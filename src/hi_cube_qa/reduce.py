from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from .io_utils import frequency_velocity, pixel_scale_asec
from .noise    import mad_rms_planes, kurtosis_planes
from .beam     import beam_stats, beam_diagnostics

##############################################################################
#
#  reduce.py: memory-light, vectorised per-cube measurement.
#
#  CubeHandle memory-maps a FITS cube so only the channels being processed are
#  read. reduce_product() makes a SINGLE chunked, vectorised pass that turns a
#  multi-GB cube into a compact bundle of 1-D spectra, two 2-D diagnostic maps,
#  and a handful of channel planes -- so the rest of the pipeline never holds a
#  whole cube, and the per-channel RMS is computed in C, not a Python loop.
#
##############################################################################


class CubeHandle:
    """Memory-mapped access to a FITS spectral cube as (nchan, ny, nx)."""

    def __init__(self, path: str | Path):
        self.path  = str(path)
        self._hdul = fits.open(self.path, memmap=True)
        self.hdr   = self._hdul[0].header
        data = np.squeeze(self._hdul[0].data)      # lazy view; drops Stokes etc.
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        if data.ndim != 3:
            raise ValueError(f"[reduce] Unexpected cube shape {data.shape}: {path}")
        self.data = data
        self.nchan, self.ny, self.nx = data.shape
        self.freq_hz, self.vel_kms   = frequency_velocity(self.hdr, self.nchan,
                                                           quiet=True)

    def close(self):
        try:
            self._hdul.close()
        except Exception:                                           # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _pix_per_beam(hdr, bmaj_arr, bmin_arr) -> float:
    """Pixels per synthesised beam, from the header pixel scale and median beam."""
    cdelt1 = abs(hdr.get("CDELT1", 6.0 / 3600.0))
    cdelt2 = abs(hdr.get("CDELT2", 6.0 / 3600.0))
    pix_sr = np.deg2rad(cdelt1) * np.deg2rad(cdelt2)
    if bmaj_arr is not None and np.any(np.isfinite(bmaj_arr)):
        bmaj_deg = np.nanmedian(bmaj_arr) / 3600.0
        bmin_deg = np.nanmedian(bmin_arr) / 3600.0
    else:
        bmaj_deg, bmin_deg = 18.0 / 3600.0, 14.0 / 3600.0
    beam_sr = (np.pi / (4.0 * np.log(2.0))) * np.deg2rad(bmaj_deg) * np.deg2rad(bmin_deg)
    return float(beam_sr / pix_sr) if pix_sr > 0 else 1.0


def _is_pbcorrected(path: str, hdr) -> bool:
    """Heuristic: is this a primary-beam-corrected cube (position-dependent noise)?"""
    name = Path(path).name.lower()
    if "pbcor" in name or "pb_corr" in name or "pbcorr" in name:
        return True
    for key in ("PBCOR", "PBCORR"):
        v = hdr.get(key)
        if isinstance(v, bool) and v:
            return True
        if isinstance(v, str) and v.strip().upper() in ("T", "TRUE", "YES"):
            return True
    return False


def _select_channels(rms, snr, selectors) -> dict:
    """Pick diagnostic channels by RMS and/or SNR; return {chan: [reasons]}."""
    picks: dict[int, list[str]] = {}

    def add(ch, reason):
        ch = int(ch)
        picks.setdefault(ch, [])
        if reason not in picks[ch]:
            picks[ch].append(reason)

    if "rms" in selectors:
        fin = np.where(np.isfinite(rms))[0]
        if fin.size:
            v = rms[fin]
            add(fin[np.argmin(v)], "min RMS")
            add(fin[np.argmax(v)], "max RMS")
            add(fin[np.argmin(np.abs(v - np.median(v)))], "median RMS")
    if "snr" in selectors:
        fin = np.where(np.isfinite(snr))[0]
        if fin.size:
            v = snr[fin]
            add(fin[np.argmax(v)], "peak SNR")
            add(fin[np.argmin(v)], "min SNR")
            add(fin[np.argmin(np.abs(v - np.median(v)))], "median SNR")
    return dict(sorted(picks.items()))


def reduce_product(
        path:            str | Path,
        *,
        selectors:       tuple = ("rms", "snr"),
        chunk:           int   = 64,
        cont_sigma:      float = 3.0,
        scatter_warn:    float = 0.5,
        compute_maps:    bool  = True,
        boresight_cfg:   dict | None = None,
        ) -> dict:
    """
    Single streaming pass over one cube. Returns a compact measurement bundle:

        nchan, ny, nx, freq_hz, vel_kms, chan_dv_kms
        rms, peak, mean, s_int            (1-D, per channel; Jy or Jy/beam)
        beam (bmaj,bmin,bpa), beam_diag, pix_per_beam
        noise_map, mean_map, persist_frac_map   (2-D; None if compute_maps=False)
        cont_metrics                      (continuum-residual diagnostics)
        thumb_planes {chan: 2-D}, thumb_reasons {chan: [..]}
        boresight                         (1-D, or None)
        pb_corrected, bunit
    """
    h = CubeHandle(path)
    try:
        nchan, ny, nx = h.nchan, h.ny, h.nx
        hdr = h.hdr

        bmaj, bmin, bpa = beam_stats(hdr, nchan)
        beam_diag = beam_diagnostics(bmaj, bmin, bpa, scatter_warn_asec=scatter_warn)
        ppb = _pix_per_beam(hdr, bmaj, bmin)

        rms    = np.full(nchan, np.nan)
        peak   = np.full(nchan, np.nan)
        mean   = np.full(nchan, np.nan)
        s_int  = np.full(nchan, np.nan)
        kurt   = np.full(nchan, np.nan)
        nfinite = np.zeros(nchan, dtype=np.int64)

        if compute_maps:
            cnt   = np.zeros((ny, nx), dtype=np.float64)
            ssum  = np.zeros((ny, nx), dtype=np.float64)
            ssq   = np.zeros((ny, nx), dtype=np.float64)
            persist = np.zeros((ny, nx), dtype=np.float64)

        for a in range(0, nchan, chunk):
            b     = min(a + chunk, nchan)
            block = np.asarray(h.data[a:b], dtype=np.float32)
            med, sig = mad_rms_planes(block)
            rms[a:b] = sig
            kurt[a:b] = kurtosis_planes(block)
            fin = np.isfinite(block)
            nfinite[a:b] = fin.sum(axis=(1, 2))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                peak[a:b] = np.nanmax(np.abs(block), axis=(1, 2))
                mean[a:b] = np.nanmean(block, axis=(1, 2))
                s_int[a:b] = np.nansum(block, axis=(1, 2)) / ppb

            if compute_maps:
                bz  = np.where(fin, block, 0.0).astype(np.float64)
                cnt  += fin.sum(axis=0)
                ssum += bz.sum(axis=0)
                ssq  += (bz * bz).sum(axis=0)
                dev  = np.abs(block - med[:, None, None])
                thr  = (cont_sigma * np.nan_to_num(sig, nan=np.inf))[:, None, None]
                persist += (fin & (dev > thr)).sum(axis=0)

        chan_dv = float(np.nanmedian(np.diff(h.vel_kms))) if nchan > 1 else np.nan

        #--- 2-D maps ---
        noise_map = mean_map = persist_map = None
        cont_metrics = {"available": False}
        if compute_maps:
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_map  = np.where(cnt > 0, ssum / cnt, np.nan)
                var_map   = np.where(cnt > 0, ssq / cnt - mean_map ** 2, np.nan)
                noise_map = np.sqrt(np.clip(var_map, 0, None))
                persist_map = np.where(cnt > 0, persist / cnt, np.nan)
            # continuum-residual indicator: pixels exceeding cont_sigma in a
            # large fraction of channels are unlikely to be noise.
            pf = persist_map[np.isfinite(persist_map)]
            pct_pix_persistent = (float(np.mean(pf > 0.5)) * 100.0) if pf.size else np.nan
            cont_metrics = {
                    "available":            True,
                    "cont_sigma":           cont_sigma,
                    "pct_pixels_persistent": pct_pix_persistent,   # >50% of channels
                    "median_persist_frac":  float(np.nanmedian(persist_map))
                                            if np.any(np.isfinite(persist_map)) else np.nan,
                    }

        #--- diagnostic channel planes (read individually from memmap) ---
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = peak / (rms + 1e-30)
        picks = _select_channels(rms, snr, selectors)
        thumb_planes = {ch: np.asarray(h.data[ch], dtype=np.float32) for ch in picks}

        #--- boresight (best-case) noise from a small central box ---
        boresight = None
        if boresight_cfg and boresight_cfg.get("enable", True):
            boresight = _boresight_from_box(h, boresight_cfg)

        nfin_pos = nfinite[nfinite > 0]
        n_finite_med = float(np.median(nfin_pos)) if nfin_pos.size else float(ny * nx)

        return {
                "hdr": hdr,
                "nchan": nchan, "ny": ny, "nx": nx,
                "freq_hz": h.freq_hz, "vel_kms": h.vel_kms, "chan_dv_kms": chan_dv,
                "rms": rms, "peak": peak, "mean": mean, "s_int": s_int,
                "kurtosis": kurt, "nfinite": nfinite,
                "n_finite_med": n_finite_med,
                "beam": (bmaj, bmin, bpa), "beam_diag": beam_diag, "pix_per_beam": ppb,
                "noise_map": noise_map, "mean_map": mean_map,
                "persist_frac_map": persist_map, "cont_metrics": cont_metrics,
                "thumb_planes": thumb_planes, "thumb_reasons": picks,
                "boresight": boresight,
                "pb_corrected": _is_pbcorrected(h.path, hdr),
                "bunit": str(hdr.get("BUNIT", "")).strip(),
                }
    finally:
        h.close()


def _boresight_from_box(h: CubeHandle, cfg: dict) -> np.ndarray:
    """Per-channel MAD-RMS in a central box, read straight from the memmap."""
    ny, nx = h.ny, h.nx
    centre = cfg.get("centre_pix")
    if centre:
        cy, cx = centre
    else:
        cy = int(h.hdr.get("CRPIX2", ny // 2 + 1)) - 1
        cx = int(h.hdr.get("CRPIX1", nx // 2 + 1)) - 1
    half = int(cfg.get("box_size", 100)) // 2
    y0, y1 = max(0, cy - half), min(ny, cy + half)
    x0, x1 = max(0, cx - half), min(nx, cx + half)

    box_cube = np.asarray(h.data[:, y0:y1, x0:x1], dtype=np.float64)  # small
    kappa = float(cfg.get("sigma_clip", 3.0))
    iters = int(cfg.get("clip_iterations", 3))

    nchan = box_cube.shape[0]
    out = np.full(nchan, np.nan)
    for i in range(nchan):
        box  = box_cube[i]
        mask = np.isfinite(box)
        for _ in range(iters):
            if mask.sum() < 10:
                break
            v   = box[mask]
            med = np.median(v)
            s   = 1.4826 * np.median(np.abs(v - med))
            if s == 0:
                break
            mask = mask & (np.abs(box - med) < kappa * s)
        v = box[mask]
        if v.size:
            med    = np.median(v)
            out[i] = 1.4826 * np.median(np.abs(v - med))
    return out
