"""Excess noise power on large angular scales.

A ripple or stripe pattern that is present in every channel (for example from a
short-baseline problem) averages to zero over the image, so the per-channel mean
does not show it, and it adds little to the pixel noise in quadrature. It does leave
excess power at large angular scales: a map smoothed to about two arcminutes is the
sum of a few hundred independent noise patches, so thermal noise gives a small rms
there and any structure stands out.

Method, for every `stride`-th channel k (a channel needs a neighbour k+1, or k-1):

1. Valid pixels are finite, not exactly zero (blank padding) and, where a primary-beam
   cube is given, at or above `pb_limit`.
2. Channel k and the difference k - neighbour are smoothed with a Gaussian by
   normalised convolution (smoothed map divided by smoothed mask), keeping pixels
   whose smoothing weight is at least MIN_WEIGHT. At least MIN_AREAS independent
   smoothing areas must remain.
3. rms is the sigma-clipped standard deviation of the smoothed channel map.
4. thermal is the same statistic of the smoothed difference map divided by sqrt(2).
   Thermal noise is independent between channels and structure that is coherent over
   a channel width cancels, so the difference is an empirical estimate of the noise
   alone in this channel, including the correlation of its own beam.
5. ratio = rms / thermal (1 for noise alone), excess = sqrt(rms^2 - thermal^2), and
   corr is the correlation of the smoothed maps of k and its neighbour (near 0 for
   noise, near 1 for structure that is coherent between channels).

Per cube the medians over channels are graded with `classify`. The limits are
judgement calls, set so that a cube of noise alone gives ratios of 1 +- 0.15.
"""

from __future__ import annotations

import glob
import os
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from .reduction import CubeHandle

WARN_RATIO = 1.5  # median ratio at or above this: Warn
FAIL_RATIO = 3.0  # median ratio at or above this: Fail
WARN_FRACTION = 0.25  # more than this fraction of channels above WARN_RATIO: Warn
MIN_WEIGHT = 0.98  # smoothing weight a pixel needs (normalised convolution)
MIN_AREAS = 25  # independent smoothing areas that must remain in a channel
CLIP_SIGMA = 3.0
CLIP_PASSES = 5
MIN_SIGMA_PIX = 2.0  # a narrower kernel is not "large scale" on this pixel grid


class CubeError(ValueError):
    """The cube cannot be assessed; the message says why."""


def pixel_scale_arcsec(header) -> float:
    """Pixel size in arcsec from the celestial WCS, with CDELT2 as a fallback."""
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            wcs = WCS(header).celestial
            scale = float(proj_plane_pixel_scales(wcs)[1]) * 3600.0 if wcs.naxis == 2 else np.nan
        except Exception:
            scale = np.nan
    if not np.isfinite(scale) or scale <= 0:
        cdelt = header.get("CDELT2")
        scale = abs(float(cdelt)) * 3600.0 if cdelt is not None else np.nan
    if not np.isfinite(scale) or scale <= 0:
        raise CubeError("no usable pixel scale in the header (no celestial WCS and no CDELT2)")
    return scale


def clipped_std(x, sigma: float = CLIP_SIGMA, passes: int = CLIP_PASSES) -> float:
    """Standard deviation after sigma-clipping, starting from the median and MAD.

    NaN for fewer than 10 values. Unlike a MAD-only estimate it is the plain rms for a
    distribution without tails (a sinusoidal ripple), and unlike the plain rms it is
    not dominated by compact sources.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    med = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med))
    if not scale > 0:
        scale = float(np.std(x))
    if not scale > 0:
        return 0.0
    keep = np.abs(x - med) <= sigma * scale
    for _ in range(passes):
        if keep.sum() < 10:
            break
        mean, std = x[keep].mean(), x[keep].std()
        new = np.abs(x - mean) <= sigma * std if std > 0 else keep
        if np.array_equal(new, keep):
            break
        keep = new
    return float(x[keep].std()) if keep.sum() >= 10 else np.nan


def smooth_masked(img, mask, sigma_pix: float, min_weight: float = MIN_WEIGHT):
    """Gaussian smoothing by normalised convolution; low-weight pixels become NaN."""
    mask = np.asarray(mask, dtype=bool)
    num = gaussian_filter(np.where(mask, img, 0.0).astype(np.float64), sigma_pix, mode="constant")
    den = gaussian_filter(mask.astype(np.float64), sigma_pix, mode="constant")
    out = np.full(num.shape, np.nan)
    ok = den >= min_weight
    out[ok] = num[ok] / den[ok]
    return out


def valid_mask(plane, pb_plane=None, pb_limit: float = 0.5):
    """Finite, not exactly zero, and above the primary-beam limit where a PB plane is given."""
    mask = np.isfinite(plane) & (plane != 0)
    if pb_plane is not None:
        mask &= np.isfinite(pb_plane) & (pb_plane >= pb_limit)
    return mask


def channel_stat(plane, other, sigma_pix: float, pb_plane=None, pb_limit: float = 0.5):
    """Large-scale statistics of one channel against a neighbouring channel.

    Returns a dict (rms, thermal, ratio, ratio_plain, excess, corr, n_areas) in the
    cube's unit, or None when the channel cannot be assessed.
    """
    mask = valid_mask(plane, pb_plane, pb_limit) & valid_mask(other, pb_plane, pb_limit)
    if mask.sum() < 100:
        return None
    a = smooth_masked(plane, mask, sigma_pix)
    d = smooth_masked(
        np.where(mask, plane.astype(np.float64) - other.astype(np.float64), 0.0), mask, sigma_pix
    )
    b = smooth_masked(other, mask, sigma_pix)
    use = np.isfinite(a) & np.isfinite(d) & np.isfinite(b)
    n_areas = use.sum() / (4.0 * np.pi * sigma_pix**2)
    if n_areas < MIN_AREAS:
        return None
    rms, thermal = clipped_std(a[use]), clipped_std(d[use]) / np.sqrt(2.0)
    if not (np.isfinite(rms) and np.isfinite(thermal)):
        return None
    ratio = rms / thermal if thermal > 0 else np.nan
    # the same without clipping: it also sees structure that covers only part of the field
    plain_std = d[use].std() / np.sqrt(2.0)
    plain = float(a[use].std() / plain_std) if plain_std > 0 else np.nan
    sa, sb = a[use].std(), b[use].std()
    corr = float(np.corrcoef(a[use], b[use])[0, 1]) if sa > 0 and sb > 0 else np.nan
    return dict(
        rms=float(rms),
        thermal=float(thermal),
        ratio=float(ratio),
        ratio_plain=plain,
        excess=float(np.sqrt(max(rms**2 - thermal**2, 0.0))),
        corr=corr,
        n_areas=float(n_areas),
    )


def classify(
    median_ratio: float,
    frac_above: float,
    warn_ratio: float = WARN_RATIO,
    fail_ratio: float = FAIL_RATIO,
    warn_fraction: float = WARN_FRACTION,
) -> str:
    """Pass / Warn / Fail from the median ratio and the fraction of channels above `warn_ratio`."""
    if not np.isfinite(median_ratio):
        return "No data"
    if median_ratio >= fail_ratio:
        return "Fail"
    if median_ratio >= warn_ratio or frac_above > warn_fraction:
        return "Warn"
    return "Pass"


def sample_channels(n: int, stride: int) -> list[int]:
    """Channel indices to assess: 0, stride, 2 stride, ..."""
    if int(stride) < 1:
        raise ValueError(f"the channel stride must be at least 1, got {stride}")
    return list(range(0, int(n), int(stride)))


def summarise(rows, sigma_arcsec: float, pixel_arcsec: float, limits: dict | None = None) -> dict:
    """Per-cube summary and grade from per-channel rows (rms, thermal, ratio, ... in mJy/beam)."""
    limits = limits or {}
    ratio = np.array([r["ratio"] for r in rows])
    rms2 = np.array([r["rms"] for r in rows]) ** 2
    th2 = np.array([r["thermal"] for r in rows]) ** 2
    plain = np.array([r["ratio_plain"] for r in rows])
    corr = np.array([r["corr"] for r in rows])
    finite = np.isfinite(ratio)
    warn_ratio = limits.get("warn_ratio", WARN_RATIO)
    median_ratio = float(np.median(ratio[finite])) if finite.any() else np.nan
    frac_above = float((ratio[finite] > warn_ratio).mean()) if finite.any() else np.nan
    summary = dict(
        available=True,
        n_channels=len(rows),
        sigma_arcsec=float(sigma_arcsec),
        pixel_arcsec=float(pixel_arcsec),
        median_rms=float(np.sqrt(np.median(rms2))),
        median_thermal=float(np.sqrt(np.median(th2))),
        median_excess=float(np.sqrt(max(np.median(rms2) - np.median(th2), 0.0))),
        median_ratio=median_ratio,
        median_ratio_plain=float(np.median(plain[np.isfinite(plain)]))
        if np.isfinite(plain).any()
        else np.nan,
        frac_above=frac_above,
        median_corr=float(np.nanmedian(corr)) if np.isfinite(corr).any() else np.nan,
    )
    summary["status"] = classify(median_ratio, frac_above, **limits)
    return summary


def localised_note(summary: dict, fail_ratio: float = FAIL_RATIO) -> str:
    """A note when the clipped ratio passes but the unclipped one is large: structure over part of the field."""
    plain = summary["median_ratio_plain"]
    if summary["status"] == "Pass" and np.isfinite(plain) and plain >= fail_ratio:
        return f"localised structure (unclipped ratio {plain:.1f}): sources or a ripple over part of the field"
    return ""


def assess_planes(
    planes,
    header,
    pb_planes=None,
    sigma_arcsec: float = 120.0,
    stride: int = 20,
    pb_limit: float = 0.5,
    min_channels: int = 5,
    limits: dict | None = None,
    unit_scale: float = 1e3,
):
    """Assess a cube given as (channels, y, x) arrays; memory maps are fine.

    Returns (summary, rows). Rms, thermal and excess are multiplied by `unit_scale`
    (1e3 turns Jy/beam into mJy/beam). Raises CubeError when the grid is too small for
    the kernel or fewer than `min_channels` channels could be assessed.
    """
    if pb_planes is not None and tuple(pb_planes.shape) != tuple(planes.shape):
        raise CubeError(
            f"the PB cube has shape {tuple(pb_planes.shape)}, the cube {tuple(planes.shape)}"
        )
    nchan, ny, nx = planes.shape
    pixel = pixel_scale_arcsec(header)
    sigma_pix = float(sigma_arcsec) / pixel
    if sigma_pix < MIN_SIGMA_PIX:
        raise CubeError(
            f"a smoothing sigma of {sigma_arcsec} arcsec is only {sigma_pix:.2f} pixels of "
            f"{pixel:.2f} arcsec: not large scale on this grid (need at least {MIN_SIGMA_PIX} pixels)"
        )
    if min(ny, nx) < 8 * sigma_pix:
        raise CubeError(
            f"the image ({nx} x {ny} pixels) is smaller than 8 smoothing sigmas ({8 * sigma_pix:.0f} pixels)"
        )
    sampled = sample_channels(nchan, stride)
    rows = []
    for k in sampled:
        result = None
        for neighbour in (k + 1, k - 1):
            if not 0 <= neighbour < nchan:
                continue
            plane = np.asarray(planes[k], dtype=np.float64)
            other = np.asarray(planes[neighbour], dtype=np.float64)
            pb = None if pb_planes is None else np.asarray(pb_planes[k], dtype=np.float64)
            result = channel_stat(plane, other, sigma_pix, pb, pb_limit)
            if result is not None:
                break
        if result is not None:
            scaled = {
                key: v * unit_scale if key in ("rms", "thermal", "excess") else v
                for key, v in result.items()
            }
            rows.append(dict(channel=k, **scaled))
    if len(rows) < min_channels:
        raise CubeError(
            f"only {len(rows)} of {len(sampled)} sampled channels could be assessed "
            f"(need {min_channels}); the others are blank or too small after the PB cut"
        )
    return summarise(rows, sigma_arcsec, pixel, limits), rows


def assess_file(path, cfg: dict, limits: dict | None = None):
    """Assess one cube file; a `*.pb.fits` cube in the same folder is used for a PB cut.

    `cfg` holds sigma_arcsec, stride, pb_limit, use_pb and min_channels. Returns (summary, rows);
    the summary carries a `note`. Raises CubeError or OSError with a message naming the problem.
    """
    from astropy.io import fits

    handle = CubeHandle(path)
    try:
        pb_planes, note, pb_hdul = None, "", None
        if cfg.get("use_pb", True):
            candidates = sorted(glob.glob(os.path.join(os.path.dirname(str(path)), "*.pb.fits")))
            if candidates:
                pb_hdul = fits.open(candidates[0], memmap=True)
                pb = np.squeeze(pb_hdul[0].data)
                if pb.shape == handle.data.shape:
                    pb_planes = pb
                else:
                    note = f"PB cube {os.path.basename(candidates[0])} has another shape: not used"
            else:
                note = "no PB cube next to the cube: no PB cut"
        try:
            summary, rows = assess_planes(
                handle.data,
                handle.hdr,
                pb_planes,
                sigma_arcsec=cfg.get("sigma_arcsec", 120.0),
                stride=cfg.get("stride", 20),
                pb_limit=cfg.get("pb_limit", 0.5),
                min_channels=cfg.get("min_channels", 5),
                limits=limits,
            )
        finally:
            if pb_hdul is not None:
                pb_hdul.close()
        pb_note = note or (f"PB >= {cfg.get('pb_limit', 0.5):g}" if pb_planes is not None else "")
        local = localised_note(summary, (limits or {}).get("fail_ratio", FAIL_RATIO))
        summary["note"] = "; ".join(t for t in (pb_note, local) if t)
        return summary, rows
    finally:
        handle.close()
