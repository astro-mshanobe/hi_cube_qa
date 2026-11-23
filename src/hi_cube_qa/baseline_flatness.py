"""
baseline_flatness.py
=====================
Checks whether the continuum-subtracted baseline (per-channel mean flux,
averaged over the whole image) is flat across the band for each HI cube --
i.e. no systematic residual slope, curvature, or offset left after
continuum subtraction.

This matters specifically for MOSAICKING: a cube can pass every criterion in
the main hi_cube_qa checklist (noise level, kurtosis, beam stability, pixel-
level continuum-residual persistence, ...) and still carry a small residual
baseline trend. When pointings are combined in their overlap regions, a
trend that differs pointing-to-pointing shows up as a seam or gradient in
the mosaic that none of the single-cube checks would catch, because none of
them look at the SHAPE of the band-averaged baseline -- only its noise level
and pixel-level outliers.

This is a standalone companion script, not folded into main.py, because:
  - the main pipeline never persists the per-channel mean anywhere (only
    rms/quality/kurtosis are written to the per-channel CSV), so there is
    nothing to reuse after the fact -- the cubes have to be re-read; and
  - discovery.py's product parsing is oriented around the CARACal imaging
    triad (dirty/image/residual, from names like '..._HI.image.fits') and
    will not find 'imsub-line.fits' (no dot before the suffix), which is
    the actual continuum-subtracted, mosaic-ready product this check needs
    to run on.

It does reuse the toolkit's own cube I/O (CubeHandle), noise estimator
(mad_rms_planes) and beam/pix-per-beam helpers, and reads session roots from
the same YAML config as the main pipeline, so it stays consistent with it.

Usage:
  hi-cube-qa-baseline --config qa_G23_all_sessions.yaml
  hi-cube-qa-baseline --config qa_G23_all_sessions.yaml --degree 2
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np

from .config import load_config
from .reduce import CubeHandle, _pix_per_beam
from .noise  import mad_rms_planes
from .beam   import beam_stats


def per_channel_mean_rms(h: CubeHandle, chunk: int = 64):
    """Chunked pass -> per-channel mean (baseline) and MAD-RMS (noise)."""
    nchan = h.nchan
    mean = np.full(nchan, np.nan)
    rms  = np.full(nchan, np.nan)
    for a in range(0, nchan, chunk):
        b = min(a + chunk, nchan)
        block = np.asarray(h.data[a:b], dtype=np.float32)
        _, sig = mad_rms_planes(block)
        rms[a:b] = sig
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean[a:b] = np.nanmean(block, axis=(1, 2))
    return mean, rms


def _robust_polyfit(x, y, degree, n_sigma=4.0, n_iter=3):
    """
    Iterative sigma-clipped polynomial fit. A handful of outlier channels
    (residual RFI, under-flagged bad channels, ...) can otherwise dominate
    an ordinary least-squares fit's slope even when the bulk of the band is
    flat -- a few high-leverage points is exactly the failure mode this
    check needs to be robust against, since it's meant to catch a genuine
    SMOOTH systematic trend (the thing that creates mosaic seams), not
    isolated bad channels (a separate, already-covered concern).

    Returns (coeffs, keep_mask) -- keep_mask is True for points used in the
    final fit. n_sigma <= 0 disables clipping (keep_mask is all True).
    """
    keep = np.ones(len(x), dtype=bool)
    coeffs = np.polyfit(x, y, degree)
    if n_sigma <= 0:
        return coeffs, keep
    for _ in range(n_iter):
        fit = np.polyval(coeffs, x)
        resid = y - fit
        med = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - med))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(resid[keep]))
        if sigma <= 0 or not np.isfinite(sigma):
            break
        new_keep = np.abs(resid - med) < n_sigma * sigma
        if new_keep.sum() < degree + 2 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= degree + 2 else keep
            break
        keep = new_keep
        coeffs = np.polyfit(x[keep], y[keep], degree)
    return coeffs, keep


def baseline_check(h: CubeHandle, degree: int = 1, chunk: int = 64,
                    n_sigma: float = 4.0) -> dict:
    """
    Fit a low-order polynomial to the per-channel mean (baseline) vs
    velocity, then express the fitted trend's peak-to-peak amplitude in
    units of the standard error on the per-channel mean (not the raw
    per-pixel RMS -- averaging over ~ny*nx/pix_per_beam independent beam
    areas per channel shrinks the noise on the MEAN well below the per-pixel
    RMS, so per-pixel RMS alone would be far too permissive a yardstick).

    The fit itself is sigma-clipped (see _robust_polyfit) so isolated
    outlier channels don't masquerade as a systematic trend.
    """
    mean, rms = per_channel_mean_rms(h, chunk=chunk)
    fin = np.isfinite(mean) & np.isfinite(rms) & (rms > 0)
    if fin.sum() < degree + 2:
        return dict(available=False)

    x = h.vel_kms[fin]
    freq = h.freq_hz[fin]
    y = mean[fin]
    coeffs, keep = _robust_polyfit(x, y, degree, n_sigma=n_sigma)
    fit = np.polyval(coeffs, x)
    baseline_amplitude = float(np.nanmax(fit[keep]) - np.nanmin(fit[keep]))
    n_clipped = int((~keep).sum())

    bmaj, bmin, _ = beam_stats(h.hdr, h.nchan)
    ppb = _pix_per_beam(h.hdr, bmaj, bmin)
    n_beams = max((h.ny * h.nx) / ppb, 1.0)
    sem = rms[fin] / np.sqrt(n_beams)
    typical_sem = float(np.nanmedian(sem[keep]))

    baseline_sigma = baseline_amplitude / typical_sem if typical_sem > 0 else np.nan
    return dict(
        available=True,
        degree=degree,
        baseline_amplitude_jy=baseline_amplitude,
        typical_sem_jy=typical_sem,
        baseline_sigma=baseline_sigma,
        n_channels_used=int(fin.sum()),
        n_clipped=n_clipped,
        # raw arrays for optional plotting -- small (nchan-length), cheap to keep
        vel_kms=x, freq_hz=freq, mean_jy=y, sem_jy=sem, fit_jy=fit, keep_mask=keep,
    )


def plot_baseline(res: dict, label: str, out_png: str) -> None:
    """Diagnostic plot: per-channel mean (baseline) with SEM error bars, and
    the fitted trend that baseline_sigma is computed from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, sem, fit = res["vel_kms"], res["mean_jy"], res["sem_jy"], res["fit_jy"]
    keep = res.get("keep_mask", np.ones(len(x), dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(x[keep], y[keep] * 1e3, yerr=sem[keep] * 1e3, fmt=".", ms=3,
                lw=0.5, alpha=0.5, color="tab:blue", label="used in fit")
    if (~keep).any():
        ax.plot(x[~keep], y[~keep] * 1e3, "x", ms=6, color="tab:orange",
                label=f"sigma-clipped ({int((~keep).sum())} channels)")
    ax.plot(x, fit * 1e3, "-", lw=2, color="tab:red",
            label=f"degree-{res['degree']} fit "
                  f"(baseline_sigma={res['baseline_sigma']:.2f})")
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("Channel-averaged flux [mJy]")
    ax.set_title(f"Baseline flatness: {label}")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def classify(baseline_sigma, warn=5.0, bad=10.0) -> str:
    if baseline_sigma is None or not np.isfinite(baseline_sigma):
        return "N/A"
    if baseline_sigma < warn:
        return "Pass"
    if baseline_sigma < bad:
        return "Warn"
    return "Fail"


def find_imsub_cubes(root: str, pattern: str = "imsub-line.fits") -> list[str]:
    """
    pattern is glob-matched inside each cube_N/ dir. Default is the
    continuum-subtracted, mosaic-ready product. Pass e.g. '*_HI.image.fits'
    to instead check the pre-continuum-subtraction cleaned image -- useful
    as an A/B comparison of what imcontsub actually bought you.
    """
    return sorted(glob.glob(os.path.join(root, "output_hi_p*", "cubes", "cube_*",
                                          pattern)))


def main():
    ap = argparse.ArgumentParser(
        description="Check per-cube baseline flatness (mosaicking prep).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="qa_G23_all_sessions.yaml",
                     help="Same YAML config used by main.py -- session roots are reused")
    ap.add_argument("--degree", type=int, default=1,
                     help="polynomial degree for the baseline fit "
                          "(1 = slope only, 2 = + curvature)")
    ap.add_argument("--warn-sigma", type=float, default=5.0,
                     help="baseline_sigma above this -> Warn (matches the "
                          "toolkit's kurtosis_warn_sigma convention)")
    ap.add_argument("--bad-sigma", type=float, default=10.0,
                     help="baseline_sigma above this -> Fail")
    ap.add_argument("--out", default=None,
                     help="output CSV (default: <output.dir>/baseline_flatness.csv)")
    ap.add_argument("--limit", type=int, default=None,
                     help="process only the first N cubes found (for quick testing)")
    ap.add_argument("--pattern", default="imsub-line.fits",
                     help="filename glob to match inside each cube_N/ dir "
                          "(default: the continuum-subtracted product). Pass "
                          "'*_HI.image.fits' to check the pre-imcontsub image "
                          "instead, e.g. for an A/B comparison.")
    ap.add_argument("--pointing", default=None,
                     help="comma-separated pointing names to restrict to, "
                          "e.g. 'p07,p11' (default: all)")
    ap.add_argument("--plot", action="store_true",
                     help="save a per-cube diagnostic PNG (mean vs velocity "
                          "with the fitted trend) next to the output CSV")
    ap.add_argument("--sigma-clip", type=float, default=4.0,
                     help="exclude channels this many robust-sigma from the "
                          "fit before measuring the trend (0 disables clipping)")
    ap.add_argument("--dump-clipped", action="store_true",
                     help="write the frequency/velocity of every sigma-clipped "
                          "channel to <out_dir>/baseline_clipped_channels.csv "
                          "(e.g. to cross-check against a static RFI mask)")
    a = ap.parse_args()
    want_pointings = set(a.pointing.split(",")) if a.pointing else None

    cfg = load_config(a.config)
    out_path = a.out or str(cfg.out_dir() / "baseline_flatness.csv")

    rows = []
    clipped_rows = []
    n_done = 0
    for s in cfg.sessions:
        if a.limit is not None and n_done >= a.limit:
            break
        root = s["root"]
        obsid = os.path.basename(root.rstrip("/").rsplit(os.sep + "caracal", 1)[0])
        cubes = find_imsub_cubes(root, pattern=a.pattern)
        if not cubes:
            print(f"{obsid}: no '{a.pattern}' cubes found under {root}")
            continue
        print(f"{obsid}: {len(cubes)} cube(s)")
        for cube_path in cubes:
            parts = cube_path.split(os.sep)
            pointing = next((p[len("output_hi_"):] for p in parts
                              if p.startswith("output_hi_p")), "?")
            if want_pointings is not None and pointing not in want_pointings:
                continue
            if a.limit is not None and n_done >= a.limit:
                break
            n_done += 1
            cube_id = next((p for p in parts if p.startswith("cube_")), "?")
            label = f"{pointing}/{cube_id}"

            try:
                h = CubeHandle(cube_path)
                res = baseline_check(h, degree=a.degree, n_sigma=a.sigma_clip)
                h.close()
            except Exception as e:                                    # noqa: BLE001
                print(f"  {label}: ! failed: {e}", file=sys.stderr)
                res = dict(available=False)

            bsig = res.get("baseline_sigma")
            status = classify(bsig, a.warn_sigma, a.bad_sigma)
            if res.get("available"):
                print(f"  {label}: {status}  (baseline_sigma={bsig:.2f}, "
                      f"amplitude={res['baseline_amplitude_jy']*1e3:.3f} mJy, "
                      f"clipped={res['n_clipped']}/{res['n_channels_used']})")
                if a.plot:
                    png_path = str(cfg.out_dir() /
                                    f"baseline_{obsid}_{pointing}_{cube_id}.png")
                    plot_baseline(res, f"{obsid} {label}", png_path)
                    print(f"    -> {png_path}")
                if a.dump_clipped:
                    keep = res["keep_mask"]
                    for fq, vv in zip(res["freq_hz"][~keep], res["vel_kms"][~keep]):
                        clipped_rows.append(dict(
                            obsid=obsid, pointing=pointing, cube=cube_id,
                            freq_mhz=f"{fq / 1e6:.4f}", vel_kms=f"{vv:.2f}",
                        ))
            else:
                print(f"  {label}: {status}  (not enough finite channels)")

            rows.append(dict(
                obsid=obsid, pointing=pointing, cube=cube_id, path=cube_path,
                status=status,
                baseline_sigma=f"{bsig:.3f}" if res.get("available") else "",
                baseline_amplitude_mjy=(f"{res['baseline_amplitude_jy']*1e3:.4f}"
                                         if res.get("available") else ""),
                typical_sem_mjy=(f"{res['typical_sem_jy']*1e3:.4f}"
                                  if res.get("available") else ""),
                degree=res.get("degree", a.degree),
                n_channels_used=res.get("n_channels_used", ""),
                n_clipped=res.get("n_clipped", ""),
            ))

    fieldnames = ["obsid", "pointing", "cube", "path", "status", "baseline_sigma",
                  "baseline_amplitude_mjy", "typical_sem_mjy", "degree",
                  "n_channels_used", "n_clipped"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    if a.dump_clipped:
        clipped_path = str(cfg.out_dir() / "baseline_clipped_channels.csv")
        with open(clipped_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["obsid", "pointing", "cube",
                                               "freq_mhz", "vel_kms"])
            w.writeheader()
            w.writerows(clipped_rows)
        print(f"{len(clipped_rows)} clipped-channel rows -> {clipped_path}")

    n_pass = sum(1 for r in rows if r["status"] == "Pass")
    n_warn = sum(1 for r in rows if r["status"] == "Warn")
    n_fail = sum(1 for r in rows if r["status"] == "Fail")
    n_na   = sum(1 for r in rows if r["status"] == "N/A")
    print(f"\n{len(rows)} cubes checked -> {out_path}")
    print(f"  Pass: {n_pass}   Warn: {n_warn}   Fail: {n_fail}   N/A: {n_na}")
    if n_fail:
        print("\n  Fail = baseline trend is a large multiple of the noise on the "
              "channel-averaged mean. Investigate before using these pointings "
              "in a mosaic -- a residual slope/curvature this size will likely "
              "show as a seam in overlap regions.")


if __name__ == "__main__":
    main()
