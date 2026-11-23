"""CSV export and structured stdout."""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np


def setup_logger(log_path: Path | None = None) -> logging.Logger:
    """Return a logger that prints to stdout and (optionally) writes to *log_path*."""
    logger = logging.getLogger("hi_cube_qa")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def write_channel_csv(
    path: Path,
    label: str,
    freq_hz: np.ndarray,
    vel_kms: np.ndarray,
    rms: np.ndarray,
    quality: np.ndarray,
    theory_rms: float | None,
    bmaj: np.ndarray | None,
    bmin: np.ndarray | None,
    bpa: np.ndarray | None,
    append: bool = False,
    kurtosis: np.ndarray | None = None,
) -> None:
    """Write (or append) per-channel QA metrics to *path* as CSV."""
    mode = "a" if append else "w"
    write_header = not (append and path.exists())

    nchan = len(rms)

    # theory_rms may be a scalar or a per-channel array (frequency/flag aware).
    ta = np.asarray(theory_rms, dtype=float) if theory_rms is not None else None

    def _theory_at(i):
        if ta is None:
            return None
        if ta.ndim == 0:
            return float(ta)
        return float(ta[i]) if i < ta.shape[0] else None

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "cube_label",
                    "channel",
                    "freq_MHz",
                    "vel_kms",
                    "rms_mjy_beam",
                    "rms_to_theory",
                    "quality",
                    "kurtosis",
                    "bmaj_asec",
                    "bmin_asec",
                    "bpa_deg",
                ]
            )
        for i in range(nchan):
            th = _theory_at(i)
            ratio = (
                float(rms[i]) / th if (th is not None and th > 0 and np.isfinite(rms[i])) else ""
            )
            writer.writerow(
                [
                    label,
                    i,
                    f"{freq_hz[i] / 1e6:.4f}",
                    f"{vel_kms[i]:.2f}",
                    f"{rms[i] * 1e3:.4f}" if np.isfinite(rms[i]) else "NaN",
                    f"{ratio:.4f}" if ratio != "" else "",
                    quality[i],
                    f"{kurtosis[i]:.3f}"
                    if kurtosis is not None and np.isfinite(kurtosis[i])
                    else "",
                    f"{bmaj[i]:.3f}" if bmaj is not None and np.isfinite(bmaj[i]) else "",
                    f"{bmin[i]:.3f}" if bmin is not None and np.isfinite(bmin[i]) else "",
                    f"{bpa[i]:.2f}" if bpa is not None and np.isfinite(bpa[i]) else "",
                ]
            )


def write_band_summary_csv(
    path: Path,
    noise_summaries: dict,
    beam_diag: dict,
    cfg,
    nhi_by_label: dict | None = None,
    cont_by_label: dict | None = None,
    kurt_by_label: dict | None = None,
    baseline_by_label: dict | None = None,
    large_by_label: dict | None = None,
) -> None:
    """Write one row per cube with band-level summary statistics."""
    nhi_by_label = nhi_by_label or {}
    cont_by_label = cont_by_label or {}
    kurt_by_label = kurt_by_label or {}
    baseline_by_label = baseline_by_label or {}
    large_by_label = large_by_label or {}
    rows = []
    for label, ns in noise_summaries.items():
        bd = beam_diag.get(label, {})
        nh = nhi_by_label.get(label, {})
        cm = cont_by_label.get(label, {})
        kt = kurt_by_label.get(label, {})
        bl = baseline_by_label.get(label) or {}
        ls = large_by_label.get(label) or {}
        row = {
            "cube_label": label,
            "n_channels": ns["n_total"],
            "median_rms_mjy_beam": f"{ns['median_rms_mjy']:.4f}",
            "min_rms_mjy_beam": f"{ns['min_rms_mjy']:.4f}",
            "max_rms_mjy_beam": f"{ns['max_rms_mjy']:.4f}",
            "std_rms_mjy_beam": f"{ns['std_rms_mjy']:.4f}",
            "theory_rms_mjy_beam": f"{ns['theory_rms_mjy']:.4f}" if ns["theory_rms_mjy"] else "",
            "rms_to_theory": f"{ns['rms_to_theory']:.3f}" if ns["rms_to_theory"] else "",
            "nhi_3sig_dv_cm2": f"{nh['nhi_dv_cm2']:.3e}"
            if nh.get("nhi_dv_cm2") and np.isfinite(nh["nhi_dv_cm2"])
            else "",
            "nhi_dv_kms": f"{nh['dv_kms']:.1f}" if nh.get("dv_kms") else "",
            "pct_pixels_persistent": f"{cm['pct_pixels_persistent']:.3f}"
            if cm.get("available")
            else "",
            "pct_bad_kurtosis": f"{kt['pct_bad']:.2f}" if kt.get("available") else "",
            "pct_warn_kurtosis": f"{kt['pct_warn']:.2f}" if kt.get("available") else "",
            "baseline_sigma": f"{bl['baseline_sigma']:.2f}" if bl.get("available") else "",
            "baseline_n_clipped": bl["n_clipped"] if bl.get("available") else "",
            "ls_status": ls["status"] if ls.get("available") else "",
            "ls_median_ratio": f"{ls['median_ratio']:.3f}" if ls.get("available") else "",
            "ls_frac_above": f"{ls['frac_above']:.3f}" if ls.get("available") else "",
            "ls_median_excess_mjy": f"{ls['median_excess']:.4f}" if ls.get("available") else "",
            "ls_median_corr": f"{ls['median_corr']:.3f}" if ls.get("available") else "",
            "n_good": ns["n_good"],
            "n_warn": ns["n_warn"],
            "n_bad": ns["n_bad"],
            "pct_bad": f"{ns['pct_bad']:.1f}",
            "overall_quality": ns["overall_quality"],
            "median_bmaj_asec": f"{bd['median_bmaj_asec']:.3f}" if bd.get("available") else "",
            "median_bmin_asec": f"{bd['median_bmin_asec']:.3f}" if bd.get("available") else "",
            "std_bmaj_asec": f"{bd['std_bmaj_asec']:.4f}" if bd.get("available") else "",
            "axial_ratio": f"{bd['axial_ratio']:.3f}" if bd.get("available") else "",
        }
        rows.append(row)

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_survey_summary(json_path: Path, csv_path: Path, rows: list[dict]) -> None:
    """Write a survey-level rollup (one row per unit/product across all sessions)
    as both JSON and CSV, for tracking quality across the whole campaign.
    """
    import json

    if not rows:
        return
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    # Union of keys, stable order from the first row then any extras.
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


CHECKLIST_CRITERIA = [
    ("Noise within 1.5x theoretical", "Band-averaged RMS / noise_theory < 1.5", "noise"),
    ("Few bad channels (< 5%)", "Fraction of Bad-quality channels < 5%", "noise"),
    ("No catastrophic channels (< 15% bad)", "Fraction of Bad-quality channels < 15%", "noise"),
    (
        "Noise floor convergence",
        "Cumulative median RMS stabilises before the last 10% of channels",
        "noise",
    ),
    ("Beam stability", "BMAJ std across band < 0.5 arcsec", "beam"),
    ("Reasonable axial ratio", "BMAJ/BMIN < 2.0 (not severely elongated)", "beam"),
    ("No severe RMS outliers", "No single channel exceeds 5x median RMS", "noise"),
    (
        "Low continuum residual",
        "Few pixels significant in >50% of channels (continuum subtraction)",
        "continuum",
    ),
    (
        "Gaussian noise (kurtosis)",
        "Few channels with |kurtosis-3| beyond the expected sampling error",
        "noise",
    ),
    (
        "Flat baseline",
        "Fitted trend of the per-channel mean < 5 standard errors of the mean",
        "baseline",
    ),
    (
        "Low large-scale noise",
        "Smoothed-map rms within 1.5x the thermal expectation from neighbouring channels",
        "large_scale",
    ),
]


def build_expert_checklist(
    noise_summaries: dict,
    beam_diags: dict,
    rms_arrays: dict,
    thresholds: dict,
    cont_by_label: dict | None = None,
    kurt_by_label: dict | None = None,
    baseline_by_label: dict | None = None,
    large_by_label: dict | None = None,
) -> list[dict]:
    """Evaluate each expert criterion for each cube."""
    rows = []
    cont_by_label = cont_by_label or {}
    kurt_by_label = kurt_by_label or {}
    baseline_by_label = baseline_by_label or {}
    large_by_label = large_by_label or {}

    for label, ns in noise_summaries.items():
        bd = beam_diags.get(label, {})
        rms = rms_arrays.get(label, np.array([np.nan]))

        # Noise within 1.5x theoretical
        ratio = ns.get("rms_to_theory")
        if ratio is not None:
            status = "Pass" if ratio < 1.5 else "Warn" if ratio < 3.0 else "Fail"
            value = f"{ratio:.2f}x"
            note = ""
        else:
            status = "N/A"
            value = "-"
            note = "No theoretical noise available"
        rows.append(_row(label, CHECKLIST_CRITERIA[0], status, value, note))

        # Few bad channels (< 5%)
        pct_bad = ns["pct_bad"]
        status = "Pass" if pct_bad < 5 else "Warn" if pct_bad < 15 else "Fail"
        rows.append(_row(label, CHECKLIST_CRITERIA[1], status, f"{pct_bad:.1f}%"))

        # No catastrophic channels (< 15%)
        status = "Pass" if pct_bad < 15 else "Fail"
        rows.append(_row(label, CHECKLIST_CRITERIA[2], status, f"{pct_bad:.1f}%"))

        # Noise floor convergence
        n = len(rms)
        last10 = rms[int(n * 0.9) :]
        first90 = rms[: int(n * 0.9)]
        med_last = float(np.nanmedian(last10))
        med_all = float(np.nanmedian(first90))
        if med_all > 0 and np.isfinite(med_last) and np.isfinite(med_all):
            conv_ratio = med_last / med_all
            status = "Pass" if conv_ratio < 1.2 else "Warn" if conv_ratio < 2.0 else "Fail"
            value = f"last-10% median = {conv_ratio:.2f}x band median"
        else:
            status = "N/A"
            value = "-"
        rows.append(_row(label, CHECKLIST_CRITERIA[3], status, value))

        # Beam stability
        if bd.get("available"):
            std_bmaj = bd["std_bmaj_asec"]
            thresh = thresholds.get("beam_scatter_warn", 0.5)
            status = "Pass" if std_bmaj < thresh else "Warn"
            value = f"{std_bmaj:.3f} arcsec std"
        else:
            status = "N/A"
            value = "No beam info"
        rows.append(_row(label, CHECKLIST_CRITERIA[4], status, value))

        # Axial ratio
        if bd.get("available"):
            ar = bd["axial_ratio"]
            status = "Pass" if ar < 2.0 else "Warn" if ar < 3.0 else "Fail"
            value = f"{ar:.2f}"
        else:
            status = "N/A"
            value = "No beam info"
        rows.append(_row(label, CHECKLIST_CRITERIA[5], status, value))

        # No severe outliers (> 5x median)
        med = float(np.nanmedian(rms[np.isfinite(rms)])) if np.any(np.isfinite(rms)) else 0
        n_outlier = int(np.sum(rms[np.isfinite(rms)] > 5 * med)) if med > 0 else 0
        status = "Pass" if n_outlier == 0 else "Warn" if n_outlier < 5 else "Fail"
        value = f"{n_outlier} channels > 5x median"
        rows.append(_row(label, CHECKLIST_CRITERIA[6], status, value))

        # Continuum residual
        cont = cont_by_label.get(label)
        if (
            cont
            and cont.get("available")
            and np.isfinite(cont.get("pct_pixels_persistent", np.nan))
        ):
            pct = cont["pct_pixels_persistent"]
            status = "Pass" if pct < 0.5 else "Warn" if pct < 2.0 else "Fail"
            value = f"{pct:.2f}% pixels persistent >50% channels"
        else:
            status, value = "N/A", "No map"
        rows.append(_row(label, CHECKLIST_CRITERIA[7], status, value))

        # Non-Gaussian noise (kurtosis)
        kt = kurt_by_label.get(label)
        if kt and kt.get("available"):
            pct_bad_k = kt["pct_bad"]
            status = "Pass" if pct_bad_k < 5 else "Warn" if pct_bad_k < 15 else "Fail"
            value = f"{pct_bad_k:.1f}% channels non-Gaussian"
            note = kt.get("note", "")
        else:
            status, value, note = "N/A", "-", "No kurtosis available"
        rows.append(_row(label, CHECKLIST_CRITERIA[8], status, value, note))

        # Flat baseline
        bl = baseline_by_label.get(label)
        if bl and bl.get("available") and np.isfinite(bl["baseline_sigma"]):
            sigma = bl["baseline_sigma"]
            warn = thresholds.get("baseline_warn_sigma", 5.0)
            bad = thresholds.get("baseline_bad_sigma", 10.0)
            status = "Pass" if sigma < warn else "Warn" if sigma < bad else "Fail"
            value = f"{sigma:.1f} standard errors"
            note = f"{bl['n_clipped']} channel(s) clipped" if bl["n_clipped"] else ""
        else:
            status, value, note = "N/A", "-", "No baseline fit available"
        rows.append(_row(label, CHECKLIST_CRITERIA[9], status, value, note))

        # Large-scale noise
        ls = large_by_label.get(label)
        if ls and ls.get("available"):
            status = {"Pass": "Pass", "Warn": "Warn", "Fail": "Fail"}.get(ls["status"], "N/A")
            value = f"{ls['median_ratio']:.2f}x thermal ({ls['frac_above']:.0%} of channels above)"
            note = ls.get("note", "")
        else:
            status, value = "N/A", "-"
            note = (ls or {}).get("note") or "Not assessed for this product"
        rows.append(_row(label, CHECKLIST_CRITERIA[10], status, value, note))

    return rows


def _row(label, criterion_tuple, status, value, note=""):
    name, desc, _cat = criterion_tuple
    return {
        "cube_label": label,
        "criterion": name,
        "description": desc,
        "status": status,
        "value": value,
        "note": note,
    }


def write_expert_checklist_csv(path: Path, checklist: list[dict]) -> None:
    """Write the checklist to CSV."""
    if not checklist:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(checklist[0].keys()))
        writer.writeheader()
        writer.writerows(checklist)


def print_summary(
    logger: logging.Logger,
    noise_summaries: dict,
    beam_diags: dict,
    checklist: list[dict],
    theory_rms: float | None,
    cfg,
    heading: str = "MeerKAT HI Cube QA - Summary Report",
) -> None:
    """Print the QA summary to the logger (stdout + log file)."""
    W = 70
    logger.info("=" * W)
    logger.info(f"   {heading}")
    logger.info(f"   Output : {cfg.output['dir']}")
    logger.info("=" * W)

    # Radiometer
    rad = cfg.radiometer
    if rad["sefd"]:
        noise_src = f"SEFD = {rad['sefd']} Jy"
    else:
        noise_src = f"Tsys = {rad['tsys']} K, eta = {rad['eta']}"
    logger.info(f"  Radiometer : {noise_src}")
    logger.info(
        f"              N_ant = {rad['n_ant']}, t_on = {rad['t_total']} h, "
        f"df = {rad['df']:.1f} Hz, nfac = {rad['nfac']}"
    )
    if theory_rms is not None:
        logger.info(f"          sigma_theory = {theory_rms * 1e3:.4f} mJy/beam")
    logger.info("-" * W)

    # Per-cube noise
    for label, ns in noise_summaries.items():
        logger.info(f"  [{label}]")
        logger.info(
            f"    Median RMS  : {ns['median_rms_mjy']:.4f} mJy/beam "
            f"(min {ns['min_rms_mjy']:.4f}, max {ns['max_rms_mjy']:.4f})"
        )
        if ns["rms_to_theory"] is not None:
            logger.info(f"    RMS/theory  : {ns['rms_to_theory']:.2f}x")
        logger.info(
            f"    Channels   : {ns['n_good']} Good / {ns['n_warn']} Warn "
            f"/ {ns['n_bad']} Bad (of {ns['n_total']} total)"
        )
        logger.info(f"    Overall   : {ns['overall_quality']}")

        bd = beam_diags.get(label, {})
        if bd.get("available"):
            logger.info(
                f'    Beam       : {bd["median_bmaj_asec"]:.2f}" x '
                f'{bd["median_bmin_asec"]:.2f}" '
                f"BPA = {bd['median_bpa_deg']:.1f} deg "
                f'(BMAJ std = {bd["std_bmaj_asec"]:.3f}")'
            )
        logger.info("-" * W)

    logger.info("   Checklist:")
    current_label = None
    for row in checklist:
        if row["cube_label"] != current_label:
            current_label = row["cube_label"]
            logger.info(f"  [{current_label}]")
        icon = {"Pass": "P", "Warn": "W", "Fail": "F", "N/A": "-"}.get(row["status"], "?")
        val = row["value"]
        logger.info(f"    {icon} {row['status']:<4}  {row['criterion']:<45}  {val}")
    logger.info("=" * W)
