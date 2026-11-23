"""Entry point for HI Cube QA (session-oriented).

For every session in the config we auto-discover its cubes, assess each
(pointing, cube_N) unit, and write one interactive HTML report per session.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np

from . import figures, obsinfo

# Local modules
from .baseline_flatness import baseline_from_spectra
from .config import QAConfig, load_config, write_example_config
from .discovery import Unit, discover_flag_reports, discover_session, summarise
from .figures import (
    fig_band_spectrum,
    fig_baseline,
    fig_beam_stats,
    fig_cumulative_rms,
    fig_flag_summary,
    fig_kurtosis,
    fig_large_scale,
    fig_nhi_sensitivity,
    fig_noise_overview,
    fig_rms_curve,
    fig_rms_histogram,
    fig_rms_vs_theory,
    fig_spectral_profile,
    fig_summary_table,
)
from .io_utils import channel_width_hz, flag_fraction_of, load_flag_report
from .large_scale import assess_file
from .noise import (
    brightness_temperature,
    channel_summary,
    classify_channels,
    classify_kurtosis,
    kurtosis_expected_se,
    nhi_sensitivity,
    theoretical_noise_spectrum,
)
from .provenance import read_provenance, resolve_nfac
from .reduction import reduce_product
from .report import build_master_index, build_session_report, export_figs_png, png_export_available
from .tables import (
    build_expert_checklist,
    setup_logger,
    write_band_summary_csv,
    write_channel_csv,
    write_expert_checklist_csv,
    write_survey_summary,
)
from .thumbnails import build_map_html, build_thumbnail_html


def _flag_fractions_by_pointing(flag_data) -> dict[str, float]:
    """Map each pointing to its target-summary flagged fraction, parsed from the
    per-pointing 'target_summary' flag reports (e.g. '...-p20-corr-flag__...').
    Calibrator flags (bpcal/gcal/static, no pointing token) are ignored.
    """
    out: dict[str, float] = {}
    if not flag_data:
        return out
    for label, content in flag_data.items():
        if "summary" not in label.lower():
            continue
        m = re.search(r"p\d+\w*", label)
        if not m:
            continue
        frac = flag_fraction_of(content)
        if frac is None:
            continue
        p = m.group()
        out[p] = max(out.get(p, 0.0), float(np.clip(frac, 0.0, 0.999)))
    return out


def _resolve_flag_fractions(cfg: QAConfig, flag_data) -> tuple[dict, float, str]:
    """Resolve flagged fractions folded into the theory.

    radiometer.flag_fraction:
      - a number   -> that value for every pointing (empty per-pointing map);
      - null       -> 0;
      - 'auto'     -> per-pointing target-summary fractions, with the session
                      median as the default for pointings without a report.
    Returns (by_pointing, default, source).
    """
    ff = cfg.radiometer.get("flag_fraction", "auto")
    if ff is None:
        return {}, 0.0, "none"
    if not (isinstance(ff, str) and ff.strip().lower() == "auto"):
        try:
            return {}, float(np.clip(float(ff), 0.0, 0.999)), "config"
        except (ValueError, TypeError):
            return {}, 0.0, "bad value -> 0"
    by_pointing = _flag_fractions_by_pointing(flag_data)
    if not by_pointing:
        return {}, 0.0, "auto (no per-pointing fractions -> 0)"
    default = float(np.median(list(by_pointing.values())))
    return by_pointing, default, "auto (per-pointing target-summary)"


# Stable colour per imaging product so legends stay consistent across units.
PRODUCT_COLOUR = {
    "dirty": "#2E75B6",  # blue
    "image": "#E04B3A",  # red
    "residual": "#27AE60",  # green
    "line": "#F39C12",  # orange
    "model": "#8E44AD",  # purple
    "psf": "#1ABC9C",  # teal
}
_FALLBACK_COLOURS = ["#F39C12", "#7F8C8D", "#D35400", "#16A085"]

# Order of preference for the "noise reference" product (used for boresight and
# the band-averaged theory comparison): residuals are the cleanest noise tracer.
_NOISE_PREF = ("residual", "dirty", "image", "model")


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MeerKAT HI Cube QA - per-session interactive HTML reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    p.add_argument(
        "--example-config", action="store_true", help="Write an annotated example config and exit"
    )
    p.add_argument("--outdir", default=None, help="Override output.dir in the config")
    p.add_argument(
        "--list", action="store_true", help="Only discover and list units per session, then exit"
    )
    res = p.add_mutually_exclusive_group()
    res.add_argument(
        "--full-res",
        dest="full_res",
        action="store_true",
        default=None,
        help="Plot every channel (override the downsampled default)",
    )
    res.add_argument(
        "--downsample",
        dest="full_res",
        action="store_false",
        default=None,
        help="Force the downsampled plots (this is the default)",
    )
    return p.parse_args()


def _colour_for(product: str, idx: int) -> str:
    return PRODUCT_COLOUR.get(product, _FALLBACK_COLOURS[idx % len(_FALLBACK_COLOURS)])


def _anchor(pointing: str, cube_id: str) -> str:
    return "u-" + re.sub(r"[^A-Za-z0-9]+", "-", f"{pointing}-{cube_id}")


def _noise_reference(labels_by_product: dict[str, str]) -> str | None:
    """Return the product label to use as the noise reference, by preference."""
    for prod in _NOISE_PREF:
        if prod in labels_by_product:
            return labels_by_product[prod]
    return next(iter(labels_by_product.values()), None)


def _large_scale_wanted(selection, product: str, is_reference: bool) -> bool:
    """Whether the large-scale check runs on this product (config large_scale.products).

    "reference" is the line cube if the unit has one, otherwise the noise reference.
    """
    if selection == "all":
        return True
    if selection == "reference":
        return is_reference
    return product in selection


# Per-unit assessment
def assess_unit(
    unit: Unit,
    cfg: QAConfig,
    logger,
    flag_fraction: float = 0.0,
    n_ant: int | None = None,
    n_ant_src: str = "",
) -> dict | None:
    """Assess one (pointing, cube_N) unit. All products share a WCS + channel grid
    and are overlaid on shared axes. Each cube is reduced in a single streaming
    pass (reduction.py); the radiometer theory is frequency- and flagging-aware.
    """
    logger.info("-" * 60)
    logger.info(f"  Unit {unit.key}: {[p.product for p in unit.products]}")

    thr = cfg.thresholds
    rad = cfg.radiometer
    sel = tuple(cfg.thumbnails["selectors"])
    chunk = int(cfg.reduce.get("chunk", 64))
    want_maps = bool(cfg.reduce.get("maps", True))

    # Noise-reference product (residual > dirty > image > ...) gets the boresight
    # box and the 2-D diagnostic maps.
    present = [cp.product for cp in unit.products]
    noise_ref_product = next(
        (p for p in _NOISE_PREF if p in present), present[0] if present else None
    )

    red = {}  # {label: reduction}
    cols = {}  # {label: colour}
    for idx, cp in enumerate(unit.products):
        is_ref = cp.product == noise_ref_product
        try:
            r = reduce_product(
                cp.path,
                selectors=sel,
                chunk=chunk,
                cont_sigma=float(cfg.continuum.get("sigma", 3.0)),
                scatter_warn=thr["beam_scatter_warn"],
                compute_maps=(want_maps and is_ref),
                boresight_cfg=cfg.boresight if is_ref else None,
            )
        except Exception as exc:
            logger.warning(f"    [{cp.label}] failed to reduce ({exc}); skipping")
            continue
        red[cp.label] = r
        cols[cp.label] = _colour_for(cp.product, idx)

    if not red:
        logger.warning(f"  Unit {unit.key}: no products reduced; skipping")
        return None

    ref_label = _noise_reference({cp.product: cp.label for cp in unit.products if cp.label in red})
    rref = red[ref_label]
    for lab in [lab for lab in red if red[lab]["nchan"] != rref["nchan"]]:
        logger.warning(
            f"    [{lab}] has {red[lab]['nchan']} channels, not {rref['nchan']}: left out of this unit"
        )
        del red[lab]
    freq, vel = rref["freq_hz"], rref["vel_kms"]
    nchan = rref["nchan"]
    chan_dv = rref["chan_dv_kms"]
    df_hdr = channel_width_hz(rref["hdr"])
    df_hz = df_hdr if df_hdr > 0 else rad["df"]

    # Provenance + weighting penalty (from the noise-reference header)
    prov = read_provenance(rref["hdr"], rref["beam_diag"])
    nfac, nfac_src = resolve_nfac(rad.get("nfac", 1.0), rref["hdr"])
    sefd_curve = rad.get("sefd_curve", "default")
    curve = (
        None if sefd_curve in (None, "default") else ([] if sefd_curve == "flat" else sefd_curve)
    )
    n_ant_use = n_ant if n_ant else int(rad.get("n_ant_fallback", 60))

    # Frequency- and flagging-aware theory (Jy/beam, per channel)
    common = dict(
        sefd=rad["sefd"],
        tsys=rad["tsys"],
        eta=rad["eta"],
        n_ant=n_ant_use,
        t_total_hr=rad["t_total"],
        df_hz=df_hz,
        nfac=nfac,
        curve=curve,
    )
    theory_flat = theoretical_noise_spectrum(freq, flag_fraction=0.0, **common)
    theory_flag = theoretical_noise_spectrum(freq, flag_fraction=flag_fraction, **common)
    logger.info(
        f"    df = {df_hz / 1e3:.3f} kHz | n_ant = {n_ant_use} ({n_ant_src}) | "
        f"nfac = {nfac:.2f} ({nfac_src}) | "
        f"flagfrac = {flag_fraction:.2f} | "
        f"theory(mid) = {np.nanmedian(theory_flag) * 1e3:.4f} mJy/beam | "
        f"{nchan} ch | PBcor={rref['pb_corrected']}"
    )

    # Per-product derived quantities
    rms_dict, beam_dict, noise_sums, beam_diags = {}, {}, {}, {}
    prof_dict, spec_dict, nhi_dict, nhi_sums = {}, {}, {}, {}
    beam_arrays = {}
    cont_by_label = {}
    kurt_dict, kurt_by_label = {}, {}
    baseline_by_label = {}
    for cp in unit.products:
        lab = cp.label
        if lab not in red:
            continue
        r = red[lab]
        col = cols[lab]
        rms = r["rms"]
        quality = classify_channels(
            rms, theory_flag, warn_factor=thr["rms_warn_factor"], bad_factor=thr["rms_bad_factor"]
        )
        ns = channel_summary(rms, quality, theory_flag)
        bmaj, bmin, bpa = r["beam"]
        bd = r["beam_diag"]

        # N_HI sensitivity from the band-median RMS and median beam
        if bd.get("available"):
            bmaj_med, bmin_med = bd["median_bmaj_asec"], bd["median_bmin_asec"]
        else:
            bmaj_med = bmin_med = np.nan
        nh = nhi_sensitivity(
            np.nanmedian(rms[np.isfinite(rms)]) if np.any(np.isfinite(rms)) else np.nan,
            float(np.nanmedian(freq)),
            bmaj_med,
            bmin_med,
            chan_dv_kms=chan_dv,
            dv_kms=float(cfg.sensitivity.get("nhi_dv_kms", 20.0)),
            nsigma=float(cfg.sensitivity.get("nsigma", 3.0)),
        )
        # per-channel N_HI sensitivity spectrum
        tb_chan = brightness_temperature(rms, freq, bmaj_med, bmin_med)
        nhi_chan = (
            1.823e18
            * float(cfg.sensitivity.get("nsigma", 3.0))
            * np.asarray(tb_chan)
            * abs(chan_dv)
        )

        # integrated-spectrum noise envelope
        ppb = r["pix_per_beam"]
        med_rms = np.nanmedian(rms[np.isfinite(rms)]) if np.any(np.isfinite(rms)) else np.nan
        sigma_int = med_rms * np.sqrt(max(r["n_finite_med"], 1.0) / ppb)

        rms_dict[lab] = (rms, col, quality)
        beam_dict[lab] = (bmaj, bmin, bpa, col)
        beam_arrays[lab] = (bmaj, bmin, bpa)
        noise_sums[lab] = ns
        beam_diags[lab] = bd
        nhi_sums[lab] = nh
        prof_dict[lab] = {"peak": r["peak"], "mean": r["mean"], "colour": col}
        if cfg.baseline.get("enable", True):
            baseline_by_label[lab] = baseline_from_spectra(
                r["vel_kms"],
                r["freq_hz"],
                r["mean"],
                rms,
                n_beams=r["ny"] * r["nx"] / ppb,
                degree=int(cfg.baseline.get("degree", 1)),
                n_sigma=float(cfg.baseline.get("sigma_clip", 4.0)),
            )
        spec_dict[lab] = {
            "s_int": r["s_int"],
            "peak": r["peak"],
            "rms": rms,
            "sigma_int": float(sigma_int),
            "colour": col,
        }
        nhi_dict[lab] = {"nhi_chan": nhi_chan, "colour": col}
        if r.get("cont_metrics", {}).get("available"):
            cont_by_label[lab] = r["cont_metrics"]

        # Non-Gaussianity (kurtosis) - shape diagnostic independent of RMS
        # n_eff = independent resolution elements (beam-corrected), not raw pixels.
        n_eff_chan = np.asarray(r["nfinite"], dtype=float) / ppb
        kurt_arr = r["kurtosis"]
        se_arr = kurtosis_expected_se(n_eff_chan)
        kurt_qual = classify_kurtosis(
            kurt_arr,
            n_eff_chan,
            warn_sigma=thr.get("kurtosis_warn_sigma", 5.0),
            bad_sigma=thr.get("kurtosis_bad_sigma", 10.0),
        )
        n_total_k = len(kurt_arr)
        pct_bad_k = float(np.sum(kurt_qual == "Bad")) / n_total_k * 100 if n_total_k else 0.0
        pct_warn_k = float(np.sum(kurt_qual == "Warn")) / n_total_k * 100 if n_total_k else 0.0

        # The CHILES cross-reference: distinguish "catastrophic" channels (bad on
        # both RMS and shape) from "mild band-wide non-Gaussianity" (kurtosis bad,
        # RMS still fine) - two different failure modes the RMS check alone
        # cannot separate.
        rms_bad_mask = quality == "Bad"
        kurt_bad_mask = kurt_qual == "Bad"
        n_both = int(np.sum(rms_bad_mask & kurt_bad_mask))
        n_kurt_only = int(np.sum(kurt_bad_mask & ~rms_bad_mask))
        n_rms_only = int(np.sum(rms_bad_mask & ~kurt_bad_mask))
        xref_note = (
            f"{n_kurt_only} ch non-Gaussian w/ RMS OK, "
            f"{n_both} ch bad on both, {n_rms_only} ch RMS-bad but Gaussian"
        )

        kurt_dict[lab] = {"kurtosis": kurt_arr, "se": se_arr, "colour": col}
        kurt_by_label[lab] = {
            "available": True,
            "pct_bad": pct_bad_k,
            "pct_warn": pct_warn_k,
            "note": xref_note,
        }

        ratio = f"{ns['rms_to_theory']:.2f}x" if ns["rms_to_theory"] else "N/A"
        logger.info(
            f"    [{lab}] median RMS = {ns['median_rms_mjy']:.4f} mJy | "
            f"RMS/theory = {ratio} | "
            f"3sig N_HI({nh['dv_kms']:.0f}) = {nh['nhi_dv_cm2']:.2e} | "
            f"G/W/B = {ns['n_good']}/{ns['n_warn']}/{ns['n_bad']}"
        )
        logger.info(f"      kurtosis: {pct_bad_k:.1f}% Bad, {pct_warn_k:.1f}% Warn | {xref_note}")

    # Large-scale noise: a separate pass over the sampled channels of the chosen products
    large_by_label, large_dict = {}, {}
    ls_cfg = cfg.large_scale
    ls_limits = {
        "warn_ratio": thr["ls_warn_ratio"],
        "fail_ratio": thr["ls_fail_ratio"],
        "warn_fraction": thr["ls_warn_fraction"],
    }
    # "reference" for this check: the continuum-subtracted line cube when the unit has one
    ls_reference = (
        "line"
        if any(cp.product == "line" and cp.label in red for cp in unit.products)
        else (next(cp.product for cp in unit.products if cp.label == ref_label))
    )
    if ls_cfg.get("enable", True):
        for cp in unit.products:
            lab = cp.label
            if lab not in red or not _large_scale_wanted(
                ls_cfg["products"], cp.product, cp.product == ls_reference
            ):
                continue
            try:
                summary, rows = assess_file(cp.path, ls_cfg, limits=ls_limits)
            except Exception as exc:
                logger.warning(f"    [{lab}] large-scale noise not assessed ({exc})")
                large_by_label[lab] = {"available": False, "note": str(exc)}
                continue
            large_by_label[lab] = summary
            large_dict[lab] = {"summary": summary, "rows": rows, "colour": cols[lab]}
            logger.info(
                f"    [{lab}] large-scale noise: ratio {summary['median_ratio']:.2f} "
                f"({summary['n_channels']} channels, sigma {summary['sigma_arcsec']:g} arcsec) "
                f"-> {summary['status']}"
            )

    # Figures
    figures = {
        "summary": fig_summary_table(noise_sums, beam_diags, theory_flag, nchan, nhi_sums),
        "rms_curve": fig_rms_curve(freq, vel, rms_dict, theory_flag),
        "rms_vs_theory": fig_rms_vs_theory(freq, vel, rms_dict, theory_flag),
        "cumulative_rms": fig_cumulative_rms(freq, vel, rms_dict, theory_flag),
        "rms_histogram": fig_rms_histogram(rms_dict),
        "noise_overview": fig_noise_overview(
            freq,
            vel,
            {
                "nchan": nchan,
                "boresight": rref.get("boresight"),
                "median": red[ref_label]["rms"],
                "peak": red[ref_label]["peak"],
                "label": ref_label,
                "colour": cols[ref_label],
            },
            theory_flat,
            theory_flag,
            title=f"Noise overview: {unit.key} ({ref_label})",
        ),
        "spectral": fig_spectral_profile(freq, vel, prof_dict),
        "bandspec": fig_band_spectrum(freq, vel, spec_dict, z_source=cfg.spectral.get("z_source")),
        "nhi": fig_nhi_sensitivity(
            freq, vel, nhi_dict, dv_kms=float(cfg.sensitivity.get("nhi_dv_kms", 20.0))
        ),
        "kurtosis": fig_kurtosis(freq, vel, kurt_dict),
        "beam": fig_beam_stats(freq, vel, beam_dict),
    }
    baseline_dict = {
        lab: {"result": res, "colour": cols[lab]}
        for lab, res in baseline_by_label.items()
        if res.get("available")
    }
    if baseline_dict:
        figures["baseline"] = fig_baseline(freq, vel, baseline_dict)
    if large_dict:
        figures["large_scale"] = fig_large_scale(freq, vel, large_dict, limits=ls_limits)

    # Thumbnails per product (pre-selected planes)
    thumb_sections = []
    for cp in unit.products:
        lab = cp.label
        if lab not in red:
            continue
        r = red[lab]
        rms, _c, quality = rms_dict[lab]
        bmaj, bmin, bpa, _ = beam_dict[lab]
        thumb_sections.append(
            build_thumbnail_html(
                lab,
                r["thumb_planes"],
                r["thumb_reasons"],
                freq,
                vel,
                rms,
                quality,
                bmaj,
                bmin,
                bpa,
                r["hdr"],
            )
        )

    # 2-D maps for the noise-reference product
    map_sections = []
    if want_maps and rref.get("noise_map") is not None:
        map_sections.append(
            build_map_html(
                ref_label, rref["noise_map"], rref["persist_frac_map"], rref["cont_metrics"]
            )
        )

    # Checklist (with continuum criterion)
    rms_arrays = {lab: rms_dict[lab][0] for lab in rms_dict}
    checklist = build_expert_checklist(
        noise_sums,
        beam_diags,
        rms_arrays,
        thr,
        cont_by_label=cont_by_label,
        kurt_by_label=kurt_by_label,
        baseline_by_label=baseline_by_label,
        large_by_label=large_by_label,
    )

    bundle = {
        "anchor": _anchor(unit.pointing, unit.cube_id),
        "pointing": unit.pointing,
        "cube_id": unit.cube_id,
        "products": [cp.product for cp in unit.products if cp.label in red],
        "nchan": nchan,
        "freq_lo": float(freq[0]) / 1e6,
        "freq_hi": float(freq[-1]) / 1e6,
        "figures": figures,
        "checklist": checklist,
        "thumb_sections": thumb_sections,
        "map_sections": map_sections,
        "provenance": prov,
        "nfac": nfac,
        "nfac_src": nfac_src,
        "n_ant": n_ant_use,
        "n_ant_src": n_ant_src,
        "flag_fraction": flag_fraction,
        "pb_corrected": rref["pb_corrected"],
        "overall": _unit_overall(noise_sums),
        # carried through for CSV / survey aggregation
        "_noise_sums": noise_sums,
        "_beam_diags": beam_diags,
        "_beam_arrays": beam_arrays,
        "_nhi_sums": nhi_sums,
        "_cont": cont_by_label,
        "_kurt": kurt_by_label,
        "_baseline": baseline_by_label,
        "_large": large_by_label,
        "_kurt_arrays": {lab: kurt_dict[lab]["kurtosis"] for lab in kurt_dict},
        "_rms_dict": rms_dict,
        "_freq": freq,
        "_vel": vel,
        "_theory": theory_flag,
    }
    return bundle


def _unit_overall(noise_sums: dict) -> str:
    """Worst overall_quality across the unit's products."""
    rank = {"Good": 0, "Warn": 1, "Bad": 2}
    worst = max((rank.get(ns["overall_quality"], 0) for ns in noise_sums.values()), default=0)
    return {0: "Good", 1: "Warn", 2: "Bad"}[worst]


def _session_overall(bundles: list[dict]) -> str:
    """Worst unit grade across the whole session."""
    rank = {"Good": 0, "Warn": 1, "Bad": 2}
    worst = max((rank.get(b.get("overall", "Good"), 0) for b in bundles), default=0)
    return {0: "Good", 1: "Warn", 2: "Bad"}[worst]


# Per-session driver
def run_session(sess_cfg: dict, cfg: QAConfig, logger) -> dict | None:
    """Assess one session; return a summary dict for the master index, or None."""
    root = sess_cfg["root"]
    session = discover_session(
        root, products=cfg.products, pointings=cfg.pointings, cubes=cfg.cubes
    )
    logger.info(summarise(session))

    if not session.units:
        logger.warning(f"  No cubes discovered under {root}; skipping session.")
        return None

    # Flagging (session-level): load once, drive both the theory and the chart
    flag_fig, flag_data = None, None
    flag_paths, flag_pat = discover_flag_reports(root, cfg.flagreport_glob)
    if flag_paths:
        flag_data = load_flag_report(flag_paths)
        flag_fig = fig_flag_summary(flag_data)
        logger.info(f"  Flag reports: {len(flag_paths)} JSON(s) via '{flag_pat}'")
    else:
        logger.warning(
            "  Flag reports: none found (tried the configured glob "
            "and fallbacks); theory will not be flag-reduced."
        )
    flag_by_pointing, flag_default, ff_src = _resolve_flag_fractions(cfg, flag_data)
    if flag_by_pointing:
        shown = ", ".join(f"{p}={v:.2f}" for p, v in sorted(flag_by_pointing.items()))
        logger.info(f"  Flagged fraction [{ff_src}]: {shown} (default {flag_default:.2f})")
    else:
        logger.info(f"  Flagged fraction [{ff_src}] = {flag_default:.3f} (all pointings)")

    # Number of antennas (session-level; sessions can differ, e.g. antennas
    # offline for maintenance)
    rad = cfg.radiometer
    n_ant_cfg = rad.get("n_ant", "auto")
    if isinstance(n_ant_cfg, str) and n_ant_cfg.strip().lower() == "auto":
        n_ant, n_ant_src = obsinfo.read_n_ant(root)
        if n_ant is None:
            n_ant_src = f"{n_ant_src} -> fallback {rad.get('n_ant_fallback', 60)}"
    else:
        try:
            n_ant, n_ant_src = int(n_ant_cfg), "config"
        except (ValueError, TypeError):
            n_ant, n_ant_src = None, "bad config value -> fallback"
    logger.info(f"  N_ant [{n_ant_src}]" + (f" = {n_ant}" if n_ant else ""))

    bundles = []
    for unit in session.units:
        ff = flag_by_pointing.get(unit.pointing, flag_default)
        b = assess_unit(unit, cfg, logger, flag_fraction=ff, n_ant=n_ant, n_ant_src=n_ant_src)
        if b is not None:
            bundles.append(b)

    if not bundles:
        logger.warning(f"  Session {session.obsid}: nothing assessed; skipping.")
        return None

    # Session-level CSVs
    if cfg.output["csv"]:
        chan_csv = cfg.session_out_path(session.obsid, "_channel_qa.csv")
        first = True
        for b in bundles:
            for lab, (rms, _c, quality) in b["_rms_dict"].items():
                bmaj, bmin, bpa = b["_beam_arrays"][lab]
                write_channel_csv(
                    chan_csv,
                    f"{b['pointing']}/{b['cube_id']}/{lab}",
                    b["_freq"],
                    b["_vel"],
                    rms,
                    quality,
                    b["_theory"],
                    bmaj,
                    bmin,
                    bpa,
                    append=not first,
                    kurtosis=b["_kurt_arrays"].get(lab),
                )
                first = False
        logger.info(f"  Per-channel CSV -> {chan_csv}")

        band_csv = cfg.session_out_path(session.obsid, "_band_summary.csv")
        merged_ns, merged_bd, merged_nhi, merged_cont, merged_kurt = {}, {}, {}, {}, {}
        merged_baseline, merged_large = {}, {}
        for b in bundles:
            for lab in b["_noise_sums"]:
                key = f"{b['pointing']}/{b['cube_id']}/{lab}"
                merged_ns[key] = b["_noise_sums"][lab]
                merged_bd[key] = b["_beam_diags"][lab]
                merged_nhi[key] = b["_nhi_sums"].get(lab, {})
                merged_cont[key] = b["_cont"].get(lab, {})
                merged_kurt[key] = b["_kurt"].get(lab, {})
                merged_baseline[key] = b["_baseline"].get(lab, {})
                merged_large[key] = b["_large"].get(lab, {})
        write_band_summary_csv(
            band_csv,
            merged_ns,
            merged_bd,
            cfg,
            nhi_by_label=merged_nhi,
            cont_by_label=merged_cont,
            kurt_by_label=merged_kurt,
            baseline_by_label=merged_baseline,
            large_by_label=merged_large,
        )
        logger.info(f"  Band summary CSV -> {band_csv}")

        check_csv = cfg.session_out_path(session.obsid, "_checklist.csv")
        all_checks = []
        for b in bundles:
            for row in b["checklist"]:
                row = dict(row)
                row["cube_label"] = f"{b['pointing']}/{b['cube_id']}/{row['cube_label']}"
                all_checks.append(row)
        write_expert_checklist_csv(check_csv, all_checks)
        logger.info(f"  Checklist CSV -> {check_csv}")

    # HTML
    html_name = None
    if cfg.output["html"]:
        html_path = cfg.session_out_path(session.obsid, "_qa_report.html")
        size_mb = build_session_report(
            html_path,
            cfg,
            session.obsid,
            bundles,
            flag_fig,
            cfg.radiometer,
            flag_fraction=flag_default,
        )
        html_name = html_path.name
        logger.info(f"  HTML report -> {html_path} ({size_mb:.1f} MB)")

    # Standalone PNGs of key figures (optional; needs kaleido)
    if cfg.output.get("save_png"):
        png_dir = cfg.out_dir() / "pngs" / session.obsid
        fig_ids = cfg.output.get("png_figures", [])
        scale = cfg.output.get("png_scale", 2)
        total = 0
        for b in bundles:
            prefix = f"{b['pointing']}_{b['cube_id']}"
            n, ok, err = export_figs_png(b["figures"], png_dir, prefix, fig_ids, scale)
            total += n
            if not ok:
                detail = next(
                    (ln for ln in (err or "").splitlines() if ln.strip()), "kaleido not available"
                )
                logger.warning(
                    f"  PNG export skipped ({detail[:120]}); "
                    f"install with 'pip install kaleido'. HTML is unaffected."
                )
                break
        else:
            logger.info(f"  PNGs -> {png_dir} ({total} files)")

    # Survey rollup rows (one per unit/product)
    survey_rows = []
    for b in bundles:
        for lab, ns in b["_noise_sums"].items():
            nh = b["_nhi_sums"].get(lab, {})
            cm = b["_cont"].get(lab, {})
            kt = b["_kurt"].get(lab, {})
            bl = b["_baseline"].get(lab) or {}
            ls = b["_large"].get(lab) or {}
            survey_rows.append(
                {
                    "obsid": session.obsid,
                    "pointing": b["pointing"],
                    "cube": b["cube_id"],
                    "product": lab,
                    "n_channels": ns["n_total"],
                    "median_rms_mjy": round(ns["median_rms_mjy"], 4),
                    "rms_to_theory": round(ns["rms_to_theory"], 3) if ns["rms_to_theory"] else None,
                    "nhi_3sig_dv_cm2": nh.get("nhi_dv_cm2"),
                    "pct_bad": round(ns["pct_bad"], 2),
                    "pct_pixels_persistent": round(cm["pct_pixels_persistent"], 3)
                    if cm.get("available")
                    else None,
                    "pct_bad_kurtosis": round(kt["pct_bad"], 2) if kt.get("available") else None,
                    "baseline_sigma": round(bl["baseline_sigma"], 2)
                    if bl.get("available")
                    else None,
                    "ls_median_ratio": round(ls["median_ratio"], 3)
                    if ls.get("available")
                    else None,
                    "ls_status": ls["status"] if ls.get("available") else None,
                    "pb_corrected": b["pb_corrected"],
                    "overall": ns["overall_quality"],
                }
            )

    return {
        "obsid": session.obsid,
        "html": html_name,
        "n_pointings": len({b["pointing"] for b in bundles}),
        "n_units": len(bundles),
        "products": cfg.products,
        "overall": _session_overall(bundles),
        "survey_rows": survey_rows,
    }


def run(cfg: QAConfig) -> None:
    t0 = time.time()
    logger = setup_logger(cfg.out_dir() / "hi_cube_qa.log" if cfg.output["log"] else None)
    logger.info("hi_cube_qa starting (session mode)")
    logger.info(f"Config : {cfg}")
    logger.info(f"Output : {cfg.output['dir']}")
    if cfg.plots.get("full_resolution"):
        figures.set_max_plot_points(None)
        logger.info("Plots  : full resolution (every channel)")
    else:
        mp = cfg.plots.get("max_points")
        figures.set_max_plot_points(mp)
        logger.info(
            f"Plots  : downsampled to <= {mp} points/trace (Bad channels kept; CSVs full-res)"
        )

    # One-time PNG-export capability check (avoids a stray empty pngs/
    # folder and repeating the same failed attempt once per session)
    if cfg.output.get("save_png"):
        ok, err = png_export_available()
        if ok:
            logger.info("PNG export: kaleido OK")
        else:
            detail = next(
                (ln for ln in (err or "").splitlines() if ln.strip()), "kaleido not available"
            )
            logger.warning(
                f"PNG export disabled for this run ({detail[:160]}); "
                f"install with 'pip install kaleido'. HTML is unaffected."
            )
            cfg._cfg["output"]["save_png"] = False

    summaries = []
    for i, sess_cfg in enumerate(cfg.sessions):
        logger.info("=" * 70)
        logger.info(f"SESSION {i + 1}/{len(cfg.sessions)} : {sess_cfg['root']}")
        logger.info("=" * 70)
        try:
            s = run_session(sess_cfg, cfg, logger)
            if s is not None:
                summaries.append(s)
        except Exception as exc:
            logger.error(f"  Session failed: {exc}", exc_info=True)

    # Survey-level rollup across all sessions
    survey_rows = [r for s in summaries for r in s.get("survey_rows", [])]
    if survey_rows:
        sj = cfg.out_dir() / "qa_survey_summary.json"
        sc = cfg.out_dir() / "qa_survey_summary.csv"
        write_survey_summary(sj, sc, survey_rows)
        logger.info(f"Survey rollup -> {sc} ({len(survey_rows)} rows) and {sj.name}")

    # Master index over all sessions
    if cfg.output["html"] and any(s["html"] for s in summaries):
        master = cfg.out_dir() / "qa_index.html"
        size_mb = build_master_index(master, summaries)
        logger.info("=" * 70)
        logger.info(f"MASTER INDEX -> {master} ({size_mb:.2f} MB)")
        logger.info("   Open this one file to browse all sessions.")
        logger.info("=" * 70)
    elif len(summaries) == 0:
        logger.warning("No sessions produced output; master index not written.")

    logger.info(f"Done. Total wall time: {time.time() - t0:.1f} s")


def main() -> None:
    args = parse_args()

    if args.example_config:
        write_example_config("qa_session_example.yaml")
        sys.exit(0)

    if args.config is None:
        print("Error: --config is required (or use --example-config).", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    if args.outdir:
        cfg._cfg["output"]["dir"] = args.outdir
    if args.full_res is not None:  # --full-res / --downsample override
        cfg._cfg["plots"]["full_resolution"] = args.full_res

    if args.list:
        for sess_cfg in cfg.sessions:
            session = discover_session(
                sess_cfg["root"], products=cfg.products, pointings=cfg.pointings, cubes=cfg.cubes
            )
            print(summarise(session))
        sys.exit(0)

    run(cfg)


if __name__ == "__main__":
    main()
