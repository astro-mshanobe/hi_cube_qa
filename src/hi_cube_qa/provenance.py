"""Read imaging provenance from FITS headers and derive the
weighting noise penalty (nfac) automatically.

WSClean records its settings in the FITS header (and HISTORY). The Briggs
robust value, niter, threshold, restoring beam, and basic observation metadata
are read from it, so the report can stamp how the cube was made and nfac can be
set from the actual weighting rather than by hand.
"""

from __future__ import annotations

import numpy as np

# Approximate natural-relative noise penalty for MeerKAT Briggs weighting.
# Natural = 1.0; uniform ~ 1.4. Robust interpolates. These are typical values and
# the exact penalty depends on the uv-coverage of each track. The only source for
# them is the NRAO VLA Observational Status Summary ("Sensitivity"), which gives
# about 1.2 for typical robust weighting with the VLA; the other entries, and the
# MeerKAT numbers, are unsourced.
_ROBUST_NFAC = [
    (-2.0, 1.40),  # ~ uniform
    (-1.0, 1.35),
    (-0.5, 1.20),
    (0.0, 1.10),
    (0.5, 1.04),
    (1.0, 1.02),
    (2.0, 1.00),  # ~ natural
]


def nfac_from_robust(robust: float | None) -> float | None:
    """Interpolate the weighting noise penalty from a Briggs robust value."""
    if robust is None:
        return None
    r = np.array([x[0] for x in _ROBUST_NFAC])
    n = np.array([x[1] for x in _ROBUST_NFAC])
    return float(np.interp(robust, r, n))


def _history_text(hdr) -> str:
    try:
        return "\n".join(str(c) for c in hdr.get("HISTORY", []))
    except Exception:
        return ""


def _find_robust(hdr) -> float | None:
    """Find the Briggs robust value. WSClean may store it as a keyword
    (WSCWEIGH = 'Briggs'(-0.5)) or only in the HISTORY (-weight briggs -0.5).
    """
    import re

    # Keyword form, e.g. WSCWEIGH = 'Briggs'(-0.5) or 'briggs -0.5'
    for key in ("WSCWEIGH", "WEIGHT", "WTSCHEME"):
        v = hdr.get(key)
        if v is None:
            continue
        m = re.search(r"[-+]?\d*\.?\d+", str(v))
        if m and "briggs" in str(v).lower():
            try:
                return float(m.group())
            except ValueError:
                pass
    # HISTORY form: "... -weight briggs -0.5 ..."
    hist = _history_text(hdr).lower()
    m = re.search(r"briggs\s+([-+]?\d*\.?\d+)", hist)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    if "natural" in hist:
        return 2.0
    if "uniform" in hist:
        return -2.0
    return None


def _find_int(hdr, keys, hist_pat) -> int | None:
    import re

    for k in keys:
        v = hdr.get(k)
        if v is not None:
            try:
                return int(float(v))
            except (ValueError, TypeError):
                pass
    m = re.search(hist_pat, _history_text(hdr).lower())
    if m:
        try:
            return int(float(m.group(1)))
        except ValueError:
            pass
    return None


def read_provenance(hdr, beam_diag: dict | None = None) -> dict:
    """Extract imaging/observation provenance from a FITS header.

    Returns a dict of human-readable strings/values (None where unknown):
        telescope, object, date_obs, origin, bunit,
        robust, niter, threshold, restoring_beam, weight_raw
    """
    robust = _find_robust(hdr)
    niter = _find_int(hdr, ("WSCNITER", "NITER"), r"-niter\s+(\d+)")

    threshold = None
    for k in ("WSCTHRES", "WSCThreshold", "THRESH"):
        v = hdr.get(k)
        if v is not None:
            try:
                threshold = float(v)
                break
            except (ValueError, TypeError):
                pass

    restoring = None
    if beam_diag and beam_diag.get("available"):
        restoring = (
            f'{beam_diag["median_bmaj_asec"]:.2f}" x '
            f'{beam_diag["median_bmin_asec"]:.2f}", '
            f"PA {beam_diag['median_bpa_deg']:.1f} deg"
        )

    return {
        "telescope": str(hdr.get("TELESCOP", "")).strip() or None,
        "object": str(hdr.get("OBJECT", "")).strip() or None,
        "date_obs": str(hdr.get("DATE-OBS", "")).strip() or None,
        "origin": str(hdr.get("ORIGIN", "")).strip() or None,
        "bunit": str(hdr.get("BUNIT", "")).strip() or None,
        "robust": robust,
        "niter": niter,
        "threshold": threshold,
        "restoring_beam": restoring,
        "weight_raw": str(hdr.get("WSCWEIGH", "")).strip() or None,
    }


def resolve_nfac(cfg_nfac, hdr) -> tuple[float, str]:
    """Resolve the radiometer weighting penalty.

    If *cfg_nfac* is the string 'auto', derive it from the header robust value
    (falling back to 1.0 if not found). Otherwise use the configured number.
    Returns (nfac, source_description).
    """
    if isinstance(cfg_nfac, str) and cfg_nfac.strip().lower() == "auto":
        robust = _find_robust(hdr)
        nf = nfac_from_robust(robust)
        if nf is not None:
            return nf, f"auto (robust={robust:+.1f})"
        return 1.0, "auto (robust unknown -> 1.0)"
    try:
        return float(cfg_nfac), "config"
    except (ValueError, TypeError):
        return 1.0, "default"
