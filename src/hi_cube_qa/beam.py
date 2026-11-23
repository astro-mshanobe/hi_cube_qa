from __future__ import annotations
import warnings

import numpy as np
from astropy.io import fits

################################################################################
#
# beam.py: beam properties
#
################################################################################
def beam_stats(
        hdr: fits.Header,
        nchan: int,
        ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Extract synthesised beam parameters from a FITS header.

    Handles three cases in priority order:
        1. Single-valued BMAJ/BMIN/BPA keywords (broadcast to all channels)
        2. Per-channel WSClean-style keywords BMAJ{i:04d} / BMIN{i:04d} / BPA{i:04d}
        3. BEAMTABLE binary extension (not yet implemented - falls through to warning)

    Returns
    -------
    bmaj_asec : (nchan,) in arcsec, or None
    bmin_asec : (nchan,) in arcsec, or None
    bpa_deg   : (nchan,) in degrees, or None
    """
    #--- Detect per-channel beam keywords first (preferred over global) ---
    # WSClean uses several conventions depending on version:
    #   BMAJ0000, BMIN0000, BPA0000   (zero-padded, 0-indexed)
    #   BMAJ1, BMIN1, BPA1            (no padding, 1-indexed)
    # When channel numbers exceed keyword length (8 chars), WSClean
    # truncates: BMAJ1000 is fine, but channel indices >9999 would need
    # BMA10000 (truncated). We also see truncations like BMA1, BMI32, etc.
    # Strategy: try both conventions, prefer the one that exists.

    per_chan_format = None
    if "BMAJ0000" in hdr:
        per_chan_format = "zero_padded_0indexed"     # BMAJ0000, BMAJ0001, ...
    elif "BMAJ1" in hdr or "BMA1" in hdr:
        per_chan_format = "wsclean_1indexed"          # BMAJ1, BMAJ2, ...

    if per_chan_format is not None:
        global_bmaj = hdr.get("BMAJ", 0)
        global_bmin = hdr.get("BMIN", 0)
        global_bpa  = hdr.get("BPA", 0.0)
        bmaj_list, bmin_list, bpa_list = [], [], []

        for i in range(nchan):
            if per_chan_format == "zero_padded_0indexed":
                key_bmaj = f"BMAJ{i:04d}"
                key_bmin = f"BMIN{i:04d}"
                key_bpa  = f"BPA{i:04d}"
            else:
                # 1-indexed, no padding
                ch = i + 1
                key_bmaj = f"BMAJ{ch}"
                key_bmin = f"BMIN{ch}"
                key_bpa  = f"BPA{ch}"
                # Handle FITS 8-char keyword truncation:
                # BMAJ9999 is fine (8 chars), but BMAJ10000 would need BMA10000
                if len(key_bmaj) > 8:
                    key_bmaj = f"BMA{ch}"
                if len(key_bmin) > 8:
                    key_bmin = f"BMI{ch}"

            # Try the computed key, then the truncated variant, then fall back to global
            bm = hdr.get(key_bmaj, None)
            if bm is None and per_chan_format == "wsclean_1indexed":
                # Try truncated form (WSClean truncates BMAJ -> BMA for some channels)
                bm = hdr.get(f"BMA{i+1}", None)
            if bm is None:
                bm = global_bmaj

            bn = hdr.get(key_bmin, None)
            if bn is None and per_chan_format == "wsclean_1indexed":
                bn = hdr.get(f"BMI{i+1}", None)
            if bn is None:
                bn = global_bmin

            bp = hdr.get(key_bpa, hdr.get(f"BPA{i+1}" if per_chan_format == "wsclean_1indexed" else key_bpa, global_bpa))

            bmaj_list.append(float(bm) * 3600.0)
            bmin_list.append(float(bn) * 3600.0)
            bpa_list.append(float(bp))

        return np.array(bmaj_list), np.array(bmin_list), np.array(bpa_list)

    #--- Fallback: single global beam ---
    bmaj = hdr.get("BMAJ", None)    # degrees
    bmin = hdr.get("BMIN", None)
    bpa  = hdr.get("BPA", 0.0)

    if bmaj is not None and bmin is not None:
        bmaj_asec = np.full(nchan, float(bmaj) * 3600.0)
        bmin_asec = np.full(nchan, float(bmin) * 3600.0)
        bpa_deg   = np.full(nchan, float(bpa))
        return bmaj_asec, bmin_asec, bpa_deg

    warnings.warn(
            "[beam] No beam info found in header (BMAJ/BMIN keywords or per-channel "
            "BMAJ{i:04d} keywords absent). Beam plots will be empty.",
            stacklevel=2
            )
    return None, None, None

def beam_diagnostics(
        bmaj : np.ndarray | None,
        bmin : np.ndarray | None,
        bpa  : np.ndarray | None,
        scatter_warn_asec: float = 0.5,
        ) -> dict:
    """
    Compute summary statistics for the synthesised beam across channels.
    """
    if bmaj is None:
        return {"available": False}

    finite = np.isfinite(bmaj)
    bmaj_f = bmaj[finite]
    bmin_f = bmin[finite] if bmin is not None else bmaj_f * np.nan
    bpa_f  = bpa[finite]  if bpa is not None else np.zeros_like(bmaj_f)

    med_bmaj = float(np.nanmedian(bmaj_f))
    med_bmin = float(np.nanmedian(bmin_f))
    med_bpa  = float(np.nanmedian(bpa_f))
    std_bmaj = float(np.nanstd(bmaj_f))
    std_bmin = float(np.nanstd(bmin_f))
    std_bpa  = float(np.nanstd(bpa_f))
    max_bmaj = float(np.nanmax(bmaj_f))
    max_bmin = float(np.nanmax(bmin_f))
    max_bpa  = float(np.nanmax(bpa_f))
    min_bmaj = float(np.nanmin(bmaj_f))
    min_bmin = float(np.nanmin(bmin_f))
    min_bpa  = float(np.nanmin(bpa_f))

    beam_area = (np.pi / (4.0 * np.log(2.0))) * med_bmaj * med_bmin
    axial     = med_bmaj / med_bmin if med_bmin > 0 else np.nan

    scatter_quality = "Warn" if std_bmaj > scatter_warn_asec else "Good"

    return {
            "available":            True,
            "median_bmaj_asec":     med_bmaj,
            "median_bmin_asec":     med_bmin,
            "median_bpa_deg":       med_bpa,
            "std_bmaj_asec":        std_bmaj,
            "std_bmin_asec":        std_bmin,
            "std_bpa_deg":          std_bpa,
            "max_bmaj_asec":        max_bmaj,
            "max_bmin_asec":        max_bmin,
            "max_bpa_deg":          max_bpa,
            "min_bmaj_asec":        min_bmaj,
            "min_bmin_asec":        min_bmin,
            "min_bpa_deg":          min_bpa,
            "beam_area_asec2":      beam_area,
            "axial_ratio":          axial,
            "bmaj_scatter_quality": scatter_quality,
            }
