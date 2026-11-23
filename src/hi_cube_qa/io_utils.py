from __future__ import annotations
import json
import os
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
##############################################################################
#
#  io_utils.py: FITS cube loading and CARACal flag-report parsing.
#
##############################################################################
HI_REST_FREQ_HZ = 1.420405752e9  # Hz
C_KMS           = 2.99792458e5   # km/s

#--- Cube Loading ---
def _find_spectral_axis(hdr):
    """
    Find the spectral axis in a FITS header.

    Returns (axis_number, axis_type) where axis_type is one of:
        'FREQ', 'VELO', 'VOPT', 'VRAD', or None if not found.

    CARACal / WSClean cubes may store the spectral axis as:
      - FREQ       : frequency in Hz
      - VELO-LSR   : radio velocity in m/s (LSR frame)
      - VOPT       : optical velocity in m/s
      - VRAD       : radio velocity in m/s
    """
    naxis = hdr.get("NAXIS", 0)
    spectral_prefixes = ("FREQ", "VELO", "VOPT", "VRAD")
    for ax in range(1, naxis + 1):
        ctype = hdr.get(f"CTYPE{ax}", "").upper().strip()
        for prefix in spectral_prefixes:
            if ctype.startswith(prefix):
                return ax, prefix
    return None, None

def _get_rest_freq(hdr):
    """Extract rest frequency from header (RESTFRQ or RESTFREQ keyword)."""
    for key in ("RESTFRQ", "RESTFREQ"):
        val = hdr.get(key, None)
        if val is not None and float(val) > 0:
            return float(val)
    return HI_REST_FREQ_HZ   # default to HI 21cm

_KMS = u.km / u.s


def _hz_to_radio_kms(freq_hz, rest_hz):
    """Frequency -> radio-convention velocity (astropy doppler_radio)."""
    return (np.asarray(freq_hz, dtype=float) * u.Hz).to(
            _KMS, u.doppler_radio(rest_hz * u.Hz)).value


def _radio_kms_to_hz(vel_kms, rest_hz):
    """Radio-convention velocity -> frequency (astropy doppler_radio)."""
    return (np.asarray(vel_kms, dtype=float) * _KMS).to(
            u.Hz, u.doppler_radio(rest_hz * u.Hz)).value


def frequency_velocity(hdr, nchan: int, quiet: bool = False) -> tuple:
    """
    Build matched (freq_hz, vel_kms) arrays of length *nchan* from a header.

    Handles spectral axes stored as FREQ, VELO/VRAD, or VOPT, using the radio
    velocity convention for the returned velocities. Falls back to a dummy
    linear frequency grid (with a warning) if no spectral axis is found, or if
    the axis length disagrees with *nchan*.
    """
    spec_ax, spec_type = _find_spectral_axis(hdr)
    rest_freq = _get_rest_freq(hdr)
    freq_hz = vel_kms = None

    if spec_ax is not None:
        n     = int(hdr.get(f"NAXIS{spec_ax}", nchan))
        crpix = hdr.get(f"CRPIX{spec_ax}", 1)
        crval = hdr.get(f"CRVAL{spec_ax}", 0)
        cdelt = hdr.get(f"CDELT{spec_ax}", 1)
        pix_vals = crval + (np.arange(1, n + 1) - crpix) * cdelt

        if spec_type == "FREQ":
            freq_hz = pix_vals
            vel_kms = _hz_to_radio_kms(freq_hz, rest_freq)
        elif spec_type in ("VELO", "VRAD"):
            cunit   = hdr.get(f"CUNIT{spec_ax}", "m/s").strip().lower()
            vel_kms = pix_vals if cunit == "km/s" else pix_vals / 1e3
            freq_hz = _radio_kms_to_hz(vel_kms, rest_freq)
        elif spec_type == "VOPT":
            cunit   = hdr.get(f"CUNIT{spec_ax}", "m/s").strip().lower()
            vel_opt = pix_vals if cunit == "km/s" else pix_vals / 1e3
            freq_hz = (vel_opt * _KMS).to(
                    u.Hz, u.doppler_optical(rest_freq * u.Hz)).value
            vel_kms = _hz_to_radio_kms(freq_hz, rest_freq)

        if freq_hz is not None and len(freq_hz) != nchan:
            freq_hz = vel_kms = None   # header axis disagrees with data

    if freq_hz is None:
        freq_hz = np.linspace(1.30e9, 1.43e9, nchan)
        vel_kms = _hz_to_radio_kms(freq_hz, rest_freq)
        if not quiet:
            warnings.warn(
                    "[io_utils] No usable spectral axis; using a dummy frequency "
                    "grid.", stacklevel=2)

    return np.asarray(freq_hz, dtype=float), np.asarray(vel_kms, dtype=float)


def load_cube(path: str | Path) -> tuple:
    """
    Loads a FITS spectral cube fully into memory.

    Handles spectral axes stored as FREQ, VELO-LSR, VOPT, or VRAD. Retained for
    callers that want the whole cube; the QA pipeline uses the streaming
    CubeHandle in reduce.py instead.
    """
    path = str(path)
    with fits.open(path) as hdul:
        hdr  = hdul[0].header
        data = hdul[0].data.squeeze()  # remove degenerate axes (Stokes etc.)
    wcs_full = WCS(hdr, naxis=hdr["NAXIS"])

    if data.ndim == 2:
        data = data[np.newaxis, :]
    elif data.ndim != 3:
        raise ValueError(
                f"[io_utils] Unexpected cube shape after squeeze: {data.shape} "
                f"from {path}")

    freq_hz, vel_kms = frequency_velocity(hdr, data.shape[0])
    return data, hdr, wcs_full, freq_hz, vel_kms

#--- Flag report ---
def load_flag_report(paths) -> dict | None:
    """
    Load one or more CARACal flag-report JSON files.

    Parameters
    ----------
    paths : str, Path, list[str|Path], or None
        A single path, a list of paths, or None.

    Returns
    -------
    dict or None
        Combined flag data keyed by a label derived from the filename.
        Each key maps to the JSON contents of that report.
    """
    if paths is None:
        return None

    # Normalise to a list
    if isinstance(paths, (str, Path)):
        paths = [paths]

    combined = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            warnings.warn(f"[io_utils] Flag report not found: {p}", stacklevel=2)
            continue
        # Derive a short label from the filename
        # e.g. "1608285672_sdp_l0-cal-flag__bpcal-flagging-summary-0.json"
        #   -> "cal-flag / bpcal"
        stem = p.stem  # without .json
        parts = stem.split("-flagging-summary")[0]  # everything before "-flagging-summary"
        # Try to extract the stage name after "__"
        if "__" in parts:
            prefix, stage = parts.rsplit("__", 1)
            # Shorten the prefix: take last meaningful segment
            prefix_short = prefix.split("-", 1)[-1] if "-" in prefix else prefix
            label = f"{prefix_short} / {stage}"
        else:
            label = parts

        with open(p) as f:
            data = json.load(f)
        combined[label] = data

    return combined if combined else None

def flag_fraction_of(content) -> float | None:
    """
    Overall flagged fraction (0..1) from one CARACal flagging-summary JSON.

    Real CARACal reports look like::

        {"Flag stats": [{"scans": {...}}, {"antennas": {...}},
                        {"fields": {...}},
                        {"corrs": {"0": {"name": "XX", "frac": 0.137,
                                         "sum": 9.6e8, "counts": 7.0e9}, ...}}]}

    Each category partitions the same data, so summing ``sum``/``counts`` over
    any one category's entries gives the overall flagged fraction. We prefer the
    smallest category ('corrs'). A generic 'flagged'/'fraction'/'frac' search is
    kept as a fallback for other formats.
    """
    if not isinstance(content, dict):
        return None

    stats = content.get("Flag stats")
    if isinstance(stats, list):
        cats: dict = {}
        for item in stats:
            if isinstance(item, dict):
                cats.update(item)
        for pref in ("corrs", "fields", "scans", "antennas"):
            cat = cats.get(pref)
            if isinstance(cat, dict) and cat:
                tot_sum = tot_cnt = 0.0
                fracs = []
                for entry in cat.values():
                    if not isinstance(entry, dict):
                        continue
                    s, c = entry.get("sum"), entry.get("counts")
                    if s is not None and c is not None:
                        try:
                            tot_sum += float(s)
                            tot_cnt += float(c)
                        except (ValueError, TypeError):
                            pass
                    elif entry.get("frac") is not None:
                        try:
                            fracs.append(float(entry["frac"]))
                        except (ValueError, TypeError):
                            pass
                if tot_cnt > 0:
                    return tot_sum / tot_cnt
                if fracs:
                    return sum(fracs) / len(fracs)

    # Generic fallback: shallowest flagged/fraction/frac value.
    from collections import deque
    queue = deque([content])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            fr = node.get("flagged", node.get("fraction", node.get("frac")))
            if fr is not None:
                try:
                    v = float(fr)
                    return v / 100.0 if v > 1.0 else v   # tolerate percentages
                except (ValueError, TypeError):
                    pass
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return None


#--- Header utilities ---
def pixel_scale_asec(hdr: fits.Header) -> float:
    """
    Return the pixel scale in arcseconds (absolute value of CDELT1/2).
    """
    cdelt1 = abs(hdr.get("CDELT1", 6.0 / 3600.0))   #   degrees/pixel
    return cdelt1 * 3600.0

def channel_width_hz(hdr: fits.Header) -> float:
    """
    Extract channel width in Hz from the FITS header.
    Handles both FREQ and VELO/VOPT/VRAD axes.
    """
    spec_ax, spec_type = _find_spectral_axis(hdr)
    if spec_ax is None:
        return 26367.19  # fallback

    cdelt = abs(hdr.get(f"CDELT{spec_ax}", 0))
    if cdelt == 0:
        return 26367.19

    if spec_type == "FREQ":
        return cdelt   # already in Hz

    # Velocity axis: convert dv to df
    # df = f0 * dv / c  (radio convention)
    rest_freq = _get_rest_freq(hdr)
    cunit = hdr.get(f"CUNIT{spec_ax}", "m/s").strip().lower()
    if cunit == "km/s":
        dv_kms = cdelt
    else:
        dv_kms = cdelt / 1e3   # m/s -> km/s
    return rest_freq * dv_kms / C_KMS
