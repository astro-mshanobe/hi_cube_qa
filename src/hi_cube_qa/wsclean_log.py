"""Read per-session observation parameters from CARACal's
obsinfo summary.json.

An earlier version of this module also parsed per-channel gridded-
visibility counts from WSClean's imaging logs, to replace the scalar
flagged-fraction proxy with a real per-channel quantity (mirroring the
CHILES survey's noise validation, Luber et al. 2025). That was abandoned:
WSClean logs a generic "Gridded visibility count: <N>" line with no
channel index, and long runs of fully-flagged channels all log the
identical value with nothing to disambiguate "one empty channel" from
"hundreds in a row" -- confirmed on real G23 data, where a single run of
1442 identical log lines (versus a typical run of 4-5) made the true
channel boundaries fundamentally unrecoverable from the log text. The
flag-fraction theory (io_utils.flag_fraction_of) and the kurtosis
diagnostic (noise.kurtosis_planes) remain the validated tools for this.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_obsinfo_summary(session_root: str | Path) -> tuple[dict | None, Path | None, str]:
    """Locate and load the CARACal obsinfo summary.json for a session."""
    session_root = Path(session_root)
    matches = sorted(session_root.glob("output_hi_1gc/obsinfo/*-summary.json"))
    if not matches:
        return None, None, "no obsinfo summary.json found"
    try:
        with open(matches[0]) as f:
            return json.load(f), matches[0], ""
    except (OSError, json.JSONDecodeError) as exc:
        return None, matches[0], f"failed to read {matches[0].name}: {exc}"


def read_n_ant(session_root: str | Path) -> tuple[int | None, str]:
    """Best-effort number of antennas for a session, read from the CARACal
    obsinfo summary.json's 'ANT' field.

    'ANT' is keyed by MS ANTENNA-table *column* name (NAME, STATION, OFFSET,
    POSITION, DISH_DIAMETER, MOUNT, TYPE, FLAG_ROW), each mapping to a list
    with one entry per antenna - confirmed directly on real G23 data, where
    the dict has exactly 8 top-level keys (the fixed column count) while each
    column's list has one entry per real antenna (58, in that case). Taking
    len(ant) would therefore silently return the column count (e.g. 8) rather
    than the antenna count - this is the bug that motivated reading the real
    per-antenna list length here instead.
    """
    d, path, err = _load_obsinfo_summary(session_root)
    if d is None:
        return None, err

    ant = d.get("ANT")
    if ant is None:
        return None, f"{path.name}: no ANT field"
    if not isinstance(ant, dict):
        return None, f"{path.name}: ANT field has unexpected type {type(ant).__name__}"

    # Prefer a column that unambiguously holds one entry per antenna.
    for col in (
        "NAME",
        "STATION",
        "OFFSET",
        "POSITION",
        "DISH_DIAMETER",
        "MOUNT",
        "TYPE",
        "FLAG_ROW",
    ):
        v = ant.get(col)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return len(v), f"{path.name} (ANT.{col}, {len(v)} antennas)"

    return None, f"{path.name}: no usable per-antenna column found in ANT"
