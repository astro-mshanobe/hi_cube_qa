"""Per-session observation parameters from CARACal's obsinfo summary.json.

Only the antenna count is read. Per-channel gridded-visibility counts from the
WSClean imaging logs were tried as a replacement for the scalar flagged
fraction and dropped: the log has one "Gridded visibility count" line per
channel group with no channel index, so runs of fully flagged channels cannot
be told apart from a single empty channel (on the G23 data one run was 1442
identical lines against a typical 4-5). The flagged-fraction theory
(io_utils.flag_fraction_of) and the kurtosis diagnostic (noise.kurtosis_planes)
are used instead.
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
