"""wsclean_log: the antenna count from CARACal's obsinfo summary."""

import json

from helpers import uses_tmp_path

from hi_cube_qa import wsclean_log as wl


def _summary(root, content):
    d = root / "output_hi_1gc" / "obsinfo"
    d.mkdir(parents=True)
    p = d / "1600000000-obsinfo-summary.json"
    p.write_text(content if isinstance(content, str) else json.dumps(content))
    return p


@uses_tmp_path
def test_antenna_count_is_column_length(
    tmp_path,
):
    cols = ("NAME", "STATION", "OFFSET", "POSITION", "DISH_DIAMETER", "MOUNT", "TYPE", "FLAG_ROW")
    _summary(tmp_path, {"ANT": {c: list(range(58)) for c in cols}})
    n, src = wl.read_n_ant(tmp_path)
    assert n == 58 and "ANT.NAME" in src and "58 antennas" in src


@uses_tmp_path
def test_fallback_column(tmp_path):
    _summary(tmp_path, {"ANT": {"NAME": [], "STATION": ["a"], "POSITION": [1, 2, 3, 4]}})
    assert wl.read_n_ant(tmp_path)[0] == 4


@uses_tmp_path
def test_missing_or_malformed_summary(tmp_path):
    assert wl.read_n_ant(tmp_path) == (None, "no obsinfo summary.json found")
    _summary(tmp_path, "{not json")
    n, why = wl.read_n_ant(tmp_path)
    assert n is None and "failed to read" in why


@uses_tmp_path
def test_bad_ant_field(tmp_path):
    p = _summary(tmp_path, {"other": 1})
    assert wl.read_n_ant(tmp_path)[1].endswith("no ANT field")
    p.write_text(json.dumps({"ANT": [1, 2]}))
    assert "unexpected type list" in wl.read_n_ant(tmp_path)[1]
    p.write_text(json.dumps({"ANT": {"NAME": ["only one"]}}))
    assert "no usable per-antenna column" in wl.read_n_ant(tmp_path)[1]
