"""obsinfo: the antenna count and the on-source time from CARACal's obsinfo summary."""

import json

from helpers import uses_tmp_path

from hi_cube_qa import obsinfo as wl


def _summary(root, content):
    d = root / "output_hi_1gc" / "obsinfo"
    d.mkdir(parents=True, exist_ok=True)
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


@uses_tmp_path
def test_on_source_hours_are_the_sum_of_the_scan_durations_of_each_field(tmp_path):
    summary = {
        "FIELD": {"NAME": ["J1939-6342", "J2302-3718", "p63", "p64"]},
        "SCAN": {
            "0": {"1": 600.0, "35": 591.75},
            "1": {"2": 104.0},
            "2": {"3": 1800.0, "4": 1800.0, "5": 360.0},
            "3": {"6": 3600.0},
        },
    }
    _summary(tmp_path, summary)
    hours, source = wl.read_on_source_hours(tmp_path)
    assert abs(hours["p63"] - 3960.0 / 3600.0) < 1e-12 and abs(hours["p64"] - 1.0) < 1e-12
    assert abs(hours["J1939-6342"] - 1191.75 / 3600.0) < 1e-12 and "4 fields" in source


@uses_tmp_path
def test_on_source_hours_report_why_they_are_missing(tmp_path):
    assert wl.read_on_source_hours(tmp_path) == ({}, "no obsinfo summary.json found")
    _summary(tmp_path, {"ANT": {"NAME": ["a", "b"]}})
    hours, why = wl.read_on_source_hours(tmp_path)
    assert hours == {} and "no FIELD names or SCAN table" in why
    _summary(tmp_path, {"FIELD": {"NAME": ["a"]}, "SCAN": {"7": {"1": 5.0}, "x": {}}})
    assert wl.read_on_source_hours(tmp_path)[1].endswith("no scan durations")


@uses_tmp_path
def test_on_source_hours_add_up_fields_with_the_same_name(tmp_path):
    _summary(
        tmp_path,
        {"FIELD": {"NAME": ["p63", "p63"]}, "SCAN": {"0": {"1": 1800.0}, "1": {"2": 900.0}}},
    )
    assert abs(wl.read_on_source_hours(tmp_path)[0]["p63"] - 0.75) < 1e-12
