"""tables: the baseline and large-scale rows of the checklist and the band summary columns."""

import csv

import numpy as np
from helpers import uses_tmp_path

from hi_cube_qa import tables

THRESHOLDS = {"baseline_warn_sigma": 5.0, "baseline_bad_sigma": 10.0}
NS = {"rms_to_theory": 1.0, "pct_bad": 0.0, "overall_quality": "Good"}


def _checks(baseline=None, large=None):
    rows = tables.build_expert_checklist(
        {"c": NS},
        {"c": {"available": False}},
        {"c": np.ones(20)},
        THRESHOLDS,
        baseline_by_label={"c": baseline} if baseline is not None else None,
        large_by_label={"c": large} if large is not None else None,
    )
    return {r["criterion"]: r for r in rows}


def _baseline(sigma, clipped=0):
    return {"available": True, "baseline_sigma": sigma, "n_clipped": clipped}


def _large(status, ratio=1.0, frac=0.0, note=""):
    return {
        "available": True,
        "status": status,
        "median_ratio": ratio,
        "frac_above": frac,
        "note": note,
    }


def test_baseline_row_limits():
    def status(sigma):
        return _checks(_baseline(sigma))["Flat baseline"]["status"]

    assert [status(v) for v in (0.0, 4.99, 5.0, 9.99, 10.0, 80.0)] == [
        "Pass", "Pass", "Warn", "Warn", "Fail", "Fail",
    ]  # fmt: skip
    row = _checks(_baseline(7.26, clipped=3))["Flat baseline"]
    assert row["value"] == "7.3 standard errors"
    assert row["note"] == "3 channel(s) clipped"


def test_baseline_row_without_a_fit():
    for baseline in (None, {"available": False}, _baseline(float("nan"))):
        row = _checks(baseline)["Flat baseline"]
        assert row["status"] == "N/A" and "No baseline fit" in row["note"]


def test_large_scale_row_follows_the_grade_of_the_assessment():
    for status in ("Pass", "Warn", "Fail"):
        row = _checks(large=_large(status, 2.0, 0.5, "PB >= 0.5"))["Low large-scale noise"]
        assert row["status"] == status and row["note"] == "PB >= 0.5"
        assert row["value"] == "2.00x thermal (50% of channels above)"
    assert _checks(large=_large("No data"))["Low large-scale noise"]["status"] == "N/A"


def test_large_scale_row_when_not_assessed():
    row = _checks()["Low large-scale noise"]
    assert row["status"] == "N/A" and row["note"] == "Not assessed for this product"
    failed = _checks(large={"available": False, "note": "image too small"})["Low large-scale noise"]
    assert failed["status"] == "N/A" and failed["note"] == "image too small"


@uses_tmp_path
def test_band_summary_columns(tmp_path):
    ns = dict(
        n_total=20, median_rms_mjy=1.0, min_rms_mjy=0.9, max_rms_mjy=1.1, std_rms_mjy=0.05,
        theory_rms_mjy=0.4, rms_to_theory=2.5, n_good=20, n_warn=0, n_bad=0, pct_bad=0.0,
        overall_quality="Good",
    )  # fmt: skip
    large = dict(_large("Fail", 3.5, 1.0), available=True, median_excess=0.0936, median_corr=0.923)
    path = tmp_path / "band.csv"
    tables.write_band_summary_csv(
        path, {"a": ns, "b": ns}, {"a": {}, "b": {}}, None,
        baseline_by_label={"a": _baseline(8.3, 2), "b": {}},
        large_by_label={"a": large, "b": {}},
    )  # fmt: skip
    rows = {r["cube_label"]: r for r in csv.DictReader(open(path))}
    a, b = rows["a"], rows["b"]
    assert a["baseline_sigma"] == "8.30" and a["baseline_n_clipped"] == "2"
    assert (
        a["ls_status"] == "Fail"
        and a["ls_median_ratio"] == "3.500"
        and a["ls_frac_above"] == "1.000"
    )
    assert a["ls_median_excess_mjy"] == "0.0936" and a["ls_median_corr"] == "0.923"
    assert all(b[k] == "" for k in b if k.startswith(("baseline_", "ls_")))


def test_baseline_limits_default_to_5_and_10():
    rows = tables.build_expert_checklist(
        {"c": NS}, {"c": {"available": False}}, {"c": np.ones(20)}, {},
        baseline_by_label={"c": _baseline(5.5)},
    )  # fmt: skip
    assert {r["criterion"]: r["status"] for r in rows}["Flat baseline"] == "Warn"
