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


def test_checklist_notes_the_channels_left_out():
    ns = dict(NS, n_excluded=103)
    rows = tables.build_expert_checklist(
        {"c": ns}, {"c": {"available": False}}, {"c": np.ones(20)}, THRESHOLDS
    )
    notes = {r["criterion"]: r["note"] for r in rows}
    assert notes["Few bad channels"] == "103 channel(s) without a beam left out"
    plain = tables.build_expert_checklist(
        {"c": NS}, {"c": {"available": False}}, {"c": np.ones(20)}, THRESHOLDS
    )
    assert {r["criterion"]: r["note"] for r in plain}["Few bad channels"] == ""


@uses_tmp_path
def test_band_summary_counts_the_channels_without_a_beam(tmp_path):
    ns = dict(
        n_total=20, median_rms_mjy=1.0, min_rms_mjy=0.9, max_rms_mjy=1.1, std_rms_mjy=0.05,
        theory_rms_mjy=0.4, rms_to_theory=2.5, n_good=20, n_warn=0, n_bad=0, pct_bad=0.0,
        overall_quality="Good", n_excluded=7,
    )  # fmt: skip
    path = tmp_path / "band.csv"
    tables.write_band_summary_csv(path, {"a": ns}, {"a": {}}, None)
    assert next(csv.DictReader(open(path)))["n_no_beam"] == "7"


def _status(criterion, ns=None, beam=None, rms=None, cont=None, kurt=None, **limits):
    rows = tables.build_expert_checklist(
        {"c": dict(NS, **(ns or {}))},
        {"c": beam or {"available": False}},
        {"c": rms if rms is not None else np.ones(20)},
        limits,
        cont_by_label={"c": cont} if cont else None,
        kurt_by_label={"c": kurt} if kurt else None,
    )
    return {r["criterion"]: r for r in rows}[criterion]


def test_noise_limits_come_from_the_thresholds():
    ns = {"rms_to_theory": 2.0}
    assert _status("Noise vs theory", ns)["status"] == "Warn"
    assert _status("Noise vs theory", ns, rms_warn_factor=2.5)["status"] == "Pass"
    assert _status("Noise vs theory", ns, rms_bad_factor=1.8)["status"] == "Fail"
    assert "< 2.5" in _status("Noise vs theory", ns, rms_warn_factor=2.5)["description"]


def test_bad_channel_limits_come_from_the_thresholds():
    ns = {"pct_bad": 8.0}
    assert _status("Few bad channels", ns)["status"] == "Warn"
    assert _status("Few bad channels", ns, bad_chan_warn_pct=10.0)["status"] == "Pass"
    assert _status("Few bad channels", ns, bad_chan_bad_pct=6.0)["status"] == "Fail"
    assert _status("No catastrophic channels", ns)["status"] == "Pass"
    assert _status("No catastrophic channels", ns, bad_chan_bad_pct=8.0)["status"] == "Fail"
    assert "< 6%" in _status("No catastrophic channels", ns, bad_chan_bad_pct=6.0)["description"]


def test_convergence_limits_and_tail_come_from_the_thresholds():
    rms = np.ones(20)
    rms[-2:] = 1.5  # the last 10% of the channels
    assert _status("Noise floor convergence", rms=rms)["status"] == "Warn"
    assert _status("Noise floor convergence", rms=rms, convergence_warn=2.0)["status"] == "Pass"
    assert _status("Noise floor convergence", rms=rms, convergence_bad=1.4)["status"] == "Fail"
    assert (
        _status("Noise floor convergence", rms=rms, convergence_tail=0.3)["status"] == "Pass"
    )  # the last 30% has the same median as the rest


def test_beam_limits_come_from_the_thresholds():
    beam = {"available": True, "std_bmaj_asec": 0.7, "axial_ratio": 2.5}
    assert _status("Beam stability", beam=beam)["status"] == "Warn"
    assert _status("Beam stability", beam=beam, beam_scatter_warn=1.0)["status"] == "Pass"
    assert _status("Reasonable axial ratio", beam=beam)["status"] == "Warn"
    assert _status("Reasonable axial ratio", beam=beam, axial_ratio_warn=3.0)["status"] == "Pass"
    assert _status("Reasonable axial ratio", beam=beam, axial_ratio_bad=2.2)["status"] == "Fail"


def test_outlier_limits_come_from_the_thresholds():
    rms = np.ones(20)
    rms[3] = 10.0
    assert _status("No severe RMS outliers", rms=rms)["status"] == "Warn"
    assert _status("No severe RMS outliers", rms=rms, outlier_warn_count=1)["status"] == "Fail"
    assert _status("No severe RMS outliers", rms=rms, outlier_factor=20.0)["status"] == "Pass"
    assert (
        _status("No severe RMS outliers", rms=rms, outlier_factor=20.0)["value"]
        == "0 channels > 20x median"
    )


def test_continuum_and_kurtosis_limits_come_from_the_thresholds():
    cont = {"available": True, "pct_pixels_persistent": 1.0}
    assert _status("Low continuum residual", cont=cont)["status"] == "Warn"
    assert _status("Low continuum residual", cont=cont, continuum_warn_pct=2.0)["status"] == "Pass"
    assert _status("Low continuum residual", cont=cont, continuum_bad_pct=0.8)["status"] == "Fail"
    kurt = {"available": True, "pct_bad": 10.0}
    assert _status("Gaussian noise (kurtosis)", kurt=kurt)["status"] == "Warn"
    assert (
        _status("Gaussian noise (kurtosis)", kurt=kurt, kurtosis_chan_warn_pct=12.0)["status"]
        == "Pass"
    )
    assert (
        _status("Gaussian noise (kurtosis)", kurt=kurt, kurtosis_chan_bad_pct=8.0)["status"]
        == "Fail"
    )


def test_every_limit_used_by_the_checklist_has_a_default_in_the_config():
    from hi_cube_qa.config import DEFAULTS

    rows = tables.build_expert_checklist(
        {"c": NS}, {"c": {"available": False}}, {"c": np.ones(20)}, {}
    )
    assert len(rows) == len(tables.CHECKLIST_CRITERIA)
    for key in (
        "convergence_warn",
        "convergence_bad",
        "convergence_tail",
        "axial_ratio_warn",
        "axial_ratio_bad",
        "outlier_factor",
        "outlier_warn_count",
        "continuum_warn_pct",
        "continuum_bad_pct",
        "kurtosis_chan_warn_pct",
        "kurtosis_chan_bad_pct",
        "warn_chan_warn_pct",
    ):
        assert key in DEFAULTS["thresholds"], key
