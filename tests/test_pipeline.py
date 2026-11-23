"""End to end: a synthetic CARACal session with known noise goes through the command line entry point."""

import contextlib
import csv
import io
import json
import sys

import numpy as np
from astropy.io import fits
from helpers import uses_tmp_path
from synthetic import DF_HZ, make_session

from hi_cube_qa import cli as qa_cli
from hi_cube_qa import noise as nz
from hi_cube_qa.baseline_flatness import baseline_check
from hi_cube_qa.reduction import CubeHandle

SIGMA_JY = 1e-3
N_ANT = 8
T_TOTAL_HR = 8.0


def _noisy_channel(car, channel, factor):
    """Multiply one channel of every cube by *factor*, as a stand-in for a badly affected channel."""
    for path in car.glob("output_hi_*/cubes/cube_*/*.fits"):
        with fits.open(path, mode="update") as hdul:
            hdul[0].data[channel] *= factor


def _run(tmp_path, *extra_args, bad_channel=None, shape=(16, 48, 48), tamper=None, extra_config=""):
    nchan, ny, nx = shape
    car = make_session(tmp_path / "data", n_ant=N_ANT, nchan=nchan, ny=ny, nx=nx, sigma=SIGMA_JY)
    if bad_channel is not None:
        _noisy_channel(car, *bad_channel)
    if tamper is not None:
        tamper(car)
    cfg = tmp_path / "qa.yaml"
    cfg.write_text(
        f"sessions:\n  - root: {car}\noutput:\n  dir: {tmp_path / 'out'}\n"
        f"radiometer:\n  t_total: {T_TOTAL_HR}\n{extra_config}"
    )
    argv, sys.argv = sys.argv, ["hi-cube-qa", "--config", str(cfg), *extra_args]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                qa_cli.main()
            except SystemExit as err:
                assert err.code in (None, 0)
    finally:
        sys.argv = argv
    return tmp_path / "out", out.getvalue()


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@uses_tmp_path
def test_session_outputs(tmp_path):
    out, _ = _run(tmp_path)
    obsid = "1600000000_sdp_l0"
    names = {p.name for p in out.iterdir()}
    assert {
        f"{obsid}_qa_report.html",
        f"{obsid}_channel_qa.csv",
        f"{obsid}_band_summary.csv",
        f"{obsid}_checklist.csv",
        "qa_index.html",
        "qa_survey_summary.csv",
        "qa_survey_summary.json",
        "hi_cube_qa.log",
    } <= names
    assert f"{obsid}_qa_report.html" in (out / "qa_index.html").read_text()
    assert (
        len(json.loads((out / "qa_survey_summary.json").read_text())) == 6
    )  # 2 pointings x 3 products


@uses_tmp_path
def test_measured_noise_and_theory(
    tmp_path,
):
    out, _ = _run(tmp_path)
    rows = _rows(out / "1600000000_sdp_l0_band_summary.csv")
    assert len(rows) == 6
    for row in rows:
        assert abs(float(row["median_rms_mjy_beam"]) - SIGMA_JY * 1e3) < 0.05 * SIGMA_JY * 1e3
        assert row["overall_quality"] == "Good" and row["n_channels"] == "16"
    expected_mjy = (
        nz.theoretical_noise(tsys=25.0, eta=0.80, n_ant=N_ANT, t_total_hr=T_TOTAL_HR, df_hz=DF_HZ)
        * 1e3
    )
    assert abs(float(rows[0]["theory_rms_mjy_beam"]) - expected_mjy) < 1e-3 * expected_mjy


@uses_tmp_path
def test_noisy_channel_is_flagged(tmp_path):
    out, _ = _run(tmp_path, bad_channel=(5, 10.0))  # 10 mJy against a 2.9 mJy theory: ratio 3.4 > 3
    rows = _rows(out / "1600000000_sdp_l0_channel_qa.csv")
    assert {int(r["channel"]) for r in rows if r["quality"] == "Bad"} == {5}
    assert {r["quality"] for r in rows if int(r["channel"]) != 5} == {"Good"}
    assert {r["overall_quality"] for r in _rows(out / "1600000000_sdp_l0_band_summary.csv")} == {
        "Warn"
    }  # 1 of 16 = 6.25 % >= 5 %


@uses_tmp_path
def test_list_mode(tmp_path):
    out, text = _run(tmp_path, "--list")
    assert (
        "Session 1600000000_sdp_l0: 2 unit(s) across 2 pointing(s)" in text and "p01/cube_1" in text
    )
    assert not out.exists()


def _slope_and_ripple(car):
    """A baseline slope in the p01 image cube and a stripe pattern in the p02 residual cube."""
    nchan = 60
    stripes = 3e-4 * np.sin(2 * np.pi * np.arange(256) / 50.0)[None, :]
    slope = np.linspace(-1, 1, nchan) * 1e-4
    for path in car.glob("output_hi_p02/cubes/cube_*/*.residual.fits"):
        with fits.open(path, mode="update") as hdul:
            hdul[0].data += stripes.astype(np.float32)
    for path in car.glob("output_hi_p01/cubes/cube_*/*.image.fits"):
        with fits.open(path, mode="update") as hdul:
            hdul[0].data += slope[:, None, None].astype(np.float32)


@uses_tmp_path
def test_baseline_and_large_scale_checks(tmp_path):
    out, _ = _run(
        tmp_path,
        shape=(60, 256, 256),
        tamper=_slope_and_ripple,
        extra_config="large_scale:\n  sigma_arcsec: 60.0\n  stride: 10\n",
    )
    obsid = "1600000000_sdp_l0"
    checks = {(r["cube_label"], r["criterion"]): r for r in _rows(out / f"{obsid}_checklist.csv")}

    def status(label, criterion):
        return checks[(f"{label}", criterion)]["status"]

    assert status("p01/cube_1/c1-image", "Flat baseline") == "Fail"
    assert status("p01/cube_1/c1-residual", "Flat baseline") == "Pass"
    assert status("p02/cube_1/c1-residual", "Flat baseline") == "Pass"
    assert status("p02/cube_1/c1-residual", "Low large-scale noise") == "Fail"
    assert status("p01/cube_1/c1-residual", "Low large-scale noise") == "Pass"
    skipped = checks[("p01/cube_1/c1-image", "Low large-scale noise")]
    assert skipped["status"] == "N/A" and "Not assessed" in skipped["note"]

    bands = {r["cube_label"]: r for r in _rows(out / f"{obsid}_band_summary.csv")}
    ripple, clean = bands["p02/cube_1/c1-residual"], bands["p01/cube_1/c1-residual"]
    assert ripple["ls_status"] == "Fail" and float(ripple["ls_median_ratio"]) > 3
    assert clean["ls_status"] == "Pass" and 0.8 < float(clean["ls_median_ratio"]) < 1.3
    assert float(ripple["ls_median_corr"]) > 0.9
    assert bands["p01/cube_1/c1-dirty"]["ls_status"] == ""
    assert float(bands["p01/cube_1/c1-image"]["baseline_sigma"]) > 10

    html = (out / f"{obsid}_qa_report.html").read_text()
    assert "fig_u-p02-cube-1_large_scale" in html and "fig_u-p01-cube-1_baseline" in html


@uses_tmp_path
def test_pipeline_baseline_matches_the_standalone_check(tmp_path):
    out, _ = _run(tmp_path, shape=(60, 256, 256), tamper=_slope_and_ripple)
    bands = {r["cube_label"]: r for r in _rows(out / "1600000000_sdp_l0_band_summary.csv")}
    path = next((tmp_path / "data").glob("**/output_hi_p01/cubes/cube_1/*.image.fits"))
    with CubeHandle(path) as handle:
        alone = baseline_check(handle)
    assert (
        abs(float(bands["p01/cube_1/c1-image"]["baseline_sigma"]) - alone["baseline_sigma"]) < 0.01
    )


def _offset_channel(car):
    """A large offset in one channel of the p01 image cube: an outlier in the per-channel mean."""
    for path in car.glob("output_hi_p01/cubes/cube_*/*.image.fits"):
        with fits.open(path, mode="update") as hdul:
            hdul[0].data[7] += 0.05


def _band(out):
    return {r["cube_label"]: r for r in _rows(out / "1600000000_sdp_l0_band_summary.csv")}


@uses_tmp_path
def test_both_checks_can_be_switched_off(tmp_path):
    out, _ = _run(
        tmp_path,
        shape=(60, 256, 256),
        extra_config="large_scale:\n  enable: false\n  sigma_arcsec: 60.0\n  stride: 10\nbaseline:\n  enable: false\n",
    )
    for row in _band(out).values():
        assert row["baseline_sigma"] == "" and row["ls_status"] == ""
    checks = _rows(out / "1600000000_sdp_l0_checklist.csv")
    criteria = ("Flat baseline", "Low large-scale noise")
    assert {r["status"] for r in checks if r["criterion"] in criteria} == {"N/A"}


@uses_tmp_path
def test_baseline_sigma_clip_setting(tmp_path):
    out, _ = _run(tmp_path, tamper=_offset_channel)
    assert int(_band(out)["p01/cube_1/c1-image"]["baseline_n_clipped"]) >= 1
    out2, _ = _run(
        tmp_path / "unclipped", tamper=_offset_channel, extra_config="baseline:\n  sigma_clip: 0\n"
    )
    assert int(_band(out2)["p01/cube_1/c1-image"]["baseline_n_clipped"]) == 0


@uses_tmp_path
def test_large_scale_limits_come_from_the_thresholds(tmp_path):
    cfg = "large_scale:\n  sigma_arcsec: 60.0\n  stride: 10\n"
    out, _ = _run(
        tmp_path, shape=(60, 256, 256), extra_config=cfg + "thresholds:\n  ls_warn_ratio: 0.5\n"
    )
    assert _band(out)["p01/cube_1/c1-residual"]["ls_status"] == "Warn"
    out, _ = _run(
        tmp_path / "fail",
        shape=(60, 256, 256),
        extra_config=cfg + "thresholds:\n  ls_fail_ratio: 0.5\n",
    )
    assert _band(out)["p01/cube_1/c1-residual"]["ls_status"] == "Fail"


def test_large_scale_product_selection():
    wanted = qa_cli._large_scale_wanted
    assert wanted("all", "dirty", False) and wanted("reference", "residual", True)
    assert not wanted("reference", "dirty", False)
    assert wanted(["residual", "image"], "image", False) and not wanted(["residual"], "dirty", True)


def _add_line_cubes(car, nchan=None):
    """An imsub-line.fits next to the cubes of each pointing, copied from its residual cube."""
    for path in car.glob("output_hi_p*/cubes/cube_*/*.residual.fits"):
        with fits.open(path) as hdul:
            data = hdul[0].data if nchan is None else hdul[0].data[:nchan]
            fits.PrimaryHDU(data=data, header=hdul[0].header).writeto(
                path.parent / "imsub-line.fits"
            )


def _stripes_and_line(car):
    _slope_and_ripple(car)
    _add_line_cubes(car)


@uses_tmp_path
def test_line_cube_is_the_reference_for_the_large_scale_check(tmp_path):
    out, _ = _run(
        tmp_path,
        shape=(60, 256, 256),
        tamper=_stripes_and_line,
        extra_config="large_scale:\n  sigma_arcsec: 60.0\n  stride: 10\n",
    )
    bands = _band(out)
    assert bands["p02/cube_1/c1-line"]["ls_status"] == "Fail"
    assert bands["p01/cube_1/c1-line"]["ls_status"] == "Pass"
    assert bands["p02/cube_1/c1-residual"]["ls_status"] == ""
    assert bands["p02/cube_1/c1-line"]["baseline_sigma"] != ""


@uses_tmp_path
def test_a_line_cube_on_another_channel_grid_is_left_out(tmp_path):
    out, _ = _run(tmp_path, tamper=lambda car: _add_line_cubes(car, nchan=10))
    assert not [label for label in _band(out) if label.endswith("c1-line")]
    assert "c1-residual" in " ".join(_band(out))


@uses_tmp_path
def test_initially_visible_products(tmp_path):
    def legendonly(config, sub):
        (tmp_path / sub).mkdir()
        out, _ = _run(tmp_path / sub, tamper=_add_line_cubes, extra_config=config)
        return (
            (out / "1600000000_sdp_l0_qa_report.html").read_text().count('"visible":"legendonly"')
        )

    assert legendonly("", "all") == 0
    assert legendonly("plots:\n  initially_visible: [residual, line]\n", "some") > 0
    assert (
        legendonly("plots:\n  initially_visible: [psf]\n", "none") == 0
    )  # never hide every product


@uses_tmp_path
def test_initially_visible_hides_the_other_products(tmp_path):
    import logging

    from hi_cube_qa.config import load_config
    from hi_cube_qa.discovery import discover_session

    car = make_session(tmp_path / "data", n_ant=N_ANT, nchan=16, ny=48, nx=48, sigma=SIGMA_JY)
    _add_line_cubes(car)
    config = tmp_path / "qa.yaml"
    config.write_text(
        f"sessions:\n  - root: {car}\noutput:\n  dir: {tmp_path / 'out'}\n"
        "plots:\n  initially_visible: [residual, line]\n"
    )
    cfg = load_config(config)
    unit = discover_session(car, products=cfg.products).units[0]
    bundle = qa_cli.assess_unit(unit, cfg, logging.getLogger("test"))
    for figure in ("rms_curve", "rms_vs_theory", "kurtosis"):
        hidden = {
            t.name.split(" ")[0]
            for t in bundle["figures"][figure].data
            if t.visible == "legendonly"
        }
        shown = {t.name.split(" ")[0] for t in bundle["figures"][figure].data if t.visible is None}
        assert {"c1-dirty", "c1-image"} <= hidden and not hidden & {"c1-residual", "c1-line"}, (
            figure
        )
        assert {"c1-residual", "c1-line"} <= shown, figure
