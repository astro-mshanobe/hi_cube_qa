"""End to end: a synthetic CARACal session with known noise goes through the command line entry point."""

import contextlib
import csv
import io
import json
import sys

from astropy.io import fits
from helpers import uses_tmp_path
from synthetic import DF_HZ, make_session

from hi_cube_qa import cli as qa_cli
from hi_cube_qa import noise as nz

SIGMA_JY = 1e-3
N_ANT = 8
T_TOTAL_HR = 8.0


def _noisy_channel(car, channel, factor):
    """Multiply one channel of every cube by *factor*, as a stand-in for a badly affected channel."""
    for path in car.glob("output_hi_*/cubes/cube_*/*.fits"):
        with fits.open(path, mode="update") as hdul:
            hdul[0].data[channel] *= factor


def _run(tmp_path, *extra_args, bad_channel=None):
    car = make_session(tmp_path / "data", n_ant=N_ANT, nchan=16, ny=48, nx=48, sigma=SIGMA_JY)
    if bad_channel is not None:
        _noisy_channel(car, *bad_channel)
    cfg = tmp_path / "qa.yaml"
    cfg.write_text(
        f"sessions:\n  - root: {car}\noutput:\n  dir: {tmp_path / 'out'}\nradiometer:\n  t_total: {T_TOTAL_HR}\n"
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
