"""io_utils: spectral-axis handling, channel width, flag-report parsing and cube loading."""

import json
import warnings

import numpy as np
from astropy.io import fits
from helpers import uses_tmp_path
from synthetic import DF_HZ, FREQ0_HZ, cube_header

from hi_cube_qa import io_utils as io

REST = io.HI_REST_FREQ_HZ


def _hdr(ctype, crval, cdelt, cunit=None, restfrq=None, n=5):
    h = fits.Header()
    h["NAXIS"] = 3
    h["NAXIS3"] = n
    h["CTYPE3"], h["CRVAL3"], h["CDELT3"], h["CRPIX3"] = ctype, crval, cdelt, 1
    if cunit:
        h["CUNIT3"] = cunit
    if restfrq:
        h["RESTFRQ"] = restfrq
    return h


def test_frequency_axis_gives_the_grid_and_the_radio_velocity():
    f, v = io.frequency_velocity(_hdr("FREQ", REST, -26367.19), 5)
    assert abs(f[0] - REST) < 1e-3 and abs(v[0]) < 1e-9
    assert np.all(np.diff(v) > 0) and abs(v[1] - io.C_KMS * 26367.19 / REST) < 1e-9


def test_radio_velocity_axis_in_metres_per_second_or_km_per_second():
    f, v = io.frequency_velocity(_hdr("VRAD", 1.0e6, 5.0e3), 5)
    assert abs(v[0] - 1000.0) < 1e-9 and abs(f[0] - REST * (1 - 1000.0 / io.C_KMS)) < 1e-3
    f2, v2 = io.frequency_velocity(_hdr("VELO-LSR", 1000.0, 5.0, cunit="km/s"), 5)
    assert np.allclose(v2, v) and np.allclose(f2, f)


def test_optical_velocity_axis_is_converted_to_radio_velocity():
    f, v = io.frequency_velocity(
        _hdr(
            "VOPT",
            3.0e7,
            0.0,
        ),
        5,
    )
    z = 3.0e4 / io.C_KMS
    assert abs(f[0] - REST / (1 + z)) < 1e-3
    assert abs(v[0] - io.C_KMS * z / (1 + z)) < 1e-6


def test_rest_frequency_is_read_from_the_header_when_present():
    f, v = io.frequency_velocity(_hdr("FREQ", 1.4e9, -1e4, restfrq=1.4e9), 5)
    assert abs(v[0]) < 1e-9


def test_a_missing_or_mismatched_spectral_axis_falls_back_to_a_dummy_grid_with_a_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f, v = io.frequency_velocity(fits.Header(), 4)
    assert (
        len(f) == 4
        and f[0] == 1.30e9
        and f[-1] == 1.43e9
        and any("dummy" in str(x.message) for x in w)
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f, _ = io.frequency_velocity(_hdr("FREQ", REST, -1e4, n=5), 7, quiet=True)
    assert len(f) == 7 and not w


def test_channel_width_from_a_frequency_axis_a_velocity_axis_and_the_fallback():
    assert io.channel_width_hz(_hdr("FREQ", REST, -26367.19)) == 26367.19
    df = io.channel_width_hz(_hdr("VRAD", 0.0, 5.0e3))
    assert abs(df - REST * 5.0 / io.C_KMS) < 1e-6
    assert abs(io.channel_width_hz(_hdr("VRAD", 0.0, 5.0, cunit="km/s")) - df) < 1e-6
    assert io.channel_width_hz(fits.Header()) == 26367.19
    assert io.channel_width_hz(_hdr("FREQ", REST, 0.0)) == 26367.19


def test_pixel_scale_is_the_absolute_cdelt1_in_arcsec_with_a_default():
    assert abs(io.pixel_scale_asec(fits.Header({"CDELT1": -2.0 / 3600})) - 2.0) < 1e-9
    assert abs(io.pixel_scale_asec(fits.Header()) - 6.0) < 1e-9


@uses_tmp_path
def test_flag_report_labels_follow_the_stage_name_in_the_file_name(tmp_path):
    a = tmp_path / "1608285672_sdp_l0-cal-flag__bpcal-flagging-summary-0.json"
    b = tmp_path / "plain-flagging-summary-1.json"
    a.write_text(json.dumps({"x": 1}))
    b.write_text(json.dumps({"y": 2}))
    out = io.load_flag_report([a, b])
    assert set(out) == {"cal-flag / bpcal", "plain"} and out["plain"] == {"y": 2}


@uses_tmp_path
def test_flag_report_skips_missing_files_with_a_warning_and_returns_none_when_empty(tmp_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert io.load_flag_report(tmp_path / "absent.json") is None
    assert w and io.load_flag_report(None) is None


def test_flag_fraction_from_a_caracal_report_sums_counts_over_the_correlations():
    rep = {
        "Flag stats": [
            {"scans": {}},
            {"corrs": {"0": {"sum": 10.0, "counts": 100.0}, "1": {"sum": 30.0, "counts": 100.0}}},
        ]
    }
    assert abs(io.flag_fraction_of(rep) - 0.2) < 1e-12


def test_flag_fraction_averages_plain_fractions_when_no_counts_are_given():
    rep = {"Flag stats": [{"corrs": {"0": {"frac": 0.1}, "1": {"frac": 0.3}}}]}
    assert abs(io.flag_fraction_of(rep) - 0.2) < 1e-12


def test_flag_fraction_prefers_corrs_over_other_categories():
    rep = {
        "Flag stats": [
            {"antennas": {"a": {"sum": 90.0, "counts": 100.0}}},
            {"corrs": {"0": {"sum": 10.0, "counts": 100.0}}},
        ]
    }
    assert abs(io.flag_fraction_of(rep) - 0.1) < 1e-12


def test_flag_fraction_generic_search_accepts_fractions_and_percentages_and_rejects_junk():
    assert abs(io.flag_fraction_of({"stage": {"flagged": 0.25}}) - 0.25) < 1e-12
    assert abs(io.flag_fraction_of({"stage": {"frac": 25.0}}) - 0.25) < 1e-12
    assert io.flag_fraction_of({"nothing": [1, 2]}) is None and io.flag_fraction_of([1]) is None


@uses_tmp_path
def test_load_cube_squeezes_a_stokes_axis_and_returns_matching_spectral_arrays(tmp_path):
    p = tmp_path / "c.fits"
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    h = cube_header(3, 4, 5)
    h["CTYPE4"], h["CRVAL4"], h["CDELT4"], h["CRPIX4"] = "STOKES", 1, 1, 1
    fits.PrimaryHDU(data=data, header=h).writeto(p)
    d, hdr, wcs, f, v = io.load_cube(p)
    assert d.shape == (3, 4, 5) and len(f) == 3 and len(v) == 3 and abs(f[1] - f[0] - DF_HZ) < 1e-3
    assert abs(f[0] - FREQ0_HZ) < 1e-3


@uses_tmp_path
def test_load_cube_promotes_an_image_to_one_channel_and_rejects_other_shapes(tmp_path):
    p = tmp_path / "img.fits"
    fits.PrimaryHDU(data=np.zeros((4, 5), dtype=np.float32)).writeto(p)
    assert io.load_cube(p)[0].shape == (1, 4, 5)
    q = tmp_path / "bad.fits"
    fits.PrimaryHDU(data=np.zeros((2, 2, 2, 2, 2), dtype=np.float32)).writeto(q)
    try:
        io.load_cube(q)
    except ValueError:
        return
    raise AssertionError("a five-dimensional array should be refused")
