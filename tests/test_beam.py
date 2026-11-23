"""beam: per-channel and global beam keywords, and the beam summary."""

import warnings

import numpy as np
from astropy.io import fits

from hi_cube_qa import beam as bm


def _h(**kw):
    h = fits.Header()
    for k, v in kw.items():
        h[k] = v
    return h


def test_global_beam_in_arcsec():
    a, b, p = bm.beam_stats(_h(BMAJ=20 / 3600, BMIN=15 / 3600, BPA=30.0), 4)
    assert (
        np.allclose(a, 20.0) and np.allclose(b, 15.0) and np.allclose(p, 30.0) and a.shape == (4,)
    )


def test_per_channel_keywords_override_global():
    h = _h(
        BMAJ=99 / 3600,
        BMIN=99 / 3600,
        BMAJ0000=20 / 3600,
        BMIN0000=15 / 3600,
        BPA0000=1.0,
        BMAJ0001=21 / 3600,
        BMIN0001=16 / 3600,
        BPA0001=2.0,
    )
    a, b, p = bm.beam_stats(h, 3)
    assert (
        np.allclose(a[:2], [20.0, 21.0])
        and np.allclose(b[:2], [15.0, 16.0])
        and list(p[:2]) == [1.0, 2.0]
    )
    assert abs(a[2] - 99.0) < 1e-9  # channel 2 has no keyword: global value


def test_one_indexed_and_truncated_keywords():
    h = _h(
        BMAJ1=20 / 3600,
        BMIN1=15 / 3600,
        BPA1=5.0,
        BMAJ2=22 / 3600,
        BMIN2=17 / 3600,
        BPA2=6.0,
        BMAJ=1 / 3600,
        BMIN=1 / 3600,
    )
    a, b, p = bm.beam_stats(h, 2)
    assert np.allclose(a, [20.0, 22.0]) and np.allclose(b, [15.0, 17.0]) and list(p) == [5.0, 6.0]
    long_h = _h(BMA1=20 / 3600, BMI1=15 / 3600)
    a, b, _ = bm.beam_stats(long_h, 1)
    assert abs(a[0] - 20.0) < 1e-9 and abs(b[0] - 15.0) < 1e-9


def test_no_beam_in_header():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert bm.beam_stats(fits.Header(), 3) == (None, None, None)
    assert w and "No beam info" in str(w[0].message)


def test_beam_area_and_axial_ratio():
    bmaj = np.array([20.0, 20.0, 22.0, np.nan])
    bmin = np.array([10.0, 10.0, 11.0, np.nan])
    bpa = np.array([0.0, 10.0, 20.0, np.nan])
    d = bm.beam_diagnostics(bmaj, bmin, bpa)
    assert (
        d["available"]
        and d["median_bmaj_asec"] == 20.0
        and d["median_bmin_asec"] == 10.0
        and d["median_bpa_deg"] == 10.0
    )
    assert abs(d["beam_area_asec2"] - np.pi / (4 * np.log(2)) * 200.0) < 1e-9
    assert d["axial_ratio"] == 2.0 and d["max_bmaj_asec"] == 22.0 and d["min_bmin_asec"] == 10.0


def test_bmaj_scatter_flag():
    steady = np.full(10, 20.0)
    wobbly = 20.0 + np.tile([-1.0, 1.0], 5)
    assert bm.beam_diagnostics(steady, steady, steady)["bmaj_scatter_quality"] == "Good"
    assert bm.beam_diagnostics(wobbly, wobbly, wobbly)["bmaj_scatter_quality"] == "Warn"
    assert (
        bm.beam_diagnostics(wobbly, wobbly, wobbly, scatter_warn_asec=2.0)["bmaj_scatter_quality"]
        == "Good"
    )


def test_diagnostics_without_beam():
    assert bm.beam_diagnostics(None, None, None) == {"available": False}


def test_valid_beam():
    bmaj = np.array([20.0, 0.0, np.nan, -1.0, 20.0])
    bmin = np.array([15.0, 15.0, 15.0, 15.0, 0.0])
    assert list(bm.valid_beam(bmaj)) == [True, False, False, False, True]
    assert list(bm.valid_beam(bmaj, bmin)) == [True, False, False, False, False]


def test_diagnostics_leave_out_channels_without_a_beam():
    bmaj = np.array([20.0, 20.1, 0.0, 0.0, 19.9, 20.0])
    bmin = np.array([15.0, 15.1, 0.0, 0.0, 14.9, 15.0])
    bpa = np.array([10.0, 10.0, 0.0, 0.0, 10.0, 10.0])
    d = bm.beam_diagnostics(bmaj, bmin, bpa)
    assert d["n_no_beam"] == 2 and d["min_bmaj_asec"] == 19.9 and d["min_bmin_asec"] == 14.9
    assert abs(d["std_bmaj_asec"] - np.std([20.0, 20.1, 19.9, 20.0])) < 1e-12
    assert d["bmaj_scatter_quality"] == "Good"  # the zeros alone would give a std of 8.5
    assert abs(d["median_bpa_deg"] - 10.0) < 1e-12 and abs(d["axial_ratio"] - 20.0 / 15.0) < 1e-3


def test_diagnostics_without_any_beam_are_unavailable():
    zeros = np.zeros(5)
    assert bm.beam_diagnostics(zeros, zeros, zeros) == {"available": False}
    assert bm.beam_diagnostics(np.full(5, np.nan), None, None) == {"available": False}


def test_pixels_per_beam_ignore_channels_without_a_beam():
    from hi_cube_qa.reduction import pix_per_beam

    hdr = _h(CDELT1=-6.0 / 3600, CDELT2=6.0 / 3600)
    with_zeros = pix_per_beam(
        hdr, np.array([0.0, 0.0, 0.0, 20.0, 20.0]), np.array([0.0, 0.0, 0.0, 15.0, 15.0])
    )
    clean = pix_per_beam(hdr, np.array([20.0, 20.0]), np.array([15.0, 15.0]))
    assert with_zeros == clean and abs(clean - np.pi / (4 * np.log(2)) * 300 / 36) < 1e-9
    assert pix_per_beam(hdr, np.zeros(3), np.zeros(3)) == pix_per_beam(hdr, None, None)
