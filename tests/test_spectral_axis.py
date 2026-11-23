"""
Validate io_utils.frequency_velocity against closed-form Doppler relations
(written out here independently of the implementation) and astropy.units.

  radio   : v = c (f0 - f) / f0
  optical : v = c (f0 - f) / f
"""

import warnings

import numpy as np
from astropy import units as u
from astropy.io import fits

from hi_cube_qa.io_utils import frequency_velocity

C = 299792.458  # km/s
F0 = 1.420405752e9  # Hz


def _header(ctype, crval, cdelt, n, cunit, crpix=1.0, axis=3, naxis=3, restfrq=F0):
    h = fits.Header()
    h["NAXIS"] = naxis
    for ax in range(1, naxis + 1):
        h[f"NAXIS{ax}"] = n if ax == axis else 10
        h[f"CTYPE{ax}"] = (
            ctype if ax == axis else ("RA---SIN", "DEC--SIN", "STOKES", "STOKES")[ax - 1]
        )
    h[f"CRVAL{axis}"] = crval
    h[f"CDELT{axis}"] = cdelt
    h[f"CRPIX{axis}"] = crpix
    h[f"CUNIT{axis}"] = cunit
    if restfrq is not None:
        h["RESTFRQ"] = restfrq
    return h


def _truth_from_freq(f):
    return C * (F0 - f) / F0


def test_frequency_axis_gives_radio_velocity():
    n = 300
    f_true = 1.40e9 + 26123.046875 * np.arange(n)
    h = _header("FREQ", f_true[0], 26123.046875, n, "Hz")
    f, v = frequency_velocity(h, n)
    assert np.allclose(f, f_true, rtol=0, atol=1e-3)
    assert np.allclose(v, _truth_from_freq(f_true), rtol=0, atol=1e-9)
    ref = (f_true * u.Hz).to(u.km / u.s, u.doppler_radio(F0 * u.Hz)).value
    assert np.allclose(v, ref, rtol=0, atol=1e-9)


def test_frequency_axis_with_crpix_offset_and_extra_axes():
    n = 200
    h = _header("FREQ", 1.41e9, -26123.046875, n, "Hz", crpix=50.0, axis=3, naxis=4)
    f, _ = frequency_velocity(h, n)
    expected = 1.41e9 + (np.arange(1, n + 1) - 50.0) * -26123.046875
    assert np.allclose(f, expected, rtol=0, atol=1e-3)


def test_radio_velocity_axis_in_m_per_s_and_km_per_s():
    n = 150
    v_true = np.linspace(-1000.0, 3000.0, n)  # km/s
    for unit, scale in (("m/s", 1e3), ("km/s", 1.0)):
        h = _header("VRAD", v_true[0] * scale, (v_true[1] - v_true[0]) * scale, n, unit)
        f, v = frequency_velocity(h, n)
        assert np.allclose(v, v_true, rtol=0, atol=1e-9)
        assert np.allclose(f, F0 * (1.0 - v_true / C), rtol=1e-12)
        ref = (v_true * u.km / u.s).to(u.Hz, u.doppler_radio(F0 * u.Hz)).value
        assert np.allclose(f, ref, rtol=1e-12)


def test_bare_velo_axis_treated_as_radio():
    n = 50
    v_true = np.linspace(0.0, 500.0, n)
    h = _header("VELO-LSR", 0.0, v_true[1] * 1e3, n, "m/s")
    _, v = frequency_velocity(h, n)
    assert np.allclose(v, v_true, rtol=0, atol=1e-9)


def test_optical_velocity_axis_converted_to_radio():
    n = 120
    v_opt = np.linspace(0.0, 4000.0, n)  # km/s, optical convention
    h = _header("VOPT", 0.0, v_opt[1] * 1e3, n, "m/s")
    f, v = frequency_velocity(h, n)
    f_true = F0 / (1.0 + v_opt / C)
    assert np.allclose(f, f_true, rtol=1e-12)
    assert np.allclose(v, _truth_from_freq(f_true), rtol=0, atol=1e-9)
    ref_f = (v_opt * u.km / u.s).to(u.Hz, u.doppler_optical(F0 * u.Hz)).value
    assert np.allclose(f, ref_f, rtol=1e-12)


def test_missing_restfrq_defaults_to_hi():
    n = 40
    f_true = 1.40e9 + 26123.046875 * np.arange(n)
    h = _header("FREQ", f_true[0], 26123.046875, n, "Hz", restfrq=None)
    _, v = frequency_velocity(h, n)
    assert np.allclose(v, _truth_from_freq(f_true), rtol=0, atol=1e-9)


def test_channel_count_mismatch_falls_back_with_warning():
    n = 100
    h = _header("FREQ", 1.40e9, 26123.046875, n, "Hz")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f, v = frequency_velocity(h, n + 7)
    assert len(f) == len(v) == n + 7
    assert any("dummy frequency" in str(x.message) for x in w)


def test_no_spectral_axis_falls_back_quietly_when_asked():
    h = fits.Header()
    h["NAXIS"] = 2
    h["CTYPE1"], h["CTYPE2"] = "RA---SIN", "DEC--SIN"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f, v = frequency_velocity(h, 30, quiet=True)
    assert len(f) == 30 and not w
