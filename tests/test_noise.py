"""noise: robust estimators, radiometer prediction, HI sensitivity and the Good/Warn/Bad rules."""

import warnings

import numpy as np

from hi_cube_qa import noise as nz

RNG = np.random.default_rng(1)


def test_mad_rms_gaussian_with_nans():
    x = RNG.normal(0.0, 2.0, 200_000)
    x[::7] = np.nan
    assert abs(nz.mad_rms(x) - 2.0) < 0.03
    assert np.isnan(nz.mad_rms(np.full(5, np.nan)))


def test_mad_rms_planes_matches_mad_rms():
    block = RNG.normal(0.0, 1.0, (4, 60, 60))
    block[2, 5, 5] = 1e6
    med, sig = nz.mad_rms_planes(block)
    for i in range(4):
        assert abs(sig[i] - nz.mad_rms(block[i])) < 1e-3
    assert abs(sig[2] - 1.0) < 0.1 and abs(med[2]) < 0.05


def test_mad_rms_planes_empty_plane():
    block = RNG.normal(0.0, 1.0, (2, 10, 10))
    block[1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        med, sig = nz.mad_rms_planes(block)
    assert np.isfinite(sig[0]) and np.isnan(sig[1]) and np.isnan(med[1])


def test_kurtosis_of_three_distributions():
    g = RNG.normal(0.0, 1.0, (3, 300, 300))
    g[1] = RNG.standard_t(3, (300, 300))
    g[2] = RNG.uniform(-1, 1, (300, 300))
    k = nz.kurtosis_planes(g)
    assert abs(k[0] - 3.0) < 0.1
    assert k[1] > 6.0
    assert abs(k[2] - 1.8) < 0.05


def test_kurtosis_se_formula():
    assert abs(nz.kurtosis_expected_se(2400.0) - 0.1) < 1e-12
    assert nz.kurtosis_expected_se(0.0) == np.sqrt(24.0)
    assert np.allclose(nz.kurtosis_expected_se(np.array([24.0, 96.0])), [1.0, 0.5])


def test_classify_kurtosis_labels():
    se = nz.kurtosis_expected_se(24.0)  # = 1
    kurt = np.array([3.0, 3.0 + 4.9 * se, 3.0 + 5.1 * se, 3.0 - 9.9 * se, 3.0 + 10.1 * se, np.nan])
    q = nz.classify_kurtosis(kurt, 24.0)
    assert list(q) == [nz.GOOD, nz.GOOD, nz.WARN, nz.WARN, nz.BAD, nz.GOOD]


def test_classify_kurtosis_custom_thresholds():
    q = nz.classify_kurtosis(np.array([3.0 + 2.5]), 24.0, warn_sigma=2.0, bad_sigma=3.0)
    assert q[0] == nz.WARN


def test_peak_channel_rms():
    d = np.zeros((2, 4, 4))
    d[0, 1, 1], d[1, 2, 2] = -3.0, 2.0
    assert list(nz.peak_channel_rms(d)) == [3.0, 2.0]


def test_boresight_ignores_bright_source():
    d = RNG.normal(0.0, 1.0, (3, 120, 120))
    d[:, 58:62, 58:62] += 500.0
    hdr = {"CRPIX1": 61, "CRPIX2": 61}
    r = nz.boresight_rms(d, hdr, box_size=100)
    assert np.all(np.abs(r - 1.0) < 0.15)


def test_boresight_clips_contaminated_fraction():
    d = RNG.normal(0.0, 1.0, (4, 100, 100))
    d[:, RNG.random((100, 100)) < 0.2] += 6.0
    assert np.all(
        np.abs(nz.boresight_rms(d, {"CRPIX1": 51, "CRPIX2": 51}, box_size=100) - 1.0) < 0.15
    )
    assert np.all(
        nz.mad_rms_planes(d)[1] > 1.2
    )  # the plain MAD over the same pixels is biased high


def test_boresight_box_at_edge():
    d = RNG.normal(0.0, 1.0, (1, 40, 40))
    assert np.isfinite(nz.boresight_rms(d, {}, box_size=100, centre_pix=(0, 0))[0])


def test_sefd_from_tsys():
    area = np.pi * 6.75**2
    expect = 2 * 1.380649e-23 * 25.0 / (0.8 * area) * 1e26
    assert abs(nz.sefd_from_tsys(25.0, 0.8) - expect) < 1e-9 * expect
    assert 590 < nz.sefd_from_tsys(25.0, 0.8) < 615


def test_sefd_spectrum_flat_cases():
    f = np.linspace(1.2e9, 1.5e9, 7)
    assert np.all(nz.sefd_spectrum(f, sefd=450.0) == 450.0)
    assert np.allclose(nz.sefd_spectrum(f, curve=[]), nz.sefd_from_tsys(25.0, 0.8))


def test_sefd_spectrum_curve_shape_and_scale():
    f = np.linspace(1.1e9, 1.5e9, 9)
    curve = [(1000.0, 400.0), (1300.0, 200.0), (1600.0, 400.0)]
    s = nz.sefd_spectrum(f, tsys=25.0, eta=0.8, curve=curve)
    assert abs(s[4] - nz.sefd_from_tsys(25.0, 0.8)) < 1e-6  # band centre
    assert (
        abs(
            s[0] / s[4]
            - np.interp(1.1e9, [1e9, 1.3e9, 1.6e9], [400, 200, 400])
            / np.interp(1.3e9, [1e9, 1.3e9, 1.6e9], [400, 200, 400])
        )
        < 1e-9
    )


def test_sefd_spectrum_sorts_curve():
    f = np.array([1.2e9, 1.3e9, 1.4e9])
    a = nz.sefd_spectrum(f, curve=[(1100.0, 500.0), (1300.0, 400.0), (1500.0, 500.0)])
    b = nz.sefd_spectrum(f, curve=[(1500.0, 500.0), (1100.0, 500.0), (1300.0, 400.0)])
    assert np.allclose(a, b)


def test_theoretical_noise_scaling():
    s = nz.theoretical_noise(
        sefd=400.0, n_ant=60, t_total_hr=1.0, df_hz=26367.19, n_pol=2, nfac=1.0
    )
    n_bl = 60 * 59 / 2
    assert abs(s - 400.0 / np.sqrt(n_bl * 3600.0 * 26367.19 * 2)) < 1e-15
    assert (
        abs(
            nz.theoretical_noise(sefd=400.0, t_total_hr=4.0)
            / nz.theoretical_noise(sefd=400.0, t_total_hr=1.0)
            - 0.5
        )
        < 1e-12
    )
    assert (
        abs(nz.theoretical_noise(sefd=400.0, nfac=1.7) / nz.theoretical_noise(sefd=400.0) - 1.7)
        < 1e-12
    )


def test_theoretical_noise_from_tsys():
    assert (
        abs(
            nz.theoretical_noise(tsys=25.0, eta=0.8)
            - nz.theoretical_noise(sefd=nz.sefd_from_tsys(25.0, 0.8))
        )
        < 1e-15
    )


def test_noise_spectrum_flat_sefd():
    f = np.linspace(1.3e9, 1.4e9, 5)
    spec = nz.theoretical_noise_spectrum(f, sefd=420.0, n_ant=58, t_total_hr=3.0)
    assert np.allclose(spec, nz.theoretical_noise(sefd=420.0, n_ant=58, t_total_hr=3.0))


def test_flagged_fraction_reduces_time():
    f = np.full(3, 1.35e9)
    flag = np.array([0.0, 0.75, 0.9999])
    s = nz.theoretical_noise_spectrum(f, sefd=420.0, flag_fraction=flag)
    assert abs(s[1] / s[0] - 2.0) < 1e-9  # 1 / sqrt(0.25)
    assert np.isfinite(s[2]) and s[2] < 1e3 * s[0]  # fraction is capped below 1
    full = nz.theoretical_noise_spectrum(f[:1], sefd=420.0, flag_fraction=1.0)[0]
    assert abs(full / s[0] - np.sqrt(1.0 / 0.001)) < 1e-6  # fully flagged: capped at 99.9 %


def test_brightness_temperature():
    tb = nz.brightness_temperature(1e-3, 1.0e9, 10.0, 10.0)  # 1 mJy/beam, 1 GHz, 10"x10"
    assert abs(tb - 1.222e3 * 1.0 / 100.0) < 1e-9
    assert abs(nz.brightness_temperature(1e-3, 2.0e9, 10.0, 10.0) - tb / 4.0) < 1e-9
    assert np.isnan(nz.brightness_temperature(1e-3, 1.4e9, 0.0, 10.0))


def test_nhi_sensitivity_scaling():
    r = nz.nhi_sensitivity(1e-4, 1.4e9, 20.0, 15.0, chan_dv_kms=5.0, dv_kms=20.0, nsigma=3.0)
    tb = nz.brightness_temperature(1e-4, 1.4e9, 20.0, 15.0)
    assert abs(r["tb_sigma_K"] - tb) < 1e-12
    assert abs(r["nhi_chan_cm2"] / (1.823e18 * 3.0 * tb * 5.0) - 1.0) < 1e-12
    assert abs(r["nhi_dv_cm2"] / r["nhi_chan_cm2"] - np.sqrt(20.0 / 5.0)) < 1e-12
    neg = nz.nhi_sensitivity(1e-4, 1.4e9, 20.0, 15.0, chan_dv_kms=-5.0)
    assert (
        abs(
            neg["nhi_dv_cm2"]
            - nz.nhi_sensitivity(1e-4, 1.4e9, 20.0, 15.0, chan_dv_kms=5.0)["nhi_dv_cm2"]
        )
        < 1e-3
    )


def test_nhi_sensitivity_without_beam():
    r = nz.nhi_sensitivity(1e-4, 1.4e9, 0.0, 0.0, 5.0)
    assert np.isnan(r["nhi_dv_cm2"]) and r["dv_kms"] == 20.0


def test_running_median_rms():
    r = np.array([1.0, 2.0, 100.0, 4.0, 5.0, np.nan, 7.0])
    out = nz.running_median_rms(r, window=2)
    assert (
        out[2] == 4.0 and out[0] == 1.5 and out[5] == 6.0
    )  # window [5, nan, 7] -> median of 5 and 7


def test_cumulative_rms():
    out = nz.cumulative_rms(np.array([3.0, 1.0, np.nan, 2.0]))
    assert list(out) == [3.0, 2.0, 2.0, 2.0]


def test_classify_channels_factors():
    rms = np.array([1.0, 1.5, 1.51, 3.0, 3.01, np.nan])
    q = nz.classify_channels(rms, 1.0)
    assert list(q) == [nz.GOOD, nz.GOOD, nz.WARN, nz.WARN, nz.BAD, nz.BAD]


def test_classify_channels_reference():
    rms = np.array([1.0, 1.0, 1.0])
    assert list(nz.classify_channels(rms, np.array([1.0, 0.5, 0.2]))) == [
        nz.GOOD,
        nz.WARN,
        nz.BAD,
    ]  # ratios 1, 2, 5
    assert list(nz.classify_channels(np.array([1.0, 1.0, 5.0]), None)) == [nz.GOOD, nz.GOOD, nz.BAD]
    assert (
        list(nz.classify_channels(rms, np.array([1.0, 1.0]))) == [nz.GOOD] * 3
    )  # wrong length -> median


def _summary(n_bad, n_warn, n_good, rms_good=1.0, theory=1.0):
    quality = np.array([nz.BAD] * n_bad + [nz.WARN] * n_warn + [nz.GOOD] * n_good, dtype=object)
    rms = np.where(quality == nz.GOOD, rms_good, 2.0 * rms_good)
    return nz.channel_summary(rms, quality, theory)


def test_channel_summary_grade():
    assert _summary(0, 0, 100)["overall_quality"] == nz.GOOD
    assert _summary(4, 0, 96)["overall_quality"] == nz.GOOD
    assert _summary(5, 0, 95)["overall_quality"] == nz.WARN
    assert _summary(0, 20, 80)["overall_quality"] == nz.WARN
    assert _summary(15, 0, 85)["overall_quality"] == nz.BAD
    assert _summary(0, 0, 100, rms_good=1.6)["overall_quality"] == nz.WARN  # ratio > 1.5
    assert _summary(0, 0, 100, rms_good=3.1)["overall_quality"] == nz.BAD  # ratio > 3


def test_channel_summary_values():
    s = _summary(2, 3, 5, rms_good=1e-3, theory=0.5e-3)
    assert (s["n_bad"], s["n_warn"], s["n_good"], s["n_total"]) == (2, 3, 5, 10)
    assert abs(s["median_rms_mjy"] - 1.0) < 1e-12 and abs(s["theory_rms_mjy"] - 0.5) < 1e-12
    assert abs(s["rms_to_theory"] - 2.0) < 1e-12 and abs(s["pct_bad"] - 20.0) < 1e-12


def test_channel_summary_without_good_channels():
    q = np.array([nz.BAD, nz.BAD], dtype=object)
    s = nz.channel_summary(np.array([2.0, 4.0]), q, None)
    assert (
        s["median_rms_mjy"] == 3000.0 and s["rms_to_theory"] is None and s["theory_rms_mjy"] is None
    )
