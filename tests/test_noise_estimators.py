"""
Validate hi_cube_qa's noise estimators against astropy and scipy.

Fast vectorised estimators in noise.py are checked, on synthetic cubes with a
known truth, against the standard implementations: astropy.stats.mad_std
(robust RMS) and scipy.stats.kurtosis / kurtosistest.
"""

import numpy as np
from astropy.stats import mad_std
from scipy import stats
from synth import make_cube

from hi_cube_qa import noise


def _cube_with_nans_and_outliers(dtype=np.float64):
    cube, sigma, _ = make_cube(dtype=dtype)
    cube[3, :10, :10] = np.nan  # partly flagged plane
    cube[7] = np.nan  # fully flagged plane
    cube[5, 40, 40] = 1.0  # bright pixel (1000 sigma)
    return cube, sigma


def test_mad_rms_planes_matches_astropy_mad_std():
    cube, _ = _cube_with_nans_and_outliers()
    expected = np.array([mad_std(p, ignore_nan=True) for p in cube])
    _, sig = noise.mad_rms_planes(cube)
    ok = np.isfinite(expected)
    assert np.array_equal(ok, np.isfinite(sig)), "NaN handling differs from astropy"
    # 1.4826 (code) vs 1.482602... (astropy): ~1.5e-6 relative
    assert np.allclose(sig[ok], expected[ok], rtol=1e-4)


def test_mad_rms_planes_median_matches_numpy():
    cube, _ = _cube_with_nans_and_outliers()
    med, _ = noise.mad_rms_planes(cube)
    ok = np.isfinite(med)
    expected = np.array([np.nanmedian(p) for p in cube[ok]])
    assert np.allclose(med[ok], expected, rtol=1e-12, atol=0)


def test_mad_rms_planes_float32_input():
    cube, _ = _cube_with_nans_and_outliers(dtype=np.float32)
    expected = np.array([mad_std(p.astype(np.float64), ignore_nan=True) for p in cube])
    _, sig = noise.mad_rms_planes(cube)
    ok = np.isfinite(expected)
    assert np.allclose(sig[ok], expected[ok], rtol=1e-3)


def test_mad_rms_scalar_matches_astropy():
    cube, _ = _cube_with_nans_and_outliers()
    for plane in cube[[0, 3, 5, 12]]:
        assert np.isclose(noise.mad_rms(plane), mad_std(plane, ignore_nan=True), rtol=1e-4)
    assert np.isnan(noise.mad_rms(cube[7]))


def test_mad_rms_recovers_true_sigma():
    cube, sigma, _ = make_cube(seed=11)
    _, sig = noise.mad_rms_planes(cube)
    # MAD estimator has ~1.2% relative scatter at 9216 pixels; allow ~6 sigma
    assert np.max(np.abs(sig / sigma - 1.0)) < 0.07


def test_mad_rms_robust_to_outliers_where_std_is_not():
    cube, sigma, _ = make_cube(seed=5)
    rng = np.random.default_rng(99)
    n_bad = int(0.01 * cube[0].size)
    for c in range(cube.shape[0]):
        idx = rng.choice(cube[c].size, n_bad, replace=False)
        cube[c].reshape(-1)[idx] += 100.0 * sigma[c]
    _, sig = noise.mad_rms_planes(cube)
    plain_std = cube.reshape(cube.shape[0], -1).std(axis=1)
    assert np.max(np.abs(sig / sigma - 1.0)) < 0.10  # MAD barely moves
    assert np.min(plain_std / sigma) > 3.0  # plain std is wrecked


def test_kurtosis_planes_matches_scipy():
    cube, _ = _cube_with_nans_and_outliers()
    cube[9] = np.random.default_rng(3).laplace(size=cube[9].shape)  # non-Gaussian plane
    k = noise.kurtosis_planes(cube)
    for i, plane in enumerate(cube):
        vals = plane[np.isfinite(plane)]
        if vals.size == 0:
            assert np.isnan(k[i])
            continue
        ref = stats.kurtosis(vals, fisher=False, bias=True)
        assert np.isclose(k[i], ref, rtol=1e-9), f"plane {i}: {k[i]} vs scipy {ref}"


def test_kurtosis_gaussian_cube_is_near_3_and_classified_good():
    cube, _, _ = make_cube(seed=21)
    k = noise.kurtosis_planes(cube)
    n_eff = cube.shape[1] * cube.shape[2]
    assert abs(np.mean(k) - 3.0) < 0.05
    assert np.all(noise.classify_kurtosis(k, n_eff) == noise.GOOD)


def test_kurtosis_heavy_tails_are_flagged_bad():
    rng = np.random.default_rng(8)
    cube = rng.laplace(size=(6, 96, 96))  # Pearson kurtosis = 6
    k = noise.kurtosis_planes(cube)
    assert np.all(np.abs(k - 6.0) < 0.4)
    assert np.all(noise.classify_kurtosis(k, 96 * 96) == noise.BAD)


def test_kurtosis_expected_se_matches_simulation():
    n, trials = 2000, 600
    rng = np.random.default_rng(4)
    ks = np.array([stats.kurtosis(rng.normal(size=n), fisher=False) for _ in range(trials)])
    assert abs(np.std(ks) / noise.kurtosis_expected_se(n) - 1.0) < 0.12


def test_kurtosis_z_agrees_with_scipy_kurtosistest_for_gaussian_data():
    n = 96 * 96
    se = noise.kurtosis_expected_se(n)
    for seed in range(10):
        x = np.random.default_rng(100 + seed).normal(size=n)
        z_ours = (stats.kurtosis(x, fisher=False) - 3.0) / se
        z_scipy = stats.kurtosistest(x).statistic
        assert abs(z_ours - z_scipy) < 0.2, (seed, z_ours, z_scipy)


def test_classify_kurtosis_thresholds():
    n_eff = 2400  # SE = sqrt(24/2400) = 0.1
    kurt = np.array([3.0, 3.2, 3.7, 4.2, 2.3])  # 0, 2, 7, 12, -7 sigma
    q = noise.classify_kurtosis(kurt, n_eff, warn_sigma=5.0, bad_sigma=10.0)
    assert list(q) == [noise.GOOD, noise.GOOD, noise.WARN, noise.BAD, noise.WARN]
