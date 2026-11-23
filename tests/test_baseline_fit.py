"""
Validate baseline_flatness._robust_polyfit.

`_legacy_robust_polyfit` is a frozen copy of the original hand-written
iterative MAD clip. It is the reference the current implementation must agree
with, so the p07/p11 conclusions (outlier channels must not drive the fit)
are protected from silent regressions.
"""

import numpy as np

from hi_cube_qa.baseline_flatness import _robust_polyfit


def _legacy_robust_polyfit(x, y, degree, n_sigma=4.0, n_iter=3):
    keep = np.ones(len(x), dtype=bool)
    coeffs = np.polyfit(x, y, degree)
    if n_sigma <= 0:
        return coeffs, keep
    for _ in range(n_iter):
        fit = np.polyval(coeffs, x)
        resid = y - fit
        med = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - med))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(resid[keep]))
        if sigma <= 0 or not np.isfinite(sigma):
            break
        new_keep = np.abs(resid - med) < n_sigma * sigma
        if new_keep.sum() < degree + 2 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= degree + 2 else keep
            break
        keep = new_keep
        coeffs = np.polyfit(x[keep], y[keep], degree)
    return coeffs, keep


SIGMA = 1e-3
OUT_IDX = [10, 50, 120]
OUT_VAL = [+12 * SIGMA, -15 * SIGMA, +9 * SIGMA]


def _series(kind, seed=0, n=200):
    """(x_kms, y, truth_poly_coeffs) for a flat/linear/quadratic baseline + outliers."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-500.0, 500.0, n)
    truth = {"flat": [0.0, 2e-4], "linear": [3e-6, 2e-4], "quad": [-8e-9, 1e-6, 2e-4]}[kind]
    y = np.polyval(truth, x) + SIGMA * rng.normal(size=n)
    y[OUT_IDX] += OUT_VAL
    return x, y, np.array(truth)


def test_matches_legacy_fit_and_flags_the_same_outliers():
    for kind, degree in (("flat", 1), ("linear", 1), ("quad", 2)):
        for seed in range(5):
            x, y, _ = _series(kind, seed)
            c_new, keep_new = _robust_polyfit(x, y, degree)
            c_old, keep_old = _legacy_robust_polyfit(x, y, degree)
            diff = np.max(np.abs(np.polyval(c_new, x) - np.polyval(c_old, x)))
            assert diff < 0.25 * SIGMA, (kind, seed, diff / SIGMA)
            assert not keep_new[OUT_IDX].any(), (kind, seed, "outlier kept")
            assert not keep_old[OUT_IDX].any()
            # masks may differ by a marginal point or two, not more
            assert np.sum(keep_new != keep_old) <= 3, (kind, seed)


def test_recovers_the_true_trend_despite_outliers():
    for kind, degree in (("flat", 1), ("linear", 1), ("quad", 2)):
        x, y, truth = _series(kind, seed=7)
        c, keep = _robust_polyfit(x, y, degree)
        clean = np.polyval(truth, x)
        assert np.max(np.abs(np.polyval(c, x) - clean)) < 1.0 * SIGMA, kind


def test_edge_outliers_would_fake_a_slope_without_clipping():
    """High-leverage outliers at the band edges -- the p07/p11 failure mode."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = np.linspace(-500.0, 500.0, 200)
        y = 2e-4 + SIGMA * rng.normal(size=200)  # truly flat baseline
        y[-4:] += 20 * SIGMA
        y[:2] -= 15 * SIGMA
        flat = np.full_like(x, 2e-4)
        c, keep = _robust_polyfit(x, y, 1)
        c_ols = np.polyfit(x, y, 1)
        assert np.max(np.abs(np.polyval(c, x) - flat)) < 0.5 * SIGMA, seed
        assert np.max(np.abs(np.polyval(c_ols, x) - flat)) > 1.5 * SIGMA, seed
        assert not keep[[0, 1, -4, -3, -2, -1]].any(), seed


def test_clipping_disabled_equals_plain_polyfit():
    x, y, _ = _series("linear")
    c, keep = _robust_polyfit(x, y, 1, n_sigma=0)
    assert keep.all()
    assert np.allclose(c, np.polyfit(x, y, 1), rtol=1e-12, atol=0)


def test_returns_polyval_ordered_coefficients_and_boolean_mask():
    x, y, _ = _series("linear")
    c, keep = _robust_polyfit(x, y, 1)
    assert keep.dtype == bool and keep.shape == x.shape
    assert len(c) == 2 and abs(c[0] - 3e-6) < 3e-7  # slope first, as np.polyval expects


def test_clean_data_clips_almost_nothing():
    rng = np.random.default_rng(2)
    x = np.linspace(-500, 500, 400)
    y = 1e-4 + SIGMA * rng.normal(size=400)
    _, keep = _robust_polyfit(x, y, 1)
    assert (~keep).sum() <= 4  # 4-sigma clip of Gaussian noise: ~0.006% expected


def test_clipping_uses_the_median_and_mad_so_a_contaminated_minority_does_not_hide_itself():
    rng = np.random.default_rng(0)
    x = np.linspace(-100.0, 100.0, 200)
    y = rng.normal(0.0, 1.0, x.size)
    bad = rng.random(x.size) < 0.15
    y[bad] += 20.0
    _, keep = _robust_polyfit(x, y, 1)
    assert not keep[bad].any()
    assert keep[~bad].all()  # a mean-centred clip would also reject some clean points


def test_a_clip_that_would_leave_too_few_points_for_the_degree_keeps_every_point():
    x = np.arange(4.0)
    y = np.array(
        [-50.7, -1.4, -0.2, -0.9]
    )  # clipping the first point would leave 3, fewer than the 5 a cubic needs
    _, keep = _robust_polyfit(x, y, 3)
    assert keep.all()
