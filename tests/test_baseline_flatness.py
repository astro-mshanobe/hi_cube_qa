"""baseline_flatness: sigma-clipped baseline fit, its amplitude in standard errors of the channel mean, and the grade."""

import numpy as np
from astropy.io import fits
from helpers import uses_tmp_path
from synthetic import cube_header

from hi_cube_qa import baseline_flatness as bf
from hi_cube_qa.reduction import CubeHandle

SIGMA = 1e-3
NCHAN, NY, NX = 64, 40, 40
BMAJ_ASEC, BMIN_ASEC, PIXEL_ASEC = 20.0, 15.0, 6.0
N_BEAMS = NY * NX / (np.pi / (4 * np.log(2)) * BMAJ_ASEC * BMIN_ASEC / PIXEL_ASEC**2)
SEM = SIGMA / np.sqrt(N_BEAMS)


def _check(tmp_path, offsets, **kw):
    rng = np.random.default_rng(3)
    data = (rng.normal(0.0, SIGMA, (NCHAN, NY, NX)) + np.asarray(offsets)[:, None, None]).astype(
        np.float32
    )
    path = tmp_path / "cube.fits"
    fits.PrimaryHDU(data=data, header=cube_header(NCHAN, NY, NX)).writeto(path, overwrite=True)
    with CubeHandle(path) as handle:
        return bf.baseline_check(handle, **kw)


def test_polyfit_ignores_edge_outliers():
    x = np.linspace(-100.0, 100.0, 80)
    y = 2e-5 * x + 1e-4 + np.random.default_rng(0).normal(0.0, 1e-4, x.size)
    y[-3:] += 0.05
    coeffs, keep = bf._robust_polyfit(x, y, 1)
    assert not keep[-3:].any() and keep.sum() >= 76
    assert abs(coeffs[0] - 2e-5) < 3e-6 and abs(coeffs[1] - 1e-4) < 5e-5
    plain = np.polyfit(x, y, 1)
    assert abs(plain[0] - 2e-5) > 10 * abs(
        coeffs[0] - 2e-5
    )  # the ordinary fit is pulled off by the outliers


def test_polyfit_clips_at_four_sigma():
    rng = np.random.default_rng(1)
    x = np.linspace(-100.0, 100.0, 200)
    y = rng.normal(0.0, 1e-4, x.size)
    y[[10, 100, 190]] += 6e-4  # 6 sigma
    _, keep = bf._robust_polyfit(x, y, 1)
    assert not keep[[10, 100, 190]].any() and keep.sum() >= 190


def test_polyfit_without_clipping_keeps_all():
    x = np.arange(10.0)
    y = x.copy()
    y[3] = 50.0
    coeffs, keep = bf._robust_polyfit(x, y, 1, n_sigma=0)
    assert keep.all() and np.allclose(coeffs, np.polyfit(x, y, 1))


def test_classify_limits():
    assert [bf.classify(v) for v in (0.0, 4.99, 5.0, 9.99, 10.0, 50.0)] == [
        "Pass",
        "Pass",
        "Warn",
        "Warn",
        "Fail",
        "Fail",
    ]
    assert bf.classify(np.nan) == "N/A" and bf.classify(None) == "N/A"
    assert bf.classify(3.0, warn=2.0, bad=4.0) == "Warn"


@uses_tmp_path
def test_flat_baseline_passes(tmp_path):
    res = _check(tmp_path, np.zeros(NCHAN))
    assert res["available"] and res["n_channels_used"] == NCHAN
    assert abs(res["typical_sem_jy"] / SEM - 1.0) < 0.05
    assert res["baseline_sigma"] < 5.0 and bf.classify(res["baseline_sigma"]) == "Pass"


@uses_tmp_path
def test_linear_trend_amplitude_and_grade(tmp_path):
    res = _check(tmp_path, np.linspace(-10.0, 10.0, NCHAN) * SEM)
    assert 18.0 < res["baseline_sigma"] < 22.0 and bf.classify(res["baseline_sigma"]) == "Fail"
    assert res["n_clipped"] == 0


@uses_tmp_path
def test_outlier_channels_are_clipped(tmp_path):
    offsets = np.zeros(NCHAN)
    offsets[[0, 1, NCHAN - 1]] = 150.0 * SEM
    res = _check(tmp_path, offsets)
    assert res["n_clipped"] >= 3 and res["baseline_sigma"] < 5.0
    unclipped = _check(tmp_path, offsets, n_sigma=0)
    assert unclipped["n_clipped"] == 0 and unclipped["baseline_sigma"] > 10.0


@uses_tmp_path
def test_unavailable_with_too_few_channels(tmp_path):
    rng = np.random.default_rng(0)
    data = np.full((NCHAN, NY, NX), np.nan, dtype=np.float32)
    data[:2] = rng.normal(0.0, SIGMA, (2, NY, NX))
    path = tmp_path / "sparse.fits"
    fits.PrimaryHDU(data=data, header=cube_header(NCHAN, NY, NX)).writeto(path)
    with CubeHandle(path) as handle:
        assert bf.baseline_check(handle) == {"available": False}
