"""large_scale on synthetic cubes: noise alone gives a ratio of 1, a coherent ripple is found."""

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from hi_cube_qa import large_scale as ls

PIX, N, NCH, SIG = 6.0, 300, 40, 60.0  # 6 arcsec pixels, smoothing sigma 60 arcsec = 10 pixels
YY, XX = np.mgrid[:N, :N]


def header(pix=PIX, n=N, nch=NCH):
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"], h["NAXIS3"] = 3, n, n, nch
    h["CTYPE1"], h["CRVAL1"], h["CDELT1"], h["CRPIX1"], h["CUNIT1"] = (
        "RA---SIN",
        348.0,
        -pix / 3600.0,
        n / 2.0,
        "deg",
    )
    h["CTYPE2"], h["CRVAL2"], h["CDELT2"], h["CRPIX2"], h["CUNIT2"] = (
        "DEC--SIN",
        -33.0,
        pix / 3600.0,
        n / 2.0,
        "deg",
    )
    return h


def noise_cube(noise=1e-3, nch=NCH, seed=0, beam_px=0.0):
    rng = np.random.default_rng(seed)
    cube = rng.normal(0.0, noise, (nch, N, N))
    if beam_px > 0:  # correlated like a restored image, renormalised to the same pixel rms
        for k in range(nch):
            cube[k] = gaussian_filter(cube[k], beam_px)
            cube[k] *= noise / cube[k].std()
    return cube


def ripple(amp, wavelength_px=67.0, angle=0.6, phase=0.3):
    kx = np.cos(angle) * 2 * np.pi / wavelength_px
    ky = np.sin(angle) * 2 * np.pi / wavelength_px
    return amp * np.sin(kx * XX + ky * YY + phase)


def assess(cube, hdr=None, pb=None, **kw):
    kw.setdefault("sigma_arcsec", SIG)
    kw.setdefault("stride", 4)
    return ls.assess_planes(cube, hdr or header(), pb, **kw)


def raises(exc_type, message, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc_type as err:
        assert message in str(err), (message, str(err))
        return
    raise AssertionError(f"{exc_type.__name__} was not raised")


def test_clipped_std():
    rng = np.random.default_rng(1)
    g = rng.normal(0, 2.0, 20000)
    assert abs(ls.clipped_std(g) - 2.0) < 0.1
    spiked = np.concatenate([g, np.full(40, 500.0)])  # 0.2% compact sources
    assert abs(ls.clipped_std(spiked) - 2.0) < 0.15 and np.std(spiked) > 10
    t = np.linspace(0, 2 * np.pi, 10001)
    assert abs(ls.clipped_std(3.0 * np.sin(t)) - 3.0 / np.sqrt(2)) < 0.02  # no tails: the true rms
    assert ls.clipped_std(np.full(100, 4.0)) == 0.0
    assert np.isnan(ls.clipped_std(np.arange(5.0))) and np.isnan(ls.clipped_std([np.nan] * 50))
    assert abs(ls.clipped_std(np.concatenate([g, [np.nan, np.inf]])) - 2.0) < 0.1


def test_normalised_smoothing():
    img = np.full((120, 120), 5.0)
    mask = np.ones_like(img, bool)
    mask[40:60, 40:60] = False  # a blank hole
    img[~mask] = np.nan
    s = ls.smooth_masked(img, mask, 4.0)
    ok = np.isfinite(s)
    assert np.allclose(s[ok], 5.0) and ok.sum() > 8000  # no dip towards the edges or the hole
    assert not ok[0, 0] and not ok[50, 50] and ok[60, 100]
    full = ls.smooth_masked(np.zeros((60, 60)), np.ones((60, 60), bool), 4.0)
    assert np.isnan(full[0, 0]) and np.isfinite(
        full[30, 30]
    )  # an edge pixel has weight 0.25 to 0.5
    assert np.isnan(ls.smooth_masked(np.ones((60, 60)), np.zeros((60, 60), bool), 4.0)).all()


def test_pixel_scale():
    assert abs(ls.pixel_scale_arcsec(header(pix=4.0)) - 4.0) < 1e-6
    h = fits.Header()
    h["CDELT2"] = 2.0 / 3600.0
    assert abs(ls.pixel_scale_arcsec(h) - 2.0) < 1e-6
    for bad in (fits.Header(), fits.Header([("CDELT2", 0.0)]), {"CDELT2": float("nan")}, {}):
        raises(ls.CubeError, "pixel scale", ls.pixel_scale_arcsec, bad)


def test_noise_alone():
    for beam_px, seed in ((0.0, 1), (3.0, 2)):  # white noise, and noise correlated like an image
        summary, rows = assess(noise_cube(beam_px=beam_px, seed=seed))
        assert summary["status"] == "Pass" and 0.85 < summary["median_ratio"] < 1.15, summary
        assert abs(summary["median_corr"]) < 0.3 and summary["frac_above"] == 0.0, summary
        assert summary["median_excess"] < 0.4 * summary["median_thermal"], summary
        assert summary["n_channels"] == len(rows) == 10
    white, _ = assess(noise_cube(seed=3))
    expect = (
        1e-3 * 1e3 / (2 * np.sqrt(np.pi) * 10.0)
    )  # sigma_pix / (2 sqrt(pi) sigma_s), in mJy/beam
    assert abs(white["median_thermal"] - expect) < 0.15 * expect
    assert abs(white["median_rms"] - expect) < 0.15 * expect


def test_coherent_ripple():
    cube = noise_cube(seed=4) + ripple(0.5e-3)  # half the pixel noise, identical in every channel
    summary, rows = assess(cube)
    attenuation = np.exp(-2 * np.pi**2 * 10.0**2 / 67.0**2)  # of this wavelength by the smoothing
    expect = 0.5 * attenuation / np.sqrt(2)  # mJy/beam rms of the smoothed ripple
    assert (
        summary["status"] == "Fail" and summary["median_ratio"] > 3 and summary["frac_above"] == 1.0
    )
    assert abs(summary["median_excess"] - expect) < 0.15 * expect
    assert summary["median_corr"] > 0.9
    assert abs(np.median([r["excess"] for r in rows]) - expect) < 0.15 * expect
    weak, _ = assess(noise_cube(seed=5) + ripple(0.03e-3))  # 3% of the noise
    assert weak["median_ratio"] < summary["median_ratio"]
    assert weak["median_excess"] < 0.25 * summary["median_excess"]


def test_drifting_and_independent_ripples():
    slow = np.stack([ripple(0.5e-3, phase=0.3 + 0.02 * k) for k in range(NCH)])
    s, _ = assess(noise_cube(seed=6) + slow)
    assert s["status"] == "Fail" and s["median_corr"] > 0.9, s
    rng = np.random.default_rng(7)
    indep = np.stack(
        [ripple(0.5e-3, angle=rng.uniform(0, 3), phase=rng.uniform(0, 6)) for _ in range(NCH)]
    )
    b, _ = assess(noise_cube(seed=8) + indep)
    # known blind spot: the difference map contains the ripple, so the thermal reference rises with it
    assert abs(b["median_corr"]) < 0.3 and b["median_ratio"] < 2.0
    assert b["median_thermal"] > 3 * assess(noise_cube(seed=8))[0]["median_thermal"]


def test_sources_blanks_and_padding():
    cube = noise_cube(seed=9)
    rng = np.random.default_rng(10)
    for _ in range(6):  # bright compact sources in every channel
        y, x = rng.integers(40, 260, 2)
        cube += 30e-3 * np.exp(-0.5 * ((XX - x) ** 2 + (YY - y) ** 2) / 3.0**2)
    s, _ = assess(cube)
    assert s["status"] == "Pass" and s["median_ratio"] < 1.3, s
    blank = cube.copy()
    blank[:, :, :20] = np.nan  # NaN strip, zero padding strip and a blank circle
    blank[:, :15, :] = 0.0
    blank[:, (XX - 150) ** 2 + (YY - 150) ** 2 < 25**2] = np.nan
    s2, _ = assess(blank)
    assert s2["status"] == "Pass" and 0.8 < s2["median_ratio"] < 1.3, s2
    blank[1::4] = np.nan  # the channel after each sampled one is blank, so k-1 is used
    blank[0] = 0.0  # channel 0 is blank itself and skipped
    s3, rows = assess(blank, min_channels=3)
    assert s3["status"] == "Pass" and [r["channel"] for r in rows] == list(range(4, 40, 4))
    offset = noise_cube(seed=19) + 5e-3  # a constant offset, with a strip of zero padding
    offset[:, :, :30] = 0.0
    so, _ = assess(offset)
    assert so["status"] == "Pass" and so["median_ratio"] < 1.3 and so["median_ratio_plain"] < 1.3, (
        so
    )
    rip, _ = assess(blank + ripple(0.5e-3))
    assert rip["status"] == "Fail"


def test_pb_cut():
    r = np.hypot(XX - 150, YY - 150)
    pb = np.broadcast_to(
        np.clip(1.0 - r / 300.0, 0, 1), (NCH, N, N)
    )  # 1 at the centre, 0.5 at 150 px
    outer = np.where(r > 155, 1.0, 0.0) * ripple(1.0e-3)  # only where the PB is below 0.5
    cube = noise_cube(seed=11) + outer
    cut, _ = assess(cube, pb=pb, pb_limit=0.5, min_channels=3)
    nocut, _ = assess(cube, pb=None, min_channels=3)
    assert cut["status"] == "Pass" and cut["median_ratio"] < 1.4 and cut["median_ratio_plain"] < 1.4
    assert abs(cut["median_corr"]) < 0.3, cut
    # a ripple over 16% of the field: the clipped ratio ignores it, the unclipped ratio and corr show it
    assert (
        nocut["median_ratio_plain"] > 2.0
        and nocut["median_corr"] > 0.5
        and nocut["median_ratio"] < 1.4
    )
    whole, _ = assess(noise_cube(seed=12) + ripple(0.5e-3), pb=pb, pb_limit=0.5, min_channels=3)
    assert whole["status"] == "Fail" and whole["median_ratio"] > 3
    inner = np.where(r < 60, 1.0, 0.0) * ripple(2.0e-3, wavelength_px=60)
    part, _ = assess(noise_cube(seed=12) + inner, pb=pb, pb_limit=0.5, min_channels=3)
    assert part["median_ratio_plain"] > 1.5 and part["median_corr"] > 0.5, part
    raises(ls.CubeError, "PB cube has shape", assess, cube, pb=pb[:, :200, :200])


def test_localised_note():
    note = ls.localised_note
    assert "localised structure" in note(dict(status="Pass", median_ratio_plain=3.5))
    assert note(dict(status="Pass", median_ratio_plain=2.9)) == ""
    assert "localised" in note(dict(status="Pass", median_ratio_plain=3.0))
    assert note(dict(status="Fail", median_ratio_plain=30.0)) == ""
    assert note(dict(status="Pass", median_ratio_plain=float("nan"))) == ""


def test_unsuitable_input_is_an_error():
    cube = noise_cube(seed=13)
    raises(ls.CubeError, "not large scale", assess, cube, sigma_arcsec=6.0)
    raises(ls.CubeError, "smaller than 8 smoothing sigmas", assess, cube, sigma_arcsec=400.0)
    raises(ValueError, "stride", assess, cube, stride=0)
    raises(ls.CubeError, "only", assess, cube, min_channels=50)
    raises(ls.CubeError, "only 0", assess, np.full((NCH, N, N), np.nan))
    tiny = noise_cube(seed=14)[:, :100, :100]  # 8 sigmas fit, but too few smoothing areas remain
    raises(
        ls.CubeError,
        "only 0",
        ls.assess_planes,
        tiny,
        header(n=100),
        None,
        sigma_arcsec=SIG,
        stride=4,
        min_channels=1,
    )


def test_classify():
    assert ls.classify(1.0, 0.0) == "Pass" and ls.classify(1.49, 0.0) == "Pass"
    assert ls.classify(1.5, 0.0) == "Warn" and ls.classify(2.99, 0.0) == "Warn"
    assert ls.classify(3.0, 0.0) == "Fail" and ls.classify(50.0, 1.0) == "Fail"
    assert ls.classify(1.2, 0.26) == "Warn" and ls.classify(1.2, 0.25) == "Pass"
    assert ls.classify(float("nan"), 0.0) == "No data"
    assert ls.classify(1.6, 0.0, warn_ratio=2.0) == "Pass"
    assert ls.classify(2.5, 0.0, fail_ratio=2.5) == "Fail"
    assert ls.classify(1.2, 0.3, warn_fraction=0.5) == "Pass"


def test_sample_channels():
    assert ls.sample_channels(100, 20) == [0, 20, 40, 60, 80]
    assert ls.sample_channels(3, 20) == [0] and ls.sample_channels(0, 5) == []
    raises(ValueError, "stride", ls.sample_channels, 10, 0)


def test_assess_file_and_pb_cube():
    with tempfile.TemporaryDirectory() as directory:
        cube_path = Path(directory) / "x.image.fits"
        cube = noise_cube(seed=15).astype(np.float32)
        fits.PrimaryHDU(data=cube, header=header()).writeto(cube_path)
        cfg = dict(sigma_arcsec=SIG, stride=4, pb_limit=0.5, use_pb=True, min_channels=3)
        summary, _ = ls.assess_file(cube_path, cfg)
        assert summary["note"] == "no PB cube next to the cube: no PB cut"
        pb = np.ones_like(cube)
        fits.PrimaryHDU(data=pb, header=header()).writeto(Path(directory) / "x.pb.fits")
        summary, _ = ls.assess_file(cube_path, cfg)
        assert summary["note"] == "PB >= 0.5"
        fits.PrimaryHDU(data=pb[:, :100, :100], header=header()).writeto(
            Path(directory) / "x.pb.fits", overwrite=True
        )
        summary, _ = ls.assess_file(cube_path, cfg)
        assert "another shape" in summary["note"]
        summary, _ = ls.assess_file(cube_path, dict(cfg, use_pb=False))
        assert summary["note"] == ""


def test_smoothing_weight_limit():
    zeros = np.zeros((60, 60))
    mask = np.ones((60, 60), bool)
    smoothed = ls.smooth_masked(zeros, mask, 4.0)
    assert np.isnan(smoothed[30, 4])  # one sigma from the edge: weight 0.84, below 0.98
    assert np.isfinite(smoothed[30, 12])  # three sigma: weight 0.9987
    assert np.isfinite(ls.smooth_masked(zeros, mask, 4.0, min_weight=0.5)[30, 4])


def _rows(ratio, rms=2.0, thermal=1.0):
    return [dict(ratio=ratio, rms=rms, thermal=thermal, ratio_plain=ratio, corr=0.5)] * 5


def test_summarise_limits():
    at_limit = ls.summarise(_rows(1.5), 120.0, 6.0)
    assert (
        at_limit["frac_above"] == 0.0 and at_limit["status"] == "Warn"
    )  # median at the limit warns
    assert ls.summarise(_rows(2.0), 120.0, 6.0)["status"] == "Warn"
    assert ls.summarise(_rows(2.0), 120.0, 6.0, limits=dict(fail_ratio=1.8))["status"] == "Fail"
    assert ls.summarise(_rows(2.0), 120.0, 6.0, limits=dict(warn_ratio=2.5))["status"] == "Pass"
    s = ls.summarise(_rows(1.2, rms=1.2, thermal=1.0), 120.0, 6.0)
    assert abs(s["median_excess"] - np.sqrt(1.2**2 - 1.0**2)) < 1e-12 and s["median_corr"] == 0.5


def test_channel_excess_is_the_difference_in_quadrature():
    _, rows = assess(noise_cube(seed=4) + ripple(0.5e-3))
    for r in rows:
        assert abs(r["excess"] ** 2 - max(r["rms"] ** 2 - r["thermal"] ** 2, 0.0)) < 1e-9
    _, noise_rows = assess(noise_cube(seed=5))
    for r in noise_rows:
        assert abs(r["excess"] ** 2 - max(r["rms"] ** 2 - r["thermal"] ** 2, 0.0)) < 1e-9
        assert r["excess"] < r["rms"]
