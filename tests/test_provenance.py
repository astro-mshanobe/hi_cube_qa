"""provenance: imaging provenance from FITS headers and the weighting penalty."""

from astropy.io import fits

from hi_cube_qa import provenance as pv


def _h(**kw):
    h = fits.Header()
    for k, v in kw.items():
        h[k] = v
    return h


def test_nfac_interpolates_the_table_and_clamps_outside_it():
    assert pv.nfac_from_robust(None) is None
    for r, n in pv._ROBUST_NFAC:
        assert abs(pv.nfac_from_robust(r) - n) < 1e-12
    assert abs(pv.nfac_from_robust(-0.25) - 0.5 * (1.20 + 1.10)) < 1e-12
    assert pv.nfac_from_robust(-9.0) == 1.40 and pv.nfac_from_robust(9.0) == 1.00


def test_robust_is_read_from_a_keyword_or_from_the_history():
    assert pv._find_robust(_h(WSCWEIGH="Briggs'(-0.5)")) == -0.5
    assert pv._find_robust(_h(WEIGHT="briggs 0.25")) == 0.25
    h = fits.Header()
    h["HISTORY"] = "wsclean -size 4096 -weight briggs -0.3 -niter 5000"
    assert pv._find_robust(h) == -0.3


def test_natural_and_uniform_weighting_map_to_the_ends_of_the_scale_and_unknown_is_none():
    h = fits.Header()
    h["HISTORY"] = "wsclean -weight natural"
    assert pv._find_robust(h) == 2.0
    g = fits.Header()
    g["HISTORY"] = "wsclean -weight uniform"
    assert pv._find_robust(g) == -2.0
    assert pv._find_robust(fits.Header()) is None
    assert (
        pv._find_robust(_h(WSCWEIGH="uniform")) is None
    )  # a keyword without a robust number is not trusted


def test_niter_and_threshold_come_from_keywords_or_history():
    assert pv._find_int(_h(WSCNITER=1500), ("WSCNITER", "NITER"), r"-niter\s+(\d+)") == 1500
    h = fits.Header()
    h["HISTORY"] = "wsclean -niter 7000 -auto-threshold 3"
    assert pv._find_int(h, ("WSCNITER",), r"-niter\s+(\d+)") == 7000
    assert pv._find_int(fits.Header(), ("WSCNITER",), r"-niter\s+(\d+)") is None
    assert pv.read_provenance(_h(WSCTHRES=2.5e-4))["threshold"] == 2.5e-4


def test_read_provenance_collects_the_observation_metadata_and_the_restoring_beam():
    hdr = _h(
        TELESCOP="MeerKAT",
        OBJECT="G23",
        **{"DATE-OBS": "2020-12-16"},
        BUNIT="JY/BEAM",
        WSCWEIGH="Briggs'(-0.5)",
    )
    bd = {
        "available": True,
        "median_bmaj_asec": 20.0,
        "median_bmin_asec": 15.0,
        "median_bpa_deg": 12.34,
    }
    p = pv.read_provenance(hdr, bd)
    assert (p["telescope"], p["object"], p["date_obs"], p["bunit"], p["robust"]) == (
        "MeerKAT",
        "G23",
        "2020-12-16",
        "JY/BEAM",
        -0.5,
    )
    assert p["restoring_beam"] == '20.00" x 15.00", PA 12.3 deg'
    empty = pv.read_provenance(fits.Header())
    assert empty["telescope"] is None and empty["restoring_beam"] is None and empty["niter"] is None


def test_resolve_nfac_uses_the_config_number_or_the_header_robust_value():
    assert pv.resolve_nfac(1.7, fits.Header()) == (1.7, "config")
    n, src = pv.resolve_nfac("auto", _h(WSCWEIGH="Briggs'(-0.5)"))
    assert abs(n - 1.20) < 1e-12 and src == "auto (robust=-0.5)"
    assert pv.resolve_nfac(" AUTO ", fits.Header()) == (1.0, "auto (robust unknown -> 1.0)")
    assert pv.resolve_nfac("abc", fits.Header()) == (1.0, "default")
