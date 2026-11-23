"""discovery: walking a CARACal session tree into pointing / cube units, with filters, plus flag-report discovery."""

from pathlib import Path, PurePath

from helpers import raises, uses_tmp_path

from hi_cube_qa import discovery as dc


def _touch(root, *parts):
    p = Path(root).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def _tree(tmp_path):
    car = tmp_path / "1600000000_sdp_l0" / "caracal"
    for ptg, cube, prods in (
        ("p13", "cube_1", ("image", "residual")),
        ("p7", "cube_1", ("dirty", "image")),
        ("p13b", "cube_2", ("image",)),
        ("p7", "cube_2", ("psf", "model", "image")),
    ):
        for prod in prods:
            _touch(car, f"output_hi_{ptg}", "cubes", cube, f"G23_x_{ptg}_HI.{prod}.fits")
    _touch(car, "output_hi_1gc", "diagnostic_plots", "a-flagging-summary-0.json")
    _touch(
        car, "output_hi_p7", "cubes", "cube_1", "G23_x_p7_HI-0001-image.fits"
    )  # channel image, no product token
    _touch(car, "output_hi_p7", "cubes", "cube_1", "notes.txt")
    return car


@uses_tmp_path
def test_unit_order_and_obsid(
    tmp_path,
):
    s = dc.discover_session(_tree(tmp_path))
    assert s.obsid == "1600000000_sdp_l0"
    assert [u.key for u in s.units] == ["p7/cube_1", "p7/cube_2", "p13/cube_1", "p13b/cube_2"]
    assert s.pointings == ["p7", "p13", "p13b"]
    assert [u.cube_short for u in s.units] == ["c1", "c2", "c1", "c2"]


@uses_tmp_path
def test_product_order_and_labels(tmp_path):
    s = dc.discover_session(_tree(tmp_path))
    u = s.units_for("p7")[1]
    assert [p.product for p in u.products] == ["image", "model", "psf"]
    assert [p.label for p in u.products] == ["c2-image", "c2-model", "c2-psf"]
    assert [p.product for p in s.units[0].products] == ["dirty", "image"]


@uses_tmp_path
def test_unparseable_files_are_skipped(tmp_path):
    s = dc.discover_session(_tree(tmp_path))
    names = [cp.path.name for u in s.units for cp in u.products]
    assert not any("flagging" in n or n.endswith(".txt") or "0001" in n for n in names)
    assert all(u.pointing.startswith("p") for u in s.units)


@uses_tmp_path
def test_product_filter(tmp_path):
    s = dc.discover_session(_tree(tmp_path), products=["residual"])
    assert [u.key for u in s.units] == ["p13/cube_1"]
    s = dc.discover_session(_tree(tmp_path), products=["DIRTY", "Image"])
    assert {p.product for u in s.units for p in u.products} == {"dirty", "image"}


@uses_tmp_path
def test_pointing_and_cube_filters(tmp_path):
    car = _tree(tmp_path)
    assert [u.key for u in dc.discover_session(car, pointings=["p13"]).units] == ["p13/cube_1"]
    for spec in (["cube_2"], ["c2"], [2], ["2"]):
        keys = [u.key for u in dc.discover_session(car, cubes=spec).units]
        assert keys == ["p7/cube_2", "p13b/cube_2"], spec
    assert len(dc.discover_session(car, cubes="all").units) == 4


def test_norm_cube():
    assert [dc._norm_cube(x) for x in ("cube_1", "Cube1", "c1", 1, " 1 ")] == ["1"] * 5


def test_parse_product():
    from pathlib import PurePath

    assert dc._parse_product(PurePath("G23_a_HI_1.residual.fits")) == "residual"
    assert dc._parse_product(PurePath("G23_a_HI.Image.fits")) == "image"
    assert dc._parse_product(PurePath("G23_a_HI-0001-image.fits")) is None
    assert dc._parse_product(PurePath("cube.txt")) is None


@uses_tmp_path
def test_missing_session_root(tmp_path):
    with raises(FileNotFoundError):
        dc.discover_session(tmp_path / "nope")


@uses_tmp_path
def test_summarise(tmp_path):
    text = dc.summarise(dc.discover_session(_tree(tmp_path)))
    assert text.splitlines()[0] == "Session 1600000000_sdp_l0: 4 unit(s) across 3 pointing(s)"
    assert "p7/cube_1" in text and "[dirty, image]" in text


@uses_tmp_path
def test_flag_report_glob_fallbacks(tmp_path):
    car = tmp_path / "caracal"
    hit = _touch(car, "output_hi_1gc", "diagnostic_plots", "s-flagging-summary-0.json")
    _touch(car, "output_hi_1gc", "diagnostic_plots", "s-crosscal-summary-0.json")
    paths, pat = dc.discover_flag_reports(
        car, "output_hi_1gc/diagnostic_plots/*-flagging-summary-*.json"
    )
    assert paths == [hit] and pat.startswith("output_hi_1gc")
    paths, pat = dc.discover_flag_reports(car, "no/such/*.json")
    assert paths == [hit] and pat != "no/such/*.json"
    paths, pat = dc.discover_flag_reports(
        car, ["a/*.json", "output_hi_1gc/**/*flagging-summary-*.json"]
    )
    assert paths == [hit]


@uses_tmp_path
def test_flag_report_discovery_disabled_or_empty(tmp_path):
    assert dc.discover_flag_reports(tmp_path, None) == ([], None)
    assert dc.discover_flag_reports(tmp_path, "") == ([], None)
    assert dc.discover_flag_reports(tmp_path, "*.json") == ([], None)


@uses_tmp_path
def test_continuum_subtracted_line_cube(tmp_path):
    car = tmp_path / "1600000000_sdp_l0" / "caracal"
    for name in ("G23_x_p7_HI.image.fits", "imsub-line.fits", "imsub.fits"):
        _touch(car, "output_hi_p7", "cubes", "cube_2", name)
    assert dc._parse_product(PurePath("imsub-line.fits")) == "line"
    assert dc._parse_product(PurePath("imsub.fits")) == "imsub"
    unit = dc.discover_session(car).units[0]
    assert [p.product for p in unit.products] == ["image", "line", "imsub"]
    assert [p.label for p in unit.products] == ["c2-image", "c2-line", "c2-imsub"]
    assert [p.product for p in dc.discover_session(car, products=["line"]).units[0].products] == [
        "line"
    ]
