"""config: defaults, merging, validation and the example file."""

import re
from pathlib import Path

import yaml
from helpers import raises, uses_tmp_path

from hi_cube_qa import config as cf

ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path, body):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


def _cfg(tmp_path, extra=""):
    return cf.load_config(_write(tmp_path, f"sessions:\n  - root: {tmp_path}\n{extra}"))


@uses_tmp_path
def test_user_file_overrides_defaults(tmp_path):
    c = _cfg(tmp_path, "radiometer:\n  tsys: 30.0\nproducts: [image]\npointings: [p1]\n")
    assert (
        c.radiometer["tsys"] == 30.0 and c.radiometer["eta"] == 0.80
    )  # nested merge keeps the other keys
    assert c.products == ["image"] and c.pointings == ["p1"] and c.cubes is None
    assert c.thresholds["kurtosis_bad_sigma"] == 10.0 and c.plots["max_points"] == 2000
    assert repr(c) == "QAConfig(sessions=1, products=['image'])"


@uses_tmp_path
def test_defaults_are_not_modified(tmp_path):
    before = yaml.safe_dump(cf.DEFAULTS)
    _cfg(tmp_path, "radiometer:\n  tsys: 99.0\n")
    assert yaml.safe_dump(cf.DEFAULTS) == before


@uses_tmp_path
def test_empty_pointing_list_means_all(tmp_path):
    assert _cfg(tmp_path, "pointings: []\ncubes: []\n").pointings is None


INVALID_SETTINGS = [
    ("products: []\n", "non-empty list"),
    ("products: some\n", "list or the string 'all'"),
    ("cubes: 3\n", "cubes must be"),
    ("thumbnails:\n  selectors: [rms, foo]\n", "selectors"),
    ("thumbnails:\n  selectors: []\n", "selectors"),
    ("radiometer:\n  n_ant: 1\n", "n_ant must be"),
    ("radiometer:\n  n_ant_fallback: 1\n", "n_ant_fallback"),
    ("radiometer:\n  sefd: null\n  tsys: 0\n", "tsys"),
    ("radiometer:\n  sefd: 0\n", "sefd"),
    ("radiometer:\n  t_total: soon\n", "t_total"),
    ("radiometer:\n  t_total: 0\n", "t_total"),
    ("radiometer:\n  t_total_fallback: -1\n", "t_total_fallback"),
    ("radiometer:\n  sefd: -5\n", "sefd"),
    ("radiometer:\n  sefd: null\n  eta: 1.5\n", "eta"),
    ("radiometer:\n  t_total: -1\n", "t_total"),
    ("radiometer:\n  df: 0\n", "df must"),
    ("large_scale:\n  products: some\n", "large_scale.products"),
    ("large_scale:\n  products: []\n", "large_scale.products"),
    ("large_scale:\n  stride: 0\n", "large_scale.stride"),
    ("large_scale:\n  sigma_arcsec: -1\n", "large_scale.sigma_arcsec"),
    ("large_scale:\n  min_channels: 0\n", "large_scale.min_channels"),
    ("baseline:\n  degree: -1\n", "baseline.degree"),
    ("plots:\n  initially_visible: []\n", "initially_visible"),
    ("plots:\n  initially_visible: residual\n", "initially_visible"),
]


@uses_tmp_path
def test_invalid_settings_are_rejected(tmp_path):
    for extra, message in INVALID_SETTINGS:
        with raises(ValueError, match=message):
            _cfg(tmp_path, extra)


@uses_tmp_path
def test_sefd_skips_tsys_and_eta_checks(tmp_path):
    assert _cfg(tmp_path, "radiometer:\n  sefd: 420.0\n  tsys: 0\n").radiometer["sefd"] == 420.0


@uses_tmp_path
def test_sessions_are_required(tmp_path):
    with raises(ValueError, match="at least one"):
        cf.load_config(_write(tmp_path, "products: all\n"))
    with raises(ValueError, match="missing required key"):
        cf.load_config(_write(tmp_path, "sessions:\n  - path: x\n"))
    with raises(FileNotFoundError, match="not a directory"):
        cf.load_config(_write(tmp_path, f"sessions:\n  - root: {tmp_path}/absent\n"))
    with raises(FileNotFoundError, match="Config file not found"):
        cf.load_config(tmp_path / "absent.yaml")


@uses_tmp_path
def test_session_output_path(tmp_path):
    c = _cfg(tmp_path, f"output:\n  dir: {tmp_path}/out/deep\n")
    p = c.session_out_path("1600000000_sdp_l0", "_channel_qa.csv")
    assert p == tmp_path / "out" / "deep" / "1600000000_sdp_l0_channel_qa.csv" and p.parent.is_dir()


def _example(tmp_path):
    p = tmp_path / "example.yaml"
    cf.write_example_config(p)
    return p


@uses_tmp_path
def test_example_file_loads(tmp_path):
    p = _example(tmp_path)
    user = yaml.safe_load(p.read_text())
    assert len(user["sessions"]) >= 1
    p.write_text(re.sub(r"root: \S+", f"root: {tmp_path}", p.read_text()))
    assert cf.load_config(p).products == ["dirty", "image", "residual", "line"]


@uses_tmp_path
def test_example_file_matches_defaults(tmp_path):
    user = yaml.safe_load(_example(tmp_path).read_text())
    for section in (
        "output",
        "plots",
        "thresholds",
        "boresight",
        "reduce",
        "sensitivity",
        "continuum",
        "thumbnails",
        "large_scale",
        "baseline",
    ):
        for key, val in user[section].items():
            assert cf.DEFAULTS[section][key] == val, f"{section}.{key}"
    for key in ("products", "pointings", "cubes", "flagreport_glob"):
        assert cf.DEFAULTS[key] == user[key], key


@uses_tmp_path
def test_shipped_example_is_current(tmp_path):
    shipped = ROOT / "qa_session_example.yaml"
    assert shipped.read_text() == _example(tmp_path).read_text()


@uses_tmp_path
def test_the_default_sefd_is_flat_425_jy_and_the_time_and_width_are_still_checked(tmp_path):
    assert _cfg(tmp_path).radiometer["sefd"] == 425.0
    with raises(ValueError, match="t_total"):
        _cfg(tmp_path, "radiometer:\n  sefd: 420.0\n  t_total: 0\n")
    with raises(ValueError, match="df must"):
        _cfg(tmp_path, "radiometer:\n  sefd: 420.0\n  df: -1\n")


@uses_tmp_path
def test_on_source_time_defaults_to_auto_with_a_one_hour_fallback_and_accepts_hours(tmp_path):
    assert _cfg(tmp_path).radiometer["t_total"] == "auto"
    assert _cfg(tmp_path).radiometer["t_total_fallback"] == 1.0
    assert _cfg(tmp_path, "radiometer:\n  t_total: 2.5\n").radiometer["t_total"] == 2.5
    assert _cfg(tmp_path, "radiometer:\n  t_total: AUTO\n").radiometer["t_total"] == "AUTO"
