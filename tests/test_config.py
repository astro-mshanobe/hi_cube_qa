"""config: defaults, merging, validation and the example file."""
import re
from pathlib import Path

import yaml

from hi_cube_qa import config as cf
from helpers import raises, uses_tmp_path

ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path, body):
    p = tmp_path / "c.yaml"
    p.write_text(body)
    return p


def _cfg(tmp_path, extra=""):
    return cf.load_config(_write(tmp_path, f"sessions:\n  - root: {tmp_path}\n{extra}"))


@uses_tmp_path
def test_load_config_merges_the_user_file_over_the_defaults(tmp_path):
    c = _cfg(tmp_path, "radiometer:\n  tsys: 30.0\nproducts: [image]\npointings: [p1]\n")
    assert c.radiometer["tsys"] == 30.0 and c.radiometer["eta"] == 0.80               # nested merge keeps the other keys
    assert c.products == ["image"] and c.pointings == ["p1"] and c.cubes is None
    assert c.thresholds["kurtosis_bad_sigma"] == 10.0 and c.plots["max_points"] == 2000
    assert repr(c) == "QAConfig(sessions=1, products=['image'])"


@uses_tmp_path
def test_load_config_does_not_modify_the_module_defaults(tmp_path):
    before = yaml.safe_dump(cf.DEFAULTS)
    _cfg(tmp_path, "radiometer:\n  tsys: 99.0\n")
    assert yaml.safe_dump(cf.DEFAULTS) == before


@uses_tmp_path
def test_an_empty_pointing_list_means_all_pointings(tmp_path):
    assert _cfg(tmp_path, "pointings: []\ncubes: []\n").pointings is None


INVALID_SETTINGS = [
    ("products: []\n", "non-empty list"),
    ("products: some\n", "list or the string 'all'"),
    ("cubes: 3\n", "cubes must be"),
    ("thumbnails:\n  selectors: [rms, foo]\n", "selectors"),
    ("thumbnails:\n  selectors: []\n", "selectors"),
    ("radiometer:\n  n_ant: 1\n", "n_ant must be"),
    ("radiometer:\n  n_ant_fallback: 1\n", "n_ant_fallback"),
    ("radiometer:\n  tsys: 0\n", "tsys"),
    ("radiometer:\n  eta: 1.5\n", "eta"),
    ("radiometer:\n  t_total: -1\n", "t_total"),
    ("radiometer:\n  df: 0\n", "df must"),
]


@uses_tmp_path
def test_invalid_settings_are_rejected_with_a_clear_message(tmp_path):
    for extra, message in INVALID_SETTINGS:
        with raises(ValueError, match=message):
            _cfg(tmp_path, extra)


@uses_tmp_path
def test_radiometer_limits_are_not_checked_when_a_sefd_is_given(tmp_path):
    assert _cfg(tmp_path, "radiometer:\n  sefd: 420.0\n  tsys: 0\n").radiometer["sefd"] == 420.0


@uses_tmp_path
def test_sessions_are_required_and_their_roots_must_exist(tmp_path):
    with raises(ValueError, match="at least one"):
        cf.load_config(_write(tmp_path, "products: all\n"))
    with raises(ValueError, match="missing required key"):
        cf.load_config(_write(tmp_path, "sessions:\n  - path: x\n"))
    with raises(FileNotFoundError, match="not a directory"):
        cf.load_config(_write(tmp_path, f"sessions:\n  - root: {tmp_path}/absent\n"))
    with raises(FileNotFoundError, match="Config file not found"):
        cf.load_config(tmp_path / "absent.yaml")


@uses_tmp_path
def test_output_paths_are_namespaced_by_session_and_the_directory_is_created(tmp_path):
    c = _cfg(tmp_path, f"output:\n  dir: {tmp_path}/out/deep\n")
    p = c.session_out_path("1600000000_sdp_l0", "_channel_qa.csv")
    assert p == tmp_path / "out" / "deep" / "1600000000_sdp_l0_channel_qa.csv" and p.parent.is_dir()


def _example(tmp_path):
    p = tmp_path / "example.yaml"
    cf.write_example_config(p)
    return p


@uses_tmp_path
def test_the_example_file_parses_and_loads_once_its_session_roots_exist(tmp_path):
    p = _example(tmp_path)
    user = yaml.safe_load(p.read_text())
    assert len(user["sessions"]) >= 1
    p.write_text(re.sub(r"root: \S+", f"root: {tmp_path}", p.read_text()))
    assert cf.load_config(p).products == ["dirty", "image", "residual"]


@uses_tmp_path
def test_the_example_file_agrees_with_the_defaults_for_every_setting_it_names(tmp_path):
    user = yaml.safe_load(_example(tmp_path).read_text())
    for section in ("output", "plots", "thresholds", "boresight", "reduce", "sensitivity", "continuum", "thumbnails"):
        for key, val in user[section].items():
            assert cf.DEFAULTS[section][key] == val, f"{section}.{key}"
    for key in ("products", "pointings", "cubes", "flagreport_glob"):
        assert cf.DEFAULTS[key] == user[key], key


@uses_tmp_path
def test_the_example_file_in_the_repository_is_what_write_example_config_writes(tmp_path):
    shipped = ROOT / "qa_session_example.yaml"
    assert shipped.read_text() == _example(tmp_path).read_text()
