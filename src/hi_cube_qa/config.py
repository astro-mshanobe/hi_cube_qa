"""YAML based configuration (session-oriented).

A config describes one or more observation *sessions* (each a CARACal
caracal/ root). Cubes are auto-discovered under each session; the user only
chooses which imaging products to plot and (optionally) which pointings.
One HTML report is produced per session.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "output": {
        "dir": "./qa_output",
        "html": True,
        "csv": True,
        "log": True,
        # Also save key figures as standalone PNGs (needs the 'kaleido' package;
        # degrades gracefully with a warning if it is missing).
        "save_png": False,
        "png_figures": [
            "summary",
            "rms_curve",
            "rms_vs_theory",
            "noise_overview",
            "bandspec",
            "nhi",
        ],
        "png_scale": 2,
    },
    # Plot resolution. By default the plotted spectra are decimated to at most
    # max_points per trace (keeps the HTML small; CSVs stay full resolution and
    # Bad channels are always kept). Set full_resolution: true (or pass
    # --full-res on the command line) to draw every channel instead.
    "plots": {
        "full_resolution": False,
        "max_points": 2000,
        # Products drawn when a figure opens: 'all', or a list such as [residual, line].
        # The others stay in the legend and a click on an entry switches them on.
        "initially_visible": "all",
    },
    # List of sessions: [{root: <path to caracal/>}, ...]
    "sessions": [],
    # Which imaging products to load per cube. 'all' loads everything found.
    "products": ["dirty", "image", "residual", "line"],
    # Optional pointing filter (e.g. [p63, p64]); null/empty -> all pointings.
    "pointings": None,
    # Optional cube filter (e.g. [cube_1] or [1]); null/'all' -> every cube_N.
    "cubes": None,
    # Flag-report JSONs, globbed relative to each session root (string or list
    # of patterns tried in order). null disables. If the configured pattern
    # finds nothing, discovery falls back to progressively broader globs.
    "flagreport_glob": "output_hi_1gc/diagnostic_plots/*-flagging-summary-*.json",
    "radiometer": {
        "sefd": None,
        "tsys": 25.0,
        "eta": 0.80,
        # Number of antennas: 'auto' reads the real count for this session
        # from the CARACal obsinfo summary.json (ANT field) - sessions can
        # differ if antennas were offline for maintenance; falls back to
        # n_ant_fallback if unavailable. Or set a fixed number directly.
        "n_ant": "auto",
        "n_ant_fallback": 60,
        "t_total": 1.0,
        "df": 26367.19,
        "nfac": 1.0,  # number, or 'auto' (from header Briggs robust)
        # Frequency dependence of the SEFD: 'default' (built-in approximate
        # MeerKAT L-band curve), 'flat' (use tsys/eta only), or a list of
        # [freq_MHz, sefd_Jy] points to interpolate.
        "sefd_curve": "default",
        # Effective flagged fraction folded into the theory (reduces t_eff):
        # 'auto' (from the flag report), a number in [0,1), or null (=0).
        "flag_fraction": "auto",
    },
    "spectral": {
        "z_source": None,
    },
    # HI column-density sensitivity reporting.
    "sensitivity": {
        "nhi_dv_kms": 20.0,  # line width over which to quote 3-sigma N_HI
        "nsigma": 3.0,
    },
    # Continuum-residual diagnostic (persistence map).
    "continuum": {
        "sigma": 3.0,  # per-channel threshold for the persistence map
    },
    # Streaming reducer.
    "reduce": {
        "chunk": 64,  # channels read/processed per block
        "maps": True,  # render 2-D noise + continuum maps
    },
    "thumbnails": {
        # Channel selectors for the thumbnail strip: any of "rms", "snr".
        "selectors": ["rms", "snr"],
    },
    "boresight": {
        "enable": True,
        "box_size": 100,
        "centre_pix": None,
        "sigma_clip": 3.0,
        "clip_iterations": 3,
    },
    # Excess noise power on large angular scales (large_scale.py). It smooths about
    # nchan/stride channels per cube, so by default only the continuum-subtracted line cube
    # (imsub-line.fits) is assessed; units without one are reported as not assessed.
    "large_scale": {
        "enable": True,
        "products": "reference",  # 'reference' (the line cube), 'all', or a list
        "sigma_arcsec": 120.0,  # Gaussian smoothing sigma
        "stride": 20,  # assess every stride-th channel
        "pb_limit": 0.5,  # with a *.pb.fits cube next to the cube
        "use_pb": True,
        "min_channels": 5,
    },
    # Baseline flatness: polynomial fit to the per-channel mean (baseline_flatness.py).
    "baseline": {
        "enable": True,
        "degree": 1,
        "sigma_clip": 4.0,
    },
    "thresholds": {
        "rms_warn_factor": 1.5,
        "rms_bad_factor": 3.0,
        "bad_chan_warn_pct": 5.0,
        "bad_chan_bad_pct": 15.0,
        "beam_scatter_warn": 0.5,
        "flag_warn_pct": 10.0,
        "flag_bad_pct": 30.0,
        # Kurtosis (non-Gaussianity): channel flagged when |kurtosis-3| exceeds
        # this many standard errors of the estimator (beam-corrected effective N).
        "kurtosis_warn_sigma": 5.0,
        "kurtosis_bad_sigma": 10.0,
        # Large-scale noise: median ratio of the smoothed-map rms to the thermal
        # estimate, and the fraction of channels above the warning ratio.
        "ls_warn_ratio": 1.5,
        "ls_fail_ratio": 3.0,
        "ls_warn_fraction": 0.25,
        # Baseline flatness: fitted amplitude in standard errors of the channel mean.
        "baseline_warn_sigma": 5.0,
        "baseline_bad_sigma": 10.0,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# Public API
class QAConfig:
    """Parsed and validated QA configuration."""

    def __init__(self, cfg: dict):
        self._cfg = cfg

    @property
    def output(self) -> dict:
        return self._cfg["output"]

    @property
    def sessions(self) -> list[dict]:
        return self._cfg["sessions"]

    @property
    def products(self) -> list[str] | str:
        return self._cfg["products"]

    @property
    def pointings(self) -> list[str] | None:
        return self._cfg.get("pointings") or None

    @property
    def cubes(self) -> list | str | None:
        return self._cfg.get("cubes") or None

    @property
    def flagreport_glob(self) -> str | None:
        return self._cfg.get("flagreport_glob")

    @property
    def radiometer(self) -> dict:
        return self._cfg["radiometer"]

    @property
    def spectral(self) -> dict:
        return self._cfg["spectral"]

    @property
    def sensitivity(self) -> dict:
        return self._cfg["sensitivity"]

    @property
    def continuum(self) -> dict:
        return self._cfg["continuum"]

    @property
    def reduce(self) -> dict:
        return self._cfg["reduce"]

    @property
    def plots(self) -> dict:
        return self._cfg["plots"]

    @property
    def thumbnails(self) -> dict:
        return self._cfg["thumbnails"]

    @property
    def boresight(self) -> dict:
        return self._cfg["boresight"]

    @property
    def large_scale(self) -> dict:
        return self._cfg["large_scale"]

    @property
    def baseline(self) -> dict:
        return self._cfg["baseline"]

    @property
    def thresholds(self) -> dict:
        return self._cfg["thresholds"]

    def out_dir(self) -> Path:
        d = Path(self.output["dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def session_out_path(self, obsid: str, suffix: str) -> Path:
        """Path inside the output dir, namespaced by session obsid."""
        d = self.out_dir()
        return d / f"{obsid}{suffix}"

    def __repr__(self) -> str:
        return f"QAConfig(sessions={len(self.sessions)}, products={self.products!r})"


def load_config(yaml_path: str | Path) -> QAConfig:
    """Load a YAML config file, merge with defaults, validate, and return a QAConfig."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULTS, user_cfg)

    # validate sessions
    if not cfg["sessions"]:
        raise ValueError("Config must define at least one entry under 'sessions:'")

    for i, sess in enumerate(cfg["sessions"]):
        if "root" not in sess:
            raise ValueError(f"sessions[{i}] is missing required key 'root'")
        if not Path(sess["root"]).is_dir():
            raise FileNotFoundError(f"sessions[{i}] root is not a directory: {sess['root']}")

    # validate products
    prods = cfg["products"]
    if isinstance(prods, str):
        if prods.lower() != "all":
            raise ValueError("products must be a list or the string 'all'")
    elif not isinstance(prods, list) or not prods:
        raise ValueError("products must be a non-empty list or 'all'")

    # validate cube filter
    cubes = cfg.get("cubes")
    if cubes is not None and not (
        (isinstance(cubes, str) and cubes.lower() == "all") or isinstance(cubes, list)
    ):
        raise ValueError("cubes must be null, 'all', or a list (e.g. [cube_1] or [1])")

    # validate thumbnail selectors
    sel = cfg["thumbnails"].get("selectors", [])
    valid_sel = {"rms", "snr"}
    if not sel or any(s not in valid_sel for s in sel):
        raise ValueError("thumbnails.selectors must be a non-empty subset of ['rms', 'snr']")

    # validate radiometer equation
    rad = cfg["radiometer"]
    n_ant_is_auto = isinstance(rad["n_ant"], str) and rad["n_ant"].strip().lower() == "auto"
    if not n_ant_is_auto and rad["n_ant"] < 2:
        raise ValueError("radiometer.n_ant must be >= 2, or 'auto'")
    if rad.get("n_ant_fallback", 60) < 2:
        raise ValueError("radiometer.n_ant_fallback must be >= 2")
    if rad["sefd"] is None:
        if rad["tsys"] <= 0:
            raise ValueError("radiometer.tsys must be positive")
        if not (0 < rad["eta"] <= 1):
            raise ValueError("radiometer.eta must be in (0, 1)")
        if rad["t_total"] <= 0:
            raise ValueError("radiometer.t_total must be positive")
        if rad["df"] <= 0:
            raise ValueError("radiometer.df must be positive (channel width in Hz)")

    shown = cfg["plots"]["initially_visible"]
    if not (shown == "all" or (isinstance(shown, list) and shown)):
        raise ValueError("plots.initially_visible must be 'all' or a non-empty list of products")

    ls = cfg["large_scale"]
    prods = ls["products"]
    if not (prods in ("reference", "all") or (isinstance(prods, list) and prods)):
        raise ValueError("large_scale.products must be 'reference', 'all' or a non-empty list")
    if int(ls["stride"]) < 1:
        raise ValueError("large_scale.stride must be at least 1")
    if float(ls["sigma_arcsec"]) <= 0:
        raise ValueError("large_scale.sigma_arcsec must be positive")
    if int(ls["min_channels"]) < 1:
        raise ValueError("large_scale.min_channels must be at least 1")
    if int(cfg["baseline"]["degree"]) < 0:
        raise ValueError("baseline.degree must be 0 or more")

    return QAConfig(cfg)


def write_example_config(path: str | Path = "qa_session_example.yaml") -> None:
    """Write a fully-commented example session config to *path*."""
    example = """\
# hi_cube_qa - example session configuration
# One HTML report is produced per session. Cubes are auto-discovered under each
# session root; you only choose which products to plot.

output:
    dir: ./qa_output
    html: true
    csv: true
    log: true
    save_png: false        # also save key figures as PNGs (needs 'kaleido')
    png_figures: [summary, rms_curve, rms_vs_theory, noise_overview, bandspec, nhi]
    png_scale: 2

plots:
    full_resolution: false # false = downsampled plots (default); true = every
                           # channel. Override per-run with --full-res / --downsample.
    max_points: 2000       # cap points/trace when downsampling (CSVs stay full-res;
                           # Bad channels always kept).
    initially_visible: all # products drawn when a figure opens: all | [residual, line].
                           # The others stay in the legend; a click switches them on.

# One or more CARACal session roots (the caracal/ directory of each obs).
sessions:
    - root: /data/G23/1610268336_sdp_l0/caracal
    - root: /data/G23/1609151470_sdp_l0/caracal

# Imaging products to load per cube. 'line' is the continuum-subtracted cube
# (imsub-line.fits) where a pointing has one. Use 'all' for every product found.
products: [dirty, image, residual, line]

# Optional pointing filter. null -> all pointings found in each session.
pointings: null
# pointings: [p63, p64]

# Optional cube filter. null/'all' -> every cube_N found in each pointing.
# Accepts cube_1, c1 or 1 (equivalent). E.g. [cube_1] to check only cube_1.
cubes: null
# cubes: [cube_1]

# Flagging-summary JSONs, globbed relative to each session root. null disables.
flagreport_glob: output_hi_1gc/diagnostic_plots/*-flagging-summary-*.json

radiometer:
    sefd:    null
    tsys:    25.0
    eta:     0.80
    n_ant:          auto  # read the real per-session antenna count from obsinfo;
                          # or set a fixed number directly
    n_ant_fallback: 60    # used only if 'auto' finds nothing
    t_total: 1.0          # hours on-source (set to the actual track length)
    df:      26367.19     # Hz; auto-overridden per cube from the FITS header
    nfac:    auto         # weighting penalty: a number, or 'auto' (from the header's Briggs robust)
    sefd_curve:    default   # 'default' freq-dependent curve | 'flat' | [[MHz,Jy],...]
    flag_fraction: auto      # fold flagging into theory: 'auto' | number | null

spectral:
    z_source: null        # set e.g. 0.05 to mark a target redshift

sensitivity:
    nhi_dv_kms: 20.0      # quote 3-sigma N_HI over this line width
    nsigma:     3.0

continuum:
    sigma: 3.0            # per-channel threshold for the continuum persistence map

reduce:
    chunk: 64             # channels read per block (memory/speed knob)
    maps:  true           # render 2-D noise + continuum maps

thumbnails:
    selectors: [rms, snr] # show best/median/worst channels by RMS and by SNR

boresight:
    enable: true
    box_size: 100
    centre_pix: null
    sigma_clip: 3.0
    clip_iterations: 3

thresholds:
    rms_warn_factor:   1.5
    rms_bad_factor:    3.0
    bad_chan_warn_pct: 5.0
    bad_chan_bad_pct:  15.0
    beam_scatter_warn: 0.5
    flag_warn_pct:     10.0
    flag_bad_pct:      30.0
    kurtosis_warn_sigma: 5.0   # |kurtosis-3| > N standard errors -> Warn
    kurtosis_bad_sigma:  10.0  # |kurtosis-3| > N standard errors -> Bad
    ls_warn_ratio:     1.5     # large-scale noise: median ratio to thermal -> Warn
    ls_fail_ratio:     3.0     # ... -> Fail
    ls_warn_fraction:  0.25    # Warn when more channels than this exceed ls_warn_ratio
    baseline_warn_sigma: 5.0   # baseline amplitude in standard errors -> Warn
    baseline_bad_sigma:  10.0  # ... -> Fail

# Excess noise power on large angular scales. Smooths about nchan/stride channels per
# cube, so by default only the line cube (imsub-line.fits) is assessed; units
# without one are reported as not assessed.
large_scale:
    enable: true
    products: reference    # 'reference' (the line cube) | 'all' | [line, residual]
    sigma_arcsec: 120.0    # Gaussian smoothing sigma
    stride: 20             # assess every stride-th channel
    pb_limit: 0.5          # used when a *.pb.fits cube sits next to the cube
    use_pb: true
    min_channels: 5

# Baseline flatness: polynomial fit to the per-channel mean, for every product.
baseline:
    enable: true
    degree: 1
    sigma_clip: 4.0        # clip outlier channels at this many sigma before fitting
"""
    with open(path, "w") as f:
        f.write(example)
    print(f"[config] Example config written -> {path}")
