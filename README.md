# hi_cube_qa

Quality assessment of MeerKAT HI cubes. For each observation session (one CARACal
`caracal/` root) it finds the imaging cubes, measures noise, beam and continuum-residual
diagnostics, and writes one interactive HTML report, CSV tables and a survey-wide summary.

New to the tool? [docs/getting_started.md](docs/getting_started.md) walks through setting up an
environment, cloning the repository and running it on a synthetic session and on your own data.

## Install

Python 3.10 or newer; numpy, scipy, astropy, plotly, matplotlib and pyyaml.

```sh
pip install -e .            # installs the hi-cube-qa and hi-cube-qa-baseline commands
pip install -e ".[png]"     # kaleido, only for output.save_png
pip install -e ".[test]"    # pytest
```

For conda, `conda env create -f environment.yml` creates an environment with the dependencies.

The minimum versions in `pyproject.toml` are loose and untested. The test suite passes with Python 3.10
and numpy 1.26, scipy 1.15, astropy 5.3, plotly 5.24, matplotlib 3.8, pyyaml 6.0, and in a fresh
environment with Python 3.12 and the newest releases of November 2026 (numpy 2.5, scipy 1.18,
astropy 8.0, plotly 7.1, matplotlib 3.11). Without installing, use
`PYTHONPATH=src python3 -m hi_cube_qa ...`.

## Usage

```sh
hi-cube-qa --config qa_G23_all_sessions.yaml --list       # list the units that would be assessed
hi-cube-qa --config qa_G23_all_sessions.yaml              # full run, plots downsampled
hi-cube-qa --config qa_G23_all_sessions.yaml --full-res   # plot every channel
hi-cube-qa --example-config                               # write ./qa_session_example.yaml
```

On the cluster, `sbatch run_qa.sh [config.yaml]` and `sbatch run_baseline_flatness.sh`. The
scripts run in the directory you submit from, which must contain the checkout including
`src/`; they start the `ASTRO-PY3.12` container with `singularity exec --env PYTHONPATH=...`.
The login node has no Singularity, so test with `sbatch --wrap`.

A unit is one pointing and one `cube_N`. Cubes are found under
`<root>/output_hi_p*/cubes/cube_*/`, and the product (`dirty`, `image`, `residual`, `model`,
`psf`) is the last dot-separated token before `.fits`. The continuum-subtracted cube
`imsub-line.fits`, which only some pointings have, is the product `line`. Products of one unit
should share a WCS and channel grid and are drawn on the same axes; a product with a different
number of channels is left out with a warning.

## Configuration

A YAML file; unspecified keys take the defaults in `src/hi_cube_qa/config.py`.
`qa_session_example.yaml` is the annotated template. The settings that matter most:

| Key | Meaning |
|-----|---------|
| `sessions` | list of `{root: <path to caracal/>}`, one report each |
| `products`, `pointings`, `cubes` | what to load; default `[dirty, image, residual, line]`, all pointings, all cubes |
| `flagreport_glob` | where to find the CARACal flag-summary JSON; `null` disables |
| `radiometer.t_total` | on-source hours, which scales the theoretical noise (default 1.0, so set it) |
| `radiometer.n_ant` | `auto` reads the antenna count from `obsinfo`, or give a number |
| `radiometer.sefd`, `tsys`, `eta`, `sefd_curve` | system sensitivity and its frequency dependence |
| `radiometer.nfac` | weighting penalty on the noise: a number, or `auto` from the Briggs robust value |
| `radiometer.flag_fraction` | `auto` (per pointing from the flag report), a number, or `null` |
| `sensitivity.nhi_dv_kms`, `nsigma` | line width and significance of the quoted N_HI sensitivity |
| `large_scale.*` | excess noise on large angular scales: `products` (`reference` = the line cube, else the residual; `all`; or a list), `sigma_arcsec` (120), `stride` (20), `pb_limit`, `min_channels`, `enable` |
| `baseline.*` | baseline flatness: `degree` (1), `sigma_clip` (4), `enable` |
| `plots.full_resolution`, `plots.max_points` | decimation of plotted traces; the CSVs are always full resolution |
| `plots.initially_visible` | `all` (default) or a list such as `[residual, line]`: the products drawn when a figure opens. The others stay in the legend and a click switches them on |
| `output.dir`, `output.save_png` | output location; PNG export of key figures |

The residual cube, when present, is the noise reference for the noise overview, the
boresight measurement and the 2-D maps.

## Outputs

In `output.dir`:

```
qa_index.html                    open this one; session sidebar
<obsid>_qa_report.html           report per session (keep beside qa_index.html)
<obsid>_channel_qa.csv           one row per channel per product
<obsid>_band_summary.csv         one row per unit and product (with baseline_sigma and ls_* columns)
<obsid>_checklist.csv            checklist results
qa_survey_summary.csv / .json    rollup over all sessions
hi_cube_qa.log
```

The reports load Plotly from a CDN, so the browser needs internet access.

In the per-channel figures the bottom axis is radio velocity in km/s and the top axis is the
channel index. Zooming moves both and the top ticks are refreshed to round channel numbers.
The velocity falls with channel number for an increasing-frequency axis, so channel numbers
then run right to left. The full-band spectrum has velocity below and frequency above. Legends
are to the right of the plots. The layout was checked for overlapping text at widths from 700
to 1400 px; narrower windows squeeze the plots.

## Grading

Per channel, RMS / theory is Good up to 1.5, Warn up to 3 and Bad above (or not finite). A
cube is Bad with 15% Bad channels or RMS/theory above 3, and Warn with 5% Bad channels, 20%
Warn channels or RMS/theory above 1.5. Channels are also graded on Pearson kurtosis (Warn
beyond 5 and Bad beyond 10 standard errors from 3), and an eleven-item checklist gives
Pass/Warn/Fail per unit. Two of its items are checks of their own, run on every cube: baseline
flatness (the fitted trend of the per-channel mean, in standard errors of the mean, for every
product) and large-scale noise (the rms of channel maps smoothed to 120 arcsec against what
neighbouring channels give for noise alone, on one product per unit: the `line` cube when the
pointing has one, otherwise the residual cube). Each has its own figure in the report.
`hi-cube-qa-baseline` runs the baseline check on its own, on any cube you point it at.
These limits are conventions chosen for this toolkit, not published values. The definitions,
and where each rule comes from, are in [docs/qa_classes.md](docs/qa_classes.md).

## Known limitations

- The radiometer expression has not been checked against a primary reference. In one stored
  run the median RMS/theory of 15 residual cubes was 0.64, below the prediction, and the
  cause is not established. Check the expression, the SEFD scale and `nfac` before quoting
  RMS/theory.
- The default `tsys`, `eta`, SEFD curve and `nfac` table have no source here.
- The flagged fraction is one number per pointing, not a per-channel spectrum.
- Beam statistics include zero-beam channels (103 in the G23 GPS band).
- Tested on synthetic cubes and Python 3.10; the cluster container (Python 3.12) passes the
  test suite but real cubes have not been run through the newest figures.

The full list is in [docs/qa_classes.md](docs/qa_classes.md).

## Tests

```sh
python3 tests/run_tests.py          # no pytest needed
python3 -m pytest                   # with pytest
```

The tests use synthetic cubes with known noise, and compare the estimators with astropy
and scipy. On ILIFU, inside the container:
`singularity exec --env PYTHONPATH=$PWD/src /software/astro/containers/ASTRO-PY3.12-latest.sif python3 tests/run_tests.py`.

## Documentation and licence

`docs/` holds the LaTeX reference manual and quickstart (`make` in `docs/`), and
`docs/qa_classes.md`. MIT licence, see `LICENSE`.
