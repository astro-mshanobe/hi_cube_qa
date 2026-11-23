# Getting started

How to set up hi_cube_qa on a laptop or a cluster and run it on your own cubes.

## What you need

- Python 3.10 or newer and git.
- Imaging cubes in the CARACal layout, one folder per observation session:

```
<session>/caracal/
    output_hi_<pXX>/cubes/cube_1/*.<product>.fits    products: dirty, image, residual, ...
    output_hi_<pXX>/cubes/cube_2/*.<product>.fits
    output_hi_<pXX>/cubes/cube_N/imsub-line.fits    continuum-subtracted cube, if present (product 'line')
    output_hi_1gc/diagnostic_plots/*-flagging-summary-*.json    optional, for the flagged fractions
    output_hi_1gc/obsinfo/*-summary.json                        optional, for the antenna count
```

A `*.pb.fits` cube next to a cube is used for a primary-beam cut in the large-scale noise check.
Without the optional files the tool still runs and the log says what it had to assume.

## 1. Clone the repository

```sh
git clone <repository URL>
cd hi_cube_qa
```

## 2. Create an environment

Pick one. All three need the repository you just cloned.

With conda or mamba:

```sh
conda env create -f environment.yml
conda activate hi_cube_qa
pip install -e .
```

With a plain virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On a cluster that already offers a container or module with numpy, scipy, astropy, plotly,
matplotlib and pyyaml, nothing needs installing. Put `src/` on the path instead and run the
same commands as `python3 -m hi_cube_qa`:

```sh
export PYTHONPATH=$PWD/src
```

`pip install -e .` provides the commands `hi-cube-qa` and `hi-cube-qa-baseline`.
To export figures as PNG as well, add `pip install kaleido`.

## 3. Check the installation

```sh
hi-cube-qa --help
python3 tests/run_tests.py
```

The tests use synthetic cubes and take a minute or two. The last line should read
`0 failure(s)`.

## 4. Try it on synthetic data

This needs no data of your own. It writes a small fake session under `/tmp/qa_demo`, runs the
tool on it and opens the result:

```sh
PYTHONPATH=src:tests python3 -c "import synthetic, pathlib; synthetic.make_session(pathlib.Path('/tmp/qa_demo'), nchan=100, ny=256, nx=256)"
printf 'sessions:\n  - root: /tmp/qa_demo/1600000000_sdp_l0/caracal\noutput:\n  dir: /tmp/qa_demo/out\nlarge_scale:\n  sigma_arcsec: 60.0\n  stride: 10\n' > /tmp/qa_demo/demo.yaml
hi-cube-qa --config /tmp/qa_demo/demo.yaml
open /tmp/qa_demo/out/qa_index.html
```

(`open` is the macOS command; use `xdg-open` on Linux.) Without an installed package, use
`PYTHONPATH=src python3 -m hi_cube_qa` instead of `hi-cube-qa`. The synthetic cubes are pure
noise, so everything grades Good or Pass.

## 5. Run it on your own data

Write a configuration from the template and edit it:

```sh
hi-cube-qa --example-config
```

This writes `qa_session_example.yaml` into the current folder. The settings you will
normally change:

- `sessions`: one `root:` line per session, pointing at its `caracal/` folder.
- `radiometer.t_total`: on-source hours of the track. The theoretical noise scales with it,
  and the default of 1 hour is rarely right.
- `output.dir`: where the reports go.
- `pointings` and `cubes`: start with one pointing, for example `pointings: [p63]`.

Check what will be assessed, then run:

```sh
hi-cube-qa --config my_config.yaml --list
hi-cube-qa --config my_config.yaml
```

Open `qa_index.html` in the output folder. The page loads Plotly from the internet, so the
browser, though not the machine that ran the job, needs a connection. Keep the
`<obsid>_qa_report.html` files next to it.

### Running as a batch job

Copy `run_qa.sh` and edit the partition, the resource lines and the `CONTAINER` path for your
site (or delete the `singularity` wrapper if your environment is already active). Submit it from
the repository folder with the name of your configuration:

```sh
sbatch run_qa.sh my_config.yaml
```

A full run reads every cube once. The large-scale noise check smooths about one channel in
twenty of one product per unit (the continuum-subtracted `imsub-line.fits` where it exists, else
the residual) as well, which adds to the time on large cubes.

## If something is missing

- `--list` shows `0 unit(s)`: the folder in `root:` is not the `caracal/` folder, or the cubes
  are not under `output_hi_p*/cubes/cube_*/`. An error saying the root "is not a directory"
  means the path itself is wrong.
- Large-scale noise says "not assessed": the image is too small for the smoothing scale (it
  needs about 450 pixels across for a 20-pixel sigma), or too few of the sampled channels have
  data. Lower `large_scale.sigma_arcsec` or `large_scale.stride`.
- Log lines saying that no flag reports or no obsinfo summary were found: the optional files
  are missing, so the theory is not reduced for flagging and the antenna count falls back to
  `n_ant_fallback` (60).
- Ratios far from 1 against the radiometer theory: check `t_total` and `n_ant` first. The
  [limitations](qa_classes.md) list what has not been verified.

The meaning of each grade is in [qa_classes.md](qa_classes.md), and the full list of settings
is in `qa_session_example.yaml`.
