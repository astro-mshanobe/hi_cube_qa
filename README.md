# hi_cube_qa — MeerKAT HI cube quality assessment (session-oriented)

Produces one self-contained interactive **HTML QA report per observation
session**. Cubes are auto-discovered under each CARACal session root; you only
choose which imaging products to plot.

## How it's organised

```
session (e.g. 1610268336_sdp_l0)  -->  one HTML report
  pointing p63
    cube_1  -->  unit block: dirty / image / residual overlaid (shared axes)
    cube_2  -->  unit block: ...
  pointing p64
    ...
  + a session-level flagging summary
```

A **unit** is one `(pointing, cube_N)` pair. Products inside the same `cube_N`
share a WCS and channel grid, so they are compared on the same axes; different
`cube_N` (different spectral windows) are kept separate.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point: loops sessions → units, writes one report each + master index + survey rollup. |
| `discovery.py` | Walks `output_hi_p*/cubes/cube_*/` and groups cubes by product. |
| `config.py` | YAML schema, defaults, validation, example-config writer. |
| `io_utils.py` | Spectral-axis (FREQ/VELO/VOPT/VRAD) + flag-report helpers. |
| `reduce.py` | Memory-mapped, vectorised single-pass cube reducer (1-D spectra, kurtosis, 2-D maps, planes). |
| `noise.py` | MAD RMS, frequency-/flagging-aware radiometer noise, N_HI sensitivity, QA classes. |
| `beam.py` | Beam stats (global + per-channel WSClean keywords) + diagnostics. |
| `provenance.py` | Reads imaging provenance (robust/niter/beam) from headers; auto `nfac`. |
| `wsclean_log.py` | Reads the real per-session antenna count from CARACal's `obsinfo`. |
| `spectrum.py` | All Plotly figures (RMS, theory curve, noise overview, spectra, N_HI, beam, flags). |
| `thumbnails.py` | Diagnostic channel maps + 2-D noise/continuum maps. |
| `tables.py` | CSV export, logger, expert checklist, survey rollup. |
| `report.py` | Per-session HTML (provenance + unit blocks) and the master tab index. |
| `qa_G23_all_sessions.yaml` | Ready-to-run config for all five G23 sessions. |
| `qa_session_example.yaml` | Annotated template (regenerate: `--example-config`). |
| `run_qa.sh` | Slurm submission script. |
| `docs/` | LaTeX documentation (science justification + usage); see `docs/README.md`. |

## Requirements

Python 3 with `numpy`, `scipy`, `astropy`, `plotly`, `matplotlib`, `pyyaml`.
All present in the ILIFU `ASTRO-PY3.12` container used by `run_qa.sh`.

## Usage

```sh
# Preview what will be assessed (fast, no heavy compute):
python3 main.py --config qa_G23_all_sessions.yaml --list

# Full run (local); plots are downsampled by default:
python3 main.py --config qa_G23_all_sessions.yaml
# Full-resolution plots (every channel) for a deep dive:
python3 main.py --config qa_G23_all_sessions.yaml --full-res

# On the cluster:
sbatch run_qa.sh                      # uses qa_G23_all_sessions.yaml
sbatch run_qa.sh my_config.yaml       # custom config

# Regenerate the annotated example config:
python3 main.py --example-config
```

Outputs land in `output.dir`:

```
qa_index.html               <- MASTER: open this one; tab through every session
qa_survey_summary.csv/.json <- rollup of every cube across ALL sessions
<obsid>_qa_report.html      (one interactive report per session)
<obsid>_channel_qa.csv      (one row per channel, per product)
<obsid>_band_summary.csv    (one row per unit/product; incl. N_HI + continuum)
<obsid>_checklist.csv       (expert QA criteria)
hi_cube_qa.log
```

`qa_index.html` is a small shell page with a session sidebar; clicking a session
loads its report on demand (in a frame), so it stays fast even with many
sessions. The per-session `<obsid>_qa_report.html` files must stay in the same
directory as `qa_index.html` (they are referenced by relative name).

## Key config knobs

| Knob | Meaning |
|------|---------|
| `sessions` | List of CARACal `caracal/` roots (one report each). |
| `products` | `[dirty, image, residual]` by default; or `all`; or any subset. |
| `pointings` | `null` = all; or a subset like `[p63, p64]`. |
| `cubes` | `null`/`all` = every `cube_N`; or a subset. Accepts `cube_1`, `c1`, or `1`. |
| `thumbnails.selectors` | `[rms, snr]` (drop one to show fewer thumbnails). |
| `flagreport_glob` | Glob(s) (relative to each session root) for flag JSONs; `null` disables. Auto-discovered per session with fallbacks if the layout differs. |
| `radiometer.t_total` | On-source **hours** — set to the real track length; it scales the theoretical-noise line. `df` is auto-read from each header. |
| `radiometer.n_ant` | Number of antennas: `auto` (default) reads the real per-session count from `obsinfo`, since sessions can differ (e.g. antennas offline); falls back to `n_ant_fallback` (60) if unavailable, or set a fixed number. |
| `radiometer.nfac` | Imaging-weight penalty (≈1.7 for Briggs robust = −0.5), or `auto` (read robust from header). |
| `radiometer.sefd_curve` | `default` frequency-dependent SEFD curve, `flat`, or `[[MHz,Jy],…]`. |
| `radiometer.flag_fraction` | Fold flagging into the theory: `auto` (from flag report), a number, or `null`. |
| `sensitivity.nhi_dv_kms` / `nsigma` | Line width and σ for the reported N_HI column-density sensitivity. |
| `continuum.sigma` | Per-channel threshold for the continuum-persistence map. |
| `thresholds.kurtosis_warn_sigma` / `kurtosis_bad_sigma` | Per-channel non-Gaussianity thresholds (standard errors from the Gaussian expectation of 3). |
| `reduce.chunk` / `maps` | Channels per I/O block (memory/speed); whether to render 2-D maps. |
| `plots.full_resolution` | `false` = downsampled plots (default); `true` = every channel. Override per-run with `--full-res` / `--downsample`. |
| `plots.max_points` | Points per trace when downsampling (default 2000). CSVs stay full-res; Bad channels always kept. |
| `output.save_png` / `png_figures` | Also save key figures as PNGs under `pngs/<obsid>/` (needs `kaleido`). |

The `residual` product (when present) is the noise reference for the noise
overview, boresight, and 2-D maps, since it is the cleanest noise tracer.

**Noise model:** the per-channel theory is frequency-dependent (SEFD curve) and
flagging-aware (effective time reduced by the flagged fraction), so the
RMS/theory comparison is fair at band edges and in RFI-heavy sub-bands. See
`docs/` for the full justification.

> **Note:** the report embeds channel-map PNGs and loads Plotly from CDN (needs
> internet in the *browser*, not on the compute node).

## First run on real data — checklist

1. `--list` first to confirm residuals (and any `model`/`psf`) are discovered.
2. Set `radiometer.t_total` to each session's real on-source time.
3. See `docs/` for the full scientific justification of every metric.
