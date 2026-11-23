# QA classes, noise model and open checks

Reference for how hi_cube_qa grades cubes and where each rule comes from. "Own choice"
means a convention of this toolkit with no reference behind it. Only Luber et al. 2025
(ApJ 985, 215) was available for checking against a paper.

## QA class definitions

There are three separate grading schemes: per-channel and cube classes (Good / Warn /
Bad), the checklist (Pass / Warn / Fail), and the baseline flatness check.

### Per-channel class (Good / Warn / Bad)

Reference noise is the per-channel theoretical noise (frequency- and flag-aware), or the
median RMS of the cube if no theory is available. With r = RMS / reference:

| Class | Rule | Config | Source |
|-------|------|--------|--------|
| Good | r <= 1.5 | `thresholds.rms_warn_factor` | own choice |
| Warn | 1.5 < r <= 3 | | own choice |
| Bad | r > 3, or RMS not finite (empty or fully flagged plane) | `thresholds.rms_bad_factor` | own choice |

Channel RMS is the MAD estimate, 1.4826 times the median absolute deviation, which equals
the standard deviation for Gaussian noise (a standard property of the estimator).

### Cube grade (Good / Warn / Bad)

From the counts of channel classes and the band ratio R (median RMS of the Good channels,
all finite channels if none is Good, divided by the median theoretical noise):

| Grade | Rule |
|-------|------|
| Bad | Bad channels >= 15 %, or R > 3 |
| Warn | Bad channels >= 5 %, or Warn channels >= 20 %, or R > 1.5 |
| Good | otherwise |

Source: own choice. These limits are fixed in the code; the config keys
`thresholds.bad_chan_warn_pct` and `bad_chan_bad_pct` exist but are not read.

### Non-Gaussianity (kurtosis)

Per channel, the Pearson kurtosis kappa = <(x-mu)^4> / <(x-mu)^2>^2 from the ordinary mean
and variance; a Gaussian has kappa = 3. A channel is Warn when |kappa - 3| exceeds 5
standard errors and Bad above 10, with SE = sqrt(24 / N_eff) and N_eff the number of
finite pixels in the channel divided by the pixels per beam.

| Element | Source |
|---------|--------|
| Using per-channel kurtosis, with 3 as the Gaussian expectation, to check cube quality | Luber et al. 2025, ApJ 985, 215, p. 6 and Fig. 3 (checked in the PDF) |
| SE = sqrt(24 / N) for N independent samples | standard result for Gaussian data; checked by simulation (standard deviation of kappa 0.1533 for N = 1000 against 0.1549; 0.04897 against 0.04899 for N = 10000) |
| Dividing N by the pixels per beam | own reasoning (neighbouring pixels are correlated); not checked against a reference |
| 5 and 10 standard errors | own choice (`thresholds.kurtosis_warn_sigma`, `kurtosis_bad_sigma`). Luber et al. quote an expected kurtosis variation of 0.004 for their data but give no threshold |

### Expert checklist (Pass / Warn / Fail)

Nine criteria per unit and product, in `<obsid>_checklist.csv`. All limits are own
choices; only the beam-scatter limit is read from the config.

| Criterion | Pass | Warn | Fail |
|-----------|------|------|------|
| Noise within 1.5x theory (R) | < 1.5 | < 3 | otherwise |
| Few bad channels | < 5 % | < 15 % | otherwise |
| No catastrophic channels | < 15 % | | otherwise |
| Noise floor convergence (median RMS of last 10 % of channels / first 90 %) | < 1.2 | < 2 | otherwise |
| Beam stability (std of BMAJ) | < 0.5" (`beam_scatter_warn`) | otherwise | |
| Axial ratio BMAJ/BMIN | < 2 | < 3 | otherwise |
| No severe outliers (channels above 5x median RMS) | 0 | < 5 | otherwise |
| Low continuum residual (% pixels above `continuum.sigma` in more than 50 % of channels) | < 0.5 % | < 2 % | otherwise |
| Gaussian noise (% of channels Bad on kurtosis) | < 5 % | < 15 % | otherwise |

### Flagged fraction (session figure)

Flag-summary entries are drawn in the warning colour above 10 % and the severe colour
above 30 %. Own choice; fixed in the code (the config keys `flag_warn_pct` and
`flag_bad_pct` are not read).

### Baseline flatness (`hi-cube-qa-baseline`)

A separate command for continuum-subtracted cubes (default pattern `imsub-line.fits`).
It fits a polynomial (default degree 1) to the per-channel mean against velocity. Outlier
channels are rejected with astropy's `FittingWithOutlierRemoval` (`sigma_clip` about the
median, sigma from the MAD, 4 sigma, 3 iterations); the trend is then an ordinary
least-squares fit to the retained channels. If rejection would leave fewer than degree + 2
channels, none is rejected. The peak-to-peak amplitude of the fit is expressed in standard
errors of the channel mean. The standard error
is the per-channel MAD RMS divided by the square root of the number of beams in the image.
Pass below 5, Warn below 10, Fail otherwise (`--warn-sigma`, `--bad-sigma`). Source: own
choice for all of it; the 5 and 10 follow the kurtosis convention above.

## Noise model and sensitivity

| Quantity | Expression in the code | Status |
|----------|------------------------|--------|
| Radiometer noise | nfac SEFD / sqrt(n_bl t_eff dnu n_pol), n_bl = N(N-1)/2, n_pol = 2, t_eff = t_total (1 - f_flag) | **not checked against a primary reference**, see limitations |
| SEFD from T_sys | 2 k T_sys / (eta pi (d/2)^2) with d = 13.5 m | arithmetic checked (602.8 Jy for 25 K and 0.80); the defaults 25 K and 0.80 have no source here |
| Frequency dependence of SEFD | built-in curve (405 to 660 Jy between 900 and 1670 MHz), rescaled so that its value at the median channel frequency equals the T_sys-derived SEFD | the curve is labelled illustrative in the code; it is not an official SARAO calibration |
| Theory from frequency-dependent SEFD and the number of unflagged visibilities | approach of Luber et al. 2025 (p. 6) | checked in the PDF. Here the flagged fraction is one scalar per pointing, not per channel |
| Weighting penalty nfac | table by Briggs robust: -2 to 1.40, -1 to 1.35, -0.5 to 1.20, 0 to 1.10, 0.5 to 1.04, 1 to 1.02, 2 to 1.00, interpolated | table is described in the code as typical values, with no source. Luber et al. 2025 report an effect below about 5 % on the noise for robust 0.5 |
| Brightness temperature | T_B = 1.222e3 S[mJy/beam] / (nu[GHz]^2 bmaj["] bmin["]) | coefficient recomputed from the Rayleigh-Jeans law for a Gaussian beam: 1222.1 |
| N_HI sensitivity | 1.823e18 nsigma T_B(sigma) sqrt(dv_chan dv) cm^-2, for N = dv/dv_chan independent channels | the coefficient is the optically thin conversion; recomputed from A10 = 2.869e-15 s^-1 (value not verified against a source) it is 1.8225e18, and 1.8127e18 with 2.8843e-15. It assumes spin temperature far above 0.07 K and no self-absorption |

## Limitations and open checks

- Radiometer expression. The comparison between measured RMS and theory has not been
  verified against Thompson, Moran and Swenson or another primary source (not available
  in this repository). In the stored `qa_output_repaired` run the median RMS/theory of the
  15 residual cubes is 0.64 (range 0.60 to 1.18), i.e. the noise is well below the prediction. Why has not
  been established; the noise also differs by a factor 2 to 3 between older and newer
  sessions. Check the expression, the SEFD scale (default T_sys and eta give about 600 Jy,
  while the built-in curve itself lies between 405 and 660 Jy) and nfac before quoting RMS/theory
  numbers.
- Default `tsys`, `eta`, the SEFD curve and the nfac table are unsourced (see above); set
  `radiometer.sefd` or `sefd_curve` from measured values for quantitative work.
- The flagged fraction is a single number per pointing, not a per-channel flag spectrum.
  Only the per-pointing target flag summaries are used (calibrator flags are ignored);
  pointings without a report get the session median.
- All class thresholds are conventions of this toolkit, not validated limits. Luber et al.
  2025 treat a deviation of 30 % in their RMS ratio as significant; that is their choice
  for CHILES data.
- Four config thresholds (`bad_chan_warn_pct`, `bad_chan_bad_pct`, `flag_warn_pct`,
  `flag_bad_pct`) are accepted but fixed values are used in the code.
- The number of antennas read from `obsinfo` is the antenna table length, i.e. antennas
  present, not antennas that survived calibration and flagging.
- Beam statistics include every channel whose beam keywords are finite, so channels with
  a zero beam keyword are not excluded and enter the scatter and axial-ratio checks. In the
  G23 cubes this applies to 103 channels in the GPS band (1379.7 to 1382.4 MHz); excluding
  them is a planned change that will alter reported numbers.
- The in-page refresh of the top (channel) axis assumes velocity is linear in channel
  number, as it is for a frequency axis; otherwise the top axis keeps the ticks of the full
  band when zooming.
- Spectral axes are handled for `FREQ`, `VELO`, `VRAD` (radio convention) and `VOPT`; other
  axis types fall back to a dummy frequency grid with a warning.
- Only tested on synthetic cubes and with Python 3.10; the ILIFU container (Python 3.12)
  has not been tested.

## Field notes (G23 data, October 2026)

Observations from the processing of the G23 sessions; they are not derived or tested by
this repository.

- Use `residual`, or compare against `dirty`, to catch deconvolution failures. In
  isolated channels of the second-pass (SoFiA-masked) `cube_2` of several pointings the
  cleaned products had an RMS 100 to 6000 times too high, while `dirty` and the first-pass
  `cube_1` were normal. A channel that is high in `dirty` too is a data problem. A worked
  example is in the reference manual (`docs/`).
- Reuse by `hi_mosaic`. `hi_mosaic/noise_table.py` imports `beam_stats`,
  `frequency_velocity` and `mad_rms_planes` from the QA folder through a `--qa-dir` path that
  assumes the old flat layout; with the package layout the modules live in `src/hi_cube_qa`
  and are imported as `hi_cube_qa.beam` and so on.

