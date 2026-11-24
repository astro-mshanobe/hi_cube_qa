"""Noise estimators, radiometer model, and HI sensitivity.

This module holds the *physics*: robust RMS, the (frequency- and flagging-
aware) radiometer-equation prediction, and the conversion of noise to an
HI column-density sensitivity. The heavy per-cube measurement loop lives in
reduction.py; the small helpers here are also used directly by it.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import median_abs_deviation

# Constants
GOOD = "Good"
WARN = "Warn"
BAD = "Bad"

K_B = 1.380649e-23  # J/K
MEERKAT_DISH = 13.5  # m
JY = 1e26  # W m^-2 Hz^-1 per Jy
HI_REST_HZ = 1.420405752e9  # Hz
C_KMS = 2.99792458e5  # km/s

# Approximate MeerKAT L-band SEFD vs frequency, per antenna [MHz, Jy].
# These are ILLUSTRATIVE values capturing the shape of the band (a broad
# minimum near 1.3 GHz, rising wings towards both edges); they are NOT the
# official SARAO numbers. Override with radiometer.sefd_curve in the config for
# quantitative work. See docs (Frequency-dependent SEFD).
DEFAULT_SEFD_CURVE = [
    (900.0, 560.0),
    (1000.0, 460.0),
    (1100.0, 430.0),
    (1200.0, 410.0),
    (1300.0, 405.0),
    (1400.0, 420.0),
    (1500.0, 460.0),
    (1600.0, 540.0),
    (1670.0, 660.0),
]


# Robust RMS helpers
def mad_rms(arr: np.ndarray) -> float:
    """Robust RMS via MAD: sigma = 1.4826 * MAD (Gaussian equivalent)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return 1.4826 * float(median_abs_deviation(finite, nan_policy="omit"))


def mad_rms_planes(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised per-plane median and MAD-RMS for a (nchan, ny, nx) block.

    NaNs are ignored. Returns (median, sigma_mad), each shape (nchan,).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN planes
        med = np.nanmedian(block, axis=(1, 2))
        dev = np.abs(block - med[:, None, None])
        sig = 1.4826 * np.nanmedian(dev, axis=(1, 2))
    return med, sig


def kurtosis_planes(block: np.ndarray) -> np.ndarray:
    """Vectorised per-plane Pearson kurtosis for a (nchan, ny, nx) block.

    Uses the ordinary mean and variance (NOT the robust MAD statistics used
    elsewhere in this module) - kurtosis exists specifically to measure
    sensitivity to outliers/tails, so a robust, outlier-resistant estimator
    would defeat its purpose. This is the biased/simple estimator
    kappa = E[(x-mu)^4] / E[(x-mu)^2]^2, for which a Gaussian distribution has
    expectation 3 (Pearson's convention, matching Luber et al. 2025 / CHILES).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN planes
        mean = np.nanmean(block, axis=(1, 2))
        dev = block - mean[:, None, None]
        m2 = np.nanmean(dev**2, axis=(1, 2))
        m4 = np.nanmean(dev**4, axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = m4 / m2**2
    return kurt


def kurtosis_expected_se(n_eff: np.ndarray | float) -> np.ndarray | float:
    """Expected standard error of the Pearson kurtosis estimator under
    normality, for *n_eff* independent samples: SE = sqrt(24 / n_eff).

    *n_eff* should be the number of independent RESOLUTION ELEMENTS (pixels
    per beam), not the raw pixel count - neighbouring pixels within a
    synthesised beam are correlated, so using the raw pixel count
    underestimates the true sampling uncertainty and makes normal channels
    look spuriously significant.
    """
    n_eff = np.asarray(n_eff, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(24.0 / np.clip(n_eff, 1.0, None))


def classify_kurtosis(
    kurt: np.ndarray,
    n_eff: np.ndarray | float,
    warn_sigma: float = 5.0,
    bad_sigma: float = 10.0,
) -> np.ndarray:
    """Label each channel Good/Warn/Bad from how many standard errors its
    kurtosis deviates from the Gaussian expectation of 3.
    """
    kurt = np.asarray(kurt, dtype=float)
    nchan = len(kurt)
    se = kurtosis_expected_se(n_eff)
    se = np.broadcast_to(se, (nchan,)) if np.ndim(se) == 0 else se
    quality = np.full(nchan, GOOD, dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_sigma = np.abs(kurt - 3.0) / se
    for i in range(nchan):
        if not np.isfinite(n_sigma[i]):
            continue
        if n_sigma[i] > bad_sigma:
            quality[i] = BAD
        elif n_sigma[i] > warn_sigma:
            quality[i] = WARN
    return quality


def peak_channel_rms(data: np.ndarray) -> np.ndarray:
    """Per-channel max |pixel| -- proxy for worst-case artefacts/RFI."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.abs(np.asarray(data)), axis=(1, 2))


def boresight_rms(
    data: np.ndarray,
    hdr,
    box_size: int = 100,
    centre_pix: tuple = None,
    sigma_clip: float = 3.0,
    clip_iterations: int = 3,
) -> np.ndarray:
    """Per-channel MAD-RMS in a box at the pointing centre (best-case noise).

    Measures noise at boresight where the primary-beam response is highest,
    after iterative MAD sigma-clipping to reject source pixels. Uses the MAD
    estimator throughout for consistency with mad_rms_planes.
    """
    data = np.asarray(data)
    nchan, ny, nx = data.shape

    if centre_pix is None:
        cy = int(hdr.get("CRPIX2", ny // 2 + 1)) - 1
        cx = int(hdr.get("CRPIX1", nx // 2 + 1)) - 1
    else:
        cy, cx = centre_pix

    half = box_size // 2
    y0, y1 = max(0, cy - half), min(ny, cy + half)
    x0, x1 = max(0, cx - half), min(nx, cx + half)

    rms = np.full(nchan, np.nan)
    for i in range(nchan):
        box = data[i, y0:y1, x0:x1].astype(np.float64)
        mask = np.isfinite(box)
        for _ in range(clip_iterations):
            if mask.sum() < 10:
                break
            valid = box[mask]
            med = np.median(valid)
            s = 1.4826 * np.median(np.abs(valid - med))
            if s == 0:
                break
            mask = mask & (np.abs(box - med) < sigma_clip * s)
        valid = box[mask]
        if valid.size:
            med = np.median(valid)
            rms[i] = 1.4826 * np.median(np.abs(valid - med))
    return rms


# Radiometer model (frequency- and flagging-aware)
def sefd_from_tsys(tsys: float, eta: float) -> float:
    """SEFD [Jy] from system temperature and aperture efficiency (single dish)."""
    a_geo = np.pi * (MEERKAT_DISH / 2.0) ** 2
    return (2.0 * K_B * tsys) / (eta * a_geo) * JY


def sefd_spectrum(
    freq_hz: np.ndarray,
    sefd: float | None = None,
    tsys: float = 25.0,
    eta: float = 0.80,
    curve: list | None = None,
) -> np.ndarray:
    """Per-channel SEFD [Jy] across the band.

    Priority:
      1. *sefd* given (scalar)            -> flat SEFD (frequency-independent).
      2. *curve* given or default curve   -> interpolate SEFD(nu), scaled so its
         band-centre value matches the tsys/eta SEFD (keeps the user's tsys as
         the normalisation while adopting the curve's *shape*).
      3. else                             -> flat tsys/eta SEFD.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    if sefd is not None and sefd > 0:
        return np.full(freq_hz.shape, float(sefd))

    sefd_flat = sefd_from_tsys(tsys, eta)
    if curve is None:
        curve = DEFAULT_SEFD_CURVE
    if not curve:
        return np.full(freq_hz.shape, sefd_flat)

    cf = np.array([c[0] for c in curve], dtype=float) * 1e6  # MHz -> Hz
    cs = np.array([c[1] for c in curve], dtype=float)
    order = np.argsort(cf)
    cf, cs = cf[order], cs[order]

    shape = np.interp(freq_hz, cf, cs)  # SEFD shape in Jy
    # Normalise the shape so its value at the band centre equals sefd_flat,
    # i.e. adopt the *shape* of the curve but the user's tsys as the scale.
    f_mid = float(np.median(freq_hz))
    s_mid = float(np.interp(f_mid, cf, cs))
    if s_mid > 0:
        shape = shape * (sefd_flat / s_mid)
    return shape


def theoretical_noise_spectrum(
    freq_hz: np.ndarray,
    sefd: float | None = None,
    tsys: float = 25.0,
    eta: float = 0.80,
    n_ant: int = 60,
    t_total_hr: float = 1.0,
    df_hz: float = 26367.19,
    n_pol: int = 2,
    nfac: float = 1.0,
    flag_fraction: float | np.ndarray = 0.0,
    curve: list | None = None,
) -> np.ndarray:
    """Per-channel radiometer noise [Jy/beam] across the band.

    sigma(nu) = nfac * SEFD(nu) / sqrt( n_bl * t_eff(nu) * dnu * n_pol )

    with t_eff(nu) = t_total * (1 - f_flag(nu)). The flagged fraction reduces
    the effective on-source time, so heavily-flagged sub-bands are *expected*
    to be noisier - this is what makes the per-channel RMS/theory comparison
    fair. *flag_fraction* may be a scalar or a per-channel array in [0, 1).
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    sefd_nu = sefd_spectrum(freq_hz, sefd=sefd, tsys=tsys, eta=eta, curve=curve)

    n_bl = n_ant * (n_ant - 1) / 2.0
    t_s = t_total_hr * 3600.0
    fflag = np.clip(np.asarray(flag_fraction, dtype=float), 0.0, 0.999)
    t_eff = t_s * (1.0 - fflag)

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = nfac * sefd_nu / np.sqrt(n_bl * t_eff * df_hz * n_pol)
    return sigma


def theoretical_noise(
    sefd: float | None = None,
    tsys: float = 25.0,
    eta: float = 0.80,
    n_ant: int = 60,
    t_total_hr: float = 1.0,
    df_hz: float = 26367.19,
    n_pol: int = 2,
    nfac: float = 1.0,
) -> float:
    """Band-centre scalar radiometer noise [Jy/beam] (flat SEFD from tsys/eta or a
    supplied SEFD). Retained for band-average summaries and logging.
    """
    if sefd is not None and sefd > 0:
        sefd_jy = float(sefd)
    else:
        sefd_jy = sefd_from_tsys(tsys, eta)
    n_bl = n_ant * (n_ant - 1) / 2.0
    t_s = t_total_hr * 3600.0
    return float(nfac * sefd_jy / np.sqrt(n_bl * t_s * df_hz * n_pol))


# HI column-density sensitivity
def brightness_temperature(
    s_jy_beam: np.ndarray | float,
    freq_hz: np.ndarray | float,
    bmaj_asec: float,
    bmin_asec: float,
) -> np.ndarray | float:
    """Convert flux density per beam [Jy/beam] to brightness temperature [K]:

        T_B = 1.222e3 * S[mJy/beam] / ( nu[GHz]^2 * bmaj["] * bmin["] )

    (the standard Rayleigh-Jeans, Gaussian-beam relation).
    """
    s_mjy = np.asarray(s_jy_beam, dtype=float) * 1e3
    nu_g = np.asarray(freq_hz, dtype=float) / 1e9
    if bmaj_asec <= 0 or bmin_asec <= 0:
        return np.nan if np.isscalar(s_jy_beam) else np.full_like(s_mjy, np.nan)
    return 1.222e3 * s_mjy / (nu_g**2 * bmaj_asec * bmin_asec)


def nhi_sensitivity(
    sigma_jy_beam: float,
    freq_hz: float,
    bmaj_asec: float,
    bmin_asec: float,
    chan_dv_kms: float,
    dv_kms: float = 20.0,
    nsigma: float = 3.0,
) -> dict:
    """HI column-density sensitivity from the per-channel noise.

    Per channel:   sigma_NHI(chan) = 1.823e18 * nsigma * T_B(sigma) * dv_chan
    Over a line of width dv (N = dv/dv_chan independent channels):
        sigma_NHI(dv) = 1.823e18 * nsigma * T_B(sigma) * sqrt(dv_chan * dv)

    Returns dict with 'tb_sigma_K', 'nhi_chan_cm2', 'nhi_dv_cm2', 'dv_kms'.
    All in atoms cm^-2. Uses |chan_dv_kms| (sign-independent).
    """
    chan_dv = abs(chan_dv_kms)
    tb = brightness_temperature(sigma_jy_beam, freq_hz, bmaj_asec, bmin_asec)
    if not np.isfinite(tb):
        return {
            "tb_sigma_K": np.nan,
            "nhi_chan_cm2": np.nan,
            "nhi_dv_cm2": np.nan,
            "dv_kms": dv_kms,
        }
    nhi_chan = 1.823e18 * nsigma * tb * chan_dv
    nhi_dv = 1.823e18 * nsigma * tb * np.sqrt(chan_dv * dv_kms)
    return {
        "tb_sigma_K": float(tb),
        "nhi_chan_cm2": float(nhi_chan),
        "nhi_dv_cm2": float(nhi_dv),
        "dv_kms": float(dv_kms),
    }


# Spectral running statistics (used by the figures)
def running_median_rms(rms: np.ndarray, window: int = 20) -> np.ndarray:
    """Centred running median of the per-channel RMS (truncated at edges)."""
    n = len(rms)
    out = np.full(n, np.nan)
    hw = window // 2
    for i in range(n):
        lo, hi = max(0, i - hw), min(n, i + hw + 1)
        valid = rms[lo:hi]
        valid = valid[np.isfinite(valid)]
        if valid.size:
            out[i] = np.nanmedian(valid)
    return out


def cumulative_rms(rms: np.ndarray) -> np.ndarray:
    """Running cumulative median RMS from channel 0 to i (floor convergence)."""
    n = len(rms)
    out = np.full(n, np.nan)
    for i in range(n):
        valid = rms[: i + 1]
        valid = valid[np.isfinite(valid)]
        if valid.size:
            out[i] = np.nanmedian(valid)
    return out


# Channel quality classification
def classify_channels(
    rms: np.ndarray,
    theory_rms: float | np.ndarray | None,
    warn_factor: float = 1.5,
    bad_factor: float = 3.0,
) -> np.ndarray:
    """Label each channel Good/Warn/Bad from RMS / reference.

    *theory_rms* may be a scalar or a per-channel array (frequency- and
    flagging-aware). If unavailable, the band-median RMS is used as the
    reference so the tool still works without radiometer inputs.
    """
    rms = np.asarray(rms, dtype=float)
    nchan = len(rms)
    quality = np.full(nchan, GOOD, dtype=object)

    theory_arr = None
    if theory_rms is not None:
        ta = np.asarray(theory_rms, dtype=float)
        if ta.ndim == 0:
            if np.isfinite(ta) and ta > 0:
                theory_arr = np.full(nchan, float(ta))
        elif ta.shape[0] == nchan:
            theory_arr = ta

    if theory_arr is None:
        med = np.nanmedian(rms[np.isfinite(rms)]) if np.any(np.isfinite(rms)) else np.nan
        if np.isfinite(med) and med > 0:
            theory_arr = np.full(nchan, med)

    for i in range(nchan):
        r = rms[i]
        if not np.isfinite(r):
            quality[i] = BAD
            continue
        if theory_arr is not None and np.isfinite(theory_arr[i]) and theory_arr[i] > 0:
            ratio = r / theory_arr[i]
            if ratio > bad_factor:
                quality[i] = BAD
            elif ratio > warn_factor:
                quality[i] = WARN
    return quality


def channel_summary(
    rms: np.ndarray,
    quality: np.ndarray,
    theory: float | np.ndarray | None,
    exclude: np.ndarray | None = None,
    limits: dict | None = None,
) -> dict:
    """Aggregate per-channel results into a band-level summary.

    Channels flagged in *exclude* (for example those with no beam, which carry no data) are
    left out of the counts, the percentages and the rms statistics; they stay in n_total.
    *limits* may hold rms_warn_factor, rms_bad_factor, bad_chan_warn_pct, bad_chan_bad_pct and
    warn_chan_warn_pct, which set the cube grade.
    """
    limits = limits or {}
    rms = np.asarray(rms, dtype=float)
    quality = np.asarray(quality, dtype=object)
    n_total = len(rms)
    keep = np.ones(n_total, dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)
    rms, quality = rms[keep], quality[keep]
    n_graded = len(rms)
    n_good = int(np.sum(quality == GOOD))
    n_warn = int(np.sum(quality == WARN))
    n_bad = int(np.sum(quality == BAD))
    pct_bad = n_bad / n_graded * 100 if n_graded else 0.0
    pct_warn = n_warn / n_graded * 100 if n_graded else 0.0

    # Headline "median RMS": the clean-channel noise floor when Good channels
    # exist, else fall back to all finite channels so a uniformly noisy (but
    # perfectly valid) cube reports its real median rather than NaN.
    fin = rms[np.isfinite(rms)]
    good_rms = rms[(quality == GOOD) & np.isfinite(rms)]
    if good_rms.size:
        med_rms = float(np.nanmedian(good_rms))
    elif fin.size:
        med_rms = float(np.nanmedian(fin))
    else:
        med_rms = np.nan
    min_rms = float(np.nanmin(fin)) if fin.size else np.nan
    max_rms = float(np.nanmax(fin)) if fin.size else np.nan
    std_rms = float(np.nanstd(fin)) if fin.size else np.nan

    # Band-average theory: scalar, or median of the per-channel theory array.
    if theory is None:
        theory_scalar = None
    else:
        ta = np.asarray(theory, dtype=float)
        theory_scalar = float(ta) if ta.ndim == 0 else float(np.nanmedian(ta))

    ratio = (
        (med_rms / theory_scalar)
        if (theory_scalar and theory_scalar > 0 and np.isfinite(med_rms))
        else None
    )

    bad_ratio = limits.get("rms_bad_factor", 3.0)
    warn_ratio = limits.get("rms_warn_factor", 1.5)
    if pct_bad >= limits.get("bad_chan_bad_pct", 15.0) or (ratio is not None and ratio > bad_ratio):
        overall = BAD
    elif (
        pct_bad >= limits.get("bad_chan_warn_pct", 5.0)
        or pct_warn >= limits.get("warn_chan_warn_pct", 20.0)
        or (ratio is not None and ratio > warn_ratio)
    ):
        overall = WARN
    else:
        overall = GOOD

    return {
        "median_rms_mjy": med_rms * 1e3,
        "min_rms_mjy": min_rms * 1e3,
        "max_rms_mjy": max_rms * 1e3,
        "std_rms_mjy": std_rms * 1e3,
        "theory_rms_mjy": theory_scalar * 1e3
        if (theory_scalar and np.isfinite(theory_scalar))
        else None,
        "rms_to_theory": ratio,
        "n_good": n_good,
        "n_warn": n_warn,
        "n_bad": n_bad,
        "n_total": n_total,
        "n_excluded": n_total - n_graded,
        "pct_bad": pct_bad,
        "pct_warn": pct_warn,
        "overall_quality": overall,
    }
