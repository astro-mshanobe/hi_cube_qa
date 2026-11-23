"""Plotly figures.

Figures consume the compact reductions produced by reduce.py (1-D spectra),
never whole cubes. The theoretical-noise reference is a per-channel array
(frequency- and flagging-aware), so it is drawn as a curve, not a flat line.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .noise import cumulative_rms, running_median_rms

HI_REST_FREQ_HZ = 1.420405752e9
C_KMS = 2.99792458e5

COL_DIRTY = "#2E75B6"
COL_CLEAN = "#E04B3A"
COL_THEORY = "#27AE60"
COL_WARN = "#F39C12"
COL_BAD = "#C0392B"
COL_BG = "#0F1117"
COL_GRID = "#2A2D35"
COL_TEXT = "#E0E0E0"

CUBE_COLOURS = [COL_DIRTY, COL_CLEAN, "#8E44AD", "#1ABC9C", "#F39C12"]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COL_BG,
        plot_bgcolor=COL_BG,
        font=dict(color=COL_TEXT, family="Inter, Arial, sans-serif", size=12),
        xaxis=dict(
            gridcolor=COL_GRID,
            zerolinecolor=COL_GRID,
            linecolor=COL_GRID,
            ticks="outside",
            ticklen=6,
            tickcolor=COL_GRID,
        ),
        yaxis=dict(
            gridcolor=COL_GRID,
            zerolinecolor=COL_GRID,
            linecolor=COL_GRID,
            ticks="outside",
            ticklen=6,
            tickcolor=COL_GRID,
        ),
        legend=dict(bgcolor="rgba(30,33,43,0.85)", bordercolor=COL_GRID, borderwidth=1),
        hoverlabel=dict(bgcolor="#1E212B", font_color=COL_TEXT),
    )
)


# --- Plot decimation (keeps files small; CSVs stay full resolution) ---
MAX_PLOT_POINTS: int | None = None


def set_max_plot_points(n) -> None:
    """Set the global plot-decimation cap (None/0 disables decimation)."""
    global MAX_PLOT_POINTS
    MAX_PLOT_POINTS = int(n) if (n and int(n) > 0) else None


def _stride(n: int) -> int:
    """Stride so that at most MAX_PLOT_POINTS points are drawn per trace."""
    if MAX_PLOT_POINTS and n > MAX_PLOT_POINTS:
        return int(np.ceil(n / MAX_PLOT_POINTS))
    return 1


def _keep_mask(n: int, extra: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of channels to plot: a uniform stride, plus any *extra*
    channels (e.g. Bad) kept in full so outliers are never hidden."""
    step = _stride(n)
    keep = np.zeros(n, dtype=bool)
    keep[::step] = True
    if extra is not None:
        keep |= extra
    return keep


def _as_scalar_theory(theory) -> float | None:
    if theory is None:
        return None
    ta = np.asarray(theory, dtype=float)
    return float(ta) if ta.ndim == 0 else float(np.nanmedian(ta))


# Spectral axes: radio velocity at the bottom, channel index at the top
VEL_AXIS_TITLE = "Radio velocity (km/s)"
CHAN_AXIS_TITLE = "Channel index"
HOVER_CH = "Ch %{customdata} | v = %{x:.1f} km/s"
_TOP_TICKS = 8
LEGEND_SIDE = dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top")
_MARGIN_R = 200
_TOP_AXIS_PX = 68  # tick labels and title of a top axis
_TITLE_PX = 35  # figure title
_SUBTITLE_PX = 22  # subplot title above the first panel


def _xy(vel_kms, chans, mask) -> dict:
    """x (velocity) and customdata (channel index) for the channels selected by *mask*."""
    return dict(x=np.asarray(vel_kms, dtype=float)[mask], customdata=chans[mask])


def _nice_step(span: float) -> int:
    """Tick step (1, 2 or 5 times a power of ten) giving about _TOP_TICKS ticks over *span* channels."""
    raw = max(span / _TOP_TICKS, 1.0)
    power = 10 ** int(np.floor(np.log10(raw)))
    for mult in (1, 2, 5, 10):
        if mult * power >= raw:
            return int(mult * power)
    return int(10 * power)


def channel_ticks(nchan: int, lo: float = 0.0, hi: float | None = None) -> np.ndarray:
    """Channel indices at which the top axis is labelled, between channels *lo* and *hi*."""
    hi = nchan - 1 if hi is None else hi
    lo, hi = max(0, int(np.ceil(lo))), min(nchan - 1, int(np.floor(hi)))
    if hi < lo:
        return np.array([], dtype=int)
    step = _nice_step(hi - lo)
    return np.arange(int(np.ceil(lo / step)) * step, hi + 1, step)


def add_top_axis(fig: go.Figure, n_xaxes: int, title: str, tickvals, ticktext, x_span) -> int:
    """Add a second x axis along the top of the first panel, tied to the main x axis
    (same range) but labelled with *ticktext* at the positions *tickvals*. *n_xaxes* is
    the number of x axes the figure already has (rows of a shared-x subplot grid).
    Room is made above the panel and the first panel's subplot title, if any, is moved
    above the new axis. Returns the number of the new axis.
    """
    name = n_xaxes + 1
    fig.update_layout(
        {
            f"xaxis{name}": dict(
                title=title,
                overlaying="x",
                side="top",
                matches="x",
                anchor="y",
                tickmode="array",
                tickvals=list(tickvals),
                ticktext=list(ticktext),
                tickangle=0,
                showgrid=False,
                zeroline=False,
            )
        }
    )
    fig.add_trace(
        go.Scatter(
            x=[float(x_span[0]), float(x_span[1])],
            y=[None, None],
            xaxis=f"x{name}",
            yaxis="y",
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
            name="top axis",
        )
    )
    first_top = fig.layout.yaxis.domain[1] if fig.layout.yaxis.domain else 1.0
    titled = [
        a
        for a in fig.layout.annotations
        if a.xref == "paper" and a.yref == "paper" and abs(a.y - first_top) < 1e-6
    ]
    for annotation in titled:
        annotation.yshift = _TOP_AXIS_PX
    top = _TITLE_PX + _TOP_AXIS_PX + (_SUBTITLE_PX if titled else 0) + 10
    height = fig.layout.height or 450
    fig.update_layout(
        margin=dict(t=top),
        height=height - (fig.layout.margin.t or 80) + top,
        title=dict(y=1.0, yref="container", yanchor="top", pad=dict(t=8)),
    )
    return name


def add_channel_axis(fig: go.Figure, vel_kms, n_xaxes: int = 1) -> None:
    """Label the velocity axis from below with VEL_AXIS_TITLE and add a channel-index
    axis along the top of the first panel. The channel axis is tied to the velocity
    axis, carries ticks at round channel numbers, and is refreshed on zoom by the
    script in report.py (it reads the figure's meta, which is only set when velocity
    is linear in channel number).

    *n_xaxes* is the number of x axes the figure already has; the velocity title goes
    on the last one.
    """
    vel = np.asarray(vel_kms, dtype=float)
    nchan = len(vel)
    ticks = channel_ticks(nchan)
    fig.update_layout({f"xaxis{n_xaxes if n_xaxes > 1 else ''}": dict(title=VEL_AXIS_TITLE)})
    name = add_top_axis(
        fig, n_xaxes, CHAN_AXIS_TITLE, vel[ticks], [str(c) for c in ticks], (vel[0], vel[-1])
    )
    step = float(vel[-1] - vel[0]) / (nchan - 1) if nchan > 1 else 1.0
    if nchan > 1 and np.allclose(
        vel, vel[0] + step * np.arange(nchan), rtol=0, atol=1e-3 * abs(step)
    ):
        # the in-page refresh assumes velocity is linear in channel number
        fig.update_layout(
            meta=dict(
                chan_axes=[
                    dict(axis=f"xaxis{name}", ref="xaxis", v0=float(vel[0]), dv=step, n=int(nchan))
                ]
            )
        )


# Per-channel RMS
def fig_rms_curve(freq_hz, vel_kms, rms_dict, theory=None) -> go.Figure:
    """Per-channel RMS, coloured by quality, with a per-channel theory curve."""
    fig = go.Figure()
    for label, (rms, col, quality) in rms_dict.items():
        n = len(rms)
        chans = np.arange(n)
        good = quality == "Good"
        warn = quality == "Warn"
        bad = quality == "Bad"
        # Decimate the dense Good/Warn/median traces; keep every Bad channel.
        keep = _keep_mask(n)
        gk = good & keep
        wk = warn & keep

        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, keep),
                y=running_median_rms(rms, 20)[keep] * 1e3,
                mode="lines",
                name=f"{label} running median",
                line=dict(color=col, width=2.5, dash="dash"),
                opacity=0.65,
                hovertemplate=HOVER_CH + "<br>Running med = %{y:.3f} mJy/beam<extra></extra>",
            )
        )
        if gk.any():
            fig.add_trace(
                go.Scatter(
                    **_xy(vel_kms, chans, gk),
                    y=rms[gk] * 1e3,
                    mode="lines",
                    name=label,
                    line=dict(color=col, width=1.2),
                    hovertemplate=HOVER_CH + "<br>RMS = %{y:.3f} mJy/beam<extra></extra>",
                )
            )
        if wk.any():
            fig.add_trace(
                go.Scatter(
                    **_xy(vel_kms, chans, wk),
                    y=rms[wk] * 1e3,
                    mode="markers",
                    name=f"{label} Warn",
                    marker=dict(color=COL_WARN, symbol="circle-open", size=5),
                    hovertemplate=HOVER_CH + " [Warn]<br>RMS = %{y:.3f} mJy/beam<extra></extra>",
                )
            )
        if bad.any():
            fig.add_trace(
                go.Scatter(
                    **_xy(vel_kms, chans, bad),
                    y=rms[bad] * 1e3,
                    mode="markers",
                    name=f"{label} Bad",
                    marker=dict(color=COL_BAD, symbol="x", size=7),
                    hovertemplate=HOVER_CH + " [Bad]<br>RMS = %{y:.3f} mJy/beam<extra></extra>",
                )
            )

    if theory is not None:
        ta = np.asarray(theory, dtype=float)
        if ta.ndim == 0:
            fig.add_hline(
                y=float(ta) * 1e3,
                line=dict(color=COL_THEORY, dash="dash", width=1.8),
                annotation_text=f"theory = {float(ta) * 1e3:.3f} mJy/beam",
                annotation_font_color=COL_THEORY,
                annotation_position="top right",
            )
        else:
            km = _keep_mask(len(ta))
            fig.add_trace(
                go.Scatter(
                    **_xy(vel_kms, np.arange(len(ta)), km),
                    y=ta[km] * 1e3,
                    mode="lines",
                    name="theory (freq+flag aware)",
                    line=dict(color=COL_THEORY, width=1.8, dash="dash"),
                    hovertemplate=HOVER_CH + "<br>theory = %{y:.3f} mJy/beam<extra></extra>",
                )
            )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Per-channel RMS noise (colour = quality class)", font=dict(size=15)),
        yaxis=dict(title="RMS noise (mJy/beam)"),
        legend=LEGEND_SIDE,
        height=440,
        margin=dict(l=65, r=_MARGIN_R, t=80, b=60),
    )
    add_channel_axis(fig, vel_kms)
    return fig


def fig_cumulative_rms(freq_hz, vel_kms, rms_dict, theory=None) -> go.Figure:
    fig = go.Figure()
    chans = np.arange(len(freq_hz))
    km = _keep_mask(len(freq_hz))
    for label, (rms, col, _q) in rms_dict.items():
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=cumulative_rms(rms)[km] * 1e3,
                mode="lines",
                name=label,
                line=dict(color=col, width=2.0),
                hovertemplate=HOVER_CH
                + "<br>Cumulative median RMS = %{y:.3f} mJy/beam<extra></extra>",
            )
        )
    ts = _as_scalar_theory(theory)
    if ts is not None:
        fig.add_hline(
            y=ts * 1e3,
            line=dict(color=COL_THEORY, dash="dot", width=1.5),
            annotation_text=f"theory = {ts * 1e3:.3f} mJy/beam",
            annotation_font_color=COL_THEORY,
            annotation_position="top right",
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Cumulative median RMS (noise-floor convergence)", font=dict(size=15)),
        yaxis=dict(title="Cumulative median RMS (mJy/beam)"),
        legend=LEGEND_SIDE,
        height=380,
        margin=dict(l=65, r=_MARGIN_R, t=80, b=60),
    )
    add_channel_axis(fig, vel_kms)
    return fig


def fig_rms_histogram(rms_dict) -> go.Figure:
    fig = go.Figure()
    for label, (rms, col, _q) in rms_dict.items():
        med = np.nanmedian(rms)
        clean = rms[(rms < 5 * med) & np.isfinite(rms)]
        fig.add_trace(
            go.Histogram(
                x=clean * 1e3,
                nbinsx=40,
                name=f"{label} (median {med * 1e3:.3f})",
                marker_color=col,
                opacity=0.72,
            )
        )
        fig.add_vline(x=med * 1e3, line=dict(color=col, dash="dash", width=1.5))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="overlay",
        title=dict(text="RMS noise distribution (dashed = median)", font=dict(size=15)),
        xaxis=dict(title="RMS noise (mJy/beam)"),
        yaxis=dict(title="Number of channels"),
        legend=LEGEND_SIDE,
        height=380,
        margin=dict(l=60, r=_MARGIN_R, t=80, b=60),
    )
    return fig


def fig_rms_vs_theory(freq_hz, vel_kms, rms_dict, theory) -> go.Figure:
    """Per-channel RMS / theory (theory may be a per-channel array)."""
    fig = go.Figure()
    ta = np.asarray(theory, dtype=float)
    for label, (rms, col, quality) in rms_dict.items():
        n = len(rms)
        chans = np.arange(n)
        denom = ta if ta.ndim and ta.shape[0] == n else np.full(n, float(ta))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = rms / denom
        cmap = {"Good": COL_THEORY, "Warn": COL_WARN, "Bad": COL_BAD}
        # keep every Bad channel; decimate the rest
        keep = _keep_mask(n, extra=(quality == "Bad"))
        mc = [cmap.get(q, COL_THEORY) for q in quality[keep]]
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, keep),
                y=ratio[keep],
                mode="lines+markers",
                name=label,
                marker=dict(color=mc, size=4),
                line=dict(color=col, width=0.8),
                hovertemplate=HOVER_CH + "<br>RMS/theory = %{y:.2f}<extra></extra>",
            )
        )
    fig.add_hline(
        y=1.0,
        line=dict(color=COL_THEORY, dash="dot", width=1.2),
        annotation_text="1x theory",
        annotation_font_color=COL_THEORY,
    )
    fig.add_hline(
        y=1.5,
        line=dict(color=COL_WARN, dash="dash", width=1.2),
        annotation_text="1.5x (Warn)",
        annotation_font_color=COL_WARN,
    )
    fig.add_hline(
        y=3.0,
        line=dict(color=COL_BAD, dash="dash", width=1.2),
        annotation_text="3x (Bad)",
        annotation_font_color=COL_BAD,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Per-channel RMS / theoretical noise", font=dict(size=15)),
        yaxis=dict(title="RMS / theory", type="log"),
        legend=LEGEND_SIDE,
        height=400,
        margin=dict(l=70, r=_MARGIN_R, t=80, b=60),
    )
    add_channel_axis(fig, vel_kms)
    return fig


def fig_noise_overview(
    freq_hz, vel_kms, noise_ref, theory_flat, theory_flag, title="Noise overview"
) -> go.Figure:
    """Measured boresight / median / peak vs the theory curve (flat and
    flagging-reduced).
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06
    )
    n = noise_ref["nchan"]
    chans = np.arange(n)
    km = _keep_mask(n)

    def line(y, name, col, dash="solid", width=1.4, row=1, show=True):
        if y is None:
            return
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=np.asarray(y)[km] * 1e3,
                mode="lines",
                name=name,
                line=dict(color=col, width=width, dash=dash),
                showlegend=show,
            ),
            row=row,
            col=1,
        )

    line(noise_ref.get("boresight"), "Boresight (best-case)", "#9B59B6")
    line(noise_ref.get("median"), "Median (full image)", COL_DIRTY, dash="dash", width=1.2)
    line(noise_ref.get("peak"), "Peak |max pixel|", COL_BAD, dash="dot", width=0.7)
    line(theory_flat, "theory (flat)", COL_THEORY, dash="dash")
    line(theory_flag, "theory (flag-reduced)", COL_WARN, dash="dot")

    # ratio panel: measured median / theory(flag-reduced)
    tflag = np.asarray(theory_flag, dtype=float) if theory_flag is not None else None
    med = noise_ref.get("median")
    if med is not None and tflag is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.asarray(med) / tflag
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=ratio[km],
                mode="lines",
                name="median / theory",
                line=dict(color=COL_DIRTY, width=1.2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#555", row=2, col=1)

    fig.update_yaxes(title_text="RMS (mJy/beam)", row=1, col=1)
    fig.update_yaxes(title_text="median / theory", row=2, col=1)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        hovermode="x unified",
        height=600,
        title=dict(text=title, font=dict(size=15)),
        legend=LEGEND_SIDE,
        margin=dict(l=70, r=_MARGIN_R, t=100, b=60),
    )
    add_channel_axis(fig, vel_kms, n_xaxes=2)
    return fig


def fig_spectral_profile(freq_hz, vel_kms, prof_dict) -> go.Figure:
    """prof_dict[label] = {'peak':arr_jy, 'mean':arr_jy, 'colour':col}."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Peak flux per channel", "Mean flux per channel"],
        vertical_spacing=0.08,
    )
    for label, d in prof_dict.items():
        n = len(d["peak"])
        chans = np.arange(n)
        km = _keep_mask(n)
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=np.asarray(d["peak"])[km] * 1e3,
                mode="lines",
                name=f"{label} peak",
                line=dict(color=d["colour"], width=1.5),
                hovertemplate=HOVER_CH + "<br>Peak = %{y:.3f} mJy/beam<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=np.asarray(d["mean"])[km] * 1e6,
                mode="lines",
                name=f"{label} mean",
                line=dict(color=d["colour"], width=1.5, dash="dot"),
                hovertemplate=HOVER_CH + "<br>Mean = %{y:.3f} uJy/beam<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Spectral profile", font=dict(size=15)),
        height=480,
        yaxis=dict(title="Peak (mJy/beam)"),
        yaxis2=dict(title="Mean (uJy/beam)"),
        legend=LEGEND_SIDE,
        margin=dict(l=65, r=_MARGIN_R, t=100, b=60),
    )
    add_channel_axis(fig, vel_kms, n_xaxes=2)
    return fig


def fig_band_spectrum(freq_hz, vel_kms, spec_dict, z_source=None) -> go.Figure:
    """spec_dict[label] = {'s_int':arr_jy, 'peak':arr_jy, 'rms':arr_jy,
    'sigma_int':float_jy, 'colour':col}.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=[
            "Spatially integrated flux density S_int(v)",
            "Peak flux per channel S_peak(v)",
        ],
        vertical_spacing=0.08,
    )
    vel_min, vel_max = float(np.nanmin(vel_kms)), float(np.nanmax(vel_kms))
    freq_lo = HI_REST_FREQ_HZ * (1.0 - vel_max / C_KMS)
    freq_hi = HI_REST_FREQ_HZ * (1.0 - vel_min / C_KMS)
    freq_ticks = np.linspace(freq_lo, freq_hi, 5)
    vel_at = C_KMS * (HI_REST_FREQ_HZ - freq_ticks) / HI_REST_FREQ_HZ

    km = _keep_mask(len(vel_kms))
    v = np.asarray(vel_kms)[km]
    for label, d in spec_dict.items():
        col = d["colour"]
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        s_int = np.asarray(d["s_int"])[km]
        peak = np.asarray(d["peak"])[km]
        env1 = np.full(len(v), float(d["sigma_int"]) * 1e3)
        env3 = env1 * 3.0
        fig.add_trace(
            go.Scatter(
                x=list(v) + list(v[::-1]),
                y=list(env3) + list(-env3[::-1]),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.08)",
                line=dict(width=0),
                name=f"{label} +/-3sig",
                legendgroup=label,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(v) + list(v[::-1]),
                y=list(env1) + list(-env1[::-1]),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.18)",
                line=dict(width=0),
                name=f"{label} +/-1sig",
                legendgroup=label,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=0, line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=v,
                y=s_int * 1e3,
                mode="lines",
                name=f"{label} S_int",
                legendgroup=label,
                line=dict(color=col, width=2.5),
                hovertemplate="v = %{x:.0f} km/s<br>S_int = %{y:.3f} mJy<extra>"
                + label
                + "</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=v,
                y=peak * 1e3,
                mode="lines",
                name=f"{label} S_peak",
                legendgroup=label,
                line=dict(color=col, width=2.0),
                hovertemplate="v = %{x:.0f} km/s<br>S_peak = %{y:.3f} mJy/beam<extra>"
                + label
                + "</extra>",
            ),
            row=2,
            col=1,
        )

    if z_source is not None:
        # radio velocity of a line at redshift z: v = c z / (1+z)
        v_src = C_KMS * z_source / (1.0 + z_source)
        f_src = HI_REST_FREQ_HZ / (1.0 + z_source)
        fig.add_vline(x=v_src, line=dict(color=COL_WARN, dash="dash", width=2.0))
        fig.add_annotation(
            x=v_src,
            y=0.97,
            xref="x",
            yref="paper",
            text=f"z={z_source:.4f}  {f_src / 1e6:.2f} MHz",
            showarrow=False,
            font=dict(color=COL_WARN, size=11),
            bgcolor="rgba(20,22,30,0.80)",
            bordercolor=COL_WARN,
            borderwidth=1,
            xanchor="left",
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=660,
        title=dict(text="Full-band HI spectrum - integrated & peak flux", font=dict(size=15)),
        xaxis=dict(tickvals=vel_at.tolist(), ticktext=[f"{v:.0f}" for v in vel_at]),
        xaxis2=dict(
            title=VEL_AXIS_TITLE, tickvals=vel_at.tolist(), ticktext=[f"{v:.0f}" for v in vel_at]
        ),
        yaxis=dict(title="S_int (mJy)"),
        yaxis2=dict(title="S_peak (mJy/beam)"),
        legend=dict(LEGEND_SIDE, tracegroupgap=8),
        margin=dict(l=75, r=_MARGIN_R, t=120, b=70),
    )
    add_top_axis(
        fig,
        2,
        "Frequency (MHz)",
        vel_at,
        [f"{f / 1e6:.1f}" for f in freq_ticks],
        (vel_min, vel_max),
    )
    return fig


def fig_nhi_sensitivity(freq_hz, vel_kms, nhi_dict, dv_kms=20.0) -> go.Figure:
    """nhi_dict[label] = {'nhi_chan':arr_cm2, 'colour':col}; per-channel 3sigma N_HI."""
    fig = go.Figure()
    for label, d in nhi_dict.items():
        y = np.asarray(d["nhi_chan"], dtype=float)
        n = len(y)
        chans = np.arange(n)
        km = _keep_mask(n)
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=y[km],
                mode="lines",
                name=label,
                line=dict(color=d["colour"], width=1.6),
                hovertemplate=HOVER_CH + "<br>3sig N_HI = %{y:.2e} cm^-2<extra></extra>",
            )
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="3σ HI column-density sensitivity (per channel)", font=dict(size=15)),
        yaxis=dict(title="N_HI (atoms cm^-2)", type="log"),
        legend=LEGEND_SIDE,
        height=360,
        margin=dict(l=80, r=_MARGIN_R, t=80, b=60),
    )
    add_channel_axis(fig, vel_kms)
    return fig


def fig_kurtosis(freq_hz, vel_kms, kurt_dict) -> go.Figure:
    """Per-channel Pearson kurtosis (expectation 3 for Gaussian noise), a shape
    diagnostic independent of RMS: a channel can pass the RMS/theory check
    while still being non-Gaussian (residual RFI, incomplete continuum
    subtraction), which this catches instead.

    kurt_dict[label] = {'kurtosis': arr, 'se': arr_or_scalar, 'colour': col}
    """
    fig = go.Figure()
    for label, d in kurt_dict.items():
        y = np.asarray(d["kurtosis"], dtype=float)
        n = len(y)
        chans = np.arange(n)
        km = _keep_mask(n)
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=y[km],
                mode="lines",
                name=label,
                line=dict(color=d["colour"], width=1.2),
                hovertemplate=HOVER_CH + "<br>kurtosis = %{y:.3f}<extra></extra>",
            )
        )
        se = d.get("se")
        if se is not None:
            se_med = float(np.nanmedian(np.asarray(se, dtype=float)))
            fig.add_hrect(
                y0=3 - 5 * se_med,
                y1=3 + 5 * se_med,
                fillcolor=COL_THEORY,
                opacity=0.06,
                layer="below",
                line_width=0,
            )
    fig.add_hline(
        y=3.0,
        line=dict(color=COL_THEORY, dash="dash", width=1.5),
        annotation_text="Gaussian expectation = 3",
        annotation_font_color=COL_THEORY,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Per-channel kurtosis (shaded: ±5 standard errors)", font=dict(size=15)),
        yaxis=dict(title="Pearson kurtosis"),
        legend=LEGEND_SIDE,
        height=360,
        margin=dict(l=70, r=_MARGIN_R, t=80, b=60),
    )
    add_channel_axis(fig, vel_kms)
    return fig


def fig_beam_stats(freq_hz, vel_kms, beam_dict) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Beam major / minor axis (arcsec)", "Beam position angle (deg)"],
        vertical_spacing=0.08,
    )
    for label, (bmaj, bmin, bpa, col) in beam_dict.items():
        if bmaj is None:
            continue
        n = len(bmaj)
        chans = np.arange(n)
        km = _keep_mask(n)
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=np.asarray(bmaj)[km],
                mode="lines",
                name=f"{label} BMAJ",
                line=dict(color=col, width=1.8),
                hovertemplate=HOVER_CH + '<br>BMAJ = %{y:.2f}"<extra></extra>',
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=np.asarray(bmin)[km],
                mode="lines",
                name=f"{label} BMIN",
                line=dict(color=col, width=1.8, dash="dot"),
                hovertemplate=HOVER_CH + '<br>BMIN = %{y:.2f}"<extra></extra>',
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                **_xy(vel_kms, chans, km),
                y=np.asarray(bpa)[km],
                mode="lines",
                name=f"{label} BPA",
                line=dict(color=col, width=1.5, dash="dash"),
                hovertemplate=HOVER_CH + "<br>BPA = %{y:.1f} deg<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Synthesised beam parameters", font=dict(size=15)),
        height=500,
        yaxis=dict(title="Beam axis (arcsec)"),
        yaxis2=dict(title="BPA (deg)"),
        legend=LEGEND_SIDE,
        margin=dict(l=65, r=_MARGIN_R, t=100, b=60),
    )
    add_channel_axis(fig, vel_kms, n_xaxes=2)
    return fig


def fig_flag_summary(flag_data) -> go.Figure | None:
    """One bar per flag report (per pointing/stage), coloured by fraction."""
    if flag_data is None:
        return None
    from .io_utils import flag_fraction_of

    labels, fracs, colours = [], [], []
    for label, content in flag_data.items():
        fr = flag_fraction_of(content)
        if fr is None:
            continue
        f = float(fr) * 100
        labels.append(label)
        fracs.append(f)
        colours.append(COL_BAD if f > 30 else COL_WARN if f > 10 else COL_THEORY)

    if not labels:
        return None
    # Sort bars by fraction (largest at top of the horizontal chart).
    order = np.argsort(fracs)
    labels = [labels[i] for i in order]
    fracs = [fracs[i] for i in order]
    colours = [colours[i] for i in order]
    fig = go.Figure(
        go.Bar(
            x=fracs,
            y=labels,
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y}<br>Flagged: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(
        x=10,
        line=dict(color=COL_WARN, dash="dash", width=1),
        annotation_text="10%",
        annotation_font_color=COL_WARN,
    )
    fig.add_vline(
        x=30,
        line=dict(color=COL_BAD, dash="dash", width=1),
        annotation_text="30%",
        annotation_font_color=COL_BAD,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Flagging fraction summary", font=dict(size=15)),
        xaxis=dict(title="Flagged data (%)"),
        height=max(300, 50 * len(labels) + 120),
        margin=dict(l=200, r=_MARGIN_R, t=80, b=60),
    )
    return fig


def fig_summary_table(
    noise_summaries, beam_diagnostics, theory, nchan, nhi_summaries=None
) -> go.Figure:
    """QA summary table; includes N_HI sensitivity rows when provided."""
    rows_h, rows_v, rows_s = [], [], []
    nhi_summaries = nhi_summaries or {}

    for label, ns in noise_summaries.items():
        rows_h.append(f"{label}: Median RMS")
        rows_v.append(f"{ns['median_rms_mjy']:.3f} mJy/beam")
        rows_s.append(ns["overall_quality"])
        rows_h.append(f"{label}: RMS range")
        rows_v.append(f"{ns['min_rms_mjy']:.3f} - {ns['max_rms_mjy']:.3f} mJy/beam")
        rows_s.append("-")
        if ns["theory_rms_mjy"] is not None:
            rows_h.append(f"{label}: Theory (band-avg)")
            rows_v.append(f"{ns['theory_rms_mjy']:.3f} mJy/beam")
            rows_s.append("-")
            ratio = ns["rms_to_theory"]
            if ratio is not None:
                rows_h.append(f"{label}: RMS / theory")
                rows_v.append(f"{ratio:.2f}x")
                rows_s.append("Good" if ratio < 1.5 else "Warn" if ratio < 3.0 else "Bad")
        nh = nhi_summaries.get(label)
        if nh is not None and np.isfinite(nh.get("nhi_dv_cm2", np.nan)):
            rows_h.append(f"{label}: 3σ N_HI ({nh['dv_kms']:.0f} km/s)")
            rows_v.append(f"{nh['nhi_dv_cm2']:.2e} cm^-2")
            rows_s.append("-")
        rows_h.append(f"{label}: Good / Warn / Bad")
        rows_v.append(f"{ns['n_good']} / {ns['n_warn']} / {ns['n_bad']} of {ns['n_total']}")
        rows_s.append("Good" if ns["pct_bad"] < 5 else "Warn" if ns["pct_bad"] < 15 else "Bad")

    for label, bd in beam_diagnostics.items():
        if not bd.get("available"):
            rows_h.append(f"{label}: Beam info")
            rows_v.append("Not available")
            rows_s.append("-")
            continue
        rows_h.append(f"{label}: Median BMAJ x BMIN")
        rows_v.append(f'{bd["median_bmaj_asec"]:.2f}" x {bd["median_bmin_asec"]:.2f}"')
        rows_s.append("-")
        rows_h.append(f"{label}: Axial ratio")
        rows_v.append(f"{bd['axial_ratio']:.2f}")
        rows_s.append("-")
        rows_h.append(f"{label}: BMAJ std across band")
        rows_v.append(f'{bd["std_bmaj_asec"]:.3f}"')
        rows_s.append(bd["bmaj_scatter_quality"])

    col_fill = []
    for s in rows_s:
        col_fill.append(
            "rgba(39,174,96,0.18)"
            if s == "Good"
            else "rgba(243,156,18,0.18)"
            if s == "Warn"
            else "rgba(192,57,43,0.18)"
            if s == "Bad"
            else "#1A1D26"
        )
    fig = go.Figure(
        go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>", "<b>Status</b>"],
                fill_color=COL_DIRTY,
                font=dict(color="white", size=13),
                align="left",
                height=32,
            ),
            cells=dict(
                values=[rows_h, rows_v, rows_s],
                fill_color=[["#1A1D26"] * len(rows_h), ["#1A1D26"] * len(rows_v), col_fill],
                font=dict(color=COL_TEXT, size=12),
                align=["left", "right", "center"],
                height=28,
            ),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="QA summary statistics", font=dict(size=15)),
        height=max(300, 30 * len(rows_h) + 100),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
