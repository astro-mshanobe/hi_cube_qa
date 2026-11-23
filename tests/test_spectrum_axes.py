"""spectrum: figures are drawn against radio velocity, with the channel index along the top axis."""

import numpy as np

from hi_cube_qa import report
from hi_cube_qa import spectrum as sp

N = 40
CHAN = np.arange(N)
VEL = 4300.0 - 5.5 * CHAN  # velocity falls with channel number, as for an increasing-frequency axis
FREQ = 1.4e9 + 26367.19 * CHAN
RMS = {"A": (1e-3 * np.ones(N), "#2E75B6", np.array(["Good"] * N, dtype=object))}
RMS["A"][2][5] = "Bad"


def _figures(vel=VEL):
    theory = 1e-3 * np.ones(N)
    per = {
        "nhi": {"A": {"nhi_chan": np.ones(N), "colour": "#2E75B6"}},
        "kurt": {"A": {"kurtosis": np.full(N, 3.0), "se": 0.1, "colour": "#2E75B6"}},
        "beam": {"A": (np.full(N, 20.0), np.full(N, 15.0), np.zeros(N), "#2E75B6")},
        "prof": {"A": {"peak": np.ones(N), "mean": np.zeros(N), "colour": "#2E75B6"}},
    }
    ref = {
        "nchan": N,
        "boresight": np.ones(N) * 1e-3,
        "median": np.ones(N) * 1e-3,
        "peak": np.ones(N) * 4e-3,
    }
    return {
        "rms_curve": (sp.fig_rms_curve(FREQ, vel, RMS, theory), 1),
        "rms_vs_theory": (sp.fig_rms_vs_theory(FREQ, vel, RMS, theory), 1),
        "cumulative_rms": (sp.fig_cumulative_rms(FREQ, vel, RMS, theory), 1),
        "noise_overview": (sp.fig_noise_overview(FREQ, vel, ref, theory, theory), 2),
        "spectral": (sp.fig_spectral_profile(FREQ, vel, per["prof"]), 2),
        "nhi": (sp.fig_nhi_sensitivity(FREQ, vel, per["nhi"]), 1),
        "kurtosis": (sp.fig_kurtosis(FREQ, vel, per["kurt"]), 1),
        "beam": (sp.fig_beam_stats(FREQ, vel, per["beam"]), 2),
    }


def _data_traces(fig):
    return [t for t in fig.data if t.name != "top axis"]


def test_channel_ticks():
    assert list(sp.channel_ticks(40)) == [0, 5, 10, 15, 20, 25, 30, 35]
    assert list(sp.channel_ticks(4783)) == [0, 1000, 2000, 3000, 4000]
    assert list(sp.channel_ticks(4783, lo=1000.4, hi=1210.7)) == [1050, 1100, 1150, 1200]
    assert list(sp.channel_ticks(10, lo=3.2, hi=3.8)) == []
    assert list(sp.channel_ticks(10, lo=-50, hi=500)) == [0, 2, 4, 6, 8]
    assert list(sp.channel_ticks(3)) == [0, 1, 2]  # fewer channels than ticks: label each one
    assert list(sp.channel_ticks(1)) == [0]


def test_figures_plot_velocity_with_channel_hover():
    for name, (fig, _) in _figures().items():
        for trace in _data_traces(fig):
            if trace.x is None or len(trace.x) == 0:
                continue
            ch = np.asarray(trace.customdata, dtype=int)
            assert np.allclose(trace.x, VEL[ch]), name
            if trace.hovertemplate:
                assert (
                    "Ch %{customdata}" in trace.hovertemplate and "km/s" in trace.hovertemplate
                ), name


def test_axis_titles_and_top_ticks():
    for name, (fig, n_x) in _figures().items():
        bottom = fig.layout[f"xaxis{n_x if n_x > 1 else ''}"]
        top = fig.layout[f"xaxis{n_x + 1}"]
        assert bottom.title.text == sp.VEL_AXIS_TITLE, name
        assert (
            top.title.text == sp.CHAN_AXIS_TITLE
            and top.side == "top"
            and top.matches == "x"
            and top.overlaying == "x"
        ), name
        chans = [int(t) for t in top.ticktext]
        assert chans == list(sp.channel_ticks(N)), name
        assert np.allclose(top.tickvals, VEL[chans]), name
        axis_trace = [t for t in fig.data if t.name == "top axis"]
        assert (
            len(axis_trace) == 1
            and axis_trace[0].xaxis == f"x{n_x + 1}"
            and axis_trace[0].yaxis == "y"
        ), name
        assert list(axis_trace[0].x) == [VEL[0], VEL[-1]] and top.tickangle == 0, (
            name
        )  # horizontal labels: rotated ones grow into the title


def test_decimation_keeps_pairs():
    sp.set_max_plot_points(7)
    try:
        fig = sp.fig_nhi_sensitivity(
            FREQ, VEL, {"A": {"nhi_chan": np.ones(N), "colour": "#2E75B6"}}
        )
    finally:
        sp.set_max_plot_points(None)
    trace = _data_traces(fig)[0]
    assert 0 < len(trace.x) < N and np.allclose(
        trace.x, VEL[np.asarray(trace.customdata, dtype=int)]
    )


def test_bad_channels_hover():
    fig = sp.fig_rms_curve(FREQ, VEL, RMS)
    bad = [t for t in fig.data if t.name == "A Bad"][0]
    assert (
        list(bad.customdata) == [5]
        and np.allclose(bad.x, [VEL[5]])
        and "[Bad]" in bad.hovertemplate
        and "Ch %{customdata}" in bad.hovertemplate
    )


def test_refresh_metadata():
    fig = sp.fig_kurtosis(
        FREQ, VEL, {"A": {"kurtosis": np.full(N, 3.0), "se": 0.1, "colour": "#2E75B6"}}
    )
    axis = fig.layout.meta["chan_axes"][0]
    assert (
        axis["axis"] == "xaxis2"
        and axis["n"] == N
        and abs(axis["dv"] + 5.5) < 1e-9
        and abs(axis["v0"] - 4300.0) < 1e-9
    )
    curved = VEL + 0.3 * CHAN**2
    fig = sp.fig_kurtosis(
        FREQ, curved, {"A": {"kurtosis": np.full(N, 3.0), "se": 0.1, "colour": "#2E75B6"}}
    )
    assert fig.layout.meta is None or not fig.layout.meta.get("chan_axes")
    assert np.allclose(
        fig.layout.xaxis2.tickvals, curved[[int(t) for t in fig.layout.xaxis2.ticktext]]
    )


def test_band_spectrum_has_no_channel_axis():
    spec = {
        "A": {
            "s_int": np.zeros(N),
            "peak": np.zeros(N),
            "rms": np.zeros(N),
            "sigma_int": 1e-3,
            "colour": "#2E75B6",
        }
    }
    fig = sp.fig_band_spectrum(FREQ, VEL, spec)
    assert fig.layout.xaxis3.title.text != sp.CHAN_AXIS_TITLE and fig.layout.meta is None


def test_renderer_adds_zoom_script():
    fig = sp.fig_kurtosis(
        FREQ, VEL, {"A": {"kurtosis": np.full(N, 3.0), "se": 0.1, "colour": "#2E75B6"}}
    )
    html = report._FigRenderer()(fig, "fig_x")
    assert "plotly_relayout" in html and "getElementById('fig_x')" in html and "chan_axes" in html


def test_legend_and_margins():
    for name, (fig, _) in _figures().items():
        assert (
            fig.layout.legend.orientation == "v"
            and fig.layout.legend.x > 1.0
            and fig.layout.legend.xanchor == "left"
        ), name
        assert fig.layout.margin.r >= 200, name
        assert (
            fig.layout.margin.t >= sp._TITLE_PX + sp._TOP_AXIS_PX
            and fig.layout.title.yref == "container"
        ), name


def test_first_subplot_title_moves_up():
    for name in ("noise_overview", "spectral", "beam"):
        fig = _figures()[name][0]
        first_top = fig.layout.yaxis.domain[1]
        titles = [
            a for a in fig.layout.annotations if a.yref == "paper" and abs(a.y - first_top) < 1e-6
        ]
        assert len(titles) == (0 if name == "noise_overview" else 1), name
        assert all(a.yshift == sp._TOP_AXIS_PX for a in titles), name
        others = [a for a in fig.layout.annotations if a not in titles]
        assert all(not a.yshift for a in others), name


def test_band_spectrum_frequency_axis():
    spec = {
        "A": {
            "s_int": np.zeros(N),
            "peak": np.zeros(N),
            "rms": np.zeros(N),
            "sigma_int": 1e-3,
            "colour": "#2E75B6",
        }
    }
    fig = sp.fig_band_spectrum(FREQ, VEL, spec)
    top = fig.layout.xaxis3
    assert (
        top.title.text == "Frequency (MHz)"
        and top.side == "top"
        and top.matches == "x"
        and len(top.ticktext) == 5
    )
    freq_at = sp.HI_REST_FREQ_HZ * (1.0 - np.asarray(top.tickvals) / sp.C_KMS)
    assert np.allclose([float(t) for t in top.ticktext], freq_at / 1e6, atol=0.051)
    assert not [
        a
        for a in fig.layout.annotations
        if a.text == "Frequency (MHz)" or (a.yref == "paper" and a.y > 1.0)
    ]
    assert fig.layout.xaxis2.title.text == sp.VEL_AXIS_TITLE


def test_histogram_medians_in_legend():
    rms = {
        "A": (np.full(N, 1e-3) + 1e-5 * CHAN, "#2E75B6", np.array(["Good"] * N, dtype=object)),
        "B": (np.full(N, 1.01e-3) + 1e-5 * CHAN, "#E04B3A", np.array(["Good"] * N, dtype=object)),
    }
    fig = sp.fig_rms_histogram(rms)
    assert [t.name for t in fig.data] == ["A (median 1.195)", "B (median 1.205)"]
    assert not fig.layout.annotations and len(fig.layout.shapes) == 2


def test_titles_have_no_html_entities():
    for name, (fig, _) in _figures().items():
        assert "&" not in fig.layout.title.text, name
    assert (
        "σ"
        in sp.fig_nhi_sensitivity(
            FREQ, VEL, {"A": {"nhi_chan": np.ones(N), "colour": "#2E75B6"}}
        ).layout.title.text
    )


def test_ticks_point_outwards():
    for axis in (sp.PLOTLY_TEMPLATE.layout.xaxis, sp.PLOTLY_TEMPLATE.layout.yaxis):
        assert axis.ticks == "outside" and axis.ticklen >= 6
