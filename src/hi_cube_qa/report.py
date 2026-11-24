"""Assemble the self-contained interactive HTML QA report.

One report per session. The session is organised by pointing; each pointing
contains one block per cube_N "unit" (its dirty/image/residual products
overlaid on shared axes), plus a session-level flagging section.
"""

from __future__ import annotations

import os
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

COL_DIRTY = "#2E75B6"
COL_TEXT = "#E0E0E0"
COL_BG = "#0F1117"
COL_GRID = "#2A2D35"
COL_WARN = "#F39C12"
COL_BAD = "#C0392B"
COL_GOOD = "#27AE60"


# Relabels the top (channel index) axis of a figure after a zoom or reset, so that
# its ticks stay at round channel numbers within the visible velocity range.
_CHANNEL_AXIS_SCRIPT = """
(function () {
  var gd = document.getElementById('{plot_id}');
  var perAxis = 8;
  function niceStep(span) {
    var raw = Math.max(span / perAxis, 1);
    var power = Math.pow(10, Math.floor(Math.log10(raw)));
    var mult = [1, 2, 5, 10];
    for (var i = 0; i < mult.length; i++) { if (mult[i] * power >= raw) { return mult[i] * power; } }
    return 10 * power;
  }
  function refresh() {
    var meta = gd.layout.meta;
    if (!meta || !meta.chan_axes) { return; }
    var update = {};
    meta.chan_axes.forEach(function (a) {
      var range = gd._fullLayout[a.ref].range;
      var c0 = (range[0] - a.v0) / a.dv, c1 = (range[1] - a.v0) / a.dv;
      var lo = Math.max(0, Math.ceil(Math.min(c0, c1))), hi = Math.min(a.n - 1, Math.floor(Math.max(c0, c1)));
      var vals = [], text = [];
      if (hi >= lo) {
        var step = niceStep(hi - lo);
        for (var c = Math.ceil(lo / step) * step; c <= hi; c += step) { vals.push(a.v0 + c * a.dv); text.push(String(c)); }
      }
      update[a.axis + '.tickvals'] = vals;
      update[a.axis + '.ticktext'] = text;
    });
    Plotly.relayout(gd, update);
  }
  gd.on('plotly_relayout', function (event) {
    if (Object.keys(event).some(function (k) { return /^xaxis\\d*\\.(range|autorange)/.test(k); })) { refresh(); }
  });
})();
"""


class _FigRenderer:
    """Render Plotly figures to HTML divs, injecting the CDN script only once."""

    def __init__(self):
        self._first = True

    def __call__(self, fig: go.Figure, div_id: str) -> str:
        inc = "cdn" if self._first else False
        self._first = False
        return pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=inc,
            div_id=div_id,
            config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
            post_script=_CHANNEL_AXIS_SCRIPT,
        )


def _checklist_html(checklist: list[dict], title: str = "QA Checklist") -> str:
    """Render the checklist as an HTML table grouped by cube label."""
    if not checklist:
        return ""

    by_label: dict[str, list[dict]] = {}
    for row in checklist:
        by_label.setdefault(row["cube_label"], []).append(row)

    blocks = []
    for label, rows in by_label.items():
        pass_n = sum(1 for r in rows if r["status"] == "Pass")
        warn_n = sum(1 for r in rows if r["status"] == "Warn")
        fail_n = sum(1 for r in rows if r["status"] == "Fail")

        tr_rows = ""
        for r in rows:
            if r["status"] == "Pass":
                icon, colour, bg = "P", COL_GOOD, "rgba(39,174,96,0.12)"
            elif r["status"] == "Warn":
                icon, colour, bg = "W", COL_WARN, "rgba(243,156,18,0.12)"
            elif r["status"] == "Fail":
                icon, colour, bg = "F", COL_BAD, "rgba(192,57,43,0.12)"
            else:
                icon, colour, bg = "-", "#888", "transparent"

            note_td = f"<td style='color:#aaa;font-size:0.85em;'>{r.get('note', '')}</td>"
            tr_rows += (
                f"<tr style='background:{bg};'>"
                f"<td style='color:{colour};font-weight:700;text-align:center;'>{icon}</td>"
                f"<td style='color:{colour};font-weight:600;'>{r['status']}</td>"
                f"<td>{r['criterion']}</td>"
                f"<td style='color:#ccc;font-size:0.88em;'>{r['description']}</td>"
                f"<td style='text-align:right;color:{COL_TEXT};'>{r['value']}</td>"
                f"{note_td}"
                f"</tr>\n"
            )

        blocks.append(f"""
        <div style='margin-bottom:18px;'>
          <h5 style='color:{COL_DIRTY};margin:0 0 6px 0;font-family:Arial;'>
            {label}  &nbsp;
            <span style='font-size:0.8em;color:{COL_GOOD};'>&#10004; {pass_n} Pass</span> &nbsp;
            <span style='font-size:0.8em;color:{COL_WARN};'>&#9888; {warn_n} Warn</span> &nbsp;
            <span style='font-size:0.8em;color:{COL_BAD};'>&#10008; {fail_n} Fail</span>
          </h5>
          <table style='border-collapse:collapse;width:100%;font-family:Arial;font-size:13px;'>
            <thead>
              <tr style='background:#1E212B;'>
                <th style='width:32px;'></th>
                <th style='text-align:left;padding:6px 10px;color:#7EC8E3;'>Status</th>
                <th style='text-align:left;padding:6px 10px;color:#7EC8E3;'>Criterion</th>
                <th style='text-align:left;padding:6px 10px;color:#7EC8E3;'>Description</th>
                <th style='text-align:right;padding:6px 10px;color:#7EC8E3;'>Measured Value</th>
                <th style='text-align:left;padding:6px 10px;color:#7EC8E3;'>Note</th>
              </tr>
            </thead>
            <tbody>{tr_rows}</tbody>
          </table>
        </div>
        """)

    return (
        f"<h4 style='color:#7EC8E3;font-family:Arial;margin:4px 0 10px 0;'>{title}</h4>"
        + "".join(blocks)
    )


# Figure ids rendered inside each unit, in display order.
_UNIT_FIG_ORDER = [
    ("summary", "Summary"),
    ("rms_curve", "Per-channel RMS"),
    ("rms_vs_theory", "RMS / theory"),
    ("cumulative_rms", "Cumulative RMS"),
    ("rms_histogram", "RMS histogram"),
    ("noise_overview", "Noise overview"),
    ("kurtosis", "Non-Gaussianity (kurtosis)"),
    ("baseline", "Baseline flatness"),
    ("large_scale", "Large-scale noise"),
    ("spectral", "Spectral profile"),
    ("bandspec", "Band spectrum"),
    ("nhi", "N_HI sensitivity"),
    ("beam", "Beam stats"),
]


def _provenance_html(unit: dict) -> str:
    """Small provenance strip: how the cube was made + a PB-correction warning."""
    p = unit.get("provenance") or {}
    bits = []
    if unit.get("pb_corrected"):
        bits.append(
            "<span style='color:#F39C12;font-weight:600;'>&#9888; "
            "PB-corrected: noise is position-dependent, the single-number "
            "RMS is only indicative</span>"
        )
    chips = []

    def chip(k, v):
        if v not in (None, ""):
            chips.append(f"<span class='chip'><b>{k}</b> {v}</span>")

    if p.get("robust") is not None:
        chip("robust", f"{p['robust']:+.1f}")
    chip("N_ant", f"{unit.get('n_ant', '')} ({unit.get('n_ant_src', '')})")
    if unit.get("t_total") is not None:
        chip("on-source", f"{unit['t_total']:.2f} h ({unit.get('t_total_src', '')})")
    chip("nfac", f"{unit.get('nfac', '')} ({unit.get('nfac_src', '')})")
    if unit.get("flag_fraction") is not None:
        chip("flagged frac", f"{unit['flag_fraction']:.2f}")
    if p.get("niter") is not None:
        chip("niter", f"{p['niter']:,}")
    chip("restoring beam", p.get("restoring_beam"))
    chip("telescope", p.get("telescope"))
    chip("object", p.get("object"))
    chip("date", p.get("date_obs"))
    chip("BUNIT", p.get("bunit"))
    if not chips and not bits:
        return ""
    warn = f"<div style='margin-bottom:4px;'>{' '.join(bits)}</div>" if bits else ""
    return f"<div class='prov'>{warn}{' '.join(chips)}</div>"


def _unit_section(render: _FigRenderer, unit: dict) -> str:
    """Render one (pointing, cube_N) unit block: provenance + figures + maps."""
    anchor = unit["anchor"]
    figures = unit["figures"]
    products = ", ".join(unit["products"])

    parts = [
        f'<section id="{anchor}" class="unit">',
        f"<h3 class='unit-title'>{unit['pointing']} &nbsp;/&nbsp; "
        f"{unit['cube_id']} "
        f"<span class='unit-meta'>[{products}] &middot; {unit['nchan']} ch &middot; "
        f"{unit['freq_lo']:.1f}&ndash;{unit['freq_hi']:.1f} MHz</span></h3>",
        _provenance_html(unit),
    ]

    for fid, _name in _UNIT_FIG_ORDER:
        fig = figures.get(fid)
        if fig is not None:
            parts.append(render(fig, f"fig_{anchor}_{fid}"))

    if unit.get("map_sections"):
        parts.append("<div class='maps'>" + "\n".join(unit["map_sections"]) + "</div>")

    if unit.get("checklist"):
        parts.append(
            "<div class='checklist'>"
            + _checklist_html(unit["checklist"], title="Checklist")
            + "</div>"
        )

    if unit.get("thumb_sections"):
        parts.append("<div class='thumbs'>" + "\n".join(unit["thumb_sections"]) + "</div>")

    parts.append("</section>")
    return "\n".join(parts)


def build_session_report(
    out_path: Path,
    cfg,
    obsid: str,
    units: list[dict],
    flag_fig,
    radiometer: dict,
    flag_fraction: float = 0.0,
) -> float:
    """Assemble and write one session's HTML QA report.

    Parameters
    ----------
    units : list of per-unit bundles (see main.assess_unit), in display order.
    flag_fig : a single session-level flagging Plotly figure, or None.

    Returns the file size in MB.
    """
    render = _FigRenderer()

    # Navigation grouped by pointing
    by_pointing: dict[str, list[dict]] = {}
    for u in units:
        by_pointing.setdefault(u["pointing"], []).append(u)

    nav_groups = []
    for pointing, plist in by_pointing.items():
        links = " ".join(f'<a href="#{u["anchor"]}">{u["cube_id"]}</a>' for u in plist)
        nav_groups.append(f'<span class="nav-group"><b>{pointing}</b> {links}</span>')
    if flag_fig is not None:
        nav_groups.append('<span class="nav-group"><a href="#flags">Flagging</a></span>')
    nav_html = "\n".join(nav_groups)

    # Radiometer note
    rad = radiometer
    if rad["sefd"]:
        noise_src = f"SEFD = {rad['sefd']} Jy"
    else:
        noise_src = f"T<sub>sys</sub> = {rad['tsys']} K, &eta;={rad['eta']}"
    sefd_curve = rad.get("sefd_curve", "default")
    flat = rad.get("sefd") is not None or sefd_curve == "flat"
    curve_note = "flat SEFD" if flat else "freq-dependent SEFD"
    t_note = (
        f"{rad['t_total']} h"
        if not isinstance(rad["t_total"], str)
        else "from obsinfo, per pointing"
    )
    theory_note = (
        f"{noise_src} &nbsp; n<sub>fac</sub> = {rad['nfac']} &nbsp; "
        f"N<sub>ant</sub> = {rad['n_ant']} &nbsp; "
        f"T<sub>on</sub> = {t_note} &nbsp; "
        f"flagged frac &asymp; {flag_fraction:.2f} (per pointing) &nbsp; ({curve_note}; "
        f"&sigma;<sub>theory</sub>(&nu;) per cube, flagging-reduced)"
    )

    # Session overview table (one row per unit)
    overview_rows = ""
    for u in units:
        overview_rows += (
            f"<tr>"
            f"<td><a href='#{u['anchor']}'>{u['pointing']} / {u['cube_id']}</a></td>"
            f"<td>{', '.join(u['products'])}</td>"
            f"<td>{u['nchan']} ch</td>"
            f"<td>{u['freq_lo']:.1f} - {u['freq_hi']:.1f} MHz</td>"
            f"<td>{u.get('overall', '-')}</td>"
            f"</tr>\n"
        )

    # Unit sections
    sections = [_unit_section(render, u) for u in units]

    if flag_fig is not None:
        sections.append(
            f'<section id="flags"><h3 class="unit-title">Flagging Summary '
            f'<span class="unit-meta">(session {obsid})</span></h3>'
            f"{render(flag_fig, 'fig_flags')}</section>"
        )

    css = f"""
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ margin:0; padding:0; background:{COL_BG}; color:{COL_TEXT};
            font-family:'Inter',Arial,sans-serif; font-size:14px; line-height:1.5; }}
    #topbar {{ position:sticky; top:0; z-index:100; background:#13161F;
                border-bottom:2px solid {COL_DIRTY}; padding:10px 24px; }}
    #topbar h1 {{ margin:0 0 6px 0; font-size:1.1em; color:{COL_TEXT}; }}
    #topbar .nav {{ display:flex; gap:14px; flex-wrap:wrap; font-size:0.85em; }}
    .nav-group {{ background:#1E212B; padding:3px 10px; border-radius:4px; }}
    .nav-group a {{ color:#7EC8E3; text-decoration:none; margin:0 3px; }}
    .nav-group a:hover {{ text-decoration:underline; }}
    .container {{ max-width:1400px; margin:0 auto; padding:24px 20px; }}
    .info-box {{ background:#13161F; border:1px solid {COL_GRID};
                 border-left:4px solid {COL_DIRTY}; border-radius:6px;
                 padding:14px 18px; margin-bottom:24px; }}
    .info-box h2 {{ margin:0 0 8px 0; font-size:1.0em; color:{COL_DIRTY}; }}
    .info-table {{ border-collapse:collapse; width:100%; font-size:0.88em; }}
    .info-table th, .info-table td {{ padding:5px 14px; border:1px solid {COL_GRID}; text-align:left; }}
    .info-table th {{ background:#1E212B; color:#7EC8E3; }}
    .info-table tr:nth-child(even) td {{ background:#13161F; }}
    section.unit {{ margin:0 0 14px 0; padding:16px 18px; scroll-margin-top:96px;
                    background:#0C0E14; border:1px solid {COL_GRID}; border-radius:8px; }}
    .unit-title {{ margin:0 0 12px 0; color:#7EC8E3; font-size:1.15em;
                   border-bottom:1px solid {COL_GRID}; padding-bottom:6px; }}
    .unit-meta {{ font-size:0.7em; color:#888; font-weight:normal; }}
    .prov {{ font-size:0.78em; color:#9aa; margin:0 0 12px 0; }}
    .prov .chip {{ display:inline-block; background:#13161F; border:1px solid {COL_GRID};
                   border-radius:4px; padding:1px 8px; margin:2px 4px 2px 0; }}
    .prov .chip b {{ color:#7EC8E3; font-weight:600; }}
    .checklist {{ margin:12px 0; }}
    .legend-note {{ font-size:0.82em; color:#aaa; margin:6px 0 0 0; }}
    footer {{ text-align:center; padding:20px; color:#555; font-size:0.8em;
              border-top:1px solid {COL_GRID}; margin-top:40px; }}
    """

    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>HI Cube QA &mdash; {obsid}</title>
  <style>{css}</style>
</head>
<body>
<div id="topbar">
  <h1>&#128225; HI Cube QA &nbsp;|&nbsp; session {obsid}</h1>
  <div class="nav">{nav_html}</div>
</div>
<div class="container">
  <div class="info-box">
    <h2>Session {obsid} &mdash; {len(units)} cube unit(s), {len(by_pointing)} pointing(s)</h2>
    <table class="info-table">
      <tr><th>Pointing / cube</th><th>Products</th><th>Channels</th>
          <th>Frequency range</th><th>Overall</th></tr>
      {overview_rows}
    </table>
    <div class="legend-note">{theory_note}</div>
  </div>

  {"".join(sections)}
</div>
<footer>
  Generated by hi_cube_qa &nbsp;|&nbsp;
  GAMA G23 MeerKAT Data Reduction using CARACal &nbsp;|&nbsp;
  <a href="https://caracal.readthedocs.io" style="color:#7EC8E3;">CARACal</a>
</footer>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    return os.path.getsize(out_path) / 1e6


# Named HTML entities used anywhere in the figures (titles, table cells,
# annotations). A real browser resolves these fine for the interactive HTML
# report, but Kaleido's bundled Chromium (used only for static PNG export)
# does not reliably decode them and fails the whole render - so we swap them
# for the literal Unicode character in a throwaway COPY used only for the PNG,
# never touching the figure object used for the HTML report.
_KALEIDO_ENTITY_MAP = {
    "&sigma;": "σ",
    "&Sigma;": "Σ",
    "&eta;": "η",
    "&Eta;": "Η",
    "&mdash;": "—",
    "&ndash;": "–",
    "&deg;": "°",
    "&times;": "×",
    "&plusmn;": "±",
    "&asymp;": "≈",
    "&middot;": "·",
}


def _sanitize_for_kaleido(fig: go.Figure) -> go.Figure:
    """Return a deep copy of *fig* with named HTML entities resolved to literal
    Unicode characters, for static PNG export only (see _KALEIDO_ENTITY_MAP)."""
    import re

    pattern = re.compile("|".join(re.escape(k) for k in _KALEIDO_ENTITY_MAP))
    if not pattern.search(str(fig.to_dict())):
        return fig  # nothing to do; avoid an unnecessary deep copy

    def repl(s: str) -> str:
        return pattern.sub(lambda m: _KALEIDO_ENTITY_MAP[m.group()], s)

    def walk(obj):
        """Mutate dicts/lists in place, replacing entities in every string."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    obj[k] = repl(v)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    obj[i] = repl(v)
                elif isinstance(v, (dict, list)):
                    walk(v)

    d = fig.to_dict()
    walk(d)
    return go.Figure(d)


def png_export_available() -> tuple[bool, str]:
    """One-time functional check that Plotly can actually render a PNG (needs
    the 'kaleido' package, and a working Chromium under it - kaleido can be
    installed but still fail in sandboxed HPC/Singularity environments, so a
    plain `import kaleido` is not sufficient). Writes a throwaway image to a
    temp file; never touches the real output directory.
    """
    import tempfile

    try:
        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
        with tempfile.TemporaryDirectory() as td:
            fig.write_image(str(Path(td) / "probe.png"))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def export_figs_png(figures, out_dir, prefix, fig_ids, scale=2):
    """Save selected Plotly figures as standalone PNGs (needs the 'kaleido'
    package). Returns (n_saved, ok, err). If kaleido is unavailable the first
    write raises; we stop and return ok=False so the caller can warn once and
    disable PNG export for the rest of the run.

    The output directory is only created once a write actually succeeds, so a
    total failure (e.g. kaleido missing) never leaves a stray empty folder.
    """
    out_dir = Path(out_dir)
    made_dir = False
    n = 0
    for fid in fig_ids:
        fig = figures.get(fid)
        if fig is None:
            continue
        if not made_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            made_dir = True
        try:
            _sanitize_for_kaleido(fig).write_image(
                str(out_dir / f"{prefix}_{fid}.png"), scale=scale
            )
        except Exception as exc:
            return n, False, str(exc)
        n += 1
    return n, True, ""


def build_master_index(
    out_path: Path,
    summaries: list[dict],
    title: str = "MeerKAT HI Cube QA",
) -> float:
    """Write a small master "shell" page that frames each per-session report.

    A session sidebar drives an <iframe>; clicking a session loads its existing
    <obsid>_qa_report.html on demand (so the master stays tiny and fast, and
    each session is only loaded when viewed). The per-session reports must live
    in the same directory as this file (they are referenced by relative name).

    Parameters
    ----------
    summaries : list of per-session dicts with keys
        obsid, html (basename), n_pointings, n_units, products, overall.

    Returns the file size in MB.
    """
    grade_col = {"Good": COL_GOOD, "Warn": COL_WARN, "Bad": COL_BAD}

    items = ""
    first_src = ""
    for i, s in enumerate(summaries):
        src = s.get("html") or ""
        if i == 0:
            first_src = src
        gcol = grade_col.get(s.get("overall", "-"), "#888")
        prods = s.get("products", "")
        if isinstance(prods, (list, tuple)):
            prods = ", ".join(prods)
        items += (
            f'<li class="sess{" active" if i == 0 else ""}" data-src="{src}">'
            f'<div class="sess-id">{s["obsid"]}</div>'
            f'<div class="sess-meta">{s["n_pointings"]} pointings &middot; '
            f"{s['n_units']} units</div>"
            f'<div class="sess-meta sess-prod">{prods}</div>'
            f'<span class="grade" style="background:{gcol};">{s.get("overall", "-")}</span>'
            f"</li>\n"
        )

    css = f"""
    html, body {{ height:100%; margin:0; }}
    body {{ background:{COL_BG}; color:{COL_TEXT};
            font-family:'Inter',Arial,sans-serif; display:flex; }}
    #sidebar {{ width:270px; min-width:270px; height:100vh; overflow-y:auto;
                background:#13161F; border-right:2px solid {COL_DIRTY};
                display:flex; flex-direction:column; }}
    #sidebar h1 {{ font-size:1.05em; margin:0; padding:16px 18px 4px 18px; }}
    #sidebar .sub {{ font-size:0.78em; color:#8aa; padding:0 18px 12px 18px;
                     border-bottom:1px solid {COL_GRID}; }}
    ul {{ list-style:none; margin:0; padding:8px; flex:1; }}
    li.sess {{ position:relative; padding:10px 12px; margin:4px 2px;
               border-radius:6px; cursor:pointer; background:#1A1D26;
               border:1px solid transparent; }}
    li.sess:hover {{ border-color:{COL_DIRTY}; }}
    li.sess.active {{ background:#222838; border-color:{COL_DIRTY}; }}
    .sess-id {{ font-weight:600; font-size:0.92em; color:#cfe6f5; }}
    .sess-meta {{ font-size:0.76em; color:#9aa; margin-top:2px; }}
    .sess-prod {{ color:#7a8; }}
    .grade {{ position:absolute; top:10px; right:10px; font-size:0.68em;
              font-weight:700; color:#fff; padding:1px 7px; border-radius:10px; }}
    #sidebar .foot {{ font-size:0.72em; color:#556; padding:12px 18px;
                      border-top:1px solid {COL_GRID}; }}
    #viewer {{ flex:1; height:100vh; border:0; background:{COL_BG}; }}
    """

    js = """
    var items = document.querySelectorAll('li.sess');
    var viewer = document.getElementById('viewer');
    items.forEach(function(li){
      li.addEventListener('click', function(){
        var src = li.getAttribute('data-src');
        if (!src) return;
        viewer.src = src;
        items.forEach(function(x){ x.classList.remove('active'); });
        li.classList.add('active');
      });
    });
    """

    n = len(summaries)
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <nav id="sidebar">
    <h1>&#128225; {title}</h1>
    <div class="sub">{n} session{"s" if n != 1 else ""} &middot; click to open</div>
    <ul>
      {items}
    </ul>
    <div class="foot">GAMA G23 &middot; CARACal &middot; one report per session</div>
  </nav>
  <iframe id="viewer" src="{first_src}" title="session report"></iframe>
  <script>{js}</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    return os.path.getsize(out_path) / 1e6
