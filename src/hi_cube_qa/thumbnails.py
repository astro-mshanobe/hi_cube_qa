"""Channel-map thumbnails and 2-D diagnostic maps.

Channel selection is done upstream (reduction.py); here we only render the
handful of planes it picked, plus the per-pixel noise map and continuum
persistence map, as base-64 PNGs embedded in HTML.
"""

from __future__ import annotations

import base64
import io
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

COL_TEXT = "#E0E0E0"
COL_GRID = "#2A2D35"
COL_WARN = "#F39C12"
COL_BAD = "#C0392B"
COL_BG = "#0F1117"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout(pad=0.4)
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _channel_thumbnail(
    plane,
    freq_hz,
    vel_kms,
    chan_idx,
    reasons,
    cmap="RdBu_r",
    rms=None,
    beam=None,
    vmin=None,
    vmax=None,
    quality="Good",
) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=150)
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)
    if vmin is None and rms is not None:
        vmin = -3 * rms
    if vmax is None and rms is not None:
        vmax = 5 * rms
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(plane, origin="lower", cmap=cmap, norm=norm, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=COL_TEXT, labelsize=6)
    cbar.set_label("Jy/beam", color=COL_TEXT, fontsize=6)
    if beam is not None:
        bmaj, bmin, bpa, cell = beam
        if cell > 0 and np.isfinite(bmaj) and np.isfinite(bmin):
            ny, nx = plane.shape
            e = Ellipse(
                xy=(nx * 0.10, ny * 0.10),
                width=bmin / cell,
                height=bmaj / cell,
                angle=-bpa,
                edgecolor="yellow",
                facecolor="None",
                lw=1.0,
            )
            ax.add_patch(e)
    badge = {"Good": "P", "Warn": "W", "Bad": "B"}.get(quality, "")
    ax.set_title(
        f"Ch {chan_idx} {badge}  [{', '.join(reasons)}]\n"
        f"{freq_hz / 1e6:.2f} MHz | v = {vel_kms:.0f} km/s",
        color=COL_TEXT,
        fontsize=6.5,
        pad=3,
    )
    ax.tick_params(colors=COL_TEXT, labelsize=6)
    for sp_ in ax.spines.values():
        sp_.set_edgecolor(COL_GRID)
    return _fig_to_b64(fig)


def build_thumbnail_html(
    label, thumb_planes, thumb_reasons, freq_hz, vel_kms, rms_arr, quality, bmaj, bmin, bpa, hdr
) -> str:
    """Render the pre-selected diagnostic channel planes as an HTML grid."""
    if not thumb_planes:
        return (
            f"<h4 style='color:{COL_TEXT};font-family:Arial;'>"
            f"Channel maps - {label} (no finite channels)</h4>"
        )
    cell_asec = abs(hdr.get("CDELT1", 0) * 3600.0) or 6.0
    med_rms = float(np.nanmedian(rms_arr))
    vmin, vmax = -3 * med_rms, 5 * med_rms

    parts = [
        f"<h4 style='color:{COL_TEXT};font-family:Arial;margin:14px 0 6px 0;'>"
        f"Channel maps - {label}"
        f"  <span style='font-size:0.75em;color:#aaa;'>"
        f"({len(thumb_planes)} diagnostic channels)</span></h4>",
        "<div style='display:flex;flex-wrap:wrap;gap:6px;background:#0F1117;"
        "padding:10px;border-radius:6px;'>",
    ]
    for ch in sorted(thumb_planes):
        reasons = thumb_reasons.get(ch, [])
        beam = (bmaj[ch], bmin[ch], bpa[ch], cell_asec) if bmaj is not None else None
        q = str(quality[ch]) if ch < len(quality) else "Good"
        b64 = _channel_thumbnail(
            thumb_planes[ch],
            freq_hz[ch],
            vel_kms[ch],
            ch,
            reasons,
            rms=rms_arr[ch],
            beam=beam,
            vmin=vmin,
            vmax=vmax,
            quality=q,
        )
        if "peak SNR" in reasons:
            border = f"3px solid {COL_WARN}"
        elif q == "Bad":
            border = f"2px solid {COL_BAD}"
        elif q == "Warn":
            border = f"2px solid {COL_WARN}"
        else:
            border = "1px solid #2A2D35"
        parts.append(
            f"<div style='border:{border};border-radius:4px;' "
            f"title='Ch {ch} [{', '.join(reasons)}] - {q}'>"
            f"<img src='data:image/png;base64,{b64}' style='display:block;'/></div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _map_png(arr, title, label_txt, cmap, vmin=None, vmax=None) -> str:
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=150)
    fig.patch.set_facecolor(COL_BG)
    ax.set_facecolor(COL_BG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if vmin is None:
            vmin = np.nanpercentile(arr, 2)
        if vmax is None:
            vmax = np.nanpercentile(arr, 98)
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=COL_TEXT, labelsize=6)
    cbar.set_label(label_txt, color=COL_TEXT, fontsize=6)
    ax.set_title(title, color=COL_TEXT, fontsize=8, pad=3)
    ax.tick_params(colors=COL_TEXT, labelsize=6)
    for sp_ in ax.spines.values():
        sp_.set_edgecolor(COL_GRID)
    return _fig_to_b64(fig)


def build_map_html(label, noise_map, persist_map, cont_metrics) -> str:
    """Render the per-pixel noise map and continuum-persistence map."""
    if noise_map is None and persist_map is None:
        return ""
    imgs = []
    if noise_map is not None:
        imgs.append(
            _map_png(
                noise_map * 1e3, "Per-pixel noise (std across channels)", "mJy/beam", "inferno"
            )
        )
    if persist_map is not None:
        imgs.append(
            _map_png(
                persist_map * 100.0,
                "Continuum persistence (% of channels)",
                "% channels",
                "magma",
                vmin=0,
                vmax=100,
            )
        )
    cont = ""
    if cont_metrics and cont_metrics.get("available"):
        cont = (
            f"<div style='font-size:0.8em;color:#aaa;margin-top:4px;'>"
            f"Pixels persistent &gt;50% of channels at "
            f"{cont_metrics['cont_sigma']:.0f}&sigma;: "
            f"<b>{cont_metrics['pct_pixels_persistent']:.2f}%</b> "
            f"(continuum-residual indicator)</div>"
        )
    grid = "".join(
        f"<div style='border:1px solid #2A2D35;border-radius:4px;'>"
        f"<img src='data:image/png;base64,{b}' style='display:block;'/></div>"
        for b in imgs
    )
    return (
        f"<h4 style='color:{COL_TEXT};font-family:Arial;margin:14px 0 6px 0;'>"
        f"Spatial diagnostic maps - {label}</h4>"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;background:#0F1117;"
        f"padding:10px;border-radius:6px;'>{grid}</div>{cont}"
    )
