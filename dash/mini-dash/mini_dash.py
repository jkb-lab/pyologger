"""
mini-dash — a stripped-down interactive viewer for one deployment.

Features (only these, kept intentionally minimal):
  - One unified timeline: a single playhead drives the plot line, the video, and
    the coverage strip. Drag it, click the plot, or use the transport controls.
  - Signal plot with per-signal color editing and drag-to-reorder ordering.
  - Synchronized Immich video (proxied server-side; local-file fallback).
  - Transport: play / pause and step by ±0.1 s or ±10 s.

Launch (from pyologger/):
    python dash/mini-dash/mini_dash.py                       # default deployment
    python dash/mini-dash/mini_dash.py --dataset <id> --deployment <id> --port 8070

Standalone: depends only on pyologger utils + DiveDB (for Immich), not on the
big integrated_dash app.
"""

import argparse
import os
import pathlib
import re

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, ctx, no_update
import dash

from pyologger.utils.folder_manager import load_configuration, resolve_deployment_context
from pyologger.utils.deployment_source import (
    resolve_deployment_source,
    resolve_plot_signal_allowlist,
)

# --------------------------------------------------------------------------- #
# Config parameters (the key knobs — edit here or pass via CLI)
# --------------------------------------------------------------------------- #
DEFAULT_DATASET = "oror-adult-orca_hr-sr-vid_sw_JKB-PP"
DEFAULT_DEPLOYMENT = "2023-06-23_oror-002"
DEFAULT_PORT = 8070
WINDOW_MINUTES = 5          # initial view window around the data start
TARGET_HZ = 10.0            # downsample target for plotting (keeps it snappy)
STEP_SMALL = 0.1            # seconds — fine step
STEP_LARGE = 10.0          # seconds — coarse step
PLAYBACK_RATES = [0.5, 1, 2, 5]

_PALETTE = [
    "#4C9BE8", "#E8794C", "#5FBF77", "#C766D6", "#E8C84C",
    "#4CD6C7", "#E85C8A", "#9B8CFF", "#8CCf5F", "#FF9F4C",
]

# --------------------------------------------------------------------------- #
# CLI + deployment load
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="mini-dash interactive viewer")
parser.add_argument("--dataset", default=DEFAULT_DATASET)
parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
parser.add_argument("--port", type=int, default=DEFAULT_PORT)
parser.add_argument("--source", choices=["auto", "immich", "local"], default="auto",
                    help="video source: auto (immich then local), or force immich/local")
args = parser.parse_args()

config, data_dir, color_mapping_path, _ = load_configuration()
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = (
    resolve_deployment_context(data_dir, dataset_id=args.dataset, deployment_id=args.deployment)
)
source = resolve_deployment_source(data_dir, dataset_id, deployment_id, deployment_folder=deployment_folder)
allowlist = resolve_plot_signal_allowlist(param_manager, source)
shell = source.build_metadata_shell(allowed_signals=allowlist)

TZ = str(shell.deployment_info.get("Time Zone", "UTC") or "UTC")
ALL_SIGNALS = [s for s in source.signal_names() if s != "location"]
DEFAULT_SIGNALS = [s for s in (allowlist or ALL_SIGNALS) if s in ALL_SIGNALS][:6] or ALL_SIGNALS[:6]

_g_start = pd.Timestamp(source.global_start)
_g_end = pd.Timestamp(source.global_end)
if _g_start.tzinfo is None:
    _g_start = _g_start.tz_localize(TZ)
if _g_end.tzinfo is None:
    _g_end = _g_end.tz_localize(TZ)

FULL_MIN = int(_g_start.timestamp())
FULL_MAX = int(_g_end.timestamp())
WIN_LO = FULL_MIN
WIN_HI = min(FULL_MAX, FULL_MIN + WINDOW_MINUTES * 60)
PLAYHEAD0 = float(WIN_LO + (WIN_HI - WIN_LO) / 2)

print(f"[mini-dash] {dataset_id} / {deployment_id}  tz={TZ}  signals={len(ALL_SIGNALS)}")


def _epoch_to_ts(epoch):
    return pd.Timestamp(float(epoch), unit="s", tz="UTC").tz_convert(TZ)


def _fmt(epoch):
    t = _epoch_to_ts(epoch)
    return t.strftime("%H:%M:%S.") + f"{int(t.microsecond/1000):03d}"


def _fmt_full(epoch):
    return _epoch_to_ts(epoch).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
# Signal color assignment (editable in the UI, seeded from the palette)
# --------------------------------------------------------------------------- #
def _default_colors():
    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(ALL_SIGNALS)}


# --------------------------------------------------------------------------- #
# Video: Immich clip index (+ local fallback) and a server-side proxy
# --------------------------------------------------------------------------- #
_immich_service = None


def _ensure_immich_env():
    if os.getenv("IMMICH_API_KEY") and os.getenv("IMMICH_BASE_URL"):
        return
    root = pathlib.Path(__file__).resolve().parents[3]  # repo root
    env_path = root / "EcoPhysVideoViz" / ".env"
    try:
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k in ("IMMICH_API_KEY", "IMMICH_BASE_URL") and not os.getenv(k):
                        os.environ[k] = v
    except Exception:
        pass


def _immich():
    global _immich_service
    if _immich_service is not None:
        return _immich_service
    try:
        _ensure_immich_env()
        from DiveDB.services.immich_service import ImmichService
        _immich_service = ImmichService()
    except Exception as exc:
        print(f"[mini-dash] Immich unavailable: {exc}")
        _immich_service = False
    return _immich_service


def _parse_dur(s):
    try:
        p = str(s).split(":")
        if len(p) == 3:
            return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    except Exception:
        pass
    return 0.0


def _build_immich_clips():
    """Immich album DepID_<deployment> (raw fileCreatedAt = true UTC), else []."""
    svc = _immich()
    if not svc:
        return []
    try:
        res = svc.find_media_by_deployment_id(f"DepID_{deployment_id}", media_type="VIDEO", shared=True)
    except Exception as exc:
        print(f"[mini-dash] Immich find_media failed: {exc}")
        return []
    if not (res and res.get("success")):
        return []
    clips = []
    for a in (res.get("data") or []):
        aid, created = a.get("id"), a.get("fileCreatedAt")
        if not aid or not created:
            continue
        ts = pd.Timestamp(created)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        s0 = float(ts.timestamp())
        clips.append({
            "name": a.get("originalFileName") or aid,
            "start_epoch": s0,
            "end_epoch": s0 + _parse_dur(a.get("duration")),
            "url": f"/mini-video/{aid}",
        })
    clips.sort(key=lambda c: c["start_epoch"])
    return clips


_LOCAL_FILENAME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<s>\d{2}-\d{2}-\d{2})_(?P<e>\d{2}-\d{2}-\d{2})\.mp4$", re.IGNORECASE
)
LOCAL_VIDEO_DIR = None


def _build_local_clips():
    """Local <media>/<dataset>/<deployment>_media/02_processed-video/*.mp4.

    Filenames encode a local-time range; interpreted in the deployment tz.
    """
    global LOCAL_VIDEO_DIR
    media_root = (config.get("paths", {}) or {}).get("local_private_media")
    if not media_root:
        return []
    d = pathlib.Path(media_root) / dataset_id / f"{deployment_id}_media" / "02_processed-video"
    LOCAL_VIDEO_DIR = d
    if not d.is_dir():
        return []
    clips = []
    for pth in sorted(d.glob("*.mp4")):
        m = _LOCAL_FILENAME_RE.search(pth.name)
        if not m:
            continue
        try:
            s = pd.Timestamp(f"{m.group('date')} {m.group('s').replace('-', ':')}").tz_localize(TZ)
            e = pd.Timestamp(f"{m.group('date')} {m.group('e').replace('-', ':')}").tz_localize(TZ)
            if e <= s:
                e = e + pd.Timedelta(days=1)
        except Exception:
            continue
        clips.append({
            "name": pth.name,
            "start_epoch": float(s.timestamp()),
            "end_epoch": float(e.timestamp()),
            "url": f"/mini-local/{pth.name}",
        })
    clips.sort(key=lambda c: c["start_epoch"])
    return clips


def _build_clip_index(prefer):
    """prefer='immich'|'local'|'auto'. auto = immich, then local fallback."""
    if prefer == "local":
        c = _build_local_clips()
        print(f"[mini-dash] {len(c)} LOCAL clip(s) (forced).")
        return c
    if prefer == "immich":
        c = _build_immich_clips()
        print(f"[mini-dash] {len(c)} Immich clip(s) (forced).")
        return c
    imm = _build_immich_clips()
    if imm:
        print(f"[mini-dash] {len(imm)} Immich clip(s).")
        return imm
    loc = _build_local_clips()
    print(f"[mini-dash] {len(loc)} LOCAL clip(s) (Immich empty).")
    return loc


CLIPS = _build_clip_index(args.source)


def _clip_for(epoch):
    for c in CLIPS:
        if c["start_epoch"] <= epoch <= c["end_epoch"]:
            return c, epoch - c["start_epoch"]
    return None, None


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
_win_cache = {}


def _load_window(lo, hi):
    """Load + lightly downsample each signal for [lo, hi]. Cached per (lo,hi)."""
    key = (int(lo), int(hi))
    if key in _win_cache:
        return _win_cache[key]
    start, end = _epoch_to_ts(lo), _epoch_to_ts(hi)
    out = {}
    span = max(1.0, hi - lo)
    target_points = int(span * TARGET_HZ)
    for sig in ALL_SIGNALS:
        try:
            df = source.load_signal_window(sig, start, end)
        except Exception:
            df = None
        if df is None or df.empty or "datetime" not in df.columns:
            continue
        if len(df) > target_points > 0:
            df = df.iloc[:: max(1, len(df) // target_points)]
        out[sig] = df
    if len(_win_cache) > 8:
        _win_cache.clear()
    _win_cache[key] = out
    return out


def build_figure(lo, hi, order, colors, playhead):
    order = [s for s in (order or DEFAULT_SIGNALS) if s in ALL_SIGNALS]
    data = _load_window(lo, hi)
    n = max(1, len(order))
    fig = go.Figure()
    row_h = 1.0 / n
    for i, sig in enumerate(order):
        df = data.get(sig)
        top = 1.0 - i * row_h
        bot = 1.0 - (i + 1) * row_h
        yax = "y" if i == 0 else f"y{i+1}"
        axis_id = "yaxis" if i == 0 else f"yaxis{i+1}"
        color = (colors or {}).get(sig, _PALETTE[i % len(_PALETTE)])
        if df is not None and not df.empty:
            chans = [c for c in df.columns if c != "datetime"]
            for j, ch in enumerate(chans[:3]):
                fig.add_trace(go.Scatter(
                    x=df["datetime"], y=df[ch], mode="lines",
                    line={"color": color, "width": 1.1, "dash": ["solid", "dot", "dash"][j % 3]},
                    name=f"{sig}:{ch}" if len(chans) > 1 else sig, yaxis=yax, showlegend=False,
                ))
        fig.update_layout({axis_id: dict(
            domain=[max(0.0, bot + 0.012), top - 0.012], title=dict(text=sig, font=dict(size=11, color=color)),
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
            tickfont=dict(size=9, color="#9db4c7"),
        )})
    ph_ts = _epoch_to_ts(playhead)
    fig.add_shape(type="line", x0=ph_ts, x1=ph_ts, y0=0, y1=1, yref="paper",
                  line=dict(color="#FFD166", width=2))
    fig.update_layout(
        xaxis=dict(range=[_epoch_to_ts(lo), _epoch_to_ts(hi)], showgrid=True,
                   gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color="#9db4c7")),
        margin=dict(l=60, r=16, t=8, b=28), height=max(260, 120 * n),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0b1f2f", uirevision="keep",
    )
    return fig


# --------------------------------------------------------------------------- #
# App + layout
# --------------------------------------------------------------------------- #
app = Dash(__name__)
app.title = "mini-dash"


def _chip(sig, color, idx):
    return html.Div(
        [
            html.Span(className="chip-grip", children="⠿"),
            dcc.Input(type="color", value=color, id={"type": "sig-color", "sig": sig}, className="chip-color"),
            html.Span(sig, className="chip-label"),
            html.Button("↑", id={"type": "sig-up", "sig": sig}, className="chip-move", n_clicks=0, title="Move up"),
            html.Button("↓", id={"type": "sig-down", "sig": sig}, className="chip-move", n_clicks=0, title="Move down"),
        ],
        className="sig-chip", id={"type": "sig-chip", "sig": sig},
    )


app.layout = html.Div(
    [
        dcc.Store(id="order", data=list(DEFAULT_SIGNALS)),
        dcc.Store(id="colors", data=_default_colors()),
        dcc.Store(id="playhead", data=PLAYHEAD0),
        dcc.Store(id="clip", data=None),
        dcc.Store(id="playing", data=False),
        dcc.Store(id="rate", data=1),
        dcc.Store(id="dummy", data=0),
        dcc.Interval(id="tick", interval=100, disabled=True),
        dcc.Input(id="key", type="text", value="", style={"display": "none"}),

        html.Div([
            html.H1("mini-dash", className="brand"),
            html.Span(f"{deployment_id}  ·  {TZ}", className="sub"),
        ], className="header"),

        html.Div([
            # left: signal order/color editor
            html.Div([
                html.Div("Signals", className="panel-title"),
                html.Div("Color swatch edits color · ↑↓ reorders", className="hint"),
                html.Div([_chip(s, _default_colors()[s], i) for i, s in enumerate(DEFAULT_SIGNALS)],
                         id="chips", className="chips"),
                html.Div("Add signal", className="panel-title", style={"marginTop": "14px"}),
                dcc.Dropdown(id="add-sig", options=[{"label": s, "value": s} for s in ALL_SIGNALS],
                             placeholder="add a signal…", className="add-dd"),
            ], className="side"),

            # right: video + plot + unified timeline
            html.Div([
                html.Div([
                    html.Video(id="video", src="", controls=True, preload="auto", className="video"),
                    html.Div(id="video-status", className="video-status"),
                ], className="video-wrap"),

                # coverage strip (unified timeline) ABOVE the plot
                html.Div([
                    html.Span("Video", className="cov-label"),
                    html.Div(
                        html.Div(
                            [html.Div(className="cov-bar",
                                      style={"--s": c["start_epoch"], "--e": c["end_epoch"]}) for c in CLIPS],
                            id="cov-strip", className="cov-strip",
                            style={"--min": WIN_LO, "--max": WIN_HI},
                        ),
                        id="cov-align", className="cov-align",
                    ),
                ], className="cov-row"),

                dcc.Graph(id="plot", className="plot", config={"displayModeBar": False}),

                # transport + unified playhead
                html.Div([
                    html.Button("⏮ −10s", id="b-back10", className="tbtn", n_clicks=0),
                    html.Button("◂ −0.1s", id="b-back01", className="tbtn", n_clicks=0),
                    html.Button("▶", id="b-play", className="tbtn play", n_clicks=0, **{"data-playing": "0"}),
                    html.Button("+0.1s ▸", id="b-fwd01", className="tbtn", n_clicks=0),
                    html.Button("+10s ⏭", id="b-fwd10", className="tbtn", n_clicks=0),
                    dcc.Dropdown(id="rate-dd", options=[{"label": f"{r}×", "value": r} for r in PLAYBACK_RATES],
                                 value=1, clearable=False, searchable=False, className="rate-dd"),
                    html.Span(id="ph-label", className="ph-label"),
                ], className="transport"),

                html.Label("Playhead", className="ph-title"),
                dcc.Slider(id="ph-slider", min=WIN_LO, max=WIN_HI, step=0.1, value=PLAYHEAD0,
                           marks=None, updatemode="drag", className="ph-slider",
                           tooltip={"always_visible": False}),

                html.Label("Window", className="ph-title"),
                dcc.RangeSlider(id="win-slider", min=FULL_MIN, max=FULL_MAX, step=1,
                                value=[WIN_LO, WIN_HI], marks=None, className="win-slider",
                                tooltip={"always_visible": False}),
            ], className="main"),
        ], className="body"),
    ],
    className="mini-app",
)


# --------------------------------------------------------------------------- #
# Video proxy (Immich) — keeps API key server-side, forwards Range for seeking
# --------------------------------------------------------------------------- #
_ASSET_RE = re.compile(r"^[0-9a-fA-F-]{16,64}$")


@app.server.route("/mini-video/<asset_id>")
def _serve_video(asset_id):
    from flask import Response, abort, request, stream_with_context
    if not _ASSET_RE.match(asset_id or ""):
        abort(404)
    svc = _immich()
    if not svc:
        abort(503)
    fwd = {"x-api-key": svc.api_key}
    if request.headers.get("Range"):
        fwd["Range"] = request.headers["Range"]
    try:
        up = svc.session.get(f"{svc.base_url}/assets/{asset_id}/video/playback",
                             headers=fwd, stream=True, timeout=60)
    except Exception:
        abort(502)
    if up.status_code not in (200, 206):
        up.close()
        abort(502)
    headers = {k: up.headers[k] for k in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges")
               if k in up.headers}
    headers.setdefault("Content-Type", "video/mp4")
    headers.setdefault("Accept-Ranges", "bytes")

    def gen():
        try:
            for chunk in up.iter_content(chunk_size=262144):
                if chunk:
                    yield chunk
        finally:
            up.close()

    return Response(stream_with_context(gen()), status=up.status_code, headers=headers)


@app.server.route("/mini-local/<path:filename>")
def _serve_local(filename):
    """Serve a local mp4 from LOCAL_VIDEO_DIR with Range support (for A/B vs Immich)."""
    from flask import abort, send_file
    if not LOCAL_VIDEO_DIR:
        abort(404)
    safe = os.path.basename(filename)
    if safe != filename or not safe.lower().endswith(".mp4"):
        abort(404)
    target = LOCAL_VIDEO_DIR / safe
    if not target.is_file():
        abort(404)
    return send_file(str(target), conditional=True, mimetype="video/mp4")


# --------------------------------------------------------------------------- #
# Callbacks — signal order / color editing
# --------------------------------------------------------------------------- #
@app.callback(
    Output("order", "data"),
    Input({"type": "sig-up", "sig": dash.ALL}, "n_clicks"),
    Input({"type": "sig-down", "sig": dash.ALL}, "n_clicks"),
    Input("add-sig", "value"),
    State("order", "data"),
    prevent_initial_call=True,
)
def edit_order(_ups, _downs, add_sig, order):
    order = list(order or DEFAULT_SIGNALS)
    trig = ctx.triggered_id
    if trig == "add-sig":
        if add_sig and add_sig not in order:
            order.append(add_sig)
        return order
    if not isinstance(trig, dict) or not ctx.triggered or ctx.triggered[0]["value"] in (None, 0):
        raise dash.exceptions.PreventUpdate
    sig = trig.get("sig")
    if sig not in order:
        raise dash.exceptions.PreventUpdate
    i = order.index(sig)
    if trig["type"] == "sig-up" and i > 0:
        order[i - 1], order[i] = order[i], order[i - 1]
    elif trig["type"] == "sig-down" and i < len(order) - 1:
        order[i + 1], order[i] = order[i], order[i + 1]
    return order


@app.callback(
    Output("colors", "data"),
    Input({"type": "sig-color", "sig": dash.ALL}, "value"),
    State({"type": "sig-color", "sig": dash.ALL}, "id"),
    State("colors", "data"),
    prevent_initial_call=True,
)
def edit_colors(values, ids, colors):
    colors = dict(colors or {})
    for val, cid in zip(values, ids):
        if val:
            colors[cid["sig"]] = val
    return colors


@app.callback(
    Output("chips", "children"),
    Input("order", "data"),
    Input("colors", "data"),
)
def render_chips(order, colors):
    order = [s for s in (order or DEFAULT_SIGNALS) if s in ALL_SIGNALS]
    colors = colors or _default_colors()
    return [_chip(s, colors.get(s, _PALETTE[i % len(_PALETTE)]), i) for i, s in enumerate(order)]


# --------------------------------------------------------------------------- #
# Callbacks — figure
# --------------------------------------------------------------------------- #
@app.callback(
    Output("plot", "figure"),
    Input("order", "data"),
    Input("colors", "data"),
    Input("win-slider", "value"),
    State("playhead", "data"),
    State("playing", "data"),
)
def draw(order, colors, win, playhead, playing):
    # The plot rebuilds on order/color/window changes. The playhead LINE is moved
    # clientside on every tick (no rebuild), so playback/drag stay smooth; the line
    # is drawn at the current playhead here so a rebuild lands it in the right spot.
    lo, hi = int(min(win)), int(max(win))
    ph = max(lo, min(hi, float(playhead if playhead is not None else PLAYHEAD0)))
    return build_figure(lo, hi, order, colors, ph)


# --------------------------------------------------------------------------- #
# Callbacks — unified playhead (slider / plot click -> playhead store)
# --------------------------------------------------------------------------- #
@app.callback(
    Output("playhead", "data", allow_duplicate=True),
    Input("ph-slider", "value"),
    prevent_initial_call=True,
)
def ph_from_slider(v):
    try:
        return float(v)
    except Exception:
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("playhead", "data", allow_duplicate=True),
    Input("plot", "clickData"),
    State("win-slider", "value"),
    prevent_initial_call=True,
)
def ph_from_plot(click, win):
    if not isinstance(click, dict) or not click.get("points"):
        raise dash.exceptions.PreventUpdate
    x = click["points"][0].get("x")
    try:
        ep = pd.Timestamp(x).tz_localize(TZ).timestamp() if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x).timestamp()
    except Exception:
        raise dash.exceptions.PreventUpdate
    return max(int(min(win)), min(int(max(win)), float(ep)))


@app.callback(
    Output("ph-slider", "value"),
    Output("ph-slider", "min"),
    Output("ph-slider", "max"),
    Input("playhead", "data"),
    Input("win-slider", "value"),
)
def slider_from_ph(ph, win):
    lo, hi = int(min(win)), int(max(win))
    v = max(lo, min(hi, float(ph if ph is not None else PLAYHEAD0)))
    return v, lo, hi


@app.callback(Output("ph-label", "children"), Input("playhead", "data"))
def ph_text(ph):
    return f"{_fmt(ph)}  ·  {_fmt_full(ph)}"


# window strip view range follows the window slider
@app.callback(Output("cov-strip", "style"), Input("win-slider", "value"))
def cov_view(win):
    return {"--min": int(min(win)), "--max": int(max(win))}


# Keep the playhead in view: if it leaves the window, slide the window to follow
# (preserving width). This keeps the yellow line + plot centered during play/scrub.
@app.callback(
    Output("win-slider", "value", allow_duplicate=True),
    Input("playhead", "data"),
    State("win-slider", "value"),
    prevent_initial_call=True,
)
def follow_playhead(ph, win):
    lo, hi = float(min(win)), float(max(win))
    p = float(ph if ph is not None else PLAYHEAD0)
    if lo <= p <= hi:
        raise dash.exceptions.PreventUpdate
    width = hi - lo
    if p < lo:
        new_lo = max(FULL_MIN, p - width * 0.1)
        new_hi = new_lo + width
    else:
        new_hi = min(FULL_MAX, p + width * 0.1)
        new_lo = new_hi - width
    new_lo = max(FULL_MIN, new_lo)
    new_hi = min(FULL_MAX, new_lo + width)
    return [int(new_lo), int(new_hi)]


# --------------------------------------------------------------------------- #
# Callbacks — transport (step / play-pause / rate)
# --------------------------------------------------------------------------- #
@app.callback(
    Output("playhead", "data", allow_duplicate=True),
    Input("b-back10", "n_clicks"),
    Input("b-back01", "n_clicks"),
    Input("b-fwd01", "n_clicks"),
    Input("b-fwd10", "n_clicks"),
    Input("key", "value"),
    State("playhead", "data"),
    State("win-slider", "value"),
    prevent_initial_call=True,
)
def step(_b10, _b01, _f01, _f10, key, ph, win):
    trig = ctx.triggered_id
    cur = float(ph if ph is not None else PLAYHEAD0)
    delta = 0.0
    if trig == "b-back10":
        delta = -STEP_LARGE
    elif trig == "b-back01":
        delta = -STEP_SMALL
    elif trig == "b-fwd01":
        delta = STEP_SMALL
    elif trig == "b-fwd10":
        delta = STEP_LARGE
    elif trig == "key":
        if not key:
            raise dash.exceptions.PreventUpdate
        d = str(key).split(":", 1)[0]
        delta = {"L": -STEP_SMALL, "R": STEP_SMALL, "DL": -STEP_LARGE, "DR": STEP_LARGE}.get(d, 0.0)
    if delta == 0.0:
        raise dash.exceptions.PreventUpdate
    return max(int(min(win)), min(int(max(win)), round(cur + delta, 3)))


app.clientside_callback(
    """
    function(n, ph, win, rate, clip) {
        const mgr = window.MiniPlayback, btn = document.getElementById("b-play"), v = document.getElementById("video");
        if (!mgr) return [false, true, "▶"];
        const willPlay = (Number(n) || 0) % 2 === 1;
        mgr.setBounds(win[0], win[1]); mgr.setRate(rate || 1);
        const hasClip = !!(clip && clip.start_epoch != null);
        if (v && !v._wired) {
            v._wired = true;
            v.addEventListener("timeupdate", function(){
                if (v.paused || v.ended || v._clipStart == null) return;
                window.dash_clientside.set_props("playhead", {data: v._clipStart + v.currentTime});
            });
        }
        if (v) v._clipStart = hasClip ? Number(clip.start_epoch) : null;
        if (willPlay) {
            if (btn) btn.setAttribute("data-playing", "1");
            if (hasClip && v) { try { v.playbackRate = Math.max(0.25, Math.min(16, rate||1)); } catch(e){} v.play().catch(function(){}); mgr.stop(); return [true, true, "⏸"]; }
            mgr.sync(ph != null ? ph : win[0]); mgr.start(); return [true, false, "⏸"];
        }
        if (btn) btn.setAttribute("data-playing", "0");
        if (v) { try { v.pause(); } catch(e){} }
        mgr.stop(); return [false, true, "▶"];
    }
    """,
    Output("playing", "data"), Output("tick", "disabled"), Output("b-play", "children"),
    Input("b-play", "n_clicks"),
    State("playhead", "data"), State("win-slider", "value"), State("rate", "data"), State("clip", "data"),
    prevent_initial_call=True,
)

app.clientside_callback(
    "function(n, playing){ if(!playing) return window.dash_clientside.no_update; const m=window.MiniPlayback; return (m&&m.playing&&m.t!=null)?m.t:window.dash_clientside.no_update; }",
    Output("playhead", "data", allow_duplicate=True),
    Input("tick", "n_intervals"), State("playing", "data"),
    prevent_initial_call=True,
)

app.clientside_callback(
    "function(r,win){ const m=window.MiniPlayback; if(m){ m.setRate(r||1); m.setBounds(win[0],win[1]); const v=document.getElementById('video'); if(v){try{v.playbackRate=Math.max(0.25,Math.min(16,r||1));}catch(e){}} } return r||1; }",
    Output("rate", "data"), Input("rate-dd", "value"), Input("win-slider", "value"),
)


# --------------------------------------------------------------------------- #
# Callbacks — video sync (server: pick clip; client: seek + move line)
# --------------------------------------------------------------------------- #
@app.callback(
    Output("video", "src"),
    Output("video-status", "children"),
    Output("clip", "data"),
    Input("playhead", "data"),
    State("clip", "data"),
    prevent_initial_call=False,
)
def video_sync(ph, current):
    if not CLIPS:
        return no_update, "No video clips for this deployment.", None
    clip, offset = _clip_for(float(ph if ph is not None else PLAYHEAD0))
    if clip is None:
        return no_update, f"No video at {_fmt(ph)}", current
    off = int(offset)
    status = f"{clip['name']}  ·  +{off//60}:{off%60:02d}"
    state = {"name": clip["name"], "start_epoch": clip["start_epoch"]}
    if isinstance(current, dict) and current.get("name") == clip["name"]:
        return no_update, status, state
    return clip["url"], status, state


# seek video (when paused) to the playhead, without fighting playback
app.clientside_callback(
    """
    function(ph, clip) {
        if (clip == null || clip.start_epoch == null || ph == null) return window.dash_clientside.no_update;
        const v = document.getElementById("video");
        if (!v) return window.dash_clientside.no_update;
        v._clipStart = Number(clip.start_epoch);
        const btn = document.getElementById("b-play"), playing = btn && btn.getAttribute("data-playing") === "1";
        if (playing) { if (v.paused) { const k=function(){v.play().catch(function(){});}; v.readyState>=2?k():v.addEventListener("canplay",k,{once:true}); } return window.dash_clientside.no_update; }
        if (!v.paused && !v.ended) return window.dash_clientside.no_update;
        const target = Math.max(0, Number(ph) - Number(clip.start_epoch));
        if (!isFinite(target)) return window.dash_clientside.no_update;
        const doSeek = function(){ if (v.seeking) { v._pending = target; return; } if (Math.abs((v.currentTime||0)-target) > 0.08) { try { v.currentTime = target; } catch(e){} } };
        if (!v._seekwired) { v._seekwired = true; v.addEventListener("seeked", function(){ if (v._pending!=null){ const t=v._pending; v._pending=null; if(!v.seeking && Math.abs((v.currentTime||0)-t)>0.08){try{v.currentTime=t;}catch(e){}} } }); }
        if (v.readyState >= 1) doSeek(); else v.addEventListener("loadedmetadata", doSeek, {once:true});
        return window.dash_clientside.no_update;
    }
    """,
    Output("video", "title"),
    Input("playhead", "data"), Input("clip", "data"),
)

# move the yellow line clientside on every playhead change (drag + play)
app.clientside_callback(
    """
    function(ph) {
        if (ph == null) return window.dash_clientside.no_update;
        const g = document.getElementById("plot"), inner = g && g.querySelector(".js-plotly-plot");
        if (!inner || !inner._fullLayout || !inner._fullLayout.shapes) return window.dash_clientside.no_update;
        const s = inner._fullLayout.shapes, xms = Number(ph) * 1000;
        for (let i=0;i<s.length;i++){ const c=((s[i].line&&s[i].line.color)||"").toUpperCase();
            if (s[i].type==="line" && s[i].yref==="paper" && c==="#FFD166"){ try{ window.Plotly.relayout(inner, {["shapes["+i+"].x0"]:xms, ["shapes["+i+"].x1"]:xms}); }catch(e){} break; } }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dummy", "data"),
    Input("playhead", "data"),
)


if __name__ == "__main__":
    # threaded=True: the /mini-video proxy streams long-lived responses; a single
    # worker would block the page's own requests behind an open video stream.
    app.run(debug=False, port=args.port, use_reloader=False, threaded=True)
