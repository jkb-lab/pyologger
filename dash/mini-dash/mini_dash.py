"""
mini-dash — a stripped-down interactive viewer for one deployment.

Features (only these, kept intentionally minimal):
  - One unified timeline: a single playhead drives the plot line, the video, and
    the coverage strip. Drag it, click the plot, or use the transport controls.
  - Signal plot with per-signal color editing and drag-to-reorder ordering.
  - Event overlays matching plot_tag_data_interactive: point events (breaths,
    accepted heartbeats/strokebeats) as markers on their target signal, and
    state events (dives, QC spans) as filled/shaded spans. Colors, symbols and
    signal targets come from color_mappings.json (__event_styles__ /
    __event_targets__), so mini-dash and the Streamlit plots agree.
  - Synchronized Immich video (proxied server-side; local-file fallback).
  - Transport: play / pause and step by ±0.1 s or ±10 s.

Launch (from pyologger/):
    python dash/mini-dash/mini_dash.py                       # default deployment
    python dash/mini-dash/mini_dash.py --dataset <id> --deployment <id> --port 8070

Standalone: depends only on pyologger utils + DiveDB (for Immich), not on the
big integrated_dash app.
"""

import argparse
import json
import os
import pathlib
import pickle
import re

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, ctx, no_update
import dash

try:
    import dash_cytoscape as cyto
    CYTO_AVAILABLE = True
except Exception:
    CYTO_AVAILABLE = False

from pyologger.plot_data.plotter import (
    _build_state_fill_to_zero_xy,
    _build_state_fill_trace_xy,
    _build_state_segments,
    _state_overlay_above_bounds,
    _state_overlay_split_bounds,
    _state_overlay_y_bounds,
)
from pyologger.utils.folder_manager import load_configuration, resolve_deployment_context
from pyologger.utils.deployment_source import (
    EVENT_COLUMNS,
    resolve_deployment_source,
    resolve_plot_signal_allowlist,
)

# --------------------------------------------------------------------------- #
# Config parameters (the key knobs — edit here or pass via CLI)
# --------------------------------------------------------------------------- #
DEFAULT_DATASET = "oror-adult-orca_hr-sr-vid_sw_JKB-PP"
DEFAULT_DEPLOYMENT = "2023-10-26_oror-001"
DEFAULT_PORT = 8070
WINDOW_MINUTES = 5          # initial view window around the data start
TARGET_HZ = 10.0            # downsample target for plotting (keeps it snappy)
STEP_SMALL = 0.1            # seconds — fine step
STEP_LARGE = 10.0          # seconds — coarse step
PLAYBACK_RATES = [0.5, 1, 2, 5]

# Auto-scroll: page the window when the playhead passes TRIGGER (fraction of the
# window width) and re-seat it at RESET, so playback cycles between the two
# instead of running off the edge.
AUTOSCROLL_TRIGGER = 0.80
AUTOSCROLL_RESET = 0.20

# Heartbeat audio: plays a beat as the playhead crosses each event, but only
# during forward 1x playback (not paused, scrubbing, or at another rate).
# First key present in the deployment wins.
HEARTBEAT_SOUND = "04_fast_heart_badum_360ms.wav"
HEARTBEAT_EVENT_KEYS = [
    "heartbeat_manual_ok",
    "heartbeat_auto_detect_accepted",
]

# Shared plot margins. Every timeline (video strip, depth context, sliders) is
# inset by these so their data areas line up with the signal plot's. The CSS
# mirrors them via --plot-ml / --plot-mr.
PLOT_ML = 60                # left margin (room for y-axis tick labels)
PLOT_MR = 16                # right margin

# Signals stored positive-down, so their y-axis is reversed (surface at top).
DEPTH_LIKE_SIGNALS = ("depth", "corrected_depth", "pressure")

_PALETTE = [
    "#4C9BE8", "#E8794C", "#5FBF77", "#C766D6", "#E8C84C",
    "#4CD6C7", "#E85C8A", "#9B8CFF", "#8CCf5F", "#FF9F4C",
]

# Events shown by default when the deployment has them. Everything else in the
# NetCDF is still listed in the sidebar, just unchecked.
# Keyboard event editing. Press the key with the playhead on a marker to remove
# it, or anywhere else to add one. Add an entry here (and the key becomes live in
# the browser automatically) to make another event type editable.
#   snap_s: how close the playhead must be to an existing marker to count as
#           "on" it — i.e. the toggle-to-delete radius.
EDIT_BINDINGS = {
    "B": {"key": "exhalation_breath", "type": "point", "snap_s": 0.5,
          "short_description": "exhalation breath (GUI)"},
    "H": {"key": "heartbeat_manual_ok", "type": "point", "snap_s": 0.15,
          "short_description": "heartbeat detection (GUI)"},
}

DEFAULT_EVENT_KEYS = [
    "dive",
    "exhalation_breath",
    "heartbeat_auto_detect_accepted",
    "heartbeat_manual_ok",
    # strokebeat detections — naming varies by processing version, and keys the
    # deployment doesn't have are filtered out, so listing all variants is safe.
    "strokebeat_auto_detect_accepted",
    "strokebeat_auto_detect_suggested",
    "strokebeat_auto_detect_rejected",
    "strokebeat_manual_ok",
]

# Signals shown on load, beyond the deployment's own allowlist ordering. These
# are appended if present so stroke traces accompany the strokebeat markers.
PINNED_SIGNALS = ["sr_smoothed", "stroke_rate"]
MAX_DEFAULT_SIGNALS = 8
STATE_FILL_OPACITY = 0.22       # shaded span alpha when a style gives none
MAX_EVENT_MARKERS = 4000        # guard: don't draw more markers than this per key
# Spacing between stacked point-event tracks, as a fraction of the row's y-span.
# 0.15 matches the ladder used in workflows/05_heartbeat_detect.py.
EVENT_OFFSET_STEP = 0.15

# --------------------------------------------------------------------------- #
# CLI + deployment load
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="mini-dash interactive viewer")
parser.add_argument("--dataset", default=DEFAULT_DATASET)
parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
parser.add_argument("--port", type=int, default=DEFAULT_PORT)
parser.add_argument("--source", choices=["auto", "immich", "local"], default="auto",
                    help="video source: auto (immich then local), or force immich/local")
parser.add_argument("--demo", action="store_true",
                    help="load the deployment's trimmed demo outputs "
                         "(outputs_demo/data_trimmed.pkl or outputs_demo/*_output_trimmed.nc) "
                         "instead of the full outputs/ -- falls back to the full outputs "
                         "if no _demo files exist for this deployment")
args = parser.parse_args()

config, data_dir, color_mapping_path, _ = load_configuration()

# --------------------------------------------------------------------------- #
# Event styling — reuse color_mappings.json so mini-dash matches the Streamlit
# plots: __event_styles__ gives color/symbol/shading, __event_targets__ says
# which signal row each event key is drawn on. Deployment-independent, so this
# is loaded once rather than per deployment.
# --------------------------------------------------------------------------- #
try:
    with open(color_mapping_path, "r") as _fh:
        COLOR_MAPPING = json.load(_fh)
except Exception as _exc:
    print(f"[mini-dash] color_mappings.json unavailable ({_exc}); using defaults.")
    COLOR_MAPPING = {}

EVENT_STYLES = COLOR_MAPPING.get("__event_styles__", {}) or {}
EVENT_TARGETS = COLOR_MAPPING.get("__event_targets__", {}) or {}


def list_datasets():
    """Dataset folders under the data dir (00_* are metadata//misc, not data)."""
    try:
        return sorted(d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("00_"))
    except Exception:
        return []


def list_deployments(ds_id):
    """Deployment folders in a dataset that actually have outputs to plot."""
    base = os.path.join(data_dir, str(ds_id or ""))
    if not os.path.isdir(base):
        return []
    out = []
    for d in sorted(os.listdir(base)):
        full = os.path.join(base, d)
        if not os.path.isdir(full) or d.startswith("00_"):
            continue
        outputs = os.path.join(full, "outputs")
        if os.path.isdir(outputs) and any(
            f.endswith(".nc") or f.endswith(".pkl") for f in os.listdir(outputs)
        ):
            out.append(d)
    return out


def _resolve_demo_paths(deployment_folder, deployment_id):
    """Find a deployment's trimmed demo outputs, if any.

    Looks in outputs_demo/ for a netcdf (preferred, matching the standard
    outputs/ lookup's own netcdf-over-pickle priority) or pickle -- e.g.
    outputs_demo/2020-04-10_mian-002_output_trimmed.nc or
    outputs_demo/data_trimmed.pkl. Returns (netcdf_path, pkl_path), either or
    both None if not found.
    """
    demo_dir = os.path.join(deployment_folder, "outputs_demo")
    if not os.path.isdir(demo_dir):
        return None, None
    netcdf_path = None
    pkl_path = None
    for fname in sorted(os.listdir(demo_dir)):
        full = os.path.join(demo_dir, fname)
        if fname.endswith(".nc") and netcdf_path is None:
            netcdf_path = full
        elif fname.endswith(".pkl") and pkl_path is None:
            pkl_path = full
    return netcdf_path, pkl_path


def load_deployment(ds_id, dep_id):
    """(Re)bind all deployment-scoped globals for the given dataset/deployment.

    mini-dash keeps deployment state in module globals for legibility; this is
    the single place they are set, so the dropdowns can switch deployments
    in-process by calling it again.
    """
    global animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder
    global param_manager, source, allowlist, shell, TZ, ALL_SIGNALS, SIGNAL_CHANNELS
    global DEFAULT_SIGNALS, EVENT_DF, ALL_EVENT_KEYS, DEFAULT_EVENTS
    global _EVENT_TYPE, _EVENT_HAS_DURATION, GUI_NOTES_DIR, GUI_NOTES_PATH
    global _g_start, _g_end, FULL_MIN, FULL_MAX, WIN_LO, WIN_HI, PLAYHEAD0
    global CLIPS, CTX_X, CTX_Y, CTX_SIG, LOCAL_VIDEO_DIR, _immich_service

    (animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder,
     param_manager) = resolve_deployment_context(data_dir, dataset_id=ds_id, deployment_id=dep_id)

    demo_netcdf_path, demo_pkl_path = (None, None)
    if args.demo:
        demo_netcdf_path, demo_pkl_path = _resolve_demo_paths(deployment_folder, deployment_id)
        if not (demo_netcdf_path or demo_pkl_path):
            print(f"[mini-dash] --demo requested but no outputs_demo/ files found for "
                  f"{deployment_id}; falling back to the full outputs/.")

    source = resolve_deployment_source(data_dir, dataset_id, deployment_id,
                                       deployment_folder=deployment_folder,
                                       netcdf_path=demo_netcdf_path,
                                       pkl_path=demo_pkl_path)
    allowlist = resolve_plot_signal_allowlist(param_manager, source)
    shell = source.build_metadata_shell(allowed_signals=allowlist)

    TZ = str(shell.deployment_info.get("Time Zone", "UTC") or "UTC")
    ALL_SIGNALS = [s for s in source.signal_names() if s != "location"]
    SIGNAL_CHANNELS = {
        s: list(getattr(meta, "channels", []) or [])
        for s, meta in (source.signal_meta or {}).items()
    }
    # Take the allowlist head, then pin the stroke signals on the end so they
    # aren't cut by the cap (they sort last in the allowlist).
    _picked = [s for s in (allowlist or ALL_SIGNALS) if s in ALL_SIGNALS] or ALL_SIGNALS
    _pinned = [s for s in PINNED_SIGNALS if s in ALL_SIGNALS]
    _head = [s for s in _picked if s not in _pinned][: max(1, MAX_DEFAULT_SIGNALS - len(_pinned))]
    DEFAULT_SIGNALS = _head + _pinned

    EVENT_DF = source.event_data if isinstance(source.event_data, pd.DataFrame) else pd.DataFrame()
    ALL_EVENT_KEYS = source.event_keys()

    # type per key ("point" -> markers, "state" -> spans). A key with any
    # non-zero duration is treated as a span even if it is tagged "point".
    _EVENT_TYPE, _EVENT_HAS_DURATION = {}, {}
    if not EVENT_DF.empty and "key" in EVENT_DF.columns:
        for _key, _grp in EVENT_DF.groupby("key"):
            _types = [str(t) for t in _grp.get("type", pd.Series(dtype=object)).dropna().unique()]
            _EVENT_TYPE[str(_key)] = "state" if "state" in _types else "point"
            _dur = pd.to_numeric(_grp.get("duration"), errors="coerce") if "duration" in _grp else None
            _EVENT_HAS_DURATION[str(_key)] = bool(_dur is not None and (_dur.fillna(0) > 0).any())

    DEFAULT_EVENTS = [k for k in DEFAULT_EVENT_KEYS if k in ALL_EVENT_KEYS]

    GUI_NOTES_DIR = pathlib.Path(deployment_folder) / NOTES_SUBFOLDER
    GUI_NOTES_PATH = GUI_NOTES_DIR / f"{deployment_id}_01_GUI_Notes.xlsx"

    _g_start, _g_end = pd.Timestamp(source.global_start), pd.Timestamp(source.global_end)
    if _g_start.tzinfo is None:
        _g_start = _g_start.tz_localize(TZ)
    if _g_end.tzinfo is None:
        _g_end = _g_end.tz_localize(TZ)

    _win_cache.clear()
    _immich_service = None
    LOCAL_VIDEO_DIR = None
    CLIPS = _build_clip_index(args.source)

    FULL_MIN, FULL_MAX = int(_g_start.timestamp()), int(_g_end.timestamp())
    WIN_LO = FULL_MIN
    WIN_HI = min(FULL_MAX, FULL_MIN + WINDOW_MINUTES * 60)
    PLAYHEAD0 = float(WIN_LO + (WIN_HI - WIN_LO) / 2)

    CTX_X, CTX_Y, CTX_SIG = _load_depth_context()

    print(f"[mini-dash] {dataset_id} / {deployment_id}  tz={TZ}  "
          f"signals={len(ALL_SIGNALS)}  events={len(ALL_EVENT_KEYS)}")


def _unmapped_event_color(key):
    """Stable distinct color for keys absent from color_mappings.json.

    Several real keys (heartbeat_manual_ok, strokebeat_auto_detect_accepted)
    have no JSON entry; a single shared fallback made them indistinguishable.
    Hashing the key keeps the choice stable across restarts. Add the key to the
    JSON to override this.
    """
    idx = sum(ord(c) for c in str(key)) % len(_PALETTE)
    return _PALETTE[idx]


def _event_color(key):
    """Marker/line color for an event key.

    Precedence matches plotter._event_style_color_for_key: a top-level entry in
    color_mappings.json wins over __event_styles__, so editing the JSON's plain
    "<event_key>" entry is what actually recolors the event.
    """
    style = EVENT_STYLES.get(key, {}) or {}
    for candidate in (COLOR_MAPPING.get(key), style.get("color"), style.get("shade_color")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return _unmapped_event_color(key)


def _event_shade_color(key):
    """Fill color for spans — __event_styles__.shade_color wins here, as in
    plotter._resolve_state_annotation_color."""
    style = EVENT_STYLES.get(key, {}) or {}
    for candidate in (COLOR_MAPPING.get(key), style.get("shade_color"), style.get("color")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return _unmapped_event_color(key)


def _event_symbol(key):
    style = EVENT_STYLES.get(key, {}) or {}
    sym = style.get("symbol")
    return sym if isinstance(sym, str) and sym.strip() else "circle"


def _event_opacity(key):
    style = EVENT_STYLES.get(key, {}) or {}
    try:
        val = float(style.get("shade_opacity"))
    except (TypeError, ValueError):
        return STATE_FILL_OPACITY
    # __event_styles__ opacities are tuned for opaque overlays in the Streamlit
    # plots; damp them here so a span never hides the signal underneath it.
    return max(0.08, min(0.45, val * 0.4))


def _rgba(color, alpha):
    """Re-alpha a color. Accepts #rgb / #rrggbb and rgb()/rgba() strings, since
    color_mappings.json uses both (e.g. heart_rate_nan is an rgba string)."""
    text = str(color).strip()
    if text.lower().startswith(("rgb(", "rgba(")):
        parts = text[text.find("(") + 1:text.rfind(")")].split(",")
        try:
            r, g, b = (int(float(p)) for p in parts[:3])
            return f"rgba({r},{g},{b},{alpha})"
        except (ValueError, IndexError):
            return text
    h = text.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return text
    return f"rgba({r},{g},{b},{alpha})"


def _event_target_signals(key):
    """Signals this event draws on, restricted to what the deployment has."""
    targets = EVENT_TARGETS.get(key)
    if isinstance(targets, str):
        targets = [targets]
    resolved = [s for s in (targets or []) if s in ALL_SIGNALS]
    if resolved:
        return resolved
    # Fallbacks for keys with no __event_targets__ entry. QC flags describe a
    # signal's usability, so they belong on that signal (QC_usable_ecg -> ecg)
    # rather than on whatever row happens to be first.
    key_lower = str(key).lower()
    if "dive" in key_lower and "depth" in ALL_SIGNALS:
        return ["depth"]
    # Heartbeat/ECG-derived events belong on ECG or heart-rate rows — never on
    # whatever row happens to be first (which put them on depth).
    if "heartbeat" in key_lower or "ecg" in key_lower:
        return [s for s in ("ecg", "heart_rate_fixed", "heart_rate") if s in ALL_SIGNALS]
    if "stroke" in key_lower:
        return [s for s in ("sr_smoothed", "stroke_rate") if s in ALL_SIGNALS]
    for sig in sorted(ALL_SIGNALS, key=len, reverse=True):
        if sig.lower() in key_lower:
            return [sig]
    return []


def _is_span_key(key):
    """Span (shaded block) vs. point (marker).

    Both fields lie on their own in real data: accepted heartbeats are tagged
    `state` but have zero duration (instantaneous beats), while accepted
    strokebeats are tagged `point` yet carry the stroke interval as a duration.
    A span therefore needs BOTH the state tag and a real duration; anything
    beat-like ends up as a marker, which is how plot_tag_data_interactive draws
    heartbeats, strokebeats and breaths.
    """
    return _EVENT_TYPE.get(key) == "state" and _EVENT_HAS_DURATION.get(key, False)


def _heartbeat_key():
    """Which heartbeat event key to sonify, or None if the deployment has none."""
    for k in HEARTBEAT_EVENT_KEYS:
        if k in ALL_EVENT_KEYS:
            return k
    return None


def _heartbeat_epochs(lo, hi, edits=None):
    """Beat times (epoch seconds) in [lo, hi], including any pending GUI edits.

    Scoped to the visible window so the browser holds a few hundred timestamps
    rather than every beat in the deployment.
    """
    key = _heartbeat_key()
    if not key:
        return []
    df = _events_for_window(lo, hi, [key], edits=edits)
    if df.empty:
        return []
    return sorted(float(t.timestamp()) for t in df["_start"])


def _source_banner():
    """Say exactly where the data on screen comes from.

    Signals/events come from whichever backing store DeploymentDataSource chose
    (NetCDF or pickle); video is Immich (proxied) or local files. Stated
    explicitly so it is never ambiguous which store is being read or edited.
    """
    backend = str(getattr(source, "backend", "") or "unknown")
    nc = pathlib.Path(getattr(source, "netcdf_path", "") or "")
    pkl = pathlib.Path(deployment_folder) / "outputs" / "data.pkl"

    if backend == "netcdf" and nc.exists():
        data_tag, data_path = "NetCDF", nc.name
    elif backend == "pickle" and pkl.exists():
        data_tag, data_path = "data.pkl", pkl.name
    else:
        data_tag, data_path = f"{backend} (unresolved)", ""

    # Video: clip URLs record which route built the index.
    if not CLIPS:
        vid_tag, vid_cls, vid_detail = "no video", "off", ""
    elif str(CLIPS[0].get("url", "")).startswith("/mini-video/"):
        vid_tag, vid_cls = "Immich", ""
        vid_detail = f"{len(CLIPS)} clip(s) · proxied"
    else:
        vid_tag, vid_cls = "local files", "warn"
        vid_detail = f"{len(CLIPS)} clip(s) · {LOCAL_VIDEO_DIR or ''}"

    children = [
        html.Span("Source", className="src-label"),
        html.Span(f"signals + events: {data_tag}", className="src-tag"),
        html.Span(data_path, className="src-path"),
        html.Span(f"video: {vid_tag}", className=f"src-tag {vid_cls}".strip()),
        html.Span(vid_detail, className="src-path"),
    ]
    # Flag the other store when both exist, since Write updates both.
    if backend == "netcdf" and pkl.exists():
        children.append(html.Span("data.pkl present (also updated on Write)", className="src-tag off"))
    children.append(html.Span(str(deployment_folder), className="src-path"))
    return children


def _event_y_offset_frac(key, index=0):
    """Fraction of the row's y-span to lift this point-event track above the trace.

    Matches plot_tag_data_interactive, where each key gets its own fraction so
    the tracks stack instead of overlapping (workflows/05_heartbeat_detect.py
    uses 0.15 / 0.30 / 0.45 / 0.60). An explicit `y_offset_frac` in
    __event_styles__ wins; otherwise keys are spaced on the same 0.15 ladder in
    the order they appear on the row.
    """
    style = EVENT_STYLES.get(key, {}) or {}
    if "y_offset_frac" in style:
        try:
            return float(style["y_offset_frac"])
        except (TypeError, ValueError):
            pass
    return EVENT_OFFSET_STEP * (index + 1)


def _event_row(key, enabled, color=None):
    """One event row: [checkbox] [color swatch] [wrapping name + count/targets].

    A dcc.Checklist per row (rather than one big Checklist) keeps the checkbox
    to the LEFT of the label and lets the name wrap, which the single-Checklist
    layout could not do.
    """
    count = int((EVENT_DF["key"].astype(str) == key).sum()) if not EVENT_DF.empty else 0
    targets = _event_target_signals(key)
    return html.Div(
        [
            dcc.Checklist(
                id={"type": "ev-on", "key": key},
                options=[{"label": "", "value": key}],
                value=[key] if enabled else [],
                className="ev-check",
            ),
            dcc.Input(
                type="color", value=color or _event_color(key),
                id={"type": "ev-color", "key": key}, className="ev-swatch",
            ),
            html.Div([
                html.Span(key, className="ev-name"),
                html.Span(f"{count}" + (f" · {', '.join(targets)}" if targets else " · unmapped"),
                          className="ev-meta"),
            ], className="ev-text"),
        ],
        className="ev-row",
    )




# --------------------------------------------------------------------------- #
# Event editing — pending edits live in a dcc.Store until written out.
#
# An edit is {"op": "add"|"del", "key": str, "epoch": float}. They are applied
# over the on-disk events to build the live view, so nothing touches the pkl/nc
# until the operator clicks Write.
# --------------------------------------------------------------------------- #
# Must match DataReader's data_folder, which workflows/00_load_data.py sets to
# <deployment>/01_raw-data — that is where import_notes looks for both the
# hand-written 00_Notes and this GUI notes file.
NOTES_SUBFOLDER = "01_raw-data"


def _apply_edits(base_df, edits):
    """Return `base_df` with pending adds appended and pending deletes removed."""
    if not edits:
        return base_df
    df = base_df.copy()
    for e in edits:
        key, epoch = str(e.get("key")), float(e.get("epoch"))
        ts = _epoch_to_ts(epoch)
        if e.get("op") == "del":
            if df.empty:
                continue
            dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(TZ)
            tol = pd.Timedelta(milliseconds=1)
            drop = (df["key"].astype(str) == key) & ((dt - ts).abs() <= tol)
            df = df.loc[~drop]
        else:
            cfg = next((c for c in EDIT_BINDINGS.values() if c["key"] == key), {})
            row = {c: pd.NA for c in (df.columns if len(df.columns) else EVENT_COLUMNS)}
            row.update({
                "datetime": ts, "type": cfg.get("type", "point"), "key": key,
                "value": pd.NA, "duration": 0.0,
                "short_description": cfg.get("short_description", f"{key} (GUI)"),
                "long_description": "",
            })
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    if df.empty:
        return df
    df = df.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    return df


def _nearest_event(key, epoch, snap_s):
    """Epoch of the nearest existing `key` event within snap_s, else None."""
    if EVENT_DF.empty:
        return None
    sub = EVENT_DF[EVENT_DF["key"].astype(str) == key]
    if sub.empty:
        return None
    dt = pd.to_datetime(sub["datetime"], errors="coerce", utc=True).dt.tz_convert(TZ)
    deltas = (dt - _epoch_to_ts(epoch)).abs()
    if deltas.empty or deltas.min() > pd.Timedelta(seconds=float(snap_s)):
        return None
    return float(dt.loc[deltas.idxmin()].timestamp())


def _write_gui_notes(edits):
    """Write pending edits to {deployment_id}_01_GUI_Notes.xlsx.

    Uses the same date/time/type/key/value/short_description schema as the
    hand-written 00_Notes file, plus a `deleted` flag that DataReader.import_notes
    honors, so a reprocess from step 00 reproduces exactly this state.
    """
    rows = []
    for e in edits:
        ts = _epoch_to_ts(float(e["epoch"]))
        cfg = next((c for c in EDIT_BINDINGS.values() if c["key"] == e["key"]), {})
        rows.append({
            "date": ts.strftime("%Y-%m-%d"),
            "time": ts.strftime("%H:%M:%S.%f"),
            "type": cfg.get("type", "point"),
            "key": e["key"],
            "value": pd.NA,
            "short_description": cfg.get("short_description", f"{e['key']} (GUI)"),
            "deleted": bool(e.get("op") == "del"),
        })
    df = pd.DataFrame(rows, columns=["date", "time", "type", "key", "value",
                                     "short_description", "deleted"])
    GUI_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    if GUI_NOTES_PATH.exists():
        # Accumulate across sessions rather than clobbering earlier GUI edits.
        try:
            prior = pd.read_excel(GUI_NOTES_PATH)
            df = pd.concat([prior, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date", "time", "key", "deleted"], keep="last")
        except Exception as exc:
            print(f"[mini-dash] could not merge existing GUI notes ({exc}); writing fresh.")
    df.to_excel(GUI_NOTES_PATH, index=False)
    return GUI_NOTES_PATH, len(df)


def _write_events_to_netcdf(nc_path, events):
    """Replace the event_data_* variables in an existing NetCDF.

    Written directly rather than via BaseExporter because that needs a full
    DataReader (i.e. a data.pkl), and NetCDF-only deployments have none. The
    events live on their own `event_data_samples` dim, so swapping just those
    variables leaves every signal untouched.
    """
    import numpy as np
    import xarray as xr

    dt = pd.to_datetime(events["datetime"], errors="coerce", utc=True)
    keep = dt.notna()
    events = events.loc[keep]
    dt = dt.loc[keep].dt.tz_localize(None)

    with xr.open_dataset(nc_path) as ds:
        ds = ds.load()                       # detach before overwriting the file
    ds = ds.drop_dims("event_data_samples", errors="ignore")

    coord = {"event_data_samples": dt.values.astype("datetime64[ns]")}
    for col in ("type", "key", "value", "duration", "short_description", "long_description"):
        if col not in events.columns:
            continue
        vals = events[col]
        if col in ("value", "duration"):
            arr = pd.to_numeric(vals, errors="coerce").to_numpy(dtype="float64")
        else:
            # Force a fixed-width unicode dtype: rows read back from the NetCDF
            # are numpy str_ while GUI-added rows are Python str, and an object
            # array mixing the two is not encodable.
            arr = np.asarray([("" if pd.isna(v) else str(v)) for v in vals], dtype=np.str_)
        ds[f"event_data_{col}"] = xr.DataArray(
            arr, dims=("event_data_samples",), coords=coord, attrs={"variable": col}
        )
    tmp = pathlib.Path(str(nc_path) + ".tmp")
    ds.to_netcdf(tmp)
    ds.close()
    os.replace(tmp, nc_path)


def _write_to_pkl_and_nc(edits):
    """Apply pending edits to the deployment's data.pkl and/or output NetCDF.

    Handles either backing store independently: pickle-backed deployments get
    data.pkl rewritten, NetCDF-backed ones get their event variables replaced.
    Each file is backed up (.bak) before being touched. The GUI notes xlsx is
    the durable record; this just makes the current outputs match so edits are
    visible without a full reprocess.

    Returns (written, problems) so the UI can report a partial or failed write
    instead of silently reporting success.
    """
    global EVENT_DF
    import shutil
    written, problems = [], []
    new_events = _apply_edits(EVENT_DF, edits)

    pkl_path = pathlib.Path(deployment_folder) / "outputs" / "data.pkl"
    if pkl_path.exists():
        try:
            shutil.copy2(pkl_path, pkl_path.with_suffix(".pkl.bak"))
            with open(pkl_path, "rb") as fh:
                data_pkl = pickle.load(fh)
            data_pkl.event_data = new_events.copy()
            with open(pkl_path, "wb") as fh:
                pickle.dump(data_pkl, fh)
            written.append(pkl_path.name)
        except Exception as exc:
            problems.append(f"data.pkl: {exc}")

    nc_path = pathlib.Path(getattr(source, "netcdf_path", "") or "")
    if nc_path.exists():
        try:
            shutil.copy2(nc_path, nc_path.with_suffix(nc_path.suffix + ".bak"))
            _write_events_to_netcdf(nc_path, new_events)
            written.append(nc_path.name)
        except Exception as exc:
            problems.append(f"{nc_path.name}: {exc}")
    else:
        problems.append("no output NetCDF found")

    # Refresh the in-process view so the plot reflects what was just written.
    source.replace_cached_events(new_events)
    EVENT_DF = new_events
    return written, problems



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
def _resolve_color_for_channel(signal, channel):
    """Port of plotter._resolve_color_for_channel — the JSON is keyed per channel.

    Tries "<sig>.<ch>", "<sig>:<ch>", then the bare channel name and its case
    variants, which is how ax/ay/az get three distinct colors from one row.
    """
    sig, ch = str(signal or ""), str(channel or "")
    for key in (f"{sig}.{ch}", f"{sig}:{ch}", ch, ch.lower(), ch.upper()):
        color = COLOR_MAPPING.get(key)
        if isinstance(color, str) and color.strip():
            return color.strip()
    return None


def _signal_color(signal, index=0):
    """Row color for a signal: the JSON's signal-level entry, else the palette."""
    sig = str(signal)
    for key in (sig, sig.lower()):
        color = COLOR_MAPPING.get(key)
        if isinstance(color, str) and color.strip():
            return color.strip()
    # Signals whose JSON entries are per-channel only (e.g. accelerometer -> ax)
    # take their first channel's color so the row label matches the trace.
    for ch in (SIGNAL_CHANNELS.get(sig) or [])[:1]:
        resolved = _resolve_color_for_channel(sig, ch)
        if resolved:
            return resolved
    return _PALETTE[index % len(_PALETTE)]


def _default_colors():
    return {s: _signal_color(s, i) for i, s in enumerate(ALL_SIGNALS)}


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
    """Immich album DepID_<deployment>, else [].

    Immich reports these uploads' fileCreatedAt as a UTC-tagged wall-clock that
    is really deployment-local (the files carry no true EXIF timezone), so
    trusting the tag puts every clip TZ-offset hours away from the signals and
    no video ever matches the playhead. Prefer the local-time range encoded in
    the filename -- same convention as _build_local_clips, and it carries a real
    end time -- and fall back to localizing fileCreatedAt to TZ.
    """
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
        name = a.get("originalFileName") or aid
        s0 = e0 = None
        m = _LOCAL_FILENAME_RE.search(name)
        if m:
            try:
                st = pd.Timestamp(f"{m.group('date')} {m.group('s').replace('-', ':')}").tz_localize(TZ)
                en = pd.Timestamp(f"{m.group('date')} {m.group('e').replace('-', ':')}").tz_localize(TZ)
                if en <= st:
                    en = en + pd.Timedelta(days=1)
                s0, e0 = float(st.timestamp()), float(en.timestamp())
            except Exception:
                s0 = e0 = None
        if s0 is None:
            # No parsable filename: treat the reported wall-clock as deployment-local.
            ts = pd.Timestamp(created)
            ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
            ts = ts.tz_localize(TZ)
            s0 = float(ts.timestamp())
            e0 = s0 + _parse_dur(a.get("duration"))
        clips.append({
            "name": name,
            "start_epoch": s0,
            "end_epoch": e0,
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


def _events_for_window(lo, hi, keys, edits=None):
    """Events overlapping [lo, hi], including spans that start before `lo`.

    Pending (unwritten) edits are applied first, so the plot shows what the
    operator has staged before it is written to disk.
    """
    base = _apply_edits(EVENT_DF, edits) if edits else EVENT_DF
    if not keys or base.empty:
        return pd.DataFrame(columns=base.columns if len(base.columns) else EVENT_COLUMNS)
    lo_ts, hi_ts = _epoch_to_ts(lo), _epoch_to_ts(hi)
    df = base[base["key"].astype(str).isin(list(keys))].copy()
    if df.empty:
        return df
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    if dt.isna().all():
        return df.iloc[0:0]
    dt = dt.dt.tz_convert(TZ)
    dur = pd.to_numeric(df.get("duration"), errors="coerce").fillna(0.0) if "duration" in df else 0.0
    # Only spans get extended by their duration; a point event's duration is a
    # measurement (e.g. the stroke interval), not an on-screen extent.
    is_span = df["key"].astype(str).map(_is_span_key).fillna(False)
    end = dt + pd.to_timedelta(dur.where(is_span, 0.0), unit="s")
    df = df.loc[dt.notna() & (end >= lo_ts) & (dt <= hi_ts)].copy()
    df["_start"] = dt.loc[df.index]
    df["_end"] = end.loc[df.index]
    return df


def _fill_trace(fig, x, y, yax, key, color, opacity, fill, hover):
    """mini-dash's equivalent of plotter._add_state_fill_trace.

    Same trace spec as the plotter — width-0 outline, solid `fillcolor` plus a
    separate `opacity` (NOT an alpha baked into the color, which is what made
    the earlier version look wrong) — but targeting a manual `yaxis` instead of
    a subplot row, since mini-dash builds one figure with stacked y-domains.
    """
    if not len(x):
        return
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line={"width": 0, "color": color},
        fill=fill, fillcolor=color, opacity=opacity, yaxis=yax,
        name=key, legendgroup=key, showlegend=False,
        hoveron="fills" if fill == "toself" else None,
        hovertemplate=f"<b>{hover}</b><extra></extra>",
        hoverlabel={"align": "left", "namelength": -1},
        connectgaps=False,
    ))


def _add_state_spans(fig, key, df, sub, yax, lo, hi, target_channel=None):
    """Draw one state-event key using the plotter's shade modes.

    Ports the dispatch in plot_tag_data_interactive: spans are merged and
    clipped to the window by _build_state_segments, then rendered per the
    __event_styles__ shade_mode — fill_trace (full height), fill_trace_above
    (a band above the signal), fill_trace_split (stripes top and bottom,
    leaving the signal readable through the middle), or fill_trace_to_zero
    (fills the signal itself down to zero, e.g. dives to the surface).
    """
    style = EVENT_STYLES.get(key, {}) or {}
    if style.get("shade_enabled") is False:
        return
    color = _event_shade_color(key)
    try:
        opacity = float(style.get("shade_opacity", STATE_FILL_OPACITY))
    except (TypeError, ValueError):
        opacity = STATE_FILL_OPACITY
    opacity = max(0.01, min(1.0, opacity))
    bridge = float(style.get("line_bridge_seconds", 0.0) or 0.0)

    segments = _build_state_segments(
        sub, window_start=_epoch_to_ts(lo), window_end=_epoch_to_ts(hi),
        gap_seconds=float(style.get("merge_gap_seconds", 0.0) or 0.0),
        default_duration_s=style.get("default_duration_s", 60.0),
    )
    if not segments:
        return

    pct_min = style.get("shade_pct_min", 0.0)
    pct_max = style.get("shade_pct_max", 100.0)
    has_data = isinstance(df, pd.DataFrame) and not df.empty

    mode = str(style.get("shade_mode", "fill_trace")).strip().lower() or "fill_trace"

    if mode == "fill_trace_to_zero" and has_data:
        x, y = _build_state_fill_to_zero_xy(df, segments, target_channel=target_channel,
                                            bridge_seconds=bridge)
        _fill_trace(fig, x, y, yax, key, color, opacity, "tozeroy", key)
        return

    if mode == "fill_trace_above" and has_data:
        bounds = [_state_overlay_above_bounds(
            df, target_channel=target_channel,
            min_factor=float(style.get("above_y_pct_min", 110.0)) / 100.0,
            max_factor=float(style.get("above_y_pct_max", 120.0)) / 100.0,
        )]
    elif mode == "fill_trace_split" and has_data:
        y_lo0, y_lo1, y_hi0, y_hi1 = _state_overlay_split_bounds(
            df, target_channel=target_channel,
            bounds_mode=str(style.get("shade_bounds_mode", "all_y")).strip().lower() or "all_y",
            shade_pct_min=pct_min, shade_pct_max=pct_max,
            split_low_pct=style.get("split_gap_low_pct", 45.0),
            split_high_pct=style.get("split_gap_high_pct", 55.0),
        )
        bounds = [(y_lo0, y_lo1), (y_hi0, y_hi1)]
    else:
        bounds = [_state_overlay_y_bounds(
            df if has_data else None, target_channel=target_channel,
            shade_mode="percent_band" if (pct_min, pct_max) != (0.0, 100.0) else "all_y",
            shade_pct_min=pct_min, shade_pct_max=pct_max,
        )]

    for y0, y1 in bounds:
        x, y = _build_state_fill_trace_xy(segments, y0=y0, y1=y1, bridge_seconds=bridge)
        _fill_trace(fig, x, y, yax, key, color, opacity, "toself", key)


def _add_events_to_row(fig, df, events, yax, lo, hi, sig=None):
    """Draw every event targeting this signal onto its row.

    Point events become markers pinned near the top of the row; state events
    are shaded via the plotter's shade modes (see _add_state_spans).
    """
    if events.empty:
        return
    has_data = df is not None and not df.empty
    chans = [c for c in df.columns if c != "datetime"] if has_data else []
    num = None
    target_channel = None
    if has_data:
        for ch in chans:
            s = pd.to_numeric(df[ch], errors="coerce")
            if s.notna().any():
                num = s
                target_channel = ch
                break
    if num is not None:
        y_lo, y_hi = float(num.min()), float(num.max())
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0
    else:
        y_lo, y_hi = 0.0, 1.0
    span = y_hi - y_lo

    # Give each point-event key its own track above the trace, using
    # plot_tag_data_interactive's formula: y = y_max + (y_offset_frac * y_span).
    # Without this every key lands on one line and they overplot each other.
    point_keys = [k for k in sorted(events["key"].astype(str).unique()) if not _is_span_key(k)]
    offsets = {k: _event_y_offset_frac(k, i) for i, k in enumerate(point_keys)}

    for key in sorted(events["key"].astype(str).unique()):
        sub = events[events["key"].astype(str) == key]
        if sub.empty:
            continue
        color = _event_color(key)
        if _is_span_key(key):
            _add_state_spans(fig, key, df, sub, yax, lo, hi, target_channel)
            continue

        # point event -> markers
        xs = list(sub["_start"])[:MAX_EVENT_MARKERS]
        if not xs:
            continue
        vals = pd.to_numeric(sub.get("value"), errors="coerce") if "value" in sub else None
        text = [
            f"<b>{key}</b><br>{t.strftime('%H:%M:%S.%f')[:-3]}"
            + (f"<br>value: {v:.3g}" if vals is not None and pd.notna(v) else "")
            for t, v in zip(xs, (vals.tolist()[:len(xs)] if vals is not None else [None] * len(xs)))
        ]
        # Offset away from the trace. On a reversed depth axis "above" means
        # smaller values, so subtract from y_min there — otherwise breaths would
        # stack below the deepest point instead of up at the surface.
        frac = offsets.get(key, 0.0)
        step = frac * span if span > 0 else frac
        marker_y = (y_lo - step) if sig in DEPTH_LIKE_SIGNALS else (y_hi + step)
        fig.add_trace(go.Scatter(
            x=xs, y=[marker_y] * len(xs), mode="markers",
            marker={"color": color, "size": 7, "symbol": _event_symbol(key),
                    "line": {"color": "rgba(0,0,0,0.5)", "width": 0.5}},
            yaxis=yax, name=key, legendgroup=key, showlegend=False,
            # hovertemplate (not hoverinfo="text") so hoverlabel.align applies.
            text=text, hovertemplate="%{text}<extra></extra>",
            hoverlabel={"align": "left", "namelength": -1},
        ))


def build_figure(lo, hi, order, colors, playhead, events=None, edits=None):
    order = [s for s in (order or DEFAULT_SIGNALS) if s in ALL_SIGNALS]
    data = _load_window(lo, hi)
    # Edited keys stay visible even if unchecked, so a new marker isn't invisible.
    keys = [k for k in (events or []) if k in ALL_EVENT_KEYS]
    for e in (edits or []):
        if e.get("key") not in keys:
            keys.append(e["key"])
    ev_df = _events_for_window(lo, hi, keys, edits=edits)
    # Map each enabled key to the rows it should be drawn on. A key with no
    # configured target still shows up, on the first visible signal.
    ev_by_signal = {}
    for key in keys:
        targets = [t for t in _event_target_signals(key) if t in order] or (order[:1] if order else [])
        for t in targets:
            ev_by_signal.setdefault(t, []).append(key)
    n = max(1, len(order))
    fig = go.Figure()
    row_h = 1.0 / n
    # A swatch edit means "override the JSON for this signal"; anything still at
    # its JSON-derived default lets per-channel colors through.
    defaults = _default_colors()
    user_colors = {
        s: c for s, c in (colors or {}).items()
        if isinstance(c, str) and c and c.lower() != str(defaults.get(s, "")).lower()
    }
    for i, sig in enumerate(order):
        df = data.get(sig)
        top = 1.0 - i * row_h
        bot = 1.0 - (i + 1) * row_h
        yax = "y" if i == 0 else f"y{i+1}"
        axis_id = "yaxis" if i == 0 else f"yaxis{i+1}"
        color = (colors or {}).get(sig) or _signal_color(sig, i)
        row_keys = ev_by_signal.get(sig, [])
        if row_keys and not ev_df.empty:
            _add_events_to_row(
                fig, df, ev_df[ev_df["key"].astype(str).isin(row_keys)], yax, lo, hi, sig
            )
        if df is not None and not df.empty:
            chans = [c for c in df.columns if c != "datetime"]
            for j, ch in enumerate(chans[:3]):
                # Per-channel color from the JSON (ax/ay/az differ); fall back to
                # the row color, which a UI swatch edit overrides.
                ch_color = _resolve_color_for_channel(sig, ch) or color
                if user_colors.get(sig):
                    ch_color = color
                label = f"{sig}:{ch}" if len(chans) > 1 else sig
                fig.add_trace(go.Scatter(
                    x=df["datetime"], y=df[ch], mode="lines",
                    line={"color": ch_color, "width": 1.1, "dash": "solid"},
                    name=label, yaxis=yax, showlegend=False,
                    # Put the channel name in the hover BODY (left-aligned) instead
                    # of the <extra> box, which right-aligns and clips the end of
                    # long concatenated names like heart_rate_fixed:manually_derived_hr.
                    hovertemplate=f"<b>{label}</b><br>%{{y:.4g}}<extra></extra>",
                    hoverlabel={"align": "left", "namelength": -1},
                ))
        axis_cfg = dict(
            domain=[max(0.0, bot + 0.012), top - 0.012], title=dict(text=sig, font=dict(size=11, color=color)),
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
            tickfont=dict(size=9, color="#9db4c7"),
        )
        # Depth reads downward: depth/pressure are stored positive-down, so
        # reversing the axis puts the surface at the top and dives below it,
        # matching plot_tag_data_interactive.
        if sig in DEPTH_LIKE_SIGNALS:
            axis_cfg["autorange"] = "reversed"
        fig.update_layout({axis_id: axis_cfg})
    ph_ts = _epoch_to_ts(playhead)
    fig.add_shape(type="line", x0=ph_ts, x1=ph_ts, y0=0, y1=1, yref="paper",
                  line=dict(color="#FFD166", width=2))
    fig.update_layout(
        xaxis=dict(range=[_epoch_to_ts(lo), _epoch_to_ts(hi)], showgrid=True,
                   gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color="#9db4c7")),
        margin=dict(l=PLOT_ML, r=PLOT_MR, t=8, b=28), height=max(260, 120 * n),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0b1f2f", uirevision="keep",
        # namelength=-1 disables Plotly's default name truncation; align="left"
        # keeps long channel names readable to their end.
        hoverlabel=dict(align="left", namelength=-1,
                        font=dict(size=11, family="monospace")),
    )
    return fig


# --------------------------------------------------------------------------- #
# Event → channel mapper (Cytoscape), ported from integrated_dash.
#
# Events on the left, signals on the right, one edge per mapping. Click an event
# node to select it, then click signal nodes to add/remove links; click an edge
# to remove it. Saves straight back to __event_targets__ in color_mappings.json.
# --------------------------------------------------------------------------- #
def _save_event_targets(targets):
    """Persist __event_targets__ to color_mappings.json (preserving the rest)."""
    try:
        with open(color_mapping_path, "r") as fh:
            mapping = json.load(fh)
    except Exception:
        mapping = dict(COLOR_MAPPING)
    mapping["__event_targets__"] = {k: list(v) for k, v in targets.items() if v}
    with open(color_mapping_path, "w") as fh:
        json.dump(mapping, fh, indent=4)
    EVENT_TARGETS.clear()
    EVENT_TARGETS.update(mapping["__event_targets__"])


def _save_event_color(key, color):
    """Persist one event's color as a top-level entry (highest precedence)."""
    try:
        with open(color_mapping_path, "r") as fh:
            mapping = json.load(fh)
    except Exception:
        mapping = dict(COLOR_MAPPING)
    mapping[key] = color
    with open(color_mapping_path, "w") as fh:
        json.dump(mapping, fh, indent=4)
    COLOR_MAPPING[key] = color


def build_cyto_elements(targets, selected, shown_events):
    """Nodes for each shown event + every signal, and an edge per mapping."""
    els = []
    events = [k for k in ALL_EVENT_KEYS if k in set(shown_events or ALL_EVENT_KEYS)]
    for i, ev in enumerate(events):
        color = _event_color(ev)
        y = 40 + i * 42
        els.append({"data": {"id": f"ev::{ev}", "label": ev, "kind": "event", "key": ev,
                             "color": color,
                             "border_width": 3 if ev == selected else 1,
                             "border_color": "#ffffff" if ev == selected else "rgba(255,255,255,0.35)"},
                    "position": {"x": 150, "y": y}})
    for i, sig in enumerate(ALL_SIGNALS):
        y = 40 + i * 30
        els.append({"data": {"id": f"sig::{sig}", "label": sig, "kind": "signal", "key": sig,
                             "color": _signal_color(sig, i)},
                    "position": {"x": 640, "y": y}})
    for ev in events:
        for tgt in (targets.get(ev) or []):
            if tgt not in ALL_SIGNALS:
                continue
            els.append({"data": {"id": f"edge::{ev}||{tgt}", "source": f"ev::{ev}",
                                 "target": f"sig::{tgt}", "event": ev, "target_key": tgt,
                                 "kind": "mapping", "color": _event_color(ev),
                                 "edge_opacity": 1.0 if (not selected or ev == selected) else 0.15,
                                 "edge_width": 3 if ev == selected else 2}})
    return els


CYTO_STYLESHEET = [
    {"selector": "node", "style": {
        "label": "data(label)", "font-size": "10px", "text-wrap": "wrap",
        "text-max-width": "180px", "text-valign": "center", "text-halign": "center",
        "color": "#dce9f3", "border-width": 1, "border-color": "#7faec7",
        "width": 200, "height": 26, "shape": "round-rectangle",
        "text-outline-width": 2, "text-outline-color": "rgba(0,0,0,0.55)"}},
    {"selector": "node[kind = 'event']", "style": {
        "background-color": "data(color)", "border-width": "data(border_width)",
        "border-color": "data(border_color)"}},
    {"selector": "node[kind = 'signal']", "style": {
        "background-color": "data(color)", "width": 150}},
    {"selector": "edge", "style": {
        "curve-style": "unbundled-bezier", "control-point-distances": "45 -45",
        "control-point-weights": "0.25 0.75", "target-arrow-shape": "triangle",
        "line-color": "data(color)", "target-arrow-color": "data(color)",
        "opacity": "data(edge_opacity)", "width": "data(edge_width)"}},
]


# --------------------------------------------------------------------------- #
# Depth context strip — whole-deployment overview with the active window
# highlighted, mirroring integrated_dash's mini depth plot.
# --------------------------------------------------------------------------- #
CONTEXT_MAX_POINTS = 2500


def _load_depth_context():
    """Coarse whole-deployment depth/pressure/ODBA series for the context strip."""
    for sig in ("corrected_depth", "depth", "pressure", "odba"):
        if sig not in ALL_SIGNALS:
            continue
        try:
            df = source.load_signal_window(sig, _g_start, _g_end)
        except Exception:
            continue
        if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
            continue
        cols = [c for c in df.columns if c != "datetime"]
        if not cols:
            continue
        col = next((c for c in cols if "depth" in str(c).lower() or str(c).lower() in ("p", "odba")), cols[0])
        x = pd.to_datetime(df["datetime"], errors="coerce")
        y = pd.to_numeric(df[col], errors="coerce")
        keep = x.notna() & y.notna()
        if not keep.any():
            continue
        x, y = x[keep], y[keep]
        if len(x) > CONTEXT_MAX_POINTS:
            step = max(1, len(x) // CONTEXT_MAX_POINTS)
            x, y = x.iloc[::step], y.iloc[::step]
        print(f"[mini-dash] depth context from '{sig}' ({len(x)} pts)")
        return x, y, sig
    print("[mini-dash] no depth context signal available")
    return None, None, None


def build_context_figure(lo, hi, playhead):
    """Full-deployment overview with the active window shaded and the playhead marked."""
    fig = go.Figure()
    if CTX_X is None or not len(CTX_X):
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           text="No depth context available", font={"color": "#73a9c4", "size": 11})
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        fig.add_trace(go.Scatter(
            x=CTX_X, y=CTX_Y, mode="lines",
            line={"color": _signal_color(CTX_SIG), "width": 1.2},
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_vrect(x0=_epoch_to_ts(lo), x1=_epoch_to_ts(hi),
                      fillcolor="rgba(133,198,255,0.28)", line_width=1,
                      line_color="#a7d7ff", layer="above")
        ph = _epoch_to_ts(playhead)
        fig.add_shape(type="line", x0=ph, x1=ph, y0=0, y1=1, xref="x", yref="paper",
                      line={"width": 1.5, "color": "#FFD166"}, layer="above")
        fig.update_yaxes(title={"text": CTX_SIG, "font": {"size": 9, "color": "#9db4c7"}},
                         showgrid=True, gridcolor="rgba(115,169,196,0.15)", zeroline=False,
                         tickfont={"size": 8, "color": "#9db4c7"},
                         # match the signal plot's depth orientation
                         autorange="reversed" if CTX_SIG in DEPTH_LIKE_SIGNALS else True)
        fig.update_xaxes(showgrid=False, tickfont={"size": 8, "color": "#9db4c7"})
    # Left/right margins match the signal plot (PLOT_ML / PLOT_MR) so the depth
    # context, the sliders and the plot below all share the same data-area edges.
    fig.update_layout(height=110, margin={"l": PLOT_ML, "r": PLOT_MR, "t": 6, "b": 18},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#08263a",
                      showlegend=False, uirevision="ctx")
    return fig


# --------------------------------------------------------------------------- #
# App + layout
# --------------------------------------------------------------------------- #
# Initial load. Deferred to here (rather than at the top) because
# load_deployment depends on the clip-index and depth-context helpers above.
load_deployment(args.dataset, args.deployment)

DATASETS = list_datasets()


def _resolve_proxy_pathname_prefix(port: int) -> str | None:
    """Path prefix Dash must prepend to every asset/callback URL when running
    behind a reverse proxy that doesn't strip its own path (NDP's JupyterHub).

    NDP's VS Code port-forwarding serves this app at
    .../vscode/proxy/<port>/ -- VSCODE_PROXY_URI is the exact template
    JupyterHub sets for that (with a `{{port}}` placeholder to fill in).
    Falls back to JUPYTERHUB_SERVICE_PREFIX (plain JupyterHub proxy, no VS
    Code layer) if that's the only one set, and to None (no prefix -- the
    normal case when running locally) if neither is present.
    """
    vscode_proxy_uri = os.environ.get("VSCODE_PROXY_URI")
    if vscode_proxy_uri:
        full_url = vscode_proxy_uri.replace("{{port}}", str(port))
        return "/" + full_url.split("://", 1)[-1].split("/", 1)[-1]

    jupyterhub_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
    if jupyterhub_prefix:
        return jupyterhub_prefix

    return None


_PROXY_PREFIX = _resolve_proxy_pathname_prefix(args.port)
if _PROXY_PREFIX:
    print(f"[mini-dash] behind a proxy; using pathname prefix {_PROXY_PREFIX}")


def _proxied_path(path: str) -> str:
    """Prefix a same-origin URL (video routes, anything else served straight
    off app.server rather than through Dash's own routing) with the proxy
    path, so it resolves correctly behind NDP's VS Code port forwarder.
    requests_pathname_prefix only affects Dash's own asset/callback URLs --
    routes added directly to the underlying Flask app need this by hand.
    """
    prefix = (_PROXY_PREFIX or "/").rstrip("/")
    return f"{prefix}{path}"

# Pin the assets folder to this file's own directory. Dash otherwise resolves it
# relative to the invoking script's location, so launching from pyologger/ picked
# up dash/assets/ (integrated_dash's) and mini-dash's css/js/wav 404'd.
app = Dash(
    __name__,
    assets_folder=str(pathlib.Path(__file__).resolve().parent / "assets"),
    requests_pathname_prefix=_PROXY_PREFIX,
)
app.title = "mini-dash"

# Slider tooltips format epochs in the deployment's timezone, not the browser's.
app.index_string = app.index_string.replace(
    "<footer>",
    "<script>"
    f"window.MINI_TZ = {json.dumps(TZ)};"
    f"window.MINI_EDIT_KEYS = {json.dumps({k: v['key'] for k, v in EDIT_BINDINGS.items()})};"
    "</script><footer>",
)


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
        dcc.Store(id="events", data=list(DEFAULT_EVENTS)),
        dcc.Store(id="edits", data=[]),
        dcc.Store(id="targets", data={k: list(v) for k, v in EVENT_TARGETS.items()}),
        dcc.Store(id="map-sel", data=None),
        dcc.Store(id="beats", data=[]),
        dcc.Store(id="colors", data=_default_colors()),
        dcc.Store(id="playhead", data=PLAYHEAD0),
        dcc.Store(id="clip", data=None),
        # Set while the user is dragging a win-slider handle, so autoscroll does
        # not overwrite the in-progress drag (see follow_playhead).
        dcc.Store(id="win-adjusting", data=0),
        dcc.Store(id="playing", data=False),
        dcc.Store(id="rate", data=1),
        dcc.Store(id="dummy", data=0),
        dcc.Interval(id="tick", interval=100, disabled=True),
        dcc.Input(id="key", type="text", value="", style={"display": "none"}),

        html.Div([
            html.H1("mini-dash", className="brand"),
            html.Div([
                dcc.Dropdown(id="dataset-dd", className="dep-dd",
                             options=[{"label": d, "value": d} for d in DATASETS],
                             value=dataset_id, clearable=False, placeholder="dataset…"),
                dcc.Dropdown(id="deployment-dd", className="dep-dd",
                             options=[{"label": d, "value": d} for d in list_deployments(dataset_id)],
                             value=deployment_id, clearable=False, placeholder="deployment…"),
            ], className="dep-pick"),
            html.Span(id="dep-label", className="sub", children=f"{TZ}"),
        ], className="header"),

        html.Div(id="src-bar", className="src-bar", children=_source_banner()),

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

                html.Div("Events", className="panel-title", style={"marginTop": "18px"}),
                html.Div(
                    "Drawn on their target signal row" if ALL_EVENT_KEYS else "No events in this deployment.",
                    className="hint",
                ),
                html.Div(
                    [_event_row(k, k in DEFAULT_EVENTS) for k in ALL_EVENT_KEYS],
                    id="event-rows", className="event-rows",
                ),
            ], className="side"),

            # right: video + plot + unified timeline
            html.Div([
                html.Div([
                    html.Video(id="video", src="", controls=True, preload="auto", className="video"),
                    html.Div(id="video-status", className="video-status"),
                ], className="video-wrap"),

                # --- all timelines live here, directly under the video ---
                html.Div([
                    # transport
                    html.Div([
                        html.Button("⏮ −10s", id="b-back10", className="tbtn", n_clicks=0),
                        html.Button("◂ −0.1s", id="b-back01", className="tbtn", n_clicks=0),
                        html.Button("▶", id="b-play", className="tbtn play", n_clicks=0, **{"data-playing": "0"}),
                        html.Button("+0.1s ▸", id="b-fwd01", className="tbtn", n_clicks=0),
                        html.Button("+10s ⏭", id="b-fwd10", className="tbtn", n_clicks=0),
                        dcc.Dropdown(id="rate-dd", options=[{"label": f"{r}×", "value": r} for r in PLAYBACK_RATES],
                                     value=1, clearable=False, searchable=False, className="rate-dd"),
                        dcc.Checklist(
                            id="autoscroll",
                            options=[{"label": "auto-scroll", "value": "on"}],
                            value=["on"], className="auto-check",
                            inputClassName="auto-box", labelClassName="auto-lbl",
                        ),
                        dcc.Checklist(
                            id="heartsound",
                            options=[{"label": "play heartbeats", "value": "on",
                                      "disabled": _heartbeat_key() is None}],
                            value=[], className="auto-check",
                            inputClassName="auto-box", labelClassName="auto-lbl",
                        ),
                        html.Span(id="ph-label", className="ph-label"),
                    ], className="transport"),

                    # 1. video coverage strip
                    html.Div([
                        html.Div("Video", className="tl-label"),
                        html.Div(
                            html.Div(
                                [html.Div(className="cov-bar",
                                          style={"--s": c["start_epoch"], "--e": c["end_epoch"]}) for c in CLIPS],
                                id="cov-strip", className="cov-strip",
                                style={"--min": WIN_LO, "--max": WIN_HI},
                            ),
                            id="cov-align", className="tl-inset",
                        ),
                    ], className="tl-row"),

                    # 2. whole-deployment depth context (full width, own margins)
                    dcc.Graph(id="ctx-plot", className="ctx-plot",
                              config={"displayModeBar": False}),

                    # 3. window selector
                    html.Div([
                        html.Div("Window", className="tl-label"),
                        html.Div([
                            dcc.RangeSlider(id="win-slider", min=FULL_MIN, max=FULL_MAX, step=1,
                                            value=[WIN_LO, WIN_HI], marks=None, className="win-slider",
                                            updatemode="mouseup",
                                            tooltip={"always_visible": False, "transform": "epochToDateTime"}),
                        ], className="tl-inset"),
                    ], className="tl-row"),

                    # 4. playhead (within the current window)
                    html.Div([
                        html.Div("Playhead", className="tl-label"),
                        html.Div([
                            dcc.Slider(id="ph-slider", min=WIN_LO, max=WIN_HI, step=0.1, value=PLAYHEAD0,
                                       marks=None, updatemode="drag", className="ph-slider",
                                       tooltip={"always_visible": False, "transform": "epochToTime"}),
                        ], className="tl-inset"),
                    ], className="tl-row"),

                    html.Div(id="win-label", className="win-label"),
                ], className="timeline-stack"),

                # pending-edit banner (hidden until an edit is made)
                html.Div([
                    html.Span(id="edit-status", className="edit-status"),
                    html.Button("Write to pkl and nc", id="b-write", className="tbtn write", n_clicks=0),
                    html.Button("Discard", id="b-discard", className="tbtn", n_clicks=0),
                    html.Span(id="write-result", className="write-result"),
                ], id="edit-bar", className="edit-bar", style={"display": "none"}),

                dcc.Graph(id="plot", className="plot", config={"displayModeBar": False}),

                # --- event → channel mapper ---
                html.Div([
                    html.Div("Event → Channel Mapper", className="panel-title"),
                    html.Div(
                        "Click an event node to select it, then click signal nodes to add/remove "
                        "links. Click an edge to remove it. Saves to color_mappings.json.",
                        className="hint",
                    ),
                    html.Div(id="map-status", className="map-status"),
                    (cyto.Cytoscape(
                        id="event-cyto", elements=[], layout={"name": "preset"},
                        style={"width": "100%", "height": "460px"},
                        stylesheet=CYTO_STYLESHEET,
                        userZoomingEnabled=True, userPanningEnabled=True,
                        boxSelectionEnabled=False, minZoom=0.4, maxZoom=2.2,
                     ) if CYTO_AVAILABLE else html.Div(
                        "dash-cytoscape not installed — run `pip install dash-cytoscape` "
                        "to enable the mapper.", className="hint")),
                ], className="mapper"),
            ], className="main"),
        ], className="body"),
    ],
    className="mini-app",
)


# --------------------------------------------------------------------------- #
# Video proxy (Immich) — keeps API key server-side, forwards Range for seeking
# --------------------------------------------------------------------------- #
_ASSET_RE = re.compile(r"^[0-9a-fA-F-]{16,64}$")


@app.server.route(_proxied_path("/mini-video/<asset_id>"))
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


@app.server.route(_proxied_path("/mini-local/<path:filename>"))
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
    return [_chip(s, colors.get(s) or _signal_color(s, i), i) for i, s in enumerate(order)]


# --------------------------------------------------------------------------- #
# Callbacks — figure
# --------------------------------------------------------------------------- #
@app.callback(
    Output("events", "data"),
    Input({"type": "ev-on", "key": dash.ALL}, "value"),
    State({"type": "ev-on", "key": dash.ALL}, "id"),
)
def set_events(values, ids):
    on = []
    for val, cid in zip(values or [], ids or []):
        if val:
            on.append(cid["key"])
    return [k for k in ALL_EVENT_KEYS if k in set(on)]


@app.callback(
    Output("map-status", "children", allow_duplicate=True),
    Input({"type": "ev-color", "key": dash.ALL}, "value"),
    State({"type": "ev-color", "key": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def set_event_colors(values, ids):
    """Swatch edits write straight to color_mappings.json (top-level = wins)."""
    changed = []
    for val, cid in zip(values or [], ids or []):
        key = cid["key"]
        if val and str(val).lower() != str(_event_color(key)).lower():
            _save_event_color(key, val)
            changed.append(key)
    if not changed:
        raise dash.exceptions.PreventUpdate
    return f"Saved color for {', '.join(changed)}."


@app.callback(
    Output("plot", "figure"),
    Input("order", "data"),
    Input("colors", "data"),
    Input("win-slider", "value"),
    Input("events", "data"),
    Input("edits", "data"),
    Input("targets", "data"),
    Input("map-status", "children"),
    State("playhead", "data"),
    State("playing", "data"),
)
def draw(order, colors, win, events, edits, _targets, _status, playhead, playing):
    # The plot rebuilds on order/color/window/event changes. The playhead LINE is
    # moved clientside on every tick (no rebuild), so playback/drag stay smooth; the
    # line is drawn at the current playhead here so a rebuild lands it in the right spot.
    lo, hi = int(min(win)), int(max(win))
    ph = max(lo, min(hi, float(playhead if playhead is not None else PLAYHEAD0)))
    return build_figure(lo, hi, order, colors, ph, events=events, edits=edits)


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


@app.callback(Output("win-label", "children"), Input("win-slider", "value"))
def win_text(win):
    lo, hi = int(min(win)), int(max(win))
    span = hi - lo
    dur = f"{span // 60}m {span % 60}s" if span >= 60 else f"{span}s"
    return f"{_fmt_full(lo)} → {_fmt_full(hi)}  ·  {dur}"


@app.callback(
    Output("ctx-plot", "figure"),
    Input("win-slider", "value"),
    Input("playhead", "data"),
)
def draw_context(win, ph):
    lo, hi = int(min(win)), int(max(win))
    return build_context_figure(lo, hi, float(ph if ph is not None else PLAYHEAD0))


@app.callback(
    Output("win-slider", "value", allow_duplicate=True),
    Input("ctx-plot", "clickData"),
    State("win-slider", "value"),
    prevent_initial_call=True,
)
def win_from_context(click, win):
    """Click the context strip to recenter the window there, keeping its width."""
    if not isinstance(click, dict) or not click.get("points"):
        raise dash.exceptions.PreventUpdate
    x = click["points"][0].get("x")
    try:
        ts = pd.Timestamp(x)
        ep = ts.tz_localize(TZ).timestamp() if ts.tzinfo is None else ts.timestamp()
    except Exception:
        raise dash.exceptions.PreventUpdate
    lo, hi = float(min(win)), float(max(win))
    width = hi - lo
    new_lo = max(FULL_MIN, min(FULL_MAX - width, ep - width / 2))
    return [int(new_lo), int(min(FULL_MAX, new_lo + width))]


# --------------------------------------------------------------------------- #
# Callbacks — event editing
# --------------------------------------------------------------------------- #
@app.callback(
    Output("edits", "data"),
    Input("key", "value"),
    State("playhead", "data"),
    State("edits", "data"),
    prevent_initial_call=True,
)
def edit_event(key_code, ph, edits):
    """Toggle an event marker at the playhead for the pressed edit key.

    On a marker (within snap_s) -> delete it; otherwise -> add one. Pressing the
    key again on a pending edit cancels it, so edits stay reversible until Write.
    """
    if not key_code:
        raise dash.exceptions.PreventUpdate
    code = str(key_code).split(":", 1)[0]
    if not code.startswith("E"):
        raise dash.exceptions.PreventUpdate
    cfg = EDIT_BINDINGS.get(code[1:])
    if not cfg:
        raise dash.exceptions.PreventUpdate

    edits = list(edits or [])
    ev_key = cfg["key"]
    epoch = float(ph if ph is not None else PLAYHEAD0)
    snap = float(cfg.get("snap_s", 0.5))

    # Cancel a pending edit at this spot before creating a competing one.
    for i, e in enumerate(edits):
        if e.get("key") == ev_key and abs(float(e.get("epoch", 0)) - epoch) <= snap:
            edits.pop(i)
            return edits

    existing = _nearest_event(ev_key, epoch, snap)
    if existing is not None:
        edits.append({"op": "del", "key": ev_key, "epoch": existing})
    else:
        edits.append({"op": "add", "key": ev_key, "epoch": round(epoch, 3)})
    return edits


@app.callback(
    Output("edit-bar", "style"),
    Output("edit-status", "children"),
    Input("edits", "data"),
)
def edit_banner(edits):
    edits = edits or []
    if not edits:
        return {"display": "none"}, ""
    parts = []
    for ev_key in sorted({e["key"] for e in edits}):
        added = sum(1 for e in edits if e["key"] == ev_key and e["op"] == "add")
        removed = sum(1 for e in edits if e["key"] == ev_key and e["op"] == "del")
        bits = [f"{added} added" if added else "", f"{removed} removed" if removed else ""]
        parts.append(f"{ev_key}: " + ", ".join(b for b in bits if b))
    return {"display": "flex"}, "  ·  ".join(parts)


@app.callback(
    Output("edits", "data", allow_duplicate=True),
    Output("write-result", "children"),
    Input("b-write", "n_clicks"),
    Input("b-discard", "n_clicks"),
    State("edits", "data"),
    prevent_initial_call=True,
)
def write_or_discard(_w, _d, edits):
    if ctx.triggered_id == "b-discard":
        return [], "Discarded."
    if not edits:
        raise dash.exceptions.PreventUpdate
    try:
        notes_path, n_notes = _write_gui_notes(edits)
        written, problems = _write_to_pkl_and_nc(edits)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return no_update, f"Write failed: {exc}"
    msg = f"Wrote {n_notes} GUI note(s) → {notes_path.name}"
    msg += f"; updated {', '.join(written)}" if written else "; no data files updated"
    if problems:
        # Keep the edits staged so a partial write can be retried.
        print(f"[mini-dash] write problems: {problems}")
        return no_update, msg + f"  ⚠️ {'; '.join(problems)}"
    print(f"[mini-dash] {msg}")
    return [], msg + "."


# --------------------------------------------------------------------------- #
# Callbacks — dataset / deployment selection
# --------------------------------------------------------------------------- #
@app.callback(
    Output("deployment-dd", "options"),
    Output("deployment-dd", "value"),
    Input("dataset-dd", "value"),
    prevent_initial_call=True,
)
def pick_dataset(ds_id):
    """Repopulate the deployment list; default to the first one."""
    deps = list_deployments(ds_id)
    return [{"label": d, "value": d} for d in deps], (deps[0] if deps else None)


@app.callback(
    Output("dep-label", "children"),
    Output("order", "data", allow_duplicate=True),
    Output("colors", "data", allow_duplicate=True),
    Output("events", "data", allow_duplicate=True),
    Output("edits", "data", allow_duplicate=True),
    Output("event-rows", "children"),
    Output("win-slider", "min"),
    Output("win-slider", "max"),
    Output("win-slider", "value", allow_duplicate=True),
    Output("playhead", "data", allow_duplicate=True),
    Output("cov-strip", "children"),
    Output("src-bar", "children"),
    Output("write-result", "children", allow_duplicate=True),
    Input("deployment-dd", "value"),
    State("dataset-dd", "value"),
    prevent_initial_call=True,
)
def switch_deployment(dep_id, ds_id):
    """Load a different deployment in-process and reset all deployment-scoped UI.

    Pending edits are dropped deliberately: they reference the previous
    deployment's timeline and must not leak across.
    """
    if not dep_id:
        raise dash.exceptions.PreventUpdate
    if dep_id == deployment_id and ds_id == dataset_id:
        raise dash.exceptions.PreventUpdate
    try:
        load_deployment(ds_id, dep_id)
    except Exception as exc:
        print(f"[mini-dash] failed to load {ds_id}/{dep_id}: {exc}")
        return (no_update,) * 12 + (f"Failed to load {dep_id}: {exc}",)
    bars = [html.Div(className="cov-bar",
                     style={"--s": c["start_epoch"], "--e": c["end_epoch"]}) for c in CLIPS]
    return (
        f"{TZ}",
        list(DEFAULT_SIGNALS),
        _default_colors(),
        list(DEFAULT_EVENTS),
        [],
        [_event_row(k, k in DEFAULT_EVENTS) for k in ALL_EVENT_KEYS],
        FULL_MIN, FULL_MAX, [WIN_LO, WIN_HI],
        PLAYHEAD0,
        bars,
        _source_banner(),
        "",
    )


# --------------------------------------------------------------------------- #
# Callbacks — event → channel mapper
# --------------------------------------------------------------------------- #
if CYTO_AVAILABLE:

    @app.callback(
        Output("event-cyto", "elements"),
        Input("targets", "data"),
        Input("map-sel", "data"),
        Input("events", "data"),
        Input("map-status", "children"),
    )
    def render_cyto(targets, selected, shown, _status):
        # Show every event key, not just the checked ones, so an unmapped event
        # can still be wired up without enabling it first.
        return build_cyto_elements(dict(targets or {}), selected, ALL_EVENT_KEYS)

    @app.callback(
        Output("map-sel", "data"),
        Input("event-cyto", "tapNodeData"),
        prevent_initial_call=True,
    )
    def select_event_node(node):
        if isinstance(node, dict) and node.get("kind") == "event":
            return str(node.get("key") or "")
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("targets", "data"),
        Output("map-status", "children"),
        Input("event-cyto", "tapNodeData"),
        Input("event-cyto", "tapEdgeData"),
        State("map-sel", "data"),
        State("targets", "data"),
        prevent_initial_call=True,
    )
    def edit_mapping(node, edge, selected, targets):
        out = {k: list(v) for k, v in (targets or {}).items()}
        prop = ctx.triggered[0].get("prop_id", "") if ctx.triggered else ""

        if prop.endswith(".tapEdgeData"):
            if not (isinstance(edge, dict) and edge.get("event") and edge.get("target_key")):
                raise dash.exceptions.PreventUpdate
            ev, tgt = str(edge["event"]), str(edge["target_key"])
            out[ev] = [v for v in out.get(ev, []) if v != tgt]
            _save_event_targets(out)
            return out, f"Unlinked {ev} → {tgt}."

        if prop.endswith(".tapNodeData"):
            if not (isinstance(node, dict) and node.get("kind") == "signal"):
                raise dash.exceptions.PreventUpdate
            ev = str(selected or "").strip()
            if not ev:
                return no_update, "Select an event node first, then click a signal."
            tgt = str(node.get("key") or "")
            existing = list(out.get(ev, []))
            if tgt in existing:
                out[ev] = [v for v in existing if v != tgt]
                msg = f"Unlinked {ev} → {tgt}."
            else:
                out[ev] = existing + [tgt]
                msg = f"Linked {ev} → {tgt}."
            _save_event_targets(out)
            return out, msg

        raise dash.exceptions.PreventUpdate


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
    State("autoscroll", "value"),
    State("win-adjusting", "data"),
    prevent_initial_call=True,
)
def follow_playhead(ph, win, autoscroll, adjusting):
    """Page the window so the playhead stays in the readable middle.

    Rather than waiting for the playhead to fall off the edge (which made it
    jump a whole window and lose context), scroll once it passes
    AUTOSCROLL_TRIGGER and re-seat it at AUTOSCROLL_RESET. Playback then cycles
    between those two positions instead of running to the edge.

    Scrubbing far outside the window still recenters in one step, and the
    window is always clamped to the deployment bounds.
    """
    if not autoscroll:
        raise dash.exceptions.PreventUpdate
    # While a win-slider handle is being dragged, autoscroll must not rewrite the
    # value: it preserves the OLD width, so it would snap the handle back and make
    # setting a new window edge impossible.
    if adjusting:
        raise dash.exceptions.PreventUpdate
    lo, hi = float(min(win)), float(max(win))
    p = float(ph if ph is not None else PLAYHEAD0)
    width = hi - lo
    if width <= 0:
        raise dash.exceptions.PreventUpdate

    frac = (p - lo) / width
    if AUTOSCROLL_RESET <= frac <= AUTOSCROLL_TRIGGER:
        raise dash.exceptions.PreventUpdate

    if frac > AUTOSCROLL_TRIGGER:
        # advancing: put the playhead back at the reset mark
        new_lo = p - width * AUTOSCROLL_RESET
    elif frac < 0.0 or frac > 1.0:
        # scrubbed outside the window entirely: center on it
        new_lo = p - width / 2.0
    else:
        # moving backwards within the window: seat at the trigger mark
        new_lo = p - width * AUTOSCROLL_TRIGGER

    new_lo = max(FULL_MIN, min(float(FULL_MAX) - width, new_lo))
    new_hi = min(FULL_MAX, new_lo + width)
    if int(new_lo) == int(lo) and int(new_hi) == int(hi):
        raise dash.exceptions.PreventUpdate
    return [int(new_lo), int(new_hi)]


# While a window handle is being dragged, drag_value fires continuously but
# value only commits on mouseup (updatemode="mouseup"). Raise the flag on drag
# and lower it once the committed value lands, so autoscroll stays out of the way
# for the whole gesture.
app.clientside_callback(
    "function(dv){ return dv ? 1 : window.dash_clientside.no_update; }",
    Output("win-adjusting", "data", allow_duplicate=True),
    Input("win-slider", "drag_value"),
    prevent_initial_call=True,
)

app.clientside_callback(
    "function(v){ return 0; }",
    Output("win-adjusting", "data", allow_duplicate=True),
    Input("win-slider", "value"),
    prevent_initial_call=True,
)


# --------------------------------------------------------------------------- #
# Callbacks — heartbeat audio
# --------------------------------------------------------------------------- #
@app.callback(
    Output("beats", "data"),
    Input("win-slider", "value"),
    Input("edits", "data"),
    Input("heartsound", "value"),
)
def collect_beats(win, edits, on):
    """Beat times for the current window; empty when the sound is off."""
    if not on:
        return []
    lo, hi = int(min(win)), int(max(win))
    return _heartbeat_epochs(lo, hi, edits=edits)


# Push the beat list and the on/off state into the browser-side audio engine.
app.clientside_callback(
    """
    function(beats, on) {
        if (!window.MiniHeart) return window.dash_clientside.no_update;
        window.MiniHeart.setEnabled(!!(on && on.length));
        window.MiniHeart.setBeats(beats || []);
        return window.dash_clientside.no_update;
    }
    """,
    Output("beats", "id"),
    Input("beats", "data"), Input("heartsound", "value"),
)

# Fire on playhead movement. MiniHeart itself decides whether this looks like
# forward 1x playback; a scrub or a rate change just re-seats its cursor.
app.clientside_callback(
    """
    function(ph, playing, rate) {
        if (window.MiniHeart) window.MiniHeart.tick(ph, !!playing, rate);
        return window.dash_clientside.no_update;
    }
    """,
    Output("heartsound", "id"),
    Input("playhead", "data"),
    State("playing", "data"), State("rate", "data"),
)


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
    return _proxied_path(clip["url"]), status, state


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
    # host="0.0.0.0" when a proxy prefix was detected: NDP's VS Code port
    # forwarder connects from outside the container's loopback interface, so
    # binding to 127.0.0.1 (Dash's default) would leave it unreachable.
    host = "0.0.0.0" if _PROXY_PREFIX else "127.0.0.1"
    app.run(debug=False, host=host, port=args.port, use_reloader=False, threaded=True)
