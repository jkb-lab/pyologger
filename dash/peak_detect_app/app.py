"""
Peak Detection Review App  (v2)
---------------------------------
Seven-stage guided workflow for HR / stroke-rate peak detection.

Stages
  1  Load       — detect existing results or choose a preset
  2  Boundaries — mark on / off animal time windows
  3  Calibrate  — click first 5 peaks in a draggable 2-min window → auto-suggest params
  4  Tune       — compare suggested params vs preset; fine-tune sliders
  5  Chunk run  — run detection on 10% chunk, inspect, iterate
  6  Full run   — run on whole deployment; lasso-select peaks to accept/reject
  7  Save       — lock in manual edits; they survive future auto-runs

Run:
    cd pyologger
    python dash/peak_detect_app/app.py [--port 8502]
"""

from __future__ import annotations

import pathlib
import pickle
import sys
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import (
    Dash, Input, Output, State, dcc, html, ctx, no_update,
    callback_context, MATCH, ALL,
)
import dash

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent
_PYOLOGGER_ROOT = _HERE.parents[1]
if str(_PYOLOGGER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYOLOGGER_ROOT))
else:
    sys.path.remove(str(_PYOLOGGER_ROOT))
    sys.path.insert(0, str(_PYOLOGGER_ROOT))

from pyologger.process_data.peak_detect import peak_detect
from pyologger.process_data.sampling import calculate_sampling_frequency
from pyologger.utils.param_manager import ParamManager
from pyologger.utils.folder_manager import load_configuration
from pyologger.utils.chunk_manager import (
    load_chunks, save_chunk_status, compute_chunk_grid,
)
from dash.exceptions import PreventUpdate

from presets import (
    BUILTIN_PRESETS, list_presets, get_preset,
    suggest_preset_for_dataset, save_preset_to_param_manager, load_dataset_presets,
)
from peak_io import (
    load_datapkl, save_datapkl,
    write_auto_peaks_to_events, apply_manual_edits,
    count_manual_edits, list_backups,
)

# ── config ────────────────────────────────────────────────────────────────────
_config, DATA_DIR, _, _ = load_configuration()

_COLOR_MAPPING_PATH = _PYOLOGGER_ROOT / "color_mappings.json"
try:
    with open(_COLOR_MAPPING_PATH, "r") as f:
        COLOR_MAPPING = json.load(f)
except (OSError, json.JSONDecodeError):
    COLOR_MAPPING = {}

# ── constants ─────────────────────────────────────────────────────────────────
STAGES = [
    ("1", "Load"),
    ("2", "Boundaries"),
    ("3", "Calibrate"),
    ("4", "Tune"),
    ("5", "Chunk Run"),
    ("6", "Full Run"),
    ("7", "Save"),
]

STATUS_COLORS = {
    "pending":   "#4a6070",
    "validated": "#55d38a",
    "gap":       "#f07070",
    "override":  "#f2b26b",
}

HR_PARAM_DEFS = {
    "BROAD_LOW_CUTOFF":               (1.0,   0.01, 25.0,   0.1,  "Broad low cutoff (Hz)"),
    "BROAD_HIGH_CUTOFF":              (25.0,  0.01, 25.0,   0.1,  "Broad high cutoff (Hz)"),
    "NARROW_LOW_CUTOFF":              (5.0,   0.01, 15.0,   0.1,  "Narrow low cutoff (Hz)"),
    "NARROW_HIGH_CUTOFF":             (15.0,  0.01, 25.0,   0.1,  "Narrow high cutoff (Hz)"),
    "FILTER_ORDER":                   (2,     1,    10,     1,    "Filter order"),
    "SPIKE_THRESHOLD":                (400,   10,   2000,   10,   "Spike threshold"),
    "SMOOTH_SEC_MULTIPLIER":          (0.36,  0.01, 15.0,   0.01, "Smooth sec multiplier"),
    "WINDOW_SIZE_MULTIPLIER":         (6.35,  0.1,  40.0,   0.05, "Window size multiplier"),
    "PEAK_HEIGHT":                    (-0.4,  -2.0, 2.0,    0.05, "Peak height threshold"),
    "PEAK_DISTANCE_SEC":              (0.16,  0.01, 5.0,    0.01, "Min peak distance (s)"),
    "SEARCH_RADIUS_SEC":              (0.2,   0.05, 2.0,    0.05, "Search radius (s)"),
    "MIN_PEAK_HEIGHT":                (70,    -10,  5000,   5,    "Min peak height"),
    "MAX_PEAK_HEIGHT":                (12000, 1,    100000, 100,  "Max peak height"),
    "HR_JUMP_FRAC":                   (0.8,   0.0,  5.0,    0.05, "HR jump frac"),
    "MIN_RR_SEC":                     (0.25,  0.01, 5.0,    0.01, "Min RR (s)"),
    "MAX_HR_BPM":                     (240,   1,    500,    1,    "Max HR (bpm)"),
    "MIN_HR_BPM":                     (0.1,   0.0,  500,    0.1,  "Min HR (bpm)"),
    "ANTI_DOUBLE_GAP_FACTOR":         (0.75,  0.1,  2.0,    0.05, "Anti-double gap factor"),
    "ANTI_DOUBLE_ROLLING_WINDOW_SEC": (10.0,  2.0,  120.0,  1.0,  "Anti-double window (s)"),
}

SR_PARAM_DEFS = {
    "BROAD_LOW_CUTOFF":              (0.1,  0.01, 10.0,  0.05, "Broad low cutoff (Hz)"),
    "BROAD_HIGH_CUTOFF":             (5.0,  0.01, 25.0,  0.1,  "Broad high cutoff (Hz)"),
    "NARROW_LOW_CUTOFF":             (0.5,  0.01, 10.0,  0.05, "Narrow low cutoff (Hz)"),
    "NARROW_HIGH_CUTOFF":            (3.0,  0.01, 15.0,  0.1,  "Narrow high cutoff (Hz)"),
    "FILTER_ORDER":                  (2,    1,    10,    1,    "Filter order"),
    "SPIKE_THRESHOLD":               (400,  10,   2000,  10,   "Spike threshold"),
    "SMOOTH_SEC_MULTIPLIER":         (0.36, 0.01, 15.0,  0.01, "Smooth sec multiplier"),
    "WINDOW_SIZE_MULTIPLIER":        (6.35, 0.1,  40.0,  0.05, "Window size multiplier"),
    "PEAK_HEIGHT":                   (-0.4, -2.0, 2.0,   0.05, "Peak height threshold"),
    "PEAK_DISTANCE_SEC":             (0.3,  0.01, 5.0,   0.01, "Min peak distance (s)"),
    "SEARCH_RADIUS_SEC":             (0.3,  0.05, 2.0,   0.05, "Search radius (s)"),
    "MIN_PEAK_HEIGHT":               (70,   -10,  5000,  5,    "Min peak height"),
    "MAX_PEAK_HEIGHT":               (12000, 1,   100000,100,  "Max peak height"),
}

BOOL_PARAMS = [
    ("enable_bandpass",      "Bandpass filter"),
    ("enable_spike_removal", "Spike removal"),
    ("enable_absolute",      "Abs transform (HR only)"),
    ("enable_smoothing",     "Smoothing"),
    ("enable_normalization", "Normalization"),
    ("enable_refinement",    "Peak refinement"),
]

CHANNEL_OPTIONS = ["normalized", "smoothed", "narrow_bandpass", "broad_bandpass", "spikeless", "raw"]

PEAK_STYLES = {
    "beat_auto_detect_accepted":  dict(color="#55d38a", symbol="triangle-up",   size=9,  name="accepted"),
    "beat_auto_detect_rejected":  dict(color="#f07070", symbol="triangle-down", size=7,  name="rejected"),
    "beat_auto_detect_suggested": dict(color="#f2b26b", symbol="diamond",       size=8,  name="suggested"),
    "heartbeat_manual_ok":        dict(color="#95ccfe", symbol="star",          size=12, name="manual ok"),
    "heartbeat_manual_reject":    dict(color="#d060a0", symbol="x",             size=10, name="manual reject"),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _list_datasets():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith("00_")
    )


def _list_deployments(dataset):
    folder = os.path.join(DATA_DIR, dataset)
    if not os.path.isdir(folder):
        return []
    return sorted(
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d)) and not d.startswith("00_")
    )


def _pm(dataset, deployment):
    dep_folder = os.path.join(DATA_DIR, dataset, deployment)
    return ParamManager(deployment_folder=dep_folder, deployment_id=deployment)


def _dark_layout(**kwargs):
    base = dict(
        paper_bgcolor="#021524",
        plot_bgcolor="#081a29",
        font=dict(family="Figtree, sans-serif", color="#8eb0cb", size=11),
    )
    base.update(kwargs)
    return base


def _param_slider(key, defn):
    default, mn, mx, step, label = defn
    return html.Div([
        html.Div([
            html.Span(label, className="param-label"),
            html.Span(str(default), id={"type": "param-val", "key": key}, className="param-value"),
        ], className="param-row-header"),
        dcc.Slider(
            id={"type": "param-slider", "key": key},
            min=mn, max=mx, step=step, value=default,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
            className="param-slider",
        ),
    ], className="param-row")


def _section(title, children, open_=True):
    return html.Details([
        html.Summary(title, className="section-summary"),
        html.Div(children, className="section-body"),
    ], open=open_, className="param-section")


def _chunk_pill(chunk, i, active):
    status = chunk.get("status", "pending")
    cls = f"chunk-pill {'chunk-pill--active' if i == active else ''}"
    return html.Div(
        str(i + 1),
        id={"type": "chunk-pill", "index": i},
        className=cls,
        style={"background": STATUS_COLORS.get(status, "#4a6070")},
        title=f"Chunk {i+1}: {chunk.get('start','')[:16]} → {chunk.get('end','')[:16]} [{status}]",
    )


def _empty_fig(msg="", color="#8eb0cb"):
    fig = go.Figure().update_layout(**_dark_layout(
        margin=dict(l=50, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, color="#4a6070"),
        yaxis=dict(showgrid=True, gridcolor="rgba(115,169,196,0.08)", color="#4a6070", fixedrange=True),
    ))
    if msg:
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color=color, size=13))
    return fig


def _first_signal_channel(signal_df: pd.DataFrame) -> str:
    channels = [c for c in signal_df.columns if c != "datetime"]
    if not channels:
        raise ValueError("Signal dataframe has no data channels")
    return channels[0]


def _color_for_signal(signal: str, channel: str | None = None, fallback: str = "#95ccfe") -> str:
    sig = str(signal or "")
    ch = str(channel or "")
    keys = (f"{sig}.{ch}", f"{sig}:{ch}", ch, ch.lower(), ch.upper(), sig, sig.lower(), sig.upper())
    for key in keys:
        color = COLOR_MAPPING.get(key)
        if isinstance(color, str) and color.strip():
            return color.strip()
    return fallback


def _hex_to_rgba(color: str, alpha: float) -> str:
    color = str(color or "").strip()
    if not color.startswith("#") or len(color) not in (4, 7):
        return f"rgba(28,61,81,{alpha})"
    if len(color) == 4:
        color = "#" + "".join(ch * 2 for ch in color[1:])
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _align_timestamp_to_reference(ts, reference_ts):
    value = pd.Timestamp(ts)
    ref = pd.Timestamp(reference_ts)
    if ref.tzinfo is not None and value.tzinfo is None:
        return value.tz_localize(ref.tzinfo)
    if ref.tzinfo is not None and value.tzinfo is not None:
        return value.tz_convert(ref.tzinfo)
    if ref.tzinfo is None and value.tzinfo is not None:
        return value.tz_localize(None)
    return value


def _window_signal_df(signal_df: pd.DataFrame, t0=None, t1=None) -> pd.DataFrame:
    if signal_df is None or "datetime" not in signal_df.columns or (t0 is None and t1 is None):
        return signal_df
    dt = _parse_datetime(signal_df["datetime"])
    reference = dt.dropna().iloc[0] if not dt.dropna().empty else None
    mask = pd.Series(True, index=signal_df.index)
    if t0 is not None:
        start = _align_timestamp_to_reference(t0, reference) if reference is not None else pd.Timestamp(t0)
        mask &= dt >= start
    if t1 is not None:
        end = _align_timestamp_to_reference(t1, reference) if reference is not None else pd.Timestamp(t1)
        mask &= dt <= end
    return signal_df[mask]


def _series_range(values) -> list[float] | None:
    y = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if y.empty:
        return None
    lo = float(y.min())
    hi = float(y.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        pad = abs(lo) * 0.05 or 1.0
    else:
        pad = (hi - lo) * 0.05
    return [lo - pad, hi + pad]


def _axis_ranges_from_traces(fig: go.Figure) -> dict[str, list[float]]:
    axis_ranges: dict[str, list[float]] = {}
    for trace in fig.data:
        if str(getattr(trace, "name", "")).startswith("__"):
            continue
        if getattr(trace, "fill", None) == "toself":
            continue
        axis_ref = getattr(trace, "yaxis", None) or "y"
        values = getattr(trace, "y", None)
        if values is None:
            continue
        yr = _series_range(values)
        if yr is None:
            continue
        if axis_ref in axis_ranges:
            axis_ranges[axis_ref] = [
                min(axis_ranges[axis_ref][0], yr[0]),
                max(axis_ranges[axis_ref][1], yr[1]),
            ]
        else:
            axis_ranges[axis_ref] = yr
    return axis_ranges


def _pick_context_signal(signal_data: dict) -> tuple[str, str, pd.DataFrame] | None:
    if "depth" in signal_data:
        return "depth", "Depth", signal_data["depth"]
    for signal in ("accelerometer", "acceleration", "acc", "corrected_acc"):
        if signal in signal_data:
            return signal, "Accelerometer", signal_data[signal]
    return None


def _pick_rate_signal(signal_data: dict, mode: str) -> tuple[str, str, pd.DataFrame] | None:
    candidates = (
        ("heart_rate", "Heart Rate"),
        ("heart_rate_fixed", "Heart Rate"),
        ("heart_rate_nan", "Heart Rate"),
    ) if mode == "heart_rate" else (
        ("stroke_rate", "Stroke Rate"),
    )
    for signal, label in candidates:
        if signal in signal_data:
            return signal, label, signal_data[signal]
    return None


def _default_boundary_periods(data_pkl) -> list[dict[str, str]]:
    if data_pkl is None:
        return []
    sig = next(iter(getattr(data_pkl, "signal_data", {}).values()), None)
    if sig is None or "datetime" not in sig.columns:
        return []
    dt = _parse_datetime(sig["datetime"]).dropna()
    if dt.empty:
        return []
    return [{"start": str(dt.min()), "end": str(dt.max())}]


def _normalize_periods(periods) -> list[dict[str, str]]:
    normalized = []
    for period in periods or []:
        start = period.get("start") if isinstance(period, dict) else None
        end = period.get("end") if isinstance(period, dict) else None
        if not start or not end:
            continue
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
            continue
        normalized.append({"start": str(start_ts), "end": str(end_ts)})
    return normalized


def _boundary_periods_from_config(dataset, deployment) -> list[dict[str, str]]:
    if not dataset or not deployment:
        return []
    pm = _pm(dataset, deployment)
    periods = _normalize_periods(pm.get_logger_attachments())
    if periods:
        return periods
    return _default_boundary_periods(load_datapkl(dataset, deployment, DATA_DIR))


def _add_signal_trace(fig: go.Figure, signal_df: pd.DataFrame, signal: str, name: str, t0=None, t1=None) -> bool:
    if signal_df is None or "datetime" not in signal_df.columns:
        return False
    signal_df = _window_signal_df(signal_df, t0=t0, t1=t1)
    channel = _first_signal_channel(signal_df)
    dt = _parse_datetime(signal_df["datetime"])
    y = pd.to_numeric(signal_df[channel], errors="coerce")
    valid = dt.notna() & y.notna()
    if not valid.any():
        return False
    step = max(1, int(valid.sum()) // 6000)
    fig.add_trace(go.Scattergl(
        x=dt[valid][::step],
        y=y[valid][::step],
        mode="lines",
        line=dict(color=_color_for_signal(signal, channel), width=0.8),
        name=name,
        hovertemplate=f"{name}<br>%{{x}}<br>%{{y}}<extra></extra>",
    ))
    return True


def _add_overview_signal_trace(
    fig: go.Figure,
    signal_df: pd.DataFrame,
    signal: str,
    name: str,
    row: int,
    fill_depth: bool = False,
) -> bool:
    if signal_df is None or "datetime" not in signal_df.columns:
        return False
    channel = _first_signal_channel(signal_df)
    dt = _parse_datetime(signal_df["datetime"])
    y = pd.to_numeric(signal_df[channel], errors="coerce")
    valid = dt.notna() & y.notna()
    if not valid.any():
        return False
    step = max(1, int(valid.sum()) // 6000)
    color = _color_for_signal(signal, channel)
    fig.add_trace(go.Scattergl(
        x=dt[valid][::step],
        y=y[valid][::step],
        mode="lines",
        line=dict(color=color, width=1.6),
        fill="tozeroy" if fill_depth else None,
        fillcolor=_hex_to_rgba(color, 0.32) if fill_depth else None,
        name=name,
        hovertemplate=f"{name}<br>%{{x}}<br>%{{y}}<extra></extra>",
    ), row=row, col=1)
    return True


def _add_active_window(fig: go.Figure, t0, t1, axis_ranges: dict[str, list[float]] | None = None) -> None:
    if t0 is None or t1 is None:
        return
    _add_active_window_fill(fig, t0, t1, axis_ranges=axis_ranges)
    for edge_time in (t0, t1):
        fig.add_shape(
            name="__active-window-edge",
            type="line",
            xref="x",
            yref="paper",
            x0=str(edge_time),
            x1=str(edge_time),
            y0=0,
            y1=1,
            line=dict(color="rgba(246,231,168,0.32)", width=8),
            layer="above",
            editable=True,
        )


def _add_active_window_fill(
    fig: go.Figure,
    t0,
    t1,
    axis_ranges: dict[str, list[float]] | None = None,
) -> None:
    axis_ranges = axis_ranges or _axis_ranges_from_traces(fig)
    for axis_ref, yr in axis_ranges.items():
        row = 1 if axis_ref == "y" else 2 if axis_ref == "y2" else None
        if row is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[str(t0), str(t1), str(t1), str(t0), str(t0)],
                y=[yr[0], yr[0], yr[1], yr[1], yr[0]],
                mode="lines",
                fill="toself",
                fillcolor="rgba(246,231,168,0.18)",
                line=dict(color="rgba(246,231,168,0)", width=0),
                name="__active-window-fill",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )


def _make_deployment_overview_fig(data_pkl, mode: str, active_start=None, active_end=None, cursor_time=None):
    if data_pkl is None:
        return _empty_fig("No data.pkl found")
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.55],
    )

    rate = _pick_rate_signal(signal_data, mode)
    rate_plotted = False
    rate_label = "HR" if mode == "heart_rate" else "SR"
    if rate is not None:
        rate_signal, rate_label, rate_df = rate
        rate_plotted = _add_overview_signal_trace(fig, rate_df, rate_signal, rate_label, row=1)

    context = _pick_context_signal(signal_data)
    context_plotted = False
    context_signal = None
    if context is not None:
        context_signal, context_name, context_df = context
        context_plotted = _add_overview_signal_trace(
            fig,
            context_df,
            context_signal,
            context_name,
            row=2,
            fill_depth=context_signal == "depth",
        )

    if not rate_plotted:
        fig.add_annotation(
            text=f"No {rate_label} signal found",
            x=0.5,
            y=0.78,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#8eb0cb", size=11),
        )
    if not context_plotted:
        fig.add_annotation(
            text="No depth or accelerometer signal found",
            x=0.5,
            y=0.22,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#8eb0cb", size=11),
        )

    axis_ranges = _axis_ranges_from_traces(fig)
    _add_active_window(fig, active_start, active_end, axis_ranges=axis_ranges)
    _add_cursor_line(fig, cursor_time, rows=2)
    _add_cursor_markers(fig, cursor_time)
    yaxis_config = dict(
        showgrid=True,
        gridcolor="rgba(115,169,196,0.08)",
        color="#4a6070",
        fixedrange=True,
        title=rate_label,
    )
    if "y" in axis_ranges:
        yaxis_config["range"] = axis_ranges["y"]
    yaxis2_config = dict(
        showgrid=True,
        gridcolor="rgba(115,169,196,0.08)",
        color="#4a6070",
        fixedrange=True,
        title="Depth" if context_signal == "depth" else "Context",
    )
    if "y2" in axis_ranges:
        yaxis2_config["range"] = (
            [axis_ranges["y2"][1], axis_ranges["y2"][0]]
            if context_signal == "depth"
            else axis_ranges["y2"]
        )
    elif context_signal == "depth":
        yaxis2_config["autorange"] = "reversed"
    else:
        yaxis2_config["autorange"] = True
    fig.update_layout(
        **_dark_layout(),
        showlegend=True,
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=9, color="#8eb0cb"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#4a6070"),
        xaxis2=dict(showgrid=False, color="#4a6070"),
        yaxis=yaxis_config,
        yaxis2=yaxis2_config,
        margin=dict(l=50, r=20, t=12, b=28),
        dragmode="pan",
    )
    return fig


def _add_cursor_line(fig: go.Figure, cursor_time, rows: int = 2) -> None:
    if cursor_time is None:
        return
    for row in range(1, rows + 1):
        fig.add_vline(
            x=str(cursor_time),
            line_color="rgba(246,231,168,0.9)",
            line_width=1,
            line_dash="dot",
            row=row,
            col=1,
        )


def _add_cursor_markers(fig: go.Figure, cursor_time) -> None:
    if cursor_time is None:
        return
    cursor_ts = pd.Timestamp(cursor_time)
    marker_traces = []
    for trace in list(fig.data):
        if getattr(trace, "mode", "") == "markers" or str(getattr(trace, "name", "")).startswith("__cursor"):
            continue
        trace_x = getattr(trace, "x", None)
        trace_y = getattr(trace, "y", None)
        x_values = list(trace_x) if trace_x is not None else []
        y_values = list(trace_y) if trace_y is not None else []
        if not x_values or not y_values:
            continue
        parsed_x = _parse_datetime(x_values)
        valid = parsed_x.notna() & pd.to_numeric(pd.Series(y_values), errors="coerce").notna()
        if not valid.any():
            continue
        parsed_series = pd.Series(parsed_x)[valid].reset_index(drop=True)
        y_series = pd.to_numeric(pd.Series(y_values), errors="coerce")[valid].reset_index(drop=True)
        compare_ts = cursor_ts
        sample_ts = parsed_series.dropna().iloc[0] if not parsed_series.dropna().empty else None
        if sample_ts is not None:
            if sample_ts.tzinfo is not None and compare_ts.tzinfo is None:
                compare_ts = compare_ts.tz_localize(sample_ts.tzinfo)
            elif sample_ts.tzinfo is not None and compare_ts.tzinfo is not None:
                compare_ts = compare_ts.tz_convert(sample_ts.tzinfo)
            elif sample_ts.tzinfo is None and compare_ts.tzinfo is not None:
                compare_ts = compare_ts.tz_localize(None)
        deltas = (parsed_series - compare_ts).abs()
        if deltas.isna().all():
            continue
        idx = int(deltas.idxmin())
        marker_traces.append(go.Scattergl(
            x=[parsed_series.iloc[idx]],
            y=[y_series.iloc[idx]],
            mode="markers",
            marker=dict(color="#ffffff", size=7, line=dict(color="#021524", width=1)),
            name="__cursor",
            showlegend=False,
            hoverinfo="skip",
            xaxis=getattr(trace, "xaxis", None),
            yaxis=getattr(trace, "yaxis", None),
        ))
    for marker in marker_traces:
        fig.add_trace(marker)


def _hover_time(hover_data) -> pd.Timestamp | None:
    if not hover_data:
        return None
    x_value = hover_data.get("points", [{}])[0].get("x")
    if x_value is None:
        return None
    ts = pd.Timestamp(x_value)
    return None if pd.isna(ts) else ts


def _timestamp_in_window(ts, start, end) -> bool:
    if ts is None or start is None or end is None:
        return False
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    cmp_ts = pd.Timestamp(ts)
    if start_ts.tzinfo is not None and cmp_ts.tzinfo is None:
        cmp_ts = cmp_ts.tz_localize(start_ts.tzinfo)
    elif start_ts.tzinfo is not None and cmp_ts.tzinfo is not None:
        cmp_ts = cmp_ts.tz_convert(start_ts.tzinfo)
    elif start_ts.tzinfo is None and cmp_ts.tzinfo is not None:
        cmp_ts = cmp_ts.tz_localize(None)
    return start_ts <= cmp_ts <= end_ts


def _add_rate_signal_window(
    fig: go.Figure,
    signal_data: dict,
    mode: str,
    t0=None,
    t1=None,
) -> bool:
    rate = _pick_rate_signal(signal_data, mode)
    if rate is None:
        return False
    rate_signal, rate_name, rate_df = rate
    rate_df = _window_signal_df(rate_df, t0=t0, t1=t1)
    if rate_df is None or rate_df.empty or "datetime" not in rate_df.columns:
        return False
    channel = _first_signal_channel(rate_df)
    dt = _parse_datetime(rate_df["datetime"])
    y = pd.to_numeric(rate_df[channel], errors="coerce")
    valid = dt.notna() & y.notna()
    if not valid.any():
        return False
    step = max(1, int(valid.sum()) // 3000)
    fig.add_trace(go.Scattergl(
        x=dt[valid][::step],
        y=y[valid][::step],
        mode="lines",
        line=dict(color=_color_for_signal(rate_signal, channel), width=1.8),
        name=rate_name,
        hovertemplate=f"{rate_name}<br>%{{x}}<br>%{{y}}<extra></extra>",
    ), row=2, col=1)
    return True


def _add_detected_rate_trace(fig: go.Figure, peak_df: pd.DataFrame, fs: float, mode: str) -> bool:
    if peak_df.empty or "datetime" not in peak_df.columns:
        return False
    acc = peak_df[peak_df["key"].isin([
        "beat_auto_detect_accepted", "beat_auto_detect_suggested", "heartbeat_manual_ok",
    ])].copy()
    if len(acc) <= 1:
        return False
    if "refined_index" in acc.columns:
        acc = acc.sort_values("refined_index")
        rr_sec = np.diff(acc["refined_index"].to_numpy()) / fs
    else:
        acc = acc.assign(_parsed_datetime=_parse_datetime(acc["datetime"])).sort_values("_parsed_datetime")
        rr_sec = acc["_parsed_datetime"].diff().dt.total_seconds().iloc[1:].to_numpy()
    valid_rr = np.isfinite(rr_sec) & (rr_sec > 0)
    if not valid_rr.any():
        return False
    rr_sec = rr_sec[valid_rr]
    rate = 60.0 / rr_sec if mode == "heart_rate" else 1.0 / rr_sec
    rate_dt = _parse_datetime(acc["datetime"].iloc[1:].tolist())[valid_rr]
    label = "Heart Rate" if mode == "heart_rate" else "Stroke Rate"
    fig.add_trace(go.Scattergl(
        x=rate_dt,
        y=rate,
        mode="lines+markers",
        line=dict(color=_color_for_signal("heart_rate" if mode == "heart_rate" else "stroke_rate"), width=1.8),
        marker=dict(size=4, color=_color_for_signal("heart_rate" if mode == "heart_rate" else "stroke_rate")),
        name=label,
        hovertemplate=f"{label}<br>%{{x}}<br>%{{y}}<extra></extra>",
    ), row=2, col=1)
    return True


def _make_initial_signal_figs(data_pkl, mode: str = "heart_rate", t0=None, t1=None, cursor_time=None):
    if data_pkl is None:
        return _empty_fig("No data.pkl found"), _empty_fig()
    signal_data = getattr(data_pkl, "signal_data", {}) or {}

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
    )
    plotted = False
    plotted_dt = []
    if "ecg" in signal_data:
        ecg_df = _window_signal_df(signal_data["ecg"], t0=t0, t1=t1)
        ecg_channel = _first_signal_channel(ecg_df)
        ecg_dt = _parse_datetime(ecg_df["datetime"])
        ecg_y = pd.to_numeric(ecg_df[ecg_channel], errors="coerce")
        valid = ecg_dt.notna() & ecg_y.notna()
        if valid.any():
            step = max(1, int(valid.sum()) // 6000)
            fig.add_trace(go.Scattergl(
                x=ecg_dt[valid][::step],
                y=ecg_y[valid][::step],
                mode="lines",
                line=dict(color=_color_for_signal("ecg", ecg_channel), width=1.5),
                name="ECG",
                hovertemplate="ECG<br>%{x}<br>%{y}<extra></extra>",
            ), row=1, col=1)
            plotted = True
            plotted_dt.append(ecg_dt)

    if _add_rate_signal_window(fig, signal_data, mode, t0=t0, t1=t1):
        plotted = True
        rate = _pick_rate_signal(signal_data, mode)
        if rate is not None:
            plotted_dt.append(_parse_datetime(_window_signal_df(rate[2], t0=t0, t1=t1)["datetime"]))

    if not plotted:
        return _empty_fig("No ECG or heart-rate signal found"), _empty_fig()

    ranges = pd.concat([pd.Series(dt).dropna() for dt in plotted_dt if len(dt) > 0], ignore_index=True)
    rate_label = "Heart Rate" if mode == "heart_rate" else "Stroke Rate"
    title = f"ECG and {rate_label}"
    if not ranges.empty:
        title = f"ECG and {rate_label} · {ranges.min():%Y-%m-%d} to {ranges.max():%Y-%m-%d}"
    if _timestamp_in_window(cursor_time, ranges.min(), ranges.max()) if not ranges.empty else False:
        _add_cursor_line(fig, cursor_time, rows=2)
        _add_cursor_markers(fig, cursor_time)

    fig.update_layout(
        **_dark_layout(),
        title=dict(text=title, font=dict(size=12, color="#8eb0cb"), x=0.01),
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10, color="#8eb0cb"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#4a6070"),
        xaxis2=dict(showgrid=False, color="#4a6070"),
        yaxis=dict(showgrid=True, gridcolor="rgba(115,169,196,0.08)", color="#4a6070", fixedrange=True, title="ECG"),
        yaxis2=dict(
            showgrid=True,
            gridcolor="rgba(115,169,196,0.08)",
            color="#4a6070",
            fixedrange=True,
            title=rate_label,
        ),
        margin=dict(l=50, r=20, t=10, b=30),
        dragmode="pan",
    )
    return fig, _empty_fig()


# ── layout ────────────────────────────────────────────────────────────────────

datasets = _list_datasets()
_preferred = next((d for d in datasets if "oror" in d), None)
default_dataset = _preferred or (datasets[0] if datasets else "")
default_deployments = _list_deployments(default_dataset) if default_dataset else []
default_deployment = default_deployments[0] if default_deployments else ""

app = Dash(
    __name__,
    assets_folder=str(_HERE / "assets"),
    title="Peak Detection Review",
    suppress_callback_exceptions=True,
    prevent_initial_callbacks="initial_duplicate",
)

def _stage_step(num, label, active):
    cls = "stage-step stage-step--active" if active else "stage-step"
    return html.Div([
        html.Div(num, className="stage-num"),
        html.Span(label, className="stage-label"),
    ], className=cls)


def _first_component_value(values):
    """Return a singleton pattern-match value, or None if the component is absent."""
    if isinstance(values, list):
        return values[0] if values else None
    return values


def _parse_datetime(values):
    """Parse serialized datetimes that may mix fractional and whole seconds."""
    return pd.to_datetime(values, format="mixed", errors="coerce")


app.layout = html.Div([

    # ── global stores ──────────────────────────────────────────────────
    dcc.Store(id="store-stage", data="1", storage_type="memory"),
    dcc.Store(id="store-params", storage_type="memory"),
    dcc.Store(id="store-peaks", storage_type="memory"),
    dcc.Store(id="store-full-peaks", storage_type="memory"),   # stage-6 full-run
    dcc.Store(id="store-chunk-grid", storage_type="memory"),
    dcc.Store(id="store-chunk-idx", data=0, storage_type="memory"),
    dcc.Store(id="store-mode", data="heart_rate", storage_type="memory"),
    dcc.Store(id="store-calib-clicks", data=[], storage_type="memory"),  # stage-3 manual clicks
    dcc.Store(id="store-manual-edits", data=[], storage_type="memory"),  # stage-6/7 manual
    dcc.Store(id="store-calib-window", data=None, storage_type="memory"),  # {start, end}
    dcc.Store(id="store-zoom-window", data=None, storage_type="memory"),  # detail zoom override {start, end}
    dcc.Store(id="store-boundary-periods", data=None, storage_type="memory"),  # stage-2 [{start, end}]
    dcc.Store(id="store-btn", data=None, storage_type="memory"),  # button click dispatcher

    # ── top bar ────────────────────────────────────────────────────────
    html.Div([
        html.Div("Peak Detection Review", className="topbar-title"),
        html.Div([
            dcc.Dropdown(
                id="dd-dataset",
                options=[{"label": d, "value": d} for d in datasets],
                value=default_dataset,
                clearable=False,
                className="topbar-dropdown",
                placeholder="Dataset…",
            ),
            dcc.Dropdown(
                id="dd-deployment",
                options=[{"label": d, "value": d} for d in default_deployments],
                value=default_deployment,
                clearable=False,
                className="topbar-dropdown",
                placeholder="Deployment…",
            ),
            dcc.RadioItems(
                id="radio-mode",
                options=[
                    {"label": "Heart Rate", "value": "heart_rate"},
                    {"label": "Stroke Rate", "value": "stroke_rate"},
                ],
                value="heart_rate",
                className="mode-radio",
                inline=True,
            ),
        ], className="topbar-right"),
    ], className="topbar"),

    # ── stage progress bar ─────────────────────────────────────────────
    html.Div(id="stage-bar", className="stage-bar"),

    # ── main body ──────────────────────────────────────────────────────
    html.Div([

        # ── left panel: context-sensitive controls ─────────────────
        html.Div([
            html.Div(id="left-panel-content"),
        ], className="left-panel"),

        # ── center panel ───────────────────────────────────────────
        html.Div([

            # stage-specific hint bar
            html.Div(id="hint-bar", className="hint-bar"),

            # chunk navigator (stages 5+)
            html.Div([
                html.Button("◀", id="btn-prev-chunk", className="btn btn-nav", n_clicks=0),
                html.Div(id="chunk-strip", className="chunk-strip"),
                html.Button("▶", id="btn-next-chunk", className="btn btn-nav", n_clicks=0),
                html.Div(id="chunk-label", className="chunk-label"),
                html.Div(id="chunk-status-badge", className="chunk-badge"),
            ], id="chunk-nav-row", className="chunk-nav-row", style={"display": "none"}),

            # gap reason
            html.Div([
                dcc.Input(id="input-gap-reason", placeholder="Gap reason (optional)…",
                          className="gap-reason-input", debounce=True),
            ], id="gap-reason-row", className="gap-reason-row", style={"display": "none"}),

            # calibration window: 2-min draggable overview for stage 3
            html.Div([
                dcc.Graph(
                    id="graph-calib-overview",
                    config={"scrollZoom": True, "displayModeBar": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                    style={"height": "20vh"},
                ),
                html.Div(
                    "Drag the shaded window to a region with clear peaks, then click 5 peaks in the zoomed view below.",
                    className="hint", style={"padding": "4px 14px"},
                ),
            ], id="calib-overview-panel", style={"display": "none"}),

            # main signal plot (ECG chunk or calibration zoom)
            dcc.Graph(
                id="graph-peaks",
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
                className="peak-graph",
                style={"height": "50vh"},
            ),

            # HR / SR derived rate plot
            dcc.Graph(
                id="graph-rate",
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "edits": {"shapePosition": True},
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
                className="rate-graph",
                style={"height": "28vh"},
            ),

        ], className="center-panel"),

    ], className="main-body"),

    # status toast
    html.Div(id="toast", className="toast", style={"display": "none"}),
    html.Div([
        html.Div(id="preset-save-status"),
        html.Div(id="preset-desc"),
        html.Div(id="boundary-status"),
        html.Div(id="calib-click-status"),
        html.Div(id="calib-suggestion"),
        html.Div(id="save-status"),
        html.Div(id="full-run-status"),
        html.Div(id="manual-edit-count"),
        html.Div(id="save7-status"),
    ], id="callback-sinks", style={"display": "none"}),

], className="app-root")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE BAR
# ══════════════════════════════════════════════════════════════════════════════

def _boundary_signal(data_pkl, mode: str) -> tuple[str, pd.DataFrame] | None:
    if data_pkl is None:
        return None
    sig_key = "ecg" if mode == "heart_rate" else "corrected_gyr"
    sig_df = data_pkl.signal_data.get(sig_key)
    if sig_df is None:
        sig_df = next(iter(data_pkl.signal_data.values()), None)
        sig_key = "signal"
    if sig_df is None or "datetime" not in sig_df.columns:
        return None
    return sig_key, sig_df


def _add_boundary_signal_window(fig: go.Figure, sig_df: pd.DataFrame, signal: str, center, row: int, col: int) -> None:
    center_ts = pd.Timestamp(center)
    t0 = center_ts - pd.Timedelta(seconds=10)
    t1 = center_ts + pd.Timedelta(seconds=10)
    sub = _window_signal_df(sig_df, t0=t0, t1=t1)
    channel = _first_signal_channel(sub)
    dt = _parse_datetime(sub["datetime"])
    y = pd.to_numeric(sub[channel], errors="coerce")
    valid = dt.notna() & y.notna()
    if valid.any():
        step = max(1, int(valid.sum()) // 2500)
        fig.add_trace(
            go.Scattergl(
                x=dt[valid][::step],
                y=y[valid][::step],
                mode="lines",
                line=dict(color=_color_for_signal(signal, channel), width=1.2),
                name=channel,
                showlegend=False,
                hovertemplate=f"{channel}<br>%{{x}}<br>%{{y}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    fig.add_vline(
        x=str(center_ts),
        line_color="rgba(246,231,168,0.9)",
        line_width=2,
        row=row,
        col=col,
    )
    fig.update_xaxes(range=[str(t0), str(t1)], row=row, col=col)


def _make_boundary_detail_fig(data_pkl, mode: str, periods: list[dict[str, str]]) -> go.Figure:
    periods = _normalize_periods(periods)
    if data_pkl is None or not periods:
        return _empty_fig("No boundary periods")
    signal = _boundary_signal(data_pkl, mode)
    if signal is None:
        return _empty_fig("No signal found for boundary review")
    sig_key, sig_df = signal
    rows = len(periods)
    titles = []
    for i in range(rows):
        titles.extend([f"Period {i + 1} start", f"Period {i + 1} end"])
    fig = make_subplots(
        rows=rows,
        cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.04,
        vertical_spacing=0.08 if rows <= 2 else 0.04,
        subplot_titles=titles,
    )
    for i, period in enumerate(periods, start=1):
        _add_boundary_signal_window(fig, sig_df, sig_key, period["start"], row=i, col=1)
        _add_boundary_signal_window(fig, sig_df, sig_key, period["end"], row=i, col=2)
    fig.update_layout(
        **_dark_layout(),
        showlegend=False,
        margin=dict(l=45, r=20, t=28, b=24),
        dragmode="pan",
    )
    for row in range(1, rows + 1):
        for col in (1, 2):
            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(115,169,196,0.08)",
                color="#4a6070",
                fixedrange=True,
                row=row,
                col=col,
            )
            fig.update_xaxes(showgrid=False, color="#4a6070", row=row, col=col)
    return fig


def _add_boundary_period_window(fig: go.Figure, period: dict[str, str], index: int, axis_ranges: dict[str, list[float]]) -> None:
    start = period["start"]
    end = period["end"]
    for axis_ref, yr in axis_ranges.items():
        row = 1 if axis_ref == "y" else 2 if axis_ref == "y2" else None
        if row is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start, start],
                y=[yr[0], yr[0], yr[1], yr[1], yr[0]],
                mode="lines",
                fill="toself",
                fillcolor="rgba(85,211,138,0.16)",
                line=dict(color="rgba(85,211,138,0)", width=0),
                name="__boundary-fill",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
    for side, edge_time in (("start", start), ("end", end)):
        fig.add_shape(
            name=f"__boundary-edge:{index}:{side}",
            type="line",
            xref="x",
            yref="paper",
            x0=edge_time,
            x1=edge_time,
            y0=0,
            y1=1,
            line=dict(color="rgba(85,211,138,0.42)", width=8),
            layer="above",
            editable=True,
        )


def _make_boundary_overview_fig(data_pkl, mode: str, periods: list[dict[str, str]]) -> go.Figure:
    fig = _make_deployment_overview_fig(data_pkl, mode)
    fig.data = tuple(trace for trace in fig.data if not str(getattr(trace, "name", "")).startswith("__active-window"))
    fig.layout.shapes = tuple(
        shape for shape in (fig.layout.shapes or [])
        if not str(getattr(shape, "name", "")).startswith("__active-window")
    )
    axis_ranges = _axis_ranges_from_traces(fig)
    for i, period in enumerate(_normalize_periods(periods)):
        _add_boundary_period_window(fig, period, i, axis_ranges)
    if "y" in axis_ranges:
        fig.update_yaxes(range=axis_ranges["y"], row=1, col=1)
    if "y2" in axis_ranges:
        fig.update_yaxes(range=[axis_ranges["y2"][1], axis_ranges["y2"][0]], row=2, col=1)
    return fig


@app.callback(
    Output("stage-bar", "children"),
    Input("store-stage", "data"),
)
def render_stage_bar(stage):
    return [_stage_step(num, lbl, num == stage) for num, lbl in STAGES]


# ══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT SELECTOR
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("dd-deployment", "options"),
    Output("dd-deployment", "value"),
    Input("dd-dataset", "value"),
)
def update_deployments(dataset):
    if not dataset:
        return [], None
    deps = _list_deployments(dataset)
    return [{"label": d, "value": d} for d in deps], (deps[0] if deps else None)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("store-stage", "data"),
    Output("store-params", "data"),
    Output("store-peaks", "data"),
    Output("store-chunk-grid", "data"),
    Input("dd-deployment", "value"),
    Input("radio-mode", "value"),
    State("dd-dataset", "value"),
    prevent_initial_call=True,
)
def stage1_load(deployment, mode, dataset):
    """On deployment or mode change, detect existing results and build stage-1 panel."""
    if not dataset or not deployment:
        raise PreventUpdate

    pm = _pm(dataset, deployment)
    section = "hr_peak_detection_settings" if mode == "heart_rate" else "stroke_peak_detection_settings"
    all_keys = list(HR_PARAM_DEFS.keys()) + [k for k, _ in BOOL_PARAMS] + ["DETECTION_DERIVATIVE_CHANNELS"]
    loaded_params = pm.get_from_config(all_keys, section=section)
    has_params = any(v is not None for v in loaded_params.values())

    data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
    has_peaks = False
    peak_count = 0
    manual_count = 0
    if data_pkl is not None and hasattr(data_pkl, "event_data") and data_pkl.event_data is not None:
        ev = data_pkl.event_data
        auto_keys = ["heartbeat_auto_detect_accepted", "heartbeat_auto_detect_rejected",
                     "heartbeat_auto_detect_suggested"] if mode == "heart_rate" else [
                     "strokebeat_auto_detect_accepted"]
        pk = ev[ev["key"].isin(auto_keys)]
        has_peaks = len(pk) > 0
        peak_count = len(pk[pk["key"].str.endswith("_accepted")])
        manual_count = len(ev[ev["key"].isin(["heartbeat_manual_ok", "heartbeat_manual_reject"])])

    # Preset options
    mode_presets = list_presets(mode)
    # Also include any dataset-saved presets
    ds_presets = load_dataset_presets(pm)
    for slug, pdata in ds_presets.items():
        if pdata.get("_mode", "heart_rate") == mode:
            mode_presets.append({"value": f"custom:{slug}", "label": pdata.get("_label", slug),
                                  "description": "saved preset"})

    suggested = suggest_preset_for_dataset(dataset, mode)

    chunks = load_chunks(pm)

    # Build left panel
    if has_peaks:
        status_block = html.Div([
            html.Div("✓ Peak detection results found", className="status-found"),
            html.Div([
                html.Span(f"{peak_count} accepted peaks", className="stat-chip stat-ok"),
                html.Span(f"{manual_count} manual edits", className="stat-chip stat-manual") if manual_count else None,
            ], className="stat-row"),
            html.Hr(className="divider"),
            html.Div("Actions:", className="param-label", style={"marginBottom": "6px"}),
            html.Button("Load params + results →", id={"type": "stage-btn", "action": "btn-load-existing"},
                        className="btn btn-primary full-width", n_clicks=0),
            html.Div("or start fresh with a preset:", className="param-label",
                     style={"marginTop": "10px", "marginBottom": "4px"}),
        ])
    else:
        status_block = html.Div([
            html.Div("No existing peak detection found", className="status-missing"),
            html.Div("Choose a starting preset:", className="param-label",
                     style={"marginTop": "8px", "marginBottom": "4px"}),
        ])

    left_panel = html.Div([
        html.Div("Step 1 — Load", className="panel-heading"),
        status_block,
        dcc.Dropdown(
            id={"type": "stage-input", "key": "preset", "index": 0},
            options=[{"label": f"{p['label']}", "value": p["value"]} for p in mode_presets],
            value=suggested,
            clearable=False,
            placeholder="Select preset…",
            className="preset-dropdown",
        ),
        html.Div(className="preset-desc"),
        html.Button("Use preset →", id={"type": "stage-btn", "action": "btn-use-preset"},
                    className="btn btn-secondary full-width", n_clicks=0,
                    style={"marginTop": "8px"}),
        html.Hr(className="divider"),
        html.Div("Save current params as preset:", className="param-label"),
        dcc.Input(id={"type": "stage-input", "key": "preset-slug", "index": 0}, placeholder="slug e.g. myspecies-hr_v2",
                  className="gap-reason-input", style={"marginBottom": "4px"}),
        dcc.Input(id={"type": "stage-input", "key": "preset-label", "index": 0}, placeholder="Label e.g. My Species HR v2",
                  className="gap-reason-input", style={"marginBottom": "4px"}),
        html.Button("Save preset", id={"type": "stage-btn", "action": "btn-save-preset"},
                    className="btn btn-secondary full-width", n_clicks=0),
        html.Div(className="save-status"),
    ])

    hint = [html.Span(
        "Select a deployment, then load existing results or choose a preset to begin.",
        className="hint",
    )]

    # If loading, return stage 1 with stores cleared so user can choose
    # left-panel and hint-bar are rendered by the unified render_left_panel callback
    return "1", no_update, no_update, chunks


@app.callback(
    Output("preset-desc", "children"),
    Input({"type": "stage-input", "key": "preset", "index": ALL}, "value"),
)
def show_preset_desc(slug):
    slug = _first_component_value(slug)
    if not slug or slug.startswith("custom:"):
        return ""
    p = BUILTIN_PRESETS.get(slug, {})
    return p.get("_description", "")


@app.callback(
    Output("preset-save-status", "children"),
    Input("store-btn", "data"),
    State("store-params", "data"),
    State({"type": "stage-input", "key": "preset-slug", "index": ALL}, "value"),
    State({"type": "stage-input", "key": "preset-label", "index": ALL}, "value"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    State("radio-mode", "value"),
    prevent_initial_call=True,
)
def save_preset(btn_data, params, slug, label, dataset, deployment, mode):
    slug = _first_component_value(slug)
    label = _first_component_value(label)
    if not btn_data or btn_data.get("action") != "btn-save-preset" or not params or not slug:
        raise PreventUpdate
    pm = _pm(dataset, deployment)
    save_preset_to_param_manager(pm, slug, params, label or slug, mode)
    return f"✓ Saved preset '{slug}'"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — BOUNDARIES (on/off animal time windows)
# ══════════════════════════════════════════════════════════════════════════════

def _build_stage2_panel(dataset, deployment, mode, periods=None):
    attachments = _normalize_periods(periods) if periods is not None else _boundary_periods_from_config(dataset, deployment)
    entries = []
    for i, att in enumerate(attachments):
        entries.append(html.Div([
            html.Span(f"Period {i+1}", className="param-label"),
            dcc.Input(value=att.get("start", ""), placeholder="start (ISO)",
                      id={"type": "att-start", "index": i},
                      className="gap-reason-input", style={"marginBottom": "2px"}),
            dcc.Input(value=att.get("end", ""), placeholder="end (ISO)",
                      id={"type": "att-end", "index": i},
                      className="gap-reason-input"),
        ], className="attachment-entry"))
    if not entries:
        data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
        start_str = end_str = ""
        if data_pkl is not None:
            sig = next(iter(data_pkl.signal_data.values()), None)
            if sig is not None and "datetime" in sig.columns:
                dt = pd.to_datetime(sig["datetime"])
                start_str = str(dt.min())
                end_str = str(dt.max())
        entries = [html.Div([
            html.Span("Period 1", className="param-label"),
            dcc.Input(value=start_str, placeholder="start (ISO)",
                      id={"type": "att-start", "index": 0}, className="gap-reason-input",
                      style={"marginBottom": "2px"}),
            dcc.Input(value=end_str, placeholder="end (ISO)",
                      id={"type": "att-end", "index": 0}, className="gap-reason-input"),
        ], className="attachment-entry")]
    left = html.Div([
        html.Div("Step 2 — Boundaries", className="panel-heading"),
        html.Div("Mark when the tag was on the animal. These define chunk windows.", className="hint"),
        html.Div(entries, id="attachment-entries", style={"marginTop": "8px"}),
        html.Button("+ Add period", id="btn-add-period", className="btn btn-secondary", n_clicks=0,
                    style={"marginTop": "6px"}),
        html.Hr(className="divider"),
        html.Button("Save boundaries →", id={"type": "stage-btn", "action": "btn-save-boundaries"},
                    className="btn btn-primary full-width", n_clicks=0),
        html.Div(className="save-status"),
    ])
    hint = [
        html.Span("Set the time windows when the tag was on the animal.", className="hint"),
        html.Span("The shaded regions will define chunk boundaries.", className="hint"),
    ]
    return left, hint


def _make_context_fig(data_pkl, mode, attachments):
    """Quick overview of signal + attachment shading for stage 2."""
    fig = go.Figure().update_layout(**_dark_layout(
        margin=dict(l=50, r=20, t=20, b=40),
        showlegend=False,
        xaxis=dict(showgrid=False, color="#4a6070"),
        yaxis=dict(showgrid=True, gridcolor="rgba(115,169,196,0.08)", color="#4a6070"),
    ))
    if data_pkl is None:
        return fig
    sig_key = "ecg" if mode == "heart_rate" else "corrected_gyr"
    sig_df = data_pkl.signal_data.get(sig_key)
    if sig_df is None:
        sig_df = next(iter(data_pkl.signal_data.values()), None)
    if sig_df is None or "datetime" not in sig_df.columns:
        return fig
    channel = [c for c in sig_df.columns if c != "datetime"][0]
    dt = pd.to_datetime(sig_df["datetime"])
    # Downsample for speed
    step = max(1, len(sig_df) // 4000)
    fig.add_trace(go.Scattergl(
        x=dt[::step], y=sig_df[channel][::step],
        mode="lines", line=dict(color="rgba(149,204,254,0.4)", width=0.8),
        name=channel, hoverinfo="skip",
    ))
    for att in attachments:
        fig.add_vrect(
            x0=att.get("start"), x1=att.get("end"),
            fillcolor="rgba(85,211,138,0.08)",
            line_color="rgba(85,211,138,0.3)", line_width=1,
        )
    return fig




@app.callback(
    Output("store-boundary-periods", "data"),
    Input("store-stage", "data"),
    Input("dd-dataset", "value"),
    Input("dd-deployment", "value"),
    Input("btn-add-period", "n_clicks"),
    Input({"type": "att-start", "index": ALL}, "value"),
    Input({"type": "att-end", "index": ALL}, "value"),
    Input("graph-rate", "relayoutData"),
    State("store-boundary-periods", "data"),
    State("graph-rate", "figure"),
    prevent_initial_call=True,
)
def update_boundary_periods(stage, dataset, deployment, add_clicks, starts, ends, relayout, periods, overview_fig):
    if stage != "2" or not dataset or not deployment:
        raise PreventUpdate
    trigger = ctx.triggered_id
    trigger_prop = ctx.triggered[0]["prop_id"].split(".")[-1] if ctx.triggered else None

    if trigger in ("store-stage", "dd-dataset", "dd-deployment") or periods is None:
        return _boundary_periods_from_config(dataset, deployment)

    current = _normalize_periods(periods)
    if trigger == "btn-add-period":
        if current:
            last = current[-1]
            start_ts = pd.Timestamp(last["start"])
            end_ts = pd.Timestamp(last["end"])
            width = end_ts - start_ts
            if width <= pd.Timedelta(0):
                width = pd.Timedelta(minutes=2)
            current.append({"start": str(end_ts), "end": str(end_ts + width)})
        else:
            current = _boundary_periods_from_config(dataset, deployment)
        return current

    if isinstance(trigger, dict) and trigger.get("type") in ("att-start", "att-end"):
        edited = [
            {"start": s.strip(), "end": e.strip()}
            for s, e in zip(starts or [], ends or [])
            if s and e and s.strip() and e.strip()
        ]
        normalized = _normalize_periods(edited)
        if normalized:
            return normalized
        raise PreventUpdate

    if trigger == "graph-rate" and trigger_prop == "relayoutData" and relayout:
        if not overview_fig:
            raise PreventUpdate
        shapes = list(((overview_fig.get("layout") or {}).get("shapes") or []))
        for key, value in relayout.items():
            if not key.startswith("shapes["):
                continue
            close = key.find("]")
            if close < 8:
                continue
            try:
                idx = int(key[7:close])
            except ValueError:
                continue
            if idx >= len(shapes):
                continue
            prop = key.split("].", 1)[-1]
            if prop in {"x0", "x1", "y0", "y1"}:
                shapes[idx] = {**shapes[idx], prop: value}
        updated = [dict(p) for p in current]
        for shape in shapes:
            name = shape.get("name")
            if not isinstance(name, str) or not name.startswith("__boundary-edge:"):
                continue
            _, period_idx, side = name.split(":")
            period_idx = int(period_idx)
            if period_idx >= len(updated):
                continue
            value = shape.get("x0") or shape.get("x1")
            if value is None:
                continue
            updated[period_idx][side] = str(pd.Timestamp(value))
        normalized = _normalize_periods(updated)
        if normalized:
            return normalized

    raise PreventUpdate


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CALIBRATE (click 5 peaks to suggest params)
# ══════════════════════════════════════════════════════════════════════════════

def _build_stage3_panel(dataset, deployment, mode, params):
    left = html.Div([
        html.Div("Step 3 — Calibrate", className="panel-heading"),
        html.Div([
            html.Div("Drag the blue window in the overview to a region with clear, regular peaks.",
                     className="hint"),
            html.Div("Then click 5 peaks in the zoomed view below.", className="hint"),
        ], style={"marginBottom": "8px"}),
        html.Div("0 / 5 peaks clicked",
                 className="save-status", style={"fontSize": "12px", "color": "#95ccfe"}),
        html.Div(className="calib-suggestion"),
        html.Hr(className="divider"),
        html.Button("← Back to boundaries", id={"type": "stage-btn", "action": "btn-back-to-2"},
                    className="btn btn-secondary full-width", n_clicks=0),
        html.Button("Skip calibration →", id={"type": "stage-btn", "action": "btn-skip-calib"},
                    className="btn btn-secondary full-width", n_clicks=0,
                    style={"marginTop": "4px"}),
        html.Button("Accept suggestion →", id={"type": "stage-btn", "action": "btn-accept-suggestion"},
                    className="btn btn-primary full-width", n_clicks=0,
                    style={"marginTop": "4px"}),
    ])
    hint = [html.Span(
        "Click 5 clear peaks in the zoomed view. The app will suggest detection parameters.",
        className="hint",
    )]
    return left, hint


def _make_calib_zoom(sig_df, channel, t0, t1, params, mode, clicked_times):
    """Raw + processed overlay for the 2-min calibration window."""
    dt = pd.to_datetime(sig_df["datetime"])
    mask = (dt >= pd.Timestamp(t0)) & (dt <= pd.Timestamp(t1))
    sub = sig_df[mask]
    if sub.empty:
        return _empty_fig("No data in window")
    sub_dt = pd.to_datetime(sub["datetime"])
    raw = sub[channel].to_numpy(dtype=float)

    fig = go.Figure().update_layout(**_dark_layout(
        margin=dict(l=50, r=20, t=10, b=30),
        xaxis=dict(showgrid=False, color="#4a6070"),
        yaxis=dict(showgrid=True, gridcolor="rgba(115,169,196,0.08)", color="#4a6070", fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        dragmode="pan",
        clickmode="event+select",
    ))
    fig.add_trace(go.Scattergl(
        x=sub_dt, y=raw,
        mode="lines", line=dict(color=_color_for_signal("ecg" if mode == "heart_rate" else "corrected_gyr", fallback="#d09191"), width=0.8),
        name="raw", hoverinfo="skip",
    ))

    # Try running processing to get smoothed overlay
    try:
        fs = calculate_sampling_frequency(sub_dt.head())
        nyquist = fs / 2.0
        p = params or {}
        bl = max(0.01, min(p.get("BROAD_LOW_CUTOFF", 1.0), nyquist * 0.9))
        bh = max(0.01, min(p.get("BROAD_HIGH_CUTOFF", min(25.0, nyquist * 0.9)), nyquist * 0.99))
        if bl >= bh:
            bl = bh * 0.5
        results = peak_detect(
            signal=sub[channel],
            sampling_rate=fs,
            datetime_series=sub_dt,
            broad_lowcut=bl, broad_highcut=bh,
            narrow_lowcut=max(0.01, min(p.get("NARROW_LOW_CUTOFF", 5.0), nyquist * 0.9)),
            narrow_highcut=max(0.01, min(p.get("NARROW_HIGH_CUTOFF", min(15.0, nyquist * 0.9)), nyquist * 0.99)),
            filter_order=int(p.get("FILTER_ORDER", 2)),
            spike_threshold=p.get("SPIKE_THRESHOLD", 400),
            smooth_sec_multiplier=p.get("SMOOTH_SEC_MULTIPLIER", 0.36),
            window_size_multiplier=p.get("WINDOW_SIZE_MULTIPLIER", 6.35),
            normalization_noise=p.get("NORMALIZATION_NOISE", 1e-10),
            peak_height=p.get("PEAK_HEIGHT", -0.4),
            peak_distance_sec=p.get("PEAK_DISTANCE_SEC", 0.16),
            search_radius_sec=p.get("SEARCH_RADIUS_SEC", 0.2),
            min_peak_height=p.get("MIN_PEAK_HEIGHT", 70),
            max_peak_height=p.get("MAX_PEAK_HEIGHT", 12000),
            enable_bandpass=p.get("enable_bandpass", True),
            enable_spike_removal=p.get("enable_spike_removal", True),
            enable_absolute=p.get("enable_absolute", True) if mode == "heart_rate" else False,
            enable_smoothing=p.get("enable_smoothing", True),
            enable_normalization=p.get("enable_normalization", True),
            enable_refinement=p.get("enable_refinement", True),
            detection_sources=p.get("DETECTION_DERIVATIVE_CHANNELS", ["normalized"]),
        )
        smoothed = results.get("smoothed", np.array([]))
        if len(smoothed) == len(sub):
            fig.add_trace(go.Scattergl(
                x=sub_dt, y=smoothed,
                mode="lines", line=dict(color=_color_for_signal("hr_smoothed" if mode == "heart_rate" else "sr_smoothed", fallback="#9f76ba"), width=1.3),
                name="processed", hoverinfo="skip",
            ))
        # Show auto-detected peaks faintly
        pk_df = results.get("peak_df", pd.DataFrame())
        if not pk_df.empty and "datetime" in pk_df.columns:
            accepted = pk_df[pk_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"])]
            if not accepted.empty:
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(accepted["datetime"]),
                    y=accepted.get("height_original", pd.Series([0]*len(accepted))).fillna(0),
                    mode="markers",
                    marker=dict(color="rgba(85,211,138,0.4)", symbol="triangle-up", size=8),
                    name="auto peaks",
                ))
    except Exception:
        pass

    # Show manually clicked peaks
    for t in clicked_times:
        fig.add_vline(x=t, line_color="#95ccfe", line_width=1.5, line_dash="dot")

    return fig




@app.callback(
    Output("store-calib-clicks", "data", allow_duplicate=True),
    Output("calib-click-status", "children", allow_duplicate=True),
    Output("calib-suggestion", "children", allow_duplicate=True),
    Input("graph-peaks", "clickData"),
    State("store-stage", "data"),
    State("store-calib-clicks", "data"),
    State("store-params", "data"),
    prevent_initial_call=True,
)
def handle_calib_click(click_data, stage, clicks, params):
    if stage != "3" or not click_data:
        raise PreventUpdate
    pt = click_data["points"][0]
    t = pt.get("x")
    if t is None:
        raise PreventUpdate
    clicks = list(clicks or [])
    if t not in clicks:
        clicks.append(t)
        clicks = clicks[-5:]  # keep last 5

    status = f"{len(clicks)} / 5 peaks clicked"
    suggestion = []

    # Try to build suggestion from clicked intervals
    if len(clicks) >= 3:
        try:
            dts = sorted(pd.to_datetime(c) for c in clicks)
            rr_secs = [(dts[i+1] - dts[i]).total_seconds() for i in range(len(dts)-1)]
            median_rr = float(np.median(rr_secs))
            implied_hr = round(60 / median_rr, 1) if median_rr > 0 else None
            suggestion = _build_suggestion(median_rr, implied_hr, params)
        except Exception:
            pass

    return clicks, status, suggestion


def _build_suggestion(median_rr_sec, implied_hr, current_params):
    """Return a list of html elements describing suggested parameter changes."""
    suggestions = []
    p = current_params or {}

    min_rr = round(median_rr_sec * 0.5, 3)
    max_hr = round(min(500, 60.0 / min_rr * 1.3), 1) if min_rr > 0 else 300

    items = []
    if abs(p.get("MIN_RR_SEC", 0.25) - min_rr) > 0.02:
        items.append(f"MIN_RR_SEC: {p.get('MIN_RR_SEC', '?')} → {min_rr}s")
    if abs(p.get("MAX_HR_BPM", 240) - max_hr) > 10:
        items.append(f"MAX_HR_BPM: {p.get('MAX_HR_BPM', '?')} → {max_hr} bpm")
    if items:
        suggestions = [
            html.Div(f"Implied HR: ~{implied_hr} bpm  ·  RR: ~{round(median_rr_sec,3)}s",
                     className="calib-stat"),
            html.Div("Suggested adjustments:", className="param-label", style={"marginTop": "6px"}),
        ] + [html.Div(it, className="calib-item") for it in items]
    else:
        suggestions = [html.Div(f"✓ Current params look aligned (implied HR ~{implied_hr} bpm)",
                                className="calib-stat")]
    return suggestions


@app.callback(
    Output("store-params", "data", allow_duplicate=True),
    Output("store-stage", "data", allow_duplicate=True),
    Output("store-peaks", "data", allow_duplicate=True),
    Output("store-chunk-grid", "data", allow_duplicate=True),
    Output("toast", "children"),
    Output("toast", "style"),
    Output("gap-reason-row", "style"),
    Input("store-btn", "data"),
    State({"type": "stage-input", "key": "preset", "index": ALL}, "value"),
    State({"type": "att-start", "index": ALL}, "value"),
    State({"type": "att-end", "index": ALL}, "value"),
    State("store-boundary-periods", "data"),
    State("store-chunk-grid", "data"),
    State("store-chunk-idx", "data"),
    State("input-gap-reason", "value"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    State("radio-mode", "value"),
    prevent_initial_call=True,
)
def handle_stage_button_action(
    btn_data,
    preset_slug,
    starts,
    ends,
    boundary_periods,
    chunks,
    idx,
    gap_reason,
    dataset,
    deployment,
    mode,
):
    """Single writer for store mutations driven only by dynamic stage buttons."""
    if not btn_data:
        raise PreventUpdate

    action = btn_data.get("action")

    if action == "btn-load-existing":
        if not dataset or not deployment:
            raise PreventUpdate
        pm = _pm(dataset, deployment)
        section = "hr_peak_detection_settings" if mode == "heart_rate" else "stroke_peak_detection_settings"
        all_keys = list(HR_PARAM_DEFS.keys()) + [k for k, _ in BOOL_PARAMS] + ["DETECTION_DERIVATIVE_CHANNELS"]
        loaded = pm.get_from_config(all_keys, section=section)
        params = {k: v for k, v in loaded.items() if v is not None}
        _fill_defaults(params, mode)

        data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
        peak_store = _eventdata_to_store(data_pkl, mode) if data_pkl else {}
        return params, "5", peak_store, no_update, no_update, no_update, no_update

    if action == "btn-use-preset":
        preset_slug = _first_component_value(preset_slug)
        if not preset_slug:
            raise PreventUpdate
        if preset_slug.startswith("custom:"):
            if not dataset or not deployment:
                raise PreventUpdate
            pm = _pm(dataset, deployment)
            ds_presets = load_dataset_presets(pm)
            slug = preset_slug[7:]
            params = {k: v for k, v in (ds_presets.get(slug) or {}).items() if not k.startswith("_")}
        else:
            params = get_preset(preset_slug) or {}
        _fill_defaults(params, mode)
        return params, "2", no_update, no_update, no_update, no_update, no_update

    if action == "btn-save-boundaries":
        if not dataset or not deployment:
            raise PreventUpdate
        attachments = _normalize_periods(boundary_periods) or [
            {"start": s.strip(), "end": e.strip()}
            for s, e in zip(starts, ends)
            if s and e and s.strip() and e.strip()
        ]
        if not attachments:
            return no_update, no_update, no_update, no_update, "⚠ No valid periods entered", {"display": "block"}, no_update
        pm = _pm(dataset, deployment)
        pm.add_to_config({"logger_attachments": attachments}, section="settings")
        new_chunks, _ = compute_chunk_grid(attachments)
        pm.add_to_config({"hr_peak_detection_chunks": new_chunks})
        status = f"✓ Saved {len(attachments)} period(s), {len(new_chunks)} chunks"
        return no_update, "3", no_update, new_chunks, status, {"display": "block"}, {"display": "none"}

    if action in ("btn-validate", "btn-gap"):
        if not dataset or not deployment:
            raise PreventUpdate
        if not chunks or idx is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
        updated_chunks = [dict(c) for c in chunks]
        show_gap = {"display": "none"}
        if action == "btn-validate":
            updated_chunks[idx]["status"] = "validated"
            updated_chunks[idx]["params_override"] = None
        else:
            updated_chunks[idx]["status"] = "gap"
            if gap_reason:
                updated_chunks[idx]["gap_reason"] = gap_reason
            show_gap = {"display": "flex"}
        pm = _pm(dataset, deployment)
        pm.add_to_config({"hr_peak_detection_chunks": updated_chunks})
        return no_update, no_update, no_update, updated_chunks, no_update, no_update, show_gap

    stage_map = {
        "btn-back-to-2": "2",
        "btn-skip-calib": "4",
        "btn-accept-suggestion": "4",
        "btn-back-to-3": "3",
        "btn-go-to-5": "5",
        "btn-back-to-4": "4",
        "btn-go-to-6": "6",
        "btn-back-to-5": "5",
        "btn-back-to-6": "6",
    }

    target = stage_map.get(action)
    if target is None:
        raise PreventUpdate

    return no_update, target, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("store-btn", "data"),
    Input({"type": "stage-btn", "action": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _collect_btn_click(n_clicks_list):
    """Collect all pattern-matched button clicks and route to store-btn."""
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        raise PreventUpdate
    
    action = ctx.triggered_id.get("action")
    if not action:
        raise PreventUpdate
    
    # Get the triggered value (n_clicks)
    triggered_prop = ctx.triggered[0]
    if not triggered_prop.get("value"):
        raise PreventUpdate
    
    return {"action": action, "t": triggered_prop["value"]}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — TUNE (param sliders + preset comparison)
# ══════════════════════════════════════════════════════════════════════════════

def _build_stage4_panel(params, mode):
    p = params or {}
    defs = HR_PARAM_DEFS if mode == "heart_rate" else SR_PARAM_DEFS

    def _slider(key):
        if key not in defs:
            return None
        default, mn, mx, step, label = defs[key]
        current = p.get(key, default)
        return html.Div([
            html.Div([
                html.Span(label, className="param-label"),
                html.Span(str(round(current, 4) if isinstance(current, float) else current),
                          id={"type": "param-val", "key": key}, className="param-value"),
            ], className="param-row-header"),
            dcc.Slider(
                id={"type": "param-slider", "key": key},
                min=mn, max=mx, step=step, value=current,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
                className="param-slider",
            ),
        ], className="param-row")

    sections = []
    if mode == "heart_rate":
        sections += [
            _section("Cleanup Rules", [
                _slider("HR_JUMP_FRAC"),
                _slider("MIN_RR_SEC"),
                _slider("MAX_HR_BPM"),
                _slider("MIN_HR_BPM"),
                _slider("ANTI_DOUBLE_GAP_FACTOR"),
                _slider("ANTI_DOUBLE_ROLLING_WINDOW_SEC"),
                html.Div([
                    dcc.Checklist(id={"type": "stage-input", "key": "pick-last", "index": 0},
                        options=[{"label": "Pick last in conflict pair", "value": "yes"}],
                        value=["yes"] if p.get("PICK_LAST_IN_CONFLICT_PAIR", True) else [],
                        className="param-check"),
                ]),
            ], open_=True),
        ]
    sections += [
        _section("Signal Processing", [
            _slider("BROAD_LOW_CUTOFF"),
            _slider("BROAD_HIGH_CUTOFF"),
            _slider("NARROW_LOW_CUTOFF"),
            _slider("NARROW_HIGH_CUTOFF"),
            _slider("FILTER_ORDER"),
            _slider("SPIKE_THRESHOLD"),
            _slider("SMOOTH_SEC_MULTIPLIER"),
            _slider("WINDOW_SIZE_MULTIPLIER"),
            _slider("PEAK_HEIGHT"),
            _slider("PEAK_DISTANCE_SEC"),
            _slider("SEARCH_RADIUS_SEC"),
            _slider("MIN_PEAK_HEIGHT"),
            _slider("MAX_PEAK_HEIGHT"),
            html.Div([
                html.Span("Detection channels", className="param-label"),
                dcc.Checklist(
                    id={"type": "stage-input", "key": "check-channels", "index": 0},
                    options=[{"label": c, "value": c} for c in CHANNEL_OPTIONS],
                    value=p.get("DETECTION_DERIVATIVE_CHANNELS", ["normalized"]),
                    className="param-check",
                ),
            ], className="param-row"),
            html.Div([
                dcc.Checklist(
                    id={"type": "stage-input", "key": "check-enables", "index": 0},
                    options=[{"label": lbl, "value": key} for key, lbl in BOOL_PARAMS],
                    value=[key for key, _ in BOOL_PARAMS if p.get(key, True)],
                    className="param-check",
                ),
            ], className="param-row"),
        ], open_=False),
    ]

    left = html.Div([
        html.Div("Step 4 — Tune Parameters", className="panel-heading"),
        html.Div([
            html.Button("Load from config", id={"type": "stage-btn", "action": "btn-load-params"},
                        className="btn btn-secondary", n_clicks=0),
            html.Button("Save to config", id={"type": "stage-btn", "action": "btn-save-params"},
                        className="btn btn-primary", n_clicks=0),
        ], className="param-action-row"),
        html.Div(className="save-status"),
        *sections,
        html.Hr(className="divider"),
        html.Button("← Back to calibrate", id={"type": "stage-btn", "action": "btn-back-to-3"},
                    className="btn btn-secondary full-width", n_clicks=0),
        html.Button("Run chunk detection →", id={"type": "stage-btn", "action": "btn-go-to-5"},
                    className="btn btn-primary full-width", n_clicks=0,
                    style={"marginTop": "4px"}),
    ])

    hint = [html.Span("Adjust parameters then run chunk detection to preview results.", className="hint")]
    return left, hint


# ── Param slider sync (stage 4) ───────────────────────────────────────────────
@app.callback(
    Output("store-params", "data", allow_duplicate=True),
    Output("save-status", "children", allow_duplicate=True),
    Input({"type": "param-slider", "key": ALL}, "value"),
    Input({"type": "stage-input", "key": "pick-last", "index": ALL}, "value"),
    Input({"type": "stage-input", "key": "check-channels", "index": ALL}, "value"),
    Input({"type": "stage-input", "key": "check-enables", "index": ALL}, "value"),
    Input("store-btn", "data"),
    State({"type": "param-slider", "key": ALL}, "id"),
    State("store-params", "data"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    State("radio-mode", "value"),
    prevent_initial_call=True,
)
def sync_params(slider_vals, pick_last, channels, enables,
                btn_data, slider_ids, current_params, dataset, deployment, mode):
    try:
        trigger = ctx.triggered_id
    except Exception:
        trigger = None
    action = btn_data.get("action") if isinstance(btn_data, dict) else None
    if trigger == "store-btn" and action not in ("btn-load-params", "btn-save-params"):
        raise PreventUpdate

    section = "hr_peak_detection_settings" if mode == "heart_rate" else "stroke_peak_detection_settings"

    if trigger == "store-btn" and action == "btn-load-params" and dataset and deployment:
        pm = _pm(dataset, deployment)
        all_keys = list(HR_PARAM_DEFS.keys()) + [k for k, _ in BOOL_PARAMS] + ["DETECTION_DERIVATIVE_CHANNELS"]
        loaded = pm.get_from_config(all_keys, section=section)
        return loaded, "✓ Loaded from config"

    pick_last = _first_component_value(pick_last) or []
    channels = _first_component_value(channels) or ["normalized"]
    enables = _first_component_value(enables) or []
    params = dict(current_params or {})
    # Update from sliders
    for sid, val in zip(slider_ids, slider_vals):
        if val is not None:
            params[sid["key"]] = val
    params["PICK_LAST_IN_CONFLICT_PAIR"] = bool(pick_last)
    params["DETECTION_DERIVATIVE_CHANNELS"] = channels or ["normalized"]
    params["HR_CONFLICT_RR_FACTOR"] = params.get("ANTI_DOUBLE_GAP_FACTOR", 0.75)
    enables_set = set(enables or [])
    for key, _ in BOOL_PARAMS:
        params[key] = key in enables_set

    if trigger == "store-btn" and action == "btn-save-params" and dataset and deployment:
        pm = _pm(dataset, deployment)
        pm.add_to_config(entries=params, section=section)
        return params, "✓ Saved to config"

    return params, no_update


@app.callback(
    Output({"type": "param-val", "key": MATCH}, "children"),
    Input({"type": "param-slider", "key": MATCH}, "value"),
)
def update_slider_label(val):
    if val is None:
        return ""
    return str(round(val, 4) if isinstance(val, float) else val)



# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — CHUNK RUN
# ══════════════════════════════════════════════════════════════════════════════

def _build_stage5_panel():
    left = html.Div([
        html.Div("Step 5 — Chunk Run", className="panel-heading"),
        html.Div("Running detection on ~10% of the deployment (one chunk).", className="hint"),
        html.Hr(className="divider"),
        html.Button("⟳ Run this chunk", id={"type": "stage-btn", "action": "btn-rerun"}, className="btn btn-primary full-width", n_clicks=0),
        html.Div([
            html.Button("✓ Validate chunk", id={"type": "stage-btn", "action": "btn-validate"},
                        className="btn btn-ok", n_clicks=0),
            html.Button("✗ Mark gap", id={"type": "stage-btn", "action": "btn-gap"},
                        className="btn btn-gap", n_clicks=0),
        ], className="param-action-row", style={"marginTop": "8px"}),
        html.Hr(className="divider"),
        html.Button("← Back to tune", id={"type": "stage-btn", "action": "btn-back-to-4"},
                    className="btn btn-secondary full-width", n_clicks=0),
        html.Button("Run full deployment →", id={"type": "stage-btn", "action": "btn-go-to-6"},
                    className="btn btn-primary full-width", n_clicks=0,
                    style={"marginTop": "4px"}),
    ])
    hint = [
        html.Span("Navigate chunks with ◀ / ▶. Run detection, then validate or mark gaps.", className="hint"),
        html.Span("When all chunks look good, proceed to full run.", className="hint"),
    ]
    return left, hint


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — FULL RUN + LASSO EDIT
# ══════════════════════════════════════════════════════════════════════════════

def _build_stage6_panel(manual_edits):
    n_manual = len(manual_edits or [])
    left = html.Div([
        html.Div("Step 6 — Full Run & Edit", className="panel-heading"),
        html.Button("⟳ Run full deployment", id={"type": "stage-btn", "action": "btn-run-full"},
                    className="btn btn-primary full-width", n_clicks=0),
        html.Div(className="save-status"),
        html.Hr(className="divider"),
        html.Div("Lasso-select peaks, then:", className="param-label"),
        html.Div([
            html.Button("✓ Accept selected", id={"type": "stage-btn", "action": "btn-manual-ok"},
                        className="btn btn-ok", n_clicks=0),
            html.Button("✗ Reject selected", id={"type": "stage-btn", "action": "btn-manual-reject"},
                        className="btn btn-gap", n_clicks=0),
        ], className="param-action-row"),
        html.Div([
            html.Div("Keyboard: A = accept, R = reject (selected peaks)", className="hint"),
        ]),
        html.Hr(className="divider"),
        html.Div(f"{n_manual} manual edit(s) pending",
                 className="save-status"),
        html.Hr(className="divider"),
        html.Button("← Back to chunk run", id={"type": "stage-btn", "action": "btn-back-to-5"},
                    className="btn btn-secondary full-width", n_clicks=0),
        html.Button("Save & lock edits →", id={"type": "stage-btn", "action": "btn-go-to-7"},
                    className="btn btn-primary full-width", n_clicks=0,
                    style={"marginTop": "4px"}),
    ])
    hint = [
        html.Span("Lasso peaks in the plot, then press Accept or Reject. Use lasso2d mode in the toolbar.",
                  className="hint"),
        html.Span("Changes are saved when you proceed to Save.", className="hint"),
    ]
    return left, hint


@app.callback(
    Output("store-full-peaks", "data", allow_duplicate=True),
    Output("full-run-status", "children"),
    Input("store-btn", "data"),
    State("store-params", "data"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    State("radio-mode", "value"),
    prevent_initial_call=True,
)
def run_full_deployment(btn_data, params, dataset, deployment, mode):
    if not btn_data or btn_data.get("action") != "btn-run-full":
        raise PreventUpdate
    if not dataset or not deployment or not params:
        raise PreventUpdate
    data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
    if data_pkl is None:
        return no_update, "⚠ No data found"

    sig_key = "ecg" if mode == "heart_rate" else "corrected_gyr"
    sig_df = data_pkl.signal_data.get(sig_key)
    if sig_df is None:
        return no_update, f"⚠ Signal '{sig_key}' not found"

    channel = [c for c in sig_df.columns if c != "datetime"][0]
    dt_col = pd.to_datetime(sig_df["datetime"])
    fs = calculate_sampling_frequency(dt_col.head())

    p = params
    nyquist = fs / 2.0
    bl = max(0.01, min(p.get("BROAD_LOW_CUTOFF", 1.0), nyquist * 0.99))
    bh = max(0.01, min(p.get("BROAD_HIGH_CUTOFF", min(25.0, nyquist * 0.9)), nyquist * 0.99))
    nl = max(0.01, min(p.get("NARROW_LOW_CUTOFF", 5.0), nyquist * 0.99))
    nh = max(0.01, min(p.get("NARROW_HIGH_CUTOFF", min(15.0, nyquist * 0.9)), nyquist * 0.99))
    if bl >= bh: bl = bh * 0.5
    if nl >= nh: nl = nh * 0.5

    try:
        results = peak_detect(
            signal=sig_df[channel],
            sampling_rate=fs,
            datetime_series=dt_col,
            broad_lowcut=bl, broad_highcut=bh,
            narrow_lowcut=nl, narrow_highcut=nh,
            filter_order=int(p.get("FILTER_ORDER", 2)),
            spike_threshold=p.get("SPIKE_THRESHOLD", 400),
            smooth_sec_multiplier=p.get("SMOOTH_SEC_MULTIPLIER", 0.36),
            window_size_multiplier=p.get("WINDOW_SIZE_MULTIPLIER", 6.35),
            normalization_noise=p.get("NORMALIZATION_NOISE", 1e-10),
            peak_height=p.get("PEAK_HEIGHT", -0.4),
            peak_distance_sec=p.get("PEAK_DISTANCE_SEC", 0.16),
            search_radius_sec=p.get("SEARCH_RADIUS_SEC", 0.2),
            min_peak_height=p.get("MIN_PEAK_HEIGHT", 70),
            max_peak_height=p.get("MAX_PEAK_HEIGHT", 12000),
            enable_bandpass=p.get("enable_bandpass", True),
            enable_spike_removal=p.get("enable_spike_removal", True),
            enable_absolute=p.get("enable_absolute", True) if mode == "heart_rate" else False,
            enable_smoothing=p.get("enable_smoothing", True),
            enable_normalization=p.get("enable_normalization", True),
            enable_refinement=p.get("enable_refinement", True),
            detection_sources=p.get("DETECTION_DERIVATIVE_CHANNELS", ["normalized"]),
        )
    except Exception as exc:
        return no_update, f"⚠ Detection failed: {exc}"

    peak_df = results.get("peak_df", pd.DataFrame())
    if not peak_df.empty and mode == "heart_rate":
        peak_df = _hr_cleanup(peak_df, results.get("smoothed", np.array([])),
                              sig_df, params, fs)

    smoothed = results.get("smoothed", np.array([]))
    store = {
        "peak_df": peak_df.to_dict("records") if not peak_df.empty else [],
        "smoothed": smoothed.tolist() if hasattr(smoothed, "tolist") else [],
        "datetime": dt_col.astype(str).tolist(),
        "signal": sig_df[channel].tolist(),
        "fs": fs,
        "mode": mode,
        "gap": False,
    }
    n_acc = len(peak_df[peak_df["key"].isin(["beat_auto_detect_accepted",
                                              "beat_auto_detect_suggested"])]) if not peak_df.empty else 0
    return store, f"✓ Full run complete — {n_acc} accepted peaks"


@app.callback(
    Output("store-manual-edits", "data", allow_duplicate=True),
    Output("manual-edit-count", "children"),
    Input("store-btn", "data"),
    State("graph-peaks", "selectedData"),
    State("store-manual-edits", "data"),
    prevent_initial_call=True,
)
def handle_manual_edit(btn_data, selected, edits):
    if not btn_data or btn_data.get("action") not in ("btn-manual-ok", "btn-manual-reject"):
        raise PreventUpdate
    if not selected:
        raise PreventUpdate
    action = "ok" if btn_data.get("action") == "btn-manual-ok" else "reject"
    edits = list(edits or [])
    for pt in selected.get("points", []):
        t = pt.get("x")
        if t:
            # Remove any prior edit for this exact time
            edits = [e for e in edits if e.get("datetime") != t]
            edits.append({"datetime": t, "action": action})
    return edits, f"{len(edits)} manual edit(s) pending"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — SAVE
# ══════════════════════════════════════════════════════════════════════════════

def _build_stage7_panel(manual_edits, dataset, deployment):
    n_manual = len(manual_edits or [])
    backups = list_backups(dataset, deployment, DATA_DIR) if dataset and deployment else []
    backup_items = [html.Div(b, className="param-label") for b in backups[:5]]

    left = html.Div([
        html.Div("Step 7 — Save", className="panel-heading"),
        html.Div(f"{n_manual} manual edit(s) to save.", className="hint"),
        html.Button("💾 Save manual edits to data.pkl", id={"type": "stage-btn", "action": "btn-save-manual"},
                    className="btn btn-primary full-width", n_clicks=0),
        html.Button("💾 Save auto peaks to data.pkl", id={"type": "stage-btn", "action": "btn-save-auto"},
                    className="btn btn-secondary full-width", n_clicks=0,
                    style={"marginTop": "4px"}),
        html.Div(className="save-status"),
        html.Hr(className="divider"),
        html.Div("Recent backups:", className="param-label"),
        html.Div(backup_items or [html.Div("No backups yet", className="param-label")],
                 style={"maxHeight": "120px", "overflowY": "auto"}),
        html.Hr(className="divider"),
        html.Button("← Back to full run", id={"type": "stage-btn", "action": "btn-back-to-6"},
                    className="btn btn-secondary full-width", n_clicks=0),
    ])
    hint = [
        html.Span("Manual edits are saved into event_data and will survive future auto-runs.",
                  className="hint"),
        html.Span("A timestamped backup is created automatically before each save.", className="hint"),
    ]
    return left, hint


# ── save edits (stage 7) ──────────────────────────────────────────────────────
@app.callback(
    Output("save7-status", "children"),
    Input("store-btn", "data"),
    State("store-manual-edits", "data"),
    State("store-full-peaks", "data"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    prevent_initial_call=True,
)
def save_edits(btn_data, manual_edits, full_peaks, dataset, deployment):
    if not btn_data or btn_data.get("action") not in ("btn-save-manual", "btn-save-auto"):
        raise PreventUpdate
    if not dataset or not deployment:
        raise PreventUpdate
    data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
    if data_pkl is None:
        return "⚠ No data.pkl found"

    action = btn_data.get("action")
    if action == "btn-save-manual":
        edits = manual_edits or []
        if not edits:
            return "No manual edits to save."
        apply_manual_edits(data_pkl, edits)
        backup = save_datapkl(data_pkl, dataset, deployment, DATA_DIR)
        return f"✓ Saved {len(edits)} manual edit(s). Backup: {os.path.basename(backup)}"

    if action == "btn-save-auto":
        if not full_peaks or not full_peaks.get("peak_df"):
            return "No full-run peaks to save. Run full deployment first."
        peak_df = pd.DataFrame(full_peaks["peak_df"])
        write_auto_peaks_to_events(data_pkl, peak_df)
        backup = save_datapkl(data_pkl, dataset, deployment, DATA_DIR)
        n = len(peak_df[peak_df["key"].isin(["beat_auto_detect_accepted",
                                              "beat_auto_detect_suggested"])])
        return f"✓ Saved {n} auto peaks. Backup: {os.path.basename(backup)}"

    raise PreventUpdate


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED LEFT PANEL + HINT DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("left-panel-content", "children"),
    Output("hint-bar", "children"),
    Output("chunk-nav-row", "style"),
    Output("calib-overview-panel", "style"),
    Input("store-stage", "data"),
    Input("store-boundary-periods", "data"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    State("radio-mode", "value"),
    State("store-params", "data"),
    State("store-manual-edits", "data"),
    prevent_initial_call=True,
)
def render_left_panel(stage, boundary_periods, dataset, deployment, mode, params, manual_edits):
    chunk_nav = {"display": "none"}
    calib_panel = {"display": "none"}

    if stage == "1":
        if not dataset or not deployment:
            raise PreventUpdate
        # Rebuild stage-1 panel (same logic as stage1_load but for display only)
        pm = _pm(dataset, deployment)
        section = "hr_peak_detection_settings" if mode == "heart_rate" else "stroke_peak_detection_settings"
        all_keys = list(HR_PARAM_DEFS.keys()) + [k for k, _ in BOOL_PARAMS] + ["DETECTION_DERIVATIVE_CHANNELS"]
        loaded_params = pm.get_from_config(all_keys, section=section)
        data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
        has_peaks = False
        peak_count = 0
        manual_count = 0
        if data_pkl is not None and hasattr(data_pkl, "event_data") and data_pkl.event_data is not None:
            ev = data_pkl.event_data
            auto_keys = (["heartbeat_auto_detect_accepted", "heartbeat_auto_detect_rejected",
                          "heartbeat_auto_detect_suggested"] if mode == "heart_rate"
                         else ["strokebeat_auto_detect_accepted"])
            pk = ev[ev["key"].isin(auto_keys)]
            has_peaks = len(pk) > 0
            peak_count = len(pk[pk["key"].str.endswith("_accepted")])
            manual_count = len(ev[ev["key"].isin(["heartbeat_manual_ok", "heartbeat_manual_reject"])])
        mode_presets = list_presets(mode)
        ds_presets = load_dataset_presets(pm)
        for slug, pdata in ds_presets.items():
            if pdata.get("_mode", "heart_rate") == mode:
                mode_presets.append({"value": f"custom:{slug}", "label": pdata.get("_label", slug),
                                     "description": "saved preset"})
        suggested = suggest_preset_for_dataset(dataset, mode)
        if has_peaks:
            status_block = html.Div([
                html.Div("✓ Peak detection results found", className="status-found"),
                html.Div([
                    html.Span(f"{peak_count} accepted peaks", className="stat-chip stat-ok"),
                    html.Span(f"{manual_count} manual edits", className="stat-chip stat-manual") if manual_count else None,
                ], className="stat-row"),
                html.Hr(className="divider"),
                html.Div("Actions:", className="param-label", style={"marginBottom": "6px"}),
                html.Button("Load params + results →", id={"type": "stage-btn", "action": "btn-load-existing"},
                            className="btn btn-primary full-width", n_clicks=0),
                html.Div("or start fresh with a preset:", className="param-label",
                         style={"marginTop": "10px", "marginBottom": "4px"}),
            ])
        else:
            status_block = html.Div([
                html.Div("No existing peak detection found", className="status-missing"),
                html.Div("Choose a starting preset:", className="param-label",
                         style={"marginTop": "8px", "marginBottom": "4px"}),
            ])
        left = html.Div([
            html.Div("Step 1 — Load", className="panel-heading"),
            status_block,
            dcc.Dropdown(id={"type": "stage-input", "key": "preset", "index": 0},
                options=[{"label": p["label"], "value": p["value"]} for p in mode_presets],
                value=suggested, clearable=False, className="preset-dropdown"),
            html.Div(className="preset-desc"),
            html.Button("Use preset →", id={"type": "stage-btn", "action": "btn-use-preset"},
                        className="btn btn-secondary full-width", n_clicks=0,
                        style={"marginTop": "8px"}),
            html.Hr(className="divider"),
            html.Div("Save current params as preset:", className="param-label"),
            dcc.Input(id={"type": "stage-input", "key": "preset-slug", "index": 0}, placeholder="slug e.g. myspecies-hr_v2",
                      className="gap-reason-input", style={"marginBottom": "4px"}),
            dcc.Input(id={"type": "stage-input", "key": "preset-label", "index": 0}, placeholder="Label e.g. My Species HR v2",
                      className="gap-reason-input", style={"marginBottom": "4px"}),
            html.Button("Save preset", id={"type": "stage-btn", "action": "btn-save-preset"},
                        className="btn btn-secondary full-width", n_clicks=0),
            html.Div(className="save-status"),
        ])
        hint = [html.Span(
            "Select a deployment, then load existing results or choose a preset to begin.",
            className="hint",
        )]
        return left, hint, chunk_nav, calib_panel

    elif stage == "2":
        if not dataset or not deployment:
            raise PreventUpdate
        left, hint = _build_stage2_panel(dataset, deployment, mode, boundary_periods)
        return left, hint, chunk_nav, calib_panel

    elif stage == "3":
        left, hint = _build_stage3_panel(dataset, deployment, mode, params)
        return left, hint, chunk_nav, {"display": "block"}

    elif stage == "4":
        left, hint = _build_stage4_panel(params, mode)
        return left, hint, chunk_nav, calib_panel

    elif stage == "5":
        left, hint = _build_stage5_panel()
        return left, hint, {"display": "flex"}, calib_panel

    elif stage == "6":
        left, hint = _build_stage6_panel(manual_edits)
        return left, hint, chunk_nav, calib_panel

    elif stage == "7":
        left, hint = _build_stage7_panel(manual_edits, dataset, deployment)
        return left, hint, chunk_nav, calib_panel

    raise PreventUpdate


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK NAVIGATOR (stages 5+)
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("store-chunk-idx", "data", allow_duplicate=True),
    Input("btn-prev-chunk", "n_clicks"),
    Input("btn-next-chunk", "n_clicks"),
    Input({"type": "chunk-pill", "index": ALL}, "n_clicks"),
    Input("graph-rate", "clickData"),
    State("store-chunk-idx", "data"),
    State("store-chunk-grid", "data"),
    prevent_initial_call=True,
)
def navigate_chunks(n_prev, n_next, pill_clicks, overview_click, current_idx, chunks):
    if not chunks:
        return 0
    try:
        trigger = ctx.triggered_id
    except Exception:
        trigger = None
    n = len(chunks)
    if trigger == "btn-prev-chunk":
        return max(0, current_idx - 1)
    if trigger == "btn-next-chunk":
        return min(n - 1, current_idx + 1)
    if isinstance(trigger, dict) and trigger.get("type") == "chunk-pill":
        return trigger["index"]
    if trigger == "graph-rate" and overview_click:
        click_time = overview_click.get("points", [{}])[0].get("x")
        if click_time is None:
            return current_idx
        click_ts = pd.Timestamp(click_time)
        for i, chunk in enumerate(chunks):
            start = pd.Timestamp(chunk["start"])
            end = pd.Timestamp(chunk["end"])
            if start.tzinfo is not None and click_ts.tzinfo is None:
                click_cmp = click_ts.tz_localize(start.tzinfo)
            elif start.tzinfo is not None and click_ts.tzinfo is not None:
                click_cmp = click_ts.tz_convert(start.tzinfo)
            else:
                click_cmp = click_ts.tz_localize(None) if click_ts.tzinfo is not None else click_ts
            if start <= click_cmp <= end:
                return i
    return current_idx


@app.callback(
    Output("store-zoom-window", "data"),
    Input("graph-rate", "selectedData"),
    Input("graph-rate", "relayoutData"),
    Input("graph-rate", "clickData"),
    State("store-zoom-window", "data"),
    State("store-chunk-grid", "data"),
    State("store-chunk-idx", "data"),
    State("graph-rate", "figure"),
    prevent_initial_call=True,
)
def update_zoom_window_from_overview(selected, relayout, click_data, current_window, chunks, chunk_idx, overview_fig):
    try:
        trigger = ctx.triggered_id
        trigger_prop = ctx.triggered[0]["prop_id"].split(".")[-1] if ctx.triggered else None
    except Exception:
        trigger = None
        trigger_prop = None

    def _window(start, end):
        if start is None or end is None:
            raise PreventUpdate
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if current_window:
            ref = pd.Timestamp(current_window["start"])
            start_ts = _align_timestamp_to_reference(start_ts, ref)
            end_ts = _align_timestamp_to_reference(end_ts, ref)
        if end_ts <= start_ts:
            raise PreventUpdate
        return {"start": str(start_ts), "end": str(end_ts)}

    def _shape_index(key: str) -> int | None:
        if not key.startswith("shapes["):
            return None
        close = key.find("]")
        if close < 8:
            return None
        try:
            return int(key[7:close])
        except ValueError:
            return None

    def _window_from_active_edges() -> dict | None:
        if not overview_fig:
            return None
        shapes = list(((overview_fig.get("layout") or {}).get("shapes") or []))
        if not shapes:
            return None
        for key, value in (relayout or {}).items():
            idx = _shape_index(key)
            if idx is None or idx >= len(shapes):
                continue
            prop = key.split("].", 1)[-1]
            if prop in {"x0", "x1", "y0", "y1"}:
                shapes[idx] = {**shapes[idx], prop: value}
        edge_values = []
        for shape in shapes:
            if shape.get("name") != "__active-window-edge":
                continue
            x0 = shape.get("x0")
            x1 = shape.get("x1")
            edge_values.extend([value for value in (x0, x1) if value is not None])
        if len(edge_values) < 2:
            return None
        return _window(min(edge_values, key=pd.Timestamp), max(edge_values, key=pd.Timestamp))

    if trigger == "graph-rate" and trigger_prop == "selectedData" and selected:
        range_x = (selected.get("range") or {}).get("x")
        if range_x and len(range_x) == 2:
            return _window(range_x[0], range_x[1])
        points = selected.get("points") or []
        xs = [p.get("x") for p in points if p.get("x") is not None]
        if len(xs) >= 2:
            return _window(min(xs), max(xs))

    if trigger == "graph-rate" and trigger_prop == "relayoutData" and relayout:
        if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            return _window(relayout["xaxis.range[0]"], relayout["xaxis.range[1]"])
        if "xaxis2.range[0]" in relayout and "xaxis2.range[1]" in relayout:
            return _window(relayout["xaxis2.range[0]"], relayout["xaxis2.range[1]"])

        shape_relayout = False
        for key, value in relayout.items():
            if key.startswith("shapes["):
                shape_relayout = True
        if shape_relayout:
            edge_window = _window_from_active_edges()
            if edge_window is not None:
                return edge_window
            if current_window:
                return {"start": current_window["start"], "end": current_window["end"]}

    if trigger == "graph-rate" and trigger_prop == "clickData" and click_data and chunks:
        click_time = click_data.get("points", [{}])[0].get("x")
        if click_time is None:
            raise PreventUpdate
        click_ts = pd.Timestamp(click_time)
        for chunk in chunks:
            start = pd.Timestamp(chunk["start"])
            end = pd.Timestamp(chunk["end"])
            if start.tzinfo is not None and click_ts.tzinfo is None:
                click_cmp = click_ts.tz_localize(start.tzinfo)
            elif start.tzinfo is not None and click_ts.tzinfo is not None:
                click_cmp = click_ts.tz_convert(start.tzinfo)
            else:
                click_cmp = click_ts.tz_localize(None) if click_ts.tzinfo is not None else click_ts
            if start <= click_cmp <= end:
                return {"start": str(start), "end": str(end)}

    raise PreventUpdate


@app.callback(
    Output("chunk-strip", "children"),
    Output("chunk-label", "children"),
    Output("chunk-status-badge", "children"),
    Output("chunk-status-badge", "style"),
    Input("store-chunk-grid", "data"),
    Input("store-chunk-idx", "data"),
)
def render_chunk_strip(chunks, active_idx):
    if not chunks:
        return [], "No chunks", "–", {}
    pills = [_chunk_pill(c, i, active_idx) for i, c in enumerate(chunks)]
    chunk = chunks[active_idx]
    start = chunk.get("start", "")[:16]
    end = chunk.get("end", "")[:16]
    label = f"Chunk {active_idx + 1} / {len(chunks)}  ·  {start} → {end}"
    status = chunk.get("status", "pending")
    badge_style = {"background": STATUS_COLORS.get(status, "#4a6070"),
                   "color": "#d1e5fa", "padding": "2px 10px",
                   "borderRadius": "999px", "fontSize": "11px", "fontWeight": 700}
    return pills, label, status, badge_style


# ══════════════════════════════════════════════════════════════════════════════
# PEAK DETECTION (chunk run — stages 5+)
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("store-peaks", "data", allow_duplicate=True),
    Input("store-btn", "data"),
    Input("store-chunk-idx", "data"),
    State("store-params", "data"),
    State("store-chunk-grid", "data"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    State("radio-mode", "value"),
    State("store-stage", "data"),
    prevent_initial_call=True,
)
def run_chunk_detect(btn_data, chunk_idx, params, chunks, dataset, deployment, mode, stage):
    if stage not in ("5",):
        raise PreventUpdate
    if not dataset or not deployment or not params:
        raise PreventUpdate
    # Allow btn-rerun to trigger detection, or store-chunk-idx change when rerun has been clicked
    trigger = ctx.triggered_id
    if isinstance(trigger, str):
        if trigger == "store-chunk-idx" and (not btn_data or btn_data.get("action") != "btn-rerun"):
            raise PreventUpdate
    elif isinstance(trigger, dict) and trigger.get("action") != "btn-rerun":
        raise PreventUpdate

    data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
    if data_pkl is None:
        raise PreventUpdate

    sig_key = "ecg" if mode == "heart_rate" else "corrected_gyr"
    sig_df = data_pkl.signal_data.get(sig_key)
    if sig_df is None:
        raise PreventUpdate
    channel = [c for c in sig_df.columns if c != "datetime"][0]
    dt_col = pd.to_datetime(sig_df["datetime"])

    if chunks and chunk_idx is not None and chunk_idx < len(chunks):
        chunk = chunks[chunk_idx]
        if chunk.get("status") == "gap":
            return {
                "peak_df": [],
                "smoothed": [],
                "gap": True,
                "chunk_idx": chunk_idx,
                "chunk_start": chunk.get("start"),
                "chunk_end": chunk.get("end"),
            }
        t0 = pd.Timestamp(chunk["start"])
        t1 = pd.Timestamp(chunk["end"])
        if dt_col.dt.tz is not None:
            t0 = t0.tz_localize(str(dt_col.dt.tz)) if t0.tzinfo is None else t0.tz_convert(str(dt_col.dt.tz))
            t1 = t1.tz_localize(str(dt_col.dt.tz)) if t1.tzinfo is None else t1.tz_convert(str(dt_col.dt.tz))
        override = chunk.get("params_override") or {}
        if override:
            params = {**params, **{k: v for k, v in override.items() if v is not None}}
    else:
        t0, t1 = dt_col.min(), dt_col.max()

    mask = (dt_col >= t0) & (dt_col <= t1)
    signal_sub = sig_df[channel][mask]
    dt_sub = dt_col[mask]
    if signal_sub.empty:
        return {
            "peak_df": [],
            "smoothed": [],
            "gap": False,
            "chunk_idx": chunk_idx,
            "chunk_start": str(t0),
            "chunk_end": str(t1),
        }

    fs = calculate_sampling_frequency(dt_sub.head())
    nyquist = fs / 2.0
    p = params
    bl = max(0.01, min(p.get("BROAD_LOW_CUTOFF", 1.0), nyquist * 0.99))
    bh = max(0.01, min(p.get("BROAD_HIGH_CUTOFF", min(25.0, nyquist * 0.9)), nyquist * 0.99))
    nl = max(0.01, min(p.get("NARROW_LOW_CUTOFF", 5.0), nyquist * 0.99))
    nh = max(0.01, min(p.get("NARROW_HIGH_CUTOFF", min(15.0, nyquist * 0.9)), nyquist * 0.99))
    if bl >= bh: bl = bh * 0.5
    if nl >= nh: nl = nh * 0.5

    try:
        results = peak_detect(
            signal=signal_sub, sampling_rate=fs, datetime_series=dt_sub,
            broad_lowcut=bl, broad_highcut=bh, narrow_lowcut=nl, narrow_highcut=nh,
            filter_order=int(p.get("FILTER_ORDER", 2)),
            spike_threshold=p.get("SPIKE_THRESHOLD", 400),
            smooth_sec_multiplier=p.get("SMOOTH_SEC_MULTIPLIER", 0.36),
            window_size_multiplier=p.get("WINDOW_SIZE_MULTIPLIER", 6.35),
            normalization_noise=p.get("NORMALIZATION_NOISE", 1e-10),
            peak_height=p.get("PEAK_HEIGHT", -0.4),
            peak_distance_sec=p.get("PEAK_DISTANCE_SEC", 0.16),
            search_radius_sec=p.get("SEARCH_RADIUS_SEC", 0.2),
            min_peak_height=p.get("MIN_PEAK_HEIGHT", 70),
            max_peak_height=p.get("MAX_PEAK_HEIGHT", 12000),
            enable_bandpass=p.get("enable_bandpass", True),
            enable_spike_removal=p.get("enable_spike_removal", True),
            enable_absolute=p.get("enable_absolute", True) if mode == "heart_rate" else False,
            enable_smoothing=p.get("enable_smoothing", True),
            enable_normalization=p.get("enable_normalization", True),
            enable_refinement=p.get("enable_refinement", True),
            detection_sources=p.get("DETECTION_DERIVATIVE_CHANNELS", ["normalized"]),
        )
    except Exception as exc:
        return {"peak_df": [], "smoothed": [], "gap": False, "error": str(exc)}

    peak_df = results.get("peak_df", pd.DataFrame())
    smoothed = results.get("smoothed", np.array([]))
    if not peak_df.empty and mode == "heart_rate":
        peak_df = _hr_cleanup(peak_df, smoothed, sig_df[mask], params, fs)

    return {
        "peak_df": peak_df.to_dict("records") if not peak_df.empty else [],
        "smoothed": smoothed.tolist() if hasattr(smoothed, "tolist") else [],
        "datetime": dt_sub.astype(str).tolist(),
        "signal": signal_sub.tolist(),
        "fs": fs,
        "mode": mode,
        "gap": False,
        "chunk_idx": chunk_idx,
        "chunk_start": str(t0),
        "chunk_end": str(t1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RENDER PLOTS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("graph-calib-overview", "figure"),
    Output("store-calib-window", "data"),
    Input("store-stage", "data"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    State("radio-mode", "value"),
    State("store-params", "data"),
    prevent_initial_call=True,
)
def render_calib_overview(stage, dataset, deployment, mode, params):
    if stage != "3" or not dataset or not deployment:
        raise PreventUpdate
    data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
    if data_pkl is None:
        return _empty_fig("No data"), None
    sig_key = "ecg" if mode == "heart_rate" else "corrected_gyr"
    sig_df = data_pkl.signal_data.get(sig_key)
    if sig_df is None:
        return _empty_fig(f"'{sig_key}' not found"), None
    channel = [c for c in sig_df.columns if c != "datetime"][0]
    dt = pd.to_datetime(sig_df["datetime"])
    calib_start = dt.min()
    calib_end = min(calib_start + pd.Timedelta(minutes=2), dt.max())
    calib_window = {"start": str(calib_start), "end": str(calib_end)}
    step = max(1, len(sig_df) // 4000)
    ov_fig = go.Figure().update_layout(**_dark_layout(
        margin=dict(l=50, r=20, t=10, b=30),
        xaxis=dict(showgrid=False, color="#4a6070"),
        yaxis=dict(showgrid=True, gridcolor="rgba(115,169,196,0.08)", color="#4a6070", fixedrange=True),
        showlegend=False, dragmode="pan",
    ))
    ov_fig.add_trace(go.Scattergl(
        x=dt[::step], y=sig_df[channel][::step],
        mode="lines", line=dict(color="rgba(149,204,254,0.4)", width=0.8), hoverinfo="skip",
    ))
    ov_fig.add_vrect(x0=str(calib_start), x1=str(calib_end),
                     fillcolor="rgba(149,204,254,0.12)", line_color="#95ccfe", line_width=1)
    return ov_fig, calib_window


@app.callback(
    Output("graph-peaks", "figure"),
    Output("graph-rate", "figure"),
    Input("store-peaks", "data"),
    Input("store-full-peaks", "data"),
    Input("store-stage", "data"),
    Input("store-calib-clicks", "data"),
    Input("store-zoom-window", "data"),
    Input("store-boundary-periods", "data"),
    State("radio-mode", "value"),
    State("store-manual-edits", "data"),
    State("dd-dataset", "value"),
    State("dd-deployment", "value"),
    State("store-params", "data"),
    State("store-calib-window", "data"),
    State("store-chunk-grid", "data"),
    State("store-chunk-idx", "data"),
    prevent_initial_call=True,
)
def render_plots(chunk_peaks, full_peaks, stage, calib_clicks, zoom_window, boundary_periods,
                 mode, manual_edits, dataset, deployment, params, calib_window, chunks, chunk_idx):
    zoom_start = zoom_window.get("start") if isinstance(zoom_window, dict) else None
    zoom_end = zoom_window.get("end") if isinstance(zoom_window, dict) else None

    # Stage 2: context overview with attachment shading
    if stage == "2":
        if not dataset or not deployment:
            return _empty_fig(), _empty_fig()
        periods = _normalize_periods(boundary_periods) or _boundary_periods_from_config(dataset, deployment)
        data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
        return _make_boundary_detail_fig(data_pkl, mode, periods), _make_boundary_overview_fig(data_pkl, mode, periods)

    # Stage 3: calibration zoom
    if stage == "3":
        if not dataset or not deployment:
            return _empty_fig("No data"), _empty_fig()
        data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
        if data_pkl is None:
            return _empty_fig("No data"), _empty_fig()
        sig_key = "ecg" if mode == "heart_rate" else "corrected_gyr"
        sig_df = data_pkl.signal_data.get(sig_key)
        if sig_df is None:
            return _empty_fig(f"'{sig_key}' not found"), _empty_fig()
        channel = [c for c in sig_df.columns if c != "datetime"][0]
        if calib_window:
            t0 = pd.Timestamp(calib_window["start"])
            t1 = pd.Timestamp(calib_window["end"])
        else:
            dt_s = pd.to_datetime(sig_df["datetime"])
            t0 = dt_s.min()
            t1 = min(t0 + pd.Timedelta(minutes=2), dt_s.max())
        return _make_calib_zoom(sig_df, channel, t0, t1, params, mode, calib_clicks or []), _empty_fig()

    # Stage 6 shows full-deployment peaks
    peak_data = full_peaks if stage == "6" and full_peaks else chunk_peaks

    def _fallback_current_signal_window():
        if dataset and deployment:
            data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
            t0 = t1 = None
            if zoom_start and zoom_end:
                t0, t1 = zoom_start, zoom_end
            elif stage == "5" and chunks and chunk_idx is not None and chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                t0 = chunk.get("start")
                t1 = chunk.get("end")
            zoom_fig, _ = _make_initial_signal_figs(data_pkl, mode=mode, t0=t0, t1=t1)
            return zoom_fig, _make_deployment_overview_fig(
                data_pkl,
                mode,
                active_start=t0,
                active_end=t1,
            )
        return _empty_fig("Select a deployment to view ECG and depth"), _empty_fig()

    if zoom_start and zoom_end and dataset and deployment:
        data_pkl = load_datapkl(dataset, deployment, DATA_DIR)
        zoom_fig, _ = _make_initial_signal_figs(
            data_pkl,
            mode=mode,
            t0=zoom_start,
            t1=zoom_end,
        )
        return zoom_fig, _make_deployment_overview_fig(
            data_pkl,
            mode,
            active_start=zoom_start,
            active_end=zoom_end,
        )

    if stage == "5" and isinstance(peak_data, dict) and chunks and chunk_idx is not None and chunk_idx < len(chunks):
        chunk = chunks[chunk_idx]
        if (
            peak_data.get("chunk_idx") is not None
            and int(peak_data["chunk_idx"]) != int(chunk_idx)
        ):
            return _fallback_current_signal_window()
        if (
            peak_data.get("chunk_start") is not None
            and str(peak_data["chunk_start"]) != str(chunk.get("start"))
        ):
            return _fallback_current_signal_window()

    if (
        not peak_data
        or not isinstance(peak_data, dict)
        or "datetime" not in peak_data
        or not peak_data.get("datetime")
        or not peak_data.get("signal")
    ):
        return _fallback_current_signal_window()

    if peak_data.get("gap"):
        data_pkl = load_datapkl(dataset, deployment, DATA_DIR) if dataset and deployment else None
        return _empty_fig("Gap — no ECG data in this chunk", color="#f07070"), _make_deployment_overview_fig(
            data_pkl,
            mode,
        )

    if peak_data.get("error"):
        data_pkl = load_datapkl(dataset, deployment, DATA_DIR) if dataset and deployment else None
        return _empty_fig(f"Error: {peak_data['error']}", color="#f07070"), _make_deployment_overview_fig(
            data_pkl,
            mode,
        )

    dt = _parse_datetime(peak_data["datetime"])
    signal = np.array(peak_data["signal"])
    smoothed = np.array(peak_data.get("smoothed", []))
    peak_records = peak_data.get("peak_df", [])
    fs = peak_data.get("fs", 100)

    peak_df = pd.DataFrame(peak_records) if peak_records else pd.DataFrame()

    # Overlay manual edits
    if manual_edits and not peak_df.empty:
        for edit in manual_edits:
            t = _parse_datetime(edit["datetime"])
            key = "heartbeat_manual_ok" if edit["action"] == "ok" else "heartbeat_manual_reject"
            # Find closest peak and relabel it in display
            if "datetime" in peak_df.columns:
                diffs = (_parse_datetime(peak_df["datetime"]) - t).abs()
                closest = diffs.idxmin()
                if diffs[closest] < pd.Timedelta(seconds=0.5):
                    peak_df.loc[closest, "key"] = key

    # ── ECG + context figure ────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(go.Scattergl(
        x=dt, y=signal,
        mode="lines", line=dict(color=_color_for_signal("ecg" if mode == "heart_rate" else "corrected_gyr", fallback="#d09191"), width=1.4),
        name="raw", hoverinfo="skip",
    ), row=1, col=1)
    if len(smoothed) == len(signal):
        fig.add_trace(go.Scattergl(
            x=dt, y=smoothed,
            mode="lines", line=dict(color=_color_for_signal("hr_smoothed" if mode == "heart_rate" else "sr_smoothed", fallback="#9f76ba"), width=2.4),
            name="processed", hoverinfo="skip",
        ), row=1, col=1)

    if not peak_df.empty and "datetime" in peak_df.columns:
        for key, style in PEAK_STYLES.items():
            sub = peak_df[peak_df["key"] == key]
            if sub.empty:
                continue
            sub_dt = _parse_datetime(sub["datetime"])
            heights = sub["height_original"].fillna(0).tolist() if "height_original" in sub.columns else [0] * len(sub)
            fig.add_trace(go.Scattergl(
                x=sub_dt, y=heights,
                mode="markers",
                marker=dict(color=style["color"], symbol=style["symbol"],
                            size=style["size"], line=dict(width=0.5, color="#021524")),
                name=style["name"],
                customdata=sub["refined_index"].tolist() if "refined_index" in sub.columns else [],
                hovertemplate=f"{style['name']}<br>%{{x}}<extra></extra>",
                # Allow lasso selection on stage 6
                selected=dict(marker=dict(opacity=1.0)),
                unselected=dict(marker=dict(opacity=0.3)),
            ), row=1, col=1)

    data_pkl = load_datapkl(dataset, deployment, DATA_DIR) if dataset and deployment else None
    signal_data = getattr(data_pkl, "signal_data", {}) if data_pkl is not None else {}
    rate_plotted = _add_detected_rate_trace(fig, peak_df, fs, mode)
    if not rate_plotted and len(dt):
        _add_rate_signal_window(fig, signal_data or {}, mode, t0=dt.min(), t1=dt.max())

    # Enable lasso on stage 6
    dragmode = "lasso" if stage == "6" else "pan"
    fig.update_layout(
        **_dark_layout(),
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10, color="#8eb0cb"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#4a6070"),
        xaxis2=dict(showgrid=False, color="#4a6070"),
        yaxis=dict(showgrid=True, gridcolor="rgba(115,169,196,0.08)", color="#4a6070", fixedrange=True, title="ECG"),
        yaxis2=dict(
            showgrid=True,
            gridcolor="rgba(115,169,196,0.08)",
            color="#4a6070",
            fixedrange=True,
            title="Heart Rate" if mode == "heart_rate" else "Stroke Rate",
        ),
        margin=dict(l=50, r=20, t=10, b=30),
        dragmode=dragmode,
    )

    active_start = peak_data.get("chunk_start") or (dt.min() if len(dt) else None)
    active_end = peak_data.get("chunk_end") or (dt.max() if len(dt) else None)
    return fig, _make_deployment_overview_fig(
        data_pkl,
        mode,
        active_start=active_start,
        active_end=active_end,
    )


app.clientside_callback(
    """
    function(peaksHover, overviewHover, peaksFig, overviewFig) {
        const noUpdate = window.dash_clientside.no_update;
        if (!peaksFig || !overviewFig) {
            return [noUpdate, noUpdate];
        }

        const ctx = window.dash_clientside.callback_context;
        const propId = ctx.triggered && ctx.triggered.length ? ctx.triggered[0].prop_id : "";
        const source = propId.indexOf("graph-rate.hoverData") === 0 ? "overview"
                     : propId.indexOf("graph-peaks.hoverData") === 0 ? "peaks"
                     : null;
        const hover = source === "overview" ? overviewHover : source === "peaks" ? peaksHover : null;
        const x = firstHoverTime(hover);
        if (!x) {
            return [noUpdate, noUpdate];
        }
        const now = Date.now();
        const hoverKey = source + ":" + x;
        window.__peakDetectHoverState = window.__peakDetectHoverState || {lastTs: 0, lastKey: ""};
        if (hoverKey === window.__peakDetectHoverState.lastKey) {
            return [noUpdate, noUpdate];
        }
        if (now - window.__peakDetectHoverState.lastTs < 45) {
            return [noUpdate, noUpdate];
        }
        window.__peakDetectHoverState.lastTs = now;
        window.__peakDetectHoverState.lastKey = hoverKey;

        function cloneFig(fig) {
            return JSON.parse(JSON.stringify(fig));
        }

        function cleanCursor(fig) {
            fig.data = (fig.data || []).filter(t => {
                return t.name !== "__cursor"
                    && t.name !== "__cursor-dot"
                    && t.name !== "__active-window-fill";
            });
            fig.layout = fig.layout || {};
            fig.layout.shapes = (fig.layout.shapes || []).filter(s => {
                return s.name !== "__cursor-line"
                    && s.name !== "__cursor-dot"
                    && s.name !== "__active-window-edge";
            });
            fig.layout.annotations = (fig.layout.annotations || []).filter(a => a.name !== "__cursor-dot");
            return fig;
        }

        function firstHoverTime(hoverData) {
            if (!hoverData || !hoverData.points || !hoverData.points.length) return null;
            return hoverData.points[0].x || null;
        }

        function parseTime(value) {
            if (typeof value === "number") return value;
            if (value instanceof Date) return value.getTime();
            const ms = Date.parse(value);
            return Number.isFinite(ms) ? ms : null;
        }

        function isOverlayTrace(trace) {
            const name = String((trace && trace.name) || "");
            return name.indexOf("__") === 0 || trace.fill === "toself";
        }

        function isCursorTargetTrace(trace) {
            if (!trace || isOverlayTrace(trace) || !trace.x || !trace.y) return false;
            return String(trace.mode || "").indexOf("lines") !== -1;
        }

        function hasBoundaryHandles(fig) {
            return (((fig.layout || {}).shapes) || []).some(shape => {
                return String((shape && shape.name) || "").indexOf("__boundary-edge:") === 0;
            });
        }

        function dataExtentValues(fig) {
            let lo = null;
            let hi = null;
            let loValue = null;
            let hiValue = null;
            (fig.data || []).forEach(trace => {
                if (isOverlayTrace(trace) || !trace.x) return;
                const candidates = [trace.x[0], trace.x[trace.x.length - 1]];
                candidates.forEach(x => {
                    const ms = parseTime(x);
                    if (ms === null) return;
                    if (lo === null || ms < lo) {
                        lo = ms;
                        loValue = x;
                    }
                    if (hi === null || ms > hi) {
                        hi = ms;
                        hiValue = x;
                    }
                });
            });
            return [loValue, hiValue];
        }

        function dataExtent(fig) {
            const [loValue, hiValue] = dataExtentValues(fig);
            return [parseTime(loValue), parseTime(hiValue)];
        }

        function visibleExtentValues(fig) {
            const layout = fig.layout || {};
            const r = (layout.xaxis && layout.xaxis.range) || (layout.xaxis2 && layout.xaxis2.range);
            if (r && r.length === 2 && parseTime(r[0]) !== null && parseTime(r[1]) !== null) {
                return parseTime(r[0]) <= parseTime(r[1]) ? [r[0], r[1]] : [r[1], r[0]];
            }
            const [loValue, hiValue] = dataExtentValues(fig);
            if (loValue === null || hiValue === null) return [null, null];
            return parseTime(loValue) <= parseTime(hiValue) ? [loValue, hiValue] : [hiValue, loValue];
        }

        function visibleExtent(fig) {
            const [loValue, hiValue] = visibleExtentValues(fig);
            return [parseTime(loValue), parseTime(hiValue)];
        }

        function addActiveWindow(fig, start, end) {
            const a = parseTime(start);
            const b = parseTime(end);
            if (a === null || b === null || a === b) return;
            const x0 = a < b ? start : end;
            const x1 = a < b ? end : start;
            fig.layout = fig.layout || {};
            fig.layout.shapes = fig.layout.shapes || [];
            ["y", "y2"].forEach(yAxis => {
                const yr = axisRange(fig, yAxis);
                if (!yr) return;
                fig.data.push({
                    type: "scatter",
                    x: [x0, x1, x1, x0, x0],
                    y: [yr[0], yr[0], yr[1], yr[1], yr[0]],
                    xaxis: yAxis === "y" ? "x" : "x2",
                    yaxis: yAxis,
                    mode: "lines",
                    fill: "toself",
                    fillcolor: "rgba(246,231,168,0.18)",
                    line: {color: "rgba(246,231,168,0)", width: 0},
                    name: "__active-window-fill",
                    showlegend: false,
                    hoverinfo: "skip"
                });
            });
            [x0, x1].forEach(edge => {
                fig.layout.shapes.push({
                    name: "__active-window-edge",
                    type: "line",
                    xref: "x",
                    yref: "paper",
                    x0: edge,
                    x1: edge,
                    y0: 0,
                    y1: 1,
                    line: {color: "rgba(246,231,168,0.32)", width: 8},
                    layer: "above",
                    editable: true
                });
            });
        }

        function inVisibleExtent(fig, x) {
            const ms = parseTime(x);
            if (ms === null) return false;
            const [lo, hi] = visibleExtent(fig);
            if (lo === null || hi === null) return true;
            return ms >= lo && ms <= hi;
        }

        function addCursorLine(fig, x) {
            fig.layout = fig.layout || {};
            fig.layout.shapes = fig.layout.shapes || [];
            fig.layout.shapes.push({
                name: "__cursor-line",
                type: "line",
                xref: "x",
                yref: "paper",
                x0: x,
                x1: x,
                y0: 0,
                y1: 1,
                line: {color: "rgba(255,255,255,0.95)", width: 1, dash: "dot"},
                layer: "above"
            });
        }

        function nearestPoint(trace, x) {
            if (!isCursorTargetTrace(trace)) return null;
            const target = parseTime(x);
            if (target === null) return null;
            const n = Math.min(trace.x.length, trace.y.length);
            if (!n) return null;
            let lo = 0;
            let hi = n - 1;
            while (lo < hi) {
                const mid = Math.floor((lo + hi) / 2);
                const ms = parseTime(trace.x[mid]);
                if (ms === null || ms < target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let best = null;
            let bestDelta = Infinity;
            [lo - 2, lo - 1, lo, lo + 1, lo + 2].forEach(i => {
                if (i < 0 || i >= n) return;
                const ms = parseTime(trace.x[i]);
                const y = Number(trace.y[i]);
                if (ms === null || !Number.isFinite(y)) return;
                const delta = Math.abs(ms - target);
                if (delta < bestDelta) {
                    bestDelta = delta;
                    best = {x: trace.x[i], y: y};
                }
            });
            return best;
        }

        function axisLayoutKey(axisRef) {
            if (!axisRef || axisRef === "y") return "yaxis";
            return "yaxis" + axisRef.slice(1);
        }

        function axisRange(fig, axisRef) {
            let lo = null;
            let hi = null;
            (fig.data || []).forEach(trace => {
                if (isOverlayTrace(trace) || (trace.yaxis || "y") !== (axisRef || "y") || !trace.y) return;
                const step = Math.max(1, Math.floor(trace.y.length / 250));
                for (let i = 0; i < trace.y.length; i += step) {
                    const y = trace.y[i];
                    const v = Number(y);
                    if (!Number.isFinite(v)) continue;
                    lo = lo === null ? v : Math.min(lo, v);
                    hi = hi === null ? v : Math.max(hi, v);
                }
            });
            if (lo === null || hi === null) {
                const key = axisLayoutKey(axisRef);
                const axis = (fig.layout || {})[key] || {};
                if (axis.range && axis.range.length === 2) {
                    const a = Number(axis.range[0]);
                    const b = Number(axis.range[1]);
                    if (Number.isFinite(a) && Number.isFinite(b)) return [Math.min(a, b), Math.max(a, b)];
                }
                return null;
            }
            const pad = lo === hi ? (Math.abs(lo) * 0.05 || 1) : (hi - lo) * 0.05;
            return [lo - pad, hi + pad];
        }

        function addCursorDots(fig, x) {
            fig.layout = fig.layout || {};
            const seenAxes = {};
            const targetTraces = (fig.data || []).filter(isCursorTargetTrace);
            targetTraces.forEach(trace => {
                const yAxis = trace.yaxis || "y";
                if (seenAxes[yAxis]) return;
                const point = nearestPoint(trace, x);
                if (!point) return;
                seenAxes[yAxis] = true;
                fig.data.push({
                    name: "__cursor-dot",
                    type: "scatter",
                    x: [point.x],
                    y: [point.y],
                    xaxis: trace.xaxis || "x",
                    yaxis: yAxis,
                    mode: "markers",
                    marker: {
                        color: "#ffffff",
                        size: 11,
                        symbol: "circle",
                        line: {color: "#021524", width: 2}
                    },
                    showlegend: false,
                    hoverinfo: "skip",
                    cliponaxis: false
                });
            });
        }

        const peaks = cleanCursor(cloneFig(peaksFig));
        const overview = cleanCursor(cloneFig(overviewFig));
        const [detailStart, detailEnd] = visibleExtentValues(peaks);
        if (!hasBoundaryHandles(overview) && detailStart !== null && detailEnd !== null) {
            addActiveWindow(overview, detailStart, detailEnd);
        }
        if (source === "overview") {
            addCursorLine(overview, x);
            addCursorDots(overview, x);
            if (inVisibleExtent(peaks, x)) {
                addCursorLine(peaks, x);
                addCursorDots(peaks, x);
            }
        } else if (source === "peaks") {
            addCursorLine(peaks, x);
            addCursorDots(peaks, x);
            addCursorLine(overview, x);
            addCursorDots(overview, x);
        }
        return [peaks, overview];
    }
    """,
    Output("graph-peaks", "figure", allow_duplicate=True),
    Output("graph-rate", "figure", allow_duplicate=True),
    Input("graph-peaks", "hoverData"),
    Input("graph-rate", "hoverData"),
    State("graph-peaks", "figure"),
    State("graph-rate", "figure"),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(relayoutData, overviewFig) {
        const noUpdate = window.dash_clientside.no_update;
        if (!relayoutData || !overviewFig) {
            return noUpdate;
        }
        const hasShapeUpdate = Object.keys(relayoutData).some(k => k.indexOf("shapes[") === 0);
        if (!hasShapeUpdate) {
            return noUpdate;
        }

        const fig = JSON.parse(JSON.stringify(overviewFig));
        const shapes = (((fig.layout || {}).shapes) || []);
        let changed = false;
        shapes.forEach(shape => {
            const name = String((shape && shape.name) || "");
            if (name !== "__active-window" && name !== "__active-window-edge" && name.indexOf("__boundary-edge:") !== 0) {
                return;
            }
            if (shape.y0 !== 0 || shape.y1 !== 1 || shape.yref !== "paper") {
                shape.yref = "paper";
                shape.y0 = 0;
                shape.y1 = 1;
                changed = true;
            }
        });
        return changed ? fig : noUpdate;
    }
    """,
    Output("graph-rate", "figure", allow_duplicate=True),
    Input("graph-rate", "relayoutData"),
    State("graph-rate", "figure"),
    prevent_initial_call=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _fill_defaults(params: dict, mode: str):
    defs = HR_PARAM_DEFS if mode == "heart_rate" else SR_PARAM_DEFS
    for key, (default, *_) in defs.items():
        params.setdefault(key, default)
    for key, _ in BOOL_PARAMS:
        params.setdefault(key, True)
    params.setdefault("DETECTION_DERIVATIVE_CHANNELS", ["normalized"])
    params.setdefault("NORMALIZATION_NOISE", 1e-10)
    params.setdefault("PICK_LAST_IN_CONFLICT_PAIR", True)
    params.setdefault("HR_CONFLICT_RR_FACTOR", params.get("ANTI_DOUBLE_GAP_FACTOR", 0.75))


def _eventdata_to_store(data_pkl, mode: str) -> dict:
    """Convert existing event_data heartbeat events into a store-peaks dict."""
    if not hasattr(data_pkl, "event_data") or data_pkl.event_data is None:
        return {}
    ev = data_pkl.event_data
    accepted_keys = (["heartbeat_auto_detect_accepted", "heartbeat_auto_detect_suggested",
                      "heartbeat_auto_detect_rejected", "heartbeat_manual_ok", "heartbeat_manual_reject"]
                     if mode == "heart_rate"
                     else ["strokebeat_auto_detect_accepted"])
    pk = ev[ev["key"].isin(accepted_keys)].copy()
    if pk.empty:
        return {}
    # Map event keys back to internal peak_df keys
    key_map = {
        "heartbeat_auto_detect_accepted":  "beat_auto_detect_accepted",
        "heartbeat_auto_detect_rejected":  "beat_auto_detect_rejected",
        "heartbeat_auto_detect_suggested": "beat_auto_detect_suggested",
        "heartbeat_manual_ok":             "heartbeat_manual_ok",
        "heartbeat_manual_reject":         "heartbeat_manual_reject",
        "strokebeat_auto_detect_accepted": "beat_auto_detect_accepted",
    }
    pk = pk.copy()
    pk["key"] = pk["key"].map(lambda k: key_map.get(k, k))
    pk = pk.rename(columns={"datetime": "datetime"})
    if "datetime" in pk.columns:
        pk["datetime"] = pk["datetime"].astype(str)
    # We don't have signal/smoothed here — just peaks
    return {
        "peak_df": pk[["key", "datetime", "value"]].to_dict("records"),
        "smoothed": [], "datetime": [], "signal": [], "fs": 100,
        "mode": mode, "gap": False,
    }


def _hr_cleanup(peak_df, smoothed, sig_subset_df, params, fs):
    """Minimal HR cleanup: conflict pair rejection."""
    CONFLICT_GAP_FACTOR = float(params.get("ANTI_DOUBLE_GAP_FACTOR", 0.75))
    CONFLICT_WINDOW_SEC = float(params.get("ANTI_DOUBLE_ROLLING_WINDOW_SEC", 10.0))
    PICK_LAST = bool(params.get("PICK_LAST_IN_CONFLICT_PAIR", True))
    CONFLICT_LOCAL_NEIGHBORS = 30

    accepted = peak_df[peak_df["key"].isin([
        "beat_auto_detect_accepted", "beat_auto_detect_suggested"
    ])].sort_values("refined_index").reset_index(drop=True)

    if len(accepted) <= 1:
        return peak_df

    idxs = accepted["refined_index"].astype(int).to_numpy()
    rr_all = np.diff(idxs) / fs
    interval_midpoints = (idxs[:-1].astype(float) + idxs[1:].astype(float)) / 2.0
    half_win = max(1.0, (CONFLICT_WINDOW_SEC * fs) / 2.0)

    for i in range(len(idxs) - 1):
        a, b = int(idxs[i]), int(idxs[i + 1])
        rr = (b - a) / fs
        local_mask = np.abs(interval_midpoints - interval_midpoints[i]) <= half_win
        local_rr = rr_all[local_mask]
        local_rr = local_rr[np.isfinite(local_rr) & (local_rr > 0)]
        if local_rr.size == 0:
            lo = max(0, i - CONFLICT_LOCAL_NEIGHBORS)
            hi = min(len(rr_all), i + CONFLICT_LOCAL_NEIGHBORS + 1)
            local_rr = rr_all[lo:hi]
            local_rr = local_rr[np.isfinite(local_rr) & (local_rr > 0)]
        if local_rr.size == 0:
            continue
        rr_ref = float(np.median(local_rr))
        if rr < CONFLICT_GAP_FACTOR * rr_ref:
            reject_idx = a if PICK_LAST else b
            peak_df.loc[peak_df["refined_index"] == reject_idx, "key"] = "beat_auto_detect_rejected"

    return peak_df


# ══════════════════════════════════════════════════════════════════════════════
# CSS + INDEX
# ══════════════════════════════════════════════════════════════════════════════

app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
body { margin:0; background:#021524; color:#d1e5fa; font-family:'Figtree',sans-serif; overflow:hidden; }
.app-root { display:flex; flex-direction:column; height:100vh; }

/* top bar */
.topbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:6px 16px; background:#081a29;
  border-bottom:1px solid rgba(115,169,196,0.18); flex-shrink:0;
}
.topbar-title { font-family:'Manrope',sans-serif; font-size:14px; font-weight:800; color:#d1e5fa; }
.topbar-right { display:flex; align-items:center; gap:10px; }
.topbar-dropdown { min-width:320px; max-width:440px; }
.mode-radio label { color:#8eb0cb; font-size:12px; margin-right:10px; }

/* stage bar */
.stage-bar {
  display:flex; align-items:center; padding:6px 16px; background:#0a1e2e;
  border-bottom:1px solid rgba(115,169,196,0.12); flex-shrink:0; gap:0;
}
.stage-step {
  display:flex; align-items:center; gap:6px; padding:4px 14px;
  opacity:0.4; transition:opacity 0.2s; cursor:default; white-space:nowrap;
}
.stage-step--active { opacity:1; }
.stage-step + .stage-step { border-left:1px solid rgba(115,169,196,0.12); }
.stage-num {
  width:20px; height:20px; border-radius:50%;
  background:rgba(149,204,254,0.12); border:1px solid rgba(149,204,254,0.25);
  display:flex; align-items:center; justify-content:center;
  font-size:10px; font-weight:800; color:#95ccfe; font-family:'Manrope',sans-serif;
}
.stage-step--active .stage-num { background:#95ccfe; color:#021524; border-color:#95ccfe; }
.stage-label { font-size:11px; font-weight:600; color:#8eb0cb; }
.stage-step--active .stage-label { color:#d1e5fa; }

/* main body */
.main-body { display:flex; flex:1; overflow:hidden; }

/* left panel */
.left-panel {
  width:270px; flex-shrink:0; background:#081a29;
  border-right:1px solid rgba(115,169,196,0.18);
  overflow-y:auto; padding:12px; display:flex; flex-direction:column; gap:6px;
}
.panel-heading { font-family:'Manrope',sans-serif; font-size:13px; font-weight:800; color:#d1e5fa; padding:2px 0 8px; }
.full-width { width:100%; margin-top:4px; }

/* status blocks */
.status-found { font-size:12px; font-weight:700; color:#55d38a; padding:6px 0; }
.status-missing { font-size:12px; font-weight:700; color:#f2b26b; padding:6px 0; }
.stat-row { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px; }
.stat-chip { font-size:10px; padding:2px 8px; border-radius:999px; font-weight:700; }
.stat-ok { background:rgba(85,211,138,0.15); color:#55d38a; border:1px solid rgba(85,211,138,0.3); }
.stat-manual { background:rgba(149,204,254,0.12); color:#95ccfe; border:1px solid rgba(149,204,254,0.2); }

.preset-dropdown { margin-bottom:4px; }
.preset-desc { font-size:10px; color:#8eb0cb; padding:2px 0 4px; min-height:16px; }
.divider { border:none; border-top:1px solid rgba(115,169,196,0.12); margin:8px 0; }

/* calib */
.calib-stat { font-size:12px; color:#95ccfe; font-weight:600; padding:4px 0; }
.calib-item { font-size:11px; color:#f2b26b; padding:2px 0 2px 10px; font-family:ui-monospace,monospace; }
.calib-suggestion { padding:4px 0; min-height:20px; }

/* attachment entry */
.attachment-entry { margin-bottom:8px; }

/* param sections */
.param-section { border:1px solid rgba(115,169,196,0.12); border-radius:10px; overflow:hidden; margin-bottom:4px; }
.section-summary { font-size:10px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#8eb0cb; padding:6px 10px; cursor:pointer; background:#0e2131; list-style:none; }
.section-summary:hover { color:#95ccfe; }
.section-body { padding:8px 10px; display:flex; flex-direction:column; gap:3px; background:#081a29; }
.param-row { display:flex; flex-direction:column; gap:2px; margin-bottom:2px; }
.param-row-header { display:flex; justify-content:space-between; align-items:center; }
.param-label { font-size:10px; color:#8eb0cb; }
.param-value { font-size:11px; font-weight:700; color:#95ccfe; font-family:ui-monospace,monospace; }
.param-slider { height:18px; }
.param-check label { font-size:11px; color:#8eb0cb; }
.param-action-row { display:flex; gap:6px; margin-bottom:4px; flex-wrap:wrap; }
.save-status { font-size:10px; color:#55d38a; min-height:14px; }

/* center */
.center-panel { flex:1; display:flex; flex-direction:column; overflow:hidden; }
.hint-bar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding:5px 14px; background:#0e2131; border-bottom:1px solid rgba(115,169,196,0.12); flex-shrink:0; }
.hint { font-size:11px; color:#8eb0cb; }

/* chunk nav */
.chunk-nav-row { display:flex; align-items:center; gap:6px; padding:5px 14px; background:#081a29; border-bottom:1px solid rgba(115,169,196,0.12); flex-shrink:0; overflow-x:auto; }
.chunk-strip { display:flex; gap:3px; flex:1; overflow-x:auto; }
.chunk-pill { min-width:20px; height:20px; border-radius:4px; font-size:9px; font-weight:700; color:#021524; display:flex; align-items:center; justify-content:center; cursor:pointer; flex-shrink:0; transition:transform 0.1s; }
.chunk-pill:hover { transform:scaleY(1.15); }
.chunk-pill--active { outline:2px solid #95ccfe; outline-offset:1px; }
.chunk-label { font-size:11px; color:#8eb0cb; white-space:nowrap; font-family:ui-monospace,monospace; }
.chunk-badge { font-size:11px; }
.gap-reason-row { padding:4px 14px; background:#081a29; border-bottom:1px solid rgba(115,169,196,0.12); }
.gap-reason-input { background:#0e2131; border:1px solid rgba(115,169,196,0.2); border-radius:8px; color:#d1e5fa; font-size:12px; padding:4px 10px; width:100%; outline:none; }
.gap-reason-input:focus { border-color:#95ccfe; }

/* plots */
.peak-graph { flex:1; min-height:0; }
.rate-graph { flex-shrink:0; border-top:1px solid rgba(115,169,196,0.12); }

/* buttons */
.btn { border:none; border-radius:8px; padding:4px 12px; font-size:11px; font-weight:600; cursor:pointer; font-family:'Figtree',sans-serif; transition:opacity 0.15s; }
.btn:hover { opacity:0.85; }
.btn-primary   { background:#3d86bd; color:#d1e5fa; }
.btn-secondary { background:#182c3d; color:#8eb0cb; border:1px solid rgba(115,169,196,0.2); }
.btn-ok  { background:rgba(85,211,138,0.18); color:#55d38a; border:1px solid rgba(85,211,138,0.3); }
.btn-gap { background:rgba(240,112,112,0.18); color:#f07070; border:1px solid rgba(240,112,112,0.3); }
.btn-nav { background:#182c3d; color:#8eb0cb; border:1px solid rgba(115,169,196,0.2); padding:3px 9px; }

/* scrollbars */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#081a29; }
::-webkit-scrollbar-thumb { background:#243747; border-radius:3px; }

/* Dash dropdown overrides */
.Select-control { background:#0e2131 !important; border-color:rgba(115,169,196,0.2) !important; }
.Select-menu-outer { background:#0e2131 !important; border-color:rgba(115,169,196,0.2) !important; z-index:9999 !important; }
.Select-option { color:#d1e5fa !important; font-size:13px !important; }
.Select-option.is-focused { background:#182c3d !important; }
.Select-value-label { color:#d1e5fa !important; font-size:13px !important; }
.Select-placeholder { color:#8eb0cb !important; }
.VirtualizedSelectOption { font-size:13px !important; color:#d1e5fa !important; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--debug", action="store_true", default=True)
    args = parser.parse_args()
    app.run(debug=args.debug, port=args.port)
