import argparse
import concurrent.futures
import colorsys
import copy
import io
import inspect
import pathlib
import sys
import json
import pickle
import os
import re
import shutil
import subprocess
import time
import html as std_html
import tempfile
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure as BasePlotlyFigure
import dash
from dash import Dash, Input, Output, State, ALL, dcc, html, ctx, no_update
try:
    import dash_cytoscape as cyto
    CYTO_AVAILABLE = True
except Exception:
    cyto = None
    CYTO_AVAILABLE = False

# Ensure repo-local pyologger package is imported before site-packages.
_LOCAL_PYOLOGGER_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_LOCAL_PYOLOGGER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PYOLOGGER_ROOT))

# Prefer local DiveDB dash component package when present so local fixes are used
# without reinstalling site-packages.
_LOCAL_REPO_ROOT = _LOCAL_PYOLOGGER_ROOT.parent
_LOCAL_THREEJS_PKG = _LOCAL_REPO_ROOT / "DiveDB" / "dash" / "three_js_orientation"
if _LOCAL_THREEJS_PKG.exists() and str(_LOCAL_THREEJS_PKG) not in sys.path:
    sys.path.insert(0, str(_LOCAL_THREEJS_PKG))

try:
    import three_js_orientation
    THREEJS_AVAILABLE = True
except Exception:
    three_js_orientation = None
    THREEJS_AVAILABLE = False

WATER_VIEWER_AVAILABLE = bool(
    THREEJS_AVAILABLE and getattr(three_js_orientation, "WebGLWaterTank", None) is not None
)

from pyologger.plot_data.plotter import plot_tag_data_interactive
from pyologger.process_data.peak_detect import peak_detect
from pyologger.process_data.sampling import calculate_sampling_frequency, downsample
from pyologger.utils.cluster_colors import (
    build_ordered_cluster_color_map,
    looks_like_ordered_cluster_key,
)
from pyologger.utils.deployment_source import (
    DeploymentDataSource,
    resolve_deployment_source,
    resolve_plot_signal_allowlist,
)
try:
    from pyologger.dash.integrated.model_3d import (
        EMPTY_ORIENTATION_JSON,
        build_orientation_data_json,
        fetch_3d_model_info,
        infer_animal_id,
    )
except ModuleNotFoundError:
    # Support direct script execution from pyologger repo root where
    # dash/integrated is not a package under pyologger/.
    from model_3d import (
        EMPTY_ORIENTATION_JSON,
        build_orientation_data_json,
        fetch_3d_model_info,
        infer_animal_id,
    )
from pyologger.utils.folder_manager import load_configuration, resolve_deployment_context, select_and_load_deployment
from pyologger.utils.streamlit_time_window import standardize_time_settings
try:
    from pyologger.dash.integrated.segmentation_helpers import (
        DEFAULT_SEGMENTATION_DATASET,
        DEFAULT_SEGMENTATION_DEPLOYMENT,
        DEFAULT_SEGMENTATION_WINDOW_HOURS,
        SEGMENTATION_WORKFLOW_ALLOWED_NODE_TYPES,
        build_find_rest_workflow_preset,
        cleanup_temp_workspace,
        default_algorithmic_cfg_from_config,
        load_algorithmic_results_from_real_pkl,
        load_segmentation_workflow_presets,
        save_segmentation_workflow_presets,
        segmentation_workflow_options,
        segmentation_workflow_to_cytoscape_elements,
        sync_workflow_nodes_from_runtime_params,
        validate_segmentation_workflow_preset,
        workflow_runtime_support_report,
        workflow_preset_to_algorithmic_cfg,
        duplicate_segmentation_workflow,
        event_group_parent,
        group_event_keys,
        run_algorithmic_segmentation_sandbox,
        save_algorithmic_results_to_real_pkl,
    )
except ModuleNotFoundError:
    from segmentation_helpers import (
        DEFAULT_SEGMENTATION_DATASET,
        DEFAULT_SEGMENTATION_DEPLOYMENT,
        DEFAULT_SEGMENTATION_WINDOW_HOURS,
        SEGMENTATION_WORKFLOW_ALLOWED_NODE_TYPES,
        build_find_rest_workflow_preset,
        cleanup_temp_workspace,
        default_algorithmic_cfg_from_config,
        load_algorithmic_results_from_real_pkl,
        load_segmentation_workflow_presets,
        save_segmentation_workflow_presets,
        segmentation_workflow_options,
        segmentation_workflow_to_cytoscape_elements,
        sync_workflow_nodes_from_runtime_params,
        validate_segmentation_workflow_preset,
        workflow_runtime_support_report,
        workflow_preset_to_algorithmic_cfg,
        duplicate_segmentation_workflow,
        event_group_parent,
        group_event_keys,
        run_algorithmic_segmentation_sandbox,
        save_algorithmic_results_to_real_pkl,
    )

MIN_WINDOW_SECONDS = 2
MAP_DEFAULT_LOCATION_MAX_POINTS = 5000
MAP_SUBDAY_WINDOW_SECONDS = 24 * 60 * 60
MAP_SUBDAY_TARGET_INTERVAL_SECONDS = 1.0
MAP_SUBDAY_LOCATION_MAX_POINTS = 90000
_MODEL_FETCH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)
# Disable server-side relayout patching to avoid stale range patches overriding
# slider-driven redraws of the main plot.
ENABLE_RELAYOUT_PATCH = True

_label_db_by_key_cache = None

PEAK_DERIVATIVE_OPTIONS = [
    {"label": "raw", "value": "raw"},
    {"label": "spikeless", "value": "spikeless"},
    {"label": "smoothed", "value": "smoothed"},
    {"label": "normalized", "value": "normalized"},
    {"label": "narrow_bandpass", "value": "narrow_bandpass"},
    {"label": "broad_bandpass", "value": "broad_bandpass"},
]

PEAK_PARAM_UI_META = {
    "peak-broad-low": {"config_key": "BROAD_LOW_CUTOFF", "min": 0.0, "max": 20.0, "step": 0.01},
    "peak-broad-high": {"config_key": "BROAD_HIGH_CUTOFF", "min": 0.1, "max": 120.0, "step": 0.01},
    "peak-narrow-low": {"config_key": "NARROW_LOW_CUTOFF", "min": 0.0, "max": 60.0, "step": 0.01},
    "peak-narrow-high": {"config_key": "NARROW_HIGH_CUTOFF", "min": 0.1, "max": 120.0, "step": 0.01},
    "peak-filter-order": {"config_key": "FILTER_ORDER", "min": 1, "max": 8, "step": 1},
    "peak-spike-threshold": {"config_key": "SPIKE_THRESHOLD", "min": 0, "max": 5000, "step": 1},
    "peak-smooth-sec": {"config_key": "SMOOTH_SEC_MULTIPLIER", "min": 0.01, "max": 10.0, "step": 0.01},
    "peak-window-mult": {"config_key": "WINDOW_SIZE_MULTIPLIER", "min": 0.1, "max": 60.0, "step": 0.01},
    "peak-norm-noise": {"config_key": "NORMALIZATION_NOISE", "min": 0.0, "max": 2.0, "step": 0.001},
    "peak-height": {"config_key": "PEAK_HEIGHT", "min": -5.0, "max": 5.0, "step": 0.01},
    "peak-distance": {"config_key": "PEAK_DISTANCE_SEC", "min": 0.01, "max": 6.0, "step": 0.01},
    "peak-search-radius": {"config_key": "SEARCH_RADIUS_SEC", "min": 0.01, "max": 6.0, "step": 0.01},
    "peak-min-height": {"config_key": "MIN_PEAK_HEIGHT", "min": 0.0, "max": 50000.0, "step": 0.01},
    "peak-max-height": {"config_key": "MAX_PEAK_HEIGHT", "min": 1.0, "max": 2000000.0, "step": 1.0},
    "peak-hr-jump-frac": {"config_key": "HR_JUMP_FRAC", "min": 0.0, "max": 2.0, "step": 0.01},
    "peak-min-rr": {"config_key": "MIN_RR_SEC", "min": 0.01, "max": 3.0, "step": 0.01},
    "peak-max-hr": {"config_key": "MAX_HR_BPM", "min": 1.0, "max": 400.0, "step": 1.0},
    "peak-min-hr": {"config_key": "MIN_HR_BPM", "min": 0.0, "max": 80.0, "step": 0.1},
    "peak-anti-double-gap": {"config_key": "ANTI_DOUBLE_GAP_FACTOR", "min": 0.0, "max": 2.0, "step": 0.01},
    "peak-anti-double-window": {"config_key": "ANTI_DOUBLE_ROLLING_WINDOW_SEC", "min": 0.1, "max": 240.0, "step": 0.1},
}

PEAK_PARAM_DESCRIPTIONS = {
    "BROAD_LOW_CUTOFF": "Low cutoff for broad bandpass to capture the general rhythm envelope.",
    "BROAD_HIGH_CUTOFF": "High cutoff for broad bandpass before narrower filtering/refinement.",
    "NARROW_LOW_CUTOFF": "Low cutoff for narrow bandpass emphasizing beat/stroke-scale peaks.",
    "NARROW_HIGH_CUTOFF": "High cutoff for narrow bandpass used by final peak candidate search.",
    "FILTER_ORDER": "Butterworth filter order; higher values sharpen cutoff but can ring more.",
    "SPIKE_THRESHOLD": "Threshold for suppressing impulsive spikes before smoothing.",
    "SMOOTH_SEC_MULTIPLIER": "Smoothing window scale (seconds x multiplier) for noise reduction.",
    "WINDOW_SIZE_MULTIPLIER": "Sliding-window size multiplier used during normalization.",
    "NORMALIZATION_NOISE": "Small stabilizer added to normalization denominator.",
    "PEAK_HEIGHT": "Minimum normalized-domain amplitude for initial peak candidates.",
    "PEAK_DISTANCE_SEC": "Minimum separation (s) between accepted peak candidates.",
    "SEARCH_RADIUS_SEC": "Refinement radius (s) used to snap candidates to local maxima.",
    "MIN_PEAK_HEIGHT": "Minimum allowed peak height after refinement in original/smoothed domain.",
    "MAX_PEAK_HEIGHT": "Maximum allowed peak height after refinement in original/smoothed domain.",
    "HR_JUMP_FRAC": "Max allowed fractional jump between consecutive instantaneous rates.",
    "MIN_RR_SEC": "Minimum allowed RR interval (s) before rejecting implausibly close peaks.",
    "MAX_HR_BPM": "Upper physiological rate cap used during cleanup.",
    "MIN_HR_BPM": "Lower physiological rate cap used during cleanup.",
    "ANTI_DOUBLE_GAP_FACTOR": "Gap factor for anti-double-beat cleanup pass.",
    "ANTI_DOUBLE_ROLLING_WINDOW_SEC": "Rolling window (s) used by anti-double-beat cleanup.",
}

PEAK_PARAM_DISPLAY_NAMES = {
    "BROAD_LOW_CUTOFF": "Broad Low Cutoff",
    "BROAD_HIGH_CUTOFF": "Broad High Cutoff",
    "NARROW_LOW_CUTOFF": "Narrow Low Cutoff",
    "NARROW_HIGH_CUTOFF": "Narrow High Cutoff",
    "FILTER_ORDER": "Filter Order",
    "SPIKE_THRESHOLD": "Spike Threshold",
    "SMOOTH_SEC_MULTIPLIER": "Smooth Sec Multiplier",
    "WINDOW_SIZE_MULTIPLIER": "Window Size Multiplier",
    "NORMALIZATION_NOISE": "Normalization Noise",
    "PEAK_HEIGHT": "Peak Height",
    "PEAK_DISTANCE_SEC": "Peak Distance Sec",
    "SEARCH_RADIUS_SEC": "Search Radius Sec",
    "MIN_PEAK_HEIGHT": "Minimum Peak Height",
    "MAX_PEAK_HEIGHT": "Maximum Peak Height",
    "HR_JUMP_FRAC": "HR Jump Fraction",
    "MIN_RR_SEC": "Min RR Sec",
    "MAX_HR_BPM": "Max HR BPM",
    "MIN_HR_BPM": "Min HR BPM",
    "ANTI_DOUBLE_GAP_FACTOR": "Anti Double Gap Factor",
    "ANTI_DOUBLE_ROLLING_WINDOW_SEC": "Anti Double Rolling Window Sec",
}

PEAK_PARAM_IDS = [
    "peak-broad-low",
    "peak-broad-high",
    "peak-narrow-low",
    "peak-narrow-high",
    "peak-filter-order",
    "peak-spike-threshold",
    "peak-smooth-sec",
    "peak-window-mult",
    "peak-norm-noise",
    "peak-height",
    "peak-distance",
    "peak-search-radius",
    "peak-min-height",
    "peak-max-height",
    "peak-hr-jump-frac",
    "peak-min-rr",
    "peak-max-hr",
    "peak-min-hr",
    "peak-anti-double-gap",
    "peak-anti-double-window",
]

PEAK_PARAM_DEFAULTS = {
    "peak-broad-low": 8.0,
    "peak-broad-high": 50.0,
    "peak-narrow-low": 15.0,
    "peak-narrow-high": 40.0,
    "peak-filter-order": 2,
    "peak-spike-threshold": 400.0,
    "peak-smooth-sec": 0.16,
    "peak-window-mult": 6.35,
    "peak-norm-noise": 0.34,
    "peak-height": -0.4,
    "peak-distance": 0.16,
    "peak-search-radius": 0.2,
    "peak-min-height": 70.0,
    "peak-max-height": 12000.0,
    "peak-hr-jump-frac": 0.8,
    "peak-min-rr": 0.25,
    "peak-max-hr": 240.0,
    "peak-min-hr": 0.1,
    "peak-anti-double-gap": 0.75,
    "peak-anti-double-window": 10.0,
}

_PEAK_REF_VALUES = {}


def _collect_peak_reference_values(config_obj, private_root):
    refs = {
        "heart_rate": {},
        "stroke_rate": {},
    }
    try:
        datasets_cfg = (config_obj or {}).get("datasets", {}) or {}
        root = pathlib.Path(private_root)
        if not root.exists():
            return refs
        for dataset_name in datasets_cfg.keys():
            p = root / str(dataset_name) / "parameter_log.json"
            if not p.exists():
                continue
            try:
                rows = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for section, mode in (
                    ("hr_peak_detection_settings", "heart_rate"),
                    ("stroke_peak_detection_settings", "stroke_rate"),
                ):
                    vals = row.get(section)
                    if not isinstance(vals, dict):
                        continue
                    for key, value in vals.items():
                        try:
                            f = float(value)
                        except Exception:
                            continue
                        if not np.isfinite(f):
                            continue
                        refs[mode].setdefault(str(key), []).append(f)
        for mode in refs:
            for key, values in list(refs[mode].items()):
                dedup = sorted(list({round(float(v), 12) for v in values}))
                refs[mode][key] = dedup
    except Exception:
        return {"heart_rate": {}, "stroke_rate": {}}
    return refs


def _format_peak_ref_number(v):
    f = float(v)
    if abs(f) >= 1000:
        return f"{f:.0f}"
    if abs(f) >= 100:
        return f"{f:.1f}"
    if abs(f) >= 10:
        return f"{f:.2f}"
    if abs(f) >= 1:
        return f"{f:.3f}".rstrip("0").rstrip(".")
    return f"{f:.4f}".rstrip("0").rstrip(".")


def _peak_slider_bounds(param_id, value):
    meta = PEAK_PARAM_UI_META.get(param_id) or {}
    lo = float(meta.get("min", 0.0))
    hi = float(meta.get("max", 1.0))
    cfg_key = str(meta.get("config_key") or "")
    ref_vals = []
    if cfg_key:
        ref_vals.extend((_PEAK_REF_VALUES.get("heart_rate", {}) or {}).get(cfg_key, []) or [])
        ref_vals.extend((_PEAK_REF_VALUES.get("stroke_rate", {}) or {}).get(cfg_key, []) or [])
    try:
        v = float(value)
        if np.isfinite(v):
            ref_vals.append(v)
    except Exception:
        pass
    if ref_vals:
        rmin = min(ref_vals)
        rmax = max(ref_vals)
        span = max(1e-9, rmax - rmin)
        lo = min(lo, rmin - (0.2 * span))
        hi = max(hi, rmax + (0.2 * span))
    if hi <= lo:
        hi = lo + 1.0
    if meta.get("step") == 1:
        lo = float(np.floor(lo))
        hi = float(np.ceil(hi))
    return lo, hi


def _peak_slider_marks(param_id, value):
    meta = PEAK_PARAM_UI_META.get(param_id) or {}
    cfg_key = str(meta.get("config_key") or "")
    marks = {}

    def _add_mark(num, label, color="#73a9c4"):
        try:
            x = float(num)
        except Exception:
            return
        if not np.isfinite(x):
            return
        marks[x] = {"label": str(label), "style": {"color": color, "fontSize": "10px"}}

    _add_mark(value, _format_peak_ref_number(value), "#ffffff")
    for mode, prefix in (("heart_rate", "hr"), ("stroke_rate", "sr")):
        vals = ((_PEAK_REF_VALUES.get(mode, {}) or {}).get(cfg_key, []) or [])[:4]
        for v in vals:
            _add_mark(v, f"{prefix}:{_format_peak_ref_number(v)}")
    return marks


def _peak_reference_text(param_id):
    meta = PEAK_PARAM_UI_META.get(param_id) or {}
    cfg_key = str(meta.get("config_key") or "")
    if not cfg_key:
        return ""
    hr_vals = ((_PEAK_REF_VALUES.get("heart_rate", {}) or {}).get(cfg_key, []) or [])[:6]
    sr_vals = ((_PEAK_REF_VALUES.get("stroke_rate", {}) or {}).get(cfg_key, []) or [])[:6]
    hr_txt = ", ".join(_format_peak_ref_number(v) for v in hr_vals) if hr_vals else "none"
    sr_txt = ", ".join(_format_peak_ref_number(v) for v in sr_vals) if sr_vals else "none"
    return f"refs hr[{hr_txt}] sr[{sr_txt}]"


def _peak_slider_control(param_id, default_value, is_open=False):
    lo, hi = _peak_slider_bounds(param_id, default_value)
    meta = PEAK_PARAM_UI_META.get(param_id) or {}
    cfg_key = str(meta.get("config_key") or "")
    step = meta.get("step", 0.01)
    display_default = _format_peak_ref_number(default_value)
    title_row = html.Div(
        [
            html.Span(
                PEAK_PARAM_DISPLAY_NAMES.get(cfg_key, cfg_key.replace("_", " ").title()),
                className="peak-param-title",
            ),
            html.Div(
                [
                    html.Span(cfg_key, className="peak-param-key"),
                    html.Span(
                        _format_peak_ref_number(default_value),
                        id={"type": "peak-param-header-value", "index": param_id},
                        className="peak-param-header-value",
                    ),
                ],
                className="peak-param-right",
            ),
        ],
        className="peak-param-title-row",
    )
    return html.Details(
        [
            html.Summary(
                [
                    html.Span("", className="chip-handle peak-param-drag-handle", **{"aria-hidden": "true"}),
                    html.Span("", className="peak-param-chevron", **{"aria-hidden": "true"}),
                    title_row,
                ],
                className="peak-param-summary",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(_format_peak_ref_number(lo), className="peak-param-min"),
                            html.Span(_format_peak_ref_number(hi), className="peak-param-max"),
                            dcc.Input(
                                id=f"{param_id}-input",
                                type="number",
                                value=float(default_value),
                                step=step,
                                className="peak-param-input",
                                placeholder=display_default,
                            ),
                        ],
                        className="peak-param-range-row",
                    ),
                    dcc.Slider(
                        id=param_id,
                        min=lo,
                        max=hi,
                        step=step,
                        value=default_value,
                        marks=_peak_slider_marks(param_id, default_value),
                        tooltip={"always_visible": False},
                    ),
                    html.Div(_peak_reference_text(param_id), className="peak-ref-values"),
                    html.Div(PEAK_PARAM_DESCRIPTIONS.get(cfg_key, ""), className="peak-param-help"),
                ],
                className="peak-param-body",
            ),
        ],
        id={"type": "peak-param-card", "index": param_id},
        className="peak-param-card",
        open=bool(is_open),
    )


def _peak_defaults_for_mode(mode: str) -> dict:
    m = str(mode or "heart_rate").strip().lower()
    if m == "stroke_rate":
        defaults = {
            "BROAD_LOW_CUTOFF": 0.6,
            "BROAD_HIGH_CUTOFF": 2.4,
            "NARROW_LOW_CUTOFF": 0.9,
            "NARROW_HIGH_CUTOFF": 2.0,
            "FILTER_ORDER": 2,
            "SPIKE_THRESHOLD": 400,
            "SMOOTH_SEC_MULTIPLIER": 0.9,
            "WINDOW_SIZE_MULTIPLIER": 15.5,
            "NORMALIZATION_NOISE": 0.025,
            "PEAK_HEIGHT": -0.9,
            "PEAK_DISTANCE_SEC": 0.5,
            "SEARCH_RADIUS_SEC": 0.5,
            "MIN_PEAK_HEIGHT": 150.0,
            "MAX_PEAK_HEIGHT": 1000000.0,
            "enable_bandpass": True,
            "enable_spike_removal": True,
            "enable_absolute": True,
            "enable_smoothing": True,
            "enable_normalization": True,
            "enable_refinement": True,
            "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],
            "HR_JUMP_FRAC": 0.8,
            "MIN_RR_SEC": 0.25,
            "MAX_HR_BPM": 240.0,
            "MIN_HR_BPM": 0.1,
            "ANTI_DOUBLE_GAP_FACTOR": 0.75,
            "ANTI_DOUBLE_ROLLING_WINDOW_SEC": 10.0,
            "PICK_LAST_IN_CONFLICT_PAIR": True,
        }
    else:
        defaults = {
            "BROAD_LOW_CUTOFF": 8,
            "BROAD_HIGH_CUTOFF": 50,
            "NARROW_LOW_CUTOFF": 15,
            "NARROW_HIGH_CUTOFF": 40,
            "FILTER_ORDER": 2,
            "SPIKE_THRESHOLD": 400,
            "SMOOTH_SEC_MULTIPLIER": 0.16,
            "WINDOW_SIZE_MULTIPLIER": 6.35,
            "NORMALIZATION_NOISE": 0.34,
            "PEAK_HEIGHT": -0.4,
            "PEAK_DISTANCE_SEC": 0.16,
            "SEARCH_RADIUS_SEC": 0.2,
            "MIN_PEAK_HEIGHT": 70.0,
            "MAX_PEAK_HEIGHT": 12000.0,
            "enable_bandpass": True,
            "enable_spike_removal": True,
            "enable_absolute": True,
            "enable_smoothing": True,
            "enable_normalization": True,
            "enable_refinement": True,
            "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],
            "HR_JUMP_FRAC": 0.8,
            "MIN_RR_SEC": 0.25,
            "MAX_HR_BPM": 240.0,
            "MIN_HR_BPM": 0.1,
            "ANTI_DOUBLE_GAP_FACTOR": 0.75,
            "ANTI_DOUBLE_ROLLING_WINDOW_SEC": 10.0,
            "PICK_LAST_IN_CONFLICT_PAIR": True,
        }

    section = "stroke_peak_detection_settings" if m == "stroke_rate" else "hr_peak_detection_settings"
    keys = list(defaults.keys()) + [
        "HR_CONFLICT_RR_FACTOR",
        "DETECTION_DERIVATIVE_CHANNELS",
        "STROKE_PARENT_SIGNAL",
        "STROKE_CHANNEL",
    ]
    try:
        saved = (param_manager.get_from_config(variable_names=keys, section=section) or {})
    except Exception:
        saved = {}
    for k, v in saved.items():
        if v is not None:
            defaults[k] = v
    for k, template in list(defaults.items()):
        val = defaults.get(k)
        if isinstance(template, bool):
            if isinstance(val, str):
                defaults[k] = val.strip().lower() in {"1", "true", "yes", "on"}
            else:
                defaults[k] = bool(val)
        elif isinstance(template, (int, float)) and not isinstance(template, bool):
            try:
                defaults[k] = float(val) if isinstance(template, float) else int(round(float(val)))
            except Exception:
                defaults[k] = template
    # Keep anti-double aliases synchronized.
    if defaults.get("ANTI_DOUBLE_GAP_FACTOR") is None and defaults.get("HR_CONFLICT_RR_FACTOR") is not None:
        defaults["ANTI_DOUBLE_GAP_FACTOR"] = defaults["HR_CONFLICT_RR_FACTOR"]
    if defaults.get("HR_CONFLICT_RR_FACTOR") is None and defaults.get("ANTI_DOUBLE_GAP_FACTOR") is not None:
        defaults["HR_CONFLICT_RR_FACTOR"] = defaults["ANTI_DOUBLE_GAP_FACTOR"]
    deriv = defaults.get("DETECTION_DERIVATIVE_CHANNELS")
    if isinstance(deriv, str):
        cleaned = [x.strip() for x in deriv.replace(";", ",").split(",") if x.strip()]
        defaults["DETECTION_DERIVATIVE_CHANNELS"] = cleaned or ["normalized"]
    elif isinstance(deriv, (list, tuple)):
        defaults["DETECTION_DERIVATIVE_CHANNELS"] = [str(x).strip() for x in deriv if str(x).strip()] or ["normalized"]
    else:
        defaults["DETECTION_DERIVATIVE_CHANNELS"] = ["normalized"]
    return defaults


def _peak_prepare_signal_subset(parent_signal, channel, start_ts, end_ts):
    df = data_pkl.signal_data.get(parent_signal)
    if (not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns) and deployment_source is not None:
        try:
            df = deployment_source.load_signal_window(parent_signal, start_ts, end_ts)
        except Exception:
            df = None
    if not isinstance(df, pd.DataFrame) or "datetime" not in df.columns:
        return None, "Parent signal is unavailable."
    if channel not in df.columns:
        return None, f"Channel '{channel}' not found in '{parent_signal}'."

    work = df[["datetime", channel]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name)
    work[channel] = pd.to_numeric(work[channel], errors="coerce")
    work = work.dropna(subset=["datetime", channel]).sort_values("datetime")
    work = work[(work["datetime"] >= start_ts) & (work["datetime"] <= end_ts)]
    if len(work) < 10:
        return None, "Not enough samples in this window."
    return work.reset_index(drop=True), None


def _peak_upsert_intermediate_signals(mode, parent_signal, subset_df, params, results):
    if subset_df is None or subset_df.empty:
        return []
    prefix = "hr" if str(mode or "heart_rate").strip().lower() == "heart_rate" else "sr"
    dt = pd.to_datetime(subset_df["datetime"], errors="coerce")
    saved = []
    save_map = [
        ("broad_bandpass", "enable_bandpass"),
        ("narrow_bandpass", "enable_bandpass"),
        ("spikeless", "enable_spike_removal"),
        ("smoothed", "enable_smoothing"),
        ("normalized", "enable_normalization"),
    ]
    for key, flag in save_map:
        arr = results.get(key)
        if arr is None or not bool(params.get(flag, False)):
            continue
        vals = np.asarray(arr)
        if vals.ndim != 1 or len(vals) != len(dt):
            continue
        sig_name = f"{prefix}_{key}"
        col_name = key
        data_pkl.signal_data[sig_name] = pd.DataFrame({"datetime": dt, col_name: vals})
        data_pkl.signal_info[sig_name] = {
            "channels": [col_name],
            "metadata": {
                col_name: {
                    "original_name": f"{sig_name} ({col_name})",
                    "unit": "signal units",
                    "parent_signal": str(parent_signal),
                }
            },
            "derived_from_signals": [str(parent_signal)],
            "transformation_log": [
                f"Derived during {mode} peak utility mode.",
                f"Parameters: {', '.join(f'{k}={v}' for k, v in sorted((params or {}).items()))}",
            ],
        }
        allowlist = _plot_signal_allowlist(data_pkl)
        if sig_name not in allowlist:
            data_pkl.plot_signal_allowlist = allowlist + [sig_name]
        saved.append(sig_name)
    return saved


def _peak_default_plot_signals(mode, parent_signal):
    mode_key = str(mode or "heart_rate").strip().lower()
    parent = str(parent_signal or "").strip()
    base = (
        [parent, "depth", "prh", "heart_rate", "hr_broad_bandpass", "hr_narrow_bandpass", "hr_smoothed", "hr_normalized", "heart_rate_fixed"]
        if mode_key == "heart_rate"
        else [parent, "depth", "prh", "stroke_rate", "sr_broad_bandpass", "sr_narrow_bandpass", "sr_smoothed", "sr_normalized"]
    )
    allowed = set(_plot_signal_allowlist(data_pkl))
    return [s for s in base if s and (s in allowed or s in data_pkl.signal_data)]


def _signal_default_channels(signal_name):
    df = data_pkl.signal_data.get(signal_name)
    if not isinstance(df, pd.DataFrame):
        return []
    options = _signal_channels(data_pkl, signal_name)
    if not options:
        return []
    info_channels = [str(c) for c in ((data_pkl.signal_info.get(signal_name, {}) or {}).get("channels") or [])]
    selected = [c for c in info_channels if c in options]
    return selected or options


def _peak_events_from_results(results, signal_subset_df, mode, cleanup_diag=None):
    peak_df = results.get("peak_df", pd.DataFrame())
    if peak_df is None or peak_df.empty:
        return []
    mode = str(mode or "heart_rate").strip().lower()
    out = []
    if mode == "stroke_rate":
        prefix = "stroke"
    else:
        prefix = "heart"

    for _, row in peak_df.iterrows():
        key = str(row.get("key") or "")
        if "accepted" in key:
            out_key = f"{prefix}beat_auto_detect_accepted"
        elif "rejected" in key:
            out_key = f"{prefix}beat_auto_detect_rejected"
        elif "suggested" in key:
            out_key = "heartbeat_auto_detect_suggested"
        else:
            continue
        out.append(
            {
                "datetime": row.get("datetime"),
                "key": out_key,
                "short_description": f"{mode} peak preview",
                "type": "point",
                "duration": 0.0,
                "value": np.nan,
            }
        )

    if mode == "heart_rate" and cleanup_diag:
        for seg in results.get("cleanup_nan_segments", []) or []:
            s, e = int(seg[0]), int(seg[1])
            dt_start = signal_subset_df["datetime"].iloc[max(0, min(s, len(signal_subset_df) - 1))]
            dt_end = signal_subset_df["datetime"].iloc[max(0, min(e - 1, len(signal_subset_df) - 1))]
            dur = max(0.0, float((dt_end - dt_start).total_seconds()))
            out.append(
                {
                    "datetime": dt_start,
                    "key": "heartbeat_auto_detect_gap",
                    "short_description": "interval where HR was invalid",
                    "type": "interval_start",
                    "duration": dur,
                    "value": np.nan,
                }
            )
            out.append(
                {
                    "datetime": dt_end,
                    "key": "heartbeat_auto_detect_gap",
                    "short_description": "interval where HR was invalid: end",
                    "type": "interval_end",
                    "duration": dur,
                    "value": np.nan,
                }
            )
    return out


def _to_tzaware(value, tz_name):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def _parse_dt_or_fallback(value, fallback, tz_name):
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz_name)
        else:
            ts = ts.tz_convert(tz_name)
        return ts
    except Exception:
        return fallback


def _signal_channels(data_pkl, sig):
    signal_info = (getattr(data_pkl, "signal_info", {}) or {}).get(sig, {}) or {}
    info_channels = [str(c) for c in (signal_info.get("channels") or []) if str(c) != "datetime"]
    if info_channels:
        return info_channels
    df = (getattr(data_pkl, "signal_data", {}) or {}).get(sig)
    if df is None:
        return []
    return [c for c in df.columns if c != "datetime"]


def _plot_signal_allowlist(data_pkl_obj=None):
    obj = data_pkl if data_pkl_obj is None else data_pkl_obj
    allowed = getattr(obj, "plot_signal_allowlist", None)
    if isinstance(allowed, (list, tuple)):
        return [str(s) for s in allowed if str(s).strip()]
    signal_data = getattr(obj, "signal_data", {}) or {}
    return [str(s) for s in signal_data.keys() if str(s) != "location"]


def _available_signal_names(data_pkl_obj=None):
    obj = data_pkl if data_pkl_obj is None else data_pkl_obj
    names = set(str(s) for s in _plot_signal_allowlist(obj))
    signal_data = getattr(obj, "signal_data", {}) or {}
    names.update(str(s) for s in signal_data.keys())
    if deployment_source is not None:
        names.update(str(s) for s in deployment_source.signal_names())
    return sorted([s for s in names if s and s != "location"])


def _preferred_plot_signals(signal_names):
    excluded = {"clock", "time", "logger_status", "dives", "location"}
    names = [str(s) for s in (signal_names or []) if str(s).strip()]
    preferred = [s for s in names if s not in excluded]
    return preferred or [s for s in names if s != "location"]


def _payload_has_signal_rows(data_pkl_obj, signals):
    signal_data = getattr(data_pkl_obj, "signal_data", {}) or {}
    for sig in (signals or []):
        df = signal_data.get(sig)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return True
    return False


def _available_event_keys(data_pkl_obj=None):
    obj = data_pkl if data_pkl_obj is None else data_pkl_obj
    keys = []
    event_df = getattr(obj, "event_data", None)
    if isinstance(event_df, pd.DataFrame) and "key" in event_df.columns:
        keys.extend(str(k) for k in event_df["key"].dropna().unique())
    if deployment_source is not None:
        try:
            real_data_pkl = deployment_source.get_real_data_pkl()
            real_event_df = getattr(real_data_pkl, "event_data", None)
            if isinstance(real_event_df, pd.DataFrame) and "key" in real_event_df.columns:
                keys.extend(str(k) for k in real_event_df["key"].dropna().unique())
        except Exception:
            pass
    return sorted(set(k for k in keys if k), key=lambda x: x.lower())


def _is_runtime_signal_frame(df):
    return isinstance(df, pd.DataFrame) and not df.empty and "datetime" in df.columns


def _ensure_real_data_pkl_loaded():
    global data_pkl
    if deployment_source is None:
        return data_pkl
    if not bool(getattr(data_pkl, "__header_only__", False)):
        return data_pkl
    allowlist = list(_plot_signal_allowlist(data_pkl))
    real = deployment_source.get_real_data_pkl()
    try:
        setattr(real, "plot_signal_allowlist", allowlist)
    except Exception:
        pass
    data_pkl = real
    return data_pkl


def _refresh_metadata_shell():
    global data_pkl
    if deployment_source is None:
        return data_pkl
    data_pkl = deployment_source.build_metadata_shell(allowed_signals=_plot_signal_allowlist(data_pkl))
    return data_pkl


def _build_orientation_runtime_payload():
    """Build a runtime payload for orientation using netCDF-backed signal windows first.

    This preserves the fast deployment_source loading path and avoids forcing a
    full pickle load just to evaluate orientation availability.
    """
    base = data_pkl
    if deployment_source is None or not bool(getattr(base, "__header_only__", False)):
        return base

    signal_data = dict(getattr(base, "signal_data", {}) or {})
    required_signals = [
        "prh", "orientation", "attitude",
        "pitch", "roll", "heading", "heading2", "head", "yaw",
        "location", "gps", "track",
        "corrected_depth", "depth", "pressure", "odba",
        "stroke_rate", "sr",
    ]
    for sig in required_signals:
        existing = signal_data.get(sig)
        if _is_runtime_signal_frame(existing):
            continue
        try:
            meta = (deployment_source.signal_meta or {}).get(sig)
            if meta is None or meta.start is None or meta.end is None:
                continue
            loaded = deployment_source.load_signal_window(sig, meta.start, meta.end)
            if _is_runtime_signal_frame(loaded):
                signal_data[sig] = loaded
        except Exception:
            continue

    return SimpleNamespace(
        signal_data=signal_data,
        signal_info=(getattr(base, "signal_info", {}) or {}),
        event_data=(getattr(base, "event_data", None)),
        deployment_info=(getattr(base, "deployment_info", {}) or {}),
        animal_info=(getattr(base, "animal_info", {}) or {}),
        dataset_info=(getattr(base, "dataset_info", {}) or {}),
        deployment_name=(getattr(base, "deployment_name", deployment_id) or deployment_id),
    )


# Cache Notion model lookups per animal. The S3 model/texture URLs are presigned
# with a 1h expiry, so we cap the TTL below that and refetch after it lapses.
_MODEL_INFO_CACHE = {}
_MODEL_INFO_TTL_S = 45 * 60


def _cached_fetch_3d_model_info(animal_id):
    now = time.monotonic()
    hit = _MODEL_INFO_CACHE.get(animal_id)
    if hit and (now - hit[0]) < _MODEL_INFO_TTL_S:
        return hit[1]
    info = fetch_3d_model_info(animal_id)
    # Only cache successful lookups (avoid pinning transient Notion failures).
    if isinstance(info, dict) and info.get("ok"):
        _MODEL_INFO_CACHE[animal_id] = (now, info)
    return info


def _start_model_info_fetch(animal_id):
    if not animal_id:
        return None
    # Serve from cache synchronously when warm (no thread/Notion round-trip).
    hit = _MODEL_INFO_CACHE.get(animal_id)
    if hit and (time.monotonic() - hit[0]) < _MODEL_INFO_TTL_S:
        fut = concurrent.futures.Future()
        fut.set_result(hit[1])
        return fut
    try:
        return _MODEL_FETCH_EXECUTOR.submit(_cached_fetch_3d_model_info, animal_id)
    except Exception:
        return None


def _resolve_model_info_from_future(model_future, animal_id):
    if not animal_id:
        return {"ok": False, "message": "No animal ID available for this deployment."}
    if model_future is None:
        return _cached_fetch_3d_model_info(animal_id)
    try:
        # A cold Notion lookup can take ~20-25s; allow headroom so the first load
        # doesn't time out and refetch. It runs behind the 3D spinner and blocks
        # nothing else (video/data render meanwhile), so a longer wait is fine.
        return model_future.result(timeout=45)
    except Exception:
        return _cached_fetch_3d_model_info(animal_id)


def _normalize_channel_order(selected, current_order):
    selected = selected or []
    current_order = current_order or []
    kept = [c for c in current_order if c in selected]
    appended = [c for c in selected if c not in kept]
    return kept + appended


def _deployment_time_bounds(data_pkl_obj, tz_name):
    hdr_start = getattr(data_pkl_obj, "global_start", None)
    hdr_end = getattr(data_pkl_obj, "global_end", None)
    if hdr_start is not None and hdr_end is not None:
        return pd.Timestamp(hdr_start), pd.Timestamp(hdr_end)
    min_ts = None
    max_ts = None
    for df in getattr(data_pkl_obj, "signal_data", {}).values():
        if df is None or "datetime" not in df.columns:
            continue
        dt = pd.to_datetime(df["datetime"], errors="coerce").dropna()
        if dt.empty:
            continue
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(tz_name)
        else:
            dt = dt.dt.tz_convert(tz_name)
        cur_min = dt.min()
        cur_max = dt.max()
        min_ts = cur_min if min_ts is None or cur_min < min_ts else min_ts
        max_ts = cur_max if max_ts is None or cur_max > max_ts else max_ts
    if min_ts is None or max_ts is None:
        now = pd.Timestamp.now(tz=tz_name)
        return now - timedelta(hours=1), now
    return min_ts, max_ts


def _to_epoch_seconds(ts):
    return float(pd.Timestamp(ts).timestamp())


def _from_epoch_seconds(seconds_value, tz_name):
    return pd.Timestamp(seconds_value, unit="s", tz="UTC").tz_convert(tz_name)


def _slider_label_children(ts, tz_name):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(tz_name)
    else:
        t = t.tz_convert(tz_name)
    hhmmss = t.strftime("%H:%M:%S") + ".000"
    ymd = t.strftime("%Y-%m-%d")
    return [
        html.Div(hhmmss, className="slider-ts-time"),
        html.Div(ymd, className="slider-ts-date"),
    ]


def _format_ts_input(ts, tz_name):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(tz_name)
    else:
        t = t.tz_convert(tz_name)
    base = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    offset = t.strftime("%z")
    if len(offset) == 5:
        offset = f"{offset[:3]}:{offset[3:]}"
    return f"{base}{offset}"


def _display_tz_name(display_mode=None):
    mode = str(display_mode or time_display_tz_mode or "local").strip().lower()
    if mode == "utc":
        return "UTC"
    return str(deployment_tz_name or tz_name or "UTC")


def _time_display_label(display_mode=None):
    active_tz = _display_tz_name(display_mode)
    if active_tz == "UTC":
        return "Time selector timezone: UTC"
    return f"Time selector timezone: {active_tz} (deployment local)"


def _extract_first_color(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, str):
            return first
    if isinstance(value, dict):
        for k in ("color", "colors", "palette"):
            if k in value:
                return _extract_first_color(value[k])
    return None


def _is_hex_color(value):
    if not isinstance(value, str):
        return False
    c = value.strip()
    if not c.startswith("#") or len(c) != 7:
        return False
    try:
        int(c[1:], 16)
        return True
    except ValueError:
        return False


def _normalize_hex_color(value):
    if not isinstance(value, str):
        return None
    c = value.strip()
    if not c:
        return None
    if not c.startswith("#"):
        c = f"#{c}"
    if len(c) != 7:
        return None
    try:
        int(c[1:], 16)
    except ValueError:
        return None
    return c.upper()


def _hex_to_hsv(color_hex):
    c = _normalize_hex_color(color_hex)
    if not c:
        return 0, 0, 0
    r = int(c[1:3], 16) / 255.0
    g = int(c[3:5], 16) / 255.0
    b = int(c[5:7], 16) / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(round(h * 360)) % 360, int(round(s * 100)), int(round(v * 100))


def _hsv_to_hex(h, s, v):
    try:
        h_f = (float(h) % 360.0) / 360.0
        s_f = max(0.0, min(1.0, float(s) / 100.0))
        v_f = max(0.0, min(1.0, float(v) / 100.0))
    except Exception:
        return None
    r, g, b = colorsys.hsv_to_rgb(h_f, s_f, v_f)
    return "#{:02X}{:02X}{:02X}".format(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _hex_to_rgba(hex_color, opacity):
    c = _normalize_hex_color(hex_color)
    if not c:
        return None
    try:
        alpha = float(opacity)
    except Exception:
        alpha = 0.2
    alpha = max(0.0, min(1.0, alpha))
    r = int(c[1:3], 16)
    g = int(c[3:5], 16)
    b = int(c[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _coerce_float_or_none(value):
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _coerce_float_default(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _apply_orientation_offsets_json(data_json, x_deg=0.0, y_deg=0.0, z_deg=0.0):
    if not data_json:
        return data_json
    x_off = _coerce_float_default(x_deg, 0.0)
    y_off = _coerce_float_default(y_deg, 0.0)
    z_off = _coerce_float_default(z_deg, 0.0)
    if x_off == 0.0 and y_off == 0.0 and z_off == 0.0:
        return data_json
    try:
        frame = pd.read_json(io.StringIO(data_json), orient="split")
        if frame.empty:
            return data_json
        if "pitch" in frame.columns:
            frame["pitch"] = pd.to_numeric(frame["pitch"], errors="coerce") + x_off
        if "roll" in frame.columns:
            frame["roll"] = pd.to_numeric(frame["roll"], errors="coerce") + y_off
        if "heading" in frame.columns:
            frame["heading"] = pd.to_numeric(frame["heading"], errors="coerce") + z_off
        return frame.to_json(orient="split", date_format="iso")
    except Exception:
        return data_json


def _load_color_mapping(path):
    try:
        mapping_path = pathlib.Path(path)
        if not mapping_path.exists():
            return {}
        raw = json.loads(mapping_path.read_text())
        if not isinstance(raw, dict):
            return {}
        out = {}
        for key, val in raw.items():
            if str(key).startswith("__"):
                continue
            color = _extract_first_color(val)
            if isinstance(color, str) and color.strip():
                out[str(key)] = color.strip()
        return out
    except Exception:
        return {}


def _load_raw_color_mapping(path):
    try:
        mapping_path = pathlib.Path(path)
        if not mapping_path.exists():
            return {}
        raw = json.loads(mapping_path.read_text())
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_json_atomic(path, payload):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _normalize_event_style_entry(entry):
    style = dict(entry or {})
    out = {}
    color = _normalize_hex_color(style.get("color"))
    if color:
        out["color"] = color
    shade_color = _normalize_hex_color(style.get("shade_color"))
    if shade_color:
        out["shade_color"] = shade_color
    symbol = str(style.get("symbol") or "")
    if symbol in EVENT_SYMBOL_OPTIONS:
        out["symbol"] = symbol
    if "shade_enabled" in style:
        out["shade_enabled"] = bool(style.get("shade_enabled"))
    if "shade_opacity" in style:
        try:
            out["shade_opacity"] = max(0.0, min(1.0, float(style.get("shade_opacity"))))
        except Exception:
            pass
    shade_mode = _normalize_event_shade_mode(style.get("shade_mode"))
    if shade_mode:
        out["shade_mode"] = shade_mode
    pct_min = _coerce_float_or_none(style.get("shade_pct_min"))
    pct_max = _coerce_float_or_none(style.get("shade_pct_max"))
    if pct_min is not None:
        out["shade_pct_min"] = max(0.0, min(100.0, pct_min))
    if pct_max is not None:
        out["shade_pct_max"] = max(0.0, min(100.0, pct_max))
    return out


def _normalize_event_styles_map(styles):
    out = {}
    for event_key, entry in dict(styles or {}).items():
        k = str(event_key or "").strip()
        if not k:
            continue
        normalized = _normalize_event_style_entry(entry)
        if normalized:
            out[k] = normalized
    return out


def _load_event_styles(path):
    raw = _load_raw_color_mapping(path)
    styles = raw.get("__event_styles__")
    if not isinstance(styles, dict):
        return {}
    return _normalize_event_styles_map(styles)


def _save_event_styles(path, styles):
    raw = _load_raw_color_mapping(path)
    raw["__event_styles__"] = _normalize_event_styles_map(styles)
    _write_json_atomic(path, raw)


def _normalize_event_targets_map(targets):
    out = {}
    for event_key, values in dict(targets or {}).items():
        ev = str(event_key or "").strip()
        if not ev:
            continue
        normalized = []
        seen = set()
        for value in (values or []):
            v = str(value or "").strip()
            if not v or v in seen:
                continue
            seen.add(v)
            normalized.append(v)
        out[ev] = normalized
    return out


def _load_event_targets(path):
    raw = _load_raw_color_mapping(path)
    targets = raw.get("__event_targets__")
    if not isinstance(targets, dict):
        return {}
    return _normalize_event_targets_map(targets)


def _save_event_targets(path, targets):
    raw = _load_raw_color_mapping(path)
    raw["__event_targets__"] = _normalize_event_targets_map(targets)
    _write_json_atomic(path, raw)


def _save_color_mapping(path, mapping):
    try:
        mapping_path = pathlib.Path(path)
        if mapping_path.exists():
            raw = json.loads(mapping_path.read_text())
            if not isinstance(raw, dict):
                raw = {}
        else:
            raw = {}
    except Exception:
        raw = {}

    for k, v in (mapping or {}).items():
        if _is_hex_color(v):
            raw[str(k)] = v

    _write_json_atomic(path, raw)


def _load_peak_detect_view_defaults(path):
    raw = _load_raw_color_mapping(path)
    data = raw.get("__peak_detect_defaults__")
    return data if isinstance(data, dict) else {}


def _save_peak_detect_view_defaults(path, mode, payload):
    raw = _load_raw_color_mapping(path)
    root = raw.get("__peak_detect_defaults__")
    if not isinstance(root, dict):
        root = {}
    root[str(mode)] = dict(payload or {})
    raw["__peak_detect_defaults__"] = root
    _write_json_atomic(path, raw)


def _normalize_signal_axis_config(raw):
    out = {}
    raw_map = raw if isinstance(raw, dict) else {}
    for sig, cfg in raw_map.items():
        sig_key = str(sig or "").strip()
        if not sig_key:
            continue
        entry = dict(cfg or {}) if isinstance(cfg, dict) else {}
        min_v = _coerce_float_or_none(entry.get("min"))
        max_v = _coerce_float_or_none(entry.get("max"))
        enabled = entry.get("enabled")
        reverse = entry.get("reverse")
        if enabled is None:
            enabled = (min_v is not None) or (max_v is not None)
        clean = {"enabled": bool(enabled)}
        if reverse is not None:
            clean["reverse"] = bool(reverse)
        if min_v is not None:
            clean["min"] = float(min_v)
        if max_v is not None:
            clean["max"] = float(max_v)
        if "enabled" in clean or "min" in clean or "max" in clean:
            out[sig_key] = clean
    return out


def _default_reverse_y_for_signal(signal_name):
    return str(signal_name or "").strip().lower() in {"pressure", "depth"}


def _read_section_key(pm, section, key, deployment_id):
    try:
        entries = pm._load_config()  # Access raw config to detect deployment-level overrides.
    except Exception:
        return None
    for entry in entries or []:
        if str(entry.get("deployment_id")) != str(deployment_id):
            continue
        section_obj = entry.get(section, {})
        if not isinstance(section_obj, dict):
            return None
        return section_obj.get(key)
    return None


def _load_signal_axis_config_from_params(pm):
    try:
        merged = (pm.get_from_config(["dash_signal_axis_config"], section="dash_plot_settings") or {}).get("dash_signal_axis_config")
    except Exception:
        merged = None
    return _normalize_signal_axis_config(merged)


def _persist_signal_axis_config(pm, cfg):
    if pm is None:
        return
    section = "dash_plot_settings"
    key = "dash_signal_axis_config"
    clean = _normalize_signal_axis_config(cfg)
    defaults_raw = _read_section_key(pm, section, key, pm.DEFAULTS_DEPLOYMENT_ID)
    defaults_clean = _normalize_signal_axis_config(defaults_raw)
    dep_raw = _read_section_key(pm, section, key, pm.deployment_id)
    has_dep_override = dep_raw is not None
    try:
        if has_dep_override:
            pm.add_to_config(entries={key: clean}, section=section, deployment_id=pm.deployment_id)
            if clean == defaults_clean:
                pm.remove_from_config(key, section=section, deployment_id=pm.deployment_id)
        else:
            pm.set_dataset_defaults(entries={key: clean}, section=section)
    except Exception:
        pass


def _normalize_model_3d_controls(cfg):
    src = dict(cfg or {}) if isinstance(cfg, dict) else {}
    pitch_offset = _coerce_float_default(src.get("pitch_offset"), 0.0)
    roll_offset = _coerce_float_default(src.get("roll_offset"), 0.0)
    heading_offset = _coerce_float_default(src.get("heading_offset"), 0.0)
    track_line_width = _coerce_float_default(src.get("track_line_width"), 7.1)
    highlight_width = _coerce_float_default(src.get("highlight_width"), 1.55)
    highlight_offset = _coerce_float_default(src.get("highlight_offset"), 0.18)

    def _norm_sign(value):
        try:
            v = float(value)
        except Exception:
            v = 1.0
        return -1 if v < 0 else 1

    return {
        "pitch_offset": float(pitch_offset),
        "roll_offset": float(roll_offset),
        "heading_offset": float(heading_offset),
        "pitch_sign": _norm_sign(src.get("pitch_sign", 1)),
        "roll_sign": _norm_sign(src.get("roll_sign", 1)),
        "heading_sign": _norm_sign(src.get("heading_sign", 1)),
        "rotation_order": _normalize_rotation_order(src.get("rotation_order")),
        "track_line_width": float(min(max(track_line_width, 1.0), 20.0)),
        "highlight_width": float(min(max(highlight_width, 0.1), 12.0)),
        "highlight_offset": float(min(max(highlight_offset, 0.0), 1.5)),
    }


def _load_model_3d_controls_from_params(pm):
    if pm is None:
        return _normalize_model_3d_controls({})
    try:
        merged = (pm.get_from_config(["dash_model_3d_controls_global"], section="settings") or {}).get("dash_model_3d_controls_global")
    except Exception:
        merged = None
    if not isinstance(merged, dict):
        # Backward-compat fallback: previous deployment-scoped storage
        try:
            merged = (pm.get_from_config(["dash_model_3d_controls"], section="dash_plot_settings") or {}).get("dash_model_3d_controls")
        except Exception:
            merged = None
    return _normalize_model_3d_controls(merged)


def _persist_model_3d_controls_dataset_defaults(pm, cfg):
    if pm is None:
        return False
    section = "settings"
    key = "dash_model_3d_controls_global"
    clean = _normalize_model_3d_controls(cfg)
    try:
        pm.set_dataset_defaults(entries={key: clean}, section=section)
        return True
    except Exception:
        return False


def _normalize_peak_param_order(order):
    base = [str(p) for p in PEAK_PARAM_IDS]
    requested = [str(p) for p in (order or []) if str(p) in base]
    if not requested:
        return base
    return requested + [p for p in base if p not in requested]


def _load_peak_param_order_from_params(pm):
    try:
        raw = (pm.get_from_config(["dash_peak_param_order"], section="dash_plot_settings") or {}).get("dash_peak_param_order")
    except Exception:
        raw = None
    if not isinstance(raw, list):
        return _normalize_peak_param_order(None)
    return _normalize_peak_param_order(raw)


def _persist_peak_param_order_dataset_defaults(pm, order):
    if pm is None:
        return False
    try:
        normalized = _normalize_peak_param_order(order)
        pm.set_dataset_defaults(entries={"dash_peak_param_order": normalized}, section="dash_plot_settings")
        return True
    except Exception:
        return False
    except Exception:
        return False


def _normalize_rotation_order(value):
    allowed = {"roll", "pitch", "heading"}
    default_order = ["roll", "pitch", "heading"]
    if not isinstance(value, (list, tuple)):
        return default_order
    out = []
    seen = set()
    for item in value:
        key = str(item or "").strip().lower()
        if key in allowed and key not in seen:
            out.append(key)
            seen.add(key)
    for key in default_order:
        if key not in seen:
            out.append(key)
    return out[:3]


def _resolve_trace_color(mapping, signal_name, channel_name):
    sig = str(signal_name or "")
    ch = str(channel_name or "")
    keys = (f"{sig}.{ch}", f"{sig}:{ch}", ch, ch.lower(), ch.upper())
    for key in keys:
        color = (mapping or {}).get(key)
        if isinstance(color, str) and color.strip():
            return color.strip()
    return None


def _fallback_color_for_key(key):
    token = str(key or "").strip()
    if not token:
        return "#87BADD"
    hue = (abs(hash(token)) % 360) / 360.0
    sat = 0.52
    val = 0.88
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def _safe_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _load_signal_display_metadata(data_dir_path):
    """
    Build signal/channel display metadata from cached Notion snapshot when available.
    Returns map keyed by signal identifiers used in data_pkl (plus common aliases).
    """
    md_path = pathlib.Path(data_dir_path) / "00_Metadata" / "metadata_snapshot.pkl"
    if not md_path.exists():
        return {}

    try:
        with md_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return {}

    metadata_obj = payload.get("metadata_obj") if isinstance(payload, dict) else payload

    def _get_table(name):
        if metadata_obj is None:
            return None
        getter = getattr(metadata_obj, "get_metadata", None)
        if callable(getter):
            try:
                return getter(name)
            except Exception:
                return None
        if isinstance(metadata_obj, dict):
            return metadata_obj.get(name) or metadata_obj.get(name.lower())
        return None

    signal_df = _get_table("signalchannel_DB")
    if not isinstance(signal_df, pd.DataFrame) or signal_df.empty:
        signal_df = _get_table("signal_DB")
    channel_df = _get_table("standardizedchannel_DB")
    if not isinstance(signal_df, pd.DataFrame) or signal_df.empty:
        return {}

    base = {}
    alias_to_key = {}

    for _, row in signal_df.iterrows():
        page_id = _safe_text(row.get("page_id"))
        name = _safe_text(row.get("Name"))
        key = name or page_id
        if not key:
            continue
        entry = base.get(key, {"label": key, "unit": "", "description": "", "channels": {}})
        entry["label"] = _safe_text(row.get("Label")) or entry["label"]
        entry["unit"] = _safe_text(row.get("Standardized Unit")) or entry["unit"]
        entry["description"] = _safe_text(row.get("Description")) or entry["description"]
        base[key] = entry
        for alias in (key, name, page_id):
            alias = _safe_text(alias)
            if alias:
                alias_to_key[alias] = key

    if isinstance(channel_df, pd.DataFrame) and not channel_df.empty:
        for _, row in channel_df.iterrows():
            parent = _safe_text(row.get("Parent signal"))
            channel_id = _safe_text(row.get("Channel ID"))
            if not parent or not channel_id:
                continue
            sig_key = alias_to_key.get(parent)
            if not sig_key:
                sig_key = parent
                if sig_key not in base:
                    base[sig_key] = {"label": sig_key, "unit": "", "description": "", "channels": {}}
                    alias_to_key[sig_key] = sig_key

            sig_meta = base[sig_key]
            suffix = _safe_text(row.get("Label Suffix"))
            ch_label = f"{sig_meta.get('label', sig_key)} {suffix}".strip() if suffix else channel_id
            ch_unit = _safe_text(row.get("Unit Override")) or _safe_text(sig_meta.get("unit"))
            ch_desc = _safe_text(row.get("Description Suffix"))
            sig_meta.setdefault("channels", {})[channel_id] = {
                "label": ch_label,
                "unit": ch_unit,
                "description": ch_desc,
                "label_suffix": suffix,
            }

    out = {}
    for alias, key in alias_to_key.items():
        if key in base:
            out[alias] = base[key]
    for key, value in base.items():
        out.setdefault(key, value)
    return out


def _is_light_hex(color):
    if not isinstance(color, str):
        return False
    c = color.strip().lstrip("#")
    if len(c) != 6:
        return False
    try:
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
    except ValueError:
        return False
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return luminance >= 186


def _signal_chip_style(signal_name):
    sig = str(signal_name or "")
    color = (
        _signal_colors.get(sig)
        or _signal_colors.get(sig.lower())
        or _signal_colors.get(sig.upper())
    )
    if not color:
        return None
    return {
        "backgroundColor": color,
        "borderColor": color,
        "color": "#0c1622" if _is_light_hex(color) else "#ffffff",
    }


def _channel_chip_style(signal_name, channel_name):
    sig = str(signal_name or "")
    ch = str(channel_name or "")
    # Prefer channel-specific colors; fall back to signal color.
    for key in (ch, ch.lower(), ch.upper(), f"{sig}.{ch}", f"{sig}:{ch}"):
        color = _signal_colors.get(key)
        if color:
            return {
                "backgroundColor": color,
                "borderColor": color,
                "color": "#0c1622" if _is_light_hex(color) else "#ffffff",
            }
    return _signal_chip_style(sig)


def _event_chip_style(event_styles, event_key, index=None):
    style = dict((event_styles or {}).get(str(event_key)) or {})
    color = _normalize_hex_color(style.get("color"))
    if not color:
        color = _normalize_hex_color(_default_event_style(event_key, int(index or 0)).get("color"))
    if not color:
        return None
    return {
        "backgroundColor": color,
        "borderColor": color,
        "color": "#0c1622" if _is_light_hex(color) else "#ffffff",
    }


def _render_sortable_chips(values, group, key, color_fn=None, editable_colors=False, removable=False, label_fn=None):
    values = values or []
    chips = [
        html.Div(
            [
                html.Span("", className="chip-handle", **{"aria-hidden": "true"}),
                html.Span((label_fn(v) if callable(label_fn) else str(v)), className="chip-label"),
                html.Button(
                    "✎",
                    id={"type": "chip-color-btn", "group": str(group), "key": str(key), "value": str(v)},
                    className="chip-color-edit",
                    n_clicks=0,
                    title="Edit trace color",
                ) if editable_colors else None,
                html.Button(
                    "×",
                    id={"type": "chip-remove-btn", "group": str(group), "key": str(key), "value": str(v)},
                    className="chip-remove-btn",
                    n_clicks=0,
                    title="Remove",
                ) if removable else None,
            ],
            className="chip-item",
            style=(color_fn(v) if callable(color_fn) else None),
            draggable="true",
            **{
                "data-value": str(v),
                "data-color-editable": "1" if editable_colors else "0",
            },
        )
        for v in values
    ]
    return html.Div(
        chips,
        className="chip-sortable",
        **{"data-order-group": str(group), "data-order-key": str(key)},
    )


def _render_peak_param_cards(order, open_map=None):
    ordered = _normalize_peak_param_order(order)
    open_map = dict(open_map or {})
    cards = []
    for pid in ordered:
        cards.append(
            html.Div(
                _peak_slider_control(pid, PEAK_PARAM_DEFAULTS.get(pid, 0.0), is_open=bool(open_map.get(pid, False))),
                className="dnd-item peak-param-dnd-item",
                draggable="true",
                **{"data-value": str(pid)},
            )
        )
    return html.Div(
        cards,
        className="signal-sortable peak-param-sortable",
        **{"data-order-group": "peak_params", "data-order-key": "__all__", "data-dnd-axis": "y"},
    )


def _render_summary_chips(values, color_fn=None, label_fn=None):
    vals = [v for v in (values or []) if str(v).strip()]
    if not vals:
        return html.Div([html.Span("None", className="summary-chip muted")], className="summary-chips")
    chips = []
    for v in vals:
        style = color_fn(v) if callable(color_fn) else None
        label = label_fn(v) if callable(label_fn) else str(v)
        chips.append(html.Span(label, className="summary-chip", style=style))
    return html.Div(chips, className="summary-chips")


def _event_symbol_value(event_key, event_styles, index):
    style_saved = dict((event_styles or {}).get(str(event_key)) or {})
    style_defaults = _default_event_style(event_key, index)
    symbol_value = str(style_saved.get("symbol") or style_defaults["symbol"])
    if symbol_value not in EVENT_SYMBOL_OPTIONS:
        symbol_value = "circle"
    return symbol_value


def _event_chip_label(event_key, event_styles, index):
    symbol = _event_symbol_value(event_key, event_styles, index)
    glyph = EVENT_SYMBOL_GLYPHS.get(symbol, "•")
    return f"{glyph} {event_key}"


def _render_event_key_chips(event_keys, group, event_styles):
    event_keys = [str(v) for v in (event_keys or []) if str(v).strip()]
    chips = []
    for idx, ev in enumerate(event_keys):
        chips.append(
            html.Div(
                [
                    html.Span("", className="chip-handle", **{"aria-hidden": "true"}),
                    html.Span(_event_chip_label(ev, event_styles, idx), className="chip-label"),
                    html.Button(
                        "✎",
                        id={"type": "event-edit-btn", "group": str(group), "event": ev},
                        className="chip-color-edit",
                        n_clicks=0,
                        title="Edit event",
                    ),
                    html.Button(
                        "×",
                        id={"type": "chip-remove-btn", "group": str(group), "key": "__all__", "value": ev},
                        className="chip-remove-btn",
                        n_clicks=0,
                        title="Remove",
                    ),
                ],
                className="chip-item",
                style=_event_chip_style(event_styles, ev, idx),
                draggable="true",
                **{"data-value": ev, "data-color-editable": "0"},
            )
        )
    return html.Div(
        chips,
        className="chip-sortable",
        **{"data-order-group": str(group), "data-order-key": "__all__"},
    )


def _panel_drag_handle():
    return html.Span("", className="panel-drag-handle", title="Drag panel")


def _comma_placeholder(values):
    vals = [str(v) for v in (values or []) if str(v).strip()]
    return ", ".join(vals) if vals else "Select..."


def _channel_base(name):
    return str(name or "").split("__", 1)[0]


def _parse_target_option(option):
    text = str(option or "")
    if "." not in text:
        return None, None
    sig, ch = text.split(".", 1)
    return sig, ch


def _match_targets(target_options, signal_candidates=None, channel_candidates=None):
    signal_candidates = [str(s) for s in (signal_candidates or []) if s]
    channel_candidates = [str(c) for c in (channel_candidates or []) if c]
    signal_set = {s.lower() for s in signal_candidates}
    channel_set = {_channel_base(c).lower() for c in channel_candidates}

    matches = []
    for opt in target_options or []:
        sig, ch = _parse_target_option(opt)
        if not sig or not ch:
            continue
        sig_ok = True if not signal_set else (str(sig).lower() in signal_set)
        ch_ok = True if not channel_set else (_channel_base(ch).lower() in channel_set)
        if sig_ok and ch_ok:
            matches.append(str(opt))
    return matches


def _get_detection_channel_defaults():
    stroke_parent = None
    stroke_channel = None
    heart_parent = None
    heart_channel = None

    try:
        stroke_cfg = param_manager.get_from_config(
            variable_names=["STROKE_PARENT_SIGNAL", "STROKE_CHANNEL"],
            section="stroke_peak_detection_settings",
        ) or {}
        stroke_parent = stroke_cfg.get("STROKE_PARENT_SIGNAL")
        stroke_channel = stroke_cfg.get("STROKE_CHANNEL")
    except Exception:
        pass

    try:
        hr_cfg = param_manager.get_from_config(
            variable_names=[
                "HEART_PARENT_SIGNAL",
                "HEART_CHANNEL",
                "HR_PARENT_SIGNAL",
                "HR_CHANNEL",
                "PEAK_PARENT_SIGNAL",
                "PEAK_CHANNEL",
            ],
            section="hr_peak_detection_settings",
        ) or {}
        heart_parent = (
            hr_cfg.get("HEART_PARENT_SIGNAL")
            or hr_cfg.get("HR_PARENT_SIGNAL")
            or hr_cfg.get("PEAK_PARENT_SIGNAL")
        )
        heart_channel = (
            hr_cfg.get("HEART_CHANNEL")
            or hr_cfg.get("HR_CHANNEL")
            or hr_cfg.get("PEAK_CHANNEL")
        )
    except Exception:
        pass

    if (not heart_parent or not heart_channel) and "heart_rate_fixed" in data_pkl.signal_info:
        try:
            meta = (data_pkl.signal_info.get("heart_rate_fixed", {}) or {}).get("metadata", {}) or {}
            info = meta.get("heart_rate_fixed", {}) if isinstance(meta, dict) else {}
            heart_parent = heart_parent or info.get("signal")
            heart_channel = heart_channel or info.get("signal")
        except Exception:
            pass

    return {
        "stroke_parent": stroke_parent,
        "stroke_channel": stroke_channel,
        "heart_parent": heart_parent,
        "heart_channel": heart_channel,
    }


def _default_targets_for_event(event_key, target_options):
    key = str(event_key or "").lower()
    options = [str(o) for o in (target_options or []) if str(o).strip()]
    if not options:
        return []

    detection = _get_detection_channel_defaults()
    out = []

    options_set = set(options)

    def add_signal_candidates(candidates):
        for c in candidates:
            sig = str(c or "").strip()
            if not sig:
                continue
            if sig in options_set and sig not in out:
                out.append(sig)

    if "stroke" in key:
        add_signal_candidates(["sr_smoothed", "stroke_rate"])
        add_signal_candidates([detection.get("stroke_parent")])
        add_signal_candidates(["corrected_gyr", "gyroscope", "dynamic_accel", "corrected_acc", "accelerometer"])
    elif "dive" in key:
        add_signal_candidates(["depth", "pressure"])
    elif "heart" in key or "beat" in key:
        add_signal_candidates(["hr_smoothed", "heart_rate", "heart_rate_fixed"])
        add_signal_candidates([detection.get("heart_parent")])
        add_signal_candidates(["ecg", "corrected_gyr", "gyroscope", "accelerometer"])
    elif "logger" in key or "status" in key:
        add_signal_candidates(["logger_status"])

    if not out:
        out = [options[0]]
    return out


EVENT_STYLE_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B"]
EVENT_SYMBOL_OPTIONS = [
    "circle",
    "diamond",
    "square",
    "triangle-up",
    "triangle-down",
    "triangle-left",
    "triangle-right",
    "cross",
    "x",
    "star",
]

EVENT_SYMBOL_GLYPHS = {
    "circle": "●",
    "diamond": "◆",
    "square": "■",
    "triangle-up": "▲",
    "triangle-down": "▼",
    "triangle-left": "◀",
    "triangle-right": "▶",
    "cross": "✚",
    "x": "✕",
    "star": "★",
}

EVENT_SYMBOL_DROPDOWN_OPTIONS = [
    {"label": EVENT_SYMBOL_GLYPHS.get(s, s), "value": s}
    for s in EVENT_SYMBOL_OPTIONS
]
EVENT_SHADE_MODE_OPTIONS = [
    {"label": "Filled trace", "value": "fill_trace"},
    {"label": "Filled trace split", "value": "fill_trace_split"},
    {"label": "Filled trace above", "value": "fill_trace_above"},
]
EVENT_SHADE_MODE_VALUES = {opt["value"] for opt in EVENT_SHADE_MODE_OPTIONS}


def _normalize_event_shade_mode(value, default="fill_trace_split"):
    shade_mode = str(value or "").strip().lower()
    if shade_mode in EVENT_SHADE_MODE_VALUES:
        return shade_mode
    if shade_mode in {"all_y", "trace_to_zero", "percent_band"}:
        return default
    return default if default in EVENT_SHADE_MODE_VALUES else "fill_trace_split"


def _default_event_style(event_key, index):
    key = str(event_key or "").lower()
    mapped_color = _mapped_event_color(event_key)
    key_token = key or str(index or 0)
    palette_idx = 0
    for i, ch in enumerate(key_token):
        palette_idx = (palette_idx + ((i + 1) * ord(ch))) % len(EVENT_STYLE_PALETTE)
    symbol = "circle"
    if "dive" in key:
        symbol = "triangle-down"
    elif "heart" in key or "beat" in key:
        symbol = "diamond"
    elif "stroke" in key:
        symbol = "x"
    return {"color": mapped_color or EVENT_STYLE_PALETTE[palette_idx], "symbol": symbol}


def _event_key_alias_candidates(event_key):
    raw = str(event_key or "").strip()
    if not raw:
        return []
    lower = raw.lower()
    cands = [raw]

    # Common singular/plural variants.
    if lower.endswith("ies") and len(raw) > 3:
        cands.append(raw[:-3] + "y")
    if lower.endswith("s") and len(raw) > 1:
        cands.append(raw[:-1])
    else:
        cands.append(raw + "s")

    # Explicit high-value aliases.
    explicit = {
        "dive": "dives",
        "dives": "dive",
    }
    mapped = explicit.get(lower)
    if mapped:
        cands.append(mapped)

    # Behavior-key compatibility aliases.
    # Supports behavior_*, raw labels (active_*/calm_*), and rf_* export keys.
    if lower.startswith("behavior_"):
        core = raw[len("behavior_") :]
        if core:
            cands.append(core)
            cands.append("rf_" + core)
    elif lower.startswith("rf_"):
        core = raw[len("rf_") :]
        if core:
            cands.append(core)
            cands.append("behavior_" + core)
    else:
        cands.append("behavior_" + raw)
        cands.append("rf_" + raw)

    out = []
    seen = set()
    for c in cands:
        k = str(c).strip()
        if not k:
            continue
        lk = k.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(k)
    return out


def _mapped_event_color(event_key):
    key = str(event_key or "").strip()
    if not key:
        return None
    mapping = globals().get("_signal_colors") or {}
    if not isinstance(mapping, dict):
        return None
    for cand in _event_key_alias_candidates(key):
        color = _normalize_hex_color(mapping.get(cand))
        if color:
            return color
    return None


def _resolve_event_key_alias(data_pkl_obj, event_key):
    raw = str(event_key or "").strip()
    if not raw:
        return raw
    if not (
        hasattr(data_pkl_obj, "event_data")
        and isinstance(data_pkl_obj.event_data, pd.DataFrame)
        and not data_pkl_obj.event_data.empty
        and "key" in data_pkl_obj.event_data.columns
    ):
        return raw

    available = [str(k) for k in data_pkl_obj.event_data["key"].dropna().unique()]
    available_set = set(available)
    available_lower = {k.lower(): k for k in available}

    if raw in available_set:
        return raw
    if raw.lower() in available_lower:
        return available_lower[raw.lower()]

    for cand in _event_key_alias_candidates(raw):
        if cand in available_set:
            return cand
        mapped = available_lower.get(cand.lower())
        if mapped:
            return mapped
    return raw


def _load_label_db_by_key():
    global _label_db_by_key_cache
    if isinstance(_label_db_by_key_cache, dict):
        return _label_db_by_key_cache
    label_db_path = pathlib.Path(_LOCAL_PYOLOGGER_ROOT) / "label_db.json"
    try:
        with open(label_db_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        by_key = payload.get("by_key") or {}
        _label_db_by_key_cache = by_key if isinstance(by_key, dict) else {}
    except Exception:
        _label_db_by_key_cache = {}
    return _label_db_by_key_cache


def _label_db_entry_for_event(event_key):
    by_key = _load_label_db_by_key()
    if not isinstance(by_key, dict) or not by_key:
        return None
    raw = str(event_key or "").strip()
    if not raw:
        return None
    if raw in by_key:
        return by_key.get(raw)
    raw_lower = raw.lower()
    for k, entry in by_key.items():
        if str(k).lower() == raw_lower:
            return entry
    for cand in _event_key_alias_candidates(raw):
        if cand in by_key:
            return by_key.get(cand)
        cand_lower = str(cand).lower()
        for k, entry in by_key.items():
            if str(k).lower() == cand_lower:
                return entry
    return None


def _has_state_type_flag(data_pkl_obj, event_key):
    if not (
        hasattr(data_pkl_obj, "event_data")
        and isinstance(data_pkl_obj.event_data, pd.DataFrame)
        and not data_pkl_obj.event_data.empty
        and "key" in data_pkl_obj.event_data.columns
    ):
        return False
    resolved_key = _resolve_event_key_alias(data_pkl_obj, event_key)
    ev_df = data_pkl_obj.event_data[data_pkl_obj.event_data["key"] == resolved_key]
    if ev_df.empty or "type" not in ev_df.columns:
        return False
    event_types = ev_df["type"].astype(str).str.strip().str.lower()
    return bool(event_types.eq("state").any())


def _is_state_event_key(data_pkl_obj, event_key):
    if _has_state_duration_events(data_pkl_obj, event_key):
        return True
    if _has_state_type_flag(data_pkl_obj, event_key):
        return True

    if deployment_source is not None:
        try:
            real_data = deployment_source.get_real_data_pkl()
        except Exception:
            real_data = None
        if real_data is not None:
            if _has_state_duration_events(real_data, event_key):
                return True
            if _has_state_type_flag(real_data, event_key):
                return True

    entry = _label_db_entry_for_event(event_key)
    if isinstance(entry, dict):
        return str(entry.get("type") or "").strip().lower() == "state"
    key_lower = str(event_key or "").strip().lower()
    if key_lower.startswith("rf_"):
        return True
    return False


def _has_state_duration_events(data_pkl_obj, event_key):
    if not (
        hasattr(data_pkl_obj, "event_data")
        and isinstance(data_pkl_obj.event_data, pd.DataFrame)
        and not data_pkl_obj.event_data.empty
        and "key" in data_pkl_obj.event_data.columns
    ):
        return False
    resolved_key = _resolve_event_key_alias(data_pkl_obj, event_key)
    ev_df = data_pkl_obj.event_data[data_pkl_obj.event_data["key"] == resolved_key]
    if ev_df.empty:
        return False
    if "type" in ev_df.columns:
        event_types = ev_df["type"].astype(str).str.strip().str.lower()
        non_empty = event_types[event_types.ne("")]
        # Treat interval-like "event" rows as state-compatible when they also
        # carry duration or end-time bounds (validated below).
        if not non_empty.empty and not (non_empty.eq("state").any() or non_empty.eq("event").any()):
            return False
    for duration_col in ["duration", "duration_pos", "duration_sec", "duration_s"]:
        if duration_col in ev_df.columns:
            dur = pd.to_numeric(ev_df[duration_col], errors="coerce")
            if dur.notna().any() and (dur > 0).any():
                return True
    for end_col in ["end_datetime", "end_time", "datetime_end", "end", "stop", "state_end"]:
        if end_col in ev_df.columns:
            end_dt = pd.to_datetime(ev_df[end_col], errors="coerce")
            if end_dt.notna().any():
                return True
    return False


def _split_event_keys_by_state(data_pkl_obj, event_keys):
    point_keys = []
    state_keys = []
    for ev in (event_keys or []):
        if _is_state_event_key(data_pkl_obj, ev):
            state_keys.append(ev)
        else:
            point_keys.append(ev)
    return point_keys, state_keys


def _extract_depth_context_series(data_pkl_obj, tz_name, max_points=2500):
    signal_data = getattr(data_pkl_obj, "signal_data", {}) or {}
    df = None
    value_col = None
    source_signal = None
    for sig_name in ["corrected_depth", "depth", "pressure", "odba"]:
        candidate = signal_data.get(sig_name)
        if not isinstance(candidate, pd.DataFrame) or candidate.empty or "datetime" not in candidate.columns:
            continue
        value_cols = [c for c in candidate.columns if c != "datetime"]
        if not value_cols:
            continue
        # Prefer depth-like columns when present; otherwise use first value column.
        chosen = value_cols[0]
        for c in value_cols:
            lc = str(c).lower()
            if "depth" in lc or lc == "p" or "odba" in lc:
                chosen = c
                break
        df = candidate
        value_col = chosen
        source_signal = sig_name
        break
    if (df is None or value_col is None) and deployment_source is not None:
        for sig_name in ["corrected_depth", "depth", "pressure", "odba"]:
            try:
                meta = (deployment_source.signal_meta or {}).get(sig_name)
                if meta is None or meta.start is None or meta.end is None:
                    continue
                candidate = deployment_source.load_signal_window(sig_name, meta.start, meta.end)
            except Exception:
                continue
            if not isinstance(candidate, pd.DataFrame) or candidate.empty or "datetime" not in candidate.columns:
                continue
            value_cols = [c for c in candidate.columns if c != "datetime"]
            if not value_cols:
                continue
            chosen = value_cols[0]
            for c in value_cols:
                lc = str(c).lower()
                if "depth" in lc or lc == "p" or "odba" in lc:
                    chosen = c
                    break
            df = candidate
            value_col = chosen
            source_signal = sig_name
            break
    if df is None or value_col is None:
        return None, None, "Depth", "m", None

    dfx = pd.to_datetime(df["datetime"], errors="coerce")
    dfy = pd.to_numeric(df[value_col], errors="coerce")
    valid = dfx.notna() & dfy.notna()
    if not valid.any():
        return None, None

    x = dfx[valid]
    y = dfy[valid]
    if getattr(x.dt, "tz", None) is None:
        x = x.dt.tz_localize(tz_name)
    else:
        x = x.dt.tz_convert(tz_name)

    if len(x) > max_points:
        step = max(1, len(x) // max_points)
        x = x.iloc[::step]
        y = y.iloc[::step]

    axis_label = _vertical_axis_label_for_signal(source_signal)
    unit = "" if str(source_signal or "").strip().lower() == "odba" else "m"
    return x, y, axis_label, unit, source_signal


def _build_depth_context_figure(start_ts, end_ts, playhead_ts=None):
    fig = BasePlotlyFigure()
    if _depth_ctx_x is None or _depth_ctx_y is None or len(_depth_ctx_x) == 0:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Depth/Pressure/ODBA context unavailable",
            showarrow=False,
            font={"color": "#73a9c4", "size": 12},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        fig.add_trace(
            go.Scatter(
                x=_depth_ctx_x,
                y=_depth_ctx_y,
                mode="lines",
                line={"color": "#4daef8", "width": 1.8},
                hoverinfo="skip",
                name=str(_depth_ctx_label or "Depth").lower(),
            )
        )
        fig.add_vrect(
            x0=pd.Timestamp(start_ts),
            x1=pd.Timestamp(end_ts),
            fillcolor="rgba(133, 198, 255, 0.30)",
            line_width=1,
            line_color="#a7d7ff",
            layer="above",
        )
        y_title = str(_depth_ctx_label or "Depth")
        if str(_depth_ctx_unit or ""):
            y_title = f"{y_title} ({_depth_ctx_unit})"
        fig.update_yaxes(
            title=y_title,
            autorange="reversed",
            showgrid=True,
            gridcolor="rgba(115,169,196,0.18)",
            zeroline=False,
        )
        fig.update_xaxes(showgrid=False)
        try:
            p_ts = pd.Timestamp(playhead_ts) if playhead_ts is not None else pd.Timestamp(start_ts) + ((pd.Timestamp(end_ts) - pd.Timestamp(start_ts)) / 2)
            if p_ts.tzinfo is None:
                p_ts = p_ts.tz_localize(tz_name)
            else:
                p_ts = p_ts.tz_convert(tz_name)
            fig.add_shape(
                type="line",
                x0=p_ts,
                x1=p_ts,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line={"width": 2, "color": "#FFD166"},
                layer="above",
                editable=True,
            )

            depth_df = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(_depth_ctx_x, errors="coerce"),
                    "depth": pd.to_numeric(_depth_ctx_y, errors="coerce"),
                }
            ).dropna(subset=["datetime", "depth"])
            if not depth_df.empty:
                if getattr(depth_df["datetime"].dt, "tz", None) is None:
                    depth_df["datetime"] = depth_df["datetime"].dt.tz_localize(tz_name)
                else:
                    depth_df["datetime"] = depth_df["datetime"].dt.tz_convert(tz_name)
                nearest_idx = (depth_df["datetime"] - p_ts).abs().idxmin()
                p_x = depth_df.loc[nearest_idx, "datetime"]
                p_y = float(depth_df.loc[nearest_idx, "depth"])
                fig.add_trace(
                    go.Scatter(
                        x=[p_x],
                        y=[p_y],
                        mode="markers",
                        marker={"size": 8, "color": "#FFD166", "line": {"width": 1, "color": "#C79A2E"}},
                        hovertemplate="Playhead<extra></extra>",
                        showlegend=False,
                    )
                )
        except Exception:
            pass

    fig.update_layout(
        height=118,
        margin={"l": 58, "r": 10, "t": 8, "b": 8},
        paper_bgcolor="#0a2b42",
        plot_bgcolor="#08263a",
        font={"color": "#ffffff"},
        showlegend=False,
    )
    return fig


def _window_span_seconds(start_ts, end_ts):
    try:
        span = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds()
        if pd.isna(span):
            return None
        return max(0.0, float(span))
    except Exception:
        return None


def _downsample_location_by_interval(loc_df, target_interval_seconds, max_points):
    if not isinstance(loc_df, pd.DataFrame) or loc_df.empty:
        return pd.DataFrame(columns=["datetime", "lat", "lon"])
    work = loc_df.sort_values("datetime").reset_index(drop=True)
    if len(work) <= 1:
        return work

    try:
        delta_seconds = pd.to_numeric(work["datetime"].diff().dt.total_seconds(), errors="coerce")
        median_delta = float(delta_seconds[delta_seconds > 0].median())
    except Exception:
        median_delta = np.nan

    # If native cadence is already coarser than target interval, preserve native rows.
    if np.isfinite(median_delta) and median_delta >= float(target_interval_seconds):
        sampled = work
    else:
        bucket_ns = max(1, int(float(target_interval_seconds) * 1_000_000_000))
        datetime_ns = work["datetime"].astype("int64")
        buckets = datetime_ns // bucket_ns
        sampled = work.loc[~buckets.duplicated(keep="first")].reset_index(drop=True)

    if len(sampled) > int(max_points):
        idx = np.linspace(0, len(sampled) - 1, num=int(max_points), dtype=int)
        sampled = sampled.iloc[np.unique(idx)].reset_index(drop=True)
    return sampled


def _extract_location_series(
    data_pkl_obj,
    tz_name,
    *,
    start_ts=None,
    end_ts=None,
    max_points=MAP_DEFAULT_LOCATION_MAX_POINTS,
):
    df = getattr(data_pkl_obj, "signal_data", {}).get("location")
    if (not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns) and deployment_source is not None:
        try:
            if (
                start_ts is not None
                and end_ts is not None
                and "location" in (deployment_source.signal_meta or {})
            ):
                span_seconds = _window_span_seconds(start_ts, end_ts)
                # Large-window map bootstrap should avoid expensive full-range netCDF
                # timestamp index decoding; use the track sampler path instead.
                if span_seconds is not None and span_seconds > MAP_SUBDAY_WINDOW_SECONDS:
                    df = deployment_source.load_location_track(max_points=int(max_points))
                else:
                    df = deployment_source.load_signal_window("location", start_ts, end_ts)
            else:
                df = deployment_source.load_location_track(max_points=int(max_points))
        except Exception:
            df = None
    if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
        return pd.DataFrame(columns=["datetime", "lat", "lon"])

    work = df.copy()
    # Parse to UTC first so naive-UTC and tz-aware strings land on a common timeline.
    dt = pd.to_datetime(work["datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name)
    work["datetime"] = dt

    value_cols = [c for c in work.columns if c != "datetime"]
    lowered = {c: str(c).lower() for c in value_cols}

    def _pick_col(tokens):
        for c in value_cols:
            lc = lowered[c]
            if any(tok in lc for tok in tokens):
                return c
        return None

    lat_col = _pick_col(["latitude", "lat", "gps_0"])
    lon_col = _pick_col(["longitude", "lon", "gps_1"])
    if lat_col is None or lon_col is None:
        # Do not guess columns here; guessing can pick unrelated variables and
        # create invalid map ranges/layout artifacts.
        return pd.DataFrame(columns=["datetime", "lat", "lon"])

    out = pd.DataFrame(
        {
            "datetime": work["datetime"],
            "lat": pd.to_numeric(work[lat_col], errors="coerce"),
            "lon": pd.to_numeric(work[lon_col], errors="coerce"),
        }
    ).dropna(subset=["datetime", "lat", "lon"])
    if start_ts is not None and end_ts is not None and not out.empty:
        out = out[(out["datetime"] >= pd.Timestamp(start_ts)) & (out["datetime"] <= pd.Timestamp(end_ts))]
    return out.sort_values("datetime").reset_index(drop=True)


def _extract_depth_series_for_map_with_source(data_pkl_obj, tz_name):
    signal_data = getattr(data_pkl_obj, "signal_data", {}) or {}

    # Prefer corrected depth when available; fall back to depth/pressure/odba.
    depth_signal_candidates = ["corrected_depth", "depth", "pressure", "odba"]
    df = None
    source_signal = None
    for sig_name in depth_signal_candidates:
        candidate = signal_data.get(sig_name)
        if isinstance(candidate, pd.DataFrame) and not candidate.empty and "datetime" in candidate.columns:
            df = candidate
            source_signal = sig_name
            break

    if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
        if deployment_source is not None:
            for sig_name in depth_signal_candidates:
                try:
                    meta = (deployment_source.signal_meta or {}).get(sig_name)
                    if meta is None or meta.start is None or meta.end is None:
                        continue
                    df = deployment_source.load_signal_window(sig_name, meta.start, meta.end)
                    if isinstance(df, pd.DataFrame) and not df.empty and "datetime" in df.columns:
                        source_signal = sig_name
                        break
                except Exception:
                    continue
        if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
            return pd.DataFrame(columns=["datetime", "depth"]), None
    work = df.copy()
    # Parse to UTC first so naive-UTC and tz-aware strings land on a common timeline.
    dt = pd.to_datetime(work["datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name)
    work["datetime"] = dt
    value_cols = [c for c in work.columns if c != "datetime"]
    if not value_cols:
        return pd.DataFrame(columns=["datetime", "depth"]), source_signal
    depth_col = value_cols[0]
    for c in value_cols:
        lc = str(c).lower()
        if "depth" in lc or lc == "p" or "odba" in lc:
            depth_col = c
            break
    out = pd.DataFrame(
        {
            "datetime": work["datetime"],
            "depth": pd.to_numeric(work[depth_col], errors="coerce"),
        }
    ).dropna(subset=["datetime", "depth"])
    return out.sort_values("datetime"), source_signal


def _extract_depth_series_for_map(data_pkl_obj, tz_name):
    out, _source_signal = _extract_depth_series_for_map_with_source(data_pkl_obj, tz_name)
    return out


def _vertical_axis_label_for_signal(source_signal):
    src = str(source_signal or "").strip().lower()
    return "ODBA" if src == "odba" else "Depth"


def _vertical_axis_value_unit_for_signal(source_signal):
    src = str(source_signal or "").strip().lower()
    if src == "odba":
        return ""
    return " m"


def _is_depth_like_signal(source_signal):
    src = str(source_signal or "").strip().lower()
    return src in {"corrected_depth", "depth", "pressure"}


def _signal_valid_window(data_pkl_obj, signal_name, tz_name):
    df = (getattr(data_pkl_obj, "signal_data", {}) or {}).get(signal_name)
    if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
        return None, None
    value_cols = [c for c in df.columns if c != "datetime"]
    if not value_cols:
        return None, None

    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name)
    value_frame = df[value_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = dt.notna() & value_frame.notna().any(axis=1)
    if not valid_mask.any():
        return None, None

    valid_dt = dt[valid_mask]
    return valid_dt.min(), valid_dt.max()


def _trim_datapkl_to_window(data_pkl_obj, start_ts, end_ts, tz_name):
    signal_data = getattr(data_pkl_obj, "signal_data", {}) or {}
    for sig_name, df in list(signal_data.items()):
        if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
            continue
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name)
        keep = dt.notna() & (dt >= start_ts) & (dt <= end_ts)
        trimmed = df.loc[keep].copy()
        if "datetime" in trimmed.columns:
            trimmed["datetime"] = dt.loc[keep].values
        signal_data[sig_name] = trimmed

    ev_df = getattr(data_pkl_obj, "event_data", None)
    if isinstance(ev_df, pd.DataFrame) and not ev_df.empty and "datetime" in ev_df.columns:
        ev_dt = pd.to_datetime(ev_df["datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name)
        keep = ev_dt.notna() & (ev_dt >= start_ts) & (ev_dt <= end_ts)
        if "end_datetime" in ev_df.columns:
            ev_end = pd.to_datetime(ev_df["end_datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name)
            overlap = ev_dt.notna() & ev_end.notna() & (ev_dt <= end_ts) & (ev_end >= start_ts)
            keep = keep | overlap
        data_pkl_obj.event_data = ev_df.loc[keep].copy()


def _signal_cache_signature(df, tz_name):
    if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
        return (0, None, None)
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name).dropna()
    if dt.empty:
        return (0, None, None)
    return (int(len(df)), int(dt.iloc[0].value), int(dt.iloc[-1].value))


def _build_downsampled_signal_data_for_plot(data_pkl_obj, signals, target_hz, tz_name):
    signal_data = getattr(data_pkl_obj, "signal_data", {}) or {}
    out = dict(signal_data)
    try:
        target_hz_val = float(target_hz)
    except Exception:
        return out
    if target_hz_val <= 0:
        return out

    dep_key = str(getattr(data_pkl_obj, "deployment_name", "") or deployment_id)
    for sig in (signals or []):
        df = signal_data.get(sig)
        if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
            continue
        sig_key = (
            dep_key,
            str(sig),
            round(target_hz_val, 6),
            _signal_cache_signature(df, tz_name),
        )
        cached = _downsample_cache.get(sig_key)
        if isinstance(cached, pd.DataFrame):
            out[sig] = cached
            continue

        work = df.copy()
        work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
        work = work.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        if work.empty:
            _downsample_cache[sig_key] = work
            out[sig] = work
            continue
        original_fs = calculate_sampling_frequency(work["datetime"])
        reduced = downsample(work, original_fs, target_hz_val)
        _downsample_cache[sig_key] = reduced
        out[sig] = reduced
    return out


def _is_map_3d_enabled(toggle_value):
    if isinstance(toggle_value, bool):
        return toggle_value
    if isinstance(toggle_value, str):
        return toggle_value.strip().lower() in {"on", "true", "1", "yes"}
    if isinstance(toggle_value, (list, tuple, set)):
        return "on" in toggle_value
    return False


def _default_playhead_epoch(window_value):
    try:
        if isinstance(window_value, (list, tuple)) and len(window_value) == 2:
            lo = float(min(window_value))
            hi = float(max(window_value))
            return lo + (0.5 * (hi - lo))
    except Exception:
        pass
    try:
        return float(slider_default[0]) + (0.5 * (float(slider_default[1]) - float(slider_default[0])))
    except Exception:
        return 0.0


def _compute_square_lon_lat_ranges(loc_df):
    if not isinstance(loc_df, pd.DataFrame) or loc_df.empty:
        return None, None
    try:
        lon = pd.to_numeric(loc_df["lon"], errors="coerce").dropna()
        lat = pd.to_numeric(loc_df["lat"], errors="coerce").dropna()
    except Exception:
        return None, None
    if lon.empty or lat.empty:
        return None, None

    lon_min = float(lon.min())
    lon_max = float(lon.max())
    lat_min = float(lat.min())
    lat_max = float(lat.max())

    lon_mid = 0.5 * (lon_min + lon_max)
    lat_mid = 0.5 * (lat_min + lat_max)
    lon_span = max(1e-9, lon_max - lon_min)
    lat_span = max(1e-9, lat_max - lat_min)
    span = max(lon_span, lat_span)
    pad = max(1e-6, 0.05 * span)
    half = 0.5 * span + pad

    return [lon_mid - half, lon_mid + half], [lat_mid - half, lat_mid + half]


def _compute_padded_range(series, pad_frac=0.05):
    try:
        vals = pd.to_numeric(series, errors="coerce").dropna()
    except Exception:
        return None
    if vals.empty:
        return None
    lo = float(vals.min())
    hi = float(vals.max())
    span = max(1e-9, hi - lo)
    pad = max(1e-6, pad_frac * span)
    return [lo - pad, hi + pad]


def _sanitize_location_df(loc_df):
    if not isinstance(loc_df, pd.DataFrame) or loc_df.empty:
        return pd.DataFrame(columns=["datetime", "lat", "lon"])
    out = loc_df.copy()
    out["lat"] = pd.to_numeric(out.get("lat"), errors="coerce")
    out["lon"] = pd.to_numeric(out.get("lon"), errors="coerce")
    out = out.dropna(subset=["datetime", "lat", "lon"])
    # Hard bounds for geographic coordinates.
    out = out[(out["lat"] >= -90.0) & (out["lat"] <= 90.0) & (out["lon"] >= -180.0) & (out["lon"] <= 180.0)]
    return out


def _trim_location_outliers(loc_df, q_lo=0.01, q_hi=0.99):
    if not isinstance(loc_df, pd.DataFrame) or loc_df.empty:
        return loc_df
    if len(loc_df) < 20:
        return loc_df
    try:
        lat = pd.to_numeric(loc_df["lat"], errors="coerce")
        lon = pd.to_numeric(loc_df["lon"], errors="coerce")
        lat_lo, lat_hi = lat.quantile([q_lo, q_hi]).tolist()
        lon_lo, lon_hi = lon.quantile([q_lo, q_hi]).tolist()
        mask = lat.between(lat_lo, lat_hi) & lon.between(lon_lo, lon_hi)
        trimmed = loc_df.loc[mask].copy()
        # Keep original when quantile clipping removes too much.
        if len(trimmed) < 5:
            return loc_df
        return trimmed
    except Exception:
        return loc_df


def _build_location_map_figure(slider_value, map_3d_enabled=False, playhead_value=None):
    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        lo = int(min(slider_value))
        hi = int(max(slider_value))
    else:
        lo = int(slider_min)
        hi = int(slider_max)
    start_ts = _from_epoch_seconds(lo, tz_name)
    end_ts = _from_epoch_seconds(hi, tz_name)
    try:
        playhead_epoch = float(playhead_value) if playhead_value is not None else _default_playhead_epoch([lo, hi])
    except Exception:
        playhead_epoch = _default_playhead_epoch([lo, hi])
    playhead_epoch = max(float(lo), min(float(hi), float(playhead_epoch)))
    playhead_ts = _from_epoch_seconds(playhead_epoch, tz_name)

    span_seconds = _window_span_seconds(start_ts, end_ts)
    use_subday_policy = span_seconds is not None and span_seconds < MAP_SUBDAY_WINDOW_SECONDS
    load_max_points = MAP_SUBDAY_LOCATION_MAX_POINTS if use_subday_policy else MAP_DEFAULT_LOCATION_MAX_POINTS
    loc = _sanitize_location_df(
        _extract_location_series(
            data_pkl,
            tz_name,
            start_ts=start_ts,
            end_ts=end_ts,
            max_points=load_max_points,
        )
    )
    if use_subday_policy and not loc.empty:
        loc = _downsample_location_by_interval(
            loc,
            target_interval_seconds=MAP_SUBDAY_TARGET_INTERVAL_SECONDS,
            max_points=MAP_SUBDAY_LOCATION_MAX_POINTS,
        )
    lon_range = None
    lat_range = None
    # Use a plain Plotly figure here (not FigureResampler) because map x-values
    # are longitude, not monotonic time, and resampler enforces monotonic x.
    fig = BasePlotlyFigure()
    if loc.empty:
        fig.add_annotation(
            text="No location signal available in deployment outputs",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#73a9c4", "size": 13},
        )
    else:
        window_all = loc[(loc["datetime"] >= start_ts) & (loc["datetime"] <= end_ts)]
        highlight = loc[(loc["datetime"] >= start_ts) & (loc["datetime"] <= playhead_ts)]
        range_source_raw = window_all if not window_all.empty else loc
        range_source = _trim_location_outliers(range_source_raw)
        plot_track = _trim_location_outliers(window_all) if not window_all.empty else range_source
        highlight = _trim_location_outliers(highlight) if not highlight.empty else highlight
        lon_range = _compute_padded_range(range_source["lon"])
        lat_range = _compute_padded_range(range_source["lat"])
        if map_3d_enabled:
            depth_df, depth_source_signal = _extract_depth_series_for_map_with_source(data_pkl, tz_name)
            vertical_label = _vertical_axis_label_for_signal(depth_source_signal)
            use_depth_3d = _is_depth_like_signal(depth_source_signal)
            if use_depth_3d and not depth_df.empty:
                def _prep_for_asof(df_in):
                    if not isinstance(df_in, pd.DataFrame):
                        return pd.DataFrame({"_asof_ns": pd.Series(dtype="int64")})
                    df_out = df_in.copy()
                    if df_out.empty or "datetime" not in df_out.columns:
                        df_out["_asof_ns"] = pd.Series(dtype="int64")
                        return df_out
                    dt = pd.to_datetime(df_out["datetime"], errors="coerce", utc=True).dropna()
                    if dt.empty:
                        out = df_out.iloc[0:0].copy()
                        out["_asof_ns"] = pd.Series(dtype="int64")
                        return out
                    df_out = df_out.loc[dt.index].copy()
                    df_out["datetime"] = dt
                    df_out["_asof_ns"] = pd.to_numeric(df_out["datetime"].astype("int64"), errors="coerce")
                    df_out = df_out.dropna(subset=["_asof_ns"])
                    if df_out.empty:
                        out = df_out.iloc[0:0].copy()
                        out["_asof_ns"] = pd.Series(dtype="int64")
                        return out
                    df_out["_asof_ns"] = df_out["_asof_ns"].astype("int64")
                    return df_out.sort_values("_asof_ns", kind="mergesort").reset_index(drop=True)

                def _merge_nearest_depth(left_df, right_df):
                    if left_df.empty or right_df.empty:
                        out = left_df.copy()
                        if "depth" not in out.columns:
                            out["depth"] = pd.Series(dtype="float64")
                        return out
                    return pd.merge_asof(
                        left_df,
                        right_df[["_asof_ns", "depth"]],
                        on="_asof_ns",
                        direction="nearest",
                        tolerance=pd.Timedelta(seconds=5).value,
                    )

                depth_sorted = _prep_for_asof(depth_df)
                loc_sorted = _prep_for_asof(loc)
                highlight_sorted = _prep_for_asof(highlight)
                window_all_sorted = _prep_for_asof(window_all)

                loc_merged = _merge_nearest_depth(loc_sorted, depth_sorted).dropna(subset=["depth"])
                highlight_merged = _merge_nearest_depth(highlight_sorted, depth_sorted).dropna(subset=["depth"])
                window_all_merged = _merge_nearest_depth(window_all_sorted, depth_sorted).dropna(subset=["depth"])
                if not loc_merged.empty:
                    merged_range_source = window_all_merged if not window_all_merged.empty else loc_merged
                    lon_range, lat_range = _compute_square_lon_lat_ranges(merged_range_source)
            else:
                loc_merged = pd.DataFrame(columns=["lon", "lat", "depth"])
                highlight_merged = pd.DataFrame(columns=["lon", "lat", "depth"])
                window_all_merged = pd.DataFrame(columns=["lon", "lat", "depth"])

            if loc_merged.empty:
                reason = "No depth/pressure data available for 3D mode"
                if str(depth_source_signal or "").strip().lower() == "odba":
                    reason = "3D mode requires depth/pressure; ODBA remains 2D context only"
                fig.add_annotation(
                    text=reason,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font={"color": "#73a9c4", "size": 13},
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=loc_merged["lon"],
                        y=loc_merged["lat"],
                        z=loc_merged["depth"],
                        mode="lines",
                        name="Full Track",
                        line={"color": "rgba(140, 180, 210, 0.35)", "width": 3},
                        hoverinfo="skip",
                    )
                )
                if not highlight_merged.empty:
                    fig.add_trace(
                        go.Scatter3d(
                            x=highlight_merged["lon"],
                            y=highlight_merged["lat"],
                            z=highlight_merged["depth"],
                            mode="lines+markers",
                            name="Current Segment",
                            line={"color": "#4DAEF8", "width": 4},
                            marker={"size": 3, "color": "#7CC2FB"},
                            hovertemplate=f"Lon %{{x:.5f}}<br>Lat %{{y:.5f}}<br>{vertical_label} %{{z:.2f}}<extra></extra>",
                        )
                    )
                if not window_all_merged.empty:
                    nearest_idx = (window_all_merged["datetime"] - playhead_ts).abs().idxmin()
                    last = window_all_merged.loc[nearest_idx]
                    fig.add_trace(
                        go.Scatter3d(
                            x=[last["lon"]],
                            y=[last["lat"]],
                            z=[last["depth"]],
                            mode="markers",
                            name="Current Position",
                            marker={"size": 5, "color": "#FFD166"},
                            hovertemplate=f"Current<br>Lon %{{x:.5f}}<br>Lat %{{y:.5f}}<br>{vertical_label} %{{z:.2f}}<extra></extra>",
                        )
                    )
                z_range = None
                try:
                    z_source = window_all_merged if not window_all_merged.empty else loc_merged
                    z_vals = pd.to_numeric(z_source["depth"], errors="coerce").dropna()
                    if not z_vals.empty:
                        z_min = float(z_vals.min())
                        z_max = float(z_vals.max())
                        z_pad = max(1e-6, 0.05 * max(1e-9, z_max - z_min))
                        # Reversed depth axis (larger depth down) with explicit range.
                        z_range = [z_max + z_pad, z_min - z_pad]
                except Exception:
                    z_range = None

                fig.update_layout(
                    scene=dict(
                        xaxis_title="Longitude",
                        yaxis_title="Latitude",
                        zaxis_title=vertical_label,
                        aspectmode="cube",
                        xaxis=dict(
                            showgrid=True,
                            gridcolor="rgba(10,43,66,0.56)",
                            showbackground=True,
                            backgroundcolor="#0d3550",
                            zerolinecolor="rgba(10,43,66,0.64)",
                            range=lon_range,
                            autorange=False if lon_range else True,
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor="rgba(10,43,66,0.56)",
                            showbackground=True,
                            backgroundcolor="#0d3550",
                            zerolinecolor="rgba(10,43,66,0.64)",
                            range=lat_range,
                            autorange=False if lat_range else True,
                        ),
                        zaxis=dict(
                            showgrid=True,
                            gridcolor="rgba(10,43,66,0.56)",
                            showbackground=True,
                            backgroundcolor="#0d3550",
                            zerolinecolor="rgba(10,43,66,0.64)",
                            autorange="reversed",
                            range=z_range,
                        ),
                        bgcolor="#08263a",
                    )
                )
        else:
            full_mode = "lines" if len(plot_track) >= 2 else "markers"
            fig.add_trace(
                go.Scattergl(
                    x=plot_track["lon"],
                    y=plot_track["lat"],
                    mode=full_mode,
                    name="Full Track",
                    line={"color": "rgba(140, 180, 210, 0.35)", "width": 1.6},
                    marker={"size": 5, "color": "rgba(140, 180, 210, 0.80)"},
                    hoverinfo="skip",
                )
            )
            if not highlight.empty:
                fig.add_trace(
                    go.Scattergl(
                        x=highlight["lon"],
                        y=highlight["lat"],
                        mode="lines+markers",
                        name="Current Segment",
                        line={"color": "#4DAEF8", "width": 2.2},
                        marker={"size": 4, "color": "#7CC2FB"},
                        hovertemplate="Lon %{x:.5f}<br>Lat %{y:.5f}<extra></extra>",
                    )
                )
            if not window_all.empty:
                nearest_idx = (window_all["datetime"] - playhead_ts).abs().idxmin()
                last = window_all.loc[nearest_idx]
                fig.add_trace(
                    go.Scattergl(
                        x=[last["lon"]],
                        y=[last["lat"]],
                        mode="markers",
                        name="Current Position",
                        marker={"size": 8, "color": "#FFD166", "line": {"color": "#1A1A1A", "width": 1}},
                        hovertemplate="Current<br>Lon %{x:.5f}<br>Lat %{y:.5f}<extra></extra>",
                    )
                )

    fig.update_layout(
        height=290,
        margin={"l": 46, "r": 12, "t": 10, "b": 34},
        paper_bgcolor="#0a2b42",
        plot_bgcolor="#08263a",
        font={"color": "#ffffff"},
        showlegend=False,
    )
    if not map_3d_enabled:
        fig.update_xaxes(
            title_text="Longitude",
            showgrid=True,
            gridcolor="rgba(115,169,196,0.18)",
            zeroline=False,
            range=lon_range,
            autorange=False if lon_range else True,
            fixedrange=True,
        )
        fig.update_yaxes(
            title_text="Latitude",
            showgrid=True,
            gridcolor="rgba(115,169,196,0.18)",
            zeroline=False,
            range=lat_range,
            autorange=False if lat_range else True,
            fixedrange=True,
        )
    return fig


def _list_datasets(data_dir):
    p = pathlib.Path(data_dir)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def _list_deployments(data_dir, dataset_name):
    if not dataset_name:
        return []
    p = pathlib.Path(data_dir) / dataset_name
    if not p.exists():
        return []
    deployments = []
    for d in p.iterdir():
        if not d.is_dir():
            continue
        if (d / "outputs" / "data.pkl").exists():
            deployments.append(d.name)
    return sorted(deployments)


def _compute_view_context(data_pkl_obj, tz_name, start_default_ts, end_default_ts, param_manager_obj=None):
    g_start, g_end = _deployment_time_bounds(data_pkl_obj, tz_name)
    s_min = int(_to_epoch_seconds(g_start))
    s_max = int(_to_epoch_seconds(g_end))

    s_def = max(pd.Timestamp(start_default_ts), g_start)
    e_def = min(pd.Timestamp(end_default_ts), g_end)
    if e_def <= s_def:
        e_def = min(s_def + timedelta(minutes=2), g_end)

    s_default = [int(_to_epoch_seconds(s_def)), int(_to_epoch_seconds(e_def))]
    mid_epoch = int((s_min + s_max) / 2)
    marks = {
        s_min: g_start.strftime("%Y-%m-%d %H:%M"),
        mid_epoch: _from_epoch_seconds(mid_epoch, tz_name).strftime("%m-%d %H:%M"),
        s_max: g_end.strftime("%Y-%m-%d %H:%M"),
    }

    sigs = sorted(_plot_signal_allowlist(data_pkl_obj))
    defaults = list(sigs)
    if not defaults and sigs:
        defaults = [sigs[0]]
    ev_opts = []
    if isinstance(getattr(data_pkl_obj, "event_data", None), pd.DataFrame) and "key" in data_pkl_obj.event_data.columns:
        ev_opts = sorted([str(k) for k in data_pkl_obj.event_data["key"].dropna().unique()])
    default_events = list(ev_opts)
    try:
        saved_cfg = (param_manager_obj.get_from_config(
            ["dash_default_events", "plotly_default_events"], section="settings"
        ) if param_manager_obj is not None else {}) or {}
        saved_events = saved_cfg.get("dash_default_events") or saved_cfg.get("plotly_default_events") or []
        if isinstance(saved_events, (list, tuple)):
            saved_filtered = [str(e) for e in saved_events if str(e) in ev_opts]
            if saved_filtered:
                default_events = saved_filtered
    except Exception:
        pass

    return {
        "global_start": g_start,
        "global_end": g_end,
        "slider_min": s_min,
        "slider_max": s_max,
        "slider_default": s_default,
        "slider_marks": marks,
        "start_default": s_def,
        "end_default": e_def,
        "all_signals": sigs,
        "default_signals": defaults,
        "event_key_options": ev_opts,
        "default_events": default_events,
    }


def _segmentation_default_scope(data_dir_path, fallback_dataset, fallback_deployment):
    ds = DEFAULT_SEGMENTATION_DATASET if DEFAULT_SEGMENTATION_DATASET in _list_datasets(data_dir_path) else fallback_dataset
    dep_options = _list_deployments(data_dir_path, ds)
    dep = DEFAULT_SEGMENTATION_DEPLOYMENT if DEFAULT_SEGMENTATION_DEPLOYMENT in dep_options else fallback_deployment
    if dep not in dep_options:
        dep = dep_options[0] if dep_options else None
    return ds, dep


def _segmentation_initial_window(data_pkl_obj, tz_name, hours=DEFAULT_SEGMENTATION_WINDOW_HOURS):
    g_start, g_end = _deployment_time_bounds(data_pkl_obj, tz_name)
    end_ts = min(g_start + timedelta(hours=float(hours)), g_end)
    if end_ts <= g_start:
        end_ts = min(g_start + timedelta(minutes=2), g_end)
    return _compute_view_context(data_pkl_obj, tz_name, g_start, end_ts, param_manager_obj=None)


def _segmentation_default_signals(data_pkl_obj):
    sigs = sorted(list((getattr(data_pkl_obj, "signal_data", {}) or {}).keys()))
    for preferred in ("depth", "corrected_depth", "pressure"):
        if preferred in sigs:
            return [preferred], sigs
    return ([sigs[0]] if sigs else []), sigs


def _segmentation_depth_target_signal(data_pkl_obj, selected_signals):
    signal_data = getattr(data_pkl_obj, "signal_data", {}) or {}
    for preferred in ("depth", "corrected_depth", "pressure"):
        if preferred in signal_data:
            return preferred
    for sig in (selected_signals or []):
        if sig in signal_data:
            return sig
    return next(iter(signal_data.keys()), None)


def _segmentation_depth_target_channel(data_pkl_obj, signal_name):
    if not signal_name:
        return None
    channels = _signal_channels(data_pkl_obj, signal_name)
    for preferred in ("depth", "corrected_depth", "pressure"):
        if preferred in channels:
            return preferred
    return channels[0] if channels else None


def _load_segmentation_signal_defaults(seg_param_manager, all_sigs):
    all_sigs = [str(s) for s in (all_sigs or []) if str(s).strip()]
    if seg_param_manager is None or not all_sigs:
        return list(all_sigs)
    try:
        saved_cfg = (seg_param_manager.get_from_config(
            ["segmentation_review_signals", "segmentation_review_signal_order"],
            section="segmentation_settings",
        ) or {})
    except Exception:
        return list(all_sigs)
    saved = saved_cfg.get("segmentation_review_signal_order") or saved_cfg.get("segmentation_review_signals") or []
    if not isinstance(saved, (list, tuple)):
        return list(all_sigs)
    saved_filtered = [str(s) for s in saved if str(s) in all_sigs]
    return saved_filtered or list(all_sigs)


def _serialize_records_for_store(rows):
    out = []
    for row in list(rows or []):
        item = dict(row)
        for key, value in list(item.items()):
            if isinstance(value, pd.Timestamp):
                item[key] = value.isoformat() if not pd.isna(value) else None
            elif isinstance(value, np.generic):
                item[key] = value.item()
            elif pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
                item[key] = None
        out.append(item)
    return out


def _event_label_display_text(event_key, max_chars=58):
    key = str(event_key or "").strip()
    if not key:
        return ""
    if len(key) <= int(max_chars):
        return key
    tail_token = re.sub(r"[^A-Za-z0-9]", "", key)
    tail = tail_token[-3:] if tail_token else key[-3:]
    head_chars = max(12, int(max_chars) - 3 - len(tail))
    head = key[:head_chars].rstrip("_- .")
    return f"{head}...{tail}"


def _cluster_preview_file_candidates(dataset_id_value):
    ds = str(dataset_id_value or "").strip()
    if not ds:
        return []
    root = pathlib.Path(data_dir) / ds / "00_Meta-Analysis" / "segmentation"
    if not root.exists():
        return []
    files = list(root.glob("*/clustering/clustered_windows.parquet"))
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return files


def _load_cluster_preview_rows(dataset_id_value, deployment_id_value, event_keys, start_ts=None, end_ts=None, max_files=6):
    keys = [str(v).strip() for v in (event_keys or []) if str(v or "").strip()]
    if not keys:
        return pd.DataFrame(columns=["datetime", "end_datetime", "key", "duration", "type", "short_description", "value"])

    dep = str(deployment_id_value or "").strip()
    key_set = set(keys)
    rows = []
    candidates = _cluster_preview_file_candidates(dataset_id_value)[: max(1, int(max_files))]
    for path_obj in candidates:
        try:
            cdf = pd.read_parquet(path_obj)
        except Exception:
            continue
        if not isinstance(cdf, pd.DataFrame) or cdf.empty:
            continue
        if "deployment_id" in cdf.columns:
            cdf = cdf.loc[cdf["deployment_id"].astype(str) == dep].copy()
        if cdf.empty:
            continue
        if "window_start" not in cdf.columns or "window_end" not in cdf.columns:
            continue

        key_cols = [c for c in ["cluster_key_pca", "cluster_key_umap", "cluster_key_tsne", "cluster_key", "__cluster_key"] if c in cdf.columns]
        if not key_cols:
            continue

        starts = pd.to_datetime(cdf["window_start"], errors="coerce")
        ends = pd.to_datetime(cdf["window_end"], errors="coerce")
        duration_s = (ends - starts).dt.total_seconds()
        rank_values = pd.to_numeric(cdf.get("cluster_rank"), errors="coerce") if "cluster_rank" in cdf.columns else pd.Series([np.nan] * len(cdf))

        for key_col in key_cols:
            sub = cdf.loc[cdf[key_col].astype(str).isin(key_set)].copy()
            if sub.empty:
                continue
            idx = sub.index
            frame = pd.DataFrame(
                {
                    "datetime": starts.loc[idx],
                    "end_datetime": ends.loc[idx],
                    "key": sub[key_col].astype(str),
                    "duration": duration_s.loc[idx],
                    "type": "state",
                    "short_description": "Cluster state",
                    "value": rank_values.loc[idx],
                }
            )
            rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["datetime", "end_datetime", "key", "duration", "type", "short_description", "value"])

    merged = pd.concat(rows, ignore_index=True)
    merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce")
    merged["end_datetime"] = pd.to_datetime(merged["end_datetime"], errors="coerce")
    merged = merged.dropna(subset=["datetime", "end_datetime", "key"]).loc[lambda d: d["end_datetime"] > d["datetime"]]
    if start_ts is not None:
        merged = merged.loc[merged["end_datetime"] >= pd.Timestamp(start_ts)]
    if end_ts is not None:
        merged = merged.loc[merged["datetime"] <= pd.Timestamp(end_ts)]
    if merged.empty:
        return pd.DataFrame(columns=["datetime", "end_datetime", "key", "duration", "type", "short_description", "value"])
    merged = merged.sort_values(["key", "datetime", "end_datetime"]).drop_duplicates(["key", "datetime", "end_datetime"]).reset_index(drop=True)
    return merged


def _append_cluster_preview_events_to_payload(data_for_plot, selected_state_keys, start_ts, end_ts):
    cluster_keys = [str(k).strip() for k in (selected_state_keys or []) if looks_like_ordered_cluster_key(str(k))]
    if not cluster_keys:
        return data_for_plot, []

    base_events = getattr(data_for_plot, "event_data", None)
    if not isinstance(base_events, pd.DataFrame):
        base_events = pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value", "end_datetime"])

    existing_keys = set(base_events["key"].dropna().astype(str).tolist()) if "key" in base_events.columns else set()
    missing = [k for k in cluster_keys if k not in existing_keys]
    if not missing:
        return data_for_plot, []

    preview_df = _load_cluster_preview_rows(dataset_id, deployment_id, missing, start_ts=start_ts, end_ts=end_ts)
    if preview_df.empty:
        return data_for_plot, []

    tmp = copy.copy(data_for_plot)
    tmp.event_data = pd.concat([base_events, preview_df], ignore_index=True)
    if "datetime" in tmp.event_data.columns:
        tmp.event_data = tmp.event_data.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    return tmp, sorted(preview_df["key"].dropna().astype(str).unique().tolist(), key=lambda x: x.lower())


def _preview_segment_rows_df(preview_store):
    seg_df = pd.DataFrame((dict(preview_store or {}).get("segment_rows") or []))
    if seg_df.empty:
        return pd.DataFrame(columns=["key", "datetime", "end_datetime", "duration"])

    key_col = next((c for c in ["key", "event_key", "state_event_key", "state_key", "label_name"] if c in seg_df.columns), None)
    start_col = next((c for c in ["datetime", "start", "start_datetime", "start_time", "datetime_start"] if c in seg_df.columns), None)
    end_col = next((c for c in ["end_datetime", "end", "end_time", "datetime_end", "stop"] if c in seg_df.columns), None)
    if not key_col or not start_col or not end_col:
        return pd.DataFrame(columns=["key", "datetime", "end_datetime", "duration"])

    out = pd.DataFrame()
    out["key"] = seg_df[key_col].astype(str).str.strip()
    out["datetime"] = pd.to_datetime(seg_df[start_col], errors="coerce")
    out["end_datetime"] = pd.to_datetime(seg_df[end_col], errors="coerce")
    out = out.loc[out["key"].ne("") & out["datetime"].notna() & out["end_datetime"].notna()].copy()
    out = out.loc[out["end_datetime"] > out["datetime"]].copy()
    if out.empty:
        return pd.DataFrame(columns=["key", "datetime", "end_datetime", "duration"])
    out["duration"] = (out["end_datetime"] - out["datetime"]).dt.total_seconds()
    return out[["key", "datetime", "end_datetime", "duration"]].reset_index(drop=True)


def _enrich_preview_event_rows_with_segment_bounds(preview_store):
    preview_df = pd.DataFrame((dict(preview_store or {}).get("event_rows") or []))
    if preview_df.empty:
        return preview_df
    if "key" not in preview_df.columns or "datetime" not in preview_df.columns:
        return preview_df

    preview_df = preview_df.copy()
    preview_df["key"] = preview_df["key"].astype(str).str.strip()
    preview_df["datetime"] = pd.to_datetime(preview_df["datetime"], errors="coerce")
    preview_df = preview_df.loc[preview_df["key"].ne("") & preview_df["datetime"].notna()].copy()
    if preview_df.empty:
        return preview_df

    if "end_datetime" in preview_df.columns:
        preview_df["end_datetime"] = pd.to_datetime(preview_df["end_datetime"], errors="coerce")
    else:
        preview_df["end_datetime"] = pd.NaT

    if "duration" in preview_df.columns:
        preview_df["duration"] = pd.to_numeric(preview_df["duration"], errors="coerce")
    else:
        preview_df["duration"] = np.nan

    seg_df = _preview_segment_rows_df(preview_store)
    if not seg_df.empty:
        merged = preview_df.merge(
            seg_df,
            how="left",
            on=["key", "datetime"],
            suffixes=("", "_seg"),
        )
        fill_end = merged["end_datetime"].isna() & merged["end_datetime_seg"].notna()
        merged.loc[fill_end, "end_datetime"] = merged.loc[fill_end, "end_datetime_seg"]
        fill_dur = merged["duration"].isna() & merged["duration_seg"].notna()
        merged.loc[fill_dur, "duration"] = merged.loc[fill_dur, "duration_seg"]
        preview_df = merged.drop(columns=["end_datetime_seg", "duration_seg"], errors="ignore")

    missing_dur = preview_df["duration"].isna() & preview_df["end_datetime"].notna()
    if missing_dur.any():
        preview_df.loc[missing_dur, "duration"] = (
            preview_df.loc[missing_dur, "end_datetime"] - preview_df.loc[missing_dur, "datetime"]
        ).dt.total_seconds()

    has_bounds = preview_df["duration"].fillna(0).gt(0) | preview_df["end_datetime"].notna()
    if has_bounds.any() and "type" not in preview_df.columns:
        preview_df["type"] = np.where(has_bounds, "state", "point")
    elif "type" in preview_df.columns:
        preview_df.loc[has_bounds, "type"] = preview_df.loc[has_bounds, "type"].fillna("state")
    return preview_df


def _preview_state_keys(preview_store):
    keys = set()
    preview_df = _enrich_preview_event_rows_with_segment_bounds(preview_store)
    if isinstance(preview_df, pd.DataFrame) and not preview_df.empty and "key" in preview_df.columns:
        duration = pd.to_numeric(preview_df.get("duration"), errors="coerce") if "duration" in preview_df.columns else None
        end_dt = pd.to_datetime(preview_df.get("end_datetime"), errors="coerce") if "end_datetime" in preview_df.columns else None
        has_state = pd.Series([False] * len(preview_df))
        if duration is not None:
            has_state = has_state | duration.fillna(0).gt(0)
        if end_dt is not None:
            has_state = has_state | end_dt.notna()
        keys.update(preview_df.loc[has_state, "key"].astype(str).tolist())
    seg_df = _preview_segment_rows_df(preview_store)
    if not seg_df.empty:
        keys.update(seg_df["key"].astype(str).tolist())
    return set(str(k).strip() for k in keys if str(k).strip())


def _segmentation_algo_cfg_from_state(
    source_channel,
    standardize,
    quantize_step,
    base_smooth_seconds,
    coarse_smooth_seconds,
    coarse_interval_threshold_s,
    coarse_resolution_threshold,
    sign_consistency_required,
    filter_out_positive_slopes,
    filter_out_negative_slopes,
    method_name,
    event_key_initial,
    event_key_filtered,
    event_key_surface_sleep,
    event_key_long_flat,
    event_key_long_drift,
    label_code_not_sleep,
    label_code_surface_sleep,
    label_code_long_flat,
    label_code_long_drift,
    label_name_not_sleep,
    label_name_surface_sleep,
    label_name_long_flat,
    label_name_long_drift,
    dive_depth_min_m,
    first_deriv_abs_max_ms,
    second_deriv_abs_max_ms2,
    min_duration_s,
    surface_sleep_min_duration_s,
    drift_rate_abs_max_ms,
    curvature_abs_max_ms,
    end_flat_abs_mean_d1_max_ms,
):
    return {
        "enabled": True,
        "method": "depth_drift_thresholds",
        "method_name": str(method_name or "find_rest").strip() or "find_rest",
        "source_channel": str(source_channel or "depth.depth").strip() or "depth.depth",
        "standardize": bool(standardize),
        "quantize_step": float(quantize_step),
        "base_smooth_seconds": float(base_smooth_seconds),
        "coarse_smooth_seconds": float(coarse_smooth_seconds),
        "coarse_interval_threshold_s": float(coarse_interval_threshold_s),
        "coarse_resolution_threshold": float(coarse_resolution_threshold),
        "sign_consistency_required": bool(sign_consistency_required),
        "filter_out_positive_slopes": bool(filter_out_positive_slopes),
        "filter_out_negative_slopes": bool(filter_out_negative_slopes),
        "event_keys": {
            "initial": str(event_key_initial or "").strip(),
            "filtered": str(event_key_filtered or "").strip(),
            "surface_sleep": str(event_key_surface_sleep or "").strip(),
            "long_flat": str(event_key_long_flat or "").strip(),
            "long_drift": str(event_key_long_drift or "").strip(),
        },
        "label_codes": {
            "not_sleep": int(label_code_not_sleep),
            "surface_sleep": int(label_code_surface_sleep),
            "long_flat": int(label_code_long_flat),
            "long_drift": int(label_code_long_drift),
        },
        "label_names": {
            "not_sleep": str(label_name_not_sleep or "").strip(),
            "surface_sleep": str(label_name_surface_sleep or "").strip(),
            "long_flat": str(label_name_long_flat or "").strip(),
            "long_drift": str(label_name_long_drift or "").strip(),
        },
        "thresholds": {
            "dive_depth_min_m": float(dive_depth_min_m),
            "first_deriv_abs_max_ms": float(first_deriv_abs_max_ms),
            "second_deriv_abs_max_ms2": float(second_deriv_abs_max_ms2),
            "min_duration_s": float(min_duration_s),
            "surface_sleep_min_duration_s": float(surface_sleep_min_duration_s),
            "drift_rate_abs_max_ms": float(drift_rate_abs_max_ms),
            "curvature_abs_max_ms": float(curvature_abs_max_ms),
            "end_flat_abs_mean_d1_max_ms": float(end_flat_abs_mean_d1_max_ms),
        },
    }


def _segmentation_event_key_options_from_cfg(algo_cfg):
    keys = list(((algo_cfg or {}).get("event_keys") or {}).values())
    return [str(k) for k in keys if str(k).strip()]


def _segmentation_style_entry(event_key, style_store=None, index=0):
    event_key = str(event_key or "")
    saved = dict((style_store or {}).get(event_key) or {})
    if not saved:
        saved = dict((_event_styles_global or {}).get(event_key) or {})
    defaults = _default_event_style(event_key, index)
    color = (
        _normalize_hex_color(saved.get("color"))
        or _normalize_hex_color(_mapped_event_color(event_key))
        or defaults["color"]
    )
    shade_color = _normalize_hex_color(saved.get("shade_color")) or color
    symbol = str(saved.get("symbol") or defaults["symbol"])
    if symbol not in EVENT_SYMBOL_OPTIONS:
        symbol = "circle"
    shade_mode = _normalize_event_shade_mode(saved.get("shade_mode"), default="fill_trace_split")
    try:
        shade_opacity = float(saved.get("shade_opacity", 0.25) or 0.25)
    except Exception:
        shade_opacity = 0.25
    return {
        "color": color,
        "shade_color": shade_color,
        "symbol": symbol,
        "shade_enabled": bool(saved.get("shade_enabled", True)),
        "shade_mode": shade_mode,
        "shade_opacity": max(0.0, min(1.0, shade_opacity)),
    }


def _workflow_editor_node_options(workflow):
    opts = []
    for node in list((workflow or {}).get("nodes") or []):
        nid = str((node or {}).get("id") or "").strip()
        if not nid:
            continue
        label = str((node or {}).get("label") or nid)
        ntype = str((node or {}).get("type") or "")
        opts.append({"label": f"{label} [{ntype}]", "value": nid})
    return opts


def _safe_workflow_name(value, fallback="workflow"):
    slug = "".join(c if c.isalnum() or c == "_" else "_" for c in str(value or "").strip().lower())
    slug = slug.strip("_")
    return slug or fallback


def _workflow_event_keys(workflow):
    runtime = workflow_preset_to_algorithmic_cfg(workflow, fallback_cfg=_segmentation_algo_defaults)
    return _segmentation_event_key_options_from_cfg(runtime)


def _set_nested_path(payload, path, value):
    parts = [str(p) for p in str(path or "").split(".") if str(p).strip()]
    if not parts:
        return
    cur = payload
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _sync_runtime_params_from_node(workflow, node):
    workflow = copy.deepcopy(workflow or {})
    node = dict(node or {})
    runtime = workflow_preset_to_algorithmic_cfg(workflow, fallback_cfg=_segmentation_algo_defaults)
    binding_path = str(node.get("binding_path") or "").strip()
    secondary_binding = str(node.get("secondary_binding_path") or "").strip()
    threshold = _coerce_float_or_none(node.get("threshold"))
    if binding_path and threshold is not None:
        _set_nested_path(runtime, binding_path, float(threshold))
    secondary_threshold = _coerce_float_or_none(node.get("secondary_threshold"))
    if secondary_binding and secondary_threshold is not None:
        _set_nested_path(runtime, secondary_binding, float(secondary_threshold))
    ntype = str(node.get("type") or "")
    if ntype == "terminal_state":
        terminal_name = str(node.get("label") or "").strip().lower()
        event_keys = dict(runtime.get("event_keys") or {})
        label_codes = dict(runtime.get("label_codes") or {})
        label_names = dict(runtime.get("label_names") or {})
        if "surface" in terminal_name:
            key = "surface_sleep"
        elif "flat" in terminal_name:
            key = "long_flat"
        elif "drift" in terminal_name:
            key = "long_drift"
        else:
            key = None
        if key:
            if str(node.get("state_key") or "").strip():
                event_keys[key] = str(node.get("state_key") or "").strip()
            if node.get("label_code") is not None:
                label_codes[key] = int(node.get("label_code") or 0)
            if str(node.get("label_name") or "").strip():
                label_names[key] = str(node.get("label_name") or "").strip()
            runtime["event_keys"] = event_keys
            runtime["label_codes"] = label_codes
            runtime["label_names"] = label_names
    workflow["runtime_params"] = runtime
    return workflow


def _normalize_workflow_for_compare(workflow):
    workflow = copy.deepcopy(workflow or {})
    workflow.pop("display_name", None)
    workflow.pop("default_review_signals", None)
    return json.loads(json.dumps(workflow, sort_keys=True, default=str))


SEGMENTATION_WORKFLOW_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "text-justification": "center",
            "text-wrap": "wrap",
            "text-max-width": 176,
            "text-margin-y": 0,
            "font-size": 15,
            "font-weight": 700,
            "line-height": 1.15,
            "width": 196,
            "height": 96,
            "shape": "round-rectangle",
            "color": "#FFFFFF",
            "border-color": "#DDEAF7",
            "border-width": 2,
            "background-color": "#7EA8E0",
        },
    },
    {"selector": ".seg-node-filter_group", "style": {"background-color": "#6D8ACF"}},
    {"selector": ".seg-node-decision", "style": {"background-color": "#7EA8E0"}},
    {"selector": ".seg-node-derive", "style": {"background-color": "#58A4B0"}},
    {"selector": ".seg-node-terminal_state", "style": {"background-color": "#294F91"}},
    {"selector": ".seg-node-terminal_reject", "style": {"background-color": "#888888"}},
    {
        "selector": ".seg-node-junction",
        "style": {
            "background-opacity": 0,
            "border-opacity": 0,
            "width": 8,
            "height": 8,
            "label": "",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "unbundled-bezier",
            "control-point-distances": 52,
            "control-point-weights": 0.5,
            "target-arrow-shape": "triangle",
            "line-color": "#BFD7EA",
            "target-arrow-color": "#BFD7EA",
            "source-distance-from-node": 18,
            "target-distance-from-node": 18,
            "label": "data(label)",
            "font-size": 12,
            "font-weight": 700,
            "color": "#FFFFFF",
            "text-background-color": "#3D5E84",
            "text-background-opacity": 1,
            "text-background-padding": 3,
            "text-border-color": "rgba(221,234,247,0.55)",
            "text-border-width": 1,
            "text-border-opacity": 0.55,
            "edge-text-rotation": "none",
            "text-margin-y": -14,
        },
    },
    {"selector": ".seg-edge-yes", "style": {"text-background-color": "#2E8B57"}},
    {"selector": ".seg-edge-no", "style": {"text-background-color": "#9B3D3D"}},
    {"selector": ".seg-edge-next", "style": {"text-background-color": "#466A8C"}},
    {"selector": ".seg-edge-surface", "style": {"text-background-color": "#3A7CA5"}},
    {"selector": ".seg-edge-drift", "style": {"text-background-color": "#7A5CA8"}},
]


def _build_segmentation_page_layout(dataset_options, default_dataset, default_deployment, algo_defaults, workflow_defaults):
    deployments = _list_deployments(data_dir, default_dataset)
    event_keys = _segmentation_event_key_options_from_cfg(algo_defaults)
    workflow_options = segmentation_workflow_options(workflow_defaults)
    default_workflow_name = workflow_options[0]["value"] if workflow_options else "find_rest"
    default_workflow = copy.deepcopy((workflow_defaults or {}).get(default_workflow_name) or build_find_rest_workflow_preset(algo_defaults))
    default_node_options = _workflow_editor_node_options(default_workflow)
    default_node_value = default_node_options[0]["value"] if default_node_options else None
    control_style = {"width": "100%", "maxWidth": "100%", "minWidth": 0}
    input_style = {"width": "100%", "maxWidth": "100%", "minWidth": 0, "boxSizing": "border-box"}
    sidebar_style = {
        "maxHeight": "calc(100vh - 96px)",
        "overflowY": "auto",
        "overflowX": "hidden",
        "minWidth": 0,
        "paddingRight": "6px",
    }
    workflow_sidebar_style = {
        "position": "fixed",
        "top": "72px",
        "right": "12px",
        "width": "420px",
        "maxWidth": "calc(100vw - 24px)",
        "maxHeight": "calc(100vh - 84px)",
        "overflowY": "auto",
        "overflowX": "hidden",
        "padding": "10px",
        "background": "#081B2A",
        "border": "1px solid rgba(255,255,255,0.12)",
        "borderRadius": "12px",
        "boxShadow": "0 16px 48px rgba(0,0,0,0.35)",
        "zIndex": 1200,
        "display": "none",
    }
    actions_style = {"display": "flex", "flexWrap": "wrap", "gap": "8px"}
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Details(
                                [
                                    html.Summary([html.Span("Deployment Selection", className="section-title")], className="collapsible-summary"),
                                    html.Div(
                                        [
                                            html.Label("Dataset"),
                                            dcc.Dropdown(
                                                id="seg-dataset-select",
                                                options=[{"label": d, "value": d} for d in dataset_options],
                                                value=default_dataset,
                                                clearable=False,
                                                style=control_style,
                                            ),
                                            html.Label("Deployment", style={"marginTop": "10px"}),
                                            dcc.Dropdown(
                                                id="seg-deployment-select",
                                                options=[{"label": d, "value": d} for d in deployments],
                                                value=default_deployment,
                                                clearable=False,
                                                style=control_style,
                                            ),
                                            html.Button("Apply", id="seg-apply-deployment", n_clicks=0, className="apply-btn"),
                                            html.Div(id="seg-deployment-status", className="window-summary"),
                                        ],
                                        className="collapsible-body",
                                    ),
                                ],
                                className="collapsible-panel",
                                open=True,
                            ),
                            html.Details(
                                [
                                    html.Summary([html.Span("Signal Selection", className="section-title")], className="collapsible-summary"),
                                    html.Div(
                                        [
                                            html.Label("Signals"),
                                            dcc.Dropdown(id="seg-signals-select", options=[], value=[], multi=True, style=control_style),
                                            html.Div(
                                                [
                                                    html.Button("Save Signals for Review", id="seg-save-signals-btn", n_clicks=0, className="apply-btn"),
                                                ],
                                                className="color-editor-actions",
                                                style=actions_style,
                                            ),
                                            html.Div(id="seg-save-signals-status", className="window-summary"),
                                            html.Div(id="seg-time-summary", className="window-summary", style={"marginTop": "10px"}),
                                        ],
                                        className="collapsible-body",
                                    ),
                                ],
                                className="collapsible-panel",
                                open=True,
                            ),
                            html.Details(
                                [
                                    html.Summary([html.Span("State Event Styling", className="section-title")], className="collapsible-summary"),
                                    html.Div(
                                        [
                                            html.Div(id="seg-state-events-panel"),
                                            html.Div(
                                                "Select parent groups with checkmarks, then use Save state events to deployment to write selected groups.",
                                                className="window-summary",
                                                style={"marginTop": "8px"},
                                            ),
                                            html.Div(
                                                "Click a state event row to edit its style in the popup editor.",
                                                className="window-summary",
                                                style={"marginTop": "8px"},
                                            ),
                                            dcc.Dropdown(
                                                id="seg-style-event-key",
                                                options=[{"label": k, "value": k} for k in event_keys],
                                                value=(event_keys[0] if event_keys else None),
                                                clearable=False,
                                                style={"display": "none"},
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Input(id="seg-style-color", type="text", value="#4DAEF8", className="peak-param-input", style=input_style),
                                                    dcc.Dropdown(
                                                        id="seg-style-shade-mode",
                                                        options=[
                                                            {"label": "fill trace", "value": "fill_trace"},
                                                            {"label": "fill trace split", "value": "fill_trace_split"},
                                                            {"label": "fill trace above", "value": "fill_trace_above"},
                                                        ],
                                                        value="fill_trace_split",
                                                        clearable=False,
                                                        style=control_style,
                                                    ),
                                                    dcc.Input(id="seg-style-opacity", type="number", min=0, max=1, step=0.05, value=0.25, className="peak-param-input", style=input_style),
                                                    html.Button("Apply Style", id="seg-apply-style-btn", n_clicks=0, className="apply-btn"),
                                                    html.Button("Open Color Picker", id="seg-open-color-editor-btn", n_clicks=0, className="apply-btn"),
                                                ],
                                                style={"display": "none"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Button("Save State Event Styles", id="seg-save-style-btn", n_clicks=0, className="apply-btn"),
                                                ],
                                                className="color-editor-actions",
                                                style=actions_style,
                                            ),
                                            html.Div(id="seg-style-status", className="window-summary"),
                                        ],
                                        className="collapsible-body",
                                    ),
                                ],
                                className="collapsible-panel",
                                open=True,
                            ),
                            html.Details(
                                [
                                    html.Summary([html.Span("Timeseries Segmentation Tool", className="section-title")], className="collapsible-summary"),
                                    html.Div(
                                        [
                                            html.Label("Workflow"),
                                            dcc.Dropdown(
                                                id="seg-workflow-select",
                                                options=workflow_options,
                                                value=default_workflow_name,
                                                clearable=False,
                                                style=control_style,
                                            ),
                                            html.Div(
                                                [
                                                    html.Button("New Workflow From Current", id="seg-new-workflow-btn", n_clicks=0, className="apply-btn"),
                                                    html.Button("Duplicate Workflow", id="seg-duplicate-workflow-btn", n_clicks=0, className="apply-btn"),
                                                    html.Button("Reset To Saved", id="seg-reset-workflow-btn", n_clicks=0, className="apply-btn"),
                                                    html.Button("Validate Workflow", id="seg-validate-workflow-btn", n_clicks=0, className="apply-btn"),
                                                    html.Button("Save Workflow Preset", id="seg-save-workflow-btn", n_clicks=0, className="apply-btn"),
                                                    html.Button("Open Workflow Editor", id="seg-workflow-sidebar-btn", n_clicks=0, className="apply-btn"),
                                                ],
                                                className="color-editor-actions",
                                                style=actions_style,
                                            ),
                                            html.Div(id="seg-workflow-status", className="window-summary"),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "Decision tree editor is shown in the workflow sidebar to the right.",
                                                        className="window-summary",
                                                    ),
                                                ]
                                            ),
                                            dcc.Tabs(
                                                id="seg-method-tabs",
                                                value="algorithmic",
                                                children=[
                                                    dcc.Tab(
                                                        label="Algorithmic",
                                                        value="algorithmic",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    html.Label("Source Channel"),
                                                                    dcc.Input(id="seg-source-channel", type="text", value=algo_defaults.get("source_channel", "depth.depth"), className="peak-param-input", style=input_style),
                                                                    html.Div(
                                                                        [
                                                                            dcc.Checklist(id="seg-standardize", options=[{"label": "standardize", "value": "on"}], value=(["on"] if algo_defaults.get("standardize", True) else [])),
                                                                            dcc.Checklist(id="seg-sign-consistency", options=[{"label": "sign consistency required", "value": "on"}], value=(["on"] if algo_defaults.get("sign_consistency_required", True) else [])),
                                                                            dcc.Checklist(id="seg-filter-positive-slopes", options=[{"label": "filter out positive slopes", "value": "on"}], value=(["on"] if algo_defaults.get("filter_out_positive_slopes", True) else [])),
                                                                            dcc.Checklist(id="seg-filter-negative-slopes", options=[{"label": "filter out negative slopes", "value": "on"}], value=(["on"] if algo_defaults.get("filter_out_negative_slopes", False) else [])),
                                                                        ],
                                                                        className="color-editor-actions",
                                                                        style=actions_style,
                                                                    ),
                                                                    html.Label("Method Name"),
                                                                    dcc.Input(id="seg-method-name", type="text", value=algo_defaults.get("method_name", "find_rest"), className="peak-param-input", style=input_style),
                                                                    html.Label("Quantize Step"),
                                                                    dcc.Input(id="seg-quantize-step", type="number", value=algo_defaults.get("quantize_step", 1.0), step=0.1, className="peak-param-input", style=input_style),
                                                                    html.Label("Base Smooth Seconds"),
                                                                    dcc.Input(id="seg-base-smooth-seconds", type="number", value=algo_defaults.get("base_smooth_seconds", 6.0), step=0.5, className="peak-param-input", style=input_style),
                                                                    html.Label("Coarse Smooth Seconds"),
                                                                    dcc.Input(id="seg-coarse-smooth-seconds", type="number", value=algo_defaults.get("coarse_smooth_seconds", 12.0), step=0.5, className="peak-param-input", style=input_style),
                                                                    html.Label("Coarse Interval Threshold (s)"),
                                                                    dcc.Input(id="seg-coarse-interval-threshold", type="number", value=algo_defaults.get("coarse_interval_threshold_s", 5.0), step=0.5, className="peak-param-input", style=input_style),
                                                                    html.Label("Coarse Resolution Threshold"),
                                                                    dcc.Input(id="seg-coarse-resolution-threshold", type="number", value=algo_defaults.get("coarse_resolution_threshold", 1.0), step=0.1, className="peak-param-input", style=input_style),
                                                                    html.Hr(),
                                                                    html.Label("Dive Depth Minimum (m)"),
                                                                    dcc.Input(id="seg-thr-dive-depth", type="number", value=algo_defaults["thresholds"].get("dive_depth_min_m", 2.0), step=0.1, className="peak-param-input", style=input_style),
                                                                    html.Label("First Derivative Max (m/s)"),
                                                                    dcc.Input(id="seg-thr-d1", type="number", value=algo_defaults["thresholds"].get("first_deriv_abs_max_ms", 0.6), step=0.01, className="peak-param-input", style=input_style),
                                                                    html.Label("Second Derivative Max (m/s²)"),
                                                                    dcc.Input(id="seg-thr-d2", type="number", value=algo_defaults["thresholds"].get("second_deriv_abs_max_ms2", 0.05), step=0.01, className="peak-param-input", style=input_style),
                                                                    html.Label("Minimum Duration (s)"),
                                                                    dcc.Input(id="seg-thr-min-duration", type="number", value=algo_defaults["thresholds"].get("min_duration_s", 180.0), step=1, className="peak-param-input", style=input_style),
                                                                    html.Label("Surface Sleep Minimum Duration (s)"),
                                                                    dcc.Input(id="seg-thr-surface-duration", type="number", value=algo_defaults["thresholds"].get("surface_sleep_min_duration_s", 600.0), step=1, className="peak-param-input", style=input_style),
                                                                    html.Label("Drift Rate Maximum (m/s)"),
                                                                    dcc.Input(id="seg-thr-drift-rate", type="number", value=algo_defaults["thresholds"].get("drift_rate_abs_max_ms", 0.30), step=0.01, className="peak-param-input", style=input_style),
                                                                    html.Label("Curvature Maximum"),
                                                                    dcc.Input(id="seg-thr-curvature", type="number", value=algo_defaults["thresholds"].get("curvature_abs_max_ms", 0.10), step=0.01, className="peak-param-input", style=input_style),
                                                                    html.Label("End Flat Mean d1 Maximum"),
                                                                    dcc.Input(id="seg-thr-end-flat", type="number", value=algo_defaults["thresholds"].get("end_flat_abs_mean_d1_max_ms", 0.01), step=0.001, className="peak-param-input", style=input_style),
                                                                    html.Hr(),
                                                                    html.Label("Event Keys"),
                                                                    dcc.Input(id="seg-event-key-initial", type="text", value=algo_defaults["event_keys"].get("initial"), className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-event-key-filtered", type="text", value=algo_defaults["event_keys"].get("filtered"), className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-event-key-surface-sleep", type="text", value=algo_defaults["event_keys"].get("surface_sleep"), className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-event-key-long-flat", type="text", value=algo_defaults["event_keys"].get("long_flat"), className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-event-key-long-drift", type="text", value=algo_defaults["event_keys"].get("long_drift"), className="peak-param-input", style=input_style),
                                                                    html.Label("Label Codes"),
                                                                    dcc.Input(id="seg-label-code-not-sleep", type="number", value=algo_defaults["label_codes"].get("not_sleep", 0), step=1, className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-label-code-surface-sleep", type="number", value=algo_defaults["label_codes"].get("surface_sleep", 1), step=1, className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-label-code-long-flat", type="number", value=algo_defaults["label_codes"].get("long_flat", 2), step=1, className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-label-code-long-drift", type="number", value=algo_defaults["label_codes"].get("long_drift", 3), step=1, className="peak-param-input", style=input_style),
                                                                    html.Label("Label Names"),
                                                                    dcc.Input(id="seg-label-name-not-sleep", type="text", value=algo_defaults["label_names"].get("not_sleep"), className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-label-name-surface-sleep", type="text", value=algo_defaults["label_names"].get("surface_sleep"), className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-label-name-long-flat", type="text", value=algo_defaults["label_names"].get("long_flat"), className="peak-param-input", style=input_style),
                                                                    dcc.Input(id="seg-label-name-long-drift", type="text", value=algo_defaults["label_names"].get("long_drift"), className="peak-param-input", style=input_style),
                                                                ],
                                                                className="collapsible-body",
                                                            )
                                                        ],
                                                    ),
                                                    dcc.Tab(
                                                        label="Unsupervised",
                                                        value="unsupervised",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    html.Label("Run Name"),
                                                                    dcc.Input(type="text", value="cluster_run_placeholder", disabled=True, className="peak-param-input", style=input_style),
                                                                    html.Label("Feature Channels"),
                                                                    dcc.Dropdown(options=[], value=[], multi=True, disabled=True, style=control_style),
                                                                    html.Div("Unsupervised segmentation is scaffolded but not executable in this page yet.", className="window-summary"),
                                                                ],
                                                                className="collapsible-body",
                                                            )
                                                        ],
                                                    ),
                                                    dcc.Tab(
                                                        label="Supervised",
                                                        value="supervised",
                                                        children=[
                                                            html.Div(
                                                                [
                                                                    html.Label("Training Label Channel"),
                                                                    dcc.Input(type="text", value="algorithmic_labels", disabled=True, className="peak-param-input", style=input_style),
                                                                    html.Label("Classifier"),
                                                                    dcc.Dropdown(options=[{"label": "Random Forest", "value": "rf"}], value="rf", disabled=True, style=control_style),
                                                                    html.Div("Supervised segmentation is scaffolded but not executable in this page yet.", className="window-summary"),
                                                                ],
                                                                className="collapsible-body",
                                                            )
                                                        ],
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                [
                                                    html.Button("Run State Event Detection", id="seg-preview-btn", n_clicks=0, className="apply-btn"),
                                                    html.Button("Calculate state events for rest of deployment", id="seg-full-btn", n_clicks=0, className="apply-btn", disabled=True),
                                                    html.Button("Save selected state groups to deployment", id="seg-save-btn", n_clicks=0, className="apply-btn", disabled=True),
                                                ],
                                                className="color-editor-actions",
                                                style=actions_style,
                                            ),
                                            html.Div(id="seg-run-status", className="window-summary"),
                                        ],
                                        className="collapsible-body",
                                    ),
                                ],
                                className="collapsible-panel",
                                open=True,
                            ),
                        ],
                        className="custom-legend-panel",
                        style=sidebar_style,
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="seg-plot", config={"displayModeBar": True}, style={"height": "78vh"}),
                            html.Div(
                                [
                                    html.Label("Time Range"),
                                    dcc.RangeSlider(id="seg-time-range-slider", min=0, max=1, value=[0, 1], allowCross=False),
                                ],
                                className="control-card",
                            ),
                        ],
                        className="segmentation-main-stage",
                        style={"minWidth": 0},
                    ),
                ],
                className="segmentation-workspace",
                style={"display": "grid", "gridTemplateColumns": "380px minmax(0, 1fr)", "gap": "12px", "alignItems": "start", "minWidth": 0},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Workflow Editor", className="section-title"),
                            html.Button("Close", id="seg-workflow-sidebar-close-btn", n_clicks=0, className="apply-btn"),
                        ],
                        className="color-editor-actions",
                        style={"justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Workflow Diagram", className="section-title"),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Move",
                                                id="seg-workflow-diagram-drag-handle",
                                                n_clicks=0,
                                                className="panel-popout-drag-handle",
                                                title="Move",
                                            ),
                                            html.Button(
                                                "Pop out",
                                                id="seg-workflow-diagram-popout-btn",
                                                n_clicks=0,
                                                className="apply-btn pip-toggle-btn",
                                            ),
                                        ],
                                        className="color-editor-actions",
                                        style={"justifyContent": "flex-end", "alignItems": "center"},
                                    ),
                                ],
                                className="card-title-row",
                                style={"display": "flex", "justifyContent": "space-between", "gap": "8px", "alignItems": "center", "marginBottom": "8px"},
                            ),
                            (
                                html.Div(
                                    [
                                        cyto.Cytoscape(
                                            id="seg-workflow-cyto",
                                            elements=segmentation_workflow_to_cytoscape_elements(default_workflow),
                                            stylesheet=SEGMENTATION_WORKFLOW_STYLESHEET,
                                            style={"width": "100%", "height": "100%", "backgroundColor": "#0B2236"},
                                            layout={"name": "preset", "fit": True, "padding": 40},
                                            autoRefreshLayout=False,
                                            userZoomingEnabled=True,
                                            wheelSensitivity=1,
                                            userPanningEnabled=True,
                                            boxSelectionEnabled=False,
                                            minZoom=0.4,
                                            maxZoom=2.2,
                                        )
                                        ,
                                        html.Button(
                                            "Save Node Positions",
                                            id="seg-save-node-positions-btn",
                                            n_clicks=0,
                                            className="apply-btn",
                                            style={
                                                "position": "absolute",
                                                "right": "10px",
                                                "bottom": "10px",
                                                "zIndex": 10,
                                                "boxShadow": "0 6px 18px rgba(0,0,0,0.35)",
                                            },
                                        ),
                                    ],
                                    className="seg-workflow-cyto-wrap",
                                    style={"width": "100%", "height": "620px", "minHeight": "320px", "position": "relative"},
                                )
                                if CYTO_AVAILABLE
                                else html.Div(
                                    "dash-cytoscape not available. Install it to edit workflow graphs.",
                                    id="seg-workflow-cyto",
                                    className="legend-empty",
                                )
                            ),
                        ],
                        id="seg-workflow-diagram-card",
                        className="control-card aux-card seg-workflow-diagram-card",
                    ),
                    html.Div(
                        [
                            html.Div("Node Editor", className="section-title", style={"marginBottom": "8px"}),
                            html.Label("Selected Node"),
                            dcc.Dropdown(
                                id="seg-selected-node",
                                options=default_node_options,
                                value=default_node_value,
                                clearable=False,
                                style=control_style,
                            ),
                            html.Label("Node Label", style={"marginTop": "8px"}),
                            dcc.Input(id="seg-node-label", type="text", value="", className="peak-param-input", style=input_style),
                            html.Label("Node Type", style={"marginTop": "8px"}),
                            dcc.Dropdown(
                                id="seg-node-type",
                                options=[{"label": t, "value": t} for t in sorted(SEGMENTATION_WORKFLOW_ALLOWED_NODE_TYPES)],
                                value=None,
                                clearable=False,
                                style=control_style,
                            ),
                            html.Label("Description", style={"marginTop": "8px"}),
                            dcc.Input(id="seg-node-description", type="text", value="", className="peak-param-input", style=input_style),
                            html.Label("Channel", style={"marginTop": "8px"}),
                            dcc.Input(id="seg-node-channel", type="text", value="", className="peak-param-input", style=input_style),
                            html.Label("Operator", style={"marginTop": "8px"}),
                            dcc.Dropdown(
                                id="seg-node-operator",
                                options=[{"label": v, "value": v} for v in ["<", "<=", ">", ">=", "between", "positive", "negative", "custom"]],
                                value="<=",
                                clearable=False,
                                style=control_style,
                            ),
                            html.Label("Threshold", style={"marginTop": "8px"}),
                            dcc.Input(id="seg-node-threshold", type="number", value=0, className="peak-param-input", style=input_style),
                            html.Label("State Key", style={"marginTop": "8px"}),
                            dcc.Input(id="seg-node-state-key", type="text", value="", className="peak-param-input", style=input_style),
                            html.Label("Label Code", style={"marginTop": "8px"}),
                            dcc.Input(id="seg-node-label-code", type="number", value=0, className="peak-param-input", style=input_style),
                            html.Label("Label Name", style={"marginTop": "8px"}),
                            dcc.Input(id="seg-node-label-name", type="text", value="", className="peak-param-input", style=input_style),
                            html.Label("Add Child Branch", style={"marginTop": "8px"}),
                            dcc.Dropdown(
                                id="seg-add-node-branch",
                                options=[{"label": v, "value": v} for v in ["yes", "no", "next", "surface", "drift"]],
                                value="yes",
                                clearable=False,
                                style=control_style,
                            ),
                            html.Div(
                                [
                                    html.Button("Apply Node Changes", id="seg-apply-node-btn", n_clicks=0, className="apply-btn"),
                                    html.Button("Add Decision", id="seg-add-decision-btn", n_clicks=0, className="apply-btn"),
                                    html.Button("Add Filter Group", id="seg-add-filter-group-btn", n_clicks=0, className="apply-btn"),
                                    html.Button("Add Terminal State", id="seg-add-terminal-btn", n_clicks=0, className="apply-btn"),
                                    html.Button("Add Reject", id="seg-add-reject-btn", n_clicks=0, className="apply-btn"),
                                    html.Button("Remove Selected", id="seg-remove-node-btn", n_clicks=0, className="apply-btn"),
                                ],
                                className="color-editor-actions",
                                style=actions_style,
                            ),
                        ],
                        id="seg-node-editor-card",
                        className="control-card aux-card seg-node-editor-card",
                        style={"marginTop": "12px"},
                    ),
                ],
                id="seg-workflow-sidebar",
                className="custom-legend-panel seg-workflow-sidebar",
                style=workflow_sidebar_style,
            ),
            dcc.Store(id="seg-preview-store", data={}),
            dcc.Store(id="seg-event-style-store", data=_event_styles_global),
            dcc.Store(id="seg-event-group-selection-store", data={}),
            dcc.Store(id="seg-workflow-store", data=workflow_defaults),
            dcc.Store(id="seg-workflow-draft", data=default_workflow),
            dcc.Store(id="seg-workflow-live-elements", data=[]),
        ],
        className="integrated-app",
    )


def _playhead_shape_dict(playhead_ts):
    x_val = pd.Timestamp(playhead_ts).isoformat()
    return {
        "type": "line",
        "x0": x_val,
        "x1": x_val,
        "y0": 0,
        "y1": 1,
        "xref": "x",
        "yref": "paper",
        "line": {"color": "#FFD166", "width": 2},
        "opacity": 0.95,
        "editable": True,
    }


parser = argparse.ArgumentParser(description="Integrated Dash interactive plot with signal/channel ordering.")
parser.add_argument("--dataset", type=str, default=None, help="Dataset folder name (defaults to the configured default deployment)")
parser.add_argument("--deployment", type=str, default=None, help="Deployment ID (defaults to the configured default deployment)")
parser.add_argument("--port", type=int, default=8061, help="Dash server port")
args = parser.parse_args()

config, data_dir, color_mapping_path, _ = load_configuration()


def _resolve_launch_target(data_dir_path, dataset_arg, deployment_arg):
    """Pick the dataset/deployment to open at launch so no args are required.

    Precedence: explicit CLI args -> configured DEFAULT_SEGMENTATION_* -> the
    first dataset on disk with a deployment that has outputs/data.pkl. Enables
    ``python integrated_dash.py`` with no flags.
    """
    ds = dataset_arg
    dep = deployment_arg
    # Configured default dataset, if present on disk.
    if ds is None and DEFAULT_SEGMENTATION_DATASET in _list_datasets(data_dir_path):
        ds = DEFAULT_SEGMENTATION_DATASET
    # Configured default deployment within the chosen dataset, if present.
    if ds is not None and dep is None:
        deps = _list_deployments(data_dir_path, ds)
        if DEFAULT_SEGMENTATION_DEPLOYMENT in deps:
            dep = DEFAULT_SEGMENTATION_DEPLOYMENT
        elif deps:
            dep = deps[0]
    # Last resort: first dataset anywhere that has a usable deployment.
    if ds is None or dep is None:
        for candidate_ds in _list_datasets(data_dir_path):
            deps = _list_deployments(data_dir_path, candidate_ds)
            if deps:
                ds, dep = candidate_ds, deps[0]
                break
    if ds is None or dep is None:
        raise SystemExit(
            "No dataset/deployment specified and no default could be resolved. "
            "Pass --dataset and --deployment, or ensure a deployment with "
            "outputs/data.pkl exists under the configured data directory."
        )
    return ds, dep


args.dataset, args.deployment = _resolve_launch_target(data_dir, args.dataset, args.deployment)
print(f"[launch] Opening dataset={args.dataset} deployment={args.deployment}")
_private_root = (config.get("paths", {}) or {}).get("local_private_data") or data_dir
_PEAK_REF_VALUES = _collect_peak_reference_values(config, _private_root)
_seg_config_path = str(pathlib.Path(_LOCAL_PYOLOGGER_ROOT) / "config.yaml")
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(
    data_dir, dataset_id=args.dataset, deployment_id=args.deployment
)
deployment_source = resolve_deployment_source(
    data_dir,
    dataset_id,
    deployment_id,
    deployment_folder=deployment_folder,
)
plot_signal_allowlist = resolve_plot_signal_allowlist(param_manager, deployment_source)
data_pkl = deployment_source.build_metadata_shell(allowed_signals=plot_signal_allowlist)

tz_name = str(data_pkl.deployment_info.get("Time Zone", "UTC") or "UTC")
deployment_tz_name = tz_name
time_display_tz_mode = "local"
std = standardize_time_settings(param_manager=param_manager, data_pkl=data_pkl, tz_name=tz_name, minutes=10, persist=False)
view_ctx = _compute_view_context(
    data_pkl,
    tz_name,
    std["zoom_window_start_time"],
    std["zoom_window_end_time"],
    param_manager_obj=param_manager,
)
global_start = view_ctx["global_start"]
global_end = view_ctx["global_end"]
slider_min = view_ctx["slider_min"]
slider_max = view_ctx["slider_max"]
slider_default = view_ctx["slider_default"]
slider_marks = view_ctx["slider_marks"]
start_default = view_ctx["start_default"]
end_default = view_ctx["end_default"]
all_signals = _available_signal_names(data_pkl)
default_signals = view_ctx["default_signals"]
event_key_options = _available_event_keys(data_pkl)
default_events = [str(e) for e in (view_ctx.get("default_events") or event_key_options)]
point_event_defaults, state_event_defaults = _split_event_keys_by_state(data_pkl, default_events)
all_event_defaults = default_events

if not all_signals:
    raise ValueError("No plot-enabled signals available for this deployment")

dataset_options = _list_datasets(data_dir)
deployment_options = _list_deployments(data_dir, dataset_id)
seg_default_dataset, seg_default_deployment = _segmentation_default_scope(data_dir, dataset_id, deployment_id)
_segmentation_algo_defaults = default_algorithmic_cfg_from_config(_seg_config_path)
_segmentation_workflow_defaults = load_segmentation_workflow_presets(_seg_config_path)

app = Dash(__name__)
app.title = "Integrated Dash"


# ---------------------------------------------------------------------------
# Local synchronized-video support
# ---------------------------------------------------------------------------
# Directory of local processed video clips for the ACTIVE deployment. Filenames
# encode a local-time range, e.g. "..._2023-06-23_11-44-33_11-49-30.mp4".
# Derived from the deployment's media folder:
#   <local_private_media>/<dataset>/<deployment>_media/02_processed-video
# (override with PYOLOGGER_VIDEO_DIR). Must be deployment-specific — a fixed path
# would load another deployment's clips and land them years off the playhead.
def _default_video_dir(dataset, deployment):
    override = os.environ.get("PYOLOGGER_VIDEO_DIR")
    if override:
        return pathlib.Path(override).resolve()
    media_root = (config.get("paths", {}) or {}).get("local_private_media")
    if media_root and dataset and deployment:
        return (pathlib.Path(media_root) / dataset / f"{deployment}_media" / "02_processed-video").resolve()
    return None


VIDEO_DIR = _default_video_dir(dataset_id, deployment_id)

_VIDEO_FILENAME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<start>\d{2}-\d{2}-\d{2})_(?P<end>\d{2}-\d{2}-\d{2})\.mp4$",
    re.IGNORECASE,
)


def _build_video_clip_index(video_dir, clip_tz_name):
    """Scan video_dir and return clips sorted by start, each with epoch bounds.

    Each entry: {"name", "start_epoch", "end_epoch"}. Times in the filename are
    interpreted in the deployment-local timezone. Clips that cross midnight
    (end < start) roll the end date forward one day.
    """
    clips = []
    if not video_dir:
        return clips
    try:
        entries = sorted(pathlib.Path(video_dir).glob("*.mp4"))
    except Exception:
        entries = []
    for p in entries:
        m = _VIDEO_FILENAME_RE.search(p.name)
        if not m:
            continue
        try:
            date = m.group("date")
            start_ts = pd.Timestamp(f"{date} {m.group('start').replace('-', ':')}").tz_localize(clip_tz_name)
            end_ts = pd.Timestamp(f"{date} {m.group('end').replace('-', ':')}").tz_localize(clip_tz_name)
            if end_ts <= start_ts:
                end_ts = end_ts + pd.Timedelta(days=1)
        except Exception:
            continue
        clips.append(
            {
                "name": p.name,
                "start_epoch": float(start_ts.timestamp()),
                "end_epoch": float(end_ts.timestamp()),
                "source": "local",
                "url": f"/local-video/{p.name}",
            }
        )
    clips.sort(key=lambda c: c["start_epoch"])
    return clips


# ---------------------------------------------------------------------------
# Immich synchronized-video support (preferred source)
# ---------------------------------------------------------------------------
# Videos live in an Immich album named "DepID_<deployment_id>". Immich transcodes
# and serves web-optimized, seekable streams, which avoids local-file container/
# encoding issues. Credentials come from IMMICH_API_KEY / IMMICH_BASE_URL (set in
# the environment, e.g. via EcoPhysVideoViz/.env). The browser never sees the API
# key: it hits our /immich-video/<asset_id> proxy, which streams from Immich with
# the key header and forwards Range requests.
_immich_service = None


def _ensure_immich_env():
    """Populate IMMICH_* from a .env file if not already in the environment.

    Checks PYOLOGGER_IMMICH_ENV, then EcoPhysVideoViz/.env at the repo root. Only
    fills missing keys; existing environment values win.
    """
    if os.getenv("IMMICH_API_KEY") and os.getenv("IMMICH_BASE_URL"):
        return
    candidates = []
    explicit = os.getenv("PYOLOGGER_IMMICH_ENV")
    if explicit:
        candidates.append(pathlib.Path(explicit))
    candidates.append(pathlib.Path(_LOCAL_PYOLOGGER_ROOT).resolve().parent / "EcoPhysVideoViz" / ".env")
    for env_path in candidates:
        try:
            if not env_path.is_file():
                continue
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k in ("IMMICH_API_KEY", "IMMICH_BASE_URL") and not os.getenv(k):
                    os.environ[k] = v
        except Exception:
            continue


def _get_immich_service():
    global _immich_service
    if _immich_service is not None:
        return _immich_service
    try:
        _ensure_immich_env()
        from DiveDB.services.immich_service import ImmichService

        _immich_service = ImmichService()
    except Exception as exc:
        print(f"[video] Immich unavailable ({exc}); will use local files if present.")
        _immich_service = False
    return _immich_service


def _parse_immich_duration(duration_str):
    """'HH:MM:SS.mmm' -> seconds (float). 0.0 on failure."""
    try:
        parts = str(duration_str).split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        pass
    return 0.0


def _build_immich_clip_index(deployment_ident):
    """Build the clip index from the Immich album DepID_<deployment_ident>.

    Uses the RAW find_media 'fileCreatedAt' (correct UTC, offset-aware) rather than
    prepare_video_options_for_react (which relabels local time as UTC). Each clip:
    {name, asset_id, start_epoch, end_epoch, source='immich', url='/immich-video/<id>'}.
    Returns [] if Immich is unavailable or the album has no videos.
    """
    svc = _get_immich_service()
    if not svc:
        return []
    album = f"DepID_{deployment_ident}"
    try:
        res = svc.find_media_by_deployment_id(album, media_type="VIDEO", shared=True)
    except Exception as exc:
        print(f"[video] Immich find_media failed for {album}: {exc}")
        return []
    if not (res and res.get("success")):
        return []
    clips = []
    for a in (res.get("data") or []):
        try:
            aid = a.get("id")
            created = a.get("fileCreatedAt")
            if not aid or not created:
                continue
            start_ts = pd.Timestamp(created)  # ISO w/ offset -> tz-aware
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            dur = _parse_immich_duration(a.get("duration"))
            start_epoch = float(start_ts.timestamp())
            clips.append(
                {
                    "name": a.get("originalFileName") or aid,
                    "asset_id": aid,
                    "start_epoch": start_epoch,
                    "end_epoch": start_epoch + dur,
                    "source": "immich",
                    "url": f"/immich-video/{aid}",
                }
            )
        except Exception:
            continue
    clips.sort(key=lambda c: c["start_epoch"])
    return clips


def _build_clip_index(deployment_ident, video_dir, clip_tz_name):
    """Prefer Immich; fall back to local files when Immich has nothing."""
    immich = _build_immich_clip_index(deployment_ident)
    if immich:
        print(f"[video] Using {len(immich)} Immich clip(s) for {deployment_ident}.")
        return immich
    local = _build_video_clip_index(video_dir, clip_tz_name)
    if local:
        print(f"[video] Using {len(local)} local clip(s) from {video_dir}.")
    else:
        print("[video] No Immich or local clips found.")
    return local


_video_clip_index = _build_clip_index(deployment_id, VIDEO_DIR, tz_name)


# ---------------------------------------------------------------------------
# Deployment/dataset availability annotations for the selection dropdowns
# ---------------------------------------------------------------------------
# Each dropdown option is labeled with: # Immich videos, # local videos, and the
# date the deployment's output.nc was last generated. Network + disk lookups are
# cached because dropdowns rebuild often.
_LOCAL_MEDIA_ROOT = (config.get("paths", {}) or {}).get("local_private_media")
_local_count_cache = {}
_nc_date_cache = {}
_immich_album_counts = None  # dict: deployment -> assetCount, built once


def _immich_album_count_map():
    """One call to /albums?shared=true -> {deployment: assetCount} for DepID_ albums.

    assetCount is per-album (video-only albums here); used as the Immich availability
    hint so the dropdowns don't fire one request per deployment.
    """
    global _immich_album_counts
    if _immich_album_counts is not None:
        return _immich_album_counts
    counts = {}
    svc = _get_immich_service()
    if svc:
        try:
            r = svc.session.get(f"{svc.base_url}/albums", params={"shared": "true"}, timeout=20)
            if r.status_code == 200:
                for a in r.json():
                    name = a.get("albumName") or ""
                    if name.startswith("DepID_"):
                        counts[name[len("DepID_"):]] = a.get("assetCount")
        except Exception:
            counts = {}
    _immich_album_counts = counts
    return counts


def _local_video_dir_for(dataset, deployment):
    """<media_root>/<dataset>/<deployment>_media/02_processed-video, if configured."""
    if not _LOCAL_MEDIA_ROOT or not dataset or not deployment:
        return None
    return pathlib.Path(_LOCAL_MEDIA_ROOT) / dataset / f"{deployment}_media" / "02_processed-video"


def _local_video_count(dataset, deployment):
    key = (dataset, deployment)
    if key in _local_count_cache:
        return _local_count_cache[key]
    n = 0
    d = _local_video_dir_for(dataset, deployment)
    try:
        if d and d.is_dir():
            n = sum(1 for p in d.glob("*.mp4") if p.is_file())
    except Exception:
        n = 0
    _local_count_cache[key] = n
    return n


def _immich_video_count(deployment):
    """Immich asset count for album DepID_<deployment>; None if Immich unavailable,
    0 if reachable but no such album."""
    counts = _immich_album_count_map()
    if not counts and _get_immich_service() is False:
        return None
    return counts.get(deployment, 0)


def _output_nc_date(dataset, deployment):
    """Date (YYYY-MM-DD) the deployment's *_output.nc was last modified, or None."""
    key = (dataset, deployment)
    if key in _nc_date_cache:
        return _nc_date_cache[key]
    date = None
    try:
        outputs = pathlib.Path(data_dir) / dataset / deployment / "outputs"
        candidates = sorted(outputs.glob(f"{deployment}_output.nc"))
        if not candidates:
            # Fall back to any *_output.nc (excluding *_copy.nc).
            candidates = [p for p in outputs.glob("*_output.nc") if "_copy" not in p.name]
        if candidates:
            mtime = max(p.stat().st_mtime for p in candidates)
            date = pd.Timestamp(mtime, unit="s").strftime("%Y-%m-%d")
    except Exception:
        date = None
    _nc_date_cache[key] = date
    return date


def _deployment_option_label(dataset, deployment):
    """'2021-04-17_mian-011 · immich 2 · local 5 · nc 2026-03-30'."""
    imm = _immich_video_count(deployment)
    loc = _local_video_count(dataset, deployment)
    nc = _output_nc_date(dataset, deployment)
    imm_txt = f"immich {imm}" if imm is not None else "immich ?"
    parts = [deployment, f"🎬 {imm_txt} · local {loc}", f"nc {nc}" if nc else "nc —"]
    return "  ·  ".join(parts)


def _deployment_options_with_meta(dataset):
    return [
        {"label": _deployment_option_label(dataset, dep), "value": dep}
        for dep in _list_deployments(data_dir, dataset)
    ]


def _dataset_option_label(dataset):
    """Aggregate across the dataset's deployments: total videos + newest nc date."""
    deps = _list_deployments(data_dir, dataset)
    if not deps:
        return dataset
    loc_total = sum(_local_video_count(dataset, d) for d in deps)
    imm_counts = [_immich_video_count(d) for d in deps]
    imm_known = [c for c in imm_counts if c is not None]
    imm_total = sum(imm_known) if imm_known else None
    nc_dates = [x for x in (_output_nc_date(dataset, d) for d in deps) if x]
    newest = max(nc_dates) if nc_dates else None
    imm_txt = f"immich {imm_total}" if imm_total is not None else "immich ?"
    parts = [f"{dataset} ({len(deps)})", f"🎬 {imm_txt} · local {loc_total}", f"nc {newest}" if newest else "nc —"]
    return "  ·  ".join(parts)


def _dataset_options_with_meta():
    return [{"label": _dataset_option_label(d), "value": d} for d in _list_datasets(data_dir)]


def _video_coverage_bars(track_class="coverage-bar"):
    """Build one CSS-positioned bar per clip (EcoPhysVideoViz style).

    Each bar carries its absolute epoch bounds as CSS variables; the containing
    strip carries --view-min/--view-max, and CSS computes left/width as a percent
    of the current view. Updating the strip's view vars re-positions every bar,
    which keeps the strip aligned to whatever x-range it represents.
    """
    bars = []
    for clip in _video_clip_index:
        bars.append(
            html.Div(
                className=track_class,
                title=clip["name"],
                style={
                    "--seg-start": clip["start_epoch"],
                    "--seg-end": clip["end_epoch"],
                },
            )
        )
    return bars


def _format_gap(seconds):
    """Human-readable gap: '12s', '3m 05s', '2h 14m', '10d 4h'."""
    s = int(round(abs(seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"


def _find_video_clip_for_epoch(epoch):
    """Return (clip, offset_seconds) for the clip containing epoch, else (None, None)."""
    if epoch is None:
        return None, None
    try:
        epoch = float(epoch)
    except Exception:
        return None, None
    for clip in _video_clip_index:
        if clip["start_epoch"] <= epoch <= clip["end_epoch"]:
            return clip, max(0.0, epoch - clip["start_epoch"])
    return None, None


# Faststart cache: these mp4s have their 'moov' atom at the END of the file, so
# the browser can't build a seek index until it downloads the tail. Range-seeks
# then render black until enough re-buffers. We remux (container copy, no
# re-encode) with 'moov' moved to the front into this cache dir and serve that.
_FASTSTART_DIR = (VIDEO_DIR / ".faststart") if VIDEO_DIR else None
_FASTSTART_TOOL = shutil.which("qt-faststart") or shutil.which("ffmpeg")


def _needs_faststart(path):
    """True if the mp4's moov atom sits after mdat (not web-optimized)."""
    try:
        import struct

        order = []
        with open(path, "rb") as f:
            pos = 0
            while len(order) < 8:
                f.seek(pos)
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                size = struct.unpack(">I", hdr[:4])[0]
                typ = hdr[4:8].decode("latin1", "replace")
                order.append(typ)
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0]
                if size == 0:
                    break
                pos += size
        if "moov" in order and "mdat" in order:
            return order.index("moov") > order.index("mdat")
    except Exception:
        pass
    return False


def _faststart_path(src):
    """Return a web-optimized copy of src, creating it once in the cache dir.

    Falls back to the original path if remuxing is unavailable or fails.
    """
    src = pathlib.Path(src)
    try:
        if not _needs_faststart(src):
            return src
        if _FASTSTART_TOOL is None:
            return src
        _FASTSTART_DIR.mkdir(parents=True, exist_ok=True)
        out = _FASTSTART_DIR / src.name
        if out.is_file() and out.stat().st_size > 0:
            return out
        tmp = out.with_suffix(".tmp.mp4")
        tool = os.path.basename(_FASTSTART_TOOL)
        if tool == "qt-faststart":
            cmd = [_FASTSTART_TOOL, str(src), str(tmp)]
        else:  # ffmpeg
            cmd = [_FASTSTART_TOOL, "-y", "-i", str(src), "-c", "copy",
                   "-movflags", "+faststart", str(tmp)]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        if res.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(out)
            return out
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return src


@app.server.route("/local-video/<path:filename>")
def _serve_local_video(filename):
    """Stream a video from VIDEO_DIR with HTTP range support (needed for seeking)."""
    from flask import abort, send_file

    # Guard against path traversal: only serve plain basenames from VIDEO_DIR.
    if not VIDEO_DIR:
        abort(404)
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.lower().endswith(".mp4"):
        abort(404)
    target = (VIDEO_DIR / safe_name)
    if not target.is_file():
        abort(404)
    served = _faststart_path(target)
    # send_file honors Range requests via conditional=True.
    return send_file(str(served), conditional=True, mimetype="video/mp4")


_IMMICH_ASSET_ID_RE = re.compile(r"^[0-9a-fA-F-]{16,64}$")


@app.server.route("/immich-video/<asset_id>")
def _serve_immich_video(asset_id):
    """Proxy an Immich video/playback stream, keeping the API key server-side.

    Forwards the browser's Range header to Immich and relays status + range
    headers back, so the <video> element can seek. The x-api-key never reaches
    the browser.
    """
    from flask import Response, abort, request, stream_with_context

    if not _IMMICH_ASSET_ID_RE.match(asset_id or ""):
        abort(404)
    svc = _get_immich_service()
    if not svc:
        abort(503)
    upstream = f"{svc.base_url}/assets/{asset_id}/video/playback"
    fwd_headers = {"x-api-key": svc.api_key}
    rng = request.headers.get("Range")
    if rng:
        fwd_headers["Range"] = rng
    try:
        up = svc.session.get(upstream, headers=fwd_headers, stream=True, timeout=60)
    except Exception:
        abort(502)
    if up.status_code not in (200, 206):
        up.close()
        abort(up.status_code if up.status_code >= 400 else 502)

    passthrough = {}
    for h in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
        if h in up.headers:
            passthrough[h] = up.headers[h]
    passthrough.setdefault("Content-Type", "video/mp4")
    passthrough.setdefault("Accept-Ranges", "bytes")

    def _generate():
        try:
            for chunk in up.iter_content(chunk_size=262144):
                if chunk:
                    yield chunk
        finally:
            up.close()

    return Response(stream_with_context(_generate()), status=up.status_code, headers=passthrough)


_current_fig = None
_depth_ctx_x, _depth_ctx_y, _depth_ctx_label, _depth_ctx_unit, _depth_ctx_source_signal = _extract_depth_context_series(data_pkl, tz_name)
_signal_colors = _load_color_mapping(color_mapping_path)
_event_styles_global = _load_event_styles(color_mapping_path)
_event_targets_global = _load_event_targets(color_mapping_path)

# In-memory cache of downsampled signal frames keyed by deployment+signal+target Hz.
_downsample_cache = {}
_signal_display_metadata = _load_signal_display_metadata(data_dir)
_initial_signal_axis_config = _load_signal_axis_config_from_params(param_manager)
_initial_model_3d_controls = _load_model_3d_controls_from_params(param_manager)
_initial_peak_param_order = _load_peak_param_order_from_params(param_manager)

color_editor_modal = html.Div(
    [
        html.Div(
            [
                html.Div(className="color-editor-edge color-editor-edge-top", **{"data-edge": "top"}),
                html.Div(className="color-editor-edge color-editor-edge-right", **{"data-edge": "right"}),
                html.Div(className="color-editor-edge color-editor-edge-bottom", **{"data-edge": "bottom"}),
                html.Div(className="color-editor-edge color-editor-edge-left", **{"data-edge": "left"}),
                html.Div(
                    [
                        html.Div("Edit Trace Color", className="drawer-title", id="color-editor-title"),
                        html.Div(
                            [
                                html.Button(
                                    "⤢",
                                    id="color-move-btn",
                                    n_clicks=0,
                                    className="color-editor-move-handle",
                                    title="Move",
                                ),
                                html.Button(
                                    "×",
                                    id="color-close-btn",
                                    n_clicks=0,
                                    className="color-editor-close-btn",
                                    title="Close",
                                ),
                            ],
                            className="color-editor-header-actions",
                        ),
                    ],
                    className="color-editor-header",
                ),
                html.Div(
                    [
                        html.Div(id="color-preview-swatch", className="color-preview-swatch"),
                        html.Div(id="color-preview-text", className="color-preview-text"),
                    ],
                    className="color-preview-row",
                ),
                dcc.Input(
                    id="color-picker-input",
                    type="text",
                    value="#4DAEF8",
                    className="color-state-input",
                    style={"display": "none"},
                ),
                dcc.Input(
                    id="color-hex-input",
                    type="text",
                    value="#4DAEF8",
                    debounce=True,
                    className="color-hex-input",
                    placeholder="#RRGGBB",
                ),
                html.Div(
                    [
                        html.Label("Event Marker Shape"),
                        dcc.Dropdown(
                            id="color-editor-event-symbol",
                            options=EVENT_SYMBOL_DROPDOWN_OPTIONS,
                            value="circle",
                            clearable=False,
                            className="event-symbol-select",
                        ),
                        dcc.Checklist(
                            id="color-editor-event-shade-enabled",
                            options=[{"label": "Shade duration", "value": "on"}],
                            value=["on"],
                            className="event-shade-toggle",
                        ),
                        html.Label("Shade Color"),
                        dcc.Input(
                            id="color-editor-event-shade-hex",
                            type="text",
                            value="#4DAEF8",
                            debounce=True,
                            className="color-hex-input",
                            placeholder="#RRGGBB",
                        ),
                        html.Label("Shade Opacity"),
                        dcc.Slider(
                            id="color-editor-event-shade-opacity",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.2,
                            marks={0: "0", 0.2: "0.2", 0.5: "0.5", 1: "1"},
                            tooltip={"always_visible": False},
                        ),
                        html.Label("Shade Placement"),
                        dcc.Dropdown(
                            id="color-editor-event-shade-mode",
                            options=EVENT_SHADE_MODE_OPTIONS,
                            value="fill_trace_split",
                            clearable=False,
                            className="event-symbol-select",
                        ),
                        html.Div(
                            [
                                html.Label("Band % Min"),
                                dcc.Input(
                                    id="color-editor-event-shade-pct-min",
                                    type="number",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=0,
                                    className="event-color-input",
                                ),
                                html.Label("Band % Max"),
                                dcc.Input(
                                    id="color-editor-event-shade-pct-max",
                                    type="number",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=100,
                                    className="event-color-input",
                                ),
                            ],
                            className="signal-axis-config-row",
                        ),
                    ],
                    id="color-editor-event-style-wrap",
                    className="color-editor-event-targets-wrap",
                    style={"display": "none"},
                ),
                html.Div(id="hue-preview-text", className="hsv-values-row"),
                html.Div(
                    [
                        html.Label("Hue"),
                        dcc.Slider(id="hue-slider", min=0, max=360, step=1, value=0, marks=None, tooltip={"always_visible": False}),
                        html.Div(
                            [
                                html.Div(id="hue-gradient-box", className="hue-gradient-box"),
                            ],
                            className="hue-preview-row",
                        ),
                        html.Label("Saturation / Value"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Value", className="axis-label-y"),
                                        dcc.Slider(
                                            id="val-slider",
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=0,
                                            marks=None,
                                            vertical=True,
                                            verticalHeight=150,
                                            tooltip={"always_visible": False},
                                        ),
                                    ],
                                    className="value-axis-col",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(id="color-sv-dot", className="color-sv-dot"),
                                            ],
                                            id="color-sv-picker",
                                            className="color-sv-picker",
                                        ),
                                        html.Div("Saturation", className="axis-label-x"),
                                        dcc.Slider(
                                            id="sat-slider",
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=0,
                                            marks=None,
                                            tooltip={"always_visible": False},
                                        ),
                                    ],
                                    className="sat-axis-col",
                                ),
                            ],
                            className="sv-axis-layout",
                        ),
                    ],
                    className="color-hsv-controls",
                ),
                html.Div(id="color-swatch-grid", className="color-swatch-grid"),
                html.Div(
                    [
                        html.Button("Apply", id="color-apply-btn", n_clicks=0, className="apply-btn"),
                        html.Button("Cancel", id="color-cancel-btn", n_clicks=0, className="apply-btn"),
                    ],
                    className="color-editor-actions",
                ),
            ],
            className="color-editor-card",
        )
    ],
    id="color-editor-modal",
    className="color-editor-modal hidden",
)


main_page_layout = html.Div(
    [
        html.Button("Datasets", id="drawer-toggle", className="drawer-toggle", n_clicks=0, style={"display": "none"}),
        html.Div(
            [
                html.Img(src="/assets/images/pyologger_logo.png", className="app-brand-logo"),
                html.Div(
                    [
                        html.Div("pyologger", className="app-brand-title", id="app-title"),
                        html.Div("visu-analyze biologging data with python", className="app-brand-tagline"),
                        html.Div(f"{dataset_id} / {deployment_id}", className="app-brand-subtitle", id="app-subtitle"),
                    ],
                    className="app-brand-text",
                ),
            ],
            className="app-title",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Details(
                                    [
                                        html.Summary(
                                            [html.Span("Deployment Selection", className="section-title")],
                                            className="collapsible-summary",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Dataset"),
                                                dcc.Dropdown(
                                                    id="dataset-select",
                                                    options=_dataset_options_with_meta(),
                                                    value=dataset_id,
                                                    multi=False,
                                                ),
                                                html.Label("Deployment", style={"marginTop": "10px"}),
                                                dcc.Dropdown(
                                                    id="deployment-select",
                                                    options=_deployment_options_with_meta(dataset_id),
                                                    value=deployment_id,
                                                    multi=False,
                                                ),
                                                html.Button("Apply", id="apply-deployment", n_clicks=0, className="apply-btn"),
                                                html.Div(id="deployment-load-status", className="window-summary"),
                                            ],
                                            className="collapsible-body",
                                        ),
                                    ],
                                    className="collapsible-panel",
                                    open=True,
                                ),
                            ],
                            id="dataset-drawer",
                            className="dataset-drawer open",
                        ),
                        html.Div(
                            [
                                html.Details(
                                    [
                                        html.Summary(
                                            html.Div(
                                                [
                                                    html.Span("Signal Selection", className="section-title"),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Div("Active Signals", className="legend-section-title"),
                                                                    html.Div(id="signals-summary-chips"),
                                                                ],
                                                                className="selection-summary-block",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Div("Active Events", className="legend-section-title"),
                                                                    html.Div(id="events-summary-chips"),
                                                                ],
                                                                className="selection-summary-block",
                                                            ),
                                                        ],
                                                        className="selection-summary-stack",
                                                    ),
                                                ],
                                                className="selection-summary-content",
                                            ),
                                            className="collapsible-summary",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        dcc.Dropdown(
                                                            id="signals-select",
                                                            options=[{"label": s, "value": s} for s in all_signals],
                                                            value=default_signals,
                                                            multi=True,
                                                            className="with-chip-order",
                                                            placeholder="Add/remove signals...",
                                                        ),
                                                        html.Div(id="signal-order-editor"),
                                                        html.Div(id="channels-editor", className="control-card dockable-panel"),
                                                        dcc.Dropdown(
                                                            id="events-select",
                                                            options=[{"label": k, "value": k} for k in event_key_options],
                                                            value=all_event_defaults,
                                                            multi=True,
                                                            className="with-chip-order",
                                                            placeholder="Add/remove events...",
                                                        ),
                                                        html.Div(id="events-key-chips"),
                                                        html.Div(id="events-target-editor"),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Save signal and event dataset defaults",
                                                            id="save-signal-event-defaults-btn",
                                                            n_clicks=0,
                                                            className="apply-btn",
                                                        ),
                                                        html.Div(id="save-signal-event-defaults-status", className="window-summary"),
                                                    ],
                                                    className="color-editor-actions",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Save signal colors to config",
                                                            id="save-signal-colors-btn",
                                                            n_clicks=0,
                                                            className="apply-btn",
                                                        ),
                                                        html.Div(id="save-signal-colors-status", className="window-summary"),
                                                    ],
                                                    className="color-editor-actions",
                                                ),
                                                html.Div("Event Selection", className="section-title legend-subsection-title"),
                                                html.Div("Active Events", className="legend-section-title legend-active-title"),
                                                html.Div(id="custom-events-panel", className="legend-active-stack"),
                                                html.Div(
                                                    [
                                                        dcc.Dropdown(
                                                            id="legend-add-event-select",
                                                            options=[],
                                                            value=None,
                                                            clearable=True,
                                                            placeholder="Add event...",
                                                            className="with-chip-order legend-add-signal-select",
                                                        ),
                                                        html.Button("Add", id="legend-add-event-btn", n_clicks=0, className="apply-btn"),
                                                    ],
                                                    className="legend-add-signal-row",
                                                ),
                                                html.Div("Signal Selection", className="section-title legend-subsection-title"),
                                                html.Div("Active Signals", className="legend-section-title legend-active-title"),
                                                html.Div(id="custom-signals-panel", className="legend-active-stack"),
                                                html.Div(
                                                    [
                                                        dcc.Dropdown(
                                                            id="legend-add-signal-select",
                                                            options=[],
                                                            value=None,
                                                            clearable=True,
                                                            placeholder="Add signal...",
                                                            className="with-chip-order legend-add-signal-select",
                                                        ),
                                                        html.Button("Add", id="legend-add-signal-btn", n_clicks=0, className="apply-btn"),
                                                    ],
                                                    className="legend-add-signal-row",
                                                ),
                                            ],
                                            className="collapsible-body",
                                        ),
                                    ],
                                    className="collapsible-panel",
                                    open=True,
                                ),
                            ],
                            className="control-card",
                        ),
                        html.Div(
                            [
                                html.Details(
                                    [
                                        html.Summary(
                                            [html.Span("Map", className="section-title")],
                                            className="collapsible-summary",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Checklist(
                                                    id="location-map-3d-toggle",
                                                    options=[{"label": "Add third dimension (vertical)", "value": "on"}],
                                                    value=[],
                                                    className="signal-custom-range-toggle",
                                                ),
                                                dcc.Graph(
                                                    id="location-map",
                                                    className="location-map",
                                                    style={"height": "300px"},
                                                    figure=_build_location_map_figure(
                                                        slider_default,
                                                        playhead_value=_default_playhead_epoch(slider_default),
                                                    ),
                                                    config={"displayModeBar": False, "scrollZoom": False, "responsive": True},
                                                ),
                                            ],
                                            className="collapsible-body",
                                        ),
                                    ],
                                    className="collapsible-panel",
                                    open=True,
                                ),
                            ],
                            id="location-map-card",
                            className="control-card legend-map-card",
                        ),
                        html.Div(
                            [
                                html.Details(
                                    [
                                        html.Summary(
                                            [html.Span("Peak Detection Tool", className="section-title")],
                                            className="collapsible-summary",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Enter peak detection mode"),
                                                        html.Div(
                                                            [
                                                                html.Button("Detect heartbeats", id="peak-detect-heart-btn", n_clicks=0, className="apply-btn"),
                                                                html.Button("Detect strokes", id="peak-detect-stroke-btn", n_clicks=0, className="apply-btn"),
                                                            ],
                                                            className="peak-enter-row",
                                                        ),
                                                        html.Fieldset(
                                                            [
                                                                html.Label("Mode"),
                                                                dcc.Dropdown(
                                                                    id="peak-mode",
                                                                    options=[
                                                                        {"label": "Heart Rate", "value": "heart_rate"},
                                                                        {"label": "Stroke Rate", "value": "stroke_rate"},
                                                                    ],
                                                                    value="heart_rate",
                                                                    clearable=False,
                                                                ),
                                                                html.Label("Parent Signal"),
                                                                dcc.Dropdown(id="peak-parent-signal", options=[], value=None, clearable=False),
                                                                html.Label("Channel"),
                                                                dcc.Dropdown(id="peak-channel", options=[], value=None, clearable=False),
                                                                html.Label("Detection Sources"),
                                                                dcc.Dropdown(
                                                                    id="peak-detection-sources",
                                                                    options=PEAK_DERIVATIVE_OPTIONS,
                                                                    value=["normalized"],
                                                                    multi=True,
                                                                    clearable=False,
                                                                ),
                                                            ],
                                                            id="peak-controls-fieldset",
                                                            disabled=True,
                                                            style={"border": "none", "padding": 0, "margin": 0},
                                                        ),
                                                    ],
                                                    className="time-row",
                                                ),
                                                html.Fieldset(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Button(
                                                                    "Expand all",
                                                                    id="peak-expand-collapse-btn",
                                                                    n_clicks=0,
                                                                    className="apply-btn",
                                                                ),
                                                            ],
                                                            className="color-editor-actions",
                                                        ),
                                                        html.Div(
                                                            html.Div(
                                                                id="peak-params-editor",
                                                                children=_render_peak_param_cards(_initial_peak_param_order),
                                                            ),
                                                            className="time-row",
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.Checklist(id="peak-enable-bandpass", options=[{"label": "enable_bandpass", "value": "on"}], value=["on"]),
                                                                dcc.Checklist(id="peak-enable-spike", options=[{"label": "enable_spike_removal", "value": "on"}], value=["on"]),
                                                                dcc.Checklist(id="peak-enable-absolute", options=[{"label": "enable_absolute", "value": "on"}], value=["on"]),
                                                                dcc.Checklist(id="peak-enable-smoothing", options=[{"label": "enable_smoothing", "value": "on"}], value=["on"]),
                                                                dcc.Checklist(id="peak-enable-normalization", options=[{"label": "enable_normalization", "value": "on"}], value=["on"]),
                                                                dcc.Checklist(id="peak-enable-refinement", options=[{"label": "enable_refinement", "value": "on"}], value=["on"]),
                                                                dcc.Checklist(id="peak-pick-last-conflict", options=[{"label": "PICK_LAST_IN_CONFLICT_PAIR", "value": "on"}], value=["on"]),
                                                            ],
                                                            className="color-editor-actions",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Button("Preview Peaks (Current Window)", id="peak-preview-btn", n_clicks=0, className="apply-btn"),
                                                                html.Button(
                                                                    "Calculate peaks for rest of deployment",
                                                                    id="peak-next-btn",
                                                                    n_clicks=0,
                                                                    className="apply-btn",
                                                                    disabled=True,
                                                                ),
                                                                html.Button(
                                                                    "Save events to deployment",
                                                                    id="peak-save-btn",
                                                                    n_clicks=0,
                                                                    className="apply-btn",
                                                                    disabled=True,
                                                                ),
                                                            ],
                                                            className="color-editor-actions",
                                                        ),
                                                    ],
                                                    id="peak-controls-fieldset-extra",
                                                    disabled=True,
                                                    style={"border": "none", "padding": 0, "margin": 0},
                                                ),
                                                html.Div(id="peak-status", className="window-summary"),
                                            ],
                                            className="collapsible-body",
                                        ),
                                    ],
                                    className="collapsible-panel",
                                    open=False,
                                )
                            ],
                            className="control-card dockable-panel",
                        ),
                    ],
                    className="custom-legend-panel",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                        html.Div(
                            [
                                html.Details(
                                    [
                                        html.Summary(
                                            [html.Div("3D Model with Track", className="section-title")],
                                            className="card-title-row model-3d-card-summary",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span("Drag", id="model-3d-drag-handle", className="panel-popout-drag-handle"),
                                                        html.Button("Stow Tools", id="model-3d-rail-toggle-btn", n_clicks=0, className="apply-btn model-3d-rail-toggle-btn"),
                                                        html.Button("Pop out", id="model-3d-popout-btn", n_clicks=0, className="apply-btn pip-toggle-btn"),
                                                    ],
                                                    className="card-title-actions model-3d-card-actions",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                (
                                                                    html.Div(
                                                                        [
                                                                            dcc.Loading(
                                                                                # Spinner while the 3D model + Notion fetch resolves.
                                                                                # This panel loads LAST (after deployment, data,
                                                                                # and video) and never blocks them.
                                                                                type="circle",
                                                                                color="#7ec5eb",
                                                                                children=html.Div(
                                                                                    (
                                                                                        three_js_orientation.ThreeJsOrientation(
                                                                                            id="model-3d-viewer",
                                                                                            data=EMPTY_ORIENTATION_JSON,
                                                                                            activeTime=float(_default_playhead_epoch(slider_default)) * 1000.0,
                                                                                            cameraFollowModel=True,
                                                                                            modelFile="",
                                                                                            textureFile="",
                                                                                            pitchOffset=_initial_model_3d_controls.get("pitch_offset", 0.0),
                                                                                            rollOffset=_initial_model_3d_controls.get("roll_offset", 0.0),
                                                                                            headingOffset=_initial_model_3d_controls.get("heading_offset", 0.0),
                                                                                            pitchSign=_initial_model_3d_controls.get("pitch_sign", 1),
                                                                                            rollSign=_initial_model_3d_controls.get("roll_sign", 1),
                                                                                            headingSign=_initial_model_3d_controls.get("heading_sign", 1),
                                                                                            rotationOrder=_initial_model_3d_controls.get("rotation_order", ["roll", "pitch", "heading"]),
                                                                                            style={
                                                                                                "width": "100%",
                                                                                                "height": "100%",
                                                                                                "--pause-stroke-threshold": "10",
                                                                                                "--show-trajectory": "1",
                                                                                                "--trajectory-line-width": str(float(_initial_model_3d_controls.get("track_line_width", 7.1))),
                                                                                                "--trajectory-highlight-width": str(float(_initial_model_3d_controls.get("highlight_width", 1.55))),
                                                                                                "--trajectory-highlight-offset": str(float(_initial_model_3d_controls.get("highlight_offset", 0.18))),
                                                                                            },
                                                                                        )
                                                                                    ),
                                                                                    className="model-3d-viewer-wrap",
                                                                                ),
                                                                            ),
                                                                            html.Div(
                                                                                three_js_orientation.ThreeJsOrientation(
                                                                                    id="model-3d-viewer-inset",
                                                                                    data=EMPTY_ORIENTATION_JSON,
                                                                                    activeTime=float(_default_playhead_epoch(slider_default)) * 1000.0,
                                                                                    cameraFollowModel=True,
                                                                                    modelFile="",
                                                                                    textureFile="",
                                                                                    pitchOffset=_initial_model_3d_controls.get("pitch_offset", 0.0),
                                                                                    rollOffset=_initial_model_3d_controls.get("roll_offset", 0.0),
                                                                                    headingOffset=_initial_model_3d_controls.get("heading_offset", 0.0),
                                                                                    pitchSign=_initial_model_3d_controls.get("pitch_sign", 1),
                                                                                    rollSign=_initial_model_3d_controls.get("roll_sign", 1),
                                                                                    headingSign=_initial_model_3d_controls.get("heading_sign", 1),
                                                                                    rotationOrder=_initial_model_3d_controls.get("rotation_order", ["roll", "pitch", "heading"]),
                                                                                    style={
                                                                                        "width": "100%",
                                                                                        "height": "100%",
                                                                                        "--pause-stroke-threshold": "10",
                                                                                        "--show-trajectory": "0",
                                                                                    },
                                                                                ),
                                                                                className="model-3d-viewer-inset-inner",
                                                                            ) if THREEJS_AVAILABLE else None,
                                                                        ],
                                                                        className="model-3d-stage",
                                                                    )
                                                                    if THREEJS_AVAILABLE
                                                                    else html.Div(
                                                                        "three_js_orientation is not available in this environment.",
                                                                        className="aux-placeholder",
                                                                    )
                                                                ),
                                                                html.Details(
                                                                    [
                                                                        html.Summary("Model Metadata", className="model-3d-panel-summary"),
                                                                        html.Div(
                                                                            id="model-3d-status",
                                                                            children=[html.Div("Loading 3D model metadata...")],
                                                                            className="model-3d-status-inline model-3d-panel-body",
                                                                        ),
                                                                    ],
                                                                    className="model-3d-panel-shell model-3d-metadata-shell",
                                                                    open=True,
                                                                ),
                                                            ],
                                                            className="model-3d-main-column",
                                                        ),
                                                        html.Details(
                                                            [
                                                                html.Summary("Tools & Controls", className="model-3d-panel-summary"),
                                                                html.Div(
                                                                    [
                                                                        html.Div(
                                                                            [
                                                                                html.Div("Order", className="model-3d-sidepanel-label"),
                                                                                html.Div(id="model-3d-rotation-order-editor"),
                                                                            ],
                                                                            className="model-3d-toolbox",
                                                                        ),
                                                                        html.Div(
                                                                            [
                                                                                html.Div(
                                                                                    [
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div("X-Axis", className="model-3d-axis-title"),
                                                                                                html.Div(
                                                                                                    dcc.Checklist(
                                                                                                        id="model-flip-pitch",
                                                                                                        options=[{"label": "Flip", "value": "on"}],
                                                                                                        value=(["on"] if int(_initial_model_3d_controls.get("pitch_sign", 1)) < 0 else []),
                                                                                                        className="model-3d-flip-toggle",
                                                                                                    ),
                                                                                                    className="model-3d-axis-toggle",
                                                                                                ),
                                                                                            ],
                                                                                            className="model-3d-axis-header",
                                                                                        ),
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div(
                                                                                                    dcc.Input(
                                                                                                        id="model-rot-x",
                                                                                                        type="number",
                                                                                                        value=_initial_model_3d_controls.get("pitch_offset", 0.0),
                                                                                                        step=1,
                                                                                                        className="model-3d-rot-input",
                                                                                                    ),
                                                                                                    className="model-3d-input-shell",
                                                                                                ),
                                                                                                html.Span("deg", className="model-3d-rot-unit"),
                                                                                            ],
                                                                                            className="model-3d-axis-inputrow",
                                                                                        ),
                                                                                    ],
                                                                                    className="model-3d-axis-card",
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div("Y-Axis", className="model-3d-axis-title"),
                                                                                                html.Div(
                                                                                                    dcc.Checklist(
                                                                                                        id="model-flip-roll",
                                                                                                        options=[{"label": "Flip", "value": "on"}],
                                                                                                        value=(["on"] if int(_initial_model_3d_controls.get("roll_sign", 1)) < 0 else []),
                                                                                                        className="model-3d-flip-toggle",
                                                                                                    ),
                                                                                                    className="model-3d-axis-toggle",
                                                                                                ),
                                                                                            ],
                                                                                            className="model-3d-axis-header",
                                                                                        ),
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div(
                                                                                                    dcc.Input(
                                                                                                        id="model-rot-y",
                                                                                                        type="number",
                                                                                                        value=_initial_model_3d_controls.get("roll_offset", 0.0),
                                                                                                        step=1,
                                                                                                        className="model-3d-rot-input",
                                                                                                    ),
                                                                                                    className="model-3d-input-shell",
                                                                                                ),
                                                                                                html.Span("deg", className="model-3d-rot-unit"),
                                                                                            ],
                                                                                            className="model-3d-axis-inputrow",
                                                                                        ),
                                                                                    ],
                                                                                    className="model-3d-axis-card",
                                                                                ),
                                                                            ],
                                                                            className="model-3d-axis-stack",
                                                                        ),
                                                                        html.Div(
                                                                            [
                                                                                html.Div(
                                                                                    [
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div("Z-Axis", className="model-3d-axis-title"),
                                                                                                html.Div(
                                                                                                    dcc.Checklist(
                                                                                                        id="model-flip-heading",
                                                                                                        options=[{"label": "Flip", "value": "on"}],
                                                                                                        value=(["on"] if int(_initial_model_3d_controls.get("heading_sign", 1)) < 0 else []),
                                                                                                        className="model-3d-flip-toggle",
                                                                                                    ),
                                                                                                    className="model-3d-axis-toggle",
                                                                                                ),
                                                                                            ],
                                                                                            className="model-3d-axis-header",
                                                                                        ),
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div(
                                                                                                    dcc.Input(
                                                                                                        id="model-rot-z",
                                                                                                        type="number",
                                                                                                        value=_initial_model_3d_controls.get("heading_offset", 0.0),
                                                                                                        step=1,
                                                                                                        className="model-3d-rot-input",
                                                                                                    ),
                                                                                                    className="model-3d-input-shell",
                                                                                                ),
                                                                                                html.Span("deg", className="model-3d-rot-unit"),
                                                                                            ],
                                                                                            className="model-3d-axis-inputrow",
                                                                                        ),
                                                                                    ],
                                                                                    className="model-3d-axis-card",
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Div("Pause animation (SR <)", className="model-3d-sidepanel-label"),
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div(
                                                                                                    dcc.Input(
                                                                                                        id="model-anim-pause-threshold",
                                                                                                        type="number",
                                                                                                        value=10.0,
                                                                                                        step=0.1,
                                                                                                        className="model-3d-rot-input",
                                                                                                    ),
                                                                                                    className="model-3d-input-shell",
                                                                                                ),
                                                                                                html.Span("spm", className="model-3d-rot-unit"),
                                                                                            ],
                                                                                            className="model-3d-axis-inputrow",
                                                                                        ),
                                                                                    ],
                                                                                    className="model-3d-utility-card",
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Div("Track width", className="model-3d-sidepanel-label"),
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div(
                                                                                                    dcc.Input(
                                                                                                        id="model-track-line-width",
                                                                                                        type="number",
                                                                                                        value=_initial_model_3d_controls.get("track_line_width", 7.1),
                                                                                                        step=0.1,
                                                                                                        min=1.0,
                                                                                                        className="model-3d-rot-input",
                                                                                                    ),
                                                                                                    className="model-3d-input-shell",
                                                                                                ),
                                                                                                html.Span("px", className="model-3d-rot-unit"),
                                                                                            ],
                                                                                            className="model-3d-axis-inputrow",
                                                                                        ),
                                                                                    ],
                                                                                    className="model-3d-utility-card",
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Div("Highlight width", className="model-3d-sidepanel-label"),
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div(
                                                                                                    dcc.Input(
                                                                                                        id="model-track-highlight-width",
                                                                                                        type="number",
                                                                                                        value=_initial_model_3d_controls.get("highlight_width", 1.55),
                                                                                                        step=0.05,
                                                                                                        min=0.1,
                                                                                                        className="model-3d-rot-input",
                                                                                                    ),
                                                                                                    className="model-3d-input-shell",
                                                                                                ),
                                                                                                html.Span("px", className="model-3d-rot-unit"),
                                                                                            ],
                                                                                            className="model-3d-axis-inputrow",
                                                                                        ),
                                                                                    ],
                                                                                    className="model-3d-utility-card",
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Div("Highlight offset", className="model-3d-sidepanel-label"),
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div(
                                                                                                    dcc.Input(
                                                                                                        id="model-track-highlight-offset",
                                                                                                        type="number",
                                                                                                        value=_initial_model_3d_controls.get("highlight_offset", 0.18),
                                                                                                        step=0.01,
                                                                                                        min=0.0,
                                                                                                        className="model-3d-rot-input",
                                                                                                    ),
                                                                                                    className="model-3d-input-shell",
                                                                                                ),
                                                                                                html.Span("u", className="model-3d-rot-unit"),
                                                                                            ],
                                                                                            className="model-3d-axis-inputrow",
                                                                                        ),
                                                                                    ],
                                                                                    className="model-3d-utility-card",
                                                                                ),
                                                                                html.Button(
                                                                                    "Save Configuration",
                                                                                    id="save-model-3d-defaults",
                                                                                    n_clicks=0,
                                                                                    className="apply-btn model-3d-save-btn",
                                                                                ),
                                                                                html.Div(id="model-3d-save-status", className="model-3d-save-status"),
                                                                                html.Details(
                                                                                    [
                                                                                        html.Summary("Show Debug", className="model-3d-debug-summary"),
                                                                                        html.Div(
                                                                                            [
                                                                                                html.Div(
                                                                                                    "Track debug: waiting for playhead",
                                                                                                    id="model-3d-track-status",
                                                                                                    className="model-3d-status-inline",
                                                                                                ),
                                                                                            ],
                                                                                            className="model-3d-debug-body",
                                                                                        ),
                                                                                    ],
                                                                                    className="model-3d-debug-panel model-3d-rail-debug-panel",
                                                                                ),
                                                                            ],
                                                                            className="model-3d-sidepanel",
                                                                        ),
                                                                    ],
                                                                    className="model-3d-panel-body",
                                                                ),
                                                            ],
                                                            className="model-3d-panel-shell model-3d-tools-shell",
                                                            open=True,
                                                        ),
                                                    ],
                                                    className="model-3d-layout",
                                                ),
                                            ],
                                            className="aux-card-body model-3d-settings-body",
                                        ),
                                    ],
                                    className="collapsible-panel aux-widget-panel",
                                    open=True,
                                ),
                            ],
                            id="model-3d-card",
                            className="control-card model-3d-card plot-top-model-card",
                        ),
                                html.Div(
                                    [
                                html.Div(
                                    [
                                        html.Details(
                                            [
                                                html.Summary(
                                                    [html.Span("Time Selector", className="section-title")],
                                                    className="collapsible-summary",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    id="time-zone-used-label",
                                                                    className="time-tz-caption",
                                                                    children=_time_display_label("local"),
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="time-display-tz-toggle",
                                                                    options=[
                                                                        {"label": "Local", "value": "local"},
                                                                        {"label": "UTC", "value": "utc"},
                                                                    ],
                                                                    value="local",
                                                                    className="time-zone-toggle",
                                                                    inputClassName="time-zone-toggle-input",
                                                                    labelClassName="time-zone-toggle-label",
                                                                ),
                                                            ],
                                                            className="time-tz-row",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Label("Start time", className="time-field-label"),
                                                                        dcc.Input(
                                                                            id="start-time",
                                                                            type="text",
                                                                            value=_format_ts_input(start_default, _display_tz_name("local")),
                                                                            debounce=True,
                                                                            className="time-input",
                                                                        ),
                                                                    ],
                                                                    className="time-field",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label("End time", className="time-field-label"),
                                                                        dcc.Input(
                                                                            id="end-time",
                                                                            type="text",
                                                                            value=_format_ts_input(end_default, _display_tz_name("local")),
                                                                            debounce=True,
                                                                            className="time-input",
                                                                        ),
                                                                    ],
                                                                    className="time-field",
                                                                ),
                                                                html.Div(id="window-summary", className="window-duration-chip"),
                                                            ],
                                                            className="time-entry-row",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Button(
                                                                            "▶",
                                                                            id="play-pause-btn",
                                                                            n_clicks=0,
                                                                            className="time-nav-btn play-pause-btn",
                                                                            title="Play / pause (space)",
                                                                            **{"data-playing": "0"},
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="playback-rate-select",
                                                                            options=[
                                                                                {"label": "0.5×", "value": 0.5},
                                                                                {"label": "1×", "value": 1},
                                                                                {"label": "2×", "value": 2},
                                                                                {"label": "5×", "value": 5},
                                                                            ],
                                                                            value=1,
                                                                            clearable=False,
                                                                            searchable=False,
                                                                            className="playback-rate-select",
                                                                        ),
                                                                    ],
                                                                    className="time-nav-group playback-group",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Button("−", id="tnav-zoom-out", n_clicks=0, className="time-nav-btn", title="Zoom out (2×)"),
                                                                        html.Button("+", id="tnav-zoom-in", n_clicks=0, className="time-nav-btn", title="Zoom in (2×)"),
                                                                        html.Button("⤢", id="tnav-full", n_clicks=0, className="time-nav-btn", title="Fit full range"),
                                                                    ],
                                                                    className="time-nav-group",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Button("‹", id="tnav-pan-left", n_clicks=0, className="time-nav-btn", title="Pan left"),
                                                                        html.Button("›", id="tnav-pan-right", n_clicks=0, className="time-nav-btn", title="Pan right"),
                                                                    ],
                                                                    className="time-nav-group",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Button(lbl, id={"type": "tnav-preset", "seconds": secs}, n_clicks=0, className="time-preset-btn", title=f"Set window to {lbl}")
                                                                        for lbl, secs in (("10s", 10), ("1m", 60), ("5m", 300), ("30m", 1800), ("1h", 3600))
                                                                    ],
                                                                    className="time-nav-group time-preset-group",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Button("⇤", id="tnav-ph-start", n_clicks=0, className="time-nav-btn", title="Playhead to window start"),
                                                                        html.Button("◂", id="tnav-ph-back", n_clicks=0, className="time-nav-btn", title="Step playhead back 5s"),
                                                                        html.Button("▸", id="tnav-ph-fwd", n_clicks=0, className="time-nav-btn", title="Step playhead forward 5s"),
                                                                        html.Button("⇥", id="tnav-ph-end", n_clicks=0, className="time-nav-btn", title="Playhead to window end"),
                                                                    ],
                                                                    className="time-nav-group",
                                                                ),
                                                            ],
                                                            className="time-nav-toolbar",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label("Target Sampling (Hz)"),
                                                                dcc.Input(
                                                                    id="target-sampling-rate-hz",
                                                                    type="number",
                                                                    min=0.01,
                                                                    step=0.01,
                                                                    value=10.0,
                                                                    debounce=True,
                                                                    className="time-input time-input-narrow",
                                                                ),
                                                                html.Div(id="target-sampling-rate-interval", className="window-summary"),
                                                            ],
                                                            className="time-row",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label("Overlap Signal A"),
                                                                dcc.Dropdown(
                                                                    id="overlap-signal-a",
                                                                    options=[{"label": s, "value": s} for s in default_signals],
                                                                    value=(default_signals[0] if default_signals else None),
                                                                    clearable=False,
                                                                ),
                                                                html.Label("Overlap Signal B"),
                                                                dcc.Dropdown(
                                                                    id="overlap-signal-b",
                                                                    options=[{"label": s, "value": s} for s in default_signals],
                                                                    value=(default_signals[1] if len(default_signals) > 1 else (default_signals[0] if default_signals else None)),
                                                                    clearable=False,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Button("Calculate Overlap", id="overlap-calc-btn", n_clicks=0, className="apply-btn"),
                                                                        html.Button("Set Time Window To Overlap", id="overlap-set-window-btn", n_clicks=0, className="apply-btn"),
                                                                        html.Button("Trim To Overlap", id="overlap-trim-btn", n_clicks=0, className="apply-btn"),
                                                                    ],
                                                                    className="color-editor-actions",
                                                                ),
                                                                html.Div(id="overlap-status", className="window-summary"),
                                                            ],
                                                            className="time-row",
                                                        ),
                                                    ],
                                                    className="collapsible-body time-row",
                                                ),
                                            ],
                                            className="collapsible-panel",
                                            open=True,
                                        ),
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="mini-depth-plot",
                                                    className="mini-context-plot",
                                                    figure=_build_depth_context_figure(
                                                        start_default,
                                                        end_default,
                                                        playhead_ts=_from_epoch_seconds(_default_playhead_epoch(slider_default), tz_name),
                                                    ),
                                                    config={
                                                        "displayModeBar": False,
                                                        "scrollZoom": False,
                                                        "editable": True,
                                                        "edits": {"shapePosition": True},
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span("Window", className="track-legend-dot track-legend-window"),
                                                        html.Span(id="window-summary-inline", className="track-caption"),
                                                    ],
                                                    className="track-row-header",
                                                ),
                                                dcc.RangeSlider(
                                                    id="time-range-slider",
                                                    min=slider_min,
                                                    max=slider_max,
                                                    step=1,
                                                    pushable=MIN_WINDOW_SECONDS,
                                                    value=slider_default,
                                                    marks=None,
                                                    className="window-slider",
                                                    tooltip={
                                                        "always_visible": False,
                                                        "placement": "bottom",
                                                    },
                                                ),
                                                html.Div(
                                                    _video_coverage_bars(),
                                                    id="coverage-strip-full",
                                                    className="coverage-strip",
                                                    style={"--view-min": slider_min, "--view-max": slider_max},
                                                    title="Video coverage across the full deployment",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(id="abs-start-label", className="slider-endpoint"),
                                                        html.Div(id="abs-end-label", className="slider-endpoint"),
                                                    ],
                                                    className="slider-endpoints",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span("Playhead", className="track-legend-dot track-legend-playhead"),
                                                        html.Span(id="playhead-summary", className="track-caption"),
                                                    ],
                                                    className="track-row-header track-row-header-playhead",
                                                ),
                                                dcc.Slider(
                                                    id="playhead-slider",
                                                    min=int(min(slider_default)),
                                                    max=int(max(slider_default)),
                                                    step=1,
                                                    value=float(_default_playhead_epoch(slider_default)),
                                                    marks=None,
                                                    className="playhead-slider",
                                                    updatemode="drag",
                                                    tooltip={
                                                        "always_visible": False,
                                                        "placement": "bottom",
                                                    },
                                                ),
                                            ],
                                            className="time-slider-row",
                                ),
                            ],
                            className="plot-toolbar control-card",
                        ),
                                        html.Div(
                                            [
                                                html.Details(
                                                    [
                                                        html.Summary(
                                                            [html.Span("Synchronized Video", className="section-title")],
                                                            className="collapsible-summary",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    html.Video(
                                                                        id="sync-video",
                                                                        src="",
                                                                        controls=True,
                                                                        # "auto" (not "metadata"): metadata-only preload
                                                                        # leaves seeks parked on a black frame because the
                                                                        # frame data at the target time is never fetched.
                                                                        preload="auto",
                                                                        className="sync-video-el",
                                                                    ),
                                                                    className="sync-video-frame",
                                                                ),
                                                                html.Div(id="sync-video-status", className="sync-video-status"),
                                                                html.Div(
                                                                    [
                                                                        dcc.Checklist(
                                                                            id="sync-video-follow",
                                                                            options=[{"label": "Follow playhead", "value": "on"}],
                                                                            value=["on"],
                                                                            className="sync-video-follow",
                                                                        ),
                                                                    ],
                                                                    className="sync-video-controls",
                                                                ),
                                                            ],
                                                            className="collapsible-body sync-video-body",
                                                        ),
                                                    ],
                                                    className="collapsible-panel",
                                                    open=True,
                                                ),
                                            ],
                                            className="control-card sync-video-card",
                                        ),
                                    ],
                                    className="time-video-row",
                                ),
                            ],
                            className="plot-top-row",
                        ),
                        html.Div(
                            [
                                html.Span("Video", className="coverage-caption"),
                                html.Div(
                                    html.Div(
                                        _video_coverage_bars(),
                                        id="coverage-strip-window",
                                        className="coverage-strip coverage-strip-window",
                                        style={"--view-min": int(min(slider_default)), "--view-max": int(max(slider_default))},
                                    ),
                                    id="coverage-plot-align",
                                    className="coverage-plot-align",
                                ),
                            ],
                            className="coverage-row",
                        ),
                        dcc.Graph(
                            id="main-plot",
                            className="main-plot",
                            config={
                                "displayModeBar": True,
                                "scrollZoom": False,
                                "editable": True,
                                "edits": {"shapePosition": True},
                            },
                        ),
                        html.Div(
                            [
                                html.Div("Event-Channel Mapper (Experimental)", className="section-title"),
                                html.Small(
                                    "Click an event node, then click signal nodes to add/remove links. Click an edge to remove it.",
                                    className="event-style-label",
                                ),
                                (
                                    cyto.Cytoscape(
                                        id="event-channel-cyto",
                                        elements=[],
                                        layout={"name": "preset"},
                                        style={"width": "100%", "height": "440px"},
                                        wheelSensitivity=1,
                                        stylesheet=[
                                            {
                                                "selector": "node",
                                                "style": {
                                                    "label": "data(label)",
                                                    "font-size": "11px",
                                                    "text-wrap": "wrap",
                                                    "text-max-width": "220px",
                                                    "text-valign": "center",
                                                    "text-halign": "center",
                                                    "color": "#dce9f3",
                                                    "border-width": 1,
                                                    "border-color": "#7faec7",
                                                    "background-color": "#173d58",
                                                    "width": 180,
                                                    "height": 24,
                                                    "padding-left": "8px",
                                                    "padding-right": "8px",
                                                },
                                            },
                                            {
                                                "selector": "node[kind = 'event']",
                                                "style": {
                                                    "shape": "round-rectangle",
                                                    "background-color": "data(color)",
                                                    "border-width": "data(border_width)",
                                                    "border-color": "data(border_color)",
                                                },
                                            },
                                            {
                                                "selector": "node[kind = 'signal']",
                                                "style": {
                                                    "shape": "round-rectangle",
                                                    "background-color": "data(color)",
                                                },
                                            },
                                            {
                                                "selector": "node[kind $= 'port']",
                                                "style": {
                                                    "width": 12,
                                                    "height": 12,
                                                    "label": "",
                                                    "background-color": "data(color)",
                                                    "border-color": "data(border_color)",
                                                    "border-width": 1,
                                                },
                                            },
                                            {
                                                "selector": "node:selected",
                                                "style": {
                                                    "border-color": "#ffffff",
                                                    "border-width": 2,
                                                },
                                            },
                                            {
                                                "selector": "edge",
                                                "style": {
                                                    "display": "none",
                                                },
                                            },
                                            {
                                                "selector": "edge[kind = 'portlink']",
                                                "style": {
                                                    "line-color": "data(color)",
                                                },
                                            },
                                            {
                                                "selector": "edge[kind = 'mapping']",
                                                "style": {
                                                    "opacity": "data(edge_opacity)",
                                                    "width": "data(edge_width)",
                                                },
                                            },
                                            {
                                                "selector": "node[kind = 'event-port'], node[kind = 'signal-port']",
                                                "style": {
                                                    "shape": "ellipse",
                                                    "width": 10,
                                                    "height": 10,
                                                    "background-color": "#ffffff",
                                                    "border-width": 2,
                                                    "border-color": "#7faec7",
                                                    "label": "",
                                                },
                                            },
                                            {
                                                "selector": "edge[kind = 'portlink']",
                                                "style": {
                                                    "display": "element",
                                                    "line-color": "rgba(127, 174, 199, 0.55)",
                                                    "width": 1.2,
                                                    "curve-style": "straight",
                                                    "target-arrow-shape": "none",
                                                },
                                            },
                                            {
                                                "selector": "edge[kind = 'mapping']",
                                                "style": {
                                                    "display": "element",
                                                    "curve-style": "unbundled-bezier",
                                                    "control-point-distances": "55 -55",
                                                    "control-point-weights": "0.25 0.75",
                                                    "target-arrow-shape": "triangle",
                                                    "line-color": "#87badd",
                                                    "target-arrow-color": "#87badd",
                                                    "width": 2,
                                                },
                                            },
                                        ],
                                        userZoomingEnabled=True,
                                        userPanningEnabled=True,
                                        boxSelectionEnabled=False,
                                        minZoom=0.4,
                                        maxZoom=2.2,
                                    )
                                    if CYTO_AVAILABLE
                                    else html.Div(
                                        "dash-cytoscape not available. Use fallback mapper below.",
                                        className="legend-empty",
                                    )
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="map-event-select",
                                            options=[],
                                            value=None,
                                            multi=False,
                                            placeholder="Event node",
                                        ),
                                        dcc.Dropdown(
                                            id="map-channel-select",
                                            options=[],
                                            value=None,
                                            multi=False,
                                            placeholder="Signal node",
                                        ),
                                        html.Button("Add Link", id="map-add-link", n_clicks=0, className="apply-btn"),
                                    ],
                                    className="event-map-controls",
                                    style={"display": "none"} if CYTO_AVAILABLE else None,
                                ),
                                html.Div(
                                    id="map-links-chips",
                                    className="chip-sortable",
                                    style={"display": "none"} if CYTO_AVAILABLE else None,
                                ),
                            ],
                            className="control-card event-map-card",
                        ),
                    ],
                    className="plot-panel",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Details(
                                    [
                                        html.Summary(
                                            [html.Span("Video Feed", className="section-title")],
                                            className="collapsible-summary",
                                        ),
                                        html.Div(
                                            [
                                                html.B("Placeholder"),
                                                html.Div("Optional synchronized video panel will go here."),
                                                html.Div("Planned source: DiveDB media assets."),
                                            ],
                                            className="aux-placeholder collapsible-body",
                                        ),
                                    ],
                                    className="collapsible-panel",
                                    open=False,
                                )
                            ],
                            className="control-card aux-card",
                        ),
                    ],
                    className="aux-panel",
                ),
            ],
            className="app-shell",
        ),
        dcc.Store(id="current-dataset", data=dataset_id),
        dcc.Store(id="current-deployment", data=deployment_id),
        dcc.Store(id="playhead-time", data=float(_default_playhead_epoch(slider_default))),
        dcc.Store(id="ordered-signals-store", data=default_signals),
        dcc.Store(id="selected-signals-store", data=default_signals),
        dcc.Store(id="channels-store", data={}),
        dcc.Store(id="channel-order-store", data={}),
        dcc.Store(id="signal-axis-config-store", data=_initial_signal_axis_config),
        dcc.Store(id="selected-events-store", data=all_event_defaults),
        dcc.Store(id="event-targets-store", data=_event_targets_global),
        dcc.Store(id="event-style-store", data=_event_styles_global),
        dcc.Store(
            id="model-3d-rotation-order-store",
            data=_normalize_rotation_order(_initial_model_3d_controls.get("rotation_order")),
        ),
        dcc.Store(id="active-event-editor", data={}),
        dcc.Store(id="color-edit-target", data={}),
        dcc.Store(id="color-map-version", data=0),
        dcc.Store(id="map-selected-event", data=None),
        dcc.Store(id="overlap-window-store", data={}),
        dcc.Store(id="sync-video-current-clip", data=None),
        dcc.Store(id="is-playing", data=False),
        dcc.Store(id="playback-rate", data=1),
        dcc.Store(id="playback-dummy", data=0),
        dcc.Interval(id="playback-interval", interval=100, n_intervals=0, disabled=True),
        dcc.Store(id="peak-detect-preview-store", data={}),
        dcc.Store(id="peak-detect-progress-store", data={}),
        dcc.Store(id="peak-view-active-store", data={"active": False, "mode": "heart_rate"}),
        dcc.Store(id="peak-param-order-store", data=_initial_peak_param_order),
        dcc.Store(id="peak-param-open-store", data={}),
        dcc.Interval(id="peak-auto-recalc-interval", interval=1000, n_intervals=0, max_intervals=0, disabled=True),
        html.Div(
            [
                dcc.Dropdown(id="point-events-select", options=[], value=[], multi=True),
                dcc.Dropdown(id="state-events-select", options=[], value=[], multi=True),
                html.Div(id="point-events-summary-chips"),
                html.Div(id="state-events-summary-chips"),
                html.Div(id="point-event-key-chips"),
                html.Div(id="state-event-key-chips"),
                html.Div(id="point-event-target-editor"),
                html.Div(id="state-event-target-editor"),
            ],
            style={"display": "none"},
        ),
        dcc.Input(id="color-editor-anchor-input", type="text", value="", style={"display": "none"}),
        dcc.Input(id="color-sv-input", type="text", value="", style={"display": "none"}),
        dcc.Input(id="chip-order-updates", type="text", value="{}", style={"display": "none"}),
        dcc.Input(id="arrow-key-input", type="text", value="", style={"display": "none"}),
    ],
    className="integrated-app",
)

segmentation_page_layout = _build_segmentation_page_layout(
    dataset_options=dataset_options,
    default_dataset=seg_default_dataset,
    default_deployment=seg_default_deployment,
    algo_defaults=_segmentation_algo_defaults,
    workflow_defaults=_segmentation_workflow_defaults,
)

water_page_layout = html.Div(
    [
        html.Div(
            [
                html.Div("Experimental Water Tank", className="section-title"),
                html.Div(
                    "WebGL water-tank viewer synced to the current dataset, deployment, playhead, model, and track. "
                    "Use the same global playhead state and arrow-key nudges from the main app.",
                    className="annotation-info",
                    style={"marginTop": "6px"},
                ),
            ],
            className="control-card",
            style={"margin": "16px"},
        ),
        html.Div(
            [
                html.Div(
                    (
                        three_js_orientation.WebGLWaterTank(
                            id="water-viewer",
                            data=EMPTY_ORIENTATION_JSON,
                            activeTime=float(_default_playhead_epoch(slider_default)) * 1000.0,
                            cameraFollowModel=True,
                            modelFile="",
                            textureFile="",
                            pitchOffset=_initial_model_3d_controls.get("pitch_offset", 0.0),
                            rollOffset=_initial_model_3d_controls.get("roll_offset", 0.0),
                            headingOffset=_initial_model_3d_controls.get("heading_offset", 0.0),
                            pitchSign=_initial_model_3d_controls.get("pitch_sign", 1),
                            rollSign=_initial_model_3d_controls.get("roll_sign", 1),
                            headingSign=_initial_model_3d_controls.get("heading_sign", 1),
                            rotationOrder=_initial_model_3d_controls.get("rotation_order", ["roll", "pitch", "heading"]),
                            showTrajectory=True,
                            trackLineWidth=float(_initial_model_3d_controls.get("track_line_width", 7.1)),
                            highlightWidth=float(_initial_model_3d_controls.get("highlight_width", 1.55)),
                            highlightOffset=float(_initial_model_3d_controls.get("highlight_offset", 0.18)),
                            style={"width": "100%", "height": "100%"},
                        )
                    )
                    if WATER_VIEWER_AVAILABLE
                    else html.Div(
                        "Water viewer component is not available in this environment.",
                        className="aux-placeholder",
                    ),
                    className="water-viewer-stage",
                    style={"height": "68vh", "minHeight": "540px"},
                ),
                html.Div(
                    id="water-viewer-status",
                    className="control-card",
                    style={"marginTop": "14px"},
                    children=[html.Div("Loading water viewer metadata...")],
                ),
            ],
            style={"margin": "0 16px 16px 16px"},
        ),
    ]
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(
            [
                dcc.Link("Main", href="/", className="apply-btn"),
                dcc.Link("Segmentation", href="/segmentation", className="apply-btn"),
                dcc.Link("Water", href="/water", className="apply-btn"),
            ],
            className="color-editor-actions",
            style={"padding": "12px 16px 0 16px"},
        ),
        color_editor_modal,
        html.Div(main_page_layout, id="main-page-shell"),
        html.Div(segmentation_page_layout, id="segmentation-page-shell", style={"display": "none"}),
        html.Div(water_page_layout, id="water-page-shell", style={"display": "none"}),
    ]
)


@app.callback(
    Output("main-page-shell", "style"),
    Output("segmentation-page-shell", "style"),
    Output("water-page-shell", "style"),
    Input("url", "pathname"),
)
def route_integrated_pages(pathname):
    route = str(pathname or "").strip().lower()
    if route == "/segmentation":
        return {"display": "none"}, {"display": "block"}, {"display": "none"}
    if route == "/water":
        return {"display": "none"}, {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}, {"display": "none"}


@app.callback(
    Output("seg-workflow-sidebar", "style"),
    Output("seg-workflow-sidebar-btn", "children"),
    Input("seg-workflow-sidebar-btn", "n_clicks"),
    Input("seg-workflow-sidebar-close-btn", "n_clicks"),
    Input("url", "pathname"),
    State("seg-workflow-sidebar", "style"),
    prevent_initial_call=False,
)
def toggle_segmentation_workflow_sidebar(_open_clicks, _close_clicks, pathname, current_style):
    base = {
        "position": "fixed",
        "top": "72px",
        "right": "12px",
        "width": "420px",
        "maxWidth": "calc(100vw - 24px)",
        "maxHeight": "calc(100vh - 84px)",
        "overflowY": "auto",
        "overflowX": "hidden",
        "padding": "10px",
        "background": "#081B2A",
        "border": "1px solid rgba(255,255,255,0.12)",
        "borderRadius": "12px",
        "boxShadow": "0 16px 48px rgba(0,0,0,0.35)",
        "zIndex": 1200,
        "display": "none",
    }
    if str(pathname or "").strip().lower() != "/segmentation":
        return base, "Open Workflow Editor"
    if ctx.triggered_id == "seg-workflow-sidebar-close-btn":
        return base, "Open Workflow Editor"
    if ctx.triggered_id == "seg-workflow-sidebar-btn":
        visible = str((current_style or {}).get("display") or "none") != "none"
        style = dict(base)
        style["display"] = "none" if visible else "block"
        return style, ("Open Workflow Editor" if visible else "Close Workflow Editor")
    return base, "Open Workflow Editor"


@app.callback(
    Output("dataset-drawer", "className"),
    Input("drawer-toggle", "n_clicks"),
    State("dataset-drawer", "className"),
)
def toggle_drawer(n_clicks, current_class):
    current_class = current_class or "dataset-drawer closed"
    if not n_clicks:
        return current_class
    return "dataset-drawer open" if "closed" in current_class else "dataset-drawer closed"


def _toggle_popout_class(current_class, open_token, base_class):
    tokens = [t for t in str(current_class or base_class).split() if t and t != open_token]
    is_open = open_token in str(current_class or "")
    if is_open:
        return " ".join(tokens)
    tokens.append(open_token)
    return " ".join(tokens)


@app.callback(
    Output("model-3d-card", "className"),
    Output("model-3d-popout-btn", "children"),
    Output("model-3d-rail-toggle-btn", "children"),
    Input("model-3d-popout-btn", "n_clicks"),
    Input("model-3d-rail-toggle-btn", "n_clicks"),
    Input("app-title", "children"),
    State("model-3d-card", "className"),
    prevent_initial_call=False,
)
def toggle_model_3d_popout(_popout_clicks, _rail_clicks, _title, current_class):
    base = "control-card aux-card model-3d-card plot-top-model-card"
    # Only toggle when the popout button is explicitly clicked.
    # For initial render / app reload / title updates, always dock.
    if ctx.triggered_id not in {"model-3d-popout-btn", "model-3d-rail-toggle-btn"}:
        return base, "Pop out", "Stow Tools"
    if ctx.triggered_id == "model-3d-popout-btn":
        updated = _toggle_popout_class(current_class, "pip-open", base)
    else:
        updated = _toggle_popout_class(current_class, "rail-stowed", base)
    return (
        updated,
        ("Dock" if "pip-open" in updated else "Pop out"),
        ("Show Tools" if "rail-stowed" in updated else "Stow Tools"),
    )


@app.callback(
    Output("seg-workflow-diagram-card", "className"),
    Output("seg-workflow-diagram-popout-btn", "children"),
    Input("seg-workflow-diagram-popout-btn", "n_clicks"),
    Input("url", "pathname"),
    State("seg-workflow-diagram-card", "className"),
    prevent_initial_call=False,
)
def toggle_segmentation_workflow_diagram_popout(_n_clicks, pathname, current_class):
    base = "control-card aux-card seg-workflow-diagram-card"
    if str(pathname or "").strip().lower() != "/segmentation":
        return base, "Pop out"
    if ctx.triggered_id != "seg-workflow-diagram-popout-btn":
        return base, "Pop out"
    updated = _toggle_popout_class(current_class, "pip-open", base)
    return updated, ("Dock" if "pip-open" in updated else "Pop out")


@app.callback(
    Output("deployment-select", "options"),
    Output("deployment-select", "value"),
    Input("dataset-select", "value"),
    State("deployment-select", "value"),
)
def update_deployment_options(selected_dataset, current_deployment):
    deployments = _list_deployments(data_dir, selected_dataset)
    opts = _deployment_options_with_meta(selected_dataset)
    value = current_deployment if current_deployment in deployments else (deployments[0] if deployments else None)
    return opts, value


@app.callback(
    Output("app-title", "children", allow_duplicate=True),
    Output("app-subtitle", "children", allow_duplicate=True),
    Output("signals-select", "options", allow_duplicate=True),
    Output("signals-select", "value", allow_duplicate=True),
    Output("selected-signals-store", "data", allow_duplicate=True),
    Output("events-select", "options", allow_duplicate=True),
    Output("events-select", "value", allow_duplicate=True),
    Output("selected-events-store", "data", allow_duplicate=True),
    Output("point-events-select", "options", allow_duplicate=True),
    Output("point-events-select", "value", allow_duplicate=True),
    Output("state-events-select", "options", allow_duplicate=True),
    Output("state-events-select", "value", allow_duplicate=True),
    Output("time-range-slider", "min", allow_duplicate=True),
    Output("time-range-slider", "max", allow_duplicate=True),
    Output("time-range-slider", "value", allow_duplicate=True),
    Output("time-range-slider", "marks", allow_duplicate=True),
    Output("playhead-slider", "min", allow_duplicate=True),
    Output("playhead-slider", "max", allow_duplicate=True),
    Output("playhead-slider", "value", allow_duplicate=True),
    Output("playhead-time", "data", allow_duplicate=True),
    Output("time-zone-used-label", "children", allow_duplicate=True),
    Output("time-display-tz-toggle", "value", allow_duplicate=True),
    Output("start-time", "value", allow_duplicate=True),
    Output("end-time", "value", allow_duplicate=True),
    Output("ordered-signals-store", "data", allow_duplicate=True),
    Output("channels-store", "data", allow_duplicate=True),
    Output("channel-order-store", "data", allow_duplicate=True),
    Output("signal-axis-config-store", "data", allow_duplicate=True),
    Output("event-targets-store", "data", allow_duplicate=True),
    Output("event-style-store", "data", allow_duplicate=True),
    Output("current-dataset", "data", allow_duplicate=True),
    Output("current-deployment", "data", allow_duplicate=True),
    Output("peak-param-order-store", "data", allow_duplicate=True),
    Output("deployment-load-status", "children", allow_duplicate=True),
    Output("peak-view-active-store", "data", allow_duplicate=True),
    Input("apply-deployment", "n_clicks"),
    State("dataset-select", "value"),
    State("deployment-select", "value"),
    prevent_initial_call=True,
)
def apply_dataset_deployment(_n_clicks, selected_dataset, selected_deployment):
    global animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder
    global data_pkl, param_manager, tz_name, deployment_tz_name, time_display_tz_mode, global_start, global_end
    global start_default, end_default, slider_min, slider_max, slider_default, slider_marks
    global default_signals, _depth_ctx_x, _depth_ctx_y, _depth_ctx_label, _depth_ctx_unit, _depth_ctx_source_signal, _event_styles_global, _event_targets_global, _downsample_cache
    global deployment_source

    if not selected_dataset or not selected_deployment:
        raise dash.exceptions.PreventUpdate

    (
        animal_id,
        dataset_id,
        deployment_id,
        dataset_folder,
        deployment_folder,
        param_manager,
    ) = resolve_deployment_context(data_dir, dataset_id=selected_dataset, deployment_id=selected_deployment)
    deployment_source = resolve_deployment_source(
        data_dir,
        dataset_id,
        deployment_id,
        deployment_folder=deployment_folder,
    )
    plot_signal_allowlist = resolve_plot_signal_allowlist(param_manager, deployment_source)
    data_pkl = deployment_source.build_metadata_shell(allowed_signals=plot_signal_allowlist)

    tz_name = str(data_pkl.deployment_info.get("Time Zone", "UTC") or "UTC")
    deployment_tz_name = tz_name
    time_display_tz_mode = "local"
    std_local = standardize_time_settings(
        param_manager=param_manager,
        data_pkl=data_pkl,
        tz_name=tz_name,
        minutes=10,
        persist=False,
    )
    ctx_local = _compute_view_context(
        data_pkl,
        tz_name,
        std_local["zoom_window_start_time"],
        std_local["zoom_window_end_time"],
        param_manager_obj=param_manager,
    )
    global_start = ctx_local["global_start"]
    global_end = ctx_local["global_end"]
    slider_min = ctx_local["slider_min"]
    slider_max = ctx_local["slider_max"]
    slider_default = ctx_local["slider_default"]
    slider_marks = ctx_local["slider_marks"]
    start_default = ctx_local["start_default"]
    end_default = ctx_local["end_default"]
    _depth_ctx_x, _depth_ctx_y, _depth_ctx_label, _depth_ctx_unit, _depth_ctx_source_signal = _extract_depth_context_series(data_pkl, tz_name)
    _event_styles_global = _load_event_styles(color_mapping_path)
    _event_targets_global = _load_event_targets(color_mapping_path)
    _downsample_cache.clear()

    all_sigs = _available_signal_names(data_pkl)
    defaults = ctx_local["default_signals"]
    default_signals = defaults
    ev_opts = _available_event_keys(data_pkl)
    default_events = [str(e) for e in (ctx_local.get("default_events") or ev_opts)]
    point_defaults, state_defaults = _split_event_keys_by_state(data_pkl, default_events)
    all_defaults = default_events
    win_lo = int(min(slider_default))
    win_hi = int(max(slider_default))
    playhead_default_epoch = float(_default_playhead_epoch(slider_default))

    return (
        "pyologger",
        f"{dataset_id} / {deployment_id}",
        [{"label": s, "value": s} for s in all_sigs],
        defaults,
        defaults,
        [{"label": k, "value": k} for k in ev_opts],
        all_defaults,
        all_defaults,
        [{"label": k, "value": k} for k in ev_opts],
        point_defaults,
        [{"label": k, "value": k} for k in ev_opts],
        state_defaults,
        slider_min,
        slider_max,
        slider_default,
        None,
        win_lo,
        win_hi,
        playhead_default_epoch,
        playhead_default_epoch,
        _time_display_label("local"),
        "local",
        _format_ts_input(start_default, _display_tz_name("local")),
        _format_ts_input(end_default, _display_tz_name("local")),
        defaults,
        {},
        {},
        _load_signal_axis_config_from_params(param_manager),
        _event_targets_global,
        _event_styles_global,
        dataset_id,
        deployment_id,
        _load_peak_param_order_from_params(param_manager),
        f"Loaded deployment {dataset_id} / {deployment_id}. Peak detection settings loaded from deployment config.",
        {"active": False, "mode": "heart_rate"},
    )


def _load_segmentation_context(selected_dataset, selected_deployment):
    _, _, _, _, _, seg_data_pkl, seg_param_manager = select_and_load_deployment(
        data_dir,
        dataset_id=selected_dataset,
        deployment_id=selected_deployment,
    )
    seg_tz = str((getattr(seg_data_pkl, "deployment_info", {}) or {}).get("Time Zone") or "UTC")
    seg_ctx = _segmentation_initial_window(seg_data_pkl, seg_tz, hours=DEFAULT_SEGMENTATION_WINDOW_HOURS)
    default_selected, all_sigs = _segmentation_default_signals(seg_data_pkl)
    saved_selected = _load_segmentation_signal_defaults(seg_param_manager, all_sigs)
    if saved_selected:
        default_selected = saved_selected
    return seg_data_pkl, seg_param_manager, seg_tz, seg_ctx, default_selected, all_sigs


@app.callback(
    Output("seg-workflow-store", "data"),
    Output("seg-workflow-select", "options"),
    Output("seg-workflow-select", "value"),
    Output("seg-workflow-draft", "data"),
    Output("seg-workflow-status", "children"),
    Input("seg-workflow-select", "value"),
    Input("seg-new-workflow-btn", "n_clicks"),
    Input("seg-duplicate-workflow-btn", "n_clicks"),
    Input("seg-reset-workflow-btn", "n_clicks"),
    Input("seg-save-workflow-btn", "n_clicks"),
    State("seg-workflow-store", "data"),
    State("seg-workflow-draft", "data"),
    prevent_initial_call=False,
)
def manage_segmentation_workflows(selected_name, _new_clicks, _dup_clicks, _reset_clicks, _save_clicks, workflow_store, workflow_draft):
    workflows = copy.deepcopy(workflow_store or {})
    if not workflows:
        workflows = {"find_rest": build_find_rest_workflow_preset(_segmentation_algo_defaults)}
    workflows = {str(name): sync_workflow_nodes_from_runtime_params(item) for name, item in workflows.items()}
    trig = ctx.triggered_id
    status = no_update
    current_name = str(selected_name or "")
    if not current_name or current_name not in workflows:
        current_name = sorted(workflows.keys())[0]
    draft = sync_workflow_nodes_from_runtime_params(copy.deepcopy(workflow_draft or workflows.get(current_name) or {}))

    if trig == "seg-new-workflow-btn":
        base = "workflow"
        idx = 1
        new_name = f"{base}_{idx}"
        while new_name in workflows:
            idx += 1
            new_name = f"{base}_{idx}"
        draft = copy.deepcopy(draft or build_find_rest_workflow_preset(_segmentation_algo_defaults))
        draft["workflow_name"] = new_name
        draft["display_name"] = str(new_name).replace("_", " ").title()
        current_name = new_name
        status = f"Created unsaved workflow draft '{new_name}'."
    elif trig == "seg-duplicate-workflow-btn":
        try:
            workflows, current_name = duplicate_segmentation_workflow(workflows, current_name)
            draft = copy.deepcopy(workflows[current_name])
            status = f"Duplicated workflow as '{current_name}'."
        except Exception as e:
            status = f"Failed to duplicate workflow: {type(e).__name__}: {e}"
    elif trig == "seg-reset-workflow-btn":
        draft = sync_workflow_nodes_from_runtime_params(copy.deepcopy(workflows.get(current_name) or build_find_rest_workflow_preset(_segmentation_algo_defaults)))
        status = f"Reset draft to saved preset '{current_name}'."
    elif trig == "seg-save-workflow-btn":
        candidate = sync_workflow_nodes_from_runtime_params(copy.deepcopy(draft or {}))
        candidate["workflow_name"] = _safe_workflow_name(candidate.get("workflow_name") or current_name or "workflow")
        if not candidate.get("display_name"):
            candidate["display_name"] = str(candidate["workflow_name"]).replace("_", " ").title()
        errors = validate_segmentation_workflow_preset(candidate)
        if errors:
            status = "Workflow preset not saved: " + "; ".join(errors[:4])
        else:
            workflows[str(candidate["workflow_name"])] = candidate
            save_segmentation_workflow_presets(_seg_config_path, workflows)
            current_name = str(candidate["workflow_name"])
            draft = copy.deepcopy(candidate)
            status = f"Saved workflow preset '{current_name}' to config.yaml."
    elif trig == "seg-workflow-select":
        draft = sync_workflow_nodes_from_runtime_params(copy.deepcopy(workflows.get(current_name) or build_find_rest_workflow_preset(_segmentation_algo_defaults)))
        status = f"Loaded workflow preset '{current_name}'."
    options = segmentation_workflow_options(workflows)
    return workflows, options, current_name, draft, status


@app.callback(
    Output("seg-workflow-status", "children", allow_duplicate=True),
    Input("seg-validate-workflow-btn", "n_clicks"),
    State("seg-workflow-store", "data"),
    State("seg-workflow-select", "value"),
    State("seg-workflow-draft", "data"),
    prevent_initial_call=True,
)
def validate_segmentation_workflow_action(_validate_clicks, workflow_store, selected_name, workflow_draft):
    workflows = copy.deepcopy(workflow_store or {})
    current_name = str(selected_name or "")
    saved = sync_workflow_nodes_from_runtime_params(copy.deepcopy(workflows.get(current_name) or {}))
    draft = sync_workflow_nodes_from_runtime_params(copy.deepcopy(workflow_draft or {}))
    changed = _normalize_workflow_for_compare(draft) != _normalize_workflow_for_compare(saved)
    report = workflow_runtime_support_report(draft)
    parts = [f"Draft changed: {'yes' if changed else 'no'}."]
    parts.append(f"Schema valid: {'yes' if report.get('valid') else 'no'}.")
    parts.append(
        "Runtime support: executable with current backend."
        if report.get("runtime_supported")
        else "Runtime support: requires backend code changes."
    )
    messages = list(report.get("messages") or [])
    if messages:
        parts.append("Notes: " + "; ".join(messages[:4]))
    return " ".join(parts)


@app.callback(
    Output("seg-workflow-cyto", "elements"),
    Output("seg-selected-node", "options"),
    Output("seg-selected-node", "value"),
    Input("seg-workflow-draft", "data"),
    State("seg-selected-node", "value"),
    prevent_initial_call=False,
)
def render_segmentation_workflow_graph(workflow_draft, selected_node):
    workflow = sync_workflow_nodes_from_runtime_params(copy.deepcopy(workflow_draft or {}))
    node_opts = _workflow_editor_node_options(workflow)
    valid_values = {str(opt.get("value")) for opt in node_opts}
    if selected_node not in valid_values:
        selected_node = node_opts[0]["value"] if node_opts else None
    elements = segmentation_workflow_to_cytoscape_elements(workflow) if CYTO_AVAILABLE else []
    return elements, node_opts, selected_node


@app.callback(
    Output("seg-workflow-live-elements", "data"),
    Input("seg-workflow-cyto", "elements"),
    prevent_initial_call=False,
)
def sync_segmentation_workflow_live_elements(elements):
    return list(elements or [])


@app.callback(
    Output("seg-workflow-draft", "data", allow_duplicate=True),
    Output("seg-workflow-status", "children", allow_duplicate=True),
    Input("seg-save-node-positions-btn", "n_clicks"),
    State("seg-workflow-live-elements", "data"),
    State("seg-workflow-draft", "data"),
    prevent_initial_call=True,
)
def save_segmentation_workflow_positions(_save_clicks, live_elements, workflow_draft):
    workflow = copy.deepcopy(workflow_draft or {})
    node_map = {str((n or {}).get("id") or ""): copy.deepcopy(n or {}) for n in (workflow.get("nodes") or [])}
    if not node_map:
        raise dash.exceptions.PreventUpdate
    updated = 0
    for el in list(live_elements or []):
        data = dict((el or {}).get("data") or {})
        if "source" in data or "target" in data:
            continue
        nid = str(data.get("id") or "").strip()
        if not nid or nid not in node_map or nid.startswith("junction__"):
            continue
        pos = dict((el or {}).get("position") or {})
        if "x" not in pos or "y" not in pos:
            continue
        node_map[nid]["position"] = {"x": float(pos["x"]), "y": float(pos["y"])}
        updated += 1
    if not updated:
        return no_update, "No node positions were available to save."
    workflow["nodes"] = list(node_map.values())
    return workflow, f"Saved positions for {updated} workflow nodes."


@app.callback(
    Output("seg-selected-node", "value", allow_duplicate=True),
    Input("seg-workflow-cyto", "tapNodeData"),
    prevent_initial_call=True,
)
def set_segmentation_selected_node(node_data):
    if not isinstance(node_data, dict):
        raise dash.exceptions.PreventUpdate
    nid = str(node_data.get("id") or "")
    if not nid:
        raise dash.exceptions.PreventUpdate
    return nid


@app.callback(
    Output("seg-node-label", "value"),
    Output("seg-node-type", "value"),
    Output("seg-node-description", "value"),
    Output("seg-node-channel", "value"),
    Output("seg-node-operator", "value"),
    Output("seg-node-threshold", "value"),
    Output("seg-node-state-key", "value"),
    Output("seg-node-label-code", "value"),
    Output("seg-node-label-name", "value"),
    Input("seg-selected-node", "value"),
    State("seg-workflow-draft", "data"),
)
def load_segmentation_node_editor(selected_node, workflow_draft):
    workflow = copy.deepcopy(workflow_draft or {})
    node_map = {str((n or {}).get("id") or ""): dict(n or {}) for n in (workflow.get("nodes") or [])}
    node = node_map.get(str(selected_node or ""), {})
    return (
        str(node.get("label") or ""),
        str(node.get("type") or "decision"),
        str(node.get("description") or ""),
        str(node.get("channel") or ""),
        str(node.get("operator") or "<="),
        node.get("threshold", 0),
        str(node.get("state_key") or ""),
        node.get("label_code", 0),
        str(node.get("label_name") or ""),
    )


@app.callback(
    Output("seg-workflow-draft", "data", allow_duplicate=True),
    Output("seg-workflow-status", "children", allow_duplicate=True),
    Output("seg-selected-node", "value", allow_duplicate=True),
    Input("seg-apply-node-btn", "n_clicks"),
    Input("seg-add-decision-btn", "n_clicks"),
    Input("seg-add-filter-group-btn", "n_clicks"),
    Input("seg-add-terminal-btn", "n_clicks"),
    Input("seg-add-reject-btn", "n_clicks"),
    Input("seg-remove-node-btn", "n_clicks"),
    State("seg-workflow-draft", "data"),
    State("seg-selected-node", "value"),
    State("seg-add-node-branch", "value"),
    State("seg-node-label", "value"),
    State("seg-node-type", "value"),
    State("seg-node-description", "value"),
    State("seg-node-channel", "value"),
    State("seg-node-operator", "value"),
    State("seg-node-threshold", "value"),
    State("seg-node-state-key", "value"),
    State("seg-node-label-code", "value"),
    State("seg-node-label-name", "value"),
    prevent_initial_call=True,
)
def edit_segmentation_workflow_nodes(
    _apply_clicks,
    _add_decision,
    _add_group,
    _add_terminal,
    _add_reject,
    _remove_node,
    workflow_draft,
    selected_node,
    add_branch,
    node_label,
    node_type,
    node_description,
    node_channel,
    node_operator,
    node_threshold,
    node_state_key,
    node_label_code,
    node_label_name,
):
    workflow = copy.deepcopy(workflow_draft or {})
    nodes = list(workflow.get("nodes") or [])
    edges = list(workflow.get("edges") or [])
    node_map = {str((n or {}).get("id") or ""): dict(n or {}) for n in nodes}
    trig = ctx.triggered_id
    if str(selected_node or "") not in node_map and nodes:
        selected_node = str(nodes[0].get("id") or "")
    if trig == "seg-apply-node-btn":
        node = node_map.get(str(selected_node or ""))
        if not node:
            raise dash.exceptions.PreventUpdate
        node.update(
            {
                "label": str(node_label or ""),
                "type": str(node_type or node.get("type") or "decision"),
                "description": str(node_description or ""),
                "channel": str(node_channel or ""),
                "operator": str(node_operator or "<="),
                "threshold": _coerce_float_or_none(node_threshold) if _coerce_float_or_none(node_threshold) is not None else node_threshold,
                "state_key": str(node_state_key or ""),
                "label_code": int(node_label_code or 0),
                "label_name": str(node_label_name or ""),
            }
        )
        node_map[str(selected_node)] = node
        workflow["nodes"] = list(node_map.values())
        workflow = _sync_runtime_params_from_node(workflow, node)
        return workflow, f"Updated node '{selected_node}'.", selected_node
    if trig in {"seg-add-decision-btn", "seg-add-filter-group-btn", "seg-add-terminal-btn", "seg-add-reject-btn"}:
        new_type = {
            "seg-add-decision-btn": "decision",
            "seg-add-filter-group-btn": "filter_group",
            "seg-add-terminal-btn": "terminal_state",
            "seg-add-reject-btn": "terminal_reject",
        }[trig]
        base = new_type.replace("_", "")
        idx = 1
        new_id = f"{base}_{idx}"
        while new_id in node_map:
            idx += 1
            new_id = f"{base}_{idx}"
        parent = node_map.get(str(selected_node or ""))
        parent_pos = dict((parent or {}).get("position") or {"x": 520, "y": 60})
        new_node = {
            "id": new_id,
            "type": new_type,
            "label": str(new_type).replace("_", " ").title(),
            "description": "",
            "channel": "",
            "operator": "<=",
            "threshold": 0,
            "state_key": "",
            "label_code": 0,
            "label_name": "",
            "position": {"x": float(parent_pos.get("x", 520)) + 180, "y": float(parent_pos.get("y", 60)) + 110},
        }
        nodes.append(new_node)
        if parent:
            edges.append({"source": str(selected_node), "target": new_id, "branch": str(add_branch or "yes")})
        workflow["nodes"] = nodes
        workflow["edges"] = edges
        return workflow, f"Added {new_type} node '{new_id}'.", new_id
    if trig == "seg-remove-node-btn":
        if not selected_node or str(selected_node) == "root":
            return workflow, "Root node cannot be removed.", selected_node
        workflow["nodes"] = [n for n in nodes if str((n or {}).get("id") or "") != str(selected_node)]
        workflow["edges"] = [
            e for e in edges
            if str((e or {}).get("source") or "") != str(selected_node)
            and str((e or {}).get("target") or "") != str(selected_node)
        ]
        next_selected = str((workflow["nodes"][0] or {}).get("id") or "") if workflow["nodes"] else None
        return workflow, f"Removed node '{selected_node}'.", next_selected
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("seg-source-channel", "value"),
    Output("seg-standardize", "value"),
    Output("seg-quantize-step", "value"),
    Output("seg-base-smooth-seconds", "value"),
    Output("seg-coarse-smooth-seconds", "value"),
    Output("seg-coarse-interval-threshold", "value"),
    Output("seg-coarse-resolution-threshold", "value"),
    Output("seg-sign-consistency", "value"),
    Output("seg-filter-positive-slopes", "value"),
    Output("seg-filter-negative-slopes", "value"),
    Output("seg-method-name", "value"),
    Output("seg-event-key-initial", "value"),
    Output("seg-event-key-filtered", "value"),
    Output("seg-event-key-surface-sleep", "value"),
    Output("seg-event-key-long-flat", "value"),
    Output("seg-event-key-long-drift", "value"),
    Output("seg-label-code-not-sleep", "value"),
    Output("seg-label-code-surface-sleep", "value"),
    Output("seg-label-code-long-flat", "value"),
    Output("seg-label-code-long-drift", "value"),
    Output("seg-label-name-not-sleep", "value"),
    Output("seg-label-name-surface-sleep", "value"),
    Output("seg-label-name-long-flat", "value"),
    Output("seg-label-name-long-drift", "value"),
    Output("seg-thr-dive-depth", "value"),
    Output("seg-thr-d1", "value"),
    Output("seg-thr-d2", "value"),
    Output("seg-thr-min-duration", "value"),
    Output("seg-thr-surface-duration", "value"),
    Output("seg-thr-drift-rate", "value"),
    Output("seg-thr-curvature", "value"),
    Output("seg-thr-end-flat", "value"),
    Input("seg-workflow-draft", "data"),
    prevent_initial_call=False,
)
def sync_algorithmic_form_from_workflow_draft(workflow_draft):
    algo_cfg = workflow_preset_to_algorithmic_cfg(workflow_draft or {}, fallback_cfg=_segmentation_algo_defaults)
    thr = dict(algo_cfg.get("thresholds") or {})
    event_keys = dict(algo_cfg.get("event_keys") or {})
    label_codes = dict(algo_cfg.get("label_codes") or {})
    label_names = dict(algo_cfg.get("label_names") or {})
    return (
        str(algo_cfg.get("source_channel") or "depth.depth"),
        (["on"] if bool(algo_cfg.get("standardize", True)) else []),
        algo_cfg.get("quantize_step", 1.0),
        algo_cfg.get("base_smooth_seconds", 6.0),
        algo_cfg.get("coarse_smooth_seconds", 12.0),
        algo_cfg.get("coarse_interval_threshold_s", 5.0),
        algo_cfg.get("coarse_resolution_threshold", 1.0),
        (["on"] if bool(algo_cfg.get("sign_consistency_required", True)) else []),
        (["on"] if bool(algo_cfg.get("filter_out_positive_slopes", True)) else []),
        (["on"] if bool(algo_cfg.get("filter_out_negative_slopes", False)) else []),
        str(algo_cfg.get("method_name") or "find_rest"),
        str(event_keys.get("initial") or ""),
        str(event_keys.get("filtered") or ""),
        str(event_keys.get("surface_sleep") or ""),
        str(event_keys.get("long_flat") or ""),
        str(event_keys.get("long_drift") or ""),
        int(label_codes.get("not_sleep", 0)),
        int(label_codes.get("surface_sleep", 1)),
        int(label_codes.get("long_flat", 2)),
        int(label_codes.get("long_drift", 3)),
        str(label_names.get("not_sleep") or ""),
        str(label_names.get("surface_sleep") or ""),
        str(label_names.get("long_flat") or ""),
        str(label_names.get("long_drift") or ""),
        thr.get("dive_depth_min_m", 2.0),
        thr.get("first_deriv_abs_max_ms", 0.6),
        thr.get("second_deriv_abs_max_ms2", 0.05),
        thr.get("min_duration_s", 180.0),
        thr.get("surface_sleep_min_duration_s", 600.0),
        thr.get("drift_rate_abs_max_ms", 0.30),
        thr.get("curvature_abs_max_ms", 0.10),
        thr.get("end_flat_abs_mean_d1_max_ms", 0.01),
    )


@app.callback(
    Output("seg-workflow-draft", "data", allow_duplicate=True),
    Input("seg-source-channel", "value"),
    Input("seg-standardize", "value"),
    Input("seg-quantize-step", "value"),
    Input("seg-base-smooth-seconds", "value"),
    Input("seg-coarse-smooth-seconds", "value"),
    Input("seg-coarse-interval-threshold", "value"),
    Input("seg-coarse-resolution-threshold", "value"),
    Input("seg-sign-consistency", "value"),
    Input("seg-filter-positive-slopes", "value"),
    Input("seg-filter-negative-slopes", "value"),
    Input("seg-method-name", "value"),
    Input("seg-event-key-initial", "value"),
    Input("seg-event-key-filtered", "value"),
    Input("seg-event-key-surface-sleep", "value"),
    Input("seg-event-key-long-flat", "value"),
    Input("seg-event-key-long-drift", "value"),
    Input("seg-label-code-not-sleep", "value"),
    Input("seg-label-code-surface-sleep", "value"),
    Input("seg-label-code-long-flat", "value"),
    Input("seg-label-code-long-drift", "value"),
    Input("seg-label-name-not-sleep", "value"),
    Input("seg-label-name-surface-sleep", "value"),
    Input("seg-label-name-long-flat", "value"),
    Input("seg-label-name-long-drift", "value"),
    Input("seg-thr-dive-depth", "value"),
    Input("seg-thr-d1", "value"),
    Input("seg-thr-d2", "value"),
    Input("seg-thr-min-duration", "value"),
    Input("seg-thr-surface-duration", "value"),
    Input("seg-thr-drift-rate", "value"),
    Input("seg-thr-curvature", "value"),
    Input("seg-thr-end-flat", "value"),
    State("seg-workflow-draft", "data"),
    prevent_initial_call=True,
)
def sync_workflow_draft_from_algorithmic_form(
    source_channel,
    standardize,
    quantize_step,
    base_smooth_seconds,
    coarse_smooth_seconds,
    coarse_interval_threshold_s,
    coarse_resolution_threshold,
    sign_consistency_required,
    filter_out_positive_slopes,
    filter_out_negative_slopes,
    method_name,
    event_key_initial,
    event_key_filtered,
    event_key_surface_sleep,
    event_key_long_flat,
    event_key_long_drift,
    label_code_not_sleep,
    label_code_surface_sleep,
    label_code_long_flat,
    label_code_long_drift,
    label_name_not_sleep,
    label_name_surface_sleep,
    label_name_long_flat,
    label_name_long_drift,
    dive_depth_min_m,
    first_deriv_abs_max_ms,
    second_deriv_abs_max_ms2,
    min_duration_s,
    surface_sleep_min_duration_s,
    drift_rate_abs_max_ms,
    curvature_abs_max_ms,
    end_flat_abs_mean_d1_max_ms,
    workflow_draft,
):
    workflow = copy.deepcopy(workflow_draft or {})
    runtime = _segmentation_algo_cfg_from_state(
        source_channel,
        bool(standardize),
        quantize_step,
        base_smooth_seconds,
        coarse_smooth_seconds,
        coarse_interval_threshold_s,
        coarse_resolution_threshold,
        bool(sign_consistency_required),
        bool(filter_out_positive_slopes),
        bool(filter_out_negative_slopes),
        method_name,
        event_key_initial,
        event_key_filtered,
        event_key_surface_sleep,
        event_key_long_flat,
        event_key_long_drift,
        label_code_not_sleep,
        label_code_surface_sleep,
        label_code_long_flat,
        label_code_long_drift,
        label_name_not_sleep,
        label_name_surface_sleep,
        label_name_long_flat,
        label_name_long_drift,
        dive_depth_min_m,
        first_deriv_abs_max_ms,
        second_deriv_abs_max_ms2,
        min_duration_s,
        surface_sleep_min_duration_s,
        drift_rate_abs_max_ms,
        curvature_abs_max_ms,
        end_flat_abs_mean_d1_max_ms,
    )
    current = workflow_preset_to_algorithmic_cfg(workflow, fallback_cfg=_segmentation_algo_defaults)
    if _normalize_workflow_for_compare(runtime) == _normalize_workflow_for_compare(current):
        raise dash.exceptions.PreventUpdate
    workflow["runtime_params"] = runtime
    return sync_workflow_nodes_from_runtime_params(workflow)


@app.callback(
    Output("seg-deployment-select", "options"),
    Output("seg-deployment-select", "value"),
    Input("seg-dataset-select", "value"),
    State("seg-deployment-select", "value"),
)
def update_segmentation_deployments(selected_dataset, current_deployment):
    deployments = _list_deployments(data_dir, selected_dataset)
    value = current_deployment if current_deployment in deployments else (deployments[0] if deployments else None)
    return [{"label": d, "value": d} for d in deployments], value


@app.callback(
    Output("seg-signals-select", "options"),
    Output("seg-signals-select", "value"),
    Output("seg-time-range-slider", "min"),
    Output("seg-time-range-slider", "max"),
    Output("seg-time-range-slider", "value"),
    Output("seg-time-range-slider", "marks"),
    Output("seg-deployment-status", "children"),
    Input("seg-apply-deployment", "n_clicks"),
    State("seg-dataset-select", "value"),
    State("seg-deployment-select", "value"),
    prevent_initial_call=False,
)
def apply_segmentation_deployment(_n_clicks, selected_dataset, selected_deployment):
    if not selected_dataset or not selected_deployment:
        raise dash.exceptions.PreventUpdate
    seg_data_pkl, _seg_pm, seg_tz, seg_ctx, default_selected, all_sigs = _load_segmentation_context(
        selected_dataset,
        selected_deployment,
    )
    return (
        [{"label": s, "value": s} for s in all_sigs],
        default_selected,
        seg_ctx["slider_min"],
        seg_ctx["slider_max"],
        seg_ctx["slider_default"],
        seg_ctx["slider_marks"],
        f"Loaded segmentation deployment {selected_dataset} / {selected_deployment} ({seg_tz}).",
    )


@app.callback(
    Output("seg-preview-store", "data", allow_duplicate=True),
    Output("seg-run-status", "children", allow_duplicate=True),
    Input("seg-apply-deployment", "n_clicks"),
    State("seg-dataset-select", "value"),
    State("seg-deployment-select", "value"),
    prevent_initial_call="initial_duplicate",
)
def load_saved_segmentation_results(_n_clicks, selected_dataset, selected_deployment):
    if not selected_dataset or not selected_deployment:
        raise dash.exceptions.PreventUpdate
    real_pkl_path = os.path.join(data_dir, selected_dataset, selected_deployment, "outputs", "data.pkl")
    algo_cfg = copy.deepcopy(_segmentation_algo_defaults)
    try:
        saved = load_algorithmic_results_from_real_pkl(real_pkl_path, algo_cfg=algo_cfg)
    except Exception as e:
        return {}, f"Loaded deployment, but failed to inspect saved algorithmic segmentation: {type(e).__name__}: {e}"
    if not saved.get("has_saved_results"):
        return {}, "No saved algorithmic segmentation found for this deployment. Run detection to preview new state events."
    payload = {
        "dataset_id": selected_dataset,
        "deployment_id": selected_deployment,
        "algorithmic_cfg": algo_cfg,
        "event_rows": _serialize_records_for_store(saved.get("event_rows") or []),
        "segment_rows": [],
        "label_rows": _serialize_records_for_store(saved.get("label_rows") or []),
        "signal_info": saved.get("signal_info") or {},
        "event_manager": saved.get("event_manager") or {},
        "summary": {},
        "full_run": False,
        "persisted": True,
    }
    return (
        payload,
        f"Loaded saved algorithmic segmentation from deployment: "
        f"{len(payload['event_rows'])} state events, {len(payload['label_rows'])} label rows. "
        f"These remain visible until you press Run again.",
    )


@app.callback(
    Output("seg-signals-select", "value", allow_duplicate=True),
    Input("seg-workflow-select", "value"),
    State("seg-workflow-store", "data"),
    State("seg-dataset-select", "value"),
    State("seg-deployment-select", "value"),
    prevent_initial_call="initial_duplicate",
)
def load_workflow_default_review_signals(selected_workflow, workflow_store, selected_dataset, selected_deployment):
    if not selected_dataset or not selected_deployment:
        raise dash.exceptions.PreventUpdate
    seg_data_pkl, _seg_pm, _seg_tz, _seg_ctx, _default_selected, all_sigs = _load_segmentation_context(
        selected_dataset,
        selected_deployment,
    )
    workflow = dict((workflow_store or {}).get(selected_workflow) or {})
    preferred = [str(s) for s in (workflow.get("default_review_signals") or []) if str(s) in all_sigs]
    if not preferred:
        preferred, _ = _segmentation_default_signals(seg_data_pkl)
    return preferred


@app.callback(
    Output("seg-time-summary", "children"),
    Input("seg-time-range-slider", "value"),
    State("seg-dataset-select", "value"),
    State("seg-deployment-select", "value"),
)
def summarize_segmentation_window(slider_value, selected_dataset, selected_deployment):
    if not selected_dataset or not selected_deployment or not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
        return "Select a deployment and time window."
    seg_data_pkl, _seg_pm, seg_tz, _seg_ctx, _default_selected, _all_sigs = _load_segmentation_context(
        selected_dataset,
        selected_deployment,
    )
    lo = _from_epoch_seconds(min(slider_value), seg_tz)
    hi = _from_epoch_seconds(max(slider_value), seg_tz)
    depth_sig = _segmentation_depth_target_signal(seg_data_pkl, [])
    return (
        f"Window: {lo.strftime('%Y-%m-%d %H:%M:%S')} to {hi.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"target overlay signal: {depth_sig or 'none'}"
    )


@app.callback(
    Output("seg-save-signals-status", "children"),
    Input("seg-save-signals-btn", "n_clicks"),
    State("seg-dataset-select", "value"),
    State("seg-deployment-select", "value"),
    State("seg-signals-select", "value"),
    prevent_initial_call=True,
)
def save_segmentation_signal_defaults(_n_clicks, selected_dataset, selected_deployment, selected_signals):
    if not selected_dataset or not selected_deployment:
        return "Select a dataset and deployment first."
    seg_data_pkl, seg_pm, _seg_tz, _seg_ctx, _default_selected, all_sigs = _load_segmentation_context(
        selected_dataset,
        selected_deployment,
    )
    _ = seg_data_pkl
    valid_sigs = set(str(s) for s in (all_sigs or []))
    selected = [str(s) for s in (selected_signals or []) if str(s) in valid_sigs]
    if not selected:
        return "Select at least one signal to save."
    try:
        seg_pm.set_dataset_defaults(
            entries={
                "segmentation_review_signals": selected,
                "segmentation_review_signal_order": selected,
            },
            section="segmentation_settings",
        )
    except Exception as e:
        return f"Failed to save segmentation signals: {type(e).__name__}: {e}"
    return f"Saved {len(selected)} segmentation review signal(s) under segmentation settings."


@app.callback(
    Output("seg-state-events-panel", "children"),
    Input("seg-style-event-key", "options"),
    Input("seg-style-event-key", "value"),
    Input("seg-event-style-store", "data"),
    Input("seg-preview-store", "data"),
    Input("seg-event-group-selection-store", "data"),
)
def render_segmentation_state_events_panel(event_options, selected_event, style_store, preview_store, group_selection_store):
    keys = [str((opt or {}).get("value")) for opt in (event_options or []) if str((opt or {}).get("value") or "").strip()]
    keys.extend([str(k) for k in _preview_state_keys(preview_store)])
    keys = list(dict.fromkeys([str(k).strip() for k in keys if str(k or "").strip()]))
    if not keys:
        return html.Div(
            [
                html.Div("State Events", className="legend-section-title"),
                html.Div([html.Div("No segmentation event keys configured.", className="legend-empty")], className="legend-list"),
            ],
            className="control-card legend-card",
        )
    rows = []
    grouped = group_event_keys(keys)
    group_selection_store = dict(group_selection_store or {})
    style_store = style_store or {}

    flat_keys = []
    for parent_name, child_keys in grouped.items():
        flat_keys.extend(child_keys)

    style_index = {ev: idx for idx, ev in enumerate(flat_keys)}
    for parent_name, child_keys in grouped.items():
        child_rows = []
        for ev in child_keys:
            resolved_style = _segmentation_style_entry(ev, style_store=style_store, index=style_index.get(ev, 0))
            event_color = resolved_style["color"]
            symbol_value = resolved_style["symbol"]
            glyph = EVENT_SYMBOL_GLYPHS.get(symbol_value, "•")
            row_children = [
                html.Span(
                    "",
                    className="legend-swatch legend-swatch-btn",
                    title=f"Edit event {ev}",
                    style={"backgroundColor": event_color},
                ),
                html.Span(
                    glyph,
                    className="legend-event-shape",
                    style={"color": event_color, "borderColor": event_color},
                    title=f"Select {ev}",
                ),
                html.Div(
                    [
                        html.Span(_event_label_display_text(ev), className="legend-main", title=ev),
                        html.Span("State", className="legend-sub"),
                    ],
                    className="legend-event-text",
                ),
            ]
            if resolved_style["shade_enabled"]:
                shade_color = resolved_style["shade_color"]
                row_children.insert(
                    2,
                    html.Span(
                        className="legend-shade",
                        style={"backgroundColor": shade_color},
                        title="Duration shading color",
                    ),
                )
            row_style = {
                "--event-color": event_color,
                "backgroundColor": "#123B59",
                "borderColor": "var(--event-color)",
                "borderWidth": "1px",
                "borderStyle": "solid",
                "color": "#FFFFFF",
            }
            if ev == selected_event:
                row_style["boxShadow"] = "inset 0 0 0 2px rgba(255,255,255,0.25)"
            child_rows.append(
                html.Button(
                    row_children,
                    className="chip-item legend-row legend-event-row",
                    id={"type": "seg-event-row-btn", "event": ev},
                    n_clicks=0,
                    style=dict(row_style, textAlign="left", width="100%", cursor="pointer"),
                    key=f"seg-legend-event-{ev}",
                )
            )

        parent_checked = bool(group_selection_store.get(parent_name))
        rows.append(
            html.Details(
                [
                    html.Summary(
                        [
                            html.Div(
                                [
                                    html.Span(parent_name, className="legend-group-label"),
                                    html.Span(f"{len(child_keys)} states", className="legend-group-count"),
                                ],
                                className="legend-group-left",
                            ),
                            html.Div(
                                dcc.Checklist(
                                    id={"type": "seg-event-group-write", "group": parent_name},
                                    options=[{"label": "Write group", "value": "write"}],
                                    value=["write"] if parent_checked else [],
                                    className="signal-custom-range-toggle",
                                ),
                                className="legend-actions",
                            ),
                        ],
                        className="legend-group-summary",
                    ),
                    html.Div(child_rows, className="legend-children-wrap"),
                ],
                className="legend-group legend-signal-group",
                open=True,
                key=f"seg-legend-group-{parent_name}",
            )
        )
    return html.Div(
        [
            html.Div("State Events", className="legend-section-title"),
            html.Div(rows, className="legend-list"),
        ],
        className="control-card legend-card",
    )


@app.callback(
    Output("seg-style-event-key", "options"),
    Output("seg-style-event-key", "value"),
    Input("seg-event-key-initial", "value"),
    Input("seg-event-key-filtered", "value"),
    Input("seg-event-key-surface-sleep", "value"),
    Input("seg-event-key-long-flat", "value"),
    Input("seg-event-key-long-drift", "value"),
    Input("seg-preview-store", "data"),
    State("seg-style-event-key", "value"),
)
def update_segmentation_style_event_options(initial_key, filtered_key, surface_key, flat_key, drift_key, preview_store, current_value):
    options = [str(v).strip() for v in [initial_key, filtered_key, surface_key, flat_key, drift_key] if str(v or "").strip()]
    options.extend(sorted(_preview_state_keys(preview_store), key=lambda x: x.lower()))
    options = list(dict.fromkeys(options))
    value = current_value if current_value in options else (options[0] if options else None)
    return [{"label": k, "value": k} for k in options], value


@app.callback(
    Output("seg-event-group-selection-store", "data"),
    Input({"type": "seg-event-group-write", "group": ALL}, "value"),
    State("seg-event-group-selection-store", "data"),
    prevent_initial_call=True,
)
def update_segmentation_group_selection(_all_values, current_store):
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    group_name = str(trig.get("group") or "").strip()
    if not group_name:
        raise dash.exceptions.PreventUpdate
    selected = dict(current_store or {})
    trig_value = ctx.triggered[0].get("value")
    selected[group_name] = bool(isinstance(trig_value, list) and ("write" in trig_value))
    return selected


@app.callback(
    Output("seg-style-event-key", "value", allow_duplicate=True),
    Output("color-edit-target", "data", allow_duplicate=True),
    Input({"type": "seg-event-row-btn", "event": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def activate_segmentation_event_row(_row_clicks):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    event_key = str(trig.get("event") or "")
    if not event_key:
        raise dash.exceptions.PreventUpdate
    return event_key, {"group": "segmentation_events", "event": event_key, "color_role": "marker"}


@app.callback(
    Output("seg-style-color", "value"),
    Output("seg-style-shade-mode", "value"),
    Output("seg-style-opacity", "value"),
    Input("seg-style-event-key", "value"),
    State("seg-event-style-store", "data"),
)
def load_segmentation_style_fields(event_key, style_store):
    if not event_key:
        return "#4DAEF8", "fill_trace_split", 0.25
    style = _segmentation_style_entry(str(event_key), style_store=style_store or {}, index=0)
    return (
        style["color"] or "#4DAEF8",
        style["shade_mode"],
        float(style["shade_opacity"] or 0.25),
    )


@app.callback(
    Output("seg-event-style-store", "data"),
    Output("seg-style-status", "children"),
    Input("seg-apply-style-btn", "n_clicks"),
    State("seg-style-event-key", "value"),
    State("seg-style-color", "value"),
    State("seg-style-shade-mode", "value"),
    State("seg-style-opacity", "value"),
    State("seg-event-style-store", "data"),
    prevent_initial_call=True,
)
def apply_segmentation_event_style(_n_clicks, event_key, color_value, shade_mode, shade_opacity, style_store):
    if not event_key:
        raise dash.exceptions.PreventUpdate
    updated = dict(style_store or {})
    updated[str(event_key)] = _normalize_event_style_entry(
        {
            "color": color_value,
            "shade_color": color_value,
            "shade_mode": shade_mode,
            "shade_enabled": True,
            "shade_opacity": shade_opacity,
        }
    )
    return updated, f"Updated style for {event_key}."


@app.callback(
    Output("seg-style-status", "children", allow_duplicate=True),
    Input("seg-save-style-btn", "n_clicks"),
    State("seg-event-style-store", "data"),
    prevent_initial_call=True,
)
def save_segmentation_event_styles(_n_clicks, style_store):
    existing = _load_event_styles(color_mapping_path)
    merged = dict(existing)
    merged.update(style_store or {})
    _save_event_styles(color_mapping_path, merged)
    return "Saved segmentation state event styles to color_mappings.json."


@app.callback(
    Output("seg-preview-store", "data"),
    Output("seg-run-status", "children"),
    Input("seg-preview-btn", "n_clicks"),
    Input("seg-full-btn", "n_clicks"),
    State("seg-method-tabs", "value"),
    State("seg-dataset-select", "value"),
    State("seg-deployment-select", "value"),
    State("seg-time-range-slider", "value"),
    State("seg-source-channel", "value"),
    State("seg-standardize", "value"),
    State("seg-quantize-step", "value"),
    State("seg-base-smooth-seconds", "value"),
    State("seg-coarse-smooth-seconds", "value"),
    State("seg-coarse-interval-threshold", "value"),
    State("seg-coarse-resolution-threshold", "value"),
    State("seg-sign-consistency", "value"),
    State("seg-filter-positive-slopes", "value"),
    State("seg-filter-negative-slopes", "value"),
    State("seg-method-name", "value"),
    State("seg-event-key-initial", "value"),
    State("seg-event-key-filtered", "value"),
    State("seg-event-key-surface-sleep", "value"),
    State("seg-event-key-long-flat", "value"),
    State("seg-event-key-long-drift", "value"),
    State("seg-label-code-not-sleep", "value"),
    State("seg-label-code-surface-sleep", "value"),
    State("seg-label-code-long-flat", "value"),
    State("seg-label-code-long-drift", "value"),
    State("seg-label-name-not-sleep", "value"),
    State("seg-label-name-surface-sleep", "value"),
    State("seg-label-name-long-flat", "value"),
    State("seg-label-name-long-drift", "value"),
    State("seg-thr-dive-depth", "value"),
    State("seg-thr-d1", "value"),
    State("seg-thr-d2", "value"),
    State("seg-thr-min-duration", "value"),
    State("seg-thr-surface-duration", "value"),
    State("seg-thr-drift-rate", "value"),
    State("seg-thr-curvature", "value"),
    State("seg-thr-end-flat", "value"),
    State("seg-workflow-draft", "data"),
    prevent_initial_call=True,
)
def run_segmentation_preview(
    preview_clicks,
    full_clicks,
    method_tab,
    selected_dataset,
    selected_deployment,
    slider_value,
    source_channel,
    standardize,
    quantize_step,
    base_smooth_seconds,
    coarse_smooth_seconds,
    coarse_interval_threshold_s,
    coarse_resolution_threshold,
    sign_consistency_required,
    filter_out_positive_slopes,
    filter_out_negative_slopes,
    method_name,
    event_key_initial,
    event_key_filtered,
    event_key_surface_sleep,
    event_key_long_flat,
    event_key_long_drift,
    label_code_not_sleep,
    label_code_surface_sleep,
    label_code_long_flat,
    label_code_long_drift,
    label_name_not_sleep,
    label_name_surface_sleep,
    label_name_long_flat,
    label_name_long_drift,
    dive_depth_min_m,
    first_deriv_abs_max_ms,
    second_deriv_abs_max_ms2,
    min_duration_s,
    surface_sleep_min_duration_s,
    drift_rate_abs_max_ms,
    curvature_abs_max_ms,
    end_flat_abs_mean_d1_max_ms,
    workflow_draft,
):
    trig = ctx.triggered_id
    if trig not in {"seg-preview-btn", "seg-full-btn"}:
        raise dash.exceptions.PreventUpdate
    if method_tab != "algorithmic":
        return no_update, f"{str(method_tab).title()} segmentation is scaffolded but not executable in this page yet."
    if not selected_dataset or not selected_deployment:
        return no_update, "Pick a dataset and deployment first."
    if isinstance(workflow_draft, dict) and workflow_draft:
        algo_cfg = workflow_preset_to_algorithmic_cfg(workflow_draft, fallback_cfg=_segmentation_algo_defaults)
    else:
        algo_cfg = _segmentation_algo_cfg_from_state(
            source_channel,
            ("on" in (standardize or [])),
            quantize_step,
            base_smooth_seconds,
            coarse_smooth_seconds,
            coarse_interval_threshold_s,
            coarse_resolution_threshold,
            ("on" in (sign_consistency_required or [])),
            ("on" in (filter_out_positive_slopes or [])),
            ("on" in (filter_out_negative_slopes or [])),
            method_name,
            event_key_initial,
            event_key_filtered,
            event_key_surface_sleep,
            event_key_long_flat,
            event_key_long_drift,
            label_code_not_sleep,
            label_code_surface_sleep,
            label_code_long_flat,
            label_code_long_drift,
            label_name_not_sleep,
            label_name_surface_sleep,
            label_name_long_flat,
            label_name_long_drift,
            dive_depth_min_m,
            first_deriv_abs_max_ms,
            second_deriv_abs_max_ms2,
            min_duration_s,
            surface_sleep_min_duration_s,
            drift_rate_abs_max_ms,
            curvature_abs_max_ms,
            end_flat_abs_mean_d1_max_ms,
        )
    real_pkl_path = os.path.join(data_dir, selected_dataset, selected_deployment, "outputs", "data.pkl")
    preview_start_ts = None
    preview_end_ts = None
    run_label = "full deployment"
    if trig == "seg-preview-btn":
        if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
            return no_update, "Pick a valid preview window first."
        seg_data_pkl, _seg_pm, seg_tz, _seg_ctx, _default_selected, _all_sigs = _load_segmentation_context(
            selected_dataset,
            selected_deployment,
        )
        preview_start_ts = _from_epoch_seconds(min(slider_value), seg_tz)
        preview_end_ts = _from_epoch_seconds(max(slider_value), seg_tz)
        run_label = f"preview window {preview_start_ts.strftime('%Y-%m-%d %H:%M:%S')} to {preview_end_ts.strftime('%Y-%m-%d %H:%M:%S')}"
    temp_root = None
    try:
        result = run_algorithmic_segmentation_sandbox(
            repo_root=str(_LOCAL_PYOLOGGER_ROOT),
            base_config_path=_seg_config_path,
            real_data_pkl_path=real_pkl_path,
            dataset_id=selected_dataset,
            deployment_id=selected_deployment,
            algo_cfg=algo_cfg,
            preview_start_ts=preview_start_ts,
            preview_end_ts=preview_end_ts,
        )
        temp_root = result.get("temp_root")
    except Exception as e:
        if temp_root:
            cleanup_temp_workspace(temp_root)
        return no_update, f"Segmentation run failed: {type(e).__name__}: {e}"
    payload = {
        "dataset_id": selected_dataset,
        "deployment_id": selected_deployment,
        "algorithmic_cfg": algo_cfg,
        "event_rows": _serialize_records_for_store(result.get("event_rows") or []),
        "segment_rows": _serialize_records_for_store(result.get("segment_rows") or []),
        "label_rows": _serialize_records_for_store(result.get("label_rows") or []),
        "signal_info": result.get("signal_info") or {},
        "event_manager": result.get("event_manager") or {},
        "summary": result.get("summary") or {},
        "full_run": (trig == "seg-full-btn"),
        "persisted": False,
    }
    cleanup_temp_workspace(temp_root)
    summary = payload["summary"]
    msg = (
        f"Completed algorithmic segmentation for {run_label}. "
        f"Initial={summary.get('n_initial_segments', 0)} "
        f"Filtered={summary.get('n_filtered_segments', 0)} "
        f"Surface={summary.get('n_surface_sleep_segments', 0)}."
    )
    if payload["full_run"]:
        msg += " Save is now enabled."
    else:
        msg += " You can now run the rest of the deployment."
    return payload, msg


@app.callback(
    Output("seg-full-btn", "disabled"),
    Output("seg-save-btn", "disabled"),
    Input("seg-preview-store", "data"),
)
def update_segmentation_action_state(preview_store):
    store = dict(preview_store or {})
    has_preview = bool(store.get("algorithmic_cfg")) or bool(store.get("summary"))
    full_done = bool(store.get("full_run"))
    return (not has_preview), (not full_done)


@app.callback(
    Output("seg-run-status", "children", allow_duplicate=True),
    Output("seg-preview-store", "data", allow_duplicate=True),
    Input("seg-save-btn", "n_clicks"),
    State("seg-preview-store", "data"),
    State("seg-event-group-selection-store", "data"),
    prevent_initial_call=True,
)
def save_segmentation_results(_n_clicks, preview_store, group_selection_store):
    payload = dict(preview_store or {})
    if not payload or not payload.get("full_run"):
        raise dash.exceptions.PreventUpdate

    selected_groups = [
        str(group).strip()
        for group, enabled in dict(group_selection_store or {}).items()
        if str(group).strip() and bool(enabled)
    ]
    if not selected_groups:
        return "Select at least one parent event group to write.", preview_store

    preview_events = pd.DataFrame(payload.get("event_rows") or [])
    if preview_events.empty or "key" not in preview_events.columns:
        return "No preview state events available to write.", preview_store
    event_keys = [str(v).strip() for v in preview_events["key"].astype(str).tolist() if str(v).strip()]
    target_event_keys = sorted(
        set(k for k in event_keys if event_group_parent(k) in set(selected_groups)),
        key=lambda x: x.lower(),
    )
    if not target_event_keys:
        return "No matching state events found for selected groups.", preview_store

    real_pkl_path = os.path.join(data_dir, payload["dataset_id"], payload["deployment_id"], "outputs", "data.pkl")
    dep_folder = os.path.join(data_dir, payload["dataset_id"], payload["deployment_id"])
    try:
        result = save_algorithmic_results_to_real_pkl(
            real_pkl_path,
            payload,
            target_event_keys=target_event_keys,
            write_netcdf=True,
            deployment_folder=dep_folder,
            deployment_id=payload.get("deployment_id"),
        )
    except Exception as e:
        return f"Failed to save segmentation results: {type(e).__name__}: {e}", preview_store
    saved_payload = dict(payload)
    saved_payload["persisted"] = True
    netcdf_error = str(result.get("netcdf_error") or "").strip()
    netcdf_suffix = ""
    if result.get("netcdf_written"):
        netcdf_suffix = " NetCDF updated."
    elif netcdf_error:
        netcdf_suffix = f" data_pkl saved, but NetCDF update failed ({netcdf_error})."
    return (
        f"Saved selected state groups ({len(selected_groups)} group(s), {len(target_event_keys)} event key(s)) to deployment "
        f"(removed {result['removed_event_count']} old events, saved {result['saved_event_count']} events and "
        f"{result['saved_label_rows']} label rows).{netcdf_suffix}",
        saved_payload,
    )


@app.callback(
    Output("seg-plot", "figure"),
    Input("seg-signals-select", "value"),
    Input("seg-event-style-store", "data"),
    Input("seg-preview-store", "data"),
    Input("seg-time-range-slider", "value"),
    State("seg-dataset-select", "value"),
    State("seg-deployment-select", "value"),
)
def update_segmentation_plot(selected_signals, style_store, preview_store, slider_value, selected_dataset, selected_deployment):
    if not selected_dataset or not selected_deployment:
        return BasePlotlyFigure()
    seg_data_pkl, _seg_pm, seg_tz, seg_ctx, default_selected, _all_sigs = _load_segmentation_context(
        selected_dataset,
        selected_deployment,
    )
    chosen_signals = [s for s in (selected_signals or []) if s in seg_data_pkl.signal_data]
    if not chosen_signals:
        chosen_signals = default_selected
    target_sig = _segmentation_depth_target_signal(seg_data_pkl, chosen_signals)
    if target_sig and target_sig not in chosen_signals:
        chosen_signals = [target_sig] + [s for s in chosen_signals if s != target_sig]
    channels = {sig: _signal_channels(seg_data_pkl, sig) for sig in chosen_signals}
    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        start_ts = _from_epoch_seconds(min(slider_value), seg_tz)
        end_ts = _from_epoch_seconds(max(slider_value), seg_tz)
    else:
        start_ts = seg_ctx["start_default"]
        end_ts = seg_ctx["end_default"]
    data_for_plot = copy.copy(seg_data_pkl)
    base_events = getattr(seg_data_pkl, "event_data", None)
    if not isinstance(base_events, pd.DataFrame):
        base_events = pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value"])
    preview_df = _enrich_preview_event_rows_with_segment_bounds(preview_store)
    preview_keys = set()
    if not preview_df.empty and "datetime" in preview_df.columns:
        preview_keys = set(preview_df["key"].astype(str).tolist()) if "key" in preview_df.columns else set()
        keep = ~base_events["key"].astype(str).isin(preview_keys) if "key" in base_events.columns else pd.Series([True] * len(base_events))
        data_for_plot.event_data = pd.concat([base_events.loc[keep].copy(), preview_df], ignore_index=True)
    else:
        data_for_plot.event_data = base_events
    target_channel = _segmentation_depth_target_channel(seg_data_pkl, target_sig)
    state_notes = {}
    event_candidates = sorted(preview_keys)
    preview_state = _preview_state_keys(preview_store)
    for idx, event_key in enumerate([k for k in event_candidates if (k in preview_state) or _has_state_duration_events(data_for_plot, k)]):
        style = dict((style_store or {}).get(event_key) or {})
        color = _normalize_hex_color(style.get("color")) or _mapped_event_color(event_key) or _default_event_style(event_key, idx)["color"]
        opacity = float(style.get("shade_opacity", 0.25) or 0.25)
        state_notes[event_key] = [{
            "signal": target_sig,
            "channel": target_channel,
            "color": _hex_to_rgba(color, opacity) or color,
            "shade_mode": _normalize_event_shade_mode(style.get("shade_mode"), default="fill_trace_split"),
            "draw_line": False,
            "start_marker": False,
            "legend_key": event_key,
            "name": event_key,
        }]
    fig = plot_tag_data_interactive(
        data_pkl=data_for_plot,
        signals=chosen_signals,
        channels=channels,
        time_range=(start_ts, end_ts),
        note_annotations=None,
        state_annotations=state_notes or None,
        color_mapping_path=color_mapping_path,
        include_blank_row=False,
        preserve_signal_order=True,
        signal_metadata=_signal_display_metadata,
        color_mapping=_signal_colors,
        persist_color_mapping=False,
        zoom_start_time=start_ts,
        zoom_end_time=end_ts,
        zoom_range_selector_channel=(target_sig or chosen_signals[0]),
        plot_event_values=[],
    )
    fig.update_layout(showlegend=False, hovermode="x unified")
    return fig


@app.callback(
    Output("peak-mode", "value", allow_duplicate=True),
    Output("peak-status", "children", allow_duplicate=True),
    Output("signals-select", "options", allow_duplicate=True),
    Output("signals-select", "value", allow_duplicate=True),
    Output("selected-signals-store", "data", allow_duplicate=True),
    Output("ordered-signals-store", "data", allow_duplicate=True),
    Output("channels-store", "data", allow_duplicate=True),
    Output("channel-order-store", "data", allow_duplicate=True),
    Output("peak-detect-preview-store", "data", allow_duplicate=True),
    Output("peak-detect-progress-store", "data", allow_duplicate=True),
    Output("peak-view-active-store", "data", allow_duplicate=True),
    Input("peak-detect-heart-btn", "n_clicks"),
    Input("peak-detect-stroke-btn", "n_clicks"),
    State("peak-parent-signal", "value"),
    State("peak-channel", "value"),
    State("time-range-slider", "value"),
    State("peak-detection-sources", "value"),
    State("peak-broad-low", "value"),
    State("peak-broad-high", "value"),
    State("peak-narrow-low", "value"),
    State("peak-narrow-high", "value"),
    State("peak-filter-order", "value"),
    State("peak-spike-threshold", "value"),
    State("peak-smooth-sec", "value"),
    State("peak-window-mult", "value"),
    State("peak-norm-noise", "value"),
    State("peak-height", "value"),
    State("peak-distance", "value"),
    State("peak-search-radius", "value"),
    State("peak-min-height", "value"),
    State("peak-max-height", "value"),
    State("peak-enable-bandpass", "value"),
    State("peak-enable-spike", "value"),
    State("peak-enable-absolute", "value"),
    State("peak-enable-smoothing", "value"),
    State("peak-enable-normalization", "value"),
    State("peak-enable-refinement", "value"),
    State("peak-hr-jump-frac", "value"),
    State("peak-min-rr", "value"),
    State("peak-max-hr", "value"),
    State("peak-min-hr", "value"),
    State("peak-anti-double-gap", "value"),
    State("peak-anti-double-window", "value"),
    State("peak-pick-last-conflict", "value"),
    prevent_initial_call=True,
)
def enter_peak_mode(
    _heart_clicks,
    _stroke_clicks,
    parent_signal,
    channel,
    slider_value,
    detection_sources,
    broad_low,
    broad_high,
    narrow_low,
    narrow_high,
    filter_order,
    spike_threshold,
    smooth_sec,
    window_mult,
    norm_noise,
    peak_height,
    peak_distance,
    search_radius,
    min_peak_height,
    max_peak_height,
    enable_bandpass,
    enable_spike,
    enable_absolute,
    enable_smoothing,
    enable_normalization,
    enable_refinement,
    hr_jump_frac,
    min_rr,
    max_hr,
    min_hr,
    anti_double_gap,
    anti_double_window,
    pick_last_conflict,
):
    trig = ctx.triggered_id
    if trig == "peak-detect-stroke-btn":
        mode = "stroke_rate"
    else:
        mode = "heart_rate"
    if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
        return mode, "Enter peak mode failed: invalid active time window.", no_update, no_update, no_update, no_update, no_update, {}, {}, {"active": False, "mode": mode}
    lo = float(min(slider_value))
    hi = float(max(slider_value))
    if not parent_signal or not channel:
        return mode, "Enter peak mode failed: select parent signal + channel.", no_update, no_update, no_update, no_update, no_update, {}, {}, {"active": False, "mode": mode}

    params = {
        "enable_bandpass": ("on" in (enable_bandpass or [])),
        "enable_spike_removal": ("on" in (enable_spike or [])),
        "enable_absolute": ("on" in (enable_absolute or [])),
        "enable_smoothing": ("on" in (enable_smoothing or [])),
        "enable_normalization": ("on" in (enable_normalization or [])),
        "enable_refinement": ("on" in (enable_refinement or [])),
    }
    subset, err = _peak_prepare_signal_subset(
        parent_signal,
        channel,
        _from_epoch_seconds(lo, tz_name),
        _from_epoch_seconds(hi, tz_name),
    )
    if err:
        return mode, f"Enter peak mode failed: {err}", no_update, no_update, no_update, no_update, no_update, {}, {}, {"active": False, "mode": mode}
    fs = float(calculate_sampling_frequency(subset["datetime"]))
    if not np.isfinite(fs) or fs <= 0:
        return mode, "Enter peak mode failed: sampling rate invalid for selected signal.", no_update, no_update, no_update, no_update, no_update, {}, {}, {"active": False, "mode": mode}

    try:
        results = peak_detect(
            signal=subset[channel].to_numpy(),
            sampling_rate=fs,
            datetime_series=subset["datetime"],
            broad_lowcut=float(broad_low),
            broad_highcut=float(broad_high),
            narrow_lowcut=float(narrow_low),
            narrow_highcut=float(narrow_high),
            filter_order=int(filter_order),
            spike_threshold=float(spike_threshold),
            smooth_sec_multiplier=float(smooth_sec),
            window_size_multiplier=float(window_mult),
            normalization_noise=float(norm_noise),
            peak_height=float(peak_height),
            peak_distance_sec=float(peak_distance),
            search_radius_sec=float(search_radius),
            min_peak_height=float(min_peak_height),
            max_peak_height=float(max_peak_height),
            enable_bandpass=params["enable_bandpass"],
            enable_spike_removal=params["enable_spike_removal"],
            enable_absolute=params["enable_absolute"],
            enable_smoothing=params["enable_smoothing"],
            enable_normalization=params["enable_normalization"],
            enable_refinement=params["enable_refinement"],
            detection_sources=list(detection_sources or ["normalized"]),
        )
    except Exception as e:
        return mode, f"Enter peak mode failed: {type(e).__name__}: {e}", no_update, no_update, no_update, no_update, no_update, {}, {}, {"active": False, "mode": mode}

    saved_now = _peak_upsert_intermediate_signals(mode, parent_signal, subset, params, results)
    all_signals = sorted(set(_plot_signal_allowlist(data_pkl)) | set(str(s) for s in getattr(data_pkl, "signal_data", {}).keys()))
    defaults_blob = _load_peak_detect_view_defaults(color_mapping_path)
    mode_blob = dict((defaults_blob.get(mode) or {})) if isinstance(defaults_blob, dict) else {}
    base_signals = _peak_default_plot_signals(mode, parent_signal)
    saved_signals = [str(s) for s in (mode_blob.get("signals") or []) if str(s) in all_signals]
    selected_signals = saved_signals or base_signals or ([all_signals[0]] if all_signals else [])
    ordered = [str(s) for s in (mode_blob.get("order") or []) if str(s) in selected_signals]
    ordered_signals = ordered + [s for s in selected_signals if s not in ordered]

    saved_channels = mode_blob.get("channels") or {}
    saved_channel_order = mode_blob.get("channel_order") or {}
    channels_store = {}
    channel_order_store = {}
    for sig in ordered_signals:
        options = _signal_channels(data_pkl, sig)
        selected = [c for c in (saved_channels.get(sig) or []) if c in options] or _signal_default_channels(sig)
        channels_store[sig] = selected
        channel_order_store[sig] = _normalize_channel_order(selected, saved_channel_order.get(sig))

    _save_peak_detect_view_defaults(
        color_mapping_path,
        mode,
        {
            "signals": selected_signals,
            "order": ordered_signals,
            "channels": channels_store,
            "channel_order": channel_order_store,
        },
    )
    events = _peak_events_from_results(results, subset, mode, cleanup_diag=None)
    for row in events:
        dt = row.get("datetime")
        row["datetime"] = pd.Timestamp(dt).isoformat() if dt is not None and not pd.isna(dt) else None
    run_sig = json.dumps(
        _peak_run_signature_dict(
            mode, parent_signal, channel, detection_sources,
            broad_low, broad_high, narrow_low, narrow_high,
            filter_order, spike_threshold, smooth_sec, window_mult, norm_noise,
            peak_height, peak_distance, search_radius, min_peak_height, max_peak_height,
            enable_bandpass, enable_spike, enable_absolute, enable_smoothing, enable_normalization, enable_refinement,
            hr_jump_frac, min_rr, max_hr, min_hr, anti_double_gap, anti_double_window, pick_last_conflict,
        ),
        sort_keys=True,
    )
    preview_out = {
        "mode": mode,
        "events": events,
        "parent_signal": parent_signal,
        "channel": channel,
    }
    progress_out = {
        "next_start_epoch": float(hi),
        "chunks_completed": 1,
        "window_seconds": max(2.0, hi - lo),
        "finished": bool(float(hi) >= float(slider_max)),
        "mode": mode,
        "run_signature": run_sig,
    }
    msg = (
        f"Entered peak detection mode ({mode.replace('_', ' ')}). "
        f"Generated/updated {len(saved_now)} intermediate signals, previewed current window, "
        f"and detected {len(events)} peaks."
    )
    return (
        mode,
        msg,
        [{"label": s, "value": s} for s in all_signals],
        selected_signals,
        selected_signals,
        ordered_signals,
        channels_store,
        channel_order_store,
        preview_out,
        progress_out,
        {"active": True, "mode": mode},
    )


@app.callback(
    Output("peak-controls-fieldset", "disabled"),
    Output("peak-controls-fieldset-extra", "disabled"),
    Input("peak-view-active-store", "data"),
)
def toggle_peak_control_enable(active_store):
    active = bool((active_store or {}).get("active"))
    disabled = not active
    return disabled, disabled


@app.callback(
    Output("peak-expand-collapse-btn", "children"),
    Output("peak-param-open-store", "data"),
    Input("peak-expand-collapse-btn", "n_clicks"),
    State("peak-param-order-store", "data"),
    prevent_initial_call=False,
)
def toggle_all_peak_param_cards(n_clicks, order_store):
    ordered = _normalize_peak_param_order(order_store)
    if not ordered:
        return "Expand all", {}
    clicks = int(n_clicks or 0)
    open_all = (clicks % 2 == 1)
    return ("Collapse all" if open_all else "Expand all"), {pid: open_all for pid in ordered}


@app.callback(
    Output({"type": "peak-param-header-value", "index": ALL}, "children"),
    *[Input(param_id, "value") for param_id in PEAK_PARAM_IDS],
    State({"type": "peak-param-header-value", "index": ALL}, "id"),
)
def sync_peak_param_header_values(*args):
    if not args:
        return []
    header_ids = args[-1] or []
    values = args[:-1]
    value_map = {pid: values[idx] for idx, pid in enumerate(PEAK_PARAM_IDS) if idx < len(values)}
    out = []
    for hid in header_ids:
        pid = str((hid or {}).get("index") or "")
        v = value_map.get(pid)
        try:
            out.append(_format_peak_ref_number(float(v)))
        except Exception:
            out.append(str(v))
    return out


@app.callback(
    Output("peak-param-open-store", "data", allow_duplicate=True),
    Input({"type": "peak-param-card", "index": ALL}, "open"),
    State({"type": "peak-param-card", "index": ALL}, "id"),
    State("peak-param-open-store", "data"),
    prevent_initial_call=True,
)
def sync_peak_param_open_store(open_values, open_ids, current_store):
    out = dict(current_store or {})
    changed = False
    for is_open, cid in zip(open_values or [], open_ids or []):
        pid = str((cid or {}).get("index") or "")
        if not pid:
            continue
        val = bool(is_open)
        if out.get(pid) != val:
            out[pid] = val
            changed = True
    if not changed:
        raise dash.exceptions.PreventUpdate
    return out


@app.callback(
    Output("peak-params-editor", "children"),
    Input("peak-param-order-store", "data"),
    Input("peak-param-open-store", "data"),
)
def render_peak_param_editor(order_store, open_store):
    return _render_peak_param_cards(order_store, open_store)


@app.callback(
    Output("peak-param-order-store", "data", allow_duplicate=True),
    Input("chip-order-updates", "value"),
    State("peak-param-order-store", "data"),
    prevent_initial_call=True,
)
def apply_peak_param_chip_order(chip_order_json, current_order):
    try:
        payload = json.loads(chip_order_json or "{}")
    except Exception:
        raise dash.exceptions.PreventUpdate
    order_payload = (((payload.get("peak_params") or {}).get("__all__")) or [])
    if not order_payload:
        raise dash.exceptions.PreventUpdate
    normalized_current = _normalize_peak_param_order(current_order)
    ordered = [v for v in order_payload if v in normalized_current]
    if not ordered:
        raise dash.exceptions.PreventUpdate
    rem = [v for v in normalized_current if v not in ordered]
    return _normalize_peak_param_order(ordered + rem)


@app.callback(
    Output("deployment-load-status", "children", allow_duplicate=True),
    Input("peak-param-order-store", "data"),
    State("deployment-load-status", "children"),
    prevent_initial_call=True,
)
def persist_peak_param_order(order_store, current_status):
    ok = _persist_peak_param_order_dataset_defaults(param_manager, order_store)
    if not ok:
        raise dash.exceptions.PreventUpdate
    return "Saved peak parameter order to dataset defaults."


@app.callback(
    Output("peak-view-active-store", "data", allow_duplicate=True),
    Input("ordered-signals-store", "data"),
    Input("channels-store", "data"),
    Input("channel-order-store", "data"),
    Input("signals-select", "value"),
    State("peak-view-active-store", "data"),
    prevent_initial_call=True,
)
def persist_peak_mode_layout(ordered_signals, channels_store, channel_order_store, selected_signals, active_store):
    active = dict(active_store or {})
    if not bool(active.get("active")):
        raise dash.exceptions.PreventUpdate
    mode = str(active.get("mode") or "heart_rate").strip().lower()
    selected = [str(s) for s in (selected_signals or []) if str(s).strip()]
    ordered = [str(s) for s in (ordered_signals or []) if str(s) in selected]
    ordered = ordered + [s for s in selected if s not in ordered]
    channels = {}
    orders = {}
    channels_store = channels_store or {}
    channel_order_store = channel_order_store or {}
    for sig in ordered:
        opts = _signal_channels(data_pkl, sig)
        chosen = [str(c) for c in (channels_store.get(sig) or []) if str(c) in opts]
        if not chosen:
            chosen = _signal_default_channels(sig)
        channels[sig] = chosen
        orders[sig] = _normalize_channel_order(chosen, channel_order_store.get(sig))
    _save_peak_detect_view_defaults(
        color_mapping_path,
        mode,
        {
            "signals": selected,
            "order": ordered,
            "channels": channels,
            "channel_order": orders,
        },
    )
    return no_update


@app.callback(
    Output("peak-broad-low", "value"),
    Output("peak-broad-high", "value"),
    Output("peak-narrow-low", "value"),
    Output("peak-narrow-high", "value"),
    Output("peak-filter-order", "value"),
    Output("peak-spike-threshold", "value"),
    Output("peak-smooth-sec", "value"),
    Output("peak-window-mult", "value"),
    Output("peak-norm-noise", "value"),
    Output("peak-height", "value"),
    Output("peak-distance", "value"),
    Output("peak-search-radius", "value"),
    Output("peak-min-height", "value"),
    Output("peak-max-height", "value"),
    Output("peak-hr-jump-frac", "value"),
    Output("peak-min-rr", "value"),
    Output("peak-max-hr", "value"),
    Output("peak-min-hr", "value"),
    Output("peak-anti-double-gap", "value"),
    Output("peak-anti-double-window", "value"),
    Output("peak-broad-low-input", "value"),
    Output("peak-broad-high-input", "value"),
    Output("peak-narrow-low-input", "value"),
    Output("peak-narrow-high-input", "value"),
    Output("peak-filter-order-input", "value"),
    Output("peak-spike-threshold-input", "value"),
    Output("peak-smooth-sec-input", "value"),
    Output("peak-window-mult-input", "value"),
    Output("peak-norm-noise-input", "value"),
    Output("peak-height-input", "value"),
    Output("peak-distance-input", "value"),
    Output("peak-search-radius-input", "value"),
    Output("peak-min-height-input", "value"),
    Output("peak-max-height-input", "value"),
    Output("peak-hr-jump-frac-input", "value"),
    Output("peak-min-rr-input", "value"),
    Output("peak-max-hr-input", "value"),
    Output("peak-min-hr-input", "value"),
    Output("peak-anti-double-gap-input", "value"),
    Output("peak-anti-double-window-input", "value"),
    Output("peak-enable-bandpass", "value"),
    Output("peak-enable-spike", "value"),
    Output("peak-enable-absolute", "value"),
    Output("peak-enable-smoothing", "value"),
    Output("peak-enable-normalization", "value"),
    Output("peak-enable-refinement", "value"),
    Output("peak-pick-last-conflict", "value"),
    Output("peak-detection-sources", "value"),
    Input("peak-mode", "value"),
    Input("current-deployment", "data"),
)
def update_peak_defaults(mode, _deployment):
    d = _peak_defaults_for_mode(mode)
    on = lambda k: (["on"] if bool(d.get(k)) else [])
    return (
        d["BROAD_LOW_CUTOFF"],
        d["BROAD_HIGH_CUTOFF"],
        d["NARROW_LOW_CUTOFF"],
        d["NARROW_HIGH_CUTOFF"],
        d["FILTER_ORDER"],
        d["SPIKE_THRESHOLD"],
        d["SMOOTH_SEC_MULTIPLIER"],
        d["WINDOW_SIZE_MULTIPLIER"],
        d["NORMALIZATION_NOISE"],
        d["PEAK_HEIGHT"],
        d["PEAK_DISTANCE_SEC"],
        d["SEARCH_RADIUS_SEC"],
        d["MIN_PEAK_HEIGHT"],
        d["MAX_PEAK_HEIGHT"],
        d["HR_JUMP_FRAC"],
        d["MIN_RR_SEC"],
        d["MAX_HR_BPM"],
        d["MIN_HR_BPM"],
        d["ANTI_DOUBLE_GAP_FACTOR"],
        d["ANTI_DOUBLE_ROLLING_WINDOW_SEC"],
        d["BROAD_LOW_CUTOFF"],
        d["BROAD_HIGH_CUTOFF"],
        d["NARROW_LOW_CUTOFF"],
        d["NARROW_HIGH_CUTOFF"],
        d["FILTER_ORDER"],
        d["SPIKE_THRESHOLD"],
        d["SMOOTH_SEC_MULTIPLIER"],
        d["WINDOW_SIZE_MULTIPLIER"],
        d["NORMALIZATION_NOISE"],
        d["PEAK_HEIGHT"],
        d["PEAK_DISTANCE_SEC"],
        d["SEARCH_RADIUS_SEC"],
        d["MIN_PEAK_HEIGHT"],
        d["MAX_PEAK_HEIGHT"],
        d["HR_JUMP_FRAC"],
        d["MIN_RR_SEC"],
        d["MAX_HR_BPM"],
        d["MIN_HR_BPM"],
        d["ANTI_DOUBLE_GAP_FACTOR"],
        d["ANTI_DOUBLE_ROLLING_WINDOW_SEC"],
        on("enable_bandpass"),
        on("enable_spike_removal"),
        on("enable_absolute"),
        on("enable_smoothing"),
        on("enable_normalization"),
        on("enable_refinement"),
        on("PICK_LAST_IN_CONFLICT_PAIR"),
        d["DETECTION_DERIVATIVE_CHANNELS"],
    )


@app.callback(
    *[Output(pid, "value", allow_duplicate=True) for pid in PEAK_PARAM_IDS],
    *[Input(f"{pid}-input", "value") for pid in PEAK_PARAM_IDS],
    *[State(pid, "value") for pid in PEAK_PARAM_IDS],
    prevent_initial_call=True,
)
def sync_peak_slider_from_inputs(*args):
    n = len(PEAK_PARAM_IDS)
    input_vals = list(args[:n])
    current_vals = list(args[n:])
    trig = str(ctx.triggered_id or "")
    if not trig.endswith("-input"):
        raise dash.exceptions.PreventUpdate
    target = trig[:-6]
    if target not in PEAK_PARAM_IDS:
        raise dash.exceptions.PreventUpdate
    idx = PEAK_PARAM_IDS.index(target)
    raw = input_vals[idx]
    meta = PEAK_PARAM_UI_META.get(target) or {}
    lo, hi = _peak_slider_bounds(target, current_vals[idx])
    step = meta.get("step", 0.01)
    try:
        val = float(raw)
    except Exception:
        raise dash.exceptions.PreventUpdate
    val = max(float(lo), min(float(hi), float(val)))
    if step == 1:
        val = float(int(round(val)))
    out = [no_update] * n
    out[idx] = val
    return tuple(out)


@app.callback(
    Output("peak-parent-signal", "options"),
    Output("peak-parent-signal", "value"),
    Input("peak-mode", "value"),
    Input("current-deployment", "data"),
)
def update_peak_parent_options(mode, _deployment):
    mode = str(mode or "heart_rate").strip().lower()
    all_sigs = [str(s) for s in getattr(data_pkl, "signal_data", {}).keys()]
    detection_defaults = _get_detection_channel_defaults()
    if mode == "heart_rate":
        saved_parent = str(detection_defaults.get("heart_parent") or "").strip()
        candidates = [saved_parent, "ecg", "corrected_gyr", "gyroscope", "accelerometer", "dynamic_accel", "corrected_acc", "calibrated_acc"]
    else:
        saved_parent = str(detection_defaults.get("stroke_parent") or "").strip()
        candidates = ["dynamic_accel", "corrected_gyr", "gyroscope", "corrected_acc", "calibrated_acc", "accelerometer"]
        if saved_parent:
            candidates = [saved_parent] + candidates
    ordered = [s for s in candidates if s in all_sigs] + [s for s in all_sigs if s not in candidates]
    if not ordered:
        return [], None
    return [{"label": s, "value": s} for s in ordered], ordered[0]


@app.callback(
    Output("peak-channel", "options"),
    Output("peak-channel", "value"),
    Input("peak-parent-signal", "value"),
    Input("peak-mode", "value"),
    Input("current-deployment", "data"),
)
def update_peak_channel_options(parent_signal, mode, _deployment):
    if not parent_signal:
        return [], None
    options = _signal_channels(data_pkl, parent_signal)
    if not options:
        return [], None
    mode = str(mode or "heart_rate").strip().lower()
    detection_defaults = _get_detection_channel_defaults()
    saved_channel = str(
        detection_defaults.get("heart_channel") if mode == "heart_rate" else detection_defaults.get("stroke_channel")
    ) if detection_defaults else ""
    saved_channel = saved_channel.strip()
    preferred = None
    if saved_channel:
        preferred = next((c for c in options if _channel_base(c).lower() == _channel_base(saved_channel).lower()), None)
    if preferred is None:
        preferred = next((c for c in ["ecg", "z", "gz", "az", "x", "y"] if c in options), options[0])
    return [{"label": c, "value": c} for c in options], preferred


@app.callback(
    Output("peak-next-btn", "disabled"),
    Output("peak-save-btn", "disabled"),
    Input("peak-detect-preview-store", "data"),
    Input("peak-detect-progress-store", "data"),
    Input("peak-mode", "value"),
)
def update_peak_action_button_state(preview_store, progress_store, mode):
    preview_store = dict(preview_store or {})
    progress_store = dict(progress_store or {})
    current_mode = str(mode or "heart_rate").strip().lower()
    preview_mode = str(preview_store.get("mode") or "").strip().lower()
    progress_mode = str(progress_store.get("mode") or "").strip().lower()
    chunks_completed = int(progress_store.get("chunks_completed", 0) or 0)
    finished = bool(progress_store.get("finished", False))
    has_preview = (
        chunks_completed > 0
        and preview_mode == current_mode
        and progress_mode == current_mode
    )
    return (not has_preview) or finished, (not has_preview)


def _peak_run_signature_dict(
    current_mode,
    parent_signal,
    channel,
    detection_sources,
    broad_low,
    broad_high,
    narrow_low,
    narrow_high,
    filter_order,
    spike_threshold,
    smooth_sec,
    window_mult,
    norm_noise,
    peak_height,
    peak_distance,
    search_radius,
    min_peak_height,
    max_peak_height,
    enable_bandpass,
    enable_spike,
    enable_absolute,
    enable_smoothing,
    enable_normalization,
    enable_refinement,
    hr_jump_frac,
    min_rr,
    max_hr,
    min_hr,
    anti_double_gap,
    anti_double_window,
    pick_last_conflict,
):
    return {
        "mode": str(current_mode or "heart_rate").strip().lower(),
        "parent_signal": str(parent_signal),
        "channel": str(channel),
        "detection_sources": sorted([str(s) for s in (detection_sources or [])]),
        "broad_low": broad_low,
        "broad_high": broad_high,
        "narrow_low": narrow_low,
        "narrow_high": narrow_high,
        "filter_order": filter_order,
        "spike_threshold": spike_threshold,
        "smooth_sec": smooth_sec,
        "window_mult": window_mult,
        "norm_noise": norm_noise,
        "peak_height": peak_height,
        "peak_distance": peak_distance,
        "search_radius": search_radius,
        "min_peak_height": min_peak_height,
        "max_peak_height": max_peak_height,
        "enable_bandpass": bool(enable_bandpass),
        "enable_spike": bool(enable_spike),
        "enable_absolute": bool(enable_absolute),
        "enable_smoothing": bool(enable_smoothing),
        "enable_normalization": bool(enable_normalization),
        "enable_refinement": bool(enable_refinement),
        "hr_jump_frac": hr_jump_frac,
        "min_rr": min_rr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "anti_double_gap": anti_double_gap,
        "anti_double_window": anti_double_window,
        "pick_last_conflict": bool(pick_last_conflict),
    }


@app.callback(
    Output("peak-detect-preview-store", "data"),
    Output("peak-detect-progress-store", "data"),
    Output("peak-status", "children"),
    Input("peak-preview-btn", "n_clicks"),
    Input("peak-next-btn", "n_clicks"),
    State("peak-detect-preview-store", "data"),
    State("peak-detect-progress-store", "data"),
    State("peak-mode", "value"),
    State("peak-parent-signal", "value"),
    State("peak-channel", "value"),
    State("time-range-slider", "value"),
    State("peak-detection-sources", "value"),
    State("peak-broad-low", "value"),
    State("peak-broad-high", "value"),
    State("peak-narrow-low", "value"),
    State("peak-narrow-high", "value"),
    State("peak-filter-order", "value"),
    State("peak-spike-threshold", "value"),
    State("peak-smooth-sec", "value"),
    State("peak-window-mult", "value"),
    State("peak-norm-noise", "value"),
    State("peak-height", "value"),
    State("peak-distance", "value"),
    State("peak-search-radius", "value"),
    State("peak-min-height", "value"),
    State("peak-max-height", "value"),
    State("peak-enable-bandpass", "value"),
    State("peak-enable-spike", "value"),
    State("peak-enable-absolute", "value"),
    State("peak-enable-smoothing", "value"),
    State("peak-enable-normalization", "value"),
    State("peak-enable-refinement", "value"),
    State("peak-hr-jump-frac", "value"),
    State("peak-min-rr", "value"),
    State("peak-max-hr", "value"),
    State("peak-min-hr", "value"),
    State("peak-anti-double-gap", "value"),
    State("peak-anti-double-window", "value"),
    State("peak-pick-last-conflict", "value"),
    prevent_initial_call=True,
)
def run_peak_preview(
    preview_clicks,
    next_clicks,
    preview_store,
    progress_store,
    mode,
    parent_signal,
    channel,
    slider_value,
    detection_sources,
    broad_low,
    broad_high,
    narrow_low,
    narrow_high,
    filter_order,
    spike_threshold,
    smooth_sec,
    window_mult,
    norm_noise,
    peak_height,
    peak_distance,
    search_radius,
    min_peak_height,
    max_peak_height,
    enable_bandpass,
    enable_spike,
    enable_absolute,
    enable_smoothing,
    enable_normalization,
    enable_refinement,
    hr_jump_frac,
    min_rr,
    max_hr,
    min_hr,
    anti_double_gap,
    anti_double_window,
    pick_last_conflict,
):
    trig = ctx.triggered_id
    if trig not in {"peak-preview-btn", "peak-next-btn"}:
        raise dash.exceptions.PreventUpdate
    if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
        return preview_store or {}, progress_store or {}, "Invalid active window."
    if not parent_signal or not channel:
        return preview_store or {}, progress_store or {}, "Pick a parent signal and channel first."

    lo = float(min(slider_value))
    hi = float(max(slider_value))
    window_seconds = max(2.0, hi - lo)

    preview_store = dict(preview_store or {})
    progress_store = dict(progress_store or {})
    current_mode = str(mode or "heart_rate").strip().lower()
    run_signature = json.dumps(
        _peak_run_signature_dict(
            current_mode, parent_signal, channel, detection_sources,
            broad_low, broad_high, narrow_low, narrow_high,
            filter_order, spike_threshold, smooth_sec, window_mult, norm_noise,
            peak_height, peak_distance, search_radius, min_peak_height, max_peak_height,
            enable_bandpass, enable_spike, enable_absolute, enable_smoothing, enable_normalization, enable_refinement,
            hr_jump_frac, min_rr, max_hr, min_hr, anti_double_gap, anti_double_window, pick_last_conflict,
        ),
        sort_keys=True,
    )

    if trig == "peak-preview-btn":
        ranges = [(lo, hi)]
        existing_events = []
        chunks_completed = 0
    else:
        progress_mode = str(progress_store.get("mode") or "").strip().lower()
        progress_sig = str(progress_store.get("run_signature") or "")
        chunks_completed = int(progress_store.get("chunks_completed", 0) or 0)
        if chunks_completed < 1 or progress_mode != current_mode or progress_sig != run_signature:
            return (
                preview_store,
                progress_store,
                "Preview current window first for the current mode/settings.",
            )
        start_epoch = float(progress_store.get("next_start_epoch", hi))
        existing_events = list(preview_store.get("events") or [])
        if start_epoch >= float(slider_max):
            return preview_store, progress_store, "No remaining deployment range to calculate."
        ranges = []
        cursor = start_epoch
        deployment_end = float(slider_max)
        while cursor < deployment_end:
            chunk_end = min(cursor + window_seconds, deployment_end)
            ranges.append((cursor, chunk_end))
            if chunk_end <= cursor:
                break
            cursor = chunk_end

    params = {
        "broad_lowcut": float(broad_low),
        "broad_highcut": float(broad_high),
        "narrow_lowcut": float(narrow_low),
        "narrow_highcut": float(narrow_high),
        "filter_order": int(filter_order),
        "spike_threshold": float(spike_threshold),
        "smooth_sec_multiplier": float(smooth_sec),
        "window_size_multiplier": float(window_mult),
        "normalization_noise": float(norm_noise),
        "peak_height": float(peak_height),
        "peak_distance_sec": float(peak_distance),
        "search_radius_sec": float(search_radius),
        "min_peak_height": float(min_peak_height),
        "max_peak_height": float(max_peak_height),
        "enable_bandpass": ("on" in (enable_bandpass or [])),
        "enable_spike_removal": ("on" in (enable_spike or [])),
        "enable_absolute": ("on" in (enable_absolute or [])),
        "enable_smoothing": ("on" in (enable_smoothing or [])),
        "enable_normalization": ("on" in (enable_normalization or [])),
        "enable_refinement": ("on" in (enable_refinement or [])),
        "detection_sources": list(detection_sources or ["normalized"]),
    }

    merged_events = list(existing_events)
    added_events = 0
    windows_processed = 0
    next_start_epoch = float(progress_store.get("next_start_epoch", hi) or hi) if trig == "peak-next-btn" else hi
    for start_epoch, end_epoch in ranges:
        start_ts = _from_epoch_seconds(start_epoch, tz_name)
        end_ts = _from_epoch_seconds(end_epoch, tz_name)
        subset, err = _peak_prepare_signal_subset(parent_signal, channel, start_ts, end_ts)
        if err:
            return preview_store, progress_store, f"Peak preview failed: {err}"

        fs = float(calculate_sampling_frequency(subset["datetime"]))
        if not np.isfinite(fs) or fs <= 0:
            return preview_store, progress_store, "Could not determine sampling rate for selected signal."

        try:
            results = peak_detect(
                signal=subset[channel].to_numpy(),
                sampling_rate=fs,
                datetime_series=subset["datetime"],
                **params,
            )
        except Exception as e:
            return preview_store, progress_store, f"Peak detection failed: {type(e).__name__}: {e}"

        events = _peak_events_from_results(results, subset, mode, cleanup_diag=None)
        for row in events:
            dt = row.get("datetime")
            if dt is not None and not pd.isna(dt):
                row["datetime"] = pd.Timestamp(dt).isoformat()
            else:
                row["datetime"] = None
        merged_events.extend(events)
        added_events += len(events)
        windows_processed += 1
        next_start_epoch = float(end_epoch)

    chunks_completed += windows_processed
    preview_out = {
        "mode": current_mode,
        "events": merged_events,
        "parent_signal": parent_signal,
        "channel": channel,
    }
    progress_out = {
        "next_start_epoch": next_start_epoch,
        "chunks_completed": chunks_completed,
        "window_seconds": window_seconds,
        "finished": bool(next_start_epoch >= float(slider_max)),
        "mode": current_mode,
        "run_signature": run_signature,
    }
    if trig == "peak-preview-btn":
        start_ts = _from_epoch_seconds(lo, tz_name)
        end_ts = _from_epoch_seconds(hi, tz_name)
        msg = (
            f"Previewed current window ({start_ts.strftime('%Y-%m-%d %H:%M:%S')} to "
            f"{end_ts.strftime('%Y-%m-%d %H:%M:%S')}) with {added_events} detected peaks. "
            f"Total preview events: {len(merged_events)}."
        )
    else:
        msg = (
            f"Calculated remaining deployment windows: {windows_processed}. "
            f"Added peaks: {added_events}. Total preview events: {len(merged_events)}."
        )
    if progress_out["finished"]:
        msg += " Reached end of deployment. Save to overwrite existing peak events."
    else:
        next_ts = _from_epoch_seconds(next_start_epoch, tz_name)
        msg += f" Next starts at {next_ts.strftime('%Y-%m-%d %H:%M:%S')}."
    return preview_out, progress_out, msg


@app.callback(
    Output("peak-auto-recalc-interval", "disabled"),
    Output("peak-auto-recalc-interval", "n_intervals"),
    Output("peak-auto-recalc-interval", "max_intervals"),
    Input("peak-view-active-store", "data"),
    Input("peak-mode", "value"),
    Input("peak-parent-signal", "value"),
    Input("peak-channel", "value"),
    Input("time-range-slider", "value"),
    Input("peak-detection-sources", "value"),
    Input("peak-broad-low", "value"),
    Input("peak-broad-high", "value"),
    Input("peak-narrow-low", "value"),
    Input("peak-narrow-high", "value"),
    Input("peak-filter-order", "value"),
    Input("peak-spike-threshold", "value"),
    Input("peak-smooth-sec", "value"),
    Input("peak-window-mult", "value"),
    Input("peak-norm-noise", "value"),
    Input("peak-height", "value"),
    Input("peak-distance", "value"),
    Input("peak-search-radius", "value"),
    Input("peak-min-height", "value"),
    Input("peak-max-height", "value"),
    Input("peak-enable-bandpass", "value"),
    Input("peak-enable-spike", "value"),
    Input("peak-enable-absolute", "value"),
    Input("peak-enable-smoothing", "value"),
    Input("peak-enable-normalization", "value"),
    Input("peak-enable-refinement", "value"),
    Input("peak-hr-jump-frac", "value"),
    Input("peak-min-rr", "value"),
    Input("peak-max-hr", "value"),
    Input("peak-min-hr", "value"),
    Input("peak-anti-double-gap", "value"),
    Input("peak-anti-double-window", "value"),
    Input("peak-pick-last-conflict", "value"),
    prevent_initial_call=True,
)
def schedule_peak_auto_recalc(
    active_store,
    mode,
    parent_signal,
    channel,
    slider_value,
    detection_sources,
    broad_low,
    broad_high,
    narrow_low,
    narrow_high,
    filter_order,
    spike_threshold,
    smooth_sec,
    window_mult,
    norm_noise,
    peak_height,
    peak_distance,
    search_radius,
    min_peak_height,
    max_peak_height,
    enable_bandpass,
    enable_spike,
    enable_absolute,
    enable_smoothing,
    enable_normalization,
    enable_refinement,
    hr_jump_frac,
    min_rr,
    max_hr,
    min_hr,
    anti_double_gap,
    anti_double_window,
    pick_last_conflict,
):
    active = bool((active_store or {}).get("active"))
    if not active or not parent_signal or not channel:
        return True, 0, 0
    if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
        return True, 0, 0
    return False, 0, 1


@app.callback(
    Output("peak-detect-preview-store", "data", allow_duplicate=True),
    Output("peak-detect-progress-store", "data", allow_duplicate=True),
    Output("peak-status", "children", allow_duplicate=True),
    Input("peak-auto-recalc-interval", "n_intervals"),
    State("peak-view-active-store", "data"),
    State("peak-mode", "value"),
    State("peak-parent-signal", "value"),
    State("peak-channel", "value"),
    State("time-range-slider", "value"),
    State("peak-detection-sources", "value"),
    State("peak-broad-low", "value"),
    State("peak-broad-high", "value"),
    State("peak-narrow-low", "value"),
    State("peak-narrow-high", "value"),
    State("peak-filter-order", "value"),
    State("peak-spike-threshold", "value"),
    State("peak-smooth-sec", "value"),
    State("peak-window-mult", "value"),
    State("peak-norm-noise", "value"),
    State("peak-height", "value"),
    State("peak-distance", "value"),
    State("peak-search-radius", "value"),
    State("peak-min-height", "value"),
    State("peak-max-height", "value"),
    State("peak-enable-bandpass", "value"),
    State("peak-enable-spike", "value"),
    State("peak-enable-absolute", "value"),
    State("peak-enable-smoothing", "value"),
    State("peak-enable-normalization", "value"),
    State("peak-enable-refinement", "value"),
    State("peak-hr-jump-frac", "value"),
    State("peak-min-rr", "value"),
    State("peak-max-hr", "value"),
    State("peak-min-hr", "value"),
    State("peak-anti-double-gap", "value"),
    State("peak-anti-double-window", "value"),
    State("peak-pick-last-conflict", "value"),
    prevent_initial_call=True,
)
def auto_recalc_peak_window(
    n_intervals,
    active_store,
    mode,
    parent_signal,
    channel,
    slider_value,
    detection_sources,
    broad_low,
    broad_high,
    narrow_low,
    narrow_high,
    filter_order,
    spike_threshold,
    smooth_sec,
    window_mult,
    norm_noise,
    peak_height,
    peak_distance,
    search_radius,
    min_peak_height,
    max_peak_height,
    enable_bandpass,
    enable_spike,
    enable_absolute,
    enable_smoothing,
    enable_normalization,
    enable_refinement,
    hr_jump_frac,
    min_rr,
    max_hr,
    min_hr,
    anti_double_gap,
    anti_double_window,
    pick_last_conflict,
):
    if int(n_intervals or 0) < 1:
        raise dash.exceptions.PreventUpdate
    if not bool((active_store or {}).get("active")):
        raise dash.exceptions.PreventUpdate
    if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
        return {}, {}, "Auto preview skipped: invalid active window."
    if not parent_signal or not channel:
        return {}, {}, "Auto preview skipped: pick a parent signal and channel."

    lo = float(min(slider_value))
    hi = float(max(slider_value))
    start_ts = _from_epoch_seconds(lo, tz_name)
    end_ts = _from_epoch_seconds(hi, tz_name)
    subset, err = _peak_prepare_signal_subset(parent_signal, channel, start_ts, end_ts)
    if err:
        return {}, {}, f"Auto preview failed: {err}"

    fs = float(calculate_sampling_frequency(subset["datetime"]))
    if not np.isfinite(fs) or fs <= 0:
        return {}, {}, "Auto preview failed: could not determine sampling rate."

    params = {
        "broad_lowcut": float(broad_low),
        "broad_highcut": float(broad_high),
        "narrow_lowcut": float(narrow_low),
        "narrow_highcut": float(narrow_high),
        "filter_order": int(filter_order),
        "spike_threshold": float(spike_threshold),
        "smooth_sec_multiplier": float(smooth_sec),
        "window_size_multiplier": float(window_mult),
        "normalization_noise": float(norm_noise),
        "peak_height": float(peak_height),
        "peak_distance_sec": float(peak_distance),
        "search_radius_sec": float(search_radius),
        "min_peak_height": float(min_peak_height),
        "max_peak_height": float(max_peak_height),
        "enable_bandpass": ("on" in (enable_bandpass or [])),
        "enable_spike_removal": ("on" in (enable_spike or [])),
        "enable_absolute": ("on" in (enable_absolute or [])),
        "enable_smoothing": ("on" in (enable_smoothing or [])),
        "enable_normalization": ("on" in (enable_normalization or [])),
        "enable_refinement": ("on" in (enable_refinement or [])),
        "detection_sources": list(detection_sources or ["normalized"]),
    }
    try:
        results = peak_detect(
            signal=subset[channel].to_numpy(),
            sampling_rate=fs,
            datetime_series=subset["datetime"],
            **params,
        )
    except Exception as e:
        return {}, {}, f"Auto preview failed: {type(e).__name__}: {e}"

    mode_key = str(mode or "heart_rate").strip().lower()
    events = _peak_events_from_results(results, subset, mode_key, cleanup_diag=None)
    for row in events:
        dt = row.get("datetime")
        row["datetime"] = pd.Timestamp(dt).isoformat() if dt is not None and not pd.isna(dt) else None

    preview_out = {"mode": mode_key, "events": events, "parent_signal": parent_signal, "channel": channel}
    run_signature = json.dumps(
        _peak_run_signature_dict(
            mode_key, parent_signal, channel, detection_sources,
            broad_low, broad_high, narrow_low, narrow_high,
            filter_order, spike_threshold, smooth_sec, window_mult, norm_noise,
            peak_height, peak_distance, search_radius, min_peak_height, max_peak_height,
            enable_bandpass, enable_spike, enable_absolute, enable_smoothing, enable_normalization, enable_refinement,
            hr_jump_frac, min_rr, max_hr, min_hr, anti_double_gap, anti_double_window, pick_last_conflict,
        ),
        sort_keys=True,
    )
    progress_out = {
        "next_start_epoch": float(hi),
        "chunks_completed": 1,
        "window_seconds": max(2.0, hi - lo),
        "finished": bool(float(hi) >= float(slider_max)),
        "mode": mode_key,
        "run_signature": run_signature,
    }
    msg = (
        f"Auto preview updated ({start_ts.strftime('%Y-%m-%d %H:%M:%S')} to "
        f"{end_ts.strftime('%Y-%m-%d %H:%M:%S')}) with {len(events)} detected peaks."
    )
    return preview_out, progress_out, msg


@app.callback(
    Output("peak-status", "children", allow_duplicate=True),
    Output("peak-detect-preview-store", "data", allow_duplicate=True),
    Output("peak-detect-progress-store", "data", allow_duplicate=True),
    Output("events-select", "options", allow_duplicate=True),
    Output("events-select", "value", allow_duplicate=True),
    Output("selected-events-store", "data", allow_duplicate=True),
    Input("peak-save-btn", "n_clicks"),
    State("peak-detect-preview-store", "data"),
    State("peak-detect-progress-store", "data"),
    State("events-select", "value"),
    prevent_initial_call=True,
)
def save_peak_events(_n, preview_store, progress_store, selected_events):
    global data_pkl
    preview_store = dict(preview_store or {})
    progress_store = dict(progress_store or {})
    rows = list(preview_store.get("events") or [])
    mode = str(preview_store.get("mode") or progress_store.get("mode") or "heart_rate").strip().lower()
    if int(progress_store.get("chunks_completed", 0) or 0) < 1:
        raise dash.exceptions.PreventUpdate

    df_new = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value"])
    if not df_new.empty and "key" not in df_new.columns:
        return "No valid preview events to save.", preview_store, progress_store, no_update, no_update

    if "datetime" in df_new.columns:
        df_new["datetime"] = pd.to_datetime(df_new["datetime"], errors="coerce")
    else:
        df_new["datetime"] = pd.NaT

    real_data_pkl = _ensure_real_data_pkl_loaded()

    if not isinstance(getattr(real_data_pkl, "event_data", None), pd.DataFrame):
        real_data_pkl.event_data = pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value"])

    if not isinstance(getattr(data_pkl, "event_data", None), pd.DataFrame):
        data_pkl.event_data = pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value"])
    existing = real_data_pkl.event_data.copy()
    if mode == "stroke_rate":
        drop_prefixes = ("strokebeat_auto_detect_",)
    else:
        drop_prefixes = ("heartbeat_auto_detect_",)
    keys = existing.get("key", pd.Series(dtype="object")).astype(str)
    removed_count = int(keys.str.startswith(drop_prefixes).sum())
    keep_mask = ~keys.str.startswith(drop_prefixes)
    existing = existing.loc[keep_mask].copy()

    if not df_new.empty:
        real_data_pkl.event_data = pd.concat([existing, df_new], ignore_index=True)
    else:
        real_data_pkl.event_data = existing
    if "datetime" in real_data_pkl.event_data.columns:
        real_data_pkl.event_data["datetime"] = pd.to_datetime(real_data_pkl.event_data["datetime"], errors="coerce")
        real_data_pkl.event_data = real_data_pkl.event_data.sort_values("datetime", kind="mergesort").reset_index(drop=True)

    try:
        if hasattr(real_data_pkl, "save_datareader_object"):
            real_data_pkl.save_datareader_object()
    except Exception:
        pass

    data_pkl = real_data_pkl
    if deployment_source is not None:
        deployment_source.replace_cached_events(real_data_pkl.event_data)

    ev_opts = sorted([str(k) for k in real_data_pkl.event_data["key"].dropna().unique()], key=lambda x: x.lower())
    sel = [str(k) for k in (selected_events or []) if str(k) in ev_opts]
    for k in sorted([str(k) for k in df_new.get("key", pd.Series(dtype="object")).dropna().unique()], key=lambda x: x.lower()):
        if k not in sel:
            sel.append(k)
    if df_new.empty:
        sel = [k for k in sel if not k.startswith(drop_prefixes)]
    return (
        f"Saved {len(df_new)} preview events to deployment (removed {removed_count} existing {mode} peak events first).",
        {},
        {},
        [{"label": k, "value": k} for k in ev_opts],
        sel,
        sel,
    )


@app.callback(
    Output("signals-select", "value", allow_duplicate=True),
    Input("selected-signals-store", "data"),
    prevent_initial_call=True,
)
def sync_hidden_signals_from_store(selected_signals):
    return [str(s) for s in (selected_signals or []) if str(s).strip()]


@app.callback(
    Output("events-select", "value", allow_duplicate=True),
    Input("selected-events-store", "data"),
    prevent_initial_call=True,
)
def sync_hidden_events_from_store(selected_events):
    return [str(e) for e in (selected_events or []) if str(e).strip()]


@app.callback(
    Output("ordered-signals-store", "data"),
    Input("selected-signals-store", "data"),
    State("ordered-signals-store", "data"),
)
def sync_signal_order(selected, current_order):
    selected = selected or []
    current_order = current_order or []
    kept = [s for s in current_order if s in selected]
    appended = [s for s in selected if s not in kept]
    return kept + appended


@app.callback(
    Output("save-signal-event-defaults-status", "children"),
    Input("save-signal-event-defaults-btn", "n_clicks"),
    State("ordered-signals-store", "data"),
    State("selected-signals-store", "data"),
    State("selected-events-store", "data"),
    prevent_initial_call=True,
)
def save_signal_event_dataset_defaults(_n_clicks, ordered_signals, selected_signals, selected_events):
    selected_sigs = [str(s) for s in (selected_signals or []) if str(s).strip()]
    order = [str(s) for s in (ordered_signals or []) if str(s) in selected_sigs]
    ordered_selected = order + [s for s in selected_sigs if s not in order]
    selected_evs = [str(e) for e in (selected_events or []) if str(e).strip()]

    try:
        param_manager.set_dataset_defaults(
            entries={
                "dash_default_signals": ordered_selected,
                "plotly_default_signals": ordered_selected,
                "dash_default_events": selected_evs,
                "plotly_default_events": selected_evs,
            },
            section="settings",
        )
    except Exception as e:
        return f"Failed to save dataset defaults: {type(e).__name__}: {e}"

    return (
        f"Saved dataset defaults: {len(ordered_selected)} signal(s) and "
        f"{len(selected_evs)} event(s)."
    )


@app.callback(
    Output("save-signal-colors-status", "children"),
    Input("save-signal-colors-btn", "n_clicks"),
    prevent_initial_call=True,
)
def save_signal_colors_to_config(_n_clicks):
    try:
        _save_color_mapping(color_mapping_path, _signal_colors)
    except Exception as e:
        return f"Failed to save signal colors: {type(e).__name__}: {e}"
    count = sum(1 for v in (_signal_colors or {}).values() if _is_hex_color(v))
    return f"Saved {count} signal color mapping(s) to {color_mapping_path}."


@app.callback(
    Output("target-sampling-rate-interval", "children"),
    Input("target-sampling-rate-hz", "value"),
)
def render_target_sampling_interval(target_hz):
    try:
        hz = float(target_hz)
    except Exception:
        hz = 10.0
    if hz <= 0:
        hz = 10.0
    interval_s = 1.0 / hz
    return html.Span(["intersample interval: ", html.I(f"{interval_s:.3f} s")])


@app.callback(
    Output("overlap-signal-a", "options"),
    Output("overlap-signal-a", "value"),
    Output("overlap-signal-b", "options"),
    Output("overlap-signal-b", "value"),
    Input("ordered-signals-store", "data"),
    State("overlap-signal-a", "value"),
    State("overlap-signal-b", "value"),
)
def sync_overlap_signal_options(ordered_signals, current_a, current_b):
    signals = [str(s) for s in (ordered_signals or []) if str(s) in (data_pkl.signal_data or {})]
    opts = [{"label": s, "value": s} for s in signals]
    if not signals:
        return [], None, [], None
    a = current_a if current_a in signals else signals[0]
    if current_b in signals:
        b = current_b
    elif len(signals) > 1:
        b = signals[1]
    else:
        b = signals[0]
    return opts, a, opts, b


@app.callback(
    Output("overlap-window-store", "data"),
    Output("overlap-status", "children", allow_duplicate=True),
    Input("overlap-calc-btn", "n_clicks"),
    State("overlap-signal-a", "value"),
    State("overlap-signal-b", "value"),
    prevent_initial_call=True,
)
def calculate_overlap_window(_n_clicks, signal_a, signal_b):
    sa = str(signal_a or "").strip()
    sb = str(signal_b or "").strip()
    if not sa or not sb:
        return {}, "Select two signals first."
    a_start, a_end = _signal_valid_window(data_pkl, sa, tz_name)
    b_start, b_end = _signal_valid_window(data_pkl, sb, tz_name)
    if a_start is None or a_end is None:
        return {}, f"No valid non-NaN datetime rows found for {sa}."
    if b_start is None or b_end is None:
        return {}, f"No valid non-NaN datetime rows found for {sb}."

    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_end <= overlap_start:
        return {}, f"No overlap between {sa} and {sb}."

    start_epoch = int(_to_epoch_seconds(overlap_start))
    end_epoch = int(_to_epoch_seconds(overlap_end))
    msg = f"Overlap found: {_format_ts_input(overlap_start, _display_tz_name())} to {_format_ts_input(overlap_end, _display_tz_name())}"
    return {
        "signal_a": sa,
        "signal_b": sb,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
    }, msg


@app.callback(
    Output("time-range-slider", "value", allow_duplicate=True),
    Output("start-time", "value", allow_duplicate=True),
    Output("end-time", "value", allow_duplicate=True),
    Output("overlap-status", "children", allow_duplicate=True),
    Input("overlap-set-window-btn", "n_clicks"),
    State("overlap-window-store", "data"),
    State("time-display-tz-toggle", "value"),
    State("time-range-slider", "min"),
    State("time-range-slider", "max"),
    prevent_initial_call=True,
)
def set_time_window_to_overlap(_n_clicks, overlap_store, display_mode, s_min, s_max):
    store = overlap_store or {}
    if "start_epoch" not in store or "end_epoch" not in store:
        raise dash.exceptions.PreventUpdate
    lo = int(store["start_epoch"])
    hi = int(store["end_epoch"])
    s_min = int(s_min if s_min is not None else slider_min)
    s_max = int(s_max if s_max is not None else slider_max)
    lo = max(s_min, min(lo, s_max))
    hi = max(s_min, min(hi, s_max))
    if hi <= lo:
        hi = min(s_max, lo + MIN_WINDOW_SECONDS)
        if hi <= lo:
            raise dash.exceptions.PreventUpdate
    display_tz = _display_tz_name(display_mode)
    lo_ts = _from_epoch_seconds(lo, display_tz)
    hi_ts = _from_epoch_seconds(hi, display_tz)
    return [lo, hi], _format_ts_input(lo_ts, display_tz), _format_ts_input(hi_ts, display_tz), "Set time window to overlap."


@app.callback(
    Output("time-range-slider", "min", allow_duplicate=True),
    Output("time-range-slider", "max", allow_duplicate=True),
    Output("time-range-slider", "value", allow_duplicate=True),
    Output("time-range-slider", "marks", allow_duplicate=True),
    Output("playhead-slider", "min", allow_duplicate=True),
    Output("playhead-slider", "max", allow_duplicate=True),
    Output("playhead-slider", "value", allow_duplicate=True),
    Output("playhead-time", "data", allow_duplicate=True),
    Output("start-time", "value", allow_duplicate=True),
    Output("end-time", "value", allow_duplicate=True),
    Output("overlap-status", "children", allow_duplicate=True),
    Input("overlap-trim-btn", "n_clicks"),
    State("overlap-window-store", "data"),
    State("time-display-tz-toggle", "value"),
    prevent_initial_call=True,
)
def trim_datapkl_to_overlap(_n_clicks, overlap_store, display_mode):
    global global_start, global_end
    global start_default, end_default, slider_min, slider_max, slider_default, slider_marks
    global _depth_ctx_x, _depth_ctx_y, _depth_ctx_label, _depth_ctx_unit, _depth_ctx_source_signal, _downsample_cache

    store = overlap_store or {}
    if "start_epoch" not in store or "end_epoch" not in store:
        raise dash.exceptions.PreventUpdate
    lo = int(store["start_epoch"])
    hi = int(store["end_epoch"])
    if hi <= lo:
        raise dash.exceptions.PreventUpdate

    start_ts = _from_epoch_seconds(lo, tz_name)
    end_ts = _from_epoch_seconds(hi, tz_name)
    _trim_datapkl_to_window(data_pkl, start_ts, end_ts, tz_name)

    ctx_local = _compute_view_context(
        data_pkl,
        tz_name,
        start_ts,
        end_ts,
        param_manager_obj=param_manager,
    )
    global_start = ctx_local["global_start"]
    global_end = ctx_local["global_end"]
    slider_min = ctx_local["slider_min"]
    slider_max = ctx_local["slider_max"]
    slider_default = ctx_local["slider_default"]
    slider_marks = ctx_local["slider_marks"]
    start_default = ctx_local["start_default"]
    end_default = ctx_local["end_default"]
    _depth_ctx_x, _depth_ctx_y, _depth_ctx_label, _depth_ctx_unit, _depth_ctx_source_signal = _extract_depth_context_series(data_pkl, tz_name)
    _downsample_cache.clear()

    win_lo = int(min(slider_default))
    win_hi = int(max(slider_default))
    playhead_default_epoch = float(_default_playhead_epoch(slider_default))
    return (
        slider_min,
        slider_max,
        slider_default,
        None,
        win_lo,
        win_hi,
        playhead_default_epoch,
        playhead_default_epoch,
        _format_ts_input(start_default, _display_tz_name(display_mode)),
        _format_ts_input(end_default, _display_tz_name(display_mode)),
        "Trimmed in-memory data to overlap window.",
    )


@app.callback(
    Output("selected-signals-store", "data", allow_duplicate=True),
    Input({"type": "chip-remove-btn", "group": "signals", "key": ALL, "value": ALL}, "n_clicks"),
    State("selected-signals-store", "data"),
    prevent_initial_call=True,
)
def remove_signal_chip(_n_clicks, selected_signals):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    sig = str(trig.get("value") or "")
    current = [str(s) for s in (selected_signals or [])]
    if sig not in current:
        raise dash.exceptions.PreventUpdate
    return [s for s in current if s != sig]


@app.callback(
    Output({"type": "sig-channels", "sig": ALL}, "value"),
    Input({"type": "chip-remove-btn", "group": "channels", "key": ALL, "value": ALL}, "n_clicks"),
    State({"type": "sig-channels", "sig": ALL}, "value"),
    State({"type": "sig-channels", "sig": ALL}, "id"),
    prevent_initial_call=True,
)
def remove_channel_chip(_n_clicks, channel_values, channel_ids):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    target_sig = str(trig.get("key") or "")
    target_ch = str(trig.get("value") or "")
    if not target_sig or not target_ch:
        raise dash.exceptions.PreventUpdate

    out = []
    changed = False
    for vals, meta in zip(channel_values or [], channel_ids or []):
        vals = [str(v) for v in (vals or [])]
        sig = str((meta or {}).get("sig") or "")
        if sig == target_sig and target_ch in vals:
            out.append([v for v in vals if v != target_ch])
            changed = True
        else:
            out.append(vals)
    if not changed:
        raise dash.exceptions.PreventUpdate
    return out


@app.callback(
    Output("point-events-select", "value", allow_duplicate=True),
    Output("state-events-select", "value", allow_duplicate=True),
    Input({"type": "chip-remove-btn", "group": ALL, "key": ALL, "value": ALL}, "n_clicks"),
    State("point-events-select", "value"),
    State("state-events-select", "value"),
    prevent_initial_call=True,
)
def remove_event_chip(_n_clicks, point_values, state_values):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    group = str(trig.get("group") or "")
    ev = str(trig.get("value") or "")
    points = [str(v) for v in (point_values or [])]
    states = [str(v) for v in (state_values or [])]
    if group == "point_events" and ev in points:
        return [v for v in points if v != ev], states
    if group == "state_events" and ev in states:
        return points, [v for v in states if v != ev]
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("selected-events-store", "data", allow_duplicate=True),
    Input({"type": "chip-remove-btn", "group": ALL, "key": ALL, "value": ALL}, "n_clicks"),
    State("selected-events-store", "data"),
    prevent_initial_call=True,
)
def remove_events_chip(_n_clicks, events_values):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    group = str(trig.get("group") or "")
    ev = str(trig.get("value") or "")
    events = [str(v) for v in (events_values or [])]
    if group == "events" and ev in events:
        return [v for v in events if v != ev]
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("point-events-select", "value", allow_duplicate=True),
    Output("state-events-select", "value", allow_duplicate=True),
    Input("chip-order-updates", "value"),
    State("point-events-select", "value"),
    State("state-events-select", "value"),
    prevent_initial_call=True,
)
def apply_event_chip_orders(chip_order_json, point_values, state_values):
    try:
        payload = json.loads(chip_order_json or "{}")
    except Exception:
        raise dash.exceptions.PreventUpdate

    points = [str(v) for v in (point_values or [])]
    states = [str(v) for v in (state_values or [])]
    changed = False

    point_payload = (((payload.get("point_events") or {}).get("__all__")) or [])
    if point_payload:
        ordered = [v for v in point_payload if v in points]
        if ordered:
            rem = [v for v in points if v not in ordered]
            points = ordered + rem
            changed = True

    state_payload = (((payload.get("state_events") or {}).get("__all__")) or [])
    if state_payload:
        ordered = [v for v in state_payload if v in states]
        if ordered:
            rem = [v for v in states if v not in ordered]
            states = ordered + rem
            changed = True

    if not changed:
        raise dash.exceptions.PreventUpdate
    return points, states


@app.callback(
    Output("selected-events-store", "data", allow_duplicate=True),
    Input("chip-order-updates", "value"),
    State("selected-events-store", "data"),
    prevent_initial_call=True,
)
def apply_events_chip_orders(chip_order_json, events_values):
    try:
        payload = json.loads(chip_order_json or "{}")
    except Exception:
        raise dash.exceptions.PreventUpdate

    events = [str(v) for v in (events_values or [])]
    events_payload = (((payload.get("events") or {}).get("__all__")) or [])
    if not events_payload:
        raise dash.exceptions.PreventUpdate

    ordered = [v for v in events_payload if v in events]
    if not ordered:
        raise dash.exceptions.PreventUpdate
    rem = [v for v in events if v not in ordered]
    return ordered + rem


@app.callback(
    Output("model-3d-rotation-order-store", "data", allow_duplicate=True),
    Input("chip-order-updates", "value"),
    State("model-3d-rotation-order-store", "data"),
    prevent_initial_call=True,
)
def apply_model_3d_rotation_order(chip_order_json, current_order):
    try:
        payload = json.loads(chip_order_json or "{}")
    except Exception:
        raise dash.exceptions.PreventUpdate
    order_payload = (((payload.get("model_3d_rotation_order") or {}).get("__all__")) or [])
    if not order_payload:
        raise dash.exceptions.PreventUpdate
    base = _normalize_rotation_order(current_order)
    ordered = [v for v in order_payload if v in base]
    if not ordered:
        raise dash.exceptions.PreventUpdate
    rem = [v for v in base if v not in ordered]
    return _normalize_rotation_order(ordered + rem)


@app.callback(
    Output("point-events-select", "value", allow_duplicate=True),
    Output("state-events-select", "value", allow_duplicate=True),
    Input("point-events-select", "value"),
    Input("state-events-select", "value"),
    prevent_initial_call=True,
)
def enforce_event_type_exclusivity(point_values, state_values):
    trig = ctx.triggered_id
    points = [str(v) for v in (point_values or []) if str(v).strip()]
    states = [str(v) for v in (state_values or []) if str(v).strip()]

    if trig == "point-events-select":
        filtered_states = [v for v in states if v not in set(points)]
        if filtered_states != states:
            return points, filtered_states
        raise dash.exceptions.PreventUpdate

    if trig == "state-events-select":
        filtered_points = [v for v in points if v not in set(states)]
        if filtered_points != points:
            return filtered_points, states
        raise dash.exceptions.PreventUpdate

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("signals-select", "placeholder"),
    Input("signals-select", "value"),
)
def update_signal_placeholder(values):
    return _comma_placeholder(values)


@app.callback(
    Output("events-select", "placeholder"),
    Input("events-select", "value"),
)
def update_events_placeholder(values):
    return _comma_placeholder(values)


@app.callback(
    Output("point-events-select", "value", allow_duplicate=True),
    Output("state-events-select", "value", allow_duplicate=True),
    Input("events-select", "value"),
    prevent_initial_call=True,
)
def sync_legacy_event_splits_from_events(events_values):
    selected = [str(v) for v in (events_values or []) if str(v).strip()]
    points = []
    states = []
    for ev in selected:
        if _has_state_duration_events(data_pkl, ev):
            states.append(ev)
        else:
            points.append(ev)
    return points, states


@app.callback(
    Output("point-events-select", "placeholder"),
    Input("point-events-select", "value"),
)
def update_point_event_placeholder(values):
    return _comma_placeholder(values)


@app.callback(
    Output("state-events-select", "placeholder"),
    Input("state-events-select", "value"),
)
def update_state_event_placeholder(values):
    return _comma_placeholder(values)


@app.callback(
    Output({"type": "sig-channels", "sig": ALL}, "placeholder"),
    Input({"type": "sig-channels", "sig": ALL}, "value"),
)
def update_channel_placeholders(values):
    return [_comma_placeholder(v) for v in (values or [])]


@app.callback(
    Output({"type": "event-targets", "event": ALL}, "placeholder"),
    Input({"type": "event-targets", "event": ALL}, "value"),
)
def update_event_target_placeholders(values):
    return [_comma_placeholder(v) for v in (values or [])]


@app.callback(
    Output("signal-order-editor", "children"),
    Input("ordered-signals-store", "data"),
)
def render_signal_order_editor(order):
    order = order or []
    if not order:
        return html.Small("No signals selected.")
    return _render_sortable_chips(
        order,
        "signals",
        "__all__",
        color_fn=lambda v: _signal_chip_style(v),
        removable=True,
    )


@app.callback(
    Output("signals-summary-chips", "children"),
    Input("ordered-signals-store", "data"),
)
def render_signal_summary_chips(order):
    return _render_summary_chips(order or [], color_fn=lambda v: _signal_chip_style(v))


@app.callback(
    Output("model-3d-rotation-order-editor", "children"),
    Input("model-3d-rotation-order-store", "data"),
)
def render_model_3d_rotation_order_editor(order):
    order = _normalize_rotation_order(order)
    label_map = {
        "roll": "roll",
        "pitch": "pitch",
        "heading": "heading",
    }
    return _render_sortable_chips(
        order,
        "model_3d_rotation_order",
        "__all__",
        label_fn=lambda v: label_map.get(str(v), str(v)),
    )


@app.callback(
    Output("channels-editor", "children"),
    Input("ordered-signals-store", "data"),
    Input("color-map-version", "data"),
    State("channels-store", "data"),
    State("channel-order-store", "data"),
)
def render_channels_editor(order, _color_map_version, channels_store, channel_order_store):
    order = order or []
    channels_store = channels_store or {}
    channel_order_store = channel_order_store or {}
    blocks = []
    summary_vals = []
    for s in order:
        options = _signal_channels(data_pkl, s)
        if not options:
            continue
        sig_meta = (
            _signal_display_metadata.get(s)
            or _signal_display_metadata.get(str(s).lower())
            or _signal_display_metadata.get(str(s).upper())
            or {}
        )
        signal_label = _safe_text((sig_meta or {}).get("label")) or str(s)
        saved = [c for c in (channels_store.get(s) or []) if c in options]
        default = saved or options
        ordered = _normalize_channel_order(default, channel_order_store.get(s))
        summary_vals.extend([f"{s}.{c}" for c in ordered])
        blocks.append(
            html.Div(
                [
                    html.Span("", className="chip-handle signal-block-drag-corner", **{"aria-hidden": "true"}),
                    html.Div(className="signal-block-edge signal-block-edge-top"),
                    html.Div(className="signal-block-edge signal-block-edge-right"),
                    html.Div(className="signal-block-edge signal-block-edge-bottom"),
                    html.Div(className="signal-block-edge signal-block-edge-left"),
                    html.Div(
                        [
                            html.Span("", className="chip-handle signal-drag-grip", **{"aria-hidden": "true"}),
                            html.Div(
                                [
                                    html.B(signal_label, className="signal-pretty-label"),
                                    html.I(str(s), className="signal-raw-label"),
                                ],
                                className="signal-drag-surface signal-title-text",
                            ),
                        ],
                        className="signal-block-title",
                    ),
                    dcc.Dropdown(
                        id={"type": "sig-channels", "sig": s},
                        options=[{"label": c, "value": c} for c in options],
                        value=default,
                        multi=True,
                        className="with-chip-order",
                        placeholder="Add/remove channels...",
                    ),
                    _render_sortable_chips(
                        ordered,
                        "channels",
                        s,
                        color_fn=lambda v, sig=s: _channel_chip_style(sig, v),
                        editable_colors=True,
                        removable=True,
                    ),
                ],
                className="signal-channel-block dnd-item",
                draggable="true",
                **{"data-value": s, "data-color-editable": "0"},
            )
        )
    return html.Details(
        [
            html.Summary(
                [
                    html.Span("Channels", className="section-title"),
                    _render_summary_chips(
                        summary_vals,
                        color_fn=lambda v: _channel_chip_style(
                            str(v).split(".", 1)[0],
                            str(v).split(".", 1)[1] if "." in str(v) else str(v),
                        ),
                    ),
                ],
                className="collapsible-summary",
            ),
            html.Div(
                blocks or [html.Small("No channels available.")],
                className="collapsible-body signal-sortable",
                **{"data-order-group": "signals", "data-order-key": "__all__", "data-dnd-axis": "y"},
            ),
        ],
        className="collapsible-panel",
        open=True,
    )


@app.callback(
    Output("color-edit-target", "data"),
    Input({"type": "chip-color-btn", "group": ALL, "key": ALL, "value": ALL}, "n_clicks"),
    Input({"type": "event-edit-btn", "group": ALL, "event": ALL}, "n_clicks"),
    Input({"type": "event-shape-btn", "group": ALL, "event": ALL}, "n_clicks"),
    Input({"type": "event-color-btn", "event": ALL}, "n_clicks"),
    Input({"type": "event-shade-color-btn", "event": ALL}, "n_clicks"),
    Input("seg-open-color-editor-btn", "n_clicks"),
    State("seg-style-event-key", "value"),
    prevent_initial_call=True,
)
def open_color_editor(_chip_clicks, _event_edit_clicks, _event_shape_clicks, _event_clicks, _event_shade_clicks, _seg_open_clicks, seg_event_key):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if trig == "seg-open-color-editor-btn":
        event_key = str(seg_event_key or "")
        if not event_key:
            raise dash.exceptions.PreventUpdate
        return {"group": "segmentation_events", "event": event_key, "color_role": "marker"}
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    if trig.get("type") == "chip-color-btn" and trig.get("group") == "channels":
        return {"group": "channels", "signal": trig.get("key"), "channel": trig.get("value")}
    if trig.get("type") in {"event-edit-btn", "event-shape-btn"}:
        event_key = str(trig.get("event") or "")
        if not event_key:
            raise dash.exceptions.PreventUpdate
        return {"group": "events", "event": event_key, "color_role": "marker"}
    if trig.get("type") == "event-color-btn":
        event_key = str(trig.get("event") or "")
        if not event_key:
            raise dash.exceptions.PreventUpdate
        return {"group": "events", "event": event_key, "color_role": "marker"}
    if trig.get("type") == "event-shade-color-btn":
        event_key = str(trig.get("event") or "")
        if not event_key:
            raise dash.exceptions.PreventUpdate
        return {"group": "events", "event": event_key, "color_role": "shade"}
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("active-event-editor", "data"),
    Input({"type": "event-edit-btn", "group": ALL, "event": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_active_event_editor(_n_clicks):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    group = str(trig.get("group") or "")
    ev = str(trig.get("event") or "")
    if group not in {"point_events", "state_events", "events"} or not ev:
        raise dash.exceptions.PreventUpdate
    return {"group": group, "event": ev}


@app.callback(
    Output("color-editor-modal", "className"),
    Output("color-editor-modal", "style"),
    Output("color-editor-title", "children"),
    Output("color-picker-input", "value"),
    Output("color-hex-input", "value"),
    Output("hue-slider", "value"),
    Output("sat-slider", "value"),
    Output("val-slider", "value"),
    Output("color-swatch-grid", "children"),
    Output("color-editor-event-style-wrap", "style"),
    Output("color-editor-event-symbol", "value"),
    Output("color-editor-event-shade-enabled", "value"),
    Output("color-editor-event-shade-hex", "value"),
    Output("color-editor-event-shade-opacity", "value"),
    Output("color-editor-event-shade-mode", "value"),
    Output("color-editor-event-shade-pct-min", "value"),
    Output("color-editor-event-shade-pct-max", "value"),
    Input("color-edit-target", "data"),
    State("color-editor-anchor-input", "value"),
    State("event-style-store", "data"),
    State("seg-event-style-store", "data"),
)
def render_color_editor(
    target,
    anchor_raw,
    event_style_store,
    seg_event_style_store,
):
    target = target or {}
    group = str(target.get("group") or "")
    signal = str(target.get("signal") or "")
    channel = str(target.get("channel") or "")
    event_key = str(target.get("event") or "")
    color_role = str(target.get("color_role") or "marker")
    if group not in {"channels", "events", "segmentation_events"}:
        return (
            "color-editor-modal hidden",
            {},
            "Edit Trace Color",
            "#4DAEF8",
            "#4DAEF8",
            203,
            68,
            97,
            [],
            {"display": "none"},
            "circle",
            ["on"],
            "#4DAEF8",
            0.2,
            "fill_trace_split",
            0,
            100,
        )

    anchor_style = {"left": "28px", "top": "120px"}
    try:
        payload = json.loads(anchor_raw or "{}")
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        if x > 0 and y > 0:
            anchor_style = {
                "left": f"{x + 12}px",
                "top": f"{max(12, y - 16)}px",
            }
    except Exception:
        pass

    if group == "channels":
        if not signal or not channel:
            return (
                "color-editor-modal hidden",
                {},
                "Edit Trace Color",
                "#4DAEF8",
                "#4DAEF8",
                203,
                68,
                97,
                [],
                {"display": "none"},
                "circle",
                ["on"],
                "#4DAEF8",
                0.2,
                "fill_trace_split",
                0,
                100,
            )
        selected = _normalize_hex_color(_resolve_trace_color(_signal_colors, signal, channel)) or "#4DAEF8"
        title = f"Edit Trace Color: {signal}.{channel}"
        event_style_wrap = {"display": "none"}
        event_symbol = "circle"
        event_shade_enabled = ["on"]
        event_shade_hex = selected
        event_shade_opacity = 0.2
        event_shade_mode = "fill_trace_split"
        event_shade_pct_min = 0
        event_shade_pct_max = 100
    else:
        style_source = seg_event_style_store if group == "segmentation_events" else event_style_store
        resolved_style = _segmentation_style_entry(event_key, style_store=style_source or {}, index=0)
        saved_style = dict((style_source or {}).get(event_key) or {})
        style_defaults = _default_event_style(event_key, 0)
        if color_role == "shade":
            selected = resolved_style["shade_color"] or resolved_style["color"] or "#4DAEF8"
            title = f"Edit Event Shade Color: {event_key}"
        else:
            selected = resolved_style["color"] or "#4DAEF8"
            title = f"Edit Event Color: {event_key}"
        event_symbol = str(saved_style.get("symbol") or resolved_style["symbol"] or style_defaults.get("symbol") or "circle")
        if event_symbol not in EVENT_SYMBOL_OPTIONS:
            event_symbol = "circle"
        event_shade_enabled = ["on"] if bool(saved_style.get("shade_enabled", resolved_style["shade_enabled"])) else []
        event_shade_hex = resolved_style["shade_color"] or selected
        try:
            event_shade_opacity = float(saved_style.get("shade_opacity", resolved_style["shade_opacity"]))
        except Exception:
            event_shade_opacity = 0.2
        event_shade_opacity = max(0.0, min(1.0, event_shade_opacity))
        event_shade_mode = _normalize_event_shade_mode(saved_style.get("shade_mode", resolved_style["shade_mode"]))
        event_shade_pct_min = _coerce_float_or_none(saved_style.get("shade_pct_min"))
        event_shade_pct_max = _coerce_float_or_none(saved_style.get("shade_pct_max"))
        event_shade_pct_min = 0 if event_shade_pct_min is None else max(0.0, min(100.0, event_shade_pct_min))
        event_shade_pct_max = 100 if event_shade_pct_max is None else max(0.0, min(100.0, event_shade_pct_max))
        event_style_wrap = {"display": "block"}

    h, s, v = _hex_to_hsv(selected)
    swatch_source = set(c for c in (_signal_colors or {}).values() if _is_hex_color(c))
    for store in (event_style_store or {}, seg_event_style_store or {}):
        for style in (store or {}).values():
            if not isinstance(style, dict):
                continue
            for key in ("color", "shade_color"):
                color = style.get(key)
                if _is_hex_color(color):
                    swatch_source.add(_normalize_hex_color(color))
    swatches = sorted(swatch_source)
    swatch_buttons = [
        html.Button(
            "",
            id={"type": "color-swatch-btn", "color": c},
            n_clicks=0,
            className="color-swatch-btn",
            title=c,
            style={"backgroundColor": c},
        )
        for c in swatches
    ]
    return (
        "color-editor-modal open",
        anchor_style,
        title,
        selected,
        selected,
        h,
        s,
        v,
        swatch_buttons,
        event_style_wrap,
        event_symbol,
        event_shade_enabled,
        event_shade_hex,
        event_shade_opacity,
        event_shade_mode,
        event_shade_pct_min,
        event_shade_pct_max,
    )


@app.callback(
    Output("color-picker-input", "value", allow_duplicate=True),
    Input({"type": "color-swatch-btn", "color": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def pick_swatch_color(_n_clicks):
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    color = trig.get("color")
    if not _is_hex_color(color):
        raise dash.exceptions.PreventUpdate
    return color


@app.callback(
    Output("color-picker-input", "value", allow_duplicate=True),
    Output("color-hex-input", "value", allow_duplicate=True),
    Output("hue-slider", "value", allow_duplicate=True),
    Output("sat-slider", "value", allow_duplicate=True),
    Output("val-slider", "value", allow_duplicate=True),
    Input("color-picker-input", "value"),
    Input("color-hex-input", "value"),
    Input("hue-slider", "value"),
    Input("sat-slider", "value"),
    Input("val-slider", "value"),
    prevent_initial_call=True,
)
def sync_color_editor_controls(picker_color, hex_color, hue, sat, val):
    trig = ctx.triggered_id

    if trig in {"color-picker-input", "color-hex-input"}:
        src = picker_color if trig == "color-picker-input" else hex_color
        norm = _normalize_hex_color(src)
        if not norm:
            raise dash.exceptions.PreventUpdate
        h, s, v = _hex_to_hsv(norm)
        return norm, norm, h, s, v

    if trig in {"hue-slider", "sat-slider", "val-slider"}:
        norm = _hsv_to_hex(hue, sat, val)
        if not norm:
            raise dash.exceptions.PreventUpdate
        h, s, v = _hex_to_hsv(norm)
        return norm, norm, h, s, v

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("sat-slider", "value", allow_duplicate=True),
    Output("val-slider", "value", allow_duplicate=True),
    Input("color-sv-input", "value"),
    prevent_initial_call=True,
)
def apply_sv_square_input(raw_value):
    text = str(raw_value or "").strip()
    if not text or "," not in text:
        raise dash.exceptions.PreventUpdate
    try:
        s_txt, v_txt = text.split(",", 1)
        s = int(round(float(s_txt)))
        v = int(round(float(v_txt)))
    except Exception:
        raise dash.exceptions.PreventUpdate
    s = max(0, min(100, s))
    v = max(0, min(100, v))
    return s, v


@app.callback(
    Output("color-sv-picker", "style"),
    Output("color-sv-dot", "style"),
    Input("hue-slider", "value"),
    Input("sat-slider", "value"),
    Input("val-slider", "value"),
)
def render_sv_square(hue, sat, val):
    h = int(hue or 0) % 360
    s = max(0, min(100, int(sat or 0)))
    v = max(0, min(100, int(val or 0)))
    picker_style = {
        "--sv-hue-color": f"hsl({h}, 100%, 50%)",
    }
    dot_style = {
        "left": f"{s}%",
        "top": f"{100 - v}%",
    }
    return picker_style, dot_style


@app.callback(
    Output("hue-gradient-box", "style"),
    Output("hue-preview-text", "children"),
    Input("hue-slider", "value"),
    Input("sat-slider", "value"),
    Input("val-slider", "value"),
)
def render_hue_preview(hue, sat, val):
    h = int(hue or 0) % 360
    s = max(0, min(100, int(sat or 0)))
    v = max(0, min(100, int(val or 0)))
    style = {
        "borderColor": f"hsl({h}, 100%, 50%)",
        "boxShadow": f"inset 0 0 0 2px hsla({h}, 100%, 60%, 0.35)",
    }
    text = f"H {h}°  |  S {s}%  |  V {v}%"
    return style, text


@app.callback(
    Output("color-preview-swatch", "style"),
    Output("color-preview-text", "children"),
    Input("color-picker-input", "value"),
    Input("color-hex-input", "value"),
)
def update_color_preview(picker_color, hex_color):
    c = _normalize_hex_color(hex_color) or _normalize_hex_color(picker_color) or "#4DAEF8"
    return {"backgroundColor": c}, c


@app.callback(
    Output("color-edit-target", "data", allow_duplicate=True),
    Output("color-map-version", "data", allow_duplicate=True),
    Output("event-style-store", "data", allow_duplicate=True),
    Output("seg-event-style-store", "data", allow_duplicate=True),
    Input("color-apply-btn", "n_clicks"),
    State("color-edit-target", "data"),
    State("color-picker-input", "value"),
    State("color-hex-input", "value"),
    State("color-map-version", "data"),
    State("event-style-store", "data"),
    State("seg-event-style-store", "data"),
    State("color-editor-event-symbol", "value"),
    State("color-editor-event-shade-enabled", "value"),
    State("color-editor-event-shade-hex", "value"),
    State("color-editor-event-shade-opacity", "value"),
    State("color-editor-event-shade-mode", "value"),
    State("color-editor-event-shade-pct-min", "value"),
    State("color-editor-event-shade-pct-max", "value"),
    prevent_initial_call=True,
)
def apply_trace_color(
    _n_clicks,
    target,
    picked_color,
    hex_color,
    version,
    event_style_store,
    seg_event_style_store,
    modal_event_symbol,
    modal_event_shade_enabled,
    modal_event_shade_hex,
    modal_event_shade_opacity,
    modal_event_shade_mode,
    modal_event_shade_pct_min,
    modal_event_shade_pct_max,
):
    global _signal_colors
    target = target or {}
    group = str(target.get("group") or "")
    signal = str(target.get("signal") or "")
    channel = str(target.get("channel") or "")
    event_key = str(target.get("event") or "")
    color_role = str(target.get("color_role") or "marker")
    final_color = _normalize_hex_color(hex_color) or _normalize_hex_color(picked_color)
    if not final_color:
        raise dash.exceptions.PreventUpdate

    if group == "channels":
        if not signal or not channel:
            raise dash.exceptions.PreventUpdate
        _signal_colors[f"{signal}.{channel}"] = final_color
        _signal_colors[f"{signal}:{channel}"] = final_color
        _signal_colors[channel] = final_color
        _save_color_mapping(color_mapping_path, _signal_colors)
        next_version = int(version or 0) + 1
        return {}, next_version, (event_style_store or {}), (seg_event_style_store or {})

    if group in {"events", "segmentation_events"}:
        if not event_key:
            raise dash.exceptions.PreventUpdate
        out = dict((seg_event_style_store if group == "segmentation_events" else event_style_store) or {})
        entry = dict(out.get(event_key) or {})
        if color_role == "shade":
            entry["shade_color"] = final_color
        else:
            entry["color"] = final_color
        symbol = str(modal_event_symbol or entry.get("symbol") or "circle")
        if symbol not in EVENT_SYMBOL_OPTIONS:
            symbol = "circle"
        entry["symbol"] = symbol
        entry["shade_enabled"] = "on" in [str(v) for v in (modal_event_shade_enabled or [])]
        shade_hex = _normalize_hex_color(modal_event_shade_hex)
        if shade_hex:
            entry["shade_color"] = shade_hex
        else:
            entry["shade_color"] = _normalize_hex_color(entry.get("shade_color")) or _normalize_hex_color(entry.get("color")) or final_color
        try:
            shade_opacity = float(modal_event_shade_opacity)
        except Exception:
            shade_opacity = 0.2
        entry["shade_opacity"] = max(0.0, min(1.0, shade_opacity))
        shade_mode = _normalize_event_shade_mode(modal_event_shade_mode)
        entry["shade_mode"] = shade_mode
        pct_min = _coerce_float_or_none(modal_event_shade_pct_min)
        pct_max = _coerce_float_or_none(modal_event_shade_pct_max)
        entry["shade_pct_min"] = 0.0 if pct_min is None else max(0.0, min(100.0, pct_min))
        entry["shade_pct_max"] = 100.0 if pct_max is None else max(0.0, min(100.0, pct_max))
        out[event_key] = entry
        if group == "events":
            _save_event_styles(color_mapping_path, out)
            return {}, int(version or 0), out, (seg_event_style_store or {})
        return {}, int(version or 0), (event_style_store or {}), out

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("color-edit-target", "data", allow_duplicate=True),
    Input("color-cancel-btn", "n_clicks"),
    Input("color-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_color_editor(_cancel_clicks, _close_clicks):
    return {}


@app.callback(
    Output("channels-store", "data"),
    Input({"type": "sig-channels", "sig": ALL}, "value"),
    State({"type": "sig-channels", "sig": ALL}, "id"),
)
def sync_channels_store(values, ids):
    out = {}
    for v, i in zip(values or [], ids or []):
        sig = i.get("sig")
        out[sig] = v or []
    return out


@app.callback(
    Output("channel-order-store", "data"),
    Input({"type": "sig-channels", "sig": ALL}, "value"),
    State({"type": "sig-channels", "sig": ALL}, "id"),
    State("channel-order-store", "data"),
)
def sync_channel_order_from_selection(values, ids, current):
    current = current or {}
    out = {}
    for v, i in zip(values or [], ids or []):
        sig = i.get("sig")
        out[sig] = _normalize_channel_order(v or [], current.get(sig))
    return out


@app.callback(
    Output("signal-axis-config-store", "data"),
    Input({"type": "sig-reverse-y", "sig": ALL}, "value"),
    Input({"type": "sig-custom-range-enabled", "sig": ALL}, "value"),
    Input({"type": "sig-y-min", "sig": ALL}, "value"),
    Input({"type": "sig-y-max", "sig": ALL}, "value"),
    State({"type": "sig-reverse-y", "sig": ALL}, "id"),
    State({"type": "sig-custom-range-enabled", "sig": ALL}, "id"),
    State({"type": "sig-y-min", "sig": ALL}, "id"),
    State({"type": "sig-y-max", "sig": ALL}, "id"),
    State("signal-axis-config-store", "data"),
)
def sync_signal_axis_config(reverse_values, enabled_values, min_values, max_values, reverse_ids, enabled_ids, min_ids, max_ids, current):
    out = dict(current or {})
    seen = set()
    for val, sid in zip(reverse_values or [], reverse_ids or []):
        sig = str((sid or {}).get("sig") or "").strip()
        if not sig:
            continue
        entry = dict(out.get(sig) or {})
        entry["reverse"] = "on" in (val or [])
        out[sig] = entry
        seen.add(sig)
    for val, sid in zip(enabled_values or [], enabled_ids or []):
        sig = str((sid or {}).get("sig") or "").strip()
        if not sig:
            continue
        entry = dict(out.get(sig) or {})
        entry["enabled"] = "on" in (val or [])
        out[sig] = entry
        seen.add(sig)
    for val, sid in zip(min_values or [], min_ids or []):
        sig = str((sid or {}).get("sig") or "").strip()
        if not sig:
            continue
        entry = dict(out.get(sig) or {})
        f = _coerce_float_or_none(val)
        if f is None:
            entry.pop("min", None)
        else:
            entry["min"] = f
        out[sig] = entry
        seen.add(sig)
    for val, sid in zip(max_values or [], max_ids or []):
        sig = str((sid or {}).get("sig") or "").strip()
        if not sig:
            continue
        entry = dict(out.get(sig) or {})
        f = _coerce_float_or_none(val)
        if f is None:
            entry.pop("max", None)
        else:
            entry["max"] = f
        out[sig] = entry
        seen.add(sig)
    for sig in list(seen):
        entry = out.get(sig) or {}
        if "reverse" not in entry and "enabled" not in entry and "min" not in entry and "max" not in entry:
            out.pop(sig, None)
    _persist_signal_axis_config(param_manager, out)
    return out


def _event_target_options(ordered_signals, channels_store, channel_order_store):
    # Event-signal mapper: targets are signal ids (not signal.channel)
    return [str(sig) for sig in (ordered_signals or []) if str(sig).strip()]


def _seed_event_target_defaults(selected_events, target_options, current_targets):
    selected_events = [str(v) for v in (selected_events or []) if str(v).strip()]
    target_options = [str(v) for v in (target_options or []) if str(v).strip()]
    current_targets = dict(current_targets or {})
    options_set = set(target_options)
    changed = False
    out = dict(current_targets)

    for ev in selected_events:
        existing = [t for t in (out.get(ev) or []) if t in options_set]
        if existing:
            if existing != (out.get(ev) or []):
                out[ev] = existing
                changed = True
            continue
        defaults = _default_targets_for_event(ev, target_options)
        defaults = [t for t in defaults if t in options_set]
        if not defaults and target_options:
            defaults = [target_options[0]]
        if defaults:
            out[ev] = defaults
            changed = True

    # Keep only valid targets for selected events; preserve all other keys as-is.
    for ev in selected_events:
        vals = [t for t in (out.get(ev) or []) if t in options_set]
        if vals != (out.get(ev) or []):
            out[ev] = vals
            changed = True

    return out, changed


@app.callback(
    Output("map-event-select", "options"),
    Output("map-event-select", "value"),
    Input("events-select", "value"),
    State("map-event-select", "value"),
)
def sync_map_event_options(events_values, current_value):
    events = sorted([str(v) for v in (events_values or []) if str(v).strip()], key=lambda x: x.lower())
    opts = [{"label": ev, "value": ev} for ev in events]
    value = current_value if current_value in events else (events[0] if events else None)
    return opts, value


@app.callback(
    Output("map-channel-select", "options"),
    Output("map-channel-select", "value"),
    Input("ordered-signals-store", "data"),
    Input("channels-store", "data"),
    Input("channel-order-store", "data"),
    State("map-channel-select", "value"),
)
def sync_map_channel_options(ordered_signals, channels_store, channel_order_store, current_value):
    target_options = _event_target_options(ordered_signals or [], channels_store or {}, channel_order_store or {})
    opts = [{"label": t, "value": t} for t in target_options]
    value = current_value if current_value in target_options else (target_options[0] if target_options else None)
    return opts, value


@app.callback(
    Output("event-targets-store", "data", allow_duplicate=True),
    Input("events-select", "value"),
    Input("ordered-signals-store", "data"),
    Input("channels-store", "data"),
    Input("channel-order-store", "data"),
    State("event-targets-store", "data"),
    prevent_initial_call=True,
)
def seed_event_targets_from_defaults(events_values, ordered_signals, channels_store, channel_order_store, current_targets):
    target_options = _event_target_options(ordered_signals or [], channels_store or {}, channel_order_store or {})
    seeded, changed = _seed_event_target_defaults(events_values, target_options, current_targets)
    if not changed:
        raise dash.exceptions.PreventUpdate
    _save_event_targets(color_mapping_path, seeded)
    return seeded


if CYTO_AVAILABLE:
    @app.callback(
        Output("event-channel-cyto", "elements"),
        Input("events-select", "value"),
        Input("ordered-signals-store", "data"),
        Input("channels-store", "data"),
        Input("channel-order-store", "data"),
        Input("event-targets-store", "data"),
        Input("event-style-store", "data"),
        Input("map-event-select", "value"),
    )
    def render_event_channel_cyto(
        events_values,
        ordered_signals,
        channels_store,
        channel_order_store,
        event_targets,
        event_styles,
        selected_map_event,
    ):
        events = [str(v) for v in (events_values or []) if str(v).strip()]
        targets = _event_target_options(ordered_signals or [], channels_store or {}, channel_order_store or {})
        target_set = set(targets)
        event_targets = dict(event_targets or {})
        event_styles = dict(event_styles or {})
        selected_map_event = str(selected_map_event or "").strip()

        elements = []
        event_step = 40
        signal_step = 30
        for i, ev in enumerate(events):
            style_saved = dict(event_styles.get(ev) or {})
            color = _normalize_hex_color(style_saved.get("color")) or _default_event_style(ev, i).get("color") or "#4C78A8"
            y = 40 + (i * event_step)
            elements.append(
                {
                    "data": {
                        "id": f"ev::{ev}",
                        "label": ev,
                        "kind": "event",
                        "key": ev,
                        "color": color,
                        "border_width": (3 if ev == selected_map_event else 1),
                        "border_color": ("#ffffff" if ev == selected_map_event else "rgba(255,255,255,0.35)"),
                    },
                    "position": {"x": 140, "y": y},
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"evport::{ev}",
                        "kind": "event-port",
                        "key": ev,
                        "color": color,
                        "border_color": color,
                    },
                    "position": {"x": 265, "y": y},
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"portlink-ev::{ev}",
                        "source": f"ev::{ev}",
                        "target": f"evport::{ev}",
                        "kind": "portlink",
                        "color": color,
                    },
                }
            )

        for i, tgt in enumerate(targets):
            sig = str(tgt or "")
            sig_channels = _normalize_channel_order(
                channels_store.get(sig) or _signal_channels(data_pkl, sig),
                channel_order_store.get(sig),
            )
            lead_ch = sig_channels[0] if sig_channels else ""
            sig_color = _resolve_trace_color(_signal_colors, sig, lead_ch) if lead_ch else None
            sig_color = _normalize_hex_color(sig_color) or _fallback_color_for_key(tgt)
            y = 40 + (i * signal_step)
            elements.append(
                {
                    "data": {
                        "id": f"sig::{tgt}",
                        "label": sig,
                        "kind": "signal",
                        "key": tgt,
                        "color": sig_color,
                    },
                    "position": {"x": 780, "y": y},
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"sigport::{tgt}",
                        "kind": "signal-port",
                        "key": tgt,
                        "color": sig_color,
                        "border_color": sig_color,
                    },
                    "position": {"x": 655, "y": y},
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"portlink-sig::{tgt}",
                        "source": f"sigport::{tgt}",
                        "target": f"sig::{tgt}",
                        "kind": "portlink",
                        "color": sig_color,
                    },
                }
            )

        for ev in events:
            for tgt in [str(v) for v in (event_targets.get(ev) or []) if str(v).strip()]:
                if tgt not in target_set:
                    continue
                edge_id = f"edge::{ev}||{tgt}"
                elements.append(
                    {
                        "data": {
                            "id": edge_id,
                            "source": f"evport::{ev}",
                            "target": f"sigport::{tgt}",
                            "event": ev,
                            "target_key": tgt,
                            "kind": "mapping",
                            "edge_opacity": (1.0 if (not selected_map_event or ev == selected_map_event) else 0.15),
                            "edge_width": (3 if ev == selected_map_event else 2),
                        },
                    }
                )

        return elements


    @app.callback(
        Output("map-selected-event", "data"),
        Input("event-channel-cyto", "tapNodeData"),
        prevent_initial_call=True,
    )
    def set_selected_event_from_cyto(node_data):
        node_data = node_data or {}
        if str(node_data.get("kind") or "") in {"event", "event-port"}:
            return str(node_data.get("key") or "")
        raise dash.exceptions.PreventUpdate


    @app.callback(
        Output("event-targets-store", "data", allow_duplicate=True),
        Input("event-channel-cyto", "tapNodeData"),
        Input("event-channel-cyto", "tapEdgeData"),
        State("map-selected-event", "data"),
        State("event-targets-store", "data"),
        prevent_initial_call=True,
    )
    def update_event_targets_from_cyto(tap_node_data, tap_edge_data, selected_event, current_targets):
        prop = (ctx.triggered[0].get("prop_id") if ctx.triggered else "")
        out = dict(current_targets or {})

        if prop.endswith(".tapEdgeData"):
            if isinstance(tap_edge_data, dict) and tap_edge_data.get("event") and tap_edge_data.get("target_key"):
                ev = str(tap_edge_data.get("event") or "").strip()
                tgt = str(tap_edge_data.get("target_key") or "").strip()
                if not ev or not tgt:
                    raise dash.exceptions.PreventUpdate
                existing = [str(v) for v in (out.get(ev) or []) if str(v).strip()]
                out[ev] = [v for v in existing if v != tgt]
                _save_event_targets(color_mapping_path, out)
                return out
            raise dash.exceptions.PreventUpdate

        if prop.endswith(".tapNodeData"):
            if isinstance(tap_node_data, dict) and str(tap_node_data.get("kind") or "") in {"signal", "signal-port"}:
                ev = str(selected_event or "").strip()
                tgt = str(tap_node_data.get("key") or "").strip()
                if not ev or not tgt:
                    raise dash.exceptions.PreventUpdate
                existing = [str(v) for v in (out.get(ev) or []) if str(v).strip()]
                if tgt in existing:
                    out[ev] = [v for v in existing if v != tgt]
                else:
                    out[ev] = existing + [tgt]
                _save_event_targets(color_mapping_path, out)
                return out
            raise dash.exceptions.PreventUpdate

        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("map-links-chips", "children"),
    Input("events-select", "value"),
    Input("event-targets-store", "data"),
)
def render_map_links(events_values, event_targets):
    events = [str(v) for v in (events_values or []) if str(v).strip()]
    targets = dict(event_targets or {})
    chips = []
    for ev in events:
        for t in [str(v) for v in (targets.get(ev) or []) if str(v).strip()]:
            chips.append(
                html.Div(
                    [
                        html.Span(f"{ev} -> {t}", className="chip-label"),
                        html.Button(
                            "×",
                            id={"type": "map-link-remove", "event": ev, "target": t},
                            className="chip-remove-btn",
                            n_clicks=0,
                            title="Remove link",
                        ),
                    ],
                    className="chip-item",
                )
            )
    if not chips:
        return [html.Small("No links. Choose an event and signal, then click Add Link.")]
    return chips


@app.callback(
    Output("event-targets-store", "data", allow_duplicate=True),
    Input("map-add-link", "n_clicks"),
    Input({"type": "map-link-remove", "event": ALL, "target": ALL}, "n_clicks"),
    State("map-event-select", "value"),
    State("map-channel-select", "value"),
    State("event-targets-store", "data"),
    prevent_initial_call=True,
)
def update_event_targets_from_mapper(_add_clicks, _remove_clicks, selected_event, selected_target, current_targets):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    out = dict(current_targets or {})

    if trig == "map-add-link":
        ev = str(selected_event or "").strip()
        tgt = str(selected_target or "").strip()
        if not ev or not tgt:
            raise dash.exceptions.PreventUpdate
        existing = [str(v) for v in (out.get(ev) or []) if str(v).strip()]
        if tgt not in existing:
            out[ev] = existing + [tgt]
        _save_event_targets(color_mapping_path, out)
        return out

    if isinstance(trig, dict) and trig.get("type") == "map-link-remove":
        ev = str(trig.get("event") or "").strip()
        tgt = str(trig.get("target") or "").strip()
        if not ev or not tgt:
            raise dash.exceptions.PreventUpdate
        existing = [str(v) for v in (out.get(ev) or []) if str(v).strip()]
        out[ev] = [v for v in existing if v != tgt]
        _save_event_targets(color_mapping_path, out)
        return out

    raise dash.exceptions.PreventUpdate


def _trace_xaxis_keys(fig_obj):
    """Return x-axis layout keys actually used by traces."""
    keys = set()
    for trace in getattr(fig_obj, "data", []):
        xa = getattr(trace, "xaxis", None) or "x"
        if xa == "x":
            keys.add("xaxis")
        elif isinstance(xa, str) and xa.startswith("x"):
            suffix = xa[1:]
            if suffix.isdigit():
                keys.add(f"xaxis{suffix}")
            else:
                keys.add("xaxis")
    return keys or {"xaxis"}


def _clean_relayout_data(relayout_data, allowed_xaxis_keys):
    """Keep relayout entries only for active x-axes."""
    if not isinstance(relayout_data, dict):
        return {}

    allowed_suffixes = {"range[0]", "range[1]", "range"}
    out = {}
    for key, value in relayout_data.items():
        if not isinstance(key, str) or not key.startswith("xaxis"):
            continue
        axis_key = key.split(".", 1)[0]
        if axis_key in allowed_xaxis_keys:
            suffix = key[len(axis_key) + 1 :] if "." in key else ""
            if suffix in allowed_suffixes:
                out[key] = value

    # Expand list-style range payloads into explicit range bounds.
    for axis_key in allowed_xaxis_keys:
        range_key = f"{axis_key}.range"
        if range_key in out and f"{axis_key}.range[0]" not in out and f"{axis_key}.range[1]" not in out:
            xr = out.get(range_key)
            if isinstance(xr, (list, tuple)) and len(xr) == 2:
                out[f"{axis_key}.range[0]"] = xr[0]
                out[f"{axis_key}.range[1]"] = xr[1]

    # Mirror primary zoom range across active subplot axes.
    if "xaxis.range[0]" in out and "xaxis.range[1]" in out:
        lo = out["xaxis.range[0]"]
        hi = out["xaxis.range[1]"]
        for axis_key in allowed_xaxis_keys:
            out.setdefault(f"{axis_key}.range[0]", lo)
            out.setdefault(f"{axis_key}.range[1]", hi)
            out.pop(f"{axis_key}.range", None)

    return out


def _extract_primary_xaxis_range(relayout_data):
    """Return (x0, x1) from Plotly relayout payload when present."""
    if not isinstance(relayout_data, dict):
        return None, None
    x0 = relayout_data.get("xaxis.range[0]")
    x1 = relayout_data.get("xaxis.range[1]")
    if x0 is not None and x1 is not None:
        return x0, x1
    xr = relayout_data.get("xaxis.range")
    if isinstance(xr, (list, tuple)) and len(xr) == 2:
        return xr[0], xr[1]
    return None, None


def _coerce_epoch_from_relayout_value(value):
    """Parse relayout x-values (datetime string, sec epoch, or ms epoch) to seconds."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            v = float(value)
            if v > 1e12:
                return v / 1000.0
            if v > 1e9:
                return v
            return float(pd.Timestamp(v, unit="s").timestamp())
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz_name)
        else:
            ts = ts.tz_convert(tz_name)
        return float(ts.tz_convert("UTC").timestamp())
    except Exception:
        return None


def _epoch_within_slider_bounds(epoch_value, lo_bound, hi_bound, *, pad_seconds=0.0):
    """Return True when epoch_value is plausibly within dataset slider bounds."""
    if epoch_value is None:
        return False
    try:
        epoch_f = float(epoch_value)
        lo_f = float(lo_bound) - float(pad_seconds)
        hi_f = float(hi_bound) + float(pad_seconds)
    except Exception:
        return False
    return lo_f <= epoch_f <= hi_f


def _estimate_axis_y_range(fig_obj, axis_ref):
    vals = []
    for trace in getattr(fig_obj, "data", []):
        tr_axis = getattr(trace, "yaxis", None) or "y"
        if tr_axis != axis_ref:
            continue
        y = getattr(trace, "y", None)
        if y is None:
            continue
        for v in y:
            try:
                f = float(v)
            except Exception:
                continue
            if pd.notna(f):
                vals.append(f)
    if not vals:
        return 0.0, 1.0
    return float(min(vals)), float(max(vals))


@app.callback(
    Output("time-zone-used-label", "children"),
    Input("time-display-tz-toggle", "value"),
)
def update_time_zone_used_label(display_mode):
    global time_display_tz_mode
    mode = str(display_mode or "local")
    time_display_tz_mode = mode
    return _time_display_label(mode)


@app.callback(
    Output("time-range-slider", "value", allow_duplicate=True),
    Output("start-time", "value", allow_duplicate=True),
    Output("end-time", "value", allow_duplicate=True),
    Input("start-time", "value"),
    Input("end-time", "value"),
    State("time-display-tz-toggle", "value"),
    State("time-range-slider", "value"),
    State("time-range-slider", "min"),
    State("time-range-slider", "max"),
    prevent_initial_call=True,
)
def sync_window_from_text(start_text, end_text, display_mode, slider_value, s_min, s_max):
    s_min = int(s_min if s_min is not None else slider_min)
    s_max = int(s_max if s_max is not None else slider_max)
    default_lo = int(_to_epoch_seconds(start_default))
    default_hi = int(_to_epoch_seconds(end_default))

    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        cur_lo = int(min(slider_value))
        cur_hi = int(max(slider_value))
    else:
        cur_lo = max(s_min, min(default_lo, s_max))
        cur_hi = max(s_min, min(default_hi, s_max))

    min_gap = MIN_WINDOW_SECONDS
    display_tz = _display_tz_name(display_mode)
    start_ts = _parse_dt_or_fallback(start_text, _from_epoch_seconds(cur_lo, display_tz), display_tz)
    end_ts = _parse_dt_or_fallback(end_text, _from_epoch_seconds(cur_hi, display_tz), display_tz)
    lo = int(_to_epoch_seconds(start_ts))
    hi = int(_to_epoch_seconds(end_ts))

    lo = max(s_min, min(lo, s_max))
    hi = max(s_min, min(hi, s_max))
    if hi < lo + min_gap:
        trigger = ctx.triggered_id
        if trigger == "start-time":
            hi = min(s_max, lo + min_gap)
            if hi < lo + min_gap:
                lo = max(s_min, hi - min_gap)
        else:
            lo = max(s_min, hi - min_gap)
            if hi < lo + min_gap:
                hi = min(s_max, lo + min_gap)

    lo = max(s_min, min(lo, s_max))
    hi = max(s_min, min(hi, s_max))
    if hi < lo:
        lo, hi = hi, lo
    if hi < lo + min_gap:
        hi = min(s_max, lo + min_gap)
        if hi < lo + min_gap:
            lo = max(s_min, hi - min_gap)

    lo_ts = _from_epoch_seconds(lo, display_tz)
    hi_ts = _from_epoch_seconds(hi, display_tz)
    return [lo, hi], _format_ts_input(lo_ts, display_tz), _format_ts_input(hi_ts, display_tz)


@app.callback(
    Output("start-time", "value", allow_duplicate=True),
    Output("end-time", "value", allow_duplicate=True),
    Input("time-range-slider", "value"),
    Input("time-display-tz-toggle", "value"),
    prevent_initial_call=True,
)
def sync_text_from_slider(slider_value, display_mode):
    if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
        raise dash.exceptions.PreventUpdate
    lo = int(min(slider_value))
    hi = int(max(slider_value))
    display_tz = _display_tz_name(display_mode)
    lo_ts = _from_epoch_seconds(lo, display_tz)
    hi_ts = _from_epoch_seconds(hi, display_tz)
    return _format_ts_input(lo_ts, display_tz), _format_ts_input(hi_ts, display_tz)


@app.callback(
    Output("time-range-slider", "value", allow_duplicate=True),
    Output("start-time", "value", allow_duplicate=True),
    Output("end-time", "value", allow_duplicate=True),
    Output("playhead-time", "data", allow_duplicate=True),
    Input("main-plot", "relayoutData"),
    State("time-display-tz-toggle", "value"),
    State("time-range-slider", "min"),
    State("time-range-slider", "max"),
    State("main-plot", "figure"),
    prevent_initial_call=True,
)
def sync_window_from_plot_zoom(relayout_data, display_mode, s_min, s_max, existing_fig):
    # Prioritize explicit playhead-shape drag updates over axis-range updates.
    # Relayout payloads can contain multiple shape keys; only accept the yellow playhead line.
    try:
        if isinstance(relayout_data, dict):
            shape_updates = {}
            for key, val in relayout_data.items():
                m = re.match(r"^shapes\[(\d+)\]\.(x0|x1)$", str(key))
                if not m:
                    continue
                idx = int(m.group(1))
                field = m.group(2)
                shape_updates.setdefault(idx, {})[field] = val
            if shape_updates:
                s_min_v = float(s_min if s_min is not None else slider_min)
                s_max_v = float(s_max if s_max is not None else slider_max)
                fig_shapes = ((existing_fig or {}).get("layout") or {}).get("shapes") or []
                for idx, vals in shape_updates.items():
                    if idx < 0 or idx >= len(fig_shapes):
                        continue
                    shp = fig_shapes[idx] if isinstance(fig_shapes[idx], dict) else {}
                    is_playhead = (
                        shp.get("type") == "line"
                        and shp.get("yref") == "paper"
                        and str(((shp.get("line") or {}).get("color") or "")).upper() == "#FFD166"
                    )
                    if not is_playhead:
                        continue
                    raw_x = vals.get("x0", vals.get("x1"))
                    epoch = _coerce_epoch_from_relayout_value(raw_x)
                    if epoch is None:
                        continue
                    p = max(s_min_v, min(s_max_v, float(epoch)))
                    return no_update, no_update, no_update, p
    except Exception:
        pass

    x0, x1 = _extract_primary_xaxis_range(relayout_data)
    # Case 1: zoom/pan changed x-axis range -> sync active window and recenter playhead.
    if x0 is not None and x1 is not None:
        lo_epoch = _coerce_epoch_from_relayout_value(x0)
        hi_epoch = _coerce_epoch_from_relayout_value(x1)
        if lo_epoch is None or hi_epoch is None:
            raise dash.exceptions.PreventUpdate
        lo = int(lo_epoch)
        hi = int(hi_epoch)
        if hi < lo:
            lo, hi = hi, lo

        s_min = int(s_min if s_min is not None else slider_min)
        s_max = int(s_max if s_max is not None else slider_max)
        # Ignore malformed numeric relayout ranges (e.g., [-1, 6]) that can appear
        # during transient redraw states and would break datetime synchronization.
        if not (
            _epoch_within_slider_bounds(lo, s_min, s_max, pad_seconds=300)
            and _epoch_within_slider_bounds(hi, s_min, s_max, pad_seconds=300)
        ):
            raise dash.exceptions.PreventUpdate
        lo = max(s_min, min(lo, s_max))
        hi = max(s_min, min(hi, s_max))
        if hi < lo + MIN_WINDOW_SECONDS:
            hi = min(s_max, lo + MIN_WINDOW_SECONDS)
            if hi < lo + MIN_WINDOW_SECONDS:
                lo = max(s_min, hi - MIN_WINDOW_SECONDS)

        display_tz = _display_tz_name(display_mode)
        lo_ts = _from_epoch_seconds(lo, display_tz)
        hi_ts = _from_epoch_seconds(hi, display_tz)
        playhead_epoch = float(lo + (0.5 * (hi - lo)))
        return [lo, hi], _format_ts_input(lo_ts, display_tz), _format_ts_input(hi_ts, display_tz), playhead_epoch

    raise dash.exceptions.PreventUpdate


def _clamp_window(lo, hi, s_min, s_max):
    """Clamp a [lo, hi] window into [s_min, s_max] preserving MIN_WINDOW_SECONDS."""
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    span = max(float(MIN_WINDOW_SECONDS), hi - lo)
    full = max(float(MIN_WINDOW_SECONDS), s_max - s_min)
    span = min(span, full)
    # Keep the window inside bounds by shifting rather than squashing where possible.
    if lo < s_min:
        lo, hi = s_min, s_min + span
    if hi > s_max:
        hi, lo = s_max, s_max - span
    lo = max(s_min, min(lo, s_max))
    hi = max(s_min, min(hi, s_max))
    if hi < lo + MIN_WINDOW_SECONDS:
        hi = min(s_max, lo + MIN_WINDOW_SECONDS)
        if hi < lo + MIN_WINDOW_SECONDS:
            lo = max(s_min, hi - MIN_WINDOW_SECONDS)
    return int(round(lo)), int(round(hi))


@app.callback(
    Output("time-range-slider", "value", allow_duplicate=True),
    Output("start-time", "value", allow_duplicate=True),
    Output("end-time", "value", allow_duplicate=True),
    Input("tnav-zoom-in", "n_clicks"),
    Input("tnav-zoom-out", "n_clicks"),
    Input("tnav-full", "n_clicks"),
    Input("tnav-pan-left", "n_clicks"),
    Input("tnav-pan-right", "n_clicks"),
    Input({"type": "tnav-preset", "seconds": ALL}, "n_clicks"),
    State("time-range-slider", "value"),
    State("time-range-slider", "min"),
    State("time-range-slider", "max"),
    State("playhead-time", "data"),
    State("time-display-tz-toggle", "value"),
    prevent_initial_call=True,
)
def time_nav_zoom_pan(_zi, _zo, _full, _pl, _pr, _presets, slider_value, s_min, s_max, playhead_time, display_mode):
    trigger = ctx.triggered_id
    if trigger is None:
        raise dash.exceptions.PreventUpdate
    # Ignore the initial 0-click fan-out that Dash emits for pattern-matched inputs.
    if not ctx.triggered or ctx.triggered[0].get("value") in (None, 0):
        raise dash.exceptions.PreventUpdate

    s_min = int(s_min if s_min is not None else slider_min)
    s_max = int(s_max if s_max is not None else slider_max)
    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        lo = float(min(slider_value))
        hi = float(max(slider_value))
    else:
        lo, hi = float(s_min), float(s_max)
    span = max(float(MIN_WINDOW_SECONDS), hi - lo)
    center = 0.5 * (lo + hi)

    if isinstance(trigger, dict) and trigger.get("type") == "tnav-preset":
        # Center the requested duration on the current playhead when available.
        try:
            ph = float(playhead_time)
        except Exception:
            ph = center
        target = float(trigger.get("seconds") or span)
        lo = ph - target / 2.0
        hi = ph + target / 2.0
    elif trigger == "tnav-zoom-in":
        new_span = max(float(MIN_WINDOW_SECONDS), span / 2.0)
        lo, hi = center - new_span / 2.0, center + new_span / 2.0
    elif trigger == "tnav-zoom-out":
        new_span = span * 2.0
        lo, hi = center - new_span / 2.0, center + new_span / 2.0
    elif trigger == "tnav-full":
        lo, hi = float(s_min), float(s_max)
    elif trigger == "tnav-pan-left":
        shift = span * 0.5
        lo, hi = lo - shift, hi - shift
    elif trigger == "tnav-pan-right":
        shift = span * 0.5
        lo, hi = lo + shift, hi + shift
    else:
        raise dash.exceptions.PreventUpdate

    lo, hi = _clamp_window(lo, hi, s_min, s_max)
    display_tz = _display_tz_name(display_mode)
    lo_ts = _from_epoch_seconds(lo, display_tz)
    hi_ts = _from_epoch_seconds(hi, display_tz)
    return [lo, hi], _format_ts_input(lo_ts, display_tz), _format_ts_input(hi_ts, display_tz)


@app.callback(
    Output("playhead-time", "data", allow_duplicate=True),
    Input("tnav-ph-start", "n_clicks"),
    Input("tnav-ph-end", "n_clicks"),
    Input("tnav-ph-back", "n_clicks"),
    Input("tnav-ph-fwd", "n_clicks"),
    State("time-range-slider", "value"),
    State("playhead-time", "data"),
    prevent_initial_call=True,
)
def time_nav_playhead(_start, _end, _back, _fwd, slider_value, playhead_time):
    trigger = ctx.triggered_id
    if trigger is None or not ctx.triggered or ctx.triggered[0].get("value") in (None, 0):
        raise dash.exceptions.PreventUpdate
    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        lo = float(min(slider_value))
        hi = float(max(slider_value))
    else:
        lo, hi = float(slider_min), float(slider_max)
    try:
        cur = float(playhead_time)
    except Exception:
        cur = _default_playhead_epoch([lo, hi])

    if trigger == "tnav-ph-start":
        cur = lo
    elif trigger == "tnav-ph-end":
        cur = hi
    elif trigger == "tnav-ph-back":
        cur = cur - 5.0
    elif trigger == "tnav-ph-fwd":
        cur = cur + 5.0
    cur = max(lo, min(hi, cur))
    return round(cur, 3)


@app.callback(
    Output("playhead-slider", "min", allow_duplicate=True),
    Output("playhead-slider", "max", allow_duplicate=True),
    Output("playhead-time", "data", allow_duplicate=True),
    Input("time-range-slider", "value"),
    State("playhead-time", "data"),
    prevent_initial_call=True,
)
def clamp_playhead_to_active_window(slider_value, playhead_time):
    if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
        raise dash.exceptions.PreventUpdate
    lo = float(min(slider_value))
    hi = float(max(slider_value))
    if hi <= lo:
        hi = lo + 1.0
    try:
        cur = float(playhead_time)
    except Exception:
        cur = float(_default_playhead_epoch([lo, hi]))
    cur = max(lo, min(hi, cur))
    return lo, hi, cur


@app.callback(
    Output("playhead-time", "data", allow_duplicate=True),
    Input("playhead-slider", "value"),
    State("playhead-slider", "min"),
    State("playhead-slider", "max"),
    prevent_initial_call=True,
)
def sync_playhead_from_slider(playhead_value, p_min, p_max):
    candidate = playhead_value
    try:
        value = float(candidate)
    except Exception:
        raise dash.exceptions.PreventUpdate
    try:
        lo = float(p_min)
        hi = float(p_max)
        value = max(lo, min(hi, value))
    except Exception:
        pass
    return value


@app.callback(
    Output("playhead-time", "data", allow_duplicate=True),
    Input("main-plot", "clickData"),
    State("playhead-slider", "min"),
    State("playhead-slider", "max"),
    prevent_initial_call=True,
)
def sync_playhead_from_plot(click_data, p_min, p_max):
    p_min = float(p_min if p_min is not None else slider_min)
    p_max = float(p_max if p_max is not None else slider_max)

    def _clamp_epoch(value):
        ts = _coerce_epoch_from_relayout_value(value)
        if ts is None:
            return None
        return max(p_min, min(p_max, ts))

    if isinstance(click_data, dict):
        points = click_data.get("points") or []
        if points:
            x = points[0].get("x")
            out = _clamp_epoch(x)
            if out is not None:
                return float(out)

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("playhead-slider", "value", allow_duplicate=True),
    Input("playhead-time", "data"),
    State("playhead-slider", "min"),
    State("playhead-slider", "max"),
    prevent_initial_call=True,
)
def sync_slider_from_playhead(playhead_time, p_min, p_max):
    try:
        val = float(playhead_time)
    except Exception:
        raise dash.exceptions.PreventUpdate
    try:
        lo = float(p_min)
        hi = float(p_max)
        val = max(lo, min(hi, val))
    except Exception:
        pass
    return val


@app.callback(
    Output("playhead-time", "data", allow_duplicate=True),
    Input("arrow-key-input", "value"),
    State("playhead-time", "data"),
    State("playhead-slider", "min"),
    State("playhead-slider", "max"),
    prevent_initial_call=True,
)
def nudge_playhead_from_arrow_key(arrow_value, current_playhead, p_min, p_max):
    if not arrow_value:
        raise dash.exceptions.PreventUpdate
    try:
        direction_text = str(arrow_value).split(":", 1)[0].strip()
        direction = int(direction_text)
        if direction not in (-1, 1):
            raise ValueError("invalid direction")
    except Exception:
        raise dash.exceptions.PreventUpdate

    try:
        cur = float(current_playhead if current_playhead is not None else _default_playhead_epoch(slider_default))
    except Exception:
        cur = float(_default_playhead_epoch(slider_default))
    try:
        lo = float(p_min if p_min is not None else slider_min)
        hi = float(p_max if p_max is not None else slider_max)
    except Exception:
        lo = float(slider_min)
        hi = float(slider_max)

    new_value = cur + (5.0 * direction)
    new_value = max(lo, min(hi, new_value))
    return round(new_value, 3)


@app.callback(
    Output("playhead-summary", "children"),
    Input("playhead-time", "data"),
)
def update_playhead_summary(playhead_value):
    try:
        playhead_ts = _from_epoch_seconds(float(playhead_value), tz_name)
    except Exception:
        playhead_ts = _from_epoch_seconds(float(_default_playhead_epoch(slider_default)), tz_name)
    return f"{playhead_ts.strftime('%H:%M:%S.000')}  ·  {playhead_ts.strftime('%Y-%m-%d')}"


@app.callback(
    Output("sync-video", "src"),
    Output("sync-video-status", "children"),
    Output("sync-video-current-clip", "data"),
    Input("playhead-time", "data"),
    Input("sync-video-follow", "value"),
    State("sync-video-current-clip", "data"),
    prevent_initial_call=False,
)
def sync_video_to_playhead(playhead_value, follow_value, current_clip):
    follow = bool(follow_value) and "on" in (follow_value or [])
    if not _video_clip_index:
        return "", f"No video clips found in {VIDEO_DIR}", None

    clip, offset = _find_video_clip_for_epoch(playhead_value)
    if clip is None:
        # Playhead is outside every clip's range: keep whatever is loaded but note
        # the gap, including the nearest footage boundary so it's clear this is a
        # real coverage gap (not a sync bug).
        try:
            ph_epoch = float(playhead_value)
            ph_ts = _from_epoch_seconds(ph_epoch, tz_name).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ph_epoch, ph_ts = None, "?"
        nearest_txt = ""
        if ph_epoch is not None and _video_clip_index:
            edges = []
            for c in _video_clip_index:
                edges.append((abs(c["end_epoch"] - ph_epoch), c["end_epoch"], "ends"))
                edges.append((abs(c["start_epoch"] - ph_epoch), c["start_epoch"], "starts"))
            gap_s, edge_epoch, kind = min(edges, key=lambda e: e[0])
            # Show the date too when the gap spans days (a multi-day gap almost
            # always means the loaded clips belong to a different deployment/date).
            fmt = "%H:%M:%S" if gap_s < 86400 else "%Y-%m-%d %H:%M:%S"
            edge_ts = _from_epoch_seconds(edge_epoch, tz_name).strftime(fmt)
            nearest_txt = f" — nearest footage {kind} {edge_ts} ({_format_gap(gap_s)} away)"
        elif not _video_clip_index:
            nearest_txt = " — no clips loaded for this deployment"
        return no_update, f"No video covers {ph_ts}{nearest_txt}", current_clip

    offset_int = int(offset)
    ph_ts = _from_epoch_seconds(clip["start_epoch"] + offset, tz_name).strftime("%H:%M:%S")
    status = f"{clip['name']}  ·  +{offset_int // 60:d}:{offset_int % 60:02d}  ·  {ph_ts}"

    # Only rewrite src when the clip changes; otherwise the <video> would reload and
    # jump back to the fragment offset on every playhead tick. In-clip seeking is
    # handled by the clientside callback below (sets video.currentTime, no reload).
    clip_state = {"name": clip["name"], "start_epoch": clip["start_epoch"]}
    if not follow:
        raise dash.exceptions.PreventUpdate
    if isinstance(current_clip, dict) and current_clip.get("name") == clip["name"]:
        return no_update, status, clip_state

    # NOTE: no "#t=" fragment. The fragment plus JS currentTime writes race each
    # other and can leave the element parked mid-seek (black frame). The clientside
    # seeker below owns positioning. The url is the Immich proxy (/immich-video/<id>)
    # or the local route (/local-video/<name>), set when the index was built.
    src = clip.get("url") or f"/local-video/{clip['name']}"
    return src, status, clip_state


# Seek the loaded video to the playhead position.
#
# CRITICAL: only seek while the video is PAUSED (user scrubbing). During playback
# the <video> is the master clock and plays untouched — writing currentTime on
# every 10 Hz playhead tick would interrupt playback with a seek each time the
# clock drifted, leaving the frame perpetually blank while audio kept buffering
# (the "audio plays, no video" symptom). The playhead is instead driven FROM the
# video via 'timeupdate' (see below) while playing.
app.clientside_callback(
    """
    function(playhead, follow, clip) {
        const on = Array.isArray(follow) && follow.indexOf("on") !== -1;
        if (!on || !clip || clip.start_epoch == null || playhead == null) {
            return window.dash_clientside.no_update;
        }
        const v = document.getElementById("sync-video");
        if (!v) { return window.dash_clientside.no_update; }
        // Keep the master-clock offset current even as clips change mid-playback.
        v._clipStart = Number(clip.start_epoch);
        // If a clip change happened while playing, resume native playback on the
        // freshly loaded source once it's ready.
        const btn = document.getElementById("play-pause-btn");
        const wantPlaying = btn && btn.getAttribute("data-playing") === "1";
        if (wantPlaying) {
            if (v.paused) {
                const kick = function () { const p = v.play(); if (p && p.catch) p.catch(function(){}); };
                if (v.readyState >= 2) { kick(); }
                else { v.addEventListener("canplay", kick, { once: true }); }
            }
            return window.dash_clientside.no_update;
        }
        // Never fight native playback.
        if (!v.paused && !v.ended) { return window.dash_clientside.no_update; }
        const target = Math.max(0, Number(playhead) - Number(clip.start_epoch));
        if (!isFinite(target)) { return window.dash_clientside.no_update; }

        const seekTo = function () {
            if (v.seeking) { v._pendingSeek = target; return; }
            if (Math.abs((v.currentTime || 0) - target) > 0.08) {
                try { v.currentTime = target; } catch (e) {}
            }
        };
        if (!v._seekWired) {
            v._seekWired = true;
            v.addEventListener("seeked", function () {
                if (v._pendingSeek != null) {
                    const t = v._pendingSeek; v._pendingSeek = null;
                    if (!v.seeking && Math.abs((v.currentTime || 0) - t) > 0.08) {
                        try { v.currentTime = t; } catch (e) {}
                    }
                }
            });
        }
        if (v.readyState >= 1) { seekTo(); }
        else { v.addEventListener("loadedmetadata", seekTo, { once: true }); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("sync-video", "title"),
    Input("playhead-time", "data"),
    Input("sync-video-follow", "value"),
    Input("sync-video-current-clip", "data"),
)


# --- Single play/pause button driving the shared playback clock -------------
# Toggles window.IntegratedPlayback, the is-playing store, the interval poller,
# the button glyph, and native <video> play/pause. Modeled on EcoPhysVideoViz.
app.clientside_callback(
    """
    function(n_clicks, playhead, ph_min, ph_max, rate, clip) {
        const mgr = window.IntegratedPlayback;
        const btn = document.getElementById("play-pause-btn");
        const v = document.getElementById("sync-video");
        if (!mgr) { return [false, true, "▶"]; }
        const willPlay = (Number(n_clicks) || 0) % 2 === 1;
        const hasClip = !!(clip && clip.start_epoch != null);
        mgr.setBounds(ph_min, ph_max);
        mgr.setPlaybackRate(rate || 1);

        // Wire the video as the master clock: while it plays, drive playhead-time
        // FROM the video via timeupdate (set_props), so the plot follows the real
        // frames and we never seek the element mid-play.
        if (v && !v._timeupdateWired) {
            v._timeupdateWired = true;
            v.addEventListener("timeupdate", function () {
                if (v.paused || v.ended) { return; }
                const cs = (v._clipStart != null) ? v._clipStart : null;
                if (cs == null) { return; }
                window.dash_clientside.set_props("playhead-time", { data: cs + v.currentTime });
            });
            v.addEventListener("ended", function () {
                const b = document.getElementById("play-pause-btn");
                if (b && b.getAttribute("data-playing") === "1") { b.click(); }
            });
        }
        if (v) { v._clipStart = hasClip ? Number(clip.start_epoch) : null; }

        if (willPlay) {
            if (btn) btn.setAttribute("data-playing", "1");
            if (hasClip && v) {
                // Video is master: play it natively; the rAF manager stays idle.
                try { v.playbackRate = Math.max(0.25, Math.min(16, Number(rate) || 1)); } catch (e) {}
                const p = v.play(); if (p && p.catch) p.catch(function () {});
                mgr.stop();
                return [true, true, "⏸"];  // interval disabled — timeupdate drives playhead
            } else {
                // No video here (gap): the rAF manager is master and advances playhead.
                mgr.syncTime(playhead != null ? playhead : ph_min);
                mgr.start();
                return [true, false, "⏸"];  // interval enabled — manager drives playhead
            }
        } else {
            mgr.stop();
            if (btn) btn.setAttribute("data-playing", "0");
            if (v) { try { v.pause(); } catch (e) {} }
            return [false, true, "▶"];
        }
    }
    """,
    Output("is-playing", "data"),
    Output("playback-interval", "disabled"),
    Output("play-pause-btn", "children"),
    Input("play-pause-btn", "n_clicks"),
    State("playhead-time", "data"),
    State("playhead-slider", "min"),
    State("playhead-slider", "max"),
    State("playback-rate", "data"),
    State("sync-video-current-clip", "data"),
    prevent_initial_call=True,
)


# The interval poll reads the manager clock into playhead-time while playing in a
# GAP (no video). When the video is master the interval is disabled and timeupdate
# drives the playhead instead.
app.clientside_callback(
    """
    function(n_intervals, is_playing) {
        if (!is_playing) { return window.dash_clientside.no_update; }
        const mgr = window.IntegratedPlayback;
        if (!mgr || !mgr.isPlaying || mgr.currentTime == null) { return window.dash_clientside.no_update; }
        return mgr.currentTime;
    }
    """,
    Output("playhead-time", "data", allow_duplicate=True),
    Input("playback-interval", "n_intervals"),
    State("is-playing", "data"),
    prevent_initial_call=True,
)


# Keep the manager's bounds and rate in sync with the window and rate selector.
app.clientside_callback(
    """
    function(rate, ph_min, ph_max) {
        const mgr = window.IntegratedPlayback;
        if (mgr) {
            mgr.setPlaybackRate(rate || 1);
            mgr.setBounds(ph_min, ph_max);
            const v = document.getElementById("sync-video");
            if (v) { try { v.playbackRate = Math.max(0.25, Math.min(16, Number(rate) || 1)); } catch (e) {} }
        }
        return rate || 1;
    }
    """,
    Output("playback-rate", "data"),
    Input("playback-rate-select", "value"),
    Input("playhead-slider", "min"),
    Input("playhead-slider", "max"),
)


# When the playhead moves from a NON-playback source (slider drag, plot click,
# arrow keys, zoom), keep the manager clock aligned so resuming play continues
# from the right spot. Gated on !is_playing to avoid fighting the poller.
app.clientside_callback(
    """
    function(playhead, is_playing) {
        const mgr = window.IntegratedPlayback;
        if (mgr && !is_playing && playhead != null) {
            mgr.syncTime(playhead);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("play-pause-btn", "title"),
    Input("playhead-time", "data"),
    State("is-playing", "data"),
)


# Move the main-plot yellow playhead line clientside (patch the paper-ref line
# shape via Plotly.relayout) on EVERY playhead change — drag, click, or playback.
# This keeps the line glued to the playhead during live scrubbing without a
# full-figure server round-trip per tick.
app.clientside_callback(
    """
    function(playhead, is_playing) {
        if (playhead == null) { return window.dash_clientside.no_update; }
        const gd = document.getElementById("main-plot");
        const inner = gd && gd.querySelector(".js-plotly-plot");
        if (!inner || !inner._fullLayout || !inner._fullLayout.shapes) {
            return window.dash_clientside.no_update;
        }
        const shapes = inner._fullLayout.shapes;
        // Plotly datetime axes expect ms since epoch for shape x coordinates.
        const xms = Number(playhead) * 1000;
        for (let i = 0; i < shapes.length; i++) {
            const s = shapes[i];
            const col = ((s.line && s.line.color) || "").toString().toUpperCase();
            if (s.type === "line" && s.yref === "paper" && col === "#FFD166") {
                const patch = {};
                patch["shapes[" + i + "].x0"] = xms;
                patch["shapes[" + i + "].x1"] = xms;
                try { window.Plotly.relayout(inner, patch); } catch (e) {}
                break;
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("playback-dummy", "data"),
    Input("playhead-time", "data"),
    State("is-playing", "data"),
)


# Keep the window-aligned coverage strip's view range matched to the plot window.
@app.callback(
    Output("coverage-strip-window", "style"),
    Input("time-range-slider", "value"),
)
def update_window_coverage_view(slider_value):
    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        lo = int(min(slider_value))
        hi = int(max(slider_value))
    else:
        lo, hi = int(slider_min), int(slider_max)
    return {"--view-min": lo, "--view-max": hi}


# Align the window coverage strip to the main plot's data area (match Plotly's
# left/right margins) so video segments line up under the corresponding x range.
app.clientside_callback(
    """
    function(fig, sliderValue) {
        const align = document.getElementById("coverage-plot-align");
        const gd = document.getElementById("main-plot");
        if (!align || !gd) { return window.dash_clientside.no_update; }
        const inner = gd.querySelector(".js-plotly-plot");
        const sz = inner && inner._fullLayout && inner._fullLayout._size;
        if (sz) {
            align.style.paddingLeft = sz.l + "px";
            align.style.paddingRight = sz.r + "px";
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("coverage-plot-align", "title"),
    Input("main-plot", "figure"),
    Input("time-range-slider", "value"),
)


@app.callback(
    Output("mini-depth-plot", "figure"),
    Input("time-range-slider", "value"),
    Input("playhead-time", "data"),
    Input("is-playing", "data"),
)
def update_depth_context_plot(slider_value, playhead_value, is_playing):
    # Skip this figure rebuild during playback (keeps the transport snappy); it
    # redraws once when playback stops (is-playing flips to False triggers this).
    if is_playing and ctx.triggered_id == "playhead-time":
        raise dash.exceptions.PreventUpdate
    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        lo = int(min(slider_value))
        hi = int(max(slider_value))
    else:
        lo = int(slider_min)
        hi = int(slider_max)
    start_ts = _from_epoch_seconds(lo, tz_name)
    end_ts = _from_epoch_seconds(hi, tz_name)
    try:
        p_epoch = float(playhead_value)
    except Exception:
        p_epoch = float(_default_playhead_epoch([lo, hi]))
    p_epoch = max(float(lo), min(float(hi), p_epoch))
    return _build_depth_context_figure(start_ts, end_ts, playhead_ts=_from_epoch_seconds(p_epoch, tz_name))


@app.callback(
    Output("playhead-time", "data", allow_duplicate=True),
    Input("mini-depth-plot", "clickData"),
    State("playhead-slider", "min"),
    State("playhead-slider", "max"),
    prevent_initial_call=True,
)
def sync_playhead_from_depth_context(click_data, p_min, p_max):
    p_min = float(p_min if p_min is not None else slider_min)
    p_max = float(p_max if p_max is not None else slider_max)
    if not isinstance(click_data, dict):
        raise dash.exceptions.PreventUpdate
    points = click_data.get("points") or []
    if not points:
        raise dash.exceptions.PreventUpdate
    x_val = points[0].get("x")
    out = _coerce_epoch_from_relayout_value(x_val)
    if out is None:
        raise dash.exceptions.PreventUpdate
    return max(p_min, min(p_max, out))


@app.callback(
    Output("playhead-time", "data", allow_duplicate=True),
    Input("mini-depth-plot", "relayoutData"),
    State("playhead-slider", "min"),
    State("playhead-slider", "max"),
    State("mini-depth-plot", "figure"),
    prevent_initial_call=True,
)
def sync_playhead_from_depth_context_drag(relayout_data, p_min, p_max, existing_fig):
    if not isinstance(relayout_data, dict) or not relayout_data:
        raise dash.exceptions.PreventUpdate
    p_min = float(p_min if p_min is not None else slider_min)
    p_max = float(p_max if p_max is not None else slider_max)

    def _to_epoch(value):
        return _coerce_epoch_from_relayout_value(value)

    fig_shapes = ((existing_fig or {}).get("layout") or {}).get("shapes") or []
    candidates = []
    for k, v in relayout_data.items():
        m = re.match(r"^shapes\[(\d+)\]\.(x0|x1)$", str(k))
        if not m:
            continue
        idx = int(m.group(1))
        if idx < 0 or idx >= len(fig_shapes):
            continue
        shp = fig_shapes[idx] if isinstance(fig_shapes[idx], dict) else {}
        is_playhead = (
            shp.get("type") == "line"
            and shp.get("yref") == "paper"
            and str(((shp.get("line") or {}).get("color") or "")).upper() == "#FFD166"
        )
        if not is_playhead:
            continue
        epoch = _to_epoch(v)
        if epoch is not None:
            candidates.append(epoch)

    if not candidates:
        raise dash.exceptions.PreventUpdate
    return max(p_min, min(p_max, float(candidates[-1])))


def _model_depth_summary():
    try:
        depth_df, source_signal = _extract_depth_series_for_map_with_source(data_pkl, tz_name)
        if depth_df is None or depth_df.empty:
            return _vertical_axis_label_for_signal(source_signal), "Unavailable", source_signal
        vals = pd.to_numeric(depth_df["depth"], errors="coerce").dropna()
        if vals.empty:
            return _vertical_axis_label_for_signal(source_signal), "Unavailable", source_signal
        d_min = float(vals.min())
        d_max = float(vals.max())
        unit = _vertical_axis_value_unit_for_signal(source_signal)
        if str(source_signal or "").strip().lower() == "odba":
            return _vertical_axis_label_for_signal(source_signal), f"{d_min:.2f} - {d_max:.2f}{unit}", source_signal
        return _vertical_axis_label_for_signal(source_signal), f"{d_min:.0f} - {d_max:.0f}{unit}", source_signal
    except Exception:
        return "Depth", "Unavailable", None


def _build_model_metadata_card(
    animal_id,
    model_filename,
    offsets_text,
    depth_text,
    mode_text,
    sync_text,
    sync_class="stable",
    message=None,
    depth_label="Depth",
):
    cells = [
        ("Animal", animal_id or "Unknown"),
        ("File", model_filename or "Unavailable"),
        ("Offsets", offsets_text),
        (depth_label, depth_text),
        ("Mode", mode_text),
        ("Sync Status", sync_text),
    ]
    body = html.Div(
        [
            html.Div(
                [
                    html.Div(label, className="model-meta-label"),
                    html.Div(
                        value,
                        className=(
                            f"model-meta-value model-meta-status model-meta-status-{sync_class}"
                            if label == "Sync Status"
                            else "model-meta-value"
                        ),
                    ),
                ],
                className="model-meta-cell",
            )
            for label, value in cells
        ],
        className="model-meta-grid",
    )
    children = [
        html.Div(
            [
                html.Div("Model Metadata", className="model-meta-title"),
            ],
            className="model-meta-header",
        ),
        body,
    ]
    if message:
        if isinstance(message, str):
            children.append(html.Div(message, className="model-meta-note"))
        else:
            children.append(message)
    return html.Div(children, className="model-meta-card")


def _build_orientation_debug_note(orient_info):
    if not isinstance(orient_info, dict):
        return None
    message = str(orient_info.get("message") or "")
    debug = orient_info.get("debug") or {}
    if not isinstance(debug, dict):
        return html.Div(message, className="model-meta-note") if message else None

    lines = []
    if message:
        lines.append(html.Div(message))
    lines.append(html.Div("Orientation debug"))
    lines.append(html.Div(f"source: {debug.get('source', 'unknown')} | status: {debug.get('status', 'unknown')}"))

    grouped_signal = debug.get("grouped_signal")
    if grouped_signal:
        lines.append(html.Div(f"grouped signal: {grouped_signal}"))

    grouped_meta = debug.get("grouped_meta")
    if isinstance(grouped_meta, dict):
        cols = []
        for key in ("datetime_col", "pitch_col", "roll_col", "heading_col"):
            val = grouped_meta.get(key)
            if val:
                cols.append(f"{key.replace('_col', '')}={val}")
        if cols:
            lines.append(html.Div("grouped cols: " + ", ".join(cols)))

    split_meta = debug.get("split_meta")
    if isinstance(split_meta, dict):
        split_signals = []
        for key in ("pitch_signal", "roll_signal", "heading_signal"):
            val = split_meta.get(key)
            if val:
                split_signals.append(f"{key.replace('_signal', '')}={val}")
        if split_signals:
            lines.append(html.Div("split signals: " + ", ".join(split_signals)))

    rows = debug.get("rows")
    start = debug.get("start")
    end = debug.get("end")
    if rows is not None:
        lines.append(html.Div(f"rows: {rows}"))
    if start and end:
        lines.append(html.Div(f"time: {start} -> {end}"))

    if debug.get("heading_all_missing"):
        lines.append(html.Div("heading values missing -> using heading=0 fallback"))

    return html.Div(lines, className="model-meta-note")


@app.callback(
    Output("location-map", "figure"),
    Input("time-range-slider", "value"),
    Input("location-map-3d-toggle", "value"),
    Input("playhead-time", "data"),
    Input("is-playing", "data"),
)
def update_location_map_plot(slider_value, map_3d_toggle, playhead_value, is_playing):
    # The map redraw is heavy; skip it during playback and redraw on stop.
    if is_playing and ctx.triggered_id == "playhead-time":
        raise dash.exceptions.PreventUpdate
    return _build_location_map_figure(
        slider_value,
        map_3d_enabled=_is_map_3d_enabled(map_3d_toggle),
        playhead_value=playhead_value,
    )


if THREEJS_AVAILABLE:
    def _has_gps_track_for_orientation():
        signal_data = getattr(data_pkl, "signal_data", {}) or {}
        for name in ("location", "gps", "track"):
            df = signal_data.get(name)
            if (df is None or getattr(df, "empty", True)) and deployment_source is not None and name == "location":
                try:
                    df = deployment_source.load_location_track(max_points=MAP_DEFAULT_LOCATION_MAX_POINTS)
                except Exception:
                    df = None
            if df is None or getattr(df, "empty", True):
                continue
            if "datetime" not in df.columns:
                continue
            cols = {str(c).lower(): c for c in df.columns}
            lat_col = None
            lon_col = None
            for candidate in ("latitude", "lat", "gps_0"):
                lat_col = cols.get(candidate)
                if lat_col is not None:
                    break
            for candidate in ("longitude", "lon", "gps_1"):
                lon_col = cols.get(candidate)
                if lon_col is not None:
                    break
            if lat_col is None or lon_col is None:
                continue
            lat_vals = pd.to_numeric(df[lat_col], errors="coerce")
            lon_vals = pd.to_numeric(df[lon_col], errors="coerce")
            valid = lat_vals.notna() & lon_vals.notna()
            valid &= (lat_vals >= -90.0) & (lat_vals <= 90.0)
            valid &= (lon_vals >= -180.0) & (lon_vals <= 180.0)
            if bool(valid.any()):
                return True
        return False


    def _compute_depth_z_offset(playhead_value):
        try:
            playhead_epoch = float(playhead_value)
        except Exception:
            playhead_epoch = float(_default_playhead_epoch(slider_default))
        playhead_ts = _from_epoch_seconds(playhead_epoch, tz_name)
        depth_df, source_signal = _extract_depth_series_for_map_with_source(data_pkl, tz_name)
        axis_label = _vertical_axis_label_for_signal(source_signal)
        if depth_df is None or depth_df.empty:
            return 0.0, html.Div(f"{axis_label} debug: no data"), source_signal
        try:
            nearest_idx = (depth_df["datetime"] - playhead_ts).abs().idxmin()
            depth_now = float(depth_df.loc[nearest_idx, "depth"])
            vals = pd.to_numeric(depth_df["depth"], errors="coerce").dropna()
            if vals.empty:
                return 0.0, html.Div(f"{axis_label} debug: invalid values"), source_signal
            d05 = float(vals.quantile(0.05))
            d95 = float(vals.quantile(0.95))
            span = max(1e-9, d95 - d05)
            norm = max(0.0, min(1.0, (depth_now - d05) / span))
            # Map low values near the top grid plane and high values downward.
            top_offset = 23.0
            bottom_offset = -23.0
            depth_offset = top_offset - norm * (top_offset - bottom_offset)
            return (
                float(depth_offset),
                html.Div(
                    [
                        html.Div(f"{axis_label} now: {depth_now:.2f}"),
                        html.Div(f"{axis_label} range (Q05-Q95): {d05:.2f} to {d95:.2f}"),
                        html.Div(f"Normalized {axis_label.lower()} position: {norm:.3f}"),
                        html.Div(f"Applied Y offset: {depth_offset:.2f}"),
                    ]
                ),
                source_signal,
            )
        except Exception:
            return 0.0, html.Div(f"{axis_label} debug: lookup error"), source_signal


    @app.callback(
        Output("model-3d-status", "children"),
        Output("model-3d-viewer", "modelFile"),
        Output("model-3d-viewer", "textureFile"),
        Output("model-3d-viewer", "data"),
        Output("model-3d-viewer", "pitchOffset"),
        Output("model-3d-viewer", "rollOffset"),
        Output("model-3d-viewer", "headingOffset"),
        Output("model-3d-viewer", "pitchSign"),
        Output("model-3d-viewer", "rollSign"),
        Output("model-3d-viewer", "headingSign"),
        Output("model-3d-viewer", "rotationOrder"),
        Output("model-3d-viewer", "style"),
        Output("model-3d-viewer-inset", "modelFile"),
        Output("model-3d-viewer-inset", "textureFile"),
        Output("model-3d-viewer-inset", "data"),
        Output("model-3d-viewer-inset", "pitchOffset"),
        Output("model-3d-viewer-inset", "rollOffset"),
        Output("model-3d-viewer-inset", "headingOffset"),
        Output("model-3d-viewer-inset", "pitchSign"),
        Output("model-3d-viewer-inset", "rollSign"),
        Output("model-3d-viewer-inset", "headingSign"),
        Output("model-3d-viewer-inset", "rotationOrder"),
        Input("current-dataset", "data"),
        Input("current-deployment", "data"),
        Input("model-rot-x", "value"),
        Input("model-rot-y", "value"),
        Input("model-rot-z", "value"),
        Input("model-flip-pitch", "value"),
        Input("model-flip-roll", "value"),
        Input("model-flip-heading", "value"),
        Input("model-anim-pause-threshold", "value"),
        Input("model-track-line-width", "value"),
        Input("model-track-highlight-width", "value"),
        Input("model-track-highlight-offset", "value"),
        Input("model-3d-rotation-order-store", "data"),
    )
    def update_model_3d_panel(
        _dataset_value,
        _deployment_value,
        rot_x,
        rot_y,
        rot_z,
        flip_pitch,
        flip_roll,
        flip_heading,
        pause_threshold,
        track_line_width,
        highlight_width,
        highlight_offset,
        rotation_order,
    ):
        animal_id_hint = infer_animal_id(data_pkl, deployment_id_fallback=deployment_id)
        model_future = _start_model_info_fetch(animal_id_hint)
        resolved_data_pkl = _build_orientation_runtime_payload()
        animal_id = infer_animal_id(resolved_data_pkl, deployment_id_fallback=deployment_id)
        orient_info = build_orientation_data_json(resolved_data_pkl)
        if animal_id and animal_id != animal_id_hint:
            model_info = _cached_fetch_3d_model_info(animal_id)
        else:
            model_info = _resolve_model_info_from_future(model_future, animal_id)
        if not (isinstance(model_info, dict) and model_info.get("ok")):
            print(f"[model3d] fetch not ok for animal={animal_id!r} hint={animal_id_hint!r}: "
                  f"{(model_info or {}).get('message')!r}")
        orientation_json = orient_info.get("data_json") or EMPTY_ORIENTATION_JSON
        x_off = _coerce_float_default(rot_x, 0.0)
        y_off = _coerce_float_default(rot_y, 0.0)
        z_off = _coerce_float_default(rot_z, 0.0)
        pitch_sign = -1 if "on" in (flip_pitch or []) else 1
        roll_sign = -1 if "on" in (flip_roll or []) else 1
        heading_sign = -1 if "on" in (flip_heading or []) else 1
        pause_threshold = _coerce_float_default(pause_threshold, 10.0)
        track_line_width = min(max(_coerce_float_default(track_line_width, 7.1), 1.0), 20.0)
        highlight_width = min(max(_coerce_float_default(highlight_width, 1.55), 0.1), 12.0)
        highlight_offset = min(max(_coerce_float_default(highlight_offset, 0.18), 0.0), 1.5)
        rotation_order = _normalize_rotation_order(rotation_order)
        depth_label, depth_summary, depth_source_signal = _model_depth_summary()
        track_enabled = _is_depth_like_signal(depth_source_signal) and depth_summary != "Unavailable"

        main_viewer_style = {
            "width": "100%",
            "height": "100%",
            "--pause-stroke-threshold": str(float(pause_threshold)),
            "--show-trajectory": ("1" if track_enabled else "0"),
            "--trajectory-line-width": str(float(track_line_width)),
            "--trajectory-highlight-width": str(float(highlight_width)),
            "--trajectory-highlight-offset": str(float(highlight_offset)),
        }

        model = model_info.get("model") or {}
        model_url = str(model.get("model_url") or "")
        texture_url = str(model.get("texture_url") or "")
        model_filename = str(model.get("model_filename") or "unknown")
        offsets_text = f"[{x_off:.1f}, {y_off:.1f}, {z_off:.1f}]"

        orientation_note = _build_orientation_debug_note(orient_info)
        if not track_enabled:
            disabled_note = html.Div(
                "Track disabled: requires depth/pressure vertical channel (ODBA stays context-only).",
                className="model-meta-note",
            )
            orientation_note = html.Div([orientation_note, disabled_note]) if orientation_note else disabled_note
        if model_info.get("ok"):
            status_children = _build_model_metadata_card(
                animal_id=animal_id,
                model_filename=model_filename,
                offsets_text=offsets_text,
                depth_text=depth_summary,
                mode_text=("Track-linked" if track_enabled else "Orientation-only"),
                sync_text="Stable",
                sync_class="stable",
                message=orientation_note,
                depth_label=depth_label,
            )
            return (
                status_children,
                model_url,
                texture_url,
                orientation_json,
                float(x_off),
                float(y_off),
                float(z_off),
                int(pitch_sign),
                int(roll_sign),
                int(heading_sign),
                rotation_order,
                main_viewer_style,
                model_url,
                texture_url,
                orientation_json,
                float(x_off),
                float(y_off),
                float(z_off),
                int(pitch_sign),
                int(roll_sign),
                int(heading_sign),
                rotation_order,
            )

        status_children = _build_model_metadata_card(
            animal_id=(animal_id or "Unknown"),
            model_filename=model_filename,
            offsets_text=offsets_text,
            depth_text=depth_summary,
            mode_text="Fallback",
            sync_text="Unavailable",
            sync_class="warning",
            message=(orientation_note or str(model_info.get("message") or "Unknown issue.")),
            depth_label=depth_label,
        )
        return (
            status_children,
            "",
            "",
            orientation_json,
            float(x_off),
            float(y_off),
            float(z_off),
            int(pitch_sign),
            int(roll_sign),
            int(heading_sign),
            rotation_order,
            main_viewer_style,
            "",
            "",
            orientation_json,
            float(x_off),
            float(y_off),
            float(z_off),
            int(pitch_sign),
            int(roll_sign),
            int(heading_sign),
            rotation_order,
        )

    if WATER_VIEWER_AVAILABLE:
        @app.callback(
            Output("water-viewer-status", "children"),
            Output("water-viewer", "modelFile"),
            Output("water-viewer", "textureFile"),
            Output("water-viewer", "data"),
            Output("water-viewer", "pitchOffset"),
            Output("water-viewer", "rollOffset"),
            Output("water-viewer", "headingOffset"),
            Output("water-viewer", "pitchSign"),
            Output("water-viewer", "rollSign"),
            Output("water-viewer", "headingSign"),
            Output("water-viewer", "rotationOrder"),
            Output("water-viewer", "showTrajectory"),
            Output("water-viewer", "trackLineWidth"),
            Output("water-viewer", "highlightWidth"),
            Output("water-viewer", "highlightOffset"),
            Input("current-dataset", "data"),
            Input("current-deployment", "data"),
            Input("model-rot-x", "value"),
            Input("model-rot-y", "value"),
            Input("model-rot-z", "value"),
            Input("model-flip-pitch", "value"),
            Input("model-flip-roll", "value"),
            Input("model-flip-heading", "value"),
            Input("model-track-line-width", "value"),
            Input("model-track-highlight-width", "value"),
            Input("model-track-highlight-offset", "value"),
            Input("model-3d-rotation-order-store", "data"),
        )
        def update_water_viewer_panel(
            _dataset_value,
            _deployment_value,
            rot_x,
            rot_y,
            rot_z,
            flip_pitch,
            flip_roll,
            flip_heading,
            track_line_width,
            highlight_width,
            highlight_offset,
            rotation_order,
        ):
            animal_id_hint = infer_animal_id(data_pkl, deployment_id_fallback=deployment_id)
            model_future = _start_model_info_fetch(animal_id_hint)
            resolved_data_pkl = _build_orientation_runtime_payload()
            animal_id = infer_animal_id(resolved_data_pkl, deployment_id_fallback=deployment_id)
            orient_info = build_orientation_data_json(resolved_data_pkl)
            if animal_id and animal_id != animal_id_hint:
                model_info = _cached_fetch_3d_model_info(animal_id)
            else:
                model_info = _resolve_model_info_from_future(model_future, animal_id)
            orientation_json = orient_info.get("data_json") or EMPTY_ORIENTATION_JSON
            x_off = _coerce_float_default(rot_x, 0.0)
            y_off = _coerce_float_default(rot_y, 0.0)
            z_off = _coerce_float_default(rot_z, 0.0)
            pitch_sign = -1 if "on" in (flip_pitch or []) else 1
            roll_sign = -1 if "on" in (flip_roll or []) else 1
            heading_sign = -1 if "on" in (flip_heading or []) else 1
            track_line_width = min(max(_coerce_float_default(track_line_width, 7.1), 1.0), 20.0)
            highlight_width = min(max(_coerce_float_default(highlight_width, 1.55), 0.1), 12.0)
            highlight_offset = min(max(_coerce_float_default(highlight_offset, 0.18), 0.0), 1.5)
            rotation_order = _normalize_rotation_order(rotation_order)

            _FALLBACK_MODEL_URL = "/assets/6_killerWhale_v017_LP.obj"
            model = model_info.get("model") or {}
            model_url = str(model.get("model_url") or "") or _FALLBACK_MODEL_URL
            texture_url = str(model.get("texture_url") or "")
            model_filename = str(model.get("model_filename") or "") or "6_killerWhale_v017_LP.obj"
            depth_label, depth_summary, depth_source_signal = _model_depth_summary()
            track_enabled = _is_depth_like_signal(depth_source_signal) and depth_summary != "Unavailable"
            offsets_text = f"[{x_off:.1f}, {y_off:.1f}, {z_off:.1f}]"
            orientation_note = _build_orientation_debug_note(orient_info)

            if model_info.get("ok"):
                status_children = _build_model_metadata_card(
                    animal_id=animal_id,
                    model_filename=model_filename,
                    offsets_text=offsets_text,
                    depth_text=depth_summary,
                    mode_text="Water tank",
                    sync_text="Experimental",
                    sync_class="stable",
                    message=(orientation_note or "Adapted from Evan Wallace's WebGL Water demo."),
                    depth_label=depth_label,
                )
                return (
                    status_children,
                    model_url,
                    texture_url,
                    orientation_json,
                    float(x_off),
                    float(y_off),
                    float(z_off),
                    int(pitch_sign),
                    int(roll_sign),
                    int(heading_sign),
                    rotation_order,
                    bool(track_enabled),
                    float(track_line_width),
                    float(highlight_width),
                    float(highlight_offset),
                )

            status_children = _build_model_metadata_card(
                animal_id=(animal_id or "Unknown"),
                model_filename=model_filename,
                offsets_text=offsets_text,
                depth_text=depth_summary,
                mode_text="Water tank",
                sync_text="Local fallback",
                sync_class="warning",
                message=(orientation_note or str(model_info.get("message") or "Using built-in whale model.")),
                depth_label=depth_label,
            )
            return (
                status_children,
                model_url,
                texture_url,
                orientation_json,
                float(x_off),
                float(y_off),
                float(z_off),
                int(pitch_sign),
                int(roll_sign),
                int(heading_sign),
                rotation_order,
                bool(track_enabled),
                float(track_line_width),
                float(highlight_width),
                float(highlight_offset),
            )


    @app.callback(
        Output("model-3d-viewer", "activeTime"),
        Output("model-3d-viewer-inset", "activeTime"),
        Input("playhead-time", "data"),
    )
    def update_model_3d_active_time(playhead_value):
        try:
            ts_ms = float(playhead_value) * 1000.0
            return ts_ms, ts_ms
        except Exception:
            ts_ms = float(_default_playhead_epoch(slider_default)) * 1000.0
            return ts_ms, ts_ms

    if WATER_VIEWER_AVAILABLE:
        @app.callback(
            Output("water-viewer", "activeTime"),
            Input("playhead-time", "data"),
        )
        def update_water_viewer_active_time(playhead_value):
            try:
                return float(playhead_value) * 1000.0
            except Exception:
                return float(_default_playhead_epoch(slider_default)) * 1000.0


    @app.callback(
        Output("model-3d-track-status", "children"),
        Input("playhead-time", "data"),
        Input("current-dataset", "data"),
        Input("current-deployment", "data"),
        Input("is-playing", "data"),
    )
    def update_model_3d_track_status(playhead_value, _dataset_value, _deployment_value, is_playing):
        # Skip the per-tick depth recompute during playback; refresh on stop.
        if is_playing and ctx.triggered_id == "playhead-time":
            raise dash.exceptions.PreventUpdate
        _depth_offset, status, source_signal = _compute_depth_z_offset(playhead_value)
        if not _is_depth_like_signal(source_signal):
            return html.Div([status, html.Div("Track is off until a depth/pressure channel is available.", style={"opacity": 0.85})])
        if not _has_gps_track_for_orientation():
            caveat = html.Div(
                "Pseudotrack with 1 m/s forward movement when stride/stroke detected.",
                style={"opacity": 0.85},
            )
            return html.Div([status, caveat])
        return status

    if WATER_VIEWER_AVAILABLE:
        @app.callback(
            Output("water-viewer", "depthOffset"),
            Input("playhead-time", "data"),
            Input("current-dataset", "data"),
            Input("current-deployment", "data"),
        )
        def update_water_viewer_depth_offset(playhead_value, _dataset_value, _deployment_value):
            depth_offset, _status, _source_signal = _compute_depth_z_offset(playhead_value)
            return depth_offset
else:
    @app.callback(
        Output("model-3d-status", "children"),
        Input("current-dataset", "data"),
        Input("current-deployment", "data"),
        Input("model-rot-x", "value"),
        Input("model-rot-y", "value"),
        Input("model-rot-z", "value"),
        Input("model-flip-pitch", "value"),
        Input("model-flip-roll", "value"),
        Input("model-flip-heading", "value"),
    )
    def update_model_3d_panel(_dataset_value, _deployment_value, rot_x, rot_y, rot_z, flip_pitch, flip_roll, flip_heading):
        animal_id = infer_animal_id(data_pkl, deployment_id_fallback=deployment_id)
        model_info = _cached_fetch_3d_model_info(animal_id)
        x_off = _coerce_float_default(rot_x, 0.0)
        y_off = _coerce_float_default(rot_y, 0.0)
        z_off = _coerce_float_default(rot_z, 0.0)
        pitch_sign = -1 if "on" in (flip_pitch or []) else 1
        roll_sign = -1 if "on" in (flip_roll or []) else 1
        heading_sign = -1 if "on" in (flip_heading or []) else 1
        return [
            html.B("3D viewer component unavailable"),
            html.Div(f"Animal: {animal_id or 'unknown'}"),
            html.Div(f"Rotation offsets (deg): X={x_off:.1f}, Y={y_off:.1f}, Z={z_off:.1f}"),
            html.Div(f"Sign flips: pitch={pitch_sign:+d}, roll={roll_sign:+d}, heading={heading_sign:+d}"),
            html.Div(str(model_info.get("message") or "Model metadata lookup complete.")),
            html.Div("Install DiveDB three_js_orientation to render the model.", style={"opacity": 0.8}),
        ]


@app.callback(
    Output("model-3d-save-status", "children"),
    Input("save-model-3d-defaults", "n_clicks"),
    State("model-rot-x", "value"),
    State("model-rot-y", "value"),
    State("model-rot-z", "value"),
    State("model-flip-pitch", "value"),
    State("model-flip-roll", "value"),
    State("model-flip-heading", "value"),
    State("model-track-line-width", "value"),
    State("model-track-highlight-width", "value"),
    State("model-track-highlight-offset", "value"),
    State("model-3d-rotation-order-store", "data"),
    prevent_initial_call=True,
)
def save_model_3d_defaults(
    _n_clicks,
    rot_x,
    rot_y,
    rot_z,
    flip_pitch,
    flip_roll,
    flip_heading,
    track_line_width,
    highlight_width,
    highlight_offset,
    rotation_order,
):
    cfg = {
        "pitch_offset": _coerce_float_default(rot_x, 0.0),
        "roll_offset": _coerce_float_default(rot_y, 0.0),
        "heading_offset": _coerce_float_default(rot_z, 0.0),
        "pitch_sign": (-1 if "on" in (flip_pitch or []) else 1),
        "roll_sign": (-1 if "on" in (flip_roll or []) else 1),
        "heading_sign": (-1 if "on" in (flip_heading or []) else 1),
        "track_line_width": _coerce_float_default(track_line_width, 7.1),
        "highlight_width": _coerce_float_default(highlight_width, 1.55),
        "highlight_offset": _coerce_float_default(highlight_offset, 0.18),
        "rotation_order": _normalize_rotation_order(rotation_order),
    }
    ok = _persist_model_3d_controls_dataset_defaults(param_manager, cfg)
    if ok:
        return "Saved 3D orientation controls to dataset-global settings."
    return "Could not save 3D orientation controls."


@app.callback(
    Output("model-rot-x", "value", allow_duplicate=True),
    Output("model-rot-y", "value", allow_duplicate=True),
    Output("model-rot-z", "value", allow_duplicate=True),
    Output("model-flip-pitch", "value", allow_duplicate=True),
    Output("model-flip-roll", "value", allow_duplicate=True),
    Output("model-flip-heading", "value", allow_duplicate=True),
    Output("model-track-line-width", "value", allow_duplicate=True),
    Output("model-track-highlight-width", "value", allow_duplicate=True),
    Output("model-track-highlight-offset", "value", allow_duplicate=True),
    Output("model-3d-rotation-order-store", "data", allow_duplicate=True),
    Output("model-3d-save-status", "children", allow_duplicate=True),
    Input("current-dataset", "data"),
    Input("current-deployment", "data"),
    prevent_initial_call=True,
)
def refresh_model_3d_controls_from_config(_dataset_value, _deployment_value):
    cfg = _load_model_3d_controls_from_params(param_manager)
    return (
        float(cfg.get("pitch_offset", 0.0)),
        float(cfg.get("roll_offset", 0.0)),
        float(cfg.get("heading_offset", 0.0)),
        (["on"] if int(cfg.get("pitch_sign", 1)) < 0 else []),
        (["on"] if int(cfg.get("roll_sign", 1)) < 0 else []),
        (["on"] if int(cfg.get("heading_sign", 1)) < 0 else []),
        float(cfg.get("track_line_width", 7.1)),
        float(cfg.get("highlight_width", 1.55)),
        float(cfg.get("highlight_offset", 0.18)),
        _normalize_rotation_order(cfg.get("rotation_order")),
        "",
    )


def _format_duration(seconds):
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


@app.callback(
    Output("abs-start-label", "children"),
    Output("abs-end-label", "children"),
    Output("window-summary", "children"),
    Output("window-summary-inline", "children"),
    Input("time-range-slider", "value"),
    Input("time-display-tz-toggle", "value"),
    State("time-range-slider", "min"),
    State("time-range-slider", "max"),
)
def update_slider_labels(slider_value, display_mode, s_min, s_max):
    if not isinstance(slider_value, (list, tuple)) or len(slider_value) != 2:
        lo = int(s_min or slider_min)
        hi = int(s_max or slider_max)
    else:
        lo = int(min(slider_value))
        hi = int(max(slider_value))

    s_min = int(s_min if s_min is not None else slider_min)
    s_max = int(s_max if s_max is not None else slider_max)

    display_tz = _display_tz_name(display_mode)
    lo_ts = _from_epoch_seconds(lo, display_tz)
    hi_ts = _from_epoch_seconds(hi, display_tz)
    abs_lo_ts = _from_epoch_seconds(s_min, display_tz)
    abs_hi_ts = _from_epoch_seconds(s_max, display_tz)

    duration = f"⏱ {_format_duration(hi - lo)}"
    inline = f"{lo_ts.strftime('%H:%M:%S')} → {hi_ts.strftime('%H:%M:%S')}  ·  {lo_ts.strftime('%Y-%m-%d')} ({display_tz})"

    return (
        _slider_label_children(abs_lo_ts, display_tz),
        _slider_label_children(abs_hi_ts, display_tz),
        duration,
        inline,
    )


def _render_event_target_blocks(event_keys, order, channels_store, channel_order_store, event_styles, event_targets):
    event_keys = event_keys or []
    event_styles = event_styles or {}
    blocks = []
    for idx, k in enumerate(event_keys):
        style_defaults = _default_event_style(k, idx)
        style_saved = dict(event_styles.get(k) or {})
        color_value = _normalize_hex_color(style_saved.get("color")) or style_defaults["color"]
        symbol_value = str(style_saved.get("symbol") or style_defaults["symbol"])
        if symbol_value not in EVENT_SYMBOL_OPTIONS:
            symbol_value = "circle"
        blocks.append(
            html.Div(
                [
                    html.B(f"{k} style"),
                    html.Small("Channel targets are configured in the event popup.", className="event-style-label"),
                    html.Div(
                        [
                            html.Div("Style", className="event-style-label"),
                            dcc.Input(
                                id={"type": "event-color", "event": k},
                                type="text",
                                value=color_value,
                                debounce=True,
                                className="event-color-input",
                                placeholder="#RRGGBB",
                            ),
                            html.Button(
                                "✎",
                                id={"type": "event-color-btn", "event": k},
                                className="chip-color-edit",
                                n_clicks=0,
                                title="Edit event color",
                            ),
                            dcc.Dropdown(
                                id={"type": "event-symbol", "event": k},
                                options=EVENT_SYMBOL_DROPDOWN_OPTIONS,
                                value=symbol_value,
                                clearable=False,
                                className="event-symbol-select",
                            ),
                            dcc.Checklist(
                                id={"type": "event-shade-enabled", "event": k},
                                options=[{"label": "Shade duration", "value": "on"}],
                                value=["on"] if bool(style_saved.get("shade_enabled", True)) else [],
                                className="event-shade-toggle",
                            ),
                            dcc.Input(
                                id={"type": "event-shade-color", "event": k},
                                type="text",
                                value=_normalize_hex_color(style_saved.get("shade_color")) or color_value,
                                debounce=True,
                                className="event-color-input",
                                placeholder="#RRGGBB",
                            ),
                            html.Button(
                                "✎",
                                id={"type": "event-shade-color-btn", "event": k},
                                className="chip-color-edit",
                                n_clicks=0,
                                title="Edit shade color",
                            ),
                        ],
                        className="event-style-row",
                    ),
                ],
                style={"marginTop": "8px"},
            )
        )
    return blocks


@app.callback(
    Output("events-target-editor", "children"),
    Output("events-summary-chips", "children"),
    Output("events-key-chips", "children"),
    Input("events-select", "value"),
    Input("event-style-store", "data"),
)
def render_events_editor(event_keys, event_styles):
    event_keys = [str(v) for v in (event_keys or []) if str(v).strip()]
    label_fn = lambda ev: _event_chip_label(ev, event_styles, event_keys.index(ev) if ev in event_keys else 0)
    return (
        [html.Small("Use the pencil on an event chip to edit marker/shading colors.")],
        _render_summary_chips(
            event_keys,
            color_fn=lambda ev: _event_chip_style(event_styles, ev, event_keys.index(ev) if ev in event_keys else 0),
            label_fn=label_fn,
        ),
        _render_event_key_chips(event_keys, "events", event_styles),
    )


@app.callback(
    Output("point-event-target-editor", "children"),
    Output("point-events-summary-chips", "children"),
    Output("point-event-key-chips", "children"),
    Input("point-events-select", "value"),
)
def render_point_events_editor(event_keys):
    return [html.Small("Legacy point events panel hidden.")], "", ""


@app.callback(
    Output("state-event-target-editor", "children"),
    Output("state-events-summary-chips", "children"),
    Output("state-event-key-chips", "children"),
    Input("state-events-select", "value"),
)
def render_state_events_editor(event_keys):
    return [html.Small("Legacy state events panel hidden.")], "", ""


@app.callback(
    Output("event-targets-store", "data"),
    Input({"type": "event-targets", "event": ALL}, "value"),
    State({"type": "event-targets", "event": ALL}, "id"),
    State("event-targets-store", "data"),
)
def sync_event_targets(values, ids, current):
    if not ids:
        raise dash.exceptions.PreventUpdate
    out = dict(current or {})
    for v, i in zip(values or [], ids or []):
        ev = i.get("event")
        if not ev:
            continue
        out[ev] = v or []
    _save_event_targets(color_mapping_path, out)
    return out


@app.callback(
    Output("event-style-store", "data"),
    Input({"type": "event-color", "event": ALL}, "value"),
    Input({"type": "event-symbol", "event": ALL}, "value"),
    Input({"type": "event-shade-enabled", "event": ALL}, "value"),
    Input({"type": "event-shade-color", "event": ALL}, "value"),
    State({"type": "event-color", "event": ALL}, "id"),
    State({"type": "event-symbol", "event": ALL}, "id"),
    State({"type": "event-shade-enabled", "event": ALL}, "id"),
    State({"type": "event-shade-color", "event": ALL}, "id"),
    State("event-style-store", "data"),
)
def sync_event_styles(color_values, symbol_values, shade_enabled_values, shade_color_values, color_ids, symbol_ids, shade_enabled_ids, shade_color_ids, current):
    out = dict(current or {})

    for val, cid in zip(color_values or [], color_ids or []):
        ev = (cid or {}).get("event")
        if not ev:
            continue
        entry = dict(out.get(ev) or {})
        normalized = _normalize_hex_color(val)
        if normalized:
            entry["color"] = normalized
        out[ev] = entry

    for val, sid in zip(symbol_values or [], symbol_ids or []):
        ev = (sid or {}).get("event")
        if not ev:
            continue
        entry = dict(out.get(ev) or {})
        symbol = str(val or "")
        if symbol in EVENT_SYMBOL_OPTIONS:
            entry["symbol"] = symbol
        out[ev] = entry

    for val, sid in zip(shade_enabled_values or [], shade_enabled_ids or []):
        ev = (sid or {}).get("event")
        if not ev:
            continue
        entry = dict(out.get(ev) or {})
        entry["shade_enabled"] = "on" in (val or [])
        out[ev] = entry

    for val, sid in zip(shade_color_values or [], shade_color_ids or []):
        ev = (sid or {}).get("event")
        if not ev:
            continue
        entry = dict(out.get(ev) or {})
        normalized = _normalize_hex_color(val)
        if normalized:
            entry["shade_color"] = normalized
        out[ev] = entry

    _save_event_styles(color_mapping_path, out)
    return out


@app.callback(
    Output("ordered-signals-store", "data", allow_duplicate=True),
    Output("channel-order-store", "data", allow_duplicate=True),
    Output("event-targets-store", "data", allow_duplicate=True),
    Input("chip-order-updates", "value"),
    State("ordered-signals-store", "data"),
    State("channel-order-store", "data"),
    State("event-targets-store", "data"),
    State("signals-select", "value"),
    State("channels-store", "data"),
    prevent_initial_call=True,
)
def apply_chip_orders(chip_order_json, ordered_signals, channel_order_store, event_targets_store, selected_signals, channels_store):
    try:
        payload = json.loads(chip_order_json or "{}")
    except Exception:
        raise dash.exceptions.PreventUpdate

    ordered_signals = [str(s) for s in (ordered_signals or []) if str(s).strip()]
    selected_signals = [str(s) for s in (selected_signals or []) if str(s).strip()]
    channel_order_store = dict(channel_order_store or {})
    event_targets_store = dict(event_targets_store or {})
    channels_store = channels_store or {}

    new_signals = ordered_signals
    new_channels = channel_order_store
    new_event_targets = event_targets_store

    sig_payload = ((payload.get("signals") or {}).get("__all__") or [])
    if sig_payload:
        # Reconcile against both stores so transient drag payload glitches cannot drop signals.
        signal_base = ordered_signals[:] + [s for s in selected_signals if s not in set(ordered_signals)]
        if not signal_base:
            signal_base = [str(s) for s in sig_payload if str(s).strip()]
        seen = set()
        filtered = []
        for s in sig_payload:
            if s in signal_base and s not in seen:
                filtered.append(s)
                seen.add(s)
        sig_payload = filtered
        remaining = [s for s in signal_base if s not in sig_payload]
        new_signals = sig_payload + remaining

    ch_payload = payload.get("channels") or {}
    if isinstance(ch_payload, dict):
        for sig, vals in ch_payload.items():
            sig = str(sig)
            available = _signal_channels(data_pkl, sig)
            if not available:
                continue
            chosen = [c for c in (channels_store.get(sig) or []) if c in available]
            selected = chosen or available
            base = _normalize_channel_order(selected, channel_order_store.get(sig))

            seen_vals = set()
            ordered_vals = []
            for v in (vals or []):
                v = str(v)
                if v in base and v not in seen_vals:
                    ordered_vals.append(v)
                    seen_vals.add(v)
            if ordered_vals:
                rem = [v for v in base if v not in seen_vals]
                new_channels[sig] = ordered_vals + rem

    ev_payload = payload.get("event_targets") or {}
    if isinstance(ev_payload, dict):
        for ev, vals in ev_payload.items():
            existing = event_targets_store.get(ev) or []
            vals = [v for v in (vals or []) if v in existing]
            if vals:
                rem = [v for v in existing if v not in vals]
                new_event_targets[ev] = vals + rem

    return new_signals, new_channels, new_event_targets


@app.callback(
    Output("main-plot", "figure"),
    Input("ordered-signals-store", "data"),
    Input("channels-store", "data"),
    Input("channel-order-store", "data"),
    Input("signal-axis-config-store", "data"),
    Input("events-select", "value"),
    Input("event-targets-store", "data"),
    Input("event-style-store", "data"),
    Input("time-range-slider", "value"),
    Input("target-sampling-rate-hz", "value"),
    Input("peak-detect-preview-store", "data"),
    Input("color-map-version", "data"),
    State("playhead-time", "data"),
    State("main-plot", "relayoutData"),
)
def update_plot(
    ordered_signals,
    channels_store,
    channel_order_store,
    signal_axis_config_store,
    event_keys,
    event_targets,
    event_styles,
    slider_value,
    target_sampling_rate_hz,
    peak_preview_store,
    _color_map_version,
    playhead_time,
    main_relayout_data,
):
    global _current_fig
    available_signals = set(_available_signal_names(data_pkl))
    allowed_signals = set(_plot_signal_allowlist(data_pkl))
    ordered_signals = [s for s in (ordered_signals or []) if s in available_signals]
    if not ordered_signals:
        fallback = [s for s in default_signals[:] if s in allowed_signals]
        if not fallback:
            fallback = _preferred_plot_signals(_plot_signal_allowlist(data_pkl))
        if not fallback:
            fallback = _preferred_plot_signals(sorted(available_signals))
        ordered_signals = fallback[:]

    channels_store = channels_store or {}
    channel_order_store = channel_order_store or {}
    signal_axis_config_store = signal_axis_config_store or {}
    plot_channels = {}
    for s in ordered_signals:
        options = _signal_channels(data_pkl, s)
        chosen = [c for c in (channels_store.get(s) or []) if c in options]
        selected = chosen or options
        plot_channels[s] = _normalize_channel_order(selected, channel_order_store.get(s))

    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        start_ts = _from_epoch_seconds(min(slider_value), tz_name)
        end_ts = _from_epoch_seconds(max(slider_value), tz_name)
    else:
        start_ts = pd.Timestamp(start_default)
        end_ts = pd.Timestamp(end_default)

    start_ts = min(max(pd.Timestamp(start_ts), global_start), global_end)
    end_ts = min(max(pd.Timestamp(end_ts), global_start), global_end)
    if end_ts <= start_ts:
        end_ts = min(start_ts + timedelta(minutes=2), global_end)

    # If slider update came from plot zoom, skip full redraw so the
    # FigureResampler relayout patch remains authoritative.
    # First use relayout payload (robust to callback ordering), then fallback
    # to current figure range comparison.
    if ctx.triggered_id == "time-range-slider" and _current_fig is not None:
        try:
            rx0, rx1 = _extract_primary_xaxis_range(main_relayout_data or {})
            if rx0 is not None and rx1 is not None:
                r_lo_epoch = _coerce_epoch_from_relayout_value(rx0)
                r_hi_epoch = _coerce_epoch_from_relayout_value(rx1)
                if r_lo_epoch is not None and r_hi_epoch is not None:
                    r_lo = _from_epoch_seconds(float(min(r_lo_epoch, r_hi_epoch)), tz_name)
                    r_hi = _from_epoch_seconds(float(max(r_lo_epoch, r_hi_epoch)), tz_name)
                    if abs((r_lo - start_ts).total_seconds()) <= 1.0 and abs((r_hi - end_ts).total_seconds()) <= 1.0:
                        raise dash.exceptions.PreventUpdate
        except dash.exceptions.PreventUpdate:
            raise
        except Exception:
            pass
        try:
            cur_range = getattr(getattr(_current_fig.layout, "xaxis", None), "range", None)
            if cur_range and len(cur_range) == 2:
                cur_lo = pd.Timestamp(cur_range[0]).tz_convert(tz_name) if pd.Timestamp(cur_range[0]).tzinfo else pd.Timestamp(cur_range[0]).tz_localize(tz_name)
                cur_hi = pd.Timestamp(cur_range[1]).tz_convert(tz_name) if pd.Timestamp(cur_range[1]).tzinfo else pd.Timestamp(cur_range[1]).tz_localize(tz_name)
                if abs((cur_lo - start_ts).total_seconds()) <= 1.0 and abs((cur_hi - end_ts).total_seconds()) <= 1.0:
                    raise dash.exceptions.PreventUpdate
        except dash.exceptions.PreventUpdate:
            raise
        except Exception:
            pass

    notes = {}
    state_notes = {}
    event_styles = event_styles or {}

    try:
        target_sampling_rate = float(target_sampling_rate_hz)
    except Exception:
        target_sampling_rate = 10.0
    if target_sampling_rate <= 0:
        target_sampling_rate = 10.0

    data_for_plot = data_pkl
    if deployment_source is not None:
        try:
            data_for_plot = deployment_source.load_plot_payload(
                ordered_signals,
                start_ts,
                end_ts,
                include_events=True,
            )
        except Exception:
            data_for_plot = data_pkl
    if deployment_source is not None and not _payload_has_signal_rows(data_for_plot, ordered_signals):
        fallback_signals = [
            s for s in _preferred_plot_signals(_plot_signal_allowlist(data_pkl))
            if s in available_signals
        ]
        if not fallback_signals:
            fallback_signals = _preferred_plot_signals(sorted(available_signals))
        if fallback_signals:
            ordered_signals = fallback_signals[:]
            plot_channels = {}
            for s in ordered_signals:
                options = _signal_channels(data_pkl, s)
                chosen = [c for c in (channels_store.get(s) or []) if c in options]
                selected = chosen or options
                plot_channels[s] = _normalize_channel_order(selected, channel_order_store.get(s))
            try:
                data_for_plot = deployment_source.load_plot_payload(
                    ordered_signals,
                    start_ts,
                    end_ts,
                    include_events=True,
                )
            except Exception:
                data_for_plot = data_pkl
    try:
        runtime_signal_data = getattr(data_pkl, "signal_data", {}) or {}
        for sig in ordered_signals:
            runtime_df = runtime_signal_data.get(sig)
            if _is_runtime_signal_frame(runtime_df):
                dt = pd.to_datetime(runtime_df["datetime"], errors="coerce", utc=True).dt.tz_convert(tz_name)
                keep = dt.notna() & (dt >= start_ts) & (dt <= end_ts)
                trimmed = runtime_df.loc[keep].copy()
                trimmed["datetime"] = dt.loc[keep].values
                data_for_plot.signal_data[sig] = trimmed.reset_index(drop=True)
                data_for_plot.signal_info[sig] = dict((getattr(data_pkl, "signal_info", {}) or {}).get(sig, {}) or {})
    except Exception:
        pass
    try:
        downsampled_signal_data = _build_downsampled_signal_data_for_plot(
            data_for_plot,
            ordered_signals,
            target_sampling_rate,
            tz_name,
        )
        data_for_plot = copy.copy(data_for_plot)
        data_for_plot.signal_data = downsampled_signal_data
    except Exception:
        pass

    # Append preview-detected peaks to plotting event_data only (until saved).
    all_events = sorted([str(e) for e in (event_keys or []) if str(e).strip()], key=lambda x: x.lower())
    preview_rows = list((peak_preview_store or {}).get("events") or [])
    if preview_rows:
        try:
            p_df = pd.DataFrame(preview_rows)
            if not p_df.empty and "datetime" in p_df.columns and "key" in p_df.columns:
                p_df["datetime"] = pd.to_datetime(p_df["datetime"], errors="coerce")
                p_df = p_df.dropna(subset=["datetime", "key"])
                if not p_df.empty:
                    tmp = copy.copy(data_for_plot)
                    base_events = getattr(data_for_plot, "event_data", None)
                    if not isinstance(base_events, pd.DataFrame):
                        base_events = pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value"])
                    tmp.event_data = pd.concat([base_events, p_df], ignore_index=True)
                    data_for_plot = tmp
                    p_keys = [str(k) for k in p_df["key"].dropna().unique()]
                    all_events = sorted(list(set(all_events + p_keys)), key=lambda x: x.lower())
        except Exception:
            pass

    preview_state_keys = []
    try:
        data_for_plot, preview_state_keys = _append_cluster_preview_events_to_payload(
            data_for_plot,
            all_events,
            start_ts,
            end_ts,
        )
        if preview_state_keys:
            all_events = sorted(list(set(all_events + preview_state_keys)), key=lambda x: x.lower())
    except Exception:
        preview_state_keys = []

    event_data_source = data_for_plot if hasattr(data_for_plot, "event_data") else data_pkl
    cluster_state_color_map = {}
    try:
        event_manager = getattr(data_for_plot, "event_manager", None) or getattr(data_pkl, "event_manager", None)
        if isinstance(event_manager, dict):
            cluster_state_color_map = dict(((event_manager.get("cluster_states") or {}).get("color_map") or {}))
    except Exception:
        cluster_state_color_map = {}

    # Build a deterministic fallback for ordered cluster-like keys when explicit
    # style colors are absent.
    cluster_like_keys = []
    for ev in all_events:
        resolved_ev = _resolve_event_key_alias(event_data_source, ev)
        if looks_like_ordered_cluster_key(resolved_ev) and _has_state_duration_events(event_data_source, resolved_ev):
            cluster_like_keys.append(str(resolved_ev))
    if cluster_like_keys:
        fallback_cluster_map = build_ordered_cluster_color_map(cluster_like_keys)
        for k, v in fallback_cluster_map.items():
            cluster_state_color_map.setdefault(k, v)

    for i, ev in enumerate(all_events):
        resolved_ev = _resolve_event_key_alias(event_data_source, ev)
        style_saved = dict(event_styles.get(ev) or {})
        style_defaults = _default_event_style(ev, i)
        cluster_color = cluster_state_color_map.get(str(resolved_ev)) or cluster_state_color_map.get(str(ev))
        event_color = (
            _normalize_hex_color(style_saved.get("color"))
            or cluster_color
            or _mapped_event_color(ev)
            or _mapped_event_color(resolved_ev)
            or style_defaults["color"]
        )
        shade_color = _normalize_hex_color(style_saved.get("shade_color")) or cluster_color or event_color
        shade_enabled = bool(style_saved.get("shade_enabled", True))
        try:
            shade_opacity = float(style_saved.get("shade_opacity", 0.2))
        except Exception:
            shade_opacity = 0.2
        shade_opacity = max(0.0, min(1.0, shade_opacity))
        shade_mode = _normalize_event_shade_mode(style_saved.get("shade_mode"))
        shade_pct_min = _coerce_float_or_none(style_saved.get("shade_pct_min"))
        shade_pct_max = _coerce_float_or_none(style_saved.get("shade_pct_max"))
        shade_pct_min = 0.0 if shade_pct_min is None else max(0.0, min(100.0, shade_pct_min))
        shade_pct_max = 100.0 if shade_pct_max is None else max(0.0, min(100.0, shade_pct_max))
        event_symbol = str(style_saved.get("symbol") or style_defaults["symbol"])
        if event_symbol not in EVENT_SYMBOL_OPTIONS:
            event_symbol = "circle"
        is_state = _has_state_duration_events(event_data_source, resolved_ev)
        y_offset_frac = 0.05 * float(i)
        raw_targets = (event_targets or {}).get(ev, [])
        parsed_targets = []
        for target in raw_targets:
            t = str(target).strip()
            if not t:
                continue
            if t.lower() in {"all", "__all__", "*"}:
                for sig in ordered_signals:
                    if plot_channels.get(sig):
                        parsed_targets.append((sig, plot_channels[sig][0]))
                continue
            if "." in t:
                sig, ch = t.split(".", 1)
                if sig in ordered_signals and ch in (plot_channels.get(sig) or []):
                    parsed_targets.append((sig, ch))
                continue
            # Event-signal mapper targets: map signal to first active channel.
            sig = t
            if sig in ordered_signals and plot_channels.get(sig):
                parsed_targets.append((sig, plot_channels[sig][0]))
        deduped_targets = []
        seen_targets = set()
        for sig, ch in parsed_targets:
            key = (sig, ch)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            deduped_targets.append(key)
        parsed_targets = deduped_targets

        if not parsed_targets and ordered_signals and plot_channels.get(ordered_signals[0]):
            parsed_targets = [(ordered_signals[0], plot_channels[ordered_signals[0]][0])]
        if not parsed_targets:
            continue
        if not is_state:
            notes[ev] = [
                {
                    "event_key": resolved_ev,
                    "legend_key": ev,
                    "name": f"<b>{std_html.escape(ch)}</b>; <i>{std_html.escape(sig)}</i>; <b>{std_html.escape(ev)}</b>",
                    "signal": sig,
                    "channel": ch,
                    "symbol": event_symbol,
                    "color": event_color,
                    "showlegend": (j == 0),
                    "y_offset_frac": y_offset_frac,
                }
                for j, (sig, ch) in enumerate(parsed_targets)
            ]
        if is_state and shade_enabled:
            shade_rgba = _hex_to_rgba(shade_color, shade_opacity) or shade_color
            state_notes[resolved_ev] = [
                {
                    "signal": sig,
                    "channel": ch,
                    "color": shade_rgba,
                    "shade_mode": shade_mode,
                    "draw_line": False,
                    "start_marker": False,
                    "shade_pct_min": shade_pct_min,
                    "shade_pct_max": shade_pct_max,
                    "name": f"<b>{std_html.escape(ch)}</b>; <i>{std_html.escape(sig)}</i>; <b>{std_html.escape(ev)} (state)</b>",
                    "legend_key": ev,
                }
                for (sig, ch) in parsed_targets
            ]

    # Downsampled signal_data already prepared in data_for_plot above.

    zoom_channel = next((s for s in ["depth", "pressure", "odba"] if s in ordered_signals), ordered_signals[0])
    plot_kwargs = dict(
        data_pkl=data_for_plot,
        signals=ordered_signals,
        channels=plot_channels,
        time_range=(start_ts, end_ts),
        note_annotations=notes or None,
        state_annotations=state_notes or None,
        color_mapping_path=color_mapping_path,
        target_sampling_rate=None,
        zoom_start_time=start_ts,
        zoom_end_time=end_ts,
        zoom_range_selector_channel=zoom_channel,
        plot_event_values=[],
    )
    try:
        params = inspect.signature(plot_tag_data_interactive).parameters
        if "include_blank_row" in params:
            plot_kwargs["include_blank_row"] = False
        if "preserve_signal_order" in params:
            plot_kwargs["preserve_signal_order"] = True
        if "signal_metadata" in params:
            plot_kwargs["signal_metadata"] = _signal_display_metadata
        if "color_mapping" in params:
            plot_kwargs["color_mapping"] = _signal_colors
        if "persist_color_mapping" in params:
            plot_kwargs["persist_color_mapping"] = True
    except Exception:
        pass

    _current_fig = plot_tag_data_interactive(**plot_kwargs)
    # Force Plotly to treat slider-window changes as a new UI revision so the
    # main graph redraws to the new window instead of preserving stale viewport.
    try:
        if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
            lo = int(min(slider_value))
            hi = int(max(slider_value))
            _current_fig.update_layout(uirevision=f"slider:{lo}:{hi}")
    except Exception:
        pass

    # Remove Plotly's internal range selector/slider; external time controls are used instead.
    for axis_key in _trace_xaxis_keys(_current_fig):
        _current_fig.update_layout(
            {
                axis_key: dict(
                    rangeselector=dict(visible=False),
                    rangeslider=dict(visible=False),
                    range=[start_ts, end_ts],
                    autorange=False,
                )
            }
        )
    try:
        if playhead_time is None:
            playhead_time = float(_default_playhead_epoch(slider_default))
        playhead_ts = _from_epoch_seconds(float(playhead_time), tz_name)
        if start_ts <= playhead_ts <= end_ts:
            # Preserve existing annotation/state shapes; only replace the playhead line.
            existing_shapes = list(getattr(_current_fig.layout, "shapes", []) or [])
            kept_shapes = []
            for shp in existing_shapes:
                try:
                    shp_dict = shp.to_plotly_json() if hasattr(shp, "to_plotly_json") else dict(shp)
                except Exception:
                    shp_dict = shp
                is_playhead = (
                    isinstance(shp_dict, dict)
                    and shp_dict.get("type") == "line"
                    and str((shp_dict.get("line") or {}).get("color") or "").upper() == "#FFD166"
                    and shp_dict.get("yref") == "paper"
                )
                if not is_playhead:
                    kept_shapes.append(shp_dict)
            kept_shapes.append(_playhead_shape_dict(playhead_ts))
            _current_fig.update_layout(shapes=kept_shapes)
    except Exception:
        pass
    for idx, sig in enumerate(ordered_signals):
        cfg = dict(signal_axis_config_store.get(sig) or {})
        reverse_y = cfg.get("reverse")
        if reverse_y is None:
            reverse_y = _default_reverse_y_for_signal(sig)
        reverse_y = bool(reverse_y)
        enabled = cfg.get("enabled")
        if enabled is None:
            enabled = ("min" in cfg) or ("max" in cfg)
        axis_key = "yaxis" if idx == 0 else f"yaxis{idx + 1}"
        axis_ref = "y" if idx == 0 else f"y{idx + 1}"
        if not bool(enabled):
            _current_fig.update_layout({axis_key: dict(autorange=("reversed" if reverse_y else True), range=None)})
            continue
        min_v = _coerce_float_or_none(cfg.get("min"))
        max_v = _coerce_float_or_none(cfg.get("max"))
        if min_v is None and max_v is None:
            _current_fig.update_layout({axis_key: dict(autorange=("reversed" if reverse_y else True), range=None)})
            continue
        existing = getattr(getattr(_current_fig.layout, axis_key, None), "range", None)
        if existing and len(existing) == 2:
            lo = _coerce_float_or_none(existing[0])
            hi = _coerce_float_or_none(existing[1])
        else:
            lo, hi = _estimate_axis_y_range(_current_fig, axis_ref)
        lo = min_v if min_v is not None else lo
        hi = max_v if max_v is not None else hi
        if lo is None or hi is None:
            continue
        if lo > hi:
            lo, hi = hi, lo
        _current_fig.update_layout({axis_key: dict(range=([hi, lo] if reverse_y else [lo, hi]), autorange=False)})
    _current_fig.update_layout(
        showlegend=False,
        hovermode="x unified",
        hoverdistance=-1,
        spikedistance=-1,
    )
    for axis_key in _trace_xaxis_keys(_current_fig):
        _current_fig.update_layout(
            {
                axis_key: dict(
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    spikecolor="rgba(255, 209, 102, 0.75)",
                    spikethickness=1,
                )
            }
        )
    _current_fig.update_traces(showlegend=False)
    return _current_fig


@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Input("playhead-time", "data"),
    State("main-plot", "figure"),
    State("time-range-slider", "value"),
    State("is-playing", "data"),
    prevent_initial_call=True,
)
def update_playhead_line_only(playhead_time, existing_fig, slider_value, is_playing):
    # The yellow playhead line is moved clientside (Plotly.relayout) on every
    # playhead change — during playback AND drag scrubbing — to avoid a full-figure
    # server round-trip per tick. This server callback only runs to ADD or REMOVE
    # the shape when the playhead crosses the window edge (clientside can only move
    # an existing shape), so live dragging stays smooth.
    if is_playing:
        raise dash.exceptions.PreventUpdate
    if not existing_fig:
        raise dash.exceptions.PreventUpdate
    try:
        playhead_ts = _from_epoch_seconds(float(playhead_time), tz_name)
    except Exception:
        raise dash.exceptions.PreventUpdate

    if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
        start_ts = _from_epoch_seconds(min(slider_value), tz_name)
        end_ts = _from_epoch_seconds(max(slider_value), tz_name)
    else:
        start_ts = pd.Timestamp(start_default)
        end_ts = pd.Timestamp(end_default)

    # If the line already exists and the playhead is inside the window, the
    # clientside mover has it covered — skip the expensive figure round-trip.
    has_playhead_shape = any(
        isinstance(shp, dict)
        and shp.get("type") == "line"
        and shp.get("yref") == "paper"
        and str((shp.get("line") or {}).get("color") or "").upper() == "#FFD166"
        for shp in (existing_fig.get("layout", {}).get("shapes") or [])
    )
    inside_window = start_ts <= playhead_ts <= end_ts
    if has_playhead_shape and inside_window:
        raise dash.exceptions.PreventUpdate

    shapes = []
    for shp in (existing_fig.get("layout", {}).get("shapes") or []):
        if (
            isinstance(shp, dict)
            and shp.get("type") == "line"
            and str((shp.get("line") or {}).get("color") or "").upper() == "#FFD166"
            and shp.get("yref") == "paper"
        ):
            continue
        shapes.append(shp)

    if start_ts <= playhead_ts <= end_ts:
        shapes.append(_playhead_shape_dict(playhead_ts))

    fig = dict(existing_fig)
    layout = dict(fig.get("layout") or {})
    layout["shapes"] = shapes
    for axis_key in _trace_xaxis_keys(fig):
        layout_axis = dict((layout.get(axis_key) or {}))
        layout_axis["range"] = [start_ts, end_ts]
        layout_axis["autorange"] = False
        layout[axis_key] = layout_axis
    fig["layout"] = layout
    return fig


@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Input("main-plot", "relayoutData"),
    prevent_initial_call=True,
)
def update_graph_on_zoom(relayoutdata):
    """Use FigureResampler patch updates with axis-safe relayout filtering."""
    global _current_fig
    if not ENABLE_RELAYOUT_PATCH:
        raise dash.exceptions.PreventUpdate
    if _current_fig is None or not relayoutdata:
        raise dash.exceptions.PreventUpdate

    clean = _clean_relayout_data(relayoutdata, _trace_xaxis_keys(_current_fig))
    if not clean:
        raise dash.exceptions.PreventUpdate
    x0, x1 = _extract_primary_xaxis_range(clean)
    if x0 is None or x1 is None:
        raise dash.exceptions.PreventUpdate
    lo_epoch = _coerce_epoch_from_relayout_value(x0)
    hi_epoch = _coerce_epoch_from_relayout_value(x1)
    if lo_epoch is None or hi_epoch is None:
        raise dash.exceptions.PreventUpdate
    if hi_epoch < lo_epoch:
        lo_epoch, hi_epoch = hi_epoch, lo_epoch

    s_min = int(slider_min)
    s_max = int(slider_max)
    if not (
        _epoch_within_slider_bounds(lo_epoch, s_min, s_max, pad_seconds=300)
        and _epoch_within_slider_bounds(hi_epoch, s_min, s_max, pad_seconds=300)
    ):
        raise dash.exceptions.PreventUpdate

    lo_iso = _from_epoch_seconds(int(lo_epoch), tz_name).isoformat()
    hi_iso = _from_epoch_seconds(int(hi_epoch), tz_name).isoformat()
    for axis_key in _trace_xaxis_keys(_current_fig):
        clean[f"{axis_key}.range[0]"] = lo_iso
        clean[f"{axis_key}.range[1]"] = hi_iso

    try:
        return _current_fig.construct_update_data_patch(clean)
    except KeyError:
        # Fallback to primary x-axis-only range.
        if "xaxis.range[0]" in clean and "xaxis.range[1]" in clean:
            try:
                return _current_fig.construct_update_data_patch(
                    {
                        "xaxis.range[0]": clean["xaxis.range[0]"],
                        "xaxis.range[1]": clean["xaxis.range[1]"],
                    }
                )
            except Exception:
                return no_update
        return no_update
    except Exception:
        return no_update


@app.callback(
    Output("custom-events-panel", "children"),
    Output("custom-signals-panel", "children"),
    Input("ordered-signals-store", "data"),
    Input("channels-store", "data"),
    Input("channel-order-store", "data"),
    Input("signal-axis-config-store", "data"),
    Input("events-select", "value"),
    Input("event-style-store", "data"),
    Input("color-map-version", "data"),
)
def render_custom_legend(
    ordered_signals,
    channels_store,
    channel_order_store,
    signal_axis_config_store,
    event_keys,
    event_styles,
    _color_map_version,
):
    ordered_signals = [s for s in (ordered_signals or []) if s in _available_signal_names(data_pkl)]
    if not ordered_signals:
        ordered_signals = default_signals[:]

    channels_store = channels_store or {}
    channel_order_store = channel_order_store or {}
    signal_axis_config_store = signal_axis_config_store or {}
    event_styles = event_styles or {}

    signal_rows = []
    for sig in ordered_signals:
        sig_meta = (
            _signal_display_metadata.get(sig)
            or _signal_display_metadata.get(str(sig).lower())
            or _signal_display_metadata.get(str(sig).upper())
            or {}
        )
        signal_label = _safe_text((sig_meta or {}).get("label")) or str(sig)
        options = _signal_channels(data_pkl, sig)
        chosen = [c for c in (channels_store.get(sig) or []) if c in options]
        selected = chosen or options
        plot_channels = _normalize_channel_order(selected, channel_order_store.get(sig))
        channel_rows = []
        for ch in plot_channels:
            color = _resolve_trace_color(_signal_colors, sig, ch) or "#9CC7E1"
            channel_rows.append(
                html.Div(
                    [
                        html.Span("", className="chip-handle", **{"aria-hidden": "true"}),
                        html.Button(
                            "",
                            id={"type": "chip-color-btn", "group": "channels", "key": sig, "value": ch},
                            className="legend-swatch legend-swatch-btn",
                            n_clicks=0,
                            title=f"Edit {sig}.{ch}",
                            style={"backgroundColor": color},
                        ),
                        html.Span(ch, className="legend-main"),
                        html.Button(
                            "×",
                            id={"type": "chip-remove-btn", "group": "channels", "key": sig, "value": ch},
                            className="chip-remove-btn",
                            n_clicks=0,
                            title=f"Remove {sig}.{ch}",
                        ),
                    ],
                    className="chip-item legend-row legend-channel-row",
                    style={
                        "--channel-color": color,
                        "backgroundColor": "#123B59",
                        "borderColor": "var(--channel-color)",
                        "borderWidth": "1px",
                        "borderStyle": "solid",
                        "color": "#FFFFFF",
                    },
                    draggable="true",
                    **{"data-value": ch, "data-color-editable": "0"},
                    key=f"legend-channel-{sig}-{ch}",
                )
            )
        count = len(channel_rows)
        sig_axis_cfg = dict(signal_axis_config_store.get(sig) or {})
        min_cfg = _coerce_float_or_none(sig_axis_cfg.get("min"))
        max_cfg = _coerce_float_or_none(sig_axis_cfg.get("max"))
        custom_range_enabled = sig_axis_cfg.get("enabled")
        reverse_y_enabled = sig_axis_cfg.get("reverse")
        if custom_range_enabled is None:
            custom_range_enabled = (min_cfg is not None) or (max_cfg is not None)
        if reverse_y_enabled is None:
            reverse_y_enabled = _default_reverse_y_for_signal(sig)
        custom_range_enabled = bool(custom_range_enabled)
        reverse_y_enabled = bool(reverse_y_enabled)
        signal_rows.append(
            html.Details(
                [
                    html.Div(className="legend-signal-edge legend-signal-edge-top"),
                    html.Div(className="legend-signal-edge legend-signal-edge-right"),
                    html.Div(className="legend-signal-edge legend-signal-edge-bottom"),
                    html.Div(className="legend-signal-edge legend-signal-edge-left"),
                    html.Summary(
                        [
                            html.Div(
                                [
                                    html.Span("", className="chip-handle legend-signal-drag-handle", **{"aria-hidden": "true"}),
                                    html.Span(
                                        [
                                            html.B(signal_label, className="legend-group-label"),
                                            html.I(str(sig), className="legend-group-raw"),
                                        ],
                                        className="legend-group-title",
                                    ),
                                    html.Span(f"{count} ch", className="legend-group-count"),
                                ],
                                className="legend-group-left legend-signal-drag-surface",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "×",
                                        id={"type": "chip-remove-btn", "group": "signals", "key": "__all__", "value": sig},
                                        className="chip-remove-btn",
                                        n_clicks=0,
                                        title=f"Remove signal {sig}",
                                    )
                                ],
                                className="legend-actions",
                            ),
                        ],
                        className="legend-group-summary",
                    ),
                    html.Div(
                        [
                            html.Div(
                                channel_rows or [html.Div("No channels.", className="legend-empty")],
                                className="chip-sortable legend-children",
                                **{"data-order-group": "channels", "data-order-key": str(sig)},
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id={"type": "sig-reverse-y", "sig": sig},
                                        options=[{"label": "Reverse Y", "value": "on"}],
                                        value=["on"] if reverse_y_enabled else [],
                                        className="signal-custom-range-toggle",
                                    ),
                                    dcc.Checklist(
                                        id={"type": "sig-custom-range-enabled", "sig": sig},
                                        options=[{"label": "Enter Custom Range", "value": "on"}],
                                        value=["on"] if custom_range_enabled else [],
                                        className="signal-custom-range-toggle",
                                    ),
                                ],
                                className="signal-custom-range-toggle-row",
                            ),
                            html.Div(
                                [
                                    html.Label("Y min", className="event-style-label"),
                                    dcc.Input(
                                        id={"type": "sig-y-min", "sig": sig},
                                        type="text",
                                        value="" if min_cfg is None else str(min_cfg),
                                        debounce=True,
                                        className="event-color-input",
                                        placeholder="auto",
                                    ),
                                    html.Label("Y max", className="event-style-label"),
                                    dcc.Input(
                                        id={"type": "sig-y-max", "sig": sig},
                                        type="text",
                                        value="" if max_cfg is None else str(max_cfg),
                                        debounce=True,
                                        className="event-color-input",
                                        placeholder="auto",
                                    ),
                                ],
                                className="signal-axis-config-row",
                                style=None if custom_range_enabled else {"display": "none"},
                            ),
                        ],
                        className="legend-children-wrap",
                    ),
                ],
                className="legend-group dnd-item legend-signal-group",
                open=True,
                draggable="true",
                **{"data-value": str(sig), "data-color-editable": "0"},
                key=f"legend-signal-{sig}",
            )
        )

    if not signal_rows:
        signal_rows = [html.Div("No channels selected.", className="legend-empty")]

    all_events = sorted([str(e) for e in (event_keys or []) if str(e).strip()], key=lambda x: x.lower())

    grouped_events = group_event_keys(all_events)
    flat_order = []
    for _parent, child_keys in grouped_events.items():
        flat_order.extend(child_keys)
    style_index = {ev: idx for idx, ev in enumerate(flat_order)}

    event_rows = []
    for parent_name, child_keys in grouped_events.items():
        child_rows = []
        for ev in child_keys:
            i = style_index.get(ev, 0)
            style_saved = dict(event_styles.get(ev) or {})
            defaults = _default_event_style(ev, i)
            event_color = _normalize_hex_color(style_saved.get("color")) or defaults["color"]
            symbol_value = str(style_saved.get("symbol") or defaults["symbol"])
            if symbol_value not in EVENT_SYMBOL_OPTIONS:
                symbol_value = "circle"
            glyph = EVENT_SYMBOL_GLYPHS.get(symbol_value, "•")
            is_state = _is_state_event_key(data_pkl, ev)

            row_children = [
                html.Button(
                    "",
                    id={"type": "event-edit-btn", "group": "events", "event": ev},
                    className="legend-swatch legend-swatch-btn",
                    n_clicks=0,
                    title=f"Edit event {ev}",
                    style={"backgroundColor": event_color},
                ),
                html.Button(
                    glyph,
                    id={"type": "event-shape-btn", "group": "events", "event": ev},
                    className="legend-event-shape",
                    n_clicks=0,
                    style={"color": event_color, "borderColor": event_color},
                    title=f"{symbol_value} marker",
                ),
                html.Div(
                    [
                        html.Span(_event_label_display_text(ev), className="legend-main", title=ev),
                        html.Span("State" if is_state else "Point", className="legend-sub"),
                    ],
                    className="legend-event-text",
                ),
                html.Button(
                    "⟲",
                    id={"type": "event-node-btn", "event": ev},
                    className="chip-node-btn",
                    n_clicks=0,
                    title="Focus this event in node diagram",
                ),
                html.Button(
                    "×",
                    id={"type": "chip-remove-btn", "group": "events", "key": "__all__", "value": ev},
                    className="chip-remove-btn",
                    n_clicks=0,
                    title=f"Remove event {ev}",
                ),
            ]

            if is_state and bool(style_saved.get("shade_enabled", True)):
                shade_color = _normalize_hex_color(style_saved.get("shade_color")) or event_color
                row_children.insert(
                    2,
                    html.Span(
                        className="legend-shade",
                        style={"backgroundColor": shade_color},
                        title="Duration shading color",
                    ),
                )

            child_rows.append(
                html.Div(
                    row_children,
                    className="chip-item legend-row legend-event-row",
                    style={
                        "--event-color": event_color,
                        "backgroundColor": "#123B59",
                        "borderColor": "var(--event-color)",
                        "borderWidth": "1px",
                        "borderStyle": "solid",
                        "color": "#FFFFFF",
                    },
                    key=f"legend-event-{ev}",
                )
            )

        if len(child_keys) <= 1:
            event_rows.extend(child_rows)
        else:
            event_rows.append(
                html.Details(
                    [
                        html.Summary(
                            [
                                html.Div(
                                    [
                                        html.Span(parent_name, className="legend-group-label"),
                                        html.Span(f"{len(child_keys)} events", className="legend-group-count"),
                                    ],
                                    className="legend-group-left",
                                ),
                            ],
                            className="legend-group-summary",
                        ),
                        html.Div(child_rows, className="legend-children-wrap"),
                    ],
                    className="legend-group legend-signal-group",
                    open=True,
                    key=f"legend-event-group-{parent_name}",
                )
            )

    if not event_rows:
        event_rows = [html.Div("No events selected.", className="legend-empty")]

    events_children = html.Div(
        [
            html.Div(event_rows, className="legend-list legend-list-active legend-list-active-events"),
            html.Div(
                [
                    html.Button(
                        "Write preview cluster states to deployment",
                        id="main-write-cluster-preview-btn",
                        n_clicks=0,
                        className="apply-btn",
                    ),
                    html.Div(id="main-write-cluster-preview-status", className="window-summary"),
                ],
                className="color-editor-actions",
            ),
        ],
        className="control-card legend-card legend-active-card",
    )
    signals_children = html.Div(
        html.Div(
            signal_rows,
            className="legend-list legend-list-active legend-list-active-signals signal-sortable",
            **{"data-order-group": "signals", "data-order-key": "__all__", "data-dnd-axis": "y"},
        ),
        className="control-card legend-card legend-active-card",
    )
    return events_children, signals_children


@app.callback(
    Output("selected-signals-store", "data", allow_duplicate=True),
    Output("legend-add-signal-select", "value"),
    Input("legend-add-signal-select", "value"),
    Input("legend-add-signal-btn", "n_clicks"),
    State("legend-add-signal-select", "value"),
    State("selected-signals-store", "data"),
    prevent_initial_call=True,
)
def add_signal_from_legend(_selected_value, _n_clicks, add_signal, selected_signals):
    sig = str(add_signal or "").strip()
    if not sig or sig not in _available_signal_names(data_pkl):
        raise dash.exceptions.PreventUpdate
    selected = [str(s) for s in (selected_signals or []) if str(s).strip()]
    if sig in selected:
        return selected, None
    return selected + [sig], None


@app.callback(
    Output("legend-add-signal-select", "options"),
    Input("selected-signals-store", "data"),
)
def update_legend_add_signal_options(selected_signals):
    opts = _available_signal_names(data_pkl)
    return [{"label": s, "value": s} for s in opts]


@app.callback(
    Output("selected-events-store", "data", allow_duplicate=True),
    Output("legend-add-event-select", "value"),
    Input("legend-add-event-btn", "n_clicks"),
    State("legend-add-event-select", "value"),
    State("selected-events-store", "data"),
    prevent_initial_call=True,
)
def add_event_from_legend(_n_clicks, add_event, selected_events):
    ev = str(add_event or "").strip()
    available = _available_event_keys(data_pkl)
    if not ev or ev not in available:
        raise dash.exceptions.PreventUpdate
    selected = [str(e) for e in (selected_events or []) if str(e).strip()]
    if ev in selected:
        return selected, None
    return selected + [ev], None


@app.callback(
    Output("map-event-select", "value", allow_duplicate=True),
    Output("map-selected-event", "data", allow_duplicate=True),
    Input({"type": "event-node-btn", "event": ALL}, "n_clicks"),
    State("selected-events-store", "data"),
    prevent_initial_call=True,
)
def focus_event_in_node_mapper(_n_clicks, selected_events):
    if not ctx.triggered or not (ctx.triggered[0].get("value") or 0):
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        raise dash.exceptions.PreventUpdate
    ev = str(trig.get("event") or "").strip()
    events = [str(v) for v in (selected_events or []) if str(v).strip()]
    if not ev or ev not in events:
        raise dash.exceptions.PreventUpdate
    return ev, ev


@app.callback(
    Output("legend-add-event-select", "options"),
    Input("selected-events-store", "data"),
)
def update_legend_add_event_options(selected_events):
    opts = _available_event_keys(data_pkl)
    return [{"label": e, "value": e} for e in opts]


@app.callback(
    Output("main-write-cluster-preview-status", "children"),
    Input("main-write-cluster-preview-btn", "n_clicks"),
    State("selected-events-store", "data"),
    State("time-range-slider", "value"),
    prevent_initial_call=True,
)
def save_main_cluster_preview_events(_n_clicks, selected_events, slider_value):
    selected = [str(v).strip() for v in (selected_events or []) if str(v or "").strip()]
    resolved = [_resolve_event_key_alias(data_pkl, ev) for ev in selected]
    cluster_keys = sorted({str(ev).strip() for ev in resolved if looks_like_ordered_cluster_key(str(ev))}, key=lambda x: x.lower())
    if not cluster_keys:
        return "Select at least one cluster state event (kN) to write."

    try:
        if isinstance(slider_value, (list, tuple)) and len(slider_value) == 2:
            lo = float(min(slider_value))
            hi = float(max(slider_value))
        else:
            lo = float(slider_min)
            hi = float(slider_max)
        start_ts = _from_epoch_seconds(lo, tz_name)
        end_ts = _from_epoch_seconds(hi, tz_name)
    except Exception:
        start_ts = global_start
        end_ts = global_end

    preview_df = _load_cluster_preview_rows(dataset_id, deployment_id, cluster_keys, start_ts=start_ts, end_ts=end_ts)
    if preview_df.empty:
        return "No preview cluster state rows found in meta-analysis outputs for the selected window."

    real_pkl_path = os.path.join(data_dir, dataset_id, deployment_id, "outputs", "data.pkl")
    deployment_folder = os.path.join(data_dir, dataset_id, deployment_id)
    payload = {
        "algorithmic_cfg": {"event_keys": {f"k{i}": key for i, key in enumerate(cluster_keys)}},
        "event_rows": preview_df.to_dict("records"),
        "label_rows": [],
        "signal_info": {},
        "event_manager": {
            "keys": cluster_keys,
            "source": "meta_analysis_cluster_preview",
            "window_start": start_ts.isoformat() if start_ts is not None else None,
            "window_end": end_ts.isoformat() if end_ts is not None else None,
        },
    }

    try:
        result = save_algorithmic_results_to_real_pkl(
            real_data_pkl_path=real_pkl_path,
            payload=payload,
            target_event_keys=cluster_keys,
            write_netcdf=True,
            deployment_folder=deployment_folder,
            deployment_id=deployment_id,
        )
    except Exception as exc:
        return f"Failed to write cluster preview events: {type(exc).__name__}: {exc}"

    if result.get("netcdf_error"):
        return (
            f"Saved {result.get('saved_event_count', 0)} cluster state events to data.pkl for {len(cluster_keys)} key(s), "
            f"but NetCDF export failed: {result.get('netcdf_error')}"
        )
    return (
        f"Saved {result.get('saved_event_count', 0)} cluster state events to data.pkl and NetCDF "
        f"for {len(cluster_keys)} key(s)."
    )


if __name__ == "__main__":
    # threaded=True is required: the /immich-video and /local-video routes stream
    # long-lived responses; a single-threaded server would block every other
    # request (including the page's own) behind an open video stream.
    app.run(debug=False, port=args.port, use_reloader=False, threaded=True)
