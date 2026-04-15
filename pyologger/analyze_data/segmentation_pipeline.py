import argparse
from collections import ChainMap
import glob
import hashlib
import html
import json
import os
import pickle
import posixpath
import re
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure direct workflow execution resolves the repo-local pyologger package.
MODULE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_DIR.parents[1]
PROJECT_ROOT = MODULE_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pyologger.analyze_data.find_segments import find_segments
from pyologger.analyze_data.segmentation_run_summaries import write_segmentation_run_summary_parquets
from pyologger.utils.event_manager import create_state_event
from pyologger.utils.folder_manager import load_combined_config
from pyologger.utils.cluster_colors import build_ordered_cluster_color_map
from pyologger.utils.workflow_netcdf import latest_processing_netcdf_path
from pyologger.utils.segmentation_qc import (
    load_standardized_channel_unit_map,
    normalize_unit_token,
)
from pyologger.utils.segmentation_run_config import normalize_segmentation_run_cfg
try:
    from sklearn.cluster import KMeans
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except Exception:
    TSNE_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import pycatch22
    CATCH22_AVAILABLE = True
except Exception:
    CATCH22_AVAILABLE = False

import yaml

_CATCH22_NAMES_CACHE = None
_SUPERVISED_LABEL_ALIAS_CACHE = None


@dataclass
class RunContext:
    run_name: str
    run_cfg: Dict
    global_cfg: Dict
    data_root: str
    output_root: str
    analysis_id: str
    scope: List[Dict]


def _load_yaml(path: str) -> Dict:
    cfg, _, _ = load_combined_config(config_path=path)
    return cfg


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_rss_gb() -> float:
    """Best-effort process RSS in GB (macOS/Linux compatible)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss > 10 ** 10:
                return rss / (1024 ** 3)
            return rss / (1024 ** 2)
        except Exception:
            return float("nan")


def _should_log(last_log_time: float, interval_s: float) -> bool:
    return (time.monotonic() - last_log_time) >= max(float(interval_s), 1.0)


def _deployment_timezone_name(data_pkl) -> str:
    return str((getattr(data_pkl, "deployment_info", {}) or {}).get("Time Zone") or "UTC").strip() or "UTC"


def _normalize_datetime_series_to_timezone(values, tz_name: str) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce")
    if not isinstance(dt, pd.Series):
        dt = pd.Series(dt)
    if not tz_name:
        return dt
    try:
        if getattr(dt.dt, "tz", None) is None:
            return dt.dt.tz_localize(tz_name)
        return dt.dt.tz_convert(tz_name)
    except Exception:
        return dt


def _normalize_window_columns(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    out = df.copy()
    for col in ("window_start", "window_end"):
        if col in out.columns:
            out[col] = _normalize_datetime_series_to_timezone(out[col], tz_name)
    return out


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _summary_focus_cfg(run_cfg: Dict) -> Dict[str, Any]:
    summary_cfg = dict((run_cfg or {}).get("summary") or {})
    actograms_cfg = dict(summary_cfg.get("actograms") or {})
    pass_mode = str(cfg.get("context_filter_pass_mode") or "measured_only").strip().lower() or "measured_only"
    if pass_mode not in {"none", "full", "measured_only"}:
        raise ValueError(
            f"Unsupported context_filter_pass_mode '{pass_mode}'. "
            "Expected one of: none, measured_only, full."
        )
    return {
        "primary_method_kind": str(summary_cfg.get("primary_method_kind") or "").strip().lower(),
        "primary_method_id": str(summary_cfg.get("primary_method_id") or "").strip(),
        "budget_label_mode": str(summary_cfg.get("budget_label_mode") or "canonical").strip().lower() or "canonical",
        "actograms": {
            "daily": bool(actograms_cfg.get("daily", True)),
            "continuous": bool(actograms_cfg.get("continuous", True)),
        },
    }


def _resolve_primary_budget_method(
    method_meta_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None,
    run_cfg: Dict,
) -> tuple[str | None, str | None]:
    if method_meta_df is None or method_meta_df.empty:
        return None, None
    focus_cfg = _summary_focus_cfg(run_cfg)
    method_meta_df = method_meta_df.copy()
    method_meta_df["method"] = method_meta_df["method"].astype(str)
    method_meta_df["method_variant"] = method_meta_df["method_variant"].astype(str)
    explicit_kind = focus_cfg["primary_method_kind"]
    explicit_id = focus_cfg["primary_method_id"]
    if explicit_kind and explicit_id:
        hit = method_meta_df[
            (method_meta_df["method"].str.lower() == explicit_kind)
            & (method_meta_df["method_variant"] == explicit_id)
        ]
        if not hit.empty:
            row = hit.iloc[0]
            return str(row["method"]), str(row["method_variant"])

    algo = method_meta_df.loc[method_meta_df["method"].str.lower() == "algorithmic"]
    if not algo.empty:
        row = algo.sort_values(["method_variant"]).iloc[0]
        return str(row["method"]), str(row["method_variant"])

    if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
        m = metrics_df.copy()
        m["metric"] = m["metric"].astype(str)
        m["label"] = m["label"].astype(str)
        target = m[(m["metric"] == "balanced_accuracy") & (m["label"] == "Overall")]
        if not target.empty and "variant_id" in target.columns:
            best_variant = str(target.sort_values("value", ascending=False).iloc[0]["variant_id"])
            sup = method_meta_df[
                (method_meta_df["method"].str.lower() == "supervised")
                & (method_meta_df["method_variant"] == best_variant)
            ]
            if not sup.empty:
                row = sup.iloc[0]
                return str(row["method"]), str(row["method_variant"])

    sup = method_meta_df.loc[method_meta_df["method"].str.lower() == "supervised"]
    if not sup.empty:
        row = sup.sort_values(["method_variant"]).iloc[0]
        return str(row["method"]), str(row["method_variant"])
    row = method_meta_df.sort_values(["method", "method_variant"]).iloc[0]
    return str(row["method"]), str(row["method_variant"])


def _detect_supervised_positive_label(labels: List[str]) -> str | None:
    label_texts = [str(x).strip() for x in labels if str(x).strip()]
    for candidate in label_texts:
        if candidate.lower() == "sleep":
            return candidate
    for candidate in label_texts:
        if "sleep" in candidate.lower():
            return candidate
    return _primary_metric_focus_label(label_texts)


def _load_repo_color_mapping() -> Dict:
    path = os.path.join(PROJECT_ROOT, "color_mappings.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _metric_label_order(labels: List[str]) -> List[str]:
    raw = [str(x).strip() for x in labels if str(x).strip()]
    unique = []
    for label in raw:
        if label not in unique:
            unique.append(label)
    non_overall = [x for x in unique if x != "Overall"]
    preferred = ["REM", "SWS", "SLEEP", "WAKE", "ACTIVE"]
    ordered = [x for x in preferred if x in non_overall]
    ordered += sorted([x for x in non_overall if x not in ordered], key=lambda x: x.upper())
    if "Overall" in unique:
        ordered.append("Overall")
    return ordered


def _metric_label_color_map(labels: List[str], repo_colors: Dict | None = None) -> Dict[str, str]:
    repo_colors = repo_colors or {}
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
    out: Dict[str, str] = {}
    fallback_idx = 0
    for label in _metric_label_order(labels):
        if label == "Overall":
            out[label] = "#6b7a88"
            continue
        lookup_chain = [
            label,
            label.strip(),
            label.replace("_", " "),
            label.replace("-", " "),
            label.title(),
            label.upper(),
        ]
        upper = label.strip().upper()
        if upper == "WAKE":
            lookup_chain.extend(["Active Waking", "ACTIVE", "WAKE"])
        elif upper == "ACTIVE":
            lookup_chain.extend(["Active Waking", "WAKE", "ACTIVE"])
        elif upper == "SLEEP":
            lookup_chain.extend(["HV Slow Wave Sleep", "SWS", "SLEEP"])
        elif upper == "SWS":
            lookup_chain.extend(["HV Slow Wave Sleep", "SLEEP", "SWS"])
        elif upper == "REM":
            lookup_chain.extend(["REM", "Certain REM", "Putative REM"])
        color = next((repo_colors.get(k) for k in lookup_chain if k in repo_colors), None)
        if color is None:
            color = palette[fallback_idx % len(palette)]
            fallback_idx += 1
        out[label] = color
    return out


def _primary_metric_focus_label(labels: List[str]) -> str | None:
    ordered = [x for x in _metric_label_order(labels) if x != "Overall"]
    for preferred in ["REM", "SWS", "SLEEP", "WAKE", "ACTIVE"]:
        if preferred in ordered:
            return preferred
    return ordered[0] if ordered else None


def _supervised_confusion_style_map() -> Dict[str, Dict[str, str]]:
    defaults = {
        "true_negative": {"label": "True Negative", "color": "#d9dde3"},
        "false_positive": {"label": "False Positive", "color": "#f6e7a8"},
        "false_negative": {"label": "False Negative", "color": "#f3c7c3"},
        "true_positive": {"label": "True Positive", "color": "#cfe9cf"},
    }
    raw = (_load_repo_color_mapping().get("__supervised_confusion_colors__") or {})
    out = {}
    for key, default in defaults.items():
        style = raw.get(key) if isinstance(raw, dict) else None
        if isinstance(style, dict):
            out[key] = {
                "label": str(style.get("label") or default["label"]),
                "color": str(style.get("color") or default["color"]),
            }
        else:
            out[key] = default.copy()
    return out


def _supervised_confusion_label(key: str, positive_label: str | None) -> str:
    positive = str(positive_label).strip() or "positive state"
    negative = f"not {positive}"
    labels = {
        "true_negative": f"True Negative: observed {negative}, predicted {negative}",
        "false_positive": f"False Positive: predicted {positive}, observed {negative}",
        "false_negative": f"False Negative: observed {positive}, predicted {negative}",
        "true_positive": f"True Positive: observed {positive}, predicted {positive}",
    }
    return labels.get(str(key), str(key).replace("_", " ").title())


def _supervised_confusion_legend_label(key: str, positive_label: str | None) -> str:
    labels = {
        "true_negative": "True Negative",
        "false_positive": "False Positive",
        "false_negative": "False Negative",
        "true_positive": "True Positive",
    }
    return labels.get(str(key), _supervised_confusion_label(key, positive_label))


def _build_supervised_confusion_overlay(
    dep_spdf: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], str | None]:
    if dep_spdf is None or dep_spdf.empty:
        return pd.DataFrame(), {}, None

    required = {"window_start", "window_end", "predicted_label"}
    if not required.issubset(dep_spdf.columns):
        return pd.DataFrame(), {}, None

    work = dep_spdf.copy()
    work = work.dropna(subset=["window_start", "window_end", "predicted_label"]).sort_values("window_start")
    if work.empty:
        return pd.DataFrame(), {}, None

    observed_series = work.get("observed_label", pd.Series(dtype=object))
    observed_nonempty = observed_series.dropna().astype(str).map(str.strip)
    observed_nonempty = observed_nonempty[~observed_nonempty.str.lower().isin({"", "nan", "none"})]
    has_observed_labels = not observed_nonempty.empty

    available_labels = pd.Index(
        observed_nonempty.tolist()
        + work["predicted_label"].dropna().astype(str).tolist()
    ).unique().tolist()
    positive_label = _detect_supervised_positive_label(available_labels)
    if positive_label is None or not has_observed_labels:
        event_df = work.rename(
            columns={"window_start": "datetime", "window_end": "end_datetime", "predicted_label": "key"}
        )[["datetime", "end_datetime", "key"]].copy()
        event_df["type"] = "state"
        event_df["duration"] = (event_df["end_datetime"] - event_df["datetime"]).dt.total_seconds()
        repo_colors = _load_repo_color_mapping()
        keys = sorted(event_df["key"].dropna().astype(str).unique().tolist())
        annotations = {
            k: {
                "signal": "all",
                "color": repo_colors.get(k),
                "shade_mode": "fill_trace_split",
                "shade_opacity": 0.25,
                "draw_line": False,
                "name": str(k),
            }
            for k in keys
        }
        return event_df.sort_values("datetime").reset_index(drop=True), annotations, positive_label

    positive_lower = str(positive_label).strip().lower()
    observed_is_positive = work["observed_label"].astype(str).str.strip().str.lower() == positive_lower
    predicted_is_positive = work["predicted_label"].astype(str).str.strip().str.lower() == positive_lower
    work["key"] = np.where(
        (~observed_is_positive) & (~predicted_is_positive),
        "true_negative",
        np.where(
            (~observed_is_positive) & predicted_is_positive,
            "false_positive",
            np.where(
                observed_is_positive & (~predicted_is_positive),
                "false_negative",
                "true_positive",
            ),
        ),
    )

    event_df = work.rename(columns={"window_start": "datetime", "window_end": "end_datetime"})[
        ["datetime", "end_datetime", "key"]
    ].copy()
    event_df["type"] = "state"
    event_df["duration"] = (event_df["end_datetime"] - event_df["datetime"]).dt.total_seconds()

    confusion_styles = _supervised_confusion_style_map()
    confusion_order = ["true_negative", "false_positive", "false_negative", "true_positive"]
    annotations = {
        key: {
            "signal": "all",
            "color": confusion_styles[key]["color"],
            "shade_mode": "fill_trace_split",
            "shade_opacity": 0.28,
            "draw_line": False,
            "name": _supervised_confusion_label(key, positive_label),
        }
        for key in confusion_order
        if key in set(event_df["key"].astype(str))
    }
    return event_df.sort_values("datetime").reset_index(drop=True), annotations, positive_label


def _interactive_state_fill_legend_html(
    state_annotations: Dict[str, Dict[str, Any]] | None,
    *,
    positive_label: str | None = None,
) -> str:
    ordered_keys = ["true_negative", "false_positive", "false_negative", "true_positive"]
    if not state_annotations and positive_label is None:
        return ""
    chunks = []
    styles = _supervised_confusion_style_map()
    for key in ordered_keys:
        cfg = (state_annotations or {}).get(key) or {}
        color = str(cfg.get("color") or "#8a99a8")
        if not cfg:
            color = str(styles.get(key, {}).get("color") or color)
        label = str(cfg.get("name") or _supervised_confusion_label(key, positive_label))
        chunks.append(
            f"<span style=\"white-space:nowrap;\"><span style=\"color:{html.escape(color)};\">■</span> {html.escape(label)}</span>"
        )
    return "<br><sup>Fill colors: " + " &nbsp;&nbsp; ".join(chunks) + "</sup>"


def _interactive_confusion_counts_html(event_df: pd.DataFrame | None) -> str:
    if event_df is None:
        return ""
    ordered_keys = ["true_negative", "false_positive", "false_negative", "true_positive"]
    counts = (
        event_df.get("key", pd.Series(dtype=object))
        .astype(str)
        .value_counts()
        .to_dict()
        if isinstance(event_df, pd.DataFrame) and not event_df.empty
        else {}
    )
    text = " | ".join(
        [
            f"TN: {int(counts.get('true_negative', 0))}",
            f"FP: {int(counts.get('false_positive', 0))}",
            f"FN: {int(counts.get('false_negative', 0))}",
            f"TP: {int(counts.get('true_positive', 0))}",
        ]
    )
    return f"<br><sup>Confusion windows: {html.escape(text)}</sup>"


def _restrict_interactive_review_legend(
    fig: go.Figure,
    *,
    keep_signal_name: str = "algorithmic_intermediate_channels",
) -> go.Figure:
    if fig is None:
        return fig
    keep_token = str(keep_signal_name or "").strip()
    any_kept = False
    for trace in getattr(fig, "data", []) or []:
        trace_name = str(getattr(trace, "name", "") or "")
        keep = bool(keep_token and keep_token in trace_name)
        try:
            trace.showlegend = keep
        except Exception:
            pass
        any_kept = any_kept or keep
    if not any_kept:
        fig.update_layout(showlegend=False)
    return fig


def _feature_source_channel(feature_name: str) -> str:
    return str(feature_name).split("__", 1)[0]


def _feature_display_name(feature_name: str) -> str:
    return str(feature_name).replace(".", " / ")


def _read_feature_subset(path: str, columns: List[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()


def _build_dataset_facet_feature_figure(
    df: pd.DataFrame,
    *,
    title: str,
    y_label: str,
    behavior_order: List[str],
    behavior_color_map: Dict[str, str],
    source_dataset_ids: List[str] | None = None,
) -> go.Figure | None:
    if df is None or df.empty:
        return None
    work = df.copy()
    source_set = {str(x).strip() for x in (source_dataset_ids or []) if str(x).strip()}
    work["dataset_role"] = np.where(
        work["dataset_id"].astype(str).isin(source_set),
        "source",
        "target",
    )
    work["dataset_display"] = work["dataset_id"].astype(str) + " [" + work["dataset_role"].astype(str) + "]"
    dataset_count = max(1, work["dataset_display"].nunique())
    fig = px.violin(
        work,
        x="observed_label",
        y="feature_value",
        color="observed_label",
        facet_col="dataset_display",
        facet_row="value_mode",
        box=True,
        points=False,
        title=title,
        labels={
            "observed_label": "Observed class",
            "feature_value": y_label,
            "dataset_display": "Dataset",
            "value_mode": "",
            "color": "Observed class",
        },
        category_orders={
            "observed_label": [x for x in behavior_order if x in set(work["observed_label"].astype(str))],
            "value_mode": ["Raw", "Z-scored"],
        },
        color_discrete_map=behavior_color_map,
        hover_data={
            "dataset_id": True,
            "dataset_role": True,
            "deployment_id": True if "deployment_id" in work.columns else False,
            "value_mode": True,
        },
    )
    fig.for_each_annotation(
        lambda ann: ann.update(
            text=str(ann.text)
            .replace("dataset_display=", "Dataset: ")
            .replace("value_mode=", "")
        )
    )
    fig.update_traces(spanmode="hard", hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    fig.update_layout(
        violinmode="group",
        legend_title_text="Observed class",
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(matches=None)
    _style_report_plot(fig, height=580)
    return fig


def _summarize_zscore_check(
    raw_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    *,
    item_name: str,
    representative_feature: str,
    review_type: str,
    norm_group_col: str,
) -> Dict[str, Any]:
    def _group_stats(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty or norm_group_col not in frame.columns:
            return pd.DataFrame()
        stats = (
            frame.groupby(norm_group_col, as_index=False)["feature_value"]
            .agg(
                mean=lambda s: float(pd.to_numeric(s, errors="coerce").mean()),
                std=lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0)),
                n="count",
            )
        )
        stats["std"] = pd.to_numeric(stats["std"], errors="coerce")
        return stats

    def _fmt_range(series: pd.Series) -> str:
        vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            return "n/a"
        return f"{float(vals.min()):.3f} to {float(vals.max()):.3f}"

    raw_stats = _group_stats(raw_df)
    norm_stats = _group_stats(norm_df)
    z_abs_mean = (
        float(pd.to_numeric(norm_stats.get("mean"), errors="coerce").abs().max())
        if not norm_stats.empty and "mean" in norm_stats.columns
        else float("nan")
    )
    z_std_vals = (
        pd.to_numeric(norm_stats.get("std"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not norm_stats.empty and "std" in norm_stats.columns
        else pd.Series(dtype=float)
    )
    z_std_min = float(z_std_vals.min()) if not z_std_vals.empty else float("nan")
    z_std_max = float(z_std_vals.max()) if not z_std_vals.empty else float("nan")
    pass_mean = np.isfinite(z_abs_mean) and z_abs_mean <= 0.05
    pass_std = np.isfinite(z_std_min) and np.isfinite(z_std_max) and z_std_min >= 0.95 and z_std_max <= 1.05
    status = "Pass" if (pass_mean and pass_std) else "Needs review"
    return {
        "review_type": review_type,
        "item_name": item_name,
        "representative_feature": representative_feature,
        "normalization_groups": int(raw_stats[norm_group_col].nunique()) if not raw_stats.empty else 0,
        "raw_mean_range": _fmt_range(raw_stats.get("mean", pd.Series(dtype=float))),
        "raw_std_range": _fmt_range(raw_stats.get("std", pd.Series(dtype=float))),
        "z_max_abs_mean": f"{z_abs_mean:.3f}" if np.isfinite(z_abs_mean) else "n/a",
        "z_std_range": _fmt_range(norm_stats.get("std", pd.Series(dtype=float))),
        "status": status,
    }


def _render_metrics_table_html(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]]) -> str:
    if not rows:
        return "<div class='empty-note'>No table rows available.</div>"
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key, _ in columns) + "</tr>"
        for row in rows
    )
    return f"<div class='metrics-table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _detect_supervised_negative_label(labels: List[str], positive_label: str | None) -> str | None:
    cleaned = [str(x).strip() for x in labels if str(x).strip()]
    if not cleaned:
        return None
    if positive_label is None:
        return None
    for label in cleaned:
        if label != positive_label:
            return label
    return None


def _observed_label_keep_mask(labels: pd.Series) -> pd.Series:
    if labels is None:
        return pd.Series(dtype=bool)
    text = labels.astype(str).str.strip()
    lowered = text.str.lower()
    keep = labels.notna() & text.ne("") & (~lowered.isin({"nan", "none", "unknown"}))
    keep &= ~lowered.str.contains("unscorable", na=False)
    return keep.fillna(False)


def _map_algorithmic_labels_to_supervised_classes(
    labels: pd.Series,
    positive_label: str | None,
    negative_label: str | None,
    algo_cfg: Dict,
) -> pd.Series:
    if labels is None:
        return pd.Series(dtype=object)
    out = labels.astype(str).copy()
    out = out.where(~out.str.strip().str.lower().isin({"", "nan", "none"}), np.nan)
    label_names = algo_cfg.get("label_names") or {}
    not_sleep_name = str(label_names.get("not_sleep") or "find_rest.not_sleep")
    active_surface_name = str(label_names.get("active_surface") or "find_rest.active_surface")
    calm_uw_gliding_name = str(label_names.get("calm_uw_gliding") or "calm_uw_gliding")
    active_uw_swimming_name = str(label_names.get("active_uw_swimming") or "active_uw_swimming")
    calm_surface_gliding_name = str(label_names.get("calm_surface_gliding") or "calm_surface_gliding")
    active_surface_swimming_name = str(label_names.get("active_surface_swimming") or "active_surface_swimming")
    sleep_like = {
        str(label_names.get("resting_surface") or label_names.get("surface_sleep") or "find_rest.surface_sleep"),
        str(label_names.get("resting_benthic") or label_names.get("long_flat") or "find_rest.long_flat"),
        str(label_names.get("long_drift") or "find_rest.long_drift"),
    }
    active_like = {
        not_sleep_name,
        active_surface_name,
        "active_diving",
        "active_surface",
        "find_rest.not_sleep",
        calm_uw_gliding_name,
        active_uw_swimming_name,
        calm_surface_gliding_name,
        active_surface_swimming_name,
        "calm_uw_gliding",
        "active_uw_swimming",
        "calm_surface_gliding",
        "active_surface_swimming",
    }
    unscorable_like = {
        str(label_names.get("unscorable") or "find_rest.unscorable"),
        "unscorable",
        "find_rest.unscorable",
    }
    if negative_label is None and positive_label is not None:
        lowered = str(positive_label).strip().lower()
        if lowered == "sleep":
            negative_label = "ACTIVE"
        elif lowered == "active":
            negative_label = "SLEEP"
    if negative_label is None:
        negative_label = "ACTIVE"
    if positive_label is None:
        positive_label = "SLEEP"
    mapped = pd.Series(np.nan, index=labels.index, dtype=object)
    out_text = out.astype(str)
    mapped.loc[out_text.isin(active_like)] = negative_label
    mapped.loc[out_text.isin(sleep_like)] = positive_label
    mapped.loc[out_text.isin(unscorable_like)] = np.nan
    return mapped


def _load_algorithmic_supervised_predictions(ctx: RunContext, window_df: pd.DataFrame, positive_label: str | None, negative_label: str | None) -> pd.Series:
    work = window_df[["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key"]].copy()
    algo_labels = None

    clustered_path = os.path.join(ctx.output_root, "clustering", "clustered_windows.parquet")
    if os.path.exists(clustered_path):
        try:
            clustered = pd.read_parquet(clustered_path)
            if "algorithmic_label_name" in clustered.columns:
                clustered = clustered[
                    [c for c in ["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "algorithmic_label_name"] if c in clustered.columns]
                ].copy()
                clustered["window_key"] = _build_window_key_series(clustered)
                algo_labels = (
                    clustered[["window_key", "algorithmic_label_name"]]
                    .drop_duplicates(subset=["window_key"], keep="last")
                    .rename(columns={"algorithmic_label_name": "__algorithmic_label_name"})
                )
        except Exception:
            algo_labels = None

    if algo_labels is None:
        detailed_path, _ = _algorithmic_segment_merge_paths(ctx)
        if os.path.exists(detailed_path):
            try:
                algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
                seg_df = pd.read_parquet(detailed_path)
                annotated = _annotate_windows_with_algorithmic_labels(
                    work.drop(columns=["window_key"]),
                    seg_df,
                    default_code=int((algo_cfg.get("label_codes") or {}).get("not_sleep", 0)),
                    default_name=str((algo_cfg.get("label_names") or {}).get("not_sleep", "find_rest.not_sleep")),
                )
                annotated["window_key"] = _build_window_key_series(annotated)
                algo_labels = (
                    annotated[["window_key", "algorithmic_label_name"]]
                    .drop_duplicates(subset=["window_key"], keep="last")
                    .rename(columns={"algorithmic_label_name": "__algorithmic_label_name"})
                )
            except Exception:
                algo_labels = None

    merged = work[["window_key"]].copy()
    if algo_labels is not None:
        merged = merged.merge(algo_labels, on="window_key", how="left")
    else:
        merged["__algorithmic_label_name"] = np.nan
    algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
    return _map_algorithmic_labels_to_supervised_classes(
        merged["__algorithmic_label_name"],
        positive_label=positive_label,
        negative_label=negative_label,
        algo_cfg=algo_cfg,
    )


def _load_algorithmic_supervised_predictions_from_segment_df(
    ctx: RunContext,
    window_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    positive_label: str | None,
    negative_label: str | None,
    class_column: str = "nominal_class",
) -> pd.Series:
    work = window_df[["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key"]].copy()
    algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
    seg_df = segment_df.copy() if segment_df is not None else pd.DataFrame()
    if seg_df.empty or class_column not in seg_df.columns:
        return pd.Series(np.nan, index=work.index, dtype=object)

    label_codes = algo_cfg.get("label_codes") or {}
    label_names = algo_cfg.get("label_names") or {}
    seg_df["label_name"] = seg_df[class_column].map(label_names).fillna(str(label_names.get("not_sleep", "find_rest.not_sleep")))
    seg_df["label_code"] = pd.to_numeric(seg_df[class_column].map(label_codes), errors="coerce").fillna(int(label_codes.get("not_sleep", 0))).astype(int)
    annotated = _annotate_windows_with_algorithmic_labels(
        work.drop(columns=["window_key"]),
        seg_df,
        default_code=int(label_codes.get("not_sleep", 0)),
        default_name=str(label_names.get("not_sleep", "find_rest.not_sleep")),
    )
    annotated["window_key"] = _build_window_key_series(annotated)
    merged = work[["window_key"]].merge(
        annotated[["window_key", "algorithmic_label_name"]].rename(columns={"algorithmic_label_name": "__algorithmic_label_name"}),
        on="window_key",
        how="left",
    )
    return _map_algorithmic_labels_to_supervised_classes(
        merged["__algorithmic_label_name"],
        positive_label=positive_label,
        negative_label=negative_label,
        algo_cfg=algo_cfg,
    )


def _substitute_filter_placeholders(obj: Any, positive_label: str | None, negative_label: str | None):
    if isinstance(obj, dict):
        return {k: _substitute_filter_placeholders(v, positive_label, negative_label) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_filter_placeholders(v, positive_label, negative_label) for v in obj]
    if isinstance(obj, str):
        if obj == "__positive_label__":
            return str(positive_label or "")
        if obj == "__negative_label__":
            return str(negative_label or "")
    return obj


def _parse_supervised_context_filters(run_cfg: Dict) -> List[Dict[str, Any]]:
    supervised_cfg = dict(run_cfg.get("supervised") or {})
    definitions = dict(
        run_cfg.get("__supervised_context_filter_definitions_merged__")
        or run_cfg.get("supervised_context_filter_definitions")
        or {}
    )
    return _resolve_named_filter_items(definitions, list(supervised_cfg.get("context_filters") or []))


def _apply_supervised_context_filters(
    ctx: RunContext,
    prediction_df: pd.DataFrame,
    positive_label: str | None,
    negative_label: str | None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    work = prediction_df.copy()
    filter_cfgs = _parse_supervised_context_filters(ctx.run_cfg)
    if work.empty or not filter_cfgs:
        return work, {"enabled": False, "status": "disabled", "n_rejected": 0}

    field_names = {
        "context_keep",
        "context_reject_reason",
        "context_filter_ids",
        "segment_mean_stroke_rate_spm",
        "inferred_trip_phase",
        "inferred_buoyancy_phase",
    }
    detailed_path, _ = _algorithmic_segment_merge_paths(ctx)
    if os.path.exists(detailed_path):
        try:
            seg_df = pd.read_parquet(detailed_path)
            work = _annotate_windows_with_algorithmic_segment_fields(work, seg_df, sorted(field_names), prefix="algorithmic_")
        except Exception:
            pass

    work["predicted_label_pre_context"] = work.get("predicted_label", pd.Series(index=work.index, dtype=object)).astype(object)
    work["rf_context_keep"] = True
    work["rf_context_filter_applied"] = False
    work["rf_context_filter_ids"] = ""
    work["rf_context_reject_reason"] = pd.NA

    resolved_filters = [_substitute_filter_placeholders(cfg, positive_label, negative_label) for cfg in filter_cfgs]
    for raw_filter in resolved_filters:
        filter_cfg = dict(raw_filter or {})
        if not bool(filter_cfg.get("enabled", False)):
            continue
        filter_id = str(filter_cfg.get("id") or f"rf_context_filter_{len(work.columns)}").strip()
        rule = dict(filter_cfg.get("rule") or {})
        mode = str(rule.get("mode") or "").strip().lower()
        action_on_fail = str(filter_cfg.get("action_on_fail") or "reject").strip().lower()
        if action_on_fail != "reject":
            raise ValueError(f"Supervised context filter '{filter_id}' uses unsupported action_on_fail '{action_on_fail}'")
        mask = work["rf_context_keep"].fillna(True)
        mask &= _filter_applies_mask(work, dict(filter_cfg.get("applies_when") or {}))
        if not mask.any():
            continue
        value_series = _resolve_filter_value(work, filter_cfg, {})
        on_missing = str(filter_cfg.get("on_missing_covariate") or "reject").strip().lower()
        if mode == "threshold_gate":
            keep_if = dict(rule.get("keep_if") or {})
            pass_mask = _compare_series(value_series, keep_if.get("comparator"), keep_if.get("threshold"))
        elif mode == "category_gate":
            allowed = {str(x) for x in (rule.get("allowed_values") or rule.get("keep_if_in") or [])}
            pass_mask = value_series.astype(str).isin(allowed)
        elif mode == "phase_gate":
            allowed = {str(x) for x in (rule.get("allowed_phases") or rule.get("keep_if_in") or [])}
            pass_mask = value_series.astype(str).isin(allowed)
        else:
            raise ValueError(f"Unsupported supervised context filter rule mode '{mode}' for '{filter_id}'")
        missing_mask = value_series.isna()
        if on_missing == "keep":
            pass_mask = pass_mask | missing_mask
        elif on_missing == "skip_filter":
            pass_mask = pass_mask | missing_mask
            mask &= ~missing_mask
        elif on_missing != "reject":
            raise ValueError(f"Unsupported on_missing_covariate '{on_missing}' for supervised filter '{filter_id}'")
        fail_mask = mask & (~pass_mask.fillna(False))
        if not fail_mask.any():
            continue
        fail_idx = work.index[fail_mask]
        work.loc[mask, "rf_context_filter_applied"] = True
        existing_ids = work.loc[fail_idx, "rf_context_filter_ids"].fillna("").astype(str)
        work.loc[fail_idx, "rf_context_filter_ids"] = existing_ids.map(lambda s: filter_id if not s else f"{s},{filter_id}")
        work.loc[fail_idx, "rf_context_keep"] = False
        reject_reason = str(filter_cfg.get("reject_reason") or filter_id)
        empty_reason = work.loc[fail_idx, "rf_context_reject_reason"].isna()
        if empty_reason.any():
            work.loc[fail_idx[empty_reason], "rf_context_reject_reason"] = reject_reason

    replacement_label = str(negative_label or "Unknown")
    work["predicted_label_context_filtered"] = work["predicted_label_pre_context"]
    reject_positive = (
        work["predicted_label_pre_context"].astype(str) == str(positive_label or "")
    ) & (~work["rf_context_keep"].fillna(True))
    work.loc[reject_positive, "predicted_label_context_filtered"] = replacement_label
    work["predicted_label"] = work["predicted_label_context_filtered"]
    work["final_behavior"] = np.where(
        work["observed_label"].notna() & (work["observed_label"].astype(str) != "Unknown"),
        work["observed_label"],
        work["predicted_label"],
    )
    summary = {
        "enabled": True,
        "status": "ok",
        "n_rows": int(len(work)),
        "n_rejected": int(reject_positive.sum()),
        "filter_ids": [str(cfg.get("id") or "") for cfg in resolved_filters],
    }
    return work, summary


def _evaluate_supervised_variant(
    y_true: pd.Series,
    y_pred: pd.Series,
    label_order: List[str],
    variant_id: str,
    variant_label: str,
    variant_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rep = classification_report(y_true, y_pred, labels=label_order, output_dict=True, zero_division=0)
    class_metrics_df = pd.DataFrame(rep).T.rename_axis("label").reset_index()
    class_metrics_df["variant_id"] = variant_id
    class_metrics_df["variant_label"] = variant_label
    class_metrics_df["variant_type"] = variant_type

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_long = (
        pd.DataFrame(cm, index=label_order, columns=label_order)
        .rename_axis("observed_label")
        .reset_index()
        .melt(id_vars="observed_label", var_name="predicted_label", value_name="count")
    )
    cm_long["variant_id"] = variant_id
    cm_long["variant_label"] = variant_label
    cm_long["variant_type"] = variant_type

    metric_rows = []
    for row in class_metrics_df[class_metrics_df["label"].isin(label_order)].to_dict(orient="records"):
        for metric_name in ["precision", "recall", "f1-score"]:
            if metric_name in row:
                metric_rows.append(
                    {
                        "variant_id": variant_id,
                        "variant_label": variant_label,
                        "variant_type": variant_type,
                        "label": str(row["label"]),
                        "metric": metric_name,
                        "value": float(row[metric_name]),
                        "support": float(row.get("support", np.nan)),
                    }
                )
    metric_rows.extend(
        [
            {
                "variant_id": variant_id,
                "variant_label": variant_label,
                "variant_type": variant_type,
                "label": "Overall",
                "metric": "accuracy",
                "value": float(accuracy_score(y_true, y_pred)),
                "support": float(len(y_true)),
            },
            {
                "variant_id": variant_id,
                "variant_label": variant_label,
                "variant_type": variant_type,
                "label": "Overall",
                "metric": "balanced_accuracy",
                "value": float(balanced_accuracy_score(y_true, y_pred)),
                "support": float(len(y_true)),
            },
        ]
    )
    metrics_long = pd.DataFrame(metric_rows)
    return class_metrics_df, cm_long, metrics_long


def _build_supervised_estimator(supervised_cfg: Dict, variant_type: str):
    rf_like = {"random_forest", "random_forest_ablation"}
    if variant_type in rf_like:
        return RandomForestClassifier(
            n_estimators=int(supervised_cfg.get("rf_n_estimators", 300)),
            random_state=42,
            min_samples_leaf=int(supervised_cfg.get("rf_min_samples_leaf", 1)),
            class_weight=supervised_cfg.get("rf_class_weight", "balanced_subsample"),
            n_jobs=-1,
        )
    if variant_type == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("lightgbm is not installed")
        return LGBMClassifier(
            n_estimators=int(supervised_cfg.get("lightgbm_n_estimators", 300)),
            learning_rate=float(supervised_cfg.get("lightgbm_learning_rate", 0.05)),
            num_leaves=int(supervised_cfg.get("lightgbm_num_leaves", 31)),
            random_state=42,
            class_weight=supervised_cfg.get("lightgbm_class_weight", "balanced"),
            n_jobs=-1,
            verbose=-1,
        )
    if variant_type == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel=str(supervised_cfg.get("svm_kernel", "rbf")),
                C=float(supervised_cfg.get("svm_c", 1.0)),
                gamma=str(supervised_cfg.get("svm_gamma", "scale")),
                probability=True,
                class_weight=supervised_cfg.get("svm_class_weight", "balanced"),
                random_state=42,
            ),
        )
    if variant_type == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=int(supervised_cfg.get("knn_n_neighbors", 25)),
                weights=str(supervised_cfg.get("knn_weights", "distance")),
            ),
        )
    raise ValueError(f"Unsupported supervised variant type: {variant_type}")


def _write_supervised_actogram_comparison_plots(ctx: RunContext, spdf: pd.DataFrame, out_dir: str) -> List[str]:
    required = {"deployment_id", "window_start", "window_end", "observed_label"}
    if not required.issubset(spdf.columns):
        return []

    plot_df = _localize_window_columns_by_deployment(ctx, spdf.copy())
    if "variant_label" not in plot_df.columns:
        predicted_col = next((c for c in ["predicted_label", "predicted_label_raw", "y_pred"] if c in plot_df.columns), None)
        if predicted_col is None:
            return []
        plot_df["variant_id"] = "rf_primary"
        plot_df["variant_label"] = "RF"
        plot_df["predicted_label"] = plot_df[predicted_col]
        plot_df["variant_rank"] = 0
    else:
        if "predicted_label" not in plot_df.columns:
            predicted_col = next((c for c in ["predicted_label_raw", "y_pred"] if c in plot_df.columns), None)
            if predicted_col is None:
                return []
            plot_df["predicted_label"] = plot_df[predicted_col]
        if "variant_rank" not in plot_df.columns:
            rank_map = {label: i for i, label in enumerate(pd.Index(plot_df["variant_label"]).dropna().astype(str).unique().tolist())}
            plot_df["variant_rank"] = plot_df["variant_label"].astype(str).map(rank_map).fillna(999).astype(int)

    plot_df["window_start"] = pd.to_datetime(plot_df["window_start"], errors="coerce")
    plot_df["window_end"] = pd.to_datetime(plot_df["window_end"], errors="coerce")
    plot_df = plot_df.dropna(subset=["deployment_id", "window_start"]).copy()
    if plot_df.empty:
        return []

    # Keep supervised review actograms on strict 30 s epochs so rapid REM/SWS
    # transitions are not smeared by gap-aware effective-duration expansion.
    plot_df["window_seconds"] = 30.0

    plot_df["observed_label"] = plot_df["observed_label"].astype(str)
    plot_df["predicted_label"] = plot_df["predicted_label"].astype(str)
    plot_df["variant_label"] = plot_df["variant_label"].astype(str)
    plot_df = plot_df[
        plot_df["observed_label"].str.strip().ne("")
        & plot_df["predicted_label"].str.strip().ne("")
        & (~plot_df["observed_label"].str.lower().isin({"nan", "none"}))
        & (~plot_df["predicted_label"].str.lower().isin({"nan", "none"}))
    ].copy()
    if plot_df.empty:
        return []

    if pd.api.types.is_datetime64tz_dtype(plot_df["window_start"]):
        local_start = plot_df["window_start"].dt.tz_localize(None)
    else:
        local_start = plot_df["window_start"]
    plot_df["date_local"] = local_start.dt.floor("D")
    plot_df["hour_of_day"] = local_start.dt.hour

    behavior_order = (
        pd.Index(plot_df["observed_label"].tolist())
        .value_counts()
        .index
        .tolist()
    )
    import matplotlib.pyplot as plt
    repo_colors = _load_repo_color_mapping()
    cmap = plt.get_cmap("tab20")
    n_colors = max(len(behavior_order), 3)
    behavior_colors = {}
    for i, label in enumerate(behavior_order):
        lookup_chain = [
            label,
            label.strip(),
            label.replace("_", " "),
            label.replace("-", " "),
            label.title(),
            label.upper(),
        ]
        color = next((repo_colors.get(k) for k in lookup_chain if k in repo_colors), None)
        if not color:
            color = cmap(i % getattr(cmap, "N", n_colors))
        behavior_colors[label] = color
    positive_label = _detect_supervised_positive_label(behavior_order)
    confusion_styles = _supervised_confusion_style_map()
    confusion_order = ["true_negative", "false_positive", "false_negative", "true_positive"]

    generated = []
    for dep_id, dep_df in plot_df.groupby("deployment_id", sort=True):
        date_order = sorted(dep_df["date_local"].dropna().unique().tolist())
        if not date_order:
            continue

        obs_cols = [c for c in ["window_key", "window_start", "window_end", "date_local", "hour_of_day", "window_seconds", "observed_label"] if c in dep_df.columns]
        obs_df = dep_df[obs_cols].copy()
        if "window_key" in obs_df.columns:
            obs_df = obs_df.drop_duplicates(subset=["window_key"]).copy()
        else:
            obs_df = obs_df.drop_duplicates(subset=[c for c in ["window_start", "window_end", "observed_label"] if c in obs_df.columns]).copy()
        obs_df["state_label"] = obs_df["observed_label"].astype(str)
        hourly_state = (
            obs_df.groupby(["date_local", "hour_of_day", "state_label"], as_index=False)["window_seconds"]
            .sum()
        )
        hourly_state["hour_fraction"] = hourly_state["window_seconds"] / 3600.0
        hourly_complete = (
            hourly_state
            .pivot_table(
                index=["date_local", "hour_of_day"],
                columns="state_label",
                values="hour_fraction",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )

        variant_order = (
            dep_df[["variant_label", "variant_rank"]]
            .drop_duplicates()
            .sort_values(["variant_rank", "variant_label"])
            ["variant_label"]
            .tolist()
        )
        confusion_by_variant = {}
        if positive_label is not None:
            positive_lower = str(positive_label).strip().lower()
            for variant_label in variant_order:
                pred_variant = dep_df[dep_df["variant_label"] == variant_label].copy()
                observed_is_positive = pred_variant["observed_label"].str.strip().str.lower() == positive_lower
                predicted_is_positive = pred_variant["predicted_label"].str.strip().str.lower() == positive_lower
                pred_variant["confusion_state"] = np.where(
                    (~observed_is_positive) & (~predicted_is_positive),
                    "true_negative",
                    np.where(
                        (~observed_is_positive) & predicted_is_positive,
                        "false_positive",
                        np.where(
                            observed_is_positive & (~predicted_is_positive),
                            "false_negative",
                            "true_positive",
                        ),
                    ),
                )
                hourly_confusion = (
                    pred_variant.groupby(["date_local", "hour_of_day", "confusion_state"], as_index=False)["window_seconds"]
                    .sum()
                )
                hourly_confusion["hour_fraction"] = hourly_confusion["window_seconds"] / 3600.0
                confusion_by_variant[variant_label] = hourly_confusion

        from matplotlib.patches import Patch

        fig_height_per_row = max(4, 0.35 * len(date_order) + 1.5)
        n_panels = 1 + len(variant_order)
        n_cols = min(4, max(1, n_panels))
        n_rows = int(np.ceil(n_panels / n_cols))
        fig_width = max(12, 4.2 * n_cols)
        fig_height = max(4, fig_height_per_row * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)

        for idx, ax in enumerate(axes[:n_panels]):
            if idx == 0:
                variant = "Observed"
                variant_df = hourly_complete.copy()
            else:
                variant = variant_order[idx - 1]
                variant_df = confusion_by_variant.get(variant, pd.DataFrame()).copy()
            if variant_df.empty:
                ax.text(0.5, 0.5, f"No rows for {variant}", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            for row_idx, date_val in enumerate(date_order):
                if idx == 0:
                    day_df = variant_df[variant_df["date_local"] == date_val].copy()
                    day_df = day_df.set_index("hour_of_day").reindex(range(24), fill_value=0).reset_index()
                    baseline = np.full(len(day_df), row_idx, dtype=float)
                    cumulative = baseline.copy()
                    for state in behavior_order:
                        heights = day_df[state].to_numpy(dtype=float) if state in day_df.columns else np.zeros(len(day_df), dtype=float)
                        ax.bar(
                            day_df["hour_of_day"],
                            heights,
                            width=1.0,
                            align="edge",
                            bottom=cumulative,
                            color=behavior_colors[state],
                            edgecolor="none",
                        )
                        cumulative = cumulative + heights
                else:
                    day_df = variant_df[variant_df["date_local"] == date_val].copy()
                    day_df = (
                        day_df.pivot_table(
                            index="hour_of_day",
                            columns="confusion_state",
                            values="hour_fraction",
                            aggfunc="sum",
                            fill_value=0,
                        )
                        .reindex(index=range(24), fill_value=0)
                        .reset_index()
                    )
                    baseline = np.full(len(day_df), row_idx, dtype=float)
                    cumulative = baseline.copy()
                    for confusion_key in confusion_order:
                        heights = day_df[confusion_key].to_numpy(dtype=float) if confusion_key in day_df.columns else np.zeros(len(day_df), dtype=float)
                        ax.bar(
                            day_df["hour_of_day"],
                            heights,
                            width=1.0,
                            align="edge",
                            bottom=cumulative,
                            color=confusion_styles[confusion_key]["color"],
                            edgecolor="none",
                        )
                        cumulative = cumulative + heights

            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 2))
            if idx // n_cols == (n_rows - 1):
                ax.set_xlabel("Local hour of day")
            else:
                ax.set_xlabel("")
            ax.set_title("Observed" if idx == 0 else variant)
            ax.grid(axis="x", alpha=0.2)
            ax.set_ylim(0, len(date_order) + 0.1)
            ax.invert_yaxis()

        for idx, ax in enumerate(axes[:n_panels]):
            if idx % n_cols == 0:
                ax.set_ylabel("Date")
                ax.set_yticks(np.arange(len(date_order)) + 0.5)
                ax.set_yticklabels([pd.Timestamp(d).strftime("%Y-%m-%d") for d in date_order], fontsize=8)
            else:
                ax.set_ylabel("")
                ax.set_yticks(np.arange(len(date_order)) + 0.5)
                ax.set_yticklabels([])
        for ax in axes[n_panels:]:
            ax.set_axis_off()
        legend_handles = [Patch(facecolor=behavior_colors[state], label=state) for state in behavior_order]
        if confusion_by_variant:
            legend_handles.extend(
                [
                    Patch(
                        facecolor=confusion_styles[key]["color"],
                        label=_supervised_confusion_legend_label(key, positive_label),
                    )
                    for key in confusion_order
                ]
            )
        axes[n_panels - 1].legend(handles=legend_handles, title="Observed / confusion state", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        plot_path = os.path.join(out_dir, f"G_supervised_actogram_comparison_{_safe_slug(dep_id)}.png")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated.append(plot_path)

    return generated


def _relative_interval_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or min(abs(a), abs(b)) == 0:
        return float("inf")
    return abs(a - b) / min(abs(a), abs(b))


def _gaussian_smooth_series(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if not np.isfinite(sigma_samples) or sigma_samples <= 0:
        return arr.copy()

    radius = max(1, int(np.ceil(float(sigma_samples) * 3.0)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / float(sigma_samples)) ** 2)
    kernel = kernel / kernel.sum()

    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    filled_pad = np.pad(filled, (radius, radius), mode="edge")
    valid_pad = np.pad(valid.astype(float), (radius, radius), mode="edge")
    numer = np.convolve(filled_pad, kernel, mode="valid")
    denom = np.convolve(valid_pad, kernel, mode="valid")
    out = np.full(arr.shape, np.nan, dtype=float)
    keep = denom > 0
    out[keep] = numer[keep] / denom[keep]
    return out


def _infer_sampling_interval_seconds(dt: pd.Series) -> float:
    if not isinstance(dt, pd.Series):
        dt = pd.Series(dt)
    delta_s = dt.sort_values().diff().dt.total_seconds()
    delta_s = delta_s[np.isfinite(delta_s) & (delta_s > 0)]
    if delta_s.empty:
        return float("nan")
    return float(delta_s.median())


def _infer_channel_resolution(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    diffs = np.abs(np.diff(arr))
    diffs = np.round(diffs[np.isfinite(diffs) & (diffs > 0)], 2)
    if diffs.size == 0:
        return float("nan")
    unique_small = np.sort(np.unique(diffs))[:10]
    if unique_small.size == 0:
        return float("nan")
    if unique_small.size == 1:
        return float(unique_small[0])
    return float(unique_small[1])


def _preprocessing_root(ctx: RunContext) -> str:
    return os.path.join(ctx.output_root, "preprocessing")


def _preprocessing_diag_paths(ctx: RunContext, dataset_id: str, deployment_id: str) -> Tuple[str, str]:
    safe_ds = _safe_slug(dataset_id)
    safe_dep = _safe_slug(deployment_id)
    base = _preprocessing_root(ctx)
    return (
        os.path.join(base, "diagnostics", f"{safe_ds}__{safe_dep}__standardization.json"),
        os.path.join(base, "plots", f"{safe_ds}__{safe_dep}__standardization.html"),
    )


def _flatten_scope(scope: List[Dict]) -> str:
    return ";".join(f"{x['dataset_id']}|{x['deployment_id']}" for x in scope)


def _clustering_scope_pairs(ctx: RunContext) -> List[Tuple[str, str]]:
    """Extract (dataset_id, deployment_id) pairs from the run context scope."""
    return [(item["dataset_id"], item["deployment_id"]) for item in ctx.scope]


def _resolve_run_context(config_path: str, run_name: str) -> RunContext:
    cfg = _load_yaml(config_path)
    run_cfg = (cfg.get("segmentation_runs") or {}).get(run_name)
    if not run_cfg:
        raise ValueError(f"segmentation_runs.{run_name} not found in {config_path}")
    run_cfg = normalize_segmentation_run_cfg(
        dict(run_cfg),
        shared_supervised_label_groups=dict(cfg.get("supervised_label_group_definitions") or {}),
    )

    global_context_defs = dict(cfg.get("context_filter_definitions") or {})
    run_context_defs = dict(run_cfg.get("context_filter_definitions") or {})
    merged_context_defs = dict(global_context_defs)
    merged_context_defs.update(run_context_defs)
    run_cfg["__context_filter_definitions_merged__"] = merged_context_defs

    global_supervised_defs = dict(cfg.get("supervised_context_filter_definitions") or {})
    run_supervised_defs = dict(run_cfg.get("supervised_context_filter_definitions") or {})
    merged_supervised_defs = dict(global_supervised_defs)
    merged_supervised_defs.update(run_supervised_defs)
    run_cfg["__supervised_context_filter_definitions_merged__"] = merged_supervised_defs
    run_cfg["__segmentation_event_definitions_merged__"] = _merge_segmentation_event_definitions(
        global_defs=dict(cfg.get("segmentation_event_definitions") or {}),
        run_defs=dict(run_cfg.get("segmentation_event_definitions") or {}),
        run_name=run_name,
        analysis_id=str(run_cfg.get("analysis_id") or run_name),
    )

    data_root = cfg["paths"]["local_private_data"]
    meta_analysis_root = cfg["paths"].get("local_private_meta_analysis_data")
    dataset_ids = list(run_cfg.get("dataset_ids") or [])
    deployment_ids_filter = set(run_cfg.get("deployment_ids") or [])
    if not dataset_ids:
        raise ValueError(f"Run '{run_name}' must define dataset_ids")

    scope = []
    for ds in dataset_ids:
        ds_path = os.path.join(data_root, ds)
        if not os.path.isdir(ds_path):
            continue
        deployments = sorted(
            d for d in os.listdir(ds_path)
            if os.path.isdir(os.path.join(ds_path, d)) and not d.startswith("00_")
        )
        if deployment_ids_filter:
            deployments = [d for d in deployments if d in deployment_ids_filter]
        for dep in deployments:
            scope.append({"dataset_id": ds, "deployment_id": dep})

    if not scope:
        raise ValueError(f"Run '{run_name}' has empty resolved scope")

    analysis_id = run_cfg.get("analysis_id") or _safe_slug(run_name)

    override = run_cfg.get("output_root_override")
    if override:
        output_root = os.path.expanduser(os.path.expandvars(override))
    elif meta_analysis_root:
        if len(dataset_ids) == 1:
            output_root = os.path.join(
                meta_analysis_root,
                dataset_ids[0],
                "segmentation",
                analysis_id,
            )
        else:
            output_root = os.path.join(
                meta_analysis_root,
                "segmentation",
                analysis_id,
            )
    else:
        if len(dataset_ids) == 1:
            output_root = os.path.join(
                data_root,
                dataset_ids[0],
                "00_Meta-Analysis",
                "segmentation",
                analysis_id,
            )
        else:
            output_root = os.path.join(
                data_root,
                "00_Meta-Analysis",
                "segmentation",
                analysis_id,
            )

    _ensure_dir(output_root)

    return RunContext(
        run_name=run_name,
        run_cfg=run_cfg,
        global_cfg=cfg,
        data_root=data_root,
        output_root=output_root,
        analysis_id=analysis_id,
        scope=scope,
    )


def _data_pkl_path(ctx: RunContext, dataset_id: str, deployment_id: str) -> str:
    return os.path.join(ctx.data_root, dataset_id, deployment_id, "outputs", "data.pkl")


def _load_data_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge_dicts(dict(out.get(key) or {}), dict(value or {}))
        else:
            out[key] = value
    return out


def _merge_segmentation_event_definitions(
    global_defs: Dict[str, Any],
    run_defs: Dict[str, Any],
    run_name: str,
    analysis_id: str,
) -> Dict[str, Any]:
    global_defs = dict(global_defs or {})
    run_defs = dict(run_defs or {})

    global_base = dict(global_defs.get("base") or {})
    global_overrides = dict(global_defs.get("run_overrides") or {})
    global_run = dict(global_overrides.get(run_name) or global_overrides.get(analysis_id) or {})

    if "base" in run_defs or "run_overrides" in run_defs:
        run_base = dict(run_defs.get("base") or {})
        run_overrides = dict(run_defs.get("run_overrides") or {})
        run_run = dict(run_overrides.get(run_name) or run_overrides.get(analysis_id) or {})
    else:
        run_base = dict(run_defs or {})
        run_run = {}

    merged = _deep_merge_dicts(global_base, global_run)
    merged = _deep_merge_dicts(merged, run_base)
    merged = _deep_merge_dicts(merged, run_run)
    return merged


def _render_event_text(template: Any, context: Dict[str, Any], default: str = "") -> str:
    text = str(template or "").strip()
    if not text:
        return str(default or "")
    try:
        return text.format_map(ChainMap(context, {"event_key": "", "label_name": "", "variant_id": ""}))
    except Exception:
        return text


def _resolve_segmentation_event_metadata(
    ctx: RunContext,
    step_name: str,
    event_key: str,
    label_name: str | None = None,
    variant_id: str | None = None,
    row_context: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    catalog = dict(ctx.run_cfg.get("__segmentation_event_definitions_merged__") or {})
    step_cfg = dict(catalog.get(step_name) or {})
    defaults = dict(step_cfg.get("defaults") or {})
    event_map = dict(step_cfg.get("event_keys") or {})
    label_map = dict(step_cfg.get("labels") or {})

    variants = dict(step_cfg.get("variants") or {})
    if variant_id is not None:
        variant_cfg = dict(variants.get(str(variant_id)) or variants.get(_normalize_label_name(str(variant_id))) or {})
        defaults = _deep_merge_dicts(defaults, dict(variant_cfg.get("defaults") or {}))
        event_map = _deep_merge_dicts(event_map, dict(variant_cfg.get("event_keys") or {}))
        label_map = _deep_merge_dicts(label_map, dict(variant_cfg.get("labels") or {}))

    raw_key = str(event_key or "").strip()
    normalized_key = _normalize_label_name(raw_key)
    raw_label = str(label_name or "").strip()
    normalized_label = _normalize_label_name(raw_label)
    context = dict(row_context or {})
    context.update(
        {
            "event_key": raw_key,
            "event_key_normalized": normalized_key,
            "label_name": raw_label,
            "label_name_normalized": normalized_label,
            "variant_id": str(variant_id or ""),
        }
    )

    resolved = dict(defaults)
    for candidate in [raw_key, normalized_key]:
        if candidate in event_map and isinstance(event_map[candidate], dict):
            resolved = _deep_merge_dicts(resolved, dict(event_map[candidate] or {}))
            break
    if raw_label:
        for candidate in [raw_label, normalized_label]:
            if candidate in label_map and isinstance(label_map[candidate], dict):
                resolved = _deep_merge_dicts(resolved, dict(label_map[candidate] or {}))
                break

    short_default = raw_label or raw_key
    return {
        "type": str(resolved.get("type") or "state"),
        "short_description": _render_event_text(resolved.get("short_description"), context, default=short_default),
        "long_description": _render_event_text(resolved.get("long_description"), context, default=""),
    }


def _feature_sets() -> Dict[str, List[str]]:
    base_min = ["mean", "std"]
    base_med = [
        "mean", "median", "min", "max", "range", "iqr", "skew", "kurtosis", "cv",
        "mad", "mean_abs_diff", "var_diff", "rms", "zero_cross_rate",
    ]
    return {
        "minimal": base_min,
        "medium": base_med,
        "full": base_med,
    }


def _catch22_names() -> List[str]:
    global _CATCH22_NAMES_CACHE
    if _CATCH22_NAMES_CACHE is not None:
        return _CATCH22_NAMES_CACHE
    if not CATCH22_AVAILABLE:
        _CATCH22_NAMES_CACHE = []
        return _CATCH22_NAMES_CACHE
    try:
        _CATCH22_NAMES_CACHE = list(pycatch22.catch22_all(np.arange(60, dtype=float)).get("names", []))
    except Exception:
        _CATCH22_NAMES_CACHE = []
    return _CATCH22_NAMES_CACHE


def _compute_stat_features(x: np.ndarray, feature_level: str) -> Dict[str, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}

    diff = np.diff(arr) if arr.size > 1 else np.array([])
    q75, q25 = np.percentile(arr, [75, 25])
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    out = {
        "mean": mean,
        "std": std,
    }
    if feature_level in {"medium", "full"}:
        out.update({
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "iqr": float(q75 - q25),
            "skew": float(pd.Series(arr).skew()),
            "kurtosis": float(pd.Series(arr).kurtosis()),
            "cv": float(std / mean) if mean not in (0.0, -0.0) else np.nan,
            "mad": float(np.median(np.abs(arr - np.median(arr)))),
            "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else np.nan,
            "var_diff": float(np.var(diff)) if diff.size else np.nan,
            "rms": float(np.sqrt(np.mean(arr ** 2))),
            "zero_cross_rate": float(np.mean(np.diff(np.signbit(arr)) != 0)) if arr.size > 1 else np.nan,
        })

    if feature_level == "full":
        if CATCH22_AVAILABLE:
            # Keep catch22 columns stable for all full-feature channels/transforms.
            # If catch22 fails for a given window, columns remain present as NaN.
            c22_names = _catch22_names()
            for n in c22_names:
                out[f"c22_{n}"] = np.nan

            try:
                c22 = pycatch22.catch22_all(arr)
                for n, v in zip(c22.get("names", []), c22.get("values", [])):
                    out[f"c22_{n}"] = float(v)
            except Exception:
                pass

    return out


def _apply_transform(series: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "raw":
        return series
    if transform_name == "diff":
        if len(series) < 2:
            return np.array([], dtype=float)
        return np.diff(series)
    if transform_name == "diff2":
        if len(series) < 3:
            return np.array([], dtype=float)
        return np.diff(series, n=2)
    if transform_name == "abs_diff":
        if len(series) < 2:
            return np.array([], dtype=float)
        return np.abs(np.diff(series))
    raise ValueError(f"Unsupported transform: {transform_name}")


def _iter_windows(df: pd.DataFrame, mode: str, run_cfg: Dict):
    # expects datetime sorted and tz-naive/aware consistently
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    sdf = df.copy()
    sdf["datetime"] = dt
    sdf = sdf.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if sdf.empty:
        return

    if mode == "fixed":
        duration_s = int(run_cfg.get("duration_s", 60))
        freq = f"{duration_s}s"
        sdf["__chunk_start"] = sdf["datetime"].dt.floor(freq)
        for chunk_start, g in sdf.groupby("__chunk_start"):
            if g.empty:
                continue
            yield chunk_start, g["datetime"].iloc[-1], g
    else:
        window_s = int(run_cfg.get("window_s", 60))
        stride_s = int(run_cfg.get("stride_s", 10))
        start = sdf["datetime"].iloc[0]
        end = sdf["datetime"].iloc[-1]
        cur = start
        while cur <= end:
            nxt = cur + pd.Timedelta(seconds=window_s)
            g = sdf[(sdf["datetime"] >= cur) & (sdf["datetime"] < nxt)]
            if not g.empty:
                yield cur, nxt, g
            cur += pd.Timedelta(seconds=stride_s)


def cmd_resolve(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    scope_hash = hashlib.md5(_flatten_scope(ctx.scope).encode("utf-8")).hexdigest()[:12]

    manifest = {
        "run_name": ctx.run_name,
        "analysis_id": ctx.analysis_id,
        "scope_hash": scope_hash,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_root": ctx.output_root,
        "config_path": os.path.abspath(args.config),
        "resolved_scope": ctx.scope,
        "run_config": ctx.run_cfg,
        "defaults": {
            "normalization_level": ctx.run_cfg.get("normalization_level", "deployment"),
            "corr_threshold": float(ctx.run_cfg.get("corr_threshold", 0.80)),
        },
    }
    _ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[resolve] wrote {args.output}")


def _parse_channel_feature_spec(run_cfg: Dict) -> Dict:
    spec = run_cfg.get("channel_feature_spec") or {}
    if not spec:
        raise ValueError("channel_feature_spec is required")
    parsed = {}
    for key, cfg in spec.items():
        if "." not in key:
            raise ValueError(f"Invalid channel_feature_spec key '{key}', expected 'signal.channel'")
        signal_id, channel_id = key.split(".", 1)
        transforms = list(cfg.get("transforms") or ["raw"])
        feat_set = cfg.get("feature_set", "minimal")
        if feat_set not in {"minimal", "medium", "full"}:
            raise ValueError(f"Invalid feature_set for {key}: {feat_set}")
        parsed[key] = {
            "signal_id": signal_id,
            "channel_id": channel_id,
            "transforms": transforms,
            "feature_set": feat_set,
            "required": bool(cfg.get("required", True)),
        }
    return parsed


def _parse_standardized_channels(run_cfg: Dict) -> Dict[str, Dict]:
    std_cfg = (((run_cfg.get("preprocessing") or {}).get("standardized_channels")) or {})
    parsed = {}
    for source_key, cfg in std_cfg.items():
        if "." not in source_key:
            raise ValueError(f"Invalid standardized channel key '{source_key}', expected 'signal.channel'")
        signal_id, channel_id = source_key.split(".", 1)
        method = str(cfg.get("method", "adaptive_gaussian_quantized"))
        if method != "adaptive_gaussian_quantized":
            raise ValueError(f"Unsupported preprocessing method for {source_key}: {method}")
        derivative_orders = [int(x) for x in (cfg.get("derivative_orders") or [0])]
        if any(x not in {0, 1, 2} for x in derivative_orders):
            raise ValueError(f"Unsupported derivative order for {source_key}: {derivative_orders}")
        fallback_source_keys = list(cfg.get("fallback_source_keys") or [])
        if source_key == "depth.depth":
            fallback_source_keys = [
                "depth.corrected_depth",
                *[key for key in fallback_source_keys if key != "depth.corrected_depth"],
            ]
        parsed[source_key] = {
            "source_key": source_key,
            "signal_id": signal_id,
            "channel_id": channel_id,
            "fallback_source_keys": fallback_source_keys,
            "method": method,
            "quantize_step": float(cfg.get("quantize_step", 1.0)) if cfg.get("quantize_step") is not None else None,
            "base_smooth_seconds": float(cfg.get("base_smooth_seconds", 6.0)),
            "coarse_smooth_seconds": float(cfg.get("coarse_smooth_seconds", 12.0)),
            "coarse_interval_threshold_s": float(cfg.get("coarse_interval_threshold_s", 5.0)),
            "coarse_resolution_threshold": float(cfg.get("coarse_resolution_threshold", 1.0)),
            "derivative_orders": sorted(set(derivative_orders)),
            "output_prefix": str(cfg.get("output_prefix") or f"standardized_{channel_id}"),
        }
    return parsed


def _normalize_spec_for_processing(run_cfg: Dict) -> Tuple[Dict, Dict]:
    spec = _parse_channel_feature_spec(run_cfg)
    standardized_cfg = _parse_standardized_channels(run_cfg)
    produced = {}
    for cfg in standardized_cfg.values():
        names = set()
        base = cfg["channel_id"]
        for order in cfg["derivative_orders"]:
            if order == 0:
                names.add(f"{base}_std")
            else:
                names.add(f"{base}_d{order}_std")
        produced[cfg["output_prefix"]] = names
    for key, scfg in spec.items():
        sig = scfg["signal_id"]
        ch = scfg["channel_id"]
        if sig not in produced:
            continue
        if ch not in produced[sig]:
            raise ValueError(
                f"channel_feature_spec entry '{key}' does not match standardized outputs for '{sig}'. "
                f"Expected one of: {sorted(produced[sig])}"
            )
    return spec, standardized_cfg


def _cluster_passes_from_run_cfg(run_cfg: Dict) -> List[Dict]:
    configured = list(run_cfg.get("cluster_passes") or [])
    if not configured:
        base_spec = _parse_channel_feature_spec(run_cfg)
        return [
            {
                "name": str(run_cfg.get("cluster_pass_name") or f"k{run_cfg.get('n_clusters', 'auto')}"),
                "channel_feature_spec": base_spec,
                "primary_input_channel_for_ranking": run_cfg.get("primary_input_channel_for_ranking") or next(iter(base_spec.keys())),
                "n_clusters": run_cfg.get("n_clusters"),
                "max_k": run_cfg.get("max_k", 10),
            }
        ]

    out = []
    for i, pass_cfg in enumerate(configured, start=1):
        if not isinstance(pass_cfg, dict):
            raise ValueError(f"cluster_passes[{i-1}] must be a mapping")
        spec_cfg = pass_cfg.get("channel_feature_spec") or run_cfg.get("channel_feature_spec")
        parsed_spec = _parse_channel_feature_spec({"channel_feature_spec": spec_cfg})
        primary = pass_cfg.get("primary_input_channel_for_ranking") or run_cfg.get("primary_input_channel_for_ranking")
        if not primary:
            primary = next(iter(parsed_spec.keys()))
        out.append(
            {
                "name": str(pass_cfg.get("name") or f"pass{i}"),
                "channel_feature_spec": parsed_spec,
                "primary_input_channel_for_ranking": str(primary),
                "n_clusters": pass_cfg.get("n_clusters", run_cfg.get("n_clusters")),
                "max_k": int(pass_cfg.get("max_k", run_cfg.get("max_k", 10))),
            }
        )
    return out


def _parse_algorithmic_segments_cfg(run_cfg: Dict) -> Dict:
    cfg = dict(run_cfg.get("algorithmic_segments") or {})
    if not cfg:
        return {"enabled": False}

    source_key = str(
        cfg.get("source_channel")
        or cfg.get("source_key")
        or run_cfg.get("primary_input_channel_for_ranking")
        or "depth.depth"
    )
    if "." not in source_key:
        raise ValueError(f"Invalid algorithmic_segments source '{source_key}', expected 'signal.channel'")
    signal_id, channel_id = source_key.split(".", 1)
    fallback_source_keys = list(cfg.get("fallback_source_keys") or [])
    if source_key == "depth.depth":
        fallback_source_keys = [
            "depth.corrected_depth",
            *[key for key in fallback_source_keys if key != "depth.corrected_depth"],
        ]

    thresholds = dict(cfg.get("thresholds") or {})
    event_keys = dict(cfg.get("event_keys") or {})
    label_codes = dict(cfg.get("label_codes") or {})
    label_names = dict(cfg.get("label_names") or {})
    covariates = dict(cfg.get("covariates") or {})
    debug_cfg = dict(cfg.get("debug") or {})
    terminal_criteria_cfg = dict(cfg.get("terminal_criteria") or {})
    filter_definitions = dict(
        run_cfg.get("__context_filter_definitions_merged__")
        or run_cfg.get("context_filter_definitions")
        or {}
    )
    method_name = str(cfg.get("method_name") or "find_rest")
    terminal_criteria = {
        "surface_sleep": {
            "uses_surface_sleep_min_duration": True,
        },
        "long_flat": {
            "uses_min_duration": True,
            "uses_end_flat_abs_mean_d1_max_ms": True,
        },
        "long_drift": {
            "uses_min_duration": True,
            "uses_drift_rate_abs_max_ms": True,
            "uses_curvature_abs_max_ms": True,
        },
        "unscorable": {
            "enabled": False,
            "uses_min_duration": True,
            "uses_nonzero_mean_d1": True,
            "uses_zero_mean_d2": True,
            "uses_surface_light_missing": False,
            "surface_light_missing_min_fraction": 1.0,
            "surface_light_missing_dataset_ids": [],
            "surface_light_missing_deployment_ids": [],
        },
    }
    for key, item in terminal_criteria_cfg.items():
        if not isinstance(item, dict):
            continue
        merged = dict(terminal_criteria.get(str(key), {}))
        merged.update(item)
        terminal_criteria[str(key)] = merged
    source_key_lower = str(source_key or "").strip().lower()
    source_is_reversed_axis = any(token in source_key_lower for token in ["depth", "pressure"])
    legacy_filter_out_positive = bool(cfg.get("filter_out_positive_slopes", False))
    legacy_filter_out_negative = bool(cfg.get("filter_out_negative_slopes", True))
    filter_out_ascent = bool(
        cfg.get(
            "filter_out_ascent",
            cfg.get(
                "filter_out_ascent_slopes",
                legacy_filter_out_negative if source_is_reversed_axis else legacy_filter_out_positive,
            ),
        )
    )
    filter_out_descent = bool(
        cfg.get(
            "filter_out_descent",
            cfg.get(
                "filter_out_descent_slopes",
                legacy_filter_out_positive if source_is_reversed_axis else legacy_filter_out_negative,
            ),
        )
    )
    pass_mode = str(cfg.get("context_filter_pass_mode") or "measured_only").strip().lower() or "measured_only"
    if pass_mode not in {"none", "full", "measured_only"}:
        raise ValueError(
            f"Unsupported context_filter_pass_mode '{pass_mode}'. "
            "Expected one of: none, measured_only, full."
        )
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "method": str(cfg.get("method") or "depth_drift_thresholds"),
        "method_name": method_name,
        "source_key": source_key,
        "signal_id": signal_id,
        "channel_id": channel_id,
        "fallback_source_keys": fallback_source_keys,
        "standardize": bool(cfg.get("standardize", True)),
        "quantize_step": float(cfg.get("quantize_step", 1.0)) if cfg.get("quantize_step") is not None else None,
        "base_smooth_seconds": float(cfg.get("base_smooth_seconds", 6.0)),
        "coarse_smooth_seconds": float(cfg.get("coarse_smooth_seconds", 12.0)),
        "coarse_interval_threshold_s": float(cfg.get("coarse_interval_threshold_s", 5.0)),
        "coarse_resolution_threshold": float(cfg.get("coarse_resolution_threshold", 1.0)),
        "end_segments_upon_d1_sign_change": bool(
            cfg.get(
                "end_segments_upon_d1_sign_change",
                cfg.get("sign_consistency_required", True),
            )
        ),
        "filter_out_ascent": filter_out_ascent,
        "filter_out_descent": filter_out_descent,
        "filter_out_ascent_slopes": filter_out_ascent,
        "filter_out_descent_slopes": filter_out_descent,
        "filter_out_positive_slopes": legacy_filter_out_positive,
        "filter_out_negative_slopes": legacy_filter_out_negative,
        "event_keys": {
            "initial": str(event_keys.get("initial") or f"{method_name}.initial"),
            "filtered": str(event_keys.get("filtered") or f"{method_name}.filtered"),
            "flat": str(event_keys.get("flat") or f"{method_name}.long_flat"),
            "surface_sleep": str(
                event_keys.get("surface_sleep")
                or event_keys.get("resting_surface")
                or f"{method_name}.surface_sleep"
            ),
            "resting_surface": str(
                event_keys.get("resting_surface")
                or event_keys.get("surface_sleep")
                or f"{method_name}.resting_surface"
            ),
            "long_drift": str(event_keys.get("long_drift") or f"{method_name}.long_drift"),
            "long_flat": str(
                event_keys.get("long_flat")
                or event_keys.get("resting_benthic")
                or f"{method_name}.long_flat"
            ),
            "resting_benthic": str(
                event_keys.get("resting_benthic")
                or event_keys.get("long_flat")
                or f"{method_name}.resting_benthic"
            ),
            "not_sleep": str(
                event_keys.get("not_sleep")
                or event_keys.get("active_uw_swimming")
                or f"{method_name}.not_sleep"
            ),
            "active_surface": str(
                event_keys.get("active_surface")
                or event_keys.get("active_surface_swimming")
                or f"{method_name}.active_surface"
            ),
            "calm_uw_gliding": str(
                event_keys.get("calm_uw_gliding")
                or event_keys.get("not_sleep")
                or f"{method_name}.calm_uw_gliding"
            ),
            "active_uw_swimming": str(
                event_keys.get("active_uw_swimming")
                or event_keys.get("not_sleep")
                or f"{method_name}.active_uw_swimming"
            ),
            "calm_surface_gliding": str(
                event_keys.get("calm_surface_gliding")
                or event_keys.get("active_surface")
                or f"{method_name}.calm_surface_gliding"
            ),
            "active_surface_swimming": str(
                event_keys.get("active_surface_swimming")
                or event_keys.get("active_surface")
                or f"{method_name}.active_surface_swimming"
            ),
            "unscorable": str(event_keys.get("unscorable") or f"{method_name}.unscorable"),
        },
        "label_codes": {
            "not_sleep": int(label_codes.get("not_sleep", 0)),
            "surface_sleep": int(label_codes.get("surface_sleep", label_codes.get("resting_surface", 1))),
            "resting_surface": int(label_codes.get("resting_surface", label_codes.get("surface_sleep", 1))),
            "long_flat": int(label_codes.get("long_flat", label_codes.get("resting_benthic", 2))),
            "resting_benthic": int(label_codes.get("resting_benthic", label_codes.get("long_flat", 2))),
            "long_drift": int(label_codes.get("long_drift", 3)),
            "active_surface": int(label_codes.get("active_surface", label_codes.get("not_sleep", 0))),
            "calm_uw_gliding": int(label_codes.get("calm_uw_gliding", label_codes.get("not_sleep", 0))),
            "active_uw_swimming": int(label_codes.get("active_uw_swimming", label_codes.get("not_sleep", 0))),
            "calm_surface_gliding": int(
                label_codes.get("calm_surface_gliding", label_codes.get("active_surface", label_codes.get("not_sleep", 0)))
            ),
            "active_surface_swimming": int(
                label_codes.get("active_surface_swimming", label_codes.get("active_surface", label_codes.get("not_sleep", 0)))
            ),
            "unscorable": int(label_codes.get("unscorable", 4)),
        },
        "label_names": {
            "not_sleep": str(
                label_names.get("not_sleep")
                or label_names.get("active_uw_swimming")
                or f"{method_name}.not_sleep"
            ),
            "surface_sleep": str(
                label_names.get("surface_sleep")
                or label_names.get("resting_surface")
                or f"{method_name}.surface_sleep"
            ),
            "resting_surface": str(
                label_names.get("resting_surface")
                or label_names.get("surface_sleep")
                or f"{method_name}.resting_surface"
            ),
            "long_flat": str(
                label_names.get("long_flat")
                or label_names.get("resting_benthic")
                or f"{method_name}.long_flat"
            ),
            "resting_benthic": str(
                label_names.get("resting_benthic")
                or label_names.get("long_flat")
                or f"{method_name}.resting_benthic"
            ),
            "long_drift": str(label_names.get("long_drift") or f"{method_name}.long_drift"),
            "active_surface": str(label_names.get("active_surface") or f"{method_name}.active_surface"),
            "calm_uw_gliding": str(label_names.get("calm_uw_gliding") or f"{method_name}.calm_uw_gliding"),
            "active_uw_swimming": str(label_names.get("active_uw_swimming") or f"{method_name}.active_uw_swimming"),
            "calm_surface_gliding": str(
                label_names.get("calm_surface_gliding") or f"{method_name}.calm_surface_gliding"
            ),
            "active_surface_swimming": str(
                label_names.get("active_surface_swimming") or f"{method_name}.active_surface_swimming"
            ),
            "unscorable": str(label_names.get("unscorable") or f"{method_name}.unscorable"),
        },
        "thresholds": {
            "dive_depth_min_m": float(thresholds.get("dive_depth_min_m", 2.0)),
            "first_deriv_abs_max_ms": float(thresholds.get("first_deriv_abs_max_ms", 0.6)),
            "first_deriv_min_ms": float(thresholds.get("first_deriv_min_ms", -abs(float(thresholds.get("first_deriv_abs_max_ms", 0.6))))),
            "first_deriv_max_ms": float(thresholds.get("first_deriv_max_ms", abs(float(thresholds.get("first_deriv_abs_max_ms", 0.6))))),
            "second_deriv_abs_max_ms2": float(thresholds.get("second_deriv_abs_max_ms2", 0.05)),
            "second_deriv_min_ms2": float(thresholds.get("second_deriv_min_ms2", -abs(float(thresholds.get("second_deriv_abs_max_ms2", 0.05))))),
            "second_deriv_max_ms2": float(thresholds.get("second_deriv_max_ms2", abs(float(thresholds.get("second_deriv_abs_max_ms2", 0.05))))),
            "min_duration_s": float(thresholds.get("min_duration_s", 180.0)),
            "surface_sleep_min_duration_s": float(thresholds.get("surface_sleep_min_duration_s", 600.0)),
            "stroke_rate_active_threshold_spm": float(thresholds.get("stroke_rate_active_threshold_spm", 10.0)),
            "drift_rate_abs_max_ms": float(thresholds.get("drift_rate_abs_max_ms", 0.55)),
            "curvature_abs_max_ms": float(thresholds.get("curvature_abs_max_ms", 0.30)),
            "end_flat_abs_mean_d1_max_ms": float(thresholds.get("end_flat_abs_mean_d1_max_ms", 0.01)),
            "unscorable_interpolated_gap_min_duration_s": float(
                thresholds.get(
                    "unscorable_interpolated_gap_min_duration_s",
                    cfg.get("unscorable_interpolated_gap_min_duration_s", 40.0),
                )
            ),
            "unscorable_interpolated_gap_min_abs_mean_d1_ms": float(
                thresholds.get(
                    "unscorable_interpolated_gap_min_abs_mean_d1_ms",
                    cfg.get("unscorable_interpolated_gap_min_abs_mean_d1_ms", 0.0),
                )
            ),
            "unscorable_interpolated_gap_max_abs_mean_d2_ms2": float(
                thresholds.get(
                    "unscorable_interpolated_gap_max_abs_mean_d2_ms2",
                    cfg.get("unscorable_interpolated_gap_max_abs_mean_d2_ms2", 0.0),
                )
            ),
            "unscorable_interpolated_gap_abs_mean_d2_tolerance_ms2": float(
                thresholds.get(
                    "unscorable_interpolated_gap_abs_mean_d2_tolerance_ms2",
                    cfg.get("unscorable_interpolated_gap_abs_mean_d2_tolerance_ms2", 1e-9),
                )
            ),
        },
        "covariates": {
            "intrinsic": dict(covariates.get("intrinsic") or {}),
            "extrinsic": dict(covariates.get("extrinsic") or {}),
        },
        "context_filters": _resolve_named_filter_items(filter_definitions, list(cfg.get("context_filters") or [])),
        "context_filter_pass_mode": pass_mode,
        "terminal_criteria": terminal_criteria,
        "debug": {
            "write_context_review_table": bool(debug_cfg.get("write_context_review_table", False)),
            "persist_context_helper_channels": bool(debug_cfg.get("persist_context_helper_channels", False)),
            "persist_intermediate_channels": bool(debug_cfg.get("persist_intermediate_channels", True)),
        },
    }


def _parse_interactive_review_cfg(run_cfg: Dict) -> Dict:
    plots_cfg = dict(run_cfg.get("plots") or {})
    interactive_cfg = dict(plots_cfg.get("interactive_review") or {})
    enabled = bool(interactive_cfg.get("enabled", True))
    deployment_scope = str(interactive_cfg.get("deployment_scope") or "all").strip().lower()
    time_range_mode = str(interactive_cfg.get("time_range_mode") or "centered_window").strip().lower()
    preview_hours = float(interactive_cfg.get("preview_hours", 6.0))
    if deployment_scope not in {"all", "first_per_dataset", "first_only"}:
        raise ValueError(
            f"plots.interactive_review.deployment_scope must be one of "
            f"'all', 'first_per_dataset', or 'first_only'; got {deployment_scope!r}"
        )
    if time_range_mode not in {"centered_window", "full_deployment"}:
        raise ValueError(
            f"plots.interactive_review.time_range_mode must be one of "
            f"'centered_window' or 'full_deployment'; got {time_range_mode!r}"
        )
    return {
        "enabled": enabled,
        "deployment_scope": deployment_scope,
        "time_range_mode": time_range_mode,
        "preview_hours": preview_hours,
    }


def _parse_window_trim_cfg(run_cfg: Dict) -> Dict:
    trim_cfg = dict(run_cfg.get("window_trim") or {})
    source_key = str(trim_cfg.get("source_channel") or trim_cfg.get("source_key") or "depth.depth")
    if "." not in source_key:
        raise ValueError(f"Invalid window_trim source '{source_key}', expected 'signal.channel'")
    signal_id, channel_id = source_key.split(".", 1)
    rule = str(trim_cfg.get("rule") or "first_last_depth_threshold").strip().lower()
    if rule not in {"first_last_depth_threshold"}:
        raise ValueError(
            f"window_trim.rule must be 'first_last_depth_threshold'; got {rule!r}"
        )
    return {
        "enabled": bool(trim_cfg.get("enabled", False)),
        "source_key": source_key,
        "signal_id": signal_id,
        "channel_id": channel_id,
        "rule": rule,
        "depth_threshold_m": float(trim_cfg.get("depth_threshold_m", 10.0)),
    }


def _parse_supervised_transfer_cfg(run_cfg: Dict) -> Dict:
    supervised_cfg = dict(run_cfg.get("supervised") or {})
    source_split_mode = str(
        supervised_cfg.get("transfer_source_split_mode")
        or "deployment_holdout_folds"
    ).strip().lower()
    if source_split_mode not in {"deployment_holdout_folds", "random_70_30"}:
        raise ValueError(
            "Unsupported supervised.transfer_source_split_mode "
            f"'{source_split_mode}'. Expected 'deployment_holdout_folds' or 'random_70_30'."
        )
    source_dataset_ids = [
        str(v).strip()
        for v in list(supervised_cfg.get("source_dataset_ids") or [])
        if str(v).strip()
    ]
    always_train = [
        str(v).strip()
        for v in list(supervised_cfg.get("source_training_deployment_ids_always") or [])
        if str(v).strip()
    ]
    raw_folds = list(supervised_cfg.get("source_holdout_folds") or [])
    parsed_folds = []
    for i, raw in enumerate(raw_folds):
        if not isinstance(raw, dict):
            raise ValueError(f"supervised.source_holdout_folds[{i}] must be a mapping")
        fold_id = str(raw.get("id") or raw.get("fold_id") or f"fold_{i+1}").strip()
        holdout_ids = [
            str(v).strip()
            for v in list(raw.get("holdout_deployment_ids") or raw.get("holdout") or [])
            if str(v).strip()
        ]
        additional_train_ids = [
            str(v).strip()
            for v in list(raw.get("additional_train_deployment_ids") or raw.get("train_additional") or [])
            if str(v).strip()
        ]
        if not holdout_ids:
            raise ValueError(f"supervised.source_holdout_folds[{i}] must define holdout deployments")
        parsed_folds.append(
            {
                "fold_id": fold_id,
                "holdout_deployment_ids": holdout_ids,
                "additional_train_deployment_ids": additional_train_ids,
            }
        )
    enabled = bool(source_dataset_ids and (parsed_folds or source_split_mode == "random_70_30"))
    transfer_normalization = str(
        supervised_cfg.get("transfer_normalization") or "per_dataset_zscore"
    ).strip().lower()
    transfer_target_scope = str(
        supervised_cfg.get("transfer_target_scope") or "all_non_source"
    ).strip().lower()
    transfer_fusion_modes = [
        str(v).strip()
        for v in list(supervised_cfg.get("transfer_fusion_modes") or ["none", "algorithmic_gate"])
        if str(v).strip()
    ]
    return {
        "enabled": enabled,
        "source_split_mode": source_split_mode,
        "source_dataset_ids": source_dataset_ids,
        "source_training_deployment_ids_always": always_train,
        "source_holdout_folds": parsed_folds,
        "transfer_normalization": transfer_normalization,
        "transfer_target_scope": transfer_target_scope,
        "transfer_fusion_modes": transfer_fusion_modes,
    }


def _compute_window_trim_interval(
    merged_signal_data: Dict[str, pd.DataFrame],
    deployment_tz: str,
    trim_cfg: Dict,
) -> Tuple[pd.Timestamp | None, pd.Timestamp | None, Dict[str, Any]]:
    summary = {
        "trim_enabled": bool(trim_cfg.get("enabled", False)),
        "trim_rule": str(trim_cfg.get("rule") or ""),
        "trim_depth_threshold_m": float(trim_cfg.get("depth_threshold_m", np.nan)),
        "trim_start_datetime": None,
        "trim_end_datetime": None,
        "trim_status": "disabled",
    }
    if not trim_cfg.get("enabled", False):
        return None, None, summary

    sig = str(trim_cfg["signal_id"])
    ch = str(trim_cfg["channel_id"])
    if sig not in merged_signal_data or ch not in merged_signal_data[sig].columns:
        summary["trim_status"] = "missing_source_channel"
        return None, None, summary
    sdf = merged_signal_data[sig]
    if "datetime" not in sdf.columns:
        summary["trim_status"] = "missing_datetime"
        return None, None, summary
    dt = _normalize_datetime_series_to_timezone(sdf["datetime"], deployment_tz)
    vals = pd.to_numeric(sdf[ch], errors="coerce")
    mask = dt.notna() & vals.notna() & (vals.abs() > float(trim_cfg["depth_threshold_m"]))
    if not mask.any():
        summary["trim_status"] = "no_in_water_interval"
        return None, None, summary
    trim_start = pd.Timestamp(dt.loc[mask].iloc[0])
    trim_end = pd.Timestamp(dt.loc[mask].iloc[-1])
    summary["trim_start_datetime"] = trim_start.isoformat()
    summary["trim_end_datetime"] = trim_end.isoformat()
    summary["trim_status"] = "ok"
    return trim_start, trim_end, summary


def _window_is_within_trim(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    trim_start: pd.Timestamp | None,
    trim_end: pd.Timestamp | None,
) -> bool:
    if trim_start is None or trim_end is None:
        return True
    start_ts = pd.Timestamp(window_start)
    end_ts = pd.Timestamp(window_end)
    return (start_ts >= trim_start) and (end_ts <= trim_end)


def _localize_window_columns_by_deployment(ctx: RunContext, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not {"dataset_id", "deployment_id"}.issubset(df.columns):
        return df
    out_frames = []
    for (dataset_id, deployment_id), sub in df.groupby(["dataset_id", "deployment_id"], sort=False):
        pkl_path = _data_pkl_path(ctx, str(dataset_id), str(deployment_id))
        deployment_tz = "UTC"
        if os.path.exists(pkl_path):
            try:
                data_pkl = _load_data_pkl(pkl_path)
                deployment_tz = _deployment_timezone_name(data_pkl)
            except Exception:
                deployment_tz = "UTC"
        localized = _normalize_window_columns(sub.copy(), deployment_tz)
        for col in ("window_start", "window_end"):
            if col in localized.columns:
                localized[col] = pd.to_datetime(localized[col], errors="coerce")
                if pd.api.types.is_datetime64tz_dtype(localized[col]):
                    localized[col] = localized[col].dt.tz_localize(None)
        out_frames.append(localized)
    if not out_frames:
        return df
    return pd.concat(out_frames, ignore_index=False).sort_index()


def _resolve_named_filter_items(definitions: Dict[str, Any], items: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    defs = dict(definitions or {})
    for item in list(items or []):
        if isinstance(item, str):
            name = str(item).strip()
            if not name:
                continue
            if name not in defs or not isinstance(defs[name], dict):
                raise ValueError(f"Unknown context filter reference '{name}'")
            resolved = dict(defs[name])
            resolved.setdefault("id", name)
            out.append(resolved)
            continue
        if isinstance(item, dict) and "use" in item:
            name = str(item.get("use") or "").strip()
            if not name or name not in defs or not isinstance(defs[name], dict):
                raise ValueError(f"Unknown context filter reference '{name}'")
            resolved = dict(defs[name])
            override = {k: v for k, v in dict(item).items() if k != "use"}
            if "rule" in override and isinstance(override["rule"], dict):
                base_rule = dict(resolved.get("rule") or {})
                base_rule.update(dict(override["rule"]))
                override["rule"] = base_rule
            resolved.update(override)
            resolved.setdefault("id", name)
            out.append(resolved)
            continue
        if isinstance(item, dict) and "id" in item:
            name = str(item.get("id") or "").strip()
            if name and name in defs and isinstance(defs[name], dict):
                resolved = dict(defs[name])
                override = dict(item)
                override.pop("id", None)
                if "rule" in override and isinstance(override["rule"], dict):
                    base_rule = dict(resolved.get("rule") or {})
                    base_rule.update(dict(override["rule"]))
                    override["rule"] = base_rule
                resolved.update(override)
                resolved.setdefault("id", name)
                out.append(resolved)
                continue
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _required_algorithmic_signal_channels(run_cfg: Dict) -> List[str]:
    raw_algo_cfg = dict(run_cfg.get("algorithmic_segments") or {})
    if not bool(raw_algo_cfg.get("enabled", False)):
        return []
    required = []
    cov_cfg = dict((raw_algo_cfg.get("covariates") or {}).get("intrinsic") or {})
    for _cov_name, single_cfg in cov_cfg.items():
        single_cfg = dict(single_cfg or {})
        if not bool(single_cfg.get("enabled", False)):
            continue
        provider = str(single_cfg.get("provider") or "").strip().lower()
        source_channel = str(single_cfg.get("source_channel") or "").strip()
        if provider == "segment_channel_stat" and source_channel and "." in source_channel:
            required.append(source_channel)
    return sorted(set(required))


def _nominal_class_to_state_key(nominal_class: str, event_keys: Dict[str, str]) -> str:
    name = str(nominal_class or "").strip()
    if name == "resting_surface":
        return str(event_keys.get("resting_surface") or event_keys.get("surface_sleep") or "")
    if name == "surface_sleep":
        return str(event_keys.get("surface_sleep") or "")
    if name == "resting_benthic":
        return str(event_keys.get("resting_benthic") or event_keys.get("long_flat") or "")
    if name == "long_flat":
        return str(event_keys.get("long_flat") or "")
    if name == "long_drift":
        return str(event_keys.get("long_drift") or "")
    if name == "calm_uw_gliding":
        return str(event_keys.get("calm_uw_gliding") or event_keys.get("not_sleep") or "")
    if name == "active_uw_swimming":
        return str(event_keys.get("active_uw_swimming") or event_keys.get("not_sleep") or "")
    if name == "calm_surface_gliding":
        return str(event_keys.get("calm_surface_gliding") or event_keys.get("active_surface") or "")
    if name == "active_surface_swimming":
        return str(event_keys.get("active_surface_swimming") or event_keys.get("active_surface") or "")
    if name == "not_sleep":
        return str(event_keys.get("not_sleep") or "")
    if name == "active_surface":
        return str(event_keys.get("active_surface") or "")
    if name == "unscorable":
        return str(event_keys.get("unscorable") or "")
    return ""


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _rolling_window_size(n_rows: int, window_fraction: float, min_periods: int = 1) -> int:
    if n_rows <= 0:
        return 1
    return max(int(np.ceil(n_rows * max(float(window_fraction), 0.0))), int(min_periods), 1)


def _apply_subset_filters(segment_df: pd.DataFrame, subset_filters: Dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=segment_df.index, dtype=bool)
    if not subset_filters:
        return mask
    if subset_filters.get("min_duration_s") is not None and "duration_s" in segment_df.columns:
        mask &= pd.to_numeric(segment_df["duration_s"], errors="coerce") >= float(subset_filters["min_duration_s"])
    if subset_filters.get("max_start_depth_m") is not None and "start_depth_m" in segment_df.columns:
        mask &= pd.to_numeric(segment_df["start_depth_m"], errors="coerce") <= float(subset_filters["max_start_depth_m"])
    if bool(subset_filters.get("exclude_long_flats", False)) and "base_nominal_class" in segment_df.columns:
        mask &= ~segment_df["base_nominal_class"].astype(str).isin(["long_flat", "resting_benthic"])
    return mask.fillna(False)


def _series_from_expr(segment_df: pd.DataFrame, expr: str | None, default_field: str | None = None) -> pd.Series:
    if segment_df is None or segment_df.empty:
        return pd.Series(dtype=float)
    if default_field and default_field in segment_df.columns and not str(expr or "").strip():
        return segment_df[default_field]
    expr_text = str(expr or "").strip()
    if not expr_text:
        return pd.Series(np.nan, index=segment_df.index, dtype=float)
    if expr_text in segment_df.columns:
        return segment_df[expr_text]

    abs_diff = re.fullmatch(r"abs\(\s*([A-Za-z0-9_\.]+)\s*-\s*([A-Za-z0-9_\.]+)\s*\)", expr_text)
    if abs_diff:
        left = abs_diff.group(1)
        right = abs_diff.group(2)
        if left not in segment_df.columns or right not in segment_df.columns:
            return pd.Series(np.nan, index=segment_df.index, dtype=float)
        return (pd.to_numeric(segment_df[left], errors="coerce") - pd.to_numeric(segment_df[right], errors="coerce")).abs()

    abs_field = re.fullmatch(r"abs\(\s*([A-Za-z0-9_\.]+)\s*\)", expr_text)
    if abs_field:
        field = abs_field.group(1)
        if field not in segment_df.columns:
            return pd.Series(np.nan, index=segment_df.index, dtype=float)
        return pd.to_numeric(segment_df[field], errors="coerce").abs()

    return pd.Series(np.nan, index=segment_df.index, dtype=float)


def _compare_series(series: pd.Series, comparator: str, threshold) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    thr = pd.to_numeric(pd.Series([threshold]), errors="coerce").iloc[0]
    op = str(comparator or "").strip()
    if op == ">":
        return values > thr
    if op == ">=":
        return values >= thr
    if op == "<":
        return values < thr
    if op == "<=":
        return values <= thr
    if op == "==":
        return values == thr
    if op == "!=":
        return values != thr
    raise ValueError(f"Unsupported comparator '{comparator}'")


def _resolve_covariate_output_fields(cov_name: str, cov_cfg: Dict[str, Any]) -> Dict[str, str]:
    outputs = dict(cov_cfg.get("outputs") or {})
    return {
        "value_field": str(outputs.get("segment_value_field") or f"{cov_name}_value"),
        "phase_field": str(outputs.get("segment_phase_field") or f"{cov_name}_phase"),
        "positive_likely_day_field": str(outputs.get("segment_positive_likely_day_field") or f"{cov_name}_positive_likely_day"),
        "trip_phase_field": str(outputs.get("segment_trip_phase_field") or f"{cov_name}_trip_phase"),
        "early_reference_field": str(outputs.get("segment_early_reference_field") or f"{cov_name}_early_reference_value"),
        "late_reference_field": str(outputs.get("segment_late_reference_field") or f"{cov_name}_late_reference_value"),
    }


def compute_intrinsic_covariate(
    segment_df: pd.DataFrame,
    sample_df: pd.DataFrame | None,
    cov_name: str,
    cov_cfg: Dict,
    runtime_params: Dict,
    deployment_context: Dict | None = None,
) -> pd.DataFrame:
    if segment_df is None or segment_df.empty:
        return segment_df

    provider = str(cov_cfg.get("provider") or "").strip().lower()
    out = segment_df.copy()
    fields = _resolve_covariate_output_fields(cov_name, cov_cfg)
    for field in fields.values():
        if field not in out.columns:
            out[field] = pd.NA

    if provider == "segment_channel_stat":
        source_channel = str(cov_cfg.get("source_channel") or "").strip()
        agg = str(cov_cfg.get("agg") or "mean").strip().lower()
        if agg != "mean":
            raise ValueError(f"Unsupported intrinsic covariate agg '{agg}' for '{cov_name}'")
        if "." not in source_channel:
            raise ValueError(f"Intrinsic covariate '{cov_name}' requires source_channel in 'signal.channel' form")
        signal_data = {}
        deployment_tz = "UTC"
        if isinstance(deployment_context, dict):
            signal_data = getattr(deployment_context.get("data_pkl"), "signal_data", None) or deployment_context.get("signal_data") or {}
            deployment_tz = str(deployment_context.get("deployment_tz") or "UTC")
        resolved_source_key, sig, ch = _resolve_available_source(signal_data, source_channel, list(cov_cfg.get("fallback_source_keys") or []))
        if not sig or sig not in signal_data or ch not in signal_data[sig].columns:
            return out
        sdf = signal_data[sig].copy()
        if "datetime" not in sdf.columns:
            return out
        sdf["datetime"] = _normalize_datetime_series_to_timezone(sdf["datetime"], deployment_tz)
        sdf[ch] = pd.to_numeric(sdf[ch], errors="coerce")
        sdf = sdf.dropna(subset=["datetime", ch]).sort_values("datetime").reset_index(drop=True)
        if sdf.empty:
            return out
        values = []
        for row in out.itertuples(index=False):
            seg_start = pd.to_datetime(getattr(row, "start_datetime", pd.NaT), errors="coerce")
            seg_end = pd.to_datetime(getattr(row, "end_datetime", pd.NaT), errors="coerce")
            if pd.isna(seg_start) or pd.isna(seg_end):
                values.append(np.nan)
                continue
            mask = (sdf["datetime"] >= seg_start) & (sdf["datetime"] <= seg_end)
            if not mask.any():
                values.append(np.nan)
                continue
            values.append(float(pd.to_numeric(sdf.loc[mask, ch], errors="coerce").mean()))
        out[fields["value_field"]] = values
        out[fields["phase_field"]] = "observed"
        out[fields["trip_phase_field"]] = "observed"
        return out

    if provider != "smoothed_segment_metric":
        raise ValueError(f"Unsupported intrinsic covariate provider '{provider}' for '{cov_name}'")

    source_metric = str(cov_cfg.get("source_metric") or "").strip()
    if not source_metric or source_metric not in out.columns:
        raise ValueError(f"Intrinsic covariate '{cov_name}' requires source metric column '{source_metric}'")

    subset_filters = dict(cov_cfg.get("subset_filters") or cov_cfg.get("include_segment_subset_filters") or {})
    subset_mask = _apply_subset_filters(out, subset_filters)
    if "post_measured_keep" in out.columns:
        subset_mask &= out["post_measured_keep"].fillna(False).astype(bool)
    elif "context_keep" in out.columns:
        subset_mask &= out["context_keep"].fillna(False).astype(bool)
    subset = out.loc[subset_mask].copy().sort_values("segment_midpoint_datetime")

    if subset.empty:
        out[fields["phase_field"]] = "unknown"
        out[fields["trip_phase_field"]] = "unknown"
        return out

    value_series = pd.to_numeric(subset[source_metric], errors="coerce")
    smoothing = dict(cov_cfg.get("smoothing") or {})
    window_fraction = float(smoothing.get("window_fraction_of_trip", 0.05))
    min_periods = int(smoothing.get("min_periods", 1))
    window_size = _rolling_window_size(len(subset), window_fraction, min_periods=min_periods)
    smoothed_subset = value_series.rolling(window=window_size, center=True, min_periods=min_periods).mean()
    if smoothed_subset.isna().all():
        smoothed_subset = value_series.copy()
    else:
        smoothed_subset = smoothed_subset.interpolate(limit_direction="both")

    subset_positions = np.arange(len(subset), dtype=float)
    full_positions = np.interp(
        pd.to_numeric(out["elapsed_days"], errors="coerce").ffill().bfill().fillna(0.0),
        pd.to_numeric(subset["elapsed_days"], errors="coerce").ffill().bfill().fillna(0.0),
        subset_positions,
        left=float(subset_positions.min()),
        right=float(subset_positions.max()),
    )
    interpolated = np.interp(full_positions, subset_positions, smoothed_subset.to_numpy(dtype=float))
    out[fields["value_field"]] = interpolated

    phase_thresholds = dict(cov_cfg.get("phase_thresholds") or {})
    negative_max = _safe_float(phase_thresholds.get("negative_max"), -0.03)
    positive_min = _safe_float(phase_thresholds.get("positive_min"), 0.0)
    out[fields["phase_field"]] = np.where(
        pd.to_numeric(out[fields["value_field"]], errors="coerce") <= negative_max,
        "negative",
        np.where(
            pd.to_numeric(out[fields["value_field"]], errors="coerce") >= positive_min,
            "positive",
            "neutral",
        ),
    )

    trip_phase_cfg = dict(cov_cfg.get("trip_phase_inference") or {})
    min_positive_elapsed_days = float(trip_phase_cfg.get("min_positive_elapsed_days", 20.0))
    backtrack_days = float(trip_phase_cfg.get("backtrack_days", 20.0))
    always_positive_max_day = float(trip_phase_cfg.get("always_positive_max_day", 20.0))
    near_arrival_exclusion_days = float(trip_phase_cfg.get("near_arrival_exclusion_days", 5.0))
    positive_candidates = subset.loc[
        (pd.to_numeric(subset["elapsed_days"], errors="coerce") > min_positive_elapsed_days)
        & (pd.to_numeric(smoothed_subset, errors="coerce") > positive_min),
        "elapsed_days",
    ]
    positive_likely_day = np.nan
    if not positive_candidates.empty:
        positive_likely_day = float(positive_candidates.min()) - backtrack_days

    max_elapsed = float(pd.to_numeric(out["elapsed_days"], errors="coerce").max()) if "elapsed_days" in out.columns else np.nan
    if np.isfinite(positive_likely_day) and positive_likely_day < always_positive_max_day and positive_likely_day < (max_elapsed - near_arrival_exclusion_days):
        trip_mode = "always_positive"
    elif np.isfinite(positive_likely_day) and positive_likely_day < (max_elapsed - near_arrival_exclusion_days):
        trip_mode = "phase_shift"
    elif subset.empty:
        trip_mode = "unknown"
    else:
        trip_mode = "never_positive"

    out[fields["positive_likely_day_field"]] = positive_likely_day
    if trip_mode == "always_positive":
        out[fields["trip_phase_field"]] = "always_positive"
    elif trip_mode == "phase_shift":
        out[fields["trip_phase_field"]] = np.where(
            pd.to_numeric(out["elapsed_days"], errors="coerce") <= positive_likely_day,
            "pre_positive",
            "post_positive",
        )
    else:
        out[fields["trip_phase_field"]] = trip_mode

    early_days = float(trip_phase_cfg.get("early_window_days", 15.0))
    late_days = float(trip_phase_cfg.get("late_window_days", 15.0))
    shallow_ref_depth = float(trip_phase_cfg.get("reference_max_start_depth_m", 500.0))
    subset_after_base = out.loc[
        subset_mask
        & out["base_keep_filtered"].fillna(False)
        & out.get("post_measured_keep", out.get("context_keep", pd.Series(True, index=out.index))).fillna(False).astype(bool)
        & pd.to_numeric(out["start_depth_m"], errors="coerce").lt(shallow_ref_depth),
        ["elapsed_days", source_metric],
    ].copy()
    early_ref = np.nan
    late_ref = np.nan
    if not subset_after_base.empty and np.isfinite(max_elapsed):
        early_mask = pd.to_numeric(subset_after_base["elapsed_days"], errors="coerce") < early_days
        late_mask = pd.to_numeric(subset_after_base["elapsed_days"], errors="coerce") > max_elapsed - late_days
        if early_mask.any():
            early_ref = float(pd.to_numeric(subset_after_base.loc[early_mask, source_metric], errors="coerce").mean())
        if late_mask.any():
            late_ref = float(pd.to_numeric(subset_after_base.loc[late_mask, source_metric], errors="coerce").mean())
    out[fields["early_reference_field"]] = early_ref
    out[fields["late_reference_field"]] = late_ref
    return out


def compute_extrinsic_covariate(
    segment_df: pd.DataFrame,
    cov_name: str,
    cov_cfg: Dict,
    deployment_context: Dict | None = None,
) -> pd.DataFrame:
    if segment_df is None or segment_df.empty:
        return segment_df
    provider = str(cov_cfg.get("provider") or "").strip().lower()
    if provider != "lookup_or_join":
        raise ValueError(f"Unsupported extrinsic covariate provider '{provider}' for '{cov_name}'")

    out = segment_df.copy()
    fields = _resolve_covariate_output_fields(cov_name, cov_cfg)
    join_df = None
    if isinstance(deployment_context, dict):
        join_df = (deployment_context.get("extrinsic_tables") or {}).get(cov_name)
    if not isinstance(join_df, pd.DataFrame) or join_df.empty:
        for field in fields.values():
            if field not in out.columns:
                out[field] = pd.NA
        return out

    key_field = str(cov_cfg.get("key_field") or "segment_midpoint_datetime")
    join_key = str(cov_cfg.get("join_key_field") or "datetime")
    value_field = str(cov_cfg.get("source_value_field") or cov_name)
    if key_field not in out.columns or join_key not in join_df.columns or value_field not in join_df.columns:
        return out
    left = out.sort_values(key_field).copy()
    right = join_df[[join_key, value_field]].copy().sort_values(join_key)
    left[key_field] = pd.to_datetime(left[key_field], errors="coerce")
    right[join_key] = pd.to_datetime(right[join_key], errors="coerce")
    merged = pd.merge_asof(left, right, left_on=key_field, right_on=join_key, direction="nearest")
    out.loc[merged.index, fields["value_field"]] = merged[value_field].to_numpy()
    return out


def annotate_segment_covariates(
    segment_df: pd.DataFrame,
    sample_df: pd.DataFrame | None,
    runtime_params: Dict,
    deployment_context: Dict | None = None,
    filter_stage: str | None = None,
) -> tuple[pd.DataFrame, Dict]:
    if segment_df is None or segment_df.empty:
        return pd.DataFrame() if segment_df is None else segment_df.copy(), {}

    cov_cfg = dict(runtime_params.get("covariates") or {})
    active_filters = [
        f for f in _stage_filtered_context_filters(runtime_params.get("context_filters") or [], filter_stage=filter_stage)
        if bool(f.get("enabled", False))
    ]
    needed_refs = {
        str(f.get("covariate_ref") or "").strip()
        for f in active_filters
        if str(f.get("covariate_ref") or "").strip()
    }
    out = segment_df.copy()
    cov_meta: Dict[str, Dict[str, str]] = {}

    for cov_name, single_cfg in dict(cov_cfg.get("intrinsic") or {}).items():
        cov_ref = f"intrinsic.{cov_name}"
        if not bool((single_cfg or {}).get("enabled", False)) or cov_ref not in needed_refs:
            continue
        out = compute_intrinsic_covariate(out, sample_df, cov_name, dict(single_cfg or {}), runtime_params, deployment_context=deployment_context)
        cov_meta[cov_ref] = _resolve_covariate_output_fields(cov_name, dict(single_cfg or {}))

    for cov_name, single_cfg in dict(cov_cfg.get("extrinsic") or {}).items():
        cov_ref = f"extrinsic.{cov_name}"
        if not bool((single_cfg or {}).get("enabled", False)) or cov_ref not in needed_refs:
            continue
        out = compute_extrinsic_covariate(out, cov_name, dict(single_cfg or {}), deployment_context=deployment_context)
        cov_meta[cov_ref] = _resolve_covariate_output_fields(cov_name, dict(single_cfg or {}))

    return out, cov_meta


def _filter_applies_mask(segment_df: pd.DataFrame, applies_when: Dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=segment_df.index, dtype=bool)
    if not applies_when:
        return mask
    for key, rule in applies_when.items():
        if key == "segment_sign":
            mask &= segment_df.get("segment_sign", pd.Series(pd.NA, index=segment_df.index)).astype(str) == str(rule)
        elif key == "min_duration_s":
            mask &= pd.to_numeric(segment_df.get("duration_s"), errors="coerce") >= float(rule)
        elif key == "max_duration_s":
            mask &= pd.to_numeric(segment_df.get("duration_s"), errors="coerce") <= float(rule)
        elif key.endswith("_in"):
            field = key[:-3]
            allowed = {str(x) for x in (rule or [])}
            mask &= segment_df.get(field, pd.Series(pd.NA, index=segment_df.index)).astype(str).isin(allowed)
        elif isinstance(rule, dict) and {"comparator", "threshold"} <= set(rule.keys()):
            mask &= _compare_series(segment_df.get(key, pd.Series(np.nan, index=segment_df.index)), rule["comparator"], rule["threshold"])
        else:
            mask &= segment_df.get(key, pd.Series(pd.NA, index=segment_df.index)).astype(str) == str(rule)
    return mask.fillna(False)


def _context_filter_stage(filter_cfg: Dict[str, Any]) -> str:
    explicit = str(
        filter_cfg.get("filter_stage")
        or filter_cfg.get("context_stage")
        or filter_cfg.get("filter_category")
        or ""
    ).strip().lower()
    if explicit in {"measured", "measured_filters", "observed"}:
        return "measured"
    if explicit in {"inferred", "inferred_filters", "derived"}:
        return "inferred"

    filter_id = str(filter_cfg.get("id") or "").strip().lower()
    if filter_id in {"low_stroke_rate_required", "stroke_rate_required", "low_stroke_rate_gate"}:
        return "measured"

    candidates = [
        filter_id,
        str(filter_cfg.get("covariate_ref") or ""),
        str(filter_cfg.get("field") or ""),
        str((filter_cfg.get("rule") or {}).get("field") or ""),
        str((filter_cfg.get("rule") or {}).get("value_expr") or ""),
        json.dumps(filter_cfg.get("applies_when") or {}, sort_keys=True),
    ]
    text = " ".join(candidates).lower()
    measured_tokens = [
        "stroke_rate",
        "segment_mean_stroke_rate_spm",
        "stroke_rate_rest_gate",
    ]
    if any(token in text for token in measured_tokens):
        return "measured"
    return "inferred"


def _stage_filtered_context_filters(filter_cfgs: List[Dict], filter_stage: str | None = None) -> List[Dict]:
    items = [dict(item or {}) for item in list(filter_cfgs or [])]
    if not filter_stage:
        staged = [cfg for cfg in items if _context_filter_stage(cfg) == "measured"]
        staged.extend(cfg for cfg in items if _context_filter_stage(cfg) == "inferred")
        return staged
    wanted = str(filter_stage).strip().lower()
    return [cfg for cfg in items if _context_filter_stage(cfg) == wanted]


def _resolve_filter_value(
    segment_df: pd.DataFrame,
    filter_cfg: Dict[str, Any],
    covariate_meta: Dict[str, Dict[str, str]],
) -> pd.Series:
    rule = dict(filter_cfg.get("rule") or {})
    expr = rule.get("value_expr", filter_cfg.get("value_expr"))
    field = str(rule.get("field") or filter_cfg.get("field") or "").strip() or None
    covariate_ref = str(filter_cfg.get("covariate_ref") or "").strip()
    if expr or field:
        return _series_from_expr(segment_df, expr, default_field=field)
    if covariate_ref and covariate_ref in covariate_meta:
        value_field = covariate_meta[covariate_ref]["value_field"]
        return segment_df.get(value_field, pd.Series(np.nan, index=segment_df.index))
    return pd.Series(np.nan, index=segment_df.index, dtype=float)


def apply_context_filters(
    segment_df: pd.DataFrame,
    filter_cfgs: List[Dict],
    covariate_meta: Dict[str, Dict[str, str]] | None = None,
    filter_stage: str | None = None,
) -> pd.DataFrame:
    if segment_df is None or segment_df.empty:
        return pd.DataFrame() if segment_df is None else segment_df.copy()

    out = segment_df.copy()
    covariate_meta = covariate_meta or {}
    if "context_filter_applied" not in out.columns:
        out["context_filter_applied"] = False
    if "context_filter_ids" not in out.columns:
        out["context_filter_ids"] = ""
    if "context_keep" not in out.columns:
        out["context_keep"] = True
    if "context_reject_reason" not in out.columns:
        out["context_reject_reason"] = pd.NA
    if "context_reject_stage" not in out.columns:
        out["context_reject_stage"] = pd.NA
    if "post_measured_keep" not in out.columns:
        out["post_measured_keep"] = out["context_keep"].fillna(True).astype(bool)

    stage_specs = []
    requested_stage = str(filter_stage).strip().lower() if filter_stage else None
    if requested_stage in {None, "", "measured"}:
        stage_specs.append(("measured", "rejected_after_measured_filters"))
    if requested_stage in {None, "", "inferred"}:
        stage_specs.append(("inferred", "rejected_after_inferred_filters"))
    for current_stage, reject_stage_label in stage_specs:
        for filter_cfg in _stage_filtered_context_filters(filter_cfgs or [], filter_stage=current_stage):
            if not bool(filter_cfg.get("enabled", False)):
                continue
            filter_id = str(filter_cfg.get("id") or f"context_filter_{len(out.columns)}").strip()
            rule = dict(filter_cfg.get("rule") or {})
            mode = str(rule.get("mode") or "").strip().lower()
            action_on_fail = str(filter_cfg.get("action_on_fail") or "reject").strip().lower()
            if action_on_fail != "reject":
                raise ValueError(f"Context filter '{filter_id}' uses unsupported action_on_fail '{action_on_fail}'")
            if mode in {"relabel", "score_modifier"}:
                raise ValueError(f"Context filter '{filter_id}' uses unsupported rule mode '{mode}'")

            target_state_keys = {str(x) for x in (filter_cfg.get("target_state_keys") or []) if str(x).strip()}
            mask = out["context_keep"].fillna(True)
            if target_state_keys:
                mask &= out.get("base_state_key", pd.Series("", index=out.index)).astype(str).isin(target_state_keys)
            mask &= _filter_applies_mask(out, dict(filter_cfg.get("applies_when") or {}))
            if not mask.any():
                continue

            value_series = _resolve_filter_value(out, filter_cfg, covariate_meta)
            on_missing = str(filter_cfg.get("on_missing_covariate") or "reject").strip().lower()
            pass_mask = pd.Series(True, index=out.index, dtype=bool)

            if mode == "threshold_gate":
                keep_if = dict(rule.get("keep_if") or {})
                pass_mask = _compare_series(value_series, keep_if.get("comparator"), keep_if.get("threshold"))
            elif mode == "category_gate":
                allowed = {str(x) for x in (rule.get("allowed_values") or rule.get("keep_if_in") or [])}
                pass_mask = value_series.astype(str).isin(allowed)
            elif mode == "phase_gate":
                allowed = {str(x) for x in (rule.get("allowed_phases") or rule.get("keep_if_in") or [])}
                pass_mask = value_series.astype(str).isin(allowed)
            else:
                raise ValueError(f"Unsupported context filter rule mode '{mode}' for '{filter_id}'")

            missing_mask = value_series.isna()
            if on_missing == "keep":
                pass_mask = pass_mask | missing_mask
            elif on_missing == "skip_filter":
                pass_mask = pass_mask | missing_mask
                mask &= ~missing_mask
            elif on_missing != "reject":
                raise ValueError(f"Unsupported on_missing_covariate '{on_missing}' for '{filter_id}'")

            apply_idx = mask[mask].index
            fail_idx = apply_idx[~pass_mask.loc[apply_idx].fillna(False)]
            out.loc[apply_idx, "context_filter_applied"] = True
            existing_ids = out.loc[apply_idx, "context_filter_ids"].fillna("").astype(str)
            out.loc[apply_idx, "context_filter_ids"] = np.where(
                existing_ids.eq(""),
                filter_id,
                existing_ids + "|" + filter_id,
            )
            out[f"{filter_id}_pass"] = pd.NA
            out[f"{filter_id}_value"] = np.nan
            out.loc[apply_idx, f"{filter_id}_pass"] = pass_mask.loc[apply_idx].fillna(False).to_numpy()
            out.loc[apply_idx, f"{filter_id}_value"] = pd.to_numeric(value_series.loc[apply_idx], errors="coerce").to_numpy()

            if len(fail_idx):
                out.loc[fail_idx, "context_keep"] = False
                out.loc[fail_idx, "context_reject_stage"] = reject_stage_label
                reason = str(filter_cfg.get("reject_reason") or f"{filter_id}_failed")
                missing_reason = out.loc[fail_idx, "context_reject_reason"].isna()
                out.loc[fail_idx[missing_reason], "context_reject_reason"] = reason
                out.loc[fail_idx[~missing_reason], "context_reject_reason"] = (
                    out.loc[fail_idx[~missing_reason], "context_reject_reason"].astype(str) + "|" + reason
                )
        if current_stage == "measured":
            out["post_measured_keep"] = out["context_keep"].fillna(False).astype(bool)

    return out


def _source_pairs_from_keys(source_key: str, fallback_source_keys: List[str] = None) -> List[Tuple[str, str, str]]:
    keys = []
    fallback_source_keys = list(fallback_source_keys or [])
    if str(source_key) == "depth.depth" and "depth.corrected_depth" in fallback_source_keys:
        keys.append("depth.corrected_depth")
    keys.append(str(source_key))
    for key in fallback_source_keys:
        key = str(key)
        if key and key not in keys:
            keys.append(key)
    pairs = []
    for key in keys:
        if "." not in key:
            continue
        sig, ch = key.split(".", 1)
        pairs.append((key, sig, ch))
    return pairs


def _resolve_available_source(
    signal_data: Dict,
    source_key: str,
    fallback_source_keys: List[str] = None,
) -> Tuple[str, str, str]:
    for key, sig, ch in _source_pairs_from_keys(source_key, fallback_source_keys):
        if sig in signal_data and ch in signal_data[sig].columns:
            return key, sig, ch
    return source_key, "", ""


def _algorithmic_segments_root(ctx: RunContext) -> str:
    return os.path.join(ctx.output_root, "segments")


def _algorithmic_segment_paths(ctx: RunContext, dataset_id: str, deployment_id: str) -> Tuple[str, str]:
    safe_ds = _safe_slug(dataset_id)
    safe_dep = _safe_slug(deployment_id)
    base = _algorithmic_segments_root(ctx)
    return (
        os.path.join(base, "by_deployment", f"{safe_ds}__{safe_dep}__algorithmic_segments.parquet"),
        os.path.join(base, "by_deployment", f"{safe_ds}__{safe_dep}__algorithmic_segments_summary.json"),
    )


def _algorithmic_exhaustive_segment_paths(ctx: RunContext, dataset_id: str, deployment_id: str) -> str:
    safe_ds = _safe_slug(dataset_id)
    safe_dep = _safe_slug(deployment_id)
    base = _algorithmic_segments_root(ctx)
    return os.path.join(base, "by_deployment", f"{safe_ds}__{safe_dep}__algorithmic_segments_exhaustive.parquet")


def _algorithmic_segment_merge_paths(ctx: RunContext) -> Tuple[str, str]:
    base = _algorithmic_segments_root(ctx)
    return (
        os.path.join(base, "algorithmic_segments.parquet"),
        os.path.join(base, "algorithmic_segment_events.parquet"),
    )


def _algorithmic_exhaustive_segment_merge_path(ctx: RunContext) -> str:
    base = _algorithmic_segments_root(ctx)
    return os.path.join(base, "algorithmic_segments_exhaustive.parquet")


def _algorithmic_label_timeseries_paths(ctx: RunContext, dataset_id: str = None, deployment_id: str = None):
    base = _algorithmic_segments_root(ctx)
    if dataset_id is None or deployment_id is None:
        return os.path.join(base, "algorithmic_label_timeseries.parquet")
    safe_ds = _safe_slug(dataset_id)
    safe_dep = _safe_slug(deployment_id)
    return os.path.join(base, "by_deployment", f"{safe_ds}__{safe_dep}__algorithmic_label_timeseries.parquet")


def _algorithmic_label_timeseries_merge_path(ctx: RunContext) -> str:
    return _algorithmic_label_timeseries_paths(ctx)


def _feature_columns_for_pass(df: pd.DataFrame, pass_spec: Dict) -> List[str]:
    prefixes = tuple(f"{full_key}__" for full_key in pass_spec.keys())
    return [
        c for c in df.columns
        if not c.startswith("__")
        and pd.api.types.is_numeric_dtype(df[c])
        and c.startswith(prefixes)
    ]


def _load_qc_signal_data_from_netcdf(netcdf_path: str | None) -> Dict[str, Any]:
    """Read channel presence from NetCDF variable attrs without loading signal arrays.

    Returns a dict compatible with signal_data consumers: {signal_id: obj_with_columns}.
    """
    if not netcdf_path or not os.path.exists(netcdf_path):
        return {}
    try:
        import xarray as xr
        with xr.open_dataset(netcdf_path) as ds:
            out: Dict[str, Any] = {}
            for var_name in ds.data_vars:
                text = str(var_name)
                if not text.startswith("signal_data_"):
                    continue
                signal_id = text[len("signal_data_"):]
                channels = (
                    _as_channel_list(ds[var_name].attrs.get("variables"))
                    or _as_channel_list(ds[var_name].attrs.get("variable"))
                )
                if channels:
                    out[signal_id] = SimpleNamespace(columns=channels)
        return out
    except Exception:
        return {}


def cmd_qc(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    spec, standardized_cfg = _normalize_spec_for_processing(ctx.run_cfg)
    extra_required_channels = _required_algorithmic_signal_channels(ctx.run_cfg)
    standardized_channel_db_path = (
        str(getattr(args, "standardized_channel_db", "") or "").strip()
        or str(ctx.run_cfg.get("standardized_channel_db_path") or "").strip()
    )
    standardized_unit_map = load_standardized_channel_unit_map(standardized_channel_db_path)
    if standardized_channel_db_path:
        if standardized_unit_map:
            print(
                f"[qc] loaded standardized channel unit map: {standardized_channel_db_path} "
                f"({len(standardized_unit_map)} channel entries)"
            )
        else:
            print(f"[qc] warning: standardized channel db provided but no unit mappings found: {standardized_channel_db_path}")
    strict = bool(ctx.run_cfg.get("strict_qc", False))
    rows = []
    failures = []
    n_scope = len(ctx.scope)
    last_ds = None
    for i, item in enumerate(ctx.scope, start=1):
        ds = item["dataset_id"]
        dep = item["deployment_id"]
        if ds != last_ds:
            print(f"[qc] dataset {ds}")
            last_ds = ds
        deployment_folder = os.path.join(ctx.data_root, str(ds), str(dep))
        netcdf_path = latest_processing_netcdf_path(deployment_folder, str(dep))
        existing_pairs = []
        existing_all_channels = []
        missing_required_pairs = []
        missing_optional_pairs = []
        unit_info = {
            "channel_units": "",
            "missing_netcdf_unit_channels": "",
            "unit_metadata_mismatch_channels": "",
            "expected_unit_mismatch_channels": "",
            "inconsistent_signal_units": "",
        }
        issues = []

        if not netcdf_path or not os.path.exists(netcdf_path):
            issues.append("missing_netcdf")
            for full_key, scfg in spec.items():
                if scfg.get("required", True):
                    missing_required_pairs.append(full_key)
                else:
                    missing_optional_pairs.append(full_key)
        else:
            try:
                signal_data = _load_qc_signal_data_from_netcdf(netcdf_path)
                fake_data_pkl = SimpleNamespace(signal_data=signal_data, signal_info={})
                existing_all_channels = _list_existing_signal_channels(signal_data)
                for full_key, scfg in spec.items():
                    if _has_required_source_for_feature(signal_data, full_key, scfg, standardized_cfg):
                        existing_pairs.append(full_key)
                    else:
                        if scfg.get("required", True):
                            missing_required_pairs.append(full_key)
                        else:
                            missing_optional_pairs.append(full_key)
                if missing_required_pairs:
                    issues.append("missing_required_channels")
                extra_existing, extra_missing = _check_explicit_required_channels(signal_data, extra_required_channels)
                existing_pairs.extend([x for x in extra_existing if x not in existing_pairs])
                for full_key in extra_missing:
                    if full_key not in missing_required_pairs:
                        missing_required_pairs.append(full_key)
                if extra_missing and "missing_required_channels" not in issues:
                    issues.append("missing_required_channels")
                if missing_optional_pairs:
                    issues.append("missing_optional_channels")
                unit_info = _evaluate_required_channel_units(
                    ctx=ctx,
                    dataset_id=ds,
                    deployment_id=dep,
                    data_pkl=fake_data_pkl,
                    spec=spec,
                    standardized_cfg=standardized_cfg,
                    extra_required_channels=extra_required_channels,
                    standardized_unit_map=standardized_unit_map,
                )
                if unit_info["missing_netcdf_unit_channels"]:
                    issues.append("missing_netcdf_unit_metadata")
                if unit_info["unit_metadata_mismatch_channels"]:
                    issues.append("unit_metadata_mismatch")
                if unit_info["expected_unit_mismatch_channels"]:
                    issues.append("expected_unit_mismatch")
                if unit_info["inconsistent_signal_units"]:
                    issues.append("inconsistent_signal_units")
            except Exception as e:
                issues.append(f"load_error:{type(e).__name__}")
                for full_key, scfg in spec.items():
                    if scfg.get("required", True):
                        missing_required_pairs.append(full_key)
                    else:
                        missing_optional_pairs.append(full_key)

        unit_fail_bits = []
        if unit_info["missing_netcdf_unit_channels"]:
            unit_fail_bits.append(f"missing_netcdf_units={unit_info['missing_netcdf_unit_channels']}")
        if unit_info["unit_metadata_mismatch_channels"]:
            unit_fail_bits.append(f"metadata_mismatch={unit_info['unit_metadata_mismatch_channels']}")
        if unit_info["expected_unit_mismatch_channels"]:
            unit_fail_bits.append(f"expected_unit_mismatch={unit_info['expected_unit_mismatch_channels']}")
        if unit_info["inconsistent_signal_units"]:
            unit_fail_bits.append(f"inconsistent_units={unit_info['inconsistent_signal_units']}")
        if strict and (missing_required_pairs or unit_fail_bits):
            fail_parts = []
            if missing_required_pairs:
                fail_parts.append(",".join(missing_required_pairs))
            fail_parts.extend(unit_fail_bits)
            failures.append(f"{ds}/{dep}: " + " | ".join(fail_parts))

        rows.append({
            "dataset_id": ds,
            "deployment_id": dep,
            "existing_channels": ",".join(sorted(existing_pairs)),
            "missing_channels": ",".join(sorted(missing_required_pairs)),
            "optional_missing_channels": ",".join(sorted(missing_optional_pairs)),
            "channel_units": unit_info["channel_units"],
            "missing_netcdf_unit_channels": unit_info["missing_netcdf_unit_channels"],
            "unit_metadata_mismatch_channels": unit_info["unit_metadata_mismatch_channels"],
            "expected_unit_mismatch_channels": unit_info["expected_unit_mismatch_channels"],
            "inconsistent_signal_units": unit_info["inconsistent_signal_units"],
            "issues": ",".join(issues),
        })
        status_parts = [f"[{i}/{n_scope}] {dep}"]
        if not missing_required_pairs:
            status_parts.append(f"req_ok={len(existing_pairs)}")
        else:
            missing_with_sources = _format_missing_channels_with_sources(
                missing_required_pairs,
                spec,
                standardized_cfg,
            )
            status_parts.append(f"missing_req={len(missing_required_pairs)}")
            status_parts.append(f"missing={','.join(missing_with_sources)}")
        if unit_info["missing_netcdf_unit_channels"]:
            status_parts.append(
                f"missing_nc_units={len([x for x in unit_info['missing_netcdf_unit_channels'].split(',') if x])}"
            )
        if unit_info["unit_metadata_mismatch_channels"]:
            status_parts.append(
                f"pkl_vs_nc_mismatch={len([x for x in unit_info['unit_metadata_mismatch_channels'].split(',') if x])}"
            )
        if unit_info["expected_unit_mismatch_channels"]:
            status_parts.append(
                f"expected_unit_mismatch={len([x for x in unit_info['expected_unit_mismatch_channels'].split(',') if x])}"
            )
        if unit_info["inconsistent_signal_units"]:
            status_parts.append(
                f"inconsistent_signals={len([x for x in unit_info['inconsistent_signal_units'].split(',') if x])}"
            )
        status_parts.append(f"RSS={_get_rss_gb():.2f} GB")
        print("[qc] " + " | ".join(status_parts))

    df = pd.DataFrame(rows)
    _ensure_dir(os.path.dirname(args.output))
    df.to_csv(args.output, index=False)
    print(f"[qc] wrote {args.output} ({len(df)} rows)")

    if failures:
        print("[qc] strict_qc failures:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(2)


def _run_qc_if_missing(ctx: RunContext) -> str:
    """Ensure QC CSV exists in run output root; regenerate if missing."""
    qc_path = os.path.join(ctx.output_root, "qc", "qc_channels.csv")
    if os.path.exists(qc_path):
        return qc_path

    print(f"[features] QC CSV missing at {qc_path}. Regenerating QC before features.")
    spec, standardized_cfg = _normalize_spec_for_processing(ctx.run_cfg)
    extra_required_channels = _required_algorithmic_signal_channels(ctx.run_cfg)
    standardized_channel_db_path = str(ctx.run_cfg.get("standardized_channel_db_path") or "").strip()
    standardized_unit_map = load_standardized_channel_unit_map(standardized_channel_db_path)
    if standardized_channel_db_path and not standardized_unit_map:
        print(f"[features] warning: standardized_channel_db_path provided but no unit mappings found: {standardized_channel_db_path}")
    strict = bool(ctx.run_cfg.get("strict_qc", False))

    rows = []
    failures = []
    for item in ctx.scope:
        ds = item["dataset_id"]
        dep = item["deployment_id"]
        pkl_path = _data_pkl_path(ctx, ds, dep)
        existing_pairs = []
        existing_all_channels = []
        missing_required_pairs = []
        missing_optional_pairs = []
        unit_info = {
            "channel_units": "",
            "missing_netcdf_unit_channels": "",
            "unit_metadata_mismatch_channels": "",
            "expected_unit_mismatch_channels": "",
            "inconsistent_signal_units": "",
        }
        issues = []

        if not os.path.exists(pkl_path):
            issues.append("missing_data_pkl")
            for full_key, scfg in spec.items():
                if scfg.get("required", True):
                    missing_required_pairs.append(full_key)
                else:
                    missing_optional_pairs.append(full_key)
        else:
            try:
                data_pkl = _load_data_pkl(pkl_path)
                patch_info = _patch_netcdf_channel_metadata_from_data_pkl(
                    ctx=ctx,
                    dataset_id=ds,
                    deployment_id=dep,
                    data_pkl=data_pkl,
                )
                if patch_info.get("written"):
                    print(
                        f"[features->qc] patched NetCDF metadata for {ds}/{dep}: "
                        f"+{patch_info.get('patched_attrs', 0)} attrs"
                    )
                signal_data = getattr(data_pkl, "signal_data", {}) or {}
                existing_all_channels = _list_existing_signal_channels(signal_data)
                for full_key, scfg in spec.items():
                    if _has_required_source_for_feature(signal_data, full_key, scfg, standardized_cfg):
                        existing_pairs.append(full_key)
                    else:
                        if scfg.get("required", True):
                            missing_required_pairs.append(full_key)
                        else:
                            missing_optional_pairs.append(full_key)
                if missing_required_pairs:
                    issues.append("missing_required_channels")
                extra_existing, extra_missing = _check_explicit_required_channels(signal_data, extra_required_channels)
                existing_pairs.extend([x for x in extra_existing if x not in existing_pairs])
                for full_key in extra_missing:
                    if full_key not in missing_required_pairs:
                        missing_required_pairs.append(full_key)
                if extra_missing and "missing_required_channels" not in issues:
                    issues.append("missing_required_channels")
                if missing_optional_pairs:
                    issues.append("missing_optional_channels")
                unit_info = _evaluate_required_channel_units(
                    ctx=ctx,
                    dataset_id=ds,
                    deployment_id=dep,
                    data_pkl=data_pkl,
                    spec=spec,
                    standardized_cfg=standardized_cfg,
                    extra_required_channels=extra_required_channels,
                    standardized_unit_map=standardized_unit_map,
                )
                if unit_info["missing_netcdf_unit_channels"]:
                    issues.append("missing_netcdf_unit_metadata")
                if unit_info["unit_metadata_mismatch_channels"]:
                    issues.append("unit_metadata_mismatch")
                if unit_info["expected_unit_mismatch_channels"]:
                    issues.append("expected_unit_mismatch")
                if unit_info["inconsistent_signal_units"]:
                    issues.append("inconsistent_signal_units")
            except Exception as e:
                issues.append(f"load_error:{type(e).__name__}")
                for full_key, scfg in spec.items():
                    if scfg.get("required", True):
                        missing_required_pairs.append(full_key)
                    else:
                        missing_optional_pairs.append(full_key)

        unit_fail_bits = []
        if unit_info["missing_netcdf_unit_channels"]:
            unit_fail_bits.append(f"missing_netcdf_units={unit_info['missing_netcdf_unit_channels']}")
        if unit_info["unit_metadata_mismatch_channels"]:
            unit_fail_bits.append(f"metadata_mismatch={unit_info['unit_metadata_mismatch_channels']}")
        if unit_info["expected_unit_mismatch_channels"]:
            unit_fail_bits.append(f"expected_unit_mismatch={unit_info['expected_unit_mismatch_channels']}")
        if unit_info["inconsistent_signal_units"]:
            unit_fail_bits.append(f"inconsistent_units={unit_info['inconsistent_signal_units']}")
        if strict and (missing_required_pairs or unit_fail_bits):
            fail_parts = []
            if missing_required_pairs:
                fail_parts.append(",".join(missing_required_pairs))
            fail_parts.extend(unit_fail_bits)
            failures.append(f"{ds}/{dep}: " + " | ".join(fail_parts))

        rows.append({
            "dataset_id": ds,
            "deployment_id": dep,
            "existing_channels": ",".join(sorted(existing_pairs)),
            "missing_channels": ",".join(sorted(missing_required_pairs)),
            "optional_missing_channels": ",".join(sorted(missing_optional_pairs)),
            "channel_units": unit_info["channel_units"],
            "missing_netcdf_unit_channels": unit_info["missing_netcdf_unit_channels"],
            "unit_metadata_mismatch_channels": unit_info["unit_metadata_mismatch_channels"],
            "expected_unit_mismatch_channels": unit_info["expected_unit_mismatch_channels"],
            "inconsistent_signal_units": unit_info["inconsistent_signal_units"],
            "issues": ",".join(issues),
        })
        if not missing_required_pairs:
            print(
                f"[features->qc] ✅ {ds}/{dep} required channels found "
                f"({len(existing_pairs)} configured channels present)"
            )
        else:
            missing_with_sources = _format_missing_channels_with_sources(
                missing_required_pairs,
                spec,
                standardized_cfg,
            )
            print(
                f"[features->qc] ⚠️ {ds}/{dep} missing required channels: "
                f"{', '.join(missing_with_sources)}"
            )
            if existing_all_channels:
                print(
                    f"[features->qc]     available channels in data.pkl ({len(existing_all_channels)}): "
                    f"{', '.join(existing_all_channels)}"
                )
            else:
                print("[features->qc]     available channels in data.pkl (0): <none>")
        if unit_info["inconsistent_signal_units"]:
            print(f"[features->qc] ⚠️ {ds}/{dep} inconsistent units by signal: {unit_info['inconsistent_signal_units']}")
        if unit_info["missing_netcdf_unit_channels"]:
            print(f"[features->qc] ⚠️ {ds}/{dep} missing unit metadata in NetCDF: {unit_info['missing_netcdf_unit_channels']}")
        if unit_info["unit_metadata_mismatch_channels"]:
            print(f"[features->qc] ⚠️ {ds}/{dep} data.pkl vs NetCDF unit mismatch: {unit_info['unit_metadata_mismatch_channels']}")
        if unit_info["expected_unit_mismatch_channels"]:
            print(f"[features->qc] ⚠️ {ds}/{dep} channel units differ from standardized db: {unit_info['expected_unit_mismatch_channels']}")

    _ensure_dir(os.path.dirname(qc_path))
    pd.DataFrame(rows).to_csv(qc_path, index=False)
    print(f"[features] regenerated QC CSV: {qc_path}")

    if failures:
        print("[features] strict_qc failures after QC regeneration:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(2)

    return qc_path


def _feature_paths(ctx: RunContext) -> Tuple[str, str]:
    return (
        os.path.join(ctx.output_root, "features", "features_raw.parquet"),
        os.path.join(ctx.output_root, "features", "feature_index.parquet"),
    )


def _write_csv_debug_outputs(run_cfg: Dict) -> bool:
    return bool(run_cfg.get("write_csv_debug_outputs", False))


def _keep_raw_merged_features(run_cfg: Dict) -> bool:
    return bool(run_cfg.get("keep_raw_merged_features", False))


def _feature_deployment_paths(ctx: RunContext, dataset_id: str, deployment_id: str) -> Tuple[str, str]:
    safe_ds = _safe_slug(dataset_id)
    safe_dep = _safe_slug(deployment_id)
    return (
        os.path.join(ctx.output_root, "features", "by_deployment", f"{safe_ds}__{safe_dep}__features.parquet"),
        os.path.join(ctx.output_root, "features", "by_deployment", f"{safe_ds}__{safe_dep}__index.parquet"),
    )


def _select_base_timeline_signal(signal_data: Dict, spec: Dict, primary_channel: str) -> str:
    # Prefer the primary signal if it exists and has datetime, otherwise fallback
    # to the first configured signal with a valid datetime column.
    p_sig = str(primary_channel).split(".", 1)[0] if primary_channel else ""
    if p_sig in signal_data and "datetime" in signal_data[p_sig].columns:
        return p_sig
    for scfg in spec.values():
        sig = scfg["signal_id"]
        if sig in signal_data and "datetime" in signal_data[sig].columns:
            return sig
    return ""


def _resolve_spec_source_for_qc(full_key: str, scfg: Dict, standardized_cfg: Dict) -> Tuple[str, str]:
    for cfg in standardized_cfg.values():
        if scfg["signal_id"] != cfg["output_prefix"]:
            continue
        base = cfg["channel_id"]
        expected = {f"{base}_std", f"{base}_d1_std", f"{base}_d2_std"}
        if scfg["channel_id"] in expected:
            return cfg["signal_id"], cfg["channel_id"]
    return scfg["signal_id"], scfg["channel_id"]


def _has_required_source_for_feature(
    signal_data: Dict,
    full_key: str,
    scfg: Dict,
    standardized_cfg: Dict,
) -> bool:
    for cfg in standardized_cfg.values():
        if scfg["signal_id"] != cfg["output_prefix"]:
            continue
        base = cfg["channel_id"]
        expected = {f"{base}_std", f"{base}_d1_std", f"{base}_d2_std"}
        if scfg["channel_id"] in expected:
            _resolved_key, sig, ch = _resolve_available_source(
                signal_data,
                cfg["source_key"],
                cfg.get("fallback_source_keys"),
            )
            return bool(sig and ch)
    sig, ch = scfg["signal_id"], scfg["channel_id"]
    return sig in signal_data and ch in signal_data[sig].columns


def _list_existing_signal_channels(signal_data: Dict) -> List[str]:
    out = []
    for sig, sdf in sorted(signal_data.items()):
        cols = list(getattr(sdf, "columns", []))
        for ch in cols:
            if ch == "datetime":
                continue
            out.append(f"{sig}.{ch}")
    return sorted(out)


def _format_missing_channels_with_sources(missing_keys: List[str], spec: Dict, standardized_cfg: Dict) -> List[str]:
    formatted = []
    for full_key in sorted(missing_keys):
        scfg = spec.get(full_key)
        if not scfg:
            formatted.append(full_key)
            continue
        src_sig, src_ch = _resolve_spec_source_for_qc(full_key, scfg, standardized_cfg)
        source_key = f"{src_sig}.{src_ch}"
        fallback_display = []
        for cfg in standardized_cfg.values():
            if scfg["signal_id"] == cfg["output_prefix"]:
                base = cfg["channel_id"]
                expected = {f"{base}_std", f"{base}_d1_std", f"{base}_d2_std"}
                if scfg["channel_id"] in expected:
                    fallback_display = list(cfg.get("fallback_source_keys") or [])
                    break
        if source_key != full_key and fallback_display:
            formatted.append(f"{full_key} (source: {source_key}; fallback: {', '.join(fallback_display)})")
        elif source_key != full_key:
            formatted.append(f"{full_key} (source: {source_key})")
        else:
            formatted.append(full_key)
    return formatted


def _check_explicit_required_channels(
    signal_data: Dict,
    required_channels: List[str],
) -> Tuple[List[str], List[str]]:
    existing = []
    missing = []
    all_existing = set(_list_existing_signal_channels(signal_data))
    for full_key in sorted(set(required_channels or [])):
        if full_key in all_existing:
            existing.append(full_key)
        else:
            missing.append(full_key)
    return existing, missing


def _load_netcdf_attrs(netcdf_path: str | None) -> Dict[str, Any]:
    if not netcdf_path or not os.path.exists(netcdf_path):
        return {}
    try:
        import xarray as xr
        with xr.open_dataset(netcdf_path) as ds:
            return dict(ds.attrs)
    except Exception:
        return {}


def _as_channel_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in list(value)]
    except Exception:
        return [str(value)]


def _is_missing_metadata_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "none", "nan", "null", "na", "n/a", "unknown"}


def _patch_netcdf_channel_metadata_from_data_pkl(
    ctx: RunContext,
    dataset_id: str,
    deployment_id: str,
    data_pkl: Any,
) -> Dict[str, Any]:
    """
    Backfill missing channel-level NetCDF attrs from data.pkl signal_info metadata.
    """
    deployment_folder = os.path.join(ctx.data_root, str(dataset_id), str(deployment_id))
    netcdf_path = latest_processing_netcdf_path(deployment_folder, str(deployment_id))
    if not netcdf_path or not os.path.exists(netcdf_path):
        return {"path": netcdf_path or "", "patched_attrs": 0, "written": False}

    signal_info = getattr(data_pkl, "signal_info", {}) or {}
    try:
        import xarray as xr
        with xr.open_dataset(netcdf_path) as ds:
            ds2 = ds.load()
    except Exception:
        return {"path": netcdf_path, "patched_attrs": 0, "written": False}

    attrs = dict(ds2.attrs)
    patched = 0
    for var_name in list(ds2.data_vars):
        var_text = str(var_name)
        if not var_text.startswith("signal_data_"):
            continue
        signal_id = var_text.replace("signal_data_", "", 1)
        sig_info = signal_info.get(signal_id) if isinstance(signal_info, dict) else {}
        sig_info = sig_info if isinstance(sig_info, dict) else {}
        meta_map = sig_info.get("metadata") if isinstance(sig_info.get("metadata"), dict) else {}

        channels = _as_channel_list(ds2[var_name].attrs.get("variables")) or _as_channel_list(ds2[var_name].attrs.get("variable"))
        for channel_id in channels:
            ch_meta = meta_map.get(channel_id) if isinstance(meta_map, dict) else {}
            ch_meta = ch_meta if isinstance(ch_meta, dict) else {}
            base = f"signal_info_{signal_id}_metadata_{channel_id}_"
            candidates = {
                "unit": ch_meta.get("unit") or sig_info.get("units") or sig_info.get("original_units"),
                "standardized_unit": ch_meta.get("standardized_unit") or sig_info.get("standardized_unit"),
                "original_name": ch_meta.get("original_name") or channel_id,
                "parent_signal": ch_meta.get("parent_signal") or signal_id,
                "label": ch_meta.get("label"),
                "description": ch_meta.get("description"),
            }
            for key, value in candidates.items():
                if _is_missing_metadata_value(value):
                    continue
                attr_key = f"{base}{key}"
                if _is_missing_metadata_value(attrs.get(attr_key)):
                    attrs[attr_key] = value
                    patched += 1

    if patched <= 0:
        return {"path": netcdf_path, "patched_attrs": 0, "written": False}

    print(
        f"[qc] warning: {dataset_id}/{deployment_id} NetCDF missing {patched} channel metadata attr(s) "
        f"(would be backfilled from data.pkl — run outside QC to patch)"
    )
    return {"path": netcdf_path, "patched_attrs": int(patched), "written": False}


def _channel_units_from_sources(
    signal_info: Dict[str, Any],
    netcdf_attrs: Dict[str, Any],
    signal_id: str,
    channel_id: str,
) -> Dict[str, str]:
    sig_info = signal_info.get(signal_id) if isinstance(signal_info, dict) else {}
    sig_info = sig_info if isinstance(sig_info, dict) else {}
    meta = sig_info.get("metadata") if isinstance(sig_info.get("metadata"), dict) else {}
    ch_meta = meta.get(channel_id) if isinstance(meta, dict) else {}
    ch_meta = ch_meta if isinstance(ch_meta, dict) else {}

    pkl_unit = normalize_unit_token(ch_meta.get("unit") or sig_info.get("units"))
    pkl_std_unit = normalize_unit_token(ch_meta.get("standardized_unit"))

    nc_unit = normalize_unit_token(
        netcdf_attrs.get(f"signal_info_{signal_id}_metadata_{channel_id}_unit")
        or netcdf_attrs.get(f"signal_info_{signal_id}_units")
    )
    nc_std_unit = normalize_unit_token(
        netcdf_attrs.get(f"signal_info_{signal_id}_metadata_{channel_id}_standardized_unit")
    )

    effective_pkl = pkl_std_unit or pkl_unit
    effective_nc = nc_std_unit or nc_unit
    effective = effective_nc or effective_pkl

    return {
        "pkl_unit": pkl_unit,
        "pkl_std_unit": pkl_std_unit,
        "nc_unit": nc_unit,
        "nc_std_unit": nc_std_unit,
        "effective_pkl": effective_pkl,
        "effective_nc": effective_nc,
        "effective": effective,
    }


def _evaluate_required_channel_units(
    ctx: RunContext,
    dataset_id: str,
    deployment_id: str,
    data_pkl: Any,
    spec: Dict,
    standardized_cfg: Dict,
    extra_required_channels: List[str],
    standardized_unit_map: Dict[Tuple[str, str], str] | None = None,
) -> Dict[str, str]:
    configured_source_pairs: Dict[Tuple[str, str], None] = {}
    for full_key, scfg in (spec or {}).items():
        src_sig, src_ch = _resolve_spec_source_for_qc(full_key, scfg, standardized_cfg)
        if src_sig and src_ch:
            configured_source_pairs[(str(src_sig), str(src_ch))] = None
    for full_key in (extra_required_channels or []):
        if "." not in str(full_key):
            continue
        sig, ch = str(full_key).split(".", 1)
        configured_source_pairs[(sig, ch)] = None

    deployment_folder = os.path.join(ctx.data_root, str(dataset_id), str(deployment_id))
    netcdf_path = latest_processing_netcdf_path(deployment_folder, str(deployment_id))
    netcdf_attrs = _load_netcdf_attrs(netcdf_path)
    signal_info = getattr(data_pkl, "signal_info", {}) or {}

    by_signal: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
    missing_netcdf_units: List[str] = []
    pkl_nc_mismatch: List[str] = []
    expected_unit_mismatch: List[str] = []
    channel_summaries: List[str] = []

    for signal_id, channel_id in sorted(configured_source_pairs.keys()):
        signal_df = ((getattr(data_pkl, "signal_data", {}) or {}).get(signal_id))
        if signal_df is None or not hasattr(signal_df, "columns") or channel_id not in signal_df.columns:
            continue
        units = _channel_units_from_sources(signal_info, netcdf_attrs, signal_id, channel_id)
        by_signal.setdefault(signal_id, []).append((channel_id, units))
        if not units["effective_nc"]:
            missing_netcdf_units.append(f"{signal_id}.{channel_id}")
        if units["effective_pkl"] and units["effective_nc"] and units["effective_pkl"] != units["effective_nc"]:
            pkl_nc_mismatch.append(
                f"{signal_id}.{channel_id}({units['effective_pkl']}!=nc:{units['effective_nc']})"
            )
        expected_unit = ""
        if isinstance(standardized_unit_map, dict):
            expected_unit = str(standardized_unit_map.get((str(signal_id).lower(), str(channel_id).lower()), "")).strip()
        if expected_unit and units["effective"] and units["effective"] != expected_unit:
            expected_unit_mismatch.append(
                f"{signal_id}.{channel_id}({units['effective']}!=expected:{expected_unit})"
            )
        unit_text = units["effective"] or "missing"
        channel_summaries.append(f"{signal_id}.{channel_id}:{unit_text}")

    inconsistent_signals: List[str] = []
    for signal_id, rows in sorted(by_signal.items()):
        distinct = sorted({u["effective"] for _, u in rows if u.get("effective")})
        if len(distinct) > 1:
            detail = ",".join(f"{ch}={u['effective'] or 'missing'}" for ch, u in rows)
            inconsistent_signals.append(f"{signal_id}[{detail}]")

    return {
        "channel_units": ";".join(channel_summaries),
        "missing_netcdf_unit_channels": ",".join(sorted(missing_netcdf_units)),
        "unit_metadata_mismatch_channels": ",".join(sorted(pkl_nc_mismatch)),
        "expected_unit_mismatch_channels": ",".join(sorted(expected_unit_mismatch)),
        "inconsistent_signal_units": ",".join(inconsistent_signals),
    }


def _write_preprocessing_diagnostics(
    ctx: RunContext,
    dataset_id: str,
    deployment_id: str,
    diagnostics_rows: List[Dict],
    plot_payloads: List[Dict],
) -> None:
    if not diagnostics_rows and not plot_payloads:
        return
    diag_path, plot_path = _preprocessing_diag_paths(ctx, dataset_id, deployment_id)
    _ensure_dir(os.path.dirname(diag_path))
    payload = {
        "dataset_id": dataset_id,
        "deployment_id": deployment_id,
        "diagnostics": diagnostics_rows,
    }
    with open(diag_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    if not plot_payloads:
        return

    fig = go.Figure()
    for row in plot_payloads:
        fig.add_trace(
            go.Scattergl(
                x=row["datetime"],
                y=row["values"],
                mode="lines",
                name=row["name"],
            )
        )
    fig.update_layout(
        title=f"Standardization diagnostics: {dataset_id}/{deployment_id}",
        template="plotly_white",
        xaxis_title="datetime",
        yaxis_title="value",
    )
    _ensure_dir(os.path.dirname(plot_path))
    fig.write_html(plot_path)


def _build_standardized_channel_frame(
    sdf: pd.DataFrame,
    dt: pd.Series,
    channel_id: str,
    cfg: Dict,
) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    vals = pd.to_numeric(sdf[channel_id], errors="coerce").to_numpy(dtype=float)
    valid = (~pd.isna(dt).to_numpy()) & np.isfinite(vals)
    if not np.any(valid):
        return pd.DataFrame(columns=["datetime"]), {
            "source_key": cfg["source_key"],
            "method": cfg["method"],
            "status": "no_valid_rows",
        }, []

    work_dt = dt[valid].reset_index(drop=True)
    work_vals = vals[valid]
    interval_s = _infer_sampling_interval_seconds(work_dt)
    resolution = _infer_channel_resolution(work_vals)
    use_coarse = (
        (np.isfinite(interval_s) and interval_s >= cfg["coarse_interval_threshold_s"])
        or (np.isfinite(resolution) and resolution >= cfg["coarse_resolution_threshold"])
    )
    smooth_seconds = cfg["coarse_smooth_seconds"] if use_coarse else cfg["base_smooth_seconds"]
    sigma_samples = smooth_seconds / interval_s if np.isfinite(interval_s) and interval_s > 0 else 0.0

    smoothed = _gaussian_smooth_series(work_vals, sigma_samples)
    quantize_step = cfg.get("quantize_step")
    if quantize_step and np.isfinite(quantize_step) and quantize_step > 0:
        quantized = np.round(smoothed / quantize_step) * quantize_step
    else:
        quantized = smoothed.copy()
    standardized = _gaussian_smooth_series(quantized, sigma_samples)

    out = pd.DataFrame({"datetime": work_dt})
    plot_payloads = [
        {"datetime": work_dt, "values": work_vals, "name": f"{cfg['source_key']} raw"},
        {"datetime": work_dt, "values": standardized, "name": f"{cfg['output_prefix']}.{channel_id}_std"},
    ]
    for order in cfg["derivative_orders"]:
        if order == 0:
            out[f"{channel_id}_std"] = standardized
            continue
        deriv = standardized.copy()
        for _ in range(order):
            if deriv.size <= 1 or not np.isfinite(interval_s) or interval_s <= 0:
                deriv = np.full(deriv.shape, np.nan, dtype=float)
                break
            deriv = np.concatenate([np.diff(deriv) / interval_s, [np.nan]])
        col = f"{channel_id}_d{order}_std"
        out[col] = deriv
        plot_payloads.append({"datetime": work_dt, "values": deriv, "name": f"{cfg['output_prefix']}.{col}"})

    diag = {
        "source_key": cfg["source_key"],
        "method": cfg["method"],
        "status": "ok",
        "sampling_interval_s": interval_s,
        "sampling_frequency_hz": float(1.0 / interval_s) if np.isfinite(interval_s) and interval_s > 0 else float("nan"),
        "observed_resolution": resolution,
        "chosen_smoothing_seconds": float(smooth_seconds),
        "quantize_step": quantize_step,
        "derived_outputs": [c for c in out.columns if c != "datetime"],
        "rows_in": int(len(sdf)),
        "rows_used": int(len(out)),
    }
    return out, diag, plot_payloads


def _standardize_signal_data(
    signal_data: Dict,
    standardized_cfg: Dict,
    deployment_tz: str,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    synthetic = {}
    diagnostics_rows = []
    plot_payloads = []
    for source_key, cfg in standardized_cfg.items():
        resolved_source_key, sig, ch = _resolve_available_source(
            signal_data,
            cfg["source_key"],
            cfg.get("fallback_source_keys"),
        )
        if not sig:
            diagnostics_rows.append({
                "source_key": source_key,
                "resolved_source_key": resolved_source_key,
                "fallback_source_keys": list(cfg.get("fallback_source_keys") or []),
                "method": cfg["method"],
                "status": "missing_signal",
            })
            continue
        sdf = signal_data[sig]
        if "datetime" not in sdf.columns or ch not in sdf.columns:
            diagnostics_rows.append({
                "source_key": source_key,
                "resolved_source_key": resolved_source_key,
                "fallback_source_keys": list(cfg.get("fallback_source_keys") or []),
                "method": cfg["method"],
                "status": "missing_channel",
            })
            continue
        dt = _normalize_datetime_series_to_timezone(sdf["datetime"], deployment_tz)
        cfg_local = dict(cfg)
        cfg_local["source_key"] = resolved_source_key
        cfg_local["signal_id"] = sig
        cfg_local["channel_id"] = ch
        std_df, diag, plot_rows = _build_standardized_channel_frame(sdf, dt, ch, cfg_local)
        diagnostics_rows.append(diag)
        if std_df.empty:
            continue
        synthetic[cfg["output_prefix"]] = std_df
        plot_payloads.extend(plot_rows)
    return synthetic, diagnostics_rows, plot_payloads


def _validate_channel_sampling_compatibility(
    signal_data: Dict,
    spec: Dict,
    standardized_cfg: Dict,
    deployment_tz: str,
    tolerance,
    dataset_id: str,
    deployment_id: str,
) -> List[Dict]:
    if tolerance is None:
        return []
    standardized_sources = set(standardized_cfg.keys())
    rows = []
    for full_key, scfg in spec.items():
        if full_key in standardized_sources or scfg["signal_id"] in {cfg["output_prefix"] for cfg in standardized_cfg.values()}:
            continue
        sig = scfg["signal_id"]
        ch = scfg["channel_id"]
        if sig not in signal_data:
            continue
        sdf = signal_data[sig]
        if "datetime" not in sdf.columns or ch not in sdf.columns:
            continue
        dt = _normalize_datetime_series_to_timezone(sdf["datetime"], deployment_tz)
        interval_s = _infer_sampling_interval_seconds(dt)
        rows.append({
            "type": "sampling_compatibility",
            "channel": full_key,
            "sampling_interval_s": interval_s,
            "sampling_frequency_hz": float(1.0 / interval_s) if np.isfinite(interval_s) and interval_s > 0 else float("nan"),
        })
    finite_intervals = [r["sampling_interval_s"] for r in rows if np.isfinite(r["sampling_interval_s"])]
    if len(finite_intervals) > 1:
        ref = finite_intervals[0]
        for row in rows[1:]:
            if _relative_interval_diff(ref, row["sampling_interval_s"]) > tolerance:
                details = ", ".join(
                    f"{r['channel']}={r['sampling_interval_s']:.6g}s/{r['sampling_frequency_hz']:.6g}Hz"
                    for r in rows
                    if np.isfinite(r["sampling_interval_s"])
                )
                raise ValueError(
                    f"Incompatible non-standardized channel sampling for {dataset_id}/{deployment_id}: "
                    f"{details}. Standardize the mismatched channels or remove them from the run."
                )
    return rows


def _build_algorithmic_source_frame(
    signal_data: Dict,
    deployment_tz: str,
    algo_cfg: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    resolved_source_key, sig, ch = _resolve_available_source(
        signal_data,
        algo_cfg["source_key"],
        algo_cfg.get("fallback_source_keys"),
    )
    if not sig:
        fallback = ", ".join(algo_cfg.get("fallback_source_keys") or [])
        if fallback:
            raise ValueError(
                f"Algorithmic segment source signal missing: {algo_cfg['source_key']} "
                f"(fallbacks tried: {fallback})"
            )
        raise ValueError(f"Algorithmic segment source signal missing: {algo_cfg['source_key']}")
    sdf = signal_data[sig]
    if "datetime" not in sdf.columns or ch not in sdf.columns:
        raise ValueError(f"Algorithmic segment source column missing: {resolved_source_key}")

    dt = _normalize_datetime_series_to_timezone(sdf["datetime"], deployment_tz)
    base_vals = pd.to_numeric(sdf[ch], errors="coerce").to_numpy(dtype=float)
    valid = (~pd.isna(dt).to_numpy()) & np.isfinite(base_vals)
    if not np.any(valid):
        raise ValueError(f"No valid rows found for algorithmic segment source: {resolved_source_key}")

    out = pd.DataFrame({"datetime": dt[valid].reset_index(drop=True)})
    source_vals = base_vals[valid]
    interval_s = _infer_sampling_interval_seconds(out["datetime"])
    resolution = _infer_channel_resolution(source_vals)

    if algo_cfg.get("standardize", True):
        std_df, diag, _ = _build_standardized_channel_frame(
            sdf=sdf[[c for c in ["datetime", ch] if c in sdf.columns]].copy(),
            dt=dt,
            channel_id=ch,
            cfg={
                "source_key": resolved_source_key,
                "signal_id": sig,
                "channel_id": ch,
                "method": algo_cfg["method"],
                "quantize_step": algo_cfg["quantize_step"],
                "base_smooth_seconds": algo_cfg["base_smooth_seconds"],
                "coarse_smooth_seconds": algo_cfg["coarse_smooth_seconds"],
                "coarse_interval_threshold_s": algo_cfg["coarse_interval_threshold_s"],
                "coarse_resolution_threshold": algo_cfg["coarse_resolution_threshold"],
                "derivative_orders": [0, 1, 2],
                "output_prefix": "algorithmic_standardized",
            },
        )
        if std_df.empty:
            raise ValueError(f"Algorithmic standardization produced no rows for {resolved_source_key}")
        out = std_df.rename(
            columns={
                f"{ch}_std": "depth_std_m",
                f"{ch}_d1_std": "depth_d1_ms",
                f"{ch}_d2_std": "depth_d2_ms2",
            }
        )
        diag["source_mode"] = "standardized"
    else:
        work_vals = source_vals.copy()
        d1 = np.concatenate([np.diff(work_vals) / interval_s, [np.nan]]) if np.isfinite(interval_s) and interval_s > 0 and work_vals.size > 1 else np.full(work_vals.shape, np.nan)
        d2 = np.concatenate([np.diff(d1[:-1]) / interval_s, [np.nan, np.nan]]) if np.isfinite(interval_s) and interval_s > 0 and work_vals.size > 2 else np.full(work_vals.shape, np.nan)
        out["depth_std_m"] = work_vals
        out["depth_d1_ms"] = d1
        out["depth_d2_ms2"] = d2
        diag = {
            "source_key": resolved_source_key,
            "requested_source_key": algo_cfg["source_key"],
            "source_mode": "raw",
            "sampling_interval_s": interval_s,
            "sampling_frequency_hz": float(1.0 / interval_s) if np.isfinite(interval_s) and interval_s > 0 else float("nan"),
            "observed_resolution": resolution,
            "rows_used": int(len(out)),
        }

    if isinstance(diag, dict):
        diag.setdefault("requested_source_key", algo_cfg["source_key"])
        diag["source_key"] = resolved_source_key

    out["depth_d1_next_ms"] = out["depth_d1_ms"].shift(-1)
    out[["depth_std_m", "depth_d1_ms", "depth_d2_ms2", "depth_d1_next_ms"]] = out[
        ["depth_std_m", "depth_d1_ms", "depth_d2_ms2", "depth_d1_next_ms"]
    ].apply(pd.to_numeric, errors="coerce")
    return out.reset_index(drop=True), diag


def _segment_window_mean(arr: np.ndarray, start_idx: int, end_idx: int, frac0: float, frac1: float) -> float:
    length = max(1, int(end_idx - start_idx + 1))
    i0 = start_idx + int(np.floor(length * frac0))
    i1 = start_idx + int(np.floor(length * frac1))
    i0 = max(start_idx, min(i0, end_idx))
    i1 = max(i0, min(i1, end_idx))
    return float(np.nanmean(arr[i0:i1 + 1]))


def _align_stroke_to_sample_times(
    sample_times: pd.Series,
    stroke_df: pd.DataFrame | None,
    stroke_value_col: str | None,
) -> pd.Series:
    sample_times = pd.to_datetime(sample_times, errors="coerce")
    if getattr(sample_times.dt, "tz", None) is not None:
        sample_times = sample_times.dt.tz_convert("UTC").dt.tz_localize(None)
    if (
        stroke_df is None
        or stroke_value_col is None
        or stroke_df.empty
        or "datetime" not in stroke_df.columns
    ):
        return pd.Series(np.nan, index=sample_times.index, dtype=float)
    src = stroke_df[["datetime", stroke_value_col]].copy()
    src["datetime"] = pd.to_datetime(src["datetime"], errors="coerce")
    if getattr(src["datetime"].dt, "tz", None) is not None:
        src["datetime"] = src["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    src[stroke_value_col] = pd.to_numeric(src[stroke_value_col], errors="coerce")
    src = src.dropna(subset=["datetime"]).sort_values("datetime")
    if src.empty:
        return pd.Series(np.nan, index=sample_times.index, dtype=float)
    target = pd.DataFrame(
        {
            "__idx": np.arange(len(sample_times), dtype=int),
            "datetime": sample_times.values,
        }
    ).dropna(subset=["datetime"])
    if target.empty:
        return pd.Series(np.nan, index=sample_times.index, dtype=float)
    diffs = pd.to_numeric(sample_times.sort_values().diff().dt.total_seconds(), errors="coerce")
    sample_dt_s = float(diffs[diffs > 0].median()) if (diffs > 0).any() else np.nan
    tol_s = max(1.0, 1.5 * sample_dt_s) if np.isfinite(sample_dt_s) and sample_dt_s > 0 else None
    merged = pd.merge_asof(
        target.sort_values("datetime"),
        src.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=(pd.to_timedelta(tol_s, unit="s") if tol_s is not None else None),
    )
    out = pd.Series(np.nan, index=sample_times.index, dtype=float)
    out.iloc[merged["__idx"].astype(int).to_numpy()] = pd.to_numeric(
        merged[stroke_value_col],
        errors="coerce",
    ).to_numpy(dtype=float)
    return out


def _build_exhaustive_segments_from_label_timeseries(
    label_ts: pd.DataFrame,
    *,
    dataset_id: str,
    deployment_id: str,
    source_key: str,
    segment_method: str,
) -> pd.DataFrame:
    if label_ts is None or label_ts.empty:
        return pd.DataFrame()
    ts = label_ts.copy().reset_index(drop=True)
    ts["datetime"] = pd.to_datetime(ts["datetime"], errors="coerce")
    ts = ts.dropna(subset=["datetime"]).reset_index(drop=True)
    if ts.empty:
        return pd.DataFrame()
    ts["__row_idx"] = np.arange(len(ts), dtype=int)
    change = (
        ts["algorithmic_label_name"].astype(str).ne(ts["algorithmic_label_name"].astype(str).shift(1))
        | ts["algorithmic_label_code"].astype("Int64").ne(ts["algorithmic_label_code"].astype("Int64").shift(1))
        | ts["algorithmic_segment_key"].astype(str).ne(ts["algorithmic_segment_key"].astype(str).shift(1))
    )
    ts["__segment_id"] = change.fillna(True).cumsum().astype(int)
    diffs = pd.to_numeric(ts["datetime"].diff().dt.total_seconds(), errors="coerce")
    median_dt_s = float(diffs[diffs > 0].median()) if (diffs > 0).any() else np.nan
    grouped = ts.groupby("__segment_id", as_index=False).agg(
        start_index=("__row_idx", "min"),
        end_index=("__row_idx", "max"),
        start_datetime=("datetime", "min"),
        end_datetime=("datetime", "max"),
        label_code=("algorithmic_label_code", "last"),
        label_name=("algorithmic_label_name", "last"),
        key=("algorithmic_segment_key", "last"),
        nominal_class=("algorithmic_nominal_class", "last"),
    )
    base_duration = (
        pd.to_datetime(grouped["end_datetime"], errors="coerce")
        - pd.to_datetime(grouped["start_datetime"], errors="coerce")
    ).dt.total_seconds().astype(float)
    if np.isfinite(median_dt_s) and median_dt_s > 0:
        grouped["duration_s"] = np.maximum(base_duration + float(median_dt_s), 0.0)
    else:
        grouped["duration_s"] = np.maximum(base_duration, 0.0)
    grouped["dataset_id"] = dataset_id
    grouped["deployment_id"] = deployment_id
    grouped["source_key"] = source_key
    grouped["segment_method"] = segment_method
    grouped["segment_rank"] = np.arange(1, len(grouped) + 1, dtype=int)
    grouped["segment_family"] = "exhaustive_partition"
    grouped["segment_sign"] = "exhaustive"
    grouped["base_nominal_class"] = grouped["nominal_class"]
    grouped["base_keep_filtered"] = True
    grouped["keep_filtered"] = True
    grouped["context_keep"] = True
    grouped["context_filter_applied"] = False
    grouped["context_filter_ids"] = ""
    grouped["context_reject_reason"] = pd.NA
    grouped["base_state_key"] = grouped["key"].astype(str)
    grouped["elapsed_days"] = (
        pd.to_datetime(grouped["start_datetime"], errors="coerce")
        - pd.to_datetime(grouped["start_datetime"], errors="coerce").min()
    ).dt.total_seconds() / 86400.0
    grouped["segment_midpoint_datetime"] = pd.to_datetime(grouped["start_datetime"], errors="coerce") + (
        pd.to_datetime(grouped["end_datetime"], errors="coerce")
        - pd.to_datetime(grouped["start_datetime"], errors="coerce")
    ) / 2
    return grouped[
        [
            "dataset_id",
            "deployment_id",
            "source_key",
            "segment_method",
            "segment_rank",
            "start_index",
            "end_index",
            "start_datetime",
            "end_datetime",
            "duration_s",
            "segment_midpoint_datetime",
            "elapsed_days",
            "segment_family",
            "segment_sign",
            "nominal_class",
            "base_nominal_class",
            "base_keep_filtered",
            "keep_filtered",
            "context_keep",
            "context_filter_applied",
            "context_filter_ids",
            "context_reject_reason",
            "base_state_key",
            "label_code",
            "label_name",
            "key",
        ]
    ].rename(columns={"key": "algorithmic_segment_key"})


def _compute_algorithmic_segments_for_deployment(
    ctx: RunContext,
    dataset_id: str,
    deployment_id: str,
    context_filter_pass_mode: str | None = None,
    data_pkl: Any | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
    if context_filter_pass_mode is not None:
        algo_cfg["context_filter_pass_mode"] = str(context_filter_pass_mode).strip().lower()
    if not algo_cfg.get("enabled"):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"enabled": False, "status": "disabled"}

    pkl_path = _data_pkl_path(ctx, dataset_id, deployment_id)
    if data_pkl is None:
        data_pkl = _load_data_pkl(pkl_path)
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    deployment_tz = _deployment_timezone_name(data_pkl)
    work_df, source_diag = _build_algorithmic_source_frame(signal_data, deployment_tz, algo_cfg)
    label_codes = algo_cfg["label_codes"]
    label_names = algo_cfg["label_names"]
    event_keys = algo_cfg["event_keys"]

    thr = algo_cfg["thresholds"]
    terminal_criteria = dict(algo_cfg.get("terminal_criteria") or {})
    surface_terminal_cfg = dict(terminal_criteria.get("surface_sleep") or {})
    flat_terminal_cfg = dict(terminal_criteria.get("long_flat") or {})
    drift_terminal_cfg = dict(terminal_criteria.get("long_drift") or {})
    unscorable_terminal_cfg = dict(terminal_criteria.get("unscorable") or {})
    band_candidate = (
        (work_df["depth_std_m"].abs() > thr["dive_depth_min_m"])
        & (work_df["depth_d1_ms"] >= thr["first_deriv_min_ms"])
        & (work_df["depth_d1_ms"] <= thr["first_deriv_max_ms"])
        & (work_df["depth_d2_ms2"] >= thr["second_deriv_min_ms2"])
        & (work_df["depth_d2_ms2"] <= thr["second_deriv_max_ms2"])
    )
    work_df["is_unfiltered_rest_candidate"] = band_candidate.fillna(False)
    candidate = band_candidate.copy()
    if algo_cfg.get("end_segments_upon_d1_sign_change", True):
        # Suppress boundary samples where the local slope flips so contiguous
        # candidate regions split at d1 sign changes.
        candidate = candidate & (np.sign(work_df["depth_d1_ms"]) == np.sign(work_df["depth_d1_next_ms"]))
    source_key_lower = str(algo_cfg.get("source_key") or "").strip().lower()
    source_is_reversed_axis = any(token in source_key_lower for token in ["depth", "pressure"])
    if algo_cfg.get("filter_out_ascent", True):
        candidate = candidate & ((work_df["depth_d1_ms"] >= 0) if source_is_reversed_axis else (work_df["depth_d1_ms"] <= 0))
    if algo_cfg.get("filter_out_descent", False):
        candidate = candidate & ((work_df["depth_d1_ms"] <= 0) if source_is_reversed_axis else (work_df["depth_d1_ms"] >= 0))
    work_df["is_signed_rest_candidate"] = candidate.fillna(False)
    work_df["is_drift_candidate"] = candidate.fillna(False)

    initial_segments = find_segments(
        data=work_df,
        column="is_drift_candidate",
        criteria=lambda x: bool(x),
        min_duration=thr["min_duration_s"],
        animal_id_col=None,
    )
    work_df["is_non_sleep_candidate"] = (~work_df["is_drift_candidate"].fillna(False)).astype(bool)
    non_candidate_segments = find_segments(
        data=work_df,
        column="is_non_sleep_candidate",
        criteria=lambda x: bool(x),
        min_duration=0.0,
        animal_id_col=None,
    )
    surface_mask = (work_df["depth_std_m"].abs() <= thr["dive_depth_min_m"]).fillna(False)
    work_df["is_surface_sleep_candidate"] = surface_mask
    surface_segments_all = find_segments(
        data=work_df,
        column="is_surface_sleep_candidate",
        criteria=lambda x: bool(x),
        min_duration=0.0,
        animal_id_col=None,
    )
    if not surface_segments_all.empty and bool(surface_terminal_cfg.get("uses_surface_sleep_min_duration", True)):
        surface_segments = surface_segments_all[
            pd.to_numeric(
                (surface_segments_all["end_datetime"] - surface_segments_all["start_datetime"]).dt.total_seconds(),
                errors="coerce",
            ) >= float(thr["surface_sleep_min_duration_s"])
        ].copy()
        active_surface_segments = surface_segments_all.drop(surface_segments.index, errors="ignore").copy()
    else:
        surface_segments = surface_segments_all.copy()
        active_surface_segments = pd.DataFrame(columns=surface_segments_all.columns if isinstance(surface_segments_all, pd.DataFrame) else [])
    if initial_segments.empty:
        initial_segments = pd.DataFrame(columns=["start_index", "end_index", "start_datetime", "end_datetime", "duration"])
    if non_candidate_segments.empty:
        non_candidate_segments = pd.DataFrame(columns=["start_index", "end_index", "start_datetime", "end_datetime", "duration"])

    initial_segments = initial_segments.copy()
    if not initial_segments.empty:
        initial_segments["dataset_id"] = dataset_id
        initial_segments["deployment_id"] = deployment_id
        initial_segments["source_key"] = algo_cfg["source_key"]
        initial_segments["segment_method"] = algo_cfg["method"]
        initial_segments["segment_rank"] = np.arange(1, len(initial_segments) + 1, dtype=int)

    depth_vals = work_df["depth_std_m"].to_numpy(dtype=float)
    d1_vals = work_df["depth_d1_ms"].to_numpy(dtype=float)
    d2_vals = work_df["depth_d2_ms2"].to_numpy(dtype=float)
    stroke_df = None
    stroke_value_col = None
    stroke_times_ns: np.ndarray | None = None
    stroke_cumsum: np.ndarray | None = None
    stroke_cumcount: np.ndarray | None = None
    if "stroke_rate" in signal_data:
        candidate_cols = [c for c in ["stroke_rate", "Stroke_Rate"] if c in signal_data["stroke_rate"].columns]
        if candidate_cols and "datetime" in signal_data["stroke_rate"].columns:
            stroke_df = signal_data["stroke_rate"][["datetime", candidate_cols[0]]].copy()
            stroke_df["datetime"] = _normalize_datetime_series_to_timezone(stroke_df["datetime"], deployment_tz)
            stroke_df[candidate_cols[0]] = pd.to_numeric(stroke_df[candidate_cols[0]], errors="coerce")
            stroke_df = stroke_df.dropna(subset=["datetime", candidate_cols[0]]).sort_values("datetime").reset_index(drop=True)
            stroke_value_col = candidate_cols[0]
            if not stroke_df.empty:
                stroke_vals = pd.to_numeric(stroke_df[stroke_value_col], errors="coerce").to_numpy(dtype=float)
                stroke_times_ns = pd.to_datetime(stroke_df["datetime"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
                valid = np.isfinite(stroke_vals)
                # Prefix sums for O(log n) range means via searchsorted.
                stroke_cumsum = np.concatenate(([0.0], np.cumsum(np.where(valid, stroke_vals, 0.0), dtype=float)))
                stroke_cumcount = np.concatenate(([0], np.cumsum(valid.astype(np.int64), dtype=np.int64)))
    light_df = None
    light_value_col = None
    light_times_ns: np.ndarray | None = None
    light_valid_cumcount: np.ndarray | None = None
    if "light" in signal_data:
        light_cols = [c for c in ["light", "Light", "lux", "Lux", "illuminance", "Illuminance"] if c in signal_data["light"].columns]
        if not light_cols:
            light_cols = [c for c in signal_data["light"].columns if c != "datetime"]
        if light_cols and "datetime" in signal_data["light"].columns:
            light_df = signal_data["light"][["datetime", light_cols[0]]].copy()
            light_df["datetime"] = _normalize_datetime_series_to_timezone(light_df["datetime"], deployment_tz)
            light_df[light_cols[0]] = pd.to_numeric(light_df[light_cols[0]], errors="coerce")
            light_df = light_df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            light_value_col = light_cols[0]
            if not light_df.empty:
                light_vals = pd.to_numeric(light_df[light_value_col], errors="coerce").to_numpy(dtype=float)
                light_times_ns = pd.to_datetime(light_df["datetime"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
                valid = np.isfinite(light_vals)
                light_valid_cumcount = np.concatenate(([0], np.cumsum(valid.astype(np.int64), dtype=np.int64)))

    def _segment_stroke_means(seg_df: pd.DataFrame) -> pd.Series:
        if seg_df is None or seg_df.empty:
            return pd.Series(dtype=float)
        if (
            stroke_df is None
            or stroke_value_col is None
            or stroke_df.empty
            or stroke_times_ns is None
            or stroke_cumsum is None
            or stroke_cumcount is None
        ):
            return pd.Series(np.nan, index=seg_df.index, dtype=float)
        starts = pd.to_datetime(seg_df["start_datetime"], errors="coerce")
        ends = pd.to_datetime(seg_df["end_datetime"], errors="coerce")
        out = np.full(len(seg_df), np.nan, dtype=float)
        valid = starts.notna().to_numpy() & ends.notna().to_numpy()
        if not np.any(valid):
            return pd.Series(out, index=seg_df.index, dtype=float)
        start_ns = starts[valid].to_numpy(dtype="datetime64[ns]").astype("int64")
        end_ns = ends[valid].to_numpy(dtype="datetime64[ns]").astype("int64")
        left = np.searchsorted(stroke_times_ns, start_ns, side="left")
        right = np.searchsorted(stroke_times_ns, end_ns, side="right")
        sums = stroke_cumsum[right] - stroke_cumsum[left]
        counts = stroke_cumcount[right] - stroke_cumcount[left]
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
        out[np.where(valid)[0]] = means
        return pd.Series(out, index=seg_df.index, dtype=float)

    def _segment_light_coverage(seg_df: pd.DataFrame) -> pd.DataFrame:
        if seg_df is None or seg_df.empty:
            return pd.DataFrame(
                {
                    "segment_light_samples_total": pd.Series(dtype="int64"),
                    "segment_light_samples_valid": pd.Series(dtype="int64"),
                    "segment_light_fraction_valid": pd.Series(dtype=float),
                }
            )
        if (
            light_df is None
            or light_value_col is None
            or light_df.empty
            or light_times_ns is None
            or light_valid_cumcount is None
        ):
            return pd.DataFrame(
                {
                    "segment_light_samples_total": pd.Series(0, index=seg_df.index, dtype="int64"),
                    "segment_light_samples_valid": pd.Series(0, index=seg_df.index, dtype="int64"),
                    "segment_light_fraction_valid": pd.Series(np.nan, index=seg_df.index, dtype=float),
                }
            )
        starts = pd.to_datetime(seg_df["start_datetime"], errors="coerce")
        ends = pd.to_datetime(seg_df["end_datetime"], errors="coerce")
        total = np.zeros(len(seg_df), dtype=np.int64)
        valid_count = np.zeros(len(seg_df), dtype=np.int64)
        frac = np.full(len(seg_df), np.nan, dtype=float)
        valid = starts.notna().to_numpy() & ends.notna().to_numpy()
        if np.any(valid):
            start_ns = starts[valid].to_numpy(dtype="datetime64[ns]").astype("int64")
            end_ns = ends[valid].to_numpy(dtype="datetime64[ns]").astype("int64")
            left = np.searchsorted(light_times_ns, start_ns, side="left")
            right = np.searchsorted(light_times_ns, end_ns, side="right")
            counts = (right - left).astype(np.int64)
            valids = (light_valid_cumcount[right] - light_valid_cumcount[left]).astype(np.int64)
            idx = np.where(valid)[0]
            total[idx] = counts
            valid_count[idx] = valids
            frac[idx] = np.divide(valids, counts, out=np.full_like(valids, np.nan, dtype=float), where=counts > 0)
        return pd.DataFrame(
            {
                "segment_light_samples_total": pd.Series(total, index=seg_df.index, dtype="int64"),
                "segment_light_samples_valid": pd.Series(valid_count, index=seg_df.index, dtype="int64"),
                "segment_light_fraction_valid": pd.Series(frac, index=seg_df.index, dtype=float),
            }
        )
    trip_start = pd.to_datetime(work_df["datetime"], errors="coerce").min()

    non_candidate_unscorable_segments = pd.DataFrame(columns=["start_index", "end_index", "start_datetime", "end_datetime", "duration_s"])
    if not non_candidate_segments.empty and bool(unscorable_terminal_cfg.get("enabled", False)):
        nc = non_candidate_segments.copy()
        nc["dataset_id"] = dataset_id
        nc["deployment_id"] = deployment_id
        nc["source_key"] = algo_cfg["source_key"]
        nc["segment_method"] = algo_cfg["method"]
        nc["segment_family"] = "non_candidate_gap"
        nc["segment_sign"] = "non_candidate"
        nc["duration_s"] = (nc["end_datetime"] - nc["start_datetime"]).dt.total_seconds()
        nc["segment_rank"] = np.arange(1, len(nc) + 1, dtype=int)
        nc["segment_midpoint_datetime"] = nc["start_datetime"] + (nc["end_datetime"] - nc["start_datetime"]) / 2
        nc["elapsed_days"] = (
            pd.to_datetime(nc["segment_midpoint_datetime"], errors="coerce") - trip_start
        ).dt.total_seconds() / 86400.0
        nc_start_idx = nc["start_index"].astype(int).to_numpy()
        nc_end_idx = nc["end_index"].astype(int).to_numpy()
        nc["mean_d1_ms"] = [float(np.nanmean(d1_vals[s:e + 1])) for s, e in zip(nc_start_idx, nc_end_idx)]
        nc["mean_d2_ms2"] = [float(np.nanmean(d2_vals[s:e + 1])) for s, e in zip(nc_start_idx, nc_end_idx)]
        nc["segment_mean_stroke_rate_spm"] = _segment_stroke_means(nc)
        nc_unscorable_mask = pd.Series(True, index=nc.index, dtype=bool)
        if bool(unscorable_terminal_cfg.get("uses_min_duration", True)):
            nc_unscorable_mask &= (
                nc["duration_s"] > float(thr["unscorable_interpolated_gap_min_duration_s"])
            )
        if bool(unscorable_terminal_cfg.get("uses_nonzero_mean_d1", True)):
            nc_unscorable_mask &= (
                nc["mean_d1_ms"].abs() > float(thr["unscorable_interpolated_gap_min_abs_mean_d1_ms"])
            )
        if bool(unscorable_terminal_cfg.get("uses_zero_mean_d2", True)):
            d2_tol = max(
                float(thr["unscorable_interpolated_gap_max_abs_mean_d2_ms2"]),
                float(thr["unscorable_interpolated_gap_abs_mean_d2_tolerance_ms2"]),
            )
            nc_unscorable_mask &= nc["mean_d2_ms2"].abs() <= d2_tol
        nc["is_unscorable_noncandidate"] = nc_unscorable_mask.fillna(False)
        nc = nc[nc["is_unscorable_noncandidate"]].copy()
        if not nc.empty:
            nc["nominal_class"] = "unscorable"
            nc["base_nominal_class"] = "unscorable"
            nc["base_keep_filtered"] = True
            nc["keep_filtered"] = True
            nc["context_keep"] = True
            nc["context_filter_applied"] = False
            nc["context_filter_ids"] = ""
            nc["context_reject_reason"] = pd.NA
            nc["base_state_key"] = _nominal_class_to_state_key("unscorable", event_keys)
            nc["label_code"] = int(label_codes["unscorable"])
            nc["label_name"] = str(label_names["unscorable"])
            non_candidate_unscorable_segments = nc
    if not initial_segments.empty:
        start_idx = initial_segments["start_index"].astype(int).to_numpy()
        end_idx = initial_segments["end_index"].astype(int).to_numpy()
        initial_segments["duration_s"] = (initial_segments["end_datetime"] - initial_segments["start_datetime"]).dt.total_seconds()
        initial_segments["segment_midpoint_datetime"] = initial_segments["start_datetime"] + (
            initial_segments["end_datetime"] - initial_segments["start_datetime"]
        ) / 2
        initial_segments["elapsed_days"] = (
            pd.to_datetime(initial_segments["segment_midpoint_datetime"], errors="coerce") - trip_start
        ).dt.total_seconds() / 86400.0
        initial_segments["start_depth_m"] = work_df.loc[start_idx, "depth_std_m"].to_numpy(dtype=float)
        initial_segments["end_depth_m"] = work_df.loc[end_idx, "depth_std_m"].to_numpy(dtype=float)
        initial_segments["mean_depth_m"] = [float(np.nanmean(depth_vals[s:e + 1])) for s, e in zip(start_idx, end_idx)]
        initial_segments["depth_std_m"] = [float(np.nanstd(depth_vals[s:e + 1])) for s, e in zip(start_idx, end_idx)]
        initial_segments["mean_d1_ms"] = [float(np.nanmean(d1_vals[s:e + 1])) for s, e in zip(start_idx, end_idx)]
        initial_segments["mean_d2_ms2"] = [float(np.nanmean(d2_vals[s:e + 1])) for s, e in zip(start_idx, end_idx)]
        initial_segments["sd_d1_ms"] = [float(np.nanstd(d1_vals[s:e + 1])) for s, e in zip(start_idx, end_idx)]
        initial_segments["sd_d2_ms2"] = [float(np.nanstd(d2_vals[s:e + 1])) for s, e in zip(start_idx, end_idx)]
        initial_segments["drift_rate_ms"] = (
            (initial_segments["start_depth_m"] - initial_segments["end_depth_m"])
            / initial_segments["duration_s"].replace(0, np.nan)
        )
        initial_segments["mean_d1_quart_to_third_ms"] = [
            _segment_window_mean(d1_vals, s, e, 0.25, 1.0 / 3.0) for s, e in zip(start_idx, end_idx)
        ]
        initial_segments["mean_d1_2third_to_3quart_ms"] = [
            _segment_window_mean(d1_vals, s, e, 2.0 / 3.0, 0.75) for s, e in zip(start_idx, end_idx)
        ]
        initial_segments["mean_d1_end_ms"] = [
            _segment_window_mean(d1_vals, s, e, 0.875, 0.8889) for s, e in zip(start_idx, end_idx)
        ]
        initial_segments["curvature_diff_ms"] = (
            initial_segments["mean_d1_2third_to_3quart_ms"] - initial_segments["mean_d1_quart_to_third_ms"]
        )
        initial_segments["segment_mean_stroke_rate_spm"] = _segment_stroke_means(initial_segments)
        initial_segments["is_long_flat"] = (
            initial_segments["mean_d1_end_ms"].abs() <= thr["end_flat_abs_mean_d1_max_ms"]
            if bool(flat_terminal_cfg.get("uses_end_flat_abs_mean_d1_max_ms", True))
            else False
        )
        initial_segments["passes_drift_rate_filter"] = (
            initial_segments["drift_rate_ms"].abs() <= thr["drift_rate_abs_max_ms"]
            if bool(drift_terminal_cfg.get("uses_drift_rate_abs_max_ms", True))
            else True
        )
        initial_segments["passes_curvature_filter"] = (
            initial_segments["curvature_diff_ms"].abs() <= thr["curvature_abs_max_ms"]
            if bool(drift_terminal_cfg.get("uses_curvature_abs_max_ms", True))
            else True
        )
        unscorable_mask = pd.Series(False, index=initial_segments.index, dtype=bool)
        if bool(unscorable_terminal_cfg.get("enabled", False)):
            unscorable_mask = pd.Series(True, index=initial_segments.index, dtype=bool)
            if bool(unscorable_terminal_cfg.get("uses_min_duration", True)):
                unscorable_mask &= (
                    initial_segments["duration_s"] > float(thr["unscorable_interpolated_gap_min_duration_s"])
                )
            if bool(unscorable_terminal_cfg.get("uses_nonzero_mean_d1", True)):
                unscorable_mask &= (
                    initial_segments["mean_d1_ms"].abs()
                    > float(thr["unscorable_interpolated_gap_min_abs_mean_d1_ms"])
                )
            if bool(unscorable_terminal_cfg.get("uses_zero_mean_d2", True)):
                d2_tol = max(
                    float(thr["unscorable_interpolated_gap_max_abs_mean_d2_ms2"]),
                    float(thr["unscorable_interpolated_gap_abs_mean_d2_tolerance_ms2"]),
                )
                unscorable_mask &= initial_segments["mean_d2_ms2"].abs() <= d2_tol
        initial_segments["is_unscorable_interpolated_gap"] = unscorable_mask.fillna(False)
        initial_segments["keep_filtered"] = (
            (initial_segments["passes_drift_rate_filter"] & initial_segments["passes_curvature_filter"])
            | initial_segments["is_long_flat"]
            | initial_segments["is_unscorable_interpolated_gap"]
        )
        initial_segments["base_nominal_class"] = np.where(
            initial_segments["is_unscorable_interpolated_gap"],
            "unscorable",
            np.where(
                initial_segments["keep_filtered"],
                np.where(initial_segments["is_long_flat"], "resting_benthic", "long_drift"),
                np.where(
                    pd.to_numeric(initial_segments["segment_mean_stroke_rate_spm"], errors="coerce")
                    > float(thr.get("stroke_rate_active_threshold_spm", 10.0)),
                    "active_uw_swimming",
                    "calm_uw_gliding",
                ),
            ),
        )
        initial_segments["base_keep_filtered"] = initial_segments["keep_filtered"].astype(bool)
        initial_segments["segment_family"] = "drift_candidate"
        initial_segments["segment_sign"] = np.where(
            pd.to_numeric(initial_segments["drift_rate_ms"], errors="coerce") > 0,
            "positive",
            np.where(
                pd.to_numeric(initial_segments["drift_rate_ms"], errors="coerce") < 0,
                "negative",
                "neutral",
            ),
        )
        initial_segments["base_state_key"] = initial_segments["base_nominal_class"].map(
            lambda x: _nominal_class_to_state_key(x, event_keys)
        )
        initial_segments["nominal_class"] = initial_segments["base_nominal_class"]
        initial_segments["context_keep"] = initial_segments["base_keep_filtered"]
        initial_segments["context_filter_applied"] = False
        initial_segments["context_filter_ids"] = ""
        initial_segments["context_reject_reason"] = pd.NA
    else:
        initial_segments = pd.DataFrame(columns=[
            "dataset_id", "deployment_id", "source_key", "segment_method", "segment_rank", "start_index", "end_index",
            "start_datetime", "end_datetime", "duration_s", "nominal_class", "label_code", "label_name"
        ])

    def _finalize_surface_segments(seg: pd.DataFrame, nominal_class: str | None, rank_offset: int = 0) -> pd.DataFrame:
        seg = seg.copy()
        if seg.empty:
            return seg
        seg["dataset_id"] = dataset_id
        seg["deployment_id"] = deployment_id
        seg["source_key"] = algo_cfg["source_key"]
        seg["segment_method"] = algo_cfg["method"]
        seg["segment_rank"] = np.arange(1 + int(rank_offset), len(seg) + 1 + int(rank_offset), dtype=int)
        seg["duration_s"] = (seg["end_datetime"] - seg["start_datetime"]).dt.total_seconds()
        trip_start = pd.to_datetime(work_df["datetime"], errors="coerce").min()
        seg["segment_midpoint_datetime"] = seg["start_datetime"] + (seg["end_datetime"] - seg["start_datetime"]) / 2
        seg["elapsed_days"] = (
            pd.to_datetime(seg["segment_midpoint_datetime"], errors="coerce") - trip_start
        ).dt.total_seconds() / 86400.0
        seg["segment_mean_stroke_rate_spm"] = _segment_stroke_means(seg)
        if nominal_class:
            seg["nominal_class"] = nominal_class
        else:
            seg["nominal_class"] = np.where(
                pd.to_numeric(seg["segment_mean_stroke_rate_spm"], errors="coerce")
                > float(thr.get("stroke_rate_active_threshold_spm", 10.0)),
                "active_surface_swimming",
                "calm_surface_gliding",
            )
        seg["base_nominal_class"] = seg["nominal_class"]
        seg["base_keep_filtered"] = True
        seg["segment_family"] = "surface_interval"
        seg["segment_sign"] = "surface"
        seg["base_state_key"] = seg["nominal_class"].map(
            lambda x: _nominal_class_to_state_key(str(x), event_keys)
        )
        seg["context_keep"] = True
        seg["context_filter_applied"] = False
        seg["context_filter_ids"] = ""
        seg["context_reject_reason"] = pd.NA
        seg["label_code"] = (
            seg["nominal_class"].map(label_codes).fillna(label_codes["not_sleep"]).astype(int)
        )
        seg["label_name"] = seg["nominal_class"].map(label_names).fillna(label_names["not_sleep"])
        return seg

    surface_segments = _finalize_surface_segments(surface_segments, "resting_surface", rank_offset=0)
    active_surface_segments = _finalize_surface_segments(
        active_surface_segments,
        None,
        rank_offset=len(surface_segments),
    )
    surface_unscorable_segments = pd.DataFrame(columns=surface_segments.columns if isinstance(surface_segments, pd.DataFrame) else [])
    light_missing_scope_ok = True
    allowed_light_dataset_ids = {
        str(x).strip()
        for x in list(unscorable_terminal_cfg.get("surface_light_missing_dataset_ids") or [])
        if str(x).strip()
    }
    allowed_light_deployment_ids = {
        str(x).strip()
        for x in list(unscorable_terminal_cfg.get("surface_light_missing_deployment_ids") or [])
        if str(x).strip()
    }
    if allowed_light_dataset_ids and str(dataset_id) not in allowed_light_dataset_ids:
        light_missing_scope_ok = False
    if allowed_light_deployment_ids and str(deployment_id) not in allowed_light_deployment_ids:
        light_missing_scope_ok = False

    missing_min_fraction = float(
        pd.to_numeric(
            unscorable_terminal_cfg.get("surface_light_missing_min_fraction", 1.0),
            errors="coerce",
        )
    )
    if not np.isfinite(missing_min_fraction):
        missing_min_fraction = 1.0
    missing_min_fraction = float(np.clip(missing_min_fraction, 0.0, 1.0))

    light_missing_enabled = (
        bool(unscorable_terminal_cfg.get("enabled", False))
        and bool(unscorable_terminal_cfg.get("uses_surface_light_missing", False))
        and light_missing_scope_ok
        and light_df is not None
        and light_value_col is not None
    )

    if (
        not surface_segments.empty
        and light_missing_enabled
    ):
        light_cov = _segment_light_coverage(surface_segments)
        surface_segments["segment_light_samples_total"] = light_cov["segment_light_samples_total"]
        surface_segments["segment_light_samples_valid"] = light_cov["segment_light_samples_valid"]
        surface_segments["segment_light_fraction_valid"] = light_cov["segment_light_fraction_valid"]
        valid_fraction = pd.to_numeric(surface_segments["segment_light_fraction_valid"], errors="coerce")
        total_samples = pd.to_numeric(surface_segments["segment_light_samples_total"], errors="coerce")
        missing_fraction = 1.0 - valid_fraction
        missing_fraction = missing_fraction.where(total_samples > 0, 1.0)
        surface_segments["segment_light_fraction_missing"] = missing_fraction
        surface_segments["is_unscorable_surface_light_missing"] = (
            pd.to_numeric(surface_segments["segment_light_fraction_missing"], errors="coerce")
            .fillna(1.0)
            >= missing_min_fraction
        )
        surface_unscorable_segments = surface_segments[surface_segments["is_unscorable_surface_light_missing"]].copy()
        surface_segments = surface_segments[~surface_segments["is_unscorable_surface_light_missing"]].copy()
        if not surface_unscorable_segments.empty:
            surface_unscorable_segments["nominal_class"] = "unscorable"
            surface_unscorable_segments["base_nominal_class"] = "unscorable"
            surface_unscorable_segments["base_keep_filtered"] = True
            surface_unscorable_segments["keep_filtered"] = True
            surface_unscorable_segments["context_keep"] = True
            surface_unscorable_segments["context_filter_applied"] = False
            surface_unscorable_segments["context_filter_ids"] = ""
            surface_unscorable_segments["context_reject_reason"] = pd.NA
            surface_unscorable_segments["base_state_key"] = _nominal_class_to_state_key("unscorable", event_keys)
            surface_unscorable_segments["label_code"] = int(label_codes["unscorable"])
            surface_unscorable_segments["label_name"] = str(label_names["unscorable"])

    pass_mode = str(algo_cfg.get("context_filter_pass_mode") or "measured_only").strip().lower()
    if pass_mode not in {"none", "full", "measured_only"}:
        raise ValueError(
            f"Unsupported context_filter_pass_mode '{pass_mode}'. "
            "Expected one of: none, measured_only, full."
        )

    if not initial_segments.empty:
        initial_segments["post_measured_keep"] = initial_segments["base_keep_filtered"].fillna(False).astype(bool)
        covariate_meta = {}
        if pass_mode != "none":
            initial_segments, measured_covariate_meta = annotate_segment_covariates(
                initial_segments,
                sample_df=work_df,
                runtime_params=algo_cfg,
                deployment_context={
                    "dataset_id": dataset_id,
                    "deployment_id": deployment_id,
                    "signal_data": signal_data,
                    "deployment_tz": deployment_tz,
                    "data_pkl": data_pkl,
                },
                filter_stage="measured",
            )
            initial_segments = apply_context_filters(
                initial_segments,
                algo_cfg.get("context_filters") or [],
                measured_covariate_meta,
                filter_stage="measured",
            )
            covariate_meta = dict(measured_covariate_meta)
            initial_segments, inferred_covariate_meta = annotate_segment_covariates(
                initial_segments,
                sample_df=work_df,
                runtime_params=algo_cfg,
                deployment_context={
                    "dataset_id": dataset_id,
                    "deployment_id": deployment_id,
                    "signal_data": signal_data,
                    "deployment_tz": deployment_tz,
                    "data_pkl": data_pkl,
                },
                filter_stage="inferred",
            )
            covariate_meta.update(inferred_covariate_meta)
            if pass_mode == "full":
                initial_segments = apply_context_filters(
                    initial_segments,
                    algo_cfg.get("context_filters") or [],
                    covariate_meta,
                    filter_stage="inferred",
                )
        initial_segments["context_keep"] = initial_segments["context_keep"].fillna(False)
        initial_segments["keep_filtered"] = initial_segments["base_keep_filtered"] & initial_segments["context_keep"]
        initial_segments["nominal_class"] = np.where(
            initial_segments["keep_filtered"],
            initial_segments["base_nominal_class"],
            np.where(
                pd.to_numeric(initial_segments["segment_mean_stroke_rate_spm"], errors="coerce")
                > float(thr.get("stroke_rate_active_threshold_spm", 10.0)),
                "active_uw_swimming",
                "calm_uw_gliding",
            ),
        )
        initial_segments["label_code"] = initial_segments["nominal_class"].map(label_codes)
        initial_segments["label_name"] = initial_segments["nominal_class"].map(label_names)
    else:
        covariate_meta = {}

    benthic_drift_unscorable_count = 0
    if light_missing_enabled and not initial_segments.empty:
        target_initial_mask = initial_segments.get("nominal_class", pd.Series(dtype=object)).astype(str).isin(
            {"resting_benthic", "long_drift", "resting_drift"}
        )
        target_initial_idx = initial_segments.index[target_initial_mask]
        if len(target_initial_idx) > 0:
            target_initial = initial_segments.loc[target_initial_idx].copy()
            light_cov = _segment_light_coverage(target_initial)
            initial_segments.loc[target_initial_idx, "segment_light_samples_total"] = light_cov["segment_light_samples_total"].to_numpy()
            initial_segments.loc[target_initial_idx, "segment_light_samples_valid"] = light_cov["segment_light_samples_valid"].to_numpy()
            initial_segments.loc[target_initial_idx, "segment_light_fraction_valid"] = light_cov["segment_light_fraction_valid"].to_numpy()
            valid_fraction = pd.to_numeric(initial_segments.loc[target_initial_idx, "segment_light_fraction_valid"], errors="coerce")
            total_samples = pd.to_numeric(initial_segments.loc[target_initial_idx, "segment_light_samples_total"], errors="coerce")
            missing_fraction = 1.0 - valid_fraction
            missing_fraction = missing_fraction.where(total_samples > 0, 1.0)
            initial_segments.loc[target_initial_idx, "segment_light_fraction_missing"] = missing_fraction.to_numpy()
            initial_segments.loc[target_initial_idx, "is_unscorable_benthic_drift_light_missing"] = (
                pd.to_numeric(initial_segments.loc[target_initial_idx, "segment_light_fraction_missing"], errors="coerce")
                .fillna(1.0)
                >= missing_min_fraction
            ).to_numpy()
            to_unscore = target_initial_idx[
                pd.to_numeric(
                    initial_segments.loc[target_initial_idx, "is_unscorable_benthic_drift_light_missing"],
                    errors="coerce",
                ).fillna(0).astype(int) > 0
            ]
            if len(to_unscore) > 0:
                benthic_drift_unscorable_count = int(len(to_unscore))
                initial_segments.loc[to_unscore, "nominal_class"] = "unscorable"
                initial_segments.loc[to_unscore, "base_nominal_class"] = "unscorable"
                initial_segments.loc[to_unscore, "base_state_key"] = _nominal_class_to_state_key("unscorable", event_keys)
                initial_segments.loc[to_unscore, "label_code"] = int(label_codes["unscorable"])
                initial_segments.loc[to_unscore, "label_name"] = str(label_names["unscorable"])

    event_rows = []
    for _, row in initial_segments.iterrows():
        event_rows.append({
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "key": event_keys["initial"],
            "segment_stage": "initial",
            "datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            "duration": row["duration_s"],
            "segment_rank": int(row["segment_rank"]),
            "source_key": algo_cfg["source_key"],
        })
        if bool(row.get("base_keep_filtered", False)):
            event_rows.append({
                "dataset_id": dataset_id,
                "deployment_id": deployment_id,
                "key": event_keys["filtered"],
                "segment_stage": "filtered",
                "datetime": row["start_datetime"],
                "end_datetime": row["end_datetime"],
                "duration": row["duration_s"],
                "segment_rank": int(row["segment_rank"]),
                "source_key": algo_cfg["source_key"],
            })
        if str(row.get("nominal_class", "")) in {"long_flat", "resting_benthic"}:
            event_rows.append({
                "dataset_id": dataset_id,
                "deployment_id": deployment_id,
                "key": event_keys.get("resting_benthic", event_keys["long_flat"]),
                "segment_stage": "flat",
                "datetime": row["start_datetime"],
                "end_datetime": row["end_datetime"],
                "duration": row["duration_s"],
                "segment_rank": int(row["segment_rank"]),
                "source_key": algo_cfg["source_key"],
                "label_code": label_codes.get("resting_benthic", label_codes["long_flat"]),
                "label_name": label_names.get("resting_benthic", label_names["long_flat"]),
            })
        elif str(row.get("nominal_class", "")) == "long_drift":
            event_rows.append({
                "dataset_id": dataset_id,
                "deployment_id": deployment_id,
                "key": event_keys["long_drift"],
                "segment_stage": "filtered_nominal",
                "datetime": row["start_datetime"],
                "end_datetime": row["end_datetime"],
                "duration": row["duration_s"],
                "segment_rank": int(row["segment_rank"]),
                "source_key": algo_cfg["source_key"],
                "label_code": label_codes["long_drift"],
                "label_name": label_names["long_drift"],
            })
        elif str(row.get("nominal_class", "")) == "unscorable":
            event_rows.append({
                "dataset_id": dataset_id,
                "deployment_id": deployment_id,
                "key": event_keys["unscorable"],
                "segment_stage": "filtered_nominal",
                "datetime": row["start_datetime"],
                "end_datetime": row["end_datetime"],
                "duration": row["duration_s"],
                "segment_rank": int(row["segment_rank"]),
                "source_key": algo_cfg["source_key"],
                "label_code": label_codes["unscorable"],
                "label_name": label_names["unscorable"],
            })
        elif str(row.get("nominal_class", "")) in {"not_sleep", "calm_uw_gliding", "active_uw_swimming"}:
            nominal_class = str(row.get("nominal_class", "not_sleep"))
            event_rows.append({
                "dataset_id": dataset_id,
                "deployment_id": deployment_id,
                "key": event_keys.get(nominal_class, event_keys["not_sleep"]),
                "segment_stage": "filtered_nominal",
                "datetime": row["start_datetime"],
                "end_datetime": row["end_datetime"],
                "duration": row["duration_s"],
                "segment_rank": int(row["segment_rank"]),
                "source_key": algo_cfg["source_key"],
                "label_code": label_codes.get(nominal_class, label_codes["not_sleep"]),
                "label_name": label_names.get(nominal_class, label_names["not_sleep"]),
            })
    for _, row in surface_segments.iterrows():
        event_rows.append({
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "key": event_keys.get("resting_surface", event_keys["surface_sleep"]),
            "segment_stage": "surface_sleep",
            "datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            "duration": row["duration_s"],
            "segment_rank": int(row["segment_rank"]),
            "source_key": algo_cfg["source_key"],
            "label_code": label_codes.get("resting_surface", label_codes["surface_sleep"]),
            "label_name": label_names.get("resting_surface", label_names["surface_sleep"]),
        })
    for _, row in active_surface_segments.iterrows():
        event_rows.append({
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "key": event_keys.get(str(row.get("nominal_class", "")), event_keys["active_surface"]),
            "segment_stage": "surface_active",
            "datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            "duration": row["duration_s"],
            "segment_rank": int(row["segment_rank"]),
            "source_key": algo_cfg["source_key"],
            "label_code": label_codes.get(str(row.get("nominal_class", "")), label_codes.get("active_surface", label_codes["not_sleep"])),
            "label_name": label_names.get(str(row.get("nominal_class", "")), label_names.get("active_surface", label_names["not_sleep"])),
        })
    for _, row in non_candidate_unscorable_segments.iterrows():
        event_rows.append({
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "key": event_keys["unscorable"],
            "segment_stage": "noncandidate_unscorable",
            "datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            "duration": row["duration_s"],
            "segment_rank": int(row.get("segment_rank", 0)),
            "source_key": algo_cfg["source_key"],
            "label_code": label_codes["unscorable"],
            "label_name": label_names["unscorable"],
        })
    for _, row in surface_unscorable_segments.iterrows():
        event_rows.append({
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "key": event_keys["unscorable"],
            "segment_stage": "surface_unscorable",
            "datetime": row["start_datetime"],
            "end_datetime": row["end_datetime"],
            "duration": row["duration_s"],
            "segment_rank": int(row.get("segment_rank", 0)),
            "source_key": algo_cfg["source_key"],
            "label_code": label_codes["unscorable"],
            "label_name": label_names["unscorable"],
        })

    event_columns = [
        "dataset_id",
        "deployment_id",
        "key",
        "segment_stage",
        "datetime",
        "end_datetime",
        "duration",
        "segment_rank",
        "source_key",
        "label_code",
        "label_name",
    ]
    event_df = pd.DataFrame(event_rows)
    if event_df.empty:
        event_df = pd.DataFrame(columns=event_columns)
    else:
        event_df = event_df.sort_values(["datetime", "key"]).reset_index(drop=True)

    stroke_at_sample = _align_stroke_to_sample_times(
        pd.to_datetime(work_df["datetime"], errors="coerce"),
        stroke_df,
        stroke_value_col,
    )
    is_surface_sample = pd.to_numeric(work_df["depth_std_m"], errors="coerce").abs() <= float(thr["dive_depth_min_m"])
    is_active_sample = pd.to_numeric(stroke_at_sample, errors="coerce") > float(thr.get("stroke_rate_active_threshold_spm", 10.0))
    stroke_missing_sample = pd.to_numeric(stroke_at_sample, errors="coerce").isna()
    fallback_nominal = np.where(
        stroke_missing_sample,
        "unscorable",
        np.where(
            is_surface_sample,
            np.where(is_active_sample, "active_surface_swimming", "calm_surface_gliding"),
            np.where(is_active_sample, "active_uw_swimming", "calm_uw_gliding"),
        ),
    )

    label_ts = work_df[["datetime"]].copy()
    label_ts["dataset_id"] = dataset_id
    label_ts["deployment_id"] = deployment_id
    label_ts["source_key"] = algo_cfg["source_key"]
    label_ts["algorithmic_method"] = algo_cfg["method_name"]
    label_ts["algorithmic_nominal_class"] = pd.Series(fallback_nominal, index=label_ts.index, dtype=object)
    label_ts["algorithmic_label_code"] = label_ts["algorithmic_nominal_class"].map(label_codes).fillna(label_codes["not_sleep"]).astype(int)
    label_ts["algorithmic_label_name"] = label_ts["algorithmic_nominal_class"].map(label_names).fillna(label_names["not_sleep"]).astype(str)
    label_ts["algorithmic_segment_key"] = label_ts["algorithmic_nominal_class"].map(event_keys).fillna(event_keys.get("not_sleep", label_names["not_sleep"])).astype(str)
    if not surface_segments.empty:
        for _, row in surface_segments.iterrows():
            s = int(row["start_index"])
            e = int(row["end_index"])
            label_ts.loc[s:e, "algorithmic_nominal_class"] = "resting_surface"
            label_ts.loc[s:e, "algorithmic_label_code"] = label_codes.get("resting_surface", label_codes["surface_sleep"])
            label_ts.loc[s:e, "algorithmic_label_name"] = label_names.get("resting_surface", label_names["surface_sleep"])
            label_ts.loc[s:e, "algorithmic_segment_key"] = event_keys.get("resting_surface", event_keys["surface_sleep"])
    if not active_surface_segments.empty:
        for _, row in active_surface_segments.iterrows():
            s = int(row["start_index"])
            e = int(row["end_index"])
            nominal = str(row.get("nominal_class") or "active_surface")
            label_ts.loc[s:e, "algorithmic_nominal_class"] = nominal
            label_ts.loc[s:e, "algorithmic_label_code"] = label_codes.get(
                nominal,
                label_codes.get("active_surface", label_codes["not_sleep"]),
            )
            label_ts.loc[s:e, "algorithmic_label_name"] = label_names.get(
                nominal,
                label_names.get("active_surface", label_names["not_sleep"]),
            )
            label_ts.loc[s:e, "algorithmic_segment_key"] = event_keys.get(
                nominal,
                event_keys.get("active_surface", event_keys.get("not_sleep", "")),
            )
    if not initial_segments.empty:
        kept = initial_segments[initial_segments["keep_filtered"]].copy()
        for _, row in kept.iterrows():
            s = int(row["start_index"])
            e = int(row["end_index"])
            nominal = str(row["nominal_class"])
            label_ts.loc[s:e, "algorithmic_nominal_class"] = nominal
            label_ts.loc[s:e, "algorithmic_label_code"] = label_codes[nominal]
            label_ts.loc[s:e, "algorithmic_label_name"] = label_names[nominal]
            label_ts.loc[s:e, "algorithmic_segment_key"] = event_keys[nominal]
    if not non_candidate_unscorable_segments.empty:
        for _, row in non_candidate_unscorable_segments.iterrows():
            s = int(row["start_index"])
            e = int(row["end_index"])
            label_ts.loc[s:e, "algorithmic_nominal_class"] = "unscorable"
            label_ts.loc[s:e, "algorithmic_label_code"] = label_codes["unscorable"]
            label_ts.loc[s:e, "algorithmic_label_name"] = label_names["unscorable"]
            label_ts.loc[s:e, "algorithmic_segment_key"] = event_keys["unscorable"]
    if not surface_unscorable_segments.empty:
        for _, row in surface_unscorable_segments.iterrows():
            s = int(row["start_index"])
            e = int(row["end_index"])
            label_ts.loc[s:e, "algorithmic_nominal_class"] = "unscorable"
            label_ts.loc[s:e, "algorithmic_label_code"] = label_codes["unscorable"]
            label_ts.loc[s:e, "algorithmic_label_name"] = label_names["unscorable"]
            label_ts.loc[s:e, "algorithmic_segment_key"] = event_keys["unscorable"]

    exhaustive_segments = _build_exhaustive_segments_from_label_timeseries(
        label_ts,
        dataset_id=dataset_id,
        deployment_id=deployment_id,
        source_key=algo_cfg["source_key"],
        segment_method=algo_cfg["method"],
    )
    if not exhaustive_segments.empty:
        exhaustive_segments["base_nominal_class"] = exhaustive_segments["nominal_class"]
        exhaustive_segments["label_code"] = pd.to_numeric(exhaustive_segments["label_code"], errors="coerce").fillna(label_codes["not_sleep"]).astype(int)
        exhaustive_segments["label_name"] = exhaustive_segments["label_name"].astype(str)

    intermediate_df = work_df[
        [
            c
            for c in [
                "datetime",
                "depth_std_m",
                "depth_d1_ms",
                "depth_d2_ms2",
                "depth_d1_next_ms",
                "is_unfiltered_rest_candidate",
                "is_signed_rest_candidate",
                "is_drift_candidate",
                "is_surface_sleep_candidate",
            ]
            if c in work_df.columns
        ]
    ].copy()

    summary = {
        "enabled": True,
        "status": "ok",
        "context_filter_pass_mode": pass_mode,
        "dataset_id": dataset_id,
        "deployment_id": deployment_id,
        "source_diagnostics": source_diag,
        "thresholds": thr,
        "source_key": algo_cfg["source_key"],
        "method_name": algo_cfg["method_name"],
        "n_initial_segments": int(len(initial_segments)),
        "n_filtered_segments": int(initial_segments["keep_filtered"].sum()) if "keep_filtered" in initial_segments.columns else 0,
        "n_base_filtered_segments": int(initial_segments["base_keep_filtered"].sum()) if "base_keep_filtered" in initial_segments.columns else 0,
        "n_context_rejected_segments": int(
            ((initial_segments["base_keep_filtered"].fillna(False)) & (~initial_segments["context_keep"].fillna(False))).sum()
        ) if {"base_keep_filtered", "context_keep"} <= set(initial_segments.columns) else 0,
        "n_long_flat_segments": int(initial_segments["nominal_class"].isin(["long_flat", "resting_benthic"]).sum()) if "nominal_class" in initial_segments.columns else 0,
        "n_resting_benthic_segments": int((initial_segments["nominal_class"] == "resting_benthic").sum()) if "nominal_class" in initial_segments.columns else 0,
        "n_long_drift_segments": int((initial_segments["nominal_class"] == "long_drift").sum()) if "nominal_class" in initial_segments.columns else 0,
        "n_active_diving_segments": int(
            initial_segments["nominal_class"].isin(["not_sleep", "calm_uw_gliding", "active_uw_swimming"]).sum()
        ) if "nominal_class" in initial_segments.columns else 0,
        "n_calm_uw_gliding_segments": int((initial_segments["nominal_class"] == "calm_uw_gliding").sum()) if "nominal_class" in initial_segments.columns else 0,
        "n_active_uw_swimming_segments": int((initial_segments["nominal_class"] == "active_uw_swimming").sum()) if "nominal_class" in initial_segments.columns else 0,
        "n_unscorable_segments": (
            int((initial_segments["nominal_class"] == "unscorable").sum()) if "nominal_class" in initial_segments.columns else 0
        ) + int(len(non_candidate_unscorable_segments)) + int(len(surface_unscorable_segments)),
        "n_unscorable_noncandidate_segments": int(len(non_candidate_unscorable_segments)),
        "n_unscorable_surface_segments": int(len(surface_unscorable_segments)),
        "n_unscorable_benthic_drift_segments": int(benthic_drift_unscorable_count),
        "n_surface_sleep_segments": int(len(surface_segments)),
        "n_resting_surface_segments": int(len(surface_segments)),
        "n_active_surface_segments": int(len(active_surface_segments)),
        "n_calm_surface_gliding_segments": int((active_surface_segments.get("nominal_class", pd.Series(dtype=object)) == "calm_surface_gliding").sum()),
        "n_active_surface_swimming_segments": int((active_surface_segments.get("nominal_class", pd.Series(dtype=object)) == "active_surface_swimming").sum()),
        "event_keys": event_keys,
        "label_codes": label_codes,
        "label_names": label_names,
        "covariate_fields": covariate_meta,
    }
    exhaustive_class_keys = [
        "active_uw_swimming",
        "calm_uw_gliding",
        "resting_surface",
        "resting_benthic",
        "long_drift",
        "active_surface_swimming",
        "calm_surface_gliding",
        "unscorable",
    ]
    if exhaustive_segments is not None and not exhaustive_segments.empty:
        summary["n_exhaustive_segments"] = int(len(exhaustive_segments))
        summary["exhaustive_total_duration_s"] = float(
            pd.to_numeric(exhaustive_segments.get("duration_s", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        )
        for class_key in exhaustive_class_keys:
            class_mask = exhaustive_segments.get("nominal_class", pd.Series(dtype=object)).astype(str) == class_key
            summary[f"n_exhaustive_{class_key}_segments"] = int(class_mask.sum())
            summary[f"exhaustive_{class_key}_duration_s"] = float(
                pd.to_numeric(exhaustive_segments.loc[class_mask, "duration_s"], errors="coerce").fillna(0.0).sum()
            )
    else:
        summary["n_exhaustive_segments"] = 0
        summary["exhaustive_total_duration_s"] = 0.0
        for class_key in exhaustive_class_keys:
            summary[f"n_exhaustive_{class_key}_segments"] = 0
            summary[f"exhaustive_{class_key}_duration_s"] = 0.0

    _concat_parts = [df for df in [initial_segments, surface_segments, active_surface_segments, non_candidate_unscorable_segments, surface_unscorable_segments] if not df.empty]
    combined_segments = pd.concat(
        _concat_parts if _concat_parts else [initial_segments],
        ignore_index=True,
        sort=False,
    )
    return combined_segments, exhaustive_segments, event_df, label_ts, intermediate_df, summary


def _annotate_windows_with_algorithmic_segments(cdf: pd.DataFrame, event_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if event_df is None or event_df.empty:
        return cdf, pd.DataFrame()

    work = cdf.copy()
    work["window_start"] = pd.to_datetime(work["window_start"], errors="coerce")
    work["window_end"] = pd.to_datetime(work["window_end"], errors="coerce")
    seg = event_df.copy()
    seg["datetime"] = pd.to_datetime(seg["datetime"], errors="coerce")
    seg["end_datetime"] = pd.to_datetime(seg["end_datetime"], errors="coerce")
    seg = seg.dropna(subset=["datetime", "end_datetime"])
    if seg.empty:
        return work, pd.DataFrame()

    summary_rows = []
    for key in sorted(seg["key"].dropna().astype(str).unique().tolist()):
        slug = _safe_slug(key)
        work[f"algorithmic_overlap_frac__{slug}"] = 0.0
        work[f"algorithmic_overlap__{slug}"] = False

    grouped_windows = work.groupby(["dataset_id", "deployment_id"], sort=False)
    for (ds, dep), g in grouped_windows:
        g_idx = g.index.to_numpy()
        win_start = pd.to_datetime(g["window_start"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
        win_end = pd.to_datetime(g["window_end"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
        denom = np.maximum(1, win_end - win_start).astype(float)
        seg_loc = seg[(seg["dataset_id"] == ds) & (seg["deployment_id"] == dep)].copy()
        if seg_loc.empty:
            continue
        for key, key_seg in seg_loc.groupby("key", sort=False):
            slug = _safe_slug(key)
            frac = np.zeros(len(g_idx), dtype=float)
            for _, row in key_seg.iterrows():
                s_ns = pd.Timestamp(row["datetime"]).value
                e_ns = pd.Timestamp(row["end_datetime"]).value
                overlap = np.maximum(0, np.minimum(win_end, e_ns) - np.maximum(win_start, s_ns)).astype(float)
                frac = np.maximum(frac, overlap / denom)
            work.loc[g.index, f"algorithmic_overlap_frac__{slug}"] = frac
            work.loc[g.index, f"algorithmic_overlap__{slug}"] = frac > 0

    overlap_frac_cols = [c for c in work.columns if c.startswith("algorithmic_overlap_frac__")]
    if overlap_frac_cols:
        work["algorithmic_overlap_label"] = pd.Series(pd.NA, index=work.index, dtype=object)
        work["algorithmic_overlap_max_frac"] = work[overlap_frac_cols].max(axis=1)
        positive_mask = work["algorithmic_overlap_max_frac"].fillna(0).astype(float) > 0
        if positive_mask.any():
            best_cols = work.loc[positive_mask, overlap_frac_cols].astype(float).idxmax(axis=1)
            work.loc[positive_mask, "algorithmic_overlap_label"] = best_cols.str.replace(
                "algorithmic_overlap_frac__", "", n=1, regex=False
            ).to_numpy()

        group_cols = ["cluster_pass", "cluster_rank"] if "cluster_pass" in work.columns and "cluster_rank" in work.columns else (["cluster_rank"] if "cluster_rank" in work.columns else ["cluster_id"])
        for frac_col in overlap_frac_cols:
            slug = frac_col.replace("algorithmic_overlap_frac__", "", 1)
            any_col = f"algorithmic_overlap__{slug}"
            agg = work.groupby(group_cols, dropna=False).agg(
                mean_overlap_frac=(frac_col, "mean"),
                any_overlap_rate=(any_col, "mean"),
                n_windows=(frac_col, "size"),
            ).reset_index()
            agg["segment_key"] = slug
            summary_rows.append(agg)

    overlap_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    return work, overlap_summary


def _annotate_windows_with_algorithmic_labels(cdf: pd.DataFrame, segment_df: pd.DataFrame, default_code: int = 0, default_name: str = "find_rest.not_sleep") -> pd.DataFrame:
    """
    Annotate each clustering window using the compact algorithmic segment table.

    This is an approximation relative to the dense label timeseries: each window
    receives the label of the segment with the greatest overlap fraction.
    Windows with no segment overlap fall back to the default not-sleep label.
    """
    if segment_df is None or segment_df.empty:
        out = cdf.copy()
        out["algorithmic_label_code"] = default_code
        out["algorithmic_label_name"] = default_name
        return out

    work = cdf.copy()
    work["window_start"] = pd.to_datetime(work["window_start"], errors="coerce")
    work["window_end"] = pd.to_datetime(work["window_end"], errors="coerce")
    out_codes = pd.Series(default_code, index=work.index, dtype="Int64")
    out_names = pd.Series(default_name, index=work.index, dtype=object)
    best_frac = pd.Series(0.0, index=work.index, dtype=float)

    seg = segment_df.copy()
    if "start_datetime" in seg.columns:
        seg["start_datetime"] = pd.to_datetime(seg["start_datetime"], errors="coerce")
    elif "datetime" in seg.columns:
        seg["start_datetime"] = pd.to_datetime(seg["datetime"], errors="coerce")
    else:
        seg["start_datetime"] = pd.NaT
    if "end_datetime" in seg.columns:
        seg["end_datetime"] = pd.to_datetime(seg["end_datetime"], errors="coerce")
    else:
        seg["end_datetime"] = pd.NaT
    seg["label_code"] = pd.to_numeric(seg.get("label_code", np.nan), errors="coerce")
    seg["label_name"] = seg.get("label_name", default_name)
    seg = seg.dropna(subset=["start_datetime", "end_datetime"])
    if seg.empty:
        work["algorithmic_label_code"] = out_codes
        work["algorithmic_label_name"] = out_names
        return work

    for (ds, dep), g in work.groupby(["dataset_id", "deployment_id"], sort=False):
        seg_loc = seg[(seg["dataset_id"] == ds) & (seg["deployment_id"] == dep)].copy()
        if seg_loc.empty:
            continue
        g_idx = g.index.to_numpy()
        win_start = pd.to_datetime(g["window_start"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
        win_end = pd.to_datetime(g["window_end"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
        denom = np.maximum(1, win_end - win_start).astype(float)
        local_best = best_frac.loc[g.index].to_numpy(dtype=float)

        for _, row in seg_loc.iterrows():
            s_ns = pd.Timestamp(row["start_datetime"]).value
            e_ns = pd.Timestamp(row["end_datetime"]).value
            overlap = np.maximum(0, np.minimum(win_end, e_ns) - np.maximum(win_start, s_ns)).astype(float)
            frac = overlap / denom
            better = frac > local_best
            if not np.any(better):
                continue
            local_best = np.where(better, frac, local_best)
            better_idx = g_idx[better]
            label_code = row.get("label_code", default_code)
            if pd.notna(label_code):
                out_codes.loc[better_idx] = int(label_code)
            out_names.loc[better_idx] = str(row.get("label_name", default_name) or default_name)
        best_frac.loc[g.index] = local_best

    work["algorithmic_label_code"] = out_codes.astype(int)
    work["algorithmic_label_name"] = out_names
    work["algorithmic_label_overlap_frac"] = best_frac
    return work


def _annotate_windows_with_algorithmic_segment_fields(
    cdf: pd.DataFrame,
    segment_df: pd.DataFrame,
    field_names: List[str],
    prefix: str = "algorithmic_",
) -> pd.DataFrame:
    work = cdf.copy()
    for field in field_names:
        out_field = f"{prefix}{field}"
        if out_field not in work.columns:
            work[out_field] = pd.NA
    if segment_df is None or segment_df.empty or not field_names:
        return work

    work["window_start"] = pd.to_datetime(work["window_start"], errors="coerce")
    work["window_end"] = pd.to_datetime(work["window_end"], errors="coerce")
    seg = segment_df.copy()
    seg["start_datetime"] = pd.to_datetime(seg.get("start_datetime", seg.get("datetime")), errors="coerce")
    seg["end_datetime"] = pd.to_datetime(seg.get("end_datetime"), errors="coerce")
    seg = seg.dropna(subset=["start_datetime", "end_datetime"])
    if seg.empty:
        return work

    for (ds, dep), g in work.groupby(["dataset_id", "deployment_id"], sort=False):
        seg_loc = seg[(seg["dataset_id"] == ds) & (seg["deployment_id"] == dep)].copy()
        if seg_loc.empty:
            continue
        g_idx = g.index.to_numpy()
        win_start = pd.to_datetime(g["window_start"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
        win_end = pd.to_datetime(g["window_end"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
        denom = np.maximum(1, win_end - win_start).astype(float)
        best_frac = np.zeros(len(g_idx), dtype=float)
        best_row = np.full(len(g_idx), -1, dtype=int)

        seg_loc = seg_loc.reset_index(drop=True)
        for seg_pos, row in enumerate(seg_loc.itertuples(index=False)):
            s_ns = pd.Timestamp(getattr(row, "start_datetime")).value
            e_ns = pd.Timestamp(getattr(row, "end_datetime")).value
            overlap = np.maximum(0, np.minimum(win_end, e_ns) - np.maximum(win_start, s_ns)).astype(float)
            frac = overlap / denom
            better = frac > best_frac
            if not np.any(better):
                continue
            best_frac = np.where(better, frac, best_frac)
            best_row = np.where(better, seg_pos, best_row)

        matched = best_row >= 0
        if not np.any(matched):
            continue
        matched_idx = g_idx[matched]
        chosen = seg_loc.iloc[best_row[matched]].reset_index(drop=True)
        for field in field_names:
            out_field = f"{prefix}{field}"
            if field not in chosen.columns:
                continue
            work.loc[matched_idx, out_field] = chosen[field].to_numpy()
    return work


def _prepare_channel_arrays(signal_data: Dict, spec: Dict, deployment_tz: str = None) -> Dict[str, Dict]:
    prepared = {}
    for full_key, scfg in spec.items():
        sig = scfg["signal_id"]
        ch = scfg["channel_id"]
        if sig not in signal_data:
            continue
        sdf = signal_data[sig]
        if ch not in sdf.columns or "datetime" not in sdf.columns:
            continue

        dt = _normalize_datetime_series_to_timezone(sdf["datetime"], deployment_tz)
        vals = pd.to_numeric(sdf[ch], errors="coerce").to_numpy(dtype=float)
        dt_ns = dt.to_numpy(dtype="datetime64[ns]").astype("int64")
        valid = (~pd.isna(dt).to_numpy()) & np.isfinite(vals)
        if not np.any(valid):
            continue

        dt_ns = dt_ns[valid]
        vals = vals[valid]
        if dt_ns.size == 0:
            continue

        order = np.argsort(dt_ns, kind="mergesort")
        prepared[full_key] = {
            "dt_ns": dt_ns[order],
            "vals": vals[order],
            "cfg": scfg,
        }
    return prepared


def _compute_features_for_deployment(
    ctx: RunContext,
    spec: Dict,
    dataset_id: str,
    deployment_id: str,
    mode: str,
    memory_log_interval_s: float,
    max_chunks_per_deployment: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    pkl_path = _data_pkl_path(ctx, dataset_id, deployment_id)
    if not os.path.exists(pkl_path):
        return pd.DataFrame(), pd.DataFrame(), {"status": "missing_data_pkl"}

    try:
        data_pkl = _load_data_pkl(pkl_path)
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), {"status": f"load_error:{type(e).__name__}"}

    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    deployment_tz = _deployment_timezone_name(data_pkl)
    standardized_cfg = _parse_standardized_channels(ctx.run_cfg)
    compatibility_tolerance = ctx.run_cfg.get("sampling_compatibility_tolerance")
    synthetic_signal_data, std_diagnostics, std_plot_payloads = _standardize_signal_data(
        signal_data,
        standardized_cfg,
        deployment_tz,
    )
    merged_signal_data = dict(signal_data)
    merged_signal_data.update(synthetic_signal_data)
    trim_cfg = _parse_window_trim_cfg(ctx.run_cfg)
    trim_start, trim_end, trim_summary = _compute_window_trim_interval(
        merged_signal_data,
        deployment_tz,
        trim_cfg,
    )
    if trim_cfg.get("enabled", False) and trim_summary.get("trim_status") != "ok":
        return pd.DataFrame(), pd.DataFrame(), {
            "status": str(trim_summary.get("trim_status") or "trim_failed"),
            "windows_seen": 0,
            "windows_after_trim": 0,
            "feature_rows": 0,
            "trim_start_datetime": trim_summary.get("trim_start_datetime"),
            "trim_end_datetime": trim_summary.get("trim_end_datetime"),
            "trim_rule": trim_summary.get("trim_rule"),
            "trim_depth_threshold_m": trim_summary.get("trim_depth_threshold_m"),
            "trim_status": trim_summary.get("trim_status"),
            "trimmed_window_count": 0,
            "dropped_prepost_land_window_count": 0,
        }
    compatibility_rows = _validate_channel_sampling_compatibility(
        signal_data,
        spec,
        standardized_cfg,
        deployment_tz,
        tolerance=float(compatibility_tolerance) if compatibility_tolerance is not None else None,
        dataset_id=dataset_id,
        deployment_id=deployment_id,
    )
    _write_preprocessing_diagnostics(
        ctx,
        dataset_id,
        deployment_id,
        diagnostics_rows=std_diagnostics + compatibility_rows,
        plot_payloads=std_plot_payloads,
    )
    primary_channel = ctx.run_cfg.get("primary_input_channel_for_ranking") or next(iter(spec.keys()))
    base_sig = _select_base_timeline_signal(merged_signal_data, spec, primary_channel)
    if not base_sig:
        return pd.DataFrame(), pd.DataFrame(), {"status": "missing_base_signal_datetime"}
    base_df = merged_signal_data[base_sig].copy()
    if "datetime" not in base_df.columns:
        return pd.DataFrame(), pd.DataFrame(), {"status": "missing_base_signal_datetime"}
    base_df["datetime"] = _normalize_datetime_series_to_timezone(base_df["datetime"], deployment_tz)

    prepared = _prepare_channel_arrays(merged_signal_data, spec, deployment_tz=deployment_tz)
    if not prepared:
        return pd.DataFrame(), pd.DataFrame(), {"status": "no_channels_available"}

    rows = []
    index_rows = []
    dep_window_count = 0
    dep_feature_rows = 0
    trimmed_window_count = 0
    dropped_prepost_land_window_count = 0
    last_log = time.monotonic()

    for w_start, w_end, _window_df in _iter_windows(base_df, mode, ctx.run_cfg):
        dep_window_count += 1
        if not _window_is_within_trim(w_start, w_end, trim_start, trim_end):
            dropped_prepost_land_window_count += 1
            continue
        if max_chunks_per_deployment is not None and dep_feature_rows >= max_chunks_per_deployment:
            break
        trimmed_window_count += 1
        w_start_ts = pd.Timestamp(w_start)
        w_end_ts = pd.Timestamp(w_end)
        w_start_ns = int(w_start_ts.value)
        w_end_ns = int(w_end_ts.value)
        feat_row = {
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "window_start": w_start_ts,
            "window_end": w_end_ts,
            "window_seconds": float((w_end_ts - w_start_ts).total_seconds()),
        }
        valid_any = False

        for full_key, p in prepared.items():
            dt_ns = p["dt_ns"]
            vals = p["vals"]
            scfg = p["cfg"]
            left = int(np.searchsorted(dt_ns, w_start_ns, side="left"))
            right = int(np.searchsorted(dt_ns, w_end_ns, side="left"))
            if right <= left:
                continue
            arr = vals[left:right]

            if full_key == primary_channel:
                feat_row["__primary_channel_mean_raw"] = float(np.mean(arr))

            for t in scfg["transforms"]:
                t_arr = _apply_transform(arr, t)
                if t_arr.size == 0:
                    continue
                stats = _compute_stat_features(t_arr, scfg["feature_set"])
                for k, v in stats.items():
                    feat_row[f"{full_key}__{t}__{k}"] = v
                valid_any = True

        if valid_any:
            rows.append(feat_row)
            index_rows.append({
                "dataset_id": dataset_id,
                "deployment_id": deployment_id,
                "window_start": feat_row["window_start"],
                "window_end": feat_row["window_end"],
                "window_seconds": feat_row["window_seconds"],
            })
            dep_feature_rows += 1

        if _should_log(last_log, memory_log_interval_s):
            print(
                f"[features] {dataset_id}/{deployment_id} | "
                f"windows={dep_window_count} features_rows={dep_feature_rows} | "
                f"RSS={_get_rss_gb():.2f} GB"
            )
            last_log = time.monotonic()

    feat_df = pd.DataFrame(rows)
    idx_df = pd.DataFrame(index_rows)
    return feat_df, idx_df, {
        "status": "ok",
        "windows_seen": dep_window_count,
        "windows_after_trim": trimmed_window_count,
        "feature_rows": dep_feature_rows,
        "trim_start_datetime": trim_summary.get("trim_start_datetime"),
        "trim_end_datetime": trim_summary.get("trim_end_datetime"),
        "trim_rule": trim_summary.get("trim_rule"),
        "trim_depth_threshold_m": trim_summary.get("trim_depth_threshold_m"),
        "trim_status": trim_summary.get("trim_status"),
        "trimmed_window_count": trimmed_window_count,
        "dropped_prepost_land_window_count": dropped_prepost_land_window_count,
    }


def cmd_algorithmic_segments_deployment(args):
    t_cmd_start = time.monotonic()
    ctx = _resolve_run_context(args.config, args.run_name)
    algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
    _ensure_dir(os.path.dirname(args.output_segments))
    _ensure_dir(os.path.dirname(args.output_summary))
    exhaustive_output_path = _algorithmic_exhaustive_segment_paths(ctx, args.dataset_id, args.deployment_id)
    _ensure_dir(os.path.dirname(exhaustive_output_path))
    # No longer saving label_timeseries files

    if not algo_cfg.get("enabled"):
        pd.DataFrame().to_parquet(args.output_segments, index=False)
        pd.DataFrame().to_parquet(exhaustive_output_path, index=False)
        with open(args.output_summary, "w") as f:
            json.dump({"enabled": False, "status": "disabled"}, f, indent=2)
        print(f"[segments] skipped {args.dataset_id}/{args.deployment_id} (disabled)")
        return

    pkl_path = _data_pkl_path(ctx, args.dataset_id, args.deployment_id)
    data_pkl = None
    t_load_start = time.monotonic()
    try:
        data_pkl = _load_data_pkl(pkl_path)
    except Exception:
        data_pkl = None
    t_after_load = time.monotonic()

    try:
        requested_pass_mode = str(getattr(args, "context_filter_pass_mode", "") or "").strip().lower() or None
        seg_df, exhaustive_seg_df, event_df, label_ts, intermediate_df, summary = _compute_algorithmic_segments_for_deployment(
            ctx,
            args.dataset_id,
            args.deployment_id,
            context_filter_pass_mode=requested_pass_mode,
            data_pkl=data_pkl,
        )
    except Exception as e:
        pd.DataFrame().to_parquet(args.output_segments, index=False)
        pd.DataFrame().to_parquet(exhaustive_output_path, index=False)
        summary = {
            "enabled": True,
            "status": "source_error",
            "dataset_id": args.dataset_id,
            "deployment_id": args.deployment_id,
            "source_key": algo_cfg.get("source_key"),
            "method_name": algo_cfg.get("method_name"),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "n_initial_segments": 0,
            "n_filtered_segments": 0,
            "n_long_flat_segments": 0,
            "n_resting_benthic_segments": 0,
            "n_surface_sleep_segments": 0,
            "n_resting_surface_segments": 0,
            "n_long_drift_segments": 0,
            "n_active_uw_swimming_segments": 0,
            "n_calm_uw_gliding_segments": 0,
            "n_active_surface_swimming_segments": 0,
            "n_calm_surface_gliding_segments": 0,
            "n_unscorable_segments": 0,
            "n_unscorable_noncandidate_segments": 0,
            "n_unscorable_surface_segments": 0,
            "event_keys": algo_cfg.get("event_keys", {}),
            "label_codes": algo_cfg.get("label_codes", {}),
            "label_names": algo_cfg.get("label_names", {}),
        }
        with open(args.output_summary, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(
            f"[segments] skipped {args.dataset_id}/{args.deployment_id} "
            f"status=source_error error={type(e).__name__}: {e}"
        )
        return
    t_after_compute = time.monotonic()

    # Coverage diagnostics: compare identified-segment coverage and full label_ts coverage
    # against total recording duration.
    coverage_flags: List[str] = []
    try:
        ts = pd.to_datetime(label_ts.get("datetime"), errors="coerce")
        ts = ts.dropna().sort_values().reset_index(drop=True)
        recording_total_s = np.nan
        label_timeseries_total_s = np.nan
        identified_segments_total_s = np.nan
        identified_candidate_segments_total_s = np.nan
        remaining_unsegmented_s = np.nan
        if len(ts) >= 2:
            diffs = ts.diff().dt.total_seconds()
            diffs = pd.to_numeric(diffs, errors="coerce")
            median_dt_s = float(diffs[diffs > 0].median()) if (diffs > 0).any() else np.nan
            span_s = float((ts.iloc[-1] - ts.iloc[0]).total_seconds())
            if np.isfinite(median_dt_s) and median_dt_s > 0:
                recording_total_s = max(span_s + median_dt_s, 0.0)
                label_timeseries_total_s = float(len(ts) * median_dt_s)
            else:
                recording_total_s = max(span_s, 0.0)
                label_timeseries_total_s = recording_total_s

            coverage_df = exhaustive_seg_df if (exhaustive_seg_df is not None and not exhaustive_seg_df.empty) else seg_df
            if coverage_df is not None and not coverage_df.empty and {"start_datetime", "end_datetime"} <= set(coverage_df.columns):
                seg_times = coverage_df[["start_datetime", "end_datetime"]].copy()
                seg_times["start_datetime"] = pd.to_datetime(seg_times["start_datetime"], errors="coerce")
                seg_times["end_datetime"] = pd.to_datetime(seg_times["end_datetime"], errors="coerce")
                seg_times = seg_times.dropna(subset=["start_datetime", "end_datetime"]).sort_values("start_datetime")
                merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
                for row in seg_times.itertuples(index=False):
                    s = row.start_datetime
                    e = row.end_datetime
                    if e < s:
                        s, e = e, s
                    if not merged:
                        merged.append((s, e))
                        continue
                    ps, pe = merged[-1]
                    if s <= pe:
                        merged[-1] = (ps, max(pe, e))
                    else:
                        merged.append((s, e))
                identified_segments_total_s = float(
                    sum(
                        max((e - s).total_seconds(), 0.0) + (median_dt_s if np.isfinite(median_dt_s) and median_dt_s > 0 else 0.0)
                        for s, e in merged
                    )
                ) if merged else 0.0
                remaining_unsegmented_s = max(recording_total_s - identified_segments_total_s, 0.0)
            if seg_df is not None and not seg_df.empty and {"start_datetime", "end_datetime"} <= set(seg_df.columns):
                seg_times_legacy = seg_df[["start_datetime", "end_datetime"]].copy()
                seg_times_legacy["start_datetime"] = pd.to_datetime(seg_times_legacy["start_datetime"], errors="coerce")
                seg_times_legacy["end_datetime"] = pd.to_datetime(seg_times_legacy["end_datetime"], errors="coerce")
                seg_times_legacy = seg_times_legacy.dropna(subset=["start_datetime", "end_datetime"]).sort_values("start_datetime")
                merged_legacy: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
                for row in seg_times_legacy.itertuples(index=False):
                    s = row.start_datetime
                    e = row.end_datetime
                    if e < s:
                        s, e = e, s
                    if not merged_legacy:
                        merged_legacy.append((s, e))
                        continue
                    ps, pe = merged_legacy[-1]
                    if s <= pe:
                        merged_legacy[-1] = (ps, max(pe, e))
                    else:
                        merged_legacy.append((s, e))
                identified_candidate_segments_total_s = float(
                    sum(
                        max((e - s).total_seconds(), 0.0) + (median_dt_s if np.isfinite(median_dt_s) and median_dt_s > 0 else 0.0)
                        for s, e in merged_legacy
                    )
                ) if merged_legacy else 0.0

            if np.isfinite(recording_total_s) and recording_total_s > 0:
                seg_cov = (
                    float(identified_segments_total_s / recording_total_s)
                    if np.isfinite(identified_segments_total_s)
                    else np.nan
                )
                ts_cov = (
                    float(label_timeseries_total_s / recording_total_s)
                    if np.isfinite(label_timeseries_total_s)
                    else np.nan
                )
                summary["recording_total_s"] = recording_total_s
                summary["label_timeseries_total_s"] = label_timeseries_total_s
                summary["label_timeseries_coverage_ratio"] = ts_cov
                summary["identified_segments_total_s"] = identified_segments_total_s
                summary["identified_segments_coverage_ratio"] = seg_cov
                summary["remaining_unsegmented_s"] = remaining_unsegmented_s
                if np.isfinite(identified_candidate_segments_total_s):
                    summary["identified_candidate_segments_total_s"] = identified_candidate_segments_total_s
                    summary["identified_candidate_segments_coverage_ratio"] = float(
                        identified_candidate_segments_total_s / recording_total_s
                    )

                # Allow small edge effects from timestamp snapping/inclusive endpoints.
                # Only flag when both absolute and relative mismatch are meaningfully large.
                ts_tol_s = max(3.0 * (median_dt_s if np.isfinite(median_dt_s) else 0.0), 30.0)
                ts_mismatch_s = (
                    abs(label_timeseries_total_s - recording_total_s)
                    if np.isfinite(label_timeseries_total_s)
                    else np.nan
                )
                ts_mismatch_ratio = (
                    float(ts_mismatch_s / recording_total_s)
                    if np.isfinite(ts_mismatch_s) and np.isfinite(recording_total_s) and recording_total_s > 0
                    else np.nan
                )
                if (
                    np.isfinite(ts_mismatch_s)
                    and ts_mismatch_s > ts_tol_s
                    and (not np.isfinite(ts_mismatch_ratio) or ts_mismatch_ratio > 0.005)
                ):
                    coverage_flags.append(
                        f"label_timeseries_mismatch({label_timeseries_total_s:.1f}s vs recording {recording_total_s:.1f}s)"
                    )
                if np.isfinite(seg_cov) and seg_cov < 0.95:
                    coverage_flags.append(f"identified_segment_coverage_low({seg_cov:.1%})")
                if np.isfinite(remaining_unsegmented_s) and remaining_unsegmented_s > max(60.0, 0.02 * recording_total_s):
                    coverage_flags.append(f"remaining_unsegmented({remaining_unsegmented_s:.1f}s)")
    except Exception as e:
        coverage_flags.append(f"coverage_check_error({type(e).__name__})")

    if coverage_flags:
        summary["coverage_flags"] = coverage_flags
    t_after_coverage = time.monotonic()

    if seg_df.empty:
        pd.DataFrame().to_parquet(args.output_segments, index=False)
    else:
        seg_df.to_parquet(args.output_segments, index=False)
    if exhaustive_seg_df is None or exhaustive_seg_df.empty:
        pd.DataFrame().to_parquet(exhaustive_output_path, index=False)
    else:
        exhaustive_seg_df.to_parquet(exhaustive_output_path, index=False)
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    t_after_parquet = time.monotonic()

    if data_pkl is None:
        data_pkl = _load_data_pkl(pkl_path)
    if not hasattr(data_pkl, "signal_data") or data_pkl.signal_data is None:
        data_pkl.signal_data = {}
    if not hasattr(data_pkl, "signal_info") or data_pkl.signal_info is None:
        data_pkl.signal_info = {}
    label_signal_name = "algorithmic_labels"
    label_signal_df = label_ts[["datetime", "algorithmic_label_code", "algorithmic_label_name", "algorithmic_segment_key"]].rename(
        columns={
            "algorithmic_label_code": "label_code",
            "algorithmic_label_name": "label_name",
            "algorithmic_segment_key": "segment_key",
        }
    ).copy()
    data_pkl.signal_data[label_signal_name] = label_signal_df
    data_pkl.signal_info[label_signal_name] = {
        "channels": ["label_code", "label_name", "segment_key"],
        "metadata": {
            "label_code": {"original_name": "Algorithmic Rest Label Code", "unit": "category_code", "parent_signal": algo_cfg["signal_id"]},
            "label_name": {"original_name": "Algorithmic Rest Label Name", "unit": "category", "parent_signal": algo_cfg["signal_id"]},
            "segment_key": {"original_name": "Algorithmic Rest Segment Key", "unit": "category", "parent_signal": algo_cfg["signal_id"]},
        },
        "derived_from_signals": [algo_cfg["signal_id"]],
        "details": f"Nominal algorithmic labels from {algo_cfg['method_name']} with codes {summary.get('label_codes', {})}",
    }

    intermediate_signal_name = "algorithmic_intermediate_channels"
    derivative_signal_name = "algorithmic_derivative_channels"
    feature_signal_name = "algorithmic_feature_channels"
    derivative_d1_signal_name = "algorithmic_depth_d1"
    derivative_d2_signal_name = "algorithmic_depth_d2"
    if algo_cfg.get("debug", {}).get("persist_intermediate_channels", True):
        intermediate_signal_df = intermediate_df.copy()
        if not intermediate_signal_df.empty:
            parent_signal_info = dict((data_pkl.signal_info or {}).get(algo_cfg["signal_id"], {}) or {})
            inferred_interval_s = _infer_sampling_interval_seconds(intermediate_signal_df["datetime"])
            inferred_sampling_frequency = (
                float(1.0 / inferred_interval_s)
                if np.isfinite(inferred_interval_s) and inferred_interval_s > 0
                else parent_signal_info.get("sampling_frequency")
            )
            data_pkl.signal_data[intermediate_signal_name] = intermediate_signal_df
            intermediate_signal_info = {
                "channels": [c for c in intermediate_signal_df.columns if c != "datetime"],
                "metadata": {
                    "depth_std_m": {
                        "original_name": "Algorithmic Depth",
                        "unit": "m",
                        "standardized_unit": "m",
                        "parent_signal": algo_cfg["signal_id"],
                        "signal": algo_cfg["signal_id"],
                    },
                    "depth_d1_ms": {
                        "original_name": "Algorithmic Depth First Derivative",
                        "unit": "m/s",
                        "standardized_unit": "m/s",
                        "parent_signal": algo_cfg["signal_id"],
                        "signal": algo_cfg["signal_id"],
                    },
                    "depth_d2_ms2": {
                        "original_name": "Algorithmic Depth Second Derivative",
                        "unit": "m/s^2",
                        "standardized_unit": "m/s^2",
                        "parent_signal": algo_cfg["signal_id"],
                        "signal": algo_cfg["signal_id"],
                    },
                    "depth_d1_next_ms": {
                        "original_name": "Algorithmic Next-Sample Depth First Derivative",
                        "unit": "m/s",
                        "standardized_unit": "m/s",
                        "parent_signal": algo_cfg["signal_id"],
                        "signal": algo_cfg["signal_id"],
                    },
                    "is_unfiltered_rest_candidate": {
                        "original_name": "Algorithmic Unfiltered Rest Candidate Mask",
                        "unit": "bool",
                        "standardized_unit": "bool",
                        "parent_signal": algo_cfg["signal_id"],
                        "signal": algo_cfg["signal_id"],
                    },
                    "is_signed_rest_candidate": {
                        "original_name": "Algorithmic Signed Rest Candidate Mask",
                        "unit": "bool",
                        "standardized_unit": "bool",
                        "parent_signal": algo_cfg["signal_id"],
                        "signal": algo_cfg["signal_id"],
                    },
                    "is_drift_candidate": {
                        "original_name": "Algorithmic Drift Candidate Mask",
                        "unit": "bool",
                        "standardized_unit": "bool",
                        "parent_signal": algo_cfg["signal_id"],
                        "signal": algo_cfg["signal_id"],
                    },
                    "is_surface_sleep_candidate": {
                        "original_name": "Algorithmic Surface Sleep Candidate Mask",
                        "unit": "bool",
                        "standardized_unit": "bool",
                        "parent_signal": algo_cfg["signal_id"],
                        "signal": algo_cfg["signal_id"],
                    },
                },
                "derived_from_signals": [algo_cfg["signal_id"]],
                "transformation_log": [
                    f"Derived intermediate channels from {algo_cfg['source_key']} for {algo_cfg['method_name']}.",
                    "Includes standardized depth, first/second derivatives, and candidate masks used by algorithmic segmentation.",
                ],
                "details": f"Intermediate algorithmic channels from {algo_cfg['method_name']}",
            }
            for passthrough_key in ["logger_manufacturer", "original_sampling_frequency"]:
                if passthrough_key in parent_signal_info:
                    intermediate_signal_info[passthrough_key] = parent_signal_info[passthrough_key]
            if inferred_sampling_frequency is not None:
                intermediate_signal_info["sampling_frequency"] = inferred_sampling_frequency
            data_pkl.signal_info[intermediate_signal_name] = intermediate_signal_info

            derivative_cols = [
                c for c in ["datetime", "depth_d1_ms", "depth_d2_ms2", "depth_d1_next_ms", "is_unfiltered_rest_candidate"] if c in intermediate_signal_df.columns
            ]
            feature_cols = [
                c for c in ["datetime", "depth_std_m", "is_unfiltered_rest_candidate", "is_signed_rest_candidate", "is_drift_candidate", "is_surface_sleep_candidate"] if c in intermediate_signal_df.columns
            ]
            derivative_df = intermediate_signal_df[derivative_cols].copy() if len(derivative_cols) > 1 else pd.DataFrame()
            feature_df = intermediate_signal_df[feature_cols].copy() if len(feature_cols) > 1 else pd.DataFrame()

            derivative_info = {
                "channels": [c for c in derivative_cols if c != "datetime"],
                "metadata": {k: v for k, v in intermediate_signal_info["metadata"].items() if k in derivative_cols},
                "derived_from_signals": [algo_cfg["signal_id"]],
                "transformation_log": [
                    f"Derived derivative-focused algorithmic channels from {algo_cfg['source_key']} for {algo_cfg['method_name']}.",
                ],
                "details": f"Derivative-focused algorithmic channels from {algo_cfg['method_name']}",
            }
            feature_info = {
                "channels": [c for c in feature_cols if c != "datetime"],
                "metadata": {k: v for k, v in intermediate_signal_info["metadata"].items() if k in feature_cols},
                "derived_from_signals": [algo_cfg["signal_id"]],
                "transformation_log": [
                    f"Derived feature-focused algorithmic channels from {algo_cfg['source_key']} for {algo_cfg['method_name']}.",
                ],
                "details": f"Feature-focused algorithmic channels from {algo_cfg['method_name']}",
            }
            for passthrough_key in ["logger_manufacturer", "original_sampling_frequency", "sampling_frequency"]:
                if passthrough_key in intermediate_signal_info:
                    derivative_info[passthrough_key] = intermediate_signal_info[passthrough_key]
                    feature_info[passthrough_key] = intermediate_signal_info[passthrough_key]
            if not derivative_df.empty:
                data_pkl.signal_data[derivative_signal_name] = derivative_df
                data_pkl.signal_info[derivative_signal_name] = derivative_info
            else:
                data_pkl.signal_data.pop(derivative_signal_name, None)
                data_pkl.signal_info.pop(derivative_signal_name, None)
            d1_cols = [c for c in ["datetime", "depth_d1_ms"] if c in intermediate_signal_df.columns]
            d2_cols = [c for c in ["datetime", "depth_d2_ms2"] if c in intermediate_signal_df.columns]
            d1_df = intermediate_signal_df[d1_cols].copy() if len(d1_cols) > 1 else pd.DataFrame()
            d2_df = intermediate_signal_df[d2_cols].copy() if len(d2_cols) > 1 else pd.DataFrame()
            d1_info = {
                "label": "d'(depth)",
                "channels": [c for c in d1_cols if c != "datetime"],
                "metadata": {
                    "depth_d1_ms": intermediate_signal_info["metadata"].get("depth_d1_ms", {})
                } if "depth_d1_ms" in d1_cols else {},
                "derived_from_signals": [algo_cfg["signal_id"]],
                "transformation_log": [
                    f"Derived standalone d1 algorithmic channel from {algo_cfg['source_key']} for {algo_cfg['method_name']}.",
                ],
                "details": f"Standalone algorithmic depth first derivative channel from {algo_cfg['method_name']}",
            }
            d2_info = {
                "label": "d''(depth)",
                "channels": [c for c in d2_cols if c != "datetime"],
                "metadata": {
                    "depth_d2_ms2": intermediate_signal_info["metadata"].get("depth_d2_ms2", {})
                } if "depth_d2_ms2" in d2_cols else {},
                "derived_from_signals": [algo_cfg["signal_id"]],
                "transformation_log": [
                    f"Derived standalone d2 algorithmic channel from {algo_cfg['source_key']} for {algo_cfg['method_name']}.",
                ],
                "details": f"Standalone algorithmic depth second derivative channel from {algo_cfg['method_name']}",
            }
            for passthrough_key in ["logger_manufacturer", "original_sampling_frequency", "sampling_frequency"]:
                if passthrough_key in intermediate_signal_info:
                    d1_info[passthrough_key] = intermediate_signal_info[passthrough_key]
                    d2_info[passthrough_key] = intermediate_signal_info[passthrough_key]
            if not d1_df.empty:
                data_pkl.signal_data[derivative_d1_signal_name] = d1_df
                data_pkl.signal_info[derivative_d1_signal_name] = d1_info
            else:
                data_pkl.signal_data.pop(derivative_d1_signal_name, None)
                data_pkl.signal_info.pop(derivative_d1_signal_name, None)
            if not d2_df.empty:
                data_pkl.signal_data[derivative_d2_signal_name] = d2_df
                data_pkl.signal_info[derivative_d2_signal_name] = d2_info
            else:
                data_pkl.signal_data.pop(derivative_d2_signal_name, None)
                data_pkl.signal_info.pop(derivative_d2_signal_name, None)
            if not feature_df.empty:
                data_pkl.signal_data[feature_signal_name] = feature_df
                data_pkl.signal_info[feature_signal_name] = feature_info
            else:
                data_pkl.signal_data.pop(feature_signal_name, None)
                data_pkl.signal_info.pop(feature_signal_name, None)
        else:
            data_pkl.signal_data.pop(intermediate_signal_name, None)
            data_pkl.signal_info.pop(intermediate_signal_name, None)
            data_pkl.signal_data.pop(derivative_signal_name, None)
            data_pkl.signal_info.pop(derivative_signal_name, None)
            data_pkl.signal_data.pop(feature_signal_name, None)
            data_pkl.signal_info.pop(feature_signal_name, None)
            data_pkl.signal_data.pop(derivative_d1_signal_name, None)
            data_pkl.signal_info.pop(derivative_d1_signal_name, None)
            data_pkl.signal_data.pop(derivative_d2_signal_name, None)
            data_pkl.signal_info.pop(derivative_d2_signal_name, None)
    else:
        data_pkl.signal_data.pop(intermediate_signal_name, None)
        data_pkl.signal_info.pop(intermediate_signal_name, None)
        data_pkl.signal_data.pop(derivative_signal_name, None)
        data_pkl.signal_info.pop(derivative_signal_name, None)
        data_pkl.signal_data.pop(feature_signal_name, None)
        data_pkl.signal_info.pop(feature_signal_name, None)
        data_pkl.signal_data.pop(derivative_d1_signal_name, None)
        data_pkl.signal_info.pop(derivative_d1_signal_name, None)
        data_pkl.signal_data.pop(derivative_d2_signal_name, None)
        data_pkl.signal_info.pop(derivative_d2_signal_name, None)

    if not hasattr(data_pkl, "event_data") or data_pkl.event_data is None:
        data_pkl.event_data = pd.DataFrame(columns=["date", "time", "value", "type", "key", "duration", "short_description", "long_description", "datetime"])
    existing = data_pkl.event_data.copy()
    existing_keys = set(str(v) for v in algo_cfg["event_keys"].values())
    if "key" in existing.columns:
        existing = existing[~existing["key"].astype(str).isin(existing_keys)].copy()
    if not event_df.empty:
        variant_id = str(algo_cfg.get("method_name") or algo_cfg.get("method") or "algorithmic")
        event_meta_rows = []
        for row in event_df.itertuples(index=False):
            row_ctx = {
                "dataset_id": getattr(row, "dataset_id", ""),
                "deployment_id": getattr(row, "deployment_id", ""),
                "segment_stage": getattr(row, "segment_stage", ""),
                "source_key": getattr(row, "source_key", ""),
                "label_code": getattr(row, "label_code", np.nan),
                "label_name": getattr(row, "label_name", ""),
            }
            event_meta_rows.append(
                _resolve_segmentation_event_metadata(
                    ctx,
                    step_name="algorithmic_segments",
                    event_key=str(getattr(row, "key", "")),
                    label_name=str(getattr(row, "label_name", "")),
                    variant_id=variant_id,
                    row_context=row_ctx,
                )
            )
        event_meta_df = pd.DataFrame(event_meta_rows)
        event_type_series = event_meta_df.get("type", pd.Series("state", index=event_df.index)).fillna("state").astype(str)
        short_desc_series = event_meta_df.get("short_description", pd.Series(index=event_df.index, dtype=object))
        short_desc_series = short_desc_series.fillna(event_df.get("label_name", event_df["key"])).astype(str)
        long_desc_series = event_meta_df.get("long_description", pd.Series(index=event_df.index, dtype=object))
        long_desc_series = long_desc_series.fillna(
            event_df.get("segment_stage", pd.Series(index=event_df.index, dtype=object)).astype(str)
        ).astype(str)
        new_events = pd.DataFrame({
            "date": pd.to_datetime(event_df["datetime"], errors="coerce").dt.floor("D").dt.strftime("%Y-%m-%d"),
            "time": pd.to_datetime(event_df["datetime"], errors="coerce").dt.strftime("%H:%M:%S.%f").str[:-3],
            "value": pd.to_numeric(event_df.get("label_code", np.nan), errors="coerce"),
            "type": event_type_series,
            "key": event_df["key"].astype(str),
            "duration": pd.to_numeric(event_df["duration"], errors="coerce"),
            "short_description": short_desc_series,
            "long_description": long_desc_series,
            "datetime": _normalize_datetime_series_to_timezone(event_df["datetime"], _deployment_timezone_name(data_pkl)),
        })
        data_pkl.event_data = pd.concat([existing, new_events], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    else:
        data_pkl.event_data = existing.sort_values("datetime").reset_index(drop=True) if "datetime" in existing.columns else existing.reset_index(drop=True)

    if not hasattr(data_pkl, "event_manager") or data_pkl.event_manager is None:
        data_pkl.event_manager = {}
    data_pkl.event_manager["algorithmic_segments"] = {
        "method_name": algo_cfg["method_name"],
        "keys": sorted(existing_keys),
        "label_codes": summary.get("label_codes", {}),
        "label_names": summary.get("label_names", {}),
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(data_pkl, f)
    t_after_pkl = time.monotonic()
    print(
        f"[segments] {args.dataset_id}/{args.deployment_id} "
        f"initial={summary.get('n_initial_segments', 0)} "
        f"base_filtered={summary.get('n_base_filtered_segments', 0)} "
        f"filtered={summary.get('n_filtered_segments', 0)} "
        f"swimming_uw={summary.get('n_exhaustive_active_uw_swimming_segments', summary.get('n_active_uw_swimming_segments', 0))} "
        f"gliding_uw={summary.get('n_exhaustive_calm_uw_gliding_segments', summary.get('n_calm_uw_gliding_segments', 0))} "
        f"resting_surface={summary.get('n_exhaustive_resting_surface_segments', summary.get('n_resting_surface_segments', summary.get('n_surface_sleep_segments', 0)))} "
        f"resting_benthic={summary.get('n_exhaustive_resting_benthic_segments', summary.get('n_resting_benthic_segments', summary.get('n_long_flat_segments', 0)))} "
        f"resting_drift={summary.get('n_exhaustive_long_drift_segments', summary.get('n_long_drift_segments', 0))} "
        f"swimming_surface={summary.get('n_exhaustive_active_surface_swimming_segments', summary.get('n_active_surface_swimming_segments', 0))} "
        f"gliding_surface={summary.get('n_exhaustive_calm_surface_gliding_segments', summary.get('n_calm_surface_gliding_segments', 0))} "
        f"unscorable={summary.get('n_exhaustive_unscorable_segments', summary.get('n_unscorable_segments', 0))} "
        f"unscorable_noncandidate={summary.get('n_unscorable_noncandidate_segments', 0)} "
        f"unscorable_surface={summary.get('n_unscorable_surface_segments', 0)} "
        f"context_rejected={summary.get('n_context_rejected_segments', 0)} "
        f"mode={summary.get('context_filter_pass_mode', 'n/a')}"
    )
    if np.isfinite(float(summary.get("recording_total_s", np.nan))):
        print(
            f"[segments] {args.dataset_id}/{args.deployment_id} "
            f"duration_check recording_s={summary.get('recording_total_s', np.nan):.1f} "
            f"identified_s={summary.get('identified_segments_total_s', np.nan):.1f} "
            f"identified_cov={100.0 * float(summary.get('identified_segments_coverage_ratio', np.nan)):.1f}% "
            f"label_ts_cov={100.0 * float(summary.get('label_timeseries_coverage_ratio', np.nan)):.1f}% "
            f"remaining_s={summary.get('remaining_unsegmented_s', np.nan):.1f}"
        )
    if np.isfinite(float(summary.get("identified_candidate_segments_total_s", np.nan))):
        print(
            f"[segments] {args.dataset_id}/{args.deployment_id} "
            f"candidate_duration_check candidate_identified_s={summary.get('identified_candidate_segments_total_s', np.nan):.1f} "
            f"candidate_identified_cov={100.0 * float(summary.get('identified_candidate_segments_coverage_ratio', np.nan)):.1f}%"
        )
    if np.isfinite(float(summary.get("exhaustive_total_duration_s", np.nan))) and np.isfinite(float(summary.get("recording_total_s", np.nan))):
        duration_delta = float(summary.get("recording_total_s", np.nan)) - float(summary.get("exhaustive_total_duration_s", np.nan))
        print(
            f"[segments] {args.dataset_id}/{args.deployment_id} "
            f"exhaustive_duration_check exhaustive_s={summary.get('exhaustive_total_duration_s', np.nan):.1f} "
            f"recording_s={summary.get('recording_total_s', np.nan):.1f} "
            f"delta_s={duration_delta:.1f}"
        )
    if coverage_flags:
        print(f"[segments] ⚠️ {args.dataset_id}/{args.deployment_id} coverage_flags={','.join(coverage_flags)}")
    if not seg_df.empty and "context_reject_reason" in seg_df.columns:
        if {"base_keep_filtered", "context_keep"}.issubset(seg_df.columns):
            base_mask = seg_df["base_keep_filtered"].fillna(False).infer_objects(copy=False).astype(bool)
            keep_mask = seg_df["context_keep"].fillna(True).infer_objects(copy=False).astype(bool)
            rejected_mask = base_mask & (~keep_mask)
        else:
            rejected_mask = seg_df["context_reject_reason"].notna()
        reasons = seg_df.loc[rejected_mask, "context_reject_reason"].dropna().astype(str)
        if not reasons.empty:
            reason_counts: Dict[str, int] = {}
            for raw in reasons:
                for reason in [part.strip() for part in str(raw).split("|") if str(part).strip()]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if reason_counts:
                top_reasons = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
                reason_text = ", ".join([f"{name}:{count}" for name, count in top_reasons])
                print(f"[segments] {args.dataset_id}/{args.deployment_id} reject_reasons={reason_text}")
    print(
        f"[segments][timing] {args.dataset_id}/{args.deployment_id} "
        f"load_pkl_s={max(0.0, t_after_load - t_load_start):.2f} "
        f"compute_s={max(0.0, t_after_compute - t_after_load):.2f} "
        f"coverage_s={max(0.0, t_after_coverage - t_after_compute):.2f} "
        f"parquet_s={max(0.0, t_after_parquet - t_after_coverage):.2f} "
        f"pkl_update_s={max(0.0, t_after_pkl - t_after_parquet):.2f} "
        f"total_s={max(0.0, t_after_pkl - t_cmd_start):.2f}"
    )


def cmd_algorithmic_segments_merge(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
    detailed_path, event_path = _algorithmic_segment_merge_paths(ctx)
    exhaustive_merge_path = _algorithmic_exhaustive_segment_merge_path(ctx)
    _ensure_dir(os.path.dirname(detailed_path))

    if not algo_cfg.get("enabled"):
        pd.DataFrame().to_parquet(detailed_path, index=False)
        pd.DataFrame().to_parquet(exhaustive_merge_path, index=False)
        pd.DataFrame().to_parquet(event_path, index=False)
        with open(args.output, "w") as f:
            json.dump({"enabled": False, "status": "disabled"}, f, indent=2)
        print("[segments] merge skipped (disabled)")
        return

    detailed_frames = []
    exhaustive_frames = []
    event_frames = []
    missing_segment_paths = []
    missing_summary_paths = []
    for item in ctx.scope:
        ds = item["dataset_id"]
        dep = item["deployment_id"]
        seg_path, summary_path = _algorithmic_segment_paths(ctx, ds, dep)
        exhaustive_path = _algorithmic_exhaustive_segment_paths(ctx, ds, dep)
        if os.path.exists(seg_path):
            try:
                df = pd.read_parquet(seg_path)
                if not df.empty:
                    detailed_frames.append(df)
            except Exception:
                pass
        else:
            missing_segment_paths.append(f"{ds}/{dep}: {seg_path}")
        if os.path.exists(exhaustive_path):
            try:
                ex_df = pd.read_parquet(exhaustive_path)
                if not ex_df.empty:
                    exhaustive_frames.append(ex_df)
            except Exception:
                pass
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                event_keys = ((summary.get("event_keys") or {}) if isinstance(summary, dict) else {}) or {}
                if os.path.exists(seg_path):
                    df = pd.read_parquet(seg_path)
                    if not df.empty:
                        for _, row in df.iterrows():
                            nominal = str(row.get("nominal_class", "not_sleep"))
                            segment_family = str(row.get("segment_family", ""))
                            base_keep_filtered = bool(row.get("base_keep_filtered", row.get("keep_filtered", False)))
                            base = {
                                "dataset_id": ds,
                                "deployment_id": dep,
                                "datetime": row["start_datetime"],
                                "end_datetime": row["end_datetime"],
                                "duration": row.get("duration_s", row.get("duration", np.nan)),
                                "segment_rank": int(row.get("segment_rank", 0)),
                                "source_key": row.get("source_key", summary.get("source_key")),
                                "label_code": row.get("label_code", np.nan),
                                "label_name": row.get("label_name", np.nan),
                            }
                            if segment_family == "drift_candidate":
                                event_frames.append(dict(base, key=event_keys.get("initial", "algorithmic_drift_initial"), segment_stage="initial"))
                            if nominal in {"surface_sleep", "resting_surface"}:
                                event_frames.append(
                                    dict(
                                        base,
                                        key=event_keys.get("resting_surface", event_keys.get("surface_sleep", "find_rest.surface_sleep")),
                                        segment_stage="surface_sleep",
                                    )
                                )
                            elif base_keep_filtered:
                                event_frames.append(dict(base, key=event_keys.get("filtered", "algorithmic_drift_filtered"), segment_stage="filtered"))
                            if nominal in {"long_flat", "resting_benthic"}:
                                event_frames.append(
                                    dict(
                                        base,
                                        key=event_keys.get("resting_benthic", event_keys.get("long_flat", "find_rest.long_flat")),
                                        segment_stage="flat",
                                    )
                                )
                            elif nominal == "long_drift":
                                event_frames.append(dict(base, key=event_keys.get("long_drift", "find_rest.long_drift"), segment_stage="filtered_nominal"))
            except Exception:
                pass
        else:
            missing_summary_paths.append(f"{ds}/{dep}: {summary_path}")

    if missing_segment_paths or missing_summary_paths:
        if missing_segment_paths:
            print("[segments-merge] missing deployment algorithmic segment parquet files:")
            for line in missing_segment_paths:
                print(f"  - {line}")
        if missing_summary_paths:
            print("[segments-merge] missing deployment algorithmic segment summary files:")
            for line in missing_summary_paths:
                print(f"  - {line}")
        raise ValueError("Incomplete algorithmic-segments outputs for one or more deployments.")

    detailed_df = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()
    exhaustive_df = pd.concat(exhaustive_frames, ignore_index=True) if exhaustive_frames else pd.DataFrame()
    merged_event_df = pd.DataFrame(event_frames).sort_values(["dataset_id", "deployment_id", "datetime", "key"]).reset_index(drop=True) if event_frames else pd.DataFrame()
    detailed_df.to_parquet(detailed_path, index=False)
    exhaustive_df.to_parquet(exhaustive_merge_path, index=False)
    merged_event_df.to_parquet(event_path, index=False)
    with open(args.output, "w") as f:
        json.dump(
            {
                "enabled": True,
                "status": "ok",
                "n_segments": int(len(detailed_df)),
                "n_exhaustive_segments": int(len(exhaustive_df)),
                "n_event_rows": int(len(merged_event_df)),
                "detailed_path": detailed_path,
                "exhaustive_path": exhaustive_merge_path,
                "event_path": event_path,
            },
            f,
            indent=2,
            default=str,
        )
    print(
        f"[segments] merged detailed={len(detailed_df)} "
        f"exhaustive={len(exhaustive_df)} event_rows={len(merged_event_df)}"
    )


def cmd_features(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    _run_qc_if_missing(ctx)
    spec, _standardized_cfg = _normalize_spec_for_processing(ctx.run_cfg)
    mode = str(ctx.run_cfg.get("cluster_length_mode", "fixed")).lower()
    if mode not in {"fixed", "variable"}:
        raise ValueError("cluster_length_mode must be fixed or variable")
    memory_log_interval_s = float(ctx.run_cfg.get("memory_log_interval_s", 30))
    max_chunks_per_deployment = ctx.run_cfg.get("max_chunks_per_deployment")
    if max_chunks_per_deployment is not None:
        max_chunks_per_deployment = int(max_chunks_per_deployment)
    max_chunks_total = ctx.run_cfg.get("max_chunks_total")
    if max_chunks_total is not None:
        max_chunks_total = int(max_chunks_total)

    feat_frames = []
    idx_frames = []
    total_rows = 0

    n_scope = len(ctx.scope)
    stop_all = False
    for i, item in enumerate(ctx.scope, start=1):
        if stop_all:
            break
        ds = item["dataset_id"]
        dep = item["deployment_id"]
        print(f"[features] [{i}/{n_scope}] start {ds}/{dep} | RSS={_get_rss_gb():.2f} GB")
        feat_df_dep, idx_df_dep, stats = _compute_features_for_deployment(
            ctx=ctx,
            spec=spec,
            dataset_id=ds,
            deployment_id=dep,
            mode=mode,
            memory_log_interval_s=memory_log_interval_s,
            max_chunks_per_deployment=max_chunks_per_deployment,
        )
        status = str((stats or {}).get("status", "unknown"))
        if status != "ok":
            print(f"[features] [{i}/{n_scope}] skip {ds}/{dep}: {status}")
            continue
        dep_rows = int(len(feat_df_dep))
        if dep_rows == 0:
            print(f"[features] [{i}/{n_scope}] skip {ds}/{dep}: no feature rows generated")
            continue
        if max_chunks_total is not None and (total_rows + dep_rows) > max_chunks_total:
            keep_rows = max(0, int(max_chunks_total - total_rows))
            if keep_rows <= 0:
                stop_all = True
                print(f"[features] reached max_chunks_total={max_chunks_total}, stopping early.")
                break
            feat_df_dep = feat_df_dep.sort_values(["window_start"]).head(keep_rows).reset_index(drop=True)
            idx_df_dep = idx_df_dep.sort_values(["window_start"]).head(keep_rows).reset_index(drop=True)
            dep_rows = keep_rows
            stop_all = True
        feat_frames.append(feat_df_dep)
        idx_frames.append(idx_df_dep)
        total_rows += dep_rows
        print(
            f"[features] [{i}/{n_scope}] done {ds}/{dep} | "
            f"windows={int((stats or {}).get('windows_seen', 0))} "
            f"trimmed_windows={int((stats or {}).get('trimmed_window_count', 0))} "
            f"dropped_land={int((stats or {}).get('dropped_prepost_land_window_count', 0))} "
            f"features_rows={dep_rows} | RSS={_get_rss_gb():.2f} GB"
        )
        if max_chunks_per_deployment is not None and dep_rows >= max_chunks_per_deployment:
            print(
                f"[features] [{i}/{n_scope}] reached max_chunks_per_deployment="
                f"{max_chunks_per_deployment} for {ds}/{dep}"
            )
        if stop_all and max_chunks_total is not None:
            print(f"[features] reached max_chunks_total={max_chunks_total}, stopping early.")
            break

    if not feat_frames:
        raise ValueError("No feature rows generated. Check inputs and channel_feature_spec.")

    feat_df = pd.concat(feat_frames, ignore_index=True, sort=False)
    idx_df = pd.concat(idx_frames, ignore_index=True, sort=False)

    # normalize only numeric feature columns (excluding metadata)
    meta_cols = {"dataset_id", "deployment_id", "window_start", "window_end", "window_seconds"}
    numeric_cols = [
        c for c in feat_df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(feat_df[c])
    ]

    norm_level = str(ctx.run_cfg.get("normalization_level", "deployment")).lower()
    def _zscore(s):
        denom = float(s.std(ddof=0))
        if not np.isfinite(denom) or denom == 0:
            denom = 1.0
        return (s - s.mean()) / denom

    if norm_level == "deployment":
        feat_df[numeric_cols] = feat_df.groupby("deployment_id")[numeric_cols].transform(_zscore)
    elif norm_level == "dataset":
        feat_df[numeric_cols] = feat_df.groupby("dataset_id")[numeric_cols].transform(_zscore)

    feature_count = len([c for c in numeric_cols if not c.startswith("__")])
    print(f"[features] total numeric features: {feature_count}")

    raw_path, idx_path = _feature_paths(ctx)
    _ensure_dir(os.path.dirname(raw_path))
    feat_df.to_parquet(raw_path, index=False)
    idx_df.to_parquet(idx_path, index=False)

    _ensure_dir(os.path.dirname(args.log_output))
    with open(args.log_output, "w") as f:
        f.write(json.dumps({
            "feature_count": feature_count,
            "rows": len(feat_df),
            "normalization_level": norm_level,
            "catch22_available": CATCH22_AVAILABLE,
            "max_chunks_per_deployment": max_chunks_per_deployment,
            "max_chunks_total": max_chunks_total,
        }, indent=2))

    print(f"[features] wrote {raw_path}")
    print(f"[features] wrote {idx_path}")


def cmd_features_deployment(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    _run_qc_if_missing(ctx)
    spec, _standardized_cfg = _normalize_spec_for_processing(ctx.run_cfg)
    mode = str(ctx.run_cfg.get("cluster_length_mode", "fixed")).lower()
    if mode not in {"fixed", "variable"}:
        raise ValueError("cluster_length_mode must be fixed or variable")
    memory_log_interval_s = float(ctx.run_cfg.get("memory_log_interval_s", 30))
    max_chunks_per_deployment = ctx.run_cfg.get("max_chunks_per_deployment")
    if max_chunks_per_deployment is not None:
        max_chunks_per_deployment = int(max_chunks_per_deployment)

    ds = args.dataset_id
    dep = args.deployment_id
    print(f"[features-deployment] start {ds}/{dep} | RSS={_get_rss_gb():.2f} GB")
    feat_df, idx_df, stats = _compute_features_for_deployment(
        ctx=ctx,
        spec=spec,
        dataset_id=ds,
        deployment_id=dep,
        mode=mode,
        memory_log_interval_s=memory_log_interval_s,
        max_chunks_per_deployment=max_chunks_per_deployment,
    )

    _ensure_dir(os.path.dirname(args.output_feature))
    _ensure_dir(os.path.dirname(args.output_index))
    feat_df.to_parquet(args.output_feature, index=False)
    idx_df.to_parquet(args.output_index, index=False)

    if args.log_output:
        _ensure_dir(os.path.dirname(args.log_output))
        with open(args.log_output, "w") as f:
            json.dump(
                {
                    "dataset_id": ds,
                    "deployment_id": dep,
                    "stats": stats,
                    "rows_feature": int(len(feat_df)),
                    "rows_index": int(len(idx_df)),
                },
                f,
                indent=2,
            )
    print(
        f"[features-deployment] done {ds}/{dep} | "
        f"rows={len(feat_df)} | out={args.output_feature}"
    )


def cmd_features_merge(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    _run_qc_if_missing(ctx)

    frames = []
    idx_frames = []
    missing_feature_paths = []
    for item in ctx.scope:
        ds = item["dataset_id"]
        dep = item["deployment_id"]
        f_path, i_path = _feature_deployment_paths(ctx, ds, dep)
        if os.path.exists(f_path):
            try:
                fdf = pd.read_parquet(f_path)
                if not fdf.empty:
                    frames.append(fdf)
            except Exception:
                pass
        else:
            missing_feature_paths.append(f"{ds}/{dep}: {f_path}")
        if os.path.exists(i_path):
            try:
                idf = pd.read_parquet(i_path)
                if not idf.empty:
                    idx_frames.append(idf)
            except Exception:
                pass

    missing_index_paths = []
    for item in ctx.scope:
        ds = item["dataset_id"]
        dep = item["deployment_id"]
        _f_path, i_path = _feature_deployment_paths(ctx, ds, dep)
        if not os.path.exists(i_path):
            missing_index_paths.append(f"{ds}/{dep}: {i_path}")

    if missing_feature_paths or missing_index_paths:
        if missing_feature_paths:
            print("[features-merge] missing deployment feature parquet files:")
            for line in missing_feature_paths:
                print(f"  - {line}")
        if missing_index_paths:
            print("[features-merge] missing deployment feature index parquet files:")
            for line in missing_index_paths:
                print(f"  - {line}")
        raise ValueError("Incomplete deployment-level feature outputs; refusing to merge a partial run.")

    if not frames:
        raise ValueError("No deployment-level feature parquet files found to merge.")

    feat_df = pd.concat(frames, ignore_index=True)
    idx_df = pd.concat(idx_frames, ignore_index=True) if idx_frames else pd.DataFrame()

    max_chunks_total = ctx.run_cfg.get("max_chunks_total")
    if max_chunks_total is not None:
        max_chunks_total = int(max_chunks_total)
        feat_df = feat_df.sort_values(["dataset_id", "deployment_id", "window_start"]).head(max_chunks_total).reset_index(drop=True)
        if not idx_df.empty:
            idx_df = idx_df.sort_values(["dataset_id", "deployment_id", "window_start"]).head(max_chunks_total).reset_index(drop=True)

    # normalize only numeric feature columns (excluding metadata)
    meta_cols = {"dataset_id", "deployment_id", "window_start", "window_end", "window_seconds"}
    numeric_cols = [
        c for c in feat_df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(feat_df[c])
    ]

    norm_level = str(ctx.run_cfg.get("normalization_level", "deployment")).lower()

    def _zscore(s):
        denom = float(s.std(ddof=0))
        if not np.isfinite(denom) or denom == 0:
            denom = 1.0
        return (s - s.mean()) / denom

    if norm_level == "deployment":
        feat_df[numeric_cols] = feat_df.groupby("deployment_id")[numeric_cols].transform(_zscore)
    elif norm_level == "dataset":
        feat_df[numeric_cols] = feat_df.groupby("dataset_id")[numeric_cols].transform(_zscore)

    raw_path, idx_path = _feature_paths(ctx)
    _ensure_dir(os.path.dirname(raw_path))
    feat_df.to_parquet(raw_path, index=False)
    idx_df.to_parquet(idx_path, index=False)

    feature_count = len([c for c in numeric_cols if not c.startswith("__")])
    _ensure_dir(os.path.dirname(args.log_output))
    with open(args.log_output, "w") as f:
        f.write(json.dumps({
            "feature_count": feature_count,
            "rows": len(feat_df),
            "normalization_level": norm_level,
            "catch22_available": CATCH22_AVAILABLE,
            "max_chunks_total": max_chunks_total,
            "merged_deployments_with_features": len(frames),
        }, indent=2))

    print(f"[features-merge] wrote {raw_path}")
    print(f"[features-merge] wrote {idx_path}")


def cmd_correlate(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    raw_path, _idx_path = _feature_paths(ctx)
    feat_df = pd.read_parquet(raw_path)

    meta_cols = ["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds"]
    feat_cols = [
        c for c in feat_df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(feat_df[c]) and not c.startswith("__")
    ]

    corr = feat_df[feat_cols].corr().abs()
    threshold = float(ctx.run_cfg.get("corr_threshold", 0.80))
    keep_std_features = bool(ctx.run_cfg.get("keep_std_features", False))

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    protected = set()
    if keep_std_features:
        protected = {c for c in feat_cols if c.endswith("__std")}
        if protected:
            to_drop = [c for c in to_drop if c not in protected]
    kept_cols = [c for c in feat_cols if c not in to_drop]

    filtered_df = feat_df[meta_cols + [c for c in feat_df.columns if c in kept_cols or c.startswith("__")]].copy()

    out_dir = os.path.join(ctx.output_root, "features")
    _ensure_dir(out_dir)
    corr_path = os.path.join(out_dir, "feature_correlation_matrix.parquet")
    drop_path = os.path.join(out_dir, "dropped_correlated_features.parquet")
    filt_path = os.path.join(out_dir, "features_filtered.parquet")
    report_path = os.path.join(out_dir, "feature_filter_report.json")
    corr_plot = os.path.join(out_dir, "correlogram.html")

    corr.to_parquet(corr_path)
    pd.DataFrame({"dropped_feature": to_drop}).to_parquet(drop_path, index=False)
    filtered_df.to_parquet(filt_path, index=False)
    if not _keep_raw_merged_features(ctx.run_cfg) and os.path.exists(raw_path):
        try:
            os.remove(raw_path)
            print(f"[correlate] removed raw merged feature parquet {raw_path}")
        except OSError as exc:
            print(f"[correlate] warning: could not remove raw merged feature parquet {raw_path}: {exc}")

    report = {
        "corr_threshold": threshold,
        "input_feature_count": len(feat_cols),
        "dropped_feature_count": len(to_drop),
        "kept_feature_count": len(kept_cols),
        "keep_std_features": keep_std_features,
        "protected_std_feature_count": len(protected),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # correlogram on top 120 original features (includes dropped so removals are visible)
    view_cols = feat_cols[:120]
    if view_cols:
        corr_view = corr.loc[view_cols, view_cols]
        fig = px.imshow(
            corr_view,
            x=view_cols,
            y=view_cols,
            color_continuous_scale="Viridis",
            zmin=0,
            zmax=1,
        )
        dropped_in_view = [c for c in view_cols if c in set(to_drop)]
        if dropped_in_view:
            tick_text = [f"<b>{c}</b>" if c in dropped_in_view else c for c in view_cols]
            fig.update_xaxes(tickmode="array", tickvals=view_cols, ticktext=tick_text)
            fig.update_yaxes(tickmode="array", tickvals=view_cols, ticktext=tick_text)
            fig.add_trace(
                go.Scatter(
                    x=dropped_in_view,
                    y=dropped_in_view,
                    mode="markers",
                    marker=dict(symbol="x", color="red", size=10),
                    name="Dropped (corr > threshold)",
                    hovertemplate="Dropped feature: %{x}<extra></extra>",
                )
            )
        fig.update_layout(title="Feature Correlation (absolute)")
        fig.write_html(corr_plot)

    print(f"[correlate] wrote {corr_path}, {drop_path}, {filt_path}, {report_path}")


def _pick_k_auto(x_scaled: np.ndarray, max_k: int) -> int:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for clustering")
    from sklearn.metrics import silhouette_score

    n = x_scaled.shape[0]
    best_k = 2
    best_score = -1
    print(f"[cluster] auto-k search starting | n_rows={n} max_k={max_k} | RSS={_get_rss_gb():.2f} GB")
    for k in range(2, min(max_k, n - 1) + 1):
        print(f"[cluster] auto-k testing k={k} | RSS={_get_rss_gb():.2f} GB")
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(x_scaled)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(x_scaled, labels)
        print(f"[cluster] auto-k k={k} silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"[cluster] auto-k selected k={best_k} (silhouette={best_score:.4f})")
    return best_k


def cmd_cluster(args):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for cluster step")
    ctx = _resolve_run_context(args.config, args.run_name)
    print(f"[cluster] start run={ctx.run_name} | output_root={ctx.output_root} | RSS={_get_rss_gb():.2f} GB")
    filt_path = os.path.join(ctx.output_root, "features", "features_filtered.parquet")
    base_df = pd.read_parquet(filt_path)
    print(f"[cluster] loaded filtered features: rows={len(base_df)} cols={len(base_df.columns)} | RSS={_get_rss_gb():.2f} GB")

    meta_cols = ["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds"]
    clustering_scope = _clustering_window_scope(ctx.run_cfg)
    if clustering_scope in {"labeled_windows_only", "positive_labeled_windows_only"}:
        supervised_cfg = ctx.run_cfg.get("supervised") or {}
        label_keys = list(supervised_cfg.get("label_event_keys") or [])
        label_prefixes = _normalized_label_prefixes(ctx.run_cfg)
        label_group_map = _build_supervised_label_group_map(supervised_cfg)
        drop_label_set = _build_supervised_drop_label_set(supervised_cfg)
        scoped_df, _scope_report = _collect_supervised_labels(
            ctx,
            base_df.copy(),
            label_keys,
            label_prefixes,
            label_group_map=label_group_map,
            drop_label_set=drop_label_set,
        )
        observed = scoped_df.get("observed_label")
        if observed is None:
            raise RuntimeError("clustering_window_scope=labeled_windows_only requested but no observed labels were collected.")
        observed = observed.astype(str)
        keep_mask = _observed_label_keep_mask(scoped_df.get("observed_label"))
        positive_label = None
        if clustering_scope == "positive_labeled_windows_only":
            positive_label = _detect_supervised_positive_label(observed.loc[keep_mask].tolist())
            if positive_label is None:
                raise RuntimeError(
                    "clustering_window_scope=positive_labeled_windows_only requested but no positive supervised label could be resolved."
                )
            positive_lower = str(positive_label).strip().lower()
            keep_mask = keep_mask & (observed.str.strip().str.lower() == positive_lower)
        kept_rows = int(keep_mask.sum())
        print(
            f"[cluster] applying clustering_window_scope={clustering_scope}"
            + (f" positive_label={positive_label}" if positive_label is not None else "")
            + " | "
            f"kept_rows={kept_rows} dropped_rows={len(scoped_df) - kept_rows} | RSS={_get_rss_gb():.2f} GB"
        )
        base_df = scoped_df.loc[keep_mask].copy()
        if base_df.empty:
            raise RuntimeError(f"clustering_window_scope={clustering_scope} left no rows to cluster.")

    run_pca = bool(ctx.run_cfg.get("run_pca", True))
    run_tsne = bool(ctx.run_cfg.get("run_tsne", False))
    run_umap = bool(ctx.run_cfg.get("run_umap", False))
    cluster_passes = _cluster_passes_from_run_cfg(ctx.run_cfg)
    clustered_frames = []
    interpret_passes = []

    for pass_cfg in cluster_passes:
        pass_name = _safe_slug(pass_cfg["name"])
        pass_spec = pass_cfg["channel_feature_spec"]
        feat_cols = _feature_columns_for_pass(base_df, pass_spec)
        if not feat_cols:
            raise ValueError(f"No feature columns available for cluster pass '{pass_name}'")
        print(f"[cluster] pass={pass_name} feature columns={len(feat_cols)}")

        work_df = base_df[meta_cols + [c for c in base_df.columns if c in feat_cols or c.startswith('__')]].copy()
        x = work_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        print(f"[cluster] pass={pass_name} scaled matrix shape={x_scaled.shape} | RSS={_get_rss_gb():.2f} GB")

        n_clusters = pass_cfg.get("n_clusters")
        if n_clusters in (None, "", "auto"):
            n_clusters = _pick_k_auto(x_scaled, int(pass_cfg.get("max_k", 10)))
        n_clusters = int(n_clusters)
        print(f"[cluster] pass={pass_name} running kmeans n_clusters={n_clusters}")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_idx = kmeans.fit_predict(x_scaled)
        work_df["cluster_id"] = cluster_idx
        print(f"[cluster] pass={pass_name} kmeans complete | RSS={_get_rss_gb():.2f} GB")

        primary_col = "__primary_channel_mean_raw"
        if primary_col not in work_df.columns:
            work_df[primary_col] = np.nan
        means_df = (
            work_df.groupby("cluster_id", as_index=False)[primary_col]
            .mean()
            .rename(columns={primary_col: "primary_mean"})
        )
        means_df = means_df.sort_values(["primary_mean", "cluster_id"], na_position="last").reset_index(drop=True)
        rank_map = {int(cid): i + 1 for i, cid in enumerate(means_df["cluster_id"].tolist())}
        work_df["cluster_rank"] = work_df["cluster_id"].map(rank_map).astype(int)
        work_df["cluster_pass"] = pass_name
        work_df["cluster_pass_n_clusters"] = n_clusters

        interpret = {
            "cluster_pass": pass_name,
            "n_rows": len(work_df),
            "n_features": len(feat_cols),
            "n_clusters": n_clusters,
            "clustering_window_scope": clustering_scope,
            "tsne_available": TSNE_AVAILABLE,
            "umap_available": UMAP_AVAILABLE,
            "cluster_ranking": means_df.assign(
                cluster_id=lambda x: x["cluster_id"].astype(int),
                cluster_rank=lambda x: x["cluster_id"].map(rank_map).astype(int),
            ).to_dict(orient="records"),
        }

        if run_pca:
            print(f"[cluster] pass={pass_name} PCA start")
            pca = PCA(n_components=min(10, x_scaled.shape[1], x_scaled.shape[0]))
            p = pca.fit_transform(x_scaled)
            work_df["pca_x"] = p[:, 0]
            work_df["pca_y"] = p[:, 1] if p.shape[1] > 1 else 0.0
            loadings = pd.DataFrame(pca.components_.T, index=feat_cols, columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])])
            top_load = []
            for pc in ["PC1", "PC2", "PC3", "PC4"]:
                if pc in loadings.columns:
                    top = loadings[pc].abs().sort_values(ascending=False).head(10).index.tolist()
                    top_load.append({"pc": pc, "top_features": top})
            interpret["pca_top_features"] = top_load
            load_path = os.path.join(ctx.output_root, "clustering", f"pca_loadings_{pass_name}.parquet")
            _ensure_dir(os.path.dirname(load_path))
            loadings.reset_index(names="feature").to_parquet(load_path, index=False)
            print(f"[cluster] pass={pass_name} PCA done, wrote loadings {load_path}")

        if run_tsne and TSNE_AVAILABLE:
            print(f"[cluster] pass={pass_name} t-SNE start")
            tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
            t = tsne.fit_transform(x_scaled)
            work_df["tsne_x"] = t[:, 0]
            work_df["tsne_y"] = t[:, 1]
            print(f"[cluster] pass={pass_name} t-SNE done")

        if run_umap and UMAP_AVAILABLE:
            umap_n_neighbors = int(ctx.run_cfg.get("umap_n_neighbors", 15))
            umap_min_dist = float(ctx.run_cfg.get("umap_min_dist", 0.1))
            umap_metric = str(ctx.run_cfg.get("umap_metric", "euclidean"))
            print(
                f"[cluster] pass={pass_name} UMAP start | n_neighbors={umap_n_neighbors} "
                f"min_dist={umap_min_dist} metric={umap_metric} | RSS={_get_rss_gb():.2f} GB"
            )
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
            )
            u = reducer.fit_transform(x_scaled)
            work_df["umap_x"] = u[:, 0]
            work_df["umap_y"] = u[:, 1]
            print(f"[cluster] pass={pass_name} UMAP done | RSS={_get_rss_gb():.2f} GB")

        input_keys = sorted(pass_spec.keys())
        input_token = "+".join(k.split(".", 1)[1] for k in input_keys)
        feat_levels = sorted({v["feature_set"] for v in pass_spec.values()})
        feat_token = "-".join(feat_levels)
        if str(ctx.run_cfg.get("cluster_length_mode", "fixed")).lower() == "fixed":
            win_token = f"fixed_W{int(ctx.run_cfg.get('duration_s', 60))}s"
        else:
            win_token = (
                f"var_S{int(ctx.run_cfg.get('stride_s', 10))}s_"
                f"W{int(ctx.run_cfg.get('window_s', 60))}s"
            )

        for method in ["PCA", "TSNE", "UMAP"]:
            if method == "TSNE" and not run_tsne:
                continue
            if method == "UMAP" and not run_umap:
                continue
            key_col = f"cluster_key_{method.lower()}"
            work_df[key_col] = work_df["cluster_rank"].map(lambda r: f"{r}_{pass_name}_{input_token}_{feat_token}_{win_token}_{method}")

        clustered_frames.append(work_df)
        interpret_passes.append(interpret)

    if not clustered_frames:
        raise ValueError("No clustering passes produced output")

    df = pd.concat(clustered_frames, ignore_index=True, sort=False)
    interpret = {
        "cluster_passes": interpret_passes,
        "run_pca": run_pca,
        "run_tsne": run_tsne,
        "run_umap": run_umap,
    }

    out_dir = os.path.join(ctx.output_root, "clustering")
    _ensure_dir(out_dir)
    clustered_path = os.path.join(out_dir, "clustered_windows.parquet")
    gap_report_path = os.path.join(out_dir, "cluster_gap_report.parquet")
    algorithmic_overlap_path = os.path.join(out_dir, "algorithmic_overlap_by_cluster.parquet")
    report_path = os.path.join(out_dir, "interpretability_report.json")

    algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
    algo_enabled = bool(algo_cfg.get("enabled", False))
    algo_detailed_path, algo_event_path = _algorithmic_segment_merge_paths(ctx)
    algo_label_path = _algorithmic_label_timeseries_merge_path(ctx)
    if algo_enabled:
        if os.path.exists(algo_event_path):
            try:
                algo_event_df = pd.read_parquet(algo_event_path)
                if not algo_event_df.empty:
                    t0 = time.monotonic()
                    print(
                        f"[cluster] annotating windows with algorithmic segment overlap | "
                        f"event_rows={len(algo_event_df)} window_rows={len(df)} | RSS={_get_rss_gb():.2f} GB"
                    )
                    df, overlap_summary = _annotate_windows_with_algorithmic_segments(df, algo_event_df)
                    if not overlap_summary.empty:
                        overlap_summary.to_parquet(algorithmic_overlap_path, index=False)
                        interpret["algorithmic_overlap_summary"] = algorithmic_overlap_path
                        print(
                            f"[cluster] wrote algorithmic overlap summary {algorithmic_overlap_path} | "
                            f"rows={len(overlap_summary)} elapsed={time.monotonic() - t0:.1f}s | RSS={_get_rss_gb():.2f} GB"
                        )
                    else:
                        print(
                            f"[cluster] algorithmic overlap annotation complete with no summary rows | "
                            f"elapsed={time.monotonic() - t0:.1f}s | RSS={_get_rss_gb():.2f} GB"
                        )
            except Exception as e:
                interpret["algorithmic_overlap_error"] = f"{type(e).__name__}: {e}"
                print(f"[cluster] warning: failed algorithmic overlap annotation: {type(e).__name__}: {e}")
        if os.path.exists(algo_detailed_path):
            try:
                algo_segment_df = pd.read_parquet(algo_detailed_path)
                if not algo_segment_df.empty:
                    t0 = time.monotonic()
                    print(
                        f"[cluster] annotating windows with segment-based algorithmic labels | "
                        f"segment_rows={len(algo_segment_df)} window_rows={len(df)} | RSS={_get_rss_gb():.2f} GB"
                    )
                    df = _annotate_windows_with_algorithmic_labels(
                        df,
                        algo_segment_df,
                        default_code=int((algo_cfg.get("label_codes") or {}).get("not_sleep", 0)),
                        default_name=str((algo_cfg.get("label_names") or {}).get("not_sleep", "find_rest.not_sleep")),
                    )
                    interpret["algorithmic_label_path"] = algo_detailed_path
                    interpret["algorithmic_label_mode"] = "segment_overlap_approximation"
                    print(
                        f"[cluster] segment-based algorithmic label annotation complete | "
                        f"elapsed={time.monotonic() - t0:.1f}s | RSS={_get_rss_gb():.2f} GB"
                    )
            except Exception as e:
                interpret["algorithmic_label_error"] = f"{type(e).__name__}: {e}"
                print(f"[cluster] warning: failed algorithmic label annotation: {type(e).__name__}: {e}")
    else:
        print("[cluster] algorithmic segment annotation disabled for this run (algorithmic_segments.enabled=false)")

    t0 = time.monotonic()
    print(
        f"[cluster] writing clustered windows parquet | "
        f"rows={len(df)} cols={len(df.columns)} | RSS={_get_rss_gb():.2f} GB"
    )
    df.to_parquet(clustered_path, index=False)
    print(
        f"[cluster] clustered windows parquet write complete | "
        f"elapsed={time.monotonic() - t0:.1f}s | RSS={_get_rss_gb():.2f} GB"
    )

    # Coverage audit: quantify true temporal gaps in clustered windows per deployment.
    t0 = time.monotonic()
    print(
        f"[cluster] starting gap audit | deployments={df[['dataset_id', 'deployment_id']].drop_duplicates().shape[0]} "
        f"| RSS={_get_rss_gb():.2f} GB"
    )
    gap_rows = []
    for (ds, dep), grp in df.groupby(["dataset_id", "deployment_id"], sort=False):
        g = grp[["window_start", "window_end"]].drop_duplicates().sort_values("window_start")
        starts = pd.to_datetime(g["window_start"], errors="coerce")
        ends = pd.to_datetime(g["window_end"], errors="coerce")
        valid = starts.notna() & ends.notna()
        starts = starts[valid].reset_index(drop=True)
        ends = ends[valid].reset_index(drop=True)
        if len(starts) < 2:
            gap_rows.append(
                {
                    "dataset_id": ds,
                    "deployment_id": dep,
                    "n_windows": int(len(starts)),
                    "max_gap_seconds": 0.0,
                    "median_step_seconds": np.nan,
                    "macro_gap_count": 0,
                }
            )
            continue
        steps = (starts.iloc[1:].reset_index(drop=True) - starts.iloc[:-1].reset_index(drop=True)).dt.total_seconds()
        gaps = (starts.iloc[1:].reset_index(drop=True) - ends.iloc[:-1].reset_index(drop=True)).dt.total_seconds()
        median_step = float(np.nanmedian(steps.to_numpy(dtype=float))) if len(steps) else float("nan")
        macro_threshold = max(5.0, median_step * 3.0) if np.isfinite(median_step) else 5.0
        gap_rows.append(
            {
                "dataset_id": ds,
                "deployment_id": dep,
                "n_windows": int(len(starts)),
                "max_gap_seconds": float(np.nanmax(gaps.to_numpy(dtype=float))) if len(gaps) else 0.0,
                "median_step_seconds": median_step,
                "macro_gap_count": int((gaps > macro_threshold).sum()) if len(gaps) else 0,
            }
        )
    pd.DataFrame(gap_rows).to_parquet(gap_report_path, index=False)
    print(
        f"[cluster] gap audit complete | rows={len(gap_rows)} "
        f"elapsed={time.monotonic() - t0:.1f}s | RSS={_get_rss_gb():.2f} GB"
    )

    with open(report_path, "w") as f:
        json.dump(interpret, f, indent=2)

    print(f"[cluster] wrote {clustered_path}")
    print(f"[cluster] wrote {gap_report_path}")


def _cluster_color_map(keys: List[str]) -> Dict[str, str]:
    return build_ordered_cluster_color_map(keys)


def cmd_export(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    clustered_path = os.path.join(ctx.output_root, "clustering", "clustered_windows.parquet")
    cdf = pd.read_parquet(clustered_path)
    memory_log_interval_s = float(ctx.run_cfg.get("memory_log_interval_s", 30))
    last_log = time.monotonic()

    # use pca key as default event key column
    key_col = "cluster_key_pca"
    if key_col not in cdf.columns:
        key_col = [c for c in cdf.columns if c.startswith("cluster_key_")][0]

    total_events = 0
    deployments_written = 0

    grouped = list(cdf.groupby(["dataset_id", "deployment_id"]))
    n_grouped = len(grouped)
    for i, ((ds, dep), g) in enumerate(grouped, start=1):
        print(f"[export] [{i}/{n_grouped}] start {ds}/{dep} | windows={len(g)} | RSS={_get_rss_gb():.2f} GB")
        pkl_path = _data_pkl_path(ctx, ds, dep)
        if not os.path.exists(pkl_path):
            print(f"[export] [{i}/{n_grouped}] skip {ds}/{dep}: missing data.pkl")
            continue

        with open(pkl_path, "rb") as f:
            data_pkl = pickle.load(f)
        deployment_tz = _deployment_timezone_name(data_pkl)

        if not hasattr(data_pkl, "event_data") or data_pkl.event_data is None:
            data_pkl.event_data = pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value"])

        ev = data_pkl.event_data.copy()
        if "key" not in ev.columns:
            ev["key"] = pd.NA
        ev = ev[~ev["key"].astype(str).str.contains("_PCA|_TSNE|_UMAP", na=False)].copy()

        new_rows = []
        for _, row in g.iterrows():
            start = _normalize_datetime_series_to_timezone(pd.Series([row["window_start"]]), deployment_tz).iloc[0]
            end = _normalize_datetime_series_to_timezone(pd.Series([row["window_end"]]), deployment_tz).iloc[0]
            dur = float((end - start).total_seconds()) if pd.notna(start) and pd.notna(end) else float(row.get("window_seconds", 0.0))
            k = str(row[key_col])
            new_rows.append({
                "datetime": start,
                "key": k,
                "short_description": f"Cluster state {k}",
                "type": "state",
                "duration": dur,
                "value": row.get("cluster_rank", np.nan),
            })

        new_df = pd.DataFrame(new_rows)
        ev = pd.concat([ev, new_df], ignore_index=True)
        ev["datetime"] = _normalize_datetime_series_to_timezone(ev["datetime"], deployment_tz)
        ev = ev.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        data_pkl.event_data = ev

        if not hasattr(data_pkl, "event_manager") or data_pkl.event_manager is None:
            data_pkl.event_manager = {}
        keys = sorted(new_df["key"].dropna().astype(str).unique().tolist())
        data_pkl.event_manager["cluster_states"] = {
            "keys": keys,
            "description": "Cluster states from dataset-level clustering pipeline",
            "color_map": _cluster_color_map(keys),
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(data_pkl, f)

        deployments_written += 1
        total_events += len(new_df)
        if _should_log(last_log, memory_log_interval_s):
            print(
                f"[export] progress {i}/{n_grouped} deployments | "
                f"last={ds}/{dep} total_events={total_events} | RSS={_get_rss_gb():.2f} GB"
            )
            last_log = time.monotonic()

    _ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        f.write(
            f"deployments_written={deployments_written}\n"
            f"total_events={total_events}\n"
        )
    print(f"[export] wrote marker {args.output}")


def _normalized_label_prefixes(run_cfg: Dict) -> List[str]:
    prefixes = list(((run_cfg.get("supervised") or {}).get("label_prefixes") or []))
    if prefixes:
        return [str(x) for x in prefixes if str(x).strip()]
    return ["active_", "calm_", "motionless_"]


def _normalize_label_name(value) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _load_supervised_label_alias_map() -> Dict[str, str]:
    global _SUPERVISED_LABEL_ALIAS_CACHE
    if _SUPERVISED_LABEL_ALIAS_CACHE is not None:
        return _SUPERVISED_LABEL_ALIAS_CACHE

    candidate_paths = [
        os.path.join(PROJECT_ROOT, "label_db.json"),
        os.path.join(str(PACKAGE_ROOT.parent), "label_db.json"),
        os.path.join(str(PACKAGE_ROOT), "label_db.json"),
    ]
    alias_map = {}
    label_db = None
    for label_db_path in candidate_paths:
        if not os.path.exists(label_db_path):
            continue
        try:
            with open(label_db_path, "r", encoding="utf-8") as f:
                label_db = json.load(f)
            break
        except Exception:
            continue
    if not isinstance(label_db, dict):
        _SUPERVISED_LABEL_ALIAS_CACHE = alias_map
        return alias_map

    by_key = label_db.get("by_key") or {}
    alias_sets = {}
    for group_aliases in (label_db.get("aliases") or {}).values():
        if not isinstance(group_aliases, dict):
            continue
        for alias, resolved in group_aliases.items():
            alias_norm = _normalize_label_name(alias)
            resolved_key = str(resolved).strip()
            if alias_norm and resolved_key:
                alias_sets.setdefault(alias_norm, set()).add(resolved_key)

    for canonical_key in by_key.keys():
        canonical_key = str(canonical_key).strip()
        if canonical_key:
            alias_map[canonical_key] = canonical_key
            alias_norm = _normalize_label_name(canonical_key)
            if alias_norm:
                alias_sets.setdefault(alias_norm, set()).add(canonical_key)

    for alias_norm, resolved_keys in alias_sets.items():
        if len(resolved_keys) == 1:
            alias_map[alias_norm] = next(iter(resolved_keys))

    _SUPERVISED_LABEL_ALIAS_CACHE = alias_map
    return alias_map


def _canonicalize_supervised_event_key(key: str) -> str:
    key_text = str(key).strip()
    if not key_text:
        return key_text

    alias_map = _load_supervised_label_alias_map()
    if key_text in alias_map:
        return alias_map[key_text]

    return alias_map.get(_normalize_label_name(key_text), key_text)


def _build_supervised_label_group_map(supervised_cfg: Dict) -> Dict[str, str]:
    out = {}
    raw_groups = (supervised_cfg or {}).get("label_groups") or {}
    if not isinstance(raw_groups, dict):
        return out

    for group_label, keys in raw_groups.items():
        group_text = str(group_label).strip()
        if not group_text:
            continue
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, (list, tuple, set)):
            continue
        for key in keys:
            key_text = str(key).strip()
            if not key_text:
                continue
            out[key_text] = group_text
            out[_canonicalize_supervised_event_key(key_text)] = group_text
            norm = _normalize_label_name(key_text)
            if norm:
                out[norm] = group_text
    return out


def _map_label_to_supervised_group(label: Any, label_group_map: Dict[str, str] | None = None) -> str:
    text = str(label).strip()
    if not text:
        return text
    group_map = label_group_map or {}
    for candidate in (
        text,
        _canonicalize_supervised_event_key(text),
        _normalize_label_name(text),
    ):
        if candidate in group_map:
            return str(group_map[candidate]).strip() or text
    return text


def _build_supervised_drop_label_set(supervised_cfg: Dict) -> set[str]:
    out = set()
    raw_drop = (supervised_cfg or {}).get("drop_labels") or []
    if isinstance(raw_drop, str):
        raw_drop = [raw_drop]
    if not isinstance(raw_drop, (list, tuple, set)):
        return out

    ignored_non_quality = []
    for label in raw_drop:
        text = str(label).strip()
        if not text:
            continue
        norm = _normalize_label_name(text)
        # drop_labels is reserved for data-quality exclusions only.
        is_quality_exclusion = ("unscorable" in norm) or ("unclear" == norm) or ("unclear" in norm)
        if not is_quality_exclusion:
            ignored_non_quality.append(text)
            continue
        out.add(text)
        out.add(_canonicalize_supervised_event_key(text))
        if norm:
            out.add(norm)
    if ignored_non_quality:
        print(
            "[supervised] warning: drop_labels only supports quality exclusions "
            f"(unscorable/unclear). Ignoring: {sorted(set(ignored_non_quality))}"
        )
    return {str(x) for x in out if str(x).strip()}


def _supervised_prediction_scope(supervised_cfg: Dict[str, Any] | None) -> str:
    scope = str((supervised_cfg or {}).get("prediction_scope") or "all_windows").strip().lower()
    if scope not in {"all_windows", "labeled_windows_only"}:
        raise ValueError(
            f"Unsupported supervised prediction_scope '{scope}'. "
            f"Expected 'all_windows' or 'labeled_windows_only'."
        )
    return scope


def _clustering_window_scope(run_cfg: Dict[str, Any] | None) -> str:
    scope = str((run_cfg or {}).get("clustering_window_scope") or "all_windows").strip().lower()
    if scope not in {"all_windows", "labeled_windows_only", "positive_labeled_windows_only"}:
        raise ValueError(
            f"Unsupported clustering_window_scope '{scope}'. "
            f"Expected 'all_windows', 'labeled_windows_only', or 'positive_labeled_windows_only'."
        )
    return scope


def _filter_supervised_prediction_rows(df: pd.DataFrame, supervised_cfg: Dict[str, Any] | None) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    work = df.copy()
    if "quality_excluded" in work.columns:
        work = work.loc[~work["quality_excluded"].fillna(False).astype(bool)].copy()
    scope = _supervised_prediction_scope(supervised_cfg)
    if scope == "all_windows":
        return work
    observed = work.get("observed_label")
    if observed is None:
        return work.iloc[0:0].copy()
    keep_mask = _observed_label_keep_mask(work.get("observed_label"))
    return work.loc[keep_mask].copy()


def _resolve_supervised_target_label(
    raw_key: str,
    label_keys: List[str],
    label_prefixes: List[str],
    label_group_map: Dict[str, str],
    drop_label_set: set[str] | None = None,
) -> str | None:
    raw_text = str(raw_key).strip()
    canonical_key = _canonicalize_supervised_event_key(raw_text)
    norm_key = _normalize_label_name(raw_text)
    drop_set = drop_label_set or set()

    def _is_dropped(label_value: str | None) -> bool:
        if label_value is None:
            return False
        label_text = str(label_value).strip()
        if not label_text:
            return False
        candidates = {
            raw_text,
            canonical_key,
            norm_key,
            label_text,
            _normalize_label_name(label_text),
            _canonicalize_supervised_event_key(label_text),
        }
        return any(candidate in drop_set for candidate in candidates if candidate)

    for candidate in (raw_text, canonical_key, norm_key):
        if candidate in label_group_map:
            resolved = label_group_map[candidate]
            return None if _is_dropped(resolved) else resolved

    if label_keys:
        wanted = {str(x) for x in label_keys}
        if raw_text in wanted:
            return None if _is_dropped(raw_text) else raw_text
        if canonical_key in wanted:
            return None if _is_dropped(canonical_key) else canonical_key
        return None

    if any(str(canonical_key).startswith(prefix) for prefix in label_prefixes):
        return None if _is_dropped(canonical_key) else canonical_key
    if any(str(raw_text).startswith(prefix) for prefix in label_prefixes):
        return None if _is_dropped(raw_text) else raw_text
    return None


def _resolve_event_end_for_supervised(event_row: pd.Series):
    start = pd.to_datetime(event_row.get("datetime"), errors="coerce")
    if pd.isna(start):
        return pd.NaT, pd.NaT
    for duration_key in ("duration", "duration_pos", "duration_sec", "duration_s"):
        if duration_key in event_row.index:
            duration_val = pd.to_numeric(pd.Series([event_row.get(duration_key)]), errors="coerce").iloc[0]
            if pd.notna(duration_val) and float(duration_val) > 0:
                return pd.Timestamp(start), pd.Timestamp(start) + pd.to_timedelta(float(duration_val), unit="s")
    for end_key in ("end_datetime", "end_time", "datetime_end", "end"):
        if end_key in event_row.index:
            end_val = pd.to_datetime(event_row.get(end_key), errors="coerce")
            if pd.notna(end_val):
                return pd.Timestamp(start), pd.Timestamp(end_val)
    return pd.Timestamp(start), pd.NaT


def _label_priority_sort_key(label: str, priority_prefixes: List[str]):
    label = str(label)
    for idx, prefix in enumerate(priority_prefixes):
        if label.startswith(prefix):
            return (idx, label)
    return (len(priority_prefixes), label)


def _derive_window_labels_from_events(
    cdf: pd.DataFrame,
    event_data: pd.DataFrame,
    label_keys: List[str],
    label_prefixes: List[str],
    label_group_map: Dict[str, str] | None = None,
    drop_label_set: set[str] | None = None,
) -> Tuple[pd.Series, Dict[str, int], pd.Series]:
    empty_report = {
        "eligible_event_rows": 0,
        "quality_exclusion_event_rows": 0,
        "windows_with_any_overlap": 0,
        "windows_quality_excluded": 0,
        "windows_with_multiple_labels": 0,
        "windows_with_ties": 0,
        "windows_labeled": 0,
    }
    if event_data is None or event_data.empty or "datetime" not in event_data.columns or "key" not in event_data.columns:
        return (
            pd.Series(np.nan, index=cdf.index, dtype=object),
            empty_report,
            pd.Series(False, index=cdf.index, dtype=bool),
        )

    edf = event_data.copy()
    edf["datetime"] = pd.to_datetime(edf["datetime"], errors="coerce")
    edf = edf.dropna(subset=["datetime"]).copy()
    edf["key"] = edf["key"].astype(str)
    group_map = label_group_map or {}
    drop_set = drop_label_set or set()
    def _is_quality_excluded_event_key(raw_key: Any) -> bool:
        text = str(raw_key).strip()
        if not text:
            return False
        candidates = {
            text,
            _canonicalize_supervised_event_key(text),
            _normalize_label_name(text),
        }
        return any(candidate in drop_set for candidate in candidates if candidate)
    edf["quality_excluded_key"] = edf["key"].apply(_is_quality_excluded_event_key)
    edf["supervised_key"] = edf["key"].apply(
        lambda x: _resolve_supervised_target_label(x, label_keys, label_prefixes, group_map, drop_set)
    )
    eligible_mask = edf["supervised_key"].notna() & edf["supervised_key"].astype(str).str.strip().ne("")

    if "type" in edf.columns:
        type_text = edf["type"].fillna("").astype(str).str.lower()
        duration_s = pd.to_numeric(edf.get("duration"), errors="coerce") if "duration" in edf.columns else pd.Series(np.nan, index=edf.index)
        end_like = pd.Series(False, index=edf.index)
        for end_key in ("end_datetime", "end_time", "datetime_end", "end"):
            if end_key in edf.columns:
                end_like = end_like | pd.to_datetime(edf[end_key], errors="coerce").notna()
        state_like_mask = type_text.eq("state")
        timed_event_mask = type_text.eq("event") & ((duration_s > 0) | end_like)
        typed_mask = state_like_mask | timed_event_mask
        edf = edf[typed_mask].copy()
    else:
        edf = edf.copy()

    if edf.empty:
        return (
            pd.Series(np.nan, index=cdf.index, dtype=object),
            empty_report,
            pd.Series(False, index=cdf.index, dtype=bool),
        )

    intervals = []
    quality_exclusion_intervals = []
    for _, row in edf.iterrows():
        start, end = _resolve_event_end_for_supervised(row)
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        if bool(row.get("quality_excluded_key", False)):
            quality_exclusion_intervals.append((start, end))
        label_key = str(row.get("supervised_key") or "").strip()
        if label_key:
            intervals.append((label_key, start, end))
    if not intervals and not quality_exclusion_intervals:
        return (
            pd.Series(np.nan, index=cdf.index, dtype=object),
            empty_report,
            pd.Series(False, index=cdf.index, dtype=bool),
        )

    labels = pd.Series(np.nan, index=cdf.index, dtype=object)
    quality_excluded = pd.Series(False, index=cdf.index, dtype=bool)
    report = dict(empty_report)
    report["eligible_event_rows"] = len(intervals)
    report["quality_exclusion_event_rows"] = len(quality_exclusion_intervals)
    starts = pd.to_datetime(cdf["window_start"], errors="coerce")
    ends = pd.to_datetime(cdf["window_end"], errors="coerce")

    for i in cdf.index:
        s = starts.loc[i]
        e = ends.loc[i]
        if pd.isna(s) or pd.isna(e) or e <= s:
            continue
        excluded_here = False
        for ev_start, ev_end in quality_exclusion_intervals:
            left = max(pd.Timestamp(s), ev_start)
            right = min(pd.Timestamp(e), ev_end)
            if right > left:
                excluded_here = True
                break
        if excluded_here:
            quality_excluded.loc[i] = True
            report["windows_quality_excluded"] += 1
            continue
        overlap_by_label = {}
        for label, ev_start, ev_end in intervals:
            left = max(pd.Timestamp(s), ev_start)
            right = min(pd.Timestamp(e), ev_end)
            overlap_s = float((right - left).total_seconds()) if right > left else 0.0
            if overlap_s <= 0:
                continue
            overlap_by_label[label] = overlap_by_label.get(label, 0.0) + overlap_s

        if not overlap_by_label:
            continue
        report["windows_with_any_overlap"] += 1
        if len(overlap_by_label) > 1:
            report["windows_with_multiple_labels"] += 1

        max_overlap = max(overlap_by_label.values())
        winners = [label for label, sec in overlap_by_label.items() if sec == max_overlap]
        if len(winners) > 1:
            report["windows_with_ties"] += 1
        winners = sorted(winners, key=lambda label: _label_priority_sort_key(label, label_prefixes))
        labels.loc[i] = winners[0]
        report["windows_labeled"] += 1

    return labels, report, quality_excluded


def _build_window_key_series(df: pd.DataFrame) -> pd.Series:
    starts = pd.to_datetime(df["window_start"], errors="coerce").astype(str)
    return (
        df["dataset_id"].astype(str)
        + "|"
        + df["deployment_id"].astype(str)
        + "|"
        + starts
    )


def _build_supervised_variant_defs(
    ctx: RunContext,
    supervised_cfg: Dict,
    feat_cols: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
    source_feature_map: Dict[str, List[str]] = {}
    for feature_name in feat_cols:
        source_feature_map.setdefault(_feature_source_channel(feature_name), []).append(feature_name)

    variant_defs: List[Dict[str, Any]] = [
        {
            "variant_id": "rf_full",
            "variant_label": "RF (all channels)",
            "variant_type": "random_forest",
            "feature_cols": list(feat_cols),
            "variant_rank": 0,
        }
    ]
    skipped_variants: List[Dict[str, Any]] = []
    if bool(supervised_cfg.get("enable_channel_ablation", False)):
        ablation_mode = str(supervised_cfg.get("ablation_mode") or "leave_one_source_channel_out").strip().lower()
        if ablation_mode == "leave_one_source_channel_out":
            requested_ablation_channels = list(supervised_cfg.get("ablation_channels") or sorted(source_feature_map))
            for source_name in requested_ablation_channels:
                source_name = str(source_name)
                if source_name not in source_feature_map:
                    continue
                keep_cols = [c for c in feat_cols if _feature_source_channel(c) != source_name]
                if not keep_cols or len(keep_cols) == len(feat_cols):
                    continue
                variant_defs.append(
                    {
                        "variant_id": f"rf_without_{_safe_slug(source_name)}",
                        "variant_label": f"RF without {source_name}",
                        "variant_type": "random_forest_ablation",
                        "feature_cols": keep_cols,
                        "held_out_source": source_name,
                        "ablation_mode": ablation_mode,
                        "variant_rank": len(variant_defs),
                    }
                )
        elif ablation_mode == "grouped_forward_addition":
            raw_groups = list(supervised_cfg.get("ablation_feature_groups") or [])
            for raw_group in raw_groups:
                if not isinstance(raw_group, dict):
                    continue
                group_label = str(raw_group.get("label") or "").strip()
                group_channels = [str(ch).strip() for ch in (raw_group.get("channels") or []) if str(ch).strip()]
                if not group_label or not group_channels:
                    continue
                selected_cols = [c for c in feat_cols if _feature_source_channel(c) in set(group_channels)]
                missing_channels = [ch for ch in group_channels if ch not in source_feature_map]
                if not selected_cols:
                    skipped_variants.append(
                        {
                            "variant_id": f"rf_group_{_safe_slug(group_label)}",
                            "reason": "no_matching_feature_columns",
                            "channels": group_channels,
                        }
                    )
                    continue
                variant_defs.append(
                    {
                        "variant_id": f"rf_group_{_safe_slug(group_label)}",
                        "variant_label": f"RF {group_label}",
                        "variant_type": "random_forest_ablation",
                        "feature_cols": selected_cols,
                        "selected_sources": group_channels,
                        "missing_sources": missing_channels,
                        "ablation_mode": ablation_mode,
                        "variant_rank": len(variant_defs),
                    }
                )
        else:
            raise ValueError(
                f"Unsupported supervised ablation_mode '{ablation_mode}'. "
                f"Expected 'leave_one_source_channel_out' or 'grouped_forward_addition'."
            )

    include_algorithmic_baseline = bool(supervised_cfg.get("include_algorithmic_baseline", True)) and bool(
        (ctx.run_cfg.get("algorithmic_segments") or {}).get("enabled", False)
    )
    if include_algorithmic_baseline:
        variant_defs.append(
            {
                "variant_id": "algorithmic_find_rest",
                "variant_label": "Algorithmic (find_rest)",
                "variant_type": "algorithmic",
                "variant_rank": len(variant_defs),
            }
        )
    return variant_defs, skipped_variants, source_feature_map


def _collect_supervised_labels(
    ctx: RunContext,
    cdf: pd.DataFrame,
    label_keys: List[str],
    label_prefixes: List[str],
    label_group_map: Dict[str, str] | None = None,
    drop_label_set: set[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.Series(np.nan, index=cdf.index, dtype=object)
    quality_excluded = pd.Series(False, index=cdf.index, dtype=bool)
    label_scan_rows = []
    grouped = list(cdf.groupby(["dataset_id", "deployment_id"]))
    n_grouped = len(grouped)
    for i, ((ds, dep), g) in enumerate(grouped, start=1):
        print(f"[supervised] [{i}/{n_grouped}] label scan {ds}/{dep} | windows={len(g)} | RSS={_get_rss_gb():.2f} GB")
        pkl_path = _data_pkl_path(ctx, ds, dep)
        if not os.path.exists(pkl_path):
            continue
        try:
            data_pkl = _load_data_pkl(pkl_path)
            deployment_tz = _deployment_timezone_name(data_pkl)
            ev = getattr(data_pkl, "event_data", pd.DataFrame())
            loc_labels, loc_report, loc_quality_excluded = _derive_window_labels_from_events(
                _normalize_window_columns(g.copy(), deployment_tz),
                ev,
                label_keys,
                label_prefixes,
                label_group_map=label_group_map,
                drop_label_set=drop_label_set,
            )
            labels.loc[g.index] = loc_labels
            quality_excluded.loc[g.index] = loc_quality_excluded
            label_scan_rows.append({"dataset_id": ds, "deployment_id": dep, **loc_report})
        except Exception as exc:
            label_scan_rows.append(
                {
                    "dataset_id": ds,
                    "deployment_id": dep,
                    "eligible_event_rows": 0,
                    "quality_exclusion_event_rows": 0,
                    "windows_with_any_overlap": 0,
                    "windows_quality_excluded": 0,
                    "windows_with_multiple_labels": 0,
                    "windows_with_ties": 0,
                    "windows_labeled": 0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
    model_df = cdf.copy()
    model_df["observed_label"] = labels
    model_df["quality_excluded"] = quality_excluded.fillna(False).astype(bool)
    return model_df, pd.DataFrame(label_scan_rows)


def _apply_transfer_normalization(
    df: pd.DataFrame,
    feat_cols: List[str],
    mode: str,
) -> pd.DataFrame:
    if not feat_cols or df.empty:
        return df
    norm_mode = str(mode or "").strip().lower()
    if norm_mode in {"none", "", "off", "disabled"}:
        return df
    if norm_mode in {"per_deployment", "per_deployment_zscore"}:
        group_col = "deployment_id"
    elif norm_mode == "per_dataset_zscore":
        group_col = "dataset_id"
    else:
        # Backward-compatible fallback to existing behavior.
        group_col = "dataset_id"

    def _zscore_series(s: pd.Series) -> pd.Series:
        mean = s.mean()
        std = s.std(ddof=0)
        if not np.isfinite(float(std)) or float(std) == 0:
            return s - mean
        return (s - mean) / float(std)

    out = df.copy()
    for col in feat_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out.groupby(group_col)[col].transform(_zscore_series)
    return out


def _resolve_supervised_stratify_labels(
    labeled_df: pd.DataFrame,
    y: pd.Series,
    supervised_cfg: Dict[str, Any],
) -> Tuple[pd.Series, str, str | None]:
    if not bool(supervised_cfg.get("stratify_split_by_dataset_label", False)):
        return y, "label", None
    if "dataset_id" not in labeled_df.columns:
        return y, "label", "dataset_id_missing"

    dataset_labels = labeled_df["dataset_id"].astype(str).str.strip()
    if dataset_labels.eq("").any():
        return y, "label", "dataset_id_blank"

    composite = dataset_labels + "||" + y.astype(str)
    composite_counts = composite.value_counts(dropna=False)
    if composite_counts.empty:
        return y, "label", "composite_empty"
    if int(composite_counts.min()) < 2:
        return y, "label", "composite_class_too_small"

    return composite, "dataset_label", None


def _split_supervised_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    labeled_df: pd.DataFrame,
    supervised_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    test_size = float(supervised_cfg.get("test_size", 0.25))
    random_state = int(supervised_cfg.get("random_state", 42))
    stratify_labels, stratify_mode, fallback_reason = _resolve_supervised_stratify_labels(
        labeled_df=labeled_df,
        y=y,
        supervised_cfg=supervised_cfg,
    )

    split_kwargs = {
        "test_size": test_size,
        "random_state": random_state,
    }
    split_summary = {
        "test_size": test_size,
        "random_state": random_state,
        "stratify_mode_requested": "dataset_label" if bool(supervised_cfg.get("stratify_split_by_dataset_label", False)) else "label",
        "stratify_mode_used": stratify_mode,
        "stratify_fallback_reason": fallback_reason,
    }

    try:
        X_train, X_test, y_train, y_test, key_train, key_test = train_test_split(
            X,
            y,
            labeled_df["window_key"],
            stratify=stratify_labels,
            **split_kwargs,
        )
    except ValueError as exc:
        if stratify_mode != "dataset_label":
            raise
        split_summary["stratify_mode_used"] = "label"
        split_summary["stratify_fallback_reason"] = f"train_test_split_failed:{exc}"
        X_train, X_test, y_train, y_test, key_train, key_test = train_test_split(
            X,
            y,
            labeled_df["window_key"],
            stratify=y,
            **split_kwargs,
        )

    return X_train, X_test, y_train, y_test, key_train, key_test, split_summary


def _resolve_supervised_prediction_event_key(
    behavior_label: Any,
    supervised_cfg: Dict[str, Any] | None = None,
) -> str:
    behavior_text = str(behavior_label or "").strip()
    event_key_map = dict((supervised_cfg or {}).get("event_keys") or {})
    uncertain_tokens = {"", "unknown", "nan", "none"}
    if _normalize_label_name(behavior_text) in uncertain_tokens:
        for uncertain_key in ("__uncertain__", "Unknown", "unknown"):
            mapped = str(event_key_map.get(uncertain_key) or "").strip()
            if mapped:
                return mapped
        return "rf_unknown"

    candidates = [
        behavior_text,
        _canonicalize_supervised_event_key(behavior_text),
        _normalize_label_name(behavior_text),
    ]
    for candidate in candidates:
        mapped = str(event_key_map.get(candidate) or "").strip()
        if mapped:
            return mapped
    return "rf_" + _normalize_label_name(behavior_text)


def _build_rf_state_events(
    pred_df: pd.DataFrame,
    supervised_cfg: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame(columns=["datetime", "end_datetime", "duration_s", "key", "behavior_label", "pred_confidence", "source"])

    dep = pred_df.copy()
    dep = dep.dropna(subset=["window_start", "window_end"]).sort_values("window_start").reset_index(drop=True)
    dep["behavior_label"] = dep["final_behavior"].fillna("Unknown").astype(str)
    dep["event_key"] = dep["behavior_label"].map(
        lambda x: _resolve_supervised_prediction_event_key(x, supervised_cfg=supervised_cfg)
    )
    dep["prev_end"] = dep["window_end"].shift(1)
    dep["prev_key"] = dep["event_key"].shift(1)
    dep["new_block"] = (dep["event_key"] != dep["prev_key"]) | (dep["window_start"] != dep["prev_end"])
    dep["block_id"] = dep["new_block"].cumsum()

    rf_events = (
        dep.groupby(["block_id", "event_key", "behavior_label"], as_index=False)
        .agg(
            datetime=("window_start", "min"),
            end_datetime=("window_end", "max"),
            pred_confidence=("pred_confidence", "mean"),
        )
    )
    rf_events["duration_s"] = (rf_events["end_datetime"] - rf_events["datetime"]).dt.total_seconds()
    rf_events["source"] = "random_forest_behavior"
    rf_events["key"] = rf_events["event_key"]
    return rf_events[["datetime", "end_datetime", "duration_s", "key", "behavior_label", "pred_confidence", "source"]].copy()


def _resolve_calibration_cv(y: pd.Series, requested_cv: int) -> Tuple[int | None, Dict[str, int]]:
    class_counts = y.astype(str).value_counts().sort_index()
    counts_dict = {str(k): int(v) for k, v in class_counts.items()}
    if class_counts.empty:
        return None, counts_dict

    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        return None, counts_dict

    return max(2, min(int(requested_cv), min_class_count)), counts_dict


def _export_rf_events_to_deployments(
    ctx: RunContext,
    prediction_df: pd.DataFrame,
    supervised_cfg: Dict[str, Any] | None = None,
) -> Dict[str, int]:
    if prediction_df is None or prediction_df.empty:
        return {"deployments_written": 0, "rf_event_rows": 0, "deployments_skipped": 0}

    deployments_written = 0
    total_event_rows = 0
    deployments_skipped = 0
    for (ds, dep), g in prediction_df.groupby(["dataset_id", "deployment_id"], sort=False):
        rf_events = _build_rf_state_events(g, supervised_cfg=supervised_cfg)
        pkl_path = _data_pkl_path(ctx, ds, dep)
        if not os.path.exists(pkl_path):
            deployments_skipped += 1
            print(f"[export] skip {ds}/{dep}: missing data.pkl")
            continue
        try:
            data_pkl = _load_data_pkl(pkl_path)
            deployment_tz = _deployment_timezone_name(data_pkl)
            rf_events["datetime"] = _normalize_datetime_series_to_timezone(rf_events["datetime"], deployment_tz)
            rf_events["end_datetime"] = _normalize_datetime_series_to_timezone(rf_events["end_datetime"], deployment_tz)

            existing = getattr(data_pkl, "event_data", pd.DataFrame()).copy()
            if existing.empty:
                existing = pd.DataFrame(columns=["date", "time", "value", "type", "key", "duration", "short_description", "long_description", "datetime"])
            if "long_description" in existing.columns:
                auto_mask = existing["long_description"].astype(str).str.startswith("Supervised prediction state:")
                existing = existing[~auto_mask].copy()
            if "key" in existing.columns:
                # Keep backward-compatible cleanup for legacy rf_* exported keys.
                existing = existing[~existing["key"].astype(str).str.startswith("rf_")].copy()

            rf_export_df = existing
            for event_key, sub in rf_events.groupby("key", sort=True):
                sub = sub.copy().sort_values("datetime")
                behavior_label = sub["behavior_label"].dropna().iloc[0] if not sub["behavior_label"].dropna().empty else event_key
                meta = _resolve_segmentation_event_metadata(
                    ctx,
                    step_name="supervised_predictions",
                    event_key=str(event_key),
                    label_name=str(behavior_label),
                    variant_id="random_forest",
                    row_context={
                        "dataset_id": ds,
                        "deployment_id": dep,
                        "behavior_label": behavior_label,
                        "event_key": event_key,
                    },
                )
                short_desc = str(meta.get("short_description") or f"supervised_{_normalize_label_name(behavior_label)}_start")
                long_desc = str(meta.get("long_description") or f"Supervised prediction state: {behavior_label}")
                rf_export_df = create_state_event(
                    state_df=sub,
                    key=event_key,
                    value_column="pred_confidence",
                    start_time_column="datetime",
                    duration_column="duration_s",
                    description=short_desc,
                    long_description=long_desc,
                    existing_events=rf_export_df,
                )
                desired_type = str(meta.get("type") or "state").strip() or "state"
                if "type" in rf_export_df.columns:
                    rf_export_df.loc[rf_export_df["key"].astype(str) == str(event_key), "type"] = desired_type

            data_pkl.event_data = rf_export_df.sort_values("datetime").reset_index(drop=True)
            if hasattr(data_pkl, "_ensure_event_data_columns"):
                data_pkl._ensure_event_data_columns()
            if not hasattr(data_pkl, "event_manager") or data_pkl.event_manager is None:
                data_pkl.event_manager = {}
            data_pkl.event_manager["rf_behavior_states"] = {
                "keys": sorted(rf_events["key"].dropna().astype(str).unique().tolist()),
                "description": "Random forest behavior states from filtered feature workflow",
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(data_pkl, f)
            deployments_written += 1
            total_event_rows += len(rf_events)
        except Exception as exc:
            deployments_skipped += 1
            print(f"[export] skip {ds}/{dep}: {type(exc).__name__}: {exc}")
            continue

    return {
        "deployments_written": deployments_written,
        "rf_event_rows": total_event_rows,
        "deployments_skipped": deployments_skipped,
    }


def cmd_supervised_export(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    supervised_dir = os.path.join(ctx.output_root, "supervised")
    prediction_path = os.path.join(supervised_dir, "supervised_predictions.parquet")
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Missing supervised predictions: {prediction_path}")

    prediction_df = pd.read_parquet(prediction_path)
    print(
        f"[export] rerunning RF event export from existing predictions | "
        f"rows={len(prediction_df)} | RSS={_get_rss_gb():.2f} GB"
    )
    export_summary = _export_rf_events_to_deployments(ctx, prediction_df, supervised_cfg=supervised_cfg)
    summary_path = os.path.join(supervised_dir, "supervised_export_report.json")
    with open(summary_path, "w") as f:
        json.dump(export_summary, f, indent=2)
    print(f"[export] wrote {summary_path}")


def _variant_is_available_for_deployment(
    dep_df: pd.DataFrame,
    variant: Dict[str, Any],
) -> Tuple[bool, str | None]:
    if variant.get("variant_type") == "algorithmic":
        return True, None
    variant_cols = [c for c in list(variant.get("feature_cols") or []) if c in dep_df.columns]
    if not variant_cols:
        return False, "no_feature_columns"
    if not dep_df[variant_cols].replace([np.inf, -np.inf], np.nan).notna().any().any():
        return False, "all_features_missing"
    required_sources = list(variant.get("selected_sources") or [])
    if required_sources:
        for source_name in required_sources:
            source_cols = [c for c in variant_cols if _feature_source_channel(c) == source_name]
            if source_cols and not dep_df[source_cols].replace([np.inf, -np.inf], np.nan).notna().any().any():
                return False, f"missing_required_source:{source_name}"
    return True, None


def _run_supervised_transfer(
    ctx: RunContext,
    model_df: pd.DataFrame,
    feat_cols: List[str],
    label_order: List[str],
    positive_label: str | None,
    negative_label: str | None,
    supervised_cfg: Dict[str, Any],
    out_dir: str,
    write_csv_debug: bool,
    label_scan_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    transfer_cfg = _parse_supervised_transfer_cfg(ctx.run_cfg)
    prediction_scope = _supervised_prediction_scope(supervised_cfg)
    variant_defs, skipped_variants, _source_feature_map = _build_supervised_variant_defs(ctx, supervised_cfg, feat_cols)
    transfer_df = _apply_transfer_normalization(model_df, feat_cols, transfer_cfg["transfer_normalization"])
    transfer_df[feat_cols] = transfer_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    scoring_df = _filter_supervised_prediction_rows(transfer_df, supervised_cfg)

    source_mask = transfer_df["dataset_id"].astype(str).isin(set(transfer_cfg["source_dataset_ids"]))
    source_df = transfer_df[source_mask].copy()
    labeled_source_df = source_df.dropna(subset=["observed_label"]).copy()
    if labeled_source_df.empty:
        raise RuntimeError("Transfer mode found no labeled source windows after filtering to source_dataset_ids.")

    holdout_variant_frames = []
    class_metric_variant_frames = []
    confusion_variant_frames = []
    metrics_long_variant_frames = []
    fold_rows = []

    algorithmic_source_preds = None
    if any(v["variant_type"] == "algorithmic" for v in variant_defs):
        algorithmic_source_preds = _load_algorithmic_supervised_predictions(
            ctx, labeled_source_df, positive_label, negative_label
        )

    fold_specs: List[Dict[str, Any]] = []
    if transfer_cfg.get("source_split_mode") == "random_70_30":
        split_test_size = float(supervised_cfg.get("test_size", 0.30))
        split_random_state = int(supervised_cfg.get("random_state", 42))
        y_source = labeled_source_df["observed_label"].astype(str)
        source_idx = labeled_source_df.index.to_numpy()
        try:
            train_idx, holdout_idx = train_test_split(
                source_idx,
                test_size=split_test_size,
                random_state=split_random_state,
                stratify=y_source,
            )
        except ValueError:
            train_idx, holdout_idx = train_test_split(
                source_idx,
                test_size=split_test_size,
                random_state=split_random_state,
                stratify=None,
            )
        fold_specs.append(
            {
                "fold_id": "random_70_30",
                "train_df": labeled_source_df.loc[train_idx].copy(),
                "holdout_df": labeled_source_df.loc[holdout_idx].copy(),
                "training_deployment_ids": sorted(
                    labeled_source_df.loc[train_idx, "deployment_id"].astype(str).unique().tolist()
                ),
                "heldout_deployment_ids": sorted(
                    labeled_source_df.loc[holdout_idx, "deployment_id"].astype(str).unique().tolist()
                ),
            }
        )
    else:
        for fold in transfer_cfg["source_holdout_folds"]:
            fold_id = str(fold["fold_id"])
            train_ids = set(transfer_cfg["source_training_deployment_ids_always"]) | set(
                fold["additional_train_deployment_ids"]
            )
            holdout_ids = set(fold["holdout_deployment_ids"])
            train_df = labeled_source_df[
                labeled_source_df["deployment_id"].astype(str).isin(train_ids)
            ].copy()
            holdout_df = labeled_source_df[
                labeled_source_df["deployment_id"].astype(str).isin(holdout_ids)
            ].copy()
            fold_specs.append(
                {
                    "fold_id": fold_id,
                    "train_df": train_df,
                    "holdout_df": holdout_df,
                    "training_deployment_ids": sorted(train_ids),
                    "heldout_deployment_ids": sorted(holdout_ids),
                }
            )

    for fold in fold_specs:
        fold_id = str(fold["fold_id"])
        train_df = fold["train_df"].copy()
        holdout_df = fold["holdout_df"].copy()
        train_ids = set(fold.get("training_deployment_ids") or [])
        holdout_ids = set(fold.get("heldout_deployment_ids") or [])
        if train_df.empty or holdout_df.empty:
            raise RuntimeError(f"Transfer fold {fold_id} has empty train or holdout source rows.")
        X_train = train_df[feat_cols].fillna(0.0)
        y_train = train_df["observed_label"].astype(str)
        X_holdout = holdout_df[feat_cols].fillna(0.0)
        y_holdout = holdout_df["observed_label"].astype(str)
        holdout_keys = holdout_df["window_key"].astype(str)

        for variant in variant_defs:
            variant_id = str(variant["variant_id"])
            variant_label = str(variant["variant_label"])
            variant_type = str(variant["variant_type"])
            if variant_type == "algorithmic":
                lookup = pd.Series(algorithmic_source_preds.values, index=labeled_source_df.index, dtype=object)
                y_pred_variant = lookup.loc[holdout_df.index]
                y_conf_variant = pd.Series(1.0, index=holdout_df.index, dtype=float)
            else:
                variant_cols = [c for c in list(variant["feature_cols"]) if c in feat_cols]
                estimator = _build_supervised_estimator(supervised_cfg, variant_type)
                estimator.fit(X_train[variant_cols], y_train)
                y_pred_variant = pd.Series(estimator.predict(X_holdout[variant_cols]), index=holdout_df.index, dtype=object)
                y_conf_variant = pd.Series(estimator.predict_proba(X_holdout[variant_cols]).max(axis=1), index=holdout_df.index, dtype=float)

            holdout_variant = pd.DataFrame(
                {
                    "window_key": holdout_keys.values,
                    "dataset_id": holdout_df["dataset_id"].values,
                    "deployment_id": holdout_df["deployment_id"].values,
                    "fold_id": fold_id,
                    "source_or_target": "source",
                    "eval_role": "source_holdout",
                    "training_deployment_ids": [sorted(train_ids)] * len(holdout_df),
                    "heldout_deployment_ids": [sorted(holdout_ids)] * len(holdout_df),
                    "y_true": y_holdout.values,
                    "y_pred": y_pred_variant.values,
                    "pred_confidence": y_conf_variant.values,
                    "correct": (y_pred_variant.values == y_holdout.values),
                    "variant_id": variant_id,
                    "variant_label": variant_label,
                    "variant_type": variant_type,
                    "variant_rank": int(variant["variant_rank"]),
                }
            )
            holdout_variant_frames.append(holdout_variant)
            y_true_eval = pd.Series(y_holdout.values, dtype=object)
            y_pred_eval = pd.Series(y_pred_variant.values, dtype=object)
            if variant_type == "algorithmic":
                valid_eval = y_pred_eval.notna()
                y_true_eval = y_true_eval.loc[valid_eval].reset_index(drop=True)
                y_pred_eval = y_pred_eval.loc[valid_eval].astype(str).reset_index(drop=True)
                if y_true_eval.empty:
                    print(
                        f"[supervised-transfer] skipping fold metrics for {variant_label}: "
                        "no scorable windows after excluding algorithmic unscorable predictions"
                    )
                    continue
            class_metrics_variant, confusion_variant, metrics_long_variant = _evaluate_supervised_variant(
                y_true=y_true_eval,
                y_pred=y_pred_eval,
                label_order=label_order,
                variant_id=variant_id,
                variant_label=variant_label,
                variant_type=variant_type,
            )
            class_metrics_variant["variant_rank"] = int(variant["variant_rank"])
            class_metrics_variant["fold_id"] = fold_id
            confusion_variant["variant_rank"] = int(variant["variant_rank"])
            confusion_variant["fold_id"] = fold_id
            metrics_long_variant["variant_rank"] = int(variant["variant_rank"])
            metrics_long_variant["fold_id"] = fold_id
            class_metric_variant_frames.append(class_metrics_variant)
            confusion_variant_frames.append(confusion_variant)
            metrics_long_variant_frames.append(metrics_long_variant)

        fold_rows.append(
            {
                "fold_id": fold_id,
                "training_deployment_ids": sorted(train_ids),
                "heldout_deployment_ids": sorted(holdout_ids),
                "train_rows": int(len(train_df)),
                "holdout_rows": int(len(holdout_df)),
            }
        )

    holdout_variant_df = pd.concat(holdout_variant_frames, ignore_index=True) if holdout_variant_frames else pd.DataFrame()
    holdout_variant_df.to_parquet(os.path.join(out_dir, "holdout_predictions_by_variant.parquet"), index=False)
    if write_csv_debug and not holdout_variant_df.empty:
        holdout_variant_df.to_csv(os.path.join(out_dir, "holdout_predictions_by_variant.csv"), index=False)

    baseline_holdout_eval = holdout_variant_df[holdout_variant_df["variant_id"] == "rf_full"].copy()
    if baseline_holdout_eval.empty:
        raise RuntimeError("Transfer mode failed to produce baseline rf_full holdout evaluation.")
    holdout_eval = baseline_holdout_eval[
        ["window_key", "dataset_id", "deployment_id", "fold_id", "y_true", "y_pred", "pred_confidence", "correct"]
    ].copy()
    holdout_eval.to_parquet(os.path.join(out_dir, "holdout_predictions.parquet"), index=False)
    if write_csv_debug:
        holdout_eval.to_csv(os.path.join(out_dir, "holdout_predictions.csv"), index=False)

    class_metrics_all_variants = pd.concat(class_metric_variant_frames, ignore_index=True) if class_metric_variant_frames else pd.DataFrame()
    confusion_all_variants = pd.concat(confusion_variant_frames, ignore_index=True) if confusion_variant_frames else pd.DataFrame()
    metrics_long_all_variants = pd.concat(metrics_long_variant_frames, ignore_index=True) if metrics_long_variant_frames else pd.DataFrame()
    if not class_metrics_all_variants.empty:
        class_metrics_all_variants.to_parquet(os.path.join(out_dir, "classification_report_variants.parquet"), index=False)
    if not confusion_all_variants.empty:
        confusion_all_variants.to_parquet(os.path.join(out_dir, "confusion_matrix_variants.parquet"), index=False)
    if not metrics_long_all_variants.empty:
        metrics_long_all_variants.to_parquet(os.path.join(out_dir, "metrics_by_variant.parquet"), index=False)

    source_all_labeled = labeled_source_df.copy()
    model_variant_frames = []
    calibration_requested = bool(supervised_cfg.get("use_probability_calibration", True))
    calibration_requested_cv = int(supervised_cfg.get("calibration_cv", 3))
    class_counts = {str(k): int(v) for k, v in source_all_labeled["observed_label"].astype(str).value_counts().sort_index().items()}

    final_predictions = []
    for variant in variant_defs:
        variant_id = str(variant["variant_id"])
        variant_label = str(variant["variant_label"])
        variant_type = str(variant["variant_type"])
        if variant_type == "algorithmic":
            algo_preds = _load_algorithmic_supervised_predictions(ctx, scoring_df, positive_label, negative_label).fillna(negative_label or label_order[0]).astype(str)
            variant_prediction = scoring_df[
                ["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key", "observed_label"]
            ].copy()
            variant_prediction["predicted_label_raw"] = algo_preds.values
            variant_prediction["pred_confidence"] = 1.0
            variant_prediction["predicted_label"] = variant_prediction["predicted_label_raw"]
            variant_prediction["final_behavior"] = np.where(
                variant_prediction["observed_label"].notna() & (variant_prediction["observed_label"].astype(str) != "Unknown"),
                variant_prediction["observed_label"],
                variant_prediction["predicted_label"],
            )
            variant_prediction["source_or_target"] = np.where(
                variant_prediction["dataset_id"].astype(str).isin(set(transfer_cfg["source_dataset_ids"])),
                "source",
                "target",
            )
            variant_prediction["eval_role"] = np.where(
                variant_prediction["source_or_target"] == "source",
                "source_inference",
                "target_inference",
            )
            variant_prediction["fold_id"] = "final_refit"
            variant_prediction["training_deployment_ids"] = [sorted(source_all_labeled["deployment_id"].astype(str).unique().tolist())] * len(variant_prediction)
            variant_prediction["heldout_deployment_ids"] = [[]] * len(variant_prediction)
            variant_prediction["variant_applied_to_deployment"] = True
            variant_prediction["variant_skip_reason"] = pd.NA
            variant_prediction["variant_id"] = variant_id
            variant_prediction["variant_label"] = variant_label
            variant_prediction["variant_type"] = variant_type
            variant_prediction["variant_rank"] = int(variant["variant_rank"])
            model_variant_frames.append(variant_prediction)
            continue

        variant_cols = [c for c in list(variant["feature_cols"]) if c in feat_cols]
        X_source = source_all_labeled[variant_cols].fillna(0.0)
        y_source = source_all_labeled["observed_label"].astype(str)
        estimator = _build_supervised_estimator(supervised_cfg, variant_type)
        estimator.fit(X_source, y_source)
        variant_prediction = scoring_df[
            ["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key", "observed_label"]
        ].copy()
        raw_pred = pd.Series("Unknown", index=variant_prediction.index, dtype=object)
        raw_conf = pd.Series(np.nan, index=variant_prediction.index, dtype=float)
        applied_mask = pd.Series(False, index=variant_prediction.index, dtype=bool)
        skip_reason = pd.Series(pd.NA, index=variant_prediction.index, dtype=object)

        for (dataset_id, deployment_id), dep_idx in variant_prediction.groupby(["dataset_id", "deployment_id"], sort=False).groups.items():
            dep_rows = scoring_df.loc[list(dep_idx)]
            available, reason = _variant_is_available_for_deployment(dep_rows, variant)
            if not available:
                skip_reason.loc[list(dep_idx)] = reason
                continue
            dep_X = dep_rows[variant_cols].fillna(0.0)
            raw_pred.loc[list(dep_idx)] = estimator.predict(dep_X)
            raw_conf.loc[list(dep_idx)] = estimator.predict_proba(dep_X).max(axis=1)
            applied_mask.loc[list(dep_idx)] = True

        confidence_threshold = float(supervised_cfg.get("confidence_threshold", 0.40))
        variant_prediction["predicted_label_raw"] = raw_pred.values
        variant_prediction["pred_confidence"] = raw_conf.values
        variant_prediction["predicted_label"] = np.where(
            pd.isna(variant_prediction["pred_confidence"]) | (variant_prediction["pred_confidence"] < confidence_threshold),
            "Unknown",
            variant_prediction["predicted_label_raw"],
        )
        variant_prediction["final_behavior"] = np.where(
            variant_prediction["observed_label"].notna() & (variant_prediction["observed_label"].astype(str) != "Unknown"),
            variant_prediction["observed_label"],
            variant_prediction["predicted_label"],
        )
        variant_prediction["source_or_target"] = np.where(
            variant_prediction["dataset_id"].astype(str).isin(set(transfer_cfg["source_dataset_ids"])),
            "source",
            "target",
        )
        variant_prediction["eval_role"] = np.where(
            variant_prediction["source_or_target"] == "source",
            "source_inference",
            "target_inference",
        )
        variant_prediction["fold_id"] = "final_refit"
        variant_prediction["training_deployment_ids"] = [sorted(source_all_labeled["deployment_id"].astype(str).unique().tolist())] * len(variant_prediction)
        variant_prediction["heldout_deployment_ids"] = [[]] * len(variant_prediction)
        variant_prediction["variant_applied_to_deployment"] = applied_mask.values
        variant_prediction["variant_skip_reason"] = skip_reason.values
        variant_prediction["variant_id"] = variant_id
        variant_prediction["variant_label"] = variant_label
        variant_prediction["variant_type"] = variant_type
        variant_prediction["variant_rank"] = int(variant["variant_rank"])
        model_variant_frames.append(variant_prediction.copy())

        if "algorithmic_gate" in set(transfer_cfg["transfer_fusion_modes"]):
            gated_prediction, _gating_summary = _apply_supervised_context_filters(
                ctx,
                variant_prediction.copy(),
                positive_label,
                negative_label,
            )
            gated_prediction["variant_id"] = f"{variant_id}_gated"
            gated_prediction["variant_label"] = f"{variant_label} + algorithmic gate"
            gated_prediction["variant_type"] = f"{variant_type}_gated"
            gated_prediction["variant_rank"] = int(variant["variant_rank"]) + 100
            model_variant_frames.append(gated_prediction)

    variant_prediction_df = pd.concat(model_variant_frames, ignore_index=True) if model_variant_frames else pd.DataFrame()
    if not variant_prediction_df.empty:
        variant_prediction_df.to_parquet(os.path.join(out_dir, "supervised_variant_predictions.parquet"), index=False)
        if write_csv_debug:
            variant_prediction_df.to_csv(os.path.join(out_dir, "supervised_variant_predictions.csv"), index=False)

    final_prediction_df = variant_prediction_df[
        variant_prediction_df["variant_id"].astype(str).eq("rf_full")
    ].copy()
    if "algorithmic_gate" in set(transfer_cfg["transfer_fusion_modes"]):
        gated_primary = variant_prediction_df[variant_prediction_df["variant_id"].astype(str).eq("rf_full_gated")].copy()
        if not gated_primary.empty:
            final_prediction_df = gated_primary
    final_prediction_df.to_parquet(os.path.join(out_dir, "supervised_predictions.parquet"), index=False)
    if write_csv_debug:
        final_prediction_df.to_csv(os.path.join(out_dir, "supervised_predictions.csv"), index=False)

    export_summary = {"deployments_written": 0, "rf_event_rows": 0}
    if bool(supervised_cfg.get("export_rf_events", False)):
        export_summary = _export_rf_events_to_deployments(ctx, final_prediction_df, supervised_cfg=supervised_cfg)

    rep = classification_report(
        baseline_holdout_eval["y_true"],
        baseline_holdout_eval["y_pred"],
        output_dict=True,
        zero_division=0,
    )
    threshold_rows = []
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        keep = baseline_holdout_eval["pred_confidence"] >= threshold
        n_keep = int(keep.sum())
        threshold_rows.append(
            {
                "threshold": threshold,
                "n_pred": n_keep,
                "coverage_pct": (100.0 * n_keep / len(baseline_holdout_eval)) if len(baseline_holdout_eval) else 0.0,
                "accuracy_on_kept": float(baseline_holdout_eval.loc[keep, "correct"].mean()) if n_keep else np.nan,
            }
        )
    pd.DataFrame(threshold_rows).to_parquet(os.path.join(out_dir, "holdout_threshold_diagnostics.parquet"), index=False)

    with open(summary_path := os.path.join(out_dir, "supervised_report.json"), "w") as f:
        json.dump(
            {
                "enabled": True,
                "status": "ok",
                "mode": "transfer",
                "source_dataset_ids": transfer_cfg["source_dataset_ids"],
                "source_split_mode": transfer_cfg.get("source_split_mode", "deployment_holdout_folds"),
                "source_training_deployment_ids_always": transfer_cfg["source_training_deployment_ids_always"],
                "source_holdout_folds": fold_rows,
                "transfer_target_scope": transfer_cfg["transfer_target_scope"],
                "transfer_normalization": transfer_cfg["transfer_normalization"],
                "transfer_fusion_modes": transfer_cfg["transfer_fusion_modes"],
                "prediction_scope": prediction_scope,
                "n_rows_labeled": int(len(labeled_source_df)),
                "n_rows_total": int(len(transfer_df)),
                "n_rows_scored": int(len(scoring_df)),
                "n_classes": int(labeled_source_df["observed_label"].nunique()),
                "class_counts": class_counts,
                "variant_definitions": variant_defs,
                "skipped_variants": skipped_variants,
                "holdout_classification_report": rep,
                "label_scan_summary": {
                    "eligible_event_rows": int(sum(int(row.get("eligible_event_rows", 0) or 0) for row in label_scan_rows)),
                    "quality_exclusion_event_rows": int(sum(int(row.get("quality_exclusion_event_rows", 0) or 0) for row in label_scan_rows)),
                    "windows_with_any_overlap": int(sum(int(row.get("windows_with_any_overlap", 0) or 0) for row in label_scan_rows)),
                    "windows_quality_excluded": int(sum(int(row.get("windows_quality_excluded", 0) or 0) for row in label_scan_rows)),
                    "windows_with_multiple_labels": int(sum(int(row.get("windows_with_multiple_labels", 0) or 0) for row in label_scan_rows)),
                    "windows_with_ties": int(sum(int(row.get("windows_with_ties", 0) or 0) for row in label_scan_rows)),
                    "windows_labeled": int(sum(int(row.get("windows_labeled", 0) or 0) for row in label_scan_rows)),
                },
                "rf_export": export_summary,
            },
            f,
            indent=2,
            default=str,
        )
    return {
        "summary_path": summary_path,
        "variant_defs": variant_defs,
        "skipped_variants": skipped_variants,
        "class_counts": class_counts,
        "export_summary": export_summary,
        "holdout_report": rep,
    }


def cmd_supervised(args):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for supervised step")
    ctx = _resolve_run_context(args.config, args.run_name)
    enable = bool(ctx.run_cfg.get("enable_supervised", False))
    out_dir = os.path.join(ctx.output_root, "supervised")
    _ensure_dir(out_dir)
    summary_path = os.path.join(out_dir, "supervised_report.json")

    if not enable:
        with open(summary_path, "w") as f:
            json.dump({"enabled": False, "status": "skipped"}, f, indent=2)
        print("[supervised] skipped (enable_supervised=false)")
        return

    filt_path = os.path.join(ctx.output_root, "features", "features_filtered.parquet")
    cdf = pd.read_parquet(filt_path)
    cdf = cdf.copy().reset_index(drop=True)
    cdf["window_key"] = _build_window_key_series(cdf)
    feat_cols = [
        c for c in cdf.columns
        if c not in {"dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key"}
        and pd.api.types.is_numeric_dtype(cdf[c])
        and not str(c).startswith("__")
    ]
    if not feat_cols:
        raise RuntimeError("No filtered numeric feature columns available for supervised learning.")

    supervised_cfg = ctx.run_cfg.get("supervised") or {}
    write_csv_debug = _write_csv_debug_outputs(ctx.run_cfg)
    label_keys = list(supervised_cfg.get("label_event_keys") or [])
    label_prefixes = _normalized_label_prefixes(ctx.run_cfg)
    label_group_map = _build_supervised_label_group_map(supervised_cfg)
    drop_label_set = _build_supervised_drop_label_set(supervised_cfg)
    memory_log_interval_s = float(ctx.run_cfg.get("memory_log_interval_s", 30))
    last_log = time.monotonic()

    print(
        f"[supervised] label targeting | prefixes={label_prefixes} "
        f"keys={label_keys} grouped_labels={sorted(set(label_group_map.values()))} "
        f"drop_labels(quality_exclusion_only)={sorted(drop_label_set)}"
    )
    model_df, label_scan_report = _collect_supervised_labels(
        ctx,
        cdf,
        label_keys,
        label_prefixes,
        label_group_map=label_group_map,
        drop_label_set=drop_label_set,
    )
    label_scan_rows = label_scan_report.to_dict("records")
    label_scan_report.to_parquet(os.path.join(out_dir, "label_scan_report.parquet"), index=False)

    transfer_cfg = _parse_supervised_transfer_cfg(ctx.run_cfg)
    labeled_keep_mask = _observed_label_keep_mask(model_df.get("observed_label"))
    if transfer_cfg.get("enabled", False):
        source_mask = model_df["dataset_id"].astype(str).isin(set(transfer_cfg["source_dataset_ids"]))
        labeled_df = model_df.loc[source_mask & labeled_keep_mask].copy()
    else:
        labeled_df = model_df.loc[labeled_keep_mask].copy()

    if labeled_df.empty:
        if transfer_cfg.get("enabled", False):
            raise RuntimeError(
                f"No eligible source supervised labels found for transfer run '{ctx.run_name}'. "
                f"Checked source_dataset_ids={transfer_cfg['source_dataset_ids']} using keys/prefixes {label_keys or label_prefixes}."
            )
        raise RuntimeError(
            f"No eligible supervised labels found for run '{ctx.run_name}'. "
            f"Expected state-event keys matching one of: {label_keys or label_prefixes}"
        )
    if labeled_df["observed_label"].nunique() < 2 or len(labeled_df) < 20:
        raise RuntimeError(
            f"Insufficient labeled windows for supervised learning: rows={len(labeled_df)} "
            f"classes={labeled_df['observed_label'].nunique()}"
        )

    print(
        f"[supervised] label scan complete | labeled_rows={len(labeled_df)} "
        f"total_rows={len(model_df)} classes={labeled_df['observed_label'].nunique()} "
        f"quality_excluded_rows={int(model_df.get('quality_excluded', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())} "
        f"| RSS={_get_rss_gb():.2f} GB"
    )

    X = labeled_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = labeled_df["observed_label"].astype(str)
    label_order = sorted(y.unique())
    positive_label = _detect_supervised_positive_label(label_order)
    negative_label = _detect_supervised_negative_label(label_order, positive_label)
    prediction_scope = _supervised_prediction_scope(supervised_cfg)
    print(
        f"[supervised] feature matrix prepared | X_shape={X.shape} "
        f"feature_cols={len(feat_cols)} | RSS={_get_rss_gb():.2f} GB"
    )

    if transfer_cfg.get("enabled", False):
        result = _run_supervised_transfer(
            ctx=ctx,
            model_df=model_df,
            feat_cols=feat_cols,
            label_order=label_order,
            positive_label=positive_label,
            negative_label=negative_label,
            supervised_cfg=supervised_cfg,
            out_dir=out_dir,
            write_csv_debug=write_csv_debug,
            label_scan_rows=label_scan_rows,
        )
        print(f"[supervised] wrote {result['summary_path']}")
        return

    X_train, X_test, y_train, y_test, key_train, key_test, split_summary = _split_supervised_train_test(
        X=X,
        y=y,
        labeled_df=labeled_df,
        supervised_cfg=supervised_cfg,
    )
    print(
        f"[supervised] train/test split complete | train_rows={len(X_train)} "
        f"test_rows={len(X_test)} stratify_mode={split_summary['stratify_mode_used']} "
        f"fallback_reason={split_summary['stratify_fallback_reason'] or 'none'} "
        f"| RSS={_get_rss_gb():.2f} GB"
    )

    variant_defs, skipped_variants, source_feature_map = _build_supervised_variant_defs(ctx, supervised_cfg, feat_cols)
    if bool(supervised_cfg.get("enable_lightgbm", False)):
        if LIGHTGBM_AVAILABLE:
            variant_defs.append(
                {
                    "variant_id": "lightgbm_full",
                    "variant_label": "LightGBM",
                    "variant_type": "lightgbm",
                    "feature_cols": list(feat_cols),
                    "variant_rank": len(variant_defs),
                }
            )
        else:
            skipped_variants.append({"variant_id": "lightgbm_full", "reason": "lightgbm_not_installed"})
            print("[supervised] skipping LightGBM comparator: lightgbm is not installed in the active environment")
    if bool(supervised_cfg.get("enable_svm", False)):
        variant_defs.append(
            {
                "variant_id": "svm_full",
                "variant_label": "SVM",
                "variant_type": "svm",
                "feature_cols": list(feat_cols),
                "variant_rank": len(variant_defs),
            }
        )
    if bool(supervised_cfg.get("enable_knn", False)):
        variant_defs.append(
            {
                "variant_id": "knn_full",
                "variant_label": "KNN",
                "variant_type": "knn",
                "feature_cols": list(feat_cols),
                "variant_rank": len(variant_defs),
            }
        )

    variant_desc = ", ".join([f"{v['variant_label']}[{v['variant_type']}]" for v in variant_defs])
    print(f"[supervised] evaluating variants | {variant_desc}")

    holdout_variant_frames = []
    class_metric_variant_frames = []
    confusion_variant_frames = []
    metrics_long_variant_frames = []
    baseline_holdout_model = None
    baseline_holdout_eval = None
    baseline_y_pred = None
    baseline_y_conf = None

    algorithmic_labeled_preds = None
    if any(v["variant_type"] == "algorithmic" for v in variant_defs):
        algorithmic_labeled_preds = _load_algorithmic_supervised_predictions(ctx, labeled_df, positive_label, negative_label)
        if algorithmic_labeled_preds.isna().all():
            print("[supervised] warning: algorithmic baseline produced no aligned labels; skipping algorithmic comparator")
            variant_defs = [v for v in variant_defs if v["variant_type"] != "algorithmic"]

    for variant in variant_defs:
        variant_id = variant["variant_id"]
        variant_label = variant["variant_label"]
        variant_type = variant["variant_type"]
        print(f"[supervised] holdout evaluation | variant={variant_label} | RSS={_get_rss_gb():.2f} GB")

        if variant_type != "algorithmic":
            variant_cols = variant["feature_cols"]
            estimator = _build_supervised_estimator(supervised_cfg, variant_type)
            estimator.fit(X_train[variant_cols], y_train)
            y_pred_variant = pd.Series(estimator.predict(X_test[variant_cols]), index=X_test.index, dtype=object)
            y_conf_variant = pd.Series(estimator.predict_proba(X_test[variant_cols]).max(axis=1), index=X_test.index, dtype=float)
            if variant_id == "rf_full":
                baseline_holdout_model = estimator
        else:
            variant_lookup = pd.Series(algorithmic_labeled_preds.values, index=labeled_df.index, dtype=object)
            y_pred_variant = variant_lookup.loc[X_test.index]
            y_conf_variant = pd.Series(1.0, index=X_test.index, dtype=float)

        holdout_variant = pd.DataFrame(
            {
                "window_key": key_test.values,
                "y_true": y_test.values,
                "y_pred": y_pred_variant.values,
                "pred_confidence": y_conf_variant.values,
                "correct": (y_pred_variant.values == y_test.values),
                "variant_id": variant_id,
                "variant_label": variant_label,
                "variant_type": variant_type,
                "variant_rank": int(variant["variant_rank"]),
            }
        )
        holdout_variant_frames.append(holdout_variant)

        y_true_eval = pd.Series(y_test.values, dtype=object)
        y_pred_eval = pd.Series(y_pred_variant.values, dtype=object)
        if variant_type == "algorithmic":
            valid_eval = y_pred_eval.notna()
            y_true_eval = y_true_eval.loc[valid_eval].reset_index(drop=True)
            y_pred_eval = y_pred_eval.loc[valid_eval].astype(str).reset_index(drop=True)
            if y_true_eval.empty:
                print(
                    f"[supervised] skipping holdout metrics for {variant_label}: "
                    "no scorable windows after excluding algorithmic unscorable predictions"
                )
                continue
        class_metrics_variant, confusion_variant, metrics_long_variant = _evaluate_supervised_variant(
            y_true=y_true_eval,
            y_pred=y_pred_eval,
            label_order=label_order,
            variant_id=variant_id,
            variant_label=variant_label,
            variant_type=variant_type,
        )
        class_metrics_variant["variant_rank"] = int(variant["variant_rank"])
        confusion_variant["variant_rank"] = int(variant["variant_rank"])
        metrics_long_variant["variant_rank"] = int(variant["variant_rank"])
        class_metric_variant_frames.append(class_metrics_variant)
        confusion_variant_frames.append(confusion_variant)
        metrics_long_variant_frames.append(metrics_long_variant)

        if variant_id == "rf_full":
            baseline_holdout_eval = holdout_variant.copy()
            baseline_y_pred = y_pred_variant
            baseline_y_conf = y_conf_variant

    if baseline_holdout_eval is None or baseline_holdout_model is None:
        raise RuntimeError("Baseline random forest evaluation failed to produce holdout outputs.")

    holdout_eval = baseline_holdout_eval[["window_key", "y_true", "y_pred", "pred_confidence", "correct"]].copy()
    holdout_eval.to_parquet(os.path.join(out_dir, "holdout_predictions.parquet"), index=False)
    if write_csv_debug:
        holdout_eval.to_csv(os.path.join(out_dir, "holdout_predictions.csv"), index=False)

    holdout_variant_df = pd.concat(holdout_variant_frames, ignore_index=True) if holdout_variant_frames else pd.DataFrame()
    holdout_variant_df.to_parquet(os.path.join(out_dir, "holdout_predictions_by_variant.parquet"), index=False)
    if write_csv_debug and not holdout_variant_df.empty:
        holdout_variant_df.to_csv(os.path.join(out_dir, "holdout_predictions_by_variant.csv"), index=False)

    class_metrics_all_variants = pd.concat(class_metric_variant_frames, ignore_index=True) if class_metric_variant_frames else pd.DataFrame()
    confusion_all_variants = pd.concat(confusion_variant_frames, ignore_index=True) if confusion_variant_frames else pd.DataFrame()
    metrics_long_all_variants = pd.concat(metrics_long_variant_frames, ignore_index=True) if metrics_long_variant_frames else pd.DataFrame()

    rep = classification_report(y_test, baseline_y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, baseline_y_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
    cm_df.to_parquet(os.path.join(out_dir, "confusion_matrix_random_forest.parquet"))
    if write_csv_debug:
        cm_df.to_csv(os.path.join(out_dir, "confusion_matrix_random_forest.csv"))

    class_metrics_df = pd.DataFrame(rep).T.rename_axis("label").reset_index()
    class_metrics_df.to_parquet(os.path.join(out_dir, "classification_report_random_forest.parquet"), index=False)
    if write_csv_debug:
        class_metrics_df.to_csv(os.path.join(out_dir, "classification_report_random_forest.csv"), index=False)

    class_rows = class_metrics_df[class_metrics_df["label"].isin(label_order)].copy()
    if not class_rows.empty:
        metrics_long = class_rows.melt(
            id_vars=["label", "support"],
            value_vars=[c for c in ["precision", "recall", "f1-score"] if c in class_rows.columns],
            var_name="metric",
            value_name="value",
        )
        metrics_fig = px.bar(
            metrics_long,
            x="label",
            y="value",
            color="metric",
            barmode="group",
            text="value",
            title="Per-Class Precision / Recall / F1",
            labels={"label": "State", "value": "Score"},
            color_discrete_map={
                "precision": "#1f77b4",
                "recall": "#2ca02c",
                "f1-score": "#ff7f0e",
            },
        )
        metrics_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        metrics_fig.update_layout(yaxis_range=[0, 1.05], xaxis_tickangle=-25)
        metrics_fig.write_html(os.path.join(out_dir, "classification_report_random_forest.html"))

    cm_fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="YlGnBu",
        aspect="auto",
        labels={"x": "Predicted", "y": "Observed", "color": "Count"},
        title="Holdout Confusion Matrix (Random Forest)",
    )
    cm_fig.update_xaxes(side="bottom")
    cm_fig.write_html(os.path.join(out_dir, "confusion_matrix_random_forest.html"))

    if not class_metrics_all_variants.empty:
        class_metrics_all_variants.to_parquet(os.path.join(out_dir, "classification_report_variants.parquet"), index=False)
        if write_csv_debug:
            class_metrics_all_variants.to_csv(os.path.join(out_dir, "classification_report_variants.csv"), index=False)
    if not confusion_all_variants.empty:
        confusion_all_variants.to_parquet(os.path.join(out_dir, "confusion_matrix_variants.parquet"), index=False)
        if write_csv_debug:
            confusion_all_variants.to_csv(os.path.join(out_dir, "confusion_matrix_variants.csv"), index=False)
    if not metrics_long_all_variants.empty:
        metrics_long_all_variants.to_parquet(os.path.join(out_dir, "metrics_by_variant.parquet"), index=False)
        if write_csv_debug:
            metrics_long_all_variants.to_csv(os.path.join(out_dir, "metrics_by_variant.csv"), index=False)

        metrics_plot_df = metrics_long_all_variants.copy()
        metrics_plot_df["label"] = pd.Categorical(metrics_plot_df["label"], categories=label_order + ["Overall"], ordered=True)
        metrics_plot_df["metric"] = pd.Categorical(
            metrics_plot_df["metric"],
            categories=["precision", "recall", "f1-score", "accuracy", "balanced_accuracy"],
            ordered=True,
        )
        variant_metrics_fig = px.bar(
            metrics_plot_df,
            x="variant_label",
            y="value",
            color="metric",
            barmode="group",
            facet_col="label",
            facet_col_wrap=min(3, max(1, len(label_order) + 1)),
            text="value",
            category_orders={"label": label_order + ["Overall"]},
            color_discrete_map={
                "precision": "#4C78A8",
                "recall": "#59A14F",
                "f1-score": "#F28E2B",
                "accuracy": "#B07AA1",
                "balanced_accuracy": "#E15759",
            },
            title="Variant Comparison: Precision, Recall, F1, Accuracy, Balanced Accuracy",
            labels={"variant_label": "Model Variant", "value": "Score"},
        )
        variant_metrics_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        variant_metrics_fig.update_layout(yaxis_range=[0, 1.05], xaxis_tickangle=-25, template="plotly_white")
        variant_metrics_fig.write_html(os.path.join(out_dir, "metrics_by_variant.html"))

    if not confusion_all_variants.empty:
        from plotly.subplots import make_subplots

        variant_meta = (
            confusion_all_variants[["variant_id", "variant_label", "variant_rank"]]
            .drop_duplicates()
            .sort_values(["variant_rank", "variant_label"])
        )
        n_variants = len(variant_meta)
        n_cols = min(3, max(1, n_variants))
        n_rows = int(np.ceil(n_variants / n_cols))
        subplot_titles = variant_meta["variant_label"].tolist()
        fig_cm_variants = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.14,
        )
        for i, row in enumerate(variant_meta.itertuples(index=False), start=1):
            sub = confusion_all_variants[confusion_all_variants["variant_id"] == row.variant_id].copy()
            mat = (
                sub.pivot_table(index="observed_label", columns="predicted_label", values="count", aggfunc="sum", fill_value=0)
                .reindex(index=label_order, columns=label_order, fill_value=0)
            )
            r = int(np.ceil(i / n_cols))
            c = ((i - 1) % n_cols) + 1
            fig_cm_variants.add_trace(
                go.Heatmap(
                    z=mat.to_numpy(),
                    x=label_order,
                    y=label_order,
                    colorscale="YlGnBu",
                    showscale=(i == 1),
                    colorbar_title="Count" if i == 1 else None,
                    zmin=0,
                ),
                row=r,
                col=c,
            )
            for yi, observed in enumerate(label_order):
                for xi, predicted in enumerate(label_order):
                    fig_cm_variants.add_annotation(
                        x=predicted,
                        y=observed,
                        text=str(int(mat.loc[observed, predicted])),
                        showarrow=False,
                        row=r,
                        col=c,
                        font=dict(size=11, color="black"),
                    )
        fig_cm_variants.update_layout(
            title="Holdout Confusion Matrices by Variant",
            template="plotly_white",
            height=max(420, 340 * n_rows),
            width=max(640, 360 * n_cols),
        )
        fig_cm_variants.write_html(os.path.join(out_dir, "confusion_matrix_variants.html"))

    threshold_rows = []
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        keep = holdout_eval["pred_confidence"] >= threshold
        n_keep = int(keep.sum())
        threshold_rows.append(
            {
                "threshold": threshold,
                "n_pred": n_keep,
                "coverage_pct": (100.0 * n_keep / len(holdout_eval)) if len(holdout_eval) else 0.0,
                "accuracy_on_kept": float(holdout_eval.loc[keep, "correct"].mean()) if n_keep else np.nan,
            }
        )
    pd.DataFrame(threshold_rows).to_parquet(os.path.join(out_dir, "holdout_threshold_diagnostics.parquet"), index=False)

    fi = pd.DataFrame({"feature": feat_cols, "importance": baseline_holdout_model.feature_importances_}).sort_values("importance", ascending=False)
    fi.to_parquet(os.path.join(out_dir, "feature_importance_random_forest.parquet"), index=False)
    fig = px.bar(fi.head(30), x="importance", y="feature", orientation="h", title="Top Feature Importances (random_forest)")
    fig.write_html(os.path.join(out_dir, "feature_importance_random_forest.html"))

    rf_full = RandomForestClassifier(
        n_estimators=int(supervised_cfg.get("rf_n_estimators", 300)),
        random_state=42,
        min_samples_leaf=int(supervised_cfg.get("rf_min_samples_leaf", 1)),
        class_weight=supervised_cfg.get("rf_class_weight", "balanced_subsample"),
        n_jobs=-1,
    )
    print(
        f"[supervised] fitting full random forest | "
        f"n_estimators={rf_full.n_estimators} rows={len(X)} | RSS={_get_rss_gb():.2f} GB"
    )
    rf_full.fit(X, y)
    print(f"[supervised] full random forest fit complete | RSS={_get_rss_gb():.2f} GB")
    predictor_model = rf_full
    predictor_name = "rf_full"
    class_counts = {str(k): int(v) for k, v in y.astype(str).value_counts().sort_index().items()}
    calibration_requested = bool(supervised_cfg.get("use_probability_calibration", True))
    calibration_requested_cv = int(supervised_cfg.get("calibration_cv", 3))
    calibration_effective_cv = None
    calibration_status = "disabled"
    calibration_reason = "use_probability_calibration=false"
    if calibration_requested:
        calibration_effective_cv, class_counts = _resolve_calibration_cv(y, calibration_requested_cv)
        if calibration_effective_cv is None:
            calibration_status = "skipped"
            calibration_reason = (
                f"insufficient class support for calibration; minimum class count={min(class_counts.values()) if class_counts else 0}"
            )
            print(f"[supervised] skipping probability calibration: {calibration_reason}")
        else:
            if calibration_effective_cv != calibration_requested_cv:
                print(
                    f"[supervised] reducing calibration_cv from {calibration_requested_cv} "
                    f"to {calibration_effective_cv} based on class counts {class_counts}"
                )
            predictor_model = CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=int(supervised_cfg.get("rf_n_estimators", 300)),
                    random_state=42,
                    min_samples_leaf=int(supervised_cfg.get("rf_min_samples_leaf", 1)),
                    class_weight=supervised_cfg.get("rf_class_weight", "balanced_subsample"),
                    n_jobs=-1,
                ),
                method=str(supervised_cfg.get("calibration_method", "sigmoid")),
                cv=calibration_effective_cv,
            )
            print(
                f"[supervised] fitting calibrated classifier | "
                f"cv={calibration_effective_cv} method={supervised_cfg.get('calibration_method', 'sigmoid')} "
                f"| RSS={_get_rss_gb():.2f} GB"
            )
            predictor_model.fit(X, y)
            predictor_name = "rf_full_calibrated"
            calibration_status = "enabled"
            calibration_reason = "ok"
            print(f"[supervised] calibrated classifier fit complete | RSS={_get_rss_gb():.2f} GB")

    scoring_df = _filter_supervised_prediction_rows(model_df, supervised_cfg)
    X_all = scoring_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(
        f"[supervised] scoring windows | scope={prediction_scope} "
        f"| rows={len(X_all)} | RSS={_get_rss_gb():.2f} GB"
    )
    pred_all = predictor_model.predict(X_all)
    prob_all = predictor_model.predict_proba(X_all).max(axis=1)
    confidence_threshold = float(supervised_cfg.get("confidence_threshold", 0.40))

    prediction_df = scoring_df[["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key", "observed_label"]].copy()
    prediction_df["predicted_label_raw"] = pred_all
    prediction_df["pred_confidence"] = prob_all
    prediction_df["predicted_label"] = np.where(
        prediction_df["pred_confidence"] >= confidence_threshold,
        prediction_df["predicted_label_raw"],
        "Unknown",
    )
    prediction_df["final_behavior"] = np.where(
        prediction_df["observed_label"].notna() & (prediction_df["observed_label"].astype(str) != "Unknown"),
        prediction_df["observed_label"],
        prediction_df["predicted_label"],
    )
    rf_pre_context_prediction_df = prediction_df.copy()
    rf_context_summary = {"enabled": False, "status": "disabled", "n_rejected": 0}
    prediction_df, rf_context_summary = _apply_supervised_context_filters(ctx, prediction_df, positive_label, negative_label)

    variant_prediction_frames = [
        rf_pre_context_prediction_df.assign(
            variant_id="rf_full_pre_context",
            variant_label="RF (all channels, pre-context)",
            variant_type="random_forest",
            variant_rank=0,
        )
    ]
    if bool(rf_context_summary.get("enabled", False)):
        variant_prediction_frames.append(
            prediction_df.assign(
                variant_id="rf_full",
                variant_label="RF (all channels, context-filtered)",
                variant_type="random_forest",
                variant_rank=1,
            )
        )
    else:
        variant_prediction_frames.append(
            prediction_df.assign(
                variant_id="rf_full",
                variant_label="RF (all channels)",
                variant_type="random_forest",
                variant_rank=0,
            )
        )

    X_all = scoring_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for variant in variant_defs:
        if variant["variant_id"] == "rf_full":
            continue
        variant_id = variant["variant_id"]
        variant_label = variant["variant_label"]
        variant_type = variant["variant_type"]
        if variant_type != "algorithmic":
            variant_cols = variant["feature_cols"]
            print(
                f"[supervised] scoring windows | scope={prediction_scope} "
                f"| variant={variant_label} rows={len(X_all)} | RSS={_get_rss_gb():.2f} GB"
            )
            estimator_full = _build_supervised_estimator(supervised_cfg, variant_type)
            estimator_full.fit(X[variant_cols], y)
            pred_variant_all = estimator_full.predict(X_all[variant_cols])
            prob_variant_all = estimator_full.predict_proba(X_all[variant_cols]).max(axis=1)
        else:
            pred_variant_all = _load_algorithmic_supervised_predictions(ctx, scoring_df, positive_label, negative_label).fillna(negative_label or label_order[0]).astype(str).values
            prob_variant_all = np.where(pd.Series(pred_variant_all).notna(), 1.0, np.nan)

        variant_prediction = scoring_df[["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key", "observed_label"]].copy()
        variant_prediction["predicted_label_raw"] = pred_variant_all
        variant_prediction["pred_confidence"] = prob_variant_all
        variant_prediction["predicted_label"] = np.where(
            pd.isna(variant_prediction["pred_confidence"]) | (variant_prediction["pred_confidence"] < confidence_threshold),
            "Unknown",
            variant_prediction["predicted_label_raw"],
        )
        if variant_type == "algorithmic":
            variant_prediction["predicted_label"] = variant_prediction["predicted_label_raw"]
        variant_prediction["final_behavior"] = np.where(
            variant_prediction["observed_label"].notna() & (variant_prediction["observed_label"].astype(str) != "Unknown"),
            variant_prediction["observed_label"],
            variant_prediction["predicted_label"],
        )
        variant_prediction["variant_id"] = variant_id
        variant_prediction["variant_label"] = variant_label
        variant_prediction["variant_type"] = variant_type
        variant_prediction["variant_rank"] = int(variant["variant_rank"])
        variant_prediction_frames.append(variant_prediction)

    print(f"[supervised] prediction dataframe assembled | rows={len(prediction_df)} | RSS={_get_rss_gb():.2f} GB")
    prediction_df.to_parquet(os.path.join(out_dir, "supervised_predictions.parquet"), index=False)
    if write_csv_debug:
        prediction_df.to_csv(os.path.join(out_dir, "supervised_predictions.csv"), index=False)
    variant_prediction_df = pd.concat(variant_prediction_frames, ignore_index=True) if variant_prediction_frames else pd.DataFrame()
    if not variant_prediction_df.empty:
        variant_prediction_df.to_parquet(os.path.join(out_dir, "supervised_variant_predictions.parquet"), index=False)
        if write_csv_debug:
            variant_prediction_df.to_csv(os.path.join(out_dir, "supervised_variant_predictions.csv"), index=False)
    print(f"[supervised] wrote prediction artifacts | RSS={_get_rss_gb():.2f} GB")

    export_summary = {"deployments_written": 0, "rf_event_rows": 0}
    if bool(supervised_cfg.get("export_rf_events", False)):
        print(f"[supervised] exporting RF events back to deployments | RSS={_get_rss_gb():.2f} GB")
        export_summary = _export_rf_events_to_deployments(ctx, prediction_df, supervised_cfg=supervised_cfg)
        print(f"[supervised] RF event export complete | RSS={_get_rss_gb():.2f} GB")

    with open(summary_path, "w") as f:
        json.dump({
            "enabled": True,
            "status": "ok",
            "n_rows_labeled": int(len(labeled_df)),
            "n_rows_total": int(len(model_df)),
            "n_rows_scored": int(len(scoring_df)),
            "n_classes": int(labeled_df["observed_label"].nunique()),
            "class_counts": class_counts,
            "predictor_name": predictor_name,
            "prediction_scope": prediction_scope,
            "confidence_threshold": confidence_threshold,
            "label_prefixes": label_prefixes,
            "label_keys": label_keys,
            "label_groups": (supervised_cfg.get("label_groups") or {}),
            "variant_definitions": variant_defs,
            "skipped_variants": skipped_variants,
            "calibration": {
                "requested": calibration_requested,
                "requested_cv": calibration_requested_cv,
                "effective_cv": calibration_effective_cv,
                "status": calibration_status,
                "reason": calibration_reason,
            },
            "rf_context_filters": rf_context_summary,
            "holdout_classification_report": rep,
            "label_scan_summary": {
                "eligible_event_rows": int(sum(int(row.get("eligible_event_rows", 0) or 0) for row in label_scan_rows)),
                "quality_exclusion_event_rows": int(sum(int(row.get("quality_exclusion_event_rows", 0) or 0) for row in label_scan_rows)),
                "windows_with_any_overlap": int(sum(int(row.get("windows_with_any_overlap", 0) or 0) for row in label_scan_rows)),
                "windows_quality_excluded": int(sum(int(row.get("windows_quality_excluded", 0) or 0) for row in label_scan_rows)),
                "windows_with_multiple_labels": int(sum(int(row.get("windows_with_multiple_labels", 0) or 0) for row in label_scan_rows)),
                "windows_with_ties": int(sum(int(row.get("windows_with_ties", 0) or 0) for row in label_scan_rows)),
                "windows_labeled": int(sum(int(row.get("windows_labeled", 0) or 0) for row in label_scan_rows)),
            },
            "rf_export": export_summary,
        }, f, indent=2)

    print(f"[supervised] wrote {summary_path}")


def _write_simple_notebook(path: str, code_cells: List[str], markdown_cells: List[str] = None) -> None:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for md in (markdown_cells or []):
        nb["cells"].append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in md.strip().splitlines()],
            }
        )
    for code in code_cells:
        nb["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.rstrip().splitlines()],
            }
        )
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(nb, f, indent=2)


def _plotly_fragment(fig, include_plotlyjs: bool = False) -> str:
    if fig is None:
        return ""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=("cdn" if include_plotlyjs else False),
        config={
            "displayModeBar": True,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "plot",
            },
        },
    )


def _describe_feature_name(feature_name: str) -> str:
    feature_name = str(feature_name)
    base, transform_part, stat_part = feature_name, "raw", ""
    parts = feature_name.split("__")
    if len(parts) >= 3:
        base = parts[0]
        transform_part = parts[1]
        stat_part = "__".join(parts[2:])
    elif len(parts) == 2:
        base = parts[0]
        stat_part = parts[1]

    signal_label = base.replace(".", " / ")
    signal_pretty = signal_label.replace(" / ", " ")
    channel_map = {
        "depth depth": "depth",
        "velocity velocity": "swim speed",
        "stroke_rate stroke_rate": "stroke rate",
        "prh pitch": "pitch",
        "prh roll": "roll",
        "prh heading": "heading",
    }
    signal_pretty = channel_map.get(signal_pretty, signal_pretty)
    transform_map = {
        "raw": "the original signal values inside each time window",
        "diff": "the step-to-step change between neighboring samples inside each time window",
        "diff2": "the change in slope, computed as the second difference inside each time window",
        "abs_diff": "the absolute step-to-step change between neighboring samples inside each time window",
        "std": "the standardized signal values inside each time window",
    }
    stat_map = {
        "mean": "the average level",
        "std": "how variable the values are",
        "var": "the variance",
        "median": "the middle value",
        "min": "the minimum value",
        "max": "the maximum value",
        "range": "the spread from minimum to maximum",
        "iqr": "the middle-50% spread",
        "p10": "the 10th percentile",
        "p25": "the 25th percentile",
        "p75": "the 75th percentile",
        "p90": "the 90th percentile",
        "skew": "how asymmetric the distribution is",
        "kurtosis": "how heavy-tailed or peaked the distribution is",
        "rms": "the root-mean-square magnitude",
        "energy": "the signal energy",
        "slope": "the overall linear trend",
        "zero_crossings": "how often the series crosses zero",
        "zero_cross_rate": "how often the series crosses zero",
        "entropy": "how irregular the values are",
        "cv": "the coefficient of variation",
        "mad": "the median absolute deviation",
        "mean_abs_diff": "the average absolute change from one sample to the next",
        "var_diff": "the variance of sample-to-sample changes",
    }
    if stat_part.startswith("c22_"):
        c22_name = stat_part.replace("c22_", "").replace("_", " ")
        return (
            f"This is a catch22 time-series descriptor computed from {transform_map.get(transform_part, 'the transformed signal')} "
            f"for {signal_pretty} within each analysis window. The specific descriptor is '{c22_name}'."
        )

    transform_desc = transform_map.get(transform_part, f"{transform_part.replace('_', ' ')} values inside each time window")
    stat_desc = stat_map.get(stat_part, stat_part.replace("_", " ")) if stat_part else "a summary statistic"

    if transform_part == "raw":
        transform_sentence = f"The workflow first takes {signal_pretty} values inside each analysis window"
    elif transform_part == "diff":
        transform_sentence = f"The workflow first converts {signal_pretty} into its first difference within each analysis window"
    elif transform_part == "diff2":
        transform_sentence = f"The workflow first converts {signal_pretty} into its second difference within each analysis window"
    elif transform_part == "abs_diff":
        transform_sentence = f"The workflow first converts {signal_pretty} into absolute sample-to-sample changes within each analysis window"
    else:
        transform_sentence = f"The workflow first derives {transform_desc} for {signal_pretty}"

    return f"{transform_sentence}, then summarizes that window using {stat_desc}."


def _render_run_config_sidebar(run_cfg: dict) -> str:
    def _kv_rows(mapping: dict, preferred_keys: list[str]) -> str:
        rows = []
        for key in preferred_keys:
            if key not in mapping:
                continue
            value = mapping.get(key)
            if isinstance(value, list):
                value_html = "<br/>".join(html.escape(str(v)) for v in value[:8])
                if len(value) > 8:
                    value_html += f"<br/>... (+{len(value) - 8} more)"
            elif isinstance(value, dict):
                preview_keys = sorted(map(str, value.keys()))
                value_html = "<br/>".join(html.escape(v) for v in preview_keys[:8])
                if len(preview_keys) > 8:
                    value_html += f"<br/>... (+{len(preview_keys) - 8} more)"
            else:
                value_html = html.escape(str(value))
            rows.append(f"<tr><td>{html.escape(str(key).replace('_', ' ').title())}</td><td>{value_html}</td></tr>")
        return "".join(rows) or "<tr><td colspan='2'>No settings recorded.</td></tr>"

    scope_rows = _kv_rows(
        run_cfg,
        ["analysis_id", "dataset_ids", "deployment_ids", "groups", "duration_s", "normalization_level", "expected_signals"],
    )
    clustering_rows = _kv_rows(
        run_cfg,
        ["n_clusters", "max_k", "run_pca", "run_umap", "run_tsne", "corr_threshold", "primary_input_channel_for_ranking", "sampling_compatibility_tolerance"],
    )
    algo_rows = _kv_rows(
        run_cfg.get("algorithmic_segments") or {},
        ["enabled", "method", "method_name", "source_channel", "standardize", "base_smooth_seconds", "coarse_smooth_seconds", "end_segments_upon_d1_sign_change", "thresholds"],
    )
    supervised_rows = _kv_rows(
        run_cfg.get("supervised") or {},
        ["test_size", "confidence_threshold", "rf_n_estimators", "rf_min_samples_leaf", "rf_class_weight", "use_probability_calibration", "enable_lightgbm", "enable_svm", "enable_knn", "enable_channel_ablation", "ablation_mode", "ablation_feature_groups", "include_algorithmic_baseline", "label_groups", "drop_labels"],
    )

    return (
        "<div class='outline-card config-card' style='margin-top:18px;'>"
        "<h3>Run Config</h3>"
        "<p>Key settings used for this report.</p>"
        "<details class='collapsible side-details' open><summary>Scope</summary>"
        f"<div class='details-body'><div class='mini-table-wrap'><table><tbody>{scope_rows}</tbody></table></div></div></details>"
        "<details class='collapsible side-details'><summary>Clustering</summary>"
        f"<div class='details-body'><div class='mini-table-wrap'><table><tbody>{clustering_rows}</tbody></table></div></div></details>"
        "<details class='collapsible side-details'><summary>Algorithmic</summary>"
        f"<div class='details-body'><div class='mini-table-wrap'><table><tbody>{algo_rows}</tbody></table></div></div></details>"
        "<details class='collapsible side-details'><summary>Supervised</summary>"
        f"<div class='details-body'><div class='mini-table-wrap'><table><tbody>{supervised_rows}</tbody></table></div></div></details>"
        "</div>"
    )


def _style_report_plot(fig, height: int | None = None):
    if fig is None:
        return fig
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Figtree, sans-serif", color="#333333", size=13),
        title_font=dict(family="Figtree, sans-serif", color="#092a42", size=20),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        legend=dict(bgcolor="rgba(255,255,255,0)", title_font=dict(color="#092a42"), font=dict(color="#333333")),
        margin=dict(l=60, r=24, t=70, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dde8ef", zeroline=False, linecolor="#cfdbe5", tickfont=dict(color="#0e3551"))
    fig.update_yaxes(showgrid=True, gridcolor="#dde8ef", zeroline=False, linecolor="#cfdbe5", tickfont=dict(color="#0e3551"))
    if height is not None:
        fig.update_layout(height=height)
    return fig


def _report_color_to_rgb(color: str | None) -> Tuple[int, int, int]:
    text = str(color or "").strip()
    if text.startswith("#"):
        text = text.lstrip("#")
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) == 6:
            try:
                return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
            except Exception:
                pass
    if text.lower().startswith("rgb"):
        nums = re.findall(r"[\d.]+", text)
        if len(nums) >= 3:
            try:
                return tuple(int(float(nums[i])) for i in range(3))
            except Exception:
                pass
    return (140, 152, 164)


def _report_rgba(color: str | None, alpha: float) -> str:
    r, g, b = _report_color_to_rgb(color)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r}, {g}, {b}, {a:.4f})"


def _build_variant_metric_figure(
    metrics_df: pd.DataFrame,
    variant_id: str,
    variant_label: str,
    positive_label: str | None,
    repo_colors: Dict[str, str],
) -> go.Figure | None:
    if metrics_df is None or metrics_df.empty:
        return None
    sub = metrics_df[metrics_df["variant_id"].astype(str) == str(variant_id)].copy()
    if sub.empty:
        return None

    metric_specs = [
        ("Overall", "accuracy", "Accuracy"),
        ("Overall", "balanced_accuracy", "Balanced Acc."),
    ]
    if positive_label is not None:
        metric_specs.extend(
            [
                (positive_label, "precision", f"{positive_label} Precision"),
                (positive_label, "recall", f"{positive_label} Recall"),
                (positive_label, "f1-score", f"{positive_label} F1"),
            ]
        )

    rows = []
    for label_name, metric_name, display_name in metric_specs:
        hit = sub[
            (sub["label"].astype(str) == str(label_name))
            & (sub["metric"].astype(str) == str(metric_name))
        ].copy()
        if hit.empty:
            continue
        rows.append(
            {
                "metric_display": display_name,
                "value": float(hit.iloc[0]["value"]),
            }
        )
    if not rows:
        return None

    plot_df = pd.DataFrame(rows)
    bar_color = repo_colors.get(str(positive_label), "#0e3551") if positive_label is not None else "#0e3551"
    fig = px.bar(
        plot_df,
        x="metric_display",
        y="value",
        text="value",
        title=f"{variant_label}: Accuracy Summary",
    )
    fig.update_traces(
        marker_color=bar_color,
        texttemplate="%{text:.3f}",
        textposition="outside",
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    )
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, 1.05],
        xaxis_title="",
        yaxis_title="Score",
    )
    fig.update_xaxes(tickangle=-18)
    _style_report_plot(fig, height=330)
    return fig


def _build_variant_confusion_figure(
    confusion_df: pd.DataFrame,
    variant_id: str,
    variant_label: str,
    positive_label: str | None,
) -> go.Figure | None:
    if confusion_df is None or confusion_df.empty or positive_label is None:
        return None
    sub = confusion_df[confusion_df["variant_id"].astype(str) == str(variant_id)].copy()
    if sub.empty:
        return None

    labels = pd.Index(
        sub["observed_label"].astype(str).tolist() + sub["predicted_label"].astype(str).tolist()
    ).drop_duplicates().tolist()
    negative_label = _detect_supervised_negative_label(labels, positive_label)
    if negative_label is None:
        return None

    order = [str(negative_label), str(positive_label)]
    mat = (
        sub.pivot_table(index="observed_label", columns="predicted_label", values="count", aggfunc="sum", fill_value=0)
        .reindex(index=order, columns=order, fill_value=0)
    )
    tn = float(mat.loc[negative_label, negative_label])
    fp = float(mat.loc[negative_label, positive_label])
    fn = float(mat.loc[positive_label, negative_label])
    tp = float(mat.loc[positive_label, positive_label])
    neg_total = max(tn + fp, 0.0)
    pos_total = max(fn + tp, 0.0)
    fp_rate = (fp / neg_total) if neg_total > 0 else 0.0
    tn_rate = (tn / neg_total) if neg_total > 0 else 0.0
    fn_rate = (fn / pos_total) if pos_total > 0 else 0.0
    tp_rate = (tp / pos_total) if pos_total > 0 else 0.0

    styles = _supervised_confusion_style_map()
    cell_meta = {
        (0, 0): {"count": tn, "rate": tn_rate, "key": "true_negative", "rate_label": "TNR"},
        (1, 0): {"count": fp, "rate": fp_rate, "key": "false_positive", "rate_label": "FPR"},
        (0, 1): {"count": fn, "rate": fn_rate, "key": "false_negative", "rate_label": "FNR"},
        (1, 1): {"count": tp, "rate": tp_rate, "key": "true_positive", "rate_label": "TPR"},
    }

    fig = go.Figure()
    for (xv, yv), meta in cell_meta.items():
        style = styles.get(meta["key"], {})
        alpha = 0.12 + (0.82 * max(0.0, min(1.0, float(meta["rate"]))))
        fig.add_shape(
            type="rect",
            x0=xv - 0.5,
            x1=xv + 0.5,
            y0=yv - 0.5,
            y1=yv + 0.5,
            line=dict(color="#d7e3eb", width=1),
            fillcolor=_report_rgba(style.get("color"), alpha),
            layer="below",
        )
        fig.add_annotation(
            x=xv,
            y=yv,
            showarrow=False,
            align="center",
            text=(
                f"<b>{int(meta['count'])}</b><br>"
                f"<span style='font-size:11px'>{html.escape(meta['rate_label'])} {100.0 * float(meta['rate']):.1f}%</span>"
            ),
            font=dict(size=13, color="#092a42"),
        )

    for edge in (-0.5, 0.5, 1.5):
        fig.add_shape(type="line", x0=-0.5, x1=1.5, y0=edge, y1=edge, line=dict(color="#d7e3eb", width=1))
        fig.add_shape(type="line", x0=edge, x1=edge, y0=-0.5, y1=1.5, line=dict(color="#d7e3eb", width=1))

    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=order,
        title_text="Predicted",
        range=[-0.5, 1.5],
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=order,
        title_text="Observed",
        range=[1.5, -0.5],
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        title=f"{variant_label}: Confusion Matrix",
        showlegend=False,
    )
    _style_report_plot(fig, height=330)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def _extract_stroke_rate_frame(data_pkl) -> Tuple[pd.DataFrame, str | None]:
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    stroke_signal = signal_data.get("stroke_rate")
    if not isinstance(stroke_signal, pd.DataFrame) or stroke_signal.empty or "datetime" not in stroke_signal.columns:
        return pd.DataFrame(), None
    candidate_cols = [c for c in ["stroke_rate", "Stroke_Rate"] if c in stroke_signal.columns]
    if not candidate_cols:
        candidate_cols = [c for c in stroke_signal.columns if c != "datetime"]
    if not candidate_cols:
        return pd.DataFrame(), None
    stroke_value_col = candidate_cols[0]
    stroke_df = stroke_signal[["datetime", stroke_value_col]].copy()
    stroke_df["datetime"] = pd.to_datetime(stroke_df["datetime"], errors="coerce")
    stroke_df[stroke_value_col] = pd.to_numeric(stroke_df[stroke_value_col], errors="coerce")
    stroke_df = stroke_df.dropna(subset=["datetime", stroke_value_col]).sort_values("datetime").reset_index(drop=True)
    return stroke_df, stroke_value_col


def _compute_stroke_overlap_seconds(
    stroke_df: pd.DataFrame,
    stroke_value_col: str,
    window_df: pd.DataFrame,
    threshold: float = 10.0,
) -> pd.Series:
    if stroke_df.empty or not stroke_value_col or window_df.empty:
        return pd.Series(0.0, index=window_df.index, dtype=float)
    work = stroke_df.copy()
    work["next_datetime"] = work["datetime"].shift(-1)
    fallback_step = work["datetime"].diff().median() if len(work) > 1 else pd.Timedelta(seconds=1)
    if pd.isna(fallback_step) or fallback_step <= pd.Timedelta(0):
        fallback_step = pd.Timedelta(seconds=1)
    work["next_datetime"] = work["next_datetime"].fillna(work["datetime"] + fallback_step)
    active = work[work[stroke_value_col] > float(threshold)].copy()
    if active.empty:
        return pd.Series(0.0, index=window_df.index, dtype=float)
    # Collapse active samples into disjoint intervals and compute overlaps in vectorized form.
    active = active.sort_values("datetime").reset_index(drop=True)
    start_ns_raw = active["datetime"].astype("int64").to_numpy()
    end_ns_raw = active["next_datetime"].astype("int64").to_numpy()
    valid_mask = np.isfinite(start_ns_raw) & np.isfinite(end_ns_raw) & (end_ns_raw > start_ns_raw)
    if not valid_mask.any():
        return pd.Series(0.0, index=window_df.index, dtype=float)
    start_ns_raw = start_ns_raw[valid_mask]
    end_ns_raw = end_ns_raw[valid_mask]
    order = np.argsort(start_ns_raw, kind="mergesort")
    start_ns_raw = start_ns_raw[order]
    end_ns_raw = end_ns_raw[order]

    merged_starts: List[int] = []
    merged_ends: List[int] = []
    for s_i, e_i in zip(start_ns_raw, end_ns_raw):
        s = int(s_i)
        e = int(e_i)
        if not merged_starts or s > merged_ends[-1]:
            merged_starts.append(s)
            merged_ends.append(e)
        elif e > merged_ends[-1]:
            merged_ends[-1] = e
    starts = np.asarray(merged_starts, dtype=np.int64)
    ends = np.asarray(merged_ends, dtype=np.int64)
    dur_ns = (ends - starts).astype(np.int64)
    prefix_ns = np.concatenate(([0], np.cumsum(dur_ns, dtype=np.int64)))

    win_start = pd.to_datetime(window_df.get("window_start"), errors="coerce")
    win_end = pd.to_datetime(window_df.get("window_end"), errors="coerce")
    valid_win = win_start.notna() & win_end.notna() & (win_end > win_start)
    overlaps_ns = np.zeros(len(window_df), dtype=np.float64)
    if not valid_win.any():
        return pd.Series(overlaps_ns, index=window_df.index, dtype=float)

    valid_pos = np.flatnonzero(valid_win.to_numpy())
    w_s = win_start.iloc[valid_pos].astype("int64").to_numpy()
    w_e = win_end.iloc[valid_pos].astype("int64").to_numpy()

    left_idx = np.searchsorted(ends, w_s, side="right")
    right_idx = np.searchsorted(starts, w_e, side="left") - 1
    has_hit = left_idx <= right_idx
    if has_hit.any():
        hp = np.flatnonzero(has_hit)
        li = left_idx[hp]
        ri = right_idx[hp]
        total = prefix_ns[ri + 1] - prefix_ns[li]
        left_cut = np.maximum(0, w_s[hp] - starts[li])
        right_cut = np.maximum(0, ends[ri] - w_e[hp])
        overlap_valid = np.maximum(0, total - left_cut - right_cut).astype(np.float64)
        overlaps_ns[valid_pos[hp]] = overlap_valid

    return pd.Series(overlaps_ns / 1e9, index=window_df.index, dtype=float)


def _build_target_weak_label_summary(
    ctx: RunContext,
    variant_prediction_df: pd.DataFrame,
    positive_label: str | None,
    stroke_threshold: float = 10.0,
    min_overlap_seconds: float = 30.0,
) -> pd.DataFrame:
    if variant_prediction_df is None or variant_prediction_df.empty or positive_label is None:
        return pd.DataFrame()

    target_df = variant_prediction_df.copy()
    if "source_or_target" in target_df.columns:
        target_df = target_df[target_df["source_or_target"].astype(str) == "target"].copy()
    else:
        transfer_cfg = _parse_supervised_transfer_cfg(ctx.run_cfg)
        source_ids = set(transfer_cfg["source_dataset_ids"])
        target_df = target_df[~target_df["dataset_id"].astype(str).isin(source_ids)].copy()
    if target_df.empty:
        return pd.DataFrame()

    target_df["window_start"] = pd.to_datetime(target_df["window_start"], errors="coerce")
    target_df["window_end"] = pd.to_datetime(target_df["window_end"], errors="coerce")
    target_df = target_df.dropna(subset=["window_start", "window_end"]).copy()
    target_df["predicted_positive"] = (
        target_df["predicted_label"].astype(str).str.strip().str.lower()
        == str(positive_label).strip().lower()
    )
    target_df["window_seconds"] = pd.to_numeric(target_df.get("window_seconds"), errors="coerce").fillna(
        (target_df["window_end"] - target_df["window_start"]).dt.total_seconds()
    )

    weak_chunks: List[pd.DataFrame] = []
    for (dataset_id, deployment_id), dep_sub in target_df.groupby(["dataset_id", "deployment_id"], sort=False):
        try:
            data_pkl = _load_data_pkl(_data_pkl_path(ctx, str(dataset_id), str(deployment_id)))
        except Exception:
            continue
        deployment_tz = _deployment_timezone_name(data_pkl)
        dep_windows = _normalize_window_columns(dep_sub, deployment_tz)
        stroke_df, stroke_value_col = _extract_stroke_rate_frame(data_pkl)
        if stroke_df.empty or not stroke_value_col:
            continue
        stroke_df["datetime"] = _normalize_datetime_series_to_timezone(stroke_df["datetime"], deployment_tz)
        dep_windows["stroke_overlap_seconds"] = _compute_stroke_overlap_seconds(
            stroke_df=stroke_df,
            stroke_value_col=stroke_value_col,
            window_df=dep_windows,
            threshold=stroke_threshold,
        )
        dep_windows["weak_negative"] = dep_windows["stroke_overlap_seconds"] > float(min_overlap_seconds)
        dep_windows["stroke_source_col"] = stroke_value_col
        weak_chunks.append(dep_windows)

    if not weak_chunks:
        return pd.DataFrame()

    weak_df = pd.concat(weak_chunks, ignore_index=True, sort=False)
    weak_df["false_positive"] = weak_df["weak_negative"] & weak_df["predicted_positive"]
    weak_df["true_negative"] = weak_df["weak_negative"] & (~weak_df["predicted_positive"])
    weak_df["candidate_positive_unchecked"] = (~weak_df["weak_negative"]) & weak_df["predicted_positive"]

    summary_df = (
        weak_df.groupby(["variant_id", "variant_label", "variant_rank"], as_index=False)
        .agg(
            target_dataset_count=("dataset_id", lambda s: int(pd.Series(s).astype(str).nunique())),
            target_deployment_count=("deployment_id", lambda s: int(pd.Series(s).astype(str).nunique())),
            weak_negative_windows=("weak_negative", lambda s: int(pd.Series(s).fillna(False).sum())),
            true_negative_windows=("true_negative", lambda s: int(pd.Series(s).fillna(False).sum())),
            false_positive_windows=("false_positive", lambda s: int(pd.Series(s).fillna(False).sum())),
            predicted_positive_windows=("predicted_positive", lambda s: int(pd.Series(s).fillna(False).sum())),
            unchecked_positive_windows=("candidate_positive_unchecked", lambda s: int(pd.Series(s).fillna(False).sum())),
            weak_negative_hours=("window_seconds", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum() / 3600.0)),
        )
        .sort_values(["variant_rank", "variant_label"])
        .reset_index(drop=True)
    )
    weak_negative_total = pd.to_numeric(summary_df["weak_negative_windows"], errors="coerce").fillna(0.0)
    summary_df["false_positive_rate"] = np.where(
        weak_negative_total > 0,
        pd.to_numeric(summary_df["false_positive_windows"], errors="coerce").fillna(0.0) / weak_negative_total,
        np.nan,
    )
    summary_df["true_negative_rate"] = np.where(
        weak_negative_total > 0,
        pd.to_numeric(summary_df["true_negative_windows"], errors="coerce").fillna(0.0) / weak_negative_total,
        np.nan,
    )
    summary_df["stroke_threshold_spm"] = float(stroke_threshold)
    summary_df["min_overlap_seconds"] = float(min_overlap_seconds)
    return summary_df


def _build_target_weak_label_confusion_figure(
    weak_summary_df: pd.DataFrame,
    variant_id: str,
    variant_label: str,
    positive_label: str | None,
) -> go.Figure | None:
    if weak_summary_df is None or weak_summary_df.empty or positive_label is None:
        return None
    sub = weak_summary_df[weak_summary_df["variant_id"].astype(str) == str(variant_id)].copy()
    if sub.empty:
        return None
    row = sub.iloc[0]

    fp = float(pd.to_numeric(row.get("false_positive_windows"), errors="coerce") or 0.0)
    tn = float(pd.to_numeric(row.get("true_negative_windows"), errors="coerce") or 0.0)
    weak_total = float(pd.to_numeric(row.get("weak_negative_windows"), errors="coerce") or 0.0)
    fp_rate = float(pd.to_numeric(row.get("false_positive_rate"), errors="coerce")) if pd.notna(row.get("false_positive_rate")) else 0.0
    tn_rate = float(pd.to_numeric(row.get("true_negative_rate"), errors="coerce")) if pd.notna(row.get("true_negative_rate")) else 0.0

    styles = _supervised_confusion_style_map()
    fig = go.Figure()
    top_row_meta = {
        (0, 0): {"count": tn, "rate": tn_rate, "key": "true_negative", "rate_label": "TNR"},
        (1, 0): {"count": fp, "rate": fp_rate, "key": "false_positive", "rate_label": "FPR"},
    }
    for (xv, yv), meta in top_row_meta.items():
        style = styles.get(meta["key"], {})
        alpha = 0.12 + (0.82 * max(0.0, min(1.0, float(meta["rate"]))))
        fig.add_shape(
            type="rect",
            x0=xv - 0.5,
            x1=xv + 0.5,
            y0=yv - 0.5,
            y1=yv + 0.5,
            line=dict(color="#d7e3eb", width=1),
            fillcolor=_report_rgba(style.get("color"), alpha),
            layer="below",
        )
        fig.add_annotation(
            x=xv,
            y=yv,
            showarrow=False,
            align="center",
            text=(
                f"<b>{int(meta['count'])}</b><br>"
                f"<span style='font-size:11px'>{html.escape(meta['rate_label'])} {100.0 * float(meta['rate']):.1f}%</span>"
            ),
            font=dict(size=13, color="#092a42"),
        )

    for xv in [0, 1]:
        fig.add_shape(
            type="rect",
            x0=xv - 0.5,
            x1=xv + 0.5,
            y0=0.5,
            y1=1.5,
            line=dict(color="#d7e3eb", width=1),
            fillcolor="rgba(221, 226, 231, 0.28)",
            layer="below",
        )
        fig.add_annotation(
            x=xv,
            y=1,
            showarrow=False,
            align="center",
            text="<span style='font-size:12px'>n/a<br>no weak positive label</span>",
            font=dict(size=12, color="#6b7a88"),
        )

    for edge in (-0.5, 0.5, 1.5):
        fig.add_shape(type="line", x0=-0.5, x1=1.5, y0=edge, y1=edge, line=dict(color="#d7e3eb", width=1))
        fig.add_shape(type="line", x0=edge, x1=edge, y0=-0.5, y1=1.5, line=dict(color="#d7e3eb", width=1))

    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=[f"Predicted not {positive_label}", f"Predicted {positive_label}"],
        title_text="Predicted",
        range=[-0.5, 1.5],
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Weak negative (stroke gate)", f"{positive_label} observed"],
        title_text="Observed / weak label",
        range=[1.5, -0.5],
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        title=f"{variant_label}: Target Weak-Label Confusion",
        showlegend=False,
        margin=dict(l=20, r=20, t=56, b=62),
        annotations=list(fig.layout.annotations) + [
            go.layout.Annotation(
                x=0.5,
                y=-0.24,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=(
                    f"Weak negatives are target windows with stroke_rate > 10 for > 30 cumulative seconds. "
                    f"Observed-positive cells are intentionally blank. Weak-negative windows scored: {int(weak_total)}."
                ),
                font=dict(size=11, color="#6b7a88"),
            )
        ],
    )
    _style_report_plot(fig, height=360)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def _assign_effective_window_duration(
    df: pd.DataFrame,
    group_cols: List[str],
    start_col: str = "window_start",
    end_col: str = "window_end",
    output_col: str = "effective_window_seconds",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out[start_col] = pd.to_datetime(out[start_col], errors="coerce")
    out[end_col] = pd.to_datetime(out[end_col], errors="coerce")
    nominal = (out[end_col] - out[start_col]).dt.total_seconds()
    nominal = nominal.where(np.isfinite(nominal) & (nominal > 0), 30.0)
    out[output_col] = nominal
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        sub = out.loc[list(idx)].sort_values(start_col)
        next_start = pd.to_datetime(sub[start_col].shift(-1), errors="coerce")
        gap_s = (next_start - pd.to_datetime(sub[start_col], errors="coerce")).dt.total_seconds()
        effective = pd.concat(
            [
                nominal.loc[sub.index],
                gap_s.where(np.isfinite(gap_s) & (gap_s > 0), nominal.loc[sub.index]),
            ],
            axis=1,
        ).min(axis=1)
        out.loc[sub.index, output_col] = effective.clip(lower=0).fillna(nominal.loc[sub.index])
    return out


def _build_flow_sankey(flow_df: pd.DataFrame, source_col: str, target_col: str, value_col: str, title: str) -> go.Figure | None:
    if flow_df is None or flow_df.empty:
        return None
    labels = (
        pd.Index(flow_df[source_col].astype(str).tolist() + flow_df[target_col].astype(str).tolist())
        .drop_duplicates()
        .tolist()
    )
    node_index = {label: i for i, label in enumerate(labels)}
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=18,
                thickness=18,
                color=["#73a9c4" if i < len(pd.Index(flow_df[source_col].astype(str).unique())) else "#0e3551" for i in range(len(labels))],
            ),
            link=dict(
                source=[node_index[str(x)] for x in flow_df[source_col]],
                target=[node_index[str(x)] for x in flow_df[target_col]],
                value=[float(x) for x in flow_df[value_col]],
            ),
        )
    )
    _style_report_plot(fig, height=420)
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _select_cluster_pass_rows(cluster_df: pd.DataFrame, pass_name: str, n_clusters: int | None = None) -> pd.DataFrame:
    if cluster_df is None or cluster_df.empty:
        return pd.DataFrame()
    work = cluster_df.copy()
    pass_text = str(pass_name).strip()
    if pass_text and "cluster_pass" in work.columns:
        matched = work[work["cluster_pass"].astype(str) == pass_text].copy()
        if not matched.empty:
            return matched
    if n_clusters is not None and "cluster_pass_n_clusters" in work.columns:
        cluster_n = pd.to_numeric(work["cluster_pass_n_clusters"], errors="coerce")
        matched = work[cluster_n == float(n_clusters)].copy()
        if not matched.empty:
            return matched
    return pd.DataFrame()


def _write_algorithmic_rejection_example_plot(
    sample_df: pd.DataFrame,
    seg_row: pd.Series,
    out_path: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    start_idx = int(seg_row.get("start_index", 0))
    end_idx = int(seg_row.get("end_index", start_idx))
    if start_idx < 0 or end_idx < start_idx or sample_df.empty:
        return False

    interval_s = _infer_sampling_interval_seconds(sample_df["datetime"])
    if not np.isfinite(interval_s) or interval_s <= 0:
        interval_s = 1.0
    duration_s = max(float(seg_row.get("duration_s", 0.0) or 0.0), 1.0)
    margin_seconds = max(600.0, duration_s * 2.0)
    margin_samples = max(10, int(np.ceil(margin_seconds / interval_s)))
    lo = max(0, start_idx - margin_samples)
    hi = min(len(sample_df) - 1, end_idx + margin_samples)
    if hi <= lo:
        return False

    view = sample_df.iloc[lo:hi + 1].copy()
    view["datetime"] = pd.to_datetime(view["datetime"], errors="coerce")
    view = view.dropna(subset=["datetime"])
    if view.empty:
        return False

    seg_start = sample_df.iloc[start_idx]["datetime"]
    seg_end = sample_df.iloc[end_idx]["datetime"]
    seg_start = pd.to_datetime(seg_start, errors="coerce")
    seg_end = pd.to_datetime(seg_end, errors="coerce")
    if pd.isna(seg_start) or pd.isna(seg_end):
        return False

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True, constrained_layout=True)
    ax_depth, ax_deriv = axes

    x = view["datetime"]
    depth = pd.to_numeric(view.get("depth_std_m"), errors="coerce")
    d1 = pd.to_numeric(view.get("depth_d1_ms"), errors="coerce")
    d2 = pd.to_numeric(view.get("depth_d2_ms2"), errors="coerce")

    ax_depth.plot(x, depth, color="#0e3551", linewidth=1.6, label="depth")
    ax_depth.axvspan(seg_start, seg_end, color="#f6e7a8", alpha=0.55)
    ax_depth.set_ylabel("Depth")
    ax_depth.invert_yaxis()
    ax_depth.grid(True, alpha=0.22)
    ax_depth.legend(loc="upper right", frameon=False)

    ax_deriv.plot(x, d1, color="#286591", linewidth=1.4, label="d1")
    ax_deriv.plot(x, d2, color="#73a9c4", linewidth=1.2, label="d2")
    ax_deriv.axvspan(seg_start, seg_end, color="#f6e7a8", alpha=0.55)
    ax_deriv.axhline(0.0, color="#9aa8b4", linewidth=0.8, linestyle="--")
    ax_deriv.set_ylabel("Derivatives")
    ax_deriv.grid(True, alpha=0.22)
    ax_deriv.legend(loc="upper right", frameon=False, ncol=2)

    reason = str(seg_row.get("context_reject_reason") or "context rejected")
    drift_rate = _safe_float(seg_row.get("drift_rate_ms"))
    buoyancy = _safe_float(seg_row.get("inferred_buoyancy_value"))
    title_bits = [
        f"{seg_row.get('dataset_id', '')} / {seg_row.get('deployment_id', '')}",
        f"segment {int(seg_row.get('segment_rank', 0))}",
        reason,
    ]
    subtitle_bits = []
    if np.isfinite(drift_rate):
        subtitle_bits.append(f"drift={drift_rate:.3f} m/s")
    if np.isfinite(buoyancy):
        subtitle_bits.append(f"buoyancy={buoyancy:.3f} m/s")
    trip_phase = str(seg_row.get("inferred_trip_phase") or "").strip()
    if trip_phase:
        subtitle_bits.append(f"phase={trip_phase}")
    fig.suptitle(" | ".join([b for b in title_bits if b]) + ("\n" + " | ".join(subtitle_bits) if subtitle_bits else ""), fontsize=11)
    axes[-1].set_xlabel("Time")

    _ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _build_algorithmic_context_report(
    ctx: RunContext,
    out_dir: str,
    segments_dir: str,
    asset_prefix: str = "",
) -> Tuple[str, str]:
    summary_html = "<div class='empty-note'>No context-filtered algorithmic segments were available.</div>"
    examples_html = ""
    try:
        by_dep_dir = os.path.join(segments_dir, "by_deployment")
        if not os.path.isdir(by_dep_dir):
            return summary_html, examples_html

        algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
        if not bool((algo_cfg.get("context_filters") or [])):
            return summary_html, examples_html

        per_dep_rows = []
        rejected_rows = []
        for name in sorted(os.listdir(by_dep_dir)):
            if not name.endswith("__algorithmic_segments.parquet"):
                continue
            seg_path = os.path.join(by_dep_dir, name)
            try:
                seg_df = pd.read_parquet(seg_path)
            except Exception:
                continue
            if seg_df.empty or "segment_family" not in seg_df.columns:
                continue
            drift_df = seg_df.loc[seg_df["segment_family"].astype(str) == "drift_candidate"].copy()
            if drift_df.empty or "base_keep_filtered" not in drift_df.columns:
                continue
            drift_df["base_keep_filtered"] = drift_df["base_keep_filtered"].fillna(False).astype(bool)
            drift_df["context_keep"] = drift_df.get("context_keep", False)
            drift_df["context_keep"] = drift_df["context_keep"].fillna(False).astype(bool)
            rejected = drift_df.loc[drift_df["base_keep_filtered"] & (~drift_df["context_keep"])].copy()
            top_reason = ""
            if not rejected.empty and "context_reject_reason" in rejected.columns:
                vc = rejected["context_reject_reason"].dropna().astype(str).value_counts()
                if not vc.empty:
                    top_reason = f"{vc.index[0]} ({int(vc.iloc[0])})"
            per_dep_rows.append(
                {
                    "dataset_id": str(drift_df["dataset_id"].iloc[0]) if "dataset_id" in drift_df.columns and not drift_df.empty else "",
                    "deployment_id": str(drift_df["deployment_id"].iloc[0]) if "deployment_id" in drift_df.columns and not drift_df.empty else "",
                    "base_pass_segments": int(drift_df["base_keep_filtered"].sum()),
                    "context_rejected_segments": int(len(rejected)),
                    "final_kept_segments": int((drift_df["base_keep_filtered"] & drift_df["context_keep"]).sum()),
                    "top_reject_reason": top_reason,
                }
            )
            if not rejected.empty:
                rejected_rows.append(rejected)

        if not per_dep_rows:
            return summary_html, examples_html

        per_dep_df = pd.DataFrame(per_dep_rows).sort_values(["dataset_id", "deployment_id"]).reset_index(drop=True)
        all_rejected = pd.concat(rejected_rows, ignore_index=True, sort=False) if rejected_rows else pd.DataFrame()
        total_base = int(per_dep_df["base_pass_segments"].sum())
        total_rejected = int(per_dep_df["context_rejected_segments"].sum())
        total_final = int(per_dep_df["final_kept_segments"].sum())
        reject_rate = (100.0 * total_rejected / total_base) if total_base > 0 else float("nan")

        reason_html = ""
        if not all_rejected.empty and "context_reject_reason" in all_rejected.columns:
            vc = all_rejected["context_reject_reason"].dropna().astype(str).value_counts().head(8)
            if not vc.empty:
                reason_html = (
                    "<div class='best-method-box'>"
                    "<p><strong>Most common reject reasons</strong></p>"
                    "<ul>"
                    + "".join(
                        f"<li>{html.escape(str(reason))} <span class='metric-pill'>{int(count)}</span></li>"
                        for reason, count in vc.items()
                    )
                    + "</ul></div>"
                )

        cols = ["dataset_id", "deployment_id", "base_pass_segments", "context_rejected_segments", "final_kept_segments", "top_reject_reason"]
        table_html = (
            "<div class='metrics-table-wrap'><table><thead><tr>"
            + "".join(f"<th>{html.escape(c)}</th>" for c in cols)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(getattr(r, c, '')))}</td>" for c in cols) + "</tr>"
                for r in per_dep_df.itertuples(index=False)
            )
            + "</tbody></table></div>"
        )
        summary_html = (
            "<div class='context-summary-grid'>"
            f"<div class='best-method-box'><p><strong>Base-pass segments</strong></p><p class='context-big-number'>{total_base:,}</p></div>"
            f"<div class='best-method-box'><p><strong>Rejected by context filters</strong></p><p class='context-big-number'>{total_rejected:,}</p></div>"
            f"<div class='best-method-box'><p><strong>Final kept segments</strong></p><p class='context-big-number'>{total_final:,}</p></div>"
            f"<div class='best-method-box'><p><strong>Context rejection rate</strong></p><p class='context-big-number'>{reject_rate:.1f}%</p></div>"
            "</div>"
            + reason_html
            + table_html
        )

        if all_rejected.empty:
            return summary_html, examples_html

        example_dir = os.path.join(out_dir, "report_assets", "algorithmic_rejections")
        _ensure_dir(example_dir)
        examples = []
        source_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        for row in all_rejected.sort_values(["dataset_id", "deployment_id", "segment_rank"]).head(4).itertuples(index=False):
            dataset_id = str(getattr(row, "dataset_id", ""))
            deployment_id = str(getattr(row, "deployment_id", ""))
            cache_key = (dataset_id, deployment_id)
            if cache_key not in source_cache:
                try:
                    data_pkl = _load_data_pkl(_data_pkl_path(ctx, dataset_id, deployment_id))
                    deployment_tz = _deployment_timezone_name(data_pkl)
                    source_df, _ = _build_algorithmic_source_frame(getattr(data_pkl, "signal_data", {}) or {}, deployment_tz, algo_cfg)
                    source_cache[cache_key] = source_df
                except Exception:
                    source_cache[cache_key] = pd.DataFrame()
            sample_df = source_cache.get(cache_key)
            if sample_df is None or sample_df.empty:
                continue
            plot_name = f"{_safe_slug(dataset_id)}__{_safe_slug(deployment_id)}__segment_{int(getattr(row, 'segment_rank', 0))}.png"
            plot_path = os.path.join(example_dir, plot_name)
            if not os.path.exists(plot_path):
                ok = _write_algorithmic_rejection_example_plot(sample_df, pd.Series(row._asdict()), plot_path)
                if not ok:
                    continue
            reason = html.escape(str(getattr(row, "context_reject_reason", "") or "context rejected"))
            filter_ids = html.escape(str(getattr(row, "context_filter_ids", "") or ""))
            phase = html.escape(str(getattr(row, "inferred_trip_phase", "") or ""))
            buoyancy_phase = html.escape(str(getattr(row, "inferred_buoyancy_phase", "") or ""))
            img_src = posixpath.join(asset_prefix, "report_assets", "algorithmic_rejections", plot_name) if asset_prefix else posixpath.join("report_assets", "algorithmic_rejections", plot_name)
            examples.append(
                "<article class='rejection-card'>"
                f"<div class='rejection-meta'><h4>{html.escape(dataset_id)} / {html.escape(deployment_id)} / segment {int(getattr(row, 'segment_rank', 0))}</h4>"
                f"<p><strong>Rejected by:</strong> {reason}</p>"
                f"<p><strong>Triggered filters:</strong> {filter_ids or 'n/a'}</p>"
                f"<p><strong>Trip phase:</strong> {phase or 'unknown'} | <strong>Buoyancy phase:</strong> {buoyancy_phase or 'unknown'}</p></div>"
                f"<img class='rejection-image' src=\"{html.escape(img_src)}\" alt=\"Rejected algorithmic segment example\"/>"
                "</article>"
            )
        if examples:
            examples_html = "<div class='rejection-gallery'>" + "".join(examples) + "</div>"
    except Exception:
        return summary_html, examples_html
    return summary_html, examples_html


def _write_supervised_review_report(
    ctx: RunContext,
    out_dir: str,
    cdf: pd.DataFrame,
    spdf: pd.DataFrame,
    supervised_actogram_paths: List[str],
    primary_daily_actogram_path: str | None = None,
    primary_continuous_actogram_path: str | None = None,
    primary_event_timing_actogram_path: str | None = None,
) -> str | None:
    if spdf is None or spdf.empty:
        return None
    from pyologger.plot_data.plotter import plot_activity_budget_stacked

    supervised_dir = os.path.join(ctx.output_root, "supervised")
    features_dir = os.path.join(ctx.output_root, "features")
    clustering_dir = os.path.join(ctx.output_root, "clustering")
    segments_dir = os.path.join(ctx.output_root, "segments")
    qc_dir = os.path.join(ctx.output_root, "qc")
    report_path = os.path.join(ctx.output_root, "00_supervised_review_report.html")
    asset_prefix = os.path.relpath(out_dir, os.path.dirname(report_path))
    asset_prefix = "" if asset_prefix == "." else asset_prefix.replace(os.sep, "/")
    summary_path = os.path.join(supervised_dir, "supervised_report.json")
    metrics_path = os.path.join(supervised_dir, "metrics_by_variant.parquet")
    confusion_path = os.path.join(supervised_dir, "confusion_matrix_variants.parquet")
    variant_prediction_path = os.path.join(supervised_dir, "supervised_variant_predictions.parquet")
    fi_path = os.path.join(supervised_dir, "feature_importance_random_forest.parquet")
    summary_dir = os.path.join(ctx.output_root, "summary")
    method_budget_daily_path = os.path.join(summary_dir, "method_budget_daily.parquet")
    method_budget_hourly_path = os.path.join(summary_dir, "method_budget_hourly.parquet")
    method_budget_group_path = os.path.join(summary_dir, "method_budget_group_comparison.parquet")
    method_budget_meta_path = os.path.join(summary_dir, "method_budget_metadata.parquet")

    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    spdf = spdf.copy()
    spdf["window_start"] = pd.to_datetime(spdf["window_start"], errors="coerce")
    spdf["window_end"] = pd.to_datetime(spdf["window_end"], errors="coerce")
    spdf = spdf.dropna(subset=["window_start", "window_end"]).copy()
    # Use fixed 30 s epochs throughout the supervised review report.
    spdf["duration_s"] = 30.0
    if "variant_label" not in spdf.columns:
        spdf["variant_id"] = "rf_full"
        spdf["variant_label"] = "RF (all channels)"
        spdf["variant_type"] = "random_forest"
        spdf["variant_rank"] = 0
    if "window_key" not in spdf.columns:
        spdf["window_key"] = _build_window_key_series(spdf)
    variant_meta = (
        spdf[["variant_id", "variant_label", "variant_type", "variant_rank"]]
        .drop_duplicates()
        .sort_values(["variant_rank", "variant_label"])
        .reset_index(drop=True)
    )
    cdf_key = cdf.copy()
    cdf_key["window_key"] = _build_window_key_series(cdf_key)

    observed_reference = spdf[
        ["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key", "observed_label", "duration_s"]
    ].drop_duplicates(subset=["window_key"]).copy()
    observed_reference["predicted_label_raw"] = observed_reference["observed_label"]
    observed_reference["pred_confidence"] = 1.0
    observed_reference["predicted_label"] = observed_reference["observed_label"]
    observed_reference["final_behavior"] = observed_reference["observed_label"]
    observed_reference["variant_id"] = "observed_reference"
    observed_reference["variant_label"] = "Observed Labels"
    observed_reference["variant_type"] = "reference"
    observed_reference["variant_rank"] = -1

    budget_frames = [observed_reference.copy()]
    if "cluster_pass" in cdf_key.columns and "cluster_rank" in cdf_key.columns:
        cluster_ref = cdf_key[["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key", "cluster_pass", "cluster_rank"]].copy()
        cluster_ref["window_start"] = pd.to_datetime(cluster_ref["window_start"], errors="coerce")
        cluster_ref["window_end"] = pd.to_datetime(cluster_ref["window_end"], errors="coerce")
        cluster_ref["duration_s"] = 30.0
        cluster_ref["cluster_pass"] = cluster_ref["cluster_pass"].astype(str)
        k_mask = cluster_ref["cluster_pass"].isin(["k2", "k5"])
        if k_mask.any():
            cluster_ref = cluster_ref.loc[k_mask].copy()
            cluster_ref["observed_label"] = pd.NA
            cluster_ref["predicted_label_raw"] = cluster_ref["cluster_rank"].map(lambda x: f"Cluster {int(x)}" if pd.notna(x) else pd.NA)
            cluster_ref["pred_confidence"] = 1.0
            cluster_ref["predicted_label"] = cluster_ref["predicted_label_raw"]
            cluster_ref["final_behavior"] = cluster_ref["predicted_label_raw"]
            cluster_ref["variant_id"] = cluster_ref["cluster_pass"]
            cluster_ref["variant_label"] = cluster_ref["cluster_pass"].str.upper() + " Clusters"
            cluster_ref["variant_type"] = "cluster_reference"
            cluster_ref["variant_rank"] = cluster_ref["cluster_pass"].map({"k2": -0.5, "k5": -0.25}).fillna(0)
            budget_cols = list(
                dict.fromkeys(
                    observed_reference.columns.tolist()
                    + ["predicted_label_raw", "pred_confidence", "predicted_label", "final_behavior", "variant_id", "variant_label", "variant_type", "variant_rank"]
                )
            )
            budget_frames.append(cluster_ref[budget_cols].copy())

    budget_spdf = pd.concat(budget_frames, ignore_index=True, sort=False)
    budget_spdf = budget_spdf.dropna(subset=["final_behavior"]).copy()
    budget_spdf["date_local"] = budget_spdf["window_start"].dt.date
    budget_spdf["hour_of_day"] = budget_spdf["window_start"].dt.hour

    dep_to_group = {}
    for gname, entries in (ctx.run_cfg.get("groups") or {}).items():
        for dep in entries:
            dep_to_group[str(dep)] = str(gname)
    budget_spdf["group"] = budget_spdf["deployment_id"].astype(str).map(dep_to_group).fillna("Unknown")
    budget_spdf["locality"] = budget_spdf["group"].astype(str)
    budget_spdf["sex"] = budget_spdf["group"].astype(str).map(
        lambda g: g.rsplit("_", 1)[-1] if "_" in g and g.rsplit("_", 1)[-1] in {"M", "F"} else "Unknown"
    )
    repo_colors = _load_repo_color_mapping()
    behavior_color_map = {}
    for label in sorted(budget_spdf["final_behavior"].dropna().astype(str).unique().tolist()):
        lookup_chain = [
            label,
            label.strip(),
            label.replace("_", " "),
            label.replace("-", " "),
            label.title(),
        ]
        upper = label.strip().upper()
        if upper == "WAKE":
            lookup_chain.extend(["Active Waking", "ACTIVE", "WAKE"])
        elif upper == "ACTIVE":
            lookup_chain.extend(["Active Waking", "WAKE", "ACTIVE"])
        elif upper == "SLEEP":
            lookup_chain.extend(["HV Slow Wave Sleep", "SWS", "SLEEP"])
        elif upper == "SWS":
            lookup_chain.extend(["HV Slow Wave Sleep", "SLEEP", "SWS"])
        elif upper == "REM":
            lookup_chain.extend(["REM", "Certain REM", "Putative REM"])
        color = next((repo_colors.get(k) for k in lookup_chain if k in repo_colors), None)
        if not color:
            if upper == "REM":
                color = "#d62728"
            elif upper == "SWS":
                color = "#1f77b4"
        if color:
            behavior_color_map[label] = color
    remainder_label = "Unscorable / Not included"
    if "Unscorable" in repo_colors:
        behavior_color_map[remainder_label] = repo_colors["Unscorable"]

    daily_budget = (
        budget_spdf.groupby(
            ["variant_id", "variant_label", "variant_type", "variant_rank", "deployment_id", "locality", "sex", "date_local", "final_behavior"],
            as_index=False,
        )["duration_s"].sum()
    )
    daily_budget["hours_per_day"] = daily_budget["duration_s"] / 3600.0
    day_totals = (
        daily_budget.groupby(["variant_id", "variant_label", "variant_type", "variant_rank", "deployment_id", "locality", "sex", "date_local"], as_index=False)["hours_per_day"]
        .sum()
        .rename(columns={"hours_per_day": "covered_hours"})
    )
    remainder_rows = day_totals.copy()
    remainder_rows["hours_per_day"] = (24.0 - remainder_rows["covered_hours"]).clip(lower=0.0)
    remainder_rows["duration_s"] = remainder_rows["hours_per_day"] * 3600.0
    remainder_rows["final_behavior"] = remainder_label
    remainder_rows = remainder_rows.loc[remainder_rows["hours_per_day"] > 1e-6].copy()
    remainder_rows = remainder_rows.reindex(columns=daily_budget.columns)
    daily_budget = pd.concat([daily_budget, remainder_rows], ignore_index=True, sort=False)
    deployment_budget = (
        daily_budget.groupby(
            ["variant_id", "variant_label", "variant_type", "variant_rank", "deployment_id", "locality", "sex", "final_behavior"],
            as_index=False,
        )["hours_per_day"].mean()
    )
    hourly_budget = (
        budget_spdf.groupby(
            ["variant_id", "variant_label", "variant_type", "variant_rank", "deployment_id", "date_local", "hour_of_day", "final_behavior"],
            as_index=False,
        )["duration_s"].sum()
    )
    hourly_budget["hours_per_day_at_hour"] = hourly_budget["duration_s"] / 3600.0
    hourly_mean = (
        hourly_budget.groupby(["variant_id", "variant_label", "variant_type", "variant_rank", "hour_of_day", "final_behavior"], as_index=False)["hours_per_day_at_hour"]
        .mean()
    )

    behavior_order = sorted(daily_budget["final_behavior"].dropna().astype(str).unique().tolist())
    daily_figs = {}
    hourly_figs = {}
    budget_meta = (
        budget_spdf[["variant_id", "variant_label", "variant_type", "variant_rank"]]
        .drop_duplicates()
        .sort_values(["variant_rank", "variant_label"])
        .reset_index(drop=True)
    )
    for row in budget_meta.itertuples(index=False):
        dep_budget_variant = deployment_budget[deployment_budget["variant_id"] == row.variant_id].copy()
        if not dep_budget_variant.empty:
            fig_daily = plot_activity_budget_stacked(
                budget_df=dep_budget_variant,
                value_col="hours_per_day",
                behavior_col="final_behavior",
                deployment_col="deployment_id",
                group_col="locality",
                metadata_label_col="sex",
                behavior_order=behavior_order,
                color_map=behavior_color_map,
                title=f"{row.variant_label}: Daily Activity Budget by Deployment",
            )
            fig_daily.update_layout(yaxis_title="h/day", yaxis_range=[0, 24])
            _style_report_plot(fig_daily)
            daily_figs[row.variant_id] = _plotly_fragment(fig_daily)

        hourly_variant = hourly_mean[hourly_mean["variant_id"] == row.variant_id].copy()
        if not hourly_variant.empty:
            fig_hourly = px.area(
                hourly_variant,
                x="hour_of_day",
                y="hours_per_day_at_hour",
                color="final_behavior",
                title=f"{row.variant_label}: Mean Hourly Activity Budget",
                category_orders={"final_behavior": behavior_order},
                labels={"hour_of_day": "Hour of Day", "hours_per_day_at_hour": "Mean h/day at hour", "final_behavior": "State"},
                color_discrete_map=behavior_color_map,
            )
            fig_hourly.update_layout(yaxis_title="Mean h/day at hour", yaxis_range=[0, 1.05], xaxis=dict(dtick=2))
            _style_report_plot(fig_hourly)
            hourly_figs[row.variant_id] = _plotly_fragment(fig_hourly)

    primary_budget_section_html = "<div class='empty-note'>No method-budget summary available.</div>"
    circadian_section_html = "<div class='empty-note'>No circadian summary available.</div>"
    primary_actogram_gallery_html = "<div class='empty-note'>No primary-method actogram assets available.</div>"
    group_contrast_section_html = "<div class='empty-note'>No group/dataset budget comparison available.</div>"
    unmapped_labels_html = ""

    metrics_fragment = ""
    per_fold_metrics_html = ""
    metrics_df = pd.DataFrame()
    best_method_html = "<div class='empty-note'>No performance summary available.</div>"
    algorithmic_performance_html = ""
    if os.path.exists(metrics_path):
            metrics_df_raw = pd.read_parquet(metrics_path)
            metrics_df = metrics_df_raw.copy()
            if not metrics_df.empty:
                if "fold_id" in metrics_df.columns:
                    group_cols = ["variant_id", "variant_label", "variant_type", "variant_rank", "label", "metric"]
                    metrics_df = metrics_df.groupby(group_cols, as_index=False)["value"].mean()
                    fold_figures = []
                    for fold_id, fold_sub in metrics_df_raw.groupby("fold_id", sort=False):
                        fold_plot_df = fold_sub.copy()
                        fold_label_order = _metric_label_order(fold_plot_df["label"].astype(str).tolist())
                        fold_plot_df["label"] = pd.Categorical(
                            fold_plot_df["label"],
                            categories=fold_label_order,
                            ordered=True,
                        )
                        fold_plot_df["metric"] = pd.Categorical(
                            fold_plot_df["metric"],
                            categories=["precision", "recall", "f1-score", "accuracy", "balanced_accuracy"],
                            ordered=True,
                        )
                        fold_plot_df["pct_label"] = (
                            pd.to_numeric(fold_plot_df["value"], errors="coerce")
                            .fillna(0.0)
                            .map(lambda v: f"{100.0 * float(v):.0f}%")
                        )
                        fold_title = str(fold_id).replace("holdout_", "").replace("_", " ")
                        fig_fold = px.bar(
                            fold_plot_df,
                            x="variant_label",
                            y="value",
                            color="label",
                            text="pct_label",
                            barmode="group",
                            facet_col="metric",
                            facet_col_wrap=min(3, max(1, len(fold_plot_df["metric"].astype(str).unique()))),
                            title=f"Held-out deployment: {fold_title}",
                            color_discrete_map=_metric_label_color_map(fold_label_order, repo_colors=repo_colors),
                            hover_data={"value":":.3f"},
                        )
                        fig_fold.update_layout(yaxis_range=[0, 1.05], bargap=0.22, bargroupgap=0.08, legend_title_text="Evaluation target")
                        fig_fold.update_traces(
                            texttemplate="%{text}",
                            textposition="outside",
                            textangle=-90,
                            cliponaxis=False,
                            textfont=dict(size=13, color="#092a42"),
                        )
                        fig_fold.update_xaxes(tickangle=-24, title_text="")
                        fig_fold.update_yaxes(title_text="Score")
                        _style_report_plot(fig_fold, height=max(500, 340 * int(np.ceil(len(fold_plot_df["metric"].astype(str).unique()) / 3))))
                        fold_figures.append(
                            "<div class='plot-shell' style='margin-top:14px;'>"
                            + _plotly_fragment(fig_fold)
                            + "</div>"
                        )
                    if fold_figures:
                        per_fold_metrics_html = (
                            "<section class='card'>"
                            "<div class='section-head'>"
                            "<h2>Held-out Deployment Performance</h2>"
                            "<p>Each panel below is a separate held-out deployment split. These are not duplicate models; they are the same method family evaluated on different held-out JKB deployments.</p>"
                            "</div>"
                            + "".join(fold_figures)
                            + "</section>"
                        )
                metrics_label_order = _metric_label_order(metrics_df["label"].astype(str).tolist())
                metrics_df["label"] = pd.Categorical(metrics_df["label"], categories=metrics_label_order, ordered=True)
                metrics_df["metric"] = pd.Categorical(
                metrics_df["metric"],
                categories=["precision", "recall", "f1-score", "accuracy", "balanced_accuracy"],
                ordered=True,
            )
            best_accuracy_label = None
            acc_sub = metrics_df[(metrics_df["metric"].astype(str) == "accuracy") & (metrics_df["label"].astype(str) == "Overall")].copy()
            if not acc_sub.empty:
                best_accuracy_label = str(acc_sub.sort_values(["value", "variant_rank"], ascending=[False, True]).iloc[0]["variant_label"])
            metrics_df["variant_display_label"] = metrics_df["variant_label"].astype(str)
            if best_accuracy_label:
                metrics_df.loc[metrics_df["variant_display_label"] == best_accuracy_label, "variant_display_label"] = (
                    "<b>" + best_accuracy_label + "</b>"
                )
            metrics_df["pct_label"] = (
                pd.to_numeric(metrics_df["value"], errors="coerce")
                .fillna(0.0)
                .map(lambda v: f"{100.0 * float(v):.0f}%")
            )
            metric_colors = _metric_label_color_map(metrics_label_order, repo_colors=repo_colors)
            fig_metrics = px.bar(
                metrics_df,
                x="variant_display_label",
                y="value",
                color="label",
                text="pct_label",
                barmode="group",
                facet_col="metric",
                facet_col_wrap=min(3, max(1, len(metrics_df["metric"].astype(str).unique()))),
                title="Method Comparison by Performance Metric (mean across held-out deployments)" if "fold_id" in metrics_df_raw.columns else "Method Comparison by Performance Metric",
                color_discrete_map=metric_colors,
                category_orders={
                    "label": metrics_label_order,
                    "metric": ["precision", "recall", "f1-score", "accuracy", "balanced_accuracy"],
                    "variant_display_label": metrics_df["variant_display_label"].drop_duplicates().tolist(),
                },
                hover_data={"variant_label": False, "value":":.3f"},
            )
            fig_metrics.update_layout(yaxis_range=[0, 1.05], bargap=0.22, bargroupgap=0.08, legend_title_text="Evaluation target")
            fig_metrics.update_traces(
                texttemplate="%{text}",
                textposition="outside",
                textangle=-90,
                cliponaxis=False,
                textfont=dict(size=13, color="#092a42"),
            )
            fig_metrics.update_xaxes(tickangle=-24, title_text="")
            fig_metrics.update_yaxes(title_text="Score")
            _style_report_plot(fig_metrics, height=max(540, 360 * int(np.ceil(len(metrics_df["metric"].astype(str).unique()) / 3))))
            metrics_fragment = _plotly_fragment(fig_metrics, include_plotlyjs=True)
            best_rows = []
            focus_label = _primary_metric_focus_label(metrics_df["label"].astype(str).tolist())
            metric_targets = [("balanced_accuracy", "Overall"), ("accuracy", "Overall")]
            if focus_label is not None:
                metric_targets.extend([("recall", focus_label), ("precision", focus_label), ("f1-score", focus_label)])
            for metric_name, label_name in metric_targets:
                sub = metrics_df[(metrics_df["metric"].astype(str) == metric_name) & (metrics_df["label"].astype(str) == label_name)].copy()
                if sub.empty:
                    continue
                top = sub.sort_values(["value", "variant_rank"], ascending=[False, True]).iloc[0]
                best_rows.append(
                    f"<li><strong>{html.escape(metric_name.replace('_', ' ').title())}</strong>: "
                    f"{html.escape(str(top['variant_label']))} "
                    f"<span class='metric-pill'>{float(top['value']):.3f}</span></li>"
                )
            if best_rows:
                best_method_html = (
                    "<div class='best-method-box'><p>The strongest methods on this run, based on held-out performance, are:</p>"
                    f"<ul>{''.join(best_rows)}</ul>"
                    f"<p class='small-note'>Balanced accuracy is the safest default ranking when class balance matters. "
                    + (
                        f"{html.escape(str(focus_label))}-specific precision, recall, and F1 are highlighted separately for this label set."
                        if focus_label is not None
                        else "Class-specific precision, recall, and F1 are highlighted separately for this label set."
                    )
                    + "</p></div>"
                )

    try:
        method_budget_daily_df = pd.read_parquet(method_budget_daily_path) if os.path.exists(method_budget_daily_path) else pd.DataFrame()
        method_budget_hourly_df = pd.read_parquet(method_budget_hourly_path) if os.path.exists(method_budget_hourly_path) else pd.DataFrame()
        method_budget_group_df = pd.read_parquet(method_budget_group_path) if os.path.exists(method_budget_group_path) else pd.DataFrame()
        method_budget_meta_df = pd.read_parquet(method_budget_meta_path) if os.path.exists(method_budget_meta_path) else pd.DataFrame()
        if not method_budget_daily_df.empty:
            meta_cols = [c for c in ["method", "method_variant", "method_label"] if c in method_budget_daily_df.columns]
            method_meta = method_budget_daily_df[meta_cols].drop_duplicates().reset_index(drop=True)
            primary_method, primary_variant = _resolve_primary_budget_method(method_meta, metrics_df, ctx.run_cfg)
            focus_cfg = _summary_focus_cfg(ctx.run_cfg)
            primary_daily = method_budget_daily_df[
                (method_budget_daily_df["method"].astype(str) == str(primary_method))
                & (method_budget_daily_df["method_variant"].astype(str) == str(primary_variant))
            ].copy()
            primary_hourly = method_budget_hourly_df[
                (method_budget_hourly_df["method"].astype(str) == str(primary_method))
                & (method_budget_hourly_df["method_variant"].astype(str) == str(primary_variant))
            ].copy()
            if not primary_daily.empty:
                primary_label = str(primary_daily.get("method_label", pd.Series(["Primary method"])).iloc[0])
                fig_primary_daily = px.bar(
                    primary_daily,
                    x="local_date",
                    y="duration_h",
                    color="state_label_canonical" if focus_cfg["budget_label_mode"] == "canonical" else "state_label_raw",
                    title=f"Primary method daily budgets: {primary_label}",
                    labels={"duration_h": "h/day", "local_date": "Local date"},
                )
                fig_primary_daily.update_layout(template="plotly_white", yaxis_range=[0, 24])
                _style_report_plot(fig_primary_daily, height=410)
                fig_primary_hourly = px.area(
                    primary_hourly,
                    x="hour_of_day",
                    y="pct_of_observed_hour",
                    color="state_label_canonical" if focus_cfg["budget_label_mode"] == "canonical" else "state_label_raw",
                    title=f"Primary method hourly circadian profile: {primary_label}",
                    labels={"pct_of_observed_hour": "% observed hour", "hour_of_day": "Hour of day"},
                ) if not primary_hourly.empty else None
                if fig_primary_hourly is not None:
                    fig_primary_hourly.update_layout(template="plotly_white")
                    fig_primary_hourly.update_xaxes(dtick=2)
                    _style_report_plot(fig_primary_hourly, height=390)
                    hourly_fragment = _plotly_fragment(fig_primary_hourly)
                else:
                    hourly_fragment = "<div class='empty-note'>No hourly profile available for primary method.</div>"
                primary_summary_cards = (
                    "<div class='context-summary-grid'>"
                    f"<div class='best-method-box'><p class='small-note'>Primary method</p><p class='context-big-number'>{html.escape(primary_label)}</p></div>"
                    f"<div class='best-method-box'><p class='small-note'>Deployments</p><p class='context-big-number'>{int(primary_daily['deployment_id'].astype(str).nunique())}</p></div>"
                    f"<div class='best-method-box'><p class='small-note'>Days</p><p class='context-big-number'>{int(primary_daily['local_date'].nunique())}</p></div>"
                    "</div>"
                )
                primary_imgs = []
                for path in [primary_daily_actogram_path, primary_continuous_actogram_path, primary_event_timing_actogram_path]:
                    if path and os.path.exists(path):
                        rel = posixpath.join(asset_prefix, os.path.basename(path)) if asset_prefix else os.path.basename(path)
                        primary_imgs.append(
                            f"<div class='plot-shell'><img src=\"{html.escape(rel)}\" class=\"actogram-image\" alt=\"Primary method actogram\"/></div>"
                        )
                primary_budget_section_html = (
                    primary_summary_cards
                    + "<div class='two-col'>"
                    + f"<div class='plot-shell'>{_plotly_fragment(fig_primary_daily)}</div>"
                    + f"<div class='plot-shell'>{hourly_fragment}</div>"
                    + "</div>"
                    + ("<div class='two-col' style='margin-top:12px;'>" + "".join(primary_imgs) + "</div>" if primary_imgs else "")
                )
                if primary_imgs:
                    primary_actogram_gallery_html = "<div class='two-col'>" + "".join(primary_imgs) + "</div>"
                solar_daily = primary_daily.drop_duplicates("local_date")[["local_date", "sunrise_local_hour", "sunset_local_hour"]].copy()
                if not solar_daily.empty:
                    fig_solar = go.Figure()
                    fig_solar.add_trace(go.Scatter(x=solar_daily["local_date"], y=solar_daily["sunrise_local_hour"], mode="lines+markers", name="Sunrise"))
                    fig_solar.add_trace(go.Scatter(x=solar_daily["local_date"], y=solar_daily["sunset_local_hour"], mode="lines+markers", name="Sunset"))
                    fig_solar.update_layout(template="plotly_white", yaxis_title="Local hour", xaxis_title="Date")
                    _style_report_plot(fig_solar, height=330)
                    circadian_section_html = _plotly_fragment(fig_solar)
            if not method_budget_group_df.empty:
                cmp_df = method_budget_group_df.copy()
                cmp_df["comparison_level"] = cmp_df["comparison_level"].astype(str)
                group_view = cmp_df[cmp_df["comparison_level"].isin(["group", "dataset"])]
                if not group_view.empty:
                    fig_group = px.bar(
                        group_view,
                        x="comparison_key",
                        y="pct_of_24h",
                        color="state_label_canonical",
                        facet_row="comparison_level",
                        barmode="stack",
                        title="Group and dataset activity budget contrasts",
                        labels={"comparison_key": "Group / dataset", "pct_of_24h": "% of 24h"},
                    )
                    fig_group.update_layout(template="plotly_white")
                    _style_report_plot(fig_group, height=560)
                    group_contrast_section_html = _plotly_fragment(fig_group)
        if not method_budget_meta_df.empty and {"raw_label", "canonical_label"}.issubset(method_budget_meta_df.columns):
            preview = method_budget_meta_df[method_budget_meta_df["canonical_label"].astype(str).str.startswith("Unmapped:")].head(20)
            if not preview.empty:
                unmapped_labels_html = (
                    "<div class='best-method-box'><p><strong>Unmapped raw labels</strong> (update canonical mapping if needed): "
                    + ", ".join(html.escape(str(x)) for x in preview["raw_label"].dropna().astype(str).tolist())
                    + "</p></div>"
                )
    except Exception:
        pass

    try:
        algo_enabled = bool((ctx.run_cfg.get("algorithmic_segments") or {}).get("enabled", False))
        observed_eval = (
            spdf[["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key", "observed_label"]]
            .drop_duplicates(subset=["window_key"])
            .dropna(subset=["observed_label"])
            .copy()
        )
        if algo_enabled and not observed_eval.empty:
            observed_eval["observed_label"] = observed_eval["observed_label"].astype(str)
            observed_eval = observed_eval.loc[_observed_label_keep_mask(observed_eval.get("observed_label"))].copy()
        if algo_enabled and not observed_eval.empty:
            label_order = sorted(observed_eval["observed_label"].dropna().astype(str).unique().tolist())
            positive_label = _detect_supervised_positive_label(label_order)
            negative_label = _detect_supervised_negative_label(label_order, positive_label)
            detailed_path, _ = _algorithmic_segment_merge_paths(ctx)
            if os.path.exists(detailed_path) and positive_label is not None:
                seg_df = pd.read_parquet(detailed_path)
                if not seg_df.empty and "base_nominal_class" in seg_df.columns and "nominal_class" in seg_df.columns:
                    y_true = observed_eval["observed_label"].astype(str)
                    base_pred = _load_algorithmic_supervised_predictions_from_segment_df(
                        ctx,
                        observed_eval,
                        seg_df,
                        positive_label=positive_label,
                        negative_label=negative_label,
                        class_column="base_nominal_class",
                    )
                    final_pred = _load_algorithmic_supervised_predictions_from_segment_df(
                        ctx,
                        observed_eval,
                        seg_df,
                        positive_label=positive_label,
                        negative_label=negative_label,
                        class_column="nominal_class",
                    )
                    algo_metric_frames = []
                    for variant_id, variant_label, pred in [
                        ("algorithmic_base", "Algorithmic without context", base_pred),
                        ("algorithmic_context", "Algorithmic with context", final_pred),
                    ]:
                        pred = pd.Series(pred, index=observed_eval.index, dtype=object)
                        eval_mask = pred.notna()
                        if not eval_mask.any():
                            continue
                        _, _, metrics_long = _evaluate_supervised_variant(
                            y_true=y_true.loc[eval_mask].astype(str).reset_index(drop=True),
                            y_pred=pred.loc[eval_mask].astype(str).reset_index(drop=True),
                            label_order=label_order,
                            variant_id=variant_id,
                            variant_label=variant_label,
                            variant_type="algorithmic_report",
                        )
                        algo_metric_frames.append(metrics_long)
                    if algo_metric_frames:
                        algo_metrics_df = pd.concat(algo_metric_frames, ignore_index=True, sort=False)
                        metric_targets = [
                            ("Overall", "accuracy", "Overall accuracy"),
                            ("Overall", "balanced_accuracy", "Overall balanced accuracy"),
                            (positive_label, "precision", f"{positive_label} precision"),
                            (positive_label, "recall", f"{positive_label} recall"),
                            (positive_label, "f1-score", f"{positive_label} F1"),
                        ]
                        algo_rows = []
                        for label_name, metric_name, display_name in metric_targets:
                            sub = algo_metrics_df[
                                (algo_metrics_df["label"].astype(str) == str(label_name))
                                & (algo_metrics_df["metric"].astype(str) == str(metric_name))
                            ].copy()
                            if sub.empty:
                                continue
                            sub["metric_display"] = display_name
                            algo_rows.append(sub[["variant_label", "metric_display", "value"]])
                        if algo_rows:
                            algo_compare = pd.concat(algo_rows, ignore_index=True, sort=False)
                            fig_algo_compare = px.bar(
                                algo_compare,
                                x="metric_display",
                                y="value",
                                color="variant_label",
                            barmode="group",
                            labels={"metric_display": "Metric", "value": "Score", "variant_label": "Algorithmic variant"},
                            title="Algorithmic Performance: Before vs After Context Filtering",
                            color_discrete_map={
                                "Algorithmic without context": "#73a9c4",
                                "Algorithmic with context": "#0e3551",
                            },
                        )
                        fig_algo_compare.update_layout(yaxis_range=[0, 1.05], legend_title_text="Algorithmic variant")
                        fig_algo_compare.update_xaxes(tickangle=-18)
                        _style_report_plot(fig_algo_compare, height=380)
                        algo_compare_table = (
                            algo_compare.pivot_table(index="metric_display", columns="variant_label", values="value", aggfunc="first")
                            .reset_index()
                            .rename_axis(None, axis=1)
                        )
                        table_cols = algo_compare_table.columns.tolist()
                        table_html = (
                            "<div class='metrics-table-wrap'><table><thead><tr>"
                            + "".join(f"<th>{html.escape(str(c))}</th>" for c in table_cols)
                            + "</tr></thead><tbody>"
                            + "".join(
                                "<tr>" + "".join(
                                    (
                                        f"<td>{float(row.get(c)):.3f}</td>"
                                        if c != "metric_display" and pd.notna(row.get(c))
                                        else f"<td>{html.escape(str(row.get(c, '')))}</td>"
                                    )
                                    for c in table_cols
                                ) + "</tr>"
                                for row in algo_compare_table.to_dict(orient="records")
                            )
                            + "</tbody></table></div>"
                        )
                        algorithmic_performance_html = (
                            "<div class='section-head'><h3>Algorithmic Baseline Comparison</h3>"
                            "<p>These metrics use the same labeled windows as the supervised methods, but compare the rule-based detector before and after buoyancy-aware context filtering.</p></div>"
                            f"<div class='plot-shell'>{_plotly_fragment(fig_algo_compare)}</div>"
                            f"{table_html}"
                        )
    except Exception:
        algorithmic_performance_html = ""

    performance_pair_cards_html = ""
    target_weak_label_cards_html = ""
    target_weak_label_table_html = "<div class='empty-note'>No target weak-label summary available.</div>"
    if os.path.exists(confusion_path):
        confusion_df = pd.read_parquet(confusion_path)
        if not confusion_df.empty:
            conf_meta = (
                confusion_df[["variant_id", "variant_label", "variant_rank"]]
                .drop_duplicates()
                .sort_values(["variant_rank", "variant_label"])
            )
            metric_labels = metrics_df["label"].astype(str).tolist() if not metrics_df.empty else []
            confusion_labels = (
                confusion_df["observed_label"].astype(str).tolist()
                + confusion_df["predicted_label"].astype(str).tolist()
            )
            positive_label = _detect_supervised_positive_label(metric_labels + confusion_labels)
            card_bits = []
            for row in conf_meta.itertuples(index=False):
                metric_fig = _build_variant_metric_figure(
                    metrics_df=metrics_df,
                    variant_id=str(row.variant_id),
                    variant_label=str(row.variant_label),
                    positive_label=positive_label,
                    repo_colors=repo_colors,
                )
                conf_fig = _build_variant_confusion_figure(
                    confusion_df=confusion_df,
                    variant_id=str(row.variant_id),
                    variant_label=str(row.variant_label),
                    positive_label=positive_label,
                )
                metric_html = _plotly_fragment(metric_fig) if metric_fig is not None else "<div class='empty-note'>No method metrics available.</div>"
                conf_html = _plotly_fragment(conf_fig) if conf_fig is not None else "<div class='empty-note'>No binary confusion view available.</div>"
                card_bits.append(
                    "<section class='card'>"
                    f"<div class='section-head'><h3>{html.escape(str(row.variant_label))}</h3>"
                    "<p>Metric summary and confusion structure for this method. False-positive and false-negative cells use their own colors and darken with the corresponding error rate.</p></div>"
                    "<div class='two-col'>"
                    f"<div class='plot-shell'>{metric_html}</div>"
                    f"<div class='plot-shell'>{conf_html}</div>"
                    "</div></section>"
                )
            performance_pair_cards_html = "".join(card_bits)

            if os.path.exists(variant_prediction_path):
                try:
                    variant_prediction_df = pd.read_parquet(variant_prediction_path)
                    print("[plots] report step: building target weak-label summary")
                    weak_summary_df = _build_target_weak_label_summary(
                        ctx=ctx,
                        variant_prediction_df=variant_prediction_df,
                        positive_label=positive_label,
                        stroke_threshold=10.0,
                        min_overlap_seconds=30.0,
                    )
                    if not weak_summary_df.empty:
                        print(f"[plots] report step: target weak-label summary built for {len(weak_summary_df)} variant(s)")
                        weak_summary_df.to_parquet(
                            os.path.join(supervised_dir, "target_weak_label_confusion_variants.parquet"),
                            index=False,
                        )
                        weak_table_rows = []
                        weak_card_bits = []
                        for row in weak_summary_df.itertuples(index=False):
                            weak_fig = _build_target_weak_label_confusion_figure(
                                weak_summary_df=weak_summary_df,
                                variant_id=str(row.variant_id),
                                variant_label=str(row.variant_label),
                                positive_label=positive_label,
                            )
                            weak_html = _plotly_fragment(weak_fig) if weak_fig is not None else "<div class='empty-note'>No target weak-label confusion available.</div>"
                            weak_card_bits.append(
                                "<section class='card'>"
                                f"<div class='section-head'><h3>{html.escape(str(row.variant_label))}</h3>"
                                "<p>Target-only weak-label review. Windows with stroke_rate > 10 for more than 30 cumulative seconds are treated as weak negatives. False negatives are intentionally omitted because no weak positive label exists.</p></div>"
                                f"<div class='plot-shell'>{weak_html}</div>"
                                "</section>"
                            )
                            weak_table_rows.append(
                                {
                                    "variant_label": str(row.variant_label),
                                    "target_dataset_count": int(getattr(row, "target_dataset_count", 0)),
                                    "target_deployment_count": int(getattr(row, "target_deployment_count", 0)),
                                    "weak_negative_windows": int(getattr(row, "weak_negative_windows", 0)),
                                    "true_negative_windows": int(getattr(row, "true_negative_windows", 0)),
                                    "false_positive_windows": int(getattr(row, "false_positive_windows", 0)),
                                    "false_positive_rate": (
                                        f"{100.0 * float(getattr(row, 'false_positive_rate', float('nan'))):.1f}%"
                                        if pd.notna(getattr(row, "false_positive_rate", np.nan))
                                        else "n/a"
                                    ),
                                    "unchecked_positive_windows": int(getattr(row, "unchecked_positive_windows", 0)),
                                }
                            )
                        target_weak_label_cards_html = "".join(weak_card_bits)
                        target_weak_label_table_html = _render_metrics_table_html(
                            weak_table_rows,
                            [
                                ("variant_label", "Method"),
                                ("target_dataset_count", "Target datasets"),
                                ("target_deployment_count", "Target deployments"),
                                ("weak_negative_windows", "Weak-negative windows"),
                                ("true_negative_windows", "True negatives"),
                                ("false_positive_windows", "False positives"),
                                ("false_positive_rate", "False-positive rate"),
                                ("unchecked_positive_windows", "Predicted sleep not contradicted"),
                            ],
                        )
                except Exception:
                    target_weak_label_cards_html = ""
                    target_weak_label_table_html = "<div class='empty-note'>Target weak-label review failed to build.</div>"

    fi_fragment = ""
    fi_df = pd.DataFrame()
    if os.path.exists(fi_path):
        fi_df = pd.read_parquet(fi_path)
        if not fi_df.empty:
            fig_fi = px.bar(
                fi_df.head(30),
                x="importance",
                y="feature",
                orientation="h",
                title="Random Forest Feature Importances",
            )
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
            _style_report_plot(fig_fi, height=720)
            fi_fragment = _plotly_fragment(fig_fi)

    qc_summary_html = "<div class='empty-note'>No QC summary available.</div>"
    qc_path = os.path.join(qc_dir, "qc_channels.csv")
    if os.path.exists(qc_path):
        try:
            qc_df = pd.read_csv(qc_path)
            if not qc_df.empty:
                cols = [c for c in ["dataset_id", "deployment_id", "status", "available_channels", "missing_required_channels"] if c in qc_df.columns]
                qc_summary_html = "<div class='metrics-table-wrap'><table><thead><tr>" + "".join([f"<th>{html.escape(c)}</th>" for c in cols]) + "</tr></thead><tbody>" + "".join(
                    [
                        "<tr>" + "".join([f"<td>{html.escape(str(getattr(r, c, '')))}</td>" for c in cols]) + "</tr>"
                        for r in qc_df.itertuples(index=False)
                    ]
                ) + "</tbody></table></div>"
        except Exception:
            pass

    feature_filter_html = "<div class='empty-note'>No feature-filter summary available.</div>"
    filter_report_path = os.path.join(features_dir, "feature_filter_report.json")
    dropped_path = os.path.join(features_dir, "dropped_correlated_features.parquet")
    if os.path.exists(filter_report_path):
        try:
            with open(filter_report_path, "r") as f:
                filter_summary = json.load(f)
            dropped_preview = []
            if os.path.exists(dropped_path):
                dropped_df = pd.read_parquet(dropped_path)
                dropped_preview = dropped_df["dropped_feature"].astype(str).head(12).tolist() if "dropped_feature" in dropped_df.columns else []
            feature_filter_html = (
                "<div class='best-method-box'>"
                f"<p><strong>Input features:</strong> {int(filter_summary.get('input_feature_count', 0)):,}</p>"
                f"<p><strong>Kept features:</strong> {int(filter_summary.get('kept_feature_count', 0)):,}</p>"
                f"<p><strong>Dropped correlated features:</strong> {int(filter_summary.get('dropped_feature_count', 0)):,}</p>"
                f"<p><strong>Correlation threshold:</strong> {float(filter_summary.get('corr_threshold', float('nan'))):.2f}</p>"
                + (
                    "<p><strong>Example dropped features:</strong><br/>" + "<br/>".join([html.escape(x) for x in dropped_preview]) + "</p>"
                    if dropped_preview else ""
                )
                + "</div>"
            )
        except Exception:
            pass

    threshold_fragment = ""
    threshold_path = os.path.join(supervised_dir, "holdout_threshold_diagnostics.parquet")
    if os.path.exists(threshold_path):
        try:
            threshold_df = pd.read_parquet(threshold_path)
            if not threshold_df.empty:
                fig_thr = px.line(
                    threshold_df,
                    x="threshold",
                    y=["coverage_pct", "accuracy_on_kept"],
                    markers=True,
                    title="Holdout Threshold Diagnostics",
                    labels={"value": "Score / Coverage", "threshold": "Confidence threshold", "variable": "Metric"},
                )
                fig_thr.update_layout(yaxis_range=[0, 100])
                _style_report_plot(fig_thr, height=360)
                threshold_fragment = _plotly_fragment(fig_thr)
        except Exception:
            pass

    label_scan_html = "<div class='empty-note'>No label-scan report available.</div>"
    label_scan_path = os.path.join(supervised_dir, "label_scan_report.parquet")
    if os.path.exists(label_scan_path):
        try:
            label_scan_df = pd.read_parquet(label_scan_path)
            if not label_scan_df.empty:
                cols = [c for c in ["dataset_id", "deployment_id", "eligible_event_rows", "windows_labeled", "windows_with_any_overlap", "error_type"] if c in label_scan_df.columns]
                label_scan_html = "<div class='metrics-table-wrap'><table><thead><tr>" + "".join([f"<th>{html.escape(c)}</th>" for c in cols]) + "</tr></thead><tbody>" + "".join(
                    [
                        "<tr>" + "".join([f"<td>{html.escape(str(getattr(r, c, '')))}</td>" for c in cols]) + "</tr>"
                        for r in label_scan_df.itertuples(index=False)
                    ]
                ) + "</tbody></table></div>"
        except Exception:
            pass

    cluster_gap_html = "<div class='empty-note'>No cluster-gap report available.</div>"
    cluster_gap_path = os.path.join(clustering_dir, "cluster_gap_report.parquet")
    if os.path.exists(cluster_gap_path):
        try:
            gap_df = pd.read_parquet(cluster_gap_path)
            if not gap_df.empty:
                preview = gap_df.head(20)
                cols = preview.columns.tolist()[:8]
                cluster_gap_html = "<div class='metrics-table-wrap'><table><thead><tr>" + "".join([f"<th>{html.escape(c)}</th>" for c in cols]) + "</tr></thead><tbody>" + "".join(
                    [
                        "<tr>" + "".join([f"<td>{html.escape(str(getattr(r, c, '')))}</td>" for c in cols]) + "</tr>"
                        for r in preview.itertuples(index=False)
                    ]
                ) + "</tbody></table></div>"
        except Exception:
            pass

    interpretability_html = "<div class='empty-note'>No interpretability report available.</div>"
    interpret_path = os.path.join(clustering_dir, "interpretability_report.json")
    if os.path.exists(interpret_path):
        try:
            with open(interpret_path, "r") as f:
                interpret = json.load(f)
            cluster_passes = interpret.get("cluster_passes") or []
            blocks = []
            for p in cluster_passes[:4]:
                p_name = p.get("cluster_pass", "pass")
                p_k = p.get("n_clusters", "")
                top_feats = []
                for item in p.get("pca_top_features") or []:
                    top_feats.append(f"{item.get('pc')}: {', '.join((item.get('top_features') or [])[:5])}")
                blocks.append(
                    f"<div class='best-method-box'><p><strong>{html.escape(str(p_name))}</strong> (k={html.escape(str(p_k))})</p><p>{'<br/>'.join([html.escape(x) for x in top_feats])}</p></div>"
                )
            if blocks:
                interpretability_html = "".join(blocks)
        except Exception:
            pass

    algorithmic_summary_html = "<div class='empty-note'>No algorithmic segment summaries available.</div>"
    try:
        algo_summary_rows = []
        by_dep_dir = os.path.join(segments_dir, "by_deployment")
        if os.path.isdir(by_dep_dir):
            for name in sorted(os.listdir(by_dep_dir)):
                if not name.endswith("__algorithmic_segments_summary.json"):
                    continue
                p = os.path.join(by_dep_dir, name)
                with open(p, "r") as f:
                    row = json.load(f)
                algo_summary_rows.append(row)
        if algo_summary_rows:
            algo_df = pd.DataFrame(algo_summary_rows)
            cols = [c for c in ["dataset_id", "deployment_id", "n_segments", "n_surface_sleep_segments", "n_long_flat_segments", "duration_hours"] if c in algo_df.columns]
            algorithmic_summary_html = "<div class='metrics-table-wrap'><table><thead><tr>" + "".join([f"<th>{html.escape(c)}</th>" for c in cols]) + "</tr></thead><tbody>" + "".join(
                [
                    "<tr>" + "".join([f"<td>{html.escape(str(getattr(r, c, '')))}</td>" for c in cols]) + "</tr>"
                    for r in algo_df.itertuples(index=False)
                ]
            ) + "</tbody></table></div>"
    except Exception:
        pass
    algorithmic_context_html, algorithmic_examples_html = _build_algorithmic_context_report(
        ctx,
        out_dir,
        segments_dir,
        asset_prefix=asset_prefix,
    )

    legacy_plots_html = []
    legacy_files = [
        ("Interactive cluster overlay notebook", "A_interactive_deployment_overlay.ipynb"),
    ]
    for label, filename in legacy_files:
        full = os.path.join(out_dir, filename)
        if os.path.exists(full):
            href = posixpath.join(asset_prefix, filename) if asset_prefix else filename
            legacy_plots_html.append(f"<li><a href=\"{html.escape(href)}\" target=\"_blank\" rel=\"noopener noreferrer\">{html.escape(label)}</a></li>")

    correlogram_fragment = ""
    corr_path = os.path.join(ctx.output_root, "features", "feature_correlation_matrix.parquet")
    if os.path.exists(corr_path):
        try:
            corr_df = pd.read_parquet(corr_path)
            if not corr_df.empty:
                if "feature" in corr_df.columns:
                    corr_df = corr_df.set_index("feature")
                corr_df.index = corr_df.index.astype(str)
                corr_df.columns = corr_df.columns.astype(str)
                top_corr_features = corr_df.columns.tolist()[:50]
                corr_view = corr_df.loc[top_corr_features, top_corr_features]
                fig_corr = px.imshow(
                    corr_view,
                    x=top_corr_features,
                    y=top_corr_features,
                    color_continuous_scale="Viridis",
                    zmin=0,
                    zmax=1,
                    aspect="auto",
                    title="Feature Correlogram (top 50 filtered features)",
                )
                _style_report_plot(fig_corr, height=820)
                correlogram_fragment = _plotly_fragment(fig_corr)
        except Exception:
            correlogram_fragment = ""

    feature_cluster_source = cdf_key.copy()
    if "cluster_pass" in feature_cluster_source.columns:
        cluster_pass_series = feature_cluster_source["cluster_pass"].astype(str)
        if (cluster_pass_series == "k5").any():
            feature_cluster_source = feature_cluster_source.loc[cluster_pass_series == "k5"].copy()
    observed_windows = spdf[["window_key", "observed_label"]].drop_duplicates(subset=["window_key"]).dropna(subset=["observed_label"]).copy()
    rf_feature_plot_sources: Dict[str, pd.DataFrame] = {}

    feature_overlap_note = ""
    signal_overlap_summary_html = "<div class='empty-note'>No top-signal overlap review available.</div>"
    signal_overlap_blocks: List[str] = []
    signal_overlap_options: List[str] = []
    signal_overlap_default = None
    signal_overlap_desc = {}
    signal_overlap_check_html = "<div class='empty-note'>No signal-level z-score checks available.</div>"
    feature_overlap_summary_html = "<div class='empty-note'>No top-feature overlap review available.</div>"
    feature_overlap_blocks: List[str] = []
    feature_overlap_options: List[str] = []
    feature_overlap_default = None
    feature_overlap_desc = {}
    feature_overlap_check_html = "<div class='empty-note'>No feature-level z-score checks available.</div>"

    norm_level = str(ctx.run_cfg.get("normalization_level", "deployment")).lower()
    norm_group_col = "deployment_id" if norm_level == "deployment" else "dataset_id"
    dataset_count = int(spdf["dataset_id"].astype(str).nunique()) if "dataset_id" in spdf.columns else 0
    feature_overlap_note = (
        f"This run uses <strong>{html.escape(norm_level)}</strong>-level z-scoring, so the QA check below tests whether each "
        f"<code>{html.escape(norm_group_col)}</code> lands near mean 0 and std 1 after normalization. "
        f"The overlap plots are faceted by dataset; current scope includes {dataset_count} dataset{'s' if dataset_count != 1 else ''}. "
        "Facet titles include <strong>[source]</strong> or <strong>[target]</strong> so transfer comparisons stay explicit."
    )
    if not fi_df.empty:
        selected_feature_rows = fi_df.copy()
        selected_feature_rows["feature"] = selected_feature_rows["feature"].astype(str)
        feature_candidates = selected_feature_rows["feature"].tolist()
        signal_rank_df = (
            selected_feature_rows.assign(signal=selected_feature_rows["feature"].map(_feature_source_channel))
            .sort_values(["importance", "feature"], ascending=[False, True])
        )
        signal_summary_df = (
            signal_rank_df.groupby("signal", as_index=False)
            .agg(
                total_importance=("importance", "sum"),
                representative_feature=("feature", "first"),
                representative_importance=("importance", "first"),
            )
            .sort_values(["total_importance", "signal"], ascending=[False, True])
        )

        top_feature_items = selected_feature_rows.head(5).to_dict(orient="records")
        top_signal_items = signal_summary_df.head(5).to_dict(orient="records")
        needed_features = sorted(
            {
                str(row.get("feature"))
                for row in top_feature_items
                if str(row.get("feature") or "").strip()
            }
            | {
                str(row.get("representative_feature"))
                for row in top_signal_items
                if str(row.get("representative_feature") or "").strip()
            }
        )
        norm_cols = ["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds"] + needed_features
        norm_path = os.path.join(features_dir, "features_filtered.parquet")
        norm_feat_df = _read_feature_subset(norm_path, columns=norm_cols)
        raw_frames = []
        raw_cols = ["dataset_id", "deployment_id", "window_start", "window_end", "window_seconds"] + needed_features
        for item in ctx.scope:
            raw_path, _raw_idx_path = _feature_deployment_paths(ctx, item["dataset_id"], item["deployment_id"])
            raw_chunk = _read_feature_subset(raw_path, columns=raw_cols)
            if raw_chunk is not None and not raw_chunk.empty:
                raw_frames.append(raw_chunk)
        raw_feat_df = pd.concat(raw_frames, ignore_index=True, sort=False) if raw_frames else pd.DataFrame()
        if not raw_feat_df.empty and not norm_feat_df.empty:
            for frame in [raw_feat_df, norm_feat_df]:
                frame["window_start"] = pd.to_datetime(frame["window_start"], errors="coerce")
                frame["window_end"] = pd.to_datetime(frame["window_end"], errors="coerce")
                frame["window_key"] = _build_window_key_series(frame)
            obs_cols = ["window_key", "dataset_id", "deployment_id", "observed_label"]
            # Defragment before repeated merges/slicing in report assembly.
            spdf = spdf.copy()
            observed_feature_windows = (
                spdf[obs_cols]
                .drop_duplicates(subset=["window_key"])
                .dropna(subset=["observed_label"])
                .copy()
            )
            raw_feature_source = raw_feat_df.merge(observed_feature_windows, on="window_key", how="inner", suffixes=("", "_obs"))
            norm_feature_source = norm_feat_df.merge(observed_feature_windows, on="window_key", how="inner", suffixes=("", "_obs"))
            for frame in [raw_feature_source, norm_feature_source]:
                for col in ["dataset_id_obs", "deployment_id_obs"]:
                    if col in frame.columns:
                        frame.drop(columns=[col], inplace=True)

            signal_summary_rows = []
            signal_check_rows = []
            for row in top_signal_items:
                signal_name = str(row.get("signal") or "").strip()
                representative_feature = str(row.get("representative_feature") or "").strip()
                if not signal_name or not representative_feature:
                    continue
                if representative_feature not in raw_feature_source.columns or representative_feature not in norm_feature_source.columns:
                    continue
                raw_sub = raw_feature_source[["dataset_id", "deployment_id", "observed_label", representative_feature]].copy()
                raw_sub = raw_sub.rename(columns={representative_feature: "feature_value"}).dropna(subset=["observed_label", "feature_value"])
                norm_sub = norm_feature_source[["dataset_id", "deployment_id", "observed_label", representative_feature]].copy()
                norm_sub = norm_sub.rename(columns={representative_feature: "feature_value"}).dropna(subset=["observed_label", "feature_value"])
                if raw_sub.empty or norm_sub.empty:
                    continue
                raw_sub["value_mode"] = "Raw"
                norm_sub["value_mode"] = "Z-scored"
                long_df = pd.concat([raw_sub, norm_sub], ignore_index=True, sort=False)
                fig_signal = _build_dataset_facet_feature_figure(
                    long_df,
                    title=f"Top signal overlap by dataset: {_feature_display_name(signal_name)}",
                    y_label=representative_feature,
                    behavior_order=behavior_order,
                    behavior_color_map=behavior_color_map,
                    source_dataset_ids=_parse_supervised_transfer_cfg(ctx.run_cfg).get("source_dataset_ids"),
                )
                if fig_signal is None:
                    continue
                signal_id = signal_name
                signal_overlap_desc[signal_id] = (
                    f"Signal {_feature_display_name(signal_name)} is represented here by "
                    f"{representative_feature} (importance {float(row.get('representative_importance', 0.0)):.4f}; "
                    f"total signal importance {float(row.get('total_importance', 0.0)):.4f}). "
                    f"{_describe_feature_name(representative_feature)}"
                )
                signal_overlap_blocks.append(
                    f"<div class='feature-plot-panel' data-feature-group='signal-overlap' data-feature='{html.escape(signal_id)}' "
                    f"style='display:{'' if signal_overlap_default is None else 'none'}'>{_plotly_fragment(fig_signal)}</div>"
                )
                signal_overlap_options.append(
                    f"<option value=\"{html.escape(signal_id)}\">{html.escape(_feature_display_name(signal_name))}</option>"
                )
                signal_summary_rows.append(
                    {
                        "signal": _feature_display_name(signal_name),
                        "total_importance": f"{float(row.get('total_importance', 0.0)):.4f}",
                        "representative_feature": representative_feature,
                    }
                )
                signal_check_rows.append(
                    _summarize_zscore_check(
                        raw_sub,
                        norm_sub,
                        item_name=_feature_display_name(signal_name),
                        representative_feature=representative_feature,
                        review_type="Signal",
                        norm_group_col=norm_group_col,
                    )
                )
                if signal_overlap_default is None:
                    signal_overlap_default = signal_id

            feature_summary_rows = []
            feature_check_rows = []
            for row in top_feature_items:
                feature_name = str(row.get("feature") or "").strip()
                if not feature_name:
                    continue
                if feature_name not in raw_feature_source.columns or feature_name not in norm_feature_source.columns:
                    continue
                raw_sub = raw_feature_source[["dataset_id", "deployment_id", "observed_label", feature_name]].copy()
                raw_sub = raw_sub.rename(columns={feature_name: "feature_value"}).dropna(subset=["observed_label", "feature_value"])
                norm_sub = norm_feature_source[["dataset_id", "deployment_id", "observed_label", feature_name]].copy()
                norm_sub = norm_sub.rename(columns={feature_name: "feature_value"}).dropna(subset=["observed_label", "feature_value"])
                if raw_sub.empty or norm_sub.empty:
                    continue
                raw_sub["value_mode"] = "Raw"
                norm_sub["value_mode"] = "Z-scored"
                long_df = pd.concat([raw_sub, norm_sub], ignore_index=True, sort=False)
                fig_feature_overlap = _build_dataset_facet_feature_figure(
                    long_df,
                    title=f"Top feature overlap by dataset: {feature_name}",
                    y_label=feature_name,
                    behavior_order=behavior_order,
                    behavior_color_map=behavior_color_map,
                    source_dataset_ids=_parse_supervised_transfer_cfg(ctx.run_cfg).get("source_dataset_ids"),
                )
                if fig_feature_overlap is None:
                    continue
                feature_overlap_desc[feature_name] = _describe_feature_name(feature_name)
                feature_overlap_blocks.append(
                    f"<div class='feature-plot-panel' data-feature-group='feature-overlap' data-feature='{html.escape(feature_name)}' "
                    f"style='display:{'' if feature_overlap_default is None else 'none'}'>{_plotly_fragment(fig_feature_overlap)}</div>"
                )
                feature_overlap_options.append(f"<option value=\"{html.escape(feature_name)}\">{html.escape(feature_name)}</option>")
                feature_summary_rows.append(
                    {
                        "feature": feature_name,
                        "importance": f"{float(row.get('importance', 0.0)):.4f}",
                        "how_calculated": _describe_feature_name(feature_name),
                    }
                )
                feature_check_rows.append(
                    _summarize_zscore_check(
                        raw_sub,
                        norm_sub,
                        item_name=feature_name,
                        representative_feature=feature_name,
                        review_type="Feature",
                        norm_group_col=norm_group_col,
                    )
                )
                if feature_overlap_default is None:
                    feature_overlap_default = feature_name

            signal_overlap_summary_html = _render_metrics_table_html(
                signal_summary_rows,
                [
                    ("signal", "Signal"),
                    ("total_importance", "Total importance"),
                    ("representative_feature", "Representative feature"),
                ],
            )
            signal_overlap_check_html = _render_metrics_table_html(
                signal_check_rows,
                [
                    ("item_name", "Signal"),
                    ("representative_feature", "Representative feature"),
                    ("normalization_groups", f"{norm_group_col} groups"),
                    ("raw_mean_range", "Raw mean range"),
                    ("raw_std_range", "Raw std range"),
                    ("z_max_abs_mean", "Max |z mean|"),
                    ("z_std_range", "Z std range"),
                    ("status", "Check"),
                ],
            )
            feature_overlap_summary_html = _render_metrics_table_html(
                feature_summary_rows,
                [
                    ("feature", "Feature"),
                    ("importance", "Importance"),
                    ("how_calculated", "How calculated"),
                ],
            )
            feature_overlap_check_html = _render_metrics_table_html(
                feature_check_rows,
                [
                    ("item_name", "Feature"),
                    ("normalization_groups", f"{norm_group_col} groups"),
                    ("raw_mean_range", "Raw mean range"),
                    ("raw_std_range", "Raw std range"),
                    ("z_max_abs_mean", "Max |z mean|"),
                    ("z_std_range", "Z std range"),
                    ("status", "Check"),
                ],
            )

            for feature_name in selected_feature_rows["feature"].astype(str).tolist():
                if feature_name not in raw_feature_source.columns or feature_name not in norm_feature_source.columns:
                    continue
                raw_sub = raw_feature_source[["dataset_id", "deployment_id", "observed_label", feature_name]].copy()
                raw_sub = raw_sub.rename(columns={feature_name: "feature_value"}).dropna(subset=["observed_label", "feature_value"])
                norm_sub = norm_feature_source[["dataset_id", "deployment_id", "observed_label", feature_name]].copy()
                norm_sub = norm_sub.rename(columns={feature_name: "feature_value"}).dropna(subset=["observed_label", "feature_value"])
                if raw_sub.empty and norm_sub.empty:
                    continue
                if not raw_sub.empty:
                    raw_sub["value_mode"] = "Raw"
                if not norm_sub.empty:
                    norm_sub["value_mode"] = "Z-scored"
                rf_feature_plot_sources[feature_name] = pd.concat(
                    [frame for frame in [raw_sub, norm_sub] if not frame.empty],
                    ignore_index=True,
                    sort=False,
                )

    pca_top_table_html = "<div class='empty-note'>No PCA loading file available.</div>"
    pca_violin_blocks = []
    pca_feature_options = []
    pca_feature_default = None
    pca_load_path = os.path.join(ctx.output_root, "clustering", "pca_loadings_k5.parquet")
    if os.path.exists(pca_load_path):
        try:
            pca_load_df = pd.read_parquet(pca_load_path)
            pca_pc_cols = [c for c in pca_load_df.columns if re.fullmatch(r"PC\d+", str(c))]
            if pca_pc_cols:
                pca_ranked = pca_load_df.copy()
                pca_ranked["max_abs_loading"] = pca_ranked[pca_pc_cols].abs().max(axis=1)
                pca_ranked["dominant_pc"] = pca_ranked[pca_pc_cols].abs().idxmax(axis=1)
                pca_top = pca_ranked.sort_values(["max_abs_loading", "feature"], ascending=[False, True]).head(10).copy()
                pca_top_table_html = "<div class='metrics-table-wrap'><table><thead><tr><th>Feature</th><th>Dominant PC</th><th>Max |loading|</th></tr></thead><tbody>" + "".join(
                    [
                        f"<tr><td>{html.escape(str(r.feature))}</td><td>{html.escape(str(r.dominant_pc))}</td><td>{float(r.max_abs_loading):.3f}</td></tr>"
                        for r in pca_top.itertuples(index=False)
                    ]
                ) + "</tbody></table></div>"
                for i, feature_name in enumerate(pca_top["feature"].astype(str).tolist()):
                    if feature_name not in feature_cluster_source.columns:
                        continue
                    sub = feature_cluster_source[["cluster_rank", feature_name, "deployment_id"]].copy()
                    sub = sub.rename(columns={feature_name: "feature_value"}).dropna(subset=["cluster_rank", "feature_value"])
                    if sub.empty:
                        continue
                    sub["cluster_rank"] = pd.to_numeric(sub["cluster_rank"], errors="coerce").astype("Int64")
                    fig_v = px.violin(
                        sub,
                        x=sub["cluster_rank"].astype(str),
                        y="feature_value",
                        color=sub["cluster_rank"].astype(str),
                        box=True,
                        points=False,
                        title=f"PCA-loaded feature by cluster: {feature_name}",
                        labels={"x": "Cluster", "feature_value": feature_name, "color": "Cluster"},
                    )
                    fig_v.update_layout(showlegend=False)
                    _style_report_plot(fig_v, height=420)
                    pca_violin_blocks.append(
                        f"<div class='feature-plot-panel' data-feature-group='pca' data-feature='{html.escape(feature_name)}' style='display:{'' if pca_feature_default is None else 'none'}'>{_plotly_fragment(fig_v)}</div>"
                    )
                    pca_feature_options.append(f"<option value=\"{html.escape(feature_name)}\">{html.escape(feature_name)}</option>")
                    if pca_feature_default is None:
                        pca_feature_default = feature_name
        except Exception:
            pass

    rf_top_table_html = "<div class='empty-note'>No RF feature-importance file available.</div>"
    rf_violin_blocks = []
    rf_feature_options = []
    rf_feature_default = None
    rf_feature_descriptions = {}
    if not fi_df.empty:
        rf_top = fi_df.head(10).copy()
        rf_top_table_html = "<div class='metrics-table-wrap'><table><thead><tr><th>Feature</th><th>Importance</th><th>How it is calculated</th></tr></thead><tbody>" + "".join(
            [
                f"<tr><td>{html.escape(str(r.feature))}</td><td>{float(r.importance):.4f}</td><td>{html.escape(_describe_feature_name(str(r.feature)))}</td></tr>"
                for r in rf_top.itertuples(index=False)
            ]
        ) + "</tbody></table></div>"
        for feature_name in rf_top["feature"].astype(str).tolist():
            long_df = rf_feature_plot_sources.get(feature_name)
            if long_df is None or long_df.empty:
                continue
            rf_feature_descriptions[feature_name] = _describe_feature_name(feature_name)
            fig_v = _build_dataset_facet_feature_figure(
                long_df,
                title=f"RF-important feature by supervised class: {feature_name}",
                y_label=feature_name,
                behavior_order=behavior_order,
                behavior_color_map=behavior_color_map,
                source_dataset_ids=_parse_supervised_transfer_cfg(ctx.run_cfg).get("source_dataset_ids"),
            )
            if fig_v is None:
                continue
            rf_violin_blocks.append(
                f"<div class='feature-plot-panel' data-feature-group='rf' data-feature='{html.escape(feature_name)}' style='display:{'' if rf_feature_default is None else 'none'}'>{_plotly_fragment(fig_v)}</div>"
            )
            rf_feature_options.append(f"<option value=\"{html.escape(feature_name)}\">{html.escape(feature_name)}</option>")
            if rf_feature_default is None:
                rf_feature_default = feature_name
    supervised_cfg = dict(ctx.run_cfg.get("supervised") or {})
    sankey_label_group_map = _build_supervised_label_group_map(supervised_cfg)
    configured_label_order = [
        str(label).strip()
        for label in list((supervised_cfg.get("label_groups") or {}).keys())
        if str(label).strip()
    ]
    configured_label_set = set(configured_label_order)
    sankey_debug = {
        "configured_label_order": configured_label_order,
        "configured_label_set": sorted(configured_label_set),
        "spdf_rows": int(len(spdf)),
        "spdf_unique_window_keys": int(spdf["window_key"].nunique()) if "window_key" in spdf.columns else 0,
        "cdf_rows": int(len(cdf_key)),
        "cdf_unique_window_keys": int(cdf_key["window_key"].nunique()) if "window_key" in cdf_key.columns else 0,
        "observed_windows_before_grouping_rows": int(len(observed_windows)),
        "observed_windows_before_grouping_labels": (
            observed_windows["observed_label"].astype(str).value_counts().to_dict()
            if not observed_windows.empty and "observed_label" in observed_windows.columns
            else {}
        ),
        "fragments_written": [],
        "errors": [],
    }
    if not observed_windows.empty:
        observed_windows = observed_windows.copy()
        observed_windows["observed_label"] = observed_windows["observed_label"].map(
            lambda x: _map_label_to_supervised_group(x, sankey_label_group_map)
        )
        if configured_label_set:
            observed_windows = observed_windows[
                observed_windows["observed_label"].astype(str).isin(configured_label_set)
            ].copy()
    sankey_debug["observed_windows_after_grouping_rows"] = int(len(observed_windows))
    sankey_debug["observed_windows_after_grouping_labels"] = (
        observed_windows["observed_label"].astype(str).value_counts().to_dict()
        if not observed_windows.empty and "observed_label" in observed_windows.columns
        else {}
    )
    observed_cluster = cdf_key.merge(observed_windows, on="window_key", how="inner")
    sankey_debug["observed_cluster_rows"] = int(len(observed_cluster))
    sankey_debug["observed_cluster_unique_window_keys"] = int(observed_cluster["window_key"].nunique()) if not observed_cluster.empty and "window_key" in observed_cluster.columns else 0
    sankey_debug["observed_cluster_cluster_pass_counts"] = (
        observed_cluster["cluster_pass"].astype(str).value_counts().to_dict()
        if not observed_cluster.empty and "cluster_pass" in observed_cluster.columns
        else {}
    )

    cluster_sankey_fragments = []
    if not observed_cluster.empty and "cluster_rank" in observed_cluster.columns:
        k5_df = _select_cluster_pass_rows(observed_cluster, "k5", n_clusters=5)
        sankey_debug["k5_rows"] = int(len(k5_df))
        if not k5_df.empty:
            try:
                flow_k5 = (
                    k5_df.groupby(["observed_label", "cluster_rank"], as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                )
                sankey_debug["k5_flow_rows"] = int(len(flow_k5))
                sankey_debug["k5_flow_preview"] = flow_k5.head(10).astype(object).to_dict("records")
                if configured_label_order:
                    flow_k5["observed_label"] = pd.Categorical(
                        flow_k5["observed_label"].astype(str),
                        categories=configured_label_order,
                        ordered=True,
                    )
                    flow_k5 = flow_k5.sort_values(["observed_label", "cluster_rank"]).reset_index(drop=True)
                flow_k5["cluster_node"] = flow_k5["cluster_rank"].astype(int).map(lambda x: f"k5: cluster {x}")
                fig_k5 = _build_flow_sankey(flow_k5, "observed_label", "cluster_node", "count", "Observed Labels to K5 Clusters")
                sankey_debug["k5_fig_is_none"] = bool(fig_k5 is None)
                if fig_k5 is not None:
                    cluster_sankey_fragments.append(("k5", _plotly_fragment(fig_k5)))
                    sankey_debug["fragments_written"].append("k5")
            except Exception as exc:
                sankey_debug["errors"].append(
                    {
                        "stage": "k5",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        k2_actual = _select_cluster_pass_rows(observed_cluster, "k2", n_clusters=2)
        sankey_debug["k2_rows"] = int(len(k2_actual))
        if not k2_actual.empty:
            try:
                flow_k2 = (
                    k2_actual.groupby(["observed_label", "cluster_rank"], as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                )
                sankey_debug["k2_flow_rows"] = int(len(flow_k2))
                sankey_debug["k2_flow_preview"] = flow_k2.head(10).astype(object).to_dict("records")
                if configured_label_order:
                    flow_k2["observed_label"] = pd.Categorical(
                        flow_k2["observed_label"].astype(str),
                        categories=configured_label_order,
                        ordered=True,
                    )
                    flow_k2 = flow_k2.sort_values(["observed_label", "cluster_rank"]).reset_index(drop=True)
                flow_k2["cluster_node"] = flow_k2["cluster_rank"].astype(int).map(lambda x: f"k2: cluster {x}")
                fig_k2 = _build_flow_sankey(flow_k2, "observed_label", "cluster_node", "count", "Observed Labels to K2 Clusters")
                sankey_debug["k2_fig_is_none"] = bool(fig_k2 is None)
                if fig_k2 is not None:
                    cluster_sankey_fragments.append(("k2", _plotly_fragment(fig_k2)))
                    sankey_debug["fragments_written"].append("k2")
            except Exception as exc:
                sankey_debug["errors"].append(
                    {
                        "stage": "k2",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        elif SKLEARN_AVAILABLE:
            feat_cols = [
                c for c in cdf_key.columns
                if c not in {"dataset_id", "deployment_id", "window_start", "window_end", "window_seconds", "window_key"}
                and pd.api.types.is_numeric_dtype(cdf_key[c])
                and not str(c).startswith("__")
                and not str(c).startswith("cluster_")
            ]
            if feat_cols:
                try:
                    k2_input = cdf_key[["window_key"] + feat_cols].drop_duplicates(subset=["window_key"]).copy()
                    x_k2 = k2_input[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    k2_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(x_k2)
                    k2_input["k2_review_cluster"] = pd.Series(k2_labels).astype(int)
                    k2_merge = observed_windows.merge(k2_input[["window_key", "k2_review_cluster"]], on="window_key", how="inner")
                    if not k2_merge.empty:
                        flow_k2d = (
                            k2_merge.groupby(["observed_label", "k2_review_cluster"], as_index=False)
                            .size()
                            .rename(columns={"size": "count"})
                        )
                        if configured_label_order:
                            flow_k2d["observed_label"] = pd.Categorical(
                                flow_k2d["observed_label"].astype(str),
                                categories=configured_label_order,
                                ordered=True,
                            )
                            flow_k2d = flow_k2d.sort_values(["observed_label", "k2_review_cluster"]).reset_index(drop=True)
                        flow_k2d["cluster_node"] = flow_k2d["k2_review_cluster"].astype(int).map(lambda x: f"k2 review: cluster {x}")
                        fig_k2d = _build_flow_sankey(
                            flow_k2d,
                            "observed_label",
                            "cluster_node",
                            "count",
                            "Observed Supervised Labels to K2 Review Clusters",
                        )
                        if fig_k2d is not None:
                            cluster_sankey_fragments.append(("k2_review", _plotly_fragment(fig_k2d)))
                            sankey_debug["fragments_written"].append("k2_review")
                except Exception:
                    sankey_debug["errors"].append(
                        {
                            "stage": "k2_review",
                            "error": traceback.format_exc().splitlines()[-1] if traceback.format_exc().splitlines() else "unknown",
                            "traceback": traceback.format_exc(),
                        }
                    )

    sankey_debug["final_fragment_count"] = int(len(cluster_sankey_fragments))
    try:
        with open(os.path.join(supervised_dir, "cluster_sankey_debug.json"), "w", encoding="utf-8") as f:
            json.dump(sankey_debug, f, indent=2, default=str)
    except Exception:
        pass

    actogram_cards = []
    for path in sorted(supervised_actogram_paths or []):
        dep_slug = Path(path).stem.replace("G_supervised_actogram_comparison_", "")
        rel_path = posixpath.join(asset_prefix, os.path.basename(path)) if asset_prefix else os.path.basename(path)
        actogram_cards.append(
            f"""
            <section class="card deployment-panel" data-deployment="{html.escape(dep_slug)}">
              <div class="section-head">
                <h3>{html.escape(dep_slug)}</h3>
                <p>Observed labels are shown on the left. Predicted methods are shown in adjacent actogram columns and colored by confusion outcome.</p>
              </div>
              <img src="{html.escape(rel_path)}" alt="Supervised actogram comparison for {html.escape(dep_slug)}" class="actogram-image"/>
            </section>
            """
        )
    if primary_daily_actogram_path and os.path.exists(primary_daily_actogram_path):
        rel_path = posixpath.join(asset_prefix, os.path.basename(primary_daily_actogram_path)) if asset_prefix else os.path.basename(primary_daily_actogram_path)
        actogram_cards.append(
            f"""
            <section class="card deployment-panel" data-deployment="primary_method_daily">
              <div class="section-head">
                <h3>Primary method daily actogram</h3>
                <p>Daily actogram for the config-resolved primary method.</p>
              </div>
              <img src="{html.escape(rel_path)}" alt="Primary method daily actogram" class="actogram-image"/>
            </section>
            """
        )
    if primary_continuous_actogram_path and os.path.exists(primary_continuous_actogram_path):
        rel_path = posixpath.join(asset_prefix, os.path.basename(primary_continuous_actogram_path)) if asset_prefix else os.path.basename(primary_continuous_actogram_path)
        actogram_cards.append(
            f"""
            <section class="card deployment-panel" data-deployment="primary_method_continuous">
              <div class="section-head">
                <h3>Primary method continuous actogram</h3>
                <p>Continuous timeline actogram with sunrise/sunset context for the primary method.</p>
              </div>
              <img src="{html.escape(rel_path)}" alt="Primary method continuous actogram" class="actogram-image"/>
            </section>
            """
        )
    if primary_event_timing_actogram_path and os.path.exists(primary_event_timing_actogram_path):
        rel_path = posixpath.join(asset_prefix, os.path.basename(primary_event_timing_actogram_path)) if asset_prefix else os.path.basename(primary_event_timing_actogram_path)
        actogram_cards.append(
            f"""
            <section class="card deployment-panel" data-deployment="primary_method_events">
              <div class="section-head">
                <h3>Primary method event timing overlay</h3>
                <p>Events of interest are mapped onto the continuous day/night timeline to show within-day timing.</p>
              </div>
              <img src="{html.escape(rel_path)}" alt="Primary method event timing overlay actogram" class="actogram-image"/>
            </section>
            """
        )

    variant_selector_options = ["<option value=\"all\">All methods</option>"] + [
        f"<option value=\"{html.escape(str(v.variant_id))}\">{html.escape(str(v.variant_label))}</option>"
        for v in budget_meta
        .sort_values(["variant_rank", "variant_label"])
        .itertuples(index=False)
    ]
    deployment_selector_options = ["<option value=\"all\">All deployments</option>"] + [
        f"<option value=\"{html.escape(Path(path).stem.replace('G_supervised_actogram_comparison_', ''))}\">{html.escape(Path(path).stem.replace('G_supervised_actogram_comparison_', ''))}</option>"
        for path in sorted(supervised_actogram_paths or [])
    ]
    if primary_daily_actogram_path and os.path.exists(primary_daily_actogram_path):
        deployment_selector_options.append("<option value=\"primary_method_daily\">Primary method daily</option>")
    if primary_continuous_actogram_path and os.path.exists(primary_continuous_actogram_path):
        deployment_selector_options.append("<option value=\"primary_method_continuous\">Primary method continuous</option>")
    if primary_event_timing_actogram_path and os.path.exists(primary_event_timing_actogram_path):
        deployment_selector_options.append("<option value=\"primary_method_events\">Primary method events</option>")

    method_sections = []
    method_meta = (
        pd.concat(
            [
                pd.DataFrame([{"variant_id": "observed_reference", "variant_label": "Observed Labels", "variant_type": "reference", "variant_rank": -1}]),
                variant_meta,
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["variant_id"])
        .sort_values(["variant_rank", "variant_label"])
    )
    for row in budget_meta.itertuples(index=False):
        daily_html = daily_figs.get(row.variant_id, "<div class='empty-note'>No daily budget available.</div>")
        hourly_html = hourly_figs.get(row.variant_id, "<div class='empty-note'>No hourly budget available.</div>")
        method_sections.append(
            f"""
            <section class="card variant-panel" data-variant="{html.escape(str(row.variant_id))}">
              <div class="section-head">
                <h3>{html.escape(str(row.variant_label))}</h3>
                <p>Daily budgets are normalized to 24 h/day. Any remainder is shown as <strong>{html.escape(remainder_label)}</strong> when labels or cluster assignments do not cover the full day.</p>
              </div>
              <div class="two-col">
                <div class="plot-shell">{daily_html}</div>
                <div class="plot-shell">{hourly_html}</div>
              </div>
            </section>
            """
        )

    metric_rows_html = ""
    if not metrics_df.empty:
        metrics_table = metrics_df.copy()
        metrics_table["value"] = metrics_table["value"].map(lambda x: f"{float(x):.3f}")
        metric_rows_html = "\n".join(
            [
                "<tr>"
                f"<td>{html.escape(str(r.variant_label))}</td>"
                f"<td>{html.escape(str(r.label))}</td>"
                f"<td>{html.escape(str(r.metric))}</td>"
                f"<td>{html.escape(str(r.value))}</td>"
                "</tr>"
                for r in metrics_table.sort_values(["variant_rank", "label", "metric"]).itertuples(index=False)
            ]
        )

    summary_cards = [
        ("Labeled windows", f"{int(summary.get('n_rows_labeled', observed_windows['window_key'].nunique())):,}"),
        ("Total windows", f"{int(summary.get('n_rows_total', spdf['window_key'].nunique())):,}"),
        ("Deployments", f"{spdf['deployment_id'].nunique():,}"),
        ("Methods compared", f"{method_meta['variant_id'].nunique():,}"),
    ]
    summary_cards_html = "\n".join(
        [
            f"<div class='stat-card'><div class='stat-label'>{html.escape(label)}</div><div class='stat-value'>{html.escape(value)}</div></div>"
            for label, value in summary_cards
        ]
    )

    sankey_html = "\n".join(
        [
            f"<section class='card'><div class='section-head'><h3>{html.escape(kind.replace('_', ' ').upper())}</h3></div><div class='plot-shell'>{frag}</div></section>"
            for kind, frag in cluster_sankey_fragments
        ]
    ) or "<div class='empty-note'>No cluster-to-supervised Sankey views were available for this run.</div>"
    rf_feature_desc_json = json.dumps(rf_feature_descriptions)
    signal_overlap_desc_json = json.dumps(signal_overlap_desc)
    feature_overlap_desc_json = json.dumps(feature_overlap_desc)
    config_sidebar_html = _render_run_config_sidebar(ctx.run_cfg)
    interactive_manifest_path = os.path.join(out_dir, "interactive_review_manifest.json")
    interactive_embed_html = "<div class='empty-note'>No embedded interactive review panes available.</div>"
    interactive_manifest_json = "[]"
    if os.path.exists(interactive_manifest_path):
        try:
            with open(interactive_manifest_path, "r", encoding="utf-8") as f:
                interactive_manifest = json.load(f)
            if interactive_manifest:
                if asset_prefix:
                    for row in interactive_manifest:
                        rel = str(row.get("relative_path") or "")
                        row["relative_path"] = posixpath.join(asset_prefix, rel.replace("\\", "/"))
                deployment_options = []
                seen_deployments = set()
                for row in interactive_manifest:
                    dep_key = str(row.get("deployment_key"))
                    if dep_key in seen_deployments:
                        continue
                    seen_deployments.add(dep_key)
                    deployment_options.append(f"<option value=\"{html.escape(dep_key)}\">{html.escape(dep_key)}</option>")
                interactive_embed_html = f"""
                <div class="controls">
                  <div class="control">
                    <label for="interactive-deployment-select">Deployment</label>
                    <select id="interactive-deployment-select">{''.join(deployment_options)}</select>
                  </div>
                  <div class="control">
                    <label for="interactive-method-select">Segmentation method</label>
                    <select id="interactive-method-select"></select>
                  </div>
                </div>
                <iframe id="interactive-review-frame" class="interactive-frame" src="" loading="lazy"></iframe>
                """
                interactive_manifest_json = json.dumps(interactive_manifest)
        except Exception:
            pass

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Supervised Review Report: {html.escape(ctx.run_name)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --blue-extra-dark: #041827;
      --blue-dark: #0e3551;
      --blue: #092a42;
      --blue-medium: #286591;
      --blue-light: #73a9c4;
      --white: #ffffff;
      --black: #333333;
      --muted: #6b7a88;
      --bg-soft: #eef4f8;
      --shadow: 0 18px 42px rgba(4, 24, 39, 0.12);
      --radius: 16px;
      --radius-sm: 10px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Figtree", sans-serif;
      color: var(--black);
      background: linear-gradient(180deg, var(--blue-extra-dark) 0%, #0a2437 26%, #f3f7fa 26%, #f3f7fa 100%);
      background-attachment: fixed;
      min-height: 100vh;
    }}
    .page {{
      width: min(1600px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 64px;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 270px minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }}
    .side-rail {{
      position: sticky;
      top: 18px;
      align-self: start;
      max-height: calc(100vh - 36px);
      overflow-y: auto;
      padding: 0 8px 16px;
      background: var(--blue-extra-dark);
      border-radius: 22px;
      box-shadow: 0 20px 48px rgba(4, 24, 39, 0.28);
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.24) transparent;
    }}
    .side-rail::-webkit-scrollbar {{
      width: 8px;
    }}
    .side-rail::-webkit-scrollbar-track {{
      background: transparent;
    }}
    .side-rail::-webkit-scrollbar-thumb {{
      background: rgba(255,255,255,0.22);
      border-radius: 999px;
    }}
    .hero {{
      color: var(--white);
      padding: 28px 6px 20px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 2rem; line-height: 1.05; }}
    .hero p {{ margin: 0; max-width: 1000px; color: rgba(255,255,255,0.82); font-size: 1rem; }}
    .nav {{
      display: flex; flex-direction: column; gap: 10px; margin-top: 18px;
    }}
    .nav a {{
      color: var(--white); text-decoration: none; border: 1px solid rgba(255,255,255,0.18);
      padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.06); font-size: 0.92rem;
    }}
    .stats-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 10px 0 22px;
    }}
    .stat-card, .card {{
      background: var(--white);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .stat-card {{ padding: 18px 18px 16px; }}
    .stat-label {{ color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .stat-value {{ margin-top: 6px; font-size: 1.7rem; font-weight: 800; color: var(--blue); }}
    .card {{ padding: 22px 22px 18px; margin: 0 0 18px; }}
    .section-head {{ margin-bottom: 12px; }}
    .section-head h2, .section-head h3 {{ margin: 0 0 6px; color: var(--blue); }}
    .section-head p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .two-col {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px;
    }}
    .plot-shell {{
      background: var(--bg-soft); border-radius: var(--radius-sm); padding: 10px; min-height: 120px;
    }}
    .controls {{
      display: flex; flex-wrap: wrap; gap: 14px; align-items: end; margin-bottom: 14px;
    }}
    .control {{
      display: flex; flex-direction: column; gap: 6px; min-width: 220px;
    }}
    .control label {{
      font-size: 0.84rem; font-weight: 700; color: var(--blue-medium); text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .control select {{
      border: 1px solid #c9d7e1; border-radius: 10px; padding: 10px 12px; font: inherit; background: var(--white);
    }}
    .metrics-table-wrap {{
      max-height: 360px; overflow: auto; border-radius: var(--radius-sm); border: 1px solid #dde8ef; background: #fff;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e7eef3; text-align: left; font-size: 0.92rem; }}
    th {{ position: sticky; top: 0; background: #f7fafc; color: var(--blue); z-index: 1; }}
    .actogram-image {{ width: 100%; height: auto; border-radius: 12px; border: 1px solid #d7e3eb; background: #fff; }}
    .empty-note {{
      display: flex; align-items: center; justify-content: center; min-height: 120px; color: var(--muted); font-style: italic;
      background: linear-gradient(135deg, #f8fbfd 0%, #eef4f8 100%); border-radius: 10px;
    }}
    .footer-note {{
      color: #dde7ee; font-size: 0.92rem; margin-top: 12px;
    }}
    .best-method-box {{
      background: linear-gradient(135deg, #eff6fb 0%, #f8fbfd 100%);
      border: 1px solid #d6e4ee;
      border-radius: 12px;
      padding: 14px 16px;
      margin-top: 14px;
    }}
    .best-method-box ul {{ margin: 10px 0 8px 18px; padding: 0; }}
    .best-method-box li {{ margin: 0 0 8px; }}
    .metric-pill {{
      display: inline-block;
      margin-left: 8px;
      padding: 2px 8px;
      border-radius: 999px;
      background: var(--blue-dark);
      color: var(--white);
      font-size: 0.82rem;
      font-weight: 700;
    }}
    .small-note {{ color: var(--muted); font-size: 0.9rem; }}
    .feature-grid {{
      display: grid;
      grid-template-columns: minmax(260px, 360px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .feature-plot-panel {{
      background: var(--bg-soft);
      border-radius: var(--radius-sm);
      padding: 10px;
    }}
    details.collapsible {{
      background: #f8fbfd;
      border: 1px solid #dde8ef;
      border-radius: 12px;
      padding: 12px 14px;
    }}
    details.collapsible[open] {{
      background: #ffffff;
    }}
    details.collapsible summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--blue);
      list-style: none;
    }}
    details.collapsible summary::-webkit-details-marker {{
      display: none;
    }}
    details.collapsible summary::after {{
      content: "Show";
      float: right;
      color: var(--blue-medium);
      font-weight: 600;
      font-size: 0.86rem;
    }}
    details.collapsible[open] summary::after {{
      content: "Hide";
    }}
    .details-body {{
      margin-top: 12px;
    }}
    .outline-card {{
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 16px;
      padding: 16px;
      backdrop-filter: blur(10px);
    }}
    .outline-card h3 {{
      margin: 0 0 8px;
      color: var(--white);
      font-size: 1rem;
    }}
    .outline-card p {{
      margin: 0;
      color: rgba(255,255,255,0.72);
      font-size: 0.9rem;
      line-height: 1.4;
    }}
    .config-card table td {{
      border-bottom: 1px solid rgba(255,255,255,0.10);
      color: rgba(255,255,255,0.9);
      font-size: 0.84rem;
      padding: 7px 0;
      vertical-align: top;
    }}
    .config-card table td:first-child {{
      color: rgba(255,255,255,0.66);
      width: 44%;
      padding-right: 10px;
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.04em;
    }}
    .mini-table-wrap {{
      max-height: 260px;
      overflow: auto;
    }}
    .side-details {{
      margin-top: 10px;
      background: rgba(255,255,255,0.06) !important;
      border-color: rgba(255,255,255,0.12) !important;
    }}
    .side-details summary {{
      color: var(--white) !important;
    }}
    .side-details summary::after {{
      color: rgba(255,255,255,0.72) !important;
    }}
    .feature-description {{
      margin: 10px 0 14px;
      padding: 12px 14px;
      border-radius: 10px;
      background: #f6fafc;
      border: 1px solid #d8e5ee;
      color: var(--muted);
      line-height: 1.45;
      min-height: 68px;
    }}
    .interactive-frame {{
      width: 100%;
      min-height: 840px;
      border: 1px solid #d7e3eb;
      border-radius: 14px;
      background: #ffffff;
    }}
    .context-summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .context-big-number {{
      margin: 6px 0 0;
      font-size: 1.55rem;
      font-weight: 800;
      color: var(--blue);
    }}
    .rejection-gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
      margin-top: 14px;
    }}
    .rejection-card {{
      background: #f8fbfd;
      border: 1px solid #dde8ef;
      border-radius: 14px;
      padding: 14px;
    }}
    .rejection-card h4 {{
      margin: 0 0 8px;
      color: var(--blue);
    }}
    .rejection-meta p {{
      margin: 0 0 6px;
      color: var(--muted);
      line-height: 1.4;
      font-size: 0.92rem;
    }}
    .rejection-image {{
      width: 100%;
      height: auto;
      display: block;
      margin-top: 10px;
      border-radius: 10px;
      border: 1px solid #d7e3eb;
      background: #fff;
    }}
    @media (max-width: 900px) {{
      .page {{ width: calc(100vw - 20px); }}
      .card {{ padding: 18px 16px 14px; }}
      .two-col {{ grid-template-columns: 1fr; }}
      .shell {{ grid-template-columns: 1fr; }}
      .side-rail {{ position: static; }}
      .feature-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="shell">
      <aside class="side-rail">
        <header class="hero">
          <h1>Supervised Segmentation Review</h1>
          <p>This report consolidates supervised-label review for <strong>{html.escape(ctx.run_name)}</strong>. It compares supervision methods, summarizes holdout performance, shows deployment-level actograms, and links behavior labels to clustering structure.</p>
          <div class="outline-card" style="margin-top:18px;">
            <h3>Outline</h3>
            <nav class="nav">
              <a href="#overview">Overview</a>
              <a href="#budgets">Budgets</a>
              <a href="#circadian">Circadian</a>
              <a href="#actograms">Actograms</a>
              <a href="#group-contrasts">Group/Dataset Contrasts</a>
              <a href="#performance">Performance</a>
              <a href="#feature-diagnostics">Feature Diagnostics</a>
              <a href="#run-diagnostics">Run Diagnostics</a>
              <a href="#interactive-review">Interactive Review</a>
              <a href="#clusters">Clusters</a>
            </nav>
            <p style="margin-top:12px;">Use the method and deployment selectors to narrow the review without leaving the page.</p>
          </div>
          {config_sidebar_html}
          <div class="footer-note">Design direction follows the Dash app palette and card system in <code>DiveDB/dash/assets/sass</code>.</div>
        </header>
      </aside>
      <main>
    <section id="overview" class="stats-grid">
      {summary_cards_html}
    </section>

    <section class="card">
      <div class="section-head">
        <h2>How To Read This Report</h2>
        <p>Start with budgets and circadian sections to evaluate day-level behavior composition and solar timing. Then review actograms and group/dataset contrasts; performance and confusion diagnostics remain below as secondary model-review sections.</p>
      </div>
    </section>

    <section id="budgets" class="card">
      <div class="section-head">
        <h2>Primary Method Budgets</h2>
        <p>This budget-first section resolves the primary method from run config (with fallback) and summarizes daily behavior budgets before model diagnostics.</p>
      </div>
      {unmapped_labels_html}
      {primary_budget_section_html}
    </section>

    <section id="circadian" class="card">
      <div class="section-head">
        <h2>Sunrise/Sunset Circadian Context</h2>
        <p>Daily sunrise and sunset tracks are shown for the primary method scope to support circadian interpretation of activity budgets.</p>
      </div>
      <div class="plot-shell">{circadian_section_html}</div>
    </section>

    <section id="actograms" class="card">
      <div class="section-head">
        <h2>Primary Method Actograms</h2>
        <p>Daily and continuous actograms for the primary method are shown here, including an event-timing overlay on the continuous day/night timeline.</p>
      </div>
      {primary_actogram_gallery_html}
    </section>

    <section id="group-contrasts" class="card">
      <div class="section-head">
        <h2>Group and Dataset Contrasts</h2>
        <p>Cross-group and cross-dataset contrasts are aggregated from the same method-budget tables used in the primary budget summary.</p>
      </div>
      <div class="plot-shell">{group_contrast_section_html}</div>
    </section>

    <section id="performance" class="card">
      <div class="section-head">
        <h2>Performance Overview</h2>
        <p>Precision, recall, F1, accuracy, and balanced accuracy are shown together so you can evaluate both class-specific behavior and overall method stability. When multiple held-out deployments were used, this top panel shows the mean across them.</p>
      </div>
      <div class="plot-shell">{metrics_fragment or "<div class='empty-note'>No variant metrics available.</div>"}</div>
      {best_method_html}
      {algorithmic_performance_html}
    </section>
    {per_fold_metrics_html}

    <section class="card">
      <div class="section-head">
        <h2>Per-Method Accuracy And Confusion</h2>
        <p>Each method now pairs its compact accuracy summary with its own confusion matrix. False-positive and false-negative cells use dedicated colors, and the label in each box shows both count and rate.</p>
      </div>
      {performance_pair_cards_html or "<div class='empty-note'>No confusion-matrix comparison available.</div>"}
    </section>

    <section class="card">
      <div class="section-head">
        <h2>Target Weak-Label Confusion</h2>
        <p>This target-only section uses a weak negative label derived from <code>stroke_rate</code>. Any target window with <code>stroke_rate &gt; 10</code> for more than one 30 s chunk is treated as a weak negative, so predicted <strong>{html.escape(str(_detect_supervised_positive_label(spdf['predicted_label'].astype(str).tolist()) or 'sleep'))}</strong> in those windows counts as a false positive. False negatives are left blank because the target datasets do not provide weak positive labels.</p>
      </div>
      {target_weak_label_cards_html or "<div class='empty-note'>No target weak-label confusion available.</div>"}
      <div class="section-head" style="margin-top:18px;">
        <h3>Weak-Label Summary Table</h3>
      </div>
      {target_weak_label_table_html}
    </section>

    <section class="card">
      <div class="section-head">
        <h2>Metric Table</h2>
        <p>This table mirrors the plots above and is useful for exact values when you want to compare small differences between methods.</p>
      </div>
      <div class="metrics-table-wrap">
        <table>
          <thead><tr><th>Method</th><th>State</th><th>Metric</th><th>Value</th></tr></thead>
          <tbody>{metric_rows_html}</tbody>
        </table>
      </div>
    </section>

    <section id="run-diagnostics" class="card">
      <div class="section-head">
        <h2>Run Diagnostics</h2>
        <p>This section collects the outputs that previously lived as separate files in the segmentation folder: QC status, feature filtering, label-assignment coverage, threshold behavior, clustering diagnostics, and algorithmic segment summaries.</p>
      </div>
      <div class="two-col">
        <div>
          <div class="section-head"><h3>QC Channels</h3></div>
          {qc_summary_html}
        </div>
        <div>
          <div class="section-head"><h3>Feature Filter Summary</h3></div>
          {feature_filter_html}
        </div>
      </div>
      <details class="collapsible">
        <summary>Expanded diagnostics</summary>
        <div class="details-body">
          <div class="two-col">
            <div>
              <div class="section-head"><h3>Label Scan Coverage</h3></div>
              {label_scan_html}
            </div>
            <div>
              <div class="section-head"><h3>Threshold Diagnostics</h3></div>
              <div class="plot-shell">{threshold_fragment or "<div class='empty-note'>No threshold diagnostics available.</div>"}</div>
            </div>
          </div>
          <div class="two-col">
            <div>
              <div class="section-head"><h3>Cluster Gap Report</h3></div>
              {cluster_gap_html}
            </div>
            <div>
              <div class="section-head"><h3>Cluster Interpretability</h3></div>
              {interpretability_html}
            </div>
          </div>
          <div class="two-col">
            <div>
              <div class="section-head"><h3>Algorithmic Segment Summaries</h3></div>
              {algorithmic_summary_html}
              <div class="section-head" style="margin-top:14px;"><h3>Context Filtering Summary</h3><p>This summarizes how many base rule-based candidates were rejected by the buoyancy-aware context layer and why.</p></div>
              {algorithmic_context_html}
            </div>
            <div>
              <div class="section-head"><h3>Legacy Plot Files</h3></div>
              <div class="best-method-box">
                <p>The original standalone outputs are still linked here so nothing from the previous folder structure is hidden.</p>
                <ul>{''.join(legacy_plots_html) or '<li>No legacy plot files found.</li>'}</ul>
              </div>
            </div>
          </div>
          <div class="section-head" style="margin-top:18px;"><h3>Rejected Segment Examples</h3><p>The first few context-rejected segments are shown below using the detector's core input channels: depth plus its first and second derivatives.</p></div>
          {algorithmic_examples_html or "<div class='empty-note'>No rejected-segment examples were available.</div>"}
        </div>
      </details>
    </section>

    <section id="feature-diagnostics" class="card">
      <div class="section-head">
        <h2>Feature Diagnostics</h2>
        <p>This section combines global feature correlation, PCA loading summaries, and RF importance summaries. The PCA violins remain cluster-facing, while the RF violins switch to supervised classes so you can see whether the most important model features separate the labeled states directly.</p>
      </div>
      <div class="plot-shell">{correlogram_fragment or "<div class='empty-note'>No correlogram available.</div>"}</div>
    </section>

    <section id="interactive-review" class="card">
      <div class="section-head">
        <h2>Interactive Review</h2>
        <p>This embedded viewer shows one 6-hour chunk per deployment in <code>plot_tag_data_interactive</code> format. Use the selectors to switch between deployments and the segmentation methods actually available for each deployment.</p>
      </div>
      {interactive_embed_html}
    </section>

    <section class="card">
      <div class="section-head">
        <h2>PCA-Loaded Features</h2>
        <p>The table lists the ten strongest PCA-loaded features in the k5 clustering pass, ranked by maximum absolute loading across principal components. Cluster numbers in the violin plots follow the run's cluster ordering, which is based on mean <code>{html.escape(str(ctx.run_cfg.get('primary_input_channel_for_ranking', 'depth.depth')))}</code> within each cluster.</p>
      </div>
      <div class="feature-grid">
        <div>{pca_top_table_html}</div>
        <div>
          <div class="controls">
            <div class="control">
              <label for="pca-feature-select">PCA feature</label>
              <select id="pca-feature-select">{''.join(pca_feature_options) or '<option value="">No PCA features</option>'}</select>
            </div>
          </div>
          {''.join(pca_violin_blocks) or "<div class='empty-note'>No PCA violin plots available.</div>"}
        </div>
      </div>
    </section>

    <section class="card">
      <div class="section-head">
        <h2>Random-Forest Features</h2>
        <p>The table lists the ten most important RF features from the primary supervised model. Each feature now includes a plain-language description of how it is calculated, and the violin plots are grouped by observed supervised class rather than unsupervised cluster.</p>
      </div>
      <div class="two-col" style="margin-bottom:18px;">
        <div class="plot-shell">{fi_fragment or "<div class='empty-note'>No feature-importance artifact available.</div>"}</div>
        <div>{rf_top_table_html}</div>
      </div>
      <div class="controls">
        <div class="control">
          <label for="rf-feature-select">RF feature</label>
          <select id="rf-feature-select">{''.join(rf_feature_options) or '<option value="">No RF features</option>'}</select>
        </div>
      </div>
      <div id="rf-feature-description" class="feature-description">{html.escape(_describe_feature_name(rf_feature_default)) if rf_feature_default else 'No RF feature description available.'}</div>
      {''.join(rf_violin_blocks) or "<div class='empty-note'>No RF violin plots available.</div>"}
    </section>

    <section class="card">
      <div class="section-head">
        <h2>Signal Overlap By Dataset</h2>
        <p>The five strongest source signals are ranked by summed RF importance. Each review plot uses the top representative feature from that signal, with raw values on one row and z-scored values on the other so you can see what the normalization is removing before model fitting.</p>
      </div>
      <div class="best-method-box" style="margin-bottom:18px;">{feature_overlap_note}</div>
      <div class="two-col" style="margin-bottom:18px;">
        <div>{signal_overlap_summary_html}</div>
        <div>{signal_overlap_check_html}</div>
      </div>
      <div class="controls">
        <div class="control">
          <label for="signal-overlap-select">Signal</label>
          <select id="signal-overlap-select">{''.join(signal_overlap_options) or '<option value="">No top signals</option>'}</select>
        </div>
      </div>
      <div id="signal-overlap-description" class="feature-description">{html.escape(signal_overlap_desc.get(signal_overlap_default, 'No signal overlap description available.'))}</div>
      {''.join(signal_overlap_blocks) or "<div class='empty-note'>No signal overlap plots available.</div>"}
    </section>

    <section class="card">
      <div class="section-head">
        <h2>Feature Overlap By Dataset</h2>
        <p>These are the five most influential RF features directly. Raw and z-scored distributions are paired in the same figure, still faceted by dataset, so you can check whether dataset-specific offsets collapse after normalization while class structure remains visible.</p>
      </div>
      <div class="two-col" style="margin-bottom:18px;">
        <div>{feature_overlap_summary_html}</div>
        <div>{feature_overlap_check_html}</div>
      </div>
      <div class="controls">
        <div class="control">
          <label for="feature-overlap-select">Feature</label>
          <select id="feature-overlap-select">{''.join(feature_overlap_options) or '<option value="">No top features</option>'}</select>
        </div>
      </div>
      <div id="feature-overlap-description" class="feature-description">{html.escape(feature_overlap_desc.get(feature_overlap_default, 'No feature overlap description available.'))}</div>
      {''.join(feature_overlap_blocks) or "<div class='empty-note'>No feature overlap plots available.</div>"}
    </section>

    <section id="budgets" class="card">
      <div class="section-head">
        <h2>Observed And Cluster Activity Budgets</h2>
        <p>This section focuses on the observed labels and the available cluster passes used for review. Daily budgets are normalized to 24 h/day, and any uncovered remainder is shown explicitly as <strong>{html.escape(remainder_label)}</strong>.</p>
      </div>
      <div class="controls">
        <div class="control">
          <label for="variant-select">Supervision method</label>
          <select id="variant-select">{''.join(variant_selector_options)}</select>
        </div>
      </div>
      {''.join(method_sections)}
    </section>

    <section id="actograms-detail" class="card">
      <div class="section-head">
        <h2>Deployment Actograms (All Methods)</h2>
        <p>Each deployment actogram places observed labels first, then the supervision-method panels. Predicted panels are colored by generic confusion outcome so false positives, false negatives, true positives, and true negatives remain interpretable for any positive-state definition.</p>
      </div>
      <div class="controls">
        <div class="control">
          <label for="deployment-select">Deployment</label>
          <select id="deployment-select">{''.join(deployment_selector_options)}</select>
        </div>
      </div>
      {''.join(actogram_cards) or "<div class='empty-note'>No supervised actogram images were found.</div>"}
    </section>

    <section id="clusters" class="card">
      <div class="section-head">
        <h2>Supervised Labels and Clusters</h2>
        <p>These Sankey plots help test whether supervised sleep labels line up with unsupervised cluster structure. When an explicit <code>k2</code> clustering pass is not present, the report attempts an exploratory two-cluster review split from the available feature matrix.</p>
      </div>
      <div class="two-col">
        {sankey_html}
      </div>
    </section>
      </main>
    </div>
  </div>

  <script>
    function filterPanels(selectId, panelClass, attrName) {{
      const select = document.getElementById(selectId);
      if (!select) return;
      const update = () => {{
        const value = select.value;
        document.querySelectorAll('.' + panelClass).forEach((node) => {{
          const keep = value === 'all' || node.getAttribute(attrName) === value;
          node.style.display = keep ? '' : 'none';
        }});
      }};
      select.addEventListener('change', update);
      update();
    }}
    function filterFeaturePanels(selectId, groupName) {{
      const select = document.getElementById(selectId);
      if (!select) return;
      const update = () => {{
        const value = select.value;
        document.querySelectorAll('[data-feature-group=\"' + groupName + '\"]').forEach((node) => {{
          const keep = node.getAttribute('data-feature') === value;
          node.style.display = keep ? '' : 'none';
        }});
      }};
      select.addEventListener('change', update);
      update();
    }}
    const rfFeatureDescriptions = {rf_feature_desc_json};
    const signalOverlapDescriptions = {signal_overlap_desc_json};
    const featureOverlapDescriptions = {feature_overlap_desc_json};
    const interactiveManifest = {interactive_manifest_json};
    function updateFeatureDescription(selectId, targetId, descriptionMap) {{
      const select = document.getElementById(selectId);
      const target = document.getElementById(targetId);
      if (!select || !target) return;
      const update = () => {{
        const value = select.value;
        target.textContent = descriptionMap[value] || 'No description available.';
      }};
      select.addEventListener('change', update);
      update();
    }}
    filterPanels('variant-select', 'variant-panel', 'data-variant');
    filterPanels('deployment-select', 'deployment-panel', 'data-deployment');
    filterFeaturePanels('pca-feature-select', 'pca');
    filterFeaturePanels('rf-feature-select', 'rf');
    filterFeaturePanels('signal-overlap-select', 'signal-overlap');
    filterFeaturePanels('feature-overlap-select', 'feature-overlap');
    updateFeatureDescription('rf-feature-select', 'rf-feature-description', rfFeatureDescriptions);
    updateFeatureDescription('signal-overlap-select', 'signal-overlap-description', signalOverlapDescriptions);
    updateFeatureDescription('feature-overlap-select', 'feature-overlap-description', featureOverlapDescriptions);
    (function initInteractiveReview() {{
      const depSelect = document.getElementById('interactive-deployment-select');
      const methodSelect = document.getElementById('interactive-method-select');
      const frame = document.getElementById('interactive-review-frame');
      if (!depSelect || !methodSelect || !frame || !interactiveManifest.length) return;
      const refreshMethods = () => {{
        const depKey = depSelect.value;
        const rows = interactiveManifest.filter((row) => row.deployment_key === depKey);
        methodSelect.innerHTML = rows.map((row) => `<option value="${{row.relative_path}}">${{row.method_label}}</option>`).join('');
        if (rows.length) {{
          frame.src = rows[0].relative_path;
        }} else {{
          frame.removeAttribute('src');
        }}
      }};
      methodSelect.addEventListener('change', () => {{
        frame.src = methodSelect.value || '';
      }});
      depSelect.addEventListener('change', refreshMethods);
      refreshMethods();
    }})();
  </script>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return report_path


def _write_interactive_review_only_report(
    ctx: RunContext,
    out_dir: str,
    cdf: pd.DataFrame,
    manifest_path: str | None = None,
) -> str | None:
    if manifest_path is None:
        manifest_path = os.path.join(out_dir, "interactive_review_manifest.json")
    manifest = []
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = []

    report_path = os.path.join(ctx.output_root, "00_supervised_review_report.html")
    asset_prefix = os.path.relpath(out_dir, os.path.dirname(report_path))
    asset_prefix = "" if asset_prefix == "." else asset_prefix.replace(os.sep, "/")

    manifest_rows = []
    for row in manifest:
        rel = str(row.get("relative_path", "")).replace(os.sep, "/")
        prefixed = "/".join([x for x in [asset_prefix, rel] if x])
        manifest_rows.append(
            {
                "dataset_id": str(row.get("dataset_id", "")),
                "deployment_id": str(row.get("deployment_id", "")),
                "deployment_key": str(row.get("deployment_key", "")),
                "method_kind": str(row.get("method_kind", "")),
                "method_id": str(row.get("method_id", "")),
                "method_label": str(row.get("method_label", "")),
                "relative_path": prefixed,
            }
        )
    deployment_keys = sorted({str(row["deployment_key"]) for row in manifest_rows if row.get("deployment_key")})
    deployment_options = "".join(
        f"<option value=\"{html.escape(dep)}\">{html.escape(dep)}</option>" for dep in deployment_keys
    )

    notebook_rel = "/".join([x for x in [asset_prefix, "A_interactive_deployment_overlay.ipynb"] if x])
    algo_summary_html = "<div class='empty-note'>No merged algorithmic segment table found.</div>"
    detailed_segments_path, _ = _algorithmic_segment_merge_paths(ctx)
    if os.path.exists(detailed_segments_path):
        try:
            seg_df = pd.read_parquet(detailed_segments_path)
            if not seg_df.empty:
                total_segments = int(len(seg_df))
                base_kept = int(seg_df.get("base_keep_filtered", pd.Series(False, index=seg_df.index)).fillna(False).sum())
                final_kept = int(seg_df.get("context_keep", pd.Series(True, index=seg_df.index)).fillna(True).sum())
                context_rejected = max(0, int(base_kept - final_kept))
                trip_phase_counts = {}
                if "inferred_trip_phase" in seg_df.columns:
                    trip_phase_counts = (
                        seg_df["inferred_trip_phase"].fillna("unknown").astype(str).value_counts().to_dict()
                    )
                reject_reason_counts = {}
                if "context_reject_reason" in seg_df.columns:
                    reject_reason_counts = (
                        seg_df["context_reject_reason"]
                        .fillna("kept")
                        .astype(str)
                        .value_counts()
                        .head(8)
                        .to_dict()
                    )
                trip_phase_rows = "".join(
                    f"<tr><td>{html.escape(str(k))}</td><td>{int(v)}</td></tr>" for k, v in trip_phase_counts.items()
                ) or "<tr><td colspan='2'>No trip-phase annotations found.</td></tr>"
                reject_rows = "".join(
                    f"<tr><td>{html.escape(str(k))}</td><td>{int(v)}</td></tr>" for k, v in reject_reason_counts.items()
                ) or "<tr><td colspan='2'>No reject reasons recorded.</td></tr>"
                algo_summary_html = f"""
                <div class="stats-grid">
                  <div class="stat-card"><div class="stat-label">Merged segments</div><div class="stat-value">{total_segments}</div></div>
                  <div class="stat-card"><div class="stat-label">Base-kept segments</div><div class="stat-value">{base_kept}</div></div>
                  <div class="stat-card"><div class="stat-label">Context-rejected</div><div class="stat-value">{context_rejected}</div></div>
                  <div class="stat-card"><div class="stat-label">Final-kept</div><div class="stat-value">{final_kept}</div></div>
                </div>
                <div class="two-col">
                  <div class="mini-table-wrap">
                    <table>
                      <thead><tr><th>Inferred trip phase</th><th>Count</th></tr></thead>
                      <tbody>{trip_phase_rows}</tbody>
                    </table>
                  </div>
                  <div class="mini-table-wrap">
                    <table>
                      <thead><tr><th>Context result / reject reason</th><th>Count</th></tr></thead>
                      <tbody>{reject_rows}</tbody>
                    </table>
                  </div>
                </div>
                """
        except Exception as e:
            algo_summary_html = f"<div class='empty-note'>Could not summarize algorithmic segments: {html.escape(str(e))}</div>"

    cluster_pass_text = ", ".join(
        sorted(cdf.get("cluster_pass", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
    ) or "None"
    summary_cards_html = "".join(
        [
            f"<div class='stat-card'><div class='stat-label'>Deployments</div><div class='stat-value'>{int(cdf[['dataset_id', 'deployment_id']].drop_duplicates().shape[0])}</div></div>",
            f"<div class='stat-card'><div class='stat-label'>Interactive views</div><div class='stat-value'>{len(manifest_rows)}</div></div>",
            f"<div class='stat-card'><div class='stat-label'>Cluster passes</div><div class='stat-value' style='font-size:1rem'>{html.escape(cluster_pass_text)}</div></div>",
            f"<div class='stat-card'><div class='stat-label'>Supervised outputs</div><div class='stat-value' style='font-size:1rem'>Disabled / missing</div></div>",
        ]
    )

    interactive_manifest_json = json.dumps(manifest_rows)
    interactive_section_html = (
        f"""
        <section id="interactive-review" class="card">
          <div class="section-head">
            <h2>Interactive Review</h2>
            <p>Use the selectors to switch between deployments and the segmentation methods available for each deployment. The iframe loads the prebuilt review HTML directly, so regenerating this wrapper is cheap.</p>
          </div>
          <div class="controls">
            <div class="control">
              <label for="interactive-deployment-select">Deployment</label>
              <select id="interactive-deployment-select">{deployment_options}</select>
            </div>
            <div class="control">
              <label for="interactive-method-select">Method</label>
              <select id="interactive-method-select"></select>
            </div>
          </div>
          <iframe id="interactive-review-frame" class="interactive-frame" loading="lazy"></iframe>
        </section>
        """
        if manifest_rows
        else """
        <section id="interactive-review" class="card">
          <div class="section-head">
            <h2>Interactive Review</h2>
            <p>Interactive review assets were not generated for this run configuration.</p>
          </div>
          <div class="empty-note">No interactive_review manifest was found. You can still use the algorithmic summaries and parquet outputs below.</div>
        </section>
        """
    )
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Interactive Segmentation Review | {html.escape(ctx.run_name)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --blue-extra-dark: #041827;
      --blue-dark: #0e3551;
      --blue: #092a42;
      --blue-medium: #286591;
      --white: #ffffff;
      --black: #333333;
      --muted: #6b7a88;
      --bg-soft: #eef4f8;
      --shadow: 0 18px 42px rgba(4, 24, 39, 0.12);
      --radius: 16px;
      --radius-sm: 10px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Figtree", sans-serif;
      color: var(--black);
      background: linear-gradient(180deg, var(--blue-extra-dark) 0%, #0a2437 24%, #f3f7fa 24%, #f3f7fa 100%);
      background-attachment: fixed;
      min-height: 100vh;
    }}
    .page {{ width: min(1500px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 64px; }}
    .shell {{ display: grid; grid-template-columns: 270px minmax(0, 1fr); gap: 22px; align-items: start; }}
    .side-rail {{ position: sticky; top: 18px; align-self: start; max-height: calc(100vh - 36px); overflow-y: auto; padding: 0 8px 16px; background: var(--blue-extra-dark); border-radius: 22px; box-shadow: 0 20px 48px rgba(4, 24, 39, 0.28); }}
    .hero {{ color: var(--white); padding: 28px 6px 20px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 2rem; line-height: 1.05; }}
    .hero p {{ margin: 0; color: rgba(255,255,255,0.82); line-height: 1.5; }}
    .nav {{ display: flex; flex-direction: column; gap: 10px; margin-top: 18px; }}
    .nav a {{ color: var(--white); text-decoration: none; border: 1px solid rgba(255,255,255,0.18); padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.06); font-size: 0.92rem; }}
    .card, .stat-card {{ background: var(--white); border-radius: var(--radius); box-shadow: var(--shadow); }}
    .card {{ padding: 22px 22px 18px; margin: 0 0 18px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 10px 0 22px; }}
    .stat-card {{ padding: 18px 18px 16px; }}
    .stat-label {{ color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .stat-value {{ margin-top: 6px; font-size: 1.6rem; font-weight: 800; color: var(--blue); }}
    .section-head {{ margin-bottom: 12px; }}
    .section-head h2, .section-head h3 {{ margin: 0 0 6px; color: var(--blue); }}
    .section-head p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: end; margin-bottom: 14px; }}
    .control {{ display: flex; flex-direction: column; gap: 6px; min-width: 260px; }}
    .control label {{ font-size: 0.84rem; font-weight: 700; color: var(--blue-medium); text-transform: uppercase; letter-spacing: 0.05em; }}
    .control select {{ border: 1px solid #c9d7e1; border-radius: 10px; padding: 10px 12px; font: inherit; background: var(--white); }}
    .interactive-frame {{ width: 100%; min-height: 900px; border: 1px solid #d7e3eb; border-radius: 14px; background: #ffffff; }}
    .plot-shell {{ background: var(--bg-soft); border-radius: var(--radius-sm); padding: 10px; min-height: 120px; }}
    .two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    .mini-table-wrap {{ max-height: 280px; overflow: auto; border-radius: var(--radius-sm); border: 1px solid #dde8ef; background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e7eef3; text-align: left; font-size: 0.92rem; }}
    th {{ position: sticky; top: 0; background: #f7fafc; color: var(--blue); z-index: 1; }}
    .best-method-box {{ background: linear-gradient(135deg, #eff6fb 0%, #f8fbfd 100%); border: 1px solid #d6e4ee; border-radius: 12px; padding: 14px 16px; }}
    .empty-note {{ display: flex; align-items: center; justify-content: center; min-height: 120px; color: var(--muted); font-style: italic; background: linear-gradient(135deg, #f8fbfd 0%, #eef4f8 100%); border-radius: 10px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    @media (max-width: 900px) {{
      .page {{ width: calc(100vw - 20px); }}
      .shell {{ grid-template-columns: 1fr; }}
      .side-rail {{ position: static; }}
      .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="shell">
      <aside class="side-rail">
        <header class="hero">
          <h1>Interactive Segmentation Review</h1>
          <p>This run did not produce supervised prediction parquet, so the report falls back to the interactive review assets and algorithmic summaries. This is the main entrypoint for <strong>{html.escape(ctx.run_name)}</strong>.</p>
          <nav class="nav">
            <a href="#overview">Overview</a>
            <a href="#interactive-review">Interactive Review</a>
            <a href="#algorithmic">Algorithmic Summary</a>
            <a href="#files">Useful Files</a>
          </nav>
        </header>
      </aside>
      <main>
        <section id="overview" class="stats-grid">{summary_cards_html}</section>
        <section class="card">
          <div class="section-head">
            <h2>What Happened</h2>
            <p>The expensive embedded interactive review pages were exported successfully, but the root HTML wrapper was previously skipped because the report code only ran when supervised predictions existed. This fallback page lets you review algorithmic and cluster outputs directly.</p>
          </div>
          <div class="best-method-box">
            <p><strong>Notebook:</strong> <a href="{html.escape(notebook_rel)}">{html.escape(notebook_rel)}</a></p>
            <p><strong>Manifest:</strong> <code>{html.escape(os.path.relpath(manifest_path, os.path.dirname(report_path)).replace(os.sep, '/'))}</code></p>
          </div>
        </section>
        {interactive_section_html}
        <section id="algorithmic" class="card">
          <div class="section-head">
            <h2>Algorithmic Summary</h2>
            <p>This summary is built from the merged algorithmic segment table, including base-kept versus context-filtered outcomes when available.</p>
          </div>
          {algo_summary_html}
        </section>
        <section id="files" class="card">
          <div class="section-head">
            <h2>Useful Files</h2>
            <p>These are the main outputs to keep using this run without regenerating expensive plots.</p>
          </div>
          <div class="best-method-box">
            <ul>
              <li><a href="{html.escape(notebook_rel)}">plots/A_interactive_deployment_overlay.ipynb</a></li>
              <li><code>{html.escape(os.path.relpath(os.path.join(out_dir, "interactive_review"), os.path.dirname(report_path)).replace(os.sep, '/'))}</code></li>
              <li><code>{html.escape(os.path.relpath(os.path.join(ctx.output_root, "segments", "algorithmic_segments.parquet"), os.path.dirname(report_path)).replace(os.sep, '/'))}</code></li>
              <li><code>{html.escape(os.path.relpath(os.path.join(ctx.output_root, "segments", "algorithmic_label_timeseries.parquet"), os.path.dirname(report_path)).replace(os.sep, '/'))}</code></li>
            </ul>
          </div>
        </section>
      </main>
    </div>
  </div>
  <script>
    const interactiveManifest = {interactive_manifest_json};
    (function initInteractiveReview() {{
      const depSelect = document.getElementById('interactive-deployment-select');
      const methodSelect = document.getElementById('interactive-method-select');
      const frame = document.getElementById('interactive-review-frame');
      if (!depSelect || !methodSelect || !frame || !interactiveManifest.length) return;
      const refreshMethods = () => {{
        const depKey = depSelect.value;
        const rows = interactiveManifest.filter((row) => row.deployment_key === depKey);
        methodSelect.innerHTML = rows.map((row) => `<option value="${{row.relative_path}}">${{row.method_label}}</option>`).join('');
        if (rows.length) {{
          frame.src = rows[0].relative_path;
        }} else {{
          frame.removeAttribute('src');
        }}
      }};
      methodSelect.addEventListener('change', () => {{
        frame.src = methodSelect.value || '';
      }});
      depSelect.addEventListener('change', refreshMethods);
      refreshMethods();
    }})();
  </script>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return report_path


def _write_embedded_interactive_review_assets(
    ctx: RunContext,
    out_dir: str,
    cdf: pd.DataFrame,
    requested_channel_map: dict,
    primary_signal: str,
    primary_channel: str,
    preview_hours: float = 6.0,
    deployment_scope: str = "all",
    time_range_mode: str = "centered_window",
) -> str | None:
    from pyologger.plot_data.plotter import plot_tag_data_interactive

    supervised_variant_path = os.path.join(ctx.output_root, "supervised", "supervised_variant_predictions.parquet")
    algorithmic_label_path = _algorithmic_label_timeseries_merge_path(ctx)
    spdf = pd.read_parquet(supervised_variant_path) if os.path.exists(supervised_variant_path) else pd.DataFrame()
    algo_df = pd.read_parquet(algorithmic_label_path) if os.path.exists(algorithmic_label_path) else pd.DataFrame()
    if not spdf.empty:
        spdf["window_start"] = pd.to_datetime(spdf["window_start"], errors="coerce")
        spdf["window_end"] = pd.to_datetime(spdf["window_end"], errors="coerce")
    if not algo_df.empty:
        algo_df["datetime"] = pd.to_datetime(algo_df["datetime"], errors="coerce")

    color_mapping_path = os.path.join(PROJECT_ROOT, "color_mappings.json")
    repo_colors = _load_repo_color_mapping()
    assets_dir = os.path.join(out_dir, "interactive_review")
    _ensure_dir(assets_dir)

    def _centered_time_range(start_ts, end_ts, hours):
        start_ts = pd.Timestamp(start_ts)
        end_ts = pd.Timestamp(end_ts)
        window = pd.Timedelta(hours=float(hours))
        if end_ts <= start_ts:
            return (start_ts, end_ts)
        total = end_ts - start_ts
        if total <= window:
            return (start_ts, end_ts)
        center = start_ts + (total / 2)
        half = window / 2
        win_start = center - half
        win_end = center + half
        if win_start < start_ts:
            win_start = start_ts
            win_end = min(start_ts + window, end_ts)
        if win_end > end_ts:
            win_end = end_ts
            win_start = max(end_ts - window, start_ts)
        return (win_start, win_end)

    def _build_plot_channels(data_pkl):
        signal_data = getattr(data_pkl, "signal_data", {}) or {}
        plot_channels = {}
        for sig, req_channels in requested_channel_map.items():
            if sig not in signal_data:
                continue
            sdf = signal_data[sig]
            if "datetime" not in sdf.columns:
                continue
            present = [c for c in req_channels if c in sdf.columns]
            if present:
                plot_channels[sig] = present
        algo_sig = "algorithmic_intermediate_channels"
        algo_default_channels = ["depth_std_m", "depth_d1_ms", "depth_d2_ms2"]
        if algo_sig in signal_data:
            sdf = signal_data[algo_sig]
            if "datetime" in sdf.columns:
                present = [c for c in algo_default_channels if c in sdf.columns]
                if present:
                    plot_channels[algo_sig] = present
        if not plot_channels:
            for sig, sdf in signal_data.items():
                if "datetime" not in sdf.columns:
                    continue
                candidate_channels = [c for c in sdf.columns if c != "datetime"]
                if candidate_channels:
                    plot_channels[sig] = [candidate_channels[0]]
                    break
        return plot_channels

    def _method_events(dep_cdf: pd.DataFrame, dataset_id: str, deployment_id: str, method_kind: str, method_id: str):
        if method_kind == "cluster":
            sub = dep_cdf[dep_cdf["cluster_pass"].astype(str) == str(method_id)].copy()
            if sub.empty:
                return pd.DataFrame(), {}
            sub["event_key"] = sub["cluster_rank"].map(lambda x: f"{method_id}: cluster {int(x)}" if pd.notna(x) else pd.NA)
            event_df = pd.DataFrame({
                "datetime": pd.to_datetime(sub["window_start"], errors="coerce"),
                "end_datetime": pd.to_datetime(sub["window_end"], errors="coerce"),
                "key": sub["event_key"].astype(str),
                "short_description": str(method_id),
                "type": "state",
                "duration": (pd.to_datetime(sub["window_end"], errors="coerce") - pd.to_datetime(sub["window_start"], errors="coerce")).dt.total_seconds(),
            }).dropna(subset=["datetime", "end_datetime", "key"])
            keys = sorted(event_df["key"].dropna().astype(str).unique().tolist())
            cmap = build_ordered_cluster_color_map(keys)
            annotations = {k: {"signal": "all", "color": cmap.get(k), "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False} for k in keys}
            return event_df.sort_values("datetime").reset_index(drop=True), annotations
        if method_kind == "algorithmic":
            dep_algo = algo_df[(algo_df["dataset_id"] == dataset_id) & (algo_df["deployment_id"] == deployment_id)].copy()
            if dep_algo.empty:
                return pd.DataFrame(), {}
            dep_algo = dep_algo.dropna(subset=["datetime", "algorithmic_label_name"]).sort_values("datetime")
            dep_algo["next_datetime"] = dep_algo["datetime"].shift(-1)
            fallback_step = dep_algo["datetime"].diff().median() if len(dep_algo) > 1 else pd.Timedelta(seconds=30)
            dep_algo["end_datetime"] = dep_algo["next_datetime"].fillna(dep_algo["datetime"] + fallback_step)
            event_df = dep_algo.rename(columns={"algorithmic_label_name": "key"})[["datetime", "end_datetime", "key"]].copy()
            event_df["short_description"] = str(method_id)
            event_df["type"] = "state"
            event_df["duration"] = (event_df["end_datetime"] - event_df["datetime"]).dt.total_seconds()
            keys = sorted(event_df["key"].dropna().astype(str).unique().tolist())
            annotations = {k: {"signal": "all", "color": repo_colors.get(k), "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False} for k in keys}
            return event_df.sort_values("datetime").reset_index(drop=True), annotations
        dep_spdf = spdf[
            (spdf["dataset_id"] == dataset_id)
            & (spdf["deployment_id"] == deployment_id)
            & (spdf["variant_id"].astype(str) == str(method_id))
        ].copy()
        if dep_spdf.empty:
            return pd.DataFrame(), {}
        event_df, annotations, _positive_label = _build_supervised_confusion_overlay(dep_spdf)
        if event_df.empty:
            return pd.DataFrame(), {}
        event_df["short_description"] = str(method_id)
        return event_df.sort_values("datetime").reset_index(drop=True), annotations

    manifest = []
    dep_rows = cdf[["dataset_id", "deployment_id"]].drop_duplicates().sort_values(["dataset_id", "deployment_id"])
    if deployment_scope == "first_per_dataset":
        dep_rows = dep_rows.groupby("dataset_id", as_index=False).head(1)
    elif deployment_scope == "first_only":
        dep_rows = dep_rows.head(1)
    for row in dep_rows.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        deployment_id = str(row.deployment_id)
        pkl_path = _data_pkl_path(ctx, dataset_id, deployment_id)
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            base_data_pkl = pickle.load(f)
        deployment_tz = _deployment_timezone_name(base_data_pkl)
        dep_cdf = cdf[(cdf["dataset_id"] == dataset_id) & (cdf["deployment_id"] == deployment_id)].copy()
        if dep_cdf.empty:
            continue
        dep_cdf = _normalize_window_columns(dep_cdf, deployment_tz)
        t0 = dep_cdf["window_start"].min()
        t1 = dep_cdf["window_end"].max()
        if pd.isna(t0) or pd.isna(t1):
            continue
        if time_range_mode == "full_deployment":
            time_range = (
                _normalize_datetime_series_to_timezone(pd.Series([t0]), deployment_tz).iloc[0],
                _normalize_datetime_series_to_timezone(pd.Series([t1]), deployment_tz).iloc[0],
            )
        else:
            time_range = _centered_time_range(t0, t1, preview_hours)

        methods = []
        if {"cluster_pass", "cluster_rank"}.issubset(dep_cdf.columns):
            for cluster_pass in sorted(dep_cdf["cluster_pass"].dropna().astype(str).unique().tolist()):
                methods.append(("cluster", cluster_pass, f"{cluster_pass.upper()} clusters"))
        if not algo_df.empty and not algo_df[(algo_df["dataset_id"] == dataset_id) & (algo_df["deployment_id"] == deployment_id)].empty:
            algo_name = str(
                algo_df.loc[(algo_df["dataset_id"] == dataset_id) & (algo_df["deployment_id"] == deployment_id), "algorithmic_method"]
                .dropna()
                .astype(str)
                .iloc[0]
            ) if "algorithmic_method" in algo_df.columns else "algorithmic"
            methods.append(("algorithmic", algo_name, f"Algorithmic ({algo_name})"))
        if not spdf.empty:
            dep_spdf = spdf[(spdf["dataset_id"] == dataset_id) & (spdf["deployment_id"] == deployment_id)].copy()
            if not dep_spdf.empty and "variant_id" in dep_spdf.columns:
                meta = dep_spdf[["variant_id", "variant_label", "variant_rank"]].drop_duplicates().sort_values(["variant_rank", "variant_label"])
                for m in meta.itertuples(index=False):
                    methods.append(("supervised", str(m.variant_id), str(m.variant_label)))

        for method_kind, method_id, method_label in methods:
            try:
                with open(pkl_path, "rb") as f:
                    data_pkl = pickle.load(f)
                signal_data = getattr(data_pkl, "signal_data", {}) or {}
                for sig_name, sdf in list(signal_data.items()):
                    if isinstance(sdf, pd.DataFrame) and "datetime" in sdf.columns:
                        sdf_local = sdf.copy()
                        sdf_local["datetime"] = _normalize_datetime_series_to_timezone(sdf_local["datetime"], deployment_tz)
                        signal_data[sig_name] = sdf_local
                data_pkl.signal_data = signal_data
                plot_channels = _build_plot_channels(data_pkl)
                if not plot_channels:
                    continue
                plot_signals = list(plot_channels.keys())
                zoom_signal = primary_signal if primary_signal in plot_channels else plot_signals[0]
                event_df, state_annotations = _method_events(dep_cdf, dataset_id, deployment_id, method_kind, method_id)
                if event_df.empty:
                    continue
                if "datetime" in event_df.columns:
                    event_df["datetime"] = _normalize_datetime_series_to_timezone(event_df["datetime"], deployment_tz)
                if "end_datetime" in event_df.columns:
                    event_df["end_datetime"] = _normalize_datetime_series_to_timezone(event_df["end_datetime"], deployment_tz)
                base_events = getattr(data_pkl, "event_data", None)
                if base_events is None or base_events.empty:
                    data_pkl.event_data = event_df.copy()
                else:
                    base_events = base_events.copy()
                    if "datetime" in base_events.columns:
                        base_events["datetime"] = _normalize_datetime_series_to_timezone(base_events["datetime"], deployment_tz)
                    for end_col in ("end_datetime", "end_time", "datetime_end", "end"):
                        if end_col in base_events.columns:
                            base_events[end_col] = _normalize_datetime_series_to_timezone(base_events[end_col], deployment_tz)
                    if "key" in base_events.columns:
                        keep = ~base_events["key"].astype(str).isin(set(event_df["key"].astype(str)))
                        data_pkl.event_data = pd.concat([base_events.loc[keep].copy(), event_df], ignore_index=True, sort=False)
                    else:
                        data_pkl.event_data = pd.concat([base_events.copy(), event_df], ignore_index=True, sort=False)
                    data_pkl.event_data = data_pkl.event_data.sort_values("datetime").reset_index(drop=True)
                positive_label = None
                if method_kind == "supervised":
                    dep_spdf = spdf[
                        (spdf["dataset_id"] == dataset_id)
                        & (spdf["deployment_id"] == deployment_id)
                        & (spdf["variant_id"].astype(str) == str(method_id))
                    ].copy()
                    _, _, positive_label = _build_supervised_confusion_overlay(dep_spdf)
                fig = plot_tag_data_interactive(
                    data_pkl=data_pkl,
                    signals=plot_signals,
                    channels=plot_channels,
                    time_range=time_range,
                    state_annotations=state_annotations,
                    target_sampling_rate=2,
                    zoom_range_selector_channel=None,
                    color_mapping_path=color_mapping_path,
                )
                fig = _restrict_interactive_review_legend(fig)
                if time_range_mode == "full_deployment":
                    review_label = "full deployment"
                else:
                    review_label = f"centered {int(preview_hours)} h"
                title_extra = (
                    _interactive_state_fill_legend_html(state_annotations, positive_label=positive_label)
                    + _interactive_confusion_counts_html(event_df)
                    if method_kind == "supervised"
                    else ""
                )
                fig.update_layout(
                    title=f"{deployment_id} | {method_label} | {review_label}{title_extra}",
                    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
                )
                safe_name = f"{_safe_slug(dataset_id)}__{_safe_slug(deployment_id)}__{_safe_slug(method_kind)}__{_safe_slug(method_id)}.html"
                full_path = os.path.join(assets_dir, safe_name)
                fig.write_html(full_path, include_plotlyjs="cdn", full_html=True)
                manifest.append(
                    {
                        "dataset_id": dataset_id,
                        "deployment_id": deployment_id,
                        "deployment_key": f"{dataset_id} / {deployment_id}",
                        "method_kind": method_kind,
                        "method_id": str(method_id),
                        "method_label": method_label,
                        "relative_path": os.path.join("interactive_review", safe_name),
                    }
                )
            except Exception as e:
                print(f"[plots] warning: could not write embedded interactive review for {dataset_id}/{deployment_id} {method_kind}:{method_id}: {e}")
                continue

    if not manifest:
        return None
    manifest_path = os.path.join(out_dir, "interactive_review_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[plots] wrote {len(manifest)} embedded interactive review view(s)")
    return manifest_path


def cmd_plots(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    cdf = pd.read_parquet(os.path.join(ctx.output_root, "clustering", "clustered_windows.parquet"))
    cdf = _localize_window_columns_by_deployment(ctx, cdf)
    algorithmic_segments_path = os.path.join(ctx.output_root, "segments", "algorithmic_segments.parquet")
    algo_seg_df = pd.read_parquet(algorithmic_segments_path) if os.path.exists(algorithmic_segments_path) else pd.DataFrame()
    algorithmic_exhaustive_segments_path = os.path.join(ctx.output_root, "segments", "algorithmic_segments_exhaustive.parquet")
    algo_exhaustive_seg_df = (
        pd.read_parquet(algorithmic_exhaustive_segments_path)
        if os.path.exists(algorithmic_exhaustive_segments_path)
        else pd.DataFrame()
    )
    out_dir = os.path.join(ctx.output_root, "plots")
    _ensure_dir(out_dir)
    interactive_cfg = _parse_interactive_review_cfg(ctx.run_cfg)
    if bool(getattr(args, "skip_interactive_review", False)):
        interactive_cfg = dict(interactive_cfg)
        interactive_cfg["enabled"] = False
        print("[plots] skipping interactive review assets (skip_interactive_review=True)")
    from pyologger.plot_data.plotter import plot_activity_budget_stacked

    spec, standardized_cfg = _normalize_spec_for_processing(ctx.run_cfg)
    primary = ctx.run_cfg.get("primary_input_channel_for_ranking") or next(iter(spec.keys()))
    primary_sig, primary_ch = primary.split(".", 1)
    psig, pch = _resolve_spec_source_for_qc(primary, {"signal_id": primary_sig, "channel_id": primary_ch}, standardized_cfg)
    requested_channel_map = {}
    for full_key, scfg in spec.items():
        sig, ch = _resolve_spec_source_for_qc(full_key, scfg, standardized_cfg)
        requested_channel_map.setdefault(sig, [])
        if ch not in requested_channel_map[sig]:
            requested_channel_map[sig].append(ch)
    requested_channel_map.setdefault(
        "algorithmic_intermediate_channels",
        ["depth_std_m", "depth_d1_ms", "depth_d2_ms2"],
    )
    interactive_default_note = (
        "full deployment view"
        if interactive_cfg["time_range_mode"] == "full_deployment"
        else f"centered {interactive_cfg['preview_hours']:g}-hour review chunk"
    )

    interactive_manifest_path = None

    # A: optional interactive overlay assets
    first_dep = cdf[["dataset_id", "deployment_id"]].drop_duplicates().iloc[0]
    ds = first_dep["dataset_id"]
    dep = first_dep["deployment_id"]
    pkl_path = _data_pkl_path(ctx, ds, dep)
    if interactive_cfg["enabled"] and os.path.exists(pkl_path):
        sub = cdf[(cdf["dataset_id"] == ds) & (cdf["deployment_id"] == dep)].copy()
        with open(pkl_path, "rb") as f:
            data_pkl = pickle.load(f)
        deployment_tz = _deployment_timezone_name(data_pkl)
        sub = _normalize_window_columns(sub, deployment_tz)
        key_col = "cluster_key_pca"
        if key_col not in sub.columns:
            key_candidates = [c for c in sub.columns if c.startswith("cluster_key_")]
            key_col = key_candidates[0] if key_candidates else None

        clustered_path = os.path.join(ctx.output_root, "clustering", "clustered_windows.parquet")
        if key_col:
            cluster_keys = sorted(sub[key_col].dropna().astype(str).unique().tolist())
        else:
            cluster_keys = sorted(sub["cluster_rank"].dropna().astype(int).astype(str).unique().tolist())

        nb_path = os.path.join(out_dir, "A_interactive_deployment_overlay.ipynb")
        deployment_records = []
        for row in cdf[["dataset_id", "deployment_id"]].drop_duplicates().sort_values(["dataset_id", "deployment_id"]).itertuples(index=False):
            deployment_records.append(
                {
                    "dataset_id": str(row.dataset_id),
                    "deployment_id": str(row.deployment_id),
                    "pkl_path": _data_pkl_path(ctx, str(row.dataset_id), str(row.deployment_id)),
                }
            )
        supervised_variant_path = os.path.join(ctx.output_root, "supervised", "supervised_variant_predictions.parquet")
        algorithmic_label_path = _algorithmic_label_timeseries_merge_path(ctx)
        markdown_cells = [
            (
                "# Interactive Segmentation Overlay Review\n"
                f"This notebook loads one deployment at a time and defaults to a "
                f"{interactive_default_note} "
                "in `plot_tag_data_interactive` format.\n"
                "Use the dropdowns to switch between deployments and the segmentation methods available for that deployment."
            )
        ]

        code1 = textwrap.dedent(
            f"""
            import pickle
            import numpy as np
            import pandas as pd
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            from pyologger.utils.cluster_colors import build_ordered_cluster_color_map
            from pyologger.plot_data.plotter import plot_tag_data_interactive
            from pyologger.pyologger.io_operations.color_mapping import load_color_mapping

            deployment_records = {json.dumps(deployment_records)}
            primary_signal = {json.dumps(str(psig))}
            primary_channel = {json.dumps(str(pch))}
            requested_channel_map = {json.dumps(requested_channel_map)}
            clustered_path = {json.dumps(clustered_path)}
            supervised_variant_path = {json.dumps(supervised_variant_path)}
            algorithmic_label_path = {json.dumps(algorithmic_label_path)}
            color_mapping_path = {json.dumps(os.path.join(os.getcwd(), "color_mappings.json"))}

            preview_hours = {interactive_cfg["preview_hours"]}
            time_range_mode = {json.dumps(interactive_cfg["time_range_mode"])}
            target_sampling_rate = 2
            """
        ).strip()

        code2 = textwrap.dedent(
            """
            cdf = pd.read_parquet(clustered_path)
            spdf = pd.read_parquet(supervised_variant_path) if pd.io.common.file_exists(supervised_variant_path) else pd.DataFrame()
            algo_df = pd.read_parquet(algorithmic_label_path) if pd.io.common.file_exists(algorithmic_label_path) else pd.DataFrame()
            color_mapping = load_color_mapping(color_mapping_path)

            if not spdf.empty:
                spdf["window_start"] = pd.to_datetime(spdf["window_start"], errors="coerce")
                spdf["window_end"] = pd.to_datetime(spdf["window_end"], errors="coerce")
            if not algo_df.empty:
                algo_df["datetime"] = pd.to_datetime(algo_df["datetime"], errors="coerce")

            deployment_lookup = {f"{r['dataset_id']} / {r['deployment_id']}": r for r in deployment_records}

            def _build_plot_channels(data_pkl):
                signal_data = getattr(data_pkl, "signal_data", {}) or {}
                plot_channels = {}
                for sig, req_channels in requested_channel_map.items():
                    if sig not in signal_data:
                        continue
                    sdf = signal_data[sig]
                    if "datetime" not in sdf.columns:
                        continue
                    present = [c for c in req_channels if c in sdf.columns]
                    if present:
                        plot_channels[sig] = present
                if not plot_channels:
                    for sig, sdf in signal_data.items():
                        if "datetime" not in sdf.columns:
                            continue
                        candidate_channels = [c for c in sdf.columns if c != "datetime"]
                        if candidate_channels:
                            plot_channels[sig] = [candidate_channels[0]]
                            break
                if not plot_channels:
                    raise ValueError("No plottable signal/channel found in data_pkl.signal_data")
                return plot_channels

            def _deployment_methods(dataset_id, deployment_id):
                methods = []
                dep_cdf = cdf[(cdf["dataset_id"] == dataset_id) & (cdf["deployment_id"] == deployment_id)].copy()
                if not dep_cdf.empty and {"cluster_pass", "cluster_rank"}.issubset(dep_cdf.columns):
                    for cluster_pass in sorted(dep_cdf["cluster_pass"].dropna().astype(str).unique().tolist()):
                        methods.append(("cluster", cluster_pass, f"{cluster_pass.upper()} clusters"))
                if not algo_df.empty:
                    dep_algo = algo_df[(algo_df["dataset_id"] == dataset_id) & (algo_df["deployment_id"] == deployment_id)]
                    if not dep_algo.empty:
                        algo_name = str(dep_algo.get("algorithmic_method", pd.Series(["algorithmic"])).dropna().astype(str).iloc[0])
                        methods.append(("algorithmic", algo_name, f"Algorithmic ({algo_name})"))
                if not spdf.empty:
                    dep_spdf = spdf[(spdf["dataset_id"] == dataset_id) & (spdf["deployment_id"] == deployment_id)].copy()
                    if not dep_spdf.empty and "variant_id" in dep_spdf.columns:
                        meta = dep_spdf[["variant_id", "variant_label", "variant_rank"]].drop_duplicates().sort_values(["variant_rank", "variant_label"])
                        for row in meta.itertuples(index=False):
                            methods.append(("supervised", str(row.variant_id), str(row.variant_label)))
                return methods

            def _detect_positive_label(labels):
                labels = [str(x).strip() for x in labels if str(x).strip()]
                for candidate in labels:
                    if candidate.lower() == "sleep":
                        return candidate
                for candidate in labels:
                    if "sleep" in candidate.lower():
                        return candidate
                for preferred in ["REM", "SWS", "SLEEP", "WAKE", "ACTIVE"]:
                    if preferred in labels:
                        return preferred
                return labels[0] if labels else None

            def _confusion_styles():
                defaults = {
                    "true_negative": {"label": "True Negative", "color": "#d9dde3"},
                    "false_positive": {"label": "False Positive", "color": "#f6e7a8"},
                    "false_negative": {"label": "False Negative", "color": "#f3c7c3"},
                    "true_positive": {"label": "True Positive", "color": "#cfe9cf"},
                }
                raw = color_mapping.get("__supervised_confusion_colors__", {})
                out = {}
                for key, default in defaults.items():
                    style = raw.get(key) if isinstance(raw, dict) else None
                    if isinstance(style, dict):
                        out[key] = {
                            "label": str(style.get("label") or default["label"]),
                            "color": str(style.get("color") or default["color"]),
                        }
                    else:
                        out[key] = default.copy()
                return out

            def _confusion_label(key, positive_label):
                positive = str(positive_label).strip() or "positive state"
                negative = f"not {positive}"
                labels = {
                    "true_negative": f"True Negative: observed {negative}, predicted {negative}",
                    "false_positive": f"False Positive: predicted {positive}, observed {negative}",
                    "false_negative": f"False Negative: observed {positive}, predicted {negative}",
                    "true_positive": f"True Positive: observed {positive}, predicted {positive}",
                }
                return labels.get(str(key), str(key).replace("_", " ").title())

            def _build_supervised_confusion_overlay_local(dep_spdf):
                if dep_spdf is None or dep_spdf.empty:
                    return pd.DataFrame(), {}, None
                work = dep_spdf.dropna(subset=["window_start", "window_end", "predicted_label"]).sort_values("window_start").copy()
                if work.empty:
                    return pd.DataFrame(), {}, None
                labels = pd.Index(
                    work.get("observed_label", pd.Series(dtype=object)).dropna().astype(str).tolist()
                    + work["predicted_label"].dropna().astype(str).tolist()
                ).unique().tolist()
                positive_label = _detect_positive_label(labels)
                if positive_label is None or "observed_label" not in work.columns:
                    event_df = work.rename(columns={"window_start": "datetime", "window_end": "end_datetime", "predicted_label": "key"})[["datetime", "end_datetime", "key"]].copy()
                    event_df["type"] = "state"
                    event_df["duration"] = (event_df["end_datetime"] - event_df["datetime"]).dt.total_seconds()
                    keys = sorted(event_df["key"].dropna().astype(str).unique().tolist())
                    annotations = {k: {"signal": "all", "color": color_mapping.get(k), "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False, "name": str(k)} for k in keys}
                    return event_df.sort_values("datetime").reset_index(drop=True), annotations, positive_label

                positive_lower = str(positive_label).strip().lower()
                observed_is_positive = work["observed_label"].astype(str).str.strip().str.lower() == positive_lower
                predicted_is_positive = work["predicted_label"].astype(str).str.strip().str.lower() == positive_lower
                work["key"] = np.where(
                    (~observed_is_positive) & (~predicted_is_positive),
                    "true_negative",
                    np.where(
                        (~observed_is_positive) & predicted_is_positive,
                        "false_positive",
                        np.where(
                            observed_is_positive & (~predicted_is_positive),
                            "false_negative",
                            "true_positive",
                        ),
                    ),
                )
                event_df = work.rename(columns={"window_start": "datetime", "window_end": "end_datetime"})[["datetime", "end_datetime", "key"]].copy()
                event_df["type"] = "state"
                event_df["duration"] = (event_df["end_datetime"] - event_df["datetime"]).dt.total_seconds()
                styles = _confusion_styles()
                annotations = {
                    key: {
                        "signal": "all",
                        "color": styles[key]["color"],
                        "shade_mode": "fill_trace_split",
                        "shade_opacity": 0.28,
                        "draw_line": False,
                        "name": _confusion_label(key, positive_label),
                    }
                    for key in ["true_negative", "false_positive", "false_negative", "true_positive"]
                    if key in set(event_df["key"].astype(str))
                }
                return event_df.sort_values("datetime").reset_index(drop=True), annotations, positive_label

            def _fill_legend_html(state_annotations, positive_label=None):
                if not state_annotations and positive_label is None:
                    return ""
                chunks = []
                styles = _confusion_styles()
                for key in ["true_negative", "false_positive", "false_negative", "true_positive"]:
                    cfg = (state_annotations or {}).get(key) or {}
                    color = str(cfg.get("color") or styles.get(key, {}).get("color") or "#8a99a8")
                    label = str(cfg.get("name") or _confusion_label(key, positive_label))
                    chunks.append(f"<span style=\\"white-space:nowrap;\\"><span style=\\"color:{color};\\">■</span> {label}</span>")
                return "<br><sup>Fill colors: " + " &nbsp;&nbsp; ".join(chunks) + "</sup>" if chunks else ""

            def _confusion_counts_html(event_df):
                counts = (
                    event_df.get("key", pd.Series(dtype=object)).astype(str).value_counts().to_dict()
                    if isinstance(event_df, pd.DataFrame) and not event_df.empty
                    else {}
                )
                text = " | ".join([
                    f"TN: {int(counts.get('true_negative', 0))}",
                    f"FP: {int(counts.get('false_positive', 0))}",
                    f"FN: {int(counts.get('false_negative', 0))}",
                    f"TP: {int(counts.get('true_positive', 0))}",
                ])
                return f"<br><sup>Confusion windows: {text}</sup>"

            def _restrict_interactive_review_legend_local(fig, keep_signal_name="algorithmic_intermediate_channels"):
                if fig is None:
                    return fig
                keep_token = str(keep_signal_name or "").strip()
                any_kept = False
                for trace in getattr(fig, "data", []) or []:
                    trace_name = str(getattr(trace, "name", "") or "")
                    keep = bool(keep_token and keep_token in trace_name)
                    try:
                        trace.showlegend = keep
                    except Exception:
                        pass
                    any_kept = any_kept or keep
                if not any_kept:
                    fig.update_layout(showlegend=False)
                return fig

            def _chunk_ranges(start_ts, end_ts, hours=6, mode="centered_window"):
                if pd.isna(start_ts) or pd.isna(end_ts):
                    return []
                if mode == "full_deployment":
                    return [(pd.Timestamp(start_ts), pd.Timestamp(end_ts))]
                starts = []
                t = pd.Timestamp(start_ts)
                end_ts = pd.Timestamp(end_ts)
                delta = pd.Timedelta(hours=float(hours))
                while t < end_ts:
                    starts.append((t, min(t + delta, end_ts)))
                    t = t + delta
                return starts or [(pd.Timestamp(start_ts), pd.Timestamp(end_ts))]

            def _centered_chunk_index(chunks):
                if not chunks:
                    return None
                return max(0, len(chunks) // 2)

            def _build_method_events(dataset_id, deployment_id, method_kind, method_id, dep_cdf):
                if method_kind == "cluster":
                    sub = dep_cdf[dep_cdf["cluster_pass"].astype(str) == str(method_id)].copy()
                    if sub.empty:
                        return pd.DataFrame(), {}
                    sub["event_key"] = sub["cluster_rank"].map(lambda x: f"{method_id}: cluster {int(x)}" if pd.notna(x) else pd.NA)
                    event_df = pd.DataFrame({
                        "datetime": pd.to_datetime(sub["window_start"], errors="coerce"),
                        "end_datetime": pd.to_datetime(sub["window_end"], errors="coerce"),
                        "key": sub["event_key"].astype(str),
                        "short_description": str(method_id),
                        "type": "state",
                        "duration": (pd.to_datetime(sub["window_end"], errors="coerce") - pd.to_datetime(sub["window_start"], errors="coerce")).dt.total_seconds(),
                    }).dropna(subset=["datetime", "end_datetime", "key"])
                    keys = sorted(event_df["key"].dropna().astype(str).unique().tolist())
                    cmap = build_ordered_cluster_color_map(keys)
                    annotations = {k: {"signal": "all", "color": cmap.get(k), "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False} for k in keys}
                    return event_df.sort_values("datetime").reset_index(drop=True), annotations
                if method_kind == "algorithmic":
                    dep_algo = algo_df[(algo_df["dataset_id"] == dataset_id) & (algo_df["deployment_id"] == deployment_id)].copy()
                    if dep_algo.empty:
                        return pd.DataFrame(), {}
                    dep_algo = dep_algo.dropna(subset=["datetime", "algorithmic_label_name"]).sort_values("datetime")
                    dep_algo["next_datetime"] = dep_algo["datetime"].shift(-1)
                    if len(dep_algo) > 1:
                        fallback_step = dep_algo["datetime"].diff().median()
                    else:
                        fallback_step = pd.Timedelta(seconds=30)
                    dep_algo["end_datetime"] = dep_algo["next_datetime"].fillna(dep_algo["datetime"] + fallback_step)
                    event_df = dep_algo.rename(columns={"algorithmic_label_name": "key"})[["datetime", "end_datetime", "key"]].copy()
                    event_df["short_description"] = str(method_id)
                    event_df["type"] = "state"
                    event_df["duration"] = (event_df["end_datetime"] - event_df["datetime"]).dt.total_seconds()
                    keys = sorted(event_df["key"].dropna().astype(str).unique().tolist())
                    annotations = {k: {"signal": "all", "color": color_mapping.get(k), "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False} for k in keys}
                    return event_df.sort_values("datetime").reset_index(drop=True), annotations
                dep_spdf = spdf[(spdf["dataset_id"] == dataset_id) & (spdf["deployment_id"] == deployment_id) & (spdf["variant_id"].astype(str) == str(method_id))].copy()
                if dep_spdf.empty:
                    return pd.DataFrame(), {}
                event_df, annotations, _positive_label = _build_supervised_confusion_overlay_local(dep_spdf)
                if event_df.empty:
                    return pd.DataFrame(), {}
                event_df["short_description"] = str(method_id)
                return event_df.sort_values("datetime").reset_index(drop=True), annotations
            """
        ).strip()

        code3 = textwrap.dedent(
            """
            deployment_dropdown = widgets.Dropdown(options=list(deployment_lookup.keys()), description="Deployment", layout=widgets.Layout(width="500px"))
            method_dropdown = widgets.Dropdown(description="Method", layout=widgets.Layout(width="360px"))
            chunk_dropdown = widgets.Dropdown(description="Time range", layout=widgets.Layout(width="280px"))
            output = widgets.Output()

            def refresh_methods(*_):
                record = deployment_lookup[deployment_dropdown.value]
                methods = _deployment_methods(record["dataset_id"], record["deployment_id"])
                method_dropdown.options = [(label, f"{kind}::{mid}") for kind, mid, label in methods]
                if methods:
                    method_dropdown.value = f"{methods[0][0]}::{methods[0][1]}"
                refresh_chunks()

            def refresh_chunks(*_):
                record = deployment_lookup[deployment_dropdown.value]
                dep_cdf = cdf[(cdf["dataset_id"] == record["dataset_id"]) & (cdf["deployment_id"] == record["deployment_id"])].copy()
                t0 = pd.to_datetime(dep_cdf["window_start"], errors="coerce").min()
                t1 = pd.to_datetime(dep_cdf["window_end"], errors="coerce").max()
                chunks = _chunk_ranges(t0, t1, hours=preview_hours, mode=time_range_mode)
                if time_range_mode == "full_deployment":
                    chunk_dropdown.options = [("Full deployment", 0)] if chunks else []
                else:
                    chunk_dropdown.options = [(f"{i + 1}: {a.strftime('%Y-%m-%d %H:%M')} to {b.strftime('%H:%M')}", i) for i, (a, b) in enumerate(chunks)]
                if chunks:
                    chunk_dropdown.value = _centered_chunk_index(chunks)
                render_plot()

            def render_plot(*_):
                with output:
                    clear_output(wait=True)
                    record = deployment_lookup[deployment_dropdown.value]
                    dep_cdf = cdf[(cdf["dataset_id"] == record["dataset_id"]) & (cdf["deployment_id"] == record["deployment_id"])].copy()
                    if dep_cdf.empty:
                        print("No clustered windows for selected deployment.")
                        return
                    with open(record["pkl_path"], "rb") as f:
                        data_pkl = pickle.load(f)
                    plot_channels = _build_plot_channels(data_pkl)
                    plot_signals = list(plot_channels.keys())
                    zoom_signal = primary_signal if primary_signal in plot_channels else plot_signals[0]
                    method_kind, method_id = method_dropdown.value.split("::", 1)
                    event_df, state_annotations = _build_method_events(record["dataset_id"], record["deployment_id"], method_kind, method_id, dep_cdf)
                    positive_label = None
                    if method_kind == "supervised":
                        dep_spdf = spdf[
                            (spdf["dataset_id"] == record["dataset_id"])
                            & (spdf["deployment_id"] == record["deployment_id"])
                            & (spdf["variant_id"].astype(str) == str(method_id))
                        ].copy()
                        _, _, positive_label = _build_supervised_confusion_overlay_local(dep_spdf)
                    base_events = getattr(data_pkl, "event_data", None)
                    if base_events is None or base_events.empty:
                        data_pkl.event_data = event_df.copy()
                    else:
                        if "key" in base_events.columns:
                            keep = ~base_events["key"].astype(str).isin(set(event_df["key"].astype(str)))
                            data_pkl.event_data = pd.concat([base_events.loc[keep].copy(), event_df], ignore_index=True, sort=False)
                        else:
                            data_pkl.event_data = pd.concat([base_events.copy(), event_df], ignore_index=True, sort=False)
                    data_pkl.event_data = data_pkl.event_data.sort_values("datetime").reset_index(drop=True)
                    chunks = _chunk_ranges(
                        pd.to_datetime(dep_cdf["window_start"], errors="coerce").min(),
                        pd.to_datetime(dep_cdf["window_end"], errors="coerce").max(),
                        hours=preview_hours,
                        mode=time_range_mode,
                    )
                    if not chunks:
                        print("No valid time range for selected deployment.")
                        return
                    chunk_idx = int(chunk_dropdown.value or 0)
                    chunk_idx = min(max(chunk_idx, 0), len(chunks) - 1)
                    time_range = chunks[chunk_idx]
                    method_label = dict(method_dropdown.options).get(method_dropdown.value, method_dropdown.value)
                    fig = plot_tag_data_interactive(
                        data_pkl=data_pkl,
                        signals=plot_signals,
                        channels=plot_channels,
                        time_range=time_range,
                        state_annotations=state_annotations,
                        target_sampling_rate=target_sampling_rate,
                        zoom_range_selector_channel=None,
                        color_mapping_path=color_mapping_path,
                    )
                    fig = _restrict_interactive_review_legend_local(fig)
                    title_suffix = "full deployment" if time_range_mode == "full_deployment" else f"chunk {chunk_idx + 1}"
                    title_extra = (
                        _fill_legend_html(state_annotations, positive_label=positive_label)
                        + _confusion_counts_html(event_df)
                        if method_kind == "supervised"
                        else ""
                    )
                    fig.update_layout(
                        title=f"{record['deployment_id']} | {method_label} | {title_suffix}{title_extra}",
                        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.0),
                    )
                    display(fig)

            deployment_dropdown.observe(refresh_methods, names="value")
            method_dropdown.observe(refresh_chunks, names="value")
            chunk_dropdown.observe(render_plot, names="value")
            refresh_methods()
            display(widgets.VBox([widgets.HBox([deployment_dropdown, method_dropdown, chunk_dropdown]), output]))
            """
        ).strip()

        _write_simple_notebook(
            path=nb_path,
            code_cells=[code1, code2, code3],
            markdown_cells=markdown_cells,
        )
        print(f"[plots] wrote notebook {nb_path}")
        interactive_manifest_path = _write_embedded_interactive_review_assets(
            ctx=ctx,
            out_dir=out_dir,
            cdf=cdf,
            requested_channel_map=requested_channel_map,
            primary_signal=str(psig),
            primary_channel=str(pch),
            preview_hours=interactive_cfg["preview_hours"],
            deployment_scope=interactive_cfg["deployment_scope"],
            time_range_mode=interactive_cfg["time_range_mode"],
        )

    # B/C: Improved actigraphy-like plots using matplotlib actograms
    work = cdf.copy()
    work["window_start"] = pd.to_datetime(work["window_start"], errors="coerce")
    work["window_end"] = pd.to_datetime(work["window_end"], errors="coerce")
    work = work.dropna(subset=["window_start", "window_end"])\
             .sort_values(["deployment_id", "window_start"]).reset_index(drop=True)
    
    if not work.empty:
        # Build hourly behavior data for actogram plotting
        work["day_idx"] = work.groupby("deployment_id")["window_start"].transform(
            lambda s: (s.dt.floor("D") - s.dt.floor("D").min()).dt.days
        )
        work["hour"] = work["window_start"].dt.hour + work["window_start"].dt.minute / 60.0
        work["date_local"] = work["window_start"].dt.date
        work["hour_of_day"] = work["window_start"].dt.hour
        work["cluster"] = work["cluster_rank"].astype(str)
        work["duration_h"] = (work["window_end"] - work["window_start"]).dt.total_seconds() / 3600.0
        
        # Aggregate to hourly proportions per deployment/day/behavior
        hourly_agg = work.groupby(
            ["deployment_id", "date_local", "hour_of_day", "cluster"], 
            as_index=False
        )["duration_h"].sum()
        hourly_totals = work.groupby(
            ["deployment_id", "date_local", "hour_of_day"], 
            as_index=False
        )["duration_h"].sum().rename(columns={"duration_h": "total_h"})
        hourly_agg = hourly_agg.merge(hourly_totals, on=["deployment_id", "date_local", "hour_of_day"])
        hourly_agg["pct_of_observed_hour"] = 100.0 * hourly_agg["duration_h"] / hourly_agg["total_h"].replace(0, np.nan)
        
        # Pivot to wide format for actogram function
        hourly_behavior_df = hourly_agg.pivot_table(
            index=["deployment_id", "date_local", "hour_of_day"],
            columns="cluster",
            values="pct_of_observed_hour",
            fill_value=0.0
        ).reset_index()
        
        # Get deployment locations from metadata or use defaults
        deployment_lat_lon = {}
        timezone_map = {}
        for dep_id in hourly_behavior_df["deployment_id"].unique():
            # Try to get from config or deployment scope
            pairs = _clustering_scope_pairs(ctx)
            matching = [p for p in pairs if p[1] == dep_id]
            if matching:
                dataset_id = matching[0][0]
                # Default locations for common datasets (can be improved with metadata lookup)
                if "kenya" in dataset_id.lower() or "pale" in dataset_id.lower():
                    deployment_lat_lon[dep_id] = (-1.3, 36.8)  # Kenya approx
                    timezone_map[dep_id] = "Africa/Nairobi"
                elif "kruger" in dataset_id.lower() or "africa" in dataset_id.lower():
                    deployment_lat_lon[dep_id] = (-24.0, 31.5)  # Kruger approx
                    timezone_map[dep_id] = "Africa/Johannesburg"
                else:
                    deployment_lat_lon[dep_id] = (0.0, 0.0)
                    timezone_map[dep_id] = "UTC"
        
        # Build behavior colors from cluster color map
        cluster_keys = sorted(hourly_behavior_df.columns.difference(["deployment_id", "date_local", "hour_of_day"]))
        from pyologger.utils.cluster_colors import build_ordered_cluster_color_map
        cluster_colors = build_ordered_cluster_color_map(cluster_keys)
        
        # Generate improved matplotlib actogram (B_improved)
        try:
            from pyologger.plot_data.plotter import plot_daily_activity_actogram
            import matplotlib.pyplot as plt
            
            fig_b_improved = plot_daily_activity_actogram(
                hourly_behavior_df=hourly_behavior_df,
                hourly_odba_df=None,  # Can add ODBA if available
                deployment_lat_lon_map=deployment_lat_lon,
                timezone_map=timezone_map,
                behavior_order=sorted(cluster_keys),
                behavior_colors=cluster_colors,
                title="Cluster Activity Actograms with Solar Context",
            )
            fig_b_improved.savefig(
                os.path.join(out_dir, "B_activity_actogram.png"), 
                dpi=150, 
                bbox_inches="tight"
            )
            plt.close(fig_b_improved)
            print(f"[plots] saved improved actogram to B_activity_actogram.png")
        except Exception as e:
            print(f"[plots] warning: could not generate improved actogram: {e}")
            # Fall back to original simple scatter plot
            work["dep_day"] = work["deployment_id"].astype(str) + "_Day" + work["day_idx"].astype(int).astype(str)
            fig_b = px.scatter(
                work,
                x="hour",
                y="dep_day",
                color="cluster",
                title="Actigraphy-like Cluster States by Deployment Day",
                opacity=0.75,
            )
            fig_b.update_traces(marker=dict(size=8, symbol="line-ew-open"))
            fig_b.update_layout(template="plotly_white", yaxis_title="Deployment Day", xaxis_title="Hour of Day (0-24)")
            fig_b.write_html(os.path.join(out_dir, "B_actigraphy_alignment.html"))

        # C: hourly stacked proportions (keep original plotly version as interactive alternative)
        tmp = work.copy()
        agg = tmp.groupby(["deployment_id", "date_local", "hour", "cluster"], as_index=False)["duration_h"].sum()
        tmp["dep_day"] = tmp["deployment_id"].astype(str) + "_Day" + tmp["day_idx"].astype(int).astype(str)
        agg = tmp.groupby(["dep_day", "hour", "cluster"], as_index=False)["duration_h"].sum()
        total = agg.groupby(["dep_day", "hour"], as_index=False)["duration_h"].sum().rename(columns={"duration_h": "total_h"})
        agg = agg.merge(total, on=["dep_day", "hour"], how="left")
        agg["prop"] = agg["duration_h"] / agg["total_h"].replace(0, np.nan)

        fig_c = px.area(
            agg,
            x="hour",
            y="prop",
            color="cluster",
            facet_row="dep_day",
            facet_row_spacing=max(0.001, min(0.01, (1.0 / max(2, agg["dep_day"].nunique() - 1)) - 1e-4)),
            title="Hourly Cluster Proportions by Deployment Day",
        )
        fig_c.update_layout(template="plotly_white", yaxis_title="Proportion", xaxis_title="Hour of Day")
        fig_c.write_html(os.path.join(out_dir, "C_hourly_cluster_proportions.html"))

    # D: group summary if groups provided
    groups = ctx.run_cfg.get("groups") or {}
    if groups and not work.empty:
        dep_to_group = {}
        for gname, entries in groups.items():
            for dep in entries:
                dep_to_group[str(dep)] = gname
        grp = work.copy()
        grp["group"] = grp["deployment_id"].astype(str).map(dep_to_group)
        grp = grp.dropna(subset=["group"]).copy()
        if not grp.empty:
            grp["duration_h"] = (grp["window_end"] - grp["window_start"]).dt.total_seconds() / 3600.0
            day_level = grp.groupby(["group", "deployment_id", "day_idx", "cluster"], as_index=False)["duration_h"].sum()
            dep_mean = day_level.groupby(["group", "deployment_id", "cluster"], as_index=False)["duration_h"].mean()
            group_mean = dep_mean.groupby(["group", "cluster"], as_index=False)["duration_h"].mean()
            denom = group_mean.groupby("group")["duration_h"].transform("sum")
            group_mean["prop"] = group_mean["duration_h"] / denom.replace(0, np.nan)

            fig_d = px.bar(group_mean, x="group", y="prop", color="cluster", title="Group Summary Activity Budgets (Deployment-weighted)")
            fig_d.update_layout(template="plotly_white", yaxis_title="Mean Proportion")
            fig_d.write_html(os.path.join(out_dir, "D_group_summary_activity_budget.html"))

    supervised_prediction_variant_path = os.path.join(ctx.output_root, "supervised", "supervised_variant_predictions.parquet")
    supervised_prediction_path = os.path.join(ctx.output_root, "supervised", "supervised_predictions.parquet")
    selected_supervised_path = (
        supervised_prediction_variant_path
        if os.path.exists(supervised_prediction_variant_path)
        else supervised_prediction_path
    )
    summary_result = {}
    primary_daily_actogram_path = None
    primary_continuous_actogram_path = None
    primary_event_timing_actogram_path = None
    spdf = pd.DataFrame()
    if os.path.exists(selected_supervised_path):
        spdf = pd.read_parquet(selected_supervised_path)
        if not spdf.empty and {"deployment_id", "window_start", "window_end", "final_behavior"}.issubset(spdf.columns):
            spdf = _localize_window_columns_by_deployment(ctx, spdf.copy())
            spdf = spdf.dropna(subset=["window_start", "window_end"]).copy()
            spdf["duration_s"] = (spdf["window_end"] - spdf["window_start"]).dt.total_seconds()

            dep_to_group = {}
            for gname, entries in (ctx.run_cfg.get("groups") or {}).items():
                for dep in entries:
                    dep_to_group[str(dep)] = str(gname)
            spdf["group"] = spdf["deployment_id"].astype(str).map(dep_to_group).fillna("Unknown")
            spdf["locality"] = spdf["group"].astype(str).map(
                lambda g: "Kruger National Park, South Africa" if g.startswith("Kruger") else ("Kenya" if g.startswith("Kenya") else g)
            )
            spdf["sex"] = spdf["group"].astype(str).map(
                lambda g: g.rsplit("_", 1)[-1] if "_" in g and g.rsplit("_", 1)[-1] in {"M", "F"} else "Unknown"
            )
            spdf["date_local"] = spdf["window_start"].dt.date
            daily_budget = (
                spdf.groupby(["deployment_id", "locality", "sex", "date_local", "final_behavior"], as_index=False)["duration_s"]
                .sum()
            )
            daily_budget["pct_of_24h"] = 100.0 * daily_budget["duration_s"] / 86400.0
            deployment_budget = (
                daily_budget.groupby(["deployment_id", "locality", "sex", "final_behavior"], as_index=False)["pct_of_24h"]
                .mean()
            )
            if not deployment_budget.empty:
                behavior_order = sorted(deployment_budget["final_behavior"].astype(str).unique().tolist())
                fig_rf_budget = plot_activity_budget_stacked(
                    budget_df=deployment_budget,
                    value_col="pct_of_24h",
                    behavior_col="final_behavior",
                    deployment_col="deployment_id",
                    group_col="locality",
                    metadata_label_col="sex",
                    behavior_order=behavior_order,
                    title="RF Daily Activity Budget by Deployment, Sex, and Locality",
                )
                fig_rf_budget.write_html(os.path.join(out_dir, "F_rf_activity_budget.html"))

            try:
                summary_result = write_segmentation_run_summary_parquets(
                    run_output_root=ctx.output_root,
                    algorithmic_seg_df=algo_seg_df,
                    algorithmic_exhaustive_seg_df=algo_exhaustive_seg_df,
                    supervised_prediction_df=spdf,
                    load_data_pkl_for_deployment=lambda dataset_id, deployment_id: _load_data_pkl(
                        _data_pkl_path(ctx, str(dataset_id), str(deployment_id))
                    ),
                    dive_depth_threshold_m=float(
                        (((ctx.run_cfg.get("algorithmic_segments") or {}).get("thresholds") or {}).get("dive_depth_min_m", 2.0))
                    ),
                    groups=ctx.run_cfg.get("groups") or {},
                )
                print(
                    "[plots] wrote segmentation summary parquets | "
                    f"segmentation={summary_result['segmentation_summary_path']} | "
                    f"putative_rest={summary_result['putative_rest_summary_path']} | "
                    f"daily={summary_result['daily_activity_summary_path']} | "
                    f"overlap={summary_result['putative_rest_overlap_summary_path']} | "
                    f"method_daily={summary_result.get('method_budget_daily_path', 'n/a')}"
                )
            except Exception as e:
                print(f"[plots] warning: could not write segmentation summary parquets before report: {type(e).__name__}: {e}")
                print(traceback.format_exc())

            try:
                method_daily_path = (summary_result or {}).get("method_budget_daily_path") or os.path.join(ctx.output_root, "summary", "method_budget_daily.parquet")
                method_hourly_path = (summary_result or {}).get("method_budget_hourly_path") or os.path.join(ctx.output_root, "summary", "method_budget_hourly.parquet")
                if os.path.exists(method_daily_path) and os.path.exists(method_hourly_path):
                    method_daily = pd.read_parquet(method_daily_path)
                    method_hourly = pd.read_parquet(method_hourly_path)
                    meta_cols = [c for c in ["method", "method_variant", "method_label"] if c in method_daily.columns]
                    method_meta = method_daily[meta_cols].drop_duplicates() if meta_cols else pd.DataFrame()
                    metrics_path = os.path.join(ctx.output_root, "supervised", "metrics_by_variant.parquet")
                    metrics_df = pd.read_parquet(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
                    primary_method, primary_variant = _resolve_primary_budget_method(method_meta, metrics_df, ctx.run_cfg)
                    focus_cfg = _summary_focus_cfg(ctx.run_cfg)
                    primary_hourly = method_hourly[
                        (method_hourly["method"].astype(str) == str(primary_method))
                        & (method_hourly["method_variant"].astype(str) == str(primary_variant))
                    ].copy()
                    if not primary_hourly.empty:
                        from pyologger.plot_data.plotter import (
                            plot_continuous_activity_actogram_from_hourly_budget,
                            plot_continuous_activity_with_event_timing,
                            plot_daily_activity_actogram,
                        )
                        import matplotlib.pyplot as plt

                        behavior_col = "state_label_canonical" if focus_cfg["budget_label_mode"] == "canonical" else "state_label_raw"
                        if focus_cfg["actograms"]["daily"]:
                            plot_frame = (
                                primary_hourly.pivot_table(
                                    index=["deployment_id", "local_date", "hour_of_day"],
                                    columns=behavior_col,
                                    values="pct_of_observed_hour",
                                    aggfunc="sum",
                                    fill_value=0.0,
                                )
                                .reset_index()
                            )
                            dep_lat_lon = {}
                            timezone_map = {}
                            daily_seed = method_daily[
                                (method_daily["method"].astype(str) == str(primary_method))
                                & (method_daily["method_variant"].astype(str) == str(primary_variant))
                            ].copy()
                            for dep_id, dep_sub in daily_seed.groupby("deployment_id", sort=False):
                                lat = pd.to_numeric(dep_sub.get("mean_latitude"), errors="coerce").dropna()
                                lon = pd.to_numeric(dep_sub.get("mean_longitude"), errors="coerce").dropna()
                                dep_lat_lon[str(dep_id)] = (
                                    float(lat.iloc[0]) if not lat.empty else np.nan,
                                    float(lon.iloc[0]) if not lon.empty else np.nan,
                                )
                                tz_vals = dep_sub.get("timezone_name", pd.Series(["UTC"])).astype(str).dropna()
                                timezone_map[str(dep_id)] = str(tz_vals.iloc[0]) if not tz_vals.empty else "UTC"
                            fig_primary_daily = plot_daily_activity_actogram(
                                hourly_behavior_df=plot_frame,
                                deployment_lat_lon_map=dep_lat_lon,
                                timezone_map=timezone_map,
                                behavior_order=sorted([c for c in plot_frame.columns if c not in {"deployment_id", "local_date", "hour_of_day"}]),
                                title=f"Primary method daily actogram: {primary_method}/{primary_variant}",
                            )
                            primary_daily_actogram_path = os.path.join(out_dir, "H_primary_method_daily_actogram.png")
                            fig_primary_daily.savefig(primary_daily_actogram_path, dpi=150, bbox_inches="tight")
                            plt.close(fig_primary_daily)
                        if focus_cfg["actograms"]["continuous"]:
                            fig_primary_cont = plot_continuous_activity_actogram_from_hourly_budget(
                                primary_hourly,
                                behavior_col=behavior_col,
                                title=f"Primary method continuous actogram: {primary_method}/{primary_variant}",
                            )
                            primary_continuous_actogram_path = os.path.join(out_dir, "I_primary_method_continuous_actogram.png")
                            fig_primary_cont.savefig(primary_continuous_actogram_path, dpi=150, bbox_inches="tight")
                            plt.close(fig_primary_cont)
                            seg_summary = (summary_result or {}).get("segmentation_summary", pd.DataFrame())
                            primary_events = pd.DataFrame()
                            if isinstance(seg_summary, pd.DataFrame) and not seg_summary.empty:
                                primary_events = seg_summary[
                                    (seg_summary["method"].astype(str) == str(primary_method))
                                    & (seg_summary["method_variant"].astype(str) == str(primary_variant))
                                ].copy()
                                if "start_datetime" in primary_events.columns:
                                    primary_events["start_datetime"] = pd.to_datetime(primary_events["start_datetime"], errors="coerce")
                                if "end_datetime" in primary_events.columns:
                                    primary_events["end_datetime"] = pd.to_datetime(primary_events["end_datetime"], errors="coerce")
                                if "positive_label" in primary_events.columns:
                                    mask = primary_events["positive_label"].fillna(False).astype(bool)
                                    if mask.any():
                                        primary_events = primary_events.loc[mask].copy()
                                    elif "state_subtype" in primary_events.columns:
                                        primary_events = primary_events.loc[primary_events["state_subtype"].notna()].copy()
                                elif "state_subtype" in primary_events.columns:
                                    primary_events = primary_events.loc[primary_events["state_subtype"].notna()].copy()
                            if not primary_events.empty:
                                fig_primary_events = plot_continuous_activity_with_event_timing(
                                    primary_hourly,
                                    primary_events,
                                    behavior_col=behavior_col,
                                    event_label_col="state_label",
                                    title=f"Primary method event timing overlay: {primary_method}/{primary_variant}",
                                )
                                primary_event_timing_actogram_path = os.path.join(out_dir, "J_primary_method_event_timing_overlay.png")
                                fig_primary_events.savefig(primary_event_timing_actogram_path, dpi=150, bbox_inches="tight")
                                plt.close(fig_primary_events)
            except Exception as e:
                print(f"[plots] warning: could not build primary method actograms: {type(e).__name__}: {e}")

            supervised_actogram_paths = _write_supervised_actogram_comparison_plots(ctx, spdf, out_dir)
            if supervised_actogram_paths:
                pd.DataFrame(
                    {
                        "deployment_id": [Path(path).stem.replace("G_supervised_actogram_comparison_", "") for path in supervised_actogram_paths],
                        "plot_path": supervised_actogram_paths,
                    }
                ).to_parquet(os.path.join(out_dir, "G_supervised_actogram_index.parquet"), index=False)
                print(f"[plots] wrote {len(supervised_actogram_paths)} supervised actogram comparison plot(s)")
            print("[plots] building supervised review report (large runs can take several minutes)")
            report_path = _write_supervised_review_report(
                ctx,
                out_dir,
                cdf,
                spdf,
                supervised_actogram_paths,
                primary_daily_actogram_path=primary_daily_actogram_path,
                primary_continuous_actogram_path=primary_continuous_actogram_path,
                primary_event_timing_actogram_path=primary_event_timing_actogram_path,
            )
            if report_path:
                print(f"[plots] wrote supervised review report {report_path}")
                for filename in [
                    "B_activity_actogram.png",
                    "B_actigraphy_alignment.html",
                    "C_hourly_cluster_proportions.html",
                    "D_group_summary_activity_budget.html",
                    "E_algorithmic_to_cluster_sankey.html",
                    "F_rf_activity_budget.html",
                    "G_supervised_actogram_index.parquet",
                ]:
                    stale_path = os.path.join(out_dir, filename)
                    if os.path.exists(stale_path):
                        try:
                            os.remove(stale_path)
                        except Exception as e:
                            print(f"[plots] warning: could not remove stale standalone plot artifact {stale_path}: {e}")
    else:
        report_path = _write_interactive_review_only_report(
            ctx=ctx,
            out_dir=out_dir,
            cdf=cdf,
            manifest_path=interactive_manifest_path,
        )
        if report_path:
            print(f"[plots] wrote interactive review report {report_path}")

    if not summary_result:
        try:
            summary_result = write_segmentation_run_summary_parquets(
                run_output_root=ctx.output_root,
                algorithmic_seg_df=algo_seg_df,
                algorithmic_exhaustive_seg_df=algo_exhaustive_seg_df,
                supervised_prediction_df=spdf,
                load_data_pkl_for_deployment=lambda dataset_id, deployment_id: _load_data_pkl(
                    _data_pkl_path(ctx, str(dataset_id), str(deployment_id))
                ),
                dive_depth_threshold_m=float(
                    (((ctx.run_cfg.get("algorithmic_segments") or {}).get("thresholds") or {}).get("dive_depth_min_m", 2.0))
                ),
                groups=ctx.run_cfg.get("groups") or {},
            )
            print(
                "[plots] wrote segmentation summary parquets | "
                f"segmentation={summary_result['segmentation_summary_path']} | "
                f"putative_rest={summary_result['putative_rest_summary_path']} | "
                f"daily={summary_result['daily_activity_summary_path']} | "
                f"overlap={summary_result['putative_rest_overlap_summary_path']} | "
                f"method_daily={summary_result.get('method_budget_daily_path', 'n/a')}"
            )
        except Exception as e:
            print(f"[plots] warning: could not write segmentation summary parquets: {type(e).__name__}: {e}")
            print(traceback.format_exc())

    # E: algorithmic labels to clusters Sankey
    if "algorithmic_label_name" in cdf.columns and "cluster_rank" in cdf.columns:
        sankey_df = cdf.copy()
        sankey_df["algorithmic_label_name"] = sankey_df["algorithmic_label_name"].fillna("find_rest.not_sleep").astype(str)
        sankey_df["cluster_pass"] = sankey_df.get("cluster_pass", "cluster").astype(str)
        sankey_df["cluster_rank"] = pd.to_numeric(sankey_df["cluster_rank"], errors="coerce")
        sankey_df = sankey_df.dropna(subset=["cluster_rank"])
        sankey_df["cluster_node"] = sankey_df.apply(
            lambda r: f"{r['cluster_pass']}: cluster {int(r['cluster_rank'])}",
            axis=1,
        )
        flow = (
            sankey_df.groupby(["algorithmic_label_name", "cluster_node"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        if not flow.empty:
            algo_order = []
            algo_cfg = _parse_algorithmic_segments_cfg(ctx.run_cfg)
            for key in [
                "not_sleep",
                "calm_uw_gliding",
                "active_uw_swimming",
                "resting_surface",
                "surface_sleep",
                "calm_surface_gliding",
                "active_surface_swimming",
                "resting_benthic",
                "long_flat",
                "long_drift",
            ]:
                name = str((algo_cfg.get("label_names") or {}).get(key, f"find_rest.{key}"))
                if name in flow["algorithmic_label_name"].values:
                    algo_order.append(name)
            extra_algo = [x for x in sorted(flow["algorithmic_label_name"].unique().tolist()) if x not in algo_order]
            algo_order.extend(extra_algo)

            pass_order = []
            for p in _cluster_passes_from_run_cfg(ctx.run_cfg):
                pname = _safe_slug(p["name"])
                pass_order.extend(
                    [f"{pname}: cluster {r}" for r in sorted(sankey_df.loc[sankey_df["cluster_pass"] == pname, "cluster_rank"].astype(int).unique().tolist())]
                )
            cluster_nodes = [x for x in pass_order if x in flow["cluster_node"].values]
            cluster_nodes.extend([x for x in sorted(flow["cluster_node"].unique().tolist()) if x not in cluster_nodes])

            labels = algo_order + cluster_nodes
            node_index = {k: i for i, k in enumerate(labels)}
            fig_e = go.Figure(
                go.Sankey(
                    arrangement="snap",
                    node=dict(label=labels, pad=18, thickness=18),
                    link=dict(
                        source=[node_index[s] for s in flow["algorithmic_label_name"]],
                        target=[node_index[t] for t in flow["cluster_node"]],
                        value=flow["count"].astype(float).tolist(),
                    ),
                )
            )
            fig_e.update_layout(
                title="Algorithmic Rest Labels to Cluster Assignments",
                template="plotly_white",
            )
            fig_e.write_html(os.path.join(out_dir, "E_algorithmic_to_cluster_sankey.html"))

    for filename in [
        "B_activity_actogram.png",
        "B_actigraphy_alignment.html",
        "C_hourly_cluster_proportions.html",
        "D_group_summary_activity_budget.html",
        "E_algorithmic_to_cluster_sankey.html",
        "F_rf_activity_budget.html",
        "G_supervised_actogram_index.parquet",
    ]:
        stale_path = os.path.join(out_dir, filename)
        if os.path.exists(stale_path):
            try:
                os.remove(stale_path)
            except Exception as e:
                print(f"[plots] warning: could not remove stale standalone plot artifact {stale_path}: {e}")

    with open(args.output, "w") as f:
        f.write("plots_done=1\n")
    print(f"[plots] wrote marker {args.output}")


def cmd_report_only(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    clustered_path = os.path.join(ctx.output_root, "clustering", "clustered_windows.parquet")
    if not os.path.exists(clustered_path):
        raise FileNotFoundError(f"Missing clustered windows parquet: {clustered_path}")
    cdf = pd.read_parquet(clustered_path)
    out_dir = os.path.join(ctx.output_root, "plots")
    _ensure_dir(out_dir)
    interactive_cfg = _parse_interactive_review_cfg(ctx.run_cfg)

    supervised_variant_path = os.path.join(ctx.output_root, "supervised", "supervised_variant_predictions.parquet")
    supervised_prediction_path = os.path.join(ctx.output_root, "supervised", "supervised_predictions.parquet")
    selected_supervised_path = (
        supervised_variant_path
        if os.path.exists(supervised_variant_path)
        else supervised_prediction_path
    )

    report_path = None
    if os.path.exists(selected_supervised_path):
        spdf = pd.read_parquet(selected_supervised_path)
        if not spdf.empty:
            actogram_paths = sorted(glob.glob(os.path.join(out_dir, "G_supervised_actogram_comparison_*.png")))
            report_path = _write_supervised_review_report(ctx, out_dir, cdf, spdf, actogram_paths)
    if report_path is None:
        report_path = _write_interactive_review_only_report(
            ctx=ctx,
            out_dir=out_dir,
            cdf=cdf,
            manifest_path=(
                os.path.join(out_dir, "interactive_review_manifest.json")
                if interactive_cfg["enabled"]
                else ""
            ),
        )
    if report_path is None:
        raise RuntimeError(
            "Could not write report-only output. Expected supervised predictions or plots/interactive_review_manifest.json."
        )
    print(f"[report-only] wrote report {report_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Dataset-level clustering workflow")
    p.add_argument("--config", required=True, help="Path to pyologger/config.yaml")
    p.add_argument("--run-name", required=True, help="segmentation_runs key")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("resolve")
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_resolve)

    s = sub.add_parser("qc")
    s.add_argument("--output", required=True)
    s.add_argument(
        "--standardized-channel-db",
        required=False,
        help="Optional CSV/Parquet standardized_channel_db path for expected unit checks.",
    )
    s.set_defaults(func=cmd_qc)

    s = sub.add_parser("algorithmic-segments-deployment")
    s.add_argument("--dataset-id", required=True)
    s.add_argument("--deployment-id", required=True)
    s.add_argument("--output-segments", required=True)
    s.add_argument("--output-summary", required=True)
    s.add_argument(
        "--context-filter-pass-mode",
        choices=["full", "measured_only"],
        default=None,
        help=(
            "Optional context-filter execution phase override. "
            "Default is measured_only. "
            "'measured_only' runs only measured context filters; "
            "'full' runs measured + inferred context filters."
        ),
    )
    s.set_defaults(func=cmd_algorithmic_segments_deployment)

    s = sub.add_parser("algorithmic-segments-merge")
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_algorithmic_segments_merge)

    s = sub.add_parser("features")
    s.add_argument("--log-output", required=True)
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("features-deployment")
    s.add_argument("--dataset-id", required=True)
    s.add_argument("--deployment-id", required=True)
    s.add_argument("--output-feature", required=True)
    s.add_argument("--output-index", required=True)
    s.add_argument("--log-output", required=False)
    s.set_defaults(func=cmd_features_deployment)

    s = sub.add_parser("features-merge")
    s.add_argument("--log-output", required=True)
    s.set_defaults(func=cmd_features_merge)

    s = sub.add_parser("correlate")
    s.set_defaults(func=cmd_correlate)

    s = sub.add_parser("cluster")
    s.set_defaults(func=cmd_cluster)

    s = sub.add_parser("export")
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("supervised")
    s.set_defaults(func=cmd_supervised)

    s = sub.add_parser("plots")
    s.add_argument("--output", required=True)
    s.add_argument(
        "--skip-interactive-review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip expensive interactive review plot/notebook generation.",
    )
    s.set_defaults(func=cmd_plots)

    s = sub.add_parser("report-only")
    s.set_defaults(func=cmd_report_only)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
