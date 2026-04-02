from __future__ import annotations

import copy
import importlib
import json
import os
import pickle
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))

for module_name, module in list(sys.modules.items()):
    if not (module_name == "pyologger" or module_name.startswith("pyologger.")):
        continue
    module_file = getattr(module, "__file__", None)
    if module_file and not str(module_file).startswith(str(LOCAL_PACKAGE_ROOT)):
        sys.modules.pop(module_name, None)

importlib.invalidate_caches()

from pyologger.plot_data.plotter import continuous_daily_activity_plot, plot_tag_data_interactive
from pyologger.utils.folder_manager import load_configuration, resolve_segmentation_runs_path
from pyologger.utils.param_manager import ParamManager
from pyologger.utils.state_event_review import (
    DEFAULT_CANDIDATE_RANK,
    DEFAULT_WINDOW_HOURS,
    _humanize_buoyancy_phase,
    build_mask_period_rows,
    build_overlay_event_rows,
    build_review_yaml_snippet_from_algo_cfg,
    candidate_filter_details,
    centered_window_bounds,
    choose_default_candidate,
    cleanup_temp_workspace,
    context_filter_stage,
    default_algorithmic_cfg_from_config,
    default_review_signals,
    discover_algorithmic_review_runs,
    find_run_config_entry,
    load_algorithmic_segments,
    positive_state_names,
    persist_preview_algorithmic_outputs,
    review_full_algorithmic_config_update,
    run_algorithmic_preview_sandbox,
    save_full_algorithmic_config_update,
    selection_options_for_positive_states,
    selector_display_name,
    stage_dropoff_summary,
    summarize_state_filters,
)


def _normalize_datetime_to_tz(values, tz_name: str) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce")
    if not isinstance(dt, pd.Series):
        dt = pd.Series(dt)
    try:
        if getattr(dt.dt, "tz", None) is None:
            return dt.dt.tz_localize(tz_name)
        return dt.dt.tz_convert(tz_name)
    except Exception:
        return dt


def _normalize_segment_datetimes(seg_df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    if seg_df is None or seg_df.empty:
        return seg_df
    out = seg_df.copy()
    for col in ["start_datetime", "end_datetime", "segment_midpoint_datetime", "datetime"]:
        if col in out.columns:
            out[col] = _normalize_datetime_to_tz(out[col], tz_name)
    return out


def _selection_key(analysis_id: str, dataset_id: str, deployment_id: str) -> str:
    return f"{analysis_id}::{dataset_id}::{deployment_id}"


def _activity_budget_mode_key(selection_key: str) -> str:
    return f"state_event_review_activity_budget_mode::{selection_key}"


def _review_phase_key(selection_key: str) -> str:
    return f"state_event_review_phase::{selection_key}"


def _config_for_filter_phase(algo_cfg: dict, review_phase: str | None = None) -> dict:
    algo_out = copy.deepcopy(algo_cfg)
    algo_out["context_filter_pass_mode"] = "full"
    return algo_out


def _run_scoped_dataset_ids(config: dict, run_cfg: dict, data_dir: str) -> list[str]:
    configured = [str(v) for v in list(run_cfg.get("dataset_ids") or []) if str(v).strip()]
    available = []
    for dataset_id in configured:
        dataset_folder = os.path.join(data_dir, dataset_id)
        if os.path.isdir(dataset_folder):
            available.append(dataset_id)
    return sorted(dict.fromkeys(available))


def _run_scoped_deployment_ids(config: dict, run_cfg: dict, dataset_id: str, data_dir: str) -> list[str]:
    configured_run_deployments = [str(v) for v in list(run_cfg.get("deployment_ids") or []) if str(v).strip()]
    dataset_cfg = dict((config.get("datasets") or {}).get(dataset_id) or {})
    dataset_cfg_deployments = []
    for item in list(dataset_cfg.get("deployments") or []):
        if isinstance(item, dict):
            dep_id = str(item.get("deployment_id") or item.get("id") or "").strip()
        else:
            dep_id = str(item).strip()
        if dep_id:
            dataset_cfg_deployments.append(dep_id)

    deployment_folder_ids = [
        dep_id for dep_id in configured_run_deployments
        if os.path.isdir(os.path.join(data_dir, dataset_id, dep_id))
    ]
    if deployment_folder_ids:
        return sorted(dict.fromkeys(deployment_folder_ids))

    if dataset_cfg_deployments:
        return sorted(
            dep_id for dep_id in dict.fromkeys(dataset_cfg_deployments)
            if os.path.isdir(os.path.join(data_dir, dataset_id, dep_id))
        )

    dataset_folder = os.path.join(data_dir, dataset_id)
    if not os.path.isdir(dataset_folder):
        return []
    return sorted(
        d for d in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, d)) and not d.startswith("00_")
    )


def _algorithmic_exhaustive_segments_path(run_info: dict, dataset_id: str, deployment_id: str) -> Path:
    run_dir = Path(str(run_info.get("run_dir") or ""))
    return run_dir / "segments" / "by_deployment" / f"{dataset_id}__{deployment_id}__algorithmic_segments_exhaustive.parquet"


def _load_exhaustive_segments_for_run(
    run_info: dict,
    dataset_id: str,
    deployment_id: str,
    deployment_tz_name: str,
) -> pd.DataFrame:
    path = _algorithmic_exhaustive_segments_path(run_info, dataset_id, deployment_id)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    return _normalize_segment_datetimes(df, deployment_tz_name)


def _exhaustive_budget_table(exhaustive_df: pd.DataFrame, pretty_map: dict[str, str] | None = None) -> pd.DataFrame:
    if exhaustive_df is None or exhaustive_df.empty:
        return pd.DataFrame(columns=["State", "Nominal Class", "Hours", "Percent", "Segments"])
    work = exhaustive_df.copy()
    if "segment_family" in work.columns:
        work = work.loc[work["segment_family"].astype(str) == "exhaustive_partition"].copy()
    if work.empty:
        return pd.DataFrame(columns=["State", "Nominal Class", "Hours", "Percent", "Segments"])
    work["nominal_class"] = work.get("nominal_class", pd.Series(index=work.index, dtype=object)).astype(str)
    work["duration_s"] = pd.to_numeric(work.get("duration_s", pd.Series(index=work.index, dtype=float)), errors="coerce").fillna(0.0)
    grouped = (
        work.groupby("nominal_class", as_index=False)
        .agg(
            duration_s=("duration_s", "sum"),
            segments=("nominal_class", "count"),
        )
    )
    total_s = float(grouped["duration_s"].sum())
    grouped["hours"] = grouped["duration_s"] / 3600.0
    grouped["percent"] = (100.0 * grouped["duration_s"] / total_s) if total_s > 0 else 0.0
    grouped["state"] = grouped["nominal_class"].map(lambda x: str((pretty_map or {}).get(str(x), str(x))))
    preferred_order = [
        "active_uw_swimming",
        "calm_uw_gliding",
        "resting_surface",
        "resting_benthic",
        "long_drift",
        "active_surface_swimming",
        "calm_surface_gliding",
        "unscorable",
    ]
    order_map = {name: idx for idx, name in enumerate(preferred_order)}
    grouped["__order"] = grouped["nominal_class"].map(lambda x: order_map.get(str(x), 999))
    grouped = grouped.sort_values(["__order", "hours"], ascending=[True, False])
    out = grouped.rename(
        columns={
            "state": "State",
            "nominal_class": "Nominal Class",
            "hours": "Hours",
            "percent": "Percent",
            "segments": "Segments",
        }
    )[["State", "Nominal Class", "Hours", "Percent", "Segments"]]
    out["Hours"] = pd.to_numeric(out["Hours"], errors="coerce").round(2)
    out["Percent"] = pd.to_numeric(out["Percent"], errors="coerce").round(1)
    out["Segments"] = pd.to_numeric(out["Segments"], errors="coerce").fillna(0).astype(int)
    return out.reset_index(drop=True)


def _exhaustive_daily_budget_table(exhaustive_df: pd.DataFrame, pretty_map: dict[str, str] | None = None) -> pd.DataFrame:
    if exhaustive_df is None or exhaustive_df.empty:
        return pd.DataFrame(columns=["date", "nominal_class", "state", "hours"])
    work = exhaustive_df.copy()
    if "segment_family" in work.columns:
        work = work.loc[work["segment_family"].astype(str) == "exhaustive_partition"].copy()
    if work.empty:
        return pd.DataFrame(columns=["date", "nominal_class", "state", "hours"])
    rows: list[dict] = []
    for row in work.itertuples(index=False):
        start = pd.to_datetime(getattr(row, "start_datetime", pd.NaT), errors="coerce")
        end = pd.to_datetime(getattr(row, "end_datetime", pd.NaT), errors="coerce")
        nominal = str(getattr(row, "nominal_class", "") or "")
        duration_s = pd.to_numeric(pd.Series([getattr(row, "duration_s", float("nan"))]), errors="coerce").iloc[0]
        if pd.isna(start) or pd.isna(end):
            continue
        if end < start:
            start, end = end, start
        raw_total_s = max(float((end - start).total_seconds()), 1e-9)
        effective_total_s = float(duration_s) if pd.notna(duration_s) and float(duration_s) > 0 else raw_total_s
        current = start
        while current < end:
            next_day = current.normalize() + pd.Timedelta(days=1)
            slice_end = min(end, next_day)
            raw_slice_s = max(float((slice_end - current).total_seconds()), 0.0)
            if raw_slice_s > 0:
                scaled_slice_s = effective_total_s * (raw_slice_s / raw_total_s)
                rows.append(
                    {
                        "date": current.normalize(),
                        "nominal_class": nominal,
                        "hours": scaled_slice_s / 3600.0,
                    }
                )
            current = slice_end
    if not rows:
        return pd.DataFrame(columns=["date", "nominal_class", "state", "hours"])
    out = pd.DataFrame(rows)
    out = out.groupby(["date", "nominal_class"], as_index=False)["hours"].sum()
    out["state"] = out["nominal_class"].map(lambda x: str((pretty_map or {}).get(str(x), str(x))))
    return out.sort_values(["date", "nominal_class"]).reset_index(drop=True)


def _load_deployment_for_state_review(data_dir: str, dataset_id: str, deployment_id: str):
    import streamlit as st

    dataset_folder = os.path.join(data_dir, dataset_id)
    deployment_folder = os.path.join(dataset_folder, deployment_id)
    try:
        animal_id = deployment_id.split("_")[1]
    except IndexError:
        st.error(f"❌ Unable to extract animal ID from deployment ID: {deployment_id}")
        st.stop()

    pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
    if not os.path.exists(pkl_path):
        st.error(f"❌ Data pickle file not found: {pkl_path}")
        st.stop()

    try:
        with open(pkl_path, "rb") as file:
            data_pkl = pickle.load(file)
    except (EOFError, pickle.UnpicklingError, ModuleNotFoundError, AttributeError, ImportError) as exc:
        st.error(
            "❌ Failed to load data.pkl (likely incomplete or corrupted).\n"
            f"Path: {pkl_path}\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            "Run recovery:\n"
            f"python workflows/00_load_data.py --dataset {dataset_id} --deployment {deployment_id}"
        )
        st.stop()

    param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)
    return animal_id, dataset_folder, deployment_folder, data_pkl, param_manager


def _load_saved_segmentation_review_signals(param_manager: ParamManager, available_signals: list[str]) -> list[str]:
    available = [str(sig) for sig in (available_signals or []) if str(sig).strip()]
    if not available:
        return []
    try:
        saved_cfg = (
            param_manager.get_from_config(
                ["segmentation_review_signals", "segmentation_review_signal_order"],
                section="segmentation_settings",
            )
            or {}
        )
    except Exception:
        return []
    saved = saved_cfg.get("segmentation_review_signal_order") or saved_cfg.get("segmentation_review_signals") or []
    if not isinstance(saved, (list, tuple)):
        return []
    normalized: list[str] = []
    for raw_sig in saved:
        sig = str(raw_sig)
        if sig == "algorithmic_derivative_channels":
            for replacement in ["algorithmic_d1_channel", "algorithmic_d2_channel"]:
                if replacement in available and replacement not in normalized:
                    normalized.append(replacement)
            continue
        if sig in available and sig not in normalized:
            normalized.append(sig)
    return normalized


def _load_supervised_predictions_for_run(run_info: dict, dataset_id: str, deployment_id: str) -> pd.DataFrame:
    run_dir = Path(str(run_info.get("run_dir") or ""))
    candidate_paths = [
        run_dir / "supervised" / "supervised_variant_predictions.parquet",
        run_dir / "supervised" / "supervised_predictions.parquet",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if {"dataset_id", "deployment_id"}.issubset(df.columns):
            df = df.loc[
                (df["dataset_id"].astype(str) == str(dataset_id))
                & (df["deployment_id"].astype(str) == str(deployment_id))
            ].copy()
        if not df.empty:
            return df
    return pd.DataFrame()


def _display_value(value):
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ", timespec="seconds")
    return str(value)


def _field_value_df(items: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Field": str(field), "Value": _display_value(value)} for field, value in items]
    )


def _merge_preview_segments_for_trip(baseline_seg_df: pd.DataFrame, active_seg_df: pd.DataFrame) -> pd.DataFrame:
    if baseline_seg_df is None or baseline_seg_df.empty:
        return pd.DataFrame()
    if active_seg_df is None or active_seg_df.empty or "segment_rank" not in baseline_seg_df.columns or "segment_rank" not in active_seg_df.columns:
        return baseline_seg_df.copy()
    baseline = baseline_seg_df.copy()
    active = active_seg_df.copy()
    baseline["_segment_rank_num"] = pd.to_numeric(baseline["segment_rank"], errors="coerce")
    active["_segment_rank_num"] = pd.to_numeric(active["segment_rank"], errors="coerce")
    active = active.dropna(subset=["_segment_rank_num"]).drop_duplicates("_segment_rank_num", keep="last")
    if active.empty:
        return baseline_seg_df.copy()
    active = active.set_index("_segment_rank_num")
    merge_cols = [
        col for col in active.columns
        if col in baseline.columns and col not in {"start_datetime", "end_datetime", "segment_midpoint_datetime", "duration_s", "segment_rank", "_segment_rank_num"}
    ]
    for col in merge_cols:
        baseline.loc[baseline["_segment_rank_num"].isin(active.index), col] = baseline.loc[
            baseline["_segment_rank_num"].isin(active.index), "_segment_rank_num"
        ].map(active[col])
    return baseline.drop(columns=["_segment_rank_num"])


def _lat_lon_source_from_data_pkl(data_pkl):
    deployment_info = getattr(data_pkl, "deployment_info", {}) or {}
    return {
        "Deployment Latitude": deployment_info.get("Deployment Latitude"),
        "Deployment Longitude": deployment_info.get("Deployment Longitude"),
    }


def _review_signal_display_name(signal_name: str) -> str:
    key = str(signal_name or "")
    overrides = {
        "algorithmic_d1_channel": "d'(depth)",
        "algorithmic_d2_channel": "d''(depth)",
        "algorithmic_depth_d1": "d'(depth)",
        "algorithmic_depth_d2": "d''(depth)",
    }
    return overrides.get(key, key)


def _augment_review_signal_data(signal_data: dict, signal_info: dict) -> tuple[dict, dict]:
    augmented_data = copy.deepcopy(signal_data or {})
    augmented_info = copy.deepcopy(signal_info or {})
    derivative_df = augmented_data.get("algorithmic_derivative_channels")
    if isinstance(derivative_df, pd.DataFrame):
        if "depth_d1_ms" in derivative_df.columns:
            d1_df = derivative_df[["datetime", "depth_d1_ms"]].copy()
            augmented_data["algorithmic_d1_channel"] = d1_df
            augmented_info["algorithmic_d1_channel"] = {
                "label": "d'(depth)",
                "channels": ["depth_d1_ms"],
                "metadata": {"depth_d1_ms": {"unit": "m/s"}},
            }
        if "depth_d2_ms2" in derivative_df.columns:
            d2_df = derivative_df[["datetime", "depth_d2_ms2"]].copy()
            augmented_data["algorithmic_d2_channel"] = d2_df
            augmented_info["algorithmic_d2_channel"] = {
                "label": "d''(depth)",
                "channels": ["depth_d2_ms2"],
                "metadata": {"depth_d2_ms2": {"unit": "m/s^2"}},
            }
    return augmented_data, augmented_info


def _augment_segment_summary_signal_data(
    signal_data: dict,
    signal_info: dict,
    seg_df: pd.DataFrame,
) -> tuple[dict, dict]:
    augmented_data = copy.deepcopy(signal_data or {})
    augmented_info = copy.deepcopy(signal_info or {})
    if seg_df is None or seg_df.empty:
        return augmented_data, augmented_info
    summary_cols = ["segment_midpoint_datetime", "inferred_buoyancy_value", "drift_rate_ms"]
    available_cols = [col for col in summary_cols if col in seg_df.columns]
    if "segment_midpoint_datetime" not in available_cols:
        return augmented_data, augmented_info
    summary_df = seg_df[available_cols].copy()
    summary_df["segment_midpoint_datetime"] = pd.to_datetime(summary_df["segment_midpoint_datetime"], errors="coerce")
    summary_df = summary_df.rename(columns={"segment_midpoint_datetime": "datetime"})
    numeric_cols = [col for col in ["inferred_buoyancy_value", "drift_rate_ms"] if col in summary_df.columns]
    for col in numeric_cols:
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
    summary_df = summary_df.dropna(subset=["datetime"], how="any")
    if numeric_cols:
        summary_df = summary_df.dropna(subset=numeric_cols, how="all")
    summary_df = summary_df.sort_values("datetime").reset_index(drop=True)
    if summary_df.empty:
        return augmented_data, augmented_info
    augmented_data["algorithmic_buoyancy_channel"] = summary_df
    augmented_info["algorithmic_buoyancy_channel"] = {
        "channels": [col for col in ["inferred_buoyancy_value", "drift_rate_ms"] if col in summary_df.columns],
        "metadata": {
            "inferred_buoyancy_value": {"unit": "m/s"},
            "drift_rate_ms": {"unit": "m/s"},
        },
    }
    return augmented_data, augmented_info


def _stage_count_barplot(stage_df: pd.DataFrame, color_map: dict[str, str]) -> go.Figure:
    final_kept_color = color_map.get("__review_final_kept__")
    if not final_kept_color:
        subtype_colors = [value for key, value in color_map.items() if str(key).startswith("__review_final_kept__::")]
        final_kept_color = subtype_colors[0] if subtype_colors else "#2C7BB6"
    rows = []
    stage_lookup = {str(row["stage"]): row for _, row in stage_df.iterrows()}
    stage_specs = [
        ("1) unfiltered_rest_candidate", "__review_unfiltered_rest__"),
        ("2) filtered_rest_candidate", "__review_initial_segments__"),
        ("3) unfiltered_putative_rest", "__review_pre_context__"),
        ("4) rejected_after_measured_filters", "__review_rejected__"),
        ("5) rejected_after_inferred_filters", "__review_rejected__"),
        ("6) rest", "__review_final_kept__"),
    ]
    for stage_name, color_key in stage_specs:
        row = stage_lookup.get(stage_name)
        if row is None:
            continue
        rows.append(
            {
                "stage": stage_name,
                "count": float(row.get("remaining_after", 0) or 0),
                "color": final_kept_color if color_key == "__review_final_kept__" else color_map.get(color_key, "#999999"),
            }
        )
    plot_df = pd.DataFrame(rows)
    fig = go.Figure()
    if not plot_df.empty:
        fig.add_trace(
            go.Bar(
                x=plot_df["stage"],
                y=plot_df["count"],
                marker_color=plot_df["color"],
                text=plot_df["count"].astype(int).astype(str),
                textposition="outside",
                hovertemplate="%{x}<br>Count=%{y}<extra></extra>",
            )
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        yaxis_title="Candidate count",
        xaxis_title="Review stage",
        showlegend=False,
    )
    return fig


def _add_context_marker_traces(fig, signal_order: list[str], seg_df: pd.DataFrame) -> None:
    if seg_df is None or seg_df.empty:
        return
    row_lookup = {signal_name: idx + 1 for idx, signal_name in enumerate(signal_order)}
    marker_specs = [
        ("inferred_buoyancy_value", "algorithmic_d1_channel", "#F39C12", "diamond", "Inferred buoyancy"),
        ("segment_mean_stroke_rate_spm", "stroke_rate", "#C0392B", "circle", "Segment mean stroke rate"),
        ("start_depth_m", "depth", "#16A085", "square", "Segment start depth"),
    ]
    for field_name, target_signal, color, symbol, label in marker_specs:
        if field_name not in seg_df.columns or target_signal not in row_lookup:
            continue
        marker_df = seg_df[["segment_midpoint_datetime", field_name]].copy()
        marker_df["segment_midpoint_datetime"] = pd.to_datetime(marker_df["segment_midpoint_datetime"], errors="coerce")
        marker_df[field_name] = pd.to_numeric(marker_df[field_name], errors="coerce")
        marker_df = (
            marker_df.dropna(subset=["segment_midpoint_datetime", field_name])
            .sort_values("segment_midpoint_datetime")
            .reset_index(drop=True)
        )
        if marker_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=marker_df["segment_midpoint_datetime"],
                y=marker_df[field_name],
                mode="markers",
                marker=dict(symbol=symbol, color=color, size=8, line=dict(color="rgba(255,255,255,0.9)", width=1)),
                name=label,
                showlegend=False,
                hovertemplate=f"{label}<br>%{{x}}<br>%{{y}}<extra></extra>",
            ),
            row=row_lookup[target_signal],
            col=1,
        )
    if "drift_rate_ms" in seg_df.columns and "algorithmic_buoyancy_channel" in row_lookup:
        marker_df = seg_df[["segment_midpoint_datetime", "drift_rate_ms"]].copy()
        marker_df["segment_midpoint_datetime"] = pd.to_datetime(marker_df["segment_midpoint_datetime"], errors="coerce")
        marker_df["drift_rate_ms"] = pd.to_numeric(marker_df["drift_rate_ms"], errors="coerce")
        marker_df = (
            marker_df.dropna(subset=["segment_midpoint_datetime", "drift_rate_ms"])
            .sort_values("segment_midpoint_datetime")
            .reset_index(drop=True)
        )
        if not marker_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=marker_df["segment_midpoint_datetime"],
                    y=marker_df["drift_rate_ms"],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=9,
                        color=marker_df["drift_rate_ms"],
                        colorscale="Magma",
                        line=dict(color="rgba(255,255,255,0.9)", width=1),
                        showscale=False,
                    ),
                    name="Drift rate",
                    showlegend=False,
                    hovertemplate="Drift rate<br>%{x}<br>%{y:.3f} m/s<extra></extra>",
                ),
                row=row_lookup["algorithmic_buoyancy_channel"],
                col=1,
            )


def _save_event_style_colors(color_mapping_path: str, updates: dict[str, str]) -> dict[str, int]:
    mapping_path = Path(color_mapping_path)
    with mapping_path.open("r", encoding="utf-8") as handle:
        color_mapping = json.load(handle)
    event_styles = dict(color_mapping.get("__event_styles__") or {})
    updated_count = 0
    mirrored_count = 0
    for event_key, color in updates.items():
        if event_key.startswith("find_rest.") and event_key not in event_styles:
            continue
        style = dict(event_styles.get(event_key) or {})
        style["color"] = str(color)
        style["shade_color"] = str(color)
        event_styles[event_key] = style
        if event_key.startswith("find_rest."):
            mirrored_count += 1
        else:
            updated_count += 1
    color_mapping["__event_styles__"] = event_styles
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(color_mapping, handle, indent=4)
        handle.write("\n")
    return {"updated_count": updated_count, "mirrored_count": mirrored_count}


def _resolved_context_filters(algo_cfg: dict, run_cfg: dict, context_filter_definitions: dict | None = None) -> list[dict]:
    definitions = (
        copy.deepcopy(dict(context_filter_definitions or {}))
        if context_filter_definitions is not None
        else copy.deepcopy(dict(run_cfg.get("context_filter_definitions") or {}))
    )
    resolved = []
    for idx, raw_item in enumerate(list(algo_cfg.get("context_filters") or [])):
        if isinstance(raw_item, str):
            filter_id = str(raw_item).strip()
            cfg = copy.deepcopy(definitions.get(filter_id) or {})
            cfg.setdefault("id", filter_id)
            resolved.append(
                {
                    "slot_type": "definition_ref",
                    "slot_ref": filter_id,
                    "display_id": filter_id,
                    "filter_stage": context_filter_stage(cfg),
                    "filter_cfg": cfg,
                    "source_index": idx,
                }
            )
        elif isinstance(raw_item, dict):
            cfg = copy.deepcopy(raw_item)
            filter_id = str(cfg.get("id") or f"context_filter_{idx + 1}").strip()
            cfg.setdefault("id", filter_id)
            resolved.append(
                {
                    "slot_type": "inline",
                    "slot_ref": filter_id,
                    "display_id": filter_id,
                    "filter_stage": context_filter_stage(cfg),
                    "filter_cfg": cfg,
                    "source_index": idx,
                }
            )
    return resolved


_FIELD_UNIT_LABELS = {
    "duration_s": "seconds",
    "elapsed_days": "days",
    "start_depth_m": "m",
    "end_depth_m": "m",
    "dive_depth_min_m": "m",
    "drift_rate_ms": "m/s",
    "depth_d1_ms": "m/s",
    "depth_d2_ms2": "m/s^2",
    "inferred_buoyancy_value": "m/s",
    "inferred_buoyancy_early_reference_value": "m/s",
    "inferred_buoyancy_late_reference_value": "m/s",
    "segment_mean_stroke_rate_spm": "spm",
    "algorithmic_segment_mean_stroke_rate_spm": "spm",
    "algorithmic_context_keep": "flag",
    "surface_sleep_min_duration_s": "seconds",
    "end_flat_abs_mean_d1_max_ms": "m/s",
    "drift_rate_abs_max_ms": "m/s",
    "curvature_abs_max_ms": "m/s",
}


def _unit_for_filter_text(text: str) -> str | None:
    text_lower = str(text or "").strip().lower()
    for field_name, unit in _FIELD_UNIT_LABELS.items():
        if field_name.lower() in text_lower:
            return unit
    return None


def _context_filter_summary(filter_cfg: dict) -> str:
    rule = dict(filter_cfg.get("rule") or {})
    mode = str(rule.get("mode") or "").strip().lower()
    def _humanize_phase_values(values):
        rendered = []
        for value in list(values or []):
            rendered.append(_humanize_buoyancy_phase(value))
        return rendered
    if mode == "threshold_gate":
        comparator = str(dict(rule.get("keep_if") or {}).get("comparator") or "").strip()
        threshold = dict(rule.get("keep_if") or {}).get("threshold")
        subject = str(rule.get("field") or rule.get("value_expr") or filter_cfg.get("covariate_ref") or "threshold")
        if "inferred_buoyancy" in subject:
            subject = subject.replace("inferred_buoyancy_value", "inferred buoyancy")
            subject = subject.replace("inferred_buoyancy_early_reference_value", "early inferred buoyancy")
            subject = subject.replace("inferred_buoyancy_late_reference_value", "late inferred buoyancy")
        unit = _unit_for_filter_text(subject)
        threshold_text = f"{threshold} {unit}" if unit and threshold is not None else str(threshold)
        return f"{subject} {comparator} {threshold_text}".strip()
    if mode == "category_gate":
        field = str(rule.get("field") or "category")
        allowed_values = list(rule.get("allowed_values") or [])
        if "trip_phase" in field or "buoyancy_phase" in field:
            allowed_values = _humanize_phase_values(allowed_values)
            field = field.replace("inferred_trip_phase", "inferred buoyancy state")
            field = field.replace("inferred_buoyancy_phase", "inferred buoyancy state")
        allowed = ", ".join(str(v) for v in allowed_values)
        return f"{field} in [{allowed}]"
    if mode == "phase_gate":
        field = str(rule.get("field") or "phase")
        allowed_values = _humanize_phase_values(rule.get("allowed_phases") or rule.get("keep_if_in") or [])
        field = field.replace("inferred_trip_phase", "inferred buoyancy state")
        field = field.replace("inferred_buoyancy_phase", "inferred buoyancy state")
        allowed = ", ".join(str(v) for v in allowed_values)
        return f"{field} in [{allowed}]"
    return str(filter_cfg.get("reject_reason") or "").strip()


def _terminal_display_name(algo_cfg: dict, terminal_key: str) -> str:
    label_names = dict(algo_cfg.get("label_names") or {})
    return str(label_names.get(terminal_key) or terminal_key)


def _state_name_matches_selector(state_name: str, selector: str | None) -> bool:
    value = str(state_name or "").strip()
    if not value:
        return False
    if not selector or str(selector).strip() == "":
        return True
    sel = str(selector)
    if sel == "__all_positive__":
        return True
    if sel.endswith(".*"):
        prefix = sel[:-2]
        return value.startswith(f"{prefix}.")
    return value == sel


def _state_label_maps(config: dict, analysis_id: str, algo_cfg: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    pretty_map: dict[str, str] = {}
    event_alias_map: dict[str, set[str]] = {}

    run_cfg_entry = find_run_config_entry(config, analysis_id)
    if isinstance(run_cfg_entry, tuple):
        run_cfg = dict(run_cfg_entry[1] or {})
    elif isinstance(run_cfg_entry, dict):
        run_cfg = dict(run_cfg_entry or {})
    else:
        run_cfg = {}
    algorithmic_block = dict(run_cfg.get("algorithmic") or {})
    segments_block = dict(
        algorithmic_block.get("segments")
        or run_cfg.get("algorithmic_segments")
        or {}
    )
    supervised_block = dict(run_cfg.get("supervised") or {})
    label_group_name = str(
        segments_block.get("label_group")
        or algorithmic_block.get("label_group")
        or supervised_block.get("label_group")
        or ""
    ).strip()
    shared_groups = dict(config.get("supervised_label_group_definitions") or {})
    terminal_cfg = dict((shared_groups.get(label_group_name) or {}).get("algorithmic_terminals") or {})

    for terminal_key, terminal_meta in terminal_cfg.items():
        meta = dict(terminal_meta or {})
        event_key = str(meta.get("event_key") or "").strip()
        label_name = str(meta.get("label_name") or terminal_key).strip()
        pretty_label = str(meta.get("pretty_label") or label_name).strip()
        for alias in [str(terminal_key).strip(), label_name, event_key]:
            if alias:
                pretty_map[alias] = pretty_label
                event_alias_map.setdefault(alias, set())
                if event_key:
                    event_alias_map[alias].add(event_key)

    label_names = dict(algo_cfg.get("label_names") or {})
    event_keys = dict(algo_cfg.get("event_keys") or {})
    for terminal_key, label_name in label_names.items():
        t_key = str(terminal_key).strip()
        l_name = str(label_name).strip()
        if not l_name:
            continue
        pretty_map.setdefault(l_name, l_name)
        pretty_map.setdefault(t_key, pretty_map.get(l_name, l_name))
        event_alias_map.setdefault(l_name, set())
        event_alias_map.setdefault(t_key, set())
        event_key = str(event_keys.get(t_key) or "").strip()
        if event_key:
            event_alias_map[l_name].add(event_key)
            event_alias_map[t_key].add(event_key)

    return pretty_map, {k: sorted(v) for k, v in event_alias_map.items() if v}


def _rest_state_label_map(pretty_map: dict[str, str], event_alias_map: dict[str, list[str]]) -> dict[str, list[str]]:
    buckets: dict[str, set[str]] = {
        "surface_sleep": {"putative_rest.surface_sleep", "find_rest.surface_sleep"},
        "benthic_sleep": {"putative_rest.benthic_sleep", "find_rest.long_flat"},
        "drift_sleep": {"putative_rest.drift_sleep", "find_rest.long_drift"},
    }
    all_names: set[str] = set(pretty_map.keys())
    for aliases in event_alias_map.values():
        all_names.update(str(v).strip() for v in list(aliases or []) if str(v).strip())
    for name in all_names:
        lower_name = str(name).lower()
        if not (("rest" in lower_name) or ("sleep" in lower_name)):
            continue
        if "surface" in lower_name:
            buckets["surface_sleep"].add(name)
        if ("benthic" in lower_name) or ("long_flat" in lower_name) or ("flat" in lower_name):
            buckets["benthic_sleep"].add(name)
        if ("drift" in lower_name) or ("long_drift" in lower_name):
            buckets["drift_sleep"].add(name)
    return {key: sorted(values) for key, values in buckets.items() if values}


def _hex_to_rgba(color_hex: str, opacity: float) -> str:
    value = str(color_hex or "").strip().lstrip("#")
    if len(value) != 6:
        return f"rgba(0, 0, 0, {opacity})"
    try:
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
    except ValueError:
        return f"rgba(0, 0, 0, {opacity})"
    return f"rgba({red}, {green}, {blue}, {opacity})"


def _subtype_review_keys(state_names: list[str]) -> list[str]:
    return [f"__review_final_kept__::{str(name)}" for name in state_names if str(name).strip()]


REVIEW_COLOR_DEFAULTS = {
    "__review_unfiltered_rest__": "#C9D7DC",
    "__review_initial_segments__": "#8A9EA6",
    "__review_pre_context__": "#346A7F",
    "__review_final_kept__": "#2C7BB6",
    "__review_rejected__": "#FF7272",
    "__review_final_kept__::resting_surface": "#EBED61",
    "__review_final_kept__::resting_benthic": "#26CEA6",
    "__review_final_kept__::resting_drift": "#93A2FF",
    "__review_final_kept__::swimming_uw": "#FF5722",
    "__review_final_kept__::gliding_uw": "#8BC34A",
    "__review_final_kept__::swimming_surface": "#FF7043",
    "__review_final_kept__::gliding_surface": "#A5D6A7",
    "__review_final_kept__::unscorable": "#BDBDBD",
}


def _review_color_label(key: str, pretty_map: dict[str, str] | None = None) -> str:
    if key.startswith("__review_final_kept__::"):
        subtype = key.split("::", 1)[1]
        pretty = (pretty_map or {}).get(subtype, subtype)
        return f"4) {pretty}"
    labels = {
        "__review_unfiltered_rest__": "1) unfiltered_rest_candidate",
        "__review_initial_segments__": "2) filtered_rest_candidate",
        "__review_pre_context__": "3) unfiltered_putative_rest",
        "__review_final_kept__": "4) rest",
        "__review_rejected__": "Rejected base-pass events",
    }
    return labels.get(key, key)


def _review_color_controls(
    include_rejected: bool,
    subtype_state_names: list[str] | None = None,
    pretty_map: dict[str, str] | None = None,
) -> dict[str, str]:
    st.markdown("**Review Event Colors**")
    subtype_keys = _subtype_review_keys(subtype_state_names or [])
    active_keys = [
        "__review_unfiltered_rest__",
        "__review_initial_segments__",
        "__review_pre_context__",
        *subtype_keys,
    ]
    if not subtype_keys:
        active_keys.append("__review_final_kept__")
    if include_rejected:
        active_keys.append("__review_rejected__")

    color_cols = st.columns(len(active_keys))
    for idx, key in enumerate(active_keys):
        with color_cols[idx]:
            widget_key = f"state_review_color_{key}"
            st.color_picker(
                _review_color_label(key, pretty_map=pretty_map),
                value=REVIEW_COLOR_DEFAULTS.get(key, "#2C7BB6"),
                key=widget_key,
            )
    return {
        key: str(st.session_state.get(f"state_review_color_{key}") or REVIEW_COLOR_DEFAULTS.get(key, "#2C7BB6"))
        for key in active_keys
    }


def _render_review_color_legend(
    color_map: dict[str, str],
    include_rejected: bool,
    subtype_state_names: list[str] | None = None,
    pretty_map: dict[str, str] | None = None,
) -> None:
    labels = [
        ("__review_unfiltered_rest__", "1) unfiltered_rest_candidate"),
        ("__review_initial_segments__", "2) filtered_rest_candidate"),
        ("__review_pre_context__", "3) unfiltered_putative_rest"),
    ]
    subtype_keys = _subtype_review_keys(subtype_state_names or [])
    if subtype_keys:
        labels.extend((key, _review_color_label(key, pretty_map=pretty_map)) for key in subtype_keys)
    else:
        labels.append(("__review_final_kept__", "4) rest"))
    if include_rejected:
        labels.append(("__review_rejected__", "Rejected base-pass events"))
    legend_html = "".join(
        (
            "<div style='display:flex;align-items:center;gap:0.45rem;"
            "padding:0.2rem 0.65rem;border:1px solid rgba(0,0,0,0.08);"
            "border-radius:999px;background:rgba(255,255,255,0.72);'>"
            f"<span style='display:inline-block;width:0.95rem;height:0.95rem;border-radius:999px;background:{color_map[key]};"
            "border:1px solid rgba(0,0,0,0.18);'></span>"
            f"<span style='font-size:0.92rem;'>{label}</span>"
            "</div>"
        )
        for key, label in labels
    )
    st.markdown(
        (
            "<div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin:0.25rem 0 0.9rem 0;'>"
            f"{legend_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _editable_preview_config(
    algo_cfg: dict,
    run_cfg: dict,
    prefix: str,
) -> tuple[dict, dict, list[str], bool, bool, bool]:
    edited_algo = copy.deepcopy(algo_cfg)
    edited_defs = copy.deepcopy(dict(run_cfg.get("context_filter_definitions") or {}))

    st.sidebar.subheader("Preview Controls")
    with st.sidebar.form(key=f"{prefix}_preview_form"):
        with st.expander("0) Signal Preparation", expanded=False):
            thresholds = dict(edited_algo.get("thresholds") or {})
            terminal_criteria = copy.deepcopy(dict(edited_algo.get("terminal_criteria") or {}))
            edited_algo["standardize"] = st.checkbox(
                "Standardize depth",
                value=bool(edited_algo.get("standardize", True)),
                key=f"{prefix}_standardize",
            )
            edited_algo["quantize_step"] = st.number_input(
                "Quantize step",
                min_value=0.0,
                value=float(edited_algo.get("quantize_step", 1.0) or 0.0),
                step=0.1,
                key=f"{prefix}_quantize_step",
            )
            edited_algo["base_smooth_seconds"] = st.number_input(
                "Base smooth seconds",
                min_value=0.0,
                value=float(edited_algo.get("base_smooth_seconds", 6.0)),
                step=0.5,
                key=f"{prefix}_base_smooth_seconds",
            )
            edited_algo["coarse_smooth_seconds"] = st.number_input(
                "Coarse smooth seconds",
                min_value=0.0,
                value=float(edited_algo.get("coarse_smooth_seconds", 12.0)),
                step=0.5,
                key=f"{prefix}_coarse_smooth_seconds",
            )
            edited_algo["coarse_interval_threshold_s"] = st.number_input(
                "Coarse interval threshold (s)",
                min_value=0.0,
                value=float(edited_algo.get("coarse_interval_threshold_s", 5.0)),
                step=0.5,
                key=f"{prefix}_coarse_interval_threshold_s",
            )
            edited_algo["coarse_resolution_threshold"] = st.number_input(
                "Coarse resolution threshold",
                min_value=0.0,
                value=float(edited_algo.get("coarse_resolution_threshold", 1.0)),
                step=0.1,
                key=f"{prefix}_coarse_resolution_threshold",
            )
            debug_cfg = dict(edited_algo.get("debug") or {})
            debug_cfg["write_context_review_table"] = st.checkbox(
                "Include debug.write_context_review_table",
                value=bool(debug_cfg.get("write_context_review_table", False)),
                key=f"{prefix}_debug_write_context_review_table",
                help="When enabled, the saved non-context config block will include `debug.write_context_review_table: true`.",
            )
            edited_algo["debug"] = debug_cfg

        thresholds = dict(edited_algo.get("thresholds") or {})
        terminal_criteria = copy.deepcopy(dict(edited_algo.get("terminal_criteria") or {}))
        with st.expander("1) unfiltered_rest_candidate", expanded=True):
            d1_min_default = float(thresholds.get("first_deriv_min_ms", -abs(float(thresholds.get("first_deriv_abs_max_ms", 0.6)))))
            d1_max_default = float(thresholds.get("first_deriv_max_ms", abs(float(thresholds.get("first_deriv_abs_max_ms", 0.6)))))
            d2_min_default = float(thresholds.get("second_deriv_min_ms2", -abs(float(thresholds.get("second_deriv_abs_max_ms2", 0.05)))))
            d2_max_default = float(thresholds.get("second_deriv_max_ms2", abs(float(thresholds.get("second_deriv_abs_max_ms2", 0.05)))))
            d1_band = st.slider(
                "D1 band (m/s)",
                min_value=-2.0,
                max_value=2.0,
                value=(d1_min_default, d1_max_default),
                step=0.01,
                key=f"{prefix}_d1_band",
                help="Samples inside this derivative band form `unfiltered_rest` before slope and duration filtering.",
            )
            d2_band = st.slider(
                "D2 band (m/s²)",
                min_value=-0.5,
                max_value=0.5,
                value=(d2_min_default, d2_max_default),
                step=0.005,
                key=f"{prefix}_d2_band",
                help="Samples inside this curvature band form `unfiltered_rest` before later segment filters.",
            )
            thresholds["first_deriv_min_ms"] = float(d1_band[0])
            thresholds["first_deriv_max_ms"] = float(d1_band[1])
            thresholds["first_deriv_abs_max_ms"] = max(abs(float(d1_band[0])), abs(float(d1_band[1])))
            thresholds["second_deriv_min_ms2"] = float(d2_band[0])
            thresholds["second_deriv_max_ms2"] = float(d2_band[1])
            thresholds["second_deriv_abs_max_ms2"] = max(abs(float(d2_band[0])), abs(float(d2_band[1])))

        with st.expander("2) filtered_rest_candidate", expanded=False):
            thresholds["min_duration_s"] = st.number_input(
                "Min rest segment duration (s)",
                value=float(thresholds.get("min_duration_s", 180.0)),
                step=1.0,
                key=f"{prefix}_thr_min_duration_s",
                help="Minimum duration for signed rest candidate segments to proceed beyond the initial band stage.",
            )
            thresholds["surface_sleep_min_duration_s"] = st.number_input(
                "Surface sleep min duration (s)",
                value=float(thresholds.get("surface_sleep_min_duration_s", 600.0)),
                step=1.0,
                key=f"{prefix}_thr_surface_sleep_min_duration_s",
            )
            thresholds["dive_depth_min_m"] = st.number_input(
                "Dive depth min (m)",
                value=float(thresholds.get("dive_depth_min_m", 2.0)),
                step=0.1,
                key=f"{prefix}_thr_dive_depth_min_m",
            )
            edited_algo["end_segments_upon_d1_sign_change"] = st.checkbox(
                "Split at d1 sign changes",
                value=bool(edited_algo.get("end_segments_upon_d1_sign_change", True)),
                key=f"{prefix}_sign_change",
            )
            slope_help = "For depth- and pressure-like channels with reversed y-axes, ascent/descent follow the visually intuitive direction rather than raw derivative sign."
            edited_algo["filter_out_ascent"] = st.checkbox(
                "Filter out ascents",
                value=bool(edited_algo.get("filter_out_ascent", True)),
                key=f"{prefix}_filter_ascent",
                help=slope_help,
            )
            edited_algo["filter_out_descent"] = st.checkbox(
                "Filter out descents",
                value=bool(edited_algo.get("filter_out_descent", False)),
                key=f"{prefix}_filter_descent",
                help=slope_help,
            )
            threshold_specs = [
                ("drift_rate_abs_max_ms", "Drift rate abs max (m/s)", 0.01),
                ("curvature_abs_max_ms", "Curvature abs max (m/s)", 0.01),
                ("end_flat_abs_mean_d1_max_ms", "End-flat abs mean d1 max (m/s)", 0.001),
            ]
            for key, label, step in threshold_specs:
                thresholds[key] = st.number_input(
                    label,
                    value=float(thresholds.get(key, 0.0)),
                    step=step,
                    key=f"{prefix}_thr_{key}",
                )

        with st.expander("3) terminal state mapping", expanded=False):
            surface_key = "surface_sleep"
            flat_key = "long_flat"
            drift_key = "long_drift"
            surface_cfg = dict(terminal_criteria.get(surface_key) or {})
            flat_cfg = dict(terminal_criteria.get(flat_key) or {})
            drift_cfg = dict(terminal_criteria.get(drift_key) or {})

            st.caption(f"{_terminal_display_name(edited_algo, surface_key)}")
            surface_cfg["uses_surface_sleep_min_duration"] = st.checkbox(
                "Use `surface_sleep_min_duration_s`",
                value=bool(surface_cfg.get("uses_surface_sleep_min_duration", True)),
                key=f"{prefix}_{surface_key}_uses_surface_sleep_min_duration",
                help="If disabled, surface-sleep labeling will no longer require the surface-sleep minimum duration threshold.",
            )

            st.caption(f"{_terminal_display_name(edited_algo, flat_key)}")
            flat_cfg["uses_min_duration"] = st.checkbox(
                "Use `min_duration_s`",
                value=bool(flat_cfg.get("uses_min_duration", True)),
                key=f"{prefix}_{flat_key}_uses_min_duration",
                help="Documentary for now: filtered rest still uses a shared minimum duration stage before terminal assignment.",
                disabled=True,
            )
            flat_cfg["uses_end_flat_abs_mean_d1_max_ms"] = st.checkbox(
                "Use `end_flat_abs_mean_d1_max_ms`",
                value=bool(flat_cfg.get("uses_end_flat_abs_mean_d1_max_ms", True)),
                key=f"{prefix}_{flat_key}_uses_end_flat_abs_mean_d1_max_ms",
                help="Controls whether the end-flat criterion is used to assign this terminal state.",
            )

            st.caption(f"{_terminal_display_name(edited_algo, drift_key)}")
            drift_cfg["uses_min_duration"] = st.checkbox(
                "Use `min_duration_s` ",
                value=bool(drift_cfg.get("uses_min_duration", True)),
                key=f"{prefix}_{drift_key}_uses_min_duration",
                help="Documentary for now: filtered rest still uses a shared minimum duration stage before terminal assignment.",
                disabled=True,
            )
            drift_cfg["uses_drift_rate_abs_max_ms"] = st.checkbox(
                "Use `drift_rate_abs_max_ms`",
                value=bool(drift_cfg.get("uses_drift_rate_abs_max_ms", True)),
                key=f"{prefix}_{drift_key}_uses_drift_rate_abs_max_ms",
                help="Controls whether the drift-rate threshold is used to assign this terminal state.",
            )
            drift_cfg["uses_curvature_abs_max_ms"] = st.checkbox(
                "Use `curvature_abs_max_ms`",
                value=bool(drift_cfg.get("uses_curvature_abs_max_ms", True)),
                key=f"{prefix}_{drift_key}_uses_curvature_abs_max_ms",
                help="Controls whether the curvature threshold is used to assign this terminal state.",
            )
            terminal_criteria[surface_key] = surface_cfg
            terminal_criteria[flat_key] = flat_cfg
            terminal_criteria[drift_key] = drift_cfg
        edited_algo["thresholds"] = thresholds
        edited_algo["terminal_criteria"] = terminal_criteria

        ordered_filter_ids: list[str] = []
        resolved_filters = _resolved_context_filters(
            edited_algo,
            run_cfg,
            context_filter_definitions=edited_defs,
        )
        measured_filters = [entry for entry in resolved_filters if str(entry.get("filter_stage")) == "measured"]
        inferred_filters = [entry for entry in resolved_filters if str(entry.get("filter_stage")) == "inferred"]
        for entry in measured_filters + inferred_filters:
            ordered_filter_ids.append(str(entry["display_id"]))

        def _render_context_filter_group(entries: list[dict], header: str, disabled: bool = False) -> None:
            if not entries:
                st.caption(f"No {header.lower()} configured.")
                return
            for entry in entries:
                filter_cfg = copy.deepcopy(entry["filter_cfg"])
                filter_id = str(entry["display_id"])
                enabled_key = f"{prefix}_{filter_id}_enabled"
                if enabled_key not in st.session_state:
                    st.session_state[enabled_key] = False
                filter_cfg["enabled"] = st.checkbox(
                    f"{filter_id} enabled",
                    value=bool(st.session_state.get(enabled_key, filter_cfg.get("enabled", True))),
                    key=enabled_key,
                    disabled=disabled,
                )
                st.caption(f"{filter_id}: {_context_filter_summary(filter_cfg)}")
                applies_when = dict(filter_cfg.get("applies_when") or {})
                if "elapsed_days" in applies_when and isinstance(applies_when["elapsed_days"], dict):
                    elapsed_rule = dict(applies_when["elapsed_days"] or {})
                    elapsed_rule["threshold"] = st.number_input(
                        f"{filter_id} applies_when elapsed_days threshold (days)",
                        value=float(elapsed_rule.get("threshold", 0.0)),
                        step=0.5,
                        key=f"{prefix}_{filter_id}_applies_elapsed_days",
                        disabled=disabled,
                    )
                    applies_when["elapsed_days"] = elapsed_rule
                    filter_cfg["applies_when"] = applies_when
                rule = dict(filter_cfg.get("rule") or {})
                mode = str(rule.get("mode") or "").strip().lower()
                if mode == "threshold_gate":
                    keep_if = dict(rule.get("keep_if") or {})
                    subject = str(rule.get("field") or rule.get("value_expr") or filter_cfg.get("covariate_ref") or "threshold")
                    unit = _unit_for_filter_text(subject)
                    label_suffix = f" ({unit})" if unit else ""
                    threshold_label = f"{filter_id} threshold{label_suffix}"
                    threshold_key = f"{prefix}_{filter_id}_threshold"
                    threshold_step = 0.01
                    threshold_help = subject
                    if str(filter_id) == "low_stroke_rate_required":
                        threshold_label = "Stroke-rate max override (spm)"
                        threshold_key = f"{prefix}_stroke_rate_max_override"
                        threshold_step = 0.1
                        threshold_help = (
                            "Per-run override for the measured stroke-rate gate. "
                            "Saved to segmentation_runs.yaml for this run."
                        )
                    keep_if["threshold"] = st.number_input(
                        threshold_label,
                        value=float(keep_if.get("threshold", 0.0)),
                        step=threshold_step,
                        key=threshold_key,
                        help=threshold_help,
                        disabled=disabled,
                    )
                    rule["keep_if"] = keep_if
                    filter_cfg["rule"] = rule
                elif mode == "category_gate":
                    allowed_values = [str(v) for v in (rule.get("allowed_values") or []) if str(v).strip()]
                    options = sorted(set(allowed_values + ["negative", "positive", "neutral", "surface"]))
                    rule["allowed_values"] = st.multiselect(
                        f"{filter_id} allowed values",
                        options=options,
                        default=allowed_values,
                        key=f"{prefix}_{filter_id}_allowed_values",
                        disabled=disabled,
                    )
                    filter_cfg["rule"] = rule
                if entry["slot_type"] == "definition_ref":
                    edited_defs[entry["slot_ref"]] = filter_cfg
                else:
                    edited_algo["context_filters"][entry["source_index"]] = filter_cfg
        with st.expander("4) measured context filters", expanded=False):
            st.caption("Applied immediately after terminal state mapping. Stroke-rate gates belong here.")
            _render_context_filter_group(measured_filters, "Measured Context Filters")
        with st.expander("5) inferred context filters (later pass)", expanded=False):
            st.caption("Applied in the same deployment run when all context filters are active.")
            _render_context_filter_group(
                inferred_filters,
                "Inferred Context Filters",
                disabled=False,
            )

        preview_cols = st.columns(3)
        rerun_preview = preview_cols[0].form_submit_button(
            "Preview window"
        )
        run_deployment = preview_cols[1].form_submit_button(
            "Run algorithmic segmentation for deployment"
        )
        save_and_rerun = preview_cols[2].form_submit_button("Save to segmentation_runs and re-run algorithmic segmentation")
    return edited_algo, edited_defs, ordered_filter_ids, rerun_preview, run_deployment, save_and_rerun


st.set_page_config(page_title="State Event Review", layout="wide")

config, data_dir, color_mapping_path, _ = load_configuration()
repo_root = config["paths"]["local_repo_path"]
config_path = os.getenv("CONFIG_PATH") or str(Path(repo_root) / "config.yaml")
segmentation_runs_path = str(resolve_segmentation_runs_path())

st.title("State Event Review")
st.caption("Review algorithmic state events, adjust thresholds and context filters, and preview how candidates drop out before saving anything.")

run_infos = discover_algorithmic_review_runs(data_dir)
if not run_infos:
    st.error("No segmentation runs with algorithmic artifacts were found for this dataset.")
    st.stop()

default_analysis_id = "mian_mile_sleep_transfer_rf"
default_run_idx = next((idx for idx, row in enumerate(run_infos) if str(row.get("analysis_id")) == default_analysis_id), 0)
selected_run = st.sidebar.selectbox("Segmentation Run", options=run_infos, index=default_run_idx, format_func=lambda row: row["label"])
analysis_id = str(selected_run["analysis_id"])
_, run_cfg = find_run_config_entry(config, analysis_id)

dataset_options = _run_scoped_dataset_ids(config, run_cfg, data_dir)
if not dataset_options:
    st.error(f"No dataset folders from config were found for segmentation run `{analysis_id}`.")
    st.stop()
dataset_state_key = f"state_review_dataset_selection::{analysis_id}"
selected_dataset_idx = 0
preferred_dataset = st.session_state.get(dataset_state_key)
if preferred_dataset in dataset_options:
    selected_dataset_idx = dataset_options.index(preferred_dataset)
dataset_id = st.sidebar.selectbox("Run Dataset", dataset_options, index=selected_dataset_idx, key=dataset_state_key)

deployment_options = _run_scoped_deployment_ids(config, run_cfg, dataset_id, data_dir)
if not deployment_options:
    st.error(f"No deployment folders were found for dataset `{dataset_id}` in run `{analysis_id}`.")
    st.stop()
deployment_state_key = f"state_review_deployment_selection::{analysis_id}::{dataset_id}"
selected_deployment_idx = 0
preferred_deployment = st.session_state.get(deployment_state_key)
if preferred_deployment in deployment_options:
    selected_deployment_idx = deployment_options.index(preferred_deployment)
deployment_id = st.sidebar.selectbox("Run Deployment", deployment_options, index=selected_deployment_idx, key=deployment_state_key)

animal_id, dataset_folder, deployment_folder, data_pkl, param_manager = _load_deployment_for_state_review(
    data_dir=data_dir,
    dataset_id=dataset_id,
    deployment_id=deployment_id,
)
deployment_tz_name = str((getattr(data_pkl, "deployment_info", {}) or {}).get("Time Zone") or "UTC")
current_selection_key = _selection_key(analysis_id, dataset_id, deployment_id)

preview_store = st.session_state.get("state_event_review_preview") or {}
config_review_state_key = f"state_event_review_config_review::{current_selection_key}"
workflow_run_state_key = f"state_event_review_workflow_run::{current_selection_key}"
if preview_store and preview_store.get("selection_key") != current_selection_key:
    cleanup_temp_workspace(preview_store.get("temp_root"))
    st.session_state.pop("state_event_review_preview", None)
    st.session_state.pop(_activity_budget_mode_key(current_selection_key), None)
    preview_store = {}
stale_review = st.session_state.get(config_review_state_key)
if stale_review and stale_review.get("selection_key") != current_selection_key:
    st.session_state.pop(config_review_state_key, None)
stale_run = st.session_state.get(workflow_run_state_key)
if stale_run and stale_run.get("selection_key") != current_selection_key:
    st.session_state.pop(workflow_run_state_key, None)

baseline_seg_df = load_algorithmic_segments(selected_run, dataset_id=dataset_id, deployment_id=deployment_id)
baseline_seg_df = _normalize_segment_datetimes(baseline_seg_df, deployment_tz_name)
if baseline_seg_df.empty:
    st.error(
        f"No algorithmic segment table was found for `{dataset_id} / {deployment_id}` in run `{analysis_id}`."
    )
    st.stop()
base_algo_cfg = default_algorithmic_cfg_from_config(config, analysis_id=analysis_id)
supervised_prediction_df = _load_supervised_predictions_for_run(selected_run, dataset_id=dataset_id, deployment_id=deployment_id)

real_data_pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
active_seg_df = preview_store.get("seg_df") if isinstance(preview_store.get("seg_df"), pd.DataFrame) and not preview_store.get("seg_df").empty else baseline_seg_df
active_seg_df = _normalize_segment_datetimes(active_seg_df, deployment_tz_name)
active_algo_cfg = copy.deepcopy(preview_store.get("algo_cfg") or base_algo_cfg)
active_context_defs = copy.deepcopy(preview_store.get("context_filter_definitions") or dict(run_cfg.get("context_filter_definitions") or {}))
active_signal_data = copy.deepcopy(getattr(data_pkl, "signal_data", {}) or {})
active_signal_info = copy.deepcopy(getattr(data_pkl, "signal_info", {}) or {})
if preview_store:
    for signal_name, signal_df in dict(preview_store.get("signal_data") or {}).items():
        active_signal_data[signal_name] = signal_df
    for signal_name, signal_meta in dict(preview_store.get("signal_info") or {}).items():
        active_signal_info[signal_name] = signal_meta
active_signal_data, active_signal_info = _augment_review_signal_data(active_signal_data, active_signal_info)
active_signal_data, active_signal_info = _augment_segment_summary_signal_data(active_signal_data, active_signal_info, active_seg_df)

positive_names = positive_state_names(active_seg_df)
baseline_positive_names = positive_state_names(baseline_seg_df)
if not baseline_positive_names:
    st.error("No positive final-kept algorithmic state events were found in the deployment-wide segmentation run.")
    st.stop()

form_prefix = f"state_review_{analysis_id}_{dataset_id}_{deployment_id}"
selection_options = selection_options_for_positive_states(baseline_positive_names)
pretty_map, event_alias_map = _state_label_maps(config, analysis_id, active_algo_cfg)
baseline_exhaustive_seg_df = _load_exhaustive_segments_for_run(
    selected_run,
    dataset_id=dataset_id,
    deployment_id=deployment_id,
    deployment_tz_name=deployment_tz_name,
)
preferred_selectors = [
    "rest.*",
    "putative_rest.*",
    "find_rest.*",
    "behavior_motionless_resting.*",
    "behavior.*",
    "resting_drift",
    "putative_rest.drift_sleep",
    "find_rest.long_drift",
]
default_selector = next((name for name in preferred_selectors if name in selection_options), selection_options[0])
selected_state_name = st.sidebar.selectbox(
    "Event View",
    options=selection_options,
    index=selection_options.index(default_selector),
    format_func=selector_display_name,
)
visible_subtype_names = []
for state_name in baseline_positive_names:
    if _state_name_matches_selector(state_name, selected_state_name):
        visible_subtype_names.append(str(state_name))

baseline_candidates, default_idx = choose_default_candidate(
    baseline_seg_df,
    state_name=selected_state_name,
    candidate_rank=DEFAULT_CANDIDATE_RANK,
)
if baseline_candidates.empty:
    st.error(f"No deployment-wide final-kept candidates were found for `{selected_state_name}`.")
    st.stop()

candidate_rank_options = [
    int(v) for v in pd.to_numeric(baseline_candidates["segment_rank"], errors="coerce").dropna().astype(int).tolist()
]
default_candidate_rank = candidate_rank_options[default_idx]
candidate_slider_key = f"{form_prefix}_{selected_state_name}_candidate_rank"
pending_candidate_rank_key = f"{candidate_slider_key}__pending"
pending_candidate_rank = st.session_state.pop(pending_candidate_rank_key, None)
if pending_candidate_rank in candidate_rank_options:
    st.session_state[candidate_slider_key] = int(pending_candidate_rank)
if st.session_state.get(candidate_slider_key) not in candidate_rank_options:
    st.session_state[candidate_slider_key] = default_candidate_rank

current_candidate_idx = candidate_rank_options.index(int(st.session_state[candidate_slider_key]))
nav_cols = st.sidebar.columns(2)
if nav_cols[0].button(
    "Previous Rest Candidate",
    key=f"{candidate_slider_key}_prev",
    disabled=current_candidate_idx <= 0,
    width="stretch",
):
    st.session_state[candidate_slider_key] = candidate_rank_options[current_candidate_idx - 1]
    st.rerun()
if nav_cols[1].button(
    "Next Rest Candidate",
    key=f"{candidate_slider_key}_next",
    disabled=current_candidate_idx >= len(candidate_rank_options) - 1,
    width="stretch",
):
    st.session_state[candidate_slider_key] = candidate_rank_options[current_candidate_idx + 1]
    st.rerun()

selected_candidate_rank = st.sidebar.select_slider(
    "Rest Segment ID",
    options=candidate_rank_options,
    value=int(st.session_state[candidate_slider_key]),
    key=candidate_slider_key,
    help="Slide from the first deployment-wide final-kept rest segment ID to the last. Defaults to the 10th final-kept candidate, or the last available one if there are fewer than 10.",
)
candidate_start_series = pd.to_datetime(baseline_candidates["start_datetime"], errors="coerce")
jump_input_key = f"{form_prefix}_{selected_state_name}_jump_start_datetime"
jump_input_pending_key = f"{jump_input_key}__pending"
jump_input_pending_value = st.session_state.pop(jump_input_pending_key, None)
if jump_input_pending_value is not None:
    st.session_state[jump_input_key] = str(jump_input_pending_value)
if jump_input_key not in st.session_state:
    current_row = baseline_candidates.loc[
        pd.to_numeric(baseline_candidates["segment_rank"], errors="coerce").astype(int) == int(selected_candidate_rank)
    ]
    current_start = pd.to_datetime(current_row.iloc[0].get("start_datetime"), errors="coerce") if not current_row.empty else pd.NaT
    st.session_state[jump_input_key] = (
        current_start.isoformat(sep=" ", timespec="seconds")
        if pd.notna(current_start)
        else ""
    )
jump_start_datetime_text = st.sidebar.text_input(
    "Jump to nearest segment start",
    key=jump_input_key,
    help="Enter a datetime and jump to the nearest identified segment start (deployment local time).",
)
if st.sidebar.button("Jump to nearest identified segment", key=f"{jump_input_key}_button", width="stretch"):
    target_ts = pd.to_datetime(jump_start_datetime_text, errors="coerce")
    if pd.isna(target_ts):
        st.sidebar.error("Could not parse datetime. Try formats like `2021-04-04 13:45:00`.")
    else:
        target_ts = _normalize_datetime_to_tz(pd.Series([target_ts]), deployment_tz_name).iloc[0]
        valid_mask = candidate_start_series.notna()
        if not bool(valid_mask.any()):
            st.sidebar.error("No valid segment start datetimes are available in this deployment.")
        else:
            deltas = (candidate_start_series.loc[valid_mask] - target_ts).abs()
            nearest_idx = deltas.idxmin()
            nearest_rank = int(pd.to_numeric(baseline_candidates.loc[nearest_idx, "segment_rank"], errors="coerce"))
            if nearest_rank in candidate_rank_options:
                st.session_state[pending_candidate_rank_key] = nearest_rank
                nearest_start = pd.to_datetime(baseline_candidates.loc[nearest_idx, "start_datetime"], errors="coerce")
                if pd.notna(nearest_start):
                    st.session_state[jump_input_pending_key] = nearest_start.isoformat(sep=" ", timespec="seconds")
                st.rerun()
candidate_row = baseline_candidates.loc[
    pd.to_numeric(baseline_candidates["segment_rank"], errors="coerce").astype(int) == int(selected_candidate_rank)
].iloc[0]
active_rank_series = (
    pd.to_numeric(active_seg_df["segment_rank"], errors="coerce")
    if "segment_rank" in active_seg_df.columns
    else pd.Series(dtype=float, index=active_seg_df.index)
)
active_candidate_matches = active_seg_df.loc[active_rank_series == int(selected_candidate_rank)].copy()
active_candidate_row = active_candidate_matches.iloc[0] if not active_candidate_matches.empty else None

window_hours = st.sidebar.slider("Window Duration (hours)", min_value=1, max_value=72, value=int(DEFAULT_WINDOW_HOURS), step=1)
include_rejected = st.sidebar.checkbox(
    "Show Rejected Base-Pass Candidates",
    value=True,
    help="Overlay drift candidates that passed base thresholds but were rejected by context filters.",
)
activity_aggregation_preset = st.sidebar.selectbox(
    "Daily Activity Aggregation",
    options=["1H", "6H", "12H", "1D", "5D", "Custom"],
    index=3,
    help="Aggregation for the continuous daily-activity and trip-wide rest summaries.",
)
activity_aggregation = (
    st.sidebar.text_input(
        "Custom aggregation",
        value="1D",
        help="Any pandas offset alias, for example `1H`, `6H`, `12H`, `1D`, or `5D`.",
    ).strip()
    if activity_aggregation_preset == "Custom"
    else activity_aggregation_preset
)
show_solar_context = st.sidebar.checkbox(
    "Show Solar Context",
    value=True,
    help="Overlay sunrise, sunset, and night shading on the new trip-scale plots.",
)
wrap_daily_activity = st.sidebar.checkbox(
    "Wrap Daily Activity By Day",
    value=False,
    help="Show the daily summary as a more traditional actogram with one row per day.",
)
size_rest_markers = st.sidebar.checkbox(
    "Scale Rest Markers By Duration",
    value=True,
    help="When on, longer rest segments are shown with larger markers in the trip-wide rest plot.",
)

window_start, window_end = centered_window_bounds(candidate_row, window_hours=window_hours)
edited_algo_cfg, edited_context_defs, ordered_filter_ids, rerun_preview, run_deployment, save_and_rerun = _editable_preview_config(
    active_algo_cfg,
    run_cfg,
    prefix=form_prefix,
)
active_ordered_filter_ids = ordered_filter_ids or [str(key) for key in active_context_defs.keys()]
if rerun_preview or run_deployment:
    cleanup_temp_workspace(preview_store.get("temp_root"))
    st.session_state.pop(config_review_state_key, None)
    preview_algo_cfg = _config_for_filter_phase(
        edited_algo_cfg,
    )
    if rerun_preview:
        preview_label = "selected review window"
    else:
        preview_label = "full deployment with all active context filters"
    with st.spinner(f"Running algorithmic preview for the {preview_label}..."):
        try:
            preview_payload = run_algorithmic_preview_sandbox(
                repo_root=repo_root,
                real_data_pkl_path=real_data_pkl_path,
                dataset_id=dataset_id,
                deployment_id=deployment_id,
                algo_cfg=preview_algo_cfg,
                context_filter_definitions=edited_context_defs,
                preview_start_ts=window_start if rerun_preview else None,
                preview_end_ts=window_end if rerun_preview else None,
            )
            preview_payload["selection_key"] = current_selection_key
            preview_payload["algo_cfg"] = copy.deepcopy(edited_algo_cfg)
            preview_payload["context_filter_definitions"] = copy.deepcopy(edited_context_defs)
            preview_payload["preview_scope"] = "window" if rerun_preview else "full_deployment"
            preview_payload["preview_window_start"] = window_start if rerun_preview else None
            preview_payload["preview_window_end"] = window_end if rerun_preview else None
            preview_payload["filter_phase"] = "full"
            st.session_state["state_event_review_preview"] = preview_payload
            preview_store = preview_payload
            st.success(f"Preview run completed for the {preview_label}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Preview run failed: {type(exc).__name__}: {exc}")
if save_and_rerun:
    with st.spinner("Saving segmentation_runs config and re-running algorithmic segmentation workflow..."):
        try:
            save_result = save_full_algorithmic_config_update(
                config_path=config_path,
                analysis_id=analysis_id,
                algo_cfg=edited_algo_cfg,
                context_filter_definitions=edited_context_defs,
            )
            run_name = str(save_result.get("run_name") or "").strip()
            merged_output_path = str(selected_run.get("merged_segments_path") or "").strip()
            if not merged_output_path:
                merged_output_path = str(Path(selected_run["run_dir"]) / "segments" / "algorithmic_segments.parquet")
            cmd = [
                sys.executable,
                "workflows/10_algorithmic_segmentation.py",
                "--config",
                config_path,
                "--run-name",
                run_name,
                "--output",
                merged_output_path,
                "--context-filter-pass-mode",
                "full",
            ]
            env = dict(os.environ)
            env["SEGMENTATION_RUNS_PATH"] = segmentation_runs_path
            proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, env=env)
            by_dep_dir = Path(str(selected_run.get("by_deployment_dir") or (Path(selected_run["run_dir"]) / "segments" / "by_deployment")))
            seg_name = f"{dataset_id}__{deployment_id}__algorithmic_segments.parquet"
            sum_name = f"{dataset_id}__{deployment_id}__algorithmic_segments_summary.json"
            st.session_state[workflow_run_state_key] = {
                "selection_key": current_selection_key,
                "run_name": run_name,
                "command": " ".join(cmd),
                "returncode": int(proc.returncode),
                "stdout": str(proc.stdout or ""),
                "stderr": str(proc.stderr or ""),
                "merged_output_path": merged_output_path,
                "deployment_segments_path": str(by_dep_dir / seg_name),
                "deployment_summary_path": str(by_dep_dir / sum_name),
            }
            if proc.returncode != 0:
                st.error("Save succeeded, but algorithmic segmentation workflow failed. See workflow outputs below.")
            else:
                st.success("Saved segmentation_runs.yaml and re-ran algorithmic segmentation workflow.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save and re-run algorithmic workflow: {type(exc).__name__}: {exc}")

workflow_run_payload = st.session_state.get(workflow_run_state_key) or {}

baseline_summary, baseline_reason_df, baseline_filter_df = summarize_state_filters(
    baseline_seg_df,
    state_name=selected_state_name,
    ordered_filter_ids=active_ordered_filter_ids,
)
active_summary, active_reason_df, active_filter_df = summarize_state_filters(
    active_seg_df,
    state_name=selected_state_name,
    ordered_filter_ids=active_ordered_filter_ids,
)
candidate_filter_df = candidate_filter_details(
    active_candidate_row if active_candidate_row is not None else candidate_row,
    ordered_filter_ids=active_ordered_filter_ids,
)
baseline_stage_df = stage_dropoff_summary(
    baseline_seg_df,
    state_name=selected_state_name,
    ordered_filter_ids=active_ordered_filter_ids,
    signal_data=getattr(data_pkl, "signal_data", {}) or {},
)
active_stage_df = stage_dropoff_summary(
    active_seg_df,
    state_name=selected_state_name,
    ordered_filter_ids=active_ordered_filter_ids,
    signal_data=active_signal_data,
)

if preview_store:
    if preview_store.get("preview_scope") == "window":
        st.info("Showing preview results from a sandbox rerun on the selected review window only. No changes have been written back to the real deployment `data.pkl`.")
    else:
        st.info("Showing preview results from a sandbox rerun across the full deployment with all active context filters. No changes have been written back to the real deployment `data.pkl`.")
    with st.expander("Preview Run Logs", expanded=False):
        st.text(preview_store.get("stdout") or "")
        if preview_store.get("stderr"):
            st.text(preview_store.get("stderr"))
else:
    st.info("Showing baseline results loaded from the existing run artifacts.")
if workflow_run_payload:
    rc = int(workflow_run_payload.get("returncode", 1))
    status = "success" if rc == 0 else "failed"
    st.subheader("Workflow Run Output")
    st.write(
        {
            "status": status,
            "run_name": workflow_run_payload.get("run_name"),
            "return_code": rc,
            "merged_output_path": workflow_run_payload.get("merged_output_path"),
            "deployment_segments_path": workflow_run_payload.get("deployment_segments_path"),
            "deployment_summary_path": workflow_run_payload.get("deployment_summary_path"),
        }
    )
    with st.expander("Workflow stdout / stderr", expanded=(rc != 0)):
        st.code(str(workflow_run_payload.get("command") or ""), language="bash")
        st.text(workflow_run_payload.get("stdout") or "")
        if workflow_run_payload.get("stderr"):
            st.text(workflow_run_payload.get("stderr"))

activity_budget_mode_state_key = _activity_budget_mode_key(current_selection_key)
activity_budget_mode = str(st.session_state.get(activity_budget_mode_state_key) or "baseline")
full_preview_available = bool(preview_store) and preview_store.get("preview_scope") == "full_deployment"
if not full_preview_available and activity_budget_mode != "baseline":
    st.session_state[activity_budget_mode_state_key] = "baseline"
    activity_budget_mode = "baseline"
show_deployment_summary_plots = full_preview_available

action_cols = st.columns([1.35, 1.15, 1.25, 4.25])
if action_cols[0].button(
    "Recalculate deployment-wide activity budgets",
    width="stretch",
    disabled=not full_preview_available or activity_budget_mode == "active",
):
    st.session_state[activity_budget_mode_state_key] = "active"
    st.rerun()
if action_cols[1].button(
    "Use baseline activity budgets",
    width="stretch",
    disabled=activity_budget_mode != "active",
):
    st.session_state[activity_budget_mode_state_key] = "baseline"
    st.rerun()
save_changes = action_cols[2].button(
    "Save to pkl and netCDF",
    width="stretch",
    disabled=not full_preview_available,
)
if full_preview_available:
    action_cols[3].caption(
        "The full-deployment preview can update the deployment-wide activity budgets and then be committed into the real deployment outputs."
    )
else:
    action_cols[3].caption(
        "Run `Accept base pass and run deployment` first. Window-only previews are diagnostic and cannot be used to recalculate deployment-wide budgets or save."
    )

if save_changes:
    preview_data_pkl_path = str(preview_store.get("temp_pkl_path") or "").strip()
    if not preview_data_pkl_path or not os.path.exists(preview_data_pkl_path):
        st.error("The preview `data.pkl` was not found. Re-run the full-deployment preview before saving.")
    else:
        with st.spinner("Saving preview outputs into the real deployment `data.pkl` and netCDF..."):
            try:
                save_result = persist_preview_algorithmic_outputs(
                    real_data_pkl_path=real_data_pkl_path,
                    preview_data_pkl_path=preview_data_pkl_path,
                    deployment_folder=deployment_folder,
                    deployment_id=deployment_id,
                    write_netcdf=True,
                )
                st.session_state[activity_budget_mode_state_key] = "active"
                save_message = (
                    f"Saved {save_result['saved_event_count']} algorithmic events and "
                    f"{save_result['saved_label_rows']} label rows to the real deployment outputs."
                )
                if save_result.get("netcdf_written"):
                    save_message += f" NetCDF refreshed at `{save_result.get('netcdf_path')}`."
                elif save_result.get("netcdf_error"):
                    save_message += f" NetCDF export failed: {save_result['netcdf_error']}."
                st.success(save_message)
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to save preview outputs: {type(exc).__name__}: {exc}")

st.subheader("Algorithmic Criteria: Base Detector")
st.caption(
    "Buttons above drive the flow: Preview window, Run algorithmic segmentation for deployment, "
    "and Save to segmentation_runs + re-run workflow."
)
st.code(
    build_review_yaml_snippet_from_algo_cfg(
        config,
        analysis_id,
        edited_algo_cfg,
        context_filter_definitions=edited_context_defs,
    ),
    language="yaml",
)

st.caption(
    "Workflow steps: 0) signal preparation, 1) `unfiltered_rest_candidate`, 2) `filtered_rest_candidate`, "
    "3) `unfiltered_putative_rest`, 4) `rest`, 5) compare drop-off across those steps."
)

st.subheader("24-Hour Review Window")
review_colors = _review_color_controls(
    include_rejected=include_rejected,
    subtype_state_names=visible_subtype_names,
    pretty_map=pretty_map,
)
_render_review_color_legend(
    review_colors,
    include_rejected=include_rejected,
    subtype_state_names=visible_subtype_names,
    pretty_map=pretty_map,
)
save_color_cols = st.columns([1.4, 4.6])
if save_color_cols[0].button("Save event colors", width="stretch"):
    color_updates: dict[str, str] = {}
    for subtype_name in visible_subtype_names:
        color_key = f"__review_final_kept__::{subtype_name}"
        chosen_color = review_colors.get(color_key)
        if not chosen_color:
            continue
        aliases = {str(subtype_name), *(event_alias_map.get(str(subtype_name)) or [])}
        for alias in sorted(aliases):
            color_updates[alias] = chosen_color
    if not color_updates:
        st.warning("No subtype colors are active in the current view.")
    else:
        try:
            save_result = _save_event_style_colors(color_mapping_path, color_updates)
            st.success(
                "Saved "
                f"{save_result.get('updated_count', 0)} event style entries"
                + (
                    f" and mirrored {save_result.get('mirrored_count', 0)} existing legacy aliases"
                    if save_result.get("mirrored_count", 0)
                    else ""
                )
                + " to color_mappings.json."
            )
        except Exception as exc:
            st.error(f"Failed to save event colors: {type(exc).__name__}: {exc}")
save_color_cols[1].caption(
    "Saves currently visible subtype colors into `color_mappings.json` for both the displayed state names and their mapped event-key aliases."
)
available_signals = list(active_signal_data.keys())
default_signals_to_plot, default_channel_map = default_review_signals(config, analysis_id, active_signal_data)
saved_review_signals = _load_saved_segmentation_review_signals(param_manager, available_signals)
signal_multiselect_default = (
    saved_review_signals
    or [sig for sig in default_signals_to_plot if sig in available_signals]
    or [sig for sig in ["depth", "stroke_rate"] if sig in available_signals]
    or ([available_signals[0]] if available_signals else [])
)
selected_review_signals = st.multiselect(
    "Signals to visualize",
    options=available_signals,
    default=signal_multiselect_default,
    format_func=_review_signal_display_name,
    help="Defaults to the signals used to derive and review the algorithmic labels. Add extras like stroke_rate, then save them for this deployment if you want that view to persist.",
)
save_signal_cols = st.columns([1, 5])
if save_signal_cols[0].button("Save to signals for review", width="stretch"):
    valid_selected = [str(sig) for sig in selected_review_signals if str(sig) in available_signals]
    if not valid_selected:
        st.warning("Select at least one signal to save.")
    else:
        try:
            param_manager.set_dataset_defaults(
                entries={
                    "segmentation_review_signals": valid_selected,
                    "segmentation_review_signal_order": valid_selected,
                },
                section="segmentation_settings",
            )
            st.success(f"Saved {len(valid_selected)} review signal(s) under segmentation settings.")
        except Exception as exc:
            st.error(f"Failed to save segmentation review signals: {type(exc).__name__}: {exc}")

signals_to_plot = [sig for sig in selected_review_signals if sig in available_signals]
if not signals_to_plot:
    signals_to_plot = [sig for sig in signal_multiselect_default if sig in available_signals]
if not signals_to_plot and available_signals:
    signals_to_plot = [available_signals[0]]
channel_map = {
    signal_name: list(default_channel_map.get(signal_name) or [])
    for signal_name in signals_to_plot
    if signal_name in default_channel_map
}

overlay_events = build_overlay_event_rows(
    active_seg_df,
    state_name=selected_state_name,
    include_rejected_base=include_rejected,
    focused_segment_rank=int(candidate_row.get("segment_rank", 0) or 0),
)
initial_stage_events = active_seg_df.copy()
if not initial_stage_events.empty:
    initial_stage_events = initial_stage_events.loc[
        initial_stage_events.get("segment_family", pd.Series("", index=initial_stage_events.index)).astype(str) == "drift_candidate"
    ].copy()
    if "state_name" in initial_stage_events.columns and selected_state_name:
        if selected_state_name.endswith("benthic"):
            initial_stage_events = initial_stage_events.loc[
                initial_stage_events.get("base_nominal_class", pd.Series("", index=initial_stage_events.index)).astype(str) == "benthic"
            ].copy()
        elif selected_state_name.endswith("drift"):
            initial_stage_events = initial_stage_events.loc[
                initial_stage_events.get("base_nominal_class", pd.Series("", index=initial_stage_events.index)).astype(str) == "drift"
            ].copy()
    initial_stage_events = initial_stage_events[["start_datetime", "end_datetime", "duration_s"]].rename(
        columns={"start_datetime": "datetime", "duration_s": "duration"}
    )
    initial_stage_events["key"] = "__review_initial_segments__"
else:
    initial_stage_events = pd.DataFrame(columns=["datetime", "end_datetime", "duration", "key"])

mask_signal_df = (
    active_signal_data.get("algorithmic_feature_channels")
    if isinstance(active_signal_data.get("algorithmic_feature_channels"), pd.DataFrame)
    else active_signal_data.get("algorithmic_intermediate_channels")
)
unfiltered_events = build_mask_period_rows(mask_signal_df, "is_unfiltered_rest_candidate", "__review_unfiltered_rest__")

review_event_data = pd.concat([overlay_events, initial_stage_events, unfiltered_events], ignore_index=True, sort=False)
review_event_data = review_event_data.dropna(subset=["datetime", "end_datetime"], how="any")
review_event_data["datetime"] = _normalize_datetime_to_tz(review_event_data["datetime"], deployment_tz_name)
review_event_data["end_datetime"] = _normalize_datetime_to_tz(review_event_data["end_datetime"], deployment_tz_name)
review_event_data["type"] = "state"
review_event_data["short_description"] = review_event_data["key"].astype(str)

baseline_overlay_events = build_overlay_event_rows(
    baseline_seg_df,
    state_name=selected_state_name,
    include_rejected_base=include_rejected,
    focused_segment_rank=int(candidate_row.get("segment_rank", 0) or 0),
)
baseline_initial_stage_events = baseline_seg_df.copy()
if not baseline_initial_stage_events.empty:
    baseline_initial_stage_events = baseline_initial_stage_events.loc[
        baseline_initial_stage_events.get("segment_family", pd.Series("", index=baseline_initial_stage_events.index)).astype(str) == "drift_candidate"
    ].copy()
    baseline_initial_stage_events = baseline_initial_stage_events[["start_datetime", "end_datetime", "duration_s"]].rename(
        columns={"start_datetime": "datetime", "duration_s": "duration"}
    )
    baseline_initial_stage_events["key"] = "__review_initial_segments__"
else:
    baseline_initial_stage_events = pd.DataFrame(columns=["datetime", "end_datetime", "duration", "key"])
baseline_review_event_data = pd.concat(
    [baseline_overlay_events, baseline_initial_stage_events, unfiltered_events],
    ignore_index=True,
    sort=False,
)
baseline_review_event_data = baseline_review_event_data.dropna(subset=["datetime", "end_datetime"], how="any")
baseline_review_event_data["datetime"] = _normalize_datetime_to_tz(baseline_review_event_data["datetime"], deployment_tz_name)
baseline_review_event_data["end_datetime"] = _normalize_datetime_to_tz(baseline_review_event_data["end_datetime"], deployment_tz_name)
baseline_review_event_data["type"] = "state"
baseline_review_event_data["short_description"] = baseline_review_event_data["key"].astype(str)

review_pkl = SimpleNamespace(
    signal_data=active_signal_data,
    signal_info=active_signal_info,
    event_data=review_event_data,
    deployment_info=getattr(data_pkl, "deployment_info", {}),
)
baseline_review_pkl = SimpleNamespace(
    signal_data=active_signal_data,
    signal_info=active_signal_info,
    event_data=baseline_review_event_data,
    deployment_info=getattr(data_pkl, "deployment_info", {}),
)

trip_plot_payload = {"daily_activity_fig": go.Figure(), "rest_trip_fig": go.Figure()}
if show_deployment_summary_plots:
    try:
        activity_budget_seg_df = active_seg_df if (full_preview_available and activity_budget_mode == "active") else baseline_seg_df
        trip_plot_payload = continuous_daily_activity_plot(
            data_pkl=review_pkl,
            seg_df=activity_budget_seg_df,
            deployment_id=deployment_id,
            aggregation=activity_aggregation,
            timezone_name=str((getattr(data_pkl, "deployment_info", {}) or {}).get("Time Zone") or "UTC"),
            lat_lon_source=_lat_lon_source_from_data_pkl(data_pkl),
            state_label_map=_rest_state_label_map(pretty_map, event_alias_map),
            dive_depth_threshold_m=float(active_algo_cfg.get("thresholds", {}).get("dive_depth_min_m", 2.0)),
            selected_segment_rank=int(selected_candidate_rank),
            color_mapping_path=color_mapping_path,
            show_solar_context=show_solar_context,
            size_markers_by_duration=size_rest_markers,
            wrap_daily=wrap_daily_activity,
            status_seg_df=active_seg_df,
            window_start=window_start,
            window_end=window_end,
            supervised_prediction_df=supervised_prediction_df,
        )
    except Exception as exc:
        st.error(f"Failed to build daily/trip summary plots for aggregation `{activity_aggregation}`: {type(exc).__name__}: {exc}")
        trip_plot_payload = {"daily_activity_fig": go.Figure(), "rest_trip_fig": go.Figure()}

if show_deployment_summary_plots:
    buoyancy_track_fig = trip_plot_payload.get("buoyancy_track_fig")
    if buoyancy_track_fig is not None and getattr(buoyancy_track_fig, "data", None):
        st.subheader("Buoyancy Shift")
        st.plotly_chart(buoyancy_track_fig, width="stretch")

    supervised_nap_overview_fig = trip_plot_payload.get("supervised_nap_overview_fig")
    if supervised_nap_overview_fig is not None and getattr(supervised_nap_overview_fig, "data", None):
        st.subheader("Putative Nap Overview")
        st.plotly_chart(supervised_nap_overview_fig, width="stretch")

    st.subheader("Daily Activity")
    st.caption(
        "Using "
        + ("full-deployment preview outputs." if full_preview_available and activity_budget_mode == "active" else "baseline saved run outputs.")
    )
    st.plotly_chart(trip_plot_payload["daily_activity_fig"], width="stretch")

    st.subheader("Rest Across Trip")
    if plotly_events is not None:
        clicked_points = plotly_events(
            trip_plot_payload["rest_trip_fig"],
            click_event=True,
            hover_event=False,
            select_event=False,
            key=f"{form_prefix}_rest_trip_clicks",
        )
        if clicked_points:
            clicked = clicked_points[0] or {}
            customdata = clicked.get("customdata")
            clicked_rank = None
            if isinstance(customdata, (list, tuple)) and len(customdata) >= 1:
                clicked_rank = pd.to_numeric(pd.Series([customdata[0]]), errors="coerce").iloc[0]
            if pd.notna(clicked_rank):
                clicked_rank = int(clicked_rank)
                if clicked_rank in candidate_rank_options and clicked_rank != int(st.session_state[candidate_slider_key]):
                    st.session_state[pending_candidate_rank_key] = clicked_rank
                    st.rerun()
    else:
        st.plotly_chart(trip_plot_payload["rest_trip_fig"], width="stretch")
        st.caption("Install or enable `streamlit-plotly-events` to click a rest point and jump directly to that candidate.")

    if plotly_events is not None:
        st.caption("Click any rest point to jump the reviewer to that rest candidate.")

    st.subheader("Exhaustive Time Budget")
    exhaustive_budget_df = _exhaustive_budget_table(baseline_exhaustive_seg_df, pretty_map=pretty_map)
    if exhaustive_budget_df.empty:
        st.caption(
            "No exhaustive segment parquet found yet for this deployment. "
            "Re-run algorithmic segmentation to generate `algorithmic_segments_exhaustive.parquet` outputs."
        )
    else:
        budget_plot_df = exhaustive_budget_df.copy()
        budget_plot_df["Label"] = budget_plot_df["State"].astype(str)
        fig_budget = go.Figure(
            data=[
                go.Bar(
                    x=budget_plot_df["Hours"],
                    y=budget_plot_df["Label"],
                    orientation="h",
                    text=[f"{v:.1f}%" for v in pd.to_numeric(budget_plot_df["Percent"], errors="coerce").fillna(0.0)],
                    textposition="outside",
                    marker=dict(color="#3E7CB1"),
                    hovertemplate=(
                        "%{y}<br>"
                        "Hours: %{x:.2f}<br>"
                        "Percent: %{text}<extra></extra>"
                    ),
                )
            ]
        )
        fig_budget.update_layout(
            height=max(280, 46 * len(budget_plot_df)),
            margin=dict(l=12, r=12, t=8, b=8),
            xaxis_title="Hours",
            yaxis_title="State",
        )
        st.plotly_chart(fig_budget, width="stretch")
        st.dataframe(exhaustive_budget_df, width="stretch", hide_index=True)

        daily_budget_df = _exhaustive_daily_budget_table(baseline_exhaustive_seg_df, pretty_map=pretty_map)
        if not daily_budget_df.empty:
            st.caption("Exhaustive actogram (daily stacked hours by state)")
            preferred_order = [
                "active_uw_swimming",
                "calm_uw_gliding",
                "resting_surface",
                "resting_benthic",
                "long_drift",
                "active_surface_swimming",
                "calm_surface_gliding",
                "unscorable",
            ]
            present_nominals = daily_budget_df["nominal_class"].astype(str).unique().tolist()
            ordered_nominals = [name for name in preferred_order if name in present_nominals] + [
                name for name in present_nominals if name not in preferred_order
            ]
            fig_actogram = go.Figure()
            for nominal in ordered_nominals:
                sub = daily_budget_df.loc[daily_budget_df["nominal_class"].astype(str) == str(nominal)].copy()
                if sub.empty:
                    continue
                pretty_label = str((pretty_map or {}).get(str(nominal), str(nominal)))
                color_key = f"__review_final_kept__::{nominal}"
                color = str(review_colors.get(color_key) or REVIEW_COLOR_DEFAULTS.get(color_key) or REVIEW_COLOR_DEFAULTS.get("__review_final_kept__", "#2C7BB6"))
                fig_actogram.add_trace(
                    go.Bar(
                        name=pretty_label,
                        x=sub["date"],
                        y=sub["hours"],
                        marker=dict(color=color),
                        hovertemplate=(
                            "%{x|%Y-%m-%d}<br>"
                            f"{pretty_label}<br>"
                            "Hours: %{y:.2f}<extra></extra>"
                        ),
                    )
                )
            fig_actogram.update_layout(
                barmode="stack",
                xaxis_title="Date",
                yaxis_title="Hours/day",
                margin=dict(l=12, r=12, t=8, b=8),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            )
            st.plotly_chart(fig_actogram, width="stretch")

state_annotations = {
    "__review_unfiltered_rest__": {
        "signal": "all",
        "color": _hex_to_rgba(review_colors["__review_unfiltered_rest__"], 0.95),
        "shade_mode": "fill_trace",
        "shade_opacity": 0.6,
        "line_width": 6,
        "name": "1) unfiltered_rest_candidate",
    },
    "__review_initial_segments__": {
        "signal": "all",
        "color": _hex_to_rgba(review_colors["__review_initial_segments__"], 0.95),
        "shade_mode": "fill_trace",
        "shade_opacity": 0.6,
        "line_width": 7,
        "name": "2) filtered_rest_candidate",
    },
    "__review_pre_context__": {
        "signal": "all",
        "color": _hex_to_rgba(review_colors["__review_pre_context__"], 0.95),
        "shade_mode": "fill_trace",
        "shade_opacity": 0.6,
        "line_width": 8,
        "name": "3) unfiltered_putative_rest",
    },
    "__review_focus__": {
        "signal": "all",
        "color": "rgba(241, 196, 15, 1.0)",
        "shade_mode": "",
        "draw_line": True,
        "line_opacity": 1.0,
        "line_width": 16,
        "start_marker": True,
        "start_marker_symbol": "circle-open",
        "start_marker_size": 12,
        "start_marker_opacity": 1.0,
        "name": "Focused candidate",
        "showlegend": False,
    },
}
final_kept_keys = sorted(
    {str(key) for key in review_event_data.get("key", pd.Series(dtype=str)).astype(str).tolist() if str(key).startswith("__review_final_kept__")}
)
for final_key in final_kept_keys:
    color_key = final_key if final_key in review_colors else "__review_final_kept__"
    state_annotations[final_key] = {
        "signal": "all",
        "color": _hex_to_rgba(review_colors.get(color_key, "#2C7BB6"), 0.95),
        "shade_mode": "fill_trace",
        "shade_opacity": 0.6,
        "line_width": 10,
        "name": _review_color_label(final_key),
    }
if include_rejected:
    state_annotations["__review_rejected__"] = {
        "signal": "all",
        "color": _hex_to_rgba(review_colors["__review_rejected__"], 0.95),
        "shade_mode": "fill_trace",
        "shade_opacity": 0.6,
        "line_width": 8,
        "name": "Rejected base-pass events",
    }

adjusted_state_annotations = copy.deepcopy(state_annotations)

st.markdown("**Adjusted Settings Algorithmic Output Review**")
st.caption(
    "This reflects the current preview settings for the selected window."
)
adjusted_fig = plot_tag_data_interactive(
    data_pkl=review_pkl,
    signals=signals_to_plot,
    channels=channel_map or None,
    time_range=(window_start, window_end),
    state_annotations=adjusted_state_annotations,
    color_mapping_path=color_mapping_path,
    target_sampling_rate=2,
    zoom_start_time=window_start,
    zoom_end_time=window_end,
    zoom_range_selector_channel=None,
    include_blank_row=False,
    preserve_signal_order=True,
    render_state_on_signal_rows=True,
    state_annotation_channel_mode=None,
    state_annotation_channel_line_width=3.0,
)
_add_context_marker_traces(adjusted_fig, signals_to_plot, active_seg_df)
adjusted_base_plot = getattr(adjusted_fig, "figure", adjusted_fig)
adjusted_base_plot.update_layout(showlegend=False)
st.plotly_chart(adjusted_base_plot, width="stretch")

st.markdown("**Previously Run Algorithmic Output Review**")
st.caption("This is from the existing run artifacts before the current adjusted preview.")
baseline_fig = plot_tag_data_interactive(
    data_pkl=baseline_review_pkl,
    signals=signals_to_plot,
    channels=channel_map or None,
    time_range=(window_start, window_end),
    state_annotations=state_annotations,
    color_mapping_path=color_mapping_path,
    target_sampling_rate=2,
    zoom_start_time=window_start,
    zoom_end_time=window_end,
    zoom_range_selector_channel=None,
    include_blank_row=False,
    preserve_signal_order=True,
    render_state_on_signal_rows=True,
    state_annotation_channel_mode=None,
    state_annotation_channel_line_width=3.0,
)
_add_context_marker_traces(baseline_fig, signals_to_plot, baseline_seg_df)
baseline_base_plot = getattr(baseline_fig, "figure", baseline_fig)
baseline_base_plot.update_layout(showlegend=False)
st.plotly_chart(baseline_base_plot, width="stretch")

stage_chart = _stage_count_barplot(active_stage_df, review_colors)
st.plotly_chart(stage_chart, width="stretch")

top_left, top_right = st.columns([1.0, 1.2], gap="large")

with top_left:
    st.subheader("Focused Candidate")
    preview_in_scope = active_candidate_row is not None
    if preview_store and not preview_in_scope:
        st.caption("This deployment-wide candidate is outside the current preview subset. Timing is from the baseline segmentation run.")
    candidate_diag = {
        "Segment Rank": int(candidate_row.get("segment_rank", 0) or 0),
        "State": str(candidate_row.get("state_name") or ""),
        "Start": pd.to_datetime(candidate_row.get("start_datetime"), errors="coerce"),
        "End": pd.to_datetime(candidate_row.get("end_datetime"), errors="coerce"),
        "Duration (s)": float(candidate_row.get("duration_s", float("nan"))),
        "Preview In Scope": "yes" if preview_in_scope else ("no" if preview_store else "baseline"),
        "Base Pass": bool(active_candidate_row.get("base_keep_filtered", False)) if preview_in_scope else bool(candidate_row.get("base_keep_filtered", False)),
        "Context Pass": bool(active_candidate_row.get("context_keep", True)) if preview_in_scope else bool(candidate_row.get("context_keep", True)),
        "Final Kept": bool(active_candidate_row.get("keep_filtered", False)) if preview_in_scope else bool(candidate_row.get("keep_filtered", False)),
        "Reject Reason(s)": str(active_candidate_row.get("context_reject_reason") or "") if preview_in_scope else str(candidate_row.get("context_reject_reason") or ""),
        "Triggered Filters": str(active_candidate_row.get("context_filter_ids") or "") if preview_in_scope else str(candidate_row.get("context_filter_ids") or ""),
    }
    st.dataframe(
        _field_value_df(list(candidate_diag.items())),
        width="stretch",
        hide_index=True,
    )

    covariate_fields = [
        "drift_rate_ms",
        "segment_mean_stroke_rate_spm",
        "inferred_buoyancy_value",
        "inferred_trip_phase",
        "elapsed_days",
        "start_depth_m",
        "end_depth_m",
        "curvature_diff_ms",
        "mean_d1_end_ms",
    ]
    covariate_source = active_candidate_row if preview_in_scope else candidate_row
    covariate_rows = [{"Field": field, "Value": covariate_source.get(field)} for field in covariate_fields if field in covariate_source.index and pd.notna(covariate_source.get(field))]
    if covariate_rows:
        st.caption("Candidate covariates")
        st.dataframe(
            _field_value_df([(row["Field"], row["Value"]) for row in covariate_rows]),
            width="stretch",
            hide_index=True,
        )
    if not candidate_filter_df.empty:
        candidate_filter_df["passed"] = candidate_filter_df["passed"].map({True: "pass", False: "fail"})
        st.caption("Per-filter result for this candidate")
        st.dataframe(candidate_filter_df, width="stretch", hide_index=True)

with top_right:
    st.subheader("Summary Comparison")
    daily_rest_map_fig = trip_plot_payload.get("daily_rest_map_fig")
    if daily_rest_map_fig is not None and getattr(daily_rest_map_fig, "data", None):
        map_clicked_points = []
        if plotly_events is not None:
            map_clicked_points = plotly_events(
                daily_rest_map_fig,
                click_event=True,
                hover_event=False,
                select_event=False,
                key=f"{form_prefix}_daily_rest_map_clicks",
            )
        else:
            st.plotly_chart(daily_rest_map_fig, width="stretch")
        if map_clicked_points:
            map_clicked = map_clicked_points[0] or {}
            map_customdata = map_clicked.get("customdata")
            map_clicked_rank = None
            if isinstance(map_customdata, (list, tuple)) and len(map_customdata) >= 1:
                map_clicked_rank = pd.to_numeric(pd.Series([map_customdata[0]]), errors="coerce").iloc[0]
            if pd.notna(map_clicked_rank):
                map_clicked_rank = int(map_clicked_rank)
                if map_clicked_rank in candidate_rank_options and map_clicked_rank != int(st.session_state[candidate_slider_key]):
                    st.session_state[pending_candidate_rank_key] = map_clicked_rank
                    st.rerun()
        st.caption("Click a putative-rest map point to inspect that candidate.")
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Base-Pass",
        f"{active_summary['base_pass_count']:,}",
        delta=active_summary["base_pass_count"] - baseline_summary["base_pass_count"],
    )
    metric_cols[1].metric(
        "Final-Kept",
        f"{active_summary['final_kept_count']:,}",
        delta=active_summary["final_kept_count"] - baseline_summary["final_kept_count"],
    )
    metric_cols[2].metric(
        "Context-Rejected",
        f"{active_summary['context_rejected_count']:,}",
        delta=active_summary["context_rejected_count"] - baseline_summary["context_rejected_count"],
    )
    metric_cols[3].metric(
        "Reject Rate",
        f"{active_summary['reject_rate_pct']:.1f}%",
        delta=f"{active_summary['reject_rate_pct'] - baseline_summary['reject_rate_pct']:.1f}%",
    )

    tab_reasons, tab_filters, tab_stages, tab_yaml = st.tabs(["Reject Reasons", "Per-Filter Impact", "Dropout Phase", "Config Snippet"])
    with tab_reasons:
        reason_cols = st.columns(2)
        with reason_cols[0]:
            st.caption("Baseline")
            st.dataframe(baseline_reason_df, width="stretch", hide_index=True)
        with reason_cols[1]:
            st.caption("Preview" if preview_store else "Active")
            st.dataframe(active_reason_df, width="stretch", hide_index=True)
    with tab_filters:
        filter_cols = st.columns(2)
        with filter_cols[0]:
            st.caption("Baseline")
            st.dataframe(baseline_filter_df, width="stretch", hide_index=True)
        with filter_cols[1]:
            st.caption("Preview" if preview_store else "Active")
            st.dataframe(active_filter_df, width="stretch", hide_index=True)
    with tab_stages:
        stage_cols = st.columns(2)
        with stage_cols[0]:
            st.caption("Baseline")
            st.dataframe(baseline_stage_df, width="stretch", hide_index=True)
        with stage_cols[1]:
            st.caption("Preview" if preview_store else "Active")
            st.dataframe(active_stage_df, width="stretch", hide_index=True)
    with tab_yaml:
        st.code(
            build_review_yaml_snippet_from_algo_cfg(
                config,
                analysis_id,
                active_algo_cfg,
            ),
            language="yaml",
        )

with st.expander("Run Context", expanded=False):
    st.write(
        {
            "analysis_id": analysis_id,
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "run_dir": selected_run["run_dir"],
            "window_start": window_start,
            "window_end": window_end,
            "signals": signals_to_plot,
            "preview_active": bool(preview_store),
        }
    )
