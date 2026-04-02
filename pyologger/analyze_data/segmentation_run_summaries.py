from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pyologger.analyze_data.dataset_state_summaries import (
    DEFAULT_REST_KEY_MAP,
    _nearest_location_at_times,
    _pick_depth_frame,
    _pick_location_frame,
    _extract_algorithmic_feature_snapshot,
)
from pyologger.utils.solar_utils import sunrise_sunset_local_hours


CANONICAL_BEHAVIOR_MAP = {
    "surface_sleep": "Resting",
    "benthic_sleep": "Resting",
    "drift_sleep": "Resting",
    "putative_rest": "Resting",
    "sleep": "Resting",
    "motionless_resting": "Resting",
    "calm_resting": "Resting",
    "resting": "Resting",
    "surfacing": "Vigilant",
    "diving": "Active",
    "active": "Active",
    "wake": "Active",
    "walking": "Active",
    "running": "Active",
    "trotting": "Active",
    "swimming": "Active",
    "feeding": "Feeding",
    "drinking": "Drinking",
    "calm": "Calm",
    "vigilant": "Vigilant",
    "scanning": "Vigilant",
}


def _normalize_dt(values, tz_name: str) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce")
    if not isinstance(dt, pd.Series):
        dt = pd.Series(dt)
    try:
        if getattr(dt.dt, "tz", None) is None:
            return dt.dt.tz_localize(tz_name)
        return dt.dt.tz_convert(tz_name)
    except Exception:
        return dt


def _canonical_algorithmic_subtype(label_text: Any, rest_key_map: dict[str, str]) -> str | None:
    text = str(label_text or "").strip().lower()
    if not text or text in {"not_sleep", "find_rest.not_sleep"}:
        return None
    for token, subtype in rest_key_map.items():
        if str(token).lower() in text:
            return str(subtype)
    return str(label_text or "").strip() or None


def _sort_values_resilient(
    df: pd.DataFrame,
    by: str | list[str],
    *,
    ascending: bool | list[bool] = True,
    kind: str = "stable",
) -> pd.DataFrame:
    by_cols = [by] if isinstance(by, str) else list(by)
    out = df
    try:
        return out.sort_values(by_cols, ascending=ascending, kind=kind)
    except TypeError:
        out = out.copy()
        for col in by_cols:
            if col in out.columns and isinstance(out[col].dtype, pd.CategoricalDtype):
                out[col] = out[col].astype("string")
        return out.sort_values(by_cols, ascending=ascending, kind=kind)


def _detect_supervised_positive_label(labels: list[str]) -> str | None:
    label_texts = [str(x).strip() for x in labels if str(x).strip()]
    for candidate in label_texts:
        if candidate.lower() == "sleep":
            return candidate
    for candidate in label_texts:
        if "sleep" in candidate.lower():
            return candidate
    return None


def _algorithmic_method_rows(seg_df: pd.DataFrame, rest_key_map: dict[str, str]) -> pd.DataFrame:
    if seg_df is None or seg_df.empty:
        return pd.DataFrame()
    work = seg_df.copy()
    work = _coerce_parquet_friendly(work)
    for col in ["start_datetime", "end_datetime", "segment_midpoint_datetime"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")
    if "segment_midpoint_datetime" not in work.columns and {"start_datetime", "end_datetime"}.issubset(work.columns):
        work["segment_midpoint_datetime"] = work["start_datetime"] + (work["end_datetime"] - work["start_datetime"]) / 2
    work["duration_s"] = pd.to_numeric(work.get("duration_s"), errors="coerce")
    state_col = "state_name" if "state_name" in work.columns else ("label_name" if "label_name" in work.columns else None)
    if state_col is None:
        return pd.DataFrame()
    work["state_label"] = work[state_col].astype(str)
    work["state_subtype"] = work["state_label"].map(lambda value: _canonical_algorithmic_subtype(value, rest_key_map))
    work["method"] = "algorithmic"
    work["method_variant"] = "algorithmic_segments"
    keep_filtered = work.get("keep_filtered", pd.Series(False, index=work.index))
    if not isinstance(keep_filtered, pd.Series):
        keep_filtered = pd.Series(keep_filtered, index=work.index)
    keep_filtered = keep_filtered.reindex(work.index)
    work["positive_label"] = work["state_subtype"].notna() & keep_filtered.eq(True)
    return work


def _algorithmic_exhaustive_method_rows(seg_df: pd.DataFrame, rest_key_map: dict[str, str]) -> pd.DataFrame:
    if seg_df is None or seg_df.empty:
        return pd.DataFrame()
    work = seg_df.copy()
    work = _coerce_parquet_friendly(work)
    for col in ["start_datetime", "end_datetime", "segment_midpoint_datetime"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")
    if "segment_midpoint_datetime" not in work.columns and {"start_datetime", "end_datetime"}.issubset(work.columns):
        work["segment_midpoint_datetime"] = work["start_datetime"] + (work["end_datetime"] - work["start_datetime"]) / 2
    work["duration_s"] = pd.to_numeric(work.get("duration_s"), errors="coerce")
    state_col = (
        "nominal_class"
        if "nominal_class" in work.columns
        else ("label_name" if "label_name" in work.columns else None)
    )
    if state_col is None:
        return pd.DataFrame()
    work["state_label"] = work[state_col].astype(str).str.strip()
    work["state_subtype"] = work["state_label"].map(lambda value: _canonical_algorithmic_subtype(value, rest_key_map))
    work["method"] = "algorithmic"
    work["method_variant"] = "algorithmic_segments"
    work["positive_label"] = work["state_subtype"].notna()
    return work


def _supervised_method_rows(spdf: pd.DataFrame) -> pd.DataFrame:
    if spdf is None or spdf.empty:
        return pd.DataFrame()
    required = {"dataset_id", "deployment_id", "window_start", "window_end"}
    if not required.issubset(spdf.columns):
        return pd.DataFrame()
    work = spdf.copy()
    work = _coerce_parquet_friendly(work)
    work["window_start"] = pd.to_datetime(work["window_start"], errors="coerce")
    work["window_end"] = pd.to_datetime(work["window_end"], errors="coerce")
    label_col = "predicted_label" if "predicted_label" in work.columns else ("final_behavior" if "final_behavior" in work.columns else None)
    if label_col is None:
        return pd.DataFrame()
    work["state_label"] = work[label_col].astype(str).str.strip()
    positive_label = _detect_supervised_positive_label(work["state_label"].tolist())
    work["duration_s"] = (work["window_end"] - work["window_start"]).dt.total_seconds()
    work["start_datetime"] = work["window_start"]
    work["end_datetime"] = work["window_end"]
    work["segment_midpoint_datetime"] = work["start_datetime"] + (work["end_datetime"] - work["start_datetime"]) / 2
    work["state_subtype"] = np.where(work["state_label"].eq(positive_label), "putative_rest", pd.NA)
    work["method"] = "supervised"
    work["method_variant"] = work.get("variant_id", pd.Series("rf_full", index=work.index)).astype(str)
    work["positive_label"] = work["state_label"].eq(positive_label)
    return work


def _build_segmentation_summary(seg_df: pd.DataFrame, spdf: pd.DataFrame, rest_key_map: dict[str, str]) -> pd.DataFrame:
    algo = _algorithmic_method_rows(seg_df, rest_key_map)
    sup = _supervised_method_rows(spdf)
    frames = []
    if not algo.empty:
        frames.append(algo)
    if not sup.empty:
        frames.append(sup)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    keep_cols = [
        "dataset_id",
        "deployment_id",
        "method",
        "method_variant",
        "start_datetime",
        "end_datetime",
        "segment_midpoint_datetime",
        "duration_s",
        "state_label",
        "state_subtype",
        "positive_label",
        "segment_rank",
        "keep_filtered",
        "base_keep_filtered",
        "context_keep",
        "nominal_class",
        "base_nominal_class",
        "drift_rate_ms",
        "pred_confidence",
        "window_key",
        "variant_label",
        "variant_type",
        "rf_context_keep",
        "rf_context_filter_ids",
        "rf_context_reject_reason",
    ]
    existing_keep_cols = [col for col in keep_cols if col in combined.columns]
    remaining = [col for col in combined.columns if col not in existing_keep_cols]
    result = combined[existing_keep_cols + remaining]
    result = _coerce_parquet_friendly(result)
    sort_cols = [c for c in ["dataset_id", "deployment_id", "start_datetime"] if c in result.columns]
    if sort_cols:
        try:
            result = _sort_values_resilient(result, sort_cols, kind="stable")
        except TypeError:
            for col in sort_cols:
                if isinstance(result[col].dtype, pd.CategoricalDtype):
                    result[col] = result[col].astype("string")
            result = _sort_values_resilient(result, sort_cols, kind="stable")
    return result.reset_index(drop=True)


def _overlap_seconds(start_a, end_a, start_b, end_b) -> float:
    if pd.isna(start_a) or pd.isna(end_a) or pd.isna(start_b) or pd.isna(end_b):
        return 0.0
    start = max(pd.Timestamp(start_a), pd.Timestamp(start_b))
    end = min(pd.Timestamp(end_a), pd.Timestamp(end_b))
    seconds = (end - start).total_seconds()
    return float(max(seconds, 0.0))


def _annotate_overlap_flags(rest_df: pd.DataFrame) -> pd.DataFrame:
    if rest_df.empty:
        return rest_df
    out = rest_df.copy()
    out["overlaps_algorithmic"] = False
    out["algorithmic_overlap_s"] = 0.0
    out["weak_confusion"] = pd.NA
    for (dataset_id, deployment_id), dep in out.groupby(["dataset_id", "deployment_id"], sort=False):
        algo = dep.loc[dep["method"].astype(str) == "algorithmic"].copy()
        sup = dep.loc[dep["method"].astype(str) == "supervised"].copy()
        if algo.empty and sup.empty:
            continue
        for idx, row in dep.iterrows():
            if str(row.get("method")) == "supervised":
                overlap_s = 0.0
                for _, algo_row in algo.iterrows():
                    overlap_s = max(overlap_s, _overlap_seconds(row["start_datetime"], row["end_datetime"], algo_row["start_datetime"], algo_row["end_datetime"]))
                out.at[idx, "overlaps_algorithmic"] = overlap_s > 0
                out.at[idx, "algorithmic_overlap_s"] = overlap_s
                out.at[idx, "weak_confusion"] = "tp" if overlap_s > 0 else "fp"
            elif str(row.get("method")) == "algorithmic":
                overlap_s = 0.0
                for _, sup_row in sup.iterrows():
                    overlap_s = max(overlap_s, _overlap_seconds(row["start_datetime"], row["end_datetime"], sup_row["start_datetime"], sup_row["end_datetime"]))
                out.at[idx, "algorithmic_overlap_s"] = overlap_s
                out.at[idx, "weak_confusion"] = "tp" if overlap_s > 0 else "fn"
    return out


def _build_daily_activity_rows(
    method_rows: pd.DataFrame,
    data_pkl,
    deployment_id: str,
    tz_name: str,
    dive_depth_threshold_m: float,
) -> pd.DataFrame:
    depth_df = _pick_depth_frame(data_pkl, tz_name)
    if depth_df.empty:
        return pd.DataFrame()
    location_df = _pick_location_frame(data_pkl, tz_name)
    deployment_info = getattr(data_pkl, "deployment_info", {}) or {}
    fallback_lat = pd.to_numeric(pd.Series([deployment_info.get("Deployment Latitude")]), errors="coerce").iloc[0]
    fallback_lon = pd.to_numeric(pd.Series([deployment_info.get("Deployment Longitude")]), errors="coerce").iloc[0]

    work = depth_df.copy()
    sample_step_s = work["datetime"].diff().dt.total_seconds().median()
    if pd.isna(sample_step_s) or float(sample_step_s) <= 0:
        sample_step_s = 60.0
    work["duration_h"] = work["datetime"].shift(-1).sub(work["datetime"]).dt.total_seconds().fillna(sample_step_s) / 3600.0
    work["state_label"] = np.where(
        pd.to_numeric(work["depth_value"], errors="coerce").fillna(np.inf) < float(dive_depth_threshold_m),
        "surfacing",
        "diving",
    )
    for _, row in method_rows.iterrows():
        start_dt = row.get("start_datetime")
        end_dt = row.get("end_datetime")
        state_label = _coalesce_state_label(row.get("state_subtype"), row.get("state_label"))
        if pd.isna(start_dt) or pd.isna(end_dt) or not state_label:
            continue
        mask = (work["datetime"] >= pd.Timestamp(start_dt)) & (work["datetime"] <= pd.Timestamp(end_dt))
        work.loc[mask, "state_label"] = str(state_label)
    work["local_date"] = work["datetime"].dt.date
    summary = (
        work.groupby(["local_date", "state_label"], dropna=False, sort=False)["duration_h"]
        .sum()
        .reset_index()
    )
    if location_df.empty:
        location_summary = pd.DataFrame({"local_date": summary["local_date"].drop_duplicates().tolist()})
        location_summary["mean_latitude"] = fallback_lat
        location_summary["mean_longitude"] = fallback_lon
    else:
        location_df = location_df.copy()
        location_df["local_date"] = location_df["datetime"].dt.date
        location_summary = (
            location_df.groupby("local_date", dropna=False, sort=False)[["latitude", "longitude"]]
            .mean()
            .rename(columns={"latitude": "mean_latitude", "longitude": "mean_longitude"})
            .reset_index()
        )
        if not pd.isna(fallback_lat):
            location_summary["mean_latitude"] = location_summary["mean_latitude"].fillna(float(fallback_lat))
        if not pd.isna(fallback_lon):
            location_summary["mean_longitude"] = location_summary["mean_longitude"].fillna(float(fallback_lon))
    out = summary.merge(location_summary, on="local_date", how="left")
    out["deployment_id"] = str(deployment_id)
    out["sunrise_local_hour"] = out.apply(
        lambda row: sunrise_sunset_local_hours(
            row["local_date"],
            row.get("mean_latitude", np.nan),
            row.get("mean_longitude", np.nan),
            tz_name,
        )[0],
        axis=1,
    )
    out["sunset_local_hour"] = out.apply(
        lambda row: sunrise_sunset_local_hours(
            row["local_date"],
            row.get("mean_latitude", np.nan),
            row.get("mean_longitude", np.nan),
            tz_name,
        )[1],
        axis=1,
    )
    return out


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _coalesce_state_label(state_subtype: Any, state_label: Any) -> str | None:
    if pd.notna(state_subtype):
        subtype_text = _safe_text(state_subtype)
        if subtype_text:
            return subtype_text
    label_text = _safe_text(state_label)
    return label_text or None


def _canonical_behavior_label(raw_label: Any, mapping: dict[str, str]) -> str:
    text = _safe_text(raw_label)
    lowered = text.lower()
    if not lowered:
        return "Unknown"
    for token, canonical in mapping.items():
        if str(token).lower() in lowered:
            return str(canonical)
    return f"Unmapped: {text}"


def _as_method_label(method: str, method_variant: str) -> str:
    method = _safe_text(method)
    variant = _safe_text(method_variant)
    if method == "algorithmic":
        return f"Algorithmic ({variant})"
    if method == "supervised":
        return f"Supervised ({variant})"
    return f"{method} ({variant})"


def _build_method_budget_rows(
    method_rows: pd.DataFrame,
    data_pkl,
    dataset_id: str,
    deployment_id: str,
    tz_name: str,
    dive_depth_threshold_m: float,
    canonical_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    depth_df = _pick_depth_frame(data_pkl, tz_name)
    if depth_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    location_df = _pick_location_frame(data_pkl, tz_name)
    deployment_info = getattr(data_pkl, "deployment_info", {}) or {}
    fallback_lat = pd.to_numeric(pd.Series([deployment_info.get("Deployment Latitude")]), errors="coerce").iloc[0]
    fallback_lon = pd.to_numeric(pd.Series([deployment_info.get("Deployment Longitude")]), errors="coerce").iloc[0]

    work = depth_df.copy()
    sample_step_s = work["datetime"].diff().dt.total_seconds().median()
    if pd.isna(sample_step_s) or float(sample_step_s) <= 0:
        sample_step_s = 60.0
    work["duration_s"] = work["datetime"].shift(-1).sub(work["datetime"]).dt.total_seconds().fillna(sample_step_s)
    work["duration_h"] = work["duration_s"] / 3600.0
    work["state_label_raw"] = np.where(
        pd.to_numeric(work["depth_value"], errors="coerce").fillna(np.inf) < float(dive_depth_threshold_m),
        "surfacing",
        "diving",
    )
    if not method_rows.empty:
        method_rows = _sort_values_resilient(method_rows, "start_datetime", kind="stable")
        for _, row in method_rows.iterrows():
            start_dt = row.get("start_datetime")
            end_dt = row.get("end_datetime")
            state_label = _coalesce_state_label(row.get("state_subtype"), row.get("state_label"))
            if pd.isna(start_dt) or pd.isna(end_dt) or not state_label:
                continue
            mask = (work["datetime"] >= pd.Timestamp(start_dt)) & (work["datetime"] <= pd.Timestamp(end_dt))
            work.loc[mask, "state_label_raw"] = str(state_label)
    method_name = _safe_text(method_rows.get("method", pd.Series(["unknown"])).iloc[0] if not method_rows.empty else "unknown")
    method_variant = _safe_text(method_rows.get("method_variant", pd.Series(["unknown"])).iloc[0] if not method_rows.empty else "unknown")
    work["state_label_canonical"] = work["state_label_raw"].map(lambda value: _canonical_behavior_label(value, canonical_map))
    work["local_date"] = work["datetime"].dt.date
    work["hour_of_day"] = work["datetime"].dt.hour.astype(int)

    if location_df.empty:
        location_summary = pd.DataFrame({"local_date": work["local_date"].drop_duplicates().tolist()})
        location_summary["mean_latitude"] = fallback_lat
        location_summary["mean_longitude"] = fallback_lon
    else:
        location_df = location_df.copy()
        location_df["local_date"] = location_df["datetime"].dt.date
        location_summary = (
            location_df.groupby("local_date", dropna=False, sort=False)[["latitude", "longitude"]]
            .mean()
            .rename(columns={"latitude": "mean_latitude", "longitude": "mean_longitude"})
            .reset_index()
        )
        if not pd.isna(fallback_lat):
            location_summary["mean_latitude"] = location_summary["mean_latitude"].fillna(float(fallback_lat))
        if not pd.isna(fallback_lon):
            location_summary["mean_longitude"] = location_summary["mean_longitude"].fillna(float(fallback_lon))

    solar_df = location_summary.copy()
    if solar_df.empty:
        solar_df = pd.DataFrame({"local_date": work["local_date"].drop_duplicates().tolist()})
        solar_df["mean_latitude"] = np.nan
        solar_df["mean_longitude"] = np.nan
    solar_df["sunrise_local_hour"] = solar_df.apply(
        lambda row: sunrise_sunset_local_hours(
            row["local_date"],
            row.get("mean_latitude", np.nan),
            row.get("mean_longitude", np.nan),
            tz_name,
        )[0],
        axis=1,
    )
    solar_df["sunset_local_hour"] = solar_df.apply(
        lambda row: sunrise_sunset_local_hours(
            row["local_date"],
            row.get("mean_latitude", np.nan),
            row.get("mean_longitude", np.nan),
            tz_name,
        )[1],
        axis=1,
    )

    daily = (
        work.groupby(["local_date", "state_label_raw", "state_label_canonical"], dropna=False, sort=False)["duration_h"]
        .sum()
        .reset_index()
        .merge(solar_df, on="local_date", how="left")
    )
    daily["dataset_id"] = str(dataset_id)
    daily["deployment_id"] = str(deployment_id)
    daily["timezone_name"] = str(tz_name)
    daily["method"] = method_name
    daily["method_variant"] = method_variant
    daily["method_label"] = _as_method_label(method_name, method_variant)
    daily["pct_of_24h"] = 100.0 * daily["duration_h"] / 24.0

    hourly = (
        work.groupby(["local_date", "hour_of_day", "state_label_raw", "state_label_canonical"], dropna=False, sort=False)["duration_h"]
        .sum()
        .reset_index()
    )
    hourly_totals = (
        hourly.groupby(["local_date", "hour_of_day"], dropna=False, sort=False)["duration_h"].sum().reset_index(name="hour_total_h")
    )
    hourly = hourly.merge(hourly_totals, on=["local_date", "hour_of_day"], how="left")
    hourly["pct_of_observed_hour"] = 100.0 * hourly["duration_h"] / hourly["hour_total_h"].replace(0, np.nan)
    hourly = hourly.merge(solar_df[["local_date", "sunrise_local_hour", "sunset_local_hour"]], on="local_date", how="left")
    hourly["dataset_id"] = str(dataset_id)
    hourly["deployment_id"] = str(deployment_id)
    hourly["timezone_name"] = str(tz_name)
    hourly["method"] = method_name
    hourly["method_variant"] = method_variant
    hourly["method_label"] = _as_method_label(method_name, method_variant)

    metadata = pd.DataFrame(
        {
            "dataset_id": [str(dataset_id)],
            "deployment_id": [str(deployment_id)],
            "method": [method_name],
            "method_variant": [method_variant],
            "method_label": [_as_method_label(method_name, method_variant)],
            "timezone_name": [str(tz_name)],
        }
    )
    return daily, hourly, metadata


def _build_method_group_comparison(
    daily_df: pd.DataFrame,
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    group_lookup = {}
    for group_name, deployments in (groups or {}).items():
        for deployment_id in deployments:
            group_lookup[str(deployment_id)] = str(group_name)
    work = daily_df.copy()
    work["group_name"] = work["deployment_id"].astype(str).map(group_lookup).fillna("Unknown")
    dep_rollup = (
        work.groupby(
            ["method", "method_variant", "method_label", "dataset_id", "deployment_id", "group_name", "state_label_canonical"],
            dropna=False,
            as_index=False,
            sort=False,
        )[["duration_h", "pct_of_24h"]]
        .mean()
    )
    dep_rollup["comparison_level"] = "deployment"
    dep_rollup["comparison_key"] = dep_rollup["deployment_id"].astype(str)

    dataset_rollup = (
        dep_rollup.groupby(
            ["method", "method_variant", "method_label", "dataset_id", "state_label_canonical"],
            dropna=False,
            as_index=False,
            sort=False,
        )[["duration_h", "pct_of_24h"]]
        .mean()
    )
    dataset_rollup["deployment_id"] = pd.NA
    dataset_rollup["group_name"] = pd.NA
    dataset_rollup["comparison_level"] = "dataset"
    dataset_rollup["comparison_key"] = dataset_rollup["dataset_id"].astype(str)

    group_rollup = (
        dep_rollup.groupby(
            ["method", "method_variant", "method_label", "group_name", "state_label_canonical"],
            dropna=False,
            as_index=False,
            sort=False,
        )[["duration_h", "pct_of_24h"]]
        .mean()
    )
    group_rollup["dataset_id"] = pd.NA
    group_rollup["deployment_id"] = pd.NA
    group_rollup["comparison_level"] = "group"
    group_rollup["comparison_key"] = group_rollup["group_name"].astype(str)
    out = pd.concat([dep_rollup, dataset_rollup, group_rollup], ignore_index=True, sort=False)
    return out


def _coerce_parquet_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize fragile dtypes (mainly categoricals) before parquet writes."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            out[col] = series.astype("string")
    return out


def _write_parquet_resilient(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write parquet and retry once with dtype coercion if pandas/pyarrow rejects categories."""
    try:
        df.to_parquet(path, index=False)
        return df
    except TypeError:
        coerced = _coerce_parquet_friendly(df)
        coerced.to_parquet(path, index=False)
        return coerced


def write_segmentation_run_summary_parquets(
    run_output_root: str,
    algorithmic_seg_df: pd.DataFrame,
    supervised_prediction_df: pd.DataFrame | None,
    load_data_pkl_for_deployment: Callable[[str, str], Any],
    algorithmic_exhaustive_seg_df: pd.DataFrame | None = None,
    *,
    dive_depth_threshold_m: float = 2.0,
    rest_key_map: dict[str, str] | None = None,
    canonical_behavior_map: dict[str, str] | None = None,
    groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    algorithmic_seg_df = _coerce_parquet_friendly(
        algorithmic_seg_df if isinstance(algorithmic_seg_df, pd.DataFrame) else pd.DataFrame()
    )
    algorithmic_exhaustive_seg_df = _coerce_parquet_friendly(
        algorithmic_exhaustive_seg_df if isinstance(algorithmic_exhaustive_seg_df, pd.DataFrame) else pd.DataFrame()
    )
    supervised_prediction_df = _coerce_parquet_friendly(
        supervised_prediction_df if isinstance(supervised_prediction_df, pd.DataFrame) else pd.DataFrame()
    )
    rest_key_map = dict(DEFAULT_REST_KEY_MAP | dict(rest_key_map or {}))
    canonical_map = dict(CANONICAL_BEHAVIOR_MAP | dict(canonical_behavior_map or {}))
    summary_dir = Path(run_output_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    segmentation_summary = _build_segmentation_summary(
        seg_df=algorithmic_seg_df,
        spdf=supervised_prediction_df,
        rest_key_map=rest_key_map,
    )
    putative_rest_summary = segmentation_summary.loc[segmentation_summary.get("positive_label", pd.Series(False, index=segmentation_summary.index)).fillna(False).astype(bool)].copy()

    enriched_frames = []
    daily_frames = []
    if {"dataset_id", "deployment_id"}.issubset(putative_rest_summary.columns):
        for (dataset_id, deployment_id), dep in putative_rest_summary.groupby(["dataset_id", "deployment_id"], sort=False):
            data_pkl = load_data_pkl_for_deployment(str(dataset_id), str(deployment_id))
            deployment_info = getattr(data_pkl, "deployment_info", {}) or {}
            tz_name = str(deployment_info.get("Time Zone") or "UTC").strip() or "UTC"
            dep = dep.copy()
            dep["start_datetime"] = _normalize_dt(dep["start_datetime"], tz_name)
            dep["end_datetime"] = _normalize_dt(dep["end_datetime"], tz_name)
            dep["segment_midpoint_datetime"] = _normalize_dt(dep["segment_midpoint_datetime"], tz_name)
            dep["local_date"] = dep["segment_midpoint_datetime"].dt.date
            dep["local_hour"] = (
                dep["segment_midpoint_datetime"].dt.hour
                + dep["segment_midpoint_datetime"].dt.minute / 60.0
                + dep["segment_midpoint_datetime"].dt.second / 3600.0
            )
            location_df = _pick_location_frame(data_pkl, tz_name)
            location_match = _nearest_location_at_times(location_df, dep["segment_midpoint_datetime"])
            dep["latitude"] = location_match["latitude"].values
            dep["longitude"] = location_match["longitude"].values
            feature_df = _extract_algorithmic_feature_snapshot(data_pkl, dep["segment_midpoint_datetime"], tz_name)
            dep = pd.concat([dep, feature_df], axis=1)
            enriched_frames.append(dep)

            for method_name, method_dep in dep.groupby("method", sort=False):
                daily = _build_daily_activity_rows(
                    method_rows=method_dep,
                    data_pkl=data_pkl,
                    deployment_id=str(deployment_id),
                    tz_name=tz_name,
                    dive_depth_threshold_m=float(dive_depth_threshold_m),
                )
                if not daily.empty:
                    daily["dataset_id"] = str(dataset_id)
                    daily["method"] = str(method_name)
                    daily_frames.append(daily)

    putative_rest_summary = pd.concat(enriched_frames, ignore_index=True, sort=False) if enriched_frames else putative_rest_summary
    putative_rest_summary = _annotate_overlap_flags(putative_rest_summary)
    daily_activity_summary = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()

    method_daily_frames = []
    method_hourly_frames = []
    method_metadata_frames = []
    if {"dataset_id", "deployment_id", "method", "method_variant"}.issubset(segmentation_summary.columns):
        for (dataset_id, deployment_id), dep in segmentation_summary.groupby(["dataset_id", "deployment_id"], sort=False):
            data_pkl = load_data_pkl_for_deployment(str(dataset_id), str(deployment_id))
            deployment_info = getattr(data_pkl, "deployment_info", {}) or {}
            tz_name = str(deployment_info.get("Time Zone") or "UTC").strip() or "UTC"
            dep = dep.copy()
            dep["start_datetime"] = _normalize_dt(dep["start_datetime"], tz_name)
            dep["end_datetime"] = _normalize_dt(dep["end_datetime"], tz_name)
            dep["segment_midpoint_datetime"] = _normalize_dt(dep["segment_midpoint_datetime"], tz_name)
            exhaustive_dep = pd.DataFrame()
            if (
                isinstance(algorithmic_exhaustive_seg_df, pd.DataFrame)
                and not algorithmic_exhaustive_seg_df.empty
                and {"dataset_id", "deployment_id"}.issubset(algorithmic_exhaustive_seg_df.columns)
            ):
                exhaustive_dep = algorithmic_exhaustive_seg_df[
                    (algorithmic_exhaustive_seg_df["dataset_id"].astype(str) == str(dataset_id))
                    & (algorithmic_exhaustive_seg_df["deployment_id"].astype(str) == str(deployment_id))
                ].copy()
                exhaustive_dep = _algorithmic_exhaustive_method_rows(exhaustive_dep, rest_key_map)
                if not exhaustive_dep.empty:
                    exhaustive_dep["start_datetime"] = _normalize_dt(exhaustive_dep["start_datetime"], tz_name)
                    exhaustive_dep["end_datetime"] = _normalize_dt(exhaustive_dep["end_datetime"], tz_name)
                    exhaustive_dep["segment_midpoint_datetime"] = _normalize_dt(exhaustive_dep["segment_midpoint_datetime"], tz_name)
            for (_, _), method_dep in dep.groupby(["method", "method_variant"], sort=False):
                method_name = _safe_text(method_dep.get("method", pd.Series([""])).iloc[0])
                method_variant = _safe_text(method_dep.get("method_variant", pd.Series([""])).iloc[0])
                method_rows = method_dep
                if method_name == "algorithmic" and method_variant == "algorithmic_segments" and not exhaustive_dep.empty:
                    method_rows = exhaustive_dep
                daily_rows, hourly_rows, metadata_rows = _build_method_budget_rows(
                    method_rows=method_rows,
                    data_pkl=data_pkl,
                    dataset_id=str(dataset_id),
                    deployment_id=str(deployment_id),
                    tz_name=tz_name,
                    dive_depth_threshold_m=float(dive_depth_threshold_m),
                    canonical_map=canonical_map,
                )
                if not daily_rows.empty:
                    method_daily_frames.append(daily_rows)
                if not hourly_rows.empty:
                    method_hourly_frames.append(hourly_rows)
                if not metadata_rows.empty:
                    method_metadata_frames.append(metadata_rows)
    method_budget_daily = pd.concat(method_daily_frames, ignore_index=True, sort=False) if method_daily_frames else pd.DataFrame()
    method_budget_hourly = pd.concat(method_hourly_frames, ignore_index=True, sort=False) if method_hourly_frames else pd.DataFrame()
    method_budget_metadata = pd.concat(method_metadata_frames, ignore_index=True, sort=False) if method_metadata_frames else pd.DataFrame()
    method_budget_group_comparison = _build_method_group_comparison(method_budget_daily, groups or {})

    if not method_budget_daily.empty:
        mapping_pairs = (
            method_budget_daily[["state_label_raw", "state_label_canonical"]]
            .drop_duplicates()
            .rename(columns={"state_label_raw": "raw_label", "state_label_canonical": "canonical_label"})
            .reset_index(drop=True)
        )
        mapping_pairs["raw_label"] = mapping_pairs["raw_label"].astype(str)
        mapping_pairs["canonical_label"] = mapping_pairs["canonical_label"].astype(str)
        mapping_pairs = _sort_values_resilient(mapping_pairs, ["raw_label", "canonical_label"], kind="stable").reset_index(drop=True)
        mapping_pairs["is_unmapped"] = mapping_pairs["canonical_label"].astype(str).str.startswith("Unmapped:")
        method_budget_metadata = pd.concat([method_budget_metadata, mapping_pairs], ignore_index=True, sort=False)

    overlap_summary = pd.DataFrame()
    if not putative_rest_summary.empty:
        overlap_summary = (
            putative_rest_summary.groupby(["dataset_id", "deployment_id", "method", "weak_confusion"], dropna=False, sort=False)
            .size()
            .reset_index(name="event_count")
        )

    segmentation_summary_path = summary_dir / "segmentation_summary.parquet"
    putative_rest_summary_path = summary_dir / "putative_rest_summary.parquet"
    daily_activity_summary_path = summary_dir / "daily_activity_summary.parquet"
    overlap_summary_path = summary_dir / "putative_rest_overlap_summary.parquet"
    method_budget_daily_path = summary_dir / "method_budget_daily.parquet"
    method_budget_hourly_path = summary_dir / "method_budget_hourly.parquet"
    method_budget_group_comparison_path = summary_dir / "method_budget_group_comparison.parquet"
    method_budget_metadata_path = summary_dir / "method_budget_metadata.parquet"
    segmentation_summary = _write_parquet_resilient(segmentation_summary, segmentation_summary_path)
    putative_rest_summary = _write_parquet_resilient(putative_rest_summary, putative_rest_summary_path)
    daily_activity_summary = _write_parquet_resilient(daily_activity_summary, daily_activity_summary_path)
    overlap_summary = _write_parquet_resilient(overlap_summary, overlap_summary_path)
    method_budget_daily = _write_parquet_resilient(method_budget_daily, method_budget_daily_path)
    method_budget_hourly = _write_parquet_resilient(method_budget_hourly, method_budget_hourly_path)
    method_budget_group_comparison = _write_parquet_resilient(
        method_budget_group_comparison, method_budget_group_comparison_path
    )
    method_budget_metadata = _write_parquet_resilient(method_budget_metadata, method_budget_metadata_path)
    return {
        "segmentation_summary_path": str(segmentation_summary_path),
        "putative_rest_summary_path": str(putative_rest_summary_path),
        "daily_activity_summary_path": str(daily_activity_summary_path),
        "putative_rest_overlap_summary_path": str(overlap_summary_path),
        "method_budget_daily_path": str(method_budget_daily_path),
        "method_budget_hourly_path": str(method_budget_hourly_path),
        "method_budget_group_comparison_path": str(method_budget_group_comparison_path),
        "method_budget_metadata_path": str(method_budget_metadata_path),
        "segmentation_summary": segmentation_summary,
        "putative_rest_summary": putative_rest_summary,
        "daily_activity_summary": daily_activity_summary,
        "putative_rest_overlap_summary": overlap_summary,
        "method_budget_daily": method_budget_daily,
        "method_budget_hourly": method_budget_hourly,
        "method_budget_group_comparison": method_budget_group_comparison,
        "method_budget_metadata": method_budget_metadata,
    }
