from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from pyologger.utils.solar_utils import sunrise_sunset_local_hours


DEFAULT_REST_KEY_MAP = {
    "surface_sleep": "surface_sleep",
    "benthic_sleep": "benthic_sleep",
    "long_flat": "benthic_sleep",
    "drift_sleep": "drift_sleep",
    "long_drift": "drift_sleep",
}


def _load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _normalize_datetime(values, tz_name: str) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce")
    if not isinstance(dt, pd.Series):
        dt = pd.Series(dt)
    try:
        if getattr(dt.dt, "tz", None) is None:
            return dt.dt.tz_localize(tz_name)
        return dt.dt.tz_convert(tz_name)
    except Exception:
        return dt


def _deployment_dirs(dataset_folder: str) -> list[Path]:
    root = Path(dataset_folder)
    if not root.exists():
        return []
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("00_")
    )


def _pick_depth_frame(data_pkl, tz_name: str) -> pd.DataFrame:
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    for signal_name in ["depth", "pressure"]:
        df = signal_data.get(signal_name)
        if not isinstance(df, pd.DataFrame) or "datetime" not in df.columns:
            continue
        value_cols = [col for col in df.columns if col != "datetime"]
        if not value_cols:
            continue
        preferred = next((col for col in value_cols if col in {"depth", "pressure"}), value_cols[0])
        out = df[["datetime", preferred]].copy()
        out["datetime"] = _normalize_datetime(out["datetime"], tz_name)
        out["depth_value"] = pd.to_numeric(out[preferred], errors="coerce")
        out = out.dropna(subset=["datetime", "depth_value"]).sort_values("datetime").reset_index(drop=True)
        if not out.empty:
            return out[["datetime", "depth_value"]]
    return pd.DataFrame(columns=["datetime", "depth_value"])


def _pick_location_frame(data_pkl, tz_name: str) -> pd.DataFrame:
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    df = signal_data.get("location")
    if not isinstance(df, pd.DataFrame) or "datetime" not in df.columns:
        return pd.DataFrame(columns=["datetime", "latitude", "longitude"])
    lat_col = next((col for col in df.columns if str(col).lower() in {"lat", "latitude"}), None)
    lon_col = next((col for col in df.columns if str(col).lower() in {"lon", "long", "longitude", "lon360"}), None)
    if not lat_col or not lon_col:
        return pd.DataFrame(columns=["datetime", "latitude", "longitude"])
    out = df[["datetime", lat_col, lon_col]].copy()
    out["datetime"] = _normalize_datetime(out["datetime"], tz_name)
    out["latitude"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["longitude"] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out[["datetime", "latitude", "longitude"]]


def _canonical_rest_subtype(event_key: Any, label_text: Any, rest_key_map: dict[str, str]) -> str | None:
    haystack = " ".join([str(event_key or ""), str(label_text or "")]).strip().lower()
    if not haystack:
        return None
    for token, subtype in rest_key_map.items():
        if str(token).lower() in haystack:
            return str(subtype)
    return None


def _event_end_datetime(row: pd.Series) -> pd.Timestamp | pd.NaT:
    start_dt = pd.to_datetime(row.get("datetime"), errors="coerce")
    if pd.isna(start_dt):
        return pd.NaT
    end_dt = pd.to_datetime(row.get("end_datetime"), errors="coerce")
    if pd.notna(end_dt):
        return pd.Timestamp(end_dt)
    duration_val = pd.to_numeric(pd.Series([row.get("duration")]), errors="coerce").iloc[0]
    if pd.notna(duration_val) and float(duration_val) > 0:
        return pd.Timestamp(start_dt) + pd.to_timedelta(float(duration_val), unit="s")
    return pd.NaT


def _extract_putative_rest_events(data_pkl, tz_name: str, rest_key_map: dict[str, str]) -> pd.DataFrame:
    event_data = getattr(data_pkl, "event_data", None)
    if not isinstance(event_data, pd.DataFrame) or event_data.empty:
        return pd.DataFrame()
    work = event_data.copy()
    if "datetime" not in work.columns:
        if {"date", "time"}.issubset(work.columns):
            work["datetime"] = pd.to_datetime(
                work["date"].astype(str) + " " + work["time"].astype(str),
                errors="coerce",
            )
        else:
            return pd.DataFrame()
    work["datetime"] = _normalize_datetime(work["datetime"], tz_name)
    work["state_subtype"] = work.apply(
        lambda row: _canonical_rest_subtype(
            row.get("key"),
            row.get("short_description"),
            rest_key_map,
        ),
        axis=1,
    )
    work = work.dropna(subset=["datetime"])
    work = work.loc[work["state_subtype"].notna()].copy()
    if work.empty:
        return work
    work["end_datetime"] = work.apply(_event_end_datetime, axis=1)
    work["segment_midpoint_datetime"] = work["datetime"] + (work["end_datetime"] - work["datetime"]) / 2
    work["duration_s"] = pd.to_numeric(work.get("duration"), errors="coerce")
    work["local_date"] = work["segment_midpoint_datetime"].dt.date
    work["local_hour"] = (
        work["segment_midpoint_datetime"].dt.hour
        + work["segment_midpoint_datetime"].dt.minute / 60.0
        + work["segment_midpoint_datetime"].dt.second / 3600.0
    )
    return work.sort_values("datetime").reset_index(drop=True)


def _nearest_location_at_times(location_df: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    if location_df.empty:
        return pd.DataFrame(index=timestamps.index, columns=["latitude", "longitude"])
    query = pd.DataFrame({"lookup_datetime": pd.to_datetime(timestamps, errors="coerce")}, index=timestamps.index)
    matched = pd.merge_asof(
        query.sort_values("lookup_datetime"),
        location_df.rename(columns={"datetime": "lookup_datetime"}).sort_values("lookup_datetime"),
        on="lookup_datetime",
        direction="nearest",
            tolerance=pd.Timedelta("12h"),
    ).set_index(query.sort_values("lookup_datetime").index)
    matched = matched.reindex(timestamps.index)
    return matched[["latitude", "longitude"]]


def _extract_algorithmic_feature_snapshot(data_pkl, timestamps: pd.Series, tz_name: str) -> pd.DataFrame:
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    signal_names = [
        "algorithmic_feature_channels",
        "algorithmic_derivative_channels",
        "algorithmic_intermediate_channels",
    ]
    out = pd.DataFrame(index=timestamps.index)
    for signal_name in signal_names:
        df = signal_data.get(signal_name)
        if not isinstance(df, pd.DataFrame) or "datetime" not in df.columns:
            continue
        work = df.copy()
        work["datetime"] = _normalize_datetime(work["datetime"], tz_name)
        work = work.dropna(subset=["datetime"]).sort_values("datetime")
        value_cols = [col for col in work.columns if col != "datetime"]
        if not value_cols:
            continue
        lookup = pd.DataFrame({"lookup_datetime": pd.to_datetime(timestamps, errors="coerce")}, index=timestamps.index)
        matched = pd.merge_asof(
            lookup.sort_values("lookup_datetime"),
            work.rename(columns={"datetime": "lookup_datetime"}),
            on="lookup_datetime",
            direction="nearest",
            tolerance=pd.Timedelta("10min"),
        ).set_index(lookup.sort_values("lookup_datetime").index)
        matched = matched.reindex(timestamps.index)
        for col in value_cols:
            out[f"{signal_name}__{col}"] = matched[col]
    return out


def _daily_location_summary(location_df: pd.DataFrame, fallback_lat: float | None, fallback_lon: float | None) -> pd.DataFrame:
    if location_df.empty:
        return pd.DataFrame(columns=["local_date", "mean_latitude", "mean_longitude"])
    work = location_df.copy()
    work["local_date"] = work["datetime"].dt.date
    summary = (
        work.groupby("local_date", dropna=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "mean_latitude", "longitude": "mean_longitude"})
        .reset_index()
    )
    if fallback_lat is not None:
        summary["mean_latitude"] = summary["mean_latitude"].fillna(float(fallback_lat))
    if fallback_lon is not None:
        summary["mean_longitude"] = summary["mean_longitude"].fillna(float(fallback_lon))
    return summary


def _build_daily_activity_summary(
    data_pkl,
    deployment_id: str,
    depth_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    location_df: pd.DataFrame,
    tz_name: str,
    dive_depth_threshold_m: float,
) -> pd.DataFrame:
    if depth_df.empty:
        return pd.DataFrame()
    work = depth_df.copy()
    sample_step_s = work["datetime"].diff().dt.total_seconds().median()
    if pd.isna(sample_step_s) or float(sample_step_s) <= 0:
        sample_step_s = 60.0
    work["duration_h"] = work["datetime"].shift(-1).sub(work["datetime"]).dt.total_seconds().fillna(sample_step_s) / 3600.0
    work["state"] = np.where(
        pd.to_numeric(work["depth_value"], errors="coerce").fillna(np.inf) < float(dive_depth_threshold_m),
        "surfacing",
        "diving",
    )
    for _, row in rest_df.iterrows():
        start_dt = row.get("datetime")
        end_dt = row.get("end_datetime")
        subtype = row.get("state_subtype")
        if pd.isna(start_dt) or pd.isna(end_dt) or not subtype:
            continue
        mask = (work["datetime"] >= pd.Timestamp(start_dt)) & (work["datetime"] <= pd.Timestamp(end_dt))
        work.loc[mask, "state"] = str(subtype)
    work["local_date"] = work["datetime"].dt.date
    state_summary = (
        work.groupby(["local_date", "state"], dropna=False)["duration_h"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for state_name in ["drift_sleep", "benthic_sleep", "surface_sleep", "surfacing", "diving"]:
        if state_name not in state_summary.columns:
            state_summary[state_name] = 0.0

    deployment_info = getattr(data_pkl, "deployment_info", {}) or {}
    fallback_lat = pd.to_numeric(pd.Series([deployment_info.get("Deployment Latitude")]), errors="coerce").iloc[0]
    fallback_lon = pd.to_numeric(pd.Series([deployment_info.get("Deployment Longitude")]), errors="coerce").iloc[0]
    location_summary = _daily_location_summary(
        location_df=location_df,
        fallback_lat=(None if pd.isna(fallback_lat) else float(fallback_lat)),
        fallback_lon=(None if pd.isna(fallback_lon) else float(fallback_lon)),
    )
    out = state_summary.merge(location_summary, on="local_date", how="left")
    out["deployment_id"] = str(deployment_id)
    out["timezone_name"] = str(tz_name)
    sunrise_vals = []
    sunset_vals = []
    for row in out.itertuples(index=False):
        sunrise_h, sunset_h = sunrise_sunset_local_hours(
            getattr(row, "local_date"),
            getattr(row, "mean_latitude", np.nan),
            getattr(row, "mean_longitude", np.nan),
            tz_name,
        )
        sunrise_vals.append(sunrise_h)
        sunset_vals.append(sunset_h)
    out["sunrise_local_hour"] = sunrise_vals
    out["sunset_local_hour"] = sunset_vals
    return out[
        [
            "deployment_id",
            "local_date",
            "timezone_name",
            "mean_latitude",
            "mean_longitude",
            "sunrise_local_hour",
            "sunset_local_hour",
            "diving",
            "surfacing",
            "benthic_sleep",
            "surface_sleep",
            "drift_sleep",
        ]
    ].sort_values(["deployment_id", "local_date"]).reset_index(drop=True)


def _build_putative_rest_summary(
    data_pkl,
    deployment_id: str,
    rest_df: pd.DataFrame,
    location_df: pd.DataFrame,
    tz_name: str,
) -> pd.DataFrame:
    if rest_df.empty:
        return pd.DataFrame()
    out = rest_df.copy()
    out["deployment_id"] = str(deployment_id)
    location_match = _nearest_location_at_times(location_df, out["segment_midpoint_datetime"])
    out["latitude"] = location_match["latitude"].values
    out["longitude"] = location_match["longitude"].values
    feature_df = _extract_algorithmic_feature_snapshot(data_pkl, out["segment_midpoint_datetime"], tz_name)
    out = pd.concat([out, feature_df], axis=1)
    preferred_cols = [
        "deployment_id",
        "datetime",
        "end_datetime",
        "segment_midpoint_datetime",
        "local_date",
        "local_hour",
        "duration_s",
        "state_subtype",
        "latitude",
        "longitude",
        "key",
        "short_description",
        "long_description",
        "value",
    ]
    remaining = [col for col in out.columns if col not in preferred_cols]
    return out[preferred_cols + remaining].sort_values(["deployment_id", "datetime"]).reset_index(drop=True)


def write_dataset_state_summary_parquets(
    dataset_folder: str,
    *,
    summary_dir: str | None = None,
    dive_depth_threshold_m: float = 2.0,
    timezone_overrides: dict[str, str] | None = None,
    rest_key_map: dict[str, str] | None = None,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_folder)
    summary_path = Path(summary_dir) if summary_dir else dataset_path / "00_Summary"
    summary_path.mkdir(parents=True, exist_ok=True)
    timezone_overrides = dict(timezone_overrides or {})
    rest_key_map = dict(DEFAULT_REST_KEY_MAP | dict(rest_key_map or {}))

    daily_frames = []
    rest_frames = []
    for deployment_path in _deployment_dirs(str(dataset_path)):
        pkl_path = deployment_path / "outputs" / "data.pkl"
        if not pkl_path.exists():
            continue
        data_pkl = _load_pickle(str(pkl_path))
        deployment_id = str(deployment_path.name)
        deployment_info = getattr(data_pkl, "deployment_info", {}) or {}
        tz_name = str(timezone_overrides.get(deployment_id) or deployment_info.get("Time Zone") or "UTC")
        depth_df = _pick_depth_frame(data_pkl, tz_name)
        if depth_df.empty:
            continue
        location_df = _pick_location_frame(data_pkl, tz_name)
        rest_df = _extract_putative_rest_events(data_pkl, tz_name, rest_key_map)
        daily_frames.append(
            _build_daily_activity_summary(
                data_pkl=data_pkl,
                deployment_id=deployment_id,
                depth_df=depth_df,
                rest_df=rest_df,
                location_df=location_df,
                tz_name=tz_name,
                dive_depth_threshold_m=float(dive_depth_threshold_m),
            )
        )
        rest_frames.append(
            _build_putative_rest_summary(
                data_pkl=data_pkl,
                deployment_id=deployment_id,
                rest_df=rest_df,
                location_df=location_df,
                tz_name=tz_name,
            )
        )

    daily_df = pd.concat([df for df in daily_frames if isinstance(df, pd.DataFrame) and not df.empty], ignore_index=True) if daily_frames else pd.DataFrame()
    rest_df = pd.concat([df for df in rest_frames if isinstance(df, pd.DataFrame) and not df.empty], ignore_index=True) if rest_frames else pd.DataFrame()

    daily_path = summary_path / "daily_activity_summary.parquet"
    rest_path = summary_path / "putative_rest_summary.parquet"
    daily_df.to_parquet(daily_path, index=False)
    rest_df.to_parquet(rest_path, index=False)
    return {
        "daily_activity_summary": daily_df,
        "putative_rest_summary": rest_df,
        "daily_activity_summary_path": str(daily_path),
        "putative_rest_summary_path": str(rest_path),
    }


def load_dataset_state_summary_parquets(dataset_folder: str, *, summary_dir: str | None = None) -> Dict[str, pd.DataFrame]:
    dataset_path = Path(dataset_folder)
    summary_path = Path(summary_dir) if summary_dir else dataset_path / "00_Summary"
    daily_path = summary_path / "daily_activity_summary.parquet"
    rest_path = summary_path / "putative_rest_summary.parquet"
    return {
        "daily_activity_summary": pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame(),
        "putative_rest_summary": pd.read_parquet(rest_path) if rest_path.exists() else pd.DataFrame(),
    }
