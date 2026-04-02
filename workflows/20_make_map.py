import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import yaml

from pyologger.plot_data.environmental_covariates import (
    detect_lat_lon_columns,
    normalize_covariate_request,
    normalize_track_extent,
    plot_continuous_track_map,
    plot_discrete_track_map,
    load_covariate_overlay,
    validate_covariate_runtime_readiness,
)
from pyologger.utils.folder_manager import load_combined_config


DEFAULT_ALGORITHMIC_REST_LABELS = {
    "find_rest.surface_sleep",
    "find_rest.long_flat",
    "find_rest.long_drift",
}


@dataclass
class MapRunContext:
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


def _safe_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def _normalize_pandas_freq(freq: Optional[str]) -> Optional[str]:
    if freq is None:
        return None
    text = str(freq).strip()
    if not text:
        return None
    return text.replace("H", "h")


def _resolve_run_context(config_path: str, run_name: str) -> MapRunContext:
    cfg = _load_yaml(config_path)
    run_cfg = dict(((cfg.get("make_map_runs") or {}).get(run_name)) or {})
    if not run_cfg:
        raise ValueError(f"make_map_runs.{run_name} not found in {config_path}")

    data_root = cfg["paths"]["local_private_data"]
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

    analysis_id = str(run_cfg.get("analysis_id") or _safe_slug(run_name))
    override = run_cfg.get("output_root_override")
    if override:
        output_root = os.path.expanduser(os.path.expandvars(str(override)))
    elif len(dataset_ids) == 1:
        output_root = os.path.join(data_root, dataset_ids[0], "00_Meta-Analysis", "maps", analysis_id)
    else:
        output_root = os.path.join(data_root, "00_Meta-Analysis", "maps", analysis_id)
    _ensure_dir(output_root)

    return MapRunContext(
        run_name=run_name,
        run_cfg=run_cfg,
        global_cfg=cfg,
        data_root=data_root,
        output_root=output_root,
        analysis_id=analysis_id,
        scope=scope,
    )


def _data_pkl_path(ctx: MapRunContext, dataset_id: str, deployment_id: str) -> str:
    return os.path.join(ctx.data_root, dataset_id, deployment_id, "outputs", "data.pkl")


def _load_data_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalize_map_specs(ctx: MapRunContext) -> List[Dict]:
    map_specs = list(ctx.run_cfg.get("maps") or [])
    if not map_specs:
        raise ValueError(f"Run '{ctx.run_name}' must define at least one map under maps")

    normalized = []
    for idx, map_cfg in enumerate(map_specs):
        if not isinstance(map_cfg, dict):
            raise ValueError(f"Map index {idx} is not a dict")
        map_id = str(map_cfg.get("map_id") or f"map_{idx + 1}")
        color_mode = str(map_cfg.get("color_mode") or "").strip().lower()
        if color_mode not in {"discrete", "continuous"}:
            raise ValueError(f"Map '{map_id}' has invalid color_mode '{color_mode}'")

        normalized_map = dict(map_cfg)
        normalized_map["map_id"] = map_id
        normalized_map["color_mode"] = color_mode
        normalized_map.setdefault("global_extent", False)
        normalized_map.setdefault("pad_deg", ((ctx.run_cfg.get("bathy") or {}).get("pad_deg", 20.0)))
        normalized_map.setdefault("max_pixels", ((ctx.run_cfg.get("bathy") or {}).get("max_pixels", 1200)))
        normalized_map.setdefault("output_dir", ctx.output_root)
        normalized_map.setdefault("agg", "mean")
        normalized_map.setdefault("time_bin", None)
        normalized_map.setdefault("color_scale", None)
        normalized_map.setdefault("point_size_by", None)
        normalized_map.setdefault("point_size_range", [1.5, 7.5])
        normalized_map.setdefault("point_alpha", 0.85)

        covariates = [normalize_covariate_request(ctx.global_cfg, map_id, x) for x in (map_cfg.get("environmental_covariates") or [])]
        if len(covariates) > 1:
            raise ValueError(f"Map '{map_id}' defines multiple environmental covariates; v1 supports at most one rendered base layer per map")
        normalized_map["environmental_covariates"] = covariates

        if color_mode == "continuous":
            derived_metric = str(map_cfg.get("derived_metric") or "").strip().lower() or None
            normalized_map["derived_metric"] = derived_metric
            if derived_metric == "algorithmic_rest_daily_hours":
                if not normalized_map.get("time_bin"):
                    normalized_map["time_bin"] = "1D"
                normalized_map.setdefault("segmentation_analysis_id", str(ctx.run_cfg.get("segmentation_analysis_id") or "").strip() or None)
                normalized_map.setdefault("rest_labels", sorted(DEFAULT_ALGORITHMIC_REST_LABELS))
                normalized_map.setdefault("value_channel", "algorithmic_rest_h_per_day")
                normalized_map.setdefault("point_size_by", "map_value")
            else:
                value_channel = str(map_cfg.get("value_channel") or "").strip()
                if "." not in value_channel:
                    raise ValueError(f"Map '{map_id}' requires value_channel in 'signal.channel' format")
                if not normalized_map.get("time_bin"):
                    normalized_map["time_bin"] = "1H"
                normalized_map["value_channel"] = value_channel
        normalized.append(normalized_map)
    return normalized


def _segments_root_for_analysis(data_root: str, segmentation_analysis_id: str) -> str:
    return os.path.join(data_root, "00_Meta-Analysis", "segmentation", segmentation_analysis_id, "segments", "by_deployment")


def _algorithmic_segments_parquet_path(data_root: str, segmentation_analysis_id: str, dataset_id: str, deployment_id: str) -> str:
    return os.path.join(
        _segments_root_for_analysis(data_root, segmentation_analysis_id),
        f"{dataset_id}__{deployment_id}__algorithmic_segments.parquet",
    )


def _allocate_interval_seconds_to_days(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[tuple[pd.Timestamp, float]]:
    allocations: List[tuple[pd.Timestamp, float]] = []
    if pd.isna(start_ts) or pd.isna(end_ts):
        return allocations
    if end_ts <= start_ts:
        return allocations
    cursor = start_ts
    while cursor < end_ts:
        next_midnight = cursor.normalize() + pd.Timedelta(days=1)
        chunk_end = end_ts if end_ts <= next_midnight else next_midnight
        secs = float((chunk_end - cursor).total_seconds())
        if secs > 0:
            allocations.append((cursor.normalize(), secs))
        cursor = chunk_end
    return allocations


def _build_algorithmic_daily_rest_dataframe(
    ctx: MapRunContext,
    loaded: Dict[str, Dict],
    location_df: pd.DataFrame,
    segmentation_analysis_id: str,
    rest_labels: List[str],
) -> pd.DataFrame:
    if not segmentation_analysis_id:
        raise ValueError("algorithmic_rest_daily_hours map requires segmentation_analysis_id")
    rest_label_set = {str(x).strip() for x in (rest_labels or []) if str(x).strip()}
    if not rest_label_set:
        rest_label_set = set(DEFAULT_ALGORITHMIC_REST_LABELS)

    frames = []
    for payload in loaded.values():
        dataset_id = payload["dataset_id"]
        deployment_id = payload["deployment_id"]
        seg_path = _algorithmic_segments_parquet_path(ctx.data_root, segmentation_analysis_id, dataset_id, deployment_id)
        if not os.path.exists(seg_path):
            print(f"[make_map] warning: skipping {deployment_id} because algorithmic segments are missing at {seg_path}")
            continue

        seg_df = pd.read_parquet(seg_path, columns=["start_datetime", "end_datetime", "label_name"])
        if seg_df.empty:
            continue
        seg_df = seg_df[seg_df["label_name"].astype(str).isin(rest_label_set)].copy()
        if seg_df.empty:
            print(f"[make_map] warning: no matching rest labels for {deployment_id} in {os.path.basename(seg_path)}")
            continue
        seg_df["start_datetime"] = pd.to_datetime(seg_df["start_datetime"], errors="coerce")
        seg_df["end_datetime"] = pd.to_datetime(seg_df["end_datetime"], errors="coerce")
        seg_df = seg_df.dropna(subset=["start_datetime", "end_datetime"])
        if seg_df.empty:
            continue

        day_rows = []
        for rec in seg_df.itertuples(index=False):
            day_rows.extend(_allocate_interval_seconds_to_days(rec.start_datetime, rec.end_datetime))
        if not day_rows:
            continue
        daily_rest = pd.DataFrame(day_rows, columns=["day", "rest_seconds"])
        daily_rest = daily_rest.groupby("day", as_index=False).agg(rest_seconds=("rest_seconds", "sum"))
        daily_rest["map_value"] = daily_rest["rest_seconds"] / 3600.0

        dep_loc = location_df[
            (location_df["dataset_id"] == dataset_id) & (location_df["deployment_id"] == deployment_id)
        ].copy()
        if dep_loc.empty:
            continue
        dep_loc["day"] = pd.to_datetime(dep_loc["datetime"], errors="coerce").dt.floor("D")
        daily_loc = dep_loc.groupby("day", as_index=False).agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            dataset_id=("dataset_id", "first"),
            deployment_id=("deployment_id", "first"),
            animal_id=("animal_id", "first"),
        )
        merged = daily_loc.merge(daily_rest[["day", "map_value"]], on="day", how="inner")
        if merged.empty:
            continue
        merged = merged.rename(columns={"day": "datetime"})
        merged["map_value"] = pd.to_numeric(merged["map_value"], errors="coerce")
        merged = merged.dropna(subset=["map_value"])
        merged["point_size"] = merged["map_value"]
        frames.append(merged[["datetime", "lat", "lon", "dataset_id", "deployment_id", "animal_id", "map_value", "point_size"]])

    if not frames:
        raise ValueError("No algorithmic daily rest map data could be built from segmentation outputs")
    return pd.concat(frames, ignore_index=True)


def _collect_loaded_deployments(ctx: MapRunContext) -> Dict[str, Dict]:
    loaded = {}
    for item in ctx.scope:
        dataset_id = item["dataset_id"]
        deployment_id = item["deployment_id"]
        path = _data_pkl_path(ctx, dataset_id, deployment_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data.pkl for {dataset_id}/{deployment_id}: {path}")
        data_pkl = _load_data_pkl(path)
        loaded[f"{dataset_id}::{deployment_id}"] = {
            "dataset_id": dataset_id,
            "deployment_id": deployment_id,
            "animal_id": deployment_id.split("_", 1)[1] if "_" in deployment_id else deployment_id,
            "data_pkl": data_pkl,
        }
    return loaded


def _downsample_location_df(df: pd.DataFrame, target_interval: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime", "lat", "lon"]).sort_values("datetime").reset_index(drop=True)
    freq = _normalize_pandas_freq(target_interval)
    if out.empty or not freq:
        return out
    out["__bin"] = out["datetime"].dt.floor(freq)
    out = (
        out.groupby("__bin", as_index=False)
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            dataset_id=("dataset_id", "first"),
            deployment_id=("deployment_id", "first"),
            animal_id=("animal_id", "first"),
        )
        .rename(columns={"__bin": "datetime"})
    )
    return out[["datetime", "lat", "lon", "dataset_id", "deployment_id", "animal_id"]]


def _collect_location_data(loaded: Dict[str, Dict], target_interval: Optional[str]) -> pd.DataFrame:
    frames = []
    for payload in loaded.values():
        data_pkl = payload["data_pkl"]
        loc_df = ((getattr(data_pkl, "signal_data", {}) or {}).get("location"))
        if loc_df is None or len(loc_df) == 0:
            print(f"[make_map] warning: skipping {payload['deployment_id']} because location signal is missing")
            continue
        loc_df = loc_df.copy()
        lat_col, lon_col = detect_lat_lon_columns(loc_df)
        loc_df["datetime"] = pd.to_datetime(loc_df["datetime"], errors="coerce")
        loc_df["lat"] = pd.to_numeric(loc_df[lat_col], errors="coerce")
        loc_df["lon"] = pd.to_numeric(loc_df[lon_col], errors="coerce")
        loc_df["dataset_id"] = payload["dataset_id"]
        loc_df["deployment_id"] = payload["deployment_id"]
        loc_df["animal_id"] = payload["animal_id"]
        loc_df = _downsample_location_df(loc_df[["datetime", "lat", "lon", "dataset_id", "deployment_id", "animal_id"]], target_interval)
        if not loc_df.empty:
            frames.append(loc_df)
    if not frames:
        raise ValueError("No usable location data found for the requested map run")
    return pd.concat(frames, ignore_index=True)


def _extract_signal_value_series(data_pkl, signal_name: str, channel_name: str) -> pd.DataFrame:
    signal_df = ((getattr(data_pkl, "signal_data", {}) or {}).get(signal_name))
    if signal_df is None or len(signal_df) == 0:
        return pd.DataFrame(columns=["datetime", "map_value"])
    if channel_name not in signal_df.columns:
        return pd.DataFrame(columns=["datetime", "map_value"])
    out = signal_df[["datetime", channel_name]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["map_value"] = pd.to_numeric(out[channel_name], errors="coerce")
    return out.drop(columns=[channel_name]).dropna(subset=["datetime", "map_value"]).sort_values("datetime")


def _build_continuous_map_dataframe(
    loaded: Dict[str, Dict],
    location_df: pd.DataFrame,
    value_channel: str,
    time_bin: str,
    agg_name: str,
) -> pd.DataFrame:
    signal_name, channel_name = value_channel.split(".", 1)
    freq = _normalize_pandas_freq(time_bin)
    frames = []
    for payload in loaded.values():
        dep_id = payload["deployment_id"]
        dep_loc = location_df[location_df["deployment_id"] == dep_id].copy()
        if dep_loc.empty:
            continue
        dep_loc["datetime"] = pd.to_datetime(dep_loc["datetime"], errors="coerce")
        dep_loc["__bin"] = dep_loc["datetime"].dt.floor(freq)
        dep_loc = dep_loc.groupby("__bin", as_index=False).agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            dataset_id=("dataset_id", "first"),
            deployment_id=("deployment_id", "first"),
            animal_id=("animal_id", "first"),
        ).rename(columns={"__bin": "datetime"})

        values = _extract_signal_value_series(payload["data_pkl"], signal_name, channel_name)
        if values.empty:
            print(f"[make_map] warning: skipping {dep_id} for continuous map because {value_channel} is missing")
            continue
        values["__bin"] = values["datetime"].dt.floor(freq)
        values = values.groupby("__bin", as_index=False).agg(map_value=("map_value", agg_name)).rename(columns={"__bin": "datetime"})

        merged = dep_loc.merge(values, on="datetime", how="inner")
        if not merged.empty:
            frames.append(merged)
    if not frames:
        raise ValueError(f"No continuous map data could be built for {value_channel}")
    return pd.concat(frames, ignore_index=True)


def _location_downsample_interval(run_cfg: Dict) -> Optional[str]:
    downsample_cfg = dict(run_cfg.get("location_downsample") or {})
    if not downsample_cfg:
        return None
    if not bool(downsample_cfg.get("enabled", False)):
        return None
    return downsample_cfg.get("target_interval") or "1H"


def _render_run(ctx: MapRunContext, map_specs: List[Dict]) -> Dict:
    loaded = _collect_loaded_deployments(ctx)
    location_df = _collect_location_data(loaded, _location_downsample_interval(ctx.run_cfg))
    overlay_cache = {}
    results = []

    for map_cfg in map_specs:
        covariates = list(map_cfg.get("environmental_covariates") or [])
        extent = normalize_track_extent(
            location_df,
            pad_deg=float(map_cfg.get("pad_deg", 20.0) or 20.0),
            global_extent=bool(map_cfg.get("global_extent", False)),
        )
        overlay = None
        if covariates:
            validate_covariate_runtime_readiness(map_cfg["map_id"], covariates[0])
            cov_request = dict(covariates[0])
            cov_request.setdefault("max_pixels", map_cfg.get("max_pixels", 1200))
            overlay = load_covariate_overlay(cov_request, extent, overlay_cache)

        if map_cfg["color_mode"] == "discrete":
            plot_discrete_track_map(
                location_df=location_df,
                output_dir=map_cfg["output_dir"],
                output_filename=map_cfg["map_id"],
                extent=extent,
                covariate_overlay=overlay,
            )
        else:
            if str(map_cfg.get("derived_metric") or "").strip().lower() == "algorithmic_rest_daily_hours":
                continuous_df = _build_algorithmic_daily_rest_dataframe(
                    ctx=ctx,
                    loaded=loaded,
                    location_df=location_df,
                    segmentation_analysis_id=str(map_cfg.get("segmentation_analysis_id") or ""),
                    rest_labels=list(map_cfg.get("rest_labels") or sorted(DEFAULT_ALGORITHMIC_REST_LABELS)),
                )
            else:
                continuous_df = _build_continuous_map_dataframe(
                    loaded=loaded,
                    location_df=location_df,
                    value_channel=map_cfg["value_channel"],
                    time_bin=str(map_cfg["time_bin"]),
                    agg_name=str(map_cfg.get("agg") or "mean"),
                )
            plot_continuous_track_map(
                binned_df=continuous_df,
                output_dir=map_cfg["output_dir"],
                output_filename=map_cfg["map_id"],
                value_label=map_cfg["value_channel"],
                covariate_overlay=overlay,
                color_scale=map_cfg.get("color_scale"),
                point_size_col=("point_size" if map_cfg.get("point_size_by") == "map_value" else map_cfg.get("point_size_by")),
                point_size_range=map_cfg.get("point_size_range"),
                point_alpha=float(map_cfg.get("point_alpha", 0.85) or 0.85),
            )

        results.append(
            {
                "map_id": map_cfg["map_id"],
                "color_mode": map_cfg["color_mode"],
                "output_dir": map_cfg["output_dir"],
                "environmental_covariates": [x["type"] for x in covariates],
            }
        )
    return {
        "run_name": ctx.run_name,
        "analysis_id": ctx.analysis_id,
        "output_root": ctx.output_root,
        "rendered_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "maps": results,
    }


def cmd_resolve(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    map_specs = _normalize_map_specs(ctx)
    for map_cfg in map_specs:
        covariates = list(map_cfg.get("environmental_covariates") or [])
        if covariates:
            validate_covariate_runtime_readiness(map_cfg["map_id"], covariates[0])
    manifest = {
        "run_name": ctx.run_name,
        "analysis_id": ctx.analysis_id,
        "output_root": ctx.output_root,
        "config_path": os.path.abspath(args.config),
        "resolved_scope": ctx.scope,
        "run_config": ctx.run_cfg,
        "maps": map_specs,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[make_map:resolve] wrote {args.output}")


def cmd_render(args):
    ctx = _resolve_run_context(args.config, args.run_name)
    map_specs = _normalize_map_specs(ctx)
    summary = _render_run(ctx, map_specs)
    _ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved maps to {ctx.output_root}")
    print(f"[make_map:render] wrote {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description="Resolve and render config-driven map runs.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--run-name", required=True, help="make_map_runs entry name")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_resolve = subparsers.add_parser("resolve", help="Resolve and validate the requested map run")
    p_resolve.add_argument("--output", required=True, help="Output JSON manifest path")
    p_resolve.set_defaults(func=cmd_resolve)

    p_render = subparsers.add_parser("render", help="Render all maps for the requested run")
    p_render.add_argument("--output", required=True, help="Output JSON summary path")
    p_render.set_defaults(func=cmd_render)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
