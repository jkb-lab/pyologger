import copy
import json
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from typing import Dict, Tuple

import pandas as pd
import yaml
from pyologger.utils.folder_manager import load_combined_config

DEFAULT_SEGMENTATION_DATASET = "mian-juv-nese_sleep_lml-ano_JKB"
DEFAULT_SEGMENTATION_DEPLOYMENT = "2021-04-17_mian-011"
DEFAULT_SEGMENTATION_WINDOW_HOURS = 10

FIND_REST_DERIVED_CHANNELS = [
    {"id": "depth_std_m", "label": "Standardized depth"},
    {"id": "depth_d1_ms", "label": "Depth first derivative (m/s)"},
    {"id": "depth_d2_ms2", "label": "Depth second derivative (m/s^2)"},
    {"id": "depth_d1_end_ms", "label": "End-of-segment mean depth derivative (m/s)"},
    {"id": "drift_rate_ms", "label": "Drift rate (m/s)"},
    {"id": "smoothed_drift_rate_ms", "label": "Smoothed drift rate (m/s)"},
    {"id": "segment_duration_s", "label": "Segment duration (s)"},
]

_MEASURED_CONTEXT_FILTER_IDS = {
    "low_stroke_rate_required",
    "stroke_rate_required",
    "gliding_must_be_true",
}


def event_group_parent(event_key: str) -> str:
    key = str(event_key or "").strip()
    if not key:
        return ""
    cluster_match = re.match(r"^\d+_k(\d+)_", key, flags=re.IGNORECASE)
    if cluster_match:
        return f"k{cluster_match.group(1)}"
    key_lower = key.lower()
    if key_lower.startswith("behavior_"):
        return "behavior"
    if key_lower.startswith("rf_behavior_") or key_lower.startswith("rf_active_") or key_lower.startswith("rf_calm_") or key_lower.startswith("rf_motionless_"):
        return "behavior"
    if key_lower.startswith("active_") or key_lower.startswith("calm_") or key_lower.startswith("motionless_"):
        return "behavior"
    if key_lower.startswith("dive-type_") or key_lower.startswith("dive_type_"):
        return "dive-type"
    if key_lower.startswith("sleep-state_") or key_lower.startswith("sleep_state."):
        return "sleep-state"
    if key_lower.startswith("find_rest"):
        return "find_rest"
    return key.split(".", 1)[0].strip() or key


def group_event_keys(event_keys) -> Dict[str, list[str]]:
    grouped: Dict[str, list[str]] = {}
    for key in [str(v).strip() for v in (event_keys or []) if str(v or "").strip()]:
        parent = event_group_parent(key)
        grouped.setdefault(parent, []).append(key)
    for parent, keys in list(grouped.items()):
        grouped[parent] = sorted(list(dict.fromkeys(keys)), key=lambda x: x.lower())
    return dict(sorted(grouped.items(), key=lambda kv: kv[0].lower()))


def _safe_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def _load_yaml(path: str) -> Dict:
    cfg, _, _ = load_combined_config(config_path=path)
    return cfg or {}


def _write_yaml(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _get_nested_path(payload: Dict, path: str):
    cur = payload
    for part in [str(p) for p in str(path or "").split(".") if str(p).strip()]:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: str, payload) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


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


def _extract_source_key(algo_cfg: Dict) -> str:
    source_key = str(
        algo_cfg.get("source_channel")
        or algo_cfg.get("source_key")
        or "depth.depth"
    )
    if "." not in source_key:
        raise ValueError(f"Invalid source channel '{source_key}', expected signal.channel")
    return source_key


def _default_algorithmic_cfg() -> Dict:
    return {
        "enabled": True,
        "method": "depth_drift_thresholds",
        "method_name": "find_rest",
        "source_channel": "depth.depth",
        "standardize": True,
        "quantize_step": 1.0,
        "base_smooth_seconds": 6.0,
        "coarse_smooth_seconds": 12.0,
        "coarse_interval_threshold_s": 5.0,
        "coarse_resolution_threshold": 1.0,
        "end_segments_upon_d1_sign_change": True,
        "filter_out_positive_slopes": False,
        "filter_out_negative_slopes": True,
        "event_keys": {
            "initial": "find_rest.initial",
            "filtered": "find_rest.filtered",
            "surface_sleep": "find_rest.surface_sleep",
            "long_flat": "find_rest.long_flat",
            "long_drift": "find_rest.long_drift",
        },
        "label_codes": {
            "not_sleep": 0,
            "surface_sleep": 1,
            "long_flat": 2,
            "long_drift": 3,
        },
        "label_names": {
            "not_sleep": "find_rest.not_sleep",
            "surface_sleep": "find_rest.surface_sleep",
            "long_flat": "find_rest.long_flat",
            "long_drift": "find_rest.long_drift",
        },
        "thresholds": {
            "dive_depth_min_m": 2.0,
            "first_deriv_abs_max_ms": 0.6,
            "second_deriv_abs_max_ms2": 0.05,
            "min_duration_s": 180.0,
            "surface_sleep_min_duration_s": 600.0,
            "drift_rate_abs_max_ms": 0.55,
            "curvature_abs_max_ms": 0.30,
            "end_flat_abs_mean_d1_max_ms": 0.01,
        },
        "covariates": {
            "intrinsic": {},
            "extrinsic": {},
        },
        "context_filter_definitions": {},
        "context_filters": [],
        "debug": {
            "write_context_review_table": False,
            "persist_context_helper_channels": False,
            "persist_intermediate_channels": True,
        },
    }


def normalize_algorithmic_cfg(algo_cfg: Dict | None) -> Dict:
    base = copy.deepcopy(_default_algorithmic_cfg())
    incoming = copy.deepcopy(algo_cfg or {})
    if "end_segments_upon_d1_sign_change" not in incoming and "sign_consistency_required" in incoming:
        incoming["end_segments_upon_d1_sign_change"] = bool(incoming.get("sign_consistency_required"))
    incoming.pop("sign_consistency_required", None)
    for key, value in incoming.items():
        if key in {"event_keys", "label_codes", "label_names", "thresholds", "covariates", "debug", "context_filter_definitions"}:
            merged = copy.deepcopy(base.get(key) or {})
            if isinstance(value, dict):
                merged.update(copy.deepcopy(value))
            base[key] = merged
        else:
            base[key] = copy.deepcopy(value)
    covariates = copy.deepcopy(base.get("covariates") or {})
    covariates["intrinsic"] = copy.deepcopy(covariates.get("intrinsic") or {})
    covariates["extrinsic"] = copy.deepcopy(covariates.get("extrinsic") or {})
    base["covariates"] = covariates
    base["context_filter_definitions"] = copy.deepcopy(base.get("context_filter_definitions") or {})
    base["context_filters"] = copy.deepcopy(base.get("context_filters") or [])
    base["debug"] = copy.deepcopy(base.get("debug") or {})
    base["debug"].setdefault("write_context_review_table", False)
    base["debug"].setdefault("persist_context_helper_channels", False)
    base["debug"].setdefault("persist_intermediate_channels", True)
    return base


def context_filter_stage(filter_cfg: Dict[str, object]) -> str:
    cfg = dict(filter_cfg or {})
    explicit = str(cfg.get("filter_stage") or cfg.get("stage") or "").strip().lower()
    if explicit in {"measured", "measured_filters", "observed"}:
        return "measured"
    if explicit in {"inferred", "inferred_filters", "derived"}:
        return "inferred"
    filter_id = str(
        cfg.get("id")
        or cfg.get("ref")
        or cfg.get("definition_ref")
        or cfg.get("filter_ref")
        or ""
    ).strip().lower()
    if filter_id in _MEASURED_CONTEXT_FILTER_IDS:
        return "measured"
    return "inferred"


def summarize_context_filter_stages(algo_cfg: Dict | None) -> Dict[str, int]:
    sections = context_rule_sections_from_algo_cfg(algo_cfg)
    definitions = dict(sections.get("context_filter_definitions") or {})
    ordered = list(sections.get("context_filters") or [])
    out = {
        "total": 0,
        "enabled_total": 0,
        "measured_total": 0,
        "measured_enabled": 0,
        "inferred_total": 0,
        "inferred_enabled": 0,
    }
    for item in ordered:
        cfg = None
        if isinstance(item, str):
            ref = item.strip()
            if not ref:
                continue
            cfg = copy.deepcopy(definitions.get(ref) or {})
            cfg.setdefault("id", ref)
        elif isinstance(item, dict):
            cfg = copy.deepcopy(item)
            ref = str(cfg.get("ref") or cfg.get("definition_ref") or cfg.get("filter_ref") or "").strip()
            if ref:
                merged = copy.deepcopy(definitions.get(ref) or {})
                merged.update(cfg)
                cfg = merged
                cfg.setdefault("id", ref)
        if not isinstance(cfg, dict):
            continue
        out["total"] += 1
        stage = context_filter_stage(cfg)
        enabled = bool(cfg.get("enabled", True))
        if stage == "measured":
            out["measured_total"] += 1
            if enabled:
                out["measured_enabled"] += 1
        else:
            out["inferred_total"] += 1
            if enabled:
                out["inferred_enabled"] += 1
        if enabled:
            out["enabled_total"] += 1
    return out


def context_rule_sections_from_algo_cfg(algo_cfg: Dict | None) -> Dict:
    algo = normalize_algorithmic_cfg(algo_cfg)
    covariates = copy.deepcopy(algo.get("covariates") or {})
    return {
        "intrinsic": copy.deepcopy(covariates.get("intrinsic") or {}),
        "extrinsic": copy.deepcopy(covariates.get("extrinsic") or {}),
        "context_filter_definitions": copy.deepcopy(algo.get("context_filter_definitions") or {}),
        "context_filters": copy.deepcopy(algo.get("context_filters") or []),
        "debug": copy.deepcopy(algo.get("debug") or {}),
    }


def apply_context_rule_sections_to_algo_cfg(
    algo_cfg: Dict | None,
    intrinsic=None,
    extrinsic=None,
    context_filter_definitions=None,
    context_filters=None,
    debug=None,
) -> Dict:
    algo = normalize_algorithmic_cfg(algo_cfg)
    algo["covariates"] = {
        "intrinsic": copy.deepcopy(intrinsic if intrinsic is not None else (algo.get("covariates") or {}).get("intrinsic") or {}),
        "extrinsic": copy.deepcopy(extrinsic if extrinsic is not None else (algo.get("covariates") or {}).get("extrinsic") or {}),
    }
    algo["context_filter_definitions"] = copy.deepcopy(
        context_filter_definitions if context_filter_definitions is not None else algo.get("context_filter_definitions") or {}
    )
    algo["context_filters"] = copy.deepcopy(context_filters if context_filters is not None else algo.get("context_filters") or [])
    algo["debug"] = copy.deepcopy(debug if debug is not None else algo.get("debug") or {})
    algo["debug"].setdefault("write_context_review_table", False)
    algo["debug"].setdefault("persist_context_helper_channels", False)
    algo["debug"].setdefault("persist_intermediate_channels", True)
    return algo


def validate_context_rule_sections(algo_cfg: Dict | None) -> list[str]:
    algo = normalize_algorithmic_cfg(algo_cfg)
    errors = []
    sections = context_rule_sections_from_algo_cfg(algo)
    definitions = dict(sections.get("context_filter_definitions") or {})
    intrinsic = dict(sections.get("intrinsic") or {})
    extrinsic = dict(sections.get("extrinsic") or {})
    for cov_group_name, cov_group in [("intrinsic", intrinsic), ("extrinsic", extrinsic)]:
        if not isinstance(cov_group, dict):
            errors.append(f"{cov_group_name.title()} covariates must be a mapping.")
            continue
        for cov_id, cov_payload in cov_group.items():
            if not isinstance(cov_payload, dict):
                errors.append(f"Covariate '{cov_group_name}.{cov_id}' must be a mapping.")
    if not isinstance(definitions, dict):
        errors.append("Context filter definitions must be a mapping.")
        definitions = {}
    else:
        for filter_id, filter_payload in definitions.items():
            if not isinstance(filter_payload, dict):
                errors.append(f"Context filter definition '{filter_id}' must be a mapping.")
                continue
            action = str(filter_payload.get("action_on_fail") or "reject").strip()
            if action and action not in {"reject"}:
                errors.append(f"Context filter definition '{filter_id}' uses unsupported action_on_fail '{action}'.")
    context_filters = sections.get("context_filters") or []
    if not isinstance(context_filters, list):
        errors.append("Ordered context filters must be a list.")
        context_filters = []
    for idx, item in enumerate(context_filters):
        if isinstance(item, str):
            if item not in definitions:
                errors.append(f"Ordered context filter '{item}' does not match any context_filter_definitions entry.")
            continue
        if not isinstance(item, dict):
            errors.append(f"Ordered context filter at index {idx} must be a string reference or mapping.")
            continue
        ref_name = str(item.get("ref") or item.get("definition_ref") or item.get("filter_ref") or "").strip()
        if ref_name and ref_name not in definitions:
            errors.append(f"Ordered context filter '{ref_name}' does not match any context_filter_definitions entry.")
        cov_ref = str(item.get("covariate_ref") or "").strip()
        if cov_ref:
            if "." not in cov_ref:
                errors.append(f"Ordered context filter covariate_ref '{cov_ref}' should be prefixed with intrinsic. or extrinsic.")
            else:
                prefix, cov_id = cov_ref.split(".", 1)
                cov_bucket = intrinsic if prefix == "intrinsic" else extrinsic if prefix == "extrinsic" else None
                if cov_bucket is None or cov_id not in cov_bucket:
                    errors.append(f"Ordered context filter covariate_ref '{cov_ref}' does not resolve to a configured covariate.")
        action = str(item.get("action_on_fail") or "").strip()
        if action and action not in {"reject"}:
            errors.append(f"Ordered context filter '{item.get('id') or ref_name or idx}' uses unsupported action_on_fail '{action}'.")
    return list(dict.fromkeys(errors))


def format_json_block(payload) -> str:
    return json.dumps(payload if payload is not None else {}, indent=2, sort_keys=False)


def default_algorithmic_cfg_from_config(config_path: str, preferred_runs=None) -> Dict:
    cfg = _load_yaml(config_path) if config_path and os.path.exists(config_path) else {}
    workflows = (cfg.get("segmentation_workflows") or {})
    find_rest = workflows.get("find_rest") if isinstance(workflows, dict) else None
    if isinstance(find_rest, dict) and isinstance(find_rest.get("runtime_params"), dict):
        algo = normalize_algorithmic_cfg(find_rest.get("runtime_params") or {})
        if algo:
            return algo
    runs = (cfg.get("segmentation_runs") or {})
    preferred_runs = list(preferred_runs or [
        "mian_mile_depth_standardized_30s",
        "mile_depth_derivatives_30s",
    ])
    run_cfg = None
    for name in preferred_runs:
        candidate = runs.get(name)
        if isinstance(candidate, dict) and candidate.get("algorithmic_segments"):
            run_cfg = candidate
            break
    if run_cfg is None:
        for candidate in runs.values():
            if isinstance(candidate, dict) and candidate.get("algorithmic_segments"):
                run_cfg = candidate
                break
    algo = normalize_algorithmic_cfg((run_cfg or {}).get("algorithmic_segments") or {})
    return algo


def crop_data_pkl_to_window(data_pkl, start_ts, end_ts, tz_name: str):
    cropped = copy.deepcopy(data_pkl)
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    signal_data = getattr(cropped, "signal_data", {}) or {}
    for sig, df in list(signal_data.items()):
        if not isinstance(df, pd.DataFrame) or "datetime" not in df.columns:
            continue
        dt = _normalize_dt(df["datetime"], tz_name)
        mask = (dt >= start_ts) & (dt < end_ts)
        signal_data[sig] = df.loc[mask.fillna(False)].reset_index(drop=True)
    event_data = getattr(cropped, "event_data", None)
    if isinstance(event_data, pd.DataFrame) and "datetime" in event_data.columns:
        dt = _normalize_dt(event_data["datetime"], tz_name)
        mask = (dt >= start_ts) & (dt < end_ts)
        cropped.event_data = event_data.loc[mask.fillna(False)].reset_index(drop=True)
    return cropped


def _build_temp_segmentation_config(base_config_path: str, temp_data_root: str, dataset_id: str, deployment_id: str, algo_cfg: Dict, run_name: str, output_root: str) -> str:
    source_key = _extract_source_key(algo_cfg)
    payload = {
        "paths": {
            "local_private_data": temp_data_root,
        },
        "segmentation_runs": {
            run_name: {
                "analysis_id": run_name,
                "output_root_override": output_root,
                "dataset_ids": [dataset_id],
                "deployment_ids": [deployment_id],
                "channel_feature_spec": {
                    source_key: {
                        "transforms": ["raw"],
                        "feature_set": "minimal",
                    }
                },
                "algorithmic_segments": algo_cfg,
            }
        },
    }
    config_path = os.path.join(output_root, "segmentation_preview_config.yaml")
    pathlib.Path(output_root).mkdir(parents=True, exist_ok=True)
    _write_yaml(config_path, payload)
    return config_path


def _snakemake_executable_cmd() -> list:
    return [sys.executable, "-m", "snakemake"]


def _pick_first_existing_column(df: pd.DataFrame, candidates) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_segment_rows(segments_path: str, event_key_values) -> list[Dict]:
    if not segments_path or not os.path.exists(segments_path):
        return []
    try:
        seg_df = pd.read_parquet(segments_path)
    except Exception:
        return []
    if not isinstance(seg_df, pd.DataFrame) or seg_df.empty:
        return []

    key_col = _pick_first_existing_column(
        seg_df,
        [
            "event_key",
            "state_event_key",
            "state_key",
            "label_name",
            "key",
            "event",
            "state",
            "segment_key",
        ],
    )
    start_col = _pick_first_existing_column(seg_df, ["start", "start_time", "start_datetime", "datetime_start", "datetime"])
    end_col = _pick_first_existing_column(seg_df, ["end", "end_time", "end_datetime", "datetime_end", "stop"])
    if not key_col or not start_col or not end_col:
        return []

    out = seg_df.copy()
    out["_event_key"] = out[key_col].astype(str).str.strip()
    valid_keys = set(str(v).strip() for v in (event_key_values or []) if str(v or "").strip())
    if valid_keys:
        out = out.loc[out["_event_key"].isin(valid_keys)].copy()
    if out.empty:
        return []

    out["_start"] = pd.to_datetime(out[start_col], errors="coerce")
    out["_end"] = pd.to_datetime(out[end_col], errors="coerce")
    out = out.loc[out["_event_key"].ne("") & out["_start"].notna() & out["_end"].notna()].copy()
    out = out.loc[out["_end"] > out["_start"]].copy()
    if out.empty:
        return []

    out["duration"] = (out["_end"] - out["_start"]).dt.total_seconds()
    out["datetime"] = out["_start"]
    out["end_datetime"] = out["_end"]
    out["key"] = out["_event_key"]
    cols = ["key", "datetime", "end_datetime", "duration"]
    extra_cols = [c for c in ["segment_id", "cluster_id", "label_code", "source", "run_name"] if c in out.columns]
    return out[cols + extra_cols].to_dict("records")


def run_algorithmic_segmentation_sandbox(
    repo_root: str,
    base_config_path: str,
    real_data_pkl_path: str,
    dataset_id: str,
    deployment_id: str,
    algo_cfg: Dict,
    preview_start_ts=None,
    preview_end_ts=None,
) -> Dict:
    temp_root = tempfile.mkdtemp(prefix="pyologger_seg_")
    dataset_root = os.path.join(temp_root, dataset_id, deployment_id, "outputs")
    pathlib.Path(dataset_root).mkdir(parents=True, exist_ok=True)
    temp_pkl_path = os.path.join(dataset_root, "data.pkl")
    temp_output_root = os.path.join(temp_root, "segmentation_outputs")
    run_name = f"dash_seg_{_safe_slug(dataset_id)}_{_safe_slug(deployment_id)}_{uuid.uuid4().hex[:8]}"

    data_pkl = _load_pickle(real_data_pkl_path)
    tz_name = str((getattr(data_pkl, "deployment_info", {}) or {}).get("Time Zone") or "UTC")
    if preview_start_ts is not None and preview_end_ts is not None:
        data_pkl = crop_data_pkl_to_window(data_pkl, preview_start_ts, preview_end_ts, tz_name)
    _save_pickle(temp_pkl_path, data_pkl)

    config_path = _build_temp_segmentation_config(
        base_config_path=base_config_path,
        temp_data_root=temp_root,
        dataset_id=dataset_id,
        deployment_id=deployment_id,
        algo_cfg=algo_cfg,
        run_name=run_name,
        output_root=temp_output_root,
    )
    merged_output_path = os.path.join(temp_output_root, "segments", "algorithmic_segments.parquet")
    cmd = [
        sys.executable,
        "workflows/10_algorithmic_segmentation.py",
        "--config",
        config_path,
        "--run-name",
        run_name,
        "--output",
        merged_output_path,
    ]
    env = dict(os.environ)
    env["SEGMENTATION_RUNS_PATH"] = config_path
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    summary_path = os.path.join(temp_output_root, "segments", "by_deployment", f"{dataset_id}__{deployment_id}__algorithmic_segments_summary.json")
    segments_path = os.path.join(temp_output_root, "segments", "by_deployment", f"{dataset_id}__{deployment_id}__algorithmic_segments.parquet")
    result_pkl = _load_pickle(temp_pkl_path) if os.path.exists(temp_pkl_path) else None
    summary = {}
    if os.path.exists(summary_path):
        try:
            summary = json.loads(pathlib.Path(summary_path).read_text())
        except Exception:
            summary = {}
    if proc.returncode != 0:
        raise RuntimeError(
            "Snakemake segmentation run failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\n\nSTDERR:\n{proc.stderr[-4000:]}"
        )
    event_rows = []
    label_rows = []
    signal_info = {}
    event_manager = {}
    segment_rows = _load_segment_rows(segments_path, (algo_cfg.get("event_keys") or {}).values())

    if result_pkl is not None:
        event_data = getattr(result_pkl, "event_data", None)
        if isinstance(event_data, pd.DataFrame) and "key" in event_data.columns:
            key_set = set(str(v) for v in ((algo_cfg.get("event_keys") or {}).values()))
            filtered = event_data.loc[event_data["key"].astype(str).isin(key_set)].copy()
            if not filtered.empty:
                if "datetime" in filtered.columns:
                    filtered["datetime"] = pd.to_datetime(filtered["datetime"], errors="coerce")
                event_rows = filtered.to_dict("records")
        sig_df = (getattr(result_pkl, "signal_data", {}) or {}).get("algorithmic_labels")
        if isinstance(sig_df, pd.DataFrame):
            if "datetime" in sig_df.columns:
                sig_df = sig_df.copy()
                sig_df["datetime"] = pd.to_datetime(sig_df["datetime"], errors="coerce")
            label_rows = sig_df.to_dict("records")
        signal_info = dict(((getattr(result_pkl, "signal_info", {}) or {}).get("algorithmic_labels") or {}))
        event_manager = dict(((getattr(result_pkl, "event_manager", {}) or {}).get("algorithmic_segments") or {}))
    return {
        "temp_root": temp_root,
        "config_path": config_path,
        "run_name": run_name,
        "summary": summary,
        "event_rows": event_rows,
        "segment_rows": segment_rows,
        "label_rows": label_rows,
        "signal_info": signal_info,
        "event_manager": event_manager,
        "segments_path": segments_path,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def save_algorithmic_results_to_real_pkl(
    real_data_pkl_path: str,
    payload: Dict,
    *,
    target_event_keys=None,
    write_netcdf: bool = False,
    deployment_folder: str | None = None,
    deployment_id: str | None = None,
) -> Dict:
    data_pkl = _load_pickle(real_data_pkl_path)
    algo_cfg = dict(payload.get("algorithmic_cfg") or {})
    configured_event_keys = set(str(v) for v in ((algo_cfg.get("event_keys") or {}).values()))
    if target_event_keys is None:
        event_keys = configured_event_keys
    else:
        event_keys = set(str(v).strip() for v in (target_event_keys or []) if str(v or "").strip())

    if not hasattr(data_pkl, "event_data") or not isinstance(data_pkl.event_data, pd.DataFrame):
        data_pkl.event_data = pd.DataFrame(columns=["date", "time", "value", "type", "key", "duration", "short_description", "long_description", "datetime"])
    existing = data_pkl.event_data.copy()
    removed_count = 0
    if "key" in existing.columns:
        mask = existing["key"].astype(str).isin(event_keys)
        removed_count = int(mask.sum())
        existing = existing.loc[~mask].copy()

    new_events = pd.DataFrame(payload.get("event_rows") or [])
    if not new_events.empty and event_keys and "key" in new_events.columns:
        new_events = new_events.loc[new_events["key"].astype(str).isin(event_keys)].copy()
    if not new_events.empty:
        if "datetime" in new_events.columns:
            new_events["datetime"] = pd.to_datetime(new_events["datetime"], errors="coerce")
        data_pkl.event_data = pd.concat([existing, new_events], ignore_index=True)
        if "datetime" in data_pkl.event_data.columns:
            data_pkl.event_data = data_pkl.event_data.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    else:
        data_pkl.event_data = existing.reset_index(drop=True)

    if not hasattr(data_pkl, "signal_data") or data_pkl.signal_data is None:
        data_pkl.signal_data = {}
    if not hasattr(data_pkl, "signal_info") or data_pkl.signal_info is None:
        data_pkl.signal_info = {}

    label_df = pd.DataFrame(payload.get("label_rows") or [])
    if not label_df.empty and "datetime" in label_df.columns:
        label_df["datetime"] = pd.to_datetime(label_df["datetime"], errors="coerce")
    data_pkl.signal_data["algorithmic_labels"] = label_df
    data_pkl.signal_info["algorithmic_labels"] = dict(payload.get("signal_info") or {})

    if not hasattr(data_pkl, "event_manager") or data_pkl.event_manager is None:
        data_pkl.event_manager = {}
    data_pkl.event_manager["algorithmic_segments"] = dict(payload.get("event_manager") or {})

    _save_pickle(real_data_pkl_path, data_pkl)
    netcdf_path = None
    netcdf_error = None
    if write_netcdf:
        try:
            from pyologger.io_operations.base_exporter import BaseExporter
            from pyologger.utils.workflow_netcdf import final_netcdf_path, latest_processing_netcdf_path

            dep_folder = str(deployment_folder or "").strip()
            dep_id = str(deployment_id or "").strip()
            if not dep_folder or not dep_id:
                raise ValueError("deployment_folder and deployment_id are required for NetCDF export")
            latest_path = latest_processing_netcdf_path(dep_folder, dep_id)
            netcdf_path = latest_path or final_netcdf_path(dep_folder, dep_id)
            BaseExporter(data_pkl).save_to_netcdf(data_pkl, filepath=netcdf_path)
        except Exception as e:
            netcdf_error = f"{type(e).__name__}: {e}"

    return {
        "removed_event_count": removed_count,
        "saved_event_count": int(len(new_events)),
        "saved_label_rows": int(len(label_df)),
        "target_event_count": int(len(event_keys)),
        "netcdf_written": bool(netcdf_path and not netcdf_error),
        "netcdf_path": netcdf_path,
        "netcdf_error": netcdf_error,
    }


def load_algorithmic_results_from_real_pkl(real_data_pkl_path: str, algo_cfg: Dict | None = None) -> Dict:
    data_pkl = _load_pickle(real_data_pkl_path)
    event_manager = dict(((getattr(data_pkl, "event_manager", {}) or {}).get("algorithmic_segments") or {}))
    event_keys = set(str(v) for v in (event_manager.get("keys") or []) if str(v).strip())
    if not event_keys:
        event_keys = set(str(v) for v in (((algo_cfg or {}).get("event_keys") or {}).values()) if str(v).strip())

    event_rows = []
    event_data = getattr(data_pkl, "event_data", None)
    if isinstance(event_data, pd.DataFrame) and "key" in event_data.columns and event_keys:
        filtered = event_data.loc[event_data["key"].astype(str).isin(event_keys)].copy()
        if not filtered.empty:
            if "datetime" in filtered.columns:
                filtered["datetime"] = pd.to_datetime(filtered["datetime"], errors="coerce")
            event_rows = filtered.to_dict("records")

    label_rows = []
    sig_df = (getattr(data_pkl, "signal_data", {}) or {}).get("algorithmic_labels")
    if isinstance(sig_df, pd.DataFrame):
        sig_df = sig_df.copy()
        if "datetime" in sig_df.columns:
            sig_df["datetime"] = pd.to_datetime(sig_df["datetime"], errors="coerce")
        label_rows = sig_df.to_dict("records")

    signal_info = dict(((getattr(data_pkl, "signal_info", {}) or {}).get("algorithmic_labels") or {}))
    return {
        "event_rows": event_rows,
        "label_rows": label_rows,
        "signal_info": signal_info,
        "event_manager": event_manager,
        "has_saved_results": bool(event_rows or label_rows or event_manager),
    }


def cleanup_temp_workspace(temp_root: str) -> None:
    if temp_root and os.path.isdir(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)


SEGMENTATION_WORKFLOW_ALLOWED_NODE_TYPES = {
    "derive",
    "decision",
    "filter_group",
    "terminal_state",
    "terminal_reject",
}


def _workflow_terminal_states_from_algo_cfg(algo_cfg: Dict) -> Dict:
    event_keys = dict(algo_cfg.get("event_keys") or {})
    label_codes = dict(algo_cfg.get("label_codes") or {})
    label_names = dict(algo_cfg.get("label_names") or {})
    return {
        "not_sleep": {
            "event_key": "",
            "label_code": int(label_codes.get("not_sleep", 0)),
            "label_name": str(label_names.get("not_sleep", "find_rest.not_sleep")),
        },
        "surface_sleep": {
            "event_key": str(event_keys.get("surface_sleep", "find_rest.surface_sleep")),
            "label_code": int(label_codes.get("surface_sleep", 1)),
            "label_name": str(label_names.get("surface_sleep", "find_rest.surface_sleep")),
        },
        "long_flat": {
            "event_key": str(event_keys.get("long_flat", "find_rest.long_flat")),
            "label_code": int(label_codes.get("long_flat", 2)),
            "label_name": str(label_names.get("long_flat", "find_rest.long_flat")),
        },
        "long_drift": {
            "event_key": str(event_keys.get("long_drift", "find_rest.long_drift")),
            "label_code": int(label_codes.get("long_drift", 3)),
            "label_name": str(label_names.get("long_drift", "find_rest.long_drift")),
        },
    }


def build_find_rest_workflow_preset(algo_cfg: Dict | None = None) -> Dict:
    algo_cfg = normalize_algorithmic_cfg(algo_cfg or default_algorithmic_cfg_from_config(""))
    thr = dict(algo_cfg.get("thresholds") or {})
    terminals = _workflow_terminal_states_from_algo_cfg(algo_cfg)
    return {
        "workflow_name": "find_rest",
        "display_name": "Find Rest",
        "workflow_type": "decision_tree",
        "source_signals": [str(algo_cfg.get("source_channel") or "depth.depth")],
        "derived_channels": copy.deepcopy(FIND_REST_DERIVED_CHANNELS),
        "default_review_signals": ["depth"],
        "ordering_channel": "depth.depth",
        "terminal_states": terminals,
        "label_codes": dict(algo_cfg.get("label_codes") or {}),
        "runtime_params": copy.deepcopy(algo_cfg),
        "nodes": [
            {
                "id": "root",
                "type": "filter_group",
                "label": "find_rest",
                "description": "Root workflow",
                "position": {"x": 520, "y": 40},
            },
            {
                "id": "surface_long",
                "type": "decision",
                "label": "Long surface interval?",
                "description": "Potential surface sleep",
                "channel": "segment_duration_s",
                "operator": ">=",
                "threshold": float(thr.get("surface_sleep_min_duration_s", 600.0)),
                "binding_path": "thresholds.surface_sleep_min_duration_s",
                "position": {"x": 220, "y": 150},
            },
            {
                "id": "surface_sleep",
                "type": "terminal_state",
                "label": "surface_sleep",
                "state_key": terminals["surface_sleep"]["event_key"],
                "label_code": terminals["surface_sleep"]["label_code"],
                "label_name": terminals["surface_sleep"]["label_name"],
                "position": {"x": 220, "y": 270},
            },
            {
                "id": "drift_candidate",
                "type": "decision",
                "label": "Drift candidate?",
                "description": "Low |d1| and |d2|",
                "channel": "depth_d1_ms/depth_d2_ms2",
                "operator": "<=",
                "threshold": float(thr.get("first_deriv_abs_max_ms", 0.6)),
                "binding_path": "thresholds.first_deriv_abs_max_ms",
                "secondary_binding_path": "thresholds.second_deriv_abs_max_ms2",
                "secondary_threshold": float(thr.get("second_deriv_abs_max_ms2", 0.05)),
                "position": {"x": 520, "y": 150},
            },
            {
                "id": "drift_long",
                "type": "decision",
                "label": "Long drift?",
                "description": "Duration threshold",
                "channel": "segment_duration_s",
                "operator": ">=",
                "threshold": float(thr.get("min_duration_s", 180.0)),
                "binding_path": "thresholds.min_duration_s",
                "position": {"x": 520, "y": 270},
            },
            {
                "id": "flat_check",
                "type": "decision",
                "label": "Flat enough?",
                "description": "Low end-of-segment slope",
                "channel": "depth_d1_end_ms",
                "operator": "<=",
                "threshold": float(thr.get("end_flat_abs_mean_d1_max_ms", 0.01)),
                "binding_path": "thresholds.end_flat_abs_mean_d1_max_ms",
                "position": {"x": 520, "y": 390},
            },
            {
                "id": "long_flat",
                "type": "terminal_state",
                "label": "long_flat",
                "state_key": terminals["long_flat"]["event_key"],
                "label_code": terminals["long_flat"]["label_code"],
                "label_name": terminals["long_flat"]["label_name"],
                "position": {"x": 770, "y": 390},
            },
            {
                "id": "plausible_drift",
                "type": "decision",
                "label": "Drift rate plausible?",
                "description": "Within smoothed drift-rate threshold",
                "channel": "drift_rate_ms",
                "operator": "<=",
                "threshold": float(thr.get("drift_rate_abs_max_ms", 0.55)),
                "binding_path": "thresholds.drift_rate_abs_max_ms",
                "position": {"x": 520, "y": 510},
                "position": {"x": 520, "y": 630},
            },
            {
                "id": "long_drift",
                "type": "terminal_state",
                "label": "long_drift",
                "state_key": terminals["long_drift"]["event_key"],
                "label_code": terminals["long_drift"]["label_code"],
                "label_name": terminals["long_drift"]["label_name"],
                "position": {"x": 770, "y": 630},
            },
            {
                "id": "not_sleep",
                "type": "terminal_reject",
                "label": "not_sleep",
                "state_key": "",
                "label_code": terminals["not_sleep"]["label_code"],
                "label_name": terminals["not_sleep"]["label_name"],
                "position": {"x": 250, "y": 630},
            },
        ],
        "edges": [
            {"source": "root", "target": "surface_long", "branch": "surface"},
            {"source": "root", "target": "drift_candidate", "branch": "drift"},
            {"source": "surface_long", "target": "surface_sleep", "branch": "yes"},
            {"source": "surface_long", "target": "not_sleep", "branch": "no"},
            {"source": "drift_candidate", "target": "drift_long", "branch": "yes"},
            {"source": "drift_candidate", "target": "not_sleep", "branch": "no"},
            {"source": "drift_long", "target": "flat_check", "branch": "yes"},
            {"source": "drift_long", "target": "not_sleep", "branch": "no"},
            {"source": "flat_check", "target": "long_flat", "branch": "yes"},
            {"source": "flat_check", "target": "plausible_drift", "branch": "no"},
            {"source": "plausible_drift", "target": "long_drift", "branch": "yes"},
            {"source": "plausible_drift", "target": "not_sleep", "branch": "no"},
        ],
    }


def sync_workflow_nodes_from_runtime_params(workflow: Dict) -> Dict:
    workflow = copy.deepcopy(workflow or {})
    runtime = workflow_preset_to_algorithmic_cfg(workflow, fallback_cfg=default_algorithmic_cfg_from_config(""))
    nodes = []
    for raw_node in list(workflow.get("nodes") or []):
        node = copy.deepcopy(raw_node or {})
        binding_path = str(node.get("binding_path") or "").strip()
        secondary_binding = str(node.get("secondary_binding_path") or "").strip()
        if binding_path:
            bound_value = _get_nested_path(runtime, binding_path)
            if bound_value is not None:
                node["threshold"] = bound_value
        if secondary_binding:
            bound_secondary = _get_nested_path(runtime, secondary_binding)
            if bound_secondary is not None:
                node["secondary_threshold"] = bound_secondary
        ntype = str(node.get("type") or "")
        label_lower = str(node.get("label") or "").strip().lower()
        if ntype == "terminal_state":
            if "surface" in label_lower:
                key = "surface_sleep"
            elif "flat" in label_lower:
                key = "long_flat"
            elif "drift" in label_lower:
                key = "long_drift"
            else:
                key = None
            if key:
                event_keys = dict(runtime.get("event_keys") or {})
                label_codes = dict(runtime.get("label_codes") or {})
                label_names = dict(runtime.get("label_names") or {})
                if event_keys.get(key):
                    node["state_key"] = event_keys.get(key)
                if key in label_codes:
                    node["label_code"] = label_codes.get(key)
                if label_names.get(key):
                    node["label_name"] = label_names.get(key)
        elif ntype == "terminal_reject":
            label_codes = dict(runtime.get("label_codes") or {})
            label_names = dict(runtime.get("label_names") or {})
            if "not_sleep" in label_codes:
                node["label_code"] = label_codes.get("not_sleep")
            if label_names.get("not_sleep"):
                node["label_name"] = label_names.get("not_sleep")
        nodes.append(node)
    workflow["nodes"] = nodes
    return workflow


def workflow_runtime_support_report(workflow: Dict) -> Dict:
    workflow = sync_workflow_nodes_from_runtime_params(workflow)
    runtime = workflow_preset_to_algorithmic_cfg(workflow, fallback_cfg=default_algorithmic_cfg_from_config(""))
    errors = validate_segmentation_workflow_preset(workflow)
    context_errors = validate_context_rule_sections(runtime)
    canonical = build_find_rest_workflow_preset(runtime) if str(workflow.get("workflow_name") or "") == "find_rest" else None

    def _norm_nodes(items):
        out = []
        for raw in list(items or []):
            node = dict(raw or {})
            out.append(
                {
                    "id": str(node.get("id") or ""),
                    "type": str(node.get("type") or ""),
                    "binding_path": str(node.get("binding_path") or ""),
                    "secondary_binding_path": str(node.get("secondary_binding_path") or ""),
                    "channel": str(node.get("channel") or ""),
                    "operator": str(node.get("operator") or ""),
                    "state_key": str(node.get("state_key") or ""),
                    "label_code": node.get("label_code"),
                    "label_name": str(node.get("label_name") or ""),
                }
            )
        return sorted(out, key=lambda item: (item["id"], item["type"]))

    def _norm_edges(items):
        out = []
        for raw in list(items or []):
            edge = dict(raw or {})
            branch = edge.get("branch")
            if isinstance(branch, bool):
                branch = "yes" if branch else "no"
            out.append(
                {
                    "source": str(edge.get("source") or ""),
                    "target": str(edge.get("target") or ""),
                    "branch": str(branch or ""),
                }
            )
        return sorted(out, key=lambda item: (item["source"], item["target"], item["branch"]))

    base_supported = not errors
    context_valid = not context_errors
    supported = base_supported and context_valid
    reasons = []
    if canonical is None:
        base_supported = False
        supported = False
        reasons.append("Only the built-in 'find_rest' runtime adapter is executable in the current backend.")
    else:
        if _norm_nodes(workflow.get("nodes")) != _norm_nodes(canonical.get("nodes")):
            base_supported = False
            supported = False
            reasons.append("Node semantics no longer match the implemented 'find_rest' runtime.")
        if _norm_edges(workflow.get("edges")) != _norm_edges(canonical.get("edges")):
            base_supported = False
            supported = False
            reasons.append("Tree topology/branches differ from the implemented 'find_rest' runtime.")
    if errors:
        reasons.extend(errors)
    if context_errors:
        reasons.extend(context_errors)
    return {
        "valid": (not errors) and (not context_errors),
        "base_detector_valid": not errors,
        "base_detector_supported": base_supported,
        "context_rules_valid": context_valid,
        "runtime_supported": supported,
        "messages": list(dict.fromkeys(reasons)),
    }


def load_segmentation_workflow_presets(config_path: str) -> Dict:
    cfg = _load_yaml(config_path) if config_path and os.path.exists(config_path) else {}
    workflows = copy.deepcopy((cfg.get("segmentation_workflows") or {}))
    if not workflows:
        workflows = {"find_rest": build_find_rest_workflow_preset(default_algorithmic_cfg_from_config(config_path))}
    for key, workflow in list(workflows.items()):
        if not isinstance(workflow, dict):
            workflows.pop(key, None)
            continue
        workflow.setdefault("workflow_name", str(key))
        workflow.setdefault("display_name", str(key).replace("_", " ").title())
        workflow.setdefault("workflow_type", "decision_tree")
        workflow.setdefault("default_review_signals", ["depth"])
        workflow.setdefault("ordering_channel", "depth.depth")
        workflow["runtime_params"] = normalize_algorithmic_cfg(workflow.get("runtime_params") or default_algorithmic_cfg_from_config(config_path))
        workflow.setdefault("nodes", [])
        workflow.setdefault("edges", [])
        workflow.setdefault("terminal_states", _workflow_terminal_states_from_algo_cfg(workflow.get("runtime_params") or {}))
        workflow.setdefault("label_codes", dict((workflow.get("runtime_params") or {}).get("label_codes") or {}))
        if str(workflow.get("workflow_name") or key) == "find_rest":
            canonical = build_find_rest_workflow_preset(workflow.get("runtime_params") or {})
            canonical["display_name"] = str(workflow.get("display_name") or canonical.get("display_name") or key)
            canonical["default_review_signals"] = list(workflow.get("default_review_signals") or canonical.get("default_review_signals") or ["depth"])
            workflows[key] = sync_workflow_nodes_from_runtime_params(canonical)
        else:
            workflows[key] = sync_workflow_nodes_from_runtime_params(workflow)
    return workflows


def save_segmentation_workflow_presets(config_path: str, presets: Dict) -> None:
    cfg = _load_yaml(config_path)
    cfg["segmentation_workflows"] = copy.deepcopy(presets or {})
    _write_yaml(config_path, cfg)


def workflow_preset_to_algorithmic_cfg(workflow: Dict, fallback_cfg: Dict | None = None) -> Dict:
    if isinstance((workflow or {}).get("runtime_params"), dict):
        return normalize_algorithmic_cfg(workflow.get("runtime_params") or {})
    return normalize_algorithmic_cfg(fallback_cfg or {})


def segmentation_workflow_options(workflows: Dict) -> list:
    opts = []
    for key, workflow in sorted((workflows or {}).items(), key=lambda kv: str((kv[1] or {}).get("display_name") or kv[0]).lower()):
        label = str((workflow or {}).get("display_name") or key)
        opts.append({"label": label, "value": str(key)})
    return opts


def duplicate_segmentation_workflow(workflows: Dict, selected_name: str) -> tuple[Dict, str]:
    workflows = copy.deepcopy(workflows or {})
    base = copy.deepcopy((workflows.get(selected_name) or {}))
    if not base:
        raise ValueError(f"Workflow '{selected_name}' not found")
    stem = f"{selected_name}_copy"
    idx = 1
    new_name = stem
    while new_name in workflows:
        idx += 1
        new_name = f"{stem}_{idx}"
    base["workflow_name"] = new_name
    base["display_name"] = f"{str(base.get('display_name') or selected_name)} Copy"
    workflows[new_name] = base
    return workflows, new_name


def new_segmentation_workflow_from_current(current_workflow: Dict, base_name: str = "new_workflow") -> tuple[Dict, str]:
    workflow = copy.deepcopy(current_workflow or build_find_rest_workflow_preset())
    workflow["workflow_name"] = base_name
    workflow["display_name"] = str(base_name).replace("_", " ").title()
    return workflow, base_name


def validate_segmentation_workflow_preset(workflow: Dict) -> list[str]:
    errors = []
    if not isinstance(workflow, dict):
        return ["Workflow must be a dictionary."]
    nodes = list(workflow.get("nodes") or [])
    edges = list(workflow.get("edges") or [])
    node_ids = [str((n or {}).get("id") or "").strip() for n in nodes]
    if not node_ids or any(not nid for nid in node_ids):
        errors.append("All nodes must have non-empty ids.")
    if len(node_ids) != len(set(node_ids)):
        errors.append("Node ids must be unique.")
    node_map = {str(n.get("id")): n for n in nodes if str(n.get("id") or "").strip()}
    incoming = {nid: 0 for nid in node_map}
    outgoing = {nid: [] for nid in node_map}
    for node in nodes:
        node_type = str((node or {}).get("type") or "")
        if node_type not in SEGMENTATION_WORKFLOW_ALLOWED_NODE_TYPES:
            errors.append(f"Invalid node type '{node_type}' for node '{node.get('id')}'.")
    for edge in edges:
        src = str((edge or {}).get("source") or "").strip()
        tgt = str((edge or {}).get("target") or "").strip()
        if src not in node_map or tgt not in node_map:
            errors.append(f"Edge '{src}->{tgt}' references missing node.")
            continue
        incoming[tgt] += 1
        outgoing[src].append(str((edge or {}).get("branch") or ""))
    roots = [nid for nid, count in incoming.items() if count == 0]
    if len(roots) != 1:
        errors.append("Workflow must have exactly one root node.")
    for nid, branches in outgoing.items():
        ntype = str((node_map.get(nid) or {}).get("type") or "")
        if ntype in {"terminal_state", "terminal_reject"} and branches:
            errors.append(f"Terminal node '{nid}' cannot have outgoing edges.")
        if ntype in {"decision", "filter_group"} and not branches:
            errors.append(f"Node '{nid}' must have at least one outgoing edge.")
        if len(branches) != len(set(branches)):
            errors.append(f"Node '{nid}' has duplicate branch labels.")
    visited = set()
    stack = set()

    def _dfs(nid: str):
        if nid in stack:
            errors.append("Workflow contains a cycle.")
            return
        if nid in visited:
            return
        visited.add(nid)
        stack.add(nid)
        for edge in edges:
            if str(edge.get("source") or "") == nid:
                _dfs(str(edge.get("target") or ""))
        stack.remove(nid)

    if roots:
        _dfs(roots[0])
    for node in nodes:
        if str((node or {}).get("type") or "") == "terminal_state":
            if not str((node or {}).get("state_key") or "").strip():
                errors.append(f"Terminal state node '{node.get('id')}' is missing state_key.")
            if (node or {}).get("label_code") is None:
                errors.append(f"Terminal state node '{node.get('id')}' is missing label_code.")
            if not str((node or {}).get("label_name") or "").strip():
                errors.append(f"Terminal state node '{node.get('id')}' is missing label_name.")
    return list(dict.fromkeys(errors))


def segmentation_workflow_to_cytoscape_elements(workflow: Dict) -> list[Dict]:
    workflow = copy.deepcopy(workflow or {})
    elements = []
    nodes = list(workflow.get("nodes") or [])
    edges = list(workflow.get("edges") or [])

    def _display_text(raw: object) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        return text.replace("_", " ").replace(".", " ")

    def _display_operator(raw: object) -> str:
        op = str(raw or "").strip()
        return {
            "<=": "≤",
            ">=": "≥",
            "!=": "≠",
            "==": "=",
        }.get(op, op)

    def _node_label(node: Dict) -> str:
        ntype = str(node.get("type") or "decision")
        title = _display_text(node.get("label") or node.get("label_name") or node.get("id"))
        if ntype in {"decision", "filter_group"} and node.get("threshold") is not None:
            channel = _display_text(node.get("channel"))
            operator = _display_operator(node.get("operator"))
            threshold = str(node.get("threshold"))
            if operator in {"", "custom"}:
                return title
            secondary_threshold = node.get("secondary_threshold")
            if secondary_threshold is not None and "/" in channel:
                lhs, rhs = [part.strip() for part in channel.split("/", 1)]
                return f"{title}\n{lhs} {operator} {threshold}\n{rhs} {operator} {secondary_threshold}".strip()
            if channel and operator:
                return f"{title}\n{channel} {operator} {threshold}".strip()
        if ntype == "terminal_state":
            return _display_text(node.get("label") or node.get("label_name") or node.get("id"))
        if ntype == "terminal_reject":
            return _display_text(node.get("label") or node.get("label_name") or "not_sleep")
        return title

    def _branch_text(raw_branch: object) -> str:
        if isinstance(raw_branch, bool):
            return "yes" if raw_branch else "no"
        return str(raw_branch or "").strip()

    def _branch_class(branch: str) -> str:
        safe = re.sub(r"[^a-z0-9]+", "-", str(branch or "").strip().lower()).strip("-")
        return f"seg-edge-{safe}" if safe else "seg-edge-unlabeled"

    for node in nodes:
        nid = str(node.get("id") or "").strip()
        if not nid:
            continue
        ntype = str(node.get("type") or "decision")
        elements.append(
            {
                "data": {
                    "id": nid,
                    "label": _node_label(node),
                    "node_type": ntype,
                },
                "position": dict((node.get("position") or {})),
                "classes": f"seg-node seg-node-{ntype}",
            }
        )
    node_pos = {str((n or {}).get("id") or ""): dict((n or {}).get("position") or {}) for n in nodes}
    pair_counts = Counter(
        (str((e or {}).get("source") or "").strip(), str((e or {}).get("target") or "").strip())
        for e in edges
        if str((e or {}).get("source") or "").strip() and str((e or {}).get("target") or "").strip()
    )
    seen_pairs = Counter()
    for edge in edges:
        src = str(edge.get("source") or "").strip()
        tgt = str(edge.get("target") or "").strip()
        if not src or not tgt:
            continue
        branch = _branch_text(edge.get("branch"))
        branch_cls = _branch_class(branch)
        pair = (src, tgt)
        seen_pairs[pair] += 1
        if pair_counts[pair] > 1:
            src_pos = node_pos.get(src) or {"x": 0, "y": 0}
            tgt_pos = node_pos.get(tgt) or {"x": 0, "y": 0}
            offset_idx = seen_pairs[pair] - 1
            bend = -40 if offset_idx % 2 == 0 else 40
            bend *= (offset_idx // 2) + 1
            mid_x = (float(src_pos.get("x", 0)) + float(tgt_pos.get("x", 0))) / 2.0
            mid_y = (float(src_pos.get("y", 0)) + float(tgt_pos.get("y", 0))) / 2.0 + bend
            junction_id = f"junction__{src}__{tgt}__{offset_idx}"
            elements.append(
                {
                    "data": {
                        "id": junction_id,
                        "label": "",
                        "node_type": "junction",
                    },
                    "position": {"x": mid_x, "y": mid_y},
                    "classes": "seg-node seg-node-junction",
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"{src}__{branch}__{junction_id}",
                        "source": src,
                        "target": junction_id,
                        "label": branch,
                    },
                    "classes": f"seg-edge {branch_cls}",
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"{junction_id}__to__{tgt}",
                        "source": junction_id,
                        "target": tgt,
                        "label": "",
                    },
                    "classes": "seg-edge",
                }
            )
            continue
        elements.append(
            {
                "data": {
                    "id": f"{src}__{branch}__{tgt}",
                    "source": src,
                    "target": tgt,
                    "label": branch,
                },
                "classes": f"seg-edge {branch_cls}",
            }
        )
    return elements
