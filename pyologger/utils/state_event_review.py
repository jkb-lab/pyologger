from __future__ import annotations

import copy
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yaml

from pyologger.utils.folder_manager import load_combined_config, resolve_segmentation_runs_path
from pyologger.utils.segmentation_run_config import normalize_segmentation_run_cfg


DEFAULT_CANDIDATE_RANK = 10
DEFAULT_WINDOW_HOURS = 24.0


def _canonical_review_rest_state_name(value: Any) -> str:
    text = str(value or "").strip()
    mapping = {
        "find_rest.surface_sleep": "putative_rest.surface_sleep",
        "find_rest.long_flat": "putative_rest.benthic_sleep",
        "find_rest.long_drift": "putative_rest.drift_sleep",
    }
    return mapping.get(text, text)


def _default_terminal_criteria() -> Dict[str, Dict[str, Any]]:
    return {
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
    }


def _source_uses_reversed_axis(source_channel: str | None) -> bool:
    source_lower = str(source_channel or "").strip().lower()
    return any(token in source_lower for token in ["depth", "pressure"])


def _humanize_buoyancy_phase(value: Any) -> str:
    text = str(value or "").strip()
    mapping = {
        "pre_positive": "negative inferred buoyancy",
        "never_positive": "negative inferred buoyancy",
        "negative_buoyancy": "negative inferred buoyancy",
        "post_positive": "positive inferred buoyancy",
        "always_positive": "positive inferred buoyancy",
        "positive_buoyancy": "positive inferred buoyancy",
        "neutral_buoyancy": "neutral inferred buoyancy",
    }
    return mapping.get(text, text)


def context_filter_stage(filter_cfg: Dict[str, Any]) -> str:
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


def _run_root_candidates(data_dir: str, dataset_id: str | None) -> List[Tuple[str, pathlib.Path, str]]:
    roots: List[Tuple[str, pathlib.Path, str]] = []
    global_root = pathlib.Path(data_dir) / "00_Meta-Analysis" / "segmentation"
    roots.append(("global", global_root, "All datasets"))
    ds = str(dataset_id or "").strip()
    if ds:
        dataset_root = pathlib.Path(data_dir) / ds / "00_Meta-Analysis" / "segmentation"
        roots.append((f"dataset:{ds}", dataset_root, ds))
        return roots
    data_root = pathlib.Path(data_dir)
    if not data_root.exists():
        return roots
    for child in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        name = str(child.name).strip()
        if not name or name.startswith(".") or name == "00_Meta-Analysis":
            continue
        dataset_root = child / "00_Meta-Analysis" / "segmentation"
        if dataset_root.exists():
            roots.append((f"dataset:{name}", dataset_root, name))
    return roots


def _safe_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    # Avoid pandas' pending silent-downcast behavior change on object arrays.
    filled = series.where(series.notna(), default)
    return filled.infer_objects(copy=False).astype(bool)


def _split_pipe_values(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def _normalize_algorithmic_cfg(algo_cfg: Dict | None) -> Dict:
    cfg = dict(algo_cfg or {})
    thresholds = dict(cfg.get("thresholds") or {})
    covariates = dict(cfg.get("covariates") or {})
    debug = dict(cfg.get("debug") or {})
    terminal_criteria = copy.deepcopy(_default_terminal_criteria()) if "copy" in globals() else _default_terminal_criteria()
    for key, value in dict(cfg.get("terminal_criteria") or {}).items():
        if not isinstance(value, dict):
            continue
        merged = dict(terminal_criteria.get(str(key), {}))
        merged.update(value)
        terminal_criteria[str(key)] = merged
    pass_mode = str(cfg.get("context_filter_pass_mode") or "measured_only").strip().lower() or "measured_only"
    if pass_mode not in {"none", "measured_only", "full"}:
        pass_mode = "measured_only"
    out = {
        "enabled": bool(cfg.get("enabled", True)),
        "method": str(cfg.get("method") or "depth_drift_thresholds"),
        "method_name": str(cfg.get("method_name") or "find_rest"),
        "source_channel": str(cfg.get("source_channel") or cfg.get("source_key") or "depth.depth"),
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
            "drift_rate_abs_max_ms": float(thresholds.get("drift_rate_abs_max_ms", 0.55)),
            "curvature_abs_max_ms": float(thresholds.get("curvature_abs_max_ms", 0.30)),
            "end_flat_abs_mean_d1_max_ms": float(thresholds.get("end_flat_abs_mean_d1_max_ms", 0.01)),
        },
        "covariates": {
            "intrinsic": dict(covariates.get("intrinsic") or {}),
            "extrinsic": dict(covariates.get("extrinsic") or {}),
        },
        "context_filters": list(cfg.get("context_filters") or []),
        "context_filter_pass_mode": pass_mode,
        "terminal_criteria": terminal_criteria,
        "debug": {
            "write_context_review_table": bool(debug.get("write_context_review_table", False)),
            "persist_context_helper_channels": bool(debug.get("persist_context_helper_channels", False)),
            "persist_intermediate_channels": bool(debug.get("persist_intermediate_channels", True)),
        },
    }
    source_is_reversed_axis = _source_uses_reversed_axis(out.get("source_channel"))
    legacy_filter_out_positive = bool(cfg.get("filter_out_positive_slopes", False))
    legacy_filter_out_negative = bool(cfg.get("filter_out_negative_slopes", True))
    out["filter_out_ascent"] = bool(
        cfg.get(
            "filter_out_ascent",
            cfg.get(
                "filter_out_ascent_slopes",
                legacy_filter_out_negative if source_is_reversed_axis else legacy_filter_out_positive,
            ),
        )
    )
    out["filter_out_descent"] = bool(
        cfg.get(
            "filter_out_descent",
            legacy_filter_out_negative if source_is_reversed_axis else legacy_filter_out_positive,
        )
    )
    if "filter_out_descent" not in cfg:
        out["filter_out_descent"] = bool(
            cfg.get(
                "filter_out_descent_slopes",
                legacy_filter_out_positive if source_is_reversed_axis else legacy_filter_out_negative,
            )
        )
    if "filter_out_ascent" not in cfg:
        out["filter_out_ascent"] = bool(
            cfg.get(
                "filter_out_ascent_slopes",
                legacy_filter_out_negative if source_is_reversed_axis else legacy_filter_out_positive,
            )
        )
    out["filter_out_ascent_slopes"] = out["filter_out_ascent"]
    out["filter_out_descent_slopes"] = out["filter_out_descent"]
    out["filter_out_positive_slopes"] = legacy_filter_out_positive
    out["filter_out_negative_slopes"] = legacy_filter_out_negative
    return out


def _extract_source_key(algo_cfg: Dict[str, Any]) -> str:
    source_key = str(algo_cfg.get("source_channel") or algo_cfg.get("source_key") or "depth.depth")
    if "." not in source_key:
        raise ValueError(f"Invalid source channel '{source_key}', expected signal.channel")
    return source_key


def find_run_config_entry(config: Dict[str, Any], analysis_id: str) -> Tuple[str | None, Dict[str, Any]]:
    runs = dict((config or {}).get("segmentation_runs") or {})
    shared_label_groups = dict((config or {}).get("supervised_label_group_definitions") or {})
    for run_name, run_cfg in runs.items():
        if not isinstance(run_cfg, dict):
            continue
        normalized_cfg = normalize_segmentation_run_cfg(
            dict(run_cfg),
            shared_supervised_label_groups=shared_label_groups,
        )
        candidate_analysis_id = str(normalized_cfg.get("analysis_id") or run_name).strip()
        if candidate_analysis_id == str(analysis_id).strip():
            return str(run_name), normalized_cfg
    return None, {}


def default_algorithmic_cfg_from_config(config: Dict[str, Any], analysis_id: str | None = None) -> Dict[str, Any]:
    if analysis_id:
        _, run_cfg = find_run_config_entry(config, analysis_id)
        if run_cfg.get("algorithmic_segments"):
            return _normalize_algorithmic_cfg(run_cfg.get("algorithmic_segments") or {})

    workflows = dict((config or {}).get("segmentation_workflows") or {})
    find_rest = dict(workflows.get("find_rest") or {})
    runtime_params = dict(find_rest.get("runtime_params") or {})
    if runtime_params:
        return _normalize_algorithmic_cfg(runtime_params)

    runs = dict((config or {}).get("segmentation_runs") or {})
    shared_label_groups = dict((config or {}).get("supervised_label_group_definitions") or {})
    for run_cfg in runs.values():
        if not isinstance(run_cfg, dict):
            continue
        normalized_cfg = normalize_segmentation_run_cfg(
            dict(run_cfg),
            shared_supervised_label_groups=shared_label_groups,
        )
        if normalized_cfg.get("algorithmic_segments"):
            return _normalize_algorithmic_cfg(normalized_cfg.get("algorithmic_segments") or {})
    return _normalize_algorithmic_cfg({})


def build_preview_algorithmic_cfg(config: Dict[str, Any], analysis_id: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    algo_cfg = default_algorithmic_cfg_from_config(config, analysis_id=analysis_id)
    overrides = dict(overrides or {})
    if not overrides:
        return algo_cfg
    if "thresholds" in overrides and isinstance(overrides["thresholds"], dict):
        merged = dict(algo_cfg.get("thresholds") or {})
        merged.update(overrides["thresholds"])
        algo_cfg["thresholds"] = merged
    if "covariates" in overrides and isinstance(overrides["covariates"], dict):
        covariates = dict(algo_cfg.get("covariates") or {})
        for bucket in ["intrinsic", "extrinsic"]:
            if isinstance(overrides["covariates"].get(bucket), dict):
                merged_bucket = dict(covariates.get(bucket) or {})
                merged_bucket.update(overrides["covariates"][bucket])
                covariates[bucket] = merged_bucket
        algo_cfg["covariates"] = covariates
    if "debug" in overrides and isinstance(overrides["debug"], dict):
        merged = dict(algo_cfg.get("debug") or {})
        merged.update(overrides["debug"])
        algo_cfg["debug"] = merged
    for key in [
        "standardize",
        "quantize_step",
        "base_smooth_seconds",
        "coarse_smooth_seconds",
        "coarse_interval_threshold_s",
        "coarse_resolution_threshold",
        "end_segments_upon_d1_sign_change",
        "filter_out_ascent",
        "filter_out_descent",
        "filter_out_ascent_slopes",
        "filter_out_descent_slopes",
        "filter_out_positive_slopes",
        "filter_out_negative_slopes",
        "context_filters",
        "context_filter_pass_mode",
        "terminal_criteria",
    ]:
        if key in overrides:
            algo_cfg[key] = overrides[key]
    return algo_cfg


def discover_algorithmic_review_runs(data_dir: str, dataset_id: str | None = None) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for scope, root, scope_label in _run_root_candidates(data_dir, dataset_id):
        if not root.exists():
            continue
        for run_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            analysis_id = str(run_dir.name).strip()
            if not analysis_id:
                continue
            key = (scope, analysis_id)
            if key in seen:
                continue
            seen.add(key)
            segments_dir = run_dir / "segments"
            runs.append(
                {
                    "scope": scope,
                    "scope_label": scope_label,
                    "analysis_id": analysis_id,
                    "run_dir": str(run_dir),
                    "segments_dir": str(segments_dir),
                    "by_deployment_dir": str(segments_dir / "by_deployment"),
                    "merged_segments_path": str(segments_dir / "algorithmic_segments.parquet"),
                    "modified_at": pd.Timestamp(run_dir.stat().st_mtime, unit="s"),
                    "label": f"{analysis_id} ({scope_label})" if scope == "global" else f"{analysis_id} ({scope_label} dataset)",
                }
            )
    runs.sort(key=lambda row: row["modified_at"], reverse=True)
    return runs


def algorithmic_segments_path(run_info: Dict[str, Any], dataset_id: str, deployment_id: str) -> pathlib.Path:
    filename = f"{dataset_id}__{deployment_id}__algorithmic_segments.parquet"
    analysis_root = pathlib.Path(str(run_info.get("run_dir") or ""))
    candidates = [
        analysis_root / "segments" / "by_deployment" / filename,   # current pipeline layout
        analysis_root / "algorithmic_segments" / filename,          # legacy layout
        pathlib.Path(str(run_info.get("by_deployment_dir") or "")) / filename,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        cstr = str(candidate)
        if not cstr or cstr in seen:
            continue
        seen.add(cstr)
        print(f"[SEGMENTS] Looking for segment file in: {candidate}")
        if candidate.exists():
            return candidate
    # Return the preferred modern path for downstream fallback logic.
    return analysis_root / "segments" / "by_deployment" / filename


def load_algorithmic_segments(run_info: Dict[str, Any], dataset_id: str, deployment_id: str) -> pd.DataFrame:
    path = algorithmic_segments_path(run_info, dataset_id, deployment_id)
    if path.exists():
        seg_df = pd.read_parquet(path)
    else:
        # Try old-style file name
        old_name = f"{dataset_id}__{deployment_id}__algorithmic_label_timeseries.parquet"
        old_dirs = [path.parent, path.parent.parent / "by_deployment"]
        found = False
        for d in old_dirs:
            old_path = d / old_name
            if old_path.exists():
                seg_df = pd.read_parquet(old_path)
                found = True
                break
        if not found:
            merged_path = pathlib.Path(str(run_info.get("merged_segments_path") or ""))
            if not merged_path.exists():
                return pd.DataFrame()
            seg_df = pd.read_parquet(merged_path)
            if seg_df.empty:
                return seg_df
            if {"dataset_id", "deployment_id"}.issubset(seg_df.columns):
                seg_df = seg_df.loc[
                    (seg_df["dataset_id"].astype(str) == str(dataset_id))
                    & (seg_df["deployment_id"].astype(str) == str(deployment_id))
                ].copy()
    return prepare_segment_table(seg_df)


def prepare_segment_table(seg_df: pd.DataFrame) -> pd.DataFrame:
    if seg_df is None or seg_df.empty:
        return pd.DataFrame()
    out = seg_df.copy()
    for col in ["start_datetime", "end_datetime", "datetime"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if "duration_s" not in out.columns:
        if "duration" in out.columns:
            out["duration_s"] = pd.to_numeric(out["duration"], errors="coerce")
        elif {"start_datetime", "end_datetime"}.issubset(out.columns):
            out["duration_s"] = (out["end_datetime"] - out["start_datetime"]).dt.total_seconds()
        else:
            out["duration_s"] = pd.NA

    if "segment_midpoint_datetime" not in out.columns:
        if {"start_datetime", "end_datetime"}.issubset(out.columns):
            out["segment_midpoint_datetime"] = out["start_datetime"] + (out["end_datetime"] - out["start_datetime"]) / 2
        elif "datetime" in out.columns:
            out["segment_midpoint_datetime"] = out["datetime"]
        else:
            out["segment_midpoint_datetime"] = pd.NaT

    if "keep_filtered" not in out.columns:
        if "context_keep" in out.columns:
            out["keep_filtered"] = out["context_keep"]
        else:
            out["keep_filtered"] = True

    if "base_keep_filtered" not in out.columns:
        if "keep_filtered" in out.columns:
            out["base_keep_filtered"] = out["keep_filtered"]
        else:
            out["base_keep_filtered"] = True

    if "label_name" not in out.columns or out["label_name"].astype(str).eq("").all():
        if "state_name" in out.columns:
            out["label_name"] = out["state_name"].astype(str)
        elif "base_state_key" in out.columns:
            out["label_name"] = out["base_state_key"].astype(str)
        elif "nominal_class" in out.columns:
            nominal_map = {
                "surface_sleep": "putative_rest.surface_sleep",
                "long_flat": "putative_rest.benthic_sleep",
                "long_drift": "putative_rest.drift_sleep",
                "not_sleep": "find_rest.not_sleep",
            }
            out["label_name"] = out["nominal_class"].astype(str).map(nominal_map).fillna("putative_rest.unfiltered")
        else:
            out["label_name"] = "putative_rest.unfiltered"

    if "state_name" not in out.columns:
        out["state_name"] = out["label_name"]
    else:
        label_name_series = out["label_name"].astype(str).str.strip()
        state_name_series = out["state_name"].astype(str).str.strip()
        out["state_name"] = state_name_series.where(label_name_series.eq(""), label_name_series)

    if "group_label" not in out.columns:
        out["group_label"] = out["state_name"].astype(str).str.split(".", n=1).str[0]
    out["group_label"] = out["group_label"].fillna("")
    return out


def positive_state_names(seg_df: pd.DataFrame) -> List[str]:
    if seg_df is None or seg_df.empty or "state_name" not in seg_df.columns:
        return []
    # Gracefully handle missing keep_filtered column
    if "keep_filtered" in seg_df.columns:
        keep_mask = _safe_bool_series(seg_df["keep_filtered"], default=True)
    else:
        keep_mask = True  # treat all as kept if column missing
    positive = seg_df.loc[
        seg_df["state_name"].astype(str).ne("")
        & ~seg_df["state_name"].astype(str).isin(["not_sleep", "find_rest.not_sleep"])
        & keep_mask
    ]
    return sorted(positive["state_name"].dropna().astype(str).unique().tolist())


def _is_group_selector(state_name: str | None) -> bool:
    return bool(state_name) and str(state_name).endswith(".*")


def _state_selector_mask(values: pd.Series, state_name: str | None = None) -> pd.Series:
    value_series = values.astype(str)
    if not state_name:
        return pd.Series(True, index=values.index)
    selector = str(state_name)
    if selector == "__all_positive__":
        return value_series.ne("") & ~value_series.isin(["not_sleep", "find_rest.not_sleep"])
    if _is_group_selector(selector):
        prefix = selector[:-2]
        return value_series.str.startswith(f"{prefix}.")
    return value_series.eq(selector)


def filter_state_segments(
    seg_df: pd.DataFrame,
    state_name: str | None = None,
    final_kept_only: bool = False,
    positive_only: bool = False,
) -> pd.DataFrame:
    if seg_df is None or seg_df.empty:
        return pd.DataFrame()
    out = seg_df.copy()
    state_mask = _matching_state_mask(out, state_name=state_name)
    if final_kept_only:
        if "keep_filtered" in out.columns:
            keep_mask = _safe_bool_series(out["keep_filtered"], default=False)
        elif "context_keep" in out.columns:
            keep_mask = _safe_bool_series(out["context_keep"], default=False)
        else:
            keep_mask = pd.Series(True, index=out.index, dtype=bool)
        state_mask &= keep_mask
    if positive_only:
        state_names = out["state_name"].astype(str) if "state_name" in out.columns else pd.Series("", index=out.index)
        state_mask &= state_names.ne("") & ~state_names.isin(["not_sleep", "find_rest.not_sleep"])
    out = out.loc[state_mask].copy()
    return out.sort_values(["segment_midpoint_datetime", "start_datetime"], na_position="last").reset_index(drop=True)


def choose_default_candidate(
    seg_df: pd.DataFrame,
    state_name: str | None = None,
    candidate_rank: int = DEFAULT_CANDIDATE_RANK,
) -> Tuple[pd.DataFrame, int]:
    candidates = filter_state_segments(
        seg_df,
        state_name=state_name,
        final_kept_only=True,
        positive_only=True,
    )
    if candidates.empty:
        return candidates, 0
    idx = min(max(int(candidate_rank), 1), len(candidates)) - 1
    return candidates, idx


def centered_window_bounds(seg_row: pd.Series | Dict[str, Any], window_hours: float = DEFAULT_WINDOW_HOURS) -> Tuple[pd.Timestamp, pd.Timestamp]:
    row = seg_row if isinstance(seg_row, pd.Series) else pd.Series(seg_row or {})
    midpoint = pd.to_datetime(row.get("segment_midpoint_datetime"), errors="coerce")
    if pd.isna(midpoint):
        start_dt = pd.to_datetime(row.get("start_datetime"), errors="coerce")
        end_dt = pd.to_datetime(row.get("end_datetime"), errors="coerce")
        if pd.notna(start_dt) and pd.notna(end_dt):
            midpoint = start_dt + (end_dt - start_dt) / 2
        else:
            midpoint = start_dt if pd.notna(start_dt) else end_dt
    half = pd.to_timedelta(float(window_hours) / 2.0, unit="h")
    return midpoint - half, midpoint + half


def _matching_state_mask(seg_df: pd.DataFrame, state_name: str | None = None) -> pd.Series:
    mask = pd.Series(True, index=seg_df.index)
    if state_name:
        mask &= _state_selector_mask(seg_df["state_name"], state_name)
    return mask


def selection_options_for_positive_states(state_names: List[str]) -> List[str]:
    options: List[str] = []
    grouped: Dict[str, List[str]] = {}
    for name in state_names:
        if "." in name:
            prefix = name.rsplit(".", 1)[0]
            grouped.setdefault(prefix, []).append(name)
    for prefix, members in sorted(grouped.items()):
        if len(members) > 1:
            options.append(f"{prefix}.*")
    options.extend(state_names)
    return options


def selector_display_name(selector: str) -> str:
    if str(selector).endswith(".*"):
        return f"{selector[:-2]} (all subtypes)"
    return str(selector)


def summarize_state_filters(
    seg_df: pd.DataFrame,
    state_name: str | None = None,
    ordered_filter_ids: Iterable[str] | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if seg_df is None or seg_df.empty:
        empty = {"base_pass_count": 0, "final_kept_count": 0, "context_rejected_count": 0, "reject_rate_pct": 0.0}
        return empty, pd.DataFrame(), pd.DataFrame()

    work = seg_df.copy()
    target_mask = _matching_state_mask(work, state_name=state_name)

    if "base_keep_filtered" in work.columns:
        base_keep = _safe_bool_series(work["base_keep_filtered"], default=False)
    elif "keep_filtered" in work.columns:
        base_keep = _safe_bool_series(work["keep_filtered"], default=False)
    else:
        base_keep = pd.Series(True, index=work.index, dtype=bool)

    if "keep_filtered" in work.columns:
        final_keep = _safe_bool_series(work["keep_filtered"], default=False)
    elif "context_keep" in work.columns:
        final_keep = _safe_bool_series(work["context_keep"], default=False)
    else:
        final_keep = base_keep.copy()

    if "context_keep" in work.columns:
        context_keep = _safe_bool_series(work["context_keep"], default=True)
    else:
        context_keep = final_keep.copy()

    base_mask = base_keep & target_mask
    final_mask = final_keep & target_mask
    rejected_mask = base_mask & (~context_keep)

    base_count = int(base_mask.sum())
    final_count = int(final_mask.sum())
    rejected_count = int(rejected_mask.sum())
    reject_rate = float((100.0 * rejected_count / base_count) if base_count else 0.0)

    reasons = (
        work.loc[rejected_mask, "context_reject_reason"]
        .dropna()
        .astype(str)
        .map(_split_pipe_values)
    )
    reason_counts: Dict[str, int] = {}
    for parts in reasons:
        for part in parts:
            reason_counts[part] = reason_counts.get(part, 0) + 1
    reason_df = (
        pd.DataFrame([{"reject_reason": key, "count": val} for key, val in reason_counts.items()])
        .sort_values(["count", "reject_reason"], ascending=[False, True])
        .reset_index(drop=True)
        if reason_counts
        else pd.DataFrame(columns=["reject_reason", "count"])
    )

    filter_ids = [str(fid) for fid in (ordered_filter_ids or []) if str(fid).strip()]
    if not filter_ids:
        filter_ids = []
        for col in work.columns:
            if col.endswith("_pass"):
                filter_ids.append(col[: -len("_pass")])
    filter_rows: List[Dict[str, Any]] = []
    for filter_id in filter_ids:
        pass_col = f"{filter_id}_pass"
        value_col = f"{filter_id}_value"
        if pass_col not in work.columns:
            continue
        applied_mask = target_mask & work[pass_col].notna()
        if not bool(applied_mask.any()):
            continue
        pass_values = work[pass_col].astype("boolean")
        failed_mask = applied_mask & (~pass_values.fillna(False))
        applied_count = int(applied_mask.sum())
        failed_count = int(failed_mask.sum())
        filter_rows.append(
            {
                "filter_id": filter_id,
                "filter_stage": "inferred" if "inferred" in str(filter_id).lower() or "buoyancy" in str(filter_id).lower() else "measured",
                "applied_count": applied_count,
                "failed_count": failed_count,
                "fail_rate_pct": float((100.0 * failed_count / applied_count) if applied_count else 0.0),
                "median_value": float(pd.to_numeric(work.loc[applied_mask, value_col], errors="coerce").median())
                if value_col in work.columns
                else float("nan"),
            }
        )
    filter_df = (
        pd.DataFrame(filter_rows)
        .sort_values(["failed_count", "applied_count", "filter_id"], ascending=[False, False, True])
        .reset_index(drop=True)
        if filter_rows
        else pd.DataFrame(columns=["filter_id", "filter_stage", "applied_count", "failed_count", "fail_rate_pct", "median_value"])
    )

    summary = {
        "base_pass_count": base_count,
        "final_kept_count": final_count,
        "context_rejected_count": rejected_count,
        "reject_rate_pct": reject_rate,
    }
    return summary, reason_df, filter_df


def candidate_filter_details(seg_row: pd.Series | Dict[str, Any], ordered_filter_ids: Iterable[str] | None = None) -> pd.DataFrame:
    row = seg_row if isinstance(seg_row, pd.Series) else pd.Series(seg_row or {})
    filter_ids = [str(fid) for fid in (ordered_filter_ids or []) if str(fid).strip()]
    if not filter_ids:
        filter_ids = [str(col[: -len("_pass")]) for col in row.index if str(col).endswith("_pass")]
    rows = []
    for filter_id in filter_ids:
        pass_col = f"{filter_id}_pass"
        value_col = f"{filter_id}_value"
        if pass_col not in row.index:
            continue
        pass_value = row.get(pass_col)
        if pd.isna(pass_value):
            continue
        rows.append(
            {
                "filter_id": filter_id,
                "filter_stage": "inferred" if "inferred" in str(filter_id).lower() or "buoyancy" in str(filter_id).lower() else "measured",
                "passed": bool(pass_value),
                "value": pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0] if value_col in row.index else pd.NA,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["filter_id", "filter_stage", "passed", "value"])
    return pd.DataFrame(rows)


def build_review_yaml_snippet(config: Dict[str, Any], analysis_id: str) -> str:
    run_name, run_cfg = find_run_config_entry(config, analysis_id)
    algo_cfg = default_algorithmic_cfg_from_config(config, analysis_id=analysis_id)
    payload: Dict[str, Any] = {"segmentation_runs": {run_name or analysis_id: {}}}
    run_section = payload["segmentation_runs"][run_name or analysis_id]
    run_section["scope"] = {"analysis_id": analysis_id}

    context_defs = dict(run_cfg.get("context_filter_definitions") or {})
    if context_defs:
        run_section.setdefault("algorithmic", {})["context_filter_definitions"] = context_defs

    debug_cfg = dict(algo_cfg.get("debug") or {})
    debug_cfg["write_context_review_table"] = True
    run_section.setdefault("algorithmic", {})["segments"] = {
        "thresholds": dict(algo_cfg.get("thresholds") or {}),
        "covariates": dict(algo_cfg.get("covariates") or {}),
        "context_filters": list(algo_cfg.get("context_filters") or []),
        "context_filter_pass_mode": str(algo_cfg.get("context_filter_pass_mode") or "measured_only"),
        "terminal_criteria": dict(algo_cfg.get("terminal_criteria") or {}),
        "debug": debug_cfg,
    }
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def _non_context_algorithmic_section_from_algo_cfg(algo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    debug_cfg = dict(algo_cfg.get("debug") or {})
    payload: Dict[str, Any] = {
        "thresholds": dict(algo_cfg.get("thresholds") or {}),
        "terminal_criteria": dict(algo_cfg.get("terminal_criteria") or {}),
        "standardize": bool(algo_cfg.get("standardize", True)),
        "quantize_step": algo_cfg.get("quantize_step"),
        "base_smooth_seconds": algo_cfg.get("base_smooth_seconds"),
        "coarse_smooth_seconds": algo_cfg.get("coarse_smooth_seconds"),
        "coarse_interval_threshold_s": algo_cfg.get("coarse_interval_threshold_s"),
        "coarse_resolution_threshold": algo_cfg.get("coarse_resolution_threshold"),
        "end_segments_upon_d1_sign_change": bool(algo_cfg.get("end_segments_upon_d1_sign_change", True)),
        "filter_out_ascent": bool(algo_cfg.get("filter_out_ascent", True)),
        "filter_out_descent": bool(algo_cfg.get("filter_out_descent", False)),
        "context_filter_pass_mode": str(algo_cfg.get("context_filter_pass_mode") or "measured_only"),
    }
    if bool(debug_cfg.get("write_context_review_table", False)):
        payload["debug"] = {"write_context_review_table": True}
    return payload


def build_non_context_algorithmic_yaml_snippet_from_algo_cfg(
    config: Dict[str, Any],
    analysis_id: str,
    algo_cfg: Dict[str, Any],
) -> str:
    run_name, _ = find_run_config_entry(config, analysis_id)
    payload: Dict[str, Any] = {"segmentation_runs": {run_name or analysis_id: {}}}
    payload["segmentation_runs"][run_name or analysis_id]["scope"] = {"analysis_id": analysis_id}
    payload["segmentation_runs"][run_name or analysis_id]["algorithmic"] = {
        "segments": _non_context_algorithmic_section_from_algo_cfg(algo_cfg)
    }
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def build_review_yaml_snippet_from_algo_cfg(
    config: Dict[str, Any],
    analysis_id: str,
    algo_cfg: Dict[str, Any],
    context_filter_definitions: Dict[str, Any] | None = None,
) -> str:
    run_name, run_cfg = find_run_config_entry(config, analysis_id)
    payload: Dict[str, Any] = {"segmentation_runs": {run_name or analysis_id: {}}}
    run_section = payload["segmentation_runs"][run_name or analysis_id]
    run_section["scope"] = {"analysis_id": analysis_id}
    context_defs = (
        copy.deepcopy(dict(context_filter_definitions or {}))
        if context_filter_definitions is not None
        else dict(run_cfg.get("context_filter_definitions") or {})
    )
    context_filter_refs, context_defs = _compact_context_filters_and_definitions(algo_cfg, context_defs)
    if context_defs:
        run_section.setdefault("algorithmic", {})["context_filter_definitions"] = context_defs
    debug_cfg = dict(algo_cfg.get("debug") or {})
    debug_cfg["write_context_review_table"] = True
    run_section.setdefault("algorithmic", {})["segments"] = {
        "thresholds": dict(algo_cfg.get("thresholds") or {}),
        "covariates": dict(algo_cfg.get("covariates") or {}),
        "context_filters": context_filter_refs,
        "context_filter_pass_mode": str(algo_cfg.get("context_filter_pass_mode") or "measured_only"),
        "terminal_criteria": dict(algo_cfg.get("terminal_criteria") or {}),
        "debug": debug_cfg,
        "standardize": bool(algo_cfg.get("standardize", True)),
        "quantize_step": algo_cfg.get("quantize_step"),
        "base_smooth_seconds": algo_cfg.get("base_smooth_seconds"),
        "coarse_smooth_seconds": algo_cfg.get("coarse_smooth_seconds"),
        "coarse_interval_threshold_s": algo_cfg.get("coarse_interval_threshold_s"),
        "coarse_resolution_threshold": algo_cfg.get("coarse_resolution_threshold"),
        "end_segments_upon_d1_sign_change": bool(algo_cfg.get("end_segments_upon_d1_sign_change", True)),
        "filter_out_ascent": bool(algo_cfg.get("filter_out_ascent", True)),
        "filter_out_descent": bool(algo_cfg.get("filter_out_descent", False)),
    }
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def _compact_context_filters_and_definitions(
    algo_cfg: Dict[str, Any],
    context_filter_definitions: Dict[str, Any] | None = None,
) -> tuple[List[str], Dict[str, Any]]:
    defs = copy.deepcopy(dict(context_filter_definitions or {}))
    refs: List[str] = []
    for idx, raw in enumerate(list(algo_cfg.get("context_filters") or [])):
        if isinstance(raw, str):
            filter_id = str(raw).strip()
            if not filter_id:
                continue
            if filter_id not in defs:
                defs[filter_id] = {"id": filter_id}
            if filter_id not in refs:
                refs.append(filter_id)
            continue
        if isinstance(raw, dict):
            filter_cfg = copy.deepcopy(raw)
            filter_id = str(filter_cfg.get("id") or f"context_filter_{idx + 1}").strip()
            if not filter_id:
                continue
            filter_cfg["id"] = filter_id
            defs[filter_id] = filter_cfg
            if filter_id not in refs:
                refs.append(filter_id)
    return refs, defs


def _full_algorithmic_section_from_algo_cfg(
    algo_cfg: Dict[str, Any],
    context_filter_definitions: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context_filter_refs, _ = _compact_context_filters_and_definitions(
        algo_cfg,
        context_filter_definitions=context_filter_definitions,
    )
    return {
        "thresholds": dict(algo_cfg.get("thresholds") or {}),
        "covariates": dict(algo_cfg.get("covariates") or {}),
        "context_filters": context_filter_refs,
        "context_filter_pass_mode": str(algo_cfg.get("context_filter_pass_mode") or "measured_only"),
        "terminal_criteria": dict(algo_cfg.get("terminal_criteria") or {}),
        "debug": dict(algo_cfg.get("debug") or {}),
        "standardize": bool(algo_cfg.get("standardize", True)),
        "quantize_step": algo_cfg.get("quantize_step"),
        "base_smooth_seconds": algo_cfg.get("base_smooth_seconds"),
        "coarse_smooth_seconds": algo_cfg.get("coarse_smooth_seconds"),
        "coarse_interval_threshold_s": algo_cfg.get("coarse_interval_threshold_s"),
        "coarse_resolution_threshold": algo_cfg.get("coarse_resolution_threshold"),
        "end_segments_upon_d1_sign_change": bool(algo_cfg.get("end_segments_upon_d1_sign_change", True)),
        "filter_out_ascent": bool(algo_cfg.get("filter_out_ascent", True)),
        "filter_out_descent": bool(algo_cfg.get("filter_out_descent", False)),
    }


def _collect_changed_paths(before: Any, after: Any, prefix: str = "") -> List[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changed: List[str] = []
        for key in sorted(set(before.keys()) | set(after.keys()), key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.append(child_prefix)
                continue
            changed.extend(_collect_changed_paths(before[key], after[key], child_prefix))
        return changed
    if before != after:
        return [prefix] if prefix else ["algorithmic_segments"]
    return []


def _ensure_run_scope_section(run_section: Dict[str, Any], analysis_id: str) -> None:
    scope_cfg = run_section.get("scope") if isinstance(run_section.get("scope"), dict) else {}
    if not scope_cfg:
        scope_cfg = {"analysis_id": analysis_id}
    else:
        scope_cfg.setdefault("analysis_id", analysis_id)
    run_section["scope"] = scope_cfg
    run_section.pop("analysis_id", None)


def _get_run_algorithmic_segments(run_section: Dict[str, Any]) -> Dict[str, Any]:
    algo_block = run_section.get("algorithmic") if isinstance(run_section.get("algorithmic"), dict) else {}
    if isinstance(algo_block.get("segments"), dict):
        return copy.deepcopy(algo_block.get("segments") or {})
    return copy.deepcopy(run_section.get("algorithmic_segments") or {})


def _set_run_algorithmic_segments(run_section: Dict[str, Any], algo_segments: Dict[str, Any]) -> None:
    algo_block = run_section.get("algorithmic") if isinstance(run_section.get("algorithmic"), dict) else {}
    algo_block["segments"] = copy.deepcopy(algo_segments or {})
    run_section["algorithmic"] = algo_block
    run_section.pop("algorithmic_segments", None)


def _get_run_context_filter_defs(run_section: Dict[str, Any]) -> Dict[str, Any]:
    algo_block = run_section.get("algorithmic") if isinstance(run_section.get("algorithmic"), dict) else {}
    if isinstance(algo_block.get("context_filter_definitions"), dict):
        return copy.deepcopy(algo_block.get("context_filter_definitions") or {})
    return copy.deepcopy(run_section.get("context_filter_definitions") or {})


def _set_run_context_filter_defs(run_section: Dict[str, Any], context_filter_definitions: Dict[str, Any]) -> None:
    algo_block = run_section.get("algorithmic") if isinstance(run_section.get("algorithmic"), dict) else {}
    if context_filter_definitions:
        algo_block["context_filter_definitions"] = copy.deepcopy(context_filter_definitions)
    else:
        algo_block.pop("context_filter_definitions", None)
    if algo_block:
        run_section["algorithmic"] = algo_block
    run_section.pop("context_filter_definitions", None)


def review_non_context_algorithmic_config_update(
    config_path: str,
    analysis_id: str,
    algo_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    config_payload, _, _ = load_combined_config(config_path=config_path)
    runs_path = resolve_segmentation_runs_path(config_path)

    runs_payload = {"segmentation_runs": copy.deepcopy(config_payload.get("segmentation_runs") or {})}
    if runs_path.exists():
        with open(runs_path, "r") as handle:
            runs_payload = yaml.safe_load(handle) or {}
    runs = runs_payload.setdefault("segmentation_runs", {})
    run_name, _ = find_run_config_entry(config_payload, analysis_id)
    if not run_name:
        run_name = str(analysis_id).strip() or "segmentation_run"
        runs.setdefault(run_name, {"scope": {"analysis_id": analysis_id}})

    run_section = runs.setdefault(run_name, {})
    _ensure_run_scope_section(run_section, analysis_id)
    existing_algo = _get_run_algorithmic_segments(run_section)
    updated_algo = copy.deepcopy(existing_algo)
    non_context_payload = _non_context_algorithmic_section_from_algo_cfg(algo_cfg)

    for key, value in non_context_payload.items():
        if key == "debug":
            debug_section = dict(updated_algo.get("debug") or {})
            debug_section.update(dict(value or {}))
            updated_algo["debug"] = debug_section
        else:
            updated_algo[key] = copy.deepcopy(value)

    _set_run_algorithmic_segments(run_section, updated_algo)
    changed_paths = _collect_changed_paths(existing_algo, updated_algo, prefix="algorithmic_segments")

    updated_runtime_config = copy.deepcopy(config_payload)
    updated_runtime_config["segmentation_runs"] = copy.deepcopy(runs)

    return {
        "config_path": os.path.abspath(str(runs_path)),
        "analysis_id": analysis_id,
        "run_name": run_name,
        "target_path": f"segmentation_runs.{run_name}.algorithmic.segments",
        "changed_paths": changed_paths,
        "existing_algorithmic_segments": existing_algo,
        "updated_algorithmic_segments": updated_algo,
        "updated_config": updated_runtime_config,
        "updated_runs_config": runs_payload,
        "yaml_snippet": build_non_context_algorithmic_yaml_snippet_from_algo_cfg(
            updated_runtime_config,
            analysis_id,
            algo_cfg,
        ),
    }


def save_non_context_algorithmic_config_update(
    config_path: str,
    analysis_id: str,
    algo_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    review_payload = review_non_context_algorithmic_config_update(config_path, analysis_id, algo_cfg)
    with open(review_payload["config_path"], "w") as handle:
        yaml.safe_dump(review_payload["updated_runs_config"], handle, sort_keys=False, default_flow_style=False)
    return review_payload


def review_full_algorithmic_config_update(
    config_path: str,
    analysis_id: str,
    algo_cfg: Dict[str, Any],
    context_filter_definitions: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    config_payload, _, _ = load_combined_config(config_path=config_path)
    runs_path = resolve_segmentation_runs_path(config_path)

    runs_payload = {"segmentation_runs": copy.deepcopy(config_payload.get("segmentation_runs") or {})}
    if runs_path.exists():
        with open(runs_path, "r") as handle:
            runs_payload = yaml.safe_load(handle) or {}
    runs = runs_payload.setdefault("segmentation_runs", {})
    run_name, run_cfg = find_run_config_entry(config_payload, analysis_id)
    if not run_name:
        run_name = str(analysis_id).strip() or "segmentation_run"
        runs.setdefault(run_name, {"scope": {"analysis_id": analysis_id}})

    run_section = runs.setdefault(run_name, {})
    _ensure_run_scope_section(run_section, analysis_id)
    existing_algo = _get_run_algorithmic_segments(run_section)
    existing_context_defs = _get_run_context_filter_defs(run_section)
    updated_algo = copy.deepcopy(existing_algo)
    merged_context_defs_seed = copy.deepcopy(existing_context_defs)
    if context_filter_definitions is not None:
        merged_context_defs_seed = copy.deepcopy(dict(context_filter_definitions or {}))
    full_payload = _full_algorithmic_section_from_algo_cfg(
        algo_cfg,
        context_filter_definitions=merged_context_defs_seed,
    )
    updated_algo.update(copy.deepcopy(full_payload))
    _set_run_algorithmic_segments(run_section, updated_algo)

    updated_context_defs = copy.deepcopy(existing_context_defs)
    compact_filter_refs, compact_context_defs = _compact_context_filters_and_definitions(
        algo_cfg,
        context_filter_definitions=merged_context_defs_seed,
    )
    updated_algo["context_filters"] = compact_filter_refs
    _set_run_algorithmic_segments(run_section, updated_algo)
    if context_filter_definitions is not None or compact_context_defs:
        updated_context_defs = compact_context_defs
        _set_run_context_filter_defs(run_section, updated_context_defs)

    changed_paths = _collect_changed_paths(existing_algo, updated_algo, prefix="algorithmic_segments")
    changed_paths.extend(
        _collect_changed_paths(
            existing_context_defs,
            updated_context_defs,
            prefix="algorithmic.context_filter_definitions",
        )
    )

    updated_runtime_config = copy.deepcopy(config_payload)
    updated_runtime_config["segmentation_runs"] = copy.deepcopy(runs)
    yaml_snippet = build_review_yaml_snippet_from_algo_cfg(
        updated_runtime_config,
        analysis_id,
        updated_algo,
        context_filter_definitions=updated_context_defs,
    )
    if context_filter_definitions is not None:
        payload_for_snippet: Dict[str, Any] = {"segmentation_runs": {run_name: {}}}
        payload_run = payload_for_snippet["segmentation_runs"][run_name]
        payload_run["scope"] = {"analysis_id": analysis_id}
        payload_run["algorithmic"] = {"segments": copy.deepcopy(updated_algo)}
        if updated_context_defs:
            payload_run["algorithmic"]["context_filter_definitions"] = copy.deepcopy(updated_context_defs)
        yaml_snippet = yaml.safe_dump(payload_for_snippet, sort_keys=False, default_flow_style=False)

    return {
        "config_path": os.path.abspath(str(runs_path)),
        "analysis_id": analysis_id,
        "run_name": run_name,
        "target_path": f"segmentation_runs.{run_name}.algorithmic.segments + algorithmic.context_filter_definitions",
        "changed_paths": changed_paths,
        "existing_algorithmic_segments": existing_algo,
        "updated_algorithmic_segments": updated_algo,
        "existing_context_filter_definitions": existing_context_defs,
        "updated_context_filter_definitions": updated_context_defs,
        "updated_config": updated_runtime_config,
        "updated_runs_config": runs_payload,
        "yaml_snippet": yaml_snippet,
    }


def save_full_algorithmic_config_update(
    config_path: str,
    analysis_id: str,
    algo_cfg: Dict[str, Any],
    context_filter_definitions: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    review_payload = review_full_algorithmic_config_update(
        config_path=config_path,
        analysis_id=analysis_id,
        algo_cfg=algo_cfg,
        context_filter_definitions=context_filter_definitions,
    )
    with open(review_payload["config_path"], "w") as handle:
        yaml.safe_dump(review_payload["updated_runs_config"], handle, sort_keys=False, default_flow_style=False)
    return review_payload


def default_review_signals(config: Dict[str, Any], analysis_id: str, signal_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:
    available = set((signal_data or {}).keys())
    algo_cfg = default_algorithmic_cfg_from_config(config, analysis_id=analysis_id)
    method_name = str(algo_cfg.get("method_name") or "find_rest")
    workflows = dict((config or {}).get("segmentation_workflows") or {})
    workflow_cfg = dict(workflows.get(method_name) or {})

    requested = [str(sig) for sig in (workflow_cfg.get("default_review_signals") or []) if str(sig).strip()]
    source_channel = str(algo_cfg.get("source_channel") or "depth.depth")
    if "." in source_channel:
        source_signal = source_channel.split(".", 1)[0]
        if source_signal not in requested:
            requested.append(source_signal)

    signals: List[str] = []
    channels: Dict[str, List[str]] = {}

    preferred_signal_order = [
        "depth",
        "pressure",
        "algorithmic_d1_channel",
        "algorithmic_d2_channel",
        "algorithmic_buoyancy_channel",
        "stroke_rate",
        "algorithmic_feature_channels",
        "algorithmic_intermediate_channels",
    ]

    if "algorithmic_d1_channel" in available:
        d1_df = signal_data.get("algorithmic_d1_channel")
        if isinstance(d1_df, pd.DataFrame):
            channels["algorithmic_d1_channel"] = [col for col in ["depth_d1_ms"] if col in d1_df.columns]

    if "algorithmic_d2_channel" in available:
        d2_df = signal_data.get("algorithmic_d2_channel")
        if isinstance(d2_df, pd.DataFrame):
            channels["algorithmic_d2_channel"] = [col for col in ["depth_d2_ms2"] if col in d2_df.columns]

    if "algorithmic_derivative_channels" in available:
        derivative_df = signal_data.get("algorithmic_derivative_channels")
        preferred_cols = ["depth_d1_ms", "depth_d2_ms2", "is_unfiltered_rest_candidate"]
        if isinstance(derivative_df, pd.DataFrame):
            channels["algorithmic_derivative_channels"] = [col for col in preferred_cols if col in derivative_df.columns]

    if "algorithmic_buoyancy_channel" in available:
        buoyancy_df = signal_data.get("algorithmic_buoyancy_channel")
        preferred_cols = ["inferred_buoyancy_value", "drift_rate_ms"]
        if isinstance(buoyancy_df, pd.DataFrame):
            channels["algorithmic_buoyancy_channel"] = [col for col in preferred_cols if col in buoyancy_df.columns]

    if "algorithmic_feature_channels" in available:
        feature_df = signal_data.get("algorithmic_feature_channels")
        preferred_cols = [
            "depth_std_m",
            "inferred_buoyancy_value",
            "inferred_buoyancy_early_reference_value",
            "inferred_buoyancy_late_reference_value",
            "segment_mean_stroke_rate_spm",
            "is_unfiltered_rest_candidate",
            "is_signed_rest_candidate",
            "is_drift_candidate",
            "is_surface_sleep_candidate",
        ]
        if isinstance(feature_df, pd.DataFrame):
            channels["algorithmic_feature_channels"] = [col for col in preferred_cols if col in feature_df.columns]

    if "algorithmic_intermediate_channels" in available:
        preferred_cols = [
            "depth_std_m",
            "depth_d1_ms",
            "depth_d2_ms2",
            "is_unfiltered_rest_candidate",
            "is_signed_rest_candidate",
            "is_drift_candidate",
            "is_surface_sleep_candidate",
        ]
        intermediate_df = signal_data.get("algorithmic_intermediate_channels")
        if isinstance(intermediate_df, pd.DataFrame):
            channels["algorithmic_intermediate_channels"] = [col for col in preferred_cols if col in intermediate_df.columns]

    preferred_seen = set()
    for sig in preferred_signal_order:
        if sig in available:
            signals.append(sig)
            preferred_seen.add(sig)

    for sig in requested:
        if sig in available and sig not in preferred_seen:
            signals.append(sig)
            preferred_seen.add(sig)

    for sig in sorted(available):
        if sig not in preferred_seen:
            signals.append(sig)
            preferred_seen.add(sig)

    deduped = []
    seen = set()
    for sig in signals:
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(sig)
    return deduped, channels


def build_overlay_event_rows(
    seg_df: pd.DataFrame,
    state_name: str | None = None,
    include_rejected_base: bool = False,
    focused_segment_rank: int | None = None,
) -> pd.DataFrame:
    if seg_df is None or seg_df.empty:
        return pd.DataFrame(columns=["datetime", "end_datetime", "duration", "key"])

    rows: List[Dict[str, Any]] = []
    positive = filter_state_segments(seg_df, state_name=state_name, positive_only=True)
    base_pass_mask = _safe_bool_series(positive["base_keep_filtered"], default=False)
    kept_mask = _safe_bool_series(positive["keep_filtered"], default=False)
    context_keep_mask = _safe_bool_series(positive["context_keep"], default=True)
    base_pass = positive.loc[base_pass_mask].copy()
    kept = positive.loc[kept_mask].copy()
    rejected = positive.loc[
        base_pass_mask & (~context_keep_mask)
    ].copy()

    for _, row in base_pass.iterrows():
        rows.append(
            {
                "datetime": row.get("start_datetime"),
                "end_datetime": row.get("end_datetime"),
                "duration": row.get("duration_s"),
                "key": "__review_pre_context__",
                "segment_rank": row.get("segment_rank"),
                "state_name": str(row.get("state_name") or ""),
            }
        )

    for _, row in kept.iterrows():
        state_label = _canonical_review_rest_state_name(row.get("state_name"))
        key = f"__review_final_kept__::{state_label}" if state_label else "__review_final_kept__"
        rows.append(
            {
                "datetime": row.get("start_datetime"),
                "end_datetime": row.get("end_datetime"),
                "duration": row.get("duration_s"),
                "key": key,
                "segment_rank": row.get("segment_rank"),
                "state_name": state_label,
            }
        )
    if include_rejected_base:
        for _, row in rejected.iterrows():
            rows.append(
                {
                    "datetime": row.get("start_datetime"),
                    "end_datetime": row.get("end_datetime"),
                    "duration": row.get("duration_s"),
                    "key": "__review_rejected__",
                    "segment_rank": row.get("segment_rank"),
                    "state_name": _canonical_review_rest_state_name(row.get("state_name")),
                }
            )
    if focused_segment_rank is not None:
        focus_rows = positive.loc[pd.to_numeric(positive.get("segment_rank"), errors="coerce") == int(focused_segment_rank)]
        for _, row in focus_rows.iterrows():
            rows.append(
                {
                    "datetime": row.get("start_datetime"),
                    "end_datetime": row.get("end_datetime"),
                    "duration": row.get("duration_s"),
                    "key": "__review_focus__",
                    "segment_rank": row.get("segment_rank"),
                    "state_name": _canonical_review_rest_state_name(row.get("state_name")),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["datetime", "end_datetime", "duration", "key"])
    event_df = pd.DataFrame(rows)
    event_df["datetime"] = pd.to_datetime(event_df["datetime"], errors="coerce")
    event_df["end_datetime"] = pd.to_datetime(event_df["end_datetime"], errors="coerce")
    return event_df.sort_values(["datetime", "key"]).reset_index(drop=True)


def build_mask_period_rows(
    signal_df: pd.DataFrame | None,
    mask_col: str,
    key: str,
) -> pd.DataFrame:
    if not isinstance(signal_df, pd.DataFrame) or signal_df.empty or "datetime" not in signal_df.columns or mask_col not in signal_df.columns:
        return pd.DataFrame(columns=["datetime", "end_datetime", "duration", "key"])
    work = signal_df[["datetime", mask_col]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["datetime", "end_datetime", "duration", "key"])
    mask = _safe_bool_series(work[mask_col], default=False)
    if not mask.any():
        return pd.DataFrame(columns=["datetime", "end_datetime", "duration", "key"])
    group_ids = (mask != mask.shift(fill_value=False)).cumsum()
    rows: List[Dict[str, Any]] = []
    for _, grp in work.loc[mask].groupby(group_ids[mask], sort=False):
        start_dt = pd.to_datetime(grp["datetime"].iloc[0], errors="coerce")
        end_dt = pd.to_datetime(grp["datetime"].iloc[-1], errors="coerce")
        if len(grp) > 1:
            diffs = grp["datetime"].diff().dropna()
            median_step = diffs.median() if not diffs.empty else pd.Timedelta(seconds=0)
        else:
            median_step = pd.Timedelta(seconds=0)
        end_dt = end_dt + (median_step if pd.notna(median_step) else pd.Timedelta(seconds=0))
        rows.append(
            {
                "datetime": start_dt,
                "end_datetime": end_dt,
                "duration": float((end_dt - start_dt).total_seconds()),
                "key": key,
            }
        )
    return pd.DataFrame(rows)


def stage_dropoff_summary(
    seg_df: pd.DataFrame,
    state_name: str | None = None,
    ordered_filter_ids: Iterable[str] | None = None,
    signal_data: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    if seg_df is None or seg_df.empty:
        return pd.DataFrame(columns=["stage", "remaining_before", "dropped_here", "remaining_after", "applied_count"])

    rows: List[Dict[str, Any]] = []
    mask_signal_df = None
    if isinstance(signal_data, dict):
        candidate_signal = signal_data.get("algorithmic_feature_channels")
        if not isinstance(candidate_signal, pd.DataFrame):
            candidate_signal = signal_data.get("algorithmic_intermediate_channels")
        if isinstance(candidate_signal, pd.DataFrame):
            mask_signal_df = candidate_signal
    if isinstance(mask_signal_df, pd.DataFrame) and "is_unfiltered_rest_candidate" in mask_signal_df.columns:
        unfiltered_rows = build_mask_period_rows(mask_signal_df, "is_unfiltered_rest_candidate", "__mask__")
        unfiltered_count = int(len(unfiltered_rows))
        rows.append(
            {
                "stage": "1) unfiltered_rest_candidate",
                "remaining_before": unfiltered_count,
                "dropped_here": 0,
                "remaining_after": unfiltered_count,
                "applied_count": unfiltered_count,
            }
        )

    work = seg_df.copy()
    work = work.loc[work.get("segment_family", pd.Series("", index=work.index)).astype(str) == "drift_candidate"].copy()
    if work.empty:
        return pd.DataFrame(columns=["stage", "remaining_before", "dropped_here", "remaining_after", "applied_count"])

    if state_name and not _is_group_selector(state_name) and str(state_name) != "__all_positive__":
        target_state = str(state_name)
        if (
            "find_rest.long_flat" in target_state
            or target_state.endswith("long_flat")
            or target_state.endswith("benthic_sleep")
        ):
            work = work.loc[work.get("base_nominal_class", pd.Series("", index=work.index)).astype(str) == "long_flat"].copy()
        elif (
            "find_rest.long_drift" in target_state
            or target_state.endswith("long_drift")
            or target_state.endswith("drift_sleep")
        ):
            work = work.loc[work.get("base_nominal_class", pd.Series("", index=work.index)).astype(str) == "long_drift"].copy()
        elif target_state.endswith("surface_sleep"):
            work = work.iloc[0:0].copy()
        elif "state_name" in work.columns:
            work = work.loc[work["state_name"].astype(str) == target_state].copy()
    if work.empty:
        return pd.DataFrame(columns=["stage", "remaining_before", "dropped_here", "remaining_after", "applied_count"])

    total_initial = int(len(work))
    base_pass_mask = _safe_bool_series(work["base_keep_filtered"], default=False)
    base_remaining = work.loc[base_pass_mask].index
    rows.append(
        {
            "stage": "2) filtered_rest_candidate",
            "remaining_before": total_initial,
            "dropped_here": 0,
            "remaining_after": total_initial,
            "applied_count": total_initial,
        }
    )
    rows.append(
        {
            "stage": "3) unfiltered_putative_rest",
            "remaining_before": total_initial,
            "dropped_here": int(total_initial - len(base_remaining)),
            "remaining_after": int(len(base_remaining)),
            "applied_count": total_initial,
        }
    )
    reject_stage = work.get("context_reject_stage", pd.Series(pd.NA, index=work.index)).astype(str)
    measured_dropped = int((base_pass_mask & reject_stage.eq("rejected_after_measured_filters")).sum())
    inferred_dropped = int((base_pass_mask & reject_stage.eq("rejected_after_inferred_filters")).sum())
    remaining_after_measured = int(len(base_remaining) - measured_dropped)
    rows.append(
        {
            "stage": "4) rejected_after_measured_filters",
            "remaining_before": int(len(base_remaining)),
            "dropped_here": measured_dropped,
            "remaining_after": remaining_after_measured,
            "applied_count": int(len(base_remaining)),
        }
    )
    rows.append(
        {
            "stage": "5) rejected_after_inferred_filters",
            "remaining_before": remaining_after_measured,
            "dropped_here": inferred_dropped,
            "remaining_after": int(remaining_after_measured - inferred_dropped),
            "applied_count": remaining_after_measured,
        }
    )
    final_mask = _safe_bool_series(work["keep_filtered"], default=False)
    final_count = int(final_mask.sum())
    rows.append(
        {
            "stage": "6) rest",
            "remaining_before": int(remaining_after_measured - inferred_dropped),
            "dropped_here": int(max(remaining_after_measured - inferred_dropped - final_count, 0)),
            "remaining_after": final_count,
            "applied_count": int(remaining_after_measured - inferred_dropped),
        }
    )
    return pd.DataFrame(rows)


def _load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _save_pickle(path: str, payload) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


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


def crop_data_pkl_to_window(data_pkl, start_ts, end_ts, tz_name: str):
    cropped = pickle.loads(pickle.dumps(data_pkl))
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


def _write_yaml(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _build_temp_segmentation_config(
    temp_data_root: str,
    dataset_id: str,
    deployment_id: str,
    algo_cfg: Dict[str, Any],
    run_name: str,
    output_root: str,
    context_filter_definitions: Dict[str, Any] | None = None,
) -> str:
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
    if context_filter_definitions:
        payload["segmentation_runs"][run_name]["context_filter_definitions"] = context_filter_definitions
    config_path = os.path.join(output_root, "segmentation_preview_config.yaml")
    pathlib.Path(output_root).mkdir(parents=True, exist_ok=True)
    _write_yaml(config_path, payload)
    return config_path


def _snakemake_executable_cmd() -> list[str]:
    return [sys.executable, "-m", "snakemake"]


def run_algorithmic_preview_sandbox(
    repo_root: str,
    real_data_pkl_path: str,
    dataset_id: str,
    deployment_id: str,
    algo_cfg: Dict[str, Any],
    context_filter_definitions: Dict[str, Any] | None = None,
    preview_start_ts=None,
    preview_end_ts=None,
) -> Dict[str, Any]:
    temp_root = tempfile.mkdtemp(prefix="pyologger_state_review_")
    dataset_root = os.path.join(temp_root, dataset_id, deployment_id, "outputs")
    pathlib.Path(dataset_root).mkdir(parents=True, exist_ok=True)
    temp_pkl_path = os.path.join(dataset_root, "data.pkl")
    temp_output_root = os.path.join(temp_root, "segmentation_outputs")
    run_name = f"state_review_{dataset_id}_{deployment_id}_{uuid.uuid4().hex[:8]}"

    data_pkl = _load_pickle(real_data_pkl_path)
    tz_name = str((getattr(data_pkl, "deployment_info", {}) or {}).get("Time Zone") or "UTC")
    if preview_start_ts is not None and preview_end_ts is not None:
        data_pkl = crop_data_pkl_to_window(data_pkl, preview_start_ts, preview_end_ts, tz_name)
    _save_pickle(temp_pkl_path, data_pkl)

    config_path = _build_temp_segmentation_config(
        temp_data_root=temp_root,
        dataset_id=dataset_id,
        deployment_id=deployment_id,
        algo_cfg=algo_cfg,
        run_name=run_name,
        output_root=temp_output_root,
        context_filter_definitions=context_filter_definitions,
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
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "Snakemake segmentation preview failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\n\nSTDERR:\n{proc.stderr[-4000:]}"
        )

    segments_path = os.path.join(
        temp_output_root,
        "segments",
        "by_deployment",
        f"{dataset_id}__{deployment_id}__algorithmic_segments.parquet",
    )
    summary_path = os.path.join(
        temp_output_root,
        "segments",
        "by_deployment",
        f"{dataset_id}__{deployment_id}__algorithmic_segments_summary.json",
    )
    summary = {}
    if os.path.exists(summary_path):
        try:
            summary = json.loads(pathlib.Path(summary_path).read_text())
        except Exception:
            summary = {}
    seg_df = pd.read_parquet(segments_path) if os.path.exists(segments_path) else pd.DataFrame()
    seg_df = prepare_segment_table(seg_df)
    preview_signal_data: Dict[str, Any] = {}
    preview_signal_info: Dict[str, Any] = {}
    if os.path.exists(temp_pkl_path):
        try:
            result_pkl = _load_pickle(temp_pkl_path)
            result_signal_data = getattr(result_pkl, "signal_data", {}) or {}
            result_signal_info = getattr(result_pkl, "signal_info", {}) or {}
            for signal_name in [
                "algorithmic_intermediate_channels",
                "algorithmic_derivative_channels",
                "algorithmic_depth_d1",
                "algorithmic_depth_d2",
                "algorithmic_feature_channels",
                "algorithmic_labels",
            ]:
                if signal_name in result_signal_data:
                    preview_signal_data[signal_name] = result_signal_data[signal_name]
                if signal_name in result_signal_info:
                    preview_signal_info[signal_name] = result_signal_info[signal_name]
        except Exception:
            preview_signal_data = {}
            preview_signal_info = {}
    return {
        "temp_root": temp_root,
        "temp_pkl_path": temp_pkl_path,
        "run_name": run_name,
        "config_path": config_path,
        "segments_path": segments_path,
        "seg_df": seg_df,
        "signal_data": preview_signal_data,
        "signal_info": preview_signal_info,
        "summary": summary,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def cleanup_temp_workspace(temp_root: str | None) -> None:
    if temp_root and os.path.isdir(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)


def persist_preview_algorithmic_outputs(
    real_data_pkl_path: str,
    preview_data_pkl_path: str,
    *,
    deployment_folder: str | None = None,
    deployment_id: str | None = None,
    write_netcdf: bool = True,
) -> Dict[str, Any]:
    real_data_pkl = _load_pickle(real_data_pkl_path)
    preview_data_pkl = _load_pickle(preview_data_pkl_path)

    if not hasattr(real_data_pkl, "signal_data") or real_data_pkl.signal_data is None:
        real_data_pkl.signal_data = {}
    if not hasattr(real_data_pkl, "signal_info") or real_data_pkl.signal_info is None:
        real_data_pkl.signal_info = {}
    if not hasattr(real_data_pkl, "event_data") or not isinstance(real_data_pkl.event_data, pd.DataFrame):
        real_data_pkl.event_data = pd.DataFrame(
            columns=["date", "time", "value", "type", "key", "duration", "short_description", "long_description", "datetime"]
        )
    if not hasattr(real_data_pkl, "event_manager") or real_data_pkl.event_manager is None:
        real_data_pkl.event_manager = {}

    preview_signal_data = getattr(preview_data_pkl, "signal_data", {}) or {}
    preview_signal_info = getattr(preview_data_pkl, "signal_info", {}) or {}
    signal_names = [
        "algorithmic_labels",
        "algorithmic_intermediate_channels",
        "algorithmic_derivative_channels",
        "algorithmic_feature_channels",
    ]
    saved_signal_names: List[str] = []
    for signal_name in signal_names:
        if signal_name in preview_signal_data:
            payload = preview_signal_data[signal_name]
            real_data_pkl.signal_data[signal_name] = payload.copy() if isinstance(payload, pd.DataFrame) else copy.deepcopy(payload)
            saved_signal_names.append(signal_name)
        else:
            real_data_pkl.signal_data.pop(signal_name, None)
        if signal_name in preview_signal_info:
            real_data_pkl.signal_info[signal_name] = copy.deepcopy(preview_signal_info[signal_name])
        else:
            real_data_pkl.signal_info.pop(signal_name, None)

    preview_event_manager = dict(((getattr(preview_data_pkl, "event_manager", {}) or {}).get("algorithmic_segments") or {}))
    event_keys = set(str(v).strip() for v in (preview_event_manager.get("keys") or []) if str(v).strip())
    preview_event_data = getattr(preview_data_pkl, "event_data", None)
    new_events = pd.DataFrame()
    if isinstance(preview_event_data, pd.DataFrame) and "key" in preview_event_data.columns and event_keys:
        new_events = preview_event_data.loc[preview_event_data["key"].astype(str).isin(event_keys)].copy()
        if "datetime" in new_events.columns:
            new_events["datetime"] = pd.to_datetime(new_events["datetime"], errors="coerce")

    existing_events = real_data_pkl.event_data.copy()
    removed_event_count = 0
    if "key" in existing_events.columns and event_keys:
        drop_mask = existing_events["key"].astype(str).isin(event_keys)
        removed_event_count = int(drop_mask.sum())
        existing_events = existing_events.loc[~drop_mask].copy()
    real_data_pkl.event_data = pd.concat([existing_events, new_events], ignore_index=True)
    if "datetime" in real_data_pkl.event_data.columns:
        real_data_pkl.event_data["datetime"] = pd.to_datetime(real_data_pkl.event_data["datetime"], errors="coerce")
        real_data_pkl.event_data = real_data_pkl.event_data.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    else:
        real_data_pkl.event_data = real_data_pkl.event_data.reset_index(drop=True)

    real_data_pkl.event_manager["algorithmic_segments"] = preview_event_manager
    _save_pickle(real_data_pkl_path, real_data_pkl)

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
            BaseExporter(real_data_pkl).save_to_netcdf(real_data_pkl, filepath=netcdf_path)
        except Exception as exc:
            netcdf_error = f"{type(exc).__name__}: {exc}"

    return {
        "saved_signal_names": saved_signal_names,
        "saved_label_rows": int(
            len(real_data_pkl.signal_data.get("algorithmic_labels", pd.DataFrame()))
            if isinstance(real_data_pkl.signal_data.get("algorithmic_labels"), pd.DataFrame)
            else 0
        ),
        "saved_event_count": int(len(new_events)),
        "removed_event_count": removed_event_count,
        "target_event_count": int(len(event_keys)),
        "netcdf_written": bool(netcdf_path and not netcdf_error),
        "netcdf_path": netcdf_path,
        "netcdf_error": netcdf_error,
    }
