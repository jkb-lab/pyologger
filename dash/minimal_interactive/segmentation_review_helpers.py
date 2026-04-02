from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _run_key(scope: str, dataset_id: str | None, analysis_id: str) -> str:
    scope_key = str(scope or "global").strip().lower() or "global"
    ds = str(dataset_id or "").strip()
    analysis = str(analysis_id or "").strip()
    if scope_key == "dataset":
        return f"dataset::{ds}::{analysis}"
    return f"global::{analysis}"


def _parse_run_key(run_key: str) -> Dict[str, str]:
    text = str(run_key or "").strip()
    if text.startswith("dataset::"):
        _, dataset_id, analysis_id = text.split("::", 2)
        return {
            "scope": "dataset",
            "dataset_id": dataset_id,
            "analysis_id": analysis_id,
        }
    if text.startswith("global::"):
        _, analysis_id = text.split("::", 1)
        return {
            "scope": "global",
            "dataset_id": "",
            "analysis_id": analysis_id,
        }
    return {
        "scope": "global",
        "dataset_id": "",
        "analysis_id": text,
    }


def _run_root_candidates(data_dir: str, dataset_id: str | None) -> List[Tuple[str, pathlib.Path, str]]:
    roots: List[Tuple[str, pathlib.Path, str]] = []
    global_root = pathlib.Path(data_dir) / "00_Meta-Analysis" / "segmentation"
    roots.append(("global", global_root, "All datasets"))
    ds = str(dataset_id or "").strip()
    if ds:
        dataset_root = pathlib.Path(data_dir) / ds / "00_Meta-Analysis" / "segmentation"
        roots.append(("dataset", dataset_root, ds))
    return roots


def _path_exists(path_str: str) -> bool:
    return bool(path_str) and os.path.exists(path_str)


def _read_json(path_str: str) -> Dict[str, Any]:
    if not _path_exists(path_str):
        return {}
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_json_list(path_str: str) -> List[Dict[str, Any]]:
    if not _path_exists(path_str):
        return []
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
    except Exception:
        return []
    return []


def _read_parquet(path_str: str) -> pd.DataFrame:
    if not _path_exists(path_str):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path_str)
    except Exception:
        return pd.DataFrame()


def _safe_iso_mtime(path_obj: pathlib.Path) -> str | None:
    try:
        return pd.Timestamp(path_obj.stat().st_mtime, unit="s").isoformat()
    except Exception:
        return None


def _run_file_manifest(run_dir: pathlib.Path) -> Dict[str, str]:
    return {
        "root": str(run_dir),
        "clustered_windows": str(run_dir / "clustering" / "clustered_windows.parquet"),
        "supervised_report": str(run_dir / "supervised" / "supervised_report.json"),
        "supervised_predictions": str(run_dir / "supervised" / "supervised_predictions.parquet"),
        "supervised_variant_predictions": str(run_dir / "supervised" / "supervised_variant_predictions.parquet"),
        "metrics_by_variant": str(run_dir / "supervised" / "metrics_by_variant.parquet"),
        "segments": str(run_dir / "segments" / "algorithmic_segments.parquet"),
        "segment_events": str(run_dir / "segments" / "algorithmic_segment_events.parquet"),
        "interactive_manifest": str(run_dir / "plots" / "interactive_review_manifest.json"),
        "html_report": str(run_dir / "00_supervised_review_report.html"),
        "interactive_notebook": str(run_dir / "plots" / "A_interactive_deployment_overlay.ipynb"),
        "qc_channels": str(run_dir / "qc" / "qc_channels.csv"),
    }


def discover_segmentation_review_runs(data_dir: str, dataset_id: str | None) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for scope, root, scope_label in _run_root_candidates(data_dir, dataset_id):
        if not root.exists():
            continue
        for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            analysis_id = str(run_dir.name)
            if not analysis_id:
                continue
            run_key = _run_key(scope, dataset_id if scope == "dataset" else "", analysis_id)
            if run_key in seen:
                continue
            seen.add(run_key)
            files = _run_file_manifest(run_dir)
            run_info = {
                "run_key": run_key,
                "analysis_id": analysis_id,
                "scope": scope,
                "scope_label": scope_label,
                "dataset_id": str(dataset_id or "") if scope == "dataset" else "",
                "run_dir": str(run_dir),
                "modified_at": _safe_iso_mtime(run_dir),
                "has_clustered_windows": _path_exists(files["clustered_windows"]),
                "has_algorithmic_segments": _path_exists(files["segments"]) or _path_exists(files["segment_events"]),
                "has_supervised": _path_exists(files["supervised_predictions"]) or _path_exists(files["supervised_variant_predictions"]),
                "has_interactive_artifacts": _path_exists(files["interactive_manifest"]),
                "has_html_report": _path_exists(files["html_report"]),
                "has_interactive_notebook": _path_exists(files["interactive_notebook"]),
                "paths": files,
            }
            run_info["label"] = (
                f"{analysis_id} ({scope_label})"
                if scope == "global"
                else f"{analysis_id} ({scope_label} dataset)"
            )
            runs.append(run_info)
    runs.sort(
        key=lambda row: (
            0 if row.get("has_supervised") else 1,
            0 if row.get("has_algorithmic_segments") else 1,
            str(row.get("modified_at") or ""),
            str(row.get("analysis_id") or ""),
        ),
        reverse=False,
    )
    runs.sort(key=lambda row: str(row.get("modified_at") or ""), reverse=True)
    return runs


def review_run_info_by_key(run_index: Iterable[Dict[str, Any]], run_key: str) -> Dict[str, Any]:
    key = str(run_key or "").strip()
    for row in list(run_index or []):
        if str((row or {}).get("run_key") or "") == key:
            return dict(row)
    return {}


def load_segmentation_review_bundle(run_info: Dict[str, Any]) -> Dict[str, Any]:
    info = dict(run_info or {})
    paths = dict(info.get("paths") or {})
    summary = _read_json(paths.get("supervised_report", ""))
    cdf = _read_parquet(paths.get("clustered_windows", ""))
    spdf = _read_parquet(
        paths.get("supervised_variant_predictions")
        if _path_exists(paths.get("supervised_variant_predictions", ""))
        else paths.get("supervised_predictions", "")
    )
    metrics = _read_parquet(paths.get("metrics_by_variant", ""))
    seg_df = _read_parquet(paths.get("segments", ""))
    seg_event_df = _read_parquet(paths.get("segment_events", ""))
    interactive_manifest = _read_json_list(paths.get("interactive_manifest", ""))
    qc_df = pd.DataFrame()
    qc_path = paths.get("qc_channels", "")
    if _path_exists(qc_path):
        try:
            qc_df = pd.read_csv(qc_path)
        except Exception:
            qc_df = pd.DataFrame()
    return {
        "run_info": info,
        "summary": summary,
        "clustered_windows": cdf,
        "supervised_predictions": spdf,
        "metrics_by_variant": metrics,
        "algorithmic_segments": seg_df,
        "algorithmic_segment_events": seg_event_df,
        "interactive_manifest": interactive_manifest,
        "qc_channels": qc_df,
    }


def _deployment_key(dataset_id: str, deployment_id: str) -> str:
    return f"{str(dataset_id)} / {str(deployment_id)}"


def split_deployment_key(value: str) -> Tuple[str, str]:
    text = str(value or "").strip()
    if " / " in text:
        ds, dep = text.split(" / ", 1)
        return ds.strip(), dep.strip()
    return "", text


def review_deployment_options(bundle: Dict[str, Any]) -> List[Dict[str, str]]:
    cdf = bundle.get("clustered_windows", pd.DataFrame())
    spdf = bundle.get("supervised_predictions", pd.DataFrame())
    seg_df = bundle.get("algorithmic_segments", pd.DataFrame())
    dep_rows: List[Tuple[str, str]] = []
    for frame in (cdf, spdf, seg_df):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if not {"dataset_id", "deployment_id"}.issubset(frame.columns):
            continue
        dep_rows.extend(
            [
                (str(ds), str(dep))
                for ds, dep in frame[["dataset_id", "deployment_id"]].dropna().drop_duplicates().itertuples(index=False)
            ]
        )
    unique = sorted(set(dep_rows), key=lambda item: (item[0], item[1]))
    return [
        {
            "label": _deployment_key(ds, dep),
            "value": _deployment_key(ds, dep),
            "dataset_id": ds,
            "deployment_id": dep,
        }
        for ds, dep in unique
    ]


def _method_value(kind: str, method_id: str) -> str:
    return f"{str(kind)}::{str(method_id)}"


def parse_method_value(value: str) -> Tuple[str, str]:
    text = str(value or "").strip()
    if "::" in text:
        kind, method_id = text.split("::", 1)
        return kind, method_id
    return "", text


def review_method_options(bundle: Dict[str, Any], dataset_id: str, deployment_id: str) -> List[Dict[str, str]]:
    methods: List[Dict[str, str]] = []
    cdf = bundle.get("clustered_windows", pd.DataFrame())
    spdf = bundle.get("supervised_predictions", pd.DataFrame())
    seg_df = bundle.get("algorithmic_segments", pd.DataFrame())

    if isinstance(spdf, pd.DataFrame) and not spdf.empty and {"dataset_id", "deployment_id"}.issubset(spdf.columns):
        dep_spdf = spdf[
            (spdf["dataset_id"].astype(str) == str(dataset_id))
            & (spdf["deployment_id"].astype(str) == str(deployment_id))
        ].copy()
        if not dep_spdf.empty and "observed_label" in dep_spdf.columns:
            observed = dep_spdf["observed_label"].dropna().astype(str)
            observed = observed[observed.ne("") & observed.ne("Unknown")]
            if not observed.empty:
                methods.append(
                    {
                        "label": "Observed labels",
                        "value": _method_value("observed", "observed"),
                    }
                )
        if not dep_spdf.empty and {"variant_id", "variant_label"}.issubset(dep_spdf.columns):
            meta = (
                dep_spdf[["variant_id", "variant_label", "variant_rank"]]
                .drop_duplicates()
                .sort_values(["variant_rank", "variant_label"], na_position="last")
            )
            for row in meta.itertuples(index=False):
                methods.append(
                    {
                        "label": str(row.variant_label),
                        "value": _method_value("supervised", str(row.variant_id)),
                    }
                )

    if isinstance(cdf, pd.DataFrame) and not cdf.empty and {"dataset_id", "deployment_id", "cluster_pass"}.issubset(cdf.columns):
        dep_cdf = cdf[
            (cdf["dataset_id"].astype(str) == str(dataset_id))
            & (cdf["deployment_id"].astype(str) == str(deployment_id))
        ].copy()
        if not dep_cdf.empty:
            for cluster_pass in sorted(dep_cdf["cluster_pass"].dropna().astype(str).unique().tolist()):
                methods.append(
                    {
                        "label": f"{cluster_pass.upper()} clusters",
                        "value": _method_value("cluster", cluster_pass),
                    }
                )

    if isinstance(seg_df, pd.DataFrame) and not seg_df.empty and {"dataset_id", "deployment_id"}.issubset(seg_df.columns):
        dep_seg = seg_df[
            (seg_df["dataset_id"].astype(str) == str(dataset_id))
            & (seg_df["deployment_id"].astype(str) == str(deployment_id))
        ].copy()
        if not dep_seg.empty:
            method_name = "algorithmic"
            if "algorithmic_method" in dep_seg.columns:
                nonnull = dep_seg["algorithmic_method"].dropna().astype(str)
                if not nonnull.empty:
                    method_name = str(nonnull.iloc[0])
            methods.append(
                {
                    "label": f"Algorithmic ({method_name})",
                    "value": _method_value("algorithmic", method_name),
                }
            )

    seen = set()
    deduped = []
    for row in methods:
        val = str(row.get("value") or "")
        if not val or val in seen:
            continue
        seen.add(val)
        deduped.append(row)
    return deduped


def localize_datetimes(values: Any, tz_name: str, strip_timezone: bool = False) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce")
    if not isinstance(dt, pd.Series):
        dt = pd.Series(dt)
    try:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz_name)
        else:
            dt = dt.dt.tz_convert(tz_name)
    except Exception:
        pass
    if strip_timezone:
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    return dt


def build_review_method_events(
    bundle: Dict[str, Any],
    dataset_id: str,
    deployment_id: str,
    method_kind: str,
    method_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    cdf = bundle.get("clustered_windows", pd.DataFrame())
    spdf = bundle.get("supervised_predictions", pd.DataFrame())
    seg_df = bundle.get("algorithmic_segments", pd.DataFrame())
    empty = pd.DataFrame(columns=["datetime", "end_datetime", "key", "short_description", "type", "duration"])
    if method_kind == "observed":
        if not isinstance(spdf, pd.DataFrame) or spdf.empty:
            return empty, {}
        dep = spdf[
            (spdf["dataset_id"].astype(str) == str(dataset_id))
            & (spdf["deployment_id"].astype(str) == str(deployment_id))
        ].copy()
        if dep.empty or not {"window_start", "window_end", "observed_label"}.issubset(dep.columns):
            return empty, {}
        dep = dep.dropna(subset=["window_start", "window_end", "observed_label"]).copy()
        dep["observed_label"] = dep["observed_label"].astype(str)
        dep = dep.loc[dep["observed_label"].ne("") & dep["observed_label"].ne("Unknown")].copy()
        if dep.empty:
            return empty, {}
        dep["datetime"] = pd.to_datetime(dep["window_start"], errors="coerce")
        dep["end_datetime"] = pd.to_datetime(dep["window_end"], errors="coerce")
        dep["duration"] = (dep["end_datetime"] - dep["datetime"]).dt.total_seconds()
        event_df = dep.rename(columns={"observed_label": "key"})[
            ["datetime", "end_datetime", "key", "duration"]
        ].copy()
        event_df["short_description"] = "observed"
        event_df["type"] = "state"
        annotations = {
            key: {"signal": "all", "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False}
            for key in sorted(event_df["key"].dropna().astype(str).unique().tolist())
        }
        return event_df.sort_values("datetime").reset_index(drop=True), annotations

    if method_kind == "cluster":
        if not isinstance(cdf, pd.DataFrame) or cdf.empty:
            return empty, {}
        dep = cdf[
            (cdf["dataset_id"].astype(str) == str(dataset_id))
            & (cdf["deployment_id"].astype(str) == str(deployment_id))
            & (cdf["cluster_pass"].astype(str) == str(method_id))
        ].copy()
        if dep.empty or not {"window_start", "window_end", "cluster_rank"}.issubset(dep.columns):
            return empty, {}
        dep["datetime"] = pd.to_datetime(dep["window_start"], errors="coerce")
        dep["end_datetime"] = pd.to_datetime(dep["window_end"], errors="coerce")
        dep["duration"] = (dep["end_datetime"] - dep["datetime"]).dt.total_seconds()
        dep["key"] = dep["cluster_rank"].map(lambda x: f"{method_id}: cluster {int(x)}" if pd.notna(x) else pd.NA)
        event_df = dep[["datetime", "end_datetime", "key", "duration"]].dropna(subset=["key"]).copy()
        event_df["short_description"] = str(method_id)
        event_df["type"] = "state"
        annotations = {
            key: {"signal": "all", "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False}
            for key in sorted(event_df["key"].dropna().astype(str).unique().tolist())
        }
        return event_df.sort_values("datetime").reset_index(drop=True), annotations

    if method_kind == "algorithmic":
        if not isinstance(seg_df, pd.DataFrame) or seg_df.empty:
            return empty, {}
        dep = seg_df[
            (seg_df["dataset_id"].astype(str) == str(dataset_id))
            & (seg_df["deployment_id"].astype(str) == str(deployment_id))
        ].copy()
        if dep.empty:
            return empty, {}
        start_col = next((c for c in ["segment_start", "start_datetime", "datetime"] if c in dep.columns), None)
        end_col = next((c for c in ["segment_end", "end_datetime", "next_datetime"] if c in dep.columns), None)
        label_col = next((c for c in ["nominal_class", "label_name", "base_nominal_class"] if c in dep.columns), None)
        if not start_col or not end_col or not label_col:
            return empty, {}
        dep["datetime"] = pd.to_datetime(dep[start_col], errors="coerce")
        dep["end_datetime"] = pd.to_datetime(dep[end_col], errors="coerce")
        dep["duration"] = (dep["end_datetime"] - dep["datetime"]).dt.total_seconds()
        dep["key"] = dep[label_col].astype(str)
        event_df = dep[["datetime", "end_datetime", "key", "duration"]].dropna(subset=["datetime", "end_datetime"]).copy()
        event_df = event_df.loc[event_df["key"].ne("") & event_df["key"].ne("not_sleep") & event_df["key"].ne("find_rest.not_sleep")].copy()
        event_df["short_description"] = str(method_id)
        event_df["type"] = "state"
        annotations = {
            key: {"signal": "all", "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False}
            for key in sorted(event_df["key"].dropna().astype(str).unique().tolist())
        }
        return event_df.sort_values("datetime").reset_index(drop=True), annotations

    if method_kind == "supervised":
        if not isinstance(spdf, pd.DataFrame) or spdf.empty:
            return empty, {}
        dep = spdf[
            (spdf["dataset_id"].astype(str) == str(dataset_id))
            & (spdf["deployment_id"].astype(str) == str(deployment_id))
            & (spdf["variant_id"].astype(str) == str(method_id))
        ].copy()
        if dep.empty:
            return empty, {}
        label_col = "predicted_label"
        if label_col not in dep.columns:
            label_col = "final_behavior" if "final_behavior" in dep.columns else label_col
        if label_col not in dep.columns or not {"window_start", "window_end"}.issubset(dep.columns):
            return empty, {}
        dep["datetime"] = pd.to_datetime(dep["window_start"], errors="coerce")
        dep["end_datetime"] = pd.to_datetime(dep["window_end"], errors="coerce")
        dep["duration"] = (dep["end_datetime"] - dep["datetime"]).dt.total_seconds()
        dep["key"] = dep[label_col].astype(str)
        event_df = dep[["datetime", "end_datetime", "key", "duration"]].dropna(subset=["datetime", "end_datetime"]).copy()
        event_df = event_df.loc[event_df["key"].ne("") & event_df["key"].ne("Unknown")].copy()
        event_df["short_description"] = str(method_id)
        event_df["type"] = "state"
        annotations = {
            key: {"signal": "all", "shade_mode": "fill_trace_split", "shade_opacity": 0.25, "draw_line": False}
            for key in sorted(event_df["key"].dropna().astype(str).unique().tolist())
        }
        return event_df.sort_values("datetime").reset_index(drop=True), annotations

    return empty, {}


def review_overview_summary(bundle: Dict[str, Any]) -> Dict[str, Any]:
    cdf = bundle.get("clustered_windows", pd.DataFrame())
    spdf = bundle.get("supervised_predictions", pd.DataFrame())
    seg_df = bundle.get("algorithmic_segments", pd.DataFrame())
    summary = dict(bundle.get("summary") or {})

    deployments = review_deployment_options(bundle)
    labeled_windows = int(summary.get("n_rows_labeled") or 0)
    if labeled_windows <= 0 and isinstance(spdf, pd.DataFrame) and not spdf.empty and "observed_label" in spdf.columns:
        obs = spdf["observed_label"].dropna().astype(str)
        labeled_windows = int(obs[obs.ne("") & obs.ne("Unknown")].shape[0])

    methods_compared = 0
    if isinstance(spdf, pd.DataFrame) and not spdf.empty and "variant_id" in spdf.columns:
        methods_compared += int(spdf["variant_id"].dropna().astype(str).nunique())
    if isinstance(cdf, pd.DataFrame) and not cdf.empty and "cluster_pass" in cdf.columns:
        methods_compared += int(cdf["cluster_pass"].dropna().astype(str).nunique())
    if isinstance(seg_df, pd.DataFrame) and not seg_df.empty:
        methods_compared += 1
    if isinstance(spdf, pd.DataFrame) and not spdf.empty and "observed_label" in spdf.columns:
        obs = spdf["observed_label"].dropna().astype(str)
        if not obs[obs.ne("") & obs.ne("Unknown")].empty:
            methods_compared += 1

    heldout = []
    for fold in list(summary.get("source_holdout_folds") or []):
        heldout.extend([str(dep) for dep in (fold.get("holdout_deployment_ids") or []) if str(dep).strip()])
    return {
        "labeled_windows": labeled_windows,
        "total_windows": int(len(cdf)) if isinstance(cdf, pd.DataFrame) else 0,
        "deployments": len(deployments),
        "methods_compared": methods_compared,
        "source_datasets": [str(v) for v in (summary.get("source_dataset_ids") or []) if str(v).strip()],
        "heldout_deployments": heldout,
        "mode": str(summary.get("mode") or "algorithmic"),
        "has_supervised": bool(bundle.get("run_info", {}).get("has_supervised")),
        "has_algorithmic": bool(bundle.get("run_info", {}).get("has_algorithmic_segments")),
        "has_interactive": bool(bundle.get("run_info", {}).get("has_interactive_artifacts")),
    }


def review_metrics_frames(bundle: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = bundle.get("metrics_by_variant", pd.DataFrame())
    summary = dict(bundle.get("summary") or {})
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = metrics.copy()
    if "fold_id" in work.columns:
        fold_label_map = {}
        for row in list(summary.get("source_holdout_folds") or []):
            fold_id = str(row.get("fold_id") or row.get("id") or "").strip()
            heldout = [str(dep) for dep in (row.get("heldout_deployment_ids") or []) if str(dep).strip()]
            if fold_id:
                fold_label_map[fold_id] = ", ".join(heldout) if heldout else fold_id
        work["heldout_deployment"] = work["fold_id"].astype(str).map(fold_label_map).fillna(work["fold_id"].astype(str))
        metric_cols = [c for c in ["variant_id", "variant_label", "variant_type", "variant_rank", "label", "metric"] if c in work.columns]
        aggregate = (
            work.groupby(metric_cols, as_index=False, dropna=False)["value"]
            .mean()
            .sort_values(["variant_rank", "variant_label", "label", "metric"], na_position="last")
        )
        return aggregate, work
    work["heldout_deployment"] = "All"
    return work, pd.DataFrame()


def review_run_diagnostics(bundle: Dict[str, Any]) -> Dict[str, Any]:
    seg_df = bundle.get("algorithmic_segments", pd.DataFrame())
    spdf = bundle.get("supervised_predictions", pd.DataFrame())
    summary = dict(bundle.get("summary") or {})
    out: Dict[str, Any] = {
        "context_rejected_segments": None,
        "algorithmic_final_kept": None,
        "algorithmic_base_candidates": None,
        "trimmed_windows": None,
        "dropped_land_windows": None,
        "variant_skip_counts": [],
    }
    if isinstance(seg_df, pd.DataFrame) and not seg_df.empty:
        if {"base_keep_filtered", "context_keep"}.issubset(seg_df.columns):
            base_keep = seg_df["base_keep_filtered"].fillna(False).astype(bool)
            context_keep = seg_df["context_keep"].fillna(True).astype(bool)
            out["context_rejected_segments"] = int((base_keep & ~context_keep).sum())
            out["algorithmic_base_candidates"] = int(base_keep.sum())
            out["algorithmic_final_kept"] = int(context_keep.sum())
        elif "nominal_class" in seg_df.columns:
            nominal = seg_df["nominal_class"].astype(str)
            out["algorithmic_final_kept"] = int(nominal[~nominal.isin(["", "not_sleep", "find_rest.not_sleep"])].shape[0])
        if "context_reject_reason" in seg_df.columns:
            reasons = (
                seg_df["context_reject_reason"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(8)
                .reset_index()
                .rename(columns={"index": "reason", "context_reject_reason": "count"})
            )
            out["context_reject_reasons"] = reasons.to_dict("records")
    if isinstance(spdf, pd.DataFrame) and not spdf.empty:
        if "trimmed_window_count" in spdf.columns:
            out["trimmed_windows"] = int(pd.to_numeric(spdf["trimmed_window_count"], errors="coerce").fillna(0).max())
        if "dropped_prepost_land_window_count" in spdf.columns:
            out["dropped_land_windows"] = int(pd.to_numeric(spdf["dropped_prepost_land_window_count"], errors="coerce").fillna(0).max())
        if {"variant_skip_reason", "variant_applied_to_deployment"}.issubset(spdf.columns):
            skipped = spdf.loc[
                (~spdf["variant_applied_to_deployment"].fillna(True).astype(bool))
                & spdf["variant_skip_reason"].notna(),
                "variant_skip_reason",
            ]
            if not skipped.empty:
                out["variant_skip_counts"] = skipped.astype(str).value_counts().head(8).rename_axis("reason").reset_index(name="count").to_dict("records")
    label_scan_summary = dict(summary.get("label_scan_summary") or {})
    if label_scan_summary:
        out["label_scan_summary"] = label_scan_summary
    return out
