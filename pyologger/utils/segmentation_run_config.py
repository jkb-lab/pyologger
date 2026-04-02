from __future__ import annotations

import copy
from typing import Any, Dict


def _setdefault_from_block(dst: Dict[str, Any], block: Dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if key in dst:
            continue
        if key in block:
            dst[key] = block[key]


def _apply_label_group_to_algorithmic_segments(
    algo_segments: Dict[str, Any],
    algorithmic_block: Dict[str, Any],
    supervised_block: Dict[str, Any],
    shared_supervised_label_groups: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    out = dict(algo_segments or {})
    shared_groups = dict(shared_supervised_label_groups or {})
    label_group_name = str(
        out.get("label_group")
        or algorithmic_block.get("label_group")
        or supervised_block.get("label_group")
        or ""
    ).strip()
    if not label_group_name:
        return out

    resolved = dict(shared_groups.get(label_group_name) or {})
    if not resolved:
        return out

    out["label_group"] = label_group_name
    terminals = dict(
        resolved.get("algorithmic_terminals")
        or resolved.get("algorithmic_terminal_mappings")
        or {}
    )
    if not terminals:
        return out

    event_keys = dict(out.get("event_keys") or {})
    label_names = dict(out.get("label_names") or {})
    label_codes = dict(out.get("label_codes") or {})
    for state_name, mapping in terminals.items():
        if not isinstance(mapping, dict):
            continue
        state = str(state_name).strip()
        if not state:
            continue
        event_key = str(mapping.get("event_key") or "").strip()
        label_name = str(mapping.get("label_name") or "").strip()
        label_code = mapping.get("label_code")
        if event_key and state not in event_keys:
            event_keys[state] = event_key
        if label_name and state not in label_names:
            label_names[state] = label_name
        if label_code is not None and state not in label_codes:
            label_codes[state] = int(label_code)

    if event_keys:
        out["event_keys"] = event_keys
    if label_names:
        out["label_names"] = label_names
    if label_codes:
        out["label_codes"] = label_codes
    return out


def normalize_segmentation_run_cfg(
    run_cfg: Dict[str, Any],
    shared_supervised_label_groups: Dict[str, Dict[str, list[str]]] | None = None,
) -> Dict[str, Any]:
    """
    Normalize block-style run config into legacy flat keys expected by workflows.

    Supported blocks:
    - scope
    - algorithmic
    - features
    - unsupervised
    - supervised
    - summary
    """
    out = dict(run_cfg or {})

    scope = out.get("scope") if isinstance(out.get("scope"), dict) else {}
    _setdefault_from_block(
        out,
        scope,
        [
            "analysis_id",
            "dataset_ids",
            "deployment_ids",
            "groups",
            "expected_signals",
            "output_root_override",
        ],
    )

    features = out.get("features") if isinstance(out.get("features"), dict) else {}
    _setdefault_from_block(
        out,
        features,
        [
            "normalization_level",
            "cluster_length_mode",
            "duration_s",
            "window_s",
            "stride_s",
            "min_samples_per_chunk",
            "max_chunks_per_deployment",
            "max_chunks_total",
            "test_run",
            "test_max_chunks_per_deployment",
            "umap_sample_frac",
            "random_state",
            "strict_qc",
            "sampling_compatibility_tolerance",
            "expected_signals",
            "memory_log_interval_s",
            "keep_raw_merged_features",
            "write_csv_debug_outputs",
            "keep_std_features",
            "include_corrected_acc",
            "keep_unified_df",
            "enable_catch22",
            "catch22_feature_cols",
            "catch22_max_chunks_per_deployment",
            "channel_feature_spec",
            "preprocessing",
            "standardized_channel_db_path",
            "umap_n_neighbors",
            "umap_min_dist",
            "umap_metric",
        ],
    )

    algorithmic = out.get("algorithmic") if isinstance(out.get("algorithmic"), dict) else {}
    algo_segments = {}
    if isinstance(algorithmic.get("segments"), dict):
        algo_segments = dict(algorithmic.get("segments") or {})
    elif isinstance(algorithmic.get("algorithmic_segments"), dict):
        algo_segments = dict(algorithmic.get("algorithmic_segments") or {})
    elif isinstance(algorithmic, dict) and any(k in algorithmic for k in ["source_channel", "thresholds", "context_filters"]):
        algo_segments = dict(algorithmic)
    if algo_segments:
        algo_segments = _apply_label_group_to_algorithmic_segments(
            algo_segments,
            algorithmic_block=algorithmic,
            supervised_block=(out.get("supervised") if isinstance(out.get("supervised"), dict) else {}),
            shared_supervised_label_groups=dict(shared_supervised_label_groups or {}),
        )
    if algo_segments and "algorithmic_segments" not in out:
        out["algorithmic_segments"] = algo_segments
    _setdefault_from_block(out, algorithmic, ["context_filter_definitions", "supervised_context_filter_definitions"])

    unsupervised = out.get("unsupervised") if isinstance(out.get("unsupervised"), dict) else {}
    _setdefault_from_block(
        out,
        unsupervised,
        [
            "clustering_window_scope",
            "cluster_passes",
            "primary_input_channel_for_ranking",
            "corr_threshold",
            "n_clusters",
            "max_k",
            "run_pca",
            "run_tsne",
            "run_umap",
            "cluster_pass_name",
        ],
    )

    supervised = out.get("supervised") if isinstance(out.get("supervised"), dict) else {}
    if "enabled" in supervised and "enable_supervised" not in out:
        out["enable_supervised"] = bool(supervised.get("enabled"))
    if "config" in supervised and isinstance(supervised.get("config"), dict):
        out["supervised"] = dict(supervised.get("config") or {})
    if isinstance(out.get("supervised"), dict):
        supervised_cfg = dict(out.get("supervised") or {})
        label_group_name = str(
            supervised_cfg.get("label_group") or supervised.get("label_group") or ""
        ).strip()
        if label_group_name:
            shared_groups = dict(shared_supervised_label_groups or {})
            resolved_cfg = shared_groups.get(label_group_name)
            if isinstance(resolved_cfg, dict):
                # Backward-compatible shortcut: group name directly maps to label_groups.
                has_label_cfg_keys = any(
                    key in resolved_cfg
                    for key in ("label_groups", "label_prefixes", "label_event_keys", "drop_labels", "event_keys")
                )
                if has_label_cfg_keys:
                    if not supervised_cfg.get("label_groups") and isinstance(resolved_cfg.get("label_groups"), dict):
                        supervised_cfg["label_groups"] = copy.deepcopy(resolved_cfg.get("label_groups") or {})
                    if "label_prefixes" not in supervised_cfg and isinstance(resolved_cfg.get("label_prefixes"), list):
                        supervised_cfg["label_prefixes"] = list(resolved_cfg.get("label_prefixes") or [])
                    if "label_event_keys" not in supervised_cfg and isinstance(resolved_cfg.get("label_event_keys"), list):
                        supervised_cfg["label_event_keys"] = list(resolved_cfg.get("label_event_keys") or [])
                    if "drop_labels" not in supervised_cfg and isinstance(resolved_cfg.get("drop_labels"), list):
                        supervised_cfg["drop_labels"] = list(resolved_cfg.get("drop_labels") or [])
                    if "event_keys" not in supervised_cfg and isinstance(resolved_cfg.get("event_keys"), dict):
                        supervised_cfg["event_keys"] = copy.deepcopy(resolved_cfg.get("event_keys") or {})
                elif not supervised_cfg.get("label_groups"):
                    supervised_cfg["label_groups"] = copy.deepcopy(resolved_cfg)
            supervised_cfg["label_group"] = label_group_name
        out["supervised"] = supervised_cfg
    if "context_filter_definitions" in supervised and "supervised_context_filter_definitions" not in out:
        defs = supervised.get("context_filter_definitions")
        if isinstance(defs, dict):
            out["supervised_context_filter_definitions"] = dict(defs)

    summary_block = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    if isinstance(summary_block.get("plots"), dict) and "plots" not in out:
        out["plots"] = dict(summary_block.get("plots") or {})
    if isinstance(summary_block.get("window_trim"), dict) and "window_trim" not in out:
        out["window_trim"] = dict(summary_block.get("window_trim") or {})
    _setdefault_from_block(
        out,
        summary_block,
        ["primary_method_kind", "primary_method_id", "budget_label_mode", "actograms"],
    )
    if isinstance(summary_block.get("config"), dict):
        merged_summary = dict(summary_block.get("config") or {})
        merged_summary.update({k: v for k, v in summary_block.items() if k in {"primary_method_kind", "primary_method_id", "budget_label_mode", "actograms"}})
        out["summary"] = merged_summary

    return out
