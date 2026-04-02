#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyologger.utils.segmentation_run_config import normalize_segmentation_run_cfg


def _escape_md(text: Any) -> str:
    s = str(text if text is not None else "").strip()
    if not s:
        return "-"
    return s.replace("|", "\\|").replace("\n", "<br>")


def _fmt_list(values: Any, max_items: int = 8) -> str:
    if values is None:
        return "-"
    if isinstance(values, str):
        values = [values]
    try:
        items = [str(v) for v in list(values) if str(v).strip()]
    except Exception:
        items = [str(values)]
    if not items:
        return "-"
    if len(items) <= max_items:
        return ", ".join(items)
    head = ", ".join(items[:max_items])
    return f"{head}, ... (+{len(items) - max_items})"


def _scope_value(cfg: dict[str, Any], max_items: int) -> dict[str, str]:
    scope = cfg.get("scope") if isinstance(cfg.get("scope"), dict) else {}
    groups = scope.get("groups") if isinstance(scope.get("groups"), dict) else {}
    return {
        "analysis_id": str(scope.get("analysis_id") or cfg.get("analysis_id") or "-"),
        "datasets": _fmt_list(scope.get("dataset_ids") or cfg.get("dataset_ids"), max_items=max_items),
        "deployments": _fmt_list(scope.get("deployment_ids") or cfg.get("deployment_ids"), max_items=max_items),
        "expected_signals": _fmt_list(scope.get("expected_signals") or cfg.get("expected_signals"), max_items=max_items),
        "groups": (
            "-"
            if not groups
            else f"{len(groups)} groups: "
            + ", ".join([f"{k} ({len(v) if isinstance(v, list) else 0})" for k, v in groups.items()])
        ),
    }


def _label_value(cfg: dict[str, Any], max_items: int) -> dict[str, str]:
    scfg = cfg.get("supervised") if isinstance(cfg.get("supervised"), dict) else {}
    label_group_name = str(scfg.get("label_group") or "-")
    label_groups = scfg.get("label_groups") if isinstance(scfg.get("label_groups"), dict) else {}
    drop_labels = scfg.get("drop_labels") if isinstance(scfg.get("drop_labels"), list) else []
    class_bits = []
    for class_name, raw in label_groups.items():
        n = len(raw) if isinstance(raw, list) else 0
        class_bits.append(f"{class_name} ({n})")
    return {
        "label_group": label_group_name,
        "label_classes": ", ".join(class_bits) if class_bits else "-",
        "drop_labels": _fmt_list(drop_labels, max_items=max_items),
    }


def _algorithmic_value(cfg: dict[str, Any], max_items: int) -> str:
    algo = cfg.get("algorithmic") if isinstance(cfg.get("algorithmic"), dict) else {}
    seg = algo.get("segments") if isinstance(algo.get("segments"), dict) else {}
    if not seg:
        return "-"
    thresholds = seg.get("thresholds") if isinstance(seg.get("thresholds"), dict) else {}
    context_filters = seg.get("context_filters") if isinstance(seg.get("context_filters"), list) else []
    bits = [
        f"enabled={bool(seg.get('enabled', True))}",
        f"method={seg.get('method') or '-'}",
        f"source={seg.get('source_channel') or '-'}",
    ]
    if thresholds:
        bits.append(
            "thresholds="
            + ", ".join([f"{k}={v}" for k, v in thresholds.items()])
        )
    if context_filters:
        bits.append(f"context_filters={len(context_filters)} [{_fmt_list(context_filters, max_items=max_items)}]")
    return "<br>".join(bits)


def _features_value(cfg: dict[str, Any], max_items: int) -> str:
    fcfg = cfg.get("features") if isinstance(cfg.get("features"), dict) else {}
    if not fcfg:
        return "-"
    spec = fcfg.get("channel_feature_spec") if isinstance(fcfg.get("channel_feature_spec"), dict) else {}
    bits = [
        f"normalization={fcfg.get('normalization_level', '-')}",
        f"cluster_length_mode={fcfg.get('cluster_length_mode', '-')}",
        f"duration_s={fcfg.get('duration_s', '-')}",
        f"strict_qc={bool(fcfg.get('strict_qc', False))}",
        f"enable_catch22={bool(fcfg.get('enable_catch22', False))}",
        f"channels={len(spec)} [{_fmt_list(spec.keys(), max_items=max_items)}]",
    ]
    return "<br>".join(bits)


def _unsupervised_value(cfg: dict[str, Any], max_items: int) -> str:
    ucfg = cfg.get("unsupervised") if isinstance(cfg.get("unsupervised"), dict) else {}
    if not ucfg:
        return "-"
    passes = ucfg.get("cluster_passes") if isinstance(ucfg.get("cluster_passes"), list) else []
    pass_labels = []
    for p in passes:
        if not isinstance(p, dict):
            continue
        nm = p.get("name", "pass")
        k = p.get("n_clusters", "?")
        pass_labels.append(f"{nm}(k={k})")
    bits = [
        f"passes={len(pass_labels)} [{_fmt_list(pass_labels, max_items=max_items)}]",
        f"primary_input={ucfg.get('primary_input_channel_for_ranking', '-')}",
        f"corr_threshold={ucfg.get('corr_threshold', '-')}",
        f"n_clusters={ucfg.get('n_clusters', '-')}, max_k={ucfg.get('max_k', '-')}",
        f"pca={bool(ucfg.get('run_pca', False))}, tsne={bool(ucfg.get('run_tsne', False))}, umap={bool(ucfg.get('run_umap', False))}",
    ]
    return "<br>".join(bits)


def _supervised_value(cfg: dict[str, Any], max_items: int) -> str:
    scfg_block = cfg.get("supervised") if isinstance(cfg.get("supervised"), dict) else {}
    if not scfg_block:
        return "-"

    # Support both block style:
    #   supervised: {enabled: true, config: {...}}
    # and normalized/flat style:
    #   supervised: {...config...}, enable_supervised: true
    has_nested_cfg = isinstance(scfg_block.get("config"), dict)
    if has_nested_cfg:
        enabled = bool(scfg_block.get("enabled", cfg.get("enable_supervised", False)))
        scfg = dict(scfg_block.get("config") or {})
    else:
        enabled = bool(cfg.get("enable_supervised", scfg_block.get("enabled", True)))
        scfg = dict(scfg_block)

    if not scfg:
        return f"enabled={enabled}"
    ablations = scfg.get("ablation_feature_groups") if isinstance(scfg.get("ablation_feature_groups"), list) else []
    bits = [
        f"enabled={enabled}",
        f"test_size={scfg.get('test_size', '-')}",
        f"rf_n_estimators={scfg.get('rf_n_estimators', '-')}",
        f"prediction_scope={scfg.get('prediction_scope', '-')}",
        f"confidence_threshold={scfg.get('confidence_threshold', '-')}",
        f"ablation={scfg.get('ablation_mode', '-')} ({len(ablations)} groups)",
        f"export_rf_events={bool(scfg.get('export_rf_events', False))}",
        f"label_group={scfg.get('label_group', '-')}",
    ]
    return "<br>".join(bits)


def _summary_value(cfg: dict[str, Any]) -> str:
    summary = cfg.get("summary") if isinstance(cfg.get("summary"), dict) else {}
    scfg = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    s = scfg if scfg else summary
    if not s:
        return "-"
    act = s.get("actograms") if isinstance(s.get("actograms"), dict) else {}
    bits = [
        f"primary_method_kind={s.get('primary_method_kind', '-')}",
        f"primary_method_id={s.get('primary_method_id', '-')}",
        f"budget_label_mode={s.get('budget_label_mode', '-')}",
    ]
    if act:
        bits.append(f"actograms: daily={bool(act.get('daily', False))}, continuous={bool(act.get('continuous', False))}")
    return "<br>".join(bits)


def build_markdown_table(config_path: str, selected_runs: list[str] | None, max_items: int) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    runs = payload.get("segmentation_runs") if isinstance(payload.get("segmentation_runs"), dict) else {}
    if not runs:
        raise ValueError("No 'segmentation_runs' section found.")

    shared_label_groups = payload.get("supervised_label_group_definitions") or {}
    shared_label_groups = shared_label_groups if isinstance(shared_label_groups, dict) else {}

    run_names = list(runs.keys())
    if selected_runs:
        wanted = [r for r in selected_runs if r in runs]
        missing = [r for r in selected_runs if r not in runs]
        if missing:
            raise ValueError(f"Unknown run names: {', '.join(missing)}")
        run_names = wanted

    normalized = {
        name: normalize_segmentation_run_cfg(runs[name], shared_supervised_label_groups=shared_label_groups)
        for name in run_names
    }

    row_defs = [
        ("Analysis ID", lambda c: _scope_value(c, max_items)["analysis_id"]),
        ("Datasets", lambda c: _scope_value(c, max_items)["datasets"]),
        ("Deployments", lambda c: _scope_value(c, max_items)["deployments"]),
        ("Expected Signals", lambda c: _scope_value(c, max_items)["expected_signals"]),
        ("Groups", lambda c: _scope_value(c, max_items)["groups"]),
        ("Label Group", lambda c: _label_value(c, max_items)["label_group"]),
        ("Label Classes", lambda c: _label_value(c, max_items)["label_classes"]),
        ("Drop Labels", lambda c: _label_value(c, max_items)["drop_labels"]),
        ("Algorithmic", lambda c: _algorithmic_value(c, max_items)),
        ("Features", lambda c: _features_value(c, max_items)),
        ("Unsupervised", lambda c: _unsupervised_value(c, max_items)),
        ("Supervised", lambda c: _supervised_value(c, max_items)),
        ("Summary", lambda c: _summary_value(c)),
    ]

    header = ["Section"] + run_names
    sep = ["---"] * len(header)
    lines = [
        "| " + " | ".join([_escape_md(x) for x in header]) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for row_name, fn in row_defs:
        row = [row_name]
        for run_name in run_names:
            row.append(fn(normalized[run_name]))
        lines.append("| " + " | ".join([_escape_md(x) for x in row]) + " |")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print segmentation-runs comparison as a Markdown table.")
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "segmentation_runs.yaml"),
        help="Path to segmentation_runs.yaml (default: pyologger/segmentation_runs.yaml)",
    )
    parser.add_argument(
        "--runs",
        default="",
        help="Optional comma-separated run names to include (default: all runs in file order).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=8,
        help="Max list items to show per cell before truncating.",
    )
    args = parser.parse_args()

    selected = [x.strip() for x in str(args.runs or "").split(",") if x.strip()]
    md = build_markdown_table(args.config, selected_runs=selected, max_items=max(args.max_items, 1))
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
