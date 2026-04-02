from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyologger.utils.cross_dataset_qc import run_cross_dataset_qc


def _parse_dataset_deployments(items: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw in items or []:
        text = str(raw).strip()
        if not text:
            continue
        if ":" not in text:
            continue
        ds, deps = text.split(":", 1)
        ds = ds.strip()
        dep_list = [d.strip() for d in deps.split(",") if d.strip()]
        if ds and dep_list:
            out[ds] = dep_list
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset channel presence/unit congruency QC.")
    parser.add_argument("--config", required=True, help="Path to pyologger/config.yaml")
    parser.add_argument("--dataset-id", action="append", default=[], help="Dataset ID to include (repeatable).")
    parser.add_argument(
        "--deployment-id",
        action="append",
        default=[],
        help="Deployment ID filter applied to selected datasets (repeatable).",
    )
    parser.add_argument(
        "--dataset-deployments",
        action="append",
        default=[],
        help="Per-dataset deployment subset, format: dataset_id:dep1,dep2 (repeatable).",
    )
    parser.add_argument("--signal", action="append", default=[], help="Signal to check (repeatable).")
    parser.add_argument(
        "--channel",
        action="append",
        default=[],
        help="Channel to check in signal.channel format (repeatable). If omitted, all channels for selected signals are used.",
    )
    parser.add_argument("--output-summary", required=True, help="Output CSV for per-channel summary QC.")
    parser.add_argument("--output-detail", required=False, help="Optional output CSV for per-scope detail QC.")
    args = parser.parse_args()

    dataset_deployments = _parse_dataset_deployments(args.dataset_deployments)
    summary_df, detail_df = run_cross_dataset_qc(
        config_path=args.config,
        dataset_ids=args.dataset_id,
        deployment_ids=args.deployment_id,
        dataset_deployments=dataset_deployments,
        signal_filters=args.signal,
        channel_filters=args.channel,
    )

    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_summary, index=False)
    print(f"[cross-dataset-qc] wrote summary: {out_summary} ({len(summary_df)} rows)")

    if args.output_detail:
        out_detail = Path(args.output_detail)
        out_detail.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(out_detail, index=False)
        print(f"[cross-dataset-qc] wrote detail: {out_detail} ({len(detail_df)} rows)")

    stats = {
        "summary_rows": int(len(summary_df)),
        "detail_rows": int(len(detail_df)),
        "matched_channels": int(summary_df["matched_across_datasets_and_deployments"].fillna(False).astype(bool).sum())
        if not summary_df.empty and "matched_across_datasets_and_deployments" in summary_df.columns
        else 0,
    }
    print(f"[cross-dataset-qc] stats: {json.dumps(stats, sort_keys=True)}")


if __name__ == "__main__":
    main()
