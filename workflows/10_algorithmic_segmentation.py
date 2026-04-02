from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyologger.analyze_data import segmentation_pipeline
from pyologger.utils.cross_dataset_qc import run_cross_dataset_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithmic segmentation step")
    parser.add_argument("--config", required=True, help="Path to pyologger/config.yaml")
    parser.add_argument("--run-name", required=True, help="segmentation_runs key")
    parser.add_argument("--output", required=True, help="Output path for merged algorithmic segments parquet")
    parser.add_argument(
        "--refresh-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate run-level QC CSV before algorithmic segmentation (default: true).",
    )
    parser.add_argument(
        "--refresh-cross-dataset-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate run-scope cross-dataset QC CSVs before algorithmic segmentation (default: true).",
    )
    parser.add_argument(
        "--context-filter-pass-mode",
        "--phase",
        dest="context_filter_pass_mode",
        choices=["full", "measured_only"],
        default="measured_only",
        help=(
            "Context-filter execution phase (default: measured_only). "
            "'measured_only' applies only measured filters; "
            "'full' applies measured and inferred filters."
        ),
    )
    args = parser.parse_args()

    ctx = segmentation_pipeline._resolve_run_context(args.config, args.run_name)
    if args.refresh_qc:
        qc_output = os.path.join(ctx.output_root, "qc", "qc_channels.csv")
        segmentation_pipeline.cmd_qc(
            argparse.Namespace(
                config=args.config,
                run_name=args.run_name,
                output=qc_output,
            )
        )
    if args.refresh_cross_dataset_qc:
        _dataset_deployments: dict = {}
        for _item in ctx.scope:
            _dataset_deployments.setdefault(_item["dataset_id"], []).append(_item["deployment_id"])
        run_cross_dataset_qc(config_path=args.config, dataset_deployments=_dataset_deployments)

    for item in ctx.scope:
        dataset_id = item["dataset_id"]
        deployment_id = item["deployment_id"]
        output_segments, output_summary = segmentation_pipeline._algorithmic_segment_paths(ctx, dataset_id, deployment_id)
        segmentation_pipeline.cmd_algorithmic_segments_deployment(
            argparse.Namespace(
                config=args.config,
                run_name=args.run_name,
                dataset_id=dataset_id,
                deployment_id=deployment_id,
                output_segments=output_segments,
                output_summary=output_summary,
                context_filter_pass_mode=args.context_filter_pass_mode,
            )
        )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        merge_report = tmp.name
    segmentation_pipeline.cmd_algorithmic_segments_merge(
        argparse.Namespace(
            config=args.config,
            run_name=args.run_name,
            output=merge_report,
        )
    )
    merged_segments_path, _ = segmentation_pipeline._algorithmic_segment_merge_paths(ctx)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(merged_segments_path, args.output)
    Path(merge_report).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
