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
    parser = argparse.ArgumentParser(description="Summary/report workflow")
    parser.add_argument("--config", required=True, help="Path to pyologger/config.yaml")
    parser.add_argument("--run-name", required=True, help="segmentation_runs key")
    parser.add_argument("--supervised", required=False, help="Accepted for compatibility; report inputs are resolved from run outputs.")
    parser.add_argument("--output", required=False, help="Optional output path for the generated review report")
    parser.add_argument(
        "--include-interactive-review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate interactive deployment overlay assets in summary workflow (default: false).",
    )
    parser.add_argument(
        "--refresh-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate run-level QC CSV before summary plots/report (default: true).",
    )
    parser.add_argument(
        "--refresh-cross-dataset-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate run-scope cross-dataset QC CSVs before summary plots/report (default: true).",
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
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        marker_path = tmp.name
    segmentation_pipeline.cmd_plots(
        argparse.Namespace(
            config=args.config,
            run_name=args.run_name,
            output=marker_path,
            skip_interactive_review=not bool(args.include_interactive_review),
        )
    )

    if args.output:
        candidate_reports = [
            Path(ctx.output_root) / "00_supervised_review_report.html",
            Path(ctx.output_root) / "00_interactive_review_report.html",
        ]
        report_path = next((path for path in candidate_reports if path.exists()), Path(marker_path))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_path, args.output)
    Path(marker_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
