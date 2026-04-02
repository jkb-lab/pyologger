from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyologger.analyze_data import segmentation_pipeline
from pyologger.utils.cross_dataset_qc import run_cross_dataset_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised segmentation workflow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--features", required=False, help="Accepted for compatibility; clustered outputs are resolved from run outputs.")
    parser.add_argument("--output", required=False, help="Optional output path for clustered windows parquet")
    parser.add_argument(
        "--refresh-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate run-level QC CSV before unsupervised segmentation (default: true).",
    )
    parser.add_argument(
        "--refresh-cross-dataset-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate run-scope cross-dataset QC CSVs before unsupervised segmentation (default: true).",
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
    segmentation_pipeline.cmd_cluster(argparse.Namespace(config=args.config, run_name=args.run_name))
    if args.output:
        clustered_path = Path(ctx.output_root) / "clustering" / "clustered_windows.parquet"
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clustered_path, args.output)


if __name__ == "__main__":
    main()
