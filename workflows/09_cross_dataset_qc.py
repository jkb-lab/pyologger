from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyologger.analyze_data import segmentation_pipeline
from pyologger.utils.cross_dataset_qc import run_cross_dataset_qc


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset QC for segmentation run scope")
    parser.add_argument("--config", required=True, help="Path to pyologger/config.yaml")
    parser.add_argument("--run-name", required=True, help="segmentation_runs key")
    parser.add_argument("--output", required=True, help="Marker output path")
    args = parser.parse_args()

    ctx = segmentation_pipeline._resolve_run_context(args.config, args.run_name)
    _dataset_deployments: dict = {}
    for _item in ctx.scope:
        _dataset_deployments.setdefault(_item["dataset_id"], []).append(_item["deployment_id"])
    run_cross_dataset_qc(config_path=args.config, dataset_deployments=_dataset_deployments)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("cross_dataset_qc_done=1\n", encoding="utf-8")
    print(f"[cross-dataset-qc] wrote marker {args.output}")


if __name__ == "__main__":
    main()
