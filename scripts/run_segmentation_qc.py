from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyologger.analyze_data import segmentation_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmentation channel/unit QC for a configured run.")
    parser.add_argument("--config", required=True, help="Path to pyologger/config.yaml")
    parser.add_argument("--run-name", required=True, help="segmentation_runs key")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--standardized-channel-db",
        required=False,
        help="Optional standardized_channel_db CSV/Parquet path with Parent signal, Channel ID, Unit Override.",
    )
    args = parser.parse_args()

    segmentation_pipeline.cmd_qc(
        argparse.Namespace(
            config=args.config,
            run_name=args.run_name,
            output=args.output,
            standardized_channel_db=args.standardized_channel_db,
        )
    )


if __name__ == "__main__":
    main()
