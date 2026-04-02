import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyologger.utils.folder_manager import create_dataset_structure


def _base_inputs(tmp_path, merged_df):
    deployments = {
        "2011019": {
            "merged_df": merged_df,
        }
    }
    mapping_results = {
        "2011019": {
            "status": "success",
            "deployment_id": "deployment-001",
            "recording_id": "recording-001",
        }
    }
    return deployments, mapping_results, str(tmp_path)


def test_create_dataset_structure_interpolates_short_gaps(tmp_path):
    merged_df = pd.DataFrame(
        {
            "stroke_time_utc": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:01Z",
                    "2024-01-01T00:00:04Z",
                    "2024-01-01T00:00:05Z",
                ],
                utc=True,
            ),
            "depth": [0.0, 1.0, 4.0, 5.0],
            "stroke": [10.0, 11.0, 14.0, 15.0],
            "toppid": ["2011019"] * 4,
        }
    )
    deployments, mapping_results, data_dir = _base_inputs(tmp_path, merged_df)

    result = create_dataset_structure(
        dataset_id="test_dataset",
        deployments=deployments,
        mapping_results=mapping_results,
        data_dir=data_dir,
        save_parquet=True,
        dry_run=False,
    )

    parquet_path = result["deployment_folders"]["deployment-001"]["parquet_files"][0]
    saved = pd.read_parquet(parquet_path)

    assert len(saved) == 6
    assert saved["stroke_time_utc"].tolist() == list(
        pd.to_datetime(
            [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:01Z",
                "2024-01-01T00:00:02Z",
                "2024-01-01T00:00:03Z",
                "2024-01-01T00:00:04Z",
                "2024-01-01T00:00:05Z",
            ],
            utc=True,
        )
    )
    assert saved["depth"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert saved["stroke"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert saved["toppid"].tolist() == ["2011019"] * 6


def test_create_dataset_structure_raises_on_long_gaps(tmp_path):
    merged_df = pd.DataFrame(
        {
            "stroke_time_utc": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:01Z",
                    "2024-01-01T00:02:02Z",
                ],
                utc=True,
            ),
            "depth": [0.0, 1.0, 122.0],
        }
    )
    deployments, mapping_results, data_dir = _base_inputs(tmp_path, merged_df)

    with pytest.raises(ValueError, match="meets or exceeds the 0 days 00:02:00 interpolation limit"):
        create_dataset_structure(
            dataset_id="test_dataset",
            deployments=deployments,
            mapping_results=mapping_results,
            data_dir=data_dir,
            save_parquet=True,
            dry_run=True,
        )
