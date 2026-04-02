from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "pyologger" / "analyze_data" / "dataset_state_summaries.py"
spec = importlib.util.spec_from_file_location("dataset_state_summaries_test", MODULE_PATH)
summary_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary_mod
assert spec.loader is not None
spec.loader.exec_module(summary_mod)


def test_write_dataset_state_summary_parquets(tmp_path):
    dataset_folder = tmp_path / "example-dataset"
    deployment_folder = dataset_folder / "2021-01-01_test-001" / "outputs"
    deployment_folder.mkdir(parents=True)

    datetimes = pd.date_range("2021-01-01 00:00:00", periods=8, freq="1h")
    data_pkl = SimpleNamespace(
        deployment_info={
            "Time Zone": "UTC",
            "Deployment Latitude": 36.0,
            "Deployment Longitude": -122.0,
        },
        signal_data={
            "depth": pd.DataFrame(
                {
                    "datetime": datetimes,
                    "depth": [0.5, 5.0, 10.0, 1.0, 0.0, 8.0, 12.0, 0.5],
                }
            ),
            "location": pd.DataFrame(
                {
                    "datetime": datetimes,
                    "latitude": [36.0, 36.1, 36.2, 36.3, 36.4, 36.5, 36.6, 36.7],
                    "longitude": [-122.0, -122.1, -122.2, -122.3, -122.4, -122.5, -122.6, -122.7],
                }
            ),
            "algorithmic_feature_channels": pd.DataFrame(
                {
                    "datetime": datetimes,
                    "depth_std_m": [0, 1, 2, 1, 0, 2, 3, 0],
                    "is_unfiltered_rest_candidate": [False, True, True, False, False, True, True, False],
                }
            ),
            "algorithmic_derivative_channels": pd.DataFrame(
                {
                    "datetime": datetimes,
                    "depth_d1_ms": [0.0, -0.1, -0.05, 0.0, 0.0, 0.02, 0.01, 0.0],
                    "depth_d2_ms2": [0.0, 0.01, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0],
                }
            ),
        },
        event_data=pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2021-01-01 01:00:00"), pd.Timestamp("2021-01-01 05:00:00")],
                "duration": [7200.0, 3600.0],
                "key": ["rest.drift", "rest.surface"],
                "short_description": ["rest.drift", "rest.surface"],
                "long_description": ["stage", "stage"],
                "value": [3, 1],
                "type": ["state", "state"],
            }
        ),
        event_manager={"algorithmic_segments": {"keys": ["find_rest.long_drift", "find_rest.surface_sleep"]}},
    )
    with open(deployment_folder / "data.pkl", "wb") as handle:
        pickle.dump(data_pkl, handle)

    result = summary_mod.write_dataset_state_summary_parquets(str(dataset_folder), dive_depth_threshold_m=2.0)
    daily_df = result["daily_activity_summary"]
    rest_df = result["putative_rest_summary"]

    assert not daily_df.empty
    assert not rest_df.empty
    assert "sunrise_local_hour" in daily_df.columns
    assert "drift" in daily_df.columns
    assert "surface" in daily_df.columns
    assert set(rest_df["state_subtype"].astype(str)) == {"drift", "surface"}
    assert "latitude" in rest_df.columns
    assert "longitude" in rest_df.columns
    assert "algorithmic_derivative_channels__depth_d1_ms" in rest_df.columns

    loaded = summary_mod.load_dataset_state_summary_parquets(str(dataset_folder))
    assert len(loaded["daily_activity_summary"]) == len(daily_df)
    assert len(loaded["putative_rest_summary"]) == len(rest_df)
