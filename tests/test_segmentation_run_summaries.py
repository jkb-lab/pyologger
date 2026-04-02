from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "pyologger" / "analyze_data" / "segmentation_run_summaries.py"
spec = importlib.util.spec_from_file_location("segmentation_run_summaries_test", MODULE_PATH)
summary_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = summary_mod
assert spec.loader is not None
spec.loader.exec_module(summary_mod)


def test_write_segmentation_run_summary_parquets(tmp_path):
    algo_seg_df = pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds1"],
            "deployment_id": ["dep1", "dep1"],
            "segment_rank": [1, 2],
            "start_datetime": [pd.Timestamp("2021-01-01 01:00:00Z"), pd.Timestamp("2021-01-01 05:00:00Z")],
            "end_datetime": [pd.Timestamp("2021-01-01 02:00:00Z"), pd.Timestamp("2021-01-01 06:00:00Z")],
            "duration_s": [3600.0, 3600.0],
            "state_name": ["putative_rest.drift_sleep", "putative_rest.surface_sleep"],
            "keep_filtered": [True, True],
            "base_keep_filtered": [True, True],
            "context_keep": [True, True],
            "drift_rate_ms": [-0.1, 0.0],
        }
    )
    spdf = pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds1"],
            "deployment_id": ["dep1", "dep1"],
            "window_start": [pd.Timestamp("2021-01-01 01:30:00Z"), pd.Timestamp("2021-01-01 08:00:00Z")],
            "window_end": [pd.Timestamp("2021-01-01 02:00:00Z"), pd.Timestamp("2021-01-01 08:30:00Z")],
            "predicted_label": ["SLEEP", "SLEEP"],
            "variant_id": ["rf_full", "rf_full"],
            "pred_confidence": [0.9, 0.8],
        }
    )
    datetimes = pd.date_range("2021-01-01 00:00:00", periods=12, freq="1h", tz="UTC")
    data_pkl = SimpleNamespace(
        deployment_info={"Time Zone": "UTC", "Deployment Latitude": 36.0, "Deployment Longitude": -122.0},
        signal_data={
            "depth": pd.DataFrame({"datetime": datetimes, "depth": [0, 5, 8, 0, 0, 7, 9, 0, 0, 4, 0, 0]}),
            "location": pd.DataFrame({"datetime": datetimes, "latitude": [36.0] * len(datetimes), "longitude": [-122.0] * len(datetimes)}),
            "algorithmic_feature_channels": pd.DataFrame(
                {"datetime": datetimes, "depth_std_m": range(len(datetimes)), "is_unfiltered_rest_candidate": [False, True] * 6}
            ),
        },
    )

    result = summary_mod.write_segmentation_run_summary_parquets(
        run_output_root=str(tmp_path / "run"),
        algorithmic_seg_df=algo_seg_df,
        supervised_prediction_df=spdf,
        load_data_pkl_for_deployment=lambda dataset_id, deployment_id: data_pkl,
        dive_depth_threshold_m=2.0,
    )

    seg_summary = result["segmentation_summary"]
    rest_summary = result["putative_rest_summary"]
    daily_summary = result["daily_activity_summary"]
    overlap_summary = result["putative_rest_overlap_summary"]
    method_daily = result["method_budget_daily"]
    method_hourly = result["method_budget_hourly"]
    method_group = result["method_budget_group_comparison"]
    method_meta = result["method_budget_metadata"]

    assert not seg_summary.empty
    assert set(seg_summary["method"].astype(str)) == {"algorithmic", "supervised"}
    assert not rest_summary.empty
    assert {"algorithmic_overlap_s", "weak_confusion", "latitude", "longitude"}.issubset(rest_summary.columns)
    assert {"tp", "fp", "fn"} & set(rest_summary["weak_confusion"].dropna().astype(str))
    assert not daily_summary.empty
    assert {"method", "state_label", "duration_h", "sunrise_local_hour", "sunset_local_hour"}.issubset(daily_summary.columns)
    assert not overlap_summary.empty
    assert not method_daily.empty
    assert not method_hourly.empty
    assert not method_group.empty
    assert not method_meta.empty
    assert {"method", "method_variant", "state_label_raw", "state_label_canonical", "pct_of_24h"}.issubset(method_daily.columns)
    assert {"hour_of_day", "pct_of_observed_hour", "sunrise_local_hour", "sunset_local_hour"}.issubset(method_hourly.columns)
    assert {"comparison_level", "comparison_key", "state_label_canonical"}.issubset(method_group.columns)
    assert {"raw_label", "canonical_label", "is_unmapped"}.issubset(method_meta.columns)


def test_method_budget_group_rollup_levels(tmp_path):
    algo_seg_df = pd.DataFrame(
        {
            "dataset_id": ["ds1"],
            "deployment_id": ["dep1"],
            "segment_rank": [1],
            "start_datetime": [pd.Timestamp("2021-01-01 00:00:00Z")],
            "end_datetime": [pd.Timestamp("2021-01-01 01:00:00Z")],
            "duration_s": [3600.0],
            "state_name": ["putative_rest.drift_sleep"],
            "keep_filtered": [True],
        }
    )
    spdf = pd.DataFrame(
        {
            "dataset_id": ["ds1"],
            "deployment_id": ["dep1"],
            "window_start": [pd.Timestamp("2021-01-01 01:00:00Z")],
            "window_end": [pd.Timestamp("2021-01-01 01:30:00Z")],
            "predicted_label": ["SLEEP"],
            "variant_id": ["rf_full"],
        }
    )
    datetimes = pd.date_range("2021-01-01 00:00:00", periods=6, freq="1h", tz="UTC")
    data_pkl = SimpleNamespace(
        deployment_info={"Time Zone": "UTC", "Deployment Latitude": 36.0, "Deployment Longitude": -122.0},
        signal_data={
            "depth": pd.DataFrame({"datetime": datetimes, "depth": [0, 5, 8, 0, 0, 4]}),
            "location": pd.DataFrame({"datetime": datetimes, "latitude": [36.0] * len(datetimes), "longitude": [-122.0] * len(datetimes)}),
        },
    )
    result = summary_mod.write_segmentation_run_summary_parquets(
        run_output_root=str(tmp_path / "run2"),
        algorithmic_seg_df=algo_seg_df,
        supervised_prediction_df=spdf,
        load_data_pkl_for_deployment=lambda dataset_id, deployment_id: data_pkl,
        groups={"Kenya_F": ["dep1"]},
    )
    levels = set(result["method_budget_group_comparison"]["comparison_level"].astype(str).tolist())
    assert {"deployment", "dataset", "group"}.issubset(levels)


def test_algorithmic_method_budget_prefers_exhaustive_segments(tmp_path):
    algo_seg_df = pd.DataFrame(
        {
            "dataset_id": ["ds1"],
            "deployment_id": ["dep1"],
            "segment_rank": [1],
            "start_datetime": [pd.Timestamp("2021-01-01 00:00:00Z")],
            "end_datetime": [pd.Timestamp("2021-01-01 00:30:00Z")],
            "duration_s": [1800.0],
            "state_name": ["putative_rest.drift_sleep"],
            "keep_filtered": [True],
        }
    )
    algo_exhaustive_df = pd.DataFrame(
        {
            "dataset_id": ["ds1"],
            "deployment_id": ["dep1"],
            "start_datetime": [pd.Timestamp("2021-01-01 00:00:00Z")],
            "end_datetime": [pd.Timestamp("2021-01-01 03:00:00Z")],
            "duration_s": [10800.0],
            "nominal_class": ["behavior_active_swimming_diving"],
        }
    )
    datetimes = pd.date_range("2021-01-01 00:00:00", periods=5, freq="1h", tz="UTC")
    data_pkl = SimpleNamespace(
        deployment_info={"Time Zone": "UTC", "Deployment Latitude": 36.0, "Deployment Longitude": -122.0},
        signal_data={
            "depth": pd.DataFrame({"datetime": datetimes, "depth": [5, 6, 7, 8, 9]}),
            "location": pd.DataFrame({"datetime": datetimes, "latitude": [36.0] * len(datetimes), "longitude": [-122.0] * len(datetimes)}),
        },
    )

    result = summary_mod.write_segmentation_run_summary_parquets(
        run_output_root=str(tmp_path / "run3"),
        algorithmic_seg_df=algo_seg_df,
        algorithmic_exhaustive_seg_df=algo_exhaustive_df,
        supervised_prediction_df=pd.DataFrame(),
        load_data_pkl_for_deployment=lambda dataset_id, deployment_id: data_pkl,
    )

    algo_budget = result["method_budget_daily"]
    algo_budget = algo_budget[
        (algo_budget["method"].astype(str) == "algorithmic")
        & (algo_budget["method_variant"].astype(str) == "algorithmic_segments")
    ].copy()
    assert not algo_budget.empty
    assert "behavior_active_swimming_diving" in set(algo_budget["state_label_raw"].astype(str))
