import importlib.util
import sys
from pathlib import Path
import tempfile

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "pyologger" / "utils" / "state_event_review.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
spec = importlib.util.spec_from_file_location("state_event_review_test", MODULE_PATH)
review = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = review
assert spec.loader is not None
spec.loader.exec_module(review)


def _segments_df():
    base = pd.Timestamp("2021-04-01T00:00:00Z")
    rows = []
    for idx in range(1, 13):
        start = base + pd.to_timedelta(idx, unit="h")
        end = start + pd.Timedelta(minutes=20)
        rows.append(
            {
                "segment_rank": idx,
                "start_datetime": start,
                "end_datetime": end,
                "duration_s": 1200.0,
                "base_keep_filtered": True,
                "context_keep": idx not in {3, 7},
                "keep_filtered": idx not in {3, 7},
                "segment_family": "drift_candidate" if idx <= 10 else "surface",
                "base_nominal_class": "long_drift" if idx <= 10 else "surface_sleep",
                "nominal_class": "long_drift" if idx <= 10 else "surface_sleep",
                "label_name": "find_rest.long_drift" if idx <= 10 else "find_rest.surface_sleep",
                "base_state_key": "find_rest.long_drift" if idx <= 10 else "find_rest.surface_sleep",
                "context_reject_reason": "stroke_rate_too_high" if idx == 3 else ("late_trip_refinement_failed" if idx == 7 else pd.NA),
                "context_reject_stage": "rejected_after_measured_filters" if idx == 3 else ("rejected_after_inferred_filters" if idx == 7 else pd.NA),
                "context_filter_ids": "low_stroke_rate_required|late_trip_refinement" if idx in {3, 7} else "low_stroke_rate_required|drift_within_intrinsic_context",
                "low_stroke_rate_required_pass": False if idx == 3 else True,
                "low_stroke_rate_required_value": 12.0 if idx == 3 else 4.0,
                "late_trip_refinement_pass": False if idx == 7 else (True if idx <= 10 else pd.NA),
                "late_trip_refinement_value": 0.20 if idx == 7 else (0.05 if idx <= 10 else pd.NA),
            }
        )
    return review.prepare_segment_table(pd.DataFrame(rows))


def test_choose_default_candidate_uses_tenth_final_kept_with_fallback():
    seg_df = _segments_df()
    candidates, idx = review.choose_default_candidate(seg_df, state_name="find_rest.long_drift", candidate_rank=10)
    assert len(candidates) == 8
    assert idx == 7
    assert int(candidates.iloc[idx]["segment_rank"]) == 10


def test_positive_state_names_handles_multiple_state_types():
    seg_df = _segments_df()
    assert review.positive_state_names(seg_df) == ["find_rest.long_drift", "find_rest.surface_sleep"]


def test_filter_state_segments_supports_group_selector_prefix():
    seg_df = _segments_df().copy()
    seg_df.loc[seg_df["nominal_class"] == "long_drift", "label_name"] = "putative_rest.drift_sleep"
    seg_df.loc[seg_df["nominal_class"] == "surface_sleep", "label_name"] = "putative_rest.surface_sleep"
    seg_df = review.prepare_segment_table(seg_df)
    grouped = review.filter_state_segments(seg_df, state_name="putative_rest.*", final_kept_only=True, positive_only=True)
    assert set(grouped["state_name"].unique().tolist()) == {"putative_rest.drift_sleep", "putative_rest.surface_sleep"}
    options = review.selection_options_for_positive_states(review.positive_state_names(seg_df))
    assert "putative_rest.*" in options


def test_summarize_state_filters_counts_rejections_and_filter_failures():
    seg_df = _segments_df()
    summary, reasons, filters = review.summarize_state_filters(
        seg_df,
        state_name="find_rest.long_drift",
        ordered_filter_ids=["low_stroke_rate_required", "late_trip_refinement"],
    )
    assert summary["base_pass_count"] == 10
    assert summary["final_kept_count"] == 8
    assert summary["context_rejected_count"] == 2
    assert round(summary["reject_rate_pct"], 1) == 20.0
    assert set(reasons["reject_reason"].tolist()) == {"stroke_rate_too_high", "late_trip_refinement_failed"}

    low_stroke = filters.loc[filters["filter_id"] == "low_stroke_rate_required"].iloc[0]
    late_refinement = filters.loc[filters["filter_id"] == "late_trip_refinement"].iloc[0]
    assert int(low_stroke["failed_count"]) == 1
    assert int(late_refinement["failed_count"]) == 1


def test_centered_window_bounds_returns_twenty_four_hour_window():
    seg_df = _segments_df()
    row = seg_df.iloc[0]
    start, end = review.centered_window_bounds(row, window_hours=24)
    assert (end - start) == pd.Timedelta(hours=24)
    assert start == pd.Timestamp("2021-03-31T13:10:00Z")


def test_build_review_yaml_snippet_uses_run_specific_context_and_enables_debug():
    config = {
        "segmentation_workflows": {
            "find_rest": {
                "runtime_params": {
                    "thresholds": {"min_duration_s": 180.0},
                    "covariates": {"intrinsic": {}, "extrinsic": {}},
                    "context_filters": [],
                    "debug": {"write_context_review_table": False},
                }
            }
        },
        "segmentation_runs": {
            "mian_mile_sleep_transfer_rf": {
                "analysis_id": "mian_mile_sleep_transfer_rf",
                "context_filter_definitions": {
                    "low_stroke_rate_required": {
                        "enabled": True,
                        "reject_reason": "stroke_rate_too_high",
                    }
                },
                "algorithmic_segments": {
                    "thresholds": {"min_duration_s": 200.0, "drift_rate_abs_max_ms": 0.3},
                    "covariates": {"intrinsic": {"foo": {"enabled": True}}, "extrinsic": {}},
                    "context_filters": ["low_stroke_rate_required"],
                    "terminal_criteria": {"long_flat": {"uses_end_flat_abs_mean_d1_max_ms": False}},
                    "debug": {"write_context_review_table": False},
                },
            }
        },
    }
    snippet = review.build_review_yaml_snippet(config, "mian_mile_sleep_transfer_rf")
    assert "context_filter_definitions:" in snippet
    assert "low_stroke_rate_required" in snippet
    assert "min_duration_s: 200.0" in snippet
    assert "terminal_criteria:" in snippet
    assert "uses_end_flat_abs_mean_d1_max_ms: false" in snippet
    assert "write_context_review_table: true" in snippet


def test_default_algorithmic_cfg_includes_terminal_criteria_defaults():
    cfg = review.default_algorithmic_cfg_from_config({})
    assert cfg["terminal_criteria"]["surface_sleep"]["uses_surface_sleep_min_duration"] is True
    assert cfg["terminal_criteria"]["long_flat"]["uses_end_flat_abs_mean_d1_max_ms"] is True
    assert cfg["terminal_criteria"]["long_drift"]["uses_drift_rate_abs_max_ms"] is True


def test_build_non_context_yaml_snippet_excludes_context_filters_and_covariates():
    config = {
        "segmentation_runs": {
            "mian_mile_sleep_transfer_rf": {
                "analysis_id": "mian_mile_sleep_transfer_rf",
                "algorithmic_segments": {
                    "thresholds": {"min_duration_s": 200.0},
                    "covariates": {"intrinsic": {"foo": {"enabled": True}}, "extrinsic": {}},
                    "context_filters": ["low_stroke_rate_required"],
                    "terminal_criteria": {"long_flat": {"uses_end_flat_abs_mean_d1_max_ms": False}},
                },
            }
        }
    }
    algo_cfg = review.default_algorithmic_cfg_from_config(config, analysis_id="mian_mile_sleep_transfer_rf")
    snippet = review.build_non_context_algorithmic_yaml_snippet_from_algo_cfg(
        config,
        "mian_mile_sleep_transfer_rf",
        algo_cfg,
    )
    assert "thresholds:" in snippet
    assert "terminal_criteria:" in snippet
    assert "context_filters:" not in snippet
    assert "covariates:" not in snippet


def test_review_non_context_config_update_preserves_context_and_other_run_keys():
    config_payload = {
        "segmentation_workflows": {
            "find_rest": {
                "runtime_params": {
                    "thresholds": {"min_duration_s": 180.0},
                }
            }
        },
        "segmentation_runs": {
            "mian_mile_sleep_transfer_rf": {
                "analysis_id": "mian_mile_sleep_transfer_rf",
                "dataset_ids": ["dataset_a"],
                "context_filter_definitions": {"foo": {"enabled": True}},
                "algorithmic_segments": {
                    "thresholds": {"min_duration_s": 200.0},
                    "covariates": {"intrinsic": {"foo": {"enabled": True}}, "extrinsic": {}},
                    "context_filters": ["foo"],
                    "terminal_criteria": {"long_flat": {"uses_end_flat_abs_mean_d1_max_ms": False}},
                    "debug": {"persist_intermediate_channels": True},
                },
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
        algo_cfg = review.default_algorithmic_cfg_from_config(config_payload, analysis_id="mian_mile_sleep_transfer_rf")
        algo_cfg["thresholds"]["min_duration_s"] = 333.0
        algo_cfg["filter_out_ascent"] = True
        review_payload = review.review_non_context_algorithmic_config_update(
            str(cfg_path),
            "mian_mile_sleep_transfer_rf",
            algo_cfg,
        )
        updated_algo = review_payload["updated_config"]["segmentation_runs"]["mian_mile_sleep_transfer_rf"]["algorithmic_segments"]
        assert updated_algo["thresholds"]["min_duration_s"] == 333.0
        assert updated_algo["context_filters"] == ["foo"]
        assert updated_algo["covariates"]["intrinsic"]["foo"]["enabled"] is True
        assert updated_algo["debug"]["persist_intermediate_channels"] is True
        assert updated_algo["filter_out_ascent"] is True
        assert "algorithmic_segments.filter_out_ascent" in review_payload["changed_paths"]


def test_save_non_context_config_update_creates_missing_algorithmic_segments_block():
    config_payload = {
        "segmentation_runs": {
            "mian_mile_sleep_transfer_rf": {
                "analysis_id": "mian_mile_sleep_transfer_rf",
                "dataset_ids": ["dataset_a"],
            }
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
        algo_cfg = review.default_algorithmic_cfg_from_config({}, analysis_id="mian_mile_sleep_transfer_rf")
        algo_cfg["thresholds"]["min_duration_s"] = 222.0
        save_result = review.save_non_context_algorithmic_config_update(
            str(cfg_path),
            "mian_mile_sleep_transfer_rf",
            algo_cfg,
        )
        runs_path = Path(save_result["config_path"])
        persisted = yaml.safe_load(runs_path.read_text())
        algo_section = persisted["segmentation_runs"]["mian_mile_sleep_transfer_rf"]["algorithmic_segments"]
        assert algo_section["thresholds"]["min_duration_s"] == 222.0
        assert "context_filters" not in algo_section
        assert save_result["run_name"] == "mian_mile_sleep_transfer_rf"
        original_main = yaml.safe_load(cfg_path.read_text())
        assert "algorithmic_segments" not in original_main["segmentation_runs"]["mian_mile_sleep_transfer_rf"]


def test_build_mask_period_rows_groups_consecutive_true_runs():
    signal_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2021-04-01T00:00:00Z", periods=6, freq="10s"),
            "is_unfiltered_rest_candidate": [False, True, True, False, True, True],
        }
    )
    periods = review.build_mask_period_rows(signal_df, "is_unfiltered_rest_candidate", "__mask__")
    assert len(periods) == 2
    assert periods.iloc[0]["datetime"] == pd.Timestamp("2021-04-01T00:00:10Z")
    assert periods.iloc[0]["end_datetime"] == pd.Timestamp("2021-04-01T00:00:30Z")
    assert periods.iloc[0]["duration"] == 20.0
    assert periods.iloc[1]["datetime"] == pd.Timestamp("2021-04-01T00:00:40Z")
    assert periods.iloc[1]["end_datetime"] == pd.Timestamp("2021-04-01T00:01:00Z")
    assert periods.iloc[1]["duration"] == 20.0


def test_stage_dropoff_summary_includes_unfiltered_rest_and_filter_progression():
    seg_df = _segments_df()
    signal_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2021-04-01T00:00:00Z", periods=8, freq="10s"),
            "is_unfiltered_rest_candidate": [False, True, True, False, True, True, True, False],
        }
    )
    stage_df = review.stage_dropoff_summary(
        seg_df,
        state_name="find_rest.long_drift",
        ordered_filter_ids=["low_stroke_rate_required", "late_trip_refinement"],
        signal_data={"algorithmic_feature_channels": signal_df},
    )
    assert stage_df["stage"].tolist() == [
        "1) unfiltered_rest_candidate",
        "2) filtered_rest_candidate",
        "3) unfiltered_putative_rest",
        "4) rejected_after_measured_filters",
        "5) rejected_after_inferred_filters",
        "6) rest",
    ]
    unfiltered = stage_df.iloc[0]
    filtered = stage_df.iloc[1]
    pre_context = stage_df.iloc[2]
    measured = stage_df.iloc[3]
    inferred = stage_df.iloc[4]
    rest = stage_df.iloc[5]
    assert int(unfiltered["remaining_after"]) == 2
    assert int(filtered["remaining_before"]) == 10
    assert int(filtered["dropped_here"]) == 0
    assert int(filtered["remaining_after"]) == 10
    assert int(pre_context["remaining_before"]) == 10
    assert int(pre_context["dropped_here"]) == 0
    assert int(pre_context["remaining_after"]) == 10
    assert int(measured["remaining_before"]) == 10
    assert int(measured["dropped_here"]) == 1
    assert int(measured["remaining_after"]) == 9
    assert int(inferred["remaining_before"]) == 9
    assert int(inferred["dropped_here"]) == 1
    assert int(inferred["remaining_after"]) == 8
    assert int(rest["remaining_before"]) == 8
    assert int(rest["dropped_here"]) == 0
    assert int(rest["remaining_after"]) == 8
