import pandas as pd

from pyologger.analyze_data import segmentation_pipeline as segmod


def _runtime_cfg():
    return {
        "algorithmic_segments": {
            "enabled": True,
            "source_channel": "depth.depth",
            "method_name": "find_rest",
            "event_keys": {
                "initial": "putative_rest.initial",
                "filtered": "putative_rest.filtered",
                "surface_sleep": "rest.surface",
                "long_flat": "rest.benthic",
                "long_drift": "rest.drift",
            },
            "label_codes": {
                "not_sleep": 0,
                "surface_sleep": 1,
                "long_flat": 2,
                "long_drift": 3,
            },
            "label_names": {
                "not_sleep": "rest.wake",
                "surface_sleep": "rest.surface",
                "long_flat": "rest.benthic",
                "long_drift": "rest.drift",
            },
            "thresholds": {},
            "covariates": {
                "intrinsic": {
                    "inferred_buoyancy": {
                        "enabled": True,
                        "provider": "smoothed_segment_metric",
                        "source_metric": "drift_rate_ms",
                        "subset_filters": {
                            "min_duration_s": 180.0,
                            "max_start_depth_m": 500.0,
                            "exclude_long_flats": True,
                        },
                        "smoothing": {
                            "window_fraction_of_trip": 0.25,
                            "min_periods": 1,
                        },
                        "phase_thresholds": {
                            "negative_max": -0.03,
                            "positive_min": 0.0,
                        },
                        "trip_phase_inference": {
                            "min_positive_elapsed_days": 20.0,
                            "backtrack_days": 20.0,
                            "always_positive_max_day": 20.0,
                            "near_arrival_exclusion_days": 5.0,
                            "early_window_days": 15.0,
                            "late_window_days": 15.0,
                            "reference_max_start_depth_m": 500.0,
                        },
                        "outputs": {
                            "segment_value_field": "inferred_buoyancy_value",
                            "segment_phase_field": "inferred_buoyancy_phase",
                            "segment_positive_likely_day_field": "inferred_positive_likely_day",
                            "segment_trip_phase_field": "inferred_trip_phase",
                            "segment_early_reference_field": "inferred_buoyancy_early_reference_value",
                            "segment_late_reference_field": "inferred_buoyancy_late_reference_value",
                        },
                    }
                },
                "extrinsic": {},
            },
            "context_filters": [
                {
                    "id": "drift_within_intrinsic_context",
                    "enabled": True,
                    "target_state_keys": ["find_rest.long_drift"],
                    "applies_when": {"segment_family": "drift_candidate"},
                    "covariate_ref": "intrinsic.inferred_buoyancy",
                    "rule": {
                        "mode": "threshold_gate",
                        "value_expr": "abs(drift_rate_ms - inferred_buoyancy_value)",
                        "keep_if": {"comparator": "<", "threshold": 0.30},
                    },
                    "on_missing_covariate": "reject",
                    "action_on_fail": "reject",
                    "reject_reason": "intrinsic_context_mismatch",
                },
                {
                    "id": "prepositive_negative_only",
                    "enabled": True,
                    "target_state_keys": ["find_rest.long_drift"],
                    "applies_when": {"inferred_trip_phase_in": ["pre_positive", "never_positive"]},
                    "rule": {
                        "mode": "category_gate",
                        "field": "segment_sign",
                        "allowed_values": ["negative"],
                    },
                    "action_on_fail": "reject",
                    "reject_reason": "prepositive_positive_drift_rejected",
                },
                {
                    "id": "postpositive_negative_deep_enough",
                    "enabled": True,
                    "target_state_keys": ["find_rest.long_drift"],
                    "applies_when": {
                        "inferred_trip_phase_in": ["post_positive", "always_positive"],
                        "segment_sign": "negative",
                    },
                    "rule": {
                        "mode": "threshold_gate",
                        "field": "start_depth_m",
                        "keep_if": {"comparator": ">=", "threshold": 200.0},
                    },
                    "action_on_fail": "reject",
                    "reject_reason": "postpositive_negative_too_shallow",
                },
                {
                    "id": "early_trip_refinement",
                    "enabled": True,
                    "target_state_keys": ["find_rest.long_drift"],
                    "applies_when": {
                        "inferred_trip_phase_in": ["pre_positive", "never_positive"],
                        "elapsed_days": {"comparator": "<", "threshold": 15.0},
                    },
                    "rule": {
                        "mode": "threshold_gate",
                        "value_expr": "abs(drift_rate_ms - inferred_buoyancy_early_reference_value)",
                        "keep_if": {"comparator": "<", "threshold": 0.15},
                    },
                    "on_missing_covariate": "skip_filter",
                    "action_on_fail": "reject",
                    "reject_reason": "early_trip_refinement_failed",
                },
            ],
            "debug": {
                "write_context_review_table": True,
                "persist_context_helper_channels": False,
                "persist_intermediate_channels": True,
            },
        }
    }


def _segment_df():
    start = pd.Timestamp("2021-01-01T00:00:00Z")
    elapsed_days = [1.0, 5.0, 45.0, 50.0]
    drifts = [0.12, -0.17, -0.10, 0.14]
    starts = [120.0, 320.0, 150.0, 350.0]
    rows = []
    for i, day in enumerate(elapsed_days, start=1):
        midpoint = start + pd.to_timedelta(day, unit="D")
        rows.append(
            {
                "segment_rank": i,
                "segment_family": "drift_candidate",
                "base_nominal_class": "long_drift",
                "base_keep_filtered": True,
                "base_state_key": "find_rest.long_drift",
                "duration_s": 240.0,
                "segment_midpoint_datetime": midpoint,
                "elapsed_days": day,
                "drift_rate_ms": drifts[i - 1],
                "start_depth_m": starts[i - 1],
                "is_long_flat": False,
                "context_keep": True,
                "segment_sign": "positive" if drifts[i - 1] > 0 else "negative",
            }
        )
    return pd.DataFrame(rows)


def test_parse_algorithmic_segments_cfg_defaults_include_context_blocks():
    parsed = segmod._parse_algorithmic_segments_cfg(
        {"algorithmic_segments": {"enabled": True, "source_channel": "depth.depth", "thresholds": {}}}
    )
    assert parsed["covariates"] == {"intrinsic": {}, "extrinsic": {}}
    assert parsed["context_filters"] == []
    assert parsed["debug"] == {
        "write_context_review_table": False,
        "persist_context_helper_channels": False,
        "persist_intermediate_channels": True,
    }


def test_named_filter_refs_resolve_from_definitions():
    resolved = segmod._resolve_named_filter_items(
        {
            "gliding_must-be-true": {
                "enabled": True,
                "rule": {"mode": "threshold_gate", "field": "algorithmic_segment_mean_stroke_rate_spm"},
            }
        },
        ["gliding_must-be-true"],
    )
    assert len(resolved) == 1
    assert resolved[0]["id"] == "gliding_must-be-true"
    assert resolved[0]["rule"]["field"] == "algorithmic_segment_mean_stroke_rate_spm"


def test_intrinsic_buoyancy_inference_assigns_trip_phase_and_outputs():
    parsed = segmod._parse_algorithmic_segments_cfg(_runtime_cfg())
    annotated, cov_meta = segmod.annotate_segment_covariates(_segment_df(), None, parsed, None)
    assert "intrinsic.inferred_buoyancy" in cov_meta
    assert "inferred_buoyancy_value" in annotated.columns
    assert "inferred_buoyancy_phase" in annotated.columns
    assert "inferred_positive_likely_day" in annotated.columns
    assert "inferred_trip_phase" in annotated.columns
    assert set(annotated["inferred_trip_phase"].astype(str)) >= {"pre_positive", "post_positive"}
    assert float(annotated["inferred_positive_likely_day"].dropna().iloc[0]) >= 0.0


def test_segment_channel_stat_intrinsic_covariate_computes_per_segment_mean():
    seg_df = pd.DataFrame(
        {
            "segment_rank": [1, 2],
            "start_datetime": [pd.Timestamp("2021-01-01T00:00:00Z"), pd.Timestamp("2021-01-01T00:05:00Z")],
            "end_datetime": [pd.Timestamp("2021-01-01T00:02:00Z"), pd.Timestamp("2021-01-01T00:07:00Z")],
        }
    )
    signal_data = {
        "stroke_rate": pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2021-01-01T00:00:00Z",
                        "2021-01-01T00:01:00Z",
                        "2021-01-01T00:02:00Z",
                        "2021-01-01T00:05:00Z",
                        "2021-01-01T00:06:00Z",
                        "2021-01-01T00:07:00Z",
                    ]
                ),
                "stroke_rate": [8.0, 10.0, 12.0, 4.0, 6.0, 8.0],
            }
        )
    }
    out = segmod.compute_intrinsic_covariate(
        seg_df,
        None,
        "stroke_rate_rest_gate",
        {
            "provider": "segment_channel_stat",
            "source_channel": "stroke_rate.stroke_rate",
            "agg": "mean",
            "outputs": {"segment_value_field": "segment_mean_stroke_rate_spm"},
        },
        runtime_params={"covariates": {"intrinsic": {}}},
        deployment_context={"signal_data": signal_data, "deployment_tz": "UTC"},
    )
    assert "segment_mean_stroke_rate_spm" in out.columns
    assert round(float(out.loc[0, "segment_mean_stroke_rate_spm"]), 3) == 10.0
    assert round(float(out.loc[1, "segment_mean_stroke_rate_spm"]), 3) == 6.0


def test_ordered_context_filters_reject_expected_segments():
    parsed = segmod._parse_algorithmic_segments_cfg(_runtime_cfg())
    annotated, cov_meta = segmod.annotate_segment_covariates(_segment_df(), None, parsed, None)
    filtered = segmod.apply_context_filters(annotated, parsed["context_filters"], cov_meta)

    by_rank = filtered.set_index("segment_rank")
    assert not bool(by_rank.loc[1, "context_keep"])
    assert "prepositive_positive_drift_rejected" in str(by_rank.loc[1, "context_reject_reason"])
    assert bool(by_rank.loc[2, "context_keep"])
    assert not bool(by_rank.loc[3, "context_keep"])
    assert "postpositive_negative_too_shallow" in str(by_rank.loc[3, "context_reject_reason"])
    assert bool(by_rank.loc[4, "context_keep"])


def test_low_stroke_rate_filter_uses_segment_mean_value_from_covariate_ref():
    segment_df = pd.DataFrame(
        {
            "segment_rank": [1, 2],
            "base_keep_filtered": [True, True],
            "context_keep": [True, True],
            "base_state_key": ["find_rest.long_drift", "find_rest.long_drift"],
            "segment_family": ["drift_candidate", "drift_candidate"],
            "segment_mean_stroke_rate_spm": [21.0, 9.5],
        }
    )
    filter_cfgs = [
        {
            "id": "low_stroke_rate_required",
            "enabled": True,
            "target_state_keys": ["find_rest.long_drift"],
            "covariate_ref": "intrinsic.stroke_rate_rest_gate",
            "rule": {
                "mode": "threshold_gate",
                "keep_if": {"comparator": "<", "threshold": 20.0},
            },
            "reject_reason": "stroke_rate_too_high",
        }
    ]
    cov_meta = {
        "intrinsic.stroke_rate_rest_gate": {
            "value_field": "segment_mean_stroke_rate_spm",
        }
    }
    filtered = segmod.apply_context_filters(segment_df, filter_cfgs, cov_meta)
    by_rank = filtered.set_index("segment_rank")
    assert not bool(by_rank.loc[1, "context_keep"])
    assert by_rank.loc[1, "low_stroke_rate_required_value"] == 21.0
    assert bool(by_rank.loc[2, "context_keep"])
    assert by_rank.loc[2, "low_stroke_rate_required_value"] == 9.5


def test_end_to_end_base_vs_final_nominal_class_reduces_long_drift_candidates():
    parsed = segmod._parse_algorithmic_segments_cfg(_runtime_cfg())
    seg_df = _segment_df()
    annotated, cov_meta = segmod.annotate_segment_covariates(seg_df, None, parsed, None)
    filtered = segmod.apply_context_filters(annotated, parsed["context_filters"], cov_meta)
    filtered["keep_filtered"] = filtered["base_keep_filtered"] & filtered["context_keep"]
    filtered["nominal_class"] = filtered["base_nominal_class"].where(filtered["keep_filtered"], "not_sleep")

    assert (filtered["base_nominal_class"] == "long_drift").all()
    assert filtered["keep_filtered"].sum() == 2
    assert (filtered["nominal_class"] == "long_drift").sum() == 2
    assert (filtered["nominal_class"] == "not_sleep").sum() == 2


def test_context_filters_apply_measured_stage_before_inferred_stage():
    segment_df = pd.DataFrame(
        {
            "segment_rank": [1, 2],
            "base_keep_filtered": [True, True],
            "context_keep": [True, True],
            "base_state_key": ["find_rest.long_drift", "find_rest.long_drift"],
            "segment_family": ["drift_candidate", "drift_candidate"],
            "segment_sign": ["negative", "positive"],
            "segment_mean_stroke_rate_spm": [12.0, 4.0],
            "inferred_trip_phase": ["pre_positive", "pre_positive"],
        }
    )
    filter_cfgs = [
        {
            "id": "low_stroke_rate_required",
            "enabled": True,
            "target_state_keys": ["find_rest.long_drift"],
            "rule": {
                "mode": "threshold_gate",
                "field": "segment_mean_stroke_rate_spm",
                "keep_if": {"comparator": "<=", "threshold": 10.0},
            },
            "reject_reason": "stroke_rate_too_high",
        },
        {
            "id": "prepositive_negative_only",
            "enabled": True,
            "target_state_keys": ["find_rest.long_drift"],
            "applies_when": {"inferred_trip_phase_in": ["pre_positive"]},
            "rule": {
                "mode": "category_gate",
                "field": "segment_sign",
                "allowed_values": ["negative"],
            },
            "reject_reason": "prepositive_positive_drift_rejected",
        },
    ]
    filtered = segmod.apply_context_filters(segment_df, filter_cfgs, {})
    by_rank = filtered.set_index("segment_rank")
    assert by_rank.loc[1, "context_reject_stage"] == "rejected_after_measured_filters"
    assert by_rank.loc[1, "context_reject_reason"] == "stroke_rate_too_high"
    assert not bool(by_rank.loc[1, "post_measured_keep"])
    assert by_rank.loc[2, "context_reject_stage"] == "rejected_after_inferred_filters"
    assert by_rank.loc[2, "context_reject_reason"] == "prepositive_positive_drift_rejected"
    assert bool(by_rank.loc[2, "post_measured_keep"])
