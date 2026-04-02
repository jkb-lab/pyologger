import pandas as pd

from pyologger.analyze_data import segmentation_pipeline as segmod


def test_parse_window_trim_cfg_defaults():
    cfg = segmod._parse_window_trim_cfg(
        {
            "window_trim": {
                "enabled": True,
                "source_channel": "depth.depth",
                "rule": "first_last_depth_threshold",
                "depth_threshold_m": 10.0,
            }
        }
    )
    assert cfg["enabled"] is True
    assert cfg["signal_id"] == "depth"
    assert cfg["channel_id"] == "depth"
    assert cfg["rule"] == "first_last_depth_threshold"
    assert cfg["depth_threshold_m"] == 10.0


def test_parse_supervised_transfer_cfg_fixed_folds():
    cfg = segmod._parse_supervised_transfer_cfg(
        {
            "supervised": {
                "source_dataset_ids": ["mian-juv-nese_sleep_lml-ano_JKB"],
                "source_training_deployment_ids_always": ["2021-04-04_mian-009"],
                "source_holdout_folds": [
                    {
                        "id": "holdout_mian_011",
                        "additional_train_deployment_ids": ["2022-04-01_mian-013"],
                        "holdout_deployment_ids": ["2021-04-17_mian-011"],
                    },
                    {
                        "id": "holdout_mian_013",
                        "additional_train_deployment_ids": ["2021-04-17_mian-011"],
                        "holdout_deployment_ids": ["2022-04-01_mian-013"],
                    },
                ],
                "transfer_target_scope": "all_non_source",
                "transfer_normalization": "per_dataset_zscore",
                "transfer_fusion_modes": ["none", "algorithmic_gate"],
            }
        }
    )
    assert cfg["enabled"] is True
    assert cfg["source_training_deployment_ids_always"] == ["2021-04-04_mian-009"]
    assert len(cfg["source_holdout_folds"]) == 2
    assert cfg["source_holdout_folds"][0]["holdout_deployment_ids"] == ["2021-04-17_mian-011"]
    assert cfg["source_holdout_folds"][1]["holdout_deployment_ids"] == ["2022-04-01_mian-013"]


def test_resolve_supervised_stratify_labels_uses_dataset_label_when_enabled():
    labeled_df = pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds1", "ds1", "ds1", "ds2", "ds2", "ds2", "ds2"],
            "window_key": ["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8"],
        }
    )
    y = pd.Series(["SLEEP", "SLEEP", "WAKE", "WAKE", "SLEEP", "SLEEP", "WAKE", "WAKE"], dtype=object)
    stratify_labels, stratify_mode, fallback_reason = segmod._resolve_supervised_stratify_labels(
        labeled_df,
        y,
        {"stratify_split_by_dataset_label": True},
    )
    assert stratify_mode == "dataset_label"
    assert fallback_reason is None
    assert stratify_labels.tolist() == [
        "ds1||SLEEP",
        "ds1||SLEEP",
        "ds1||WAKE",
        "ds1||WAKE",
        "ds2||SLEEP",
        "ds2||SLEEP",
        "ds2||WAKE",
        "ds2||WAKE",
    ]


def test_resolve_supervised_stratify_labels_falls_back_for_sparse_composite_classes():
    labeled_df = pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds1", "ds2"],
            "window_key": ["w1", "w2", "w3"],
        }
    )
    y = pd.Series(["SLEEP", "SLEEP", "WAKE"], dtype=object)
    stratify_labels, stratify_mode, fallback_reason = segmod._resolve_supervised_stratify_labels(
        labeled_df,
        y,
        {"stratify_split_by_dataset_label": True},
    )
    assert stratify_mode == "label"
    assert fallback_reason == "composite_class_too_small"
    assert stratify_labels.tolist() == y.tolist()


def test_compute_window_trim_interval_and_filtering():
    signal_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2021-01-01", periods=8, freq="1min"),
            "depth": [0.0, 1.0, 12.0, 15.0, 9.0, 11.0, 0.0, 0.0],
        }
    )
    trim_cfg = {
        "enabled": True,
        "signal_id": "depth",
        "channel_id": "depth",
        "rule": "first_last_depth_threshold",
        "depth_threshold_m": 10.0,
    }
    trim_start, trim_end, summary = segmod._compute_window_trim_interval(
        {"depth": signal_df},
        "UTC",
        trim_cfg,
    )
    assert summary["trim_status"] == "ok"
    assert trim_start == pd.Timestamp("2021-01-01 00:02:00", tz="UTC")
    assert trim_end == pd.Timestamp("2021-01-01 00:05:00", tz="UTC")
    assert segmod._window_is_within_trim(trim_start, trim_end, trim_start, trim_end) is True
    assert segmod._window_is_within_trim(
        pd.Timestamp("2021-01-01 00:01:00", tz="UTC"),
        pd.Timestamp("2021-01-01 00:03:00", tz="UTC"),
        trim_start,
        trim_end,
    ) is False


def test_compute_window_trim_interval_handles_no_in_water_interval():
    signal_df = pd.DataFrame(
        {
            "datetime": pd.date_range("2021-01-01", periods=4, freq="1min"),
            "depth": [0.0, 1.0, 2.0, 3.0],
        }
    )
    trim_start, trim_end, summary = segmod._compute_window_trim_interval(
        {"depth": signal_df},
        "UTC",
        {
            "enabled": True,
            "signal_id": "depth",
            "channel_id": "depth",
            "rule": "first_last_depth_threshold",
            "depth_threshold_m": 10.0,
        },
    )
    assert trim_start is None
    assert trim_end is None
    assert summary["trim_status"] == "no_in_water_interval"


def test_supervised_confusion_legend_label_stays_generic_for_non_sleep_positive_class():
    assert segmod._supervised_confusion_legend_label("false_positive", "REM") == "False Positive"
    assert segmod._supervised_confusion_legend_label("false_negative", "SWS") == "False Negative"


def test_select_cluster_pass_rows_prefers_explicit_cluster_pass_name():
    cluster_df = pd.DataFrame(
        {
            "cluster_pass": ["k5", "k2", "review"],
            "cluster_pass_n_clusters": [2, 5, 2],
            "cluster_rank": [1, 0, 1],
        }
    )
    selected = segmod._select_cluster_pass_rows(cluster_df, "k2", n_clusters=2)
    assert selected["cluster_pass"].astype(str).tolist() == ["k2"]
    assert selected["cluster_rank"].astype(int).tolist() == [0]
