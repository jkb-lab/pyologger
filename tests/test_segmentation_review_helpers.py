import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "dash" / "minimal_interactive" / "segmentation_review_helpers.py"
spec = importlib.util.spec_from_file_location("segmentation_review_helpers_test", MODULE_PATH)
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
assert spec.loader is not None
spec.loader.exec_module(helpers)


def test_discover_segmentation_review_runs_finds_dataset_and_global(tmp_path):
    data_dir = tmp_path / "data"
    dataset_root = data_dir / "mian-juv-nese_sleep_lml-ano_JKB" / "00_Meta-Analysis" / "segmentation" / "dataset_run"
    global_root = data_dir / "00_Meta-Analysis" / "segmentation" / "global_run"
    (dataset_root / "clustering").mkdir(parents=True)
    (global_root / "segments").mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(dataset_root / "clustering" / "clustered_windows.parquet", index=False)
    pd.DataFrame({"x": [1]}).to_parquet(global_root / "segments" / "algorithmic_segments.parquet", index=False)

    runs = helpers.discover_segmentation_review_runs(str(data_dir), "mian-juv-nese_sleep_lml-ano_JKB")
    keys = {row["run_key"] for row in runs}
    assert "dataset::mian-juv-nese_sleep_lml-ano_JKB::dataset_run" in keys
    assert "global::global_run" in keys


def test_review_metrics_frames_aggregates_holdout_folds():
    bundle = {
        "summary": {
            "source_holdout_folds": [
                {"fold_id": "holdout_mian_011", "heldout_deployment_ids": ["2021-04-17_mian-011"]},
                {"fold_id": "holdout_mian_013", "heldout_deployment_ids": ["2022-04-01_mian-013"]},
            ]
        },
        "metrics_by_variant": pd.DataFrame(
            {
                "variant_id": ["rf_full", "rf_full", "rf_full", "rf_full"],
                "variant_label": ["RF full", "RF full", "RF full", "RF full"],
                "variant_type": ["random_forest"] * 4,
                "variant_rank": [0] * 4,
                "label": ["Overall", "Overall", "SLEEP", "SLEEP"],
                "metric": ["accuracy", "accuracy", "recall", "recall"],
                "value": [0.8, 0.6, 0.7, 0.5],
                "fold_id": ["holdout_mian_011", "holdout_mian_013", "holdout_mian_011", "holdout_mian_013"],
            }
        ),
    }
    aggregate, per_holdout = helpers.review_metrics_frames(bundle)
    assert not aggregate.empty
    assert not per_holdout.empty
    acc = aggregate[(aggregate["label"] == "Overall") & (aggregate["metric"] == "accuracy")]["value"].iloc[0]
    assert acc == 0.7
    assert set(per_holdout["heldout_deployment"].astype(str)) == {
        "2021-04-17_mian-011",
        "2022-04-01_mian-013",
    }


def test_review_method_options_collects_observed_cluster_algorithmic_and_supervised():
    bundle = {
        "clustered_windows": pd.DataFrame(
            {
                "dataset_id": ["ds1", "ds1"],
                "deployment_id": ["dep1", "dep1"],
                "cluster_pass": ["k2", "k5"],
            }
        ),
        "supervised_predictions": pd.DataFrame(
            {
                "dataset_id": ["ds1", "ds1"],
                "deployment_id": ["dep1", "dep1"],
                "observed_label": ["WAKE", "SLEEP"],
                "variant_id": ["rf_full", "rf_depth"],
                "variant_label": ["RF full", "RF depth"],
                "variant_rank": [0, 1],
            }
        ),
        "algorithmic_segments": pd.DataFrame(
            {
                "dataset_id": ["ds1"],
                "deployment_id": ["dep1"],
                "algorithmic_method": ["find_rest"],
            }
        ),
    }
    options = helpers.review_method_options(bundle, "ds1", "dep1")
    labels = [row["label"] for row in options]
    assert "Observed labels" in labels
    assert "K2 clusters" in labels
    assert "K5 clusters" in labels
    assert "RF full" in labels
    assert "Algorithmic (find_rest)" in labels


def test_localize_datetimes_handles_naive_and_aware_values():
    naive = pd.Series(pd.date_range("2021-01-01", periods=2, freq="1h"))
    localized = helpers.localize_datetimes(naive, "America/Los_Angeles")
    assert str(localized.dt.tz) == "America/Los_Angeles"

    aware = pd.Series(pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC"))
    converted = helpers.localize_datetimes(aware, "America/Los_Angeles")
    assert str(converted.dt.tz) == "America/Los_Angeles"
