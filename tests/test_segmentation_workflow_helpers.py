import importlib.util
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "dash" / "minimal_interactive" / "segmentation_helpers.py"
spec = importlib.util.spec_from_file_location("segmentation_helpers_test", MODULE_PATH)
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
assert spec.loader is not None
spec.loader.exec_module(helpers)


def test_normalize_algorithmic_cfg_migrates_legacy_flag_and_defaults():
    normalized = helpers.normalize_algorithmic_cfg(
        {
            "sign_consistency_required": False,
            "thresholds": {"min_duration_s": 240.0},
        }
    )
    assert "sign_consistency_required" not in normalized
    assert normalized["end_segments_upon_d1_sign_change"] is False
    assert normalized["thresholds"]["min_duration_s"] == 240.0
    assert normalized["covariates"] == {"intrinsic": {}, "extrinsic": {}}
    assert normalized["context_filters"] == []
    assert normalized["debug"]["persist_intermediate_channels"] is True


def test_build_find_rest_workflow_preset_is_base_detector_only():
    workflow = helpers.build_find_rest_workflow_preset()
    node_ids = {node["id"] for node in workflow["nodes"]}
    assert "positive_buoyancy" not in node_ids
    assert "negative_buoyancy_criteria" not in node_ids
    assert "sign_split_negative" not in node_ids
    assert {"root", "surface_long", "drift_candidate", "drift_long", "flat_check", "plausible_drift", "long_drift"}.issubset(node_ids)
    derived_ids = [row["id"] for row in workflow["derived_channels"]]
    assert derived_ids == [
        "depth_std_m",
        "depth_d1_ms",
        "depth_d2_ms2",
        "depth_d1_end_ms",
        "drift_rate_ms",
        "smoothed_drift_rate_ms",
        "segment_duration_s",
    ]


def test_workflow_runtime_support_report_separates_base_and_context_rules():
    workflow = helpers.build_find_rest_workflow_preset()
    runtime = helpers.workflow_preset_to_algorithmic_cfg(workflow, fallback_cfg={})
    runtime["context_filter_definitions"] = {
        "known_filter": {
            "enabled": True,
            "action_on_fail": "reject",
        }
    }
    runtime["context_filters"] = ["missing_filter"]
    workflow["runtime_params"] = runtime
    report = helpers.workflow_runtime_support_report(workflow)
    assert report["base_detector_valid"] is True
    assert report["context_rules_valid"] is False
    assert report["runtime_supported"] is False
    assert any("missing_filter" in msg for msg in report["messages"])


def test_load_segmentation_workflow_presets_normalizes_find_rest_graph(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_payload = {
        "segmentation_workflows": {
            "find_rest": {
                "workflow_name": "find_rest",
                "display_name": "Find Rest",
                "runtime_params": {
                    "method_name": "find_rest",
                    "source_channel": "depth.depth",
                    "sign_consistency_required": True,
                    "context_filters": ["known_filter"],
                    "context_filter_definitions": {
                        "known_filter": {"enabled": True, "action_on_fail": "reject"}
                    },
                },
                "nodes": [
                    {"id": "root", "type": "filter_group", "label": "root"},
                    {"id": "positive_buoyancy", "type": "decision", "label": "Positively buoyant?"},
                ],
                "edges": [{"source": "root", "target": "positive_buoyancy", "branch": "yes"}],
            }
        }
    }
    cfg_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False))
    workflows = helpers.load_segmentation_workflow_presets(str(cfg_path))
    workflow = workflows["find_rest"]
    node_ids = {node["id"] for node in workflow["nodes"]}
    assert "positive_buoyancy" not in node_ids
    assert "plausible_drift" in node_ids
    runtime = workflow["runtime_params"]
    assert runtime["end_segments_upon_d1_sign_change"] is True
    assert runtime["context_filters"] == ["known_filter"]
    assert runtime["context_filter_definitions"]["known_filter"]["action_on_fail"] == "reject"


def test_apply_context_rule_sections_round_trips_named_filters():
    base = helpers.normalize_algorithmic_cfg({})
    updated = helpers.apply_context_rule_sections_to_algo_cfg(
        base,
        intrinsic={"inferred_buoyancy": {"enabled": True, "provider": "smoothed_segment_metric"}},
        extrinsic={},
        context_filter_definitions={"aligned": {"enabled": True, "action_on_fail": "reject"}},
        context_filters=["aligned"],
        debug={"write_context_review_table": True, "persist_context_helper_channels": False, "persist_intermediate_channels": True},
    )
    sections = helpers.context_rule_sections_from_algo_cfg(updated)
    assert "inferred_buoyancy" in sections["intrinsic"]
    assert sections["context_filters"] == ["aligned"]
    assert sections["debug"]["write_context_review_table"] is True
    assert helpers.validate_context_rule_sections(updated) == []
