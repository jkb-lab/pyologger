

def _target_to_marker(target):
    # Map shorthand target to the correct marker file
    mapping = {
        "segment_label_and_summarize": "08_plots.done",
        "summary": "08_summary.done",
        "supervised": "07_supervised.done",
        "unsupervised": "06_cluster.done",
        "feature_generation": "04_correlate.done",
    }
    return mapping.get(target, f"{target}.done")
configfile: "config.yaml"
import os
import sys
import shutil
import pathlib

import yaml
from pyologger.utils.segmentation_run_config import normalize_segmentation_run_cfg

# Use the project venv python so ete3/biopython are available regardless of the
# shell's active environment. Falls back to sys.executable if the venv isn't present.
_venv_python = pathlib.Path(workflow.basedir).parent / "venv" / "bin" / "python3"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

ACTIVE_SEGMENTATION_CONFIG = getattr(workflow, "overwrite_configfiles", None) or getattr(workflow, "configfiles", None) or ["config.yaml"]
ACTIVE_SEGMENTATION_CONFIG = ACTIVE_SEGMENTATION_CONFIG[0]
SEGMENTATION_RUNS_CONFIG = os.getenv("SEGMENTATION_RUNS_PATH") or str(
    pathlib.Path(ACTIVE_SEGMENTATION_CONFIG).resolve().with_name("segmentation_runs.yaml")
)
SEGMENTATION_CONTEXT_FILTER_PASS_MODE = os.getenv("SEGMENTATION_CONTEXT_FILTER_PASS_MODE", "full")

def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _load_yaml_file(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


if os.path.exists(SEGMENTATION_RUNS_CONFIG):
    segmentation_runs_payload = _load_yaml_file(SEGMENTATION_RUNS_CONFIG)
    if isinstance(segmentation_runs_payload.get("segmentation_runs"), dict):
        config["segmentation_runs"] = segmentation_runs_payload["segmentation_runs"]
    for shared_key in (
        "context_filter_definitions",
        "supervised_context_filter_definitions",
        "supervised_label_group_definitions",
    ):
        if isinstance(segmentation_runs_payload.get(shared_key), dict):
            config[shared_key] = segmentation_runs_payload[shared_key]

# Extract dataset and deployment names from the config file
private_data_root = config["paths"]["local_private_data"]
meta_analysis_data_root = config["paths"]["local_private_meta_analysis_data"]

def _step_marker(step):
    return (
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/"
        f".snakemake_markers/{{deployment}}_step{step:02d}.done"
    )

# Generate a list of dataset-specific deployment paths
dataset_deployment_pairs = []
for dataset, details in (config.get("datasets") or {}).items():
    deployments = []
    if isinstance(details, dict):
        deployments = details.get("deployments") or []
    if not isinstance(deployments, list):
        continue
    for deployment in deployments:
        if deployment:
            dataset_deployment_pairs.append((dataset, deployment))

if dataset_deployment_pairs:
    first_dataset, first_deployment = dataset_deployment_pairs[0]
else:
    first_dataset, first_deployment = "", ""
dash_port = int(config.get("dash", {}).get("port", 8061))
enable_segmentation_marker_invalidation = _as_bool(
    ((config.get("segmentation") or {}).get("invalidate_stale_markers", False))
)
force_segmentation_rerun = _as_bool(config.get("force_segmentation_rerun", False))
force_segmentation_rerun_runs_cfg = config.get("force_segmentation_rerun_runs", [])
if isinstance(force_segmentation_rerun_runs_cfg, str):
    force_segmentation_rerun_runs = {
        token.strip() for token in force_segmentation_rerun_runs_cfg.split(",") if token.strip()
    }
elif isinstance(force_segmentation_rerun_runs_cfg, list):
    force_segmentation_rerun_runs = {
        str(token).strip() for token in force_segmentation_rerun_runs_cfg if str(token).strip()
    }
else:
    force_segmentation_rerun_runs = set()
run_selection_cfg = config.get("run_selection") or {}
enable_processing_runs = _as_bool(run_selection_cfg.get("processing", True))
enable_segmentation_runs = _as_bool(run_selection_cfg.get("segmentation", False))
enable_make_map_runs = _as_bool(run_selection_cfg.get("make_map", False))
enable_make_phylogeny_runs = _as_bool(run_selection_cfg.get("make_phylogeny", False))


def _segmentation_runs():
    return list((config.get("segmentation_runs") or {}).keys())

def _segmentation_run_cfg(run_name):
    run_cfg = (config.get("segmentation_runs") or {}).get(run_name, {})
    if not isinstance(run_cfg, dict):
        return {}
    return normalize_segmentation_run_cfg(
        dict(run_cfg),
        shared_supervised_label_groups=dict(config.get("supervised_label_group_definitions") or {}),
    )


def _make_map_runs():
    return list((config.get("make_map_runs") or {}).keys())


def _make_phylogeny_runs():
    return list((config.get("make_phylogeny_runs") or {}).keys())


def _segmentation_base(run_name):
    run_cfg = _segmentation_run_cfg(run_name)
    override = run_cfg.get("output_root_override")
    if override:
        return override
    dataset_ids = list(run_cfg.get("dataset_ids") or [])
    analysis_id = run_cfg.get("analysis_id", run_name)
    if len(dataset_ids) == 1:
        return f"{private_data_root}/{dataset_ids[0]}/00_Meta-Analysis/segmentation/{analysis_id}"
    return f"{private_data_root}/00_Meta-Analysis/segmentation/{analysis_id}"


def _segmentation_scope_pairs(run_name):
    run_cfg = _segmentation_run_cfg(run_name)
    dataset_ids = list(run_cfg.get("dataset_ids") or [])
    deployment_filter = set(run_cfg.get("deployment_ids") or [])
    pairs = []
    for ds in dataset_ids:
        ds_path = os.path.join(private_data_root, ds)
        if not os.path.isdir(ds_path):
            continue
        deployments = sorted(
            d for d in os.listdir(ds_path)
            if os.path.isdir(os.path.join(ds_path, d)) and not d.startswith("00_")
        )
        if deployment_filter:
            deployments = [d for d in deployments if d in deployment_filter]
        for dep in deployments:
            pairs.append((ds, dep))
    return pairs


def _segmentation_expected_feature_outputs(run_name):
    base = _segmentation_base(run_name)
    return [
        os.path.join(base, "features", "by_deployment", f"{dataset}__{deployment}__features.parquet")
        for dataset, deployment in _segmentation_scope_pairs(run_name)
    ]


def _segmentation_expected_segment_outputs(run_name):
    base = _segmentation_base(run_name)
    return [
        os.path.join(base, "segments", "by_deployment", f"{dataset}__{deployment}__algorithmic_segments.parquet")
        for dataset, deployment in _segmentation_scope_pairs(run_name)
    ]


segmentation_run_names = _segmentation_runs()
segmentation_plot_markers = [f"{meta_analysis_data_root}/segmentation/{r}/14_summary.done" for r in segmentation_run_names]
make_map_run_names = _make_map_runs()
make_map_markers = [f".snakemake_maps/{r}/02_render.done" for r in make_map_run_names]
make_phylogeny_run_names = _make_phylogeny_runs()
make_phylogeny_markers = [f".snakemake_phylogenies/{r}/02_render.done" for r in make_phylogeny_run_names]
processing_output_targets = [
    f"{private_data_root}/{dataset}/{deployment}/outputs/{deployment}_output.nc"
    for dataset, deployment in dataset_deployment_pairs
]
default_run_targets = []
if enable_processing_runs:
    default_run_targets.extend(processing_output_targets)
if enable_segmentation_runs:
    default_run_targets.extend(segmentation_plot_markers)
if enable_make_map_runs:
    default_run_targets.extend(make_map_markers)
if enable_make_phylogeny_runs:
    default_run_targets.extend(make_phylogeny_markers)


def _invalidate_stale_segmentation_markers():
    """
    If users delete a run's output folder/artifacts manually, remove the
    corresponding .snakemake markers so the DAG reruns that run.
    """
    for run_name in segmentation_run_names:
        run_marker_dir = f"{meta_analysis_data_root}/segmentation/{run_name}"
        run_base = _segmentation_base(run_name)
        scope_pairs = _segmentation_scope_pairs(run_name)

        expected_artifacts = [
            os.path.join(run_base, "qc", "qc_channels.csv"),
            os.path.join(run_base, "features", "features_raw.parquet"),
            os.path.join(run_base, "features", "features_filtered.parquet"),
            os.path.join(run_base, "clustering", "clustered_windows.parquet"),
        ]
        expected_feature_outputs = _segmentation_expected_feature_outputs(run_name)
        expected_segment_outputs = _segmentation_expected_segment_outputs(run_name)
        missing_any = (not os.path.isdir(run_base)) or any(not os.path.exists(p) for p in expected_artifacts)
        partial_scope = bool(scope_pairs) and (
            any(not os.path.exists(p) for p in expected_feature_outputs)
            or any(not os.path.exists(p) for p in expected_segment_outputs)
        )
        if (missing_any or partial_scope) and os.path.isdir(run_marker_dir):
            reason = "missing output artifacts" if missing_any else "partial per-deployment segmentation outputs"
            print(f"[segmentation] invalidating stale markers for run '{run_name}' ({reason})")
            shutil.rmtree(run_marker_dir, ignore_errors=True)


def _force_invalidate_segmentation_markers():
    if not (force_segmentation_rerun or force_segmentation_rerun_runs):
        return

    target_runs = set(segmentation_run_names)
    if force_segmentation_rerun_runs:
        unknown = sorted(force_segmentation_rerun_runs - target_runs)
        if unknown:
            raise ValueError(
                f"Unknown force_segmentation_rerun_runs entries: {unknown}. "
                f"Known runs: {sorted(target_runs)}"
            )
        target_runs &= force_segmentation_rerun_runs

    if not target_runs:
        return

    for run_name in sorted(target_runs):
        run_marker_dir = f"{meta_analysis_data_root}/segmentation/{run_name}"
        if os.path.isdir(run_marker_dir):
            print(f"[segmentation] force invalidating markers for run '{run_name}'")
            shutil.rmtree(run_marker_dir, ignore_errors=True)
        else:
            print(f"[segmentation] force invalidation requested for run '{run_name}' (no marker dir present)")


if enable_segmentation_marker_invalidation:
    _invalidate_stale_segmentation_markers()
_force_invalidate_segmentation_markers()

# Define the final target rule
rule all:
    input:
        default_run_targets


rule processing_all:
    input:
        processing_output_targets


rule segmentation_all:
    input:
        segmentation_plot_markers


rule segmentation_features_all:
    input:
        [f"{meta_analysis_data_root}/segmentation/{r}/11_features.done" for r in segmentation_run_names]


rule segmentation_learning_all:
    input:
        segmentation_plot_markers


rule make_map_all:
    input:
        make_map_markers


rule make_maps:
    input:
        make_map_markers

# Processing rules are marker-based on purpose.
# Each workflow script now decides whether to reuse the latest existing NetCDF
# or emit a new step NetCDF only when data changed, while Snakemake tracks
# completion via lightweight marker files.

# Step 00: Load Data
rule load_data:
    output:
        _step_marker(0)
    shell:
        "python3 workflows/00_load_data.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 01: Calibrate Pressure
rule calibrate_pressure:
    input:
        _step_marker(0)
    output:
        _step_marker(1)
    shell:
        "python3 workflows/01_calibrate_pressure.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 02: Calibrate Accelerometer & Magnetometer
rule calibrate_accmag:
    input:
        _step_marker(1)
    output:
        _step_marker(2)
    shell:
        "python3 workflows/02_calibrate_accmag.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 03: Convert to Animal Reference Frame
rule tag2animal:
    input:
        _step_marker(2)
    output:
        _step_marker(3)
    shell:
        "python3 workflows/03_tag2animal.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 04: Stroke detection
rule stroke_detect:
    input:
        _step_marker(3)
    output:
        _step_marker(4)
    shell:
        "python3 workflows/04_stroke_detect.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 05: Heartbeat detection
rule heartbeat_detect:
    input:
        _step_marker(4)
    output:
        _step_marker(5)
    shell:
        "python3 workflows/05_heartbeat_detect.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 06: Export Data
rule export_data:
    input:
        _step_marker(5)
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_output.nc"
    shell:
        "python3 workflows/06_export_data.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 06b: Export signal and event data CSVs (optional, runs independently of the main pipeline)
rule export_csvs:
    input:
        _step_marker(5)
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_signal_data.csv",
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_event_data.csv"
    shell:
        "python3 workflows/06_export_data.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} --export-csvs"


rule export_csvs_all:
    input:
        expand(
            [
                f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_signal_data.csv",
                f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_event_data.csv",
            ],
            zip,
            dataset=[d for d, _ in dataset_deployment_pairs],
            deployment=[dep for _, dep in dataset_deployment_pairs],
        )


# Launch only the minimal interactive Dash app (no workflow processing).
rule dash_app:
    params:
        dataset="mile-adult-sese_vdr_argentina_RD-KM",
        deployment="2015-11-05_mile-011",
        port=dash_port
    shell:
        "python dash/integrated/integrated_dash.py --dataset {params.dataset} --deployment {params.deployment} --port {params.port}"


rule segmentation_qc:
    output:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/09_qc.done"
    params:
        qc_csv=lambda wildcards: f"{_segmentation_base(wildcards.run_name)}/qc/qc_channels.csv"
    shell:
        "SEGMENTATION_RUNS_PATH={SEGMENTATION_RUNS_CONFIG} python3 pyologger/analyze_data/segmentation_pipeline.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} qc --output {params.qc_csv} && mkdir -p $(dirname {output}) && touch {output}"


rule segmentation_algorithmic_segments:
    input:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/09_qc.done"
    output:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/10_algorithmic.done"
    params:
        merged_output=lambda wildcards: f"{_segmentation_base(wildcards.run_name)}/segments/algorithmic_segments.parquet"
    shell:
        "SEGMENTATION_RUNS_PATH={SEGMENTATION_RUNS_CONFIG} "
        "python3 workflows/10_algorithmic_segmentation.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} --output {params.merged_output} "
        "--context-filter-pass-mode {SEGMENTATION_CONTEXT_FILTER_PASS_MODE} "
        "--no-refresh-qc --no-refresh-cross-dataset-qc "
        "&& mkdir -p $(dirname {output}) && touch {output}"


rule segmentation_features:
    input:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/10_algorithmic.done"
    output:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/11_features.done"
    params:
        merged_output=lambda wildcards: f"{_segmentation_base(wildcards.run_name)}/features/features_filtered.parquet"
    shell:
        "SEGMENTATION_RUNS_PATH={SEGMENTATION_RUNS_CONFIG} python3 workflows/11_feature_generation.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} --output {params.merged_output} && mkdir -p $(dirname {output}) && touch {output}"


rule segmentation_cluster:
    input:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/11_features.done"
    output:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/12_unsupervised.done"
    shell:
        "SEGMENTATION_RUNS_PATH={SEGMENTATION_RUNS_CONFIG} python3 workflows/12_unsupervised_segmentation.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} && mkdir -p $(dirname {output}) && touch {output}"


rule segmentation_supervised:
    input:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/11_features.done"
    output:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/13_supervised.done"
    shell:
        "SEGMENTATION_RUNS_PATH={SEGMENTATION_RUNS_CONFIG} python3 workflows/13_supervised_segmentation.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} && mkdir -p $(dirname {output}) && touch {output}"


rule segmentation_plots:
    input:
        clustered=f"{meta_analysis_data_root}/segmentation/{{run_name}}/12_unsupervised.done",
        supervised=f"{meta_analysis_data_root}/segmentation/{{run_name}}/13_supervised.done"
    output:
        f"{meta_analysis_data_root}/segmentation/{{run_name}}/14_summary.done"
    shell:
        "SEGMENTATION_RUNS_PATH={SEGMENTATION_RUNS_CONFIG} python3 workflows/14_summary.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} --output /tmp/{wildcards.run_name}_plots_done.marker && mkdir -p $(dirname {output}) && touch {output}"


rule make_map_resolve:
    output:
        ".snakemake_maps/{run_name}/01_resolve.done"
    shell:
        "python3 workflows/20_make_map.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} resolve --output /tmp/{wildcards.run_name}_make_map_manifest.json && mkdir -p $(dirname {output}) && touch {output}"


rule make_map_render:
    input:
        ".snakemake_maps/{run_name}/01_resolve.done"
    output:
        ".snakemake_maps/{run_name}/02_render.done"
    params:
        summary_output=lambda wildcards: f"/tmp/{wildcards.run_name}_make_map_render_summary.json"
    shell:
        "python3 workflows/20_make_map.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} render --output {params.summary_output} && mkdir -p $(dirname {output}) && touch {output}"


rule make_phylogeny_all:
    input:
        make_phylogeny_markers


rule make_phylogeny_resolve:
    output:
        ".snakemake_phylogenies/{run_name}/01_resolve.done"
    params:
        python=PYTHON
    shell:
        "{params.python} workflows/21_make_phylogeny.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} resolve --output /tmp/{wildcards.run_name}_phylo_manifest.json && mkdir -p $(dirname {output}) && touch {output}"


rule make_phylogeny_render:
    input:
        ".snakemake_phylogenies/{run_name}/01_resolve.done"
    output:
        ".snakemake_phylogenies/{run_name}/02_render.done"
    params:
        python=PYTHON,
        summary_output=lambda wildcards: f"/tmp/{wildcards.run_name}_phylo_render_summary.json"
    shell:
        "{params.python} workflows/21_make_phylogeny.py --config {ACTIVE_SEGMENTATION_CONFIG} --run-name {wildcards.run_name} render --output {params.summary_output} && mkdir -p $(dirname {output}) && touch {output}"
