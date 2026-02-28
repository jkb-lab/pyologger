configfile: "config.yaml"

def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

# Extract dataset and deployment names from the config file
private_data_root = config["paths"]["local_private_data"]

def _step_marker(step):
    return (
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/"
        f".snakemake_markers/{{deployment}}_step{step:02d}.done"
    )

# Generate a list of dataset-specific deployment paths
dataset_deployment_pairs = []
for dataset, details in config["datasets"].items():
    for deployment in details["deployments"]:
        dataset_deployment_pairs.append((dataset, deployment))

if not dataset_deployment_pairs:
    raise ValueError("No dataset/deployment entries found in config['datasets'].")

first_dataset, first_deployment = dataset_deployment_pairs[0]
dash_port = int(config.get("dash", {}).get("port", 8061))

# Define the final target rule
rule all:
    input: # f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_output.nc"
        [f"{private_data_root}/{dataset}/{deployment}/outputs/{deployment}_output.nc"
         for dataset, deployment in dataset_deployment_pairs]

# Step 00: Load Data
rule load_data:
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/data.pkl"
    shell:
        "python workflows/00_load_data.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 01: Calibrate Pressure
rule calibrate_pressure:
    input:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/data.pkl"
    output:
        _step_marker(1)
    shell:
        "python workflows/01_calibrate_pressure.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 02: Calibrate Accelerometer & Magnetometer
rule calibrate_accmag:
    input:
        _step_marker(1)
    output:
        _step_marker(2)
    shell:
        "python workflows/02_calibrate_accmag.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 03: Convert to Animal Reference Frame
rule tag2animal:
    input:
        _step_marker(2)
    output:
        _step_marker(3)
    shell:
        "python workflows/03_tag2animal.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 04: Stroke detection
rule stroke_detect:
    input:
        _step_marker(3)
    output:
        _step_marker(4)
    shell:
        "python workflows/04_stroke_detect.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 05: Heartbeat detection
rule heartbeat_detect:
    input:
        _step_marker(4)
    output:
        _step_marker(5)
    shell:
        "python workflows/05_heartbeat_detect.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && mkdir -p $(dirname {output}) && touch {output}"

# Step 06: Export Data
rule export_data:
    input:
        _step_marker(5)
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_output.nc"
    shell:
        "python workflows/06_export_data.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Launch only the minimal interactive Dash app (no workflow processing).
rule dash_app:
    params:
        dataset=first_dataset,
        deployment=first_deployment,
        port=dash_port
    shell:
        "python dash/minimal_interactive/app.py --dataset {params.dataset} --deployment {params.deployment} --port {params.port}"
