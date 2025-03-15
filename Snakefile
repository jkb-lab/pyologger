configfile: "config.yaml"

# Extract dataset and deployment names from the config file
private_data_root = config["paths"]["local_private_data"]

# Generate a list of dataset-specific deployment paths
dataset_deployment_pairs = []
for dataset, details in config["datasets"].items():
    for deployment in details["deployments"]:
        dataset_deployment_pairs.append((dataset, deployment))

# Define the final target rule
rule all:
    input:
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
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step01.nc"
    shell:
        "python workflows/01_calibrate_pressure.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 02: Calibrate Accelerometer & Magnetometer
rule calibrate_accmag:
    input:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step01.nc"
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step02.nc"
    shell:
        "python workflows/02_calibrate_accmag.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 03: Convert to Animal Reference Frame
rule tag2animal:
    input:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step02.nc"
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step03.nc"
    shell:
        "python workflows/03_tag2animal.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 04: Stroke detection
rule stroke_detect:
    input:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step03.nc"
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step04.nc"
    shell:
        "python workflows/04_stroke_detect.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 05: Heartbeat detection
rule heartbeat_detect:
    input:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step04.nc"
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step05.nc"
    shell:
        "python workflows/05_heartbeat_detect.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 06: Export Data
rule export_data:
    input:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_step05.nc"
    output:
        f"{private_data_root}/{{dataset}}/{{deployment}}/outputs/{{deployment}}_output.nc"
    shell:
        "python workflows/06_export_data.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"
