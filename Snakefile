# Snakefile

configfile: "config.yaml"

# Extract dataset and deployment names from the config file
datasets = list(config["datasets"].keys())
deployments = [deployment for dataset in datasets for deployment in config["datasets"][dataset]["deployments"]]


# Define the final target rule
rule all:
    input:
        expand("data/{dataset}/{deployment}/outputs/{deployment}_step02.nc", dataset=datasets, deployment=deployments)

# Step 1: Load Data
rule load_data:
    output:
        "data/{dataset}/{deployment}/outputs/data.pkl"
    shell:
        "python workflows/00_load_data.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 2: Calibrate Pressure
rule calibrate_pressure:
    input:
        "data/{dataset}/{deployment}/outputs/data.pkl"
    output:
        "data/{dataset}/{deployment}/outputs/{deployment}_step01.nc"
    shell:
        "python workflows/01_calibrate_pressure.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"

# Step 3: Calibrate Accelerometer & Magnetometer
rule calibrate_accmag:
    input:
        "data/{dataset}/{deployment}/outputs/{deployment}_step01.nc"
    output:
        "data/{dataset}/{deployment}/outputs/{deployment}_step02.nc"
    shell:
        "python workflows/02_calibrate_accmag.py --dataset {wildcards.dataset} --deployment {wildcards.deployment} && touch {output}"