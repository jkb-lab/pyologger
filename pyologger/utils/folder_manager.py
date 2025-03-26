import os
import yaml
import pickle
import streamlit as st
from dotenv import load_dotenv

from pyologger.utils.param_manager import ParamManager

def load_configuration():
    load_dotenv()
    CONFIG_PATH = os.getenv("CONFIG_PATH")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    data_dir = config["paths"]["local_private_data"]
    color_mapping_path = os.path.join(config["paths"]["local_repo_path"], "color_mappings.json")
    montage_path = os.path.join(config["paths"]["local_repo_path"], "montage_log.json")
    return config, data_dir, color_mapping_path, montage_path

def select_folder(base_dir, prompt="Select a folder:"):
    """Prompts the user to select a dataset or deployment folder."""
    folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith("00_")])
    
    if not folders:
        raise ValueError(f"No valid folders found in {base_dir}.")
    
    print(prompt)
    for i, folder in enumerate(folders):
        print(f"{i}: {folder}")
    
    selected_index = int(input("Enter the number of the folder you want to select: "))
    
    if 0 <= selected_index < len(folders):
        return os.path.join(base_dir, folders[selected_index])
    else:
        raise ValueError("Invalid selection. Please restart and choose a valid folder.")
    
def select_and_load_deployment_streamlit(data_dir):
    """Streamlit-based deployment selection with dataset and deployment filtering."""
    
    # Get available datasets (excluding those starting with "00_")
    datasets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("00_")])
    if not datasets:
        st.error("❌ No valid datasets found.")
        st.stop()

    # Dataset selection
    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets, key="dataset_selection")
    dataset_folder = os.path.join(data_dir, selected_dataset)

    # Get available deployments (excluding those starting with "00_")
    deployments = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d)) and not d.startswith("00_")])
    if not deployments:
        st.error("❌ No valid deployments found in the selected dataset.")
        st.stop()

    # Deployment selection
    selected_deployment = st.sidebar.selectbox("Select Deployment", deployments, key="deployment_selection")
    deployment_folder = os.path.join(dataset_folder, selected_deployment)

    # Extract metadata
    deployment_id = selected_deployment
    dataset_id = selected_dataset
    try:
        animal_id = deployment_id.split("_")[1]
    except IndexError:
        st.error(f"❌ Unable to extract animal ID from deployment ID: {deployment_id}")
        st.stop()

    # Load data.pkl
    pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
    if not os.path.exists(pkl_path):
        st.error(f"❌ Data pickle file not found: {pkl_path}")
        st.stop()

    with open(pkl_path, "rb") as file:
        data_pkl = pickle.load(file)

    # Initialize ParamManager
    param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)

    return animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager

def select_and_load_deployment(data_dir, dataset_id=None, deployment_id=None):
    """Command-line or function-based deployment selection. Allows selection via index or folder name."""

    # Get available datasets (excluding those starting with "00_")
    datasets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("00_")])
    if not datasets:
        raise ValueError("❌ No valid datasets found.")

    # Allow selection by either index or folder name
    if dataset_id is None:
        print("\nAvailable Datasets:")
        for i, dataset in enumerate(datasets):
            print(f"{i}: {dataset}")
        dataset_input = input("Enter dataset index or name: ")

        if dataset_input.isdigit() and 0 <= int(dataset_input) < len(datasets):
            dataset_id = datasets[int(dataset_input)]
        elif dataset_input in datasets:
            dataset_id = dataset_input
        else:
            raise ValueError("❌ Invalid dataset selection.")

    dataset_folder = os.path.join(data_dir, dataset_id)

    # Get available deployments (excluding those starting with "00_")
    deployments = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d)) and not d.startswith("00_")])
    if not deployments:
        raise ValueError(f"❌ No valid deployments found in {dataset_id}.")

    # Allow selection by either index or folder name
    if deployment_id is None:
        print("\nAvailable Deployments:")
        for i, deployment in enumerate(deployments):
            print(f"{i}: {deployment}")
        deployment_input = input("Enter deployment index or name: ")

        if deployment_input.isdigit() and 0 <= int(deployment_input) < len(deployments):
            deployment_id = deployments[int(deployment_input)]
        elif deployment_input in deployments:
            deployment_id = deployment_input
        else:
            raise ValueError("❌ Invalid deployment selection.")

    deployment_folder = os.path.join(dataset_folder, deployment_id)

    # Extract metadata
    try:
        animal_id = deployment_id.split("_")[1]
    except IndexError:
        raise ValueError(f"❌ Unable to extract animal ID from deployment ID: {deployment_id}")

    # Load data.pkl
    pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"❌ Data pickle file not found: {pkl_path}")

    with open(pkl_path, "rb") as file:
        data_pkl = pickle.load(file)

    # Initialize ParamManager
    param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)

    return animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager
