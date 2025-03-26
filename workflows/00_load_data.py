# Run with shell command: python pyologger/workflows/00_load_data.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
import os
import re
import pickle
import argparse
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# Import pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.param_manager import ParamManager
from pyologger.load_data.datareader import DataReader
from pyologger.load_data.metadata import Metadata
from pyologger.io_operations.base_exporter import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Load data")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load important file paths and configurations
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Step 1: Select dataset folder
if args.dataset:
    dataset_folder = os.path.join(data_dir, args.dataset)
else:
    print("No dataset provided. Selecting from available datasets.")
    dataset_folder = select_folder(data_dir, "Select a dataset folder:")

if not os.path.exists(dataset_folder):
    raise ValueError(f"❌ Dataset folder {dataset_folder} not found!")

overwrite_metadata = False
# Define the path to the metadata pickle file
metadata_pickle_path = os.path.join(data_dir, "00_Metadata/metadata_snapshot.pkl")

# Check if the metadata pickle file exists and is more than 5 days old
if overwrite_metadata or not os.path.exists(metadata_pickle_path) or (datetime.now() - datetime.fromtimestamp(os.path.getmtime(metadata_pickle_path))) > timedelta(days=14):
    
    metadata = Metadata()

    # Save database variables
    deployment_db = metadata.get_metadata("deployment_DB")
    logger_db = metadata.get_metadata("logger_DB")
    recording_db = metadata.get_metadata("recording_DB")
    animal_db = metadata.get_metadata("animal_DB")
    dataset_db = metadata.get_metadata("dataset_DB")
    procedure_db = metadata.get_metadata("procedure_DB")
    observation_db = metadata.get_metadata("observation_DB")
    collaborator_db = metadata.get_metadata("collaborator_DB")
    location_db = metadata.get_metadata("location_DB")
    montage_db = metadata.get_metadata("montage_DB")
    sensor_db = metadata.get_metadata("sensor_DB")
    attachment_db = metadata.get_metadata("attachment_DB")
    originalchannel_db = metadata.get_metadata("originalchannel_DB")
    standardizedchannel_db = metadata.get_metadata("standardizedchannel_DB")
    derivedsignal_db = metadata.get_metadata("derivedsignal_DB")
    derivedchannel_db = metadata.get_metadata("derivedchannel_DB")

    # Get the relations map
    relations_map = metadata.relations_map

    # Define the path to save the relations map
    relations_map_path = os.path.join(config['paths']['local_repo_path'], 'relations_map.json')

    # Save the relations map as a JSON file
    with open(relations_map_path, 'w') as file:
        json.dump(relations_map, file, indent=4)

    print(f"Relations map saved at: {relations_map_path}")

    ## OPTIONAL: Save metadata snapshot as a pickle file
    metadata.notion = None  # Temporarily remove the Notion client
    # Save metadata snapshot as a pickle file
    with open(metadata_pickle_path, "wb") as file:
        pickle.dump(metadata, file)

    print(f"Metadata snapshot saved at: {metadata_pickle_path}")
else:
    print(f"Recent metadata snapshot loaded, already present at: {metadata_pickle_path}")
    
    # Load the metadata snapshot from the pickle file
    with open(metadata_pickle_path, "rb") as file:
        metadata = pickle.load(file)
    
    # Save database variables
    deployment_db = metadata.get_metadata("deployment_DB")
    logger_db = metadata.get_metadata("logger_DB")
    recording_db = metadata.get_metadata("recording_DB")
    animal_db = metadata.get_metadata("animal_DB")
    dataset_db = metadata.get_metadata("dataset_DB")
    procedure_db = metadata.get_metadata("procedure_DB")
    observation_db = metadata.get_metadata("observation_DB")
    collaborator_db = metadata.get_metadata("collaborator_DB")
    location_db = metadata.get_metadata("location_DB")
    montage_db = metadata.get_metadata("montage_DB")
    sensor_db = metadata.get_metadata("sensor_DB")
    attachment_db = metadata.get_metadata("attachment_DB")
    originalchannel_db = metadata.get_metadata("originalchannel_DB")
    standardizedchannel_db = metadata.get_metadata("standardizedchannel_DB")
    derivedsignal_db = metadata.get_metadata("derivedsignal_DB")
    derivedchannel_db = metadata.get_metadata("derivedchannel_DB")

# Step 3: Select deployment folder
if args.deployment:
    deployment_id = args.deployment
    deployment_folder = os.path.join(dataset_folder, deployment_id)
    print(f"✅ Using provided deployment ID: {deployment_id}")
else:
    print("No deployment provided. Selecting from available deployments.")
    deployment_folder = select_folder(dataset_folder, "Select a deployment folder:")

if not os.path.exists(deployment_folder):
    raise ValueError(f"❌ Deployment folder {deployment_folder} not found!")

# Extract deployment_id and animal_id from the folder name
match = re.match(r"(\d{4}-\d{2}-\d{2}_[a-z]{4}-\d{3})", os.path.basename(deployment_folder), re.IGNORECASE)
if match:
    deployment_id = match.group(1)  # Extract YYYY-MM-DD_animalID
    animal_id = deployment_id.split("_")[1]  # Extract animal ID
    print(f"✅ Extracted deployment ID: {deployment_id}, Animal ID: {animal_id}")
else:
    raise ValueError(f"❌ Unable to extract deployment ID from folder: {deployment_folder}")

deployment_info, loggers_used = metadata.extract_essential_metadata(deployment_id)

# Step 5: Initialize DataReader with dataset folder, deployment ID, and optional data subfolder
data_pkl = DataReader(dataset_folder=dataset_folder, deployment_id=deployment_id, data_subfolder="01_raw-data", montage_path=montage_path)

# Step 6: Initialize config manager
param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)
param_manager.add_to_config("current_processing_step", "Processing Step 00: Data import pending.")
param_manager.export_config()

overwrite_data = False
pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
if os.path.exists(pkl_path) and not overwrite_data:
    with open(pkl_path, "rb") as f:
        data_pkl = pickle.load(f)
    print(f"📦 Loaded processed DataReader object from: {pkl_path}")
else:
    data_pkl = DataReader(
        dataset_folder=dataset_folder,
        deployment_id=deployment_id,
        data_subfolder="01_raw-data",
        montage_path=montage_path
    )
    param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)
    param_manager.add_to_config("current_processing_step", "Processing Step 00: Data import pending.")

    data_pkl.read_files(
        deployment_info=deployment_info,
        loggers_used=loggers_used,
        save_parq=False,
        save_netcdf=True
    )

# Get timezone
timezone = data_pkl.deployment_info.get("Time Zone", "UTC")

# Load time settings
time_settings = param_manager.get_from_config(
    ["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],
    section="settings"
)

if time_settings:
    print("Time settings present.")
# If any required time settings are missing, compute and update them
if not any(v is None for v in time_settings.values()):
    print("Time settings not empty.")
else:
    print("Adding timestamps to config.")
    zoom_time_window = 5  # minutes

    # Extract start and end times for all sensors
    start_times = [df['datetime'].min() for df in data_pkl.sensor_data.values()]
    end_times = [df['datetime'].max() for df in data_pkl.sensor_data.values()]

    # Compute common start, end, and zoom window
    overlap_start_time = max(start_times)
    overlap_end_time = min(end_times)
    midpoint = overlap_start_time + (overlap_end_time - overlap_start_time) / 2
    zoom_window_start, zoom_window_end = midpoint - timedelta(minutes=zoom_time_window / 2), midpoint + timedelta(minutes=zoom_time_window / 2)

    # Update settings
    time_settings = {
        "overlap_start_time": str(overlap_start_time),
        "overlap_end_time": str(overlap_end_time),
        "zoom_window_start_time": str(zoom_window_start),
        "zoom_window_end_time": str(zoom_window_end),
    }
    param_manager.add_to_config(entries=time_settings, section="settings")

if any(v is None for v in time_settings.values()):
    print("YES")
time_settings

# Check if selected start and end times exist in the config file
truncate_times = param_manager.get_from_config(
    ["selected_start_time", "selected_end_time"],
    section="settings"
)

if not any(v is None for v in truncate_times.values()):
    print("Truncating with provided cropping times.")
    # Update overlap window with selected range
    OVERLAP_START_TIME = pd.Timestamp(truncate_times['selected_start_time']).tz_convert(timezone)
    OVERLAP_END_TIME = pd.Timestamp(truncate_times['selected_end_time']).tz_convert(timezone)

    # Truncate sensor data
    for sensor, df in data_pkl.sensor_data.items():
        # Truncate based on selected time range
        truncated_df = df[(df.iloc[:, 0] >= OVERLAP_START_TIME) & (df.iloc[:, 0] <= OVERLAP_END_TIME)].copy()
        data_pkl.sensor_data[sensor] = truncated_df  # Save truncated version to new variable

    # Recalculate Zoom Window (5-minute window in the middle)
    midpoint = OVERLAP_START_TIME + (OVERLAP_END_TIME - OVERLAP_START_TIME) / 2
    ZOOM_WINDOW_START_TIME = midpoint - timedelta(minutes=2.5)
    ZOOM_WINDOW_END_TIME = midpoint + timedelta(minutes=2.5)

    # Save new time settings
    time_settings_update = {
        "overlap_start_time": str(OVERLAP_START_TIME),
        "overlap_end_time": str(OVERLAP_END_TIME),
        "zoom_window_start_time": str(ZOOM_WINDOW_START_TIME),
        "zoom_window_end_time": str(ZOOM_WINDOW_END_TIME)
    }
    param_manager.add_to_config(entries=time_settings_update, section="settings")

    pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)

# Step 8: Update processing step
param_manager.add_to_config("current_processing_step", "Processing Step 00: Data imported.")

exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
netcdf_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_step00.nc') # Define the export path
exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format
