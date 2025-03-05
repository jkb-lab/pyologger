# Run with shell command: python pyologger/workflows/00_load_data.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
import os
import re
import pickle
import argparse
import pandas as pd
import xarray as xr
from datetime import timedelta

# Import pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.json_manager import ConfigManager
from pyologger.load_data.datareader import DataReader
from pyologger.load_data.metadata import Metadata

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Load data")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load important file paths and configurations
config, data_dir, color_mapping_path, channel_mapping_path = load_configuration()

# Step 1: Select dataset folder
if args.dataset:
    dataset_folder = os.path.join(data_dir, args.dataset)
else:
    print("No dataset provided. Selecting from available datasets.")
    dataset_folder = select_folder(data_dir, "Select a dataset folder:")

if not os.path.exists(dataset_folder):
    raise ValueError(f"❌ Dataset folder {dataset_folder} not found!")

# Step 2: Load metadata
metadata = Metadata()
metadata.fetch_databases(verbose=False)
metadata.find_relations(verbose=False)

# Fetch databases
deployment_db = metadata.get_metadata("deployment_DB")
logger_db = metadata.get_metadata("logger_DB")
recording_db = metadata.get_metadata("recording_DB")
animal_db = metadata.get_metadata("animal_DB")
dataset_db = metadata.get_metadata("dataset_DB")

# Combine DataFrames into a dictionary
metadata_databases = {
    'deployment_db': deployment_db,
    'logger_db': logger_db,
    'recording_db': recording_db,
    'animal_db': animal_db,
    'dataset_db': dataset_db
}

# Save metadata snapshot
metadata_pickle_path = os.path.join(dataset_folder, "metadata_snapshot.pkl")
with open(metadata_pickle_path, "wb") as file:
    pickle.dump(metadata_databases, file)

print(f"📂 Metadata snapshot saved at: {metadata_pickle_path}")

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

# Step 5: Initialize DataReader with dataset folder, deployment ID, and optional data subfolder
datareader = DataReader(dataset_folder=dataset_folder, deployment_id=deployment_id, data_subfolder="01_raw-data")

# Step 6: Initialize config manager
config_manager = ConfigManager(deployment_folder=deployment_folder, deployment_id=deployment_id)
config_manager.add_to_config("current_processing_step", "Processing Step 00: Data import pending.")

# Step 7: Read deployment files
datareader.read_files(
    metadata,
    save_csv=False,
    save_parq=True,
    save_edf=False,
    custom_mapping_path=channel_mapping_path,
    save_netcdf=True,
)

data_pkl = datareader
# Get timezone
timezone = data_pkl.deployment_info.get("Time Zone", "UTC")

# Load time settings
time_settings = config_manager.get_from_config(
    ["latest_common_start_time", "earliest_common_end_time", "zoom_window_start_time", "zoom_window_end_time"],
    section="settings"
)

# If any required time settings are missing, compute and update them
if not time_settings or any(v is None for v in time_settings.values()):
    zoom_time_window = 5  # minutes

    # Extract start and end times for all sensors
    start_times = [info["sensor_start_datetime"] for info in data_pkl.sensor_info.values()]
    end_times = [info["sensor_end_datetime"] for info in data_pkl.sensor_info.values()]

    # Compute common start, end, and zoom window
    earliest_common_start = max(start_times)
    latest_common_end = min(end_times)
    midpoint = earliest_common_start + (latest_common_end - earliest_common_start) / 2
    zoom_window_start, zoom_window_end = midpoint - timedelta(minutes=zoom_time_window / 2), midpoint + timedelta(minutes=zoom_time_window / 2)

    # Update settings
    time_settings = {
        "earliest_common_start_time": str(earliest_common_start),
        "latest_common_end_time": str(latest_common_end),
        "zoom_time_window": zoom_time_window,
        "zoom_window_start_time": str(zoom_window_start),
        "zoom_window_end_time": str(zoom_window_end),
    }
    config_manager.add_to_config(entries=time_settings, section="settings")

# Convert timestamps and validate
try:
    OVERLAP_START_TIME = pd.Timestamp(time_settings["earliest_common_start_time"]).tz_convert(timezone)
    OVERLAP_END_TIME = pd.Timestamp(time_settings["latest_common_end_time"]).tz_convert(timezone)
    ZOOM_WINDOW_START_TIME = pd.Timestamp(time_settings["zoom_window_start_time"]).tz_convert(timezone)
    ZOOM_WINDOW_END_TIME = pd.Timestamp(time_settings["zoom_window_end_time"]).tz_convert(timezone)
except KeyError as e:
    raise ValueError(f"Missing time setting: {e}")

# Display values
print(f"OVERLAP_START_TIME: {OVERLAP_START_TIME}")
print(f"OVERLAP_END_TIME: {OVERLAP_END_TIME}")
print(f"ZOOM_START_TIME: {ZOOM_WINDOW_START_TIME}")
print(f"ZOOM_END_TIME: {ZOOM_WINDOW_END_TIME}")

# Step 8: Update processing step
config_manager.add_to_config("current_processing_step", "Processing Step 00: Data imported.")

# Step 9: Open NetCDF file
netcdf_path = os.path.join(deployment_folder, "outputs", f'{deployment_id}_00_processed.nc')
if os.path.exists(netcdf_path):
    data = xr.open_dataset(netcdf_path)
    print(f"📊 NetCDF file loaded: {netcdf_path}")
else:
    print(f"⚠ NetCDF file not found at {netcdf_path}.")

# Step 10: Inspect sampling frequencies from DataReader object
pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
if os.path.exists(pkl_path):
    with open(pkl_path, "rb") as file:
        data_pkl = pickle.load(file)

    for logger_id, info in data_pkl.logger_info.items():
        sampling_frequency = info.get("datetime_metadata", {}).get("fs", None)
        if sampling_frequency is not None:
            print(f"📡 Sampling frequency for {logger_id}: {sampling_frequency:.5f} Hz")
        else:
            print(f"⚠ No sampling frequency available for {logger_id}")
else:
    print(f"⚠ Data pickle file not found at {pkl_path}.")
