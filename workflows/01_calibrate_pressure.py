# Run with shell command: python pyologger/workflows/01_calibrate_pressure.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
# Zero offset correction: calibrate pressure sensor
import os
import pickle
import argparse
import pandas as pd

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.event_manager import *
from pyologger.plot_data.plotter import *
from pyologger.calibrate_data.zoc import *
from pyologger.io_operations.base_exporter import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, channel_mapping_path = load_configuration()

# Load data with optional arguments
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, config_manager = select_and_load_deployment(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, config_manager = select_and_load_deployment(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')

# Load key time points
timezone = data_pkl.deployment_info.get('Time Zone', 'UTC')
settings = config_manager.get_from_config(variable_names=["earliest_common_start_time", "latest_common_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
OVERLAP_START_TIME = pd.Timestamp(settings["earliest_common_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["latest_common_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)

# Confirm the values or raise an error if any are missing
if None in {OVERLAP_START_TIME, OVERLAP_END_TIME, ZOOM_WINDOW_START_TIME, ZOOM_WINDOW_END_TIME}:
    raise ValueError("One or more required time values were not found in the config file.")

current_processing_step = "Processing Step 01 IN PROGRESS."
config_manager.add_to_config("current_processing_step", current_processing_step)

# Print sampling frequency info
for logger_id, info in data_pkl.logger_info.items():
    sampling_frequency = info.get("datetime_metadata", {}).get("fs", None)
    print(f"📡 Sampling frequency for {logger_id}: {sampling_frequency} Hz" if sampling_frequency else f"⚠ No sampling frequency available for {logger_id}")

# Step 4: Load depth and temperature data
depth_data = data_pkl.sensor_data["pressure"]["pressure"]
depth_datetime = data_pkl.sensor_data["pressure"]["datetime"]
depth_fs = data_pkl.sensor_info["pressure"]["sampling_frequency"]
temp_data = data_pkl.sensor_data['temperature']['temp']
temp_fs = data_pkl.sensor_info['temperature']['sampling_frequency']

# **Step 2: Load Configuration Parameters**
dive_detection_settings = config_manager.get_from_config(
    variable_names=[
        "first_deriv_threshold", "min_duration", "depth_threshold",
        "apply_temp_correction", "min_depth_threshold", "dive_duration_threshold",
        "smoothing_window", "downsampled_sampling_rate", "baseline_adjust"
    ],
    section="dive_detection_settings"
)

# Default settings
default_settings = {
    "first_deriv_threshold": 0.1, "min_duration": 60, "depth_threshold": 5,
    "apply_temp_correction": False, "min_depth_threshold": 0.5,
    "dive_duration_threshold": 10, "smoothing_window": 5,
    "downsampled_sampling_rate": 1, "baseline_adjust": 0.0
}

# If settings are missing or None, initialize them
if dive_detection_settings is None:
    dive_detection_settings = default_settings
elif any(v is None for v in dive_detection_settings.values()):  
    # Fill in only missing/None values
    dive_detection_settings = {k: v if dive_detection_settings.get(k) is not None else default_settings[k] for k, v in dive_detection_settings.items()}

# Save if changes were made
config_manager.add_to_config(entries=dive_detection_settings, section="dive_detection_settings")

# Print to confirm
print(f"✅ Loaded downsampled_sampling_rate: {dive_detection_settings['downsampled_sampling_rate']}")


# Step 6: Process depth data - Downsample, smooth, adjust baseline, and calculate first derivative
depth_processing_params = {
    "original_sampling_rate": depth_fs,
    "downsampled_sampling_rate": int(dive_detection_settings["downsampled_sampling_rate"]),
    "baseline_adjust": dive_detection_settings["baseline_adjust"]  # New parameter
}

first_derivative, downsampled_depth = smooth_downsample_derivative(depth_data, **depth_processing_params)

# Adjust datetime indexing based on the new downsample rate
depth_downsampled_datetime = depth_datetime.iloc[::int(depth_fs / dive_detection_settings["downsampled_sampling_rate"])]

# Ensure indexing does not go out of bounds
if len(depth_downsampled_datetime) > len(downsampled_depth):
    depth_downsampled_datetime = depth_downsampled_datetime[:len(downsampled_depth)]

# Print summary of processing
print(f"✅ Depth processing complete: Downsampled to {dive_detection_settings['downsampled_sampling_rate']} Hz")
print(f"✅ Baseline adjustment applied: {dive_detection_settings['baseline_adjust']} meters")

# Detect flat chunks (potential surface intervals)
flat_chunk_params = {
    "depth": downsampled_depth,
    "datetime_data": depth_downsampled_datetime,
    "first_derivative": first_derivative,
    "threshold": dive_detection_settings["first_deriv_threshold"],
    "min_duration": dive_detection_settings["min_duration"],
    "depth_threshold": dive_detection_settings["depth_threshold"],
    "original_sampling_rate": depth_fs,
    "downsampled_sampling_rate": dive_detection_settings["downsampled_sampling_rate"]
}
flat_chunks = detect_flat_chunks(**flat_chunk_params)

# Apply zero offset correction
zoc_params = {
    "depth": downsampled_depth,
    "temp": temp_data.values if temp_data is not None else None,
    "flat_chunks": flat_chunks
}
corrected_depth_temp, corrected_depth_no_temp, depth_correction = apply_zero_offset_correction(**zoc_params)

# Choose corrected depth based on temperature correction setting
if dive_detection_settings["apply_temp_correction"]:
    corrected_depth = corrected_depth_temp
else:
    corrected_depth = corrected_depth_no_temp

# Detect dives in the corrected depth data
dive_detection_params = {
    "depth_series": corrected_depth,
    "datetime_data": depth_downsampled_datetime,
    "min_depth_threshold": dive_detection_settings["min_depth_threshold"],
    "sampling_rate": dive_detection_settings["downsampled_sampling_rate"],
    "duration_threshold": dive_detection_settings["dive_duration_threshold"],
    "smoothing_window": dive_detection_settings["smoothing_window"]
}
dives = find_dives(**dive_detection_params)
corrected_depth = enforce_surface_before_after_dives(corrected_depth, depth_downsampled_datetime, dives)

# Calculate dive duration in seconds
dives['dive_duration'] = (dives['end_time'] - dives['start_time']).dt.total_seconds()

# Log transformations
transformation_log = [
    f"downsampled_{dive_detection_settings['downsampled_sampling_rate']}Hz",
    f"smoothed_{dive_detection_settings['smoothing_window']}s",
    f"ZOC_settings__first_deriv_threshold_{dive_detection_settings['first_deriv_threshold']}mps__min_duration_{dive_detection_settings['min_duration']}s__depth_threshold_{dive_detection_settings['depth_threshold']}m",
    f"DIVE_detection_settings__min_depth_threshold_{dive_detection_settings['min_depth_threshold']}m__dive_duration_threshold_{dive_detection_settings['dive_duration_threshold']}s__smoothing_window_{dive_detection_settings['smoothing_window']}"
]

# Outputs
print(f"✅ {len(flat_chunks)} surface intervals detected.")
print(f"✅ {len(dives)} dives detected.")
print("📖 Transformation Log:", transformation_log)

# Step 11: Save processed data
data_pkl.event_data = create_state_event(
    state_df=dives,
    key="dive",
    value_column="max_depth",
    start_time_column="start_time",
    duration_column="dive_duration",
    description="dive_start",
    existing_events=data_pkl.event_data
)

data_pkl.event_info = list(data_pkl.event_data["key"].unique())

# Step 12: Store derived depth data
depth_df = pd.DataFrame({"datetime": depth_downsampled_datetime, "depth": corrected_depth})

derived_from_sensors = ["pressure"]
original_name = "Temp-corrected Depth (m)" if dive_detection_settings["apply_temp_correction"] else "Corrected Depth (m)"

derived_info = {
    "channels": ["depth"],
    "metadata": {
        "depth": {
            "original_name": original_name,
            "unit": "m",
            "sensor": "pressure"
        }
    },
    "derived_from_sensors": derived_from_sensors + (["temperature"] if dive_detection_settings["apply_temp_correction"] else []),
    "transformation_log": transformation_log + (["temperature_correction"] if dive_detection_settings["apply_temp_correction"] else [])
}

data_pkl.derived_data["depth"] = depth_df
data_pkl.derived_info["depth"] = derived_info

# Load key time points
timezone = data_pkl.deployment_info.get('Time Zone', 'UTC')
settings = config_manager.get_from_config(variable_names=["earliest_common_start_time", "latest_common_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
OVERLAP_START_TIME = pd.Timestamp(settings["earliest_common_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["latest_common_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)

fig = plot_tag_data_interactive(
    data_pkl=data_pkl,
    sensors=['pressure'],
    derived_data_signals=['depth'],
    time_range=(depth_downsampled_datetime.min(), depth_downsampled_datetime.max()),
    note_annotations={"dive": {"signal": "depth", "symbol": "triangle-down", "color": "blue"}},
    state_annotations={"dive": {"signal": "depth", "color": "rgba(150, 150, 150, 0.3)"}},
    color_mapping_path=color_mapping_path,
    target_sampling_rate=1
)
fig.show()

config_manager.add_to_config("current_processing_step", "Processing Step 01: Pressure sensor calibration complete.")
with open(pkl_path, "wb") as file:
    pickle.dump(data_pkl, file)

# exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
# netcdf_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_step01.nc') # Define the export path
# exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format

print("✅ Data processing complete. Pickle file updated.")