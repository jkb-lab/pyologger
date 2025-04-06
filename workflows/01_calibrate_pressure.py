# Run with shell command: python pyologger/workflows/01_calibrate_pressure.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
# Zero offset correction: calibrate pressure sensor
import os
import pickle
import argparse
import pandas as pd
import numpy as np

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.event_manager import *
from pyologger.plot_data.plotter import *
from pyologger.calibrate_data.zoc import *
from pyologger.io_operations.base_exporter import *
from pyologger.analyze_data.find_segments import *
from pyologger.analyze_data.analyze_segments import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Load data with optional arguments
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')

# Load key time points
timezone = data_pkl.deployment_info.get('Time Zone', 'UTC')
settings = param_manager.get_from_config(variable_names=["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
OVERLAP_START_TIME = pd.Timestamp(settings["overlap_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["overlap_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)

# Confirm the values or raise an error if any are missing
if None in {OVERLAP_START_TIME, OVERLAP_END_TIME, ZOOM_WINDOW_START_TIME, ZOOM_WINDOW_END_TIME}:
    raise ValueError("One or more required time values were not found in the config file.")

current_processing_step = "Processing Step 01 IN PROGRESS."
param_manager.add_to_config("current_processing_step", current_processing_step)

# **Step 1: Clean and prepare data
# 0. Check that units are in meters or convert if necessary
original_pressure_unit = data_pkl.sensor_info['pressure']['original_units']
pressure_unit = data_pkl.sensor_info['pressure']['units']

if original_pressure_unit == 'bar' and pressure_unit != 'm': # if bar to m and hasn't been converted yet
    print("Converting pressure from bar to m")
    data_pkl.sensor_data['pressure']['pressure'] *= 10
    data_pkl.sensor_info['pressure']['units'] = 'm'
    print("✅ Pressure unit changed from bar to m")
elif pressure_unit in ['m', '100bar_1', 'msw']: # including CATS format weird 100bar_1 which seems to be m
    print("✅ Pressure unit already in m")
    data_pkl.sensor_info['pressure']['units'] = 'm'
    pass
else:
    print(f"Unknown pressure unit: {pressure_unit}")
    raise ValueError(f"Unknown pressure unit: {pressure_unit}")

new_pressure_unit = data_pkl.sensor_info['pressure']['units']

# 1. Check if logger is known to produce extreme pressure values
if data_pkl.sensor_info['pressure']['logger_manufacturer'] == 'Evolocus':
    # 2. Check if logger_restart events have already been added
    if data_pkl.event_data is None or data_pkl.event_data.empty or not any(data_pkl.event_data['key'] == 'logger_restart'):
        # 3. Identify bad segments based on unrealistic negative pressure
        pressure_df = data_pkl.sensor_data['pressure'].copy()
        restarts = find_segments(
            data=pressure_df,
            column='pressure',
            criteria=lambda x: x < -500,
            min_duration=None
        )

        # 4. Add restart events using standardized event creation
        if not restarts.empty:
            data_pkl.event_data = create_state_event(
                state_df=restarts,
                key="logger_restart",
                start_time_column="start_datetime",
                duration_column="duration",
                description="Detected logger restart from extreme pressure",
                long_description="Logger restart inferred from pressure values dropping below -500, typically inserted by logger hardware during reboot.",
                existing_events=data_pkl.event_data
            )
            print(f"🟠 Added {len(restarts)} logger_restart event(s) to event_data.")

            # 5. Replace pressure values with NaN around each segment
            pressure_series = data_pkl.sensor_data['pressure']
            datetimes = pressure_series['datetime']
            for _, row in restarts.iterrows():
                start = row['start_datetime']
                end = row['end_datetime']
                mask = (datetimes >= start) & (datetimes <= end)
                buffer_before = datetimes.shift(1)
                buffer_after = datetimes.shift(-1)
                buffer_mask = (buffer_before >= start) & (buffer_before <= end) | (buffer_after >= start) & (buffer_after <= end)
                full_mask = mask | buffer_mask
                data_pkl.sensor_data['pressure'].loc[full_mask, 'pressure'] = np.nan
            print("⚠️ Replaced extreme pressure values (< -500) and surrounding buffer with NaN.")
        else:
            print("✅ No restart segments detected.")
    else:
        print("✅ Logger restart events already exist in event_data.")

    # 6. Check again in case any extreme values remain outside known segments
    extreme_exists = (data_pkl.sensor_data['pressure']['pressure'] < -500).any()
    if extreme_exists:
        data_pkl.sensor_data['pressure'].loc[
            data_pkl.sensor_data['pressure']['pressure'] < -500, 'pressure'
        ] = np.nan
        print("⚠️ Replaced residual extreme pressure values (< -500) with NaN.")
    else:
        print("✅ No extreme pressure values found.")
else:
    print("✅ No logger restart check needed for this logger.")

# Step 4: Load depth and temperature data
depth_data = data_pkl.sensor_data["pressure"]["pressure"]
depth_datetime = data_pkl.sensor_data["pressure"]["datetime"]
depth_fs = data_pkl.sensor_info["pressure"]["sampling_frequency"]
if 'temperature-ext' in data_pkl.sensor_data:
    temp_data = data_pkl.sensor_data['temperature-ext']['temp-ext']
    temp_fs = data_pkl.sensor_info['temperature-ext']['sampling_frequency']
elif 'temperature-int' in data_pkl.sensor_data:
    temp_data = data_pkl.sensor_data['temperature-int']['temp-int']
    temp_fs = data_pkl.sensor_info['temperature-int']['sampling_frequency']
else:
    temp_data = None
    temp_fs = None

# **Step 2: Load Configuration Parameters**
dive_detection_settings = param_manager.get_from_config(
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
param_manager.add_to_config(entries=dive_detection_settings, section="dive_detection_settings")

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

corrected_depth = corrected_depth_temp if dive_detection_settings["apply_temp_correction"] else corrected_depth_no_temp

# Detect dives using find_segments
depth_df = pd.DataFrame({
    'datetime': depth_downsampled_datetime,
    'depth': corrected_depth
})

dives = find_segments(
    data=depth_df,
    column='depth',
    criteria=lambda x: x > dive_detection_settings['min_depth_threshold'],
    min_duration=dive_detection_settings['dive_duration_threshold'],
)

nan_mask = depth_data.isna().reindex(depth_downsampled_datetime.index, method='nearest')
dives['has_nans'] = dives.apply(lambda row: nan_mask.loc[(depth_downsampled_datetime >= row['start_datetime']) & (depth_downsampled_datetime <= row['end_datetime'])].any(), axis=1)
dives['short_description'] = dives['has_nans'].apply(lambda x: 'dive-with-nan_start' if x else 'dive_start')

corrected_depth = enforce_surface_before_after_dives(corrected_depth, depth_downsampled_datetime, dives)

dives['dive_duration'] = (dives['end_datetime'] - dives['start_datetime']).dt.total_seconds()

transformation_log = [
    f"downsampled_{dive_detection_settings['downsampled_sampling_rate']}Hz",
    f"smoothed_{dive_detection_settings['smoothing_window']}s",
    f"ZOC_settings__first_deriv_threshold_{dive_detection_settings['first_deriv_threshold']}mps__min_duration_{dive_detection_settings['min_duration']}s__depth_threshold_{dive_detection_settings['depth_threshold']}m",
    f"DIVE_detection_settings__min_depth_threshold_{dive_detection_settings['min_depth_threshold']}m__dive_duration_threshold_{dive_detection_settings['dive_duration_threshold']}s__smoothing_window_{dive_detection_settings['smoothing_window']}"
]

print(f"✅ {len(flat_chunks)} surface intervals detected.")
print(f"✅ {len(dives)} dives detected.")
print("📖 Transformation Log:", transformation_log)

# Append max depth for each dive segment
dives = append_stats(
    data=depth_df, 
    segment_df=dives, 
    statistics=[("max", "depth")]
)

# Generate and update dive events
data_pkl.event_data = create_state_event(
    state_df=dives,
    key='dive',
    value_column='depth_max',
    start_time_column='start_datetime',
    duration_column='dive_duration', # in seconds
    description='dive_start',
    existing_events=data_pkl.event_data  # Pass existing events for overwrite and concatenation
)

# Update event_info with unique keys
data_pkl.event_info = list(data_pkl.event_data['key'].unique())

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
settings = param_manager.get_from_config(variable_names=["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
OVERLAP_START_TIME = pd.Timestamp(settings["overlap_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["overlap_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)

# fig = plot_tag_data_interactive(
#     data_pkl=data_pkl,
#     sensors=['pressure'],
#     derived_data_signals=['depth'],
#     time_range=(depth_downsampled_datetime.min(), depth_downsampled_datetime.max()),
#     note_annotations={"dive": {"signal": "depth", "symbol": "triangle-down", "color": "blue"}},
#     state_annotations={"dive": {"signal": "depth", "color": "rgba(150, 150, 150, 0.3)"}},
#     color_mapping_path=color_mapping_path,
#     target_sampling_rate=1
# )
# fig.show()

param_manager.add_to_config("current_processing_step", "Processing Step 01: Pressure sensor calibration complete.")
with open(pkl_path, "wb") as file:
    pickle.dump(data_pkl, file)

exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
netcdf_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_step01.nc') # Define the export path
exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format

print("✅ Data processing complete. Pickle file updated.")