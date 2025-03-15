# Script to calibrate accelerometer and magnetometer data: generates calibrated_acc and calibrated_mag in derived_data
# See 02_calibrate_accmag.ipynb notebook for more detailed description and view intermediate outputs
# Run with shell command: python pyologger/workflows/02_calibrate_accmag.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
import os
import argparse
import pandas as pd

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.event_manager import *
from pyologger.plot_data.plotter import *
from pyologger.io_operations.base_exporter import *
from pyologger.utils.data_manager import *
from pyologger.calibrate_data.calibrate_acc_mag import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()

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
settings = config_manager.get_from_config(variable_names=["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
OVERLAP_START_TIME = pd.Timestamp(settings["overlap_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["overlap_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)
if None in {OVERLAP_START_TIME, OVERLAP_END_TIME, ZOOM_WINDOW_START_TIME, ZOOM_WINDOW_END_TIME}:
    raise ValueError("One or more required time values were not found in the config file.")

current_processing_step = "Processing Step 02 IN PROGRESS."
config_manager.add_to_config("current_processing_step", current_processing_step)

# Check sampling frequencies of accelerometer and magnetometer
acc_fs = data_pkl.sensor_info['accelerometer']['sampling_frequency']
mag_fs = data_pkl.sensor_info['magnetometer']['sampling_frequency']
print(f"Sampling frequency of Accelerometer is {acc_fs/mag_fs}X of Magnetometer.")

acc_data = data_pkl.sensor_data['accelerometer'][['ax','ay','az']]
mag_data = data_pkl.sensor_data['magnetometer'][['mx', 'my', 'mz']]

upsampled_columns = []
for col in mag_data.columns:
    upsampled_col = upsample(mag_data[col].values, acc_fs / mag_fs, len(acc_data))  # Apply upsample to each column
    upsampled_columns.append(upsampled_col)  # Append the upsampled column to the list

# Combine the upsampled columns back into a NumPy array
mag_data_upsampled = np.column_stack(upsampled_columns)
mag_data = mag_data_upsampled

acc_data = acc_data.values

# Assuming acc_data and mag_data_upsampled are NumPy arrays
print(f"Magnetometer upsampled to match accelerometer length: acc_data shape= {acc_data.shape}, upsampled mag_data shape = {mag_data.shape}")
sampling_rate = acc_fs

# Call the check_AM function
AMcheck = compute_field_intensity_and_inclination(acc_data, mag_data, sampling_rate)

# Access the field intensity and inclination angle
field_intensity_acc = AMcheck['field_intensity'][:, 0]  # Field intensity of accelerometer data
field_intensity_mag = AMcheck['field_intensity'][:, 1]  # Field intensity of magnetometer data
inclination_angle = AMcheck['inclination_angle']

# Print results
print("Pre-calibration values:")
print("Field Intensity (Accelerometer):\n", field_intensity_acc)
print("Field Intensity (Magnetometer):\n", field_intensity_mag)
print("Inclination Angle (degrees):\n", inclination_angle)

# Calibration for Accelerometer (field_intensity_acc)
calibration_acc = pd.DataFrame({
    'datetime': data_pkl.sensor_data['accelerometer']['datetime'],
    'field_intensity_acc': field_intensity_acc
})
data_pkl.derived_data['calibration_acc'] = calibration_acc
data_pkl.derived_info['calibration_acc'] = {
    "channels": ["field_intensity_acc"],
    "metadata": {
        'field_intensity_acc': {'original_name': 'Field Intensity Acc (m/s^2)',
                                'unit': 'm/s^2',
                                'sensor': 'accelerometer'}
    },
    "derived_from_sensors": ["accelerometer"],
    "transformation_log": ["checked_field_intensity"]
}

# Calibration for Magnetometer (field_intensity_mag)
calibration_mag = pd.DataFrame({
    'datetime': data_pkl.sensor_data['accelerometer']['datetime'],
    'field_intensity_mag': field_intensity_mag
})
data_pkl.derived_data['calibration_mag'] = calibration_mag
data_pkl.derived_info['calibration_mag'] = {
    "channels": ["field_intensity_mag"],
    "metadata": {
        'field_intensity_mag': {'original_name': 'Field Intensity Mag (uT)',
                                'unit': 'uT',
                                'sensor': 'magnetometer'}
    },
    "derived_from_sensors": ["magnetometer"],
    "transformation_log": ["checked_field_intensity"]
}

# Inclination Angle
inclination_angle_df = pd.DataFrame({
    'datetime': data_pkl.sensor_data['accelerometer']['datetime'],
    'inclination_angle': inclination_angle
})
data_pkl.derived_data['inclination_angle'] = inclination_angle_df
data_pkl.derived_info['inclination_angle'] = {
    "channels": ["inclination_angle"],
    "metadata": {
        'inclination_angle': {'original_name': 'Inclination Angle (deg)',
                              'unit': 'deg',
                              'sensor': 'extra'}
    },
    "derived_from_sensors": ["accelerometer", "magnetometer"],
    "transformation_log": ["calculated_inclination_angle"]
}

# Apply the fix_offset_3d function to adjust accelerometer data
result = estimate_offset_triaxial(acc_data)

# Extract the adjusted data and calibration info
adjusted_data_acc = result['X']
calibration_info_acc = result['G']

print("Adjusted Data:\n", adjusted_data_acc)
print("Calibration Info:\n", calibration_info_acc)

# Calibration for Accelerometer (field_intensity_acc and ax, ay, az)
calibration_acc = pd.DataFrame({
    'datetime': data_pkl.sensor_data['accelerometer']['datetime'],
    'ax': adjusted_data_acc[:,0],
    'ay': adjusted_data_acc[:,1],
    'az': adjusted_data_acc[:,2]
})

data_pkl.derived_data['calibrated_acc'] = calibration_acc
data_pkl.derived_info['calibrated_acc'] = {
    "channels": ["ax", "ay", "az"],
    "metadata": {
        'ax': {'original_name': 'Acceleration X (m/s^2)',
               'unit': 'm/s^2',
               'sensor': 'accelerometer'},
        'ay': {'original_name': 'Acceleration Y (m/s^2)',
               'unit': 'm/s^2',
               'sensor': 'accelerometer'},
        'az': {'original_name': 'Acceleration Z (m/s^2)',
               'unit': 'm/s^2',
               'sensor': 'accelerometer'}
    },
    "derived_from_sensors": ["accelerometer"],
    "transformation_log": ["estimated_offset_triaxial"]
}

# Apply the fix_offset_3d function to adjust magnetometer data
result = estimate_offset_triaxial(mag_data)

# Extract the adjusted data and calibration info
adjusted_data_mag = result['X']
calibration_info_mag = result['G']

print("Adjusted Data:\n", adjusted_data_mag)
print("Calibration Info:\n", calibration_info_mag)

# Re-run check AM

# Calibration for Magnetometer (field_intensity_mag and mx, my, mz)
calibration_mag = pd.DataFrame({
    'datetime': data_pkl.sensor_data['accelerometer']['datetime'],
    'mx': adjusted_data_mag[:,0],
    'my': adjusted_data_mag[:,1],
    'mz': adjusted_data_mag[:,2]
})

data_pkl.derived_data['calibrated_mag'] = calibration_mag
data_pkl.derived_info['calibrated_mag'] = {
    "channels": ["mx", "my", "mz"],
    "metadata": {
        'mx': {'original_name': 'Magnetometer (uT)',
               'unit': 'uT',
               'sensor': 'magnetometer'},
        'my': {'original_name': 'Magnetometer (uT)',
               'unit': 'uT',
               'sensor': 'magnetometer'},
        'mz': {'original_name': 'Magnetometer (uT)',
               'unit': 'uT',
               'sensor': 'magnetometer'}
    },
    "derived_from_sensors": ["magnetometer"],
    "transformation_log": ["estimated_offset_triaxial"]
}

# Assuming acc_data and mag_data are extracted from the calibrated accelerometer and magnetometer data
acc_data = data_pkl.derived_data['calibrated_acc'][['ax', 'ay', 'az']].values
mag_data = data_pkl.derived_data['calibrated_mag'][['mx', 'my', 'mz']].values
sampling_rate = 100  # Adjust this to the correct sampling rate of your data

# Call the check_AM function
AMcheck = compute_field_intensity_and_inclination(acc_data, mag_data, sampling_rate)

# Access the field intensity and inclination angle
field_intensity_acc = AMcheck['field_intensity'][:, 0]  # Field intensity of accelerometer data
field_intensity_mag = AMcheck['field_intensity'][:, 1]  # Field intensity of magnetometer data
inclination_angle = AMcheck['inclination_angle']

# Append the new field intensity and inclination angle to calibration_acc and calibration_mag

# Calibration for Accelerometer (append field_intensity_acc)
data_pkl.derived_data['calibration_acc']['calibrated_field_intensity_acc'] = field_intensity_acc

# Calibration for Magnetometer (append field_intensity_mag)
data_pkl.derived_data['calibration_mag']['calibrated_field_intensity_mag'] = field_intensity_mag

# Inclination Angle (append inclination_angle)
data_pkl.derived_data['inclination_angle']['calibrated_inclination_angle'] = inclination_angle

# Update the derived_info to reflect the new columns for accelerometer and magnetometer
data_pkl.derived_info['calibration_acc']["channels"].append("calibrated_field_intensity_acc")
data_pkl.derived_info['calibration_acc']["metadata"]['calibrated_field_intensity_acc'] = {
    'original_name': 'Calibrated Field Intensity Acc (m/s^2)',
    'unit': 'm/s^2',
    'sensor': 'accelerometer'
}

data_pkl.derived_info['calibration_mag']["channels"].append("calibrated_field_intensity_mag")
data_pkl.derived_info['calibration_mag']["metadata"]['calibrated_field_intensity_mag'] = {
    'original_name': 'Calibrated Field Intensity Mag (uT)',
    'unit': 'uT',
    'sensor': 'magnetometer'
}

# Update the derived_info for inclination_angle
data_pkl.derived_info['inclination_angle']["channels"].append("calibrated_inclination_angle")
data_pkl.derived_info['inclination_angle']["metadata"]['calibrated_inclination_angle'] = {
    'original_name': 'Calibrated Inclination Angle (deg)',
    'unit': 'deg',
    'sensor': 'accelerometer, magnetometer'
}

# Visualize results
# Retrieve necessary time settings from the settings section
time_settings = config_manager.get_from_config(
    ["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],
    section="settings"
)

# Assign retrieved values to variables
OVERLAP_START_TIME = time_settings.get("overlap_start_time")
OVERLAP_END_TIME = time_settings.get("overlap_end_time")
ZOOM_START_TIME = time_settings.get("zoom_window_start_time")
ZOOM_END_TIME = time_settings.get("zoom_window_end_time")
TARGET_SAMPLING_RATE = int(10)

notes_to_plot = {
    'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
    'heartbeat_auto_detect_accepted': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
    'heartbeat_auto_detect_rejected': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'red'}
}

# fig = plot_tag_data_interactive(
#     data_pkl=data_pkl,
#     sensors=['accelerometer', 'magnetometer'],
#     derived_data_signals=['depth', 'calibrated_acc', 'calibrated_mag', 'inclination_angle', 'calibration_acc', 'calibration_mag'],
#     channels={},
#     time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
#     note_annotations=notes_to_plot,
#     color_mapping_path=color_mapping_path,
#     target_sampling_rate=TARGET_SAMPLING_RATE,
#     zoom_start_time=ZOOM_START_TIME,
#     zoom_end_time=ZOOM_END_TIME,
#     zoom_range_selector_channel='depth',
#     plot_event_values=[],
# )
#fig.show()

# Delete intermediate signals
keys_to_remove = ["calibration_acc", "calibration_mag", "inclination_angle"]

# Clear the specified keys
clear_intermediate_signals(data_pkl, remove_keys=keys_to_remove)

current_processing_step = "Processing Step 02. Calibration of accelerometer and magnetometer complete."
print(current_processing_step)

# Add or update the current_processing_step for the specified deployment
config_manager.add_to_config("current_processing_step", current_processing_step)

# Optional: save new pickle file
with open(pkl_path, 'wb') as file:
        pickle.dump(data_pkl, file)
print("Pickle file updated.")

exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
netcdf_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_step00.nc') # Define the export path
exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format
