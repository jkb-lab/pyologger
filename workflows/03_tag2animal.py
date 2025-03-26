import os
import argparse
import pandas as pd

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.event_manager import *
from pyologger.plot_data.plotter import *
from pyologger.io_operations.base_exporter import *
from pyologger.utils.data_manager import *
from pyologger.calibrate_data.tag2animal import *

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
if None in {OVERLAP_START_TIME, OVERLAP_END_TIME, ZOOM_WINDOW_START_TIME, ZOOM_WINDOW_END_TIME}:
    raise ValueError("One or more required time values were not found in the config file.")

current_processing_step = "Processing Step 03 IN PROGRESS."
param_manager.add_to_config("current_processing_step", current_processing_step)

acc_df = data_pkl.derived_data['calibrated_acc']
mag_df = data_pkl.derived_data['calibrated_mag']
gyr_df = data_pkl.sensor_data['gyroscope']

# Calculate and print sampling frequency for each dataframe
acc_fs = calculate_sampling_frequency(acc_df['datetime'])
print(f"Accelerometer Sampling frequency: {acc_fs} Hz")

mag_fs = calculate_sampling_frequency(mag_df['datetime'])
print(f"Magnetometer Sampling frequency: {mag_fs} Hz")

gyr_fs = calculate_sampling_frequency(gyr_df['datetime'])
print(f"Gyroscope Sampling frequency: {gyr_fs} Hz")

acc_data = acc_df[['ax','ay','az']]
mag_data = mag_df[['mx','my','mz']]
gyr_data = gyr_df[['gx', 'gy', 'gz']]

upsampled_columns = []
for col in gyr_data.columns:
    upsampled_col = upsample(gyr_data[col].values, acc_fs / gyr_fs, len(acc_data))  # Apply upsample to each column
    upsampled_columns.append(upsampled_col)  # Append the upsampled column to the list

# Combine the upsampled columns back into a NumPy array
gyr_data_upsampled = np.column_stack(upsampled_columns)
gyr_data = gyr_data_upsampled

acc_data = acc_data.values
mag_data = mag_data.values

# Assuming acc_data and mag_data_upsampled are NumPy arrays
print(f"Gyroscope upsampled to match accelerometer length: acc_data shape= {acc_data.shape}, upsampled gyr_data shape = {gyr_data.shape}")
sampling_rate = acc_fs

# Retrieve values from config
variables = ["calm_horizontal_start_time", "calm_horizontal_end_time", 
             "zoom_window_start_time", "zoom_window_end_time", 
             "overlap_start_time", "overlap_end_time"]
settings = param_manager.get_from_config(variables, section="settings")

# Assign retrieved values to variables
CALM_HORIZONTAL_START_TIME = settings.get("calm_horizontal_start_time")
CALM_HORIZONTAL_END_TIME = settings.get("calm_horizontal_end_time")
ZOOM_START_TIME = settings.get("zoom_window_start_time")
ZOOM_END_TIME = settings.get("zoom_window_end_time")
OVERLAP_START_TIME = settings.get("overlap_start_time")
OVERLAP_END_TIME = settings.get("overlap_end_time")

# Check if manual update is required for CALM_HORIZONTAL_START_TIME and CALM_HORIZONTAL_END_TIME
requires_manual_update = False
if not CALM_HORIZONTAL_START_TIME or CALM_HORIZONTAL_START_TIME == "PLACEHOLDER":
    requires_manual_update = True
if not CALM_HORIZONTAL_END_TIME or CALM_HORIZONTAL_END_TIME == "PLACEHOLDER":
    requires_manual_update = True

# Use ZOOM_WINDOW values as defaults if CALM_HORIZONTAL times are placeholders
if requires_manual_update:
    CALM_HORIZONTAL_START_TIME = CALM_HORIZONTAL_START_TIME or ZOOM_START_TIME
    CALM_HORIZONTAL_END_TIME = CALM_HORIZONTAL_END_TIME or ZOOM_END_TIME

# Display values to the user
print("CALM_HORIZONTAL_START_TIME (current or default):", CALM_HORIZONTAL_START_TIME)
print("CALM_HORIZONTAL_END_TIME (current or default):", CALM_HORIZONTAL_END_TIME)

# Display a message based on whether manual update is needed
if requires_manual_update:
    print("Calm horizontal start and end times require manual update. Proceed to Cell 2 to set placeholders.")
else:
    print("Calm horizontal start and end times are already set. No further action is required.")

# Retrieve timezone from deployment info
timezone = data_pkl.deployment_info['Time Zone']

# Define placeholder timestamps for calm period in the retrieved timezone
placeholder_start_time = ZOOM_WINDOW_START_TIME
placeholder_end_time = ZOOM_WINDOW_END_TIME

# Set this to True if we want to override the placeholders regardless of manual update status
override_required = False

# Only update if manual update is required or if override is enabled
if requires_manual_update or override_required:
    CALM_HORIZONTAL_START_TIME = str(placeholder_start_time)
    CALM_HORIZONTAL_END_TIME = str(placeholder_end_time)
    
    # Use ParamManager to add placeholders to the config
    param_manager.add_to_config("calm_horizontal_start_time",
        value=CALM_HORIZONTAL_START_TIME,
        section="settings"
    )
    param_manager.add_to_config("calm_horizontal_end_time",
        value=CALM_HORIZONTAL_END_TIME,
        section="settings"
    )

    print("Timestamps for calm horizontal start and end times have been set and saved.")
else:
    print("Manual update not required. Placeholders were not set.")

start_time = CALM_HORIZONTAL_START_TIME
end_time   = CALM_HORIZONTAL_END_TIME

# Filter the DataFrame to include only rows within the specified time range
accelerometer_df = data_pkl.sensor_data['accelerometer']
filtered_df = accelerometer_df[(accelerometer_df['datetime'] >= start_time) & (accelerometer_df['datetime'] <= end_time)]

# Calculate the mean of ax, ay, and az columns within the filtered time range
mean_values = filtered_df[['ax', 'ay', 'az']].mean()

# Display the results
print(f"Average values between {start_time} and {end_time}:")
print(mean_values)

abar0 = [mean_values['ax'], mean_values['ay'], mean_values['az']]
# abar0 = [0, 0, -9.8] # to override - use this or similar for orca and other standard CATS tags
deploy_latitude = data_pkl.deployment_info["Deployment Latitude"]
deploy_longitude = data_pkl.deployment_info["Deployment Longitude"]

print(f"Using location Lat: {deploy_latitude}, Lon: {deploy_longitude} and stationary readings of abar0: {str(abar0)} to orient tag.")

# Use the function to get corrected orientation and heading for the entire dataset
pitch_deg, roll_deg, heading_deg, corrected_acc, corrected_mag, corrected_gyr = orientation_and_heading_correction(
    abar0, 
    latitude= deploy_latitude,
    longitude= deploy_longitude,
    acc_data=acc_data, 
    mag_data=mag_data, 
    gyr_data=gyr_data)

# Define multiple key-value pairs to add under a section
settings_to_add = {
    "declination_latitude": deploy_latitude,
    "declination_longitude": deploy_longitude,
    "abar0": ', '.join(f"{value:.3f}" for value in mean_values)
}

# Add the settings under a specific section
param_manager.add_to_config(entries=settings_to_add, section="03_tagtoanimal_settings")

print(f"Tag to animal correction settings saved and added to config file.")

# One datetime column from highest sampled data that was matched by other sensors
datetime_data = data_pkl.sensor_data['accelerometer']['datetime']

# Step 1: Create a DataFrame for pitch, roll, and heading
prh_df = pd.DataFrame({
    'datetime': datetime_data,
    'pitch': pitch_deg,
    'roll': roll_deg,
    'heading': heading_deg
})

# Store the 'prh' variable in derived_data
data_pkl.derived_data['prh'] = prh_df
data_pkl.derived_info['prh'] = {
    "channels": ["pitch", "roll", "heading"],
    "metadata": {
        'pitch': {'original_name': 'Pitch (degrees)',
                  'unit': 'degrees',
                  'sensor': 'accelerometer'},
        'roll': {'original_name': 'Roll (degrees)',
                 'unit': 'degrees',
                 'sensor': 'accelerometer'},
        'heading': {'original_name': 'Heading (degrees)',
                    'unit': 'degrees',
                    'sensor': 'magnetometer'}
    },
    "derived_from_sensors": ["accelerometer", "magnetometer"],
    "transformation_log": [f"calculated_pitch_roll_heading using abar0: {abar0} from calibration period with start time: {start_time} and end time: {end_time} at Deployment Latitude: {deploy_latitude} and Deployment Longitude: {deploy_longitude}."]
}

# Step 2: Create DataFrames for corrected accelerometer, magnetometer, and gyroscope data
corrected_acc_df = pd.DataFrame({
    'datetime': datetime_data,
    'ax': corrected_acc[:, 0],
    'ay': corrected_acc[:, 1],
    'az': corrected_acc[:, 2]
})

corrected_mag_df = pd.DataFrame({
    'datetime': datetime_data,
    'mx': corrected_mag[:, 0],
    'my': corrected_mag[:, 1],
    'mz': corrected_mag[:, 2]
})

corrected_gyr_df = pd.DataFrame({
    'datetime': datetime_data,
    'gx': corrected_gyr[:, 0],
    'gy': corrected_gyr[:, 1],
    'gz': corrected_gyr[:, 2]
})

# Step 3: Store the corrected accelerometer, magnetometer, and gyroscope data into derived_data
data_pkl.derived_data['corrected_acc'] = corrected_acc_df
data_pkl.derived_info['corrected_acc'] = {
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
    "transformation_log": ["corrected_orientation"]
}

data_pkl.derived_data['corrected_mag'] = corrected_mag_df
data_pkl.derived_info['corrected_mag'] = {
    "channels": ["mx", "my", "mz"],
    "metadata": {
        'mx': {'original_name': 'Magnetometer X (µT)',
               'unit': 'µT',
               'sensor': 'magnetometer'},
        'my': {'original_name': 'Magnetometer Y (µT)',
               'unit': 'µT',
               'sensor': 'magnetometer'},
        'mz': {'original_name': 'Magnetometer Z (µT)',
               'unit': 'µT',
               'sensor': 'magnetometer'}
    },
    "derived_from_sensors": ["magnetometer"],
    "transformation_log": ["corrected_orientation"]
}

data_pkl.derived_data['corrected_gyr'] = corrected_gyr_df
data_pkl.derived_info['corrected_gyr'] = {
    "channels": ["gx", "gy", "gz"],
    "metadata": {
        'gx': {'original_name': 'Gyroscope X (deg/s)',
               'unit': 'deg/s',
               'sensor': 'gyroscope'},
        'gy': {'original_name': 'Gyroscope Y (deg/s)',
               'unit': 'deg/s',
               'sensor': 'gyroscope'},
        'gz': {'original_name': 'Gyroscope Z (deg/s)',
               'unit': 'deg/s',
               'sensor': 'gyroscope'}
    },
    "derived_from_sensors": ["gyroscope"],
    "transformation_log": ["corrected_orientation"]
}

TARGET_SAMPLING_RATE = 10

notes_to_plot = {
    'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
    'heartbeat_auto_detect_accepted': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
    'heartbeat_auto_detect_rejected': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'red'}
}

# fig = plot_tag_data_interactive(
#     data_pkl=data_pkl,
#     sensors=['ecg', 'accelerometer', 'magnetometer'],
#     derived_data_signals=['depth', 'corrected_acc', 'corrected_mag', 'prh'],
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

keys_to_remove = ['calibrated_acc','calibrated_mag']

# Clear the specified keys
clear_intermediate_signals(data_pkl, remove_keys=keys_to_remove)

current_processing_step = "Processing Step 03. Tag frame to animal frame transformation complete."
print(current_processing_step)

# Add or update the current_processing_step for the specified deployment
print(current_processing_step)
param_manager.add_to_config("current_processing_step", current_processing_step)

# Optional: save new pickle file
with open(pkl_path, 'wb') as file:
        pickle.dump(data_pkl, file)
print("Pickle file updated.")

exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
netcdf_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_step03.nc') # Define the export path
exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format
