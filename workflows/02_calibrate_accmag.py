# Script to calibrate accelerometer and magnetometer data: generates calibrated_acc and calibrated_mag in signal_data
# See 02_calibrate_accmag.ipynb notebook for more detailed description and view intermediate outputs
# Run with shell command: python3 pyologger/workflows/02_calibrate_accmag.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
import os
import pickle
import argparse
import sys
import pandas as pd

# Ensure direct workflow execution resolves the repo-local pyologger package.
WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.event_manager import *
from pyologger.plot_data.plotter import *
from pyologger.io_operations.base_exporter import *
from pyologger.utils.data_manager import *
from pyologger.calibrate_data.calibrate_acc_mag import *
from pyologger.utils.workflow_netcdf import (
    latest_processing_netcdf_path,
    netcdf_has_signal,
    save_step_netcdf_if_changed,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Resolve deployment first so missing critical signals can short-circuit via NetCDF metadata.
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
latest_netcdf_path = latest_processing_netcdf_path(deployment_folder, deployment_id)

if not netcdf_has_signal(latest_netcdf_path, "magnetometer"):
    print("Skipping Step 02 based on NetCDF metadata: magnetometer signal not available.")
    param_manager.add_to_config("current_processing_step", "Processing Step 02 skipped: missing magnetometer input.")
    raise SystemExit(0)

with open(pkl_path, "rb") as file:
    data_pkl = pickle.load(file)

# Load key time points
timezone = data_pkl.deployment_info.get('Time Zone', 'UTC')
settings = param_manager.get_from_config(variable_names=["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
OVERLAP_START_TIME = pd.Timestamp(settings["overlap_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["overlap_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)
if None in {OVERLAP_START_TIME, OVERLAP_END_TIME, ZOOM_WINDOW_START_TIME, ZOOM_WINDOW_END_TIME}:
    raise ValueError("One or more required time values were not found in the config file.")

current_processing_step = "Processing Step 02 IN PROGRESS."
param_manager.add_to_config("current_processing_step", current_processing_step)
data_changed = False

# Check accelerometer units: if not , convert to g. This code works fine for either unit, but we prefer to work in g for consistency.

critical_signal = 'magnetometer'
# If critical signal doesn't exist, create flag to skip step.
if critical_signal not in data_pkl.signal_data or data_pkl.signal_data[critical_signal] is None:
    print(f"⚠️ Signal: {critical_signal} not found. Skipping processing.")
    skip_step = True
else:
    # signals exists and can be processed normally
    skip_step = False

if not skip_step:
    data_changed = True
    acc_unit = data_pkl.signal_info['accelerometer']['units']

    if acc_unit == 'g':
        # Convert accelerometer data from g to m/s^2
        conversion_factor = 9.80665  # 1 g = 9.80665 m/s^2
        data_pkl.signal_data['accelerometer'][['ax', 'ay', 'az']] *= conversion_factor
        acc_unit = 'm/s^2'
        data_pkl.signal_info['accelerometer']['units'] = acc_unit
        print("Accelerometer data converted to m/s^2.")
    elif acc_unit == 'm/s²' or acc_unit == 'm/s^2':
        print("Accelerometer data is already in m/s^2.")

    # Check magnetometer units: if not Gauss, convert to Gauss. This code works fine for either unit, but we prefer to work in Gauss for consistency.
    mag_unit = data_pkl.signal_info['magnetometer']['units']

    if mag_unit == 'Gauss':
        # Convert magnetometer data from Gauss to microtesla
        conversion_factor = 100  # 1 Gauss = 100 microtesla
        data_pkl.signal_data['magnetometer'][['mx', 'my', 'mz']] *= conversion_factor
        mag_unit = 'µT'
        data_pkl.signal_info['magnetometer']['units'] = mag_unit
        print("Magnetometer data converted to microtesla.")
    elif mag_unit == 'µT' or mag_unit == 'µt' or mag_unit == 'ut' or mag_unit == 'microtesla':
        print("Magnetometer data is already in microtesla.")

    # Check gyroscope units: if not in mrad/s, convert to mrad/s. This code works fine for either unit, but we prefer to work in mrad/s for consistency.
    gyr_unit = data_pkl.signal_info['gyroscope']['units']

    if gyr_unit == 'rps':
        # Convert gyroscope data from rps to mrad/s
        conversion_factor = 2 * np.pi * 1000  # 1 rps = 2π × 1000 mrad/s
        data_pkl.signal_data['gyroscope'][['gx', 'gy', 'gz']] *= conversion_factor
        gyr_unit = 'mrad/s'
        data_pkl.signal_info['gyroscope']['units'] = gyr_unit
        print("Gyroscope data converted to mrad/s.")
    elif gyr_unit == 'mrad/s':
        print("Gyroscope data is already in mrad/s.")

    # Deal with unreliable data inserted during logger_restart events
    # This code will mask the data during logger_restart events and interpolate across the gaps

    restart_sensitive_signals = ['accelerometer', 'gyroscope', 'magnetometer']

    # Proceed only if logger_restart events are present
    if (
        data_pkl.event_data is not None and
        not data_pkl.event_data.empty and
        any(data_pkl.event_data['key'] == 'logger_restart')
    ):
        restarts = data_pkl.event_data[data_pkl.event_data['key'] == 'logger_restart'].copy()
        restarts['end_datetime'] = restarts['datetime'] + pd.to_timedelta(restarts['duration'], unit='s')

        for signal in restart_sensitive_signals:
            if signal not in data_pkl.signal_data:
                print(f"⚠️ {signal} not found in signal_data. Skipping.")
                continue

            df = data_pkl.signal_data[signal]
            datetime_col = df['datetime']

            # Make a working copy of the data so we can interpolate safely
            df_interp = df.copy()
            channels = [col for col in df.columns if col != 'datetime']

            for _, row in restarts.iterrows():
                start = row['datetime']
                end = row['end_datetime']

                # Buffer to include one sample before and after (optional but good practice)
                mask = (datetime_col >= start) & (datetime_col <= end)
                buffer_before = datetime_col.shift(1)
                buffer_after = datetime_col.shift(-1)
                buffer_mask = ((buffer_before >= start) & (buffer_before <= end)) | \
                            ((buffer_after >= start) & (buffer_after <= end))
                full_mask = mask | buffer_mask

                # Mask the data (set to NaN)
                df_interp.loc[full_mask, channels] = np.nan

            # Interpolate linearly across the gaps
            df_interp[channels] = df_interp[channels].interpolate(method='linear', limit_direction='both')

            # Replace signal data in-place (optional: store separately for later restoration)
            data_pkl.signal_data[signal] = df_interp
            print(f"🔧 Interpolated {signal} data through logger_restart windows.")
    else:
        print("✅ No logger_restart events to apply to signal masks.")

    # Check sampling frequencies of accelerometer and magnetometer
    acc_fs = data_pkl.signal_info['accelerometer']['sampling_frequency']
    acc_fs
    mag_fs = data_pkl.signal_info['magnetometer']['sampling_frequency']
    mag_fs
    print(f"Sampling frequency of Accelerometer is {acc_fs/mag_fs}X of Magnetometer.")

    acc_data = data_pkl.signal_data['accelerometer'][['ax', 'ay', 'az']]
    mag_data = data_pkl.signal_data['magnetometer'][['mx', 'my', 'mz']]

    if acc_fs != mag_fs:
        upsampled_columns = []
        for col in mag_data.columns:
            upsampled_col = upsample(mag_data[col].values, acc_fs / mag_fs, len(acc_data))  # Apply upsample to each column
            upsampled_columns.append(upsampled_col)  # Append the upsampled column to the list

        # Combine the upsampled columns back into a NumPy array
        mag_data_upsampled = np.column_stack(upsampled_columns)
        mag_data = mag_data_upsampled
        print("Magnetometer data upsampled to match accelerometer data length.")
    else:
        mag_data = mag_data.values
        print("Magnetometer data is already at the same sampling frequency as accelerometer data.")

    acc_data = acc_data.values

    # Assuming acc_data and mag_data_upsampled are NumPy arrays
    print(f"Magnetometer matches accelerometer length: acc_data shape= {acc_data.shape}, upsampled mag_data shape = {mag_data.shape}")
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
        'datetime': data_pkl.signal_data['accelerometer']['datetime'],
        'field_intensity_acc': field_intensity_acc
    })
    data_pkl.signal_data['calibration_acc'] = calibration_acc
    data_pkl.signal_info['calibration_acc'] = {
        "channels": ["field_intensity_acc"],
        "metadata": {
            'field_intensity_acc': {'original_name': 'Field Intensity Acc (m/s^2)',
                                    'unit': 'm/s^2',
                                    'signal': 'accelerometer'}
        },
        "derived_from_signals": ["accelerometer"],
        "transformation_log": ["checked_field_intensity"]
    }

    # Calibration for Magnetometer (field_intensity_mag)
    calibration_mag = pd.DataFrame({
        'datetime': data_pkl.signal_data['accelerometer']['datetime'],
        'field_intensity_mag': field_intensity_mag
    })
    data_pkl.signal_data['calibration_mag'] = calibration_mag
    data_pkl.signal_info['calibration_mag'] = {
        "channels": ["field_intensity_mag"],
        "metadata": {
            'field_intensity_mag': {'original_name': 'Field Intensity Mag (uT)',
                                    'unit': 'uT',
                                    'signal': 'magnetometer'}
        },
        "derived_from_signals": ["magnetometer"],
        "transformation_log": ["checked_field_intensity"]
    }

    # Inclination Angle
    inclination_angle_df = pd.DataFrame({
        'datetime': data_pkl.signal_data['accelerometer']['datetime'],
        'inclination_angle': inclination_angle
    })
    data_pkl.signal_data['inclination_angle'] = inclination_angle_df
    data_pkl.signal_info['inclination_angle'] = {
        "channels": ["inclination_angle"],
        "metadata": {
            'inclination_angle': {'original_name': 'Inclination Angle (deg)',
                                'unit': 'deg',
                                'signal': 'extra'}
        },
        "derived_from_signals": ["accelerometer", "magnetometer"],
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
        'datetime': data_pkl.signal_data['accelerometer']['datetime'],
        'ax': adjusted_data_acc[:,0],
        'ay': adjusted_data_acc[:,1],
        'az': adjusted_data_acc[:,2]
    })

    data_pkl.signal_data['calibrated_acc'] = calibration_acc
    data_pkl.signal_info['calibrated_acc'] = {
        "channels": ["ax", "ay", "az"],
        "metadata": {
            'ax': {'original_name': 'Acceleration X (m/s^2)',
                'unit': 'm/s^2',
                'signal': 'accelerometer'},
            'ay': {'original_name': 'Acceleration Y (m/s^2)',
                'unit': 'm/s^2',
                'signal': 'accelerometer'},
            'az': {'original_name': 'Acceleration Z (m/s^2)',
                'unit': 'm/s^2',
                'signal': 'accelerometer'}
        },
        "derived_from_signals": ["accelerometer"],
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
        'datetime': data_pkl.signal_data['accelerometer']['datetime'],
        'mx': adjusted_data_mag[:,0],
        'my': adjusted_data_mag[:,1],
        'mz': adjusted_data_mag[:,2]
    })

    data_pkl.signal_data['calibrated_mag'] = calibration_mag
    data_pkl.signal_info['calibrated_mag'] = {
        "channels": ["mx", "my", "mz"],
        "metadata": {
            'mx': {'original_name': 'Magnetometer (uT)',
                'unit': 'uT',
                'signal': 'magnetometer'},
            'my': {'original_name': 'Magnetometer (uT)',
                'unit': 'uT',
                'signal': 'magnetometer'},
            'mz': {'original_name': 'Magnetometer (uT)',
                'unit': 'uT',
                'signal': 'magnetometer'}
        },
        "derived_from_signals": ["magnetometer"],
        "transformation_log": ["estimated_offset_triaxial"]
    }

    # Assuming acc_data and mag_data are extracted from the calibrated accelerometer and magnetometer data
    acc_data = data_pkl.signal_data['calibrated_acc'][['ax', 'ay', 'az']].values
    mag_data = data_pkl.signal_data['calibrated_mag'][['mx', 'my', 'mz']].values
    sampling_rate = 100  # Adjust this to the correct sampling rate of your data

    # Call the check_AM function
    AMcheck = compute_field_intensity_and_inclination(acc_data, mag_data, sampling_rate)

    # Access the field intensity and inclination angle
    field_intensity_acc = AMcheck['field_intensity'][:, 0]  # Field intensity of accelerometer data
    field_intensity_mag = AMcheck['field_intensity'][:, 1]  # Field intensity of magnetometer data
    inclination_angle = AMcheck['inclination_angle']

    # Append the new field intensity and inclination angle to calibration_acc and calibration_mag

    # Calibration for Accelerometer (append field_intensity_acc)
    data_pkl.signal_data['calibration_acc']['calibrated_field_intensity_acc'] = field_intensity_acc

    # Calibration for Magnetometer (append field_intensity_mag)
    data_pkl.signal_data['calibration_mag']['calibrated_field_intensity_mag'] = field_intensity_mag

    # Inclination Angle (append inclination_angle)
    data_pkl.signal_data['inclination_angle']['calibrated_inclination_angle'] = inclination_angle

    # Update the signal_info to reflect the new columns for accelerometer and magnetometer
    data_pkl.signal_info['calibration_acc']["channels"].append("calibrated_field_intensity_acc")
    data_pkl.signal_info['calibration_acc']["metadata"]['calibrated_field_intensity_acc'] = {
        'original_name': 'Calibrated Field Intensity Acc (m/s^2)',
        'unit': 'm/s^2',
        'signal': 'accelerometer'
    }

    data_pkl.signal_info['calibration_mag']["channels"].append("calibrated_field_intensity_mag")
    data_pkl.signal_info['calibration_mag']["metadata"]['calibrated_field_intensity_mag'] = {
        'original_name': 'Calibrated Field Intensity Mag (uT)',
        'unit': 'uT',
        'signal': 'magnetometer'
    }

    # Update the signal_info for inclination_angle
    data_pkl.signal_info['inclination_angle']["channels"].append("calibrated_inclination_angle")
    data_pkl.signal_info['inclination_angle']["metadata"]['calibrated_inclination_angle'] = {
        'original_name': 'Calibrated Inclination Angle (deg)',
        'unit': 'deg',
        'signal': 'accelerometer, magnetometer'
    }

    # Visualize results
    # Retrieve necessary time settings from the settings section
    time_settings = param_manager.get_from_config(
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
    #     signals=['accelerometer', 'magnetometer','depth', 'calibrated_acc', 'calibrated_mag', 'inclination_angle', 'calibration_acc', 'calibration_mag'],
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
else:
    print(f"Skipping step due to missing critical signal: {critical_signal}.")

current_processing_step = "Processing Step 02. Calibration of accelerometer and magnetometer complete."
print(current_processing_step)

# Add or update the current_processing_step for the specified deployment
param_manager.add_to_config("current_processing_step", current_processing_step)

# Optional: save new pickle file
if data_changed:
    with open(pkl_path, 'wb') as file:
            pickle.dump(data_pkl, file)
    print("Pickle file updated.")

save_step_netcdf_if_changed(data_pkl, deployment_folder, deployment_id, 2, changed=data_changed)
