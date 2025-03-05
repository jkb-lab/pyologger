import os
import argparse
import pandas as pd

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.event_manager import *
from pyologger.plot_data.plotter import *
from pyologger.io_operations.base_exporter import *
from pyologger.utils.data_manager import *
from pyologger.process_data.peak_detect import *

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

# Retrieve values from config
variables = ["calm_horizontal_start_time", "calm_horizontal_end_time", 
             "zoom_window_start_time", "zoom_window_end_time", 
             "earliest_common_start_time", "latest_common_end_time"]
settings = config_manager.get_from_config(variables, section="settings")

# Assign retrieved values to variables
CALM_HORIZONTAL_START_TIME = settings.get("calm_horizontal_start_time")
CALM_HORIZONTAL_END_TIME = settings.get("calm_horizontal_end_time")
ZOOM_START_TIME = settings.get("zoom_window_start_time")
ZOOM_END_TIME = settings.get("zoom_window_end_time")
OVERLAP_START_TIME = settings.get("earliest_common_start_time")
OVERLAP_END_TIME = settings.get("latest_common_end_time")

# CHANGE AS NEEDED

detection_mode="heart_rate"

# Define parent signal options
parent_signal_options = list(data_pkl.sensor_data.keys()) + list(data_pkl.derived_data.keys())
default_parent_signal = "ecg" if detection_mode == "heart_rate" else "corrected_gyr"

# User input for parent signal
print(f"Available parent signals: {parent_signal_options}")
parent_signal = input(f"Choose parent signal (default: {default_parent_signal}): ").strip()
if not parent_signal or parent_signal not in parent_signal_options:
    parent_signal = default_parent_signal

# Get available channels
if parent_signal in data_pkl.sensor_data:
    available_channels = list(data_pkl.sensor_data[parent_signal].columns)
elif parent_signal in data_pkl.derived_data:
    available_channels = list(data_pkl.derived_data[parent_signal].columns)
else:
    available_channels = []

# Default channel
default_channel = "ecg" if detection_mode == "heart_rate" else "gy"

# User input for channel
print(f"Available channels: {available_channels}")
channel = input(f"Choose channel (default: {default_channel}): ").strip()
if not channel or channel not in available_channels:
    channel = default_channel

# Configure signals
signal_df = data_pkl.sensor_data[parent_signal] if parent_signal in data_pkl.sensor_data else data_pkl.derived_data[parent_signal]
signal = data_pkl.sensor_data[parent_signal][channel] if parent_signal in data_pkl.sensor_data else data_pkl.derived_data[parent_signal][channel]
datetime_signal = data_pkl.sensor_data[parent_signal]['datetime'] if parent_signal in data_pkl.sensor_data else data_pkl.derived_data[parent_signal]['datetime']
sampling_rate = data_pkl.sensor_info.get(parent_signal, {}).get('sampling_frequency', calculate_sampling_frequency(datetime_signal))

# Define the default time range based on the signal's datetime column
signal_start = datetime_signal.min()
signal_end = datetime_signal.max()

# User input for time range
print(f"Signal time range: {signal_start} to {signal_end}")
start_time_input = input(f"Enter start time (default: {signal_start}): ").strip()
end_time_input = input(f"Enter end time (default: {signal_end}): ").strip()

# Determine time range based on user input
start_datetime = pd.Timestamp(start_time_input) if start_time_input else signal_start
end_datetime = pd.Timestamp(end_time_input) if end_time_input else signal_end

# Filter signal based on the selected time range
time_mask = (datetime_signal >= start_datetime) & (datetime_signal <= end_datetime)
signal_subset = signal[time_mask]
datetime_subset = datetime_signal[time_mask]
signal_subset_df = signal_df[
    (signal_df['datetime'] >= start_datetime) & 
    (signal_df['datetime'] <= end_datetime)
]

# Output the results
print(f"Time range selected: {start_datetime} to {end_datetime}")
print(f"Signal subset size: {len(signal_subset)}")

# Retrieve parameters for peak detection
params = config_manager.get_from_config(
    variable_names=[
        "BROAD_LOW_CUTOFF", "BROAD_HIGH_CUTOFF", "NARROW_LOW_CUTOFF", "NARROW_HIGH_CUTOFF",
        "FILTER_ORDER", "SPIKE_THRESHOLD", "SMOOTH_SEC_MULTIPLIER", "WINDOW_SIZE_MULTIPLIER",
        "NORMALIZATION_NOISE", "PEAK_HEIGHT", "PEAK_DISTANCE_SEC", "SEARCH_RADIUS_SEC",
        "MIN_PEAK_HEIGHT", "MAX_PEAK_HEIGHT", "enable_bandpass", "enable_spike_removal",
        "enable_absolute", "enable_smoothing", "enable_normalization", "enable_refinement"
    ],
    section="hr_peak_detection_settings" if detection_mode == "heart_rate" else "stroke_peak_detection_settings"
)

overwrite=False # If needed, change to true and rewrite settings here

# Add parameters to the config file (if not already present)
if overwrite | any(value is None for value in params.values()):
    # Define parameters for peak detection with updated values
    params = {
        "BROAD_LOW_CUTOFF": 1.0,  # Hz, lower cutoff for the broad bandpass filter
        "BROAD_HIGH_CUTOFF": 35.0,  # Hz, upper cutoff for the broad bandpass filter
        "NARROW_LOW_CUTOFF": 5.0,  # Hz, lower cutoff for the narrow bandpass filter
        "NARROW_HIGH_CUTOFF": 20.0,  # Hz, upper cutoff for the narrow bandpass filter
        "FILTER_ORDER": 2,  # Order of the bandpass filter, affects sharpness
        "SPIKE_THRESHOLD": 400,  # Threshold for removing large spikes (e.g., noise or artifacts)
        "SMOOTH_SEC_MULTIPLIER": 0.36,  # Multiplier for calculating the smoothing window size
        "WINDOW_SIZE_MULTIPLIER": 6.35,  # Multiplier for calculating sliding window size
        "NORMALIZATION_NOISE": 1e-10,  # Small constant to avoid division by zero in normalization
        "PEAK_HEIGHT": -0.4,  # Minimum amplitude (height) for peak detection
        "PEAK_DISTANCE_SEC": 0.71,  # Minimum time between detected peaks (in seconds)
        "SEARCH_RADIUS_SEC": 0.35,  # Time range for refining the peak location (in seconds)
        "MIN_PEAK_HEIGHT": 70,  # Minimum acceptable amplitude for detected peaks
        "MAX_PEAK_HEIGHT": 12000,  # Maximum acceptable amplitude for detected peaks
        "enable_bandpass": True,  # Enable/disable bandpass filtering
        "enable_spike_removal": True,  # Enable/disable spike removal
        "enable_absolute": True,  # Enable/disable abs() transformation of signal (only use if HR, not for stroke rate)
        "enable_smoothing": True,  # Enable/disable smoothing
        "enable_normalization": True,  # Enable/disable sliding window normalization
        "enable_refinement": True,  # Enable/disable peak refinement
    }
    # Add updated parameters to the config file
    config_manager.add_to_config(entries=params, section="hr_peak_detection_settings")
else:
    print("Settings loaded from config file, not overwritten.")

# Use the updated parameters in peak detection
results = peak_detect(
    signal=signal_subset,
    sampling_rate=sampling_rate,
    datetime_series=datetime_subset,
    broad_lowcut=params["BROAD_LOW_CUTOFF"],
    broad_highcut=params["BROAD_HIGH_CUTOFF"],
    narrow_lowcut=params["NARROW_LOW_CUTOFF"],
    narrow_highcut=params["NARROW_HIGH_CUTOFF"],
    filter_order=params["FILTER_ORDER"],
    spike_threshold=params["SPIKE_THRESHOLD"],
    smooth_sec_multiplier=params["SMOOTH_SEC_MULTIPLIER"],
    window_size_multiplier=params["WINDOW_SIZE_MULTIPLIER"],
    normalization_noise=params["NORMALIZATION_NOISE"],
    peak_height=params["PEAK_HEIGHT"],
    peak_distance_sec=params["PEAK_DISTANCE_SEC"],
    search_radius_sec=params["SEARCH_RADIUS_SEC"],
    min_peak_height=params["MIN_PEAK_HEIGHT"],
    max_peak_height=params["MAX_PEAK_HEIGHT"],
    enable_bandpass=params["enable_bandpass"],
    enable_spike_removal=params["enable_spike_removal"],
    enable_absolute=params["enable_absolute"],
    enable_smoothing=params["enable_smoothing"],
    enable_normalization=params["enable_normalization"],
    enable_refinement=params["enable_refinement"]
)

process_rate(data_pkl, results, signal_subset_df, parent_signal,
             params, sampling_rate, detection_mode)

TARGET_SAMPLING_RATE = 10

notes_to_plot = {
    'heartbeat_manual_ok': {'signal': 'hr_normalized', 'symbol': 'triangle-down', 'color': 'blue'},
    'heartbeat_auto_detect_accepted': {'signal': 'hr_normalized', 'symbol': 'triangle-up', 'color': 'green'},
    'heartbeat_auto_detect_rejected': {'signal': 'hr_normalized', 'symbol': 'triangle-up', 'color': 'red'},
    'strokebeat_auto_detect_accepted': {'signal': 'sr_smoothed', 'symbol': 'triangle-up', 'color': 'green'},
}

fig = plot_tag_data_interactive(
    data_pkl=data_pkl,
    sensors=['ecg', 'hr_broad_bandpass', 'hr_narrow_bandpass','hr_smoothed', 'hr_normalized'],
    derived_data_signals=['depth', 'prh', 'stroke_rate', 'heart_rate','sr_smoothed'],
    channels={}, #'corrected_gyr': ['broad_bandpassed_signal']
    time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
    note_annotations=notes_to_plot,
    color_mapping_path=color_mapping_path,
    target_sampling_rate=TARGET_SAMPLING_RATE,
    zoom_start_time=ZOOM_START_TIME,
    zoom_end_time=ZOOM_END_TIME,
    zoom_range_selector_channel='depth',
    plot_event_values=[],
)

fig.show()

# Clean-up events

# Clear the specified keys
keys_to_remove = ['hr_broad_bandpass','hr_narrow_bandpass', 'hr_smoothed'] # KEEPING 'hr_normalized' because it is clearest
clear_intermediate_signals(data_pkl, remove_keys=keys_to_remove)

initial_event_count = len(data_pkl.event_data)
# Remove events with keys ending in '_rejected'
data_pkl.event_data = data_pkl.event_data[~data_pkl.event_data['key'].str.endswith('_rejected', na=False)]
# Get the final count of events
final_event_count = len(data_pkl.event_data)
# Print the number of removed events
removed_event_count = initial_event_count - final_event_count
print(f"Removed {removed_event_count} events with keys ending in '_rejected'.")

current_processing_step = "Processing Step 05. Heart rate calculation complete."
print(current_processing_step)

# Add or update the current_processing_step for the specified deployment
config_manager.add_to_config("current_processing_step", current_processing_step)

# Optional: save new pickle file
with open(pkl_path, 'wb') as file:
        pickle.dump(data_pkl, file)
print("Pickle file updated.")