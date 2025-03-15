import streamlit as st
import pandas as pd
from datetime import timedelta

# Import pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.data_manager import *
from pyologger.calibrate_data.zoc import *
from pyologger.plot_data.plotter import *
from pyologger.process_data.sampling import *
from pyologger.process_data.peak_detect import *

# Load important file paths and configurations
config, data_dir, color_mapping_path, montage_path = load_configuration()

# **Step 2: Deployment Selection (Dropdown Menus)**
st.sidebar.title("Deployment Selection")

# Streamlit load data
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, config_manager = select_and_load_deployment_streamlit(data_dir)
timezone = data_pkl.deployment_info.get('Time Zone', 'UTC')

st.sidebar.write(f"📂 Selected Deployment: {deployment_id}")

# Load relevant configuration settings
settings = config_manager.get_from_config(
    variable_names=["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],
    section="settings"
)
OVERLAP_START_TIME = pd.Timestamp(settings["overlap_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["overlap_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)

# Function to get parameters for heart rate or stroke rate
def get_params(mode):
    section = "hr_peak_detection_settings" if mode == "heart_rate" else "stroke_peak_detection_settings"
    return config_manager.get_from_config(variable_names=[
        "BROAD_LOW_CUTOFF", "BROAD_HIGH_CUTOFF", "NARROW_LOW_CUTOFF", "NARROW_HIGH_CUTOFF",
        "FILTER_ORDER", "SPIKE_THRESHOLD", "SMOOTH_SEC_MULTIPLIER", "WINDOW_SIZE_MULTIPLIER",
        "NORMALIZATION_NOISE", "PEAK_HEIGHT", "PEAK_DISTANCE_SEC", "SEARCH_RADIUS_SEC",
        "MIN_PEAK_HEIGHT", "MAX_PEAK_HEIGHT", "enable_bandpass", "enable_spike_removal",
        "enable_absolute", "enable_smoothing", "enable_normalization", "enable_refinement"
    ], section=section)

# User selection for detection type
st.sidebar.title("Detection Mode")
detection_mode = st.sidebar.selectbox("Choose mode:", ["stroke_rate", "heart_rate"])

# Adjust parameters based on the selected mode
params = get_params(detection_mode)

# Select parent signal and channel
st.sidebar.subheader("Signal Configuration")
parent_signal_options = list(data_pkl.sensor_data.keys()) + list(data_pkl.derived_data.keys())
default_parent_signal = "ecg" if detection_mode == "heart_rate" else "corrected_gyr"
parent_signal = st.sidebar.selectbox("Parent Signal", parent_signal_options, index=parent_signal_options.index(default_parent_signal))

# Get available channels for the selected parent signal
if parent_signal in data_pkl.sensor_data:
    available_channels = list(data_pkl.sensor_data[parent_signal].columns)
elif parent_signal in data_pkl.derived_data:
    available_channels = list(data_pkl.derived_data[parent_signal].columns)
else:
    available_channels = []

# Select specific channel
default_channel = "ecg" if detection_mode == "heart_rate" else "gy"
channel = st.sidebar.selectbox("Channel", available_channels, index=available_channels.index(default_channel) if default_channel in available_channels else 0)

# Configure signals
signal_df = data_pkl.sensor_data[parent_signal] if parent_signal in data_pkl.sensor_data else data_pkl.derived_data[parent_signal]
signal = data_pkl.sensor_data[parent_signal][channel] if parent_signal in data_pkl.sensor_data else data_pkl.derived_data[parent_signal][channel]
datetime_signal = data_pkl.sensor_data[parent_signal]['datetime'] if parent_signal in data_pkl.sensor_data else data_pkl.derived_data[parent_signal]['datetime']
sampling_rate = data_pkl.sensor_info.get(parent_signal, {}).get('sampling_frequency', calculate_sampling_frequency(datetime_signal))

# Streamlit UI
st.title(f"{detection_mode} Peak Detection")

# Convert datetime range to a list of allowable values with 10-second steps
start_time = ZOOM_WINDOW_START_TIME.to_pydatetime()
end_time = ZOOM_WINDOW_END_TIME.to_pydatetime()
time_values = [start_time + timedelta(seconds=i) for i in range(0, int((end_time - start_time).total_seconds()) + 1, 10)]

# Double-sided slider for time range selection
time_range = st.sidebar.select_slider(
    "Select Time Range",
    options=time_values,
    value=(ZOOM_WINDOW_START_TIME.to_pydatetime(), ZOOM_WINDOW_END_TIME.to_pydatetime()),
    format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
)

# Extract start and end times from the slider
start_datetime = time_range[0]
end_datetime = time_range[1]

# Filter signal based on the selected time range
time_mask = (datetime_signal >= start_datetime) & (datetime_signal <= end_datetime)
signal_subset = signal[time_mask]
datetime_subset = datetime_signal[time_mask]
signal_subset_df = signal_df[
    (signal_df['datetime'] >= start_datetime) & 
    (signal_df['datetime'] <= end_datetime)
]

# Slider configurations
slider_config = [
    ("Broad Low Cutoff (Hz)", "BROAD_LOW_CUTOFF", 0.01, 25.0, float, 0.01),
    ("Broad High Cutoff (Hz)", "BROAD_HIGH_CUTOFF", 0.01, 25.0, float, 0.01),
    ("Narrow Low Cutoff (Hz)", "NARROW_LOW_CUTOFF", 0.01, 15.0, float, 0.01),
    ("Narrow High Cutoff (Hz)", "NARROW_HIGH_CUTOFF", 0.01, 15.0, float, 0.01),
    ("Filter Order", "FILTER_ORDER", 1, 10, int, 1),
    ("Spike Threshold", "SPIKE_THRESHOLD", 100, 1000, int, 10),
    ("Smoothing Window (s)", "SMOOTH_SEC_MULTIPLIER", 0.01, 2.0, float, 0.05),
    ("Normalization Window Size (s)", "WINDOW_SIZE_MULTIPLIER", 0.1, 20.0, float, 0.05),
    ("Normalization Noise", "NORMALIZATION_NOISE", 1e-12, 1e-8, float, 1e-12),
    ("Peak Height", "PEAK_HEIGHT", -2.0, 2.0, float, 0.1),
    ("Peak Distance (s)", "PEAK_DISTANCE_SEC", 0.01, 5.0, float, 0.05),
    ("Search Radius (s)", "SEARCH_RADIUS_SEC", 0.1, 2.0, float, 0.05),
    ("Minimum Peak Height", "MIN_PEAK_HEIGHT", 0.1, 1000, int, 10),
    ("Maximum Peak Height", "MAX_PEAK_HEIGHT", 50, 50000, int, 10),
]

# Checkbox configurations
checkbox_config = [
    ("Enable Bandpass Filtering", "enable_bandpass"),
    ("Enable Spike Removal", "enable_spike_removal"),
    ("Enable Absolute Transformation", "enable_absolute"),
    ("Enable Smoothing", "enable_smoothing"),
    ("Enable Normalization", "enable_normalization"),
    ("Enable Refinement", "enable_refinement"),
]

# Retrieve current parameters from the configuration file
default_params = config_manager.get_from_config(
    variable_names=[key for _, key, _, _, _, _ in slider_config] +
                   [key for _, key in checkbox_config],
    section="hr_peak_detection_settings" if detection_mode == "heart_rate" else "stroke_peak_detection_settings"
)

# Streamlit UI for sliders and checkboxes
st.sidebar.subheader("Adjust Peak Detection Parameters")

# Adjust sliders
for label, key, min_val, max_val, dtype, step in slider_config:
    default_value = default_params.get(key, dtype(min_val))
    params[key] = st.sidebar.slider(label, dtype(min_val), dtype(max_val), dtype(default_value), step)

# Adjust checkboxes
for label, key in checkbox_config:
    default_value = default_params.get(key, True)
    params[key] = st.sidebar.checkbox(label, value=default_value)

# Save updated configuration
if st.sidebar.button("Save Configuration"):
    section = "hr_peak_detection_settings" if detection_mode == "heart_rate" else "stroke_peak_detection_settings"
    config_manager.add_to_config(entries=params, section=section)
    st.success(f"{detection_mode} configuration saved successfully!")

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

notes_to_plot = {
    'heartbeat_manual_ok': {'signal': 'hr_narrow_bandpass', 'symbol': 'triangle-down', 'color': 'blue'},
    'heartbeat_auto_detect_accepted': {'signal': 'hr_narrow_bandpass', 'symbol': 'triangle-up', 'color': 'green'},
    'heartbeat_auto_detect_rejected': {'signal': 'hr_narrow_bandpass', 'symbol': 'triangle-up', 'color': 'red'},
    'strokebeat_auto_detect_accepted': {'signal': 'sr_narrow_bandpass', 'symbol': 'triangle-up', 'color': 'green'},
    'strokebeat_auto_detect_rejected': {'signal': 'sr_narrow_bandpass', 'symbol': 'triangle-up', 'color': 'red'}
}

TARGET_SAMPLING_RATE = 25 if detection_mode == "heart_rate" else 10

if detection_mode == "heart_rate":
    fig = plot_tag_data_interactive(
        data_pkl=data_pkl,
        sensors=['ecg'],
        derived_data_signals=['depth', 'prh', 'heart_rate', 'hr_broad_bandpass',
                            'hr_narrow_bandpass', 'hr_smoothed',
                            'hr_normalized'],
        channels={}, #'corrected_gyr': ['broad_bandpassed_signal']
        time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
        note_annotations=notes_to_plot,
        color_mapping_path=color_mapping_path,
        target_sampling_rate=TARGET_SAMPLING_RATE,
        zoom_start_time=start_datetime,
        zoom_end_time=end_datetime,
        zoom_range_selector_channel='depth',
        plot_event_values=[],
    )

    # Update the legend position
    fig.update_layout(
        legend=dict(
            visible=False,
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
else:
    fig = plot_tag_data_interactive(
        data_pkl=data_pkl,
        sensors=['ecg'],
        derived_data_signals=['depth', 'prh', 'stroke_rate', 'sr_broad_bandpass',
                            'sr_narrow_bandpass', 'sr_smoothed',
                            'sr_normalized'],
        channels={}, #'corrected_gyr': ['broad_bandpassed_signal']
        time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
        note_annotations=notes_to_plot,
        color_mapping_path=color_mapping_path,
        target_sampling_rate=25,
        zoom_start_time=start_datetime,
        zoom_end_time=end_datetime,
        zoom_range_selector_channel='depth',
        plot_event_values=[],
    )

    # Update the legend position
    fig.update_layout(
        legend=dict(
            visible=False,
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

#fig.show()

st.plotly_chart(fig)

# Debugging: Display updated params
st.write("Updated Parameters:", params)

# Add a button to clear intermediate signals in the Streamlit UI
if st.sidebar.button("Clear Intermediate Signals"):
    clear_intermediate_signals(data_pkl)
    st.sidebar.success("Intermediate signals cleared successfully!")
