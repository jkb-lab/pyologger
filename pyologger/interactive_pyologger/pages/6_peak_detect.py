import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from pyologger.process_data.peak_detect import peak_detect
from pyologger.plot_data.plotter import plot_tag_data_interactive4

# Define paths
root_dir = "/Users/jessiekb/Documents/GitHub/pyologger"
data_dir = os.path.join(root_dir, "data")
deployment_folder = os.path.join(data_dir, "2024-01-16_oror-002a_Shuka-HR")
pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')

# Load the data_reader object from the pickle file
with open(pkl_path, 'rb') as file:
    data_pkl = pickle.load(file)

# Load color mappings
color_mapping_path = os.path.join(root_dir, 'color_mappings.json')

OVERLAP_START_TIME = '2024-01-16 09:55:00'  # Start time for plotting
OVERLAP_END_TIME = '2024-01-16 10:30:00'  # End time for plotting
ZOOM_START_TIME = '2024-01-16 10:00:00'  # Start time for zooming
ZOOM_END_TIME = '2024-01-16 10:02:30'  # End time for zooming

# Set up the layout with two columns
col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed (1:2 means sliders take 1/3, plot takes 2/3)

# Column 1: Sliders for parameter adjustment
with col1:
    st.title("ECG Peak Detection")

    # Slider controls for each parameter
    low_cutoff = st.slider("Low Cutoff Frequency (Hz)", 0.1, 50.0, 2.0)
    high_cutoff = st.slider("High Cutoff Frequency (Hz)", 0.1, 50.0, 20.0)
    filter_order = st.slider("Filter Order", 1, 10, 2)
    spike_threshold = st.slider("Spike Threshold", 100, 1000, 400)
    smooth_sec_multiplier = st.slider("Smoothing Window Multiplier", 1, 10, 5)
    window_size_multiplier = st.slider("Window Size Multiplier", 1, 10, 5)
    normalization_noise = st.slider("Normalization Noise", 1e-12, 1e-8, 1e-10, format="%.2e")
    peak_height = st.slider("Peak Height", -1.0, 1.0, 0.1)
    peak_distance_sec = st.slider("Peak Distance (s)", 0.01, 1.0, 0.2)
    search_radius_sec = st.slider("Search Radius (s)", 0.1, 1.0, 0.5)
    min_peak_height = st.slider("Minimum Peak Height", 100, 1000, 500)
    max_peak_height = st.slider("Maximum Peak Height", 10000, 50000, 30000)

# Column 2: Plot the ECG data with detected peaks
with col2:
    # Extract ECG signal and sampling frequency from the loaded data
    ecg_signal = data_pkl.sensor_data['ecg']['ecg']
    sampling_rate = data_pkl.sensor_info['ecg']['sampling_frequency']

    # Run the peak detection with the current slider values
    results = peak_detect(
        signal=ecg_signal,
        sampling_rate=sampling_rate,
        lowcut=low_cutoff,
        highcut=high_cutoff,
        filter_order=filter_order,
        spike_threshold=spike_threshold,
        smooth_sec_multiplier=smooth_sec_multiplier,
        window_size_multiplier=window_size_multiplier,
        normalization_noise=normalization_noise,
        peak_height=peak_height,
        peak_distance_sec=peak_distance_sec,
        search_radius_sec=search_radius_sec,
        min_peak_height=min_peak_height,
        max_peak_height=max_peak_height
    )

    # Update the data_pkl with intermediate results for visualization
    data_pkl.sensor_data['ecg']['bandpassed_signal'] = results.get('bandpassed_signal', None)
    data_pkl.sensor_data['ecg']['spike_removed_signal'] = results.get('spike_removed_signal', None)
    data_pkl.sensor_data['ecg']['smoothed_signal'] = results.get('smoothed_signal', None) / 10
    data_pkl.sensor_data['ecg']['normalized_signal'] = results.get('normalized_signal', None) * 250 + 4000

    # Calculate RR intervals (in seconds) between successive peaks and assign to hr_data
    if len(results['peak_df']) > 1:
        rr_intervals = np.diff(results['peak_df']['refined_index']) / sampling_rate
        heart_rate = 60 / rr_intervals
        hr_data = np.full(len(ecg_signal), np.nan)
        for i in range(len(heart_rate)):
            start_idx = results['peak_df']['refined_index'].iloc[i]
            end_idx = results['peak_df']['refined_index'].iloc[i + 1]
            hr_data[start_idx:end_idx] = heart_rate[i]
        if len(results['peak_df']) > 0 and len(heart_rate) > 0:
            hr_data[results['peak_df']['refined_index'].iloc[-1]:] = heart_rate[-1]
        data_pkl.sensor_data['ecg']['hr_data'] = hr_data

    # Convert refined indices to datetime for event annotations
    matching_datetimes = data_pkl.sensor_data['ecg'].loc[results['peak_df']['refined_index'], 'datetime'].values
    utc_datetimes = pd.to_datetime(matching_datetimes).tz_localize('UTC')
    local_timezone = data_pkl.sensor_info['ecg']['sensor_start_datetime'].tz
    results['peak_df']['datetime'] = utc_datetimes.tz_convert(local_timezone)

    # Append detected peaks to the event data with heart rate values at each peak
    hr_values = hr_data[results['peak_df']['refined_index']]
    hr_events = pd.DataFrame({
        'datetime': results['peak_df']['datetime'],
        'key': results['peak_df']['key'],
        'short_description': 'calculated heart rate from detected peaks',
        'type': 'point',
        'value': hr_values
    })

    # Clear any existing events with keys that start with 'heartbeat_auto_detect'
    data_pkl.event_data['key'] = data_pkl.event_data['key'].astype(str)  # Ensure 'key' is string type
    data_pkl.event_data = data_pkl.event_data[~data_pkl.event_data['key'].str.startswith('heartbeat_auto_detect', na=False)]

    # Concatenate with hr_events
    data_pkl.event_data = pd.concat([data_pkl.event_data, hr_events], ignore_index=True)

    # Visualization: Define Notes and Plot the Results
    notes_to_plot = {
        'heartbeat_manual_ok': {'sensor': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
        'heartbeat_auto_detect_accepted': {'sensor': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
        'heartbeat_auto_detect_rejected': {'sensor': 'ecg', 'symbol': 'triangle-up', 'color': 'red'}
    }

    fig = plot_tag_data_interactive4(
        data_pkl=data_pkl,
        sensors=['ecg'],
        channels={'ecg': ['bandpassed_signal', 'spike_removed_signal', 'smoothed_signal', 'normalized_signal', 'hr_data']},
        time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
        note_annotations=notes_to_plot,
        color_mapping_path=color_mapping_path,
        target_sampling_rate=sampling_rate,
        zoom_start_time=ZOOM_START_TIME,
        zoom_end_time=ZOOM_END_TIME,
        plot_event_values=['heartbeat_manual_ok', 'heartbeat_auto_detect_accepted']
    )

    # Display the figure
    st.plotly_chart(fig)
