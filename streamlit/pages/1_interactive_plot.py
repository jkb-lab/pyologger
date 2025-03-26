import os
import streamlit as st
import pickle
import pandas as pd
from datetime import timedelta

# Import pyologger utilities
from pyologger.utils.event_manager import *
from pyologger.process_data.sampling import *
from pyologger.utils.folder_manager import *
from pyologger.calibrate_data.zoc import *
from pyologger.plot_data.plotter import plot_tag_data_interactive

# Load configuration
config, data_dir, color_mapping_path, montage_path = load_configuration()

# **Step 1: Deployment Selection**
st.sidebar.title("Deployment Selection")

# Load dataset & deployment
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment_streamlit(data_dir)

if not dataset_id or not deployment_id:
    st.sidebar.warning("⚠ Please select a dataset and deployment.")
    st.stop()

st.sidebar.write(f"📂 Selected Deployment: {deployment_id}")

# Get timezone from metadata
timezone = data_pkl.deployment_info.get("Time Zone", "UTC")

# Load time settings
time_settings = param_manager.get_from_config(
    ["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],
    section="settings"
)

# Convert timestamps and round to the nearest second
def round_to_nearest_second(dt):
    return dt.replace(microsecond=0)

try:
    OVERLAP_START_TIME = round_to_nearest_second(pd.Timestamp(time_settings["overlap_start_time"]).tz_convert(timezone))
    OVERLAP_END_TIME = round_to_nearest_second(pd.Timestamp(time_settings["overlap_end_time"]).tz_convert(timezone))
    ZOOM_WINDOW_START_TIME = round_to_nearest_second(pd.Timestamp(time_settings["zoom_window_start_time"]).tz_convert(timezone))
    ZOOM_WINDOW_END_TIME = round_to_nearest_second(pd.Timestamp(time_settings["zoom_window_end_time"]).tz_convert(timezone))
except (KeyError, TypeError) as e:
    raise ValueError(f"Missing time setting: {e}")

# **Step 2: Time Range Selection Slider**
st.subheader("Select Time Range for Truncation")

# Convert datetime range to a list of allowable values with 10-second steps
start_time = OVERLAP_START_TIME.to_pydatetime()
end_time = OVERLAP_END_TIME.to_pydatetime()
time_values = [start_time + timedelta(seconds=i) for i in range(0, int((end_time - start_time).total_seconds()) + 1, 1)]

# Double-sided slider for time range selection
time_range = st.select_slider(
    "Select Time Range",
    options=time_values,
    value=((OVERLAP_START_TIME + timedelta(minutes=4)).to_pydatetime(), (OVERLAP_END_TIME - timedelta(minutes=4)).to_pydatetime()),
    format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
)

selected_start_time = time_range[0]
selected_end_time = time_range[1]

st.write(f"📌 Selected Time Range: {selected_start_time} → {selected_end_time}")

notes_to_plot = {
    'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
    'heartbeat_auto_detect_accepted': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
    'heartbeat_auto_detect_rejected': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'red'},
    'strokebeat_auto_detect_accepted': {'signal': 'prh', 'symbol': 'triangle-up', 'color': 'green'},
    'exhalation_breath': {'signal': 'heart_rate', 'symbol': 'triangle-up', 'color': 'orange'}
}


TARGET_SAMPLING_RATE = 25
# **Step 2: Interactive Plot with Zoom**
fig = plot_tag_data_interactive(
    data_pkl=data_pkl,
    sensors=['ecg'],
    derived_data_signals=['depth','heart_rate', 'prh', 'stroke_rate'],
    note_annotations=notes_to_plot,
    zoom_start_time=selected_start_time,
    zoom_end_time=selected_end_time,
    time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
    color_mapping_path=color_mapping_path,
    target_sampling_rate=TARGET_SAMPLING_RATE,
    zoom_range_selector_channel='depth'
)

st.plotly_chart(fig, use_container_width=True)

# **Step 6: Update Configuration JSON**
if st.button("Update configuration JSON"):
    param_manager.add_to_config(entries={"selected_start_time": str(selected_start_time),
                                          "selected_end_time": str(selected_end_time)}, 
                                 section="settings")
    st.success("✅ Configuration JSON updated.")

# **Step 4: Button to Truncate Data**
if st.button("Truncate Data and Save Pickle"):
    # Update overlap window with selected range
    OVERLAP_START_TIME = pd.Timestamp(selected_start_time).tz_convert(timezone)
    OVERLAP_END_TIME = pd.Timestamp(selected_end_time).tz_convert(timezone)

    # **Truncate sensor data**
    for sensor, df in data_pkl.sensor_data.items():

        # Truncate based on selected time range
        truncated_df = df[(df.iloc[:, 0] >= OVERLAP_START_TIME) & (df.iloc[:, 0] <= OVERLAP_END_TIME)].copy()
        data_pkl.sensor_data[sensor] = truncated_df  # Save truncated version to new variable

    # **Recalculate Zoom Window** (5-minute window in the middle)
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

    st.success(f"✅ Data truncated to {OVERLAP_START_TIME} → {OVERLAP_END_TIME}")
    st.success("✅ Data processing complete. Pickle file updated.")
    st.write(f"🔍 New Zoom Window: {ZOOM_WINDOW_START_TIME} → {ZOOM_WINDOW_END_TIME}")



