import os
import streamlit as st
import pickle

# Import pyologger utilities
from pyologger.utils.event_manager import *
from pyologger.utils.folder_manager import *
from pyologger.plot_data.plotter import plot_tag_data_interactive

# Load configuration
config, data_dir, color_mapping_path, channel_mapping_path = load_configuration()

# **Step 1: Deployment Selection**
st.sidebar.title("Deployment Selection")

# Load dataset & deployment
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, config_manager = select_and_load_deployment_streamlit(data_dir)

if not dataset_id or not deployment_id:
    st.sidebar.warning("⚠ Please select a dataset and deployment.")
    st.stop()

st.sidebar.write(f"📂 Selected Deployment: {deployment_id}")

# Sidebar for time range selection
imu_logger_to_use = 'CC-96'
ephys_logger_to_use = 'UF-01'

imu_df = data_pkl.data[imu_logger_to_use]
ephys_df = data_pkl.data[ephys_logger_to_use]
overlap_start_time = max(imu_df['datetime'].min(), ephys_df['datetime'].min()).to_pydatetime()
overlap_end_time = min(imu_df['datetime'].max(), ephys_df['datetime'].max()).to_pydatetime()

st.sidebar.title("Select Time Range")
start_time = st.sidebar.slider("Start Time", value=overlap_start_time, min_value=overlap_start_time, max_value=overlap_end_time, format="YYYY-MM-DD HH:mm:ss")
end_time = st.sidebar.slider("End Time", value=overlap_end_time, min_value=overlap_start_time, max_value=overlap_end_time, format="YYYY-MM-DD HH:mm:ss")

# Interactive color pickers
st.sidebar.header("Customize Colors")
for key in color_mapping.keys():
    new_color = st.sidebar.color_picker(f"Select color for {key}", value=color_mapping[key])
    if new_color != color_mapping[key]:
        color_mapping[key] = new_color  # Update color in memory

# Define notes to plot
notes_to_plot = {
    'heartbeat_manual_ok': 'ecg',
    'exhalation_breath': 'depth'
}

# Display title
st.title('Interactive Plot Customization')

# Plotting
fig = plot_tag_data_interactive(data_pkl, 
                                imu_channels=['depth', 'corrdepth', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ', 'magX', 'magY', 'magZ'], 
                                ephys_channels=['ecg'], 
                                imu_logger=imu_logger_to_use, 
                                ephys_logger=ephys_logger_to_use, 
                                time_range=(start_time, end_time), 
                                note_annotations=notes_to_plot, 
                                color_mapping=color_mapping)

# Display the plot
st.plotly_chart(fig)

# Save color mapping changes if needed
if st.sidebar.button('Save Color Mappings'):
    save_color_mapping(color_mapping, color_mapping_path)
    st.success("Color mappings saved.")
