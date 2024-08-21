Examples
========

Installation/Usage:
*******************
Currently, ``pyologger`` is not available on PyPi, so it cannot be installed using pip.

For now, the suggested method is to clone the repository from GitHub and place the necessary modules in your project directory. Then, you can import the required classes and functions from ``pyologger`` as shown in the example below.

Make sure you have all the dependencies installed, including `Streamlit`, `Plotly`, `Pandas`, `Numpy`, and others that ``pyologger`` relies on.

Plotting Interactive Sensor Data
********************************

.. code-block:: python

    """This example demonstrates how to use `pyologger` to load and visualize multimodal sensor data from animal-borne sensors, including depth, accelerometry, gyroscope, magnetometer, and ECG data.

    The example leverages Streamlit for an interactive interface and Plotly for plotting.
    """

    import os
    import streamlit as st
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import pickle
    from datareader import DataReader
    import pandas as pd
    import json

    # Set the root directory and paths
    os.chdir("/Users/jessiekb/Documents/GitHub/pyologger")
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, "data")
    deployment_folder = os.path.join(data_dir, "2024-01-16_oror-002a_Shuka-HR")

    # Load the data_reader object from the pickle file
    pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')

    with open(pkl_path, 'rb') as file:
        data_pkl = pickle.load(file)

    # Set Streamlit page config for light mode and wide layout
    st.set_page_config(page_title="Interactive Plot", layout="wide", initial_sidebar_state="auto")

    # Load color mappings from JSON file
    color_mapping_path = os.path.join(root_dir, 'color_mappings.json')

    # Load or initialize color mappings
    if os.path.exists(color_mapping_path):
        with open(color_mapping_path, 'r') as f:
            color_mapping = json.load(f)
    else:
        color_mapping = {
            'ECG': '#FFCCCC',
            'Depth': '#00008B',
            'Corrected Depth': '#6CA1C3',
            'Accelerometer X [m/s²]': '#6CA1C3',
            'Accelerometer Y [m/s²]': '#98FB98',
            'Accelerometer Z [m/s²]': '#FF6347',
            'Gyroscope X [mrad/s]': '#9370DB',
            'Gyroscope Y [mrad/s]': '#BA55D3',
            'Gyroscope Z [mrad/s]': '#8A2BE2',
            'Magnetometer X [µT]': '#FFD700',
            'Magnetometer Y [µT]': '#FFA500',
            'Magnetometer Z [µT]': '#FF8C00',
            'Filtered Heartbeats': '#808080',
            'Exhalation Breath': '#0000FF',
        }

    # Function to save updated color mappings to JSON
    def save_color_mapping(mapping, path):
        with open(path, 'w') as f:
            json.dump(mapping, f, indent=4)

    # Function to plot sensor data interactively
    def plot_tag_data_interactive(data_pkl, imu_channels, ephys_channels=None, imu_logger=None, ephys_logger=None, imu_sampling_rate=10, ephys_sampling_rate=50, time_range=None, note_annotations=None):
        # Streamlit color pickers for each channel type
        st.sidebar.header("Customize Colors")
        for key in color_mapping.keys():
            new_color = st.sidebar.color_picker(f"Select color for {key}", value=color_mapping[key])
            if new_color != color_mapping[key]:
                color_mapping[key] = new_color  # Update color in memory
                save_color_mapping(color_mapping, color_mapping_path)  # Save the updated color mapping

        # Plotting code (details omitted for brevity)
        # ...

    # Example usage in Streamlit
    imu_channels_to_plot = ['depth', 'corrdepth', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ', 'magX', 'magY', 'magZ']
    ephys_channels_to_plot = ['ecg']
    imu_logger_to_use = 'CC-96'
    ephys_logger_to_use = 'UF-01'

    # Get the overlapping time range
    imu_df = data_pkl.data[imu_logger_to_use]
    ephys_df = data_pkl.data[ephys_logger_to_use]
    overlap_start_time = max(imu_df['datetime'].min(), ephys_df['datetime'].min()).to_pydatetime()
    overlap_end_time = min(imu_df['datetime'].max(), ephys_df['datetime'].max()).to_pydatetime()

    # Add Streamlit slider for time range selection
    st.sidebar.title("Select Time Range")
    start_time = st.sidebar.slider("Start Time", value=overlap_start_time, min_value=overlap_start_time, max_value=overlap_end_time, format="YYYY-MM-DD HH:mm:ss")
    end_time = st.sidebar.slider("End Time", value=overlap_end_time, min_value=overlap_start_time, max_value=overlap_end_time, format="YYYY-MM-DD HH:mm:ss")

    # Define notes to plot
    notes_to_plot = {
        'heartbeat_manual_ok': 'ecg',
        'exhalation_breath': 'depth'
    }

    st.title('Interactive Plot Customization')
    plot_tag_data_interactive(data_pkl, imu_channels_to_plot, ephys_channels=ephys_channels_to_plot, imu_logger=imu_logger_to_use, ephys_logger=ephys_logger_to_use, time_range=(start_time, end_time), note_annotations=notes_to_plot)
