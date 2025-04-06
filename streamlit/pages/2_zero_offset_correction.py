import os
import streamlit as st
import pickle

# Import pyologger utilities
from pyologger.utils.event_manager import *
from pyologger.utils.folder_manager import *
from pyologger.calibrate_data.zoc import *
from pyologger.plot_data.plotter import plot_tag_data_interactive_st

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
    "first_deriv_threshold": 0.1, "min_duration": 30, "depth_threshold": 5,
    "apply_temp_correction": False, "min_depth_threshold": 0.5,
    "dive_duration_threshold": 10, "smoothing_window": 5,
    "downsampled_sampling_rate": 1, "baseline_adjust": 1.0
}

# If settings are missing or None, initialize them
if dive_detection_settings is None:
    dive_detection_settings = default_settings
elif any(v is None for v in dive_detection_settings.values()):  
    # Fill in only missing/None values
    dive_detection_settings = {k: v if dive_detection_settings.get(k) is not None else default_settings[k] for k, v in dive_detection_settings.items()}

# Save if changes were made
param_manager.add_to_config(entries=dive_detection_settings, section="dive_detection_settings")

# **Step 3: Store Parameters in Session State**
if "calibration_params" not in st.session_state:
    st.session_state.calibration_params = dive_detection_settings

# **Step 4: Sliders & Checkboxes for Parameters**
st.sidebar.title("Calibration Parameters")

# Calibration Parameters
params = st.session_state.calibration_params
params["first_deriv_threshold"] = st.sidebar.slider("Flat Chunk Threshold", 0.01, 1.0, dive_detection_settings["first_deriv_threshold"], 0.01)
params["min_duration"] = st.sidebar.slider("Minimum Duration (s)", 1, 100, dive_detection_settings["min_duration"], 1)
params["depth_threshold"] = st.sidebar.slider("Max Depth for Surface Interval (m)", -10, 25, dive_detection_settings["depth_threshold"], 1)
params["baseline_adjust"] = st.sidebar.slider("Baseline Adjustment (m)", -10.0, 10.0, dive_detection_settings["baseline_adjust"], 0.5)
params["apply_temp_correction"] = st.sidebar.checkbox("Apply Temperature Correction", dive_detection_settings["apply_temp_correction"])

# Dive Detection Parameters
st.sidebar.title("Dive Detection Parameters")
params["min_depth_threshold"] = st.sidebar.slider("Min Depth for Dives (m)", 0.1, 10.0, dive_detection_settings["min_depth_threshold"], 0.1)
params["dive_duration_threshold"] = st.sidebar.slider("Min Dive Duration (s)", 1, 100, dive_detection_settings["dive_duration_threshold"], 1)
params["smoothing_window"] = st.sidebar.slider("Smoothing Window (samples)", 1, 20, dive_detection_settings["smoothing_window"], 1)
params["downsampled_sampling_rate"] = st.sidebar.slider("Downsample Rate (Hz)", 1, 25, dive_detection_settings["downsampled_sampling_rate"], 1)

# **Step 5: Process Depth Data**
depth_data = data_pkl.sensor_data["pressure"]["pressure"]
depth_datetime = data_pkl.sensor_data["pressure"]["datetime"]
depth_fs = data_pkl.sensor_info["pressure"]["sampling_frequency"]
temp_data = data_pkl.sensor_data['temperature']['temp']

depth_processing_params = {
    "original_sampling_rate": depth_fs,
    "downsampled_sampling_rate": st.session_state.calibration_params["downsampled_sampling_rate"],
    "baseline_adjust": st.session_state.calibration_params["baseline_adjust"]
}

first_derivative, downsampled_depth = smooth_downsample_derivative(depth_data, **depth_processing_params)

# Adjust datetime indexing
depth_downsampled_datetime = depth_datetime.iloc[::int(depth_fs / st.session_state.calibration_params["downsampled_sampling_rate"])]
if len(depth_downsampled_datetime) > len(downsampled_depth):
    depth_downsampled_datetime = depth_downsampled_datetime[:len(downsampled_depth)]

# **Step 6: Detect Flat Chunks**
flat_chunks = detect_flat_chunks(
    depth=downsampled_depth,
    datetime_data=depth_downsampled_datetime,
    first_derivative=first_derivative,
    threshold=st.session_state.calibration_params["first_deriv_threshold"],
    min_duration=st.session_state.calibration_params["min_duration"],
    depth_threshold=st.session_state.calibration_params["depth_threshold"],
    original_sampling_rate=depth_fs,
    downsampled_sampling_rate=st.session_state.calibration_params["downsampled_sampling_rate"]
)

# **Step 7: Apply Zero Offset Correction**
corrected_depth_temp, corrected_depth_no_temp, depth_correction = apply_zero_offset_correction(
    depth=downsampled_depth,
    temp=temp_data.values if temp_data is not None else None,
    flat_chunks=flat_chunks
)

corrected_depth = corrected_depth_temp if st.session_state.calibration_params["apply_temp_correction"] else corrected_depth_no_temp

# **Step 8: Detect Dives**
dives = find_dives(
    depth_series=corrected_depth,
    datetime_data=depth_downsampled_datetime,
    min_depth_threshold=st.session_state.calibration_params["min_depth_threshold"],
    sampling_rate=st.session_state.calibration_params["downsampled_sampling_rate"],
    duration_threshold=st.session_state.calibration_params["dive_duration_threshold"],
    smoothing_window=st.session_state.calibration_params["smoothing_window"]
)

# Update dive duration
dives['dive_duration'] = (dives['end_time'] - dives['start_time']).dt.total_seconds()

# **Step 9: Update Event Data**
data_pkl.event_data = create_state_event(
    state_df=dives,
    key="dive",
    value_column="max_depth",
    start_time_column="start_time",
    duration_column="dive_duration",
    description="dive_start",
    existing_events=data_pkl.event_data
)

# **Step 10: Interactive Plot**
st.sidebar.title("Processing Results")
st.sidebar.write(f"✅ {len(flat_chunks)} surface intervals detected.")
st.sidebar.write(f"✅ {len(dives)} dives detected.")

fig = plot_tag_data_interactive_st(
    data_pkl=data_pkl,
    sensors=['pressure'],
    derived_data_signals=['depth'],
    time_range=(depth_downsampled_datetime.min(), depth_downsampled_datetime.max()),
    note_annotations={"dive": {"signal": "depth", "symbol": "triangle-down", "color": "blue"}},
    state_annotations={"dive": {"signal": "depth", "color": "rgba(150, 150, 150, 0.3)"}},
    color_mapping_path=color_mapping_path,
    target_sampling_rate=1
)
st.plotly_chart(fig)

# **Step 11: Save Pickle**
if st.sidebar.button("Update pickle"):
    pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)

    st.success("✅ Data processing complete. Pickle file updated.")

# **Step 12: Update Configuration JSON**
if st.sidebar.button("Update configuration JSON"):
    param_manager.add_to_config(entries=st.session_state.calibration_params, section="dive_detection_settings")
    st.success("✅ Configuration JSON updated.")
