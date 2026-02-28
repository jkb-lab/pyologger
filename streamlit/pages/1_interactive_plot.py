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
from pyologger.utils.streamlit_time_window import standardize_time_settings

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
DATA_TZ = str(timezone or "UTC")


def _to_tzaware(value, tz_name):
    """Normalize saved timestamps into the deployment timezone."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def _to_data_naive(value):
    return _to_tzaware(value, DATA_TZ).tz_localize(None)


def _from_data_naive(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(DATA_TZ)
    return ts.tz_convert(DATA_TZ)

def round_to_nearest_second(dt):
    return dt.replace(microsecond=0)

try:
    standardized = standardize_time_settings(
        param_manager=param_manager,
        data_pkl=data_pkl,
        tz_name=DATA_TZ,
        minutes=30,
    )
    OVERLAP_START_TIME = round_to_nearest_second(standardized["overlap_start_time"])
    OVERLAP_END_TIME = round_to_nearest_second(standardized["overlap_end_time"])
    ZOOM_WINDOW_START_TIME = round_to_nearest_second(standardized["zoom_window_start_time"])
    ZOOM_WINDOW_END_TIME = round_to_nearest_second(standardized["zoom_window_end_time"])
except (KeyError, TypeError, ValueError) as e:
    st.error(f"❌ Error with time settings: {e}")
    st.stop()

# **Step 2: Time Range Selection Slider**
st.subheader("Select Time Range for Truncation")

# Convert datetime range to deployment-local naive values for slider display.
start_time = _to_data_naive(OVERLAP_START_TIME).to_pydatetime()
end_time = _to_data_naive(OVERLAP_END_TIME).to_pydatetime()
time_values = [start_time + timedelta(seconds=i) for i in range(0, int((end_time - start_time).total_seconds()) + 1, 1)]

# Double-sided slider for time range selection
time_range = st.select_slider(
    "Select Time Range",
    options=time_values,
    value=(
        _to_data_naive(ZOOM_WINDOW_START_TIME).to_pydatetime(),
        _to_data_naive(ZOOM_WINDOW_END_TIME).to_pydatetime()
    ),
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
    'exhalation_breath': {'signal': 'heart_rate', 'symbol': 'triangle-up', 'color': 'orange'},
    'dive': {"signal": "depth", "symbol": "triangle-down", "color": "blue"},
}


available_signals = list(data_pkl.signal_data.keys())
preferred_defaults = ['pressure','depth', 'prh', 'ecg', 'heart_rate', 'heart_rate_fixed']
default_signals = [sig for sig in preferred_defaults if sig in available_signals]
if not default_signals and available_signals:
    default_signals = [available_signals[0]]

saved_signal_config = param_manager.get_from_config(
    ["plotly_default_signals"], section="settings"
)
stored_signals = saved_signal_config.get("plotly_default_signals") or default_signals
stored_signals = [sig for sig in stored_signals if sig in available_signals]
if not stored_signals:
    stored_signals = default_signals

selected_signals = st.multiselect(
    "Signals to plot",
    options=available_signals,
    default=stored_signals,
    help="Default selection (pressure, depth, prh, ecg, heart_rate, heart_rate_fixed) is saved per deployment.",
)

if selected_signals and set(selected_signals) != set(stored_signals):
    param_manager.add_to_config(
        entries={"plotly_default_signals": selected_signals},
        section="settings",
    )

if not selected_signals:
    st.warning("Please select at least one signal to display.")
    st.stop()

filtered_notes = {
    key: value for key, value in notes_to_plot.items()
    if value['signal'] in selected_signals
}
state_annotations = {
    "dive": {"signal": "depth", "color": "rgba(150, 150, 150, 0.3)"}
} if "depth" in selected_signals else {}

TARGET_SAMPLING_RATE = 10
selected_start_ts = _from_data_naive(selected_start_time)
selected_end_ts = _from_data_naive(selected_end_time)

fig = plot_tag_data_interactive(
    data_pkl=data_pkl,
    signals=selected_signals,
    note_annotations=filtered_notes or None,
    state_annotations=state_annotations or None,
    zoom_start_time=selected_start_ts,
    zoom_end_time=selected_end_ts,
    time_range=(selected_start_ts, selected_end_ts),
    color_mapping_path=color_mapping_path,
    target_sampling_rate=TARGET_SAMPLING_RATE,
    zoom_range_selector_channel=selected_signals[0]
)

base_plot = getattr(fig, "figure", fig)
st.plotly_chart(base_plot, use_container_width=True)

# **Step 6: Update Configuration JSON**
if st.button("Update configuration JSON"):
    param_manager.add_to_config(entries={"selected_start_time": str(selected_start_ts),
                                          "selected_end_time": str(selected_end_ts)}, 
                                 section="settings")
    st.success("✅ Configuration JSON updated.")

# **Step 4: Button to Truncate Data**
if st.button("Truncate Data and Save Pickle"):
    # Update overlap window with selected range
    for signal, df in data_pkl.signal_data.items():

        # Truncate based on selected time range
        truncated_df = df[(df.iloc[:, 0] >= selected_start_ts) & (df.iloc[:, 0] <= selected_end_ts)].copy()
        data_pkl.signal_data[signal] = truncated_df  # Save truncated version to new variable

    # **Recalculate Zoom Window** (5-minute window in the middle)
    midpoint = selected_start_ts + (selected_end_ts - selected_start_ts) / 2
    ZOOM_WINDOW_START_TIME = midpoint - timedelta(minutes=2.5)
    ZOOM_WINDOW_END_TIME = midpoint + timedelta(minutes=2.5)

    # Save new time settings
    time_settings_update = {
        "overlap_start_time": str(selected_start_ts),
        "overlap_end_time": str(selected_end_ts),
        "zoom_window_start_time": str(ZOOM_WINDOW_START_TIME),
        "zoom_window_end_time": str(ZOOM_WINDOW_END_TIME)
    }
    param_manager.add_to_config(entries=time_settings_update, section="settings")

    pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)

    st.success(f"✅ Data truncated to {selected_start_ts} → {selected_end_ts}")
    st.success("✅ Data processing complete. Pickle file updated.")
    st.write(f"🔍 New Zoom Window: {ZOOM_WINDOW_START_TIME} → {ZOOM_WINDOW_END_TIME}")
