import os
import streamlit as st
import pickle
import threading

import pandas as pd
from datetime import timedelta
from multiprocessing import Process

import streamlit.components.v1 as components

# Import pyologger utilities
from pyologger.utils.event_manager import *
from pyologger.process_data.sampling import *
from pyologger.utils.folder_manager import *
from pyologger.calibrate_data.zoc import *
from pyologger.plot_data.plotter import plot_tag_data_interactive_st
from pyologger.plot_data.plotter import plot_tag_data_interactive
from pyologger.utils.streamlit_time_window import standardize_time_settings

# If you need FigureResampler directly (only if you build the fig here)
# from plotly_resampler import FigureResampler

# -----------------------------
#  Load configuration
# -----------------------------
config, data_dir, color_mapping_path, montage_path = load_configuration()

# **Step 1: Deployment Selection**
st.sidebar.title("Deployment Selection")

# Load dataset & deployment
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = (
    select_and_load_deployment_streamlit(data_dir)
)

if not dataset_id or not deployment_id:
    st.sidebar.warning("⚠ Please select a dataset and deployment.")
    st.stop()

st.sidebar.write(f"📂 Selected Deployment: {deployment_id}")

# Get timezone from metadata
timezone = data_pkl.deployment_info.get("Time Zone", "UTC")

def round_to_nearest_second(dt):
    return dt.replace(microsecond=0)

def _to_tzaware(value, tz_name):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def _to_display_naive(value):
    return _to_tzaware(value, timezone).tz_localize(None)


def _from_display_naive(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(timezone)
    return ts.tz_convert(timezone)


standardized = standardize_time_settings(
    param_manager=param_manager,
    data_pkl=data_pkl,
    tz_name=str(timezone),
    minutes=30,
)
OVERLAP_START_TIME = round_to_nearest_second(standardized["overlap_start_time"])
OVERLAP_END_TIME = round_to_nearest_second(standardized["overlap_end_time"])
ZOOM_WINDOW_START_TIME = round_to_nearest_second(standardized["zoom_window_start_time"])
ZOOM_WINDOW_END_TIME = round_to_nearest_second(standardized["zoom_window_end_time"])

# -----------------------------
#  Step 2: Time Range Selection Slider
# -----------------------------
st.subheader("Select Time Range for Truncation")

start_time = _to_display_naive(OVERLAP_START_TIME).to_pydatetime()
end_time = _to_display_naive(OVERLAP_END_TIME).to_pydatetime()
time_values = [
    start_time + timedelta(seconds=i)
    for i in range(0, int((end_time - start_time).total_seconds()) + 1, 1)
]

time_range = st.select_slider(
    "Select Time Range",
    options=time_values,
    value=(
        _to_display_naive(ZOOM_WINDOW_START_TIME).to_pydatetime(),
        _to_display_naive(ZOOM_WINDOW_END_TIME).to_pydatetime(),
    ),
    format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"),
)

selected_start_time = time_range[0]
selected_end_time = time_range[1]
selected_start_ts = _from_display_naive(selected_start_time)
selected_end_ts = _from_display_naive(selected_end_time)

st.write(f"📌 Selected Time Range: {selected_start_time} → {selected_end_time}")

notes_to_plot = {
    "heartbeat_manual_ok": {"signal": "ecg", "symbol": "triangle-down", "color": "blue"},
    "heartbeat_auto_detect_accepted": {
        "signal": "ecg",
        "symbol": "triangle-up",
        "color": "green",
    },
    "heartbeat_auto_detect_rejected": {
        "signal": "ecg",
        "symbol": "triangle-up",
        "color": "red",
    },
    "strokebeat_auto_detect_accepted": {
        "signal": "prh",
        "symbol": "triangle-up",
        "color": "green",
    },
    "exhalation_breath": {
        "signal": "heart_rate",
        "symbol": "triangle-up",
        "color": "orange",
    },
    "dive": {"signal": "depth", "symbol": "triangle-down", "color": "blue"},
}

TARGET_SAMPLING_RATE = 25

# -----------------------------
#  Step 3: Build interactive figure
#  (this should already be using FigureResampler internally)
# -----------------------------
fig = plot_tag_data_interactive(
    data_pkl=data_pkl,
    # signals=['ecg','depth','corrected_acc','heart_rate', 'prh', 'stroke_rate'],
    note_annotations=notes_to_plot,
    state_annotations={"dive": {"signal": "depth", "color": "rgba(150, 150, 150, 0.3)"}},
    zoom_start_time=selected_start_ts,
    zoom_end_time=selected_end_ts,
    time_range=(selected_start_ts, selected_end_ts),
    color_mapping_path=color_mapping_path,
    target_sampling_rate=TARGET_SAMPLING_RATE,
    zoom_range_selector_channel="depth",
)

# -----------------------------
#  Step 3b: Run plotly-resampler Dash app in a subprocess
#           and embed via iframe in Streamlit
# -----------------------------
# IMPORTANT:
#   This assumes that `fig` is a FigureResampler or another object
#   that has a `.show_dash(...)` method.
#   If your helper currently returns a plain go.Figure, you’ll need
#   to adjust `plot_tag_data_interactive` to construct and return
#   a FigureResampler instead.

BASE_DASH_PORT = 9025  # base port; we'll offset this per new fig

# Build a key that changes when deployment or selected range changes
fig_key = f"{deployment_id}_{selected_start_time.isoformat()}_{selected_end_time.isoformat()}"

# Initialize session state for Dash tracking
if "dash_fig_key" not in st.session_state:
    st.session_state["dash_fig_key"] = None
    st.session_state["dash_port"] = None
    st.session_state["dash_thread_counter"] = 0

if hasattr(fig, "show_dash"):
    # If the figure (deployment/time window) changed, start a new Dash server
    if st.session_state["dash_fig_key"] != fig_key:
        st.session_state["dash_thread_counter"] += 1
        port = BASE_DASH_PORT + st.session_state["dash_thread_counter"]

        # Start Dash in a background thread with the *current* fig
        t = threading.Thread(
            target=fig.show_dash,
            kwargs={"mode": "external", "port": port},
            daemon=True,
        )
        t.start()

        # Update session_state so future reruns reuse this port for this fig
        st.session_state["dash_fig_key"] = fig_key
        st.session_state["dash_port"] = port

    # Use the latest port from session_state
    components.iframe(f"http://localhost:{st.session_state['dash_port']}", height=700)

else:
    st.warning(
        "Figure does not support `.show_dash` (not a FigureResampler?). "
        "Falling back to `st.plotly_chart`."
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
#  Step 6: Update Configuration JSON
# -----------------------------
if st.button("Update configuration JSON"):
    param_manager.add_to_config(
        entries={
            "selected_start_time": str(selected_start_ts),
            "selected_end_time": str(selected_end_ts),
        },
        section="settings",
    )
    st.success("✅ Configuration JSON updated.")

# -----------------------------
#  Step 4: Button to Truncate Data
# -----------------------------
if st.button("Truncate Data and Save Pickle"):
    OVERLAP_START_TIME = selected_start_ts
    OVERLAP_END_TIME = selected_end_ts

    # Truncate signal data
    for signal, df in data_pkl.signal_data.items():
        truncated_df = df[
            (df.iloc[:, 0] >= OVERLAP_START_TIME)
            & (df.iloc[:, 0] <= OVERLAP_END_TIME)
        ].copy()
        data_pkl.signal_data[signal] = truncated_df

    # Recalculate Zoom Window (5-minute window in the middle)
    midpoint = OVERLAP_START_TIME + (OVERLAP_END_TIME - OVERLAP_START_TIME) / 2
    ZOOM_WINDOW_START_TIME = midpoint - timedelta(minutes=2.5)
    ZOOM_WINDOW_END_TIME = midpoint + timedelta(minutes=2.5)

    time_settings_update = {
        "overlap_start_time": str(OVERLAP_START_TIME),
        "overlap_end_time": str(OVERLAP_END_TIME),
        "zoom_window_start_time": str(ZOOM_WINDOW_START_TIME),
        "zoom_window_end_time": str(ZOOM_WINDOW_END_TIME),
    }
    param_manager.add_to_config(entries=time_settings_update, section="settings")

    pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)

    st.success(f"✅ Data truncated to {OVERLAP_START_TIME} → {OVERLAP_END_TIME}")
    st.success("✅ Data processing complete. Pickle file updated.")
    st.write(f"🔍 New Zoom Window: {ZOOM_WINDOW_START_TIME} → {ZOOM_WINDOW_END_TIME}")
