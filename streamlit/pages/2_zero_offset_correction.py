import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Import pyologger utilities
from pyologger.utils.event_manager import *
from pyologger.utils.folder_manager import *
from pyologger.calibrate_data.zoc import *
from pyologger.analyze_data.find_segments import *
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
        "smoothing_window", "downsampled_sampling_rate", "baseline_adjust",
        "use_flat_chunks", "min_flat_chunks_for_zoc",
        "disable_automatic_sign_flipping", "conversion_factor",
        "logger_restart_pressure_threshold"
    ],
    section="dive_detection_settings"
)

# Default settings
default_settings = {
    "first_deriv_threshold": 0.1, "min_duration": 30, "depth_threshold": 5,
    "apply_temp_correction": False, "min_depth_threshold": 0.5,
    "dive_duration_threshold": 10, "smoothing_window": 5,
    "downsampled_sampling_rate": 1, "baseline_adjust": 1.0,
    "use_flat_chunks": True, "min_flat_chunks_for_zoc": 10,
    "disable_automatic_sign_flipping": False, "conversion_factor": 1.0,
    "logger_restart_pressure_threshold": -500.0
}

# If settings are missing or None, initialize them
if dive_detection_settings is None:
    dive_detection_settings = {}
elif any(v is None for v in dive_detection_settings.values()):  
    # Fill in only missing/None values
    dive_detection_settings = {k: v if dive_detection_settings.get(k) is not None else default_settings[k] for k, v in dive_detection_settings.items()}

# Build resolved settings from config with defaults only for missing keys.
resolved_settings = {
    k: dive_detection_settings.get(k, default_settings[k])
    if dive_detection_settings.get(k) is not None else default_settings[k]
    for k in default_settings
}

# **Step 3: Store Parameters in Session State**
deployment_session_key = f"{dataset_id}::{deployment_id}"
if st.session_state.get("calibration_params_deployment") != deployment_session_key:
    st.session_state.calibration_params = resolved_settings.copy()
    st.session_state.calibration_params_deployment = deployment_session_key

# **Step 4: Sliders & Checkboxes for Parameters**
st.sidebar.title("Calibration Parameters")
widget_prefix = deployment_session_key.replace(":", "_")

# Normalize config values to slider-safe numeric types.
def _safe_num(raw_value, min_val, max_val, cast):
    try:
        value = cast(raw_value)
    except (TypeError, ValueError):
        value = cast(min_val)
    if value < min_val:
        value = cast(min_val)
    if value > max_val:
        value = cast(max_val)
    return value

# Combined slider + manual number input control.
def _slider_with_manual(label, key, min_val, max_val, default_raw, cast, step):
    default_value = _safe_num(default_raw, min_val, max_val, cast)
    slider_value = st.sidebar.slider(
        label,
        cast(min_val),
        cast(max_val),
        cast(default_value),
        step,
        key=f"{widget_prefix}_slider_{key}"
    )
    manual_value = st.sidebar.number_input(
        f"{label} (Manual)",
        min_value=cast(min_val),
        max_value=cast(max_val),
        value=cast(slider_value),
        step=step,
        key=f"{widget_prefix}_manual_{key}"
    )
    return cast(manual_value)

# Calibration Parameters
params = st.session_state.calibration_params
params["first_deriv_threshold"] = _slider_with_manual(
    "Flat Chunk Threshold",
    "first_deriv_threshold",
    0.01,
    1.0,
    params.get("first_deriv_threshold"),
    float,
    0.01
)
params["min_duration"] = _slider_with_manual(
    "Minimum Duration (s)",
    "min_duration",
    1,
    100,
    params.get("min_duration"),
    int,
    1
)
params["depth_threshold"] = _slider_with_manual(
    "Max Depth for Surface Interval (m)",
    "depth_threshold",
    -10.0,
    25.0,
    params.get("depth_threshold"),
    float,
    0.1
)
params["baseline_adjust"] = _slider_with_manual(
    "Baseline Adjustment (m)",
    "baseline_adjust",
    -10.0,
    10.0,
    params.get("baseline_adjust"),
    float,
    0.1
)
params["conversion_factor"] = _slider_with_manual(
    "Conversion Factor",
    "conversion_factor",
    0.0,
    100.0,
    params.get("conversion_factor"),
    float,
    0.1
)
params["logger_restart_pressure_threshold"] = _slider_with_manual(
    "Logger Restart Pressure Threshold (m)",
    "logger_restart_pressure_threshold",
    -10000.0,
    0.0,
    params.get("logger_restart_pressure_threshold"),
    float,
    1.0
)
params["apply_temp_correction"] = st.sidebar.checkbox(
    "Apply Temperature Correction",
    bool(params.get("apply_temp_correction", False)),
    key=f"{widget_prefix}_apply_temp_correction"
)
params["disable_automatic_sign_flipping"] = st.sidebar.checkbox(
    "Disable Automatic Sign Flipping",
    bool(params.get("disable_automatic_sign_flipping", False)),
    key=f"{widget_prefix}_disable_automatic_sign_flipping"
)
params["use_flat_chunks"] = st.sidebar.checkbox(
    "Use Flat Chunks For ZOC",
    bool(params.get("use_flat_chunks", True)),
    key=f"{widget_prefix}_use_flat_chunks"
)

# Dive Detection Parameters
st.sidebar.title("Dive Detection Parameters")
params["min_depth_threshold"] = _slider_with_manual(
    "Min Depth for Dives (m)",
    "min_depth_threshold",
    0.1,
    10.0,
    params.get("min_depth_threshold"),
    float,
    0.1
)
params["dive_duration_threshold"] = _slider_with_manual(
    "Min Dive Duration (s)",
    "dive_duration_threshold",
    1,
    100,
    params.get("dive_duration_threshold"),
    int,
    1
)
params["smoothing_window"] = _slider_with_manual(
    "Smoothing Window (samples)",
    "smoothing_window",
    1,
    20,
    params.get("smoothing_window"),
    int,
    1
)
params["downsampled_sampling_rate"] = _slider_with_manual(
    "Downsample Rate (Hz)",
    "downsampled_sampling_rate",
    1,
    25,
    params.get("downsampled_sampling_rate"),
    int,
    1
)
params["min_flat_chunks_for_zoc"] = _slider_with_manual(
    "Min Flat Chunks For ZOC",
    "min_flat_chunks_for_zoc",
    1,
    100,
    params.get("min_flat_chunks_for_zoc"),
    int,
    1
)

# **Step 5: Process Depth Data**
depth_data = data_pkl.signal_data["pressure"]["pressure"].copy()
depth_datetime = data_pkl.signal_data["pressure"]["datetime"]
depth_fs = data_pkl.signal_info["pressure"]["sampling_frequency"]
original_pressure_unit = data_pkl.signal_info["pressure"].get("original_units", "unknown")
target_pressure_unit = data_pkl.signal_info["pressure"].get("units", "unknown")
logger_manufacturer = str(data_pkl.signal_info["pressure"].get("logger_manufacturer", "")).strip()
st.sidebar.write(f"Original pressure units: `{original_pressure_unit}`")
st.sidebar.write(f"Target pressure units: `{target_pressure_unit}`")
if 'temperature-ext' in data_pkl.signal_data:
    temp_data = data_pkl.signal_data['temperature-ext']['temp-ext']
elif 'temperature-int' in data_pkl.signal_data:
    temp_data = data_pkl.signal_data['temperature-int']['temp-int']
elif 'temperature' in data_pkl.signal_data and 'temp' in data_pkl.signal_data['temperature'].columns:
    temp_data = data_pkl.signal_data['temperature']['temp']
else:
    temp_data = None

restart_threshold = float(st.session_state.calibration_params.get("logger_restart_pressure_threshold", -500.0))
if logger_manufacturer == "Evolocus":
    pressure_df = pd.DataFrame({
        "datetime": depth_datetime,
        "pressure": depth_data
    })
    restarts = find_segments(
        data=pressure_df,
        column="pressure",
        criteria=lambda x: x < restart_threshold,
        min_duration=None
    )
    if restarts is not None and not restarts.empty:
        data_pkl.event_data = create_state_event(
            state_df=restarts,
            key="logger_restart",
            start_time_column="start_datetime",
            duration_column="duration",
            description="Detected logger restart from extreme pressure",
            long_description=(
                "Logger restart inferred from pressure values dropping below "
                f"{restart_threshold}, typically inserted by logger hardware during reboot."
            ),
            existing_events=data_pkl.event_data
        )
        for _, row in restarts.iterrows():
            start = row["start_datetime"]
            end = row["end_datetime"]
            mask = (depth_datetime >= start) & (depth_datetime <= end)
            buffer_before = depth_datetime.shift(1)
            buffer_after = depth_datetime.shift(-1)
            buffer_mask = (
                ((buffer_before >= start) & (buffer_before <= end)) |
                ((buffer_after >= start) & (buffer_after <= end))
            )
            depth_data.loc[mask | buffer_mask] = np.nan
        st.sidebar.info(
            f"Applied logger restart cleanup at threshold < {restart_threshold} m "
            f"({len(restarts)} segment(s))."
        )
    if (depth_data < restart_threshold).any():
        depth_data.loc[depth_data < restart_threshold] = np.nan
        st.sidebar.info(f"Removed residual pressure values < {restart_threshold} m.")

depth_processing_params = {
    "original_sampling_rate": depth_fs,
    "downsampled_sampling_rate": st.session_state.calibration_params["downsampled_sampling_rate"],
    "baseline_adjust": st.session_state.calibration_params["baseline_adjust"]
}

interpolated_depth_data = depth_data.interpolate(limit_direction='both')
conversion_factor = float(st.session_state.calibration_params.get("conversion_factor", 1.0))
conversion_factor = max(0.0, min(100.0, conversion_factor))
st.session_state.calibration_params["conversion_factor"] = conversion_factor
if conversion_factor != 1.0:
    depth_data = depth_data * conversion_factor
    interpolated_depth_data = interpolated_depth_data * conversion_factor
    st.sidebar.info(f"Applied conversion factor: {conversion_factor}")
first_derivative, downsampled_depth = smooth_downsample_derivative(interpolated_depth_data, **depth_processing_params)

# Ensure sign check happens after baseline adjustment.
downsampled_valid = pd.Series(downsampled_depth).dropna()
if bool(st.session_state.calibration_params.get("disable_automatic_sign_flipping", False)):
    st.sidebar.info("Automatic sign flipping disabled.")
elif downsampled_valid.empty:
    st.sidebar.warning("No valid baseline-adjusted depth values for sign check.")
else:
    neg_count = int((downsampled_valid < 0).sum())
    pos_count = int((downsampled_valid > 0).sum())
    if neg_count > pos_count:
        depth_data = depth_data * -1
        interpolated_depth_data = interpolated_depth_data * -1
        downsampled_depth = downsampled_depth * -1
        first_derivative = first_derivative * -1
        st.sidebar.info(f"Pressure sign flipped after baseline adjust ({neg_count} negative vs {pos_count} positive samples).")

# Adjust datetime indexing
downsample_step = int(depth_fs / st.session_state.calibration_params["downsampled_sampling_rate"])
if downsample_step <= 0:
    depth_downsampled_datetime = depth_datetime.copy()
else:
    depth_downsampled_datetime = depth_datetime.iloc[::downsample_step]

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
use_flat_chunks = bool(st.session_state.calibration_params["use_flat_chunks"])
min_flat_chunks_for_zoc = int(st.session_state.calibration_params["min_flat_chunks_for_zoc"])
enough_flat_chunks = len(flat_chunks) >= min_flat_chunks_for_zoc

if use_flat_chunks and enough_flat_chunks:
    corrected_depth_temp, corrected_depth_no_temp, depth_correction = apply_zero_offset_correction(
        depth=downsampled_depth,
        temp=temp_data.values if temp_data is not None else None,
        flat_chunks=flat_chunks
    )
    corrected_depth = (
        corrected_depth_temp
        if st.session_state.calibration_params["apply_temp_correction"]
        else corrected_depth_no_temp
    )
    st.sidebar.info(
        f"Applied ZOC using {len(flat_chunks)} flat chunks "
        f"(minimum required: {min_flat_chunks_for_zoc})."
    )
else:
    corrected_depth = downsampled_depth.copy()
    depth_correction = np.zeros_like(downsampled_depth)
    if not use_flat_chunks:
        st.sidebar.warning("ZOC skipped because 'Use Flat Chunks For ZOC' is disabled.")
    else:
        st.sidebar.warning(
            f"ZOC skipped: {len(flat_chunks)} flat chunks found, "
            f"{min_flat_chunks_for_zoc} required. Using baseline-adjusted depth only."
        )
data_pkl.signal_data["depth"] = pd.DataFrame({"datetime": depth_downsampled_datetime, "depth": corrected_depth})

# **Step 8: Detect Dives**
try:
    dives = find_dives(
        depth_series=corrected_depth,
        datetime_data=depth_downsampled_datetime,
        min_depth_threshold=st.session_state.calibration_params["min_depth_threshold"],
        sampling_rate=st.session_state.calibration_params["downsampled_sampling_rate"],
        duration_threshold=st.session_state.calibration_params["dive_duration_threshold"],
        smoothing_window=st.session_state.calibration_params["smoothing_window"]
    )
except ValueError as e:
    # Defensive fallback for edge cases from older find_dives implementations (e.g., empty argmin windows).
    st.sidebar.warning(f"Dive detection skipped due to edge-case windowing error: {e}")
    dives = pd.DataFrame(columns=["start_time", "end_time", "max_depth", "dive_duration"])

# Update dive duration safely (find_dives returns empty df with no columns when no dives are found)
required_dive_cols = {"start_time", "end_time", "max_depth"}
has_dive_columns = isinstance(dives, pd.DataFrame) and required_dive_cols.issubset(set(dives.columns))

if has_dive_columns and len(dives) > 0:
    dives["dive_duration"] = (dives["end_time"] - dives["start_time"]).dt.total_seconds()

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
else:
    dives = pd.DataFrame(columns=["start_time", "end_time", "max_depth", "dive_duration"])
    st.sidebar.warning(
        "No dives detected with current settings. "
        "Try lowering 'Min Depth for Dives (m)' or adjusting baseline/flat-chunk parameters."
    )

# **Step 10: Interactive Plot**
st.sidebar.title("Processing Results")
st.sidebar.write(f"✅ {len(flat_chunks)} surface intervals detected.")
st.sidebar.write(f"✅ {len(dives)} dives detected.")
st.sidebar.write(
    f"Depth range after correction: {float(np.nanmin(corrected_depth)):.3f} to {float(np.nanmax(corrected_depth)):.3f} m"
)

# Ensure plot time comparisons use a single timezone context.
deployment_tz_name = str(data_pkl.deployment_info.get("Time Zone", "")).strip()
if not deployment_tz_name:
    st.error("No Time Zone set for deployment.")
    st.stop()
try:
    _ = pd.Timestamp.now(tz=deployment_tz_name)
except Exception:
    st.error(f"Invalid Time Zone set for deployment: {deployment_tz_name}")
    st.stop()

def _normalize_one_timestamp(value, tz_name):
    if pd.isna(value):
        return pd.NaT
    ts = pd.Timestamp(value)
    return ts.tz_localize(tz_name) if ts.tzinfo is None else ts.tz_convert(tz_name)

if isinstance(getattr(data_pkl, "event_data", None), pd.DataFrame) and "datetime" in data_pkl.event_data.columns:
    data_pkl.event_data["datetime"] = data_pkl.event_data["datetime"].apply(
        lambda x: _normalize_one_timestamp(x, deployment_tz_name)
    )

depth_plot_dt = pd.to_datetime(depth_downsampled_datetime)
if isinstance(depth_plot_dt, pd.Series):
    if depth_plot_dt.dt.tz is None:
        depth_plot_dt = depth_plot_dt.dt.tz_localize(deployment_tz_name)
    else:
        depth_plot_dt = depth_plot_dt.dt.tz_convert(deployment_tz_name)
else:
    if depth_plot_dt.tz is None:
        depth_plot_dt = depth_plot_dt.tz_localize(deployment_tz_name)
    else:
        depth_plot_dt = depth_plot_dt.tz_convert(deployment_tz_name)

plot_time_range = (depth_plot_dt.min(), depth_plot_dt.max())

fig = plot_tag_data_interactive_st(
    data_pkl=data_pkl,
    signals=['pressure','depth'],
    time_range=plot_time_range,
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
