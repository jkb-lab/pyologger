import os
import pickle
import argparse
import sys
import pandas as pd
import numpy as np

# Ensure direct workflow execution resolves the repo-local pyologger package.
WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.event_manager import *
from pyologger.plot_data.plotter import *
from pyologger.io_operations.base_exporter import *
from pyologger.utils.data_manager import *
from pyologger.process_data.peak_detect import *
from pyologger.process_data.odba import *
from pyologger.utils.workflow_netcdf import (
    latest_processing_netcdf_path,
    netcdf_attr,
    netcdf_has_signal,
    save_step_netcdf_if_changed,
)

def _resolve_channel_alias(requested_channel, available_channels):
    """Allow x/y/z <-> ax/ay/az aliases for accel channels."""
    if requested_channel in available_channels:
        return requested_channel
    req_text = str(requested_channel)
    req_base = req_text.split("__", 1)[0]
    for ch in available_channels:
        if str(ch).split("__", 1)[0] == req_base:
            return ch
    alias_map = {
        "ax": "x", "ay": "y", "az": "z",
        "x": "ax", "y": "ay", "z": "az",
        "gx": "x", "gy": "y", "gz": "z",
    }
    alias = alias_map.get(req_base.lower())
    if alias:
        for ch in available_channels:
            if str(ch).split("__", 1)[0] == alias:
                return ch
    return requested_channel


def _normalize_signal_channel_names(data_pkl, signal_name):
    """
    Normalize suffixed channel names (e.g., gy__corrected_gyr -> gy) for one signal
    when the base names are unique.
    """
    if signal_name not in data_pkl.signal_data:
        return
    df = data_pkl.signal_data.get(signal_name)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    data_cols = [c for c in df.columns if c != "datetime"]
    if not data_cols:
        return

    rename_map = {}
    seen = set()
    collisions = set()
    for col in data_cols:
        base = str(col).split("__", 1)[0]
        if base in seen:
            collisions.add(base)
        seen.add(base)
        rename_map[col] = base

    if collisions:
        # Keep original names if normalization would collide.
        print(
            f"[stroke_detect] Channel normalization skipped for '{signal_name}' due to base-name collisions: {sorted(collisions)}"
        )
        return

    if all(str(old) == str(new) for old, new in rename_map.items()):
        return

    data_pkl.signal_data[signal_name] = df.rename(columns=rename_map)

    sig_info = data_pkl.signal_info.get(signal_name, {})
    if isinstance(sig_info, dict):
        old_channels = list(sig_info.get("channels", []))
        if old_channels:
            sig_info["channels"] = [rename_map.get(c, c) for c in old_channels]
        old_meta = sig_info.get("metadata", {})
        if isinstance(old_meta, dict) and old_meta:
            new_meta = {}
            for old_key, val in old_meta.items():
                new_meta[rename_map.get(old_key, old_key)] = val
            sig_info["metadata"] = new_meta
        data_pkl.signal_info[signal_name] = sig_info

    print(f"[stroke_detect] Normalized channel names for '{signal_name}': {rename_map}")


def _resolve_time_settings(param_manager, data_pkl):
    """Load required time settings; derive and persist fallback values when missing."""
    variables = [
        "calm_horizontal_start_time",
        "calm_horizontal_end_time",
        "zoom_window_start_time",
        "zoom_window_end_time",
        "overlap_start_time",
        "overlap_end_time",
    ]
    settings = param_manager.get_from_config(variables, section="settings")

    required_keys = [
        "overlap_start_time",
        "overlap_end_time",
        "zoom_window_start_time",
        "zoom_window_end_time",
    ]

    def _has_all_required(values):
        return all(values.get(k) is not None for k in required_keys)

    if _has_all_required(settings):
        return settings

    timezone = (getattr(data_pkl, "deployment_info", {}) or {}).get("Time Zone", "UTC")
    start_times = []
    end_times = []
    for df in data_pkl.signal_data.values():
        if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
            continue
        dt = pd.to_datetime(df["datetime"], errors="coerce").dropna()
        if dt.empty:
            continue
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(timezone)
        else:
            dt = dt.dt.tz_convert(timezone)
        start_times.append(dt.min())
        end_times.append(dt.max())

    if not start_times or not end_times:
        raise ValueError(
            "One or more required time values were not found in the config file, "
            "and no valid datetime values were available to derive fallback settings."
        )

    overlap_start_time = max(start_times)
    overlap_end_time = min(end_times)
    if overlap_start_time > overlap_end_time:
        overlap_start_time = min(start_times)
        overlap_end_time = max(end_times)

    midpoint = overlap_start_time + (overlap_end_time - overlap_start_time) / 2
    zoom_window_start = midpoint - pd.Timedelta(minutes=2.5)
    zoom_window_end = midpoint + pd.Timedelta(minutes=2.5)

    fallback_settings = {
        "overlap_start_time": str(overlap_start_time),
        "overlap_end_time": str(overlap_end_time),
        "zoom_window_start_time": str(zoom_window_start),
        "zoom_window_end_time": str(zoom_window_end),
    }
    param_manager.add_to_config(entries=fallback_settings, section="settings")
    settings.update(fallback_settings)
    print("Missing time settings were derived from signal datetimes and saved to config.")
    return settings

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Resolve deployment first so we can decide from NetCDF metadata whether the step is needed.
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
latest_netcdf_path = latest_processing_netcdf_path(deployment_folder, deployment_id)

stroke_rate_unit = str(netcdf_attr(latest_netcdf_path, "signal_info_stroke_rate_metadata_stroke_rate_unit", "") or "").lower()
stroke_rate_exists = netcdf_has_signal(latest_netcdf_path, "stroke_rate")
critical_signal_candidates = ['dynamic_accel', 'corrected_acc', 'calibrated_acc', 'accelerometer']
critical_signal_available = any(netcdf_has_signal(latest_netcdf_path, sig) for sig in critical_signal_candidates)

if stroke_rate_exists and "hz" not in stroke_rate_unit:
    print("Skipping Step 04 based on NetCDF metadata: stroke_rate already exists in final units.")
    param_manager.add_to_config("current_processing_step", "Processing Step 04 skipped: stroke_rate already available.")
    raise SystemExit(0)

if not stroke_rate_exists and not critical_signal_available:
    print(f"Skipping Step 04 based on NetCDF metadata: none of {critical_signal_candidates} are available.")
    param_manager.add_to_config("current_processing_step", "Processing Step 04 skipped: missing stroke input.")
    raise SystemExit(0)

with open(pkl_path, "rb") as file:
    data_pkl = pickle.load(file)

# Retrieve values from config (derive fallback values when missing)
settings = _resolve_time_settings(param_manager, data_pkl)

# Assign retrieved values to variables
CALM_HORIZONTAL_START_TIME = settings.get("calm_horizontal_start_time")
CALM_HORIZONTAL_END_TIME = settings.get("calm_horizontal_end_time")
ZOOM_WINDOW_START_TIME = settings.get("zoom_window_start_time")
ZOOM_WINDOW_END_TIME = settings.get("zoom_window_end_time")
OVERLAP_START_TIME = settings.get("overlap_start_time")
OVERLAP_END_TIME = settings.get("overlap_end_time")

if None in {OVERLAP_START_TIME, OVERLAP_END_TIME, ZOOM_WINDOW_START_TIME, ZOOM_WINDOW_END_TIME}:
    raise ValueError("One or more required time values were not found in the config file.")

current_processing_step = "Processing Step 04 IN PROGRESS."
param_manager.add_to_config("current_processing_step", current_processing_step)
data_changed = False

# For stroke workflow we can operate with corrected/dynamic/calibrated accelerometer signals.
critical_signal_candidates = ['dynamic_accel', 'corrected_acc', 'calibrated_acc', 'accelerometer']
critical_signal = next(
    (sig for sig in critical_signal_candidates
     if sig in data_pkl.signal_data and data_pkl.signal_data[sig] is not None),
    None
)

# Check if stroke_rate exists; if so, ensure units are spm.
converted = False
if 'stroke_rate' in data_pkl.signal_data and data_pkl.signal_data['stroke_rate'] is not None:
    sr_df = data_pkl.signal_data['stroke_rate']
    sr_info = data_pkl.signal_info.get('stroke_rate', {})
    metadata = sr_info.get('metadata', {})
    channels = sr_info.get('channels', [c for c in sr_df.columns if c != 'datetime'])

    hz_aliases = {'hz', 'hertz', '1/s', '1/sec', 'sec^-1', 's^-1', 'per second'}
    spm_aliases = {'spm', '1/min', 'min^-1', 'per minute'}

    for ch in channels:
        if ch not in sr_df.columns:
            continue
        unit = str(metadata.get(ch, {}).get('unit', '')).lower()
        is_hz = (unit in hz_aliases) or ('hz' in unit)
        is_spm = (unit in spm_aliases) or ('spm' in unit) or (('stroke' in unit) and ('min' in unit))

        if is_hz and not is_spm:
            sr_df[ch] = sr_df[ch] * 60.0
            metadata.setdefault(ch, {})
            metadata[ch]['unit'] = 'spm'
            converted = True

    if converted:
        data_changed = True
        sr_info['metadata'] = metadata
        sr_info['transformation_log'] = sr_info.get('transformation_log', [])
        sr_info['transformation_log'].append('converted_stroke_rate_hz_to_spm')
        data_pkl.signal_data['stroke_rate'] = sr_df
        data_pkl.signal_info['stroke_rate'] = sr_info
        print("Converted stroke_rate from Hz to spm.")

# If stroke_rate was converted above, persist the change immediately
if converted:
    print("✅ stroke_rate converted to spm. Saving updated data.")
    print(current_processing_step)
    param_manager.add_to_config("current_processing_step", current_processing_step)
    with open(pkl_path, 'wb') as file:
        pickle.dump(data_pkl, file)
    print("Pickle file updated.")

# Only proceed with critical signal check if stroke_rate does not already exist
if 'stroke_rate' in data_pkl.signal_data and data_pkl.signal_data['stroke_rate'] is not None:
    skip_step = True
    print("✅ stroke_rate already exists. Skipping processing.")
else:
    # If critical signal doesn't exist, create flag to skip step.
    if critical_signal is None:
        print(f"⚠️ None of the required signals were found: {critical_signal_candidates}. Skipping processing.")
        skip_step = True
        print(f'‼️ DO NOT PROCEED - Skip_step: {skip_step} due to missing required signals: {critical_signal_candidates}')
    else:
        # signals exists and can be processed normally
        skip_step = False
        print(f'✅ Proceed - Skip_step: {skip_step}. Using accelerometer source signal: {critical_signal}.')

if not skip_step:
    data_changed = True
    # Retrieve timezone from deployment info
    timezone = data_pkl.deployment_info['Time Zone']

    # Define placeholder timestamps for calm period in the retrieved timezone
    stroking_start_time = OVERLAP_START_TIME
    stroking_end_time = OVERLAP_END_TIME

    # Use ParamManager to add both stroking start and end times to the config in the desired section
    param_manager.add_to_config(
        entries={
            "stroking_start_time": str(stroking_start_time),
            "stroking_end_time": str(stroking_end_time)
        },
        section="settings"
    )

    # CHANGE AS NEEDED

    detection_mode="stroke_rate"
    overwrite = False

    # Define parent signal options
    parent_signal_options = list(data_pkl.signal_data.keys())
    if detection_mode == "heart_rate":
        default_parent_signal = "ecg"
    else:
        stroke_parent_from_config = param_manager.get_from_config(
            variable_names=["STROKE_PARENT_SIGNAL"],
            section="stroke_peak_detection_settings"
        ).get("STROKE_PARENT_SIGNAL")
        # Prefer gyro for stroke detection, but allow corrected/calibrated acc-only datasets.
        stroke_parent_candidates = ["dynamic_accel", "corrected_gyr", "gyroscope", "corrected_acc", "calibrated_acc", "accelerometer"]
        if stroke_parent_from_config in data_pkl.signal_data:
            default_parent_signal = stroke_parent_from_config
        else:
            default_parent_signal = next((sig for sig in stroke_parent_candidates if sig in data_pkl.signal_data), "corrected_gyr")

    animal_id = data_pkl.animal_info['Animal_ID']
    print(f"Detected animal ID: {animal_id}")

    # User input for parent signal
    if overwrite:
        print(f"Available parent signals: {parent_signal_options}")
        parent_signal = input(f"Choose parent signal (default: {default_parent_signal}): ").strip()
        if not parent_signal or parent_signal not in parent_signal_options:
            parent_signal = default_parent_signal
    else:
        parent_signal = default_parent_signal
    _normalize_signal_channel_names(data_pkl, parent_signal)

    # Get available channels
    if parent_signal in data_pkl.signal_data:
        available_channels = [c for c in data_pkl.signal_data[parent_signal].columns if c != 'datetime']
    elif parent_signal in data_pkl.signal_data:
        available_channels = [c for c in data_pkl.signal_data[parent_signal].columns if c != 'datetime']
    else:
        available_channels = []
    print(f"[stroke_detect] parent_signal='{parent_signal}'")
    print(f"[stroke_detect] available_channels({len(available_channels)}): {available_channels}")

    # Default channel logic
    if detection_mode == "heart_rate":
        default_channel = "ecg"
    elif detection_mode == "stroke_rate":
        stroke_channel_from_config = param_manager.get_from_config(
            variable_names=["STROKE_CHANNEL"],
            section="stroke_peak_detection_settings"
        ).get("STROKE_CHANNEL")
        print(f"[stroke_detect] STROKE_CHANNEL from config: {stroke_channel_from_config}")

        if stroke_channel_from_config:
            default_channel = _resolve_channel_alias(stroke_channel_from_config, available_channels)
            print(f"[stroke_detect] resolved config channel -> {default_channel}")
            if default_channel in available_channels:
                pass
            else:
                default_channel = None
        else:
            default_channel = None

        if default_channel is None:
            if parent_signal in ("corrected_acc", "calibrated_acc", "accelerometer", "dynamic_accel"):
                if animal_id.startswith(('nesc', 'mian')):
                    preferred_channels = ['ax', 'x', 'ay', 'y', 'az', 'z']
                elif animal_id.startswith(('oror', 'bamu')):
                    preferred_channels = ['ay', 'y', 'ax', 'x', 'az', 'z']
                else:
                    preferred_channels = ['ax', 'x', 'ay', 'y', 'az', 'z']
            else:
                if animal_id.startswith(('nesc', 'mian')):
                    preferred_channels = ['gx', 'gy', 'gz']
                elif animal_id.startswith(('oror', 'bamu')):
                    preferred_channels = ['gy', 'gx', 'gz']
                else:
                    preferred_channels = ['gy', 'gx', 'gz']
            default_channel = next((ch for ch in preferred_channels if ch in available_channels), None)
            if default_channel is None and available_channels:
                default_channel = available_channels[0]
    else:
        default_channel = available_channels[0] if available_channels else None
    print(f"[stroke_detect] default_channel candidate: {default_channel}")

    # User input for channel
    if overwrite:
        print(f"Available channels: {available_channels}")
        channel = input(f"Choose channel (default: {default_channel}): ").strip()
        if not channel or channel not in available_channels:
            channel = default_channel
    else:
        channel = default_channel
    print(f"[stroke_detect] final selected channel: {channel}")

    # Configure signals
    signal_df = data_pkl.signal_data[parent_signal]
    if channel is None or channel not in signal_df.columns:
        raise ValueError(f"Channel '{channel}' not found in parent signal '{parent_signal}'. Available channels: {available_channels}")
    signal = pd.to_numeric(signal_df[channel], errors="coerce")
    total_n = int(len(signal))
    non_na_n = int(signal.notna().sum())
    finite_n = int(np.isfinite(signal.to_numpy(dtype=float)).sum()) if total_n > 0 else 0
    print(
        f"[stroke_detect] source stats for {parent_signal}.{channel}: "
        f"n={total_n}, non_na={non_na_n}, finite={finite_n}"
    )
    if total_n == 0:
        raise ValueError(
            f"Selected source channel '{channel}' in '{parent_signal}' has zero samples. "
            "Check STROKE_PARENT_SIGNAL/STROKE_CHANNEL configuration."
        )
    if non_na_n == 0 or finite_n == 0:
        raise ValueError(
            f"Selected source channel '{channel}' in '{parent_signal}' has no valid numeric data "
            f"(non_na={non_na_n}, finite={finite_n}). "
            "This usually means the wrong source channel was selected."
        )
    if non_na_n > 0:
        sig_non_na = signal.dropna()
        print(
            f"[stroke_detect] source value range: min={sig_non_na.min():.6g}, "
            f"max={sig_non_na.max():.6g}, std={sig_non_na.std():.6g}"
        )
    datetime_signal = signal_df['datetime']
    sampling_rate = calculate_sampling_frequency(datetime_signal.head())

    # Define the default time range based on the signal's datetime column
    signal_start = datetime_signal.min()
    signal_end = datetime_signal.max()

    # Determine time range based on user input if overwrite is True
    if overwrite:
        print(f"Signal time range: {signal_start} to {signal_end}")
        start_time_input = input(f"Enter start time (default: {signal_start}): ").strip()
        end_time_input = input(f"Enter end time (default: {signal_end}): ").strip()
        start_datetime = pd.Timestamp(start_time_input) if start_time_input else signal_start
        end_datetime = pd.Timestamp(end_time_input) if end_time_input else signal_end
    else:
        start_datetime = signal_start
        end_datetime = signal_end
        
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
    subset_non_na = int(signal_subset.notna().sum())
    subset_finite = int(np.isfinite(signal_subset.to_numpy(dtype=float)).sum()) if len(signal_subset) else 0
    print(
        f"[stroke_detect] subset stats for {parent_signal}.{channel}: "
        f"n={len(signal_subset)}, non_na={subset_non_na}, finite={subset_finite}"
    )
    if len(signal_subset) == 0 or subset_non_na == 0 or subset_finite == 0:
        raise ValueError(
            f"Selected source channel '{channel}' in '{parent_signal}' has no usable data in selected time window. "
            f"subset_n={len(signal_subset)}, subset_non_na={subset_non_na}, subset_finite={subset_finite}"
        )

    # Repair NaN gaps before filtering/peak detection; scipy filters cannot operate on NaN streams.
    signal_subset = signal_subset.astype(float)
    nan_before = int(signal_subset.isna().sum())
    if nan_before > 0:
        signal_subset = signal_subset.interpolate(limit_direction="both")
        signal_subset = signal_subset.ffill().bfill()
        nan_after = int(signal_subset.isna().sum())
        print(
            f"[stroke_detect] repaired NaNs in detection input: before={nan_before}, after={nan_after}"
        )
        if nan_after > 0:
            raise ValueError(
                f"Unable to repair NaNs for detection input channel '{channel}' in '{parent_signal}'. "
                f"Remaining NaNs={nan_after}"
            )

    # Retrieve parameters for peak detection
    params = param_manager.get_from_config(
        variable_names=[
            "BROAD_LOW_CUTOFF", "BROAD_HIGH_CUTOFF", "NARROW_LOW_CUTOFF", "NARROW_HIGH_CUTOFF",
            "FILTER_ORDER", "SPIKE_THRESHOLD", "SMOOTH_SEC_MULTIPLIER", "WINDOW_SIZE_MULTIPLIER",
            "NORMALIZATION_NOISE", "PEAK_HEIGHT", "PEAK_DISTANCE_SEC", "SEARCH_RADIUS_SEC",
            "MIN_PEAK_HEIGHT", "MAX_PEAK_HEIGHT", "enable_bandpass", "enable_spike_removal",
            "enable_absolute", "enable_smoothing", "enable_normalization", "enable_refinement",
            "ANTI_DOUBLE_GAP_FACTOR", "HR_CONFLICT_RR_FACTOR", "PICK_LAST_IN_CONFLICT_PAIR",
            "DETECTION_DERIVATIVE_CHANNELS",
        ],
        section="hr_peak_detection_settings" if detection_mode == "Heart Rate" else "stroke_peak_detection_settings"
    )

    overwrite=False # If needed, change to true and rewrite settings here

    default_params = {
        "BROAD_LOW_CUTOFF": 0.05,  # Hz, lower cutoff for the broad bandpass filter
        "BROAD_HIGH_CUTOFF": 10,  # Hz, upper cutoff for the broad bandpass filter
        "NARROW_LOW_CUTOFF": 0.1,  # Hz, lower cutoff for the narrow bandpass filter
        "NARROW_HIGH_CUTOFF": 2.0,  # Hz, upper cutoff for the narrow bandpass filter
        "FILTER_ORDER": 2,  # Order of the bandpass filter, affects sharpness
        "SPIKE_THRESHOLD": 400,  # Threshold for removing large spikes (e.g., noise or artifacts)
        "SMOOTH_SEC_MULTIPLIER": 0.41,  # Multiplier for calculating the smoothing window size (3 for HR)
        "WINDOW_SIZE_MULTIPLIER": 15.5,  # Multiplier for calculating sliding window size (if this is too big it will lump all strokes into a plateau)
        "NORMALIZATION_NOISE": 1e-10,  # Small constant to avoid division by zero in normalization
        "PEAK_HEIGHT": -0.9,  # Minimum amplitude (height) for peak detection
        "PEAK_DISTANCE_SEC": 0.5,  # Minimum time between detected peaks (in seconds)
        "SEARCH_RADIUS_SEC": 0.3,  # Time range for refining the peak location (in seconds)
        "MIN_PEAK_HEIGHT": 150,  # Minimum acceptable amplitude for detected peaks; original units
        "MAX_PEAK_HEIGHT": 1000000,  # Maximum acceptable amplitude for detected peaks; original units
        "enable_bandpass": True,  # Enable/disable bandpass filtering
        "enable_spike_removal": False,  # Enable/disable spike removal
        "enable_absolute": False,  # Enable/disable abs() transformation of signal (only use if HR, not for stroke rate)
        "enable_smoothing": True,  # Enable/disable smoothing
        "enable_normalization": True,  # Enable/disable sliding window normalization
        "enable_refinement": True,  # Enable/disable peak refinement
        "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],  # candidate detection sources
        "ANTI_DOUBLE_GAP_FACTOR": 0.75,  # Generalized anti-double detection factor (kept for cross-workflow consistency)
        "HR_CONFLICT_RR_FACTOR": 0.75,  # Legacy alias for backward compatibility
        "PICK_LAST_IN_CONFLICT_PAIR": True,  # keep later peak in conflict pairs by default
    }

    if overwrite:
        params = default_params.copy()
        # Explicit overwrite writes deployment-specific values by intent.
        param_manager.add_to_config(entries=params, section="stroke_peak_detection_settings")
    else:
        # Fill only missing keys at runtime, preserving dataset-default -> deployment override precedence.
        params = {
            key: params.get(key) if params.get(key) is not None else value
            for key, value in default_params.items()
        }
        if params.get("ANTI_DOUBLE_GAP_FACTOR") is None and params.get("HR_CONFLICT_RR_FACTOR") is not None:
            params["ANTI_DOUBLE_GAP_FACTOR"] = params["HR_CONFLICT_RR_FACTOR"]
        if params.get("HR_CONFLICT_RR_FACTOR") is None and params.get("ANTI_DOUBLE_GAP_FACTOR") is not None:
            params["HR_CONFLICT_RR_FACTOR"] = params["ANTI_DOUBLE_GAP_FACTOR"]
        print("Settings loaded from config file with runtime fallback for missing values.")

    # Run peak detection
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
        enable_refinement=params["enable_refinement"],
        detection_sources=params.get("DETECTION_DERIVATIVE_CHANNELS"),
    )

    # Debug diagnostics for zero-acceptance cases.
    peak_df_dbg = results.get("peak_df", pd.DataFrame())
    n_detected = len(results.get("detected_peaks", []))
    n_refined = len(results.get("refined_peaks", [])) if "refined_peaks" in results else n_detected
    n_total = len(peak_df_dbg) if isinstance(peak_df_dbg, pd.DataFrame) else 0
    n_acc = int((peak_df_dbg.get("key", pd.Series(dtype=str)) == "beat_auto_detect_accepted").sum()) if n_total else 0
    n_rej = int((peak_df_dbg.get("key", pd.Series(dtype=str)) == "beat_auto_detect_rejected").sum()) if n_total else 0
    print(
        f"[stroke_detect] peak diagnostics: detected={n_detected}, refined={n_refined}, "
        f"total_scored={n_total}, accepted={n_acc}, rejected={n_rej}"
    )
    if n_total and "height_original" in peak_df_dbg.columns:
        h = pd.to_numeric(peak_df_dbg["height_original"], errors="coerce").dropna()
        if not h.empty:
            q = h.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
            print(
                "[stroke_detect] height_original quantiles: "
                f"p05={q.get(0.05):.6g}, p25={q.get(0.25):.6g}, p50={q.get(0.5):.6g}, "
                f"p75={q.get(0.75):.6g}, p95={q.get(0.95):.6g}; "
                f"MIN_PEAK_HEIGHT={params.get('MIN_PEAK_HEIGHT')}, MAX_PEAK_HEIGHT={params.get('MAX_PEAK_HEIGHT')}"
            )

    process_rate(data_pkl, results, signal_subset_df, parent_signal,
                params, sampling_rate, detection_mode)

    TARGET_SAMPLING_RATE = 10

    notes_to_plot = {
        'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
        'heartbeat_auto_detect_accepted': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
        'heartbeat_auto_detect_rejected': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'red'},
        'strokebeat_auto_detect_accepted': {'signal': 'sr_narrow_bandpass', 'symbol': 'triangle-up', 'color': 'green'},
        'strokebeat_auto_detect_rejected': {'signal': 'sr_narrow_bandpass', 'symbol': 'triangle-up', 'color': 'red'}
    }

    # fig = plot_tag_data_interactive(
    #     data_pkl=data_pkl,
    #     signals=['ecg', 'gyroscope','depth', 'corrected_gyr', 'prh','stroke_rate', 'sr_broad_bandpass',
    #                           'sr_narrow_bandpass', 'sr_smoothed',
    #                           'sr_normalized'],
    #     channels={}, #'corrected_gyr': ['broad_bandpassed_signal']
    #     time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
    #     note_annotations=notes_to_plot,
    #     color_mapping_path=color_mapping_path,
    #     target_sampling_rate=TARGET_SAMPLING_RATE,
    #     zoom_start_time=stroking_start_time,
    #     zoom_end_time=stroking_end_time,
    #     zoom_range_selector_channel='depth',
    #     plot_event_values=[],
    # )

    # fig.show()

    # Clear the specified keys
    keys_to_remove = ['sr_broad_bandpass','sr_narrow_bandpass', 'sr_normalized']
    clear_intermediate_signals(data_pkl, remove_keys=keys_to_remove)

    initial_event_count = len(data_pkl.event_data)
    # Remove events with keys ending in '_rejected'
    data_pkl.event_data = data_pkl.event_data[~data_pkl.event_data['key'].str.endswith('_rejected', na=False)]
    # Get the final count of events
    final_event_count = len(data_pkl.event_data)
    # Print the number of removed events
    removed_event_count = initial_event_count - final_event_count
    print(f"Removed {removed_event_count} events with keys ending in '_rejected'.")

    # Prefer corrected_acc for ODBA; fall back to calibrated_acc if needed.
    odba_source_signal = 'corrected_acc' if 'corrected_acc' in data_pkl.signal_data else 'calibrated_acc'
    corrected_acc = data_pkl.signal_data[odba_source_signal]
    acc_sampling_rate = calculate_sampling_frequency(corrected_acc['datetime'])

    stroke_rate_subset = data_pkl.signal_data['stroke_rate'][
            (data_pkl.signal_data['stroke_rate']['datetime'] >= stroking_start_time) &
            (data_pkl.signal_data['stroke_rate']['datetime'] <= stroking_end_time)
        ]

    # Calculate mean stroke rate
    mean_stroke_rate = stroke_rate_subset['stroke_rate'].mean()
    stroke_hz = mean_stroke_rate / 60.0 if pd.notna(mean_stroke_rate) else float("nan")
    print(f"Stroke rate in Hz: {stroke_hz} Hz.")
    print(f"[stroke_detect] acc_sampling_rate: {acc_sampling_rate}")

    can_compute_odba = (
        pd.notna(acc_sampling_rate) and float(acc_sampling_rate) > 0
        and pd.notna(stroke_hz) and float(stroke_hz) > 0
    )

    if not can_compute_odba:
        print(
            "⚠️ Skipping ODBA computation because stroke_hz or acc_sampling_rate is invalid. "
            f"stroke_hz={stroke_hz}, acc_sampling_rate={acc_sampling_rate}"
        )
    else:
        fh = stroke_hz / 2.0
        n = max(1, int(4 * round(float(acc_sampling_rate) / float(fh))))
        print(f"[stroke_detect] ODBA window n: {n}")

        # Calculate ODBA using VeDBA method.
        odba_df = compute_odba(corrected_acc, fs=acc_sampling_rate, method='wilson', n=n)

        # Print the first few rows of the resulting ODBA DataFrame
        print(odba_df.head())

        # Optionally store it in signal_data
        data_pkl.signal_data['odba'] = odba_df
        data_pkl.signal_info['odba'] = {
            "channels": ["odba"],
            "metadata": {
                "odba": {"original_name": "Overall Dynamic Body Acceleration (VeDBA)", "unit": "g"}
            },
            "derived_from_signals": [odba_source_signal],
            "transformation_log": [f"VeDBA calculated with n={n} from {odba_source_signal}"]
        }

    TARGET_SAMPLING_RATE = 10

    notes_to_plot = {
        'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
        'heartbeat_auto_detect_accepted': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
        'heartbeat_auto_detect_rejected': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'red'},
        'strokebeat_auto_detect_accepted': {'signal': 'sr_smoothed', 'symbol': 'triangle-up', 'color': 'green'},
        'strokebeat_auto_detect_rejected': {'signal': 'sr_smoothed', 'symbol': 'triangle-up', 'color': 'red'}
    }

    # fig = plot_tag_data_interactive(
    #     data_pkl=data_pkl,
    #     signals=['ecg', 'gyroscope','depth', 'corrected_gyr', 'prh', 'stroke_rate', 'sr_smoothed','odba'],
    #     channels={}, #'corrected_gyr': ['broad_bandpassed_signal']
    #     time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
    #     note_annotations=notes_to_plot,
    #     color_mapping_path=color_mapping_path,
    #     target_sampling_rate=TARGET_SAMPLING_RATE,
    #     zoom_start_time=stroking_start_time,
    #     zoom_end_time=stroking_end_time,
    #     zoom_range_selector_channel='depth',
    #     plot_event_values=[],
    # )

    # fig.show()
else:
    print(f"Skipping step due to missing required signals: {critical_signal_candidates}.")

current_processing_step = "Processing Step 04. Stroke rate and ODBA calculation complete."
print(current_processing_step)

# Add or update the current_processing_step for the specified deployment
param_manager.add_to_config("current_processing_step", current_processing_step)

# Optional: save new pickle file
if data_changed:
    with open(pkl_path, 'wb') as file:
            pickle.dump(data_pkl, file)
    print("Pickle file updated.")

save_step_netcdf_if_changed(data_pkl, deployment_folder, deployment_id, 4, changed=data_changed)
