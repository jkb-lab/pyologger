import os
import pickle
import argparse
import sys
import numpy as np
import pandas as pd

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
from pyologger.utils.event_manager import create_state_event
from pyologger.utils.workflow_netcdf import (
    latest_processing_netcdf_path,
    netcdf_has_signal,
    save_step_netcdf_if_changed,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Resolve deployment first so metadata-only skip paths can avoid loading data.pkl.
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
latest_netcdf_path = latest_processing_netcdf_path(deployment_folder, deployment_id)

if not netcdf_has_signal(latest_netcdf_path, "ecg"):
    print("Skipping Step 05 based on NetCDF metadata: ecg signal not available.")
    param_manager.add_to_config("current_processing_step", "Processing Step 05 skipped: missing ECG input.")
    raise SystemExit(0)

with open(pkl_path, "rb") as file:
    data_pkl = pickle.load(file)

# Retrieve values from config
variables = ["calm_horizontal_start_time", "calm_horizontal_end_time", 
             "zoom_window_start_time", "zoom_window_end_time", 
             "overlap_start_time", "overlap_end_time"]
settings = param_manager.get_from_config(variables, section="settings")

# Assign retrieved values to variables
CALM_HORIZONTAL_START_TIME = settings.get("calm_horizontal_start_time")
CALM_HORIZONTAL_END_TIME = settings.get("calm_horizontal_end_time")
ZOOM_START_TIME = settings.get("zoom_window_start_time")
ZOOM_END_TIME = settings.get("zoom_window_end_time")
OVERLAP_START_TIME = settings.get("overlap_start_time")
OVERLAP_END_TIME = settings.get("overlap_end_time")


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}

# CHANGE AS NEEDED

detection_mode = "heart_rate"
overwrite = False
data_changed = False

critical_signal = 'ecg'
# If critical signal doesn't exist, create flag to skip step.
if critical_signal not in data_pkl.signal_data or data_pkl.signal_data[critical_signal] is None:
    print(f"⚠️ Signal: {critical_signal} not found. Skipping processing.")
    skip_step = True
else:
    # signals exists and can be processed normally
    skip_step = False

if not skip_step:
    data_changed = True
    # Define parent signal options
    # parent_signal_options = list(data_pkl.signal_data.keys()) + list(data_pkl.signal_data.keys())

    # Handle default signal based on availability
    if detection_mode == "heart_rate":
        if "ecg" in data_pkl.signal_data:
            default_parent_signal = "ecg"
            default_channel = "ecg"
        else:
            print("⚠️ ECG not found. Using accelerometer (ax) as fallback for peak detection.")
            default_parent_signal = "accelerometer" if "accelerometer" in data_pkl.signal_data else parent_signal_options[0]
            default_channel = "ax" if "ax" in data_pkl.signal_data.get(default_parent_signal, {}).columns else None
    else:
        default_parent_signal = "corrected_gyr"
        default_channel = "gy"

    parent_signal = default_parent_signal
    channel = default_channel

    if parent_signal not in data_pkl.signal_data:
        print(f"⚠️ Parent signal '{parent_signal}' not found. Skipping processing.")
        skip_step = True
    elif channel is None or channel not in data_pkl.signal_data[parent_signal].columns:
        print(f"⚠️ Channel '{channel}' not found for signal '{parent_signal}'. Skipping processing.")
        skip_step = True
    elif 'datetime' not in data_pkl.signal_data[parent_signal].columns:
        print(f"⚠️ 'datetime' column missing for signal '{parent_signal}'. Skipping processing.")
        skip_step = True

    if not skip_step:

        # Configure signals
        signal_df = data_pkl.signal_data[parent_signal]
        signal = data_pkl.signal_data[parent_signal][channel]
        datetime_signal = data_pkl.signal_data[parent_signal]['datetime']
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

        # Retrieve parameters for peak detection
        params = param_manager.get_from_config(
            variable_names=[
                "BROAD_LOW_CUTOFF", "BROAD_HIGH_CUTOFF", "NARROW_LOW_CUTOFF", "NARROW_HIGH_CUTOFF",
                "FILTER_ORDER", "SPIKE_THRESHOLD", "SMOOTH_SEC_MULTIPLIER", "WINDOW_SIZE_MULTIPLIER",
                "NORMALIZATION_NOISE", "PEAK_HEIGHT", "PEAK_DISTANCE_SEC", "SEARCH_RADIUS_SEC",
                "MIN_PEAK_HEIGHT", "MAX_PEAK_HEIGHT", "enable_bandpass", "enable_spike_removal",
                "enable_absolute", "enable_smoothing", "enable_normalization", "enable_refinement",
                "HR_JUMP_FRAC", "MIN_RR_SEC", "MAX_HR_BPM", "MIN_HR_BPM",
                "ANTI_DOUBLE_GAP_FACTOR", "HR_CONFLICT_RR_FACTOR", "PICK_LAST_IN_CONFLICT_PAIR",
                "ANTI_DOUBLE_ROLLING_WINDOW_SEC", "DETECTION_DERIVATIVE_CHANNELS"
            ],
            section="hr_peak_detection_settings" if detection_mode == "heart_rate" else "stroke_peak_detection_settings"
        )

        alt_settings = param_manager.get_from_config(
            variable_names=[
                "method",
                "sampling_rate_hz",
                "xqrs_hr_init",
                "xqrs_hr_max",
                "xqrs_hr_min",
                "xqrs_qrs_width",
                "xqrs_qrs_thr_init",
                "xqrs_qrs_thr_min",
                "xqrs_ref_period",
                "xqrs_t_inspect_period",
                "xqrs_learn",
                "xqrs_verbose",
            ],
            section="alt_peak_detect_settings",
        )
        alt_switch = param_manager.get_from_config(
            variable_names=["use_alt_peak_detect_settings"],
            section="settings",
        )
        use_alt_peak_detect = _as_bool(alt_switch.get("use_alt_peak_detect_settings"), default=False)

        overwrite=False # If needed, change to true and rewrite settings here

        default_params = {
            "BROAD_LOW_CUTOFF": 1.0,  # Hz, lower cutoff for the broad bandpass filter
            "BROAD_HIGH_CUTOFF": 35.0,  # Hz, upper cutoff for the broad bandpass filter. Per the Nyquist theorem, this should be less than half the sampling rate.
            "NARROW_LOW_CUTOFF": 5.0,  # Hz, lower cutoff for the narrow bandpass filter
            "NARROW_HIGH_CUTOFF": 20.0,  # Hz, upper cutoff for the narrow bandpass filter
            "FILTER_ORDER": 2,  # Order of the bandpass filter, affects sharpness
            "SPIKE_THRESHOLD": 400,  # Threshold for removing large spikes (e.g., noise or artifacts)
            "SMOOTH_SEC_MULTIPLIER": 0.36,  # Multiplier for calculating the smoothing window size
            "WINDOW_SIZE_MULTIPLIER": 6.35,  # Multiplier for calculating sliding window size
            "NORMALIZATION_NOISE": 1e-10,  # Small constant to avoid division by zero in normalization
            "PEAK_HEIGHT": -0.4,  # Minimum amplitude (height) for peak detection
            "PEAK_DISTANCE_SEC": 0.16,  # Minimum time between detected peaks (in seconds)
            "SEARCH_RADIUS_SEC": 0.2,  # Time range for refining the peak location (in seconds)
            "MIN_PEAK_HEIGHT": 70,  # Minimum acceptable amplitude for detected peaks
            "MAX_PEAK_HEIGHT": 12000,  # Maximum acceptable amplitude for detected peaks
            "enable_bandpass": True,  # Enable/disable bandpass filtering
            "enable_spike_removal": True,  # Enable/disable spike removal
            "enable_absolute": True,  # Enable/disable abs() transformation of signal (only use if HR, not for stroke rate)
            "enable_smoothing": True,  # Enable/disable smoothing
            "enable_normalization": True,  # Enable/disable sliding window normalization
            "enable_refinement": True,  # Enable/disable peak refinement
            "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],  # candidate detection sources
            "HR_JUMP_FRAC": 0.8,  # >80% change is suspicious
            "MIN_RR_SEC": 0.25,  # discard RR < 0.25 s (i.e. > 240 bpm)
            "MAX_HR_BPM": 240,  # 60 for turtles
            "MIN_HR_BPM": 0.1,
            "ANTI_DOUBLE_GAP_FACTOR": 0.75,  # Generalized anti-double detection factor (relative spacing rule)
            "HR_CONFLICT_RR_FACTOR": 0.75,  # Legacy alias for backward compatibility
            "PICK_LAST_IN_CONFLICT_PAIR": True,  # keep later peak in a close conflict pair by default
            "ANTI_DOUBLE_ROLLING_WINDOW_SEC": 10.0,  # local RR window for anti-double baseline
        }

        if overwrite:
            params = default_params.copy()
            # Explicit overwrite writes deployment-specific values by intent.
            param_manager.add_to_config(entries=params, section="hr_peak_detection_settings")
        else:
            params_raw = params.copy()
            # Fill only missing keys at runtime, preserving dataset-default -> deployment override precedence.
            params = {
                key: params.get(key) if params.get(key) is not None else value
                for key, value in default_params.items()
            }
            if detection_mode == "heart_rate":
                # Keep anti-double factor aliases synchronized.
                if params.get("ANTI_DOUBLE_GAP_FACTOR") is None and params.get("HR_CONFLICT_RR_FACTOR") is not None:
                    params["ANTI_DOUBLE_GAP_FACTOR"] = params["HR_CONFLICT_RR_FACTOR"]
                if params.get("HR_CONFLICT_RR_FACTOR") is None and params.get("ANTI_DOUBLE_GAP_FACTOR") is not None:
                    params["HR_CONFLICT_RR_FACTOR"] = params["ANTI_DOUBLE_GAP_FACTOR"]
                # Persist missing HR cleanup keys so they exist in parameter_log.json.
                hr_cleanup_keys = [
                    "HR_JUMP_FRAC", "MIN_RR_SEC", "MAX_HR_BPM", "MIN_HR_BPM",
                    "ANTI_DOUBLE_GAP_FACTOR", "HR_CONFLICT_RR_FACTOR", "PICK_LAST_IN_CONFLICT_PAIR",
                    "ANTI_DOUBLE_ROLLING_WINDOW_SEC",
                ]
                missing_cleanup_entries = {
                    key: params[key]
                    for key in hr_cleanup_keys
                    if params_raw.get(key) is None
                }
                if missing_cleanup_entries:
                    param_manager.add_to_config(entries=missing_cleanup_entries, section="hr_peak_detection_settings")
            print("Settings loaded from config file with runtime fallback for missing values.")

        # Use the updated parameters in peak detection
        detector_method = "legacy"
        xqrs_conf = None
        xqrs_learn = True
        xqrs_verbose = False
        if detection_mode == "heart_rate" and use_alt_peak_detect:
            alt_method = str(alt_settings.get("method") or "").strip().lower()
            if alt_method == "wfdb_xqrs":
                detector_method = "wfdb_xqrs"
                xqrs_conf = {
                    "hr_init": float(alt_settings.get("xqrs_hr_init") or 75.0),
                    "hr_max": float(alt_settings.get("xqrs_hr_max") or 200.0),
                    "hr_min": float(alt_settings.get("xqrs_hr_min") or 25.0),
                    "qrs_width": float(alt_settings.get("xqrs_qrs_width") or 0.1),
                    "qrs_thr_init": float(alt_settings.get("xqrs_qrs_thr_init") or 0.13),
                    "qrs_thr_min": float(alt_settings.get("xqrs_qrs_thr_min") or 0.0),
                    "ref_period": float(alt_settings.get("xqrs_ref_period") or 0.2),
                    "t_inspect_period": float(alt_settings.get("xqrs_t_inspect_period") or 0.0),
                }
                xqrs_learn = _as_bool(alt_settings.get("xqrs_learn"), default=True)
                xqrs_verbose = _as_bool(alt_settings.get("xqrs_verbose"), default=False)
                print("Using alternate detector from dataset defaults: wfdb_xqrs")
            else:
                print("use_alt_peak_detect_settings=True, but alt method is not wfdb_xqrs. Falling back to legacy detector.")

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
            detector_method=detector_method,
            xqrs_conf=xqrs_conf,
            xqrs_learn=xqrs_learn,
            xqrs_verbose=xqrs_verbose,
        )

        process_rate(data_pkl, results, signal_subset_df, parent_signal,
                    params, sampling_rate, detection_mode)
    
        # ============================================================
        # HEARTBEAT CLEANUP + EVENT SYNC (all together)
        # ============================================================
        # Assumptions:
        # - you already ran peak detection and rate processing, so you have:
        #     results["peak_df"]
        #     results["smoothed"]
        #     signal_subset_df  (with a 'datetime' column, same len as signal)
        #     data_pkl          (with .event_data and .event_info or we create them)
        #     fs                (sampling rate, e.g. 100 or 400)
        #     parent_signal     (e.g. "ecg")
        # - this goes AFTER you’ve run your original detect code
        # ============================================================

        # ------------------------------------------------------------
        # 0. CONFIG
        # ------------------------------------------------------------
        HR_JUMP_FRAC = params["HR_JUMP_FRAC"]  # >80% change is suspicious
        MIN_SUGGESTED_PEAK_HEIGHT = params["MIN_PEAK_HEIGHT"]  # change to your MIN_PEAK_HEIGHT
        MIN_RR_SEC = params["MIN_RR_SEC"]  # discard RR < 0.25 s (i.e. > 240 bpm)
        MAX_HR_BPM = params["MAX_HR_BPM"]  # 60 for turtles
        MIN_HR_BPM = params["MIN_HR_BPM"]
        fs = sampling_rate  # e.g. 100 or 400
        UP_JUMP_SEARCH_RADIUS_SEC = params.get("SEARCH_RADIUS_SEC", 0.2)
        try:
            UP_JUMP_SEARCH_RADIUS_SEC = float(UP_JUMP_SEARCH_RADIUS_SEC)
        except (TypeError, ValueError):
            UP_JUMP_SEARCH_RADIUS_SEC = 0.2
        UP_JUMP_SEARCH_RADIUS_SAMPLES = max(1, int(round(UP_JUMP_SEARCH_RADIUS_SEC * fs)))
        CONFLICT_GAP_FACTOR = params.get("ANTI_DOUBLE_GAP_FACTOR")
        if CONFLICT_GAP_FACTOR is None:
            CONFLICT_GAP_FACTOR = params.get("HR_CONFLICT_RR_FACTOR")
        try:
            CONFLICT_GAP_FACTOR = float(CONFLICT_GAP_FACTOR) if CONFLICT_GAP_FACTOR is not None else 0.75
        except (TypeError, ValueError):
            CONFLICT_GAP_FACTOR = 0.75
        try:
            CONFLICT_ROLLING_WINDOW_SEC = float(params.get("ANTI_DOUBLE_ROLLING_WINDOW_SEC", 10.0))
            if CONFLICT_ROLLING_WINDOW_SEC <= 0:
                CONFLICT_ROLLING_WINDOW_SEC = 10.0
        except (TypeError, ValueError):
            CONFLICT_ROLLING_WINDOW_SEC = 10.0
        CONFLICT_LOCAL_NEIGHBORS = 30
        pick_last_raw = params.get("PICK_LAST_IN_CONFLICT_PAIR", True)
        if isinstance(pick_last_raw, bool):
            PICK_LAST_IN_CONFLICT_PAIR = pick_last_raw
        elif isinstance(pick_last_raw, (int, float)):
            PICK_LAST_IN_CONFLICT_PAIR = bool(pick_last_raw)
        else:
            PICK_LAST_IN_CONFLICT_PAIR = str(pick_last_raw).strip().lower() in {"true", "1", "yes", "y", "on"}
        FILL_METHOD = "interp"      # "ffill" or "interp"

        # ============================================================
        # 1. PULL PEAKS AND BUILD BASE HR TABLE
        # ============================================================
        peak_df = results["peak_df"].copy()
        smoothed = results["smoothed"]

        # accepted peaks (before we alter for DOWN/UP jumps)
        accepted = peak_df[peak_df["key"] == "beat_auto_detect_accepted"].sort_values(
            "refined_index"
        ).reset_index(drop=True)

        # make an HR table from accepted only, to detect jumps
        if len(accepted) > 1:
            rr_sec = np.diff(accepted["refined_index"].to_numpy()) / fs
            inst_hr = 60.0 / rr_sec
            hr_df = pd.DataFrame({
                "idx": accepted["refined_index"].iloc[1:].to_numpy(),
                "datetime": accepted["datetime"].iloc[1:].to_numpy() if "datetime" in accepted else None,
                "hr_bpm": inst_hr
            })
        else:
            hr_df = pd.DataFrame(columns=["idx", "datetime", "hr_bpm"])

        # detect jumps
        if not hr_df.empty:
            hr_df["prev_hr_bpm"] = hr_df["hr_bpm"].shift(1)
            hr_df["frac_change"] = (hr_df["hr_bpm"] - hr_df["prev_hr_bpm"]) / hr_df["prev_hr_bpm"]
            big_jump_mask = hr_df["prev_hr_bpm"].notna() & (hr_df["frac_change"].abs() > HR_JUMP_FRAC)
            jumps = hr_df[big_jump_mask].copy()
            down_jumps = jumps[jumps["frac_change"] < 0].copy()
            up_jumps = jumps[jumps["frac_change"] > 0].copy()
        else:
            down_jumps = pd.DataFrame()
            up_jumps = pd.DataFrame()


        # ============================================================
        # 2. HANDLE DOWN JUMPS: try to insert a SUGGESTED beat
        # ============================================================
        suggested_rows = []

        for _, row in down_jumps.iterrows():
            this_idx = int(row["idx"])
            # find prev accepted beat index
            prev_idx = int(accepted.loc[accepted["refined_index"] < this_idx, "refined_index"].max())

            search_sig = smoothed[prev_idx:this_idx]
            if len(search_sig) < 3:
                continue

            # find local max
            local_rel_idx = np.argmax(search_sig)
            local_abs_idx = prev_idx + local_rel_idx
            local_val = smoothed[local_abs_idx]

            if local_val >= MIN_SUGGESTED_PEAK_HEIGHT:
                suggested_rows.append({
                    "refined_index": local_abs_idx,
                    "height_original": local_val,
                    "height_normalized": np.nan,
                    "datetime": signal_subset_df["datetime"].iloc[local_abs_idx]
                        if "datetime" in signal_subset_df
                        else pd.NaT,
                    "key": "beat_auto_detect_suggested"
                })

        if suggested_rows:
            suggested_df = pd.DataFrame(suggested_rows)
            peak_df = pd.concat([peak_df, suggested_df], ignore_index=True)

        # re-sort after adding suggestions
        peak_df = peak_df.sort_values("refined_index").reset_index(drop=True)

        def _apply_conflict_pair_rejection(df):
            accepted_tmp = df[df["key"].isin([
                "beat_auto_detect_accepted",
                "beat_auto_detect_suggested"
            ])].sort_values("refined_index").reset_index(drop=True)
            if len(accepted_tmp) <= 1:
                return df
            idxs = accepted_tmp["refined_index"].astype(int).to_numpy()
            rr_all = np.diff(idxs) / fs if len(idxs) > 1 else np.array([])
            if rr_all.size == 0:
                return df
            interval_midpoints = (idxs[:-1].astype(float) + idxs[1:].astype(float)) / 2.0
            half_window_samples = max(1.0, (CONFLICT_ROLLING_WINDOW_SEC * fs) / 2.0)
            for i in range(len(idxs) - 1):
                a = int(idxs[i])
                b = int(idxs[i + 1])
                rr = (b - a) / fs
                pair_mid = (a + b) / 2.0
                local_mask = np.abs(interval_midpoints - pair_mid) <= half_window_samples
                local_rr = rr_all[local_mask]
                local_rr = local_rr[np.isfinite(local_rr) & (local_rr > 0)]
                if local_rr.size == 0:
                    lo = max(0, i - CONFLICT_LOCAL_NEIGHBORS)
                    hi = min(len(rr_all), i + CONFLICT_LOCAL_NEIGHBORS + 1)
                    local_rr = rr_all[lo:hi]
                    local_rr = local_rr[np.isfinite(local_rr) & (local_rr > 0)]
                if local_rr.size == 0:
                    continue
                rr_ref = float(np.median(local_rr))
                if rr < (CONFLICT_GAP_FACTOR * rr_ref):
                    reject_idx = a if PICK_LAST_IN_CONFLICT_PAIR else b
                    df.loc[df["refined_index"] == reject_idx, "key"] = "beat_auto_detect_rejected"
            return df

        # First anti-double pass before up-jump cleanup.
        peak_df = _apply_conflict_pair_rejection(peak_df)

        # Recompute up jumps from the conflict-cleaned sequence.
        accepted_after_conflict = peak_df[peak_df["key"].isin([
            "beat_auto_detect_accepted",
            "beat_auto_detect_suggested"
        ])].sort_values("refined_index").reset_index(drop=True)
        if len(accepted_after_conflict) > 1:
            rr_sec2 = np.diff(accepted_after_conflict["refined_index"].to_numpy()) / fs
            inst_hr2 = 60.0 / rr_sec2
            hr_df2 = pd.DataFrame({
                "idx": accepted_after_conflict["refined_index"].iloc[1:].to_numpy(),
                "hr_bpm": inst_hr2
            })
            hr_df2["prev_hr_bpm"] = hr_df2["hr_bpm"].shift(1)
            hr_df2["frac_change"] = (hr_df2["hr_bpm"] - hr_df2["prev_hr_bpm"]) / hr_df2["prev_hr_bpm"]
            big_jump_mask2 = hr_df2["prev_hr_bpm"].notna() & (hr_df2["frac_change"].abs() > HR_JUMP_FRAC)
            up_jumps = hr_df2[big_jump_mask2 & (hr_df2["frac_change"] > 0)].copy()
        else:
            up_jumps = pd.DataFrame()


        # ============================================================
        # 3. HANDLE UP JUMPS: reject spurious early beat, maybe mark gap
        # ============================================================
        nan_segments = []        # explicit gaps we want to be NaN
        auto_nan_intervals = []  # gaps the rebuild later discovers

        for _, row in up_jumps.iterrows():
            this_idx = int(row["idx"])
            this_is_active = ((peak_df["refined_index"] == this_idx) &
                              (peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"]))).any()
            if not this_is_active:
                continue
            upjump_prev_candidates = peak_df[
                (peak_df["refined_index"] < this_idx) &
                (peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"]))
            ]["refined_index"]
            prev_upjump_idx = int(upjump_prev_candidates.max()) if not upjump_prev_candidates.empty else None
            # For up-jumps, enforce pair resolution first: keep later peak by default.
            if prev_upjump_idx is not None:
                reject_idx = prev_upjump_idx if PICK_LAST_IN_CONFLICT_PAIR else this_idx
            else:
            # 1) within the configured search window, find local max in hr_smoothed.
            #    Then reject the nearest accepted/suggested beat to that local max.
                lo = max(0, this_idx - UP_JUMP_SEARCH_RADIUS_SAMPLES)
                hi = min(len(smoothed) - 1, this_idx + UP_JUMP_SEARCH_RADIUS_SAMPLES)
                local_search_sig = smoothed[lo:hi + 1]
                if len(local_search_sig) == 0:
                    reject_idx = this_idx
                else:
                    local_max_idx = lo + int(np.argmax(local_search_sig))
                    candidates = peak_df[
                        (peak_df["refined_index"] >= lo) &
                        (peak_df["refined_index"] <= hi) &
                        (peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"]))
                    ]["refined_index"].astype(int).to_numpy()
                    if len(candidates) == 0:
                        reject_idx = this_idx
                    else:
                        reject_idx = int(candidates[np.argmin(np.abs(candidates - local_max_idx))])

            peak_df.loc[peak_df["refined_index"] == reject_idx, "key"] = "beat_auto_detect_rejected"

            # 2) get next accepted/suggested beat AFTER this_idx
            later = peak_df[
                (peak_df["refined_index"] > reject_idx) &
                (peak_df["key"].str.contains("accepted|suggested"))
            ].sort_values("refined_index")

            if later.empty:
                continue

            next_idx = int(later.iloc[0]["refined_index"])

            prev_candidates = peak_df[
                (peak_df["refined_index"] < reject_idx) &
                (peak_df["key"].str.contains("accepted|suggested"))
            ]["refined_index"]
            if prev_candidates.empty:
                continue
            prev_idx = int(prev_candidates.max())

            # 3) check if the interval prev_idx -> next_idx is too short
            rr_fixed_sec = (next_idx - prev_idx) / fs
            if rr_fixed_sec < MIN_RR_SEC:
                # force this to be a NaN gap later
                nan_segments.append((prev_idx, next_idx))
                continue
            # else we let the rebuild compute it later

        # Second anti-double pass after up-jump cleanup.
        peak_df = _apply_conflict_pair_rejection(peak_df)

        # stash the updated peak_df
        results["peak_df"] = peak_df


        # ============================================================
        # 4. REBUILD HR (strict) -> hr_series with NaNs
        # ============================================================
        # use accepted + suggested ONLY
        peak_for_hr = peak_df[peak_df["key"].isin([
            "beat_auto_detect_accepted",
            "beat_auto_detect_suggested"
        ])].sort_values("refined_index").reset_index(drop=True)

        n = len(signal_subset_df)
        hr_series = np.full(n, np.nan, dtype=float)

        if len(peak_for_hr) > 1:
            for i in range(len(peak_for_hr) - 1):
                s = int(peak_for_hr["refined_index"].iloc[i])
                e = int(peak_for_hr["refined_index"].iloc[i+1])

                if e <= s:
                    hr_series[s:e] = np.nan
                    auto_nan_intervals.append((s, e, "non_increasing"))
                    continue

                rr_sec = (e - s) / fs

                # too short -> NaN + remember
                if rr_sec < MIN_RR_SEC:
                    hr_series[s:e] = np.nan
                    auto_nan_intervals.append((s, e, f"rr_too_short={rr_sec:.3f}"))
                    continue

                hr_val = 60.0 / rr_sec
                # clamp
                hr_val = max(min(hr_val, MAX_HR_BPM), MIN_HR_BPM)
                hr_series[s:e] = hr_val

        # apply explicit NaN segments from UP-jump phase
        for (s, e) in nan_segments:
            hr_series[s:e] = np.nan


        # ============================================================
        # 5. BUILD DERIVED DATAFRAMES (NaN + fixed)
        # ============================================================
        heart_rate_nan = pd.DataFrame({
            "datetime": signal_subset_df["datetime"],
            "heart_rate_nan": hr_series
        })

        # fill version
        s_hr = pd.Series(hr_series)
        if FILL_METHOD == "ffill":
            heart_rate_fixed_vals = s_hr.ffill().to_numpy()
        else:  # "interp"
            heart_rate_fixed_vals = s_hr.interpolate(limit_direction="both").to_numpy()

        # clamp again after fill
        heart_rate_fixed_vals = np.clip(heart_rate_fixed_vals, MIN_HR_BPM, MAX_HR_BPM)

        heart_rate_fixed = pd.DataFrame({
            "datetime": signal_subset_df["datetime"],
            "heart_rate_fixed": heart_rate_fixed_vals
        })

        # save into data_pkl signal_data / signal_info
        data_pkl.signal_data["heart_rate_nan"] = heart_rate_nan
        data_pkl.signal_info["heart_rate_nan"] = {
            "channels": ["heart_rate_nan"],
            "metadata": {"heart_rate_nan": {"unit": "bpm", "signal": parent_signal}},
            "derived_from_signals": [parent_signal],
            "transformation_log": [
                "recomputed HR after beat cleanup",
                "inserted NaN for suspect intervals (>50% jump or too-short RR)"
            ],
        }

        data_pkl.signal_data["heart_rate_fixed"] = heart_rate_fixed
        data_pkl.signal_info["heart_rate_fixed"] = {
            "channels": ["heart_rate_fixed"],
            "metadata": {"heart_rate_fixed": {"unit": "bpm", "signal": parent_signal}},
            "derived_from_signals": [parent_signal],
            "transformation_log": [
                "recomputed HR after beat cleanup",
                f"NaN gaps filled using {FILL_METHOD}"
            ],
        }


        # ============================================================
        # 6. EVENTS: push accepted/rejected/suggested + gaps via event manager
        # ============================================================

        # --- 6a. ensure event_data exists ---
        if not hasattr(data_pkl, "event_data") or data_pkl.event_data is None:
            data_pkl.event_data = pd.DataFrame(columns=[
                "datetime", "key", "short_description", "type", "duration"
            ])

        # --- 6b. accepted/rejected/suggested (point events) ---
        point_events = []

        KEY_MAP = {
            "beat_auto_detect_accepted":  ("heartbeat_auto_detect_accepted",  "auto-detected heartbeat (accepted)"),
            "beat_auto_detect_rejected":  ("heartbeat_auto_detect_rejected",  "auto-detected heartbeat (rejected as spurious / atrial)"),
            "beat_auto_detect_suggested": ("heartbeat_auto_detect_suggested", "auto-detected heartbeat (suggested, missed-beat fix)"),
        }

        for _, row in results["peak_df"].iterrows():
            k = row.get("key", "")
            if k not in KEY_MAP:
                continue
            dt = row.get("datetime", pd.NaT)
            new_key, desc = KEY_MAP[k]
            point_events.append({
                "datetime": dt,
                "key": new_key,
                "short_description": desc,
                "type": "point",
                "duration": 0.0,
            })

        point_df = pd.DataFrame(point_events)


        # --- 6c. gaps (interval events with duration) ---
        gap_key = "heartbeat_auto_detect_gap"
        gap_desc = "interval where HR was invalid (>50% jump / too-short RR)"
        gap_events = []

        def _add_gap_event(s, e, reason=None):
            dt_start = signal_subset_df["datetime"].iloc[s]
            dt_end   = signal_subset_df["datetime"].iloc[min(e - 1, len(signal_subset_df) - 1)]
            duration_sec = (dt_end - dt_start).total_seconds()
            desc = f"{gap_desc} ({reason})" if reason else gap_desc
            gap_events.append({
                "datetime": dt_start,
                "key": gap_key,
                "short_description": desc,
                "type": "interval_start",
                "duration": duration_sec,
            })
            gap_events.append({
                "datetime": dt_end,
                "key": gap_key,
                "short_description": f"{desc}: end",
                "type": "interval_end",
                "duration": duration_sec,
            })

        # from explicit and auto-detected gaps
        for (s, e) in nan_segments:
            _add_gap_event(s, e, reason="nan_segment")
        for (s, e, reason) in auto_nan_intervals:
            _add_gap_event(s, e, reason=reason)

        gap_df = pd.DataFrame(gap_events)

        # --- 6d. merge and deduplicate using EventManager-style call ---
        # Combine point and gap events first
        combined_events = pd.concat([point_df, gap_df], ignore_index=True) if not gap_df.empty else point_df

        # Use create_state_event to safely append into existing event_data
        if not combined_events.empty:
            for k in combined_events["key"].unique():
                subset = combined_events[combined_events["key"] == k]
                data_pkl.event_data = create_state_event(
                    state_df=subset.rename(columns={"datetime": "start_time"}),
                    key=k,
                    start_time_column="start_time",
                    duration_column="duration",
                    description=subset["short_description"].iloc[0],
                    existing_events=data_pkl.event_data
                )

        # --- 6e. register keys via EventManager ---
        if not hasattr(data_pkl, "event_manager"):
            data_pkl.event_manager = {}

        # ensure a sub-dict for HR
        data_pkl.event_manager["heart_rate"] = {
            "keys": [
                "heartbeat_auto_detect_accepted",
                "heartbeat_auto_detect_rejected",
                "heartbeat_auto_detect_suggested",
                "heartbeat_auto_detect_gap",
            ],
            "description": "Events related to heartbeat detection and HR gap handling",
            "color_map": {
                "heartbeat_auto_detect_accepted": "#4caf50",
                "heartbeat_auto_detect_rejected": "#f44336",
                "heartbeat_auto_detect_suggested": "#ffeb3b",
                "heartbeat_auto_detect_gap": "#9e9e9e",
            },
        }

        print("✅ Heartbeat events synced via EventManager.")



        # ============================================================
        # done 🎉
        # you now have:
        # - results["peak_df"] with accepted/rejected/suggested
        # - data_pkl.signal_data["heart_rate_nan"]  (NaN where bad)
        # - data_pkl.signal_data["heart_rate_fixed"] (filled)
        # - data_pkl.event_data with heartbeat_* and heartbeat_*_gap (with duration)
        # ============================================================

        TARGET_SAMPLING_RATE = 25

        notes_to_plot = {
            'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
            'heartbeat_auto_detect_accepted': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
            'heartbeat_auto_detect_rejected': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'red'},
            'strokebeat_auto_detect_accepted': {'signal': 'sr_smoothed', 'symbol': 'triangle-up', 'color': 'green'},
        }

        # fig = plot_tag_data_interactive(
        #     data_pkl=data_pkl,
        #     signals=['hr_broad_bandpass', 'hr_smoothed', 'hr_normalized', 'ecg', 'depth', 'prh', 'stroke_rate', 'heart_rate','sr_smoothed'],
        #     channels={}, #'corrected_gyr': ['broad_bandpassed_signal']
        #     time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
        #     note_annotations=notes_to_plot,
        #     color_mapping_path=color_mapping_path,
        #     target_sampling_rate=TARGET_SAMPLING_RATE,
        #     zoom_start_time=ZOOM_START_TIME,
        #     zoom_end_time=ZOOM_END_TIME,
        #     zoom_range_selector_channel='depth',
        #     plot_event_values=[],
        # )

        # fig.show()

        # Clean-up events

        # Clear the specified keys
        keys_to_remove = ['hr_broad_bandpass','hr_narrow_bandpass', 'hr_smoothed'] # KEEPING 'hr_normalized' because it is clearest
        clear_intermediate_signals(data_pkl, remove_keys=keys_to_remove)

        initial_event_count = len(data_pkl.event_data)
        # Remove events with keys ending in '_rejected'
        data_pkl.event_data = data_pkl.event_data[~data_pkl.event_data['key'].str.endswith('_rejected', na=False)]
        # Get the final count of events
        final_event_count = len(data_pkl.event_data)
        # Print the number of removed events
        removed_event_count = initial_event_count - final_event_count
        print(f"Removed {removed_event_count} events with keys ending in '_rejected'.")
else:
    print(f"Skipping step due to missing critical signal: {critical_signal}.")

current_processing_step = "Processing Step 05. Heart rate calculation complete."
print(current_processing_step)

# Add or update the current_processing_step for the specified deployment
param_manager.add_to_config("current_processing_step", current_processing_step)

# Optional: save new pickle file
if data_changed:
    with open(pkl_path, 'wb') as file:
            pickle.dump(data_pkl, file)
    print("Pickle file updated.")

save_step_netcdf_if_changed(data_pkl, deployment_folder, deployment_id, 5, changed=data_changed)
