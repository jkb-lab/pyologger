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
from pyologger.utils.chunk_manager import attachment_time_mask

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

        # ── Resolve deployment-level params and alt-detector config ──────
        # These are loaded once; per-chunk overrides are applied as sparse
        # patches inside the loop below.
        base_params = param_manager.get_from_config(
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
                "method", "sampling_rate_hz",
                "xqrs_hr_init", "xqrs_hr_max", "xqrs_hr_min",
                "xqrs_qrs_width", "xqrs_qrs_thr_init", "xqrs_qrs_thr_min",
                "xqrs_ref_period", "xqrs_t_inspect_period",
                "xqrs_learn", "xqrs_verbose",
            ],
            section="alt_peak_detect_settings",
        )
        alt_switch = param_manager.get_from_config(
            variable_names=["use_alt_peak_detect_settings"],
            section="settings",
        )
        use_alt_peak_detect = _as_bool(alt_switch.get("use_alt_peak_detect_settings"), default=False)

        overwrite = False  # If needed, change to true and rewrite settings here

        default_params = {
            "BROAD_LOW_CUTOFF": 1.0,
            "BROAD_HIGH_CUTOFF": 35.0,
            "NARROW_LOW_CUTOFF": 5.0,
            "NARROW_HIGH_CUTOFF": 20.0,
            "FILTER_ORDER": 2,
            "SPIKE_THRESHOLD": 400,
            "SMOOTH_SEC_MULTIPLIER": 0.36,
            "WINDOW_SIZE_MULTIPLIER": 6.35,
            "NORMALIZATION_NOISE": 1e-10,
            "PEAK_HEIGHT": -0.4,
            "PEAK_DISTANCE_SEC": 0.16,
            "SEARCH_RADIUS_SEC": 0.2,
            "MIN_PEAK_HEIGHT": 70,
            "MAX_PEAK_HEIGHT": 12000,
            "enable_bandpass": True,
            "enable_spike_removal": True,
            "enable_absolute": True,
            "enable_smoothing": True,
            "enable_normalization": True,
            "enable_refinement": True,
            "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],
            "HR_JUMP_FRAC": 0.8,
            "MIN_RR_SEC": 0.25,
            "MAX_HR_BPM": 240,
            "MIN_HR_BPM": 0.1,
            "ANTI_DOUBLE_GAP_FACTOR": 0.75,
            "HR_CONFLICT_RR_FACTOR": 0.75,
            "PICK_LAST_IN_CONFLICT_PAIR": True,
            "ANTI_DOUBLE_ROLLING_WINDOW_SEC": 10.0,
        }

        if overwrite:
            base_params = default_params.copy()
            param_manager.add_to_config(entries=base_params, section="hr_peak_detection_settings")
        else:
            base_params_raw = base_params.copy()
            base_params = {
                key: base_params.get(key) if base_params.get(key) is not None else value
                for key, value in default_params.items()
            }
            if detection_mode == "heart_rate":
                if base_params.get("ANTI_DOUBLE_GAP_FACTOR") is None and base_params.get("HR_CONFLICT_RR_FACTOR") is not None:
                    base_params["ANTI_DOUBLE_GAP_FACTOR"] = base_params["HR_CONFLICT_RR_FACTOR"]
                if base_params.get("HR_CONFLICT_RR_FACTOR") is None and base_params.get("ANTI_DOUBLE_GAP_FACTOR") is not None:
                    base_params["HR_CONFLICT_RR_FACTOR"] = base_params["ANTI_DOUBLE_GAP_FACTOR"]
                hr_cleanup_keys = [
                    "HR_JUMP_FRAC", "MIN_RR_SEC", "MAX_HR_BPM", "MIN_HR_BPM",
                    "ANTI_DOUBLE_GAP_FACTOR", "HR_CONFLICT_RR_FACTOR", "PICK_LAST_IN_CONFLICT_PAIR",
                    "ANTI_DOUBLE_ROLLING_WINDOW_SEC",
                ]
                missing_cleanup_entries = {
                    key: base_params[key]
                    for key in hr_cleanup_keys
                    if base_params_raw.get(key) is None
                }
                if missing_cleanup_entries:
                    param_manager.add_to_config(entries=missing_cleanup_entries, section="hr_peak_detection_settings")
            print("Settings loaded from config file with runtime fallback for missing values.")

        # ── Build alt-detector config once (shared across chunks) ────────
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

        # ── Chunk-aware time window resolution ──────────────────────────
        chunk_grid = param_manager.get_or_create_chunk_grid()
        if chunk_grid:
            run_chunks = [
                (i, c) for i, c in enumerate(chunk_grid)
                if c.get("status") not in ("gap",)
            ]
            print(f"Chunk mode: {len(run_chunks)} processable chunks out of {len(chunk_grid)} total.")
        else:
            signal_start = datetime_signal.min()
            signal_end = datetime_signal.max()
            if overwrite:
                print(f"Signal time range: {signal_start} to {signal_end}")
                start_time_input = input(f"Enter start time (default: {signal_start}): ").strip()
                end_time_input = input(f"Enter end time (default: {signal_end}): ").strip()
                signal_start = pd.Timestamp(start_time_input) if start_time_input else signal_start
                signal_end = pd.Timestamp(end_time_input) if end_time_input else signal_end
            run_chunks = [(None, {"start": str(signal_start), "end": str(signal_end),
                                  "status": "pending", "params_override": None})]
            print("No chunk grid — running single-pass over full signal.")

        all_peak_rows = []
        all_signal_subset_dfs = []
        all_smoothed_parts = []

        # Precompute once for chunk lookup. A monotonic datetime column lets each
        # chunk be sliced by binary search rather than a full-length boolean mask.
        _datetime_is_sorted = bool(datetime_signal.is_monotonic_increasing)
        if _datetime_is_sorted:
            _dt_utc = datetime_signal
            if datetime_signal.dt.tz is not None:
                _dt_utc = datetime_signal.dt.tz_convert("UTC").dt.tz_localize(None)
            _datetime_values = _dt_utc.to_numpy(dtype="datetime64[ns]")
        else:
            _datetime_values = None
            print("⚠️ datetime column is not monotonic; using boolean masks per chunk.")

        # ── Per-chunk detection loop ─────────────────────────────────────
        for _chunk_idx, _chunk in run_chunks:
            _chunk_start = pd.Timestamp(_chunk["start"])
            _chunk_end = pd.Timestamp(_chunk["end"])

            if datetime_signal.dt.tz is not None:
                if _chunk_start.tzinfo is None:
                    _chunk_start = _chunk_start.tz_localize(str(datetime_signal.dt.tz))
                else:
                    _chunk_start = _chunk_start.tz_convert(str(datetime_signal.dt.tz))
                if _chunk_end.tzinfo is None:
                    _chunk_end = _chunk_end.tz_localize(str(datetime_signal.dt.tz))
                else:
                    _chunk_end = _chunk_end.tz_convert(str(datetime_signal.dt.tz))

            # The datetime column is monotonic, so locate the chunk by binary search
            # instead of building a full-length boolean mask per chunk (which is O(n)
            # over the whole ECG series for each of ~100+ chunks).
            if _datetime_is_sorted:
                _lo_key = _chunk_start
                _hi_key = _chunk_end
                if _lo_key.tzinfo is not None:
                    _lo_key = _lo_key.tz_convert("UTC").tz_localize(None)
                    _hi_key = _hi_key.tz_convert("UTC").tz_localize(None)
                # side='left'/'right' makes the span inclusive on both ends, matching
                # the original (datetime >= start) & (datetime <= end) mask.
                _i0 = np.searchsorted(_datetime_values, np.datetime64(_lo_key), side="left")
                _i1 = np.searchsorted(_datetime_values, np.datetime64(_hi_key), side="right")
                signal_subset = signal.iloc[_i0:_i1]
                datetime_subset = datetime_signal.iloc[_i0:_i1]
                signal_subset_df = signal_df.iloc[_i0:_i1]
            else:
                time_mask = (datetime_signal >= _chunk_start) & (datetime_signal <= _chunk_end)
                signal_subset = signal[time_mask]
                datetime_subset = datetime_signal[time_mask]
                signal_subset_df = signal_df[time_mask]

            if signal_subset.empty:
                print(f"  Chunk {_chunk_idx}: no signal samples in [{_chunk_start}, {_chunk_end}] — skipping.")
                continue

            print(f"  Chunk {_chunk_idx}: [{_chunk_start}] → [{_chunk_end}] | {len(signal_subset)} samples")

            # Apply sparse per-chunk param overrides on top of deployment defaults.
            params = dict(base_params)
            _chunk_overrides = _chunk.get("params_override") or {}
            if _chunk_overrides:
                params.update({k: v for k, v in _chunk_overrides.items() if v is not None})

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

            all_peak_rows.append(results["peak_df"].copy())
            all_signal_subset_dfs.append(signal_subset_df)
            # Accumulate and concatenate once after the loop; growing the array per
            # chunk re-copies everything already collected (quadratic).
            all_smoothed_parts.append(results.get("smoothed", np.array([])))

        # end of per-chunk loop ─────────────────────────────────────────

        if not all_peak_rows:
            print("⚠️ No chunks produced any peaks. Skipping HR cleanup.")
            skip_step = True
        else:
            peak_df_merged = pd.concat(all_peak_rows, ignore_index=True).sort_values("refined_index").reset_index(drop=True)
            signal_subset_df = pd.concat(all_signal_subset_dfs, ignore_index=True)
            smoothed = (
                np.concatenate(all_smoothed_parts) if all_smoothed_parts else np.array([])
            )
            results = {"peak_df": peak_df_merged, "smoothed": smoothed}
            params = base_params  # use deployment-level params for cleanup config

    if not skip_step:
        # ============================================================
        # HEARTBEAT CLEANUP + EVENT SYNC (all together)
        # ============================================================

        # ------------------------------------------------------------
        # 0. CONFIG
        # ------------------------------------------------------------
        HR_JUMP_FRAC = params["HR_JUMP_FRAC"]
        MIN_SUGGESTED_PEAK_HEIGHT = params["MIN_PEAK_HEIGHT"]
        MIN_RR_SEC = params["MIN_RR_SEC"]
        MAX_HR_BPM = params["MAX_HR_BPM"]
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
            # interval_midpoints is sorted (idxs is), so the rolling window for every
            # pair can be located with two vectorized binary searches instead of
            # building a full-length boolean mask per iteration.
            win_lo = np.searchsorted(
                interval_midpoints, interval_midpoints - half_window_samples, side="left"
            )
            win_hi = np.searchsorted(
                interval_midpoints, interval_midpoints + half_window_samples, side="right"
            )
            reject_indices = []
            for i in range(len(idxs) - 1):
                a = int(idxs[i])
                b = int(idxs[i + 1])
                rr = (b - a) / fs
                local_rr = rr_all[win_lo[i]:win_hi[i]]
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
                    reject_indices.append(a if PICK_LAST_IN_CONFLICT_PAIR else b)
            if reject_indices:
                # One vectorized membership test rather than a full frame scan per reject.
                df.loc[
                    df["refined_index"].isin(reject_indices),
                    "key",
                ] = "beat_auto_detect_rejected_conflict"
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

        # This loop mutates peak_df as it goes (rejecting beats), and later iterations
        # depend on earlier rejections, so it cannot be vectorized wholesale. Instead
        # keep the lookups it needs as sorted numpy arrays and refresh them only when a
        # rejection actually changes the active set — the original re-scanned the whole
        # frame several times per row (~190 ms/row at 334k beats).
        _AUTO_ACTIVE = ("beat_auto_detect_accepted", "beat_auto_detect_suggested")
        _peak_idx_all = peak_df["refined_index"].astype(int).to_numpy()

        def _active_sorted(strict):
            """Sorted refined_index of active beats.

            strict=True mirrors .isin(['..._accepted', '..._suggested']);
            strict=False mirrors .str.contains('accepted|suggested'), which also
            matches manually-added keys such as 'beat_manual_accepted'.
            """
            keys = peak_df["key"].astype(str)
            mask = keys.isin(_AUTO_ACTIVE) if strict else keys.str.contains("accepted|suggested", regex=True)
            vals = peak_df.loc[mask, "refined_index"].astype(int).to_numpy()
            vals.sort()
            return vals

        _strict_active = _active_sorted(True)
        _loose_active = _active_sorted(False)

        for _, row in up_jumps.iterrows():
            this_idx = int(row["idx"])
            _pos = np.searchsorted(_strict_active, this_idx)
            this_is_active = _pos < _strict_active.size and _strict_active[_pos] == this_idx
            if not this_is_active:
                continue
            _prev_pos = np.searchsorted(_strict_active, this_idx, side="left") - 1
            prev_upjump_idx = int(_strict_active[_prev_pos]) if _prev_pos >= 0 else None
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
                    _c0 = np.searchsorted(_strict_active, lo, side="left")
                    _c1 = np.searchsorted(_strict_active, hi, side="right")
                    candidates = _strict_active[_c0:_c1]
                    if len(candidates) == 0:
                        reject_idx = this_idx
                    else:
                        reject_idx = int(candidates[np.argmin(np.abs(candidates - local_max_idx))])

            peak_df.loc[peak_df["refined_index"] == reject_idx, "key"] = "beat_auto_detect_rejected"
            # reject_idx just left the active set; drop it from both cached views.
            _rp = np.searchsorted(_strict_active, reject_idx)
            if _rp < _strict_active.size and _strict_active[_rp] == reject_idx:
                _strict_active = np.delete(_strict_active, _rp)
            _rl = np.searchsorted(_loose_active, reject_idx)
            if _rl < _loose_active.size and _loose_active[_rl] == reject_idx:
                _loose_active = np.delete(_loose_active, _rl)

            # 2) get next accepted/suggested beat AFTER this_idx
            _np_pos = np.searchsorted(_loose_active, reject_idx, side="right")
            if _np_pos >= _loose_active.size:
                continue

            next_idx = int(_loose_active[_np_pos])

            _pp_pos = np.searchsorted(_loose_active, reject_idx, side="left") - 1
            if _pp_pos < 0:
                continue
            prev_idx = int(_loose_active[_pp_pos])

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
        # 3b. CONSECUTIVE-REJECT GAPS → QC state events
        # When 2+ rejected beats appear in a row with no accepted/suggested
        # beat between them, the detector was uncertain — mark that span as
        # QC_unusable_ecg and NaN out HR.  Everything outside those spans
        # is emitted as QC_usable_ecg.
        # ============================================================
        MIN_CONSECUTIVE_REJECTS_FOR_GAP = 2

        # Any rejection reason → gap threshold of 1.
        # Conflict-pair rejections (double-beat artefact) → threshold of 2
        # because one conflict rejection between two valid beats is expected.
        REJECT_KEYS         = {"beat_auto_detect_rejected", "beat_auto_detect_rejected_conflict"}
        CONFLICT_REJECT_KEY = "beat_auto_detect_rejected_conflict"

        all_beats = peak_df[peak_df["key"].isin(
            {"beat_auto_detect_accepted", "beat_auto_detect_suggested"} | REJECT_KEYS
        )].sort_values("refined_index").reset_index(drop=True)

        consecutive_reject_count    = 0
        consecutive_conflict_count  = 0
        reject_run_start_idx        = None
        unusable_sample_spans       = []

        def _close_reject_run(n_reject, n_conflict, run_start_idx, gap_end_sample):
            # Threshold: 1 for any plain rejection, 2 if the run is ALL conflict-pair rejections
            all_conflict = (n_reject == n_conflict)
            threshold = MIN_CONSECUTIVE_REJECTS_FOR_GAP if all_conflict else 1
            if n_reject >= threshold:
                prev_good = peak_df[
                    (peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"])) &
                    (peak_df["refined_index"] < run_start_idx)
                ]["refined_index"]
                gap_start = int(prev_good.max()) if not prev_good.empty else run_start_idx
                if gap_end_sample > gap_start:
                    nan_segments.append((gap_start, gap_end_sample))
                    unusable_sample_spans.append((gap_start, gap_end_sample))
                    print(f"  Reject gap: samples {gap_start}–{gap_end_sample} "
                          f"({n_reject} rejected, {n_conflict} conflict)")

        for _, beat_row in all_beats.iterrows():
            if beat_row["key"] in REJECT_KEYS:
                if consecutive_reject_count == 0:
                    reject_run_start_idx = int(beat_row["refined_index"])
                consecutive_reject_count += 1
                if beat_row["key"] == CONFLICT_REJECT_KEY:
                    consecutive_conflict_count += 1
            else:
                if consecutive_reject_count > 0:
                    _close_reject_run(
                        consecutive_reject_count, consecutive_conflict_count,
                        reject_run_start_idx, int(beat_row["refined_index"])
                    )
                    consecutive_reject_count   = 0
                    consecutive_conflict_count = 0
                    reject_run_start_idx       = None

        # handle a reject run that reaches the end of the signal
        if consecutive_reject_count > 0:
            _close_reject_run(
                consecutive_reject_count, consecutive_conflict_count,
                reject_run_start_idx, len(signal_subset) - 1
            )


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

        # NaN out QC_unusable spans so heart_rate_fixed respects ECG quality flags
        for s, e in unusable_sample_spans:
            heart_rate_fixed_vals[s:e] = np.nan

        # --- manually_derived_hr: HR from manual-OK beats only (if present) ---
        manually_derived_hr_vals = np.full(n, np.nan, dtype=float)
        _event_data = getattr(data_pkl, "event_data", None)
        if _event_data is not None and not _event_data.empty:
            _manual_ok = _event_data[_event_data["key"] == "heartbeat_manual_ok"].copy()
            _manual_ok = _manual_ok.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            if len(_manual_ok) > 1:
                _sig_dts = signal_subset_df["datetime"].reset_index(drop=True)
                _sig_ns  = _sig_dts.values.astype("int64")
                _man_ns  = pd.to_datetime(_manual_ok["datetime"])
                if _sig_dts.dt.tz is not None:
                    tz = str(_sig_dts.dt.tz)
                    _man_ns = _man_ns.dt.tz_convert(tz) if _man_ns.dt.tz is not None else _man_ns.dt.tz_localize(tz)
                _man_ns = _man_ns.values.astype("int64")
                _snapped_idxs = []
                for ts_ns in _man_ns:
                    diff = np.abs(_sig_ns - ts_ns)
                    nearest = int(np.argmin(diff))
                    if diff[nearest] / 1e6 <= 500:  # within 500 ms
                        _snapped_idxs.append(nearest)
                _snapped_idxs = sorted(set(_snapped_idxs))
                for _i in range(len(_snapped_idxs) - 1):
                    s = _snapped_idxs[_i]
                    e = _snapped_idxs[_i + 1]
                    if e > s:
                        rr_sec = (e - s) / fs
                        if MIN_RR_SEC <= rr_sec <= (60.0 / MIN_HR_BPM + 1.0):
                            hr_val = np.clip(60.0 / rr_sec, MIN_HR_BPM, MAX_HR_BPM)
                            manually_derived_hr_vals[s:e] = hr_val
                print(f"  manually_derived_hr: {np.sum(~np.isnan(manually_derived_hr_vals))} samples from "
                      f"{len(_snapped_idxs)} snapped manual beats")

        # heart_rate_nan (NaN-broken) lives as a channel inside heart_rate_fixed,
        # not as a separate signal, so all HR channels share one subplot.
        heart_rate_fixed = pd.DataFrame({
            "datetime":            signal_subset_df["datetime"],
            "heart_rate_nan":      hr_series,                # NaN-gapped (top channel)
            "heart_rate_fixed":    heart_rate_fixed_vals,    # interpolated (middle)
            "manually_derived_hr": manually_derived_hr_vals, # manual beats (bottom, thick)
        })

        # Remove heart_rate_nan as a standalone signal if it exists from a prior run.
        data_pkl.signal_data.pop("heart_rate_nan", None)
        data_pkl.signal_info.pop("heart_rate_nan", None)

        data_pkl.signal_data["heart_rate_fixed"] = heart_rate_fixed
        data_pkl.signal_info["heart_rate_fixed"] = {
            "channels": ["heart_rate_nan", "heart_rate_fixed", "manually_derived_hr"],
            "metadata": {
                "heart_rate_nan":      {"unit": "bpm", "signal": parent_signal},
                "heart_rate_fixed":    {"unit": "bpm", "signal": parent_signal},
                "manually_derived_hr": {"unit": "bpm", "signal": parent_signal, "line_width": 3},
            },
            "derived_from_signals": [parent_signal],
            "transformation_log": [
                "recomputed HR after beat cleanup",
                f"NaN gaps filled using {FILL_METHOD}",
                "manually_derived_hr added from heartbeat_manual_ok events"
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
            "beat_auto_detect_accepted":          ("heartbeat_auto_detect_accepted",          "auto-detected heartbeat (accepted)"),
            "beat_auto_detect_rejected":          ("heartbeat_auto_detect_rejected",          "auto-detected heartbeat (rejected as spurious / atrial)"),
            "beat_auto_detect_rejected_conflict": ("heartbeat_auto_detect_rejected_conflict", "auto-detected heartbeat (rejected: conflict pair / double-beat)"),
            "beat_auto_detect_suggested":         ("heartbeat_auto_detect_suggested",         "auto-detected heartbeat (suggested, missed-beat fix)"),
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


        # --- 6c. QC state events: unusable spans + usable spans between them ---
        # All NaN sources (RR too short, up-jump, consecutive-reject runs) contribute
        # to unusable spans.  Usable spans fill the gaps between them.
        def _sample_to_dt(idx):
            idx = max(0, min(idx, len(signal_subset_df) - 1))
            return signal_subset_df["datetime"].iloc[idx]

        # Collect all unusable sample spans: explicit nan_segments + auto_nan_intervals
        all_unusable_spans = list(unusable_sample_spans)
        for (s, e) in nan_segments:
            if (s, e) not in all_unusable_spans:
                all_unusable_spans.append((s, e))
        for (s, e, _reason) in auto_nan_intervals:
            all_unusable_spans.append((s, e))

        # Merge overlapping unusable spans
        all_unusable_spans.sort(key=lambda x: x[0])
        merged_unusable = []
        for s, e in all_unusable_spans:
            if merged_unusable and s <= merged_unusable[-1][1]:
                merged_unusable[-1] = (merged_unusable[-1][0], max(merged_unusable[-1][1], e))
            else:
                merged_unusable.append((s, e))

        # Build QC_unusable_ecg state rows
        unusable_rows = []
        for s, e in merged_unusable:
            dt_start = _sample_to_dt(s)
            dt_end   = _sample_to_dt(e)
            duration_sec = (dt_end - dt_start).total_seconds()
            if duration_sec > 0:
                unusable_rows.append({
                    "datetime":          dt_start,
                    "key":               "QC_unusable_ecg",
                    "short_description": "ECG quality unusable: consecutive rejected beats or invalid RR interval",
                    "type":              "state",
                    "duration":          duration_sec,
                })

        # Build QC_usable_ecg state rows (spans between unusable regions)
        sig_start = _sample_to_dt(0)
        sig_end   = _sample_to_dt(len(signal_subset_df) - 1)
        usable_rows = []
        cursor = sig_start
        for s, e in merged_unusable:
            gap_start_dt = _sample_to_dt(s)
            gap_end_dt   = _sample_to_dt(e)
            if gap_start_dt > cursor:
                duration_sec = (gap_start_dt - cursor).total_seconds()
                if duration_sec > 0:
                    usable_rows.append({
                        "datetime":          cursor,
                        "key":               "QC_usable_ecg",
                        "short_description": "ECG quality usable",
                        "type":              "state",
                        "duration":          duration_sec,
                    })
            cursor = gap_end_dt
        if cursor < sig_end:
            duration_sec = (sig_end - cursor).total_seconds()
            if duration_sec > 0:
                usable_rows.append({
                    "datetime":          cursor,
                    "key":               "QC_usable_ecg",
                    "short_description": "ECG quality usable",
                    "type":              "state",
                    "duration":          duration_sec,
                })

        qc_df = pd.DataFrame(unusable_rows + usable_rows)

        # --- 6d. merge all events and write into event_data ---
        combined_events = pd.concat(
            [df for df in [point_df, qc_df] if not df.empty],
            ignore_index=True,
        )

        if not combined_events.empty:
            for k in combined_events["key"].unique():
                subset = combined_events[combined_events["key"] == k]
                data_pkl.event_data = create_state_event(
                    state_df=subset.rename(columns={"datetime": "start_time"}),
                    key=k,
                    start_time_column="start_time",
                    duration_column="duration",
                    description=subset["short_description"].iloc[0],
                    existing_events=data_pkl.event_data,
                )

        # --- 6e. register keys via EventManager ---
        if not hasattr(data_pkl, "event_manager"):
            data_pkl.event_manager = {}

        data_pkl.event_manager["heart_rate"] = {
            "keys": [
                "heartbeat_auto_detect_accepted",
                "heartbeat_auto_detect_rejected",
                "heartbeat_auto_detect_rejected_conflict",
                "heartbeat_auto_detect_suggested",
                "QC_unusable_ecg",
                "QC_usable_ecg",
            ],
            "description": "Events related to heartbeat detection and ECG quality control",
            "color_map": {
                "heartbeat_auto_detect_accepted":          "#4caf50",
                "heartbeat_auto_detect_rejected":          "#f44336",
                "heartbeat_auto_detect_rejected_conflict": "#ff9800",
                "heartbeat_auto_detect_suggested":         "#ffeb3b",
                "QC_unusable_ecg":                         "rgba(80, 80, 80, 0.5)",
                "QC_usable_ecg":                           "rgba(100, 220, 100, 0.20)",
            },
        }

        print(f"✅ Heartbeat events synced. "
              f"{len(unusable_rows)} unusable spans, {len(usable_rows)} usable spans.")



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
            'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'circle', 'color': 'blue', 'y_offset_frac': -1.5},
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
