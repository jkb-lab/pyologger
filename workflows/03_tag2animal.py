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
from pyologger.calibrate_data.tag2animal import *
from pyologger.utils.workflow_netcdf import (
    latest_processing_netcdf_path,
    netcdf_attr,
    netcdf_has_signal,
    save_step_netcdf_if_changed,
)
from pyologger.analyze_data.activity_periodicity import compute_series_aggregates

# Minimum samples required for the 2 s static window to be meaningful. Below
# this the rolling mean approaches the signal itself and dynamic acceleration
# collapses to ~0 (exactly 0 at a 1-sample window).
MIN_STATIC_WINDOW_SAMPLES = 3

# Static-window width used instead of 2 s on slow archival tags.
LOW_RATE_STATIC_WINDOW_SEC = 3600.0  # 1 hour

# Below this sampling rate, per-sample ODBA is not a usable activity metric
# (fluke/fin beats are at or above Nyquist), so a binned activity channel is
# derived instead. 1 Hz sits well above any archival tag and well below any
# high-rate logger.
LOW_RATE_ACTIVITY_THRESHOLD_HZ = 1.0

# Bin width for the derived low-rate activity channel. Uses the same
# aggregation method as the Wildlife Computers "Series" product (mean of the
# acceleration vector magnitude per bin), at a round 5-minute interval.
DERIVED_ACTIVITY_INTERVAL = "300s"


def _normalize_signal_channel_names(data_pkl, signal_name):
    """Normalize suffixed channels (e.g., gy__corrected_gyr -> gy) if unambiguous."""
    if signal_name not in data_pkl.signal_data:
        return False
    df = data_pkl.signal_data.get(signal_name)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False
    data_cols = [c for c in df.columns if c != "datetime"]
    if not data_cols:
        return False

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
        print(f"[tag2animal] skipped normalization for '{signal_name}' due to collisions: {sorted(collisions)}")
        return False
    if all(str(k) == str(v) for k, v in rename_map.items()):
        return False

    data_pkl.signal_data[signal_name] = df.rename(columns=rename_map)
    sig_info = data_pkl.signal_info.get(signal_name, {})
    if isinstance(sig_info, dict):
        channels = sig_info.get("channels", [])
        if isinstance(channels, list):
            sig_info["channels"] = [rename_map.get(c, c) for c in channels]
        meta = sig_info.get("metadata", {})
        if isinstance(meta, dict):
            new_meta = {}
            for k, v in meta.items():
                new_meta[rename_map.get(k, k)] = v
            sig_info["metadata"] = new_meta
        data_pkl.signal_info[signal_name] = sig_info
    print(f"[tag2animal] normalized channels for '{signal_name}': {rename_map}")
    return True


def _compute_norm_jerk(corrected_acc_df: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
    """Compute full-rate norm-jerk from corrected accelerometer components."""
    required = ["ax", "ay", "az"]
    missing = [c for c in required if c not in corrected_acc_df.columns]
    if missing:
        raise KeyError(f"Cannot compute jerk. Missing corrected_acc columns: {missing}")

    acc = corrected_acc_df[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if acc.shape[0] == 0:
        return pd.DataFrame(columns=["datetime", "jerk"])

    if sampling_rate_hz is None or not np.isfinite(float(sampling_rate_hz)) or float(sampling_rate_hz) <= 0:
        raise ValueError(f"Invalid sampling_rate_hz for jerk computation: {sampling_rate_hz}")

    fs = float(sampling_rate_hz)

    # Forward difference with first sample padded to preserve length.
    dacc = np.diff(acc, axis=0, prepend=acc[[0], :])
    jerk_components = dacc * fs
    norm_jerk = np.linalg.norm(jerk_components, axis=1)

    return pd.DataFrame({
        # Use the Series (not .values) so tz-awareness is preserved. Taking
        # .values on a tz-aware column yields naive datetime64[ns], which later
        # breaks comparisons against the tz-aware analysis window in step 06
        # ("Cannot compare tz-naive and tz-aware datetime-like objects").
        "datetime": corrected_acc_df["datetime"].reset_index(drop=True),
        "jerk": norm_jerk,
    })

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Resolve deployment first so metadata-only skip paths do not require loading data.pkl.
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
latest_netcdf_path = latest_processing_netcdf_path(deployment_folder, deployment_id)

prh_pitch_unit = str(netcdf_attr(latest_netcdf_path, "signal_info_prh_metadata_pitch_unit", "") or "").lower()
prh_roll_unit = str(netcdf_attr(latest_netcdf_path, "signal_info_prh_metadata_roll_unit", "") or "").lower()
prh_exists = netcdf_has_signal(latest_netcdf_path, "prh")
accelerometer_exists = netcdf_has_signal(latest_netcdf_path, "accelerometer")

if prh_exists and "rad" not in prh_pitch_unit and "rad" not in prh_roll_unit:
    print("Skipping Step 03 based on NetCDF metadata: PRH already exists with non-radian units.")
    param_manager.add_to_config("current_processing_step", "Processing Step 03 skipped: PRH already available.")
    raise SystemExit(0)

if not prh_exists and not accelerometer_exists:
    print("Skipping Step 03 based on NetCDF metadata: accelerometer signal not available.")
    param_manager.add_to_config("current_processing_step", "Processing Step 03 skipped: missing accelerometer input.")
    raise SystemExit(0)

with open(pkl_path, "rb") as file:
    data_pkl = pickle.load(file)

# Load key time points
timezone = data_pkl.deployment_info.get('Time Zone', 'UTC')
settings = param_manager.get_from_config(variable_names=["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
OVERLAP_START_TIME = pd.Timestamp(settings["overlap_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["overlap_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)
if None in {OVERLAP_START_TIME, OVERLAP_END_TIME, ZOOM_WINDOW_START_TIME, ZOOM_WINDOW_END_TIME}:
    raise ValueError("One or more required time values were not found in the config file.")

current_processing_step = "Processing Step 03 IN PROGRESS."
param_manager.add_to_config("current_processing_step", current_processing_step)
data_changed = False

# Normalize known suffixed channels even if this step later short-circuits.
normalized_any = False
for _sig in ("corrected_gyr", "corrected_acc", "corrected_mag", "gyroscope", "accelerometer", "magnetometer"):
    normalized_any = _normalize_signal_channel_names(data_pkl, _sig) or normalized_any
if normalized_any:
    data_changed = True
    with open(pkl_path, 'wb') as file:
        pickle.dump(data_pkl, file)
    print("[tag2animal] Saved channel normalization updates to pickle.")

critical_signal = 'calibrated_acc'
converted = False

# If PRH exists, ensure its units are degrees
if 'prh' in data_pkl.signal_data and data_pkl.signal_data['prh'] is not None:
    prh_df = data_pkl.signal_data['prh']
    prh_info = data_pkl.signal_info.get('prh', {})
    metadata = prh_info.get('metadata', {})
    channels = prh_info.get('channels', [c for c in prh_df.columns if c != 'datetime'])

    rad_aliases = {'rad', 'radian', 'radians'}
    for ch in channels:
        if ch not in prh_df.columns:
            continue
        unit = str(metadata.get(ch, {}).get('unit', '')).lower()
        is_radians = (unit in rad_aliases) or ('rad' in unit and unit not in {'deg', 'degree', 'degrees'})
        if is_radians:
            prh_df[ch] = prh_df[ch] * (180.0 / 3.141592653589793)
            if ch in metadata:
                metadata[ch]['unit'] = 'degrees'
            converted = True

    if converted:
        data_changed = True
        prh_info['metadata'] = metadata
        prh_info['transformation_log'] = prh_info.get('transformation_log', [])
        prh_info['transformation_log'].append('converted_prh_angles_to_degrees')
        data_pkl.signal_data['prh'] = prh_df
        data_pkl.signal_info['prh'] = prh_info
        print("Converted PRH from radians to degrees.")

# If PRH was converted to degrees above, persist the change immediately
if converted:
    print("✅ PRH converted to degrees. Saving updated data.")
    print(current_processing_step)
    param_manager.add_to_config("current_processing_step", current_processing_step)
    with open(pkl_path, 'wb') as file:
        pickle.dump(data_pkl, file)
    print("Pickle file updated.")

# Only proceed with critical signal check if PRH does not already exist
if 'prh' in data_pkl.signal_data and data_pkl.signal_data['prh'] is not None:
    skip_step = True
    print("✅ PRH already exists. Skipping processing.")
else:
    # Check for accelerometer data (primary requirement)
    if 'accelerometer' not in data_pkl.signal_data or data_pkl.signal_data['accelerometer'] is None:
        print(f"⚠️ Accelerometer data not found. Cannot proceed with tag2animal processing.")
        skip_step = True
        print(f'‼️ DO NOT PROCEED - Skip_step: {skip_step} due to missing accelerometer data')
    else:
        # Check if we have full sensor suite (accelerometer + magnetometer) or accelerometer-only
        has_mag = (critical_signal in data_pkl.signal_data and 
                   data_pkl.signal_data[critical_signal] is not None and
                   'calibrated_mag' in data_pkl.signal_data and 
                   data_pkl.signal_data['calibrated_mag'] is not None)
        
        if has_mag:
            # Full sensor suite available - use complete calibration workflow
            skip_step = False
            accel_only_mode = False
            print(f'✅ Proceed - Full sensor suite detected. Using complete calibration workflow.')
        else:
            # Accelerometer-only mode - use assumed neutral orientation
            skip_step = False
            accel_only_mode = True
            print(f'⚠️ Accelerometer-only mode: Magnetometer not available.')
            print(f'✅ Proceed - Will use assumed neutral orientation for tag2animal transformation.')

if not skip_step:
    data_changed = True
    if accel_only_mode:
        # ================================================================
        # ACCELEROMETER-ONLY MODE: Assumed Neutral Orientation Processing
        # ================================================================
        print("\n" + "="*60)
        print("ACCELEROMETER-ONLY MODE")
        print("="*60)
        
        # Get accelerometer data
        acc_df = data_pkl.signal_data['accelerometer']
        datetime_data = acc_df['datetime']
        acc_sampling_rate = calculate_sampling_frequency(datetime_data.head())
        print(f"Accelerometer Sampling frequency: {acc_sampling_rate} Hz")
        
        # Get abar0 from config or use default [1.0, 0.0, -1.0] (g units, normalized direction)
        try:
            abar0_config = param_manager.get_from_config(variable_names=["abar0"], section="03_tagtoanimal_settings")
            if abar0_config and abar0_config.get("abar0"):
                abar0_str = abar0_config["abar0"]
                abar0 = [float(x.strip()) for x in abar0_str.split(',')]
                print(f"Using abar0 from config: {abar0}")
            else:
                raise ValueError("abar0 not in config")
        except:
            # Default neutral orientation [1g forward, 0g right, -1g down]
            abar0 = [1.0, 0.0, -1.0]
            print(f"Using default assumed neutral orientation abar0: {abar0}")
        
        # Normalize abar0
        abar = np.array(abar0, dtype=float)
        abar = abar / np.linalg.norm(abar)
        
        # Calculate pitch and roll from abar0 for Euler rotation
        p0 = -np.arcsin(abar[0])
        r0 = np.arctan2(abar[1], abar[2])
        
        # Constrain p to [-pi/2, pi/2]
        if p0 > np.pi / 2:
            p0 = np.pi / 2 - p0
            r0 = r0 + np.pi
        
        print(f"Initial orientation - Pitch: {np.degrees(p0):.2f}°, Roll: {np.degrees(r0):.2f}°")
        
        # Define rotation matrices
        def rotP(p):
            return np.array([[np.cos(p), 0, np.sin(p)],
                           [0, 1, 0],
                           [-np.sin(p), 0, np.cos(p)]])
        
        def rotR(r):
            return np.array([[1, 0, 0],
                           [0, np.cos(r), -np.sin(r)],
                           [0, np.sin(r), np.cos(r)]])
        
        # Calculate rotation matrix W = rotP(p0) * rotR(r0), then transpose
        W = np.matmul(rotP(p0), rotR(r0)).T
        
        # Get accelerometer data as numpy array using standardized channel ids only.
        required_axes = ["ax", "ay", "az"]
        missing_axes = [c for c in required_axes if c not in acc_df.columns]
        if missing_axes:
            raise KeyError(
                "Standardized accelerometer channels are missing in 03_tag2animal. "
                f"Missing: {missing_axes}. Available columns: {list(acc_df.columns)}. "
                "Upstream importer must provide ax/ay/az."
            )
        acc_data = acc_df[required_axes].values
        
        # Apply rotation to get corrected accelerometer
        corrected_acc = np.matmul(acc_data, W)
        
        print(f"✅ Applied Euler rotation to accelerometer data")
        
        # ================================================================
        # Calculate Static Components (2s running average)
        # ================================================================
        # A 2 s static window is only meaningful for high-rate loggers. On slow
        # archival tags (e.g. Wildlife Computers MiniPAT at 1/3 Hz) it rounds to
        # a single sample, making static == raw and dynamic identically zero.
        # Widen the window in that case so the subtraction stays meaningful.
        static_window_sec = 2.0
        n_samples = int(round(static_window_sec * acc_sampling_rate))
        if n_samples < MIN_STATIC_WINDOW_SAMPLES:
            static_window_sec = LOW_RATE_STATIC_WINDOW_SEC
            n_samples = int(round(static_window_sec * acc_sampling_rate))
            print(
                f"⚠️ 2s static window is only {int(round(2.0 * acc_sampling_rate))} sample(s) at "
                f"{acc_sampling_rate:.4f} Hz (dynamic acceleration would be identically zero). "
                f"Using a {static_window_sec:.0f}s window instead ({n_samples} samples)."
            )
        n_samples = max(n_samples, 1)
        min_periods = max(1, n_samples // 5)

        corrected_acc_df = pd.DataFrame(corrected_acc, columns=['ax', 'ay', 'az'])
        static_x = corrected_acc_df['ax'].rolling(window=n_samples, center=True, min_periods=min_periods).mean()
        static_y = corrected_acc_df['ay'].rolling(window=n_samples, center=True, min_periods=min_periods).mean()
        static_z = corrected_acc_df['az'].rolling(window=n_samples, center=True, min_periods=min_periods).mean()

        print(f"✅ Calculated static components ({static_window_sec:g}s window at {acc_sampling_rate} Hz)")
        
        # ================================================================
        # Calculate Dynamic Components
        # ================================================================
        dyn_x = corrected_acc_df['ax'] - static_x
        dyn_y = corrected_acc_df['ay'] - static_y
        dyn_z = corrected_acc_df['az'] - static_z
        
        # Store dynamic acceleration as separate signal
        dynamic_accel_df = pd.DataFrame({
            'datetime': datetime_data,
            'dynX': dyn_x.values,
            'dynY': dyn_y.values,
            'dynZ': dyn_z.values
        })
        
        data_pkl.signal_data['dynamic_accel'] = dynamic_accel_df
        data_pkl.signal_info['dynamic_accel'] = {
            "channels": ["dynX", "dynY", "dynZ"],
            "metadata": {
                'dynX': {'original_name': 'Dynamic Acceleration X', 'unit': 'g', 'signal': 'accelerometer'},
                'dynY': {'original_name': 'Dynamic Acceleration Y', 'unit': 'g', 'signal': 'accelerometer'},
                'dynZ': {'original_name': 'Dynamic Acceleration Z', 'unit': 'g', 'signal': 'accelerometer'}
            },
            "derived_from_signals": ["accelerometer"],
            "transformation_log": [
                f"Calculated dynamic components using {static_window_sec}s running average on rotated accelerometer. "
                f"Rotation based on assumed neutral orientation abar0={abar0}"
            ]
        }
        
        print(f"✅ Calculated dynamic components (dynX, dynY, dynZ)")
        
        # ================================================================
        # Calculate ODBA (Wilson method, 1-norm) at full sampling rate
        # ================================================================
        odba_values = dyn_x.abs() + dyn_y.abs() + dyn_z.abs()
        
        odba_df = pd.DataFrame({
            'datetime': datetime_data,
            'odba': odba_values.values
        })
        
        data_pkl.signal_data['odba'] = odba_df
        data_pkl.signal_info['odba'] = {
            "channels": ["odba"],
            "metadata": {
                "odba": {
                    "original_name": "Overall Dynamic Body Acceleration (Wilson)",
                    "unit": "g"
                }
            },
            "derived_from_signals": ["accelerometer"],
            "transformation_log": [
                f"Wilson ODBA (1-norm) calculated at {acc_sampling_rate} Hz using dynamic components. "
                f"Static window: {static_window_sec}s on rotated accelerometer with abar0={abar0}"
            ]
        }
        
        print(f"✅ Calculated ODBA at full sampling rate ({acc_sampling_rate} Hz)")

        # ================================================================
        # Low-rate fallback: binned activity channel
        # ================================================================
        # Per-sample ODBA is not usable on slow archival tags, so derive a
        # binned activity metric using the same method as the Wildlife
        # Computers "Series" export: mean of the acceleration vector magnitude
        # over each bin, plus its peak-to-peak range. Gravity is included, so
        # values sit near 1 g at rest -- matching the vendor product.
        if acc_sampling_rate < LOW_RATE_ACTIVITY_THRESHOLD_HZ:
            activity_source = pd.DataFrame({
                'datetime': datetime_data.values,
                'ax': corrected_acc_df['ax'].values,
                'ay': corrected_acc_df['ay'].values,
                'az': corrected_acc_df['az'].values,
            })
            activity_binned = compute_series_aggregates(
                activity_source, interval=DERIVED_ACTIVITY_INTERVAL
            )
            activity_df = activity_binned[['datetime', 'activity', 'activity_range']].copy()

            data_pkl.signal_data['activity'] = activity_df
            data_pkl.signal_info['activity'] = {
                "channels": ["activity", "activity_range"],
                "metadata": {
                    "activity": {
                        "original_name": "Binned activity (mean acceleration magnitude)",
                        "unit": "g",
                        "standardized_unit": "g",
                        "parent_signal": "accelerometer",
                    },
                    "activity_range": {
                        "original_name": "Binned activity range (peak-to-peak magnitude)",
                        "unit": "g",
                        "standardized_unit": "g",
                        "parent_signal": "accelerometer",
                    },
                },
                "derived_from_signals": ["accelerometer"],
                "transformation_log": [
                    f"Derived because acc sampling rate {acc_sampling_rate:.4f} Hz is below "
                    f"{LOW_RATE_ACTIVITY_THRESHOLD_HZ} Hz, where per-sample ODBA is not a usable "
                    f"activity metric. Wildlife Computers 'Series' method: mean of "
                    f"||(ax, ay, az)|| per {DERIVED_ACTIVITY_INTERVAL} bin (gravity included, "
                    f"so ~1 g at rest); activity_range is the per-bin peak-to-peak of the same "
                    f"magnitude. Bins are left-closed and labelled by start time."
                ],
            }
            print(
                f"✅ Derived binned 'activity' channel at {DERIVED_ACTIVITY_INTERVAL} "
                f"({len(activity_df):,} bins) because {acc_sampling_rate:.4f} Hz < "
                f"{LOW_RATE_ACTIVITY_THRESHOLD_HZ} Hz"
            )
        
        # ================================================================
        # Calculate Pitch and Roll from Static Components
        # ================================================================
        # Calculate magnitude of static acceleration
        A_static = np.sqrt(static_x**2 + static_y**2 + static_z**2)
        
        # Calculate pitch and roll from static components
        pitch_deg_full = -np.degrees(np.arcsin(static_x / A_static))
        roll_deg_full = np.degrees(np.arctan2(static_y, static_z))
        
        print(f"✅ Calculated pitch and roll from static components")
        
        # ================================================================
        # Downsample Pitch and Roll to 1Hz
        # ================================================================
        # Create DataFrame with full-rate pitch/roll
        prh_full_df = pd.DataFrame({
            'datetime': datetime_data,
            'pitch': pitch_deg_full.values,
            'roll': roll_deg_full.values
        })
        
        # Resample to 1Hz
        prh_full_df.set_index('datetime', inplace=True)
        prh_1hz_df = prh_full_df.resample('1S').mean()
        prh_1hz_df.reset_index(inplace=True)
        
        # Drop any NaN rows
        prh_1hz_df = prh_1hz_df.dropna()
        
        print(f"✅ Downsampled pitch and roll to 1Hz ({len(prh_1hz_df)} samples from {len(prh_full_df)})")
        
        # Store PRH signal (no heading in accelerometer-only mode)
        data_pkl.signal_data['prh'] = prh_1hz_df
        data_pkl.signal_info['prh'] = {
            "channels": ["pitch", "roll"],
            "metadata": {
                'pitch': {
                    'original_name': 'Pitch (degrees)',
                    'unit': 'degrees',
                    'signal': 'accelerometer'
                },
                'roll': {
                    'original_name': 'Roll (degrees)',
                    'unit': 'degrees',
                    'signal': 'accelerometer'
                }
            },
            "derived_from_signals": ["accelerometer"],
            "transformation_log": [
                f"Calculated pitch/roll from static components (2s window) using Euler rotation with assumed abar0={abar0}. "
                f"Downsampled to 1Hz. No heading (magnetometer not available)."
            ]
        }
        
        # ================================================================
        # Store Corrected Accelerometer
        # ================================================================
        corrected_acc_df = pd.DataFrame({
            'datetime': datetime_data,
            'ax': corrected_acc[:, 0],
            'ay': corrected_acc[:, 1],
            'az': corrected_acc[:, 2]
        })
        
        data_pkl.signal_data['corrected_acc'] = corrected_acc_df
        data_pkl.signal_info['corrected_acc'] = {
            "channels": ["ax", "ay", "az"],
            "metadata": {
                'ax': {'original_name': 'Acceleration X', 'unit': 'g', 'signal': 'accelerometer'},
                'ay': {'original_name': 'Acceleration Y', 'unit': 'g', 'signal': 'accelerometer'},
                'az': {'original_name': 'Acceleration Z', 'unit': 'g', 'signal': 'accelerometer'}
            },
            "derived_from_signals": ["accelerometer"],
            "transformation_log": [
                f"Euler rotation applied with assumed neutral orientation abar0={abar0}"
            ]
        }

        # ================================================================
        # Calculate norm-jerk (full rate) from corrected acceleration
        # ================================================================
        jerk_df = _compute_norm_jerk(corrected_acc_df, acc_sampling_rate)
        data_pkl.signal_data['jerk'] = jerk_df
        data_pkl.signal_info['jerk'] = {
            "channels": ["jerk"],
            "metadata": {
                "jerk": {
                    "original_name": "Norm-jerk",
                    "unit": "g/s",
                    "signal": "corrected_acc",
                }
            },
            "derived_from_signals": ["corrected_acc"],
            "transformation_log": [
                f"Norm-jerk calculated at full rate ({acc_sampling_rate} Hz) from first difference of corrected_acc"
            ],
        }
        
        # Save abar0 to config
        settings_to_add = {
            "abar0": ', '.join(f"{val:.3f}" for val in abar0),
            "accel_only_mode": "true"
        }
        param_manager.add_to_config(entries=settings_to_add, section="03_tagtoanimal_settings")
        
        print("\n" + "="*60)
        print("ACCELEROMETER-ONLY PROCESSING COMPLETE")
        print("="*60)
        print(f"Outputs:")
        print(f"  - corrected_acc: Rotated accelerometer at {acc_sampling_rate} Hz")
        print(f"  - dynamic_accel: dynX, dynY, dynZ at {acc_sampling_rate} Hz")
        print(f"  - odba: ODBA at {acc_sampling_rate} Hz")
        print(f"  - jerk: Norm-jerk at {acc_sampling_rate} Hz")
        print(f"  - prh: pitch, roll at 1 Hz (no heading)")
        print("="*60 + "\n")
        
    else:
        # ================================================================
        # FULL SENSOR SUITE MODE: Complete Calibration Workflow
        # ================================================================
        print("\n" + "="*60)
        print("FULL SENSOR SUITE MODE")
        print("="*60)
        
        acc_df = data_pkl.signal_data['calibrated_acc']
        mag_df = data_pkl.signal_data['calibrated_mag']
        gyr_df = data_pkl.signal_data['gyroscope']

        # Calculate and print sampling frequency for each dataframe
        acc_fs = calculate_sampling_frequency(acc_df['datetime'].head())
        print(f"Accelerometer Sampling frequency: {acc_fs} Hz")

        mag_fs = calculate_sampling_frequency(mag_df['datetime'].head())
        print(f"Magnetometer Sampling frequency: {mag_fs} Hz")

        gyr_fs = calculate_sampling_frequency(gyr_df['datetime'].head())
        print(f"Gyroscope Sampling frequency: {gyr_fs} Hz")

        acc_data = acc_df[['ax','ay','az']]
        mag_data = mag_df[['mx','my','mz']]
        gyr_data = gyr_df[['gx', 'gy', 'gz']]

        upsampled_columns = []
        for col in gyr_data.columns:
            upsampled_col = upsample(gyr_data[col].values, acc_fs / gyr_fs, len(acc_data))  # Apply upsample to each column
            upsampled_columns.append(upsampled_col)  # Append the upsampled column to the list

    # Combine the upsampled columns back into a NumPy array
        gyr_data_upsampled = np.column_stack(upsampled_columns)
        gyr_data = gyr_data_upsampled

        acc_data = acc_data.values
        mag_data = mag_data.values

    # Assuming acc_data and mag_data_upsampled are NumPy arrays
        print(f"Gyroscope upsampled to match accelerometer length: acc_data shape= {acc_data.shape}, upsampled gyr_data shape = {gyr_data.shape}")
        sampling_rate = acc_fs

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

    # Check if manual update is required for CALM_HORIZONTAL_START_TIME and CALM_HORIZONTAL_END_TIME
        requires_manual_update = False
        if not CALM_HORIZONTAL_START_TIME or CALM_HORIZONTAL_START_TIME == "PLACEHOLDER":
            requires_manual_update = True
        if not CALM_HORIZONTAL_END_TIME or CALM_HORIZONTAL_END_TIME == "PLACEHOLDER":
            requires_manual_update = True

    # Use ZOOM_WINDOW values as defaults if CALM_HORIZONTAL times are placeholders
        if requires_manual_update:
            CALM_HORIZONTAL_START_TIME = CALM_HORIZONTAL_START_TIME or ZOOM_START_TIME
            CALM_HORIZONTAL_END_TIME = CALM_HORIZONTAL_END_TIME or ZOOM_END_TIME

    # Display values to the user
        print("CALM_HORIZONTAL_START_TIME (current or default):", CALM_HORIZONTAL_START_TIME)
        print("CALM_HORIZONTAL_END_TIME (current or default):", CALM_HORIZONTAL_END_TIME)

    # Display a message based on whether manual update is needed
        if requires_manual_update:
            print("Calm horizontal start and end times require manual update. Proceed to Cell 2 to set placeholders.")
        else:
            print("Calm horizontal start and end times are already set. No further action is required.")

    # Retrieve timezone from deployment info
        timezone = data_pkl.deployment_info['Time Zone']

    # Define placeholder timestamps for calm period in the retrieved timezone
        placeholder_start_time = ZOOM_WINDOW_START_TIME
        placeholder_end_time = ZOOM_WINDOW_END_TIME

    # Set this to True if we want to override the placeholders regardless of manual update status
        override_required = False

    # Only update if manual update is required or if override is enabled
        if requires_manual_update or override_required:
            CALM_HORIZONTAL_START_TIME = str(placeholder_start_time)
            CALM_HORIZONTAL_END_TIME = str(placeholder_end_time)
        
        # Use ParamManager to add placeholders to the config
            param_manager.add_to_config("calm_horizontal_start_time",
                value=CALM_HORIZONTAL_START_TIME,
                section="settings"
            )
            param_manager.add_to_config("calm_horizontal_end_time",
                value=CALM_HORIZONTAL_END_TIME,
                section="settings"
            )

            print("Timestamps for calm horizontal start and end times have been set and saved.")
        else:
            print("Manual update not required. Placeholders were not set.")

        start_time = CALM_HORIZONTAL_START_TIME
        end_time   = CALM_HORIZONTAL_END_TIME

    # Filter the calibrated accelerometer to include only rows within the specified calm time range
        calm_source_df = data_pkl.signal_data.get('calibrated_acc', data_pkl.signal_data['accelerometer'])
        calm_dt = pd.to_datetime(calm_source_df['datetime'], errors='coerce')
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        if getattr(calm_dt.dt, "tz", None) is not None and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(calm_dt.dt.tz)
            end_ts = end_ts.tz_localize(calm_dt.dt.tz)
        elif getattr(calm_dt.dt, "tz", None) is None and start_ts.tzinfo is not None:
            calm_dt = calm_dt.dt.tz_localize(start_ts.tzinfo)
        filtered_df = calm_source_df[(calm_dt >= start_ts) & (calm_dt <= end_ts)]
        finite_counts = filtered_df[['ax', 'ay', 'az']].apply(pd.to_numeric, errors='coerce').notna().sum()
        print(
            f"[tag2animal] calm window rows={len(filtered_df)}, "
            f"finite ax/ay/az={finite_counts.to_dict()}"
        )

    # Calculate robust finite means for ax/ay/az in calm window
        mean_values = filtered_df[['ax', 'ay', 'az']].apply(pd.to_numeric, errors='coerce').mean()
        mean_is_finite = np.isfinite(mean_values.values).all()

        print(f"Average values between {start_time} and {end_time}:")
        print(mean_values)

        if not mean_is_finite:
            print(
                "⚠️ Calm-window abar0 is invalid (NaN/Inf). "
                "Falling back to prior config abar0, then full finite calibrated_acc mean, then default [1,0,-1]."
            )

            fallback_abar0 = None
            prior = param_manager.get_from_config(variable_names=["abar0"], section="03_tagtoanimal_settings")
            prior_abar0_str = (prior or {}).get("abar0")
            if prior_abar0_str:
                try:
                    parsed = np.array([float(x.strip()) for x in str(prior_abar0_str).split(',')], dtype=float)
                    if parsed.shape == (3,) and np.isfinite(parsed).all() and np.linalg.norm(parsed) > 0:
                        fallback_abar0 = parsed
                        print(f"✅ Using prior config abar0 fallback: {fallback_abar0.tolist()}")
                except Exception:
                    pass

            if fallback_abar0 is None:
                full_mean = calm_source_df[['ax', 'ay', 'az']].apply(pd.to_numeric, errors='coerce').mean()
                if np.isfinite(full_mean.values).all() and np.linalg.norm(full_mean.values) > 0:
                    fallback_abar0 = full_mean.values.astype(float)
                    print(f"✅ Using full calibrated_acc mean fallback: {fallback_abar0.tolist()}")

            if fallback_abar0 is None:
                fallback_abar0 = np.array([1.0, 0.0, -1.0], dtype=float)
                print("⚠️ Using hard fallback abar0=[1.0, 0.0, -1.0].")

            abar0 = fallback_abar0.tolist()
            mean_values = pd.Series(abar0, index=['ax', 'ay', 'az'])
        else:
            abar0 = [mean_values['ax'], mean_values['ay'], mean_values['az']]
    # abar0 = [0, 0, -9.8] # to override - use this or similar for orca and other standard CATS tags
        deploy_latitude = data_pkl.deployment_info["Deployment Latitude"]
        deploy_longitude = data_pkl.deployment_info["Deployment Longitude"]

        print(f"Using location Lat: {deploy_latitude}, Lon: {deploy_longitude} and stationary readings of abar0: {str(abar0)} to orient tag.")

    # Use the function to get corrected orientation and heading for the entire dataset
        pitch_deg, roll_deg, heading_deg, corrected_acc, corrected_mag, corrected_gyr = orientation_and_heading_correction(
            abar0, 
            latitude= deploy_latitude,
            longitude= deploy_longitude,
            acc_data=acc_data, 
            mag_data=mag_data, 
            gyr_data=gyr_data)

    # Define multiple key-value pairs to add under a section
        settings_to_add = {
            "declination_latitude": deploy_latitude,
            "declination_longitude": deploy_longitude,
            "abar0": ', '.join(f"{value:.3f}" for value in mean_values)
        }

    # Add the settings under a specific section
        param_manager.add_to_config(entries=settings_to_add, section="03_tagtoanimal_settings")

        print(f"Tag to animal correction settings saved and added to config file.")

    # One datetime column from highest sampled data that was matched by other signals
        datetime_data = data_pkl.signal_data['accelerometer']['datetime']

    # Step 1: Create a DataFrame for pitch, roll, and heading
        prh_df = pd.DataFrame({
            'datetime': datetime_data,
            'pitch': pitch_deg,
            'roll': roll_deg,
            'heading': heading_deg
        })

    # Store the 'prh' variable in signal_data
        data_pkl.signal_data['prh'] = prh_df
        data_pkl.signal_info['prh'] = {
            "channels": ["pitch", "roll", "heading"],
            "metadata": {
                'pitch': {'original_name': 'Pitch (degrees)',
                        'unit': 'degrees',
                        'signal': 'accelerometer'},
                'roll': {'original_name': 'Roll (degrees)',
                        'unit': 'degrees',
                        'signal': 'accelerometer'},
                'heading': {'original_name': 'Heading (degrees)',
                            'unit': 'degrees',
                            'signal': 'magnetometer'}
            },
            "derived_from_signals": ["accelerometer", "magnetometer"],
            "transformation_log": [f"calculated_pitch_roll_heading using abar0: {abar0} from calibration period with start time: {start_time} and end time: {end_time} at Deployment Latitude: {deploy_latitude} and Deployment Longitude: {deploy_longitude}."]
        }

    # Step 2: Create DataFrames for corrected accelerometer, magnetometer, and gyroscope data
        corrected_acc_df = pd.DataFrame({
            'datetime': datetime_data,
            'ax': corrected_acc[:, 0],
            'ay': corrected_acc[:, 1],
            'az': corrected_acc[:, 2]
        })

        corrected_mag_df = pd.DataFrame({
            'datetime': datetime_data,
            'mx': corrected_mag[:, 0],
            'my': corrected_mag[:, 1],
            'mz': corrected_mag[:, 2]
        })

        corrected_gyr_df = pd.DataFrame({
            'datetime': datetime_data,
            'gx': corrected_gyr[:, 0],
            'gy': corrected_gyr[:, 1],
            'gz': corrected_gyr[:, 2]
        })

    # Step 3: Store the corrected accelerometer, magnetometer, and gyroscope data into signal_data
        data_pkl.signal_data['corrected_acc'] = corrected_acc_df
        data_pkl.signal_info['corrected_acc'] = {
            "channels": ["ax", "ay", "az"],
            "metadata": {
                'ax': {'original_name': 'Acceleration X (m/s^2)',
                    'unit': 'm/s^2',
                    'signal': 'accelerometer'},
                'ay': {'original_name': 'Acceleration Y (m/s^2)',
                    'unit': 'm/s^2',
                    'signal': 'accelerometer'},
                'az': {'original_name': 'Acceleration Z (m/s^2)',
                    'unit': 'm/s^2',
                    'signal': 'accelerometer'}
            },
            "derived_from_signals": ["accelerometer"],
            "transformation_log": ["corrected_orientation"]
        }

        data_pkl.signal_data['corrected_mag'] = corrected_mag_df
        data_pkl.signal_info['corrected_mag'] = {
            "channels": ["mx", "my", "mz"],
            "metadata": {
                'mx': {'original_name': 'Magnetometer X (µT)',
                    'unit': 'µT',
                    'signal': 'magnetometer'},
                'my': {'original_name': 'Magnetometer Y (µT)',
                    'unit': 'µT',
                    'signal': 'magnetometer'},
                'mz': {'original_name': 'Magnetometer Z (µT)',
                    'unit': 'µT',
                    'signal': 'magnetometer'}
            },
            "derived_from_signals": ["magnetometer"],
            "transformation_log": ["corrected_orientation"]
        }

        data_pkl.signal_data['corrected_gyr'] = corrected_gyr_df
        data_pkl.signal_info['corrected_gyr'] = {
            "channels": ["gx", "gy", "gz"],
            "metadata": {
                'gx': {'original_name': 'Gyroscope X (deg/s)',
                    'unit': 'deg/s',
                    'signal': 'gyroscope'},
                'gy': {'original_name': 'Gyroscope Y (deg/s)',
                    'unit': 'deg/s',
                    'signal': 'gyroscope'},
                'gz': {'original_name': 'Gyroscope Z (deg/s)',
                    'unit': 'deg/s',
                    'signal': 'gyroscope'}
            },
            "derived_from_signals": ["gyroscope"],
            "transformation_log": ["corrected_orientation"]
        }

        # Calculate norm-jerk (full rate) from corrected acceleration.
        jerk_df = _compute_norm_jerk(corrected_acc_df, acc_fs)
        data_pkl.signal_data['jerk'] = jerk_df
        data_pkl.signal_info['jerk'] = {
            "channels": ["jerk"],
            "metadata": {
                "jerk": {
                    "original_name": "Norm-jerk",
                    "unit": "m/s^3",
                    "signal": "corrected_acc",
                }
            },
            "derived_from_signals": ["corrected_acc"],
            "transformation_log": [
                f"Norm-jerk calculated at full rate ({acc_fs} Hz) from first difference of corrected_acc"
            ],
        }

        TARGET_SAMPLING_RATE = 10

        notes_to_plot = {
            'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
            'heartbeat_auto_detect_accepted': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
            'heartbeat_auto_detect_rejected': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'red'}
        }

    # fig = plot_tag_data_interactive(
    #     data_pkl=data_pkl,
    #     signals=['ecg', 'accelerometer', 'magnetometer','depth', 'corrected_acc', 'corrected_mag', 'prh'],
    #     channels={},
    #     time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
    #     note_annotations=notes_to_plot,
    #     color_mapping_path=color_mapping_path,
    #     target_sampling_rate=TARGET_SAMPLING_RATE,
    #     zoom_start_time=ZOOM_START_TIME,
    #     zoom_end_time=ZOOM_END_TIME,
    #     zoom_range_selector_channel='depth',
    #     plot_event_values=[],
    # )
    #fig.show()

        keys_to_remove = ['calibrated_acc','calibrated_mag']

    # Clear the specified keys
        clear_intermediate_signals(data_pkl, remove_keys=keys_to_remove)

else:
    print(f"Skipping step due to missing critical data.")

current_processing_step = "Processing Step 03. Tag frame to animal frame transformation complete."
print(current_processing_step)

# Add or update the current_processing_step for the specified deployment
print(current_processing_step)
param_manager.add_to_config("current_processing_step", current_processing_step)

# Optional: save new pickle file
if data_changed:
    with open(pkl_path, 'wb') as file:
            pickle.dump(data_pkl, file)
    print("Pickle file updated.")

save_step_netcdf_if_changed(data_pkl, deployment_folder, deployment_id, 3, changed=data_changed)
