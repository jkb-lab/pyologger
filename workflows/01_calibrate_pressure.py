# Run with shell command: python3 pyologger/workflows/01_calibrate_pressure.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
# Zero offset correction: calibrate pressure signal
import os
import pickle
import argparse
import sys
import pandas as pd
import numpy as np
import xarray as xr
import ast
import pytz

# Ensure direct workflow execution resolves the repo-local pyologger package.
WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)

# How deep the baseline-adjusted median must be, in metres, before a negative median is
# read as an inverted signal rather than a surface offset. Shallow coastal deployments
# hover just below zero when already correct; deep divers sit tens to hundreds of metres
# down when inverted. See the post-baseline sign check for the observed separation.
SIGN_FLIP_MIN_MEDIAN_DEPTH_M = 10.0
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.event_manager import *
from pyologger.plot_data.plotter import *
from pyologger.calibrate_data.zoc import *
from pyologger.io_operations.base_exporter import *
from pyologger.analyze_data.find_segments import *
from pyologger.analyze_data.analyze_segments import *
from pyologger.utils.workflow_netcdf import (
    latest_processing_netcdf_path,
    netcdf_has_signal,
    netcdf_signal_has_channel,
    save_step_netcdf_if_changed,
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite data_pkl pressure channel from outputs/{deployment}_00_processed.nc before processing."
)
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

# Resolve deployment first so critical-signal checks can inspect NetCDF metadata.
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
latest_netcdf_path = latest_processing_netcdf_path(deployment_folder, deployment_id)

has_pressure_signal = netcdf_has_signal(latest_netcdf_path, "pressure")
has_pressure_channel = netcdf_signal_has_channel(latest_netcdf_path, "pressure", "pressure")
has_depth_signal = netcdf_has_signal(latest_netcdf_path, "depth")
has_depth_channel = (
    netcdf_signal_has_channel(latest_netcdf_path, "depth", "depth") or
    netcdf_signal_has_channel(latest_netcdf_path, "depth", "corrected_depth")
)

if not ((has_pressure_signal and has_pressure_channel) or (has_depth_signal and has_depth_channel)):
    print(
        "Skipping Step 01 based on NetCDF metadata: neither a usable pressure signal "
        "nor a usable depth signal is available."
    )
    param_manager.add_to_config("current_processing_step", "Processing Step 01 skipped: missing pressure/depth input.")
    raise SystemExit(0)

with open(pkl_path, "rb") as file:
    data_pkl = pickle.load(file)


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except Exception:
                pass
        return [txt]
    return [str(value)]


def overwrite_pressure_from_netcdf(data_obj, deployment_folder_path, deployment_name):
    nc_path = os.path.join(deployment_folder_path, "outputs", f"{deployment_name}_00_processed.nc")
    if not os.path.exists(nc_path):
        print(f"⚠️ Missing NetCDF for overwrite: {nc_path}")
        return False

    with xr.open_dataset(nc_path) as ds:
        var_name = "signal_data_pressure"
        if var_name not in ds.variables:
            print(f"⚠️ '{var_name}' not found in NetCDF: {nc_path}")
            return False

        da = ds[var_name]
        if da.ndim < 1:
            raise ValueError(f"{var_name} has invalid dimensions: {da.dims}")

        sample_dim = da.dims[0]
        datetimes = pd.to_datetime(da.coords[sample_dim].values)

        channel_names = _to_list(da.attrs.get("variables")) or _to_list(da.attrs.get("variable"))
        values = np.asarray(da.values)

        if values.ndim == 1:
            pressure_values = values
        else:
            if not channel_names:
                channel_names = [f"ch_{i}" for i in range(values.shape[1])]
            if "pressure" in channel_names:
                pressure_idx = channel_names.index("pressure")
            else:
                pressure_idx = 0
                print(f"⚠️ 'pressure' channel not found in {var_name} attrs; using first channel '{channel_names[0]}'.")
            pressure_values = values[:, pressure_idx]

        pressure_df = pd.DataFrame({
            "datetime": datetimes,
            "pressure": pd.to_numeric(pressure_values, errors="coerce")
        })

        data_obj.signal_data["pressure"] = pressure_df

        if not hasattr(data_obj, "signal_info") or not isinstance(data_obj.signal_info, dict):
            data_obj.signal_info = {}

        pressure_info = data_obj.signal_info.get("pressure", {})
        if not isinstance(pressure_info, dict):
            pressure_info = {}

        # Pull pressure metadata from NetCDF attrs when available.
        nc_attrs = ds.attrs
        original_units_from_nc = nc_attrs.get("signal_info_pressure_original_units")
        units_from_nc = nc_attrs.get("signal_info_pressure_units")
        if pressure_info.get("original_units") in (None, "", "unknown") and original_units_from_nc not in (None, "", "unknown"):
            pressure_info["original_units"] = original_units_from_nc
        if pressure_info.get("units") in (None, "", "unknown") and units_from_nc not in (None, "", "unknown"):
            pressure_info["units"] = units_from_nc
        pressure_info["sampling_frequency"] = pressure_info.get("sampling_frequency", nc_attrs.get("signal_info_pressure_sampling_frequency"))
        pressure_info["original_sampling_frequency"] = pressure_info.get("original_sampling_frequency", nc_attrs.get("signal_info_pressure_original_sampling_frequency"))
        pressure_info["logger_manufacturer"] = pressure_info.get("logger_manufacturer", nc_attrs.get("signal_info_pressure_logger_manufacturer"))
        pressure_info["channels"] = ["pressure"]
        pressure_info["metadata"] = pressure_info.get("metadata", {})
        if isinstance(pressure_info["metadata"], dict):
            pressure_info["metadata"]["pressure"] = pressure_info["metadata"].get("pressure", {})
            if isinstance(pressure_info["metadata"]["pressure"], dict):
                pressure_info["metadata"]["pressure"]["parent_signal"] = pressure_info["metadata"]["pressure"].get("parent_signal", "pressure")

        # Fallback frequency estimation if missing from metadata.
        if pressure_info.get("sampling_frequency") in (None, "", "unknown"):
            dt_seconds = pressure_df["datetime"].diff().dt.total_seconds().dropna()
            dt_seconds = dt_seconds[dt_seconds > 0]
            if not dt_seconds.empty:
                inferred_fs = float(round(1.0 / dt_seconds.median(), 6))
                pressure_info["sampling_frequency"] = inferred_fs
                if pressure_info.get("original_sampling_frequency") in (None, "", "unknown"):
                    pressure_info["original_sampling_frequency"] = inferred_fs

        data_obj.signal_info["pressure"] = pressure_info

    print(f"✅ Overwrote pressure channel from NetCDF: {nc_path} ({len(data_obj.signal_data['pressure'])} samples)")
    return True


overwrite_from_config = _as_bool(config.get("overwrite_step01_from_nc", False))
if args.overwrite or overwrite_from_config:
    pressure_present = (
        hasattr(data_pkl, "signal_data")
        and isinstance(data_pkl.signal_data, dict)
        and "pressure" in data_pkl.signal_data
        and data_pkl.signal_data["pressure"] is not None
    )
    if pressure_present:
        print("ℹ️ Pressure signal already exists in data.pkl; skipping NetCDF overwrite.")
    else:
        print("ℹ️ Pressure signal missing in data.pkl; attempting NetCDF overwrite.")
        overwrite_ok = overwrite_pressure_from_netcdf(data_pkl, deployment_folder, deployment_id)
        if not overwrite_ok:
            print("ℹ️ NetCDF overwrite unavailable; continuing with in-memory critical signal checks.")


def normalize_event_datetimes(event_df, deployment_tz_name):
    """
    Ensure event datetimes are consistently timezone-aware in deployment timezone.
    - Naive timestamps are assumed to be deployment-local and localized.
    - Aware timestamps are converted to deployment timezone.
    """
    if not isinstance(event_df, pd.DataFrame) or event_df.empty or "datetime" not in event_df.columns:
        return event_df

    tz_name = str(deployment_tz_name).strip() if deployment_tz_name else ""
    if not tz_name:
        raise ValueError("No Time Zone set for deployment.")
    try:
        target_tz = pytz.timezone(tz_name)
    except Exception as e:
        raise ValueError(f"Invalid Time Zone set for deployment: '{tz_name}'") from e

    def _normalize_one(value):
        if pd.isna(value):
            return pd.NaT
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize(target_tz)
        return ts.tz_convert(target_tz)

    out = event_df.copy()
    out["datetime"] = out["datetime"].apply(_normalize_one)

    # Keep UTC mirror fields in sync when present.
    if "datetime_utc" in out.columns:
        out["datetime_utc"] = out["datetime"].dt.tz_convert("UTC")
    if "time_unix_ms" in out.columns:
        out["time_unix_ms"] = (out["datetime"].astype("int64") // 10**6).astype("Int64")

    print(f"✅ Normalized event_data datetimes to timezone: {tz_name}")
    return out


def normalize_datetime_series_to_tz(datetime_series, deployment_tz_name):
    """
    Normalize a datetime series to deployment timezone.
    - Naive timestamps are localized to deployment timezone.
    - Aware timestamps are converted to deployment timezone.
    """
    tz_name = str(deployment_tz_name).strip() if deployment_tz_name else ""
    if not tz_name:
        raise ValueError("No Time Zone set for deployment.")
    try:
        _ = pytz.timezone(tz_name)
    except Exception as e:
        raise ValueError(f"Invalid Time Zone set for deployment: '{tz_name}'") from e

    dt = pd.to_datetime(datetime_series, errors="coerce")
    if isinstance(dt, pd.Series):
        if dt.dt.tz is None:
            return dt.dt.tz_localize(tz_name)
        return dt.dt.tz_convert(tz_name)
    if dt.tz is None:
        return dt.tz_localize(tz_name)
    return dt.tz_convert(tz_name)

# Load key time points
timezone = str(data_pkl.deployment_info.get('Time Zone', '')).strip()
if not timezone:
    raise ValueError("No Time Zone set for deployment.")
try:
    pytz.timezone(timezone)
except Exception as e:
    raise ValueError(f"Invalid Time Zone set for deployment: '{timezone}'") from e
settings = param_manager.get_from_config(variable_names=["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
OVERLAP_START_TIME = pd.Timestamp(settings["overlap_start_time"]).tz_convert(timezone)
OVERLAP_END_TIME = pd.Timestamp(settings["overlap_end_time"]).tz_convert(timezone)
ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)

# Confirm the values or raise an error if any are missing
if None in {OVERLAP_START_TIME, OVERLAP_END_TIME, ZOOM_WINDOW_START_TIME, ZOOM_WINDOW_END_TIME}:
    raise ValueError("One or more required time values were not found in the config file.")

current_processing_step = "Processing Step 01 IN PROGRESS."
param_manager.add_to_config("current_processing_step", current_processing_step)
data_changed = False

critical_signal = 'pressure'
has_derived_depth = (
    hasattr(data_pkl, 'signal_data')
    and isinstance(data_pkl.signal_data, dict)
    and 'depth' in data_pkl.signal_data
    and data_pkl.signal_data['depth'] is not None
)

# Check if pressure is missing
pressure_is_missing = (
    critical_signal not in data_pkl.signal_data 
    or data_pkl.signal_data[critical_signal] is None
)

if has_derived_depth and pressure_is_missing:
    print("✅ data_pkl.signal_data['depth'] exists and pressure is missing.")
    print("✅ Copying depth to pressure...")
    newdepth_df = data_pkl.signal_data['depth'].copy()
    if 'depth' in newdepth_df.columns and 'pressure' not in newdepth_df.columns:
        newdepth_df = newdepth_df.rename(columns={'depth': 'pressure'})
    if 'corrected_depth' in newdepth_df.columns and 'pressure' not in newdepth_df.columns:
        newdepth_df = newdepth_df.rename(columns={'corrected_depth': 'pressure'})
    data_pkl.signal_data['pressure'] = newdepth_df
    if hasattr(data_pkl, 'signal_info') and isinstance(data_pkl.signal_info, dict):
        depth_info = data_pkl.signal_info.get('depth')
        if depth_info is not None:
            if 'channels' in depth_info and depth_info['channels'] == ['depth']:
                depth_info['channels'] = ['pressure']
            if 'metadata' in depth_info and 'depth' in depth_info['metadata']:
                depth_info['metadata']['pressure'] = depth_info['metadata'].pop('depth')
            data_pkl.signal_info['depth'] = depth_info
            data_pkl.signal_info['pressure'] = depth_info
    print("✅ Copied depth data and metadata to pressure, with pressure column.")
    skip_step = False
    data_changed = True
elif has_derived_depth and not pressure_is_missing:
    print("✅ data_pkl.signal_data['depth'] exists but pressure already exists.")
    print("✅ Keeping existing pressure data without overwriting.")
    skip_step = False
else:
    # If critical signal doesn't exist, create flag to skip step.
    if critical_signal not in data_pkl.signal_data or data_pkl.signal_data[critical_signal] is None:
        print(f"⚠️ Signal: {critical_signal} not found. Skipping processing.")
        skip_step = True
        print(f'‼️ DO NOT PROCEED - Skip_step: {skip_step} due to missing critical signal: {critical_signal}')
    elif 'pressure' not in data_pkl.signal_data[critical_signal].columns:
        print("⚠️ Pressure column not found in pressure signal. Skipping processing.")
        skip_step = True
    else:
        # signals exists and can be processed normally
        skip_step = False
        print(f'✅ Proceed - Skip_step: {skip_step}. Critical signal: {critical_signal} found.')

if not skip_step:
    data_changed = True
    # **Step 1: Clean and prepare data**
    # Keep raw pressure channel unchanged; all conversions/corrections apply only
    # to this local working copy used to derive depth.
    depth_data = data_pkl.signal_data["pressure"]["pressure"].copy()
    depth_datetime = normalize_datetime_series_to_tz(
        data_pkl.signal_data["pressure"]["datetime"],
        data_pkl.deployment_info.get("Time Zone", "")
    )
    depth_fs = data_pkl.signal_info["pressure"]["sampling_frequency"]

    # 0. Convert working depth copy to meters if necessary
    original_pressure_unit = str(data_pkl.signal_info['pressure'].get('original_units', 'unknown')).strip().lower()
    pressure_unit = str(data_pkl.signal_info['pressure'].get('units', 'unknown')).strip().lower()

    if original_pressure_unit == 'bar':
        print("Converting working depth copy from bar to m")
        depth_data *= 10
        print("✅ Working depth copy converted from bar to m")
    elif original_pressure_unit == 'cm':
        print("Converting working depth copy from cm to m")
        depth_data /= 100
        print("✅ Working depth copy converted from cm to m")
    elif original_pressure_unit in ['m', '100bar', '100bar_1', '30bar_1', 'msw'] or pressure_unit == 'm': # including CATS format weird 100bar_1 which seems to be m
        print("✅ Working depth copy already in m")
    else:
        print(f"Unknown pressure unit: {original_pressure_unit}")
        raise ValueError(f"Unknown pressure unit: {original_pressure_unit}")

    restart_threshold_setting = param_manager.get_from_config(
        variable_names=["logger_restart_pressure_threshold"],
        section="dive_detection_settings"
    ).get("logger_restart_pressure_threshold")
    try:
        logger_restart_pressure_threshold = float(restart_threshold_setting)
    except (TypeError, ValueError):
        logger_restart_pressure_threshold = -500.0

    # 1. Check if logger is known to produce extreme pressure values
    if data_pkl.signal_info['pressure']['logger_manufacturer'] == 'Evolocus':
        # 2. Check if logger_restart events have already been added
        if data_pkl.event_data is None or data_pkl.event_data.empty or not any(data_pkl.event_data['key'] == 'logger_restart'):
            # 3. Identify bad segments based on unrealistic negative pressure
            pressure_df = pd.DataFrame({
                "datetime": depth_datetime,
                "pressure": depth_data
            })
            restarts = find_segments(
                data=pressure_df,
                column='pressure',
                criteria=lambda x: x < logger_restart_pressure_threshold,
                min_duration=None
            )

            # 4. Add restart events using standardized event creation
            if not restarts.empty:
                data_pkl.event_data = create_state_event(
                    state_df=restarts,
                    key="logger_restart",
                    start_time_column="start_datetime",
                    duration_column="duration",
                    description="Detected logger restart from extreme pressure",
                    long_description=(
                        "Logger restart inferred from pressure values dropping below "
                        f"{logger_restart_pressure_threshold}, typically inserted by logger hardware during reboot."
                    ),
                    existing_events=data_pkl.event_data
                )
                print(f"🟠 Added {len(restarts)} logger_restart event(s) to event_data.")

                # 5. Replace working depth values with NaN around each segment
                datetimes = depth_datetime
                for _, row in restarts.iterrows():
                    start = row['start_datetime']
                    end = row['end_datetime']
                    mask = (datetimes >= start) & (datetimes <= end)
                    buffer_before = datetimes.shift(1)
                    buffer_after = datetimes.shift(-1)
                    buffer_mask = (buffer_before >= start) & (buffer_before <= end) | (buffer_after >= start) & (buffer_after <= end)
                    full_mask = mask | buffer_mask
                    depth_data.loc[full_mask] = np.nan
                print(
                    "⚠️ Replaced extreme working depth values "
                    f"(< {logger_restart_pressure_threshold}) and surrounding buffer with NaN."
                )
            else:
                print("✅ No restart segments detected.")
        else:
            print("✅ Logger restart events already exist in event_data.")

        # 6. Check again in case any extreme values remain outside known segments
        extreme_exists = (depth_data < logger_restart_pressure_threshold).any()
        if extreme_exists:
            depth_data.loc[depth_data < logger_restart_pressure_threshold] = np.nan
            print(
                "⚠️ Replaced residual extreme working depth values "
                f"(< {logger_restart_pressure_threshold}) with NaN."
            )
        else:
            print("✅ No extreme pressure values found.")
    else:
        print("✅ No logger restart check needed for this logger.")

    # Step 4: Load temperature data (depth data already loaded as working copy above)
    if 'temperature-ext' in data_pkl.signal_data:
        temp_data = data_pkl.signal_data['temperature-ext']['temp-ext']
        temp_fs = data_pkl.signal_info['temperature-ext']['sampling_frequency']
    elif 'temperature-int' in data_pkl.signal_data:
        temp_data = data_pkl.signal_data['temperature-int']['temp-int']
        temp_fs = data_pkl.signal_info['temperature-int']['sampling_frequency']
    else:
        temp_data = None
        temp_fs = None

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
        "first_deriv_threshold": 0.1, "min_duration": 60, "depth_threshold": 5,
        "apply_temp_correction": False, "min_depth_threshold": 0.5,
        "dive_duration_threshold": 10, "smoothing_window": 5,
        "downsampled_sampling_rate": 1, "baseline_adjust": 0.0,
        "use_flat_chunks": True, "min_flat_chunks_for_zoc": 10,
        "disable_automatic_sign_flipping": False, "conversion_factor": 1.0,
        "logger_restart_pressure_threshold": -500.0
    }

    # If settings are missing or None, initialize them
    if dive_detection_settings is None:
        dive_detection_settings = default_settings.copy()
    else:
        # Fill missing/None keys from defaults.
        dive_detection_settings = {
            k: dive_detection_settings.get(k) if dive_detection_settings.get(k) is not None else v
            for k, v in default_settings.items()
        }

    # Do not auto-write resolved defaults into deployment config.
    # Dataset defaults should remain the fallback source unless a deployment-specific
    # value is explicitly set in parameter_log.json.

    # Print to confirm
    print(f"✅ Loaded downsampled_sampling_rate: {dive_detection_settings['downsampled_sampling_rate']}")

    conversion_factor = float(dive_detection_settings.get("conversion_factor", 1.0))
    # Allow -1.0 for sign flipping, otherwise clamp to reasonable positive range
    if conversion_factor != -1.0:
        conversion_factor = max(0.01, min(100.0, conversion_factor))
    dive_detection_settings["conversion_factor"] = conversion_factor
    try:
        logger_restart_pressure_threshold = float(
            dive_detection_settings.get("logger_restart_pressure_threshold", -500.0)
        )
    except (TypeError, ValueError):
        logger_restart_pressure_threshold = -500.0
    dive_detection_settings["logger_restart_pressure_threshold"] = logger_restart_pressure_threshold
    if conversion_factor != 1.0:
        depth_data = depth_data * conversion_factor
        print(f"✅ Applied pressure conversion_factor={conversion_factor}")
    else:
        print("✅ conversion_factor is 1.0; no additional pressure scaling applied.")

    # Step 6: Process depth data - Downsample, smooth, adjust baseline, and calculate first derivative
    # Interpolate NaNs for processing
    interpolated_depth_data = depth_data.interpolate(limit_direction='both')

    # Step 6: Process depth data - Downsample, smooth, adjust baseline, and calculate first derivative
    depth_processing_params = {
        "original_sampling_rate": depth_fs,
        "downsampled_sampling_rate": int(dive_detection_settings["downsampled_sampling_rate"]),
        "baseline_adjust": dive_detection_settings["baseline_adjust"]  # New parameter
    }

    try:
        first_derivative, downsampled_depth = smooth_downsample_derivative(
            interpolated_depth_data, **depth_processing_params
        )
    except ValueError as exc:
        # SciPy decimate can fail on very short vectors (e.g., len(x) <= padlen).
        # Fall back to stride-based downsampling with median smoothing so we can proceed.
        if "padlen" not in str(exc):
            raise
        target_rate = int(dive_detection_settings["downsampled_sampling_rate"])
        downsample_factor = int(depth_fs / target_rate) if target_rate > 0 else 1
        raw_depth = pd.to_numeric(interpolated_depth_data, errors="coerce").to_numpy()
        if downsample_factor <= 1:
            downsampled_depth = raw_depth.copy()
        else:
            downsampled_depth = raw_depth[::downsample_factor]
        if downsampled_depth.size == 0:
            downsampled_depth = raw_depth[:1]

        downsampled_depth = (
            pd.Series(downsampled_depth)
            .rolling(window=5, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        downsampled_depth = downsampled_depth + float(dive_detection_settings["baseline_adjust"])
        if downsampled_depth.size <= 1:
            first_derivative = np.zeros_like(downsampled_depth, dtype=float)
        else:
            first_derivative = np.gradient(downsampled_depth)
        print(
            "⚠️ Depth decimation fallback used due to short input length "
            f"(len={len(raw_depth)}, downsample_factor={downsample_factor}): {exc}"
        )

    # Ensure pressure/depth values are mostly positive after manual baseline adjustment.
    # Run this check on the baseline-adjusted downsampled signal so sign decision reflects user baseline settings.
    disable_automatic_sign_flipping = bool(dive_detection_settings.get("disable_automatic_sign_flipping", False))
    downsampled_valid = pd.Series(downsampled_depth).dropna()
    if conversion_factor == -1.0:
        # conversion_factor=-1.0 already applied the inversion above; skip the auto-sign
        # check to prevent a double-flip on reruns.
        print("ℹ️ conversion_factor=-1.0 already applied; skipping post-baseline sign check.")
    elif disable_automatic_sign_flipping:
        print("ℹ️ Automatic sign flipping is disabled by config; skipping sign check.")
    elif downsampled_valid.empty:
        print("⚠️ No valid baseline-adjusted depth values to evaluate sign (all NaN).")
    else:
        neg_count = (downsampled_valid < 0).sum()
        pos_count = (downsampled_valid > 0).sum()
        zero_count = (downsampled_valid == 0).sum()
        median_depth = float(downsampled_valid.median())
        print(f"📊 Post-baseline sign check — negative: {neg_count}, positive: {pos_count}, zero: {zero_count}, median: {median_depth:.4f}")

        # Use median rather than neg/pos count: in pool/coastal deployments the animal
        # is near the surface most of the time, so neg and pos counts can be nearly equal
        # even when the signal is fully inverted (surface offset ≈ -1.5 m, dives go more
        # negative). The median of an upright signal is always ≥ 0 (surface = 0); if it
        # is negative the signal must be flipped.
        #
        # A negative median alone is not enough. Shallow deployments sit near the surface
        # with a small negative offset, so an already-correct signal can show a median of
        # a few tens of centimetres below zero — flipping those inverts good data, and the
        # -1.0 then persists to parameter_log.json where it can never self-correct
        # (a recorded -1.0 skips this check entirely on the next run).
        #
        # Require the median to be genuinely deep before trusting it. Observed |median|:
        # shallow coastal orca 0.00-1.00 m vs. deep-diving elephant seal 53-353 m, so 10 m
        # separates the two by a wide margin in both directions.
        if median_depth < -SIGN_FLIP_MIN_MEDIAN_DEPTH_M:
            # Keep all depth representations consistent if sign flip is needed.
            depth_data *= -1
            interpolated_depth_data *= -1
            downsampled_depth *= -1
            first_derivative *= -1
            print(f"🔄 Baseline-adjusted depth has negative median ({median_depth:.4f} m); multiplied by -1 to make it positive.")
            # Persist the inversion so reruns apply it via conversion_factor and never
            # double-flip. Only write if conversion_factor wasn't already -1.0 (i.e. this
            # flip was auto-detected, not already recorded from a prior run).
            if conversion_factor != -1.0:
                param_manager.add_to_config(
                    "conversion_factor", -1.0,
                    section="dive_detection_settings"
                )
                print("💾 Wrote conversion_factor=-1.0 to parameter_log.json (dive_detection_settings) to prevent double-flip on rerun.")
        elif median_depth < 0:
            print(
                f"✅ Baseline-adjusted depth median is negative ({median_depth:.4f} m) but "
                f"shallower than the {SIGN_FLIP_MIN_MEDIAN_DEPTH_M:.0f} m inversion "
                "threshold; treating as a surface offset, not an inverted signal. "
                "Set conversion_factor=-1.0 explicitly if this signal really is inverted."
            )
        else:
            print(f"✅ Baseline-adjusted depth median is non-negative ({median_depth:.4f} m); no sign change applied.")

    # Adjust datetime indexing based on the new downsample rate
    downsample_step = int(depth_fs / dive_detection_settings["downsampled_sampling_rate"])
    if downsample_step <= 0:
        # Data is already at or below target rate, no datetime downsampling needed
        depth_downsampled_datetime = depth_datetime.copy()
    else:
        depth_downsampled_datetime = depth_datetime.iloc[::downsample_step]

    # Ensure indexing does not go out of bounds
    if len(depth_downsampled_datetime) > len(downsampled_depth):
        depth_downsampled_datetime = depth_downsampled_datetime[:len(downsampled_depth)]

    # Print summary of processing
    print(f"✅ Depth processing complete: Downsampled to {dive_detection_settings['downsampled_sampling_rate']} Hz")
    print(f"✅ Baseline adjustment applied: {dive_detection_settings['baseline_adjust']} meters")

    # Detect flat chunks (potential surface intervals)
    flat_chunk_params = {
        "depth": downsampled_depth,
        "datetime_data": depth_downsampled_datetime,
        "first_derivative": first_derivative,
        "threshold": dive_detection_settings["first_deriv_threshold"],
        "min_duration": dive_detection_settings["min_duration"],
        "depth_threshold": dive_detection_settings["depth_threshold"],
        "original_sampling_rate": depth_fs,
        "downsampled_sampling_rate": dive_detection_settings["downsampled_sampling_rate"]
    }
    flat_chunks = detect_flat_chunks(**flat_chunk_params)

    use_flat_chunks = bool(dive_detection_settings.get("use_flat_chunks", True))
    min_flat_chunks_for_zoc = int(dive_detection_settings.get("min_flat_chunks_for_zoc", 10))
    enough_flat_chunks = len(flat_chunks) >= min_flat_chunks_for_zoc

    if use_flat_chunks and enough_flat_chunks:
        # Apply zero offset correction
        zoc_params = {
            "depth": downsampled_depth,
            "temp": temp_data.values if temp_data is not None else None,
            "flat_chunks": flat_chunks
        }
        corrected_depth_temp, corrected_depth_no_temp, depth_correction = apply_zero_offset_correction(**zoc_params)
        pressure = corrected_depth_temp if dive_detection_settings["apply_temp_correction"] else corrected_depth_no_temp
        print(f"✅ Applied ZOC using {len(flat_chunks)} flat chunks (minimum required: {min_flat_chunks_for_zoc}).")
    else:
        # Fallback: use baseline-adjusted depth directly when ZOC is disabled or too few flat chunks are available.
        pressure = downsampled_depth.copy()
        depth_correction = np.zeros_like(downsampled_depth)
        if not use_flat_chunks:
            print("ℹ️ Skipped ZOC because use_flat_chunks is False. Using baseline-adjusted depth only.")
        else:
            print(
                f"ℹ️ Skipped ZOC due to insufficient flat chunks: {len(flat_chunks)} found, "
                f"{min_flat_chunks_for_zoc} required. Using baseline-adjusted depth only."
            )

    # Detect dives using find_segments
    depth_df = pd.DataFrame({
        'datetime': depth_downsampled_datetime,
        'depth': pressure
    })

    print(f'Number of unique depth values detected: {len(np.unique(pressure))}')
    print(depth_df.head())

    dives = find_segments(
        data=depth_df,
        column='depth',
        criteria=lambda x: x > dive_detection_settings['min_depth_threshold'],
        min_duration=dive_detection_settings['dive_duration_threshold'],
    )

    dives_detected = dives is not None and not dives.empty
    if dives_detected:
        nan_mask = depth_data.isna().reindex(depth_downsampled_datetime.index, method='nearest')
        dives['has_nans'] = dives.apply(
            lambda row: nan_mask.loc[
                (depth_downsampled_datetime >= row['start_datetime']) &
                (depth_downsampled_datetime <= row['end_datetime'])
            ].any(),
            axis=1
        )
        dives['short_description'] = dives['has_nans'].apply(lambda x: 'dive-with-nan_start' if x else 'dive_start')
    else:
        print("⚠️ No dives met the detection thresholds. Skipping dive event creation and clearing existing 'dive' events if present.")
        if isinstance(data_pkl.event_data, pd.DataFrame) and 'key' in data_pkl.event_data.columns:
            before = len(data_pkl.event_data)
            data_pkl.event_data = data_pkl.event_data[data_pkl.event_data['key'] != 'dive'].reset_index(drop=True)
            if before != len(data_pkl.event_data):
                print("ℹ️ Removed prior 'dive' events from event_data.")
            data_pkl.event_info = list(data_pkl.event_data['key'].unique()) if not data_pkl.event_data.empty else []

    pressure = enforce_surface_before_after_dives(pressure, depth_downsampled_datetime, dives)

    if dives_detected:
        dives['dive_duration'] = (dives['end_datetime'] - dives['start_datetime']).dt.total_seconds()

    transformation_log = [
        f"downsampled_{dive_detection_settings['downsampled_sampling_rate']}Hz",
        f"smoothed_{dive_detection_settings['smoothing_window']}s",
        f"conversion_factor_{dive_detection_settings['conversion_factor']}",
        f"ZOC_settings__first_deriv_threshold_{dive_detection_settings['first_deriv_threshold']}mps__min_duration_{dive_detection_settings['min_duration']}s__depth_threshold_{dive_detection_settings['depth_threshold']}m",
        f"DIVE_detection_settings__min_depth_threshold_{dive_detection_settings['min_depth_threshold']}m__dive_duration_threshold_{dive_detection_settings['dive_duration_threshold']}s__smoothing_window_{dive_detection_settings['smoothing_window']}"
    ]

    print(f"✅ {len(flat_chunks)} surface intervals detected.")
    print(f"✅ {len(dives)} dives detected.")
    print("📖 Transformation Log:", transformation_log)

    # Append max depth for each dive segment
    if dives_detected:
        dives = append_stats(
            data=depth_df, 
            segment_df=dives, 
            statistics=[("max", "depth")]
        )

        # Generate and update dive events
        data_pkl.event_data = create_state_event(
            state_df=dives,
            key='dive',
            value_column='depth_max',
            start_time_column='start_datetime',
            duration_column='dive_duration', # in seconds
            description='dive_start',
            existing_events=data_pkl.event_data  # Pass existing events for overwrite and concatenation
        )

        # Update event_info with unique keys
        data_pkl.event_info = list(data_pkl.event_data['key'].unique())

    # Step 12: Store derived depth data
    depth_df = pd.DataFrame({"datetime": depth_downsampled_datetime, "depth": pressure})

    derived_from_signals = ["pressure"]
    original_name = "Temp-corrected Depth (m)" if dive_detection_settings["apply_temp_correction"] else "Corrected Depth (m)"

    signal_info = {
        "channels": ["depth"],
        "metadata": {
            "depth": {
                "original_name": original_name,
                "unit": "m",
                "parent_signal": "pressure"
            }
        },
        "derived_from_signals": derived_from_signals + (["temperature"] if dive_detection_settings["apply_temp_correction"] else []),
        "transformation_log": transformation_log + (["temperature_correction"] if dive_detection_settings["apply_temp_correction"] else [])
    }

    data_pkl.signal_data["depth"] = depth_df
    data_pkl.signal_info["depth"] = signal_info

    # Load key time points
    timezone = data_pkl.deployment_info.get('Time Zone', 'UTC')
    settings = param_manager.get_from_config(variable_names=["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],section="settings")
    OVERLAP_START_TIME = pd.Timestamp(settings["overlap_start_time"]).tz_convert(timezone)
    OVERLAP_END_TIME = pd.Timestamp(settings["overlap_end_time"]).tz_convert(timezone)
    ZOOM_WINDOW_START_TIME = pd.Timestamp(settings["zoom_window_start_time"]).tz_convert(timezone)
    ZOOM_WINDOW_END_TIME = pd.Timestamp(settings["zoom_window_end_time"]).tz_convert(timezone)

    # fig = plot_tag_data_interactive(
    #     data_pkl=data_pkl,
    #     signals=['pressure','depth'],
    #     time_range=(depth_downsampled_datetime.min(), depth_downsampled_datetime.max()),
    #     note_annotations={"dive": {"signal": "depth", "symbol": "triangle-down", "color": "blue"}},
    #     state_annotations={"dive": {"signal": "depth", "color": "rgba(150, 150, 150, 0.3)"}},
    #     color_mapping_path=color_mapping_path,
    #     target_sampling_rate=1
    # )
    # fig.show()
else:
    print(f"Skipping step due to missing critical signal: {critical_signal}.")

if data_changed:
    data_pkl.event_data = normalize_event_datetimes(
        getattr(data_pkl, "event_data", None),
        data_pkl.deployment_info.get("Time Zone", "UTC")
    )

param_manager.add_to_config("current_processing_step", "Processing Step 01: Pressure signal calibration complete.")
if data_changed:
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)

save_step_netcdf_if_changed(data_pkl, deployment_folder, deployment_id, 1, changed=data_changed)

print("✅ Data processing complete. Pickle file updated.")
