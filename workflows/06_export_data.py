import argparse
import os
import sys

from datetime import datetime
import glob

# Ensure direct workflow execution resolves the repo-local pyologger package.
WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.io_operations.base_exporter import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
parser.add_argument("--export-csvs", action="store_true", help="Export signal and event data CSVs")
parser.add_argument("--csvs-only", action="store_true", help="Export CSVs only, skip NetCDF export and pkl save")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

# Load data with optional arguments
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')

# Retrieve values from config
variables = ["calm_horizontal_start_time", "calm_horizontal_end_time", 
            "zoom_window_start_time", "zoom_window_end_time", 
            "overlap_start_time", "overlap_end_time",
            "analysis_start_time", "analysis_end_time"]
settings = param_manager.get_from_config(variables, section="settings")

# Assign retrieved values to variables
CALM_HORIZONTAL_START_TIME = settings.get("calm_horizontal_start_time")
CALM_HORIZONTAL_END_TIME = settings.get("calm_horizontal_end_time")
ZOOM_START_TIME = settings.get("zoom_window_start_time")
ZOOM_END_TIME = settings.get("zoom_window_end_time")
OVERLAP_START_TIME = settings.get("overlap_start_time")
OVERLAP_END_TIME = settings.get("overlap_end_time")
ANALYSIS_START_TIME = settings.get("analysis_start_time")
ANALYSIS_END_TIME = settings.get("analysis_end_time")

pkl_size_gb = os.path.getsize(pkl_path) / 1e9 if os.path.exists(pkl_path) else 0

if args.export_csvs or args.csvs_only:
    # Example usage
    signal_data_keys = ['depth','corrected_depth','o2_pressure', 'temperature_ext', 'temperature_int']
    # signal_data_keys = ['pressure','prh', 'odba', 'heart_rate', 'stroke_rate']
    output_frequency = 1  # Hz

    # Calculate sampling frequencies for reference
    # pressure_fs = calculate_sampling_frequency(data_pkl.signal_data['pressure']['datetime'])
    # heart_rate_fs = calculate_sampling_frequency(data_pkl.signal_data['heart_rate']['datetime'])
    # stroke_rate_fs = calculate_sampling_frequency(data_pkl.signal_data['stroke_rate']['datetime'])

    if pkl_size_gb >= 1.0:
        print(f"Skipping signal data CSV export: data.pkl is {pkl_size_gb:.2f} GB (>= 1 GB limit).")
    else:
        # Run the function
        collated_df = collate_data(data_pkl, signal_data_keys, output_frequency)

        # Add another 'datetime' column without the timezone information
        collated_df['datetime'] = collated_df['datetime'].dt.tz_localize(None)
        if OVERLAP_START_TIME and OVERLAP_END_TIME:
            start_time = pd.Timestamp(OVERLAP_START_TIME).tz_localize(None)
            end_time = pd.Timestamp(OVERLAP_END_TIME).tz_localize(None)
            collated_df = collated_df[(collated_df['datetime'] >= start_time) & (collated_df['datetime'] <= end_time)]
            print(f"Cropping signal CSV to overlap window: {start_time} to {end_time}")
        else:
            print("No overlap window defined; exporting full signal data.")

        csv_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_signal_data.csv')
        collated_df.to_csv(csv_file_path, index=False)
        print(f"Signal data saved to {csv_file_path}")

    # Export ECG at native resolution (not resampled — too high frequency for collate_data)
    if 'ecg' in data_pkl.signal_data:
        ecg_df = data_pkl.signal_data['ecg'].copy()
        ecg_df['datetime'] = ecg_df['datetime'].dt.tz_localize(None)
        if OVERLAP_START_TIME and OVERLAP_END_TIME:
            start_time = pd.Timestamp(OVERLAP_START_TIME).tz_localize(None)
            end_time = pd.Timestamp(OVERLAP_END_TIME).tz_localize(None)
            ecg_df = ecg_df[(ecg_df['datetime'] >= start_time) & (ecg_df['datetime'] <= end_time)]
        ecg_csv_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_ecg_data.csv')
        ecg_df.to_csv(ecg_csv_path, index=False)
        print(f"ECG data saved to {ecg_csv_path}")
    else:
        print("No ECG signal found in data; skipping ECG CSV export.")

    # Filter the event data
    filtered_event_data = data_pkl.event_data[
        data_pkl.event_data['key'].isin(['heartbeat_auto_detect_accepted', 'strokebeat_auto_detect_accepted', 'exhalation_breath'])
    ]

    # Keep only the 'datetime' and 'key' columns
    filtered_event_data = filtered_event_data[['datetime', 'key']].copy()
    filtered_event_data['datetime'] = filtered_event_data['datetime'].dt.tz_localize(None)
    if OVERLAP_START_TIME and OVERLAP_END_TIME:
        filtered_event_data = filtered_event_data[(filtered_event_data['datetime'] >= start_time) & (filtered_event_data['datetime'] <= end_time)]
        print(f"Cropping event CSV to overlap window: {start_time} to {end_time}")
    else:
        print("No overlap window defined; exporting full event data.")

    csv_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_event_data.csv')
    filtered_event_data.to_csv(csv_file_path, index=False)
    print(f"Event data saved to {csv_file_path}")

if args.csvs_only:
    print("--csvs-only: skipping NetCDF export and pkl save.")
    exit(0)

# signal_data is deliberately NOT cropped to the analysis window.
#
# This step used to trim every signal to [analysis_start_time, analysis_end_time] and
# then save the trimmed result back to data.pkl. Two problems followed:
#
#  1. The trim was destructive and permanent. Across the oror dataset it removed 28-91%
#     of the record (2023-10-18_oror-002 kept 1.8 min of a 20.4 min deployment), and the
#     raw signal was only recoverable by reprocessing from step 00.
#  2. Step 05 builds its heartbeat chunk grid from the wider `selected` window. On a
#     rerun those chunks no longer lined up with the shortened ECG, so no chunk yielded
#     peaks, and heart_rate/hr_normalized were silently dropped from the export.
#
# The analysis window is still recorded in parameter_log.json for downstream consumers
# that want to restrict their own analysis; it just no longer mutates stored data.
if ANALYSIS_START_TIME and ANALYSIS_END_TIME:
    print(
        f"ℹ️ Analysis window {ANALYSIS_START_TIME} to {ANALYSIS_END_TIME} recorded but "
        "not applied; exporting the full signal record."
    )

# Save the updated data_pkl
with open(pkl_path, 'wb') as f:
    pickle.dump(data_pkl, f)
print(f"Updated data.pkl saved to {pkl_path}")

exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
current_date = datetime.now().strftime("%Y-%m-%d") # Get the current date in YYYY-MM-DD format
netcdf_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_output.nc') # Define the export path
exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format

# Optional cleanup of interim step NetCDF files.
# Keep disabled by default so Snakemake marker dependencies remain satisfied.
cleanup_intermediate = _as_bool(config.get("cleanup_intermediate_step_nc", False))
if cleanup_intermediate:
    nc_files = glob.glob(os.path.join(deployment_folder, 'outputs', '*_step[0-9][0-9].nc'))
    for nc_file in nc_files:
        os.remove(nc_file)
        print(f"Deleted interim processing file: {nc_file}")
else:
    print("Keeping interim step NetCDF files (cleanup_intermediate_step_nc=false).")
