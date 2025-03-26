import argparse

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.io_operations.base_exporter import *
from datetime import datetime
import glob

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Load data with optional arguments
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment(data_dir)

pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')

# Example usage
sensor_data_keys = ['pressure']
derived_data_keys = ['prh', 'odba', 'heart_rate', 'stroke_rate']
output_frequency = 5  # Hz

# Calculate sampling frequencies for reference
pressure_fs = calculate_sampling_frequency(data_pkl.sensor_data['pressure']['datetime'])
heart_rate_fs = calculate_sampling_frequency(data_pkl.derived_data['heart_rate']['datetime'])
stroke_rate_fs = calculate_sampling_frequency(data_pkl.derived_data['stroke_rate']['datetime'])

# Run the function
collated_df = collate_data(data_pkl, sensor_data_keys, derived_data_keys, output_frequency)

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

# Add another 'datetime' column without the timezone information
collated_df['datetime'] = collated_df['datetime'].dt.tz_localize(None)
start_time = pd.Timestamp(OVERLAP_START_TIME).tz_localize(None)
end_time = pd.Timestamp(OVERLAP_END_TIME).tz_localize(None)
cropped_collated_df = collated_df[(collated_df['datetime'] >= start_time) & (collated_df['datetime'] <= end_time)]

cropped_collated_df
# Save the filtered event data to a CSV file in the deployment folder
csv_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_signal_data.csv')
cropped_collated_df.to_csv(csv_file_path, index=False)
print(f"Filtered event data saved to {csv_file_path}")

# Filter the event data
filtered_event_data = data_pkl.event_data[
    data_pkl.event_data['key'].isin(['heartbeat_auto_detect_accepted', 'strokebeat_auto_detect_accepted', 'exhalation_breath'])
]

# Keep only the 'datetime' and 'key' columns
filtered_event_data = filtered_event_data[['datetime', 'key']]

# Add another 'datetime' column without the timezone information
filtered_event_data['datetime'] = filtered_event_data['datetime'].dt.tz_localize(None)
filtered_event_data = filtered_event_data[(filtered_event_data['datetime'] >= start_time) & (filtered_event_data['datetime'] <= end_time)]
filtered_event_data
# Save the filtered event data to a CSV file in the deployment folder
csv_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_event_data.csv')
filtered_event_data.to_csv(csv_file_path, index=False)
print(f"Filtered event data saved to {csv_file_path}")

# Crop all derived dataframes to analysis time window if defined
if ANALYSIS_START_TIME and ANALYSIS_END_TIME:
    analysis_start = pd.Timestamp(ANALYSIS_START_TIME)
    analysis_end = pd.Timestamp(ANALYSIS_END_TIME)
    for key, df in data_pkl.derived_data.items():
        if 'datetime' in df.columns:
            data_pkl.derived_data[key] = df[
                (df['datetime'] >= analysis_start) &
                (df['datetime'] <= analysis_end)
            ]
    print(f"Cropped derived_data to analysis window: {analysis_start} to {analysis_end}")
else:
    print("No analysis window defined; skipping cropping of derived_data.")

# Save the updated data_pkl with cropped derived_data
with open(pkl_path, 'wb') as f:
    pickle.dump(data_pkl, f)
print(f"Updated data.pkl saved to {pkl_path}")

exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
current_date = datetime.now().strftime("%Y-%m-%d") # Get the current date in YYYY-MM-DD format
netcdf_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_output.nc') # Define the export path
exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format

# Delete all files in the folder that end in _stepXX.nc
nc_files = glob.glob(os.path.join(deployment_folder, 'outputs', '*_step[0-9][0-9].nc'))
for nc_file in nc_files:
    os.remove(nc_file)
    print(f"Deleted interim processing file: {nc_file}")
