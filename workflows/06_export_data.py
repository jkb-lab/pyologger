import argparse

# Import necessary pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.io_operations.base_exporter import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Zero Offset Correction - Calibrate Pressure Sensor")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load environment variables
config, data_dir, color_mapping_path, channel_mapping_path = load_configuration()

# Load data with optional arguments
if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, config_manager = select_and_load_deployment(
        data_dir, dataset_id=args.dataset, deployment_id=args.deployment
    )
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, config_manager = select_and_load_deployment(data_dir)

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

# Add another 'datetime' column without the timezone information
collated_df['datetime'] = collated_df['datetime'].dt.tz_localize(None)
start_time = pd.to_datetime("2024-01-16 10:02:00")
end_time = pd.to_datetime("2024-01-16 10:10:00")
cropped_collated_df = collated_df[(collated_df['datetime'] >= start_time) & (collated_df['datetime'] <= end_time)]

cropped_collated_df
# Save the filtered event data to a CSV file in the deployment folder
csv_file_path = os.path.join(deployment_folder, f'{deployment_id}_signal_data.csv')
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
csv_file_path = os.path.join(deployment_folder, f'{deployment_id}_event_data.csv')
filtered_event_data.to_csv(csv_file_path, index=False)
print(f"Filtered event data saved to {csv_file_path}")