import xarray as xr
import pandas as pd
import numpy as np
import pytz
import os
import mne
from datetime import datetime, date, time
from pyologger.process_data.sampling import *

class BaseExporter:
    """Handles exporting data from the DataReader object."""

    def __init__(self, datareader):
        self.datareader = datareader

    def save_data(self, data, logger_id, filename, save_csv=True, save_parq=False):
        """Saves the processed data to disk in CSV and/or Parquet format."""
        self.datareader.logger_data[logger_id] = {}
        output_folder = os.path.join(self.datareader.files_info['deployment_folder_path'], 'outputs')
        os.makedirs(output_folder, exist_ok=True)

        self.datareader.logger_data[logger_id] = data

        if save_csv:
            csv_path = os.path.join(output_folder, filename)
            data.to_csv(csv_path, index=False)
            print(f"Data for {logger_id} saved as CSV to: {csv_path}")

        if save_parq:
            parq_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.parquet")
            data.to_parquet(parq_path, index=False)
            print(f"Data for {logger_id} saved as Parquet to: {parq_path}")

    def save_to_netcdf(self, datareader, filepath):
            """Saves the current state of the DataReader object to a NetCDF file."""
            def convert_to_compatible_array(df):
                """Convert DataFrame columns to compatible numpy arrays."""
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Handle datetime objects by converting them to strings
                        if isinstance(df[col].iloc[0], (datetime, date, time)):
                            df[col] = df[col].astype(str)
                        elif pd.api.types.is_datetime64_any_dtype(df[col]):
                            df[col] = pd.to_datetime(df[col])
                        else:
                            # Attempt to convert to float, if fails convert to string
                            try:
                                df[col] = df[col].astype(float)
                            except ValueError:
                                df[col] = df[col].astype(str)
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col])

                # Check the number of columns in the DataFrame
                if df.shape[1] == 1:
                    # If there is only one column, return a flat array
                    return df.iloc[:, 0].to_numpy()
                else:
                    # If there are multiple columns, return nested arrays
                    return df.apply(pd.to_numeric, errors='ignore').to_numpy()

            def serialize_value(value):
                """Helper function to serialize values to be JSON-compatible."""
                if isinstance(value, (datetime, date, time)):
                    return value.isoformat()
                elif isinstance(value, (list, tuple)):
                    return [serialize_value(item) for item in value]
                elif isinstance(value, dict):
                    return {k: serialize_value(v) for k, v in value.items()}
                else:
                    return value

            def flatten_dict(prefix, d):
                """Flattens a dictionary and adds it to dataset attributes."""
                for key, value in d.items():
                    flattened_key = f"{prefix}_{key}"
                    try:
                        serialized_value = serialize_value(value)
                        if isinstance(serialized_value, (str, int, float, list, tuple, np.ndarray)):
                            ds.attrs[flattened_key] = serialized_value
                        else:
                            raise TypeError("Invalid value type for NetCDF serialization")
                    except (TypeError, ValueError):
                        ds.attrs[flattened_key] = "Invalid entry"
                        print(f"Invalid entry recognized and placed in {flattened_key}")

            def create_coords(ndim, datetime_coord, variables, name):
                """Creates an xarray DataArray with appropriate dimensions and coordinates."""
                if ndim == 1:
                    dims = [f"{name}_samples"]
                    coords = {f"{name}_samples": datetime_coord}
                else:
                    dims = [f"{name}_samples", f"{name}_variables"]
                    coords = {f"{name}_samples": datetime_coord, f"{name}_variables": variables}

                return dims, coords
                
            def create_data_array(data, dims, coords):
                """Creates an xarray DataArray with appropriate dimensions and coordinates."""
                return xr.DataArray(data, dims=dims, coords=coords)

            def set_variables_attr(ds, var_name, variables):
                """Sets the 'variables' or 'variable' attribute based on the type of 'variables'."""
                if isinstance(variables, list):
                    ds[var_name].attrs['variables'] = variables
                else:
                    ds[var_name].attrs['variable'] = variables

            # Create an empty xarray dataset
            ds = xr.Dataset()

            # Flatten the dictionaries into xarray DataArrays
            for sensor_name, df in self.datareader.sensor_data.items():
                sensor_data = df.copy()
                # Saving datetime as timezone-aware
                datetime_coord = pd.to_datetime(sensor_data['datetime'])
                sensor_data = sensor_data.drop(columns=['datetime'])
                variables = [col for col in sensor_data.columns]
                data_array = convert_to_compatible_array(sensor_data)
                var_name = f'sensor_data_{sensor_name}'
                ndim = data_array.ndim
                dims, coords = create_coords(ndim, datetime_coord, variables, sensor_name)
                ds[var_name] = create_data_array(data_array, dims, coords)
                set_variables_attr(ds, var_name, variables)

            for logger_id, df in self.datareader.logger_data.items():
                # Remove specified columns
                logger_data = df.copy()
                columns_to_remove = ['date_utc', 'time_utc', 'date', 'time']
                logger_data = logger_data.drop(columns=[col for col in columns_to_remove if col in logger_data.columns])
                # Saving datetime as timezone-aware
                datetime_coord = pd.to_datetime(logger_data['datetime'])
                logger_data = logger_data.drop(columns=['datetime'])
                # Remove string type columns
                logger_data = logger_data.select_dtypes(exclude=['object'])
                variables = [col for col in logger_data.columns]
                data_array = convert_to_compatible_array(logger_data)
                var_name = f'logger_data_{logger_id}'
                ndim = data_array.ndim
                dims, coords = create_coords(ndim, datetime_coord, variables, logger_id)
                ds[var_name] = create_data_array(data_array, dims, coords)
                set_variables_attr(ds, var_name, variables)

            for derived_name, df in self.datareader.derived_data.items():
                derived_data = df.copy()
                # Saving datetime as timezone-aware
                datetime_coord = pd.to_datetime(derived_data['datetime'])
                derived_data = derived_data.drop(columns=['datetime'])
                variables = [col for col in derived_data.columns]
                data_array = convert_to_compatible_array(derived_data)
                var_name = f'derived_data_{derived_name}'
                ndim = data_array.ndim
                dims, coords = create_coords(ndim, datetime_coord, variables, derived_name)
                ds[var_name] = create_data_array(data_array, dims, coords)
                set_variables_attr(ds, var_name, variables)

            columns_to_keep = ["type", "key", "value", "duration", "short_description", "long_description"]
            
            if isinstance(self.datareader.event_data, pd.DataFrame):
                for var in columns_to_keep:
                    event_data = self.datareader.event_data.copy()
                    datetime_coord = pd.to_datetime(event_data['datetime'])
                    event_data = event_data[[var]]
                    data_array = convert_to_compatible_array(event_data)
                    var_name = f'event_data_{var}'
                    ndim = data_array.ndim
                    if var == columns_to_keep[0]:
                        dims, coords = create_coords(ndim, datetime_coord, variables, 'event_data')
                    ds[var_name] = create_data_array(data_array, dims, coords)
                    set_variables_attr(ds, var_name, var)

            # Flatten and add global attributes
            flatten_dict('deployment_info', self.datareader.deployment_info)
            flatten_dict('files_info', self.datareader.files_info)
            flatten_dict('animal_info', self.datareader.animal_info)
            flatten_dict('dataset_info', self.datareader.dataset_info[0] if self.datareader.dataset_info else {})

            for logger_id, logger_info in self.datareader.logger_info.items():
                flatten_dict(f'logger_info_{logger_id}', logger_info)

            for sensor_name, sensor_info in self.datareader.sensor_info.items():
                flatten_dict(f'sensor_info_{sensor_name}', sensor_info)

            # Store the Dataset as a NetCDF file
            ds.to_netcdf(filepath)
            print(f"NetCDF file saved at {filepath}")

    def create_mne_raw_object(self, sensor, selected_channels=None):
            """
            Creates an MNE Raw object from the data of a specific sensor.

            Parameters:
            - sensor: The sensor name to include in the Raw object.
            - selected_channels: List of channels to include for the sensor. If None, include all channels.

            Returns:
            - MNE Raw object containing the sensor data.
            """
            sensor_df = self.sensor_data[sensor]
            ch_names = self.sensor_info[sensor]['channels']
            
            # If no specific channels are selected, use all available channels for this sensor
            if selected_channels is None:
                selected_channels = ch_names
            
            # Extract the relevant data for the selected channels
            selected_data = sensor_df[selected_channels].values.T  # Transpose to match MNE shape requirements
            
            # Create MNE info dictionary
            info = mne.create_info(
                ch_names=selected_channels,
                sfreq=self.sensor_info[sensor]['sampling_frequency'],  # Assume uniform sampling frequency for the sensor
                ch_types='misc'  # Adjust based on actual sensor types if known
            )
            
            # Convert the start datetime string to a UTC datetime object
            start_datetime = self.sensor_info[sensor]['sensor_start_datetime']
            if isinstance(start_datetime, pd.Timestamp):
                start_datetime_local = start_datetime.to_pydatetime()
                start_datetime_utc = start_datetime_local.astimezone(pytz.UTC)
            elif isinstance(start_datetime, str):
                start_datetime_local = pd.to_datetime(start_datetime)
                start_datetime_utc = start_datetime_local.tz_convert('UTC')
            else:
                raise ValueError(f"Unexpected format for sensor_start_datetime: {start_datetime}")
            
            # Convert to (seconds, microseconds) tuple
            meas_date = (int(start_datetime_utc.timestamp()), int((start_datetime_utc.timestamp() % 1) * 1e6))

            # Set the measurement date using the converted UTC datetime
            raw = mne.io.RawArray(selected_data, info)
            raw.set_meas_date(meas_date)
            
            # Add custom metadata to the MNE info object
            for i, ch_name in enumerate(selected_channels):
                ch_metadata = self.sensor_info[sensor]['metadata'][ch_name]
                
                # Store original unit and other details in the channel description
                description = f"{ch_metadata['original_name']} ({ch_metadata['unit']})"
                info['chs'][i]['desc'] = description  # Use the description field for storing extra information

            # Concatenate other deployment data into a plaintext string
            deployment_info = "\n".join([f"{key}: {value}" for key, value in self.deployment_info.items()])
            info['description'] = f"Sensor: {sensor}\nDeployment Data:\n{deployment_info}"
            
            return raw

    def export_to_edf_from_data(self, edf_filename_template):
        """
        Exports a single EDF file by concatenating data from all loggers stored in `self.logger_data`.

        Parameters:
        - edf_filename_template: Template string for the EDF filename.
                                The string should contain `{sensor}` to be replaced with 'ALL' for this function.
        """
        logger_data_info = {}

        # Step 1: Extract start time, end time, and sampling frequency for each logger
        for logger_id, df in self.logger_data.items():
            if not isinstance(df, pd.DataFrame):
                print(f"Logger {logger_id} does not contain a valid DataFrame. Skipping.")
                continue

            if 'datetime' not in df.columns:
                print(f"Logger {logger_id} does not have a 'datetime' column. Skipping.")
                continue

            start_time = df['datetime'].iloc[0]
            end_time = df['datetime'].iloc[-1]
            sampling_frequency = round(1 / df['datetime'].diff().dt.total_seconds().mean())

            logger_data_info[logger_id] = {
                'start_time': start_time,
                'end_time': end_time,
                'sampling_frequency': sampling_frequency
            }

            print(f"Logger {logger_id}: start_time={start_time}, end_time={end_time}, sampling_frequency={sampling_frequency} Hz")

        if not logger_data_info:
            print("No valid logger data found. Exiting.")
            return

        # Step 2: Determine the latest start time, earliest end time, and highest sampling frequency
        latest_start_time = max(info['start_time'] for info in logger_data_info.values())
        earliest_end_time = min(info['end_time'] for info in logger_data_info.values())
        highest_sampling_frequency = max(info['sampling_frequency'] for info in logger_data_info.values())

        print(f"Latest start time: {latest_start_time}")
        print(f"Earliest end time: {earliest_end_time}")
        print(f"Highest sampling frequency: {highest_sampling_frequency} Hz")

        # Step 3: Initialize an empty DataFrame for concatenation
        concatenated_df = pd.DataFrame()

        # Step 4: Crop dataframes, upsample as necessary, and concatenate
        for logger_id, df in self.logger_data.items():
            if not isinstance(df, pd.DataFrame):
                continue

            # Crop dataframe
            df_cropped = df[(df['datetime'] >= latest_start_time) & (df['datetime'] <= earliest_end_time)]
            print(f"Logger {logger_id}: Cropped data from {len(df)} rows to {len(df_cropped)} rows.")

            # Determine upsampling factor
            upsampling_factor = highest_sampling_frequency / logger_data_info[logger_id]['sampling_frequency']

            if upsampling_factor > 1:
                original_length = len(df_cropped)
                df_cropped = df_cropped.set_index('datetime')

                # Upsample each sensor column that is not "extra"
                for column in df_cropped.columns:
                    sensor_info = None

                    # Search for the sensor type in `sensor_info`
                    for sensor_name, sensor_details in self.sensor_info.items():
                        if column in sensor_details['channels']:
                            sensor_info = sensor_details
                            break

                    if not sensor_info:
                        continue

                    sensor_type = sensor_info['metadata'][column]['sensor']
                    if sensor_type != 'extra':
                        print(f"Upsampling column {column} from logger {logger_id} by factor {upsampling_factor}.")
                        df_cropped[column] = upsample(df_cropped[column].values, int(upsampling_factor), original_length)

                df_cropped = df_cropped.reset_index()

            # Remove "extra" sensor columns and append to the concatenated DataFrame
            columns_to_keep = []
            for column in df_cropped.columns:
                sensor_info = None

                # Search for the sensor type in `sensor_info`
                for sensor_name, sensor_details in self.sensor_info.items():
                    if column in sensor_details['channels']:
                        sensor_info = sensor_details
                        break

                if not sensor_info:
                    continue

                sensor_type = sensor_info['metadata'][column]['sensor']
                if sensor_type != 'extra':
                    columns_to_keep.append(column)

            df_filtered = df_cropped[columns_to_keep]

            # Step 5: Replace NaNs with zeros
            df_filtered = df_filtered.fillna(100000)

            # Concatenate to the overall DataFrame
            concatenated_df = pd.concat([concatenated_df, df_filtered], axis=1)

            print(f"Logger {logger_id}: Concatenated DataFrame shape now {concatenated_df.shape}.")

        # Step 6: Ensure no NaNs in the final concatenated DataFrame
        concatenated_df = concatenated_df.fillna(0)

        # Step 7: Create MNE info and Raw object from concatenated data
        ch_names = concatenated_df.columns.tolist()
        sfreq = highest_sampling_frequency

        # Check if there are any channels to process
        if len(ch_names) == 0:
            print("No valid channels found to export. Exiting.")
            return

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='misc')  # Adjust ch_types as necessary

        # Convert datetime to (seconds, microseconds) tuple for the latest start time
        meas_date = (int(latest_start_time.timestamp()), int((latest_start_time.timestamp() % 1) * 1e6))

        # Create MNE RawArray
        data = concatenated_df.values.T
        raw = mne.io.RawArray(data, info)
        raw.set_meas_date(meas_date)

        # Step 8: Define the EDF filename and save the EDF file
        edf_filename = edf_filename_template.format(sensor='ALL')

        print(f"Saving EDF file as {edf_filename} with shape {data.shape}.")

        # Ensure that data is within the physical range EDF expects
        raw.export(edf_filename, fmt='edf')

        print(f"EDF file saved as {edf_filename}")

    def export_to_edf(self, filename_template, selected_sensors=None, selected_channels=None):
        """
        High-level method to export the current DataReader object's sensors to separate EDF files.

        Parameters:
        - filename_template: A template string for the filename where '{sensor}' will be replaced by the sensor name.
        - selected_sensors: List of sensor names to export to EDF files. If None, include all sensors.
        - selected_channels: Dictionary specifying which channels to include for each sensor (e.g., {'accelerometer': ['ax', 'ay']}).
                            If None, include all channels for the selected sensors.
        """
        # If no specific sensors are selected, use all available sensors
        if selected_sensors is None:
            selected_sensors = list(self.sensor_data.keys())

        # Iterate through each sensor and export to an EDF file
        for sensor in selected_sensors:
            if sensor not in self.sensor_data:
                print(f"Sensor {sensor} not found in sensor_data. Skipping.")
                continue
            
            # Determine which channels to include for the current sensor
            if selected_channels and sensor in selected_channels:
                channels_to_include = selected_channels[sensor]
            else:
                channels_to_include = self.sensor_info[sensor]['channels']

            # Create the MNE Raw object for the current sensor
            raw = self.create_mne_raw_object(sensor, selected_channels=channels_to_include)

            # Define the EDF filename for the current sensor, replacing '{sensor}' in the template
            edf_filename = filename_template.format(sensor=sensor)

            # Save the Raw object as an EDF file
            raw.export(edf_filename, fmt='edf')
            
            print(f"EDF file for {sensor} saved as {edf_filename}")