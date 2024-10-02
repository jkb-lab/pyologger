import os
import struct
import pickle
import pytz
import re
import mne
import json
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, date, time
from pyologger.process_data.sampling import *
from pyologger.load_data.metadata import *

class DataReader:
    """A class for handling the reading and processing of deployment data files."""
    
    def __init__(self, deployment_folder_path=None):
        self.deployment_info = {}  # Store selected deployment metadata here
        self.files_info = {}
        self.animal_info = {}
        self.dataset_info = {}
        
        self.logger_data = {}  # Store data by logger ID
        self.logger_info = {}  # Store info (metadata) by logger ID
        self.sensor_data = {}  
        self.sensor_info = {}  # Initialize sensor_info to store sensor metadata
        self.event_data = {}
        self.event_info = {}

        self.derived_data = {} # Holds derived data like pitch, roll, heading, HR, Stroke Rate, track
        self.derived_info = {}
        
        self.files_info['deployment_folder_path'] = deployment_folder_path

    def read_files(self, metadata, save_csv=True, save_parq=True, save_edf=False, 
                   custom_mapping_path=None, edf_filename_template=None, selected_sensors=None, 
                   selected_channels=None, edf_save_from='sensor_data', save_netcdf=False):
        """
        Reads and processes the deployment data files based on manufacturer.

        Parameters:
        - metadata: Metadata object containing deployment information.
        - save_csv: Whether to save the processed data as CSV files.
        - save_parq: Whether to save the processed data as Parquet files.
        - save_edf: Whether to export the data to an EDF file.
        - edf_filename_template: The file path template where the EDF files will be saved.
        - selected_sensors: List of sensor names to include in the EDF file. If None, include all sensors.
        - selected_channels: Dictionary specifying which channels to include for each sensor.
                            If None, include all channels for the selected sensors.
        - edf_save_from: Specifies whether to export EDF files from 'sensor_data' or 'data'.
        - save_netcdf: Whether to save the processed data and metadata into a NetCDF file.
        """
        if not self.files_info['deployment_folder_path']:
            print("No deployment folder set. Please use check_deployment_folder first.")
            return

        print(f"Step 2: Deployment folder initialized at: {self.files_info['deployment_folder_path']}")

        # Save databases
        logger_db = metadata.get_metadata("logger_DB")
        recording_db = metadata.get_metadata("recording_DB")
        animal_db = metadata.get_metadata("animal_DB")
        dataset_db = metadata.get_metadata("dataset_DB")

        # Step 1: Fill in self.animal_info
        self.animal_info = self.get_animal_info(animal_db)

        print(dataset_db)
        # Step 2: Fill in self.dataset_info
        self.dataset_info = self.get_dataset_info(dataset_db)

        event_data = self.import_notes()
        if event_data is not None:
            self.event_data = event_data

        logger_files = self.organize_files_by_logger_id(logger_db)

        # Check if output files are already processed
        if self.check_outputs_folder(logger_files.keys()):
            print("All necessary files are already processed. Skipping further processing.")

            # Load the DataReader object from the pickle file
            pkl_path = os.path.join(self.files_info['deployment_folder_path'], 'outputs', 'data.pkl')
            with open(pkl_path, 'rb') as file:
                data_pkl = pickle.load(file)

            # Update self with loaded data
            self.logger_data = data_pkl.logger_data
            self.logger_info = data_pkl.logger_info
            self.sensor_data = data_pkl.sensor_data
            self.sensor_info = data_pkl.sensor_info

            # If export_edf is True, proceed to export the data to an EDF file
            if save_edf and edf_filename_template:
                if edf_save_from == 'sensor_data':
                    self.export_to_edf(edf_filename_template, selected_sensors=selected_sensors, selected_channels=selected_channels)
                elif edf_save_from == 'data':
                    self.export_to_edf_from_data(edf_filename_template)
            return

        # Continue processing if files are not already present
        for logger_id, files in logger_files.items():
            manufacturer = logger_db.loc[logger_db['Logger ID'] == logger_id, 'Manufacturer'].values[0]

            # Save logger metadata
            self.logger_info[logger_id]['logger_metadata'] = logger_db.loc[logger_db['Logger ID'] == logger_id].to_dict('records')[0]
            # Find matching rows in recording_db
            deployment_id = self.deployment_info['Deployment ID'].split('_')[0]
            recording_matches = recording_db[(recording_db['Recording ID'].str.startswith(deployment_id)) &
                                            (recording_db['Recording ID'].str.contains(logger_id))]

            if len(recording_matches) == 0:
                print(f"No matching recording found for Logger ID {logger_id} in Deployment ID {deployment_id}.")
            elif len(recording_matches) > 1:
                raise ValueError(f"Multiple recordings found for Logger ID {logger_id} in Deployment ID {deployment_id}. This should not happen.")
            else:
                # Save the matching row of recording_db in self.logger_info[logger_id]['recording_info']
                self.logger_info[logger_id]['recording_info'] = recording_matches.to_dict('records')[0]

            if manufacturer == "CATS":
                processor = CATSManufacturer(self, logger_id, manufacturer="CATS", custom_mapping_path=custom_mapping_path)
            elif manufacturer == "UFI":
                processor = UFIManufacturer(self, logger_id, manufacturer="UFI", custom_mapping_path=custom_mapping_path)
            else:
                print(f"Manufacturer {manufacturer} is not supported.")
                continue

            result = processor.process_files(files)

            final_df, column_metadata, datetime_metadata, sensor_groups, sensor_info = result

            if final_df is not None and 'datetime' in final_df.columns:
                self.logger_info[logger_id]['datetime_metadata'] = datetime_metadata
                self.logger_info[logger_id]['channelinfo'] = column_metadata
                self.save_data(final_df, logger_id, f"{logger_id}.csv", save_csv, save_parq)
                print(f"Files saved for logger {logger_id}.")
            else:
                print("Issue with file saving.")

        self.save_datareader_object()

        # If export_edf is True, export the data to an EDF file
        if save_edf and edf_filename_template:
            if edf_save_from == 'sensor_data':
                self.export_to_edf(edf_filename_template, selected_sensors=selected_sensors, selected_channels=selected_channels)
            elif edf_save_from == 'data':
                self.export_to_edf_from_data(edf_filename_template)

        # If save_netcdf is True, save the data to a NetCDF file
        if save_netcdf:
            netcdf_filename = os.path.join(self.files_info['deployment_folder_path'], 'outputs', 'deployment_data.nc')
            self.save_to_netcdf(netcdf_filename)

    def save_to_netcdf(self, filepath):
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

        def create_data_array(data, datetime_coord, variables, name):
            """Creates an xarray DataArray with appropriate dimensions and coordinates."""
            if data.ndim == 1:
                dims = [f"{name}_samples"]
                coords = {f"{name}_samples": datetime_coord}
            else:
                dims = [f"{name}_samples", f"{name}_variables"]
                coords = {f"{name}_samples": datetime_coord, f"{name}_variables": variables}
            
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
        for sensor_name, df in self.sensor_data.items():
            sensor_data = df.copy()
            # Saving datetime as timezone-aware
            datetime_coord = pd.to_datetime(sensor_data['datetime'])
            sensor_data = sensor_data.drop(columns=['datetime'])
            variables = [col for col in sensor_data.columns]
            print(variables)
            data_array = convert_to_compatible_array(sensor_data)
            var_name = f'sensor_data_{sensor_name}'
            ds[var_name] = create_data_array(data_array, datetime_coord, variables, sensor_name)
            set_variables_attr(ds, var_name, variables)

        for logger_id, df in self.logger_data.items():
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
            ds[var_name] = create_data_array(data_array, datetime_coord, variables, logger_id)
            set_variables_attr(ds, var_name, variables)

        for derived_name, df in self.derived_data.items():
            derived_data = df.copy()
            # Saving datetime as timezone-aware
            datetime_coord = pd.to_datetime(derived_data['datetime'])
            derived_data = derived_data.drop(columns=['datetime'])
            variables = [col for col in derived_data.columns]
            data_array = convert_to_compatible_array(derived_data)
            var_name = f'derived_data_{derived_name}'
            ds[var_name] = create_data_array(data_array, datetime_coord, variables, derived_name)
            set_variables_attr(ds, var_name, variables)

        if isinstance(self.event_data, pd.DataFrame):
            event_data = self.event_data.copy()
            # Saving datetime as timezone-aware
            datetime_coord = pd.to_datetime(event_data['datetime_utc'])
            vars_to_keep = ["type", "key", "short_description", "long_description"]
            event_data = event_data[vars_to_keep]
            variables = [col for col in event_data.columns]
            data_array = convert_to_compatible_array(event_data)
            var_name = 'event_data'
            ds[var_name] = create_data_array(data_array, datetime_coord, variables, "event")
            set_variables_attr(ds, var_name, variables)

        # Flatten and add global attributes
        flatten_dict('deployment_info', self.deployment_info)
        flatten_dict('files_info', self.files_info)
        flatten_dict('animal_info', self.animal_info)
        flatten_dict('dataset_info', self.dataset_info)

        for logger_id, logger_info in self.logger_info.items():
            flatten_dict(f'logger_info_{logger_id}', logger_info)

        for sensor_name, sensor_info in self.sensor_info.items():
            flatten_dict(f'sensor_info_{sensor_name}', sensor_info)

        # Store the Dataset as a NetCDF file
        ds.to_netcdf(filepath)
        print(f"NetCDF file saved at {filepath}")

    def collect_file_info(self):
        """Collect information about the files in the outputs folder and store in self.files_info."""
        output_folder = os.path.join(self.files_info['deployment_folder_path'], 'outputs')
        if not os.path.exists(output_folder):
            print(f"Outputs folder does not exist at {output_folder}.")
            return

        file_info = {
            'filenames': [],
            'file_types': set(),  # Use a set to avoid duplicates
            'read_in_dates': []
        }

        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                file_info['filenames'].append(file)
                file_extension = os.path.splitext(file)[1].lower()
                file_info['file_types'].add(file_extension)

                # Get the file's last modification time as the read-in date
                read_in_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                file_info['read_in_dates'].append(read_in_date)

        # Convert file types set to a sorted list
        file_info['file_types'] = sorted(list(file_info['file_types']))

        self.files_info['outputs'] = file_info

        print(f"Collected file information: {self.files_info['outputs']}")

    def get_animal_info(self, animal_db):
        """Retrieve and store information about the animal(s) in the current deployment."""
        deployment_folder = self.files_info['deployment_folder_path']
        if not deployment_folder:
            print("Deployment folder not set.")
            return {}

        # Extract Animal IDs from the deployment folder path
        matched_animals = []
        for animal_id in animal_db['Animal ID'].unique():
            if animal_id in deployment_folder:
                matched_animals.append(animal_id)

        if not matched_animals:
            print("No matching Animal IDs found in the deployment folder path.")
            return {}

        # Retrieve corresponding rows from animal_db
        animal_info = animal_db[animal_db['Animal ID'].isin(matched_animals)]
        if len(animal_info) == 1:
            return animal_info.to_dict('records')[0]
        else:
            return animal_info.to_dict('records')

    def get_dataset_info(self, dataset_db):
        """Retrieve and store dataset information based on the matched Animal IDs."""
        if not self.animal_info:
            print("No animal information available.")
            return {}

        # If animal_info is a list (multiple animals), flatten the list of Animal IDs
        if isinstance(self.animal_info, list):
            animal_ids = [animal['Animal ID'] for animal in self.animal_info]
        else:
            animal_ids = [self.animal_info['Animal ID']]

        # Retrieve datasets associated with these Animal IDs, handling None/NaN values
        dataset_info = dataset_db[dataset_db['Animal ID'].notna() &
                                  dataset_db['Animal ID'].apply(lambda x: any(aid in x for aid in animal_ids) if x else False)]

        if dataset_info.empty:
            print("No matching datasets found for the animal(s).")
            return {}

        # Return the dataset info as a list of records
        return dataset_info.to_dict('records')

    def process_datetime(self, df, time_zone=None):
        """Processes datetime columns in the DataFrame and calculates sampling frequency."""
        metadata = {'datetime_created_from': None, 'fs': None}

        if 'datetime' in df.columns:
            print("'datetime' column found.")
            metadata['datetime_created_from'] = 'datetime'
            if time_zone and df['datetime'].dt.tz is None:
                print(f"Localizing datetime using timezone {time_zone}.")
                tz = pytz.timezone(time_zone)
                df['datetime'] = df['datetime'].dt.tz_localize(tz)

            df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')
            df['time_unix_ms'] = df['datetime_utc'].astype(np.int64) // 10**6
            df['sec_diff'] = df['datetime_utc'].diff().dt.total_seconds()
            if len(df) > 1:
                mean_diff = df['sec_diff'].mean()
                sampling_frequency = 1 / mean_diff if mean_diff else None
                max_timediff = np.max(df['sec_diff'])
                formatted_fs = f"{sampling_frequency:.5g}"
                print(f"Sampling frequency: {formatted_fs} Hz with a maximum time difference of {max_timediff} seconds")
                metadata['fs'] = formatted_fs
            else:
                print("Insufficient data points to calculate sampling frequency.")
                metadata['fs'] = None

            return df, metadata

        elif 'time' in df.columns and 'date' in df.columns:
            print("'datetime' column not found. Combining 'date' and 'time' columns.")
            dates = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
            times = pd.to_timedelta(df['time'].astype(str))
            df['datetime'] = dates + times
            metadata['datetime_created_from'] = 'date and time'

        elif 'time_local' in df.columns and 'date_local' in df.columns:
            print("'datetime' and 'date/time' columns not found. Combining 'date_local' and 'time_local' columns.")
            dates = pd.to_datetime(df['date_local'], format='%d.%m.%Y', errors='coerce')
            times = pd.to_timedelta(df['time_local'].astype(str))
            df['datetime'] = dates + times
            metadata['datetime_created_from'] = 'date_local and time_local'

        else:
            print("No suitable columns found to create a 'datetime' column.")
            return df, metadata

        if time_zone and df['datetime'].dt.tz is None:
            print(f"Localizing datetime using timezone {time_zone}.")
            tz = pytz.timezone(time_zone)
            df['datetime'] = df['datetime'].dt.tz_localize(tz)

        print("Converting to UTC and Unix.")
        df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')
        df['time_unix_ms'] = df['datetime_utc'].astype(np.int64) // 10**6
        df['sec_diff'] = df['datetime_utc'].diff().dt.total_seconds()
        if len(df) > 1:
            mean_diff = df['sec_diff'].mean()
            sampling_frequency = 1 / mean_diff if mean_diff else None
            max_timediff = np.max(df['sec_diff'])
            formatted_fs = f"{sampling_frequency:.5g}"
            print(f"Sampling frequency: {formatted_fs} Hz with a maximum time difference of {max_timediff} seconds")
            metadata['fs'] = formatted_fs
        else:
            print("Insufficient data points to calculate sampling frequency.")
            metadata['fs'] = None

        return df, metadata

    def organize_files_by_logger_id(self, logger_db):
        """Organizes files by logger ID based on the deployment folder."""
        logger_ids = set(logger_db['Logger ID'])
        logger_files = {logger_id: [] for logger_id in logger_ids}

        for file in os.listdir(self.files_info['deployment_folder_path']):
            for logger_id in logger_ids:
                if logger_id in file:
                    logger_files[logger_id].append(file)
                    break

        for logger_id in logger_files:
            logger_files[logger_id].sort()

        loggers_with_files = {logger_id: files for logger_id, files in logger_files.items() if files}
        loggers_without_files = [logger_id for logger_id in logger_ids if logger_id not in loggers_with_files]

        print("Loggers with files:", ", ".join(loggers_with_files.keys()))
        print("Loggers without files:", ", ".join(loggers_without_files))

        for logger_id in loggers_with_files:
            self.logger_data[logger_id] = None
            self.logger_info[logger_id] = {"channelinfo": {}}

        return loggers_with_files

    def save_datareader_object(self):
        """Saves the DataReader object as a pickle file."""
        pickle_filename = os.path.join(self.files_info['deployment_folder_path'], 'outputs', 'data.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"DataReader object successfully saved to {pickle_filename}.")

    def check_deployment_folder(self, deployment_db, data_dir):
        """Checks the deployment folder and allows the user to select a deployment."""
        print("Step 1: Displaying deployments to help you select one.")
        print(deployment_db[['Deployment ID', 'Notes']])

        selected_index = int(input("Enter the index of the deployment you want to work with: "))

        if 0 <= selected_index < len(deployment_db):
            selected_deployment = deployment_db.iloc[selected_index]
            self.deployment_info = selected_deployment  # Save selected deployment to self
            print(f"Step 1: You selected the deployment: {selected_deployment['Deployment ID']}")
            print(f"Description: {selected_deployment['Notes']}")
        else:
            print("Invalid index selected.")
            return None

        deployment_folder = os.path.join(data_dir, selected_deployment['Deployment ID'])
        print(f"Step 2: Deployment folder path: {deployment_folder}")

        if os.path.exists(deployment_folder):
            print(f"Deployment folder found: {deployment_folder}")
        else:
            print(f"Folder {deployment_folder} not found. Searching for folders with a similar name...")
            possible_folders = [folder for folder in os.listdir(data_dir) 
                                if folder.startswith(selected_deployment['Deployment ID'])]

            if len(possible_folders) == 1:
                deployment_folder = os.path.join(data_dir, possible_folders[0])
                print(f"Using the found folder: {deployment_folder}")
            elif len(possible_folders) > 1:
                print("Multiple matching folders found. Please select one:")
                for i, folder in enumerate(possible_folders):
                    print(f"{i}: {folder}")
                selected_index = int(input("Enter the index of the folder you want to use: "))
                if 0 <= selected_index < len(possible_folders):
                    deployment_folder = os.path.join(data_dir, possible_folders[selected_index])
                    print(f"Using the selected folder: {deployment_folder}")
                else:
                    print("Invalid selection. Aborting.")
                    return None
            else:
                print("Error: Folder not found.")
                return None

        self.files_info['deployment_folder_path'] = deployment_folder
        print(f"Ready to process deployment folder: {self.files_info['deployment_folder_path']}")
        return self.files_info['deployment_folder_path']

    def import_notes(self):
        """Imports and processes notes associated with the selected deployment."""
        if self.deployment_info is None or self.deployment_info.empty:
            print("Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None
        print("import_notes")
        notes_filename = f"{self.deployment_info['Deployment ID']}_00_Notes.xlsx"
        rec_date = self.deployment_info['Recording Date']
        start_time = self.deployment_info.get('Start Time', "00:00:00")
        time_zone = self.deployment_info.get('Time Zone')

        if not time_zone:
            print(f"No time zone information found for the selected deployment.")
            return None

        notes_filepath = os.path.join(self.files_info['deployment_folder_path'], notes_filename)

        if not os.path.exists(notes_filepath):
            print(f"Notes file {notes_filename} not found in {self.files_info['deployment_folder_path']}.")
            return None

        try:
            event_df = pd.read_excel(notes_filepath)
        except Exception as e:
            print(f"Error reading {notes_filename}: {e}")
            return None

        event_df, datetime_metadata = self.process_datetime(event_df, time_zone=time_zone)
        
        if event_df['datetime'].isna().any():
            print(f"WARNING: Some timestamps could not be parsed.")
            return event_df

        event_df = event_df.sort_values(by='datetime').reset_index(drop=True)
        print(f"Notes imported, processed, and sorted chronologically from {notes_filename}.")
        return event_df

    def check_outputs_folder(self, logger_ids):
        """Checks if the processed data files for the given loggers already exist in the outputs folder."""
        output_folder = os.path.join(self.files_info['deployment_folder_path'], 'outputs')
        if not os.path.exists(output_folder):
            print("Outputs folder does not exist. Processing required.")
            return False

        existing_files = os.listdir(output_folder)
        print(f"Existing files in output folder: {existing_files}")

        for logger_id in logger_ids:
            matching_files = [filename for filename in existing_files if logger_id in filename]
            if not matching_files:
                print(f"No files found for logger ID {logger_id} in the output folder. Processing required.")
                return False
            else:
                print(f"Files found for logger ID {logger_id}: {matching_files}")

        print("All necessary files are already processed and available in the outputs folder.")
        return True

    def read_csv(self, csv_path):
        """Reads a CSV file with multiple encoding attempts."""
        encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                print(f"Attempting to read {csv_path} with encoding {encoding}")
                return pd.read_csv(csv_path, encoding=encoding)
            except UnicodeDecodeError as e:
                print(f"Error reading {csv_path} with encoding {encoding}: {e}")
        raise UnicodeDecodeError(f"Failed to read {csv_path} with available encodings.")

    def save_data(self, data, logger_id, filename, save_csv=True, save_parq=False):
        """Saves the processed data to disk in CSV and/or Parquet format."""
        self.logger_data[logger_id] = {}
        output_folder = os.path.join(self.files_info['deployment_folder_path'], 'outputs')
        os.makedirs(output_folder, exist_ok=True)

        self.logger_data[logger_id] = data

        if save_csv:
            csv_path = os.path.join(output_folder, f"{filename}")
            data.to_csv(csv_path, index=False)
            print(f"Data for {logger_id} successfully saved as CSV to: {csv_path}")

        if save_parq:
            if hasattr(data, 'attrs'):
                data.attrs = {key: str(value) for key, value in data.attrs.items()}
            parq_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.parquet")
            data.to_parquet(parq_path, index=False)
            print(f"Data for {logger_id} successfully saved as Parquet to: {parq_path}")

        if not save_csv and not save_parq:
            print(f"Data for {logger_id} saved to attribute {filename} but not to CSV or Parquet.")

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

class BaseManufacturer:
    """Base class for handling manufacturer-specific processing."""

    def __init__(self, data_reader, logger_id, manufacturer, custom_mapping_path):
        self.logger_id = logger_id
        self.logger_manufacturer = manufacturer
        self.data_reader = data_reader
        self.expected_frequencies = {}  # This will hold the expected frequencies parsed from the .txt file

        # Load the custom JSON mapping for column names if deployment folder is provided
        if custom_mapping_path:
            self.load_custom_mapping(custom_mapping_path)

    def load_custom_mapping(self, custom_mapping_path):
        """Loads custom JSON mapping for column names."""
        try:
            with open(custom_mapping_path, 'r') as json_file:
                self.column_mapping = json.load(json_file)
                print(f"Custom column mapping loaded from {custom_mapping_path}")
                
                # Check if the mapping contains expected keys
                if not isinstance(self.column_mapping, dict):
                    raise ValueError("Column mapping is not a dictionary.")
                if 'CATS' not in self.column_mapping and 'UFI' not in self.column_mapping:
                    raise ValueError("Expected keys 'CATS' or 'UFI' are missing in the column mapping.")
                print("Column mapping verified successfully.")
        except FileNotFoundError:
            print(f"Custom mapping file not found at {custom_mapping_path}. Proceeding without it.")
            self.column_mapping = None
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading or verifying JSON from {custom_mapping_path}: {e}")
            self.column_mapping = None

    def rename_columns(self, df, logger_id, manufacturer):
        """Renames columns based on the provided column mapping."""
        column_metadata = {}
        new_columns = {}
        seen_names = set()

        # Ensure you are using the correct sub-dictionary for column mapping
        mapping_sub_dict = self.column_mapping.get(manufacturer, {})

        for original_name in df.columns:
            clean_name = original_name.strip().lower().replace(" ", "_").replace(".", "")  # Normalize column names

            # Extract and store units
            unit = None
            if "[" in clean_name and "]" in clean_name:
                name, square_unit = clean_name.split("[", 1)
                square_unit = square_unit.replace("]", "").strip().lower()
                name = name.strip("_")
                clean_name = name
                unit = square_unit
            if "(" in clean_name and ")" in clean_name:
                name, round_unit = clean_name.split("(", 1)
                round_unit = round_unit.replace(")", "").strip().lower()
                clean_name = f"{name.strip('_')}_{round_unit}"
                unit = round_unit
            
            # Ensure local and UTC times remain distinct
            if "local" in original_name.lower() and "local" not in clean_name:
                clean_name = f"{clean_name}_local"
            elif "utc" in original_name.lower() and "utc" not in clean_name:
                clean_name = f"{clean_name}_utc"

            # Check for duplicates and handle them
            if clean_name in seen_names:
                if unit:
                    clean_name = f"{clean_name}_{unit}"
                else:
                    clean_name = f"{clean_name}_dup"  # Add a suffix to distinguish duplicates
            seen_names.add(clean_name)

            # Apply custom mapping from the correct sub-dictionary
            mapping_info = mapping_sub_dict.get(clean_name, None)
            if mapping_info is None:
                print(f"Warning: {clean_name} not found in column mapping. Skipping.")
                continue  # Skip columns not found in the mapping

            mapped_name = mapping_info.get("column_name", clean_name)
            sensor_type = mapping_info.get("sensor_type", "extra").strip().lower()  # Use 'sensor_type' instead of 'sensor'

            print(f"Original name: {original_name}, Clean name: {clean_name}, Mapped name: {mapped_name}, Sensor type: {sensor_type}")

            column_metadata[mapped_name] = {
                "original_name": original_name,
                "unit": unit or "unknown",
                "sensor": sensor_type  # Updated to use 'sensor_type'
            }
            new_columns[original_name] = mapped_name

        # Rename the columns in the DataFrame
        df.rename(columns=new_columns, inplace=True)

        return df, column_metadata

    def map_data_to_sensors(self, df, logger_id, column_metadata):
        """Groups data columns to sensors and downsamples based on expected frequencies."""
        sensor_groups = {}
        sensor_info = {}

        for sensor_name in set(v['sensor'].strip().lower() for v in column_metadata.values()):
            if sensor_name == 'extra':
                continue  # Skip 'extra' sensor type

            # Skip already processed sensors
            if sensor_name in self.data_reader.sensor_data:
                print(f"Sensor '{sensor_name}' has already been processed. Skipping reprocessing.")
                continue

            # Group columns by sensor
            sensor_cols = [col for col, meta in column_metadata.items() if meta['sensor'].strip().lower() == sensor_name]
            sensor_df = df[['datetime'] + sensor_cols].copy()

            # Determine the data type of the sensor columns
            data_type = sensor_df[sensor_cols].dtypes[0]  # Assuming all sensor columns have the same dtype
            data_type_str = str(data_type)

            # Standardized metadata collection
            start_time = sensor_df['datetime'].iloc[0]
            end_time = sensor_df['datetime'].iloc[-1]
            max_value = sensor_df[sensor_cols].max().max()
            min_value = sensor_df[sensor_cols].min().min()
            mean_value = sensor_df[sensor_cols].mean().mean()

            # Get the original unit from the column metadata
            original_units = list({column_metadata[col]['unit'] for col in sensor_cols})

            # Get sampling frequency for each sensor (logger-specific methods or default)
            expected_frequency = self.expected_frequencies.get(sensor_name)
            if not expected_frequency and sensor_name == 'ecg':
                # For UFI ECG, use the overall frequency (e.g., determined earlier)
                expected_frequency = float(self.data_reader.logger_info[logger_id]['datetime_metadata']['fs'])

            if expected_frequency:
                # Calculate the actual sampling frequency based on the 'datetime' column
                current_frequency = round(1 / df['datetime'].diff().dt.total_seconds().mean())
                # Calculate the decimation factor
                decimation_factor = max(1, int(round(current_frequency / expected_frequency)))

                # Print the downsampling information
                if decimation_factor > 1:
                    print(f"Downsampling {sensor_name} data by {decimation_factor} X from {current_frequency:.2f} Hz to {expected_frequency:.2f}Hz.")
                    # Apply downsampling
                    sensor_df = sensor_df.iloc[::decimation_factor]
                else:
                    print(f"No downsampling needed for {sensor_name}. Expected frequency is close to the actual frequency {current_frequency:.2f} Hz.")
            
            # Store the processed data and standardized metadata
            self.data_reader.sensor_data[sensor_name] = sensor_df
            self.data_reader.sensor_info[sensor_name] = {
                'channels': sensor_cols,
                'metadata': {col: column_metadata[col] for col in sensor_cols},
                'sensor_start_datetime': start_time,
                'sensor_end_datetime': end_time,
                'max_value': max_value,
                'min_value': min_value,
                'mean_value': mean_value,
                'data_type': data_type_str,
                'original_units': original_units,
                'sampling_frequency': expected_frequency,
                'logger_id': self.logger_id,
                'logger_manufacturer': self.logger_manufacturer,
                'processing_step': 'Raw data uploaded',
                'last_updated': datetime.now().astimezone(pytz.timezone(self.data_reader.deployment_info['Time Zone'])),
                'details': 'Initial, raw sensor-specific data and metadata loaded.',
            }

        # Print final mapping and downsampling results
        for sensor_name, df in self.data_reader.sensor_data.items():
            print(f"Sensor '{sensor_name}' data processed and stored with shape {df.shape}.")

        return sensor_groups, sensor_info

    def parse_txt_for_intervals(self, txt_file_path):
        """Parses the .txt file to extract expected sampling frequencies for sensors."""
        print(f"Attempting to parse intervals from {txt_file_path}")

        try:
            with open(txt_file_path, 'r') as file:
                content = file.read()

            # Extract sensor information from the 'activated sensors' section
            activated_sensors_section = re.search(r'\[activated sensors\](.*?)\n\n', content, re.DOTALL)
            if activated_sensors_section:
                activated_sensors_content = activated_sensors_section.group(1)

                # Find all sensors' names and their corresponding intervals
                sensor_info = re.findall(r'(\d{2})_name=(.*?)\n.*?\1_interval=(\d+)', activated_sensors_content, re.DOTALL)
                if not sensor_info:
                    print("No sensor information found in the file. Please check the file format.")
                    return
                
                print(f"Sensor info read from txt: {sensor_info}")

                # Use the correct sub-dictionary for column mapping, assuming you're processing data for a CATS logger
                mapping_sub_dict = self.column_mapping.get(self.logger_manufacturer, {})

                for sensor_id, sensor_name, interval in sensor_info:
                    normalized_sensor_name = sensor_name.strip().lower()  # Normalize the sensor name

                    # Reverse lookup in the mapping sub-dictionary to find the matching manufacturer_sensor_name
                    for clean_name, mapping in mapping_sub_dict.items():
                        # Get the manufacturer sensor name associated with this column
                        manufacturer_sensor_name = mapping['manufacturer_sensor_name'].strip().lower()  # Use 'manufacturer_sensor_name'

                        # Match based on normalized sensor name and manufacturer sensor name
                        if manufacturer_sensor_name == normalized_sensor_name:
                            frequency = int(interval)
                            # Store the frequency using the sensor_type (normalized)
                            sensor_type = mapping['sensor_type'].strip().lower()
                            self.expected_frequencies[sensor_type] = frequency  
                            print(f"Sensor '{sensor_type}' found matching sensor name in config: '{sensor_name}'. Expected sampling frequency: {frequency} Hz")
                            break
                    else:
                        print(f"Sensor name '{sensor_name}' not found in column mapping. Ignoring this sensor.")

            else:
                print("No 'activated sensors' section found in the file.")

        except Exception as e:
            print(f"Failed to parse {txt_file_path} due to: {e}")


    def process_files(self, files):
        """Process files in the subclass. This should be overridden."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def concatenate_and_save_csvs(self, csv_files):
        """Base method for concatenating and saving CSVs."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class CATSManufacturer(BaseManufacturer):
    """CATS-specific processing."""

    def process_files(self, files):
        """Process CATS files, specifically handling .txt, ignoring .ubx, .ubc, and .bin files."""
        # Filter out .ubx, .ubc, and .bin files
        files = [f for f in files if not f.endswith(('.ubc', '.bin', '.ubx'))]
        
        # Parse the .txt file for expected intervals
        txt_file = next((f for f in files if f.endswith('.txt')), None)
        if txt_file:
            print(f"Parsing {txt_file} for expected sensor intervals.")
            self.parse_txt_for_intervals(os.path.join(self.data_reader.files_info['deployment_folder_path'], txt_file))

        if not files:
            print(f"No valid files found for {self.logger_manufacturer} logger.")
            return None, None, None, None  # Return four None values

        # Remove the .txt files from the list after processing them
        files = [f for f in files if not f.endswith('.txt')]

        # Concatenate the remaining files into one DataFrame
        final_df = self.concatenate_and_save_csvs(files)

        # Rename columns
        final_df, column_metadata = self.rename_columns(final_df, self.logger_id, self.logger_manufacturer)
        
        # Process datetime and return metadata
        final_df, datetime_metadata = self.data_reader.process_datetime(final_df, time_zone=self.data_reader.deployment_info['Time Zone'])
        self.data_reader.logger_info[self.logger_id]['datetime_metadata'] = datetime_metadata

        # Map data to sensors and return sensor information
        sensor_groups, sensor_info = self.map_data_to_sensors(final_df, self.logger_id, column_metadata)

        return final_df, column_metadata, datetime_metadata, sensor_groups, sensor_info

    def concatenate_and_save_csvs(self, csv_files):
        """Concatenates multiple CSV files into one DataFrame."""
        dfs = []
        for file in csv_files:
            file_path = os.path.join(self.data_reader.files_info['deployment_folder_path'], file)
            try:
                data = self.data_reader.read_csv(file_path)
                dfs.append(data)
                print(f"{self.logger_manufacturer} file: {file} - Successfully processed.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        if len(dfs) > 1:
            concatenated_df = pd.concat(dfs, ignore_index=True)
        else:
            concatenated_df = dfs[0]

        return concatenated_df

    def print_txt_content(self, txt_file):
        """Prints the content of a .txt file."""
        file_path = os.path.join(self.data_reader.files_info['deployment_folder_path'], txt_file)
        with open(file_path, 'r') as file:
            print(file.read())

class UFIManufacturer(BaseManufacturer):
    """UFI-specific processing."""

    def process_files(self, files):
        """Process UFI files, specifically looking for .ube files."""
        # Find the first .ube file in the files list
        ube_file = next((f for f in files if f.endswith('.ube')), None)
        
        if ube_file:
            print(f"Processing .ube file: {ube_file}")
            final_df = self.process_ube_file(ube_file)

            # Rename columns
            final_df, column_metadata = self.rename_columns(final_df, self.logger_id, self.logger_manufacturer)
            
            # Process datetime and return metadata
            final_df, datetime_metadata = self.data_reader.process_datetime(final_df, time_zone=self.data_reader.deployment_info['Time Zone'])
            self.data_reader.logger_info[self.logger_id]['datetime_metadata'] = datetime_metadata
            
            # Map data to sensors and return sensor information
            sensor_groups, sensor_info = self.map_data_to_sensors(final_df, self.logger_id, column_metadata)

            return final_df, column_metadata, datetime_metadata, sensor_groups, sensor_info
        else:
            print(f"No .ube file found for UFI logger.")
            return None, None, None, None  # Return four None values

    def process_ube_file(self, ube_file):
        """Processes a UBE file and extracts data."""
        file_path = os.path.join(self.data_reader.files_info['deployment_folder_path'], ube_file)
        print(f"Processing UBE file: {file_path}")

        try:
            with open(file_path, 'rb') as file:
                ube_raw = file.read()

            # Parse the download timestamp
            dl_time_str = ube_raw[0:32].decode('utf-8').strip()
            print(f"Parsed download timestamp string: '{dl_time_str}'")
            try:
                dl_time = datetime.strptime(dl_time_str, "%m-%d-%Y, %H:%M:%S")
            except ValueError as e:
                print(f"Error parsing timestamp: '{dl_time_str}' - {e}")
                return pd.DataFrame(), {}

            # Extract the record start time components
            mdhms = struct.unpack('BBBBB', ube_raw[32:37])
            now = datetime.now()
            record_start = datetime(now.year, mdhms[0], mdhms[1], mdhms[2], mdhms[3], mdhms[4])

            timezone = self.data_reader.deployment_info.get('Time Zone')
            if timezone:
                tz = pytz.timezone(timezone)
                record_start = tz.localize(record_start)
                print(f"Recording start time (localized): {record_start}")

            rec_date = pd.to_datetime(self.data_reader.deployment_info['Recording Date']).date()
            if record_start.date() != rec_date:
                print(f"Error: Recording start date {record_start.date()} does not match Recording Date {rec_date}.")
                return pd.DataFrame(), {}

            data_raw = ube_raw[40:]

            ecg_channel = 0x20
            ecg_data = []

            for i in range(0, len(data_raw), 2):
                channel = data_raw[i]
                value = data_raw[i + 1]
                if (channel & 0xF0) == ecg_channel:
                    ecg_value = (channel & 0x0F) << 8 | value
                    ecg_data.append(ecg_value)

            print(f"Total ECG data points: {len(ecg_data)}")

            if len(ecg_data) == 0:
                print("No ECG data extracted. Exiting.")
                return pd.DataFrame(), {}

            # Generate the datetime column for the ECG data
            ecg_time = [record_start + timedelta(seconds=i/100) for i in range(len(ecg_data))]

            result = pd.DataFrame({
                'datetime': ecg_time,
                'ecg': ecg_data
            })
            result.attrs['created'] = dl_time

            return result

        except Exception as e:
            print(f"Error processing UBE file {ube_file}: {e}")
            return pd.DataFrame(), {}
