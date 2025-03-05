import os
import pickle
import pytz
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta, date, time
from pyologger.process_data.sampling import *
from pyologger.load_data.metadata import *
from pyologger.utils.json_manager import ConfigManager
from pyologger.io_operations import *
from pyologger.io_operations.base_exporter import *
from pyologger.io_operations.cats_importer import *
from pyologger.io_operations.ufi_importer import *
from pyologger.io_operations.wc_importer import *
from pyologger.io_operations.ll_importer import *

class DataReader:
    """A class for handling the reading and processing of deployment data files."""
    
    def __init__(self, dataset_folder: str, deployment_id: str, data_subfolder: str = None):
        """
        Initializes DataReader.

        Parameters:
        - dataset_folder: Path to the dataset folder containing deployments.
        - deployment_id: Name of the deployment folder inside the dataset folder.
        - data_subfolder: (Optional) Subfolder inside the deployment folder where data is stored.
        """
        self.deployment_id = deployment_id
        self.deployment_folder = os.path.join(dataset_folder, deployment_id)
        self.data_folder = os.path.join(self.deployment_folder, data_subfolder) if data_subfolder else self.deployment_folder

        self.deployment_info = {}  # Store selected deployment metadata here
        self.files_info = {'deployment_folder_path': self.deployment_folder}
        self.animal_info = {}
        self.dataset_info = {}

        self.logger_data = {}  # Store data by logger ID
        self.logger_info = {}  # Store metadata by logger ID
        self.sensor_data = {}  
        self.sensor_info = {}  # Store sensor metadata
        self.event_data = {}
        self.event_info = {}

        self.derived_data = {}  # Holds derived data like pitch, roll, heading, HR, stroke rate, track
        self.derived_info = {}

        self.exporter = BaseExporter(self)  # Initialize exporter

        print(f"DataReader initialized with deployment folder: {self.deployment_folder}")
        print(f"Using data folder: {self.data_folder}")

    def read_files(self, metadata, save_csv=True, save_parq=True, save_edf=False, 
                custom_mapping_path=None, edf_filename_template=None, selected_sensors=None, 
                selected_channels=None, edf_save_from='sensor_data', save_netcdf=False):
        """
        Reads and processes deployment data files from the specified folder.

        Parameters:
        - metadata: Metadata object containing deployment information.
        - save_csv, save_parq, save_edf, save_netcdf: Flags for saving output formats.
        - custom_mapping_path: Optional path to a custom mapping file.
        - edf_filename_template: File path template for EDF file saving.
        - selected_sensors, selected_channels: Options to filter sensors/channels for EDF export.
        - edf_save_from: Source data for EDF export ('sensor_data' or 'data').
        """
        if not os.path.exists(self.data_folder):
            print(f"❌ Error: The specified data folder '{self.data_folder}' does not exist.")
            return

        print(f"🔄 Reading files from: {self.data_folder}")

        # Load metadata databases
        logger_db = metadata.get_metadata("logger_DB")
        recording_db = metadata.get_metadata("recording_DB")
        dataset_db = metadata.get_metadata("dataset_DB")

        # Restore deployment_info from metadata
        deployment_db = metadata.get_metadata("deployment_DB")
        if self.deployment_id in deployment_db["Deployment ID"].values:
            self.deployment_info = deployment_db.loc[deployment_db["Deployment ID"] == self.deployment_id].to_dict("records")[0]
            print(f"✅ Deployment info loaded for {self.deployment_id}.")
        else:
            print(f"⚠ Deployment ID '{self.deployment_id}' not found in deployment database.")

        # Step 1: Fill in self.dataset_info
        self.dataset_info = self.get_dataset_info(dataset_db)

        # Step 2: Import event notes from data subfolder
        self.event_data = self.import_notes()

        # Step 3: Organize files by logger ID
        logger_files = self.organize_files_by_logger_id(logger_db)

        # Step 4: Check if outputs exist to skip unnecessary processing
        if self.check_outputs_folder(logger_files.keys()):
            print("✅ All necessary files are already processed. Skipping further processing.")
            
            # Attempt to load existing processed data
            pkl_path = os.path.join(self.deployment_folder, "outputs", "data.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as file:
                    data_pkl = pickle.load(file)

                # Restore data from the saved pickle file
                self.logger_data = data_pkl.logger_data
                self.logger_info = data_pkl.logger_info
                self.sensor_data = data_pkl.sensor_data
                self.sensor_info = data_pkl.sensor_info
                self.derived_data = data_pkl.derived_data
                self.derived_info = data_pkl.derived_info

                print(f"📦 Loaded previously processed data from {pkl_path}.")
                
                # If export options are selected, export from loaded data
                if save_edf and edf_filename_template:
                    if edf_save_from == 'sensor_data':
                        self.exporter.export_to_edf(edf_filename_template, selected_sensors=selected_sensors, selected_channels=selected_channels)
                    elif edf_save_from == 'data':
                        self.exporter.export_to_edf_from_data(edf_filename_template)

                return

        # Step 5: Process each logger if not already processed
        for logger_id, files in logger_files.items():
            manufacturer = logger_db.loc[logger_db['Logger ID'] == logger_id, 'Manufacturer'].values[0]

            # Save logger metadata
            self.logger_info[logger_id] = {
                'logger_metadata': logger_db.loc[logger_db['Logger ID'] == logger_id].to_dict('records')[0]
            }

            # Step 6: Match recordings to deployment
            deployment_id_prefix = self.deployment_id.split('_')[0]
            recording_matches = recording_db[
                (recording_db['Recording ID'].str.startswith(deployment_id_prefix)) &
                (recording_db['Recording ID'].str.contains(logger_id))
            ]

            if len(recording_matches) == 0:
                print(f"⚠ No matching recording found for Logger ID {logger_id} in Deployment ID {deployment_id_prefix}.")
            elif len(recording_matches) > 1:
                raise ValueError(f"❌ Multiple recordings found for Logger ID {logger_id} in Deployment ID {deployment_id_prefix}. This should not happen.")
            else:
                self.logger_info[logger_id]['recording_info'] = recording_matches.to_dict('records')[0]

            # Step 7: Select manufacturer-specific processor
            processor_classes = {"CATS": CATSImporter, "UFI": UFIImporter, "LL": LLImporter, "WC": WCImporter}
            processor_class = processor_classes.get(manufacturer)

            if processor_class:
                processor_instance = processor_class(self, logger_id, manufacturer=manufacturer, custom_mapping_path=custom_mapping_path)
                result = processor_instance.process_files(files)

                if result:
                    final_df, column_metadata, datetime_metadata, sensor_groups, sensor_info = result

                    if final_df is not None and 'datetime' in final_df.columns:
                        # Store extracted metadata
                        self.logger_info[logger_id]['datetime_metadata'] = datetime_metadata
                        self.logger_info[logger_id]['channelinfo'] = column_metadata

                        # Save data in requested formats
                        self.exporter.save_data(final_df, logger_id, f"{logger_id}.csv", save_csv, save_parq)

                        print(f"✅ Processed and saved files for logger {logger_id}.")
                    else:
                        print(f"⚠ Issue with file saving for logger {logger_id}.")
            else:
                print(f"⚠ Manufacturer {manufacturer} is not supported.")

        # Step 8: Save the DataReader object to a pickle file
        self.save_datareader_object()

        # Step 9: Optionally export to EDF and NetCDF
        if save_edf and edf_filename_template:
            if edf_save_from == 'sensor_data':
                self.exporter.export_to_edf(edf_filename_template, selected_sensors=selected_sensors, selected_channels=selected_channels)
            elif edf_save_from == 'data':
                self.exporter.export_to_edf_from_data(edf_filename_template)

        if save_netcdf:
            netcdf_filename = os.path.join(self.deployment_folder, 'outputs', f'{self.deployment_id}_00_processed.nc')
            self.exporter.save_to_netcdf(self, netcdf_filename)
            print(f"📊 Saved deployment data to NetCDF: {netcdf_filename}")

    def check_outputs_folder(self, logger_ids):
        """
        Checks if the processed data files for the given loggers already exist.

        Returns:
            bool: True if all necessary files are found, False otherwise.
        """
        output_folder = os.path.join(self.deployment_folder, "outputs")

        if not os.path.exists(output_folder):
            print(f"❌ Outputs folder '{output_folder}' does not exist. Processing required.")
            return False

        existing_files = set(os.listdir(output_folder))
        print(f"📂 Checking output folder: {output_folder}")
        print(f"📝 Existing files: {existing_files}")

        # Define expected file formats
        required_extensions = {".csv", ".parquet", ".edf", ".nc"}

        missing_loggers = []
        for logger_id in logger_ids:
            matching_files = [file for file in existing_files if logger_id in file]
            if not matching_files:
                print(f"⚠ No files found for Logger ID {logger_id} in {output_folder}. Processing required.")
                missing_loggers.append(logger_id)
            else:
                # Check if all expected file types exist
                found_extensions = {os.path.splitext(file)[1].lower() for file in matching_files}
                missing_extensions = required_extensions - found_extensions

                if missing_extensions:
                    print(f"⚠ Logger ID {logger_id} is missing expected files: {missing_extensions}")
                    missing_loggers.append(logger_id)
                else:
                    print(f"✅ All expected files found for Logger ID {logger_id}: {matching_files}")

        if missing_loggers:
            print(f"🚨 Missing processed files for loggers: {', '.join(missing_loggers)}. Processing required.")
            return False

        print("✅ All necessary files are already processed and available.")
        return True


    def import_notes(self):
        """Imports and processes notes associated with the selected deployment."""
        
        if not self.deployment_info:
            print("❌ Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None

        notes_filename = f"{self.deployment_info['Deployment ID']}_00_Notes.xlsx"
        time_zone = self.deployment_info.get("Time Zone")

        if not time_zone:
            print(f"⚠ No time zone information found for deployment {self.deployment_id}. Defaulting to UTC.")
            time_zone = "UTC"  # Fallback to UTC

        # Look in the data subfolder (self.data_folder) instead of deployment folder
        notes_filepath = os.path.join(self.data_folder, notes_filename)

        if not os.path.exists(notes_filepath):
            print(f"❌ Notes file '{notes_filename}' not found in {self.data_folder}. Skipping import.")
            return None

        try:
            event_df = pd.read_excel(notes_filepath)
            print(f"📂 Successfully loaded notes file: {notes_filepath}")
        except Exception as e:
            print(f"❌ Error reading {notes_filename}: {e}")
            return None

        # Process timestamps
        event_df, datetime_metadata = self.process_datetime(event_df, time_zone=time_zone)

        if event_df["datetime"].isna().any():
            print(f"⚠ WARNING: Some timestamps could not be parsed correctly.")
        
        # Ensure the dataframe is sorted by time
        event_df = event_df.sort_values(by="datetime").reset_index(drop=True)

        # Ensure 'duration' column exists and is properly set
        if "duration" not in event_df.columns:
            event_df["duration"] = 0

        event_df.loc[event_df["type"] == "point", "duration"] = 0  # Set duration to 0 for 'point' events

        print(f"✅ Notes imported and processed from {notes_filename}. Sorted chronologically.")
        return event_df


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
        """Retrieve dataset information based on the deployment folder path."""
        
        # Extract parent directory name from deployment folder path
        dataset_folder = os.path.basename(os.path.dirname(self.deployment_folder))

        if not dataset_folder:
            print("❌ Dataset folder could not be determined from deployment path.")
            return {}

        print(f"🔍 Searching for dataset matching folder: {dataset_folder}")

        # Find matching entry in dataset_db based on the 'Folder' column
        dataset_info = dataset_db[dataset_db["Folder"].astype(str) == dataset_folder]

        if dataset_info.empty:
            print(f"⚠ No matching dataset found for folder: {dataset_folder}")
            return {}

        print(f"✅ Found {len(dataset_info)} matching dataset(s) for folder: {dataset_folder}")

        # Return dataset info as a dictionary
        return dataset_info.to_dict("records")[0] if len(dataset_info) == 1 else dataset_info.to_dict("records")

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
                metadata['fs'] = int(formatted_fs)
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
        datetime_utc = df['datetime'].dt.tz_convert('UTC')
        sec_diff = datetime_utc.diff().dt.total_seconds()
        if len(df) > 1:
            mean_diff = sec_diff.mean()
            sampling_frequency = 1 / mean_diff if mean_diff else None
            max_timediff = np.max(sec_diff)
            formatted_fs = f"{sampling_frequency:.5g}"
            print(f"Sampling frequency: {formatted_fs} Hz with a maximum time difference of {max_timediff} seconds")
            metadata['fs'] = int(sampling_frequency)
        else:
            print("Insufficient data points to calculate sampling frequency.")
            metadata['fs'] = None

        return df, metadata

    def organize_files_by_logger_id(self, logger_db):
        """Organizes files by logger ID based on the deployment folder."""
        logger_ids = set(logger_db['Logger ID'])
        logger_files = {logger_id: [] for logger_id in logger_ids}

        for file in os.listdir(self.data_folder):
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