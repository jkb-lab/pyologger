import os
import pickle
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
from pyologger.process_data.sampling import *
from pyologger.load_data.metadata import *
from pyologger.utils.config_manager import ConfigManager
from pyologger.io_operations import *
from pyologger.io_operations.base_exporter import *
from pyologger.io_operations.cats_importer import *
from pyologger.io_operations.ufi_importer import *
from pyologger.io_operations.wc_importer import *
from pyologger.io_operations.ll_importer import *

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
        
        self.exporter = BaseExporter(self)  # Initialize the exporter with self
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
        - selected_channels: Dictionary specifying which channels to include for each sensor. If None, include all channels for the selected sensors.
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
                    self.exporter.export_to_edf(edf_filename_template, selected_sensors=selected_sensors, selected_channels=selected_channels)
                elif edf_save_from == 'data':
                    self.exporter.export_to_edf_from_data(edf_filename_template)
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
                processor = CATSImporter(self, logger_id, manufacturer="CATS", custom_mapping_path=custom_mapping_path)
            elif manufacturer == "UFI":
                processor = UFIImporter(self, logger_id, manufacturer="UFI", custom_mapping_path=custom_mapping_path)
            elif manufacturer == "LL":
                processor = LLImporter(self, logger_id, manufacturer="LL", custom_mapping_path=custom_mapping_path)
            elif manufacturer == "WC":
                processor = WCImporter(self, logger_id, manufacturer="WC", custom_mapping_path=custom_mapping_path)
            else:
                print(f"Manufacturer {manufacturer} is not supported.")
                continue

            result = processor.process_files(files)

            final_df, column_metadata, datetime_metadata, sensor_groups, sensor_info = result

            if final_df is not None and 'datetime' in final_df.columns:
                self.logger_info[logger_id]['datetime_metadata'] = datetime_metadata
                self.logger_info[logger_id]['channelinfo'] = column_metadata
                self.exporter.save_data(final_df, logger_id, f"{logger_id}.csv", save_csv, save_parq)
                print(f"Files saved for logger {logger_id}.")
            else:
                print("Issue with file saving.")

        self.save_datareader_object()
        
        config_manager = ConfigManager(deployment_folder='deployment_folder_path', deployment_id=deployment_id)
        config_manager.add_deployment_to_log(logger_ids = list(self.logger_info.keys()))

        # If export_edf is True, export the data to an EDF file
        if save_edf and edf_filename_template:
            if edf_save_from == 'sensor_data':
                self.exporter.export_to_edf(edf_filename_template, selected_sensors=selected_sensors, selected_channels=selected_channels)
            elif edf_save_from == 'data':
                self.exporter.export_to_edf_from_data(edf_filename_template)

        # If save_netcdf is True, save the data to a NetCDF file
        if save_netcdf:
            netcdf_filename = os.path.join(self.files_info['deployment_folder_path'], 'outputs', 'deployment_data.nc')
            self.exporter.save_to_netcdf(self, netcdf_filename)

    def import_notes(self):
        """Imports and processes notes associated with the selected deployment."""
        if self.deployment_info is None or self.deployment_info.empty:
            print("Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None
        print("import_notes")
        notes_filename = f"{self.deployment_info['Deployment ID']}_00_Notes.xlsx"
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
        # Check if the 'duration' column exists; if not, create it
        if 'duration' not in event_df.columns:
            # Initialize duration as 0 by default
            event_df['duration'] = 0

        # Set duration to 0 for 'point' events only
        event_df.loc[event_df['type'] == 'point', 'duration'] = 0

        # Leave the duration unchanged for 'state' events
        # (no need for additional action, as we want to retain existing values)

        # Print the dataframe to confirm changes (optional)
        print(event_df)
        print(f"Notes imported, processed, and sorted chronologically from {notes_filename}.")
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
            selected_deployment_id = selected_deployment['Deployment ID']
            
            print(f"Step 1: You selected the deployment: {selected_deployment_id}")
            print(f"Description: {selected_deployment['Notes']}")
        else:
            print("Invalid index selected.")
            return None, None

        deployment_folder = os.path.join(data_dir, selected_deployment_id)
        print(f"Step 2: Deployment folder path: {deployment_folder}")

        if os.path.exists(deployment_folder):
            print(f"Deployment folder found: {deployment_folder}")
        else:
            print(f"Folder {deployment_folder} not found. Searching for folders with a similar name...")
            possible_folders = [folder for folder in os.listdir(data_dir) 
                                if folder.startswith(selected_deployment_id)]

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
                    return None, None
            else:
                print("Error: Folder not found.")
                return None, None

        self.files_info['deployment_folder_path'] = deployment_folder
        print(f"Ready to process deployment folder: {self.files_info['deployment_folder_path']}")
        
        # Return both the folder path and the selected deployment ID
        return self.files_info['deployment_folder_path'], selected_deployment_id

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
