import os
import struct
import pickle
import pytz
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataReader:
    """A class for handling the reading and processing of deployment data files."""
    
    def __init__(self, deployment_folder_path=None, custom_mapping_path=None, sensor_mapping_path=None):
        self.deployment_data_folder = deployment_folder_path
        self.selected_deployment = None  # Store selected deployment metadata here
        self.data = {}  # Store data by logger ID
        self.info = {}  # Store info (metadata) by logger ID
        self.sensor_data = {}
        self.column_mapping = None  # Initialize the column mapping
        self.sensor_mapping_path = sensor_mapping_path  # Store the sensor mapping path

        # Load the custom JSON mapping for column names if deployment folder is provided
        if custom_mapping_path:
            self.load_custom_mapping(custom_mapping_path)

    def load_custom_mapping(self, custom_mapping_path):
        """Loads custom JSON mapping for column names."""
        try:
            with open(custom_mapping_path, 'r') as json_file:
                self.column_mapping = json.load(json_file)
                print(f"Custom column mapping loaded from {custom_mapping_path}")
        except FileNotFoundError:
            print(f"Custom mapping file not found at {custom_mapping_path}. Proceeding without it.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {custom_mapping_path}: {e}")
            self.column_mapping = None

    def read_files(self, metadata, save_csv=True, save_parq=True):
        """Reads and processes the deployment data files based on manufacturer."""
        if not self.deployment_data_folder:
            print("No deployment folder set. Please use check_deployment_folder first.")
            return

        print(f"Step 2: Deployment folder initialized at: {self.deployment_data_folder}")

        metadata.fetch_databases(verbose=False)
        logger_db = metadata.get_metadata("logger_DB")

        notes_df = self.import_notes()
        if notes_df is not None:
            self.notes_df = notes_df

        logger_files = self.organize_files_by_logger_id(logger_db)

        if self.check_outputs_folder(logger_files.keys()):
            print("All necessary files are already processed. Skipping further processing.")
            return

        for logger_id, files in logger_files.items():
            manufacturer = logger_db.loc[logger_db['LoggerID'] == logger_id, 'Manufacturer'].values[0]
            if manufacturer == "CATS":
                processor = CATSManufacturer(self, logger_id, sensor_mapping_path=self.sensor_mapping_path)
            elif manufacturer == "UFI":
                processor = UFIManufacturer(self, logger_id, sensor_mapping_path=self.sensor_mapping_path)
            else:
                print(f"Manufacturer {manufacturer} is not supported.")
                continue

            # Update to properly handle the returned values from process_files
            result = processor.process_files(files)
            
            # Handle the case when process_files returns None or an unexpected number of results
            if result is None or len(result) != 5:
                print(f"Failed to process files for logger {logger_id}. Skipping this logger.")
                continue

            final_df, channel_info, datetime_metadata, sensor_groups, sensor_fs = result

            if final_df is not None and 'datetime' in final_df.columns:
                self.info[logger_id]['channelinfo'] = channel_info
                self.info[logger_id]['datetime_metadata'] = datetime_metadata
                self.info[logger_id]['sensor_fs'] = sensor_fs
                self.sensor_data[logger_id] = sensor_groups
                self.save_data(final_df, logger_id, f"{logger_id}.csv", save_csv, save_parq)
                print(f"Files saved for logger {logger_id}.")
            else:
                print("Issue with file saving.")

        self.save_datareader_object()


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
        logger_ids = set(logger_db['LoggerID'])
        logger_files = {logger_id: [] for logger_id in logger_ids}

        for file in os.listdir(self.deployment_data_folder):
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
            self.data[logger_id] = None
            self.info[logger_id] = {"channelinfo": {}}

        return loggers_with_files

    def save_datareader_object(self):
        """Saves the DataReader object as a pickle file."""
        pickle_filename = os.path.join(self.deployment_data_folder, 'outputs', 'data.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"DataReader object successfully saved to {pickle_filename}.")

    def check_deployment_folder(self, dep_db, data_dir):
        """Checks the deployment folder and allows the user to select a deployment."""
        print("Step 1: Displaying deployments to help you select one.")
        print(dep_db[['Deployment Name', 'Notes']])

        selected_index = int(input("Enter the index of the deployment you want to work with: "))

        if 0 <= selected_index < len(dep_db):
            selected_deployment = dep_db.iloc[selected_index]
            self.selected_deployment = selected_deployment  # Save selected deployment to self
            print(f"Step 1: You selected the deployment: {selected_deployment['Deployment Name']}")
            print(f"Description: {selected_deployment['Notes']}")
        else:
            print("Invalid index selected.")
            return None

        deployment_folder = os.path.join(data_dir, selected_deployment['Deployment Name'])
        print(f"Step 2: Deployment folder path: {deployment_folder}")

        if os.path.exists(deployment_folder):
            print(f"Deployment folder found: {deployment_folder}")
        else:
            print(f"Folder {deployment_folder} not found. Searching for folders with a similar name...")
            possible_folders = [folder for folder in os.listdir(data_dir) 
                                if folder.startswith(selected_deployment['Deployment Name'])]

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

        self.deployment_data_folder = deployment_folder
        print(f"Ready to process deployment folder: {self.deployment_data_folder}")
        return self.deployment_data_folder

    def import_notes(self):
        """Imports and processes notes associated with the selected deployment."""
        if self.selected_deployment is None or self.selected_deployment.empty:
            print("Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None

        notes_filename = f"{self.selected_deployment['Deployment Name']}_00_Notes.xlsx"
        rec_date = self.selected_deployment['Rec Date']
        start_time = self.selected_deployment.get('Start Time', "00:00:00")
        time_zone = self.selected_deployment.get('Time Zone')

        if not time_zone:
            print(f"No time zone information found for the selected deployment.")
            return None

        notes_filepath = os.path.join(self.deployment_data_folder, notes_filename)

        if not os.path.exists(notes_filepath):
            print(f"Notes file {notes_filename} not found in {self.deployment_data_folder}.")
            return None

        try:
            notes_df = pd.read_excel(notes_filepath)
        except Exception as e:
            print(f"Error reading {notes_filename}: {e}")
            return None

        notes_df, datetime_metadata = self.process_datetime(notes_df, time_zone=time_zone)
        
        if notes_df['datetime'].isna().any():
            print(f"WARNING: Some timestamps could not be parsed.")
            return notes_df

        notes_df = notes_df.sort_values(by='datetime').reset_index(drop=True)
        print(f"Notes imported, processed, and sorted chronologically from {notes_filename}.")
        return notes_df

    def check_outputs_folder(self, logger_ids):
        """Checks if the processed data files for the given loggers already exist in the outputs folder."""
        output_folder = os.path.join(self.deployment_data_folder, 'outputs')
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
        self.data[logger_id] = {}
        output_folder = os.path.join(self.deployment_data_folder, 'outputs')
        os.makedirs(output_folder, exist_ok=True)

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

class BaseManufacturer:
    """Base class for handling manufacturer-specific processing."""

    def __init__(self, data_reader, logger_id, sensor_mapping_path=None):
        self.logger_id = logger_id
        self.data_reader = data_reader
        self.column_mapping = data_reader.column_mapping
        self.sensor_mapping = self.load_sensor_mapping(sensor_mapping_path)
        self.expected_intervals = {}  # This will hold the expected intervals parsed from the .txt file

    def load_sensor_mapping(self, sensor_mapping_path):
        """Loads the sensor mapping JSON file."""
        if sensor_mapping_path:
            try:
                with open(sensor_mapping_path, 'r') as json_file:
                    sensor_mapping = json.load(json_file)
                    print(f"Sensor mapping loaded from {sensor_mapping_path}")
                    return sensor_mapping
            except FileNotFoundError:
                print(f"Sensor mapping file not found at {sensor_mapping_path}. Proceeding without it.")
        return {}

    def parse_txt_for_intervals(self, txt_file_path):
        """Parses the .txt file to extract expected sampling intervals for sensors."""
        with open(txt_file_path, 'r') as file:
            content = file.read()

        sensor_info = re.findall(r'\d{2}_name=(\w+).*?interval=(\d+)', content, re.DOTALL)
        for name, interval in sensor_info:
            sensor_key = name.lower()
            self.expected_intervals[sensor_key] = int(interval)
            print(f"Sensor {name} expected sampling interval: {interval} Hz")

    def group_by_sensors(self, df):
        """Groups data columns into sensors based on the sensor mapping and downsamples based on expected intervals."""
        sensor_groups = {}
        sensor_fs = {}

        # Group columns by sensors
        for sensor_key, sensor_name in self.sensor_mapping.items():
            sensor_cols = [col for col in df.columns if col.startswith(sensor_key)]
            if sensor_cols:
                # Create a DataFrame with datetime and sensor columns
                sensor_df = df[['datetime'] + sensor_cols].copy()
                sensor_groups[sensor_name] = sensor_df

                # Use the expected sampling interval to calculate frequency and downsample
                expected_interval = self.expected_intervals.get(sensor_key)
                if expected_interval:
                    sensor_freq = 1000 / expected_interval  # Convert ms to Hz
                    sensor_fs[sensor_name] = sensor_freq

                    # Downsample the sensor data to its frequency
                    step = max(1, int(round(sensor_freq / sensor_freq)))  # No additional downsampling
                    downsampled_df = sensor_df.iloc[::step]
                    sensor_groups[sensor_name] = downsampled_df

                    print(f"Channels {', '.join(sensor_cols)} grouped by {sensor_name} sensor with a frequency of {sensor_freq:.2f} Hz.")
                else:
                    sensor_fs[sensor_name] = None
                    print(f"No expected interval found for sensor {sensor_name}.")
                    sensor_groups[sensor_name] = sensor_df  # No downsampling if no interval is found

        return sensor_groups, sensor_fs
    
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
            self.parse_txt_for_intervals(os.path.join(self.data_reader.deployment_data_folder, txt_file))

        if not files:
            print("No valid files found for CATS logger.")
            return None, None, None, None, None  # Return five None values

        # Remove the .txt files from the list after processing them
        files = [f for f in files if not f.endswith('.txt')]

        # Process the remaining files
        final_df, channel_info = self.concatenate_and_save_csvs(files)
        final_df, datetime_metadata = self.data_reader.process_datetime(final_df, time_zone=self.data_reader.selected_deployment['Time Zone'])

        # Group data by sensors and downsample based on expected intervals
        sensor_groups, sensor_fs = self.group_by_sensors(final_df)

        return final_df, channel_info, datetime_metadata, sensor_groups, sensor_fs  # Return five values


    def print_txt_content(self, txt_file):
        """Prints the content of a .txt file."""
        file_path = os.path.join(self.data_reader.deployment_data_folder, txt_file)
        with open(file_path, 'r') as file:
            print(file.read())

    def concatenate_and_save_csvs(self, csv_files):
        """Concatenates multiple CSV files, normalizes column names, applies custom mappings, and handles duplicates."""
        dfs = []
        for file in csv_files:
            file_path = os.path.join(self.data_reader.deployment_data_folder, file)
            try:
                data = self.data_reader.read_csv(file_path)
                dfs.append(data)
                print(f"File: {file} - Successfully processed.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        if len(dfs) > 1:
            concatenated_df = pd.concat(dfs, ignore_index=True)
        else:
            concatenated_df = dfs[0]

        # Initialize dictionaries for mapping and metadata
        column_metadata = {}
        new_columns = {}
        seen_names = set()

        for original_name in concatenated_df.columns:
            clean_name = original_name.strip().lower().replace(" ", "_").replace(".", "")  # Remove spaces, periods, and convert to lowercase

            # Extract and store units
            unit = None
            # Handle units in []
            if "[" in clean_name and "]" in clean_name:
                name, square_unit = clean_name.split("[", 1)
                square_unit = square_unit.replace("]", "").strip().lower()
                name = name.strip("_")
                clean_name = name  # Exclude units from column name
                unit = square_unit  # Store the [] unit
            # Handle units in ()
            if "(" in clean_name and ")" in clean_name:
                name, round_unit = clean_name.split("(", 1)
                round_unit = round_unit.replace(")", "").strip().lower()
                clean_name = f"{name.strip('_')}_{round_unit}"  # Include () units as a suffix
                unit = round_unit  # Overwrite the unit if () unit is present

            # Apply custom mapping
            if self.column_mapping and clean_name in self.column_mapping.get("CATS", {}):
                mapped_name = self.column_mapping["CATS"][clean_name]
            else:
                mapped_name = clean_name

            # Ensure local and UTC times remain distinct
            if "local" in original_name.lower() and "local" not in mapped_name:
                mapped_name = f"{mapped_name}_local"
            elif "utc" in original_name.lower() and "utc" not in mapped_name:
                mapped_name = f"{mapped_name}_utc"

            # Check for duplicates and handle them
            if mapped_name in seen_names:
                if unit:
                    mapped_name = f"{mapped_name}_{unit}"
                else:
                    mapped_name = f"{mapped_name}_dup"  # Add a suffix to distinguish duplicates
            seen_names.add(mapped_name)

            column_metadata[mapped_name] = {
                "original_name": original_name,
                "unit": unit or "unknown"
            }
            new_columns[original_name] = mapped_name

        # Rename the columns in the DataFrame
        concatenated_df.rename(columns=new_columns, inplace=True)

        # Print the final column mapping for review
        print("Final column mapping from original to final names:")
        for original, final in new_columns.items():
            print(f"{original} -> {final}")

        return concatenated_df, column_metadata

class UFIManufacturer(BaseManufacturer):
    """UFI-specific processing."""

    def process_files(self, files):
        """Process UFI files, specifically looking for .ube files."""
        # Find the first .ube file in the files list
        ube_file = next((f for f in files if f.endswith('.ube')), None)
        
        if ube_file:
            print(f"Processing .ube file: {ube_file}")
            final_df, channel_info = self.process_ube_file(ube_file)

            # Process datetime and return the final dataframe and channel info
            final_df, datetime_metadata = self.data_reader.process_datetime(final_df, time_zone=self.data_reader.selected_deployment['Time Zone'])
            return final_df, channel_info, datetime_metadata
        else:
            print(f"No .ube file found for UFI logger.")
            return None, None, None

    def process_ube_file(self, ube_file):
        """Processes a UBE file and extracts data."""
        file_path = os.path.join(self.data_reader.deployment_data_folder, ube_file)
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

            timezone = self.data_reader.selected_deployment.get('Time Zone')
            if timezone:
                tz = pytz.timezone(timezone)
                record_start = tz.localize(record_start)
                print(f"Recording start time (localized): {record_start}")

            rec_date = pd.to_datetime(self.data_reader.selected_deployment['Rec Date']).date()
            if record_start.date() != rec_date:
                print(f"Error: Recording start date {record_start.date()} does not match Rec Date {rec_date}.")
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

            # Define a mapping of original column names to computer-friendly names
            column_mapping = {
                "datetime": "datetime",
                "ecg": "ecg",
            }

            # Define units for the computer-friendly names
            units_mapping = {
                "datetime": "datetime",
                "ecg": "microVolts",
            }

            column_metadata = {}

            for original_name in result.columns:
                friendly_name = column_mapping.get(original_name, original_name)
                column_metadata[friendly_name] = {
                    "original_name": original_name,
                    "unit": units_mapping.get(friendly_name, "unknown")
                }

            result.rename(columns={v["original_name"]: k for k, v in column_metadata.items()}, inplace=True)

            return result, column_metadata

        except Exception as e:
            print(f"Error processing UBE file {ube_file}: {e}")
            return pd.DataFrame(), {}
