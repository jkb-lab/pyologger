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
    
    def __init__(self, deployment_folder_path=None, custom_mapping_path=None):
        self.deployment_data_folder = deployment_folder_path
        self.selected_deployment = None  # Store selected deployment metadata here
        self.data = {}  # Store data by logger ID
        self.info = {}  # Store info (metadata) by logger ID
        self.sensor_data = {}
        self.sensor_info = {}  # Initialize sensor_info to store sensor metadata
        self.column_mapping = None  # Initialize the column mapping

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
                processor = CATSManufacturer(self, logger_id, 
                                             manufacturer="CATS")
            elif manufacturer == "UFI":
                processor = UFIManufacturer(self, logger_id, 
                                            manufacturer="UFI")
            else:
                print(f"Manufacturer {manufacturer} is not supported.")
                continue

            # Update to properly handle the returned values from process_files
            result = processor.process_files(files)

            final_df, datetime_metadata, sensor_groups, sensor_fs = result

            if final_df is not None and 'datetime' in final_df.columns:
                self.info[logger_id]['datetime_metadata'] = datetime_metadata
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

    def __init__(self, data_reader, logger_id, manufacturer):
        self.logger_id = logger_id
        self.logger_manufacturer = manufacturer
        self.data_reader = data_reader
        self.column_mapping = data_reader.column_mapping
        self.expected_frequencies = {}  # This will hold the expected frequencies parsed from the .txt file

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

            print(f"Original name: {original_name}, Clean name: {clean_name}")
            
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
            sensor_type = mapping_info.get("sensor", "extra")

            print(f"Original name: {original_name}, Clean name: {clean_name}, Mapped name: {mapped_name}, Sensor type: {sensor_type}")

            column_metadata[mapped_name] = {
                "original_name": original_name,
                "unit": unit or "unknown",
                "sensor": sensor_type
            }
            new_columns[original_name] = mapped_name

        # Rename the columns in the DataFrame
        df.rename(columns=new_columns, inplace=True)

        return df, column_metadata

    def map_data_to_sensors(self, df, logger_id, column_metadata):
            """Groups data columns to sensors and downsamples based on expected frequencies."""
            sensor_groups = {}
            sensor_fs = {}

            for sensor_name in set(v['sensor'].lower() for v in column_metadata.values()):
                if sensor_name == 'extra':
                    continue  # Skip 'extra' sensor type

                # Group columns by sensor
                sensor_cols = [col for col, meta in column_metadata.items() if meta['sensor'].lower() == sensor_name]
                sensor_df = df[['datetime'] + sensor_cols].copy()

                # Get sampling frequency for each sensor (logger-specific methods or default)
                expected_frequency = self.expected_frequencies.get(sensor_name)
                if not expected_frequency and sensor_name == 'ecg':
                    # For UFI ECG, use the overall frequency (e.g., determined earlier)
                    expected_frequency = float(self.data_reader.info[logger_id]['datetime_metadata']['fs'])

                if expected_frequency:
                    sensor_fs[sensor_name] = expected_frequency
                    step = max(1, int(round(df['datetime'].diff().mean().total_seconds() * expected_frequency)))
                    sensor_df = sensor_df.iloc[::step]  # Downsample data

                self.data_reader.sensor_data[sensor_name] = sensor_df
                self.data_reader.sensor_info[sensor_name] = {
                    'channels': sensor_cols,
                    'metadata': {col: column_metadata[col] for col in sensor_cols},
                    'sampling_frequency': sensor_fs.get(sensor_name),
                    'logger_id': self.logger_id,
                    'logger_manufacturer': self.logger_manufacturer,
                }

            # Print final mapping and downsampling results
            for sensor_name, df in self.data_reader.sensor_data.items():
                print(f"Sensor '{sensor_name}' data processed and stored with shape {df.shape}.")

            return sensor_groups, sensor_fs

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

                # Use the correct sub-dictionary for column mapping, assuming you're processing data for a CATS logger
                mapping_sub_dict = self.column_mapping.get(self.logger_manufacturer, {})

                for sensor_id, name, interval in sensor_info:
                    name = name.strip().lower()  # Clean up the sensor name

                    # Reverse lookup in the mapping sub-dictionary to find the matching sensor name
                    for clean_name, mapping in mapping_sub_dict.items():
                        if mapping['sensor'].lower() == name:
                            sensor_key = clean_name
                            frequency = int(interval)
                            self.expected_frequencies[sensor_key] = frequency
                            print(f"Sensor {sensor_key.capitalize()} expected sampling frequency: {frequency} Hz")
                            break
                    else:
                        print(f"Sensor name '{name}' not found in column mapping. Ignoring this sensor.")

            else:
                print("No 'activated sensors' section found in the file.")

        except Exception as e:
            print(f"Failed to parse {txt_file_path} due to: {e}")

    def group_by_sensors(self, df):
        """Groups data columns into sensors based on the sensor mapping and downsamples based on expected frequencies."""
        sensor_groups = {}
        sensor_fs = {}

        # Group columns by sensors using the metadata gathered during CSV mapping
        for column_name, metadata in df.items():
            sensor_name = metadata.get("sensor", "extra").lower()

            if sensor_name == 'extra' or sensor_name == 'none':
                print(f"SKIPPING {column_name} because sensor name is: {sensor_name}")
                continue  # Skip extra sensors or columns that don't belong to any sensor

            sensor_cols = [col for col in df.columns if col == column_name]
            if sensor_cols:
                # Create a DataFrame with datetime and sensor columns
                sensor_df = df[['datetime'] + sensor_cols].copy()

                # Use the expected sampling frequency to downsample the data
                expected_frequency = self.expected_frequencies.get(sensor_name)
                if expected_frequency:
                    sensor_freq = expected_frequency
                    sensor_fs[sensor_name] = sensor_freq

                    # Downsample the sensor data to its frequency
                    step = max(1, int(round(df['datetime'].diff().mean().total_seconds() * sensor_freq)))
                    downsampled_df = sensor_df.iloc[::step]
                    sensor_groups[sensor_name] = downsampled_df

                    print(f"Channels {', '.join(sensor_cols)} grouped by {sensor_name.capitalize()} sensor with a frequency of {sensor_freq:.2f} Hz.")
                else:
                    sensor_fs[sensor_name] = None
                    print(f"No expected frequency found for sensor {sensor_name.capitalize()}.")
                    sensor_groups[sensor_name] = sensor_df  # No downsampling if no frequency is found

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
            print(f"No valid files found for {self.logger_manufacturer} logger.")
            return None, None, None, None  # Return four None values

        # Remove the .txt files from the list after processing them
        files = [f for f in files if not f.endswith('.txt')]

        # Concatenate the remaining files into one DataFrame
        final_df = self.concatenate_and_save_csvs(files)

        # Rename columns
        final_df, column_metadata = self.rename_columns(final_df, self.logger_id, self.logger_manufacturer)
        
        # Process datetime and return metadata
        final_df, datetime_metadata = self.data_reader.process_datetime(final_df, time_zone=self.data_reader.selected_deployment['Time Zone'])
        self.data_reader.info[self.logger_id]['datetime_metadata'] = datetime_metadata

        # Map data to sensors and return sensor information
        sensor_groups, sensor_fs = self.map_data_to_sensors(final_df, self.logger_id, column_metadata)

        return final_df, datetime_metadata, sensor_groups, sensor_fs

    def concatenate_and_save_csvs(self, csv_files):
        """Concatenates multiple CSV files into one DataFrame."""
        dfs = []
        for file in csv_files:
            file_path = os.path.join(self.data_reader.deployment_data_folder, file)
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
        file_path = os.path.join(self.data_reader.deployment_data_folder, txt_file)
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
            final_df, datetime_metadata = self.data_reader.process_datetime(final_df, time_zone=self.data_reader.selected_deployment['Time Zone'])
            self.data_reader.info[self.logger_id]['datetime_metadata'] = datetime_metadata
            
            # Map data to sensors and return sensor information
            sensor_groups, sensor_fs = self.map_data_to_sensors(final_df, self.logger_id, column_metadata)

            return final_df, datetime_metadata, sensor_groups, sensor_fs
        else:
            print(f"No .ube file found for UFI logger.")
            return None, None, None, None  # Return four None values

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

            return result

        except Exception as e:
            print(f"Error processing UBE file {ube_file}: {e}")
            return pd.DataFrame(), {}
