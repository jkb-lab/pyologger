import os
import struct
import pickle
import pytz
import json
import pandas as pd
from datetime import datetime, timedelta

class DataReader:
    def __init__(self, deployment_folder_path=None):
        self.deployment_data_folder = deployment_folder_path
        self.selected_deployment = None  # Store selected deployment metadata here
        self.data = {}  # To store both original and concatenated data
        self.metadata = {"channelnames": {}}  # Initialize metadata storage for channel names

    def read_files(self, metadata, save_csv=True):
        if not self.deployment_data_folder:
            print("No deployment folder set. Please use check_deployment_folder first.")
            return

        print(f"Step 2: Deployment folder initialized at: {self.deployment_data_folder}")
        
        # Step 3: Fetch metadata
        print("Step 3: Fetching metadata...")
        metadata.fetch_databases(verbose=False)
        logger_db = metadata.get_metadata("logger_DB")
        print("Metadata fetched successfully.")

        # New Step: Import notes
        print("Step 3.5: Importing notes...")
        notes_df = self.import_notes()
        if notes_df is not None:
            print("Notes imported successfully.")
            # You can now store the notes_df in a class attribute or process it further as needed
            self.notes_df = notes_df
        else:
            print("No valid notes were imported.")

        # Step 4: Organize files by logger ID
        print("Step 4: Organizing files by logger ID...")
        logger_ids = set(logger_db['LoggerID'])
        logger_files = {logger_id: [] for logger_id in logger_ids}

        for file in os.listdir(self.deployment_data_folder):
            for logger_id in logger_ids:
                if logger_id in file:
                    logger_files[logger_id].append(file)
                    break

        # Sort the files for each logger ID to ensure they are in the correct order
        for logger_id in logger_files:
            logger_files[logger_id].sort()

        # Filter down to loggers that actually have files
        loggers_with_files = {logger_id: files for logger_id, files in logger_files.items() if files}
        loggers_without_files = [logger_id for logger_id in logger_ids if logger_id not in loggers_with_files]

        # Print the summary of loggers with and without files
        print("Loggers with files:", ", ".join(loggers_with_files.keys()))
        print("Loggers without files:", ", ".join(loggers_without_files))

        # Step 5: Check for existing processed files only for loggers with files
        print("Step 5: Checking for existing processed files in the output folder for loggers with files...")
        if self.check_outputs_folder(loggers_with_files.keys()):
            print("Step 5 complete: Skipping reprocessing.")
            return
        
        # Step 6 & 7: Process each logger's files
        for logger_id, files in loggers_with_files.items():
            manufacturer = logger_db.loc[logger_db['LoggerID'] == logger_id, 'Manufacturer'].values[0]

            print(f"Step 6: Processing UBE files for logger: {logger_id} (Manufacturer: {manufacturer})")
            ube_files = [f for f in files if f.endswith('.ube')]
            csv_files = [f for f in files if f.endswith('.csv')]

            if ube_files:
                for ube_file in ube_files:
                    self.process_ube_file(ube_file, logger_id, save_csv)

            if csv_files:
                print(f"Step 7: Processing CSV files for logger: {logger_id}")
                self.concatenate_and_save_csvs(csv_files, logger_id, save_csv)

        print("Step 8: All processing complete.")
        
        # Step 9: Save the DataReader object as a pickle file
        pickle_filename = os.path.join(self.deployment_data_folder, 'outputs', 'data_reader.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Step 9: DataReader object successfully saved to {pickle_filename}.")


    def check_deployment_folder(self, dep_db, data_dir):
        # Step 1: Display relevant information to help the user decide
        print("Step 1: Displaying deployments to help you select one.")
        print(dep_db[['Deployment Name', 'Notes']])

        # Step 1: Prompt the user for input
        selected_index = int(input("Enter the index of the deployment you want to work with: "))

        # Step 2: Process the user's selection
        if 0 <= selected_index < len(dep_db):
            selected_deployment = dep_db.iloc[selected_index]
            self.selected_deployment = selected_deployment  # Save selected deployment to self
            print(f"Step 2: You selected the deployment: {selected_deployment['Deployment Name']}")
            print(f"Description: {selected_deployment['Notes']}")
        else:
            print("Invalid index selected.")
            return None

        # Get to deployment folder
        deployment_folder = os.path.join(data_dir, selected_deployment['Deployment Name'])
        # Verify the current working directory
        print(f"Step 2: Deployment folder path: {deployment_folder}")

        # Step 3: Check if the folder exists
        if os.path.exists(deployment_folder):
            print(f"Deployment folder found: {deployment_folder}")
        else:
            # If not found, search for a folder that starts with the deployment name
            print(f"Folder {deployment_folder} not found. Searching for folders with a similar name...")
            
            # Get a list of all folders in the data directory
            possible_folders = [folder for folder in os.listdir(data_dir) 
                                if folder.startswith(selected_deployment['Deployment Name'])]
            
            if len(possible_folders) == 1:
                # If exactly one match is found, use that folder
                deployment_folder = os.path.join(data_dir, possible_folders[0])
                print(f"Using the found folder: {deployment_folder}")
            elif len(possible_folders) > 1:
                # If multiple matches are found, ask the user to select one
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
                # If no matches are found, return an error
                print("Error: Folder not found.")
                return None

        # Continue processing if a valid folder was found
        self.deployment_data_folder = deployment_folder
        print(f"Ready to process deployment folder: {self.deployment_data_folder}")
        return self.deployment_data_folder

    def import_notes(self):
        if self.selected_deployment is None or self.selected_deployment.empty:
            print("Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None

        # Construct the filename based on the format: "{Deployment Name}_00_Notes.xlsx"
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

        # Read the .xlsx file
        try:
            notes_df = pd.read_excel(notes_filepath)
        except Exception as e:
            print(f"Error reading {notes_filename}: {e}")
            return None

        # Combine Rec Date and Start Time to form a base datetime
        rec_date_str = pd.to_datetime(rec_date).strftime('%Y-%m-%d')
        timezone = time_zone

        # Add Rec Date to time entries without a date
        def add_date_if_missing(time_str):
            try:
                # Try to parse directly as datetime (to handle full datetime entries)
                datetime_value = pd.to_datetime(time_str, errors='coerce')
                if pd.isna(datetime_value):
                    # If parsing as full datetime fails, treat it as time and prepend the Rec Date
                    datetime_value = pd.to_datetime(f"{rec_date_str} {time_str}", errors='coerce')
                return datetime_value
            except Exception as e:
                print(f"Error parsing time '{time_str}': {e}")
                return pd.NaT  # Return Not-a-Time if parsing fails

        # Apply the function to create the datetime column
        notes_df['datetime'] = notes_df.iloc[:, 0].apply(add_date_if_missing)

        # Convert base datetime to the specified timezone
        tz = pytz.timezone(timezone)
        notes_df['datetime'] = notes_df['datetime'].apply(lambda dt: tz.localize(dt) if pd.notna(dt) else dt)

        if notes_df['datetime'].isna().any():
            print(f"Error: Some timestamps could not be parsed.")
            return None

        print(f"Notes imported and processed from {notes_filename}.")
        return notes_df




    def check_outputs_folder(self, logger_ids):
        """Check if all expected processed files are already in the outputs folder."""
        output_folder = os.path.join(self.deployment_data_folder, 'outputs')
        if not os.path.exists(output_folder):
            print("Outputs folder does not exist. Processing required.")
            return False  # Outputs folder doesn't exist, so we need to process files

        existing_files = os.listdir(output_folder)
        print(f"Existing files in output folder: {existing_files}")

        for logger_id in logger_ids:
            # Check if any file in the outputs folder contains the logger ID
            matching_files = [filename for filename in existing_files if logger_id in filename]
            if not matching_files:
                print(f"No files found for logger ID {logger_id} in the output folder. Processing required.")
                return False  # If any logger ID doesn't have a file containing its ID, we need to process files
            else:
                print(f"Files found for logger ID {logger_id}: {matching_files}")

        print("All necessary files are already processed and available in the outputs folder.")
        return True  # All logger IDs have corresponding files in the outputs folder

    def concatenate_and_save_csvs(self, csv_files, logger_id, save_csv):
        dfs = []
        for file in csv_files:
            file_path = os.path.join(self.deployment_data_folder, file)
            try:
                data = self.read_csv(file_path)
                dfs.append(data)
                print(f"File: {file} - Successfully processed.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        # Concatenate DataFrames
        if len(dfs) > 1:
            concatenated_df = pd.concat(dfs, ignore_index=True)
        else:
            concatenated_df = dfs[0]

        # Define a mapping of original column names to computer-friendly names
        column_mapping = {
            "Date (UTC)": "datetime-utc",
            "Time (UTC)": "time-utc",
            "Date (local)": "date",
            "Time (local)": "time",
            "Accelerometer X [m/s²]": "accX",
            "Accelerometer Y [m/s²]": "accY",
            "Accelerometer Z [m/s²]": "accZ",
            "Gyroscope X [mrad/s]": "gyrX",
            "Gyroscope Y [mrad/s]": "gyrY",
            "Gyroscope Z [mrad/s]": "gyrZ",
            "Magnetometer X [µT]": "magX",
            "Magnetometer Y [µT]": "magY",
            "Magnetometer Z [µT]": "magZ",
            "Temperature (imu) [°C]": "tempIMU",
            "Depth (100bar) 1 [m]": "depth1",
            "Depth (100bar) 2 [°C]": "depth2",
            "Light intensity 1 [raw]": "light1",
            "Light intensity 2 [raw]": "light2",
            "System error": "sysError",
            "BATT [V]": "battV",
            "BATT [mA]": "battA",
            "BATT [mAh]": "battAh",
            "Camera": "camera",
            "Flags": "flags",
            "LED": "led",
            "Camera time": "cameraTime",
            "GPS": "gps",
            "CC status": "ccStatus",
            "CC vid. size [kBytes]": "ccVidSize",
            # Add more mappings as needed
        }

        # Define units for the computer-friendly names
        units_mapping = {
            "datetime-utc": "datetime",
            "time-utc": "time",
            "datetime": "date",
            "time": "time",
            "accX": "m/s²",
            "accY": "m/s²",
            "accZ": "m/s²",
            "gyrX": "mrad/s",
            "gyrY": "mrad/s",
            "gyrZ": "mrad/s",
            "magX": "µT",
            "magY": "µT",
            "magZ": "µT",
            "tempIMU": "°C",
            "depth1": "m",
            "depth2": "°C",
            "light1": "raw",
            "light2": "raw",
            "sysError": "unknown",
            "battV": "V",
            "battA": "mA",
            "battAh": "mAh",
            "camera": "unknown",
            "flags": "unknown",
            "led": "unknown",
            "cameraTime": "unknown",
            "gps": "unknown",
            "ccStatus": "unknown",
            "ccVidSize": "kBytes",
            # Add more units as needed
        }


        # Create a dictionary to store original names, friendly names, and units
        column_metadata = {}

        for original_name in concatenated_df.columns:
            friendly_name = column_mapping.get(original_name, original_name)  # Use original name if not mapped
            column_metadata[friendly_name] = {
                "original_name": original_name,
                "unit": units_mapping.get(friendly_name, "unknown")  # Default to "unknown" if not mapped
            }

        # Store the metadata for this logger
        self.metadata["channelnames"][logger_id] = column_metadata

        # Apply the renaming to the DataFrame
        concatenated_df.rename(columns={v["original_name"]: k for k, v in column_metadata.items()}, inplace=True)

        # Save the concatenated DataFrame and metadata
        output_filename = f"{'_'.join(csv_files[0].split('_')[:-1])}_ALL.csv"
        self.save_data(concatenated_df, output_filename, save_csv)

        # Save the metadata to a JSON file
        metadata_filename = f"{logger_id}_metadata.json"
        metadata_filepath = os.path.join(self.deployment_data_folder, 'outputs', metadata_filename)
        with open(metadata_filepath, 'w') as json_file:
            json.dump(self.metadata["channelnames"][logger_id], json_file, indent=4)
        print(f"Metadata for logger {logger_id} saved to {metadata_filepath}")

    def process_ube_file(self, ube_file, logger_id, save_csv):
        file_path = os.path.join(self.deployment_data_folder, ube_file)
        try:
            data = self.read_ube(file_path)
            output_filename = f"{'_'.join(ube_file.split('_')[:-1])}_only.csv"
            self.data[logger_id + '_ube_' + os.path.splitext(ube_file)[0]] = data  # Store the UBE data
            self.save_data(data, output_filename, save_csv)
            print(f"UBE file for logger {logger_id} saved to {output_filename}")
        except Exception as e:
            print(f"Error processing UBE file {ube_file} for logger {logger_id}: {e}")

    def read_ube(self, ube_path):
        if self.selected_deployment is None or self.selected_deployment.empty:
            print("Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None

        with open(ube_path, 'rb') as file:
            ube_raw = file.read()

        dl_time_str = ube_raw[0:32].decode('utf-8').strip()
        print(f"Parsed download timestamp string: '{dl_time_str}'")
        try:
            dl_time = datetime.strptime(dl_time_str, "%m-%d-%Y, %H:%M:%S")
        except ValueError as e:
            print(f"Error parsing timestamp: {dl_time_str} - {e}")
            raise

        # Extracting the record start time components
        mdhms = struct.unpack('BBBBB', ube_raw[32:37])
        now = datetime.now()
        record_start = datetime(now.year, mdhms[0], mdhms[1], mdhms[2], mdhms[3], mdhms[4])

        # Convert record_start to the correct timezone
        timezone = self.selected_deployment.get('Time Zone')
        if timezone:
            tz = pytz.timezone(timezone)
            record_start = tz.localize(record_start)
            print(f"Recording start time (localized): {record_start}")

        # Ensure the record_start date matches the Rec Date
        rec_date = pd.to_datetime(self.selected_deployment['Rec Date']).date()
        if record_start.date() != rec_date:
            print(f"Error: Recording start date {record_start.date()} does not match Rec Date {rec_date}.")
            raise ValueError(f"Recording start date {record_start.date()} does not match Rec Date {rec_date}.")

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

        # Generate the datetime column for the ECG data
        ecg_time = [record_start + timedelta(seconds=i/100) for i in range(len(ecg_data))]

        result = pd.DataFrame({
            'datetime': ecg_time,
            'ecg': ecg_data
        })
        result.attrs['created'] = dl_time

        return result


    def read_csv(self, csv_path):
        encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                print(f"Attempting to read {csv_path} with encoding {encoding}")
                return pd.read_csv(csv_path, encoding=encoding)
            except UnicodeDecodeError as e:
                print(f"Error reading {csv_path} with encoding {encoding}: {e}")
        raise UnicodeDecodeError(f"Failed to read {csv_path} with available encodings.")

    def save_data(self, data, filename, save_csv=True):
        attribute_name = os.path.splitext(filename)[0]
        self.data[attribute_name] = data  # Save everything to data
        
        if save_csv:
            output_folder = os.path.join(self.deployment_data_folder, 'outputs')
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, filename)
            data.to_csv(output_path, index=False)
            print(f"Data successfully saved to: {output_path}")
        else:
            print(f"Data saved to attribute {attribute_name} but not to CSV.")
