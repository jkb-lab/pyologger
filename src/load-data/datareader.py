import os
import struct
import pickle
import pandas as pd
from datetime import datetime, timedelta

class DataReader:
    def __init__(self, deployment_folder_path=None):
        self.deployment_data_folder = deployment_folder_path
        self.data_new = {}  # To store both original and concatenated data

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

    def check_deployment_folder(self, dep_db, data_dir):
        # Step 1: Display relevant information to help the user decide
        print("Step 1: Displaying deployments to help you select one.")
        print(dep_db[['Deployment Name', 'Notes']])

        # Step 1: Prompt the user for input
        selected_index = int(input("Enter the index of the deployment you want to work with: "))

        # Step 2: Process the user's selection
        if 0 <= selected_index < len(dep_db):
            selected_deployment = dep_db.iloc[selected_index]
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

        # Concatenate and save the DataFrames
        if len(dfs) > 1:
            concatenated_df = pd.concat(dfs, ignore_index=True)
            original_filename = csv_files[0]  # Use the first file's name as the base for the output filename
            output_filename = f"{'_'.join(original_filename.split('_')[:-1])}_ALL.csv"  # Replace the suffix with "_ALL.csv"
            self.save_data(concatenated_df, output_filename, save_csv)
        else:
            # Only one file, save without changing the name
            concatenated_df = dfs[0]
            output_filename = csv_files[0]
            self.save_data(concatenated_df, output_filename, save_csv)

        print(f"Data for logger {logger_id} saved to {output_filename}")

    def process_ube_file(self, ube_file, logger_id, save_csv):
        file_path = os.path.join(self.deployment_data_folder, ube_file)
        try:
            data = self.read_ube(file_path)
            output_filename = f"{'_'.join(ube_file.split('_')[:-1])}_only.csv"
            self.data_new[logger_id + '_ube_' + os.path.splitext(ube_file)[0]] = data  # Store the UBE data
            self.save_data(data, output_filename, save_csv)
            print(f"UBE file for logger {logger_id} saved to {output_filename}")
        except Exception as e:
            print(f"Error processing UBE file {ube_file} for logger {logger_id}: {e}")

    def read_ube(self, ube_path):
        with open(ube_path, 'rb') as file:
            ube_raw = file.read()

        dl_time_str = ube_raw[0:32].decode('utf-8').strip()
        print(f"Parsed download timestamp string: '{dl_time_str}'")
        try:
            dl_time = datetime.strptime(dl_time_str, "%m-%d-%Y, %H:%M:%S")
        except ValueError as e:
            print(f"Error parsing timestamp: {dl_time_str} - {e}")
            raise

        mdhms = struct.unpack('BBBBB', ube_raw[32:37])
        now = datetime.now()
        record_start = datetime(now.year, mdhms[0], mdhms[1], mdhms[2], mdhms[3], mdhms[4])
        print(f"Recording start time: {record_start}")

        data_new = ube_raw[40:]
        
        ecg_channel = 0x20
        ecg_data = []

        for i in range(0, len(data_new), 2):
            channel = data_new[i]
            value = data_new[i + 1]
            if (channel & 0xF0) == ecg_channel:
                ecg_value = (channel & 0x0F) << 8 | value
                ecg_data.append(ecg_value)
        
        print(f"Total ECG data points: {len(ecg_data)}")

        ecg_time = [record_start + timedelta(seconds=i/100) for i in range(len(ecg_data))]

        result = pd.DataFrame({
            'timestamp': ecg_time,
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
        self.data_new[attribute_name] = data  # Save everything to data_new
        
        if save_csv:
            output_folder = os.path.join(self.deployment_data_folder, 'outputs')
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, filename)
            data.to_csv(output_path, index=False)
            print(f"Data successfully saved to: {output_path}")
        else:
            print(f"Data saved to attribute {attribute_name} but not to CSV.")
