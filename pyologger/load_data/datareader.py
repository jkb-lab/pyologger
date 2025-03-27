import os
import pickle
import pandas as pd
from datetime import datetime
from pyologger.utils.time_manager import *
from pyologger.process_data.sampling import *
from pyologger.load_data.metadata import *
from pyologger.io_operations import *
from pyologger.io_operations.base_exporter import *
from pyologger.io_operations.cats_importer import *
from pyologger.io_operations.ufi_importer import *
from pyologger.io_operations.wc_importer import *
from pyologger.io_operations.ll_importer import *
from pyologger.io_operations.evolocus_importer import *
from pyologger.io_operations.manitty_importer import *

class DataReader:
    """A class for handling the reading and processing of deployment data files."""
    
    def __init__(self, dataset_folder: str, deployment_id: str, data_subfolder: str = None, montage_path: str = None):
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
        self.montage_path = montage_path
        self.deployment_info = {}  # Store selected deployment metadata here
        self.output_folder = os.path.join(self.deployment_folder, 'outputs')
        
        self.animal_info = {'Animal_ID': self.deployment_id.split('_', 1)[1] if '_' in self.deployment_id else None}
        self.dataset_info = {'Dataset_ID': os.path.basename(os.path.normpath(dataset_folder))}

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
    
    def read_files(self, deployment_info, loggers_used, save_parq=True,
                save_netcdf=False):
        """
        Reads and processes deployment data files from the specified folder.

        Parameters:
        - deployment_info: Dict containing "Deployment Latitude", "Deployment Longitude", and "Time Zone".
        - loggers_used: List of dictionaries, each containing "Logger ID", "Manufacturer", and "Montage ID".
        - save_parq, save_netcdf: Flags for saving output formats.
        """

        print(f"🔄 Reading files from: {self.data_folder}")
        print(f"📁 Created data folder: {self.output_folder}")

        # Store minimal metadata explicitly instead of update()
        self.deployment_info["Deployment Date"] = deployment_info.get("Deployment Date")
        self.deployment_info["Deployment Latitude"] = deployment_info.get("Deployment Latitude")
        self.deployment_info["Deployment Longitude"] = deployment_info.get("Deployment Longitude")
        self.deployment_info["Time Zone"] = deployment_info.get("Time Zone")

        # Store logger information including Montage IDs
        self.logger_info = {
            logger["Logger ID"]: {
                "ID": logger["Logger ID"],
                "Manufacturer": logger["Manufacturer"],
                "Montage ID": logger.get("Montage ID", "Unknown")  # Default to "Unknown" if missing
            }
            for logger in loggers_used
        }

        # Step 2: Import event notes from data subfolder
        self.event_data = self.import_notes()

        # Step 3: Organize files by logger ID
        logger_files = {logger["Logger ID"]: [] for logger in loggers_used}
        for file in os.listdir(self.data_folder):
            for logger in loggers_used:
                if logger["Logger ID"] in file:
                    logger_files[logger["Logger ID"]].append(file)
                    break

        for logger_id in logger_files:
            logger_files[logger_id].sort()

        loggers_with_files = {logger_id: files for logger_id, files in logger_files.items() if files}
        loggers_without_files = [logger["Logger ID"] for logger in loggers_used if logger["Logger ID"] not in loggers_with_files]

        print("📟 Loggers with files:", ", ".join(loggers_with_files.keys()))
        print("⚠ Loggers without files:", ", ".join(loggers_without_files) if loggers_without_files else "None")

        # Step 5: Process each logger if not already processed
        for logger in loggers_used:
            logger_id = logger["Logger ID"]
            manufacturer = logger["Manufacturer"]

            if logger_id in loggers_with_files:
                files = loggers_with_files[logger_id]

                # Step 6: Select manufacturer-specific processor
                processor_classes = {
                    "CATS": CATSImporter,
                    "UFI": UFIImporter,
                    "LL": LLImporter,
                    "WC": WCImporter,
                    "Evolocus": EvolocusImporter,
                    "Manitty": ManittyImporter  # if/when implemented
                }

                processor_class = processor_classes.get(manufacturer)

                if processor_class:
                    processor_instance = processor_class(self, logger_id)
                    result = processor_instance.process_files(files)

                    if result:
                        if manufacturer in ['Evolocus', 'Manitty']:
                            # Expecting dictionaries by frequency
                            final_dfs_dict, channel_metadata_dict, datetime_metadata_dict, sensor_groups_dict, sensor_info_dict = result

                            for freq in final_dfs_dict:
                                df = final_dfs_dict[freq]
                                if df is not None and 'datetime' in df.columns:
                                    filename = f"{logger_id}_{freq}Hz.csv"
                                    self.exporter.save_data(df, logger_id, filename, save_parq)

                            print(f"✅ Processed and saved multi-frequency data for logger {logger_id}.")
                        else:
                            # Standard single final_df path
                            final_df, channel_metadata, datetime_metadata, sensor_groups, sensor_info = result

                            if final_df is not None and 'datetime' in final_df.columns:
                                self.exporter.save_data(final_df, logger_id, f"{logger_id}.csv", save_parq)
                                print(f"✅ Processed and saved files for logger {logger_id}.")
                            else:
                                print(f"⚠ Issue with file saving for logger {logger_id}.")
                else:
                    print(f"⚠ Manufacturer {manufacturer} is not supported.")


        # Step 7: Save the DataReader object to a pickle file
        self.save_datareader_object()

        if save_netcdf:
            netcdf_filename = os.path.join(self.deployment_folder, 'outputs', f'{self.deployment_id}_00_processed.nc')
            self.exporter.save_to_netcdf(self, netcdf_filename)
            print(f"📊 Saved deployment data to NetCDF: {netcdf_filename}")


    def check_outputs_folder(self):
        """
        Checks if the processed data files for the given loggers already exist.

        Returns:
            bool: True if the outputs folder exists and contains a .pkl file, False otherwise.
        """

        if not os.path.exists(self.output_folder):
            print(f"❌ Outputs folder '{self.output_folder}' does not exist. Processing required.")
            os.makedirs(self.output_folder, exist_ok=True)
            return False

        if not any(fname.endswith('.pkl') for fname in os.listdir(self.output_folder)):
            print(f"❌ Outputs folder '{self.output_folder}' does not contain a .pkl file. Processing required.")
            return False

        print(f"✅ Outputs folder '{self.output_folder}' exists and contains a .pkl file.")
        return True

    def import_notes(self):
        """Imports and processes notes associated with the selected deployment."""
        
        if not self.deployment_info:
            print("❌ Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None

        notes_filename = f"{self.deployment_id}_00_Notes.xlsx"
        time_zone = self.deployment_info.get("Time Zone")

        if not time_zone:
            print(f"⚠ No time zone information found for deployment {self.deployment_id}. Defaulting to UTC.")
            time_zone = "UTC"  # Fallback to UTC

        # Look in the data subfolder (self.data_folder) instead of deployment folder
        notes_filepath = os.path.join(self.data_folder, notes_filename)

        if not os.path.exists(notes_filepath):
            print(f"⚠️ Notes file '{notes_filename}' not found in {self.data_folder}. Skipping import.")
            return None

        try:
            event_df = pd.read_excel(notes_filepath)
            print(f"📂 Successfully loaded notes file: {notes_filepath}")
        except Exception as e:
            print(f"❌ Error reading {notes_filename}: {e}")
            return None

        # Process timestamps
        event_df, _ = process_datetime(event_df, time_zone=time_zone)

        if event_df["datetime"].isna().any():
            print(f"⚠️ WARNING: Some timestamps could not be parsed correctly.")
        
        # Ensure the dataframe is sorted by time
        event_df = event_df.sort_values(by="datetime").reset_index(drop=True)

        # Ensure 'duration' column exists and is properly set
        if "duration" not in event_df.columns:
            event_df["duration"] = 0

        event_df.loc[event_df["type"] == "point", "duration"] = 0  # Set duration to 0 for 'point' events

        print(f"✅ Notes imported and processed from {notes_filename}. Sorted chronologically.")
        return event_df

    def save_datareader_object(self):
        """Saves the DataReader object as a pickle file."""
        pickle_filename = os.path.join(self.output_folder, 'data.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"DataReader object successfully saved to {pickle_filename}.")