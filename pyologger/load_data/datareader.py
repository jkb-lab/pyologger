import os
import pickle
import pandas as pd
from datetime import datetime
from pyologger.utils.time_manager import process_datetime
from pyologger.process_data.sampling import *
from pyologger.load_data.metadata import *
from pyologger.io_operations import *
from pyologger.io_operations.base_exporter import BaseExporter
from pyologger.io_operations.cats_importer import CATSImporter
from pyologger.io_operations.ufi_importer import UFIImporter
from pyologger.io_operations.wc_importer import WCImporter  # if you still have a WC-specific importer; if not, remove this import
from pyologger.io_operations.ll_importer import LLImporter
from pyologger.io_operations.evolocus_importer import EvolocusImporter
from pyologger.io_operations.manitty_importer import ManittyImporter
from pyologger.io_operations.csv_importer import CSVImporter


class DataReader:
    """Reads and processes all raw files for a deployment, 1 pass per logger."""

    def __init__(self, dataset_folder: str, deployment_id: str, data_subfolder: str = None, montage_path: str = None):
        self.deployment_id = deployment_id
        self.deployment_folder = os.path.join(dataset_folder, deployment_id)
        self.data_folder = os.path.join(self.deployment_folder, data_subfolder) if data_subfolder else self.deployment_folder
        self.montage_path = montage_path

        # Selected deployment metadata (lat/lon/time zone/date)
        self.deployment_info = {}

        # Output folder
        self.output_folder = os.path.join(self.deployment_folder, 'outputs')

        # Dataset / animal tracking info for downstream metadata
        self.animal_info = {'Animal_ID': self.deployment_id.split('_', 1)[1] if '_' in self.deployment_id else None}
        self.dataset_info = {'Dataset_ID': os.path.basename(os.path.normpath(dataset_folder))}

        # Per-logger metadata like Manufacturer, Montage ID, fs, etc.
        self.logger_info = {}

        # Sensor data after grouping (populated by importer.group_data_by_sensors())
        self.sensor_data = {}
        self.sensor_info = {}

        # Event notes (00_Notes.xlsx) and extracted behaviors/events
        self.event_data = {}
        self.event_info = {}

        # Derived / higher-level signals (pitch/roll/HR/etc.)
        self.derived_data = {}
        self.derived_info = {}

        # Exporter helper
        self.exporter = BaseExporter(self)

        self._has_run = False

        print(f"DataReader initialized with deployment folder: {self.deployment_folder}")
        print(f"Using data folder: {self.data_folder}")

    def read_files(self, deployment_info, loggers_used, save_parq=True, save_netcdf=False):
        
        """
        Main ingestion routine:
          1. Cache deployment metadata
          2. Cache logger_info (Manufacturer, Montage ID, etc.)
          3. Import notes (events)
          4. Match files to each logger simply by substring match on Logger ID
          5. For each logger:
             - Choose the correct importer class
             - Run importer ONCE (no double-processing)
             - Save outputs
          6. Save DataReader snapshot (.pkl)
          7. [optional] Save NetCDF
        """

        print(f"🔄 Reading files from: {self.data_folder}")
        print(f"📁 Created data folder: {self.output_folder}")

        # 1. Store deployment metadata we care about
        self.deployment_info["Deployment Date"]      = deployment_info.get("Deployment Date")
        self.deployment_info["Deployment Latitude"]  = deployment_info.get("Deployment Latitude")
        self.deployment_info["Deployment Longitude"] = deployment_info.get("Deployment Longitude")
        self.deployment_info["Time Zone"]            = deployment_info.get("Time Zone")

        # 2. Record logger_info (Manufacturer, Montage ID, etc.)
        self.logger_info = {
            logger["Logger ID"]: {
                "ID": logger["Logger ID"],
                "Manufacturer": logger["Manufacturer"],
                "Montage ID": logger.get("Montage ID", "Unknown")
            }
            for logger in loggers_used
        }

        # 3. Import notes, if present
        self.event_data = self.import_notes()

        # 4. Build mapping logger_id -> files
        #    Rule: file goes to logger if the literal logger_id string is in filename
        raw_files = sorted(os.listdir(self.data_folder))
        logger_files = {logger["Logger ID"]: [] for logger in loggers_used}

        for fname in raw_files:
            for logger in loggers_used:
                lid = logger["Logger ID"]
                if lid in fname:
                    logger_files[lid].append(fname)

        # 5. Report file association
        print("📟 Loggers with files:")
        for lid, files in logger_files.items():
            if files:
                print(f"   {lid}: {files}")
            else:
                print(f"   {lid}: (no matching files)")

        # 6. Importer resolution logic
        def choose_importer(manufacturer: str, files_for_logger: list):
            """
            Priority rules:
              - CATS always uses CATSImporter.
              - For UFI:
                    if there's any .ube/.ubf -> UFIImporter
                    elif only .csv -> CSVImporter
              - For other manufacturers:
                    if any .csv -> CSVImporter
                    else fallback by known manufacturer class
            """

            # (a) CATS is special
            if manufacturer == "CATS":
                return CATSImporter

            # (b) UFI is also special (binary ECG tags)
            if manufacturer == "UFI":
                has_ube = any(f.lower().endswith((".ube", ".ubf")) for f in files_for_logger)
                has_csv = any(f.lower().endswith(".csv") for f in files_for_logger)
                if has_ube:
                    return UFIImporter
                if has_csv:
                    return CSVImporter
                # default if some weird future format shows up
                return UFIImporter

            # (c) Other known manufacturers
            # Wildlife Computers, Star Oddi, etc. are basically CSV loggers
            if any(f.lower().endswith(".csv") for f in files_for_logger):
                return CSVImporter

            if manufacturer == "Little Leonardo":
                return LLImporter
            if manufacturer == "Evolocus":
                return EvolocusImporter
            if manufacturer == "Manitty":
                return ManittyImporter

            # e.g. "Wildlife Computers", "Star Oddi"
            if manufacturer == "Wildlife Computers":
                return CSVImporter
            if manufacturer == "Star Oddi":
                return CSVImporter

            # final fallback
            return CSVImporter

        # 7. Process each logger once
        processed_logger_ids = set()

        for logger in loggers_used:
            lid = logger["Logger ID"]
            manufacturer = logger["Manufacturer"]

            if lid in processed_logger_ids:
                # already handled this logger
                continue
            processed_logger_ids.add(lid)

            files_for_this_logger = logger_files.get(lid, [])
            if not files_for_this_logger:
                print(f"⚠ No files to process for {lid} ({manufacturer}). Skipping.")
                continue

            importer_class = choose_importer(manufacturer, files_for_this_logger)
            importer_instance = importer_class(self, lid)

            print(f"🔄 Processing {lid} ({manufacturer}) with importer {importer_class.__name__}")
            print(f"   Files: {files_for_this_logger}")

            try:
                result = importer_instance.process_files(files_for_this_logger)
            except Exception as e:
                print(f"❌ ERROR while processing {lid} ({manufacturer}): {e}")
                continue

            if not result:
                print(f"⚠ Importer for {lid} returned no data.")
                continue

            # Two shapes are supported:
            #  A) multi-frequency dicts (Evolocus / Manitty)
            #  B) single dataframe (CSVImporter, UFIImporter, etc.)
            if (
                manufacturer in ["Evolocus", "Manitty"]
                and isinstance(result, tuple)
                and len(result) == 5
                and isinstance(result[0], dict)
            ):
                (final_dfs_dict,
                 channel_metadata_dict,
                 datetime_metadata_dict,
                 sensor_groups_dict,
                 sensor_info_dict) = result

                for freq, df in final_dfs_dict.items():
                    if df is not None and "datetime" in df.columns:
                        filename = f"{lid}_{freq}Hz.csv"
                        self.exporter.save_data(df, lid, filename, save_parq)

                print(f"✅ Processed and saved multi-frequency data for {lid} ({manufacturer}).")

            else:
                (final_df,
                 channel_metadata,
                 datetime_metadata,
                 sensor_groups,
                 sensor_info) = result

                if final_df is not None and "datetime" in final_df.columns:
                    outname = f"{lid}.csv"
                    self.exporter.save_data(final_df, lid, outname, save_parq)
                    print(f"✅ Processed and saved files for logger {lid} ({manufacturer}).")
                else:
                    print(f"⚠ Issue: {lid} ({manufacturer}) produced no final_df with 'datetime'.")

        # 8. Save the DataReader snapshot to a pickle file (once)
        self.save_datareader_object()

        # 9. Optionally write NetCDF for the whole deployment (once)
        if save_netcdf:
            netcdf_filename = os.path.join(
                self.deployment_folder,
                'outputs',
                f'{self.deployment_id}_00_processed.nc'
            )
            self.exporter.save_to_netcdf(self, netcdf_filename)
            print(f"📊 Saved deployment data to NetCDF: {netcdf_filename}")

    def check_outputs_folder(self):
        """
        Checks if the processed data files for the given loggers already exist.
        Returns True if outputs folder has a .pkl, else False.
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
        """Imports and processes {deployment_id}_00_Notes.xlsx in the data folder, if present."""

        if not self.deployment_info:
            print("❌ Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None

        notes_filename = f"{self.deployment_id}_00_Notes.xlsx"
        time_zone = self.deployment_info.get("Time Zone") or "UTC"

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

        # timestamp handling
        event_df, _ = process_datetime(event_df, time_zone=time_zone)

        if event_df["datetime"].isna().any():
            print(f"⚠️ WARNING: Some timestamps could not be parsed correctly.")

        # sort chronologically
        event_df = event_df.sort_values(by="datetime").reset_index(drop=True)

        # ensure 'duration' exists
        if "duration" not in event_df.columns:
            event_df["duration"] = 0

        # zero duration for 'point' events
        event_df.loc[event_df["type"] == "point", "duration"] = 0

        print(f"✅ Notes imported and processed from {notes_filename}. Sorted chronologically.")
        return event_df

    def save_datareader_object(self):
        """Saves the entire DataReader object for later quick reload."""
        pickle_filename = os.path.join(self.output_folder, 'data.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"DataReader object successfully saved to {pickle_filename}.")
