import pandas as pd
import numpy as np
import re
import warnings
import json
import pytz
from datetime import datetime
from pyologger.utils.time_manager import *

class BaseImporter:
    """Base class for handling manufacturer-specific processing."""

    def __init__(self, data_reader, logger_id):
        self.data_reader = data_reader
        self.logger_id = logger_id
        self.logger_manufacturer = self.data_reader.logger_info[logger_id]['Manufacturer']
        self.montage_id = self.data_reader.logger_info[logger_id]['Montage ID']
        self.expected_frequencies = {}  # Stores expected sensor frequencies from .txt files
        self.montage_path = self.data_reader.montage_path

        # Load the custom JSON mapping for column names if available
        if self.montage_path:
            self.load_custom_mapping()
        
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

    def import_netcdf(data_reader, filepath):
            """Imports a NetCDF file to the pickle format used by pyologger."""
            
            print(f"NetCDF file imported from {filepath} and returning pickle object.")

    def import_edf(data_reader, filepath):
            """Imports a NetCDF file to the pickle format used by pyologger."""
            
            print(f"NetCDF file imported from {filepath} and returning pickle object.")
    
    def load_custom_mapping(self):
        """Loads custom JSON mapping for column names based on manufacturer and montage ID."""
        try:
            with open(self.montage_path, 'r') as json_file:
                full_mapping = json.load(json_file)
                print(f"Custom column mapping loaded from {self.montage_path}")

                # Ensure it's a dictionary
                if not isinstance(full_mapping, dict):
                    raise ValueError("Column mapping JSON is not a dictionary.")

                # Validate manufacturer
                manufacturer = self.logger_manufacturer
                if manufacturer not in full_mapping:
                    raise ValueError(f"Manufacturer '{manufacturer}' not found in column mapping.")

                # Validate montage ID
                montage_id = self.montage_id
                if montage_id not in full_mapping[manufacturer]:
                    raise ValueError(f"Montage ID '{montage_id}' not found under manufacturer '{manufacturer}'.")

                # Extract only the relevant part of the mapping
                self.montage = full_mapping[manufacturer][montage_id]
                print(f"Column mapping loaded for manufacturer '{manufacturer}', montage '{montage_id}'.")
                print(f"Mapping content {self.montage}.")
        except FileNotFoundError:
            print(f"Custom mapping file not found at {self.montage_path}. Proceeding without it.")
            self.montage = None
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading or verifying JSON from {self.montage_path}: {e}")
            self.montage = None

    def rename_channels(self, channel_names):
        """Maps raw channel names to standardized sensor names."""
        channel_metadata = {}
        new_channels = {}

        print(f"🔄 Renaming channels for {self.logger_manufacturer} logger with montage ID {self.montage_id}.")
        print(f"📜 Available mapping keys: {list(self.montage.keys())}")  # Show available keys in the mapping

        for original_name in channel_names:
            # Take out spaces from channel names and make lowercase
            clean_name = original_name.strip().lower().replace(" ", "_").replace(".", "")  # Normalize

            # Extract unit (if present)
            unit = None
            if "[" in clean_name and "]" in clean_name:
                name, unit = clean_name.split("[", 1)
                unit = unit.replace("]", "").strip().lower()
                clean_name = name.strip("_")

            if "(" in clean_name and ")" in clean_name:
                name, unit = clean_name.split("(", 1)
                unit = unit.replace(")", "").strip().lower()
                clean_name = f"{name.strip('_')}_{unit}"

            # Ensure local/UTC times are distinct
            if "local" in original_name.lower() and "local" not in clean_name:
                clean_name = f"{clean_name}_local"
            elif "utc" in original_name.lower() and "utc" not in clean_name:
                clean_name = f"{clean_name}_utc"

            # Dictionary lookup
            if clean_name in self.montage:
                mapping_info = self.montage[clean_name]
                print(f"🎯 Found match in montage: {mapping_info}")  # Show full mapping info
            else:
                print(f"⚠ Warning: {clean_name} not found in channel mapping. Skipping.")
                continue

            mapped_name = mapping_info.get("standardized_channel_id", clean_name)
            sensor_type = mapping_info.get("standardized_sensor_type", "extra").strip().lower()

            print(f"📝 Dictionary: original name: {original_name} → standardized name: {mapped_name} (Sensor type: {sensor_type})")

            channel_metadata[mapped_name] = {
                "original_name": original_name,
                "unit": unit or "unknown",
                "sensor": sensor_type
            }
            new_channels[original_name] = mapped_name

        print(f"✅ Final renamed channels: {new_channels}")
        return new_channels, channel_metadata


    def group_data_by_sensors(self, df, logger_id, channel_metadata):
        """Groups data columns to sensors and downsamples based on expected frequencies."""
        sensor_groups = {}
        sensor_info = {}

        for sensor_name in set(v['sensor'].strip().lower() for v in channel_metadata.values()):
            if sensor_name == 'extra':
                continue  # Skip 'extra' sensor type

            if sensor_name in self.data_reader.sensor_data:
                print(f"Sensor '{sensor_name}' has already been processed. Skipping reprocessing.")
                continue

            # Group columns by sensor
            sensor_cols = [col for col, meta in channel_metadata.items() if meta['sensor'].strip().lower() == sensor_name]
            sensor_df = df[['datetime'] + sensor_cols].copy()

            # Check if all sensor columns are numeric
            non_numeric_cols = sensor_df[sensor_cols].select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"❌ Skipping sensor '{sensor_name}': non-numeric data found in columns: {non_numeric_cols}")
                continue

            # Determine the data type of the sensor columns
            data_type = sensor_df[sensor_cols].dtypes.iloc[0]
            data_type_str = str(data_type)

            # Standardized metadata collection
            start_time = sensor_df['datetime'].iloc[0]
            end_time = sensor_df['datetime'].iloc[-1]
            max_value = sensor_df[sensor_cols].max().max()
            min_value = sensor_df[sensor_cols].min().min()
            mean_value = sensor_df[sensor_cols].mean().mean()

            # Get the original unit from the column metadata
            original_units = {channel_metadata[col]['unit'] for col in sensor_cols}
            if len(original_units) > 1:
                warnings.warn(f"Conflicting units found for sensor '{sensor_name}': {original_units}. Using the first one.")
            original_unit = original_units.pop() if original_units else "unknown"

            # Calculate current frequency - beware this does not fix gaps, just uses the first few values to calculate freq.
            original_frequency = calculate_sampling_frequency(sensor_df['datetime'].head()) # round(1 / sensor_df['datetime'].diff().dt.total_seconds().mean())
            print(f"Original frequency for {sensor_name}: {original_frequency} Hz")

            expected_frequency = self.expected_frequencies.get(sensor_name)
            if not expected_frequency and self.logger_manufacturer == 'LL':
                expected_frequency = int(self.data_reader.logger_info[logger_id]['fs'])

            max_desired_frequency = None
            if self.logger_manufacturer in ['Evolocus', 'UFI']:
                max_freq_lookup = {'eeg': 100, 'eog': 100, 'ecg': 250, 'emg': 250}
                max_desired_frequency = max_freq_lookup.get(sensor_name, None)

            downsample_target = max_desired_frequency or expected_frequency
            if not expected_frequency and not max_desired_frequency:
                print(f"⚠️ No frequency target found for {sensor_name}. Using original frequency {original_frequency} Hz.")

            if downsample_target and downsample_target < original_frequency:
                decimation_factor = max(1, int(round(original_frequency / downsample_target)))
                print(f"Downsampling {sensor_name} by {decimation_factor}x from {original_frequency:.2f}Hz to {downsample_target:.2f}Hz.")
                sensor_df = sensor_df.iloc[::decimation_factor]
                # Calculate new frequency after downsampling - beware this does not fix gaps, just uses the first few values to calculate freq.
                new_frequency = calculate_sampling_frequency(sensor_df['datetime'].head()) # round(1 / sensor_df['datetime'].diff().dt.total_seconds().mean())
                print(f"New frequency after downsampling: {new_frequency} Hz")
            else:
                new_frequency = original_frequency
                print(f"No downsampling required for {sensor_name}. Current: {original_frequency:.2f}Hz, Target: {downsample_target}Hz")

            details = 'Initial, raw sensor-specific data and metadata loaded.'
            if new_frequency != original_frequency:
                details += f' Original frequency: {original_frequency} Hz; downsampled to {new_frequency} Hz.'
            else:
                details += f' Original frequency: {original_frequency} Hz; no downsampling applied.'

            self.data_reader.sensor_data[sensor_name] = sensor_df
            self.data_reader.sensor_info[sensor_name] = {
                'channels': sensor_cols,
                'metadata': {col: channel_metadata[col] for col in sensor_cols},
                'sensor_start_datetime': start_time,
                'sensor_end_datetime': end_time,
                'max_value': float(max_value),
                'min_value': float(min_value),
                'mean_value': float(mean_value),
                'data_type': data_type_str,
                'original_units': original_unit,
                'units': original_unit,
                'original_sampling_frequency': original_frequency,
                'sampling_frequency': new_frequency,
                'logger_id': self.logger_id,
                'logger_manufacturer': self.logger_manufacturer,
                'processing_step': 'Raw data uploaded',
                'last_updated': pd.Timestamp(datetime.now().astimezone(pytz.timezone(self.data_reader.deployment_info['Time Zone']))),
                'details': details,
            }

        for sensor_name, df in self.data_reader.sensor_data.items():
            print(f"Sensor '{sensor_name}' data processed and stored with shape {df.shape}.")

        return sensor_groups, sensor_info


    def process_files(self, files):
        """Process files in the subclass. This should be overridden."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def concatenate_and_save_csvs(self, csv_files):
        """Base method for concatenating and saving CSVs."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_expected_frequencies(self, parsed_sensors, enforce_frequency=True):
        """
        Matches parsed sensors with the channel mapping and sets expected frequencies.
        
        Args:
            parsed_sensors (dict): Sensor name -> expected interval (Hz)
            enforce_frequency (bool): Whether to enforce setting expected frequencies. Default: True.
        """
        if not parsed_sensors:
            print("⚠ No sensors parsed from txt file, skipping frequency matching.")
            return

        if not self.montage:
            print(f"⚠ Warning: No valid channel mapping found for Manufacturer '{self.logger_manufacturer}' with Montage ID '{self.montage_id}'.")
            return

        print(f"🔍 Matching parsed sensors with channel mapping. Enforce frequency: {enforce_frequency}")

        for sensor_name, frequency in parsed_sensors.items():
            found_match = False
            for clean_name, mapping in self.montage.items():
                manufacturer_sensor_name = mapping['manufacturer_sensor_name'].strip().lower()

                if manufacturer_sensor_name == sensor_name:
                    sensor_type = mapping['standardized_sensor_type'].strip().lower()

                    if enforce_frequency:
                        self.expected_frequencies[sensor_type] = frequency  
                        print(f"✅ Matched '{sensor_name}' -> '{sensor_type}' with expected frequency: {frequency} Hz.")
                    else:
                        print(f"🔍 Matched '{sensor_name}' -> '{sensor_type}', but skipping frequency enforcement.")

                    found_match = True
                    break  # Stop once a match is found

            if not found_match:
                print(f"⚠ Sensor name '{sensor_name}' not found in channel mapping. Ignoring this sensor.")



