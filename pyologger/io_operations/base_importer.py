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
        self.expected_frequencies = {}  # Stores expected signal frequencies from .txt files
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
        """Maps original channel IDs to standardized channel IDs."""
        channel_metadata = {}
        new_channels = {}

        print(f"🔄 Renaming channels for {self.logger_manufacturer} logger with montage ID {self.montage_id}.")

        # Be defensive: self.montage may be None if no mapping file was provided or it failed to load.
        if not getattr(self, 'montage', None):
            print(f"⚠️ No montage mapping loaded for logger {self.logger_id} ({self.logger_manufacturer}). Falling back to identity mapping.")
            self.montage = {}
        else:
            try:
                print(f"📜 Available mapping keys: {list(self.montage.keys())}")  # Show available keys in the mapping
            except Exception:
                # Defensive fallback if montage is not a dict-like object
                print("⚠️ Montage mapping is not iterable. Falling back to empty mapping.")
                self.montage = {}

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

            # Dictionary lookup: if no mapping entry exists, fall back to using the cleaned name
            if clean_name in self.montage:
                mapping_info = self.montage[clean_name] or {}
                print(f"🎯 Found match in montage: {mapping_info}")  # Show full mapping info
            else:
                mapping_info = {}
                print(f"ℹ️ No mapping for '{clean_name}' — using cleaned name as standardized id.")

            mapped_name = mapping_info.get("standardized_channel_id", clean_name)
            parent_signal = mapping_info.get("parent_signal", "extra").strip().lower()
            original_unit = mapping_info.get("original_unit", "extra").strip().lower()

            print(f"📝 Dictionary: original name: {original_name} → standardized name: {mapped_name} (Signal type: {parent_signal})")

            channel_metadata[mapped_name] = {
                "original_name": original_name,
                "unit": original_unit or "unknown",
                "parent_signal": parent_signal
            }
            new_channels[original_name] = mapped_name

        print(f"✅ Final renamed channels: {new_channels}")
        return new_channels, channel_metadata


    def group_data_by_signals(self, df, logger_id, channel_metadata):
        """Groups data columns to signals and downsamples based on expected frequencies."""
        signal_groups = {}
        signal_info = {}

        for signal_name in set(v['parent_signal'].strip().lower() for v in channel_metadata.values()):
            if signal_name == 'extra':
                continue  # Skip 'extra' signal type

            if signal_name in self.data_reader.signal_data:
                print(f"Sensor '{signal_name}' has already been processed. Skipping reprocessing.")
                continue

            # Group columns by signal
            signal_cols = [col for col, meta in channel_metadata.items() if meta['parent_signal'].strip().lower() == signal_name]
            signal_df = df[['datetime'] + signal_cols].copy()

            # Check if all signal columns are numeric
            non_numeric_cols = signal_df[signal_cols].select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"❌ Skipping signal '{signal_name}': non-numeric data found in columns: {non_numeric_cols}")
                continue

            # Determine the data type of the signal columns
            data_type = signal_df[signal_cols].dtypes.iloc[0]
            data_type_str = str(data_type)

            # Standardized metadata collection
            start_time = signal_df['datetime'].iloc[0]
            end_time = signal_df['datetime'].iloc[-1]
            max_value = signal_df[signal_cols].max().max()
            min_value = signal_df[signal_cols].min().min()
            mean_value = signal_df[signal_cols].mean().mean()

            # Get the original unit from the column metadata
            original_units = {channel_metadata[col]['unit'] for col in signal_cols}
            if len(original_units) > 1:
                warnings.warn(f"Conflicting units found for signal '{signal_name}': {original_units}. Using the first one.")
            original_unit = original_units.pop() if original_units else "unknown"

            # Calculate current frequency - beware this does not fix gaps, just uses the first few values to calculate freq.
            original_frequency = calculate_sampling_frequency(signal_df['datetime'].head()) # round(1 / signal_df['datetime'].diff().dt.total_seconds().mean())
            print(f"Original frequency for {signal_name}: {original_frequency} Hz")

            expected_frequency = self.expected_frequencies.get(signal_name)
            if not expected_frequency and self.logger_manufacturer == 'LL':
                expected_frequency = int(self.data_reader.logger_info[logger_id]['fs'])

            max_desired_frequency = None
            if self.logger_manufacturer in ['Evolocus', 'Manitty', 'UFI']:
                max_freq_lookup = {'eeg': 100, 'eog': 100, 'ecg': 250, 'emg': 250}
                max_desired_frequency = max_freq_lookup.get(signal_name, None)

            downsample_target = max_desired_frequency or expected_frequency
            if not expected_frequency and not max_desired_frequency:
                print(f"⚠️ No frequency target found for {signal_name}. Using original frequency {original_frequency} Hz.")

            if downsample_target and downsample_target < original_frequency:
                decimation_factor = max(1, int(round(original_frequency / downsample_target)))
                print(f"Downsampling {signal_name} by {decimation_factor}x from {original_frequency:.2f}Hz to {downsample_target:.2f}Hz.")
                signal_df = signal_df.iloc[::decimation_factor]
                # Calculate new frequency after downsampling - beware this does not fix gaps, just uses the first few values to calculate freq.
                new_frequency = calculate_sampling_frequency(signal_df['datetime'].head()) # round(1 / signal_df['datetime'].diff().dt.total_seconds().mean())
                print(f"New frequency after downsampling: {new_frequency} Hz")
            else:
                new_frequency = original_frequency
                print(f"No downsampling required for {signal_name}. Current: {original_frequency:.2f}Hz, Target: {downsample_target}Hz")

            details = 'Initial, raw signal-specific data and metadata loaded.'
            if new_frequency != original_frequency:
                details += f' Original frequency: {original_frequency} Hz; downsampled to {new_frequency} Hz.'
            else:
                details += f' Original frequency: {original_frequency} Hz; no downsampling applied.'

            self.data_reader.signal_data[signal_name] = signal_df
            self.data_reader.signal_info[signal_name] = {
                'channels': signal_cols,
                'metadata': {col: channel_metadata[col] for col in signal_cols},
                'signal_start_datetime': start_time,
                'signal_end_datetime': end_time,
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

        for signal_name, df in self.data_reader.signal_data.items():
            print(f"Sensor '{signal_name}' data processed and stored with shape {df.shape}.")

        return signal_groups, signal_info


    def process_files(self, files):
        """Process files in the subclass. This should be overridden."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def concatenate_and_save_csvs(self, csv_files):
        """Base method for concatenating and saving CSVs."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_expected_frequencies(self, parsed_signals, enforce_frequency=True):
        """
        Matches parsed signals with the channel mapping and sets expected frequencies.
        
        Args:
            parsed_signals (dict): Sensor name -> expected interval (Hz)
            enforce_frequency (bool): Whether to enforce setting expected frequencies. Default: True.
        """
        if not parsed_signals:
            print("⚠ No signals parsed from txt file, skipping frequency matching.")
            return

        if not self.montage:
            print(f"⚠ Warning: No valid channel mapping found for Manufacturer '{self.logger_manufacturer}' with Montage ID '{self.montage_id}'.")
            return

        print(f"🔍 Matching parsed signals with channel mapping. Enforce frequency: {enforce_frequency}")

        for signal_name, frequency in parsed_signals.items():
            found_match = False
            for clean_name, mapping in self.montage.items():
                manufacturer_signal_name = mapping['manufacturer_signal_name'].strip().lower()

                if manufacturer_signal_name == signal_name:
                    parent_signal = mapping['parent_signal'].strip().lower()

                    if enforce_frequency:
                        self.expected_frequencies[parent_signal] = frequency  
                        print(f"✅ Matched '{signal_name}' -> '{parent_signal}' with expected frequency: {frequency} Hz.")
                    else:
                        print(f"🔍 Matched '{signal_name}' -> '{parent_signal}', but skipping frequency enforcement.")

                    found_match = True
                    break  # Stop once a match is found

            if not found_match:
                print(f"⚠ Sensor name '{signal_name}' not found in channel mapping. Ignoring this signal.")



