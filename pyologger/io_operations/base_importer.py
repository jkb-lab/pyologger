import pandas as pd
import numpy as np
import re
import json
import pytz
from datetime import datetime
from pyologger.utils.time_manager import *

class BaseImporter:
    """Base class for handling manufacturer-specific processing."""

    def __init__(self, data_reader, logger_id, manufacturer):
        self.logger_id = logger_id
        self.logger_manufacturer = manufacturer
        self.data_reader = data_reader
        self.expected_frequencies = {}  # This will hold the expected frequencies parsed from .txt file
        montage_path = self.data_reader.montage_path

        # Load the custom JSON mapping for column names if deployment folder is provided
        if montage_path:
            self.load_custom_mapping(montage_path)

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
    
    def load_custom_mapping(self, montage_path):
        """Loads custom JSON mapping for column names."""
        try:
            with open(montage_path, 'r') as json_file:
                self.column_mapping = json.load(json_file)
                print(f"Custom column mapping loaded from {montage_path}")
                
                # Check if the mapping contains expected keys
                if not isinstance(self.column_mapping, dict):
                    raise ValueError("Column mapping is not a dictionary.")
                if 'CATS' not in self.column_mapping and 'UFI' not in self.column_mapping:
                    raise ValueError("Expected keys 'CATS' or 'UFI' are missing in the column mapping.")
                print("Column mapping verified successfully.")
        except FileNotFoundError:
            print(f"Custom mapping file not found at {montage_path}. Proceeding without it.")
            self.column_mapping = None
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading or verifying JSON from {montage_path}: {e}")
            self.column_mapping = None

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
            sensor_type = mapping_info.get("sensor_type", "extra").strip().lower()  # Use 'sensor_type' instead of 'sensor'

            print(f"Original name: {original_name}, Clean name: {clean_name}, Mapped name: {mapped_name}, Sensor type: {sensor_type}")

            column_metadata[mapped_name] = {
                "original_name": original_name,
                "unit": unit or "unknown",
                "sensor": sensor_type  # Updated to use 'sensor_type'
            }
            new_columns[original_name] = mapped_name

        # Rename the columns in the DataFrame
        df.rename(columns=new_columns, inplace=True)

        return df, column_metadata
    
    def group_data_by_sensors(self, df, logger_id, column_metadata):
        """Groups data columns to sensors and downsamples based on expected frequencies."""
        sensor_groups = {}
        sensor_info = {}

        for sensor_name in set(v['sensor'].strip().lower() for v in column_metadata.values()):
            if sensor_name == 'extra':
                continue  # Skip 'extra' sensor type

            # Skip already processed sensors
            if sensor_name in self.data_reader.sensor_data:
                print(f"Sensor '{sensor_name}' has already been processed. Skipping reprocessing.")
                continue

            # Group columns by sensor
            sensor_cols = [col for col, meta in column_metadata.items() if meta['sensor'].strip().lower() == sensor_name]
            sensor_df = df[['datetime'] + sensor_cols].copy()

            # Determine the data type of the sensor columns
            data_type = sensor_df[sensor_cols].dtypes.iloc[0]  # Assuming all sensor columns have the same dtype
            data_type_str = str(data_type)

            # Standardized metadata collection
            start_time = sensor_df['datetime'].iloc[0]
            end_time = sensor_df['datetime'].iloc[-1]
            max_value = sensor_df[sensor_cols].max().max()
            min_value = sensor_df[sensor_cols].min().min()
            mean_value = sensor_df[sensor_cols].mean().mean()

            # Get the original unit from the column metadata
            original_units = list({column_metadata[col]['unit'] for col in sensor_cols})

            # Get sampling frequency for each sensor (logger-specific methods or default)
            expected_frequency = self.expected_frequencies.get(sensor_name)
            if not expected_frequency and sensor_name == 'ecg':
                # For UFI ECG, use the overall frequency (e.g., determined earlier)
                expected_frequency = int((self.data_reader.logger_info[logger_id]['datetime_metadata']['fs']))

            if expected_frequency:
                # Calculate the actual sampling frequency based on the 'datetime' column
                current_frequency = round(1 / df['datetime'].diff().dt.total_seconds().mean())
                # Calculate the decimation factor
                decimation_factor = max(1, int(round(current_frequency / expected_frequency)))

                # Print the downsampling information
                if decimation_factor > 1:
                    print(f"Downsampling {sensor_name} data by {decimation_factor} X from {current_frequency:.2f} Hz to {expected_frequency:.2f}Hz.")
                    # Apply downsampling
                    sensor_df = sensor_df.iloc[::decimation_factor]
                else:
                    print(f"No downsampling needed for {sensor_name}. Expected frequency is close to the actual frequency {current_frequency:.2f} Hz.")
            
            # Store the processed data and standardized metadata
            self.data_reader.sensor_data[sensor_name] = sensor_df
            self.data_reader.sensor_info[sensor_name] = {
                'channels': sensor_cols,
                'metadata': {col: column_metadata[col] for col in sensor_cols},
                'sensor_start_datetime': start_time,
                'sensor_end_datetime': end_time,
                'max_value': max_value,
                'min_value': min_value,
                'mean_value': mean_value,
                'data_type': data_type_str,
                'original_units': original_units,
                'sampling_frequency': expected_frequency,
                'logger_id': self.logger_id,
                'logger_manufacturer': self.logger_manufacturer,
                'processing_step': 'Raw data uploaded',
                'last_updated': datetime.now().astimezone(pytz.timezone(self.data_reader.deployment_info['Time Zone'])),
                'details': 'Initial, raw sensor-specific data and metadata loaded.',
            }

        # Print final mapping and downsampling results
        for sensor_name, df in self.data_reader.sensor_data.items():
            print(f"Sensor '{sensor_name}' data processed and stored with shape {df.shape}.")

        return sensor_groups, sensor_info

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
                
                print(f"Sensor info read from txt: {sensor_info}")

                # Use the correct sub-dictionary for column mapping, assuming you're processing data for a CATS logger
                mapping_sub_dict = self.column_mapping.get(self.logger_manufacturer, {})

                for sensor_id, sensor_name, interval in sensor_info:
                    normalized_sensor_name = sensor_name.strip().lower()  # Normalize the sensor name

                    # Reverse lookup in the mapping sub-dictionary to find the matching manufacturer_sensor_name
                    for clean_name, mapping in mapping_sub_dict.items():
                        # Get the manufacturer sensor name associated with this column
                        manufacturer_sensor_name = mapping['manufacturer_sensor_name'].strip().lower()  # Use 'manufacturer_sensor_name'

                        # Match based on normalized sensor name and manufacturer sensor name
                        if manufacturer_sensor_name == normalized_sensor_name:
                            frequency = int(interval)
                            # Store the frequency using the sensor_type (normalized)
                            sensor_type = mapping['sensor_type'].strip().lower()
                            self.expected_frequencies[sensor_type] = frequency  
                            print(f"Sensor '{sensor_type}' found matching sensor name in config: '{sensor_name}'. Expected sampling frequency: {frequency} Hz")
                            break
                    else:
                        print(f"Sensor name '{sensor_name}' not found in column mapping. Ignoring this sensor.")

            else:
                print("No 'activated sensors' section found in the file.")

        except Exception as e:
            print(f"Failed to parse {txt_file_path} due to: {e}")

    def process_files(self, files):
        """Process files in the subclass. This should be overridden."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def concatenate_and_save_csvs(self, csv_files):
        """Base method for concatenating and saving CSVs."""
        raise NotImplementedError("This method should be implemented by subclasses.")



