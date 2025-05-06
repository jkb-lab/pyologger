from pyologger.io_operations.base_importer import BaseImporter
import os
import re
import pandas as pd
from pyologger.utils.time_manager import process_datetime

class CATSImporter(BaseImporter):
    """CATS-specific processing."""

    def process_files(self, files, enforce_frequency=True):
        """
        Process CATS files, handling .txt separately and ignoring .ubx, .ubc, .bin, .cfg files.
        
        Args:
            files (list): List of files to process.
            enforce_frequency (bool): Whether to enforce expected frequencies. Default is True.
        """
        # Filter out unwanted files
        files = [f for f in files if not f.endswith(('.ubc', '.bin', '.ubx', '.cfg', '.obs', '.pos', '.stat', '.nsl', '.kml', '.conv'))]

        # Step 1: Parse .txt file for sensor intervals
        txt_file = next((f for f in files if f.endswith('.txt')), None)
        parsed_sensors = {}
        if txt_file:
            print(f"🔍 Parsing {txt_file} for expected sensor intervals.")
            parsed_sensors = self.parse_txt_file(os.path.join(self.data_reader.data_folder, txt_file))

        # Step 2: Set expected frequencies (can be overridden)
        self.set_expected_frequencies(parsed_sensors, enforce_frequency=enforce_frequency)

        # Remove .txt files from the list after processing them
        files = [f for f in files if not f.endswith('.txt')]

        if not files:
            print(f"⚠ No valid files found for {self.logger_manufacturer} logger.")
            return None, None, None, None, None  # Return five None values

        # Step 3: Concatenate remaining files into one DataFrame
        final_df = self.concatenate_and_save_csvs(files)
        
        # Step 4: Rename columns
        original_channel_names = final_df.columns.tolist()
        new_channel_names, channel_metadata = self.rename_channels(original_channel_names)
        final_df.rename(columns=new_channel_names, inplace=True)
        print(f"✅ Renamed columns: {new_channel_names}")

        # Step 5: Process datetime and return metadata
        final_df, datetime_metadata = process_datetime(final_df, time_zone=self.data_reader.deployment_info['Time Zone'])
        self.data_reader.logger_info[self.logger_id]['datetime_created_from'] = datetime_metadata.get('datetime_created_from', None)
        self.data_reader.logger_info[self.logger_id]['fs'] = [datetime_metadata.get('fs', None)]

        # Step 6: Map data to sensors and return sensor information
        sensor_groups, sensor_info = self.group_data_by_sensors(final_df, self.logger_id, channel_metadata)

        return final_df, channel_metadata, datetime_metadata, sensor_groups, sensor_info

    def concatenate_and_save_csvs(self, csv_files):
        """Concatenates multiple CSV files into one DataFrame."""
        dfs = []
        for file in csv_files:
            file_path = os.path.join(self.data_reader.data_folder, file)
            try:
                data = self.read_csv(file_path)
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
        file_path = os.path.join(self.data_reader.data_folder, txt_file)
        with open(file_path, 'r') as file:
            print(file.read())

    def parse_txt_file(self, txt_file_path):
        """Parses the .txt file to extract sensor names and sampling intervals."""
        print(f"🔍 Attempting to parse intervals from {txt_file_path}")

        try:
            with open(txt_file_path, 'r') as file:
                content = file.read()

            # Extract sensor information from the '[activated sensors]' section
            activated_sensors_section = re.search(r'\[activated sensors\](.*?)\n\n', content, re.DOTALL)
            if not activated_sensors_section:
                print("⚠ No 'activated sensors' section found in the file.")
                return {}

            activated_sensors_content = activated_sensors_section.group(1)

            # Find all sensors' names and their corresponding intervals
            sensor_info = re.findall(r'(\d{2})_name=(.*?)\n.*?\1_interval=(\d+)', activated_sensors_content, re.DOTALL)
            if not sensor_info:
                print("⚠ No sensor information found in the file. Please check the file format.")
                return {}

            print(f"✅ Parsed sensor info from txt: {sensor_info}")

            # Convert to dictionary {sensor_name: interval}
            parsed_sensors = {sensor_name.strip().lower(): int(interval) for _, sensor_name, interval in sensor_info}
            return parsed_sensors

        except Exception as e:
            print(f"❌ Failed to parse {txt_file_path} due to: {e}")
            return {}
