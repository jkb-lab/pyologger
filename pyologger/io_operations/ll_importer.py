from pyologger.io_operations.base_importer import BaseImporter
import os
import pandas as pd
from datetime import datetime

# TODO: Finish writing Little Leonardo-specific methods for .TXT files (will require getting data from Gdrive)
class LLImporter(BaseImporter):
    """Little Leonardo-specific processing."""

    def process_files(self, files):
        """Process CATS files, specifically handling .txt, ignoring .ubx, .ubc, and .bin files."""
        # Filter out .ubx, .ubc, and .bin files
        files = [f for f in files if not f.endswith(('.ubc', '.bin', '.ubx'))]
        
        # Parse the .txt file for expected intervals
        txt_file = next((f for f in files if f.endswith('.txt')), None)
        if txt_file:
            print(f"Parsing {txt_file} for expected sensor intervals.")
            self.parse_txt_for_intervals(os.path.join(self.data_reader.files_info['deployment_folder_path'], txt_file))

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
        final_df, datetime_metadata = self.data_reader.process_datetime(final_df, time_zone=self.data_reader.deployment_info['Time Zone'])
        self.data_reader.logger_info[self.logger_id]['datetime_metadata'] = datetime_metadata

        # Map data to sensors and return sensor information
        sensor_groups, sensor_info = self.group_data_by_sensors(final_df, self.logger_id, column_metadata)

        return final_df, column_metadata, datetime_metadata, sensor_groups, sensor_info

    def concatenate_and_save_csvs(self, csv_files):
        """Concatenates multiple CSV files into one DataFrame."""
        dfs = []
        for file in csv_files:
            file_path = os.path.join(self.data_reader.files_info['deployment_folder_path'], file)
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
        file_path = os.path.join(self.data_reader.files_info['deployment_folder_path'], txt_file)
        with open(file_path, 'r') as file:
            print(file.read())