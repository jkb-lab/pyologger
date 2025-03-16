from pyologger.io_operations.base_importer import BaseImporter
import os
from edfio import Edf, EdfSignal
import pandas as pd
from datetime import datetime

# TODO: Finish writing Neurologger-specific methods for .EDF files
class NLImporter(BaseImporter):
    """Neurologger-specific processing."""
    
    def __init__(self, edf_file_path, time_zone='UTC'):
        self.edf_file_path = edf_file_path
        self.time_zone = time_zone
        self.edf = None
        self.metadata = {}
        self.signal_groups = {}
        montage_id = self.data_reader.logger_info[self.logger_id]['montage_id']
    
    def load_edf_metadata(self):
        """Load metadata and signal information from the EDF file."""
        self.edf = Edf.read(self.edf_file_path)
        self.metadata = {
            'patient': self.edf.patient.__dict__,
            'recording': self.edf.recording.__dict__,
            'start_datetime': self.edf.start_time.astimezone(self.time_zone),
            'signals': {}
        }
        
        # Group signals by sampling frequency
        freq_dict = {}
        for signal in self.edf.signals:
            freq = signal.samples_per_second
            if freq not in freq_dict:
                freq_dict[freq] = []
            freq_dict[freq].append(signal)
        
        self.signal_groups = freq_dict
    
    def print_signal_info(self):
        """Print available signals and their sampling frequencies."""
        print(f"\n📁 Loaded EDF: {os.path.basename(self.edf_file_path)}")
        print(f"🕒 Start Time: {self.metadata['start_datetime']}")
        print(f"📊 Signals and Sampling Frequencies:")
        for freq, signals in self.signal_groups.items():
            print(f"  - {freq} Hz: {[s.label for s in signals]}")
    
    def extract_signal_data(self):
        """Convert signal groups into pandas DataFrames indexed by datetime."""
        dfs = {}
        for freq, signals in self.signal_groups.items():
            df_data = {}
            start_time = self.metadata['start_datetime']
            timestamps = pd.date_range(start=start_time, periods=len(signals[0].samples), freq=pd.Timedelta(seconds=1/freq))
            for signal in signals:
                df_data[signal.label] = signal.samples
            
            df = pd.DataFrame(df_data, index=timestamps)
            df.index = df.index.tz_localize(self.time_zone)
            dfs[freq] = df
        return dfs
    
    def run(self, test_run=True):
        """Run the import process."""
        self.load_edf_metadata()
        self.print_signal_info()
        if test_run:
            print("\n🔍 Test run mode enabled: Data not yet loaded.")
            return None
        return self.extract_signal_data()

    # def process_files(self, files):
    #     """Process CATS files, specifically handling .txt, ignoring .ubx, .ubc, and .bin files."""
    #     # Filter out .ubx, .ubc, and .bin files
    #     files = [f for f in files if not f.endswith(('.ubc', '.bin', '.ubx'))]
        
    #     # Parse the .txt file for expected intervals
    #     txt_file = next((f for f in files if f.endswith('.txt')), None)
    #     if txt_file:
    #         print(f"Parsing {txt_file} for expected sensor intervals.")
    #         self.parse_txt_for_intervals(os.path.join(self.data_reader.data_folder, txt_file))

    #     if not files:
    #         print(f"No valid files found for {self.logger_manufacturer} logger.")
    #         return None, None, None, None  # Return four None values

    #     # Remove the .txt files from the list after processing them
    #     files = [f for f in files if not f.endswith('.txt')]

    #     # Concatenate the remaining files into one DataFrame
    #     final_df = self.concatenate_and_save_csvs(files)

    #     # Rename columns
    #     final_df, column_metadata = self.rename_columns(final_df, self.logger_id, self.logger_manufacturer)
        
    #     # Process datetime and return metadata
    #     final_df, datetime_metadata = self.data_reader.process_datetime(final_df, time_zone=self.data_reader.deployment_info['Time Zone'])
    #     self.data_reader.logger_info[self.logger_id]['datetime_metadata'] = datetime_metadata

    #     # Map data to sensors and return sensor information
    #     sensor_groups, sensor_info = self.group_data_by_sensors(final_df, self.logger_id, column_metadata)

    #     return final_df, column_metadata, datetime_metadata, sensor_groups, sensor_info

    # def concatenate_and_save_csvs(self, csv_files):
    #     """Concatenates multiple CSV files into one DataFrame."""
    #     dfs = []
    #     for file in csv_files:
    #         file_path = os.path.join(self.data_reader.data_folder, file)
    #         try:
    #             data = self.data_reader.read_csv(file_path)
    #             dfs.append(data)
    #             print(f"{self.logger_manufacturer} file: {file} - Successfully processed.")
    #         except Exception as e:
    #             print(f"Error processing file {file}: {e}")

    #     if len(dfs) > 1:
    #         concatenated_df = pd.concat(dfs, ignore_index=True)
    #     else:
    #         concatenated_df = dfs[0]

    #     return concatenated_df

    # def print_txt_content(self, txt_file):
    #     """Prints the content of a .txt file."""
    #     file_path = os.path.join(self.data_reader.data_folder, txt_file)
    #     with open(file_path, 'r') as file:
    #         print(file.read())