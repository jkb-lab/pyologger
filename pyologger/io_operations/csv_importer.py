import os
import pandas as pd
from pyologger.io_operations.base_importer import BaseImporter
from pyologger.utils.time_manager import process_datetime

class CSVImporter(BaseImporter):
    """Generic CSV importer for any logger that outputs CSV-like tabular data."""

    def process_files(self, files, enforce_frequency=True):
        logger_id = self.logger_id
        tz = self.data_reader.deployment_info.get("Time Zone")

        # Only consider .csv files
        selected_files = [f for f in files if f.lower().endswith(".csv")]

        print(f"🔄 Processing {self.logger_manufacturer} logger {logger_id} with {len(selected_files)} CSV file(s).")

        if not selected_files:
            print(f"⚠ No valid CSV files found for {logger_id} ({self.logger_manufacturer}).")
            return None, None, None, None, None

        # --- 1. read each file once
        dfs = []
        for fname in selected_files:
            file_path = os.path.join(self.data_reader.data_folder, fname)
            df_part = self.read_csv(file_path)  # <- BaseImporter.read_csv()
            dfs.append(df_part)
            print(f"{self.logger_manufacturer} file for {logger_id}: {fname} - Successfully read.")

        # --- 2. concat once
        final_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # --- 3. rename columns once
        original_cols = final_df.columns.tolist()
        new_cols, channel_metadata = self.rename_channels(original_cols)
        final_df.rename(columns=new_cols, inplace=True)
        print(f"✅ Renamed columns for {logger_id}: {new_cols}")

        # --- 4. build datetime once
        final_df, datetime_metadata = process_datetime(final_df, time_zone=tz)
        self.data_reader.logger_info[logger_id]['datetime_created_from'] = datetime_metadata.get('datetime_created_from', None)
        self.data_reader.logger_info[logger_id]['fs'] = [datetime_metadata.get('fs', None)]

        # --- 5. group into sensors once
        sensor_groups, sensor_info = self.group_data_by_sensors(
            final_df,
            logger_id,
            channel_metadata
        )

        # --- 6. return once
        return final_df, channel_metadata, datetime_metadata, sensor_groups, sensor_info
