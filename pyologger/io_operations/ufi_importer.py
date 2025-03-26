from pyologger.io_operations.base_importer import BaseImporter
import os
import pytz
import struct
import pandas as pd
from datetime import datetime, timedelta
from pyologger.utils.time_manager import *

class UFIImporter(BaseImporter):
    """UFI-specific processing."""

    def process_files(self, files):
        """Process UFI files, specifically looking for .ube files."""
        # Find the first .ube file in the files list
        ube_file = next((f for f in files if f.endswith('.ube')), None)
        
        if ube_file:
            print(f"Processing .ube file: {ube_file}")
            final_df = self.process_ube_file(ube_file)

            # Rename columns
            print(final_df)

            # Step 4: Rename columns
            original_channel_names = final_df.columns.tolist()
            new_channel_names, channel_metadata = self.rename_channels(original_channel_names)
            final_df.rename(columns=new_channel_names, inplace=True)
            print(f"✅ Renamed columns: {new_channel_names}")
            
            # Process datetime and return metadata
            final_df, datetime_metadata = process_datetime(final_df, time_zone=self.data_reader.deployment_info['Time Zone'])
            self.data_reader.logger_info[self.logger_id]['datetime_created_from'] = datetime_metadata.get('datetime_created_from', None)
            self.data_reader.logger_info[self.logger_id]['fs'] = datetime_metadata.get('fs', None)
            
            # Map data to sensors and return sensor information
            sensor_groups, sensor_info = self.group_data_by_sensors(final_df, self.logger_id, channel_metadata)

            return final_df, channel_metadata, datetime_metadata, sensor_groups, sensor_info
        else:
            print(f"No .ube file found for UFI logger.")
            return None, None, None, None, None  # Return four None values

    def process_ube_file(self, ube_file):
        """Processes a UBE file and extracts data."""
        file_path = os.path.join(self.data_reader.data_folder, ube_file)
        print(f"Processing UBE file: {file_path}")

        try:
            with open(file_path, 'rb') as file:
                ube_raw = file.read()

            # Parse the download timestamp
            dl_time_str = ube_raw[0:32].decode('utf-8').strip()
            print(f"Parsed download timestamp string: '{dl_time_str}'")
            try:
                dl_time = datetime.strptime(dl_time_str, "%m-%d-%Y, %H:%M:%S")
            except ValueError as e:
                print(f"Error parsing timestamp: '{dl_time_str}' - {e}")
                return pd.DataFrame(), {}

            # Extract the record start time components
            mdhms = struct.unpack('BBBBB', ube_raw[32:37])
            record_month, record_day, record_hour, record_minute, record_second = mdhms

            # Determine the year for the record start time
            record_year = dl_time.year
            record_start = datetime(record_year, record_month, record_day, record_hour, record_minute, record_second)

            # Handle cases where the recording is from the previous year
            if record_start > dl_time:
                record_start = record_start.replace(year=record_year - 1)

            timezone = self.data_reader.deployment_info.get('Time Zone')
            if timezone:
                tz = pytz.timezone(timezone)
                record_start = tz.localize(record_start)
                print(f"Deployment start time (localized): {record_start}")

            rec_date = pd.to_datetime(self.data_reader.deployment_info['Deployment Date']).date()
            if record_start.date() != rec_date:
                print(f"Error: Deployment start date {record_start.date()} does not match Deployment Date {rec_date}.")
                return pd.DataFrame(), {}

            # Extract data after the header
            data_raw = ube_raw[40:]

            ecg_channel = 0x20
            ecg_data = []

            for i in range(0, len(data_raw), 2):
                channel = data_raw[i]
                value = data_raw[i + 1]
                if (channel & 0xF0) == ecg_channel:
                    ecg_value = (channel & 0x0F) << 8 | value
                    ecg_data.append(ecg_value)

            print(f"Total ECG data points: {len(ecg_data)}")

            if len(ecg_data) == 0:
                print("No ECG data extracted. Exiting.")
                return pd.DataFrame(), {}

            # Generate the datetime column for the ECG data
            ecg_time = [record_start + timedelta(seconds=i/100) for i in range(len(ecg_data))]

            result = pd.DataFrame({
                'datetime': ecg_time,
                'ecg': ecg_data
            })
            result.attrs['created'] = dl_time

            return result

        except Exception as e:
            print(f"Error processing UBE file {ube_file}: {e}")
            return pd.DataFrame(), {}