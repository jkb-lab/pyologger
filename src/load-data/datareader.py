import os
import struct
import pandas as pd
from datetime import datetime, timedelta

class DataReader:
    def __init__(self, deployment_folder_name):
        self.deployment_folder_name = deployment_folder_name
        self.current_dir = os.path.dirname(__file__)
        self.data_folder = os.path.join(self.current_dir, '../../data/')
        self.deployment_data_folder = os.path.join(self.data_folder, deployment_folder_name)
        self.file_list = os.listdir(self.deployment_data_folder)
        self.data_raw = {}
    
    def read_ube(self, ube_path):
        with open(ube_path, 'rb') as file:
            ube_raw = file.read()

        dl_time_str = ube_raw[0:32].decode('utf-8').strip()
        print(f"Parsed download timestamp string: '{dl_time_str}'")
        try:
            dl_time = datetime.strptime(dl_time_str, "%m-%d-%Y, %H:%M:%S")
        except ValueError as e:
            print(f"Error parsing timestamp: {dl_time_str} - {e}")
            raise

        mdhms = struct.unpack('BBBBB', ube_raw[32:37])
        now = datetime.now()
        record_start = datetime(now.year, mdhms[0], mdhms[1], mdhms[2], mdhms[3], mdhms[4])
        print(f"Recording start time: {record_start}")

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

        ecg_time = [record_start + timedelta(seconds=i/100) for i in range(len(ecg_data))]

        result = pd.DataFrame({
            'timestamp': ecg_time,
            'ecg': ecg_data
        })
        result.attrs['created'] = dl_time

        return result

    def read_csv(self, csv_path):
        encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                print(f"Attempting to read {csv_path} with encoding {encoding}")
                return pd.read_csv(csv_path, encoding=encoding)
            except UnicodeDecodeError as e:
                print(f"Error reading {csv_path} with encoding {encoding}: {e}")
        raise UnicodeDecodeError(f"Failed to read {csv_path} with available encodings.")

    def read_files(self, metadata, deployment_folder_name, save_csv=True):
        self.deployment_folder_name = deployment_folder_name
        self.current_dir = os.path.dirname(__file__)
        self.data_folder = os.path.join(self.current_dir, '../../data/')
        self.deployment_data_folder = os.path.join(self.data_folder, deployment_folder_name)
        self.file_list = os.listdir(self.deployment_data_folder)
        
        metadata.fetch_databases()
        logger_db = metadata.get_metadata("logger_DB")
        
        logger_ids = set(logger_db['LoggerID'])
        logger_files = {}
        for logger_id in logger_ids:
            logger_files[logger_id] = [file for file in self.file_list if logger_id in file]
        
        manufacturer_files = {}
        for logger_id, files in logger_files.items():
            manufacturer = logger_db.loc[logger_db['LoggerID'] == logger_id, 'Manufacturer'].values[0]
            if manufacturer not in manufacturer_files:
                manufacturer_files[manufacturer] = []
            manufacturer_files[manufacturer].extend(files)
        
        if manufacturer_files:
            total_files = sum(len(files) for files in manufacturer_files.values())
            print()
            print(f"SUCCESS: {total_files} Data files found.")
            for manufacturer, files in manufacturer_files.items():
                print(f"{len(files)} {manufacturer} files: {files}")
                print()

        else:
            print("No logger files found.")

        for manufacturer, files in manufacturer_files.items():
            print(f"Manufacturer: {manufacturer}")
            for file in files:
                file_path = os.path.join(self.deployment_data_folder, file)
                if file.endswith('.ube'):
                    try:
                        data = self.read_ube(file_path)
                        print(data.head())
                        if save_csv:
                            self.save_data(data, file, save_csv=True)
                        else:
                            self.save_data(data, file, save_csv=False)
                        print()
                        print(f"File: {file} - Successfully processed.")
                    except ValueError as e:
                        print()
                        print(f"Error processing file {file}: {e}")
                elif file.endswith('.csv'):
                    try:
                        data = self.read_csv(file_path)
                        print(data.head())
                        if save_csv:
                            self.save_data(data, file, save_csv=True)
                        else:
                            self.save_data(data, file, save_csv=False)
                        print()
                        print(f"File: {file} - Successfully processed.")
                    except UnicodeDecodeError as e:
                        print()
                        print(f"Error processing file {file}: {e}")
                else:
                    print()
                    print(f"File: {file} - NOT a supported file type.")
    
    def save_data(self, data, file, save_csv=True):
        attribute_name = ""
        if file is not None:
            attribute_name = os.path.splitext(file)[0]
        self.data_raw[attribute_name] = data
        if save_csv and file is not None:
            output_folder = os.path.join(os.path.dirname(__file__), '../outputs')
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.csv")
            data.to_csv(output_path, index=False)
            print(f"Data successfully saved to: {output_path}")
        else:
            self.data_raw[attribute_name] = data
            print("Data saved to attribute and not to CSV. *Change save_csv to True to save to CSV.")
