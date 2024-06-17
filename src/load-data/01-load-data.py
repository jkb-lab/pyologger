import os
import pandas as pd
from datareader import DataReader
from metadata import Metadata

# Initialize the Metadata class
metadata = Metadata()
metadata.fetch_databases()

# Get the logger database
logger_db = metadata.get_metadata("logger_DB")

# Determine unique LoggerIDs from the logger metadata dataframe
logger_ids = set(logger_db['LoggerID'])
print(f"Unique Logger IDs: {logger_ids}")

# Breakdown of loggers by type
logger_breakdown = logger_db.groupby(['Manufacturer', 'Type']).size().reset_index(name='Count')
print("Logger Breakdown by Manufacturer and Type:")
print(logger_breakdown)

if __name__ == "__main__":
    deployment_folder_name = '00_renamed-raw-data'
    data_reader = DataReader(deployment_folder_name)
    
    # Use the DataReader to process the files
    data_reader.read_files(metadata, deployment_folder_name, save_csv=False)
    print(data_reader.data_raw['2024-01-16_oror-002a_CC-96_001'])