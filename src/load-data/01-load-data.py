import os
import pandas as pd
from datareader import DataReader
from metadata import Metadata
import dill as pickle

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
    deployment_folder_name = '00_renamed-raw-data/2024-06-17_oror-002-001a'
    data_reader = DataReader(deployment_folder_name)
    
    # Use the DataReader to process the files
    data_reader.read_files(metadata, deployment_folder_name, save_csv=True)
    print(data_reader.data_raw['2024-06-17_oror-002-001a_CO-68_001'])

    # Create the outputs directory if it doesn't exist
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data_reader
    try:
        with open(os.path.join(output_dir, 'data_reader.pkl'), 'wb') as f:
            pickle.dump(data_reader, f)
        print("data_reader.pkl saved successfully.")
    except Exception as e:
        print(f"Error saving data_reader.pkl: {e}")
    
    # Save metadata
    try:
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        print("metadata.pkl saved successfully.")
    except Exception as e:
        print(f"Error saving metadata.pkl: {e}")

        