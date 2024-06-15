import os
import numpy as np
import pandas as pd
from metadata import Metadata

deployment_folder_name = '00_renamed-raw-data'

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the path to the data folder
data_folder = os.path.join(current_dir, '../../data/')
deployment_data_folder = os.path.join(data_folder, deployment_folder_name)
file_list = os.listdir(deployment_data_folder)
print(file_list)

# Load deployment notes
notes_file = [file for file in file_list if file.endswith('_00_Notes.csv')]
if notes_file:
    notes = pd.read_csv(os.path.join(deployment_data_folder, notes_file[0]))
    print(notes.head())
else:
    print("No notes file found.")

# Initialize the Metadata class
metadata = Metadata()
metadata.fetch_databases()

# Get the logger database
logger_db = metadata.get_metadata("logger_DB")

# Determine unique LoggerIDs from the logger metadata dataframe
logger_ids = set(logger_db['LoggerID'])

# Find files corresponding to logger IDs in metadata
logger_files = {}
for logger_id in logger_ids:
    logger_files[logger_id] = [file for file in file_list if logger_id in file]

# Categorize files based on Manufacturer
manufacturer_files = {}
for logger_id, files in logger_files.items():
    manufacturer = logger_db.loc[logger_db['LoggerID'] == logger_id, 'Manufacturer'].values[0]
    if manufacturer not in manufacturer_files:
        manufacturer_files[manufacturer] = []
    manufacturer_files[manufacturer].extend(files)

# Print the categorized files and file count per manufacturer
if manufacturer_files:
    total_files = sum(len(files) for files in manufacturer_files.values())
    print(f"SUCCESS: {total_files} Data files found.")
    for manufacturer, files in manufacturer_files.items():
        print(f"{len(files)} {manufacturer} files: {files}")
else:
    print("No logger files found.")


# Check for files with *_CC-* in the filename
# CATS_files = [file for file in file_list if '_CC-' in file] # CATS loggers
# UFI_files = [file for file in file_list if '_UF-' in file] # UFI loggers
# WCH_files = [file for file in file_list if '_WC-' in file] # Wildlife Computers loggers
# LL_files = [file for file in file_list if '_LL-' in file] # Little Leonardo loggers
# NL_files = [file for file in file_list if '_NL-' in file] # Neurologgers
# MN_files = [file for file in file_list if '_MN-' in file] # Manitty loggers

# print(f"Number of unique loggers: {len(unique_loggers)}")

# # Count the number of files per logger and list unique file extensions
# logger_files = {}
# for logger in unique_loggers:
#     logger_files[logger] = []
#     for file in file_list:
#         if logger in file:
#             logger_files[logger].append(file)

# for logger, files in logger_files.items():
#     unique_extensions = set()
#     for file in files:
#         extension = file.split('.')[-1]
#         unique_extensions.add(extension)
#     print(f"Logger {logger}: {len(files)} files, unique extensions: {', '.join(unique_extensions)}")


# # Define a function to load CSV files into a list of DataFrames
# def load_csv_files(file_list):
#     data_frames = []
#     for file in file_list:
#         file_path = os.path.join(data_folder, '00_renamed-raw-data', file)
#         data_frames.append(pd.read_csv(file_path))
#     return data_frames

# # If there are matching CC files
# if CC_files:
#     print(f"Found {len(CC_files)} CATS files."))
#     # Extract the loggerID from the first matching CC file
#     loggerID = CC_files[0].split('_CC-')[1][:5]  # Assuming loggerID is always 5 characters long
#     # Filter the matching CC files based on loggerID
#     CC_files = [file for file in CC_files if loggerID in file]
#     # Load the CC CSV files into a list of DataFrames
#     CC_data_frames = load_csv_files(CC_files)
#     # Concatenate the CC DataFrames into a single DataFrame
#     combined_CC_data = pd.concat(CC_data_frames)
#     # Print the combined CC data
#     print(combined_CC_data.head())
# # If there are matching UF files
# elif UF_files:
#     # Extract the loggerID from the first matching UF file
#     loggerID = UF_files[0].split('_UF-')[1][:5]  # Assuming loggerID is always 5 characters long
#     # Filter the matching UF files based on loggerID
#     UF_files = [file for file in UF_files if loggerID in file]
#     # Load the UF CSV files into a list of DataFrames
#     UF_data_frames = load_csv_files(UF_files)
#     # Concatenate the UF DataFrames into a single DataFrame
#     combined_UF_data = pd.concat(UF_data_frames)
#     # Print the combined UF data
#     print(combined_UF_data.head())
# else:
#     print("No matching files found.")



