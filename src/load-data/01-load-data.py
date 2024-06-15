import os
import numpy as np
import pandas as pd

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the path to the data folder
data_folder = os.path.join(current_dir, '../../data/')

# Define the file path relative to the root directory
notes_path = os.path.join(data_folder, '00_renamed-raw-data', '2024-01-16_oror-002a_00_Notes.csv')

# Check for files with *_CC-* in the filename
CC_files = [file for file in file_list if '*_CC-*' in file]

# Check for files with *_UF-* in the filename
UF_files = [file for file in file_list if '*_UF-*' in file]

# Load the CSV file into a DataFrame
notes = pd.read_csv(notes_path)

# If there are matching CC files
if CC_files:
    # Extract the loggerID from the first matching CC file
    loggerID = CC_files[0].split('_CC-')[1][:5]  # Assuming loggerID is always 5 characters long

    # Filter the matching CC files based on loggerID
    CC_files = [file for file in CC_files if loggerID in file]

    # Load the CC CSV files into a list of DataFrames
    CC_data_frames = []
    for file in CC_files:
        file_path = os.path.join(data_folder, '00_renamed-raw-data', file)
        CC_data_frames.append(pd.read_csv(file_path))

    # Concatenate the CC DataFrames into a single DataFrame
    combined_CC_data = pd.concat(CC_data_frames)

    # Print the combined CC data
    print(combined_CC_data.head())

# If there are matching UF files
elif UF_files:
    # Extract the loggerID from the first matching UF file
    loggerID = UF_files[0].split('_UF-')[1][:5]  # Assuming loggerID is always 5 characters long

    # Filter the matching UF files based on loggerID
    UF_files = [file for file in UF_files if loggerID in file]

    # Load the UF CSV files into a list of DataFrames
    UF_data_frames = []
    for file in UF_files:
        file_path = os.path.join(data_folder, '00_renamed-raw-data', file)
        UF_data_frames.append(pd.read_csv(file_path))

    # Concatenate the UF DataFrames into a single DataFrame
    combined_UF_data = pd.concat(UF_data_frames)

    # Print the combined UF data
    print(combined_UF_data.head())

else:
    print("No matching files found.")



# Print the first few rows of the DataFrame
print(notes.head())
