import os
import pandas as pd
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