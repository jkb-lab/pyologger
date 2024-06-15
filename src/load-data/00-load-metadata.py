import os
from notion_client import Client
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
notion_token = os.getenv("notion_token") # uses environmental variable to store token
# Initialize the Notion client with integration token
notion = Client(auth=notion_token) # Secret token

databases = {
    "dep_DB": os.getenv("databases.dep_DB"), # uses environmental variable to store database ID
    "rec_DB": os.getenv("databases.rec_DB"), # uses environmental variable to store database ID
    "logger_DB": os.getenv("databases.logger_DB"), # uses environmental variable to store database ID
    "animal_DB": os.getenv("databases.animal_DB"), # uses environmental variable to store database ID
}
os.getenv("databases") # uses environmental variable to store database IDs

# Initialize a dictionary to hold each DataFrame
metadataframes = {}

# Fetch and process each database
for db_name, db_id in databases.items():
    db = notion.databases.query(database_id=db_id)
    rows_list = []
    for page in db["results"]:
        row_data = {}
        for prop_name, prop in page["properties"].items():
            prop_type = prop["type"]
            # Handle different property types here
            if prop_type == "title":
                row_data[prop_name] = prop[prop_type][0]["text"]["content"] if prop[prop_type] else None
            elif prop_type == "rich_text":
                row_data[prop_name] = prop[prop_type][0]["plain_text"] if prop[prop_type] else None
            elif prop_type == "number":
                row_data[prop_name] = prop[prop_type]
            elif prop_type == "select":
                row_data[prop_name] = prop[prop_type]["name"] if prop[prop_type] else None
            elif prop_type == "date":
                row_data[prop_name] = prop[prop_type]["start"] if prop[prop_type] else None
            # Add more property types as needed
        rows_list.append(row_data)
    metadataframes[db_name] = pd.DataFrame(rows_list)

# Print the metadata
for db_name, df in metadataframes.items():
    print(f"Dataframe: {db_name}")
    print(df)
    print()

# Example for how to reference metadata:
# Get the AnimalID for every Common name matching "boat"
animal_ids = metadataframes['animal_DB'].loc[metadataframes['animal_DB']['CommonName'] == 'boat', 'AnimalID']

# Print the AnimalIDs
print("Animal IDs:")
print(animal_ids)
