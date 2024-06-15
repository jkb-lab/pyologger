from notion_client import Client
import pandas as pd

# Initialize the Notion client with your integration token
notion = Client(auth="***REMOVED-NOTION-TOKEN***") # Secret token

# The ID of your Notion databases
databases = {
    "dep_DB": "657cae511066439ea9499085e3443406",  # Link to Deployments database
    "rec_DB": "0a86a1b1756c46afa84429e8ff13c79a",  # Link to Recordings database
    "logger_DB": "8aa29755491c4357b87e43cb9a505a70",  # Link to Loggers database
    "animal_DB": "b57495033c4d4391aa5ab75de0291802",  # Link to Animal database
}

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
