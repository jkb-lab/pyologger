import os
import re
from notion_client import Client
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

class Metadata:
    def __init__(self):
        load_dotenv()
        notion_token = os.getenv("notion_token")
        if not notion_token:
            raise ValueError("Notion token not found in environment variables")
        print(f"Loaded Notion secret token.")

        self.notion = Client(auth=notion_token)
        self.databases = {
            "deployment_DB": os.getenv("databases.deployment_DB"),
            "recording_DB": os.getenv("databases.recording_DB"),
            "logger_DB": os.getenv("databases.logger_DB"),
            "animal_DB": os.getenv("databases.animal_DB"),
            "dataset_DB": os.getenv("databases.dataset_DB")
        }

        # Check if all database IDs are loaded correctly
        for db_name, db_id in self.databases.items():
            if not db_id:
                raise ValueError(f"Database ID for {db_name} not found in environment variables")
            print(f"Loaded database ID for {db_name}.")

        self.metadata = {}

    def parse_metadata_value(self, prop, prop_type, column_name):
        if prop is None:
            return None

        if prop_type == "title" or prop_type == "rich_text":
            return ", ".join([text.get("plain_text", "") for text in prop.get(prop_type, []) if text.get("plain_text", "") is not None])
        elif prop_type == "number":
            return str(prop.get("number", None)) if prop.get("number") is not None else None
        elif prop_type == "select":
            return prop.get("select", {}).get("name", None)
        elif prop_type == "multi_select":
            return ", ".join([opt.get("name", "") for opt in prop.get("multi_select", []) if opt.get("name", "") is not None])
        elif prop_type == "url":
            return prop.get("url", None)
        elif prop_type == "rollup":
            rollup_type = prop.get("rollup", {}).get("type")
            if rollup_type == "array":
                array_values = [self.parse_metadata_value(item, item.get("type", ""), column_name) for item in prop.get("rollup", {}).get("array", [])]
                return ", ".join([str(val) for val in array_values if val is not None])
            elif rollup_type == "number":
                return str(prop.get("rollup", {}).get("number", None)) if prop.get("rollup", {}).get("number") is not None else None
            elif rollup_type == "date":
                return self.parse_metadata_value(prop.get("rollup", {}), "date", column_name)
            return str(prop.get("rollup", {}))
        elif prop_type == "relation":
            return ", ".join([rel.get("id", "") for rel in prop.get("relation", []) if rel.get("id", "") is not None])
        elif prop_type == "date":
            date_info = prop.get("date", {})
            start_date = date_info.get("start")
            end_date = date_info.get("end")

            if start_date:
                try:
                    parsed_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                    return str(parsed_date)
                except ValueError:
                    return start_date  # Return as-is if parsing fails
            return None
        elif prop_type == "people":
            return ", ".join([person.get("name", "") for person in prop.get("people", []) if person.get("name", "") is not None])
        elif isinstance(prop, dict) and "number" in prop:
            number = str(prop.get("number", ""))
            prefix = prop.get("prefix", "")
            return f"{prefix}{number}".strip() if prefix else number
        elif isinstance(prop, dict) and "string" in prop:
            return prop.get("string", "")
        elif isinstance(prop, dict) and "id" in prop:
            return prop.get("id", "")
        else:
            return str(prop) if prop is not None else None


    def fetch_databases(self, verbose=True):
        for db_name, db_id in self.databases.items():
            if verbose:
                print(f"Fetching data for {db_name} with ID {db_id}")
            if db_id is None:
                print(f"Database ID for {db_name} is None. Skipping.")
                continue
            try:
                db = self.notion.databases.query(database_id=db_id)
                rows_list = []
                for page in db["results"]:
                    row_data = {"page_id": page["id"]}  # Store the page ID for each row
                    for prop_name, prop in page["properties"].items():
                        prop_type = prop.get("type")
                        parsed_value = self.parse_metadata_value(prop, prop_type, prop_name)
                        row_data[prop_name] = parsed_value
                    rows_list.append(row_data)
                self.metadata[db_name] = pd.DataFrame(rows_list)
                if verbose:
                    print(f"Successfully fetched data for {db_name}.")
            except Exception as e:
                if verbose:
                    print(f"Error fetching data for {db_name}: {e}")

    def get_metadata(self, db_name):
        return self.metadata.get(db_name)

    def print_metadata(self):
        for db_name, df in self.metadata.items():
            print(f"Dataframe: {db_name}")
            print(df)
            print()

    def find_relations(self, verbose=True):
        uuid_pattern = re.compile(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}')
        
        id_columns = {
            "deployment_DB": "Deployment ID",
            "recording_DB": "Recording ID",
            "logger_DB": "Logger ID",
            "animal_DB": "Animal ID",
            "dataset_DB": "Dataset ID"
        }
        
        page_id_lookup = {}
        
        for db_name, df in self.metadata.items():
            id_column = id_columns.get(db_name)
            if id_column and id_column in df.columns:
                for index, row in df.iterrows():
                    page_id_lookup[row['page_id']] = row[id_column]
        
        for db_name, df in self.metadata.items():
            if verbose:
                print(f"Searching for UUID patterns in {db_name}")
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    for index, value in df[col].items():
                        if pd.notna(value) and isinstance(value, str):
                            matches = uuid_pattern.findall(value)
                            for match in matches:
                                if match in page_id_lookup:
                                    id_value = page_id_lookup[match]
                                    if pd.notna(id_value):
                                        clean_value = value.replace(match, str(id_value))
                                        df.at[index, col] = clean_value
                                        if verbose:
                                            print(f"Replaced UUID-like string '{match}' in column '{col}' of '{db_name}' with '{id_value}'")
                                    else:
                                        if verbose:
                                            print(f"Skipped replacement for UUID '{match}' in column '{col}' of '{db_name}' because the matching ID value is None or NaN.")
