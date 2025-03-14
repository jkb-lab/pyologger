import os
import re
import numpy as np
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
            "dataset_DB": os.getenv("databases.dataset_DB"),
            "procedure_DB": os.getenv("databases.procedure_DB"),
            "observation_DB": os.getenv("databases.observation_DB"),
            "collaborator_DB": os.getenv("databases.collaborator_DB"),
            "location_DB": os.getenv("databases.location_DB"),
            "montage_DB": os.getenv("databases.montage_DB"),
            "sensor_DB": os.getenv("databases.sensor_DB"),
            "sensorchannel_DB": os.getenv("databases.sensorchannel_DB"),
            "derivedsignal_DB": os.getenv("databases.derivedsignal_DB"),
            "derivedchannel_DB": os.getenv("databases.derivedchannel_DB")
        }

        self.metadata = {}
        self.page_id_lookup = {}  # Store ID -> Name mapping
        self.relations_map = {}

        self.fetch_databases(verbose=True)
        self.find_relations(verbose=False)  # Preprocess relations immediately
        

    def parse_metadata_value(self, prop, prop_type, column_name):
        """Parse Notion API metadata fields safely, ensuring Pandas-friendly outputs."""
        if prop is None or prop_type is None:
            return np.nan

        try:
            if prop_type in ["title", "rich_text"]:
                value = ", ".join([text.get("plain_text", "") for text in prop.get(prop_type, []) if text.get("plain_text")])
                return value if value else np.nan
            elif prop_type == "number":
                return prop.get("number", np.nan)
            elif prop_type == "select":
                return prop.get("select", {}).get("name", np.nan) if prop.get("select") else np.nan
            elif prop_type == "multi_select":
                value = ", ".join([opt.get("name", "") for opt in prop.get("multi_select", []) if opt.get("name")])
                return value if value else np.nan
            elif prop_type == "url":
                return prop.get("url", np.nan)
            elif prop_type == "rollup":
                return np.nan
            elif prop_type == "relation":
                # Store raw page IDs as a comma-separated string; `find_relations()` will replace them later
                related_ids = [rel.get("id") for rel in prop.get("relation", []) if rel.get("id")]
                return ", ".join(related_ids) if related_ids else np.nan
            elif prop_type == "date":
                date_info = prop.get("date", {})
                start_date = date_info.get("start")
                if start_date:
                    try:
                        return datetime.strptime(start_date, "%Y-%m-%d").date()
                    except ValueError:
                        return start_date  # Return as-is if parsing fails
                return np.nan
            elif prop_type == "people":
                value = ", ".join([person.get("name", "") for person in prop.get("people", []) if person.get("name")])
                return value if value else np.nan
            elif isinstance(prop, dict) and "number" in prop:
                number = str(prop.get("number", ""))
                prefix = prop.get("prefix", "")
                return f"{prefix}{number}".strip() if prefix else (number if number else np.nan)
            elif isinstance(prop, dict) and "string" in prop:
                return prop.get("string", np.nan)
            elif isinstance(prop, dict) and "id" in prop:
                return prop.get("id", np.nan)
            else:
                return str(prop) if prop is not None else np.nan
        except Exception as e:
            print(f"Error parsing {column_name}: {e}")
            return np.nan

    def fetch_databases(self, verbose=True):
        """Fetch all databases and store metadata while removing rollup columns."""
        self.metadata_types = {}  # Store column types

        for db_name, db_id in self.databases.items():
            if verbose:
                print(f"Fetching data for {db_name} with ID {db_id}")
            if db_id is None:
                print(f"Database ID for {db_name} is None. Skipping.")
                continue

            try:
                db = self.notion.databases.query(database_id=db_id)
                rows_list = []
                column_types = {}

                for page in db["results"]:
                    row_data = {"page_id": page["id"]}  # Store page ID for each row

                    for prop_name, prop in page["properties"].items():
                        if prop is None:
                            continue
                        prop_type = prop.get("type")

                        # **Skip rollup columns entirely**
                        if prop_type == "rollup":
                            continue  # Do not store rollup data at all

                        column_types[prop_name] = prop_type  # Store type for lookup

                        try:
                            parsed_value = self.parse_metadata_value(prop, prop_type, prop_name)
                            row_data[prop_name] = parsed_value
                        except Exception as parse_error:
                            print(f"Error parsing '{prop_name}' in {db_name}: {parse_error}")

                    rows_list.append(row_data)

                # **Create DataFrame without rollup columns**
                df = pd.DataFrame(rows_list)

                # **Remove any lingering rollup columns**
                rollup_columns = [col for col in df.columns if column_types.get(col) == "rollup"]
                if rollup_columns:
                    df.drop(columns=rollup_columns, inplace=True)
                    if verbose:
                        print(f"[DEBUG] Removed rollup columns from {db_name}: {rollup_columns}")

                self.metadata[db_name] = df
                self.metadata_types[db_name] = column_types  # Store column types

            except Exception as e:
                if verbose:
                    print(f"Error fetching data for {db_name}: {e}")



    def find_relations(self, verbose=True):
        """Pre-fetch all page names and replace relation IDs with human-readable names while keeping commas between values."""
        uuid_pattern = re.compile(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}')

        # Reset page ID lookup
        self.page_id_lookup = {}

        if verbose:
            print("\n[DEBUG] Building Page ID -> Title lookup dictionary...")

        # Step 1: Build Page ID -> Title lookup dictionary
        for db_name, df in self.metadata.items():
            if "page_id" in df.columns:
                if verbose:
                    print(f"[DEBUG] Processing {db_name} to extract title fields...")

                # Get column types for this database
                column_types = self.metadata_types.get(db_name, {})

                # Find the title column
                title_column = None
                for col, col_type in column_types.items():
                    if col_type == "title":
                        title_column = col
                        break

                if title_column is None:
                    if verbose:
                        print(f"[WARNING] No title column found in {db_name}. Skipping.")
                    continue

                # Extract page ID -> Title mapping
                for _, row in df.iterrows():
                    page_id = row["page_id"]
                    name_field = row.get(title_column, None)

                    if pd.notna(name_field):
                        self.page_id_lookup[page_id] = name_field
                        if verbose:
                            print(f"[DEBUG] Mapped {page_id} -> {name_field}")

        if verbose:
            print("\n[DEBUG] Completed Page ID -> Title mapping.")
            print(f"[DEBUG] Total IDs stored: {len(self.page_id_lookup)}\n")

        # Step 2: Replace relation IDs with names in all DataFrames
        if verbose:
            print("\n[DEBUG] Replacing relation IDs with human-readable names...")

        for db_name, df in self.metadata.items():
            if verbose:
                print(f"[DEBUG] Processing {db_name} for relation replacements...")

            for col in df.columns:
                if df[col].dtype == 'object':  # Only process string columns
                    for index, value in df[col].items():
                        if pd.notna(value) and isinstance(value, str):
                            matches = uuid_pattern.findall(value)

                            if not matches:
                                continue  # Skip if no UUIDs found

                            # Split by commas, replace UUIDs, and rejoin
                            parts = [self.page_id_lookup.get(part.strip(), part.strip()) for part in value.split(",")]
                            new_value = ", ".join(parts)

                            if new_value != value:
                                df.at[index, col] = new_value
                                if verbose:
                                    print(f"[DEBUG] Updated {db_name}.{col} (Row {index}): '{value}' → '{new_value}'")

        if verbose:
            print("\n[DEBUG] Completed relation replacements.\n")

    def map_database_relations(self):
            """Map relations between all databases."""
            for db_name, db_id in self.databases.items():
                db_schema = self.notion.databases.retrieve(database_id=db_id)
                self.relations_map[db_name] = {
                    prop_name: prop_details["relation"]["database_id"]
                    for prop_name, prop_details in db_schema["properties"].items()
                    if prop_details["type"] == "relation"
                }
            print("Database Relations Map:", self.relations_map)

    def get_metadata(self, db_name, update_relations=False):
        """Retrieve a DataFrame for a given database.
        
        If update_relations=True, it will update relation IDs to human-readable names before returning.
        """
        if db_name in self.metadata:
            if update_relations:
                self.find_relations()  # Only update if explicitly requested
            return self.metadata[db_name]
        return None

    def print_metadata(self):
        """Print stored metadata."""
        for db_name, df in self.metadata.items():
            print(f"Dataframe: {db_name}")
            print(df)
            print()
