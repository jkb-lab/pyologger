import os
import re
import json
import requests
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from notion_client import Client


# Databases this class pulls. Each maps to NOTION_<NAME>_DB in the environment.
NOTION_DATABASES = (
    "deployment",
    "recording",
    "logger",
    "animal",
    "dataset",
    "procedure",
    "observation",
    "collaborator",
    "location",
    "montage",
    "signal",
    "attachment",
    "originalchannel",
    "standardizedchannel",
)


# Databases whose env var name differs from a plain upper-casing of the key above.
_ENV_ALIASES = {
    "animal": "ORGANISM",  # the Notion database is titled "Organism DB"
    "originalchannel": "ORIGINAL_CHANNEL",
    "standardizedchannel": "STANDARDIZED_CHANNEL",
}


def _db_env(name: str):
    """Database ID for `name`, read as NOTION_DB_<NAME>.

    Falls back to the older NOTION_<NAME>_DB and databases.<name>_DB spellings so
    unmigrated .env files keep working. `databases.x` is not a valid shell
    identifier and cannot be exported or passed through Docker, which is part of
    why the NOTION_DB_* prefix form is now canonical.
    """
    suffix = _ENV_ALIASES.get(name, name.upper())
    return (
        os.getenv(f"NOTION_DB_{suffix}")
        or os.getenv(f"NOTION_{name.upper()}_DB")
        or os.getenv(f"databases.{name}_DB")
    )


class Metadata:
    def __init__(self):
        load_dotenv()
        notion_token = os.getenv("NOTION_TOKEN") or os.getenv("notion_token")
        if not notion_token:
            raise ValueError(
                "Notion token not found. Set NOTION_TOKEN in your .env file."
            )
        print("Loaded Notion secret token.")

        # Use the DS API release
        self.notion_version = "2025-09-03"
        self.notion_token = notion_token

        # Keep the SDK for retrieve/schema calls
        self.notion = Client(
            auth=notion_token,
            notion_version=self.notion_version,
        )

        # DATABASE IDs from .env, keyed as <name>_DB for backwards compatibility
        # with callers that index self.databases directly.
        self.databases = {f"{name}_DB": _db_env(name) for name in NOTION_DATABASES}

        # runtime state
        self.data_source_cache = {}   # {database_id: data_source_id}
        self.metadata = {}            # {logical_name: DataFrame}
        self.metadata_types = {}      # {logical_name: {col_name: notion_type}}
        self.page_id_lookup = {}      # {page_id: title}
        self.relations_map = {}       # {db_name: {prop_name: {database_id, data_source_id}}}

        # pull everything
        self.fetch_databases(verbose=True)

        # resolve relation UUIDs to human-readable names
        self.find_relations(verbose=False)


    def _headers(self):
        """Headers for raw HTTP calls to Notion."""
        return {
            "Authorization": f"Bearer {self.notion_token}",
            "Notion-Version": self.notion_version,
            "Content-Type": "application/json",
        }

    # ---------------------------
    # STEP 1: Discover data_source_id
    # ---------------------------
    def discover_data_source_id(self, database_id):
        """
        Call GET /v1/databases/:database_id via the official client.
        Cache the first data_sources[].id so we know which data source is
        considered the "main" for this database.
        """
        if not database_id:
            return None

        if database_id in self.data_source_cache:
            return self.data_source_cache[database_id]

        try:
            db_info = self.notion.databases.retrieve(database_id=database_id)
        except Exception as e:
            print(
                f"[ERROR] Could not retrieve database {database_id}. "
                "Either wrong ID or the integration is not shared on it.\n ->", e
            )
            self.data_source_cache[database_id] = None
            return None

        data_sources = db_info.get("data_sources", [])
        if not data_sources:
            print(
                f"[ERROR] Database {database_id} returned no data_sources[]. "
                "Either no data sources exist or access is restricted."
            )
            self.data_source_cache[database_id] = None
            return None

        ds_id = data_sources[0].get("id")
        if not ds_id:
            print(
                f"[ERROR] Database {database_id} had a data_source with no id."
            )
            self.data_source_cache[database_id] = None
            return None

        self.data_source_cache[database_id] = ds_id
        return ds_id

    # ---------------------------
    # Parse Notion property blobs into usable values
    # ---------------------------
    def parse_metadata_value(self, prop, prop_type, column_name):
        if prop is None or prop_type is None:
            return np.nan
        try:
            if prop_type in ["title", "rich_text"]:
                value = ", ".join(
                    [
                        text.get("plain_text", "")
                        for text in prop.get(prop_type, [])
                        if text.get("plain_text")
                    ]
                )
                return value if value else np.nan

            elif prop_type == "number":
                return prop.get("number", np.nan)

            elif prop_type == "select":
                return (
                    prop.get("select", {}).get("name", np.nan)
                    if prop.get("select")
                    else np.nan
                )

            elif prop_type == "multi_select":
                value = ", ".join(
                    [
                        opt.get("name", "")
                        for opt in prop.get("multi_select", [])
                        if opt.get("name")
                    ]
                )
                return value if value else np.nan

            elif prop_type == "url":
                return prop.get("url", np.nan)

            elif prop_type == "rollup":
                return np.nan  # skipped

            elif prop_type == "relation":
                related_ids = [
                    rel.get("id")
                    for rel in prop.get("relation", [])
                    if rel.get("id")
                ]
                return ", ".join(related_ids) if related_ids else np.nan

            elif prop_type == "date":
                date_info = prop.get("date", {})
                start_date = date_info.get("start")
                if start_date:
                    try:
                        return datetime.strptime(start_date, "%Y-%m-%d").date()
                    except ValueError:
                        return start_date  # ISO timestamp fallback
                return np.nan

            elif prop_type == "people":
                # Notion user objects typically include "name"; fallback if missing
                names = []
                for person in prop.get("people", []):
                    name = person.get("name")
                    if not name:
                        # Older payloads may require a further lookup; keep id/email if present
                        name = person.get("id") or person.get("person", {}).get("email")
                    if name:
                        names.append(name)
                value = ", ".join(names)
                return value if value else np.nan

            # fallbacks
            elif isinstance(prop, dict) and "number" in prop:
                number = str(prop.get("number", ""))
                prefix = prop.get("prefix", "")
                return (
                    f"{prefix}{number}".strip()
                    if prefix
                    else (number if number else np.nan)
                )

            elif isinstance(prop, dict) and "string" in prop:
                return prop.get("string", np.nan)

            elif isinstance(prop, dict) and "id" in prop:
                return prop.get("id", np.nan)

            else:
                return str(prop) if prop is not None else np.nan

        except Exception as e:
            print(f"Error parsing {column_name}: {e}")
            return np.nan

    def _extract_cover_url(self, page):
        """
        Extract Notion page cover URL from page-level metadata.
        Supports both external and Notion-hosted file cover payloads.
        """
        if not isinstance(page, dict):
            return np.nan
        cover = page.get("cover") or {}
        if not isinstance(cover, dict):
            return np.nan
        cover_type = cover.get("type")
        if cover_type == "external":
            return cover.get("external", {}).get("url", np.nan)
        if cover_type == "file":
            return cover.get("file", {}).get("url", np.nan)
        return np.nan

    def _extract_icon(self, page):
        """
        Extract Notion page icon as (emoji, image_url).
        """
        if not isinstance(page, dict):
            return np.nan, np.nan
        icon = page.get("icon") or {}
        if not isinstance(icon, dict):
            return np.nan, np.nan
        icon_type = icon.get("type")
        if icon_type == "emoji":
            return icon.get("emoji", np.nan), np.nan
        if icon_type == "external":
            return np.nan, icon.get("external", {}).get("url", np.nan)
        if icon_type == "file":
            return np.nan, icon.get("file", {}).get("url", np.nan)
        return np.nan, np.nan

    # ---------------------------
    # STEP 2 (legacy): query database pages via old endpoint
    # ---------------------------
    def _http_query_database_all_pages(self, database_id):
        """
        Raw HTTP pagination loop:
        POST /v1/databases/{database_id}/query
        (Kept for compatibility; not used by the DS path.)
        """
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        all_results = []
        start_cursor = None

        while True:
            payload = {}
            if start_cursor:
                payload["start_cursor"] = start_cursor

            resp = requests.post(url, headers=self._headers(), json=payload)

            if resp.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {resp.status_code} while querying {database_id}: {resp.text}"
                )

            data = resp.json()
            all_results.extend(data.get("results", []))

            if data.get("has_more"):
                start_cursor = data.get("next_cursor")
            else:
                break

        return all_results

    # ---------------------------
    # STEP 2 (new): query pages via Data Source
    # ---------------------------
    def query_data_source_pages(self, database_id, data_source_id, page_size=100, filter_properties=None, extra_body=None):
        """
        Query all pages (rows) from a specific data source using the new
        Data Sources API: POST /v1/data_sources/{data_source_id}/query

        Args:
          database_id: only used for logging
          data_source_id: DS id discovered from the database
          page_size: batch size (<= 100)
          filter_properties: optional iterable of property names to slim payload
          extra_body: optional dict merged into the request body (e.g., filters)

        Returns:
          list of page objects
        """
        base_url = f"https://api.notion.com/v1/data_sources/{data_source_id}/query"
        params = []
        if filter_properties:
            for prop in filter_properties:
                params.append(("filter_properties", prop))

        all_pages = []
        start_cursor = None

        while True:
            body = {
                "page_size": page_size,
                # default stable sort by creation time so pagination is deterministic
                "sorts": [
                    {"timestamp": "created_time", "direction": "ascending"}
                ],
            }
            if start_cursor:
                body["start_cursor"] = start_cursor
            if extra_body:
                # merge with precedence to extra_body
                body.update(extra_body)

            resp = requests.post(base_url, headers=self._headers(), params=params, json=body)

            if resp.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {resp.status_code} while querying DS {data_source_id} "
                    f"(from DB {database_id}): {resp.text}"
                )

            data = resp.json()
            results = data.get("results", [])
            all_pages.extend(results)

            start_cursor = data.get("next_cursor")
            if not start_cursor:
                break

        return all_pages

    # ---------------------------
    # Pull everything into DataFrames
    # ---------------------------
    def fetch_databases(self, verbose=True):
        for db_name, db_id in self.databases.items():
            if verbose:
                print(f"Fetching data for {db_name} with database_id {db_id}")

            if not db_id or db_id.strip() == "":
                print(f"[WARN] {db_name} has no database_id in env. Skipping.")
                continue

            # 1. discover data_source_id for this database
            data_source_id = self.discover_data_source_id(db_id)
            if not data_source_id:
                print(f"[ERROR] {db_name}: couldn't resolve data_source_id (no access / not shared?)")
                continue

            if verbose:
                print(f"[DEBUG] {db_name}: using data_source_id {data_source_id}")

            # 2. query rows FROM THAT DATA SOURCE
            try:
                pages = self.query_data_source_pages(db_id, data_source_id)
            except Exception as e:
                print(
                    f"[ERROR] {db_name}: data source {data_source_id} query failed.\n -> {e}"
                )
                continue

            # 3. parse rows into a dataframe exactly like before
            rows_list = []
            column_types = {}

            for page in pages:
                row = {"page_id": page["id"]}
                # Page-level media metadata from Notion (not a database property).
                row["cover_image_url"] = self._extract_cover_url(page)
                icon_emoji, icon_image_url = self._extract_icon(page)
                row["icon_emoji"] = icon_emoji
                row["icon_image_url"] = icon_image_url

                for prop_name, prop in page.get("properties", {}).items():
                    if prop is None:
                        continue

                    prop_type = prop.get("type")

                    # skip rollup props, we don't store them
                    if prop_type == "rollup":
                        continue

                    column_types[prop_name] = prop_type

                    try:
                        row[prop_name] = self.parse_metadata_value(
                            prop, prop_type, prop_name
                        )
                    except Exception as parse_error:
                        print(
                            f"Error parsing '{prop_name}' in {db_name}: {parse_error}"
                        )

                rows_list.append(row)

            df = pd.DataFrame(rows_list)

            # clean up lingering rollup columns, just in case
            rollup_cols = [
                col for col in df.columns if column_types.get(col) == "rollup"
            ]
            if rollup_cols:
                df.drop(columns=rollup_cols, inplace=True, errors="ignore")
                if verbose:
                    print(f"[DEBUG] Removed rollup columns from {db_name}: {rollup_cols}")

            # 4. store per-database (FIX: this used to happen after the loop)
            self.metadata[db_name] = df
            self.metadata_types[db_name] = column_types

    # ---------------------------
    # Convert relation UUIDs -> readable titles
    # ---------------------------
    def find_relations(self, verbose=True):
        uuid_pattern = re.compile(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
        )

        self.page_id_lookup = {}

        if verbose:
            print("\n[DEBUG] Building Page ID -> Title lookup dictionary...")

        for db_name, df in self.metadata.items():
            if "page_id" not in df.columns:
                continue

            if verbose:
                print(f"[DEBUG] Processing {db_name} to extract title fields...")

            col_types = self.metadata_types.get(db_name, {})

            # find the title column in this df
            title_col = None
            for col, col_type in col_types.items():
                if col_type == "title":
                    title_col = col
                    break

            if title_col is None:
                if verbose:
                    print(f"[WARNING] No title column found in {db_name}. Skipping.")
                continue

            for _, row in df.iterrows():
                pid = row["page_id"]
                title_val = row.get(title_col, None)

                if pd.notna(title_val):
                    self.page_id_lookup[pid] = title_val
                    if verbose:
                        print(f"[DEBUG] Mapped {pid} -> {title_val}")

        if verbose:
            print("\n[DEBUG] Completed Page ID -> Title mapping.")
            print(f"[DEBUG] Total IDs stored: {len(self.page_id_lookup)}\n")

        if verbose:
            print("\n[DEBUG] Replacing relation IDs with human-readable names...")

        for db_name, df in self.metadata.items():
            if verbose:
                print(f"[DEBUG] Processing {db_name} for relation replacements...")

            for col in df.columns:
                if df[col].dtype == "object":
                    for idx, value in df[col].items():
                        if pd.notna(value) and isinstance(value, str):
                            matches = uuid_pattern.findall(value)
                            if not matches:
                                continue

                            parts = [
                                self.page_id_lookup.get(part.strip(), part.strip())
                                for part in value.split(",")
                            ]
                            new_value = ", ".join(parts)

                            if new_value != value:
                                df.at[idx, col] = new_value
                                if verbose:
                                    print(
                                        f"[DEBUG] Updated {db_name}.{col} (Row {idx}): "
                                        f"'{value}' → '{new_value}'"
                                    )

        if verbose:
            print("\n[DEBUG] Completed relation replacements.\n")

    # ---------------------------
    # Schema map for relations
    # ---------------------------
    def map_database_relations(self):
        rel_map = {}
        for db_name, db_id in self.databases.items():
            if not db_id:
                continue
            try:
                db_schema = self.notion.databases.retrieve(database_id=db_id)
            except Exception as e:
                print(
                    f"[WARN] couldn't retrieve schema for {db_name} ({db_id}): {e}"
                )
                continue

            rels = {}
            for prop_name, prop_details in db_schema.get("properties", {}).items():
                if prop_details.get("type") == "relation":
                    rels[prop_name] = {
                        "database_id": prop_details["relation"].get("database_id"),
                        "data_source_id": prop_details["relation"].get("data_source_id"),
                    }
            rel_map[db_name] = rels

        self.relations_map = rel_map
        print("Database Relations Map:", json.dumps(self.relations_map, indent=2))

    # ---------------------------
    # Convenience getters
    # ---------------------------
    def get_metadata(self, db_name, update_relations=False):
        if db_name in self.metadata:
            if update_relations:
                self.find_relations()
            return self.metadata[db_name]
        return None

    def print_metadata(self):
        for db_name, df in self.metadata.items():
            print(f"Dataframe: {db_name}")
            print(df)
            print()

    # ---------------------------
    # Your deployment summarizer
    # ---------------------------
    def extract_essential_metadata(self, deployment_id):
        """
        Extract deployment lat/long/tz and logger info for a given Deployment ID.
        """

        print(f"🔍 Extracting essential metadata for Deployment ID: {deployment_id}")

        deployment_db = self.get_metadata("deployment_DB")
        recording_db = self.get_metadata("recording_DB")
        logger_db = self.get_metadata("logger_DB")
        procedure_db = self.get_metadata("procedure_DB")
        location_db = self.get_metadata("location_DB")

        def _normalize_col_name(name):
            return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

        def _find_column(df, aliases):
            if df is None or not isinstance(df, pd.DataFrame):
                return None
            normalized = {_normalize_col_name(col): col for col in df.columns}
            for alias in aliases:
                match = normalized.get(_normalize_col_name(alias))
                if match is not None:
                    return match
            return None

        def _to_scalar_or_none(value):
            if pd.isna(value):
                return None
            if isinstance(value, np.generic):
                return value.item()
            return value

        # Step 1: recordings for this deployment
        if (
            deployment_db is None
            or "Deployment ID" not in deployment_db.columns
            or "Recordings" not in deployment_db.columns
        ):
            print("⚠ Required columns not found in deployment_DB.")
            return None, None

        deployment_recordings = deployment_db.loc[
            deployment_db["Deployment ID"] == deployment_id, "Recordings"
        ].dropna()

        if deployment_recordings.empty:
            print(f"⚠ No recordings found for Deployment ID: {deployment_id}")
            return None, None

        recording_ids = deployment_recordings.iloc[0].split(", ")

        recording_tz_overrides = {}
        config_path = os.getenv("CONFIG_PATH")
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
                recording_tz_overrides = (
                    (config.get("recording_metadata_overrides") or {}).get("time_zones")
                    or {}
                )
            except Exception as e:
                print(f"⚠ Failed to load recording time zone overrides from {config_path}: {e}")

        # Step 2: logger IDs from recording IDs
        logger_ids = [
            rec_id.split("_")[2]
            for rec_id in recording_ids
            if len(rec_id.split("_")) > 2
        ]

        # Step 3: map Recording ID -> Montage ID
        montage_map = {}
        if (
            recording_db is not None
            and "Recording ID" in recording_db.columns
            and "Montage ID" in recording_db.columns
        ):
            montage_map = (
                recording_db.set_index("Recording ID")["Montage ID"].to_dict()
            )

        def _lookup_recording_time_zone(logger_id):
            if (
                recording_db is None
                or "Recording ID" not in recording_db.columns
                or "Logger ID" not in recording_db.columns
                or "Time Zone" not in recording_db.columns
            ):
                return None

            rec_rows = recording_db.loc[
                recording_db["Recording ID"].isin(recording_ids)
                & (recording_db["Logger ID"] == logger_id),
                ["Time Zone"],
            ]
            if rec_rows.empty:
                return None
            tz_series = rec_rows["Time Zone"].dropna()
            return tz_series.iloc[0] if not tz_series.empty else None

        # Step 4: assemble logger info
        loggers_used = []
        for logger_id in logger_ids:
            logger_entry = {}

            if (
                logger_db is not None
                and "Logger ID" in logger_db.columns
                and "Manufacturer" in logger_db.columns
            ):
                sub = logger_db.loc[
                    logger_db["Logger ID"] == logger_id,
                    ["Logger ID", "Manufacturer"],
                ]
                if not sub.empty:
                    logger_entry = sub.iloc[0].to_dict()

            logger_entry["Montage ID"] = next(
                (montage_map.get(rid) for rid in recording_ids if logger_id in rid),
                None,
            )
            tz_override = recording_tz_overrides.get(logger_id)
            tz_from_recording = _lookup_recording_time_zone(logger_id)
            logger_entry["Time Zone"] = tz_override or tz_from_recording

            if logger_entry:
                loggers_used.append(logger_entry)

        # Step 5: collect all procedure rows for this deployment and store them by unique procedure ID.
        procedure_rows = pd.DataFrame()
        procedures_series = pd.Series(dtype=object)
        if "Procedures" in deployment_db.columns:
            procedures_series = deployment_db.loc[
                deployment_db["Deployment ID"] == deployment_id, "Procedures"
            ].dropna()

        candidate_procedure_ids = []
        if not procedures_series.empty:
            candidate_procedure_ids = procedures_series.iloc[0].split(", ")

        if procedure_db is not None and isinstance(procedure_db, pd.DataFrame):
            if "Deployment ID" in procedure_db.columns:
                procedure_rows = procedure_db.loc[
                    procedure_db["Deployment ID"] == deployment_id
                ].copy()
            elif candidate_procedure_ids and "Procedure ID" in procedure_db.columns:
                procedure_rows = procedure_db.loc[
                    procedure_db["Procedure ID"].isin(candidate_procedure_ids)
                ].copy()

        procedure_id_col = _find_column(procedure_rows, ["Procedure ID", "Name", "name"])
        procedure_location_id_col = _find_column(procedure_rows, ["Location ID", "location_id"])
        procedure_start_dt_col = _find_column(
            procedure_rows,
            ["Start Datetime", "start_datetime", "start date time"],
        )
        procedure_end_dt_col = _find_column(
            procedure_rows,
            ["End Datetime", "end_datetime", "end date time"],
        )
        mass_kg_col = _find_column(
            procedure_rows,
            ["mass_kg", "mass kg", "mass (kg)", "procedure_info_mass", "mass"],
        )
        mass_estimated_kg_col = _find_column(
            procedure_rows,
            ["mass_estimated_kg", "mass estimated kg", "estimated mass kg", "estimated mass (kg)"],
        )
        location_name_col = _find_column(location_db, ["Location Name", "location_name", "Name", "name"])
        location_latitude_col = _find_column(location_db, ["Latitude", "latitude", "lat"])
        location_longitude_col = _find_column(location_db, ["Longitude", "longitude", "lon", "long"])

        procedure_info = {}
        preferred_attachment_id = None
        if not procedure_rows.empty and procedure_id_col is not None:
            procedure_rows = procedure_rows.drop_duplicates(subset=[procedure_id_col], keep="first")

            for _, row in procedure_rows.iterrows():
                procedure_key = str(row.get(procedure_id_col, "")).strip()
                if not procedure_key:
                    continue

                entry = {}
                for col in procedure_rows.columns:
                    value = _to_scalar_or_none(row[col])
                    entry[_normalize_col_name(col)] = "" if value is None else value

                mass_kg_value = _to_scalar_or_none(row[mass_kg_col]) if mass_kg_col is not None else None
                mass_estimated_value = _to_scalar_or_none(row[mass_estimated_kg_col]) if mass_estimated_kg_col is not None else None
                start_dt_value = pd.to_datetime(
                    row[procedure_start_dt_col], errors="coerce"
                ) if procedure_start_dt_col is not None else pd.NaT
                end_dt_value = pd.to_datetime(
                    row[procedure_end_dt_col], errors="coerce"
                ) if procedure_end_dt_col is not None else pd.NaT
                if pd.notna(start_dt_value) and pd.isna(end_dt_value):
                    end_dt_value = start_dt_value + pd.Timedelta(hours=1)

                procedure_location_id = _to_scalar_or_none(
                    row[procedure_location_id_col] if procedure_location_id_col is not None else None
                )
                location_name_value = ""
                location_latitude_value = np.nan
                location_longitude_value = np.nan
                if (
                    procedure_location_id is not None
                    and location_db is not None
                    and isinstance(location_db, pd.DataFrame)
                    and "Location ID" in location_db.columns
                ):
                    loc_match = location_db.loc[
                        location_db["Location ID"] == procedure_location_id
                    ]
                    if not loc_match.empty:
                        loc_row = loc_match.iloc[0]
                        if location_name_col is not None and pd.notna(loc_row.get(location_name_col)):
                            location_name_value = str(loc_row.get(location_name_col))
                        if location_latitude_col is not None and pd.notna(loc_row.get(location_latitude_col)):
                            location_latitude_value = _to_scalar_or_none(loc_row.get(location_latitude_col))
                        if location_longitude_col is not None and pd.notna(loc_row.get(location_longitude_col)):
                            location_longitude_value = _to_scalar_or_none(loc_row.get(location_longitude_col))

                entry["id"] = procedure_key
                entry["mass"] = np.nan if mass_kg_value is None else mass_kg_value
                entry["mass_kg"] = np.nan if mass_kg_value is None else mass_kg_value
                entry["mass_estimated_kg"] = np.nan if mass_estimated_value is None else mass_estimated_value
                entry["start_datetime"] = "" if pd.isna(start_dt_value) else start_dt_value.isoformat()
                entry["end_datetime"] = "" if pd.isna(end_dt_value) else end_dt_value.isoformat()
                entry["location_name"] = location_name_value
                entry["location_latitude"] = location_latitude_value
                entry["location_longitude"] = location_longitude_value
                procedure_info[procedure_key] = entry

                if procedure_key.endswith("_attachment") and preferred_attachment_id is None:
                    preferred_attachment_id = procedure_key

        # Step 6: use that to get Location ID
        location_id = None
        if (
            preferred_attachment_id
            and procedure_db is not None
            and "Procedure ID" in procedure_db.columns
            and "Location ID" in procedure_db.columns
        ):
            loc_series = procedure_db.loc[
                procedure_db["Procedure ID"] == preferred_attachment_id, "Location ID"
            ].dropna()
            if not loc_series.empty:
                location_id = loc_series.iloc[0]

        # Step 7: from Location DB → lat / lon / tz
        deployment_latitude = None
        deployment_longitude = None
        time_zone = None

        if (
            location_id
            and location_db is not None
            and "Location ID" in location_db.columns
            and "Latitude" in location_db.columns
            and "Longitude" in location_db.columns
            and "Time Zone" in location_db.columns
        ):
            loc_row = location_db.loc[
                location_db["Location ID"] == location_id,
                ["Latitude", "Longitude", "Time Zone"],
            ]
            if not loc_row.empty:
                deployment_latitude = float(loc_row["Latitude"].iloc[0])
                deployment_longitude = float(loc_row["Longitude"].iloc[0])
                time_zone = loc_row["Time Zone"].iloc[0]

        # Step 8: bundle
        deployment_date = (
            deployment_id.split("_")[0] if "_" in deployment_id else None
        )

        deployment_info = {
            "Deployment Date": deployment_date,
            "Deployment Latitude": deployment_latitude,
            "Deployment Longitude": deployment_longitude,
            "Time Zone": time_zone,
            "Procedure Info": procedure_info,
        }

        print(f"📍 Deployment Metadata: {deployment_info}")
        print(f"📟 Loggers Used: {loggers_used}")

        return deployment_info, loggers_used
