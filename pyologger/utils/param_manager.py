import os
import json
from typing import Any, Optional, List, Dict, Union
import pandas as pd

class ParamManager:
    DEFAULTS_DEPLOYMENT_ID = "__dataset_defaults__"

    def __init__(self, deployment_folder: str, deployment_id: str):
        """
        Initializes ParamManager with the path to config_log.json inside the dataset folder.
        
        The dataset folder is assumed to be the parent directory of the deployment folder.
        """
        self.deployment_folder = deployment_folder
        self.deployment_id = deployment_id
        self.dataset_folder = os.path.dirname(deployment_folder)  # Parent directory as dataset folder
        self.config_log_path = os.path.join(self.dataset_folder, "parameter_log.json")

        # Ensure dataset folder exists
        os.makedirs(self.dataset_folder, exist_ok=True)

        # Initialize config log if it doesn't exist
        if not os.path.exists(self.config_log_path):
            self._initialize_config()
        else:
            # Ensure the deployment exists in the config log
            self._ensure_deployment_entry()

    def _initialize_config(self):
        """Creates a new config file inside the dataset folder with the current deployment."""
        initial_config = [self._build_defaults_entry(), self._build_deployment_entry(self.deployment_id, self.deployment_folder)]
        with open(self.config_log_path, "w") as file:
            json.dump(initial_config, file, indent=4)
        print(f"Initialized new config log at {self.config_log_path}")

    def _build_defaults_entry(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.DEFAULTS_DEPLOYMENT_ID,
            "deployment_folder_path": self.dataset_folder,
            "logger_ids": [],
            "settings": {}
        }

    @staticmethod
    def _build_deployment_entry(deployment_id: str, deployment_folder: str) -> Dict[str, Any]:
        return {
            "deployment_id": deployment_id,
            "deployment_folder_path": deployment_folder,
            "logger_ids": [],
            "settings": {}
        }

    def _load_config(self) -> List[Dict[str, Any]]:
        """Loads the config log from the JSON file."""
        if os.path.exists(self.config_log_path):
            with open(self.config_log_path, "r") as f:
                return json.load(f)
        return []

    def _save_config(self, config_log: List[Dict[str, Any]]):
        """Saves the provided config log back to the JSON file."""
        with open(self.config_log_path, "w") as f:
            json.dump(config_log, f, indent=4)

    def _ensure_defaults_entry(self):
        config_log = self._load_config()
        has_defaults = any(entry.get("deployment_id") == self.DEFAULTS_DEPLOYMENT_ID for entry in config_log)
        if not has_defaults:
            config_log.insert(0, self._build_defaults_entry())
            self._save_config(config_log)

    def _ensure_deployment_entry(self):
        """Ensures the deployment exists in the config log. Adds it if missing."""
        self._ensure_defaults_entry()
        config_log = self._load_config()
        for entry in config_log:
            if entry.get("deployment_id") == self.deployment_id:
                return  # Deployment already exists
        
        # Add the deployment if it was missing
        new_deployment_entry = self._build_deployment_entry(self.deployment_id, self.deployment_folder)
        config_log.append(new_deployment_entry)
        self._save_config(config_log)
        print(f"Added missing deployment '{self.deployment_id}' to config log.")

    def _get_dataset_defaults(self, section: Optional[str] = None) -> Dict[str, Any]:
        config_log = self._load_config()
        defaults_entry = next(
            (entry for entry in config_log if entry.get("deployment_id") == self.DEFAULTS_DEPLOYMENT_ID),
            None
        )
        if not defaults_entry:
            return {}
        if section:
            value = defaults_entry.get(section, {})
            return value if isinstance(value, dict) else {}
        return defaults_entry

    @staticmethod
    def _merge_non_none(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge dicts while treating None in overrides as "not explicitly set".
        This preserves dataset defaults unless a deployment value is provided.
        """
        merged = dict(defaults or {})
        for key, value in (overrides or {}).items():
            if value is not None:
                merged[key] = value
        return merged

    def set_dataset_defaults(self, entries: Dict[str, Any], section: Optional[str] = None):
        """Set dataset-wide defaults that apply to all deployments unless overridden."""
        self._ensure_defaults_entry()
        self.add_to_config(entries=entries, section=section, deployment_id=self.DEFAULTS_DEPLOYMENT_ID)

    def add_to_config(self, entries: Union[Dict[str, Any], str], value: Optional[Any] = None, section: Optional[str] = None, deployment_id: Optional[str] = None):
        """
        Adds or updates key-value pairs in the specified section of the config_log JSON file.
        
        Parameters:
        - entries (Dict or str): Dictionary of key-value pairs to add/update, or a single key as a string.
        - value (Any, optional): If `entries` is a single key, this is the value to set.
        - section (str, optional): Section within the config to add entries to. Defaults to top level.
        - deployment_id (str, optional): Specific deployment ID to target. Defaults to class-level deployment_id.
        """
        deployment_id = deployment_id or self.deployment_id
        config_log = self._load_config()

        # Ensure the deployment exists before modifying
        if deployment_id != self.DEFAULTS_DEPLOYMENT_ID:
            self._ensure_deployment_entry()
        else:
            self._ensure_defaults_entry()
        config_log = self._load_config()  # Reload after ensuring entry exists

        # Ensure entries is a dictionary if adding a single key-value pair
        if isinstance(entries, str) and value is not None:
            entries = {entries: value}

        for entry in config_log:
            if entry["deployment_id"] == deployment_id:
                if section:
                    entry.setdefault(section, {}).update(entries)
                else:
                    entry.update(entries)
                break
        
        self._save_config(config_log)
        print(f"Updated entries in deployment '{deployment_id}' under '{section or 'top level'}'.")

    def get_from_config(self, variable_names: List[str], section: Optional[str] = None, deployment_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieves values for specified variable names from the config_log JSON file."""
        deployment_id = deployment_id or self.deployment_id
        if deployment_id != self.DEFAULTS_DEPLOYMENT_ID:
            self._ensure_deployment_entry()
        else:
            self._ensure_defaults_entry()
        config_log = self._load_config()
        
        for entry in config_log:
            if entry["deployment_id"] == deployment_id:
                if section:
                    defaults = self._get_dataset_defaults(section=section) if deployment_id != self.DEFAULTS_DEPLOYMENT_ID else {}
                    settings = entry.get(section, {})
                    merged = self._merge_non_none(
                        defaults if isinstance(defaults, dict) else {},
                        settings if isinstance(settings, dict) else {}
                    )
                    return {var: merged.get(var) for var in variable_names}
                else:
                    defaults_entry = self._get_dataset_defaults(section=None) if deployment_id != self.DEFAULTS_DEPLOYMENT_ID else {}
                    merged = self._merge_non_none(
                        defaults_entry if isinstance(defaults_entry, dict) else {},
                        entry if isinstance(entry, dict) else {}
                    )
                    return {var: merged.get(var) for var in variable_names}
        raise ValueError(f"Deployment ID '{deployment_id}' not found in config log.")

    def remove_from_config(self, key: str, section: Optional[str] = None, deployment_id: Optional[str] = None):
        """Removes a key from the specified section or top level in the config_log JSON file."""
        deployment_id = deployment_id or self.deployment_id
        config_log = self._load_config()
        
        for entry in config_log:
            if entry["deployment_id"] == deployment_id:
                if section and section in entry and key in entry[section]:
                    del entry[section][key]
                elif key in entry:
                    del entry[key]
                break
        else:
            raise ValueError(f"Deployment ID '{deployment_id}' not found in config log.")
        
        self._save_config(config_log)
        print(f"Removed {key} from deployment '{deployment_id}' under '{section or 'top level'}'.")
        
    def get_logger_attachments(self) -> list:
        """Return logger_attachments for this deployment, falling back to selected_start/end_time.

        Returns a list of {"start": <str>, "end": <str>} dicts.  Merges dataset-default
        attachments with deployment-level overrides the same way get_from_config does.
        Falls back to constructing a single period from selected_start_time /
        selected_end_time when logger_attachments is absent.
        """
        result = self.get_from_config(
            ["logger_attachments", "selected_start_time", "selected_end_time"],
            section="settings",
        )
        attachments = result.get("logger_attachments")
        if attachments and isinstance(attachments, list) and len(attachments) > 0:
            return attachments

        # fallback: build a single period from the old scalar fields
        start = result.get("selected_start_time")
        end = result.get("selected_end_time")
        if start and end:
            return [{"start": str(start), "end": str(end)}]

        return []

    def get_or_create_chunk_grid(self, chunk_size_sec: int | None = None) -> list:
        """Return the existing chunk grid, or build and persist it if absent.

        Uses logger_attachments (via get_logger_attachments) to compute the
        grid.  The resolved chunk_size_sec is stored in settings for
        reproducibility.

        Returns the list of chunk dicts (see chunk_manager.compute_chunk_grid).
        Returns an empty list if no attachment windows can be determined.
        """
        from pyologger.utils.chunk_manager import compute_chunk_grid, load_chunks

        # return existing grid unchanged
        existing = load_chunks(self)
        if existing is not None:
            return existing

        attachments = self.get_logger_attachments()
        if not attachments:
            return []

        chunks, resolved_size = compute_chunk_grid(attachments, chunk_size_sec)

        # persist both the grid and the resolved chunk size
        self.add_to_config(
            entries={"hr_detection_chunk_size_sec": resolved_size},
            section="settings",
        )
        self.add_to_config(
            entries={"hr_peak_detection_chunks": chunks},
            section=None,
        )
        return chunks

    def export_config(self):
        """Exports the config log to a CSV file."""
        json_path = self.config_log_path

        # Load JSON data
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # Flatten JSON into a structured DataFrame
        data_list = []

        for entry in json_data:
            deployment_id = entry.get("deployment_id", None)
            settings = entry.get("settings", {})
            for key, value in settings.items():
                if isinstance(value, str) and "time" in key.lower():
                    try:
                        value = pd.to_datetime(value).tz_localize(None)
                    except Exception as e:
                        print(f"Error converting {key}: {e}")
                settings[key] = value
            
            row = {"deployment_id": deployment_id}
            for key, value in settings.items():
                row[key] = value
            
            data_list.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(data_list)

        # Define the path to save the CSV file
        csv_path = os.path.join(self.dataset_folder, 'parameter_log.csv')

        # Save the DataFrame as a CSV file
        df.to_csv(csv_path, index=False)

        print(f"CSV file saved at: {csv_path}")
