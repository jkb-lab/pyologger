import os
import json
from typing import Any, Optional, List, Dict, Union

class ConfigManager:
    def __init__(self, deployment_folder: str, deployment_id: str):
        """
        Initializes ConfigManager with the path to config_log.json inside the dataset folder.
        
        The dataset folder is assumed to be the parent directory of the deployment folder.
        """
        self.deployment_folder = deployment_folder
        self.deployment_id = deployment_id
        self.dataset_folder = os.path.dirname(deployment_folder)  # Parent directory as dataset folder
        self.config_log_path = os.path.join(self.dataset_folder, "deployment_config_log.json")

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
        initial_config = [{
            "deployment_id": self.deployment_id,
            "deployment_folder_path": self.deployment_folder,
            "logger_ids": [],
            "settings": {}
        }]
        with open(self.config_log_path, "w") as file:
            json.dump(initial_config, file, indent=4)
        print(f"Initialized new config log at {self.config_log_path}")

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

    def _ensure_deployment_entry(self):
        """Ensures the deployment exists in the config log. Adds it if missing."""
        config_log = self._load_config()
        for entry in config_log:
            if entry.get("deployment_id") == self.deployment_id:
                return  # Deployment already exists
        
        # Add the deployment if it was missing
        new_deployment_entry = {
            "deployment_id": self.deployment_id,
            "deployment_folder_path": self.deployment_folder,
            "logger_ids": [],
            "settings": {}
        }
        config_log.append(new_deployment_entry)
        self._save_config(config_log)
        print(f"Added missing deployment '{self.deployment_id}' to config log.")

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
        self._ensure_deployment_entry()
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
        config_log = self._load_config()
        
        for entry in config_log:
            if entry["deployment_id"] == deployment_id:
                if section:
                    settings = entry.get(section, {})
                    return {var: settings.get(var) for var in variable_names}
                else:
                    return {var: entry.get(var) for var in variable_names}
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
