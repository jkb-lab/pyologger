import os
import json
from typing import Any, Optional, List, Dict, Union

class ConfigManager:
    def __init__(self, deployment_folder: str, deployment_id: str):
        """Initializes ConfigManager with the path to config_log.json based on the deployment folder and deployment_id."""
        self.config_log_path = os.path.join(os.path.dirname(deployment_folder), 'deployment_config_log.json')
        self.deployment_id = deployment_id

    def _load_config(self) -> List[Dict[str, Any]]:
        """Loads the config log from the JSON file."""
        if os.path.exists(self.config_log_path):
            with open(self.config_log_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Config log file not found at {self.config_log_path}")

    def _save_config(self, config_log: List[Dict[str, Any]]):
        """Saves the provided config log back to the JSON file."""
        with open(self.config_log_path, 'w') as f:
            json.dump(config_log, f, indent=4)

    def add_deployment_to_log(self, logger_ids: list):
        if not isinstance(self.config_log_path, (str, os.PathLike)):
            raise TypeError("The 'config_log_path' argument must be a string or os.PathLike object.")

        if os.path.exists(self.config_log_path):
            with open(self.config_log_path, 'r') as file:
                log_data = json.load(file)
        else:
            log_data = []

        # Check if deployment ID already exists
        for entry in log_data:
            if entry.get("deployment_id") == self.deployment_id:
                print(f"Deployment {self.deployment_id} already exists in the log.")
                return

        # If not, append the new deployment entry
        new_deployment_entry = {
            "deployment_id": self.deployment_id,
            "logger_ids": logger_ids,
            "settings": {}
        }
        log_data.append(new_deployment_entry)

        # Save the updated log back to the JSON file
        with open(self.config_log_path, 'w') as file:
            json.dump(log_data, file, indent=4)

        print(f"Deployment {self.deployment_id} added with loggers {logger_ids}.")

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

        # Ensure entries is a dictionary if adding a single key-value pair
        if isinstance(entries, str) and value is not None:
            entries = {entries: value}

        for entry in config_log:
            if entry["deployment_id"] == deployment_id:
                if section:
                    if section not in entry:
                        entry[section] = {}
                    entry[section].update(entries)
                else:
                    entry.update(entries)
                break
        else:
            raise ValueError(f"Deployment ID '{deployment_id}' not found in config log.")
        
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
                if section:
                    if section in entry and key in entry[section]:
                        del entry[section][key]
                elif key in entry:
                    del entry[key]
                break
        else:
            raise ValueError(f"Deployment ID '{deployment_id}' not found in config log.")
        
        self._save_config(config_log)
        print(f"Removed {key} from deployment '{deployment_id}' under '{section or 'top level'}'.")
