import os
import json
from typing import Any, Optional, Dict, Union

class MontageManager:
	def __init__(self, montage_folder: str):
		"""
		Initializes MontageManager with the path to montage_log.json inside the montage folder.
		"""
		self.montage_folder = montage_folder
		self.montage_log_path = os.path.join(montage_folder, "montage_log.json")

		# Ensure montage folder exists
		os.makedirs(self.montage_folder, exist_ok=True)

		# Initialize montage log if it doesn't exist
		if not os.path.exists(self.montage_log_path):
			self._initialize_montage_log()

	def _initialize_montage_log(self):
		"""Creates a new montage log file inside the montage folder."""
		initial_montage_log = {}
		with open(self.montage_log_path, "w") as file:
			json.dump(initial_montage_log, file, indent=4)
		print(f"Initialized new montage log at {self.montage_log_path}")

	def _load_montage_log(self) -> Dict[str, Any]:
		"""Loads the montage log from the JSON file."""
		if os.path.exists(self.montage_log_path):
			with open(self.montage_log_path, "r") as file:
				return json.load(file)
		return {}

	def _save_montage_log(self, montage_log: Dict[str, Any]):
		"""Saves the provided montage log back to the JSON file."""
		with open(self.montage_log_path, "w") as file:
			json.dump(montage_log, file, indent=4)
		print(f"Updated JSON saved to {self.montage_log_path}")

	def add_montage(self, manufacturer: str, montage_id: str, montage_data: Dict[str, Any]):
		"""
		Adds a new montage entry to the montage log.
		
		Parameters:
		- manufacturer (str): The manufacturer of the montage (e.g., "CATS", "UFI").
		- montage_id (str): The name of the montage.
		- montage_data (Dict[str, Any]): The data for the montage.
		"""
		montage_log = self._load_montage_log()

		if manufacturer not in montage_log:
			montage_log[manufacturer] = {}

		montage_log[manufacturer][montage_id] = montage_data

		self._save_montage_log(montage_log)
		print(f"Added montage '{montage_id}' under manufacturer '{manufacturer}'.")

	def get_montage(self, manufacturer: str, montage_id: str) -> Optional[Dict[str, Any]]:
		"""
		Retrieves a montage entry from the montage log.
		
		Parameters:
		- manufacturer (str): The manufacturer of the montage (e.g., "CATS", "UFI").
		- montage_id (str): The name of the montage.
		
		Returns:
		- Optional[Dict[str, Any]]: The data for the montage if found, otherwise None.
		"""
		montage_log = self._load_montage_log()
		return montage_log.get(manufacturer, {}).get(montage_id)

	def remove_montage(self, manufacturer: str, montage_id: str):
		"""
		Removes a montage entry from the montage log.
		
		Parameters:
		- manufacturer (str): The manufacturer of the montage (e.g., "CATS", "UFI").
		- montage_id (str): The name of the montage.
		"""
		montage_log = self._load_montage_log()

		if manufacturer in montage_log and montage_id in montage_log[manufacturer]:
			del montage_log[manufacturer][montage_id]
			self._save_montage_log(montage_log)
			print(f"Removed montage '{montage_id}' from manufacturer '{manufacturer}'.")
		else:
			print(f"Montage '{montage_id}' not found in manufacturer '{manufacturer}'.")

# Example usage:
# montage_manager = MontageManager("/path/to/montage/folder")
# montage_manager.add_montage("CATS", "new_montage", {"sensor": "data"})
# montage_data = montage_manager.get_montage("CATS", "new_montage")
# montage_manager.remove_montage("CATS", "new_montage")