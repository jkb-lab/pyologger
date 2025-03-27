import os
import pandas as pd
import json
from typing import List, Any, Optional, Dict, Union

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

	def convert_df_to_montage_dict(self, montage_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
		"""
		Converts a DataFrame of montage channel mappings to the JSON-compatible dictionary format.

		Expected columns:
		- original_channel_id
		- original_unit
		- manufacturer_sensor_name
		- standardized_channel_id
		- standardized_unit
		- standardized_sensor_type
		"""
		montage_dict = {}
		required_cols = [
			"original_channel_id", "original_unit", "manufacturer_sensor_name",
			"standardized_channel_id", "standardized_unit", "standardized_sensor_type"
		]

		if not all(col in montage_df.columns for col in required_cols):
			raise ValueError(f"❌ Missing one or more required columns in montage_df: {required_cols}")

		for _, row in montage_df.iterrows():
			montage_dict[row["original_channel_id"]] = {
				"original_unit": row["original_unit"],
				"manufacturer_sensor_name": row["manufacturer_sensor_name"],
				"standardized_channel_id": row["standardized_channel_id"],
				"standardized_unit": row["standardized_unit"],
				"standardized_sensor_type": row["standardized_sensor_type"],
			}

		return montage_dict
	
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

	def add_missing_montages_per_logger(
		self,
		loggers_used: List[Dict[str, str]],
		montage_inputs: Dict[str, Union[pd.DataFrame, str]]
	) -> List[Dict[str, str]]:
		"""
		Adds missing montages per logger using individual montage_df or CSV path per Logger ID.

		Parameters:
		- loggers_used: list of dicts with 'Logger ID', 'Manufacturer', 'Montage ID'
		- montage_inputs: dict mapping Logger ID -> montage_df or CSV path

		Returns:
		- montages_metadata: list of metadata about added or existing montages
		"""
		montages_metadata = []

		for logger in loggers_used:
			logger_id = logger["Logger ID"]
			manufacturer = logger["Manufacturer"]
			montage_id = logger["Montage ID"]

			# Check for existing montage
			existing = self.get_montage(manufacturer, montage_id)
			if existing:
				print(f"✅ Found existing montage for {manufacturer} - {montage_id} ({len(existing)} channels)")
				montages_metadata.append({
					"Logger ID": logger_id,
					"Manufacturer": manufacturer,
					"Montage ID": montage_id,
					"Number of Channels": len(existing),
					"Status": "existing"
				})
				continue

			# Get montage_df or CSV path
			montage_input = montage_inputs.get(logger_id)
			if montage_input is None:
				print(f"⚠️ No montage input found for Logger ID: {logger_id}. Skipping.")
				continue

			try:
				if isinstance(montage_input, str):
					montage_df = pd.read_csv(montage_input)
					print(f"📄 Loaded montage from CSV for {logger_id}: {montage_input}")
				elif isinstance(montage_input, pd.DataFrame):
					montage_df = montage_input
					print(f"📋 Using provided DataFrame for {logger_id}")
				else:
					raise ValueError("montage_input must be a DataFrame or path to CSV")

				montage_dict = self.convert_df_to_montage_dict(montage_df)
				self.add_montage(manufacturer, montage_id, montage_dict)

				montages_metadata.append({
					"Logger ID": logger_id,
					"Manufacturer": manufacturer,
					"Montage ID": montage_id,
					"Number of Channels": len(montage_dict),
					"Status": "added"
				})
				print(f"➕ Added montage for {manufacturer} - {montage_id} ({len(montage_dict)} channels)")

			except Exception as e:
				print(f"❌ Failed to create montage for {logger_id}: {e}")

		return montages_metadata
# Example usage:
# montage_manager = MontageManager("/path/to/montage/folder")
# montage_manager.add_montage("CATS", "new_montage", {"sensor": "data"})
# montage_data = montage_manager.get_montage("CATS", "new_montage")
# montage_manager.remove_montage("CATS", "new_montage")