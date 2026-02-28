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
		- manufacturer_signal_name
		- standardized_channel_id
		- standardized_unit
		- parent_signal
		"""
		montage_dict = {}
		required_cols = [
			"original_channel_id", "original_unit", "manufacturer_signal_name",
			"standardized_channel_id", "standardized_unit", "parent_signal"
		]

		if not all(col in montage_df.columns for col in required_cols):
			raise ValueError(f"❌ Missing one or more required columns in montage_df: {required_cols}")

		df = montage_df.copy()
		df["original_channel_id"] = df["original_channel_id"].astype(str).str.strip()
		df["parent_signal"] = df["parent_signal"].astype(str).str.strip().str.lower()
		df["standardized_channel_id"] = df["standardized_channel_id"].astype(str).str.strip().str.lower()

		# A montage dict is keyed by original_channel_id; keep last row if repeated.
		dups = df[df.duplicated("original_channel_id", keep=False)].copy()
		if not dups.empty:
			conflicts = []
			for channel_id, group in dups.groupby("original_channel_id", dropna=False):
				unique_defs = group[[
					"standardized_channel_id",
					"parent_signal",
					"manufacturer_signal_name",
				]].drop_duplicates()
				if len(unique_defs) > 1:
					conflicts.append((channel_id, unique_defs.to_dict("records")))

			if conflicts:
				preview = ", ".join([c[0] for c in conflicts[:10]])
				raise ValueError(
					"❌ Conflicting duplicate original_channel_id rows in montage input. "
					"Each original_channel_id must map to exactly one parent_signal/standardized_channel_id "
					f"before writing JSON. Examples: {preview}"
				)

			# Exact duplicates are fine; keep one.
			df = df.drop_duplicates(subset=["original_channel_id"], keep="last")

		for _, row in df.iterrows():
			montage_dict[row["original_channel_id"]] = {
				"original_unit": row["original_unit"],
				"manufacturer_signal_name": row["manufacturer_signal_name"],
				"standardized_channel_id": row["standardized_channel_id"],
				"standardized_unit": row["standardized_unit"],
				"parent_signal": row["parent_signal"],
			}

		return montage_dict

	@staticmethod
	def _prepare_montage_df(
		montage_df: pd.DataFrame,
		manufacturer: str,
		montage_id: str
	) -> pd.DataFrame:
		"""
		Normalize dataframe columns and optionally filter rows when input came
		directly from original_channel_db-style tables.
		"""
		df = montage_df.copy()
		col_map = {
			"Original Channel ID": "original_channel_id",
			"Original Unit": "original_unit",
			"Manufacturer Signal Name": "manufacturer_signal_name",
			"Standardized Channel ID": "standardized_channel_id",
			"Standardized Unit": "standardized_unit",
			"Parent Signal": "parent_signal",
			"Montages": "montages",
			"Manufacturer": "manufacturer",
		}
		df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

		if "manufacturer" in df.columns:
			manu_mask = df["manufacturer"].astype(str).str.strip().str.lower() == str(manufacturer).strip().lower()
			if manu_mask.any():
				df = df.loc[manu_mask].copy()

		if "montages" in df.columns:
			target = str(montage_id).strip().lower()
			montage_mask = df["montages"].astype(str).str.lower().apply(
				lambda x: target in [m.strip() for m in x.split(",")]
			)
			if montage_mask.any():
				df = df.loc[montage_mask].copy()

		return df
	
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

			# Get montage_df or CSV path
			montage_input = montage_inputs.get(logger_id)
			if montage_input is None:
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

				montage_df = self._prepare_montage_df(
					montage_df,
					manufacturer=manufacturer,
					montage_id=montage_id,
				)
				montage_dict = self.convert_df_to_montage_dict(montage_df)
				existing = self.get_montage(manufacturer, montage_id)

				if existing == montage_dict:
					print(f"ℹ️ no changes detected in montage {montage_id}")
					montages_metadata.append({
						"Logger ID": logger_id,
						"Manufacturer": manufacturer,
						"Montage ID": montage_id,
						"Number of Channels": len(montage_dict),
						"Status": "unchanged"
					})
					continue

				self.add_montage(manufacturer, montage_id, montage_dict)

				status = "updated" if existing else "added"
				print_prefix = "♻️ Updated" if existing else "➕ Added"
				montages_metadata.append({
					"Logger ID": logger_id,
					"Manufacturer": manufacturer,
					"Montage ID": montage_id,
					"Number of Channels": len(montage_dict),
					"Status": status
				})
				print(f"{print_prefix} montage for {manufacturer} - {montage_id} ({len(montage_dict)} channels)")

			except Exception as e:
				print(f"❌ Failed to create montage for {logger_id}: {e}")

		return montages_metadata
# Example usage:
# montage_manager = MontageManager("/path/to/montage/folder")
# montage_manager.add_montage("CATS", "new_montage", {"parent_signal": "data"})
# montage_data = montage_manager.get_montage("CATS", "new_montage")
# montage_manager.remove_montage("CATS", "new_montage")
