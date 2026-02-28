import pandas as pd
import numpy as np
import re
import warnings
import json
import pytz
from datetime import datetime
from pyologger.utils.time_manager import *

class BaseImporter:
    """Base class for handling manufacturer-specific processing."""

    def __init__(self, data_reader, logger_id):
        self.data_reader = data_reader
        self.logger_id = logger_id
        self.logger_manufacturer = self.data_reader.logger_info[logger_id]['Manufacturer']
        self.montage_id = self.data_reader.logger_info[logger_id]['Montage ID']
        self.expected_frequencies = {}  # Stores expected signal frequencies from .txt files
        self.montage_path = self.data_reader.montage_path

        # Load the custom JSON mapping for column names if available
        if self.montage_path:
            self.load_custom_mapping()
        
    def read_csv(self, csv_path):
        """Reads a CSV file with multiple encoding attempts."""
        # First, check if this is a Vectronics file with metadata header
        skiprows = 0
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
        except UnicodeDecodeError:
            with open(csv_path, 'r', encoding='ISO-8859-1') as f:
                first_line = f.readline().strip()
        
        # Detect Vectronics metadata row
        if (first_line.startswith("Collar:") or 
            first_line.startswith("DeviceID:") or 
            "Sensor range" in first_line):
            skiprows = 1
            print(f"   🔪 Detected Vectronics metadata row - skipping first row")
        
        # Try reading with different encodings
        encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                print(f"Attempting to read {csv_path} with encoding {encoding}")
                return pd.read_csv(csv_path, encoding=encoding, skiprows=skiprows)
            except UnicodeDecodeError as e:
                print(f"Error reading {csv_path} with encoding {encoding}: {e}")
        raise UnicodeDecodeError(f"Failed to read {csv_path} with available encodings.")

    def _read_file(self, path):
        """Dispatch to CSV or Parquet reader based on extension."""
        low = path.lower()
        if low.endswith(('.csv',)):
            # Uses BaseImporter.read_csv() (handles messy headers, comments, etc.)
            return self.read_csv(path)
        elif low.endswith(('.parquet', '.parq', '.pq')):
            return self._read_parquet(path)
        raise ValueError(f"Unsupported file type for: {path}")

    def _read_parquet(self, path):
        """Parquet reader with a safe engine fallback."""
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            # Fallback to fastparquet if pyarrow isn't available
            return pd.read_parquet(path, engine="fastparquet")

    def import_netcdf(data_reader, filepath):
            """Imports a NetCDF file to the pickle format used by pyologger."""
            
            print(f"NetCDF file imported from {filepath} and returning pickle object.")

    def import_edf(data_reader, filepath):
            """Imports a NetCDF file to the pickle format used by pyologger."""
            
            print(f"NetCDF file imported from {filepath} and returning pickle object.")
    
    def load_custom_mapping(self):
        """Loads custom JSON mapping for column names based on manufacturer and montage ID."""
        try:
            with open(self.montage_path, 'r') as json_file:
                full_mapping = json.load(json_file)
                print(f"Custom column mapping loaded from {self.montage_path}")

                # Ensure it's a dictionary
                if not isinstance(full_mapping, dict):
                    raise ValueError("Column mapping JSON is not a dictionary.")

                # Validate manufacturer
                manufacturer = self.logger_manufacturer
                if manufacturer not in full_mapping:
                    raise ValueError(f"Manufacturer '{manufacturer}' not found in column mapping.")

                # Validate montage ID
                montage_id = self.montage_id
                montage_id_text = str(montage_id).strip().lower() if montage_id is not None else ""
                if pd.isna(montage_id) or montage_id_text in {"", "nan", "none"}:
                    print(
                        f"No valid montage ID for logger '{self.logger_id}' "
                        f"({manufacturer}); proceeding without custom mapping."
                    )
                    self.montage = None
                    return
                if montage_id not in full_mapping[manufacturer]:
                    raise ValueError(f"Montage ID '{montage_id}' not found under manufacturer '{manufacturer}'.")

                # Extract only the relevant part of the mapping
                self.montage = full_mapping[manufacturer][montage_id]
                print(f"Column mapping loaded for manufacturer '{manufacturer}', montage '{montage_id}'.")
                print(f"Mapping content {self.montage}.")
        except FileNotFoundError:
            print(f"Custom mapping file not found at {self.montage_path}. Proceeding without it.")
            self.montage = None
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading or verifying JSON from {self.montage_path}: {e}")
            self.montage = None

    def rename_channels(self, channel_names):
        """Maps original channel IDs to standardized channel IDs."""
        channel_metadata = {}
        new_channels = {}
        used_names = {}
        name_parent_owner = {}

        def _slug(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

        def ensure_unique_name(candidate_name: str, source_column: str, parent_signal: str) -> str:
            """
            Guarantee that the standardized channel name we emit is unique.

            If another column already claimed the same standardized name,
            append a suffix so pandas doesn't create duplicate column labels.
            """
            count = used_names.get(candidate_name, 0)
            if count == 0:
                used_names[candidate_name] = 1
                name_parent_owner[candidate_name] = parent_signal
                return candidate_name

            # Prefer a deterministic parent-signal suffix when collision is across
            # different parent signals (e.g., ax in accelerometer vs corrected_acc).
            owner_parent = name_parent_owner.get(candidate_name)
            parent_slug = _slug(parent_signal) or "extra"
            owner_slug = _slug(owner_parent) if owner_parent else None

            if owner_slug and owner_slug != parent_slug:
                preferred_name = f"{candidate_name}__{parent_slug}"
                if preferred_name not in used_names:
                    used_names[preferred_name] = 1
                    return preferred_name

            unique_name = f"{candidate_name}__dup{count}"
            used_names[candidate_name] = count + 1
            print(
                f"⚠️ Duplicate standardized name '{candidate_name}' detected for '{source_column}'. "
                f"Using unique column id '{unique_name}'."
            )
            return unique_name

        print(f"🔄 Renaming channels for {self.logger_manufacturer} logger with montage ID {self.montage_id}.")

        # Be defensive: self.montage may be None if no mapping file was provided or it failed to load.
        if not getattr(self, 'montage', None):
            print(f"⚠️ No montage mapping loaded for logger {self.logger_id} ({self.logger_manufacturer}). Falling back to identity mapping.")
            self.montage = {}
        else:
            try:
                print(f"📜 Available mapping keys: {list(self.montage.keys())}")  # Show available keys in the mapping
            except Exception:
                # Defensive fallback if montage is not a dict-like object
                print("⚠️ Montage mapping is not iterable. Falling back to empty mapping.")
                self.montage = {}

        for original_name in channel_names:
            # Take out spaces from channel names and make lowercase
            clean_name = original_name.strip().lower().replace(" ", "_").replace(".", "")  # Normalize

            # Extract unit (if present)
            unit = None
            if "[" in clean_name and "]" in clean_name:
                name, unit = clean_name.split("[", 1)
                unit = unit.replace("]", "").strip().lower()
                clean_name = name.strip("_")

            if "(" in clean_name and ")" in clean_name:
                name, unit = clean_name.split("(", 1)
                unit = unit.replace(")", "").strip().lower()
                clean_name = f"{name.strip('_')}_{unit}"

            # Ensure local/UTC times are distinct
            if "local" in original_name.lower() and "local" not in clean_name:
                clean_name = f"{clean_name}_local"
            elif "utc" in original_name.lower() and "utc" not in clean_name:
                clean_name = f"{clean_name}_utc"

            # Dictionary lookup: if no mapping entry exists, fall back to using the cleaned name
            if clean_name in self.montage:
                mapping_info = self.montage[clean_name] or {}
                print(f"🎯 Found match in montage: {mapping_info}")  # Show full mapping info
            else:
                mapping_info = {}
                print(f"ℹ️ No mapping for '{clean_name}' — using cleaned name as standardized id.")

            mapped_name = mapping_info.get("standardized_channel_id", clean_name)
            parent_signal_raw = mapping_info.get("parent_signal")
            parent_signal = (parent_signal_raw if parent_signal_raw else "extra")
            parent_signal = str(parent_signal).strip().lower()
            unique_mapped_name = ensure_unique_name(mapped_name, original_name, parent_signal)

            original_unit_raw = mapping_info.get("original_unit")
            original_unit = (original_unit_raw if original_unit_raw else "unknown")
            original_unit = str(original_unit).strip().lower()

            standardized_unit_raw = mapping_info.get("standardized_unit")
            standardized_unit = (standardized_unit_raw if standardized_unit_raw else "unknown")
            standardized_unit = str(standardized_unit).strip()

            if unique_mapped_name != mapped_name:
                print(f"   ↳ Reassigned to unique standardized name: {unique_mapped_name}")

            print(f"📝 Dictionary: original name: {original_name} → standardized name: {unique_mapped_name} (Signal type: {parent_signal})")

            channel_metadata[unique_mapped_name] = {
                "original_name": original_name,
                "unit": original_unit or "unknown",
                "standardized_unit": standardized_unit or "unknown",
                "parent_signal": parent_signal
            }
            new_channels[original_name] = unique_mapped_name
        print(f"🐻Channel metadata: {channel_metadata}")
        print(f"✅ Final renamed channels: {new_channels}")
        return new_channels, channel_metadata


    def group_data_by_signals(self, df, logger_id, channel_metadata):
        """Groups data columns to signals and downsamples based on expected frequencies."""
        signal_groups = {}
        signal_info = {}

        def _norm_col(value):
            return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

        # Build robust column lookup once (handles case, whitespace, punctuation variants).
        df_cols = list(df.columns)
        df_lookup = {}
        for c in df_cols:
            c_str = str(c)
            candidates = {
                c_str,
                c_str.strip(),
                c_str.lower(),
                c_str.strip().lower(),
                _norm_col(c_str),
            }
            for key in candidates:
                if key and key not in df_lookup:
                    df_lookup[key] = c

        print("16a")
        signal_names = set()
        print(f"Channel metadata received for grouping: {channel_metadata}")
        for v in channel_metadata.values():
            try:
                signal_names.add(v['parent_signal'].strip().lower())
            except Exception as e:
                print(f"⚠️ Skipping invalid channel metadata entry: {v}. Error: {e}")

        for signal_name in signal_names:

            print("16b")
            if signal_name == 'extra':
                continue  # Skip 'extra' signal type

            if signal_name in self.data_reader.signal_data:
                print(f"Sensor '{signal_name}' has already been processed. Skipping reprocessing.")
                continue


            print("16c")
            # Group columns by signal and normalize channel ids back to base names.
            signal_orig_cols = []
            resolved_meta_keys = {}
            for std_col, meta in channel_metadata.items():
                if meta['parent_signal'].strip().lower() != signal_name:
                    continue
                # Prefer standardized channel id first; then fallback to original source name.
                source_name = meta.get("original_name")
                probes = [
                    std_col,
                    str(std_col).strip(),
                    str(std_col).lower(),
                    _norm_col(std_col),
                    source_name,
                    str(source_name).strip() if source_name is not None else None,
                    str(source_name).lower() if source_name is not None else None,
                    _norm_col(source_name) if source_name is not None else None,
                ]
                resolved = None
                for p in probes:
                    if not p:
                        continue
                    if p in df_lookup:
                        resolved = df_lookup[p]
                        break
                if resolved is not None and resolved not in signal_orig_cols:
                    signal_orig_cols.append(resolved)
                    resolved_meta_keys[resolved] = std_col
            if not signal_orig_cols:
                expected = [
                    str(col) for col, meta in channel_metadata.items()
                    if meta['parent_signal'].strip().lower() == signal_name
                ]
                print(
                    f"⚠️ No columns found for signal '{signal_name}'. "
                    f"Expected one of {expected}. Available columns: {list(df.columns)}. Skipping."
                )
                continue
            print("16d")
            signal_df = df[['datetime'] + signal_orig_cols].copy()

            # Strip importer uniqueness suffixes (e.g., gy__corrected_gyr -> gy)
            # at the per-signal level so users interact with canonical channel IDs.
            rename_map = {}
            signal_channel_metadata = {}
            used_base_names = set()
            for orig_col in signal_orig_cols:
                meta_key = resolved_meta_keys.get(orig_col, orig_col)
                if meta_key not in channel_metadata:
                    # Fallback: try normalized lookup against channel_metadata keys.
                    norm_target = _norm_col(meta_key)
                    alt_key = next(
                        (k for k in channel_metadata.keys() if _norm_col(k) == norm_target),
                        None,
                    )
                    meta_key = alt_key if alt_key is not None else meta_key
                if meta_key not in channel_metadata:
                    print(
                        f"⚠️ Missing channel metadata for '{orig_col}' "
                        f"(resolved key '{meta_key}') in signal '{signal_name}'. Skipping this channel."
                    )
                    continue
                # Use the standardized metadata key as the canonical channel id.
                base_name = str(meta_key).split("__", 1)[0]
                target_name = base_name
                if target_name in used_base_names:
                    # Keep this deterministic and safe if a true same-signal collision occurs.
                    i = 2
                    while f"{base_name}__dup{i}" in used_base_names:
                        i += 1
                    target_name = f"{base_name}__dup{i}"
                used_base_names.add(target_name)
                rename_map[orig_col] = target_name
                signal_channel_metadata[target_name] = channel_metadata[meta_key]

            signal_df.rename(columns=rename_map, inplace=True)
            signal_cols = [rename_map[c] for c in signal_orig_cols if c in rename_map]
            if any(str(c).find("__") >= 0 for c in signal_orig_cols):
                print(
                    f"[group_data_by_signals] '{signal_name}' channel normalization: "
                    f"{signal_orig_cols} -> {signal_cols}"
                )

            if not signal_cols:
                print(f"⚠️ No valid channels remained for signal '{signal_name}' after metadata resolution. Skipping.")
                continue

            print("16e")
            # Check if all signal columns are numeric
            non_numeric_cols = signal_df[signal_cols].select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"❌ Skipping signal '{signal_name}': non-numeric data found in columns: {non_numeric_cols}")
                continue

            print("16f")
            # Determine the data type of the signal columns
            data_type = signal_df[signal_cols].dtypes.iloc[0]
            data_type_str = str(data_type)

            print("16g")
            # Standardized metadata collection
            start_time = signal_df['datetime'].iloc[0]
            end_time = signal_df['datetime'].iloc[-1]
            max_value = signal_df[signal_cols].max().max()
            min_value = signal_df[signal_cols].min().min()
            mean_value = signal_df[signal_cols].mean().mean()

            print("16h")
            # Get the original unit from the column metadata
            original_units = {signal_channel_metadata[col]['unit'] for col in signal_cols}
            if len(original_units) > 1:
                warnings.warn(f"Conflicting units found for signal '{signal_name}': {original_units}. Using the first one.")
            original_unit = original_units.pop() if original_units else "unknown"

            print("16i")
            # Calculate current frequency - beware this does not fix gaps, just uses the first few values to calculate freq.
            original_frequency = calculate_sampling_frequency(signal_df['datetime'].head()) # round(1 / signal_df['datetime'].diff().dt.total_seconds().mean())
            print(f"Original frequency for {signal_name}: {original_frequency} Hz")

            print("16j")
            expected_frequency = self.expected_frequencies.get(signal_name)
            if not expected_frequency and self.logger_manufacturer == 'LL':
                expected_frequency = int(self.data_reader.logger_info[logger_id]['fs'])

            print("16k")
            max_desired_frequency = None
            if self.logger_manufacturer in ['Evolocus', 'Manitty', 'UFI']:
                max_freq_lookup = {'eeg': 100, 'eog': 100, 'ecg': 250, 'emg': 250}
                max_desired_frequency = max_freq_lookup.get(signal_name, None)

            print("16l")
            downsample_target = max_desired_frequency or expected_frequency
            if not expected_frequency and not max_desired_frequency:
                print(f"⚠️ No frequency target found for {signal_name}. Using original frequency {original_frequency} Hz.")

            print("16m")
            if downsample_target and downsample_target < original_frequency:
                decimation_factor = max(1, int(round(original_frequency / downsample_target)))
                print(f"Downsampling {signal_name} by {decimation_factor}x from {original_frequency:.2f}Hz to {downsample_target:.2f}Hz.")
                signal_df = signal_df.iloc[::decimation_factor]
                # Calculate new frequency after downsampling - beware this does not fix gaps, just uses the first few values to calculate freq.
                new_frequency = calculate_sampling_frequency(signal_df['datetime'].head()) # round(1 / signal_df['datetime'].diff().dt.total_seconds().mean())
                print(f"New frequency after downsampling: {new_frequency} Hz")
            else:
                new_frequency = original_frequency
                print(f"No downsampling required for {signal_name}. Current: {original_frequency:.2f}Hz, Target: {downsample_target}Hz")

            print("16n")
            details = 'Initial, raw signal-specific data and metadata loaded.'
            if new_frequency != original_frequency:
                details += f' Original frequency: {original_frequency} Hz; downsampled to {new_frequency} Hz.'
            else:
                details += f' Original frequency: {original_frequency} Hz; no downsampling applied.'

            if signal_name in {"accelerometer", "accelerometer2"}:
                try:
                    preview_cols = ["datetime"] + signal_cols
                    preview_cols = [c for c in preview_cols if c in signal_df.columns]
                    print(f"🔎 DEBUG {signal_name} dataframe columns: {list(signal_df.columns)}")
                    print(
                        f"🔎 DEBUG {signal_name} dataframe dtypes: "
                        f"{ {col: str(dtype) for col, dtype in signal_df.dtypes.items()} }"
                    )
                    print(
                        f"🔎 DEBUG {signal_name} dataframe header "
                        f"(cols={preview_cols}, rows={len(signal_df)}):"
                    )
                    print(signal_df[preview_cols].head(3).to_string(index=False))
                except Exception as e:
                    print(f"⚠️ Failed to print {signal_name} debug header: {type(e).__name__}: {e}")

            print
            self.data_reader.signal_data[signal_name] = signal_df
            time_zone_raw = self.data_reader.deployment_info.get('Time Zone')
            tz_name = str(time_zone_raw).strip() if time_zone_raw is not None else "UTC"
            if not tz_name:
                tz_name = "UTC"
            try:
                tz = pytz.timezone(tz_name)
            except Exception:
                print(f"⚠️ Invalid timezone '{tz_name}'. Falling back to UTC.")
                tz = pytz.UTC

            self.data_reader.signal_info[signal_name] = {
                'channels': signal_cols,
                'metadata': {col: signal_channel_metadata[col] for col in signal_cols},
                'signal_start_datetime': start_time,
                'signal_end_datetime': end_time,
                'max_value': float(max_value),
                'min_value': float(min_value),
                'mean_value': float(mean_value),
                'data_type': data_type_str,
                'original_units': original_unit,
                'units': original_unit,
                'original_sampling_frequency': original_frequency,
                'sampling_frequency': new_frequency,
                'logger_id': self.logger_id,
                'logger_manufacturer': self.logger_manufacturer,
                'processing_step': 'Raw data uploaded',
                'last_updated': pd.Timestamp(datetime.now().astimezone(tz)),
                'details': details,
            }
            print(f"[group_data_by_signals] stored signal_info['{signal_name}']['channels'] = {signal_cols}")

        for signal_name, df in self.data_reader.signal_data.items():
            print(f"Sensor '{signal_name}' data processed and stored with shape {df.shape}.")

        return signal_groups, signal_info


    def process_files(self, files):
        """Process files in the subclass. This should be overridden."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def concatenate_and_save_csvs(self, csv_files):
        """Base method for concatenating and saving CSVs."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_expected_frequencies(self, parsed_signals, enforce_frequency=True):
        """
        Matches parsed signals with the channel mapping and sets expected frequencies.
        
        Args:
            parsed_signals (dict): Sensor name -> expected interval (Hz)
            enforce_frequency (bool): Whether to enforce setting expected frequencies. Default: True.
        """
        if not parsed_signals:
            print("⚠ No signals parsed from txt file, skipping frequency matching.")
            return

        if not self.montage:
            print(f"⚠ Warning: No valid channel mapping found for Manufacturer '{self.logger_manufacturer}' with Montage ID '{self.montage_id}'.")
            return

        print(f"🔍 Matching parsed signals with channel mapping. Enforce frequency: {enforce_frequency}")

        for signal_name, frequency in parsed_signals.items():
            found_match = False
            for clean_name, mapping in self.montage.items():
                manufacturer_signal_name = mapping['manufacturer_signal_name'].strip().lower()

                if manufacturer_signal_name == signal_name:
                    parent_signal = mapping['parent_signal'].strip().lower()

                    if enforce_frequency:
                        self.expected_frequencies[parent_signal] = frequency  
                        print(f"✅ Matched '{signal_name}' -> '{parent_signal}' with expected frequency: {frequency} Hz.")
                    else:
                        print(f"🔍 Matched '{signal_name}' -> '{parent_signal}', but skipping frequency enforcement.")

                    found_match = True
                    break  # Stop once a match is found

            if not found_match:
                print(f"⚠ Sensor name '{signal_name}' not found in channel mapping. Ignoring this signal.")
