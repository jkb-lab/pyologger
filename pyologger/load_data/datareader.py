import os
import pickle
import re
import json
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from pyologger.utils.time_manager import process_datetime
from pyologger.utils.event_manager import append_logger_status_events
from pyologger.analyze_data.find_segments import find_segments
from pyologger.process_data.sampling import *
from pyologger.load_data.metadata import *
from pyologger.io_operations import *
from pyologger.io_operations.base_exporter import BaseExporter
from pyologger.io_operations.cats_importer import CATSImporter
from pyologger.io_operations.ufi_importer import UFIImporter
from pyologger.io_operations.wc_importer import WCImporter  # if you still have a WC-specific importer; if not, remove this import
from pyologger.io_operations.ll_importer import LLImporter
from pyologger.io_operations.evolocus_importer import EvolocusImporter
from pyologger.io_operations.manitty_importer import ManittyImporter
from pyologger.io_operations.csv_importer import CSVImporter
from pyologger.io_operations.dt_importer import DTImporter
from pyologger.io_operations.pd_importer import PDImporter
from pyologger.io_operations.star_oddi_importer import StarOddiImporter
from pyologger.io_operations.vectronics_importer import VectronicsImporter


class DataReader:
    """Reads and processes all raw files for a deployment, 1 pass per logger."""
    _label_db_cache = None

    def __init__(self, dataset_folder: str, deployment_id: str, data_subfolder: str = None, montage_path: str = None):
        self.deployment_id = deployment_id
        
        # Look for deployment folder that starts with deployment_id (allows suffixes)
        self.deployment_folder = self._find_deployment_folder(dataset_folder, deployment_id)
        self.data_folder = os.path.join(self.deployment_folder, data_subfolder) if data_subfolder else self.deployment_folder
        self.montage_path = montage_path

        # Selected deployment metadata (lat/lon/time zone/date)
        self.deployment_info = {}
        self.procedure_info = {}

        # Output folder
        self.output_folder = os.path.join(self.deployment_folder, 'outputs')

        # Dataset / animal tracking info for downstream metadata
        self.animal_info = {'Animal_ID': self.deployment_id.split('_', 1)[1] if '_' in self.deployment_id else None}
        self.dataset_info = {'Dataset_ID': os.path.basename(os.path.normpath(dataset_folder))}

        # Per-logger metadata like Manufacturer, Montage ID, fs, etc.
        self.logger_info = {}

        # Signal data after grouping (populated by importer.group_data_by_signals())
        self.signal_data = {}
        self.signal_info = {}

        # Event notes (00_Notes.xlsx) and extracted behaviors/events
        self.event_data = self._create_empty_event_dataframe()
        self.event_info = {}

        # Exporter helper
        self.exporter = BaseExporter(self)

        self._has_run = False

        print(f"DataReader initialized with deployment folder: {self.deployment_folder}")
        print(f"Using data folder: {self.data_folder}")
    
    def _find_deployment_folder(self, dataset_folder: str, deployment_id: str) -> str:
        """
        Find deployment folder that matches deployment_id.
        Prefers folders with suffixes (e.g., 2015-07-01_pale-007_Chunga-18832-M) over exact matches.
        """
        exact_path = os.path.join(dataset_folder, deployment_id)
        found_folders = []
        
        # Look for all folders starting with deployment_id
        try:
            for entry in os.listdir(dataset_folder):
                full_path = os.path.join(dataset_folder, entry)
                if os.path.isdir(full_path) and entry.startswith(deployment_id):
                    # Make sure it's not just a prefix match (require _ or end of string after ID)
                    if entry == deployment_id:
                        found_folders.append((entry, full_path, False))  # exact match
                    elif len(entry) > len(deployment_id) and entry[len(deployment_id)] == '_':
                        found_folders.append((entry, full_path, True))  # has suffix
        except (FileNotFoundError, PermissionError):
            pass
        
        # Prefer folders with suffixes
        folders_with_suffix = [f for f in found_folders if f[2]]
        if folders_with_suffix:
            entry, full_path, _ = folders_with_suffix[0]
            print(f"   📁 Found deployment folder with suffix: {entry}")
            return full_path
        
        # Fall back to exact match if no suffix folders found
        exact_matches = [f for f in found_folders if not f[2]]
        if exact_matches:
            return exact_matches[0][1]
        
        # If we get here, deployment folder was not found
        raise ValueError(f"❌ Deployment folder not found for ID '{deployment_id}' in dataset '{dataset_folder}'. "
                       f"Tried exact match and folders with suffix pattern '{deployment_id}_*'.")

    def get_logger_time_zone(self, logger_id, default="UTC"):
        deployment_tz = self.deployment_info.get("Time Zone") or self.deployment_info.get("TimeZone")
        deployment_tz = str(deployment_tz).strip() if deployment_tz else ""

        logger_tz = (self.logger_info.get(logger_id, {}) or {}).get("Time Zone")
        logger_tz = str(logger_tz).strip() if logger_tz else ""

        if logger_tz and deployment_tz and logger_tz.lower() != deployment_tz.lower():
            return logger_tz

        if deployment_tz:
            return deployment_tz

        return default

    def _has_vectronics_logger(self) -> bool:
        for logger_meta in (self.logger_info or {}).values():
            manufacturer = str((logger_meta or {}).get("Manufacturer", "")).strip()
            if manufacturer in {"VT", "Vectronics"}:
                return True
        return False

    def _crop_vectronics_data_to_accelerometer_range(self):
        """
        For Vectronics deployments, crop datetime-based data to the accelerometer window.
        Raise if no location samples overlap that accelerometer range.
        """
        if not self._has_vectronics_logger():
            return

        accel_df = self.signal_data.get("accelerometer")
        if not isinstance(accel_df, pd.DataFrame) or accel_df.empty or "datetime" not in accel_df.columns:
            raise ValueError(
                "Vectronics deployment requires signal_data['accelerometer'] with datetime values for range cropping."
            )

        accel_dt = pd.to_datetime(accel_df["datetime"], errors="coerce").dropna()
        if accel_dt.empty:
            raise ValueError(
                "Vectronics deployment has no valid accelerometer datetime values for range cropping."
            )

        crop_start = accel_dt.min()
        crop_end = accel_dt.max()
        print(
            "[Vectronics] Cropping DataReader contents to accelerometer datetime range: "
            f"{crop_start} -> {crop_end}"
        )

        location_df = self.signal_data.get("location")
        if not isinstance(location_df, pd.DataFrame) or location_df.empty or "datetime" not in location_df.columns:
            raise ValueError(
                "Vectronics deployment requires location data with datetime values aligned to the accelerometer range."
            )

        location_dt = pd.to_datetime(location_df["datetime"], errors="coerce")
        location_mask = location_dt.between(crop_start, crop_end, inclusive="both")
        aligned_location_df = location_df.loc[location_mask].copy()
        if aligned_location_df.empty:
            raise ValueError(
                "No location data aligns with the Vectronics accelerometer datetime range "
                f"{crop_start} to {crop_end}."
            )

        for signal_name, signal_df in list(self.signal_data.items()):
            if not isinstance(signal_df, pd.DataFrame) or signal_df.empty or "datetime" not in signal_df.columns:
                continue

            signal_dt = pd.to_datetime(signal_df["datetime"], errors="coerce")
            signal_mask = signal_dt.between(crop_start, crop_end, inclusive="both")
            cropped_df = signal_df.loc[signal_mask].copy()
            self.signal_data[signal_name] = cropped_df

            signal_meta = self.signal_info.get(signal_name)
            if isinstance(signal_meta, dict):
                if not cropped_df.empty:
                    cropped_dt = pd.to_datetime(cropped_df["datetime"], errors="coerce").dropna()
                    if not cropped_dt.empty:
                        signal_meta["signal_start_datetime"] = cropped_dt.iloc[0]
                        signal_meta["signal_end_datetime"] = cropped_dt.iloc[-1]
                signal_meta["details"] = (
                    f"{signal_meta.get('details', '').strip()} "
                    f"Cropped to Vectronics accelerometer datetime range."
                ).strip()
                self.signal_info[signal_name] = signal_meta

            print(
                f"[Vectronics] Cropped signal '{signal_name}' from {len(signal_df)} to {len(cropped_df)} rows."
            )

        if isinstance(self.event_data, pd.DataFrame) and not self.event_data.empty and "datetime" in self.event_data.columns:
            event_dt = pd.to_datetime(self.event_data["datetime"], errors="coerce")
            event_mask = event_dt.between(crop_start, crop_end, inclusive="both")
            original_event_count = len(self.event_data)
            self.event_data = self.event_data.loc[event_mask].copy()
            print(
                f"[Vectronics] Cropped event_data from {original_event_count} to {len(self.event_data)} rows."
            )

    def _standardize_little_leonardo_stroke_rate(self):
        """Convert Little Leonardo stroke rate from count_per_5s to spm during ingest."""
        if not any(
            str((info or {}).get("Manufacturer") or "").strip().lower() == "little leonardo"
            for info in (self.logger_info or {}).values()
        ):
            return False
        stroke_df = self.signal_data.get("stroke_rate")
        stroke_info = self.signal_info.get("stroke_rate")
        if not isinstance(stroke_df, pd.DataFrame) or stroke_df.empty or not isinstance(stroke_info, dict):
            return False

        value_col = next((col for col in ["stroke_rate", "Stroke_Rate"] if col in stroke_df.columns), None)
        if value_col is None:
            return False

        meta = dict((stroke_info.get("metadata") or {}).get(value_col) or {})
        unit_candidates = [
            str(meta.get("unit") or "").strip().lower(),
            str(meta.get("original_unit") or "").strip().lower(),
            str(stroke_info.get("unit") or "").strip().lower(),
            str(stroke_info.get("standardized_unit") or "").strip().lower(),
        ]
        if "count_per_5s" not in unit_candidates:
            return False

        stroke_df = stroke_df.copy()
        stroke_df[value_col] = pd.to_numeric(stroke_df[value_col], errors="coerce") * 12.0
        meta["unit"] = "spm"
        if meta.get("original_unit") is None:
            meta["original_unit"] = "count_per_5s"
        metadata = dict(stroke_info.get("metadata") or {})
        metadata[value_col] = meta
        stroke_info = dict(stroke_info)
        stroke_info["metadata"] = metadata
        stroke_info["standardized_unit"] = "spm"
        transformation_log = list(stroke_info.get("transformation_log") or [])
        if "converted_stroke_rate_count_per_5s_to_spm" not in transformation_log:
            transformation_log.append("converted_stroke_rate_count_per_5s_to_spm")
        stroke_info["transformation_log"] = transformation_log
        self.signal_data["stroke_rate"] = stroke_df
        self.signal_info["stroke_rate"] = stroke_info
        print("Converted Little Leonardo stroke_rate from count_per_5s to spm during ingest.")
        return True

    def read_files(self, deployment_info, loggers_used, save_parq=True, save_netcdf=False):
        
        """
        Main ingestion routine:
          1. Cache deployment metadata
          2. Cache logger_info (Manufacturer, Montage ID, etc.)
          3. Import notes (events)
          4. Match files to each logger simply by substring match on Logger ID
          5. For each logger:
             - Choose the correct importer class
             - Run importer ONCE (no double-processing)
             - Save outputs
          6. Save DataReader snapshot (.pkl)
          7. [optional] Save NetCDF
        """

        print(f"🔄 Reading files from: {self.data_folder}")
        print(f"📁 Created data folder: {self.output_folder}")

        # 1. Store deployment metadata we care about
        self.deployment_info["Deployment Date"]      = deployment_info.get("Deployment Date")
        self.deployment_info["Deployment Latitude"]  = deployment_info.get("Deployment Latitude")
        self.deployment_info["Deployment Longitude"] = deployment_info.get("Deployment Longitude")
        tz_raw = deployment_info.get("Time Zone")
        self.deployment_info["Time Zone"] = str(tz_raw).strip() if tz_raw is not None else None
        self.procedure_info = dict(deployment_info.get("Procedure Info") or {})

        # 2. Record logger_info (Manufacturer, Montage ID, etc.)
        self.logger_info = {
            logger["Logger ID"]: {
                "ID": logger["Logger ID"],
                "Manufacturer": logger["Manufacturer"],
                "Montage ID": logger.get("Montage ID", "Unknown"),
                "Time Zone": logger.get("Time Zone")
            }
            for logger in loggers_used
        }

        # 3. Import notes, if present
        imported_events = self.import_notes()
        if imported_events is not None:
            self.event_data = imported_events

        # 4. Build mapping logger_id -> files
        #    Rule: file goes to logger if the literal logger_id string is in filename
        raw_files = self._collect_raw_files()
        logger_files = {logger["Logger ID"]: [] for logger in loggers_used}

        subfolder_matches = self._collect_logger_subfolder_files(loggers_used)
        for lid, matched_files in subfolder_matches.items():
            logger_files[lid].extend(matched_files)

        for fname in raw_files:
            for logger in loggers_used:
                lid = logger["Logger ID"]
                if lid in fname:
                    logger_files[lid].append(fname)

        for lid, files in logger_files.items():
            if not files:
                continue
            seen = set()
            deduped = []
            for path in files:
                if path in seen:
                    continue
                seen.add(path)
                deduped.append(path)
            logger_files[lid] = deduped

        # 5. Report file association
        print("📟 Loggers with files:")
        for lid, files in logger_files.items():
            if files:
                print(f"   {lid}: {files}")
            else:
                print(f"   {lid}: (no matching files)")

        # 6. Importer resolution logic
        def choose_importer(manufacturer: str, files_for_logger: list):
            """
            Priority rules:
              - CATS always uses CATSImporter.
              - For UFI:
                    if there's any .ube/.ubf -> UFIImporter
                    elif only .csv -> CSVImporter
              - For other manufacturers:
                    if any .csv -> CSVImporter
                    else fallback by known manufacturer class
            """

            # (a) CATS is special
            if manufacturer == "CATS":
                return CATSImporter

            # Vectronics / VT loggers
            if manufacturer in ("VT", "Vectronics"):
                return VectronicsImporter

            # (b) UFI is also special (binary ECG tags)
            if manufacturer == "UFI":
                has_ube = any(f.lower().endswith((".ube", ".ubf")) for f in files_for_logger)
                has_csv = any(f.lower().endswith(".csv") for f in files_for_logger)
                if has_ube:
                    return UFIImporter
                if has_csv:
                    return CSVImporter
                # default if some weird future format shows up
                return UFIImporter

            # (c) Other known manufacturers
            # Wildlife Computers, Star Oddi, Little Leonardo, etc. are basically CSV/Parquet loggers
            if manufacturer == "Star Oddi":
                return StarOddiImporter

            if manufacturer == "PD":
                return PDImporter

            has_dt_netcdf = any(
                f.lower().endswith(".nc")
                and any(token in os.path.basename(f).lower() for token in ("sens", "trk", "prof"))
                for f in files_for_logger
            )
            if manufacturer in {"DT", "DTAG"} or has_dt_netcdf:
                return DTImporter

            if any(f.lower().endswith((".csv", ".parquet", ".parq", ".pq")) for f in files_for_logger):
                return CSVImporter

            if manufacturer == "Little Leonardo":
                return CSVImporter
            if manufacturer == "Evolocus":
                return EvolocusImporter
            if manufacturer == "Manitty":
                return ManittyImporter

            # e.g. "Wildlife Computers", "Star Oddi"
            if manufacturer == "Wildlife Computers":
                return CSVImporter

            # final fallback
            return CSVImporter

        # 7. Process each logger once
        processed_logger_ids = set()

        for logger in loggers_used:
            lid = logger["Logger ID"]
            manufacturer = logger["Manufacturer"]

            if lid in processed_logger_ids:
                # already handled this logger
                continue
            processed_logger_ids.add(lid)

            files_for_this_logger = logger_files.get(lid, [])
            if not files_for_this_logger:
                print(f"⚠ No files to process for {lid} ({manufacturer}). Skipping.")
                continue

            importer_class = choose_importer(manufacturer, files_for_this_logger)
            importer_instance = importer_class(self, lid)

            print(f"🔄 Processing {lid} ({manufacturer}) with importer {importer_class.__name__}")
            print(f"   Files: {files_for_this_logger}")

            # Early montage validation to catch missing parent_signal before heavy processing
            try:
                expected_channels = importer_instance.montage
            except AttributeError:
                print('there was an attribute error')
                expected_channels = None

            if expected_channels:
                print("There were expected channels")
                unmapped = [
                    ch for ch, mapping in expected_channels.items()
                    if not mapping or not mapping.get("parent_signal")
                ]
                print(f"Unmapped channels: {unmapped}")
                if unmapped:
                    raise ValueError(
                        f"Montage '{self.montage_path}' for logger '{lid}' is missing parent_signal for: {unmapped[:10]}"
                    )

            try:
                result = importer_instance.process_files(files_for_this_logger)
            except Exception as e:
                print(f"❌ ERROR while processing files from {lid} ({manufacturer}): {e}")
                print(type(e))
                print(e.__dict__)
                continue

            if not result:
                print(f"⚠ Importer for {lid} returned no data.")
                continue

            # Two shapes are supported:
            #  A) multi-frequency dicts (Evolocus / Manitty)
            #  B) single dataframe (CSVImporter, UFIImporter, etc.)
            if (
                manufacturer in ["Evolocus", "Manitty"]
                and isinstance(result, tuple)
                and len(result) == 5
                and isinstance(result[0], dict)
            ):
                (final_dfs_dict,
                 channel_metadata_dict,
                 datetime_metadata_dict,
                 signal_groups_dict,
                 signal_info_dict) = result

                for freq_key, metadata in channel_metadata_dict.items():
                    self._validate_channel_metadata(metadata, lid, context=f"{freq_key}Hz")

                for freq, df in final_dfs_dict.items():
                    self._append_label_state_events_from_dataframe(
                        df,
                        channel_metadata_dict.get(freq, {}),
                        logger_id=lid,
                    )
                    if df is not None and "datetime" in df.columns:
                        filename = f"{lid}_{freq}Hz.csv"
                        self.exporter.save_data(df, lid, filename, save_parq)

                print(f"✅ Processed and saved multi-frequency data for {lid} ({manufacturer}).")

                start_dt, end_dt = self._compute_logger_time_bounds(final_dfs_dict.values())
                self.event_data = append_logger_status_events(
                    existing_events=self.event_data,
                    logger_id=lid,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    timezone=self.deployment_info.get("Time Zone")
                )

            else:
                (final_df,
                 channel_metadata,
                 datetime_metadata,
                 signal_groups,
                 signal_info) = result

                self._validate_channel_metadata(channel_metadata, lid)
                self._append_label_state_events_from_dataframe(final_df, channel_metadata, logger_id=lid)

                if final_df is not None and "datetime" in final_df.columns:
                    outname = f"{lid}.csv"
                    self.exporter.save_data(final_df, lid, outname, save_parq)
                    print(f"✅ Processed and saved files for logger {lid} ({manufacturer}).")
                else:
                    print(f"⚠ Issue: {lid} ({manufacturer}) produced no final_df with 'datetime'.")

                start_dt, end_dt = self._compute_logger_time_bounds([final_df])
                self.event_data = append_logger_status_events(
                    existing_events=self.event_data,
                    logger_id=lid,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    timezone=self.deployment_info.get("Time Zone")
                )
        
        self._ensure_event_data_columns()
        self._crop_vectronics_data_to_accelerometer_range()
        self._standardize_little_leonardo_stroke_rate()

        # 8. Save the DataReader snapshot to a pickle file (once)
        self.save_datareader_object()

        # 9. Optionally write NetCDF for the whole deployment (once)
        if save_netcdf:
            netcdf_filename = os.path.join(
                self.deployment_folder,
                'outputs',
                f'{self.deployment_id}_00_processed.nc'
            )
            self.exporter.save_to_netcdf(self, netcdf_filename)
            print(f"📊 Saved deployment data to NetCDF: {netcdf_filename}")

    def _collect_raw_files(self) -> list[str]:
        """
        Enumerate files in the data folder, including nested files inside directories
        that start with two letters (e.g., VT_accelerometer) but ignoring XX_* folders.
        """
        collected: list[str] = []
        try:
            two_letter_dir = re.compile(r"^[A-Za-z]{2}_")
            with os.scandir(self.data_folder) as entries:
                for entry in entries:
                    if entry.is_file():
                        collected.append(entry.name)
                    elif entry.is_dir():
                        name_lower = entry.name.lower()
                        if two_letter_dir.match(entry.name) and not name_lower.startswith("xx_"):
                            for root, _, files in os.walk(entry.path):
                                for filename in files:
                                    abs_path = os.path.join(root, filename)
                                    rel_path = os.path.relpath(abs_path, self.data_folder)
                                    collected.append(rel_path)
            print(f"Found {len(collected)} raw files in the data folder.")
        except FileNotFoundError:
            print(f"⚠️ Data folder not found: {self.data_folder}")
        return sorted(collected)

    def _collect_logger_subfolder_files(self, loggers_used) -> dict[str, list[str]]:
        """
        Find files located inside subfolders that follow the pattern:
            <LOGGER_ID>_<signal[-signal2[-...]]>

        All files inside those subfolders are attributed to the matching logger.
        """
        matches = {logger["Logger ID"]: [] for logger in loggers_used}
        lid_order = [logger["Logger ID"] for logger in loggers_used]
        try:
            with os.scandir(self.data_folder) as entries:
                for entry in entries:
                    if not entry.is_dir():
                        continue
                    entry_name_lower = entry.name.lower()
                    if entry_name_lower.startswith("xx_"):
                        continue

                    for lid in lid_order:
                        lid_lower = lid.lower()
                        prefix = f"{lid_lower}_"
                        if entry_name_lower.startswith(prefix):
                            matches[lid].extend(self._walk_relative_files(entry.path))
                            break
        except FileNotFoundError:
            print(f"⚠️ Data folder not found: {self.data_folder}")
        return matches

    def _walk_relative_files(self, start_path: str) -> list[str]:
        collected: list[str] = []
        for root, _, files in os.walk(start_path):
            for filename in files:
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, self.data_folder)
                collected.append(rel_path)
        return collected


    def _validate_channel_metadata(self, channel_metadata, logger_id, context: str | None = None):
        """Ensure each channel entry includes a parent_signal; raise helpful error otherwise."""
        if not isinstance(channel_metadata, dict):
            return

        missing = []
        for channel_name, metadata in channel_metadata.items():
            parent_signal = (metadata or {}).get("parent_signal")
            if not parent_signal:
                missing.append({
                    "channel": channel_name,
                    "metadata": metadata
                })

        if missing:
            ctx = f" ({context})" if context else ""
            preview = "\n".join(
                f" - {item['channel']}: {item['metadata']}"
                for item in missing[:10]
            )
            raise ValueError(
                f"Channel mapping for logger '{logger_id}'{ctx} is missing 'parent_signal' "
                f"for {len(missing)} channel(s). Please update the montage mapping.\n"
                f"{preview}"
            )

    def _ensure_event_data_columns(self):
        """Ensure event_data exists, has required columns, and tz-aware datetimes."""
        required_columns = ['datetime', 'key', 'value', 'note', 'type', 'duration']
        timezone = self.deployment_info.get("Time Zone") or "UTC"

        if not isinstance(self.event_data, pd.DataFrame):
            self.event_data = self._create_empty_event_dataframe(timezone)

        # Add any missing columns with sensible defaults
        for column in required_columns:
            if column not in self.event_data.columns:
                default_dtype = 'float64' if column == 'duration' else 'object'
                self.event_data[column] = pd.Series([], dtype=default_dtype)

        # Normalize datetime column and enforce timezone awareness even if empty
        datetime_series = pd.to_datetime(self.event_data.get('datetime', pd.Series([], dtype='datetime64[ns]')), errors='coerce')
        try:
            tz = pytz.timezone(timezone)
        except Exception:
            print(f"⚠️ Invalid timezone '{timezone}'. Using UTC instead.")
            tz = pytz.UTC

        if datetime_series.dt.tz is None:
            datetime_series = datetime_series.dt.tz_localize(tz)
        else:
            datetime_series = datetime_series.dt.tz_convert(tz)

        self.event_data['datetime'] = datetime_series

        computed_types = self.event_data.apply(
            lambda row: (
                'dive' if 'dive' in str(row.get('key', '')).lower()
                else 'note' if pd.notna(row.get('note')) and str(row.get('note')).strip()
                else 'event'
            ),
            axis=1
        )

        if 'type' in self.event_data.columns:
            mask = self.event_data['type'].isna() | (self.event_data['type'].astype(str).str.strip() == '')
            self.event_data.loc[mask, 'type'] = computed_types[mask]
        else:
            self.event_data['type'] = computed_types

    @staticmethod
    def _normalize_label_name(value) -> str:
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    @classmethod
    def _load_label_db(cls):
        if cls._label_db_cache is not None:
            return cls._label_db_cache
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        label_db_path = os.path.join(repo_root, "label_db.json")
        try:
            with open(label_db_path, "r", encoding="utf-8") as f:
                cls._label_db_cache = json.load(f)
        except FileNotFoundError:
            cls._label_db_cache = {}
        except Exception as exc:
            print(f"⚠️ Failed to load label_db.json: {type(exc).__name__}: {exc}")
            cls._label_db_cache = {}
        return cls._label_db_cache

    def _resolve_label_db_entry(self, label_group: str, decoded_text: str | None, code_value=None):
        label_db = self._load_label_db()
        alias_groups = (label_db.get("aliases") or {})
        candidate_groups = [label_group]
        if label_group.endswith("-project"):
            candidate_groups.append(label_group[: -len("-project")])
        candidate_groups.append(label_group.replace("_", "-"))
        if label_group.endswith("_project"):
            candidate_groups.append(label_group[: -len("_project")].replace("_", "-"))

        alias_candidates = []
        if decoded_text:
            alias_candidates.append(self._normalize_label_name(decoded_text))
        if code_value is not None and pd.notna(code_value):
            numeric_code = pd.to_numeric(pd.Series([code_value]), errors="coerce").iloc[0]
            if pd.notna(numeric_code):
                if float(numeric_code).is_integer():
                    alias_candidates.append(str(int(numeric_code)))
                alias_candidates.append(str(numeric_code))

        resolved_key = None
        for group_name in candidate_groups:
            aliases = alias_groups.get(group_name, {})
            for alias_key in alias_candidates:
                resolved_key = aliases.get(alias_key)
                if resolved_key:
                    break
            if resolved_key:
                break
        if not resolved_key:
            return None
        return (label_db.get("by_key") or {}).get(resolved_key)

    @staticmethod
    def _label_sample_seconds(df: pd.DataFrame) -> float:
        if not isinstance(df, pd.DataFrame) or "datetime" not in df.columns or len(df) < 2:
            return 1.0
        dt = pd.to_datetime(df["datetime"], errors="coerce").sort_values()
        diffs = dt.diff().dt.total_seconds().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            return 1.0
        return float(diffs.median())

    def _append_label_state_events_from_dataframe(self, df: pd.DataFrame, channel_metadata: dict | None, logger_id: str):
        """
        Convert numeric label signals (e.g., sleep_state) into state events by
        collapsing consecutive runs of the same code. If matching text-code
        columns exist in the imported dataframe, use them to name the state key.
        """
        if not isinstance(df, pd.DataFrame) or df.empty or "datetime" not in df.columns:
            return 0

        metadata = channel_metadata or {}
        if not metadata:
            return 0

        work_df = df.copy()
        work_df["datetime"] = pd.to_datetime(work_df["datetime"], errors="coerce")
        work_df = work_df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        if work_df.empty:
            return 0

        original_name_lookup = {}
        for standardized_col, meta in metadata.items():
            if standardized_col not in work_df.columns:
                continue
            original_name = self._normalize_label_name((meta or {}).get("original_name", standardized_col))
            if original_name and original_name not in original_name_lookup:
                original_name_lookup[original_name] = standardized_col

        label_columns = []
        for standardized_col, meta in metadata.items():
            if standardized_col not in work_df.columns:
                continue
            parent_signal = str((meta or {}).get("parent_signal", "")).strip().lower()
            if parent_signal != "label":
                continue
            if not pd.api.types.is_numeric_dtype(work_df[standardized_col]):
                continue
            label_columns.append(standardized_col)

        if not label_columns:
            return 0

        generated_rows = []
        generated_prefixes = []
        generated_keys = set()
        dive_type_generated_keys = set()
        sample_seconds = self._label_sample_seconds(work_df)

        for numeric_col in sorted(set(label_columns)):
            series = pd.to_numeric(work_df[numeric_col], errors="coerce")
            valid_codes = sorted(series.dropna().unique().tolist())
            if not valid_codes:
                continue

            prefix = f"{numeric_col}."
            generated_prefixes.append(prefix)

            text_candidates = []
            if numeric_col.endswith("_state"):
                base_name = numeric_col[: -len("_state")]
                text_candidates.extend([
                    f"{base_name}_code",
                    f"{base_name}code",
                ])
            text_col = next(
                (original_name_lookup[candidate] for candidate in text_candidates if candidate in original_name_lookup),
                None,
            )

            code_to_text = {}
            if text_col and text_col in work_df.columns:
                mapping_df = work_df[[numeric_col, text_col]].copy()
                mapping_df[numeric_col] = pd.to_numeric(mapping_df[numeric_col], errors="coerce")
                mapping_df[text_col] = mapping_df[text_col].astype(str).str.strip()
                mapping_df = mapping_df.dropna(subset=[numeric_col])
                mapping_df = mapping_df[mapping_df[text_col].ne("") & mapping_df[text_col].ne("nan")]
                if not mapping_df.empty:
                    for code_value, g in mapping_df.groupby(numeric_col, dropna=True):
                        modes = g[text_col].mode(dropna=True)
                        if not modes.empty:
                            code_to_text[code_value] = str(modes.iloc[0]).strip()

            for code_value in valid_codes:
                segments = find_segments(
                    work_df,
                    column=numeric_col,
                    criteria=lambda x, target=code_value: pd.notna(x) and x == target,
                    min_duration=None,
                    animal_id_col=None,
                )
                if segments.empty:
                    continue

                decoded_text = code_to_text.get(code_value)
                label_group = str(numeric_col).replace("_", "-")
                label_db_entry = self._resolve_label_db_entry(label_group, decoded_text, code_value=code_value)
                if label_db_entry:
                    event_key = str(label_db_entry.get("key"))
                    event_note = str(label_db_entry.get("event_name") or decoded_text or "")
                    short_description = str(label_db_entry.get("short_description") or event_key)
                    long_description = label_db_entry.get("long_description")
                elif decoded_text:
                    slug = self._normalize_label_name(decoded_text)
                    event_key = f"{numeric_col}.{slug}" if slug else f"{numeric_col}.code_{int(code_value)}"
                    event_note = decoded_text
                    short_description = event_key
                    long_description = f"Derived from {numeric_col} label signal: {decoded_text}"
                else:
                    if float(code_value).is_integer():
                        code_label = str(int(code_value))
                    else:
                        code_label = str(code_value).replace(".", "_")
                    event_key = f"{numeric_col}.code_{code_label}"
                    event_note = np.nan
                    short_description = event_key
                    long_description = f"Derived from {numeric_col} label signal"
                generated_keys.add(str(event_key))
                if "dive_type_project" in self._normalize_label_name(numeric_col):
                    dive_type_generated_keys.add(str(event_key))
                for _, seg in segments.iterrows():
                    start_dt = pd.to_datetime(seg["start_datetime"], errors="coerce")
                    if pd.isna(start_dt):
                        continue
                    generated_rows.append(
                        {
                            "datetime": start_dt,
                            "key": event_key,
                            "value": code_value,
                            "note": event_note,
                            "type": "state",
                            "duration": float(seg["duration"]) + sample_seconds,
                            "short_description": short_description,
                            "long_description": long_description,
                            "date": start_dt.strftime("%Y-%m-%d"),
                            "time": start_dt.strftime("%H:%M:%S"),
                        }
                    )

        generated_rows.extend(
            self._build_unscorable_dive_type_rows(
                work_df=work_df,
                metadata=metadata,
                sample_seconds=sample_seconds,
                dive_type_generated_keys=dive_type_generated_keys,
            )
        )
        if generated_rows:
            for row in generated_rows:
                generated_keys.add(str(row.get("key") or "").strip())

        if not generated_rows:
            return 0

        generated_df = pd.DataFrame(generated_rows)
        if not isinstance(self.event_data, pd.DataFrame):
            self.event_data = self._create_empty_event_dataframe(self.deployment_info.get("Time Zone"))

        existing_events = self.event_data.copy()
        if not existing_events.empty and "key" in existing_events.columns:
            for prefix in generated_prefixes:
                existing_events = existing_events[
                    ~existing_events["key"].astype(str).str.startswith(prefix, na=False)
                ]
            if generated_keys:
                existing_events = existing_events[
                    ~existing_events["key"].astype(str).isin(generated_keys)
                ]

        self.event_data = pd.concat([existing_events, generated_df], ignore_index=True)

        if not hasattr(self, "event_manager") or self.event_manager is None:
            self.event_manager = {}
        self.event_manager["derived_label_states"] = {
            "keys": sorted(generated_df["key"].dropna().astype(str).unique().tolist()),
            "description": "State events derived from consecutive numeric label signals",
            "logger_id": logger_id,
        }

        print(
            f"✅ Derived {len(generated_df)} label state events from logger {logger_id}: "
            f"{sorted(generated_df['key'].dropna().astype(str).unique().tolist())}"
        )
        return len(generated_df)

    def _build_unscorable_dive_type_rows(
        self,
        *,
        work_df: pd.DataFrame,
        metadata: dict,
        sample_seconds: float,
        dive_type_generated_keys: set[str],
    ) -> list[dict]:
        """Synthesize UNSCORABLE dive-type states for dives present in dive_num_all but missing dive_num."""
        if not isinstance(work_df, pd.DataFrame) or work_df.empty:
            return []
        if "datetime" not in work_df.columns:
            return []

        dive_type_col = None
        dive_num_all_col = None
        dive_num_col = None

        for standardized_col, meta in metadata.items():
            if standardized_col not in work_df.columns:
                continue
            parent_signal = str((meta or {}).get("parent_signal", "")).strip().lower()
            normalized_col = self._normalize_label_name(standardized_col)
            if parent_signal == "label" and "dive_type_project" in normalized_col:
                dive_type_col = standardized_col
            if parent_signal == "dives":
                if normalized_col.startswith("dive_num_all") and "inacc" in normalized_col and "depth" in normalized_col:
                    dive_num_all_col = standardized_col
                elif normalized_col == "dive_num":
                    dive_num_col = standardized_col

        if dive_type_col is None or dive_num_all_col is None or dive_num_col is None:
            return []

        dive_type_unscorable_key = self._resolve_unscorable_dive_type_key(dive_type_col, dive_type_generated_keys)
        dive_type_unscorable_note = "UNSCORABLE"

        scope = work_df[["datetime", dive_num_all_col, dive_num_col]].copy()
        scope["datetime"] = pd.to_datetime(scope["datetime"], errors="coerce")
        scope = scope.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        if scope.empty:
            return []

        scope["_dive_num_all"] = pd.to_numeric(scope[dive_num_all_col], errors="coerce")
        scope["_dive_num"] = pd.to_numeric(scope[dive_num_col], errors="coerce")
        scope = scope.loc[scope["_dive_num_all"].notna()].copy()
        if scope.empty:
            return []

        scope["_grp"] = scope["_dive_num_all"].ne(scope["_dive_num_all"].shift()).cumsum()
        rows = []
        for _, group in scope.groupby("_grp", sort=False):
            if group["_dive_num"].notna().any():
                continue
            start_dt = pd.to_datetime(group["datetime"].iloc[0], errors="coerce")
            end_dt = pd.to_datetime(group["datetime"].iloc[-1], errors="coerce")
            if pd.isna(start_dt) or pd.isna(end_dt):
                continue
            duration_s = max(
                float(sample_seconds),
                float((end_dt - start_dt).total_seconds()) + float(sample_seconds),
            )
            rows.append(
                {
                    "datetime": start_dt,
                    "key": dive_type_unscorable_key,
                    "value": float(group["_dive_num_all"].iloc[0]),
                    "note": dive_type_unscorable_note,
                    "type": "state",
                    "duration": duration_s,
                    "short_description": "dive-type: UNSCORABLE",
                    "long_description": (
                        f"Derived from {dive_num_all_col} where {dive_num_col} is missing "
                        f"for the full dive interval."
                    ),
                    "date": start_dt.strftime("%Y-%m-%d"),
                    "time": start_dt.strftime("%H:%M:%S"),
                }
            )
        return rows

    @staticmethod
    def _resolve_unscorable_dive_type_key(dive_type_col: str, generated_keys: set[str]) -> str:
        """Choose an UNSCORABLE key style consistent with existing dive-type keys for this dataframe."""
        key_candidates = sorted(str(v).strip() for v in (generated_keys or set()) if str(v or "").strip())
        if not key_candidates:
            return f"{dive_type_col}.unscorable"

        sample = key_candidates[0]
        if sample.startswith("dive-type_"):
            return "dive-type_unscorable"
        if "." in sample:
            return f"{sample.rsplit('.', 1)[0]}.unscorable"
        return f"{dive_type_col}.unscorable"

    def _create_empty_event_dataframe(self, timezone: str | None = None) -> pd.DataFrame:
        """Return a standardized empty event_data DataFrame with tz-aware datetime."""
        tz_name = timezone or "UTC"
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            print(f"⚠️ Invalid timezone '{tz_name}'. Using UTC instead.")
            tz = pytz.UTC

        return pd.DataFrame({
            'datetime': pd.Series([], dtype=pd.DatetimeTZDtype(tz=tz)),
            'key': pd.Series([], dtype='object'),
            'value': pd.Series([], dtype='object'),
            'note': pd.Series([], dtype='object'),
            'type': pd.Series([], dtype='object'),
            'duration': pd.Series([], dtype='float64')
        })

    def _compute_logger_time_bounds(self, dataframes) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Return min/max datetimes across one or more logger dataframes."""
        start_dt = None
        end_dt = None
        for df in dataframes:
            if df is None or not isinstance(df, pd.DataFrame) or 'datetime' not in df.columns:
                continue
            dt_series = pd.to_datetime(df['datetime'], errors='coerce').dropna()
            if dt_series.empty:
                continue
            current_start = dt_series.min()
            current_end = dt_series.max()
            start_dt = current_start if start_dt is None or current_start < start_dt else start_dt
            end_dt = current_end if end_dt is None or current_end > end_dt else end_dt
        return start_dt, end_dt

    def check_outputs_folder(self):
        """
        Checks if the processed data files for the given loggers already exist.
        Returns True if outputs folder has a .pkl, else False.
        """
        if not os.path.exists(self.output_folder):
            print(f"❌ Outputs folder '{self.output_folder}' does not exist. Processing required.")
            return False

        if not any(fname.endswith('.pkl') for fname in os.listdir(self.output_folder)):
            print(f"❌ Outputs folder '{self.output_folder}' does not contain a .pkl file. Processing required.")
            return False

        print(f"✅ Outputs folder '{self.output_folder}' exists and contains a .pkl file.")
        return True

    def import_notes(self):
        """Imports and processes {deployment_id}_00_Notes.xlsx in the data folder, if present."""

        if not self.deployment_info:
            print("❌ Selected deployment metadata not found. Please ensure you have selected a deployment.")
            return None

        notes_filename = f"{self.deployment_id}_00_Notes.xlsx"
        time_zone = self.deployment_info.get("Time Zone") or "UTC"

        notes_filepath = os.path.join(self.data_folder, notes_filename)
        if not os.path.exists(notes_filepath):
            print(f"⚠️ Notes file '{notes_filename}' not found in {self.data_folder}. Skipping import.")
            return None

        try:
            event_df = pd.read_excel(notes_filepath)
            print(f"📂 Successfully loaded notes file: {notes_filepath}")
        except Exception as e:
            print(f"❌ Error reading {notes_filename}: {e}")
            return None

        # timestamp handling
        event_df, _ = process_datetime(event_df, time_zone=time_zone)

        if event_df["datetime"].isna().any():
            print(f"⚠️ WARNING: Some timestamps could not be parsed correctly.")

        # sort chronologically
        event_df = event_df.sort_values(by="datetime").reset_index(drop=True)

        # ensure 'duration' exists
        if "duration" not in event_df.columns:
            event_df["duration"] = 0

        # zero duration for 'point' events
        event_df.loc[event_df["type"] == "point", "duration"] = 0

        print(f"✅ Notes imported and processed from {notes_filename}. Sorted chronologically.")
        return event_df

    def save_datareader_object(self):
        """Saves the entire DataReader object for later quick reload."""
        pickle_filename = os.path.join(self.output_folder, 'data.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"DataReader object successfully saved to {pickle_filename}.")
