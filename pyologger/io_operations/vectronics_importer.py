import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import pytz

from pyologger.io_operations.base_importer import BaseImporter
from pyologger.io_operations.csv_importer import CSVImporter
from pyologger.io_operations.kml_importer import KMLImporter


class VectronicsImporter(BaseImporter):
    """
    Handles VT/Vectronics collars whose raw data live inside sensor-specific folders
    (e.g., 01_raw-data/VT_accelerometer, 01_raw-data/VT_GPS).
    """

    ACC_SUFFIX = "accelerometer"
    GPS_SUFFIX = "gps"

    def process_files(self, files, enforce_frequency: bool = True):
        raw_root = Path(self.data_reader.data_folder)
        logger_id = self.logger_id
        
        # Debug: Check if data folder exists
        if not raw_root.exists():
            print(f"⚠️ Data folder not found: {raw_root}")
            return None

        logger_specific_dirs = self._find_logger_specific_dirs()
        acc_dirs = logger_specific_dirs.get("accelerometer", [])
        gps_dirs = logger_specific_dirs.get("gps", [])
        
        # Debug output
        if acc_dirs:
            print(f"   Found logger-specific accelerometer dirs: {[str(d) for d in acc_dirs]}")
        if gps_dirs:
            print(f"   Found logger-specific GPS dirs: {[str(d) for d in gps_dirs]}")
        
        cleaned_acc_files: List[str] = []

        acc_dataframes = []
        if acc_dirs:
            acc_dataframes.extend(self._prepare_accelerometer_files(acc_dirs))

        acc_root = self._find_sensor_root(raw_root, self.ACC_SUFFIX)
        gps_root = self._find_sensor_root(raw_root, self.GPS_SUFFIX)

        acc_dir = self._resolve_vt_dir(acc_root, logger_id)
        if acc_dir and not acc_dataframes:
            print(f"📥 Found VT accelerometer folder for {logger_id}: {acc_dir}")
            acc_dataframes.extend(self._prepare_accelerometer_files([acc_dir]))

        # Prefer GPS CSV files over KML when both are available.
        gps_location_loaded = False
        gps_csv_files = self._collect_gps_csv_files(gps_dirs, gps_root)
        if gps_csv_files:
            print(f"📍 Found GPS CSV files for logger {logger_id}; preferring CSV over KML.")
            gps_location_loaded = self._process_gps_csv_files(gps_csv_files)

        # Fall back to KML only if no usable GPS CSV location points were loaded.
        if not gps_location_loaded:
            kml_files = self._collect_kml_files(gps_dirs, gps_root)
            if kml_files:
                print(f"📍 Processing KML GPS data for logger {logger_id}")
                kml_importer = KMLImporter(self.data_reader, logger_id)
                kml_importer.process_files(kml_files)

        # If we have accelerometer dataframes, process them directly
        if acc_dataframes:
            print(f"📥 Processing {len(acc_dataframes)} VT accelerometer dataset(s)")
            # Combine all accelerometer dataframes
            combined_df = pd.concat(acc_dataframes, ignore_index=True)

            # Guard against accidental double-standardization:
            # if raw VT IDs exist, prefer those over helper canonical ids.
            helper_drop_cols = []
            if "utc_time" in combined_df.columns and "datetime_utc" in combined_df.columns:
                helper_drop_cols.append("datetime_utc")
            if ("x_axis_scaled" in combined_df.columns or "x_axis" in combined_df.columns) and "ax" in combined_df.columns:
                helper_drop_cols.append("ax")
            if ("y_axis_scaled" in combined_df.columns or "y_axis" in combined_df.columns) and "ay" in combined_df.columns:
                helper_drop_cols.append("ay")
            if ("z_axis_scaled" in combined_df.columns or "z_axis" in combined_df.columns) and "az" in combined_df.columns:
                helper_drop_cols.append("az")
            if helper_drop_cols:
                combined_df = combined_df.drop(columns=helper_drop_cols)
                print(f"   🧹 Dropped helper columns to avoid duplicate mapping: {helper_drop_cols}")
            
            # Get channel metadata from montage
            original_cols = combined_df.columns.tolist()
            csv_importer = CSVImporter(self.data_reader, logger_id)
            new_cols, channel_metadata = csv_importer.rename_channels(original_cols)
            # Apply standardized channel ids (e.g., ax/ay/az) before datetime/grouping.
            # This ensures downstream signals are stored with standardized channel names.
            rename_map = {
                str(src): str(dst)
                for src, dst in (new_cols or {}).items()
                if str(src) in combined_df.columns and str(src) != str(dst)
            }
            if rename_map:
                combined_df = combined_df.rename(columns=rename_map)
                print(f"   🏷️ Applied standardized channel rename for {len(rename_map)} columns.")
                acc_preview = [c for c in ["ax", "ay", "az", "ax2", "ay2", "az2"] if c in combined_df.columns]
                print(f"   🔎 Vectronics standardized accel columns now present: {acc_preview}")

            # Standardize acceleration to the montage's declared unit.
            combined_df = csv_importer.convert_acceleration_to_standard_unit(
                combined_df, channel_metadata
            )
            
            # Process through the standard pipeline
            from pyologger.utils.time_manager import process_datetime
            tz = self.data_reader.deployment_info.get("Time Zone") or "UTC"
            
            combined_df, datetime_metadata = process_datetime(
                combined_df,
                time_zone=tz,
                channel_metadata=channel_metadata
            )
            
            self.data_reader.logger_info[logger_id]['datetime_created_from'] = datetime_metadata.get('datetime_created_from', None)
            self.data_reader.logger_info[logger_id]['fs'] = [datetime_metadata.get('fs', None)]
            
            signal_groups, signal_info = csv_importer.group_data_by_signals(
                combined_df,
                logger_id,
                channel_metadata
            )
            self._ensure_vectronics_acc_signal(combined_df, channel_metadata, "accelerometer", ["ax", "ay", "az"])
            self._ensure_vectronics_acc_signal(combined_df, channel_metadata, "accelerometer2", ["ax2", "ay2", "az2"])
            
            return combined_df, channel_metadata, datetime_metadata, signal_groups, signal_info

        # If we only had GPS data and already loaded location directly, we are done.
        if gps_location_loaded:
            print(f"✅ GPS location loaded from CSV for logger {logger_id}.")
            return None
        
        # Otherwise check for other GPS CSV files
        gps_files = self._collect_files_from_dirs(gps_dirs)
        if not gps_files:
            gps_files = self._collect_sensor_files(gps_root, logger_id)

        if gps_files:
            print(f"📥 Prepared VT GPS files for CSV importer: {gps_files}")
            csv_importer = CSVImporter(self.data_reader, logger_id)
            return csv_importer.process_files(gps_files, enforce_frequency=enforce_frequency)

        print(f"⚠️ No VT accelerometer or GPS data found for logger {logger_id}.")
        fallback_candidates = [
            f for f in files
            if f.lower().endswith((".csv", ".parquet", ".parq", ".pq"))
        ]
        if fallback_candidates:
            print("   ℹ️ Falling back to CSV importer for VT logger.")
            csv_importer = CSVImporter(self.data_reader, logger_id)
            return csv_importer.process_files(fallback_candidates, enforce_frequency=enforce_frequency)
        return None

    # ------------------------------------------------------------------
    # Accelerometer helpers
    # ------------------------------------------------------------------
    def _prepare_accelerometer_files(self, vt_dirs: List[Path]) -> List[pd.DataFrame]:
        """
        Clean raw VT accelerometer CSVs, align their timestamps, and return dataframes.
        """
        prepared_dfs: List[pd.DataFrame] = []
        if not vt_dirs:
            return prepared_dfs

        tz_name = self.data_reader.deployment_info.get("Time Zone") or "UTC"
        try:
            local_tz = pytz.timezone(tz_name)
        except Exception:
            print(f"   ⚠️ Invalid timezone '{tz_name}'. Falling back to UTC.")
            tz_name = "UTC"
            local_tz = pytz.UTC

        for vt_dir in vt_dirs:
            print(f"🔧 VectronicsImporter: Processing accelerometer dir: {vt_dir}")
            
            # Delete any existing parquet files to force regeneration
            existing_parquets = list(vt_dir.glob("*.parquet"))
            for pq_file in existing_parquets:
                print(f"   🗑️ Deleting old parquet file: {pq_file.name}")
                pq_file.unlink()
            
            csv_files = sorted(vt_dir.glob("*.csv"), key=self._sort_csv_key)
            if not csv_files:
                print(f"⚠️ No CSV files found under {vt_dir}.")
                continue

            print(f"   Found {len(csv_files)} VT CSV files inside {vt_dir.name}:")
            for path in csv_files:
                print(f"    • {path.name}")

            default_date = self._extract_date_from_name(csv_files[0])
            dfs: List[pd.DataFrame] = []

            for csv_path in csv_files:
                df = self._load_and_standardize_csv(csv_path, default_date)
                if df is not None:
                    dfs.append(df)

            if not dfs:
                print(f"   ❌ All VT CSVs in {vt_dir} failed format checks; skipping.")
                continue

            big_df = pd.concat(dfs, ignore_index=True)
            manual = big_df["datetime_raw_utc"]

            if manual.isna().all():
                print("   ❌ All timestamps are NaT; cannot continue.")
                continue

            diffs = manual.diff().dropna()
            diffs = diffs[diffs > pd.Timedelta(0)]
            if diffs.empty:
                print("   ❌ Unable to infer cadence from timestamps.")
                continue

            raw_step = diffs.median()
            raw_fs = 1.0 / raw_step.total_seconds()
            int_fs = max(1, int(round(raw_fs)))
            if int_fs <= 0:
                print(f"   ❌ Invalid inferred sampling rate: {raw_fs}")
                continue

            ideal_step = pd.Timedelta(seconds=1 / int_fs)
            print("   ⏱ VT accelerometer cadence:")
            print(f"      Median step: {raw_step}  (~{raw_fs:.3f} Hz)")
            print(f"      Integer-enforced fs: {int_fs} Hz (Δ={ideal_step})")

            first_idx = manual.first_valid_index()
            start_ts = manual.loc[first_idx]
            aligned_index = pd.date_range(start=start_ts, periods=len(big_df), freq=ideal_step)
            aligned_series = pd.Series(aligned_index, index=big_df.index)
            # For plain-combined VT files, keep `utc_time` as the source timestamp
            # so montage mapping can map it to datetime_utc without collisions.
            has_plain_utc_time = "utc_time" in big_df.columns
            if not has_plain_utc_time:
                big_df["datetime_utc"] = aligned_series
                big_df["datetime"] = big_df["datetime_utc"].dt.tz_convert(local_tz)
            else:
                big_df["datetime"] = aligned_series.dt.tz_convert(local_tz)

            tol_check = pd.Timedelta(milliseconds=5)
            dt_diff = (aligned_series - manual).abs()
            n_mismatch = (dt_diff > tol_check).sum()
            print(f"   🔍 |raw - aligned| > {tol_check}: {n_mismatch} rows")

            rename_map = {
                "ACC X [g]": "ax",
                "ACC Y [g]": "ay",
                "ACC Z [g]": "az",
                "Acc X [g]": "ax",
                "Acc Y [g]": "ay",
                "Acc Z [g]": "az",
            }
            big_df = big_df.rename(columns=rename_map)

            drop_cols = [
                "Date ISO String",
                "UTC Milliseconds since 1970",
                "UTC DateTime",
                "Milliseconds",
                "datetime_raw_utc",
                "datetime_iso_utc",
            ]
            for col in drop_cols:
                if col in big_df.columns:
                    big_df = big_df.drop(columns=col)

            ordered_cols = [col for col in ("datetime", "datetime_utc", "ax", "ay", "az") if col in big_df.columns]
            remaining_cols = [c for c in big_df.columns if c not in ordered_cols]
            final_df = big_df[ordered_cols + remaining_cols]

            # Note: We don't save to parquet here because the cleaned data will be
            # processed directly by CSVImporter. Saving would cause duplicate data.
            prepared_dfs.append(final_df)
            print(f"   ✅ Prepared {len(final_df)} accelerometer rows from {vt_dir.name}")

        return prepared_dfs

    # ------------------------------------------------------------------
    # GPS/Location KML helpers
    # ------------------------------------------------------------------
    def _collect_gps_csv_files(self, gps_dirs: List[Path], gps_root: Optional[Path]) -> List[str]:
        """
        Collect GPS CSV files from GPS directories.
        Returns list of file paths relative to data_folder.
        """
        csv_files: List[str] = []
        data_root = Path(self.data_reader.data_folder)

        # Search in logger-specific GPS directories
        for gps_dir in gps_dirs:
            for csv_path in gps_dir.rglob("*.csv"):
                if not csv_path.is_file():
                    continue
                try:
                    rel = csv_path.relative_to(data_root)
                    csv_files.append(str(rel))
                except ValueError:
                    continue

        # Search in general GPS root folder (logger-specific subfolder if available)
        if gps_root and gps_root.exists():
            gps_dir = self._resolve_vt_dir(gps_root, self.logger_id)
            if gps_dir:
                for csv_path in gps_dir.rglob("*.csv"):
                    if not csv_path.is_file():
                        continue
                    try:
                        rel = csv_path.relative_to(data_root)
                        rel_str = str(rel)
                        if rel_str not in csv_files:
                            csv_files.append(rel_str)
                    except ValueError:
                        continue

        return sorted(csv_files)

    def _process_gps_csv_files(self, gps_csv_files: List[str]) -> bool:
        """
        Parse VT GPS CSV files and store as location signal.
        Returns True if at least one valid location point was loaded.
        """
        data_root = Path(self.data_reader.data_folder)
        all_location_data: List[pd.DataFrame] = []

        for rel_path in gps_csv_files:
            csv_path = data_root / rel_path
            df = self._parse_gps_csv_file(csv_path)
            if df is not None and not df.empty:
                all_location_data.append(df)

        if not all_location_data:
            print("⚠️ GPS CSV files found, but no valid location points were extracted.")
            return False

        location_df = pd.concat(all_location_data, ignore_index=True)
        location_df = location_df.sort_values("datetime").reset_index(drop=True)

        # Reuse KML importer storage path so signal_info['location'] is standardized.
        kml_importer = KMLImporter(self.data_reader, self.logger_id)
        kml_importer._store_location_data(location_df)
        return True

    def _parse_gps_csv_file(self, csv_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse a VT GPS CSV into datetime/latitude/longitude.
        Expects UTC_Date + UTC_Time and Latitude/Longitude columns.
        """
        try:
            df = self.read_csv(str(csv_path))
        except Exception as e:
            print(f"   ❌ Error reading GPS CSV {csv_path.name}: {e}")
            return None

        if df is None or df.empty:
            print(f"   ⚠️ GPS CSV {csv_path.name} is empty.")
            return None

        def _norm_col(name: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

        col_map = {_norm_col(col): col for col in df.columns}

        utc_date_col = col_map.get("utc_date")
        utc_time_col = col_map.get("utc_time")
        latitude_col = next((orig for norm, orig in col_map.items() if norm.startswith("latitude")), None)
        longitude_col = next((orig for norm, orig in col_map.items() if norm.startswith("longitude")), None)

        if not (utc_date_col and utc_time_col and latitude_col and longitude_col):
            print(
                f"   ⚠️ GPS CSV {csv_path.name} missing required columns "
                "(UTC_Date, UTC_Time, Latitude, Longitude)."
            )
            return None

        dt_utc = pd.to_datetime(
            df[utc_date_col].astype(str).str.strip() + " " + df[utc_time_col].astype(str).str.strip(),
            utc=True,
            errors="coerce",
        )
        lat = pd.to_numeric(df[latitude_col], errors="coerce")
        lon = pd.to_numeric(df[longitude_col], errors="coerce")

        tz_name = self.data_reader.deployment_info.get("Time Zone") or "UTC"
        try:
            local_tz = pytz.timezone(tz_name)
        except Exception:
            print(f"   ⚠️ Invalid timezone '{tz_name}'. Using UTC.")
            local_tz = pytz.UTC

        parsed = pd.DataFrame({
            "datetime": dt_utc.dt.tz_convert(local_tz),
            "latitude": lat,
            "longitude": lon,
        })
        parsed = parsed.dropna(subset=["datetime", "latitude", "longitude"])
        parsed = parsed[~((parsed["latitude"] == 0) & (parsed["longitude"] == 0))]

        if parsed.empty:
            print(f"   ⚠️ No valid GPS points extracted from CSV {csv_path.name}")
            return None

        print(f"   ✅ Parsed {len(parsed)} GPS points from CSV {csv_path.name}")
        return parsed

    def _collect_kml_files(self, gps_dirs: List[Path], gps_root: Optional[Path]) -> List[str]:
        """
        Collect KML files from GPS directories.
        Returns list of file paths relative to data_folder.
        """
        kml_files = []
        data_root = Path(self.data_reader.data_folder)
        
        # Search in logger-specific GPS directories
        for gps_dir in gps_dirs:
            for kml_path in gps_dir.rglob("*.kml"):
                if kml_path.is_file():
                    try:
                        rel = kml_path.relative_to(data_root)
                        kml_files.append(str(rel))
                    except ValueError:
                        continue
            for kml_path in gps_dir.rglob("*.KML"):
                if kml_path.is_file():
                    try:
                        rel = kml_path.relative_to(data_root)
                        kml_files.append(str(rel))
                    except ValueError:
                        continue
        
        # Search in general GPS root folder
        if gps_root and gps_root.exists():
            gps_dir = self._resolve_vt_dir(gps_root, self.logger_id)
            if gps_dir:
                for kml_path in gps_dir.rglob("*.kml"):
                    if kml_path.is_file():
                        try:
                            rel = kml_path.relative_to(data_root)
                            rel_str = str(rel)
                            if rel_str not in kml_files:
                                kml_files.append(rel_str)
                        except ValueError:
                            continue
                for kml_path in gps_dir.rglob("*.KML"):
                    if kml_path.is_file():
                        try:
                            rel = kml_path.relative_to(data_root)
                            rel_str = str(rel)
                            if rel_str not in kml_files:
                                kml_files.append(rel_str)
                        except ValueError:
                            continue
        
        return sorted(kml_files)

    # ------------------------------------------------------------------
    # Accelerometer helpers
    # ------------------------------------------------------------------
    def _load_and_standardize_csv(
        self,
        csv_path: Path,
        default_date_from_name: Optional[str]
    ) -> Optional[pd.DataFrame]:
        # Read and manually trim first row if it contains metadata
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with csv_path.open("r", encoding="latin-1") as f:
                lines = f.readlines()

        if not lines:
            print(f"   ⚠️ {csv_path.name} is empty.")
            return None

        first_line_stripped = lines[0].strip()
        has_metadata_line = (
            first_line_stripped.startswith("Collar:") or
            first_line_stripped.startswith("DeviceID:") or
            "Sensor range" in first_line_stripped
        )
        
        # If metadata line exists, skip it and read from second line onwards
        if has_metadata_line:
            print(f"   🔪 Trimming metadata row from {csv_path.name}")
            lines = lines[1:]  # Remove first line
        
        # Write the trimmed content to a StringIO and read with pandas
        from io import StringIO
        csv_content = "".join(lines)
        df = pd.read_csv(StringIO(csv_content))
        
        # Debug: print what columns we actually got
        print(f"   🔍 DEBUG: Columns after reading CSV: {list(df.columns)}")
        print(f"   🔍 DEBUG: First row of data: {df.iloc[0].to_dict() if len(df) > 0 else 'EMPTY'}")
        
        fmt = self._detect_format(df)
        if fmt == "unknown":
            print(f"   ⚠️ Skipping {csv_path.name}: unknown schema.")
            return None

        if fmt == "clean":
            iso_col = "Date ISO String"
            ms_col = "UTC Milliseconds since 1970"
            dt_iso = pd.to_datetime(df[iso_col], utc=True, errors="coerce")
            dt_ms = pd.to_datetime(df[ms_col], unit="ms", utc=True, errors="coerce")

            diff = (dt_iso - dt_ms).abs()
            tolerance = pd.Timedelta(milliseconds=5)
            n_bad = (diff > tolerance).sum()
            if n_bad > 0:
                print(
                    f"   ⚠️ {csv_path.name}: {n_bad} rows where ISO and ms differ by > {tolerance}. "
                    "Using milliseconds as source of truth."
                )

            df["datetime_raw_utc"] = dt_ms
            return df
        
        if fmt == "clean_iso_only":
            # Format with ISO timestamp only (no separate milliseconds column)
            iso_col = "Date ISO String"
            dt_iso = pd.to_datetime(df[iso_col], utc=True, errors="coerce")
            df["datetime_raw_utc"] = dt_iso
            print(f"   ✅ {csv_path.name}: Using ISO timestamp (no separate ms column)")
            return df

        if fmt == "plain_combined":
            # Preserve Vectronics channel identity by converting dotted headers
            # (e.g., "X.Axis.Scaled") to snake-case with underscores.
            vt_name_map = {
                col: self._vectronics_channel_id(col)
                for col in df.columns
            }
            df = df.rename(columns=vt_name_map)

            if "utc_time" not in df.columns:
                print(f"   ⚠️ {csv_path.name}: plain schema missing UTC.Time; skipping.")
                return None

            x_col = "x_axis_scaled" if "x_axis_scaled" in df.columns else "x_axis" if "x_axis" in df.columns else None
            y_col = "y_axis_scaled" if "y_axis_scaled" in df.columns else "y_axis" if "y_axis" in df.columns else None
            z_col = "z_axis_scaled" if "z_axis_scaled" in df.columns else "z_axis" if "z_axis" in df.columns else None
            if not (x_col and y_col and z_col):
                print(f"   ⚠️ {csv_path.name}: plain schema missing X/Y/Z axis columns; skipping.")
                return None

            df["datetime_raw_utc"] = pd.to_datetime(df["utc_time"], utc=True, errors="coerce")
            # Keep original VT channel IDs so montage mapping can own standardization.
            df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
            df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
            df[z_col] = pd.to_numeric(df[z_col], errors="coerce")

            source_label = "scaled" if all(c.endswith("_scaled") for c in (x_col, y_col, z_col)) else "raw"
            print(f"   ✅ {csv_path.name}: Using plain combined schema ({source_label} axes + UTC.Time).")
            return df

        # Legacy format
        date_str = self._extract_date_from_name(csv_path)
        if not date_str:
            date_str = default_date_from_name or self._get_date_from_header(csv_path)

        if not date_str:
            print(f"   ⚠️ {csv_path.name}: unable to determine date; skipping.")
            return None

        dt_str = (
            date_str + " " +
            df["UTC DateTime"].astype(str) + "." +
            df["Milliseconds"].astype(int).astype(str).str.zfill(3)
        )
        df["datetime_raw_utc"] = pd.to_datetime(
            dt_str,
            format="%Y-%m-%d %H:%M:%S.%f",
            utc=True,
            errors="coerce",
        )
        return df

    @staticmethod
    def _detect_format(df: pd.DataFrame) -> str:
        cols = set(df.columns)
        normalized = {VectronicsImporter._normalize_col_name(col) for col in cols}
        clean_required = {
            "Date ISO String",
            "UTC Milliseconds since 1970",
            "ACC X [g]",
            "ACC Y [g]",
            "ACC Z [g]",
        }
        # Clean format without milliseconds column (timestamp is in ISO string)
        clean_iso_only = {
            "Date ISO String",
            "ACC X [g]",
            "ACC Y [g]",
            "ACC Z [g]",
        }
        legacy_required = {
            "UTC DateTime",
            "Milliseconds",
            "Acc X [g]",
            "Acc Y [g]",
            "Acc Z [g]",
        }
        plain_combined_required = {
            "utc_time",
        }
        plain_combined_axes_scaled = {
            "x_axis_scaled",
            "y_axis_scaled",
            "z_axis_scaled",
        }
        plain_combined_axes_raw = {
            "x_axis",
            "y_axis",
            "z_axis",
        }
        if clean_required.issubset(cols):
            return "clean"
        if clean_iso_only.issubset(cols):
            return "clean_iso_only"
        if legacy_required.issubset(cols):
            return "legacy"
        if plain_combined_required.issubset(normalized) and (
            plain_combined_axes_scaled.issubset(normalized)
            or plain_combined_axes_raw.issubset(normalized)
        ):
            return "plain_combined"
        return "unknown"

    @staticmethod
    def _normalize_col_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

    @staticmethod
    def _vectronics_channel_id(name: str) -> str:
        """
        Vectronics-specific column normalization for channel IDs:
        keep identity but make dots and spaces compatible with montage keys.
        """
        return str(name).strip().lower().replace(".", "_").replace(" ", "_")

    @staticmethod
    def _sort_csv_key(csv_path: Path):
        name = csv_path.name
        m = re.match(r"(\d{4}-\d{2}-\d{2})-(?:Par|Part)(\d+)\.csv$", name)
        if m:
            date_str, part_str = m.groups()
            return (date_str, int(part_str))

        m = re.match(r"(\d{4}-\d{2}-\d{2})\.csv$", name)
        if m:
            return (m.group(1), 0)
        return (name, 0)

    @staticmethod
    def _extract_date_from_name(csv_path: Path) -> Optional[str]:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", csv_path.name)
        return m.group(1) if m else None

    @staticmethod
    def _get_date_from_header(csv_path: Path) -> Optional[str]:
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        except Exception:
            return None

        parts = [p.strip() for p in first_line.split(",")]
        for part in parts:
            if part.startswith("Date:"):
                candidate = part.replace("Date:", "").strip()
                return candidate or None
        return None

    # ------------------------------------------------------------------
    # Sensor folder helpers
    # ------------------------------------------------------------------
    def _find_sensor_root(self, raw_root: Path, suffix: str) -> Optional[Path]:
        if not raw_root.exists():
            return None

        suffix_lower = suffix.lower()
        entries = [entry for entry in raw_root.iterdir() if entry.is_dir()]
        candidates = []
        manufacturer_clean = (self.logger_manufacturer or "").replace(" ", "")
        if manufacturer_clean:
            candidates.append(f"{manufacturer_clean}_{suffix_lower}")
        candidates.append(f"VT_{suffix_lower}")

        for name in candidates:
            for entry in entries:
                if entry.name.lower() == name.lower():
                    return entry

        for entry in entries:
            if entry.name.lower().endswith(f"_{suffix_lower}"):
                return entry
        return None

    def _resolve_vt_dir(self, sensor_root: Optional[Path], logger_id: str) -> Optional[Path]:
        if not sensor_root or not sensor_root.is_dir():
            return None

        direct = sensor_root / logger_id
        if direct.is_dir():
            return direct

        matches = [
            child for child in sensor_root.iterdir()
            if child.is_dir() and logger_id in child.name
        ]
        if not matches:
            return None
        if len(matches) > 1:
            print(f"   ⚠️ Multiple VT folders found for {logger_id}: {matches}. Using the first match.")
        return sorted(matches)[0]

    def _collect_sensor_files(self, sensor_root: Optional[Path], logger_id: str) -> List[str]:
        if not sensor_root or not sensor_root.is_dir():
            return []

        logger_lower = logger_id.lower()
        collected: List[str] = []
        for path in sensor_root.rglob("*"):
            if not path.is_file():
                continue
            if logger_lower not in path.name.lower():
                continue
            if path.suffix.lower() not in (".csv", ".parquet", ".parq", ".pq"):
                continue
            rel = path.relative_to(self.data_reader.data_folder)
            collected.append(str(rel))

        return sorted(collected)

    def _collect_files_from_dirs(self, dirs: List[Path]) -> List[str]:
        data_root = Path(self.data_reader.data_folder)
        collected: List[str] = []
        for directory in dirs:
            for path in directory.rglob("*"):
                if path.is_file():
                    try:
                        rel = path.relative_to(data_root)
                    except ValueError:
                        continue
                    collected.append(str(rel))
        unique: List[str] = []
        seen = set()
        for item in sorted(collected):
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _find_logger_specific_dirs(self) -> dict[str, List[Path]]:
        matches: dict[str, List[Path]] = {
            "accelerometer": [],
            "gps": [],
        }
        data_root = Path(self.data_reader.data_folder)
        prefix = f"{self.logger_id.lower()}_"
        try:
            for child in data_root.iterdir():
                if not child.is_dir():
                    continue
                name_lower = child.name.lower()
                if not name_lower.startswith(prefix):
                    continue
                sensor_part = name_lower[len(prefix):]
                tokens = [token for token in sensor_part.split("-") if token]
                for token in tokens:
                    normalized = self._normalize_sensor_token(token)
                    if normalized and normalized in matches:
                        matches[normalized].append(child)
        except FileNotFoundError:
            pass
        return matches

    @staticmethod
    def _normalize_sensor_token(token: str) -> Optional[str]:
        token = token.strip().lower()
        if token in {"acc", "accel", "accelerometer"}:
            return "accelerometer"
        if token in {"gps"}:
            return "gps"
        return None

    def _ensure_vectronics_acc_signal(self, df: pd.DataFrame, channel_metadata: dict, signal_name: str, channels: List[str]):
        """
        Ensure Vectronics accelerometer signals are present even if generic grouping misses them.
        """
        if signal_name in self.data_reader.signal_data:
            return
        if "datetime" not in df.columns:
            return

        present = [c for c in channels if c in df.columns]
        if len(present) < 1:
            return

        sig_df = df[["datetime"] + present].copy()
        for c in present:
            sig_df[c] = pd.to_numeric(sig_df[c], errors="coerce")
        sig_df = sig_df.dropna(subset=present, how="all")
        if sig_df.empty:
            return

        freq = np.nan
        try:
            from pyologger.utils.time_manager import calculate_sampling_frequency
            freq = calculate_sampling_frequency(sig_df["datetime"].head())
        except Exception:
            pass

        meta = {}
        for c in present:
            entry = channel_metadata.get(c) or {}
            if not entry:
                entry = {
                    "original_name": c,
                    "unit": "unknown",
                    "standardized_unit": "unknown",
                    "parent_signal": signal_name,
                }
            meta[c] = entry

        self.data_reader.signal_data[signal_name] = sig_df
        self.data_reader.signal_info[signal_name] = {
            "channels": present,
            "metadata": meta,
            "sampling_frequency": freq,
            "logger_id": self.logger_id,
            "logger_manufacturer": self.logger_manufacturer,
            "processing_step": "Raw data uploaded (Vectronics fallback materialization)",
            "details": f"Ensured {signal_name} from channels {present}.",
        }
        print(f"✅ Ensured signal_data['{signal_name}'] with channels {present} (rows={len(sig_df)}).")
