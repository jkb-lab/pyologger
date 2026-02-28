from pyologger.io_operations.base_importer import BaseImporter
import os
import re
import pandas as pd
import numpy as np
import pytz
from scipy.io import loadmat
from pyologger.utils.time_manager import process_datetime
from pyologger.io_operations.kml_importer import KMLImporter

class CATSImporter(BaseImporter):
    """CATS-specific processing."""

    def process_files(self, files, enforce_frequency=True):
        """
        Process CATS files, handling .txt separately and ignoring .ubx, .ubc, .bin, .cfg files.
        
        Args:
            files (list): List of files to process.
            enforce_frequency (bool): Whether to enforce expected frequencies. Default is True.
        """
        # Filter out unwanted files
        files = [f for f in files if not f.endswith(('.ubc', '.bin', '.ubx', '.cfg', '.obs', '.pos', '.stat', '.nsl', '.kml', '.conv'))]

        # Dedicated path for processed CATS MATLAB files.
        mat_files = [f for f in files if f.lower().endswith(".mat")]
        if mat_files:
            print(f"📥 Detected {len(mat_files)} CATS processed .mat file(s). Using MAT importer path.")
            return self.process_processed_mat_files(mat_files, enforce_frequency=enforce_frequency)

        # Step 1: Parse .txt file for signal intervals
        txt_file = next((f for f in files if f.endswith('.txt')), None)
        parsed_signals = {}
        if txt_file:
            print(f"🔍 Parsing {txt_file} for expected signal intervals.")
            parsed_signals = self.parse_txt_file(os.path.join(self.data_reader.data_folder, txt_file))

        # Step 2: Set expected frequencies (can be overridden)
        self.set_expected_frequencies(parsed_signals, enforce_frequency=enforce_frequency)

        # Remove .txt files from the list after processing them
        files = [f for f in files if not f.endswith('.txt')]

        if not files:
            print(f"⚠ No valid files found for {self.logger_manufacturer} logger.")
            return None, None, None, None, None  # Return five None values

        # Step 3: Concatenate remaining files into one DataFrame
        final_df = self.concatenate_and_save_csvs(files)
        
        # Step 4: Rename columns
        original_channel_names = final_df.columns.tolist()
        new_channel_names, channel_metadata = self.rename_channels(original_channel_names)
        final_df.rename(columns=new_channel_names, inplace=True)
        print(f"✅ Renamed columns: {new_channel_names}")

        # Step 5: Process datetime and return metadata
        final_df, datetime_metadata = process_datetime(final_df, time_zone=self.data_reader.deployment_info['Time Zone'])
        self.data_reader.logger_info[self.logger_id]['datetime_created_from'] = datetime_metadata.get('datetime_created_from', None)
        self.data_reader.logger_info[self.logger_id]['fs'] = [datetime_metadata.get('fs', None)]

        # Step 6: Map data to signals and return signal information
        signal_groups, signal_info = self.group_data_by_signals(final_df, self.logger_id, channel_metadata)

        return final_df, channel_metadata, datetime_metadata, signal_groups, signal_info

    def process_processed_mat_files(self, mat_files, enforce_frequency=True):
        """
        Process processed CATS .mat files (e.g., cats-processed-mat_V1).

        DN is interpreted as local wall-clock MATLAB datenum.
        UTC is derived via INFO.UTC where local = UTC + offset_hours.
        """
        tz_name = self.data_reader.deployment_info.get("Time Zone")
        if tz_name:
            tz_name = str(tz_name).strip()
        if not tz_name:
            tz_name = "UTC"

        all_frames = []
        info_blobs = []
        utc_offsets = []

        for rel_path in mat_files:
            file_path = os.path.join(self.data_reader.data_folder, rel_path)
            print(f"📥 Loading CATS processed MAT: {file_path}")
            try:
                mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
            except Exception as e:
                print(f"❌ Failed to read {rel_path}: {e}")
                continue

            dn = mat.get("DN")
            if dn is None:
                print(f"⚠️ Skipping {rel_path}: missing DN.")
                continue

            dn = np.asarray(dn).astype(float).ravel()
            # Ensure Series (not DatetimeIndex) so downstream .dt operations are valid.
            dt_local = pd.Series(pd.to_datetime(dn - 719529, unit="D", errors="coerce"))

            info = mat.get("INFO")
            info_utc = self._extract_info_utc_offset(info)
            if info_utc is None:
                print(f"⚠️ INFO.UTC missing in {rel_path}; using UTC offset 0.")
                info_utc = 0.0
            utc_offsets.append(info_utc)

            dt_utc = (dt_local - pd.to_timedelta(info_utc, unit="h")).dt.tz_localize("UTC")
            try:
                dt_local_tz = dt_utc.dt.tz_convert(tz_name)
            except Exception:
                print(f"⚠️ Invalid deployment timezone '{tz_name}'. Falling back to UTC.")
                tz_name = "UTC"
                dt_local_tz = dt_utc

            self._check_timezone_vs_info_utc(dt_utc, tz_name, info_utc, rel_path)

            row_count = len(dt_utc)
            df = pd.DataFrame({
                "datetime": dt_local_tz,
                "datetime_utc": dt_utc,
                "dn": dn,
            })

            for key in ["At", "Aw", "Gt", "Gw", "Mt", "Mw", "GPS", "GPSerr", "Ptrack", "geoPtrack"]:
                arr = mat.get(key)
                if arr is None:
                    continue
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                if arr.ndim != 2 or arr.shape[0] != row_count:
                    continue

                base = self._mat_key_to_base(key)
                for i in range(arr.shape[1]):
                    df[f"{base}_{i}"] = pd.to_numeric(arr[:, i], errors="coerce")

            for key in [
                "Light", "T", "p", "pitch", "roll", "head", "flownoise", "tagon",
                "camon", "audon", "vidDN", "vidDurs", "viddeploy"
            ]:
                arr = mat.get(key)
                if arr is None:
                    continue
                arr = np.asarray(arr).ravel()
                if len(arr) == row_count:
                    df[key.lower()] = pd.to_numeric(arr, errors="coerce")

            # Keep vidNam if vector length aligns, even though it's string-like.
            vidnam = mat.get("vidNam")
            if vidnam is not None:
                arr = np.asarray(vidnam).ravel()
                if len(arr) == row_count:
                    df["vidnam"] = arr.astype(str)

            all_frames.append(df)
            info_blobs.append({
                "file": rel_path,
                "info": info,
                "utc_offset": info_utc,
                "gps": mat.get("GPS"),
            })

        if not all_frames:
            print("❌ No valid MAT frames produced.")
            return None, None, None, None, None

        final_df = pd.concat(all_frames, ignore_index=True).sort_values("datetime_utc").reset_index(drop=True)

        # Rename columns via montage mapping.
        original_channel_names = final_df.columns.tolist()
        new_channel_names, channel_metadata = self.rename_channels(original_channel_names)
        final_df.rename(columns=new_channel_names, inplace=True)
        print(f"✅ Renamed columns: {new_channel_names}")

        # Keep datetime columns as authoritative (already timezone-correct).
        final_df, datetime_metadata = process_datetime(
            final_df,
            time_zone=tz_name,
            channel_metadata=channel_metadata
        )
        self.data_reader.logger_info[self.logger_id]['datetime_created_from'] = datetime_metadata.get('datetime_created_from', None)
        self.data_reader.logger_info[self.logger_id]['fs'] = [datetime_metadata.get('fs', None)]

        self.data_reader.logger_info[self.logger_id]['cats_processed_mat'] = {
            "source_files": [b["file"] for b in info_blobs],
            "utc_offsets_hours": utc_offsets,
            "timezone_used": tz_name,
        }

        # For processed CATS tags, derive a location track from geoptrack offsets
        # when start location is available in INFO/deployment metadata.
        self._add_geoptrack_location(final_df, info_blobs)

        signal_groups, signal_info = self.group_data_by_signals(final_df, self.logger_id, channel_metadata)
        return final_df, channel_metadata, datetime_metadata, signal_groups, signal_info

    @staticmethod
    def _mat_key_to_base(key: str) -> str:
        mapping = {
            "At": "at",
            "Aw": "aw",
            "Gt": "gt",
            "Gw": "gw",
            "Mt": "mt",
            "Mw": "mw",
            "GPS": "gps",
            "GPSerr": "gpserr",
            "Ptrack": "ptrack",
            "geoPtrack": "geoptrack",
        }
        return mapping.get(key, key.lower())

    @staticmethod
    def _extract_info_utc_offset(info_obj):
        """
        Extract INFO.UTC offset hours from scipy-loaded MATLAB struct.
        """
        if info_obj is None:
            return None
        try:
            value = getattr(info_obj, "UTC", None)
            if value is None and isinstance(info_obj, dict):
                value = info_obj.get("UTC")
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                value = value.ravel()
                if len(value) == 0:
                    return None
                value = value[0]
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _check_timezone_vs_info_utc(dt_utc: pd.Series, tz_name: str, info_utc_hours: float, source_name: str):
        """
        Compare deployment timezone UTC offset to INFO.UTC and warn on mismatch.
        local = UTC + INFO.UTC
        """
        if dt_utc is None or len(dt_utc) == 0:
            return
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            return
        first_utc = pd.Timestamp(dt_utc.iloc[0]).to_pydatetime()
        local_dt = first_utc.astimezone(tz)
        tz_offset = local_dt.utcoffset()
        if tz_offset is None:
            return
        tz_offset_hours = tz_offset.total_seconds() / 3600.0
        if abs(tz_offset_hours - float(info_utc_hours)) > 0.5:
            print(
                f"⚠️ Timezone offset mismatch in {source_name}: "
                f"INFO.UTC={info_utc_hours:+.2f}h vs deployment TZ '{tz_name}' "
                f"offset={tz_offset_hours:+.2f}h at {local_dt.isoformat()}."
            )

    def _add_geoptrack_location(self, final_df: pd.DataFrame, info_blobs):
        """
        Build/augment location (lat/lon) from geoptrack meter offsets.
        Expects standardized columns geopos_x/geopos_y in meters.
        """
        if "geopos_x" not in final_df.columns or "geopos_y" not in final_df.columns:
            return
        if "datetime" not in final_df.columns:
            return

        start_lat, start_lon = self._extract_start_location(info_blobs)
        if start_lat is None or start_lon is None:
            print("⚠️ Could not determine start lat/lon from INFO/deployment metadata for geoptrack conversion.")
            return

        geo = final_df[["datetime", "geopos_x", "geopos_y"]].copy()
        geo["geopos_x"] = pd.to_numeric(geo["geopos_x"], errors="coerce")
        geo["geopos_y"] = pd.to_numeric(geo["geopos_y"], errors="coerce")
        geo = geo.dropna(subset=["datetime", "geopos_x", "geopos_y"])
        if geo.empty:
            return

        lat_arr, lon_arr = self._meters_offset_to_latlon(
            start_lat,
            start_lon,
            geo["geopos_x"].to_numpy(),
            geo["geopos_y"].to_numpy(),
        )
        location_df = pd.DataFrame({
            "datetime": geo["datetime"].values,
            "latitude": lat_arr,
            "longitude": lon_arr,
        })
        location_df = location_df.dropna(subset=["datetime", "latitude", "longitude"])
        if location_df.empty:
            return

        first_aug = location_df.iloc[0]
        last_aug = location_df.iloc[-1]
        print(
            "🔎 Geoptrack-derived location span: "
            f"first=({first_aug['latitude']:.6f}, {first_aug['longitude']:.6f}), "
            f"last=({last_aug['latitude']:.6f}, {last_aug['longitude']:.6f}), "
            f"n={len(location_df)}"
        )

        kml_importer = KMLImporter(self.data_reader, self.logger_id)
        kml_importer._store_location_data(location_df)
        print(
            f"✅ Added geoptrack-derived location from start ({start_lat:.6f}, {start_lon:.6f}) "
            f"with {len(location_df)} points."
        )

    def _extract_start_location(self, info_blobs):
        """
        Prefer INFO-provided start coordinates, then deployment metadata.
        Returns (lat, lon) or (None, None).
        """
        gps_start = self._extract_single_unique_gps_start(info_blobs)
        if gps_start is not None:
            return gps_start

        # If GPS was provided but not uniquely stable, skip geoptrack-based construction.
        any_gps = any((blob.get("gps") is not None) for blob in info_blobs if isinstance(blob, dict))
        if any_gps:
            print("⚠️ GPS field has multiple unique valid coordinates; skipping geoptrack-derived location.")
            return None, None

        for blob in info_blobs:
            info = blob.get("info") if isinstance(blob, dict) else None
            lat, lon = self._extract_lat_lon_from_info(info)
            if lat is not None and lon is not None:
                return lat, lon

        dep_lat = self._coerce_float(self.data_reader.deployment_info.get("Deployment Latitude"))
        dep_lon = self._coerce_float(self.data_reader.deployment_info.get("Deployment Longitude"))
        if self._is_valid_lat_lon(dep_lat, dep_lon):
            return dep_lat, dep_lon
        return None, None

    def _extract_single_unique_gps_start(self, info_blobs):
        """
        Parse GPS arrays from MAT blobs and select a start coordinate when:
        - exactly one unique valid coordinate exists -> use it
        - exactly two unique valid coordinates exist -> use first observed
        Otherwise return None.
        """
        points = []
        for blob in info_blobs:
            if not isinstance(blob, dict):
                continue
            gps = blob.get("gps")
            if gps is None:
                continue
            arr = np.asarray(gps)
            if arr.size == 0:
                continue

            if arr.ndim == 1:
                if arr.size >= 2:
                    arr = arr.reshape(-1, 2)
                else:
                    continue
            elif arr.ndim >= 2 and arr.shape[1] >= 2:
                arr = arr[:, :2]
            else:
                continue

            for lat, lon in arr:
                try:
                    latf = float(lat)
                    lonf = float(lon)
                except Exception:
                    continue
                if not (np.isfinite(latf) and np.isfinite(lonf)):
                    continue
                if self._is_valid_lat_lon(latf, lonf):
                    points.append((latf, lonf))

        if not points:
            return None

        first_raw = points[0]
        last_raw = points[-1]
        print(
            f"🔎 MAT GPS raw span before geoptrack augmentation: "
            f"first=({first_raw[0]:.6f}, {first_raw[1]:.6f}), "
            f"last=({last_raw[0]:.6f}, {last_raw[1]:.6f}), n={len(points)}"
        )

        # Round for stable uniqueness under tiny floating jitter, while preserving order.
        unique_ordered = []
        seen = set()
        for lat, lon in points:
            key = (round(lat, 6), round(lon, 6))
            if key not in seen:
                seen.add(key)
                unique_ordered.append(key)

        if len(unique_ordered) == 1:
            lat, lon = unique_ordered[0]
            print(f"✅ Using single unique GPS start coordinate from MAT GPS field: ({lat}, {lon})")
            return float(lat), float(lon)

        if len(unique_ordered) == 2:
            lat, lon = unique_ordered[0]
            print(
                f"✅ GPS has exactly two unique points; using first observed as start: "
                f"({lat}, {lon}). Second unique: ({unique_ordered[1][0]}, {unique_ordered[1][1]})"
            )
            return float(lat), float(lon)

        return None

    def _extract_lat_lon_from_info(self, info_obj):
        if info_obj is None:
            return None, None

        field_pairs = [
            ("startlat", "startlon"),
            ("start_lat", "start_lon"),
            ("deploylat", "deploylon"),
            ("deploymentlat", "deploymentlon"),
            ("latitude", "longitude"),
            ("lat", "lon"),
        ]

        for lat_key, lon_key in field_pairs:
            lat = self._coerce_float(self._get_info_field(info_obj, lat_key))
            lon = self._coerce_float(self._get_info_field(info_obj, lon_key))
            if self._is_valid_lat_lon(lat, lon):
                return lat, lon

        gps_val = self._get_info_field(info_obj, "gps")
        if gps_val is not None:
            arr = np.asarray(gps_val).astype(float).ravel()
            if arr.size >= 2:
                lat, lon = float(arr[0]), float(arr[1])
                if self._is_valid_lat_lon(lat, lon):
                    return lat, lon

        return None, None

    @staticmethod
    def _get_info_field(info_obj, key: str):
        if info_obj is None:
            return None
        # dict-like
        if isinstance(info_obj, dict):
            for k, v in info_obj.items():
                if str(k).strip().lower() == key:
                    return v
            return None
        # scipy mat_struct
        try:
            if hasattr(info_obj, "_fieldnames"):
                for field in getattr(info_obj, "_fieldnames", []):
                    if str(field).strip().lower() == key:
                        return getattr(info_obj, field, None)
            if hasattr(info_obj, key):
                return getattr(info_obj, key, None)
        except Exception:
            return None
        return None

    @staticmethod
    def _coerce_float(value):
        if value is None:
            return None
        try:
            if isinstance(value, np.ndarray):
                flat = value.astype(float).ravel()
                if flat.size == 0:
                    return None
                return float(flat[0])
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _is_valid_lat_lon(lat, lon):
        if lat is None or lon is None:
            return False
        return -90 <= lat <= 90 and -180 <= lon <= 180

    @staticmethod
    def _meters_offset_to_latlon(start_lat, start_lon, east_m, north_m):
        """
        Convert local East/North meter offsets to latitude/longitude.
        """
        lat_deg_per_m = 1.0 / 111320.0
        lon_deg_per_m = 1.0 / (111320.0 * np.cos(np.deg2rad(start_lat)))
        lon_deg_per_m = np.where(np.isfinite(lon_deg_per_m), lon_deg_per_m, 0.0)

        lat = start_lat + (north_m * lat_deg_per_m)
        lon = start_lon + (east_m * lon_deg_per_m)
        return lat, lon

    def concatenate_and_save_csvs(self, csv_files):
        """Concatenates multiple CSV files into one DataFrame."""
        dfs = []
        for file in csv_files:
            file_path = os.path.join(self.data_reader.data_folder, file)
            try:
                data = self.read_csv(file_path)
                dfs.append(data)
                print(f"{self.logger_manufacturer} file: {file} - Successfully processed.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")

        if len(dfs) > 1:
            concatenated_df = pd.concat(dfs, ignore_index=True)
        else:
            concatenated_df = dfs[0]

        return concatenated_df

    def print_txt_content(self, txt_file):
        """Prints the content of a .txt file."""
        file_path = os.path.join(self.data_reader.data_folder, txt_file)
        with open(file_path, 'r') as file:
            print(file.read())

    def parse_txt_file(self, txt_file_path):
        """Parses the .txt file to extract sensor names and sampling intervals."""
        print(f"🔍 Attempting to parse intervals from {txt_file_path}")

        try:
            with open(txt_file_path, 'r') as file:
                content = file.read()

            # Extract sensor information from the '[activated sensors]' section
            activated_signals_section = re.search(r'\[activated sensors\](.*?)\n\n', content, re.DOTALL)
            if not activated_signals_section:
                print("⚠ No 'activated signals' section found in the file.")
                return {}

            activated_signals_content = activated_signals_section.group(1)

            # Find all signals' names and their corresponding intervals
            signal_info = re.findall(r'(\d{2})_name=(.*?)\n.*?\1_interval=(\d+)', activated_signals_content, re.DOTALL)
            if not signal_info:
                print("⚠ No signal information found in the file. Please check the file format.")
                return {}

            print(f"✅ Parsed signal info from txt: {signal_info}")

            # Convert to dictionary {signal_name: interval}
            parsed_signals = {signal_name.strip().lower(): int(interval) for _, signal_name, interval in signal_info}
            return parsed_signals

        except Exception as e:
            print(f"❌ Failed to parse {txt_file_path} due to: {e}")
            return {}
