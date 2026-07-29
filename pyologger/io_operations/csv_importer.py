import os
import pandas as pd
from pyologger.io_operations.base_importer import BaseImporter
from pyologger.utils.time_manager import process_datetime

class CSVImporter(BaseImporter):
    """Generic tabular importer for CSV *and* Parquet files."""

    PARQUET_EXTS = (".parquet", ".parq", ".pq")
    CSV_EXTS = (".csv",)

    def process_files(self, files, enforce_frequency=True):
        logger_id = self.logger_id
        # accept either "Time Zone" or "TimeZone"
        tz = (
            self.data_reader.deployment_info.get("Time Zone")
            or self.data_reader.deployment_info.get("TimeZone")
        )

        # select CSV + Parquet files
        selected_files = [
            f for f in files
            if f.lower().endswith(self.CSV_EXTS + self.PARQUET_EXTS)
        ]

        # Optional montage-driven filename filter. Loggers whose exports split
        # incompatible layouts across sibling files (e.g. Wildlife Computers
        # ArchivedSeries vs Series vs GPE3) must ingest only the files matching
        # the active montage, since all selected files are concatenated below.
        include_patterns = self.get_file_include_patterns()
        if include_patterns:
            kept = [
                f for f in selected_files
                if any(p.lower() in os.path.basename(f).lower() for p in include_patterns)
            ]
            skipped = [f for f in selected_files if f not in kept]
            if skipped:
                print(
                    f"ℹ️ Montage file filter {include_patterns} for {logger_id}: "
                    f"keeping {[os.path.basename(f) for f in kept]}, "
                    f"skipping {[os.path.basename(f) for f in skipped]}."
                )
            # Geolocation products are excluded from the sensor montage (their
            # layout is incompatible), but still carry the deployment's track.
            # Ingest them separately into signal_data['location'].
            self._import_geolocation_files(skipped)
            selected_files = kept

        selected_files.sort()  # reproducible concatenation order

        n_csv = sum(f.lower().endswith(self.CSV_EXTS) for f in selected_files)
        n_parq = sum(f.lower().endswith(self.PARQUET_EXTS) for f in selected_files)
        print(
            f"🔄 Processing {self.logger_manufacturer} logger {logger_id} with "
            f"{n_csv} CSV and {n_parq} Parquet file(s)."
        )

        if not selected_files:
            print(f"⚠ No valid CSV/Parquet files found for {logger_id} ({self.logger_manufacturer}).")
            return None, None, None, None, None

        # --- 1) read each file once
        dfs = []
        for fname in selected_files:
            file_path = os.path.join(self.data_reader.data_folder, fname)
            df_part = self._read_file(file_path)
            dfs.append(df_part)
            print(f"{self.logger_manufacturer} file for {logger_id}: {fname} - successfully read.")

        # --- 2) concat once
        final_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # --- 2.5) drop completely empty rows
        drop_mask = final_df.isna().all(axis=1)
        if drop_mask.any():
            final_df = final_df.loc[~drop_mask].reset_index(drop=True)

        # --- 3) rename columns once
        original_cols = final_df.columns.tolist()
        new_cols, channel_metadata = self.rename_channels(original_cols)
        if new_cols:
            final_df.rename(columns=new_cols, inplace=True)
            print(f"✅ Renamed columns for {logger_id}: {new_cols}")
        else:
            print(f"ℹ️ No column renames applied for {logger_id}.")

        # Manufacturer-specific transforms can be injected by subclasses.
        final_df = self.apply_post_rename_transforms(final_df, channel_metadata)

        # --- 4) build datetime once
        final_df, datetime_metadata = process_datetime(
            final_df,
            time_zone=tz,
            channel_metadata=channel_metadata
        )
        self.data_reader.logger_info[logger_id]['datetime_created_from'] = datetime_metadata.get('datetime_created_from', None)
        self.data_reader.logger_info[logger_id]['fs'] = [datetime_metadata.get('fs', None)]

        # --- 5) group into signals once
        signal_groups, signal_info = self.group_data_by_signals(
            final_df,
            logger_id,
            channel_metadata
        )

        # --- 6) return once
        return final_df, channel_metadata, datetime_metadata, signal_groups, signal_info

    def apply_post_rename_transforms(self, df: pd.DataFrame, channel_metadata: dict) -> pd.DataFrame:
        """Hook for manufacturer-specific transforms after channel renaming."""
        return self.convert_acceleration_to_standard_unit(df, channel_metadata)

    def _import_geolocation_files(self, candidate_files) -> None:
        """
        Ingest light/SST geolocation products (Wildlife Computers GPE3) into
        ``signal_data['location']``.

        These files are excluded from the sensor montage because their layout is
        incompatible with the per-sample sensor exports, but they carry the
        deployment's position track. The track is smoothed before storage:
        light-based geolocation error is one to two orders of magnitude larger
        than Argos/GPS, so the raw fixes are noisy enough that path length is
        substantially inflated (see utils.geolocation_smoother).
        """
        from pyologger.io_operations.gpe3_reader import find_gpe3_files, read_gpe3
        from pyologger.io_operations.kml_importer import KMLImporter
        from pyologger.utils.geolocation_smoother import smooth_geolocation_track

        data_folder = self.data_reader.data_folder
        gpe3_files = [
            os.path.join(data_folder, f)
            for f in (candidate_files or [])
            if "gpe3" in os.path.basename(f).lower() and f.lower().endswith(".csv")
        ]
        if not gpe3_files:
            # Fall back to scanning, in case no montage filter was applied.
            gpe3_files = find_gpe3_files(data_folder)
        if not gpe3_files:
            return

        frames = []
        for path in sorted(set(gpe3_files)):
            try:
                frames.append(read_gpe3(path))
            except Exception as error:
                print(f"⚠️ Could not read GPE3 file {os.path.basename(path)}: {error}")
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return

        raw = pd.concat(frames, ignore_index=True).sort_values("datetime")
        try:
            smoothed = smooth_geolocation_track(raw)
        except Exception as error:
            print(f"⚠️ Geolocation smoothing failed ({error}); storing raw GPE3 positions.")
            smoothed = raw.assign(
                latitude_smoothed=raw["latitude"], longitude_smoothed=raw["longitude"]
            )

        location_df = pd.DataFrame({
            "datetime": smoothed["datetime"],
            # Prefer the smoothed track, falling back to the raw fix where a
            # position was flagged as an outlier and therefore not smoothed.
            "latitude": smoothed["latitude_smoothed"].fillna(smoothed["latitude"]),
            "longitude": smoothed["longitude_smoothed"].fillna(smoothed["longitude"]),
        }).dropna(subset=["datetime", "latitude", "longitude"])

        tz = (
            self.data_reader.deployment_info.get("Time Zone")
            or self.data_reader.deployment_info.get("TimeZone")
            or "UTC"
        )
        location_df["datetime"] = pd.to_datetime(location_df["datetime"])
        if location_df["datetime"].dt.tz is None:
            location_df["datetime"] = location_df["datetime"].dt.tz_localize(tz)

        n_outliers = int(smoothed["outlier"].sum()) if "outlier" in smoothed else 0
        print(
            f"🌍 GPE3 geolocation: {len(location_df):,} positions from "
            f"{[os.path.basename(p) for p in gpe3_files]} "
            f"({n_outliers} speed outlier(s) excluded from smoothing)."
        )
        KMLImporter(self.data_reader, self.logger_id)._store_location_data(location_df)

    def _get_signal_columns(self, channel_metadata: dict, parent_signal: str, available_columns) -> list[str]:
        return [
            column_name
            for column_name, metadata in channel_metadata.items()
            if metadata.get("parent_signal") == parent_signal and column_name in available_columns
        ]

    def _pick_preferred_column(self, columns: list[str], preferred_name: str) -> str | None:
        if preferred_name in columns:
            return preferred_name
        if len(columns) == 1:
            return columns[0]
        return None

    # -------- helpers --------
    def _read_file(self, path: str) -> pd.DataFrame:
        """Dispatch to CSV or Parquet reader based on extension."""
        low = path.lower()
        if low.endswith(self.CSV_EXTS):
            # Uses BaseImporter.read_csv() (handles messy headers, comments, etc.)
            return self.read_csv(path)
        elif low.endswith(self.PARQUET_EXTS):
            return self._read_parquet(path)
        raise ValueError(f"Unsupported file type for: {path}")

    def _read_parquet(self, path: str) -> pd.DataFrame:
        """Parquet reader with a safe engine fallback."""
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            # Fallback to fastparquet if pyarrow isn't available
            return pd.read_parquet(path, engine="fastparquet")
