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
        return df

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
