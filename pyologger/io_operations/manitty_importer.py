from pyologger.io_operations.base_importer import BaseImporter
from edfio import read_edf
import pandas as pd
import numpy as np
from math import floor
import os
import re
from pyologger.utils.time_manager import process_datetime

class ManittyImporter(BaseImporter):
    """Manitty processing for EDF files, tabular exports, or both."""

    TABULAR_EXTS = (".csv", ".parquet", ".parq", ".pq")

    def process_files(self, files, enforce_frequency=True):
        edf_file = next((f for f in files if f.endswith('.edf')), None)
        tabular_files = [f for f in files if f.lower().endswith(self.TABULAR_EXTS)]

        if not edf_file and not tabular_files:
            print("❌ No EDF or CSV/Parquet file found for Manitty logger.")
            return {}, {}, {}, {}, {}

        if not edf_file:
            print(f"ℹ️ No EDF for {self.logger_id}; importing {len(tabular_files)} tabular file(s).")
            return self._process_tabular_files(tabular_files, enforce_frequency)

        edf_result = self._process_edf_file(edf_file, enforce_frequency)

        if tabular_files:
            print(
                f"ℹ️ {self.logger_id} also has {len(tabular_files)} tabular file(s); "
                f"importing alongside the EDF."
            )
            self._process_tabular_files(tabular_files, enforce_frequency)

        return edf_result

    def _process_tabular_files(self, tabular_files, enforce_frequency=True):
        """Delegate CSV/Parquet handling to the generic tabular importer."""
        from pyologger.io_operations.csv_importer import CSVImporter

        importer = CSVImporter(self.data_reader, self.logger_id)
        try:
            return importer.process_files(tabular_files, enforce_frequency)
        except Exception as exc:
            print(f"⚠️ Failed to import tabular files for {self.logger_id}: {type(exc).__name__}: {exc}")
            return {}, {}, {}, {}, {}

    def _process_edf_file(self, edf_file, enforce_frequency=True):
        edf_path = os.path.join(self.data_reader.data_folder, edf_file)
        print(f"📥 Reading EDF: {edf_path}")
        edf = read_edf(edf_path)

        if not isinstance(getattr(self, "montage", None), dict) or not self.montage:
            raise ValueError(
                f"No valid montage mapping loaded for logger '{self.logger_id}' "
                f"({self.logger_manufacturer}, requested montage '{self.montage_id}')."
            )

        normalized_montage = {
            self.normalize_channel_key(key): value
            for key, value in self.montage.items()
        }

        # Filter, map and retain
        retained_signals = []
        channel_metadata_all = {}

        for signal in edf.signals:
            cleaned = self.normalize_channel_key(signal.label)
            if cleaned in normalized_montage:
                mapping = normalized_montage[cleaned]
                parent_signal = mapping['parent_signal'].lower()
                if parent_signal in ['exg', 'logger_status']:
                    continue
                standardized_id = mapping['standardized_channel_id']
                original_label = signal.label
                signal.label = standardized_id
                retained_signals.append(signal)
                channel_metadata_all[standardized_id] = {
                    'original_name': original_label,
                    'unit': mapping.get('original_unit', 'unknown'),
                    'signal': parent_signal,
                    'parent_signal': parent_signal,
                }
            else:
                print(f"ℹ️ Skipping unmapped Manitty EDF channel '{signal.label}' (normalized '{cleaned}').")

        if not retained_signals:
            print("❌ No valid signals retained after montage mapping.")
            return {}, {}, {}, {}, {}

        # Reorder signals by label group
        order_prefixes = ['ecg', 'eog', 'emg', 'eeg', 'a', 'm', 'g']
        def sort_key(label):
            for i, p in enumerate(order_prefixes):
                if label.startswith(p):
                    return i
            return len(order_prefixes)

        retained_signals = sorted(retained_signals, key=lambda s: sort_key(s.label))

        # === Build and return outputs ===
        time_zone = self.data_reader.deployment_info.get("Time Zone")

        # Grouping by frequency is shared with the other EDF importers; see
        # BaseImporter.group_edf_signals_by_frequency for the memory-conscious build.
        grouped_dfs = self.group_edf_signals_by_frequency(
            retained_signals,
            startdate=edf.startdate,
            starttime=edf.starttime,
            time_zone=time_zone,
            skip_full=False,
            channel_metadata=channel_metadata_all
        )

        final_dfs = {}
        channel_metadata = {}
        datetime_metadata = {}
        signal_groups = {}
        signal_info = {}

        for freq, df in grouped_dfs.items():
            print(f"\n📦 Processing frequency group: {freq} Hz")

            df, dt_meta = process_datetime(df, time_zone)
            final_dfs[freq] = df
            datetime_metadata[freq] = dt_meta

            # Per-frequency metadata
            cols_in_df = set(df.columns) - {'datetime'}
            metadata_for_df = {col: channel_metadata_all[col] for col in cols_in_df if col in channel_metadata_all}
            channel_metadata[freq] = metadata_for_df

            g, i = self.group_data_by_signals(df, self.logger_id, metadata_for_df)

            # Only store signals not already processed
            signal_groups[freq] = {k: v for k, v in g.items() if k not in self.data_reader.signal_data}
            signal_info[freq] = {k: v for k, v in i.items() if k not in self.data_reader.signal_info}

            self.data_reader.logger_info[self.logger_id]['datetime_created_from'] = dt_meta.get('datetime_created_from', None)
            self.data_reader.logger_info[self.logger_id]['fs'] = list(final_dfs.keys())

        return final_dfs, channel_metadata, datetime_metadata, signal_groups, signal_info
