from pyologger.io_operations.base_importer import BaseImporter
from edfio import read_edf
import pandas as pd
import numpy as np
from math import floor
import os
import re
from pyologger.utils.time_manager import process_datetime

class EvolocusImporter(BaseImporter):
    """Evolocus-specific processing for EDF files with multiple frequency outputs."""

    def process_files(self, files, enforce_frequency=True):
        edf_file = next((f for f in files if f.endswith('.edf')), None)
        if not edf_file:
            print("❌ No EDF file found for Evolocus logger.")
            return {}, {}, {}, {}, {}

        edf_path = os.path.join(self.data_reader.data_folder, edf_file)
        print(f"📥 Reading EDF: {edf_path}")
        edf = read_edf(edf_path)

        # Clean + normalize label
        def clean_label(label):
            return re.sub(r'[^\w]', '', label.lower().replace(' ', ''))

        # Filter, map and retain
        retained_signals = []
        channel_metadata_all = {}

        for signal in edf.signals:
            cleaned = clean_label(signal.label)
            if cleaned in self.montage:
                mapping = self.montage[cleaned]
                sensor_type = mapping['standardized_sensor_type'].lower()
                if sensor_type in ['exg', 'logger_status']:
                    continue
                standardized_id = mapping['standardized_channel_id']
                signal.label = standardized_id
                retained_signals.append(signal)
                channel_metadata_all[standardized_id] = {
                    'original_name': signal.label,
                    'unit': mapping.get('original_unit', 'unknown'),
                    'sensor': sensor_type
                }

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

        # === Grouping Logic (by frequency) ===
        def construct_datetime_array(signals, sampling_frequency, startdate, starttime, time_zone='UTC'):
            start_time = pd.to_datetime(f"{startdate} {starttime}").tz_localize(time_zone)
            num_samples = len(next(s for s in signals if s.__dict__.get('_sampling_frequency') == sampling_frequency).data)
            time_offsets = np.arange(num_samples) / sampling_frequency
            return start_time + pd.to_timedelta(time_offsets, unit='s')

        def group_signals_by_frequency(signals, startdate, starttime, time_zone='UTC', skip_full=False):
            freq_dict = {}
            for s in signals:
                freq = s.__dict__.get('_sampling_frequency')
                freq_dict.setdefault(freq, []).append(s)

            grouped_data = {}
            for freq, sigs in freq_dict.items():
                print(f"\n🔍 Grouping {len(sigs)} signals at {freq} Hz")
                timestamps = construct_datetime_array(sigs, freq, startdate, starttime, time_zone)

                # Always build preview
                signal_preview = {s.label: s.data[:10] for s in sigs}
                preview_df = pd.DataFrame(signal_preview)
                preview_df['datetime'] = timestamps[:10]
                preview_df = preview_df.reset_index(drop=True)

                if skip_full:
                    print(f"⚡ Skipping full data for {freq} Hz — using preview only")
                    grouped_data[freq] = preview_df
                    continue

                # Build full data
                int_freq = floor(freq)
                full_df = pd.DataFrame({s.label: s.data for s in sigs})
                full_df['datetime'] = timestamps

                if not np.isclose(freq, int_freq):
                    print(f"⏬ Resampling from {freq:.4f} Hz to {int_freq} Hz...")
                    df = self.resample_mean_dataframe(full_df, int_freq)
                    grouped_data[int_freq] = df
                else:
                    grouped_data[int_freq] = full_df.reset_index(drop=True)

            return grouped_data

        # === Build and return outputs ===
        time_zone = self.data_reader.deployment_info.get("Time Zone")

        grouped_dfs = group_signals_by_frequency(
            retained_signals,
            startdate=edf.startdate,
            starttime=edf.starttime,
            time_zone=time_zone,
            skip_full=False
        )

        final_dfs = {}
        channel_metadata = {}
        datetime_metadata = {}
        sensor_groups = {}
        sensor_info = {}

        for freq, df in grouped_dfs.items():
            print(f"\n📦 Processing frequency group: {freq} Hz")

            df, dt_meta = process_datetime(df, time_zone)
            final_dfs[freq] = df
            datetime_metadata[freq] = dt_meta

            # Per-frequency metadata
            cols_in_df = set(df.columns) - {'datetime'}
            metadata_for_df = {col: channel_metadata_all[col] for col in cols_in_df if col in channel_metadata_all}
            channel_metadata[freq] = metadata_for_df

            g, i = self.group_data_by_sensors(df, self.logger_id, metadata_for_df)

            # Only store sensors not already processed
            sensor_groups[freq] = {k: v for k, v in g.items() if k not in self.data_reader.sensor_data}
            sensor_info[freq] = {k: v for k, v in i.items() if k not in self.data_reader.sensor_info}

            self.data_reader.logger_info[self.logger_id]['datetime_created_from'] = dt_meta.get('datetime_created_from', None)
            self.data_reader.logger_info[self.logger_id]['fs'] = list(final_dfs.keys())

        return final_dfs, channel_metadata, datetime_metadata, sensor_groups, sensor_info

    def resample_mean_dataframe(self, df, target_freq_hz):
        df = df.copy()

        df = df.set_index('datetime')
        target_period_ms = int(round(1000 / target_freq_hz))

        df_resampled = df.resample(f"{target_period_ms}ms").mean()

        return df_resampled.reset_index()

