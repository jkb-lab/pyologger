import os
from datetime import timedelta

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from pyologger.io_operations.base_importer import BaseImporter
from pyologger.utils.time_manager import process_datetime


class DTImporter(BaseImporter):
    """Importer for DTAG/TagTools NetCDF files."""

    DEFAULT_TARGET_HZ = 1.0
    DEFAULT_MAX_ROWS = 2_000_000

    def __init__(self, data_reader, logger_id, target_hz: float | None = None, max_rows: int | None = None):
        super().__init__(data_reader, logger_id)
        self.target_hz = self.DEFAULT_TARGET_HZ if target_hz is None else target_hz
        self.max_rows = self.DEFAULT_MAX_ROWS if max_rows is None else max_rows

        if not getattr(self, "montage", None):
            self.montage = self._default_montage()

    def process_files(self, files, enforce_frequency=True):
        nc_files = [f for f in files if f.lower().endswith(".nc")]
        if not nc_files:
            print(f"⚠ No NetCDF files found for {self.logger_id} (DT).")
            return None, None, None, None, None

        sens_files = sorted(f for f in nc_files if "sens" in os.path.basename(f).lower())
        trk_files = sorted(f for f in nc_files if "trk" in os.path.basename(f).lower())
        prof_files = sorted(f for f in nc_files if "prof" in os.path.basename(f).lower())

        if not sens_files:
            raise FileNotFoundError(f"No *sens*.nc file found for logger {self.logger_id}.")

        sens_path = os.path.join(self.data_reader.data_folder, sens_files[0])
        print(f"🔄 Processing DT logger {self.logger_id} from NetCDF: {sens_files[0]}")

        signal_frames, signal_info = self._load_sens_frames(sens_path)

        if trk_files:
            trk_path = os.path.join(self.data_reader.data_folder, trk_files[0])
            print(f"📍 Loading DT track file: {trk_files[0]}")
            location_df, location_info = self._load_trk_frame(trk_path)
            if location_df is not None and not location_df.empty:
                signal_frames.append(location_df)
                signal_info["location"] = location_info

        if prof_files:
            print(f"ℹ️ DT profile file present but not yet imported: {prof_files[0]}")

        if not signal_frames:
            print(f"⚠ No DT signal frames were created for {self.logger_id}.")
            return None, None, None, None, None

        final_df = self._merge_signal_frames(signal_frames)
        if final_df.empty:
            print(f"⚠ DT final dataframe is empty for {self.logger_id}.")
            return None, None, None, None, None

        original_cols = final_df.columns.tolist()
        new_cols, channel_metadata = self.rename_channels(original_cols)
        if new_cols:
            final_df.rename(columns=new_cols, inplace=True)
            print(f"✅ Renamed DT columns for {self.logger_id}: {new_cols}")

        final_df, datetime_metadata = process_datetime(
            final_df,
            time_zone=self.data_reader.deployment_info.get("Time Zone"),
            channel_metadata=channel_metadata,
        )

        self.data_reader.logger_info[self.logger_id]["datetime_created_from"] = datetime_metadata.get(
            "datetime_created_from",
            None,
        )
        self.data_reader.logger_info[self.logger_id]["fs"] = [datetime_metadata.get("fs", None)]

        signal_groups, grouped_signal_info = self.group_data_by_signals(
            final_df,
            self.logger_id,
            channel_metadata,
        )

        for signal_name, metadata in signal_info.items():
            existing = self.data_reader.signal_info.get(signal_name, {})
            existing.update({k: v for k, v in metadata.items() if pd.notna(v)})
            self.data_reader.signal_info[signal_name] = existing

        return final_df, channel_metadata, datetime_metadata, signal_groups, grouped_signal_info

    def _default_montage(self):
        return {
            "datetime": {
                "standardized_channel_id": "datetime_utc",
                "original_unit": "YYYY-MM-DD HH:MM:SS.000",
                "standardized_unit": "YYYY-MM-DD HH:MM:SS.000",
                "parent_signal": "clock",
                "manufacturer_signal_name": "clock",
            },
            "depth": {
                "standardized_channel_id": "depth",
                "original_unit": "m",
                "standardized_unit": "m",
                "parent_signal": "depth",
                "manufacturer_signal_name": "depth",
            },
            "temperature": {
                "standardized_channel_id": "temp_ext",
                "original_unit": "deg C",
                "standardized_unit": "deg C",
                "parent_signal": "temperature_ext",
                "manufacturer_signal_name": "temperature",
            },
            "jerk": {
                "standardized_channel_id": "jerk",
                "original_unit": "unknown",
                "standardized_unit": "unknown",
                "parent_signal": "jerk",
                "manufacturer_signal_name": "jerk",
            },
            "ax": {
                "standardized_channel_id": "ax",
                "original_unit": "m/s^2",
                "standardized_unit": "m/s^2",
                "parent_signal": "accelerometer",
                "manufacturer_signal_name": "accelerometer",
            },
            "ay": {
                "standardized_channel_id": "ay",
                "original_unit": "m/s^2",
                "standardized_unit": "m/s^2",
                "parent_signal": "accelerometer",
                "manufacturer_signal_name": "accelerometer",
            },
            "az": {
                "standardized_channel_id": "az",
                "original_unit": "m/s^2",
                "standardized_unit": "m/s^2",
                "parent_signal": "accelerometer",
                "manufacturer_signal_name": "accelerometer",
            },
            "mx": {
                "standardized_channel_id": "mx",
                "original_unit": "uT",
                "standardized_unit": "uT",
                "parent_signal": "magnetometer",
                "manufacturer_signal_name": "magnetometer",
            },
            "my": {
                "standardized_channel_id": "my",
                "original_unit": "uT",
                "standardized_unit": "uT",
                "parent_signal": "magnetometer",
                "manufacturer_signal_name": "magnetometer",
            },
            "mz": {
                "standardized_channel_id": "mz",
                "original_unit": "uT",
                "standardized_unit": "uT",
                "parent_signal": "magnetometer",
                "manufacturer_signal_name": "magnetometer",
            },
            "lat": {
                "standardized_channel_id": "latitude",
                "original_unit": "decimal-degrees",
                "standardized_unit": "DD",
                "parent_signal": "location",
                "manufacturer_signal_name": "location",
            },
            "lon": {
                "standardized_channel_id": "longitude",
                "original_unit": "decimal-degrees",
                "standardized_unit": "DD",
                "parent_signal": "location",
                "manufacturer_signal_name": "location",
            },
        }

    def _clean_scalar(self, value):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            value = value.item()
        text = str(value).strip()
        if text in {"", "UNKNOWN", "Unknown", "unknown", "nan", "None"}:
            return pd.NA
        return value

    def _global_attrs(self, nc: Dataset):
        return {name: self._clean_scalar(nc.getncattr(name)) for name in nc.ncattrs()}

    def _parse_device_start(self, attrs):
        raw = attrs.get("dephist_device_datetime_start", pd.NA)
        if pd.isna(raw):
            raise ValueError("Missing dephist_device_datetime_start; cannot build DT datetimes.")

        tz_offset_h = attrs.get("dephist_device_tzone", 0)
        try:
            tz_offset_h = float(tz_offset_h)
        except Exception:
            tz_offset_h = 0.0

        start_naive = pd.to_datetime(str(raw), format="%Y/%m/%d %H:%M:%S", errors="coerce")
        if pd.isna(start_naive):
            start_naive = pd.to_datetime(str(raw), errors="coerce")
        if pd.isna(start_naive):
            raise ValueError(f"Could not parse dephist_device_datetime_start={raw}")

        start_utc = pd.Timestamp(start_naive) - timedelta(hours=tz_offset_h)
        return start_utc.tz_localize("UTC")

    def _compute_stride(self, fs: float, n: int) -> int:
        stride = 1
        if self.target_hz and fs and fs > self.target_hz:
            stride = max(1, int(round(fs / self.target_hz)))
        if self.max_rows and n > self.max_rows * stride:
            stride = max(stride, int(np.ceil(n / self.max_rows)))
        return stride

    def _build_regular_time(self, start_utc: pd.Timestamp, n: int, fs: float, start_offset_s: float, stride: int):
        idx = np.arange(0, n, stride, dtype=float)
        t_s = start_offset_s + (idx / fs)
        return start_utc + pd.to_timedelta(t_s, unit="s")

    def _to_float_array(self, arr):
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        return np.asarray(arr, dtype=float)

    def _load_sens_frames(self, sens_path):
        signal_frames = []
        signal_info = {}

        specs = {
            "P": ("depth", ["depth"]),
            "T": ("temperature", ["temperature"]),
            "J": ("jerk", ["jerk"]),
            "A": ("accelerometer", ["ax", "ay", "az"]),
            "M": ("magnetometer", ["mx", "my", "mz"]),
        }

        with Dataset(sens_path, "r") as nc:
            gattrs = self._global_attrs(nc)
            start_utc = self._parse_device_start(gattrs)
            self._update_reader_metadata(gattrs)

            for var_name, (signal_name, cols) in specs.items():
                if var_name not in nc.variables:
                    continue

                var = nc.variables[var_name]
                arr = self._to_float_array(var[:])
                fs = float(var.getncattr("sampling_rate")) if "sampling_rate" in var.ncattrs() else 1.0
                start_offset_s = float(var.getncattr("start_offset")) if "start_offset" in var.ncattrs() else 0.0
                n = arr.shape[-1] if arr.ndim == 2 else arr.shape[0]
                stride = self._compute_stride(fs, n)
                dt = self._build_regular_time(start_utc, n=n, fs=fs, start_offset_s=start_offset_s, stride=stride)

                if arr.ndim == 1:
                    values = arr[::stride]
                    frame = pd.DataFrame({"datetime": dt, cols[0]: values})
                else:
                    values = arr[:, ::stride]
                    frame = pd.DataFrame({"datetime": dt})
                    for idx, col in enumerate(cols):
                        frame[col] = values[idx, :] if values.shape[0] > idx else np.nan

                value_cols = [c for c in frame.columns if c != "datetime"]
                frame = frame.dropna(how="all", subset=value_cols)
                if frame.empty:
                    continue

                signal_frames.append(frame)
                signal_info[signal_name] = {
                    "source_var": var_name,
                    "sampling_rate_hz": fs,
                    "stride": stride,
                    "rows": int(len(frame)),
                    "units": self._clean_scalar(var.getncattr("unit")) if "unit" in var.ncattrs() else pd.NA,
                    "description": self._clean_scalar(var.getncattr("description")) if "description" in var.ncattrs() else pd.NA,
                }

        return signal_frames, signal_info

    def _load_trk_frame(self, trk_path):
        with Dataset(trk_path, "r") as nc:
            gattrs = self._global_attrs(nc)
            start_utc = self._parse_device_start(gattrs)

            if "POS" not in nc.variables:
                return None, None

            pos = self._to_float_array(nc.variables["POS"][:])
            if pos.ndim != 2 or pos.shape[0] < 3:
                return None, None

            t_offset_s = pos[0, :]
            lat = pos[1, :]
            lon = pos[2, :]
            good = np.isfinite(t_offset_s)
            if not good.any():
                return None, None

            dt = start_utc + pd.to_timedelta(t_offset_s[good], unit="s")
            loc_df = pd.DataFrame({"datetime": dt, "lat": lat[good], "lon": lon[good]})
            loc_df = loc_df.dropna(how="all", subset=["lat", "lon"])
            if loc_df.empty:
                return None, None

            loc_info = {
                "source_var": "POS",
                "sampling": "irregular",
                "rows": int(len(loc_df)),
                "description": self._clean_scalar(nc.variables["POS"].getncattr("description"))
                if "description" in nc.variables["POS"].ncattrs()
                else pd.NA,
            }
            return loc_df, loc_info

    def _merge_signal_frames(self, signal_frames):
        merged = signal_frames[0].sort_values("datetime").reset_index(drop=True).copy()
        for frame in signal_frames[1:]:
            merged = merged.merge(
                frame.sort_values("datetime").reset_index(drop=True),
                on="datetime",
                how="outer",
            )
        merged = merged.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        return merged

    def _update_reader_metadata(self, attrs):
        logger_details = {
            "Manufacturer": "DT",
            "Tag Type": "DTAG",
            "Model": attrs.get("device_model", pd.NA),
            "device_make": attrs.get("device_make", pd.NA),
            "device_model": attrs.get("device_model", pd.NA),
            "device_serial": attrs.get("device_serial", pd.NA),
            "device_type": attrs.get("device_type", pd.NA),
        }
        self.data_reader.logger_info[self.logger_id].update(
            {key: value for key, value in logger_details.items() if pd.notna(value)}
        )

        deployment_updates = {
            "Deployment ID": attrs.get("depid", pd.NA),
            "Deployment Date": attrs.get("dephist_deploy_datetime_start", pd.NA),
            "Deployment Latitude": attrs.get("dephist_deploy_location_lat", pd.NA),
            "Deployment Longitude": attrs.get("dephist_deploy_location_lon", pd.NA),
        }
        for key, value in deployment_updates.items():
            if pd.notna(value) and pd.isna(self.data_reader.deployment_info.get(key)):
                self.data_reader.deployment_info[key] = value

        tz_value = attrs.get("dephist_device_tzone", pd.NA)
        if pd.notna(tz_value):
            try:
                tz_float = float(tz_value)
                if np.isclose(tz_float, 0.0):
                    tz_text = "UTC"
                else:
                    sign = "+" if tz_float >= 0 else "-"
                    hours = int(abs(tz_float))
                    minutes = int(round((abs(tz_float) - hours) * 60))
                    tz_text = f"UTC{sign}{hours:02d}:{minutes:02d}"
            except Exception:
                tz_text = str(tz_value)

            if not self.data_reader.deployment_info.get("Time Zone"):
                self.data_reader.deployment_info["Time Zone"] = tz_text
