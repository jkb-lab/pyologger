from __future__ import annotations

import os
import pickle
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pyologger.utils.workflow_netcdf import latest_processing_netcdf_path


EVENT_COLUMNS = [
    "datetime",
    "type",
    "key",
    "value",
    "duration",
    "short_description",
    "long_description",
]

DEFAULT_PLOT_SIGNAL_PRIORITY = [
    "depth",
    "corrected_depth",
    "ecg",
    "heart_rate",
    "prh",
    "stroke_rate",
    "pressure",
    "accelerometer",
    "magnetometer",
    "gyroscope",
    "light",
    "temperature",
    "odba",
    "speed",
    "velocity",
    "position",
]

NON_PLOT_DEFAULT_SIGNALS = {
    "clock",
    "time",
    "logger_status",
    "dives",
    "location",
    # derived/intermediate signals — shown only when explicitly added
    "calibrated_acc",
    "calibrated_mag",
    "corrected_acc",
    "corrected_gyr",
    "corrected_mag",
    "heart_rate_fixed",
    "heart_rate_nan",
    "hr_normalized",
    "sr_smoothed",
}

NETCDF_IO_LOCK = threading.RLock()


def _to_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [str(v) for v in value]
    return [str(value)]


def _normalize_datetimes(values, tz_name: str | None = None) -> pd.Series:
    series = pd.to_datetime(pd.Series(values), errors="coerce", utc=True)
    if tz_name:
        try:
            return series.dt.tz_convert(tz_name)
        except Exception:
            return series
    return series


def _event_frame_from_rows(rows: list[dict[str, Any]], tz_name: str | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.DataFrame(rows)
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["datetime"] = _normalize_datetimes(df["datetime"], tz_name)
    return df[EVENT_COLUMNS].dropna(subset=["datetime"]).sort_values("datetime", kind="mergesort").reset_index(drop=True)


def _subsample_preserve_ends(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or max_points <= 0 or len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    return df.iloc[idx].reset_index(drop=True)


def _deserialize_attr_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _extract_prefixed_attrs(attrs: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    plen = len(prefix)
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        out[key[plen:]] = _deserialize_attr_value(value)
    return out


def _extract_signal_info_from_attrs(attrs: dict[str, Any], signal_name: str, channels: list[str]) -> dict[str, Any]:
    base_prefix = f"signal_info_{signal_name}_"
    signal_info = _extract_prefixed_attrs(attrs, base_prefix)
    meta_prefix = "metadata_"
    metadata: dict[str, Any] = {}
    for ch in channels:
        ch_prefix = f"{meta_prefix}{ch}_"
        metadata[ch] = _extract_prefixed_attrs(signal_info, ch_prefix)
    cleaned = {}
    for key, value in signal_info.items():
        if key.startswith(meta_prefix):
            continue
        cleaned[key] = value
    cleaned["channels"] = list(channels)
    cleaned["metadata"] = metadata
    return cleaned


@dataclass
class SignalMetadata:
    name: str
    variable_name: str
    sample_dim: str
    sample_count: int
    channels: list[str] = field(default_factory=list)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    signal_info: dict[str, Any] = field(default_factory=dict)


class DeploymentDataSource:
    def __init__(
        self,
        deployment_folder: str,
        deployment_id: str,
        *,
        real_data_pkl=None,
        netcdf_path: str | None = None,
        pkl_path: str | None = None,
    ):
        """
        netcdf_path / pkl_path: explicit overrides for which processed file to
        read, bypassing the standard outputs/{deployment_id}_output.nc /
        outputs/data.pkl lookup -- e.g. to load a trimmed demo slice
        (outputs_demo/*_output_trimmed.nc, outputs_demo/data_trimmed.pkl)
        instead of the full deployment's real outputs. When netcdf_path is
        given it always wins (netcdf takes priority over pickle, same as the
        default lookup); pkl_path is only used if no netcdf is found.
        """
        self.deployment_folder = deployment_folder
        self.deployment_id = deployment_id
        self._pkl_path_override = pkl_path
        self.netcdf_path = netcdf_path or latest_processing_netcdf_path(deployment_folder, deployment_id)
        self._real_data_pkl = real_data_pkl
        self._metadata_shell = None
        self._location_cache: dict[int, pd.DataFrame] = {}
        self._event_df_cache: pd.DataFrame | None = None
        self.signal_meta: dict[str, SignalMetadata] = {}
        self.deployment_info: dict[str, Any] = {}
        self.animal_info: dict[str, Any] = {}
        self.dataset_info: dict[str, Any] = {}
        self.event_data = pd.DataFrame(columns=EVENT_COLUMNS)
        self.global_start: pd.Timestamp | None = None
        self.global_end: pd.Timestamp | None = None
        self.backend = "pickle"
        if self.netcdf_path and os.path.exists(self.netcdf_path):
            self.backend = "netcdf"
            self._scan_netcdf()
        else:
            self._scan_pickle()

    def _scan_pickle(self):
        data_pkl = self.get_real_data_pkl()
        self.deployment_info = dict(getattr(data_pkl, "deployment_info", {}) or {})
        self.animal_info = dict(getattr(data_pkl, "animal_info", {}) or {})
        self.dataset_info = dict(getattr(data_pkl, "dataset_info", {}) or {})
        signal_data = getattr(data_pkl, "signal_data", {}) or {}
        signal_info = getattr(data_pkl, "signal_info", {}) or {}
        for sig, df in signal_data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            dt = _normalize_datetimes(df.get("datetime", pd.Series(dtype="object")), self.tz_name)
            dt = dt.dropna()
            channels = [str(c) for c in df.columns if c != "datetime"]
            self.signal_meta[str(sig)] = SignalMetadata(
                name=str(sig),
                variable_name=f"signal_data_{sig}",
                sample_dim=f"{sig}_samples",
                sample_count=len(df),
                channels=channels,
                start=(dt.min() if not dt.empty else None),
                end=(dt.max() if not dt.empty else None),
                signal_info=dict(signal_info.get(sig, {}) or {}),
            )
        self.event_data = getattr(data_pkl, "event_data", pd.DataFrame(columns=EVENT_COLUMNS)).copy()
        if "datetime" in self.event_data.columns:
            self.event_data["datetime"] = _normalize_datetimes(self.event_data["datetime"], self.tz_name)
        self._finalize_bounds()

    def _scan_netcdf(self):
        with NETCDF_IO_LOCK:
            with xr.open_dataset(self.netcdf_path) as ds:
                attrs = {str(k): _deserialize_attr_value(v) for k, v in dict(ds.attrs).items()}
                self.deployment_info = _extract_prefixed_attrs(attrs, "deployment_info_")
                self.animal_info = _extract_prefixed_attrs(attrs, "animal_info_")
                self.dataset_info = _extract_prefixed_attrs(attrs, "dataset_info_")

                for var_name in ds.data_vars:
                    if not str(var_name).startswith("signal_data_"):
                        continue
                    signal_name = str(var_name)[len("signal_data_") :]
                    da = ds[var_name]
                    sample_dim = next((dim for dim in da.dims if str(dim).endswith("_samples")), "")
                    sample_coord = da.coords.get(sample_dim)
                    coord_values = sample_coord.values if sample_coord is not None else np.array([])
                    dt = _normalize_datetimes(coord_values, self.tz_name).dropna()
                    channels = _to_list(da.attrs.get("variables")) or _to_list(da.attrs.get("variable"))
                    if not channels and len(da.dims) > 1:
                        var_dim = next((dim for dim in da.dims if str(dim).endswith("_variables")), None)
                        if var_dim and var_dim in da.coords:
                            channels = [str(v) for v in da.coords[var_dim].values.tolist()]
                    self.signal_meta[signal_name] = SignalMetadata(
                        name=signal_name,
                        variable_name=str(var_name),
                        sample_dim=str(sample_dim),
                        sample_count=int(len(coord_values)),
                        channels=[str(c) for c in channels],
                        start=(dt.min() if not dt.empty else None),
                        end=(dt.max() if not dt.empty else None),
                        signal_info=_extract_signal_info_from_attrs(attrs, signal_name, [str(c) for c in channels]),
                    )

                self.event_data = self._read_all_events_from_dataset(ds)
        self._finalize_bounds()

    def _finalize_bounds(self):
        starts = [meta.start for meta in self.signal_meta.values() if meta.start is not None]
        ends = [meta.end for meta in self.signal_meta.values() if meta.end is not None]
        self.global_start = min(starts) if starts else None
        self.global_end = max(ends) if ends else None

    @property
    def tz_name(self) -> str:
        return str(self.deployment_info.get("Time Zone") or self.deployment_info.get("TimeZone") or "UTC")

    def get_real_data_pkl(self):
        if self._real_data_pkl is None:
            pkl_path = self._pkl_path_override or os.path.join(self.deployment_folder, "outputs", "data.pkl")
            with open(pkl_path, "rb") as fh:
                self._real_data_pkl = pickle.load(fh)
        return self._real_data_pkl

    def _read_all_events_from_dataset(self, ds: xr.Dataset | None = None) -> pd.DataFrame:
        if self._event_df_cache is not None:
            return self._event_df_cache.copy()
        with NETCDF_IO_LOCK:
            close_ds = False
            if ds is None:
                ds = xr.open_dataset(self.netcdf_path)
                close_ds = True
            try:
                event_vars = [str(name) for name in ds.data_vars if str(name).startswith("event_data_")]
                if not event_vars:
                    self._event_df_cache = pd.DataFrame(columns=EVENT_COLUMNS)
                    return self._event_df_cache.copy()
                sample_var = ds[event_vars[0]]
                sample_dim = next((dim for dim in sample_var.dims if str(dim).endswith("_samples")), None)
                if sample_dim is None:
                    self._event_df_cache = pd.DataFrame(columns=EVENT_COLUMNS)
                    return self._event_df_cache.copy()
                datetimes = _normalize_datetimes(sample_var.coords[sample_dim].values, self.tz_name)
                rows = []
                column_map = {}
                for var_name in event_vars:
                    col = str(var_name)[len("event_data_") :]
                    values = ds[var_name].values
                    if np.ndim(values) > 1:
                        values = np.asarray(values).reshape(-1)
                    column_map[col] = list(values)
                n_rows = len(datetimes)
                for idx in range(n_rows):
                    row = {"datetime": datetimes.iloc[idx]}
                    for col, values in column_map.items():
                        row[col] = values[idx] if idx < len(values) else pd.NA
                    rows.append(row)
                self._event_df_cache = _event_frame_from_rows(rows, self.tz_name)
                return self._event_df_cache.copy()
            finally:
                if close_ds:
                    ds.close()

    def event_keys(self) -> list[str]:
        if not isinstance(self.event_data, pd.DataFrame) or "key" not in self.event_data.columns:
            return []
        return sorted([str(k) for k in self.event_data["key"].dropna().unique()], key=lambda x: x.lower())

    def signal_names(self) -> list[str]:
        return sorted(self.signal_meta.keys())

    def build_metadata_shell(self, allowed_signals: list[str] | None = None):
        allowed = [str(s) for s in (allowed_signals or self.signal_names()) if str(s) in self.signal_meta]
        signal_data = {}
        signal_info = {}
        for sig, meta in self.signal_meta.items():
            cols = ["datetime"] + list(meta.channels)
            signal_data[sig] = pd.DataFrame(columns=cols)
            signal_info[sig] = dict(meta.signal_info or {"channels": list(meta.channels), "metadata": {}})
            signal_info[sig].setdefault("channels", list(meta.channels))
            signal_info[sig].setdefault("metadata", {})
        shell = SimpleNamespace(
            signal_data=signal_data,
            signal_info=signal_info,
            event_data=self.event_data.copy(),
            deployment_info=dict(self.deployment_info),
            animal_info=dict(self.animal_info),
            dataset_info=dict(self.dataset_info),
            deployment_name=self.deployment_id,
            plot_signal_allowlist=list(allowed),
            global_start=self.global_start,
            global_end=self.global_end,
            __header_only__=True,
        )
        self._metadata_shell = shell
        return shell

    def refresh_shell(self, allowed_signals: list[str] | None = None):
        self._metadata_shell = None
        return self.build_metadata_shell(allowed_signals=allowed_signals)

    def replace_cached_events(self, event_df: pd.DataFrame):
        self.event_data = event_df.copy() if isinstance(event_df, pd.DataFrame) else pd.DataFrame(columns=EVENT_COLUMNS)
        self._event_df_cache = self.event_data.copy()
        if self._metadata_shell is not None:
            self._metadata_shell.event_data = self.event_data.copy()

    def load_signal_window(self, signal_name: str, start_ts, end_ts) -> pd.DataFrame:
        signal_name = str(signal_name)
        if signal_name not in self.signal_meta:
            return pd.DataFrame(columns=["datetime"])
        if self.backend == "pickle":
            data_pkl = self.get_real_data_pkl()
            df = (getattr(data_pkl, "signal_data", {}) or {}).get(signal_name)
            if not isinstance(df, pd.DataFrame) or "datetime" not in df.columns:
                return pd.DataFrame(columns=["datetime"] + list(self.signal_meta[signal_name].channels))
            work = df.copy()
            dt = _normalize_datetimes(work["datetime"], self.tz_name)
            keep = dt.notna() & (dt >= pd.Timestamp(start_ts)) & (dt <= pd.Timestamp(end_ts))
            work = work.loc[keep].copy()
            work["datetime"] = dt.loc[keep].values
            return work.reset_index(drop=True)

        meta = self.signal_meta[signal_name]
        start_ts = pd.Timestamp(start_ts)
        end_ts = pd.Timestamp(end_ts)
        with NETCDF_IO_LOCK:
            with xr.open_dataset(self.netcdf_path) as ds:
                da = ds[meta.variable_name]
                indexer = {meta.sample_dim: slice(start_ts.tz_convert("UTC").tz_localize(None) if start_ts.tzinfo else start_ts, end_ts.tz_convert("UTC").tz_localize(None) if end_ts.tzinfo else end_ts)}
                sliced = da.sel(indexer)
                if sliced.size == 0:
                    return pd.DataFrame(columns=["datetime"] + list(meta.channels))
                dt = _normalize_datetimes(sliced.coords[meta.sample_dim].values, self.tz_name)
                values = np.asarray(sliced.values)
                if values.ndim == 1:
                    col = meta.channels[0] if meta.channels else signal_name
                    out = pd.DataFrame({"datetime": dt, col: values})
                else:
                    channels = list(meta.channels) or [str(v) for v in range(values.shape[1])]
                    out = pd.DataFrame(values, columns=channels)
                    out.insert(0, "datetime", dt)
                out["datetime"] = dt
                return out.dropna(subset=["datetime"]).reset_index(drop=True)

    def load_events_window(self, start_ts, end_ts) -> pd.DataFrame:
        if not isinstance(self.event_data, pd.DataFrame) or self.event_data.empty or "datetime" not in self.event_data.columns:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        start_ts = pd.Timestamp(start_ts)
        end_ts = pd.Timestamp(end_ts)
        dt = _normalize_datetimes(self.event_data["datetime"], self.tz_name)
        keep = dt.notna() & (dt >= start_ts) & (dt <= end_ts)
        out = self.event_data.loc[keep].copy()
        out["datetime"] = dt.loc[keep]
        return out.reset_index(drop=True)

    def load_plot_payload(self, signals: list[str], start_ts, end_ts, *, include_events: bool = True):
        shell = SimpleNamespace(
            signal_data={},
            signal_info={},
            event_data=self.load_events_window(start_ts, end_ts) if include_events else pd.DataFrame(columns=EVENT_COLUMNS),
            deployment_info=dict(self.deployment_info),
            animal_info=dict(self.animal_info),
            dataset_info=dict(self.dataset_info),
            deployment_name=self.deployment_id,
        )
        for sig in [str(s) for s in (signals or []) if str(s) in self.signal_meta]:
            shell.signal_data[sig] = self.load_signal_window(sig, start_ts, end_ts)
            shell.signal_info[sig] = dict(self.signal_meta[sig].signal_info or {})
            shell.signal_info[sig].setdefault("channels", list(self.signal_meta[sig].channels))
            shell.signal_info[sig].setdefault("metadata", {})
        return shell

    def load_location_track(self, max_points: int = 5000) -> pd.DataFrame:
        max_points = max(1, int(max_points))
        cached = self._location_cache.get(max_points)
        if isinstance(cached, pd.DataFrame):
            return cached.copy()
        if "location" not in self.signal_meta:
            df = pd.DataFrame(columns=["datetime", "latitude", "longitude"])
        else:
            meta = self.signal_meta["location"]
            start = meta.start or self.global_start
            end = meta.end or self.global_end
            if start is None or end is None:
                df = pd.DataFrame(columns=["datetime", "latitude", "longitude"])
            elif self.backend == "netcdf":
                with NETCDF_IO_LOCK:
                    with xr.open_dataset(self.netcdf_path, create_default_indexes=False) as ds:
                        da = ds.get(meta.variable_name)
                        if da is None:
                            df = pd.DataFrame(columns=["datetime"] + list(meta.channels))
                        else:
                            total = int(da.sizes.get(meta.sample_dim, 0))
                            if total <= 0:
                                df = pd.DataFrame(columns=["datetime"] + list(meta.channels))
                            else:
                                step = max(1, total // max_points)
                                sampled = da.isel({meta.sample_dim: slice(0, None, step)})
                                dt = _normalize_datetimes(sampled.coords[meta.sample_dim].values, self.tz_name)
                                values = np.asarray(sampled.values)
                                if values.ndim == 1:
                                    col = meta.channels[0] if meta.channels else "location"
                                    df = pd.DataFrame({"datetime": dt, col: values})
                                else:
                                    channels = list(meta.channels) or [str(v) for v in range(values.shape[1])]
                                    df = pd.DataFrame(values, columns=channels)
                                    df.insert(0, "datetime", dt)

                                # Preserve the exact last point when stride sampling skips it.
                                if total > 1 and (total - 1) % step != 0:
                                    tail = da.isel({meta.sample_dim: [total - 1]})
                                    tail_dt = _normalize_datetimes(tail.coords[meta.sample_dim].values, self.tz_name)
                                    tail_values = np.asarray(tail.values)
                                    if tail_values.ndim == 1:
                                        tail_col = meta.channels[0] if meta.channels else "location"
                                        tail_df = pd.DataFrame({"datetime": tail_dt, tail_col: tail_values})
                                    else:
                                        tail_channels = list(meta.channels) or [str(v) for v in range(tail_values.shape[1])]
                                        tail_df = pd.DataFrame(tail_values, columns=tail_channels)
                                        tail_df.insert(0, "datetime", tail_dt)
                                    df = pd.concat([df, tail_df], ignore_index=True)

                                df = df.dropna(subset=["datetime"]).reset_index(drop=True)
            else:
                df = self.load_signal_window("location", start, end)
                df = _subsample_preserve_ends(df, max_points)
        self._location_cache[max_points] = df.copy()
        return df


def resolve_deployment_source(
    data_dir: str | None,
    dataset_id: str | None,
    deployment_id: str,
    *,
    deployment_folder: str | None = None,
    netcdf_path: str | None = None,
    pkl_path: str | None = None,
) -> DeploymentDataSource:
    folder = deployment_folder
    if not folder:
        folder = os.path.join(str(data_dir or ""), str(dataset_id or ""), str(deployment_id))
    return DeploymentDataSource(folder, deployment_id, netcdf_path=netcdf_path, pkl_path=pkl_path)


def resolve_plot_signal_allowlist(param_manager, source: DeploymentDataSource) -> list[str]:
    available = [sig for sig in source.signal_names() if sig != "location"]
    saved_cfg = {}
    try:
        saved_cfg = (param_manager.get_from_config(
            ["dash_default_signals", "plotly_default_signals"],
            section="settings",
        ) or {})
    except Exception:
        saved_cfg = {}
    saved_signals = saved_cfg.get("dash_default_signals") or saved_cfg.get("plotly_default_signals") or []
    filtered = [str(sig) for sig in (saved_signals or []) if str(sig) in available]
    if filtered:
        return filtered
    available_lower = {str(sig).lower(): sig for sig in available}
    preferred = [available_lower[sig] for sig in DEFAULT_PLOT_SIGNAL_PRIORITY if sig in available_lower]
    remaining = [
        sig for sig in available
        if sig not in preferred and str(sig).lower() not in NON_PLOT_DEFAULT_SIGNALS
    ]
    fallback = preferred + remaining
    if fallback:
        return fallback
    return list(available)
