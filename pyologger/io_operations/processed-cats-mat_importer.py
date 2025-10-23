from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy.io import loadmat

from pyologger.io_operations.base_importer import BaseImporter

# -----------------------------
# Helpers
# -----------------------------

def _matlab_datenum_to_datetime(dn: np.ndarray, tz: Optional[str] = "UTC") -> pd.Series:
    """Convert MATLAB datenum array to timezone-aware pandas Timestamps.

    MATLAB datenum is days since 0000-01-00. Pandas/NumPy use the proleptic Gregorian calendar with
    different epoch; the canonical conversion is:
      dt = datetime.fromordinal(int(dn)) \
           + timedelta(days=float(dn)%1) \
           - timedelta(days=366)

    Parameters
    ----------
    dn : np.ndarray
        1D array-like of MATLAB datenums (float days).
    tz : Optional[str]
        IANA timezone name (e.g., "America/Los_Angeles"). Defaults to "UTC".

    Returns
    -------
    pd.Series of tz-aware timestamps (ns precision).
    """
    dn = np.asarray(dn).astype(float).ravel()
    out = []
    for d in dn:
        if np.isnan(d):
            out.append(pd.NaT)
            continue
        py_dt = datetime.fromordinal(int(d)) + timedelta(days=float(d) % 1) - timedelta(days=366)
        # note: we'll localize later via pandas tz_localize (safer for DST boundaries)
        out.append(pd.Timestamp(py_dt))
    s = pd.Series(out)
    if tz:
        s = s.dt.tz_localize("UTC").dt.tz_convert(tz) if s.dt.tz is None else s.dt.tz_convert(tz)
    return s


def _ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and 1 in x.shape:
        return x.reshape(-1)
    if x.ndim > 1:
        # flatten column-major (MATLAB-style)
        return x.reshape(-1, order="F")
    return x


def _safe_get(mat: Dict, key: str) -> Optional[np.ndarray]:
    return mat.get(key, None)


def _make_df(datetime_series: pd.Series, data: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame({"datetime": datetime_series})
    for name, arr in data.items():
        if arr is None:
            continue
        a = _ensure_1d(arr)
        if len(a) == len(df):
            df[name] = a
        else:
            # length mismatch; skip with notice
            # (we keep code silent here; caller can log if needed)
            pass
    return df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


# -----------------------------
# Importer
# -----------------------------

class ProcessedMATImporter(BaseImporter):
    """Importer for processed CATS PRH-style .mat files.

    Goal: mirror the existing CATS importer *behaviorally* where appropriate,
    but:
      • Anything that is a processed/estimated quantity (orientation, heading,
        speeds, jiggle, rotated sensor frames, etc.) is stored under
        `data_reader.derived_data` + `derived_info`.
      • Raw or near-raw environmental channels (depth p, temperature T,
        light) and GPS are stored under `sensor_data` + `sensor_info`.

    Expected variables in .mat (subset will be used if present):
      DN, Aw, Gw, Mw, At, Gt, Mt, pitch, roll, head, p, T, Light,
      GPS, speed, speedstats, JigRMS, flownoise, tagon, camon,
      vidDN, vidNam, vidDurs, viddeploy, INFO, fs
    """

    DERIVED_KEYS = {
        # orientation and rotated frames
        "Aw": ("acc_animal", ["ax", "ay", "az"], "g or m/s^2"),
        "Gw": ("gyr_animal", ["gx", "gy", "gz"], "deg/s or rad/s"),
        "Mw": ("mag_animal", ["mx", "my", "mz"], "uT"),
        # Euler angles
        "pitch": ("pitch", ["pitch"], "deg"),
        "roll": ("roll", ["roll"], "deg"),
        "head": ("heading", ["heading"], "deg"),
        # speed proxies & estimates
        "flownoise": ("speed_flownoise", ["speed_flownoise"], "m/s"),
        "JigRMS": ("jig_rms", ["jig_rms"], "g or m/s^2"),
        "speed": ("speed", ["speed"], "m/s"),
    }

    SENSOR_KEYS = {
        # environmental channels (treated as sensors)
        "p": ("depth", ["depth"], "m"),
        "T": ("temp", ["temp"], "degC"),
        "Light": ("light", ["light"], "arb"),
        # optionally expose raw triads if present (still sensors)
        "At": ("acc_raw", ["ax", "ay", "az"], "g or m/s^2"),
        "Gt": ("gyr_raw", ["gx", "gy", "gz"], "deg/s or rad/s"),
        "Mt": ("mag_raw", ["mx", "my", "mz"], "uT"),
        # GPS block (lat, lon, maybe time)
        "GPS": ("gps", ["lat", "lon"], "deg"),
    }

    EVENT_KEYS = {
        # event-like vectors or video metadata
        "tagon": "tagon",
        "camon": "camon",
        "vidDN": "video_start",
        "vidNam": "video_name",
        "vidDurs": "video_duration_s",
        "viddeploy": "video_deploy_meta",
    }

    def process_files(self, files: List[str], enforce_frequency: bool = True):
        """Process provided .mat files and register into the pickle structure.

        Notes
        -----
        • If multiple .mat files are passed, we load and merge based on DN.
        • Timezone is taken from `self.data_reader.deployment_info['Time Zone']`
          when available, else 'UTC'.
        • Sampling frequency `fs` in PRH files can either be scalar (common) or
          array per stream—here we record what we can in info blocks.
        """
        mat_files = [f for f in files if f.lower().endswith(".mat")]
        if not mat_files:
            print("⚠️ No .mat files provided to ProcessedMATImporter; skipping.")
            return None

        tz = self.data_reader.deployment_info.get("Time Zone", "UTC")

        # Load and concatenate by DN
        frames: List[pd.DataFrame] = []
        aux_blobs: List[Dict] = []  # INFO / speedstats etc per-file for metadata aggregation

        for fname in mat_files:
            path = os.path.join(self.data_reader.data_folder, fname)
            print(f"📥 Loading processed MAT: {path}")
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)

            # Pull DN first – must exist to build a chronological index
            DN = _safe_get(mat, "DN")
            if DN is None:
                print(f"❌ Skipping {fname}: 'DN' not found.")
                continue
            dt = _matlab_datenum_to_datetime(DN, tz)

            # Prepare per-file data frame with everything we might need later
            cols: Dict[str, np.ndarray] = {}

            # DERIVED (store later)
            for key in self.DERIVED_KEYS:
                if key in mat:
                    cols[key] = mat[key]

            # SENSORS (store later)
            for key in self.SENSOR_KEYS:
                if key in mat:
                    cols[key] = mat[key]

            # Also bring event/gps ancillary arrays to the temporary frame
            for key in self.EVENT_KEYS:
                if key in mat:
                    cols[key] = mat[key]

            # A lightweight union frame – single copy of DN as datetime
            df = _make_df(dt, cols)
            frames.append(df)

            # Stash info-like blocks for metadata
            blob = {
                "INFO": mat.get("INFO"),
                "speedstats": mat.get("speedstats"),
                "fs": mat.get("fs"),
                "source_file": fname,
            }
            aux_blobs.append(blob)

        if not frames:
            print("❌ No valid frames produced from .mat files.")
            return None

        # Merge all frames on datetime (outer to retain everything)
        all_df = pd.concat(frames, axis=0, ignore_index=True)
        all_df = all_df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

        # -----------------------------
        # Register SENSOR streams
        # -----------------------------
        for key, (sensor_name, channel_names, units) in self.SENSOR_KEYS.items():
            if key not in all_df.columns:
                continue
            arr = all_df[key]
            if arr.dtype == object and isinstance(arr.iloc[0], (np.ndarray, list)):
                # handle triads e.g., At, Gt, Mt shaped (N,3)
                tri = np.vstack(arr.values)
            else:
                tri = np.asarray(arr).reshape(-1, 1)

            if tri.ndim == 1:
                tri = tri[:, None]
            ncols = tri.shape[1]

            # Build DataFrame
            data_cols = {}
            for i in range(min(ncols, len(channel_names))):
                data_cols[channel_names[i]] = tri[:, i]
            sdf = pd.DataFrame({"datetime": all_df["datetime"], **data_cols})
            sdf = sdf.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

            if len(sdf.columns) == 1:  # no data columns made it
                continue

            # Store into sensor_data / sensor_info
            self.data_reader.sensor_data[sensor_name] = sdf
            self.data_reader.sensor_info[sensor_name] = {
                "channels": [c for c in sdf.columns if c != "datetime"],
                "metadata": {"source": "processed_mat", "mat_key": key},
                "sensor_start_datetime": sdf["datetime"].iloc[0],
                "sensor_end_datetime": sdf["datetime"].iloc[-1],
                "units": units,
                "sampling_frequency": None,  # unknown/variable; can infer later if desired
                "original_sampling_frequency": None,
                "logger_id": self.logger_id,
                "logger_manufacturer": self.logger_manufacturer,
                "processing_step": "Processed MAT import",
                "details": f"Imported from PRH processed file key '{key}'.",
            }
            print(f"📦 Sensor '{sensor_name}' stored with columns {self.data_reader.sensor_info[sensor_name]['channels']}")

        # -----------------------------
        # Register DERIVED streams
        # -----------------------------
        for key, (derived_name, channel_names, units) in self.DERIVED_KEYS.items():
            if key not in all_df.columns:
                continue
            arr = all_df[key]
            if arr.dtype == object and isinstance(arr.iloc[0], (np.ndarray, list)):
                tri = np.vstack(arr.values)
            else:
                tri = np.asarray(arr).reshape(-1, 1)
            if tri.ndim == 1:
                tri = tri[:, None]
            ncols = tri.shape[1]

            data_cols = {}
            for i in range(min(ncols, len(channel_names))):
                data_cols[channel_names[i]] = tri[:, i]
            ddf = pd.DataFrame({"datetime": all_df["datetime"], **data_cols})
            ddf = ddf.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            if len(ddf.columns) == 1:
                continue

            # Store
            self.data_reader.derived_data[derived_name] = ddf
            self.data_reader.derived_info[derived_name] = {
                "channels": [c for c in ddf.columns if c != "datetime"],
                "metadata": {"source": "processed_mat", "mat_key": key},
                "derived_start_datetime": ddf["datetime"].iloc[0],
                "derived_end_datetime": ddf["datetime"].iloc[-1],
                "units": units,
                "sampling_frequency": None,
                "logger_id": self.logger_id,
                "logger_manufacturer": self.logger_manufacturer,
                "processing_step": "Processed MAT import (derived)",
                "details": f"Imported derived stream from key '{key}'.",
            }
            print(f"🧮 Derived '{derived_name}' stored with columns {self.data_reader.derived_info[derived_name]['channels']}")

        # -----------------------------
        # Register EVENT-like / ancillary tables (optional): tagon/camon and video
        # -----------------------------
        events_cols = {k: v for k, v in self.EVENT_KEYS.items() if k in all_df.columns}
        if events_cols:
            ev = pd.DataFrame({"datetime": all_df["datetime"]})
            for k, out_name in events_cols.items():
                ev[out_name] = all_df[k]
            ev = ev.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            self.data_reader.event_data = ev
            self.data_reader.event_info = {
                "processing_step": "Processed MAT import (events)",
                "details": f"Registered event-like vectors: {list(events_cols.values())}",
            }
            print(f"🎬 Event table stored with columns: {list(events_cols.values())}")

        # -----------------------------
        # Aggregate INFO-like metadata from aux blobs
        # -----------------------------
        # We don’t attempt to fully re-serialize MATLAB structs; store minimally useful pieces.
        metainfo = {
            "source_files": [b.get("source_file") for b in aux_blobs],
            "fs": [b.get("fs") for b in aux_blobs if b.get("fs") is not None],
            "has_speedstats": any(b.get("speedstats") is not None for b in aux_blobs),
            "has_INFO": any(b.get("INFO") is not None for b in aux_blobs),
        }
        self.data_reader.logger_info[self.logger_id].update({
            "processed_mat_import": metainfo,
            "CellNum": "ProcessedMAT",  # marker akin to MainCATSprhTool cells
        })

        return {
            "sensors": list(self.data_reader.sensor_data.keys()),
            "derived": list(self.data_reader.derived_data.keys()),
            "events": list(events_cols.values()) if events_cols else [],
        }
