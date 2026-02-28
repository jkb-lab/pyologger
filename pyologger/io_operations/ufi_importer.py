from pyologger.io_operations.base_importer import BaseImporter
import os
import pytz
import struct
import pandas as pd
from datetime import datetime, timedelta
from pyologger.utils.time_manager import process_datetime
import re
from pathlib import Path
from functools import lru_cache

Hz = 100  # UFI ECG sampling rate in Hz
_UBC_RATE_PATTERN = re.compile(r"SmplRat\s+(\d+(?:\.\d+)?)\s*Hz", re.IGNORECASE)


def _read_text_lossy(p: Path) -> str:
    try:
        data = p.read_bytes()
    except Exception:
        return ""
    # Robust, lossy decoding: drop NULs and unknown bytes
    return data.replace(b"\x00", b"").decode("utf-8", errors="ignore")


@lru_cache(maxsize=256)
def parse_ubc_sampling_rate(ubc_path: str) -> int | None:
    """
    Parse sampling rate from a .ubc header.
    Looks for a line like: 'SmplRat  50 Hz              ECG'
    Returns an integer Hz, or None if not found.
    """
    p = Path(ubc_path)
    if not p.is_file():
        return None

    text = _read_text_lossy(p)
    if not text:
        return None

    m = _UBC_RATE_PATTERN.search(text)
    if not m:
        return None

    try:
        rate = float(m.group(1))
        # UFI typically uses integer Hz; round to nearest int
        return max(1, int(round(rate)))
    except Exception:
        return None


def _find_sibling_ubc(file_path: str, logger_id: str | None = None) -> Path | None:
    """
    Given a .ube/.ubf path, find the most likely .ubc file:
      1) Same stem, .ubc extension (preferred)
      2) Any .ubc in the same folder that contains logger_id
      3) Any .ubc in the same folder (fallback)
    """
    p = Path(file_path)
    cand = p.with_suffix(".ubc")
    if cand.is_file():
        return cand

    ubcs = sorted(p.parent.glob("*.ubc"))
    if logger_id:
        for u in ubcs:
            if logger_id in u.name:
                return u
    return ubcs[0] if ubcs else None


def resolve_sampling_rate_from_ubc(file_path: str, logger_id: str | None = None, default: int = Hz) -> tuple[int, bool]:
    """
    Resolve sampling rate for a .ube/.ubf by checking a nearby .ubc header.
    Returns (hz, from_ubc_flag).
    """
    ubc = _find_sibling_ubc(file_path, logger_id=logger_id)
    if not ubc:
        return default, False
    rate = parse_ubc_sampling_rate(str(ubc))
    if rate:
        return rate, True
    return default, False

class UFIImporter(BaseImporter):
    """Importer for UFI ECG loggers (.ube/.ubf high-rate cardiac tags)."""

    def process_files(self, files, enforce_frequency=True):
        """
        One-pass UFI pipeline:
          1. pick .ube/.ubf/.csv files for this logger
          2. parse them to raw dfs
          3. concat
          4. rename columns
          5. process datetime
          6. group channels into signals (writes into data_reader.signal_data/info)
          7. return standard 5-tuple
        """

        logger_id = self.logger_id
        tz_name = self.data_reader.deployment_info.get("Time Zone")

        print(f"🔄 TEST - processing UFI logger {logger_id} with {len(files)} file(s).")
        unique_files = list(dict.fromkeys(files))  # order-preserving dedupe

        # 1. keep only UFI-relevant extensions for THIS logger
        allowed_exts = (".ube", ".ubf", ".csv")
        selected_files = [
            f for f in unique_files
            if f.lower().endswith(allowed_exts) and logger_id in f
        ]

        # DEDUPE AGAIN just in case (paranoia + visibility)
        selected_files = list(dict.fromkeys(selected_files))

        print(f"[UFIImporter] {logger_id} selected_files after dedupe: {selected_files}")


        # 1. Pick only UFI files for this logger
        allowed_exts = (".ube", ".ubf", ".csv")
        selected_files = [
            f for f in files
            if f.lower().endswith(allowed_exts) and logger_id in f
        ]

        if not selected_files:
            print(f"⚠ No valid UFI files found for {logger_id} (UFI).")
            return None, None, None, None, None

        dfs = []

        # 2. Parse each selected file once
        for fname in selected_files:
            file_path = os.path.join(self.data_reader.data_folder, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in (".ube", ".ubf"):
                df_part = self._parse_ube_file(file_path)
            elif ext == ".csv":
                # some UFI devices export CSV already
                df_part = self.read_csv(file_path)
            else:
                print(f"⚠ Skipping unsupported UFI file extension: {fname}")
                continue

            if df_part is None or df_part.empty:
                print(f"⚠ Parsed empty data from {fname}, skipping.")
                continue

            dfs.append(df_part)
            print(f"UFI file for {logger_id}: {fname} - Successfully parsed.")

        if not dfs:
            print(f"⚠ Could not parse any usable UFI data for {logger_id}.")
            return None, None, None, None, None

        # 3. Concatenate all parts into one dataframe
        final_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        print(f"UFI logger {logger_id} combined shape: {final_df.shape}")

        # 4. Rename columns ONCE using mapping / montage
        original_cols = final_df.columns.tolist()
        new_col_names, channel_metadata = self.rename_channels(original_cols)
        final_df.rename(columns=new_col_names, inplace=True)
        print(f"✅ Renamed columns for {logger_id}: {new_col_names}")

        # 5. Standardize / localize / utc convert datetime ONCE
        final_df, datetime_metadata = process_datetime(
            final_df,
            time_zone=tz_name,
            channel_metadata=channel_metadata
        )

        # Store metadata for this logger in the DataReader
        self.data_reader.logger_info[self.logger_id]['datetime_created_from'] = datetime_metadata.get(
            'datetime_created_from', None
        )
        # fs in process_datetime() ends up as a scalar or None, but everywhere else you store it in a list.
        # We'll be consistent with CSVImporter: wrap in a list.
        self.data_reader.logger_info[self.logger_id]['fs'] = [datetime_metadata.get('fs', None)]

        # 6. Group data into signals ONCE (this populates signal_data / signal_info)
        signal_groups, signal_info = self.group_data_by_signals(
            final_df,
            self.logger_id,
            channel_metadata
        )

        # 7. Return standard 5-tuple exactly like CSVImporter does
        return final_df, channel_metadata, datetime_metadata, signal_groups, signal_info

    # ---------------------------------------------------------------------
    # Internal helper: parse one UBE/UBF file into a raw dataframe.
    # NO renaming here, NO process_datetime, NO grouping here.
    # Just return a DataFrame with columns like ['datetime', 'ecg'].
    # ---------------------------------------------------------------------

    def _parse_ube_file(self, file_path: str) -> pd.DataFrame:
        """
        Parse a single .ube/.ubf file into a pandas DataFrame with at least:
            'datetime' (tz-aware if possible)
            'ecg'      (raw integer samples)
        """

        print(f"Processing UBE file: {file_path}")
        
        # TODO: FIX THIS
        try:
            with open(file_path, 'rb') as file:
                ube_raw = file.read()

            # First 32 bytes: download timestamp (ASCII)
            dl_time_str = ube_raw[0:32].decode('utf-8').strip()
            print(f"Parsed download timestamp string: '{dl_time_str}'")

            try:
                dl_time = datetime.strptime(dl_time_str, "%m-%d-%Y, %H:%M:%S")
            except ValueError as e:
                print(f"Error parsing timestamp '{dl_time_str}': {e}")
                return pd.DataFrame()

            # Next bytes 32..36 (5 bytes): month/day/hour/min/sec at recording start
            record_month, record_day, record_hour, record_minute, record_second = struct.unpack(
                'BBBBB', ube_raw[32:37]
            )

            record_year = dl_time.year
            record_start = datetime(
                record_year,
                record_month,
                record_day,
                record_hour,
                record_minute,
                record_second
            )

            # If record_start > dl_time, assume it rolled over new year (logger still thinks it's next year)
            if record_start > dl_time:
                record_start = record_start.replace(year=record_year - 1)

            # Localize start time using deployment time zone, if available
            tz_name = self.data_reader.deployment_info.get('Time Zone')
            if tz_name:
                tz = pytz.timezone(tz_name)
                record_start = tz.localize(record_start)
                print(f"Deployment start time (localized): {record_start}")

            # Sanity-check vs declared deployment date
            deployment_date_str = self.data_reader.deployment_info.get('Deployment Date')
            if deployment_date_str is not None:
                rec_date = pd.to_datetime(deployment_date_str).date()
                if record_start.date() != rec_date:
                    print(f"⚠ Deployment start date {record_start.date()} "
                          f"does not match Deployment Date {rec_date}.")

            # Everything after byte 40 is interleaved channel/value pairs
            data_raw = ube_raw[40:]

            # ECG channel mask/ID
            ECG_CHAN_MASK = 0xF0
            ECG_CHAN_VAL  = 0x20  # upper nibble 0x2?
            ecg_data = []

            # Walk byte pairs [channel, value]
            # channel high 4 bits are channel ID, low 4 bits are high data bits
            # value byte is low 8 bits
            for i in range(0, len(data_raw), 2):
                if i + 1 >= len(data_raw):
                    break

                chan_byte = data_raw[i]
                val_byte  = data_raw[i + 1]

                if (chan_byte & ECG_CHAN_MASK) == ECG_CHAN_VAL:
                    # 12-bit sample = low 4 bits of chan_byte as MSBs + val_byte as LSBs
                    sample_val = ((chan_byte & 0x0F) << 8) | val_byte
                    ecg_data.append(sample_val)

            print(f"Total ECG data points: {len(ecg_data)}")

            if not ecg_data:
                print("⚠ No ECG data extracted from UBE file.")
                return pd.DataFrame()

            # --- NEW: resolve sampling rate via sibling .ubc (fallback to default Hz) ---
            sr, sr_from_ubc = resolve_sampling_rate_from_ubc(file_path, logger_id=self.logger_id, default=Hz)
            if sr_from_ubc:
                print(f"✅ Sampling rate parsed from .ubc: {sr} Hz")
            else:
                print(f"ℹ Using default sampling rate: {sr} Hz (no .ubc override found)")

            # Generate timestamps at detected sampling rate
            ecg_time = [
                record_start + timedelta(seconds=i / float(sr))
                for i in range(len(ecg_data))
            ]

            df = pd.DataFrame({
                'datetime': ecg_time,
                'ecg': ecg_data
            })

            # Attach provenance + fs
            df.attrs['created'] = dl_time
            df.attrs['fs'] = sr  # <- handy downstream

            return df

        except Exception as e:
            print(f"❌ Error processing UBE file {file_path}: {e}")
            return pd.DataFrame()
