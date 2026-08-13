import pandas as pd
import numpy as np
import re
import warnings
import json
import pytz
from datetime import datetime
from pyologger.utils.time_manager import *

class BaseImporter:
    """Base class for handling manufacturer-specific processing."""

    def __init__(self, data_reader, logger_id):
        self.data_reader = data_reader
        self.logger_id = logger_id
        self.logger_manufacturer = self.data_reader.logger_info[logger_id]['Manufacturer']
        self.montage_id = self.data_reader.logger_info[logger_id]['Montage ID']
        self.expected_frequencies = {}  # Stores expected signal frequencies from .txt files
        self.montage_path = self.data_reader.montage_path
        # Optional montage-level "__files__" filter; empty means "read every file".
        self.file_include_patterns = []

        # Load the custom JSON mapping for column names if available
        if self.montage_path:
            self.load_custom_mapping()

    def get_file_include_patterns(self):
        """Filename substrings this montage should ingest ([] = no filtering)."""
        return list(getattr(self, "file_include_patterns", []) or [])

    # Default import rates per signal type (Hz). Applied only when the source rate is
    # higher; a signal type absent here keeps its native rate. Override per deployment
    # via parameter_log.json ("edf_target_frequencies" in the "settings" section), or
    # dataset-wide via the config.yaml `edf_import.target_frequencies` block.
    DEFAULT_EDF_TARGET_FREQUENCIES = {
        "eeg": 100,
        "eog": 100,
        "emg": 100,
        "ecg": 250,
    }

    # Signals that pyologger recomputes from raw data later in the pipeline. When a
    # "derived" logger (e.g. NL-D1, montage juv-nese-sleep-derived) supplies one of
    # these, it is imported under a "<signal>_2" name so the canonical name stays free
    # for pyologger's own, more precise version. Labels and analysis products that
    # pyologger does not recompute (sleep_state, eeg_*_analysis, ...) keep their names.
    DERIVED_SIGNAL_SUFFIX = "_2"
    RECOMPUTED_SIGNALS = {
        "stroke_rate",
        "heart_rate",
        "depth",
        "prh",
        "velocity",
        "position",
        "location",
    }

    def is_derived_logger(self):
        """True if this logger carries previously-derived products, not raw sensor data."""
        montage_id = str(getattr(self, "montage_id", "") or "").strip().lower()
        return "derived" in montage_id

    def resolve_signal_name(self, signal_name):
        """Map a signal to its stored name, suffixing derived duplicates."""
        if signal_name in self.RECOMPUTED_SIGNALS and self.is_derived_logger():
            return f"{signal_name}{self.DERIVED_SIGNAL_SUFFIX}"
        return signal_name

    def get_edf_target_frequencies(self):
        """Per-signal-type decimation targets for EDF import, as {parent_signal: target_hz}.

        Resolution order (later wins): class defaults → config.yaml
        `edf_import.target_frequencies` → parameter_log.json `settings.edf_target_frequencies`
        (which itself merges dataset defaults under the deployment entry).

        Set a value to null/None to disable decimation for that signal type.
        """
        resolved = dict(self.DEFAULT_EDF_TARGET_FREQUENCIES)

        config = getattr(self.data_reader, "config", None) or {}
        layers = [(config.get("edf_import") or {}).get("target_frequencies") or {}]

        param_manager = getattr(self.data_reader, "param_manager", None)
        if param_manager is not None:
            try:
                override = param_manager.get_from_config(
                    ["edf_target_frequencies"], section="settings"
                ).get("edf_target_frequencies")
                if isinstance(override, dict):
                    layers.append(override)
            except Exception as exc:
                print(f"⚠️ Could not read edf_target_frequencies from parameter log ({exc}).")

        for layer in layers:
            for signal_type, target_hz in layer.items():
                key = str(signal_type).strip().lower()
                if target_hz is None:
                    resolved.pop(key, None)
                    continue
                try:
                    resolved[key] = float(target_hz)
                except (TypeError, ValueError):
                    print(f"⚠️ Ignoring malformed edf target frequency: {signal_type!r}: {target_hz!r}")
        return resolved

    def _edf_decimation_step(self, signal_label, source_hz, targets, channel_metadata):
        """Integer decimation factor for one signal, or 1 to keep its native rate.

        Only integer factors are used: EDF sources here are integer-or-repeating rates,
        and a non-integer factor would force a resample that downstream code (which
        does `int(logger_info['fs'])`) cannot represent.
        """
        meta = (channel_metadata or {}).get(signal_label) or {}
        parent = str(meta.get("parent_signal", "")).strip().lower()
        target_hz = targets.get(parent)
        if not target_hz or target_hz >= source_hz:
            return 1

        step = int(round(source_hz / target_hz))
        if step < 2:
            return 1
        achieved = source_hz / step
        if not np.isclose(achieved, round(achieved)):
            print(
                f"⚠️ {signal_label}: {source_hz} Hz → {target_hz} Hz would give a "
                f"non-integer rate ({achieved:.4f} Hz); keeping native rate."
            )
            return 1
        return step

    def group_edf_signals_by_frequency(
        self, signals, startdate, starttime, time_zone="UTC", skip_full=False,
        channel_metadata=None,
    ):
        """Group EDF signals into one DataFrame per resulting sampling frequency.

        Signals are keyed by the rate they end up at, not the rate they were recorded
        at. Signal types with a decimation target (see `get_edf_target_frequencies`)
        are anti-alias filtered and decimated with `scipy.signal.decimate`, so a single
        source group can split into several output frames — e.g. a 500 Hz group holding
        both ECG and EEG becomes a 250 Hz frame and a 100 Hz frame. Related channels
        that share a resulting rate stay together in one matrix.

        Memory-conscious: `EdfSignal.data` is a property that re-materializes a full
        float64 array on every access, so it is read exactly once per signal and
        accumulated into a preallocated float32 matrix. Peak usage is roughly one
        output matrix plus one signal's float64 temporary, rather than every signal's
        float64 array at once.
        """
        from math import floor

        targets = self.get_edf_target_frequencies()

        # Bucket by (resulting_rate, decimation_step) so channels that end up at the
        # same rate share a matrix.
        buckets = {}
        for signal in signals:
            source_hz = signal.__dict__.get("_sampling_frequency")
            step = self._edf_decimation_step(signal.label, source_hz, targets, channel_metadata)
            buckets.setdefault((source_hz / step, step, source_hz), []).append(signal)

        # edfio reads each signal with a separate strided gather over the whole data-record
        # buffer, which costs a full pass per signal. Signals recorded at the same rate sit
        # in adjacent column spans, so one bulk read fills them all in a single pass.
        self._prefetch_edf_signals(signals)

        grouped_data = {}
        for (out_freq, step, source_hz), sigs in sorted(buckets.items(), key=lambda kv: -kv[0][0]):
            labels = [s.label for s in sigs]
            if step > 1:
                print(
                    f"\n🔍 Grouping {len(sigs)} signals at {source_hz} Hz → "
                    f"{out_freq:g} Hz (factor {step}, anti-aliased): {', '.join(labels)}"
                )
            else:
                print(f"\n🔍 Grouping {len(sigs)} signals at {source_hz} Hz: {', '.join(labels)}")

            if skip_full:
                preview = {s.label: np.asarray(s.data[:10], dtype=np.float32) for s in sigs}
                preview_df = pd.DataFrame(preview)
                preview_df["datetime"] = self._edf_timestamps(10, out_freq, startdate, starttime, time_zone)
                grouped_data[out_freq] = preview_df.reset_index(drop=True)
                print(f"⚡ Skipping full data for {out_freq} Hz — using preview only")
                continue

            # len(digital) reads the int16 buffer, which is 4x smaller than float64 data.
            n_full = len(sigs[0].digital)
            n_out = -(-n_full // step) if step > 1 else n_full

            arr = np.empty((n_out, len(sigs)), dtype=np.float32)
            for i, signal in enumerate(sigs):
                print(f"   [{i + 1}/{len(sigs)}] {signal.label}", flush=True)
                column = self._edf_signal_column(signal, step, n_out)
                # Signals in a group are nominally the same length, but guard against
                # off-by-one differences (e.g. decimate's output length convention).
                if column.size < n_out:
                    arr[: column.size, i] = column
                    arr[column.size:, i] = np.nan
                else:
                    arr[:, i] = column[:n_out]
                del column

            df = pd.DataFrame(arr, columns=labels, copy=False)
            df["datetime"] = self._edf_timestamps(n_out, out_freq, startdate, starttime, time_zone)

            int_out_freq = floor(out_freq)
            if not np.isclose(out_freq, int_out_freq):
                print(f"⏬ Resampling from {out_freq:.4f} Hz to {int_out_freq} Hz...")
                df = self.resample_mean_dataframe(df, int_out_freq)
            if int_out_freq in grouped_data:
                # Two source rates landed on the same integer rate; merge on datetime
                # rather than silently dropping one.
                existing = grouped_data[int_out_freq]
                df = existing.merge(df, on="datetime", how="outer")
            grouped_data[int_out_freq] = df.reset_index(drop=True)

        return grouped_data

    def resample_mean_dataframe(self, df, target_freq_hz):
        """Resample to an integer rate by averaging. Subclasses may override."""
        df = df.set_index('datetime')
        target_period_ms = int(round(1000 / target_freq_hz))
        return df.resample(f"{target_period_ms}ms").mean().reset_index()

    @staticmethod
    def _prefetch_edf_signals(signals):
        """Populate `_digital` for lazily-loaded EDF signals using bulk reads.

        `edfio.LazyLoader.load()` slices one signal's columns out of the shared
        data-record buffer, so reading N signals means N strided passes over the whole
        file (~74 s each on a 12 GB EDF). Signals sharing a sampling rate occupy
        adjacent column spans, so a single contiguous read covers the whole run and is
        an order of magnitude faster.

        Falls back silently to edfio's per-signal path if the layout is not as expected,
        so behaviour is unchanged when this optimization does not apply.
        """
        pending = [
            s for s in signals
            if getattr(s, "_digital", None) is None and getattr(s, "_lazy_loader", None) is not None
        ]
        if len(pending) < 2:
            return

        # Group by the shared buffer; a deployment may mix loggers/files.
        by_buffer = {}
        for signal in pending:
            by_buffer.setdefault(id(signal._lazy_loader.buffer), []).append(signal)

        for group in by_buffer.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda s: s._lazy_loader.start_sample)
            buffer = group[0]._lazy_loader.buffer

            # Split into maximal contiguous column runs.
            runs, run = [], [group[0]]
            for signal in group[1:]:
                if signal._lazy_loader.start_sample == run[-1]._lazy_loader.end_sample:
                    run.append(signal)
                else:
                    runs.append(run)
                    run = [signal]
            runs.append(run)

            for run in runs:
                if len(run) < 2:
                    continue
                lo = run[0]._lazy_loader.start_sample
                hi = run[-1]._lazy_loader.end_sample
                n_records = buffer.shape[0]
                approx_gb = n_records * (hi - lo) * buffer.dtype.itemsize / 1e9
                print(
                    f"⚡ Bulk-reading {len(run)} signals in one pass ({approx_gb:.2f} GB)...",
                    flush=True,
                )
                try:
                    # Copy in row chunks rather than one strided `ascontiguousarray`
                    # over the whole buffer: same result, but far better page/cache
                    # locality (~1.6x faster on a 12 GB EDF).
                    block = np.empty((n_records, hi - lo), dtype=buffer.dtype)
                    chunk = 8192
                    for start in range(0, n_records, chunk):
                        stop = min(start + chunk, n_records)
                        block[start:stop] = buffer[start:stop, lo:hi]
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"⚠️ Bulk EDF read failed ({exc}); using per-signal reads.")
                    continue
                for signal in run:
                    loader = signal._lazy_loader
                    start = loader.start_sample - lo
                    end = loader.end_sample - lo
                    signal._digital = block[:, start:end].flatten()
                    signal._lazy_loader = None
                del block

    @staticmethod
    def _edf_signal_column(signal, step, n_out):
        """Return one signal as float32, decimated by `step` with anti-aliasing."""
        data = signal.data
        if step == 1:
            return np.asarray(data, dtype=np.float32)

        from scipy.signal import decimate

        # decimate() applies an anti-alias filter before downsampling; plain slicing
        # would fold content above the new Nyquist back into the retained band.
        try:
            reduced = decimate(data, step, ftype="fir", zero_phase=True)
        except ValueError as exc:
            # Very short signals can trip the filter's padding requirements.
            print(f"⚠️ Anti-aliased decimation failed for '{signal.label}' ({exc}); falling back to slicing.")
            reduced = data[::step]
        return np.asarray(reduced[:n_out], dtype=np.float32)

    @staticmethod
    def _edf_timestamps(n_samples, sampling_frequency, startdate, starttime, time_zone="UTC"):
        """Build the datetime index for an EDF frequency group."""
        start_time = pd.to_datetime(f"{startdate} {starttime}").tz_localize(time_zone)
        offsets = np.arange(n_samples) / sampling_frequency
        return start_time + pd.to_timedelta(offsets, unit="s")

    @staticmethod
    def normalize_channel_key(value):
        """Normalize channel identifiers so importer labels and JSON keys compare reliably."""
        normalized = re.sub(r"[^\w]", "", str(value).strip().lower().replace(" ", ""))
        if normalized.startswith("magn"):
            normalized = "mag" + normalized[4:]
        return normalized
        
    def read_csv(self, csv_path):
        """Reads a CSV file with multiple encoding attempts."""
        # First, check if this is a Vectronics file with metadata header
        skiprows = 0
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
        except UnicodeDecodeError:
            with open(csv_path, 'r', encoding='ISO-8859-1') as f:
                first_line = f.readline().strip()
        
        # Detect Vectronics metadata row
        if (first_line.startswith("Collar:") or 
            first_line.startswith("DeviceID:") or 
            "Sensor range" in first_line):
            skiprows = 1
            print(f"   🔪 Detected Vectronics metadata row - skipping first row")
        
        # Try reading with different encodings
        encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                print(f"Attempting to read {csv_path} with encoding {encoding}")
                return pd.read_csv(csv_path, encoding=encoding, skiprows=skiprows)
            except UnicodeDecodeError as e:
                print(f"Error reading {csv_path} with encoding {encoding}: {e}")
        raise UnicodeDecodeError(f"Failed to read {csv_path} with available encodings.")

    def _read_file(self, path):
        """Dispatch to CSV or Parquet reader based on extension."""
        low = path.lower()
        if low.endswith(('.csv',)):
            # Uses BaseImporter.read_csv() (handles messy headers, comments, etc.)
            return self.read_csv(path)
        elif low.endswith(('.parquet', '.parq', '.pq')):
            return self._read_parquet(path)
        raise ValueError(f"Unsupported file type for: {path}")

    def _read_parquet(self, path):
        """Parquet reader with a safe engine fallback."""
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception:
            # Fallback to fastparquet if pyarrow isn't available
            return pd.read_parquet(path, engine="fastparquet")

    def import_netcdf(data_reader, filepath):
            """Imports a NetCDF file to the pickle format used by pyologger."""
            
            print(f"NetCDF file imported from {filepath} and returning pickle object.")

    def import_edf(data_reader, filepath):
            """Imports a NetCDF file to the pickle format used by pyologger."""
            
            print(f"NetCDF file imported from {filepath} and returning pickle object.")
    
    def load_custom_mapping(self):
        """Loads custom JSON mapping for column names based on manufacturer and montage ID."""
        try:
            with open(self.montage_path, 'r') as json_file:
                full_mapping = json.load(json_file)
                print(f"Custom column mapping loaded from {self.montage_path}")

                # Ensure it's a dictionary
                if not isinstance(full_mapping, dict):
                    raise ValueError("Column mapping JSON is not a dictionary.")

                # Validate manufacturer
                manufacturer = self.logger_manufacturer
                if manufacturer not in full_mapping:
                    raise ValueError(f"Manufacturer '{manufacturer}' not found in column mapping.")

                # Validate montage ID
                montage_id = self.montage_id
                montage_id_text = str(montage_id).strip().lower() if montage_id is not None else ""
                if pd.isna(montage_id) or montage_id_text in {"", "nan", "none"}:
                    print(
                        f"No valid montage ID for logger '{self.logger_id}' "
                        f"({manufacturer}); proceeding without custom mapping."
                    )
                    self.montage = None
                    return

                manufacturer_mappings = full_mapping[manufacturer]
                resolved_montage_id = montage_id
                if montage_id not in manufacturer_mappings:
                    case_insensitive_match = next(
                        (candidate for candidate in manufacturer_mappings if str(candidate).strip().lower() == montage_id_text),
                        None,
                    )
                    if case_insensitive_match is not None:
                        resolved_montage_id = case_insensitive_match
                    elif len(manufacturer_mappings) == 1:
                        resolved_montage_id = next(iter(manufacturer_mappings))
                        print(
                            f"⚠️ Montage ID '{montage_id}' not found under manufacturer '{manufacturer}'. "
                            f"Falling back to the only available montage '{resolved_montage_id}'."
                        )
                        self.data_reader.logger_info[self.logger_id]["Requested Montage ID"] = montage_id
                        self.data_reader.logger_info[self.logger_id]["Montage ID"] = resolved_montage_id
                        self.montage_id = resolved_montage_id
                    else:
                        available = sorted(str(candidate) for candidate in manufacturer_mappings.keys())
                        raise ValueError(
                            f"Montage ID '{montage_id}' not found under manufacturer '{manufacturer}'. "
                            f"Available montages: {available}"
                        )

                # Extract only the relevant part of the mapping
                self.montage = manufacturer_mappings[resolved_montage_id]
                # "__files__" is montage-level configuration, not a channel. Pull it
                # out so channel iteration/validation never sees it as a signal.
                if isinstance(self.montage, dict) and "__files__" in self.montage:
                    self.montage = {
                        key: value for key, value in self.montage.items()
                        if key != "__files__"
                    }
                    self.file_include_patterns = list(
                        manufacturer_mappings[resolved_montage_id]["__files__"] or []
                    )
                    print(f"Montage file filter: {self.file_include_patterns}")
                print(f"Column mapping loaded for manufacturer '{manufacturer}', montage '{resolved_montage_id}'.")
                print(f"Mapping content {self.montage}.")
        except FileNotFoundError:
            print(f"Custom mapping file not found at {self.montage_path}. Proceeding without it.")
            self.montage = None
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading or verifying JSON from {self.montage_path}: {e}")
            self.montage = None

    def rename_channels(self, channel_names):
        """Maps original channel IDs to standardized channel IDs."""
        channel_metadata = {}
        new_channels = {}
        used_names = {}
        name_parent_owner = {}

        def _slug(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

        def ensure_unique_name(candidate_name: str, source_column: str, parent_signal: str) -> str:
            """
            Guarantee that the standardized channel name we emit is unique.

            If another column already claimed the same standardized name,
            append a suffix so pandas doesn't create duplicate column labels.
            """
            count = used_names.get(candidate_name, 0)
            if count == 0:
                used_names[candidate_name] = 1
                name_parent_owner[candidate_name] = parent_signal
                return candidate_name

            # Prefer a deterministic parent-signal suffix when collision is across
            # different parent signals (e.g., ax in accelerometer vs corrected_acc).
            owner_parent = name_parent_owner.get(candidate_name)
            parent_slug = _slug(parent_signal) or "extra"
            owner_slug = _slug(owner_parent) if owner_parent else None

            if owner_slug and owner_slug != parent_slug:
                preferred_name = f"{candidate_name}__{parent_slug}"
                if preferred_name not in used_names:
                    used_names[preferred_name] = 1
                    return preferred_name

            unique_name = f"{candidate_name}__dup{count}"
            used_names[candidate_name] = count + 1
            print(
                f"⚠️ Duplicate standardized name '{candidate_name}' detected for '{source_column}'. "
                f"Using unique column id '{unique_name}'."
            )
            return unique_name

        print(f"🔄 Renaming channels for {self.logger_manufacturer} logger with montage ID {self.montage_id}.")

        # Be defensive: self.montage may be None if no mapping file was provided or it failed to load.
        if not getattr(self, 'montage', None):
            print(f"⚠️ No montage mapping loaded for logger {self.logger_id} ({self.logger_manufacturer}). Falling back to identity mapping.")
            self.montage = {}
        else:
            try:
                print(f"📜 Available mapping keys: {list(self.montage.keys())}")  # Show available keys in the mapping
            except Exception:
                # Defensive fallback if montage is not a dict-like object
                print("⚠️ Montage mapping is not iterable. Falling back to empty mapping.")
                self.montage = {}

        for original_name in channel_names:
            # Take out spaces from channel names and make lowercase
            clean_name = original_name.strip().lower().replace(" ", "_").replace(".", "")  # Normalize

            # Extract unit (if present)
            unit = None
            if "[" in clean_name and "]" in clean_name:
                name, unit = clean_name.split("[", 1)
                unit = unit.replace("]", "").strip().lower()
                clean_name = name.strip("_")

            if "(" in clean_name and ")" in clean_name:
                name, unit = clean_name.split("(", 1)
                unit = unit.replace(")", "").strip().lower()
                clean_name = f"{name.strip('_')}_{unit}"

            # Ensure local/UTC times are distinct
            if "local" in original_name.lower() and "local" not in clean_name:
                clean_name = f"{clean_name}_local"
            elif "utc" in original_name.lower() and "utc" not in clean_name:
                clean_name = f"{clean_name}_utc"

            # Dictionary lookup: if no mapping entry exists, fall back to using the cleaned name
            if clean_name in self.montage:
                mapping_info = self.montage[clean_name] or {}
                print(f"🎯 Found match in montage: {mapping_info}")  # Show full mapping info
            else:
                mapping_info = {}
                print(f"ℹ️ No mapping for '{clean_name}' — using cleaned name as standardized id.")

            mapped_name = mapping_info.get("standardized_channel_id", clean_name)
            parent_signal_raw = mapping_info.get("parent_signal")
            parent_signal = (parent_signal_raw if parent_signal_raw else "extra")
            parent_signal = str(parent_signal).strip().lower()
            unique_mapped_name = ensure_unique_name(mapped_name, original_name, parent_signal)

            original_unit_raw = mapping_info.get("original_unit")
            original_unit = (original_unit_raw if original_unit_raw else "unknown")
            original_unit = str(original_unit).strip().lower()

            standardized_unit_raw = mapping_info.get("standardized_unit")
            standardized_unit = (standardized_unit_raw if standardized_unit_raw else "unknown")
            standardized_unit = str(standardized_unit).strip()

            if unique_mapped_name != mapped_name:
                print(f"   ↳ Reassigned to unique standardized name: {unique_mapped_name}")

            print(f"📝 Dictionary: original name: {original_name} → standardized name: {unique_mapped_name} (Signal type: {parent_signal})")

            channel_metadata[unique_mapped_name] = {
                "original_name": original_name,
                "unit": original_unit or "unknown",
                "standardized_unit": standardized_unit or "unknown",
                "parent_signal": parent_signal
            }
            new_channels[original_name] = unique_mapped_name
        print(f"🐻Channel metadata: {channel_metadata}")
        print(f"✅ Final renamed channels: {new_channels}")
        return new_channels, channel_metadata


    # Standard gravity (CODATA / ISO 80000-3), used to standardize accelerometer
    # channels recorded in g to the m/s^2 target unit.
    STANDARD_GRAVITY_MS2 = 9.80665

    # Accelerometer parent signals whose channels are stored in physical
    # acceleration units and therefore participate in g -> m/s^2 conversion.
    _ACCEL_PARENT_SIGNALS = ("accelerometer", "corrected_acc", "dynamic_accel", "calibrated_acc")

    @staticmethod
    def _is_g_unit(unit) -> bool:
        """True for units meaning 'multiples of standard gravity'."""
        text = str(unit or "").strip().lower()
        return text in {"g", "gs", "g-force", "gforce", "gravity"}

    @staticmethod
    def _is_ms2_unit(unit) -> bool:
        text = str(unit or "").strip().lower().replace(" ", "")
        return text in {"m/s^2", "m/s2", "m/s²", "ms^-2", "m·s^-2"}

    def convert_acceleration_to_standard_unit(self, df, channel_metadata):
        """
        Standardize accelerometer channels to their declared standardized_unit.

        Mirrors how the pressure pipeline standardizes depth to metres: the
        montage declares the target unit, and the value is actually converted
        rather than merely relabelled. Without this, `standardized_unit` is just
        a claim, and stored values can be off by ~9.81x from what it says.

        The project standard is **g**, matching the convention that ODBA/VeDBA
        are reported in g. Conversion runs in whichever direction the montage
        requires (g -> m/s^2 or m/s^2 -> g), and is a no-op when the source
        already matches the target, so the transform is idempotent.
        """
        if not channel_metadata:
            return df

        updated = df
        converted = []
        for column_name, metadata in channel_metadata.items():
            if not isinstance(metadata, dict):
                continue
            if metadata.get("parent_signal") not in self._ACCEL_PARENT_SIGNALS:
                continue
            if column_name not in getattr(updated, "columns", []):
                continue

            raw_unit = metadata.get("unit")
            target_unit = metadata.get("standardized_unit")

            if self._is_g_unit(raw_unit) and self._is_ms2_unit(target_unit):
                factor, achieved = self.STANDARD_GRAVITY_MS2, "m/s^2"
            elif self._is_ms2_unit(raw_unit) and self._is_g_unit(target_unit):
                factor, achieved = 1.0 / self.STANDARD_GRAVITY_MS2, "g"
            else:
                continue

            if updated is df:
                updated = updated.copy()
            updated[column_name] = (
                pd.to_numeric(updated[column_name], errors="coerce") * factor
            )
            # Record the achieved unit so a second pass is a no-op.
            metadata["unit"] = achieved
            channel_metadata[column_name] = metadata
            converted.append((column_name, achieved))

        if converted:
            for achieved in sorted({unit for _, unit in converted}):
                cols = [name for name, unit in converted if unit == achieved]
                print(f"🔁 Standardized acceleration to {achieved} for column(s): {cols}")
        return updated

    def group_data_by_signals(self, df, logger_id, channel_metadata):
        """Groups data columns to signals and downsamples based on expected frequencies."""
        signal_groups = {}
        signal_info = {}

        def _norm_col(value):
            return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

        # Build robust column lookup once (handles case, whitespace, punctuation variants).
        df_cols = list(df.columns)
        df_lookup = {}
        for c in df_cols:
            c_str = str(c)
            candidates = {
                c_str,
                c_str.strip(),
                c_str.lower(),
                c_str.strip().lower(),
                _norm_col(c_str),
            }
            for key in candidates:
                if key and key not in df_lookup:
                    df_lookup[key] = c

        print("16a")
        signal_names = set()
        print(f"Channel metadata received for grouping: {channel_metadata}")
        for v in channel_metadata.values():
            try:
                signal_names.add(v['parent_signal'].strip().lower())
            except Exception as e:
                print(f"⚠️ Skipping invalid channel metadata entry: {v}. Error: {e}")

        for signal_name in signal_names:

            print("16b")
            if signal_name == 'extra':
                continue  # Skip 'extra' signal type

            # Derived loggers store recomputed signals under a "_2" name so pyologger's
            # own version can claim the canonical one later in the pipeline.
            stored_signal_name = self.resolve_signal_name(signal_name)

            if stored_signal_name in self.data_reader.signal_data:
                alt_name = f"{stored_signal_name}_2"
                if alt_name not in self.data_reader.signal_data:
                    print(f"Sensor '{stored_signal_name}' already claimed; storing this logger's version as '{alt_name}'.")
                    stored_signal_name = alt_name
                else:
                    print(f"Sensor '{stored_signal_name}' (and '{alt_name}') already processed. Skipping.")
                    continue


            print("16c")
            # Group columns by signal and normalize channel ids back to base names.
            signal_orig_cols = []
            resolved_meta_keys = {}
            for std_col, meta in channel_metadata.items():
                if meta['parent_signal'].strip().lower() != signal_name:
                    continue
                # Prefer standardized channel id first; then fallback to original source name.
                source_name = meta.get("original_name")
                probes = [
                    std_col,
                    str(std_col).strip(),
                    str(std_col).lower(),
                    _norm_col(std_col),
                    source_name,
                    str(source_name).strip() if source_name is not None else None,
                    str(source_name).lower() if source_name is not None else None,
                    _norm_col(source_name) if source_name is not None else None,
                ]
                resolved = None
                for p in probes:
                    if not p:
                        continue
                    if p in df_lookup:
                        resolved = df_lookup[p]
                        break
                if resolved is not None and resolved not in signal_orig_cols:
                    signal_orig_cols.append(resolved)
                    resolved_meta_keys[resolved] = std_col
            if not signal_orig_cols:
                expected = [
                    str(col) for col, meta in channel_metadata.items()
                    if meta['parent_signal'].strip().lower() == signal_name
                ]
                print(
                    f"⚠️ No columns found for signal '{signal_name}'. "
                    f"Expected one of {expected}. Available columns: {list(df.columns)}. Skipping."
                )
                continue
            print("16d")
            signal_df = df[['datetime'] + signal_orig_cols].copy()

            # Strip importer uniqueness suffixes (e.g., gy__corrected_gyr -> gy)
            # at the per-signal level so users interact with canonical channel IDs.
            rename_map = {}
            signal_channel_metadata = {}
            used_base_names = set()
            for orig_col in signal_orig_cols:
                meta_key = resolved_meta_keys.get(orig_col, orig_col)
                if meta_key not in channel_metadata:
                    # Fallback: try normalized lookup against channel_metadata keys.
                    norm_target = _norm_col(meta_key)
                    alt_key = next(
                        (k for k in channel_metadata.keys() if _norm_col(k) == norm_target),
                        None,
                    )
                    meta_key = alt_key if alt_key is not None else meta_key
                if meta_key not in channel_metadata:
                    print(
                        f"⚠️ Missing channel metadata for '{orig_col}' "
                        f"(resolved key '{meta_key}') in signal '{signal_name}'. Skipping this channel."
                    )
                    continue
                # Use the standardized metadata key as the canonical channel id.
                base_name = str(meta_key).split("__", 1)[0]
                target_name = base_name
                if target_name in used_base_names:
                    # Keep this deterministic and safe if a true same-signal collision occurs.
                    i = 2
                    while f"{base_name}__dup{i}" in used_base_names:
                        i += 1
                    target_name = f"{base_name}__dup{i}"
                used_base_names.add(target_name)
                rename_map[orig_col] = target_name
                signal_channel_metadata[target_name] = channel_metadata[meta_key]

            signal_df.rename(columns=rename_map, inplace=True)
            signal_cols = [rename_map[c] for c in signal_orig_cols if c in rename_map]
            if any(str(c).find("__") >= 0 for c in signal_orig_cols):
                print(
                    f"[group_data_by_signals] '{signal_name}' channel normalization: "
                    f"{signal_orig_cols} -> {signal_cols}"
                )

            if not signal_cols:
                print(f"⚠️ No valid channels remained for signal '{signal_name}' after metadata resolution. Skipping.")
                continue

            print("16e")
            # Check if all signal columns are numeric
            non_numeric_cols = signal_df[signal_cols].select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"❌ Skipping signal '{signal_name}': non-numeric data found in columns: {non_numeric_cols}")
                continue

            print("16f")
            # Determine the data type of the signal columns
            data_type = signal_df[signal_cols].dtypes.iloc[0]
            data_type_str = str(data_type)

            print("16g")
            # Standardized metadata collection
            start_time = signal_df['datetime'].iloc[0]
            end_time = signal_df['datetime'].iloc[-1]
            max_value = signal_df[signal_cols].max().max()
            min_value = signal_df[signal_cols].min().min()
            mean_value = signal_df[signal_cols].mean().mean()

            print("16h")
            # Get the original unit from the column metadata
            original_units = {signal_channel_metadata[col]['unit'] for col in signal_cols}
            if len(original_units) > 1:
                warnings.warn(f"Conflicting units found for signal '{signal_name}': {original_units}. Using the first one.")
            original_unit = original_units.pop() if original_units else "unknown"

            print("16i")
            # Estimate cadence from the full sorted datetime series.
            original_frequency = calculate_sampling_frequency(signal_df['datetime'])
            print(f"Original frequency for {signal_name}: {original_frequency} Hz")

            print("16j")
            expected_frequency = self.expected_frequencies.get(signal_name)
            if not expected_frequency and self.logger_manufacturer == 'LL':
                expected_frequency = int(self.data_reader.logger_info[logger_id]['fs'])

            print("16k")
            max_desired_frequency = None
            if self.logger_manufacturer in ['Evolocus', 'Manitty', 'UFI']:
                max_freq_lookup = {'eeg': 100, 'eog': 100, 'ecg': 250, 'emg': 250}
                max_desired_frequency = max_freq_lookup.get(signal_name, None)

            print("16l")
            downsample_target = max_desired_frequency or expected_frequency
            if not expected_frequency and not max_desired_frequency:
                print(f"⚠️ No frequency target found for {signal_name}. Using original frequency {original_frequency} Hz.")

            print("16m")
            if downsample_target and downsample_target < original_frequency:
                decimation_factor = max(1, int(round(original_frequency / downsample_target)))
                print(f"Downsampling {signal_name} by {decimation_factor}x from {original_frequency:.2f}Hz to {downsample_target:.2f}Hz.")
                signal_df = signal_df.iloc[::decimation_factor]
                new_frequency = calculate_sampling_frequency(signal_df['datetime'])
                print(f"New frequency after downsampling: {new_frequency} Hz")
            else:
                new_frequency = original_frequency
                print(f"No downsampling required for {signal_name}. Current: {original_frequency:.2f}Hz, Target: {downsample_target}Hz")

            print("16n")
            details = 'Initial, raw signal-specific data and metadata loaded.'
            if new_frequency != original_frequency:
                details += f' Original frequency: {original_frequency} Hz; downsampled to {new_frequency} Hz.'
            else:
                details += f' Original frequency: {original_frequency} Hz; no downsampling applied.'

            if signal_name in {"accelerometer", "accelerometer2"}:
                try:
                    preview_cols = ["datetime"] + signal_cols
                    preview_cols = [c for c in preview_cols if c in signal_df.columns]
                    print(f"🔎 DEBUG {signal_name} dataframe columns: {list(signal_df.columns)}")
                    print(
                        f"🔎 DEBUG {signal_name} dataframe dtypes: "
                        f"{ {col: str(dtype) for col, dtype in signal_df.dtypes.items()} }"
                    )
                    print(
                        f"🔎 DEBUG {signal_name} dataframe header "
                        f"(cols={preview_cols}, rows={len(signal_df)}):"
                    )
                    print(signal_df[preview_cols].head(3).to_string(index=False))
                except Exception as e:
                    print(f"⚠️ Failed to print {signal_name} debug header: {type(e).__name__}: {e}")

            if stored_signal_name != signal_name:
                print(
                    f"🏷️  Storing derived '{signal_name}' from {logger_id} as "
                    f"'{stored_signal_name}' (canonical name reserved for pyologger)."
                )
            self.data_reader.signal_data[stored_signal_name] = signal_df
            time_zone_raw = self.data_reader.deployment_info.get('Time Zone')
            tz_name = str(time_zone_raw).strip() if time_zone_raw is not None else "UTC"
            if not tz_name:
                tz_name = "UTC"
            try:
                tz = pytz.timezone(tz_name)
            except Exception:
                print(f"⚠️ Invalid timezone '{tz_name}'. Falling back to UTC.")
                tz = pytz.UTC

            self.data_reader.signal_info[stored_signal_name] = {
                'channels': signal_cols,
                'metadata': {col: signal_channel_metadata[col] for col in signal_cols},
                'signal_start_datetime': start_time,
                'signal_end_datetime': end_time,
                'max_value': float(max_value),
                'min_value': float(min_value),
                'mean_value': float(mean_value),
                'data_type': data_type_str,
                'original_units': original_unit,
                'units': original_unit,
                'original_sampling_frequency': original_frequency,
                'sampling_frequency': new_frequency,
                'logger_id': self.logger_id,
                'logger_manufacturer': self.logger_manufacturer,
                'processing_step': 'Raw data uploaded',
                'last_updated': pd.Timestamp(datetime.now().astimezone(tz)),
                'details': details,
            }
            print(f"[group_data_by_signals] stored signal_info['{signal_name}']['channels'] = {signal_cols}")

        for signal_name, df in self.data_reader.signal_data.items():
            print(f"Sensor '{signal_name}' data processed and stored with shape {df.shape}.")

        return signal_groups, signal_info


    def process_files(self, files):
        """Process files in the subclass. This should be overridden."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def concatenate_and_save_csvs(self, csv_files):
        """Base method for concatenating and saving CSVs."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_expected_frequencies(self, parsed_signals, enforce_frequency=True):
        """
        Matches parsed signals with the channel mapping and sets expected frequencies.
        
        Args:
            parsed_signals (dict): Sensor name -> expected interval (Hz)
            enforce_frequency (bool): Whether to enforce setting expected frequencies. Default: True.
        """
        if not parsed_signals:
            print("⚠ No signals parsed from txt file, skipping frequency matching.")
            return

        if not self.montage:
            print(f"⚠ Warning: No valid channel mapping found for Manufacturer '{self.logger_manufacturer}' with Montage ID '{self.montage_id}'.")
            return

        print(f"🔍 Matching parsed signals with channel mapping. Enforce frequency: {enforce_frequency}")

        for signal_name, frequency in parsed_signals.items():
            found_match = False
            for clean_name, mapping in self.montage.items():
                manufacturer_signal_name = mapping['manufacturer_signal_name'].strip().lower()

                if manufacturer_signal_name == signal_name:
                    parent_signal = mapping['parent_signal'].strip().lower()

                    if enforce_frequency:
                        self.expected_frequencies[parent_signal] = frequency  
                        print(f"✅ Matched '{signal_name}' -> '{parent_signal}' with expected frequency: {frequency} Hz.")
                    else:
                        print(f"🔍 Matched '{signal_name}' -> '{parent_signal}', but skipping frequency enforcement.")

                    found_match = True
                    break  # Stop once a match is found

            if not found_match:
                print(f"⚠ Sensor name '{signal_name}' not found in channel mapping. Ignoring this signal.")
