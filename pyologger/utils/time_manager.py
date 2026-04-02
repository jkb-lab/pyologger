import pytz
import numpy as np
import pandas as pd
from datetime import timedelta

def calculate_sampling_frequency(datetime_series: pd.Series) -> float:
    """
    Estimate sampling frequency (Hz) from a datetime series.

    Parameters
    ----------
    datetime_series : pandas.Series
        A pandas Series of datetime-like values. Can be tz-aware or naive.

    Returns
    -------
    float
        Estimated sampling frequency in Hz (can be <1 for slow signals).
        Returns np.nan if it cannot be determined.
    """
    if datetime_series is None or len(datetime_series) < 2:
        return np.nan

    # Ensure datetime dtype
    dt = pd.to_datetime(datetime_series, errors='coerce')
    diffs = dt.diff().dropna().dt.total_seconds()

    # Remove zero or negative diffs (duplicates or disorder)
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return np.nan

    # Use median for robustness
    median_interval = np.median(diffs)

    if median_interval <= 0 or np.isnan(median_interval):
        return np.nan

    return 1.0 / median_interval

def _infer_nominal_interval(datetime_series: pd.Series) -> float:
    """
    Infer the nominal sampling interval (in seconds) from a datetime series.

    Uses the median of consecutive diffs in seconds, ignoring NaT and 0s.
    Returns np.nan if it can't infer.
    """
    diffs = datetime_series.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]  # ignore 0 or negative (duplicates/out of order)
    if len(diffs) == 0:
        return np.nan
    return float(np.median(diffs))


def _infer_sampling_frequency(datetime_series: pd.Series) -> float:
    """
    Infer sampling frequency in Hz from a datetime series.
    Handles sub-second and <1 Hz data.
    """
    nominal_interval = _infer_nominal_interval(datetime_series)
    if np.isnan(nominal_interval) or nominal_interval <= 0:
        return np.nan
    return 1.0 / nominal_interval


def _find_gaps(datetime_series: pd.Series,
               nominal_interval_sec: float,
               tolerance_factor: float = 1.5,
               min_report_seconds: float = None):
    """
    Find gaps larger than expected.

    A gap is flagged if:
    - gap_sec > nominal_interval_sec * tolerance_factor
    - AND (if min_report_seconds is set) gap_sec >= min_report_seconds

    If nominal_interval_sec is nan or <=0, we can't infer cadence,
    so every positive gap is reported.
    """
    gaps = []
    diffs = datetime_series.diff()

    for idx in range(1, len(datetime_series)):
        gap = diffs.iloc[idx]
        if pd.isna(gap):
            continue

        gap_sec = gap.total_seconds()
        if gap_sec <= 0:
            continue  # backwards or duplicate timestamp, handled separately

        if np.isnan(nominal_interval_sec) or nominal_interval_sec <= 0:
            is_gap = True  # no cadence info -> report all positive jumps
        else:
            is_gap = gap_sec > (nominal_interval_sec * tolerance_factor)

        if min_report_seconds is not None:
            is_gap = is_gap and (gap_sec >= min_report_seconds)

        if is_gap:
            gaps.append((idx, gap_sec))

    return gaps


def process_datetime(df, time_zone=None,
                     tolerance_factor=1.5,
                     min_report_seconds=None,
                     channel_metadata=None):
    """
    Processes datetime columns in the DataFrame:
    - creates datetime if needed
    - localizes to provided time_zone and converts to UTC
    - checks monotonicity
    - infers sampling frequency (Hz, can be <1)
    - detects gaps relative to inferred cadence

    Returns
    -------
    df : pandas.DataFrame
        With 'datetime', 'datetime_utc', 'datetimenum_utc'
    metadata : dict
        {
            'datetime_created_from': str,
            'fs': float  # Hz, may be np.nan if can't infer
        }
    channel_metadata : dict, optional
        Mapping of standardized column names to metadata entries with
        'unit' (original_unit) and 'standardized_unit'.
    """

    metadata = {'datetime_created_from': None, 'fs': None}
    if time_zone is not None:
        print(f"time_zone value/type: {time_zone} ({type(time_zone)})")
    if time_zone is not None:
        time_zone = str(time_zone).strip() or None

    def _first_valid(series: pd.Series):
        if series is None:
            return None
        non_null = series.dropna()
        if non_null.empty:
            return None
        return non_null.iloc[0]

    def _log_conversion(label, before, after=None, tz_label=None):
        if before is not None:
            print(f"First {label} timestamp: {before}")
        if after is not None and after is not before:
            if tz_label:
                print(f"First {label} after conversion to {tz_label}: {after}")
            else:
                print(f"First {label} after conversion: {after}")

    def _parse_unix(series: pd.Series):
        # Expect Unix seconds; auto-detect ms based on magnitude.
        numeric = pd.to_numeric(series, errors='coerce')
        non_null = numeric.dropna()
        if non_null.empty:
            return pd.to_datetime(series, unit='s', errors='coerce', utc=True)

        median_val = float(non_null.median())
        unit = 'ms' if median_val >= 1e11 else 's'
        if unit == 'ms':
            print("Detected Unix timestamps in milliseconds.")
        return pd.to_datetime(numeric, unit=unit, errors='coerce', utc=True)

    def _get_unit(col_name: str):
        if not channel_metadata:
            return None
        entry = channel_metadata.get(col_name, {})
        return entry.get("unit")

    def _get_standardized_unit(col_name: str):
        if not channel_metadata:
            return None
        entry = channel_metadata.get(col_name, {})
        unit = entry.get("standardized_unit")
        return None if unit in (None, "unknown") else unit

    def _date_format_from_unit(unit: str):
        if not unit or str(unit).lower() in ("unknown", "nan", "none"):
            return "%m/%d/%y"
        fmt = str(unit).strip().upper().replace(" ", "")
        fmt = fmt.replace("YYYY", "%Y").replace("YY", "%y")
        fmt = fmt.replace("DD", "%d").replace("MM", "%m")
        return fmt if "%" in fmt else "%m/%d/%y"

    def _time_format_from_unit(unit: str):
        if not unit or str(unit).lower() in ("unknown", "nan", "none"):
            return None
        fmt = str(unit).strip().upper().replace(" ", "")
        fmt = fmt.replace("HH", "%H").replace("MM", "%M").replace("SS", "%S")
        if ".000" in fmt or ".SSS" in fmt:
            fmt = fmt.replace(".000", ".%f").replace(".SSS", ".%f")
        return fmt if "%" in fmt else None

    def _parse_date_series(series: pd.Series, unit: str):
        sample = series.dropna().head(3).tolist()
        if sample:
            print(f"Sample date values: {sample} (types: {[type(v) for v in sample]})")
        series = series.apply(lambda v: str(v) if isinstance(v, np.str_) else v).astype("string")
        fmt = _date_format_from_unit(unit)
        parsed = pd.to_datetime(series, format=fmt, errors='coerce')
        nat_ratio = parsed.isna().mean()
        if nat_ratio > 0.2:
            print(f"⚠️ {nat_ratio:.1%} date values failed to parse with format {fmt}.")
            alt_fmt = "%m/%d/%Y"
            alt_parsed = pd.to_datetime(series, format=alt_fmt, errors='coerce')
            recovered_mask = parsed.isna() & alt_parsed.notna()
            if recovered_mask.any():
                recovered_pct = recovered_mask.mean()
                print(f"Recovered {recovered_pct:.1%} dates using alternate format {alt_fmt}.")
                parsed = parsed.fillna(alt_parsed)
            if not unit or str(unit).lower() in ("unknown", "nan", "none"):
                parsed = pd.to_datetime(series.astype(str), errors='coerce')
        if parsed.isna().any():
            failed = series[parsed.isna()].dropna().head(2).tolist()
            if failed:
                print(f"Failed date values (sample): {failed}")
        return parsed

    def _parse_time_series(series: pd.Series, unit: str):
        sample = series.dropna().head(3).tolist()
        if sample:
            print(f"Sample time values: {sample} (types: {[type(v) for v in sample]})")
        series = series.apply(lambda v: str(v) if isinstance(v, np.str_) else v).astype("string")
        fmt = _time_format_from_unit(unit)
        if fmt:
            parsed = pd.to_datetime(series, format=fmt, errors='coerce')
            if parsed.isna().mean() <= 0.2:
                return pd.to_timedelta(parsed.dt.strftime("%H:%M:%S.%f"), errors='coerce')
        if not unit or str(unit).lower() in ("unknown", "nan", "none"):
            return pd.to_timedelta(series.astype(str), errors='coerce')
        print(f"⚠️ Falling back to time parsing without format for unit {unit}.")
        td = pd.to_timedelta(series.astype(str), errors='coerce')
        if td.isna().any():
            failed = series[td.isna()].dropna().head(2).tolist()
            if failed:
                print(f"Failed time values (sample): {failed}")
        return td

    def _precision_from_unit(unit: str):
        if not unit:
            return None
        upper = str(unit).upper()
        if ".000" in upper or "SSS" in upper or "%f" in upper:
            return "ms"
        if "SS" in upper:
            return "s"
        return None

    # Step 1: Create datetime column if needed
    if 'datetime' in df.columns:
        print("'datetime' column found.")
        # First, try strict parsing with common format 'YYYY-MM-DD HH:MM:SS'
        strict_parsed = pd.to_datetime(
            df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce'
        )
        # For any rows that failed strict parsing, fall back to pandas' general parser
        if strict_parsed.isna().any():
            fallback_parsed = pd.to_datetime(df['datetime'], errors='coerce')
            # Prefer strict where available; otherwise use fallback
            df['datetime'] = strict_parsed.fillna(fallback_parsed)
        else:
            df['datetime'] = strict_parsed

        _log_conversion("parsed datetime", _first_valid(df['datetime']))
        metadata['datetime_created_from'] = 'datetime'

    elif 'datetimenum' in df.columns:
        print("'datetimenum' column found. Converting from Unix time.")
        dt_utc = _parse_unix(df['datetimenum'])
        if time_zone:
            df['datetime'] = dt_utc.dt.tz_convert(time_zone)
            _log_conversion(
                "datetimenum (UTC)", _first_valid(dt_utc),
                _first_valid(df['datetime']), time_zone
            )
        else:
            df['datetime'] = dt_utc
            _log_conversion("datetimenum (UTC)", _first_valid(df['datetime']))
        metadata['datetime_created_from'] = 'datetimenum'

    elif 'datetimenum_utc' in df.columns:
        print("'datetimenum_utc' column found. Converting from Unix time.")
        dt_utc = _parse_unix(df['datetimenum_utc'])
        if time_zone:
            df['datetime'] = dt_utc.dt.tz_convert(time_zone)
            _log_conversion(
                "datetimenum_utc (UTC)", _first_valid(dt_utc),
                _first_valid(df['datetime']), time_zone
            )
        else:
            df['datetime'] = dt_utc
            _log_conversion("datetimenum_utc (UTC)", _first_valid(df['datetime']))
        metadata['datetime_created_from'] = 'datetimenum_utc'

    elif 'datetime_utc' in df.columns:
        # Check if already datetime type
        if pd.api.types.is_datetime64_any_dtype(df['datetime_utc']):
            print("'datetime_utc' column is already datetime format.")
            dt_utc = pd.to_datetime(df['datetime_utc'], utc=True)
            if time_zone:
                df['datetime'] = dt_utc.dt.tz_convert(time_zone)
                _log_conversion(
                    "datetime_utc (already datetime)", _first_valid(dt_utc),
                    _first_valid(df['datetime']), time_zone
                )
            else:
                df['datetime'] = dt_utc
                _log_conversion("datetime_utc (already datetime)", _first_valid(df['datetime']))
            metadata['datetime_created_from'] = 'datetime_utc'
        else:
            numeric = pd.to_numeric(df['datetime_utc'], errors='coerce')
            if numeric.notna().any():
                print("'datetime_utc' column looks numeric. Converting from Unix time.")
                dt_utc = _parse_unix(df['datetime_utc'])
                if time_zone:
                    df['datetime'] = dt_utc.dt.tz_convert(time_zone)
                    _log_conversion(
                        "datetime_utc (UTC)", _first_valid(dt_utc),
                        _first_valid(df['datetime']), time_zone
                    )
                else:
                    df['datetime'] = dt_utc
                    _log_conversion("datetime_utc (UTC)", _first_valid(df['datetime']))
                metadata['datetime_created_from'] = 'datetime_utc_unix'
            else:
                print("'datetime_utc' column found. Parsing as datetime.")
                parsed = pd.to_datetime(df['datetime_utc'].astype(str), errors='coerce')
                df['datetime'] = parsed
                _log_conversion("parsed datetime_utc", _first_valid(df['datetime']))
                metadata['datetime_created_from'] = 'datetime_utc'

    elif 'time_utc' in df.columns and 'date_utc' in df.columns:
        print("'datetime' column not found. Combining 'date_utc' and 'time_utc' columns.")
        date_unit = _get_unit('date_utc')
        time_unit = _get_unit('time_utc')
        print(f"Using original_unit for date_utc: {date_unit}")
        print(f"Using original_unit for time_utc: {time_unit}")
        dates = _parse_date_series(df['date_utc'], date_unit)
        times = _parse_time_series(df['time_utc'], time_unit)
        # Optional sub-second sampling column (e.g., time_hz with values 1..16)
        hz_col = None
        for candidate in ("time_hz", "time_Hz"):
            if candidate in df.columns:
                hz_col = candidate
                break

        if hz_col:
            samples = pd.to_numeric(df[hz_col], errors='coerce')
            max_sample = samples.dropna().max()
            if pd.notna(max_sample) and max_sample > 1:
                sample_rate = int(round(max_sample))
                offsets = (samples - 1) / sample_rate
                times = times + pd.to_timedelta(offsets, unit='s')
                print(f"Applied sub-second offsets from {hz_col} using {sample_rate} Hz.")
            else:
                print(f"⚠️ {hz_col} present but sample rate could not be inferred.")

        try:
            dt_utc = (dates + times).dt.tz_localize('UTC')
        except Exception as e:
            print(f"❌ Failed combining date_utc/time_utc: {e}")
            print(f"date_utc dtype: {dates.dtype}, time_utc dtype: {times.dtype}")
            print(f"date_utc sample: {dates.head(3).tolist()}")
            print(f"time_utc sample: {times.head(3).tolist()}")
            raise
        if time_zone:
            df['datetime'] = dt_utc.dt.tz_convert(time_zone)
            _log_conversion(
                "date_utc/time_utc (UTC)", _first_valid(dt_utc),
                _first_valid(df['datetime']), time_zone
            )
        else:
            df['datetime'] = dt_utc
            _log_conversion("date_utc/time_utc (UTC)", _first_valid(df['datetime']))
        metadata['datetime_created_from'] = 'date_utc and time_utc'

    elif 'time' in df.columns and 'date' in df.columns:
        print("'datetime' column not found. Combining 'date' and 'time' columns.")
        date_unit = _get_unit('date')
        time_unit = _get_unit('time')
        dates = _parse_date_series(df['date'], date_unit)
        times = _parse_time_series(df['time'], time_unit)
        df['datetime'] = dates + times
        metadata['datetime_created_from'] = 'date and time'

    elif 'time_local' in df.columns and 'date_local' in df.columns:
        print("'datetime' and 'date/time' columns not found. Combining 'date_local' and 'time_local' columns.")
        date_unit = _get_unit('date_local')
        time_unit = _get_unit('time_local')
        dates = _parse_date_series(df['date_local'], date_unit)
        times = _parse_time_series(df['time_local'], time_unit)
        df['datetime'] = dates + times
        metadata['datetime_created_from'] = 'date_local and time_local'

    else:
        print("No suitable columns found to create a 'datetime' column.")
        return df, metadata

    # Step 2: Localize and convert to UTC
    # note: .dt.tz is deprecated-ish; use .dt.tz is None logic while guarding for tz-aware types
    if time_zone and (df['datetime'].dt.tz is None):
        _log_conversion("datetime before localization", _first_valid(df['datetime']))
        print(f"Localizing datetime using timezone {time_zone}.")
        tz = pytz.timezone(time_zone)
        df['datetime'] = df['datetime'].dt.tz_localize(tz)
        _log_conversion("datetime localized", None, _first_valid(df['datetime']), time_zone)
    else:
        # if already tz-aware, we assume it's correct and leave it
        if df['datetime'].dt.tz is None:
            _log_conversion("datetime before localization", _first_valid(df['datetime']))
            # still naive and no timezone provided
            print("⚠️ Datetimes are naive (no timezone). Assuming they are already in UTC.")
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
            _log_conversion("datetime localized", None, _first_valid(df['datetime']), 'UTC')

    df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')
    _log_conversion("datetime_utc", None, _first_valid(df['datetime_utc']), 'UTC')

    # Standardize datetime precision based on standardized_unit (if provided)
    standardized_dt_unit = (
        _get_standardized_unit('datetime')
        or _get_standardized_unit('datetime_utc')
        or (
            (_get_standardized_unit('date_utc') or "")
            + (" " if _get_standardized_unit('time_utc') else "")
            + (_get_standardized_unit('time_utc') or "")
        ).strip()
    )
    precision = _precision_from_unit(standardized_dt_unit)
    if precision:
        df['datetime'] = df['datetime'].dt.round(precision)
        df['datetime_utc'] = df['datetime_utc'].dt.round(precision)
        print(f"Standardized datetime precision to {precision} using unit '{standardized_dt_unit}'.")

    # Use int64 ns -> seconds for a stable Unix timestamp column
    df['datetimenum_utc'] = df['datetime_utc'].astype('int64') // 10**9

    # Step 3: enforce chronological row order before downstream grouping.
    if not df['datetime'].is_monotonic_increasing:
        print("⚠️ The 'datetime' column is not monotonically increasing. Sorting rows by datetime.")
        df = df.sort_values('datetime', kind='stable').reset_index(drop=True)

    if not df['datetime'].is_monotonic_increasing:
        print("❌ The 'datetime' column is still not monotonically increasing after sorting.")
    else:
        print("✅ The 'datetime' column is monotonically increasing.")

    # Step 4: Infer sampling frequency for metadata['fs']
    fs_est = _infer_sampling_frequency(df['datetime'])
    metadata['fs'] = fs_est
    if np.isnan(fs_est):
        print("⚠️ Could not infer sampling frequency (fs).")
    else:
        print(f"Estimated sampling frequency: {fs_est:.6f} Hz")

    # Step 5: Detect gaps using inferred cadence
    nominal_interval_sec = np.nan if np.isnan(fs_est) or fs_est <= 0 else (1.0 / fs_est)

    gaps = _find_gaps(
        df['datetime'],
        nominal_interval_sec=nominal_interval_sec,
        tolerance_factor=tolerance_factor,
        min_report_seconds=min_report_seconds
    )

    if len(gaps) > 0:
        gap_secs = [g[1] for g in gaps]
        print(f"⚠️ WARNING: {len(gaps)} gaps detected in 'datetime' column")
        print(f"   Gaps range from {min(gap_secs)}s to {max(gap_secs)}s in duration.")

        # collect rows around each gap for context
        gap_rows = []
        for idx, gap_sec in gaps:
            before_row = df.iloc[idx - 1].to_dict() if idx - 1 >= 0 else None
            after_row = df.iloc[idx].to_dict() if idx < len(df) else None
            gap_rows.append({
                "idx_before": idx - 1,
                "idx_after": idx,
                "datetime_before": before_row['datetime'] if before_row else None,
                "datetime_after": after_row['datetime'] if after_row else None,
                "gap_seconds": gap_sec
            })

        gap_df = pd.DataFrame(gap_rows)
        print("Rows surrounding gaps:")
        print(gap_df[['idx_before', 'idx_after', 'datetime_before', 'datetime_after', 'gap_seconds']])

    else:
        print("✅ No unexpected gaps detected in 'datetime' column.")

    # Step 6: Drop redundant datetime component columns
    keep_cols = {"datetime", "datetime_utc", "datetimenum_utc"}
    redundant_cols = {
        "date", "time", "date_utc", "time_utc", "date_local", "time_local",
        "time_hz", "timehz", "time_s", "time_ms", "time_unix_ms",
        "date_gmt", "time_gmt", "datetimenum", "datetimenumeric"
    }
    drop_cols = [
        col for col in df.columns
        if col.lower() in redundant_cols and col.lower() not in keep_cols
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df, metadata
