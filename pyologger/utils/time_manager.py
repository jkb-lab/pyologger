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
                     min_report_seconds=None):
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
        With 'datetime', 'datetime_utc', 'time_unix_ms'
    metadata : dict
        {
            'datetime_created_from': str,
            'fs': float  # Hz, may be np.nan if can't infer
        }
    """

    metadata = {'datetime_created_from': None, 'fs': None}

    # Step 1: Create datetime column if needed
    if 'datetime' in df.columns:
        print("'datetime' column found.")
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        print(f"First few entries in 'datetime' column:\n{df['datetime'].head()}")
        metadata['datetime_created_from'] = 'datetime'

    elif 'time' in df.columns and 'date' in df.columns:
        print("'datetime' column not found. Combining 'date' and 'time' columns.")
        dates = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        times = pd.to_timedelta(df['time'].astype(str), errors='coerce')
        df['datetime'] = dates + times
        metadata['datetime_created_from'] = 'date and time'

    elif 'time_local' in df.columns and 'date_local' in df.columns:
        print("'datetime' and 'date/time' columns not found. Combining 'date_local' and 'time_local' columns.")
        dates = pd.to_datetime(df['date_local'], format='%d.%m.%Y', errors='coerce')
        times = pd.to_timedelta(df['time_local'].astype(str), errors='coerce')
        df['datetime'] = dates + times
        metadata['datetime_created_from'] = 'date_local and time_local'

    else:
        print("No suitable columns found to create a 'datetime' column.")
        return df, metadata

    # Step 2: Localize and convert to UTC
    # note: .dt.tz is deprecated-ish; use .dt.tz is None logic while guarding for tz-aware types
    if time_zone and (df['datetime'].dt.tz is None):
        print(f"Localizing datetime using timezone {time_zone}.")
        tz = pytz.timezone(time_zone)
        df['datetime'] = df['datetime'].dt.tz_localize(tz)
    else:
        # if already tz-aware, we assume it's correct and leave it
        if df['datetime'].dt.tz is None:
            # still naive and no timezone provided
            print("⚠️ Datetimes are naive (no timezone). Assuming they are already in UTC.")
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')

    df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')

    # Use int64 ns -> ms
    df['time_unix_ms'] = df['datetime_utc'].astype('int64') // 10**6

    # Step 3: Check monotonicity
    if not df['datetime'].is_monotonic_increasing:
        print("❌ The 'datetime' column is not monotonically increasing.")
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

    return df, metadata
