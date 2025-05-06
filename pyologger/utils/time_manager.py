import pytz
import numpy as np
import pandas as pd
from datetime import timedelta
from pyologger.process_data.sampling import calculate_sampling_frequency

def process_datetime(df, time_zone=None):
    """Processes datetime columns in the DataFrame, calculates sampling frequency, and checks for gaps."""
    metadata = {'datetime_created_from': None, 'fs': None}

    # Step 1: Create datetime column if needed
    if 'datetime' in df.columns:
        print("'datetime' column found.")
        metadata['datetime_created_from'] = 'datetime'
    elif 'time' in df.columns and 'date' in df.columns:
        print("'datetime' column not found. Combining 'date' and 'time' columns.")
        dates = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        times = pd.to_timedelta(df['time'].astype(str))
        df['datetime'] = dates + times
        metadata['datetime_created_from'] = 'date and time'
    elif 'time_local' in df.columns and 'date_local' in df.columns:
        print("'datetime' and 'date/time' columns not found. Combining 'date_local' and 'time_local' columns.")
        dates = pd.to_datetime(df['date_local'], format='%d.%m.%Y', errors='coerce')
        times = pd.to_timedelta(df['time_local'].astype(str))
        df['datetime'] = dates + times
        metadata['datetime_created_from'] = 'date_local and time_local'
    else:
        print("No suitable columns found to create a 'datetime' column.")
        return df, metadata

    # Step 2: Localize and convert to UTC
    if time_zone and df['datetime'].dt.tz is None:
        print(f"Localizing datetime using timezone {time_zone}.")
        tz = pytz.timezone(time_zone)
        df['datetime'] = df['datetime'].dt.tz_localize(tz)

    df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')
    df['time_unix_ms'] = df['datetime_utc'].astype(np.int64) // 10**6

    # Step 3: Check monotonicity
    if not df['datetime'].is_monotonic_increasing:
        print("❌ The 'datetime' column is not monotonically increasing.")
    else:
        print("✅ The 'datetime' column is monotonically increasing.")

    # Step 5: Detect large gaps > 20s
    actual_intervals = df['datetime'].diff().dropna()
    gap_threshold = timedelta(seconds=20)
    gap_indices = actual_intervals[actual_intervals > gap_threshold].index

    if not gap_indices.empty:
        durations = actual_intervals[gap_indices]
        print(f"⚠️ WARNING: {len(gap_indices)} gaps detected in 'datetime' column")
        print(f"   Gaps range from {durations.min()} to {durations.max()} in duration.")

        gap_rows = []
        for idx in gap_indices:
            gap_rows.append(df.iloc[idx - 1].to_dict())  # Row before
            gap_rows.append(df.iloc[idx].to_dict())      # Row after
        gap_df = pd.DataFrame(gap_rows)
        print("Rows surrounding gaps:")
        print(gap_df[['datetime', 'datetime_utc']])

        metadata['fs'] = calculate_sampling_frequency(df['datetime'].head())
        print(f"Sampling frequency: {metadata['fs']} Hz calculated using head() only.")
    else:
        print("✅ No gaps > 20s detected in 'datetime' column.")

        # Step 4: Calculate sampling frequency from head
        metadata['fs'] = calculate_sampling_frequency(df['datetime'])
        print(f"Sampling frequency: {metadata['fs']} Hz for whole column.")

    return df, metadata