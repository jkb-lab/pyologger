import pytz
import numpy as np
import pandas as pd
from pyologger.process_data.sampling import calculate_sampling_frequency

def process_datetime(df, time_zone=None):
    """Processes datetime columns in the DataFrame and calculates sampling frequency."""
    metadata = {'datetime_created_from': None, 'fs': None}

    if 'datetime' in df.columns:
        print("'datetime' column found.")
        metadata['datetime_created_from'] = 'datetime'
        if time_zone and df['datetime'].dt.tz is None:
            print(f"Localizing datetime using timezone {time_zone}.")
            tz = pytz.timezone(time_zone)
            df['datetime'] = df['datetime'].dt.tz_localize(tz)

        df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')
        df['time_unix_ms'] = df['datetime_utc'].astype(np.int64) // 10**6
        metadata['fs'] = calculate_sampling_frequency(df['datetime_utc'])
        return df, metadata

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
        return df

    if time_zone and df['datetime'].dt.tz is None:
        print(f"Localizing datetime using timezone {time_zone}.")
        tz = pytz.timezone(time_zone)
        df['datetime'] = df['datetime'].dt.tz_localize(tz)

    print("Converting to UTC and Unix.")
    df['datetime_utc'] = df['datetime'].dt.tz_convert('UTC')
    df['time_unix_ms'] = df['datetime_utc'].astype(np.int64) // 10**6
    metadata['fs'] = calculate_sampling_frequency(df['datetime_utc'])

    return df, metadata