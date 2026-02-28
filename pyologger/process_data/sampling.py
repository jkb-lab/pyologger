import numpy as np
import pandas as pd

def calculate_sampling_frequency(datetime_series):
    """
    Calculate the sampling frequency from a series of datetime values.
    For frequencies >= 1 Hz, rounds to nearest integer. For sub-1Hz, preserves precision.

    Parameters
    ----------
    datetime_series : pandas.Series or numpy.array
        A series or array containing datetime values.

    Returns
    -------
    float or None
        The calculated sampling frequency in Hz, or None if not enough valid data points.
    """
    # Ensure the input is in datetime format
    datetime_series = pd.to_datetime(datetime_series)

    # Calculate the time differences between consecutive values in seconds
    sec_diff = datetime_series.diff().dt.total_seconds().dropna()  # Drop NaNs here
    
    # Filter out zero and negative intervals
    sec_diff = sec_diff[sec_diff > 0]

    # Calculate the mean difference and sampling frequency
    if len(sec_diff) < 1:
        print("Insufficient data points to calculate sampling frequency.")
        return None
    
    mean_diff = sec_diff.mean()
    
    if mean_diff == 0 or not np.isfinite(mean_diff):
        print(f"Invalid mean interval: {mean_diff}")
        return None
    
    sampling_frequency = 1 / mean_diff
    
    # Round to nearest integer if >= 1 Hz, otherwise keep precision
    if sampling_frequency >= 1.0:
        sampling_frequency = round(sampling_frequency)

    return sampling_frequency


def upsample(data, upsampling_factor, original_length):
    """
    Upsamples the input data by repeating each value upsampling_factor times 
    and adjusts the length to match the original length.
    
    Parameters:
    - data: numpy array of the data to be upsampled.
    - upsampling_factor: int, the factor by which to upsample the data.
    - original_length: int, the length of the original data before downsampling.
    
    Returns:
    - numpy array of the upsampled data adjusted to the original length.
    """
    # Step 1: Repeat the data to upsample
    upsampled_data = np.repeat(data, upsampling_factor)
    
    # Step 2: Adjust the length to match the original length
    if len(upsampled_data) > original_length:
        upsampled_data = upsampled_data[:original_length]
    elif len(upsampled_data) < original_length:
        upsampled_data = np.pad(upsampled_data, (0, original_length - len(upsampled_data)), 'edge')
    
    return upsampled_data

def downsample(df, original_fs, target_fs):
    """Downsample dataframe by selecting every nth row.
    
    Parameters:
    - df: pandas DataFrame
    - original_fs: float, original sampling frequency in Hz
    - target_fs: float, target sampling frequency in Hz
    
    Returns:
    - pandas DataFrame downsampled to target frequency
    """
    # If sampling frequency cannot be estimated, fallback to time-based decimation
    # using the requested target sampling interval.
    if original_fs is None or target_fs is None:
        return _downsample_by_target_interval(df, target_fs)
    try:
        original_fs = float(original_fs)
        target_fs = float(target_fs)
    except (TypeError, ValueError):
        return _downsample_by_target_interval(df, target_fs)
    if not np.isfinite(original_fs) or not np.isfinite(target_fs) or original_fs <= 0 or target_fs <= 0:
        return _downsample_by_target_interval(df, target_fs)

    if target_fs >= original_fs:
        return df
    conversion_factor = max(1, int(round(original_fs / target_fs)))
    print(f"Original FS: {original_fs:.6f} Hz, Target FS: {target_fs:.6f} Hz, Conversion Factor: {conversion_factor}")
    return df.iloc[::conversion_factor, :]


def _downsample_by_target_interval(df, target_fs):
    """
    Downsample using datetime spacing only, keeping approximately one row per
    target sampling interval when original_fs is unavailable.
    """
    try:
        target_fs = float(target_fs)
    except (TypeError, ValueError):
        return df
    if not np.isfinite(target_fs) or target_fs <= 0:
        return df
    if "datetime" not in df.columns:
        return df

    work = df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["datetime"]).sort_values("datetime")
    if work.empty:
        return work

    min_delta = pd.Timedelta(seconds=(1.0 / target_fs))
    dt = work["datetime"]
    keep = np.zeros(len(work), dtype=bool)
    keep[0] = True
    last_kept = dt.iloc[0]
    for i in range(1, len(work)):
        if (dt.iloc[i] - last_kept) >= min_delta:
            keep[i] = True
            last_kept = dt.iloc[i]

    out = work.loc[keep]
    print(
        f"Downsample fallback by interval: target_fs={target_fs:.6f} Hz, "
        f"kept {len(out)} / {len(work)} rows."
    )
    return out

def resample_df(df, target_fs, original_fs=None):
    """
    Resamples the DataFrame to the target frequency, adjusting the datetime
    index accordingly.

    Parameters:
    - df: pandas DataFrame with a DatetimeIndex.
    - target_fs: float, the target frequency in Hz.
    - original_fs: float, optional, the original frequency in Hz.

    Returns:
    - pandas DataFrame resampled to the target frequency.
    """
    df.index = pd.to_datetime(df.index)

    # Ensure the index is sorted
    df = df.sort_values(by="datetime")

    if original_fs is None:
        # Estimate the original frequency from the datetime index
        original_intervals = df["datetime"].diff().dropna()

        # Filter out zero and negative intervals
        original_intervals = original_intervals[original_intervals > pd.Timedelta(0)]

        if len(original_intervals) == 0:
            raise ValueError(
                "Cannot estimate original frequency: all time intervals are zero or negative."
            )

        # Use median to estimate the interval
        original_interval = original_intervals.median()

        if original_interval.total_seconds() == 0:
            raise ValueError(
                "Original interval is zero after filtering; cannot estimate original frequency."
            )

        original_fs = 1 / original_interval.total_seconds()

    if original_fs == 0:
        raise ValueError("Original frequency is zero, cannot resample.")

    if target_fs == original_fs:
        return df
    elif target_fs < original_fs:
        return downsample_df(df, original_fs, target_fs)
    else:
        return upsample_df(df, original_fs, target_fs)
    
def upsample_df(df, original_fs, target_fs):
    """
    Upsamples the DataFrame to the target frequency by forward-filling data
    and adjusting the datetime index accordingly.

    Parameters:
    - df: pandas DataFrame with a DatetimeIndex.
    - target_fs: float, the target frequency in Hz.

    Returns:
    - pandas DataFrame upsampled to the target frequency.
    """
    original_length = len(df)
    upsampling_factor = int(target_fs / original_fs)

    # Step 1: Repeat the data to upsample
    new_df = pd.DataFrame()

    # For all columns that aren't datetime, repeat the data
    for col in df.columns:
        if col != "datetime":
            new_df[col] = np.repeat(df[col], upsampling_factor)
        else:
            number_of_seconds = (df[col].iloc[-1] - df[col].iloc[0]).total_seconds() + 1
            seconds_elapsed = np.arange(0, number_of_seconds, 1 / target_fs)

            new_df[col] = df[col].iloc[0] + pd.to_timedelta(seconds_elapsed, unit="ms")

    # Step 2: Adjust the length to match the original length
    if len(new_df) > original_length:
        new_df = new_df[:original_length]
    elif len(new_df) < original_length:
        new_df = np.pad(new_df, (0, original_length - len(new_df)), "edge")

    return new_df


def downsample_df(df, original_fs, target_fs):
    """
    Downsamples the DataFrame to the target frequency by taking every nth row.

    Parameters:
    - df: pandas DataFrame with a DatetimeIndex.
    - original_fs: float, the original frequency in Hz.
    - target_fs: float, the target frequency in Hz.

    Returns:
    - pandas DataFrame downsampled to the target frequency.
    """
    # If frequency metadata is missing/invalid, keep data unchanged.
    if original_fs is None or target_fs is None:
        return df
    try:
        original_fs = float(original_fs)
        target_fs = float(target_fs)
    except (TypeError, ValueError):
        return df
    if not np.isfinite(original_fs) or not np.isfinite(target_fs) or original_fs <= 0 or target_fs <= 0:
        return df

    if target_fs >= original_fs:
        return df
    conversion_factor = max(1, int(round(original_fs / target_fs)))
    print(
        f"Original FS: {original_fs:.6f} Hz, Target FS: {target_fs:.6f} Hz, Conversion Factor: {conversion_factor}"
    )
    return df.iloc[::conversion_factor, :]
