"""Utilities for estimating time offset between two signals using cross-correlation.

This module provides a simple, robust function `estimate_time_lag` which will
resample/align two input signals (pandas Series or numeric arrays with optional
datetime indices), compute the cross-correlation, and return the lag in seconds
and a shifted version of the second signal aligned to the first.

The implementation keeps dependencies minimal (numpy, pandas, scipy optional)
and uses numpy.correlate for the cross-correlation computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _to_series(x, name="x") -> pd.Series:
    """Convert input to pandas Series with a numeric (float) index in seconds.

    If x is a pandas Series with a datetime index, the index is converted to
    seconds relative to its first timestamp. If x is a numpy array or list,
    it is converted to a Series with an integer index (interpreted as samples).
    """
    if isinstance(x, pd.Series):
        s = x.copy()
        if isinstance(s.index, pd.DatetimeIndex):
            t0 = s.index[0]
            seconds = (s.index - t0).total_seconds()
            s.index = seconds
        else:
            # keep numeric index as-is
            s.index = s.index.astype(float)
        s.name = name
        return s
    else:
        return pd.Series(np.asarray(x).ravel(), name=name)


def estimate_time_lag(
    ref,
    targ,
    resample_rate: Optional[float] = 10.0,
    max_lag_seconds: Optional[float] = None,
    return_shifted: bool = True,
) -> Tuple[float, int, Optional[pd.Series]]:
    """Estimate time lag between two signals using normalized cross-correlation.

    Parameters
    - ref: reference signal (pandas Series or array). If Series and has a
      DatetimeIndex it will be converted to seconds from its first timestamp.
    - targ: target signal to align to reference (same types supported).
    - resample_rate: target sampling rate in Hz for resampling both signals
      before correlation (default 10 Hz). If None, signals are left at their
      native sampling (integer index).
    - max_lag_seconds: maximum absolute lag (in seconds) to consider. If None,
      the full length is used.
    - return_shifted: if True, returns targ shifted to align with ref as a
      pandas Series; otherwise returns None for the shifted signal.

    Returns
    - lag_seconds: estimated lag in seconds (positive means targ lags behind
      ref and needs to be shifted earlier to align).
    - lag_samples: integer number of samples of the resampled signals
      corresponding to the lag (positive means targ lags behind ref).
    - shifted_targ: targ shifted (resampled) and aligned to ref (or None).

    Notes
    - The function zero-mean normalizes both signals before cross-correlation.
    - For signals with different timebases, resampling to a common rate is
      recommended (controlled by resample_rate).
    """
    ref_s = _to_series(ref, name="ref")
    targ_s = _to_series(targ, name="targ")

    # If both have numeric indices that look like seconds, try to build a
    # common datetime-like index for resampling. Otherwise we will treat them
    # as arrays and resample via linear interpolation on a uniform grid.
    # Determine start time for both (in seconds)
    t0 = 0.0
    if isinstance(ref_s.index, pd.Index) and len(ref_s.index) > 0:
        try:
            # if index values are seconds floats
            t0 = float(ref_s.index[0])
        except Exception:
            t0 = 0.0

    # Build uniform time grid
    if resample_rate is None or resample_rate <= 0:
        # use native integer samples
        ref_vals = ref_s.values
        targ_vals = targ_s.values
        sample_dt = 1.0
    else:
        sample_dt = 1.0 / float(resample_rate)
        # determine overlapping time window
        ref_start = float(ref_s.index[0])
        ref_end = float(ref_s.index[-1])
        targ_start = float(targ_s.index[0])
        targ_end = float(targ_s.index[-1])

        start = max(ref_start, targ_start)
        end = min(ref_end, targ_end)
        if end <= start:
            # fallback to union range
            start = min(ref_start, targ_start)
            end = max(ref_end, targ_end)

        n = max(2, int(np.floor((end - start) * resample_rate)))
        grid = np.linspace(start, end, n)
        ref_vals = np.interp(grid, ref_s.index.astype(float), ref_s.values)
        targ_vals = np.interp(grid, targ_s.index.astype(float), targ_s.values)

    # zero-mean normalize
    ref_vals = np.asarray(ref_vals, dtype=float)
    targ_vals = np.asarray(targ_vals, dtype=float)
    ref_vals = ref_vals - np.nanmean(ref_vals)
    targ_vals = targ_vals - np.nanmean(targ_vals)

    # handle NaNs
    ref_vals = np.nan_to_num(ref_vals)
    targ_vals = np.nan_to_num(targ_vals)

    # compute cross-correlation
    corr = np.correlate(ref_vals, targ_vals, mode="full")
    lags = np.arange(-len(targ_vals) + 1, len(ref_vals))

    # optionally limit to max_lag_seconds
    if max_lag_seconds is not None and resample_rate is not None and resample_rate > 0:
        max_lag_samples = int(np.round(max_lag_seconds * resample_rate))
        center = len(corr) // 2
        low = max(0, center - max_lag_samples)
        high = min(len(corr), center + max_lag_samples + 1)
        sub = corr[low:high]
        sublags = lags[low - (len(targ_vals) - 1):high - (len(targ_vals) - 1)]
        argmax = np.argmax(sub)
        lag_samples = sublags[argmax]
    else:
        argmax = np.argmax(corr)
        lag_samples = lags[argmax]

    # compute lag in seconds
    lag_seconds = float(lag_samples * sample_dt)

    shifted = None
    if return_shifted:
        # shift targ_vals by -lag_samples so that targ aligns with ref
        if lag_samples > 0:
            shifted_vals = np.concatenate((targ_vals[lag_samples:], np.full(lag_samples, np.nan)))
        elif lag_samples < 0:
            shifted_vals = np.concatenate((np.full(-lag_samples, np.nan), targ_vals[:lag_samples]))
        else:
            shifted_vals = targ_vals.copy()

        # build a pandas Series mirroring the ref grid if resampled, otherwise plain indices
        if resample_rate is None or resample_rate <= 0:
            shifted = pd.Series(shifted_vals)
        else:
            shifted = pd.Series(shifted_vals, index=grid)

    return lag_seconds, int(lag_samples), shifted


__all__ = ["estimate_time_lag"]
