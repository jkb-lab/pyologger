"""
Activity/temperature analysis helpers for slow-sampling archival tags.

Built for Wildlife Computers MiniPAT deployments (3 s sampling), where the
standard high-rate accelerometer pipeline does not apply:

- ODBA/dynamic acceleration from `03_tag2animal` is degenerate at this rate.
  That step derives the static component with a 2 s rolling mean, which at
  1/3 Hz rounds to a **1-sample** window, so dynamic = raw - raw = exactly 0.
  Activity must therefore be recomputed with a window defined in *hours*,
  not seconds.
- Stroke/fin-beat detection is impossible: beat frequencies sit at or above
  the 0.167 Hz Nyquist and are aliased away (Step 04 skips for this reason).

What *is* resolvable at 3 s sampling is long-period structure -- diel (24 h),
semi-diel/tidal (12 h), and multi-day to lunar (~29.5 d) cycles -- which is
what the wavelet spectrogram here targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Wavelet analysis defaults. A Morlet wavelet with a moderate centre frequency
# trades time localisation for frequency resolution, which suits multi-day
# periodicity far better than the short bursts stroke detection looks for.
DEFAULT_WAVELET = "cmor1.5-1.0"

# Periods of biological interest, in hours.
PERIODS_OF_INTEREST_HOURS = {
    "6 h": 6.0,
    "12 h (semi-diel / tidal)": 12.0,
    "24 h (diel)": 24.0,
    "7 d (weekly)": 24.0 * 7,
    "14.8 d (spring-neap)": 24.0 * 14.765,
    "29.5 d (lunar)": 24.0 * 29.53,
}


def compute_activity_from_acc(
    acc_df: pd.DataFrame,
    window_hours: float = 1.0,
    method: str = "vedba",
) -> pd.DataFrame:
    """
    Recompute a dynamic-acceleration activity index for slow-sampled data.

    The static (postural) component is estimated with a centred rolling mean
    whose width is expressed in hours, guaranteeing a multi-sample window at
    3 s sampling. Activity is the norm of the residual.

    method: 'vedba' (2-norm, default) or 'odba' (1-norm).
    """
    required = {"datetime", "ax", "ay", "az"}
    missing = required - set(acc_df.columns)
    if missing:
        raise ValueError(f"acc_df is missing columns: {sorted(missing)}")

    work = acc_df[["datetime", "ax", "ay", "az"]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    work = work.sort_values("datetime").reset_index(drop=True)

    fs = estimate_sampling_rate_hz(work["datetime"])
    n_samples = int(round(window_hours * 3600.0 * fs))
    if n_samples < 3:
        raise ValueError(
            f"window_hours={window_hours} gives only {n_samples} samples at "
            f"fs={fs:.4f} Hz; choose a longer window."
        )
    # Odd window keeps the rolling mean centred without a half-sample shift.
    if n_samples % 2 == 0:
        n_samples += 1

    dyn = {}
    for axis in ("ax", "ay", "az"):
        static = work[axis].rolling(
            window=n_samples, center=True, min_periods=max(3, n_samples // 10)
        ).mean()
        dyn[axis] = work[axis] - static

    stacked = np.vstack([dyn["ax"], dyn["ay"], dyn["az"]])
    if method == "vedba":
        activity = np.sqrt(np.nansum(stacked ** 2, axis=0))
    elif method == "odba":
        activity = np.nansum(np.abs(stacked), axis=0)
    else:
        raise ValueError(f"Unknown method: {method!r} (expected 'vedba' or 'odba')")

    # Rows where every axis was NaN must stay NaN rather than collapse to 0.
    all_nan = np.all(~np.isfinite(stacked), axis=0)
    activity = np.where(all_nan, np.nan, activity)

    out = pd.DataFrame({
        "datetime": work["datetime"],
        "activity": activity,
        "dyn_x": dyn["ax"],
        "dyn_y": dyn["ay"],
        "dyn_z": dyn["az"],
    })
    out.attrs["sampling_rate_hz"] = fs
    out.attrs["static_window_samples"] = n_samples
    out.attrs["method"] = method
    return out


# --------------------------------------------------------------------------
# Wildlife Computers "Series" aggregation, reverse-engineered
# --------------------------------------------------------------------------
# Derived empirically by comparing 167205-Series.csv against its parent
# 167205-ArchivedSeries.csv (1748 full windows, 99.94% exact agreement):
#
#   bin width   = 450 s (7.5 min) = 150 ArchivedSeries samples at 3 s
#   timestamp   = bin START, aligned to the 7.5-minute grid
#   window      = [t, t + 450s)  -- left-closed, right-open
#   Activity    = mean over the bin of ||(Ax, Ay, Az)||   (vector magnitude)
#   ARange      = peak-to-peak of that same magnitude
#   Depth       = mean(Depth);        DRange = peak-to-peak(Depth)
#   Temperature = mean(Temperature);  TRange = peak-to-peak(Temperature)
#
# Depth/Temperature are rounded to 1 dp in the export; Activity is full
# precision. Note Activity is the *raw* magnitude, so it includes the ~1 g
# gravity component -- it is NOT a dynamic-acceleration (ODBA/VeDBA) metric,
# and values therefore sit near 1.0 for a stationary animal.
WC_SERIES_INTERVAL_SECONDS = 450.0


def compute_series_aggregates(
    acc_df: pd.DataFrame,
    interval: str | float = "450s",
    depth_df: pd.DataFrame | None = None,
    temperature_df: pd.DataFrame | None = None,
    depth_col: str = "depth",
    temperature_col: str = "temp_ext",
    label: bool = True,
) -> pd.DataFrame:
    """
    Reproduce the Wildlife Computers Series aggregation at any interval.

    Uses the method decoded from this deployment's own Series export (see
    module notes above), so results are directly comparable to the vendor
    product while allowing a different bin width.

    Parameters
    ----------
    acc_df : DataFrame with datetime, ax, ay, az.
    interval : pandas offset string (e.g. '450s', '30min', '1h') or seconds.
        Defaults to the vendor's 450 s so output matches Series.csv.
    depth_df, temperature_df : optional frames to aggregate alongside.
    label : if True, bins are labelled by their START timestamp, matching the
        vendor convention. Set False for centre-of-bin labels.

    Returns
    -------
    DataFrame with datetime plus activity/activity_range (and depth/temperature
    equivalents when supplied), one row per bin.

    Note: `activity` here includes gravity, exactly as the vendor's does. For a
    gravity-removed activity index use :func:`compute_activity_from_acc`.
    """
    required = {"datetime", "ax", "ay", "az"}
    missing = required - set(acc_df.columns)
    if missing:
        raise ValueError(f"acc_df is missing columns: {sorted(missing)}")

    rule = f"{float(interval)}s" if isinstance(interval, (int, float)) else interval

    work = acc_df[["datetime", "ax", "ay", "az"]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    magnitude = np.sqrt(
        work["ax"].to_numpy() ** 2
        + work["ay"].to_numpy() ** 2
        + work["az"].to_numpy() ** 2
    )
    work = work.assign(_magnitude=magnitude).set_index("datetime")

    grouped = work["_magnitude"].resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "activity": grouped.mean(),
        "activity_range": grouped.apply(
            lambda s: s.max() - s.min() if s.notna().any() else np.nan
        ),
        "n_samples": grouped.count(),
    })

    for frame, col, prefix in (
        (depth_df, depth_col, "depth"),
        (temperature_df, temperature_col, "temperature"),
    ):
        if frame is None:
            continue
        if col not in frame.columns:
            raise ValueError(f"Expected column {col!r} in the {prefix} frame.")
        other = frame[["datetime", col]].copy()
        other["datetime"] = pd.to_datetime(other["datetime"])
        g = other.set_index("datetime")[col].resample(rule, label="left", closed="left")
        out[prefix] = g.mean()
        out[f"{prefix}_range"] = g.apply(
            lambda s: s.max() - s.min() if s.notna().any() else np.nan
        )

    out = out.reset_index().rename(columns={"index": "datetime"})
    if not label:
        offset = pd.Timedelta(rule) / 2
        out["datetime"] = out["datetime"] + offset

    out.attrs["interval"] = rule
    out.attrs["gravity_included"] = True
    return out


def estimate_sampling_rate_hz(datetime_series: pd.Series) -> float:
    """Median-interval sampling rate in Hz (robust to gaps and duplicates)."""
    dt = pd.to_datetime(datetime_series)
    diffs = dt.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float("nan")
    return 1.0 / float(np.median(diffs))


def resample_regular(
    df: pd.DataFrame,
    value_col: str,
    rule: str = "10min",
    how: str = "mean",
) -> pd.DataFrame:
    """
    Resample onto a regular grid — a prerequisite for wavelet analysis, which
    assumes uniform spacing. Returns columns [datetime, <value_col>].
    """
    work = df[["datetime", value_col]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    grouped = work.set_index("datetime")[value_col].resample(rule)
    series = getattr(grouped, how)()
    return pd.DataFrame({"datetime": series.index, value_col: series.to_numpy()})


def correlate_temperature_activity(
    temp_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    rule: str = "1h",
    temp_col: str = "temp_ext",
    activity_col: str = "activity",
    max_lag_hours: float = 48.0,
) -> dict:
    """
    Correlate temperature against activity on a common time grid.

    Reports Pearson and Spearman on aligned bins, plus a lagged
    cross-correlation. Lag matters here: a shark changing depth alters its
    thermal environment, so temperature may follow activity rather than drive
    it, and the sign/direction of any lag is the interesting part.

    Returns a dict with 'n', 'pearson_r', 'spearman_rho', both p-values,
    'best_lag_hours', 'best_lag_r', and a 'lag_curve' DataFrame.
    """
    from scipy.stats import pearsonr, spearmanr

    temp_rs = resample_regular(temp_df, temp_col, rule=rule)
    act_rs = resample_regular(activity_df, activity_col, rule=rule)
    merged = temp_rs.merge(act_rs, on="datetime", how="inner").dropna()
    if len(merged) < 10:
        raise ValueError(f"Only {len(merged)} aligned bins; need at least 10.")

    t = merged[temp_col].to_numpy()
    a = merged[activity_col].to_numpy()
    pear_r, pear_p = pearsonr(t, a)
    spear_r, spear_p = spearmanr(t, a)

    step_hours = pd.Timedelta(rule).total_seconds() / 3600.0
    max_lag_steps = int(round(max_lag_hours / step_hours))
    lags, corrs = [], []
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        shifted = np.roll(a, lag)
        # Exclude wrapped samples so the correlation uses only real overlap.
        if lag > 0:
            valid_t, valid_a = t[lag:], shifted[lag:]
        elif lag < 0:
            valid_t, valid_a = t[:lag], shifted[:lag]
        else:
            valid_t, valid_a = t, shifted
        if len(valid_t) < 10:
            continue
        lags.append(lag * step_hours)
        corrs.append(float(np.corrcoef(valid_t, valid_a)[0, 1]))

    lag_curve = pd.DataFrame({"lag_hours": lags, "r": corrs})
    best_idx = int(np.nanargmax(np.abs(lag_curve["r"].to_numpy())))
    return {
        "n": int(len(merged)),
        "bin": rule,
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho": float(spear_r),
        "spearman_p": float(spear_p),
        "best_lag_hours": float(lag_curve["lag_hours"].iloc[best_idx]),
        "best_lag_r": float(lag_curve["r"].iloc[best_idx]),
        "lag_curve": lag_curve,
        "merged": merged,
    }


def wavelet_power_spectrum(
    df: pd.DataFrame,
    value_col: str = "activity",
    rule: str = "30min",
    min_period_hours: float = 2.0,
    max_period_hours: float = 24.0 * 40,
    n_scales: int = 128,
    wavelet: str = DEFAULT_WAVELET,
) -> dict:
    """
    Continuous wavelet transform of a time series, returned in period space.

    Uses a complex Morlet CWT so power can be read directly against period
    (hours), which is how diel/tidal/lunar structure is interpreted. The
    series is resampled to a regular grid, mean-filled across short gaps, and
    linearly detrended so a long-term drift does not dominate low frequencies.

    Returns dict with 'power' (n_scales x n_times), 'periods_hours',
    'times', 'global_power' (time-averaged), 'cone_of_influence_hours',
    and 'dt_hours'.
    """
    import pywt

    regular = resample_regular(df, value_col, rule=rule)
    dt_hours = pd.Timedelta(rule).total_seconds() / 3600.0

    values = regular[value_col].to_numpy(dtype="float64")
    finite = np.isfinite(values)
    if finite.sum() < 32:
        raise ValueError(f"Only {int(finite.sum())} finite samples; need >= 32.")
    # Interpolate interior gaps, then fill any remaining edges with the mean.
    values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    values = np.where(np.isfinite(values), values, np.nanmean(values))
    values = values - np.nanmean(values)
    # Remove linear trend so slow drift doesn't masquerade as a long period.
    x = np.arange(values.size, dtype="float64")
    slope, intercept = np.polyfit(x, values, 1)
    values = values - (slope * x + intercept)

    centre_freq = pywt.central_frequency(wavelet)
    periods = np.logspace(
        np.log10(min_period_hours), np.log10(max_period_hours), n_scales
    )
    # period = scale * dt / centre_frequency  =>  scale = period * cf / dt
    scales = periods * centre_freq / dt_hours
    keep = scales >= 1.0
    if not keep.all():
        periods, scales = periods[keep], scales[keep]

    coeffs, _ = pywt.cwt(values, scales, wavelet, sampling_period=dt_hours)
    power = np.abs(coeffs) ** 2

    # Cone of influence: edge-affected region where the wavelet overlaps the
    # series boundary. Periods beyond this are not trustworthy.
    total_hours = values.size * dt_hours
    coi = total_hours / (2.0 * np.sqrt(2.0))

    return {
        "power": power,
        "periods_hours": periods,
        "times": regular["datetime"].to_numpy(),
        "global_power": power.mean(axis=1),
        "cone_of_influence_hours": float(coi),
        "dt_hours": dt_hours,
        "n_samples": int(values.size),
        "wavelet": wavelet,
    }


def dominant_periods(spectrum: dict, top_n: int = 5) -> pd.DataFrame:
    """
    Rank periods by time-averaged wavelet power, flagging those beyond the
    cone of influence as unreliable.
    """
    periods = spectrum["periods_hours"]
    power = spectrum["global_power"]
    coi = spectrum["cone_of_influence_hours"]

    # Local maxima in the global power spectrum.
    peak_idx = [
        i for i in range(1, len(power) - 1)
        if power[i] > power[i - 1] and power[i] >= power[i + 1]
    ]
    if not peak_idx:
        peak_idx = [int(np.argmax(power))]
    peak_idx.sort(key=lambda i: power[i], reverse=True)

    rows = []
    for i in peak_idx[:top_n]:
        rows.append({
            "period_hours": float(periods[i]),
            "period_days": float(periods[i] / 24.0),
            "power": float(power[i]),
            "within_cone_of_influence": bool(periods[i] <= coi),
            "nearest_known_cycle": _nearest_known_cycle(periods[i]),
        })
    return pd.DataFrame(rows)


def _nearest_known_cycle(period_hours: float, tolerance: float = 0.25) -> str:
    """Label a period with the nearest biological cycle within ±tolerance."""
    best_label, best_ratio = "", np.inf
    for label, hours in PERIODS_OF_INTEREST_HOURS.items():
        ratio = abs(period_hours - hours) / hours
        if ratio < best_ratio:
            best_label, best_ratio = label, ratio
    return best_label if best_ratio <= tolerance else "—"
