"""
Smoothing pipeline for light/SST-based geolocation tracks (e.g. Wildlife
Computers GPE3), as an alternative to the Argos/GPS aniMotum CRW workflow.

Why this is separate from aniMotum
----------------------------------
aniMotum's state-space models are parameterised for Argos/GPS error structure:
small, well-characterised error ellipses supplied per fix (semi-major axis,
semi-minor axis, orientation) or an Argos location class. Light-based
geolocation has none of that and its error is one to two orders of magnitude
larger -- roughly 0.5-1 deg in longitude (driven by the timing of dawn/dusk)
and considerably worse in latitude, degrading severely near the equinoxes when
day length is nearly uniform with latitude.

Feeding such positions to an Argos-tuned SSM would be badly misspecified: the
model would treat coarse light estimates as precise observations and produce
an over-confident track. Instead this module applies explicit, documented
smoothing with anisotropic (latitude-weaker-than-longitude) error assumptions.

The output mirrors the aniMotum workflow's contract -- regularised fixes on a
fixed time step -- so downstream map/track consumers can use either source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default 1-sigma observation error for light-based geolocation, in degrees.
# Latitude is deliberately much weaker than longitude; see module docstring.
DEFAULT_LON_SIGMA_DEG = 0.7
DEFAULT_LAT_SIGMA_DEG = 2.0

# Equinox dates (month, day) where latitude from day length is least reliable.
_EQUINOXES = ((3, 20), (9, 22))
# Within this many days of an equinox, inflate latitude sigma by the factor below.
EQUINOX_WINDOW_DAYS = 21
EQUINOX_LAT_INFLATION = 2.5


def days_from_equinox(timestamps: pd.Series) -> pd.Series:
    """Days to the nearest equinox for each timestamp (0 = on the equinox)."""
    dt = pd.to_datetime(timestamps)
    out = []
    for value in dt:
        if pd.isna(value):
            out.append(np.nan)
            continue
        best = None
        for year in (value.year - 1, value.year, value.year + 1):
            for month, day in _EQUINOXES:
                delta = abs((value - pd.Timestamp(year=year, month=month, day=day)).days)
                best = delta if best is None else min(best, delta)
        out.append(float(best))
    return pd.Series(out, index=dt.index, dtype="float64")


def latitude_sigma(
    timestamps: pd.Series,
    base_sigma_deg: float = DEFAULT_LAT_SIGMA_DEG,
) -> pd.Series:
    """
    Per-fix latitude 1-sigma, inflated near equinoxes.

    Day length varies little with latitude around an equinox, so light-based
    latitude is least trustworthy there and is explicitly down-weighted.
    """
    distance = days_from_equinox(timestamps)
    inflation = np.where(
        distance <= EQUINOX_WINDOW_DAYS,
        EQUINOX_LAT_INFLATION,
        1.0,
    )
    return pd.Series(base_sigma_deg * inflation, index=distance.index, dtype="float64")


def _weighted_moving_average(
    values: np.ndarray,
    times_hours: np.ndarray,
    weights: np.ndarray,
    window_hours: float,
) -> np.ndarray:
    """
    Gaussian-kernel weighted moving average over an irregular time series.

    Each observation is weighted by 1/sigma^2 and by a Gaussian in time, so
    coarse fixes and distant fixes both contribute less. Operating in the time
    domain (not sample index) keeps the smoothing physically meaningful when
    the fix interval varies -- here from ~2 h to 12 h.
    """
    smoothed = np.empty_like(values, dtype="float64")
    tau = max(float(window_hours), 1e-6)
    for i in range(values.size):
        dt = times_hours - times_hours[i]
        kernel = np.exp(-0.5 * (dt / tau) ** 2)
        w = kernel * weights
        total = w.sum()
        smoothed[i] = np.dot(w, values) / total if total > 0 else values[i]
    return smoothed


def smooth_geolocation_track(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    window_hours: float = 24.0,
    lon_sigma_deg: float = DEFAULT_LON_SIGMA_DEG,
    lat_sigma_deg: float = DEFAULT_LAT_SIGMA_DEG,
    max_speed_kmh: float | None = 10.0,
) -> pd.DataFrame:
    """
    Smooth a light-based geolocation track.

    Steps
    -----
    1. Drop rows without a timestamp or position, sort by time.
    2. Optionally flag fixes implying an implausible speed from the previous
       accepted fix, and exclude them from the smoother.
    3. Apply an error-weighted, time-domain Gaussian smoother separately to
       latitude and longitude, using anisotropic sigmas (latitude weaker,
       inflated near equinoxes).

    Returns a copy with added ``latitude_smoothed`` / ``longitude_smoothed``
    columns, the per-fix sigmas used, and an ``outlier`` flag. Original
    positions are preserved so the smoothing can always be audited.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=[datetime_col, lat_col, lon_col,
                     "latitude_smoothed", "longitude_smoothed",
                     "lat_sigma_deg", "lon_sigma_deg", "outlier"]
        )

    work = df.copy()
    work[datetime_col] = pd.to_datetime(work[datetime_col], errors="coerce")
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = (
        work.dropna(subset=[datetime_col, lat_col, lon_col])
        .sort_values(datetime_col)
        .reset_index(drop=True)
    )
    if work.empty:
        return work

    work["lat_sigma_deg"] = latitude_sigma(work[datetime_col], lat_sigma_deg).to_numpy()
    work["lon_sigma_deg"] = float(lon_sigma_deg)

    work["outlier"] = False
    if max_speed_kmh is not None and len(work) > 1:
        work["outlier"] = _flag_speed_outliers(
            work[datetime_col].to_numpy(),
            work[lat_col].to_numpy(),
            work[lon_col].to_numpy(),
            max_speed_kmh,
        )

    keep = ~work["outlier"].to_numpy()
    if keep.sum() < 2:
        # Not enough clean fixes to smooth; pass positions through unchanged.
        work["latitude_smoothed"] = work[lat_col]
        work["longitude_smoothed"] = work[lon_col]
        return work

    times = work.loc[keep, datetime_col]
    hours = (times - times.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    lat_w = 1.0 / np.square(work.loc[keep, "lat_sigma_deg"].to_numpy())
    lon_w = 1.0 / np.square(work.loc[keep, "lon_sigma_deg"].to_numpy())

    lat_s = _weighted_moving_average(
        work.loc[keep, lat_col].to_numpy(), hours, lat_w, window_hours
    )
    lon_s = _weighted_moving_average(
        work.loc[keep, lon_col].to_numpy(), hours, lon_w, window_hours
    )

    work["latitude_smoothed"] = np.nan
    work["longitude_smoothed"] = np.nan
    work.loc[keep, "latitude_smoothed"] = lat_s
    work.loc[keep, "longitude_smoothed"] = lon_s
    return work


def _flag_speed_outliers(
    times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    max_speed_kmh: float,
) -> np.ndarray:
    """
    Flag fixes requiring an implausible swim speed from the last accepted fix.

    Compared against the last *accepted* fix rather than the immediately
    previous one, so a single wild position doesn't cascade into flagging the
    good fixes that follow it.
    """
    flags = np.zeros(lat.size, dtype=bool)
    if lat.size < 2:
        return flags
    anchor = 0
    for i in range(1, lat.size):
        hours = (times[i] - times[anchor]) / np.timedelta64(1, "h")
        if hours <= 0:
            flags[i] = True
            continue
        km = _haversine_km(lat[anchor], lon[anchor], lat[i], lon[i])
        if km / hours > max_speed_kmh:
            flags[i] = True
        else:
            anchor = i
    return flags


def _haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def regularize_track(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    lat_col: str = "latitude_smoothed",
    lon_col: str = "longitude_smoothed",
    step: str = "3h",
) -> pd.DataFrame:
    """
    Resample a smoothed track onto a fixed time step by time interpolation,
    matching the regularised output of the aniMotum CRW workflow (3 h default).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["datetime", "latitude", "longitude"])

    work = df.dropna(subset=[datetime_col, lat_col, lon_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["datetime", "latitude", "longitude"])

    work = work.set_index(pd.to_datetime(work[datetime_col])).sort_index()
    grid = pd.date_range(
        work.index.min().ceil(step), work.index.max().floor(step), freq=step
    )
    if len(grid) == 0:
        return pd.DataFrame(columns=["datetime", "latitude", "longitude"])

    union = work.index.union(grid)
    lat = work[lat_col].reindex(union).interpolate(method="time").reindex(grid)
    lon = work[lon_col].reindex(union).interpolate(method="time").reindex(grid)
    return pd.DataFrame(
        {"datetime": grid, "latitude": lat.to_numpy(), "longitude": lon.to_numpy()}
    )
