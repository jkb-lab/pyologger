"""
Solar and astronomical utilities for behavior analysis.

Provides functions to calculate sunrise/sunset times, lunar phase,
and other astronomical context for deployment timeseries analysis.
"""

import math
from datetime import date as dt_date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

try:
    from astral import LocationInfo
    from astral.sun import sun as astral_sun
    ASTRAL_AVAILABLE = True
except ImportError:
    ASTRAL_AVAILABLE = False

try:
    from scipy.interpolate import PchipInterpolator
    PCHIP_AVAILABLE = True
except ImportError:
    PCHIP_AVAILABLE = False


def sunrise_sunset_local_hours(day_value, lat, lon, tz_name):
    """
    Calculate sunrise and sunset times in local hours for a given day and location.
    
    Parameters
    ----------
    day_value : date, datetime, or pd.Timestamp
        The date for which to calculate sunrise/sunset
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    tz_name : str
        IANA timezone name (e.g., 'Africa/Nairobi')
    
    Returns
    -------
    tuple of float
        (sunrise_hour, sunset_hour) in local time as decimal hours (0-24).
        Returns (np.nan, np.nan) if location is invalid or calculation fails.
    
    Notes
    -----
    Uses astral library if available, otherwise falls back to simplified
    astronomical calculations based on NOAA solar calculator algorithms.
    """
    if not np.isfinite(lat) or not np.isfinite(lon):
        return np.nan, np.nan
    
    if isinstance(day_value, pd.Timestamp):
        day_obj = day_value.date()
    elif isinstance(day_value, datetime):
        day_obj = day_value.date()
    elif isinstance(day_value, dt_date):
        day_obj = day_value
    else:
        day_obj = pd.Timestamp(day_value).date()

    # Prefer astral library for accurate calculations
    if ASTRAL_AVAILABLE:
        try:
            loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz_name)
            sun_times = astral_sun(loc.observer, date=day_obj, tzinfo=ZoneInfo(tz_name))
            sunrise_dt = sun_times.get('sunrise')
            sunset_dt = sun_times.get('sunset')
            sunrise_h = np.nan if sunrise_dt is None else sunrise_dt.hour + (sunrise_dt.minute / 60.0) + (sunrise_dt.second / 3600.0)
            sunset_h = np.nan if sunset_dt is None else sunset_dt.hour + (sunset_dt.minute / 60.0) + (sunset_dt.second / 3600.0)
            return sunrise_h, sunset_h
        except Exception:
            pass

    # Fallback: simplified NOAA algorithm
    def _calc(is_sunrise):
        n = day_obj.timetuple().tm_yday
        lng_hour = lon / 15.0
        approx_t = n + (((6.0 if is_sunrise else 18.0) - lng_hour) / 24.0)
        m = (0.9856 * approx_t) - 3.289
        l = m + (1.916 * math.sin(math.radians(m))) + (0.020 * math.sin(math.radians(2 * m))) + 282.634
        l = l % 360.0
        ra = math.degrees(math.atan(0.91764 * math.tan(math.radians(l)))) % 360.0
        l_quadrant = (math.floor(l / 90.0)) * 90.0
        ra_quadrant = (math.floor(ra / 90.0)) * 90.0
        ra = (ra + (l_quadrant - ra_quadrant)) / 15.0
        sin_dec = 0.39782 * math.sin(math.radians(l))
        cos_dec = math.cos(math.asin(sin_dec))
        cos_h = (math.cos(math.radians(90.833)) - (sin_dec * math.sin(math.radians(lat)))) / (cos_dec * math.cos(math.radians(lat)))
        if cos_h < -1.0 or cos_h > 1.0:
            return np.nan
        h = (360.0 - math.degrees(math.acos(cos_h))) if is_sunrise else math.degrees(math.acos(cos_h))
        h /= 15.0
        local_mean_t = h + ra - (0.06571 * approx_t) - 6.622
        ut = (local_mean_t - lng_hour) % 24.0
        utc_dt = datetime(day_obj.year, day_obj.month, day_obj.day, tzinfo=timezone.utc) + timedelta(hours=ut)
        local_dt = utc_dt.astimezone(ZoneInfo(tz_name))
        return local_dt.hour + (local_dt.minute / 60.0) + (local_dt.second / 3600.0)

    return _calc(True), _calc(False)


def format_local_hour_hhmm(hour_value):
    """
    Format decimal hour as HH:MM string.
    
    Parameters
    ----------
    hour_value : float
        Decimal hour (0-24)
    
    Returns
    -------
    str
        Formatted time string 'HH:MM' or 'NA' if invalid
    """
    if not np.isfinite(hour_value):
        return 'NA'
    minutes_total = int(round(float(hour_value) * 60.0))
    hh = (minutes_total // 60) % 24
    mm = minutes_total % 60
    return f'{hh:02d}:{mm:02d}'


def format_local_hour_label(hour_value, tz_name):
    """
    Format decimal hour with timezone abbreviation.
    
    Parameters
    ----------
    hour_value : float
        Decimal hour (0-24)
    tz_name : str
        IANA timezone name
    
    Returns
    -------
    str
        Formatted time string 'HH:MM TZ' or 'NA' if invalid
    """
    if not np.isfinite(hour_value):
        return 'NA'
    minutes_total = int(round(float(hour_value) * 60.0))
    hh = (minutes_total // 60) % 24
    mm = minutes_total % 60
    tz_label = datetime(2020, 1, 1, tzinfo=ZoneInfo(tz_name)).tzname() or tz_name
    return f'{hh:02d}:{mm:02d} {tz_label}'


def smooth_xy(x_vals, y_vals, points_per_hour=16):
    """
    Smooth x-y data using PCHIP interpolation or linear interpolation.
    
    Parameters
    ----------
    x_vals : array-like
        X coordinates
    y_vals : array-like
        Y coordinates
    points_per_hour : int, default=16
        Density of interpolated points per unit x
    
    Returns
    -------
    tuple of ndarray
        (x_smooth, y_smooth) arrays
    """
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    if len(x_arr) < 2:
        return x_arr, y_arr
    x_dense = np.linspace(float(x_arr.min()), float(x_arr.max()), int((x_arr.max() - x_arr.min()) * points_per_hour) + 1)
    if PCHIP_AVAILABLE and len(np.unique(x_arr)) >= 2:
        y_dense = PchipInterpolator(x_arr, y_arr)(x_dense)
    else:
        y_dense = np.interp(x_dense, x_arr, y_arr)
    return x_dense, y_dense


def smooth_bounds(x_vals, y1_vals, y2_vals, points_per_hour=16):
    """
    Smooth upper and lower bounds using PCHIP interpolation.
    
    Parameters
    ----------
    x_vals : array-like
        X coordinates
    y1_vals : array-like
        Lower bound Y coordinates
    y2_vals : array-like
        Upper bound Y coordinates
    points_per_hour : int, default=16
        Density of interpolated points per unit x
    
    Returns
    -------
    tuple of ndarray
        (x_smooth, y1_smooth, y2_smooth) arrays
    """
    x_arr = np.asarray(x_vals, dtype=float)
    y1_arr = np.asarray(y1_vals, dtype=float)
    y2_arr = np.asarray(y2_vals, dtype=float)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y1_arr) & np.isfinite(y2_arr)
    x_arr = x_arr[finite_mask]
    y1_arr = y1_arr[finite_mask]
    y2_arr = y2_arr[finite_mask]
    if len(x_arr) < 2:
        return x_arr, y1_arr, y2_arr
    x_dense = np.linspace(float(x_arr.min()), float(x_arr.max()), int((x_arr.max() - x_arr.min()) * points_per_hour) + 1)
    if PCHIP_AVAILABLE and len(np.unique(x_arr)) >= 2:
        y1_dense = PchipInterpolator(x_arr, y1_arr)(x_dense)
        y2_dense = PchipInterpolator(x_arr, y2_arr)(x_dense)
    else:
        y1_dense = np.interp(x_dense, x_arr, y1_arr)
        y2_dense = np.interp(x_dense, x_arr, y2_arr)
    return x_dense, y1_dense, y2_dense


def smooth_stacked_boundaries(x_vals, boundary_matrix, points_per_hour=16):
    """
    Smooth multiple stacked boundaries (for area charts) using PCHIP interpolation.
    
    Parameters
    ----------
    x_vals : array-like
        X coordinates
    boundary_matrix : ndarray, shape (n_layers, n_points)
        Matrix where each row is a cumulative boundary
    points_per_hour : int, default=16
        Density of interpolated points per unit x
    
    Returns
    -------
    tuple of ndarray
        (x_smooth, boundary_matrix_smooth) where boundary_matrix_smooth has shape (n_layers, n_dense_points)
    """
    x_arr = np.asarray(x_vals, dtype=float)
    bounds = np.asarray(boundary_matrix, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != len(x_arr):
        return x_arr, bounds
    finite_mask = np.isfinite(x_arr)
    for row in bounds:
        finite_mask &= np.isfinite(row)
    x_arr = x_arr[finite_mask]
    bounds = bounds[:, finite_mask]
    if len(x_arr) < 2:
        return x_arr, bounds
    x_dense = np.linspace(float(x_arr.min()), float(x_arr.max()), int((x_arr.max() - x_arr.min()) * points_per_hour) + 1)
    dense_rows = []
    for row in bounds:
        if PCHIP_AVAILABLE and len(np.unique(x_arr)) >= 2:
            dense_rows.append(PchipInterpolator(x_arr, row)(x_dense))
        else:
            dense_rows.append(np.interp(x_dense, x_arr, row))
    return x_dense, np.vstack(dense_rows)
