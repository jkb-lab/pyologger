import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly_resampler import FigureWidgetResampler, FigureResampler, register_plotly_resampler, MinMaxLTTB
from pyologger.process_data.sampling import *
from datetime import timedelta, datetime
from contextlib import contextmanager
import json
import os
import pandas as pd
import numpy as np
import pytz
import html
import re
import fnmatch
import matplotlib.pyplot as plt
import threading
import time
from pyologger.utils.cluster_colors import (
    build_ordered_cluster_color_map,
    looks_like_ordered_cluster_key,
    parse_cluster_rank_from_key,
)
from pyologger.utils.solar_utils import (
    sunrise_sunset_local_hours,
    format_local_hour_hhmm,
    format_local_hour_label,
    smooth_xy,
    smooth_bounds,
    smooth_stacked_boundaries,
)


_BATHY_DATASET_CACHE = {}


@contextmanager
def _progress_log(operation_name: str, interval_s: float = 60.0):
    stop_event = threading.Event()
    start = time.monotonic()

    def _worker():
        while not stop_event.wait(interval_s):
            elapsed = int(time.monotonic() - start)
            print(f"[timing] {operation_name} still running after {elapsed}s")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        elapsed = time.monotonic() - start
        print(f"[timing] {operation_name} completed in {elapsed:.1f}s")


def _open_bathy_dataset_cached(gebco_nc_path):
    import xarray as xr

    cache_key = os.path.abspath(os.path.expanduser(str(gebco_nc_path)))
    cached = _BATHY_DATASET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _progress_log(f"open bathymetry dataset header {cache_key}"):
        ds = xr.open_dataset(cache_key, chunks="auto", cache=False)

        z_name = "elevation" if "elevation" in ds.data_vars else list(ds.data_vars)[0]
        z = ds[z_name]
        lon_name = "lon" if "lon" in z.coords else ("x" if "x" in z.coords else None)
        lat_name = "lat" if "lat" in z.coords else ("y" if "y" in z.coords else None)
        if lon_name is None or lat_name is None:
            raise ValueError(f"Could not find lon/lat coords. coords={list(z.coords)}")

        # Inspect only a small header slice to infer coordinate ordering.
        lat_probe = np.asarray(z[lat_name].isel({lat_name: slice(0, 2)}).values, dtype=float)
        lat_ascending = True
        if lat_probe.size >= 2 and np.isfinite(lat_probe).all():
            lat_ascending = bool(lat_probe[1] > lat_probe[0])

        cached = {
            "path": cache_key,
            "ds": ds,
            "z_name": z_name,
            "lon_name": lon_name,
            "lat_name": lat_name,
            "lat_ascending": lat_ascending,
        }
        _BATHY_DATASET_CACHE[cache_key] = cached
    return cached


def plot_activity_budget_stacked(
    budget_df,
    value_col,
    behavior_col,
    deployment_col="deployment_id",
    group_col=None,
    metadata_label_col=None,
    behavior_order=None,
    color_map=None,
    title="Activity Budget",
):
    """
    Build a stacked activity-budget bar chart from a prepared long-format table.
    Required columns are deployment, behavior, and value; group and metadata columns are optional.
    """
    if budget_df is None or len(budget_df) == 0:
        raise ValueError("budget_df is empty")

    df = budget_df.copy()
    required_cols = {deployment_col, behavior_col, value_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required budget_df columns: {missing}")

    if behavior_order:
        present = [b for b in behavior_order if b in df[behavior_col].astype(str).unique().tolist()]
        extras = [b for b in df[behavior_col].astype(str).unique().tolist() if b not in present]
        df[behavior_col] = pd.Categorical(df[behavior_col].astype(str), categories=present + sorted(extras), ordered=True)
    else:
        df[behavior_col] = df[behavior_col].astype(str)

    if metadata_label_col and metadata_label_col in df.columns:
        df["__x_label"] = df[deployment_col].astype(str) + "<br>(" + df[metadata_label_col].fillna("NA").astype(str) + ")"
    else:
        df["__x_label"] = df[deployment_col].astype(str)

    plot_kwargs = dict(
        data_frame=df,
        x="__x_label",
        y=value_col,
        color=behavior_col,
        title=title,
        color_discrete_map=(color_map or None),
    )
    if group_col and group_col in df.columns:
        plot_kwargs["facet_col"] = group_col

    fig = px.bar(**plot_kwargs)
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        xaxis_title="Deployment",
        yaxis_title=value_col,
        legend_title=behavior_col,
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_daily_activity_actogram(
    hourly_behavior_df,
    hourly_odba_df=None,
    deployment_lat_lon_map=None,
    timezone_map=None,
    behavior_order=None,
    behavior_colors=None,
    keep_incomplete_days=True,
    figsize=(18, None),
    title="Hourly Activity Budget Actograms with Solar Context",
):
    """
    Create matplotlib actogram plots showing daily behavior budgets with solar context.
    
    This function creates multi-day activity plots (actograms) where each row represents
    one day, with hours on the x-axis. Behaviors are shown as stacked proportions with
    smooth interpolation, sunrise/sunset markers are overlaid, and optionally ODBA
    (overall dynamic body acceleration) is plotted as a separate track.
    
    Parameters
    ----------
    hourly_behavior_df : pd.DataFrame
        Hourly behavior budget data with columns:
        - deployment_id : str
        - date_local : date
        - hour_of_day : int (0-23)
        - final_behavior or behavior columns : float (percent of observed hour)
    hourly_odba_df : pd.DataFrame, optional
        Hourly ODBA summary data with columns:
        - deployment_id : str
        - date_local : date
        - hour_of_day : int
        - odba_mean : float
    deployment_lat_lon_map : dict, optional
        Mapping from deployment_id to (lat, lon) tuple for sunrise/sunset calculations
    timezone_map : dict, optional
        Mapping from deployment_id to timezone name (e.g., 'Africa/Nairobi')
    behavior_order : list of str, optional
        Ordered list of behaviors for stacking (bottom to top)
    behavior_colors : dict, optional
        Mapping from behavior name to matplotlib color
    keep_incomplete_days : bool, default=True
        If True, plot all available days (including days with missing hours).
        If False, keep only complete days with all 24 local hours present.
    figsize : tuple, default=(18, None)
        Figure size (width, height). Height is auto-calculated if None.
    title : str, optional
        Overall figure title
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure with actogram subplots
    
    Notes
    -----
    - Each deployment gets its own subplot
    - Days are stacked vertically (newest at top by default)
    - Sunrise/sunset times are calculated using pyologger.utils.solar_utils
    - Night periods are shaded in gray
    - Mean sunrise/sunset across days shown as dotted lines
    - ODBA (if provided) is plotted as line+shade below the actogram
    
    Examples
    --------
    >>> fig = plot_daily_activity_actogram(
    ...     hourly_behavior_df=budget_df,
    ...     hourly_odba_df=odba_df,
    ...     deployment_lat_lon_map={'dep001': (-1.3, 36.8)},
    ...     timezone_map={'dep001': 'Africa/Nairobi'},
    ...     behavior_order=['Resting', 'Calm', 'Active'],
    ...     behavior_colors={'Resting': '#1A608B', 'Calm': '#8D7392', 'Active': '#B12A40'}
    ... )
    >>> plt.show()
    """
    import math
    from zoneinfo import ZoneInfo
    
    if hourly_behavior_df is None or len(hourly_behavior_df) == 0:
        raise ValueError('hourly_behavior_df is empty')
    
    # Identify behavior columns
    exclude_cols = {'deployment_id', 'group', 'date_local', 'hour_of_day', 'hour_local', 
                    'observed_hour_seconds', 'pct_of_observed_hour', 'chunk_seconds'}
    behavior_cols = [c for c in hourly_behavior_df.columns if c not in exclude_cols]
    
    if not behavior_cols:
        raise ValueError('No behavior columns found in hourly_behavior_df')
    
    # Setup defaults
    if behavior_order is None:
        behavior_order = sorted(behavior_cols)
    else:
        # Add any extra columns not in behavior_order
        behavior_order = [b for b in behavior_order if b in behavior_cols]
        behavior_order += sorted([b for b in behavior_cols if b not in behavior_order])
    
    if behavior_colors is None:
        # Default color palette
        behavior_colors = {
            'Resting': '#1A608B',
            'Calm': '#8D7392',
            'Vigilant': '#C9C27A',
            'Drinking': '#8BE4CA',
            'Feeding': '#F4B183',
            'Active': '#B12A40',
            'Unknown': '#D3D3D3',
        }
    
    deployment_lat_lon_map = deployment_lat_lon_map or {}
    timezone_map = timezone_map or {}
    
    # Group by deployment
    deployments = sorted(hourly_behavior_df['deployment_id'].dropna().unique())
    if len(deployments) == 0:
        raise ValueError('No deployments found in hourly_behavior_df')
    
    ncols = 2
    nrows = int(math.ceil(len(deployments) / ncols))
    fig_height = max(7, 4.2 * nrows) if figsize[1] is None else figsize[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], fig_height), sharex=True, squeeze=False)
    axes_flat = axes.ravel()
    
    # Build legend handles
    legend_handles = []
    for behavior in behavior_order:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=behavior_colors.get(behavior, '#999999'), 
                                           alpha=0.9, label=behavior))
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='#9ca3af', alpha=0.18, 
                                        label='Mean activity (all days)'))
    if hourly_odba_df is not None and len(hourly_odba_df) > 0:
        legend_handles.append(plt.Line2D([0], [0], color='black', linewidth=1.4, label='Mean ODBA'))
        legend_handles.append(plt.Line2D([0], [0], color='black', linewidth=1.0, linestyle=':', label='ODBA ±1 SD'))
    legend_handles.append(plt.Line2D([0], [0], color='#f59e0b', linewidth=1.0, linestyle='--', label='Sunrise'))
    legend_handles.append(plt.Line2D([0], [0], color='#1d4ed8', linewidth=1.0, linestyle='--', label='Sunset'))
    legend_handles.append(plt.Line2D([0], [0], color='#f59e0b', linewidth=1.2, linestyle=':', label='Median sunrise'))
    legend_handles.append(plt.Line2D([0], [0], color='#1d4ed8', linewidth=1.2, linestyle=':', label='Median sunset'))
    
    for ax, dep_id in zip(axes_flat, deployments):
        dep_data = hourly_behavior_df[hourly_behavior_df['deployment_id'] == dep_id].copy()
        days = sorted(dep_data['date_local'].dropna().unique())

        if not keep_incomplete_days and len(days) > 0:
            complete_days = (
                dep_data.groupby('date_local')['hour_of_day']
                .nunique()
                .loc[lambda s: s >= 24]
                .index
            )
            dep_data = dep_data[dep_data['date_local'].isin(complete_days)].copy()
            days = sorted(dep_data['date_local'].dropna().unique())
        
        if len(days) == 0:
            ax.set_visible(False)
            continue
        
        # Get location and timezone
        dep_lat, dep_lon = deployment_lat_lon_map.get(dep_id, (np.nan, np.nan))
        tz_name = timezone_map.get(dep_id, 'UTC')
        
        # Layout parameters
        row_gap = 1.02
        stack_height = 1.0
        x_hours = np.arange(24, dtype=float)
        
        # Calculate ODBA track dimensions if available
        day_area = ((len(days) - 1) * row_gap) + stack_height
        odba_height = day_area * (2.0 / 3.0) if hourly_odba_df is not None else 0
        odba_band_bottom = -0.10 if odba_height > 0 else 0
        odba_band_top = odba_band_bottom - odba_height if odba_height > 0 else 0
        
        sunrise_values = []
        sunset_values = []
        
        # Plot each day
        for day_idx, day_val in enumerate(days):
            baseline = day_idx * row_gap
            day_slice = dep_data[dep_data['date_local'] == day_val]
            
            # Build pivot with behavior proportions
            pivot = day_slice.pivot_table(
                index='hour_of_day',
                columns='final_behavior' if 'final_behavior' in day_slice.columns else behavior_cols[0],
                values='pct_of_observed_hour' if 'pct_of_observed_hour' in day_slice.columns else behavior_cols[0],
                aggfunc='sum',
                fill_value=0,
            ).reindex(range(24), fill_value=0) if 'final_behavior' in day_slice.columns else day_slice.groupby('hour_of_day')[behavior_cols].sum().reindex(range(24), fill_value=0)
            
            # Order columns
            cols = [c for c in behavior_order if c in pivot.columns]
            cols += [c for c in pivot.columns if c not in cols]
            pivot = pivot[cols]
            observed_hour_mask = pivot.sum(axis=1).to_numpy(dtype=float) > 0
            missing_hour_mask = ~observed_hour_mask
            
            # Calculate sunrise/sunset
            sunrise_h, sunset_h = sunrise_sunset_local_hours(day_val, dep_lat, dep_lon, tz_name)
            if np.isfinite(sunrise_h):
                sunrise_values.append(float(sunrise_h))
                ax.vlines(sunrise_h, baseline, baseline + stack_height, colors='#f59e0b', 
                         linewidth=1.0, linestyles='--', alpha=0.85, zorder=2)
                ax.text(sunrise_h + 0.08, baseline + 0.18, format_local_hour_hhmm(sunrise_h), 
                       color='white', fontsize=6.5, va='center', ha='left', zorder=6)
            if np.isfinite(sunset_h):
                sunset_values.append(float(sunset_h))
                ax.vlines(sunset_h, baseline, baseline + stack_height, colors='#1d4ed8', 
                         linewidth=1.0, linestyles='--', alpha=0.85, zorder=2)
                ax.text(sunset_h + 0.08, baseline + 0.82, format_local_hour_hhmm(sunset_h), 
                       color='white', fontsize=6.5, va='center', ha='left', zorder=6)
            
            # Build stacked area boundaries
            boundary_rows = [np.full(len(x_hours), baseline, dtype=float)]
            running = boundary_rows[0].copy()
            for behavior in cols:
                running = running + (pivot[behavior].to_numpy(dtype=float) / 100.0) * stack_height
                boundary_rows.append(running.copy())
            
            # Smooth boundaries
            x_fill, smoothed_bounds = smooth_stacked_boundaries(x_hours, np.vstack(boundary_rows))
            smoothed_bounds = np.clip(smoothed_bounds, baseline, baseline + stack_height)
            smoothed_bounds[0, :] = baseline
            smoothed_bounds[-1, :] = baseline + stack_height
            smoothed_bounds = np.maximum.accumulate(smoothed_bounds, axis=0)
            
            # Fill stacked areas
            for behavior_idx, behavior in enumerate(cols):
                ax.fill_between(
                    x_fill,
                    smoothed_bounds[behavior_idx],
                    smoothed_bounds[behavior_idx + 1],
                    color=behavior_colors.get(behavior, '#999999'),
                    alpha=0.9,
                    linewidth=0,
                )

            # If an hour has no observed data at all, force it to render as white (no data)
            # instead of inheriting the last behavior color from boundary smoothing.
            if missing_hour_mask.any():
                ax.fill_between(
                    x_hours,
                    baseline,
                    baseline + stack_height,
                    where=missing_hour_mask,
                    step='mid',
                    color='white',
                    alpha=1.0,
                    linewidth=0,
                    zorder=1.5,
                )
            
            ax.plot(x_hours, baseline + np.zeros_like(x_hours), color='#d0d0d0', linewidth=0.5, alpha=0.6)
        
        # Add mean activity budget band (if multiple days)
        if len(days) > 1:
            mean_behavior = dep_data.groupby('hour_of_day')[behavior_cols].mean().reindex(range(24), fill_value=0.0)
            cols_avg = [c for c in behavior_order if c in mean_behavior.columns]
            mean_missing_hour_mask = ~(mean_behavior.sum(axis=1).to_numpy(dtype=float) > 0)
            top_band_span = abs(odba_band_bottom - odba_band_top) if odba_height > 0 else stack_height * 0.3
            activity_bounds = [np.full(len(x_hours), odba_band_top if odba_height > 0 else -0.5, dtype=float)]
            activity_running = activity_bounds[0].copy()
            for behavior in cols_avg:
                activity_running = activity_running + (mean_behavior[behavior].to_numpy(dtype=float) / 100.0) * top_band_span
                activity_bounds.append(activity_running.copy())
            x_fill_top, activity_smoothed = smooth_stacked_boundaries(x_hours, np.vstack(activity_bounds))
            for behavior_idx, behavior in enumerate(cols_avg):
                ax.fill_between(
                    x_fill_top,
                    activity_smoothed[behavior_idx],
                    activity_smoothed[behavior_idx + 1],
                    color=behavior_colors.get(behavior, '#999999'),
                    alpha=0.18,
                    linewidth=0,
                    zorder=1,
                )

            if mean_missing_hour_mask.any():
                band_base = odba_band_top if odba_height > 0 else -0.5
                band_top = band_base + top_band_span
                ax.fill_between(
                    x_hours,
                    band_base,
                    band_top,
                    where=mean_missing_hour_mask,
                    step='mid',
                    color='white',
                    alpha=1.0,
                    linewidth=0,
                    zorder=1.2,
                )
        
        # Add ODBA if available
        if hourly_odba_df is not None and len(hourly_odba_df) > 0 and odba_height > 0:
            dep_odba = hourly_odba_df[hourly_odba_df['deployment_id'] == dep_id]
            if len(dep_odba) > 0:
                odba_hourly = dep_odba.groupby('hour_of_day')['odba_mean'].agg(mean='mean', std='std').reset_index()
                odba_hourly = odba_hourly.set_index('hour_of_day').reindex(range(24))
                odba_hourly['std'] = odba_hourly['std'].fillna(0.0)
                
                scale_lo = 0.0
                scale_hi = max(1.0, odba_hourly['mean'].max() + odba_hourly['std'].max())
                
                def _scale_odba(vals):
                    vals = np.asarray(vals, dtype=float)
                    scaled = np.full(vals.shape, np.nan, dtype=float)
                    finite_mask = np.isfinite(vals)
                    scaled[finite_mask] = odba_band_bottom - ((vals[finite_mask] - scale_lo) / (scale_hi - scale_lo)) * (odba_band_bottom - odba_band_top)
                    return scaled
                
                mean_y = _scale_odba(odba_hourly['mean'].to_numpy(dtype=float))
                low_y = _scale_odba((odba_hourly['mean'] - odba_hourly['std']).to_numpy(dtype=float))
                high_y = _scale_odba((odba_hourly['mean'] + odba_hourly['std']).to_numpy(dtype=float))
                
                x_odba, low_y_smooth, high_y_smooth = smooth_bounds(x_hours, low_y, high_y)
                _, mean_y_smooth = smooth_xy(x_hours, mean_y)
                
                ax.plot(x_odba, low_y_smooth, color='black', linewidth=1.0, linestyle=':', alpha=0.65, zorder=3)
                ax.plot(x_odba, high_y_smooth, color='black', linewidth=1.0, linestyle=':', alpha=0.65, zorder=3)
                ax.plot(x_odba, mean_y_smooth, color='black', linewidth=1.4, alpha=0.95, zorder=4)
        
        # Add median sunrise/sunset lines and night shading
        if sunrise_values and sunset_values:
            sunrise_med = float(np.nanmedian(sunrise_values))
            sunset_med = float(np.nanmedian(sunset_values))
            
            # Shade night periods
            if odba_height > 0:
                upper_night_top = odba_band_top - 1.68
                upper_night_bottom = odba_band_top - 0.02
                for x0, x1 in [(-0.5, sunrise_med), (sunset_med, 23.5)]:
                    if x1 > x0:
                        ax.fill_between(np.array([x0, x1]), upper_night_top, upper_night_bottom, 
                                      color='#6b7280', alpha=0.48, linewidth=0, zorder=2)
            
            # Median lines
            ax.vlines(sunrise_med, odba_band_top if odba_height > 0 else 0, 
                     (len(days) - 1) * row_gap + stack_height,
                     colors='#f59e0b', linewidth=1.2, linestyles=':', alpha=0.95, zorder=3)
            ax.vlines(sunset_med, odba_band_top if odba_height > 0 else 0, 
                     (len(days) - 1) * row_gap + stack_height,
                     colors='#1d4ed8', linewidth=1.2, linestyles=':', alpha=0.95, zorder=3)
        
        # Configure axes
        ax.set_xlim(-0.5, 23.5)
        bottom_lim = odba_band_top - 0.05 if odba_height > 0 else -0.5
        ax.set_ylim(bottom_lim, (len(days) - 1) * row_gap + stack_height + 0.15)
        ax.set_yticks([idx * row_gap + (stack_height / 2.0) for idx in range(len(days))])
        ax.set_yticklabels([pd.Timestamp(day).strftime('%Y-%m-%d') for day in days], fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.18)
        
        # Title with sunrise/sunset info
        sunrise_label = format_local_hour_label(float(np.nanmedian(sunrise_values)), tz_name) if sunrise_values else 'NA'
        sunset_label = format_local_hour_label(float(np.nanmedian(sunset_values)), tz_name) if sunset_values else 'NA'
        loc_txt = f' | {dep_lat:.2f}, {dep_lon:.2f}' if np.isfinite(dep_lat) and np.isfinite(dep_lon) else ''
        ax.set_title(f'{dep_id} | Sunrise {sunrise_label} | Sunset {sunset_label}{loc_txt}', fontsize=11)
    
    # Hide unused subplots
    for ax in axes_flat[len(deployments):]:
        ax.set_visible(False)
    
    # X-axis labels on bottom row
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xticks(np.arange(0, 24, 2))
            ax.set_xlabel('Hour of local day')
    
    # Overall title and legend
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.legend(handles=legend_handles, loc='upper center', ncol=min(len(legend_handles), 9), 
              fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    
    return fig


def plot_continuous_activity_actogram_from_hourly_budget(
    hourly_budget_df,
    *,
    behavior_col="state_label_canonical",
    value_col="pct_of_observed_hour",
    title="Continuous Activity Actogram",
    behavior_order=None,
    behavior_colors=None,
    deployment_order=None,
):
    """
    Build a continuous (multi-day concatenated) actogram from hourly budget rows.
    Expected columns:
      deployment_id, local_date, hour_of_day, behavior_col, value_col
    Optional sorting controls:
      deployment_order (explicit deployment_id order)
      deployment_sort_value (lower values are plotted first)
    Optional columns for solar overlays:
      sunrise_local_hour, sunset_local_hour
    """
    if hourly_budget_df is None or len(hourly_budget_df) == 0:
        raise ValueError("hourly_budget_df is empty")
    required = {"deployment_id", "local_date", "hour_of_day", behavior_col, value_col}
    missing = [col for col in required if col not in hourly_budget_df.columns]
    if missing:
        raise ValueError(f"Missing required hourly budget columns: {missing}")

    work = hourly_budget_df.copy()
    work["deployment_id"] = work["deployment_id"].astype(str)
    work["local_date"] = pd.to_datetime(work["local_date"], errors="coerce").dt.date
    work["hour_of_day"] = pd.to_numeric(work["hour_of_day"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["local_date", "hour_of_day"])
    if work.empty:
        raise ValueError("No valid hourly rows available after datetime coercion.")

    if behavior_order is None:
        behavior_order = sorted(work[behavior_col].dropna().astype(str).unique().tolist())
    else:
        seen = set(work[behavior_col].dropna().astype(str).unique().tolist())
        behavior_order = [b for b in behavior_order if b in seen] + sorted([b for b in seen if b not in set(behavior_order)])
    if not behavior_order:
        raise ValueError("No behaviors found for continuous actogram.")

    if behavior_colors is None:
        behavior_colors = {
            "Resting": "#1A608B",
            "Calm": "#8D7392",
            "Vigilant": "#C9C27A",
            "Drinking": "#8BE4CA",
            "Feeding": "#F4B183",
            "Active": "#B12A40",
            "Unknown": "#D3D3D3",
        }

    if deployment_order is not None:
        requested = [str(d) for d in deployment_order]
        present = set(work["deployment_id"].astype(str).unique().tolist())
        deployments = [d for d in requested if d in present]
        remaining = sorted([d for d in present if d not in set(deployments)])
        deployments = deployments + remaining
    elif "deployment_sort_value" in work.columns:
        dep_order_df = (
            work[["deployment_id", "deployment_sort_value"]]
            .dropna(subset=["deployment_id"])
            .copy()
        )
        dep_order_df["deployment_sort_value"] = pd.to_numeric(
            dep_order_df["deployment_sort_value"], errors="coerce"
        )
        dep_order_df = (
            dep_order_df.groupby("deployment_id", as_index=False)["deployment_sort_value"]
            .min()
            .sort_values(["deployment_sort_value", "deployment_id"], ascending=[True, True])
        )
        deployments = dep_order_df["deployment_id"].astype(str).tolist()
    else:
        deployments = sorted(work["deployment_id"].unique().tolist())
    n_rows = max(1, len(deployments))
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, max(5.5, 3.8 * n_rows)), squeeze=False)
    axes_flat = axes.ravel()

    for ax, dep_id in zip(axes_flat, deployments):
        dep = work.loc[work["deployment_id"] == dep_id].copy()
        days = sorted(dep["local_date"].dropna().unique().tolist())
        if not days:
            ax.set_visible(False)
            continue
        day_index = {day: idx for idx, day in enumerate(days)}
        dep["x_hour"] = dep.apply(lambda row: 24.0 * day_index.get(row["local_date"], 0) + float(row["hour_of_day"]), axis=1)

        pivot = dep.pivot_table(index="x_hour", columns=behavior_col, values=value_col, aggfunc="sum", fill_value=0.0).sort_index()
        pivot = pivot.reindex(columns=behavior_order, fill_value=0.0)
        if pivot.empty:
            ax.set_visible(False)
            continue

        # Ensure a complete hourly axis so smoothing is stable and gaps are rendered cleanly.
        x_min = int(np.floor(float(pivot.index.min())))
        x_max = int(np.ceil(float(pivot.index.max())))
        x_vals = np.arange(x_min, x_max + 1, dtype=float)
        pivot = pivot.reindex(x_vals, fill_value=0.0)

        boundary_rows = [np.zeros(len(x_vals), dtype=float)]
        running = boundary_rows[0].copy()
        for behavior in behavior_order:
            running = running + (pivot[behavior].to_numpy(dtype=float) / 100.0)
            boundary_rows.append(running.copy())

        x_fill, smoothed_bounds = smooth_stacked_boundaries(x_vals, np.vstack(boundary_rows))
        smoothed_bounds = np.clip(smoothed_bounds, 0.0, 1.0)
        smoothed_bounds[0, :] = 0.0
        smoothed_bounds[-1, :] = 1.0
        smoothed_bounds = np.maximum.accumulate(smoothed_bounds, axis=0)

        for behavior_idx, behavior in enumerate(behavior_order):
            ax.fill_between(
                x_fill,
                smoothed_bounds[behavior_idx],
                smoothed_bounds[behavior_idx + 1],
                color=behavior_colors.get(behavior, "#999999"),
                alpha=0.9,
                linewidth=0,
            )

        if {"sunrise_local_hour", "sunset_local_hour"}.issubset(dep.columns):
            solar = dep[["local_date", "sunrise_local_hour", "sunset_local_hour"]].drop_duplicates("local_date")
            for row in solar.itertuples(index=False):
                day_i = day_index.get(getattr(row, "local_date"))
                if day_i is None:
                    continue
                x0 = 24.0 * day_i
                x1 = x0 + 24.0
                sunrise_h = pd.to_numeric(pd.Series([getattr(row, "sunrise_local_hour")]), errors="coerce").iloc[0]
                sunset_h = pd.to_numeric(pd.Series([getattr(row, "sunset_local_hour")]), errors="coerce").iloc[0]
                if pd.notna(sunrise_h):
                    xs = x0 + float(sunrise_h)
                    ax.axvline(xs, color="#f59e0b", linewidth=0.8, linestyle="--", alpha=0.6)
                    ax.axvspan(x0, xs, color="#6b7280", alpha=0.10, linewidth=0)
                if pd.notna(sunset_h):
                    xs = x0 + float(sunset_h)
                    ax.axvline(xs, color="#1d4ed8", linewidth=0.8, linestyle="--", alpha=0.6)
                    ax.axvspan(xs, x1, color="#6b7280", alpha=0.10, linewidth=0)

        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Fraction")
        tick_locs = [(24.0 * idx) for idx in range(len(days))]
        tick_labels = [str(day) for day in days]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(str(dep_id), fontsize=10)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)

    for ax in axes_flat[len(deployments):]:
        ax.set_visible(False)
    axes_flat[-1].set_xlabel("Local date (continuous timeline)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=behavior_colors.get(label, "#999999"), alpha=0.9, label=label) for label in behavior_order]
    fig.legend(handles=handles, loc="upper center", ncol=min(7, max(1, len(handles))), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_continuous_activity_with_event_timing(
    hourly_budget_df,
    event_df,
    *,
    behavior_col="state_label_canonical",
    value_col="pct_of_observed_hour",
    event_label_col="state_label",
    event_start_col="start_datetime",
    event_end_col="end_datetime",
    title="Continuous Activity Actogram with Event Timing",
    behavior_order=None,
    behavior_colors=None,
):
    """
    Build a continuous activity actogram with event-of-interest timing overlaid.
    Event markers are placed at each event midpoint on the day/night timeline.
    """
    fig = plot_continuous_activity_actogram_from_hourly_budget(
        hourly_budget_df=hourly_budget_df,
        behavior_col=behavior_col,
        value_col=value_col,
        title=title,
        behavior_order=behavior_order,
        behavior_colors=behavior_colors,
    )
    if event_df is None or len(event_df) == 0:
        return fig

    work = hourly_budget_df.copy()
    work["deployment_id"] = work["deployment_id"].astype(str)
    work["local_date"] = pd.to_datetime(work["local_date"], errors="coerce").dt.date
    day_index_by_dep = {}
    for dep_id, dep_sub in work.groupby("deployment_id", sort=False):
        day_index_by_dep[str(dep_id)] = {
            day: idx for idx, day in enumerate(sorted(dep_sub["local_date"].dropna().unique().tolist()))
        }

    event_work = event_df.copy()
    required_cols = {"deployment_id", event_start_col}
    if not required_cols.issubset(event_work.columns):
        return fig
    event_work["deployment_id"] = event_work["deployment_id"].astype(str)
    event_work[event_start_col] = pd.to_datetime(event_work[event_start_col], errors="coerce")
    if event_end_col in event_work.columns:
        event_work[event_end_col] = pd.to_datetime(event_work[event_end_col], errors="coerce")
    else:
        event_work[event_end_col] = event_work[event_start_col]
    event_work = event_work.dropna(subset=[event_start_col]).copy()
    if event_work.empty:
        return fig
    event_work["event_midpoint"] = event_work[event_start_col] + (
        (event_work[event_end_col] - event_work[event_start_col]).fillna(pd.Timedelta(seconds=0)) / 2
    )
    event_work["event_date"] = event_work["event_midpoint"].dt.date
    event_work["event_hour"] = (
        event_work["event_midpoint"].dt.hour
        + event_work["event_midpoint"].dt.minute / 60.0
        + event_work["event_midpoint"].dt.second / 3600.0
    )
    event_work["event_label"] = (
        event_work.get(event_label_col, pd.Series("event", index=event_work.index))
        .fillna("event")
        .astype(str)
    )
    event_work["duration_h"] = (
        (event_work[event_end_col] - event_work[event_start_col]).dt.total_seconds().fillna(0.0).clip(lower=0.0) / 3600.0
    )
    if len(fig.axes) == 0:
        return fig
    axes = fig.axes
    for ax in axes:
        title_text = str(ax.get_title() or "")
        dep_id = title_text.split(" | ")[0].strip() if " | " in title_text else title_text.strip()
        if dep_id not in day_index_by_dep:
            continue
        dep_events = event_work[event_work["deployment_id"] == dep_id].copy()
        if dep_events.empty:
            continue
        day_index = day_index_by_dep[dep_id]
        dep_events["x_hour"] = dep_events.apply(
            lambda row: (24.0 * day_index.get(row["event_date"], -1.0)) + float(row["event_hour"]),
            axis=1,
        )
        dep_events = dep_events[dep_events["x_hour"] >= 0].copy()
        if dep_events.empty:
            continue
        labels = sorted(dep_events["event_label"].dropna().astype(str).unique().tolist())
        cmap = plt.get_cmap("tab10")
        color_map = {label: cmap(i % 10) for i, label in enumerate(labels)}
        max_dur = float(dep_events["duration_h"].max()) if not dep_events["duration_h"].empty else 0.0
        for label in labels:
            subset = dep_events[dep_events["event_label"] == label]
            if subset.empty:
                continue
            if max_dur > 0:
                sizes = 30.0 + 120.0 * (subset["duration_h"].astype(float) / max_dur)
            else:
                sizes = 45.0
            ax.scatter(
                subset["x_hour"].to_numpy(dtype=float),
                np.full(len(subset), 1.03, dtype=float),
                s=sizes,
                c=[color_map[label]],
                marker="o",
                edgecolors="white",
                linewidths=0.4,
                alpha=0.9,
                label=f"Event: {label}",
                clip_on=False,
                zorder=7,
            )
        y0, y1 = ax.get_ylim()
        if y1 < 1.12:
            ax.set_ylim(y0, 1.12)
    return fig


def _normalize_plot_datetime(values, tz_name="UTC"):
    dt = pd.to_datetime(values, errors="coerce")
    if not isinstance(dt, pd.Series):
        dt = pd.Series(dt)
    try:
        if getattr(dt.dt, "tz", None) is None:
            return dt.dt.tz_localize(tz_name)
        return dt.dt.tz_convert(tz_name)
    except Exception:
        return dt


def _resolve_plot_lat_lon(lat_lon_source, deployment_info=None):
    info = dict(deployment_info or {})
    if isinstance(lat_lon_source, (tuple, list)) and len(lat_lon_source) >= 2:
        return float(lat_lon_source[0]), float(lat_lon_source[1])
    if isinstance(lat_lon_source, dict):
        lat_candidates = [
            lat_lon_source.get("lat"),
            lat_lon_source.get("latitude"),
            lat_lon_source.get("Deployment Latitude"),
        ]
        lon_candidates = [
            lat_lon_source.get("lon"),
            lat_lon_source.get("longitude"),
            lat_lon_source.get("Deployment Longitude"),
        ]
        for lat_value in lat_candidates:
            for lon_value in lon_candidates:
                lat_num = pd.to_numeric(pd.Series([lat_value]), errors="coerce").iloc[0]
                lon_num = pd.to_numeric(pd.Series([lon_value]), errors="coerce").iloc[0]
                if pd.notna(lat_num) and pd.notna(lon_num):
                    return float(lat_num), float(lon_num)
    lat_num = pd.to_numeric(
        pd.Series([info.get("Deployment Latitude") or info.get("latitude") or info.get("lat")]),
        errors="coerce",
    ).iloc[0]
    lon_num = pd.to_numeric(
        pd.Series([info.get("Deployment Longitude") or info.get("longitude") or info.get("lon")]),
        errors="coerce",
    ).iloc[0]
    return float(lat_num) if pd.notna(lat_num) else np.nan, float(lon_num) if pd.notna(lon_num) else np.nan


def _canonical_rest_label_map(state_label_map):
    default_map = {
        "surface_sleep": {
            "putative_rest.surface_sleep",
            "find_rest.surface_sleep",
            "surface_sleep",
        },
        "benthic_sleep": {
            "putative_rest.benthic_sleep",
            "find_rest.long_flat",
            "long_flat",
            "benthic_sleep",
        },
        "drift_sleep": {
            "putative_rest.drift_sleep",
            "find_rest.long_drift",
            "long_drift",
            "drift_sleep",
        },
    }
    label_to_category = {}
    raw = state_label_map or {}
    if isinstance(raw, dict):
        if raw and all(isinstance(v, (list, tuple, set)) for v in raw.values()):
            for category, labels in raw.items():
                for label in labels:
                    label_to_category[str(label)] = str(category)
        else:
            for label, category in raw.items():
                label_to_category[str(label)] = str(category)
    for category, labels in default_map.items():
        for label in labels:
            label_to_category.setdefault(str(label), category)
    return label_to_category


def _event_style_color_for_key(color_mapping, event_key, fallback):
    mapping = color_mapping or {}
    if isinstance(mapping.get(event_key), str) and mapping.get(event_key).strip():
        return mapping.get(event_key).strip()
    event_styles = mapping.get("__event_styles__", {}) or {}
    style = event_styles.get(str(event_key), {}) if isinstance(event_styles, dict) else {}
    style_color = style.get("color") or style.get("shade_color")
    if isinstance(style_color, str) and style_color.strip():
        return style_color.strip()
    return fallback


def _pick_depth_frame(data_pkl):
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    for signal_name, preferred_cols in [("depth", ["depth"]), ("pressure", ["pressure"])]:
        frame = signal_data.get(signal_name)
        if not isinstance(frame, pd.DataFrame) or frame.empty or "datetime" not in frame.columns:
            continue
        numeric_cols = [col for col in preferred_cols if col in frame.columns]
        if not numeric_cols:
            numeric_cols = [
                col for col in frame.columns
                if col != "datetime" and pd.api.types.is_numeric_dtype(frame[col])
            ]
        if numeric_cols:
            out = frame[["datetime", numeric_cols[0]]].copy()
            out.columns = ["datetime", "depth_value"]
            return out
    raise ValueError("continuous_daily_activity_plot requires a `depth` or `pressure` signal with a datetime column.")


def _pick_location_frame(data_pkl):
    signal_data = getattr(data_pkl, "signal_data", {}) or {}
    frame = signal_data.get("location")
    if not isinstance(frame, pd.DataFrame) or frame.empty or "datetime" not in frame.columns:
        return pd.DataFrame(columns=["datetime", "latitude", "longitude"])
    lat_col = next((col for col in ["latitude", "lat", "Latitude", "Lat"] if col in frame.columns), None)
    lon_col = next((col for col in ["longitude", "lon", "Longitude", "Long"] if col in frame.columns), None)
    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=["datetime", "latitude", "longitude"])
    out = frame[["datetime", lat_col, lon_col]].copy()
    out.columns = ["datetime", "latitude", "longitude"]
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    return out.dropna(subset=["datetime", "latitude", "longitude"]).reset_index(drop=True)


def _segment_rest_index(seg_df, state_label_map):
    if seg_df is None or seg_df.empty:
        return pd.DataFrame(
            columns=[
                "segment_rank",
                "state_name",
                "start_datetime",
                "end_datetime",
                "segment_midpoint_datetime",
                "duration_s",
                "drift_rate_ms",
                "local_date",
                "local_hour",
                "rest_category",
            ]
        )
    label_map = _canonical_rest_label_map(state_label_map)
    work = seg_df.copy()
    work["state_name"] = work.get("state_name", work.get("label_name", pd.Series("", index=work.index))).fillna("").astype(str)
    work["base_nominal_class"] = work.get("base_nominal_class", pd.Series("", index=work.index)).fillna("").astype(str)
    work["keep_filtered"] = pd.Series(work.get("keep_filtered", False)).fillna(False).astype(bool)
    work["post_measured_keep"] = pd.Series(
        work.get("post_measured_keep", work.get("base_keep_filtered", False))
    ).fillna(False).astype(bool)
    work["rest_category"] = work["state_name"].map(label_map)
    work["rest_category"] = work["rest_category"].where(
        work["rest_category"].notna(),
        work["base_nominal_class"].map(label_map),
    )
    work["plot_state_name"] = work["state_name"]
    work["plot_state_name"] = work["plot_state_name"].where(
        work["plot_state_name"].map(label_map).notna(),
        work["base_nominal_class"].map(
            {
                "surface_sleep": "rest.surface",
                "long_flat": "rest.benthic",
                "long_drift": "rest.drift",
            }
        ),
    )
    work = work.loc[work["post_measured_keep"] & work["rest_category"].notna()].copy()
    for col in ["start_datetime", "end_datetime", "segment_midpoint_datetime"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")
    if "segment_midpoint_datetime" not in work.columns:
        work["segment_midpoint_datetime"] = work["start_datetime"] + (work["end_datetime"] - work["start_datetime"]) / 2
    if "duration_s" not in work.columns:
        work["duration_s"] = (work["end_datetime"] - work["start_datetime"]).dt.total_seconds()
    keep_cols = [
        "segment_rank",
        "state_name",
        "start_datetime",
        "end_datetime",
        "segment_midpoint_datetime",
        "duration_s",
        "drift_rate_ms",
        "plot_state_name",
        "rest_category",
    ]
    for col in keep_cols:
        if col not in work.columns:
            work[col] = pd.NA
    return work[keep_cols].sort_values(
        ["segment_midpoint_datetime", "start_datetime", "segment_rank"],
        na_position="last",
    ).reset_index(drop=True)


def _canonical_buoyancy_phase(phase_value):
    text = str(phase_value or "").strip().lower()
    if not text:
        return ""
    if text in {"negative", "pre_positive", "never_positive"}:
        return "negative_buoyancy"
    if text in {"neutral", "transition", "neutral_buoyancy"}:
        return "neutral_buoyancy"
    if text in {"positive", "post_positive", "always_positive"}:
        return "positive_buoyancy"
    return text


def _build_buoyancy_track_fig(seg_df, deployment_id, reference_max_start_depth_m=500.0, tolerance_ms=0.15):
    if seg_df is None or seg_df.empty:
        return go.Figure()
    work = seg_df.copy()
    for col in ["start_datetime", "end_datetime", "segment_midpoint_datetime"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")
    phase_series = None
    for col in ["inferred_buoyancy_phase", "inferred_trip_phase"]:
        if col in work.columns:
            phase_series = work[col]
            break
    if phase_series is None:
        return go.Figure()
    work["buoyancy_phase"] = phase_series.map(_canonical_buoyancy_phase)
    work = work.loc[work["buoyancy_phase"].isin(["negative_buoyancy", "neutral_buoyancy", "positive_buoyancy"])].copy()
    if work.empty:
        return go.Figure()
    if "segment_midpoint_datetime" not in work.columns:
        work["segment_midpoint_datetime"] = work["start_datetime"] + (work["end_datetime"] - work["start_datetime"]) / 2
    work = work.sort_values(["segment_midpoint_datetime", "start_datetime"], na_position="last").reset_index(drop=True)
    phase_order = ["negative_buoyancy", "neutral_buoyancy", "positive_buoyancy"]
    phase_colors = {
        "negative_buoyancy": "#000004",
        "neutral_buoyancy": "#B5367A",
        "positive_buoyancy": "#FCFDBF",
    }
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.58],
    )
    for _, row in work.iterrows():
        start_dt = row.get("start_datetime")
        end_dt = row.get("end_datetime")
        phase_name = str(row.get("buoyancy_phase") or "")
        if pd.isna(start_dt) or pd.isna(end_dt) or not phase_name:
            continue
        fig.add_vrect(
            x0=start_dt,
            x1=end_dt,
            fillcolor=phase_colors.get(phase_name, "#888888"),
            line_width=0,
            opacity=0.95,
            layer="below",
            row=1,
            col=1,
        )
    midpoint_df = work.dropna(subset=["segment_midpoint_datetime"]).copy()
    midpoint_df["phase_index"] = midpoint_df["buoyancy_phase"].map({name: idx for idx, name in enumerate(phase_order)})
    fig.add_trace(
        go.Scatter(
            x=midpoint_df["segment_midpoint_datetime"],
            y=midpoint_df["phase_index"],
            mode="markers",
            marker=dict(
                size=7,
                color=[phase_colors.get(v, "#888888") for v in midpoint_df["buoyancy_phase"]],
                line=dict(color="rgba(0,0,0,0.22)", width=0.5),
            ),
            customdata=midpoint_df[["buoyancy_phase"]].values,
            hovertemplate="%{x}<br>%{customdata[0]}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    candidate_df = work.loc[
        work.get("post_measured_keep", pd.Series(False, index=work.index)).fillna(False).astype(bool)
        & work.get("base_nominal_class", pd.Series("", index=work.index)).astype(str).eq("long_drift")
        & pd.to_numeric(work.get("start_depth_m", pd.Series(np.nan, index=work.index)), errors="coerce").lt(
            float(reference_max_start_depth_m)
        )
    ].copy()
    if "segment_midpoint_datetime" in candidate_df.columns and "inferred_buoyancy_value" in candidate_df.columns:
        candidate_df["inferred_buoyancy_value"] = pd.to_numeric(candidate_df["inferred_buoyancy_value"], errors="coerce")
        candidate_df["drift_rate_ms"] = pd.to_numeric(candidate_df.get("drift_rate_ms"), errors="coerce")
        candidate_df = candidate_df.dropna(subset=["segment_midpoint_datetime", "inferred_buoyancy_value"]).sort_values(
            "segment_midpoint_datetime"
        )
        if not candidate_df.empty:
            candidate_df["upper_band"] = candidate_df["inferred_buoyancy_value"] + float(tolerance_ms)
            candidate_df["lower_band"] = candidate_df["inferred_buoyancy_value"] - float(tolerance_ms)
            x_hours = (
                candidate_df["segment_midpoint_datetime"].astype("int64").to_numpy(dtype=float) / 3_600_000_000_000.0
            )
            x_smooth, center_smooth = smooth_xy(x_hours, candidate_df["inferred_buoyancy_value"].to_numpy(dtype=float))
            _, lower_smooth, upper_smooth = smooth_bounds(
                x_hours,
                candidate_df["lower_band"].to_numpy(dtype=float),
                candidate_df["upper_band"].to_numpy(dtype=float),
            )
            x_smooth_dt = pd.to_datetime(x_smooth, unit="h", origin="unix")
            fig.add_trace(
                go.Scatter(
                    x=x_smooth_dt,
                    y=upper_smooth,
                    mode="lines",
                    line=dict(color="rgba(80,80,80,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_smooth_dt,
                    y=lower_smooth,
                    mode="lines",
                    line=dict(color="rgba(80,80,80,0)"),
                    fill="tonexty",
                    fillcolor="rgba(60,60,60,0.22)",
                    hoverinfo="skip",
                    name="Allowed around inferred buoyancy",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_smooth_dt,
                    y=center_smooth,
                    mode="lines",
                    line=dict(color="#2F4858", width=2.6),
                    hoverinfo="skip",
                    name="Smoothed inferred buoyancy",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=candidate_df["segment_midpoint_datetime"],
                    y=candidate_df["inferred_buoyancy_value"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=candidate_df["drift_rate_ms"],
                        colorscale="Magma",
                        showscale=False,
                        line=dict(color="rgba(0,0,0,0.18)", width=0.5),
                    ),
                    customdata=(
                        candidate_df[["segment_rank", "drift_rate_ms"]].values
                        if {"segment_rank", "drift_rate_ms"} <= set(candidate_df.columns)
                        else None
                    ),
                    hovertemplate=(
                        "%{x}<br>Inferred buoyancy %{y:.3f} m/s"
                        "<br>Drift rate %{customdata[1]:.3f} m/s"
                        "<br>Rank %{customdata[0]}<extra></extra>"
                    ),
                    name="Inferred buoyancy",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
    fig.update_layout(
        template="plotly_white",
        title=f"{deployment_id} Buoyancy Shift",
        xaxis_title="Local time across trip",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["negative", "neutral", "positive"],
            range=[-0.6, 2.6],
            title="Buoyancy state",
        ),
        yaxis2=dict(title="Inferred buoyancy (m/s)"),
        height=280,
        margin=dict(l=40, r=20, t=45, b=25),
    )
    return fig


def _build_daily_rest_map_fig(data_pkl, rest_index_df, deployment_id, tz_name):
    location_df = _pick_location_frame(data_pkl)
    if location_df.empty:
        return go.Figure()
    location_df["datetime"] = _normalize_plot_datetime(location_df["datetime"], tz_name)
    location_df = location_df.dropna(subset=["datetime", "latitude", "longitude"]).sort_values("datetime").reset_index(drop=True)
    if location_df.empty:
        return go.Figure()

    work = location_df.copy()
    sample_step_s = work["datetime"].diff().dt.total_seconds().median()
    if pd.isna(sample_step_s) or float(sample_step_s) <= 0:
        sample_step_s = 3600.0
    work["duration_h"] = work["datetime"].shift(-1).sub(work["datetime"]).dt.total_seconds().fillna(sample_step_s) / 3600.0
    work["local_date"] = work["datetime"].dt.date
    work["rest_hours"] = 0.0

    if isinstance(rest_index_df, pd.DataFrame) and not rest_index_df.empty:
        for _, row in rest_index_df.loc[rest_index_df["status_keep_filtered"].fillna(False).astype(bool)].iterrows():
            start_dt = row.get("start_datetime")
            end_dt = row.get("end_datetime")
            if pd.isna(start_dt) or pd.isna(end_dt):
                continue
            mask = (work["datetime"] >= start_dt) & (work["datetime"] <= end_dt)
            work.loc[mask, "rest_hours"] = work.loc[mask, "duration_h"]

    daily = (
        work.groupby("local_date", dropna=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            rest_hours=("rest_hours", "sum"),
            track_hours=("duration_h", "sum"),
        )
        .reset_index()
    )
    daily = daily.dropna(subset=["latitude", "longitude"])
    if daily.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["longitude"],
            y=work["latitude"],
            mode="lines",
            line=dict(color="rgba(120,120,120,0.35)", width=1),
            name="Track",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    point_sizes = np.clip(8 + (pd.to_numeric(daily["rest_hours"], errors="coerce").fillna(0.0) * 2.5), 8, 28)
    fig.add_trace(
        go.Scatter(
            x=daily["longitude"],
            y=daily["latitude"],
            mode="markers",
            marker=dict(
                size=point_sizes,
                color=pd.to_numeric(daily["rest_hours"], errors="coerce").fillna(0.0),
                colorscale="Magma",
                colorbar=dict(title="Rest h/day", len=0.75, y=0.5),
                line=dict(color="rgba(0,0,0,0.25)", width=0.8),
                opacity=0.95,
            ),
            customdata=daily[["local_date", "rest_hours", "track_hours"]].astype(str).values,
            hovertemplate=(
                "Date %{customdata[0]}<br>Putative rest %{customdata[1]} h"
                "<br>Observed %{customdata[2]} h<br>Lon %{x:.3f}<br>Lat %{y:.3f}<extra></extra>"
            ),
            name="Daily putative rest",
            showlegend=False,
        )
    )
    if isinstance(rest_index_df, pd.DataFrame) and not rest_index_df.empty:
        rest_points = rest_index_df.copy()
        rest_points = rest_points.dropna(subset=["segment_midpoint_datetime"]).sort_values("segment_midpoint_datetime").reset_index(drop=True)
        location_sorted = work[["datetime", "latitude", "longitude"]].sort_values("datetime").reset_index(drop=True)
        if not rest_points.empty and not location_sorted.empty:
            mapped = pd.merge_asof(
                rest_points,
                location_sorted,
                left_on="segment_midpoint_datetime",
                right_on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta("12H"),
            )
            mapped = mapped.dropna(subset=["latitude", "longitude"])
            if not mapped.empty:
                kept_mask = mapped["status_keep_filtered"].fillna(False).astype(bool)
                if kept_mask.any():
                    kept = mapped.loc[kept_mask].copy()
                    kept_sizes = np.clip(np.sqrt(pd.to_numeric(kept["duration_s"], errors="coerce").fillna(0.0)) / 3.0, 7, 18)
                    fig.add_trace(
                        go.Scatter(
                            x=kept["longitude"],
                            y=kept["latitude"],
                            mode="markers",
                            marker=dict(
                                size=kept_sizes,
                                color=pd.to_numeric(
                                    kept["status_drift_rate_ms"].where(pd.notna(kept["status_drift_rate_ms"]), kept["drift_rate_ms"]),
                                    errors="coerce",
                                ),
                                colorscale="Magma",
                                showscale=False,
                                line=dict(color="rgba(0,0,0,0.35)", width=0.8),
                                symbol=np.where(kept["rest_category"].eq("surface_sleep"), "diamond", np.where(kept["rest_category"].eq("benthic_sleep"), "square", "circle")),
                                opacity=0.95,
                            ),
                            customdata=kept[["segment_rank", "state_name"]].values,
                            hovertemplate="Rank %{customdata[0]}<br>%{customdata[1]}<br>Lon %{x:.3f}<br>Lat %{y:.3f}<extra></extra>",
            name="Post-measured putative rest",
            showlegend=False,
        )
                    )
                if (~kept_mask).any():
                    overwritten = mapped.loc[~kept_mask].copy()
                    overwritten_sizes = np.clip(np.sqrt(pd.to_numeric(overwritten["duration_s"], errors="coerce").fillna(0.0)) / 3.0, 7, 18)
                    fig.add_trace(
                        go.Scatter(
                            x=overwritten["longitude"],
                            y=overwritten["latitude"],
                            mode="markers",
                            marker=dict(
                                size=overwritten_sizes,
                                color="#D62728",
                                line=dict(color="rgba(0,0,0,0.35)", width=0.8),
                                symbol=np.where(overwritten["rest_category"].eq("surface_sleep"), "diamond-open", np.where(overwritten["rest_category"].eq("benthic_sleep"), "square-open", "circle-open")),
                                opacity=0.95,
                            ),
                            customdata=overwritten[["segment_rank", "state_name"]].values,
                            hovertemplate="Rank %{customdata[0]}<br>%{customdata[1]}<br>Overwritten<br>Lon %{x:.3f}<br>Lat %{y:.3f}<extra></extra>",
                            name="Overwritten putative rest",
                            showlegend=False,
                        )
                    )
    fig.update_layout(
        template="plotly_white",
        title=f"{deployment_id} Daily Rest Map",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=300,
        margin=dict(l=35, r=15, t=45, b=30),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _detect_supervised_positive_label_plot(labels):
    label_texts = [str(x).strip() for x in labels if str(x).strip()]
    for candidate in label_texts:
        if candidate.lower() == "sleep":
            return candidate
    for candidate in label_texts:
        if "sleep" in candidate.lower():
            return candidate
    return None


def _extract_supervised_nap_windows(supervised_prediction_df, deployment_id, tz_name):
    if not isinstance(supervised_prediction_df, pd.DataFrame) or supervised_prediction_df.empty:
        return pd.DataFrame()
    required = {"window_start", "window_end"}
    if not required.issubset(supervised_prediction_df.columns):
        return pd.DataFrame()
    dep = supervised_prediction_df.copy()
    if "deployment_id" in dep.columns:
        dep = dep.loc[dep["deployment_id"].astype(str) == str(deployment_id)].copy()
    if dep.empty:
        return pd.DataFrame()
    if "variant_id" in dep.columns:
        variant_ids = dep["variant_id"].dropna().astype(str).tolist()
        preferred_variant = None
        if "rf_full" in variant_ids:
            preferred_variant = "rf_full"
        else:
            rf_full_like = next((v for v in variant_ids if v.startswith("rf_full")), None)
            if rf_full_like:
                preferred_variant = rf_full_like
            elif "variant_type" in dep.columns:
                rf_variants = dep.loc[dep["variant_type"].astype(str) == "random_forest", "variant_id"].dropna().astype(str).tolist()
                preferred_variant = rf_variants[0] if rf_variants else None
        if preferred_variant:
            dep = dep.loc[dep["variant_id"].astype(str) == preferred_variant].copy()
    if dep.empty:
        return pd.DataFrame()
    label_col = "predicted_label" if "predicted_label" in dep.columns else ("final_behavior" if "final_behavior" in dep.columns else None)
    if not label_col:
        return pd.DataFrame()
    dep["window_start"] = _normalize_plot_datetime(dep["window_start"], tz_name)
    dep["window_end"] = _normalize_plot_datetime(dep["window_end"], tz_name)
    dep = dep.dropna(subset=["window_start", "window_end"]).copy()
    dep[label_col] = dep[label_col].astype(str).str.strip()
    dep = dep.loc[dep[label_col].ne("") & dep[label_col].ne("Unknown") & (~dep[label_col].str.lower().isin({"nan", "none"}))].copy()
    if dep.empty:
        return pd.DataFrame()
    positive_label = _detect_supervised_positive_label_plot(dep[label_col].tolist())
    if not positive_label:
        return pd.DataFrame()
    dep = dep.loc[dep[label_col].astype(str) == str(positive_label)].copy()
    if dep.empty:
        return pd.DataFrame()
    dep["segment_midpoint_datetime"] = dep["window_start"] + (dep["window_end"] - dep["window_start"]) / 2
    dep["duration_s"] = (dep["window_end"] - dep["window_start"]).dt.total_seconds()
    dep["local_hour"] = (
        dep["segment_midpoint_datetime"].dt.hour
        + dep["segment_midpoint_datetime"].dt.minute / 60.0
        + dep["segment_midpoint_datetime"].dt.second / 3600.0
    )
    return dep[["window_start", "window_end", "segment_midpoint_datetime", "duration_s", "local_hour"]].rename(
        columns={"window_start": "start_datetime", "window_end": "end_datetime"}
    )


def _build_supervised_nap_overview_fig(
    work_df,
    deployment_id,
    tz_name,
    lat,
    lon,
    supervised_prediction_df=None,
    window_start=None,
    window_end=None,
):
    if not isinstance(work_df, pd.DataFrame) or work_df.empty:
        return go.Figure()
    work_df = work_df.copy()
    work_df["datetime"] = pd.to_datetime(work_df["datetime"], errors="coerce")
    work_df = work_df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    algo_points = work_df.loc[
        work_df["activity_state"].astype(str).isin(["drift_sleep", "benthic_sleep", "surface_sleep"]),
        ["datetime", "local_hour"],
    ].dropna().sort_values("datetime").copy()
    if algo_points.empty:
        return go.Figure()
    if len(algo_points) > 30000:
        stride = int(np.ceil(len(algo_points) / 30000.0))
        algo_points = algo_points.iloc[::max(stride, 1)].copy()

    rf_points = _extract_supervised_nap_windows(supervised_prediction_df, deployment_id, tz_name)
    if not rf_points.empty:
        rf_points = rf_points.sort_values("segment_midpoint_datetime").copy()
    if rf_points.empty:
        return go.Figure()
    if len(rf_points) > 30000:
        stride = int(np.ceil(len(rf_points) / 30000.0))
        rf_points = rf_points.iloc[::max(stride, 1)].copy()

    fig = go.Figure()
    day_starts = pd.date_range(
        start=work_df["datetime"].dt.floor("D").min(),
        end=work_df["datetime"].dt.floor("D").max(),
        freq="1D",
        tz=work_df["datetime"].dt.tz,
    )
    for day_start in day_starts:
        sunrise_h, sunset_h = sunrise_sunset_local_hours(day_start, lat, lon, tz_name)
        if np.isfinite(sunrise_h):
            sunrise_dt = day_start + pd.to_timedelta(float(sunrise_h), unit="h")
            fig.add_trace(
                go.Scatter(
                    x=[sunrise_dt, sunrise_dt],
                    y=[0, 24],
                    mode="lines",
                    line=dict(color="#f59e0b", width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_vrect(x0=day_start, x1=sunrise_dt, fillcolor="rgba(107, 114, 128, 0.10)", line_width=0, layer="below")
        if np.isfinite(sunset_h):
            sunset_dt = day_start + pd.to_timedelta(float(sunset_h), unit="h")
            next_day = day_start + pd.Timedelta(days=1)
            fig.add_trace(
                go.Scatter(
                    x=[sunset_dt, sunset_dt],
                    y=[0, 24],
                    mode="lines",
                    line=dict(color="#1d4ed8", width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_vrect(x0=sunset_dt, x1=next_day, fillcolor="rgba(107, 114, 128, 0.10)", line_width=0, layer="below")

    fig.add_trace(
        go.Scattergl(
            x=algo_points["datetime"],
            y=algo_points["local_hour"],
            mode="markers",
            name="Algorithmic putative naps",
            marker=dict(color="rgba(255, 215, 0, 0.5)", size=4),
            hovertemplate="%{x}<br>Local hour %{y:.2f}<br>Algorithmic putative nap<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=rf_points["segment_midpoint_datetime"],
            y=rf_points["local_hour"],
            mode="markers",
            name="RF putative naps",
            marker=dict(color="rgba(0, 102, 255, 0.5)", size=4),
            hovertemplate="%{x}<br>Local hour %{y:.2f}<br>RF putative nap<extra></extra>",
        )
    )
    if window_start is not None and window_end is not None:
        try:
            fig.add_vrect(
                x0=pd.Timestamp(window_start),
                x1=pd.Timestamp(window_end),
                fillcolor="rgba(241, 196, 15, 0.10)",
                line_color="#F1C40F",
                line_width=1,
                layer="above",
            )
        except Exception:
            pass
    fig.update_layout(
        template="plotly_white",
        title=f"{deployment_id} Putative Nap Overview",
        xaxis_title="Local time across trip",
        yaxis_title="Local time of day",
        yaxis=dict(range=[0, 24], tickmode="array", tickvals=list(range(0, 25, 2))),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def continuous_daily_activity_plot(
    data_pkl,
    seg_df,
    deployment_id,
    aggregation="1D",
    timezone_name=None,
    lat_lon_source=None,
    state_label_map=None,
    dive_depth_threshold_m=2.0,
    selected_segment_rank=None,
    color_mapping_path=None,
    show_solar_context=True,
    size_markers_by_duration=True,
    wrap_daily=False,
    status_seg_df=None,
    window_start=None,
    window_end=None,
    supervised_prediction_df=None,
    buoyancy_reference_max_start_depth_m=500.0,
    buoyancy_tolerance_ms=0.15,
):
    color_mapping = load_color_mapping(color_mapping_path) if color_mapping_path else {}
    deployment_info = getattr(data_pkl, "deployment_info", {}) or {}
    tz_name = str(timezone_name or deployment_info.get("Time Zone") or "UTC")
    lat, lon = _resolve_plot_lat_lon(lat_lon_source, deployment_info=deployment_info)

    depth_df = _pick_depth_frame(data_pkl)
    depth_df["datetime"] = _normalize_plot_datetime(depth_df["datetime"], tz_name)
    depth_df = depth_df.dropna(subset=["datetime", "depth_value"]).sort_values("datetime").reset_index(drop=True)
    if depth_df.empty:
        raise ValueError("No depth/pressure timeseries samples were available for continuous daily activity plotting.")

    rest_index_df = _segment_rest_index(seg_df, state_label_map=state_label_map)
    rest_index_df["start_datetime"] = _normalize_plot_datetime(rest_index_df["start_datetime"], tz_name)
    rest_index_df["end_datetime"] = _normalize_plot_datetime(rest_index_df["end_datetime"], tz_name)
    rest_index_df["segment_midpoint_datetime"] = _normalize_plot_datetime(rest_index_df["segment_midpoint_datetime"], tz_name)
    rest_index_df["local_date"] = rest_index_df["segment_midpoint_datetime"].dt.date
    rest_index_df["local_hour"] = (
        rest_index_df["segment_midpoint_datetime"].dt.hour
        + rest_index_df["segment_midpoint_datetime"].dt.minute / 60.0
        + rest_index_df["segment_midpoint_datetime"].dt.second / 3600.0
    )
    rest_index_df = rest_index_df.sort_values("segment_midpoint_datetime").reset_index(drop=True)
    status_lookup = {}
    if isinstance(status_seg_df, pd.DataFrame) and not status_seg_df.empty and "segment_rank" in status_seg_df.columns:
        status_work = status_seg_df.copy()
        status_work["segment_rank_num"] = pd.to_numeric(status_work["segment_rank"], errors="coerce")
        status_work = status_work.dropna(subset=["segment_rank_num"]).sort_values(["segment_rank_num"]).drop_duplicates("segment_rank_num", keep="last")
        status_lookup = status_work.set_index("segment_rank_num").to_dict("index")
    rest_index_df["segment_rank_num"] = pd.to_numeric(rest_index_df["segment_rank"], errors="coerce")
    rest_index_df["status_keep_filtered"] = rest_index_df["segment_rank_num"].map(
        lambda rank: bool((status_lookup.get(rank) or {}).get("keep_filtered", True))
    )
    rest_index_df["status_post_measured_keep"] = rest_index_df["segment_rank_num"].map(
        lambda rank: bool((status_lookup.get(rank) or {}).get("post_measured_keep", True))
    )
    rest_index_df["status_drift_rate_ms"] = rest_index_df["segment_rank_num"].map(
        lambda rank: (status_lookup.get(rank) or {}).get("drift_rate_ms", pd.NA)
    )
    rest_index_df["status_state_name"] = rest_index_df["segment_rank_num"].map(
        lambda rank: (status_lookup.get(rank) or {}).get("state_name", "")
    )

    work = depth_df.copy()
    sample_step_s = work["datetime"].diff().dt.total_seconds().median()
    if pd.isna(sample_step_s) or float(sample_step_s) <= 0:
        sample_step_s = 60.0
    work["duration_h"] = work["datetime"].shift(-1).sub(work["datetime"]).dt.total_seconds().fillna(sample_step_s) / 3600.0
    work["activity_state"] = np.where(
        pd.to_numeric(work["depth_value"], errors="coerce").fillna(np.inf) < float(dive_depth_threshold_m),
        "surfacing",
        "diving",
    )
    for _, row in rest_index_df.iterrows():
        start_dt = row.get("start_datetime")
        end_dt = row.get("end_datetime")
        category = row.get("rest_category")
        if pd.isna(start_dt) or pd.isna(end_dt) or not str(category):
            continue
        mask = (work["datetime"] >= start_dt) & (work["datetime"] <= end_dt)
        work.loc[mask, "activity_state"] = str(category)

    activity_order = ["drift_sleep", "benthic_sleep", "surface_sleep", "surfacing", "diving"]
    activity_colors = {
        "drift_sleep": _event_style_color_for_key(color_mapping, "putative_rest.drift_sleep", "#7B61FF"),
        "benthic_sleep": _event_style_color_for_key(color_mapping, "putative_rest.benthic_sleep", "#3F72AF"),
        "surface_sleep": _event_style_color_for_key(color_mapping, "putative_rest.surface_sleep", "#00B8A9"),
        "surfacing": "#BFD7EA",
        "diving": "#334E68",
    }
    work["bin_start"] = work["datetime"].dt.floor(aggregation)
    work["local_date"] = work["datetime"].dt.date
    work["local_hour"] = (
        work["datetime"].dt.hour
        + work["datetime"].dt.minute / 60.0
        + work["datetime"].dt.second / 3600.0
    )
    budget = (
        work.groupby(["bin_start", "activity_state"], dropna=False)["duration_h"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=activity_order, fill_value=0.0)
        .sort_index()
    )

    if wrap_daily:
        day_budget = (
            work.groupby(["local_date", "bin_start", "activity_state"], dropna=False)["duration_h"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(columns=activity_order, fill_value=0.0)
            .reset_index()
        )
        day_labels = [str(day) for day in sorted(day_budget["local_date"].dropna().unique().tolist())]
        if not day_labels:
            daily_activity_fig = go.Figure()
        else:
            daily_activity_fig = make_subplots(
                rows=len(day_labels),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.015,
                subplot_titles=day_labels,
            )
            freq_td = pd.to_timedelta(aggregation)
            bin_hours = max(freq_td.total_seconds() / 3600.0, 1.0)
            for row_idx, day_label in enumerate(day_labels, start=1):
                day_df = day_budget.loc[day_budget["local_date"].astype(str) == day_label].copy()
                if day_df.empty:
                    continue
                day_df["bin_local_hour"] = (
                    pd.to_datetime(day_df["bin_start"], errors="coerce").dt.hour
                    + pd.to_datetime(day_df["bin_start"], errors="coerce").dt.minute / 60.0
                    + pd.to_datetime(day_df["bin_start"], errors="coerce").dt.second / 3600.0
                    + (bin_hours / 2.0)
                )
                for state_name in activity_order:
                    daily_activity_fig.add_trace(
                        go.Bar(
                            x=day_df["bin_local_hour"],
                            y=day_df[state_name],
                            width=[bin_hours] * len(day_df),
                            name=state_name,
                            marker_color=activity_colors[state_name],
                            hovertemplate=(
                                f"{day_label}<br>%{{x:.2f}} h<br>{state_name}: %{{y:.2f}} h<extra></extra>"
                            ),
                            showlegend=row_idx == 1,
                        ),
                        row=row_idx,
                        col=1,
                    )
                daily_activity_fig.update_yaxes(title_text="h", row=row_idx, col=1)
                if show_solar_context:
                    day_ts = pd.Timestamp(day_label)
                    sunrise_h, sunset_h = sunrise_sunset_local_hours(day_ts, lat, lon, tz_name)
                    if np.isfinite(sunrise_h):
                        daily_activity_fig.add_vline(
                            x=float(sunrise_h),
                            line_color="#f59e0b",
                            line_width=1,
                            line_dash="dash",
                            row=row_idx,
                            col=1,
                        )
                        daily_activity_fig.add_vrect(
                            x0=0.0,
                            x1=float(sunrise_h),
                            fillcolor="rgba(107, 114, 128, 0.10)",
                            line_width=0,
                            layer="below",
                            row=row_idx,
                            col=1,
                        )
                    if np.isfinite(sunset_h):
                        daily_activity_fig.add_vline(
                            x=float(sunset_h),
                            line_color="#1d4ed8",
                            line_width=1,
                            line_dash="dash",
                            row=row_idx,
                            col=1,
                        )
                        daily_activity_fig.add_vrect(
                            x0=float(sunset_h),
                            x1=24.0,
                            fillcolor="rgba(107, 114, 128, 0.10)",
                            line_width=0,
                            layer="below",
                            row=row_idx,
                            col=1,
                        )
            daily_activity_fig.update_layout(
                barmode="stack",
                template="plotly_white",
                title=f"{deployment_id} Daily Activity Actogram",
                legend_title="Activity",
                height=max(300, 120 * len(day_labels)),
                margin=dict(l=50, r=20, t=50, b=40),
            )
            daily_activity_fig.update_xaxes(range=[0, 24], title_text="Hour of local day", row=len(day_labels), col=1)
    else:
        budget = budget.sort_index()
        daily_activity_fig = go.Figure()
        for state_name in activity_order:
            daily_activity_fig.add_trace(
                go.Bar(
                    x=budget.index,
                    y=budget[state_name],
                    name=state_name,
                    marker_color=activity_colors[state_name],
                    hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f} h<extra></extra>",
                )
            )
        daily_activity_fig.update_layout(
            barmode="stack",
            template="plotly_white",
            title=f"{deployment_id} Daily Activity",
            xaxis_title="Local time",
            yaxis_title="Hours per bin",
            legend_title="Activity",
            height=360,
            margin=dict(l=40, r=20, t=50, b=40),
        )

    work = work.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    rest_trip_fig = go.Figure()
    day_starts = pd.date_range(
        start=work["datetime"].dt.floor("D").min(),
        end=work["datetime"].dt.floor("D").max(),
        freq="1D",
        tz=work["datetime"].dt.tz,
    )
    if show_solar_context:
        for day_start in day_starts:
            sunrise_h, sunset_h = sunrise_sunset_local_hours(day_start, lat, lon, tz_name)
            if np.isfinite(sunrise_h):
                sunrise_dt = day_start + pd.to_timedelta(float(sunrise_h), unit="h")
                if not wrap_daily:
                    daily_activity_fig.add_vline(x=sunrise_dt, line_color="#f59e0b", line_width=1, line_dash="dash")
                rest_trip_fig.add_trace(
                    go.Scatter(
                        x=[sunrise_dt, sunrise_dt],
                        y=[0, 24],
                        mode="lines",
                        line=dict(color="#f59e0b", width=1, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                if not wrap_daily:
                    daily_activity_fig.add_vrect(x0=day_start, x1=sunrise_dt, fillcolor="rgba(107, 114, 128, 0.10)", line_width=0, layer="below")
                rest_trip_fig.add_vrect(x0=day_start, x1=sunrise_dt, fillcolor="rgba(107, 114, 128, 0.10)", line_width=0, layer="below")
            if np.isfinite(sunset_h):
                sunset_dt = day_start + pd.to_timedelta(float(sunset_h), unit="h")
                next_day = day_start + pd.Timedelta(days=1)
                if not wrap_daily:
                    daily_activity_fig.add_vline(x=sunset_dt, line_color="#1d4ed8", line_width=1, line_dash="dash")
                rest_trip_fig.add_trace(
                    go.Scatter(
                        x=[sunset_dt, sunset_dt],
                        y=[0, 24],
                        mode="lines",
                        line=dict(color="#1d4ed8", width=1, dash="dash"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                if not wrap_daily:
                    daily_activity_fig.add_vrect(x0=sunset_dt, x1=next_day, fillcolor="rgba(107, 114, 128, 0.10)", line_width=0, layer="below")
                rest_trip_fig.add_vrect(x0=sunset_dt, x1=next_day, fillcolor="rgba(107, 114, 128, 0.10)", line_width=0, layer="below")

    kept = rest_index_df.loc[rest_index_df["status_keep_filtered"].fillna(False).astype(bool)].copy()
    overwritten = rest_index_df.loc[
        rest_index_df["status_post_measured_keep"].fillna(False).astype(bool)
        & ~rest_index_df["status_keep_filtered"].fillna(False).astype(bool)
    ].copy()
    if not kept.empty:
        kept = kept.dropna(subset=["segment_midpoint_datetime", "local_hour"]).sort_values("segment_midpoint_datetime").copy()
        marker_sizes = np.clip(np.sqrt(pd.to_numeric(kept["duration_s"], errors="coerce").fillna(0.0)) / 2.5, 6, 24)
        if not size_markers_by_duration:
            marker_sizes = np.full(len(kept), 10.0)
        focus_mask = pd.to_numeric(kept["segment_rank"], errors="coerce").fillna(-1).astype(int) == int(selected_segment_rank or -1)
        drift_color = pd.to_numeric(
            kept["status_drift_rate_ms"].where(pd.notna(kept["status_drift_rate_ms"]), kept["drift_rate_ms"]),
            errors="coerce",
        )
        rest_trip_fig.add_trace(
            go.Scatter(
                x=kept["segment_midpoint_datetime"],
                y=kept["local_hour"],
                mode="markers",
                name="Final kept rest",
                customdata=kept[["segment_rank", "state_name"]].values,
                marker=dict(
                    size=marker_sizes,
                    color=drift_color,
                    colorscale="Magma",
                    colorbar=dict(title="Drift rate (m/s)"),
                    showscale=True,
                    line=dict(
                        width=np.where(focus_mask, 3.0, 0.6).tolist(),
                        color=np.where(focus_mask, "#F1C40F", "rgba(0,0,0,0.18)").tolist(),
                    ),
                    symbol=np.where(kept["rest_category"].eq("surface_sleep"), "diamond", np.where(kept["rest_category"].eq("benthic_sleep"), "square", "circle")),
                    opacity=0.95,
                ),
                hovertemplate=(
                    "Rank %{customdata[0]}<br>%{customdata[1]}<br>%{x}"
                    "<br>Local hour %{y:.2f}<br>Drift rate %{marker.color:.3f} m/s<extra></extra>"
                ),
            )
        )
    if not overwritten.empty:
        overwritten = overwritten.dropna(subset=["segment_midpoint_datetime", "local_hour"]).sort_values("segment_midpoint_datetime").copy()
        marker_sizes = np.clip(np.sqrt(pd.to_numeric(overwritten["duration_s"], errors="coerce").fillna(0.0)) / 2.5, 6, 24)
        if not size_markers_by_duration:
            marker_sizes = np.full(len(overwritten), 10.0)
        focus_mask = pd.to_numeric(overwritten["segment_rank"], errors="coerce").fillna(-1).astype(int) == int(selected_segment_rank or -1)
        rest_trip_fig.add_trace(
            go.Scatter(
                x=overwritten["segment_midpoint_datetime"],
                y=overwritten["local_hour"],
                mode="markers",
                name="Rejected after inferred filters",
                customdata=overwritten[["segment_rank", "plot_state_name"]].values,
                marker=dict(
                    size=marker_sizes,
                    color="#D62728",
                    line=dict(
                        width=np.where(focus_mask, 3.0, 0.6).tolist(),
                        color=np.where(focus_mask, "#F1C40F", "rgba(0,0,0,0.18)").tolist(),
                    ),
                    symbol=np.where(overwritten["rest_category"].eq("surface_sleep"), "diamond-open", np.where(overwritten["rest_category"].eq("benthic_sleep"), "square-open", "circle-open")),
                    opacity=0.95,
                ),
                hovertemplate=(
                    "Rank %{customdata[0]}<br>%{customdata[1]}<br>%{x}"
                    "<br>Local hour %{y:.2f}<br>Rejected after inferred filters<extra></extra>"
                ),
            )
        )
    if window_start is not None and window_end is not None:
        try:
            daily_activity_fig.add_vrect(
                x0=pd.Timestamp(window_start),
                x1=pd.Timestamp(window_end),
                fillcolor="rgba(241, 196, 15, 0.10)",
                line_color="#F1C40F",
                line_width=1,
                layer="above",
            )
        except Exception:
            pass
        try:
            rest_trip_fig.add_vrect(
                x0=pd.Timestamp(window_start),
                x1=pd.Timestamp(window_end),
                fillcolor="rgba(241, 196, 15, 0.10)",
                line_color="#F1C40F",
                line_width=1,
                layer="above",
            )
        except Exception:
            pass
    rest_trip_fig.update_layout(
        template="plotly_white",
        title=f"{deployment_id} Rest Across Trip",
        xaxis_title="Local time across trip",
        yaxis_title="Local time of day",
        yaxis=dict(range=[0, 24], tickmode="array", tickvals=list(range(0, 25, 2))),
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    buoyancy_track_fig = _build_buoyancy_track_fig(
        seg_df,
        deployment_id,
        reference_max_start_depth_m=buoyancy_reference_max_start_depth_m,
        tolerance_ms=buoyancy_tolerance_ms,
    )
    if window_start is not None and window_end is not None and getattr(buoyancy_track_fig, "data", None):
        try:
            buoyancy_track_fig.add_vrect(
                x0=pd.Timestamp(window_start),
                x1=pd.Timestamp(window_end),
                fillcolor="rgba(241, 196, 15, 0.10)",
                line_color="#F1C40F",
                line_width=1,
                layer="above",
            )
        except Exception:
            pass
    daily_rest_map_fig = _build_daily_rest_map_fig(data_pkl, rest_index_df, deployment_id, tz_name)
    supervised_nap_overview_fig = _build_supervised_nap_overview_fig(
        work_df=work,
        deployment_id=deployment_id,
        tz_name=tz_name,
        lat=lat,
        lon=lon,
        supervised_prediction_df=supervised_prediction_df,
        window_start=window_start,
        window_end=window_end,
    )

    return {
        "buoyancy_track_fig": buoyancy_track_fig,
        "supervised_nap_overview_fig": supervised_nap_overview_fig,
        "daily_activity_fig": daily_activity_fig,
        "daily_rest_map_fig": daily_rest_map_fig,
        "rest_trip_fig": rest_trip_fig,
        "rest_index_df": rest_index_df,
    }


def _safe_text(value):
    """Return a normalized string for display; empty string for missing values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _resolve_signal_display_metadata(signal, signal_info, signal_metadata=None):
    """
    Resolve display metadata for a signal and its channels.
    Returns dict with signal-level label/unit/description and per-channel metadata.
    """
    pretty_signal_label_overrides = {
        "algorithmic_d1_channel": "d'(depth)",
        "algorithmic_d2_channel": "d''(depth)",
        "algorithmic_depth_d1": "d'(depth)",
        "algorithmic_depth_d2": "d''(depth)",
    }
    signal_key = str(signal)
    out = {
        "signal_label": signal,
        "signal_unit": "",
        "signal_description": "",
        "channels": {},
    }
    if signal_key in pretty_signal_label_overrides:
        out["signal_label"] = pretty_signal_label_overrides[signal_key]

    # Signal-level fallback from signal_info.
    info_units = (signal_info or {}).get("units")
    if isinstance(info_units, str):
        out["signal_unit"] = _safe_text(info_units)
    elif isinstance(info_units, (list, tuple)) and info_units:
        out["signal_unit"] = _safe_text(info_units[0])
    elif isinstance(info_units, dict) and info_units:
        first = next(iter(info_units.values()))
        out["signal_unit"] = _safe_text(first)
    out["signal_description"] = _safe_text((signal_info or {}).get("details"))

    # Start with signal_info channel-level units.
    info_meta = (signal_info or {}).get("metadata", {}) or {}
    for ch, meta in info_meta.items():
        meta = meta or {}
        unit = _safe_text(meta.get("standardized_unit")) or _safe_text(meta.get("unit"))
        ch_label = _safe_text(meta.get("original_name")) or str(ch)
        out["channels"][str(ch)] = {
            "label": ch_label,
            "unit": unit,
            "description": "",
            "label_suffix": "",
        }
        if not out["signal_unit"] and unit:
            out["signal_unit"] = unit

    # Overlay from optional external metadata map.
    if isinstance(signal_metadata, dict):
        sm = signal_metadata.get(signal) or signal_metadata.get(str(signal))
        if isinstance(sm, dict):
            out["signal_label"] = _safe_text(sm.get("label")) or out["signal_label"]
            out["signal_unit"] = _safe_text(sm.get("unit")) or out["signal_unit"]
            out["signal_description"] = _safe_text(sm.get("description")) or out["signal_description"]
            ch_map = sm.get("channels", {}) or {}
            if isinstance(ch_map, dict):
                for ch, ch_meta in ch_map.items():
                    if not isinstance(ch_meta, dict):
                        continue
                    base = out["channels"].get(str(ch), {"label": str(ch), "unit": "", "description": ""})
                    out["channels"][str(ch)] = {
                        "label": _safe_text(ch_meta.get("label")) or base.get("label", str(ch)),
                        "unit": _safe_text(ch_meta.get("unit")) or base.get("unit", ""),
                        "description": _safe_text(ch_meta.get("description")) or base.get("description", ""),
                        "label_suffix": _safe_text(ch_meta.get("label_suffix")) or base.get("label_suffix", ""),
                    }

    return out


def _format_signal_axis_title(display_meta):
    label = _safe_text((display_meta or {}).get("signal_label"))
    unit = _safe_text((display_meta or {}).get("signal_unit"))
    if label and unit:
        return f"{label} ({unit})"
    return label or ""


def _add_signal_axis_label_annotation(fig, row, total_rows, label, unit):
    """
    Render horizontal left-aligned signal labels per subplot row:
    bold signal name and italicized unit on the next line.
    """
    label_txt = _safe_text(label)
    unit_txt = _safe_text(unit)
    if not label_txt and not unit_txt:
        return
    if total_rows <= 0:
        return
    row_height = 1.0 / float(total_rows)
    y_mid = 1.0 - ((row - 0.5) * row_height)
    if label_txt and unit_txt:
        text = (
            f"<span style='font-size:15px; font-weight:700; color:#4b5563;'>{html.escape(label_txt)}</span>"
            f"<br><span style='font-size:12px; font-style:italic; color:#4b5563;'>({html.escape(unit_txt)})</span>"
        )
    elif label_txt:
        text = f"<span style='font-size:15px; font-weight:700; color:#4b5563;'>{html.escape(label_txt)}</span>"
    else:
        text = f"<span style='font-size:12px; font-style:italic; color:#4b5563;'>({html.escape(unit_txt)})</span>"

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=y_mid,
        text=text,
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        align="right",
        xshift=-24,
        font=dict(size=14, color="#4b5563", family="Figtree, sans-serif"),
    )


def _add_signal_help_annotation(fig, row, total_rows, text):
    """
    Add a small '(?)' hover help marker near the top-left of a subplot row.
    This is ignored when description text is missing.
    """
    desc = _safe_text(text)
    if not desc:
        return
    if total_rows <= 0:
        return
    row_height = 1.0 / float(total_rows)
    y_top = 1.0 - ((row - 1) * row_height) - 0.01
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.006,
        y=y_top,
        text="(?)",
        showarrow=False,
        font=dict(size=10, color="#8ecfff"),
        align="left",
        hovertext=desc,
        hoverlabel=dict(bgcolor="#0d2f49", bordercolor="#6fb7ea", font=dict(color="#ffffff")),
    )


def _build_trace_hovertemplate(
    signal_name,
    channel_name,
    signal_label,
    channel_label_suffix,
    channel_description,
    channel_unit,
):
    desc = _safe_text(channel_description)
    safe_signal = html.escape(_safe_text(signal_name))
    safe_channel = html.escape(_safe_text(channel_name))
    safe_signal_label = html.escape(_safe_text(signal_label))
    safe_suffix = html.escape(_safe_text(channel_label_suffix))
    safe_unit = html.escape(_safe_text(channel_unit))
    value_line = "<span style='font-size:1.35em;'><b>%{y:.2g}"
    if safe_unit:
        value_line += f" {safe_unit}"
    value_line += "</b></span>"
    lines = [
        value_line,
        "<b>%{x|%Y-%m-%d %H:%M:%S.%L}</b>",
        f"<b>{safe_channel}</b> <i>({safe_signal})</i>",
    ]
    if safe_signal_label and safe_suffix:
        lines.append(f"<b>{safe_signal_label}</b> + <i>{safe_suffix}</i>")
    elif safe_signal_label:
        lines.append(f"<b>{safe_signal_label}</b>")
    elif safe_suffix:
        lines.append(f"<i>{safe_suffix}</i>")
    if desc:
        safe_desc = html.escape(desc)
        lines.append(safe_desc)
    return "<br>".join(lines) + "<extra></extra>"


def _parse_color_to_rgb(color_value):
    """Best-effort parser for hex/rgb/rgba colors used in hover styling."""
    if not isinstance(color_value, str):
        return None
    color_text = color_value.strip()
    if not color_text:
        return None

    if color_text.startswith("#"):
        hex_value = color_text.lstrip("#")
        if len(hex_value) == 3:
            hex_value = "".join(ch * 2 for ch in hex_value)
        if len(hex_value) == 6:
            try:
                return tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return None
        return None

    match = re.match(r"rgba?\(([^)]+)\)", color_text, flags=re.IGNORECASE)
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split(",")]
    if len(parts) < 3:
        return None
    try:
        rgb = []
        for part in parts[:3]:
            value = float(part)
            rgb.append(int(max(0, min(255, round(value)))))
        return tuple(rgb)
    except ValueError:
        return None


def _build_state_hoverlabel(color_value):
    """Use the state color for hover labels with readable foreground text."""
    color_text = _safe_text(color_value) or "#4b5563"
    rgb = _parse_color_to_rgb(color_text)
    if rgb is None:
        font_color = "#ffffff"
    else:
        r, g, b = rgb
        luminance = ((0.299 * r) + (0.587 * g) + (0.114 * b)) / 255.0
        font_color = "#0f172a" if luminance >= 0.62 else "#ffffff"

    return dict(
        bgcolor=color_text,
        bordercolor=color_text,
        font=dict(color=font_color),
    )


def _build_trace_legend_name(signal_name, channel_name, signal_label, channel_label_suffix):
    """
    Human-readable legend label aligned with hover semantics and styling.
    """
    sig = html.escape(_safe_text(signal_name))
    ch = html.escape(_safe_text(channel_name))
    sig_label = html.escape(_safe_text(signal_label))
    suffix = html.escape(_safe_text(channel_label_suffix))
    if sig_label and suffix:
        human = f"<b>{sig_label}</b> + <i>{suffix}</i>"
    elif sig_label:
        human = f"<b>{sig_label}</b>"
    elif suffix:
        human = f"<i>{suffix}</i>"
    else:
        human = ch or "channel"
    return f"<b>{ch}</b>; <i>{sig}</i>; {human}"


def _coerce_datetime_series(series):
    """Coerce datetime series and resolve mixed tz-aware/naive values."""
    try:
        return pd.to_datetime(series, errors="coerce")
    except (ValueError, TypeError):
        target_tz = None
        parsed = []
        for value in series:
            if pd.isna(value):
                parsed.append(pd.NaT)
                continue
            ts = pd.Timestamp(value)
            if target_tz is None and ts.tzinfo is not None:
                target_tz = ts.tzinfo
            parsed.append(ts)
        if target_tz is None:
            return pd.to_datetime(pd.Series(parsed, index=series.index), errors="coerce")
        normalized = []
        for ts in parsed:
            if pd.isna(ts):
                normalized.append(pd.NaT)
            elif ts.tzinfo is None:
                normalized.append(ts.tz_localize(target_tz))
            else:
                normalized.append(ts.tz_convert(target_tz))
        return pd.Series(normalized, index=series.index)

def _align_range_to_series(datetime_series, start_time, end_time):
    """Align start/end timestamps to the timezone-naive/aware state of datetime_series."""
    dt = _coerce_datetime_series(datetime_series)
    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    series_tz = dt.dt.tz

    if series_tz is None:
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_localize(None)
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_localize(None)
    else:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(series_tz)
        else:
            start_ts = start_ts.tz_convert(series_tz)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(series_tz)
        else:
            end_ts = end_ts.tz_convert(series_tz)
    return dt, start_ts, end_ts

def _filter_df_by_time(df, start_time, end_time):
    """Filter dataframe by datetime using timezone-safe comparisons."""
    dt, start_ts, end_ts = _align_range_to_series(df["datetime"], start_time, end_time)
    mask = (dt >= start_ts) & (dt <= end_ts)
    filtered = df.loc[mask].copy()
    # Preserve timezone-aware dtype; `.values` can coerce to tz-naive UTC.
    filtered["datetime"] = dt.loc[mask]
    return filtered


def _resolve_state_event_end(event_row, default_duration_s=None):
    """
    Resolve state-event end time from duration-like fields (seconds) or explicit end-time fields.
    Returns (start_time, end_time) or (start_time, None) when unavailable.
    """
    start_time = event_row.get("datetime")
    if pd.isna(start_time):
        return start_time, None
    start_ts = pd.Timestamp(start_time)

    # Accept multiple duration column conventions used across deployments.
    for duration_key in ("duration", "duration_pos", "duration_sec", "duration_s"):
        if duration_key in event_row.index:
            duration_val = pd.to_numeric(pd.Series([event_row.get(duration_key)]), errors="coerce").iloc[0]
            if pd.notna(duration_val) and float(duration_val) > 0:
                return start_ts, start_ts + pd.to_timedelta(float(duration_val), unit="s")

    for end_key in ("end_datetime", "end_time", "datetime_end", "end"):
        if end_key in event_row.index:
            end_raw = event_row.get(end_key)
            end_ts = pd.to_datetime(end_raw, errors="coerce")
            if pd.notna(end_ts):
                end_ts = pd.Timestamp(end_ts)
                if start_ts.tzinfo is None and end_ts.tzinfo is not None:
                    end_ts = end_ts.tz_localize(None)
                elif start_ts.tzinfo is not None and end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize(start_ts.tzinfo)
                elif start_ts.tzinfo is not None and end_ts.tzinfo is not None:
                    end_ts = end_ts.tz_convert(start_ts.tzinfo)
                return start_ts, end_ts

    if default_duration_s is not None:
        try:
            dflt = float(default_duration_s)
            if np.isfinite(dflt) and dflt > 0:
                return start_ts, start_ts + pd.to_timedelta(dflt, unit="s")
        except Exception:
            pass

    return start_ts, None


def _merge_state_event_spans(state_events, gap_seconds=0.0, default_duration_s=None):
    """
    Merge adjacent/overlapping state-event intervals to reduce shape count.
    Expects a dataframe containing datetime/duration-style columns for one key.
    """
    if state_events is None or len(state_events) == 0:
        return []

    try:
        gap_td = pd.to_timedelta(float(gap_seconds), unit="s")
    except Exception:
        gap_td = pd.to_timedelta(0.0, unit="s")

    ordered = state_events.copy()
    if "datetime" in ordered.columns:
        ordered["datetime"] = _coerce_datetime_series(ordered["datetime"])
        ordered = ordered.sort_values("datetime")

    spans = []
    for _, event in ordered.iterrows():
        start_time, end_time = _resolve_state_event_end(event, default_duration_s=default_duration_s)
        if pd.isna(start_time) or end_time is None or pd.isna(end_time) or end_time <= start_time:
            continue

        if not spans:
            spans.append([start_time, end_time])
            continue

        last_start, last_end = spans[-1]
        if start_time <= (last_end + gap_td):
            if end_time > last_end:
                spans[-1][1] = end_time
        else:
            spans.append([start_time, end_time])

    return [(s, e) for s, e in spans]


def _build_state_segments(
    state_events,
    window_start=None,
    window_end=None,
    gap_seconds=0.0,
    default_duration_s=None,
):
    """Return sorted (x0, x1) spans clipped to the optional plotting window."""
    spans = _merge_state_event_spans(
        state_events,
        gap_seconds=gap_seconds,
        default_duration_s=default_duration_s,
    )
    if not spans:
        return []

    ref_start = spans[0][0]
    if window_start is not None:
        window_start = pd.Timestamp(window_start)
        if pd.Timestamp(ref_start).tzinfo is None:
            if window_start.tzinfo is not None:
                window_start = window_start.tz_localize(None)
        else:
            if window_start.tzinfo is None:
                window_start = window_start.tz_localize(pd.Timestamp(ref_start).tzinfo)
            else:
                window_start = window_start.tz_convert(pd.Timestamp(ref_start).tzinfo)
    if window_end is not None:
        window_end = pd.Timestamp(window_end)
        if pd.Timestamp(ref_start).tzinfo is None:
            if window_end.tzinfo is not None:
                window_end = window_end.tz_localize(None)
        else:
            if window_end.tzinfo is None:
                window_end = window_end.tz_localize(pd.Timestamp(ref_start).tzinfo)
            else:
                window_end = window_end.tz_convert(pd.Timestamp(ref_start).tzinfo)

    # Build segments as tuples first to allow sorting
    segments = []
    for start_time, end_time in spans:
        x0 = start_time
        x1 = end_time
        if window_start is not None:
            if x1 <= window_start:
                continue
            if x0 < window_start:
                x0 = window_start
        if window_end is not None:
            if x0 >= window_end:
                continue
            if x1 > window_end:
                x1 = window_end
        if pd.isna(x0) or pd.isna(x1) or x1 <= x0:
            continue
        segments.append((x0, x1))
    
    # Sort segments by start time to keep a monotonic x-axis for plotting.
    segments.sort(key=lambda seg: seg[0])

    return segments


def _build_state_segment_x(
    state_events,
    window_start=None,
    window_end=None,
    gap_seconds=0.0,
    default_duration_s=None,
):
    """Build [x0, x1, None, ...] segments for legacy callers."""
    segments = _build_state_segments(
        state_events,
        window_start=window_start,
        window_end=window_end,
        gap_seconds=gap_seconds,
        default_duration_s=default_duration_s,
    )
    if not segments:
        return []

    # Flatten to [x0, x1, None, x2, x3, None, ...] format
    x_parts = []
    for x0, x1 in segments:
        x_parts.extend([x0, x1, None])

    return x_parts


def _signal_overlay_center(signal_data):
    """Use signal median as the overlay baseline (same approach as lightweight cluster plots)."""
    if not isinstance(signal_data, pd.DataFrame) or signal_data.empty:
        return 0.0
    value_cols = [c for c in signal_data.columns if c != "datetime"]
    if not value_cols:
        return 0.0
    vals = pd.to_numeric(signal_data[value_cols].stack(), errors="coerce")
    vals = vals[np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))]
    if len(vals) == 0:
        return 0.0
    return float(np.nanmedian(vals.to_numpy(dtype=float)))


def _state_overlay_y_bounds(signal_data, target_channel=None, shade_mode="all_y", shade_pct_min=0.0, shade_pct_max=100.0):
    """Resolve y-range for state shading from signal data and shade style."""
    if not isinstance(signal_data, pd.DataFrame) or signal_data.empty:
        return 0.0, 1.0

    series = None
    if target_channel and target_channel in signal_data.columns:
        series = pd.to_numeric(signal_data[target_channel], errors="coerce")
    if series is None:
        value_cols = [c for c in signal_data.columns if c != "datetime"]
        if not value_cols:
            return 0.0, 1.0
        stacked = pd.to_numeric(signal_data[value_cols].stack(), errors="coerce")
        series = pd.Series(stacked.to_numpy(), dtype=float)

    vals = series[np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))]
    if vals.empty:
        return 0.0, 1.0

    y_min = float(vals.min())
    y_max = float(vals.max())
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return 0.0, 1.0
    if y_max <= y_min:
        return y_min - 0.5, y_max + 0.5

    mode = str(shade_mode or "all_y")
    if mode == "percent_band":
        try:
            p0 = float(shade_pct_min)
        except Exception:
            p0 = 0.0
        try:
            p1 = float(shade_pct_max)
        except Exception:
            p1 = 100.0
        p0 = max(0.0, min(100.0, p0))
        p1 = max(0.0, min(100.0, p1))
        if p1 < p0:
            p0, p1 = p1, p0
        span = y_max - y_min
        return y_min + span * (p0 / 100.0), y_min + span * (p1 / 100.0)

    if mode == "trace_to_zero":
        return min(y_min, 0.0), max(y_max, 0.0)

    return y_min, y_max


def _build_state_line_trace_xy(segments, y_level, bridge_seconds=0.0):
    """Build monotonic x / NaN-broken y vectors for thick state line overlays."""
    if not segments:
        return [], []
    try:
        bridge_td = pd.to_timedelta(float(bridge_seconds), unit="s")
    except Exception:
        bridge_td = pd.to_timedelta(0.0, unit="s")

    x_vals = []
    y_vals = []
    for i, (x0, x1) in enumerate(segments):
        if bridge_td > pd.Timedelta(0):
            x0 = x0 - bridge_td
            x1 = x1 + bridge_td
        x_vals.extend([x0, x1])
        y_vals.extend([y_level, y_level])
        if i < (len(segments) - 1):
            # Keep x monotonic while forcing a visual break between spans.
            x_vals.append(x1)
            y_vals.append(np.nan)
    return x_vals, y_vals


def _build_state_fill_trace_xy(segments, y0, y1, bridge_seconds=0.0):
    """Build batched rectangle polygons for trace-based state shading."""
    if not segments:
        return [], []
    try:
        bridge_td = pd.to_timedelta(float(bridge_seconds), unit="s")
    except Exception:
        bridge_td = pd.to_timedelta(0.0, unit="s")

    x_vals = []
    y_vals = []
    for x0, x1 in segments:
        if bridge_td > pd.Timedelta(0):
            x0 = x0 - bridge_td
            x1 = x1 + bridge_td
        # Keep x monotonic (x0,x0,x1,x1) so resampler-compatible traces remain valid.
        x_vals.extend([x0, x0, x1, x1, None])
        y_vals.extend([y0, y1, y1, y0, None])
    return x_vals, y_vals


def _state_overlay_above_bounds(signal_data, target_channel=None, min_factor=1.10, max_factor=1.20):
    """Resolve an above-signal band (e.g., 110%-120% of top signal level)."""
    y_min, y_max = _state_overlay_y_bounds(
        signal_data,
        target_channel=target_channel,
        shade_mode="all_y",
        shade_pct_min=0.0,
        shade_pct_max=100.0,
    )
    span = float(max(abs(y_max - y_min), 1e-9))
    try:
        f0 = float(min_factor)
    except Exception:
        f0 = 1.10
    try:
        f1 = float(max_factor)
    except Exception:
        f1 = 1.20
    if f1 < f0:
        f0, f1 = f1, f0

    if y_max > 0:
        y0 = y_max * f0
        y1 = y_max * f1
    else:
        # For zero/negative-top ranges, place an offset band above the top.
        y0 = y_max + ((f0 - 1.0) * span)
        y1 = y_max + ((f1 - 1.0) * span)

    if y1 <= y0:
        y0 = y_max + 0.10 * span
        y1 = y_max + 0.20 * span
    return y0, y1


def _state_overlay_split_bounds(
    signal_data,
    target_channel=None,
    bounds_mode="all_y",
    shade_pct_min=0.0,
    shade_pct_max=100.0,
    split_low_pct=45.0,
    split_high_pct=55.0,
):
    """Resolve lower/upper stripe bounds split around a center gap (e.g., 45%-55%)."""
    y_min, y_max = _state_overlay_y_bounds(
        signal_data,
        target_channel=target_channel,
        shade_mode=bounds_mode,
        shade_pct_min=shade_pct_min,
        shade_pct_max=shade_pct_max,
    )
    if y_max <= y_min:
        return y_min, y_min, y_max, y_max

    try:
        p_low = float(split_low_pct)
    except Exception:
        p_low = 45.0
    try:
        p_high = float(split_high_pct)
    except Exception:
        p_high = 55.0
    p_low = max(0.0, min(100.0, p_low))
    p_high = max(0.0, min(100.0, p_high))
    if p_high < p_low:
        p_low, p_high = p_high, p_low

    span = y_max - y_min
    y_lo0 = y_min
    y_lo1 = y_min + span * (p_low / 100.0)
    y_hi0 = y_min + span * (p_high / 100.0)
    y_hi1 = y_max
    return y_lo0, y_lo1, y_hi0, y_hi1


def _add_state_fill_trace(fig, row, col, segments, line_color, shade_opacity, line_bridge_seconds, y0, y1, legendgroup, hover_label=None):
    """Add trace-based state fill polygons to a subplot row."""
    fill_x, fill_y = _build_state_fill_trace_xy(
        segments,
        y0=y0,
        y1=y1,
        bridge_seconds=line_bridge_seconds,
    )
    if not fill_x:
        return
    fill_trace = go.Scatter(
        x=fill_x,
        y=fill_y,
        mode="lines",
        line=dict(width=0, color=line_color),
        fill="toself",
        fillcolor=line_color,
        opacity=shade_opacity,
        showlegend=False,
        hoveron="fills",
        hovertemplate=f"{_safe_text(hover_label) or _safe_text(legendgroup)}<extra></extra>",
        hoverlabel=_build_state_hoverlabel(line_color),
        legendgroup=str(legendgroup),
    )

    if isinstance(fig, FigureResampler):
        # Fill polygons are not intended for resampling; keep them as-is.
        fig.add_trace(
            fill_trace,
            row=row,
            col=1,
            max_n_samples=max(len(fill_x) + 10, 5000),
        )
    else:
        fig.add_trace(
            fill_trace,
            row=row,
            col=1,
        )


def _build_state_fill_to_zero_xy(signal_data, segments, target_channel=None, bridge_seconds=0.0):
    """Build x/y vectors that fill to y=0 only within the provided state segments."""
    if (
        signal_data is None
        or not isinstance(signal_data, pd.DataFrame)
        or signal_data.empty
        or not segments
        or "datetime" not in signal_data.columns
    ):
        return [], []

    channel = None
    if target_channel and target_channel in signal_data.columns:
        channel = target_channel
    else:
        value_cols = [c for c in signal_data.columns if c != "datetime"]
        if value_cols:
            channel = value_cols[0]
    if channel is None or channel not in signal_data.columns:
        return [], []

    dt = _coerce_datetime_series(signal_data["datetime"])
    vals = pd.to_numeric(signal_data[channel], errors="coerce")
    finite_vals = np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))
    valid_mask = dt.notna() & finite_vals
    if not valid_mask.any():
        return [], []

    work = pd.DataFrame(
        {
            "datetime": dt.loc[valid_mask],
            "value": vals.loc[valid_mask],
        }
    )
    if work.empty:
        return [], []
    work = work.sort_values("datetime")

    try:
        bridge_td = pd.to_timedelta(float(bridge_seconds), unit="s")
    except Exception:
        bridge_td = pd.to_timedelta(0.0, unit="s")

    in_segment = np.zeros(len(work), dtype=bool)
    work_dt = work["datetime"]
    series_tz = work_dt.dt.tz
    for x0, x1 in segments:
        seg_start = pd.Timestamp(x0)
        seg_end = pd.Timestamp(x1)

        if series_tz is None:
            if seg_start.tzinfo is not None:
                seg_start = seg_start.tz_localize(None)
            if seg_end.tzinfo is not None:
                seg_end = seg_end.tz_localize(None)
        else:
            if seg_start.tzinfo is None:
                seg_start = seg_start.tz_localize(series_tz)
            else:
                seg_start = seg_start.tz_convert(series_tz)
            if seg_end.tzinfo is None:
                seg_end = seg_end.tz_localize(series_tz)
            else:
                seg_end = seg_end.tz_convert(series_tz)

        if bridge_td > pd.Timedelta(0):
            seg_start = seg_start - bridge_td
            seg_end = seg_end + bridge_td

        if seg_end <= seg_start:
            continue
        in_segment |= ((work_dt >= seg_start) & (work_dt <= seg_end)).to_numpy()

    if not in_segment.any():
        return [], []

    y = work["value"].where(in_segment, np.nan)
    return work_dt.tolist(), y.tolist()


def _add_state_fill_trace_to_zero(
    fig,
    row,
    col,
    signal_data,
    segments,
    line_color,
    shade_opacity,
    line_bridge_seconds,
    legendgroup,
    hover_label=None,
    target_channel=None,
):
    """Add state fill trace that fills between a signal trace and y=0."""
    fill_x, fill_y = _build_state_fill_to_zero_xy(
        signal_data=signal_data,
        segments=segments,
        target_channel=target_channel,
        bridge_seconds=line_bridge_seconds,
    )
    if not fill_x:
        return

    fill_trace = go.Scatter(
        x=fill_x,
        y=fill_y,
        mode="lines",
        line=dict(width=0, color=line_color),
        fill="tozeroy",
        fillcolor=line_color,
        opacity=shade_opacity,
        showlegend=False,
        hovertemplate=f"{_safe_text(hover_label) or _safe_text(legendgroup)}<extra></extra>",
        hoverlabel=_build_state_hoverlabel(line_color),
        legendgroup=str(legendgroup),
        connectgaps=False,
    )

    if isinstance(fig, FigureResampler):
        fig.add_trace(
            fill_trace,
            row=row,
            col=1,
            max_n_samples=max(len(fill_x) + 10, 5000),
        )
    else:
        fig.add_trace(
            fill_trace,
            row=row,
            col=1,
        )


def _resolve_numeric_channel(signal_data, target_channel=None):
    if not isinstance(signal_data, pd.DataFrame) or signal_data.empty:
        return None
    if target_channel and target_channel in signal_data.columns:
        return target_channel
    for col in signal_data.columns:
        if col == "datetime":
            continue
        vals = pd.to_numeric(signal_data[col], errors="coerce")
        if vals.notna().any():
            return col
    return None


def _add_event_highlight_trace(
    fig,
    row,
    col,
    signal_data,
    segments,
    line_color,
    line_width,
    line_opacity,
    legendgroup,
    hover_label=None,
    target_channel=None,
    showlegend=False,
):
    """Overlay thicker, colored trace portions for highlighted event/state segments."""
    channel = _resolve_numeric_channel(signal_data, target_channel=target_channel)
    if channel is None:
        return

    hi_x, hi_y = _build_state_fill_to_zero_xy(
        signal_data=signal_data,
        segments=segments,
        target_channel=channel,
        bridge_seconds=0.0,
    )
    if not hi_x:
        return

    trace = go.Scattergl(
        x=hi_x,
        y=hi_y,
        mode="lines",
        line=dict(width=max(1.0, float(line_width)), color=line_color),
        opacity=max(0.01, min(1.0, float(line_opacity))),
        name=_safe_text(hover_label) or _safe_text(legendgroup) or "highlight",
        legendgroup=str(legendgroup),
        showlegend=bool(showlegend),
        connectgaps=False,
        hovertemplate=f"{_safe_text(hover_label) or _safe_text(legendgroup)}<extra></extra>",
        hoverlabel=_build_state_hoverlabel(line_color),
    )
    fig.add_trace(trace, row=row, col=col)


def _add_state_hover_markers(fig, row, col, segments, y0, y1, hover_label, legendgroup, swatch_color=None):
    """Add invisible midpoint markers so state segments expose detailed hover text."""
    if not segments:
        return
    x_vals = []
    y_vals = []
    text_vals = []
    y_mid = float(y0 + ((y1 - y0) / 2.0))
    label = _safe_text(hover_label) or _safe_text(legendgroup) or "state"
    safe_label = html.escape(label)
    safe_swatch = html.escape(_safe_text(swatch_color) or "rgba(150,150,150,0.7)")
    for x0, x1 in segments:
        try:
            mid = pd.Timestamp(x0) + ((pd.Timestamp(x1) - pd.Timestamp(x0)) / 2)
        except Exception:
            continue
        x_vals.append(mid)
        y_vals.append(y_mid)
        text_vals.append(
            "".join(
                [
                    "<span style='display:inline-block;width:10px;height:10px;",
                    "margin-right:6px;border:1px solid rgba(255,255,255,0.65);",
                    "vertical-align:middle;background:",
                    safe_swatch,
                    ";'></span>",
                    f"<b>{safe_label}</b>",
                ]
            )
        )
    if not x_vals:
        return
    hover_trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(size=16, color="rgba(0,0,0,0)"),
        opacity=1.0,
        showlegend=False,
        legendgroup=str(legendgroup),
        hovertemplate="%{text}<extra></extra>",
        hoverlabel=_build_state_hoverlabel(swatch_color),
        text=text_vals,
    )
    if isinstance(fig, FigureResampler):
        fig.add_trace(
            hover_trace,
            row=row,
            col=1,
            max_n_samples=max(len(x_vals) + 10, 5000),
        )
    else:
        fig.add_trace(
            hover_trace,
            row=row,
            col=1,
        )


def _build_state_channel_groups(state_annotations, state_event_keys, mode):
    """Build dedicated state-channel row groups, allowing multiple event keys per channel slot."""
    if not state_event_keys:
        return [], {}
    if mode == "combined":
        return ["__combined__"], {"__combined__": list(state_event_keys)}
    if mode != "separate":
        return [], {}

    groups = {}
    for event_key in state_event_keys:
        cfg = (state_annotations or {}).get(event_key, {})
        cfg0 = cfg[0] if isinstance(cfg, list) and cfg else (cfg if isinstance(cfg, dict) else {})
        slot = cfg0.get("state_channel")
        slot_key = _safe_text(slot) or str(event_key)
        groups.setdefault(slot_key, []).append(str(event_key))
    return list(groups.keys()), groups


def _resolve_state_numeric_level(event_type, state_events, fallback_level):
    """Resolve a numeric state level for accessibility channel plotting."""
    if isinstance(state_events, pd.DataFrame) and "value" in state_events.columns:
        vals = pd.to_numeric(state_events["value"], errors="coerce")
        vals = vals[np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))]
        if len(vals) > 0:
            return float(np.nanmedian(vals.to_numpy(dtype=float)))
    parsed = parse_cluster_rank_from_key(str(event_type))
    if parsed is not None:
        return float(parsed)
    return float(fallback_level)


def load_color_mapping(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Gracefully recover from concatenated/partially written JSON by
        # decoding the first valid JSON object prefix.
        try:
            with open(path, 'r') as f:
                raw = f.read()
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(raw.lstrip())
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    except Exception:
        return {}

def save_color_mapping(mapping, path):
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=4)

def generate_random_color():
    """Generate a random pastel color in HEX format."""
    import random
    r = lambda: random.randint(100, 255)
    return f'#{r():02x}{r():02x}{r():02x}'


def _resolve_color_for_channel(color_mapping, signal, channel):
    sig = str(signal or "")
    ch = str(channel or "")
    keys = (f"{sig}.{ch}", f"{sig}:{ch}", ch, ch.lower(), ch.upper())
    for key in keys:
        color = (color_mapping or {}).get(key)
        if isinstance(color, str) and color.strip():
            return color.strip()
    return None


def _is_placeholder_state_color(color_value):
    if color_value is None:
        return True
    c = str(color_value).strip().lower()
    if not c:
        return True
    # Existing generic placeholder grays used in notebooks/tools.
    if c in {"rgba(130,130,130,0.25)", "rgba(150, 150, 150, 0.3)", "rgba(150,150,150,0.3)"}:
        return True
    if c in {"#808080", "#999999", "#a0a0a0", "#b0b0b0", "gray", "grey"}:
        return True
    return False


def _normalize_state_annotations_input(state_annotations):
    """
    Normalize state_annotations to a dict[event_key] -> dict|list[dict].
    Accepts:
    - dict mapping event keys to cfg dict/list[dict]
    - iterable of event keys (auto-creates empty cfg dict)
    """
    if state_annotations is None:
        return {}
    if isinstance(state_annotations, dict):
        out = {}
        for k, v in state_annotations.items():
            if isinstance(v, list):
                clean = [cfg if isinstance(cfg, dict) else {} for cfg in v]
                out[str(k)] = clean
            elif isinstance(v, dict):
                out[str(k)] = v
            else:
                out[str(k)] = {}
        return out
    if isinstance(state_annotations, (list, tuple, set)):
        return {str(k): {} for k in state_annotations}
    return {}


def _has_wildcard_pattern(value):
    text = str(value or "")
    return any(ch in text for ch in ("*", "?", "[", "]"))


def _normalize_annotation_cfg_list(cfg):
    if isinstance(cfg, list):
        return [c if isinstance(c, dict) else {} for c in cfg]
    if isinstance(cfg, dict):
        return [cfg]
    return [{}]


def _append_annotation_cfg(existing_cfg, new_cfg):
    left = _normalize_annotation_cfg_list(existing_cfg)
    right = _normalize_annotation_cfg_list(new_cfg)
    merged = left + right
    return merged if len(merged) != 1 else merged[0]


def _expand_state_annotation_patterns(state_annotations, available_event_keys):
    """Expand wildcard state annotation keys (e.g., behavior_*) to matching event keys."""
    normalized = _normalize_state_annotations_input(state_annotations)
    if not normalized:
        return {}

    known = [str(k) for k in (available_event_keys or [])]
    expanded = {}
    for raw_key, cfg in normalized.items():
        key = str(raw_key)
        if _has_wildcard_pattern(key):
            matches = [k for k in known if fnmatch.fnmatch(k, key)]
            if not matches:
                expanded[key] = _append_annotation_cfg(expanded.get(key), cfg)
                continue
            for match_key in matches:
                expanded[match_key] = _append_annotation_cfg(expanded.get(match_key), cfg)
            continue
        expanded[key] = _append_annotation_cfg(expanded.get(key), cfg)
    return expanded


def _expand_note_annotation_patterns(note_annotations, available_event_keys):
    """Expand wildcard note annotation keys and default event_key to each matched key."""
    if not isinstance(note_annotations, dict):
        return note_annotations
    known = [str(k) for k in (available_event_keys or [])]
    expanded = {}
    for raw_key, cfg in note_annotations.items():
        key = str(raw_key)
        cfg_list = _normalize_annotation_cfg_list(cfg)
        if _has_wildcard_pattern(key):
            matches = [k for k in known if fnmatch.fnmatch(k, key)]
            if not matches:
                expanded[key] = _append_annotation_cfg(expanded.get(key), cfg_list)
                continue
            for match_key in matches:
                out_list = []
                for c in cfg_list:
                    c2 = dict(c or {})
                    c2.setdefault("event_key", match_key)
                    c2.setdefault("name", match_key)
                    out_list.append(c2)
                expanded[match_key] = _append_annotation_cfg(expanded.get(match_key), out_list)
            continue
        expanded[key] = _append_annotation_cfg(expanded.get(key), cfg_list)
    return expanded


def _state_signal_is_all_token(value):
    token = _safe_text(value).lower()
    return token in {"all", "__all__", "*"}


def _normalize_state_target_tokens(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    out = []
    seen = set()
    for raw in raw_values:
        token = _safe_text(raw)
        if not token:
            continue
        if "." in token:
            token = token.split(".", 1)[0].strip()
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _fallback_state_signal(event_key, signals_sorted, signal_data):
    available_signals = set(str(s) for s in (signal_data or {}).keys())
    ordered_signals = [str(s) for s in (signals_sorted or [])]

    key_lower = str(event_key).lower()
    if "dive" in key_lower and "depth" in available_signals:
        return "depth"

    for sig in ordered_signals:
        if not available_signals or sig in available_signals:
            return sig
    if available_signals:
        return next(iter(available_signals))
    return ""


def _resolve_state_target_signals(event_key, cfg, signals_sorted, signal_data, color_mapping):
    available_signals = set(str(s) for s in (signal_data or {}).keys())
    ordered_signals = [str(s) for s in (signals_sorted or [])]

    explicit_targets = _normalize_state_target_tokens((cfg or {}).get("signal"))
    event_targets = (color_mapping or {}).get("__event_targets__", {}) or {}
    mapped_targets = _normalize_state_target_tokens(event_targets.get(str(event_key), []))

    candidate_targets = explicit_targets or mapped_targets
    if candidate_targets and any(_state_signal_is_all_token(token) for token in candidate_targets):
        return [
            sig for sig in ordered_signals
            if not available_signals or sig in available_signals
        ]

    resolved = []
    seen = set()
    for sig in candidate_targets:
        sig_txt = str(sig)
        if sig_txt in seen:
            continue
        if sig_txt in ordered_signals and (not available_signals or sig_txt in available_signals):
            seen.add(sig_txt)
            resolved.append(sig_txt)
            continue
        if sig_txt in available_signals:
            seen.add(sig_txt)
            resolved.append(sig_txt)

    if resolved:
        return resolved

    fallback = _fallback_state_signal(
        event_key,
        signals_sorted=signals_sorted,
        signal_data=signal_data,
    )
    return [fallback] if fallback else []


def _pick_default_state_signal(event_key, cfg, signals_sorted, signal_data, color_mapping):
    targets = _resolve_state_target_signals(
        event_key,
        cfg,
        signals_sorted=signals_sorted,
        signal_data=signal_data,
        color_mapping=color_mapping,
    )
    return targets[0] if targets else ""


def _resolve_state_annotation_color(event_key, cfg, color_mapping):
    explicit_color = (cfg or {}).get("color")
    if isinstance(explicit_color, str) and explicit_color.strip() and not _is_placeholder_state_color(explicit_color):
        return explicit_color.strip()

    event_styles = (color_mapping or {}).get("__event_styles__", {}) or {}
    style = event_styles.get(str(event_key), {}) if isinstance(event_styles, dict) else {}

    mapped_color = (color_mapping or {}).get(str(event_key))
    if isinstance(mapped_color, str) and mapped_color.strip():
        return mapped_color.strip()

    style_color = style.get("shade_color") or style.get("color")
    if isinstance(style_color, str) and style_color.strip():
        return style_color.strip()

    if isinstance(explicit_color, str) and explicit_color.strip():
        return explicit_color.strip()
    return explicit_color


def _resolve_state_annotations_with_mapping(state_annotations, color_mapping, signals_sorted, signal_data):
    """
    Fill missing state annotation display config from color mapping metadata:
    - signal target from __event_targets__
    - color from top-level event key color, then __event_styles__
    - line_opacity from __event_styles__.shade_opacity when not provided
    """
    normalized = _normalize_state_annotations_input(state_annotations)
    if not normalized:
        return {}

    event_styles = (color_mapping or {}).get("__event_styles__", {}) or {}
    resolved = {}
    for event_key, raw_cfg in normalized.items():
        cfg_list = raw_cfg if isinstance(raw_cfg, list) else [raw_cfg]
        out_cfg_list = []
        for cfg in cfg_list:
            cfg0 = dict(cfg or {})
            cfg0["color"] = _resolve_state_annotation_color(
                event_key,
                cfg0,
                color_mapping=color_mapping,
            )

            style = event_styles.get(str(event_key), {}) if isinstance(event_styles, dict) else {}
            if "line_opacity" not in cfg0 and "event_alpha" not in cfg0 and isinstance(style, dict):
                shade_opacity = style.get("shade_opacity")
                if shade_opacity is not None:
                    cfg0["line_opacity"] = shade_opacity

            target_signals = _resolve_state_target_signals(
                event_key,
                cfg0,
                signals_sorted=signals_sorted,
                signal_data=signal_data,
                color_mapping=color_mapping,
            )
            if not target_signals:
                target_signals = [""]

            for target_signal in target_signals:
                expanded_cfg = dict(cfg0)
                expanded_cfg["signal"] = target_signal
                out_cfg_list.append(expanded_cfg)

        if isinstance(raw_cfg, list) or len(out_cfg_list) != 1:
            resolved[event_key] = out_cfg_list
        else:
            resolved[event_key] = out_cfg_list[0]
    return resolved


def plot_tag_data_interactive(data_pkl, signals=None, channels=None, 
                               time_range=None, note_annotations=None, state_annotations=None, color_mapping_path=None, 
                               target_sampling_rate=10, zoom_start_time=None, zoom_end_time=None, 
                               plot_event_values=None, zoom_range_selector_channel=None,
                               include_blank_row=None, preserve_signal_order=False,
                               signal_metadata=None, color_mapping=None,
                               persist_color_mapping=False,
                               render_state_on_signal_rows=True,
                               state_annotation_channel_mode=None,
                               state_annotation_channel_height_ratio=0.2,
                               state_annotation_channel_line_width=3.0):
    """
    Function to plot tag data interactively using Plotly with optional initial zooming into a specific time range.
    """
    register_plotly_resampler(mode='auto')
    # Default signal order
    default_order = ['ecg', 'pressure', 'accelerometer', 'magnetometer', 'gyroscope', 
                     'prh', 'temperature', 'light']

    # Load the color mapping
    if isinstance(color_mapping, dict):
        base = load_color_mapping(color_mapping_path) if color_mapping_path else {}
        base.update(color_mapping)
        color_mapping = base
    else:
        color_mapping = load_color_mapping(color_mapping_path) if color_mapping_path else {}

    # Determine the signals to plot
    if signals is None:
        signals = list(data_pkl.signal_data.keys())

    # Sort signals with the range selector signal on top if specified, unless UI order is explicit.
    if preserve_signal_order:
        signals_sorted = list(signals)
    elif zoom_range_selector_channel and zoom_range_selector_channel in signals:
        signals_sorted = [zoom_range_selector_channel] + [s for s in signals if s != zoom_range_selector_channel]
    else:
        signals_sorted = sorted(signals, key=lambda x: (default_order.index(x) 
                                                        if x in default_order else len(default_order) + signals.index(x)))

    available_event_keys = []
    if (
        hasattr(data_pkl, "event_data")
        and isinstance(data_pkl.event_data, pd.DataFrame)
        and "key" in data_pkl.event_data.columns
    ):
        available_event_keys = sorted(data_pkl.event_data["key"].astype(str).dropna().unique().tolist())

    state_annotations = _expand_state_annotation_patterns(
        state_annotations,
        available_event_keys=available_event_keys,
    )
    note_annotations = _expand_note_annotation_patterns(
        note_annotations,
        available_event_keys=available_event_keys,
    )

    state_annotations = _resolve_state_annotations_with_mapping(
        state_annotations=state_annotations,
        color_mapping=color_mapping,
        signals_sorted=signals_sorted,
        signal_data=getattr(data_pkl, "signal_data", {}) or {},
    )

    # Optional accessibility channel(s) for state annotations.
    mode = str(state_annotation_channel_mode or "").strip().lower()
    if mode not in {"combined", "separate"}:
        mode = None
    state_event_keys = [str(k) for k in (state_annotations or {}).keys()]
    state_channel_keys, state_channel_groups = _build_state_channel_groups(
        state_annotations=state_annotations,
        state_event_keys=state_event_keys,
        mode=mode,
    )
    state_channel_row_count = len(state_channel_keys)

    # Keep legacy spacer-row behavior for regular plots, but default to no spacer
    # when explicit state-annotation channel rows are requested.
    if include_blank_row is None:
        include_blank_row = state_channel_row_count == 0
    include_blank_row = bool(include_blank_row)

    # Add subplots: One row per signal, plus optional blank row, optional state channel rows,
    # and event value rows.
    extra_rows = len(plot_event_values) if plot_event_values else 0
    total_rows = len(signals_sorted) + extra_rows + (1 if include_blank_row else 0) + state_channel_row_count
    row_heights = [1.0] * total_rows
    if state_channel_row_count > 0:
        state_h = max(0.05, min(0.5, float(state_annotation_channel_height_ratio)))
        state_row_start = len(signals_sorted) + (1 if include_blank_row else 0) + 1
        for rr in range(state_row_start, state_row_start + state_channel_row_count):
            row_heights[rr - 1] = state_h
    fig = FigureResampler(
        make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights),
        default_downsampler=MinMaxLTTB(nan_policy="keep"),
    )
    row_counter = 1

    signal_row_meta = {}

    def plot_signal_data(signal, signal_data, signal_info):
        """General function to handle plotting signal data."""
        display_meta = _resolve_signal_display_metadata(signal, signal_info, signal_metadata=signal_metadata)
        # Determine the channels to plot for the current signal
        if channels is None or signal not in channels:
            signal_channels = signal_info['channels']
        else:
            signal_channels = channels[signal]

        # Filter data to the specified time range
        if time_range:
            start_time, end_time = time_range
            signal_data_filtered = _filter_df_by_time(signal_data, start_time, end_time)
        else:
            signal_data_filtered = signal_data.copy()
            signal_data_filtered["datetime"] = _coerce_datetime_series(signal_data_filtered["datetime"])

        # Drop invalid timestamps before sampling/plotting and keep order stable.
        signal_data_filtered = signal_data_filtered.dropna(subset=["datetime"])
        # Ensure datetime is sorted
        signal_data_filtered = signal_data_filtered.sort_values('datetime').reset_index(drop=True)

        # Calculate original sampling rate
        original_fs = calculate_sampling_frequency(signal_data_filtered['datetime'])

        # Downsample the data if needed
        signal_data_filtered = downsample(signal_data_filtered, original_fs, target_sampling_rate)

        for channel in signal_channels:
            if channel in signal_data_filtered.columns:
                # Use plain Python lists to avoid narwhals/duckdb inspection paths
                # that can fail in some environments with partially initialized duckdb.
                x_data = signal_data_filtered['datetime'].tolist()
                y_data = signal_data_filtered[channel].tolist()
                # Skip traces that contain no finite values to avoid downstream
                # empty-slice warnings in plotting/stat helpers.
                y_numeric = pd.to_numeric(pd.Series(y_data), errors="coerce")
                if y_numeric.notna().sum() == 0:
                    continue

                # Set labels and line properties
                ch_meta = (display_meta.get("channels", {}) or {}).get(channel, {})
                ch_label = _safe_text(ch_meta.get("label")) or channel
                unit = _safe_text(ch_meta.get("unit")) or _safe_text(signal_info['metadata'].get(channel, {}).get('unit', ''))
                ch_desc = _safe_text(ch_meta.get("description"))
                label_suffix = _safe_text(ch_meta.get("label_suffix"))
                y_label = _build_trace_legend_name(
                    signal,
                    channel,
                    display_meta.get("signal_label", signal),
                    label_suffix,
                )
                color = _resolve_color_for_channel(color_mapping, signal, channel) or generate_random_color()
                color_mapping[channel] = color

                line_width = ch_meta.get("line_width")
                line_kwargs = dict(color=color)
                if line_width is not None:
                    try:
                        line_kwargs["width"] = float(line_width)
                    except (TypeError, ValueError):
                        pass

                hovertemplate = _build_trace_hovertemplate(
                    signal,
                    channel,
                    display_meta.get("signal_label", signal),
                    label_suffix,
                    ch_desc,
                    unit,
                )
                fig.add_trace(
                    go.Scattergl(
                        name=y_label,
                        mode='lines',
                        line=dict(**line_kwargs),
                        hovertemplate=hovertemplate,
                        connectgaps=False,
                    ),
                              hf_x=x_data, hf_y=y_data,
                              row=row_counter, col=1,
                )
        signal_row_meta[row_counter] = {
            "title": _format_signal_axis_title(display_meta) or signal,
            "description": _safe_text(display_meta.get("signal_description")),
            "label": _safe_text(display_meta.get("signal_label")) or signal,
            "unit": _safe_text(display_meta.get("signal_unit")),
        }

    def _determine_annotation_window(requested_range):
        if requested_range and len(requested_range) == 2:
            return requested_range[0], requested_range[1]

        if (
            hasattr(data_pkl, 'event_data') and isinstance(data_pkl.event_data, pd.DataFrame)
            and not data_pkl.event_data.empty and 'datetime' in data_pkl.event_data
        ):
            return data_pkl.event_data['datetime'].min(), data_pkl.event_data['datetime'].max()

        return None, None

    missing_note_warnings = set()
    missing_state_warnings = set()
    plotted_state_legends = set()

    # Iterate through signals and plot relevant timeseries and events
    for signal in signals_sorted:
        if signal in data_pkl.signal_data:
            signal_data = data_pkl.signal_data[signal]
            signal_info = data_pkl.signal_info[signal]
            signal_plot_row = row_counter

            plot_signal_data(signal, signal_data, signal_info)
            # Reverse y-axis for depth or pressure signals
            if signal in ['pressure']:
                fig.update_yaxes(autorange="reversed", row=row_counter, col=1)
            
            if signal in ['depth']:
                fig.update_yaxes(autorange="reversed", row=row_counter, col=1)

            if include_blank_row and row_counter == 1:  # Right after the first plot
                # Add blank plot with height of 200 pixels after the first plot
                fig.add_trace(go.Scatter(x=[], y=[], mode='markers', showlegend=False), row=row_counter+1, col=1)
                fig.update_yaxes(showticklabels=False, row=row_counter+1, col=1)  # Hide tick labels
                fig.update_xaxes(showticklabels=False, row=row_counter+1, col=1)  # Hide tick labels
                row_counter += 1  # Skip to the next row after the blank plot

        # Plot note annotations if available
        if note_annotations:
            plotted_annotations = set()

            # Ensure time_range is valid
            start_time, end_time = _determine_annotation_window(time_range)
            if start_time is None or end_time is None:
                print("⚠ Cannot determine annotation window; skipping note annotations.")
                continue

            for note_type, note_cfg in (note_annotations or {}).items():
                configs = note_cfg if isinstance(note_cfg, list) else [note_cfg]
                for note_params in configs:
                    # Check if the annotation applies to the current signal
                    if signal != note_params.get("signal"):
                        continue  # Skip annotations not tied to this signal

                    symbol = note_params.get("symbol", "circle")
                    color = note_params.get("color", "rgba(128, 128, 128, 0.5)")
                    event_key = note_params.get("event_key", note_type)
                    legend_key = note_params.get("legend_key", note_type)
                    showlegend = note_params.get("showlegend", True)
                    label = note_params.get("name", note_type)

                    # Filter annotations based on the specified time range
                    notes_by_key = data_pkl.event_data[data_pkl.event_data["key"] == event_key]
                    filtered_notes = _filter_df_by_time(notes_by_key, start_time, end_time)

                    if filtered_notes.empty:
                        if event_key not in missing_note_warnings:
                            print(f"⚠ No '{event_key}' events found for plotting.")
                            missing_note_warnings.add(event_key)
                        continue

                    target_channel = note_params.get("channel")
                    if target_channel is None:
                        if "channels" in signal_info and signal_info["channels"]:
                            target_channel = signal_info["channels"][0]
                        else:
                            print(f"⚠ Warning: No valid channels found for signal '{signal}'. Skipping annotation.")
                            continue

                    if target_channel not in signal_data.columns:
                        print(f"⚠ Warning: Annotation channel '{target_channel}' not found for signal '{signal}'.")
                        continue

                    y_series = pd.to_numeric(signal_data[target_channel], errors="coerce").dropna()
                    if y_series.empty:
                        y_min, y_max = 0.0, 1.0
                    else:
                        y_min = float(y_series.min())
                        y_max = float(y_series.max())
                    y_span = y_max - y_min
                    y_offset_frac = float(note_params.get("y_offset_frac", 0.0) or 0.0)
                    if np.isfinite(y_span) and y_span > 0:
                        y_fixed = y_max + (y_offset_frac * y_span)
                    elif np.isfinite(y_max):
                        y_fixed = y_max + y_offset_frac
                    else:
                        y_fixed = 1.0
                    scatter_x = filtered_notes["datetime"]
                    scatter_y = [y_fixed] * len(filtered_notes)

                    # Add point event markers
                    fig.add_trace(go.Scatter(
                        x=scatter_x,
                        y=scatter_y,
                        mode="markers",
                        marker=dict(symbol=symbol, color=color, size=10),
                        name=label,
                        opacity=0.5,
                        showlegend=showlegend and (legend_key not in plotted_annotations)
                    ), row=row_counter, col=1)

                    if bool(note_params.get("highlight_trace", False)):
                        try:
                            highlight_width = float(note_params.get("highlight_line_width", 3.0))
                        except Exception:
                            highlight_width = 3.0
                        try:
                            highlight_opacity = float(note_params.get("highlight_opacity", 0.95))
                        except Exception:
                            highlight_opacity = 0.95
                        try:
                            highlight_window_seconds = float(
                                note_params.get(
                                    "highlight_window_seconds",
                                    note_params.get("default_duration_s", 60.0),
                                )
                            )
                        except Exception:
                            highlight_window_seconds = 60.0
                        try:
                            highlight_merge_gap_seconds = float(note_params.get("highlight_merge_gap_seconds", 0.0))
                        except Exception:
                            highlight_merge_gap_seconds = 0.0

                        highlight_segments = _build_state_segments(
                            filtered_notes,
                            window_start=start_time,
                            window_end=end_time,
                            gap_seconds=max(0.0, highlight_merge_gap_seconds),
                            default_duration_s=max(0.1, highlight_window_seconds),
                        )
                        _add_event_highlight_trace(
                            fig,
                            row=signal_plot_row,
                            col=1,
                            signal_data=signal_data,
                            segments=highlight_segments,
                            line_color=color,
                            line_width=max(1.0, highlight_width),
                            line_opacity=max(0.01, min(1.0, highlight_opacity)),
                            legendgroup=legend_key,
                            hover_label=f"{label} highlight",
                            target_channel=target_channel,
                            showlegend=bool(note_params.get("highlight_showlegend", False)),
                        )

                    # Mark the annotation as plotted to avoid duplicate legends
                    plotted_annotations.add(legend_key)

        # Plot state annotations independently from note annotations.
        if state_annotations and render_state_on_signal_rows:
            state_event_keys = [str(k) for k in (state_annotations or {}).keys()]
            ordered_cluster_color_map = {}
            if state_event_keys and all(looks_like_ordered_cluster_key(k) for k in state_event_keys):
                ordered_cluster_color_map = build_ordered_cluster_color_map(state_event_keys)
            for event_type, event_cfg in state_annotations.items():
                # allow either a single dict or a list of dicts
                if isinstance(event_cfg, list):
                    configs = event_cfg
                else:
                    configs = [event_cfg]

                # same events for this event_type, independent of signal
                start_time, end_time = _determine_annotation_window(time_range)
                if start_time is None or end_time is None:
                    print("⚠ Cannot determine annotation window; skipping state annotations.")
                    break

                state_events = data_pkl.event_data[data_pkl.event_data["key"] == event_type].copy()
                if "datetime" in state_events.columns:
                    state_events["datetime"] = _coerce_datetime_series(state_events["datetime"])

                for event_params in configs:
                    # only draw shapes on the currently plotted signal
                    if signal != event_params.get("signal"):
                        continue

                    line_color = event_params.get("color")
                    if _is_placeholder_state_color(line_color):
                        line_color = ordered_cluster_color_map.get(str(event_type), "rgba(150, 150, 150, 0.35)")

                    try:
                        line_width = float(event_params.get("line_width", event_params.get("event_line_width", 12)))
                    except Exception:
                        line_width = 12.0
                    line_width = max(1.0, min(2000.0, line_width))

                    try:
                        line_opacity = float(
                            event_params.get(
                                "line_opacity",
                                event_params.get("event_alpha", event_params.get("shade_opacity", 0.4)),
                            )
                        )
                    except Exception:
                        line_opacity = 0.4
                    line_opacity = max(0.01, min(1.0, line_opacity))
                    try:
                        merge_gap_seconds = float(event_params.get("merge_gap_seconds", 0.0))
                    except Exception:
                        merge_gap_seconds = 0.0
                    merge_gap_seconds = max(0.0, merge_gap_seconds)
                    try:
                        line_bridge_seconds = float(event_params.get("line_bridge_seconds", 0.0))
                    except Exception:
                        line_bridge_seconds = 0.0
                    line_bridge_seconds = max(0.0, line_bridge_seconds)
                    legend_key = str(event_params.get("legend_key", event_type))
                    label = event_params.get("name", str(event_type))
                    safe_hover_label = "".join(
                        [
                            "<span style='display:inline-block;width:10px;height:10px;",
                            "margin-right:6px;border:1px solid rgba(255,255,255,0.65);",
                            "vertical-align:middle;background:",
                            html.escape(_safe_text(line_color) or "rgba(150,150,150,0.7)"),
                            ";'></span>",
                            f"<b>{html.escape(_safe_text(label) or str(event_type))}</b>",
                        ]
                    )

                    default_duration_s = event_params.get("default_duration_s", 60.0)
                    segments = _build_state_segments(
                        state_events,
                        window_start=start_time,
                        window_end=end_time,
                        gap_seconds=merge_gap_seconds,
                        default_duration_s=default_duration_s,
                    )
                    if not segments:
                        if event_type not in missing_state_warnings:
                            print(f"⚠ No '{event_type}' state events found for plotting.")
                            missing_state_warnings.add(event_type)
                        continue

                    shade_mode = str(event_params.get("shade_mode", "")).strip().lower()
                    if shade_mode in {"fill_trace", "fill_trace_above", "fill_trace_split", "fill_trace_to_zero"}:
                        try:
                            shade_opacity = float(
                                event_params.get(
                                    "shade_opacity",
                                    event_params.get("event_alpha", line_opacity),
                                )
                            )
                        except Exception:
                            shade_opacity = line_opacity
                        shade_opacity = max(0.01, min(1.0, shade_opacity))

                        if shade_mode == "fill_trace_to_zero":
                            _add_state_fill_trace_to_zero(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                signal_data=signal_data,
                                segments=segments,
                                line_color=line_color,
                                shade_opacity=shade_opacity,
                                line_bridge_seconds=line_bridge_seconds,
                                legendgroup=event_params.get("legend_key", event_type),
                                hover_label=label,
                                target_channel=event_params.get("channel"),
                            )
                            y0, y1 = _state_overlay_y_bounds(
                                signal_data,
                                target_channel=event_params.get("channel"),
                                shade_mode="trace_to_zero",
                                shade_pct_min=event_params.get("shade_pct_min", 0.0),
                                shade_pct_max=event_params.get("shade_pct_max", 100.0),
                            )
                            _add_state_hover_markers(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                y0=y0,
                                y1=y1,
                                hover_label=label,
                                legendgroup=event_params.get("legend_key", event_type),
                                swatch_color=line_color,
                            )
                        elif shade_mode == "fill_trace_above":
                            y0, y1 = _state_overlay_above_bounds(
                                signal_data,
                                target_channel=event_params.get("channel"),
                                min_factor=float(event_params.get("above_y_pct_min", 110.0)) / 100.0,
                                max_factor=float(event_params.get("above_y_pct_max", 120.0)) / 100.0,
                            )
                            _add_state_fill_trace(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                line_color=line_color,
                                shade_opacity=shade_opacity,
                                line_bridge_seconds=line_bridge_seconds,
                                y0=y0,
                                y1=y1,
                                legendgroup=event_params.get("legend_key", event_type),
                                hover_label=label,
                            )
                            _add_state_hover_markers(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                y0=y0,
                                y1=y1,
                                hover_label=label,
                                legendgroup=event_params.get("legend_key", event_type),
                                swatch_color=line_color,
                            )
                        elif shade_mode == "fill_trace_split":
                            bounds_mode = str(event_params.get("shade_bounds_mode", "all_y")).strip().lower() or "all_y"
                            y_lo0, y_lo1, y_hi0, y_hi1 = _state_overlay_split_bounds(
                                signal_data,
                                target_channel=event_params.get("channel"),
                                bounds_mode=bounds_mode,
                                shade_pct_min=event_params.get("shade_pct_min", 0.0),
                                shade_pct_max=event_params.get("shade_pct_max", 100.0),
                                split_low_pct=event_params.get("split_gap_low_pct", 45.0),
                                split_high_pct=event_params.get("split_gap_high_pct", 55.0),
                            )
                            _add_state_fill_trace(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                line_color=line_color,
                                shade_opacity=shade_opacity,
                                line_bridge_seconds=line_bridge_seconds,
                                y0=y_lo0,
                                y1=y_lo1,
                                legendgroup=event_params.get("legend_key", event_type),
                                hover_label=label,
                            )
                            _add_state_fill_trace(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                line_color=line_color,
                                shade_opacity=shade_opacity,
                                line_bridge_seconds=line_bridge_seconds,
                                y0=y_hi0,
                                y1=y_hi1,
                                legendgroup=event_params.get("legend_key", event_type),
                                hover_label=label,
                            )
                            _add_state_hover_markers(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                y0=y_lo0,
                                y1=y_lo1,
                                hover_label=label,
                                legendgroup=event_params.get("legend_key", event_type),
                                swatch_color=line_color,
                            )
                            _add_state_hover_markers(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                y0=y_hi0,
                                y1=y_hi1,
                                hover_label=label,
                                legendgroup=event_params.get("legend_key", event_type),
                                swatch_color=line_color,
                            )
                        else:
                            bounds_mode = str(event_params.get("shade_bounds_mode", "all_y")).strip().lower() or "all_y"
                            y0, y1 = _state_overlay_y_bounds(
                                signal_data,
                                target_channel=event_params.get("channel"),
                                shade_mode=bounds_mode,
                                shade_pct_min=event_params.get("shade_pct_min", 0.0),
                                shade_pct_max=event_params.get("shade_pct_max", 100.0),
                            )
                            _add_state_fill_trace(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                line_color=line_color,
                                shade_opacity=shade_opacity,
                                line_bridge_seconds=line_bridge_seconds,
                                y0=y0,
                                y1=y1,
                                legendgroup=event_params.get("legend_key", event_type),
                                hover_label=label,
                            )
                            _add_state_hover_markers(
                                fig,
                                row=signal_plot_row,
                                col=1,
                                segments=segments,
                                y0=y0,
                                y1=y1,
                                hover_label=label,
                                legendgroup=event_params.get("legend_key", event_type),
                                swatch_color=line_color,
                            )
                    elif shade_mode:
                        y0, y1 = _state_overlay_y_bounds(
                            signal_data,
                            target_channel=event_params.get("channel"),
                            shade_mode=shade_mode,
                            shade_pct_min=event_params.get("shade_pct_min", 0.0),
                            shade_pct_max=event_params.get("shade_pct_max", 100.0),
                        )
                        try:
                            shade_opacity = float(
                                event_params.get(
                                    "shade_opacity",
                                    event_params.get("event_alpha", line_opacity),
                                )
                            )
                        except Exception:
                            shade_opacity = line_opacity
                        shade_opacity = max(0.01, min(1.0, shade_opacity))

                        for x0, x1 in segments:
                            fig.add_shape(
                                type="rect",
                                x0=x0,
                                x1=x1,
                                y0=y0,
                                y1=y1,
                                fillcolor=line_color,
                                opacity=shade_opacity,
                                line=dict(width=0),
                                layer="below",
                                row=signal_plot_row,
                                col=1,
                            )

                    if bool(event_params.get("highlight_trace", False)):
                        try:
                            highlight_width = float(event_params.get("highlight_line_width", max(3.0, line_width)))
                        except Exception:
                            highlight_width = max(3.0, line_width)
                        try:
                            highlight_opacity = float(event_params.get("highlight_opacity", 0.95))
                        except Exception:
                            highlight_opacity = 0.95
                        highlight_channel = event_params.get("highlight_channel", event_params.get("channel"))
                        _add_event_highlight_trace(
                            fig,
                            row=signal_plot_row,
                            col=1,
                            signal_data=signal_data,
                            segments=segments,
                            line_color=line_color,
                            line_width=max(1.0, highlight_width),
                            line_opacity=max(0.01, min(1.0, highlight_opacity)),
                            legendgroup=event_params.get("legend_key", event_type),
                            hover_label=f"{label} highlight",
                            target_channel=highlight_channel,
                            showlegend=bool(event_params.get("highlight_showlegend", False)),
                        )

                    show_start_marker = bool(
                        event_params.get(
                            "start_marker",
                            event_params.get("show_start_marker", False),
                        )
                    )
                    draw_line = bool(event_params.get("draw_line", True))
                    y_center = _signal_overlay_center(signal_data)
                    if show_start_marker:
                        marker_x = [x0 for x0, _ in segments]
                        marker_y = [y_center] * len(marker_x)
                        try:
                            marker_size = float(event_params.get("start_marker_size", 8.0))
                        except Exception:
                            marker_size = 8.0
                        marker_size = max(2.0, min(64.0, marker_size))
                        marker_symbol = str(event_params.get("start_marker_symbol", "circle"))
                        try:
                            marker_opacity = float(event_params.get("start_marker_opacity", 0.95))
                        except Exception:
                            marker_opacity = 0.95
                        marker_opacity = max(0.01, min(1.0, marker_opacity))

                        fig.add_trace(
                            go.Scatter(
                                x=marker_x,
                                y=marker_y,
                                mode="markers",
                                marker=dict(
                                    symbol=marker_symbol,
                                    size=marker_size,
                                    color=line_color,
                                    line=dict(color="rgba(255,255,255,0.9)", width=1),
                                ),
                                opacity=marker_opacity,
                                name=f"{label} start",
                                legendgroup=legend_key,
                                showlegend=False,
                                hovertemplate=f"{safe_hover_label}<extra></extra>",
                                hoverlabel=_build_state_hoverlabel(line_color),
                            ),
                            row=signal_plot_row,
                            col=1,
                        )

                    showlegend = bool(event_params.get("showlegend", True)) and legend_key not in plotted_state_legends
                    if draw_line:
                        seg_x, seg_y = _build_state_line_trace_xy(
                            segments,
                            y_center,
                            bridge_seconds=line_bridge_seconds,
                        )
                        if not seg_x:
                            continue

                        state_trace = go.Scatter(
                            x=seg_x,
                            y=seg_y,
                            mode="lines",
                            line=dict(width=line_width, color=line_color),
                            opacity=line_opacity,
                            zorder=-1,
                            name=label,
                            legendgroup=legend_key,
                            showlegend=showlegend,
                            hovertemplate=f"{safe_hover_label}<extra></extra>",
                            hoverlabel=_build_state_hoverlabel(line_color),
                        )
                        if isinstance(fig, FigureResampler):
                            fig.add_trace(
                                state_trace,
                                row=signal_plot_row,
                                col=1,
                                max_n_samples=max(len(seg_x) + 10, 5000),
                            )
                        else:
                            fig.add_trace(
                                state_trace,
                                row=signal_plot_row,
                                col=1,
                            )
                        if showlegend:
                            plotted_state_legends.add(legend_key)

        # Update y-axis label for each subplot
        axis_title = signal_row_meta.get(row_counter, {}).get("title", signal)
        if include_blank_row and row_counter == 2:
            # Align the title of the blank plot (row 2) with the first plot (row 1)
            fig.update_yaxes(title_text="", row=1, col=1)
        else:
            # Keep the title where it is for the other rows
            fig.update_yaxes(title_text="", row=row_counter, col=1)
        row_counter += 1

    # Optional accessibility channel(s): render state annotations as numeric line tracks.
    if state_channel_row_count > 0 and state_annotations:
        start_time, end_time = _determine_annotation_window(time_range)
        ordered_cluster_color_map = {}
        if state_event_keys and all(looks_like_ordered_cluster_key(k) for k in state_event_keys):
            ordered_cluster_color_map = build_ordered_cluster_color_map(state_event_keys)

        state_key_to_level = {}
        for i, k in enumerate(sorted(state_event_keys), start=1):
            key_events = data_pkl.event_data[data_pkl.event_data["key"] == k].copy()
            if "datetime" in key_events.columns:
                key_events["datetime"] = _coerce_datetime_series(key_events["datetime"])
            state_key_to_level[k] = _resolve_state_numeric_level(k, key_events, fallback_level=i)

        for channel_key in state_channel_keys:
            row_for_channel = row_counter
            keys_for_row = state_channel_groups.get(
                channel_key,
                state_event_keys if channel_key == "__combined__" else [channel_key],
            )
            any_trace = False
            for j, event_type in enumerate(keys_for_row, start=1):
                cfg = (state_annotations or {}).get(event_type, {})
                cfg0 = cfg[0] if isinstance(cfg, list) and cfg else (cfg if isinstance(cfg, dict) else {})
                state_events = data_pkl.event_data[data_pkl.event_data["key"] == event_type].copy()
                if "datetime" in state_events.columns:
                    state_events["datetime"] = _coerce_datetime_series(state_events["datetime"])
                segments = _build_state_segments(
                    state_events,
                    window_start=start_time,
                    window_end=end_time,
                    gap_seconds=float(cfg0.get("merge_gap_seconds", 0.0) or 0.0),
                    default_duration_s=cfg0.get("default_duration_s", 60.0),
                )
                if not segments:
                    continue
                line_color = cfg0.get("color")
                if _is_placeholder_state_color(line_color):
                    line_color = ordered_cluster_color_map.get(str(event_type), "rgba(120,120,120,0.9)")
                safe_hover_label = "".join(
                    [
                        "<span style='display:inline-block;width:10px;height:10px;",
                        "margin-right:6px;border:1px solid rgba(255,255,255,0.65);",
                        "vertical-align:middle;background:",
                        html.escape(_safe_text(line_color) or "rgba(120,120,120,0.9)"),
                        ";'></span>",
                        f"<b>{html.escape(_safe_text(event_type) or 'state')}</b>",
                    ]
                )
                y_level = state_key_to_level.get(event_type, float(j))
                bridge_seconds = float(cfg0.get("line_bridge_seconds", 0.0) or 0.0)
                channel_draw_mode = str(cfg0.get("state_channel_draw_mode", "line")).strip().lower()
                try:
                    channel_fill_opacity = float(cfg0.get("state_channel_fill_opacity", cfg0.get("shade_opacity", 0.25)))
                except Exception:
                    channel_fill_opacity = 0.25
                channel_fill_opacity = max(0.01, min(1.0, channel_fill_opacity))

                if channel_draw_mode in {"fill_trace", "fill_trace_split"}:
                    try:
                        half_h = float(cfg0.get("state_channel_fill_half_height", 0.42))
                    except Exception:
                        half_h = 0.42
                    half_h = max(0.02, min(2.0, half_h))
                    y0 = y_level - half_h
                    y1 = y_level + half_h

                    if channel_draw_mode == "fill_trace_split":
                        try:
                            p_low = float(cfg0.get("split_gap_low_pct", 45.0))
                        except Exception:
                            p_low = 45.0
                        try:
                            p_high = float(cfg0.get("split_gap_high_pct", 55.0))
                        except Exception:
                            p_high = 55.0
                        p_low = max(0.0, min(100.0, p_low))
                        p_high = max(0.0, min(100.0, p_high))
                        if p_high < p_low:
                            p_low, p_high = p_high, p_low

                        span = y1 - y0
                        y_lo0, y_lo1 = y0, y0 + span * (p_low / 100.0)
                        y_hi0, y_hi1 = y0 + span * (p_high / 100.0), y1
                        _add_state_fill_trace(
                            fig,
                            row=row_for_channel,
                            col=1,
                            segments=segments,
                            line_color=line_color,
                            shade_opacity=channel_fill_opacity,
                            line_bridge_seconds=bridge_seconds,
                            y0=y_lo0,
                            y1=y_lo1,
                            legendgroup=f"state_access_{event_type}",
                        )
                        _add_state_fill_trace(
                            fig,
                            row=row_for_channel,
                            col=1,
                            segments=segments,
                            line_color=line_color,
                            shade_opacity=channel_fill_opacity,
                            line_bridge_seconds=bridge_seconds,
                            y0=y_hi0,
                            y1=y_hi1,
                            legendgroup=f"state_access_{event_type}",
                        )
                    else:
                        _add_state_fill_trace(
                            fig,
                            row=row_for_channel,
                            col=1,
                            segments=segments,
                            line_color=line_color,
                            shade_opacity=channel_fill_opacity,
                            line_bridge_seconds=bridge_seconds,
                            y0=y0,
                            y1=y1,
                            legendgroup=f"state_access_{event_type}",
                        )
                    any_trace = True

                draw_channel_line = bool(cfg0.get("state_channel_draw_line", channel_draw_mode == "line"))
                if draw_channel_line:
                    seg_x, seg_y = _build_state_line_trace_xy(
                        segments,
                        y_level=y_level,
                        bridge_seconds=bridge_seconds,
                    )
                    if seg_x:
                        any_trace = True
                        fig.add_trace(
                            go.Scattergl(
                                x=seg_x,
                                y=seg_y,
                                mode="lines",
                                line=dict(
                                    width=max(1.0, float(state_annotation_channel_line_width)),
                                    color=line_color,
                                ),
                                opacity=max(0.05, min(1.0, float(cfg0.get("line_opacity", cfg0.get("event_alpha", 0.4)) or 0.4))),
                                name=f"{event_type} state",
                                legendgroup=f"state_access_{event_type}",
                                showlegend=True,
                                hovertemplate=f"{safe_hover_label}<extra></extra>",
                                hoverlabel=_build_state_hoverlabel(line_color),
                            ),
                            row=row_for_channel,
                            col=1,
                        )
            fig.update_yaxes(
                title_text=("State rank" if channel_key == "__combined__" else f"State channel {channel_key}"),
                row=row_for_channel,
                col=1,
            )
            if not any_trace:
                fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False), row=row_for_channel, col=1)
            row_counter += 1

    # Add event values as separate subplots
    if plot_event_values:
        for event_type in plot_event_values:
            event_data = data_pkl.event_data[data_pkl.event_data['key'] == event_type]
            if not event_data.empty:
                fig.add_trace(go.Scatter(
                    x=event_data['datetime'],
                    y=[1] * len(event_data),
                    mode='markers',
                    name=f"{event_type} events"
                ), row=row_counter, col=1)
                fig.update_yaxes(title_text=f"{event_type} events", row=row_counter, col=1)
                row_counter += 1

    # Apply zoom and configure rangeslider for the bottom subplot
    if zoom_start_time and zoom_end_time:
        fig.update_xaxes(range=[zoom_start_time, zoom_end_time])

    # Configure shared x-axis and only expose range controls when requested.
    show_range_controls = bool(zoom_range_selector_channel)
    fig.update_layout(
        title='Tag Data Visualization',
        xaxis=dict(
            rangeselector=dict(
                visible=show_range_controls,
                buttons=[
                    dict(count=30, label="30s", step="second", stepmode="backward"),
                    dict(count=5, label="5m", step="minute", stepmode="backward"),
                    dict(count=10, label="10m", step="minute", stepmode="backward"),
                    dict(count=30, label="30m", step="minute", stepmode="backward"),
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            ),
            rangeslider=dict(
                visible=show_range_controls,
                thickness=0.15
            ),
            type="date"
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=170, r=20, t=72, b=90),
        height=600 + 50 * (len(signals_sorted) + extra_rows) + int(120 * state_channel_row_count),
        showlegend=True
    )

    for row_num, meta in signal_row_meta.items():
        _add_signal_axis_label_annotation(
            fig,
            row_num,
            total_rows,
            meta.get("label"),
            meta.get("unit"),
        )
        _add_signal_help_annotation(fig, row_num, total_rows, meta.get("description"))

    if persist_color_mapping and color_mapping_path:
        try:
            save_color_mapping(color_mapping, color_mapping_path)
        except Exception:
            pass

    return fig

def plot_tag_data_interactive_st(data_pkl, signals=None, channels=None, 
                               time_range=None, note_annotations=None, state_annotations=None, color_mapping_path=None, 
                               target_sampling_rate=10, zoom_start_time=None, zoom_end_time=None, 
                               plot_event_values=None, zoom_range_selector_channel=None,
                               preserve_signal_order=False,
                               include_blank_row=None,
                               render_state_on_signal_rows=True,
                               state_annotation_channel_mode=None,
                               state_annotation_channel_height_ratio=0.2,
                               state_annotation_channel_line_width=3.0):
    """
    Streamlit-oriented wrapper for plot_tag_data_interactive.

    Keeps the legacy API surface while delegating all rendering logic to the
    canonical implementation so state-event behavior stays consistent.
    """
    return plot_tag_data_interactive(
        data_pkl=data_pkl,
        signals=signals,
        channels=channels,
        time_range=time_range,
        note_annotations=note_annotations,
        state_annotations=state_annotations,
        color_mapping_path=color_mapping_path,
        target_sampling_rate=target_sampling_rate,
        zoom_start_time=zoom_start_time,
        zoom_end_time=zoom_end_time,
        plot_event_values=plot_event_values,
        zoom_range_selector_channel=zoom_range_selector_channel,
        include_blank_row=include_blank_row,
        preserve_signal_order=preserve_signal_order,
        render_state_on_signal_rows=render_state_on_signal_rows,
        state_annotation_channel_mode=state_annotation_channel_mode,
        state_annotation_channel_height_ratio=state_annotation_channel_height_ratio,
        state_annotation_channel_line_width=state_annotation_channel_line_width,
    )

def get_bathy(
    lat_min,
    lat_max,
    lon0_180,
    lon1_180,
    gebco_nc_path=None,
    max_pixels=1200,
):
    """
    Load and process GEBCO bathymetry/elevation data for a specified extent.

    Parameters
    ----------
    lat_min : float
        Minimum latitude in degrees (-90 to 90).
    lat_max : float
        Maximum latitude in degrees (-90 to 90).
    lon0_180 : float
        Minimum longitude in degrees (-180 to 180).
    lon1_180 : float
        Maximum longitude in degrees (-180 to 180).
    gebco_nc_path : str, optional
        Path to GEBCO NetCDF file. If None, uses default GEBCO 2025 (with ice elevation).
    max_pixels : int, default 1200
        Maximum dimension for coarsening bathymetry data.

    Returns
    -------
    bathy_df : pd.DataFrame
        DataFrame with columns: ['lon360', 'lat', 'bathy'] containing bathymetry/elevation data.
    """
    import numpy as np
    import pandas as pd
    def _coarsen_to_max_pixels(da, max_n=1200):
        ny, nx = da.shape
        factor = int(np.ceil(max((nx * ny) / (max_n * max_n), 1.0)) ** 0.5)
        if factor <= 1:
            return da
        return da.coarsen(lon=factor, lat=factor, boundary="trim").mean()

    def _wrap_lon180_edges(lon0_180, lon1_180):
        """Return lon slices in [-180,180] handling dateline crossing."""
        if lon0_180 <= lon1_180:
            return [(lon0_180, lon1_180)]
        # crosses dateline: [-180, lon1] U [lon0, 180]
        return [(-180.0, lon1_180), (lon0_180, 180.0)]

    # Default to GEBCO 2025 with ice elevation
    if gebco_nc_path is None:
        gebco_nc_path = "/Volumes/WORK-SSD/Datasets/Published/GEBCO/gebco_2025/GEBCO_2025.nc"

    print(f"Loading bathymetry from: {gebco_nc_path}")
    ds_meta = _open_bathy_dataset_cached(gebco_nc_path)
    z = ds_meta["ds"][ds_meta["z_name"]].rename({ds_meta["lon_name"]: "lon", ds_meta["lat_name"]: "lat"})

    # The cached dataset metadata is derived from the header so later calls can
    # jump straight to spatial querying without reopening the NetCDF.
    lat_slice = slice(lat_min, lat_max) if ds_meta["lat_ascending"] else slice(lat_max, lat_min)

    # Subset with dateline handling
    lon_slices = _wrap_lon180_edges(lon0_180, lon1_180)

    with _progress_log(
        (
            f"query bathymetry slice lat=[{lat_min:.2f},{lat_max:.2f}] "
            f"lon180=[{lon0_180:.2f},{lon1_180:.2f}] max_pixels={max_pixels}"
        )
    ):
        parts = []
        for lo, hi in lon_slices:
            parts.append(z.sel(lon=slice(lo, hi), lat=lat_slice))

        if len(parts) > 1:
            import xarray as xr
            z_sub = xr.concat(parts, dim="lon")
        else:
            z_sub = parts[0]

        # Coarsen the queried slice before materializing to dataframe.
        z_sub = _coarsen_to_max_pixels(z_sub, max_n=max_pixels)

        # Convert lon back to 0..360
        z_sub = z_sub.assign_coords(lon=(z_sub["lon"] % 360).astype(float)).sortby("lon")

        # Build DataFrame
        bathy_df = (
            z_sub
            .to_dataframe(name="bathy")
            .reset_index()[["lon", "lat", "bathy"]]
            .dropna()
        )

    if bathy_df.empty:
        raise ValueError(
            f"No bathymetry data found for the specified extent. "
            f"Lat: [{lat_min:.1f}, {lat_max:.1f}], Lon180: [{lon0_180:.1f}, {lon1_180:.1f}]. "
            f"Check that the GEBCO file path is correct and contains data for this region."
        )

    bathy_df["lon360"] = (bathy_df["lon"].to_numpy(dtype=float) % 360.0)
    bathy_df = bathy_df.sort_values(["lat", "lon360"]).reset_index(drop=True)

    return bathy_df


def get_ice_elevation(
    lat_min,
    lat_max,
    lon0_180,
    lon1_180,
    ice_elevation_path=None,
    sub_ice_path=None,
    max_pixels=1200,
):
    """
    Get ice surface elevation where ice exists (ice_elevation > sub_ice_elevation).
    
    Returns ice elevation values only where ice thickness > 0, for visualization as a 
    colored overlay showing the ice surface.

    Parameters
    ----------
    lat_min : float
        Minimum latitude in degrees (-90 to 90).
    lat_max : float
        Maximum latitude in degrees (-90 to 90).
    lon0_180 : float
        Minimum longitude in degrees (-180 to 180).
    lon1_180 : float
        Maximum longitude in degrees (-180 to 180).
    ice_elevation_path : str, optional
        Path to GEBCO ice elevation NetCDF. If None, uses default GEBCO 2025.
    sub_ice_path : str, optional
        Path to GEBCO sub-ice elevation NetCDF. If None, uses default GEBCO 2025 sub-ice.
    max_pixels : int, default 1200
        Maximum dimension for coarsening data.

    Returns
    -------
    ice_df : pd.DataFrame
        DataFrame with columns: ['lon360', 'lat', 'ice_elevation'] where ice exists.
        Returns empty DataFrame if no ice found in extent.
    """
    import numpy as np
    import pandas as pd

    # Load both datasets using get_bathy
    if ice_elevation_path is None:
        ice_elevation_path = "/Volumes/WORK-SSD/Datasets/Published/GEBCO/gebco_2025/GEBCO_2025.nc"
    
    if sub_ice_path is None:
        sub_ice_path = "/Volumes/WORK-SSD/Datasets/Published/GEBCO/gebco_2025_sub_ice_topo/GEBCO_2025_sub_ice.nc"

    print("Loading ice elevation data...")
    ice_elev_df = get_bathy(lat_min, lat_max, lon0_180, lon1_180, 
                            gebco_nc_path=ice_elevation_path, max_pixels=max_pixels)
    ice_elev_df = ice_elev_df.rename(columns={"bathy": "ice_elevation"})

    print("Loading sub-ice elevation data...")
    sub_ice_df = get_bathy(lat_min, lat_max, lon0_180, lon1_180,
                           gebco_nc_path=sub_ice_path, max_pixels=max_pixels)
    sub_ice_df = sub_ice_df.rename(columns={"bathy": "sub_ice_elevation"})

    # Merge on coordinates
    merged = pd.merge(
        ice_elev_df[["lon360", "lat", "ice_elevation"]],
        sub_ice_df[["lon360", "lat", "sub_ice_elevation"]],
        on=["lon360", "lat"],
        how="inner"
    )

    # Calculate ice thickness to identify where ice exists
    merged["ice_thickness"] = merged["ice_elevation"] - merged["sub_ice_elevation"]

    # Keep only where ice exists (thickness > 0), but return ice_elevation for plotting
    ice_df = merged[merged["ice_thickness"] > 0][["lon360", "lat", "ice_elevation"]].copy()

    if not ice_df.empty:
        print(f"Found {len(ice_df)} grid cells with ice")
        print(f"Ice thickness range: {merged[merged['ice_thickness'] > 0]['ice_thickness'].min():.1f} to {merged[merged['ice_thickness'] > 0]['ice_thickness'].max():.1f} m")
        print(f"Ice elevation range: {ice_df['ice_elevation'].min():.1f} to {ice_df['ice_elevation'].max():.1f} m")
    else:
        print("No ice found in this extent")

    return ice_df


def plot_track_with_bathy(
    lat,
    lon,
    toppid=None,
    global_extent=False,
    pad_deg=20.0,
    bathy_df=None,
    ice_df=None,
    gebco_nc_path=None,
    output_dir=None,
    output_filename="track_bathy_map",
    max_pixels=1200,
    show_ice=True,
):
    """
    Plot animal track overlaid on GEBCO bathymetry with optional ice thickness overlay.

    Parameters
    ----------
    lat : array-like
        Latitude values (in degrees, -90 to 90).
    lon : array-like
        Longitude values (in degrees, any format; will be converted to 0-360).
    toppid : array-like, optional
        TOPPID identifiers for grouping tracks. If None, treats all as one track.
    global_extent : bool, default False
        If True, plot full global extent (lat: -90 to 90, lon: 0 to 360).
        If False, crop to track bounds with padding.
    pad_deg : float, default 20.0
        Degrees of padding around track bounds when global_extent=False.
    bathy_df : pd.DataFrame, optional
        Pre-loaded bathymetry DataFrame with columns ['lon360', 'lat', 'bathy'].
        If None, will call get_bathy() to load data.
    ice_df : pd.DataFrame, optional
        Pre-loaded ice elevation DataFrame with columns ['lon360', 'lat', 'ice_elevation'].
        If None and show_ice=True, will call get_ice_elevation() to load data.
    gebco_nc_path : str, optional
        Path to GEBCO NetCDF file for bathymetry. If None, uses default GEBCO 2025.
    output_dir : str or Path, optional
        Directory to save output figure. If None, doesn't save.
    output_filename : str, default "track_bathy_map"
        Base filename for saved figure (without extension).
    max_pixels : int, default 1200
        Maximum dimension for coarsening bathymetry data.
    show_ice : bool, default True
        If True, overlay ice elevation data (blue-white colorscale) where ice exists.

    Returns
    -------
    fig : plotnine.ggplot
        The generated figure object.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from plotnine import (
        ggplot, aes, geom_raster, geom_point,
        scale_fill_gradientn, theme_bw, labs, ggtitle,
        scale_x_continuous, scale_y_continuous, coord_fixed, theme
    )

    # --- Helper functions ---
    def _to_lon360(lon_vals):
        lon_vals = np.asarray(lon_vals, dtype=float)
        return lon_vals % 360.0

    def lon360_to_lon180(x):
        x = np.asarray(x, dtype=float)
        return ((x + 180) % 360) - 180

    # --- Prepare track data ---
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lon360 = _to_lon360(lon)

    if toppid is not None:
        toppid = np.asarray(toppid, dtype=str)
    else:
        toppid = np.array(["ALL"] * len(lat))

    track_df = pd.DataFrame({
        "Lat": lat,
        "Lon360": lon360,
        "TOPPID": toppid
    }).dropna(subset=["Lat", "Lon360"]).reset_index(drop=True)

    # Downsample if too many points
    if len(track_df) > 8000:
        stride = max(1, len(track_df) // 8000)
        track_df = track_df.iloc[::stride].reset_index(drop=True)

    # --- Determine extent ---
    if global_extent:
        LAT_MIN, LAT_MAX = -90.0, 90.0
        lon0_180_exp, lon1_180_exp = -180.0, 180.0
    else:
        lon0_360 = float(track_df["Lon360"].min())
        lon1_360 = float(track_df["Lon360"].max())
        lat0 = float(track_df["Lat"].min())
        lat1 = float(track_df["Lat"].max())

        LAT_MIN = max(-90.0, lat0 - pad_deg)
        LAT_MAX = min(90.0, lat1 + pad_deg)
        lon0_360_exp = (lon0_360 - pad_deg) % 360.0
        lon1_360_exp = (lon1_360 + pad_deg) % 360.0

        # Convert to lon180 for GEBCO indexing
        lon0_180_exp = float(lon360_to_lon180(lon0_360_exp))
        lon1_180_exp = float(lon360_to_lon180(lon1_360_exp))

    print(f"Extent: global={global_extent} | Lat: [{LAT_MIN:.1f}, {LAT_MAX:.1f}] | Lon180: [{lon0_180_exp:.1f}, {lon1_180_exp:.1f}]")

    # --- Load bathymetry if not provided ---
    if bathy_df is None:
        bathy_df = get_bathy(
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
            lon0_180=lon0_180_exp,
            lon1_180=lon1_180_exp,
            gebco_nc_path=gebco_nc_path,
            max_pixels=max_pixels
        )
    
    bathy_wrapped = bathy_df.copy()

    # --- Load ice thickness if requested and not provided ---
    if show_ice and ice_df is None:
        try:
            ice_df = get_ice_elevation(
                lat_min=LAT_MIN,
                lat_max=LAT_MAX,
                lon0_180=lon0_180_exp,
                lon1_180=lon1_180_exp,
                max_pixels=max_pixels
            )
        except Exception as e:
            print(f"Warning: Could not load ice elevation data: {e}")
            ice_df = None

    # --- Build color scale ---
    eps = 50  # meters around 0 that will be white

    bmin = float(np.nanmin(bathy_wrapped["bathy"]))
    bmax = float(np.nanmax(bathy_wrapped["bathy"]))

    def norm(v):
        return (v - bmin) / (bmax - bmin)

    fill_values = [
        norm(bmin),     # deepest
        norm(-eps),     # start white
        norm(0),        # exact sea level
        norm(eps),      # end white
        norm(500),      # low elevation stretch
        norm(bmax)      # shallowest / land
    ]

    fill_colors = [
        "#01665e",   # deep water
        "#c7eae5",   # shallow water
        "#FFFFFF",   # white at 0
        "#999999",   # grey coastline stretch
        "#636363",   # grey coastline stretch
        "#FCFCFC"    # land / ice
    ]

    # --- Determine plot bounds ---
    if global_extent:
        XMIN, XMAX = 0.0, 360.0
    else:
        # For cropped extent, use actual data bounds to avoid white space
        XMIN = float(min(bathy_wrapped["lon360"].min(), track_df["Lon360"].min()))
        XMAX = float(max(bathy_wrapped["lon360"].max(), track_df["Lon360"].max()))
    
    pad = 0.0

    ymin = float(min(bathy_wrapped["lat"].min(), track_df["Lat"].min()) - pad)
    ymax = float(max(bathy_wrapped["lat"].max(), track_df["Lat"].max()) + pad)

    # Aspect ratio correction
    mean_lat = 0.5 * (ymin + ymax)
    ratio = 1.0 / np.cos(np.deg2rad(mean_lat))

    # Create track grouping and assign categorical colors
    track_df["track_group"] = track_df["TOPPID"].astype(str)
    unique_tracks = track_df["track_group"].unique()
    n_tracks = len(unique_tracks)
    
    # Use categorical colormap
    from plotnine import scale_color_manual
    import matplotlib.cm as cm
    
    # Choose colormap based on number of tracks
    if n_tracks <= 10:
        cmap = cm.get_cmap("tab10")
    elif n_tracks <= 20:
        cmap = cm.get_cmap("tab20")
    else:
        cmap = cm.get_cmap("hsv")
    
    # Generate colors for each track
    track_colors = {}
    for i, track_id in enumerate(unique_tracks):
        if n_tracks <= 20:
            color_idx = i
        else:
            color_idx = i / n_tracks
        rgba = cmap(color_idx)
        track_colors[track_id] = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"

    # --- Build plot ---
    fig = ggplot()
    
    # Base bathymetry layer
    fig = fig + geom_raster(bathy_wrapped, aes(x="lon360", y="lat", fill="bathy"))
    
    # Ice elevation overlay (if available) - show ice with gradient
    if show_ice and ice_df is not None and not ice_df.empty:
        print(f"Adding ice elevation overlay ({len(ice_df)} points)")
        # Create ice visualization with gradient based on elevation
        # Map ice elevation to alpha for visualization
        ice_df_plot = ice_df.copy()
        ice_min = float(ice_df_plot["ice_elevation"].min())
        ice_max = float(ice_df_plot["ice_elevation"].max())
        
        # Normalize ice elevation for alpha mapping (higher ice = more opaque)
        if ice_max > ice_min:
            ice_df_plot["ice_alpha"] = ((ice_df_plot["ice_elevation"] - ice_min) / (ice_max - ice_min) * 0.6) + 0.3
        else:
            ice_df_plot["ice_alpha"] = 0.7
        
        # Plot ice with gradient - use light blue-white gradient
        fig = fig + geom_raster(
            ice_df_plot,
            aes(x="lon360", y="lat", alpha="ice_alpha"),
            fill="#D0E8F5",  # Light blue for ice
            show_legend=False  # Don't show ice in legend
        )
    
    # Track points
    fig = fig + geom_point(
        track_df,
        aes(x="Lon360", y="Lat", color="track_group"),
        alpha=0.8,
        size=0.2,
    )
    
    # Scales
    fig = (
        fig
        + scale_fill_gradientn(
            colors=fill_colors,
            values=fill_values,
            limits=(bmin, bmax),
            name="Depth (m)"
        )
        + scale_color_manual(
            values=track_colors,
            name="Track/Deployment"
        )
    )
    
    fig = (
        fig
        + scale_x_continuous(limits=(XMIN, XMAX), expand=(0, 0))
        + scale_y_continuous(limits=(ymin, ymax), expand=(0, 0))
        + coord_fixed(ratio=ratio)
        + theme_bw()
        + theme(
            figure_size=(8, 6.5),
            dpi=300,
            legend_position="right",
            legend_box="vertical"
        )
        + ggtitle("Track over GEBCO 2025 Bathymetry" + (" + Ice" if show_ice and ice_df is not None and not ice_df.empty else ""))
        + labs(x="Longitude (0–360)", y="Latitude")
    )

    # --- Save output ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / f"{output_filename}.png"
        svg_path = output_dir / f"{output_filename}.svg"

        fig.save(png_path, dpi=300, verbose=False)
        fig.save(svg_path, dpi=300, verbose=False)

        print(f"Saved: {png_path}")
        print(f"Saved: {svg_path}")

    return fig
