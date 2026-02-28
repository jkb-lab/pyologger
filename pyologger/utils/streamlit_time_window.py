import pandas as pd


TIME_KEYS = [
    "overlap_start_time",
    "overlap_end_time",
    "zoom_window_start_time",
    "zoom_window_end_time",
]


def _to_aware(ts_value, tz_name: str):
    ts = pd.Timestamp(ts_value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def deployment_time_span(data_pkl, tz_name: str):
    mins = []
    maxs = []
    for sig_df in getattr(data_pkl, "signal_data", {}).values():
        if not isinstance(sig_df, pd.DataFrame):
            continue
        if "datetime" not in sig_df.columns or sig_df.empty:
            continue
        dt = pd.to_datetime(sig_df["datetime"], errors="coerce").dropna()
        if dt.empty:
            continue
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(tz_name)
        else:
            dt = dt.dt.tz_convert(tz_name)
        mins.append(dt.min())
        maxs.append(dt.max())
    if not mins or not maxs:
        return None, None
    return min(mins), max(maxs)


def middle_window(start_ts, end_ts, minutes: int = 30):
    total = end_ts - start_ts
    target = pd.Timedelta(minutes=minutes)
    if total <= target:
        return start_ts, end_ts
    mid = start_ts + total / 2
    half = target / 2
    return mid - half, mid + half


def standardize_time_settings(param_manager, data_pkl, tz_name: str, minutes: int = 30, persist: bool = True):
    """
    Standard deployment window policy for Streamlit:
    - overlap = full deployment span across all signal_data datetime columns
    - zoom = centered `minutes` window (or full span if shorter)
    """
    start_ts, end_ts = deployment_time_span(data_pkl, tz_name)
    if start_ts is None or end_ts is None:
        raise ValueError("No valid datetime values found across signal_data.")

    zoom_start, zoom_end = middle_window(start_ts, end_ts, minutes=minutes)
    settings = {
        "overlap_start_time": str(start_ts),
        "overlap_end_time": str(end_ts),
        "zoom_window_start_time": str(zoom_start),
        "zoom_window_end_time": str(zoom_end),
    }
    if persist:
        param_manager.add_to_config(entries=settings, section="settings")
    return {k: _to_aware(v, tz_name) for k, v in settings.items()}
