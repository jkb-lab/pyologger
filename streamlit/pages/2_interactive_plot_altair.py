import os
import pickle
from datetime import timedelta
from itertools import cycle

import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px

from pyologger.process_data.sampling import calculate_sampling_frequency
from pyologger.process_data.sampling import downsample
from pyologger.utils.folder_manager import (
    load_configuration,
    select_and_load_deployment_streamlit,
)
from pyologger.utils.streamlit_time_window import standardize_time_settings

alt.data_transformers.disable_max_rows()


def infer_signal_for_event(event_key: str, available_signals: list[str]) -> str | None:
    """Heuristic mapping from event key to the most relevant signal."""
    if not available_signals:
        return None

    key = (event_key or "").lower()
    keyword_map = (
        ("dive", ["depth", "pressure"]),
        ("stroke", ["stroke_rate", "prh"]),
        ("heart", ["ecg", "heart_rate"]),
        ("beat", ["ecg", "heart_rate"]),
        ("breath", ["o2_pressure", "pressure"]),
        ("logger", ["logger_status"]),
    )

    for keyword, candidates in keyword_map:
        if keyword in key:
            for candidate in candidates:
                if candidate in available_signals:
                    return candidate

    return available_signals[0]


def build_event_annotations(data_pkl, available_signals):
    """Create note/state annotation dictionaries for every event in event_data."""
    if not hasattr(data_pkl, "event_data") or data_pkl.event_data is None:
        return {}, {}

    event_df = data_pkl.event_data.copy()
    if event_df.empty or "datetime" not in event_df.columns or "key" not in event_df.columns:
        return {}, {}

    palette = cycle(px.colors.qualitative.Light24)
    note_annotations = {}
    state_annotations = {}

    for event_key, group in event_df.groupby("key"):
        target_signal = infer_signal_for_event(event_key, available_signals)
        if not target_signal:
            continue

        color = next(palette)
        event_type_series = group.get("type")
        duration_series = group.get("duration")
        is_state = False
        if event_type_series is not None:
            is_state = (event_type_series.astype(str).str.lower() != "point").any()
        if duration_series is not None:
            is_state = is_state or (duration_series.fillna(0) > 0).any()

        if is_state:
            state_annotations[event_key] = {
                "signal": target_signal,
                "color": color,
            }
        else:
            note_annotations[event_key] = {
                "signal": target_signal,
                "symbol": "circle",
                "color": color,
            }

    return note_annotations, state_annotations


def _filter_signal_df(signal_df, time_range):
    if not time_range:
        return signal_df
    start_time, end_time = time_range
    out = signal_df.copy()
    dt = pd.to_datetime(out["datetime"], errors="coerce")
    if start_time.tzinfo is not None:
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(start_time.tzinfo)
        else:
            dt = dt.dt.tz_convert(start_time.tzinfo)
    out["datetime"] = dt
    return out[(out["datetime"] >= start_time) & (out["datetime"] <= end_time)]


def _filter_event_df(event_df, event_key, time_range):
    subset = event_df[event_df["key"] == event_key].copy()
    if not time_range or subset.empty:
        return subset
    start_time, end_time = time_range
    dt = pd.to_datetime(subset["datetime"], errors="coerce")
    if start_time.tzinfo is not None:
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize(start_time.tzinfo)
        else:
            dt = dt.dt.tz_convert(start_time.tzinfo)
    subset["datetime"] = dt
    return subset[(subset["datetime"] >= start_time) & (subset["datetime"] <= end_time)]


def _altair_symbol(symbol: str) -> str:
    mapping = {
        "triangle-up": "triangle-up",
        "triangle-down": "triangle-down",
        "square": "square",
        "diamond": "diamond",
        "x": "cross",
    }
    return mapping.get(symbol, "circle")


def plot_tag_data_interactive_alt(
    data_pkl,
    signals=None,
    channels=None,
    time_range=None,
    note_annotations=None,
    state_annotations=None,
    target_sampling_rate=10,
):
    default_order = [
        "ecg",
        "pressure",
        "accelerometer",
        "magnetometer",
        "gyroscope",
        "prh",
        "temperature",
        "light",
    ]

    if signals is None:
        signals = list(data_pkl.signal_data.keys())

    if not signals:
        return None

    def sort_key(sig):
        return (
            default_order.index(sig)
            if sig in default_order
            else len(default_order) + signals.index(sig)
        )

    signals_sorted = sorted(signals, key=sort_key)

    event_df = getattr(data_pkl, "event_data", pd.DataFrame())
    charts = []

    for signal in signals_sorted:
        if signal not in data_pkl.signal_data:
            continue

        signal_df = data_pkl.signal_data[signal].copy()
        signal_info = data_pkl.signal_info.get(signal, {})
        signal_channels = (
            signal_info.get("channels", [])
            if not channels or signal not in channels
            else channels[signal]
        )

        if not signal_channels:
            continue

        signal_df = _filter_signal_df(signal_df, time_range)
        signal_df = signal_df.sort_values("datetime").reset_index(drop=True)

        original_fs = calculate_sampling_frequency(signal_df["datetime"])
        signal_df = downsample(signal_df, original_fs, target_sampling_rate)

        long_df = signal_df[["datetime"] + signal_channels].melt(
            "datetime", var_name="channel", value_name="value"
        )

        if long_df.empty:
            continue

        base_chart = (
            alt.Chart(long_df)
            .mark_line()
            .encode(
                x=alt.X("datetime:T", title="Datetime"),
                y=alt.Y("value:Q", title=signal),
                color=alt.Color("channel:N", legend=alt.Legend(title="Channel")),
            )
            .properties(height=220)
        )

        signal_layers = [base_chart]
        y_min = float(long_df["value"].min())
        y_max = float(long_df["value"].max())

        if note_annotations and not event_df.empty:
            for event_key, config in note_annotations.items():
                if config.get("signal") != signal:
                    continue
                subset = _filter_event_df(event_df, event_key, time_range)
                if subset.empty:
                    continue
                subset["y_position"] = y_max
                symbol = _altair_symbol(config.get("symbol", "circle"))
                note_chart = (
                    alt.Chart(subset)
                    .mark_point(
                        size=80,
                        filled=True,
                        shape=symbol,
                        color=config.get("color", "#666"),
                    )
                    .encode(
                        x="datetime:T",
                        y=alt.Y("y_position:Q", title=None),
                        tooltip=["key", "datetime", "short_description"],
                    )
                )
                signal_layers.append(note_chart)

        if state_annotations and not event_df.empty:
            for event_key, config in state_annotations.items():
                if config.get("signal") != signal:
                    continue
                subset = _filter_event_df(event_df, event_key, time_range)
                if subset.empty:
                    continue
                subset["start"] = subset["datetime"]
                subset["end"] = subset["datetime"] + pd.to_timedelta(
                    subset.get("duration", 0).fillna(0), unit="s"
                )
                subset["y_min"] = y_min
                subset["y_max"] = y_max
                rect_chart = (
                    alt.Chart(subset)
                    .mark_rect(opacity=0.2, color=config.get("color", "#BBB"))
                    .encode(
                        x="start:T",
                        x2="end:T",
                        y="y_min:Q",
                        y2="y_max:Q",
                        tooltip=["key", "start", "end"],
                    )
                )
                signal_layers.append(rect_chart)

        charts.append(alt.layer(*signal_layers))

    if not charts:
        return None

    return alt.vconcat(*charts).resolve_scale(y="independent")


# --- Streamlit Page --- #
config, data_dir, color_mapping_path, montage_path = load_configuration()

st.sidebar.title("Deployment Selection")
(
    animal_id,
    dataset_id,
    deployment_id,
    dataset_folder,
    deployment_folder,
    data_pkl,
    param_manager,
) = select_and_load_deployment_streamlit(data_dir)

if not dataset_id or not deployment_id:
    st.sidebar.warning("⚠ Please select a dataset and deployment.")
    st.stop()

st.sidebar.write(f"📂 Selected Deployment: {deployment_id}")

timezone = data_pkl.deployment_info.get("Time Zone", "UTC")


def _round_second(dt):
    return dt.replace(microsecond=0)


def _to_tzaware(value, tz_name):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def _to_display_naive(value):
    return _to_tzaware(value, timezone).tz_localize(None)


def _from_display_naive(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(timezone)
    return ts.tz_convert(timezone)


standardized = standardize_time_settings(
    param_manager=param_manager,
    data_pkl=data_pkl,
    tz_name=str(timezone),
    minutes=30,
)
OVERLAP_START_TIME = _round_second(standardized["overlap_start_time"])
OVERLAP_END_TIME = _round_second(standardized["overlap_end_time"])
ZOOM_WINDOW_START_TIME = _round_second(standardized["zoom_window_start_time"])
ZOOM_WINDOW_END_TIME = _round_second(standardized["zoom_window_end_time"])

st.subheader("Altair Signal Overview")

time_values = [
    _to_display_naive(OVERLAP_START_TIME) + timedelta(seconds=i)
    for i in range(
        0, int((OVERLAP_END_TIME - OVERLAP_START_TIME).total_seconds()) + 1, 1
    )
]

time_range_selection = st.select_slider(
    "Select Time Range",
    options=time_values,
    value=(_to_display_naive(ZOOM_WINDOW_START_TIME), _to_display_naive(ZOOM_WINDOW_END_TIME)),
    format_func=lambda x: x.strftime("%Y-%m-%d %H:%M:%S"),
)

selected_start_time, selected_end_time = time_range_selection
selected_start_ts = _from_display_naive(selected_start_time)
selected_end_ts = _from_display_naive(selected_end_time)

available_signals = list(data_pkl.signal_data.keys())

preferred_defaults = ["depth", "prh", "ecg", "heart_rate"]
default_signals = [sig for sig in preferred_defaults if sig in available_signals]
if not default_signals and available_signals:
    default_signals = [available_signals[0]]

saved_signal_config = param_manager.get_from_config(
    ["altair_default_signals"], section="settings"
)
stored_signals = saved_signal_config.get("altair_default_signals") or default_signals
stored_signals = [sig for sig in stored_signals if sig in available_signals]
if not stored_signals:
    stored_signals = default_signals

selected_signals = st.multiselect(
    "Signals to plot",
    options=available_signals,
    default=stored_signals,
    help="Default selection (depth, prh, ecg, heart_rate) is saved per deployment.",
)

if selected_signals and set(selected_signals) != set(stored_signals):
    param_manager.add_to_config(
        entries={"altair_default_signals": selected_signals},
        section="settings",
    )

if not selected_signals:
    st.warning("Please select at least one signal to display.")
    st.stop()

note_annotations, state_annotations = build_event_annotations(
    data_pkl, selected_signals
)

chart = plot_tag_data_interactive_alt(
    data_pkl=data_pkl,
    signals=selected_signals,
    time_range=(selected_start_ts, selected_end_ts),
    note_annotations=note_annotations,
    state_annotations=state_annotations,
    target_sampling_rate=10,
)

if chart is not None:
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No signals available to plot.")
