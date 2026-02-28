import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureWidgetResampler, FigureResampler, register_plotly_resampler
from pyologger.process_data.sampling import *
from datetime import timedelta, datetime
import json
import os
import pandas as pd
import numpy as np
import pytz
import html


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
    out = {
        "signal_label": signal,
        "signal_unit": "",
        "signal_description": "",
        "channels": {},
    }

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
):
    desc = _safe_text(channel_description)
    safe_signal = html.escape(_safe_text(signal_name))
    safe_channel = html.escape(_safe_text(channel_name))
    safe_signal_label = html.escape(_safe_text(signal_label))
    safe_suffix = html.escape(_safe_text(channel_label_suffix))
    lines = [
        f"<b>{safe_channel}</b>; <i>{safe_signal}</i>; %{{y}}",
        "<b>%{x|%H:%M:%S.%L}</b>",
        "<i>%{x|%Y-%m-%d}</i>",
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


def _resolve_state_event_end(event_row):
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

    return start_ts, None


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


def plot_tag_data_interactive(data_pkl, signals=None, channels=None, 
                               time_range=None, note_annotations=None, state_annotations=None, color_mapping_path=None, 
                               target_sampling_rate=10, zoom_start_time=None, zoom_end_time=None, 
                               plot_event_values=None, zoom_range_selector_channel=None,
                               include_blank_row=True, preserve_signal_order=False,
                               signal_metadata=None, color_mapping=None,
                               persist_color_mapping=False):
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

    # Add subplots: One row per signal, plus extra row for the blank plot and event values if needed
    extra_rows = len(plot_event_values) if plot_event_values else 0
    total_rows = len(signals_sorted) + extra_rows + (1 if include_blank_row else 0)
    fig = FigureResampler(
        make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03)
        # Specify the subplot rows that will be used for the overview axis of each column
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
        print(f"Original sampling frequency for {signal} calculated as {original_fs} Hz.")

        # Downsample the data if needed
        signal_data_filtered = downsample(signal_data_filtered, original_fs, target_sampling_rate)

        for channel in signal_channels:
            if channel in signal_data_filtered.columns:
                # Use plain Python lists to avoid narwhals/duckdb inspection paths
                # that can fail in some environments with partially initialized duckdb.
                x_data = signal_data_filtered['datetime'].tolist()
                y_data = signal_data_filtered[channel].tolist()

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

                hovertemplate = _build_trace_hovertemplate(
                    signal,
                    channel,
                    display_meta.get("signal_label", signal),
                    label_suffix,
                    ch_desc,
                )
                fig.add_trace(
                    go.Scattergl(
                        name=y_label,
                        mode='lines',
                        line=dict(color=color),
                        hovertemplate=hovertemplate,
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

                    # Mark the annotation as plotted to avoid duplicate legends
                    plotted_annotations.add(legend_key)

            # **Plot State Events (Rectangles for Continuous Events)**
            if state_annotations:
                if state_annotations:
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

                        state_by_key = data_pkl.event_data[data_pkl.event_data["key"] == event_type]
                        state_events = _filter_df_by_time(state_by_key, start_time, end_time)

                        if state_events.empty and event_type not in missing_state_warnings:
                            print(f"⚠ No '{event_type}' state events found for plotting.")
                            missing_state_warnings.add(event_type)
                            continue

                        for event_params in configs:
                            # only draw shapes on the currently plotted signal
                            if signal != event_params.get("signal"):
                                continue

                            # y-range for THIS signal only
                            y_min = signal_data.iloc[:, 1:].min().min()
                            y_max = signal_data.iloc[:, 1:].max().max()
                            shade_mode = str(event_params.get("shade_mode", "all_y"))
                            if shade_mode == "trace_to_zero":
                                y0_draw = min(0, y_min)
                                y1_draw = max(0, y_max)
                            elif shade_mode == "percent_band":
                                try:
                                    p0 = float(event_params.get("shade_pct_min", 0.0))
                                except Exception:
                                    p0 = 0.0
                                try:
                                    p1 = float(event_params.get("shade_pct_max", 100.0))
                                except Exception:
                                    p1 = 100.0
                                p0 = max(0.0, min(100.0, p0))
                                p1 = max(0.0, min(100.0, p1))
                                if p0 > p1:
                                    p0, p1 = p1, p0
                                span = float(y_max - y_min) if pd.notna(y_max) and pd.notna(y_min) else 0.0
                                y0_draw = float(y_min) + (span * (p0 / 100.0))
                                y1_draw = float(y_min) + (span * (p1 / 100.0))
                            else:
                                y0_draw = y_min
                                y1_draw = y_max

                            for _, event in state_events.iterrows():
                                start_time, end_time = _resolve_state_event_end(event)
                                if pd.isna(start_time) or end_time is None or pd.isna(end_time) or end_time <= start_time:
                                    continue

                                fig.add_shape(
                                    type="rect",
                                    x0=start_time,
                                    x1=end_time,
                                    y0=y0_draw,
                                    y1=y1_draw,
                                    fillcolor=event_params.get("color", "rgba(150, 150, 150, 0.3)"),
                                    line=dict(width=0),
                                    row=signal_plot_row,
                                    col=1,
                                    layer="below",
                                )

        # Update y-axis label for each subplot
        axis_title = signal_row_meta.get(row_counter, {}).get("title", signal)
        if include_blank_row and row_counter == 2:
            # Align the title of the blank plot (row 2) with the first plot (row 1)
            fig.update_yaxes(title_text="", row=1, col=1)
        else:
            # Keep the title where it is for the other rows
            fig.update_yaxes(title_text="", row=row_counter, col=1)
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

    # Configure shared x-axis and rangeslider at the bottom
    fig.update_layout(
        title='Tag Data Visualization',
        xaxis=dict(
            rangeselector=dict(
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
                visible=True,
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
        height=600 + 50 * (len(signals_sorted) + extra_rows),
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
                               preserve_signal_order=False):
    """
    Function to plot tag data interactively using Plotly with optional initial zooming into a specific time range.
    """
    # Default signal order
    default_order = ['ecg', 'pressure', 'accelerometer', 'magnetometer', 'gyroscope', 
                     'prh', 'temperature', 'light']

    # Load the color mapping
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

    # Add subplots: One row per signal, plus extra row for the blank plot and event values if needed
    extra_rows = len(plot_event_values) if plot_event_values else 0
    total_rows = len(signals_sorted) + extra_rows + 1  # +1 for the blank plot
    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    row_counter = 1

    def plot_signal_data(signal, signal_data, signal_info):
        """General function to handle plotting data."""
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

        # Drop invalid timestamps before sampling/plotting.
        signal_data_filtered = signal_data_filtered.dropna(subset=["datetime"])
        signal_data_filtered = signal_data_filtered.sort_values('datetime').reset_index(drop=True)

        # Calculate original sampling rate
        original_fs = calculate_sampling_frequency(signal_data_filtered['datetime'])
        print(f"Original sampling frequency for {signal} calculated as {original_fs} Hz.")

        # Downsample the data if needed
        signal_data_filtered = downsample(signal_data_filtered, original_fs, target_sampling_rate)

        for channel in signal_channels:
            if channel in signal_data_filtered.columns:
                # Use plain Python lists to avoid narwhals/duckdb inspection paths
                # that can fail in some environments with partially initialized duckdb.
                x_data = signal_data_filtered['datetime'].tolist()
                y_data = signal_data_filtered[channel].tolist()

                # Set labels and line properties
                unit = signal_info['metadata'].get(channel, {}).get('unit', '')
                y_label = f"{channel} [{unit}]" if unit else f"{channel} [unk]"
                color = color_mapping.get(channel, generate_random_color())
                color_mapping[channel] = color

                fig.add_trace(
                    go.Scatter(
                        name=y_label,
                        mode='lines',
                        line=dict(color=color),
                        x=x_data,
                        y=y_data
                    ),
                    row=row_counter,
                    col=1
                )

    # Iterate through signal data and plot relevant timeseries and events
    for signal in signals_sorted:
        if signal in data_pkl.signal_data:
            signal_data = data_pkl.signal_data[signal]
            signal_info = data_pkl.signal_info[signal]

            plot_signal_data(signal, signal_data, signal_info)
            # Reverse y-axis for depth or pressure signals
            if signal in ['pressure', 'depth', 'corrected_depth']:
                fig.update_yaxes(autorange="reversed", row=row_counter, col=1)

            if row_counter == 1:  # Right after the first plot
                # Add blank plot with height of 200 pixels after the first plot
                fig.add_trace(go.Scatter(x=[], y=[], mode='markers', showlegend=False), row=row_counter+1, col=1)
                fig.update_yaxes(showticklabels=False, row=row_counter+1, col=1)  # Hide tick labels
                fig.update_xaxes(showticklabels=False, row=row_counter+1, col=1)  # Hide tick labels
                row_counter += 1  # Skip to the next row after the blank plot

        # Plot note annotations if available
        if note_annotations:
            plotted_annotations = set()

            # Ensure time_range is valid
            if time_range and len(time_range) == 2:
                start_time, end_time = time_range
            else:
                start_time, end_time = data_pkl.event_data["datetime"].min(), data_pkl.event_data["datetime"].max()

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

                    # Mark the annotation as plotted to avoid duplicate legends
                    plotted_annotations.add(legend_key)

            # **Plot State Events (Rectangles for Continuous Events)**
            if state_annotations:
                for event_type, event_cfg in state_annotations.items():
                    # allow either a single dict or a list of dicts
                    if isinstance(event_cfg, list):
                        configs = event_cfg
                    else:
                        configs = [event_cfg]

                    # same events for this event_type, independent of signal
                    state_by_key = data_pkl.event_data[data_pkl.event_data["key"] == event_type]
                    if time_range and len(time_range) == 2:
                        state_events = _filter_df_by_time(state_by_key, time_range[0], time_range[1])
                    else:
                        state_events = state_by_key.copy()
                        if "datetime" in state_events.columns:
                            state_events["datetime"] = _coerce_datetime_series(state_events["datetime"])

                    for event_params in configs:
                        # only draw shapes on the currently plotted signal
                        if signal != event_params.get("signal"):
                            continue

                        # Row index for this signal in the subplot grid
                        signal_row = signals_sorted.index(signal)

                        # y-range for THIS signal only
                        y_min = signal_data.iloc[:, 1:].min().min()
                        y_max = signal_data.iloc[:, 1:].max().max()
                        shade_mode = str(event_params.get("shade_mode", "all_y"))
                        if shade_mode == "trace_to_zero":
                            y0_draw = min(0, y_min)
                            y1_draw = max(0, y_max)
                        elif shade_mode == "percent_band":
                            try:
                                p0 = float(event_params.get("shade_pct_min", 0.0))
                            except Exception:
                                p0 = 0.0
                            try:
                                p1 = float(event_params.get("shade_pct_max", 100.0))
                            except Exception:
                                p1 = 100.0
                            p0 = max(0.0, min(100.0, p0))
                            p1 = max(0.0, min(100.0, p1))
                            if p0 > p1:
                                p0, p1 = p1, p0
                            span = float(y_max - y_min) if pd.notna(y_max) and pd.notna(y_min) else 0.0
                            y0_draw = float(y_min) + (span * (p0 / 100.0))
                            y1_draw = float(y_min) + (span * (p1 / 100.0))
                        else:
                            y0_draw = y_min
                            y1_draw = y_max

                        for _, event in state_events.iterrows():
                            start_time, end_time = _resolve_state_event_end(event)
                            if pd.isna(start_time) or end_time is None or pd.isna(end_time) or end_time <= start_time:
                                continue

                            fig.add_shape(
                                type="rect",
                                x0=start_time,
                                x1=end_time,
                                y0=y0_draw,
                                y1=y1_draw,
                                fillcolor=event_params.get("color", "rgba(150, 150, 150, 0.3)"),
                                line=dict(width=0),
                                row=signal_row + 2,  # keeping your original offset
                                col=1,
                                layer="below",
                            )

        # Update y-axis label for each subplot
        if row_counter == 2:
            # Align the title of the blank plot (row 2) with the first plot (row 1)
            fig.update_yaxes(title_text=signal, row=1, col=1)
        else:
            # Keep the title where it is for the other rows
            fig.update_yaxes(title_text=signal, row=row_counter, col=1)
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

    # Configure shared x-axis and rangeslider at the bottom
    fig.update_layout(
        title='Tag Data Visualization',
        xaxis=dict(
            rangeselector=dict(
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
                visible=True,
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
        margin=dict(l=80, r=20, t=72, b=90),
        height=600 + 50 * (len(signals_sorted) + extra_rows),
        showlegend=True
    )

    return fig


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
    import xarray as xr

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

    ds = xr.open_dataset(gebco_nc_path, chunks="auto", cache=False)

    # Get elevation variable
    z_name = "elevation" if "elevation" in ds.data_vars else list(ds.data_vars)[0]
    z = ds[z_name]

    lon_name = "lon" if "lon" in z.coords else ("x" if "x" in z.coords else None)
    lat_name = "lat" if "lat" in z.coords else ("y" if "y" in z.coords else None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Could not find lon/lat coords. coords={list(z.coords)}")

    z = z.rename({lon_name: "lon", lat_name: "lat"})

    # Ensure lat slice direction matches dataset ordering
    lat_vals = z["lat"].values
    lat_ascending = bool(np.all(np.diff(lat_vals[:min(1000, lat_vals.size)]) > 0))
    lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)

    # Subset with dateline handling
    lon_slices = _wrap_lon180_edges(lon0_180, lon1_180)

    parts = []
    for lo, hi in lon_slices:
        parts.append(z.sel(lon=slice(lo, hi), lat=lat_slice))

    z_sub = xr.concat(parts, dim="lon") if len(parts) > 1 else parts[0]

    # Coarsen
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
