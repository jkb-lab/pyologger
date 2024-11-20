import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pyologger.process_data.sampling import *
from datetime import timedelta
import json
import os
import pandas as pd
import numpy as np

def load_color_mapping(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_color_mapping(mapping, path):
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=4)

def generate_random_color():
    """Generate a random pastel color in HEX format."""
    import random
    r = lambda: random.randint(100, 255)
    return f'#{r():02x}{r():02x}{r():02x}'

def plot_tag_data_interactive5(data_pkl, sensors=None, derived_data_signals=None, channels=None, 
                               time_range=None, note_annotations=None, color_mapping_path=None, 
                               target_sampling_rate=10, zoom_start_time=None, zoom_end_time=None, 
                               plot_event_values=None, zoom_range_selector_channel=None):
    """
    Function to plot tag data interactively using Plotly with optional initial zooming into a specific time range.
    Includes both sensor_data and derived_data.
    """
    
    # Default sensor and derived data order
    default_order = ['ecg', 'pressure', 'accelerometer', 'magnetometer', 'gyroscope', 
                     'prh', 'temperature', 'light']

    # Load the color mapping
    color_mapping = load_color_mapping(color_mapping_path) if color_mapping_path else {}

    # Determine the sensors and derived data to plot
    if sensors is None:
        sensors = list(data_pkl.sensor_data.keys())
        
    if derived_data_signals is None:
        derived_data_signals = list(data_pkl.derived_data.keys())

    # Combine sensors and derived data
    signals = sensors + derived_data_signals

    # Sort signals with the range selector signal on top if specified
    if zoom_range_selector_channel and zoom_range_selector_channel in signals:
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
        """General function to handle plotting both sensor and derived data."""
        # Determine the channels to plot for the current signal
        if channels is None or signal not in channels:
            signal_channels = signal_info['channels']
        else:
            signal_channels = channels[signal]

        # Filter data to the specified time range
        if time_range:
            start_time, end_time = time_range
            signal_data_filtered = signal_data[(signal_data['datetime'] >= start_time) & (signal_data['datetime'] <= end_time)]
        else:
            signal_data_filtered = signal_data

        # Calculate original sampling rate
        original_fs = round(1 / signal_data_filtered['datetime'].diff().dt.total_seconds().mean())
        print(f"Original sampling frequency for {signal} calculated as {original_fs} Hz.")

        # Downsample the data if needed
        signal_data_filtered = downsample(signal_data_filtered, original_fs, target_sampling_rate)

        for channel in signal_channels:
            if channel in signal_data_filtered.columns:
                x_data = signal_data_filtered['datetime']
                y_data = signal_data_filtered[channel]

                # Set labels and line properties
                original_name = signal_info['metadata'].get(channel, {}).get('original_name', channel)
                unit = signal_info['metadata'].get(channel, {}).get('unit', '')
                y_label = f"{original_name} [{unit}]" if unit else original_name
                color = color_mapping.get(original_name, generate_random_color())
                color_mapping[original_name] = color

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=y_label,
                    line=dict(color=color)
                ), row=row_counter, col=1)

    # Iterate through both sensor data and derived data and plot
    for signal in signals_sorted:
        if signal in data_pkl.sensor_data:
            signal_data = data_pkl.sensor_data[signal]
            signal_info = data_pkl.sensor_info[signal]

            plot_signal_data(signal, signal_data, signal_info)
            # Reverse y-axis for depth or pressure signals
            if signal in ['pressure']:
                fig.update_yaxes(autorange="reversed", row=row_counter, col=1)

            if row_counter == 1:  # Right after the first plot
                # Add blank plot with height of 200 pixels after the first plot
                fig.add_trace(go.Scatter(x=[], y=[], mode='markers', showlegend=False), row=row_counter+1, col=1)
                fig.update_yaxes(showticklabels=False, row=row_counter+1, col=1)  # Hide tick labels
                fig.update_xaxes(showticklabels=False, row=row_counter+1, col=1)  # Hide tick labels
                row_counter += 1  # Skip to the next row after the blank plot

        elif signal in data_pkl.derived_data:
            signal_data = data_pkl.derived_data[signal]
            signal_info = data_pkl.derived_info[signal]

            plot_signal_data(signal, signal_data, signal_info)
            if signal in ['depth']:
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

            for note_type, note_params in note_annotations.items():
                # Check if the annotation applies to the current signal
                if signal != note_params['signal']:
                    continue  # Skip annotations not tied to this signal

                symbol = note_params.get('symbol', 'circle')
                color = note_params.get('color', 'rgba(128, 128, 128, 0.5)')

                # Filter annotations based on the specified time range
                filtered_notes = data_pkl.event_data[
                    (data_pkl.event_data['key'] == note_type) &
                    (data_pkl.event_data['datetime'] >= time_range[0]) &
                    (data_pkl.event_data['datetime'] <= time_range[1])
                ]

                if not filtered_notes.empty:
                    # Use the first channel's data to determine y-axis values for the annotations
                    first_channel = signal_info['channels'][0]
                    if first_channel not in signal_data.columns:
                        print(f"Warning: First channel '{first_channel}' not found in data for signal '{signal}'.")
                        continue

                    y_fixed = signal_data[first_channel].max() if not signal_data[first_channel].empty else 1
                    scatter_x = filtered_notes['datetime']
                    scatter_y = [y_fixed] * len(filtered_notes)

                    # Add the marker trace for this annotation type
                    fig.add_trace(go.Scatter(
                        x=scatter_x,
                        y=scatter_y,
                        mode='markers',
                        marker=dict(symbol=symbol, color=color, size=10),
                        name=note_type,
                        opacity=0.5,
                        showlegend=(note_type not in plotted_annotations)
                    ), row=row_counter, col=1)

                    # Mark the annotation as plotted to avoid duplicate legends
                    plotted_annotations.add(note_type)

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
        height=600 + 50 * (len(signals_sorted) + extra_rows),
        showlegend=True
    )

    return fig


def plot_depth_correction(datetime_data, dec_factor, depth_data, first_derivative, repeated_corrected_depth_temp, 
                          repeated_corrected_depth_no_temp, depth_correction, dives, flat_chunks, 
                          temp_data=None, apply_temp_correction=False):
    """
    Plot the depth correction process including original, corrected depths, and temperature effects.
    
    Parameters:
    - datetime_data: Series of datetime objects corresponding to depth measurements- will use to calculate sampling rate and align indices of dives and SIs.
    - dec_factor: Decimation factor for plotting (try at least 400 for depth data at 400 Hz)
    - depth_data: Array of original depth measurements.
    - first_derivative: Array of first derivative of depth.
    - repeated_corrected_depth_temp: Array of temperature-corrected depth measurements, upsampled to match original sampling frequency of depth_data.
    - repeated_corrected_depth_no_temp: Array of depth measurements corrected without temperature adjustments, upsampled to match original sampling frequency of depth_data.
    - depth_correction: Array of depth correction values.
    - dives: DataFrame of detected dives with columns 'start', 'end', 'max', 'tmax'.
    - flat_chunks: List of tuples representing flat chunks detected as (start_index, end_index, median_depth).
    - temp_data: Array of temperature data (optional).
    - apply_temp_correction: Boolean flag to indicate if temperature correction was applied.
    
    Returns:
    - fig: Plotly figure object.
    """

    # Check the size of each array
    arrays = {
        "depth_data": depth_data[::dec_factor],
        "first_derivative": first_derivative[::dec_factor],
        "repeated_corrected_depth_temp": repeated_corrected_depth_temp[::dec_factor],
        "repeated_corrected_depth_no_temp": repeated_corrected_depth_no_temp[::dec_factor],
        "depth_correction": depth_correction[::dec_factor]
    }

    for name, array in arrays.items():
        if len(array) > 10000:
            raise ValueError(f"The downsampled array '{name}' has more than 10,000 values ({len(array)} values). Please downsample the data further or increase the dec_factor.")

    sampling_rate = 1 / datetime_data.diff().dt.total_seconds().mean()
    downsampled_datetime = datetime_data[::dec_factor]

    # Create subplots with four rows
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Original vs. Corrected Depth",
                                        "First Derivative of Depth",
                                        "Depth Correction Over Time",
                                        "Temperature Correction" if temp_data is not None else None))

    # Downsample arrays for plotting
    depth_data_downsampled = depth_data[::dec_factor]
    first_derivative_downsampled = first_derivative #downsample not needed
    depth_correction_downsampled = depth_correction #downsample not needed
    repeated_corrected_depth_temp_downsampled = repeated_corrected_depth_temp[::dec_factor]
    repeated_corrected_depth_no_temp_downsampled = repeated_corrected_depth_no_temp[::dec_factor]
   

    # Plot 1: Original Depth (reversed y-axis)
    fig.add_trace(go.Scatter(x=downsampled_datetime, y=depth_data_downsampled, 
                             mode='lines', name='Original Depth', 
                             line=dict(color='LightBlue')), row=1, col=1)

    # Add temperature-corrected depth with low opacity if the checkbox is selected
    if apply_temp_correction:
        fig.add_trace(go.Scatter(x=downsampled_datetime, y=repeated_corrected_depth_temp_downsampled, 
                                 mode='lines', name='Temp Corrected Depth', 
                                 line=dict(color='red', dash='dot')), row=1, col=1)

    # Add the no temperature-corrected depth as the main corrected depth
    fig.add_trace(go.Scatter(x=downsampled_datetime, y=repeated_corrected_depth_no_temp_downsampled, 
                             mode='lines', name='Corrected Depth (no Temp Correction)', 
                             line=dict(color='DarkBlue')), row=1, col=1)
    fig.update_yaxes(title_text="Depth (meters)", autorange="reversed", row=1, col=1)

    # Highlight detected dives in blue
    for _, row in dives.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']

        fig.add_shape(
            type="rect",
            x0=start_time, x1=end_time,
            y0=0, y1=max(depth_data),
            fillcolor="DarkBlue", opacity=0.2,
            layer="below", line_width=0,
            row=1, col=1
        )

    # Highlight flat chunks in green
    for _, row in flat_chunks.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        
        fig.add_shape(
            type="rect",
            x0=start_time, x1=end_time,
            y0=1, y1=1.5,
            fillcolor="LightBlue", opacity=0.5,
            layer="below", line_width=0,
            row=1, col=1
        )

    # Plot 2: First Derivative of Depth with Flat Chunks
    fig.add_trace(go.Scatter(x=downsampled_datetime, y=first_derivative_downsampled, 
                             mode='lines', name='First Derivative of Depth', 
                             line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=downsampled_datetime, y=[-0.1]*len(downsampled_datetime), 
                             mode='lines', name='Flat Chunk Threshold (-0.1)', 
                             line=dict(color='LightGreen', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=downsampled_datetime, y=[0.1]*len(downsampled_datetime), 
                             mode='lines', name='Flat Chunk Threshold (0.1)', 
                             line=dict(color='LightGreen', dash='dot')), row=2, col=1)
    fig.update_yaxes(title_text="Rate of Depth Change", row=2, col=1)

    # Plot 3: Depth Correction Over Time
    fig.add_trace(go.Scatter(x=downsampled_datetime, y=depth_correction_downsampled, 
                             mode='lines', name='Depth Correction', 
                             line=dict(color='DarkBlue')), row=3, col=1)
    fig.update_yaxes(title_text="Depth Correction (m)", row=3, col=1)

    if temp_data is not None:
        # Plot 4: Temperature Correction Over Time
        fig.add_trace(go.Scatter(x=downsampled_datetime, y=repeated_corrected_depth_temp_downsampled, 
                                 mode='lines', name='Temperature Correction', 
                                 line=dict(color='orange')), row=4, col=1)
        fig.update_yaxes(title_text="Temperature Correction (m)", row=4, col=1)

    # Update layout
    fig.update_layout(title="Depth Data Analysis", 
                      height=800,
                      hoversubplots="axis",  # Synchronize hover across subplots
                      hovermode="x unified") # This enables the vertical line across subplots

    return fig


