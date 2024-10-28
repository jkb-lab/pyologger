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

def plot_tag_data_interactive3(data_pkl, sensors=None, channels=None, time_range=None, note_annotations=None, 
                              color_mapping_path=None, target_sampling_rate=10):
    """
    Function to plot tag data interactively using Plotly.

    Parameters
    ----------
    data_pkl : object
        The object containing the sensor data and metadata.
    sensors : list, optional
        List of sensors to plot. If None, plot all available sensors.
    channels : dict, optional
        Dictionary specifying the channels to plot for each sensor.
        E.g., {'ecg': ['ecg'], 'depth': ['depth']}
        If None, plot all channels for the specified sensors.
    time_range : tuple, optional
        Tuple specifying the start and end time for plotting.
    note_annotations : dict, optional
        Dictionary of annotations to plot. E.g., {'heartbeat_manual_ok': 'ecg'}
    color_mapping_path : str, optional
        Path to the JSON file containing the color mappings.
    target_sampling_rate : int, optional
        The target sampling rate to downsample the data for plotting.
    """
    
    # Load the color mapping
    color_mapping = load_color_mapping(color_mapping_path) if color_mapping_path else {}

    # Determine the sensors to plot
    if sensors is None:
        sensors = list(data_pkl.sensor_data.keys())

    # Set up the figure
    fig = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True, vertical_spacing=0.03)
    
    row_counter = 1

    for sensor in sensors:
        sensor_df = data_pkl.sensor_data[sensor]
        sensor_info = data_pkl.sensor_info[sensor]

        # Determine the channels to plot for the current sensor
        if channels is None or sensor not in channels:
            sensor_channels = sensor_info['channels']
        else:
            sensor_channels = channels[sensor]

        # Filter data to the time range
        if time_range:
            start_time, end_time = time_range
            sensor_df_filtered = sensor_df[(sensor_df['datetime'] >= start_time) & (sensor_df['datetime'] <= end_time)]
        else:
            sensor_df_filtered = sensor_df

        # Calculate original sampling rate
        original_fs = 1 / sensor_df_filtered['datetime'].diff().dt.total_seconds().mean()

        # Downsample the data
        def downsample(df, original_fs, target_fs):
            if target_fs >= original_fs:
                return df
            conversion_factor = int(original_fs / target_fs)
            return df.iloc[::conversion_factor, :]

        sensor_df_filtered = downsample(sensor_df_filtered, original_fs, target_sampling_rate)

        # Plot each channel
        for channel in sensor_channels:
            if channel in sensor_df_filtered.columns:
                x_data = sensor_df_filtered['datetime']
                y_data = sensor_df_filtered[channel]

                original_name = sensor_info['metadata'][channel]['original_name']
                unit = sensor_info['metadata'][channel]['unit']
                y_label = f"{original_name} [{unit}]"

                color = color_mapping.get(original_name, generate_random_color())
                color_mapping[original_name] = color

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=y_label,
                    line=dict(color=color)
                ), row=row_counter, col=1)

        # Handle annotations for this sensor
        if note_annotations:
            for note_type, note_channel in note_annotations.items():
                if note_channel in sensor_df_filtered.columns:
                    filtered_notes = data_pkl.event_data[data_pkl.event_data['key'] == note_type]
                    if not filtered_notes.empty:
                        for dt in filtered_notes['datetime']:
                            fig.add_trace(go.Scatter(
                                x=[dt, dt],
                                y=[sensor_df_filtered[note_channel].min(), sensor_df_filtered[note_channel].max()],
                                mode='lines',
                                line=dict(color=color_mapping.get(note_type, 'rgba(128, 128, 128, 0.5)'), width=1, dash='dot'),
                                showlegend=False
                            ), row=row_counter, col=1)

        fig.update_yaxes(title_text=sensor, row=row_counter, col=1)
        row_counter += 1

    fig.update_layout(
        height=200 * len(sensors),
        width=1200,
        hovermode="x unified",  # Enables the vertical hover line across subplots
        title_text=f"{data_pkl.deployment_info['Deployment ID']}",
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend
            xanchor='center',  # Anchor the legend horizontally at the center
            yanchor='top'   # Anchor the legend vertically at the top of the legend box
        )
    )

    fig.update_xaxes(title_text="Datetime", row=row_counter-1, col=1)

    return fig

def plot_tag_data_interactive4(data_pkl, sensors=None, derived_data_signals=None, channels=None, 
                               time_range=None, note_annotations=None, color_mapping_path=None, 
                               target_sampling_rate=10, zoom_start_time=None, zoom_end_time=None, 
                               plot_event_values=None, zoom_range_selector_channel=None):
    """
    Function to plot tag data interactively using Plotly with optional initial zooming into a specific time range.
    Includes both sensor_data and derived_data.
    
    Parameters
    ----------
    data_pkl : object
        The object containing the sensor data and metadata.
    sensors : list, optional
        List of sensors to plot. If None, plot all available sensors.
    derived_data_signals : list, optional
        List of derived data signals to plot. If None, plot all available derived data.
    channels : dict, optional
        Dictionary specifying the channels to plot for each sensor.
    time_range : tuple, optional
        Tuple specifying the start and end time for plotting.
    note_annotations : dict, optional
        Dictionary of annotations to plot.
    color_mapping_path : str, optional
        Path to the JSON file containing the color mappings.
    target_sampling_rate : int, optional
        The target sampling rate to downsample the data for plotting.
    zoom_start_time : datetime, optional
        Start time for zooming into the plot initially.
    zoom_end_time : datetime, optional
        End time for zooming into the plot initially.
    zoom_range_selector_channel : str, optional
        Specify the channel to show the range selector.
    plot_event_values : list, optional
        List of event types to plot values associated with.
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

    # If a zoom_range_selector_channel is specified, move it to the top of the plot list
    if zoom_range_selector_channel and zoom_range_selector_channel in signals:
        signals_sorted = [zoom_range_selector_channel] + [s for s in signals if s != zoom_range_selector_channel]
    else:
        signals_sorted = sorted(signals, key=lambda x: (default_order.index(x) 
                                                        if x in default_order else len(default_order) + signals.index(x)))

    # If plotting event values, add an extra subplot row
    extra_rows = len(plot_event_values) if plot_event_values else 0

    # Set up the figure with one subplot per signal, plus extra for event values
    fig = make_subplots(rows=len(signals_sorted) + extra_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    row_counter = 1

    for signal in signals_sorted:
        if signal in data_pkl.sensor_data:
            # Plot sensor data
            sensor_df = data_pkl.sensor_data[signal]
            sensor_info = data_pkl.sensor_info[signal]

            # Determine the channels to plot for the current sensor
            if channels is None or signal not in channels:
                sensor_channels = sensor_info['channels']
            else:
                sensor_channels = channels[signal]

            # Filter data to the specified time range
            if time_range:
                start_time, end_time = time_range
                sensor_df_filtered = sensor_df[(sensor_df['datetime'] >= start_time) & (sensor_df['datetime'] <= end_time)]
            else:
                sensor_df_filtered = sensor_df

            # Calculate original sampling rate
            original_fs = 1 / sensor_df_filtered['datetime'].diff().dt.total_seconds().mean()

            # Downsample the data
            def downsample(df, original_fs, target_fs):
                if target_fs >= original_fs:
                    return df
                conversion_factor = int(original_fs / target_fs)
                return df.iloc[::conversion_factor, :]

            sensor_df_filtered = downsample(sensor_df_filtered, original_fs, target_sampling_rate)

            # Plot each channel
            for channel in sensor_channels:
                if channel in sensor_df_filtered.columns:
                    x_data = sensor_df_filtered['datetime']
                    y_data = sensor_df_filtered[channel]

                    # Use the original name if available, else default to channel name
                    original_name = sensor_info['metadata'].get(channel, {}).get('original_name', channel)
                    unit = sensor_info['metadata'].get(channel, {}).get('unit', '')
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

        elif signal in data_pkl.derived_data:
            # Plot derived data
            derived_df = data_pkl.derived_data[signal]

            # Filter data to the specified time range
            if time_range:
                start_time, end_time = time_range
                derived_df_filtered = derived_df[(derived_df['datetime'] >= start_time) & (derived_df['datetime'] <= end_time)]
            else:
                derived_df_filtered = derived_df

            # Plot each channel in derived data
            for channel in derived_df_filtered.columns:
                if channel != 'datetime':  # Skip datetime column
                    x_data = derived_df_filtered['datetime']
                    y_data = derived_df_filtered[channel]

                    # Use channel name as the label
                    y_label = f"{signal} - {channel}"

                    color = color_mapping.get(y_label, generate_random_color())
                    color_mapping[y_label] = color

                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines',
                        name=y_label,
                        line=dict(color=color)
                    ), row=row_counter, col=1)

        if note_annotations:
            plotted_annotations = set()
            all_shapes = []
            y_offsets = {}

            def get_signal_data(signal, data_pkl):
                """Helper function to retrieve the data (either sensor_data or derived_data) for the given signal."""
                if signal in data_pkl.sensor_data:
                    return data_pkl.sensor_data[signal], data_pkl.sensor_info[signal]
                elif signal in data_pkl.derived_data:
                    return data_pkl.derived_data[signal], data_pkl.derived_info[signal]
                else:
                    return None, None
                
            for note_type, note_params in note_annotations.items():
                note_channel = note_params['sensor']
                
                # Get the data and info for the signal (from either sensor_data or derived_data)
                signal_data, signal_info = get_signal_data(signal, data_pkl)

                # If the signal exists and the note_channel exists in its columns, proceed
                if signal_data is not None and note_channel in signal_data.columns:
                    # Filter notes by the specified time range
                    filtered_notes = data_pkl.event_data[data_pkl.event_data['key'] == note_type]
                    filtered_notes = filtered_notes[(filtered_notes['datetime'] >= start_time) &
                                                    (filtered_notes['datetime'] <= end_time)]
                    
                    if not filtered_notes.empty:
                        symbol = note_params.get('symbol', 'circle')
                        color = note_params.get('color', 'rgba(128, 128, 128, 0.5)')
                        
                        # Calculate the fixed y-value (half of the maximum y-value)
                        y_fixed = (2/3)*data_pkl.sensor_data[signal][note_channel].max()
                        scatter_x = []
                        scatter_y = []

                        offset_threshold = timedelta(milliseconds=50)

                        for dt in filtered_notes['datetime']:
                            # Check if there's an existing datetime within 10 milliseconds
                            close_times = [existing_dt for existing_dt in y_offsets if abs(dt - existing_dt) <= offset_threshold]

                            if close_times:
                                # If there's a close datetime, apply an additional offset
                                closest_time = max(close_times)  # Get the most recent close time
                                y_current = y_offsets[closest_time] + 0.15 * data_pkl.sensor_data[signal][note_channel].max()
                            else:
                                y_current = y_fixed

                            scatter_x.append(dt)
                            scatter_y.append(y_current)

                            # Track the y-offset for this datetime to avoid overlap
                            y_offsets[dt] = y_current

                        # Add the Scatter trace for this note_type
                        showlegend = note_type not in plotted_annotations
                        plotted_annotations.add(note_type)

                        fig.add_trace(go.Scatter(
                            x=scatter_x,
                            y=scatter_y,
                            mode='markers',
                            marker=dict(symbol=symbol, color=color, size=10),
                            name=note_type,
                            opacity=0.5,
                            showlegend=showlegend
                        ), row=row_counter, col=1)
                else:
                    print(f"Warning: note_channel '{note_channel}' not found for signal '{signal}'")

        # Update y-axis title
        fig.update_yaxes(title_text=signal, row=row_counter, col=1)
        row_counter += 1

    # Handle event values
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

    # Handle initial zoom
    if zoom_start_time and zoom_end_time:
        fig.update_xaxes(range=[zoom_start_time, zoom_end_time])

    # Configure layout with shared x-axis and range slider
    fig.update_layout(
        title='Tag Data Visualization',
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            type="date"
        ),
        xaxis1=dict(
            rangeslider=dict(visible=True),
            type="date"
        ),
        height=600 + 50 * (len(signals_sorted) + extra_rows),
        showlegend=True
    )

    fig.show()

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
        original_fs = int(1 / signal_data_filtered['datetime'].diff().dt.total_seconds().mean())
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

            if row_counter == 1:  # Right after the first plot
                # Add blank plot with height of 200 pixels after the first plot
                fig.add_trace(go.Scatter(x=[], y=[], mode='markers', showlegend=False), row=row_counter+1, col=1)
                fig.update_yaxes(showticklabels=False, row=row_counter+1, col=1)  # Hide tick labels
                fig.update_xaxes(showticklabels=False, row=row_counter+1, col=1)  # Hide tick labels
                row_counter += 1  # Skip to the next row after the blank plot

        # Plot note annotations if available
        if note_annotations:
            plotted_annotations = set()
            y_offsets = {}

            for note_type, note_params in note_annotations.items():
                note_channel = note_params['sensor']
                signal_data, signal_info = (data_pkl.sensor_data.get(signal), data_pkl.sensor_info.get(signal)) if signal in data_pkl.sensor_data else (data_pkl.derived_data.get(signal), data_pkl.derived_info.get(signal))

                if signal_data is not None and note_channel in signal_data.columns:
                    filtered_notes = data_pkl.event_data[(data_pkl.event_data['key'] == note_type) & (data_pkl.event_data['datetime'] >= time_range[0]) & (data_pkl.event_data['datetime'] <= time_range[1])]

                    if not filtered_notes.empty:
                        symbol = note_params.get('symbol', 'circle')
                        color = note_params.get('color', 'rgba(128, 128, 128, 0.5)')
                        y_fixed = (2 / 3) * signal_data[note_channel].max()

                        scatter_x, scatter_y = [], []
                        for dt in filtered_notes['datetime']:
                            y_current = y_offsets.get(dt, y_fixed)
                            scatter_x.append(dt)
                            scatter_y.append(y_current)
                            y_offsets[dt] = y_current + 0.15 * y_fixed

                        fig.add_trace(go.Scatter(
                            x=scatter_x,
                            y=scatter_y,
                            mode='markers',
                            marker=dict(symbol=symbol, color=color, size=10),
                            name=note_type,
                            opacity=0.5,
                            showlegend=(note_type not in plotted_annotations)
                        ), row=row_counter, col=1)
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

    fig.show()

   
def plot_tag_data_interactive2(data_pkl, imu_channels, ephys_channels=None, imu_logger=None, 
                              ephys_logger=None, imu_sampling_rate=10, ephys_sampling_rate=50, 
                              time_range=None, note_annotations=None, color_mapping_path=None):
    """
    Function to plot tag data interactively using Plotly.

    Parameters
    ----------
    data_pkl : object
        The pickle object containing the data.
    imu_channels : list
        List of IMU channels to plot.
    ephys_channels : list, optional
        List of electrophysiology channels to plot.
    imu_logger : str, optional
        IMU logger identifier.
    ephys_logger : str, optional
        Electrophysiology logger identifier.
    imu_sampling_rate : int, optional
        Sampling rate for IMU data.
    ephys_sampling_rate : int, optional
        Sampling rate for electrophysiology data.
    time_range : tuple, optional
        Tuple specifying the start and end time for plotting.
    note_annotations : dict, optional
        Dictionary of annotations to plot.
    color_mapping_path : str, optional
        Path to the JSON file containing the color mappings.
    """

    # Load the color mapping
    color_mapping = load_color_mapping(color_mapping_path) if color_mapping_path else {}

    # Define and order channels
    ordered_channels = []
    if ephys_channels:
        if 'ecg' in [ch.lower() for ch in ephys_channels]:
            ordered_channels.append(('ECG', 'ecg'))
    if 'depth' in [ch.lower() for ch in imu_channels]:
        ordered_channels.append(('Depth', 'depth'))
    if 'corrdepth' in imu_channels:
        ordered_channels.append(('Corrected Depth', 'corrdepth'))

    if any(ch.lower() in ['accx', 'accy', 'accz'] for ch in imu_channels):
        ordered_channels.append(('Accelerometer', ['accX', 'accY', 'accZ']))
    if any(ch.lower() in ['accx_adjusted', 'accy_adjusted', 'accz_adjusted'] for ch in imu_channels):
        ordered_channels.append(('Accel_Adjusted', ['accX', 'accY', 'accZ']))
    if any(ch.lower() in ['corr_accx', 'corr_accy', 'corr_accz'] for ch in imu_channels):
        ordered_channels.append(('Accel_Corrected', ['corr_accX', 'corr_accY', 'corr_accZ']))
        
    if any(ch.lower() in ['gyrx', 'gyry', 'gyrz'] for ch in imu_channels):
        ordered_channels.append(('Gyroscope', ['gyrX', 'gyrY', 'gyrZ']))
    if any(ch.lower() in ['gyrx_adjusted', 'gyry_adjusted', 'gyrz_adjusted'] for ch in imu_channels):
        ordered_channels.append(('Gyro_Adjusted', ['gyrX', 'gyrY', 'gyrZ']))
    if any(ch.lower() in ['corr_gyrx', 'corr_gyry', 'corr_gyrz'] for ch in imu_channels):
        ordered_channels.append(('Gyro_Corrected', ['corr_gyrX', 'corr_gyrY', 'corr_gyrZ']))

    if any(ch.lower() in ['magx', 'magy', 'magz'] for ch in imu_channels):
        ordered_channels.append(('Magnetometer', ['magX', 'magY', 'magZ']))
    if any(ch.lower() in ['magx_adjusted', 'magy_adjusted', 'magz_adjusted'] for ch in imu_channels):
        ordered_channels.append(('Mag_Adjusted', ['magX', 'magY', 'magZ']))
    if any(ch.lower() in ['corr_magx', 'corr_magy', 'corr_magz'] for ch in imu_channels):
        ordered_channels.append(('Mag_Corrected', ['corr_magX', 'corr_magY', 'corr_magZ']))
    
    if 'pitch' in [ch.lower() for ch in imu_channels]:
        ordered_channels.append(('Pitch', 'pitch'))
    if 'roll' in [ch.lower() for ch in imu_channels]:
        ordered_channels.append(('Roll', 'roll'))
    if 'heading' in [ch.lower() for ch in imu_channels]:
        ordered_channels.append(('Heading', 'heading'))

    #if any(ch.lower() in ['pitch', 'roll', 'heading'] for ch in imu_channels):
    #    ordered_channels.append(('Pitch Roll Heading', ['pitch', 'roll', 'heading']))

    # Add any undefined channels to the bottom of the list
    defined_channels = {ch.lower() for channel_type, channels in ordered_channels for ch in (channels if isinstance(channels, list) else [channels])}
    undefined_imu_channels = [ch for ch in imu_channels if ch.lower() not in defined_channels]
    undefined_ephys_channels = [ch for ch in ephys_channels if ch.lower() not in defined_channels]

    for ch in undefined_imu_channels:
        ordered_channels.append(('Undefined_IMU', ch))
    for ch in undefined_ephys_channels:
        ordered_channels.append(('Undefined_EPHYS', ch))

    # Get the datetime overlap between IMU and ePhys data
    imu_df = data_pkl.logger_data[imu_logger] if imu_logger else None
    ephys_df = data_pkl.logger_data[ephys_logger] if ephys_logger else None

    imu_start_time = imu_df['datetime'].min().to_pydatetime() if imu_df is not None else None
    imu_end_time = imu_df['datetime'].max().to_pydatetime() if imu_df is not None else None
    ephys_start_time = ephys_df['datetime'].min().to_pydatetime() if ephys_df is not None else None
    ephys_end_time = ephys_df['datetime'].max().to_pydatetime() if ephys_df is not None else None

    # Determine overlapping time range
    start_time = max(imu_start_time, ephys_start_time) if imu_start_time and ephys_start_time else imu_start_time or ephys_start_time
    end_time = min(imu_end_time, ephys_end_time) if imu_end_time and ephys_end_time else imu_end_time or ephys_end_time

    # Use the user-defined time range if provided
    if time_range:
        start_time = max(start_time, time_range[0])
        end_time = min(end_time, time_range[1])

    # Filter data to the overlapping time range
    def get_time_filtered_df(df, start_time, end_time):
        return df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
    
    def downsample(df, original_fs, target_fs):
        if target_fs >= original_fs:
            return df
        conversion_factor = int(original_fs / target_fs)
        print(f"Original FS: {original_fs}, Target FS: {target_fs}, Conversion Factor: {conversion_factor}")
        return df.iloc[::conversion_factor, :]

    if imu_logger:
        imu_fs = 1 / imu_df['datetime'].diff().dt.total_seconds().mean()
        imu_df_downsampled = downsample(imu_df, imu_fs, imu_sampling_rate)
        imu_df_filtered = get_time_filtered_df(imu_df_downsampled, start_time, end_time)
        imu_info = data_pkl.logger_info[imu_logger]['channelinfo']
        print(f"IMU was downsampled from {imu_fs} to {imu_sampling_rate}")

    if ephys_logger:
        ephys_fs = 1 / ephys_df['datetime'].diff().dt.total_seconds().mean()
        ephys_df_downsampled = downsample(ephys_df, ephys_fs, ephys_sampling_rate)
        ephys_df_filtered = get_time_filtered_df(ephys_df_downsampled, start_time, end_time)
        ephys_info = data_pkl.logger_info[ephys_logger]['channelinfo']

    # Prepare the data and layout structure
    data = []
    yaxis_counter = 1
    yaxis_map = {}

    for channel_type, channels in ordered_channels:
        for sub_channel in channels if isinstance(channels, list) else [channels]:
            if channel_type == 'ECG':
                df = ephys_df_filtered
                info = ephys_info
            else:
                df = imu_df_filtered
                info = imu_info

            if sub_channel in df.columns:
                y_data = df[sub_channel]
                x_data = df['datetime']

                # Handle undefined channels and assign a random color if necessary
                original_name = info[sub_channel]['original_name']
                unit = info[sub_channel]['unit']
                y_label = f"{original_name}"

                if original_name not in color_mapping:
                    color = generate_random_color()
                    color_mapping[original_name] = color
                    if color_mapping_path:
                        save_color_mapping(color_mapping, color_mapping_path)
                else:
                    color = color_mapping[original_name]

                if channel_type not in yaxis_map:
                    yaxis_map[channel_type] = f'y{yaxis_counter}'
                    yaxis_counter += 1

                yaxis = yaxis_map[channel_type]
                
                data.append(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=y_label,
                    line=dict(color=color),
                    yaxis=yaxis
                ))

    # Calculate the height based on the number of rows
    row_height = 120  # Height per row in pixels
    total_height = row_height * (yaxis_counter - 1)
    
    layout = {
        'title': f"{data_pkl.deployment_info['Deployment ID']}",
        'hovermode': 'x unified',
        'height': total_height,
        'margin': dict(l=10, r=10, t=40, b=20),
        'grid': {'rows': yaxis_counter - 1, 'columns': 1},
        'showlegend': True,
        'legend': {
            'xanchor': 'center',
        }
    }

    for i, channel_type in enumerate(yaxis_map, start=1):
        yaxis_name = f'yaxis{i}' if i > 1 else 'yaxis'
        layout[yaxis_name] = {'title': channel_type}
        if channel_type == 'Depth':
            layout[yaxis_name]['autorange'] = 'reversed'

    fig = go.Figure(data=data, layout=layout)

    if note_annotations:
        for note_type, note_channel in note_annotations.items():
            if note_channel in imu_df_filtered.columns or note_channel in ephys_df_filtered.columns:
                filtered_notes = data_pkl.event_data[data_pkl.event_data['key'] == note_type]
                for dt in filtered_notes['datetime']:
                    # Use color from color_mapping or default to 50% opaque gray
                    color = color_mapping.get(note_type, 'rgba(128, 128, 128, 0.5)')
                    fig.add_vline(
                        x=dt,
                        line=dict(color=color, width=1, dash='dot')
                    )

    return fig


def plot_tag_data_interactive(data_pkl, imu_channels, ephys_channels=None, imu_logger=None, 
                              ephys_logger=None, imu_sampling_rate=10, ephys_sampling_rate=50, 
                              time_range=None, note_annotations=None, color_mapping_path=None):
    """
    Function to plot tag data interactively using Plotly.

    Parameters
    ----------
    data_pkl : object
        The pickle object containing the data.
    imu_channels : list
        List of IMU channels to plot.
    ephys_channels : list, optional
        List of electrophysiology channels to plot.
    imu_logger : str, optional
        IMU logger identifier.
    ephys_logger : str, optional
        Electrophysiology logger identifier.
    imu_sampling_rate : int, optional
        Sampling rate for IMU data.
    ephys_sampling_rate : int, optional
        Sampling rate for electrophysiology data.
    time_range : tuple, optional
        Tuple specifying the start and end time for plotting.
    note_annotations : dict, optional
        Dictionary of annotations to plot.
    color_mapping_path : str, optional
        Path to the JSON file containing the color mappings.
    """
    
    # Load the color mapping
    color_mapping = load_color_mapping(color_mapping_path) if color_mapping_path else {}

    # Ensure the order of channels: ECG, HR, Depth, Accel, Gyro, Mag
    ordered_channels = []
    if ephys_channels:
        if 'ecg' in [ch.lower() for ch in ephys_channels]:
            ordered_channels.append(('ECG', 'ecg'))
    if 'depth' in [ch.lower() for ch in imu_channels]:
        ordered_channels.append(('Depth', 'depth'))
    if 'corrdepth' in imu_channels:
        ordered_channels.append(('Corrected Depth', 'corrdepth'))

    if any(ch.lower() in ['accx', 'accy', 'accz'] for ch in imu_channels):
        ordered_channels.append(('Accel', ['accX', 'accY', 'accZ']))
    if any(ch.lower() in ['accX_adjusted', 'accY_adjusted', 'accZ_adjusted'] for ch in imu_channels):
        ordered_channels.append(('Accel_Adjusted', ['accX', 'accY', 'accZ']))
    if any(ch.lower() in ['corr_accx', 'corr_accy', 'corr_accz'] for ch in imu_channels):
        ordered_channels.append(('Accel_Corrected', ['corr_accX', 'corr_accY', 'corr_accZ']))
        
    if any(ch.lower() in ['gyrx', 'gyry', 'gyrz'] for ch in imu_channels):
        ordered_channels.append(('Gyro', ['gyrX', 'gyrY', 'gyrZ']))
    if any(ch.lower() in ['gyrx_adjusted', 'gyry_adjusted', 'gyrz_adjusted'] for ch in imu_channels):
        ordered_channels.append(('Gyro_Adjusted', ['gyrX', 'gyrY', 'gyrZ']))
    if any(ch.lower() in ['corr_gyrx', 'corr_gyry', 'corr_gyrz'] for ch in imu_channels):
        ordered_channels.append(('Gyro_Corrected', ['corr_gyrX', 'corr_gyrY', 'corr_gyrZ']))

    if any(ch.lower() in ['magx', 'magy', 'magz'] for ch in imu_channels):
        ordered_channels.append(('Mag', ['magX', 'magY', 'magZ']))
    if any(ch.lower() in ['magx_adjusted', 'magy_adjusted', 'magz_adjusted'] for ch in imu_channels):
        ordered_channels.append(('Mag_Adjusted', ['magX', 'magY', 'magZ']))
    if any(ch.lower() in ['corr_magx', 'corr_magy', 'corr_magz'] for ch in imu_channels):
        ordered_channels.append(('Mag_Corrected', ['corr_magX', 'corr_magY', 'corr_magZ']))

    if any(ch.lower() in ['pitch', 'roll', 'heading'] for ch in imu_channels):
        ordered_channels.append(('prh', ['pitch', 'roll', 'heading']))

    # Add any undefined channels to the bottom of the list
    defined_channels = {ch.lower() for channel_type, channels in ordered_channels for ch in (channels if isinstance(channels, list) else [channels])}
    undefined_imu_channels = [ch for ch in imu_channels if ch.lower() not in defined_channels]
    undefined_ephys_channels = [ch for ch in ephys_channels if ch.lower() not in defined_channels]

    for ch in undefined_imu_channels:
        ordered_channels.append(('Undefined_IMU', ch))
    for ch in undefined_ephys_channels:
        ordered_channels.append(('Undefined_EPHYS', ch))

    # Get the datetime overlap between IMU and ePhys data
    imu_df = data_pkl.logger_data[imu_logger] if imu_logger else None
    ephys_df = data_pkl.logger_data[ephys_logger] if ephys_logger else None

    # Determine overlapping time range based on the data
    imu_start_time = imu_df['datetime'].min().to_pydatetime() if imu_df is not None else None
    imu_end_time = imu_df['datetime'].max().to_pydatetime() if imu_df is not None else None
    ephys_start_time = ephys_df['datetime'].min().to_pydatetime() if ephys_df is not None else None
    ephys_end_time = ephys_df['datetime'].max().to_pydatetime() if ephys_df is not None else None

    # Determine overlapping time range
    start_time = max(imu_start_time, ephys_start_time) if imu_start_time and ephys_start_time else imu_start_time or ephys_start_time
    end_time = min(imu_end_time, ephys_end_time) if imu_end_time and ephys_end_time else imu_end_time or ephys_end_time

    # Apply the user-defined time range if provided
    if time_range:
        # Constrain within the overlap time range
        user_start_time = time_range[0].to_pydatetime() if time_range[0].tzinfo is None else time_range[0]
        user_end_time = time_range[1].to_pydatetime() if time_range[1].tzinfo is None else time_range[1]
        
        start_time = max(start_time, user_start_time)
        end_time = min(end_time, user_end_time)

    # Filter data to the overlapping time range
    def get_time_filtered_df(df, start_time, end_time):
        return df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
    
    def downsample(df, original_fs, target_fs):
        if target_fs >= original_fs:
            return df
        conversion_factor = int(original_fs / target_fs)
        return df.iloc[::conversion_factor, :]

    if imu_logger:
        imu_fs = 1 / imu_df['datetime'].diff().dt.total_seconds().mean()
        imu_df_downsampled = downsample(imu_df, imu_fs, imu_sampling_rate)
        imu_df_filtered = get_time_filtered_df(imu_df_downsampled, start_time, end_time)
        imu_info = data_pkl.logger_info[imu_logger]['channelinfo']

    if ephys_logger:
        ephys_fs = 1 / ephys_df['datetime'].diff().dt.total_seconds().mean()
        ephys_df_downsampled = downsample(ephys_df, ephys_fs, ephys_sampling_rate)
        ephys_df_filtered = get_time_filtered_df(ephys_df_downsampled, start_time, end_time)
        ephys_info = data_pkl.logger_info[ephys_logger]['channelinfo']

    # Set up plotting
    num_rows = len(ordered_channels)
    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    row_counter = 1

    for channel_type, channels in ordered_channels:
        if channel_type in ['ECG'] and ephys_channels:
            channel = channels.lower()
            df = ephys_df_filtered
            info = ephys_info
            original_name = info[channel]['original_name']
            unit = info[channel]['unit']

            y_data = df[channel]
            x_data = df['datetime']

            y_label = f"{original_name} [{unit}]"
            color = color_mapping.get(channel.upper(), '#FFCCCC' if channel == 'ecg' else '#FF6347')

            # Add vertical lines for annotations if ECG is being plotted
            if channel == 'ecg' and note_annotations and 'heartbeat_manual_ok' in note_annotations:
                filtered_notes = data_pkl.event_data[data_pkl.event_data['key'] == 'heartbeat_manual_ok']
                if not filtered_notes.empty:
                    for dt in filtered_notes['datetime']:
                        fig.add_trace(go.Scatter(
                            x=[dt, dt],
                            y=[y_data.min(), y_data.max()],
                            mode='lines',
                            line=dict(color=color_mapping.get('Filtered Heartbeats', '#808080'), width=1, dash='dot'),
                            showlegend=False
                        ), row=row_counter, col=1)

            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=y_label,
                line=dict(color=color)
            ), row=row_counter, col=1)

            fig.update_yaxes(title_text=y_label, row=row_counter, col=1)
            row_counter += 1

        elif channel_type == 'Depth' and 'depth' in [ch.lower() for ch in imu_channels]:
            channel = 'depth'
            df = imu_df_filtered
            info = imu_info
            original_name = info[channel]['original_name']
            unit = info[channel]['unit']

            y_data = df[channel]
            x_data = df['datetime']

            y_label = f"{original_name} [{unit}]"
            color = color_mapping.get('Depth', '#00008B')

            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=y_label,
                line=dict(color=color)
            ), row=row_counter, col=1)

            if note_annotations and 'exhalation_breath' in note_annotations:
                filtered_notes = data_pkl.event_data[data_pkl.event_data['key'] == 'exhalation_breath']
                if not filtered_notes.empty:  
                    for dt in filtered_notes['datetime']:
                        fig.add_trace(go.Scatter(
                            x=[dt, dt],
                            y=[y_data.min(), y_data.max()],
                            mode='lines',
                            line=dict(color=color_mapping.get('Exhalation Breath', '#0000FF'), width=1, dash='dot'),
                            showlegend=False
                        ), row=row_counter, col=1)

            fig.update_yaxes(title_text=y_label, autorange="reversed", row=row_counter, col=1)
            row_counter += 1

        elif channel_type == 'Corrected Depth' and 'corrdepth' in imu_df_filtered.columns:
            channel = 'corrdepth'
            df = imu_df_filtered
            original_name = "Corrected Depth"
            unit = 'm'

            y_data = df[channel]
            x_data = df['datetime']

            y_label = f"{original_name} [{unit}]"
            color = color_mapping.get('Corrected Depth', '#6CA1C3')

            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=y_label,
                line=dict(color=color)
            ), row=row_counter-1, col=1)  # Plot on the same row as original depth

        elif channel_type in ['Accel', 'Gyro', 'Mag', 
                              'Accel_Adjusted', 'Gyro_Adjusted', 'Mag_Adjusted', 
                              'Accel_Corrected', 'Gyro_Corrected', 'Mag_Corrected', 'prh']:
            for sub_channel in channels:
                if sub_channel in imu_df_filtered.columns:
                    df = imu_df_filtered
                    info = imu_info
                    original_name = info[sub_channel]['original_name']
                    unit = info[sub_channel]['unit']

                    y_data = df[sub_channel]
                    x_data = df['datetime']

                    y_label = f"{original_name} [{unit}]"
                    color = color_mapping.get(original_name, '#000000')

                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines',
                        name=y_label,
                        line=dict(color=color)
                    ), row=row_counter, col=1)

            fig.update_yaxes(title_text=f"{channel_type} [{unit}]", row=row_counter, col=1)
            row_counter += 1

        elif channel_type == 'Undefined_IMU':
            channel = channels  # Single channel
            df = imu_df_filtered

            if channel in df.columns:
                y_data = df[channel]
                x_data = df['datetime']

                y_label = channel  # Use the channel name directly
                color = color_mapping.get(y_label)

                # If no color mapping exists, create one
                if color is None:
                    color = generate_random_color()
                    color_mapping[y_label] = color
                    if color_mapping_path:
                        save_color_mapping(color_mapping, color_mapping_path)

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=y_label,
                    line=dict(color=color)
                ), row=row_counter, col=1)

                fig.update_yaxes(title_text=y_label, row=row_counter, col=1)
                row_counter += 1

        elif channel_type == 'Undefined_EPHYS':
            channel = channels  # Single channel
            df = ephys_df_filtered

            if channel in df.columns:
                y_data = df[channel]
                x_data = df['datetime']

                y_label = channel  # Use the channel name directly
                color = color_mapping.get(y_label)

                # If no color mapping exists, create one
                if color is None:
                    color = generate_random_color()
                    color_mapping[y_label] = color
                    if color_mapping_path:
                        save_color_mapping(color_mapping, color_mapping_path)

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=y_label,
                    line=dict(color=color)
                ), row=row_counter, col=1)

                fig.update_yaxes(title_text=y_label, row=row_counter, col=1)
                row_counter += 1

    fig.update_layout(
        height=200 * num_rows,
        width=1200,
        hovermode="x unified",  # Enables the vertical hover line across subplots
        title_text=f"{data_pkl.deployment_info['Deployment ID']}",
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend
            xanchor='center',  # Anchor the legend horizontally at the center
            yanchor='top'   # Anchor the legend vertically at the top of the legend box
        )
    )

    fig.update_xaxes(title_text="Datetime", row=row_counter-1, col=1)

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


