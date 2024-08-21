import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Color mapping dictionary with pastel, colorblind-friendly colors
color_mapping = {
    'Heart Rate': '#808080',       # Gray
    'ECG': '#FFCCCC',              # Light Red with alpha in rgba
    'Depth': '#00008B',            # Dark Blue
    'Accel X': '#87CEFA',          # Light Blue
    'Accel Y': '#98FB98',          # Pale Green
    'Accel Z': '#FF6347',          # Light Coral
    'Gyro Y': '#9370DB',           # Medium Purple
    'Filtered Heartbeats': '#808080',  # Gray for dotted lines
}

__all__ = ["plot_tag_data"]

def plot_tag_data(data_pkl, imu_channels, ephys_channels=None, imu_logger=None, ephys_logger=None, imu_sampling_rate=10, ephys_sampling_rate=50, draw=True):
    if not imu_logger and not ephys_logger:
        raise ValueError("At least one logger (imu_logger or ephys_logger) must be specified.")

    fig = make_subplots(rows=len(imu_channels) + (len(ephys_channels) if ephys_channels else 0), cols=1, shared_xaxes=True, vertical_spacing=0.03)
    
    def downsample(df, original_fs, target_fs):
        if target_fs >= original_fs:
            return df
        conversion_factor = int(original_fs / target_fs)
        return df.iloc[::conversion_factor, :]

    if imu_logger:
        imu_df = data_pkl.data[imu_logger]
        imu_fs = 1 / imu_df['datetime'].diff().dt.total_seconds().mean()
        imu_df_downsampled = downsample(imu_df, imu_fs, imu_sampling_rate)
        imu_info = data_pkl.info[imu_logger]['channelinfo']
    
    if ephys_logger:
        ephys_df = data_pkl.data[ephys_logger]
        ephys_fs = 1 / ephys_df['datetime'].diff().dt.total_seconds().mean()
        ephys_df_downsampled = downsample(ephys_df, ephys_fs, ephys_sampling_rate)
        ephys_info = data_pkl.info[ephys_logger]['channelinfo']

    row_counter = 1
    
    # Plot IMU channels
    for channel in imu_channels:
        if channel in imu_df_downsampled.columns:
            df = imu_df_downsampled
            info = imu_info
        else:
            raise ValueError(f"IMU Channel {channel} not found in the specified loggers' DataFrames.")
        
        original_name = info[channel]['original_name']
        unit = info[channel]['unit']
        is_depth = 'depth' in channel.lower() or channel.lower() == 'p'

        y_data = df[channel]
        x_data = df['datetime']

        y_label = f"{original_name} [{unit}]"
        color = color_mapping.get(original_name, '#000000')  # Default to black if not found

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            name=y_label,
            line=dict(color=color)
        ), row=row_counter, col=1)

        if is_depth:
            fig.update_yaxes(autorange="reversed", row=row_counter, col=1)

        fig.update_yaxes(title_text=y_label, row=row_counter, col=1)
        row_counter += 1

    # Plot ePhys channels
    if ephys_channels and ephys_logger:
        for channel in ephys_channels:
            if channel in ephys_df_downsampled.columns:
                df = ephys_df_downsampled
                info = ephys_info
            else:
                raise ValueError(f"ePhys Channel {channel} not found in the specified loggers' DataFrames.")
            
            original_name = info[channel]['original_name']
            unit = info[channel]['unit']

            y_data = df[channel]
            x_data = df['datetime']

            y_label = f"{original_name} [{unit}]"
            color = color_mapping.get(original_name, '#00FF00')  # Default to green if not found

            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=y_label,
                line=dict(color=color)
            ), row=row_counter, col=1)

            fig.update_yaxes(title_text=y_label, row=row_counter, col=1)

            # Add vertical lines for heartbeats if ECG is plotted
            if 'ecg' in channel.lower():
                filtered_notes = data_pkl.notes_df[data_pkl.notes_df['key'] == 'heartbeat_manual_ok']
                if not filtered_notes.empty:
                    for dt in filtered_notes['datetime']:
                        fig.add_trace(go.Scatter(
                            x=[dt, dt],
                            y=[y_data.min(), y_data.max()],
                            mode='lines',
                            line=dict(color=color_mapping['Filtered Heartbeats'], width=1, dash='dot'),
                            showlegend=False
                        ), row=row_counter, col=1)

            row_counter += 1

    fig.update_layout(
        height=200 * (len(imu_channels) + (len(ephys_channels) if ephys_channels else 0)),
        width=1200,
        title_text=f"{data_pkl.selected_deployment['Deployment Name']}",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Datetime", row=row_counter-1, col=1)

    if draw:
        fig.show()
    else:
        return fig
