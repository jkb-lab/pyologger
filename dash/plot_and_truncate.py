import os
import pickle
from datetime import timedelta
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, ctx
import plotly.graph_objs as go

# Your pyologger imports
from pyologger.utils.event_manager import *
from pyologger.process_data.sampling import *
from pyologger.utils.folder_manager import *
from pyologger.calibrate_data.zoc import *
from pyologger.plot_data.plotter import plot_tag_data_interactive

# Load configuration
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Initialize Dash app
app = Dash(__name__)
app.title = "Deployment Viewer"

# Load dataset & deployment
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment_streamlit(data_dir)

timezone = data_pkl.deployment_info.get("Time Zone", "UTC")

time_settings = param_manager.get_from_config(
    ["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],
    section="settings"
)

def round_to_nearest_second(dt):
    return dt.replace(microsecond=0)

OVERLAP_START_TIME = round_to_nearest_second(pd.Timestamp(time_settings["overlap_start_time"]).tz_convert(timezone))
OVERLAP_END_TIME = round_to_nearest_second(pd.Timestamp(time_settings["overlap_end_time"]).tz_convert(timezone))
ZOOM_WINDOW_START_TIME = round_to_nearest_second(pd.Timestamp(time_settings["zoom_window_start_time"]).tz_convert(timezone))
ZOOM_WINDOW_END_TIME = round_to_nearest_second(pd.Timestamp(time_settings["zoom_window_end_time"]).tz_convert(timezone))

time_values = [OVERLAP_START_TIME + timedelta(seconds=i) for i in range(0, int((OVERLAP_END_TIME - OVERLAP_START_TIME).total_seconds()) + 1)]

notes_to_plot = {
    'heartbeat_manual_ok': {'signal': 'ecg', 'symbol': 'triangle-down', 'color': 'blue'},
    'heartbeat_auto_detect_accepted': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'green'},
    'heartbeat_auto_detect_rejected': {'signal': 'ecg', 'symbol': 'triangle-up', 'color': 'red'},
    'strokebeat_auto_detect_accepted': {'signal': 'prh', 'symbol': 'triangle-up', 'color': 'green'},
    'exhalation_breath': {'signal': 'heart_rate', 'symbol': 'triangle-up', 'color': 'orange'}
}

TARGET_SAMPLING_RATE = 25

app.layout = html.Div([
    html.H1("Deployment Viewer"),
    
    html.Label("Select Time Range"),
    dcc.RangeSlider(
        id='time-slider',
        min=0,
        max=len(time_values)-1,
        step=1,
        value=[240, len(time_values) - 240],
        marks={i: time_values[i].strftime('%H:%M:%S') for i in range(0, len(time_values), len(time_values)//10)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Div(id='selected-range'),

    dcc.Graph(id='tag-plot'),

    html.Button("Update Config JSON", id='update-config', n_clicks=0),
    html.Div(id='config-status'),

    html.Button("Truncate and Save Pickle", id='truncate-save', n_clicks=0),
    html.Div(id='truncate-status')
])

@app.callback(
    Output('selected-range', 'children'),
    Output('tag-plot', 'figure'),
    Input('time-slider', 'value')
)
def update_plot(time_indices):
    start = time_values[time_indices[0]]
    end = time_values[time_indices[1]]
    fig = plot_tag_data_interactive(
        data_pkl=data_pkl,
        sensors=['ecg'],
        derived_data_signals=['depth','heart_rate', 'prh', 'stroke_rate'],
        note_annotations=notes_to_plot,
        zoom_start_time=start,
        zoom_end_time=end,
        time_range=(OVERLAP_START_TIME, OVERLAP_END_TIME),
        color_mapping_path=color_mapping_path,
        target_sampling_rate=TARGET_SAMPLING_RATE,
        zoom_range_selector_channel='depth'
    )
    return f"Selected range: {start} → {end}", fig

@app.callback(
    Output('config-status', 'children'),
    Input('update-config', 'n_clicks'),
    State('time-slider', 'value')
)
def update_config(n_clicks, time_indices):
    if n_clicks == 0:
        return ""
    start = time_values[time_indices[0]]
    end = time_values[time_indices[1]]
    param_manager.add_to_config(
        entries={"selected_start_time": str(start), "selected_end_time": str(end)},
        section="settings"
    )
    return "✅ Configuration JSON updated."

@app.callback(
    Output('truncate-status', 'children'),
    Input('truncate-save', 'n_clicks'),
    State('time-slider', 'value')
)
def truncate_and_save(n_clicks, time_indices):
    if n_clicks == 0:
        return ""
    start = time_values[time_indices[0]]
    end = time_values[time_indices[1]]
    overlap_start = pd.Timestamp(start).tz_convert(timezone)
    overlap_end = pd.Timestamp(end).tz_convert(timezone)

    for sensor, df in data_pkl.sensor_data.items():
        truncated_df = df[(df.iloc[:, 0] >= overlap_start) & (df.iloc[:, 0] <= overlap_end)].copy()
        data_pkl.sensor_data[sensor] = truncated_df

    midpoint = overlap_start + (overlap_end - overlap_start) / 2
    zoom_start = midpoint - timedelta(minutes=2.5)
    zoom_end = midpoint + timedelta(minutes=2.5)

    param_manager.add_to_config(entries={
        "overlap_start_time": str(overlap_start),
        "overlap_end_time": str(overlap_end),
        "zoom_window_start_time": str(zoom_start),
        "zoom_window_end_time": str(zoom_end)
    }, section="settings")

    pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)

    return f"✅ Truncated and saved from {overlap_start} to {overlap_end}."

if __name__ == '__main__':
    app.run_server(debug=True)
