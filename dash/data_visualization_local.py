import os
import dash
from dash import dcc, html, Output, Input, State
import pandas as pd
import pickle
from pyologger.utils.json_manager import ConfigManager
from pyologger.load_data.datareader import DataReader
from pyologger.load_data.metadata import Metadata
from pyologger.plot_data.plotter import *
from pyologger.utils.data_manager import *
from pyologger.utils.event_manager import *
import three_js_orientation
import video_preview

# Define paths
root_dir = "/Users/jessiekb/Documents/GitHub/pyologger"
data_dir = os.path.join(root_dir, "data")
color_mapping_path = os.path.join(root_dir, "color_mappings.json")
deployment_id = "2024-01-16_oror-002a"
deployment_folder = os.path.join(data_dir, deployment_id)
pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
config_manager = ConfigManager(deployment_folder=deployment_folder, deployment_id=deployment_id)


# Retrieve values from config
variables = ["calm_horizontal_start_time", "calm_horizontal_end_time", 
             "zoom_window_start_time", "zoom_window_end_time", 
             "earliest_common_start_time", "latest_common_end_time",
             "video_start_time", "video_end_time", "video_filename"]
settings = config_manager.get_from_config(variables, section="settings")

# Assign retrieved values to variables
CALM_HORIZONTAL_START_TIME = settings.get("calm_horizontal_start_time")
CALM_HORIZONTAL_END_TIME = settings.get("calm_horizontal_end_time")
ZOOM_START_TIME = settings.get("zoom_window_start_time")
ZOOM_END_TIME = settings.get("zoom_window_end_time")
OVERLAP_START_TIME = settings.get("earliest_common_start_time")
OVERLAP_END_TIME = settings.get("latest_common_end_time")
VIDEO_START_TIME = settings.get("video_start_time")
VIDEO_END_TIME = settings.get("video_end_time")
VIDEO_FILENAME = settings.get("video_filename")

# Load the data_reader object from the pickle file
with open(pkl_path, 'rb') as file:
    data_pkl = pickle.load(file)

timezone = data_pkl.deployment_info['Time Zone']

app = dash.Dash(__name__)

df_all = data_pkl.derived_data['prh']

# Convert to UTC
#dff["datetime"] = dff["datetime"].dt.tz_localize("UTC")
# convert the datetime in dff to timezone
#dff["datetime"] = dff["datetime"] + pd.Timedelta(hours=-8)
print(f'')

dff = df_all[(df_all['datetime'] >= VIDEO_START_TIME) & (df_all['datetime'] <= VIDEO_END_TIME)]
dff.reset_index(drop=True, inplace=True)
# Convert datetime to timestamp (seconds since epoch) for slider control
dff["timestamp"] = dff["datetime"].apply(lambda x: x.timestamp())
# dff["depth"] = dff["derived_data_depth"].apply(lambda x: x * -1)

ORIGINAL_SAMPLING_RATE = 400
TARGET_SAMPLING_RATE = 10

notes_to_plot = {
    'heartbeat_manual_ok': {'signal': 'hr_normalized', 'symbol': 'triangle-down', 'color': 'blue'},
    'heartbeat_auto_detect_accepted': {'signal': 'hr_normalized', 'symbol': 'triangle-up', 'color': 'green'},
    'strokebeat_auto_detect_accepted': {'signal': 'sr_smoothed', 'symbol': 'triangle-up', 'color': 'green'},
    'exhalation_breath': {'signal': 'heart_rate', 'symbol': 'triangle-up', 'color': 'blue'}
}

fig = plot_tag_data_interactive5(
    data_pkl=data_pkl,
    sensors=['ecg', 'hr_normalized'],
    derived_data_signals=['depth', 'prh', 'stroke_rate', 'heart_rate','sr_smoothed', 'odba'],
    channels={}, #'corrected_gyr': ['broad_bandpassed_signal']
    time_range=(VIDEO_START_TIME, VIDEO_END_TIME),
    note_annotations=notes_to_plot,
    color_mapping_path=color_mapping_path,
    target_sampling_rate=TARGET_SAMPLING_RATE,
    zoom_start_time=VIDEO_START_TIME,
    zoom_end_time=VIDEO_END_TIME,
    zoom_range_selector_channel='depth',
    plot_event_values=[],
)

# Set x-axis range to data range and set uirevision
fig.update_layout(
    xaxis=dict(range=[dff["datetime"].min(), dff["datetime"].max()]),
    uirevision="constant",  # Maintain UI state across updates
)

# Convert DataFrame to JSON
data_json = dff[["datetime", "pitch", "roll", "heading"]].to_json(orient="split")

# Define the app layout
app.layout = html.Div(
    [
        html.Div(
            [
                three_js_orientation.ThreeJsOrientation(
                    id="three-d-model",
                    data=data_json,
                    activeTime=0,
                    objFile="/assets/6_killerWhale_v017_LP.obj",
                    textureFile="/assets/killerWhale_LP.png",
                    style={"width": "50vw", "height": "40vw"},
                ),
                video_preview.VideoPreview(
                    id="video-trimmer",
                    # Video file must be downloaded from https://figshare.com/ndownloader/files/50061327
                    videoSrc=f'/assets/{VIDEO_FILENAME}',
                    # activeTime=0,
                    playheadTime=dff["timestamp"].min(),
                    isPlaying=False,
                    style={"width": "50vw", "height": "40vw"},
                ),
            ],
            style={"display": "flex"},
        ),
        html.Div(
            "Playhead Control",
            style={
                "margin-top": "32px",
                "margin-right": "10px",
                "font-weight": "bold",
                "font-family": "sans-serif",
            },
        ),
        html.Div(
            [
                html.Button(
                    "Play", id="play-button", n_clicks=0, style={"margin-right": "10px"}
                ),
                html.Div(
                    dcc.Slider(
                        id="playhead-slider",
                        min=dff["timestamp"].min(),
                        max=dff["timestamp"].max(),
                        value=dff["timestamp"].min(),
                        marks=None,
                        tooltip={"placement": "bottom"},
                    ),
                    style={
                        "flex": "1",
                        "align-items": "center",
                        "margin-top": "25px",
                        "margin-right": "180px",
                    },
                ),
            ],
            style={"display": "flex", "align-items": "center", "width": "100%"},
        ),
        dcc.Interval(
            id="interval-component",
            interval=1 * 1000,  # Base interval of 1 second
            n_intervals=0,
            disabled=True,  # Start with the interval disabled
        ),
        dcc.Store(id="playhead-time", data=dff["timestamp"].min()),
        dcc.Store(id="is-playing", data=False),
        dcc.Graph(id="graph-content", figure=fig),
    ]
)


# Callback to update VideoPreview props
@app.callback(
    Output("video-trimmer", "playheadTime"),
    Output("video-trimmer", "isPlaying"),
    Input("playhead-time", "data"),
    Input("is-playing", "data"),
)
def update_video_preview(playhead_time, is_playing):
    return playhead_time, is_playing


@app.callback(
    Output("three-d-model", "activeTime"), [Input("playhead-slider", "value")]
)
def update_active_time(slider_value):
    # Find the nearest datetime to the slider value
    nearest_idx = dff["timestamp"].sub(slider_value).abs().idxmin()
    return nearest_idx


# Callback to toggle play/pause state
@app.callback(
    Output("is-playing", "data"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("is-playing", "data"),
)
def play_pause(n_clicks, is_playing):
    if n_clicks % 2 == 1:
        return True, "Pause"  # Switch to playing
    else:
        return False, "Play"  # Switch to paused


# Callback to enable/disable the interval component based on play state
@app.callback(Output("interval-component", "disabled"), Input("is-playing", "data"))
def update_interval_component(is_playing):
    return not is_playing  # Interval is disabled when not playing


# Callback to update playhead time based on interval or slider input
@app.callback(
    Output("playhead-time", "data"),
    Output("playhead-slider", "value"),
    Input("interval-component", "n_intervals"),
    Input("playhead-slider", "value"),
    State("is-playing", "data"),
    prevent_initial_call=True,
)
def update_playhead(n_intervals, slider_value, is_playing):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "interval-component" and is_playing:
        # Find the current index based on the slider value
        current_idx = dff["timestamp"].sub(slider_value).abs().idxmin()
        print(f'Current index: {current_idx}.')
        next_idx = (
            current_idx + ORIGINAL_SAMPLING_RATE if current_idx + ORIGINAL_SAMPLING_RATE < len(dff) else 0
        )  # Loop back to start
        print(f'Next index: {next_idx}.')
        new_time = dff["timestamp"].iloc[next_idx]
        print(f'New time: {new_time}.')
        return new_time, new_time
        
    elif trigger_id == "playhead-slider":
        return slider_value, slider_value
    else:
        raise dash.exceptions.PreventUpdate


# Callback to update the graph with the playhead line
@app.callback(
    Output("graph-content", "figure"),
    Input("playhead-time", "data"),
    State("graph-content", "figure"),
)
def update_graph(playhead_timestamp, existing_fig):
    playhead_time = pd.to_datetime(playhead_timestamp, unit="s").tz_localize("UTC")
    playhead_time = playhead_time.tz_convert(timezone)
    existing_fig["layout"]["shapes"] = []
    print(f'Playhead time: {playhead_time}.')
    existing_fig["layout"]["shapes"].append(
        dict(
            type="line",
            x0=playhead_time,
            x1=playhead_time,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(width=2, dash="solid"),
        )
    )
    existing_fig["layout"]["uirevision"] = "constant"
    return existing_fig


if __name__ == "__main__":
    app.run_server(debug=True)
