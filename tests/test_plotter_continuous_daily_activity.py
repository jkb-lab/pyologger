import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pandas as pd
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "pyologger" / "plot_data" / "plotter.py"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-test-cache")
sys.path.insert(0, str(REPO_ROOT))
spec = importlib.util.spec_from_file_location("plotter_test", MODULE_PATH)
plotter = importlib.util.module_from_spec(spec)
sys.modules.setdefault(
    "plotly_resampler",
    types.SimpleNamespace(
        FigureWidgetResampler=go.Figure,
        FigureResampler=lambda fig=None, **_: fig if fig is not None else go.Figure(),
        register_plotly_resampler=lambda **_: None,
    ),
)
sys.modules[spec.name] = plotter
assert spec.loader is not None
spec.loader.exec_module(plotter)


def _fake_data_pkl():
    dt = pd.date_range("2021-04-01 00:00:00", periods=48, freq="1H", tz="UTC")
    depth = [0.0] * 8 + [50.0] * 16 + [1.0] * 8 + [70.0] * 16
    return SimpleNamespace(
        signal_data={"depth": pd.DataFrame({"datetime": dt, "depth": depth})},
        signal_info={"depth": {"channels": ["depth"], "metadata": {"depth": {"unit": "m"}}}},
        deployment_info={
            "Time Zone": "UTC",
            "Deployment Latitude": 36.6,
            "Deployment Longitude": -122.1,
        },
    )


def _seg_df():
    rows = [
        {
            "segment_rank": 1,
            "state_name": "putative_rest.surface_sleep",
            "keep_filtered": True,
            "start_datetime": pd.Timestamp("2021-04-01T01:00:00Z"),
            "end_datetime": pd.Timestamp("2021-04-01T03:00:00Z"),
            "segment_midpoint_datetime": pd.Timestamp("2021-04-01T02:00:00Z"),
            "duration_s": 7200.0,
            "drift_rate_ms": pd.NA,
        },
        {
            "segment_rank": 2,
            "state_name": "putative_rest.benthic_sleep",
            "keep_filtered": True,
            "start_datetime": pd.Timestamp("2021-04-01T10:00:00Z"),
            "end_datetime": pd.Timestamp("2021-04-01T12:00:00Z"),
            "segment_midpoint_datetime": pd.Timestamp("2021-04-01T11:00:00Z"),
            "duration_s": 7200.0,
            "drift_rate_ms": -0.05,
        },
        {
            "segment_rank": 3,
            "state_name": "putative_rest.drift_sleep",
            "keep_filtered": True,
            "start_datetime": pd.Timestamp("2021-04-02T14:00:00Z"),
            "end_datetime": pd.Timestamp("2021-04-02T16:00:00Z"),
            "segment_midpoint_datetime": pd.Timestamp("2021-04-02T15:00:00Z"),
            "duration_s": 7200.0,
            "drift_rate_ms": 0.12,
        },
    ]
    return pd.DataFrame(rows)


def test_continuous_daily_activity_plot_builds_rest_index_and_figures():
    payload = plotter.continuous_daily_activity_plot(
        data_pkl=_fake_data_pkl(),
        seg_df=_seg_df(),
        deployment_id="dep001",
        aggregation="1D",
        timezone_name="UTC",
        lat_lon_source={"lat": 36.6, "lon": -122.1},
        state_label_map=None,
        dive_depth_threshold_m=2.0,
        selected_segment_rank=2,
        color_mapping_path=None,
    )
    rest_index_df = payload["rest_index_df"]
    assert rest_index_df["segment_rank"].tolist() == [1, 2, 3]
    assert rest_index_df["state_name"].tolist() == [
        "putative_rest.surface_sleep",
        "putative_rest.benthic_sleep",
        "putative_rest.drift_sleep",
    ]
    assert "local_date" in rest_index_df.columns
    assert "local_hour" in rest_index_df.columns
    assert len(payload["daily_activity_fig"].data) == 5
    assert len(payload["rest_trip_fig"].data) >= 2


def test_continuous_daily_activity_plot_aggregation_changes_bin_count():
    payload_daily = plotter.continuous_daily_activity_plot(
        data_pkl=_fake_data_pkl(),
        seg_df=_seg_df(),
        deployment_id="dep001",
        aggregation="1D",
        timezone_name="UTC",
        lat_lon_source=None,
        state_label_map=None,
        dive_depth_threshold_m=2.0,
        color_mapping_path=None,
        show_solar_context=False,
    )
    payload_six_hour = plotter.continuous_daily_activity_plot(
        data_pkl=_fake_data_pkl(),
        seg_df=_seg_df(),
        deployment_id="dep001",
        aggregation="6H",
        timezone_name="UTC",
        lat_lon_source=None,
        state_label_map=None,
        dive_depth_threshold_m=2.0,
        color_mapping_path=None,
        show_solar_context=False,
    )
    assert len(payload_daily["daily_activity_fig"].data[0]["x"]) == 2
    assert len(payload_six_hour["daily_activity_fig"].data[0]["x"]) == 8


def test_continuous_daily_activity_plot_surface_sleep_stays_separate_without_drift_rate():
    payload = plotter.continuous_daily_activity_plot(
        data_pkl=_fake_data_pkl(),
        seg_df=_seg_df(),
        deployment_id="dep001",
        aggregation="1D",
        timezone_name="UTC",
        lat_lon_source=None,
        state_label_map=None,
        dive_depth_threshold_m=2.0,
        color_mapping_path=None,
        show_solar_context=False,
    )
    trace_names = [trace.name for trace in payload["rest_trip_fig"].data]
    assert "Surface sleep" in trace_names
    assert "Underwater rest" in trace_names
