import pathlib
import sys

import pandas as pd
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyologger.utils.deployment_source import DeploymentDataSource, resolve_deployment_source, resolve_plot_signal_allowlist


class DummyParamManager:
    def __init__(self, saved_signals):
        self.saved_signals = saved_signals

    def get_from_config(self, variable_names=None, section=None):
        return {
            "dash_default_signals": list(self.saved_signals),
            "plotly_default_signals": list(self.saved_signals),
        }


def _write_test_netcdf(tmp_path):
    outputs = tmp_path / "dataset" / "deployment" / "outputs"
    outputs.mkdir(parents=True)
    t = pd.date_range("2024-01-01T00:00:00Z", periods=6, freq="1min").tz_localize(None)
    event_t = pd.DatetimeIndex([t[1], t[4]])
    ds = xr.Dataset(
        data_vars={
            "signal_data_depth": xr.DataArray(
                [1, 2, 3, 4, 5, 6],
                dims=["depth_samples"],
                coords={"depth_samples": t},
                attrs={"variables": ["depth"]},
            ),
            "signal_data_accelerometer": xr.DataArray(
                [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7]],
                dims=["accelerometer_samples", "accelerometer_variables"],
                coords={"accelerometer_samples": t, "accelerometer_variables": ["x", "y"]},
                attrs={"variables": ["x", "y"]},
            ),
            "signal_data_location": xr.DataArray(
                [[40.0, -120.0], [40.1, -120.1], [40.2, -120.2], [40.3, -120.3], [40.4, -120.4], [40.5, -120.5]],
                dims=["location_samples", "location_variables"],
                coords={"location_samples": t, "location_variables": ["latitude", "longitude"]},
                attrs={"variables": ["latitude", "longitude"]},
            ),
            "event_data_key": xr.DataArray(["a", "b"], dims=["event_data_samples"], coords={"event_data_samples": event_t}),
            "event_data_type": xr.DataArray(["point", "state"], dims=["event_data_samples"], coords={"event_data_samples": event_t}),
            "event_data_duration": xr.DataArray([0.0, 60.0], dims=["event_data_samples"], coords={"event_data_samples": event_t}),
        },
        attrs={
            "deployment_info_Time Zone": "UTC",
            "animal_info_Animal_ID": "animal-001",
            "dataset_info_Dataset_ID": "dataset",
            "signal_info_depth_channels_0": "depth",
            "signal_info_accelerometer_channels_0": "x",
            "signal_info_accelerometer_channels_1": "y",
            "signal_info_location_channels_0": "latitude",
            "signal_info_location_channels_1": "longitude",
        },
    )
    path = outputs / "deployment_output.nc"
    ds.to_netcdf(path)
    return outputs.parent, path


def test_netcdf_source_reads_header_and_window(tmp_path):
    deployment_folder, _path = _write_test_netcdf(tmp_path)
    source = DeploymentDataSource(str(deployment_folder), "deployment")

    assert source.backend == "netcdf"
    assert source.deployment_info["Time Zone"] == "UTC"
    assert source.signal_meta["accelerometer"].channels == ["x", "y"]
    assert source.event_keys() == ["a", "b"]

    start = pd.Timestamp("2024-01-01T00:01:00Z")
    end = pd.Timestamp("2024-01-01T00:03:00Z")
    depth_df = source.load_signal_window("depth", start, end)
    assert list(depth_df.columns) == ["datetime", "depth"]
    assert len(depth_df) == 3
    assert depth_df["depth"].tolist() == [2, 3, 4]
    assert str(depth_df["datetime"].dtype).startswith("datetime64[ns,")

    events_df = source.load_events_window(start, end)
    if not events_df.empty:
        assert str(events_df["datetime"].dtype).startswith("datetime64[ns,")


def test_metadata_shell_and_allowlist(tmp_path):
    deployment_folder, _path = _write_test_netcdf(tmp_path)
    source = DeploymentDataSource(str(deployment_folder), "deployment")
    allowlist = resolve_plot_signal_allowlist(DummyParamManager(["depth", "missing"]), source)

    assert allowlist == ["depth"]

    shell = source.build_metadata_shell(allowed_signals=allowlist)
    assert shell.plot_signal_allowlist == ["depth"]
    assert "depth" in shell.signal_data
    assert "location" in shell.signal_data
    assert list(shell.signal_data["depth"].columns) == ["datetime", "depth"]


def test_location_track_is_available_even_when_not_in_allowlist(tmp_path):
    deployment_folder, _path = _write_test_netcdf(tmp_path)
    source = DeploymentDataSource(str(deployment_folder), "deployment")
    shell = source.build_metadata_shell(allowed_signals=["depth"])

    assert shell.plot_signal_allowlist == ["depth"]
    track = source.load_location_track(max_points=3)
    assert len(track) == 3
    assert "latitude" in track.columns
    assert "longitude" in track.columns


def test_resolve_deployment_source_accepts_explicit_deployment_folder(tmp_path):
    deployment_folder, _path = _write_test_netcdf(tmp_path)
    source = resolve_deployment_source(
        data_dir="ignored",
        dataset_id="ignored",
        deployment_id="deployment",
        deployment_folder=str(deployment_folder),
    )

    assert source.backend == "netcdf"
    assert "depth" in source.signal_meta
