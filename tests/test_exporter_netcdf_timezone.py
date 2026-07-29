"""
Regression tests for NetCDF export of timezone-aware signal data.

NetCDF/CF has no tz-aware datetime type, so xarray raises
"Cannot interpret 'datetime64[ns, UTC]' as a data type" when a tz-aware
coordinate reaches to_netcdf(). This previously produced a metadata-only
NetCDF that downstream steps read as "no usable signal available".
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyologger.io_operations.base_exporter import BaseExporter


class _FakeReader:
    """Minimal stand-in for DataReader with only what save_to_netcdf touches."""

    def __init__(self, signal_data):
        self.signal_data = signal_data
        self.signal_info = {}
        self.deployment_info = {"Time Zone": "UTC"}
        self.procedure_info = {}
        self.animal_info = {"Animal_ID": "cafa-001"}
        self.dataset_info = {"Dataset_ID": "cafa-adult_tdr-imu_BT"}
        self.event_data = pd.DataFrame()


def _signals(tz="UTC", n=200):
    dt = pd.date_range("2021-08-03 17:34:21", periods=n, freq="3s", tz=tz)
    return {
        "accelerometer": pd.DataFrame({
            "datetime": dt,
            "ax": np.random.rand(n),
            "ay": np.random.rand(n),
            "az": np.random.rand(n),
        }),
        "depth": pd.DataFrame({
            "datetime": dt,
            "corrected_depth": np.random.rand(n) * 100.0,
        }),
    }


def test_tz_aware_signals_are_written(tmp_path):
    path = tmp_path / "tz_aware.nc"
    reader = _FakeReader(_signals(tz="UTC"))
    BaseExporter(reader).save_to_netcdf(reader, str(path))

    with xr.open_dataset(path) as ds:
        written = sorted(str(v) for v in ds.variables if str(v).startswith("signal_data_"))
        assert written == ["signal_data_accelerometer", "signal_data_depth"]
        # Coordinates must be stored tz-naive to match existing exports.
        assert ds["accelerometer_samples"].dtype == np.dtype("datetime64[ns]")


def test_tz_aware_export_preserves_utc_instants(tmp_path):
    path = tmp_path / "instants.nc"
    reader = _FakeReader(_signals(tz="UTC"))
    BaseExporter(reader).save_to_netcdf(reader, str(path))

    with xr.open_dataset(path) as ds:
        first = pd.Timestamp(ds["accelerometer_samples"].values[0])
    assert first == pd.Timestamp("2021-08-03 17:34:21")


def test_non_utc_timezone_is_converted_not_truncated(tmp_path):
    """A non-UTC tz must be converted to UTC, not have its tzinfo dropped."""
    path = tmp_path / "non_utc.nc"
    reader = _FakeReader(_signals(tz="America/New_York"))
    BaseExporter(reader).save_to_netcdf(reader, str(path))

    with xr.open_dataset(path) as ds:
        first = pd.Timestamp(ds["accelerometer_samples"].values[0])
    expected = (
        pd.Timestamp("2021-08-03 17:34:21", tz="America/New_York")
        .tz_convert("UTC")
        .tz_localize(None)
    )
    assert first == expected


def test_tz_naive_signals_still_written(tmp_path):
    path = tmp_path / "naive.nc"
    reader = _FakeReader(_signals(tz=None))
    BaseExporter(reader).save_to_netcdf(reader, str(path))

    with xr.open_dataset(path) as ds:
        written = [str(v) for v in ds.variables if str(v).startswith("signal_data_")]
    assert len(written) == 2


def test_all_nan_object_signal_is_still_written(tmp_path):
    """An all-NaN label column must not silently drop the whole signal."""
    n = 50
    dt = pd.date_range("2021-08-03 17:34:21", periods=n, freq="3s", tz="UTC")
    reader = _FakeReader({
        "label": pd.DataFrame({"datetime": dt, "deployment_phase": [np.nan] * n}),
    })
    path = tmp_path / "label.nc"
    BaseExporter(reader).save_to_netcdf(reader, str(path))

    with xr.open_dataset(path) as ds:
        assert "signal_data_label" in ds.variables


def test_export_raises_when_signals_present_but_none_written(tmp_path):
    """
    The core regression: a metadata-only NetCDF must never be reported as a
    successful export when signal_data was non-empty.
    """
    reader = _FakeReader({
        "depth": pd.DataFrame(columns=["datetime", "corrected_depth"]),
    })
    with pytest.raises(ValueError, match="no signal variables"):
        BaseExporter(reader).save_to_netcdf(reader, str(tmp_path / "empty.nc"))


def test_export_of_genuinely_empty_reader_is_allowed(tmp_path):
    """A deployment with no signals at all may still write a placeholder."""
    reader = _FakeReader({})
    path = tmp_path / "placeholder.nc"
    BaseExporter(reader).save_to_netcdf(reader, str(path))
    assert path.exists()
