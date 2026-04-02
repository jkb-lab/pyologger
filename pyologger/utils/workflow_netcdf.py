import os
from typing import Iterable

import numpy as np
import xarray as xr


def _to_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]


def outputs_dir(deployment_folder: str) -> str:
    return os.path.join(deployment_folder, "outputs")


def step_netcdf_path(deployment_folder: str, deployment_id: str, step_number: int) -> str:
    return os.path.join(outputs_dir(deployment_folder), f"{deployment_id}_step{step_number:02d}.nc")


def final_netcdf_path(deployment_folder: str, deployment_id: str) -> str:
    return os.path.join(outputs_dir(deployment_folder), f"{deployment_id}_output.nc")


def latest_processing_netcdf_path(deployment_folder: str, deployment_id: str) -> str | None:
    candidates = [final_netcdf_path(deployment_folder, deployment_id)]
    candidates.extend(
        step_netcdf_path(deployment_folder, deployment_id, step_number)
        for step_number in range(5, -1, -1)
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def netcdf_has_signal(netcdf_path: str | None, signal_name: str) -> bool:
    if not netcdf_path or not os.path.exists(netcdf_path):
        return False
    with xr.open_dataset(netcdf_path) as ds:
        return f"signal_data_{signal_name}" in ds.variables


def netcdf_signal_names(netcdf_path: str | None) -> list[str]:
    """Return a sorted list of all signal names stored in a pyologger NetCDF file.

    Signal variables follow the ``signal_data_{name}`` naming convention.
    """
    prefix = "signal_data_"
    if not netcdf_path or not os.path.exists(netcdf_path):
        return []
    with xr.open_dataset(netcdf_path) as ds:
        return sorted({str(v)[len(prefix):] for v in ds.variables if str(v).startswith(prefix)})


def netcdf_signal_channels(netcdf_path: str | None, signal_name: str) -> list[str]:
    if not netcdf_path or not os.path.exists(netcdf_path):
        return []
    var_name = f"signal_data_{signal_name}"
    with xr.open_dataset(netcdf_path) as ds:
        if var_name not in ds.variables:
            return []
        da = ds[var_name]
        channels = _to_list(da.attrs.get("variables")) or _to_list(da.attrs.get("variable"))
    return [str(channel) for channel in channels]


def netcdf_signal_has_channel(netcdf_path: str | None, signal_name: str, channel_name: str) -> bool:
    return str(channel_name) in netcdf_signal_channels(netcdf_path, signal_name)


def netcdf_attr(netcdf_path: str | None, attr_name: str, default=None):
    if not netcdf_path or not os.path.exists(netcdf_path):
        return default
    with xr.open_dataset(netcdf_path) as ds:
        return ds.attrs.get(attr_name, default)


def save_step_netcdf_if_changed(data_pkl, deployment_folder: str, deployment_id: str, step_number: int, *, changed: bool) -> str | None:
    if not changed:
        print(f"Skipping NetCDF export for Step {step_number:02d}; no data changes were made.")
        return None
    from pyologger.io_operations.base_exporter import BaseExporter

    target_path = step_netcdf_path(deployment_folder, deployment_id, step_number)
    BaseExporter(data_pkl).save_to_netcdf(data_pkl, filepath=target_path)
    return target_path
