from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    text = str(value).strip()
    if text in {"", "UNKNOWN", "Unknown", "unknown", "nan", "None"}:
        return pd.NA
    return value


def _global_attrs(nc: Dataset) -> dict[str, Any]:
    return {name: _clean_scalar(nc.getncattr(name)) for name in nc.ncattrs()}


def _first_present(*values: Any) -> Any:
    for value in values:
        if pd.notna(value):
            return value
    return pd.NA


def _deployment_date(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    ts = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(ts):
        return value
    return ts.date().isoformat()


def _normalize_timezone(value: Any, fallback: str | None = "UTC") -> Any:
    if pd.isna(value):
        return fallback if fallback is not None else pd.NA

    text = str(value).strip()
    if text == "":
        return fallback if fallback is not None else pd.NA

    try:
        numeric = float(text)
    except Exception:
        return text

    if np.isclose(numeric, 0.0):
        return "UTC"

    sign = "+" if numeric >= 0 else "-"
    abs_hours = abs(numeric)
    whole_hours = int(abs_hours)
    minutes = int(round((abs_hours - whole_hours) * 60))
    return f"UTC{sign}{whole_hours:02d}:{minutes:02d}"


def _slugify(value: Any, default: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or default


def _tagtools_signal_specs() -> dict[str, dict[str, Any]]:
    return {
        "A": {
            "rows": [
                ("ax", "accelerometer", "ax", "m/s^2", "accelerometer"),
                ("ay", "accelerometer", "ay", "m/s^2", "accelerometer"),
                ("az", "accelerometer", "az", "m/s^2", "accelerometer"),
            ],
        },
        "M": {
            "rows": [
                ("mx", "magnetometer", "mx", "uT", "magnetometer"),
                ("my", "magnetometer", "my", "uT", "magnetometer"),
                ("mz", "magnetometer", "mz", "uT", "magnetometer"),
            ],
        },
        "P": {
            "rows": [
                ("depth", "depth", "depth", "m", "depth"),
            ],
        },
        "T": {
            "rows": [
                ("temperature", "temperature", "temp_ext", "deg C", "temperature_ext"),
            ],
        },
        "J": {
            "rows": [
                ("jerk", "jerk", "jerk", "unknown", "jerk"),
            ],
        },
    }


def build_tagtools_montage_df(
    netcdf_path: str | Path,
    *,
    include_datetime: bool = True,
    include_location: bool = True,
) -> pd.DataFrame:
    """
    Build a montage template DataFrame for TagTools/DTAG netCDF content.

    The returned DataFrame matches the CSV schema expected by
    MontageManager.convert_df_to_montage_dict(...):
      - original_channel_id
      - original_unit
      - manufacturer_signal_name
      - standardized_channel_id
      - standardized_unit
      - parent_signal

    This template is aligned with the notebook-local DTAG importer output
    columns (`datetime`, `depth`, `temperature`, `ax`, `ay`, `az`, etc.).
    """
    netcdf_path = Path(netcdf_path)
    rows: list[dict[str, Any]] = []

    with Dataset(netcdf_path, "r") as nc:
        if include_datetime:
            rows.append(
                {
                    "original_channel_id": "datetime",
                    "original_unit": "YYYY-MM-DD HH:MM:SS.000",
                    "manufacturer_signal_name": "clock",
                    "standardized_channel_id": "datetime_utc",
                    "standardized_unit": "YYYY-MM-DD HH:MM:SS.000",
                    "parent_signal": "clock",
                }
            )

        for var_name, spec in _tagtools_signal_specs().items():
            if var_name not in nc.variables:
                continue

            var = nc.variables[var_name]
            original_unit = _first_present(
                _clean_scalar(var.getncattr("unit")) if "unit" in var.ncattrs() else pd.NA,
                _clean_scalar(var.getncattr("units")) if "units" in var.ncattrs() else pd.NA,
                "unknown",
            )

            for original_channel_id, manufacturer_signal_name, standardized_channel_id, standardized_unit, parent_signal in spec["rows"]:
                rows.append(
                    {
                        "original_channel_id": original_channel_id,
                        "original_unit": original_unit,
                        "manufacturer_signal_name": manufacturer_signal_name,
                        "standardized_channel_id": standardized_channel_id,
                        "standardized_unit": standardized_unit,
                        "parent_signal": parent_signal,
                    }
                )

        if include_location and "POS" in nc.variables:
            for original_channel_id, standardized_channel_id in (
                ("lat", "latitude"),
                ("lon", "longitude"),
            ):
                rows.append(
                    {
                        "original_channel_id": original_channel_id,
                        "original_unit": "decimal-degrees",
                        "manufacturer_signal_name": "location",
                        "standardized_channel_id": standardized_channel_id,
                        "standardized_unit": "DD",
                        "parent_signal": "location",
                    }
                )

    montage_df = pd.DataFrame(rows)
    if montage_df.empty:
        return pd.DataFrame(
            columns=[
                "original_channel_id",
                "original_unit",
                "manufacturer_signal_name",
                "standardized_channel_id",
                "standardized_unit",
                "parent_signal",
            ]
        )

    return montage_df.drop_duplicates(subset=["original_channel_id"], keep="last").reset_index(drop=True)


def extract_essential_metadata_from_tagtools_netcdf(
    netcdf_path: str | Path,
    *,
    manufacturer: str = "DT",
    tag_type: str = "DTAG",
    montage_id: str | None = None,
    logger_id: str | None = None,
    fallback_time_zone: str | None = "UTC",
):
    """
    Extract the minimum metadata needed to drive `00_load_data`-style notebook
    loading from a TagTools/DTAG NetCDF file.

    Returns:
      deployment_info, loggers_used, montage_df
    """
    netcdf_path = Path(netcdf_path)
    with Dataset(netcdf_path, "r") as nc:
        attrs = _global_attrs(nc)

    deployment_id = _first_present(
        attrs.get("depid"),
        netcdf_path.stem,
    )
    time_zone = _normalize_timezone(
        _first_present(attrs.get("dephist_device_tzone"), attrs.get("time_zone")),
        fallback=fallback_time_zone,
    )

    deployment_info = {
        "Deployment ID": deployment_id,
        "Deployment Date": _deployment_date(
            _first_present(
                attrs.get("dephist_deploy_datetime_start"),
                attrs.get("dephist_device_datetime_start"),
                attrs.get("time_coverage_start"),
            )
        ),
        "Deployment Latitude": _first_present(
            attrs.get("dephist_deploy_location_lat"),
            attrs.get("geospatial_lat_min"),
            attrs.get("lat"),
        ),
        "Deployment Longitude": _first_present(
            attrs.get("dephist_deploy_location_lon"),
            attrs.get("geospatial_lon_min"),
            attrs.get("lon"),
        ),
        "Time Zone": time_zone,
        "Procedure Info": {},
    }

    derived_logger_id = logger_id
    if derived_logger_id is None:
        derived_logger_id = _first_present(
            attrs.get("device_serial"),
            attrs.get("depid"),
            netcdf_path.stem,
        )

    derived_montage_id = montage_id
    if derived_montage_id is None:
        model_slug = _slugify(_first_present(attrs.get("device_model"), tag_type), default=_slugify(tag_type, "dtag"))
        derived_montage_id = f"tagtools-{model_slug}_V1"

    loggers_used = [
        {
            "Logger ID": str(derived_logger_id),
            "Manufacturer": manufacturer,
            "Montage ID": derived_montage_id,
            "Time Zone": time_zone,
        }
    ]

    montage_df = build_tagtools_montage_df(netcdf_path)

    return deployment_info, loggers_used, montage_df
