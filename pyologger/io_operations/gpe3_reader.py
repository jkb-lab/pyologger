"""
Reader for Wildlife Computers GPE3 light/SST geolocation exports.

GPE3 files are a modelled *product*, not raw sensor data: the header carries a
few comment lines (grid size, model score, reference datasets) before the real
column header, and positions arrive as "Most Likely Latitude/Longitude".

These tracks are handled separately from the sensor montage path because a
GPE3 export shares a folder with ArchivedSeries/Series/DailyData files whose
layouts are mutually incompatible, and because the appropriate cleaning is
smoothing rather than an Argos-tuned SSM (see utils.geolocation_smoother).
"""

from __future__ import annotations

import os

import pandas as pd

# GPE3 timestamps are non-zero-padded, e.g. "8/3/2021 17:38" (no seconds).
GPE3_DATETIME_FORMAT = "%m/%d/%Y %H:%M"

LAT_COLUMN = "Most Likely Latitude"
LON_COLUMN = "Most Likely Longitude"


def find_header_row(path: str, max_scan: int = 40) -> int:
    """
    Locate the real header row, skipping the leading comment block.

    Comment lines start with ';' (sometimes quoted, since trailing commas make
    the CSV writer quote the whole field) and are followed by a blank row.
    Detecting the header by content rather than a fixed skiprows keeps this
    robust to GPE3 runs that emit a different number of preamble lines.
    """
    with open(path, "r", errors="replace") as handle:
        for index, line in enumerate(handle):
            if index >= max_scan:
                break
            probe = line.lstrip().lstrip('"').lstrip()
            if probe.startswith(";"):
                continue
            if LAT_COLUMN.lower() in line.lower():
                return index
    raise ValueError(
        f"Could not locate the GPE3 header row (expected a column named "
        f"'{LAT_COLUMN}') within the first {max_scan} lines of {path}."
    )


def read_gpe3(path: str) -> pd.DataFrame:
    """
    Read one GPE3 CSV into a tidy location frame.

    Returns columns: datetime (UTC-naive), latitude, longitude, plus the
    GPE3 provenance fields (observation_type, observation_score) when present.
    """
    header_row = find_header_row(path)
    df = pd.read_csv(path, skiprows=header_row)

    missing = [c for c in (LAT_COLUMN, LON_COLUMN, "Date") if c not in df.columns]
    if missing:
        raise ValueError(f"GPE3 file {os.path.basename(path)} is missing columns: {missing}")

    parsed = pd.to_datetime(df["Date"], format=GPE3_DATETIME_FORMAT, errors="coerce")
    if parsed.isna().mean() > 0.2:
        # Fall back to inference only if the documented format largely fails,
        # so a format change is visible rather than silently absorbed.
        print(
            f"⚠️ {parsed.isna().mean():.1%} of GPE3 dates failed format "
            f"{GPE3_DATETIME_FORMAT}; falling back to inferred parsing."
        )
        parsed = pd.to_datetime(df["Date"], errors="coerce")

    out = pd.DataFrame(
        {
            "datetime": parsed,
            "latitude": pd.to_numeric(df[LAT_COLUMN], errors="coerce"),
            "longitude": pd.to_numeric(df[LON_COLUMN], errors="coerce"),
        }
    )
    if "Observation Type" in df.columns:
        out["observation_type"] = df["Observation Type"]
    if "Observation Score" in df.columns:
        out["observation_score"] = pd.to_numeric(df["Observation Score"], errors="coerce")

    dropped = out["datetime"].isna() | out["latitude"].isna() | out["longitude"].isna()
    if dropped.any():
        print(f"ℹ️ Dropping {int(dropped.sum())} GPE3 rows lacking datetime/position.")
    return out.loc[~dropped].sort_values("datetime").reset_index(drop=True)


def find_gpe3_files(folder: str) -> list[str]:
    """Return GPE3 CSV paths under a logger folder (recursively)."""
    hits: list[str] = []
    for root, _, files in os.walk(folder):
        for name in files:
            if "gpe3" in name.lower() and name.lower().endswith(".csv"):
                hits.append(os.path.join(root, name))
    return sorted(hits)
