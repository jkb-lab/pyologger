"""
geoai_data.py — Environmental data download utilities for EcoViz/DiveDB.

Provides functions to download and explore environmental datasets for a
geographic bounding box:

  Land / vegetation (via geoai-py + Microsoft Planetary Computer STAC):
    - NDVI from Sentinel-2 (bands B08 NIR, B04 Red)
    - ESA WorldCover land-cover (10 m categorical, 2020/2021)

  Topo-bathymetry:
    - Copernicus DEM GLO-30 / GLO-90 (30 m / 90 m, Planetary Computer)
    - NOAA ETOPO 2022 (15 arc-second global topo-bathy, NCEI)

  Ocean (via ERDDAP):
    - Sea Surface Temperature — NOAA OISST v2.1 (0.25°, daily)
    - Chlorophyll-a — MODIS-Aqua (4 km, 8-day composite)

  Discovery:
    - List all STAC collections that intersect a bbox / date range
      (Planetary Computer + Earth Search on AWS)

All raster functions return an xarray.DataArray so results drop straight
into the existing plotter / environmental-covariate pipeline.

Typical usage
-------------
>>> from pyologger.utils.geoai_data import download_ndvi, download_sst
>>> ndvi = download_ndvi(bbox=(-122.5, 37.5, -121.5, 38.5),
...                      date_range=("2023-06-01", "2023-08-31"))
>>> sst  = download_sst(bbox=(-130, 30, -110, 50),
...                     date_range=("2023-06-01", "2023-08-31"))
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

BBox = Tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_to_aoi(bbox: BBox) -> Dict:
    """Return a GeoJSON polygon dict for pystac-client spatial filter."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ]],
    }


def _require(package: str, extra_hint: str = "") -> None:
    """Raise a clear ImportError if *package* is not installed."""
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        msg = (
            f"'{package}' is required for this function but is not installed. "
            f"Run: pip install {package}"
        )
        if extra_hint:
            msg += f"\n{extra_hint}"
        raise ImportError(msg) from None


# ---------------------------------------------------------------------------
# STAC discovery
# ---------------------------------------------------------------------------

def discover_available_stac_data(
    bbox: BBox,
    date_range: Tuple[str, str],
    catalogs: Optional[List[str]] = None,
    max_items_per_collection: int = 3,
) -> pd.DataFrame:
    """List STAC collections that have data for *bbox* and *date_range*.

    Parameters
    ----------
    bbox:
        (lon_min, lat_min, lon_max, lat_max) in WGS-84 degrees.
    date_range:
        ("YYYY-MM-DD", "YYYY-MM-DD") start / end strings.
    catalogs:
        STAC root URLs to query.  Defaults to Planetary Computer + Earth Search.
    max_items_per_collection:
        How many items to retrieve per collection (just for counting).

    Returns
    -------
    pd.DataFrame with columns: catalog, collection_id, item_count, description.
    """
    _require("pystac_client")

    import pystac_client

    if catalogs is None:
        catalogs = [
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            "https://earth-search.aws.element84.com/v1",
        ]

    date_str = f"{date_range[0]}/{date_range[1]}"
    rows: List[Dict] = []

    for catalog_url in catalogs:
        try:
            # Planetary Computer needs a modifier to sign asset HREFs
            modifier = None
            if "planetarycomputer" in catalog_url:
                try:
                    import planetary_computer
                    modifier = planetary_computer.sign_inplace
                except ImportError:
                    pass

            cat = pystac_client.Client.open(catalog_url, modifier=modifier)
            search = cat.search(
                bbox=list(bbox),
                datetime=date_str,
                max_items=max_items_per_collection * 50,  # broad first pass
            )
            # group by collection
            collection_counts: Dict[str, int] = {}
            for item in search.items():
                cid = item.collection_id or "unknown"
                collection_counts[cid] = collection_counts.get(cid, 0) + 1

            for cid, count in sorted(collection_counts.items()):
                try:
                    coll = cat.get_collection(cid)
                    description = (coll.description or "")[:120] if coll else ""
                except Exception:
                    description = ""
                rows.append({
                    "catalog": catalog_url.split("/")[2],
                    "collection_id": cid,
                    "item_count_sample": count,
                    "description": description,
                })
        except Exception as exc:
            warnings.warn(f"Could not query {catalog_url}: {exc}")

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["catalog", "collection_id", "item_count_sample", "description"]
    )


# ---------------------------------------------------------------------------
# NDVI — Sentinel-2 via Planetary Computer
# ---------------------------------------------------------------------------

def download_ndvi(
    bbox: BBox,
    date_range: Tuple[str, str],
    output_dir: Optional[Union[str, Path]] = None,
    max_cloud_cover: float = 20.0,
    max_items: int = 5,
    resolution: int = 10,
) -> "xarray.DataArray":  # type: ignore[name-defined]
    """Download Sentinel-2 imagery and return a median NDVI DataArray.

    NDVI = (B08_NIR − B04_Red) / (B08_NIR + B04_Red)

    Parameters
    ----------
    bbox:
        (lon_min, lat_min, lon_max, lat_max).
    date_range:
        ("YYYY-MM-DD", "YYYY-MM-DD").
    output_dir:
        If provided, save the NDVI GeoTIFF here.
    max_cloud_cover:
        Maximum cloud cover percentage to accept (0–100).
    max_items:
        Maximum number of Sentinel-2 scenes to composite.
    resolution:
        Output spatial resolution in metres (default 10 m).

    Returns
    -------
    xarray.DataArray  (y, x) with CRS attached via rioxarray, values in [-1, 1].
    """
    for pkg in ("pystac_client", "planetary_computer", "rioxarray", "stackstac"):
        _require(pkg)

    import planetary_computer
    import pystac_client
    import rioxarray  # noqa: F401 — registers .rio accessor
    import stackstac
    import xarray as xr

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=max_items,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for bbox={bbox}, dates={date_range}, "
            f"cloud_cover<{max_cloud_cover}%"
        )

    stack = stackstac.stack(
        items,
        assets=["B04", "B08"],
        bounds_latlon=list(bbox),
        resolution=resolution,
        dtype="float32",
    )
    # Median composite across time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        median = stack.median(dim="time").compute()

    red = median.sel(band="B04").astype("float32")
    nir = median.sel(band="B08").astype("float32")

    ndvi: xr.DataArray = (nir - red) / (nir + red + 1e-9)
    ndvi = ndvi.rename("ndvi")
    ndvi.attrs["long_name"] = "NDVI (Sentinel-2 median composite)"
    ndvi.attrs["valid_range"] = [-1.0, 1.0]
    ndvi.attrs["date_range"] = list(date_range)
    ndvi.attrs["bbox"] = list(bbox)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"ndvi_{date_range[0]}_{date_range[1]}.tif"
        ndvi.rio.to_raster(str(fname))
        print(f"[geoai_data] NDVI saved → {fname}")

    return ndvi


# ---------------------------------------------------------------------------
# Land cover — ESA WorldCover via Planetary Computer
# ---------------------------------------------------------------------------

# Mapping of ESA WorldCover integer codes to human-readable labels
ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def download_land_cover(
    bbox: BBox,
    output_dir: Optional[Union[str, Path]] = None,
    year: int = 2021,
    resolution: int = 10,
) -> "xarray.DataArray":  # type: ignore[name-defined]
    """Download ESA WorldCover land-cover for *bbox*.

    Parameters
    ----------
    bbox:
        (lon_min, lat_min, lon_max, lat_max).
    output_dir:
        If provided, save the result as a GeoTIFF here.
    year:
        2020 or 2021 (only two WorldCover editions).
    resolution:
        Output resolution in metres (default 10 m, the native resolution).

    Returns
    -------
    xarray.DataArray  (y, x) with integer class codes and ``attrs["classes"]``
    containing the ESA_WORLDCOVER_CLASSES dict.
    """
    for pkg in ("pystac_client", "planetary_computer", "rioxarray", "stackstac"):
        _require(pkg)

    import planetary_computer
    import pystac_client
    import rioxarray  # noqa: F401
    import stackstac

    if year not in (2020, 2021):
        raise ValueError(f"ESA WorldCover is only available for 2020 or 2021, got {year}")

    collection = f"esa-worldcover"
    date_range = (f"{year}-01-01", f"{year}-12-31")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{date_range[0]}/{date_range[1]}",
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No ESA WorldCover items found for bbox={bbox}, year={year}"
        )

    stack = stackstac.stack(
        items,
        assets=["map"],
        bounds_latlon=list(bbox),
        resolution=resolution,
        dtype="uint8",
    )
    # WorldCover tiles don't overlap in time — just squeeze
    lc = stack.squeeze("time", drop=True).squeeze("band", drop=True).compute()
    lc = lc.rename("land_cover")
    lc.attrs["long_name"] = f"ESA WorldCover {year}"
    lc.attrs["classes"] = ESA_WORLDCOVER_CLASSES
    lc.attrs["year"] = year
    lc.attrs["bbox"] = list(bbox)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"land_cover_esa_worldcover_{year}.tif"
        lc.rio.to_raster(str(fname))
        print(f"[geoai_data] Land cover saved → {fname}")

    return lc


# ---------------------------------------------------------------------------
# SST — NOAA OISST v2.1 via ERDDAP
# ---------------------------------------------------------------------------

_OISST_ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap"
_OISST_DATASET_ID = "ncdcOisst21Agg_LonPM180"


def download_sst(
    bbox: BBox,
    date_range: Tuple[str, str],
    output_dir: Optional[Union[str, Path]] = None,
    altitude: float = 0.0,
) -> "xarray.DataArray":  # type: ignore[name-defined]
    """Download NOAA OISST v2.1 sea-surface temperature.

    Daily 0.25° global SST grid, September 1981 – present.

    Parameters
    ----------
    bbox:
        (lon_min, lat_min, lon_max, lat_max) in WGS-84 degrees (-180 to +180).
    date_range:
        ("YYYY-MM-DD", "YYYY-MM-DD").
    output_dir:
        If provided, save result as NetCDF here.
    altitude:
        ERDDAP altitude slice (0.0 for surface, the only meaningful value).

    Returns
    -------
    xarray.DataArray  (time, latitude, longitude), SST in °C.
    """
    _require("erddapy", extra_hint="pip install erddapy")

    from erddapy import ERDDAP

    e = ERDDAP(server=_OISST_ERDDAP_URL, protocol="griddap")
    e.dataset_id = _OISST_DATASET_ID
    e.griddap_initialize()

    lon_min, lat_min, lon_max, lat_max = bbox
    e.constraints = {
        "time>=": date_range[0],
        "time<=": date_range[1],
        "altitude>=": altitude,
        "altitude<=": altitude,
        "latitude>=": lat_min,
        "latitude<=": lat_max,
        "longitude>=": lon_min,
        "longitude<=": lon_max,
    }
    e.variables = ["sst"]

    ds = e.to_xarray()
    da: "xarray.DataArray" = ds["sst"]
    da.attrs["source"] = "NOAA OISST v2.1"
    da.attrs["erddap_url"] = _OISST_ERDDAP_URL
    da.attrs["erddap_dataset_id"] = _OISST_DATASET_ID
    da.attrs["date_range"] = list(date_range)
    da.attrs["bbox"] = list(bbox)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"sst_oisst_{date_range[0]}_{date_range[1]}.nc"
        ds[["sst"]].to_netcdf(str(fname))
        print(f"[geoai_data] SST saved → {fname}")

    return da


# ---------------------------------------------------------------------------
# Chlorophyll-a — MODIS-Aqua 8-day composite via ERDDAP
# ---------------------------------------------------------------------------

_CHLA_ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap"
_CHLA_DATASET_ID = "erdMH1chla8day"  # MODIS-Aqua 8-day, 4 km


def download_chlorophyll(
    bbox: BBox,
    date_range: Tuple[str, str],
    output_dir: Optional[Union[str, Path]] = None,
    dataset_id: Optional[str] = None,
) -> "xarray.DataArray":  # type: ignore[name-defined]
    """Download MODIS-Aqua chlorophyll-a (8-day composite, 4 km).

    Parameters
    ----------
    bbox:
        (lon_min, lat_min, lon_max, lat_max) in WGS-84 degrees (-180 to +180).
    date_range:
        ("YYYY-MM-DD", "YYYY-MM-DD").
    output_dir:
        If provided, save result as NetCDF here.
    dataset_id:
        Override the default ERDDAP dataset ID (``erdMH1chla8day``).
        Use ``erdMH1chlamday`` for monthly composites.

    Returns
    -------
    xarray.DataArray  (time, latitude, longitude), chlorophyll in mg m⁻³.
    """
    _require("erddapy", extra_hint="pip install erddapy")

    from erddapy import ERDDAP

    ds_id = dataset_id or _CHLA_DATASET_ID
    e = ERDDAP(server=_CHLA_ERDDAP_URL, protocol="griddap")
    e.dataset_id = ds_id
    e.griddap_initialize()

    lon_min, lat_min, lon_max, lat_max = bbox
    e.constraints = {
        "time>=": date_range[0],
        "time<=": date_range[1],
        "latitude>=": lat_min,
        "latitude<=": lat_max,
        "longitude>=": lon_min,
        "longitude<=": lon_max,
    }
    e.variables = ["chlorophyll"]

    ds = e.to_xarray()
    da: "xarray.DataArray" = ds["chlorophyll"]
    da.attrs["source"] = "MODIS-Aqua via ERDDAP (erdMH1chla8day)"
    da.attrs["erddap_url"] = _CHLA_ERDDAP_URL
    da.attrs["erddap_dataset_id"] = ds_id
    da.attrs["units"] = "mg m-3"
    da.attrs["date_range"] = list(date_range)
    da.attrs["bbox"] = list(bbox)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"chlorophyll_modis_{date_range[0]}_{date_range[1]}.nc"
        ds[["chlorophyll"]].to_netcdf(str(fname))
        print(f"[geoai_data] Chlorophyll saved → {fname}")

    return da


# ---------------------------------------------------------------------------
# Convenience: summarise what's available for a bbox (all layers)
# ---------------------------------------------------------------------------

def summarise_region(
    bbox: BBox,
    date_range: Tuple[str, str],
    check_stac: bool = True,
    check_erddap: bool = True,
) -> None:
    """Print a human-readable summary of available data for *bbox*.

    Queries Planetary Computer STAC and ERDDAP to show item counts and
    availability. Useful for scoping a new region before downloading.

    Parameters
    ----------
    bbox:
        (lon_min, lat_min, lon_max, lat_max).
    date_range:
        ("YYYY-MM-DD", "YYYY-MM-DD").
    check_stac:
        Query Planetary Computer + Earth Search STAC catalogs.
    check_erddap:
        Probe NOAA ERDDAP endpoints for SST and chlorophyll availability.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    print(
        f"\n{'='*60}\n"
        f"Region summary\n"
        f"  bbox     : lon [{lon_min:.2f}, {lon_max:.2f}]  "
        f"lat [{lat_min:.2f}, {lat_max:.2f}]\n"
        f"  dates    : {date_range[0]} → {date_range[1]}\n"
        f"{'='*60}"
    )

    if check_stac:
        print("\n--- STAC collections (Planetary Computer + Earth Search) ---")
        try:
            df = discover_available_stac_data(bbox, date_range)
            if df.empty:
                print("  (no results)")
            else:
                for _, row in df.iterrows():
                    print(
                        f"  [{row['catalog']}]  {row['collection_id']}"
                        f"  ({row['item_count_sample']} items sampled)"
                    )
                    if row["description"]:
                        print(f"      {row['description']}")
        except Exception as exc:
            print(f"  STAC query failed: {exc}")

    if check_erddap:
        print("\n--- ERDDAP availability ---")
        try:
            import requests as _requests
        except ImportError:
            _requests = None

        for label, server_url, ds_id in [
            ("SST (NOAA OISST v2.1)",       _OISST_ERDDAP_URL, _OISST_DATASET_ID),
            ("Chlorophyll-a (MODIS 8-day)", _CHLA_ERDDAP_URL,  _CHLA_DATASET_ID),
        ]:
            # Lightweight check: just fetch the dataset info page (no metadata download)
            info_url = f"{server_url}/info/{ds_id}/index.json"
            try:
                if _requests is not None:
                    resp = _requests.get(info_url, timeout=8)
                    resp.raise_for_status()
                    print(f"  ✓  {label}")
                    print(f"      {server_url}/griddap/{ds_id}.html")
                else:
                    # Fall back to urllib if requests isn't available
                    import urllib.request
                    urllib.request.urlopen(info_url, timeout=8)
                    print(f"  ✓  {label}")
                    print(f"      {server_url}/griddap/{ds_id}.html")
            except Exception as exc:
                print(f"  ✗  {label}  — {exc}")

    print()
