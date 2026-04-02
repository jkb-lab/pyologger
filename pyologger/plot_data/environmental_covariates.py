from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from pyologger.plot_data.plotter import get_bathy, get_ice_elevation, plot_track_with_bathy


SUPPORTED_COVARIATE_TYPES = {
    "bathymetry",
    "bathymetry_ice",
    "sst",
    "land_cover",
    "human_impact",
}
TIME_VARYING_COVARIATE_TYPES = {"sst", "human_impact"}
CATEGORICAL_COVARIATE_TYPES = {"land_cover"}
EXECUTABLE_COVARIATE_TYPES = {"bathymetry", "bathymetry_ice"}


def default_environmental_covariate_catalog() -> Dict:
    return {
        "dataset_catalog": {
            "bathymetry": {
                "default_source": "gebco_2025",
                "suggestions": {
                    "gebco_2025": {
                        "kind": "raster_static",
                        "provider": "GEBCO",
                        "url": "https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2025-grid",
                        "notes": "Global bathymetry/elevation grid; already used by current plotting code.",
                    }
                },
            },
            "bathymetry_ice": {
                "default_source": "gebco_2025_surface_plus_subice",
                "suggestions": {
                    "gebco_2025_surface_plus_subice": {
                        "kind": "raster_static",
                        "provider": "GEBCO",
                        "url": "https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2025-grid",
                        "notes": "Use GEBCO surface grid plus sub-ice grid to derive ice presence/elevation.",
                    }
                },
            },
            "sst": {
                "default_source": "noaa_oisst_v2_1",
                "suggestions": {
                    "noaa_oisst_v2_1": {
                        "kind": "raster_timevarying",
                        "provider": "NOAA NCEI",
                        "url": "https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
                        "notes": "Daily 0.25 degree global SST, September 1 1981 to present.",
                    }
                },
            },
            "land_cover": {
                "default_source": "esa_worldcover_2021",
                "suggestions": {
                    "esa_worldcover_2021": {
                        "kind": "raster_static_categorical",
                        "provider": "ESA WorldCover",
                        "url": "https://esa-worldcover.org/en/data-access",
                        "notes": "Global 10 m categorical land cover for 2021.",
                    },
                    "esa_worldcover_2020": {
                        "kind": "raster_static_categorical",
                        "provider": "ESA WorldCover",
                        "url": "https://esa-worldcover.org/en/data-access",
                        "notes": "Global 10 m categorical land cover for 2020.",
                    },
                },
            },
            "human_impact": {
                "default_source": "wcs_human_impact_index",
                "suggestions": {
                    "wcs_human_impact_index": {
                        "kind": "raster_timevarying",
                        "provider": "WCS",
                        "variable": "Human impact index",
                        "spatial_resolution": "300m",
                        "temporal_resolution": "Annual",
                        "year_range": [2001, 2020],
                        "notes": "User-specified WCS Human Impact Index, annual 300 m layer from 2001–2020.",
                    }
                },
            },
        }
    }


def merged_environmental_covariate_catalog(global_cfg: Dict) -> Dict:
    base = default_environmental_covariate_catalog()
    override = dict((global_cfg or {}).get("environmental_covariates") or {})
    if not override:
        return base
    merged = dict(base)
    merged_catalog = dict(base.get("dataset_catalog") or {})
    override_catalog = dict(override.get("dataset_catalog") or {})
    for cov_type, cfg in override_catalog.items():
        existing = dict(merged_catalog.get(cov_type) or {})
        suggestions = dict(existing.get("suggestions") or {})
        suggestions.update(dict((cfg or {}).get("suggestions") or {}))
        existing.update(cfg or {})
        existing["suggestions"] = suggestions
        merged_catalog[cov_type] = existing
    merged.update(override)
    merged["dataset_catalog"] = merged_catalog
    return merged


def resolve_covariate_source(global_cfg: Dict, cov_type: str, source: Optional[str] = None) -> Tuple[str, Dict]:
    catalog = merged_environmental_covariate_catalog(global_cfg).get("dataset_catalog") or {}
    type_cfg = dict(catalog.get(cov_type) or {})
    if not type_cfg:
        raise ValueError(f"Unknown environmental covariate type '{cov_type}'")
    source_name = source or type_cfg.get("default_source")
    suggestions = dict(type_cfg.get("suggestions") or {})
    source_cfg = dict(suggestions.get(source_name) or {})
    if not source_cfg:
        raise ValueError(f"Unknown source '{source_name}' for environmental covariate '{cov_type}'")
    return str(source_name), source_cfg


def normalize_covariate_request(global_cfg: Dict, map_id: str, cov_cfg: Dict) -> Dict:
    if not isinstance(cov_cfg, dict):
        raise ValueError(f"Map '{map_id}' has a non-dict environmental covariate entry")
    cov_type = str(cov_cfg.get("type") or "").strip()
    if cov_type not in SUPPORTED_COVARIATE_TYPES:
        raise ValueError(f"Map '{map_id}' uses unsupported environmental covariate type '{cov_type}'")
    source_name, source_cfg = resolve_covariate_source(global_cfg, cov_type, cov_cfg.get("source"))
    normalized = dict(cov_cfg)
    normalized["type"] = cov_type
    normalized["source"] = source_name
    normalized["source_config"] = source_cfg
    normalized.setdefault("time_strategy", "map_bin" if cov_type in TIME_VARYING_COVARIATE_TYPES else None)
    normalized.setdefault("year_strategy", "nearest" if cov_type == "human_impact" else None)
    normalized.setdefault("aggregation", cov_cfg.get("agg") or "nearest")
    if cov_type in TIME_VARYING_COVARIATE_TYPES and not normalized.get("time_strategy"):
        raise ValueError(f"Map '{map_id}' covariate '{cov_type}' requires a time_strategy")
    if cov_type == "human_impact":
        year_range = source_cfg.get("year_range") or []
        if len(year_range) == 2:
            normalized["year_range"] = [int(year_range[0]), int(year_range[1])]
    return normalized


def normalize_track_extent(location_df: pd.DataFrame, pad_deg: float = 20.0, global_extent: bool = False) -> Dict[str, float]:
    if location_df is None or location_df.empty:
        raise ValueError("Cannot compute map extent from empty location data")
    lat_vals = pd.to_numeric(location_df["lat"], errors="coerce").to_numpy(dtype=float)
    lon_vals = pd.to_numeric(location_df["lon"], errors="coerce").to_numpy(dtype=float)
    lat_vals = lat_vals[np.isfinite(lat_vals)]
    lon_vals = lon_vals[np.isfinite(lon_vals)]
    if lat_vals.size == 0 or lon_vals.size == 0:
        raise ValueError("Location data is missing finite lat/lon values")
    if global_extent:
        lat_min, lat_max = -90.0, 90.0
        lon0_180, lon1_180 = -180.0, 180.0
    else:
        lat_min = max(-90.0, float(lat_vals.min() - pad_deg))
        lat_max = min(90.0, float(lat_vals.max() + pad_deg))
        lon_360 = lon_vals % 360.0
        lon0_360 = (float(lon_360.min()) - pad_deg) % 360.0
        lon1_360 = (float(lon_360.max()) + pad_deg) % 360.0
        lon0_180 = float(((lon0_360 + 180.0) % 360.0) - 180.0)
        lon1_180 = float(((lon1_360 + 180.0) % 360.0) - 180.0)
    return {
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "lon0_180": float(lon0_180),
        "lon1_180": float(lon1_180),
        "pad_deg": float(pad_deg),
        "global_extent": bool(global_extent),
    }


def load_covariate_overlay(
    covariate_cfg: Dict,
    extent: Dict[str, float],
    cache: Dict,
) -> Dict:
    cov_type = covariate_cfg["type"]
    source = covariate_cfg["source"]
    cache_key = (
        cov_type,
        source,
        round(extent["lat_min"], 6),
        round(extent["lat_max"], 6),
        round(extent["lon0_180"], 6),
        round(extent["lon1_180"], 6),
        bool(extent["global_extent"]),
        int(covariate_cfg.get("max_pixels", 1200) or 1200),
    )
    if cache_key in cache:
        return cache[cache_key]

    if cov_type == "bathymetry":
        bathy_df = get_bathy(
            lat_min=extent["lat_min"],
            lat_max=extent["lat_max"],
            lon0_180=extent["lon0_180"],
            lon1_180=extent["lon1_180"],
            max_pixels=int(covariate_cfg.get("max_pixels", 1200) or 1200),
        )
        overlay = {"type": cov_type, "bathy_df": bathy_df, "ice_df": None}
    elif cov_type == "bathymetry_ice":
        bathy_df = get_bathy(
            lat_min=extent["lat_min"],
            lat_max=extent["lat_max"],
            lon0_180=extent["lon0_180"],
            lon1_180=extent["lon1_180"],
            max_pixels=int(covariate_cfg.get("max_pixels", 1200) or 1200),
        )
        ice_df = get_ice_elevation(
            lat_min=extent["lat_min"],
            lat_max=extent["lat_max"],
            lon0_180=extent["lon0_180"],
            lon1_180=extent["lon1_180"],
            max_pixels=int(covariate_cfg.get("max_pixels", 1200) or 1200),
        )
        overlay = {"type": cov_type, "bathy_df": bathy_df, "ice_df": ice_df}
    elif cov_type in {"sst", "land_cover", "human_impact"}:
        raise NotImplementedError(
            f"Environmental covariate '{cov_type}' with source '{source}' is schema-supported but not wired yet. "
            f"Only GEBCO-backed 'bathymetry' and 'bathymetry_ice' are executable in this pass."
        )
    else:
        raise ValueError(f"Unsupported environmental covariate type '{cov_type}'")

    cache[cache_key] = overlay
    return overlay


def validate_covariate_runtime_readiness(map_id: str, covariate_cfg: Dict) -> None:
    cov_type = covariate_cfg["type"]
    if cov_type not in EXECUTABLE_COVARIATE_TYPES:
        raise NotImplementedError(
            f"Map '{map_id}' requests environmental covariate '{cov_type}', which is schema-supported but not executable yet. "
            f"Only 'bathymetry' and 'bathymetry_ice' are wired in this implementation."
        )


def _save_plotnine_figure(fig, output_dir: str | Path, output_filename: str) -> Tuple[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_filename}.png"
    svg_path = output_dir / f"{output_filename}.svg"
    fig.save(png_path, dpi=300, verbose=False)
    fig.save(svg_path, dpi=300, verbose=False)
    return str(png_path), str(svg_path)


def plot_discrete_track_map(
    location_df: pd.DataFrame,
    output_dir: str | Path,
    output_filename: str,
    extent: Dict[str, float],
    covariate_overlay: Optional[Dict] = None,
):
    if covariate_overlay and covariate_overlay["type"] in {"bathymetry", "bathymetry_ice"}:
        fig = plot_track_with_bathy(
            lat=location_df["lat"].to_numpy(dtype=float),
            lon=location_df["lon"].to_numpy(dtype=float),
            toppid=location_df["deployment_id"].astype(str).to_numpy(),
            global_extent=bool(extent.get("global_extent", False)),
            pad_deg=float(extent.get("pad_deg", 20.0)),
            bathy_df=covariate_overlay.get("bathy_df"),
            ice_df=covariate_overlay.get("ice_df"),
            output_dir=output_dir,
            output_filename=output_filename,
            show_ice=bool(covariate_overlay["type"] == "bathymetry_ice"),
        )
        return fig

    from plotnine import aes, coord_fixed, geom_path, geom_point, ggplot, labs, scale_color_brewer, theme, theme_bw

    plot_df = location_df.copy()
    plot_df["lon360"] = plot_df["lon"].to_numpy(dtype=float) % 360.0
    mean_lat = float(plot_df["lat"].mean())
    ratio = 1.0 / max(math.cos(math.radians(mean_lat)), 1e-6)
    fig = (
        ggplot(plot_df, aes(x="lon360", y="lat", group="deployment_id", color="deployment_id"))
        + geom_path(alpha=0.45, size=0.4)
        + geom_point(alpha=0.7, size=0.35)
        + scale_color_brewer(type="qual", palette="Set1")
        + coord_fixed(ratio=ratio)
        + theme_bw()
        + theme(figure_size=(8, 6.5), dpi=300, legend_position="right")
        + labs(x="Longitude (0-360)", y="Latitude", color="Deployment", title=output_filename)
    )
    _save_plotnine_figure(fig, output_dir, output_filename)
    return fig


def plot_continuous_track_map(
    binned_df: pd.DataFrame,
    output_dir: str | Path,
    output_filename: str,
    value_label: str,
    covariate_overlay: Optional[Dict] = None,
    color_scale: Optional[str | list[str]] = None,
    point_size_col: Optional[str] = None,
    point_size_range: Optional[tuple[float, float] | list[float]] = None,
    point_alpha: float = 0.85,
):
    from plotnine import (
        aes,
        coord_fixed,
        geom_point,
        geom_raster,
        ggplot,
        labs,
        scale_color_gradientn,
        scale_fill_gradientn,
        scale_size_continuous,
        theme,
        theme_bw,
    )

    plot_df = binned_df.copy()
    plot_df["lon360"] = plot_df["lon"].to_numpy(dtype=float) % 360.0
    mean_lat = float(plot_df["lat"].mean())
    ratio = 1.0 / max(math.cos(math.radians(mean_lat)), 1e-6)

    fig = ggplot()
    if covariate_overlay and covariate_overlay["type"] in {"bathymetry", "bathymetry_ice"}:
        bathy_df = covariate_overlay.get("bathy_df")
        if bathy_df is not None and not bathy_df.empty:
            fig = fig + geom_raster(
                bathy_df,
                aes(x="lon360", y="lat", fill="bathy"),
                alpha=0.85,
            ) + scale_fill_gradientn(
                colors=["#01665e", "#c7eae5", "#FFFFFF", "#999999", "#636363", "#FCFCFC"],
                name="Depth (m)",
            )

    if isinstance(color_scale, str) and color_scale.strip().lower() == "magma":
        color_values = ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]
    elif isinstance(color_scale, (list, tuple)) and len(color_scale) >= 2:
        color_values = [str(x) for x in color_scale]
    else:
        color_values = ["#440154", "#31688E", "#35B779", "#FDE725"]

    if point_size_col and point_size_col in plot_df.columns:
        size_range = tuple(point_size_range or (1.5, 7.5))
        fig = (
            fig
            + geom_point(
                plot_df,
                aes(x="lon360", y="lat", color="map_value", size=point_size_col),
                alpha=float(point_alpha),
            )
            + scale_color_gradientn(colors=color_values, name=value_label)
            + scale_size_continuous(name=point_size_col, range=size_range)
        )
    else:
        fig = (
            fig
            + geom_point(
                plot_df,
                aes(x="lon360", y="lat", color="map_value"),
                alpha=float(point_alpha),
                size=1.2,
            )
            + scale_color_gradientn(colors=color_values, name=value_label)
        )

    fig = (
        fig
        + coord_fixed(ratio=ratio)
        + theme_bw()
        + theme(figure_size=(8, 6.5), dpi=300, legend_position="right")
        + labs(x="Longitude (0-360)", y="Latitude", title=output_filename)
    )
    _save_plotnine_figure(fig, output_dir, output_filename)
    return fig


def detect_lat_lon_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lat_candidates = ["latitude", "lat", "Latitude", "LAT", "Lat"]
    lon_candidates = ["longitude", "lon", "Longitude", "LON", "Lon", "Long"]
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find lat/lon columns. Available columns: {list(df.columns)}")
    return str(lat_col), str(lon_col)
