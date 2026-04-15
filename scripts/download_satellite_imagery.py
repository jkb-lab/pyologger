"""
Download Sentinel-2 imagery for any DEM extent
===============================================
Generalizable approach:
  1. Search all tiles intersecting the DEM bbox
  2. Group results by acquisition date
  3. For each date, estimate combined spatial coverage of the DEM
  4. Pick the date with best coverage + lowest cloud, download all its tiles
  5. Warp each tile to DEM extent and mosaic into one texture

Run in terminal (not Blender):
  python3 download_sentinel2_monterey.py

Requirements:
  pip install rasterio numpy pystac-client planetary-computer shapely
"""

import os
import numpy as np
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEM_PATH   = "/Users/jessiekb/Downloads/monterey_bay_DEM_V2.tiff"
OUTPUT_DIR = "/Users/jessiekb/Downloads/sentinel2_monterey"
OUTPUT_TEX = "/Users/jessiekb/Downloads/sentinel2_monterey/monterey_rgb_utm.tif"
TARGET_CRS = "EPSG:32610"
MAX_CLOUD  = 20           # max per-tile cloud %
DATE_RANGE = "2022-05-01/2023-09-30"
MIN_COVERAGE = 0.90       # require 90% of DEM bbox covered
OUT_RES    = 10           # output pixel size in metres (Sentinel-2 native)
# ──────────────────────────────────────────────────────────────────────────────


def get_dem_info(dem_path, target_crs):
    import rasterio
    from rasterio.warp import calculate_default_transform, transform_bounds
    from rasterio.crs import CRS
    from shapely.geometry import box

    dst_crs = CRS.from_string(target_crs)
    with rasterio.open(dem_path) as src:
        bbox_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        t, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

    left, top = t.c, t.f
    right  = left + t.a * w
    bottom = top  + t.e * h
    utm_bounds = (left, bottom, right, top)

    print(f"  WGS84 bbox : {[round(x, 4) for x in bbox_wgs84]}")
    print(f"  UTM bounds : {left:.0f}, {bottom:.0f}, {right:.0f}, {top:.0f}")
    print(f"  UTM extent : {(right-left)/1000:.1f} x {(top-bottom)/1000:.1f} km")

    dem_shape_wgs84 = box(*bbox_wgs84)
    return bbox_wgs84, utm_bounds, dem_shape_wgs84


def search_all_tiles(bbox_wgs84, date_range, max_cloud):
    """Return all Sentinel-2 items intersecting the bbox, no limit."""
    from pystac_client import Client
    import planetary_computer as pc
    from shapely.geometry import box

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    results = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=box(*bbox_wgs84),
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=200,
    )
    items = list(results.items())
    print(f"  Found {len(items)} total scenes within bbox and cloud threshold")
    return items


def group_by_date(items):
    """
    Group items by acquisition date (YYYY-MM-DD).
    Items from the same orbit pass share the same date — these are the tiles
    we want to mosaic together.
    """
    groups = defaultdict(list)
    for item in items:
        dt = item.properties.get("datetime", "")[:10]
        groups[dt].append(item)
    return groups


def coverage_fraction(items, dem_shape_wgs84):
    """
    Estimate what fraction of the DEM bbox is covered by the union
    of all tile footprints on this date.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    footprints = []
    for item in items:
        geom = item.geometry
        if geom:
            footprints.append(shape(geom))

    if not footprints:
        return 0.0

    union    = unary_union(footprints)
    covered  = dem_shape_wgs84.intersection(union).area
    total    = dem_shape_wgs84.area
    return covered / total if total > 0 else 0.0


def pick_best_date(groups, dem_shape_wgs84, min_coverage):
    """
    Score each date by coverage fraction and mean cloud cover.
    Return the date + items that best covers the DEM with least cloud.
    """
    print(f"\n  Evaluating {len(groups)} dates for coverage >= {min_coverage*100:.0f}%...")

    candidates = []
    for date, items in sorted(groups.items()):
        cov  = coverage_fraction(items, dem_shape_wgs84)
        mean_cloud = np.mean([i.properties.get("eo:cloud_cover", 99) for i in items])
        if cov >= min_coverage:
            candidates.append((date, items, cov, mean_cloud))
            print(f"    {date}  tiles={len(items)}  coverage={cov*100:.0f}%  "
                  f"cloud={mean_cloud:.1f}%  ✓")
        else:
            print(f"    {date}  tiles={len(items)}  coverage={cov*100:.0f}%  "
                  f"cloud={mean_cloud:.1f}%")

    if not candidates:
        return None, None

    # Sort by cloud cover ascending (coverage already >= threshold for all)
    candidates.sort(key=lambda x: x[3])
    best_date, best_items, best_cov, best_cloud = candidates[0]
    print(f"\n  → Best date: {best_date}  coverage={best_cov*100:.0f}%  "
          f"cloud={best_cloud:.1f}%  ({len(best_items)} tiles)")
    return best_date, best_items


def warp_band(asset_href, utm_bounds, target_crs):
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    dst_crs = CRS.from_string(target_crs)
    l, b, r, t = utm_bounds
    out_w = int(round((r - l) / OUT_RES))
    out_h = int(round((t - b) / OUT_RES))
    dst_t = from_bounds(l, b, r, t, out_w, out_h)

    dst_arr = np.zeros((out_h, out_w), dtype=np.uint16)
    with rasterio.open(asset_href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_t,
            dst_crs=dst_crs,
            resampling=Resampling.lanczos,
            dst_nodata=0,
        )
    return dst_arr, dst_t, out_w, out_h


def process_item(item, utm_bounds, target_crs, out_path):
    import rasterio
    from rasterio.crs import CRS
    import planetary_computer as pc

    signed  = pc.sign(item)
    dst_crs = CRS.from_string(target_crs)
    arrays  = []
    dst_t = out_w = out_h = None

    for band in ["B04", "B03", "B02"]:
        asset = signed.assets.get(band)
        if asset is None:
            raise ValueError(f"Band {band} missing from {item.id}")
        print(f"    {band}...", end=" ", flush=True)
        arr, dst_t, out_w, out_h = warp_band(asset.href, utm_bounds, target_crs)
        arrays.append(arr)
        print("ok")

    rgb    = np.stack(arrays, axis=0)
    rgb_u8 = (np.clip(rgb, 0, 3500).astype(np.float32) / 3500.0 * 255).astype(np.uint8)
    coverage = (rgb_u8[0] > 0).mean()
    print(f"    Tile coverage of DEM extent: {coverage*100:.0f}%")

    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": out_w, "height": out_h, "count": 3,
        "crs": dst_crs, "transform": dst_t,
        "compress": "deflate", "nodata": 0,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb_u8)
    return out_path


def mosaic(tile_paths, output_path):
    import rasterio
    from rasterio.merge import merge

    if len(tile_paths) == 1:
        import shutil
        shutil.copy(tile_paths[0], output_path)
        return

    print(f"  Mosaicking {len(tile_paths)} tiles...")
    datasets = [rasterio.open(p) for p in tile_paths]
    arr, out_transform = merge(datasets, method="first")
    profile = datasets[0].profile.copy()
    profile.update({
        "height": arr.shape[1], "width": arr.shape[2],
        "transform": out_transform, "compress": "deflate",
    })
    for ds in datasets:
        ds.close()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading DEM extent...")
    bbox_wgs84, utm_bounds, dem_shape = get_dem_info(DEM_PATH, TARGET_CRS)

    print(f"\nSearching Sentinel-2 (cloud < {MAX_CLOUD}%, {DATE_RANGE})...")
    items = search_all_tiles(bbox_wgs84, DATE_RANGE, MAX_CLOUD)

    if not items:
        print("No items found. Try increasing MAX_CLOUD or widening DATE_RANGE.")
        return

    groups     = group_by_date(items)
    best_date, best_items = pick_best_date(groups, dem_shape, MIN_COVERAGE)

    if best_date is None:
        print(f"\nNo single date reaches {MIN_COVERAGE*100:.0f}% coverage.")
        print("Try lowering MIN_COVERAGE or widening DATE_RANGE.")
        return

    print(f"\nDownloading {len(best_items)} tiles for {best_date}...")
    tile_paths = []
    for i, item in enumerate(best_items):
        # Use the UTM grid tile ID (e.g. T10SFG) as the temp filename
        tile_id  = next(
            (p for p in item.id.split("_") if p.startswith("T") and len(p) == 6),
            f"tile{i}"
        )
        tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{tile_id}.tif")
        print(f"\n  [{i+1}/{len(best_items)}] {item.id[:55]}  tile={tile_id}")
        process_item(item, utm_bounds, TARGET_CRS, tmp_path)
        tile_paths.append(tmp_path)

    print("\nMosaicking...")
    mosaic(tile_paths, OUTPUT_TEX)

    for p in tile_paths:
        if os.path.exists(p) and p != OUTPUT_TEX:
            os.remove(p)

    import rasterio
    with rasterio.open(OUTPUT_TEX) as ds:
        print(f"\nFinal texture:")
        print(f"  Bounds : {ds.bounds}")
        print(f"  Shape  : {ds.shape}  "
              f"({ds.width * OUT_RES / 1000:.1f} x {ds.height * OUT_RES / 1000:.1f} km at {OUT_RES}m/px)")
    print(f"\nDone → {OUTPUT_TEX}")
    print("Now run monterey_dem_blender_textured.py in Blender.")


if __name__ == "__main__":
    main()