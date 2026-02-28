import os
import pathlib
import re
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd


_MODEL_INFO_CACHE = {}
_ANIMAL_ID_PATTERN = re.compile(r"([a-z]{4}-\d{3}[a-z]?)", re.IGNORECASE)


def _build_empty_orientation_json():
    empty = pd.DataFrame({"datetime": [], "pitch": [], "roll": [], "heading": []}).set_index("datetime")
    return empty.to_json(orient="split", date_format="iso")


EMPTY_ORIENTATION_JSON = _build_empty_orientation_json()


def _extract_animal_id(value):
    text = str(value or "").strip()
    if not text:
        return None
    if _ANIMAL_ID_PATTERN.fullmatch(text):
        return text.lower()
    match = _ANIMAL_ID_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return None


def infer_animal_id(data_pkl_obj, deployment_id_fallback=None):
    info = getattr(data_pkl_obj, "deployment_info", {}) or {}
    for key in ("animal_id", "Animal ID", "animal", "Animal", "animalid", "animalId"):
        value = _extract_animal_id(info.get(key))
        if value:
            return value
    for key in ("deployment_id", "deployment", "Name", "name", "recording_id"):
        value = _extract_animal_id(info.get(key))
        if value:
            return value
    if deployment_id_fallback:
        value = _extract_animal_id(deployment_id_fallback)
        if value:
            return value
    return None


def build_orientation_data_json(data_pkl_obj):
    prh_df = (getattr(data_pkl_obj, "signal_data", {}) or {}).get("prh")
    if prh_df is None or prh_df.empty:
        return {
            "ok": False,
            "message": "No PRH signal found; using empty orientation stream.",
            "data_json": EMPTY_ORIENTATION_JSON,
        }

    cols = {str(c): c for c in prh_df.columns}
    pitch_col = cols.get("pitch")
    roll_col = cols.get("roll")
    heading_col = cols.get("heading") or cols.get("heading2") or cols.get("head")

    if pitch_col is None or roll_col is None:
        return {
            "ok": False,
            "message": "PRH is missing pitch/roll columns; using empty orientation stream.",
            "data_json": EMPTY_ORIENTATION_JSON,
        }

    if "datetime" not in prh_df.columns:
        return {
            "ok": False,
            "message": "PRH is missing datetime; using empty orientation stream.",
            "data_json": EMPTY_ORIENTATION_JSON,
        }

    heading_series = (
        pd.to_numeric(prh_df[heading_col], errors="coerce")
        if heading_col is not None
        else pd.Series(0.0, index=prh_df.index, dtype="float64")
    )
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(prh_df["datetime"], errors="coerce", utc=True),
            "pitch": pd.to_numeric(prh_df[pitch_col], errors="coerce"),
            "roll": pd.to_numeric(prh_df[roll_col], errors="coerce"),
            "heading": heading_series,
        }
    ).dropna(subset=["datetime"])

    if frame.empty:
        return {
            "ok": False,
            "message": "PRH datetime parsing produced no rows; using empty orientation stream.",
            "data_json": EMPTY_ORIENTATION_JSON,
        }

    frame = frame.sort_values("datetime").set_index("datetime")
    frame = _augment_orientation_with_track_columns(data_pkl_obj, frame)
    message = ""
    if heading_col is None:
        message = "PRH heading column not found; using heading=0 while applying pitch/roll."

    return {
        "ok": True,
        "message": message,
        "data_json": frame.to_json(orient="split", date_format="iso"),
    }


def _first_existing_signal(data_pkl_obj, signal_names):
    signal_data = getattr(data_pkl_obj, "signal_data", {}) or {}
    for name in signal_names:
        df = signal_data.get(name)
        if df is not None and not df.empty:
            return df
    return None


def _extract_location_for_track(data_pkl_obj):
    loc_df = _first_existing_signal(data_pkl_obj, ["location", "gps", "track"])
    if loc_df is None or loc_df.empty:
        return pd.DataFrame(columns=["datetime", "lat", "lon"])
    if "datetime" not in loc_df.columns:
        return pd.DataFrame(columns=["datetime", "lat", "lon"])
    cols = {str(c).lower(): c for c in loc_df.columns}
    lat_col = None
    lon_col = None
    for candidate in ("latitude", "lat", "gps_0"):
        lat_col = cols.get(candidate)
        if lat_col is not None:
            break
    for candidate in ("longitude", "lon", "gps_1"):
        lon_col = cols.get(candidate)
        if lon_col is not None:
            break
    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=["datetime", "lat", "lon"])
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(loc_df["datetime"], errors="coerce", utc=True),
            "lat": pd.to_numeric(loc_df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(loc_df[lon_col], errors="coerce"),
        }
    ).dropna(subset=["datetime", "lat", "lon"])
    if out.empty:
        return pd.DataFrame(columns=["datetime", "lat", "lon"])
    out = out[(out["lat"] >= -90.0) & (out["lat"] <= 90.0) & (out["lon"] >= -180.0) & (out["lon"] <= 180.0)]
    return out.sort_values("datetime")


def _extract_depth_for_track(data_pkl_obj):
    dep_df = _first_existing_signal(data_pkl_obj, ["corrected_depth", "depth", "pressure", "odba"])
    if dep_df is None or dep_df.empty:
        return pd.DataFrame(columns=["datetime", "depth"])
    if "datetime" not in dep_df.columns:
        return pd.DataFrame(columns=["datetime", "depth"])
    value_cols = [c for c in dep_df.columns if str(c).lower() != "datetime"]
    if not value_cols:
        return pd.DataFrame(columns=["datetime", "depth"])
    depth_col = value_cols[0]
    for c in value_cols:
        lc = str(c).lower()
        if "depth" in lc or lc == "p":
            depth_col = c
            break
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(dep_df["datetime"], errors="coerce", utc=True),
            "depth": pd.to_numeric(dep_df[depth_col], errors="coerce"),
        }
    ).dropna(subset=["datetime", "depth"])
    if out.empty:
        return pd.DataFrame(columns=["datetime", "depth"])
    return out.sort_values("datetime")


def _extract_stroke_rate_for_track(data_pkl_obj):
    sr_df = _first_existing_signal(data_pkl_obj, ["stroke_rate", "sr"])
    if sr_df is None or sr_df.empty:
        return pd.DataFrame(columns=["datetime", "stroke_rate"])
    if "datetime" not in sr_df.columns:
        return pd.DataFrame(columns=["datetime", "stroke_rate"])
    value_cols = [c for c in sr_df.columns if str(c).lower() != "datetime"]
    if not value_cols:
        return pd.DataFrame(columns=["datetime", "stroke_rate"])
    sr_col = value_cols[0]
    for c in value_cols:
        lc = str(c).lower()
        if lc in {"stroke_rate", "sr", "spm"}:
            sr_col = c
            break
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(sr_df["datetime"], errors="coerce", utc=True),
            "stroke_rate": pd.to_numeric(sr_df[sr_col], errors="coerce"),
        }
    ).dropna(subset=["datetime", "stroke_rate"])
    if out.empty:
        return pd.DataFrame(columns=["datetime", "stroke_rate"])
    return out.sort_values("datetime")


def _augment_orientation_with_track_columns(data_pkl_obj, orientation_df):
    if orientation_df is None or orientation_df.empty:
        return orientation_df
    if "datetime" in orientation_df.columns:
        orient = orientation_df.copy()
    else:
        orient = orientation_df.reset_index().copy()
    if "datetime" not in orient.columns:
        return orientation_df
    orient["datetime"] = pd.to_datetime(orient["datetime"], errors="coerce", utc=True)
    orient = orient.dropna(subset=["datetime"]).sort_values("datetime")
    if orient.empty:
        return orientation_df

    loc = _extract_location_for_track(data_pkl_obj)
    if not loc.empty:
        try:
            orient = pd.merge_asof(
                orient.sort_values("datetime"),
                loc[["datetime", "lat", "lon"]].sort_values("datetime"),
                on="datetime",
                direction="nearest",
            )
        except Exception:
            pass

    dep = _extract_depth_for_track(data_pkl_obj)
    if not dep.empty:
        try:
            orient = pd.merge_asof(
                orient.sort_values("datetime"),
                dep[["datetime", "depth"]].sort_values("datetime"),
                on="datetime",
                direction="nearest",
            )
            vals = pd.to_numeric(orient.get("depth"), errors="coerce").dropna()
            if not vals.empty:
                d05 = float(vals.quantile(0.05))
                d95 = float(vals.quantile(0.95))
                span = max(1e-9, d95 - d05)
                norm = (pd.to_numeric(orient["depth"], errors="coerce") - d05) / span
                norm = norm.clip(lower=0.0, upper=1.0)
                top_offset = 23.0
                bottom_offset = -23.0
                orient["y_depth"] = top_offset - norm * (top_offset - bottom_offset)
        except Exception:
            pass

    sr = _extract_stroke_rate_for_track(data_pkl_obj)
    if not sr.empty:
        try:
            orient = pd.merge_asof(
                orient.sort_values("datetime"),
                sr[["datetime", "stroke_rate"]].sort_values("datetime"),
                on="datetime",
                direction="nearest",
            )
        except Exception:
            pass

    # Fallback track when GPS is unavailable:
    # keep depth-driven Y movement and synthesize a forward path.
    # Move forward at 1 m/s when:
    # - stroke_rate > 10 spm, OR
    # - stroke_rate is NA and |d(depth)/dt| > 0.05 m/s
    # Otherwise do not advance forward.
    try:
        has_latlon = ("lat" in orient.columns) and ("lon" in orient.columns)
        depth_vals = pd.to_numeric(orient.get("depth"), errors="coerce")
        has_depth = depth_vals.notna().any()
        if (not has_latlon) and has_depth and len(orient) > 0:
            dt_s = pd.to_numeric(
                orient["datetime"].diff().dt.total_seconds(),
                errors="coerce",
            ).fillna(0.0)
            dt_s = dt_s.clip(lower=0.0)
            if dt_s.sum() <= 0.0:
                dt_s = pd.Series(1.0, index=orient.index, dtype="float64")

            stroke = pd.to_numeric(orient.get("stroke_rate"), errors="coerce")
            stroke_valid = stroke.notna()
            moving_by_stroke = stroke.fillna(0.0) > 10.0

            depth_speed = (depth_vals.diff() / dt_s.replace(0.0, pd.NA)).abs()
            depth_speed = pd.to_numeric(depth_speed, errors="coerce").fillna(0.0)
            moving_by_depth = depth_speed > 0.05

            moving = moving_by_stroke.where(stroke_valid, moving_by_depth)
            forward_speed_ms = 1.0
            forward_step_m = dt_s * forward_speed_ms * moving.astype("float64")
            forward_dist_m = forward_step_m.cumsum()
            orient["lat"] = 0.0
            orient["lon"] = forward_dist_m.astype("float64")
    except Exception:
        pass

    orient = orient.set_index("datetime")
    return orient


def fetch_3d_model_info(animal_id):
    animal_key = str(animal_id or "").strip()
    if not animal_key:
        return {"ok": False, "message": "No animal ID available for this deployment."}
    cached = _MODEL_INFO_CACHE.get(animal_key)
    if cached and cached.get("ok"):
        model_url = str(((cached.get("model") or {}).get("model_url")) or "")
        if model_url and not _is_presigned_url_expired(model_url):
            return cached
    elif cached and not cached.get("ok"):
        # Do not pin failed lookups forever; retry on next request.
        cached = None

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=False)
        load_dotenv(repo_root / "DiveDB" / ".env", override=False)
    except Exception:
        pass

    notion_token = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_API_KEY")
    asset_db = os.getenv("NOTION_ASSETS_DB") or os.getenv("NOTION_ASSET_DB")
    db_map = {
        "Deployment DB": os.getenv("NOTION_DEPLOYMENT_DB"),
        "Recording DB": os.getenv("NOTION_RECORDING_DB"),
        "Logger DB": os.getenv("NOTION_LOGGER_DB"),
        "Animal DB": os.getenv("NOTION_ANIMAL_DB"),
        "Species DB": os.getenv("NOTION_SPECIES_DB"),
        "Asset DB": asset_db,
        "Dataset DB": os.getenv("NOTION_DATASET_DB"),
        "Signal DB": os.getenv("NOTION_SIGNAL_DB"),
        "Standardized Channel DB": os.getenv("NOTION_STANDARDIZEDCHANNEL_DB"),
    }
    missing = [k for k, v in db_map.items() if not v]
    if not notion_token or missing:
        result = {
            "ok": False,
            "message": "Missing Notion environment variables for DiveDB model lookup.",
            "details": f"missing token={not bool(notion_token)}; missing DB keys={missing}",
        }
        _MODEL_INFO_CACHE[animal_key] = result
        return result

    divedb_root = repo_root / "DiveDB"
    if str(divedb_root) not in sys.path:
        sys.path.insert(0, str(divedb_root))

    try:
        from DiveDB.services.duck_pond import DuckPond
        from DiveDB.services.notion_orm import NotionORMManager
    except Exception as e:
        result = {
            "ok": False,
            "message": "DiveDB imports failed in current runtime.",
            "details": f"{type(e).__name__}: {e}",
        }
        _MODEL_INFO_CACHE[animal_key] = result
        return result

    try:
        notion_manager = NotionORMManager(token=notion_token, db_map=db_map)
        duck_pond = DuckPond.from_environment(notion_manager=notion_manager)
        model = duck_pond.get_3d_model_for_animal(animal_key, use_cache=True) or {}
        model_url = model.get("model_url")
        result = {
            "ok": bool(model_url),
            "message": ("3D model found." if model_url else "No 3D model returned for this animal."),
            "animal_id": animal_key,
            "model": model,
        }
    except Exception as e:
        result = {
            "ok": False,
            "message": "3D model lookup failed (likely network/API/runtime issue).",
            "details": f"{type(e).__name__}: {e}",
            "animal_id": animal_key,
        }

    _MODEL_INFO_CACHE[animal_key] = result
    return result


def _is_presigned_url_expired(url, skew_seconds=60):
    """Return True when an S3-style presigned URL is expired or about to expire."""
    try:
        query = parse_qs(urlparse(str(url)).query or "")
        date_raw = (query.get("X-Amz-Date") or query.get("x-amz-date") or [None])[0]
        expires_raw = (query.get("X-Amz-Expires") or query.get("x-amz-expires") or [None])[0]
        if not date_raw or not expires_raw:
            return False
        signed_at = datetime.strptime(date_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        expires_at = signed_at + timedelta(seconds=int(expires_raw))
        return datetime.now(timezone.utc) >= (expires_at - timedelta(seconds=max(0, int(skew_seconds))))
    except Exception:
        return False
