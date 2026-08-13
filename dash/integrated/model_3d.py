import os
import pathlib
import re
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd
import numpy as np


_MODEL_INFO_CACHE = {}
_ANIMAL_ID_PATTERN = re.compile(r"([a-z]{4}-\d{3}[a-z]?)", re.IGNORECASE)
_TRACK_MERGE_TOLERANCE = pd.Timedelta(seconds=5)
_ORIENTATION_MAX_RAW_ROWS = int(os.getenv("PYOLOGGER_ORIENTATION_MAX_RAW_ROWS", "15000"))
_ORIENTATION_MAX_OUTPUT_ROWS = int(os.getenv("PYOLOGGER_ORIENTATION_MAX_OUTPUT_ROWS", "5000"))

_ORIENTATION_SIGNAL_ALIASES = {
    "prh",
    "orientation",
    "attitude",
}

_ORIENTATION_COLUMN_ALIASES = {
    "pitch": ("pitch", "pitchdeg", "pitchdegrees", "prhpitch"),
    "roll": ("roll", "rolldeg", "rolldegrees", "prhroll"),
    "heading": (
        "heading",
        "heading2",
        "head",
        "yaw",
        "yawdeg",
        "yawdegrees",
        "prhheading",
    ),
}

_DATETIME_COLUMN_ALIASES = ("datetime", "timestamp", "time")


def _build_empty_orientation_json():
    empty = pd.DataFrame({"datetime": [], "pitch": [], "roll": [], "heading": []}).set_index("datetime")
    return empty.to_json(orient="split", date_format="iso")


EMPTY_ORIENTATION_JSON = _build_empty_orientation_json()


def _downsample_frame_preserve_ends(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame
    max_rows = int(max_rows or 0)
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    idx = np.linspace(0, len(frame) - 1, num=max_rows, dtype=int)
    idx = np.unique(idx)
    return frame.iloc[idx].copy()


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


def _normalize_token(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _first_matching_column(df, aliases):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    alias_tokens = {_normalize_token(a) for a in aliases}
    for col in df.columns:
        if _normalize_token(col) in alias_tokens:
            return col
    return None


def _first_matching_signal_df(signal_data, aliases):
    name, frame = _first_matching_signal(signal_data, aliases)
    if name is None:
        return None
    return frame


def _first_matching_signal(signal_data, aliases):
    if not isinstance(signal_data, dict):
        return None, None
    alias_tokens = {_normalize_token(a) for a in aliases}
    for key, value in signal_data.items():
        if _normalize_token(key) not in alias_tokens:
            continue
        if isinstance(value, pd.DataFrame) and not value.empty:
            return str(key), value
    return None, None


def _extract_scalar_signal(signal_data, signal_aliases, value_aliases):
    frame = _first_matching_signal_df(signal_data, signal_aliases)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    datetime_col = _first_matching_column(frame, _DATETIME_COLUMN_ALIASES)
    if datetime_col is None:
        return None

    value_col = _first_matching_column(frame, value_aliases)
    if value_col is None:
        non_time_cols = [c for c in frame.columns if c != datetime_col]
        if len(non_time_cols) != 1:
            return None
        value_col = non_time_cols[0]

    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(frame[datetime_col], errors="coerce", utc=True),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
        }
    ).dropna(subset=["datetime", "value"])
    if out.empty:
        return None
    return out.sort_values("datetime").reset_index(drop=True)


def _build_orientation_frame_from_grouped_signal(prh_df):
    if not isinstance(prh_df, pd.DataFrame) or prh_df.empty:
        return None, {"status": "missing_grouped_signal"}

    datetime_col = _first_matching_column(prh_df, _DATETIME_COLUMN_ALIASES)
    pitch_col = _first_matching_column(prh_df, _ORIENTATION_COLUMN_ALIASES["pitch"])
    roll_col = _first_matching_column(prh_df, _ORIENTATION_COLUMN_ALIASES["roll"])
    heading_col = _first_matching_column(prh_df, _ORIENTATION_COLUMN_ALIASES["heading"])

    if datetime_col is None or pitch_col is None or roll_col is None:
        return None, {
            "status": "missing_grouped_columns",
            "datetime_col": datetime_col,
            "pitch_col": pitch_col,
            "roll_col": roll_col,
            "heading_col": heading_col,
        }

    heading_series = (
        pd.to_numeric(prh_df[heading_col], errors="coerce")
        if heading_col is not None
        else pd.Series(0.0, index=prh_df.index, dtype="float64")
    )
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(prh_df[datetime_col], errors="coerce", utc=True),
            "pitch": pd.to_numeric(prh_df[pitch_col], errors="coerce"),
            "roll": pd.to_numeric(prh_df[roll_col], errors="coerce"),
            "heading": heading_series,
        }
    ).dropna(subset=["datetime", "pitch", "roll"])
    if frame.empty:
        return None, {
            "status": "grouped_frame_empty_after_parse",
            "datetime_col": datetime_col,
            "pitch_col": pitch_col,
            "roll_col": roll_col,
            "heading_col": heading_col,
        }
    return frame.sort_values("datetime").reset_index(drop=True), {
        "status": "ok",
        "datetime_col": str(datetime_col),
        "pitch_col": str(pitch_col),
        "roll_col": str(roll_col),
        "heading_col": (str(heading_col) if heading_col is not None else None),
        "heading_fallback_zero": bool(heading_col is None),
    }


def _build_orientation_frame_from_split_signals(signal_data):
    pitch_name, _ = _first_matching_signal(signal_data, ("pitch",))
    roll_name, _ = _first_matching_signal(signal_data, ("roll",))
    heading_name, _ = _first_matching_signal(signal_data, ("heading", "heading2", "head", "yaw"))

    pitch = _extract_scalar_signal(signal_data, ("pitch",), _ORIENTATION_COLUMN_ALIASES["pitch"])
    roll = _extract_scalar_signal(signal_data, ("roll",), _ORIENTATION_COLUMN_ALIASES["roll"])
    heading = _extract_scalar_signal(
        signal_data,
        ("heading", "heading2", "head", "yaw"),
        _ORIENTATION_COLUMN_ALIASES["heading"],
    )

    if pitch is None or roll is None:
        return None, {
            "status": "split_missing_pitch_or_roll",
            "pitch_signal": pitch_name,
            "roll_signal": roll_name,
            "heading_signal": heading_name,
        }

    merged = pd.merge_asof(
        pitch.rename(columns={"value": "pitch"}).sort_values("datetime"),
        roll.rename(columns={"value": "roll"}).sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1),
    ).dropna(subset=["pitch", "roll"])
    if merged.empty:
        return None, {
            "status": "split_merge_empty",
            "pitch_signal": pitch_name,
            "roll_signal": roll_name,
            "heading_signal": heading_name,
        }

    if heading is None:
        merged["heading"] = 0.0
    else:
        merged = pd.merge_asof(
            merged.sort_values("datetime"),
            heading.rename(columns={"value": "heading"}).sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=1),
        )
        merged["heading"] = pd.to_numeric(merged.get("heading"), errors="coerce").fillna(0.0)

    return merged.sort_values("datetime").reset_index(drop=True), {
        "status": "ok",
        "pitch_signal": pitch_name,
        "roll_signal": roll_name,
        "heading_signal": heading_name,
        "heading_fallback_zero": bool(heading is None),
    }


def build_orientation_data_json(data_pkl_obj):
    signal_data = getattr(data_pkl_obj, "signal_data", {}) or {}
    grouped_name, prh_df = _first_matching_signal(signal_data, _ORIENTATION_SIGNAL_ALIASES)
    frame, grouped_meta = _build_orientation_frame_from_grouped_signal(prh_df)

    debug = {
        "source": "none",
        "grouped_signal": grouped_name,
        "grouped_meta": grouped_meta,
        "available_signals": sorted([str(k) for k in signal_data.keys()]),
    }

    source_msg = ""
    if frame is None:
        frame, split_meta = _build_orientation_frame_from_split_signals(signal_data)
        debug["source"] = "split"
        debug["split_meta"] = split_meta
        source_msg = "Orientation assembled from split pitch/roll/heading signals."
    else:
        debug["source"] = "grouped"

    if frame is None or frame.empty:
        debug["status"] = "empty"
        return {
            "ok": False,
            "message": "No usable orientation data found; using empty orientation stream.",
            "data_json": EMPTY_ORIENTATION_JSON,
            "debug": debug,
        }

    frame = frame.sort_values("datetime").reset_index(drop=True)
    raw_rows_before_cap = int(len(frame))
    frame = _downsample_frame_preserve_ends(frame, _ORIENTATION_MAX_RAW_ROWS)
    raw_rows_after_cap = int(len(frame))

    frame = frame.set_index("datetime")
    frame = _augment_orientation_with_track_columns(data_pkl_obj, frame)
    output_rows_before_cap = int(len(frame))
    frame = _downsample_frame_preserve_ends(frame.reset_index(), _ORIENTATION_MAX_OUTPUT_ROWS).set_index("datetime")
    output_rows_after_cap = int(len(frame))

    message = source_msg
    heading_all_missing = "heading" in frame.columns and not pd.to_numeric(frame["heading"], errors="coerce").notna().any()
    if heading_all_missing:
        message = "PRH heading column not found; using heading=0 while applying pitch/roll."
    if output_rows_after_cap < output_rows_before_cap:
        ds_msg = (
            f"Orientation stream downsampled {output_rows_before_cap:,} -> {output_rows_after_cap:,} rows "
            "for browser stability."
        )
        message = f"{message} {ds_msg}".strip()

    debug["status"] = "ok"
    debug["rows"] = int(len(frame))
    debug["raw_rows_before_cap"] = raw_rows_before_cap
    debug["raw_rows_after_cap"] = raw_rows_after_cap
    debug["output_rows_before_cap"] = output_rows_before_cap
    debug["output_rows_after_cap"] = output_rows_after_cap
    debug["raw_row_cap"] = int(_ORIENTATION_MAX_RAW_ROWS)
    debug["output_row_cap"] = int(_ORIENTATION_MAX_OUTPUT_ROWS)
    debug["columns"] = [str(c) for c in frame.columns]
    try:
        debug["start"] = frame.index.min().isoformat()
        debug["end"] = frame.index.max().isoformat()
    except Exception:
        pass
    debug["heading_all_missing"] = bool(heading_all_missing)

    return {
        "ok": True,
        "message": message,
        "data_json": frame.to_json(orient="split", date_format="iso"),
        "debug": debug,
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
                tolerance=_TRACK_MERGE_TOLERANCE,
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
                tolerance=_TRACK_MERGE_TOLERANCE,
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
                tolerance=_TRACK_MERGE_TOLERANCE,
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

    notion_token = os.getenv("NOTION_TOKEN")
    asset_db = os.getenv("NOTION_DB_ASSET") or os.getenv("NOTION_ASSET_DB")
    db_map = {
        "Deployment DB": os.getenv("NOTION_DB_DEPLOYMENT"),
        "Recording DB": os.getenv("NOTION_DB_RECORDING"),
        "Logger DB": os.getenv("NOTION_DB_LOGGER"),
        "Animal DB": os.getenv("NOTION_DB_ORGANISM"),
        "Species DB": os.getenv("NOTION_DB_SPECIES"),
        "Asset DB": asset_db,
        "Dataset DB": os.getenv("NOTION_DB_DATASET"),
        "Signal DB": os.getenv("NOTION_DB_SIGNAL"),
        "Standardized Channel DB": os.getenv("NOTION_DB_STANDARDIZED_CHANNEL"),
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
