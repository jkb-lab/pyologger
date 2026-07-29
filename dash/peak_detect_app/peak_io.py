"""
peak_io.py — save/load data_pkl with backups and event_data utilities.

All writes go through save_datapkl() which creates a timestamped backup first.
Manual edits are stored with keys: heartbeat_manual_ok / heartbeat_manual_reject.
These are written into event_data and survive automated re-runs because the
workflow reads them before cleanup and never removes them.
"""

from __future__ import annotations

import os
import pickle
import shutil
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np


# ── backup / save ─────────────────────────────────────────────────────────────

def _outputs_dir(dataset: str, deployment: str, data_dir: str) -> str:
    return os.path.join(data_dir, dataset, deployment, "outputs")


def _pkl_path(dataset: str, deployment: str, data_dir: str) -> str:
    return os.path.join(_outputs_dir(dataset, deployment, data_dir), "data.pkl")


def load_datapkl(dataset: str, deployment: str, data_dir: str):
    path = _pkl_path(dataset, deployment, data_dir)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_datapkl(data_pkl, dataset: str, deployment: str, data_dir: str) -> str:
    """Write data_pkl to disk, creating a timestamped backup first.

    Returns the path of the backup created.
    """
    path = _pkl_path(dataset, deployment, data_dir)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = os.path.join(_outputs_dir(dataset, deployment, data_dir), "backups")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"data_{stamp}.pkl")
    if os.path.exists(path):
        shutil.copy2(path, backup_path)
    with open(path, "wb") as f:
        pickle.dump(data_pkl, f)
    return backup_path


# ── event_data helpers ────────────────────────────────────────────────────────

_AUTO_KEYS = {
    "heartbeat_auto_detect_accepted",
    "heartbeat_auto_detect_rejected",
    "heartbeat_auto_detect_suggested",
    "heartbeat_auto_detect_gap",
}

_MANUAL_KEYS = {
    "heartbeat_manual_ok",
    "heartbeat_manual_reject",
}


def ensure_event_data(data_pkl):
    """Make sure data_pkl.event_data is a proper DataFrame."""
    if not hasattr(data_pkl, "event_data") or data_pkl.event_data is None:
        data_pkl.event_data = pd.DataFrame(columns=[
            "type", "key", "value", "short_description", "long_description",
            "datetime", "datetime_utc", "duration", "time_unix_ms", "note",
            "date", "time",
        ])
    return data_pkl


def write_auto_peaks_to_events(data_pkl, peak_df: pd.DataFrame) -> None:
    """Replace all auto-detect heartbeat events with those in peak_df.

    Manual events are preserved untouched.
    """
    ensure_event_data(data_pkl)
    # Drop old auto events
    ev = data_pkl.event_data
    ev = ev[~ev["key"].isin(_AUTO_KEYS)].copy()

    key_map = {
        "beat_auto_detect_accepted": "heartbeat_auto_detect_accepted",
        "beat_auto_detect_rejected": "heartbeat_auto_detect_rejected",
        "beat_auto_detect_suggested": "heartbeat_auto_detect_suggested",
    }
    desc_map = {
        "heartbeat_auto_detect_accepted": "auto-detected heartbeat (accepted)",
        "heartbeat_auto_detect_rejected": "auto-detected heartbeat (rejected)",
        "heartbeat_auto_detect_suggested": "auto-detected heartbeat (suggested)",
    }

    rows = []
    for _, row in peak_df.iterrows():
        event_key = key_map.get(row.get("key", ""), None)
        if event_key is None:
            continue
        dt = pd.to_datetime(row.get("datetime"))
        rows.append({
            "type": "state",
            "key": event_key,
            "value": np.nan,
            "short_description": desc_map.get(event_key, event_key),
            "long_description": np.nan,
            "datetime": dt,
            "datetime_utc": dt.tz_convert("UTC") if dt.tzinfo else dt,
            "duration": 0.0,
            "time_unix_ms": pd.NA,
            "note": np.nan,
            "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "",
            "time": dt.strftime("%H:%M:%S.%f")[:-3] if pd.notna(dt) else "",
        })

    if rows:
        new_auto = pd.DataFrame(rows)
        ev = pd.concat([ev, new_auto], ignore_index=True)

    ev = ev.sort_values("datetime", na_position="last").reset_index(drop=True)
    data_pkl.event_data = ev


def apply_manual_edits(data_pkl, manual_edits: List[dict]) -> None:
    """Write manual_ok / manual_reject events into event_data.

    manual_edits: list of {datetime: str, action: 'ok'|'reject'}
    """
    ensure_event_data(data_pkl)
    ev = data_pkl.event_data
    # Drop existing manual entries for these exact datetimes
    edit_dts = {pd.to_datetime(e["datetime"]) for e in manual_edits}

    def _is_manual_at_dt(row):
        return row["key"] in _MANUAL_KEYS and pd.to_datetime(row["datetime"]) in edit_dts

    ev = ev[~ev.apply(_is_manual_at_dt, axis=1)].copy()

    rows = []
    for edit in manual_edits:
        dt = pd.to_datetime(edit["datetime"])
        key = "heartbeat_manual_ok" if edit["action"] == "ok" else "heartbeat_manual_reject"
        rows.append({
            "type": "state",
            "key": key,
            "value": np.nan,
            "short_description": f"manually marked {edit['action']}",
            "long_description": np.nan,
            "datetime": dt,
            "datetime_utc": dt.tz_convert("UTC") if dt.tzinfo else dt,
            "duration": 0.0,
            "time_unix_ms": pd.NA,
            "note": np.nan,
            "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "",
            "time": dt.strftime("%H:%M:%S.%f")[:-3] if pd.notna(dt) else "",
        })

    if rows:
        ev = pd.concat([ev, pd.DataFrame(rows)], ignore_index=True)
    ev = ev.sort_values("datetime", na_position="last").reset_index(drop=True)
    data_pkl.event_data = ev


def count_manual_edits(data_pkl) -> dict:
    """Return {ok: N, reject: N} counts of existing manual edits."""
    ensure_event_data(data_pkl)
    ev = data_pkl.event_data
    ok_n = len(ev[ev["key"] == "heartbeat_manual_ok"])
    rej_n = len(ev[ev["key"] == "heartbeat_manual_reject"])
    return {"ok": ok_n, "reject": rej_n}


def list_backups(dataset: str, deployment: str, data_dir: str) -> list[str]:
    """Return sorted list of backup pkl filenames (newest first)."""
    backup_dir = os.path.join(_outputs_dir(dataset, deployment, data_dir), "backups")
    if not os.path.exists(backup_dir):
        return []
    files = sorted(
        [f for f in os.listdir(backup_dir) if f.endswith(".pkl")],
        reverse=True,
    )
    return files
