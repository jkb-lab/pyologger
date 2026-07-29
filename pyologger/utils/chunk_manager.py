"""
chunk_manager.py — utilities for dividing a deployment into validated time chunks.

No Streamlit or Dash imports — importable from workflows and both UI layers.

Chunk grid lifecycle:
  1. compute_chunk_grid()  — build the grid from logger_attachments (or fallback window)
  2. load_chunks()         — read current grid from parameter_log (None if absent)
  3. save_chunk_status()   — write one chunk's status/override back to parameter_log
  4. merge_attachment_gaps() — inject automatic "gap" chunks between attachment periods
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ── interval rounding table ───────────────────────────────────────────────────
_CLEAN_INTERVALS_SEC = [60, 120, 300, 600, 900, 1800, 3600]


def _round_to_clean_interval(seconds: float) -> int:
    """Round *seconds* up to the nearest entry in _CLEAN_INTERVALS_SEC."""
    for iv in _CLEAN_INTERVALS_SEC:
        if seconds <= iv:
            return iv
    return _CLEAN_INTERVALS_SEC[-1]


# ── core grid builder ─────────────────────────────────────────────────────────

def compute_chunk_grid(
    logger_attachments: List[Dict[str, str]],
    chunk_size_sec: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Build a flat chunk list from one or more logger_attachments periods.

    Parameters
    ----------
    logger_attachments:
        List of {"start": <iso-str>, "end": <iso-str>} dicts (tz-aware strings).
    chunk_size_sec:
        Fixed chunk duration in seconds. When None, computed automatically as
        total on-animal duration ÷ 10, rounded to the nearest clean interval.

    Returns
    -------
    chunks : list of chunk dicts
        Each dict has: start, end, status ("pending"), params_override (None).
        Automatic "gap" chunks are inserted between non-contiguous attachment
        periods via merge_attachment_gaps().
    chunk_size_sec : int
        The resolved chunk size (seconds) — store this in the parameter_log for
        reproducibility.
    """
    if not logger_attachments:
        raise ValueError("logger_attachments is empty — cannot build chunk grid.")

    periods = [
        (pd.Timestamp(p["start"]), pd.Timestamp(p["end"]))
        for p in logger_attachments
    ]
    periods.sort(key=lambda t: t[0])

    # total on-animal seconds (sum of all attachment periods)
    total_sec = sum((e - s).total_seconds() for s, e in periods)

    if chunk_size_sec is None:
        raw = total_sec / 10.0
        chunk_size_sec = _round_to_clean_interval(raw)

    chunks: List[Dict[str, Any]] = []

    for period_start, period_end in periods:
        t = period_start
        while t < period_end:
            t_end = min(t + pd.Timedelta(seconds=chunk_size_sec), period_end)
            chunks.append({
                "start": str(t),
                "end": str(t_end),
                "status": "pending",
                "params_override": None,
            })
            t = t_end

    chunks = merge_attachment_gaps(chunks, periods)
    return chunks, chunk_size_sec


def merge_attachment_gaps(
    chunks: List[Dict[str, Any]],
    periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Dict[str, Any]]:
    """Insert automatic 'gap' chunks for time between attachment periods.

    Gaps are inserted in sorted order among the on-animal chunks so the full
    list is chronological.
    """
    if len(periods) <= 1:
        return chunks

    gap_chunks: List[Dict[str, Any]] = []
    for i in range(len(periods) - 1):
        gap_start = periods[i][1]
        gap_end = periods[i + 1][0]
        if gap_end > gap_start:
            gap_chunks.append({
                "start": str(gap_start),
                "end": str(gap_end),
                "status": "gap",
                "gap_reason": "between logger attachment periods (tag off animal)",
                "params_override": None,
            })

    all_chunks = chunks + gap_chunks
    all_chunks.sort(key=lambda c: pd.Timestamp(c["start"]))
    return all_chunks


# ── parameter_log I/O ─────────────────────────────────────────────────────────

def load_chunks(param_manager) -> Optional[List[Dict[str, Any]]]:
    """Return the hr_peak_detection_chunks list for this deployment, or None."""
    try:
        result = param_manager.get_from_config(
            ["hr_peak_detection_chunks"],
            section=None,
        )
        return result.get("hr_peak_detection_chunks")
    except (KeyError, ValueError):
        return None


def save_chunk_status(
    param_manager,
    chunk_index: int,
    status: str,
    gap_reason: Optional[str] = None,
    params_override: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the status (and optionally override params) for a single chunk.

    Reads the current chunk list, mutates the entry at *chunk_index*, then
    writes the whole list back. No-op if the chunk list does not exist yet.

    Parameters
    ----------
    status : "pending" | "validated" | "gap" | "override"
    gap_reason : free-text explanation (only meaningful when status=="gap")
    params_override : sparse dict of param keys that differ from deployment default
    """
    chunks = load_chunks(param_manager)
    if chunks is None or chunk_index >= len(chunks):
        return

    entry = chunks[chunk_index]
    entry["status"] = status

    if status == "gap" and gap_reason is not None:
        entry["gap_reason"] = gap_reason
    elif "gap_reason" in entry and status != "gap":
        del entry["gap_reason"]

    if params_override is not None:
        entry["params_override"] = params_override
    elif status != "override":
        entry["params_override"] = None

    param_manager.add_to_config(
        entries={"hr_peak_detection_chunks": chunks},
        section=None,
    )


# ── helper: flatten attachment periods for signal masking ────────────────────

def attachment_time_mask(
    datetime_series: pd.Series,
    logger_attachments: List[Dict[str, str]],
) -> pd.Series:
    """Return a boolean mask: True where datetime_series falls within any attachment."""
    mask = pd.Series(False, index=datetime_series.index)
    for p in logger_attachments:
        s = pd.Timestamp(p["start"])
        e = pd.Timestamp(p["end"])
        # coerce timezone if needed
        if datetime_series.dt.tz is not None and s.tzinfo is None:
            s = s.tz_localize(datetime_series.dt.tz)
            e = e.tz_localize(datetime_series.dt.tz)
        elif datetime_series.dt.tz is not None and s.tzinfo is not None:
            s = s.tz_convert(datetime_series.dt.tz)
            e = e.tz_convert(datetime_series.dt.tz)
        mask |= (datetime_series >= s) & (datetime_series <= e)
    return mask
