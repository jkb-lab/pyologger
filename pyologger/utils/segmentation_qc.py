from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def normalize_unit_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null", "unknown", "n/a", "na"}:
        return ""
    text = "".join(text.split())
    alias = {
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "sec": "s",
        "second": "s",
        "seconds": "s",
    }
    return alias.get(text, text)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        hit = cols.get(str(c).strip().lower())
        if hit is not None:
            return hit
    return None


def load_standardized_channel_unit_map(path: str | None) -> Dict[Tuple[str, str], str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    parent_col = _pick_col(df, ["Parent signal", "parent_signal"])
    channel_col = _pick_col(df, ["Channel ID", "Standardized Channel ID", "standardized_channel_id", "channel_id"])
    unit_override_col = _pick_col(df, ["Unit Override", "unit_override"])
    std_unit_col = _pick_col(df, ["Standardized Unit", "standardized_unit"])
    if parent_col is None or channel_col is None:
        return {}

    out: Dict[Tuple[str, str], str] = {}
    for _, row in df.iterrows():
        parent_signal = str(row.get(parent_col, "")).strip().lower()
        channel_id = str(row.get(channel_col, "")).strip().lower()
        if not parent_signal or not channel_id:
            continue
        override = str(row.get(unit_override_col, "")).strip() if unit_override_col else ""
        std_unit = str(row.get(std_unit_col, "")).strip() if std_unit_col else ""
        expected = normalize_unit_token(override or std_unit)
        if expected:
            out[(parent_signal, channel_id)] = expected
    return out
