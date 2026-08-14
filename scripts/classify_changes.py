#!/usr/bin/env python3
"""Classify per-deployment differences between outputs/ and outputs_backup/.

compare_outputs.py reports *what* changed; this says whether the change is a repair,
an expected consequence of a pipeline change, or something that needs review.

Categories:
  corrected  depth sign now agrees with the raw pressure convention (the backup was
             inverted by a bad conversion_factor=-1.0)
  recovered  the new run has materially more signal than the backup, which had been
             truncated by the old analysis-window crop
  expected   only calibrated_acc / calibrated_mag differ -- step 03 drops these on
             purpose (see 03_tag2animal.py), and most backups predate that change
  review     anything else
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import yaml

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
INTENTIONAL_DROPS = {"signal_data_calibrated_acc", "signal_data_calibrated_mag"}


def _root() -> str:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)["paths"]["local_private_data"]


def _depth_stats(path: str):
    try:
        with xr.open_dataset(path) as ds:
            if "signal_data_depth" not in ds.data_vars:
                return None
            v = np.asarray(ds["signal_data_depth"].values, dtype=float)
            v = v[np.isfinite(v)]
            return (float(v.mean()), float(v.min()), float(v.max())) if v.size else None
    except Exception:
        return None


def _raw_pressure_positive_down(base: str) -> bool | None:
    """True if raw pressure reads positive with depth (so no sign flip is wanted)."""
    for rel in ("outputs_backup/data.pkl", "outputs/data.pkl"):
        p = os.path.join(base, rel)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            df = d.signal_data.get("pressure")
            if df is None:
                continue
            col = next(c for c in df.columns if c != "datetime")
            v = np.asarray(df[col], dtype=float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            return abs(v.max()) > abs(v.min())
        except Exception:
            continue
    return None


def _span_minutes(path: str, var: str = "signal_data_ecg") -> float | None:
    try:
        with xr.open_dataset(path) as ds:
            dim = next((d for d in ds.dims if d.startswith("ecg")), None)
            if dim is None or dim not in ds.coords:
                return None
            t = pd.to_datetime(ds[dim].values)
            return float((t.max() - t.min()).total_seconds() / 60)
    except Exception:
        return None


def classify(dataset: str, deployment: str) -> dict:
    base = os.path.join(_root(), dataset, deployment)
    new_p = os.path.join(base, "outputs", f"{deployment}_output.nc")
    old_p = os.path.join(base, "outputs_backup", f"{deployment}_output.nc")

    out = {"deployment": deployment, "category": "review", "notes": []}
    if not os.path.isfile(old_p):
        out["category"] = "no_backup"
        return out
    if not os.path.isfile(new_p):
        out["category"] = "missing_new"
        return out

    with xr.open_dataset(old_p) as o, xr.open_dataset(new_p) as n:
        old_vars = {v for v in o.data_vars if str(v).startswith("signal_data")}
        new_vars = {v for v in n.data_vars if str(v).startswith("signal_data")}

    lost = old_vars - new_vars
    gained = new_vars - old_vars
    unexpected_lost = lost - INTENTIONAL_DROPS

    if lost & INTENTIONAL_DROPS:
        out["notes"].append("calibrated_acc/mag dropped by step 03 (intentional)")
    if unexpected_lost:
        out["notes"].append(f"unexpectedly lost: {', '.join(sorted(unexpected_lost))}")
    if gained:
        out["notes"].append(f"gained: {', '.join(sorted(gained))}")

    # Depth: did the sign come into agreement with the raw pressure convention?
    o_d, n_d = _depth_stats(old_p), _depth_stats(new_p)
    depth_corrected = False
    if o_d and n_d:
        pos_down = _raw_pressure_positive_down(base)
        flipped = (
            abs(o_d[1] + n_d[2]) < 0.01 * max(1.0, abs(n_d[2]))
            and abs(o_d[2] + n_d[1]) < 0.01 * max(1.0, abs(n_d[2]))
        )
        if flipped and pos_down and n_d[2] > 0:
            depth_corrected = True
            out["notes"].append(
                f"depth sign corrected: max {o_d[2]:.2f} -> {n_d[2]:.2f} m"
            )
        elif abs(n_d[0] - o_d[0]) > 0.01 * max(1.0, abs(o_d[0])):
            out["notes"].append(f"depth mean {o_d[0]:.3f} -> {n_d[0]:.3f}")

    # Record length: did the crop removal restore truncated signal?
    o_s, n_s = _span_minutes(old_p), _span_minutes(new_p)
    recovered = bool(o_s and n_s and n_s > o_s * 1.1)
    if recovered:
        out["notes"].append(f"record recovered: {o_s:.1f} -> {n_s:.1f} min")

    if unexpected_lost:
        out["category"] = "review"
    elif recovered:
        out["category"] = "recovered"
    elif depth_corrected:
        out["category"] = "corrected"
    elif lost <= INTENTIONAL_DROPS and not gained:
        out["category"] = "expected"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    deployments = (cfg["datasets"].get(args.dataset) or {}).get("deployments") or []

    results = [classify(args.dataset, d) for d in deployments]
    order = ["expected", "corrected", "recovered", "review", "missing_new", "no_backup"]
    for cat in order:
        rows = [r for r in results if r["category"] == cat]
        if not rows:
            continue
        print(f"\n=== {cat.upper()} ({len(rows)}) ===")
        for r in rows:
            print(f"  {r['deployment']}")
            for note in r["notes"]:
                print(f"      - {note}")

    counts = {c: sum(1 for r in results if r["category"] == c) for c in order}
    print(f"\nSUMMARY {args.dataset}: "
          f"{ {k: v for k, v in counts.items() if v} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
