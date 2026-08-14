#!/usr/bin/env python3
"""Compare a deployment's freshly built outputs/ against its outputs_backup/.

Reports per-variable drift in mean, min, and max. A deployment is "unchanged" when
every shared variable stays within the tolerance AND the variable set and sample
counts match. Anything else is flagged for review rather than silently accepted.

Usage:
    compare_outputs.py --dataset <id> --deployment <id> [--tolerance 0.01]
    compare_outputs.py --all --datasets <id> [<id> ...]   # every configured deployment
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import yaml

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _root() -> str:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)["paths"]["local_private_data"]


def _rel(new: float, old: float, scale: float) -> float:
    """Relative difference, scaled by `scale` to stay meaningful near zero."""
    if not np.isfinite(new) or not np.isfinite(old):
        return float("nan")
    denom = abs(scale)
    if denom < 1e-12:
        return 0.0 if abs(new - old) < 1e-12 else float("inf")
    return abs(new - old) / denom


def _stats(ds: xr.Dataset) -> dict:
    """Per-variable mean/min/max/count for every numeric data variable."""
    out = {}
    for name, var in ds.data_vars.items():
        try:
            arr = np.asarray(var.values, dtype=float)
        except (ValueError, TypeError):
            continue  # non-numeric (strings, event blobs)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        out[str(name)] = {
            "mean": float(np.mean(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "n": int(arr.size),
        }
    return out


def compare(dataset: str, deployment: str, tolerance: float = 0.01) -> dict:
    root = _root()
    base = os.path.join(root, dataset, deployment)
    new_p = os.path.join(base, "outputs", f"{deployment}_output.nc")
    old_p = os.path.join(base, "outputs_backup", f"{deployment}_output.nc")

    result = {
        "dataset": dataset,
        "deployment": deployment,
        "status": None,
        "issues": [],
        "n_vars_compared": 0,
        "worst": None,
    }

    if not os.path.isfile(old_p):
        result["status"] = "no_backup"
        result["issues"].append("no outputs_backup to compare against")
        return result
    if not os.path.isfile(new_p):
        result["status"] = "missing_new"
        result["issues"].append("outputs/ has no output.nc (build failed?)")
        return result

    with xr.open_dataset(old_p) as old_ds, xr.open_dataset(new_p) as new_ds:
        old_s, new_s = _stats(old_ds), _stats(new_ds)

    only_old = sorted(set(old_s) - set(new_s))
    only_new = sorted(set(new_s) - set(old_s))
    if only_old:
        result["issues"].append(f"variables lost: {', '.join(only_old[:6])}")
    if only_new:
        result["issues"].append(f"variables added: {', '.join(only_new[:6])}")

    worst = {"var": None, "metric": None, "rel": 0.0, "old": None, "new": None}
    for name in sorted(set(old_s) & set(new_s)):
        o, n = old_s[name], new_s[name]
        result["n_vars_compared"] += 1
        if o["n"] != n["n"]:
            result["issues"].append(
                f"{name}: sample count {o['n']} -> {n['n']}"
            )
        scale = max(abs(o["max"] - o["min"]), abs(o["mean"]), 1e-12)
        for metric in ("mean", "min", "max"):
            rel = _rel(n[metric], o[metric], scale)
            if np.isfinite(rel) and rel > worst["rel"]:
                worst = {
                    "var": name, "metric": metric, "rel": rel,
                    "old": o[metric], "new": n[metric],
                }
            if not np.isfinite(rel) or rel > tolerance:
                result["issues"].append(
                    f"{name}.{metric}: {o[metric]:.6g} -> {n[metric]:.6g} "
                    f"({rel:.2%})"
                )

    result["worst"] = worst if worst["var"] else None
    result["status"] = "unchanged" if not result["issues"] else "changed"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--deployment")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    pairs = []
    if args.all:
        with open(CONFIG) as fh:
            cfg = yaml.safe_load(fh)
        for ds, det in (cfg.get("datasets") or {}).items():
            if args.datasets and ds not in args.datasets:
                continue
            for dep in ((det or {}).get("deployments") or []):
                pairs.append((ds, dep))
    else:
        if not (args.dataset and args.deployment):
            ap.error("--dataset and --deployment required unless --all")
        pairs.append((args.dataset, args.deployment))

    results = [compare(d, p, args.tolerance) for d, p in pairs]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    by_status: dict[str, list] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)

    for status in ("unchanged", "changed", "missing_new", "no_backup"):
        rows = by_status.get(status, [])
        if not rows:
            continue
        print(f"\n=== {status.upper()} ({len(rows)}) ===")
        for r in rows:
            w = r["worst"]
            tail = ""
            if w and w["var"]:
                tail = f"  worst: {w['var']}.{w['metric']} {w['rel']:.3%}"
            print(f"  {r['deployment']:28} vars={r['n_vars_compared']:3}{tail}")
            for issue in r["issues"][:5]:
                print(f"      - {issue}")
            if len(r["issues"]) > 5:
                print(f"      ... {len(r['issues']) - 5} more")

    counts = {k: len(v) for k, v in by_status.items()}
    print(f"\nSUMMARY: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
