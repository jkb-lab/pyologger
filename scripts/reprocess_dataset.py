#!/usr/bin/env python3
"""Back up, rebuild, and compare a dataset's deployments.

For each configured deployment:
  1. move outputs/ -> outputs_backup/   (skipped if a backup already exists)
  2. rebuild via snakemake, one deployment at a time (a failure does not stop the run)
  3. compare the rebuild against the backup

Nothing is deleted. Review the comparison, then remove backups separately.

Usage:
    reprocess_dataset.py --dataset <id> [--deployments <id> ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from compare_outputs import compare  # noqa: E402

import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG = os.path.join(REPO, "config.yaml")
PYTHON = os.path.join(os.path.dirname(REPO), "venv", "bin", "python")


def _cfg() -> dict:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)


def backup(base: str, deployment: str) -> str:
    """Move outputs/ aside. Returns a status string."""
    out = os.path.join(base, "outputs")
    bak = os.path.join(base, "outputs_backup")
    if os.path.isdir(bak):
        return "backup_exists"
    if not os.path.isdir(out):
        return "no_outputs"
    shutil.move(out, bak)
    return "backed_up"


def rebuild(root: str, dataset: str, deployment: str) -> tuple[bool, str]:
    target = os.path.join(
        root, dataset, deployment, "outputs", f"{deployment}_output.nc"
    )
    proc = subprocess.run(
        [PYTHON, "-m", "snakemake", "--cores", "1", "--nolock", target],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return True, ""
    tail = [
        ln for ln in proc.stderr.splitlines()
        if ln.strip() and "it/s]" not in ln
    ][-3:]
    return False, " | ".join(tail)[:300]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--deployments", nargs="*", default=None)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = _cfg()
    root = cfg["paths"]["local_private_data"]
    configured = (cfg["datasets"].get(args.dataset) or {}).get("deployments") or []
    deployments = args.deployments or configured

    print(f"{args.dataset}: {len(deployments)} deployment(s)")
    if args.dry_run:
        for d in deployments:
            print(f"  would rebuild {d}")
        return 0

    results = []
    for i, dep in enumerate(deployments, 1):
        base = os.path.join(root, args.dataset, dep)
        print(f"\n[{i}/{len(deployments)}] {dep}", flush=True)

        b = backup(base, dep)
        print(f"    backup: {b}", flush=True)
        if b == "no_outputs" and not os.path.isdir(base):
            results.append({"deployment": dep, "stage": "missing", "detail": base})
            continue

        t0 = time.time()
        ok, err = rebuild(root, args.dataset, dep)
        print(f"    rebuild: {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s)",
              flush=True)
        if not ok:
            print(f"      {err}", flush=True)
            results.append({"deployment": dep, "stage": "rebuild_failed",
                            "detail": err})
            continue

        cmp_res = compare(args.dataset, dep, args.tolerance)
        print(f"    compare: {cmp_res['status']} "
              f"({cmp_res['n_vars_compared']} vars)", flush=True)
        for issue in cmp_res["issues"][:3]:
            print(f"      - {issue}", flush=True)
        results.append({"deployment": dep, "stage": cmp_res["status"],
                        "detail": cmp_res})

    print("\n" + "=" * 66)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["stage"]] = counts.get(r["stage"], 0) + 1
    print(f"SUMMARY {args.dataset}: {counts}")
    for r in results:
        if r["stage"] not in ("unchanged",):
            print(f"  {r['stage']:16} {r['deployment']}")

    out = os.path.join(REPO, f".reprocess_{args.dataset}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nfull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
