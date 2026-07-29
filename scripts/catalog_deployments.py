#!/usr/bin/env python3
"""
catalog_deployments.py
======================
Audit preparedness of all tagged-animal deployments across three datasets.
Cross-references on-disk pipeline stages and nc/pkl outputs against known
session status and priority notes.

Usage:
    python catalog_deployments.py              # full report, all datasets
    python catalog_deployments.py --dataset KW # one dataset
    python catalog_deployments.py --json        # machine-readable JSON
"""

import os
import csv
import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path

try:
    import xarray as xr
    HAS_XR = True
except ImportError:
    HAS_XR = False

# ─── DATASET REGISTRY ─────────────────────────────────────────────────────────

DATASETS = {
    "KW": {
        "label":    "KW — Killer Whales (oror)",
        "dir":      Path("/Volumes/WORK-SSD/Datasets/Unpublished/oror-adult-orca_hr-sr-vid_sw_JKB-PP"),
        "species":  "oror",
        "type":     "captive, short deployments",
        # Folders matching this regex are real deployments (skip TEST/hosa/etc.)
        "dep_re":   re.compile(r"^(\d{4}-\d{2}-\d{2})_(oror-\d{3})$"),
        "pipeline_stages": {
            "01_raw-data": "raw",
            "outputs":     "nc/pkl",
        },
        "stage_completion": {
            "01_raw-data": lambda p: any(p.glob("*.bin")) or any(p.glob("*.csv")) or any(p.glob("*.edf")),
            "outputs":     lambda p: any(p.glob("*.nc")) and (p / "data.pkl").exists(),
        },
        "metadata_fn": None,  # pulled from nc attrs + event counts
    },
    "NES": {
        "label":    "NES — N. Elephant Seals (mian)",
        "dir":      Path("/Volumes/WORK-SSD/Datasets/Unpublished/mian-juv-nese_sleep_lml-ano_JKB"),
        "species":  "mian",
        "type":     "captive + wild, multi-day deployments",
        "dep_re":   re.compile(r"^(\d{4}-\d{2}-\d{2})_(mian-\d{3})$"),
        "pipeline_stages": {
            "01_raw-data":                 "raw",
            "02_preprocessed-motion-data": "preproc",
            "05_ica-processing":           "ica",
            "06_sleep-scoring":            "scored",
            "10_sleep-estimation":         "sleep_est",
            "outputs":                     "nc/pkl",
        },
        "stage_completion": {
            "02_preprocessed-motion-data": lambda p: any(p.glob("*.csv")) or any(p.glob("*.edf")),
            "05_ica-processing":           lambda p: (p / "output-edf").exists() or any(p.glob("edf_*")),
            "06_sleep-scoring":            lambda p: any(p.glob("*Hypnogram*")),
            "10_sleep-estimation":         lambda p: any(p.glob("*SleepStats*")) or any(p.glob("*.png")),
            "outputs":                     lambda p: any(p.glob("*.nc")) and (p / "data.pkl").exists(),
        },
        "metadata_fn": "00_Metadata/00_Sleep_Study_Metadata.csv",
    },
    "BW": {
        "label":    "BW — Blue Whales (bamu)",
        "dir":      Path("/Volumes/WORK-SSD/Datasets/Unpublished/wild-whale-adult_hr-sr_JG-PP"),
        "species":  "bamu",
        "type":     "wild, short deployments",
        "dep_re":   re.compile(r"^(\d{4}-\d{2}-\d{2})_(bamu-\d{3})$"),
        "pipeline_stages": {
            "01_raw-data": "raw",
            "outputs":     "nc/pkl",
        },
        "stage_completion": {
            "01_raw-data": lambda p: any(p.glob("*.csv")) or any(p.glob("*.mat")) or any(p.glob("*.ube")),
            "outputs":     lambda p: any(p.glob("*.nc")) and (p / "data.pkl").exists(),
        },
        "metadata_fn": None,
    },
}

# ─── SESSION STATE — edit as needed ───────────────────────────────────────────

IN_PIPELINE = {
    "2021-04-04_mian-009",   # NES Fatigued Fiona
    "2023-06-23_oror-002",   # KW — in Snakemake run
}

LOADED_IN_SESSION = {
    # NES
    "2021-04-04_mian-009",
    "2021-04-17_mian-011",
    "2022-04-01_mian-013",
    # KW (loaded, not all in pipeline)
    "2023-06-23_oror-002",
    "2023-09-29_oror-002",
    "2023-10-18_oror-001",
    "2023-12-12_oror-001",
    "2024-01-16_oror-002",
    "2024-01-24_oror-001",
    "2025-01-30_oror-001",
    # KW — excluded
    "2023-06-13_oror-002",
    "2023-10-18_oror-002",
    "2024-06-06_oror-002",
    "2024-12-19_oror-001",
}

EXCLUDE = {
    # deploy_id: reason
    "2023-06-13_oror-002":  "0 breath cycles",
    "2023-10-18_oror-002":  "too short",
    "2024-06-06_oror-002":  "0 breaths",
    "2024-12-19_oror-001":  "0 breaths",
}

PRIORITY = {
    "2021-04-10_mian-010": "⭐ Goodnight Gerty — WILD weanling, clear gyro breath signal",
    "2020-10-08_mian-005": "⭐ Bertha the Sleeping Beauty — CAPTIVE, no device failure",
}

# ─── METADATA LOADERS ─────────────────────────────────────────────────────────

def load_nes_metadata(ds_dir: Path, metadata_fn: str) -> dict:
    """Parse transposed NES CSV → {mian-NNN: {attr: value}}."""
    csv_path = ds_dir / metadata_fn
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    name_col = next(c for c in rows[0].keys() if "Name" in c)
    col_names = [c for c in rows[0].keys() if c and c != name_col]
    animals = {}
    for col in col_names:
        attrs = {r[name_col]: r[col] for r in rows}
        dep_num = attrs.get("Deployment", "").strip()
        if not dep_num or not dep_num.isdigit():
            continue
        key = f"mian-{int(dep_num):03d}"
        animals[key] = {
            "nickname":        col.strip(),
            "recording_id":    attrs.get("Recording ID", "").strip(),
            "duration_h":      attrs.get("Duration_ON_ANIMAL_h", "").strip(),
            "device_failure":  attrs.get("Device Failure", "").strip(),
            "ica_quality":     attrs.get("ICA Decomposition Quality", "").strip(),
            "sex":             attrs.get("Sex", "").strip(),
            "age":             attrs.get("Age Estimate", "").strip(),
        }
    return animals


def read_nc_stats(nc_path: Path) -> dict:
    """Extract breath count, HR beat count, and duration from output nc."""
    if not HAS_XR or not nc_path.exists():
        return {}
    try:
        ds = xr.open_dataset(nc_path)
        keys = ds["event_data_key"].values if "event_data_key" in ds else []
        breaths = sum(1 for k in keys if k in ("exhalation_breath", "uw_exhalation"))
        hr_beats = sum(1 for k in keys if "heartbeat" in str(k) and "accepted" in str(k))
        # Duration from time coordinate
        dur_min = None
        if "time" in ds and len(ds.time) > 1:
            span = ds.time.values[-1] - ds.time.values[0]
            dur_min = float(span / np.timedelta64(1, "m"))
        # Animal/dataset info from attrs
        animal_id = ds.attrs.get("animal_info_Animal_ID", "")
        dataset_id = ds.attrs.get("dataset_info_Dataset_ID", "")
        ds.close()
        return {"breaths": breaths, "hr_beats": hr_beats, "dur_min": dur_min,
                "animal_id": animal_id, "dataset_id": dataset_id}
    except Exception:
        return {}

# ─── STAGE CHECKER ────────────────────────────────────────────────────────────

def check_stage(dep_path: Path, stage: str, completion_fns: dict) -> str:
    stage_path = dep_path / stage
    if not stage_path.exists():
        return "—"
    checker = completion_fns.get(stage)
    if checker:
        return "✅" if checker(stage_path) else "⚠️ empty"
    return "✅" if any(stage_path.iterdir()) else "⚠️ empty"

# ─── READINESS TIER ───────────────────────────────────────────────────────────

def assign_tier(deploy_id: str, stages: dict, ds_key: str) -> tuple:
    if deploy_id in IN_PIPELINE:
        return "in_pipeline",    "🔵"
    if deploy_id in EXCLUDE:
        return "excluded",       "🚫"
    nc_ready = stages.get("nc/pkl") == "✅"
    scored   = stages.get("scored") == "✅"
    preproc  = stages.get("preproc") == "✅"
    raw      = stages.get("raw") == "✅"
    if ds_key == "NES":
        if nc_ready and scored:  return "pipeline_ready", "🟢"
        if scored:               return "scored_only",    "🟡"
        if preproc:              return "preproc_only",   "🟠"
        if raw:                  return "raw_only",       "🔴"
    else:  # KW / BW — no scoring step
        if nc_ready:             return "pipeline_ready", "🟢"
        if raw:                  return "raw_only",       "🔴"
    return "unknown", "❓"

# ─── SCANNER ──────────────────────────────────────────────────────────────────

def scan_dataset(ds_key: str) -> list:
    cfg = DATASETS[ds_key]
    ds_dir = cfg["dir"]
    if not ds_dir.exists():
        return []

    meta = {}
    if cfg["metadata_fn"]:
        meta = load_nes_metadata(ds_dir, cfg["metadata_fn"])

    deps = []
    for folder in sorted(ds_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = cfg["dep_re"].match(folder.name)
        if not m:
            continue
        date_str, animal_id = m.group(1), m.group(2)
        deploy_id = folder.name.lower()

        stages = {
            label: check_stage(folder, stage, cfg["stage_completion"])
            for stage, label in cfg["pipeline_stages"].items()
        }

        # Try to enrich from nc
        nc_path = folder / "outputs" / f"{folder.name}_output.nc"
        nc_stats = read_nc_stats(nc_path)

        animal_meta = meta.get(animal_id, {})
        tier, tier_sym = assign_tier(deploy_id, stages, ds_key)

        deps.append({
            "dataset":        ds_key,
            "deploy_id":      deploy_id,
            "date":           date_str,
            "animal_id":      animal_id,
            "nickname":       animal_meta.get("nickname", ""),
            "recording_id":   animal_meta.get("recording_id", ""),
            "duration_h":     animal_meta.get("duration_h", ""),
            "dur_min":        nc_stats.get("dur_min"),
            "breaths":        nc_stats.get("breaths"),
            "hr_beats":       nc_stats.get("hr_beats"),
            "device_failure": animal_meta.get("device_failure", ""),
            "ica_quality":    animal_meta.get("ica_quality", ""),
            "sex":            animal_meta.get("sex", ""),
            "age":            animal_meta.get("age", ""),
            "stages":         stages,
            "tier":           tier,
            "tier_sym":       tier_sym,
            "in_session":     deploy_id in LOADED_IN_SESSION,
            "in_pipeline":    deploy_id in IN_PIPELINE,
            "is_priority":    deploy_id in PRIORITY,
            "priority_note":  PRIORITY.get(deploy_id, ""),
            "exclude_reason": EXCLUDE.get(deploy_id, ""),
        })
    return deps

# ─── DISPLAY ──────────────────────────────────────────────────────────────────

def hr(label, w=76):
    print(f"\n{'═'*w}\n  {label}\n{'═'*w}")


def dur_str(d: dict) -> str:
    if d.get("dur_min") is not None:
        return f"{d['dur_min']/60:.1f}h"
    if d.get("duration_h"):
        try:
            return f"{float(d['duration_h']):.1f}h"
        except ValueError:
            pass
    return "  ?"


def print_dataset_table(deps: list, ds_key: str):
    cfg = DATASETS[ds_key]
    stage_labels = list(cfg["pipeline_stages"].values())

    hr(cfg["label"])
    if not cfg["dir"].exists():
        print(f"  ❌  {cfg['dir']} — SSD not mounted")
        return
    if not deps:
        print("  (no matching deployment folders found)")
        return

    has_breaths = any(d.get("breaths") is not None for d in deps)
    has_beats   = any(d.get("hr_beats") is not None for d in deps)

    # Header
    stage_hdr = "  ".join(f"{s:>9}" for s in stage_labels)
    extras = ("  breaths  hr_beats" if has_beats else "  breaths") if has_breaths else ""
    print(f"\n  {'Deploy ID':<30} {'Nickname':<26} {'Dur':>5}  {stage_hdr}{extras}  Flags")
    print(f"  {'─'*30} {'─'*26} {'─'*5}  {'  '.join(['─'*9]*len(stage_labels))}"
          + ("  ───────  ────────" if has_beats else "  ───────" if has_breaths else "") + "  ─────")

    for d in deps:
        stages_str = "  ".join(f"{d['stages'].get(s,'—'):>9}" for s in stage_labels)
        dur = dur_str(d)
        nickname = (d["nickname"] or d["recording_id"] or "")[:24]

        flags = []
        if d["in_pipeline"]:    flags.append("🔵IN-PIPE")
        if d["in_session"] and not d["in_pipeline"]:  flags.append("✅loaded")
        if d["is_priority"]:    flags.append("⭐")
        if d["exclude_reason"]: flags.append(f"🚫{d['exclude_reason'][:18]}")
        if d["device_failure"] == "Yes": flags.append("⚡devfail")
        flag_str = " ".join(flags)

        breaths_col = ""
        if has_breaths:
            b = d.get("breaths")
            breaths_col = f"  {b if b is not None else '?':>7}"
        beats_col = ""
        if has_beats:
            b = d.get("hr_beats")
            beats_col = f"  {b if b is not None else '?':>8}"

        print(f"  {d['tier_sym']} {d['deploy_id']:<28} {nickname:<26} {dur:>5}  {stages_str}{breaths_col}{beats_col}  {flag_str}")


def print_report(all_deps: list, ds_filter: str | None):
    print("\n" + "█"*76)
    print("  DEPLOYMENT PREPAREDNESS CATALOG")
    print("█"*76)

    selected = {k: v for k, v in DATASETS.items() if not ds_filter or k == ds_filter}

    # ── Per-dataset tables ─────────────────────────────────────────────────────
    for ds_key in selected:
        deps = [d for d in all_deps if d["dataset"] == ds_key]
        print_dataset_table(deps, ds_key)

    # ── Cross-dataset summary ──────────────────────────────────────────────────
    hr("SUMMARY")
    for ds_key in selected:
        deps = [d for d in all_deps if d["dataset"] == ds_key]
        cfg  = DATASETS[ds_key]
        if not cfg["dir"].exists():
            print(f"\n  {cfg['label']}")
            print(f"    ❌  {cfg['dir']} — SSD not mounted")
            continue

        from collections import Counter
        tier_ct = Counter(d["tier"] for d in deps)
        print(f"\n  {cfg['label']}")
        print(f"    Total on disk         : {len(deps)}")
        print(f"    🔵 In pipeline        : {tier_ct['in_pipeline']}")
        print(f"    🟢 Pipeline-ready     : {tier_ct['pipeline_ready']}")
        if ds_key == "NES":
            print(f"    🟡 Scored, need nc/pkl: {tier_ct['scored_only']}")
            print(f"    🟠 Preproc only       : {tier_ct['preproc_only']}")
        print(f"    🔴 Raw only           : {tier_ct['raw_only']}")
        print(f"    🚫 Excluded           : {tier_ct['excluded']}")

        # Breath cycle summary for KW/BW
        if ds_key in ("KW", "BW"):
            usable = [d for d in deps if d["tier"] not in ("excluded","unknown")
                      and d.get("breaths") is not None and d["breaths"] > 0]
            total_b = sum(d["breaths"] for d in usable)
            print(f"    🫁 Total breath cycles: {total_b}  across {len(usable)} deployments")

        # Device failures for NES
        if ds_key == "NES":
            fails = [d for d in deps if d.get("device_failure") == "Yes"]
            if fails:
                print(f"    ⚡ Device failures     : {len(fails)}  ({', '.join(d['animal_id'] for d in fails)})")

    # ── Priority queue ─────────────────────────────────────────────────────────
    priority = [d for d in all_deps if d["is_priority"]]
    if priority and (not ds_filter):
        hr("⭐ PRIORITY DEPLOYMENTS")
        for d in priority:
            print(f"  {d['tier_sym']}  {d['deploy_id']}  ({d['dataset']})")
            print(f"       {d['priority_note']}")
            stage_str = "  ".join(f"{k}:{v}" for k, v in d["stages"].items())
            print(f"       stages: {stage_str}")

    print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-dataset deployment preparedness catalog")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Limit to one dataset (KW, NES, BW)")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    args = parser.parse_args()

    ds_filter = args.dataset

    all_deps = []
    for ds_key in DATASETS:
        if ds_filter and ds_key != ds_filter:
            continue
        all_deps.extend(scan_dataset(ds_key))

    if args.json:
        print(json.dumps(all_deps, indent=2, default=str))
    else:
        print_report(all_deps, ds_filter)


if __name__ == "__main__":
    main()
