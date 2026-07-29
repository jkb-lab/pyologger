#!/usr/bin/env python3
"""
catalog_nes_preparedness.py
===========================
Audit preparedness of NES sleep study deployments for analysis.
Cross-references on-disk pipeline stages against known metadata.

Usage:
    python catalog_nes_preparedness.py
    python catalog_nes_preparedness.py --json   # machine-readable output
"""

import os
import csv
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

NES_DIR = Path("/Volumes/WORK-SSD/Datasets/Unpublished/mian-juv-nese_sleep_lml-ano_JKB")
METADATA_CSV = NES_DIR / "00_Metadata" / "00_Sleep_Study_Metadata.csv"

# Pipeline stages to check (folder name → short label)
PIPELINE_STAGES = {
    "01_raw-data":                 "raw",
    "02_preprocessed-motion-data": "preproc",
    "05_ica-processing":           "ica",
    "06_sleep-scoring":            "scored",
    "10_sleep-estimation":         "sleep_est",
    "outputs":                     "nc/pkl",
}

# Key output file patterns that mark a stage as complete
STAGE_COMPLETION = {
    "02_preprocessed-motion-data": lambda p: any(p.glob("*.csv")) or any(p.glob("*.edf")),
    "05_ica-processing":           lambda p: (p / "output-edf").exists() or any(p.glob("edf_*")),
    "06_sleep-scoring":            lambda p: any(p.glob("*Hypnogram*")),
    "10_sleep-estimation":         lambda p: any(p.glob("*SleepStats*")) or any(p.glob("*.png")),
    "outputs":                     lambda p: any(p.glob("*.nc")) and any(p.glob("*.pkl")),
}

# Manually curated status — edit as needed
LOADED_IN_SESSION = {
    "2021-04-04_mian-009",  # Fatigued Fiona  — IN Snakemake run
    "2021-04-17_mian-011",  # Hypoactive Heidi
    "2022-04-01_mian-013",  # Jaunting Juliette
}
IN_PIPELINE = {
    "2021-04-04_mian-009",  # Fatigued Fiona
}
EXCLUDE = {
    # deploy_id: reason
}
PRIORITY = {
    # deploy_id: note (⭐ = top priority)
    "2021-04-10_mian-010": "⭐ Goodnight Gerty — WILD weanling, clear gyro breath signal",
    "2020-10-08_mian-005": "⭐ Bertha the Sleeping Beauty — CAPTIVE, no device failure",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_metadata():
    """Parse transposed metadata CSV → {mian-NNN: {attr: value}}."""
    if not METADATA_CSV.exists():
        return {}
    with open(METADATA_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    animals = {}
    col_names = [c for c in rows[0].keys() if c and c != "﻿Name"]
    for col in col_names:
        attrs = {r["Name"] if "Name" in r else r.get("﻿Name", ""): r[col] for r in rows}
        dep_num = attrs.get("Deployment", "").strip()
        if not dep_num or not dep_num.isdigit():
            continue
        key = f"mian-{int(dep_num):03d}"
        animals[key] = {
            "nickname":       col.strip(),
            "nickname_short": attrs.get("Nickname", "").strip(),
            "recording_id":   attrs.get("Recording ID", "").strip(),
            "duration_h":     attrs.get("Duration_ON_ANIMAL_h", "").strip(),
            "device_failure": attrs.get("Device Failure", "").strip(),
            "ica_quality":    attrs.get("ICA Decomposition Quality", "").strip(),
            "scorable_no_ica":attrs.get("Scorable Without ICA", "").strip(),
            "age":            attrs.get("Age Estimate", "").strip(),
            "sex":            attrs.get("Sex", "").strip(),
        }
    return animals


def check_stage(dep_path: Path, stage: str) -> str:
    """Return '✅', '⚠️ empty', or '—' for a pipeline stage."""
    stage_path = dep_path / stage
    if not stage_path.exists():
        return "—"
    checker = STAGE_COMPLETION.get(stage)
    if checker:
        return "✅" if checker(stage_path) else "⚠️ empty"
    return "✅" if any(stage_path.iterdir()) else "⚠️ empty"


def scan_deployments():
    """Return list of deployment dicts, sorted by deploy_id."""
    if not NES_DIR.exists():
        return []

    meta = load_metadata()
    deps = []

    for folder in sorted(NES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        # Match YYYY-MM-DD_mian-NNN
        import re
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(mian-\d{3})", name)
        if not m:
            continue

        date_str, animal_id = m.group(1), m.group(2)
        deploy_id = name.lower()
        animal_meta = meta.get(animal_id, {})

        stages = {
            label: check_stage(folder, stage)
            for stage, label in PIPELINE_STAGES.items()
        }

        # Infer overall readiness tier
        nc_pkl_ready = stages["nc/pkl"] == "✅"
        scored_ready = stages["scored"] == "✅"
        preproc_ready = stages["preproc"] == "✅"

        if deploy_id in IN_PIPELINE:
            tier, tier_sym = "in_pipeline",  "🔵"
        elif deploy_id in EXCLUDE:
            tier, tier_sym = "excluded",     "🚫"
        elif nc_pkl_ready and scored_ready:
            tier, tier_sym = "pipeline_ready", "🟢"
        elif scored_ready:
            tier, tier_sym = "scored_only",  "🟡"
        elif preproc_ready:
            tier, tier_sym = "preproc_only", "🟠"
        elif stages["raw"] == "✅":
            tier, tier_sym = "raw_only",     "🔴"
        else:
            tier, tier_sym = "unknown",      "❓"

        deps.append({
            "deploy_id":    deploy_id,
            "date":         date_str,
            "animal_id":    animal_id,
            "nickname":     animal_meta.get("nickname", ""),
            "recording_id": animal_meta.get("recording_id", ""),
            "duration_h":   animal_meta.get("duration_h", ""),
            "device_failure": animal_meta.get("device_failure", ""),
            "ica_quality":  animal_meta.get("ica_quality", ""),
            "sex":          animal_meta.get("sex", ""),
            "age":          animal_meta.get("age", ""),
            "stages":       stages,
            "tier":         tier,
            "tier_sym":     tier_sym,
            "in_session":   deploy_id in LOADED_IN_SESSION,
            "in_pipeline":  deploy_id in IN_PIPELINE,
            "is_priority":  deploy_id in PRIORITY,
            "priority_note":PRIORITY.get(deploy_id, ""),
            "exclude_reason": EXCLUDE.get(deploy_id, ""),
        })

    return deps


# ─── DISPLAY ──────────────────────────────────────────────────────────────────

def hr(label, w=72):
    print(f"\n{'═'*w}\n  {label}\n{'═'*w}")


def print_table(deps):
    stage_labels = list(PIPELINE_STAGES.values())
    header_stages = "  ".join(f"{s:>8}" for s in stage_labels)
    print(f"\n  {'Deploy ID':<30} {'Nickname':<32} {'RecType':<24}  {header_stages}  Status")
    print(f"  {'─'*30} {'─'*32} {'─'*24}  {'  '.join(['─'*8]*len(stage_labels))}  ──────")

    for d in deps:
        stages_str = "  ".join(f"{d['stages'].get(s,'—'):>8}" for s in stage_labels)
        badges = []
        if d["in_pipeline"]:  badges.append("🔵 PIPELINE")
        if d["in_session"]:   badges.append("✅ loaded")
        if d["is_priority"]:  badges.append("⭐")
        if d["device_failure"] == "Yes": badges.append("⚡ dev_fail")
        extra = " | ".join(badges)

        nickname_trunc = d["nickname"][:30] if d["nickname"] else "—"
        rectype_trunc  = d["recording_id"][:22] if d["recording_id"] else "—"
        print(f"  {d['tier_sym']} {d['deploy_id']:<28} {nickname_trunc:<32} {rectype_trunc:<24}  {stages_str}  {extra}")


def print_report(deps):
    print("\n" + "█"*72)
    print("  NES SLEEP STUDY — DEPLOYMENT PREPAREDNESS CATALOG")
    print(f"  Source: {NES_DIR}")
    print("█"*72)

    if not NES_DIR.exists():
        print("\n  ❌  SSD not mounted — check /Volumes/WORK-SSD/")
        return

    # ── Full table ────────────────────────────────────────────────────────────
    hr("ALL DEPLOYMENTS")
    print_table(deps)

    # ── Pipeline-ready (can be added to Snakemake now) ────────────────────────
    ready = [d for d in deps if d["tier"] == "pipeline_ready" and not d["in_pipeline"]]
    if ready:
        hr("🟢 PIPELINE-READY — have scored hypnogram + nc/pkl outputs")
        for d in ready:
            dur = f"{float(d['duration_h']):.1f}h" if d["duration_h"].replace(".","",1).isdigit() else "?"
            fail = " ⚡dev_fail" if d["device_failure"] == "Yes" else ""
            print(f"  🟢  {d['deploy_id']}  {d['nickname']:<34} {dur}{fail}")

    # ── Scored but no nc/pkl (need Snakemake run) ────────────────────────────
    scored = [d for d in deps if d["tier"] == "scored_only"]
    if scored:
        hr("🟡 SCORED — hypnogram exists but no nc/pkl (run Snakemake)")
        for d in scored:
            print(f"  🟡  {d['deploy_id']}  {d['nickname']}")

    # ── Priority animals ──────────────────────────────────────────────────────
    priority = [d for d in deps if d["is_priority"]]
    if priority:
        hr("⭐ PRIORITY DEPLOYMENTS")
        for d in priority:
            print(f"  {d['tier_sym']}  {d['deploy_id']}")
            print(f"       {d['priority_note']}")
            stage_str = "  ".join(f"{k}:{v}" for k, v in d["stages"].items())
            print(f"       stages: {stage_str}")

    # ── Currently in Snakemake pipeline ───────────────────────────────────────
    in_pipe = [d for d in deps if d["in_pipeline"]]
    hr("🔵 IN SNAKEMAKE PIPELINE")
    if in_pipe:
        for d in in_pipe:
            print(f"  🔵  {d['deploy_id']}  {d['nickname']}")
    else:
        print("  (none)")

    # ── Summary ───────────────────────────────────────────────────────────────
    hr("SUMMARY")
    tier_counts = defaultdict(int)
    for d in deps:
        tier_counts[d["tier"]] += 1

    total = len(deps)
    print(f"  Total deployments on disk : {total}")
    print(f"  🔵 In Snakemake pipeline  : {tier_counts['in_pipeline']}")
    print(f"  🟢 Pipeline-ready (nc+pkl): {tier_counts['pipeline_ready']}")
    print(f"  🟡 Scored, need nc/pkl    : {tier_counts['scored_only']}")
    print(f"  🟠 Preprocessed only      : {tier_counts['preproc_only']}")
    print(f"  🔴 Raw data only          : {tier_counts['raw_only']}")
    print(f"  🚫 Excluded               : {tier_counts['excluded']}")
    print(f"  ❓ Unknown/empty          : {tier_counts['unknown']}")
    print()

    # device failure breakdown
    fail_deps = [d for d in deps if d["device_failure"] == "Yes"]
    if fail_deps:
        print(f"  ⚡ Device failure recorded: {len(fail_deps)}  ({', '.join(d['animal_id'] for d in fail_deps)})")
    print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NES deployment preparedness catalog")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    deps = scan_deployments()

    if args.json:
        print(json.dumps(deps, indent=2, default=str))
    else:
        print_report(deps)


if __name__ == "__main__":
    main()
