"""
presets.py — named HR/SR parameter presets for peak_detect_app.

Presets are keyed by a short slug like "oror-hr_v1". Built-in presets live
here as BUILTIN_PRESETS. Per-dataset saved presets are stored in the dataset's
parameter_log.json under __dataset_defaults__ -> settings -> named_presets.
"""

from __future__ import annotations
from typing import Dict, Any

# ── built-in presets ──────────────────────────────────────────────────────────

BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "oror-hr_v1": {
        "_label": "Orca HR v1",
        "_description": "Orca (oror) heart rate detection — broadband ECG, moderate cleanup",
        "_mode": "heart_rate",
        "BROAD_LOW_CUTOFF": 1.0,
        "BROAD_HIGH_CUTOFF": 25.0,
        "NARROW_LOW_CUTOFF": 5.0,
        "NARROW_HIGH_CUTOFF": 15.0,
        "FILTER_ORDER": 2,
        "SPIKE_THRESHOLD": 400,
        "SMOOTH_SEC_MULTIPLIER": 0.36,
        "WINDOW_SIZE_MULTIPLIER": 6.35,
        "PEAK_HEIGHT": -0.4,
        "PEAK_DISTANCE_SEC": 0.16,
        "SEARCH_RADIUS_SEC": 0.2,
        "MIN_PEAK_HEIGHT": 70,
        "MAX_PEAK_HEIGHT": 12000,
        "HR_JUMP_FRAC": 0.8,
        "MIN_RR_SEC": 0.25,
        "MAX_HR_BPM": 240,
        "MIN_HR_BPM": 0.1,
        "ANTI_DOUBLE_GAP_FACTOR": 0.75,
        "ANTI_DOUBLE_ROLLING_WINDOW_SEC": 10.0,
        "PICK_LAST_IN_CONFLICT_PAIR": True,
        "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],
        "enable_bandpass": True,
        "enable_spike_removal": True,
        "enable_absolute": True,
        "enable_smoothing": True,
        "enable_normalization": True,
        "enable_refinement": True,
    },
    "mian-hr_v1": {
        "_label": "NESE HR v1",
        "_description": "Northern elephant seal heart rate — higher HR range, tighter filters",
        "_mode": "heart_rate",
        "BROAD_LOW_CUTOFF": 1.0,
        "BROAD_HIGH_CUTOFF": 25.0,
        "NARROW_LOW_CUTOFF": 5.0,
        "NARROW_HIGH_CUTOFF": 20.0,
        "FILTER_ORDER": 2,
        "SPIKE_THRESHOLD": 600,
        "SMOOTH_SEC_MULTIPLIER": 0.2,
        "WINDOW_SIZE_MULTIPLIER": 5.0,
        "PEAK_HEIGHT": -0.3,
        "PEAK_DISTANCE_SEC": 0.1,
        "SEARCH_RADIUS_SEC": 0.15,
        "MIN_PEAK_HEIGHT": 50,
        "MAX_PEAK_HEIGHT": 15000,
        "HR_JUMP_FRAC": 0.8,
        "MIN_RR_SEC": 0.15,
        "MAX_HR_BPM": 300,
        "MIN_HR_BPM": 0.1,
        "ANTI_DOUBLE_GAP_FACTOR": 0.7,
        "ANTI_DOUBLE_ROLLING_WINDOW_SEC": 8.0,
        "PICK_LAST_IN_CONFLICT_PAIR": True,
        "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],
        "enable_bandpass": True,
        "enable_spike_removal": True,
        "enable_absolute": True,
        "enable_smoothing": True,
        "enable_normalization": True,
        "enable_refinement": True,
    },
    "apfo-hr_v1": {
        "_label": "Penguin HR v1",
        "_description": "African penguin heart rate — slower HR, aggressive spike removal",
        "_mode": "heart_rate",
        "BROAD_LOW_CUTOFF": 0.5,
        "BROAD_HIGH_CUTOFF": 15.0,
        "NARROW_LOW_CUTOFF": 2.0,
        "NARROW_HIGH_CUTOFF": 10.0,
        "FILTER_ORDER": 2,
        "SPIKE_THRESHOLD": 300,
        "SMOOTH_SEC_MULTIPLIER": 0.5,
        "WINDOW_SIZE_MULTIPLIER": 8.0,
        "PEAK_HEIGHT": -0.5,
        "PEAK_DISTANCE_SEC": 0.2,
        "SEARCH_RADIUS_SEC": 0.25,
        "MIN_PEAK_HEIGHT": 60,
        "MAX_PEAK_HEIGHT": 10000,
        "HR_JUMP_FRAC": 0.75,
        "MIN_RR_SEC": 0.2,
        "MAX_HR_BPM": 200,
        "MIN_HR_BPM": 0.1,
        "ANTI_DOUBLE_GAP_FACTOR": 0.7,
        "ANTI_DOUBLE_ROLLING_WINDOW_SEC": 12.0,
        "PICK_LAST_IN_CONFLICT_PAIR": True,
        "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],
        "enable_bandpass": True,
        "enable_spike_removal": True,
        "enable_absolute": True,
        "enable_smoothing": True,
        "enable_normalization": True,
        "enable_refinement": True,
    },
    "sr-generic_v1": {
        "_label": "Generic Stroke Rate v1",
        "_description": "Generic stroke rate for cetaceans — low-frequency fluke motion",
        "_mode": "stroke_rate",
        "BROAD_LOW_CUTOFF": 0.1,
        "BROAD_HIGH_CUTOFF": 5.0,
        "NARROW_LOW_CUTOFF": 0.5,
        "NARROW_HIGH_CUTOFF": 3.0,
        "FILTER_ORDER": 2,
        "SPIKE_THRESHOLD": 400,
        "SMOOTH_SEC_MULTIPLIER": 0.36,
        "WINDOW_SIZE_MULTIPLIER": 6.35,
        "PEAK_HEIGHT": -0.4,
        "PEAK_DISTANCE_SEC": 0.3,
        "SEARCH_RADIUS_SEC": 0.3,
        "MIN_PEAK_HEIGHT": 70,
        "MAX_PEAK_HEIGHT": 12000,
        "DETECTION_DERIVATIVE_CHANNELS": ["normalized"],
        "enable_bandpass": True,
        "enable_spike_removal": True,
        "enable_absolute": False,
        "enable_smoothing": True,
        "enable_normalization": True,
        "enable_refinement": True,
    },
}


def list_presets(mode: str = "heart_rate") -> list[dict]:
    """Return list of {value, label, description} for the given mode."""
    out = []
    for slug, params in BUILTIN_PRESETS.items():
        if params.get("_mode", "heart_rate") != mode:
            continue
        out.append({
            "value": slug,
            "label": params.get("_label", slug),
            "description": params.get("_description", ""),
        })
    return out


def get_preset(slug: str) -> dict | None:
    """Return a copy of preset params (without internal _ keys), or None."""
    raw = BUILTIN_PRESETS.get(slug)
    if raw is None:
        return None
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def suggest_preset_for_dataset(dataset_name: str, mode: str = "heart_rate") -> str | None:
    """Heuristic: pick the best built-in preset slug based on dataset name prefix."""
    name = dataset_name.lower()
    if mode == "stroke_rate":
        return "sr-generic_v1"
    if "oror" in name:
        return "oror-hr_v1"
    if "mian" in name or "nese" in name:
        return "mian-hr_v1"
    if "apfo" in name or "penguin" in name:
        return "apfo-hr_v1"
    return "oror-hr_v1"


def save_preset_to_param_manager(pm, slug: str, params: dict, label: str, mode: str):
    """Persist a user-defined preset into the dataset-level __dataset_defaults__."""
    named = {k: v for k, v in params.items() if not k.startswith("_")}
    named["_label"] = label
    named["_mode"] = mode
    # Load existing presets dict
    result = pm.get_from_config(["named_presets"], section="settings",
                                deployment_id=pm.DEFAULTS_DEPLOYMENT_ID)
    existing = result.get("named_presets") or {}
    existing[slug] = named
    pm.add_to_config(entries={"named_presets": existing}, section="settings",
                     deployment_id=pm.DEFAULTS_DEPLOYMENT_ID)


def load_dataset_presets(pm) -> dict:
    """Return {slug: params} dict of user-saved presets for this dataset."""
    result = pm.get_from_config(["named_presets"], section="settings",
                                deployment_id=pm.DEFAULTS_DEPLOYMENT_ID)
    return result.get("named_presets") or {}
