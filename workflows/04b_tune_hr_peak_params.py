"""
04b_tune_hr_peak_params.py  —  optional pre-step

Learn peak-detection thresholds from manually identified heartbeat timestamps
(heartbeat_manual_ok events in data.pkl / event_data) so that Step 05
(05_heartbeat_detect.py) has data-driven starting values.

What it does
------------
1. Loads the pkl for the target deployment.
2. Extracts manually-labelled heartbeat timestamps (heartbeat_manual_ok).
3. Runs the full peak_detect preprocessing pipeline (bandpass → spike-remove →
   abs → smooth → normalize) on the ECG signal, using the current parameter
   defaults.
4. For every manual timestamp it snaps to the nearest sample in the processed
   signal and reads off:
       - height_original  (smoothed signal value at that sample)
       - height_normalized (normalized signal value at that sample)
       - rr_interval_sec   (to the *next* manual beat)
5. Computes percentile-based thresholds:
       MIN_PEAK_HEIGHT  = p5  of height_original
       MAX_PEAK_HEIGHT  = p99 of height_original
       PEAK_HEIGHT      = p5  of height_normalized   (used by find_peaks)
       PEAK_DISTANCE_SEC = p5  of rr_interval_sec     (= min spacing allowed)
       MIN_RR_SEC        = p1  of rr_interval_sec     (post-detection cleanup)
       MAX_HR_BPM        = p99 of 60 / rr_interval_sec
       MIN_HR_BPM        = p1  of 60 / rr_interval_sec
6. Prints a summary table and writes the suggested values into the deployment's
   hr_peak_detection_settings section of parameter_log.json (if --write is
   passed) so Step 05 picks them up automatically.

Usage
-----
# dry-run (prints thresholds, does not write)
python3 workflows/04b_tune_hr_peak_params.py --dataset wild-whale-adult_hr-sr_JG-PP --deployment 2018-08-27_bamu-002

# write suggestions to parameter_log
python3 workflows/04b_tune_hr_peak_params.py --dataset wild-whale-adult_hr-sr_JG-PP --deployment 2018-08-27_bamu-002 --write
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyologger.utils.folder_manager import load_configuration, resolve_deployment_context
from pyologger.utils.time_manager import calculate_sampling_frequency
from pyologger.process_data.peak_detect import (
    bandpass_filter,
    remove_spikes,
    smooth_signal,
    sliding_window_normalization,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Learn HR peak-detection parameters from manual heartbeat labels.")
parser.add_argument("--dataset",    type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
parser.add_argument("--write",      action="store_true",
                    help="Write suggested parameters into parameter_log.json")
parser.add_argument("--snap-radius-ms", type=float, default=200.0,
                    help="Max ms to snap a manual timestamp to the nearest ECG sample (default 200 ms)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config + deployment context
# ---------------------------------------------------------------------------
config, data_dir, color_mapping_path, montage_path = load_configuration()

if args.dataset and args.deployment:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = \
        resolve_deployment_context(data_dir, dataset_id=args.dataset, deployment_id=args.deployment)
else:
    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = \
        resolve_deployment_context(data_dir)

pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
print(f"Loading {pkl_path} …")
with open(pkl_path, "rb") as f:
    data_pkl = pickle.load(f)

# ---------------------------------------------------------------------------
# Pull manual heartbeat timestamps
# ---------------------------------------------------------------------------
event_df = getattr(data_pkl, "event_data", None)
if event_df is None or event_df.empty:
    raise SystemExit("No event_data found in pkl — cannot tune parameters.")

manual_ok = event_df[event_df["key"] == "heartbeat_manual_ok"].copy()
if manual_ok.empty:
    raise SystemExit("No 'heartbeat_manual_ok' events found — nothing to learn from.")

manual_ok = manual_ok.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
print(f"Found {len(manual_ok)} manual heartbeat_ok events "
      f"({manual_ok['datetime'].min()} → {manual_ok['datetime'].max()})")

# ---------------------------------------------------------------------------
# Load ECG signal and compute sampling rate
# ---------------------------------------------------------------------------
ecg_df = data_pkl.signal_data.get("ecg")
if ecg_df is None:
    raise SystemExit("No 'ecg' signal found in data.pkl.")

ecg_signal   = ecg_df["ecg"].to_numpy(dtype=float)
ecg_datetime = ecg_df["datetime"]
fs           = calculate_sampling_frequency(ecg_datetime.head())
print(f"ECG: {len(ecg_signal):,} samples @ {fs:.1f} Hz")

# ---------------------------------------------------------------------------
# Run preprocessing (same chain as 05_heartbeat_detect / peak_detect)
# using current deployment parameters as defaults
# ---------------------------------------------------------------------------
base_params = param_manager.get_from_config(
    variable_names=[
        "BROAD_LOW_CUTOFF", "BROAD_HIGH_CUTOFF",
        "NARROW_LOW_CUTOFF", "NARROW_HIGH_CUTOFF",
        "FILTER_ORDER", "SPIKE_THRESHOLD",
        "SMOOTH_SEC_MULTIPLIER", "WINDOW_SIZE_MULTIPLIER",
        "NORMALIZATION_NOISE",
    ],
    section="hr_peak_detection_settings",
)
DEFAULT_PREPROCESS = {
    "BROAD_LOW_CUTOFF":      1.0,
    "BROAD_HIGH_CUTOFF":     35.0,
    "NARROW_LOW_CUTOFF":     5.0,
    "NARROW_HIGH_CUTOFF":    20.0,
    "FILTER_ORDER":          2,
    "SPIKE_THRESHOLD":       400,
    "SMOOTH_SEC_MULTIPLIER": 0.36,
    "WINDOW_SIZE_MULTIPLIER":6.35,
    "NORMALIZATION_NOISE":   1e-10,
}
pp = {k: (base_params.get(k) if base_params.get(k) is not None else v)
      for k, v in DEFAULT_PREPROCESS.items()}

print("\nPreprocessing parameters used:")
for k, v in pp.items():
    print(f"  {k}: {v}")

# Restrict preprocessing to the manual-annotation window to keep memory reasonable
t_start = pd.Timestamp(manual_ok["datetime"].min())
t_end   = pd.Timestamp(manual_ok["datetime"].max())

# ensure tz compatibility
if ecg_datetime.dt.tz is not None:
    tz = str(ecg_datetime.dt.tz)
    if t_start.tzinfo is None:
        t_start = t_start.tz_localize(tz)
    else:
        t_start = t_start.tz_convert(tz)
    if t_end.tzinfo is None:
        t_end = t_end.tz_localize(tz)
    else:
        t_end = t_end.tz_convert(tz)

window_mask = (ecg_datetime >= t_start) & (ecg_datetime <= t_end)
ecg_window  = ecg_signal[window_mask]
dt_window   = ecg_datetime[window_mask].reset_index(drop=True)

print(f"\nAnnotation window: {len(ecg_window):,} ECG samples "
      f"({t_start} → {t_end})")

print("Running bandpass filter …")
narrow_bp = bandpass_filter(ecg_window,
                            lowcut=pp["NARROW_LOW_CUTOFF"],
                            highcut=pp["NARROW_HIGH_CUTOFF"],
                            fs=fs,
                            order=int(pp["FILTER_ORDER"]))

print("Removing spikes …")
spikeless = remove_spikes(narrow_bp, threshold=pp["SPIKE_THRESHOLD"])

print("Smoothing (abs) …")
smoothed = smooth_signal(np.abs(spikeless),
                         smooth_sec=pp["SMOOTH_SEC_MULTIPLIER"],
                         fs=fs)

print("Normalizing …")
normalized = sliding_window_normalization(smoothed,
                                          int(pp["WINDOW_SIZE_MULTIPLIER"] * fs),
                                          noise=pp["NORMALIZATION_NOISE"])

# ---------------------------------------------------------------------------
# Snap each manual timestamp to the nearest ECG sample in the window
# ---------------------------------------------------------------------------
snap_radius_samples = int(args.snap_radius_ms * 1e-3 * fs)

# Convert manual datetimes to the same tz
manual_dts = pd.to_datetime(manual_ok["datetime"])
if ecg_datetime.dt.tz is not None:
    tz = str(ecg_datetime.dt.tz)
    manual_dts = manual_dts.dt.tz_convert(tz) if manual_dts.dt.tz is not None else manual_dts.dt.tz_localize(tz)

dt_window_ns = dt_window.values.astype("int64")   # nanoseconds

records = []
snap_failures = 0
for ts in manual_dts:
    ts_ns = int(pd.Timestamp(ts).value)
    diff   = np.abs(dt_window_ns - ts_ns)
    nearest_idx = int(np.argmin(diff))
    nearest_diff_ms = diff[nearest_idx] / 1e6
    if nearest_diff_ms > args.snap_radius_ms:
        snap_failures += 1
        continue
    records.append({
        "manual_datetime":   ts,
        "ecg_sample_idx":    nearest_idx,
        "snap_error_ms":     nearest_diff_ms,
        "height_original":   float(smoothed[nearest_idx]),
        "height_normalized": float(normalized[nearest_idx]),
    })

snap_df = pd.DataFrame(records).sort_values("manual_datetime").reset_index(drop=True)
print(f"\nSnapped {len(snap_df)} / {len(manual_ok)} manual beats to ECG samples "
      f"({snap_failures} outside ±{args.snap_radius_ms:.0f} ms window)")
if snap_df.empty:
    raise SystemExit("No manual beats could be snapped to the ECG — check time zones or snap radius.")

# ---------------------------------------------------------------------------
# Compute RR intervals between consecutive snapped beats
# ---------------------------------------------------------------------------
snap_df["rr_interval_sec"] = (
    snap_df["ecg_sample_idx"].diff() / fs
).where(snap_df["ecg_sample_idx"].diff() > 0)

rr_vals   = snap_df["rr_interval_sec"].dropna().to_numpy()
hr_vals   = 60.0 / rr_vals

h_orig    = snap_df["height_original"].to_numpy()
h_norm    = snap_df["height_normalized"].to_numpy()

# ---------------------------------------------------------------------------
# Compute percentile-based suggestions
# ---------------------------------------------------------------------------
def pct(arr, p):
    return float(np.percentile(arr[np.isfinite(arr)], p))

suggestions = {
    "MIN_PEAK_HEIGHT":    round(pct(h_orig,   5), 1),
    "MAX_PEAK_HEIGHT":    round(pct(h_orig,  99), 1),
    "PEAK_HEIGHT":        round(pct(h_norm,   5), 4),
    "PEAK_DISTANCE_SEC":  round(pct(rr_vals,  5), 3),
    "MIN_RR_SEC":         round(pct(rr_vals,  1), 3),
    "MAX_HR_BPM":         round(pct(hr_vals, 99), 1),
    "MIN_HR_BPM":         round(pct(hr_vals,  1), 2),
}

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"  ECG peak-detection parameter suggestions")
print(f"  Deployment : {deployment_id}")
print(f"  N beats    : {len(snap_df)}")
print(f"  N RR pairs : {len(rr_vals)}")
print("=" * 60)

col_w = 26
print(f"  {'Parameter':<{col_w}} {'Suggested':>12}   {'Basis'}")
print(f"  {'-'*col_w}  {'-'*12}   {'-'*30}")

basis = {
    "MIN_PEAK_HEIGHT":   "p5  of smoothed height at manual beats",
    "MAX_PEAK_HEIGHT":   "p99 of smoothed height at manual beats",
    "PEAK_HEIGHT":       "p5  of normalized height at manual beats",
    "PEAK_DISTANCE_SEC": "p5  of RR interval (min spacing)",
    "MIN_RR_SEC":        "p1  of RR interval (cleanup floor)",
    "MAX_HR_BPM":        "p99 of instantaneous HR (bpm)",
    "MIN_HR_BPM":        "p1  of instantaneous HR (bpm)",
}
for k, v in suggestions.items():
    print(f"  {k:<{col_w}} {v:>12.4g}   {basis[k]}")

print()
print("Distribution summary (height_original at manual beats):")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  p{p:02d}: {pct(h_orig, p):.2f}")

print()
print("Distribution summary (RR interval, s):")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  p{p:02d}: {pct(rr_vals, p):.3f}  ({60/pct(rr_vals,p):.1f} bpm)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Optionally write to parameter_log
# ---------------------------------------------------------------------------
if args.write:
    print("\nWriting suggestions to parameter_log (hr_peak_detection_settings) …")
    param_manager.add_to_config(entries=suggestions, section="hr_peak_detection_settings")
    print("Done — Step 05 will pick up these values on next run.")
else:
    print("\nDry run — pass --write to persist these into hr_peak_detection_settings.")
