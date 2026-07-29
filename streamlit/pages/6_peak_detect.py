import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import pytz
import time
import os
import plotly.graph_objects as go

# Import pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.data_manager import *
from pyologger.calibrate_data.zoc import *
from pyologger.plot_data.plotter import *
from pyologger.process_data.sampling import *
from pyologger.process_data.peak_detect import *
from pyologger.utils.streamlit_time_window import standardize_time_settings

# Load important file paths and configurations
config, data_dir, color_mapping_path, montage_path = load_configuration()

# **Step 2: Deployment Selection (Dropdown Menus)**
st.sidebar.title("Deployment Selection")

# Streamlit load data
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment_streamlit(data_dir)
timezone = data_pkl.deployment_info.get('Time Zone', 'UTC')
DEPLOYMENT_TZ = timezone or "UTC"
DISPLAY_TZ = DEPLOYMENT_TZ

st.sidebar.write(f"📂 Selected Deployment: {deployment_id}")


def _init_timing():
    st.session_state["peak_detect_timing"] = []
    return time.perf_counter()


def _log_timing(start_ts, label):
    elapsed = time.perf_counter() - start_ts
    msg = f"{label}: {elapsed:.2f}s"
    st.session_state.setdefault("peak_detect_timing", []).append(msg)
    print(f"[peak-detect] {msg}")
    return time.perf_counter()


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return bool(default)


def _safe_int(value, fallback=30):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _coerce_string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
        return [v for v in items if v]
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _cleanup_heartbeat_peaks(results, signal_subset_df, params, sampling_rate, parent_signal):
    """
    Apply heartbeat cleanup heuristics (suggestions, jump-based rejections, gap detection)
    and return diagnostics for rejection attribution.
    """
    diag = {
        "rejected_below_min_peak_height": 0,
        "rejected_above_max_peak_height": 0,
        "rejected_conflict_pair_score": 0,
        "rejected_up_jump_cleanup": 0,
        "rejected_rr_too_short": 0,
        "suggested_added": 0,
        "final_accepted_or_suggested": 0,
        "final_rejected": 0,
    }
    cleanup_rejected_indices = set()

    if "peak_df" not in results or "smoothed" not in results:
        return results, [], diag

    peak_df = results["peak_df"].copy().sort_values("refined_index").reset_index(drop=True)
    smoothed = np.asarray(results["smoothed"])
    fs = float(sampling_rate)

    min_peak = _safe_float(params.get("MIN_PEAK_HEIGHT"))
    max_peak = _safe_float(params.get("MAX_PEAK_HEIGHT"))
    if "height_original" in peak_df.columns:
        if min_peak is not None:
            diag["rejected_below_min_peak_height"] = int((peak_df["height_original"] < min_peak).sum())
        if max_peak is not None:
            diag["rejected_above_max_peak_height"] = int((peak_df["height_original"] > max_peak).sum())

    hr_jump_frac = _safe_float(params.get("HR_JUMP_FRAC")) or 0.8
    min_rr_sec = _safe_float(params.get("MIN_RR_SEC")) or 0.25
    max_hr_bpm = _safe_float(params.get("MAX_HR_BPM")) or 240.0
    min_hr_bpm = _safe_float(params.get("MIN_HR_BPM")) or 0.1
    up_jump_search_radius_sec = _safe_float(params.get("SEARCH_RADIUS_SEC")) or 0.2
    up_jump_search_radius_samples = max(1, int(round(up_jump_search_radius_sec * fs)))
    min_suggested_peak_height = _safe_float(params.get("MIN_PEAK_HEIGHT")) or 0.0
    conflict_rr_factor = _safe_float(params.get("ANTI_DOUBLE_GAP_FACTOR"))
    if conflict_rr_factor is None:
        # Backward compatibility with older config key name.
        conflict_rr_factor = _safe_float(params.get("HR_CONFLICT_RR_FACTOR"))
    if conflict_rr_factor is None:
        conflict_rr_factor = 0.75
    conflict_rolling_window_sec = _safe_float(params.get("ANTI_DOUBLE_ROLLING_WINDOW_SEC")) or 10.0
    if conflict_rolling_window_sec <= 0:
        conflict_rolling_window_sec = 10.0
    conflict_local_neighbors = 30
    pick_last_in_conflict_pair = _safe_bool(params.get("PICK_LAST_IN_CONFLICT_PAIR"), default=True)

    accepted = peak_df[peak_df["key"] == "beat_auto_detect_accepted"].sort_values("refined_index").reset_index(drop=True)
    if len(accepted) <= 1:
        results["peak_df"] = peak_df
        diag["final_accepted_or_suggested"] = int((peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"])).sum())
        diag["final_rejected"] = int((peak_df["key"] == "beat_auto_detect_rejected").sum())
        return results, [], diag

    rr_sec = np.diff(accepted["refined_index"].to_numpy()) / fs
    inst_hr = 60.0 / rr_sec
    hr_df = pd.DataFrame({
        "idx": accepted["refined_index"].iloc[1:].to_numpy(),
        "hr_bpm": inst_hr,
    })
    hr_df["prev_hr_bpm"] = hr_df["hr_bpm"].shift(1)
    hr_df["frac_change"] = (hr_df["hr_bpm"] - hr_df["prev_hr_bpm"]) / hr_df["prev_hr_bpm"]
    big_jump_mask = hr_df["prev_hr_bpm"].notna() & (hr_df["frac_change"].abs() > hr_jump_frac)
    down_jumps = hr_df[big_jump_mask & (hr_df["frac_change"] < 0)].copy()

    signal_len = int(len(smoothed))
    accepted_indices = accepted["refined_index"].astype(int).to_numpy()
    # Guard against out-of-range/invalid indices from prior processing steps.
    accepted_indices = accepted_indices[(accepted_indices >= 0) & (accepted_indices < signal_len)]
    accepted_indices = np.unique(accepted_indices)

    suggested_rows = []
    for _, row in down_jumps.iterrows():
        this_idx = int(row["idx"])
        prev_candidates = accepted[accepted["refined_index"] < this_idx]["refined_index"]
        if prev_candidates.empty:
            continue
        prev_idx = int(prev_candidates.max())
        if prev_idx < 0 or prev_idx >= signal_len:
            continue
        search_sig = smoothed[prev_idx:this_idx]
        if len(search_sig) < 3:
            continue

        local_abs_idx = prev_idx + int(np.argmax(search_sig))
        if local_abs_idx < 0 or local_abs_idx >= signal_len:
            continue
        local_val = float(smoothed[local_abs_idx])
        if local_val < min_suggested_peak_height:
            continue

        suggested_rows.append({
            "refined_index": local_abs_idx,
            "height_original": local_val,
            "height_normalized": np.nan,
            "datetime": signal_subset_df["datetime"].iloc[local_abs_idx] if "datetime" in signal_subset_df else pd.NaT,
            "key": "beat_auto_detect_suggested",
        })

    if suggested_rows:
        peak_df = pd.concat([peak_df, pd.DataFrame(suggested_rows)], ignore_index=True)
        peak_df = peak_df.sort_values("refined_index").reset_index(drop=True)
    diag["suggested_added"] = len(suggested_rows)

    accepted_scored = peak_df[
        peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"])
    ].sort_values("refined_index").reset_index(drop=True)

    if len(accepted_scored) > 1:
        idxs = accepted_scored["refined_index"].astype(int).to_numpy()
        rr_all = np.diff(idxs) / fs if len(idxs) > 1 else np.array([])
        interval_midpoints = (idxs[:-1].astype(float) + idxs[1:].astype(float)) / 2.0
        half_window_samples = max(1.0, (conflict_rolling_window_sec * fs) / 2.0)

        for i in range(len(idxs) - 1):
            a = int(idxs[i])
            b = int(idxs[i + 1])
            rr = (b - a) / fs
            pair_mid = (a + b) / 2.0
            local_mask = np.abs(interval_midpoints - pair_mid) <= half_window_samples
            local_rr = rr_all[local_mask]
            local_rr = local_rr[np.isfinite(local_rr) & (local_rr > 0)]
            if local_rr.size == 0:
                lo = max(0, i - conflict_local_neighbors)
                hi = min(len(rr_all), i + conflict_local_neighbors + 1)
                local_rr = rr_all[lo:hi]
                local_rr = local_rr[np.isfinite(local_rr) & (local_rr > 0)]
            if local_rr.size == 0:
                continue
            rr_ref = float(np.median(local_rr))
            if rr >= (conflict_rr_factor * rr_ref):
                continue

            ka = peak_df.loc[peak_df["refined_index"] == a, "key"]
            kb = peak_df.loc[peak_df["refined_index"] == b, "key"]
            if ka.empty or kb.empty:
                continue
            if ka.iloc[0] not in {"beat_auto_detect_accepted", "beat_auto_detect_suggested"}:
                continue
            if kb.iloc[0] not in {"beat_auto_detect_accepted", "beat_auto_detect_suggested"}:
                continue

            # Deterministic conflict resolution: keep later peak by default.
            reject_idx = a if pick_last_in_conflict_pair else b
            peak_df.loc[peak_df["refined_index"] == reject_idx, "key"] = "beat_auto_detect_rejected"
            cleanup_rejected_indices.add(int(reject_idx))
            diag["rejected_conflict_pair_score"] += 1

    # Recompute jump table after conflict correction so up-jump cleanup uses corrected beat sequence.
    accepted_after_conflict = peak_df[
        peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"])
    ].sort_values("refined_index").reset_index(drop=True)
    if len(accepted_after_conflict) > 1:
        rr_sec2 = np.diff(accepted_after_conflict["refined_index"].to_numpy()) / fs
        inst_hr2 = 60.0 / rr_sec2
        hr_df2 = pd.DataFrame(
            {
                "idx": accepted_after_conflict["refined_index"].iloc[1:].to_numpy(),
                "hr_bpm": inst_hr2,
            }
        )
        hr_df2["prev_hr_bpm"] = hr_df2["hr_bpm"].shift(1)
        hr_df2["frac_change"] = (hr_df2["hr_bpm"] - hr_df2["prev_hr_bpm"]) / hr_df2["prev_hr_bpm"]
        big_jump_mask2 = hr_df2["prev_hr_bpm"].notna() & (hr_df2["frac_change"].abs() > hr_jump_frac)
        up_jumps = hr_df2[big_jump_mask2 & (hr_df2["frac_change"] > 0)].copy()
    else:
        up_jumps = pd.DataFrame()

    nan_segments = []
    for _, row in up_jumps.iterrows():
        this_idx = int(row["idx"])
        this_is_active = ((peak_df["refined_index"] == this_idx) &
                          (peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"]))).any()
        if not this_is_active:
            continue
        upjump_prev_candidates = peak_df[
            (peak_df["refined_index"] < this_idx) &
            (peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"]))
        ]["refined_index"]
        prev_upjump_idx = int(upjump_prev_candidates.max()) if not upjump_prev_candidates.empty else None
        # For up-jumps, enforce pair resolution first: keep later peak by default.
        if prev_upjump_idx is not None:
            reject_idx = prev_upjump_idx if pick_last_in_conflict_pair else this_idx
        else:
            lo = max(0, this_idx - up_jump_search_radius_samples)
            hi = min(signal_len - 1, this_idx + up_jump_search_radius_samples)
            local_search_sig = smoothed[lo:hi + 1]
            if local_search_sig.size == 0:
                reject_idx = this_idx
            else:
                local_max_idx = lo + int(np.argmax(local_search_sig))
                candidates = peak_df[
                    (peak_df["refined_index"] >= lo) &
                    (peak_df["refined_index"] <= hi) &
                    (peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"]))
                ]["refined_index"].astype(int).to_numpy()
                if candidates.size == 0:
                    reject_idx = this_idx
                else:
                    reject_idx = int(candidates[np.argmin(np.abs(candidates - local_max_idx))])

        matched = peak_df["refined_index"] == reject_idx
        if matched.any():
            changed = int((peak_df.loc[matched, "key"] != "beat_auto_detect_rejected").sum())
            diag["rejected_up_jump_cleanup"] += changed
            peak_df.loc[matched, "key"] = "beat_auto_detect_rejected"
            cleanup_rejected_indices.add(int(reject_idx))

        later = peak_df[
            (peak_df["refined_index"] > reject_idx) &
            (peak_df["key"].str.contains("accepted|suggested", na=False))
        ].sort_values("refined_index")
        if later.empty:
            continue
        next_idx = int(later.iloc[0]["refined_index"])
        prev_candidates = peak_df[
            (peak_df["refined_index"] < reject_idx) &
            (peak_df["key"].str.contains("accepted|suggested", na=False))
        ]["refined_index"]
        if prev_candidates.empty:
            continue
        prev_idx = int(prev_candidates.max())
        rr_fixed_sec = (next_idx - prev_idx) / fs
        if rr_fixed_sec < min_rr_sec:
            nan_segments.append((prev_idx, next_idx))
            diag["rejected_rr_too_short"] += 1

    # Run anti-double once more after up-jump cleanup on the cleaned sequence.
    accepted_post_upjump = peak_df[
        peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"])
    ].sort_values("refined_index").reset_index(drop=True)
    if len(accepted_post_upjump) > 1:
        idxs2 = accepted_post_upjump["refined_index"].astype(int).to_numpy()
        rr_all2 = np.diff(idxs2) / fs if len(idxs2) > 1 else np.array([])
        interval_midpoints2 = (idxs2[:-1].astype(float) + idxs2[1:].astype(float)) / 2.0
        half_window_samples2 = max(1.0, (conflict_rolling_window_sec * fs) / 2.0)
        for i in range(len(idxs2) - 1):
            a = int(idxs2[i])
            b = int(idxs2[i + 1])
            rr = (b - a) / fs
            pair_mid = (a + b) / 2.0
            local_mask = np.abs(interval_midpoints2 - pair_mid) <= half_window_samples2
            local_rr = rr_all2[local_mask]
            local_rr = local_rr[np.isfinite(local_rr) & (local_rr > 0)]
            if local_rr.size == 0:
                lo = max(0, i - conflict_local_neighbors)
                hi = min(len(rr_all2), i + conflict_local_neighbors + 1)
                local_rr = rr_all2[lo:hi]
                local_rr = local_rr[np.isfinite(local_rr) & (local_rr > 0)]
            if local_rr.size == 0:
                continue
            rr_ref = float(np.median(local_rr))
            if rr >= (conflict_rr_factor * rr_ref):
                continue

            ka = peak_df.loc[peak_df["refined_index"] == a, "key"]
            kb = peak_df.loc[peak_df["refined_index"] == b, "key"]
            if ka.empty or kb.empty:
                continue
            if ka.iloc[0] not in {"beat_auto_detect_accepted", "beat_auto_detect_suggested"}:
                continue
            if kb.iloc[0] not in {"beat_auto_detect_accepted", "beat_auto_detect_suggested"}:
                continue

            reject_idx = a if pick_last_in_conflict_pair else b
            matched = peak_df["refined_index"] == reject_idx
            if not matched.any():
                continue
            changed = int((peak_df.loc[matched, "key"] != "beat_auto_detect_rejected").sum())
            if changed:
                peak_df.loc[matched, "key"] = "beat_auto_detect_rejected"
                cleanup_rejected_indices.add(int(reject_idx))
                diag["rejected_conflict_pair_score"] += changed

    # Build heart_rate_fixed from accepted+suggested with explicit NaN gaps.
    peak_for_hr = peak_df[peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"])].sort_values("refined_index").reset_index(drop=True)
    n = len(signal_subset_df)
    hr_series = np.full(n, np.nan, dtype=float)
    if len(peak_for_hr) > 1:
        for i in range(len(peak_for_hr) - 1):
            s = int(peak_for_hr["refined_index"].iloc[i])
            e = int(peak_for_hr["refined_index"].iloc[i + 1])
            if e <= s:
                continue
            rr = (e - s) / fs
            if rr < min_rr_sec:
                continue
            hr_val = 60.0 / rr
            hr_series[s:e] = max(min(hr_val, max_hr_bpm), min_hr_bpm)

    for s, e in nan_segments:
        hr_series[s:e] = np.nan

    heart_rate_fixed_vals = pd.Series(hr_series).interpolate(limit_direction="both").to_numpy()
    heart_rate_fixed_vals = np.clip(heart_rate_fixed_vals, min_hr_bpm, max_hr_bpm)
    heart_rate_fixed = pd.DataFrame({
        "datetime": signal_subset_df["datetime"],
        "heart_rate_fixed": heart_rate_fixed_vals,
    })

    results["peak_df"] = peak_df
    results["cleanup_nan_segments"] = nan_segments
    results["heart_rate_fixed_df"] = heart_rate_fixed
    results["cleanup_rejected_indices"] = sorted(cleanup_rejected_indices)

    diag["final_accepted_or_suggested"] = int((peak_df["key"].isin(["beat_auto_detect_accepted", "beat_auto_detect_suggested"])).sum())
    diag["final_rejected"] = int((peak_df["key"] == "beat_auto_detect_rejected").sum())

    return results, nan_segments, diag


def _upsert_heartbeat_cleanup_events(data_pkl, results, signal_subset_df):
    if not hasattr(data_pkl, "event_data") or data_pkl.event_data is None:
        data_pkl.event_data = pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value"])

    cleanup_keys = {
        "heartbeat_auto_detect_suggested",
        "heartbeat_auto_detect_gap",
        "heartbeat_auto_detect_cleanup_rejected",
    }
    if "key" in data_pkl.event_data.columns:
        data_pkl.event_data = data_pkl.event_data[~data_pkl.event_data["key"].isin(cleanup_keys)].copy()

    peak_df = results.get("peak_df", pd.DataFrame())
    suggested = peak_df[peak_df.get("key", pd.Series(dtype=str)) == "beat_auto_detect_suggested"].copy()
    suggested_events = []
    for _, row in suggested.iterrows():
        suggested_events.append({
            "datetime": row.get("datetime", pd.NaT),
            "key": "heartbeat_auto_detect_suggested",
            "short_description": "auto-detected heartbeat (suggested, missed-beat fix)",
            "type": "point",
            "duration": 0.0,
            "value": np.nan,
        })

    cleanup_rejected_events = []
    peak_df = results.get("peak_df", pd.DataFrame())
    cleanup_rej_set = set(results.get("cleanup_rejected_indices", []))
    if not peak_df.empty and cleanup_rej_set:
        rej_rows = peak_df[peak_df.get("refined_index", pd.Series(dtype=int)).isin(cleanup_rej_set)]
        for _, row in rej_rows.iterrows():
            cleanup_rejected_events.append(
                {
                    "datetime": row.get("datetime", pd.NaT),
                    "key": "heartbeat_auto_detect_cleanup_rejected",
                    "short_description": "auto-detected heartbeat rejected by cleanup",
                    "type": "point",
                    "duration": 0.0,
                    "value": np.nan,
                }
            )

    gap_events = []
    for s, e in results.get("cleanup_nan_segments", []):
        dt_start = signal_subset_df["datetime"].iloc[s]
        dt_end = signal_subset_df["datetime"].iloc[min(e - 1, len(signal_subset_df) - 1)]
        duration_sec = (dt_end - dt_start).total_seconds()
        gap_events.append({
            "datetime": dt_start,
            "key": "heartbeat_auto_detect_gap",
            "short_description": "interval where HR was invalid (>50% jump / too-short RR)",
            "type": "interval_start",
            "duration": duration_sec,
            "value": np.nan,
        })
        gap_events.append({
            "datetime": dt_end,
            "key": "heartbeat_auto_detect_gap",
            "short_description": "interval where HR was invalid (>50% jump / too-short RR): end",
            "type": "interval_end",
            "duration": duration_sec,
            "value": np.nan,
        })

    appended = pd.DataFrame(suggested_events + cleanup_rejected_events + gap_events)
    if not appended.empty:
        data_pkl.event_data = pd.concat([data_pkl.event_data, appended], ignore_index=True)

    if not hasattr(data_pkl, "event_manager"):
        data_pkl.event_manager = {}
    data_pkl.event_manager["heart_rate"] = {
        "keys": [
            "heartbeat_auto_detect_accepted",
            "heartbeat_auto_detect_rejected",
            "heartbeat_auto_detect_cleanup_rejected",
            "heartbeat_auto_detect_suggested",
            "heartbeat_auto_detect_gap",
        ],
        "description": "Events related to heartbeat detection and HR gap handling",
        "color_map": {
            "heartbeat_auto_detect_accepted": "#4caf50",
            "heartbeat_auto_detect_rejected": "#f44336",
            "heartbeat_auto_detect_cleanup_rejected": "#ff9800",
            "heartbeat_auto_detect_suggested": "#ffeb3b",
            "heartbeat_auto_detect_gap": "#9e9e9e",
        },
    }

def _get_middle_acc_window_defaults(data_pkl_obj, minutes=10):
    """Build a centered fallback time window from available accelerometer-like signals."""
    acc_candidates = ["corrected_acc", "calibrated_acc", "accelerometer"]
    source_signal = next((s for s in acc_candidates if s in data_pkl_obj.signal_data), None)
    if source_signal is None:
        # Fallback to any signal with a datetime column.
        for sig_name, sig_df in data_pkl_obj.signal_data.items():
            if isinstance(sig_df, pd.DataFrame) and "datetime" in sig_df.columns and not sig_df.empty:
                source_signal = sig_name
                break
    if source_signal is None:
        raise ValueError("No signal with datetime data available to build fallback time window.")

    dt = pd.to_datetime(data_pkl_obj.signal_data[source_signal]["datetime"], errors="coerce").dropna()
    if dt.empty:
        raise ValueError(f"Signal '{source_signal}' has no valid datetime values for fallback window.")

    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(DEPLOYMENT_TZ)
    else:
        dt = dt.dt.tz_convert(DEPLOYMENT_TZ)

    sig_start = dt.min()
    sig_end = dt.max()
    midpoint = sig_start + (sig_end - sig_start) / 2
    half_window = pd.Timedelta(minutes=minutes / 2)
    win_start = midpoint - half_window
    win_end = midpoint + half_window

    # Clamp if signal span is shorter than desired window.
    if win_start < sig_start:
        shift = sig_start - win_start
        win_start += shift
        win_end += shift
    if win_end > sig_end:
        shift = win_end - sig_end
        win_start -= shift
        win_end -= shift
    win_start = max(win_start, sig_start)
    win_end = min(win_end, sig_end)

    return {
        "overlap_start_time": str(win_start),
        "overlap_end_time": str(win_end),
        "zoom_window_start_time": str(win_start),
        "zoom_window_end_time": str(win_end),
        "_source_signal": source_signal,
    }


def _to_deployment_timestamp(value):
    """Normalize saved timestamps into deployment timezone."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(DEPLOYMENT_TZ)
    return ts.tz_convert(DEPLOYMENT_TZ)


def _to_display_naive(value):
    """Return display wall-clock time (deployment local or UTC) without tzinfo."""
    ts = _to_deployment_timestamp(value).tz_convert(DISPLAY_TZ)
    return ts.tz_localize(None)


def _display_naive_to_deployment_aware(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(DISPLAY_TZ)
    return ts.tz_convert(DEPLOYMENT_TZ)


standardized = standardize_time_settings(
    param_manager=param_manager,
    data_pkl=data_pkl,
    tz_name=str(DEPLOYMENT_TZ),
    minutes=2,
    persist=False,
)
page_t0 = _init_timing()
page_t0 = _log_timing(page_t0, "Time window standardization")
OVERLAP_START_TIME = standardized["overlap_start_time"]
OVERLAP_END_TIME = standardized["overlap_end_time"]
ZOOM_WINDOW_START_TIME = standardized["zoom_window_start_time"]
ZOOM_WINDOW_END_TIME = standardized["zoom_window_end_time"]
OVERLAP_START_DISPLAY = _to_display_naive(OVERLAP_START_TIME)
OVERLAP_END_DISPLAY = _to_display_naive(OVERLAP_END_TIME)
ZOOM_WINDOW_START_DISPLAY = _to_display_naive(ZOOM_WINDOW_START_TIME)
ZOOM_WINDOW_END_DISPLAY = _to_display_naive(ZOOM_WINDOW_END_TIME)

DETECTION_PARAM_KEYS = [
    "BROAD_LOW_CUTOFF", "BROAD_HIGH_CUTOFF", "NARROW_LOW_CUTOFF", "NARROW_HIGH_CUTOFF",
    "FILTER_ORDER", "SPIKE_THRESHOLD", "SMOOTH_SEC_MULTIPLIER", "WINDOW_SIZE_MULTIPLIER",
    "NORMALIZATION_NOISE", "PEAK_HEIGHT", "PEAK_DISTANCE_SEC", "SEARCH_RADIUS_SEC",
    "MIN_PEAK_HEIGHT", "MAX_PEAK_HEIGHT", "enable_bandpass", "enable_spike_removal",
    "enable_absolute", "enable_smoothing", "enable_normalization", "enable_refinement",
    "HR_JUMP_FRAC", "MIN_RR_SEC", "MAX_HR_BPM", "MIN_HR_BPM",
    "ANTI_DOUBLE_GAP_FACTOR",
    "ANTI_DOUBLE_ROLLING_WINDOW_SEC",
    "PICK_LAST_IN_CONFLICT_PAIR",
    "HR_CONFLICT_RR_FACTOR",
    "DETECTION_DERIVATIVE_CHANNELS",
    "STROKE_PARENT_SIGNAL", "STROKE_CHANNEL",
]


def _get_detection_params_cached(mode):
    section = "hr_peak_detection_settings" if mode == "heart_rate" else "stroke_peak_detection_settings"
    cache_bucket = st.session_state.setdefault("peak_detect_param_cache", {})
    cache_key = f"{dataset_id}::{deployment_id}::{section}"
    config_path = getattr(param_manager, "config_log_path", None)
    config_mtime = None
    if config_path and os.path.exists(config_path):
        config_mtime = os.path.getmtime(config_path)

    cached = cache_bucket.get(cache_key)
    if cached and cached.get("mtime") == config_mtime:
        return cached["params"].copy(), section

    loaded = param_manager.get_from_config(variable_names=DETECTION_PARAM_KEYS, section=section)
    cache_bucket[cache_key] = {"mtime": config_mtime, "params": loaded.copy()}
    return loaded, section


def get_pipeline_default_params(mode):
    """Defaults aligned with workflow pipelines when config values are missing."""
    if mode == "stroke_rate":
        # Mirrors defaults in workflows/04_stroke_detect.py
        return {
            "BROAD_LOW_CUTOFF": 0.05,
            "BROAD_HIGH_CUTOFF": 10.0,
            "NARROW_LOW_CUTOFF": 0.1,
            "NARROW_HIGH_CUTOFF": 2.0,
            "FILTER_ORDER": 2,
            "SPIKE_THRESHOLD": 400,
            "SMOOTH_SEC_MULTIPLIER": 0.41,
            "WINDOW_SIZE_MULTIPLIER": 15.5,
            "NORMALIZATION_NOISE": 0.02,
            "PEAK_HEIGHT": -0.9,
            "PEAK_DISTANCE_SEC": 0.5,
            "SEARCH_RADIUS_SEC": 0.3,
            "MIN_PEAK_HEIGHT": 150.0,
            "MAX_PEAK_HEIGHT": 1000000.0,
            "enable_bandpass": True,
            "enable_spike_removal": False,
            "enable_absolute": False,
            "enable_smoothing": True,
            "enable_normalization": True,
            "enable_refinement": True,
        }
    # Mirrors defaults in workflows/05_heartbeat_detect.py
    return {
        "BROAD_LOW_CUTOFF": 1.0,
        "BROAD_HIGH_CUTOFF": 35.0,
        "NARROW_LOW_CUTOFF": 5.0,
        "NARROW_HIGH_CUTOFF": 20.0,
        "FILTER_ORDER": 2,
        "SPIKE_THRESHOLD": 400,
        "SMOOTH_SEC_MULTIPLIER": 0.36,
        "WINDOW_SIZE_MULTIPLIER": 6.35,
        "NORMALIZATION_NOISE": 0.02,
        "PEAK_HEIGHT": -0.4,
        "PEAK_DISTANCE_SEC": 0.16,
        "SEARCH_RADIUS_SEC": 0.2,
        "MIN_PEAK_HEIGHT": 70.0,
        "MAX_PEAK_HEIGHT": 12000.0,
        "enable_bandpass": True,
        "enable_spike_removal": True,
        "enable_absolute": True,
        "enable_smoothing": True,
        "enable_normalization": True,
        "enable_refinement": True,
        "HR_JUMP_FRAC": 0.8,
        "MIN_RR_SEC": 0.25,
        "MAX_HR_BPM": 240.0,
        "MIN_HR_BPM": 0.1,
        "ANTI_DOUBLE_GAP_FACTOR": 0.75,
        "ANTI_DOUBLE_ROLLING_WINDOW_SEC": 10.0,
        "PICK_LAST_IN_CONFLICT_PAIR": True,
        "HR_CONFLICT_RR_FACTOR": 0.75,
    }


def _resolve_channel_alias(requested_channel, available_channels):
    if requested_channel in available_channels:
        return requested_channel
    req_text = str(requested_channel)
    req_base = req_text.split("__", 1)[0]

    # Prefer base-name match so stored config can remain simple (e.g., "gy").
    for ch in available_channels:
        if str(ch).split("__", 1)[0] == req_base:
            return ch

    alias_map = {
        "ax": "x", "ay": "y", "az": "z",
        "x": "ax", "y": "ay", "z": "az",
        "gx": "x", "gy": "y", "gz": "z",
    }
    alias = alias_map.get(req_base.lower())
    if alias:
        for ch in available_channels:
            ch_base = str(ch).split("__", 1)[0]
            if ch_base == alias:
                return ch
    return requested_channel


def _channel_label(channel_name: str) -> str:
    return str(channel_name).split("__", 1)[0]

# User selection for detection type
st.sidebar.title("Detection Mode")
detection_mode = st.sidebar.selectbox("Choose mode:", ["stroke_rate", "heart_rate"])
widget_prefix = f"{dataset_id}_{deployment_id}_{detection_mode}".replace(":", "_").replace("/", "_")
section = "hr_peak_detection_settings" if detection_mode == "heart_rate" else "stroke_peak_detection_settings"
pipeline_defaults = get_pipeline_default_params(detection_mode)
load_saved_params = st.sidebar.checkbox(
    "Load saved params from config",
    value=True,
    key=f"{widget_prefix}_load_saved_params",
    help="Enable only if you need deployment-saved values. Disabled keeps the page responsive.",
)

if load_saved_params:
    # Load params once per deployment/mode and reuse across reruns when config is unchanged.
    default_params, section = _get_detection_params_cached(detection_mode)
    page_t0 = _log_timing(page_t0, "Loaded peak-detect params from config")
else:
    default_params = {}
    page_t0 = _log_timing(page_t0, "Skipped config param load")

params = pipeline_defaults.copy()
for k, v in default_params.items():
    if v is not None:
        params[k] = v

# Backward/forward compatibility for anti-double cleanup factor naming.
if params.get("ANTI_DOUBLE_GAP_FACTOR") is None and params.get("HR_CONFLICT_RR_FACTOR") is not None:
    params["ANTI_DOUBLE_GAP_FACTOR"] = params["HR_CONFLICT_RR_FACTOR"]
if params.get("HR_CONFLICT_RR_FACTOR") is None and params.get("ANTI_DOUBLE_GAP_FACTOR") is not None:
    params["HR_CONFLICT_RR_FACTOR"] = params["ANTI_DOUBLE_GAP_FACTOR"]

# Select parent signal and channel
st.sidebar.subheader("Signal Configuration")
parent_signal_options = list(data_pkl.signal_data.keys())
if not parent_signal_options:
    st.error("No signals found in data_pkl.signal_data.")
    st.stop()

if detection_mode == "heart_rate":
    default_parent_signal = "ecg" if "ecg" in parent_signal_options else parent_signal_options[0]
else:
    stroke_parent_from_config = params.get("STROKE_PARENT_SIGNAL")
    stroke_parent_candidates = ["dynamic_accel", "corrected_gyr", "gyroscope", "corrected_acc", "calibrated_acc", "accelerometer"]
    if stroke_parent_from_config in parent_signal_options:
        default_parent_signal = stroke_parent_from_config
    else:
        default_parent_signal = next((sig for sig in stroke_parent_candidates if sig in parent_signal_options), parent_signal_options[0])

default_parent_index = parent_signal_options.index(default_parent_signal) if default_parent_signal in parent_signal_options else 0
parent_signal = st.sidebar.selectbox("Parent Signal", parent_signal_options, index=default_parent_index)

# Get available channels for the selected parent signal
if parent_signal in data_pkl.signal_data:
    available_channels = [c for c in data_pkl.signal_data[parent_signal].columns if c != "datetime"]
else:
    available_channels = []

# Select specific channel
if not available_channels:
    st.error(f"No channels found for parent signal '{parent_signal}'.")
    st.stop()

animal_id = data_pkl.animal_info.get("Animal_ID", "")
if detection_mode == "heart_rate":
    default_channel = "ecg"
else:
    stroke_channel_from_config = params.get("STROKE_CHANNEL")
    if stroke_channel_from_config:
        default_channel = _resolve_channel_alias(stroke_channel_from_config, available_channels)
        if default_channel not in available_channels:
            default_channel = None
    else:
        default_channel = None

    if default_channel is None:
        if parent_signal in ("corrected_acc", "calibrated_acc", "accelerometer", "dynamic_accel"):
            if str(animal_id).startswith(('nesc', 'mian')):
                preferred_channels = ['ax', 'x', 'ay', 'y', 'az', 'z']
            elif str(animal_id).startswith(('oror', 'bamu')):
                preferred_channels = ['ay', 'y', 'ax', 'x', 'az', 'z']
            else:
                preferred_channels = ['ax', 'x', 'ay', 'y', 'az', 'z']
        else:
            if str(animal_id).startswith(('nesc', 'mian')):
                preferred_channels = ['gx', 'gy', 'gz']
            elif str(animal_id).startswith(('oror', 'bamu')):
                preferred_channels = ['gy', 'gx', 'gz']
            else:
                preferred_channels = ['gy', 'gx', 'gz']
        default_channel = next((ch for ch in preferred_channels if ch in available_channels), available_channels[0])

channel_index = available_channels.index(default_channel) if default_channel in available_channels else 0
channel = st.sidebar.selectbox(
    "Channel",
    available_channels,
    index=channel_index,
    format_func=_channel_label,
)

# Persist stroke signal/channel selection alongside detection parameters.
if detection_mode == "stroke_rate":
    params["STROKE_PARENT_SIGNAL"] = parent_signal
    params["STROKE_CHANNEL"] = str(channel).split("__", 1)[0]

derivative_options = ["normalized", "smoothed", "narrow_bandpass", "broad_bandpass", "spikeless", "raw"]
stored_derivative_channels = _coerce_string_list(params.get("DETECTION_DERIVATIVE_CHANNELS"))
default_derivative_channels = [d for d in stored_derivative_channels if d in derivative_options] or ["normalized"]
detection_derivative_channels = st.sidebar.multiselect(
    "Derivative Inputs for Detection",
    options=derivative_options,
    default=default_derivative_channels,
    key=f"{widget_prefix}_derivative_channels",
    help="Run peak detection on one or more derived versions of this channel and merge candidates.",
)
if not detection_derivative_channels:
    detection_derivative_channels = ["normalized"]
params["DETECTION_DERIVATIVE_CHANNELS"] = detection_derivative_channels

# Configure signals
signal_df = data_pkl.signal_data[parent_signal]
signal = data_pkl.signal_data[parent_signal][channel]
datetime_signal = data_pkl.signal_data[parent_signal]['datetime']
sampling_rate = data_pkl.signal_info.get(parent_signal, {}).get('sampling_frequency', calculate_sampling_frequency(datetime_signal.head()))
if sampling_rate is None:
    st.error(f"Could not determine sampling rate for parent signal '{parent_signal}'.")
    st.stop()

# Streamlit UI
st.title(f"{detection_mode} Peak Detection")

# Time range controls (main page): step paging + editable datetimes
range_start = _to_display_naive(OVERLAP_START_TIME).to_pydatetime()
range_end = _to_display_naive(OVERLAP_END_TIME).to_pydatetime()
default_start = _to_display_naive(ZOOM_WINDOW_START_TIME).to_pydatetime()
default_end = _to_display_naive(ZOOM_WINDOW_END_TIME).to_pydatetime()

window_start_value_key = f"{widget_prefix}_window_start_value"
window_end_value_key = f"{widget_prefix}_window_end_value"
if window_start_value_key not in st.session_state:
    st.session_state[window_start_value_key] = default_start.strftime("%Y-%m-%d %H:%M:%S")
if window_end_value_key not in st.session_state:
    st.session_state[window_end_value_key] = default_end.strftime("%Y-%m-%d %H:%M:%S")

def _clamp_window(start_dt, end_dt, min_dt, max_dt):
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(seconds=1)
    window = end_dt - start_dt
    total = max_dt - min_dt
    if window >= total:
        return min_dt, max_dt
    if start_dt < min_dt:
        start_dt = min_dt
        end_dt = start_dt + window
    if end_dt > max_dt:
        end_dt = max_dt
        start_dt = end_dt - window
    return start_dt, end_dt

def _parse_display_datetime(value):
    ts = pd.to_datetime(value, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()

top_left, top_right = st.columns([1, 1])
with top_right:
    st.markdown("### Peak Rejection Attribution")
    attribution_caption_placeholder = st.empty()
    attribution_plot_placeholder = st.empty()
    attribution_caption_placeholder.caption("Will populate after cleanup for current window.")

with top_left:
    st.markdown("### Processing Window")
    window_options = {
        "2 min": timedelta(minutes=2),
        "5 min": timedelta(minutes=5),
        "10 min": timedelta(minutes=10),
        "30 min": timedelta(minutes=30),
        "1 hr": timedelta(hours=1),
        "6 hr": timedelta(hours=6),
        "1 d": timedelta(days=1),
    }
    step_options = {
        "30 sec": timedelta(seconds=30),
        "1 min": timedelta(minutes=1),
        "10 min": timedelta(minutes=10),
        "30 min": timedelta(minutes=30),
        "1 hr": timedelta(hours=1),
        "6 hr": timedelta(hours=6),
        "1 d": timedelta(days=1),
    }
    window_label = st.selectbox(
        "Window Duration",
        list(window_options.keys()),
        index=0,
        key=f"{widget_prefix}_window_size",
    )
    window_size = window_options[window_label]
    step_label = st.selectbox(
        "Page Step",
        list(step_options.keys()),
        index=0,
        key=f"{widget_prefix}_jump_step",
    )
    step_size = step_options[step_label]
    st.caption(
        f"Available range: {range_start.strftime('%Y-%m-%d %H:%M:%S')} to {range_end.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    typed_start = _parse_display_datetime(st.session_state[window_start_value_key])
    typed_end = _parse_display_datetime(st.session_state[window_end_value_key])
    if typed_start is None:
        st.warning("Use YYYY-MM-DD HH:MM:SS format for Start. Reverting to current default.")
        typed_start = default_start
    if typed_end is None:
        typed_end = typed_start + window_size
    typed_start, typed_end = _clamp_window(typed_start, typed_end, range_start, range_end)

    start_text = st.text_input(
        "Start (YYYY-MM-DD HH:MM:SS)",
        value=st.session_state[window_start_value_key],
    )
    st.text_input(
        "End (auto from window)",
        value=st.session_state[window_end_value_key],
        disabled=True,
    )

    nav_buttons = st.columns(2)
    with nav_buttons[0]:
        prev_clicked = st.button(f"Previous {step_label}", key=f"{widget_prefix}_prev_window")
    with nav_buttons[1]:
        next_clicked = st.button(f"Next {step_label}", key=f"{widget_prefix}_next_window")

    if prev_clicked or next_clicked:
        delta = step_size if next_clicked else -step_size
        typed_start, typed_end = _clamp_window(typed_start + delta, typed_end + delta, range_start, range_end)
        st.session_state[window_start_value_key] = typed_start.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state[window_end_value_key] = typed_end.strftime("%Y-%m-%d %H:%M:%S")

start_display = _parse_display_datetime(start_text) or typed_start
end_display = start_display + window_size
start_display, end_display = _clamp_window(start_display, end_display, range_start, range_end)
st.session_state[window_start_value_key] = start_display.strftime("%Y-%m-%d %H:%M:%S")
st.session_state[window_end_value_key] = end_display.strftime("%Y-%m-%d %H:%M:%S")

# Convert selected display-naive timestamps to deployment-aware timestamps for processing
start_datetime = _display_naive_to_deployment_aware(start_display)
end_datetime = _display_naive_to_deployment_aware(end_display)

# Filter signal based on the selected time range in deployment timezone
datetime_signal = pd.to_datetime(datetime_signal, errors="coerce")
if datetime_signal.dt.tz is None:
    datetime_signal = datetime_signal.dt.tz_localize(DEPLOYMENT_TZ)
else:
    datetime_signal = datetime_signal.dt.tz_convert(DEPLOYMENT_TZ)
signal_df_datetime = pd.to_datetime(signal_df["datetime"], errors="coerce")
if signal_df_datetime.dt.tz is None:
    signal_df_datetime = signal_df_datetime.dt.tz_localize(DEPLOYMENT_TZ)
else:
    signal_df_datetime = signal_df_datetime.dt.tz_convert(DEPLOYMENT_TZ)

time_mask = (datetime_signal >= start_datetime) & (datetime_signal <= end_datetime)
signal_subset = signal[time_mask]
datetime_subset = datetime_signal[time_mask]
signal_subset_df = signal_df[(signal_df_datetime >= start_datetime) & (signal_df_datetime <= end_datetime)]

# Slider configurations
PARAM_HELP = {
    "BROAD_LOW_CUTOFF": "Low cutoff of the broad bandpass used to isolate the general rhythm signal.",
    "BROAD_HIGH_CUTOFF": "High cutoff of the broad bandpass used before narrower filtering/refinement.",
    "NARROW_LOW_CUTOFF": "Low cutoff of the narrow bandpass that emphasizes beat/stroke-like peaks.",
    "NARROW_HIGH_CUTOFF": "High cutoff of the narrow bandpass used for final peak finding.",
    "FILTER_ORDER": "Butterworth filter order. Higher values are sharper but can increase ringing.",
    "SPIKE_THRESHOLD": "Large raw spikes above this magnitude are treated as artifacts in spike-removal step.",
    "SMOOTH_SEC_MULTIPLIER": "Smoothing window length in seconds for the filtered signal.",
    "WINDOW_SIZE_MULTIPLIER": "Window length in seconds for local normalization.",
    "NORMALIZATION_NOISE": "Std floor used in normalization to avoid division by tiny local variance.",
    "PEAK_HEIGHT": "Minimum peak height threshold applied to the normalized signal.",
    "PEAK_DISTANCE_SEC": "Minimum time separation between detected peaks.",
    "SEARCH_RADIUS_SEC": "Radius used during refinement to snap each peak to a nearby local maximum.",
    "MIN_PEAK_HEIGHT": "Minimum allowed peak height in the original/smoothed domain after detection.",
    "MAX_PEAK_HEIGHT": "Maximum allowed peak height in the original/smoothed domain after detection.",
    "enable_bandpass": "Apply broad and narrow bandpass filters.",
    "enable_spike_removal": "Suppress impulsive spikes before peak detection.",
    "enable_absolute": "Take absolute value before smoothing/normalization (useful for bipolar waveforms).",
    "enable_smoothing": "Apply smoothing to reduce noise before normalization and peak search.",
    "enable_normalization": "Apply local normalization so thresholds are less sensitive to amplitude drift.",
    "enable_refinement": "Refine initial peaks to nearby local maxima in the chosen search radius.",
    "DETECTION_DERIVATIVE_CHANNELS": "Derived signal representations to use for candidate peak detection.",
    "HR_JUMP_FRAC": "Reject intervals with abrupt rate jumps larger than this fractional change.",
    "MIN_RR_SEC": "Minimum allowed interval (seconds) between adjacent accepted events.",
    "MAX_HR_BPM": "Upper bound for displayed/fixed rate output.",
    "MIN_HR_BPM": "Lower bound for displayed/fixed rate output.",
    "ANTI_DOUBLE_GAP_FACTOR": "Relative gap factor for anti-double detection. If two candidates are too close (< local median interval x factor), keep only the stronger one.",
    "ANTI_DOUBLE_ROLLING_WINDOW_SEC": "Duration (seconds) of local rolling window used to estimate RR baseline for anti-double cleanup.",
    "PICK_LAST_IN_CONFLICT_PAIR": "If enabled, keep the later peak in a close conflict pair; if disabled, keep the earlier peak.",
}

slider_config = [
    ("Broad Low Cutoff (Hz)", "BROAD_LOW_CUTOFF", 0.01, 25.0, float, 0.01),
    ("Broad High Cutoff (Hz)", "BROAD_HIGH_CUTOFF", 0.01, 25.0, float, 0.01),
    ("Narrow Low Cutoff (Hz)", "NARROW_LOW_CUTOFF", 0.01, 15.0, float, 0.01),
    ("Narrow High Cutoff (Hz)", "NARROW_HIGH_CUTOFF", 0.01, 15.0, float, 0.01),
    ("Filter Order", "FILTER_ORDER", 1, 10, int, 1),
    ("Spike Threshold", "SPIKE_THRESHOLD", 100, 1000, int, 10),
    ("Smoothing Window (s)", "SMOOTH_SEC_MULTIPLIER", 0.01, 15.0, float, 0.05),
    ("Normalization Window Size (s)", "WINDOW_SIZE_MULTIPLIER", 0.1, 40.0, float, 0.1),
    ("Normalization Std Floor (fraction)", "NORMALIZATION_NOISE", 0.0, 0.5, float, 0.01),
    ("Normalized Peak Height", "PEAK_HEIGHT", -2.0, 2.0, float, 0.1),
    ("Peak Distance (s)", "PEAK_DISTANCE_SEC", 0.01, 5.0, float, 0.05),
    ("Search Radius (s)", "SEARCH_RADIUS_SEC", 0.1, 2.0, float, 0.05),
    ("Minimum Peak Height", "MIN_PEAK_HEIGHT", -10.0, 100.0, float, 1.0),
    ("Maximum Peak Height", "MAX_PEAK_HEIGHT", 1.0, 1000000.0, float, 10.0),
]

# Checkbox configurations
checkbox_config = [
    ("Enable Bandpass Filtering", "enable_bandpass"),
    ("Enable Spike Removal", "enable_spike_removal"),
    ("Enable Absolute Transformation", "enable_absolute"),
    ("Enable Smoothing", "enable_smoothing"),
    ("Enable Normalization", "enable_normalization"),
    ("Enable Refinement", "enable_refinement"),
]

hr_cleanup_config = [
    ("HR Jump Fraction", "HR_JUMP_FRAC", 0.0, 5.0, float, 0.05),
    ("Minimum RR (s)", "MIN_RR_SEC", 0.01, 5.0, float, 0.01),
    ("Maximum HR (bpm)", "MAX_HR_BPM", 1.0, 500.0, float, 1.0),
    ("Minimum HR (bpm)", "MIN_HR_BPM", 0.0, 500.0, float, 0.1),
    ("Anti-Double Gap Factor (relative)", "ANTI_DOUBLE_GAP_FACTOR", 0.1, 2.0, float, 0.01),
    ("Anti-Double Rolling Window (s)", "ANTI_DOUBLE_ROLLING_WINDOW_SEC", 2.0, 120.0, float, 0.5),
]
hr_cleanup_checkbox_config = [
    ("Pick Later Peak In Conflict Pair", "PICK_LAST_IN_CONFLICT_PAIR"),
]

page_t0 = _log_timing(page_t0, "Resolved defaults/missing params")

# Streamlit UI for sliders and checkboxes
st.sidebar.subheader("Adjust Peak Detection Parameters")

# Coerce possibly-missing config values into safe numeric defaults.
def _safe_slider_default(raw_value, min_val, max_val, dtype):
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        value = min_val
    else:
        try:
            value = dtype(raw_value)
        except (TypeError, ValueError):
            value = min_val
    if value < min_val:
        value = min_val
    if value > max_val:
        value = max_val
    return dtype(value)


# Collect suggested ranges from all deployments' parameter logs.
def _collect_param_suggestions(key, section="hr_peak_detection_settings"):
    """Return (min, max) across all deployments that have this param, or None."""
    try:
        import glob, json, os as _os
        values = []
        pattern = _os.path.join(data_dir, "**", "parameter_log.json")
        for path in glob.glob(pattern, recursive=True):
            try:
                with open(path) as f:
                    log = json.load(f)
                v = log.get(section, {}).get(key)
                if v is not None:
                    values.append(float(v))
            except Exception:
                pass
        if len(values) >= 1:
            return min(values), max(values)
    except Exception:
        pass
    return None


# Peak detection parameters — single number_input per param (no slider).
for label, key, min_val, max_val, dtype, step in slider_config:
    raw_default = default_params.get(key)
    if raw_default is None and key in pipeline_defaults:
        raw_default = pipeline_defaults[key]

    # Expand bounds so the current value is always reachable.
    input_min = min_val
    input_max = max_val
    if raw_default is not None:
        try:
            raw_num = float(raw_default)
            input_min = min(input_min, raw_num)
            input_max = max(input_max, raw_num)
        except (TypeError, ValueError):
            pass

    default_value = _safe_slider_default(raw_default, input_min, input_max, dtype)

    # Build help string with suggested range from cross-deployment param logs.
    base_help = PARAM_HELP.get(key) or ""
    suggestion = _collect_param_suggestions(key)
    if suggestion is not None:
        sug_min, sug_max = suggestion
        sug_str = f"{sug_min:.4g}" if sug_min == sug_max else f"{sug_min:.4g} – {sug_max:.4g}"
        range_help = f"Suggested range across deployments: {sug_str}. Typical bounds: [{min_val}, {max_val}]."
    else:
        range_help = f"Typical bounds: [{min_val}, {max_val}]."
    help_text = f"{base_help}  {range_help}".strip()

    params[key] = dtype(st.sidebar.number_input(
        label,
        min_value=dtype(input_min),
        max_value=dtype(input_max),
        value=default_value,
        step=step,
        key=f"{widget_prefix}_param_{key}",
        help=help_text,
    ))

# Adjust checkboxes
for label, key in checkbox_config:
    default_value = default_params.get(key)
    if default_value is None and key in pipeline_defaults:
        default_value = pipeline_defaults[key]
    if default_value is None:
        default_value = True
    params[key] = st.sidebar.checkbox(
        label,
        value=default_value,
        key=f"{widget_prefix}_checkbox_{key}",
        help=PARAM_HELP.get(key),
    )

if detection_mode == "heart_rate":
    st.sidebar.subheader("HR Cleanup Parameters")
    for label, key, min_val, max_val, dtype, step in hr_cleanup_config:
        raw_default = default_params.get(key)
        if raw_default is None and key in pipeline_defaults:
            raw_default = pipeline_defaults[key]
        default_value = _safe_slider_default(raw_default, min_val, max_val, dtype)
        params[key] = dtype(st.sidebar.number_input(
            label,
            min_value=dtype(min_val),
            max_value=dtype(max_val),
            value=dtype(default_value),
            step=step,
            key=f"{widget_prefix}_hr_cleanup_{key}",
            help=PARAM_HELP.get(key),
        ))
    for label, key in hr_cleanup_checkbox_config:
        default_value = default_params.get(key)
        if default_value is None and key in pipeline_defaults:
            default_value = pipeline_defaults[key]
        params[key] = st.sidebar.checkbox(
            label,
            value=_safe_bool(default_value, default=True),
            key=f"{widget_prefix}_hr_cleanup_checkbox_{key}",
            help=PARAM_HELP.get(key),
        )
    # Keep legacy key in sync for workflow/backward compatibility.
    if params.get("ANTI_DOUBLE_GAP_FACTOR") is not None:
        params["HR_CONFLICT_RR_FACTOR"] = params["ANTI_DOUBLE_GAP_FACTOR"]

# Save updated configuration
if st.sidebar.button("Save Configuration"):
    param_manager.add_to_config(entries=params, section=section)
    st.session_state.pop("peak_detect_param_cache", None)
    st.success(f"{detection_mode} configuration saved successfully!")

overwrite_existing_prompt = st.sidebar.checkbox(
    "Overwrite deployment-specific settings too",
    value=False,
    key=f"{widget_prefix}_set_dataset_default_overwrite",
    help=(
        "If enabled, deployments in this dataset that already have "
        f"'{section}' entries will be overwritten with these values."
    ),
)

if st.sidebar.button("Set Default for Dataset"):
    # Save dataset-level defaults first.
    param_manager.set_dataset_defaults(entries=params, section=section)

    overwritten_count = 0
    if overwrite_existing_prompt:
        config_log = param_manager._load_config()
        for entry in config_log:
            dep_id = entry.get("deployment_id")
            if dep_id in (None, param_manager.DEFAULTS_DEPLOYMENT_ID):
                continue
            sec = entry.get(section, {})
            if isinstance(sec, dict) and len(sec) > 0:
                param_manager.add_to_config(entries=params, section=section, deployment_id=dep_id)
                overwritten_count += 1

    st.session_state.pop("peak_detect_param_cache", None)
    if overwrite_existing_prompt:
        st.success(
            f"Set dataset default for '{section}' and overwrote {overwritten_count} deployment-specific setting blocks."
        )
    else:
        st.success(f"Set dataset default for '{section}'. Existing deployment-specific settings were not changed.")

page_t0 = _log_timing(page_t0, "UI ready / auto compute")

def _validate_filter_params(params_dict, fs):
    if not params_dict.get("enable_bandpass", True):
        return None
    if fs is None or fs <= 0:
        return "Sampling rate must be positive for bandpass filtering."
    nyquist = 0.5 * fs
    cutoffs = [
        ("BROAD_LOW_CUTOFF", params_dict["BROAD_LOW_CUTOFF"]),
        ("BROAD_HIGH_CUTOFF", params_dict["BROAD_HIGH_CUTOFF"]),
        ("NARROW_LOW_CUTOFF", params_dict["NARROW_LOW_CUTOFF"]),
        ("NARROW_HIGH_CUTOFF", params_dict["NARROW_HIGH_CUTOFF"]),
    ]
    for name, value in cutoffs:
        if value <= 0 or value >= nyquist:
            return (
                f"{name}={value:.4g} must satisfy 0 < cutoff < Nyquist ({nyquist:.4g} Hz). "
                "Adjust filter cutoffs or disable bandpass."
            )
    if params_dict["BROAD_LOW_CUTOFF"] >= params_dict["BROAD_HIGH_CUTOFF"]:
        return "BROAD_LOW_CUTOFF must be smaller than BROAD_HIGH_CUTOFF."
    if params_dict["NARROW_LOW_CUTOFF"] >= params_dict["NARROW_HIGH_CUTOFF"]:
        return "NARROW_LOW_CUTOFF must be smaller than NARROW_HIGH_CUTOFF."
    return None

# Use the updated parameters in peak detection
param_error = _validate_filter_params(params, sampling_rate)
if param_error:
    st.error(f"Peak detection parameter error: {param_error}")
    st.stop()

try:
    run_t0 = time.perf_counter()
    results = peak_detect(
        signal=signal_subset,
        sampling_rate=sampling_rate,
        datetime_series=datetime_subset,
        broad_lowcut=params["BROAD_LOW_CUTOFF"],
        broad_highcut=params["BROAD_HIGH_CUTOFF"],
        narrow_lowcut=params["NARROW_LOW_CUTOFF"],
        narrow_highcut=params["NARROW_HIGH_CUTOFF"],
        filter_order=params["FILTER_ORDER"],
        spike_threshold=params["SPIKE_THRESHOLD"],
        smooth_sec_multiplier=params["SMOOTH_SEC_MULTIPLIER"],
        window_size_multiplier=params["WINDOW_SIZE_MULTIPLIER"],
        normalization_noise=params["NORMALIZATION_NOISE"],
        peak_height=params["PEAK_HEIGHT"],
        peak_distance_sec=params["PEAK_DISTANCE_SEC"],
        search_radius_sec=params["SEARCH_RADIUS_SEC"],
        min_peak_height=params["MIN_PEAK_HEIGHT"],
        max_peak_height=params["MAX_PEAK_HEIGHT"],
        enable_bandpass=params["enable_bandpass"],
        enable_spike_removal=params["enable_spike_removal"],
        enable_absolute=params["enable_absolute"],
        enable_smoothing=params["enable_smoothing"],
        enable_normalization=params["enable_normalization"],
        enable_refinement=params["enable_refinement"],
        detection_sources=params.get("DETECTION_DERIVATIVE_CHANNELS"),
    )
    run_t0 = _log_timing(run_t0, "peak_detect()")
except Exception as e:
    st.error(f"Peak detection failed: {type(e).__name__}: {e}")
    st.stop()

process_rate(data_pkl, results, signal_subset_df, parent_signal, 
             params, sampling_rate, detection_mode)
run_t0 = _log_timing(run_t0, "process_rate()")

cleanup_diag = None
if detection_mode == "heart_rate":
    results, nan_segments, cleanup_diag = _cleanup_heartbeat_peaks(
        results=results,
        signal_subset_df=signal_subset_df,
        params=params,
        sampling_rate=sampling_rate,
        parent_signal=parent_signal,
    )
    if "heart_rate_fixed_df" in results:
        # Patch the newly-computed heart_rate_fixed values back into the existing
        # multi-channel DataFrame (which also contains heart_rate_nan and
        # manually_derived_hr) by aligning on datetime.  Never use upsert_rate_signal
        # here — it resets signal_info to a single-channel layout.
        new_fixed_df = results["heart_rate_fixed_df"]
        existing_hr_df = data_pkl.signal_data.get("heart_rate_fixed")
        if (existing_hr_df is not None
                and isinstance(existing_hr_df, pd.DataFrame)
                and "heart_rate_fixed" in existing_hr_df.columns
                and "datetime" in existing_hr_df.columns
                and "datetime" in new_fixed_df.columns):
            existing_hr_df = existing_hr_df.copy()
            # Align by datetime index — new_fixed_df covers only the detection window
            new_idx = pd.to_datetime(new_fixed_df["datetime"]).values
            exist_idx = pd.to_datetime(existing_hr_df["datetime"]).values
            if len(new_idx) == len(exist_idx):
                # Same length: direct assignment
                existing_hr_df["heart_rate_fixed"] = new_fixed_df["heart_rate_fixed"].values
            else:
                # Different lengths (subset window): update matching rows via merge
                new_s = pd.Series(
                    new_fixed_df["heart_rate_fixed"].values,
                    index=pd.to_datetime(new_fixed_df["datetime"]),
                )
                existing_hr_df = existing_hr_df.set_index(pd.to_datetime(existing_hr_df["datetime"]))
                existing_hr_df.update(new_s.rename("heart_rate_fixed"))
                existing_hr_df = existing_hr_df.reset_index(drop=True)
            data_pkl.signal_data["heart_rate_fixed"] = existing_hr_df
            # Preserve existing signal_info channels/metadata; just update log.
            si = data_pkl.signal_info.get("heart_rate_fixed", {})
            si["transformation_log"] = [
                "recomputed HR after beat cleanup in Streamlit",
                "NaN gaps filled using interpolation",
            ]
            data_pkl.signal_info["heart_rate_fixed"] = si
        else:
            upsert_rate_signal(
                data_pkl=data_pkl,
                rate_df=new_fixed_df,
                rate_key="heart_rate_fixed",
                parent_signal=parent_signal,
                transformation_log=[
                    "recomputed HR after beat cleanup in Streamlit",
                    "NaN gaps filled using interpolation",
                ],
            )
        # Make heart_rate reflect post-cleanup RR output for this Streamlit workflow.
        heart_rate_clean = results["heart_rate_fixed_df"].rename(columns={"heart_rate_fixed": "heart_rate"}).copy()
        upsert_rate_signal(
            data_pkl=data_pkl,
            rate_df=heart_rate_clean,
            rate_key="heart_rate",
            parent_signal=parent_signal,
            transformation_log=[
                "overwritten by post-cleanup RR output in Streamlit page 6",
                "derived from heart_rate_fixed cleanup series",
            ],
        )
    _upsert_heartbeat_cleanup_events(data_pkl, results, signal_subset_df)
    run_t0 = _log_timing(run_t0, "heartbeat cleanup + event sync")

TARGET_SAMPLING_RATE = 25 if detection_mode == "heart_rate" else 10
st.markdown("### Plot Configuration")
_base_plot_signals_raw = (
    [
        parent_signal, "ecg", "depth", "prh", "heart_rate", "hr_broad_bandpass",
        "hr_narrow_bandpass", "hr_smoothed", "hr_normalized", "heart_rate_fixed",
    ] if detection_mode == "heart_rate" else [
        parent_signal, "depth", "prh", "stroke_rate", "sr_broad_bandpass",
        "sr_narrow_bandpass", "sr_smoothed", "sr_normalized",
    ]
)
base_plot_signals = list(dict.fromkeys(s for s in _base_plot_signals_raw if s in data_pkl.signal_data))
all_plot_signals = list(data_pkl.signal_data.keys())
saved_plot_cfg = param_manager.get_from_config(
    [f"peak_detect_default_signals_{detection_mode}", f"peak_detect_default_events_{detection_mode}"],
    section="settings",
)
stored_plot_signals = saved_plot_cfg.get(f"peak_detect_default_signals_{detection_mode}") or base_plot_signals
stored_plot_signals = [sig for sig in stored_plot_signals if sig in all_plot_signals]
if not stored_plot_signals:
    stored_plot_signals = base_plot_signals or ([all_plot_signals[0]] if all_plot_signals else [])

selected_plot_signals = st.multiselect(
    "Signals to plot",
    options=all_plot_signals,
    default=stored_plot_signals,
    key=f"{widget_prefix}_plot_signals",
    help="Choose which signals appear in the main plot.",
)
if not selected_plot_signals:
    st.warning("Please select at least one signal to plot.")
    st.stop()

# Order selected signals for plotting.
if len(selected_plot_signals) > 1:
    st.caption("Signal plot order")
    order_df = pd.DataFrame({"signal": selected_plot_signals, "order": list(range(1, len(selected_plot_signals) + 1))})
    edited_order_df = st.data_editor(
        order_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["signal"],
        key=f"{widget_prefix}_signal_order_editor",
    )
    try:
        selected_plot_signals = (
            edited_order_df.sort_values("order", kind="stable")["signal"].astype(str).tolist()
        )
    except Exception:
        pass

if set(selected_plot_signals) != set(stored_plot_signals):
    param_manager.add_to_config(
        entries={f"peak_detect_default_signals_{detection_mode}": selected_plot_signals},
        section="settings",
    )

plot_channels = {}
with st.expander("Channel Configuration", expanded=False):
    for sig in selected_plot_signals:
        sig_df = data_pkl.signal_data.get(sig)
        if not isinstance(sig_df, pd.DataFrame):
            continue
        channel_options = [c for c in sig_df.columns if c != "datetime"]
        if not channel_options:
            continue
        default_sig_channels = data_pkl.signal_info.get(sig, {}).get("channels", channel_options)
        default_sig_channels = [c for c in default_sig_channels if c in channel_options] or channel_options
        chosen_sig_channels = st.multiselect(
            f"{sig} channels",
            options=channel_options,
            default=default_sig_channels,
            key=f"{widget_prefix}_plot_channels_{sig}",
        )
        chosen_sig_channels = chosen_sig_channels or channel_options
        if len(chosen_sig_channels) > 1:
            ch_order_df = pd.DataFrame({"channel": chosen_sig_channels, "order": list(range(1, len(chosen_sig_channels) + 1))})
            edited_ch_order_df = st.data_editor(
                ch_order_df,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                disabled=["channel"],
                key=f"{widget_prefix}_plot_channel_order_{sig}",
            )
            try:
                chosen_sig_channels = (
                    edited_ch_order_df.sort_values("order", kind="stable")["channel"].astype(str).tolist()
                )
            except Exception:
                pass
        plot_channels[sig] = chosen_sig_channels

event_style_defaults = {
    "beat_auto_detect_accepted": {"symbol": "triangle-up", "color": "green", "signal": "ecg"},
    "beat_auto_detect_rejected": {"symbol": "triangle-up", "color": "red", "signal": "ecg"},
    "heartbeat_manual_ok": {"symbol": "circle", "color": "blue", "signal": "ecg", "y_offset_frac": -0.05},
    "heartbeat_auto_detect_accepted": {"symbol": "triangle-up", "color": "green", "signal": "ecg"},
    "heartbeat_auto_detect_rejected": {"symbol": "triangle-up", "color": "red", "signal": "ecg"},
    "heartbeat_auto_detect_rejected_conflict": {"symbol": "triangle-up", "color": "orange", "signal": "ecg"},
    "heartbeat_auto_detect_cleanup_rejected": {"symbol": "triangle-down", "color": "orange", "signal": "ecg"},
    "heartbeat_auto_detect_suggested": {"symbol": "diamond", "color": "orange", "signal": "ecg"},
    "heartbeat_auto_detect_gap": {"symbol": "circle", "color": "gray", "signal": "heart_rate_fixed", "state_color": "rgba(140, 140, 140, 0.25)"},
    "QC_unusable_ecg": {"signal": "ecg", "state_color": "rgba(80, 80, 80, 0.45)"},
    "QC_usable_ecg":   {"signal": "ecg", "state_color": "rgba(100, 220, 100, 0.15)"},
    "strokebeat_auto_detect_accepted": {"symbol": "triangle-up", "color": "green", "signal": "sr_narrow_bandpass"},
    "strokebeat_auto_detect_rejected": {"symbol": "triangle-up", "color": "red", "signal": "sr_narrow_bandpass"},
    "dive": {"symbol": "triangle-down", "color": "blue", "signal": "depth", "state_color": "rgba(150, 150, 150, 0.3)"},
    "exhalation_breath": {"symbol": "triangle-up", "color": "orange", "signal": "depth"},
    "uw_exhalation": {"symbol": "triangle-down", "color": "cyan", "signal": "depth"},
}

def _infer_event_signal(event_key, chosen_signals):
    key = str(event_key or "").lower()
    signal_pool = chosen_signals or list(data_pkl.signal_data.keys())
    keyword_map = [
        ("dive", ["depth", "pressure"]),
        ("stroke", ["stroke_rate", "sr_narrow_bandpass", "prh"]),
        ("heart", ["heart_rate_fixed", "heart_rate", "hr_narrow_bandpass", "ecg"]),
        ("beat", ["heart_rate_fixed", "hr_narrow_bandpass", "sr_narrow_bandpass", "ecg"]),
        ("breath", ["depth", "pressure", "heart_rate_fixed"]),
    ]
    for keyword, candidates in keyword_map:
        if keyword in key:
            for candidate in candidates:
                if candidate in signal_pool:
                    return candidate
    return signal_pool[0] if signal_pool else None


def _resolve_marker_signals(event_key, chosen_signals):
    fallback = _infer_event_signal(event_key, chosen_signals)
    return [fallback] if fallback else (list(chosen_signals) if chosen_signals else [])

event_key_options = []
event_df = None
base_default_events = [
    "beat_auto_detect_accepted",
    "beat_auto_detect_rejected",
    "heartbeat_manual_ok",
    "heartbeat_auto_detect_accepted",
    "heartbeat_auto_detect_rejected",
    "heartbeat_auto_detect_cleanup_rejected",
    "heartbeat_auto_detect_suggested",
    "heartbeat_auto_detect_gap",
    "QC_unusable_ecg",
    "QC_usable_ecg",
    "strokebeat_auto_detect_accepted",
    "strokebeat_auto_detect_rejected",
    "dive",
]
if isinstance(data_pkl.event_data, pd.DataFrame) and "key" in data_pkl.event_data.columns:
    event_df = data_pkl.event_data.copy()
    event_key_options = sorted([str(k) for k in event_df["key"].dropna().unique()])
    event_key_counts = data_pkl.event_data["key"].value_counts(dropna=True)
    with st.expander("Event Keys", expanded=False):
        if event_key_counts.empty:
            st.write("No event keys found.")
        else:
            for k, v in event_key_counts.items():
                st.write(f"{k}: {int(v)}")
else:
    event_key_counts = pd.Series(dtype=int)

if event_key_options:
    breath_like = [k for k in event_key_options if "breath" in k.lower()]
    dive_like = [k for k in event_key_options if "dive" in k.lower()]
    stored_default_events = saved_plot_cfg.get(f"peak_detect_default_events_{detection_mode}") or []
    stored_default_events = [k for k in stored_default_events if k in event_key_options]
    default_event_selection = [k for k in (base_default_events + breath_like + dive_like) if k in event_key_options]
    default_event_selection = sorted(set(default_event_selection + stored_default_events))
    selected_event_keys = st.multiselect(
        "Events/States to plot",
        options=event_key_options,
        default=default_event_selection,
        key=f"{widget_prefix}_event_overlay_keys",
        help="Choose point and interval events to overlay on the plot.",
    )
    if set(selected_event_keys) != set(default_event_selection):
        param_manager.add_to_config(
            entries={f"peak_detect_default_events_{detection_mode}": selected_event_keys},
            section="settings",
        )
else:
    selected_event_keys = []

event_channel_map_key = f"peak_detect_event_channel_map_{detection_mode}"
saved_event_channel_cfg = param_manager.get_from_config([event_channel_map_key], section="settings").get(event_channel_map_key) or {}
event_channel_options = []
for sig in selected_plot_signals:
    for ch in plot_channels.get(sig, []):
        event_channel_options.append(f"{sig}.{ch}")

event_target_map = {}
if selected_event_keys and event_channel_options:
    with st.expander("Event Placement (signals/channels)", expanded=False):
        for event_key in selected_event_keys:
            inferred_signals = _resolve_marker_signals(event_key, selected_plot_signals)
            inferred_signals = [s for s in inferred_signals if s in selected_plot_signals]
            inferred_labels = []
            for s in inferred_signals:
                for c in plot_channels.get(s, []):
                    inferred_labels.append(f"{s}.{c}")
            saved_labels = saved_event_channel_cfg.get(event_key, [])
            saved_labels = [lbl for lbl in saved_labels if lbl in event_channel_options]
            default_labels = saved_labels or inferred_labels or event_channel_options[:1]
            picked_labels = st.multiselect(
                f"{event_key} targets",
                options=event_channel_options,
                default=default_labels,
                key=f"{widget_prefix}_event_target_{event_key}",
                help="Choose one or more signal.channel targets where this event marker should appear.",
            )
            event_target_map[event_key] = picked_labels or default_labels

    if event_target_map != saved_event_channel_cfg:
        param_manager.add_to_config(entries={event_channel_map_key: event_target_map}, section="settings")

notes_to_plot = {}
state_annotations = {}
if event_df is not None and selected_event_keys:
    palette = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
        "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AB",
    ]
    for i, event_key in enumerate(selected_event_keys):
        style = event_style_defaults.get(event_key, {})
        target_signal = style.get("signal")
        if target_signal not in selected_plot_signals:
            target_signal = _infer_event_signal(event_key, selected_plot_signals)

        color = style.get("color", palette[i % len(palette)])
        state_color = style.get("state_color", "rgba(150, 150, 150, 0.25)")
        symbol = style.get("symbol", "circle")
        y_offset_frac = style.get("y_offset_frac", None)
        subset = event_df[event_df["key"] == event_key]
        type_series = subset.get("type")
        duration_series = subset.get("duration")
        is_state = False
        if type_series is not None:
            type_clean = type_series.astype(str).str.lower().str.strip()
            has_explicit_point = (type_clean == "point").any()
            has_explicit_state = (type_clean != "point").any()
            # Respect explicit event type first: point events may still carry duration.
            if has_explicit_point and not has_explicit_state:
                is_state = False
            elif has_explicit_state:
                is_state = True
            elif duration_series is not None:
                is_state = (pd.to_numeric(duration_series, errors="coerce").fillna(0) > 0).any()
        elif duration_series is not None:
            is_state = (pd.to_numeric(duration_series, errors="coerce").fillna(0) > 0).any()

        target_pairs = []
        if event_target_map.get(event_key):
            for label in event_target_map[event_key]:
                if "." not in label:
                    continue
                sig, ch = label.split(".", 1)
                if sig in selected_plot_signals and ch in plot_channels.get(sig, []):
                    target_pairs.append((sig, ch))
        if not target_pairs:
            # Saved targets didn't match current signal/channel selection — re-infer.
            marker_targets = _resolve_marker_signals(event_key, selected_plot_signals)
            marker_targets = [s for s in marker_targets if s in selected_plot_signals]
            if not marker_targets and target_signal is not None and target_signal in selected_plot_signals:
                marker_targets = [target_signal]
            for sig in marker_targets:
                chs = plot_channels.get(sig, [])
                if chs:
                    target_pairs.append((sig, chs[0]))
                    break  # one signal is enough for the fallback
        if not target_pairs:
            # Last resort: place on first plotted signal that has any channels.
            for sig in selected_plot_signals:
                chs = plot_channels.get(sig, [])
                if chs:
                    target_pairs.append((sig, chs[0]))
                    break
        if not target_pairs:
            continue

        # Keep heartbeat gaps visible both as points and interval shading.
        if event_key == "heartbeat_auto_detect_gap":
            notes_to_plot[event_key] = [
                {
                    "event_key": event_key,
                    "legend_key": event_key,
                    "name": event_key,
                    "signal": sig,
                    "channel": ch,
                    "symbol": symbol,
                    "color": color,
                    "showlegend": (j == 0),
                    **({} if y_offset_frac is None else {"y_offset_frac": y_offset_frac}),
                }
                for j, (sig, ch) in enumerate(target_pairs)
            ]
            state_annotations[event_key] = {"signal": target_pairs[0][0], "color": state_color}
            continue

        if is_state:
            state_annotations[event_key] = {"signal": target_pairs[0][0], "color": state_color}
        else:
            notes_to_plot[event_key] = [
                {
                    "event_key": event_key,
                    "legend_key": event_key,
                    "name": event_key,
                    "signal": sig,
                    "channel": ch,
                    "symbol": symbol,
                    "color": color,
                    "showlegend": (j == 0),
                    **({} if y_offset_frac is None else {"y_offset_frac": y_offset_frac}),
                }
                for j, (sig, ch) in enumerate(target_pairs)
            ]

if detection_mode == "heart_rate":
    zoom_channel = "depth" if "depth" in selected_plot_signals else selected_plot_signals[0]
    fig = plot_tag_data_interactive(
        data_pkl=data_pkl,
        signals=selected_plot_signals,
        channels=plot_channels,
        time_range=(start_datetime, end_datetime),
        note_annotations=notes_to_plot or None,
        state_annotations=state_annotations or None,
        color_mapping_path=color_mapping_path,
        target_sampling_rate=TARGET_SAMPLING_RATE,
        zoom_start_time=start_datetime,
        zoom_end_time=end_datetime,
        zoom_range_selector_channel=zoom_channel,
        plot_event_values=[],
    )
    run_t0 = _log_timing(run_t0, "Build heart-rate plot")

    # Update the legend position
    fig.update_layout(
        legend=dict(
            visible=False,
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
else:
    zoom_channel = "depth" if "depth" in selected_plot_signals else selected_plot_signals[0]
    fig = plot_tag_data_interactive_st(
        data_pkl=data_pkl,
        signals=selected_plot_signals,
        channels=plot_channels,
        time_range=(start_datetime, end_datetime),
        note_annotations=notes_to_plot or None,
        state_annotations=state_annotations or None,
        color_mapping_path=color_mapping_path,
        target_sampling_rate=TARGET_SAMPLING_RATE,
        zoom_start_time=start_datetime,
        zoom_end_time=end_datetime,
        zoom_range_selector_channel=zoom_channel,
        plot_event_values=[],
    )
    run_t0 = _log_timing(run_t0, "Build stroke-rate plot")

    # Update the legend position
    fig.update_layout(
        legend=dict(
            visible=False,
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

#fig.show()

st.plotly_chart(fig)
run_t0 = _log_timing(run_t0, "Render plotly chart")

if detection_mode == "heart_rate" and cleanup_diag:
    metric_order = [
        "rejected_below_min_peak_height",
        "rejected_above_max_peak_height",
        "rejected_conflict_pair_score",
        "rejected_up_jump_cleanup",
        "rejected_rr_too_short",
        "suggested_added",
        "final_accepted_or_suggested",
    ]
    metric_labels = {
        "rejected_below_min_peak_height": "Rejected: below MIN_PEAK_HEIGHT",
        "rejected_above_max_peak_height": "Rejected: above MAX_PEAK_HEIGHT",
        "rejected_conflict_pair_score": "Rejected: close-pair conflict cleanup",
        "rejected_up_jump_cleanup": "Rejected: HR up-jump cleanup",
        "rejected_rr_too_short": "Rejected: RR too short cleanup",
        "suggested_added": "Suggested beats added",
        "final_accepted_or_suggested": "Final accepted/suggested peaks",
    }
    metric_df = pd.DataFrame(
        {
            "metric": [metric_labels[k] for k in metric_order],
            "count": [int(cleanup_diag.get(k, 0)) for k in metric_order],
        }
    ).sort_values("count", ascending=False)
    def _bar_color(metric_name):
        name = str(metric_name).lower()
        if "rejected" in name:
            return "rgba(198, 104, 104, 0.85)"  # unsaturated red
        if "accepted" in name:
            return "rgba(104, 168, 124, 0.85)"  # unsaturated green
        if "suggested" in name:
            return "rgba(209, 176, 93, 0.85)"   # muted amber
        return "rgba(130, 130, 130, 0.75)"

    metric_df["color"] = metric_df["metric"].map(_bar_color)
    attribution_caption_placeholder.caption("Counts in current time window. Ordered by count.")
    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=metric_df["metric"],
                y=metric_df["count"],
                marker_color=metric_df["color"],
            )
        ]
    )
    bar_fig.update_layout(
        xaxis_title="Rule / Outcome",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=20, b=120),
        xaxis=dict(tickangle=-25),
    )
    attribution_plot_placeholder.plotly_chart(bar_fig, use_container_width=True)
elif detection_mode != "heart_rate":
    attribution_caption_placeholder.caption("Peak rejection attribution is available in heart_rate mode.")
    attribution_plot_placeholder.empty()
st.sidebar.markdown("### Timing (this run)")
for line in st.session_state.get("peak_detect_timing", []):
    st.sidebar.caption(line)
st.sidebar.caption(f"Total (auto-run to plot): {sum(float(x.split(': ')[1][:-1]) for x in st.session_state.get('peak_detect_timing', []) if x.endswith('s')):.2f}s")

# Debugging: Display updated params
st.write("Updated Parameters:", params)

# Add a button to clear intermediate signals in the Streamlit UI
if st.sidebar.button("Clear Intermediate Signals"):
    clear_intermediate_signals(data_pkl)
    st.sidebar.success("Intermediate signals cleared successfully!")
