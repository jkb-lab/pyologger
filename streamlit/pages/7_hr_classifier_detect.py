import os
import itertools
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from pyologger.process_data.peak_detect import peak_detect
from pyologger.process_data.sampling import calculate_sampling_frequency
from pyologger.plot_data.plotter import plot_tag_data_interactive
from pyologger.utils.folder_manager import load_configuration, select_and_load_deployment_streamlit
from pyologger.utils.streamlit_time_window import standardize_time_settings

try:
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    _SKTIME_AVAILABLE = True
except Exception:
    _SKTIME_AVAILABLE = False
    TimeSeriesForestClassifier = None

try:
    from sleepecg import detect_heartbeats as _sleepecg_detect_heartbeats
    _SLEEPECG_AVAILABLE = True
except Exception:
    _SLEEPECG_AVAILABLE = False
    _sleepecg_detect_heartbeats = None

try:
    import wfdb.processing as _wfdb_processing
    _WFDB_AVAILABLE = True
except Exception:
    _WFDB_AVAILABLE = False
    _wfdb_processing = None


def _to_tzaware(value, tz_name):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def _to_display_naive(value, tz_name):
    return _to_tzaware(value, tz_name).tz_localize(None)


def _from_display_naive(value, tz_name):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(tz_name)
    return ts.tz_convert(tz_name)


def _build_candidate_features(peak_df, raw_signal, smoothed, normalized, fs, half_window_samples):
    if peak_df.empty:
        return pd.DataFrame()
    raw = np.asarray(raw_signal, dtype=float)
    n = len(smoothed)
    rows = []
    for _, row in peak_df.iterrows():
        idx = int(row["refined_index"])
        if idx < 0 or idx >= n:
            continue
        lo = max(0, idx - half_window_samples)
        hi = min(n - 1, idx + half_window_samples)
        win_raw = raw[lo:hi + 1]
        win_sm = smoothed[lo:hi + 1]
        win_n = normalized[lo:hi + 1]
        left = normalized[lo:idx + 1]
        right = normalized[idx:hi + 1]
        prev_idx = int(row.get("prev_idx", -1))
        next_idx = int(row.get("next_idx", -1))
        rr_prev = (idx - prev_idx) / fs if prev_idx >= 0 else np.nan
        rr_next = (next_idx - idx) / fs if next_idx >= 0 else np.nan
        left_trough = float(np.min(left)) if left.size else np.nan
        right_trough = float(np.min(right)) if right.size else np.nan
        peak_norm = float(normalized[idx])
        peak_sm = float(smoothed[idx])
        prominence = peak_norm - max(left_trough, right_trough) if not np.isnan(left_trough) and not np.isnan(right_trough) else np.nan
        def _catch22_like(prefix, x):
            arr = np.asarray(x, dtype=float).ravel()
            if arr.size < 4:
                return {
                    f"{prefix}_ac1": np.nan,
                    f"{prefix}_zero_cross_rate": np.nan,
                    f"{prefix}_mean_abs_diff": np.nan,
                    f"{prefix}_entropy_bins10": np.nan,
                    f"{prefix}_above_mean_runmax": np.nan,
                }
            x0 = arr - np.nanmean(arr)
            v = np.nanvar(x0)
            ac1 = float(np.nanmean(x0[:-1] * x0[1:]) / v) if v > 1e-12 else np.nan
            sgn = np.sign(x0)
            zc = float(np.sum(sgn[:-1] * sgn[1:] < 0) / max(arr.size - 1, 1))
            mad1 = float(np.nanmean(np.abs(np.diff(arr))))
            hist, _ = np.histogram(arr[~np.isnan(arr)], bins=10)
            p = hist.astype(float)
            p = p / np.sum(p) if p.sum() > 0 else p
            p = p[p > 0]
            ent = float(-np.sum(p * np.log(p))) if p.size else np.nan
            above = x0 > 0
            run_max = 0
            run = 0
            for b in above:
                if b:
                    run += 1
                    run_max = max(run_max, run)
                else:
                    run = 0
            return {
                f"{prefix}_ac1": ac1,
                f"{prefix}_zero_cross_rate": zc,
                f"{prefix}_mean_abs_diff": mad1,
                f"{prefix}_entropy_bins10": ent,
                f"{prefix}_above_mean_runmax": float(run_max),
            }

        c_ecg = _catch22_like("c22_ecg", win_raw)
        c_sm = _catch22_like("c22_sm", win_sm)
        c_norm = _catch22_like("c22_norm", win_n)

        rows.append(
            {
                "refined_index": idx,
                "datetime": row.get("datetime", pd.NaT),
                "height_original": float(row.get("height_original", np.nan)),
                "height_normalized": float(row.get("height_normalized", np.nan)),
                "peak_ecg": float(raw[idx]),
                "peak_smoothed": peak_sm,
                "peak_normalized": peak_norm,
                "mean_ecg": float(np.mean(win_raw)) if win_raw.size else np.nan,
                "std_ecg": float(np.std(win_raw)) if win_raw.size else np.nan,
                "mean_smoothed": float(np.mean(win_sm)) if win_sm.size else np.nan,
                "std_smoothed": float(np.std(win_sm)) if win_sm.size else np.nan,
                "mean_norm": float(np.mean(win_n)) if win_n.size else np.nan,
                "std_norm": float(np.std(win_n)) if win_n.size else np.nan,
                "left_trough_norm": left_trough,
                "right_trough_norm": right_trough,
                "prominence_norm": prominence,
                "rr_prev_sec": rr_prev,
                "rr_next_sec": rr_next,
                "window_ecg_series": win_raw.astype(float),
                "window_series": win_n.astype(float),
                "window_smoothed_series": win_sm.astype(float),
                **c_ecg,
                **c_sm,
                **c_norm,
            }
        )
    feat = pd.DataFrame(rows).sort_values("refined_index").reset_index(drop=True)
    return feat


def _attach_neighbor_indices(df):
    if df.empty:
        return df
    idx = df["refined_index"].astype(int).to_numpy()
    prev_idx = np.roll(idx, 1)
    next_idx = np.roll(idx, -1)
    prev_idx[0] = -1
    next_idx[-1] = -1
    out = df.copy()
    out["prev_idx"] = prev_idx
    out["next_idx"] = next_idx
    return out


def _label_candidates(feat_df, peak_df, event_df, tolerance_sec=0.2, manual_only=False):
    if feat_df.empty:
        return pd.Series(dtype=int), "none"

    if manual_only:
        y = pd.Series(0, index=feat_df.index, dtype=int)
        label_source = "manual_ok_only"
    else:
        # Start with peak_detect labels (accepted/rejected by amplitude gates)
        base = peak_df.set_index("refined_index")["key"].to_dict()
        y = feat_df["refined_index"].map(lambda i: 1 if base.get(i) == "beat_auto_detect_accepted" else 0).astype(int)
        label_source = "peak_detect"

    if event_df is None or event_df.empty or "datetime" not in event_df or "key" not in event_df:
        return y, label_source

    if manual_only:
        accepted_keys = {"heartbeat_manual_ok"}
        rejected_keys = set()
    else:
        accepted_keys = {"heartbeat_auto_detect_accepted", "heartbeat_manual_ok"}
        rejected_keys = {"heartbeat_auto_detect_rejected"}
    subset = event_df[event_df["key"].isin(accepted_keys | rejected_keys)].copy()
    if subset.empty:
        return y, label_source

    subset["datetime"] = pd.to_datetime(subset["datetime"], errors="coerce")
    subset = subset.dropna(subset=["datetime"])
    if subset.empty:
        return y, label_source

    cand_time = pd.to_datetime(feat_df["datetime"], errors="coerce")
    tol = pd.Timedelta(seconds=tolerance_sec)
    for i, dt in enumerate(cand_time):
        if pd.isna(dt):
            continue
        delta = (subset["datetime"] - dt).abs()
        j = delta.idxmin()
        if pd.isna(delta.loc[j]) or delta.loc[j] > tol:
            continue
        k = subset.loc[j, "key"]
        if k in accepted_keys:
            y.iloc[i] = 1
            label_source = "event_overrides"
        elif k in rejected_keys:
            y.iloc[i] = 0
            label_source = "event_overrides"

    return y, label_source


def _train_classifier(X_feat, X_series, y, use_sktime, X_feat_pred=None, X_series_pred=None, rf_params=None):
    if len(np.unique(y)) < 2:
        return None, "Need both accepted and rejected labels in current window.", None
    if X_feat_pred is None:
        X_feat_pred = X_feat
    if X_series_pred is None:
        X_series_pred = X_series

    if use_sktime and _SKTIME_AVAILABLE:
        X_nested = pd.DataFrame({"ts": [pd.Series(v) for v in X_series]})
        X_nested_pred = pd.DataFrame({"ts": [pd.Series(v) for v in X_series_pred]})
        clf = TimeSeriesForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_nested, y)
        prob = _positive_class_proba(clf, X_nested_pred)
        return clf, None, prob

    rf_params = rf_params or {}
    clf = RandomForestClassifier(
        n_estimators=int(rf_params.get("n_estimators", 300)),
        min_samples_leaf=int(rf_params.get("min_samples_leaf", 2)),
        max_depth=(None if rf_params.get("max_depth") in (None, 0) else int(rf_params.get("max_depth"))),
        random_state=42,
        class_weight=rf_params.get("class_weight", "balanced_subsample"),
    )
    clf.fit(X_feat, y)
    prob = _positive_class_proba(clf, X_feat_pred)
    return clf, None, prob


def _positive_class_proba(clf, X):
    proba = clf.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba.astype(float)
    if proba.shape[1] == 0:
        return np.zeros(proba.shape[0], dtype=float)
    if proba.shape[1] == 1:
        classes = getattr(clf, "classes_", np.array([0]))
        cls = int(np.asarray(classes).ravel()[0]) if np.asarray(classes).size else 0
        fill = 1.0 if cls == 1 else 0.0
        return np.full(proba.shape[0], fill, dtype=float)
    classes = np.asarray(getattr(clf, "classes_", [0, 1]))
    if classes.size:
        pos_idx = np.where(classes == 1)[0]
        if pos_idx.size:
            return proba[:, int(pos_idx[0])]
    return proba[:, 1]


def _best_threshold_fbeta(y_true, prob, beta=2.0):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob).astype(float)
    if y.size == 0 or np.unique(y).size < 2:
        return 0.5

    beta2 = float(beta) ** 2
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if tp == 0:
            score = 0.0
        else:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            denom = (beta2 * precision) + recall
            score = ((1 + beta2) * precision * recall / denom) if denom > 0 else 0.0
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def _flatten_series_windows(series_list):
    n = len(series_list)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    max_len = max((len(np.asarray(v)) for v in series_list), default=0)
    out = np.zeros((n, max_len), dtype=float)
    for i, v in enumerate(series_list):
        arr = np.asarray(v, dtype=float)
        out[i, : min(len(arr), max_len)] = arr[:max_len]
    return out


def _compose_series_inputs(feat_df, selected_inputs):
    if feat_df.empty:
        return []
    input_map = {
        "ecg": "window_ecg_series",
        "hr_smoothed": "window_smoothed_series",
        "hr_normalized": "window_series",
    }
    cols = [input_map[k] for k in selected_inputs if input_map.get(k) in feat_df.columns]
    if not cols:
        cols = ["window_series"] if "window_series" in feat_df.columns else []
    out = []
    for i in range(len(feat_df)):
        parts = []
        for col in cols:
            arr = np.asarray(feat_df.iloc[i][col], dtype=float).ravel()
            if arr.size:
                parts.append(arr)
        if parts:
            out.append(np.concatenate(parts))
        else:
            out.append(np.array([], dtype=float))
    return out


def _build_fused_windows(feat_df, selected_inputs):
    """
    Build a single fused time-series window per candidate by averaging z-scored
    windows from selected channels (ecg, hr_smoothed, hr_normalized).
    """
    if feat_df.empty:
        return []
    input_map = {
        "ecg": "window_ecg_series",
        "hr_smoothed": "window_smoothed_series",
        "hr_normalized": "window_series",
    }
    cols = [input_map[k] for k in selected_inputs if input_map.get(k) in feat_df.columns]
    if not cols:
        cols = [c for c in ["window_series", "window_smoothed_series", "window_ecg_series"] if c in feat_df.columns]

    def _zscore(arr):
        x = np.asarray(arr, dtype=float).ravel()
        if x.size == 0:
            return x
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd <= 1e-12:
            return x - mu
        return (x - mu) / sd

    fused = []
    for i in range(len(feat_df)):
        parts = []
        for col in cols:
            arr = _zscore(feat_df.iloc[i][col])
            if arr.size:
                parts.append(arr)
        if not parts:
            fused.append(np.array([], dtype=float))
            continue
        max_len = max(len(p) for p in parts)
        padded = np.zeros((len(parts), max_len), dtype=float)
        for j, p in enumerate(parts):
            padded[j, : len(p)] = p
        fused.append(np.mean(padded, axis=0))
    return fused


def _predict_series_proba(model, series_list):
    if model is None:
        return np.zeros(len(series_list), dtype=float)

    if _SKTIME_AVAILABLE and isinstance(model, TimeSeriesForestClassifier):
        X_nested = pd.DataFrame({"ts": [pd.Series(v) for v in series_list]})
        return _positive_class_proba(model, X_nested)

    X = _flatten_series_windows(series_list)
    n_required = int(getattr(model, "n_features_in_", X.shape[1]))
    if X.shape[1] < n_required:
        pad = np.zeros((X.shape[0], n_required - X.shape[1]), dtype=float)
        X = np.hstack([X, pad])
    elif X.shape[1] > n_required:
        X = X[:, :n_required]
    return _positive_class_proba(model, X)


def _downsample_majority_indices(y, neg_pos_ratio=3.0, seed=42):
    y = np.asarray(y).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))
    max_neg = int(max(1, round(float(neg_pos_ratio) * len(pos))))
    if len(neg) <= max_neg:
        keep_neg = neg
    else:
        rng = np.random.default_rng(seed)
        keep_neg = rng.choice(neg, size=max_neg, replace=False)
    keep = np.sort(np.concatenate([pos, keep_neg]))
    return keep


def _fit_stickleback_style(X_series_train, y_train, X_series_pred, n_folds=4, beta=2.0, neg_pos_ratio=3.0):
    """
    Stickleback-like workflow:
    - under-sample majority (non-event) class
    - train a time-series classifier
    - tune decision threshold via internal CV (F-beta)
    """
    y = np.asarray(y_train).astype(int)
    if np.unique(y).size < 2:
        return None, "Need both classes in training span after labeling.", None, 0.5

    keep = _downsample_majority_indices(y, neg_pos_ratio=neg_pos_ratio, seed=42)
    y_ds = y[keep]
    X_train_sel = [X_series_train[i] for i in keep]
    if np.unique(y_ds).size < 2:
        return None, "Need both classes after downsampling.", None, 0.5

    n_splits = int(max(2, min(int(n_folds), int(np.bincount(y_ds).min()))))
    if n_splits < 2:
        return None, "Insufficient examples per class for CV.", None, 0.5

    if _SKTIME_AVAILABLE:
        X_nested = pd.DataFrame({"ts": [pd.Series(v) for v in X_train_sel]})
        oof_prob = np.zeros(len(y_ds), dtype=float)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, va in cv.split(np.zeros(len(y_ds)), y_ds):
            clf_cv = TimeSeriesForestClassifier(n_estimators=200, random_state=42)
            clf_cv.fit(X_nested.iloc[tr], y_ds[tr])
            oof_prob[va] = _positive_class_proba(clf_cv, X_nested.iloc[va])
        tuned_thresh = _best_threshold_fbeta(y_ds, oof_prob, beta=beta)

        clf = TimeSeriesForestClassifier(n_estimators=300, random_state=42)
        clf.fit(X_nested, y_ds)
        X_pred_nested = pd.DataFrame({"ts": [pd.Series(v) for v in X_series_pred]})
        prob_pred = _positive_class_proba(clf, X_pred_nested)
        return clf, None, prob_pred, tuned_thresh

    # Fallback if sktime is unavailable: flattened RF on sequence windows.
    X_flat = _flatten_series_windows(X_train_sel)
    X_pred_flat = _flatten_series_windows(X_series_pred)
    oof_prob = np.zeros(len(y_ds), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, va in cv.split(np.zeros(len(y_ds)), y_ds):
        clf_cv = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
        )
        clf_cv.fit(X_flat[tr], y_ds[tr])
        oof_prob[va] = _positive_class_proba(clf_cv, X_flat[va])
    tuned_thresh = _best_threshold_fbeta(y_ds, oof_prob, beta=beta)

    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(X_flat, y_ds)
    prob_pred = _positive_class_proba(clf, X_pred_flat)
    return clf, None, prob_pred, tuned_thresh


def _autobalance_single_class_labels(feat_df, y):
    """
    If labels collapse to a single class in the current window, synthesize the
    opposite class from extreme-ranked candidates so training can proceed.
    """
    y_bal = pd.Series(y).astype(int).copy()
    uniq = np.unique(y_bal)
    if uniq.size >= 2 or len(y_bal) < 6:
        return y_bal.to_numpy(), False

    # Prefer prominence; fall back to normalized then original height.
    score_col = None
    for c in ["prominence_norm", "peak_normalized", "height_normalized", "height_original"]:
        if c in feat_df.columns:
            col = pd.to_numeric(feat_df[c], errors="coerce")
            if col.notna().sum() >= 3:
                score_col = c
                break
    if score_col is None:
        return y_bal.to_numpy(), False

    score = pd.to_numeric(feat_df[score_col], errors="coerce")
    q = max(1, int(round(0.2 * len(y_bal))))  # 20% tail
    valid = score.dropna()
    if valid.empty:
        return y_bal.to_numpy(), False

    if uniq[0] == 1:
        # All positive -> lowest tail as pseudo negatives.
        low_idx = valid.nsmallest(q).index
        y_bal.loc[low_idx] = 0
    else:
        # All negative -> highest tail as pseudo positives.
        hi_idx = valid.nlargest(q).index
        y_bal.loc[hi_idx] = 1

    return y_bal.to_numpy(), (np.unique(y_bal).size >= 2)


def _manual_training_mask(feat_df, event_df):
    """
    Training mask restricted to the time span bounded by first/last heartbeat_manual_ok.
    """
    if feat_df.empty or event_df is None or event_df.empty:
        return np.zeros(len(feat_df), dtype=bool), None, None
    if "key" not in event_df.columns or "datetime" not in event_df.columns:
        return np.zeros(len(feat_df), dtype=bool), None, None

    manual = event_df[event_df["key"] == "heartbeat_manual_ok"].copy()
    if manual.empty:
        return np.zeros(len(feat_df), dtype=bool), None, None

    manual["datetime"] = pd.to_datetime(manual["datetime"], errors="coerce")
    manual = manual.dropna(subset=["datetime"])
    if manual.empty:
        return np.zeros(len(feat_df), dtype=bool), None, None

    t0 = manual["datetime"].min()
    t1 = manual["datetime"].max()
    cand_time = pd.to_datetime(feat_df["datetime"], errors="coerce")
    mask = cand_time.notna() & (cand_time >= t0) & (cand_time <= t1)
    return mask.to_numpy(), t0, t1


def _manual_positive_mask(feat_df, event_df, tolerance_sec=0.2):
    if feat_df.empty or event_df is None or event_df.empty:
        return np.zeros(len(feat_df), dtype=bool)
    if "datetime" not in event_df.columns or "key" not in event_df.columns:
        return np.zeros(len(feat_df), dtype=bool)
    manual = event_df[event_df["key"] == "heartbeat_manual_ok"].copy()
    if manual.empty:
        return np.zeros(len(feat_df), dtype=bool)
    manual_dt = pd.to_datetime(manual["datetime"], errors="coerce", utc=True).dropna()
    if manual_dt.empty:
        return np.zeros(len(feat_df), dtype=bool)
    cand_dt = pd.to_datetime(feat_df["datetime"], errors="coerce", utc=True)
    tol = pd.Timedelta(seconds=tolerance_sec)
    out = np.zeros(len(feat_df), dtype=bool)
    for i, dt in enumerate(cand_dt):
        if pd.isna(dt):
            continue
        delta = (manual_dt - dt).abs()
        if not delta.empty and delta.min() <= tol:
            out[i] = True
    return out


def _fit_unsupervised_tuned(feat_df, train_mask, manual_pos_mask, selected_inputs=None):
    """
    Unsupervised clustering on fused windows from selected inputs
    (ecg/hr_smoothed/hr_normalized), tuned to maximize F1 overlap with manual_ok.
    """
    n = len(feat_df)
    if n == 0:
        return np.zeros(0, dtype=bool), {"score": 0.0, "inputs": []}

    fused = _build_fused_windows(feat_df, selected_inputs or ["ecg", "hr_smoothed", "hr_normalized"])
    max_len = max((len(x) for x in fused), default=0)
    if max_len == 0:
        return np.zeros(n, dtype=bool), {"score": 0.0, "inputs": selected_inputs or []}

    def _pad(v, L):
        arr = np.asarray(v, dtype=float)
        if L <= 0:
            return np.array([], dtype=float)
        if arr.size >= L:
            return arr[:L]
        out = np.zeros(L, dtype=float)
        out[:arr.size] = arr
        return out

    X = np.vstack([_pad(v, max_len) for v in fused]) if max_len > 0 else np.zeros((n, 0))

    train_mask = np.asarray(train_mask, dtype=bool)
    manual_pos_mask = np.asarray(manual_pos_mask, dtype=bool)
    if train_mask.sum() < 4:
        return np.zeros(n, dtype=bool), {"score": 0.0, "inputs": selected_inputs or []}

    if X.shape[1] == 0:
        return np.zeros(n, dtype=bool), {"score": 0.0, "inputs": selected_inputs or []}

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    km = KMeans(n_clusters=2, random_state=42, n_init=20)
    km.fit(Xz[train_mask])
    labels_all = km.predict(Xz)

    y_true = manual_pos_mask[train_mask].astype(int)
    lab = labels_all[train_mask]
    best_local_score = -1.0
    best_pos_cluster = 0
    for pos_cluster in [0, 1]:
        pred = (lab == pos_cluster).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if score > best_local_score:
            best_local_score = score
            best_pos_cluster = pos_cluster

    accepted = labels_all == best_pos_cluster
    return accepted, {"score": float(best_local_score), "inputs": selected_inputs or []}


def _build_hr_from_ml(feat_df, accepted_mask, signal_subset_df, fs, extra_indices=None):
    if feat_df.empty:
        return pd.DataFrame(columns=["datetime", "heart_rate_ml"]), []
    picked = feat_df.loc[accepted_mask].sort_values("refined_index").reset_index(drop=True)
    if extra_indices:
        extra = pd.DataFrame({"refined_index": [int(i) for i in extra_indices]})
        picked = pd.concat([picked, extra], ignore_index=True).drop_duplicates(subset=["refined_index"]).sort_values("refined_index").reset_index(drop=True)
    n = len(signal_subset_df)
    hr = np.full(n, np.nan, dtype=float)
    if len(picked) > 1:
        for i in range(len(picked) - 1):
            s = int(picked.loc[i, "refined_index"])
            e = int(picked.loc[i + 1, "refined_index"])
            if e <= s:
                continue
            rr = (e - s) / fs
            if rr <= 0:
                continue
            hr[s:e] = 60.0 / rr
        hr[picked.loc[len(picked) - 1, "refined_index"]:] = hr[np.where(~np.isnan(hr))[0][-1]] if np.any(~np.isnan(hr)) else np.nan
    hr = pd.Series(hr).interpolate(limit_direction="both").to_numpy()
    return pd.DataFrame({"datetime": signal_subset_df["datetime"], "heart_rate_ml": hr}), picked["refined_index"].astype(int).tolist()


def _build_prob_signal_from_candidates(feat_df, signal_subset_df, rolling_window_sec=1.0):
    """
    Build continuous 25 Hz rolling probability across the whole selected window,
    and sample that local probability at candidate times.
    """
    n_cand = len(feat_df)
    cand_prob = np.full(n_cand, np.nan, dtype=float)
    dt_full = pd.to_datetime(signal_subset_df.get("datetime"), errors="coerce")
    dt_full = dt_full.dropna()
    if dt_full.empty:
        return pd.DataFrame(columns=["datetime", "heart_rate_ml_prob"]), cand_prob

    start = dt_full.min()
    end = dt_full.max()
    if end <= start:
        end = start + pd.Timedelta(milliseconds=40)
    grid = pd.date_range(start=start, end=end, freq="40ms")
    if grid.empty:
        grid = pd.DatetimeIndex([start])

    if feat_df.empty or "accept_prob" not in feat_df.columns:
        out = pd.DataFrame({"datetime": grid, "heart_rate_ml_prob": np.full(len(grid), np.nan, dtype=float)})
        return out, cand_prob

    cand_dt = pd.to_datetime(feat_df.get("datetime"), errors="coerce")
    cand_raw_prob = pd.to_numeric(feat_df.get("accept_prob"), errors="coerce")
    valid = cand_dt.notna() & cand_raw_prob.notna()
    if not valid.any():
        out = pd.DataFrame({"datetime": grid, "heart_rate_ml_prob": np.full(len(grid), np.nan, dtype=float)})
        return out, cand_prob

    s = pd.Series(cand_raw_prob[valid].to_numpy(dtype=float), index=pd.DatetimeIndex(cand_dt[valid]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        out = pd.DataFrame({"datetime": grid, "heart_rate_ml_prob": np.full(len(grid), np.nan, dtype=float)})
        return out, cand_prob

    s25 = s.reindex(grid.union(s.index)).sort_index().interpolate(method="time").reindex(grid).ffill().bfill()
    win_n = max(1, int(round(float(rolling_window_sec) * 25.0)))
    local25 = s25.rolling(window=win_n, min_periods=1, center=True).mean().clip(0.0, 1.0)
    out = pd.DataFrame({"datetime": local25.index, "heart_rate_ml_prob": local25.to_numpy(dtype=float)})

    cand_dt_all = pd.to_datetime(feat_df.get("datetime"), errors="coerce")
    ok = cand_dt_all.notna().to_numpy()
    if ok.any() and not out.empty:
        x = out["datetime"].astype("int64").to_numpy()
        y = out["heart_rate_ml_prob"].to_numpy(dtype=float)
        cand_x = cand_dt_all[ok].astype("int64").to_numpy()
        cand_prob[ok] = np.interp(cand_x, x, y, left=y[0], right=y[-1])

    return out, cand_prob


def _build_dense_peak_df(smoothed, normalized, datetime_series, fs, base_peak_distance_sec):
    norm = np.asarray(normalized, dtype=float)
    sm = np.asarray(smoothed, dtype=float)
    dt = pd.to_datetime(datetime_series, errors="coerce")
    if norm.size == 0:
        return pd.DataFrame(columns=["refined_index", "height_original", "height_normalized", "datetime"])
    min_dist = max(1, int(round(max(0.04, float(base_peak_distance_sec) * 0.5) * float(fs))))
    dense_idx, _ = find_peaks(norm, distance=min_dist)
    if dense_idx.size == 0:
        return pd.DataFrame(columns=["refined_index", "height_original", "height_normalized", "datetime"])
    out = pd.DataFrame(
        {
            "refined_index": dense_idx.astype(int),
            "height_original": sm[dense_idx],
            "height_normalized": norm[dense_idx],
            "datetime": dt.iloc[dense_idx].to_numpy(),
        }
    )
    return out.sort_values("refined_index").reset_index(drop=True)


def _find_down_jump_insertions(
    feat_df,
    accepted_mask,
    raw_signal,
    smoothed,
    signal_subset_df,
    fs,
    prob_signal_df,
    prob_threshold,
    hr_jump_frac=0.8,
    min_peak_height=0.0,
    gap_repair_method="auto",
):
    """
    Detect large down-jumps in interval HR among accepted peaks, then re-search the
    prior interval for a missed local max and insert if probability/height criteria pass.
    """
    if feat_df.empty:
        return pd.DataFrame(columns=["datetime", "refined_index", "accept_prob"])
    kept = feat_df.loc[np.asarray(accepted_mask, dtype=bool)].copy()
    if len(kept) < 3:
        return pd.DataFrame(columns=["datetime", "refined_index", "accept_prob"])
    kept["refined_index"] = pd.to_numeric(kept["refined_index"], errors="coerce")
    kept = kept.dropna(subset=["refined_index"]).sort_values("refined_index").reset_index(drop=True)
    if len(kept) < 3:
        return pd.DataFrame(columns=["datetime", "refined_index", "accept_prob"])

    idxs = kept["refined_index"].astype(int).to_numpy()
    rr = np.diff(idxs).astype(float) / max(float(fs), 1e-9)
    rr = np.where(rr <= 0, np.nan, rr)
    hr = 60.0 / rr
    if len(hr) < 2:
        return pd.DataFrame(columns=["datetime", "refined_index", "accept_prob"])
    frac = (hr[1:] - hr[:-1]) / np.where(np.abs(hr[:-1]) < 1e-9, np.nan, hr[:-1])
    down_pos = np.where(np.isfinite(frac) & (frac < -float(hr_jump_frac)))[0]
    if down_pos.size == 0:
        return pd.DataFrame(columns=["datetime", "refined_index", "accept_prob"])

    # Build interpolation function for local probability by datetime.
    pdt = pd.to_datetime(prob_signal_df.get("datetime"), errors="coerce")
    pp = pd.to_numeric(prob_signal_df.get("heart_rate_ml_prob"), errors="coerce")
    pvalid = pdt.notna() & pp.notna()
    if pvalid.any():
        p_x = pdt[pvalid].astype("int64").to_numpy()
        p_y = pp[pvalid].to_numpy(dtype=float)
    else:
        p_x = np.array([], dtype=np.int64)
        p_y = np.array([], dtype=float)

    n = len(signal_subset_df)
    sm = np.asarray(smoothed, dtype=float)
    raw = np.asarray(raw_signal, dtype=float)
    occupied = set(idxs.tolist())
    rows = []

    def _library_rel_candidates(seg_raw, fs_hz, method):
        rel = []
        if seg_raw.size < 8:
            return rel
        m = str(method or "auto").lower()
        want_sleep = m in {"auto", "sleepecg"}
        want_wfdb = m in {"auto", "wfdb_xqrs"}
        if want_sleep and _SLEEPECG_AVAILABLE:
            try:
                c = _sleepecg_detect_heartbeats(seg_raw.astype(float), fs=int(round(fs_hz)))
                if c is not None and len(c) > 0:
                    rel.extend(np.asarray(c, dtype=int).tolist())
            except Exception:
                pass
        if want_wfdb and _WFDB_AVAILABLE:
            try:
                xqrs = _wfdb_processing.XQRS(sig=seg_raw.astype(float), fs=float(fs_hz))
                xqrs.detect(verbose=False)
                c = np.asarray(getattr(xqrs, "qrs_inds", []), dtype=int)
                if c.size > 0:
                    rel.extend(c.tolist())
            except Exception:
                pass
        if not rel:
            return []
        rel = sorted(set(int(x) for x in rel if 0 <= int(x) < seg_raw.size))
        return rel
    for pos in down_pos:
        this_pos = int(pos + 2)  # align with accepted index that received the down-jump
        if this_pos >= len(idxs):
            continue
        prev_idx = int(idxs[this_pos - 1])
        this_idx = int(idxs[this_pos])
        if this_idx - prev_idx < 3:
            continue
        if prev_idx < 0 or this_idx > len(sm):
            continue
        seg_sm = sm[prev_idx:this_idx]
        seg_raw = raw[prev_idx:this_idx] if raw.size >= this_idx else np.array([], dtype=float)
        if seg_sm.size < 3:
            continue
        cand_abs = {int(prev_idx + np.argmax(seg_sm))}  # local fallback
        if str(gap_repair_method).lower() != "local_only":
            for rel_idx in _library_rel_candidates(seg_raw, fs, gap_repair_method):
                cand_abs.add(prev_idx + int(rel_idx))

        best = None
        for ins_idx in sorted(cand_abs):
            if ins_idx in occupied:
                continue
            if ins_idx < 0 or ins_idx >= n:
                continue
            peak_h = float(sm[ins_idx])
            if peak_h < float(min_peak_height):
                continue
            dt = pd.to_datetime(signal_subset_df["datetime"].iloc[ins_idx], errors="coerce")
            if pd.isna(dt):
                continue
            if p_x.size > 0:
                p_val = float(np.interp(int(dt.value), p_x, p_y, left=p_y[0], right=p_y[-1]))
            else:
                p_val = np.nan
            if np.isfinite(p_val) and p_val < float(prob_threshold):
                continue
            candidate = {"datetime": dt, "refined_index": ins_idx, "accept_prob": p_val, "peak_h": peak_h}
            if best is None:
                best = candidate
            else:
                bp = -np.inf if not np.isfinite(best["accept_prob"]) else best["accept_prob"]
                cp = -np.inf if not np.isfinite(candidate["accept_prob"]) else candidate["accept_prob"]
                if (cp > bp) or (cp == bp and candidate["peak_h"] > best["peak_h"]):
                    best = candidate
        if best is not None:
            rows.append({"datetime": best["datetime"], "refined_index": best["refined_index"], "accept_prob": best["accept_prob"]})
            occupied.add(int(best["refined_index"]))

    if not rows:
        return pd.DataFrame(columns=["datetime", "refined_index", "accept_prob"])
    return pd.DataFrame(rows).drop_duplicates(subset=["refined_index"]).sort_values("refined_index").reset_index(drop=True)


def _apply_ml_peak_cleanup(
    feat_df,
    accepted_mask,
    sampling_rate,
    do_conflict_cleanup=True,
    do_up_jump_cleanup=True,
    conflict_rr_factor=0.75,
    hr_jump_frac=0.8,
    pick_last_in_conflict=True,
):
    """
    Optional post-ML cleanup on accepted peaks:
    - conflict cleanup for too-close pairs
    - up-jump cleanup for abrupt HR increases
    """
    kept = np.asarray(accepted_mask, dtype=bool).copy()
    if feat_df.empty or kept.sum() < 2:
        return kept

    def _ordered_kept_indices(mask):
        sub = feat_df.loc[mask, ["refined_index", "accept_prob_local", "accept_prob"]].copy()
        sub["refined_index"] = pd.to_numeric(sub["refined_index"], errors="coerce")
        sub = sub.dropna(subset=["refined_index"]).sort_values("refined_index").reset_index()
        return sub

    # 1) Conflict cleanup: remove too-close peaks based on local median RR.
    if do_conflict_cleanup:
        changed = True
        while changed:
            changed = False
            sub = _ordered_kept_indices(kept)
            if len(sub) < 3:
                break
            idx = sub["refined_index"].astype(int).to_numpy()
            rr = np.diff(idx).astype(float)
            med_rr = float(np.nanmedian(rr)) if rr.size else np.nan
            if not np.isfinite(med_rr) or med_rr <= 0:
                break
            min_rr = max(1.0, float(conflict_rr_factor) * med_rr)
            bad_pos = np.where(rr < min_rr)[0]
            if bad_pos.size == 0:
                break
            p = int(bad_pos[0])
            left_row = sub.iloc[p]
            right_row = sub.iloc[p + 1]
            if pick_last_in_conflict:
                drop_global = int(left_row["index"])
            else:
                lp = float(pd.to_numeric(left_row.get("accept_prob_local"), errors="coerce") if pd.notna(left_row.get("accept_prob_local")) else pd.to_numeric(left_row.get("accept_prob"), errors="coerce"))
                rp = float(pd.to_numeric(right_row.get("accept_prob_local"), errors="coerce") if pd.notna(right_row.get("accept_prob_local")) else pd.to_numeric(right_row.get("accept_prob"), errors="coerce"))
                drop_global = int(left_row["index"] if lp <= rp else right_row["index"])
            kept[drop_global] = False
            changed = True

    # 2) Up-jump cleanup: remove peaks driving abrupt HR increases.
    if do_up_jump_cleanup:
        changed = True
        fs = float(sampling_rate)
        while changed:
            changed = False
            sub = _ordered_kept_indices(kept)
            if len(sub) < 4:
                break
            idx = sub["refined_index"].astype(int).to_numpy()
            rr = np.diff(idx).astype(float) / max(fs, 1e-9)
            rr = np.where(rr <= 0, np.nan, rr)
            hr = 60.0 / rr
            if len(hr) < 2:
                break
            jump_pos = np.where((np.isfinite(hr[1:])) & (np.isfinite(hr[:-1])) & (hr[1:] > hr[:-1] * (1.0 + float(hr_jump_frac))))[0]
            if jump_pos.size == 0:
                break
            j = int(jump_pos[0] + 1)
            # hr[j] is interval between idx[j] and idx[j+1]; drop lower-prob endpoint.
            left_row = sub.iloc[j]
            right_row = sub.iloc[j + 1]
            lp = float(pd.to_numeric(left_row.get("accept_prob_local"), errors="coerce") if pd.notna(left_row.get("accept_prob_local")) else pd.to_numeric(left_row.get("accept_prob"), errors="coerce"))
            rp = float(pd.to_numeric(right_row.get("accept_prob_local"), errors="coerce") if pd.notna(right_row.get("accept_prob_local")) else pd.to_numeric(right_row.get("accept_prob"), errors="coerce"))
            drop_global = int(left_row["index"] if lp <= rp else right_row["index"])
            kept[drop_global] = False
            changed = True

    return kept


def _apply_smoothness_penalty(
    feat_df,
    base_score,
    hr_jump_frac=0.8,
    smoothness_penalty=0.0,
    min_hr_bpm=0.1,
    max_hr_bpm=240.0,
):
    """
    Reduce candidate probability when local HR transitions are too abrupt.
    Penalty is soft (exponential), not a hard reject.
    """
    score = np.asarray(base_score, dtype=float).copy()
    if score.size == 0:
        return score
    lam = max(0.0, float(smoothness_penalty))
    if lam <= 0:
        return np.clip(score, 0.0, 1.0)

    rr_prev = pd.to_numeric(feat_df.get("rr_prev_sec"), errors="coerce").to_numpy(dtype=float)
    rr_next = pd.to_numeric(feat_df.get("rr_next_sec"), errors="coerce").to_numpy(dtype=float)
    prev_hr = np.where(rr_prev > 0, 60.0 / rr_prev, np.nan)
    next_hr = np.where(rr_next > 0, 60.0 / rr_next, np.nan)

    # Soft penalties for abrupt jump and implausible HR range.
    jump = np.abs(next_hr - prev_hr) / np.maximum(np.abs(prev_hr), 1e-9)
    jump_excess = np.where(np.isfinite(jump), np.maximum(0.0, jump - float(hr_jump_frac)), 0.0)

    low_excess_prev = np.maximum(0.0, float(min_hr_bpm) - np.nan_to_num(prev_hr, nan=float(min_hr_bpm)))
    high_excess_prev = np.maximum(0.0, np.nan_to_num(prev_hr, nan=float(max_hr_bpm)) - float(max_hr_bpm))
    low_excess_next = np.maximum(0.0, float(min_hr_bpm) - np.nan_to_num(next_hr, nan=float(min_hr_bpm)))
    high_excess_next = np.maximum(0.0, np.nan_to_num(next_hr, nan=float(max_hr_bpm)) - float(max_hr_bpm))
    range_excess = (low_excess_prev + high_excess_prev + low_excess_next + high_excess_next) / max(float(max_hr_bpm), 1.0)

    total_excess = np.nan_to_num(jump_excess, nan=0.0) + np.nan_to_num(range_excess, nan=0.0)
    penalized = score * np.exp(-lam * total_excess)
    return np.clip(penalized, 0.0, 1.0)


def _precision_recall_f1(y_true, y_pred):
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def _event_match_precision_recall(accepted_dt, manual_dt, tolerance_sec=0.2):
    """
    Event-level precision/recall with greedy one-to-one matching in time.
    """
    acc = pd.to_datetime(pd.Series(accepted_dt), errors="coerce", utc=True).dropna().sort_values().to_numpy()
    man = pd.to_datetime(pd.Series(manual_dt), errors="coerce", utc=True).dropna().sort_values().to_numpy()
    if len(acc) == 0 and len(man) == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0
    if len(man) == 0:
        return 0.0, 0.0, 0.0, 0, len(acc), 0
    if len(acc) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(man)

    tol_ns = int(pd.Timedelta(seconds=float(tolerance_sec)).value)
    ai = 0
    mi = 0
    tp = 0
    used_a = np.zeros(len(acc), dtype=bool)
    used_m = np.zeros(len(man), dtype=bool)
    while ai < len(acc) and mi < len(man):
        da = int(pd.Timestamp(acc[ai]).value)
        dm = int(pd.Timestamp(man[mi]).value)
        delta = da - dm
        if abs(delta) <= tol_ns:
            if not used_a[ai] and not used_m[mi]:
                tp += 1
                used_a[ai] = True
                used_m[mi] = True
            ai += 1
            mi += 1
        elif da < dm - tol_ns:
            ai += 1
        else:
            mi += 1

    fp = int((~used_a).sum())
    fn = int((~used_m).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, int(tp), int(fp), int(fn)


def _upsert_ml_events(data_pkl, feat_df, accepted_mask, suggested_df=None):
    if not hasattr(data_pkl, "event_data") or data_pkl.event_data is None:
        data_pkl.event_data = pd.DataFrame(columns=["datetime", "key", "short_description", "type", "duration", "value"])
    cleanup = {"heartbeat_ml_detect_accepted", "heartbeat_ml_detect_rejected", "heartbeat_ml_detect_suggested"}
    if "key" in data_pkl.event_data.columns:
        data_pkl.event_data = data_pkl.event_data[~data_pkl.event_data["key"].isin(cleanup)].copy()

    ev = feat_df.copy()
    ev["key"] = np.where(accepted_mask, "heartbeat_ml_detect_accepted", "heartbeat_ml_detect_rejected")
    ev["short_description"] = np.where(
        accepted_mask,
        "ML classifier accepted heartbeat candidate",
        "ML classifier rejected heartbeat candidate",
    )
    ev["type"] = "point"
    ev["duration"] = 0.0
    ev["value"] = np.nan
    add = ev[["datetime", "key", "short_description", "type", "duration", "value"]]
    if suggested_df is not None and not suggested_df.empty:
        sug = suggested_df.copy()
        sug["key"] = "heartbeat_ml_detect_suggested"
        sug["short_description"] = "ML suggested peak outside base detector candidates"
        sug["type"] = "point"
        sug["duration"] = 0.0
        sug["value"] = np.nan
        add = pd.concat([add, sug[["datetime", "key", "short_description", "type", "duration", "value"]]], ignore_index=True)
    data_pkl.event_data = pd.concat([data_pkl.event_data, add], ignore_index=True)


config, data_dir, color_mapping_path, montage_path = load_configuration()
st.sidebar.title("Deployment Selection")
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment_streamlit(data_dir)
if not dataset_id or not deployment_id:
    st.sidebar.warning("Select dataset/deployment.")
    st.stop()
st.sidebar.write(f"📂 Selected Deployment: {deployment_id}")

timezone = data_pkl.deployment_info.get("Time Zone", "UTC")
DEPLOYMENT_TZ = str(timezone or "UTC")

st.title("HR Classifier Detect (Alternative)")
st.caption("Windowed classifier approach seeded by hr_smoothed/hr_normalized features.")
widget_prefix = f"{dataset_id}_{deployment_id}_hr_classifier".replace(":", "_").replace("/", "_")

ml_cfg = param_manager.get_from_config(
    [
        "HR_ML_ACCEPT_PROB_THRESHOLD",
        "HR_ML_FEATURE_WINDOW_SEC",
        "HR_ML_INPUT_SIGNALS",
        "ANTI_DOUBLE_GAP_FACTOR",
        "HR_JUMP_FRAC",
        "HR_SMOOTHNESS_PENALTY",
        "PROMINENCE_NORM_MIN",
        "MIN_PEAK_HEIGHT",
        "MIN_HR_BPM",
        "MAX_HR_BPM",
        "RF_N_ESTIMATORS",
        "RF_MIN_SAMPLES_LEAF",
        "RF_MAX_DEPTH",
        "RF_CLASS_WEIGHT",
    ],
    section="hr_peak_detection_settings",
)

standardized = standardize_time_settings(param_manager, data_pkl, tz_name=DEPLOYMENT_TZ, minutes=2, persist=False)
start_default = _to_display_naive(standardized["zoom_window_start_time"], DEPLOYMENT_TZ).to_pydatetime()
end_default = _to_display_naive(standardized["zoom_window_end_time"], DEPLOYMENT_TZ).to_pydatetime()
overlap_start = _to_display_naive(standardized["overlap_start_time"], DEPLOYMENT_TZ).to_pydatetime()
overlap_end = _to_display_naive(standardized["overlap_end_time"], DEPLOYMENT_TZ).to_pydatetime()

st.sidebar.subheader("ML Parameters")
start_key = f"{widget_prefix}_start_text"
end_key = f"{widget_prefix}_end_text"
pending_start_key = f"{widget_prefix}_pending_start_text"
pending_end_key = f"{widget_prefix}_pending_end_text"
jump_msg_key = f"{widget_prefix}_jump_message"
# Apply pending jump values before widgets are instantiated.
if pending_start_key in st.session_state and pending_end_key in st.session_state:
    st.session_state[start_key] = st.session_state[pending_start_key]
    st.session_state[end_key] = st.session_state[pending_end_key]
    del st.session_state[pending_start_key]
    del st.session_state[pending_end_key]
start_text = st.sidebar.text_input(
    "Start (YYYY-MM-DD HH:MM:SS)",
    value=start_default.strftime("%Y-%m-%d %H:%M:%S"),
    key=start_key,
)
end_text = st.sidebar.text_input(
    "End (YYYY-MM-DD HH:MM:SS)",
    value=end_default.strftime("%Y-%m-%d %H:%M:%S"),
    key=end_key,
)
classifier_options = ["stickleback_style", "unsupervised_tuned", "sklearn_rf"]
if _SKTIME_AVAILABLE:
    classifier_options.append("sktime_tsf")
classifier_mode = "sklearn_rf"
st.sidebar.caption("Classifier: Random Forest (focused mode)")
prob_thresh = st.sidebar.slider(
    "ML_ACCEPT_PROB",
    min_value=0.0,
    max_value=1.0,
    value=float(ml_cfg.get("HR_ML_ACCEPT_PROB_THRESHOLD") or 0.5),
    step=0.01,
)
window_sec = st.sidebar.number_input(
    "FEATURE_WINDOW_SEC",
    min_value=0.05,
    max_value=5.0,
    value=float(ml_cfg.get("HR_ML_FEATURE_WINDOW_SEC") or 0.6),
    step=0.05,
)
manual_only_labels = st.sidebar.checkbox("Train from heartbeat_manual_ok only", value=True)
ml_input_sources = st.sidebar.multiselect(
    "ML input signals",
    options=["ecg", "hr_smoothed", "hr_normalized"],
    default=(ml_cfg.get("HR_ML_INPUT_SIGNALS") if isinstance(ml_cfg.get("HR_ML_INPUT_SIGNALS"), list) else ["ecg", "hr_smoothed", "hr_normalized"]),
    help="Windows from selected signals are concatenated for classifier input.",
)
if not ml_input_sources:
    ml_input_sources = ["hr_normalized"]

st.sidebar.markdown("**RF Controls**")
rf_n_estimators = st.sidebar.number_input("RF_N_ESTIMATORS", min_value=50, max_value=2000, value=int(ml_cfg.get("RF_N_ESTIMATORS") or 400), step=50)
rf_min_samples_leaf = st.sidebar.number_input("RF_MIN_SAMPLES_LEAF", min_value=1, max_value=50, value=int(ml_cfg.get("RF_MIN_SAMPLES_LEAF") or 2), step=1)
rf_max_depth = st.sidebar.number_input("RF_MAX_DEPTH (0=None)", min_value=0, max_value=100, value=int(ml_cfg.get("RF_MAX_DEPTH") or 0), step=1)
_rf_cw_saved = str(ml_cfg.get("RF_CLASS_WEIGHT") or "balanced_subsample")
rf_class_weight_mode = st.sidebar.selectbox("RF_CLASS_WEIGHT", ["balanced_subsample", "balanced", "none"], index=(["balanced_subsample", "balanced", "none"].index(_rf_cw_saved) if _rf_cw_saved in ["balanced_subsample", "balanced", "none"] else 0))

st.sidebar.markdown("**RF Feature Groups**")
use_feature_rr = st.sidebar.checkbox("Use RR Features", value=True)
use_feature_prom = st.sidebar.checkbox("Use Prominence/Trough Features", value=True)
use_feature_peak = st.sidebar.checkbox("Use Peak Amplitude Features", value=True)
use_feature_stats = st.sidebar.checkbox("Use Local Mean/Std Features", value=True)
use_feature_catch22 = st.sidebar.checkbox("Use Catch22-like Features", value=True)

auto_sensitive_thresh = True
sensitivity_beta = 2.0
min_user_thresh = 0.10
sb_neg_pos_ratio = 3.0
sb_folds = 4
sb_suggest = False
sb_suggest_thresh = 0.45
jump_to_first_manual = False
do_conflict_cleanup = True
do_up_jump_cleanup = True
do_down_jump_insert = True
gap_repair_method = "auto"
conflict_rr_factor = 0.75
up_jump_frac = 0.8
smoothness_penalty = 0.0
prominence_norm_min = 0.0
min_peak_height_ml = 0.0
min_hr_bpm_ml = 0.1
max_hr_bpm_ml = 240.0

conflict_rr_factor = st.sidebar.number_input("ANTI_DOUBLE_GAP_FACTOR", min_value=0.1, max_value=1.5, value=float(ml_cfg.get("ANTI_DOUBLE_GAP_FACTOR") or 0.75), step=0.05)
up_jump_frac = st.sidebar.number_input("HR_JUMP_FRAC", min_value=0.1, max_value=3.0, value=float(ml_cfg.get("HR_JUMP_FRAC") or 0.8), step=0.1)
smoothness_penalty = st.sidebar.number_input("HR_SMOOTHNESS_PENALTY", min_value=0.0, max_value=10.0, value=float(ml_cfg.get("HR_SMOOTHNESS_PENALTY") or 1.0), step=0.1)
prominence_norm_min = st.sidebar.number_input("PROMINENCE_NORM_MIN", min_value=-10.0, max_value=20.0, value=float(ml_cfg.get("PROMINENCE_NORM_MIN") or 0.0), step=0.05)
min_peak_height_ml = st.sidebar.number_input("MIN_PEAK_HEIGHT", min_value=-100000.0, max_value=100000.0, value=float(ml_cfg.get("MIN_PEAK_HEIGHT") or 70.0), step=1.0)
min_hr_bpm_ml = st.sidebar.number_input("MIN_HR_BPM", min_value=0.01, max_value=120.0, value=float(ml_cfg.get("MIN_HR_BPM") or 0.1), step=0.1)
max_hr_bpm_ml = st.sidebar.number_input("MAX_HR_BPM", min_value=1.0, max_value=400.0, value=float(ml_cfg.get("MAX_HR_BPM") or 240.0), step=1.0)
min_user_thresh = st.sidebar.number_input("MIN_PROB_THRESHOLD", min_value=0.01, max_value=0.99, value=0.10, step=0.01)
do_conflict_cleanup = st.sidebar.checkbox("ENABLE_CONFLICT_CLEANUP", value=True)
do_up_jump_cleanup = st.sidebar.checkbox("ENABLE_UP_JUMP_CLEANUP", value=True)
do_down_jump_insert = st.sidebar.checkbox("ENABLE_DOWN_JUMP_INSERT", value=True)
gap_repair_method = st.sidebar.selectbox("GAP_REPAIR_METHOD", ["auto", "sleepecg", "wfdb_xqrs", "local_only"], index=0)
jump_to_first_manual = st.sidebar.checkbox("JUMP_TO_FIRST_MANUAL_OK", value=False)

with st.sidebar.expander("Method Specific", expanded=False):
    if classifier_mode == "stickleback_style":
        sb_suggest = st.checkbox("ENABLE_SB_SUGGEST", value=True)
        sb_suggest_thresh = st.slider("SB_SUGGEST_PROB", min_value=0.0, max_value=1.0, value=0.45, step=0.01)

with st.sidebar.expander("Parameter Search", expanded=False):
    search_objective = st.selectbox(
        "SEARCH_OBJECTIVE",
        options=["balanced_pr", "f1", "precision", "recall"],
        index=0,
        help="balanced_pr maximizes precision*recall.",
    )
    search_points = st.number_input("SEARCH_POINTS_PER_PARAM", min_value=2, max_value=7, value=3, step=1)
    search_max_combos = st.number_input("SEARCH_MAX_COMBOS", min_value=50, max_value=5000, value=800, step=50)
    run_param_search = st.button("Run Parameter Search")

start_dt = pd.to_datetime(start_text, errors="coerce")
end_dt = pd.to_datetime(end_text, errors="coerce")
if pd.isna(start_dt) or pd.isna(end_dt):
    st.error("Invalid Start/End datetime format. Use YYYY-MM-DD HH:MM:SS.")
    st.stop()
start_dt = max(min(start_dt.to_pydatetime(), overlap_end), overlap_start)
end_dt = min(max(end_dt.to_pydatetime(), overlap_start), overlap_end)
if end_dt <= start_dt:
    end_dt = start_dt + timedelta(seconds=1)

if jump_to_first_manual:
    manual_dt = None
    if hasattr(data_pkl, "event_data") and isinstance(data_pkl.event_data, pd.DataFrame) and not data_pkl.event_data.empty:
        manual_events = data_pkl.event_data[data_pkl.event_data.get("key", pd.Series(dtype=str)) == "heartbeat_manual_ok"].copy()
        if not manual_events.empty and "datetime" in manual_events.columns:
            ev_dt = pd.to_datetime(manual_events["datetime"], errors="coerce")
            if getattr(ev_dt.dt, "tz", None) is None:
                ev_dt = ev_dt.dt.tz_localize(DEPLOYMENT_TZ)
            else:
                ev_dt = ev_dt.dt.tz_convert(DEPLOYMENT_TZ)
            ev_dt = ev_dt.dropna().sort_values()
            if not ev_dt.empty:
                manual_dt = ev_dt.iloc[0].to_pydatetime()
    if manual_dt is not None:
        manual_dt_display = _to_display_naive(manual_dt, DEPLOYMENT_TZ).to_pydatetime()
        win = max(end_dt - start_dt, timedelta(seconds=1))
        start_dt = max(min(manual_dt_display, overlap_end - win), overlap_start)
        end_dt = min(start_dt + win, overlap_end)
        target_start = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        target_end = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        # Defer widget updates to next rerun to avoid post-instantiation mutation error.
        if start_text != target_start or end_text != target_end:
            st.session_state[pending_start_key] = target_start
            st.session_state[pending_end_key] = target_end
            st.session_state[jump_msg_key] = f"Jumped to first heartbeat_manual_ok at {manual_dt}."
            st.rerun()
    else:
        st.caption("No heartbeat_manual_ok events found to jump to.")

if jump_msg_key in st.session_state:
    st.caption(st.session_state[jump_msg_key])
    del st.session_state[jump_msg_key]

start_ts = _from_display_naive(start_dt, DEPLOYMENT_TZ)
end_ts = _from_display_naive(end_dt, DEPLOYMENT_TZ)

parent_signal_options = list(data_pkl.signal_data.keys())
default_parent = "ecg" if "ecg" in parent_signal_options else parent_signal_options[0]
parent_signal = st.sidebar.selectbox("Parent Signal", parent_signal_options, index=parent_signal_options.index(default_parent))
channels = [c for c in data_pkl.signal_data[parent_signal].columns if c != "datetime"]
if not channels:
    st.error(f"No channels for {parent_signal}.")
    st.stop()
default_channel = "ecg" if "ecg" in channels else channels[0]
channel = st.sidebar.selectbox("Channel", channels, index=channels.index(default_channel))

signal_df = data_pkl.signal_data[parent_signal].copy()
dt = pd.to_datetime(signal_df["datetime"], errors="coerce")
if dt.dt.tz is None:
    dt = dt.dt.tz_localize(DEPLOYMENT_TZ)
else:
    dt = dt.dt.tz_convert(DEPLOYMENT_TZ)
mask = (dt >= start_ts) & (dt <= end_ts)
signal_subset_df = signal_df.loc[mask].reset_index(drop=True)
if signal_subset_df.empty:
    st.error("No data in selected time window.")
    st.stop()

signal = signal_subset_df[channel]
datetime_subset = pd.to_datetime(signal_subset_df["datetime"], errors="coerce")
sampling_rate = data_pkl.signal_info.get(parent_signal, {}).get("sampling_frequency", calculate_sampling_frequency(datetime_subset.head()))
if sampling_rate is None or sampling_rate <= 0:
    st.error("Could not determine sampling rate.")
    st.stop()

params = {
    "BROAD_LOW_CUTOFF": 1.0,
    "BROAD_HIGH_CUTOFF": 35.0,
    "NARROW_LOW_CUTOFF": 5.0,
    "NARROW_HIGH_CUTOFF": 20.0,
    "FILTER_ORDER": 2,
    "SPIKE_THRESHOLD": 400,
    "SMOOTH_SEC_MULTIPLIER": 0.36,
    "WINDOW_SIZE_MULTIPLIER": 6.35,
    "NORMALIZATION_NOISE": 1e-10,
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
}
cfg = param_manager.get_from_config(list(params.keys()), section="hr_peak_detection_settings")
for k, v in cfg.items():
    if v is not None:
        params[k] = v

res = peak_detect(
    signal=signal,
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
)

if "smoothed" not in res or "normalized" not in res or "peak_df" not in res:
    st.error("peak_detect did not return required arrays (smoothed/normalized/peak_df).")
    st.stop()

peak_df = _attach_neighbor_indices(res["peak_df"].copy())
half_window = max(2, int(float(window_sec) * float(sampling_rate) / 2))
feat_df = _build_candidate_features(
    peak_df,
    np.asarray(signal),
    np.asarray(res["smoothed"]),
    np.asarray(res["normalized"]),
    float(sampling_rate),
    half_window,
)
if feat_df.empty:
    st.warning("No candidate peaks in this window.")
    st.stop()

y, label_source = _label_candidates(
    feat_df=feat_df,
    peak_df=peak_df,
    event_df=getattr(data_pkl, "event_data", pd.DataFrame()),
    tolerance_sec=0.2,
    manual_only=manual_only_labels,
)

feature_cols = [
    "height_original",
    "height_normalized",
    "peak_ecg",
    "peak_smoothed",
    "peak_normalized",
    "mean_ecg",
    "std_ecg",
    "mean_smoothed",
    "std_smoothed",
    "mean_norm",
    "std_norm",
    "left_trough_norm",
    "right_trough_norm",
    "prominence_norm",
    "rr_prev_sec",
    "rr_next_sec",
    "c22_ecg_ac1",
    "c22_ecg_zero_cross_rate",
    "c22_ecg_mean_abs_diff",
    "c22_ecg_entropy_bins10",
    "c22_ecg_above_mean_runmax",
    "c22_sm_ac1",
    "c22_sm_zero_cross_rate",
    "c22_sm_mean_abs_diff",
    "c22_sm_entropy_bins10",
    "c22_sm_above_mean_runmax",
    "c22_norm_ac1",
    "c22_norm_zero_cross_rate",
    "c22_norm_mean_abs_diff",
    "c22_norm_entropy_bins10",
    "c22_norm_above_mean_runmax",
]
selected_feature_cols = []
if use_feature_peak:
    selected_feature_cols += ["height_original", "height_normalized", "peak_ecg", "peak_smoothed", "peak_normalized"]
if use_feature_stats:
    selected_feature_cols += ["mean_ecg", "std_ecg", "mean_smoothed", "std_smoothed", "mean_norm", "std_norm"]
if use_feature_prom:
    selected_feature_cols += ["left_trough_norm", "right_trough_norm", "prominence_norm"]
if use_feature_rr:
    selected_feature_cols += ["rr_prev_sec", "rr_next_sec"]
if use_feature_catch22:
    selected_feature_cols += [
        "c22_ecg_ac1",
        "c22_ecg_zero_cross_rate",
        "c22_ecg_mean_abs_diff",
        "c22_ecg_entropy_bins10",
        "c22_ecg_above_mean_runmax",
        "c22_sm_ac1",
        "c22_sm_zero_cross_rate",
        "c22_sm_mean_abs_diff",
        "c22_sm_entropy_bins10",
        "c22_sm_above_mean_runmax",
        "c22_norm_ac1",
        "c22_norm_zero_cross_rate",
        "c22_norm_mean_abs_diff",
        "c22_norm_entropy_bins10",
        "c22_norm_above_mean_runmax",
    ]
selected_feature_cols = [c for c in selected_feature_cols if c in feature_cols]
if not selected_feature_cols:
    selected_feature_cols = feature_cols.copy()

allowed_prefixes = []
if "ecg" in ml_input_sources:
    allowed_prefixes += ["peak_ecg", "mean_ecg", "std_ecg", "height_original", "c22_ecg_"]
if "hr_smoothed" in ml_input_sources:
    allowed_prefixes += ["peak_smoothed", "mean_smoothed", "std_smoothed", "c22_sm_"]
if "hr_normalized" in ml_input_sources:
    allowed_prefixes += ["peak_normalized", "mean_norm", "std_norm", "height_normalized", "left_trough_norm", "right_trough_norm", "prominence_norm", "c22_norm_"]

if allowed_prefixes:
    filtered = []
    for c in selected_feature_cols:
        if c in {"rr_prev_sec", "rr_next_sec"}:
            filtered.append(c)
            continue
        if any(c.startswith(pref) for pref in allowed_prefixes):
            filtered.append(c)
    if filtered:
        selected_feature_cols = filtered

rf_params = {
    "n_estimators": int(rf_n_estimators),
    "min_samples_leaf": int(rf_min_samples_leaf),
    "max_depth": int(rf_max_depth),
    "class_weight": None if rf_class_weight_mode == "none" else rf_class_weight_mode,
}
X_feat = feat_df[selected_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_series = _compose_series_inputs(feat_df, ml_input_sources)

# In manual-only mode, train only on candidates between first and last manual_ok timestamp.
train_mask = np.ones(len(feat_df), dtype=bool)
train_span_start = None
train_span_end = None
if manual_only_labels:
    train_mask, train_span_start, train_span_end = _manual_training_mask(
        feat_df=feat_df,
        event_df=getattr(data_pkl, "event_data", pd.DataFrame()),
    )
    if not train_mask.any():
        st.error("No heartbeat_manual_ok span found for training in current data.")
        st.stop()

X_feat_train = X_feat.loc[train_mask].reset_index(drop=True)
X_series_train = [X_series[i] for i, m in enumerate(train_mask) if m]
y_train = np.asarray(y)[train_mask]
y_train, auto_balanced = _autobalance_single_class_labels(feat_df.loc[train_mask].reset_index(drop=True), y_train)
if auto_balanced:
    label_source = f"{label_source}+auto_balanced"

use_sktime = classifier_mode.startswith("sktime") and _SKTIME_AVAILABLE
model = None
effective_thresh = float(prob_thresh)
suggested_df = pd.DataFrame()
manual_pos = _manual_positive_mask(
    feat_df=feat_df,
    event_df=getattr(data_pkl, "event_data", pd.DataFrame()),
    tolerance_sec=0.2,
)
if classifier_mode == "stickleback_style":
    y_sb = manual_pos.astype(int)
    y_sb_train = y_sb[train_mask]
    X_series_pred = X_series
    X_series_train = [X_series_pred[i] for i, m in enumerate(train_mask) if m]
    model, err, prob, tuned_thresh = _fit_stickleback_style(
        X_series_train=X_series_train,
        y_train=y_sb_train,
        X_series_pred=X_series_pred,
        n_folds=int(sb_folds),
        beta=float(sensitivity_beta),
        neg_pos_ratio=float(sb_neg_pos_ratio),
    )
    if err:
        st.error(err)
        st.stop()
    feat_df["accept_prob"] = prob
    if sb_suggest:
        dense_peak_df = _build_dense_peak_df(
            smoothed=np.asarray(res["smoothed"]),
            normalized=np.asarray(res["normalized"]),
            datetime_series=datetime_subset,
            fs=float(sampling_rate),
            base_peak_distance_sec=float(params.get("PEAK_DISTANCE_SEC", 0.16)),
        )
        if not dense_peak_df.empty:
            dense_peak_df = _attach_neighbor_indices(dense_peak_df)
            dense_feat_df = _build_candidate_features(
                dense_peak_df,
                np.asarray(signal),
                np.asarray(res["smoothed"]),
                np.asarray(res["normalized"]),
                float(sampling_rate),
                half_window,
            )
            if not dense_feat_df.empty:
                if manual_only_labels:
                    dense_train_mask, _, _ = _manual_training_mask(
                        feat_df=dense_feat_df,
                        event_df=getattr(data_pkl, "event_data", pd.DataFrame()),
                    )
                else:
                    dense_train_mask = np.ones(len(dense_feat_df), dtype=bool)
                dense_series = _compose_series_inputs(dense_feat_df, ml_input_sources)
                dense_prob = _predict_series_proba(model, dense_series)
                dense_feat_df["accept_prob"] = dense_prob
                existing_idx = set(pd.to_numeric(feat_df["refined_index"], errors="coerce").dropna().astype(int).tolist())
                is_new = ~pd.to_numeric(dense_feat_df["refined_index"], errors="coerce").astype(int).isin(existing_idx)
                is_suggested = (
                    dense_train_mask
                    & is_new.to_numpy()
                    & (np.asarray(dense_prob) >= float(sb_suggest_thresh))
                )
                suggested_df = dense_feat_df.loc[is_suggested, ["datetime", "refined_index", "accept_prob"]].copy()
    label_source = f"stickleback_style(manual_ok,ratio={float(sb_neg_pos_ratio):.1f},folds={int(sb_folds)})"
elif classifier_mode == "unsupervised_tuned":
    accepted_mask, unsup_meta = _fit_unsupervised_tuned(
        feat_df=feat_df,
        train_mask=train_mask,
        manual_pos_mask=manual_pos,
        selected_inputs=ml_input_sources,
    )
    feat_df["accept_prob"] = accepted_mask.astype(float)
    label_source = f"unsupervised_tuned_f1={unsup_meta['score']:.2f},inputs={unsup_meta['inputs']}"
else:
    model, err, prob = _train_classifier(
        X_feat_train,
        X_series_train,
        np.asarray(y_train),
        use_sktime,
        X_feat_pred=X_feat,
        X_series_pred=X_series,
        rf_params=rf_params,
    )
    if err:
        st.error(err)
        st.stop()

    feat_df["accept_prob"] = prob

prob_ml_df, prob_at_candidates = _build_prob_signal_from_candidates(
    feat_df=feat_df,
    signal_subset_df=signal_subset_df,
    rolling_window_sec=1.0,
)
feat_df["accept_prob_local"] = prob_at_candidates
raw_score_for_threshold = pd.to_numeric(feat_df["accept_prob_local"], errors="coerce").fillna(
    pd.to_numeric(feat_df["accept_prob"], errors="coerce")
).to_numpy(dtype=float)
score_for_threshold = raw_score_for_threshold.copy()
height_gate = pd.to_numeric(feat_df.get("height_original"), errors="coerce").fillna(-np.inf).to_numpy(dtype=float) >= float(min_peak_height_ml)
prom_gate = pd.to_numeric(feat_df.get("prominence_norm"), errors="coerce").fillna(-np.inf).to_numpy(dtype=float) >= float(prominence_norm_min)
gate_ok = height_gate & prom_gate
score_for_threshold = np.where(gate_ok, score_for_threshold, 0.0)
score_for_threshold = _apply_smoothness_penalty(
    feat_df=feat_df,
    base_score=score_for_threshold,
    hr_jump_frac=float(up_jump_frac),
    smoothness_penalty=float(smoothness_penalty),
    min_hr_bpm=float(min_hr_bpm_ml),
    max_hr_bpm=float(max_hr_bpm_ml),
)

y_tune = np.asarray(manual_pos if manual_only_labels else y).astype(int)
if auto_sensitive_thresh:
    tuned = _best_threshold_fbeta(
        y_true=y_tune[train_mask],
        prob=score_for_threshold[train_mask],
        beta=float(sensitivity_beta),
    )
    effective_thresh = max(float(min_user_thresh), min(float(prob_thresh), float(tuned)))
accepted_mask = score_for_threshold >= effective_thresh
accepted_mask = _apply_ml_peak_cleanup(
    feat_df=feat_df,
    accepted_mask=accepted_mask,
    sampling_rate=float(sampling_rate),
    do_conflict_cleanup=bool(do_conflict_cleanup),
    do_up_jump_cleanup=bool(do_up_jump_cleanup),
    conflict_rr_factor=float(conflict_rr_factor),
    hr_jump_frac=float(up_jump_frac),
    pick_last_in_conflict=True,
)
downjump_suggested_df = pd.DataFrame(columns=["datetime", "refined_index", "accept_prob"])
if do_down_jump_insert:
    downjump_suggested_df = _find_down_jump_insertions(
        feat_df=feat_df,
        accepted_mask=accepted_mask,
        raw_signal=np.asarray(signal),
        smoothed=np.asarray(res["smoothed"]),
        signal_subset_df=signal_subset_df,
        fs=float(sampling_rate),
        prob_signal_df=prob_ml_df,
        prob_threshold=float(effective_thresh),
        hr_jump_frac=float(up_jump_frac),
        min_peak_height=float(min_peak_height_ml),
        gap_repair_method=gap_repair_method,
    )
feat_df["ml_key"] = np.where(accepted_mask, "heartbeat_ml_detect_accepted", "heartbeat_ml_detect_rejected")

suggested_merge = pd.concat([suggested_df, downjump_suggested_df], ignore_index=True) if not suggested_df.empty or not downjump_suggested_df.empty else pd.DataFrame(columns=["datetime", "refined_index", "accept_prob"])
extra_indices = (
    pd.to_numeric(suggested_merge["refined_index"], errors="coerce").dropna().astype(int).tolist()
    if not suggested_merge.empty
    else []
)
hr_ml_df, accepted_indices = _build_hr_from_ml(
    feat_df,
    accepted_mask,
    signal_subset_df,
    float(sampling_rate),
    extra_indices=extra_indices,
)
data_pkl.signal_data["heart_rate_ml"] = hr_ml_df
data_pkl.signal_info["heart_rate_ml"] = {
    "channels": ["heart_rate_ml"],
    "metadata": {"heart_rate_ml": {"unit": "bpm", "signal": parent_signal}},
    "derived_from_signals": [parent_signal],
    "transformation_log": [
        "alternative ML heartbeat detector in streamlit page 7",
        f"classifier={'sktime_tsf' if use_sktime else 'sklearn_rf'} label_source={label_source}",
    ],
}
data_pkl.signal_data["heart_rate_ml_prob"] = prob_ml_df
data_pkl.signal_info["heart_rate_ml_prob"] = {
    "channels": ["heart_rate_ml_prob"],
    "metadata": {"heart_rate_ml_prob": {"unit": "probability", "signal": parent_signal, "sampling_frequency": 25.0, "range": [0.0, 1.0]}},
    "derived_from_signals": [parent_signal, "hr_smoothed", "hr_normalized"],
    "transformation_log": [
        "ML heartbeat local probability at 25 Hz across full selected window",
        "rolling mean window = 1.0 sec",
        f"classifier={classifier_mode} label_source={label_source}",
    ],
}
_upsert_ml_events(data_pkl, feat_df, accepted_mask, suggested_df=suggested_merge)

st.markdown("### Classifier Summary")
eval_mask = np.ones(len(feat_df), dtype=bool)
event_df_all = getattr(data_pkl, "event_data", pd.DataFrame())
manual_eval_dt = pd.Series(dtype="datetime64[ns]")
if isinstance(event_df_all, pd.DataFrame) and not event_df_all.empty and "key" in event_df_all.columns and "datetime" in event_df_all.columns:
    manual_ev = event_df_all[event_df_all["key"] == "heartbeat_manual_ok"].copy()
    manual_ev["datetime"] = pd.to_datetime(manual_ev["datetime"], errors="coerce")
    if not manual_ev.empty:
        if getattr(manual_ev["datetime"].dt, "tz", None) is None:
            manual_ev["datetime"] = manual_ev["datetime"].dt.tz_localize(DEPLOYMENT_TZ)
        else:
            manual_ev["datetime"] = manual_ev["datetime"].dt.tz_convert(DEPLOYMENT_TZ)
    manual_ev = manual_ev.dropna(subset=["datetime"])
    # Always evaluate against manual beats inside the active calculation window.
    manual_ev = manual_ev[(manual_ev["datetime"] >= pd.Timestamp(start_ts)) & (manual_ev["datetime"] <= pd.Timestamp(end_ts))]
    if manual_only_labels and train_span_start is not None and train_span_end is not None:
        manual_ev = manual_ev[(manual_ev["datetime"] >= pd.Timestamp(train_span_start)) & (manual_ev["datetime"] <= pd.Timestamp(train_span_end))]
    manual_eval_dt = manual_ev["datetime"]
accepted_eval_dt = pd.to_datetime(feat_df.loc[accepted_mask & eval_mask, "datetime"], errors="coerce")
prec, rec, f1, tp, fp, fn = _event_match_precision_recall(accepted_eval_dt, manual_eval_dt, tolerance_sec=0.2)
# Candidate-level labels kept for PR/ROC score curves.
y_eval = np.asarray(manual_pos).astype(int)[eval_mask]
if manual_only_labels and train_span_start is not None and train_span_end is not None:
    st.caption(f"Training span (manual_ok bounded): {train_span_start} to {train_span_end}")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Precision", f"{prec*100:.1f}%")
m2.metric("Recall", f"{rec*100:.1f}%")
m3.metric("F1", f"{f1*100:.1f}%")
m4.metric("Threshold", f"{effective_thresh*100:.1f}%")
m5.metric("Accepted", int(accepted_mask.sum()))
m6.metric("Candidates", int(len(feat_df)))
st.caption(f"Manual-ok events in selected window: {int(len(manual_eval_dt))}; TP={tp}, FP={fp}, FN={fn}")

annotation_signals = [s for s in ["hr_smoothed", "hr_normalized"] if s in data_pkl.signal_data]
if not annotation_signals:
    annotation_signals = [parent_signal]

def _anno_configs(symbol, color, label):
    out = []
    for i, sig in enumerate(annotation_signals):
        out.append({
            "signal": sig,
            "symbol": symbol,
            "color": color,
            "name": label,
            "legend_key": label,
            "showlegend": i == 0,
        })
    return out

notes_to_plot = {
    "heartbeat_manual_ok": _anno_configs("triangle-down", "blue", "heartbeat_manual_ok"),
    "heartbeat_ml_detect_accepted": _anno_configs("triangle-up", "green", "heartbeat_ml_detect_accepted"),
    "heartbeat_ml_detect_rejected": _anno_configs("x", "red", "heartbeat_ml_detect_rejected"),
    "heartbeat_ml_detect_suggested": _anno_configs("diamond", "orange", "heartbeat_ml_detect_suggested"),
}
signals = [
    s
    for s in [
        parent_signal,
        "depth",
        "hr_smoothed",
        "hr_normalized",
        "heart_rate",
        "heart_rate_fixed",
        "heart_rate_ml",
        "heart_rate_ml_prob",
    ]
    if s in data_pkl.signal_data
]
zoom_channel = "depth" if "depth" in data_pkl.signal_data else parent_signal
fig = plot_tag_data_interactive(
    data_pkl=data_pkl,
    signals=signals,
    time_range=(start_ts, end_ts),
    note_annotations=notes_to_plot,
    color_mapping_path=color_mapping_path,
    target_sampling_rate=25,
    zoom_start_time=start_ts,
    zoom_end_time=end_ts,
    zoom_range_selector_channel=zoom_channel,
    plot_event_values=[],
)
st.plotly_chart(fig, use_container_width=True)

valid_curve = np.isfinite(score_for_threshold[eval_mask]) & np.isfinite(y_eval)
if valid_curve.sum() >= 5 and np.unique(y_eval[valid_curve]).size >= 2:
    pr_precision, pr_recall, _ = precision_recall_curve(y_eval[valid_curve], score_for_threshold[eval_mask][valid_curve])
    roc_fpr, roc_tpr, _ = roc_curve(y_eval[valid_curve], score_for_threshold[eval_mask][valid_curve])
    pr_auc = auc(pr_recall, pr_precision)
    roc_auc = auc(roc_fpr, roc_tpr)

    curve_cols = st.columns(2)
    with curve_cols[0]:
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=pr_recall, y=pr_precision, mode="lines", name=f"PR AUC={pr_auc:.3f}"))
        pr_fig.update_layout(
            title="Precision-Recall (manual-ok bounded span)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(pr_fig, use_container_width=True)
    with curve_cols[1]:
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=roc_fpr, y=roc_tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="chance"))
        roc_fig.update_layout(
            title="ROC (manual-ok bounded span)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(roc_fig, use_container_width=True)

if classifier_mode == "sklearn_rf":
    fi = pd.Series(model.feature_importances_, index=selected_feature_cols).sort_values(ascending=False)
    bar = go.Figure([go.Bar(x=fi.index, y=fi.values)])
    bar.update_layout(title="Feature Importance (RandomForest)", xaxis_tickangle=-30, margin=dict(l=20, r=20, t=40, b=120))
    st.plotly_chart(bar, use_container_width=True)

# Sensitivity histogram: accepted peak count across thresholds.
th_grid = np.linspace(0.0, 1.0, 41)
accepted_counts = np.array([(score_for_threshold >= t).sum() for t in th_grid], dtype=int)
hist = go.Figure([go.Bar(x=th_grid, y=accepted_counts)])
hist.add_vline(x=float(effective_thresh), line_dash="dash", line_color="red")
hist.update_layout(
    title="Threshold Sensitivity: Accepted Peaks vs Probability Threshold",
    xaxis_title="Probability Threshold",
    yaxis_title="Accepted Peak Count",
    margin=dict(l=30, r=20, t=45, b=40),
)
st.plotly_chart(hist, use_container_width=True)

if run_param_search:
    def _grid_around(v, lo, hi, n, span):
        vv = float(v)
        a = max(lo, vv - span)
        b = min(hi, vv + span)
        if n <= 2:
            return np.array([a, b], dtype=float)
        return np.linspace(a, b, int(n))

    thr_grid = np.linspace(0.05, 0.95, max(9, int(search_points) * 4))
    conflict_grid = _grid_around(conflict_rr_factor, 0.1, 1.5, int(search_points), 0.2)
    jump_grid = _grid_around(up_jump_frac, 0.1, 3.0, int(search_points), 0.5)
    smooth_grid = _grid_around(smoothness_penalty, 0.0, 10.0, int(search_points), max(0.5, 0.7 * max(0.5, smoothness_penalty)))
    prom_grid = _grid_around(prominence_norm_min, -10.0, 20.0, int(search_points), max(0.15, 0.5 * max(0.2, abs(prominence_norm_min))))
    minpeak_span = max(10.0, 0.35 * max(10.0, abs(min_peak_height_ml)))
    minpeak_grid = _grid_around(min_peak_height_ml, -100000.0, 100000.0, int(search_points), minpeak_span)

    combos = list(itertools.product(thr_grid, conflict_grid, jump_grid, smooth_grid, prom_grid, minpeak_grid))
    if len(combos) > int(search_max_combos):
        stride = int(np.ceil(len(combos) / float(search_max_combos)))
        combos = combos[::max(1, stride)]

    rows = []
    manual_eval_series = pd.to_datetime(manual_eval_dt, errors="coerce").dropna()
    for thr, cfac, jfac, spal, pmin, hmin in combos:
        score = raw_score_for_threshold.copy()
        h_gate = pd.to_numeric(feat_df.get("height_original"), errors="coerce").fillna(-np.inf).to_numpy(dtype=float) >= float(hmin)
        p_gate = pd.to_numeric(feat_df.get("prominence_norm"), errors="coerce").fillna(-np.inf).to_numpy(dtype=float) >= float(pmin)
        score = np.where(h_gate & p_gate, score, 0.0)
        score = _apply_smoothness_penalty(
            feat_df=feat_df,
            base_score=score,
            hr_jump_frac=float(jfac),
            smoothness_penalty=float(spal),
            min_hr_bpm=float(min_hr_bpm_ml),
            max_hr_bpm=float(max_hr_bpm_ml),
        )
        pred_mask = score >= float(thr)
        pred_mask = _apply_ml_peak_cleanup(
            feat_df=feat_df,
            accepted_mask=pred_mask,
            sampling_rate=float(sampling_rate),
            do_conflict_cleanup=bool(do_conflict_cleanup),
            do_up_jump_cleanup=bool(do_up_jump_cleanup),
            conflict_rr_factor=float(cfac),
            hr_jump_frac=float(jfac),
            pick_last_in_conflict=True,
        )
        accepted_dt_loop = pd.to_datetime(feat_df.loc[pred_mask, "datetime"], errors="coerce")
        p, r, f, tp_i, fp_i, fn_i = _event_match_precision_recall(
            accepted_dt_loop,
            manual_eval_series,
            tolerance_sec=0.2,
        )
        if search_objective == "precision":
            obj = p
        elif search_objective == "recall":
            obj = r
        elif search_objective == "f1":
            obj = f
        else:
            obj = p * r
        rows.append(
            {
                "objective": float(obj),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "threshold": float(thr),
                "ANTI_DOUBLE_GAP_FACTOR": float(cfac),
                "HR_JUMP_FRAC": float(jfac),
                "HR_SMOOTHNESS_PENALTY": float(spal),
                "PROMINENCE_NORM_MIN": float(pmin),
                "MIN_PEAK_HEIGHT": float(hmin),
                "accepted": int(pred_mask.sum()),
                "tp": int(tp_i),
                "fp": int(fp_i),
                "fn": int(fn_i),
            }
        )

    search_df = pd.DataFrame(rows).sort_values(["objective", "f1", "precision", "recall"], ascending=False).reset_index(drop=True)
    st.markdown("### Parameter Search Results")
    st.caption(f"Evaluated {len(search_df)} parameter combinations (objective={search_objective}).")
    if not search_df.empty:
        st.dataframe(search_df.head(20), use_container_width=True)
        pr_scatter = go.Figure(
            data=[
                go.Scatter(
                    x=search_df["recall"],
                    y=search_df["precision"],
                    mode="markers",
                    marker=dict(
                        size=np.clip(6 + np.sqrt(np.maximum(search_df["objective"], 0)) * 12, 6, 20),
                        color=search_df["objective"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="objective"),
                    ),
                    text=[
                        f"thr={t:.2f}, f1={f:.2f}"
                        for t, f in zip(search_df["threshold"], search_df["f1"])
                    ],
                    hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<br>%{text}<extra></extra>",
                )
            ]
        )
        pr_scatter.update_layout(
            title="Precision vs Recall Across Parameter Combinations",
            xaxis_title="Recall",
            yaxis_title="Precision",
            margin=dict(l=30, r=20, t=45, b=35),
        )
        st.plotly_chart(pr_scatter, use_container_width=True)

if st.sidebar.button("Save ML Outputs to Config"):
    param_manager.add_to_config(
        entries={
            "HR_ML_CLASSIFIER": classifier_mode,
            "HR_ML_ACCEPT_PROB_THRESHOLD": float(prob_thresh),
            "HR_ML_FEATURE_WINDOW_SEC": float(window_sec),
            "HR_ML_INPUT_SIGNALS": list(ml_input_sources),
            "RF_N_ESTIMATORS": int(rf_n_estimators),
            "RF_MIN_SAMPLES_LEAF": int(rf_min_samples_leaf),
            "RF_MAX_DEPTH": int(rf_max_depth),
            "RF_CLASS_WEIGHT": rf_class_weight_mode,
            "ANTI_DOUBLE_GAP_FACTOR": float(conflict_rr_factor),
            "HR_JUMP_FRAC": float(up_jump_frac),
            "HR_SMOOTHNESS_PENALTY": float(smoothness_penalty),
            "PROMINENCE_NORM_MIN": float(prominence_norm_min),
            "MIN_PEAK_HEIGHT": float(min_peak_height_ml),
            "MIN_HR_BPM": float(min_hr_bpm_ml),
            "MAX_HR_BPM": float(max_hr_bpm_ml),
            "GAP_REPAIR_METHOD": gap_repair_method,
        },
        section="hr_peak_detection_settings",
    )
    st.sidebar.success("Saved ML detector settings to hr_peak_detection_settings.")
