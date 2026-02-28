import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import resample_poly

from pyologger.utils.folder_manager import load_configuration, select_and_load_deployment_streamlit
from pyologger.utils.streamlit_time_window import standardize_time_settings

try:
    from sleepecg import detect_heartbeats as _sleepecg_detect_heartbeats
    _SLEEPECG_AVAILABLE = True
except Exception:
    _sleepecg_detect_heartbeats = None
    _SLEEPECG_AVAILABLE = False

try:
    import wfdb.processing as _wfdb_processing
    _WFDB_AVAILABLE = True
except Exception:
    _wfdb_processing = None
    _WFDB_AVAILABLE = False


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


def _time_mask(dt_series, start_ts, end_ts, tz_name):
    dt = pd.to_datetime(dt_series, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(tz_name)
    else:
        dt = dt.dt.tz_convert(tz_name)
    start = _to_tzaware(start_ts, tz_name)
    end = _to_tzaware(end_ts, tz_name)
    return (dt >= start) & (dt <= end), dt


def _estimate_fs(dt):
    t = pd.Series(dt).dropna()
    if len(t) < 2:
        return np.nan
    sec = t.view("int64").to_numpy() / 1e9
    diff = np.diff(sec)
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if diff.size == 0:
        return np.nan
    return float(1.0 / np.median(diff))


def _run_sleepecg(sig, fs, backend):
    if not _SLEEPECG_AVAILABLE:
        raise RuntimeError("sleepecg is not installed in this environment.")
    sig_arr = np.asarray(sig, dtype=float)
    fs_in = float(fs)
    min_safe_fs = 61.0
    target_fs = 128.0

    if fs_in < min_safe_fs:
        up = int(round(target_fs))
        down = max(1, int(round(fs_in)))
        sig_rs = resample_poly(sig_arr, up=up, down=down)
        fs_rs = fs_in * (up / down)
        idx_rs = np.asarray(_sleepecg_detect_heartbeats(sig_rs, fs=fs_rs, backend=backend), dtype=int)
        idx = np.rint(idx_rs * (fs_in / fs_rs)).astype(int)
        return idx, fs_rs, True

    idx = np.asarray(_sleepecg_detect_heartbeats(sig_arr, fs=fs_in, backend=backend), dtype=int)
    return idx, fs_in, False


def _run_wfdb_xqrs(sig, fs, conf_kwargs, learn, verbose):
    if not _WFDB_AVAILABLE:
        raise RuntimeError("wfdb is not installed in this environment.")
    conf = _wfdb_processing.XQRS.Conf(**conf_kwargs)
    return np.asarray(
        _wfdb_processing.xqrs_detect(
            sig=np.asarray(sig, dtype=float),
            fs=float(fs),
            sampfrom=0,
            sampto="end",
            conf=conf,
            learn=bool(learn),
            verbose=bool(verbose),
        ),
        dtype=int,
    )


config, data_dir, _, _ = load_configuration()

st.title("ECG Library Peak Detection")
st.caption("Stripped-down detector page for SleepECG and WFDB XQRS.")

st.sidebar.title("Deployment")
animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = select_and_load_deployment_streamlit(data_dir)
timezone = data_pkl.deployment_info.get("Time Zone", "UTC")
DISPLAY_TZ = str(timezone or "UTC")
st.sidebar.write(f"Selected deployment: {deployment_id}")

ALT_SECTION = "alt_peak_detect_settings"
DEFAULT_KEYS = [
    "method",
    "signal_name",
    "channel_name",
    "sampling_rate_hz",
    "xqrs_hr_init",
    "xqrs_hr_max",
    "xqrs_hr_min",
    "xqrs_qrs_width",
    "xqrs_qrs_thr_init",
    "xqrs_qrs_thr_min",
    "xqrs_ref_period",
    "xqrs_t_inspect_period",
    "xqrs_learn",
    "xqrs_verbose",
    "sleepecg_backend",
]
saved_defaults = param_manager.get_from_config(DEFAULT_KEYS, section=ALT_SECTION)
saved_settings = param_manager.get_from_config(["use_alt_peak_detect_settings"], section="settings")

std = standardize_time_settings(
    param_manager=param_manager,
    data_pkl=data_pkl,
    tz_name=DISPLAY_TZ,
    minutes=10,
    persist=False,
)
default_start = _to_display_naive(std["zoom_window_start_time"], DISPLAY_TZ).to_pydatetime()
default_end = _to_display_naive(std["zoom_window_end_time"], DISPLAY_TZ).to_pydatetime()

st.sidebar.title("Window")
start_input = st.sidebar.text_input("Start (YYYY-MM-DD HH:MM:SS)", value=default_start.strftime("%Y-%m-%d %H:%M:%S"))
end_input = st.sidebar.text_input("End (YYYY-MM-DD HH:MM:SS)", value=default_end.strftime("%Y-%m-%d %H:%M:%S"))

try:
    start_dt = _from_display_naive(pd.Timestamp(start_input).to_pydatetime(), DISPLAY_TZ)
    end_dt = _from_display_naive(pd.Timestamp(end_input).to_pydatetime(), DISPLAY_TZ)
except Exception:
    st.error("Invalid start/end datetime format.")
    st.stop()

if end_dt <= start_dt:
    st.error("End time must be after start time.")
    st.stop()

st.sidebar.title("Signal")
signals = sorted(list(data_pkl.signal_data.keys()))
if not signals:
    st.error("No signals found.")
    st.stop()
default_signal_idx = signals.index("ecg") if "ecg" in signals else 0
saved_signal_name = saved_defaults.get("signal_name")
if saved_signal_name in signals:
    default_signal_idx = signals.index(saved_signal_name)
signal_name = st.sidebar.selectbox("Signal", options=signals, index=default_signal_idx)

signal_info = data_pkl.signal_info.get(signal_name, {})
channels = list(signal_info.get("channels", []))
if not channels:
    channels = [c for c in data_pkl.signal_data[signal_name].columns if c != "datetime"]
if not channels:
    st.error("Selected signal has no channels.")
    st.stop()
saved_channel_name = saved_defaults.get("channel_name")
if saved_channel_name in channels:
    channel_name = st.sidebar.selectbox("Channel", options=channels, index=channels.index(saved_channel_name))
else:
    channel_name = st.sidebar.selectbox("Channel", options=channels, index=0)

raw_df = data_pkl.signal_data[signal_name].copy()
if "datetime" not in raw_df.columns or channel_name not in raw_df.columns:
    st.error("Selected signal/channel columns are missing.")
    st.stop()

mask, dt_aligned = _time_mask(raw_df["datetime"], start_dt, end_dt, DISPLAY_TZ)
df = raw_df.loc[mask, ["datetime", channel_name]].copy()
df["datetime"] = dt_aligned.loc[mask]
df = df.dropna().sort_values("datetime").reset_index(drop=True)

if df.empty:
    st.warning("No data in selected window.")
    st.stop()

fs_est = _estimate_fs(df["datetime"])
if not np.isfinite(fs_est) or fs_est <= 0:
    st.error("Could not estimate sampling rate from selected data.")
    st.stop()

st.sidebar.title("Method")
method_options = ["wfdb_xqrs", "sleepecg"]
saved_method = saved_defaults.get("method")
default_method_idx = method_options.index(saved_method) if saved_method in method_options else 0
method = st.sidebar.radio("Detector", method_options, index=default_method_idx)
saved_fs_override = saved_defaults.get("sampling_rate_hz")
if saved_fs_override is None:
    saved_fs_override = float(fs_est)
fs_override = st.sidebar.number_input(
    "Sampling rate (Hz)",
    min_value=0.1,
    value=float(saved_fs_override),
    step=1.0,
    help="Sampling rate passed into the detector function. Use the true sample rate for best results.",
)
use_alt_peak_detect = st.sidebar.checkbox(
    "Use alt peak settings in workflow",
    value=bool(saved_settings.get("use_alt_peak_detect_settings")),
    help="When enabled, workflow 05 heartbeat detection will use alt_peak_detect_settings (WFDB XQRS) instead of legacy peak_detect.",
)

xqrs_conf = {}
learn = True
verbose = False
sleepecg_backend = str(saved_defaults.get("sleepecg_backend") or "c")
if method == "wfdb_xqrs":
    st.sidebar.subheader("XQRS.Conf")
    xqrs_conf["hr_init"] = float(
        st.sidebar.number_input(
            "hr_init",
            value=float(saved_defaults.get("xqrs_hr_init") or 75.0),
            step=1.0,
            help="Initial HR guess (bpm) used to initialize RR tracking at detector startup.",
        )
    )
    xqrs_conf["hr_max"] = float(
        st.sidebar.number_input(
            "hr_max",
            value=float(saved_defaults.get("xqrs_hr_max") or 200.0),
            step=1.0,
            help="Hard upper HR bound (bpm). Prevents accepting beats that are unrealistically close together.",
        )
    )
    xqrs_conf["hr_min"] = float(
        st.sidebar.number_input(
            "hr_min",
            value=float(saved_defaults.get("xqrs_hr_min") or 25.0),
            step=1.0,
            help="Hard lower HR bound (bpm). Used for expected RR tracking and search-back behavior.",
        )
    )
    xqrs_conf["qrs_width"] = float(
        st.sidebar.number_input(
            "qrs_width (sec)",
            value=float(saved_defaults.get("xqrs_qrs_width") or 0.1),
            step=0.01,
            format="%.3f",
            help="Expected QRS width in seconds; used by internal filters and scoring windows.",
        )
    )
    xqrs_conf["qrs_thr_init"] = float(
        st.sidebar.number_input(
            "qrs_thr_init (mV)",
            value=float(saved_defaults.get("xqrs_qrs_thr_init") or 0.13),
            step=0.01,
            format="%.3f",
            help="Initial detection threshold. Higher values are stricter; lower values are more sensitive.",
        )
    )
    xqrs_conf["qrs_thr_min"] = float(
        st.sidebar.number_input(
            "qrs_thr_min",
            value=float(saved_defaults.get("xqrs_qrs_thr_min") or 0.0),
            step=0.01,
            format="%.3f",
            help="Lower bound for adaptive threshold. Set >0 to stop threshold from dropping too low in noisy regions.",
        )
    )
    xqrs_conf["ref_period"] = float(
        st.sidebar.number_input(
            "ref_period (sec)",
            value=float(saved_defaults.get("xqrs_ref_period") or 0.2),
            step=0.01,
            format="%.3f",
            help="Refractory period after each detection. Blocks immediate duplicate detections.",
        )
    )
    xqrs_conf["t_inspect_period"] = float(
        st.sidebar.number_input(
            "t_inspect_period (sec)",
            value=float(saved_defaults.get("xqrs_t_inspect_period") or 0.0),
            step=0.01,
            format="%.3f",
            help="If >0, inspects early candidate peaks to reject likely T-waves.",
        )
    )
    learn = st.sidebar.checkbox(
        "learn",
        value=bool(saved_defaults.get("xqrs_learn") if saved_defaults.get("xqrs_learn") is not None else True),
        help="Enable learning phase to adapt QRS/noise thresholds from the signal.",
    )
    verbose = st.sidebar.checkbox(
        "verbose",
        value=bool(saved_defaults.get("xqrs_verbose") if saved_defaults.get("xqrs_verbose") is not None else False),
        help="Print WFDB XQRS detector logs.",
    )
else:
    st.sidebar.subheader("SleepECG")
    sleepecg_backend = st.sidebar.selectbox(
        "backend",
        options=["c", "numba", "python"],
        index=(["c", "numba", "python"].index(sleepecg_backend) if sleepecg_backend in {"c", "numba", "python"} else 0),
        help="SleepECG implementation backend from docs: c=fastest, numba=fast fallback, python=slow fallback.",
    )

sig = df[channel_name].to_numpy(dtype=float)
fs = float(fs_override)

detected_idx = np.array([], dtype=int)
err = None
method_note = ""
try:
    if method == "sleepecg":
        detected_idx, fs_used, was_resampled = _run_sleepecg(sig, fs, sleepecg_backend)
        if was_resampled:
            method_note = f"SleepECG ran on internally resampled signal at {fs_used:.2f} Hz (mapped back to original samples)."
    else:
        detected_idx = _run_wfdb_xqrs(sig, fs, xqrs_conf, learn, verbose)
except Exception as e:
    err = str(e)

if method == "sleepecg" and not _SLEEPECG_AVAILABLE:
    st.warning("sleepecg is not installed. Install it to use this method: `pip install sleepecg`")
if method == "wfdb_xqrs" and not _WFDB_AVAILABLE:
    st.warning("wfdb is not installed. Install it to use this method: `pip install wfdb`")
if err:
    st.error(f"Detection failed: {err}")

detected_idx = detected_idx[(detected_idx >= 0) & (detected_idx < len(df))]
detected_idx = np.unique(detected_idx)
detected_times = df["datetime"].iloc[detected_idx] if len(detected_idx) else pd.Series(dtype="datetime64[ns]")
detected_vals = df[channel_name].iloc[detected_idx] if len(detected_idx) else pd.Series(dtype=float)

manual = pd.DataFrame()
if hasattr(data_pkl, "event_data") and isinstance(data_pkl.event_data, pd.DataFrame):
    event_df = data_pkl.event_data.copy()
    if "datetime" in event_df.columns and "key" in event_df.columns:
        emask, edt = _time_mask(event_df["datetime"], start_dt, end_dt, DISPLAY_TZ)
        manual = event_df.loc[emask & (event_df["key"] == "heartbeat_manual_ok"), ["datetime", "key"]].copy()
        manual["datetime"] = edt.loc[manual.index]

st.markdown(f"Estimated fs: `{fs_est:.2f} Hz` | Using fs: `{fs:.2f} Hz` | Detected peaks: `{len(detected_idx)}`")
if method_note:
    st.info(method_note)

if st.sidebar.button("Save as dataset defaults"):
    payload = {
        "method": method,
        "signal_name": signal_name,
        "channel_name": channel_name,
        "sampling_rate_hz": float(fs),
        "xqrs_hr_init": float(xqrs_conf.get("hr_init", 75.0)),
        "xqrs_hr_max": float(xqrs_conf.get("hr_max", 200.0)),
        "xqrs_hr_min": float(xqrs_conf.get("hr_min", 25.0)),
        "xqrs_qrs_width": float(xqrs_conf.get("qrs_width", 0.1)),
        "xqrs_qrs_thr_init": float(xqrs_conf.get("qrs_thr_init", 0.13)),
        "xqrs_qrs_thr_min": float(xqrs_conf.get("qrs_thr_min", 0.0)),
        "xqrs_ref_period": float(xqrs_conf.get("ref_period", 0.2)),
        "xqrs_t_inspect_period": float(xqrs_conf.get("t_inspect_period", 0.0)),
        "xqrs_learn": bool(learn),
        "xqrs_verbose": bool(verbose),
        "sleepecg_backend": str(sleepecg_backend),
    }
    param_manager.set_dataset_defaults(entries=payload, section=ALT_SECTION)
    param_manager.set_dataset_defaults(
        entries={"use_alt_peak_detect_settings": bool(use_alt_peak_detect)},
        section="settings",
    )
    st.sidebar.success("Saved dataset defaults to alt_peak_detect_settings.")

fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=df["datetime"],
        y=df[channel_name],
        mode="lines",
        name=f"{signal_name}.{channel_name}",
        line=dict(color="#C96E67", width=1),
    )
)
if len(detected_idx):
    fig.add_trace(
        go.Scattergl(
            x=detected_times,
            y=detected_vals,
            mode="markers",
            name=f"{method} peaks",
            marker=dict(symbol="triangle-up", size=8, color="#2E8B57"),
        )
    )
if not manual.empty:
    y_manual = np.interp(
        manual["datetime"].view("int64").to_numpy(),
        df["datetime"].view("int64").to_numpy(),
        df[channel_name].to_numpy(),
    )
    fig.add_trace(
        go.Scattergl(
            x=manual["datetime"],
            y=y_manual,
            mode="markers",
            name="heartbeat_manual_ok",
            marker=dict(symbol="triangle-down", size=8, color="#2F6CC0"),
        )
    )
fig.update_layout(height=540, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Time", yaxis_title=channel_name)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Detected peak timestamps"):
    if len(detected_idx):
        out = pd.DataFrame(
            {
                "datetime": pd.to_datetime(detected_times).astype(str).to_list(),
                "sample_index_in_window": detected_idx.astype(int).tolist(),
                "value": detected_vals.astype(float).tolist(),
            }
        )
        st.dataframe(out, use_container_width=True, height=240)
    else:
        st.write("No peaks detected.")
