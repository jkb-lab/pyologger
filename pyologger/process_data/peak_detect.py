import numpy as np
import pandas as pd
import ast
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal.windows import bartlett
from scipy import stats
from wfdb import processing

QUADRUPED_SPECIES_CODES = { # For these species codes, detected stroke_rate will be converted to stride_rate by dividing by 2.
    "pale",  # Panthera leo
}


def _get_species_code_from_animal_id(animal_id):
    """Extract the species code prefix from an animal ID like 'pale-001'."""
    if animal_id is None:
        return ""
    animal_id_text = str(animal_id).strip().lower()
    if not animal_id_text:
        return ""
    return animal_id_text.split("-", 1)[0]


def _should_convert_stroke_to_stride_rate(data_pkl, mode):
    """Return True when stroke detections should be reported as stride rate."""
    if mode != "stroke_rate":
        return False
    animal_info = getattr(data_pkl, "animal_info", {}) or {}
    species_code = _get_species_code_from_animal_id(animal_info.get("Animal_ID"))
    return species_code in QUADRUPED_SPECIES_CODES

# Configuration Section
BROAD_LOW_CUTOFF = 1  # Hz for bandpass filter
BROAD_HIGH_CUTOFF = 35  # Hz for bandpass filter
NARROW_LOW_CUTOFF = 5  # Hz for bandpass filter
NARROW_HIGH_CUTOFF = 20  # Hz for bandpass filter
FILTER_ORDER = 2  # Order of the bandpass filter
SPIKE_THRESHOLD = 400  # Threshold for spike removal
SMOOTH_SEC_MULTIPLIER = 3  # Multiplier for smoothing window size
WINDOW_SIZE_MULTIPLIER = 5  # Multiplier for sliding window normalization
NORMALIZATION_NOISE = 1e-10  # Noise level for sliding window normalization
PEAK_HEIGHT = -0.4  # Minimum peak height for detection
PEAK_DISTANCE_SEC = 0.2  # Minimum distance between peaks in seconds
SEARCH_RADIUS_SEC = 0.5  # Search radius for peak refinement in seconds
MIN_PEAK_HEIGHT = 500  # Minimum acceptable peak height
MAX_PEAK_HEIGHT = 12000  # Maximum acceptable peak height

# Bandpass Filter Function
def bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Spike Removal Function
def remove_spikes(signal, threshold=400):
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    return np.where(np.abs(signal - median) > threshold * mad, median, signal)

# Signal Smoothing Function
def smooth_signal(signal, smooth_sec, fs):
    window = int(smooth_sec * fs)
    bartlett_window = bartlett(window)
    bartlett_window /= bartlett_window.sum()  # Normalize the window
    return np.convolve(signal, bartlett_window, mode='same')

# Sliding Window Normalization Function
def sliding_window_normalization(signal, window_size, noise=1e-10):
    """
    Local z-normalization with a configurable std floor.

    Backward-compatible behavior:
    - Very small values (<= 1e-6) are treated as legacy absolute epsilon.
    - Typical Streamlit tuning values (> 1e-6 and < 1) are treated as a fraction of global std.
    - Values >= 1 are treated as an absolute std floor.
    """
    half_window = max(1, int(window_size) // 2)
    sig = np.asarray(signal, dtype=float)
    global_std = float(np.nanstd(sig)) if sig.size else 0.0
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = 1.0

    if noise <= 1e-6:
        std_floor = float(noise)
    elif noise < 1.0:
        std_floor = float(noise) * global_std
    else:
        std_floor = float(noise)
    std_floor = max(std_floor, 1e-12)

    n = len(sig)
    if n == 0:
        return np.empty(0, dtype=float)

    # Rolling mean/std over the half-open window sig[max(0, i-half) : min(n, i+half)]
    # via prefix sums, rather than one np.mean/np.std call per sample. Same result
    # (differences at float round-off), ~50x faster on multi-million-sample ECG.
    csum = np.concatenate(([0.0], np.cumsum(sig)))
    csum_sq = np.concatenate(([0.0], np.cumsum(sig * sig)))

    idx = np.arange(n)
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(n, idx + half_window)
    count = (hi - lo).astype(float)

    window_sum = csum[hi] - csum[lo]
    window_sum_sq = csum_sq[hi] - csum_sq[lo]
    mu = window_sum / count
    # Clamp to zero: catastrophic cancellation can make this marginally negative.
    variance = np.maximum(window_sum_sq / count - mu * mu, 0.0)
    sigma = np.sqrt(variance)

    return (sig - mu) / np.maximum(sigma, std_floor)

# Peak Refinement with WFDB
def refine_peaks_with_wfdb(cleaned_signal, rpeaks, fs, search_radius=0.5, sample_rate=1000, peak_dir="compare"):
    return processing.correct_peaks(cleaned_signal, rpeaks, smooth_window_size = int(0.5 * fs), search_radius=int(search_radius * sample_rate), peak_dir=peak_dir)

# Absolute Maxima Search Function
def absolute_maxima_search(refined_peaks, original_signal, QRS_width, sample_rate):
    QRS_samples = int(QRS_width * sample_rate)
    return [np.argmax(original_signal[max(0, peak - QRS_samples):peak + 1]) + max(0, peak - QRS_samples) for peak in refined_peaks]


# Combined Peak Detection Function
def peak_detect(signal, sampling_rate, datetime_series=None,
                broad_lowcut=BROAD_LOW_CUTOFF, broad_highcut=BROAD_HIGH_CUTOFF,
                narrow_lowcut=NARROW_LOW_CUTOFF, narrow_highcut=NARROW_HIGH_CUTOFF, 
                filter_order=FILTER_ORDER, spike_threshold=SPIKE_THRESHOLD, 
                smooth_sec_multiplier=SMOOTH_SEC_MULTIPLIER, window_size_multiplier=WINDOW_SIZE_MULTIPLIER, 
                normalization_noise=NORMALIZATION_NOISE, peak_height=PEAK_HEIGHT, 
                peak_distance_sec=PEAK_DISTANCE_SEC, search_radius_sec=SEARCH_RADIUS_SEC,
                min_peak_height=MIN_PEAK_HEIGHT, max_peak_height=MAX_PEAK_HEIGHT,
                enable_bandpass=True, enable_spike_removal=True, enable_absolute=True, enable_smoothing=True, 
                enable_normalization=True, enable_refinement=True, detection_sources=None,
                detector_method="legacy", xqrs_conf=None, xqrs_learn=True, xqrs_verbose=False):
    
    results = {}

    # Bandpass filter
    if enable_bandpass:
        broad_bandpassed_signal = bandpass_filter(signal, lowcut=broad_lowcut, highcut=broad_highcut, fs=sampling_rate, order=filter_order)
        results['broad_bandpass'] = broad_bandpassed_signal
        narrow_bandpassed_signal = bandpass_filter(signal, lowcut=narrow_lowcut, highcut=narrow_highcut, fs=sampling_rate, order=filter_order)
        results['narrow_bandpass'] = narrow_bandpassed_signal
    else:
        narrow_bandpassed_signal = signal

    # Spike removal
    if enable_spike_removal:
        spike_removed_signal = remove_spikes(narrow_bandpassed_signal, threshold=spike_threshold)
        results['spikeless'] = spike_removed_signal
    else:
        spike_removed_signal = narrow_bandpassed_signal

    # Smoothing
    if enable_smoothing:
        if enable_absolute:
            signal_to_smooth = np.abs(spike_removed_signal)
        else:
            signal_to_smooth = spike_removed_signal
        smoothed_signal = smooth_signal(signal_to_smooth, smooth_sec= smooth_sec_multiplier, fs=sampling_rate)
        results['smoothed'] = smoothed_signal
    else:
        smoothed_signal = spike_removed_signal

    # Normalization
    if enable_normalization:
        normalized_signal = sliding_window_normalization(smoothed_signal, int(window_size_multiplier * sampling_rate), noise=normalization_noise)
        results['normalized'] = normalized_signal
    else:
        normalized_signal = smoothed_signal

    # Peak Detection
    method = str(detector_method or "legacy").strip().lower()
    run_legacy_detector = True
    if method == "wfdb_xqrs":
        xqrs_conf = xqrs_conf or {}
        try:
            conf = processing.XQRS.Conf(**xqrs_conf)
            detected_peaks = processing.xqrs_detect(
                sig=np.asarray(signal, dtype=float),
                fs=float(sampling_rate),
                sampfrom=0,
                sampto='end',
                conf=conf,
                learn=bool(xqrs_learn),
                verbose=bool(xqrs_verbose),
            )
            detected_peaks = np.asarray(detected_peaks, dtype=int)
            detected_by_source = {"wfdb_xqrs": detected_peaks}
            run_legacy_detector = False
        except Exception:
            # Safe fallback: if XQRS fails, keep behavior deterministic by using legacy normalized peaks.
            method = "legacy"
    if run_legacy_detector:
        selected_sources = detection_sources
        if selected_sources is None:
            selected_sources = ["normalized"]
        elif isinstance(selected_sources, str):
            text = selected_sources.strip()
            parsed = None
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    parsed = None
            if isinstance(parsed, (list, tuple, set)):
                selected_sources = [str(v).strip() for v in parsed if str(v).strip()]
            else:
                selected_sources = [v.strip() for v in text.split(",") if v.strip()]
        else:
            selected_sources = [str(v).strip() for v in selected_sources if str(v).strip()]
        if not selected_sources:
            selected_sources = ["normalized"]

        source_signals = {
            "raw": np.asarray(signal),
            "spikeless": np.asarray(spike_removed_signal),
            "smoothed": np.asarray(smoothed_signal),
            "normalized": np.asarray(normalized_signal),
            "narrow_bandpass": np.asarray(narrow_bandpassed_signal),
        }
        if enable_bandpass and "broad_bandpass" in results:
            source_signals["broad_bandpass"] = np.asarray(results["broad_bandpass"])

        distance_samples = max(1, int(peak_distance_sec * sampling_rate))
        detected_by_source = {}
        detected_pool = []
        for source_name in selected_sources:
            source_key = str(source_name).strip().lower()
            if source_key not in source_signals:
                continue

            if source_key == "normalized":
                detection_signal = normalized_signal
            else:
                detection_signal = sliding_window_normalization(
                    source_signals[source_key],
                    int(window_size_multiplier * sampling_rate),
                    noise=normalization_noise,
                )
                results[f"detection_{source_key}"] = detection_signal

            source_peaks = find_peaks(
                detection_signal,
                height=peak_height,
                distance=distance_samples,
            )[0]
            detected_by_source[source_key] = source_peaks
            detected_pool.append(source_peaks)

        if detected_pool:
            detected_peaks = np.unique(np.concatenate(detected_pool))
        else:
            # Safe fallback when user-configured sources are unavailable.
            detected_peaks = find_peaks(
                normalized_signal,
                height=peak_height,
                distance=distance_samples,
            )[0]
            detected_by_source = {"normalized": detected_peaks}

    results['detected_peaks_by_source'] = detected_by_source
    results['detected_peaks'] = detected_peaks
    results['detector_method'] = method

    # Peak Refinement
    if enable_refinement:
        refined_peaks = refine_peaks_with_wfdb(smoothed_signal, detected_peaks, fs=sampling_rate, search_radius=search_radius_sec, sample_rate=sampling_rate, peak_dir="both")
        results['refined_peaks'] = refined_peaks
        refined_indices = refined_peaks
    else:
        refined_indices = detected_peaks

    # Remove duplicate refined_indices while keeping corresponding heights aligned
    unique_refined_indices, index_positions = np.unique(refined_indices, return_index=True)
    results['unique_refined_indices'] = unique_refined_indices

    # Align heights with unique refined indices
    height_original = smoothed_signal[unique_refined_indices]
    height_normalized = normalized_signal[refined_indices[index_positions]]
    results['height_original'] = height_original
    results['height_normalized'] = height_normalized

    # Filter out peaks that are too close to each other.
    # Keep the latest peak in a close pair by default (replace prior).
    min_distance_samples = int(peak_distance_sec * sampling_rate)
    filtered_indices = [unique_refined_indices[0]] if len(unique_refined_indices) > 0 else []
    for i in range(1, len(unique_refined_indices)):
        if unique_refined_indices[i] - filtered_indices[-1] >= min_distance_samples:
            filtered_indices.append(unique_refined_indices[i])
        else:
            filtered_indices[-1] = unique_refined_indices[i]

    # Update the DataFrame with the filtered indices
    peak_df = pd.DataFrame({
        'refined_index': filtered_indices,
        'height_original': smoothed_signal[filtered_indices],
        'height_normalized': normalized_signal[filtered_indices]
    })

    # Add datetime column if datetime_series is provided
    if datetime_series is not None:
        if len(datetime_series) != len(signal):
            raise ValueError("Length of datetime_series must match the length of the signal.")
        datetimes = pd.to_datetime(datetime_series.iloc[filtered_indices])
        datetimes_dropped = datetimes.reset_index(drop=True)
        peak_df['datetime'] = datetimes_dropped

    # Filter out peaks based on both minimum and maximum peak height
    filtered_peak_df = peak_df[(peak_df['height_original'] >= min_peak_height) & 
                               (peak_df['height_original'] <= max_peak_height)].reset_index(drop=True)
    results['filtered_peak_df'] = filtered_peak_df

    # Label peaks as accepted or rejected
    peak_df['key'] = np.where(peak_df['refined_index'].isin(filtered_peak_df['refined_index']), 
                              'beat_auto_detect_accepted', 
                              'beat_auto_detect_rejected')
    peak_df['key'] = peak_df['key'].fillna('beat_auto_detect_unknown')  # Ensure no NaN values in 'key'
    results['peak_df'] = peak_df

    return results


def upsert_rate_signal(data_pkl, rate_df, rate_key, parent_signal, transformation_log=None):
    """
    Write/update a derived rate-like signal and its signal_info metadata.
    """
    if transformation_log is None:
        transformation_log = [f"{rate_key} updated."]

    signal_info = {
        "channels": [rate_key],
        "metadata": {
            rate_key: {
                "original_name": f"Derived {rate_key.capitalize()} (bpm)",
                "unit": "bpm",
                "parent_signal": parent_signal,
            }
        },
        "derived_from_signals": [parent_signal],
        "transformation_log": transformation_log,
    }
    data_pkl.signal_data[rate_key] = rate_df
    data_pkl.signal_info[rate_key] = signal_info


def process_rate(
    data_pkl, results, signal_subset_df, parent_signal, params, sampling_rate, mode
):
    """
    Process heart rate or stroke rate, and save intermediate results, metadata, and events.

    Args:
        data_pkl: Data structure to store derived data and metadata.
        results: Results from the peak detection function.
        signal_subset_df: The dataframe used for peak detection.
        parent_signal: The name of the signal used for peak detection.
        params: Dictionary of parameters used in peak detection.
        sampling_rate: Sampling rate of the signal.
        mode: "heart_rate" or "stroke_rate".
        timezone: Timezone information for timestamps.

    Returns:
        None
    """
    # Define keys based on mode
    rate_key = "heart_rate" if mode == "heart_rate" else "stroke_rate"
    keyword = "beat_auto_detect" 
    description = (
        "calculated heart rate from detected peaks"
        if mode == "heart_rate"
        else "calculated stroke rate from detected peaks"
    )
    convert_stroke_to_stride_rate = _should_convert_stroke_to_stride_rate(data_pkl, mode)
    if convert_stroke_to_stride_rate:
        animal_info = getattr(data_pkl, "animal_info", {}) or {}
        species_code = _get_species_code_from_animal_id(animal_info.get("Animal_ID"))
        description = "calculated stride rate from detected peaks"
        print(
            f"[peak_detect] Converting stroke_rate to stride_rate for quadruped species code '{species_code}'."
        )

    peak_df = results["peak_df"]

    # Filter down to accepted peaks only
    accepted_peaks = peak_df[peak_df["key"].str.contains(f"{keyword}_accepted", na=False)]

    # Calculate RR intervals (in seconds) and corresponding rate (bpm) for accepted peaks
    if len(accepted_peaks) > 1:
        rr_intervals = np.diff(accepted_peaks["refined_index"]) / sampling_rate
        rate_values = 60 / rr_intervals
    else:
        rr_intervals = np.array([])
        rate_values = np.array([])

    if convert_stroke_to_stride_rate and len(rate_values) > 0:
        rate_values = rate_values / 2.0

    # Initialize the rate data array
    rate_data = np.full(len(signal_subset_df), np.nan)

    # Interpolate rate values between detected peaks
    for i in range(len(rate_values)):
        start_idx = accepted_peaks["refined_index"].iloc[i]
        end_idx = accepted_peaks["refined_index"].iloc[i + 1]
        rate_data[start_idx:end_idx] = rate_values[i]

    # Fill the remaining segment with the last rate value (if applicable)
    if len(accepted_peaks) > 0 and len(rate_values) > 0:
        rate_data[accepted_peaks["refined_index"].iloc[-1] :] = rate_values[-1]

    print(f"Rate values are: {len(rate_values)} values long and accepted peaks are: {len(accepted_peaks)}")
    # Use datetime from signal_subset_df or accepted peaks df
    datetime = signal_subset_df["datetime"] if "datetime" in signal_subset_df else accepted_peaks["datetime"]

    # Create a DataFrame for the rate
    rate_df = pd.DataFrame({"datetime": datetime, rate_key: rate_data})

    # Save intermediate results (conditionally based on params)
    intermediate_keys = {
        "broad_bandpass": "enable_bandpass",
        "narrow_bandpass": "enable_bandpass",
        "spike_removed_signal": "enable_spike_removal",
        "smoothed": "enable_smoothing",
        "normalized": "enable_normalization",
    }
    for key, param_flag in intermediate_keys.items():
        if key in results and params.get(param_flag, False):  # Only save if enabled in params
            derived_key = f"hr_{key}" if mode == "heart_rate" else f"sr_{key}"  # e.g., heart_rate_smoothed_signal
            derived_df = pd.DataFrame({"datetime": datetime, key: results[key]})
            data_pkl.signal_data[derived_key] = derived_df

            # Add signal_info for the intermediate key
            intermediate_info = {
                "channels": [key],
                "metadata": {
                    key: {
                        "original_name": f"{rate_key.capitalize()} {key.replace('_', ' ').capitalize()}",
                        "unit": "signal units",
                        "parent_signal": parent_signal,
                    }
                },
                "derived_from_signals": [parent_signal],
                "transformation_log": [
                    f"Derived from {parent_signal} during {rate_key} processing.",
                    f"Parameters: {', '.join(f'{k}={v}' for k, v in params.items())}",
                ],
            }
            data_pkl.signal_info[derived_key] = intermediate_info
            print(f"Saved derived info for intermediate signal: {derived_key}")

    # Construct transformation log
    transformation_log = [
        f"Parameters used: {', '.join(f'{k}={v}' for k, v in params.items())}.",
        f"{rate_key} was calculated using RR intervals derived from peaks.",
    ]
    if convert_stroke_to_stride_rate:
        transformation_log.append(
            "Quadruped adjustment applied: divided detected stroke rate by 2 to estimate stride rate."
        )

    # Save rate data and metadata via shared utility.
    upsert_rate_signal(
        data_pkl=data_pkl,
        rate_df=rate_df,
        rate_key=rate_key,
        parent_signal=parent_signal,
        transformation_log=transformation_log,
    )
    print(f"Derived {rate_key} data and metadata saved successfully.")

    # Create DataFrame for accepted events
    accepted_events = pd.DataFrame(
        {
            "datetime": accepted_peaks["datetime"].iloc[:-1],  # Use all but the last timestamp for intervals
            "key": accepted_peaks["key"].iloc[:-1].values,  # Use all but the last key
            "duration": rr_intervals if len(rr_intervals) > 0 else np.nan,
            "short_description": description,
            "type": "point",
            "value": rate_values if len(rate_values) > 0 else np.nan,
        }
    )

    # Select only 'datetime' and 'key' for rejected peaks
    rejected_events = peak_df[peak_df["key"].str.contains(f"{keyword}_rejected", na=False)][["datetime", "key"]].copy()

    rejected_events["duration"] = np.nan  # No duration for rejected peaks
    rejected_events["value"] = np.nan  # No heart rate for rejected peaks
    rejected_events["short_description"] = description
    rejected_events["type"] = "point"

    # Combine accepted and rejected events
    rate_events = pd.concat([accepted_events, rejected_events], ignore_index=True).sort_values(
        by="datetime"
    )

    # Process detected peaks for event data
    rate_events.reset_index(drop=True, inplace=True)
    # Prepend "heart" or "stroke" to the key based on the mode
    rate_events["key"] = ("stroke" + rate_events["key"] if mode == "stroke_rate" else "heart" + rate_events["key"])
    mode_specific_keyword = ("stroke" + keyword if mode == "stroke_rate" else "heart" + keyword)

    # Remove existing events with the same prefix
    data_pkl.event_data["key"] = data_pkl.event_data["key"].astype(str)
    data_pkl.event_data = data_pkl.event_data[
        ~data_pkl.event_data["key"].str.contains(mode_specific_keyword, na=False)
    ]

    # Append new events
    data_pkl.event_data = pd.concat([data_pkl.event_data, rate_events], ignore_index=True)
    print(f"Appended {rate_key} events successfully.")
