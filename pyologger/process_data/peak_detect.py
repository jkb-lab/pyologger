import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal.windows import bartlett
from scipy import stats
import plotly.graph_objs as go

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

# Combined Peak Detection Function
def peak_detect(signal, sampling_rate, QRS_width=0.200, 
                broad_lowcut= BROAD_LOW_CUTOFF, broad_highcut = BROAD_HIGH_CUTOFF,
                narrow_lowcut=NARROW_LOW_CUTOFF, narrow_highcut=NARROW_HIGH_CUTOFF, 
                filter_order=FILTER_ORDER,
                spike_threshold=SPIKE_THRESHOLD, smooth_sec_multiplier=SMOOTH_SEC_MULTIPLIER,
                window_size_multiplier=WINDOW_SIZE_MULTIPLIER, normalization_noise=NORMALIZATION_NOISE,
                peak_height=PEAK_HEIGHT, peak_distance_sec=PEAK_DISTANCE_SEC, search_radius_sec=SEARCH_RADIUS_SEC,
                min_peak_height=MIN_PEAK_HEIGHT, max_peak_height=MAX_PEAK_HEIGHT,
                enable_bandpass=True, enable_spike_removal=True, enable_smoothing=True, 
                enable_normalization=True, enable_refinement=True):
    
    results = {}

    # Bandpass filter
    if enable_bandpass:
        broad_bandpassed_signal = bandpass_filter(signal, lowcut=broad_lowcut, highcut=broad_highcut, fs=sampling_rate, order=filter_order)
        results['broad_bandpassed_signal'] = broad_bandpassed_signal
        narrow_bandpassed_signal = bandpass_filter(signal, lowcut=narrow_lowcut, highcut=narrow_highcut, fs=sampling_rate, order=filter_order)
        results['narrow_bandpassed_signal'] = narrow_bandpassed_signal
    else:
        narrow_bandpassed_signal = signal

    # Spike removal
    if enable_spike_removal:
        spike_removed_signal = remove_spikes(narrow_bandpassed_signal, threshold=spike_threshold)
        results['spike_removed_signal'] = spike_removed_signal
    else:
        spike_removed_signal = narrow_bandpassed_signal

    # Smoothing
    if enable_smoothing:
        smoothed_signal = smooth_signal(spike_removed_signal, smooth_sec=QRS_width * smooth_sec_multiplier, fs=sampling_rate)
        results['smoothed_signal'] = smoothed_signal
    else:
        smoothed_signal = spike_removed_signal

    # Normalization
    if enable_normalization:
        normalized_signal = sliding_window_normalization(smoothed_signal, int(window_size_multiplier * sampling_rate), noise=normalization_noise)
        results['normalized_signal'] = normalized_signal
    else:
        normalized_signal = smoothed_signal

    # Peak Detection
    detected_peaks = find_peaks(normalized_signal, height=peak_height, distance=int(peak_distance_sec * sampling_rate))[0]
    results['detected_peaks'] = detected_peaks

    # Peak Refinement
    if enable_refinement:
        refined_peaks = refine_peaks_with_wfdb(spike_removed_signal, detected_peaks, fs = sampling_rate, search_radius=search_radius_sec, sample_rate=sampling_rate,
                                               peak_dir = "both")
        results['refined_peaks'] = refined_peaks
        refined_indices = refined_peaks
        #refined_indices = absolute_maxima_search(refined_peaks, spike_removed_signal, QRS_width * 2, sampling_rate)
    else:
        refined_indices = detected_peaks

    # Remove duplicate refined_indices while keeping corresponding heights aligned
    unique_refined_indices, index_positions = np.unique(refined_indices, return_index=True)
    results['unique_refined_indices'] = unique_refined_indices

    # Align heights with unique refined indices
    height_original = smoothed_signal[unique_refined_indices]
    height_normalized = normalized_signal[refined_peaks[index_positions]]
    results['height_original'] = height_original
    results['height_normalized'] = height_normalized

    # Filter out peaks that are too close to each other
    min_distance_samples = int(peak_distance_sec * sampling_rate)
    filtered_indices = [unique_refined_indices[0]] if len(unique_refined_indices) > 0 else []  # Always include the first peak if available
    for i in range(1, len(unique_refined_indices)):
        if unique_refined_indices[i] - filtered_indices[-1] >= min_distance_samples:
            filtered_indices.append(unique_refined_indices[i])
    
    # Update the DataFrame with the filtered indices
    peak_df = pd.DataFrame({
        'refined_index': filtered_indices,
        'height_original': smoothed_signal[filtered_indices],
        'height_normalized': normalized_signal[filtered_indices]
    })

    # Filter out peaks based on both minimum and maximum peak height
    filtered_peak_df = peak_df[(peak_df['height_original'] >= min_peak_height) & 
                               (peak_df['height_original'] <= max_peak_height)].reset_index(drop=True)
    results['filtered_peak_df'] = filtered_peak_df

    # Label peaks as accepted or rejected
    peak_df['key'] = np.where(peak_df['refined_index'].isin(filtered_peak_df['refined_index']), 
                              'heartbeat_auto_detect_accepted', 
                              'heartbeat_auto_detect_rejected')
    # Ensure no NaN values in 'key'
    peak_df['key'].fillna('heartbeat_auto_detect_unknown', inplace=True)
    results['peak_df'] = peak_df

    return results
