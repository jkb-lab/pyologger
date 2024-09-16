import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, bartlett, find_peaks
import wfdb
from wfdb import processing

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
    return np.convolve(np.abs(signal), bartlett(window), mode='same')

# Sliding Window Normalization Function
def sliding_window_normalization(signal, window_size, noise=1e-10):
    half_window = window_size // 2
    normalized_signal = np.array([(signal[i] - np.mean(signal[max(0, i - half_window):min(len(signal), i + half_window)])) / 
                     (np.std(signal[max(0, i - half_window):min(len(signal), i + half_window)]) + noise) 
                     for i in range(len(signal))])
    return normalized_signal

# Peak Refinement with WFDB
def refine_peaks_with_wfdb(cleaned_signal, rpeaks, fs, search_radius=0.5, peak_dir="compare"):
    return wfdb.processing.correct_peaks(cleaned_signal, rpeaks, smooth_window_size = int(0.5 * fs), 
                                         search_radius=int(search_radius * fs), peak_dir=peak_dir)


# Absolute Maxima Search Function
def absolute_maxima_search(refined_peaks, original_signal, QRS_width, sample_rate):
    QRS_samples = int(QRS_width * sample_rate)
    return [np.argmax(original_signal[max(0, peak - QRS_samples):peak + 1]) + max(0, peak - QRS_samples) for peak in refined_peaks]

# Combined Peak Detection Function
def peak_detect(signal, sampling_rate, QRS_width=0.200, 
                lowcut=5, highcut=20, filter_order=2,
                spike_threshold=400, smooth_sec_multiplier=3,
                window_size_multiplier=5, normalization_noise=1e-10,
                peak_height=-0.4, peak_distance_sec=0.2, search_radius_sec=0.5,
                min_peak_height=500, max_peak_height=12000,
                enable_bandpass=True, enable_spike_removal=True, enable_smoothing=True, 
                enable_normalization=True, enable_refinement=True):
    
    results = {}

    # Bandpass filter
    if enable_bandpass:
        bandpassed_signal = bandpass_filter(signal, lowcut=lowcut, highcut=highcut, fs=sampling_rate, order=filter_order)
        results['bandpassed_signal'] = bandpassed_signal
    else:
        bandpassed_signal = signal

    # Spike removal
    if enable_spike_removal:
        spike_removed_signal = remove_spikes(bandpassed_signal, threshold=spike_threshold)
        results['spike_removed_signal'] = spike_removed_signal
    else:
        spike_removed_signal = bandpassed_signal

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
        refined_peaks = refine_peaks_with_wfdb(spike_removed_signal, detected_peaks, fs = sampling_rate, search_radius=search_radius_sec, 
                                               peak_dir = "compare")
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
