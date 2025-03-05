import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt

def compute_odba(data, fs, method='vedba', n=None, fh=None):
    """
    Compute Overall Dynamic Body Acceleration (ODBA) or VeDBA (Vectorial Dynamic Body Acceleration).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns ['datetime', 'ax', 'ay', 'az'].
    fs : float
        Sampling rate in Hz.
    method : str, optional
        'wilson' for 1-norm ODBA or 'vedba' for 2-norm VeDBA. Default is 'vedba'.
    n : int, optional
        Length of the rectangular window (samples) for high-pass filtering. Required if `method` is 'wilson' or 'vedba'.
    fh : float, optional
        High-pass filter cutoff frequency in Hz. Required if using FIR filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['datetime', 'odba'], where 'odba' is the computed ODBA or VeDBA.
    """
    if not all(col in data.columns for col in ['datetime', 'ax', 'ay', 'az']):
        raise ValueError("Input data must have columns: ['datetime', 'ax', 'ay', 'az']")

    acc_data = data[['ax', 'ay', 'az']].to_numpy()

    if method in ['wilson', 'vedba']:
        if n is None:
            raise ValueError("Window size 'n' must be provided for 'wilson' or 'vedba' methods.")
        n = 2 * (n // 2) + 1  # Ensure n is odd
        nz = n // 2
        h = np.concatenate([np.zeros(nz), [1], np.zeros(nz)]) - np.ones(n) / n

        # Apply high-pass filter using a moving average method
        padded_acc = np.vstack([np.tile(acc_data[0, :], (nz, 1)), acc_data, np.tile(acc_data[-1, :], (nz, 1))])
        Ah = np.array([np.convolve(padded_acc[:, i], h, mode='valid') for i in range(acc_data.shape[1])]).T

        if method == 'vedba':
            odba_values = np.sqrt(np.sum(Ah**2, axis=1))  # VeDBA: Use 2-norm
        elif method == 'wilson':
            odba_values = np.sum(np.abs(Ah), axis=1)      # Wilson: Use 1-norm
        else:
            raise ValueError(f"Unknown method: {method}")

    elif fh is not None:
        if acc_data.shape[0] <= 2 * (fs / fh):
            raise ValueError(f"Data needs at least {2 * (fs / fh) + 1} rows to compute FIR filter.")
        # Design FIR high-pass filter
        n = 4 * int(fs / fh)
        b = firwin(n, cutoff=fh / (fs / 2), pass_zero=False)
        Ah = np.array([filtfilt(b, 1, acc_data[:, i]) for i in range(acc_data.shape[1])]).T
        odba_values = np.sqrt(np.sum(Ah**2, axis=1))  # Use 2-norm
    else:
        raise ValueError("Either 'fh' or 'n' must be provided to calculate ODBA.")

    # Combine datetime with ODBA values into a DataFrame
    odba_df = pd.DataFrame({
        'datetime': data['datetime'],
        'odba': odba_values
    })

    return odba_df
