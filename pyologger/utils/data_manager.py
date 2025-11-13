import pickle
import pandas as pd
import numpy as np

def clear_intermediate_signals(data_pkl, remove_keys=None):
    """
    Clears specified intermediate signals from the `data_pkl.signal_data` structure.
    
    Args:
        data_pkl: The data structure containing derived data and metadata.
        remove_keys (list): A list of keys to remove from `data_pkl.signal_data`.
                            If None, no keys will be removed.
    
    Returns:
        None
    """
    if remove_keys is None:
        print("No keys specified for removal. No intermediate signals cleared.")
        return

    # Identify and remove keys
    for key in remove_keys:
        if key in data_pkl.signal_data:
            del data_pkl.signal_data[key]
            if key in data_pkl.signal_info:
                del data_pkl.signal_info[key]
            print(f"Removed intermediate signal: {key}")
        else:
            print(f"Key not found, no removal performed for: {key}")

    print("Specified intermediate signals have been cleared.")