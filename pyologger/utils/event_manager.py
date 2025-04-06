import pandas as pd
import numpy as np

def create_state_event(state_df, key, value_column=None, start_time_column=None,
                      duration_column=None, description=None, long_description=None,
                      existing_events=None):
    """
    Create a state event DataFrame for a given state, such as dives or other intervals, and update the existing events.

    Parameters
    ----------
    state_df : pandas.DataFrame
        DataFrame containing state information, including start time and relevant metrics.
    key : str
        Key representing the type of state (e.g., 'dive', 'rest', etc.).
    value_column : str, optional
        Column name in state_df containing the primary value (e.g., 'max_depth').
    start_time_column : str
        Column name in state_df containing the start time of the state.
    duration_column : str, optional
        Column name in state_df containing the duration of the state (default is None).
    description : str, optional
        Short description for the state events (default is key+'_start').
    long_description : str, optional
        Long description to include with events (default is None).
    existing_events : pandas.DataFrame, optional
        Existing event data to check for duplicate keys (default is None).

    Returns
    -------
    pandas.DataFrame
        Updated event DataFrame with new events appended to existing events, replacing any with the same key.
    """
    import numpy as np
    import pandas as pd

    # Default descriptions
    if description is None:
        description = f"{key}_start"
    if long_description is None:
        long_description = np.nan

    # Check for existing events with the same key
    if existing_events is not None and not existing_events.empty:
        existing_key_events = existing_events[existing_events['key'] == key]
        if not existing_key_events.empty:
            print(f"Overwriting {len(existing_key_events)} existing events with key: '{key}'")
        # Remove existing events with the same key
        existing_events = existing_events[existing_events['key'] != key]
    else:
        # Initialize empty DataFrame if no existing events are provided
        existing_events = pd.DataFrame()

    # Build the new event DataFrame
    new_events = pd.DataFrame({
        'date': state_df[start_time_column].dt.floor('D').dt.strftime('%Y-%m-%d'),
        'time': state_df[start_time_column].dt.strftime('%H:%M:%S.%f').str[:-3],
        'value': state_df[value_column] if value_column else np.nan,
        'type': 'state',
        'key': key,
        'duration': state_df[duration_column] if duration_column else np.nan,
        'short_description': description,
        'long_description': long_description,
        'datetime': state_df[start_time_column]
    })

    # Combine new and existing events
    updated_events = pd.concat([existing_events, new_events], ignore_index=True)

    return updated_events
