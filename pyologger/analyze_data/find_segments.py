import pandas as pd

def find_segments(data, column, criteria, min_duration=None, animal_id_col='animal_id'):
    """
    Loop-based function to find segments in a dataset based on a column and criteria,
    while handling changes in animal ID to avoid grouping across individuals.

    Parameters:
        data (pd.DataFrame): Input dataset with a datetime column.
        column (str): Name of column to apply the criteria (e.g., `turn_direction` or `Depth`).
        criteria (callable): Function to evaluate whether a value satisfies the segment condition.
        min_duration (float, optional): Minimum duration (in seconds) to include a segment.
                                         Requires 'datetime' column in the dataset.
        animal_id_col (str, optional): Name of the column containing animal IDs. Defaults to 'animal_id'.

    Returns:
        pd.DataFrame: DataFrame where each row corresponds to a detected segment, with relevant attributes.
    """
    # If no animal_id_col is provided or the column doesn't exist, assign a temporary ID for all rows
    if not animal_id_col or animal_id_col not in data.columns:
        data['TEMP_ANIMAL_ID'] = "TEMP_ANIMAL_ID"
        animal_id_col = 'TEMP_ANIMAL_ID'

    segments = []
    segment_start_index = None
    current_satisfied = False
    current_animal_id = None

    # Loop through the dataset
    for i in range(len(data)):
        value = data[column].iloc[i]
        animal_id = data[animal_id_col].iloc[i]
        satisfies_criteria = criteria(value)

        # Handle transitions between animal IDs
        if current_animal_id is not None and animal_id != current_animal_id:
            if current_satisfied:
                segment_end_index = i - 1
                start_time = data['datetime'].iloc[segment_start_index]
                end_time = data['datetime'].iloc[segment_end_index]
                duration = (end_time - start_time).total_seconds()

                if min_duration is None or duration >= min_duration:
                    segments.append({
                        animal_id_col: current_animal_id,
                        'start_index': segment_start_index,
                        'end_index': segment_end_index,
                        'start_datetime': start_time,
                        'end_datetime': end_time,
                        'duration': duration
                    })
            current_satisfied = False
            segment_start_index = None

        # Process the current row
        if satisfies_criteria:
            if not current_satisfied:
                segment_start_index = i
            current_satisfied = True
        elif current_satisfied:
            segment_end_index = i - 1
            start_time = data['datetime'].iloc[segment_start_index]
            end_time = data['datetime'].iloc[segment_end_index]
            duration = (end_time - start_time).total_seconds()

            if min_duration is None or duration >= min_duration:
                segments.append({
                    animal_id_col: animal_id,
                    'start_index': segment_start_index,
                    'end_index': segment_end_index,
                    'start_datetime': start_time,
                    'end_datetime': end_time,
                    'duration': duration
                })
            current_satisfied = False
            segment_start_index = None

        current_animal_id = animal_id

    # Handle the last segment
    if current_satisfied:
        segment_end_index = len(data) - 1
        start_time = data['datetime'].iloc[segment_start_index]
        end_time = data['datetime'].iloc[segment_end_index]
        duration = (end_time - start_time).total_seconds()

        if min_duration is None or duration >= min_duration:
            segments.append({
                animal_id_col: current_animal_id,
                'start_index': segment_start_index,
                'end_index': segment_end_index,
                'start_datetime': start_time,
                'end_datetime': end_time,
                'duration': duration
            })

    # Convert to DataFrame
    segments_df = pd.DataFrame(segments)

    return segments_df

def find_adjusted_segments(df_all, parent_intervals, child_intervals, min_duration, animal_id_col='SealID'):
    """
    Uses a combined criteria approach to extract segments where both the parent 
    and child conditions are met.

    Parameters:
        df_all (pd.DataFrame): Full dataset with datetime.
        parent_intervals (pd.DataFrame): Parent intervals (e.g., dives/surfacings).
        child_intervals (pd.DataFrame): Child intervals (e.g., spins/non-spins).
        min_duration (float): Minimum segment duration in seconds.
        animal_id_col (str): Name of the column containing animal IDs.

    Returns:
        pd.DataFrame: Detected segments with adjusted start and end times.
    """
    # Initialize criteria column
    df_all["IN_PARENT"] = 0
    df_all["IN_CHILD"] = 0
    
    # Mark timepoints that fall within parent intervals
    for _, parent in parent_intervals.iterrows():
        mask = (df_all["datetime"] >= parent["start_datetime"]) & (df_all["datetime"] <= parent["end_datetime"])
        df_all.loc[mask, "IN_PARENT"] = 1

    # Mark timepoints that fall within child intervals
    for _, child in child_intervals.iterrows():
        mask = (df_all["datetime"] >= child["start_datetime"]) & (df_all["datetime"] <= child["end_datetime"])
        df_all.loc[mask, "IN_CHILD"] = 1

    # Final column where both parent and child conditions are met
    df_all["IN_PARENT_AND_CHILD"] = (df_all["IN_PARENT"] == 1) & (df_all["IN_CHILD"] == 1)

    # Run the existing function to extract segments
    adjusted_segments = find_segments(
        data=df_all,
        column="IN_PARENT_AND_CHILD",
        criteria=lambda x: x == True,
        min_duration=min_duration,
        animal_id_col=animal_id_col
    )

    return adjusted_segments

def find_gaps_between_segments(dive_segments, original_data, animal_id_col='SealID'):
    """
    Identify gaps between segments (surface intervals) for each animal,
    excluding periods before the first dive and after the last dive.

    Parameters:
        dive_segments (pd.DataFrame): Detected dive segments with start and end indices.
        original_data (pd.DataFrame): Original dataset with datetime column.
        animal_id_col (str): Name of the column containing animal IDs.

    Returns:
        pd.DataFrame: DataFrame of surface intervals (gaps) with start, end, and duration.
    """
    gaps = []
    
    # Group dive segments by animal ID
    grouped_dive_segments = dive_segments.groupby(animal_id_col)
    
    for animal_id, group in grouped_dive_segments:
        group = group.sort_values(by="start_index")
        
        # Exclude data before the first dive and after the last dive
        first_dive_end = group.iloc[0]['end_index']
        last_dive_start = group.iloc[-1]['start_index']
        
        # Find surface intervals (gaps) between consecutive dives
        for i in range(len(group) - 1):
            current_dive_end = group.iloc[i]['end_index']
            next_dive_start = group.iloc[i + 1]['start_index']
            
            # Calculate surface interval details
            gap_start_index = current_dive_end + 1
            gap_end_index = next_dive_start - 1
            gap_start_time = original_data.loc[gap_start_index, 'datetime']
            gap_end_time = original_data.loc[gap_end_index, 'datetime']
            gap_duration = (gap_end_time - gap_start_time).total_seconds()
            
            gaps.append({
                animal_id_col: animal_id,
                'start_index': gap_start_index,
                'end_index': gap_end_index,
                'start_datetime': gap_start_time,
                'end_datetime': gap_end_time,
                'duration': gap_duration
            })
    
    # Convert gaps list to DataFrame
    gaps_df = pd.DataFrame(gaps)
    
    return gaps_df

