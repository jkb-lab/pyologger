import pandas as pd
import numpy as np
from pyologger.analyze_data.find_segments import *

def append_stats(data, segment_df, statistics):
    """
    Append statistics to segments based on the original data and specified operations.

    Parameters:
        data (pd.DataFrame): Original dataset with detailed columns for calculations.
        segment_df (pd.DataFrame): DataFrame with segment information, including start and end indices.
        statistics (list of tuples): Each tuple contains the operation ('max', 'mean', 'mode', etc.) and the column name.

    Returns:
        pd.DataFrame: Updated segment DataFrame with appended statistics.
    """
    stats_results = []

    for _, segment in segment_df.iterrows():
        start, end = segment['start_index'], segment['end_index']
        segment_data = data.iloc[start:end+1]  # Slice the original data based on start and end indices

        stats = {}
        stats.update(segment)  # Copy the original segment information

        for operation, column in statistics:
            if column not in segment_data:
                continue
            if operation == "mean":
                stats[f"{column}_mean"] = segment_data[column].mean()
            elif operation == "max":
                stats[f"{column}_max"] = segment_data[column].max()
            elif operation == "min":
                stats[f"{column}_min"] = segment_data[column].min()
            elif operation == "mode":
                stats[f"{column}_mode"] = segment_data[column].mode().iloc[0] if not segment_data[column].mode().empty else None
            elif operation == "cumsum":
                stats[f"{column}_sum"] = segment_data[column].cumsum()
            elif operation == "sum":
                stats[f"{column}_sum"] = segment_data[column].sum()
            elif operation == "std":
                stats[f"{column}_std"] = segment_data[column].std()
            elif operation == "abs_mean":
                stats[f"{column}_abs_mean"] = segment_data[column].abs().mean()
            elif operation == "abs_max":
                stats[f"{column}_abs_max"] = segment_data[column].abs().max()
            elif operation == "abs_min":
                stats[f"{column}_abs_min"] = segment_data[column].abs().min()
            elif operation == "abs_sum":
                stats[f"{column}_abs_sum"] = segment_data[column].abs().sum()
            elif operation == "abs_cumsum":
                stats[f"{column}_abs_cumsum"] = segment_data[column].abs().cumsum()
            else:
                raise ValueError(f"Unsupported operation: {operation}")

        stats_results.append(stats)

    # Convert to DataFrame
    updated_segment_df = pd.DataFrame(stats_results)
    return updated_segment_df

def calculate_tortuosity(data, segment_df, x_col, y_col):
    """
    Calculate tortuosity for segments based on x and y positions.

    Parameters:
        data (pd.DataFrame): Original dataset with x and y positions.
        segment_df (pd.DataFrame): DataFrame with segment information, including start and end indices.
        x_col (str): Column name for x positions in the `data` DataFrame.
        y_col (str): Column name for y positions in the `data` DataFrame.

    Returns:
        pd.DataFrame: Updated segment DataFrame with tortuosity values.
    """
    tortuosity_results = []

    for _, segment in segment_df.iterrows():
        start, end = segment['start_index'], segment['end_index']
        segment_data = data.iloc[start:end+1]  # Slice the original data based on start and end indices

        if len(segment_data) < 2:  # Tortuosity requires at least two points
            segment['tortuosity'] = np.nan
        else:
            # Calculate total distance traveled
            # TODO: Check math for this calculation
            # PATH LENGTH / STRAIGHT LINE DISTANCE
            distances = np.sqrt(np.diff(segment_data[x_col])**2 + np.diff(segment_data[y_col])**2)
            total_distance = np.sum(distances)

            # Calculate straight-line distance
            start_point = segment_data.iloc[0]
            end_point = segment_data.iloc[-1]
            straight_line_distance = np.sqrt((end_point[x_col] - start_point[x_col])**2 + \
                                            (end_point[y_col] - start_point[y_col])**2)

            # Calculate tortuosity
            segment['tortuosity'] = total_distance / straight_line_distance if straight_line_distance > 0 else np.nan

        tortuosity_results.append(segment)

    # Convert to DataFrame
    updated_segment_df = pd.DataFrame(tortuosity_results)
    return updated_segment_df

def filter_and_adjust_children(parent_intervals, child_intervals):
    """
    Adjusts child interval start and end times to fit within parent intervals.

    Parameters:
        parent_intervals (pd.DataFrame): Parent intervals with start_datetime and end_datetime.
        child_intervals (pd.DataFrame): Child intervals with start_datetime, end_datetime.

    Returns:
        pd.DataFrame: Adjusted child intervals.
    """
    adjusted_start_count = 0
    adjusted_end_count = 0
    multi_parent_count = 0
    multi_parent_cases = []

    adjusted_children = []

    for _, parent in parent_intervals.iterrows():
        parent_start = parent['start_datetime']
        parent_end = parent['end_datetime']

        # Identify child intervals overlapping with this parent interval
        overlapping_children = child_intervals[
            (child_intervals['end_datetime'] > parent_start) & 
            (child_intervals['start_datetime'] < parent_end)
        ].copy()

        for i, child in overlapping_children.iterrows():
            original_start, original_end = child['start_datetime'], child['end_datetime']
            adjusted_start, adjusted_end = original_start, original_end

            if original_start < parent_start and original_end <= parent_end:
                adjusted_start = parent_start
                adjusted_start_count += 1

            if original_end > parent_end and original_start >= parent_start:
                adjusted_end = parent_end
                adjusted_end_count += 1

            if original_start < parent_start and original_end > parent_end:
                overlapping_parents = parent_intervals[
                    (parent_intervals['start_datetime'] <= original_end) & 
                    (parent_intervals['end_datetime'] >= original_start)
                ]
                multi_parent_count += 1
                multi_parent_cases.append(
                    f"Child interval {original_start} to {original_end} spans {len(overlapping_parents)} parent intervals."
                )
                continue  # Skip this child since it requires special handling

            child['start_datetime'] = adjusted_start
            child['end_datetime'] = adjusted_end
            adjusted_children.append(child)

    print(f"{adjusted_start_count} child intervals had their start times adjusted.")
    print(f"{adjusted_end_count} child intervals had their end times adjusted.")
    print(f"{multi_parent_count} child intervals spanned multiple parent intervals.")

    return pd.DataFrame(adjusted_children)

def summarize_children_per_parent(parent_intervals, child_intervals_dict):
    """
    Summarizes multiple mutually exclusive child intervals within each parent interval.

    Parameters:
        parent_intervals (pd.DataFrame): Parent intervals (e.g., surface intervals).
        child_intervals_dict (dict): Dictionary where keys are child interval names (e.g., 'spin', 'nonspin')
                                     and values are DataFrames with start_datetime, end_datetime, and duration.

    Returns:
        pd.DataFrame: Parent intervals with child interval statistics summarized.
    """
    results = []

    for _, parent in parent_intervals.iterrows():
        parent_start = parent['start_datetime']
        parent_end = parent['end_datetime']
        parent_duration = (parent_end - parent_start).total_seconds()

        stats = parent.to_dict()

        for child_name, child_df in child_intervals_dict.items():
            children_in_parent = child_df[
                (child_df['start_datetime'] >= parent_start) & 
                (child_df['end_datetime'] <= parent_end)
            ]

            total_child_duration = children_in_parent['duration'].sum() if not children_in_parent.empty else 0

            stats[f'percent_time_{child_name}'] = (
                total_child_duration / parent_duration * 100 if parent_duration > 0 else 0
            )
            stats[f'count_{child_name}'] = len(children_in_parent)
            stats[f'duration_{child_name}'] = total_child_duration
            stats[f'tortuosity_mean_{child_name}'] = (
                children_in_parent['tortuosity'].mean() if 'tortuosity' in child_df.columns and not children_in_parent.empty else np.nan
            )

        results.append(stats)

    return pd.DataFrame(results)

