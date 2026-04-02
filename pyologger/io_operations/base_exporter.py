import xarray as xr
import pandas as pd
import numpy as np
import pytz
import os
import mne
from datetime import datetime, date, time
from pyologger.process_data.sampling import *

def collate_data(data_pkl, signal_data_keys, output_frequency):
    """
    Collates specified signal data, resampling to a given output frequency.
    
    Parameters:
        data_pkl: Data object containing signal data as DataFrames.
        signal_data_keys: List of signal data keys to include.
        output_frequency: Desired output frequency in Hz (assumes datetime index).
        
    Returns:
        A collated DataFrame with all specified data resampled to output_frequency.
    """
    # Initialize base DataFrame with datetime index from 'pressure' signal
    base_df = data_pkl.signal_data['pressure'][['datetime']].copy()
    base_df.set_index('datetime', inplace=True)

    # Create a new time index based on the output frequency
    start_time = base_df.index.min()
    end_time = base_df.index.max()
    time_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{int(1 / output_frequency * 1000)}ms"  # Convert Hz to ms
    )
    collated_df = pd.DataFrame(index=time_index)

    def resample_data(df, frequency, original_frequency):
        """Resamples data to the specified frequency using interpolation."""
        df = df.set_index('datetime')

        # Convert Hz to millisecond intervals
        resample_interval = f"{int(1 / frequency * 1000)}ms"
        df_resampled = df.resample(resample_interval).mean()

        # Interpolate missing values to match the new frequency
        return df_resampled.interpolate()

    # Process signal data
    for key in signal_data_keys:
        if key in data_pkl.signal_data:
            original_fs = calculate_sampling_frequency(data_pkl.signal_data[key]['datetime'])
            if original_fs:
                resampled_data = resample_data(data_pkl.signal_data[key], output_frequency, original_fs)
                collated_df = collated_df.merge(resampled_data, left_index=True, right_index=True, how='left')

    # Reset index to include datetime
    collated_df.reset_index(inplace=True)
    collated_df.rename(columns={'index': 'datetime'}, inplace=True)

    return collated_df


class BaseExporter:
    """Handles exporting data from the DataReader object."""

    def __init__(self, datareader):
        self.datareader = datareader

    def save_data(self, data, logger_id, filename, save_csv=True, save_parq=False):
        """Saves the processed data to disk in CSV and/or Parquet format."""
        output_folder = self.datareader.output_folder
        os.makedirs(output_folder, exist_ok=True)

        if save_csv:
            csv_path = os.path.join(output_folder, filename)
            data.to_csv(csv_path, index=False)
            print(f"Data for {logger_id} saved as CSV to: {csv_path}")

        if save_parq:
            parq_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.parquet")

            # Convert non-serializable attributes to strings
            for key, value in data.attrs.items():
                if isinstance(value, (datetime, date, time)):
                    data.attrs[key] = value.isoformat()  # Convert to string format

            try:
                data.to_parquet(parq_path, index=False)
                print(f"Data for {logger_id} saved as Parquet to: {parq_path}")
            except Exception as e:
                print(f"❌ Error saving Parquet file: {e}")
                print(f"Attempting to remove all attributes and retry...")
                data.attrs = {}  # Remove attributes entirely
                data.to_parquet(parq_path, index=False)
                print(f"✅ Parquet file saved without attributes: {parq_path}")

    def save_to_netcdf(self, datareader, filepath):
        """Saves the current state of the DataReader object to a NetCDF file."""

        def convert_to_compatible_array(df: pd.DataFrame):
            """Convert DataFrame columns to compatible numpy arrays."""
            # Early exit for None or empty
            if df is None or df.empty:
                return np.array([])

            df = df.copy()

            def _stringify(series: pd.Series) -> pd.Series:
                return series.astype("string").fillna("")

            for col in df.columns:
                if df[col].dtype == 'object':
                    # Look at a non-NA example if possible
                    non_na = df[col].dropna()
                    first = non_na.iloc[0] if not non_na.empty else None

                    # Handle datetime-like object column
                    if isinstance(first, (datetime, date, time)):
                        df[col] = _stringify(df[col])
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col])
                    else:
                        # Attempt to convert to float, if fails convert to string
                        try:
                            df[col] = df[col].astype(float)
                        except (ValueError, TypeError):
                            df[col] = _stringify(df[col])
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col])

            # Check the number of columns in the DataFrame
            if df.shape[1] == 1:
                # If there is only one column, return a flat array
                series = df.iloc[:, 0]
                if series.dtype == 'object' or pd.api.types.is_string_dtype(series.dtype):
                    return _stringify(series).to_numpy(dtype=str)
                return series.to_numpy()
            else:
                # Multi-column signal arrays require one consistent dtype.
                if any(
                    (dtype == 'object') or pd.api.types.is_string_dtype(dtype)
                    for dtype in df.dtypes
                ):
                    return df.apply(_stringify).to_numpy(dtype=str)

                def safe_to_numeric(series):
                    try:
                        return pd.to_numeric(series)
                    except ValueError:
                        return series

                return df.apply(safe_to_numeric).to_numpy()

        def serialize_value(value):
            """Helper function to serialize values to be JSON-compatible."""
            if isinstance(value, (datetime, date, time)):
                return value.isoformat()
            elif isinstance(value, (list, tuple)):
                return [serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            else:
                return value

        def flatten_dict(prefix, d):
            """Flattens a dictionary and adds it to dataset attributes."""
            for key, value in d.items():
                flattened_key = f"{prefix}_{key}"
                try:
                    serialized_value = serialize_value(value)
                    if isinstance(serialized_value, (str, int, float, list, tuple, np.ndarray)):
                        ds.attrs[flattened_key] = serialized_value
                    else:
                        raise TypeError("Invalid value type for NetCDF serialization")
                except (TypeError, ValueError):
                    ds.attrs[flattened_key] = "Invalid entry"
                    print(f"Invalid entry recognized and placed in {flattened_key}")

        def create_coords(ndim, datetime_coord, variables, name):
            """Creates an xarray DataArray with appropriate dimensions and coordinates."""
            if ndim == 1:
                dims = [f"{name}_samples"]
                coords = {f"{name}_samples": datetime_coord}
            else:
                dims = [f"{name}_samples", f"{name}_variables"]
                coords = {
                    f"{name}_samples": datetime_coord,
                    f"{name}_variables": variables,
                }
            return dims, coords

        def create_data_array(data, dims, coords):
            """Creates an xarray DataArray with appropriate dimensions and coordinates."""
            return xr.DataArray(data, dims=dims, coords=coords)

        def set_variables_attr(ds, var_name, variables):
            """Sets the 'variables' or 'variable' attribute based on the type of 'variables'."""
            if isinstance(variables, list):
                ds[var_name].attrs['variables'] = variables
            else:
                ds[var_name].attrs['variable'] = variables

        # Create an empty xarray dataset
        ds = xr.Dataset()

        # Flatten the signal_data dictionaries into xarray DataArrays
        for signal_name, df in self.datareader.signal_data.items():
            signal_data = df.copy()
            if signal_data.empty:
                continue

            # Saving datetime as timezone-aware
            datetime_coord = pd.to_datetime(signal_data['datetime'])
            signal_data = signal_data.drop(columns=['datetime'])
            variables = [col for col in signal_data.columns]

            data_array = convert_to_compatible_array(signal_data)
            if data_array.size == 0:
                # Nothing to write for this signal
                continue

            var_name = f'signal_data_{signal_name}'
            ndim = data_array.ndim
            dims, coords = create_coords(ndim, datetime_coord, variables, signal_name)
            ds[var_name] = create_data_array(data_array, dims, coords)
            set_variables_attr(ds, var_name, variables)

        # Event data: export selected columns as separate variables (if present)
        columns_to_keep = ["type", "key", "value", "duration", "short_description", "long_description"]

        if isinstance(self.datareader.event_data, pd.DataFrame) and not self.datareader.event_data.empty:
            event_df = self.datareader.event_data.copy()
            # Ensure datetime exists and is usable
            if 'datetime' not in event_df.columns:
                print("⚠️ event_data has no 'datetime' column; skipping event export.")
            else:
                datetime_coord = pd.to_datetime(event_df['datetime'])

                for var in columns_to_keep:
                    if var not in event_df.columns:
                        continue

                    col_df = event_df[[var]].copy()
                    data_array = convert_to_compatible_array(col_df)
                    if data_array.size == 0:
                        # Skip empty column
                        continue

                    var_name = f'event_data_{var}'
                    ndim = data_array.ndim
                    # For event data, variables are just the column name
                    variables_event = [var]
                    dims, coords = create_coords(ndim, datetime_coord, variables_event, 'event_data')
                    ds[var_name] = create_data_array(data_array, dims, coords)
                    set_variables_attr(ds, var_name, var)

        def recursive_flatten_dict(prefix, d):
            """Recursively flattens a dictionary and adds it to dataset attributes."""
            if isinstance(d, dict):
                for key, value in d.items():
                    flattened_key = f"{prefix}_{key}"
                    if isinstance(value, dict):
                        recursive_flatten_dict(flattened_key, value)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                recursive_flatten_dict(f"{flattened_key}_{i}", item)
                            else:
                                try:
                                    serialized_value = serialize_value(item)
                                    if isinstance(serialized_value, (str, int, float, list, tuple, np.ndarray)):
                                        ds.attrs[f"{flattened_key}_{i}"] = serialized_value
                                    else:
                                        raise TypeError("Invalid value type for NetCDF serialization")
                                except (TypeError, ValueError):
                                    ds.attrs[f"{flattened_key}_{i}"] = "Invalid entry"
                                    print(f"⚠️ Invalid entry recognized and placed in {flattened_key}_{i}")
                    else:
                        try:
                            serialized_value = serialize_value(value)
                            if isinstance(serialized_value, (str, int, float, list, tuple, np.ndarray)):
                                ds.attrs[flattened_key] = serialized_value
                            else:
                                raise TypeError("Invalid value type for NetCDF serialization")
                        except (TypeError, ValueError):
                            ds.attrs[flattened_key] = "Invalid entry"
                            print(f"⚠️ Invalid entry recognized and placed in {flattened_key}")
            else:
                # If it's not a dictionary, just store the value
                try:
                    serialized_value = serialize_value(d)
                    if isinstance(serialized_value, (str, int, float, list, tuple, np.ndarray)):
                        ds.attrs[prefix] = serialized_value
                    else:
                        raise TypeError("Invalid value type for NetCDF serialization")
                except (TypeError, ValueError):
                    ds.attrs[prefix] = "Invalid entry"
                    print(f"⚠️ Invalid entry recognized and placed in {prefix}")

        # Flatten and add global attributes
        recursive_flatten_dict('deployment_info', self.datareader.deployment_info)
        recursive_flatten_dict('procedure_info', getattr(self.datareader, 'procedure_info', {}) or {})
        recursive_flatten_dict('animal_info', self.datareader.animal_info)
        recursive_flatten_dict('dataset_info', self.datareader.dataset_info if self.datareader.dataset_info else {})

        for signal_name, signal_info in self.datareader.signal_info.items():
            recursive_flatten_dict(f'signal_info_{signal_name}', signal_info)
            if 'metadata' in signal_info:
                recursive_flatten_dict(f'signal_info_{signal_name}_metadata', signal_info['metadata'])

        # Store the Dataset as a NetCDF file
        ds.to_netcdf(filepath)
        print(f"NetCDF file saved at {filepath}")

    def create_mne_raw_object(self, signal, selected_channels=None):
        """
        Creates an MNE Raw object from the data of a specific signal.

        Parameters:
        - signal: The signal name to include in the Raw object.
        - selected_channels: List of channels to include for the signal. If None, include all channels.

        Returns:
        - MNE Raw object containing the signal data.
        """
        signal_df = self.signal_data[signal]
        ch_names = self.signal_info[signal]['channels']
        
        # If no specific channels are selected, use all available channels for this signal
        if selected_channels is None:
            selected_channels = ch_names
        
        # Extract the relevant data for the selected channels
        selected_data = signal_df[selected_channels].values.T  # Transpose to match MNE shape requirements
        
        # Create MNE info dictionary
        info = mne.create_info(
            ch_names=selected_channels,
            sfreq=self.signal_info[signal]['sampling_frequency'],  # Assume uniform sampling frequency for the signal
            ch_types='misc'  # Adjust based on actual signal types if known
        )
        
        # Convert the start datetime string to a UTC datetime object
        start_datetime = self.signal_info[signal]['signal_start_datetime']
        if isinstance(start_datetime, pd.Timestamp):
            start_datetime_local = start_datetime.to_pydatetime()
            start_datetime_utc = start_datetime_local.astimezone(pytz.UTC)
        elif isinstance(start_datetime, str):
            start_datetime_local = pd.to_datetime(start_datetime)
            start_datetime_utc = start_datetime_local.tz_convert('UTC')
        else:
            raise ValueError(f"Unexpected format for signal_start_datetime: {start_datetime}")
        
        # Convert to (seconds, microseconds) tuple
        meas_date = (int(start_datetime_utc.timestamp()), int((start_datetime_utc.timestamp() % 1) * 1e6))

        # Set the measurement date using the converted UTC datetime
        raw = mne.io.RawArray(selected_data, info)
        raw.set_meas_date(meas_date)
        
        # Add custom metadata to the MNE info object
        for i, ch_name in enumerate(selected_channels):
            ch_metadata = self.signal_info[signal]['metadata'][ch_name]
            
            # Store original unit and other details in the channel description
            description = f"{ch_metadata['original_name']} ({ch_metadata['unit']})"
            info['chs'][i]['desc'] = description  # Use the description field for storing extra information

        # Concatenate other deployment data into a plaintext string
        deployment_info = "\n".join([f"{key}: {value}" for key, value in self.deployment_info.items()])
        info['description'] = f"Sensor: {signal}\nDeployment Data:\n{deployment_info}"
        
        return raw

    def export_to_edf(self, filename_template, selected_signals=None, selected_channels=None):
        """
        High-level method to export the current DataReader object's signals to separate EDF files.

        Parameters:
        - filename_template: A template string for the filename where '{signal}' will be replaced by the signal name.
        - selected_signals: List of signal names to export to EDF files. If None, include all signals.
        - selected_channels: Dictionary specifying which channels to include for each signal (e.g., {'accelerometer': ['ax', 'ay']}).
                             If None, include all channels for the selected signals.
        """
        # If no specific signals are selected, use all available signals
        if selected_signals is None:
            selected_signals = list(self.signal_data.keys())

        # Iterate through each signal and export to an EDF file
        for signal in selected_signals:
            if signal not in self.signal_data:
                print(f"Sensor {signal} not found in signal_data. Skipping.")
                continue
            
            # Determine which channels to include for the current signal
            if selected_channels and signal in selected_channels:
                channels_to_include = selected_channels[signal]
            else:
                channels_to_include = self.signal_info[signal]['channels']

            # Create the MNE Raw object for the current signal
            raw = self.create_mne_raw_object(signal, selected_channels=channels_to_include)

            # Define the EDF filename for the current signal, replacing '{signal}' in the template
            edf_filename = filename_template.format(signal=signal)

            # Save the Raw object as an EDF file
            raw.export(edf_filename, fmt='edf')
            
            print(f"EDF file for {signal} saved as {edf_filename}")
