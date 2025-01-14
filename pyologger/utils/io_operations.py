import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, date, time

def save_to_netcdf(data_reader, filepath):
        """Saves the current state of the DataReader object to a NetCDF file."""
        def convert_to_compatible_array(df):
            """Convert DataFrame columns to compatible numpy arrays."""
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Handle datetime objects by converting them to strings
                    if isinstance(df[col].iloc[0], (datetime, date, time)):
                        df[col] = df[col].astype(str)
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col])
                    else:
                        # Attempt to convert to float, if fails convert to string
                        try:
                            df[col] = df[col].astype(float)
                        except ValueError:
                            df[col] = df[col].astype(str)
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col])

            # Check the number of columns in the DataFrame
            if df.shape[1] == 1:
                # If there is only one column, return a flat array
                return df.iloc[:, 0].to_numpy()
            else:
                # If there are multiple columns, return nested arrays
                return df.apply(pd.to_numeric, errors='ignore').to_numpy()

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
                coords = {f"{name}_samples": datetime_coord, f"{name}_variables": variables}

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

        # Flatten the dictionaries into xarray DataArrays
        for sensor_name, df in self.sensor_data.items():
            sensor_data = df.copy()
            # Saving datetime as timezone-aware
            datetime_coord = pd.to_datetime(sensor_data['datetime'])
            sensor_data = sensor_data.drop(columns=['datetime'])
            variables = [col for col in sensor_data.columns]
            data_array = convert_to_compatible_array(sensor_data)
            var_name = f'sensor_data_{sensor_name}'
            ndim = data_array.ndim
            dims, coords = create_coords(ndim, datetime_coord, variables, sensor_name)
            ds[var_name] = create_data_array(data_array, dims, coords)
            set_variables_attr(ds, var_name, variables)

        for logger_id, df in self.logger_data.items():
            # Remove specified columns
            logger_data = df.copy()
            columns_to_remove = ['date_utc', 'time_utc', 'date', 'time']
            logger_data = logger_data.drop(columns=[col for col in columns_to_remove if col in logger_data.columns])
            # Saving datetime as timezone-aware
            datetime_coord = pd.to_datetime(logger_data['datetime'])
            logger_data = logger_data.drop(columns=['datetime'])
            # Remove string type columns
            logger_data = logger_data.select_dtypes(exclude=['object'])
            variables = [col for col in logger_data.columns]
            data_array = convert_to_compatible_array(logger_data)
            var_name = f'logger_data_{logger_id}'
            ndim = data_array.ndim
            dims, coords = create_coords(ndim, datetime_coord, variables, logger_id)
            ds[var_name] = create_data_array(data_array, dims, coords)
            set_variables_attr(ds, var_name, variables)

        for derived_name, df in self.derived_data.items():
            derived_data = df.copy()
            # Saving datetime as timezone-aware
            datetime_coord = pd.to_datetime(derived_data['datetime'])
            derived_data = derived_data.drop(columns=['datetime'])
            variables = [col for col in derived_data.columns]
            data_array = convert_to_compatible_array(derived_data)
            var_name = f'derived_data_{derived_name}'
            ndim = data_array.ndim
            dims, coords = create_coords(ndim, datetime_coord, variables, derived_name)
            ds[var_name] = create_data_array(data_array, dims, coords)
            set_variables_attr(ds, var_name, variables)

        columns_to_keep = ["type", "key", "value", "duration", "short_description", "long_description"]
        
        if isinstance(self.event_data, pd.DataFrame):
            for var in columns_to_keep:
                event_data = self.event_data.copy()
                datetime_coord = pd.to_datetime(event_data['datetime'])
                event_data = event_data[[var]]
                data_array = convert_to_compatible_array(event_data)
                var_name = f'event_data_{var}'
                ndim = data_array.ndim
                if var == columns_to_keep[0]:
                    dims, coords = create_coords(ndim, datetime_coord, variables, 'event_data')
                ds[var_name] = create_data_array(data_array, dims, coords)
                set_variables_attr(ds, var_name, var)

        # Flatten and add global attributes
        flatten_dict('deployment_info', self.deployment_info)
        flatten_dict('files_info', self.files_info)
        flatten_dict('animal_info', self.animal_info)
        flatten_dict('dataset_info', self.dataset_info[0] if self.dataset_info else {})

        for logger_id, logger_info in self.logger_info.items():
            flatten_dict(f'logger_info_{logger_id}', logger_info)

        for sensor_name, sensor_info in self.sensor_info.items():
            flatten_dict(f'sensor_info_{sensor_name}', sensor_info)

        # Store the Dataset as a NetCDF file
        ds.to_netcdf(filepath)
        print(f"NetCDF file saved at {filepath}")