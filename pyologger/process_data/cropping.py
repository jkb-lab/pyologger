import pandas as pd
import pytz

def crop_data(data_pkl, imu_logger=None, ephys_logger=None, start_time=None, end_time=None):
    """
    Crop both the ephys and IMU datasets based on specified start and end times.

    Parameters
    ----------
    data_pkl : object
        The structured data object containing IMU and ephys data.
    imu_logger : str, optional
        The ID of the IMU logger. Default is None.
    ephys_logger : str, optional
        The ID of the ephys logger. Default is None.
    start_time : datetime, optional
        The start time for cropping the data. Default is None.
    end_time : datetime, optional
        The end time for cropping the data. Default is None.

    Returns
    -------
    data_pkl_crop : object
        A new structured data object containing the cropped datasets and metadata.
    """
    # Copy the original data_pkl to retain the structure
    data_pkl_crop = data_pkl

    # Get the time zone from the deployment
    time_zone_str = data_pkl.selected_deployment['Time Zone']
    time_zone = pytz.timezone(time_zone_str)

    # Localize start_time and end_time to the specified time zone if not already timezone-aware
    if start_time.tzinfo is None:
        start_time = time_zone.localize(start_time)
    if end_time.tzinfo is None:
        end_time = time_zone.localize(end_time)

    if imu_logger:
        imu_df = data_pkl.data[imu_logger]
        imu_fs = int(data_pkl.info[imu_logger]['datetime_metadata']['fs'])
        print(f"IMU Logger {imu_logger} sampled at: {imu_fs} Hz")

        # Crop the IMU data based on start and end time
        cropped_imu_df = imu_df[(imu_df['datetime'] >= start_time) & (imu_df['datetime'] <= end_time)]

        # Overwrite the data in data_pkl_crop
        data_pkl_crop.data[imu_logger] = cropped_imu_df

        # Update the metadata to reflect the original start and end times
        data_pkl_crop.info[imu_logger]['datetime_metadata']['original_start_time'] = imu_df['datetime'].min()
        data_pkl_crop.info[imu_logger]['datetime_metadata']['original_end_time'] = imu_df['datetime'].max()

    if ephys_logger:
        ephys_df = data_pkl.data[ephys_logger]
        ephys_fs = int(data_pkl.info[ephys_logger]['datetime_metadata']['fs'])
        print(f"ePhys Logger {ephys_logger} sampled at: {ephys_fs} Hz")

        # Crop the ephys data based on start and end time
        cropped_ephys_df = ephys_df[(ephys_df['datetime'] >= start_time) & (ephys_df['datetime'] <= end_time)]

        # Overwrite the data in data_pkl_crop
        data_pkl_crop.data[ephys_logger] = cropped_ephys_df

        # Update the metadata to reflect the original start and end times
        data_pkl_crop.info[ephys_logger]['datetime_metadata']['original_start_time'] = ephys_df['datetime'].min()
        data_pkl_crop.info[ephys_logger]['datetime_metadata']['original_end_time'] = ephys_df['datetime'].max()

    return data_pkl_crop