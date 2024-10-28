import numpy as np
from magnetic_field_calculator import MagneticFieldCalculator

def orientation_and_heading_correction(abar0, acc_data, mag_data, latitude, longitude, gyr_data=None):
    """
    Corrects the orientation and heading of tag data to align with the reference frame of an animal.

    Parameters
    ----------
    abar0 : array_like
        A 3-element vector representing the accelerometer readings when the animal is stationary on its belly.
    acc_data : array_like
        A 2D array where each row represents a 3-element accelerometer reading.
    mag_data : array_like
        A 2D array where each row represents a 3-element magnetometer reading.
    latitude : float
        Latitude of the location where the tag data was recorded.
    longitude : float
        Longitude of the location where the tag data was recorded.
    gyr_data : array_like, optional
        A 2D array where each row represents a 3-element gyroscope reading. If provided, gyroscope data will also be corrected.

    Returns
    -------
    pitch_deg : array_like
        The pitch angles in degrees for the entire dataset.
    roll_deg : array_like
        The roll angles in degrees for the entire dataset.
    heading_deg : array_like
        The heading angles in degrees for the entire dataset.
    acc_corr : array_like
        The corrected accelerometer data.
    mag_corr : array_like
        The corrected magnetometer data.
    gyr_corr : array_like, optional
        The corrected gyroscope data. Only returned if gyr_data is provided.

    Notes
    -----
    This function seeks to rotate tag data to the reference frame of an animal.
    The function first normalizes the provided stationary accelerometer vector, computes the pitch and roll angles,
    and then applies the corresponding rotation matrices to correct the input accelerometer, magnetometer, and optionally,
    gyroscope data. The function returns the corrected orientation in terms of pitch, roll, and heading angles.

    Examples
    --------
    >>> abar0 = np.array([0.1, 0.2, -0.98])
    >>> acc_data = np.random.rand(100, 3)
    >>> mag_data = np.random.rand(100, 3)
    >>> pitch_deg, roll_deg, heading_deg, acc_corr, mag_corr = orientation_and_heading_correction(abar0, acc_data, mag_data, 32.764567, -117.228665)
    >>> pitch_deg, roll_deg, heading_deg, acc_corr, mag_corr, gyr_corr = orientation_and_heading_correction(abar0, acc_data, mag_data, 32.764567, -117.228665, gyr_data=np.random.rand(100, 3))
    """
    # Normalize abar0 to create abar
    abar = abar0 / np.linalg.norm(abar0)
    
    # Calculate initial pitch (p0) and roll (r0)
    p0 = -np.arcsin(abar[0])
    r0 = np.arctan2(abar[1], abar[2])
    # Constrain p to [-pi / 2, pi / 2]
    if p0 > np.pi / 2:
        p0 = np.pi / 2 - p0
        r0 = r0 + np.pi

    # Define rotation matrices for pitch and roll
    def rotP(p):
        return np.array([[np.cos(p), 0, np.sin(p)],
                         [0, 1, 0],
                         [-np.sin(p), 0, np.cos(p)]])
    
    def rotR(r):
        return np.array([[1, 0, 0],
                         [0, np.cos(r), -np.sin(r)],
                         [0, np.sin(r), np.cos(r)]])
    
    # Calculate rotation matrix W
    W = np.matmul(rotP(p0), rotR(r0)).T

    # Correct the accelerometer and magnetometer data for the entire dataset
    acc_corr = np.matmul(acc_data, W)
    mag_corr = np.matmul(mag_data, W)
    
    # Correct the gyroscope data if provided
    if gyr_data is not None:
        gyr_corr = np.matmul(gyr_data, W)
    else:
        gyr_corr = None
    
    # Calculate magnitude of the corrected accelerometer vectors
    A = np.linalg.norm(acc_corr, axis=1)
    
    # Calculate pitch and roll in degrees from corrected accelerometer data
    pitch_deg = -np.degrees(np.arcsin(acc_corr[:, 0] / A))
    roll_deg = np.degrees(np.arctan2(acc_corr[:, 1], acc_corr[:, 2]))
    
    # Initialize an array to hold the gimbaled magnetic data
    mag_horiz = np.zeros_like(mag_corr)
    
    # Apply the un-roll and un-pitch rotation for each time step
    for i in range(len(pitch_deg)):
        mag_horiz[i, :] = np.matmul(np.matmul(mag_corr[i, :], rotR(np.deg2rad(roll_deg[i])).T), rotP(np.deg2rad(pitch_deg[i])).T)

    # Get the declination using MagneticFieldCalculator
    calculator = MagneticFieldCalculator()
    result = calculator.calculate(latitude=latitude, longitude=longitude)
    declination = result['field-value']['declination']

    print(f"The declination at latitude {latitude} and longitude {longitude} is {declination} degrees.")
    
    # Calculate heading in degrees from corrected magnetometer data
    heading_deg = np.degrees(np.arctan2(mag_horiz[:, 1], mag_horiz[:, 0])) + declination['value']
    
    # Return the corrected pitch, roll, and heading for the entire dataset
    if gyr_corr is not None:
        return pitch_deg, roll_deg, heading_deg, acc_corr, mag_corr, gyr_corr
    else:
        return pitch_deg, roll_deg, heading_deg, acc_corr, mag_corr