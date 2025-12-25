"""
Data loader utilities for GPS Kalman filter.

Handles loading and processing of MATLAB .mat data files.
"""

import numpy as np
from scipy.io import loadmat


def load_gps_data(mat_file_path):
    """
    Load GPS data from MATLAB .mat file.

    Parameters
    ----------
    mat_file_path : str
        Path to the GPSdata.mat file

    Returns
    -------
    tuple
        (gps_data, ref_data_struct) where:
        - gps_data is a list of M dictionaries with satellite data
        - ref_data_struct is a dictionary with reference data
    """
    # Load MATLAB file
    mat_data = loadmat(mat_file_path)

    # Extract GPS data
    # In MATLAB, gps_data is a structured array of size (1, M)
    gps_data_mat = mat_data['gps_data']

    # Convert to list of dictionaries
    gps_data = []
    for i in range(gps_data_mat.shape[1]):
        satellite_data = {
            'Satellite': gps_data_mat[0, i]['Satellite'][0],
            'Satellite_Position_NED': gps_data_mat[0, i]['Satellite_Position_NED'],
            'PseudoRange': gps_data_mat[0, i]['PseudoRange'].flatten()
        }
        gps_data.append(satellite_data)

    # Extract reference data
    ref_data_mat = mat_data['ref_data_struct']

    ref_data_struct = {
        's2r': float(ref_data_mat['s2r'][0, 0][0, 0]),
        'PSD_clk': ref_data_mat['PSD_clk'][0, 0].flatten(),
        'Ts': float(ref_data_mat['Ts'][0, 0][0, 0]),
        'c': float(ref_data_mat['c'][0, 0][0, 0]),
        'x_clk': ref_data_mat['x_clk'][0, 0][:, 0],
        'traj_ned': ref_data_mat['traj_ned'][0, 0]
    }

    return gps_data, ref_data_struct
