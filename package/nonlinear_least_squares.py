"""
Nonlinear Least Squares estimator for GPS position.

Calculates single point position solution from GPS pseudorange measurements
using a nonlinear least squares approach.
"""

import numpy as np


def nonlinear_least_squares(gps_data, s2r):
    """
    Calculate GPS position using nonlinear least squares.

    Parameters
    ----------
    gps_data : list of dict
        List of M satellite data dictionaries, each containing:
        - 'Satellite': Name of satellite
        - 'Satellite_Position_NED': ndarray of shape (3, N) with satellite positions
        - 'PseudoRange': ndarray of shape (N,) with pseudorange measurements
    s2r : float
        Variance of range measurement error

    Returns
    -------
    dict
        Dictionary with fields:
        - 'x_h': ndarray of shape (4, N) with estimated [x, y, z, clock_offset] for each time
        - 'P': ndarray of shape (4, N) with diagonal elements of covariance matrix
    """
    N = len(gps_data[0]['PseudoRange'])  # length of data
    M = len(gps_data)  # number of satellites (=30)

    est = {
        'x_h': np.zeros((4, N)),
        'P': np.zeros((4, N))
    }

    for n in range(N):
        x = np.zeros(4)
        dx = np.inf * np.ones(4)
        res = np.zeros(M)
        H = np.zeros((M, 4))
        itr_ctr = 0

        # Nonlinear least squares iteration
        while np.linalg.norm(dx) > 0.01 and itr_ctr < 10:

            for m in range(M):
                if not np.isnan(gps_data[m]['PseudoRange'][n]):
                    dR_h = gps_data[m]['Satellite_Position_NED'][:, n] - x[0:3]
                    res[m] = gps_data[m]['PseudoRange'][n] - (np.linalg.norm(dR_h) + x[3])
                    H[m, 0:3] = -dR_h / np.linalg.norm(dR_h)
                    H[m, 3] = 1
                else:
                    res[m] = 0
                    H[m, :] = np.zeros(4)

            # Calculate the correction to the state vector
            dx = np.linalg.solve(H.T @ H, H.T @ res)

            # Update the state vector
            x = x + dx

            # Update the iteration counter
            itr_ctr += 1

        # Store the estimate
        est['x_h'][:, n] = x
        est['P'][:, n] = s2r * np.diag(np.linalg.inv(H.T @ H))

    return est
