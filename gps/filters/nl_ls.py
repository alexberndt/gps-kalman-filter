"""
Nonlinear Least Squares estimator for GPS position.

Calculates single point position solution from GPS pseudorange measurements
using a nonlinear least squares approach.
"""

import numpy as np

from gps.data.loader import GPSDataset


def nonlinear_least_squares(dataset: GPSDataset) -> dict:
    """
    Calculate GPS position using nonlinear least squares.

    Parameters
    ----------
    dataset : GPSDataset
        GPS dataset containing satellite measurements and measurement parameters

    Returns
    -------
    dict
        Dictionary with fields:
        - 'x_h': ndarray of shape (4, N) with estimated [x, y, z, clock_offset] for each time
        - 'P': ndarray of shape (4, N) with diagonal elements of covariance matrix
    """
    N = dataset.num_timesteps
    M = dataset.num_satellites
    s2r = dataset.measurement_params.range_variance

    est = {"x_h": np.zeros((4, N)), "P": np.zeros((4, N))}

    for n in range(N):
        x = np.zeros(4)
        dx = np.inf * np.ones(4)
        res = np.zeros(M)
        H = np.zeros((M, 4))
        itr_ctr = 0

        # Nonlinear least squares iteration
        while np.linalg.norm(dx) > 0.01 and itr_ctr < 10:
            for m in range(M):
                sat = dataset.satellites[m]
                if sat.is_available_at(n):
                    sat_pos = sat.get_position_at(n)
                    pseudorange = sat.get_pseudorange_at(n)
                    dR_h = sat_pos - x[0:3]
                    res[m] = pseudorange - (np.linalg.norm(dR_h) + x[3])
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
        est["x_h"][:, n] = x
        est["P"][:, n] = s2r * np.diag(np.linalg.inv(H.T @ H))

    return est
