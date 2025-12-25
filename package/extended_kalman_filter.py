"""
Extended Kalman Filter for GPS position estimation.

Calculates single point position solution from GPS pseudorange measurements
using an Extended Kalman Filter approach.
"""

import numpy as np
from scipy.linalg import solve_discrete_are, block_diag
from .measurement_functions import h_func, h_prime_func


def extended_kalman_filter(gps_data, ref_data, sigma2_pos=0.1):
    """
    Calculate GPS position using Extended Kalman Filter.

    Parameters
    ----------
    gps_data : list of dict
        List of M satellite data dictionaries, each containing:
        - 'Satellite': Name of satellite
        - 'Satellite_Position_NED': ndarray of shape (3, N) with satellite positions
        - 'PseudoRange': ndarray of shape (N,) with pseudorange measurements
    ref_data : dict
        Reference data dictionary containing:
        - 's2r': Variance of range measurement error
        - 'PSD_clk': Power spectral density of clock [S_phi, S_f]
        - 'Ts': Sample time
        - 'c': Speed of light
        - 'x_clk': Initial clock state [delta_t, delta_t_dot]
        - 'traj_ned': True trajectory (optional, for comparison)
    sigma2_pos : float, optional
        Process noise variance for position states (x, y, z).
        Default: 0.1
        - Lower values (e.g., 0.01-0.1): Filter trusts model more, smoother but slower to adapt
        - Higher values (e.g., 1-10): Filter trusts measurements more, faster adaptation but noisier

    Returns
    -------
    dict
        Dictionary with fields:
        - 'x_h': ndarray of shape (7, N) with estimated state [x, vx, y, vy, z, delta_t, delta_t_dot]
        - 'P': ndarray of shape (7, 7, N) with covariance matrices
    """
    N = len(gps_data[0]['PseudoRange'])  # length of data
    M = len(gps_data)  # number of satellites (=30)

    est = {
        'x_h': np.zeros((7, N)),
        'P': np.zeros((7, 7, N))
    }

    # Get data from ref_data
    s2r = ref_data['s2r']
    PSD_clk = ref_data['PSD_clk']
    Ts = ref_data['Ts']

    # State transition matrix F
    F_x = np.array([[1, Ts], [0, 1]])
    F_y = np.array([[1, Ts], [0, 1]])
    F_z = np.array([[1]])
    F_clk = np.array([[1, Ts], [0, 1]])

    F = block_diag(F_x, F_y, F_z, F_clk)

    # Process noise input matrix G
    G_x = np.array([[Ts**2 / 2], [Ts]])
    G_y = np.array([[Ts**2 / 2], [Ts]])
    G_z = np.array([[Ts]])

    G = block_diag(G_x, G_y, G_z, np.eye(2))

    # Initial state estimate
    xhat_k_km1 = np.zeros(7)
    xhat_k_km1[5] = ref_data['x_clk'][0]
    xhat_k_km1[6] = ref_data['x_clk'][1]

    # Initial covariance (will be initialized via DARE)
    P_k = None

    # Clock process noise covariance
    S_phi = PSD_clk[0]
    S_f = PSD_clk[1]
    Q_k_clk = np.array([
        [S_phi * Ts + S_f * (Ts**3) / 3, Ts**2 * S_f],
        [Ts**2 * S_f, S_f * Ts]
    ])

    # Measurement covariance matrix
    R_k = s2r * np.eye(M)

    # Position process noise variances
    sigma2_x = sigma2_pos
    sigma2_y = sigma2_pos
    sigma2_z = sigma2_pos

    Q_k_x = sigma2_x
    Q_k_y = sigma2_y
    Q_k_z = sigma2_z

    Q_k = block_diag(Q_k_x, Q_k_y, Q_k_z, Q_k_clk)

    for n in range(N):

        # Get measurement from satellites
        y_i_tilde_vec = np.zeros(M)
        y_i_vec = np.zeros(M)
        h_nonlinear = np.zeros(M)
        H = np.zeros((M, 7))

        satellite_avail = np.zeros(M, dtype=bool)

        for i in range(M):
            # Check if satellite measurement i is available (is NOT NAN)
            if not np.isnan(gps_data[i]['PseudoRange'][n]):

                satellite_avail[i] = True

                # Position (x, y, z) of satellite i
                p_i = gps_data[i]['Satellite_Position_NED'][:, n]

                # Partial derivative elements of h'(x)
                h_p_1 = h_prime_func(p_i, xhat_k_km1, 'x')
                h_p_2 = 0
                h_p_3 = h_prime_func(p_i, xhat_k_km1, 'y')
                h_p_4 = 0
                h_p_5 = h_prime_func(p_i, xhat_k_km1, 'z')
                h_p_6 = ref_data['c']
                h_p_7 = 0

                # Partial derivative vector h'(x) evaluated at xhat_k_km1
                h_prime = np.array([h_p_1, h_p_2, h_p_3, h_p_4, h_p_5, h_p_6, h_p_7])

                # Measurement from satellite
                y_i = gps_data[i]['PseudoRange'][n]

                # LHS of linearization y^i - h(xh) + h'(xh)xh
                # where xh = \hat{x}
                y_i_tilde = y_i - h_func(p_i, xhat_k_km1, ref_data) + h_prime @ xhat_k_km1

                h_nonlinear[i] = h_func(p_i, xhat_k_km1, ref_data)

                y_i_vec[i] = y_i
                y_i_tilde_vec[i] = y_i_tilde
                H[i, :] = h_prime
            else:
                y_i_tilde_vec[i] = 0
                H[i, :] = np.zeros(7)

        # LINEAR KF
        idxs = np.where(satellite_avail)[0]

        H_sub = H[idxs, :]
        R_k_sub = R_k[np.ix_(idxs, idxs)]
        y_i_vec_sub = y_i_vec[idxs]
        h_nonlinear_sub = h_nonlinear[idxs]

        e_k = y_i_vec_sub - h_nonlinear_sub

        # Initialize P_k using DARE if not yet initialized
        if P_k is None:
            print("P_k is None - solving DARE")
            P_k = solve_discrete_are(F.T, H_sub.T, G @ Q_k @ G.T, R_k_sub)

        R_ek = H_sub @ P_k @ H_sub.T + R_k_sub

        # Kalman gain
        K_k = F @ P_k @ H_sub.T @ np.linalg.inv(R_ek)

        # Covariance update
        P_kp1 = F @ P_k @ F.T + G @ Q_k @ G.T - K_k @ R_ek @ K_k.T

        # State update
        xhat_kp1_k = F @ xhat_k_km1 + K_k @ e_k

        # Update time-index for next step
        xhat_k_km1 = xhat_kp1_k
        P_k = P_kp1

        # Store the estimate
        est['x_h'][:, n] = xhat_kp1_k
        est['P'][:, :, n] = P_kp1

    return est
