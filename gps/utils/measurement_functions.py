"""
Measurement functions for GPS Kalman filter.

Contains the nonlinear measurement function h(x) and its Jacobian h'(x).
"""

import numpy as np


def h_func(p_i, xhat_k_km1, ref_data):
    """
    Nonlinear measurement function for GPS pseudorange.

    Computes the expected pseudorange from receiver to satellite i.

    Parameters
    ----------
    p_i : ndarray, shape (3,)
        Position of satellite i in NED coordinates [x_i, y_i, z_i]
    xhat_k_km1 : ndarray, shape (7,)
        State estimate vector [x, vx, y, vy, z, delta_t, delta_t_dot]
        where (x, y, z) is receiver position, delta_t is clock bias
    ref_data : dict
        Reference data containing 'c' (speed of light)

    Returns
    -------
    float
        Expected pseudorange measurement
    """
    x_rec = xhat_k_km1[0]
    y_rec = xhat_k_km1[2]
    z_rec = xhat_k_km1[4]
    delta_t = xhat_k_km1[5]

    x_i = p_i[0]
    y_i = p_i[1]
    z_i = p_i[2]

    val = np.sqrt((x_i - x_rec)**2 + (y_i - y_rec)**2 + (z_i - z_rec)**2) + ref_data['c'] * delta_t

    return val


def h_prime_func(p_i, xhat_k_km1, var):
    """
    Partial derivative of measurement function h with respect to position variable.

    Computes ∂h/∂var where var is 'x', 'y', or 'z'.

    Parameters
    ----------
    p_i : ndarray, shape (3,)
        Position of satellite i in NED coordinates [x_i, y_i, z_i]
    xhat_k_km1 : ndarray, shape (7,)
        State estimate vector [x, vx, y, vy, z, delta_t, delta_t_dot]
    var : str
        Variable for partial derivative: 'x', 'y', or 'z'

    Returns
    -------
    float
        Partial derivative ∂h/∂var
    """
    x_rec = xhat_k_km1[0]
    y_rec = xhat_k_km1[2]
    z_rec = xhat_k_km1[4]

    x_i = p_i[0]
    y_i = p_i[1]
    z_i = p_i[2]

    h_den = np.sqrt((x_i - x_rec)**2 + (y_i - y_rec)**2 + (z_i - z_rec)**2)

    if var == 'x':
        h_num = x_rec - x_i
    elif var == 'y':
        h_num = y_rec - y_i
    else:  # var == 'z'
        h_num = z_rec - z_i

    h_prime_ = h_num / h_den

    return h_prime_
