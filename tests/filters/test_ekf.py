import numpy as np
from gps.filters.ekf import extended_kalman_filter


def test_ekf_output_structure(gps_dataset, small_sigma):
    result = extended_kalman_filter(gps_dataset, sigma2_pos=small_sigma)

    assert "x_h" in result
    assert "P" in result


def test_ekf_output_dimensions(gps_dataset, small_sigma):
    result = extended_kalman_filter(gps_dataset, sigma2_pos=small_sigma)

    assert result["x_h"].shape == (7, gps_dataset.num_timesteps), (
        "State should be (7, N): [x, vx, y, vy, z, delta_t, delta_t_dot]"
    )
    assert result["P"].shape == (7, 7, gps_dataset.num_timesteps), (
        "Covariance should be (7, 7, N)"
    )


def test_ekf_produces_valid_estimates(gps_dataset, small_sigma):
    result = extended_kalman_filter(gps_dataset, sigma2_pos=small_sigma)

    x_h = result["x_h"]

    assert not np.allclose(x_h, 0), "Expecting estimates to not all be zeros"
    assert np.all(np.isfinite(x_h)), "Expecting all estimates to be finite"


def test_ekf_covariance_positive_semidefinite(gps_dataset, small_sigma):
    result = extended_kalman_filter(gps_dataset, sigma2_pos=small_sigma)

    P = result["P"]

    for i in [0, len(P[0, 0]) // 2, len(P[0, 0]) - 1]:
        P_i = P[:, :, i]
        eigenvalues = np.linalg.eigvals(P_i)
        assert np.all(eigenvalues >= -1e-10), (
            f"Covariance has negative eigenvalues at timestep {i}"
        )


def test_ekf_sigma_parameter_effect(gps_dataset, small_sigma, large_sigma):
    result_small = extended_kalman_filter(gps_dataset, sigma2_pos=small_sigma)
    result_large = extended_kalman_filter(gps_dataset, sigma2_pos=large_sigma)

    assert not np.allclose(result_small["x_h"], result_large["x_h"]), (
        "Results should be different for different sigma values"
    )


def test_ekf_velocity_estimates(gps_dataset, large_sigma):
    result = extended_kalman_filter(gps_dataset, sigma2_pos=large_sigma)

    x_h = result["x_h"]

    vx = x_h[1, :]
    vy = x_h[3, :]

    assert not np.allclose(vx, 0), "Expecting x-velocities to not all be zero"
    assert not np.allclose(vy, 0), "Expecting y-velocities to not all be zero"
    assert np.all(np.isfinite(vx)), "Expecting all x-velocities to be finite"
    assert np.all(np.isfinite(vy)), "Expecting all y-velocities to be finite"
