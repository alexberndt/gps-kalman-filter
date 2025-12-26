import numpy as np
from gps.filters.nl_ls import nonlinear_least_squares


def test_nls_output_structure(gps_dataset):
    result = nonlinear_least_squares(gps_dataset)

    assert "x_h" in result
    assert "P" in result


def test_nls_output_dimensions(gps_dataset):
    result = nonlinear_least_squares(gps_dataset)

    assert result["x_h"].shape == (4, gps_dataset.num_timesteps), (
        "State should be (4, N): [x, y, z, clock_offset]"
    )
    assert result["P"].shape == (4, gps_dataset.num_timesteps), (
        "Covariance diagonal should be (4, N)"
    )


def test_nls_produces_valid_estimates(gps_dataset):
    result = nonlinear_least_squares(gps_dataset)

    x_h = result["x_h"]

    assert not np.allclose(x_h, 0), "Expecting estimates to not all be zeros"
    assert np.all(np.isfinite(x_h)), "Expecting all estimates to be finite"


def test_nls_covariance_positive(gps_dataset):
    result = nonlinear_least_squares(gps_dataset)
    assert np.all(result["P"] > 0), "All covariance values should be positive"
