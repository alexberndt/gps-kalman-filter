#!/usr/bin/env python3
"""
Test script to verify the structured API works correctly.
"""

from pathlib import Path
from package.data_loader import load_gps_data
from package.extended_kalman_filter import extended_kalman_filter
from package.nonlinear_least_squares import nonlinear_least_squares


def test_data_loading():
    """Test that data loads correctly with structured format."""
    print("Testing data loading...")
    data_path = Path(__file__).parent / "data" / "GPSdata.mat"
    dataset = load_gps_data(str(data_path))

    # Check basic properties
    assert dataset.num_satellites == 30, f"Expected 30 satellites, got {dataset.num_satellites}"
    assert dataset.num_timesteps == 2015, f"Expected 2015 timesteps, got {dataset.num_timesteps}"

    # Check clock parameters
    assert hasattr(dataset.clock_params, 'initial_bias')
    assert hasattr(dataset.clock_params, 'initial_drift')
    assert hasattr(dataset.clock_params, 'psd_phase')
    assert hasattr(dataset.clock_params, 'psd_frequency')

    # Check measurement parameters
    assert hasattr(dataset.measurement_params, 'range_variance')
    assert hasattr(dataset.measurement_params, 'sample_time')
    assert hasattr(dataset.measurement_params, 'speed_of_light')

    # Check ground truth
    assert dataset.ground_truth is not None
    assert dataset.ground_truth.num_timesteps == 2015

    print("✓ Data loading test passed")
    return dataset


def test_satellite_access(dataset):
    """Test satellite data access methods."""
    print("\nTesting satellite data access...")

    # Get first satellite
    sat = dataset.satellites[0]

    # Test methods
    pos = sat.get_position_at(0)
    assert pos.shape == (3,), f"Expected position shape (3,), got {pos.shape}"

    pseudorange = sat.get_pseudorange_at(0)
    assert isinstance(pseudorange, (float, int)), f"Expected float/int, got {type(pseudorange)}"

    is_available = sat.is_available_at(0)
    assert isinstance(is_available, bool), f"Expected bool, got {type(is_available)}"

    print("✓ Satellite access test passed")


def test_nonlinear_least_squares(dataset):
    """Test NLS estimator with structured data."""
    print("\nTesting Nonlinear Least Squares...")

    result = nonlinear_least_squares(dataset)

    # Check output structure
    assert 'x_h' in result
    assert 'P' in result

    # Check dimensions
    assert result['x_h'].shape == (4, dataset.num_timesteps)
    assert result['P'].shape == (4, dataset.num_timesteps)

    print(f"✓ NLS test passed - Output shape: {result['x_h'].shape}")


def test_extended_kalman_filter(dataset):
    """Test EKF with structured data."""
    print("\nTesting Extended Kalman Filter...")

    result = extended_kalman_filter(dataset, sigma2_pos=5.0)

    # Check output structure
    assert 'x_h' in result
    assert 'P' in result

    # Check dimensions
    assert result['x_h'].shape == (7, dataset.num_timesteps)
    assert result['P'].shape == (7, 7, dataset.num_timesteps)

    print(f"✓ EKF test passed - Output shape: {result['x_h'].shape}")


def main():
    """Run all tests."""
    print("="*60)
    print("Running Structured API Tests")
    print("="*60)

    dataset = test_data_loading()
    test_satellite_access(dataset)
    test_nonlinear_least_squares(dataset)
    test_extended_kalman_filter(dataset)

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    main()
