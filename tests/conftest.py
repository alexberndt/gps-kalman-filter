"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from gps_kalman_filter.data.loader import load_gps_data


@pytest.fixture(scope="session")
def data_path():
    """Path to test data file."""
    return Path(__file__).parent.parent / "data" / "GPSdata.mat"


@pytest.fixture(scope="session")
def gps_dataset(data_path):
    """Load GPS dataset for tests."""
    return load_gps_data(str(data_path))


@pytest.fixture
def small_sigma():
    """Small process noise variance for testing."""
    return 0.1


@pytest.fixture
def large_sigma():
    """Large process noise variance for testing."""
    return 5.0
