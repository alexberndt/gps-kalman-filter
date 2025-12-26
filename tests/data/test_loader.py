import numpy as np

from gps.data.loader import GPSDataset, load_gps_data


def test_load_gps_data_returns_dataset(data_path):
    """Test that load_gps_data returns a GPSDataset instance."""
    dataset = load_gps_data(str(data_path))
    assert isinstance(dataset, GPSDataset)


def test_dataset_has_expected_dimensions(gps_dataset):
    """Test that dataset has the expected number of satellites and timesteps."""
    assert gps_dataset.num_satellites == 30
    assert gps_dataset.num_timesteps == 2015


def test_satellite_data_structure(gps_dataset):
    """Test that satellite data has correct structure."""
    sat = gps_dataset.satellites[0]

    assert hasattr(sat, "satellite_id"), "Satellite should have satellite_id attribute"
    assert hasattr(sat, "positions_ned"), "Satellite should have positions_ned attribute"
    assert hasattr(sat, "pseudoranges"), "Satellite should have pseudoranges attribute"

    assert sat.positions_ned.shape == (3, 2015), "Position data should have shape (3, 2015)"
    assert sat.pseudoranges.shape == (2015,), "Pseudorange data should have shape (2015,)"


def test_satellite_access_methods(gps_dataset):
    """Test satellite data access methods."""
    sat = gps_dataset.satellites[0]

    pos = sat.get_position_at(0)
    assert isinstance(pos, np.ndarray), "Position should be a numpy array"
    assert pos.shape == (3,), "Position should have shape (3,)"

    pr = sat.get_pseudorange_at(0)
    assert isinstance(pr, (float, np.floating)), "Pseudorange should be a float"

    available = sat.is_available_at(0)
    assert isinstance(available, (bool, np.bool_)), "Availability should be a boolean"


def test_clock_parameters(gps_dataset):
    """Test clock parameters structure."""
    clock = gps_dataset.clock_params

    assert hasattr(clock, "initial_bias"), "Clock should have initial_bias attribute"
    assert hasattr(clock, "initial_drift"), "Clock should have initial_drift attribute"
    assert hasattr(clock, "psd_phase"), "Clock should have psd_phase attribute"
    assert hasattr(clock, "psd_frequency"), "Clock should have psd_frequency attribute"

    psd_array = clock.to_psd_array()
    assert psd_array.shape == (2,), "PSD array should have shape (2,)"

    state_array = clock.to_initial_state_array()
    assert state_array.shape == (2,), "Initial state array should have shape (2,)"


def test_measurement_parameters(gps_dataset):
    """Test measurement parameters structure."""
    params = gps_dataset.measurement_params

    assert hasattr(params, "range_variance"), "Params should have range_variance attribute"
    assert hasattr(params, "sample_time"), "Params should have sample_time attribute"
    assert hasattr(params, "speed_of_light"), "Params should have speed_of_light attribute"

    assert params.range_variance > 0, "Range variance should be positive"
    assert params.sample_time > 0, "Sample time should be positive"
    assert params.speed_of_light > 0, "Speed of light should be positive"


def test_ground_truth(gps_dataset):
    """Test ground truth data."""
    assert gps_dataset.ground_truth is not None, "Ground truth should exist"

    gt = gps_dataset.ground_truth
    assert gt.trajectory_ned.shape == (3, 2015), "Ground truth trajectory should have shape (3, 2015)"

    pos = gt.get_position_at(0)
    assert pos.shape == (3,), "Ground truth position should have shape (3,)"


def test_nan_handling(gps_dataset):
    """Test that NaN values in pseudoranges are handled correctly."""
    for sat in gps_dataset.satellites:
        nan_indices = np.where(np.isnan(sat.pseudoranges))[0]
        if len(nan_indices) > 0:
            idx = nan_indices[0]
            assert not sat.is_available_at(idx), "is_available_at should return False for NaN values"
            break
