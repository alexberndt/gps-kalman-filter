from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat


@dataclass
class SatelliteTimeSeries:
    """Time series data for a single satellite."""

    satellite_id: str
    positions_ned: np.ndarray  # Shape (3, N): satellite positions over time in NED coordinates
    pseudoranges: np.ndarray  # Shape (N,): pseudorange measurements over time

    @property
    def num_timesteps(self) -> int:
        return len(self.pseudoranges)

    def get_position_at(self, time_idx: int) -> np.ndarray:
        return self.positions_ned[:, time_idx]

    def get_pseudorange_at(self, time_idx: int) -> float:
        return self.pseudoranges[time_idx]

    def is_available_at(self, time_idx: int) -> bool:
        return not np.isnan(self.pseudoranges[time_idx])


@dataclass
class ClockParameters:
    """GPS receiver clock parameters."""

    initial_bias: float  # Initial clock bias (delta_t)
    initial_drift: float  # Initial clock drift rate (delta_t_dot)
    psd_phase: float  # Power spectral density of phase noise (S_phi)
    psd_frequency: float  # Power spectral density of frequency noise (S_f)

    @classmethod
    def from_arrays(cls, x_clk: np.ndarray, PSD_clk: np.ndarray) -> "ClockParameters":
        return cls(
            initial_bias=float(x_clk[0]),
            initial_drift=float(x_clk[1]),
            psd_phase=float(PSD_clk[0]),
            psd_frequency=float(PSD_clk[1]),
        )

    def to_psd_array(self) -> np.ndarray:
        return np.array([self.psd_phase, self.psd_frequency])

    def to_initial_state_array(self) -> np.ndarray:
        return np.array([self.initial_bias, self.initial_drift])


@dataclass
class MeasurementParameters:
    """Measurement noise and system parameters."""

    range_variance: float  # Variance of range measurement error (s2r)
    sample_time: float  # Sample time interval (Ts)
    speed_of_light: float  # Speed of light (c)

    @classmethod
    def from_dict(cls, ref_data: dict) -> "MeasurementParameters":
        return cls(
            range_variance=float(ref_data["s2r"]),
            sample_time=float(ref_data["Ts"]),
            speed_of_light=float(ref_data["c"]),
        )


@dataclass
class GroundTruth:
    """Ground truth trajectory data."""

    trajectory_ned: np.ndarray  # Shape (3, N): true trajectory in NED coordinates

    @property
    def num_timesteps(self) -> int:
        return self.trajectory_ned.shape[1]

    def get_position_at(self, time_idx: int) -> np.ndarray:
        return self.trajectory_ned[:, time_idx]


@dataclass
class GPSDataset:
    """Complete GPS dataset with all satellite and reference data."""

    satellites: list[SatelliteTimeSeries]
    clock_params: ClockParameters
    measurement_params: MeasurementParameters
    ground_truth: GroundTruth | None = None

    @property
    def num_satellites(self) -> int:
        return len(self.satellites)

    @property
    def num_timesteps(self) -> int:
        return self.satellites[0].num_timesteps if self.satellites else 0


def load_gps_data(mat_file_path) -> GPSDataset:
    """
    Load GPS data from MATLAB .mat file.

    Parameters
    ----------
    mat_file_path : str
        Path to the GPSdata.mat file

    Returns
    -------
    GPSDataset
        Structured GPS dataset object
    """
    mat_data = loadmat(mat_file_path)

    gps_data_mat = mat_data["gps_data"]

    satellites = []
    for i in range(gps_data_mat.shape[1]):
        satellite = SatelliteTimeSeries(
            satellite_id=gps_data_mat[0, i]["Satellite"][0],
            positions_ned=gps_data_mat[0, i]["Satellite_Position_NED"],
            pseudoranges=gps_data_mat[0, i]["PseudoRange"].flatten(),
        )
        satellites.append(satellite)

    ref_data_mat = mat_data["ref_data_struct"]

    measurement_params = MeasurementParameters(
        range_variance=float(ref_data_mat["s2r"][0, 0][0, 0]),
        sample_time=float(ref_data_mat["Ts"][0, 0][0, 0]),
        speed_of_light=float(ref_data_mat["c"][0, 0][0, 0]),
    )

    clock_params = ClockParameters.from_arrays(
        x_clk=ref_data_mat["x_clk"][0, 0][:, 0],
        PSD_clk=ref_data_mat["PSD_clk"][0, 0].flatten(),
    )

    ground_truth = GroundTruth(trajectory_ned=ref_data_mat["traj_ned"][0, 0])

    return GPSDataset(
        satellites=satellites,
        clock_params=clock_params,
        measurement_params=measurement_params,
        ground_truth=ground_truth,
    )
