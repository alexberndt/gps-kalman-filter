"""
Extended Kalman Filter for GPS position estimation.

Calculates single point position solution from GPS pseudorange measurements
using an Extended Kalman Filter approach.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are

from gps.data.loader import GPSDataset
from gps.utils.measurement_functions import h_prime_func


@dataclass
class State:
    x: float = 0.0
    vx: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    z: float = 0.0
    delta_t: float = 0.0
    delta_t_dot: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.vx, self.y, self.vy, self.z, self.delta_t, self.delta_t_dot])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "State":
        return cls(
            x=vec[0],
            vx=vec[1],
            y=vec[2],
            vy=vec[3],
            z=vec[4],
            delta_t=vec[5],
            delta_t_dot=vec[6],
        )


@dataclass
class Position3D:
    x: float
    y: float
    z: float

    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "Position3D":
        return cls(x=vec[0], y=vec[1], z=vec[2])


@dataclass
class ClockState:
    delta_t: float
    delta_t_dot: float

    def to_vector(self) -> np.ndarray:
        return np.array([self.delta_t, self.delta_t_dot])


@dataclass
class ProcessNoise:
    sigma2_x: float
    sigma2_y: float
    sigma2_z: float
    Q_clk: np.ndarray

    def to_matrix(self) -> np.ndarray:
        return block_diag(self.sigma2_x, self.sigma2_y, self.sigma2_z, self.Q_clk)


@dataclass
class SystemMatrices:
    F: np.ndarray
    G: np.ndarray
    Q: np.ndarray

    @classmethod
    def create(cls, Ts: float, sigma2_pos: float, PSD_clk: np.ndarray) -> "SystemMatrices":
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

        # Clock process noise covariance
        S_phi, S_f = PSD_clk[0], PSD_clk[1]
        Q_clk = np.array([[S_phi * Ts + S_f * (Ts**3) / 3, Ts**2 * S_f], [Ts**2 * S_f, S_f * Ts]])

        # Process noise
        process_noise = ProcessNoise(sigma2_x=sigma2_pos, sigma2_y=sigma2_pos, sigma2_z=sigma2_pos, Q_clk=Q_clk)
        Q = process_noise.to_matrix()

        return cls(F=F, G=G, Q=Q)


@dataclass
class SatelliteMeasurement:
    satellite_id: str
    position: Position3D
    pseudorange: float
    available: bool

    @classmethod
    def from_satellite_timeseries(cls, satellite, time_idx: int) -> "SatelliteMeasurement":
        """Create measurement from SatelliteTimeSeries at specific time index."""
        pseudorange = satellite.get_pseudorange_at(time_idx)
        available = satellite.is_available_at(time_idx)

        if available:
            position = Position3D.from_vector(satellite.get_position_at(time_idx))
        else:
            position = Position3D(0, 0, 0)
            pseudorange = 0.0

        return cls(
            satellite_id=satellite.satellite_id,
            position=position,
            pseudorange=pseudorange,
            available=available,
        )


@dataclass
class LinearizedMeasurement:
    y_tilde: float
    H_row: np.ndarray
    y_actual: float
    h_nonlinear: float

    @classmethod
    def compute(cls, measurement: SatelliteMeasurement, state: State, speed_of_light: float) -> "LinearizedMeasurement":
        if not measurement.available:
            return cls(y_tilde=0.0, H_row=np.zeros(7), y_actual=0.0, h_nonlinear=0.0)

        state_vec = state.to_vector()
        p_i = measurement.position.to_vector()

        # Compute partial derivatives
        h_prime = np.array(
            [
                h_prime_func(p_i, state_vec, "x"),
                0,
                h_prime_func(p_i, state_vec, "y"),
                0,
                h_prime_func(p_i, state_vec, "z"),
                speed_of_light,
                0,
            ]
        )

        # Nonlinear measurement function (h_func internally computes distance + c*delta_t)
        x_rec = state_vec[0]
        y_rec = state_vec[2]
        z_rec = state_vec[4]
        delta_t = state_vec[5]

        x_i, y_i, z_i = p_i[0], p_i[1], p_i[2]
        h_nonlinear = np.sqrt((x_i - x_rec) ** 2 + (y_i - y_rec) ** 2 + (z_i - z_rec) ** 2) + speed_of_light * delta_t

        # Linearized measurement
        y_tilde = measurement.pseudorange - h_nonlinear + h_prime @ state_vec

        return cls(
            y_tilde=y_tilde,
            H_row=h_prime,
            y_actual=measurement.pseudorange,
            h_nonlinear=h_nonlinear,
        )


@dataclass
class MeasurementBatch:
    measurements: list[SatelliteMeasurement]
    linearized: list[LinearizedMeasurement]

    def get_available_indices(self) -> np.ndarray:
        return np.array([i for i, m in enumerate(self.measurements) if m.available])

    def build_measurement_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build H, y, and h_nonlinear vectors for available measurements."""
        idxs = self.get_available_indices()

        H_sub = np.array([self.linearized[i].H_row for i in idxs])
        y_vec_sub = np.array([self.linearized[i].y_actual for i in idxs])
        h_nonlinear_sub = np.array([self.linearized[i].h_nonlinear for i in idxs])

        return H_sub, y_vec_sub, h_nonlinear_sub, idxs


@dataclass
class FilterState:
    state: State
    P: np.ndarray

    @classmethod
    def initialize(cls, initial_clock: ClockState) -> "FilterState":
        state = State(delta_t=initial_clock.delta_t, delta_t_dot=initial_clock.delta_t_dot)
        return cls(state=state, P=None)


def extended_kalman_filter(dataset: "GPSDataset", sigma2_pos: float = 0.1) -> dict:
    """
    Calculate GPS position using Extended Kalman Filter.

    Parameters
    ----------
    dataset : GPSDataset
        GPS dataset containing satellite measurements, clock parameters, and measurement parameters
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
    N = dataset.num_timesteps
    M = dataset.num_satellites

    # Initialize system matrices
    system = SystemMatrices.create(
        Ts=dataset.measurement_params.sample_time,
        sigma2_pos=sigma2_pos,
        PSD_clk=dataset.clock_params.to_psd_array(),
    )

    # Initialize filter state
    initial_clock = ClockState(
        delta_t=dataset.clock_params.initial_bias,
        delta_t_dot=dataset.clock_params.initial_drift,
    )
    filter_state = FilterState.initialize(initial_clock)

    # Measurement covariance matrix
    R_k = dataset.measurement_params.range_variance * np.eye(M)

    # Storage for estimates
    est = {"x_h": np.zeros((7, N)), "P": np.zeros((7, 7, N))}

    for n in range(N):
        # Collect satellite measurements
        measurements = [SatelliteMeasurement.from_satellite_timeseries(sat, n) for sat in dataset.satellites]

        # Linearize measurements
        linearized = [
            LinearizedMeasurement.compute(meas, filter_state.state, dataset.measurement_params.speed_of_light)
            for meas in measurements
        ]

        # Create measurement batch
        batch = MeasurementBatch(measurements=measurements, linearized=linearized)

        # Build measurement matrices for available satellites
        H_sub, y_vec_sub, h_nonlinear_sub, idxs = batch.build_measurement_matrices()

        # Innovation (measurement residual)
        e_k = y_vec_sub - h_nonlinear_sub

        # Extract measurement covariance for available satellites
        R_k_sub = R_k[np.ix_(idxs, idxs)]

        # Initialize P using DARE if not yet initialized
        if filter_state.P is None:
            print("P_k is None - solving DARE")
            filter_state.P = solve_discrete_are(system.F.T, H_sub.T, system.G @ system.Q @ system.G.T, R_k_sub)

        # Innovation covariance
        R_ek = H_sub @ filter_state.P @ H_sub.T + R_k_sub

        # Kalman gain
        K_k = system.F @ filter_state.P @ H_sub.T @ np.linalg.inv(R_ek)

        # Covariance update
        P_kp1 = system.F @ filter_state.P @ system.F.T + system.G @ system.Q @ system.G.T - K_k @ R_ek @ K_k.T

        # State update
        state_vec = filter_state.state.to_vector()
        xhat_kp1_k = system.F @ state_vec + K_k @ e_k

        # Update filter state for next iteration
        filter_state.state = State.from_vector(xhat_kp1_k)
        filter_state.P = P_kp1

        # Store the estimate
        est["x_h"][:, n] = xhat_kp1_k
        est["P"][:, :, n] = P_kp1

    return est
