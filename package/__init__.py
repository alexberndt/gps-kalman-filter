"""
GPS Kalman Filter Package

A Python implementation of Kalman filter design for GPS receiver.

Authors: Alexander Berndt, Rebecka Winqvist
Python conversion: 2025
"""

from .measurement_functions import h_func, h_prime_func
from .nonlinear_least_squares import nonlinear_least_squares
from .extended_kalman_filter import (
    extended_kalman_filter,
    State,
    Position3D,
    ClockState,
    ProcessNoise,
    SystemMatrices,
    SatelliteMeasurement,
    LinearizedMeasurement,
    MeasurementBatch,
    FilterState,
)
from .data_loader import (
    load_gps_data,
    GPSDataset,
    SatelliteTimeSeries,
    ClockParameters,
    MeasurementParameters,
    GroundTruth,
)

__all__ = [
    # Measurement functions
    'h_func',
    'h_prime_func',
    # Estimators
    'nonlinear_least_squares',
    'extended_kalman_filter',
    # EKF data structures
    'State',
    'Position3D',
    'ClockState',
    'ProcessNoise',
    'SystemMatrices',
    'SatelliteMeasurement',
    'LinearizedMeasurement',
    'MeasurementBatch',
    'FilterState',
    # Data loader
    'load_gps_data',
    'GPSDataset',
    'SatelliteTimeSeries',
    'ClockParameters',
    'MeasurementParameters',
    'GroundTruth',
]
