# GPS Kalman Filter

A Python implementation of Kalman filtering techniques for GPS position estimation using real GPS data with clock offsets.

## Features

- **Multiple Filtering Algorithms**:
  - Extended Kalman Filter (EKF) with velocity estimation
  - Nonlinear Least Squares (NLS) estimator

- **Structured Data Types**: Type-safe dataclasses for all GPS data structures
- **Comprehensive Testing**: Full test suite with pytest
- **Optional Visualization**: Matplotlib-based plotting (optional dependency)
- **CLI Interface**: Easy-to-use command-line interface

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd gps-kalman-filter

# Install the package
uv pip install -e .

# For development with plotting support
uv pip install -e '.[dev]'
```

## Usage

### Command Line

```bash
# Run both filters without plots (fast)
uv run python main.py

# Run with plots
uv run python main.py --plot

# Run only EKF with custom sigma
uv run python main.py --filter ekf --sigma 5.0

# Save plots without showing
uv run python main.py --save-plots ./my_plots

# Get help
uv run python main.py --help
```

### Python API

```python
from gps_kalman_filter import load_gps_data, extended_kalman_filter, nonlinear_least_squares

# Load data
dataset = load_gps_data("data/GPSdata.mat")

# Run Extended Kalman Filter
ekf_result = extended_kalman_filter(dataset, sigma2_pos=5.0)

# Run Nonlinear Least Squares
nls_result = nonlinear_least_squares(dataset)

# Access results
positions_ekf = ekf_result['x_h']  # Shape: (7, N) - [x, vx, y, vy, z, delta_t, delta_t_dot]
covariance_ekf = ekf_result['P']    # Shape: (7, 7, N)

positions_nls = nls_result['x_h']   # Shape: (4, N) - [x, y, z, clock_offset]
covariance_nls = nls_result['P']    # Shape: (4, N)
```

## Package Structure

```
gps-kalman-filter/
├── gps_kalman_filter/          # Main package
│   ├── data/                   # Data loading utilities
│   │   ├── loader.py           # GPS dataset loader
│   │   └── __init__.py
│   ├── filters/                # Filtering algorithms
│   │   ├── extended_kalman.py  # Extended Kalman Filter
│   │   ├── nonlinear_least_squares.py  # NLS estimator
│   │   └── __init__.py
│   ├── utils/                  # Utility functions
│   │   ├── measurement_functions.py
│   │   └── __init__.py
│   ├── cli.py                  # Command-line interface
│   └── __init__.py
├── tests/                      # Test suite
│   ├── test_data_loader.py
│   ├── test_filters.py
│   ├── conftest.py
│   └── __init__.py
├── data/                       # GPS data files
├── pyproject.toml              # Package configuration
└── README.md
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_filters.py -v

# Run with coverage
uv run pytest tests/ --cov=gps_kalman_filter
```

## Data Structures

### GPSDataset
Complete GPS dataset with satellite measurements, clock parameters, and ground truth.

```python
dataset.num_satellites     # Number of satellites
dataset.num_timesteps      # Number of time steps
dataset.satellites         # List of SatelliteTimeSeries
dataset.clock_params       # ClockParameters
dataset.measurement_params # MeasurementParameters
dataset.ground_truth       # GroundTruth
```

### Satellite Access
```python
sat = dataset.satellites[0]
pos = sat.get_position_at(time_idx)      # Get position at time index
pr = sat.get_pseudorange_at(time_idx)    # Get pseudorange
available = sat.is_available_at(time_idx) # Check if measurement is valid
```

## Algorithm Parameters

### Extended Kalman Filter

The EKF includes a tunable parameter `sigma2_pos` that controls the process noise variance:

- **Lower values (0.01-0.1)**: Smoother trajectory, trusts model more, slower adaptation
- **Higher values (1-10)**: Faster adaptation, trusts measurements more, noisier estimates
- **Recommended**: 5.0 (achieves ~3.5m RMS error for this dataset)

```bash
# Try different sigma values
uv run python main.py --filter ekf --sigma 0.1  # Smooth
uv run python main.py --filter ekf --sigma 5.0  # Balanced (recommended)
uv run python main.py --filter ekf --sigma 10.0 # Responsive
```

## Results

With the provided dataset and optimal parameters:

| Filter | 2D RMS Error | 3D RMS Error | Max Error |
|--------|-------------|-------------|-----------|
| EKF (σ²=5.0) | 3.49 m | 4.00 m | 9.25 m |
| NLS | 4.47 m | 10.27 m | 29.51 m |

## Authors

- Alexander Berndt
- Rebecka Winqvist

Python conversion: 2025

## License

FEM3200 - Optimal Filtering Project
