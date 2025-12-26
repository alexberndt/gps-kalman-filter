# GPS Kalman Filter

Python implementation of Extended Kalman Filter (EKF) and Nonlinear Least Squares (NLS) for GPS position estimation using real satellite data.

## Installation

```bash
# Install dependencies
uv sync

# For development with plotting and testing
uv sync --group dev
```

## Usage

```bash
# Run both filters (EKF and NLS)
uv run python -m gps.cli

# Run only EKF with custom sigma
uv run python -m gps.cli --filter ekf --sigma 5.0

# Run with plots (requires dev dependencies)
uv run python -m gps.cli --plot

# Get help
uv run python -m gps.cli --help
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=gps
```

## Results

With the provided dataset (σ²=5.0 for EKF):

| Filter | 2D RMS Error | 3D RMS Error |
|--------|-------------|-------------|
| EKF | 3.49 m | 4.00 m |
| NLS | 4.47 m | 10.27 m |

## License

FEM3200 - Optimal Filtering Project
