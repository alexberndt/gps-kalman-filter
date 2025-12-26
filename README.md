# GPS Kalman Filter

Python implementation of Extended Kalman Filter (EKF) and Nonlinear Least Squares (NLS) for GPS position estimation using real satellite data.

## Installation

```bash
uv sync

uv sync --group dev
```

## Usage

```bash
uv run python -m gps.cli

uv run python -m gps.cli --filter ekf --sigma 5.0

uv run python -m gps.cli --plot

uv run python -m gps.cli --help
```

## Development

```bash
uv run pytest
```

## Results

With the provided dataset ($\sigma^2 = 5.0$ for EKF):

| Filter | 2D RMS Error | 3D RMS Error |
|--------|-------------|-------------|
| EKF | 3.49 m | 4.00 m |
| NLS | 4.47 m | 10.27 m |
