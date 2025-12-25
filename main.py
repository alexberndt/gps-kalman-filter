#!/usr/bin/env python3
"""
FEM3200 - Optimal Filtering

Project: Kalman Filter design for GPS receiver

Authors: Alexander Berndt, Rebecka Winqvist
Python conversion: 2025

This script demonstrates GPS position estimation using:
1. Nonlinear Least Squares (commented out)
2. Extended Kalman Filter
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from package.data_loader import load_gps_data
from package.nonlinear_least_squares import nonlinear_least_squares
from package.extended_kalman_filter import extended_kalman_filter


def main():
    """Main script for GPS Kalman filter demonstration."""

    # Load data
    print("Loading GPS data...")
    current_folder = Path(__file__).parent
    data_path = current_folder / "data" / "GPSdata.mat"

    gps_data, ref_data_struct = load_gps_data(str(data_path))
    print(f"Loaded data with {len(gps_data)} satellites and {len(gps_data[0]['PseudoRange'])} time steps")

    # ========================================================================
    # Nonlinear Least Squares Algorithm
    # ========================================================================
    print("\nRunning Nonlinear Least Squares estimator...")
    est_nls = nonlinear_least_squares(gps_data, ref_data_struct['s2r'])
    x_h_nls = est_nls['x_h']

    x_nls = x_h_nls[0, :]
    y_nls = x_h_nls[1, :]
    z_nls = x_h_nls[2, :]

    # True trajectory
    x_t = ref_data_struct['traj_ned'][0, :]
    y_t = ref_data_struct['traj_ned'][1, :]
    z_t = ref_data_struct['traj_ned'][2, :]

    # Create plots directory
    plots_dir = current_folder / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot NLS 2D trajectory
    plt.figure(1, figsize=(10, 8))
    plt.plot(x_nls, y_nls, label='NL Estimator')
    plt.plot(x_t, y_t, 'r-.', label='True')
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("NL Estimator")
    plt.legend()
    plt.savefig(plots_dir / "nl_estimator_traj_2D.png", dpi=300, bbox_inches='tight')
    print(f"Saved NLS 2D trajectory plot to {plots_dir / 'nl_estimator_traj_2D.png'}")

    # Plot NLS 3D trajectory
    fig = plt.figure(2, figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_nls, y_nls, z_nls, label='NL Estimator')
    ax.plot(x_t, y_t, z_t, 'r-.', label='True')
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("NL Estimator")
    ax.set_zlim([-100, 100])
    ax.legend()
    plt.savefig(plots_dir / "nl_estimator_traj_3D.png", dpi=300, bbox_inches='tight')
    print(f"Saved NLS 3D trajectory plot to {plots_dir / 'nl_estimator_traj_3D.png'}")

    # ========================================================================
    # Extended Kalman Filter
    # ========================================================================
    print("\nRunning Extended Kalman Filter...")

    # EKF Tuning Parameter
    # sigma2_pos: Process noise variance for position (x, y, z)
    #   - Lower (0.01-0.1): Smoother trajectory, trusts model more
    #   - Higher (1-10): Faster adaptation, trusts measurements more
    #   - Optimal for this dataset: 5.0 (gives ~3.5m RMS error vs 10m with 0.1)
    sigma2_pos = 5.0  # Tuned value - run tune_ekf.py to experiment with different values

    est = extended_kalman_filter(gps_data, ref_data_struct, sigma2_pos=sigma2_pos)
    print(f"EKF sigma2_pos: {sigma2_pos}")

    x_h = est['x_h']
    P = est['P']

    # Extract coordinate values
    x = x_h[0, :]  # x position
    y = x_h[2, :]  # y position
    z = x_h[4, :]  # z position

    # Plot x-y 2D trajectory
    print("\nGenerating EKF plots...")
    plt.figure(3, figsize=(10, 8))
    plt.clf()
    plt.plot(x, y, label='EKF')
    plt.plot(x_t, y_t, 'r-.', label='True')
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("EKF Position Estimation")
    plt.legend()
    plt.savefig(plots_dir / "ekf_traj_2D.png", dpi=300, bbox_inches='tight')
    print(f"Saved EKF 2D trajectory plot to {plots_dir / 'ekf_traj_2D.png'}")

    # Plot x-y-z 3D trajectory
    fig = plt.figure(4, figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='EKF')
    ax.plot(x_t, y_t, z_t, 'r-.', label='True')
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("EKF Position Estimation")
    ax.set_zlim([-100, 100])
    ax.legend()
    plt.savefig(plots_dir / "ekf_traj_3D.png", dpi=300, bbox_inches='tight')
    print(f"Saved EKF 3D trajectory plot to {plots_dir / 'ekf_traj_3D.png'}")

    # Calculate error statistics
    error_x = x - x_t
    error_y = y - y_t
    error_z = z - z_t
    error_2d = np.sqrt(error_x**2 + error_y**2)
    error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)

    print("\n" + "="*60)
    print("Error Statistics")
    print("="*60)
    print(f"2D RMS Error: {np.sqrt(np.mean(error_2d**2)):.2f} m")
    print(f"3D RMS Error: {np.sqrt(np.mean(error_3d**2)):.2f} m")
    print(f"Max 2D Error: {np.max(error_2d):.2f} m")
    print(f"Max 3D Error: {np.max(error_3d):.2f} m")
    print("="*60)

    plt.show()


if __name__ == "__main__":
    main()
