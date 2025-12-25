#!/usr/bin/env python3
"""
Script to compare different EKF tuning parameters.

This script runs the EKF with different sigma2_pos values and generates
comparison plots to help you find the optimal tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from package.data_loader import load_gps_data
from package.extended_kalman_filter import extended_kalman_filter


def main():
    """Compare EKF with different tuning parameters."""

    # Load data
    print("Loading GPS data...")
    current_folder = Path(__file__).parent
    data_path = current_folder / "data" / "GPSdata.mat"
    gps_data, ref_data_struct = load_gps_data(str(data_path))

    # True trajectory
    x_t = ref_data_struct['traj_ned'][0, :]
    y_t = ref_data_struct['traj_ned'][1, :]
    z_t = ref_data_struct['traj_ned'][2, :]

    # Test different sigma2_pos values
    sigma2_values = [0.01, 0.1, 1.0, 5.0]
    colors = ['blue', 'green', 'orange', 'purple']

    results = []

    for sigma2_pos in sigma2_values:
        print(f"\nRunning EKF with sigma2_pos = {sigma2_pos}...")
        est = extended_kalman_filter(gps_data, ref_data_struct, sigma2_pos=sigma2_pos)

        x = est['x_h'][0, :]
        y = est['x_h'][2, :]
        z = est['x_h'][4, :]

        # Calculate errors
        error_x = x - x_t
        error_y = y - y_t
        error_z = z - z_t
        error_2d = np.sqrt(error_x**2 + error_y**2)
        error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)

        rms_2d = np.sqrt(np.mean(error_2d**2))
        rms_3d = np.sqrt(np.mean(error_3d**2))

        results.append({
            'sigma2_pos': sigma2_pos,
            'x': x,
            'y': y,
            'z': z,
            'rms_2d': rms_2d,
            'rms_3d': rms_3d,
            'max_2d': np.max(error_2d),
            'max_3d': np.max(error_3d)
        })

        print(f"  2D RMS Error: {rms_2d:.2f} m")
        print(f"  3D RMS Error: {rms_3d:.2f} m")

    # Create comparison plots
    plots_dir = current_folder / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 2D trajectory comparison
    plt.figure(figsize=(14, 10))
    plt.plot(x_t, y_t, 'r-.', linewidth=2, label='True', alpha=0.7)

    for i, result in enumerate(results):
        plt.plot(result['x'], result['y'],
                color=colors[i],
                alpha=0.6,
                label=f"σ²={result['sigma2_pos']} (RMS={result['rms_2d']:.2f}m)")

    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]", fontsize=12)
    plt.ylabel("y [m]", fontsize=12)
    plt.title("EKF Tuning Comparison - 2D Trajectory", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.savefig(plots_dir / "ekf_tuning_comparison_2D.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {plots_dir / 'ekf_tuning_comparison_2D.png'}")

    # Error statistics comparison
    plt.figure(figsize=(12, 6))

    sigma_labels = [f"{r['sigma2_pos']}" for r in results]
    rms_2d_values = [r['rms_2d'] for r in results]
    rms_3d_values = [r['rms_3d'] for r in results]

    x_pos = np.arange(len(sigma_labels))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(x_pos - width/2, rms_2d_values, width, label='2D RMS', color='skyblue')
    plt.bar(x_pos + width/2, rms_3d_values, width, label='3D RMS', color='lightcoral')
    plt.xlabel('σ² (Process Noise Variance)', fontsize=12)
    plt.ylabel('RMS Error [m]', fontsize=12)
    plt.title('RMS Error vs Process Noise', fontsize=14)
    plt.xticks(x_pos, sigma_labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.subplot(1, 2, 2)
    max_2d_values = [r['max_2d'] for r in results]
    max_3d_values = [r['max_3d'] for r in results]
    plt.bar(x_pos - width/2, max_2d_values, width, label='2D Max', color='skyblue')
    plt.bar(x_pos + width/2, max_3d_values, width, label='3D Max', color='lightcoral')
    plt.xlabel('σ² (Process Noise Variance)', fontsize=12)
    plt.ylabel('Max Error [m]', fontsize=12)
    plt.title('Max Error vs Process Noise', fontsize=14)
    plt.xticks(x_pos, sigma_labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plots_dir / "ekf_tuning_error_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved error comparison to {plots_dir / 'ekf_tuning_error_comparison.png'}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: EKF Tuning Parameter Comparison")
    print("="*70)
    print(f"{'σ²':<10} {'2D RMS [m]':<15} {'3D RMS [m]':<15} {'Max 2D [m]':<15} {'Max 3D [m]':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['sigma2_pos']:<10} {r['rms_2d']:<15.2f} {r['rms_3d']:<15.2f} {r['max_2d']:<15.2f} {r['max_3d']:<15.2f}")
    print("="*70)

    # Find best
    best_idx = np.argmin([r['rms_3d'] for r in results])
    print(f"\nBest σ² for lowest 3D RMS error: {results[best_idx]['sigma2_pos']}")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()
