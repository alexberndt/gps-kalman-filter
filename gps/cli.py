import argparse
import sys
from pathlib import Path
import numpy as np

from gps.data.loader import load_gps_data
from gps.filters.ekf import extended_kalman_filter
from gps.filters.nl_ls import nonlinear_least_squares


def main():
    parser = argparse.ArgumentParser(
        description="GPS position estimation using Kalman filtering"
    )
    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to GPSdata.mat file (default: ./data/GPSdata.mat)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate and display plots"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="EKF process noise variance (default: 5.0)",
    )
    parser.add_argument(
        "--filter",
        choices=["nls", "ekf", "both"],
        default="both",
        help="Which filter(s) to run (default: both)",
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        default=None,
        help="Directory to save plots (default: ./plots)",
    )

    args = parser.parse_args()

    # Determine data path
    if args.data_path is None:
        data_path = Path("data") / "GPSdata.mat"
    else:
        data_path = Path(args.data_path)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        return 1

    # Load data
    print(f"Loading GPS data from {data_path}...")
    dataset = load_gps_data(str(data_path))
    print(
        f"Loaded {dataset.num_satellites} satellites, {dataset.num_timesteps} timesteps"
    )

    # Run filters
    results = {}

    if args.filter in ["nls", "both"]:
        print("\nRunning Nonlinear Least Squares...")
        results["nls"] = nonlinear_least_squares(dataset)
        print_error_statistics("NLS", results["nls"], dataset, is_ekf=False)

    if args.filter in ["ekf", "both"]:
        print(f"\nRunning Extended Kalman Filter (sigma2_pos={args.sigma})...")
        results["ekf"] = extended_kalman_filter(dataset, sigma2_pos=args.sigma)
        print_error_statistics("EKF", results["ekf"], dataset, is_ekf=True)

    # Generate plots if requested
    if args.plot or args.save_plots:
        try:
            import matplotlib

            if not args.plot:
                # Non-interactive backend if only saving
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            plot_dir = Path(args.save_plots) if args.save_plots else Path("plots")
            plot_dir.mkdir(exist_ok=True)

            generate_plots(results, dataset, plot_dir, show=args.plot)

        except ImportError:
            print(
                "\nWarning: matplotlib not installed. Install with: uv pip install matplotlib"
            )
            print("Or install plot extras: uv pip install -e '.[plot]'")

    return 0


def print_error_statistics(name, result, dataset, is_ekf=False):
    # Extract position estimates
    if is_ekf:
        # EKF: [x, vx, y, vy, z, delta_t, delta_t_dot]
        x = result["x_h"][0, :]
        y = result["x_h"][2, :]
        z = result["x_h"][4, :]
    else:
        # NLS: [x, y, z, clock]
        x = result["x_h"][0, :]
        y = result["x_h"][1, :]
        z = result["x_h"][2, :]

    # Ground truth
    x_t = dataset.ground_truth.trajectory_ned[0, :]
    y_t = dataset.ground_truth.trajectory_ned[1, :]
    z_t = dataset.ground_truth.trajectory_ned[2, :]

    # Calculate errors
    error_x = x - x_t
    error_y = y - y_t
    error_z = z - z_t
    error_2d = np.sqrt(error_x**2 + error_y**2)
    error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)

    print(f"\n{name} Error Statistics:")
    print(f"  2D RMS Error: {np.sqrt(np.mean(error_2d**2)):.2f} m")
    print(f"  3D RMS Error: {np.sqrt(np.mean(error_3d**2)):.2f} m")
    print(f"  Max 2D Error: {np.max(error_2d):.2f} m")
    print(f"  Max 3D Error: {np.max(error_3d):.2f} m")


def generate_plots(results, dataset, plot_dir, show=True):
    import matplotlib.pyplot as plt

    # Ground truth
    x_t = dataset.ground_truth.trajectory_ned[0, :]
    y_t = dataset.ground_truth.trajectory_ned[1, :]
    z_t = dataset.ground_truth.trajectory_ned[2, :]

    # Plot NLS if available
    if "nls" in results:
        x_nls = results["nls"]["x_h"][0, :]
        y_nls = results["nls"]["x_h"][1, :]
        z_nls = results["nls"]["x_h"][2, :]

        # 2D plot
        plt.figure(figsize=(10, 8))
        plt.plot(x_nls, y_nls, label="NLS")
        plt.plot(x_t, y_t, "r-.", label="Ground Truth", alpha=0.7)
        plt.grid(True)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("NLS Position Estimation (2D)")
        plt.legend()
        plt.savefig(plot_dir / "nls_2d.png", dpi=300, bbox_inches="tight")
        print(f"Saved {plot_dir / 'nls_2d.png'}")

        # 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_nls, y_nls, z_nls, label="NLS")
        ax.plot(x_t, y_t, z_t, "r-.", label="Ground Truth", alpha=0.7)
        ax.grid(True)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("NLS Position Estimation (3D)")
        ax.set_zlim([-100, 100])
        ax.legend()
        plt.savefig(plot_dir / "nls_3d.png", dpi=300, bbox_inches="tight")
        print(f"Saved {plot_dir / 'nls_3d.png'}")

    # Plot EKF if available
    if "ekf" in results:
        x_ekf = results["ekf"]["x_h"][0, :]
        y_ekf = results["ekf"]["x_h"][2, :]
        z_ekf = results["ekf"]["x_h"][4, :]

        # 2D plot
        plt.figure(figsize=(10, 8))
        plt.plot(x_ekf, y_ekf, label="EKF")
        plt.plot(x_t, y_t, "r-.", label="Ground Truth", alpha=0.7)
        plt.grid(True)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("EKF Position Estimation (2D)")
        plt.legend()
        plt.savefig(plot_dir / "ekf_2d.png", dpi=300, bbox_inches="tight")
        print(f"Saved {plot_dir / 'ekf_2d.png'}")

        # 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_ekf, y_ekf, z_ekf, label="EKF")
        ax.plot(x_t, y_t, z_t, "r-.", label="Ground Truth", alpha=0.7)
        ax.grid(True)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("EKF Position Estimation (3D)")
        ax.set_zlim([-100, 100])
        ax.legend()
        plt.savefig(plot_dir / "ekf_3d.png", dpi=300, bbox_inches="tight")
        print(f"Saved {plot_dir / 'ekf_3d.png'}")

    if show:
        print("\nDisplaying plots... (close windows to continue)")
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    sys.exit(main())
