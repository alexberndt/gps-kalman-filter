import argparse
import sys
from pathlib import Path

import numpy as np

from gps.data.loader import load_gps_data
from gps.filters.ekf import extended_kalman_filter
from gps.filters.nl_ls import nonlinear_least_squares


def main():
    parser = argparse.ArgumentParser(description="GPS position estimation using Kalman filtering")
    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to GPSdata.mat file (default: ./data/GPSdata.mat)",
    )
    parser.add_argument("--plot", action="store_true", help="Generate and display plots")
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
    data_path = Path("data") / "GPSdata.mat" if args.data_path is None else Path(args.data_path)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        return 1

    # Load data
    print(f"Loading GPS data from {data_path}...")
    dataset = load_gps_data(str(data_path))
    print(f"Loaded {dataset.num_satellites} satellites, {dataset.num_timesteps} timesteps")

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
            plot_dir = Path(args.save_plots) if args.save_plots else Path("plots")
            plot_dir.mkdir(exist_ok=True)

            generate_plots(results, dataset, plot_dir, show=args.plot)

        except ImportError as e:
            print("\nWarning: bokeh not installed. Install with: uv add bokeh")
            print(f"Error: {e}")

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
    import plotly.graph_objects as go
    from bokeh.layouts import column
    from bokeh.models import HoverTool
    from bokeh.plotting import figure, output_file, save
    from bokeh.plotting import show as bokeh_show

    # Ground truth
    x_t = dataset.ground_truth.trajectory_ned[0, :]
    y_t = dataset.ground_truth.trajectory_ned[1, :]
    z_t = dataset.ground_truth.trajectory_ned[2, :]

    # Color palette for visually appealing plots
    colors = {
        "nls": "#2E86AB",  # Blue
        "ekf": "#A23B72",  # Purple
        "truth": "#F18F01",  # Orange
        "grid": "#E8E9EB",  # Light gray
    }

    plots_to_show = []
    plots_3d_to_show = []

    # Plot NLS if available
    if "nls" in results:
        x_nls = results["nls"]["x_h"][0, :]
        y_nls = results["nls"]["x_h"][1, :]
        z_nls = results["nls"]["x_h"][2, :]

        # 2D plot
        p_nls_2d = figure(
            width=800,
            height=700,
            title="NLS Position Estimation (2D)",
            x_axis_label="x [m]",
            y_axis_label="y [m]",
            background_fill_color="#FAFAFA",
            border_fill_color="white",
        )

        # Add ground truth
        p_nls_2d.line(
            x_t,
            y_t,
            legend_label="Ground Truth",
            line_color=colors["truth"],
            line_width=2.5,
            line_dash="dashed",
            alpha=0.8,
        )

        # Add NLS estimate
        p_nls_2d.line(
            x_nls,
            y_nls,
            legend_label="NLS Estimate",
            line_color=colors["nls"],
            line_width=2.5,
            alpha=0.9,
        )

        # Style the plot
        p_nls_2d.legend.location = "top_right"
        p_nls_2d.legend.background_fill_alpha = 0.8
        p_nls_2d.legend.border_line_color = "#CCCCCC"
        p_nls_2d.grid.grid_line_color = colors["grid"]
        p_nls_2d.grid.grid_line_alpha = 0.5
        p_nls_2d.title.text_font_size = "16pt"
        p_nls_2d.axis.axis_label_text_font_size = "12pt"

        # Add hover tool
        hover_nls = HoverTool(tooltips=[("x", "$x{0.2f} m"), ("y", "$y{0.2f} m")])
        p_nls_2d.add_tools(hover_nls)

        # Save 2D plot
        output_file(plot_dir / "nls_2d.html")
        save(p_nls_2d)
        print(f"Saved {plot_dir / 'nls_2d.html'}")

        plots_to_show.append(p_nls_2d)

        # 3D plot with Plotly
        fig_nls_3d = go.Figure()

        # Add ground truth trajectory
        fig_nls_3d.add_trace(
            go.Scatter3d(
                x=x_t,
                y=y_t,
                z=z_t,
                mode="lines",
                name="Ground Truth",
                line=dict(color=colors["truth"], width=4, dash="dash"),
                opacity=0.8,
            )
        )

        # Add NLS estimate trajectory
        fig_nls_3d.add_trace(
            go.Scatter3d(
                x=x_nls,
                y=y_nls,
                z=z_nls,
                mode="lines",
                name="NLS Estimate",
                line=dict(color=colors["nls"], width=4),
                opacity=0.9,
            )
        )

        # Update layout for visual appeal
        fig_nls_3d.update_layout(
            title=dict(text="NLS Position Estimation (3D)", font=dict(size=20)),
            scene=dict(
                xaxis=dict(
                    title="x [m]",
                    backgroundcolor="#FAFAFA",
                    gridcolor=colors["grid"],
                    showbackground=True,
                ),
                yaxis=dict(
                    title="y [m]",
                    backgroundcolor="#FAFAFA",
                    gridcolor=colors["grid"],
                    showbackground=True,
                ),
                zaxis=dict(
                    title="z [m]",
                    backgroundcolor="#FAFAFA",
                    gridcolor=colors["grid"],
                    showbackground=True,
                    range=[-100, 100],
                ),
                aspectmode="auto",
            ),
            showlegend=True,
            legend=dict(x=0.7, y=0.9, bgcolor="rgba(255,255,255,0.8)"),
            width=900,
            height=700,
            paper_bgcolor="white",
        )

        # Save 3D plot
        fig_nls_3d.write_html(plot_dir / "nls_3d.html")
        print(f"Saved {plot_dir / 'nls_3d.html'}")
        plots_3d_to_show.append(fig_nls_3d)

    # Plot EKF if available
    if "ekf" in results:
        x_ekf = results["ekf"]["x_h"][0, :]
        y_ekf = results["ekf"]["x_h"][2, :]
        z_ekf = results["ekf"]["x_h"][4, :]

        # 2D plot
        p_ekf_2d = figure(
            width=800,
            height=700,
            title="EKF Position Estimation (2D)",
            x_axis_label="x [m]",
            y_axis_label="y [m]",
            background_fill_color="#FAFAFA",
            border_fill_color="white",
        )

        # Add ground truth
        p_ekf_2d.line(
            x_t,
            y_t,
            legend_label="Ground Truth",
            line_color=colors["truth"],
            line_width=2.5,
            line_dash="dashed",
            alpha=0.8,
        )

        # Add EKF estimate
        p_ekf_2d.line(
            x_ekf,
            y_ekf,
            legend_label="EKF Estimate",
            line_color=colors["ekf"],
            line_width=2.5,
            alpha=0.9,
        )

        # Style the plot
        p_ekf_2d.legend.location = "top_right"
        p_ekf_2d.legend.background_fill_alpha = 0.8
        p_ekf_2d.legend.border_line_color = "#CCCCCC"
        p_ekf_2d.grid.grid_line_color = colors["grid"]
        p_ekf_2d.grid.grid_line_alpha = 0.5
        p_ekf_2d.title.text_font_size = "16pt"
        p_ekf_2d.axis.axis_label_text_font_size = "12pt"

        # Add hover tool
        hover_ekf = HoverTool(tooltips=[("x", "$x{0.2f} m"), ("y", "$y{0.2f} m")])
        p_ekf_2d.add_tools(hover_ekf)

        # Save 2D plot
        output_file(plot_dir / "ekf_2d.html")
        save(p_ekf_2d)
        print(f"Saved {plot_dir / 'ekf_2d.html'}")

        plots_to_show.append(p_ekf_2d)

        # 3D plot with Plotly
        fig_ekf_3d = go.Figure()

        # Add ground truth trajectory
        fig_ekf_3d.add_trace(
            go.Scatter3d(
                x=x_t,
                y=y_t,
                z=z_t,
                mode="lines",
                name="Ground Truth",
                line=dict(color=colors["truth"], width=4, dash="dash"),
                opacity=0.8,
            )
        )

        # Add EKF estimate trajectory
        fig_ekf_3d.add_trace(
            go.Scatter3d(
                x=x_ekf,
                y=y_ekf,
                z=z_ekf,
                mode="lines",
                name="EKF Estimate",
                line=dict(color=colors["ekf"], width=4),
                opacity=0.9,
            )
        )

        # Update layout for visual appeal
        fig_ekf_3d.update_layout(
            title=dict(text="EKF Position Estimation (3D)", font=dict(size=20)),
            scene=dict(
                xaxis=dict(
                    title="x [m]",
                    backgroundcolor="#FAFAFA",
                    gridcolor=colors["grid"],
                    showbackground=True,
                ),
                yaxis=dict(
                    title="y [m]",
                    backgroundcolor="#FAFAFA",
                    gridcolor=colors["grid"],
                    showbackground=True,
                ),
                zaxis=dict(
                    title="z [m]",
                    backgroundcolor="#FAFAFA",
                    gridcolor=colors["grid"],
                    showbackground=True,
                    range=[-100, 100],
                ),
                aspectmode="auto",
            ),
            showlegend=True,
            legend=dict(x=0.7, y=0.9, bgcolor="rgba(255,255,255,0.8)"),
            width=900,
            height=700,
            paper_bgcolor="white",
        )

        # Save 3D plot
        fig_ekf_3d.write_html(plot_dir / "ekf_3d.html")
        print(f"Saved {plot_dir / 'ekf_3d.html'}")
        plots_3d_to_show.append(fig_ekf_3d)

    if show:
        if plots_to_show:
            print("\nOpening 2D plots in browser...")
            # Create a combined dashboard view for Bokeh plots
            dashboard = column(*plots_to_show)
            output_file(plot_dir / "dashboard_2d.html")
            bokeh_show(dashboard)

        if plots_3d_to_show:
            print("\nOpening 3D plots in browser...")
            # Show Plotly 3D plots
            for fig_3d in plots_3d_to_show:
                fig_3d.show()


if __name__ == "__main__":
    sys.exit(main())
