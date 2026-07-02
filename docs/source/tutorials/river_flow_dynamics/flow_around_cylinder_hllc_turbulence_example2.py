"""Run a depth-averaged HLLC flow-around-cylinder turbulence example.

This script builds the steady subcritical cylinder benchmark as a raised-bed
obstacle on a RasterModelGrid, runs RiverFlowDynamics_HLLC with an optional
algebraic turbulence model, and writes visual diagnostics intended to reveal
wake vorticity, shear layers, and possible vortex-street-like unsteadiness.

The case is primarily a shallow-water hydraulic obstruction test. A resolved
von Karman vortex street is not guaranteed because the model is depth-averaged,
the obstacle is represented as a dry raised-bed footprint, and the cylinder
boundary is not a no-slip viscous wall. The script therefore focuses on
vorticity and cross-stream velocity diagnostics rather than claiming full
viscous wake validation.

Example
-------
Run a moderate-resolution exploratory case::

    python flow_around_cylinder_hllc_turbulence_example.py \
        --dx 0.25 \
        --total-time 250 \
        --plot-interval 5 \
        --turbulence-model smagorinsky

For a faster smoke test::

    python flow_around_cylinder_hllc_turbulence_example.py \
        --dx 0.5 \
        --total-time 30 \
        --plot-interval 10
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from landlab import RasterModelGrid
from landlab.components import RiverFlowDynamics_HLLC

G = 9.81


def normal_depth_from_manning(
    unit_discharge: float,
    mannings_n: float,
    slope: float,
) -> float:
    """Calculate Manning normal depth for wide-channel unit discharge."""
    return (unit_discharge * mannings_n / math.sqrt(slope)) ** (3.0 / 5.0)


def nearest_node(grid: RasterModelGrid, x: float, y: float) -> int:
    """Return the node nearest to coordinates ``x`` and ``y``."""
    distance2 = (grid.x_of_node - x) ** 2 + (grid.y_of_node - y) ** 2
    return int(np.argmin(distance2))


def build_grid_and_initial_state(args: argparse.Namespace):
    """Create the channel, raised cylinder, initial fields, and boundary nodes."""
    ncols = int(round(args.length / args.dx)) + 1
    nrows = int(round(args.width / args.dx)) + 1
    grid = RasterModelGrid((nrows, ncols), xy_spacing=args.dx)

    x = grid.x_of_node.reshape(grid.shape)
    y = grid.y_of_node.reshape(grid.shape)

    unit_discharge = args.discharge / args.width
    h_normal = normal_depth_from_manning(unit_discharge, args.mannings_n, args.slope)
    u_normal = unit_discharge / h_normal
    eta_base = -args.slope * x + h_normal

    z_base = -args.slope * x
    z = z_base.copy()

    radius = 0.5 * args.cylinder_diameter
    cylinder = (x - args.cylinder_x) ** 2 + (y - args.cylinder_y) ** 2 <= radius**2
    z[cylinder] = z_base[cylinder] + args.obstacle_height

    h = np.maximum(eta_base - z, 0.0)
    u = np.where(h > args.dry_depth, u_normal, 0.0)
    v = np.zeros_like(u)

    if args.seed_asymmetry > 0.0:
        rng = np.random.default_rng(args.random_seed)
        wake = np.exp(
            -(
                (
                    x
                    - (args.cylinder_x + 2.5 * args.cylinder_diameter)
                )
                / (4.0 * args.cylinder_diameter)
            )
            ** 2
            - ((y - args.cylinder_y) / (2.5 * args.cylinder_diameter)) ** 2
        )
        random_sign = rng.normal(0.0, 1.0, size=v.shape)
        v += args.seed_asymmetry * u_normal * wake * random_sign
        v[h <= args.dry_depth] = 0.0

    grid.add_field("topographic__elevation", z.ravel(), at="node", clobber=True)
    grid.add_field("surface_water__depth", h.ravel(), at="node", clobber=True)
    grid.add_field("surface_water__elevation", (h + z).ravel(), at="node", clobber=True)
    grid.add_field("surface_water__x_velocity", u.ravel(), at="node", clobber=True)
    grid.add_field("surface_water__y_velocity", v.ravel(), at="node", clobber=True)
    grid.add_field("surface_water__x_momentum", (h * u).ravel(), at="node", clobber=True)
    grid.add_field("surface_water__y_momentum", (h * v).ravel(), at="node", clobber=True)

    rows = np.arange(1, nrows - 1)
    entry_nodes = rows * ncols
    exit_nodes = rows * ncols + (ncols - 1)

    inlet_h = np.full(rows.size, h_normal)
    inlet_u = np.full(rows.size, u_normal)
    inlet_v = np.zeros(rows.size)

    outlet_eta = np.full(rows.size, eta_base[rows, -1].mean())

    reference = {
        "h_normal": h_normal,
        "u_normal": u_normal,
        "froude_normal": u_normal / math.sqrt(G * h_normal),
        "h_stagnation": h_normal + u_normal**2 / (2.0 * G),
        "unit_discharge": unit_discharge,
    }

    return grid, entry_nodes, inlet_h, inlet_u, inlet_v, exit_nodes, outlet_eta, reference


def make_solver(
    args: argparse.Namespace,
    grid,
    entry_nodes,
    inlet_h,
    inlet_u,
    inlet_v,
    exit_nodes,
    outlet_eta,
):
    """Instantiate RiverFlowDynamics_HLLC with turbulence parameters."""
    return RiverFlowDynamics_HLLC(
        grid,
        mannings_n=args.mannings_n,
        cfl=args.cfl,
        order=args.order,
        fixed_entry_nodes=entry_nodes,
        entry_nodes_h_values=inlet_h,
        entry_nodes_u_values=inlet_u,
        entry_nodes_v_values=inlet_v,
        fixed_exit_nodes=exit_nodes,
        exit_nodes_eta_values=outlet_eta,
        wall_edges={"top", "bottom"},
        update_link_fields=True,
        turbulence_model=args.turbulence_model,
        smagorinsky_cs=args.smagorinsky_cs,
        filter_width_model=args.filter_width_model,
        filter_width_coefficient=args.filter_width_coefficient,
        parabolic_alpha=args.parabolic_alpha,
        eddy_viscosity=args.eddy_viscosity,
        eddy_viscosity_background=args.eddy_viscosity_background,
        eddy_viscosity_max=args.eddy_viscosity_max,
        mask_dry_cells=True,
    )


def field_2d(grid: RasterModelGrid, name: str) -> np.ndarray:
    """Return a node field reshaped to grid shape."""
    return grid.at_node[name].reshape(grid.shape)


def calculate_vorticity(grid: RasterModelGrid, dry_depth: float) -> np.ndarray:
    """Calculate depth-averaged vertical vorticity from nodal velocities."""
    h = field_2d(grid, "surface_water__depth")
    u = field_2d(grid, "surface_water__x_velocity")
    v = field_2d(grid, "surface_water__y_velocity")

    dvdx = np.gradient(v, grid.dx, axis=1)
    dudy = np.gradient(u, grid.dy, axis=0)
    omega = dvdx - dudy
    omega = np.where(h > dry_depth, omega, np.nan)
    return omega


def calculate_speed(grid: RasterModelGrid, dry_depth: float) -> np.ndarray:
    """Calculate velocity magnitude and mask dry cells."""
    h = field_2d(grid, "surface_water__depth")
    u = field_2d(grid, "surface_water__x_velocity")
    v = field_2d(grid, "surface_water__y_velocity")
    speed = np.sqrt(u**2 + v**2)
    return np.where(h > dry_depth, speed, np.nan)


def robust_symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    """Return a robust symmetric color limit for signed fields."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    return max(limit, 1.0e-12)


def plot_frame(
    grid: RasterModelGrid,
    args: argparse.Namespace,
    elapsed_time: float,
    output_path: Path,
    title: str,
):
    """Write a two-panel velocity and vorticity frame."""
    x = grid.x_of_node.reshape(grid.shape)
    y = grid.y_of_node.reshape(grid.shape)
    speed = calculate_speed(grid, args.dry_depth)
    omega = calculate_vorticity(grid, args.dry_depth)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.0, 4.2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    speed_im = axes[0].pcolormesh(
        x,
        y,
        speed,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=2.5,
    )
    axes[0].set_title("velocity magnitude")
    speed_cbar = fig.colorbar(speed_im, ax=axes[0], pad=0.01)
    speed_cbar.set_label("velocity magnitude (m/s)")

    omega_im = axes[1].pcolormesh(
        x,
        y,
        omega,
        shading="auto",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )
    axes[1].set_title("vertical vorticity")
    omega_cbar = fig.colorbar(omega_im, ax=axes[1], pad=0.01)
    omega_cbar.set_label("vertical vorticity (1/s)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(args.x_plot_min, args.x_plot_max)
        ax.set_ylim(0.0, args.width)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.suptitle(f"{title}, t = {elapsed_time:.1f} s")
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def plot_final_diagnostics(
    grid: RasterModelGrid,
    args: argparse.Namespace,
    reference: dict,
    output_path: Path,
):
    """Write a four-panel final hydraulic and turbulence diagnostic figure."""
    x = grid.x_of_node.reshape(grid.shape)
    y = grid.y_of_node.reshape(grid.shape)
    h = field_2d(grid, "surface_water__depth")
    speed = calculate_speed(grid, args.dry_depth)
    omega = calculate_vorticity(grid, args.dry_depth)

    if "surface_water__eddy_viscosity" in grid.at_node:
        nu_t = field_2d(grid, "surface_water__eddy_viscosity")
        nu_t = np.where(h > args.dry_depth, nu_t, np.nan)
    else:
        nu_t = np.full_like(h, np.nan)

    data = [
        (h, "water depth (m)", "viridis", None),
        (speed, "speed (m/s)", "viridis", None),
        (omega, "vertical vorticity (1/s)", "coolwarm", robust_symmetric_limit(omega, 98.5)),
        (nu_t, "eddy viscosity (m2/s)", "magma", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), constrained_layout=True)
    for ax, (arr, label, cmap, signed_limit) in zip(axes.ravel(), data):
        if signed_limit is None:
            im = ax.pcolormesh(x, y, arr, shading="auto", cmap=cmap)
        else:
            im = ax.pcolormesh(
                x,
                y,
                arr,
                shading="auto",
                cmap=cmap,
                vmin=-signed_limit,
                vmax=signed_limit,
            )
        ax.contour(x, y, h, levels=[args.dry_depth], colors="k", linewidths=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(args.x_plot_min, args.x_plot_max)
        ax.set_ylim(0.0, args.width)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, pad=0.01)

    axes[0, 1].text(
        0.02,
        0.98,
        (
            f"hn = {reference['h_normal']:.3f} m\n"
            f"Un = {reference['u_normal']:.3f} m/s\n"
            f"Fr = {reference['froude_normal']:.3f}"
        ),
        transform=axes[0, 1].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def plot_probe_timeseries(records: list[dict], output_path: Path, dpi: int):
    """Write wake-probe cross-stream velocity time series."""
    if not records:
        return

    time = np.array([row["time"] for row in records])

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for key in ("v_upper", "v_lower", "v_center"):
        ax.plot(time, [row[key] for row in records], label=key)

    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cross-stream velocity (m/s)")
    ax.set_title("Wake probe cross-stream velocity")
    ax.legend()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_probe_csv(records: list[dict], output_path: Path):
    """Write probe records to CSV."""
    if not records:
        return

    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_summary(
    args: argparse.Namespace,
    reference: dict,
    grid: RasterModelGrid,
    output_path: Path,
):
    """Write a compact text summary of final diagnostics."""
    h = field_2d(grid, "surface_water__depth")
    speed = calculate_speed(grid, args.dry_depth)
    omega = calculate_vorticity(grid, args.dry_depth)

    if "surface_water__eddy_viscosity" in grid.at_node:
        nu_t = field_2d(grid, "surface_water__eddy_viscosity")
        nu_max = float(np.nanmax(nu_t))
        nu_mean = float(np.nanmean(np.where(h > args.dry_depth, nu_t, np.nan)))
    else:
        nu_max = float("nan")
        nu_mean = float("nan")

    with output_path.open("w") as fp:
        fp.write("Flow around cylinder HLLC turbulence example\n")
        fp.write("============================================\n\n")
        fp.write(f"turbulence_model: {args.turbulence_model}\n")
        fp.write(f"dx: {args.dx} m\n")
        fp.write(f"total_time: {args.total_time} s\n")
        fp.write(f"normal_depth: {reference['h_normal']:.6f} m\n")
        fp.write(f"normal_velocity: {reference['u_normal']:.6f} m/s\n")
        fp.write(f"normal_froude: {reference['froude_normal']:.6f}\n")
        fp.write(f"stagnation_depth_reference: {reference['h_stagnation']:.6f} m\n")
        fp.write(
            "final_depth_min_wet: "
            f"{float(np.nanmin(np.where(h > args.dry_depth, h, np.nan))):.6f} m\n"
        )
        fp.write(f"final_depth_max: {float(np.nanmax(h)):.6f} m\n")
        fp.write(f"final_speed_max: {float(np.nanmax(speed)):.6f} m/s\n")
        fp.write(f"final_abs_vorticity_max: {float(np.nanmax(np.abs(omega))):.6e} 1/s\n")
        fp.write(f"final_eddy_viscosity_max: {nu_max:.6e} m2/s\n")
        fp.write(f"final_eddy_viscosity_mean_wet: {nu_mean:.6e} m2/s\n")
        fp.write("\nInterpretation note:\n")
        fp.write(
            "This is a depth-averaged raised-obstacle experiment. It can show wake "
            "vorticity and unsteady cross-stream velocity if the numerical and "
            "turbulence settings allow instability, but it is not a 3D viscous "
            "vortex-shedding validation.\n"
        )


def maybe_make_gif(frame_paths: list[Path], gif_path: Path, fps: float):
    """Create an animated GIF if imageio is installed."""
    if not frame_paths:
        return

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio is not installed; skipping GIF creation.")
        return

    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(gif_path, images, duration=1.0 / fps)


def run(args: argparse.Namespace) -> None:
    """Run the cylinder example and write visualizations."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    (
        grid,
        entry_nodes,
        inlet_h,
        inlet_u,
        inlet_v,
        exit_nodes,
        outlet_eta,
        reference,
    ) = build_grid_and_initial_state(args)

    flow = make_solver(
        args,
        grid,
        entry_nodes,
        inlet_h,
        inlet_u,
        inlet_v,
        exit_nodes,
        outlet_eta,
    )

    probe_nodes = {
        "upper": nearest_node(
            grid,
            args.probe_x,
            args.cylinder_y + 0.75 * args.cylinder_diameter,
        ),
        "lower": nearest_node(
            grid,
            args.probe_x,
            args.cylinder_y - 0.75 * args.cylinder_diameter,
        ),
        "center": nearest_node(grid, args.probe_x, args.cylinder_y),
    }

    next_plot = 0.0
    next_probe = 0.0
    frame_paths: list[Path] = []
    records: list[dict] = []
    frame_index = 0

    print("Reference state:")
    print(f"  h_normal = {reference['h_normal']:.4f} m")
    print(f"  u_normal = {reference['u_normal']:.4f} m/s")
    print(f"  Fr_normal = {reference['froude_normal']:.4f}")
    print(f"  h_stagnation = {reference['h_stagnation']:.4f} m")
    print(
        f"Running to t = {args.total_time:.1f} s "
        f"with turbulence_model={args.turbulence_model!r}"
    )

    while flow.elapsed_time < args.total_time:
        dt = min(flow.current_dt, args.max_dt, args.total_time - flow.elapsed_time)

        if next_plot > flow.elapsed_time:
            dt = min(dt, next_plot - flow.elapsed_time)

        if dt <= 0.0:
            dt = min(flow.current_dt, args.max_dt)

        flow.run_one_step(dt=dt)

        if flow.elapsed_time + 1.0e-9 >= next_probe:
            v = grid.at_node["surface_water__y_velocity"]
            omega = calculate_vorticity(grid, args.dry_depth).ravel()
            records.append(
                {
                    "time": flow.elapsed_time,
                    "v_upper": float(v[probe_nodes["upper"]]),
                    "v_lower": float(v[probe_nodes["lower"]]),
                    "v_center": float(v[probe_nodes["center"]]),
                    "omega_upper": float(omega[probe_nodes["upper"]]),
                    "omega_lower": float(omega[probe_nodes["lower"]]),
                    "omega_center": float(omega[probe_nodes["center"]]),
                }
            )
            next_probe += args.probe_interval

        if flow.elapsed_time + 1.0e-9 >= next_plot:
            frame_path = frames_dir / f"vorticity_{frame_index:04d}.png"
            plot_frame(grid, args, flow.elapsed_time, frame_path, args.turbulence_model)
            frame_paths.append(frame_path)
            frame_index += 1
            next_plot += args.plot_interval
            print(f"  wrote frame {frame_index:04d} at t = {flow.elapsed_time:.2f} s")

    plot_final_diagnostics(
        grid,
        args,
        reference,
        output_dir / "final_hydraulic_turbulence_fields.png",
    )
    plot_probe_timeseries(records, output_dir / "wake_probe_timeseries.png", args.dpi)
    write_probe_csv(records, output_dir / "wake_probe_timeseries.csv")
    write_summary(args, reference, grid, output_dir / "summary.txt")

    if args.make_gif:
        maybe_make_gif(frame_paths, output_dir / "vorticity_animation.gif", args.gif_fps)

    print(f"Finished. Results written to: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run a RiverFlowDynamics_HLLC turbulence example for flow around a cylinder."
    )
    parser.add_argument("--output-dir", default="flow_around_cylinder_turbulence_output")
    parser.add_argument("--length", type=float, default=100.0)
    parser.add_argument("--width", type=float, default=20.0)
    parser.add_argument("--dx", type=float, default=0.25)
    parser.add_argument("--cylinder-diameter", type=float, default=2.0)
    parser.add_argument("--cylinder-x", type=float, default=30.0)
    parser.add_argument("--cylinder-y", type=float, default=10.0)
    parser.add_argument("--obstacle-height", type=float, default=2.0)
    parser.add_argument("--slope", type=float, default=0.001)
    parser.add_argument("--mannings-n", type=float, default=0.025)
    parser.add_argument("--discharge", type=float, default=20.0)
    parser.add_argument("--dry-depth", type=float, default=1.0e-4)
    parser.add_argument("--total-time", type=float, default=250.0)
    parser.add_argument("--max-dt", type=float, default=0.05)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--order", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--turbulence-model",
        default="smagorinsky",
        choices=("none", "constant", "smagorinsky", "parabolic", "hybrid_additive"),
    )
    parser.add_argument("--smagorinsky-cs", type=float, default=0.08)
    parser.add_argument(
        "--filter-width-model",
        default="grid",
        choices=("grid", "depth", "depth_limited_grid"),
    )
    parser.add_argument("--filter-width-coefficient", type=float, default=1.0)
    parser.add_argument("--parabolic-alpha", type=float, default=None)
    parser.add_argument("--eddy-viscosity", type=float, default=0.0)
    parser.add_argument("--eddy-viscosity-background", type=float, default=1.0e-6)
    parser.add_argument("--eddy-viscosity-max", type=float, default=0.25)
    parser.add_argument("--seed-asymmetry", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--plot-interval", type=float, default=5.0)
    parser.add_argument("--probe-interval", type=float, default=0.5)
    parser.add_argument("--probe-x", type=float, default=42.0)
    parser.add_argument("--x-plot-min", type=float, default=20.0)
    parser.add_argument("--x-plot-max", type=float, default=65.0)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--gif-fps", type=float, default=8.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())