# Auto-generated from Jupyter Notebook
# Source: river_temperature_dynamics_tutorial.ipynb


# ===== Cell Separator =====
import matplotlib.pyplot as plt
import numpy as np

from landlab import RasterModelGrid
from landlab.components import RiverFlowDynamics, RiverTemperatureDynamics

# ===== Cell Separator =====
help(RiverTemperatureDynamics)

# ===== Cell Separator =====
# Channel parameters
mannings_n = 0.025  # Manning's roughness coefficient [s/m^(1/3)]
channel_slope = 0.001  # Channel slope [m/m]
channel_width = 20.0  # Channel width [m]
channel_length = 200.0  # Channel length [m]

# Grid parameters
dx = 2.0  # Grid spacing [m]
nrows = 13  # 2 bank rows + 11 channel rows = 20 m wide channel
ncols = 101  # 0 to 200 m

# Hydraulic parameters
h_normal = 0.5  # Normal flow depth [m]
v_normal = (1.0 / mannings_n) * h_normal ** (2.0 / 3.0) * channel_slope**0.5
print(f"Normal flow velocity: {v_normal:.4f} m/s")
print(f"Froude number: {v_normal / np.sqrt(9.81 * h_normal):.3f}")

# Simulation parameters
dt_hydro = 1.0  # Hydrodynamic time step [s]
n_hydro_steps = 300  # Steps to reach steady state
dt_temp = 0.5  # Temperature time step [s]
t_total = 200.0  # Total temperature simulation time [s]

# ===== Cell Separator =====
# Create and set up the grid
grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)

# ===== Cell Separator =====
# The grid represents a rectangular channel with slope = 0.001 m/m
te = grid.add_field(
    "topographic__elevation",
    (dx * (ncols - 1) - grid.x_of_node) * channel_slope,
    at="node",
)

# Elevated banks (rows 0 and 12)
bank_mask = (grid.y_of_node < dx) | (grid.y_of_node > dx * (nrows - 2))
te[bank_mask] = 5.0

# ===== Cell Separator =====
# Showing the topography
grid.imshow("topographic__elevation")

# ===== Cell Separator =====
# Initial conditions: pre-fill channel at normal depth
h = grid.add_zeros("surface_water__depth", at="node")
vel = grid.add_zeros("surface_water__velocity", at="link")
wse = grid.add_zeros("surface_water__elevation", at="node")

# Fill channel nodes with normal depth
channel_mask = ~bank_mask
h[channel_mask] = h_normal
wse[:] = te + h

# Pre-set velocity on horizontal links within the channel
horiz_links = grid.horizontal_links
link_y = grid.y_of_node[grid.node_at_link_tail[horiz_links]]
channel_hlinks = horiz_links[(link_y >= dx) & (link_y <= dx * (nrows - 2))]
vel[channel_hlinks] = v_normal

# ===== Cell Separator =====
# Entry boundary conditions: left edge of the channel (column 0, rows 1-11)
channel_rows = np.arange(1, nrows - 1)
fixed_entry_nodes = channel_rows * ncols  # Column 0 of each channel row
fixed_entry_links = grid.links_at_node[fixed_entry_nodes][:, 0]

# Fixed depth and velocity at the inlet
entry_nodes_h_values = np.full(len(fixed_entry_nodes), h_normal)
entry_links_vel_values = np.full(len(fixed_entry_links), v_normal)

# ===== Cell Separator =====
# Construct the hydrodynamic component
rfd = RiverFlowDynamics(
    grid,
    dt=dt_hydro,
    mannings_n=mannings_n,
    eddy_viscosity=1e-3,
    threshold_depth=0.005,
    theta=0.7,
    fixed_entry_nodes=fixed_entry_nodes,
    fixed_entry_links=fixed_entry_links,
    entry_nodes_h_values=entry_nodes_h_values,
    entry_links_vel_values=entry_links_vel_values,
)

# Spin up to steady state
print("Running hydrodynamics to steady state...")
for i in range(n_hydro_steps):
    rfd.run_one_step()
    if (i + 1) % 100 == 0:
        print(
            f"  Step {i + 1}/{n_hydro_steps}: "
            f"h_max = {h[channel_mask].max():.4f} m, "
            f"v_mean = {vel[channel_hlinks].mean():.4f} m/s"
        )

print("\nSteady state reached.")
print(
    f"  Channel velocity: {vel[channel_hlinks].mean():.4f} m/s (expected {v_normal:.4f})"
)

# ===== Cell Separator =====
grid.imshow("surface_water__depth")

# ===== Cell Separator =====
# Temperature parameters
T_background = 15.0  # Background temperature [deg C]
T_peak = 25.0  # Peak temperature of the warm patch [deg C]
x0_patch = 40.0  # Center of the warm patch [m]
sigma_x = 6.0  # Standard deviation of the warm patch [m]

# Initialize temperature field
T = grid.add_zeros("surface_water__temperature", at="node")
T[:] = T_background

# Apply Gaussian warm patch (uniform in y within the channel)
for node in range(grid.number_of_nodes):
    if not bank_mask[node]:
        dist_x = grid.x_of_node[node] - x0_patch
        T[node] = T_background + (T_peak - T_background) * np.exp(
            -(dist_x**2) / (2 * sigma_x**2)
        )
T[bank_mask] = T_background

# Set up the advection velocity field (copy from steady-state hydraulics)
adv_vel = grid.add_zeros("advection__velocity", at="link")
adv_vel[:] = vel[:]

# Atmospheric forcing: all zeros (no heat exchange for this example)
_ = grid.add_zeros("air__temperature", at="node")
_ = grid.add_zeros("air__relative_humidity", at="node")
_ = grid.add_zeros("air__velocity", at="node")
_ = grid.add_zeros("radiation__incoming_shortwave_flux", at="node")
_ = grid.add_zeros("solar__altitude_angle", at="node")

# ===== Cell Separator =====
grid.imshow(
    "surface_water__temperature", vmin=T_background - 0.5, vmax=T_peak, cmap="RdYlBu_r"
)
plt.title("Initial temperature field")

# ===== Cell Separator =====
# Construct the temperature component
rtd = RiverTemperatureDynamics(
    grid,
    shade_factor=0.0,
    alpha_L=10.0,
    alpha_T=0.6,
    ustar_fraction=0.1,
)

# ===== Cell Separator =====
# Time stepping
n_temp_steps = int(t_total / dt_temp)
snapshot_times = [0, 25, 50, 100, 150, 200]
snapshots = {0.0: T.copy()}

print(
    f"Running {n_temp_steps} temperature steps (dt = {dt_temp} s, total = {t_total} s)..."
)
for step in range(n_temp_steps):
    t_sim = (step + 1) * dt_temp

    rtd.run_one_step(dt_temp)

    # Boundary conditions
    T[bank_mask] = T_background
    T[fixed_entry_nodes] = T_background  # Cold water entering

    # Save snapshots
    if t_sim in snapshot_times:
        snapshots[t_sim] = T.copy()
        T_channel = T[channel_mask]
        print(
            f"  t = {t_sim:6.1f} s: T_min = {T_channel.min():.3f}, "
            f"T_max = {T_channel.max():.3f} deg C"
        )

print("Done!")

# ===== Cell Separator =====
grid.imshow(
    "surface_water__temperature", vmin=T_background - 0.5, vmax=T_peak, cmap="RdYlBu_r"
)
plt.title(f"Temperature field at t = {t_total:.0f} s")

# ===== Cell Separator =====
fig, axes = plt.subplots(len(snapshot_times), 1, figsize=(14, 3 * len(snapshot_times)))
plt.subplots_adjust(hspace=0.4)

for ax, t_snap in zip(axes, snapshot_times):
    T_2d = snapshots[t_snap].reshape((nrows, ncols))
    T_channel_2d = T_2d[1:-1, :]  # Only channel rows

    im = ax.imshow(
        T_channel_2d,
        origin="lower",
        aspect="auto",
        extent=[0, dx * (ncols - 1), 0, dx * (nrows - 3)],
        vmin=T_background - 0.2,
        vmax=T_peak,
        cmap="RdYlBu_r",
        interpolation="bilinear",
    )

    # Expected patch center from advection
    x_expected = x0_patch + v_normal * t_snap
    if x_expected < dx * (ncols - 1):
        ax.axvline(x_expected, color="k", ls="--", lw=1, alpha=0.6)

    ax.set_title(f"t = {t_snap:.0f} s", fontsize=11, fontweight="bold")
    ax.set_ylabel("y (m)")
    plt.colorbar(im, ax=ax, label="T (°C)", shrink=0.8)

axes[-1].set_xlabel("x (m)")
fig.suptitle(
    "2D Temperature Field — Advection + Anisotropic Dispersion\n"
    f"V = {v_normal:.3f} m/s | No atmospheric heat exchange",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)
plt.show()

# ===== Cell Separator =====
fig, ax = plt.subplots(figsize=(12, 5))
center_row = nrows // 2
x_arr = np.arange(ncols) * dx
colors = plt.cm.viridis(np.linspace(0, 0.9, len(snapshot_times)))

for i, t_snap in enumerate(snapshot_times):
    T_center = snapshots[t_snap].reshape((nrows, ncols))[center_row, :]
    ax.plot(x_arr, T_center, color=colors[i], lw=2, label=f"t = {t_snap:.0f} s")

ax.set_xlabel("x (m)", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.set_title("Centerline Temperature Profiles", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, dx * (ncols - 1))
ax.axhline(T_background, color="gray", ls=":", alpha=0.5)
plt.tight_layout()
plt.show()

# ===== 2nd PART =====
# Domain: 600 m x 40 m (large enough to avoid boundary effects)
dx_a = 2.0
nrows_a = 23  # rows 0,22 = banks; 1-21 = 40 m wide channel
ncols_a = 301  # 0 to 600 m

grid_a = RasterModelGrid((nrows_a, ncols_a), xy_spacing=dx_a)

# Flat bed (no need for RiverFlowDynamics — prescribed velocity)
te_a = grid_a.add_zeros("topographic__elevation", at="node")

# Bank mask
bank_mask_a = (grid_a.y_of_node < dx_a) | (grid_a.y_of_node > dx_a * (nrows_a - 2))
channel_mask_a = ~bank_mask_a

# Prescribed uniform fields
h_a = grid_a.add_zeros("surface_water__depth", at="node")
h_a[channel_mask_a] = 0.5

vel_a = grid_a.add_zeros("surface_water__velocity", at="link")
vel_a[grid_a.horizontal_links] = v_normal

adv_vel_a = grid_a.add_zeros("advection__velocity", at="link")
adv_vel_a[grid_a.horizontal_links] = v_normal

# Atmospheric fields (all zeros — pure advection-diffusion)
_ = grid_a.add_zeros("air__temperature", at="node")
_ = grid_a.add_zeros("air__relative_humidity", at="node")
_ = grid_a.add_zeros("air__velocity", at="node")
_ = grid_a.add_zeros("radiation__incoming_shortwave_flux", at="node")
_ = grid_a.add_zeros("solar__altitude_angle", at="node")

# ===== Cell Separator =====
# Temperature IC: 2D Gaussian
T_bg_a = 15.0
dT_a = 10.0
x0_a = 100.0  # Patch center [m]
y0_a = dx_a * (nrows_a - 1) / 2.0  # Channel center [m]
sigma_x0 = 8.0  # Initial x-spread [m]
sigma_y0 = 100.0  # Very wide in y (effectively uniform)

T_a = grid_a.add_zeros("surface_water__temperature", at="node")
T_a[:] = T_bg_a

for node in range(grid_a.number_of_nodes):
    if not bank_mask_a[node]:
        rx = grid_a.x_of_node[node] - x0_a
        ry = grid_a.y_of_node[node] - y0_a
        T_a[node] = T_bg_a + dT_a * np.exp(
            -(rx**2) / (2 * sigma_x0**2) - ry**2 / (2 * sigma_y0**2)
        )
T_a[bank_mask_a] = T_bg_a

print(f"Initial peak temperature: {T_a.max():.3f} °C")
print(f"Patch center: x = {x0_a} m, y = {y0_a:.1f} m")

# ===== Cell Separator =====
# Construct RiverTemperatureDynamics (no atmospheric exchange)
rtd_a = RiverTemperatureDynamics(
    grid_a,
    shade_factor=0.0,
    sigma_lw_factor=1.0,
    alpha_L=10.0,
    alpha_T=0.6,
    ustar_fraction=0.1,
)

# Compute the dispersion coefficients (for the analytical solution)
u_star = 0.1 * v_normal
D_L = 10.0 * 0.5 * u_star  # alpha_L * h * u_star
D_T = 0.6 * 0.5 * u_star  # alpha_T * h * u_star
print(f"Dispersion coefficients: D_L = {D_L:.4f} m²/s, D_T = {D_T:.5f} m²/s")


# Analytical solution function
def T_analytical(x, y, t):
    sx = np.sqrt(sigma_x0**2 + 2 * D_L * t)
    sy = np.sqrt(sigma_y0**2 + 2 * D_T * t)
    amplitude = dT_a * (sigma_x0 * sigma_y0) / (sx * sy)
    return T_bg_a + amplitude * np.exp(
        -((x - x0_a - v_normal * t) ** 2) / (2 * sx**2) - (y - y0_a) ** 2 / (2 * sy**2)
    )


# Run and compare
dt_a = 0.5
t_total_a = 300.0
n_steps_a = int(t_total_a / dt_a)
compare_times = [0, 50, 100, 150, 200, 300]
results = {0.0: {"numerical": T_a.copy()}}

print(f"\nRunning {n_steps_a} steps...")
print(f"{'t (s)':>8s}  {'Peak (num)':>12s}  {'Peak (exact)':>12s}  {'L2 error':>10s}")
print("-" * 50)

# Compute initial analytical for comparison
T_exact_0 = T_analytical(grid_a.x_of_node, grid_a.y_of_node, 0.0)
L2_0 = np.sqrt(np.mean((T_a[channel_mask_a] - T_exact_0[channel_mask_a]) ** 2))
print(f"{0.0:8.1f}  {T_a.max():12.4f}  {T_exact_0.max():12.4f}  {L2_0:10.6f}")

for step in range(n_steps_a):
    t_sim = (step + 1) * dt_a

    rtd_a.run_one_step(dt_a)
    T_a[bank_mask_a] = T_bg_a

    if t_sim in compare_times:
        T_exact = T_analytical(grid_a.x_of_node, grid_a.y_of_node, t_sim)
        L2_err = np.sqrt(np.mean((T_a[channel_mask_a] - T_exact[channel_mask_a]) ** 2))
        results[t_sim] = {
            "numerical": T_a.copy(),
            "analytical": T_exact.copy(),
            "L2": L2_err,
        }
        print(
            f"{t_sim:8.1f}  {T_a[channel_mask_a].max():12.4f}  "
            f"{T_exact[channel_mask_a].max():12.4f}  {L2_err:10.6f}"
        )

print("\nDone!")

# ===== Cell Separator =====
fig, ax = plt.subplots(figsize=(14, 6))
center_row_a = nrows_a // 2
x_arr_a = np.arange(ncols_a) * dx_a
colors_a = plt.cm.viridis(np.linspace(0, 0.9, len(compare_times)))

for i, t_snap in enumerate(compare_times):
    # Numerical
    T_num = results[t_snap]["numerical"].reshape((nrows_a, ncols_a))
    ax.plot(
        x_arr_a,
        T_num[center_row_a, :],
        color=colors_a[i],
        lw=2,
        label=f"Numerical, t = {t_snap:.0f} s",
    )

    # Analytical (skip t=0 to avoid clutter — they overlap perfectly)
    if t_snap > 0:
        T_ex = results[t_snap]["analytical"].reshape((nrows_a, ncols_a))
        ax.plot(
            x_arr_a,
            T_ex[center_row_a, :],
            color=colors_a[i],
            lw=1.5,
            ls="--",
        )

# Dummy lines for legend
ax.plot([], [], "k-", lw=2, label="Numerical")
ax.plot([], [], "k--", lw=1.5, label="Analytical")

ax.set_xlabel("x (m)", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.set_title(
    "Centerline Temperature: Numerical vs. Analytical Solution",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=9, ncol=3, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, dx_a * (ncols_a - 1))
ax.axhline(T_bg_a, color="gray", ls=":", alpha=0.5)
plt.tight_layout()
plt.show()

# ===== Cell Separator =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

times_plot = sorted([t for t in results.keys() if t > 0])
peaks_num = [results[t]["numerical"][channel_mask_a].max() for t in times_plot]
peaks_exact = [results[t]["analytical"][channel_mask_a].max() for t in times_plot]
L2_errors = [results[t]["L2"] for t in times_plot]

# Peak temperature comparison
ax1.plot(times_plot, peaks_num, "bo-", lw=2, markersize=8, label="Numerical")
ax1.plot(times_plot, peaks_exact, "r^--", lw=2, markersize=8, label="Analytical")
ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Peak Temperature (°C)", fontsize=12)
ax1.set_title("Peak Temperature Over Time", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# L2 error
ax2.plot(times_plot, L2_errors, "ks-", lw=2, markersize=8)
ax2.set_xlabel("Time (s)", fontsize=12)
ax2.set_ylabel("L2 Error Norm (°C)", fontsize=12)
ax2.set_title("L2 Error: Numerical vs. Analytical", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary
print("Summary of numerical vs. analytical comparison:")
print(
    f"{'Time (s)':>10s}  {'Peak Num':>10s}  {'Peak Exact':>10s}  {'Rel Error':>10s}  {'L2':>10s}"
)
print("-" * 55)
for t in times_plot:
    pn = results[t]["numerical"][channel_mask_a].max()
    pe = results[t]["analytical"][channel_mask_a].max()
    re = abs(pn - pe) / (pe - T_bg_a) * 100
    print(f"{t:10.0f}  {pn:10.4f}  {pe:10.4f}  {re:9.2f}%  {results[t]['L2']:10.6f}")
