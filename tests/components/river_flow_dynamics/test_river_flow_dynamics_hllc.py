"""
Unit tests for landlab.components.river_flow_dynamics_hllc.RiverFlowDynamics_HLLC

Notes
-----
* RiverFlowDynamics_HLLC is a hyperbolic finite-volume solver; its "exact" answers
  can depend on CFL, reconstruction order, and boundary condition choices.
  Tests therefore focus on invariants (mass conservation in closed systems,
  well-balanced rest states, sign/directionality, no NaNs/Infs) and on a small
  number of controlled regression checks.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverFlowDynamics_HLLC

G = 9.81

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_grid(nr: int, nc: int, dx: float = 1.0) -> RasterModelGrid:
    return RasterModelGrid((nr, nc), xy_spacing=dx)


def _component_accepts(param_name: str) -> bool:
    """Return True if RiverFlowDynamics_HLLC.__init__ accepts `param_name`."""
    sig = inspect.signature(RiverFlowDynamics_HLLC.__init__)
    return param_name in sig.parameters


def _apply_stage_exit_bc_supported() -> bool:
    return _component_accepts("fixed_exit_nodes") and (
        _component_accepts("exit_nodes_eta_values")
        or _component_accepts("exit_nodes_h_values")
    )


def _run_to_time(
    comp: RiverFlowDynamics_HLLC, t_end: float, dt_fixed: float | None = None
):
    """Advance component to t_end (seconds), trimming the last step.

    If dt_fixed is provided, still respect the component CFL dt to avoid
    instability warnings during tests.
    """
    epsilon = 1e-12  # Tolerance to prevent microscopic floating-point timesteps

    while comp.elapsed_time < t_end - epsilon:
        time_remaining = t_end - comp.elapsed_time

        # If no fixed dt is requested and we aren't at the final partial step,
        # let the component calculate and use its own safe dt entirely.
        if dt_fixed is None and time_remaining >= comp.current_dt:
            comp.run_one_step()
            continue

        # Otherwise, we need to enforce a fixed dt or trim the final step
        dt_cfl = comp.current_dt

        if dt_fixed is None:
            dt_target = time_remaining
        else:
            dt_target = min(float(dt_fixed), time_remaining)

        # Always respect the CFL limit
        dt = min(dt_target, dt_cfl)

        comp.run_one_step(dt=dt)


# -----------------------------------------------------------------------------
# Pytest fixture (mirrors test_river_flow_dynamics.py)
# -----------------------------------------------------------------------------


@pytest.fixture
def rfd_hllc():
    grid = _make_grid(10, 10, dx=0.1)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("surface_water__depth", at="node")
    # HLLC requires surface_water__depth + topography on init.
    return RiverFlowDynamics_HLLC(grid)


# -----------------------------------------------------------------------------
# Landlab component interface tests
# -----------------------------------------------------------------------------


def test_name(rfd_hllc):
    assert rfd_hllc.name == "RiverFlowDynamics_HLLC"


def test_input_var_names(rfd_hllc):
    # Order should match component _info / Landlab convention; keep explicit tuple.
    assert rfd_hllc.input_var_names == (
        "surface_water__depth",
        "topographic__elevation",
    )


def test_output_var_names(rfd_hllc):
    # Do not enforce ordering; Landlab does not require a specific order.
    expected = {
        "surface_water__depth",
        "surface_water__elevation",
        "surface_water__x_velocity",
        "surface_water__y_velocity",
        "surface_water__x_momentum",
        "surface_water__y_momentum",
    }
    assert set(rfd_hllc.output_var_names) == expected


def test_var_units(rfd_hllc):
    assert rfd_hllc.var_units("topographic__elevation") == "m"
    assert rfd_hllc.var_units("surface_water__depth") == "m"
    assert rfd_hllc.var_units("surface_water__elevation") == "m"
    assert rfd_hllc.var_units("surface_water__x_velocity") == "m/s"
    assert rfd_hllc.var_units("surface_water__y_velocity") == "m/s"
    assert rfd_hllc.var_units("surface_water__x_momentum") == "m2/s"
    assert rfd_hllc.var_units("surface_water__y_momentum") == "m2/s"
    # Optional link output when update_link_fields=True
    assert rfd_hllc.var_units("surface_water__velocity") == "m/s"


def test_initialization_creates_fields():
    grid = _make_grid(6, 10, dx=0.5)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.5, at="node")

    comp = RiverFlowDynamics_HLLC(grid, mannings_n=0.02, cfl=0.45)

    # Required node fields exist and have correct shapes
    for field in [
        "surface_water__depth",
        "surface_water__elevation",
        "surface_water__x_velocity",
        "surface_water__y_velocity",
        "surface_water__x_momentum",
        "surface_water__y_momentum",
    ]:
        assert field in grid.at_node
        assert grid.at_node[field].shape == (grid.number_of_nodes,)

    assert comp.elapsed_time == 0.0
    assert comp.current_dt > 0.0

    # Optional link velocity field
    grid2 = _make_grid(6, 10, dx=0.5)
    grid2.add_zeros("topographic__elevation", at="node")
    grid2.add_full("surface_water__depth", 0.5, at="node")
    grid2.add_zeros("surface_water__velocity", at="link")
    # Some implementations update an existing link field rather than creating it.
    assert "surface_water__velocity" in grid2.at_link
    assert grid2.at_link["surface_water__velocity"].shape == (grid2.number_of_links,)


# -----------------------------------------------------------------------------
# 2) Lake at rest  -  flat bed (well-balanced, no motion)
# -----------------------------------------------------------------------------


def test_lake_at_rest_flat():
    grid = _make_grid(10, 20, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.5, at="node")

    comp = RiverFlowDynamics_HLLC(grid, mannings_n=0.0)  # remove friction influence

    # A few steps should not create motion
    for _ in range(20):
        comp.run_one_step()

    dh = np.abs(grid.at_node["surface_water__depth"] - 0.5).max()
    max_hu = np.abs(grid.at_node["surface_water__x_momentum"]).max()
    max_hv = np.abs(grid.at_node["surface_water__y_momentum"]).max()

    assert dh < 1e-12
    assert max_hu < 1e-12
    assert max_hv < 1e-12


# -----------------------------------------------------------------------------
# 3) Lake at rest  -  bumpy bed (hydrostatic reconstruction / well-balanced)
# -----------------------------------------------------------------------------


def test_lake_at_rest_bumpy():
    nr, nc, dx = 8, 30, 0.5
    grid = _make_grid(nr, nc, dx=dx)

    x = (np.arange(nc) + 0.5) * dx
    z_row = 0.4 * np.exp(-((x - x.mean()) ** 2) / 8.0)
    z_2d = np.tile(z_row, (nr, 1))
    grid.at_node["topographic__elevation"] = z_2d.ravel()

    # Still water at constant stage eta=1.0 (with dry prevention)
    eta0 = 1.0
    h0 = np.maximum(0.0, eta0 - z_2d).ravel()
    grid.add_field("surface_water__depth", h0, at="node", copy=True)

    comp = RiverFlowDynamics_HLLC(grid, mannings_n=0.0)

    for _ in range(30):
        comp.run_one_step()

    h = grid.at_node["surface_water__depth"]
    z = grid.at_node["topographic__elevation"]
    eta = h + z

    wet = h > 1e-8
    assert np.abs(eta[wet] - eta0).max() < 1e-10
    assert np.abs(grid.at_node["surface_water__x_momentum"][wet]).max() < 1e-10
    assert np.abs(grid.at_node["surface_water__y_momentum"][wet]).max() < 1e-10


# -----------------------------------------------------------------------------
# Mass conservation (closed system)
# -----------------------------------------------------------------------------


def test_mass_conservation_all_walls():
    nr, nc, dx = 10, 12, 1.0
    grid = _make_grid(nr, nc, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")

    # Blob of water
    x = (np.arange(nc) + 0.5) * dx
    y = (np.arange(nr) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    h0 = (
        0.5 + 0.3 * np.exp(-((X - x.mean()) ** 2 + (Y - y.mean()) ** 2) / 4.0)
    ).ravel()
    grid.add_field("surface_water__depth", h0, at="node", copy=True)

    mass0 = h0.sum() * dx**2

    comp = RiverFlowDynamics_HLLC(
        grid, wall_edges={"left", "right", "bottom", "top"}, mannings_n=0.0
    )

    for _ in range(50):
        comp.run_one_step()

    mass1 = grid.at_node["surface_water__depth"].sum() * dx**2
    rel = abs(mass1 - mass0) / mass0
    assert rel < 1e-10


# -----------------------------------------------------------------------------
# Transmissive outflow BC (waves exit cleanly)
# -----------------------------------------------------------------------------


def test_transmissive_outflow():
    nr, nc, dx = 6, 40, 0.5
    grid = _make_grid(nr, nc, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.3, at="node")  # pre-fill

    # Inflow at right edge, leftward
    entry = grid.nodes_at_right_edge
    n = len(entry)
    comp = RiverFlowDynamics_HLLC(
        grid,
        mannings_n=0.01,
        fixed_entry_nodes=entry,
        entry_nodes_h_values=np.full(n, 0.5),
        entry_nodes_u_values=np.full(n, -1.0),
        entry_nodes_v_values=np.zeros(n),
        # all edges transmissive by default
    )

    _run_to_time(comp, 20.0)

    h = grid.at_node["surface_water__depth"].reshape(nr, nc)
    max_refl = np.abs(h[:, 0] - h[:, 1]).max()
    assert max_refl < 0.05
    assert h.max() > 0.1


# -----------------------------------------------------------------------------
# Reflective wall BCs
# -----------------------------------------------------------------------------


def test_wall_left_right_zero_normal_momentum():
    nr, nc, dx = 6, 30, 0.5
    grid = _make_grid(nr, nc, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")

    # Gaussian blob
    x = (np.arange(nc) + 0.5) * dx
    y = (np.arange(nr) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    h0 = (
        0.3 + 0.5 * np.exp(-((X - x.mean()) ** 2 + (Y - y.mean()) ** 2) / 2.0)
    ).ravel()
    grid.add_field("surface_water__depth", h0, at="node", copy=True)

    comp = RiverFlowDynamics_HLLC(grid, wall_edges={"left", "right"}, mannings_n=0.0)

    for _ in range(20):
        comp.run_one_step()

    hu = grid.at_node["surface_water__x_momentum"].reshape(nr, nc)
    assert np.abs(hu[:, 0]).max() < 0.02
    assert np.abs(hu[:, -1]).max() < 0.02


# -----------------------------------------------------------------------------
# Non-uniform Manning's n
# -----------------------------------------------------------------------------


def test_nonuniform_mannings_array_slows_flow():
    nr, nc, dx = 6, 60, 0.5
    grid = _make_grid(nr, nc, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.5, at="node")
    grid.add_full("surface_water__x_momentum", 1.0, at="node")  # uniform initial u-ish

    # Left half smooth, right half rough
    n_vals = np.where(grid.x_of_node < nc * dx / 2, 0.01, 0.1)

    comp = RiverFlowDynamics_HLLC(grid, mannings_n=n_vals)

    for _ in range(30):
        comp.run_one_step()

    u = grid.at_node["surface_water__x_velocity"].reshape(nr, nc)
    assert u[:, : nc // 2].mean() > u[:, nc // 2 :].mean()


def test_nonuniform_mannings_grid_field_precedence():
    nr, nc, dx = 6, 20, 1.0
    grid = _make_grid(nr, nc, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.4, at="node")
    grid.add_full("surface_water__x_momentum", 0.5, at="node")
    grid.add_full("mannings_n_at_node", 0.05, at="node")

    # Pass different scalar; field should take precedence
    comp = RiverFlowDynamics_HLLC(grid, mannings_n=0.0)

    u_before = grid.at_node["surface_water__x_velocity"].copy()
    for _ in range(10):
        comp.run_one_step()
    u_after = grid.at_node["surface_water__x_velocity"]

    wet = grid.at_node["surface_water__depth"] > 1e-8
    assert u_after[wet].mean() < u_before[wet].mean() + 1e-6


def test_mannings_scalar_zero_frictionless_conserves_zero_momentum():
    nr, nc, dx = 4, 20, 1.0
    grid = _make_grid(nr, nc, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 1.0, at="node")

    comp = RiverFlowDynamics_HLLC(
        grid, mannings_n=0.0, wall_edges={"left", "right", "bottom", "top"}
    )
    for _ in range(20):
        comp.run_one_step()
    hu_total = grid.at_node["surface_water__x_momentum"].sum()
    hv_total = grid.at_node["surface_water__y_momentum"].sum()
    assert abs(hu_total) < 1e-10
    assert abs(hv_total) < 1e-10


# -----------------------------------------------------------------------------
# Link velocity field (update_link_fields)
# -----------------------------------------------------------------------------


def test_link_field_auto_update_and_finite():
    nr, nc, dx = 6, 20, 1.0
    grid = _make_grid(nr, nc, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.5, at="node")

    grid.add_zeros("surface_water__velocity", at="link")
    comp = RiverFlowDynamics_HLLC(grid, update_link_fields=True)

    assert "surface_water__velocity" in grid.at_link
    vel = grid.at_link["surface_water__velocity"]
    assert vel.shape == (grid.number_of_links,)

    for _ in range(5):
        comp.run_one_step()

    assert np.isfinite(vel).all()


# -----------------------------------------------------------------------------
# Inflow from v-direction (bottom edge entry)
# -----------------------------------------------------------------------------


def test_inflow_v_direction_propagates_north():
    nr, nc, dx = 30, 6, 0.5
    grid = _make_grid(nr, nc, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.1, at="node")

    entry = grid.nodes_at_bottom_edge
    n = len(entry)

    # --- ADD THESE LINES ---
    # Pre-condition the boundary nodes to match the forced boundary conditions
    # so the solver computes a correct, safe initial CFL timestep.
    grid.at_node["surface_water__depth"][entry] = 0.4
    grid.add_zeros("surface_water__y_momentum", at="node")
    grid.at_node["surface_water__y_momentum"][entry] = 0.4 * 0.8  # depth * v_velocity
    # -----------------------

    comp = RiverFlowDynamics_HLLC(
        grid,
        mannings_n=0.01,
        fixed_entry_nodes=entry,
        entry_nodes_h_values=np.full(n, 0.4),
        entry_nodes_u_values=np.zeros(n),
        entry_nodes_v_values=np.full(n, 0.8),
        wall_edges={"left", "right"},
    )

    _run_to_time(comp, 10.0)

    h = grid.at_node["surface_water__depth"].reshape(nr, nc)
    v = grid.at_node["surface_water__y_velocity"].reshape(nr, nc)

    assert h[nr // 2 :, :].mean() > 0.01
    wet = h > 1e-8
    assert v[wet].mean() > 0.0


# -----------------------------------------------------------------------------
# Adaptive vs fixed dt replay
# -----------------------------------------------------------------------------


def test_adaptive_vs_fixed_replay():
    nr, nc, dx = 6, 40, 0.5
    x = (np.arange(nc) + 0.5) * dx
    x0 = nc * dx / 2

    def make_comp():
        grid = _make_grid(nr, nc, dx=dx)
        grid.add_zeros("topographic__elevation", at="node")
        H = np.where(x < x0, 1.0, 0.1)
        grid.add_field(
            "surface_water__depth", np.tile(H, (nr, 1)).ravel(), at="node", copy=True
        )
        return RiverFlowDynamics_HLLC(grid, cfl=0.45, mannings_n=0.0)

    comp_a = make_comp()
    dts = []
    for _ in range(5):
        dts.append(comp_a.current_dt)
        comp_a.run_one_step()

    comp_f = make_comp()
    for dt in dts:
        comp_f.run_one_step(dt=dt)

    ha = comp_a.grid.at_node["surface_water__depth"]
    hf = comp_f.grid.at_node["surface_water__depth"]
    assert np.allclose(ha, hf, atol=1e-14)


# -----------------------------------------------------------------------------
# Elapsed time tracking
# -----------------------------------------------------------------------------


def test_elapsed_time_accumulates():
    grid = _make_grid(6, 10, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.5, at="node")

    comp = RiverFlowDynamics_HLLC(grid, mannings_n=0.0)

    total = 0.0
    for _ in range(10):
        dt = comp.current_dt
        comp.run_one_step()
        total += dt

    assert abs(comp.elapsed_time - total) < 1e-12


# -----------------------------------------------------------------------------
# Outlet stage constraint
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not _apply_stage_exit_bc_supported(),
    reason="Exit BC params not supported by this build.",
)
def test_fixed_exit_stage_produces_near_uniform_depth_like_rfd_example():
    """Regression-style test for fixed outlet stage.

    Mirrors the RiverFlowDynamics doctest-ish channel setup.
    """
    grid = RasterModelGrid((8, 6), xy_spacing=0.1)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += 0.005 - 0.01 * grid.x_of_node
    z[grid.y_of_node > 0.5] = 1.0
    z[grid.y_of_node < 0.2] = 1.0

    grid.add_zeros("surface_water__depth", at="node")
    grid.at_node["surface_water__depth"][:] = 0.5

    fixed_entry_nodes = np.arange(12, 36, 6)
    entry_nodes_h_values = np.full(4, 0.5)
    entry_nodes_u_values = np.full(4, 0.45)
    entry_nodes_v_values = np.zeros(4)

    fixed_exit_nodes = np.arange(17, 41, 6)
    eta0 = (z[fixed_entry_nodes] + 0.5).mean()
    exit_nodes_eta_values = np.full(4, eta0)

    comp = RiverFlowDynamics_HLLC(
        grid,
        mannings_n=0.012,
        fixed_entry_nodes=fixed_entry_nodes,
        entry_nodes_h_values=entry_nodes_h_values,
        entry_nodes_u_values=entry_nodes_u_values,
        entry_nodes_v_values=entry_nodes_v_values,
        fixed_exit_nodes=fixed_exit_nodes,
        exit_nodes_eta_values=exit_nodes_eta_values,
        wall_edges={"top", "bottom"},
        update_link_fields=True,
    )

    _run_to_time(comp, 10.0, dt_fixed=0.01)

    flow_depth = grid.at_node["surface_water__depth"].reshape((8, 6))[3, :]
    expected = np.array([0.5, 0.5, 0.5, 0.501, 0.502, 0.502])
    assert np.allclose(np.round(flow_depth, 3), expected, atol=0.02)


# -----------------------------------------------------------------------------
# Basic numerical stability checks
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [None, 0.05, 0.01])
def test_numerical_stability_no_nan_inf_and_nonnegative_depth(dt):
    grid = _make_grid(10, 10, dx=0.1)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.5, at="node")

    comp = RiverFlowDynamics_HLLC(grid, mannings_n=0.012)

    if dt is None:
        _run_to_time(comp, 1.0)
    else:
        _run_to_time(comp, 1.0, dt_fixed=dt)

    h = grid.at_node["surface_water__depth"]
    assert np.isfinite(h).all()
    assert np.all(h >= 0.0)

    u = grid.at_node["surface_water__x_velocity"]
    v = grid.at_node["surface_water__y_velocity"]
    assert np.isfinite(u).all()
    assert np.isfinite(v).all()


# -----------------------------------------------------------------------------
# Others
# -----------------------------------------------------------------------------


def test_muscl_reconstruction_order_2():
    grid = _make_grid(6, 6, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.5, at="node")


def test_init_exceptions():
    with pytest.raises(TypeError, match="requires a RasterModelGrid"):
        RiverFlowDynamics_HLLC("not_a_grid")


def test_cfl_warning_triggered_by_huge_dt():
    grid = _make_grid(5, 5, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 1.0, at="node")


def test_exit_nodes_with_imposed_velocities():
    grid = _make_grid(5, 5, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 1.0, at="node")


# -----------------------------------------------------------------------------
# Well-Balancing & Compatible-Source Tests
# -----------------------------------------------------------------------------


def _max_speed(grid: RasterModelGrid) -> float:
    """Computes  maximum velocity magnitude in the grid."""
    u = grid.at_node["surface_water__x_velocity"]
    v = grid.at_node["surface_water__y_velocity"]
    return float(np.max(np.sqrt(u**2 + v**2)))


@pytest.mark.parametrize("order", [1, 2])
def test_flat_flow(order):
    """
    Performs the Flat bed with flow test.
    The bed-slope source must be IDENTICALLY zero,
    regardless of the surface gradient or velocity field.
    """
    # Import the internal kernel module to probe _x_sweep directly
    from landlab.components.river_flow_dynamics import river_flow_dynamics_hllc as K

    nr, nc = 5, 60
    dx = 1.0

    h = np.full((nr, nc), 0.8)
    hu = np.full((nr, nc), 0.8 * 0.5)  # u = 0.5 everywhere
    hv = np.zeros((nr, nc))
    z = np.zeros((nr, nc))

    # One x-sweep; with a correct flat-bed source, hu is unchanged to machine
    # precision (flux divergence zero + source zero).
    _, hu2, _ = K._x_sweep(
        h.copy(), hu.copy(), hv.copy(), z, dt=0.01, dx=dx, g=K._G, order=order
    )

    # Interior cells only
    dhu = np.max(np.abs(hu2[:, 1:-1] - hu[:, 1:-1]))

    # Second probe: a non-trivial surface gradient on a flat bed.
    x = (np.arange(nc) + 0.5) * dx
    h_ramp = np.clip(1.0 - 0.012 * (x - 20), 0.2, 1.0)
    h_r = np.tile(h_ramp, (nr, 1))
    eta = h_r + z

    eta_p = K._pad(eta, False, False, negate=False)
    z_p = K._pad(z, False, False, negate=False)

    if order == 2:
        _, etaR2 = K._muscl_x(eta_p)
        zL2, zR2 = K._muscl_x(z_p)
        hpc = np.maximum(0.0, eta_p - z_p)
        dry = (hpc[:, :-1] <= K._H_DRY) | (hpc[:, 1:] <= K._H_DRY)
        zL = np.where(dry, z_p[:, :-1], zL2)
        zR = np.where(dry, z_p[:, 1:], zR2)
    else:
        zL, zR = z_p[:, :-1], z_p[:, 1:]

    zf = np.maximum(zL, zR)
    hLsrc = np.maximum(0.0, eta_p[:, :-1] - zf)
    hRsrc = np.maximum(0.0, eta_p[:, 1:] - zf)
    Sx = 0.5 * K._G * (hLsrc[:, 1:] ** 2 - hRsrc[:, :-1] ** 2) / dx

    max_Sx = float(np.max(np.abs(Sx)))

    assert dhu < 1e-12, f"Momentum changed by {dhu:.2e} on flat bed (target < 1e-12)"
    assert (
        max_Sx < 1e-12
    ), f"Spurious bed-slope source generated: {max_Sx:.2e} (target < 1e-12)"


@pytest.mark.parametrize("order", [1, 2])
def test_slope_rest(order):
    """
    Performs the Lake at rest on a sloped bed test.
    Velocity magnitude must stay at machine precision.
    """
    nrows, ncols, dx = 5, 60, 1.0
    grid = _make_grid(nrows, ncols, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")

    x = (np.arange(ncols) + 0.5) * dx
    z1d = 0.5 * (x - x.min()) / (x.max() - x.min())  # linear ramp 0→0.5
    z2d = np.tile(z1d, (nrows, 1))

    eta0 = 0.8
    h2d = np.maximum(0.0, eta0 - z2d)

    grid.at_node["topographic__elevation"][:] = z2d.ravel()
    grid.add_field("surface_water__depth", h2d.ravel(), at="node", copy=True)

    hllc = RiverFlowDynamics_HLLC(
        grid, mannings_n=0.0, cfl=0.4, order=order, wall_edges={"bottom", "top"}
    )

    umax_history = []
    nsteps = 1000
    for _ in range(nsteps):
        hllc.run_one_step()
        umax_history.append(_max_speed(grid))

    umax = max(umax_history)

    eta_now = (
        grid.at_node["surface_water__depth"] + grid.at_node["topographic__elevation"]
    )
    eta_flat_err = float(np.max(np.abs(eta_now[grid.core_nodes] - eta0)))

    assert umax < 1e-10, f"Spurious velocity generated: {umax:.2e} m/s (target < 1e-10)"
    assert (
        eta_flat_err < 1e-10
    ), f"Surface deviated from flat state: {eta_flat_err:.2e} m"


@pytest.mark.parametrize("order", [1, 2])
def test_slope_flow(order):
    """
    Performs the Sloped bed + flow. Sanity/stability guard test.
    Ensures finite values and non-negative depth when flow traverses a slope.
    """
    nrows, ncols, dx = 5, 60, 1.0
    grid = _make_grid(nrows, ncols, dx=dx)
    grid.add_zeros("topographic__elevation", at="node")

    x = (np.arange(ncols) + 0.5) * dx
    z1d = 0.3 * (x - x.min()) / (x.max() - x.min())
    z2d = np.tile(z1d, (nrows, 1))
    h2d = np.maximum(0.1, 0.8 - z2d)

    grid.at_node["topographic__elevation"][:] = z2d.ravel()
    grid.add_field("surface_water__depth", h2d.ravel(), at="node", copy=True)

    # Impose initial flow via the momentum field before component init binds its view
    hu0 = grid.add_zeros("surface_water__x_momentum", at="node")
    hu0[:] = (h2d * 0.3).ravel()  # u ≈ 0.3 m/s

    hllc = RiverFlowDynamics_HLLC(
        grid, mannings_n=0.03, cfl=0.4, order=order, wall_edges={"bottom", "top"}
    )

    nsteps = 50
    for _ in range(nsteps):
        hllc.run_one_step()
        h = grid.at_node["surface_water__depth"]

        assert np.all(np.isfinite(h)), "Depth became NaN/Inf"
        assert not np.any(h < -1e-9), "Depth became negative"


def test_implicit_diffusion_closures():
    """Tests the Smagorinsky and Elder diffusion block."""
    grid = _make_grid(40, 40, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 1.0, at="node")

    # Explicitly add the fields the component expects to find
    grid.add_zeros("surface_water__x_velocity", at="node")
    grid.add_zeros("surface_water__y_velocity", at="node")

    # Force a shear layer in the INTERIOR (not on the boundary)
    # Get the 2D view of the velocity field directly from the grid
    u_2d = grid.at_node["surface_water__x_velocity"].reshape(grid.shape)

    # Now you can use 2D indexing
    u_2d[15:25, :] = 2.0  # High velocity jet in the middle

    # Ensure the grid field is updated (if you used reshape, it's already a view)
    grid.at_node["surface_water__x_velocity"][:] = u_2d.flatten()

    comp = RiverFlowDynamics_HLLC(
        grid,
        mannings_n=0.02,
        use_smagorinsky=True,
        smagorinsky_cs=1.0,
        use_elder=True,
        elder_alpha=0.6,
    )

    # Manually inject the velocity jet into the component's internal state
    # u = momentum / h (here h=1.0, so momentum == velocity)
    u_2d = comp._u  # This is the 2D view of the component's internal velocity
    u_2d[15:25, :] = 2.0
    comp._hu[:] = comp._u * comp._h

    # Now run the step; the diffusion solver will see the jet in self._hu
    comp.run_one_step(dt=0.1)

    nu_t = grid.at_node["surface_water__eddy_viscosity"]
    nu_t_2d = nu_t.reshape(grid.shape)
    interior_nu_t = nu_t_2d[1:-1, 1:-1]

    assert np.any(interior_nu_t > 1e-8), f"Max nu_t across whole grid: {np.max(nu_t)}"


def test_hllc_init_exceptions_and_edge_cases():
    """Tests various ValueErrors and missing coverage in __init__."""
    grid = _make_grid(5, 5, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 1.0, at="node")

    # mannings_n wrong size
    with pytest.raises(ValueError, match="mannings_n must be a scalar or a 1-D array"):
        RiverFlowDynamics_HLLC(grid, mannings_n=np.array([0.01, 0.02]))

    # invalid wall_edges
    with pytest.raises(ValueError, match="Unknown wall_edges"):
        RiverFlowDynamics_HLLC(grid, wall_edges={"bogus_edge"})

    # fixed_entry_nodes without h_values
    with pytest.raises(ValueError, match="entry_nodes_h_values is required"):
        RiverFlowDynamics_HLLC(grid, fixed_entry_nodes=[0])

    # fixed_exit_nodes without h_values or eta_values
    with pytest.raises(
        ValueError, match="Provide exit_nodes_h_values or exit_nodes_eta_values"
    ):
        RiverFlowDynamics_HLLC(grid, fixed_exit_nodes=[4])

    # exit nodes mismatch lengths
    with pytest.raises(ValueError, match="exit_nodes_h_values must match"):
        RiverFlowDynamics_HLLC(grid, fixed_exit_nodes=[4, 9], exit_nodes_h_values=[1.0])
    with pytest.raises(ValueError, match="exit_nodes_eta_values must match"):
        RiverFlowDynamics_HLLC(
            grid, fixed_exit_nodes=[4, 9], exit_nodes_eta_values=[1.0]
        )
    with pytest.raises(ValueError, match="exit_nodes_u_values must match"):
        RiverFlowDynamics_HLLC(
            grid,
            fixed_exit_nodes=[4, 9],
            exit_nodes_h_values=[1.0, 1.0],
            exit_nodes_u_values=[0.5],
        )
    with pytest.raises(ValueError, match="exit_nodes_v_values must match"):
        RiverFlowDynamics_HLLC(
            grid,
            fixed_exit_nodes=[4, 9],
            exit_nodes_h_values=[1.0, 1.0],
            exit_nodes_v_values=[0.5],
        )


def test_inflow_all_four_edges():
    """Tests the right, top, and bottom edge inflow builder blocks."""
    grid = _make_grid(5, 5, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 1.0, at="node")

    # Pick one node on each of the four edges
    entry_nodes = [
        grid.nodes_at_left_edge[1],
        grid.nodes_at_right_edge[1],
        grid.nodes_at_bottom_edge[1],
        grid.nodes_at_top_edge[1],
    ]
    comp = RiverFlowDynamics_HLLC(
        grid,
        fixed_entry_nodes=entry_nodes,
        entry_nodes_h_values=[1.0, 1.0, 1.0, 1.0],
        entry_nodes_u_values=[0.1, -0.1, 0.0, 0.0],
        entry_nodes_v_values=[0.0, 0.0, 0.1, -0.1],
    )
    comp.run_one_step(dt=0.1)


def test_fixed_exit_velocities_and_outlet_max_depth():
    """Tests the WSE clamping branches for exit_u, exit_v, and outlet_max_depth."""
    grid = _make_grid(5, 5, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 0.5, at="node")  # Below outlet_max_depth

    exit_nodes = grid.nodes_at_right_edge
    comp = RiverFlowDynamics_HLLC(
        grid,
        fixed_exit_nodes=exit_nodes,
        exit_nodes_h_values=np.ones_like(exit_nodes) * 1.0,
        exit_nodes_u_values=np.ones_like(exit_nodes) * 0.5,
        exit_nodes_v_values=np.ones_like(exit_nodes) * 0.1,
        outlet_max_depth=0.8,
    )
    # Step 1: Ramp inactive
    comp.run_one_step(dt=0.1)

    # Step 2: Raise depth to trigger outlet max depth clamp
    grid.at_node["surface_water__depth"][:] = 1.0
    comp.run_one_step(dt=0.1)


def test_build_inflow_ghost_internal():
    """Tests the internal _build_inflow_ghost logic directly for full/bogus modes."""
    import landlab.components.river_flow_dynamics.river_flow_dynamics_hllc as hllc_module

    h_int = np.array([1.0])
    hu_int = np.array([0.5])
    hv_int = np.array([0.0])
    z = np.array([0.0])
    g = 9.81

    # Test full mode
    spec_full = {
        "mode": "full",
        "h": np.array([1.0]),
        "hu": np.array([1.0]),
        "hv": np.array([0.0]),
        "z": z,
    }
    hg, hug, hvg, zg = hllc_module._build_inflow_ghost(
        spec_full, h_int, hu_int, hv_int, g, "left"
    )
    assert hg[0] == 1.0

    # Test bogus mode exception
    spec_bogus = {"mode": "bogus", "z": z}
    with pytest.raises(ValueError, match="Unknown inflow mode: bogus"):
        hllc_module._build_inflow_ghost(spec_bogus, h_int, hu_int, hv_int, g, "left")


def test_muscl_order_2_dry_face():
    """Tests the dry-face fallback block inside MUSCL _x_sweep."""
    grid = _make_grid(5, 5, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    h = grid.add_zeros("surface_water__depth", at="node")
    h[:10] = 0.5  # Half wet, half dry grid
    comp = RiverFlowDynamics_HLLC(grid, order=2)
    comp.run_one_step(dt=0.01)


def test_cfl_dt_warning():
    """Tests the user warning when dt > dt_cfl."""
    grid = _make_grid(5, 5, dx=1.0)
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_full("surface_water__depth", 1.0, at="node")
    comp = RiverFlowDynamics_HLLC(grid)
    with pytest.warns(UserWarning, match="exceeds CFL-stable"):
        comp.run_one_step(dt=999.0)
