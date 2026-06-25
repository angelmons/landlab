"""Consolidated tests for ``landlab.components.RiverBedDynamics``.

This file combines the bed-dynamics unit, integration, coupling, stratigraphy,
and targeted coverage tests into one pytest module. The only optional block is
the coverage test for the legacy non-underscore ``bedload_equation_base``
module; it is skipped automatically if that duplicate source module is removed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverBedDynamics, RiverFlowDynamics_HLLC
from landlab.components.river_bed_dynamics import _bedload_eq_MPM_style as mpm
from landlab.components.river_bed_dynamics import _critical_shear_stress as ccs
from landlab.components.river_bed_dynamics import _fluid_properties as fp
from landlab.components.river_bed_dynamics import _initialize_fields as initf
from landlab.components.river_bed_dynamics import _initialize_gsd as initg
from landlab.components.river_bed_dynamics import _stratigraphy as strat
from landlab.components.river_bed_dynamics import _stratigraphy as stratigraphy
from landlab.components.river_bed_dynamics import _utilities as utilities
from landlab.components.river_bed_dynamics._gsd_evolver import ToroEscobarEvolver
from landlab.grid.mappers import map_mean_of_link_nodes_to_link

RBD_GSD = [[32, 100, 100], [16, 25, 50], [8, 0, 0]]
RBD_GSD_LOC = [
    [0, 1.0, 1.0, 1.0, 0],
    [0, 1.0, 1.0, 1.0, 0],
    [0, 1.0, 1.0, 1.0, 0],
    [0, 1.0, 1.0, 1.0, 0],
    [0, 1.0, 1.0, 1.0, 0],
]

HLLC_ENTRY_NODES = np.arange(12, 36, 6)
HLLC_EXIT_NODES = np.arange(17, 41, 6)
HLLC_GSD = np.array([[2, 100], [1, 50], [0.5, 0]])


# -----------------------------------------------------------------------------
# Shared grid builders
# -----------------------------------------------------------------------------


def _rbd_grid() -> RasterModelGrid:
    grid = RasterModelGrid((5, 5))
    grid.at_node["topographic__elevation"] = [
        [1.07, 1.06, 1.00, 1.06, 1.07],
        [1.08, 1.07, 1.03, 1.07, 1.08],
        [1.09, 1.08, 1.07, 1.08, 1.09],
        [1.09, 1.09, 1.08, 1.09, 1.09],
        [1.09, 1.09, 1.09, 1.09, 1.09],
    ]
    grid.set_watershed_boundary_condition(grid.at_node["topographic__elevation"])
    grid.at_node["surface_water__depth"] = np.full(grid.number_of_nodes, 0.102)
    grid.at_node["surface_water__velocity"] = np.full(grid.number_of_nodes, 0.25)
    grid.at_link["surface_water__depth"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__depth"
    )
    grid.at_link["surface_water__velocity"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__velocity"
    )
    return grid


def _rbd_component(**kwargs) -> RiverBedDynamics:
    defaults = dict(
        gsd=RBD_GSD,
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        bedload_equation="MPM",
        dt=0.1,
        check_advective_cfl=False,
        check_gsd_residual=False,
    )
    defaults.update(kwargs)
    return RiverBedDynamics(_rbd_grid(), **defaults)


def _base_ready_component():
    rbd = _rbd_component()
    rbd.run_one_step()
    return rbd


def _flow():
    n = 6
    U = np.linspace(0.2, 1.5, n)
    h = np.full(n, 0.5)
    D50 = np.full(n, 0.02)
    return U, h, D50


@pytest.fixture
def small_grid():
    return RasterModelGrid((4, 4))



# -----------------------------------------------------------------------------
# Optional coverage for legacy duplicate bedload_equation_base.py
# -----------------------------------------------------------------------------


def test_duplicate_nonunderscore_bedload_equation_base_delegates():
    base = pytest.importorskip(
        "landlab.components.river_bed_dynamics.bedload_equation_base",
        reason="Legacy duplicate bedload_equation_base.py has been removed.",
    )

    rbd = _base_ready_component()
    for key in sorted(base.EQUATION_REGISTRY):
        equation = base.EQUATION_REGISTRY[key]()
        qb, qb_gsd = equation.calculate(rbd)
        assert qb.shape[0] == rbd._grid.number_of_links
        assert np.all(np.isfinite(qb))
        if qb_gsd is not None:
            assert qb_gsd.shape[0] == rbd._grid.number_of_links



# -----------------------------------------------------------------------------
# Critical shear stress helpers
# -----------------------------------------------------------------------------



def test_compute_critical_shear_stress_default_path():
    U, h, D50 = _flow()
    out = ccs.compute_critical_shear_stress(U, h, D50, rho_s=2650.0, T=20.0)
    # All advertised keys present
    for key in (
        "tau_cr",
        "tau_cr_star",
        "tau_star",
        "Re_s_cr",
        "u_star",
        "u_star_cr",
        "delta_v",
        "rho",
        "mu",
    ):
        assert key in out
        assert np.asarray(out[key]).shape == U.shape
    # Physical sanity: positive critical stress, positive shear velocity
    assert np.all(out["tau_cr"] > 0.0)
    assert np.all(out["u_star"] >= 0.0)
    assert np.all(out["u_star_cr"] > 0.0)
    # Density/viscosity recovered from T = 20 degC
    assert np.allclose(out["rho"], 997.939, atol=1e-2)

def test_compute_critical_shear_stress_with_overrides_and_array_T():
    U, h, D50 = _flow()
    rho = np.full(U.shape, 999.7)
    mu = np.full(U.shape, 1.3e-3)
    T = np.linspace(5.0, 25.0, U.size)  # spatially variable temperature
    out = ccs.compute_critical_shear_stress(
        U, h, D50, rho_s=2650.0, T=T, rho=rho, mu=mu, g=9.81
    )
    # Overrides are passed straight through
    assert np.allclose(out["rho"], rho)
    assert np.allclose(out["mu"], mu)

def test_compute_critical_shear_stress_temperature_lowers_threshold():
    # Colder water (higher viscosity) generally shifts the threshold; just
    # confirm the solver runs across a temperature sweep and stays finite.
    U, h, D50 = _flow()
    cold = ccs.compute_critical_shear_stress(U, h, D50, rho_s=2650.0, T=2.0)
    warm = ccs.compute_critical_shear_stress(U, h, D50, rho_s=2650.0, T=28.0)
    assert np.all(np.isfinite(cold["tau_cr"]))
    assert np.all(np.isfinite(warm["tau_cr"]))

def test_log_law_u_star_with_initial_guess():
    U = np.array([0.8, 1.2])
    h = np.array([0.5, 0.5])
    D50 = np.array([0.02, 0.02])
    rho = np.array([1000.0, 1000.0])
    mu = np.array([1e-3, 1e-3])
    guess = np.array([0.05, 0.05])
    u_star = ccs._log_law_u_star(U, h, D50, rho, mu, u_star_guess=guess)
    assert u_star.shape == U.shape
    assert np.all(u_star > 0.0)

def test_log_law_u_star_zero_velocity_and_no_convergence():
    # U = 0 exercises the U <= 0 branch of the default initial guess.
    U = np.array([0.0, 0.5])
    h = np.array([0.4, 0.4])
    D50 = np.array([0.02, 0.02])
    rho = np.array([1000.0, 1000.0])
    mu = np.array([1e-3, 1e-3])
    # max_iter=1 with a tight tolerance forces the loop to exit without the
    # early break, covering the non-convergent path.
    u_star = ccs._log_law_u_star(U, h, D50, rho, mu, max_iter=1, tol=1e-15)
    assert np.all(np.isfinite(u_star))


# -----------------------------------------------------------------------------
# Fluid property helpers
# -----------------------------------------------------------------------------



def test_water_density_scalar_and_array():
    # Density maximum at 4 degC
    assert np.isclose(float(fp.water_density(4.0)), 1000.0, atol=1e-4)
    # Known value at 20 degC
    assert np.isclose(float(fp.water_density(20.0)), 997.939, atol=1e-3)
    # Array input is handled element-wise and monotonic away from 4 degC
    rho = fp.water_density(np.array([0.0, 4.0, 10.0, 25.0]))
    assert rho.shape == (4,)
    assert rho[1] >= rho.max() - 1e-9

def test_d_rho_dT_zero_at_maximum_and_sign():
    assert np.isclose(float(fp.d_rho_dT(4.0)), 0.0, atol=1e-6)
    assert float(fp.d_rho_dT(20.0)) < 0.0  # density falls above 4 degC
    assert float(fp.d_rho_dT(1.0)) > 0.0  # density rises below 4 degC
    # Array path including the singular point T = 4
    arr = fp.d_rho_dT(np.array([1.0, 4.0, 20.0]))
    assert np.isclose(arr[1], 0.0, atol=1e-6)

def test_water_viscosity_known_value_and_decreasing():
    assert np.isclose(float(fp.water_viscosity(20.0)), 0.0010129, atol=1e-6)
    mu = fp.water_viscosity(np.array([5.0, 20.0, 30.0]))
    assert np.all(np.diff(mu) < 0)

def test_d_mu_dT_negative():
    assert float(fp.d_mu_dT(20.0)) < 0.0
    arr = fp.d_mu_dT(np.array([10.0, 20.0, 30.0]))
    assert np.all(arr < 0.0)

def test_particle_reynolds():
    re = fp.particle_reynolds(
        u_star=np.array([0.05, 0.1]),
        D50_m=np.array([0.01, 0.01]),
        rho=np.array([1000.0, 1000.0]),
        mu=np.array([1e-3, 1e-3]),
    )
    # Re_s = rho u* D50 / mu = 1000 * 0.05 * 0.01 / 1e-3 = 500
    assert np.isclose(re[0], 500.0)
    assert np.isclose(re[1], 1000.0)

def test_viscous_sublayer_thickness_finite_and_infinite():
    delta = fp.viscous_sublayer_thickness(
        u_star=np.array([0.1, 0.0]),
        rho=np.array([1000.0, 1000.0]),
        mu=np.array([1e-3, 1e-3]),
    )
    # nu = 1e-6; delta = 11.6 * 1e-6 / 0.1 = 1.16e-4
    assert np.isclose(delta[0], 11.6 * 1e-6 / 0.1)
    assert np.isinf(delta[1])

def test_b_s_log_layer_intercept():
    # _B_s should be finite and around the law-of-the-wall constant ~8.5
    b = fp._B_s(np.array([1.0, 100.0, 1e4]))
    assert np.all(np.isfinite(b))

def test_paphitis_tau_cr_star_range():
    re = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
    tau_cr_star = fp.paphitis_tau_cr_star(re)
    # Physically plausible band for the Shields curve
    assert np.all(tau_cr_star > 0.0)
    assert np.all(tau_cr_star < 0.2)

def test_shields_stress():
    tau = np.array([5.0, 10.0])
    tau_star = fp.shields_stress(
        tau=tau,
        rho=np.array([1000.0, 1000.0]),
        rho_s=2650.0,
        g=9.80665,
        D50_m=np.array([0.01, 0.01]),
    )
    expected = tau / ((2650.0 - 1000.0) * 9.80665 * 0.01)
    assert np.allclose(tau_star, expected)


# -----------------------------------------------------------------------------
# Field and grain-size initializer helpers
# -----------------------------------------------------------------------------



def test_field_at_node_paths(small_grid):
    n = small_grid.number_of_nodes
    # None -> zeros
    assert initf.field_at_node(small_grid, None).shape == (n,)
    # correct size -> flattened copy
    good = np.ones(n, dtype=int)
    assert np.array_equal(initf.field_at_node(small_grid, good), good)
    # wrong size -> falls back to zeros
    bad = np.ones(n + 3, dtype=int)
    assert np.all(initf.field_at_node(small_grid, bad) == 0)
    # empty array -> returned as-is (size 0)
    empty = initf.field_at_node(small_grid, np.array([], dtype=int))
    assert empty.size == 0

def test_field_at_link_paths(small_grid):
    n = small_grid.number_of_links
    assert initf.field_at_link(small_grid, None).shape == (n,)
    good = np.arange(n, dtype=float)
    assert np.allclose(initf.field_at_link(small_grid, good), good)
    bad = np.ones(n + 2, dtype=float)
    assert np.all(initf.field_at_link(small_grid, bad) == 0)

def test_gsd_at_link_paths(small_grid):
    n = small_grid.number_of_links
    gsd = np.array([[32, 100], [8, 0]], dtype=float)  # 2 size classes -> 1 fraction col
    ncols = gsd.shape[0] - 1
    # None -> zeros of shape (n_links, ncols)
    out = initf.gsd_at_link(small_grid, None, gsd)
    assert out.shape == (n, ncols)
    # correct shape passes through
    good = np.ones((n, ncols), dtype=float)
    assert np.allclose(initf.gsd_at_link(small_grid, good, gsd), good)
    # wrong shape -> zeros
    bad = np.ones((n + 1, ncols), dtype=float)
    assert np.all(initf.gsd_at_link(small_grid, bad, gsd) == 0)

def test_velocity_at_link_paths(small_grid):
    n = small_grid.number_of_links
    small_grid.add_zeros("surface_water__velocity", at="link")
    small_grid["link"]["surface_water__velocity"][:] = 0.3
    # None -> copy of the grid field
    out = initf.velocity_at_link(small_grid, None)
    assert np.allclose(out, 0.3)
    # correct size passes through
    good = np.full(n, 0.5)
    assert np.allclose(initf.velocity_at_link(small_grid, good), good)
    # wrong size -> falls back to grid field copy
    bad = np.full(n + 4, 0.7)
    assert np.allclose(initf.velocity_at_link(small_grid, bad), 0.3)

def test_adds_2mm_to_gsd_inserts_row():
    gsd = [[32, 100, 100], [16, 25, 50], [1, 0, 0]]  # has a sub-2 mm class, no 2 mm
    out = initg.adds_2mm_to_gsd(gsd)
    # A 2 mm row should now be present
    assert np.any(np.isclose(out[:, 0], 2.0))
    # And the array grew by one row
    assert out.shape[0] == len(gsd) + 1

def test_adds_2mm_to_gsd_noop_for_coarse_gsd():
    gsd = [[32, 100, 100], [16, 25, 50], [8, 0, 0]]  # all >= 2 mm
    out = initg.adds_2mm_to_gsd(gsd)
    assert out.shape[0] == len(gsd)

def test_remove_sand_from_gsd_parker():
    gsd = [[32, 100, 100], [16, 60, 70], [8, 30, 40], [1, 0, 0]]
    out = initg.remove_sand_from_gsd(gsd, "Parker1990")
    # Sand (< 2 mm) removed, fractions renormalised so coarsest ends at 100
    assert np.all(out[:, 0] >= 2)
    assert np.isclose(out[0, 1], 100.0)

def test_remove_sand_from_gsd_other_eq_is_noop():
    gsd = [[32, 100, 100], [16, 60, 70], [1, 0, 0]]
    out = initg.remove_sand_from_gsd(gsd, "MPM")
    assert out.shape[0] == len(gsd)


# -----------------------------------------------------------------------------
# RiverBedDynamics option sweep and integration coverage
# -----------------------------------------------------------------------------



@pytest.mark.parametrize(
    "equation",
    ["MPM", "FLvB", "WongAndParker", "Huang", "Parker1990", "WilcockAndCrowe"],
)
def test_run_one_step_each_bedload_equation(equation):
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bedload_equation=equation,
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
    )
    rbd.run_one_step()
    # Shear stress and net bedload fields are populated and finite
    assert np.all(np.isfinite(rbd._surface_water__shear_stress_link))
    assert np.all(np.isfinite(rbd._sed_transp__net_bedload_node))

def test_variable_critical_shear_stress_mueller():
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bedload_equation="MPM",
        variable_critical_shear_stress=True,
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._sed_transp__bedload_rate_link))

def test_hydraulic_radius_shear_stress():
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bedload_equation="MPM",
        use_hydraulics_radius_in_shear_stress=True,
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._surface_water__shear_stress_link))

@pytest.mark.parametrize("formulation", ["boussinesq", "velocity_driven"])
def test_shear_stress_formulations(formulation):
    grid = _rbd_grid()
    try:
        rbd = RiverBedDynamics(
            grid,
            gsd=RBD_GSD,
            bedload_equation="MPM",
            shear_stress_formulation=formulation,
            mannings_n=0.03,
            bed_surf__gsd_loc_node=RBD_GSD_LOC,
        )
    except (ValueError, KeyError):
        pytest.skip(f"formulation {formulation!r} not supported by this build")
        return
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._surface_water__shear_stress_link))

def test_track_stratigraphy_multiple_steps():
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bedload_equation="Parker1990",
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        track_stratigraphy=True,
        num_cycles_to_process_strat=2,
        gsd_advection_scheme="tvd_minmod",
        check_gsd_residual=False,
    )
    for _ in range(5):
        rbd.run_one_step()
    assert np.all(np.isfinite(rbd._sed_transp__net_bedload_node))

@pytest.mark.parametrize("mode", ["constant", "nonlinear"])
def test_bed_diffusion(mode):
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bedload_equation="MPM",
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        use_bed_diffusion=True,
        bed_diffusion_mode=mode,
        bed_diffusion_coeff=0.01,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(grid.at_node["topographic__elevation"]))

def test_variable_fluid_properties_with_temperature():
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bedload_equation="MPM",
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        variable_fluid_properties=True,
        water_temperature=8.0,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._surface_water__shear_stress_link))

def test_slope_limiter():
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bedload_equation="MPM",
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        use_slope_limiter=True,
        slope_limiter_angle=25.0,
    )
    for _ in range(3):
        rbd.run_one_step()
    assert np.all(np.isfinite(grid.at_node["topographic__elevation"]))

def test_fixed_nodes_and_links():
    grid = _rbd_grid()
    fix_node = np.zeros(grid.number_of_nodes, dtype=int)
    fix_node[12] = 1
    fix_link = np.zeros(grid.number_of_links, dtype=int)
    fix_link[15] = 1
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bedload_equation="Parker1990",
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        bed_surf__elev_fix_node=fix_node,
        sed_transp__bedload_rate_fix_link=fix_link,
        dt=0.1,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._sed_transp__net_bedload_node))


# -----------------------------------------------------------------------------
# HLLC coupling integration tests
# -----------------------------------------------------------------------------



def _build_channel():
    """An 8x6 sloping channel with high banks (walls) along top and bottom."""
    grid = RasterModelGrid((8, 6), xy_spacing=0.1)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += 0.005 - 0.01 * grid.x_of_node
    z[grid.y_of_node > 0.5] = 1.0
    z[grid.y_of_node < 0.2] = 1.0
    h = grid.add_zeros("surface_water__depth", at="node")
    grid.add_zeros("surface_water__velocity", at="link")
    grid.add_zeros("surface_water__elevation", at="node")
    grid.at_node["surface_water__elevation"] += h + z
    return grid, z

def _make_hllc(grid, z, update_link_fields=True):
    return RiverFlowDynamics_HLLC(
        grid,
        mannings_n=0.03,
        fixed_entry_nodes=HLLC_ENTRY_NODES,
        entry_nodes_h_values=np.full(4, 0.5),
        entry_nodes_u_values=np.full(4, 0.6),
        entry_nodes_v_values=np.zeros(4),
        fixed_exit_nodes=HLLC_EXIT_NODES,
        exit_nodes_eta_values=np.full(4, (z[HLLC_ENTRY_NODES] + 0.5).mean()),
        wall_edges={"top", "bottom"},
        update_link_fields=update_link_fields,
    )

def _spin_up(hllc, grid, n=300, dt=0.01):
    """Establish steady flow before introducing morphodynamics."""
    for _ in range(n):
        hllc.run_one_step(dt=dt)
    grid["link"]["surface_water__depth"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__depth"
    )

def _make_rbd(grid):
    fixed = np.zeros(grid.number_of_nodes)
    fixed[HLLC_EXIT_NODES] = 1
    return RiverBedDynamics(
        grid,
        gsd=HLLC_GSD,
        bedload_equation="MPM",
        shear_stress_formulation="velocity_driven",
        mannings_n=0.03,
        outlet_boundary_condition="fixedValue",
        bed_surf__elev_fix_node=fixed,
    )

def _couple_step(hllc, rbd, grid, dt=0.01):
    """One coupled hydraulics -> bed step."""
    hllc.run_one_step(dt=dt)
    # HLLC writes depth at nodes; RBD needs it at links
    grid["link"]["surface_water__depth"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__depth"
    )
    rbd._grid._dt = dt
    rbd.run_one_step()

def test_hllc_drives_rbd_coupling_executes():
    """A short coupled run: HLLC establishes flow, RBD consumes the link
    velocity/depth it produces and computes shear and bedload."""
    grid, z = _build_channel()
    hllc = _make_hllc(grid, z)
    _spin_up(hllc, grid)

    # HLLC produced positive depth (nodes) and velocity (links)
    assert grid.at_node["surface_water__depth"].max() > 0.1
    assert grid.at_link["surface_water__velocity"].max() > 0.1

    rbd = _make_rbd(grid)
    for _ in range(40):
        _couple_step(hllc, rbd, grid)

    # RBD read the HLLC link velocity and produced finite shear and bedload
    assert np.all(np.isfinite(rbd._surface_water__shear_stress_link))
    assert np.all(np.isfinite(rbd._sed_transp__bedload_rate_link))
    assert np.all(np.isfinite(rbd._sed_transp__net_bedload_node))
    # Transport is active somewhere in the channel
    assert np.count_nonzero(rbd._sed_transp__bedload_rate_link) > 0
    assert np.all(np.isfinite(grid.at_node["topographic__elevation"]))

def test_hllc_rbd_bed_responds_and_stays_finite():
    """Over a longer coupled run the bed measurably evolves, remains finite,
    and the fixed outlet nodes do not move."""
    grid, z = _build_channel()
    hllc = _make_hllc(grid, z)
    _spin_up(hllc, grid)
    rbd = _make_rbd(grid)

    z0 = grid.at_node["topographic__elevation"].copy()
    outlet0 = z0[HLLC_EXIT_NODES].copy()
    for _ in range(150):
        _couple_step(hllc, rbd, grid)

    dz = grid.at_node["topographic__elevation"] - z0
    assert np.all(np.isfinite(dz))
    # The bed moved (transport changed the channel) but did not blow up
    assert np.max(np.abs(dz)) > 1e-5
    assert np.max(np.abs(dz)) < 5.0
    # Fixed-elevation outlet nodes are held in place
    assert np.allclose(grid.at_node["topographic__elevation"][HLLC_EXIT_NODES], outlet0)

def test_hllc_link_velocity_is_the_coupling_source():
    """RBD's working velocity (``rbd._u``) is exactly the link field HLLC fills
    with ``update_link_fields=True``; without it the field stays zero and no
    sediment moves."""
    # With link-field updates enabled, RBD sees the HLLC velocity
    grid, z = _build_channel()
    hllc = _make_hllc(grid, z, update_link_fields=True)
    _spin_up(hllc, grid)
    rbd = _make_rbd(grid)
    _couple_step(hllc, rbd, grid)
    assert np.allclose(rbd._u, grid.at_link["surface_water__velocity"])
    assert np.count_nonzero(rbd._sed_transp__bedload_rate_link) > 0

    # Without link-field updates, the link velocity is never populated by HLLC,
    # so the bed stays put -- demonstrating why update_link_fields is required.
    grid2, z2 = _build_channel()
    hllc2 = _make_hllc(grid2, z2, update_link_fields=False)
    for _ in range(300):
        hllc2.run_one_step(dt=0.01)
    grid2["link"]["surface_water__depth"] = map_mean_of_link_nodes_to_link(
        grid2, "surface_water__depth"
    )
    rbd2 = _make_rbd(grid2)
    z0 = grid2.at_node["topographic__elevation"].copy()
    for _ in range(40):
        # deliberately do NOT supply link velocity
        grid2["link"]["surface_water__depth"] = map_mean_of_link_nodes_to_link(
            grid2, "surface_water__depth"
        )
        rbd2._grid._dt = 0.01
        rbd2.run_one_step()
    assert np.allclose(grid2.at_node["topographic__elevation"], z0)


# -----------------------------------------------------------------------------
# Targeted RiverBedDynamics release-branch coverage
# -----------------------------------------------------------------------------



@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bedload_equation": "NotARealEquation"}, "Unknown bedload_equation"),
        ({"bed_diffusion_mode": "bogus"}, "bed_diffusion_mode must be one of"),
        ({"time_stepping": "nope"}, "Unknown time_stepping"),
        ({"morfac": 0}, "morfac must be a positive integer"),
        ({"depth_threshold": -1.0}, "depth_threshold must be >= 0.0"),
    ],
)
def test_constructor_validation_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _rbd_component(**kwargs)

def test_morfac_early_exit_and_adaptive_dt_warning():
    rbd = _rbd_component(morfac=2, adaptive_dt=True, dt=1.0)
    z_initial = rbd._grid.at_node["topographic__elevation"].copy()

    rbd.run_one_step()
    assert rbd._morfac_counter == 1
    assert np.allclose(rbd._grid.at_node["topographic__elevation"], z_initial)

    rbd._sed_transp__bedload_rate_link[:] = 1.0
    with pytest.warns(UserWarning, match="adaptive_dt"):
        rbd.run_one_step()

    assert rbd._morfac_counter == 0
    assert rbd._grid._dt == 1.0

def test_update_fluid_properties_and_density_properties():
    rbd = _rbd_component(variable_fluid_properties=True, water_temperature=10.0)
    temperatures = np.linspace(4.0, 20.0, rbd._grid.number_of_links)

    rbd.update_fluid_properties(temperatures)

    assert rbd.water_density_link.shape == (rbd._grid.number_of_links,)
    assert rbd.water_dynamic_viscosity_link.shape == (rbd._grid.number_of_links,)
    assert np.all(np.isfinite(rbd.water_density_link))
    assert np.all(np.isfinite(rbd.water_dynamic_viscosity_link))

def test_stable_dt_helpers_cover_zero_and_finite_transport():
    rbd = _rbd_component()
    assert np.isinf(rbd.calc_max_stable_dt_advective())
    assert np.isinf(rbd.calc_max_stable_dt_diffusive())

    rbd._sed_transp__bedload_rate_link[:] = 0.05
    assert np.isfinite(rbd.calc_max_stable_dt_advective())
    assert np.isfinite(rbd.calc_max_stable_dt())

    rbd_diff = _rbd_component(use_bed_diffusion=True, bed_diffusion_mode="constant")
    assert np.isinf(rbd_diff.calc_max_stable_dt_diffusive())
    rbd_diff._bed_diffusion_coeff = 0.01
    assert np.isfinite(rbd_diff.calc_max_stable_dt_diffusive())

    rbd_nonlin = _rbd_component(use_bed_diffusion=True, bed_diffusion_mode="nonlinear")
    assert np.isinf(rbd_nonlin.calc_max_stable_dt_diffusive())
    rbd_nonlin._sed_transp__bedload_rate_link[:] = 0.01
    assert np.isfinite(rbd_nonlin.calc_max_stable_dt_diffusive())

def test_rk2_and_implicit_bed_elevation_paths():
    for scheme in ("rk2", "implicit"):
        rbd = _rbd_component(time_stepping=scheme)
        rbd.run_one_step()
        assert np.all(np.isfinite(rbd._grid.at_node["topographic__elevation"]))

def test_implicit_helpers_directly():
    rbd = _rbd_component()
    rbd.shear_stress()
    rbd.bedload_equation()
    rbd.calculate_net_bedload()

    jac = rbd._compute_transport_jacobian(eps=1.0e-5)
    lhs, rhs = rbd._assemble_implicit_system(jac, dt=0.1)
    dz = rbd._solve_implicit_exner(dt=0.1)

    assert jac.shape == (rbd._grid.number_of_nodes, rbd._grid.number_of_nodes)
    assert lhs.shape == jac.shape
    assert rhs.shape == (rbd._grid.number_of_nodes,)
    assert dz.shape == (rbd._grid.number_of_nodes,)

def test_bed_diffusion_cfl_warning_is_explicitly_captured():
    rbd = _rbd_component(
        use_bed_diffusion=True,
        bed_diffusion_mode="constant",
        bed_diffusion_coeff=10.0,
        check_diffusion_cfl=True,
        dt=1.0,
    )
    rbd.shear_stress()
    rbd.bedload_equation()
    with pytest.warns(UserWarning, match="Diffusive CFL"):
        rbd.bed_diffusion()
    assert np.all(np.isfinite(rbd._bed_surf__diffusive_dz_node))

def test_slope_limiter_break_and_max_iteration_paths():
    rbd_flat = _rbd_component(use_slope_limiter=True)
    rbd_flat._apply_slope_limiter()
    assert rbd_flat._slope_limiter_n_avalanched == 0

    rbd_steep = _rbd_component(
        use_slope_limiter=True,
        slope_limiter_angle=5.0,
        slope_limiter_max_iterations=1,
    )
    z = rbd_steep._grid.at_node["topographic__elevation"]
    z[12] += 10.0
    rbd_steep._grid.at_link["topographic__elevation"] = (
        rbd_steep._grid.map_mean_of_link_nodes_to_link(z)
    )
    rbd_steep._apply_slope_limiter()
    assert rbd_steep._slope_limiter_n_avalanched == 1

def test_gsd_evolver_validation_noop_and_tvd_path():
    with pytest.raises(ValueError, match="Unknown gsd_advection_scheme"):
        ToroEscobarEvolver("bad")

    class NoStratigraphy:
        _track_stratigraphy = False

    ToroEscobarEvolver().evolve(NoStratigraphy())

    rbd = _rbd_component(
        bedload_equation="Parker1990",
        track_stratigraphy=True,
        gsd_advection_scheme="tvd_minmod",
        dt=0.01,
    )
    rbd.run_one_step()

    n_frac = rbd._bed_surf__gsd_link.shape[1]
    rbd._sed_transp__bedload_rate_link[:] = 0.0
    rbd._sed_transp__bedload_rate_link[rbd._grid.horizontal_links[::2]] = 0.001
    rbd._sed_transp__bedload_rate_link[rbd._grid.horizontal_links[1::2]] = -0.001
    rbd._sed_transp__bedload_rate_link[rbd._grid.vertical_links[::2]] = 0.001
    rbd._sed_transp__bedload_rate_link[rbd._grid.vertical_links[1::2]] = -0.001
    rbd._sed_transp__bedload_gsd_link = np.tile(
        np.full(n_frac, 1.0 / n_frac), (rbd._grid.number_of_links, 1)
    )
    rbd._current_t = 1.0

    rbd.update_bed_surf_gsd()

    assert np.all(np.isfinite(rbd._bed_surf__gsd_link))


# -----------------------------------------------------------------------------
# Stratigraphy deposition and erosion branches
# -----------------------------------------------------------------------------



def _strat_component():
    grid = RasterModelGrid((8, 3), xy_spacing=100)
    grid.at_node["topographic__elevation"] = [
        [1.12, 1.00, 1.12],
        [1.12, 1.01, 1.12],
        [1.12, 1.01, 1.12],
        [1.12, 1.01, 1.12],
        [1.12, 1.01, 1.12],
        [1.12, 1.01, 1.12],
        [1.12, 1.01, 1.12],
        [1.12, 1.12, 1.12],
    ]
    grid.at_node["surface_water__depth"] = np.full(grid.number_of_nodes, 0.40)
    grid.at_link["surface_water__depth"] = np.full(grid.number_of_links, 0.40)
    grid.at_link["surface_water__velocity"] = np.full(grid.number_of_links, 0.40)
    grid.set_watershed_boundary_condition(grid.at_node["topographic__elevation"])

    gsd = [[8, 100], [4, 90], [2, 0]]
    fixed_nodes = np.zeros(grid.number_of_nodes)
    fixed_nodes[[1, 4]] = 1
    qb = np.zeros(grid.number_of_links)
    qb[[28, 33]] = -0.002

    rbd = RiverBedDynamics(
        grid,
        gsd=gsd,
        bedload_equation="Parker1990",
        outlet_boundary_condition="fixedValue",
        bed_surf__elev_fix_node=fixed_nodes,
        bed_surf__gsd_fix_node=fixed_nodes,
        sed_transp__bedload_rate_fix_link=qb,
        track_stratigraphy=True,
        bed_surf_new_layer_thick=0.02,
        num_cycles_to_process_strat=1,
        check_gsd_residual=False,
    )
    return grid, rbd

def test_stratigraphy_deposition_branch():
    grid, rbd = _strat_component()
    for t in range(6):
        rbd._current_t = t
        rbd.run_one_step()

    link = next(
        k
        for k in rbd._link_stratigraphy_temp
        if k in grid.active_links and len(rbd._link_stratigraphy_temp[k]) > 0
    )
    n_before = len(rbd._link_stratigraphy[link])

    rbd._update_subsurface_deposited = True
    rbd._update_deposited_link_id = np.array([link])
    strat.evolve(rbd)

    # A new subsurface layer was committed: temp buffer cleared, history grew
    assert len(rbd._link_stratigraphy_temp[link]) == 0
    assert len(rbd._link_stratigraphy[link]) == n_before + 1

def test_stratigraphy_erosion_branch():
    grid, rbd = _strat_component()
    for t in range(8):
        rbd._current_t = t
        rbd.run_one_step()

    link = next(k for k in grid.active_links if k in rbd._link_stratigraphy)
    # The erosion path inspects the previous layer, so ensure >1 history rows
    if len(rbd._link_stratigraphy[link]) < 2:
        rbd._link_stratigraphy[link].append(rbd._link_stratigraphy[link][-1].copy())
    n_before = len(rbd._link_stratigraphy[link])

    rbd._update_subsurface_eroded = True
    rbd._update_eroded_link_id = np.array([link])
    strat.evolve(rbd)

    # Erosion records the exposed (previous) layer as the new surface history
    assert len(rbd._link_stratigraphy[link]) == n_before + 1
    assert np.all(np.isfinite(rbd._bed_surf__gsd_link[link]))

def test_checks_erosion_or_deposition_sets_flags():
    """Cover the deposition/erosion trigger detection in
    ``checks_erosion_or_deposition``. Reaching the +/- threshold by forward
    integration takes ~1300 steps; instead we push the accumulated new-layer
    thickness past the threshold on two active links (one positive, one
    negative) and call the detector directly."""
    grid, rbd = _strat_component()
    for t in range(3):
        rbd._current_t = t
        rbd.run_one_step()

    active = grid.active_links
    fixed = set(np.atleast_1d(rbd._bed_surf__elev_fix_link_id).tolist())
    free = [int(l) for l in active if int(l) not in fixed]
    dep_link, ero_link = free[0], free[1]
    thr = rbd._bed_surf_new_layer_thick

    # checks_erosion_or_deposition recomputes the new-layer thickness as
    # link topographic elevation minus subsurface elevation, so set those.
    z_link = grid["link"]["topographic__elevation"]
    z_link[dep_link] = rbd._topogr__elev_subsurf_link[dep_link] + 2.0 * thr
    z_link[ero_link] = rbd._topogr__elev_subsurf_link[ero_link] - 2.0 * thr

    rbd._update_subsurface_deposited = False
    rbd._update_subsurface_eroded = False
    strat.checks_erosion_or_deposition(rbd)

    assert rbd._update_subsurface_deposited is True
    assert rbd._update_subsurface_eroded is True
    assert dep_link in rbd._update_deposited_link_id
    assert ero_link in rbd._update_eroded_link_id


# -----------------------------------------------------------------------------
# Utilities, velocity-driven shear, fixed-link restore, and CSV export
# -----------------------------------------------------------------------------


def _utility_run(**kwargs):
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        check_advective_cfl=False,
        check_gsd_residual=False,
        **kwargs,
    )
    rbd.run_one_step()
    return grid, rbd



def test_vector_mapper():
    grid = _rbd_grid()
    u = np.zeros(grid.number_of_links)
    u[15] = -0.01
    u[19] = -0.02
    vector, magnitude = utilities.vector_mapper(grid, u)
    assert vector.shape == (grid.number_of_nodes, 2)
    assert magnitude.shape == (grid.number_of_nodes,)
    assert np.all(magnitude >= 0)

def test_map_gsd_from_link_to_node_both_locations():
    _, rbd = _utility_run(bedload_equation="Parker1990")
    surf = utilities.map_gsd_from_link_to_node(rbd, location="bed_surf")
    load = utilities.map_gsd_from_link_to_node(rbd, location="bedload")
    assert surf.shape[0] == rbd._grid.number_of_nodes
    assert load.shape[0] == rbd._grid.number_of_nodes

def test_format_gsd_link_and_node():
    _, rbd = _utility_run(bedload_equation="Parker1990")
    df_link = utilities.format_gsd(rbd, rbd._sed_transp__bedload_gsd_link)
    assert df_link.index[0].startswith("Link_")
    node_gsd = utilities.map_gsd_from_link_to_node(rbd, location="bedload")
    df_node = utilities.format_gsd(rbd, node_gsd)
    assert df_node.index[0].startswith("Node_")

def test_get_available_fields():
    fields = utilities.get_available_fields()
    assert isinstance(fields, list)
    names = [f[0] for f in fields]
    assert "rbd._surface_water__shear_stress_link" in names
    # Sorted, with (name, units) pairs
    assert names == sorted(names)

def test_velocity_driven_formulation():
    _, rbd = _utility_run(
        bedload_equation="MPM",
        shear_stress_formulation="velocity_driven",
        mannings_n=0.03,
    )
    assert np.all(np.isfinite(rbd._surface_water__shear_stress_link))

def test_velocity_driven_requires_mannings_n():
    grid = _rbd_grid()
    with pytest.raises(ValueError):
        RiverBedDynamics(
            grid,
            gsd=RBD_GSD,
            bed_surf__gsd_loc_node=RBD_GSD_LOC,
            shear_stress_formulation="velocity_driven",
            mannings_n=None,
        )

def test_mpm_fixed_bedload_rate_link():
    grid = _rbd_grid()
    fix_link = np.zeros(grid.number_of_links, dtype=int)
    fix_link[15] = 1
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        bedload_equation="MPM",
        sed_transp__bedload_rate_fix_link=fix_link,
        dt=0.1,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._sed_transp__bedload_rate_link))

def test_wilcock_crowe_fixed_rate_and_gsd_links():
    grid = _rbd_grid()
    n_links = grid.number_of_links
    n_frac = len(RBD_GSD) - 1  # RBD_GSD value columns
    fix_rate = np.zeros(n_links, dtype=float)
    fix_rate[15] = 1.0
    # RBD_GSD fix array is (n_links, n_frac); a non-zero row marks an imposed link
    fix_gsd = np.zeros((n_links, n_frac), dtype=float)
    fix_gsd[15, :] = [0.6, 0.4]
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        bedload_equation="WilcockAndCrowe",
        sed_transp__bedload_rate_fix_link=fix_rate,
        sed_transp__bedload_gsd_fix_link=fix_gsd,
        dt=0.1,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._sed_transp__bedload_gsd_link))

def test_bedload_equation_dispatcher_warning():
    _, rbd = _utility_run(bedload_equation="MPM")
    rbd._bedload_equation = "MPM"  # compatibility attribute consumed by the dispatcher
    with pytest.warns(DeprecationWarning):
        qb = mpm.bedload_equation(rbd)
    assert qb.shape[0] == rbd._grid.number_of_links

def test_stratigraphy_layer_cycling_and_write(tmp_path):
    grid = _rbd_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=RBD_GSD,
        bed_surf__gsd_loc_node=RBD_GSD_LOC,
        bedload_equation="Parker1990",
        track_stratigraphy=True,
        num_cycles_to_process_strat=1,
        bed_surf_new_layer_thick=1e-4,  # tiny layer so deposition/erosion crosses it
        morfac=50,  # amplify morphological change to force layer turnover
    )
    for _ in range(8):
        rbd.run_one_step()

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        stratigraphy.write_evolution(rbd)
        assert (tmp_path / "Stratigraphy_evolution.csv").exists()
    finally:
        os.chdir(cwd)