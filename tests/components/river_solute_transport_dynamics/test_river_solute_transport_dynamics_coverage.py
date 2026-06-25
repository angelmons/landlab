"""Additional coverage tests for RiverSoluteTransportDynamics."""

from __future__ import annotations

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverSoluteTransportDynamics


def _make_grid(nr: int = 5, nc: int = 6, dx: float = 1.0) -> RasterModelGrid:
    grid = RasterModelGrid((nr, nc), xy_spacing=dx)
    grid.add_zeros("surface_water__depth", at="node")
    grid.add_zeros("surface_water__velocity", at="link")
    grid.add_zeros("advection__velocity", at="link")
    grid.at_node["surface_water__depth"][:] = 0.5
    return grid


def test_invalid_outlet_boundary_condition():
    grid = _make_grid()
    with pytest.raises(ValueError, match="outlet_boundary_condition must be one of"):
        RiverSoluteTransportDynamics(grid, outlet_boundary_condition="bad_bc")


def test_existing_lateral_inflow_and_anisotropic_dispersion():
    grid = _make_grid()
    q_lat = grid.add_zeros("lateral__water_specific_discharge", at="node")
    q_lat[:] = 1.0e-4

    comp = RiverSoluteTransportDynamics(
        grid,
        solutes=["tracer"],
        dispersion_mode="anisotropic",
        alpha_L=1.0,
        alpha_T=0.5,
    )

    assert comp._q_lat is q_lat
    comp._advection_dispersion(dt=1.0e-12)
    assert np.all(np.isfinite(grid.at_node["surface_water__tracer__concentration"]))


def test_isotropic_node_array_dispersion_and_gradient_boundary():
    grid = _make_grid()
    d_long = np.linspace(0.01, 0.02, grid.number_of_nodes)
    d_tran = np.linspace(0.02, 0.03, grid.number_of_nodes)

    comp = RiverSoluteTransportDynamics(
        grid,
        solutes=["tracer"],
        dispersion_mode="isotropic",
        dispersion_coefficient=d_long,
        transverse_dispersion_coefficient=d_tran,
        outlet_boundary_condition="gradient_preserving",
    )

    C = grid.at_node["surface_water__tracer__concentration"]
    C[:] = np.arange(grid.number_of_nodes, dtype=float)
    comp._advection_dispersion(dt=1.0e-12)
    comp._apply_boundaries()

    outlet = grid.nodes_at_right_edge
    interior_1 = outlet - 1
    interior_2 = outlet - 2
    assert np.allclose(C[outlet], 2.0 * C[interior_1] - C[interior_2])


def test_array_storage_and_sorption_parameters_update_masked_nodes():
    grid = _make_grid()
    n = grid.number_of_nodes
    core = grid.core_nodes

    alpha = np.zeros(n)
    alpha[core] = 0.1
    lam_hat = np.zeros(n)
    lam_hat[core] = 0.05
    h_storage = np.full(n, 0.2)
    kd = np.full(n, 0.1)

    comp = RiverSoluteTransportDynamics(
        grid,
        solutes=["tracer"],
        alpha_exchange={"tracer": alpha},
        h_storage={"tracer": h_storage},
        lambda_hat_sorption={"tracer": lam_hat},
        kd_sorption={"tracer": kd},
        rho_sediment={"tracer": 2.0},
    )

    C = grid.at_node["surface_water__tracer__concentration"]
    Cs = grid.at_node["storage_zone__tracer__concentration"]
    Csed = grid.at_node["streambed__tracer__sorbate_concentration"]
    C[:] = 10.0
    Cs[:] = 0.0
    Csed[:] = 0.0

    comp._otis_reactions(dt=0.1)

    assert np.any(Cs[core] > 0.0)
    assert np.any(Csed[core] > 0.0)


def test_scalar_sorption_uses_scalar_denominator_branch():
    grid = _make_grid()
    core = grid.core_nodes

    comp = RiverSoluteTransportDynamics(
        grid,
        solutes=["tracer"],
        lambda_hat_sorption={"tracer": 0.05},
        kd_sorption={"tracer": 0.1},
        rho_sediment=2.0,
    )

    C = grid.at_node["surface_water__tracer__concentration"]
    Csed = grid.at_node["streambed__tracer__sorbate_concentration"]
    C[:] = 10.0
    Csed[:] = 0.0

    comp._otis_reactions(dt=0.1)

    assert np.any(Csed[core] > 0.0)
