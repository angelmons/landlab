"""Tests for HLLC depth-averaged turbulence closures."""

from __future__ import annotations

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverFlowDynamics_HLLC
from landlab.components.river_flow_dynamics._turbulence import (
    DepthAveragedTurbulenceModel,
)


def _channel_grid():
    grid = RasterModelGrid((6, 6), xy_spacing=0.1)
    z = grid.add_zeros("topographic__elevation", at="node")
    z += 0.003 - 0.005 * grid.x_of_node
    h = grid.add_zeros("surface_water__depth", at="node")
    h += 0.2
    return grid, z


def test_turbulence_model_validation():
    """Reject unsupported closure and filter-width names."""
    with pytest.raises(ValueError, match="turbulence_model"):
        DepthAveragedTurbulenceModel(model="bogus")
    with pytest.raises(ValueError, match="filter_width_model"):
        DepthAveragedTurbulenceModel(model="smagorinsky", filter_width_model="bad")


def test_turbulence_model_safeguards_and_dry_masking():
    """Apply floors, caps, and dry-cell masks consistently."""
    model = DepthAveragedTurbulenceModel(
        model="constant",
        constant_eddy_viscosity=10.0,
        background_eddy_viscosity=0.01,
        max_eddy_viscosity=0.05,
        dry_depth_threshold=0.1,
    )
    h = np.array([[0.0, 0.2], [0.3, 0.4]])
    u = np.zeros_like(h)
    v = np.zeros_like(h)

    nu_t = model.update(h, u, v)

    assert nu_t[0, 0] == 0.0
    assert np.allclose(nu_t[h > 0.1], 0.05)


@pytest.mark.parametrize("model", ["smagorinsky", "parabolic", "hybrid_additive"])
def test_hllc_turbulence_closures_create_eddy_viscosity_field(model):
    """Run one HLLC step with each algebraic turbulence closure."""
    grid, z = _channel_grid()
    entry_nodes = grid.nodes_at_left_edge[1:-1]
    exit_nodes = grid.nodes_at_right_edge[1:-1]

    rfd = RiverFlowDynamics_HLLC(
        grid,
        mannings_n=0.03,
        fixed_entry_nodes=entry_nodes,
        entry_nodes_h_values=np.full(entry_nodes.size, 0.2),
        entry_nodes_u_values=np.full(entry_nodes.size, 0.3),
        entry_nodes_v_values=np.zeros(entry_nodes.size),
        fixed_exit_nodes=exit_nodes,
        exit_nodes_eta_values=z[exit_nodes] + 0.2,
        wall_edges={"top", "bottom"},
        turbulence_model=model,
        smagorinsky_cs=0.15,
        parabolic_alpha=0.067,
        eddy_viscosity_background=1.0e-6,
        eddy_viscosity_max=0.1,
    )

    rfd.run_one_step(dt=0.001)

    assert "surface_water__eddy_viscosity" in grid.at_node
    nu_t = grid.at_node["surface_water__eddy_viscosity"]
    assert np.all(np.isfinite(nu_t))
    assert np.all(nu_t >= 0.0)
