"""Tests for eddy-viscosity coupling in solute transport."""

from __future__ import annotations

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverSoluteTransportDynamics


def _grid():
    grid = RasterModelGrid((5, 5), xy_spacing=1.0)
    grid.add_zeros("surface_water__depth", at="node")[:] = 1.0
    grid.add_zeros("surface_water__velocity", at="link")
    grid.add_zeros("advection__velocity", at="link")
    return grid


def test_solute_turbulent_diffusivity_requires_eddy_viscosity_field():
    """Raise a clear error if turbulent diffusivity is requested without nu_t."""
    with pytest.raises(ValueError, match="surface_water__eddy_viscosity"):
        RiverSoluteTransportDynamics(_grid(), use_turbulent_diffusivity=True)


def test_solute_turbulent_diffusivity_spreads_concentration():
    """Use surface_water__eddy_viscosity as an added scalar diffusivity."""
    grid = _grid()
    grid.add_zeros("surface_water__eddy_viscosity", at="node")[:] = 0.2
    comp = RiverSoluteTransportDynamics(
        grid,
        solutes=["tracer"],
        dispersion_mode="isotropic",
        dispersion_coefficient=0.0,
        transverse_dispersion_coefficient=0.0,
        use_turbulent_diffusivity=True,
        turbulent_schmidt_number=1.0,
    )
    C = grid.at_node["surface_water__tracer__concentration"]
    C[12] = 1.0
    before = C.copy()

    comp.run_one_step(dt=0.1)

    assert not np.allclose(C, before)
    assert np.all(np.isfinite(C))
