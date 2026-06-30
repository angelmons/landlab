"""Tests for eddy-viscosity coupling in temperature transport."""

from __future__ import annotations

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverTemperatureDynamics


def _grid():
    grid = RasterModelGrid((5, 5), xy_spacing=1.0)
    grid.add_zeros("surface_water__depth", at="node")[:] = 1.0
    grid.add_zeros("surface_water__velocity", at="link")
    grid.add_zeros("advection__velocity", at="link")
    grid.add_zeros("surface_water__temperature", at="node")
    return grid


def test_temperature_turbulent_diffusivity_requires_eddy_viscosity_field():
    """Raise a clear error if turbulent diffusivity is requested without nu_t."""
    with pytest.raises(ValueError, match="surface_water__eddy_viscosity"):
        RiverTemperatureDynamics(
            _grid(), heat_exchange=False, use_turbulent_diffusivity=True
        )


def test_temperature_turbulent_diffusivity_spreads_temperature():
    """Use surface_water__eddy_viscosity as an added thermal diffusivity."""
    grid = _grid()
    grid.add_zeros("surface_water__eddy_viscosity", at="node")[:] = 0.2
    comp = RiverTemperatureDynamics(
        grid,
        heat_exchange=False,
        alpha_L=0.0,
        alpha_T=0.0,
        use_turbulent_diffusivity=True,
        turbulent_schmidt_number=1.0,
    )
    T = grid.at_node["surface_water__temperature"]
    T[:] = 10.0
    T[12] = 20.0
    before = T.copy()

    comp.run_one_step(dt=0.1)

    assert not np.allclose(T, before)
    assert np.all(np.isfinite(T))
