"""
Unit tests for landlab.components.river_temperature_dynamics.RiverTemperatureDynamics

This suite evaluates the depth-averaged 2D temperature model, testing the atmospheric
heat budget, dynamic water property recalculation, and passive advection behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverTemperatureDynamics


# -----------------------------------------------------------------------------
# Helpers & Fixtures
# -----------------------------------------------------------------------------


def _make_grid(nr: int, nc: int, dx: float = 1.0) -> RasterModelGrid:
    grid = RasterModelGrid((nr, nc), xy_spacing=dx)
    grid.add_zeros("surface_water__depth", at="node")
    grid.add_zeros("surface_water__velocity", at="link")
    grid.add_zeros("advection__velocity", at="link")
    grid.add_zeros("surface_water__temperature", at="node")
    return grid


@pytest.fixture
def rtd_grid():
    """Provides a grid initialized with required hydraulic and atmospheric fields."""
    grid = _make_grid(5, 10, dx=1.0)
    grid.at_node["surface_water__depth"][:] = 0.5
    
    # Initialize necessary atmospheric forcing fields
    grid.add_zeros("air__temperature", at="node")
    grid.add_zeros("air__relative_humidity", at="node")
    grid.add_zeros("air__velocity", at="node")
    grid.add_zeros("radiation__incoming_shortwave_flux", at="node")
    grid.add_zeros("solar__altitude_angle", at="node")
    return grid


# -----------------------------------------------------------------------------
# Initialization & Setup
# -----------------------------------------------------------------------------


def test_name(rtd_grid):
    comp = RiverTemperatureDynamics(rtd_grid)
    assert comp.name == "RiverTemperatureDynamics"


def test_no_heat_exchange_auto_creates_fields():
    """
    If heat_exchange=False, the component should automatically generate zeroed
    atmospheric fields so validation passes without user input.
    """
    grid = _make_grid(5, 5)
    # Instantiate without manually adding 'air__temperature', etc.
    comp = RiverTemperatureDynamics(grid, heat_exchange=False)
    
    assert "air__temperature" in grid.at_node
    assert "radiation__incoming_shortwave_flux" in grid.at_node
    assert np.all(grid.at_node["air__temperature"] == 0.0)


# -----------------------------------------------------------------------------
# Passive Transport (Heat Exchange Disabled)
# -----------------------------------------------------------------------------


def test_passive_transport_conserves_heat(rtd_grid):
    """
    With heat_exchange=False, temperature acts as a conservative tracer.
    Total heat energy (T * h) should be conserved in a closed domain.
    """
    rtd_grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
    
    comp = RiverTemperatureDynamics(rtd_grid, heat_exchange=False)
    
    T = rtd_grid.at_node["surface_water__temperature"]
    h = rtd_grid.at_node["surface_water__depth"]
    
    # Inject a warm pulse
    T[:] = 10.0
    T[rtd_grid.number_of_nodes // 2] = 30.0
    
    initial_heat = np.sum(T * h)
    
    for _ in range(10):
        comp.run_one_step(dt=1.0)
        
    final_heat = np.sum(T * h)
    
    assert abs(final_heat - initial_heat) < 1e-10
    # Confirm flux diagnostic arrays remained zero
    assert np.all(comp.Q_sw_net == 0.0)
    assert np.all(comp.Q_net == 0.0)


# -----------------------------------------------------------------------------
# Thermodynamic Behavior
# -----------------------------------------------------------------------------


def test_heating_response(rtd_grid):
    """A net positive radiation flux should increase water temperature."""
    rtd_grid.at_node["air__temperature"][:] = 30.0
    rtd_grid.at_node["radiation__incoming_shortwave_flux"][:] = 1000.0  # Intense sun
    rtd_grid.at_node["solar__altitude_angle"][:] = np.radians(80.0)     # High noon
    rtd_grid.at_node["air__relative_humidity"][:] = 80.0                # Suppress evap
    
    T = rtd_grid.at_node["surface_water__temperature"]
    T[:] = 15.0  # Cold water
    
    comp = RiverTemperatureDynamics(rtd_grid, heat_exchange=True)
    comp.run_one_step(dt=60.0)
    
    assert np.all(T > 15.0), "Temperature did not increase under intense heating"
    assert np.all(comp.Q_net > 0.0), "Net heat flux is not positive"


def test_cooling_response_via_evaporation(rtd_grid):
    """Dry, windy conditions should cause evaporative cooling."""
    rtd_grid.at_node["air__temperature"][:] = 10.0
    rtd_grid.at_node["air__relative_humidity"][:] = 10.0                # Very dry
    rtd_grid.at_node["air__velocity"][:] = 10.0                         # High wind
    rtd_grid.at_node["radiation__incoming_shortwave_flux"][:] = 0.0     # Night
    
    T = rtd_grid.at_node["surface_water__temperature"]
    T[:] = 20.0  # Warm water
    
    comp = RiverTemperatureDynamics(rtd_grid, heat_exchange=True)
    comp.run_one_step(dt=60.0)
    
    assert np.all(T < 20.0), "Temperature did not decrease under evaporative conditions"
    assert np.all(comp.Q_evap > 0.0), "Evaporative heat loss is zero"
    assert np.all(comp.Q_net < 0.0), "Net heat flux is not negative"


def test_variable_water_properties(rtd_grid):
    """If enabled, water density (rho) should dynamically update based on temperature."""
    comp = RiverTemperatureDynamics(rtd_grid, variable_water_properties=True)
    
    T = rtd_grid.at_node["surface_water__temperature"]
    
    # 4 deg C represents max density (~1000 kg/m^3)
    T[:] = 4.0
    comp.run_one_step(dt=1.0)
    rho_at_4C = np.mean(comp._rho)
    
    # 30 deg C represents much lower density
    T[:] = 30.0
    comp.run_one_step(dt=1.0)
    rho_at_30C = np.mean(comp._rho)
    
    assert rho_at_4C > rho_at_30C, "Density did not decrease as water warmed from 4C to 30C"
    # Max density at 4C should be extremely close to 1000.0 based on Heggen (1983)
    assert np.allclose(rho_at_4C, 1000.0, atol=1e-3)


# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------


def test_fixed_outlet_temperature(rtd_grid):
    """Outlet boundary nodes should remain clamped to the user-specified temperature."""
    fixed_T = 12.5
    comp = RiverTemperatureDynamics(
        rtd_grid,
        outlet_boundary_condition="fixed_value",
        fixed_outlet_temperature=fixed_T,
        heat_exchange=False  # Isolate BC testing
    )
    
    T = rtd_grid.at_node["surface_water__temperature"]
    T[:] = 20.0
    
    comp.run_one_step(dt=1.0)
    
    outlet_nodes = rtd_grid.nodes_at_right_edge
    assert np.all(T[outlet_nodes] == fixed_T)