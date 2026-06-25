"""Additional coverage tests for RiverTemperatureDynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverTemperatureDynamics


def _make_grid(nr: int = 5, nc: int = 5, dx: float = 1.0) -> RasterModelGrid:
    grid = RasterModelGrid((nr, nc), xy_spacing=dx)
    for name in (
        "surface_water__depth",
        "surface_water__temperature",
        "air__temperature",
        "air__relative_humidity",
        "air__velocity",
        "radiation__incoming_shortwave_flux",
        "solar__altitude_angle",
        "cloud_cover__fraction",
        "groundwater__specific_discharge",
        "groundwater__temperature",
        "sediment__temperature",
    ):
        grid.add_zeros(name, at="node")
    grid.add_zeros("surface_water__velocity", at="link")
    grid.add_zeros("advection__velocity", at="link")
    grid.at_node["surface_water__depth"][:] = 0.5
    grid.at_node["surface_water__temperature"][:] = 15.0
    grid.at_node["air__temperature"][:] = 15.0
    grid.at_node["air__relative_humidity"][:] = 50.0
    grid.at_node["air__velocity"][:] = 2.0
    grid.at_node["groundwater__temperature"][:] = 15.0
    grid.at_node["sediment__temperature"][:] = 15.0
    return grid


def test_invalid_outlet_boundary_condition():
    grid = _make_grid()
    with pytest.raises(ValueError, match="outlet_boundary_condition must be one of"):
        RiverTemperatureDynamics(grid, outlet_boundary_condition="bad_bc")


def test_heat_exchange_property_and_var_name_properties():
    grid = _make_grid()
    comp = RiverTemperatureDynamics(grid, heat_exchange=False)

    assert comp.heat_exchange is False
    comp.heat_exchange = True
    assert comp.heat_exchange is True
    with pytest.raises(TypeError, match="heat_exchange must be a bool"):
        comp.heat_exchange = 1

    assert "surface_water__depth" in comp.input_var_names
    assert "surface_water__temperature" in comp.output_var_names


def test_meteorology_with_cloud_cover_and_run_one_step_t_sim(tmp_path):
    grid = _make_grid()
    met_file = tmp_path / "met.csv"
    pd.DataFrame(
        {
            "time_sec": [0.0, 10.0],
            "T_air": [10.0, 20.0],
            "RH": [40.0, 60.0],
            "u_wind": [1.0, 3.0],
            "Q_sw": [100.0, 300.0],
            "cloud_cover": [0.2, 0.6],
        }
    ).to_csv(met_file, index=False)

    comp = RiverTemperatureDynamics(grid, met_file=met_file, heat_exchange=False)
    comp.update_meteorology(5.0)
    assert np.allclose(grid.at_node["air__temperature"], 15.0)
    assert np.allclose(grid.at_node["cloud_cover__fraction"], 0.4)

    comp.run_one_step(dt=1.0e-12, t_sim=5.0)
    assert np.all(np.isfinite(grid.at_node["surface_water__temperature"]))


def test_meteorology_without_cloud_cover_uses_zero_default(tmp_path):
    grid = _make_grid()
    met_file = tmp_path / "met_no_cloud.csv"
    pd.DataFrame(
        {
            "time_sec": [0.0, 10.0],
            "T_air": [10.0, 20.0],
            "RH": [40.0, 60.0],
            "u_wind": [1.0, 3.0],
            "Q_sw": [100.0, 300.0],
        }
    ).to_csv(met_file, index=False)

    comp = RiverTemperatureDynamics(grid, met_file=met_file, heat_exchange=False)
    comp.update_meteorology(5.0)
    assert np.allclose(grid.at_node["cloud_cover__fraction"], 0.0)


def test_atmospheric_exchange_without_wind_height_correction():
    grid = _make_grid()
    grid.at_node["radiation__incoming_shortwave_flux"][:] = 100.0
    grid.at_node["solar__altitude_angle"][:] = np.radians(45.0)

    comp = RiverTemperatureDynamics(grid, h_ws=7.0, heat_exchange=True)
    comp.atmospheric_net_heat_exchange(dt=1.0e-12)

    assert np.all(np.isfinite(comp.Q_net))


def test_outlet_gradient_and_fixed_value_without_value_branches():
    grid = _make_grid()
    comp = RiverTemperatureDynamics(
        grid, heat_exchange=False, outlet_boundary_condition="gradient_preserving"
    )
    T = grid.at_node["surface_water__temperature"]
    T[:] = np.arange(grid.number_of_nodes, dtype=float)
    comp._apply_outlet_boundary_conditions()
    outlet = grid.nodes_at_right_edge
    assert np.allclose(T[outlet], 2.0 * T[outlet - 1] - T[outlet - 2])

    grid2 = _make_grid()
    comp2 = RiverTemperatureDynamics(
        grid2,
        heat_exchange=False,
        outlet_boundary_condition="fixed_value",
        fixed_outlet_temperature=None,
    )
    before = grid2.at_node["surface_water__temperature"].copy()
    comp2._apply_outlet_boundary_conditions()
    assert np.allclose(grid2.at_node["surface_water__temperature"], before)
