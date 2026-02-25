"""
Unit tests for
landlab.components.river_temperature_dynamics.RiverTemperatureDynamics

Tests are organized into four groups:

    1. Component metadata (name, fields, units, initialization)
    2. Transport physics (conservation, advection, dispersion, BCs)
    3. Energy budget (one test per flux term, independently verified)
    4. Integration (analytical solution, coupled run, stability)

Reference values for the energy budget tests were computed by hand and
cross-checked against published
saturation-vapor-pressure tables (WMO, 2018).

last updated: 02/23/2026
"""

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverTemperatureDynamics

# ======================================================================
# Helpers
# ======================================================================


def _make_grid(nrows=10, ncols=10, dx=10.0):
    """Return a grid with every required field already created."""
    grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
    grid.add_zeros("surface_water__depth", at="node")
    grid.add_zeros("surface_water__velocity", at="link")
    grid.add_zeros("advection__velocity", at="link")
    grid.add_zeros("surface_water__temperature", at="node")
    grid.add_zeros("air__temperature", at="node")
    grid.add_zeros("air__relative_humidity", at="node")
    grid.add_zeros("air__velocity", at="node")
    grid.add_zeros("radiation__incoming_shortwave_flux", at="node")
    grid.add_zeros("solar__altitude_angle", at="node")
    grid.add_zeros("cloud_cover__fraction", at="node")
    grid.add_zeros("groundwater__specific_discharge", at="node")
    grid.add_zeros("groundwater__temperature", at="node")
    grid.add_zeros("sediment__temperature", at="node")
    return grid


def _make_grid_with_conditions(
    Tw=20.0,
    Ta=25.0,
    RH=50.0,
    wind=2.0,
    Q_sw=800.0,
    alt_deg=60.0,
    h=0.5,
    shade=0.3,
    sigma_lw_factor=1.0,
):
    """Return a grid, component, and reference values for energy budget
    tests. All nodes get identical, spatially uniform conditions.
    Neutralizes the new physics (clouds, GW, bed) by default to preserve
    the baseline WMO/Tetens flux validation.
    """
    grid = _make_grid(nrows=5, ncols=8, dx=10.0)
    grid.at_node["surface_water__depth"][:] = h
    grid.at_node["surface_water__temperature"][:] = Tw
    grid.at_node["air__temperature"][:] = Ta
    grid.at_node["air__relative_humidity"][:] = RH
    grid.at_node["air__velocity"][:] = wind
    grid.at_node["radiation__incoming_shortwave_flux"][:] = Q_sw
    grid.at_node["solar__altitude_angle"][:] = np.radians(alt_deg)

    # Neutralize new physics for baseline tests
    grid.at_node["cloud_cover__fraction"][:] = 0.0
    grid.at_node["groundwater__specific_discharge"][:] = 0.0
    grid.at_node["groundwater__temperature"][:] = Tw
    grid.at_node["sediment__temperature"][:] = Tw  # Prevents bed flux

    rtd = RiverTemperatureDynamics(
        grid,
        shade_factor=shade,
        sigma_lw_factor=sigma_lw_factor,
        h_ws=2.0,
        rug_terreno=0.01,
        wind_adj=1.0,
    )
    return grid, rtd


# ======================================================================
# Shared fixture
# ======================================================================


@pytest.fixture
def rtd():
    grid = _make_grid()
    grid.at_node["surface_water__depth"][:] = 0.5
    return RiverTemperatureDynamics(grid)


# ======================================================================
# 1.  COMPONENT METADATA
# ======================================================================


def test_name(rtd):
    """Component name must match the class attribute."""
    assert rtd.name == "RiverTemperatureDynamics"


def test_input_var_names(rtd):
    """All declared input fields must be listed."""
    expected = {
        "advection__velocity",
        "air__relative_humidity",
        "air__temperature",
        "air__velocity",
        "radiation__incoming_shortwave_flux",
        "solar__altitude_angle",
        "surface_water__depth",
        "surface_water__temperature",
        "surface_water__velocity",
        "cloud_cover__fraction",
        "groundwater__specific_discharge",
        "groundwater__temperature",
        "sediment__temperature",
    }
    assert set(rtd.input_var_names) == expected


def test_output_var_names(rtd):
    """Both water and sediment temperature have intent 'inout'."""
    expected = {"surface_water__temperature", "sediment__temperature"}
    assert set(rtd.output_var_names) == expected


def test_var_units(rtd):
    """Spot-check units for key fields."""
    assert rtd.var_units("surface_water__temperature") == "deg C"
    assert rtd.var_units("surface_water__depth") == "m"
    assert rtd.var_units("advection__velocity") == "m/s"
    assert rtd.var_units("air__temperature") == "deg C"
    assert rtd.var_units("air__relative_humidity") == "%"
    assert rtd.var_units("radiation__incoming_shortwave_flux") == "W/m^2"
    assert rtd.var_units("solar__altitude_angle") == "rad"
    assert rtd.var_units("sediment__temperature") == "deg C"


def test_initialization():
    """Fields are created during init if missing, and re-init works."""
    grid = _make_grid()
    grid.at_node["surface_water__depth"][:] = 0.5

    rtd = RiverTemperatureDynamics(grid)

    # All fields must exist
    for field_name, meta in rtd._info.items():
        loc = meta["mapping"]
        assert field_name in grid[loc], f"Missing field: {field_name} at {loc}"

    # Re-initializing on the same grid (fields already exist) must not raise
    rtd2 = RiverTemperatureDynamics(grid)
    assert rtd2.name == "RiverTemperatureDynamics"


def test_invalid_outlet_bc_raises():
    """Passing an invalid outlet_boundary_condition must raise ValueError."""
    grid = _make_grid()
    grid.at_node["surface_water__depth"][:] = 0.5
    with pytest.raises(ValueError, match="outlet_boundary_condition"):
        RiverTemperatureDynamics(grid, outlet_boundary_condition="invalid")


# ======================================================================
# 2.  TRANSPORT PHYSICS
# ======================================================================


def test_thermal_mass_conservation():
    """With no advection, no dispersion source, and no atmospheric exchange,
    total thermal energy must be conserved within the domain."""
    grid = _make_grid(nrows=10, ncols=10, dx=10.0)
    grid.at_node["surface_water__depth"][:] = 0.5
    T = grid.at_node["surface_water__temperature"]
    T[:] = 15.0
    center = grid.number_of_nodes // 2
    T[center] = 25.0

    # No velocity → no advection; set k_bed=0 to isolate transport physics
    rtd = RiverTemperatureDynamics(
        grid, shade_factor=0.0, sigma_lw_factor=1.0, k_bed=0.0
    )

    core = grid.core_nodes
    initial_mass = np.sum(T[core] * grid.at_node["surface_water__depth"][core])

    for _ in range(50):
        rtd.run_one_step(1.0)

    final_mass = np.sum(T[core] * grid.at_node["surface_water__depth"][core])

    rel_change = abs(final_mass - initial_mass) / initial_mass
    assert rel_change < 0.01, f"Thermal mass change: {rel_change:.4%}"


def test_uniform_temperature_stays_uniform():
    """Spatially uniform temperature must remain uniform."""
    grid = _make_grid(nrows=6, ncols=8, dx=10.0)
    grid.at_node["surface_water__depth"][:] = 0.5
    grid.at_node["surface_water__temperature"][:] = 18.0
    grid.at_node["sediment__temperature"][:] = 18.0

    rtd = RiverTemperatureDynamics(
        grid, shade_factor=0.0, sigma_lw_factor=1.0, k_bed=0.0
    )

    for _ in range(20):
        rtd.run_one_step(1.0)

    T = grid.at_node["surface_water__temperature"]
    core = grid.core_nodes
    T_range = T[core].max() - T[core].min()
    assert T_range < 0.01, f"Spatial variation: {T_range:.6f}°C"


def test_advection_downstream():
    """A warm patch must move in the direction of flow."""
    dx = 2.0
    nrows, ncols = 7, 101
    grid = _make_grid(nrows=nrows, ncols=ncols, dx=dx)
    grid.at_node["surface_water__depth"][:] = 0.5

    T = grid.at_node["surface_water__temperature"]
    T[:] = 15.0
    x0 = 30.0
    for n in grid.core_nodes:
        T[n] = 15.0 + 10.0 * np.exp(-((grid.x_of_node[n] - x0) ** 2) / (2 * 5.0**2))

    v = 0.5
    grid.at_link["advection__velocity"][grid.horizontal_links] = v

    rtd = RiverTemperatureDynamics(
        grid, shade_factor=0.0, sigma_lw_factor=1.0, k_bed=0.0
    )

    center_row = nrows // 2
    T_center_0 = T.reshape(nrows, ncols)[center_row, :].copy()
    x_peak_0 = np.arange(ncols)[np.argmax(T_center_0)] * dx

    for _ in range(100):
        rtd.run_one_step(0.5)

    T_center_1 = T.reshape(nrows, ncols)[center_row, :]
    x_peak_1 = np.arange(ncols)[np.argmax(T_center_1)] * dx

    assert x_peak_1 > x_peak_0, f"Peak did not move downstream: {x_peak_0} → {x_peak_1}"

    expected_shift = v * 50.0  # 25 m
    actual_shift = x_peak_1 - x_peak_0
    assert (
        abs(actual_shift - expected_shift) / expected_shift < 0.30
    ), f"Peak shift {actual_shift:.1f} m vs expected {expected_shift:.1f} m"


def test_dispersion_spreading():
    """A warm patch must spread over time (peak decreases)."""
    dx = 2.0
    nrows, ncols = 7, 101
    grid = _make_grid(nrows=nrows, ncols=ncols, dx=dx)
    grid.at_node["surface_water__depth"][:] = 0.5
    grid.at_link["advection__velocity"][grid.horizontal_links] = 0.5

    T = grid.at_node["surface_water__temperature"]
    T[:] = 15.0
    for n in grid.core_nodes:
        T[n] = 15.0 + 10.0 * np.exp(-((grid.x_of_node[n] - 50.0) ** 2) / (2 * 6.0**2))
    T_max_0 = T.max()

    rtd = RiverTemperatureDynamics(
        grid, shade_factor=0.0, sigma_lw_factor=1.0, k_bed=0.0
    )
    for _ in range(200):
        rtd.run_one_step(0.5)

    T_max_1 = T.max()
    assert T_max_1 < T_max_0, f"Peak did not decrease: {T_max_0} → {T_max_1}"


def test_anisotropic_dispersion():
    """Longitudinal dispersion (D_L) must exceed transverse dispersion (D_T)."""
    dx = 2.0
    nrows, ncols = 21, 101
    grid = _make_grid(nrows=nrows, ncols=ncols, dx=dx)
    grid.at_node["surface_water__depth"][:] = 0.5

    T = grid.at_node["surface_water__temperature"]
    T[:] = 15.0
    cx, cy = 50.0, dx * (nrows - 1) / 2.0
    for n in grid.core_nodes:
        rx = grid.x_of_node[n] - cx
        ry = grid.y_of_node[n] - cy
        T[n] = 15.0 + 10.0 * np.exp(-(rx**2 + ry**2) / (2 * 4.0**2))

    grid.at_link["advection__velocity"][grid.horizontal_links] = 0.5

    rtd = RiverTemperatureDynamics(
        grid,
        shade_factor=0.0,
        sigma_lw_factor=1.0,
        alpha_L=10.0,
        alpha_T=0.6,
        k_bed=0.0,
    )

    for _ in range(200):
        rtd.run_one_step(0.5)

    T_2d = T.reshape(nrows, ncols)
    warm = T_2d > 15.5
    rows_warm, cols_warm = np.where(warm)
    if len(cols_warm) > 0 and len(rows_warm) > 0:
        x_spread = (cols_warm.max() - cols_warm.min()) * dx
        y_spread = (rows_warm.max() - rows_warm.min()) * dx
        assert (
            x_spread > y_spread
        ), f"Longitudinal spread ({x_spread} m) should exceed transverse ({y_spread} m)"


def test_outlet_bc_zero_gradient():
    """Zero-gradient BC: outlet temperature must equal the adjacent interior column."""
    dx = 2.0
    nrows, ncols = 7, 51
    grid = _make_grid(nrows=nrows, ncols=ncols, dx=dx)
    grid.at_node["surface_water__depth"][:] = 0.5
    grid.at_link["advection__velocity"][grid.horizontal_links] = 0.8

    T = grid.at_node["surface_water__temperature"]
    T[:] = 15.0
    for n in range(grid.number_of_nodes):
        T[n] = 15.0 + 10.0 * np.exp(-((grid.x_of_node[n] - 20.0) ** 2) / (2 * 6.0**2))

    rtd = RiverTemperatureDynamics(
        grid,
        shade_factor=0.0,
        sigma_lw_factor=1.0,
        k_bed=0.0,
        outlet_boundary_condition="zero_gradient",
    )
    for _ in range(200):
        rtd.run_one_step(0.5)

    T_2d = T.reshape(nrows, ncols)
    np.testing.assert_array_almost_equal(T_2d[:, -1], T_2d[:, -2], decimal=10)


def test_outlet_bc_fixed_value():
    """Fixed-value BC: outlet temperature must equal the prescribed value."""
    dx = 2.0
    nrows, ncols = 7, 51
    grid = _make_grid(nrows=nrows, ncols=ncols, dx=dx)
    grid.at_node["surface_water__depth"][:] = 0.5
    grid.at_link["advection__velocity"][grid.horizontal_links] = 0.8

    T = grid.at_node["surface_water__temperature"]
    T[:] = 15.0
    for n in range(grid.number_of_nodes):
        T[n] = 15.0 + 10.0 * np.exp(-((grid.x_of_node[n] - 20.0) ** 2) / (2 * 6.0**2))

    T_fixed = 12.0
    rtd = RiverTemperatureDynamics(
        grid,
        shade_factor=0.0,
        sigma_lw_factor=1.0,
        k_bed=0.0,
        outlet_boundary_condition="fixed_value",
        fixed_outlet_temperature=T_fixed,
    )
    for _ in range(200):
        rtd.run_one_step(0.5)

    T_outlet = T.reshape(nrows, ncols)[:, -1]
    np.testing.assert_array_almost_equal(T_outlet, T_fixed, decimal=10)


def test_outlet_bc_gradient_preserving():
    """Gradient-preserving BC: T_boundary = 2*T_{i-1} - T_{i-2}."""
    dx = 2.0
    nrows, ncols = 7, 51
    grid = _make_grid(nrows=nrows, ncols=ncols, dx=dx)
    grid.at_node["surface_water__depth"][:] = 0.5
    grid.at_link["advection__velocity"][grid.horizontal_links] = 0.8

    T = grid.at_node["surface_water__temperature"]
    T[:] = 15.0
    for n in range(grid.number_of_nodes):
        T[n] = 15.0 + 10.0 * np.exp(-((grid.x_of_node[n] - 20.0) ** 2) / (2 * 6.0**2))

    rtd = RiverTemperatureDynamics(
        grid,
        shade_factor=0.0,
        sigma_lw_factor=1.0,
        k_bed=0.0,
        outlet_boundary_condition="gradient_preserving",
    )
    for _ in range(200):
        rtd.run_one_step(0.5)

    T_2d = T.reshape(nrows, ncols)
    expected = 2.0 * T_2d[:, -2] - T_2d[:, -3]
    np.testing.assert_array_almost_equal(T_2d[:, -1], expected, decimal=10)


# ======================================================================
# 3.  ENERGY BUDGET — individual flux terms
# ======================================================================


def test_vapor_pressure_formulation_A():
    vp = RiverTemperatureDynamics._vapor_pressure_mmHg
    expected_20 = -7.348 + 20.0 * 1.2441
    assert abs(vp(np.array([20.0]))[0] - expected_20) < 1e-10


def test_vapor_pressure_formulation_B():
    vp = RiverTemperatureDynamics._vapor_pressure_mbar
    expected_20 = (-193.717 + 20.0 * 126.5763) * 0.01
    assert abs(vp(np.array([20.0]))[0] - expected_20) < 1e-10


def test_shortwave_net_mid_altitude():
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0,
        Ta=25.0,
        RH=50.0,
        Q_sw=800.0,
        alt_deg=70.0,
        shade=0.3,
    )
    rtd.atmospheric_net_heat_exchange(60.0)
    expected = (800.0 - 2.1 * 800.0 / 100.0) * (1.0 - 0.3)
    np.testing.assert_allclose(rtd.Q_sw_net[grid.core_nodes], expected, rtol=1e-6)


def test_shortwave_net_low_altitude():
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0,
        Ta=25.0,
        RH=50.0,
        Q_sw=800.0,
        alt_deg=30.0,
        shade=0.3,
    )
    rtd.atmospheric_net_heat_exchange(60.0)
    x = 4.0
    refl = (
        284.2
        - 286.793 * x
        + 132.161 * x**2
        - 34.3354 * x**3
        + 5.17431 * x**4
        - 0.42125 * x**5
        + 0.0143056 * x**6
    )
    expected = (800.0 - refl * 800.0 / 100.0) * (1.0 - 0.3)
    np.testing.assert_allclose(rtd.Q_sw_net[grid.core_nodes], expected, rtol=1e-6)


def test_shortwave_net_high_altitude():
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0,
        Ta=25.0,
        RH=50.0,
        Q_sw=800.0,
        alt_deg=85.0,
        shade=0.3,
    )
    rtd.atmospheric_net_heat_exchange(60.0)
    refl = -0.01 * 85.0 + 2.9
    expected = (800.0 - refl * 800.0 / 100.0) * 0.7
    np.testing.assert_allclose(rtd.Q_sw_net[grid.core_nodes], expected, rtol=1e-6)


def test_shortwave_nighttime():
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0,
        Ta=25.0,
        RH=50.0,
        Q_sw=0.0,
        alt_deg=0.0,
        shade=0.3,
    )
    rtd.atmospheric_net_heat_exchange(60.0)
    np.testing.assert_array_equal(rtd.Q_sw_net, 0.0)


def test_longwave_incoming():
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0, Ta=25.0, RH=50.0, sigma_lw_factor=1.0
    )
    rtd.atmospheric_net_heat_exchange(60.0)
    sigma = 5.67e-8
    e_sat_air = -16.583 + 25.0 * 1.6135
    e_air = e_sat_air * 50.0 / 100.0
    eps_a = 0.7 + 0.031 * np.sqrt(e_air)
    expected = eps_a * sigma * (25.0 + 273.15) ** 4
    np.testing.assert_allclose(rtd.Q_lw_in[grid.core_nodes], expected, rtol=1e-6)


def test_longwave_incoming_legacy_factor():
    _, rtd_std = _make_grid_with_conditions(sigma_lw_factor=1.0)
    _, rtd_leg = _make_grid_with_conditions(sigma_lw_factor=1.159)
    rtd_std.atmospheric_net_heat_exchange(60.0)
    rtd_leg.atmospheric_net_heat_exchange(60.0)
    core = rtd_std._grid.core_nodes
    ratio = rtd_leg.Q_lw_in[core[0]] / rtd_std.Q_lw_in[core[0]]
    np.testing.assert_allclose(ratio, 1.159, rtol=1e-6)


def test_longwave_reflected():
    grid, rtd = _make_grid_with_conditions(sigma_lw_factor=1.0)
    rtd.atmospheric_net_heat_exchange(60.0)
    core = grid.core_nodes
    np.testing.assert_allclose(
        rtd.Q_lw_reflected[core], 0.03 * rtd.Q_lw_in[core], rtol=1e-12
    )


def test_longwave_outgoing():
    grid, rtd = _make_grid_with_conditions(Tw=20.0, sigma_lw_factor=1.0)
    rtd.atmospheric_net_heat_exchange(60.0)
    sigma = 5.67e-8
    expected = 0.97 * sigma * (20.0 + 273.15) ** 4
    np.testing.assert_allclose(rtd.Q_lw_out[grid.core_nodes], expected, rtol=1e-6)


def test_longwave_outgoing_scales_with_temperature():
    _, rtd_cool = _make_grid_with_conditions(Tw=10.0)
    _, rtd_warm = _make_grid_with_conditions(Tw=30.0)
    rtd_cool.atmospheric_net_heat_exchange(60.0)
    rtd_warm.atmospheric_net_heat_exchange(60.0)
    core = rtd_cool._grid.core_nodes
    assert rtd_warm.Q_lw_out[core[0]] > rtd_cool.Q_lw_out[core[0]]


def test_evaporative_heat_loss():
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0, Ta=25.0, RH=50.0, wind=2.0, sigma_lw_factor=1.0
    )
    rtd.atmospheric_net_heat_exchange(60.0)
    rug = 0.01
    v_w = 2.0 * np.log(7.0 / rug) / np.log(2.0 / rug)
    e_sat_w = (-193.717 + 20.0 * 126.5763) * 0.01
    e_sat_a = (-979.786 + 25.0 * 165.8797) * 0.01
    e_air = e_sat_a * 50.0 / 100.0
    LW = 1000.0 * (2499.0 - 2.36 * 20.0)
    f_w = 2.81e-9 + 0.14e-9 * v_w
    E = f_w * (e_sat_w - e_air)
    expected = E * LW * 1000.0
    np.testing.assert_allclose(rtd.Q_evap[grid.core_nodes], expected, rtol=1e-6)


def test_evaporative_increases_with_deficit():
    _, rtd_humid = _make_grid_with_conditions(RH=90.0)
    _, rtd_dry = _make_grid_with_conditions(RH=20.0)
    rtd_humid.atmospheric_net_heat_exchange(60.0)
    rtd_dry.atmospheric_net_heat_exchange(60.0)
    core = rtd_humid._grid.core_nodes
    assert rtd_dry.Q_evap[core[0]] > rtd_humid.Q_evap[core[0]]


def test_convective_heat_loss():
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0, Ta=25.0, RH=50.0, wind=2.0, sigma_lw_factor=1.0
    )
    rtd.atmospheric_net_heat_exchange(60.0)
    rug = 0.01
    v_w = 2.0 * np.log(7.0 / rug) / np.log(2.0 / rug)
    LW = 1000.0 * (2499.0 - 2.36 * 20.0)
    f_w = 2.81e-9 + 0.14e-9 * v_w
    expected = 1000.0 * LW * f_w * 0.61 * (20.0 - 25.0)
    np.testing.assert_allclose(rtd.Q_conv[grid.core_nodes], expected, rtol=1e-6)


def test_convective_changes_sign():
    _, rtd_warm = _make_grid_with_conditions(Tw=30.0, Ta=20.0)
    _, rtd_cool = _make_grid_with_conditions(Tw=15.0, Ta=25.0)
    rtd_warm.atmospheric_net_heat_exchange(60.0)
    rtd_cool.atmospheric_net_heat_exchange(60.0)
    core = rtd_warm._grid.core_nodes
    assert rtd_warm.Q_conv[core[0]] > 0
    assert rtd_cool.Q_conv[core[0]] < 0


def test_net_heat_balance():
    """Net heat flux must equal the algebraic sum of all components."""
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0,
        Ta=25.0,
        RH=50.0,
        wind=2.0,
        Q_sw=800.0,
        alt_deg=60.0,
        shade=0.3,
        sigma_lw_factor=1.0,
    )
    rtd.atmospheric_net_heat_exchange(60.0)
    core = grid.core_nodes

    expected_net = (
        rtd.Q_sw_net[core]
        + (rtd.Q_lw_in[core] - rtd.Q_lw_reflected[core])
        - rtd.Q_lw_out[core]
        - rtd.Q_conv[core]
        - rtd.Q_evap[core]
        + rtd.Q_bed[core]
        + rtd.Q_gw[core]
    )
    np.testing.assert_allclose(rtd.Q_net[core], expected_net, rtol=1e-12)

    # Baseline calculation with Q_bed=0 and Q_gw=0 matches original validation
    np.testing.assert_allclose(rtd.Q_net[core[0]], 457.9573, rtol=1e-4)


def test_temperature_update_from_heat_flux():
    grid, rtd = _make_grid_with_conditions(
        Tw=20.0,
        Ta=25.0,
        RH=50.0,
        wind=2.0,
        Q_sw=800.0,
        alt_deg=60.0,
        shade=0.3,
        sigma_lw_factor=1.0,
    )
    T_before = grid.at_node["surface_water__temperature"].copy()
    dt = 60.0
    rtd.atmospheric_net_heat_exchange(dt)

    T_after = grid.at_node["surface_water__temperature"]
    core = grid.core_nodes
    dT_obtained = T_after[core] - T_before[core]
    dT_expected = rtd.Q_net[core] * dt / (0.5 * 4186.0 * 1000.0)

    np.testing.assert_allclose(dT_obtained, dT_expected, rtol=1e-10)


def test_wind_height_correction():
    grid7, rtd7 = _make_grid_with_conditions(Tw=20.0, Ta=25.0, wind=2.0)
    rtd7._h_ws = 7.0
    grid2, rtd2 = _make_grid_with_conditions(Tw=20.0, Ta=25.0, wind=2.0)

    rtd7.atmospheric_net_heat_exchange(60.0)
    rtd2.atmospheric_net_heat_exchange(60.0)

    core = grid7.core_nodes
    assert rtd2.Q_evap[core[0]] > rtd7.Q_evap[core[0]]


def test_shade_factor():
    _, rtd_no_shade = _make_grid_with_conditions(shade=0.0)
    _, rtd_half_shade = _make_grid_with_conditions(shade=0.5)
    _, rtd_full_shade = _make_grid_with_conditions(shade=1.0)

    rtd_no_shade.atmospheric_net_heat_exchange(60.0)
    rtd_half_shade.atmospheric_net_heat_exchange(60.0)
    rtd_full_shade.atmospheric_net_heat_exchange(60.0)

    core = rtd_no_shade._grid.core_nodes
    np.testing.assert_allclose(rtd_full_shade.Q_sw_net[core], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        rtd_half_shade.Q_sw_net[core], 0.5 * rtd_no_shade.Q_sw_net[core], rtol=1e-12
    )


def test_longwave_cloud_cover():
    """Cloud cover fraction should increase incoming longwave radiation."""
    grid_clear, rtd_clear = _make_grid_with_conditions(
        Ta=25.0, RH=50.0, sigma_lw_factor=1.0
    )
    grid_cloud, rtd_cloud = _make_grid_with_conditions(
        Ta=25.0, RH=50.0, sigma_lw_factor=1.0
    )

    grid_cloud.at_node["cloud_cover__fraction"][:] = 0.5

    rtd_clear.atmospheric_net_heat_exchange(60.0)
    rtd_cloud.atmospheric_net_heat_exchange(60.0)

    core = grid_clear.core_nodes

    # Check expected multiplier: 1 + 0.17 * (0.5)^2 = 1.0425
    expected_cloudy_lw = rtd_clear.Q_lw_in[core[0]] * 1.0425
    np.testing.assert_allclose(
        rtd_cloud.Q_lw_in[core[0]], expected_cloudy_lw, rtol=1e-6
    )


def test_bed_conduction_flux():
    """Warmer sediment bed should inject heat into the water column."""
    grid, rtd = _make_grid_with_conditions(Tw=20.0)

    # Make the bed warmer than the water
    grid.at_node["sediment__temperature"][:] = 25.0

    # Force the flux
    rtd.atmospheric_net_heat_exchange(60.0)

    core = grid.core_nodes
    # Q_bed = (k_bed / (0.5 * dz_bed)) * (T_bed - T) = (1.5 / 0.25) * 5.0 = 30.0 W/m^2
    np.testing.assert_allclose(rtd.Q_bed[core], 30.0, rtol=1e-6)

    # Check that the bed cooled down over the time step
    assert grid.at_node["sediment__temperature"][core[0]] < 25.0


def test_groundwater_exchange_flux():
    """Cold groundwater discharge should extract heat from the water column."""
    grid, rtd = _make_grid_with_conditions(Tw=20.0)

    # Add cold groundwater discharge
    grid.at_node["groundwater__temperature"][:] = 10.0
    grid.at_node["groundwater__specific_discharge"][:] = 1e-5  # m/s

    rtd.atmospheric_net_heat_exchange(60.0)

    core = grid.core_nodes
    # Q_gw = rho * cp * q_gw * (T_gw - T)
    expected_Q_gw = 1000.0 * 4186.0 * 1e-5 * (10.0 - 20.0)  # -418.6 W/m^2
    np.testing.assert_allclose(rtd.Q_gw[core], expected_Q_gw, rtol=1e-6)


def test_dynamic_meteorology_from_file(tmp_path):
    """Component should successfully interpolate and apply pandas time-series data."""
    import pandas as pd

    # Create a dummy meteorological CSV
    df = pd.DataFrame(
        {
            "time_sec": [0, 3600, 7200],
            "T_air": [10.0, 15.0, 20.0],
            "RH": [50.0, 55.0, 60.0],
            "u_wind": [1.0, 2.0, 3.0],
            "Q_sw": [0.0, 400.0, 800.0],
            "cloud_cover": [0.0, 0.5, 1.0],
        }
    )
    csv_path = tmp_path / "met_data.csv"
    df.to_csv(csv_path, index=False)

    grid = _make_grid()
    rtd = RiverTemperatureDynamics(grid, met_file=str(csv_path), k_bed=0.0)

    # Run the component advancing to t = 3600 s
    rtd.run_one_step(60.0, t_sim=3600.0)

    core = grid.core_nodes
    # Verify the atmospheric fields were updated correctly via interpolation
    np.testing.assert_allclose(grid.at_node["air__temperature"][core], 15.0)
    np.testing.assert_allclose(grid.at_node["cloud_cover__fraction"][core], 0.5)


# ======================================================================
# 4.  INTEGRATION TESTS
# ======================================================================


def test_analytical_advection_diffusion():
    """Compare numerical solution against the exact analytical solution."""
    dx = 2.0
    nrows, ncols = 13, 301
    grid = _make_grid(nrows=nrows, ncols=ncols, dx=dx)

    h_val = 0.5
    grid.at_node["surface_water__depth"][:] = h_val
    v = 0.7968
    grid.at_link["advection__velocity"][grid.horizontal_links] = v
    grid.at_link["surface_water__velocity"][grid.horizontal_links] = v

    T = grid.at_node["surface_water__temperature"]
    T_bg = 15.0
    dT_amp = 10.0
    x0 = 100.0
    sx0 = 8.0

    T[:] = T_bg
    for n in grid.core_nodes:
        rx = grid.x_of_node[n] - x0
        T[n] = T_bg + dT_amp * np.exp(-(rx**2) / (2 * sx0**2))

    # Disable bed conduction to test pure transport physics against analytical
    rtd = RiverTemperatureDynamics(
        grid,
        shade_factor=0.0,
        sigma_lw_factor=1.0,
        alpha_L=10.0,
        alpha_T=0.6,
        ustar_fraction=0.1,
        outlet_boundary_condition="zero_gradient",
        k_bed=0.0,
    )

    u_star = 0.1 * v
    D_L = 10.0 * h_val * u_star

    dt = 0.5
    t_final = 100.0
    bank_mask = (grid.y_of_node < dx) | (grid.y_of_node > dx * (nrows - 2))

    for _step in range(int(t_final / dt)):
        rtd.run_one_step(dt)
        T[bank_mask] = T_bg

    sx = np.sqrt(sx0**2 + 2 * D_L * t_final)
    center_row = nrows // 2
    x_arr = np.arange(ncols) * dx
    T_exact = T_bg + dT_amp * (sx0 / sx) * np.exp(
        -((x_arr - x0 - v * t_final) ** 2) / (2 * sx**2)
    )

    T_numerical = T.reshape(nrows, ncols)[center_row, :]

    x_peak_num = x_arr[np.argmax(T_numerical)]
    x_peak_exact = x0 + v * t_final
    assert (
        abs(x_peak_num - x_peak_exact) <= dx
    ), f"Peak position error: {x_peak_num} vs {x_peak_exact}"

    peak_num = T_numerical.max()
    peak_exact = T_exact.max()
    rel_err = abs(peak_num - peak_exact) / (peak_exact - T_bg)
    assert rel_err < 0.05, f"Peak amplitude error: {rel_err:.2%}"

    interior = (x_arr > 20) & (x_arr < dx * (ncols - 1) - 20)
    L2 = np.sqrt(np.mean((T_numerical[interior] - T_exact[interior]) ** 2))
    assert L2 < 0.3, f"L2 error too large: {L2:.4f} °C"


def test_run_one_step():
    """Integration test: run 10 steps and check temperature remains physical."""
    grid = _make_grid(nrows=5, ncols=8, dx=10.0)
    grid.at_node["surface_water__depth"][:] = 0.5
    grid.at_link["advection__velocity"][grid.horizontal_links] = 0.25
    grid.at_node["air__temperature"][:] = 25.0
    grid.at_node["air__relative_humidity"][:] = 50.0
    grid.at_node["air__velocity"][:] = 2.0
    grid.at_node["radiation__incoming_shortwave_flux"][:] = 800.0
    grid.at_node["solar__altitude_angle"][:] = np.radians(60.0)
    grid.at_node["surface_water__temperature"][:] = 15.0

    # Add baseline values for new physics
    grid.at_node["cloud_cover__fraction"][:] = 0.0
    grid.at_node["groundwater__specific_discharge"][:] = 0.0
    grid.at_node["groundwater__temperature"][:] = 15.0
    grid.at_node["sediment__temperature"][:] = 15.0

    rtd = RiverTemperatureDynamics(grid, shade_factor=0.3, sigma_lw_factor=1.0)

    for _ in range(10):
        rtd.run_one_step(60.0)

    T = grid.at_node["surface_water__temperature"]

    assert np.all(np.isfinite(T))
    assert np.all(T > -5)
    assert np.all(T < 60)


def test_time_step_sensitivity():
    """Results should not be heavily dependent on time step choice."""
    dt_coarse = 1.0
    dt_fine = 0.25
    t_total = 50.0

    def run_case(dt):
        grid = _make_grid(nrows=7, ncols=51, dx=2.0)
        grid.at_node["surface_water__depth"][:] = 0.5
        grid.at_link["advection__velocity"][grid.horizontal_links] = 0.5
        T = grid.at_node["surface_water__temperature"]
        T[:] = 15.0
        for n in grid.core_nodes:
            T[n] = 15.0 + 5.0 * np.exp(
                -((grid.x_of_node[n] - 30.0) ** 2) / (2 * 6.0**2)
            )
        rtd = RiverTemperatureDynamics(
            grid,
            shade_factor=0.0,
            sigma_lw_factor=1.0,
            k_bed=0.0,
        )
        for _ in range(int(t_total / dt)):
            rtd.run_one_step(dt)
        return T.copy()

    T_coarse = run_case(dt_coarse)
    T_fine = run_case(dt_fine)

    max_diff = np.max(np.abs(T_coarse - T_fine))
    assert (
        max_diff < 0.5
    ), f"Time step sensitivity too high: max diff = {max_diff:.4f} °C"


def test_numerical_stability():
    """Test numerical stability under various time step sizes."""
    time_steps = [1.0, 0.5, 0.25]
    t_total = 10.0

    for dt in time_steps:
        grid = _make_grid(nrows=7, ncols=21, dx=2.0)
        grid.at_node["surface_water__depth"][:] = 0.5
        grid.at_link["advection__velocity"][grid.horizontal_links] = 0.5
        T = grid.at_node["surface_water__temperature"]
        T[:] = 15.0
        T[grid.core_nodes[len(grid.core_nodes) // 2]] = 30.0

        rtd = RiverTemperatureDynamics(
            grid,
            shade_factor=0.0,
            sigma_lw_factor=1.0,
            k_bed=0.0,
        )

        for _ in range(int(t_total / dt)):
            rtd.run_one_step(dt)

            assert np.all(np.isfinite(T)), f"Non-finite T at dt={dt}"
            assert np.all(T > 0), f"Negative temperature at dt={dt}"
            assert np.all(T < 50), f"Temperature > 50°C at dt={dt}"
