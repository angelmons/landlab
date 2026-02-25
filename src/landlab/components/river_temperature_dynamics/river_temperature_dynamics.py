"""Simulate river temperature dynamics using 2D advection-dispersion and
atmospheric energy exchange.

This component implements a depth-averaged 2D temperature model for rivers
and streams, coupling TVD advection and anisotropic dispersion with a full
atmospheric energy budget (shortwave, longwave, evaporative, convective,
bed conduction, and groundwater exchange). It is designed to be coupled with
Landlab's ``RiverFlowDynamics`` or "OverlandFlow" component.

Written by Angel Monsalve.

Key physics
-----------
Governing equation (2D depth-averaged):

dT/dt + u * dT/dx + v * dT/dy = d/dx(D_L * dT/dx) + d/dy(D_T * dT/dy)
                                     + Phi_net / (rho * c_p * h)

where Phi_net is the sum of shortwave, longwave, evaporative, convective,
bed conduction, and groundwater heat fluxes at the water surface, and D_L,
D_T are the longitudinal and transverse dispersion coefficients respectively.

Examples
--------

>>> import numpy as np
>>> from landlab import RasterModelGrid

Create a small 5 x 8 grid with 10 m spacing.

>>> grid = RasterModelGrid((5, 8), xy_spacing=10.0)

Set up hydraulic fields (uniform steady flow).

>>> _ = grid.add_zeros("surface_water__depth", at="node")
>>> grid.at_node["surface_water__depth"][:] = 0.5
>>> _ = grid.add_zeros("surface_water__velocity", at="link")
>>> grid.at_link["surface_water__velocity"][grid.horizontal_links] = 0.25
>>> _ = grid.add_zeros("advection__velocity", at="link")
>>> grid.at_link["advection__velocity"][grid.horizontal_links] = 0.25

Set up atmospheric forcing (clear-sky summer noon).

>>> _ = grid.add_zeros("air__temperature", at="node")
>>> grid.at_node["air__temperature"][:] = 25.0
>>> _ = grid.add_zeros("air__relative_humidity", at="node")
>>> grid.at_node["air__relative_humidity"][:] = 50.0
>>> _ = grid.add_zeros("air__velocity", at="node")
>>> grid.at_node["air__velocity"][:] = 2.0
>>> _ = grid.add_zeros("radiation__incoming_shortwave_flux", at="node")
>>> grid.at_node["radiation__incoming_shortwave_flux"][:] = 800.0
>>> _ = grid.add_zeros("solar__altitude_angle", at="node")
>>> grid.at_node["solar__altitude_angle"][:] = np.radians(60.0)

Initialize water temperature with a warm inflow at the left edge.

>>> _ = grid.add_zeros("surface_water__temperature", at="node")
>>> grid.at_node["surface_water__temperature"][:] = 15.0
>>> left_edge = grid.nodes_at_left_edge
>>> grid.at_node["surface_water__temperature"][left_edge] = 20.0

Instantiate and run for 10 minutes with 60 s steps.

>>> rtd = RiverTemperatureDynamics(grid, shade_factor=0.3)
>>> dt = 60.0
>>> for _ in range(10):
...     rtd.run_one_step(dt)
...     grid.at_node["surface_water__temperature"][left_edge] = 20.0
...

Temperature at the left edge should still be pinned.

>>> np.allclose(grid.at_node["surface_water__temperature"][left_edge], 20.0)
True

Temperature everywhere should remain within physically reasonable bounds.

>>> T = grid.at_node["surface_water__temperature"]
>>> bool(np.all((T > 0) & (T < 50)))
True

"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from landlab import Component
from landlab.components import AdvectionSolverTVD


class RiverTemperatureDynamics(Component):
    """Simulate the temporal and spatial evolution of river temperature.

    This Landlab component solves the depth-averaged 2D advection-dispersion
    equation coupled with a full atmospheric energy budget. The energy budget
    includes net shortwave radiation, net longwave radiation (with cloud cover
    corrections), evaporative heat loss, convective (sensible) heat loss,
    groundwater/hyporheic exchange, and bed heat conduction.

    Parameters
    ----------
    grid : RasterModelGrid
        A grid.
    rho : float, optional
        Water density. Default = 1000 kg/m^3
    cp : float, optional
        Specific heat capacity of water. Default = 4186 J/(kg*C)
    shade_factor : float, optional
        Fraction of shortwave radiation blocked by riparian shading [0-1].
        Default: 0.2
    h_ws : float, optional
        Wind measurement height above the water surface [m].
        Default: 2.0
    rug_terreno : float, optional
        Aerodynamic roughness length for the wind height log-law correction [m].
        Default: 0.01
    wind_adj : float, optional
        Empirical multiplier applied to evaporative and convective fluxes.
        Default: 1.0 (no adjustment).
    h_min : float, optional
        Minimum depth used to prevent division by zero in the heat budget [m].
        Default: 0.01
    sigma_lw_factor : float, optional
        Multiplicative correction on the Stefan-Boltzmann constant. Default: 1.159
    alpha_L : float, optional
        Longitudinal dispersion scaling coefficient. Default: 10.0
    alpha_T : float, optional
        Transverse dispersion scaling coefficient. Default: 0.6
    ustar_fraction : float, optional
        Shear velocity approximation. Default: 0.1
    outlet_boundary_condition : str, optional
        Type of boundary condition applied to the downstream edge.
        Options: "zero_gradient", "gradient_preserving", "fixed_value".
        Default: "zero_gradient".
    fixed_outlet_temperature : float or None, optional
        Temperature value [deg C] for the outlet nodes. Default: None
    met_file : str or None, optional
        Path to a CSV file containing meteorological time-series data.
        Expected columns: 'time_sec', 'T_air', 'RH', 'u_wind', 'Q_sw',
        'cloud_cover'. If provided, the component will interpolate and update
        these fields automatically during `run_one_step`. Default: None
    dz_bed : float, optional
        Thickness of the active sediment layer for bed conduction [m]. Default: 0.5
    k_bed : float, optional
        Thermal conductivity of the sediment bed [W/(m*C)]. Default: 1.5
    rho_bed : float, optional
        Density of the sediment bed [kg/m^3]. Default: 1500.0
    cp_bed : float, optional
        Specific heat capacity of the sediment bed [J/(kg*C)]. Default: 800.0
    """

    _name = "RiverTemperatureDynamics"

    _unit_agnostic = False

    _info = {
        "surface_water__temperature": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "deg C",
            "mapping": "node",
            "doc": "Depth-averaged water temperature",
        },
        "surface_water__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of water on the surface",
        },
        "surface_water__velocity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/s",
            "mapping": "link",
            "doc": "Speed of water flow above the surface",
        },
        "advection__velocity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/s",
            "mapping": "link",
            "doc": "Link-parallel advection velocity",
        },
        "air__temperature": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "deg C",
            "mapping": "node",
            "doc": "Air temperature",
        },
        "air__relative_humidity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "%",
            "mapping": "node",
            "doc": "Air relative humidity [0-100]",
        },
        "air__velocity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "Wind speed at measurement height h_ws",
        },
        "radiation__incoming_shortwave_flux": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "W/m^2",
            "mapping": "node",
            "doc": "Total incident shortwave radiation at the water surface",
        },
        "solar__altitude_angle": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "rad",
            "mapping": "node",
            "doc": "Solar altitude angle above the horizon [radians]",
        },
        "cloud_cover__fraction": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "Fraction of sky covered by clouds [0.0 - 1.0]",
        },
        "groundwater__specific_discharge": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m/s",
            "mapping": "node",
            "doc": "Specific discharge of groundwater into the channel (positive = gaining)",
        },
        "groundwater__temperature": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "deg C",
            "mapping": "node",
            "doc": "Temperature of the incoming groundwater",
        },
        "sediment__temperature": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "deg C",
            "mapping": "node",
            "doc": "Temperature of the active sediment bed layer",
        },
    }

    def __init__(
        self,
        grid,
        rho=1000.0,
        cp=4186.0,
        shade_factor=0.2,
        h_ws=2.0,
        rug_terreno=0.01,
        wind_adj=1.0,
        h_min=0.01,
        sigma_lw_factor=1.159,
        alpha_L=10.0,
        alpha_T=0.6,
        ustar_fraction=0.1,
        outlet_boundary_condition="zero_gradient",
        fixed_outlet_temperature=None,
        met_file=None,
        dz_bed=0.5,
        k_bed=1.5,
        rho_bed=1500.0,
        cp_bed=800.0,
    ):
        super().__init__(grid)

        # Physical parameters
        self._rho = rho
        self._cp = cp
        self._shade_factor = shade_factor
        self._h_ws = h_ws
        self._rug_terreno = rug_terreno
        self._wind_adj = wind_adj
        self._h_min = h_min
        self._sigma = 5.67e-8
        self._sigma_lw_factor = sigma_lw_factor

        # Dispersion parameters
        self._alpha_L = alpha_L
        self._alpha_T = alpha_T
        self._ustar_fraction = ustar_fraction

        # Bed conduction parameters
        self._dz_bed = dz_bed
        self._k_bed = k_bed
        self._rho_bed = rho_bed
        self._cp_bed = cp_bed

        # Outlet boundary condition
        valid_bcs = ("zero_gradient", "gradient_preserving", "fixed_value")
        if outlet_boundary_condition not in valid_bcs:
            raise ValueError(
                f"outlet_boundary_condition must be one of {valid_bcs}, "
                f"got {outlet_boundary_condition}"
            )
        self._outlet_bc = outlet_boundary_condition
        self._fixed_outlet_temperature = fixed_outlet_temperature

        self._outlet_nodes = grid.nodes_at_right_edge
        self._outlet_interior_1 = self._outlet_nodes - 1
        self._outlet_interior_2 = self._outlet_nodes - 2

        # Initialize core and optional fields
        for field_name, meta in self._info.items():
            if field_name not in self.grid.at_node and meta["mapping"] == "node":
                grid.add_zeros(field_name, at="node", units=meta["units"])
            elif field_name not in self.grid.at_link and meta["mapping"] == "link":
                grid.add_zeros(field_name, at="link", units=meta["units"])

        # Mapping fields
        self._T = self._grid.at_node["surface_water__temperature"]
        self._h = self._grid.at_node["surface_water__depth"]
        self._vel = self._grid.at_link["surface_water__velocity"]
        self._adv_vel = self._grid.at_link["advection__velocity"]
        self._T_air = self._grid.at_node["air__temperature"]
        self._RH = self._grid.at_node["air__relative_humidity"]
        self._v_wind = self._grid.at_node["air__velocity"]
        self._Q_sw_inc = self._grid.at_node["radiation__incoming_shortwave_flux"]
        self._alt_rad = self._grid.at_node["solar__altitude_angle"]

        # New mapping fields for upgraded physics
        self._C_cloud = self._grid.at_node["cloud_cover__fraction"]
        self._q_gw = self._grid.at_node["groundwater__specific_discharge"]
        self._T_gw = self._grid.at_node["groundwater__temperature"]
        self._T_bed = self._grid.at_node["sediment__temperature"]

        # Diagnostic flux fields
        self.Q_sw_net = np.zeros(grid.number_of_nodes)
        self.Q_lw_in = np.zeros(grid.number_of_nodes)
        self.Q_lw_reflected = np.zeros(grid.number_of_nodes)
        self.Q_lw_out = np.zeros(grid.number_of_nodes)
        self.Q_evap = np.zeros(grid.number_of_nodes)
        self.Q_conv = np.zeros(grid.number_of_nodes)
        self.Q_bed = np.zeros(grid.number_of_nodes)
        self.Q_gw = np.zeros(grid.number_of_nodes)
        self.Q_net = np.zeros(grid.number_of_nodes)

        self._advector = AdvectionSolverTVD(
            grid,
            fields_to_advect="surface_water__temperature",
        )

        # Setup meteorological time-series if provided
        self._setup_meteorology(met_file)

    @property
    def input_var_names(self):
        """Names of fields used as inputs."""
        return sorted(
            name
            for name, meta in self._info.items()
            if meta["intent"] in ("in", "inout")
        )

    @property
    def output_var_names(self):
        """Names of fields modified or produced by the component."""
        return sorted(
            name
            for name, meta in self._info.items()
            if meta["intent"] in ("out", "inout")
        )

    def _setup_meteorology(self, met_file):
        """Reads a CSV file and creates interpolation functions for forcing data."""
        if met_file is None:
            self._dynamic_met = False
            return

        self._dynamic_met = True
        df = pd.read_csv(met_file)

        self._interp_Ta = interp1d(
            df["time_sec"], df["T_air"], fill_value="extrapolate"
        )
        self._interp_RH = interp1d(df["time_sec"], df["RH"], fill_value="extrapolate")
        self._interp_wind = interp1d(
            df["time_sec"], df["u_wind"], fill_value="extrapolate"
        )
        self._interp_Qsw = interp1d(
            df["time_sec"], df["Q_sw"], fill_value="extrapolate"
        )

        if "cloud_cover" in df.columns:
            self._interp_cloud = interp1d(
                df["time_sec"], df["cloud_cover"], fill_value="extrapolate"
            )
        else:
            self._interp_cloud = lambda t: 0.0

    def update_meteorology(self, t_sim):
        """Updates the grid fields with interpolated values at current time."""
        if not self._dynamic_met:
            return

        self._T_air[:] = self._interp_Ta(t_sim)
        self._RH[:] = self._interp_RH(t_sim)
        self._v_wind[:] = self._interp_wind(t_sim)
        self._Q_sw_inc[:] = self._interp_Qsw(t_sim)
        self._C_cloud[:] = self._interp_cloud(t_sim)

    @staticmethod
    def _vapor_pressure_mmHg(T):
        """Saturation vapor pressure in mmHg (Formulation A)."""
        T = np.asarray(T, dtype=float)
        conditions = [
            T < 0,
            (T >= 0) & (T < 5),
            (T >= 5) & (T < 10),
            (T >= 10) & (T < 15),
            (T >= 15) & (T < 20),
            (T >= 20) & (T < 25),
            (T >= 25) & (T < 30),
            (T >= 30) & (T < 35),
            T >= 35,
        ]
        aa = [4.579, 4.579, 3.832, 2.051, -1.453, -7.348, -16.583, -30.280, -49.864]
        bb = [0.2832, 0.3928, 0.5332, 0.7157, 0.9493, 1.2441, 1.6135, 2.0700, 2.6296]
        a_sel = np.select(conditions, aa, default=aa[-1])
        b_sel = np.select(conditions, bb, default=bb[-1])
        return a_sel + T * b_sel

    @staticmethod
    def _vapor_pressure_mbar(T):
        """Saturation vapor pressure in mbar (Formulation B)."""
        T = np.asarray(T, dtype=float)
        conditions = [
            T <= 0,
            (T > 0) & (T <= 5),
            (T > 5) & (T <= 10),
            (T > 10) & (T <= 15),
            (T > 15) & (T <= 20),
            (T > 20) & (T <= 25),
            (T > 25) & (T <= 30),
            (T > 30) & (T <= 35),
            T > 35,
        ]
        a1 = [
            610.483,
            610.483,
            516.891,
            273.444,
            -193.717,
            -979.786,
            -2211.018,
            -4037.268,
            -6648.520,
        ]
        b1 = [
            37.7569,
            52.3690,
            71.0875,
            95.4322,
            126.5763,
            165.8797,
            215.1290,
            276.0040,
            350.6112,
        ]
        a_sel = np.select(conditions, a1, default=a1[-1])
        b_sel = np.select(conditions, b1, default=b1[-1])
        return (a_sel + T * b_sel) * 0.01

    def atmospheric_net_heat_exchange(self, dt):
        """Calculate the net heat flux (including bed and GW) and update temperature."""
        T = self._T
        Ta = self._T_air
        RH = self._RH
        h = np.maximum(self._h, self._h_min)
        C_cloud = self._C_cloud

        # 1. Shortwave radiation with reflectance and shading
        Q_sw_inc = self._Q_sw_inc
        alt = np.degrees(self._alt_rad)
        daytime = (Q_sw_inc > 0) & (alt > 0)
        refl = np.zeros_like(alt)

        mask_high = daytime & (alt >= 80)
        refl[mask_high] = -0.01 * alt[mask_high] + 2.9
        mask_mid = daytime & (alt >= 60) & (alt < 80)
        refl[mask_mid] = 2.1
        mask_low = daytime & (alt < 60)
        x = 1.0 + alt[mask_low] / 10.0
        refl[mask_low] = (
            284.2
            - 286.793 * x
            + 132.161 * x**2
            - 34.3354 * x**3
            + 5.17431 * x**4
            - 0.42125 * x**5
            + 0.0143056 * x**6
        )
        Q_sw_net = (Q_sw_inc - refl * Q_sw_inc / 100.0) * (1.0 - self._shade_factor)

        # 2. Atmospheric longwave incoming (with Cloud Correction)
        e_sat_air_mmHg = self._vapor_pressure_mmHg(Ta)
        e_air_mmHg = e_sat_air_mmHg * RH / 100.0
        eps_a = 0.7 + 0.031 * np.sqrt(e_air_mmHg)
        sigma_eff = self._sigma * self._sigma_lw_factor

        # Apply cloud cover adjustment (1 + 0.17 * C^2)
        cloud_factor = 1.0 + 0.17 * (C_cloud**2)
        Q_lw_in = eps_a * sigma_eff * (Ta + 273.15) ** 4 * cloud_factor

        # 3. Longwave reflected
        Q_lw_reflected = 0.03 * Q_lw_in

        # 4. Longwave emitted by water surface
        Q_lw_out = 0.97 * self._sigma * (T + 273.15) ** 4

        # 5. Wind height correction
        v_w = self._v_wind.copy()
        if self._h_ws != 7.0:
            v_w *= np.log(7.0 / self._rug_terreno) / np.log(
                self._h_ws / self._rug_terreno
            )

        # 6. Evaporative heat loss
        e_sat_w_mbar = self._vapor_pressure_mbar(T)
        e_air_mbar = self._vapor_pressure_mbar(Ta) * RH / 100.0
        LW = 1000.0 * (2499.0 - 2.36 * T)
        f_w = 2.81e-9 + 0.14e-9 * v_w
        E = f_w * (e_sat_w_mbar - e_air_mbar)
        Q_evap = E * LW * self._rho

        # 7. Convective (sensible) heat loss
        CBowen = 0.61
        Q_conv = self._rho * LW * f_w * CBowen * (T - Ta)

        # 8. Bed Heat Conduction
        Q_bed = (self._k_bed / (0.5 * self._dz_bed)) * (self._T_bed - T)

        # 9. Groundwater / Hyporheic Exchange
        Q_gw = self._rho * self._cp * self._q_gw * (self._T_gw - T)

        # 10. Net balance
        Q_net = (
            Q_sw_net
            + (Q_lw_in - Q_lw_reflected)
            - Q_lw_out
            - self._wind_adj * Q_conv
            - self._wind_adj * Q_evap
            + Q_bed
            + Q_gw
        )

        # 11. Apply temperature changes
        # Update water temperature
        self._T += Q_net * dt / (h * self._cp * self._rho)

        # Update active sediment layer temperature
        self._T_bed -= Q_bed * dt / (self._rho_bed * self._cp_bed * self._dz_bed)

        # Store diagnostics
        self.Q_sw_net[:] = Q_sw_net
        self.Q_lw_in[:] = Q_lw_in
        self.Q_lw_reflected[:] = Q_lw_reflected
        self.Q_lw_out[:] = Q_lw_out
        self.Q_evap[:] = Q_evap
        self.Q_conv[:] = Q_conv
        self.Q_bed[:] = Q_bed
        self.Q_gw[:] = Q_gw
        self.Q_net[:] = Q_net

    def temperature_advection_dispersion(self, dt):
        """Solve the advection-dispersion equation for temperature."""
        grid = self._grid

        self._advector.run_one_step(dt)

        h_link = grid.map_mean_of_link_nodes_to_link("surface_water__depth")
        h_link = np.maximum(h_link, self._h_min)
        u_star = np.abs(self._adv_vel) * self._ustar_fraction

        D_link = np.zeros(grid.number_of_links, dtype=float)
        D_link[grid.horizontal_links] = (
            self._alpha_L
            * h_link[grid.horizontal_links]
            * u_star[grid.horizontal_links]
        )
        D_link[grid.vertical_links] = (
            self._alpha_T * h_link[grid.vertical_links] * u_star[grid.vertical_links]
        )

        grad_T = grid.calc_grad_at_link("surface_water__temperature")
        diff_flux = D_link * grad_T
        dTdt_diff = grid.calc_flux_div_at_node(diff_flux)

        core = grid.core_nodes
        self._T[core] += dTdt_diff[core] * dt

    def _apply_outlet_boundary_conditions(self):
        """Apply the user-selected boundary condition at the outlet."""
        T = self._T

        if self._outlet_bc == "zero_gradient":
            T[self._outlet_nodes] = T[self._outlet_interior_1]
        elif self._outlet_bc == "gradient_preserving":
            T[self._outlet_nodes] = (
                2.0 * T[self._outlet_interior_1] - T[self._outlet_interior_2]
            )
        elif self._outlet_bc == "fixed_value":
            if self._fixed_outlet_temperature is not None:
                T[self._outlet_nodes] = self._fixed_outlet_temperature

    def run_one_step(self, dt, t_sim=None):
        """Advance the temperature field by one time step *dt* [s].

        Parameters
        ----------
        dt : float
            Time step [s].
        t_sim : float, optional
            Current simulation time [s]. Used to update meteorological
            boundary conditions if a met_file was provided.
        """
        if t_sim is not None:
            self.update_meteorology(t_sim)

        self.temperature_advection_dispersion(dt)
        self.atmospheric_net_heat_exchange(dt)
        self._apply_outlet_boundary_conditions()
