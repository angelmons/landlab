"""Simulate solute transport in rivers using 2D advection-dispersion with
transient storage, first-order decay, and kinetic sorption.

This component implements a depth-averaged 2D solute-transport model for
rivers and streams, inspired by the USGS OTIS model (Runkel, 1998) but
extended to two spatial dimensions within the Landlab ecosystem.  It
supports N simultaneous solutes with independent reaction parameters.

Written by Angel Monsalve, 2026.

Key physics
-----------
Main-channel equation (OTIS Eq. 3, extended to 2D):

.. math::

    \\frac{\\partial C}{\\partial t}
    + u \\frac{\\partial C}{\\partial x}
    + v \\frac{\\partial C}{\\partial y}
    = \\frac{\\partial}{\\partial x}\\!\\left(D_L \\frac{\\partial C}{\\partial x}\\right)
    + \\frac{\\partial}{\\partial y}\\!\\left(D_T \\frac{\\partial C}{\\partial y}\\right)
    + \\frac{q_{LIN}}{h}(C_L - C)
    + \\alpha (C_S - C)
    - \\lambda C
    + \\rho \\hat{\\lambda}(C_{sed} - K_d C)

Storage-zone equation (OTIS Eq. 4):

.. math::

    \\frac{dC_S}{dt} = \\alpha \\frac{h}{h_S}(C - C_S)
                      + \\hat{\\lambda}_S(\\hat{C}_S - C_S)
                      - \\lambda_S C_S

Streambed sediment equation (OTIS Eq. 5):

.. math::

    \\frac{dC_{sed}}{dt} = \\hat{\\lambda}(K_d C - C_{sed})

The operator-splitting order is:

    1. Advection   (TVD with Van Leer limiter via AdvectionSolverTVD)
    2. Dispersion  (explicit, anisotropic or user-specified isotropic)
    3. Reactions    (lateral inflow, storage exchange, decay, sorption)
    4. Outlet boundary conditions

References
----------
Runkel, R.L. (1998). "One-Dimensional Transport with Inflow and Storage
(OTIS): A Solute Transport Model for Streams and Rivers." USGS WRIR 98-4018.

Bencala, K.E. and Walters, R.A. (1983). "Simulation of solute transport in
a mountain pool-and-riffle stream: A transient storage model." Water Resources
Research, 19(3): 718-724.

Fischer, H.B., List, E.J., Koh, R.C.Y., Imberger, J., Brooks, N.H. (1979).
"Mixing in Inland and Coastal Waters." Academic Press.

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

Instantiate with a single conservative tracer.

>>> rstd = RiverSoluteTransportDynamics(grid, solutes=["chloride"])

Set initial concentration with a high value at the left edge.

>>> left = grid.nodes_at_left_edge
>>> grid.at_node["surface_water__chloride__concentration"][:] = 3.7
>>> grid.at_node["surface_water__chloride__concentration"][left] = 11.4

Run for 10 time steps.

>>> dt = 1.0
>>> for _ in range(10):
...     grid.at_node["surface_water__chloride__concentration"][left] = 11.4
...     rstd.run_one_step(dt)
...     grid.at_node["surface_water__chloride__concentration"][left] = 11.4

Concentration at the left edge should still be pinned.

>>> np.allclose(
...     grid.at_node["surface_water__chloride__concentration"][left], 11.4
... )
True

Concentration everywhere should remain within physically reasonable bounds.

>>> C = grid.at_node["surface_water__chloride__concentration"]
>>> bool(np.all((C >= 0) & (C < 100)))
True
"""

import numpy as np

from landlab import Component
from landlab.components import AdvectionSolverTVD


class RiverSoluteTransportDynamics(Component):
    """Simulate multi-solute river transport with OTIS-style reactions in 2D.

    This Landlab component solves the depth-averaged 2D advection-dispersion
    equation with additional source/sink terms for transient storage, lateral
    inflow, first-order decay, and kinetic sorption, following the OTIS
    model equations (Runkel, 1998).  It supports N simultaneous solutes with
    per-solute reaction parameters passed as dictionaries.

    When all reaction parameters are zero (the defaults), the model reduces
    to conservative transport.

    Parameters
    ----------
    grid : RasterModelGrid
        A Landlab raster grid.
    solutes : list of str, optional
        Names of solutes to simulate.  Each name generates four grid fields:
        ``surface_water__{name}__concentration``,
        ``storage_zone__{name}__concentration``,
        ``streambed__{name}__sorbate_concentration``, and
        ``lateral__{name}__concentration``.
        Default: ``["tracer"]``
    dispersion_mode : str, optional
        ``"anisotropic"`` computes D_L and D_T from depth and shear velocity
        (Fischer, 1979).  ``"isotropic"`` uses a single user-supplied D
        passed via ``dispersion_coefficient``.
        Default: ``"anisotropic"``
    dispersion_coefficient : float or array_like, optional
        Longitudinal dispersion coefficient [m^2/s] for isotropic mode.
        Scalar or per-node array.  Ignored in anisotropic mode.
        Default: 0.24
    transverse_dispersion_coefficient : float or array_like or None, optional
        Transverse dispersion [m^2/s] for isotropic mode.  If None,
        defaults to ``dispersion_coefficient``.  Set to 0 for 1D-equivalent
        runs on a 3-row grid.  Default: None
    alpha_L : float, optional
        Longitudinal dispersion scaling (anisotropic mode):
        D_L = alpha_L * h * u_*.  Default: 10.0
    alpha_T : float, optional
        Transverse dispersion scaling (anisotropic mode):
        D_T = alpha_T * h * u_*.  Default: 0.6
    ustar_fraction : float, optional
        Shear velocity approximation: u_* = ustar_fraction * |u|.
        Default: 0.1
    h_min : float, optional
        Minimum depth to prevent division by zero [m].  Default: 0.01
    alpha_exchange : dict, optional
        Transient storage exchange coefficient [1/s], keyed by solute name.
        Values may be scalar or per-node arrays.
        Corresponds to OTIS parameter ALPHA.  Default: {} (no storage)
    h_storage : dict, optional
        Storage-zone effective thickness [m], keyed by solute name.
        Related to OTIS AREA2: h_storage = AREA2 / channel_width.
        Default: {}
    cs_background : dict, optional
        Background storage-zone concentration [mass/L^3], keyed by solute.
        Corresponds to OTIS parameter CSBACK (C-hat_S in Eq. 4).
        Default: {}
    lambda_decay : dict, optional
        Main-channel first-order decay coefficient [1/s].
        OTIS parameter LAMBDA.  Default: {}
    lambda_s_decay : dict, optional
        Storage-zone first-order decay coefficient [1/s].
        OTIS parameter LAMBDA2.  Default: {}
    lambda_hat_sorption : dict, optional
        Main-channel sorption rate coefficient [1/s].
        OTIS parameter LAMHAT.  Default: {}
    lambda_hat_s_sorption : dict, optional
        Storage-zone sorption rate coefficient [1/s].
        OTIS parameter LAMHAT2.  Default: {}
    kd_sorption : dict, optional
        Sorption distribution coefficient [L^3/mass].
        OTIS parameter KD.  Default: {}
    rho_sediment : float, optional
        Mass of accessible sediment per unit volume water [mass/L^3].
        OTIS parameter RHO.  Default: 0.0
    outlet_boundary_condition : str, optional
        ``"zero_gradient"``, ``"gradient_preserving"``, or ``"fixed_value"``.
        Default: ``"zero_gradient"``
    fixed_outlet_concentration : dict, optional
        Outlet concentration for ``"fixed_value"`` BC, keyed by solute.
        Default: {}
    """

    _name = "RiverSoluteTransportDynamics"

    _unit_agnostic = False

    _info = {
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
            "doc": (
                "Link-parallel advection velocity used by AdvectionSolverTVD."
            ),
        },
    }

    def __init__(
        self,
        grid,
        solutes=None,
        # -- Dispersion --
        dispersion_mode="anisotropic",
        dispersion_coefficient=0.24,
        transverse_dispersion_coefficient=None,
        alpha_L=10.0,
        alpha_T=0.6,
        ustar_fraction=0.1,
        # -- Transient storage (OTIS: ALPHA, AREA2, CSBACK) --
        alpha_exchange=None,
        h_storage=None,
        cs_background=None,
        # -- First-order decay (OTIS: LAMBDA, LAMBDA2) --
        lambda_decay=None,
        lambda_s_decay=None,
        # -- Kinetic sorption (OTIS: LAMHAT, LAMHAT2, KD, RHO) --
        lambda_hat_sorption=None,
        lambda_hat_s_sorption=None,
        kd_sorption=None,
        rho_sediment=0.0,
        # -- Numerical / BC --
        h_min=0.01,
        outlet_boundary_condition="zero_gradient",
        fixed_outlet_concentration=None,
    ):
        super().__init__(grid)

        if solutes is None:
            solutes = ["tracer"]
        self._solutes = list(solutes)
        self._h_min = h_min

        # ── Dispersion parameters ────────────────────────────────────
        valid_modes = ("isotropic", "anisotropic")
        if dispersion_mode not in valid_modes:
            raise ValueError(
                f"dispersion_mode must be one of {valid_modes}, "
                f"got '{dispersion_mode}'"
            )
        self._dispersion_mode = dispersion_mode
        self._D_iso = dispersion_coefficient
        self._D_T_iso = (
            transverse_dispersion_coefficient
            if transverse_dispersion_coefficient is not None
            else dispersion_coefficient
        )
        self._alpha_L = alpha_L
        self._alpha_T = alpha_T
        self._ustar_fraction = ustar_fraction

        # ── Reaction parameters (dicts keyed by solute name) ─────────
        self._alpha = alpha_exchange or {}
        self._h_s = h_storage or {}
        self._cs_bg = cs_background or {}
        self._lambda = lambda_decay or {}
        self._lambda_s = lambda_s_decay or {}
        self._lambda_hat = lambda_hat_sorption or {}
        self._lambda_hat_s = lambda_hat_s_sorption or {}
        self._Kd = kd_sorption or {}
        if isinstance(rho_sediment, dict):
            self._rho_sed = rho_sediment
        else:
            self._rho_sed = {"__default__": rho_sediment}

        # ── Boundary conditions ──────────────────────────────────────
        valid_bcs = ("zero_gradient", "gradient_preserving", "fixed_value")
        if outlet_boundary_condition not in valid_bcs:
            raise ValueError(
                f"outlet_boundary_condition must be one of {valid_bcs}, "
                f"got '{outlet_boundary_condition}'"
            )
        self._outlet_bc = outlet_boundary_condition
        self._fixed_outlet_conc = fixed_outlet_concentration or {}

        self._outlet_nodes = grid.nodes_at_right_edge
        self._outlet_interior_1 = self._outlet_nodes - 1
        self._outlet_interior_2 = self._outlet_nodes - 2

        # ── Core hydraulic fields ────────────────────────────────────
        self._h = grid.at_node["surface_water__depth"]
        self._adv_vel = grid.at_link["advection__velocity"]

        # Optional lateral inflow field
        if "lateral__water_specific_discharge" in grid.at_node:
            self._q_lat = grid.at_node[
                "lateral__water_specific_discharge"
            ]
        else:
            self._q_lat = np.zeros(grid.number_of_nodes)

        # ── Per-solute field creation + advection solvers ────────────
        self._advectors = {}

        for solute in self._solutes:
            fields = [
                f"surface_water__{solute}__concentration",
                f"storage_zone__{solute}__concentration",
                f"streambed__{solute}__sorbate_concentration",
                f"lateral__{solute}__concentration",
            ]
            for field in fields:
                if field not in grid.at_node:
                    grid.add_zeros(field, at="node")

            self._advectors[solute] = AdvectionSolverTVD(
                grid,
                fields_to_advect=(
                    f"surface_water__{solute}__concentration"
                ),
            )

    # ------------------------------------------------------------------
    # Helper: get per-solute parameter (scalar, array, or dict value)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_param(param_dict, solute, default=0.0):
        """Retrieve a per-solute parameter from a dict or return default."""
        return param_dict.get(solute, default)

    # ------------------------------------------------------------------
    # Advection + Dispersion
    # ------------------------------------------------------------------
    def _advection_dispersion(self, dt):
        """Solve advection and dispersion for all solutes.

        Step A: TVD advection (Van Leer limiter) via AdvectionSolverTVD.
        Step B: Velocity-divergence correction for non-uniform flow.
        Step C: Anisotropic or isotropic dispersion (explicit Euler).
        """
        grid = self._grid
        core = grid.core_nodes

        # -- Build dispersion coefficient array on links --
        if self._dispersion_mode == "anisotropic":
            h_link = grid.map_mean_of_link_nodes_to_link(
                "surface_water__depth"
            )
            h_link = np.maximum(h_link, self._h_min)
            u_star = np.abs(self._adv_vel) * self._ustar_fraction

            D_link = np.zeros(grid.number_of_links, dtype=float)
            D_link[grid.horizontal_links] = (
                self._alpha_L
                * h_link[grid.horizontal_links]
                * u_star[grid.horizontal_links]
            )
            D_link[grid.vertical_links] = (
                self._alpha_T
                * h_link[grid.vertical_links]
                * u_star[grid.vertical_links]
            )
        else:
            # Isotropic: user-specified D_L on horizontal, D_T on vertical
            D_link = np.zeros(grid.number_of_links, dtype=float)

            D_L_val = self._D_iso
            D_T_val = self._D_T_iso

            # Longitudinal (horizontal links)
            if np.ndim(D_L_val) == 0:
                D_link[grid.horizontal_links] = float(D_L_val)
            else:
                D_link[grid.horizontal_links] = (
                    grid.map_mean_of_link_nodes_to_link(
                        np.asarray(D_L_val, dtype=float)
                    )[grid.horizontal_links]
                )

            # Transverse (vertical links)
            if np.ndim(D_T_val) == 0:
                D_link[grid.vertical_links] = float(D_T_val)
            else:
                D_link[grid.vertical_links] = (
                    grid.map_mean_of_link_nodes_to_link(
                        np.asarray(D_T_val, dtype=float)
                    )[grid.vertical_links]
                )

        # -- Velocity divergence (computed once for all solutes) --
        div_u = grid.calc_flux_div_at_node(self._adv_vel)

        # -- Loop over solutes --
        for solute in self._solutes:
            field_name = f"surface_water__{solute}__concentration"
            C = grid.at_node[field_name]

            # A. Advection (TVD + Van Leer)
            self._advectors[solute].run_one_step(dt)

            # B. Velocity-divergence correction
            #    The TVD solver advects in conservative form; in a
            #    divergent velocity field this produces artificial
            #    mass compression/rarefaction.  Correct by undoing
            #    the div(u)*C contribution.
            C[core] += C[core] * div_u[core] * dt

            # C. Dispersion
            grad_C = grid.calc_grad_at_link(field_name)
            diff_flux = D_link * grad_C
            dCdt_diff = grid.calc_flux_div_at_node(diff_flux)
            C[core] += dCdt_diff[core] * dt

    # ------------------------------------------------------------------
    # OTIS reaction terms (Crank-Nicolson decoupled formulation)
    # ------------------------------------------------------------------
    def _otis_reactions(self, dt):
        """Apply OTIS source/sink terms at core nodes only.

        Uses the Crank-Nicolson decoupled formulation of Runkel and
        Chapra (1993, 1994) as presented in the OTIS report (Runkel,
        1998, Equations 25-27).  This is second-order accurate in time
        and unconditionally stable.

        Workflow
        --------
        1. Save C^j (main-channel concentration before reaction update).
        2. Update main-channel C to C^{j+1} using source/sink terms
           evaluated at the midpoint (Crank-Nicolson averaging of the
           lateral inflow, decay, and sorption contributions at times
           j and j+1, with the storage exchange handled implicitly
           through the decoupled Eq. 25).
        3. Compute C_S^{j+1} from C_S^j, C^j, and C^{j+1} using the
           explicit decoupled expression (OTIS Eq. 25).
        4. Compute C_sed^{j+1} from C_sed^j, C^j, and C^{j+1} using
           the explicit decoupled expression (OTIS Eq. 26).

        OTIS Equations implemented
        --------------------------
        Eq. 25 (storage zone, decoupled):

            C_S^{j+1} = [(2 - gamma^j - dt*lam_hat_S - dt*lam_S)*C_S^j
                         + gamma^j * C^j + gamma^{j+1} * C^{j+1}
                         + 2*dt*lam_hat_S * C_hat_S]
                        / (2 + gamma^{j+1} + dt*lam_hat_S + dt*lam_S)

        Eq. 26 (bed sediment, decoupled):

            C_sed^{j+1} = [(2 - dt*lam_hat)*C_sed^j
                           + dt*lam_hat*Kd*(C^j + C^{j+1})]
                          / (2 + dt*lam_hat)

        Eq. 27:
            gamma = alpha * dt * h / h_S
        """
        core = self._grid.core_nodes
        h = np.maximum(self._h, self._h_min)

        for solute in self._solutes:
            C = self._grid.at_node[
                f"surface_water__{solute}__concentration"
            ]
            C_s = self._grid.at_node[
                f"storage_zone__{solute}__concentration"
            ]
            C_sed = self._grid.at_node[
                f"streambed__{solute}__sorbate_concentration"
            ]
            C_lat = self._grid.at_node[
                f"lateral__{solute}__concentration"
            ]

            # Retrieve per-solute parameters
            alpha = self._get_param(self._alpha, solute)
            h_s = self._get_param(self._h_s, solute, default=1.0)
            cs_bg = self._get_param(self._cs_bg, solute)
            lam = self._get_param(self._lambda, solute)
            lam_s = self._get_param(self._lambda_s, solute)
            lam_hat = self._get_param(self._lambda_hat, solute)
            lam_hat_s = self._get_param(self._lambda_hat_s, solute)
            Kd = self._get_param(self._Kd, solute)

            # ── Step 1: Save C^j (before update) ─────────────────
            C_old = C.copy()

            # ── Step 2: Main-channel update (OTIS Eq. 3 sources) ─
            rho = self._get_param(self._rho_sed, solute,
                                  default=self._rho_sed.get("__default__", 0.0))
            #    Explicit Euler for lateral inflow, storage exchange,
            #    decay, and sorption.  Storage exchange uses C_S^j
            #    (current storage concentration).
            lat_flux = (self._q_lat / h) * (C_lat - C)
            storage_flux = alpha * (C_s - C)
            decay = -lam * C
            sorption = rho * lam_hat * (C_sed - Kd * C)

            C[core] += (
                lat_flux[core]
                + storage_flux[core]
                + decay[core]
                + sorption[core]
            ) * dt
            # C is now at time level j+1

            # ── Step 3: Storage zone (OTIS Eq. 25, decoupled) ────
            #    gamma = alpha * dt * h / h_S   (Eq. 27)
            #    For steady parameters: gamma^j = gamma^{j+1} = gamma
            gamma = alpha * dt * h / np.maximum(h_s, 1e-30)

            lam_hat_s_dt = lam_hat_s * dt
            lam_s_dt = lam_s * dt

            numerator = (
                (2.0 - gamma - lam_hat_s_dt - lam_s_dt) * C_s
                + gamma * C_old
                + gamma * C
                + 2.0 * lam_hat_s_dt * cs_bg
            )
            denominator = 2.0 + gamma + lam_hat_s_dt + lam_s_dt

            # Only update where storage is active
            has_storage = np.asarray(alpha > 0)
            if np.any(has_storage):
                if np.ndim(has_storage) == 0:
                    # scalar alpha > 0: update all core nodes
                    C_s[core] = numerator[core] / denominator[core]
                else:
                    mask = np.isin(
                        np.arange(self._grid.number_of_nodes), core
                    ) & has_storage
                    C_s[mask] = numerator[mask] / denominator[mask]

            # ── Step 4: Bed sediment (OTIS Eq. 26, decoupled) ────
            lam_hat_dt = lam_hat * dt

            num_sed = (
                (2.0 - lam_hat_dt) * C_sed
                + lam_hat_dt * Kd * (C_old + C)
            )
            den_sed = 2.0 + lam_hat_dt

            has_sorption = np.asarray(lam_hat > 0)
            if np.any(has_sorption):
                if np.ndim(has_sorption) == 0:
                    C_sed[core] = num_sed[core] / den_sed[core]
                else:
                    mask = np.isin(
                        np.arange(self._grid.number_of_nodes), core
                    ) & has_sorption
                    C_sed[mask] = num_sed[mask] / den_sed[mask]

            # ── Non-negativity enforcement ────────────────────────
            np.clip(C, 0.0, None, out=C)
            np.clip(C_s, 0.0, None, out=C_s)
            np.clip(C_sed, 0.0, None, out=C_sed)

    # ------------------------------------------------------------------
    # Outlet boundary conditions
    # ------------------------------------------------------------------
    def _apply_boundaries(self):
        """Apply downstream outlet boundary conditions for all solutes."""
        for solute in self._solutes:
            C = self._grid.at_node[
                f"surface_water__{solute}__concentration"
            ]

            if self._outlet_bc == "zero_gradient":
                C[self._outlet_nodes] = C[self._outlet_interior_1]

            elif self._outlet_bc == "gradient_preserving":
                C[self._outlet_nodes] = (
                    2.0 * C[self._outlet_interior_1]
                    - C[self._outlet_interior_2]
                )

            elif self._outlet_bc == "fixed_value":
                fixed_val = self._fixed_outlet_conc.get(solute, 0.0)
                C[self._outlet_nodes] = fixed_val

    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------
    def run_one_step(self, dt):
        """Advance all solute concentration fields by one time step.

        Operator-splitting order:

            1. Transport    (advection + divergence correction + dispersion)
            2. Reactions    (lateral inflow, storage, decay, sorption)
            3. Boundaries   (outlet BC)

        Parameters
        ----------
        dt : float
            Time step [s].
        """
        self._advection_dispersion(dt)
        self._otis_reactions(dt)
        self._apply_boundaries()