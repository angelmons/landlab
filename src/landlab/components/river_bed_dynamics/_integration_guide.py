"""
Integration guide: wiring temperature effects into RiverBedDynamics v2
=======================================================================

Three existing files need targeted changes.  Nothing else in the
architecture has to move.  All changes are backward-compatible:
when ``water_temperature`` is not supplied the behaviour is
identical to the current v1 code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 1 — river_bed_dynamics.py   (the main component)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A) Add the new parameter to __init__
-------------------------------------

    def __init__(
        self,
        grid,
        ...,                              # existing params unchanged
        water_temperature: float = 20.0,  # NEW — scalar OR per-link ndarray
        variable_fluid_properties: bool = False,  # NEW — enable T effects
    ):

B) Store fluid properties at construction (after existing self._rho = rho)
---------------------------------------------------------------------------

    from ._fluid_properties import water_density, water_viscosity

    self._water_temperature = water_temperature   # stored, may be updated by RiverTemperatureDynamics
    self._variable_fluid_properties = variable_fluid_properties

    if variable_fluid_properties:
        # Initialise per-link arrays from the scalar or array temperature
        T = np.broadcast_to(
            np.asarray(water_temperature, dtype=float),
            (grid.number_of_links,),
        )
        self._rho = water_density(T)         # overrides the scalar rho param
        self._mu  = water_viscosity(T)
    else:
        # Backward-compatible: scalar constants, mu defaults to 20 °C value
        self._mu = water_viscosity(20.0)     # scalar — only used for Re_s

C) Add a public update method (called by the coupled driver loop)
-----------------------------------------------------------------

    def update_fluid_properties(self, T: float | np.ndarray) -> None:
        """Update rho and mu from a new temperature field.

        Call this every timestep when RiverTemperatureDynamics is active:

            temp_component.run_one_step(dt)
            rbd.update_fluid_properties(
                grid.at_link["surface_water__temperature"]
            )
            rbd.run_one_step(dt)

        Parameters
        ----------
        T : float or ndarray
            Water temperature [°C].  Scalar or per-link array.
        """
        from ._fluid_properties import water_density, water_viscosity
        T_arr = np.broadcast_to(
            np.asarray(T, dtype=float), (self._grid.number_of_links,)
        )
        self._water_temperature = T_arr
        self._rho = water_density(T_arr)
        self._mu  = water_viscosity(T_arr)

D) Expose rho and mu as public properties (optional but convenient)
-------------------------------------------------------------------

    @property
    def water_density_link(self) -> np.ndarray:
        return np.broadcast_to(
            np.asarray(self._rho), (self._grid.number_of_links,)
        )

    @property
    def water_dynamic_viscosity_link(self) -> np.ndarray:
        return np.broadcast_to(
            np.asarray(self._mu), (self._grid.number_of_links,)
        )


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 2 — shear_stress.py   (ShearStressCalculator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The current code computes:

    tau = rho * g * h * sf          (depth-slope product, scalar rho)

Change: read rho from rbd (already per-link after the patch above)

    rho = np.broadcast_to(rbd._rho, h_links.shape)   # per-link or scalar
    tau = rho * g * h_links * sf

That single line change gives every downstream consumer (bedload equations,
tau_star, tau_cr) the temperature-corrected density.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 3 — bedload_eq_MPM_style.py  (and all other equation files)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current tau_star calculation (line ~55 in your file):

    tau_star = shear_stress / (rho * R * g * (gs_D50 / 1000))

R = (rho_s - rho) / rho, so this is equivalent to:

    tau_star = tau / ((rho_s - rho) * g * D50)     ✓  correct formula

But rho is the scalar stored at construction.  After the patch, rho is a
per-link array, so the formula already works without any change — NumPy
broadcasting handles it.

The only targeted change needed is in the critical Shields stress path.
Replace the current block:

    # OLD — fixed or slope-based tau_star_cr
    tau_star_cr = np.full(tau_star.shape, 0.047)
    if var_cr_shear_stress:
        tau_star_cr = np.where(tau_star_cr_0 < 0.021, tau_star_cr, tau_star_cr_0)

With the temperature-aware Paphitis solve (when variable_fluid_properties=True):

    # NEW — Paphitis (2001) critical Shields stress via Re_s
    if self._variable_fluid_properties:
        from ._critical_shear_stress import compute_critical_shear_stress
        _result = compute_critical_shear_stress(
            U=self._u,                             # depth-averaged velocity
            h=self._grid.at_link["surface_water__depth"],
            D50_m=self._bed_surf__median_size_link / 1000.0,
            rho_s=self._rhos,
            rho=self._rho,                         # per-link, temperature-aware
            mu=self._mu,                           # per-link
            g=self._g,
        )
        tau_star_cr = _result["tau_cr_star"]
        # Store diagnostics for inspection / output fields
        self._delta_v_link         = _result["delta_v"]
        self._Re_s_cr_link         = _result["Re_s_cr"]
        self._u_star_cr_link        = _result["u_star_cr"]
    else:
        # Unchanged backward-compatible path
        tau_star_cr = np.full(tau_star.shape, 0.047)
        if var_cr_shear_stress:
            tau_star_cr = np.where(tau_star_cr_0 < 0.021, tau_star_cr, tau_star_cr_0)

This is the complete, minimal integration.  Parker 1990 and Wilcock-Crowe
2003 use their own reference Shields stresses (tau*_rsgo = 0.0386 and
tau*_rsg0 = 0.021 + 0.015 exp(-20 Fs)), which already depend on rho only
through shear_stress.  Because shear_stress is now temperature-corrected, both
fractional equations gain temperature sensitivity automatically with no changes
to their own code.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New Landlab output fields to register in _info
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add to the _info dict in river_bed_dynamics.py:

    "surface_water__density": {
        "dtype": float,
        "intent": "out",
        "optional": True,
        "units": "kg m^-3",
        "mapping": "link",
        "doc": "Water density at links, temperature-corrected (Heggen 1983)",
    },
    "surface_water__dynamic_viscosity": {
        "dtype": float,
        "intent": "out",
        "optional": True,
        "units": "Pa s",
        "mapping": "link",
        "doc": "Dynamic viscosity at links, temperature-corrected (Heggen 1983)",
    },
    "surface_water__viscous_sublayer_thickness": {
        "dtype": float,
        "intent": "out",
        "optional": True,
        "units": "m",
        "mapping": "link",
        "doc": "Viscous sublayer thickness delta_v = 11.6 nu / u*",
    },
    "surface_water__critical_shields_stress": {
        "dtype": float,
        "intent": "out",
        "optional": True,
        "units": "-",
        "mapping": "link",
        "doc": "Dimensionless critical Shields stress (Paphitis 2001, temperature-aware)",
    },


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Coupling to RiverTemperatureDynamics (driver loop pattern)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    flow  = RiverFlowDynamics(grid, ...)
    bed   = RiverBedDynamics(grid, ..., variable_fluid_properties=True)
    temp  = RiverTemperatureDynamics(grid, ...)

    for t in range(n_steps):
        flow.run_one_step(dt)
        temp.run_one_step(dt)               # writes surface_water__temperature
        # Pull temperature field from the shared Landlab grid and push to bed:
        bed.update_fluid_properties(
            grid.at_link["surface_water__temperature"]
        )
        bed.run_one_step(dt)                # uses T-corrected rho and mu

When RiverTemperatureDynamics is absent, simply omit the two temperature lines.
RiverBedDynamics falls back to rho=1000, mu=mu(20°C), tau_cr_star=0.047.
"""
