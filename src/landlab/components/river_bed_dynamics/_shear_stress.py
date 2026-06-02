"""
Shear stress calculator for RiverBedDynamics.

Extracts the unsteady friction-slope shear stress computation from the main
component into a focused, testable class.  The logic is identical to the
original ``RiverBedDynamics.shear_stress()`` method; only the housing changed.

Unsteady shear stress at links
-------------------------------
Two formulations are supported (selected at component construction time):

**Depth-based** (default)::

    τ = ρ g h sf

**Hydraulic-radius-based**::

    τ = ρ g R_h sf

where the unsteady friction slope is::

    sf = S₀ − ∂h/∂s − (U/g) ∂U/∂s − (1/g) ∂U/∂t

.. codeauthor:: Angel Monsalve (original implementation)
"""

from __future__ import annotations

import numpy as np


class ShearStressCalculator:
    """Computes unsteady shear stress at every link of a Landlab raster grid.

    Parameters
    ----------
    use_hydraulics_radius : bool
        When ``True``, uses ``τ = ρ g R_h sf``; when ``False`` (default),
        uses ``τ = ρ g h sf``.

    Notes
    -----
    All grid-topology arrays (cached link index sets, scratch arrays) are
    read directly from the ``rbd`` component passed to :meth:`calculate`.
    The calculator holds no mutable state of its own between calls.

    Attribute contract
    ~~~~~~~~~~~~~~~~~~
    This class reads the following private attributes from the
    ``RiverBedDynamics`` instance.  Every name here matches exactly what
    ``RiverBedDynamics.__init__`` and ``cache_topology()`` store:

    ==========================================  =====================================
    Attribute on ``rbd``                        Set in ``RiverBedDynamics`` as
    ==========================================  =====================================
    ``rbd._grid``                               ``self._grid``
    ``rbd._g``                                  ``self._g = scipy.constants.g``
    ``rbd._rho``                                ``self._rho``  (scalar or per-link
                                                ndarray when
                                                ``variable_fluid_properties=True``)
    ``rbd._topo_du_ds_scratch``                 ``self._topo_du_ds_scratch``
    ``rbd._topo_horizontal_links``              ``self._topo_horizontal_links``
    ``rbd._topo_vertical_links``                ``self._topo_vertical_links``
    ``rbd._scratch_area``                       ``self._scratch_area``
    ``rbd._scratch_perimeter``                  ``self._scratch_perimeter``
    ``rbd._surface_water__velocity_prev_time_link``
                                                ``self._surface_water__velocity_prev_time_link``
    ``rbd._boundary_links``                     ``self._boundary_links``
    ==========================================  =====================================

    Written back to ``rbd`` each call:

    ==========================================  =====================================
    ``rbd._dz_ds``                              bed-slope gradient [m m⁻¹]
    ``rbd._u``                                  link velocity [m s⁻¹]
    ``rbd._shear_stress``                       signed shear stress [Pa]
    ``rbd._surface_water__shear_stress_link``   absolute shear stress [Pa]
    ==========================================  =====================================

    Depth threshold
    ~~~~~~~~~~~~~~~
    The wetting-drying cutoff (zeroing shear stress at links shallower than
    ``_depth_threshold``) is applied by ``RiverBedDynamics.shear_stress()``
    **after** this calculator returns, not here.  That keeps ownership clear:
    this class is responsible only for the physics; the caller decides the
    transport mask.

    Temperature-aware density
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ``rbd._rho`` may be a scalar float (default) or a per-link ndarray (when
    ``variable_fluid_properties=True``).  ``np.broadcast_to`` handles both
    cases identically without conditional logic or data copies.
    """

    def __init__(
        self,
        use_hydraulics_radius: bool = False,
        formulation: str = "boussinesq",
        mannings_n: float | None = None,
    ) -> None:
        """Initialise the shear stress calculator.

        Parameters
        ----------
        use_hydraulics_radius : bool, optional
            When ``True``, compute shear stress using hydraulic radius.
        formulation : str, optional
            "boussinesq" (default) uses the full unsteady energy equation.
            "velocity_driven" uses pure velocity-driven friction.
        mannings_n : float, optional
            Required if formulation="velocity_driven".
        """
        self._use_hydraulics_radius = use_hydraulics_radius
        self._formulation = formulation
        self._mannings_n = mannings_n

        if self._formulation == "velocity_driven" and self._mannings_n is None:
            raise ValueError(
                "mannings_n must be provided when using the 'velocity_driven' formulation."
            )

    def calculate(self, rbd) -> None:
        g = rbd._grid

        # ── S₀ = −∂z/∂s ──────────────────────────────────────────────────
        # Always computed: required by downstream bedload equations for local slope corrections.
        z = g.at_node["topographic__elevation"]
        rbd._dz_ds = -g.calc_grad_at_link(z)

        rbd._u = g["link"]["surface_water__velocity"]
        h_links = g.at_link["surface_water__depth"]
        rho = np.broadcast_to(np.asarray(rbd._rho, dtype=float), h_links.shape)

        if self._formulation == "velocity_driven":

            h_safe = np.maximum(h_links, 1e-6)
            tau_mag = (
                rho
                * rbd._g
                * (self._mannings_n**2)
                * (rbd._u**2)
                / (h_safe ** (1.0 / 3.0))
            )

            # Direction strictly follows the water velocity vector
            rbd._shear_stress = tau_mag * np.sign(rbd._u)

        else:
            # ── Original Unsteady Boussinesq Formulation ───────────────────
            h = g["node"]["surface_water__depth"]
            dh_ds = g.calc_grad_at_link(h)

            # ∂U/∂s — velocity gradient at links
            du_ds = rbd._topo_du_ds_scratch
            du_ds[:] = 0.0

            u_nodes_h = g.map_mean_of_horizontal_links_to_node(rbd._u)
            hl = rbd._topo_horizontal_links
            du_ds[hl] = g.calc_grad_at_link(u_nodes_h)[hl]

            u_nodes_v = g.map_mean_of_vertical_links_to_node(rbd._u)
            u_nodes_v = u_nodes_v.reshape(g._shape[0], g._shape[1])[::-1, :]
            du_ds_v = -np.diff(u_nodes_v, axis=0) / g.dy
            vl = rbd._topo_vertical_links
            du_ds[vl] = np.flip(du_ds_v.T, axis=1).flatten(order="F")

            # ∂U/∂t — temporal velocity gradient
            u_prev = rbd._surface_water__velocity_prev_time_link
            du_dt = (rbd._u - u_prev) / g._dt

            # Friction slope sf
            sf = rbd._dz_ds - dh_ds - (rbd._u / rbd._g) * du_ds - du_dt / rbd._g

            if self._use_hydraulics_radius:
                area = rbd._scratch_area
                perimeter = rbd._scratch_perimeter
                area[hl] = h_links[hl] * g.dx
                area[vl] = h_links[vl] * g.dy
                perimeter[hl] = g.dx + 2.0 * h_links[hl]
                perimeter[vl] = g.dy + 2.0 * h_links[vl]
                rh = area / perimeter
                rbd._shear_stress = rho * rbd._g * rh * sf
            else:
                rbd._shear_stress = rho * rbd._g * h_links * sf

        # ── Zero shear stress at boundary links ────────────────────────────
        rbd._shear_stress[rbd._boundary_links] = 0.0
        rbd._surface_water__shear_stress_link = np.abs(rbd._shear_stress)

    def __repr__(self) -> str:  # pragma: no cover
        """Return a short string showing the active shear-stress formulation."""
        mode = "hydraulic-radius" if self._use_hydraulics_radius else "depth"
        return f"<ShearStressCalculator mode={mode!r}>"
