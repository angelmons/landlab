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
.. codeauthor:: Phase-3B refactor — class extraction
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
    """

    def __init__(self, use_hydraulics_radius: bool = False) -> None:
        """Initialise the shear stress calculator.

        Parameters
        ----------
        use_hydraulics_radius : bool, optional
            When ``True``, compute shear stress as ``τ = ρ g R_h sf`` using
            the hydraulic radius ``R_h = A / P`` (cross-sectional area over
            wetted perimeter).  When ``False`` (default), use the simpler
            depth-slope product ``τ = ρ g h sf``.  The hydraulic-radius
            formulation is more accurate for wide channels where the aspect
            ratio is not very large.
        """
        self._use_hydraulics_radius = use_hydraulics_radius

    def calculate(self, rbd) -> None:
        """Compute shear stress at every link and store results on *rbd*.

        Reads all required fields from the RiverBedDynamics component
        instance ``rbd`` and writes back:

        * ``rbd._dz_ds`` — bed-slope gradient at links [m m⁻¹]
        * ``rbd._u`` — current link velocity [m s⁻¹] (alias for grid field)
        * ``rbd._shear_stress`` — signed shear stress at links [Pa]
        * ``rbd._surface_water__shear_stress_link`` — absolute value [Pa]

        Parameters
        ----------
        rbd : RiverBedDynamics
            The component instance.  Must have been initialised so that all
            fields (topographic elevation, water depth/velocity, cached
            topology arrays) are available.
        """
        g = rbd._grid

        # ── S₀ = −∂z/∂s ──────────────────────────────────────────────────
        z = g.at_node["topographic__elevation"]
        rbd._dz_ds = -g.calc_grad_at_link(z)

        # ── ∂h/∂s ────────────────────────────────────────────────────────
        h = g["node"]["surface_water__depth"]
        dh_ds = g.calc_grad_at_link(h)
        h_links = g.at_link["surface_water__depth"]

        # ── ∂U/∂s — velocity gradient at links ───────────────────────────
        rbd._u = g["link"]["surface_water__velocity"]
        du_ds = rbd._topo_du_ds_scratch
        du_ds[:] = 0.0  # reset pre-allocated scratch (no malloc)

        # Horizontal component — use Landlab's built-in horizontal mapper
        u_nodes_h = g.map_mean_of_horizontal_links_to_node(rbd._u)
        hl = rbd._topo_horizontal_links
        du_ds[hl] = g.calc_grad_at_link(u_nodes_h)[hl]

        # Vertical component — reshape, reverse, finite-difference, restore
        u_nodes_v = g.map_mean_of_vertical_links_to_node(rbd._u)
        u_nodes_v = u_nodes_v.reshape(g._shape[0], g._shape[1])[::-1, :]
        du_ds_v = -np.diff(u_nodes_v, axis=0) / g.dy  # vectorised 1B
        vl = rbd._topo_vertical_links
        du_ds[vl] = np.flip(du_ds_v.T, axis=1).flatten(order="F")

        # ── ∂U/∂t — rate of change of velocity ───────────────────────────
        u_prev = rbd._surface_water__velocity_prev_time_link
        du_dt = (rbd._u - u_prev) / g._dt

        # ── Friction slope sf ─────────────────────────────────────────────
        sf = rbd._dz_ds - dh_ds - (rbd._u / rbd._g) * du_ds - du_dt / rbd._g

        # ── Shear stress at links ─────────────────────────────────────────
        if self._use_hydraulics_radius:
            hl = rbd._topo_horizontal_links
            vl = rbd._topo_vertical_links
            area = rbd._scratch_area  # pre-allocated scratch (1D)
            perimeter = rbd._scratch_perimeter
            area[hl] = h_links[hl] * g.dx
            area[vl] = h_links[vl] * g.dy
            perimeter[hl] = g.dx + 2 * h_links[hl]
            perimeter[vl] = g.dy + 2 * h_links[vl]
            rh = area / perimeter
            rbd._shear_stress = rbd._rho * rbd._g * rh * sf
        else:
            rbd._shear_stress = rbd._rho * rbd._g * h_links * sf

        # ── Boundary condition — zero flux at border links ────────────────
        rbd._shear_stress[rbd._boundary_links] = 0
        rbd._surface_water__shear_stress_link = np.abs(rbd._shear_stress)

    def __repr__(self) -> str:  # pragma: no cover
        """Return a short string showing the active shear-stress formulation."""
        mode = "hydraulic-radius" if self._use_hydraulics_radius else "depth"
        return f"<ShearStressCalculator mode={mode!r}>"
