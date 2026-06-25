"""Simulate surface fluid flow with an HLLC shallow-water solver.

This component implements a finite-volume Godunov-type approximation of the
depth-averaged 2D shallow-water equations on a RasterModelGrid. Numerical fluxes
are computed with an HLLC Riemann solver, with optional second-order MUSCL
reconstruction and hydrostatic reconstruction for well-balanced treatment of
topography. Time integration uses Strang splitting with optional implicit
Manning friction.

Written by Angel Monsalve.

References
----------
Monsalve et al., (2025). RiverFlowDynamics v1.0: A Landlab component for
computing two-dimensional river flow dynamics. Journal of Open Source Software,
10(110), 7823, https://doi.org/10.21105/joss.07823

Toro, E. F. (2001). *Shock-Capturing Methods for Free-Surface Shallow Flows*.

Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R., & Perthame, B. (2004).
A fast and stable well-balanced scheme with hydrostatic reconstruction for
shallow water flows. *SIAM J. Sci. Comput.* https://doi.org/10.1137/S1064827503431090

Notes
-----
* HLLC Riemann flux  -  correct shocks, hydraulic jumps, transcritical flow
* Audusse hydrostatic reconstruction  -  exact well-balancedness
* Strang operator splitting  -  second-order isotropy
* Transmissive (zero-gradient) outflow BCs on all edges by default
* Optional per-edge reflective wall BCs  (wall_edges parameter)
* Inflow BCs on any edge: depth + u + v at specified nodes
* Non-uniform Manning's n  -  scalar or per-node array / grid field
* Implicit Manning friction  -  no stiffness at shallow depths
* Auto-populate surface_water__velocity at links (update_link_fields=True)
* Adaptive CFL time-stepping (or user-supplied fixed dt)
* Positive-depth guarantee throughout

Examples
--------
This example mirrors the ``RiverFlowDynamics`` doctest and demonstrates a
simple sloped channel with a fixed inflow (left) and a fixed-depth outlet
(right), producing a near-uniform depth along the centerline after 10 seconds.

>>> import numpy as np
>>> from landlab import RasterModelGrid
>>> from landlab.components import RiverFlowDynamics_HLLC

Create a small grid for demonstration purposes:

>>> grid = RasterModelGrid((8, 6), xy_spacing=0.1)

Set up a sloped channel with elevated sides (slope of 0.01):

>>> z = grid.add_zeros("topographic__elevation", at="node")
>>> z += 0.005 - 0.01 * grid.x_of_node
>>> z[grid.y_of_node > 0.5] = 1.0
>>> z[grid.y_of_node < 0.2] = 1.0

Instantiating the Component. To check the names of the required inputs, use
the 'input_var_names' class property.

>>> RiverFlowDynamics_HLLC.input_var_names
('surface_water__depth', 'topographic__elevation')

Initialize the required depth field (other fields are created by the component):

>>> h = grid.add_zeros("surface_water__depth", at="node")
>>> vel = grid.add_zeros("surface_water__velocity", at="link")
>>> wse = grid.add_zeros("surface_water__elevation", at="node")
>>> wse += h + z

Set up inlet boundary conditions (left side of channel):
Water flows from left to right at a depth of 0.5 m with x-velocity of 0.45 m/s.

>>> fixed_entry_nodes = np.arange(12, 36, 6)
>>> entry_nodes_h_values = np.full(4, 0.5)
>>> entry_nodes_u_values = np.full(4, 0.45)
>>> entry_nodes_v_values = np.zeros(4)

Fix the outlet depth on the downstream edge (right side of channel):

>>> fixed_exit_nodes = np.arange(17, 41, 6)
>>> exit_nodes_eta_values = np.full(4, (z[fixed_entry_nodes] + 0.5).mean())

Instantiate RiverFlowDynamics_HLLC (enable link-velocity output for coupling):

>>> rfd = RiverFlowDynamics_HLLC(
...     grid,
...     mannings_n=0.012,
...     fixed_entry_nodes=fixed_entry_nodes,
...     entry_nodes_h_values=entry_nodes_h_values,
...     entry_nodes_u_values=entry_nodes_u_values,
...     entry_nodes_v_values=entry_nodes_v_values,
...     fixed_exit_nodes=fixed_exit_nodes,
...     exit_nodes_eta_values=exit_nodes_eta_values,
...     wall_edges={"top", "bottom"},
...     update_link_fields=True,
... )

Run the simulation for 10 seconds:

>>> target_time = 10.0
>>> dt = 0.01
>>> while rfd.elapsed_time < target_time:
...     rfd.run_one_step(dt=min(dt, target_time - rfd.elapsed_time))
...

>>> bool(np.all(grid.at_node["surface_water__depth"] >= 0.0))
True
>>> "surface_water__elevation" in grid.at_node
True
>>> "surface_water__velocity" in grid.at_link
True

Examine the flow depth at the center of the channel after 10 seconds.
Expected values are from RiverBedDynamics

>>> expected = np.array([0.5, 0.5, 0.5, 0.501, 0.502, 0.502])
>>> flow_depth = np.reshape(grid.at_node["surface_water__depth"], (8, 6))[3, :]
>>> bool(np.allclose(np.round(flow_depth, 3), expected, atol=0.02))
True

And the velocity at links along the center of the channel.
Expected values are from RiverBedDynamics

>>> expected = np.array([0.45, 0.457, 0.455, 0.452, 0.453])
>>> linksAtCenter = grid.links_at_node[np.array(np.arange(24, 30))][:-1, 0]
>>> flow_velocity = grid["link"]["surface_water__velocity"][linksAtCenter]
>>> bool(np.allclose(np.round(flow_velocity, 3), expected, atol=0.02))
True

"""

import warnings

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import diags

from landlab import Component
from landlab import RasterModelGrid

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_G = 9.80665  # gravitational acceleration [m/s²]
_H_DRY = 1e-4  # depth threshold for wet/dry distinction [m]


# ─────────────────────────────────────────────────────────────────────────────
# HLLC Riemann kernel
# ─────────────────────────────────────────────────────────────────────────────


def _swe_flux_x(h, hu, hv, g):
    """Calculate the physical shallow-water flux in the x direction."""
    inv_h = np.where(h > 0.0, 1.0 / np.where(h > 0.0, h, 1.0), 0.0)
    u = hu * inv_h
    return hu, hu * u + 0.5 * g * h**2, hu * (hv * inv_h)


def _wave_speeds(hL, uL, hR, uR, g):
    """HEstimate HLLC wave-speeds with dry-front correction.

    Uses Einfeldt-Roe estimates on wet/wet interfaces (the standard choice)
    and the two-rarefaction (dry-front) estimates of Brufau et al. (2002) /
    Toro (2001 §10.5.1) on wet/dry interfaces.  The dry-front correction is
    essential for correct propagation speed of the wetting front in
    dam-break problems on a dry bed: at a wet/dry interface the standard
    Einfeldt-Roe estimate collapses to S_R ≈ u_L + c_L/√2, whereas the
    physically correct value from the positive Riemann invariant across
    the rarefaction is S_R = u_L + 2 c_L (and symmetrically for dry/wet).

    References
    ----------
    Brufau, Vázquez-Cendón, García-Navarro (2002).  A numerical model for
        flooding and drying of irregular domains.  IJNMF 39, 247-275.
    Toro, E.F. (2001).  *Shock-Capturing Methods for Free-Surface Shallow
        Flows*, §10.5.1.
    """
    hL = np.maximum(hL, 0.0)
    hR = np.maximum(hR, 0.0)
    cL = np.sqrt(g * hL)
    cR = np.sqrt(g * hR)

    wetL = hL > _H_DRY
    wetR = hR > _H_DRY

    # --- wet/wet : Einfeldt-Roe ---
    sqL = np.sqrt(hL)
    sqR = np.sqrt(hR)
    den = sqL + sqR
    safe = den > 0.0
    u_roe = np.where(safe, (sqL * uL + sqR * uR) / np.where(safe, den, 1.0), 0.0)
    c_roe = np.sqrt(g * 0.5 * (hL + hR))
    SL_ww = np.minimum(uL - cL, u_roe - c_roe)
    SR_ww = np.maximum(uR + cR, u_roe + c_roe)

    # --- wet/dry : two-rarefaction (rarefaction reaches dry front) ---
    SL_wd = uL - cL
    SR_wd = uL + 2.0 * cL

    # --- dry/wet : two-rarefaction, mirror of above ---
    SL_dw = uR - 2.0 * cR
    SR_dw = uR + cR

    # --- per-face selection ---
    wet_wet = wetL & wetR
    wet_dry = wetL & ~wetR
    dry_wet = ~wetL & wetR
    # dry/dry → SL = SR = 0; flux is then identically zero (face skipped)

    SL = np.where(
        wet_wet, SL_ww, np.where(wet_dry, SL_wd, np.where(dry_wet, SL_dw, 0.0))
    )
    SR = np.where(
        wet_wet, SR_ww, np.where(wet_dry, SR_wd, np.where(dry_wet, SR_dw, 0.0))
    )

    # --- contact-wave speed S* (standard formula, clamped to [SL, SR]) ---
    num = hR * uR * (uR - SR) - hL * uL * (uL - SL) + 0.5 * g * (hR**2 - hL**2)
    dstar = hR * (uR - SR) - hL * (uL - SL)
    sf = np.abs(dstar) > 1e-14
    S_star = np.where(sf, num / np.where(sf, dstar, 1.0), 0.5 * (uL + uR))
    S_star = np.minimum(np.maximum(S_star, SL), SR)
    return SL, SR, S_star


def _hllc_star_flux(h, hu, hv, Fh, Fhu, Fhv, S, S_star, g):
    """Calculate HLLC star-region fluxes."""
    inv_h = np.where(h > 0.0, 1.0 / np.where(h > 0.0, h, 1.0), 0.0)
    v = hv * inv_h
    dss = S - S_star
    sf = np.abs(dss) > 1e-14
    c = h * (S - hu * inv_h) / np.where(sf, dss, np.sign(dss + 1e-30) * 1e-14)
    return (Fh + S * (c - h), Fhu + S * (c * S_star - hu), Fhv + S * (c * v - hv))


def _hllc_flux_x(hL, huL, hvL, hR, huR, hvR, g=_G):
    """Calculate the vectorized HLLC flux in the x-direction across N faces."""
    hL, huL, hvL = np.asarray(hL, float), np.asarray(huL, float), np.asarray(hvL, float)
    hR, huR, hvR = np.asarray(hR, float), np.asarray(huR, float), np.asarray(hvR, float)
    inv_hL = np.where(hL > 0, 1.0 / np.where(hL > 0, hL, 1.0), 0.0)
    inv_hR = np.where(hR > 0, 1.0 / np.where(hR > 0, hR, 1.0), 0.0)
    uL, uR = huL * inv_hL, huR * inv_hR
    FhL, FhuL, FhvL = _swe_flux_x(hL, huL, hvL, g)
    FhR, FhuR, FhvR = _swe_flux_x(hR, huR, hvR, g)
    SL, SR, SS = _wave_speeds(hL, uL, hR, uR, g)
    FLh, FLhu, FLhv = _hllc_star_flux(hL, huL, hvL, FhL, FhuL, FhvL, SL, SS, g)
    FRh, FRhu, FRhv = _hllc_star_flux(hR, huR, hvR, FhR, FhuR, FhvR, SR, SS, g)
    uL = SL >= 0.0
    uLs = (~uL) & (SS >= 0.0)
    uRs = (~uL) & (~uLs) & (SR >= 0.0)

    def w(a, b, c, d):
        """Return the wave-speed estimate for one state."""
        return np.where(uL, a, np.where(uLs, b, np.where(uRs, c, d)))

    return w(FhL, FLh, FRh, FhR), w(FhuL, FLhu, FRhu, FhuR), w(FhvL, FLhv, FRhv, FhvR)


# ─────────────────────────────────────────────────────────────────────────────
# MUSCL + Audusse reconstruction
# ─────────────────────────────────────────────────────────────────────────────


def _vanleer(a, b):
    """Calculate a Van Leer limited slope."""
    ab = a * b
    s = a + b
    return np.where(ab > 0.0, 2.0 * ab / np.where(np.abs(s) > 1e-14, s, 1.0), 0.0)


def _muscl_x(q_p):
    """Reconstruct MUSCL left and right interface states."""
    dq = q_p[:, 1:] - q_p[:, :-1]
    slp = np.zeros_like(q_p)
    slp[:, 1:-1] = 0.5 * _vanleer(dq[:, :-1], dq[:, 1:])
    return q_p[:, :-1] + slp[:, :-1], q_p[:, 1:] - slp[:, 1:]


def _hydro_recon(etaL, etaR, zL, zR):
    """Apply hydrostatic reconstruction to neighboring states."""
    zf = np.maximum(zL, zR)
    return np.maximum(0.0, etaL - zf), np.maximum(0.0, etaR - zf), zf


# ─────────────────────────────────────────────────────────────────────────────
# Directional sweeps with configurable BCs
# ─────────────────────────────────────────────────────────────────────────────


def _solve_subcritical_inflow(
    q_target,
    h_interior,
    hu_interior,
    hv_interior,
    z_ghost,
    hv_ghost_target,
    g,
    side="left",
):
    """Solve for the ghost state at a subcritical inflow boundary that is
    consistent with the prescribed discharge and the interior's outgoing
    Riemann invariant.

    SWE characteristics in subcritical flow (Fr < 1):
      * Outgoing characteristic (left boundary: goes left, out of domain):
            R⁻ = u - 2√(g h)
        carries information from the interior.
      * Incoming characteristic (left boundary: goes right, into domain):
            R⁺ = u + 2√(g h)
        carries the boundary condition.

    The user prescribes the discharge `q = h u` (one condition, matching the
    number of incoming characteristics).  We extract R⁻ from the interior
    cell and solve for the consistent ghost (h, u) satisfying:
        q = h u           and       u - 2√(g h) = R⁻_interior
    (with sign of R⁻ flipped at the right boundary).

    For the right boundary the outgoing characteristic is R⁺ = u + 2√(g h),
    and the equations become:
        q = h u           and       u + 2√(g h) = R⁺_interior

    Returns (h_ghost, hu_ghost) arrays.  `hv_ghost` is taken as
    `h_ghost * (hv_ghost_target / max(h_interior, eps))` if the user
    didn't pre-scale (i.e. it tracks the depth so v is the prescribed
    transverse velocity).

    Parameters
    ----------
    q_target : ndarray
        Prescribed discharge per unit width [m^2/s] at each row of the boundary.
        Positive values mean inflow into the domain.
    h_interior, hu_interior, hv_interior : ndarray
        State of the boundary-adjacent interior cells.
    z_ghost : ndarray
        Bed elevation at the ghost cell (typically matches the interior bed).
    hv_ghost_target : ndarray
        Prescribed transverse momentum at the boundary [m^2/s].
    g : float
    side : "left" or "right"
        Which boundary; flips the sign convention of the Riemann invariant.

    Returns
    -------
    h_ghost, hu_ghost, hv_ghost : arrays (nr,)
    """
    # u_int from interior; safe division
    h_int_safe = np.maximum(h_interior, _H_DRY)
    u_int = hu_interior / h_int_safe
    c_int = np.sqrt(g * h_int_safe)
    if side == "left":
        # Outgoing characteristic value extracted from interior:
        Rminus = u_int - 2.0 * c_int
        # Solve for ghost h from:  h * (Rminus + 2 sqrt(g h)) = q_target
        # Let s = sqrt(h); cubic in s: 2 sqrt(g) s^3 + Rminus s^2 - q = 0
        sign = +1.0
        R = Rminus
    else:  # "right"
        Rplus = u_int + 2.0 * c_int
        sign = -1.0
        R = Rplus

    # Newton iteration on the cubic; initial guess from interior h
    s = np.sqrt(h_int_safe).copy()
    for _ in range(20):
        f = 2.0 * np.sqrt(g) * sign * s**3 + R * s**2 - q_target
        df = 6.0 * np.sqrt(g) * sign * s**2 + 2.0 * R * s
        # Where df ~ 0 (degenerate), keep current s
        step = np.where(np.abs(df) > 1e-12, f / df, 0.0)
        s = s - step
        s = np.maximum(s, 1e-9)
        if np.max(np.abs(step)) < 1e-12:
            break
    h_ghost = s * s
    # u from R invariant (more accurate than q/h for tiny h)
    if side == "left":
        u_ghost = R + 2.0 * np.sqrt(g * h_ghost)
    else:
        u_ghost = R - 2.0 * np.sqrt(g * h_ghost)
    hu_ghost = h_ghost * u_ghost
    # Transverse momentum: prescribed v scaled by ghost depth
    # (hv_ghost_target is interpreted as h*v specified at the boundary,
    # but if the user supplied it already as h_in*v_in their original
    # h_in may not equal the resolved h_ghost; rescale by depth ratio
    # so the *velocity* is what the user intended).
    hv_ghost = hv_ghost_target / np.maximum(h_interior, _H_DRY) * h_ghost
    return h_ghost, hu_ghost, hv_ghost


def _build_inflow_ghost(spec, h_int_edge, hu_int_edge, hv_int_edge, g, side):
    """Resolve an inflow spec into concrete ghost arrays.

    The spec can specify the ghost state in one of two modes:

      * Mode "full"  (spec has key 'mode' == 'full' or omitted with all
        of 'h','hu','hv'):
            The ghost is the prescribed (h, hu, hv).  Only correct for
            supercritical inflow (Fr > 1) where all three variables are
            prescribed; use with care for subcritical flow because it
            over-determines the Riemann problem.

      * Mode "discharge"  (spec has key 'mode' == 'discharge' or has key
        'q' but not 'h'):
            Subcritical inflow.  The user specifies q (and optionally
            hv_target).  The ghost h and u are resolved from q together
            with the outgoing Riemann invariant from the interior, which
            is the textbook Godunov-correct subcritical inflow BC.
    """
    z = spec["z"]
    mode = spec.get("mode", None)
    if mode is None:
        mode = "discharge" if "q" in spec else "full"
    if mode == "full":
        return spec["h"], spec["hu"], spec["hv"], z
    elif mode == "discharge":
        q = spec["q"]
        hv_target = spec.get("hv", np.zeros_like(q))
        h_g, hu_g, hv_g = _solve_subcritical_inflow(
            q,
            h_int_edge,
            hu_int_edge,
            hv_int_edge,
            z,
            hv_target,
            g,
            side=side,
        )
        return h_g, hu_g, hv_g, z
    else:
        raise ValueError(f"Unknown inflow mode: {mode}")


def _pad(
    q,
    left_reflect=False,
    right_reflect=False,
    negate=False,
    left_ghost=None,
    right_ghost=None,
):
    """Pad q with ghost cells in the x-direction.

    Modes (per side, independent):
      * Prescribed ghost (`left_ghost` / `right_ghost` is an array):
        use the prescribed array as the ghost.  This is the Godunov-correct
        way to impose a Dirichlet inflow / outflow boundary: the HLLC
        Riemann problem at the boundary face is then between the prescribed
        ghost state and the interior cell, which delivers the exact
        prescribed inflow flux.  Overrides the reflect/transmissive choice.
      * Reflective (`left_reflect=True`):
        mirror the interior with momentum negation when `negate=True`.
      * Transmissive (default):
        copy the interior boundary cell as a zero-gradient extrapolation.
    """
    sign = -1.0 if negate else 1.0
    if left_ghost is not None:
        lg = left_ghost
        if lg.ndim == 1:
            lg = lg[:, None]
    else:
        lg = q[:, :1] * (sign if left_reflect else 1.0)
    if right_ghost is not None:
        rg = right_ghost
        if rg.ndim == 1:
            rg = rg[:, None]
    else:
        rg = q[:, -1:] * (sign if right_reflect else 1.0)
    return np.concatenate([lg, q, rg], axis=1)


def _x_sweep(
    h,
    hu,
    hv,
    z,
    dt,
    dx,
    g=_G,
    order=1,
    left_wall=False,
    right_wall=False,
    inflow_left=None,
    inflow_right=None,
):
    """Advance the conservative state by one x-direction sub-step.

    inflow_left / inflow_right (optional): dict with keys
        'h', 'hu', 'hv', 'z'  (each a 1-D array of length nr)
    If provided, the corresponding x-boundary ghost cells are set to these
    prescribed values rather than transmissive/reflective copies of the
    interior.  This implements a Godunov-correct Dirichlet inflow / outflow
    boundary: the HLLC Riemann problem at the boundary face is solved
    between the prescribed ghost state and the interior cell, delivering
    the exact prescribed mass and momentum flux.

    Overrides left_wall / right_wall when inflow_left / inflow_right is given.
    """
    nr, nc = h.shape
    eta = h + z

    # Prepare optional prescribed ghost arrays via the characteristic-based
    # builder when the spec is provided
    eta_lg = z_lg = hu_lg = hv_lg = None
    eta_rg = z_rg = hu_rg = hv_rg = None
    if inflow_left is not None:
        h_int_edge = h[:, 0]
        hu_int_edge = hu[:, 0]
        hv_int_edge = hv[:, 0]
        h_g, hu_g, hv_g, z_g = _build_inflow_ghost(
            inflow_left, h_int_edge, hu_int_edge, hv_int_edge, g, side="left"
        )
        eta_lg = h_g + z_g
        z_lg = z_g
        hu_lg = hu_g
        hv_lg = hv_g
    if inflow_right is not None:
        h_int_edge = h[:, -1]
        hu_int_edge = hu[:, -1]
        hv_int_edge = hv[:, -1]
        h_g, hu_g, hv_g, z_g = _build_inflow_ghost(
            inflow_right, h_int_edge, hu_int_edge, hv_int_edge, g, side="right"
        )
        eta_rg = h_g + z_g
        z_rg = z_g
        hu_rg = hu_g
        hv_rg = hv_g

    # When inflow is specified, override wall flag on that side
    eff_left_wall = left_wall and (inflow_left is None)
    eff_right_wall = right_wall and (inflow_right is None)

    eta_p = _pad(
        eta,
        eff_left_wall,
        eff_right_wall,
        negate=False,
        left_ghost=eta_lg,
        right_ghost=eta_rg,
    )
    z_p = _pad(
        z,
        eff_left_wall,
        eff_right_wall,
        negate=False,
        left_ghost=z_lg,
        right_ghost=z_rg,
    )
    hu_p = _pad(
        hu,
        eff_left_wall,
        eff_right_wall,
        negate=True,
        left_ghost=hu_lg,
        right_ghost=hu_rg,
    )
    hv_p = _pad(
        hv,
        eff_left_wall,
        eff_right_wall,
        negate=False,
        left_ghost=hv_lg,
        right_ghost=hv_rg,
    )

    if order == 2:
        # MUSCL second-order reconstruction
        etaL_2, etaR_2 = _muscl_x(eta_p)
        zL_2, zR_2 = _muscl_x(z_p)
        huL_2, huR_2 = _muscl_x(hu_p)
        hvL_2, hvR_2 = _muscl_x(hv_p)
        # First-order (cell-centered) face values for the wet/dry fallback
        etaL_1, etaR_1 = eta_p[:, :-1], eta_p[:, 1:]
        zL_1, zR_1 = z_p[:, :-1], z_p[:, 1:]
        huL_1, huR_1 = hu_p[:, :-1], hu_p[:, 1:]
        hvL_1, hvR_1 = hv_p[:, :-1], hv_p[:, 1:]
        # I2 wet/dry safety (Liang & Borthwick 2009 / Toro 2001 §11):
        # at any face whose left or right cell is dry, fall back to first
        # order.  MUSCL gradients and limiters are unreliable on
        # near-vanishing depths and inject spurious face states that
        # severely throttle wet/dry mass fluxes.
        h_p_centered = np.maximum(0.0, eta_p - z_p)
        dry_face = (h_p_centered[:, :-1] <= _H_DRY) | (h_p_centered[:, 1:] <= _H_DRY)
        etaL = np.where(dry_face, etaL_1, etaL_2)
        etaR = np.where(dry_face, etaR_1, etaR_2)
        zL = np.where(dry_face, zL_1, zL_2)
        zR = np.where(dry_face, zR_1, zR_2)
        huL = np.where(dry_face, huL_1, huL_2)
        huR = np.where(dry_face, huR_1, huR_2)
        hvL = np.where(dry_face, hvL_1, hvL_2)
        hvR = np.where(dry_face, hvR_1, hvR_2)
    else:
        etaL, etaR = eta_p[:, :-1], eta_p[:, 1:]
        zL, zR = z_p[:, :-1], z_p[:, 1:]
        huL, huR = hu_p[:, :-1], hu_p[:, 1:]
        hvL, hvR = hv_p[:, :-1], hv_p[:, 1:]

    hL_s, hR_s, _ = _hydro_recon(etaL, etaR, zL, zR)
    hLr = np.maximum(0.0, etaL - zL)
    hRr = np.maximum(0.0, etaR - zR)
    iL = np.where(hLr > _H_DRY, 1.0 / np.where(hLr > _H_DRY, hLr, 1.0), 0.0)
    iR = np.where(hRr > _H_DRY, 1.0 / np.where(hRr > _H_DRY, hRr, 1.0), 0.0)
    uL, vL = huL * iL, hvL * iL
    uR, vR = huR * iR, hvR * iR

    Fh, Fhu, Fhv = _hllc_flux_x(
        hL_s.ravel(),
        (hL_s * uL).ravel(),
        (hL_s * vL).ravel(),
        hR_s.ravel(),
        (hR_s * uR).ravel(),
        (hR_s * vR).ravel(),
        g=g,
    )
    Fh = Fh.reshape(nr, nc + 1)
    Fhu = Fhu.reshape(nr, nc + 1)
    Fhv = Fhv.reshape(nr, nc + 1)

    # Compatible (well-balanced) bed-slope source.
    #
    # The source must be the exact partner of the HLLC hydrostatic pressure
    # flux so that "lake at rest" on a non-flat bed is preserved to machine
    # precision, while contributing nothing on a flat bed (where there is
    # no bed slope), regardless of the surface gradient.
    #
    # The key requirement (Audusse et al. 2004 §2.2; Liang & Borthwick 2009
    # "compatible discretization") is that the source be built from the
    # CELL-CENTERED surface elevation referenced to the same raised-bed face
    # step `zf = max(zL, zR)` that the flux uses  -  NOT from the MUSCL-
    # reconstructed face values.  Using reconstructed eta here couples the
    # well-balancing to the slope limiter and injects spurious momentum on a
    # flat bed whenever the surface varies .
    #
    # At rest (eta = const) hLsrc = hRsrc = eta - zf at every face, so the
    # source telescopes exactly against div(F_hu); on a flat bed zf = 0 makes
    # hLsrc/hRsrc equal the cell-centered depths and the source vanishes.
    eta_cell_L = eta_p[:, :-1]
    eta_cell_R = eta_p[:, 1:]
    zf_face = np.maximum(zL, zR)
    hLsrc = np.maximum(0.0, eta_cell_L - zf_face)
    hRsrc = np.maximum(0.0, eta_cell_R - zf_face)
    Sx = 0.5 * g * (hLsrc[:, 1:] ** 2 - hRsrc[:, :-1] ** 2) / dx
    return (
        h - dt / dx * (Fh[:, 1:] - Fh[:, :-1]),
        hu - dt / dx * (Fhu[:, 1:] - Fhu[:, :-1]) + dt * Sx,
        hv - dt / dx * (Fhv[:, 1:] - Fhv[:, :-1]),
    )


def _y_sweep(
    h,
    hu,
    hv,
    z,
    dt,
    dy,
    g=_G,
    order=1,
    bottom_wall=False,
    top_wall=False,
    inflow_bottom=None,
    inflow_top=None,
):
    """Advance the conservative state by one y-direction sub-step. Implemented
    by calling _x_sweep on transposed arrays with hu/hv swapped (so the
    "x-momentum" of the sweep is the y physical momentum).  Inflow specs are
    likewise swapped: bottom -> left
    (and the v-component of inflow becomes the swept-direction "hu").
    """

    def _swap_spec(s):
        """Transpose a discharge-mode or full-mode inflow spec so that the
        sweep direction's "x-momentum" (hu) carries the physical y-momentum.
        In discharge mode the spec's q already represents the discharge
        normal to the boundary, which is what the swept-direction _x_sweep
        expects (no swap needed for q); but hv (tangential) in the original
        frame becomes the swept frame's transverse, which is again hv."""
        if s is None:
            return None
        if s.get("mode") == "discharge" or "q" in s:
            # q is already normal-to-boundary; hv stores the tangential
            # momentum in the ORIGINAL frame.  In the swept frame, the
            # tangential is the orthogonal direction, also called hv there.
            return {"mode": "discharge", "q": s["q"], "hv": s["hv"], "z": s["z"]}
        return {"h": s["h"], "hu": s["hv"], "hv": s["hu"], "z": s["z"]}

    h_T, hv_T, hu_T = _x_sweep(
        h.T,
        hv.T,
        hu.T,
        z.T,
        dt,
        dy,
        g,
        order,
        left_wall=bottom_wall,
        right_wall=top_wall,
        inflow_left=_swap_spec(inflow_bottom),
        inflow_right=_swap_spec(inflow_top),
    )
    return h_T.T, hu_T.T, hv_T.T


def _friction(h, hu, hv, dt, n_2d, g=_G):
    """Apply implicit Manning friction to momentum fields."""
    wet = h > _H_DRY
    ih = np.where(wet, 1.0 / np.where(wet, h, 1.0), 0.0)
    spd = np.sqrt((hu * ih) ** 2 + (hv * ih) ** 2)
    Cf = np.where(wet, g * n_2d**2 * spd / (h ** (4.0 / 3.0) + 1e-30), 0.0)
    fac = 1.0 / (1.0 + Cf * dt)
    return h, hu * fac, hv * fac


def _pos(h, hu, hv):
    """Enforce nonnegative water depth and mask dry-cell momentum."""
    h = np.maximum(h, 0.0)
    hu = np.where(h > _H_DRY, hu, 0.0)
    hv = np.where(h > _H_DRY, hv, 0.0)
    return h, hu, hv


def _dt(h, hu, hv, dx, dy, cfl=0.45, g=_G):
    """Calculate a stable timestep from the CFL condition."""
    wet = h > _H_DRY
    ih = np.where(wet, 1.0 / np.where(wet, h, 1.0), 0.0)
    c = np.sqrt(g * np.maximum(h, 0.0))
    mx = (np.abs(hu * ih) + c).max() if wet.any() else 0.0
    my = (np.abs(hv * ih) + c).max() if wet.any() else 0.0
    return min(cfl * dx / (mx + 1e-12), cfl * dy / (my + 1e-12))


def _step(
    h,
    hu,
    hv,
    z,
    dx,
    dy,
    dt,
    n_2d=0.0,
    g=_G,
    step_count=0,
    order=1,
    left_wall=False,
    right_wall=False,
    bottom_wall=False,
    top_wall=False,
    inflow_left=None,
    inflow_right=None,
    inflow_bottom=None,
    inflow_top=None,
):
    """Advance the HLLC solution by one internal timestep."""
    kx = {
        "g": g,
        "order": order,
        "left_wall": left_wall,
        "right_wall": right_wall,
        "inflow_left": inflow_left,
        "inflow_right": inflow_right,
    }
    ky = {
        "g": g,
        "order": order,
        "bottom_wall": bottom_wall,
        "top_wall": top_wall,
        "inflow_bottom": inflow_bottom,
        "inflow_top": inflow_top,
    }
    P = _pos
    if step_count % 2 == 0:
        h, hu, hv = P(*_x_sweep(h, hu, hv, z, dt / 2, dx, **kx))
        h, hu, hv = P(*_y_sweep(h, hu, hv, z, dt, dy, **ky))
        h, hu, hv = P(*_x_sweep(h, hu, hv, z, dt / 2, dx, **kx))
    else:
        h, hu, hv = P(*_y_sweep(h, hu, hv, z, dt / 2, dy, **ky))
        h, hu, hv = P(*_x_sweep(h, hu, hv, z, dt, dx, **kx))
        h, hu, hv = P(*_y_sweep(h, hu, hv, z, dt / 2, dy, **ky))
    if np.any(n_2d > 0):
        h, hu, hv = P(*_friction(h, hu, hv, dt, n_2d, g))
    return h, hu, hv


# ─────────────────────────────────────────────────────────────────────────────
# Landlab Component
# ─────────────────────────────────────────────────────────────────────────────


class RiverFlowDynamics_HLLC(Component):
    """Simulate 2D shallow-water flow using an HLLC solver on a RasterModelGrid.

    Parameters
    ----------
    grid : RasterModelGrid

    mannings_n : float or array_like, optional
        Manning roughness [s/m^1/3].  Scalar (uniform) or 1-D array of
        length ``grid.number_of_nodes`` (per-node).  If the grid already
        carries a ``"mannings_n_at_node"`` node field, that field takes
        precedence and this argument is ignored.  Default 0.0 (frictionless).

    cfl : float
        Courant number for adaptive time-stepping.  Default 0.45.

    g : float
        Gravity [m/s²].  Default 9.80665.

    order : int
        Spatial order: ``1`` (default, robust) or ``2`` (MUSCL Van Leer).

    fixed_entry_nodes : array_like of int, optional
        Node indices where inflow Dirichlet conditions are imposed.
        Works on any grid edge.

    entry_nodes_h_values : array_like, optional
        Water depth [m] at each ``fixed_entry_node``.

    entry_nodes_u_values : array_like, optional
        Depth-averaged x-velocity [m/s] at each entry node.
        Positive = rightward.  Default 0.

    entry_nodes_v_values : array_like, optional
        Depth-averaged y-velocity [m/s] at each entry node.
        Positive = upward.  Default 0.

    fixed_exit_nodes : array_like of int, optional
        Node indices where an outlet Dirichlet condition is imposed.
        Typically used on the downstream edge to fix stage (or depth).

    exit_nodes_eta_values : array_like, optional
        Water-surface elevation [m] imposed at each ``fixed_exit_node`` (stage-based outlet).
        If provided, depth is set as ``max(eta - z, 0)`` at those nodes.

    exit_nodes_u_values, exit_nodes_v_values : array_like, optional
        Optional velocities [m/s] imposed at outlet nodes. If omitted, the
        solver preserves the current local velocity when applying the outlet
        depth/stage (i.e., momentum is adjusted consistently with the imposed
        depth/stage).

    outlet_max_depth : float or None, optional
        Ramp-up outlet depth cap [m]. When set, the outlet depth is clamped
        to ``min(h_local, outlet_max_depth)``  -  the target depth is only
        enforced once the local water depth reaches or exceeds this value.
        This prevents the fixed-stage BC from pulling water out of a dry or
        partially-wet outlet at the start of a simulation.
        Applies only when ``fixed_exit_nodes`` is also provided.
        Default ``None`` (standard hard Dirichlet outlet).

    wall_edges : set of str, optional
        Edges treated as **reflective walls** (zero normal velocity).
        Subset of ``{'left', 'right', 'bottom', 'top'}``.
        Edges not listed use **transmissive** (zero-gradient) outflow BCs.
        Default : empty set (all edges transmissive)

    update_link_fields : bool, optional
        If ``True``, ``surface_water__velocity`` at links is updated every
        call to ``run_one_step()``.  Required for ``RiverTemperatureDynamics``
        coupling.  Default ``False``.

    Notes
    -----
    Outflow (transmissive) BC  -  the default for every edge.
    Ghost cells are set to the boundary cell value (zero-gradient), which
    allows waves and flow to exit without numerical reflection.  This is the
    standard first-order transmissive (Sommerfeld-like) outflow condition for
    hyperbolic systems and is appropriate for all open boundaries.

    Wall (reflective) BC  -  enabled via ``wall_edges``.
    The normal momentum component is negated in the ghost cell, producing
    zero normal flux at that face.  Use for physical walls or closed ends.

    Inflow BC  -  specified via ``fixed_entry_nodes``.
    Depth and velocity are overwritten before and after each step
    (Dirichlet enforcement).  Can be applied on any edge or interior nodes.

    Outlet (fixed stage/depth) BC  -  specified via ``fixed_exit_nodes``.
    Depth (or stage) is overwritten before and after each step (Dirichlet).
    Use this to constrain downstream water-surface elevation / depth for
    controlled outflow comparisons or steady channel tests.

    Non-uniform roughness  -  pass a per-node array or pre-populate
    ``"mannings_n_at_node"`` before creating the component.  The roughness
    array is used directly in the implicit friction solve each step; values
    of zero give frictionless cells.

    Link velocity field  -  set ``update_link_fields=True`` to
    automatically populate ``surface_water__velocity`` (scalar speed at
    links) after each step.  Values are the face-normal component of the
    node-averaged velocity.
    """

    _name = "RiverFlowDynamics_HLLC"
    _unit_agnostic = False

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "surface_water__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of water on the surface",
        },
        "surface_water__elevation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Water surface elevation at time N",
        },
        "surface_water__x_velocity": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "Depth-averaged x-velocity",
        },
        "surface_water__y_velocity": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "Depth-averaged y-velocity",
        },
        "surface_water__x_momentum": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m2/s",
            "mapping": "node",
            "doc": "Depth-integrated x-momentum (hu)",
        },
        "surface_water__y_momentum": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m2/s",
            "mapping": "node",
            "doc": "Depth-integrated y-momentum (hv)",
        },
        "surface_water__velocity": {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "m/s",
            "mapping": "link",
            "doc": "Speed of water flow above the surface",
        },
        "mannings_n_at_node": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "s/m^(1/3)",
            "mapping": "node",
            "doc": "Per-node Manning roughness coefficient",
        },
    }

    def __init__(
        self,
        grid,
        mannings_n=0.0,
        cfl=0.45,
        g=_G,
        order=1,
        fixed_entry_nodes=None,
        entry_nodes_h_values=None,
        entry_nodes_u_values=None,
        entry_nodes_v_values=None,
        fixed_exit_nodes=None,
        exit_nodes_h_values=None,
        exit_nodes_eta_values=None,
        exit_nodes_u_values=None,
        exit_nodes_v_values=None,
        outlet_max_depth=None,
        wall_edges=None,
        update_link_fields=False,
        use_smagorinsky=False,
        smagorinsky_cs=0.15,
        use_elder=False,
        elder_alpha=0.6,
    ):
        """Initialize RiverFlowDynamics_HLLC."""
        if not isinstance(grid, RasterModelGrid):
            raise TypeError("RiverFlowDynamics_HLLC requires a RasterModelGrid.")
        super().__init__(grid)

        self._g = float(g)
        self._cfl = float(cfl)
        self._order = int(order)
        self._step_n = 0
        self._t = 0.0
        self._update_links = bool(update_link_fields)

        nr, nc = grid.shape
        self._nr = nr
        self._nc = nc
        self._dx = grid.dx
        self._dy = grid.dy

        self._n = mannings_n  # Useful for tests
        self._mannings_n = mannings_n  # Useful for tests

        # ── Topography ────────────────────────────────────────────────────
        if "topographic__elevation" not in grid.at_node:
            raise ValueError(
                "'topographic__elevation' must be set before initialising "
                "RiverFlowDynamics_HLLC."
            )

        # ── Create output node fields if absent ───────────────────────────
        for name in [
            "surface_water__depth",
            "surface_water__elevation",
            "surface_water__x_velocity",
            "surface_water__y_velocity",
            "surface_water__x_momentum",
            "surface_water__y_momentum",
        ]:
            if name not in grid.at_node:
                grid.add_zeros(name, at="node")

        if update_link_fields and "surface_water__velocity" not in grid.at_link:
            grid.add_zeros("surface_water__velocity", at="link")

        # ── 2-D views into flat node arrays (no copy) ─────────────────────
        self._z = grid.at_node["topographic__elevation"].reshape(nr, nc)
        self._h = grid.at_node["surface_water__depth"].reshape(nr, nc)
        self._eta = grid.at_node["surface_water__elevation"].reshape(nr, nc)
        self._u = grid.at_node["surface_water__x_velocity"].reshape(nr, nc)
        self._v = grid.at_node["surface_water__y_velocity"].reshape(nr, nc)
        self._hu = grid.at_node["surface_water__x_momentum"].reshape(nr, nc)
        self._hv = grid.at_node["surface_water__y_momentum"].reshape(nr, nc)

        # ── Manning's n ───────────────────────────────────────────────────
        if "mannings_n_at_node" in grid.at_node:
            # Live view  -  updates automatically if the field changes
            self._n_2d = grid.at_node["mannings_n_at_node"].reshape(nr, nc)
        else:
            n_arr = np.asarray(mannings_n, dtype=float)
            if n_arr.ndim == 0:
                self._n_2d = float(n_arr)  # scalar fast-path
            elif n_arr.size == grid.number_of_nodes:
                self._n_2d = n_arr.reshape(nr, nc).copy()
            else:
                raise ValueError(
                    f"mannings_n must be a scalar or a 1-D array of length "
                    f"{grid.number_of_nodes} (number_of_nodes). "
                    f"Got size {n_arr.size}."
                )

        # ── Wall BCs ──────────────────────────────────────────────────────
        walls = set(wall_edges) if wall_edges else set()
        bad = walls - {"left", "right", "bottom", "top"}
        if bad:
            raise ValueError(
                f"Unknown wall_edges: {bad}. "
                "Choose from {{'left','right','bottom','top'}}."
            )
        self._left_wall = "left" in walls
        self._right_wall = "right" in walls
        self._bottom_wall = "bottom" in walls
        self._top_wall = "top" in walls

        # ── Inflow BCs ────────────────────────────────────────────────────
        # The inflow is enforced as a flux boundary condition (Godunov-correct):
        # the prescribed (h, u, v) values populate the boundary ghost cells,
        # and the HLLC Riemann problem at the boundary face then delivers the
        # exact prescribed (h u, h u^2 + ½ g h^2, h u v) flux into the
        # adjacent interior cell.  This is the canonical way to impose a
        # subcritical inflow in a finite-volume SWE scheme  -  directly setting
        # the interior cell state (as a Dirichlet on the cell) leaves a
        # spurious Riemann problem at the face to the next interior cell
        # whenever the interior deviates from the prescribed state, with
        # mass and momentum lost to that internal wave structure.
        self._inflow_left_spec = None
        self._inflow_right_spec = None
        self._inflow_bottom_spec = None
        self._inflow_top_spec = None
        if fixed_entry_nodes is not None:
            self._entry_nodes = np.asarray(fixed_entry_nodes, dtype=int)
            n = len(self._entry_nodes)
            if entry_nodes_h_values is None:
                raise ValueError(
                    "entry_nodes_h_values is required with fixed_entry_nodes."
                )
            self._entry_h = np.asarray(entry_nodes_h_values, dtype=float)
            self._entry_u = (
                np.zeros(n)
                if entry_nodes_u_values is None
                else np.asarray(entry_nodes_u_values, dtype=float)
            )
            self._entry_v = (
                np.zeros(n)
                if entry_nodes_v_values is None
                else np.asarray(entry_nodes_v_values, dtype=float)
            )
            self._entry_rows = self._entry_nodes // nc
            self._entry_cols = self._entry_nodes % nc

            # Build per-boundary inflow specs by classifying entry nodes
            # into the boundary they sit on.  An entry node belongs to a
            # boundary if its row/col index is 0 (left/bottom) or n-1
            # (right/top).  Bed elevation comes from grid.z at the boundary
            # row/column so the ghost is consistent with the interior bed.
            z_2d = self._z
            for side, mask in (
                ("left", self._entry_cols == 0),
                ("right", self._entry_cols == nc - 1),
                ("bottom", self._entry_rows == 0),
                ("top", self._entry_rows == nr - 1),
            ):
                if not mask.any():
                    continue
                # Build full-edge arrays (one entry per row for left/right,
                # per column for bottom/top), filled with the interior cell
                # bed elevation and zero flow; then overwrite the entries
                # specified by the user.  Non-specified edge cells default to
                # h=interior, hu=0, hv=0 (treated as transmissive only at
                # those rows where the user did not impose an inflow).
                if side in ("left", "right"):
                    edge_len = nr
                    col_idx = 0 if side == "left" else nc - 1
                    q_edge = np.zeros(edge_len)
                    hv_edge = np.zeros(edge_len)
                    z_edge = z_2d[:, col_idx].copy()
                    rows_here = self._entry_rows[mask]
                    h_vals = self._entry_h[mask]
                    u_vals = self._entry_u[mask]
                    v_vals = self._entry_v[mask]
                    q_edge[rows_here] = h_vals * u_vals  # discharge = h * u
                    hv_edge[rows_here] = h_vals * v_vals  # transverse momentum target
                    spec = {
                        "mode": "discharge",
                        "q": q_edge,
                        "hv": hv_edge,
                        "z": z_edge,
                    }
                    if side == "left":
                        self._inflow_left_spec = spec
                    else:
                        self._inflow_right_spec = spec
                else:  # bottom / top
                    edge_len = nc
                    row_idx = 0 if side == "bottom" else nr - 1
                    q_edge = np.zeros(edge_len)
                    hv_edge = np.zeros(edge_len)
                    z_edge = z_2d[row_idx, :].copy()
                    cols_here = self._entry_cols[mask]
                    h_vals = self._entry_h[mask]
                    u_vals = self._entry_u[mask]
                    v_vals = self._entry_v[mask]
                    # For bottom/top inflow, the "discharge" is in the y direction,
                    # which after the y-sweep transpose becomes hu in the swept frame.
                    # We store the spec under the original (x, y) convention; the
                    # _swap_spec function in _y_sweep handles the transpose.
                    # Here q = h * v_n (normal velocity into domain).
                    q_edge[cols_here] = h_vals * v_vals
                    hv_edge[cols_here] = h_vals * u_vals  # tangential = x-momentum
                    spec = {
                        "mode": "discharge",
                        "q": q_edge,
                        "hv": hv_edge,
                        "z": z_edge,
                    }
                    if side == "bottom":
                        self._inflow_bottom_spec = spec
                    else:
                        self._inflow_top_spec = spec
        else:
            self._entry_nodes = None

        # ── Outlet (fixed stage/depth) BCs ─────────────────────────────────
        if fixed_exit_nodes is not None:
            self._exit_nodes = np.asarray(fixed_exit_nodes, dtype=int)
            m = len(self._exit_nodes)

            if (exit_nodes_h_values is None) and (exit_nodes_eta_values is None):
                raise ValueError(
                    "Provide exit_nodes_h_values or exit_nodes_eta_values "
                    "with fixed_exit_nodes."
                )

            self._exit_h = (
                None
                if exit_nodes_h_values is None
                else np.asarray(exit_nodes_h_values, dtype=float)
            )
            self._exit_eta = (
                None
                if exit_nodes_eta_values is None
                else np.asarray(exit_nodes_eta_values, dtype=float)
            )

            self._exit_u = (
                None
                if exit_nodes_u_values is None
                else np.asarray(exit_nodes_u_values, dtype=float)
            )
            self._exit_v = (
                None
                if exit_nodes_v_values is None
                else np.asarray(exit_nodes_v_values, dtype=float)
            )

            self._exit_rows = self._exit_nodes // nc
            self._exit_cols = self._exit_nodes % nc

            # Basic length checks
            if self._exit_h is not None and self._exit_h.size != m:
                raise ValueError(
                    "exit_nodes_h_values must match fixed_exit_nodes length."
                )
            if self._exit_eta is not None and self._exit_eta.size != m:
                raise ValueError(
                    "exit_nodes_eta_values must match fixed_exit_nodes length."
                )
            if self._exit_u is not None and self._exit_u.size != m:
                raise ValueError(
                    "exit_nodes_u_values must match fixed_exit_nodes length."
                )
            if self._exit_v is not None and self._exit_v.size != m:
                raise ValueError(
                    "exit_nodes_v_values must match fixed_exit_nodes length."
                )
        else:
            self._exit_nodes = None

        # Ramp-up outlet: only clamp depth when local h >= this threshold
        self._outlet_max_depth = (
            None if outlet_max_depth is None else float(outlet_max_depth)
        )

        self._update_derived()

        self._use_smagorinsky = use_smagorinsky
        self._smagorinsky_cs = smagorinsky_cs
        self._use_elder = use_elder
        self._elder_alpha = elder_alpha

        # Initialize diagnostic field for Eddy Viscosity
        if self._use_smagorinsky or self._use_elder:
            if "surface_water__eddy_viscosity" not in self._grid.at_node:
                self._grid.add_zeros("surface_water__eddy_viscosity", at="node")
            self._nu_t_flat = self._grid.at_node["surface_water__eddy_viscosity"]

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def elapsed_time(self):
        # Total simulated time [s] since component creation.
        """Return elapsed model time."""
        return self._t

    @property
    def current_dt(self):
        # CFL-based adaptive time step for the next call [s].
        """Return the most recent timestep."""
        return _dt(self._h, self._hu, self._hv, self._dx, self._dy, self._cfl, self._g)

    # ──────────────────────────────────────────────────────────────────────
    # Main stepping method
    # ──────────────────────────────────────────────────────────────────────

    def run_one_step(self, dt=None):
        """Advance the hydraulic state by one time step.

        Parameters
        ----------
        dt : float or None
            Time step [s].  ``None`` → adaptive CFL step (recommended).
            A warning is issued when a user-supplied ``dt`` exceeds the
            CFL-stable limit.

        Notes
        -----
        Inflow Dirichlet boundary conditions are enforced as a Godunov-correct
        flux BC: the prescribed (h, u, v) populates the boundary ghost cells
        inside _x_sweep/_y_sweep, so the HLLC Riemann problem at the boundary
        face delivers exactly the prescribed (h u, h u^2 + ½ g h^2, h u v)
        flux into the adjacent interior cell.  No interior cell is overwritten.
        Outlet BCs remain a stage Dirichlet on the boundary cell (handled by
        _apply_outlet).
        """
        # Outlet stage BC is still applied as a cell-state Dirichlet
        self._apply_outlet()

        if dt is None:
            dt = _dt(
                self._h, self._hu, self._hv, self._dx, self._dy, self._cfl, self._g
            )
        else:
            dt = float(dt)
            dt_cfl = _dt(self._h, self._hu, self._hv, self._dx, self._dy, 1.0, self._g)
            if dt > dt_cfl:
                warnings.warn(
                    f"Supplied dt={dt:.4g} s exceeds CFL-stable "
                    f"dt={dt_cfl:.4g} s.  Results may be unstable.",
                    stacklevel=2,
                )

        h_new, hu_new, hv_new = _step(
            self._h,
            self._hu,
            self._hv,
            self._z,
            self._dx,
            self._dy,
            dt,
            n_2d=self._n_2d,
            g=self._g,
            step_count=self._step_n,
            order=self._order,
            left_wall=self._left_wall,
            right_wall=self._right_wall,
            bottom_wall=self._bottom_wall,
            top_wall=self._top_wall,
            inflow_left=self._inflow_left_spec,
            inflow_right=self._inflow_right_spec,
            inflow_bottom=self._inflow_bottom_spec,
            inflow_top=self._inflow_top_spec,
        )

        self._h[:] = h_new
        self._hu[:] = hu_new
        self._hv[:] = hv_new

        # === 2D IMPLICIT DIFFUSION ===
        if self._use_smagorinsky or self._use_elder:
            self._apply_implicit_diffusion(dt)

        self._apply_outlet()
        self._update_derived()

        if self._update_links:
            self._populate_link_velocity()

        self._step_n += 1
        self._t += dt

    # ──────────────────────────────────────────────────────────────────────
    # Link velocity helpers
    # ──────────────────────────────────────────────────────────────────────

    def map_velocities_to_links(self):
        """Signed velocity projected onto each link direction.

        The value at each link is the arithmetic mean of the two end-node
        velocity components projected in the link direction.  Positive
        values follow the positive link direction (east for horizontal
        links, north for vertical links).

        Returns
        -------
        vel : ndarray
        """
        grid = self._grid
        u_flat = self._u.ravel()
        v_flat = self._v.ravel()
        vel = np.zeros(grid.number_of_links)
        hl = grid.horizontal_links
        vel[hl] = 0.5 * (
            u_flat[grid.node_at_link_head[hl]] + u_flat[grid.node_at_link_tail[hl]]
        )
        vl = grid.vertical_links
        vel[vl] = 0.5 * (
            v_flat[grid.node_at_link_head[vl]] + v_flat[grid.node_at_link_tail[vl]]
        )
        return vel

    def _populate_link_velocity(self):
        """Populate the link velocity field from nodal velocity components."""
        if "surface_water__velocity" not in self._grid.at_link:
            self._grid.add_zeros("surface_water__velocity", at="link")
        self._grid.at_link["surface_water__velocity"][:] = np.abs(
            self.map_velocities_to_links()
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _apply_inflow(self):
        """Apply fixed inflow boundary conditions."""
        if self._entry_nodes is None:
            return
        r, c = self._entry_rows, self._entry_cols
        self._h[r, c] = self._entry_h
        self._hu[r, c] = self._entry_h * self._entry_u
        self._hv[r, c] = self._entry_h * self._entry_v

    def _apply_outlet(self):
        """Apply fixed outlet boundary conditions."""
        if self._exit_nodes is None:
            return
        r, c = self._exit_rows, self._exit_cols

        # Determine imposed depth from eta or h target
        if self._exit_eta is not None:
            h_set = np.maximum(0.0, self._exit_eta - self._z[r, c])
        else:
            h_set = np.maximum(0.0, self._exit_h)

        # Ramp-up: only clamp depth when local h has reached the target.
        # While the outlet is still dry or shallower than the target,
        # keep the local depth (transmissive behaviour) so no water is
        # artificially created or destroyed at a dry/partial outlet.
        if self._outlet_max_depth is not None:
            h_local = self._h[r, c]
            h_set = np.where(h_local >= self._outlet_max_depth, h_set, h_local)

        # Determine outlet momentum to impose.
        # If user supplies exit velocities, convert to momentum. Otherwise keep
        # momentum (zero-gradient) and only adjust depth/stage. This avoids
        # injecting/removing discharge when only stage is prescribed.
        if self._exit_u is None:
            hu_set = self._hu[r, c]
        else:
            hu_set = h_set * self._exit_u

        if self._exit_v is None:
            hv_set = self._hv[r, c]
        else:
            hv_set = h_set * self._exit_v

        self._h[r, c] = h_set
        self._hu[r, c] = hu_set
        self._hv[r, c] = hv_set

    def _update_derived(self):
        """Update derived water-surface and velocity fields."""
        np.add(self._h, self._z, out=self._eta)
        wet = self._h > _H_DRY
        ih = np.where(wet, 1.0 / np.where(wet, self._h, 1.0), 0.0)
        np.multiply(self._hu, ih, out=self._u)
        np.multiply(self._hv, ih, out=self._v)

    def _apply_implicit_diffusion(self, dt):
        """Perform implicit integration of horizontal momentum diffusion."""
        h = self._h
        u = np.where(h > 1e-6, self._hu / h, 0.0)
        v = np.where(h > 1e-6, self._hv / h, 0.0)

        nu_t = np.zeros_like(h)
        dx, dy = self._dx, self._dy

        # ── 1. Smagorinsky Closure (Shear-driven mixing) ───────────
        if self._use_smagorinsky:
            dudx = np.zeros_like(u)
            dudy = np.zeros_like(u)
            dvdx = np.zeros_like(v)
            dvdy = np.zeros_like(v)

            # Central differences (boundaries naturally remain 0)
            dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
            dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
            dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
            dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)

            S_mag = np.sqrt(2 * (dudx**2 + dvdy**2) + (dudy + dvdx) ** 2)
            nu_smag = (self._smagorinsky_cs * np.sqrt(dx * dy)) ** 2 * S_mag
            nu_t += nu_smag

        # ── 2. Elder's Closure (Bed-friction-driven mixing) ────────
        if self._use_elder:
            mag_u = np.sqrt(u**2 + v**2)
            # ``_n_2d`` is either the scalar fast path or the active
            # per-node roughness array (including a live grid field).
            n_field = self._n_2d
            h_safe = np.maximum(h, 1e-6)
            # nu_elder = alpha * u_* * h; where u_* = sqrt(g) * n * |U| / h^(1/6)
            nu_elder = (
                self._elder_alpha
                * np.sqrt(self._g)
                * n_field
                * mag_u
                * (h_safe ** (5.0 / 6.0))
            )
            nu_t += np.where(h > 1e-6, nu_elder, 0.0)

        # Store for diagnostics
        self._nu_t_flat[:] = nu_t.flatten()

        # ── 3. Sparse Matrix Assembly ──────────────────────────────
        # We solve:  h u^{n+1} - dt * ∇ · (h nu_t ∇ u^{n+1}) = h u^n
        Gamma = h * nu_t
        nrows, ncols = h.shape
        N = nrows * ncols

        # Face Diffusivities
        GE = np.zeros_like(Gamma)
        GW = np.zeros_like(Gamma)
        GN = np.zeros_like(Gamma)
        GS = np.zeros_like(Gamma)

        # Internal faces (Edges remain 0 -> Zero-gradient no-flux BCs,
        # which also brilliantly prevents 1D sparse array wrap-around)
        GE[:, :-1] = 0.5 * (Gamma[:, :-1] + Gamma[:, 1:])
        GW[:, 1:] = 0.5 * (Gamma[:, 1:] + Gamma[:, :-1])
        GN[:-1, :] = 0.5 * (Gamma[:-1, :] + Gamma[1:, :])
        GS[1:, :] = 0.5 * (Gamma[1:, :] + Gamma[:-1, :])

        dt_dx2 = dt / (dx**2)
        dt_dy2 = dt / (dy**2)

        aE = GE * dt_dx2
        aW = GW * dt_dx2
        aN = GN * dt_dy2
        aS = GS * dt_dy2

        aP = h + aE + aW + aN + aS

        # Safeguard dry cells: keeps matrix strictly diagonally dominant
        # and prevents singular matrix inversion failures.
        aP = np.where(h < 1e-6, 1.0, aP)

        diagonals = [
            aP.flatten(),
            -aE.flatten()[:-1],
            -aW.flatten()[1:],
            -aN.flatten()[:-ncols],
            -aS.flatten()[ncols:],
        ]
        offsets = [0, 1, -1, ncols, -ncols]

        # CSR format is optimized for arithmetic operations and matrix-vector products
        A = diags(diagonals, offsets, shape=(N, N), format="csr")

        RHS_u = np.where(h < 1e-6, 0.0, h * u).flatten()
        RHS_v = np.where(h < 1e-6, 0.0, h * v).flatten()

        # ── 4. Solve Sparse Linear System ──────────────────────────
        try:
            # We solve for u and v simultaneously by stacking the RHS
            RHS = np.column_stack((RHS_u, RHS_v))
            UV_new = spla.spsolve(A, RHS)

            u_new = UV_new[:, 0].reshape((nrows, ncols))
            v_new = UV_new[:, 1].reshape((nrows, ncols))

            # Reconstruct conservative momenta
            # Write in place so the arrays remain live views of the
            # corresponding Landlab node fields. Rebinding ``self._hu`` or
            # ``self._hv`` would detach the solver state from the grid.
            self._hu[:] = np.where(h > 1e-6, h * u_new, 0.0)
            self._hv[:] = np.where(h > 1e-6, h * v_new, 0.0)

        except RuntimeError:
            # Fallback pass in the highly unlikely event of a solver condition failure
            pass
