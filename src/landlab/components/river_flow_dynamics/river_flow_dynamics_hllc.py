""" Simulate surface fluid flow with an HLLC shallow-water solver.

This component implements a finite-volume Godunov-type approximation of the
depth-averaged 2D shallow-water equations on a RasterModelGrid. Numerical fluxes
are computed with an HLLC Riemann solver, with optional second-order MUSCL
reconstruction and hydrostatic reconstruction for well-balanced treatment of
topography. Time integration uses Strang splitting with optional implicit
Manning friction.

Written by Angel Monsalve.

References
----------
Monsalve et al., (2025). RiverFlowDynamics v1.0: A Landlab component for computing two-dimensional river flow dynamics. Journal of Open Source Software, 10(110), 7823, https://doi.org/10.21105/joss.07823

Toro, E. F. (2001). *Shock-Capturing Methods for Free-Surface Shallow Flows*.

Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R., & Perthame, B. (2004).
A fast and stable well-balanced scheme with hydrostatic reconstruction for
shallow water flows. *SIAM J. Sci. Comput.* https://doi.org/10.1137/S1064827503431090

Capabilities
------------
* HLLC Riemann flux — correct shocks, hydraulic jumps, transcritical flow
* Audusse hydrostatic reconstruction — exact well-balancedness
* Strang operator splitting — second-order isotropy
* Transmissive (zero-gradient) outflow BCs on all edges by default
* Optional per-edge reflective wall BCs  (wall_edges parameter)
* Inflow BCs on any edge: depth + u + v at specified nodes
* Non-uniform Manning's n — scalar or per-node array / grid field
* Implicit Manning friction — no stiffness at shallow depths
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

>>> bool(np.all(grid.at_node["surface_water__depth"] >= 0.0))
True
>>> "surface_water__elevation" in grid.at_node
True
>>> "surface_water__velocity" in grid.at_link
True

Examine the flow depth at the center of the channel after 10 seconds.
Expected values are from RiverBedDynamics

>>> expected = np.array([0.5  , 0.5  , 0.5  , 0.501, 0.502, 0.502])
>>> flow_depth = np.reshape(grid.at_node["surface_water__depth"], (8, 6))[3, :]
>>> bool(np.allclose(np.round(flow_depth, 3), expected, atol=0.02))
True

And the velocity at links along the center of the channel.
Expected values are from RiverBedDynamics

>>> expected = np.array([0.45 , 0.457, 0.455, 0.452, 0.453])
>>> linksAtCenter = grid.links_at_node[np.array(np.arange(24, 30))][:-1, 0]
>>> flow_velocity = grid["link"]["surface_water__velocity"][linksAtCenter]
>>> bool(np.allclose(np.round(flow_velocity, 3), expected, atol=0.02))
True

"""

import warnings
import numpy as np
from landlab import Component, RasterModelGrid

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_G     = 9.80665   # gravitational acceleration [m/s²]
_H_DRY = 1e-4      # depth threshold for wet/dry distinction [m]


# ─────────────────────────────────────────────────────────────────────────────
# HLLC Riemann kernel
# ─────────────────────────────────────────────────────────────────────────────

def _swe_flux_x(h, hu, hv, g):
    inv_h = np.where(h > 0.0, 1.0 / np.where(h > 0.0, h, 1.0), 0.0)
    u = hu * inv_h
    return hu, hu * u + 0.5 * g * h**2, hu * (hv * inv_h)


def _wave_speeds(hL, uL, hR, uR, g):
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    sqL = np.sqrt(np.maximum(hL, 0.0))
    sqR = np.sqrt(np.maximum(hR, 0.0))
    den = sqL + sqR
    safe = den > 0.0
    u_roe = np.where(safe, (sqL * uL + sqR * uR) / np.where(safe, den, 1.0), 0.0)
    c_roe = np.sqrt(g * 0.5 * (np.maximum(hL, 0.0) + np.maximum(hR, 0.0)))
    SL = np.minimum(uL - cL, u_roe - c_roe)
    SR = np.maximum(uR + cR, u_roe + c_roe)
    num  = hR * uR * (uR - SR) - hL * uL * (uL - SL) + 0.5 * g * (hR**2 - hL**2)
    dstar = hR * (uR - SR) - hL * (uL - SL)
    sf   = np.abs(dstar) > 1e-14
    S_star = np.where(sf, num / np.where(sf, dstar, 1.0), 0.5 * (uL + uR))
    return SL, SR, S_star


def _hllc_star_flux(h, hu, hv, Fh, Fhu, Fhv, S, S_star, g):
    inv_h = np.where(h > 0.0, 1.0 / np.where(h > 0.0, h, 1.0), 0.0)
    v     = hv * inv_h
    dss   = S - S_star
    sf    = np.abs(dss) > 1e-14
    c     = h * (S - hu * inv_h) / np.where(sf, dss, np.sign(dss + 1e-30) * 1e-14)
    return (Fh  + S * (c         - h ),
            Fhu + S * (c * S_star - hu),
            Fhv + S * (c * v      - hv))


def _hllc_flux_x(hL, huL, hvL, hR, huR, hvR, g=_G):
    """Vectorized HLLC flux in the x-direction across N faces."""
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
    uL = SL >= 0.0;  uLs = (~uL) & (SS >= 0.0);  uRs = (~uL) & (~uLs) & (SR >= 0.0)
    w = lambda a, b, c, d: np.where(uL, a, np.where(uLs, b, np.where(uRs, c, d)))
    return w(FhL, FLh, FRh, FhR), w(FhuL, FLhu, FRhu, FhuR), w(FhvL, FLhv, FRhv, FhvR)


# ─────────────────────────────────────────────────────────────────────────────
# MUSCL + Audusse reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _vanleer(a, b):
    ab = a * b;  s = a + b
    return np.where(ab > 0.0, 2.0 * ab / np.where(np.abs(s) > 1e-14, s, 1.0), 0.0)


def _muscl_x(q_p):
    dq = q_p[:, 1:] - q_p[:, :-1]
    slp = np.zeros_like(q_p)
    slp[:, 1:-1] = 0.5 * _vanleer(dq[:, :-1], dq[:, 1:])
    return q_p[:, :-1] + slp[:, :-1], q_p[:, 1:] - slp[:, 1:]


def _hydro_recon(etaL, etaR, zL, zR):
    zf = np.maximum(zL, zR)
    return np.maximum(0.0, etaL - zf), np.maximum(0.0, etaR - zf), zf


# ─────────────────────────────────────────────────────────────────────────────
# Directional sweeps with configurable BCs
# ─────────────────────────────────────────────────────────────────────────────

def _pad(q, left_reflect=False, right_reflect=False, negate=False):
    #Pad q with transmissive or reflective ghost cells in x.
    sign = -1.0 if negate else 1.0
    left_ghost  = q[:, :1]  * (sign if left_reflect  else 1.0)
    right_ghost = q[:, -1:] * (sign if right_reflect else 1.0)
    return np.concatenate([left_ghost, q, right_ghost], axis=1)


def _x_sweep(h, hu, hv, z, dt, dx, g=_G, order=1,
             left_wall=False, right_wall=False):
    nr, nc = h.shape
    eta = h + z

    # transmissive by default; reflective (negated u-mom) for walls
    eta_p = _pad(eta, left_wall, right_wall, negate=False)
    z_p   = _pad(z,   left_wall, right_wall, negate=False)
    hu_p  = _pad(hu,  left_wall, right_wall, negate=True)   # negate normal momentum
    hv_p  = _pad(hv,  left_wall, right_wall, negate=False)

    if order == 2:
        etaL, etaR = _muscl_x(eta_p);  zL, zR = _muscl_x(z_p)
        huL, huR   = _muscl_x(hu_p);   hvL, hvR = _muscl_x(hv_p)
    else:
        etaL, etaR = eta_p[:, :-1], eta_p[:, 1:]
        zL,   zR   =   z_p[:, :-1],   z_p[:, 1:]
        huL,  huR  =  hu_p[:, :-1],  hu_p[:, 1:]
        hvL,  hvR  =  hv_p[:, :-1],  hv_p[:, 1:]

    hL_s, hR_s, _ = _hydro_recon(etaL, etaR, zL, zR)
    hLr = np.maximum(0.0, etaL - zL);  hRr = np.maximum(0.0, etaR - zR)
    iL  = np.where(hLr > _H_DRY, 1.0 / np.where(hLr > _H_DRY, hLr, 1.0), 0.0)
    iR  = np.where(hRr > _H_DRY, 1.0 / np.where(hRr > _H_DRY, hRr, 1.0), 0.0)
    uL, vL = huL * iL, hvL * iL
    uR, vR = huR * iR, hvR * iR

    Fh, Fhu, Fhv = _hllc_flux_x(
        hL_s.ravel(), (hL_s * uL).ravel(), (hL_s * vL).ravel(),
        hR_s.ravel(), (hR_s * uR).ravel(), (hR_s * vR).ravel(), g=g)
    Fh  = Fh.reshape(nr, nc + 1)
    Fhu = Fhu.reshape(nr, nc + 1)
    Fhv = Fhv.reshape(nr, nc + 1)

    Sx = 0.5 * g * (hL_s[:, 1:]**2 - hR_s[:, :-1]**2) / dx
    return (h  - dt / dx * (Fh[:,  1:] - Fh[:,  :-1]),
            hu - dt / dx * (Fhu[:, 1:] - Fhu[:, :-1]) + dt * Sx,
            hv - dt / dx * (Fhv[:, 1:] - Fhv[:, :-1]))


def _y_sweep(h, hu, hv, z, dt, dy, g=_G, order=1,
             bottom_wall=False, top_wall=False):
    h_T, hv_T, hu_T = _x_sweep(
        h.T, hv.T, hu.T, z.T, dt, dy, g, order,
        left_wall=bottom_wall, right_wall=top_wall)
    return h_T.T, hu_T.T, hv_T.T


def _friction(h, hu, hv, dt, n_2d, g=_G):
    wet  = h > _H_DRY
    ih   = np.where(wet, 1.0 / np.where(wet, h, 1.0), 0.0)
    spd  = np.sqrt((hu * ih)**2 + (hv * ih)**2)
    Cf   = np.where(wet, g * n_2d**2 * spd / (h**(4.0 / 3.0) + 1e-30), 0.0)
    fac  = 1.0 / (1.0 + Cf * dt)
    return h, hu * fac, hv * fac


def _pos(h, hu, hv):
    h  = np.maximum(h, 0.0)
    hu = np.where(h > _H_DRY, hu, 0.0)
    hv = np.where(h > _H_DRY, hv, 0.0)
    return h, hu, hv


def _dt(h, hu, hv, dx, dy, cfl=0.45, g=_G):
    wet  = h > _H_DRY
    ih   = np.where(wet, 1.0 / np.where(wet, h, 1.0), 0.0)
    c    = np.sqrt(g * np.maximum(h, 0.0))
    mx   = (np.abs(hu * ih) + c).max() if wet.any() else 0.0
    my   = (np.abs(hv * ih) + c).max() if wet.any() else 0.0
    return min(cfl * dx / (mx + 1e-12), cfl * dy / (my + 1e-12))


def _step(h, hu, hv, z, dx, dy, dt, n_2d=0.0, g=_G, step_count=0, order=1,
          left_wall=False, right_wall=False, bottom_wall=False, top_wall=False):
    kx = dict(g=g, order=order, left_wall=left_wall,   right_wall=right_wall)
    ky = dict(g=g, order=order, bottom_wall=bottom_wall, top_wall=top_wall)
    P  = _pos
    if step_count % 2 == 0:
        h, hu, hv = P(*_x_sweep(h, hu, hv, z, dt / 2, dx, **kx))
        h, hu, hv = P(*_y_sweep(h, hu, hv, z, dt,     dy, **ky))
        h, hu, hv = P(*_x_sweep(h, hu, hv, z, dt / 2, dx, **kx))
    else:
        h, hu, hv = P(*_y_sweep(h, hu, hv, z, dt / 2, dy, **ky))
        h, hu, hv = P(*_x_sweep(h, hu, hv, z, dt,     dx, **kx))
        h, hu, hv = P(*_y_sweep(h, hu, hv, z, dt / 2, dy, **ky))
    if np.any(n_2d > 0):
        h, hu, hv = P(*_friction(h, hu, hv, dt, n_2d, g))
    return h, hu, hv


# ─────────────────────────────────────────────────────────────────────────────
# Landlab Component
# ─────────────────────────────────────────────────────────────────────────────

class RiverFlowDynamics_HLLC(Component):
    """2D shallow-water HLLC solver on a RasterModelGrid.

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
        Water depth [m] imposed at each ``fixed_exit_node`` (depth-based outlet).

    exit_nodes_eta_values : array_like, optional
        Water-surface elevation [m] imposed at each ``fixed_exit_node`` (stage-based outlet).
        If provided, depth is set as ``max(eta - z, 0)`` at those nodes.

    exit_nodes_u_values, exit_nodes_v_values : array_like, optional
        Optional velocities [m/s] imposed at outlet nodes. If omitted, the
        solver preserves the current local velocity when applying the outlet
        depth/stage (i.e., momentum is adjusted consistently with the imposed
        depth/stage).
    wall_edges : set of str, optional
        Edges treated as **reflective walls** (zero normal velocity).
        Subset of ``{'left', 'right', 'bottom', 'top'}``.
        Edges not listed use **transmissive** (zero-gradient) outflow BCs.
        Default: empty set (all edges transmissive).

    update_link_fields : bool, optional
        If ``True``, ``surface_water__velocity`` at links is updated every
        call to ``run_one_step()``.  Required for ``RiverTemperatureDynamics``
        coupling.  Default ``False``.

    Notes
    -----
    Outflow (transmissive) BC — the default for every edge.
    Ghost cells are set to the boundary cell value (zero-gradient), which
    allows waves and flow to exit without numerical reflection.  This is the
    standard first-order transmissive (Sommerfeld-like) outflow condition for
    hyperbolic systems and is appropriate for all open boundaries.

    Wall (reflective) BC — enabled via ``wall_edges``.
    The normal momentum component is negated in the ghost cell, producing
    zero normal flux at that face.  Use for physical walls or closed ends.

    Inflow BC — specified via ``fixed_entry_nodes``.
    Depth and velocity are overwritten before and after each step
    (Dirichlet enforcement).  Can be applied on any edge or interior nodes.

    Outlet (fixed stage/depth) BC — specified via ``fixed_exit_nodes``.
    Depth (or stage) is overwritten before and after each step (Dirichlet).
    Use this to constrain downstream water-surface elevation / depth for
    controlled outflow comparisons or steady channel tests.

    Non-uniform roughness — pass a per-node array or pre-populate
    ``"mannings_n_at_node"`` before creating the component.  The roughness
    array is used directly in the implicit friction solve each step; values
    of zero give frictionless cells.

    Link velocity field — set ``update_link_fields=True`` to
    automatically populate ``surface_water__velocity`` (scalar speed at
    links) after each step.  Values are the face-normal component of the
    node-averaged velocity.
    """

    _name = "RiverFlowDynamics_HLLC"
    _unit_agnostic = False

    _info = {
        "topographic__elevation": {
            "dtype": float, "intent": "in", "optional": False,
            "units": "m", "mapping": "node",
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
            "dtype": float, "intent": "out", "optional": False,
            "units": "m/s", "mapping": "node",
            "doc": "Depth-averaged x-velocity",
        },
        "surface_water__y_velocity": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "m/s", "mapping": "node",
            "doc": "Depth-averaged y-velocity",
        },
        "surface_water__x_momentum": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "m2/s", "mapping": "node",
            "doc": "Depth-integrated x-momentum (hu)",
        },
        "surface_water__y_momentum": {
            "dtype": float, "intent": "out", "optional": False,
            "units": "m2/s", "mapping": "node",
            "doc": "Depth-integrated y-momentum (hv)",
        },
        "surface_water__velocity": {
            "dtype": float, "intent": "out", "optional": True,
            "units": "m/s", "mapping": "link",
            "doc": "Speed of water flow above the surface",
        },
        "mannings_n_at_node": {
            "dtype": float, "intent": "in", "optional": True,
            "units": "s/m^(1/3)", "mapping": "node",
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
        wall_edges=None,
        update_link_fields=False,
    ):
        if not isinstance(grid, RasterModelGrid):
            raise TypeError("RiverFlowDynamics_HLLC requires a RasterModelGrid.")
        super().__init__(grid)

        self._g      = float(g)
        self._cfl    = float(cfl)
        self._order  = int(order)
        self._step_n = 0
        self._t      = 0.0
        self._update_links = bool(update_link_fields)

        nr, nc = grid.shape
        self._nr = nr;  self._nc = nc
        self._dx = grid.dx;  self._dy = grid.dy

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
        self._z   = grid.at_node["topographic__elevation"].reshape(nr, nc)
        self._h   = grid.at_node["surface_water__depth"].reshape(nr, nc)
        self._eta = grid.at_node["surface_water__elevation"].reshape(nr, nc)
        self._u   = grid.at_node["surface_water__x_velocity"].reshape(nr, nc)
        self._v   = grid.at_node["surface_water__y_velocity"].reshape(nr, nc)
        self._hu  = grid.at_node["surface_water__x_momentum"].reshape(nr, nc)
        self._hv  = grid.at_node["surface_water__y_momentum"].reshape(nr, nc)

        # ── Manning's n ───────────────────────────────────────────────────
        if "mannings_n_at_node" in grid.at_node:
            # Live view — updates automatically if the field changes
            self._n_2d = grid.at_node["mannings_n_at_node"].reshape(nr, nc)
        else:
            n_arr = np.asarray(mannings_n, dtype=float)
            if n_arr.ndim == 0:
                self._n_2d = float(n_arr)               # scalar fast-path
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
        bad   = walls - {"left", "right", "bottom", "top"}
        if bad:
            raise ValueError(
                f"Unknown wall_edges: {bad}. "
                "Choose from {{'left','right','bottom','top'}}."
            )
        self._left_wall   = "left"   in walls
        self._right_wall  = "right"  in walls
        self._bottom_wall = "bottom" in walls
        self._top_wall    = "top"    in walls

        # ── Inflow BCs ────────────────────────────────────────────────────
        if fixed_entry_nodes is not None:
            self._entry_nodes = np.asarray(fixed_entry_nodes, dtype=int)
            n = len(self._entry_nodes)
            if entry_nodes_h_values is None:
                raise ValueError(
                    "entry_nodes_h_values is required with fixed_entry_nodes."
                )
            self._entry_h = np.asarray(entry_nodes_h_values, dtype=float)
            self._entry_u = (np.zeros(n) if entry_nodes_u_values is None
                             else np.asarray(entry_nodes_u_values, dtype=float))
            self._entry_v = (np.zeros(n) if entry_nodes_v_values is None
                             else np.asarray(entry_nodes_v_values, dtype=float))
            self._entry_rows = self._entry_nodes // nc
            self._entry_cols = self._entry_nodes  % nc
        else:
            self._entry_nodes = None

        # ── Outlet (fixed stage/depth) BCs ─────────────────────────────────
        if fixed_exit_nodes is not None:
            self._exit_nodes = np.asarray(fixed_exit_nodes, dtype=int)
            m = len(self._exit_nodes)

            if (exit_nodes_h_values is None) and (exit_nodes_eta_values is None):
                raise ValueError(
                    "Provide exit_nodes_h_values or exit_nodes_eta_values with fixed_exit_nodes."
                )

            self._exit_h = (None if exit_nodes_h_values is None
                            else np.asarray(exit_nodes_h_values, dtype=float))
            self._exit_eta = (None if exit_nodes_eta_values is None
                              else np.asarray(exit_nodes_eta_values, dtype=float))

            self._exit_u = (None if exit_nodes_u_values is None
                            else np.asarray(exit_nodes_u_values, dtype=float))
            self._exit_v = (None if exit_nodes_v_values is None
                            else np.asarray(exit_nodes_v_values, dtype=float))

            self._exit_rows = self._exit_nodes // nc
            self._exit_cols = self._exit_nodes  % nc

            # Basic length checks
            if self._exit_h is not None and self._exit_h.size != m:
                raise ValueError("exit_nodes_h_values must match fixed_exit_nodes length.")
            if self._exit_eta is not None and self._exit_eta.size != m:
                raise ValueError("exit_nodes_eta_values must match fixed_exit_nodes length.")
            if self._exit_u is not None and self._exit_u.size != m:
                raise ValueError("exit_nodes_u_values must match fixed_exit_nodes length.")
            if self._exit_v is not None and self._exit_v.size != m:
                raise ValueError("exit_nodes_v_values must match fixed_exit_nodes length.")
        else:
            self._exit_nodes = None

        self._update_derived()

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def elapsed_time(self):
        #Total simulated time [s] since component creation.
        return self._t

    @property
    def current_dt(self):
        #CFL-based adaptive time step for the next call [s].
        return _dt(self._h, self._hu, self._hv,
                   self._dx, self._dy, self._cfl, self._g)

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
        """
        self._apply_inflow()
        self._apply_outlet()

        if dt is None:
            dt = _dt(self._h, self._hu, self._hv,
                     self._dx, self._dy, self._cfl, self._g)
        else:
            dt = float(dt)
            dt_cfl = _dt(self._h, self._hu, self._hv,
                         self._dx, self._dy, 1.0, self._g)
            if dt > dt_cfl:
                warnings.warn(
                    f"Supplied dt={dt:.4g} s exceeds CFL-stable "
                    f"dt={dt_cfl:.4g} s.  Results may be unstable.",
                    stacklevel=2,
                )

        h_new, hu_new, hv_new = _step(
            self._h, self._hu, self._hv, self._z,
            self._dx, self._dy, dt,
            n_2d=self._n_2d, g=self._g,
            step_count=self._step_n, order=self._order,
            left_wall=self._left_wall,    right_wall=self._right_wall,
            bottom_wall=self._bottom_wall, top_wall=self._top_wall,
        )

        self._h[:]  = h_new
        self._hu[:] = hu_new
        self._hv[:] = hv_new
        self._apply_inflow()
        self._apply_outlet()
        self._update_derived()

        if self._update_links:
            self._populate_link_velocity()

        self._step_n += 1
        self._t      += dt

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
        vel : ndarray (n_links,)
        """
        grid   = self._grid
        u_flat = self._u.ravel()
        v_flat = self._v.ravel()
        vel    = np.zeros(grid.number_of_links)
        hl = grid.horizontal_links
        vel[hl] = 0.5 * (u_flat[grid.node_at_link_head[hl]] +
                         u_flat[grid.node_at_link_tail[hl]])
        vl = grid.vertical_links
        vel[vl] = 0.5 * (v_flat[grid.node_at_link_head[vl]] +
                         v_flat[grid.node_at_link_tail[vl]])
        return vel

    def _populate_link_velocity(self):
        if "surface_water__velocity" not in self._grid.at_link:
            self._grid.add_zeros("surface_water__velocity", at="link")
        self._grid.at_link["surface_water__velocity"][:] = \
            np.abs(self.map_velocities_to_links())

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _apply_inflow(self):
        if self._entry_nodes is None:
            return
        r, c = self._entry_rows, self._entry_cols
        self._h[r, c]  = self._entry_h
        self._hu[r, c] = self._entry_h * self._entry_u
        self._hv[r, c] = self._entry_h * self._entry_v

    def _apply_outlet(self):
        if self._exit_nodes is None:
            return
        r, c = self._exit_rows, self._exit_cols

        # Determine imposed depth
        if self._exit_eta is not None:
            h_set = np.maximum(0.0, self._exit_eta - self._z[r, c])
        else:
            h_set = np.maximum(0.0, self._exit_h)

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
        np.add(self._h, self._z, out=self._eta)
        wet  = self._h > _H_DRY
        ih   = np.where(wet, 1.0 / np.where(wet, self._h, 1.0), 0.0)
        np.multiply(self._hu, ih, out=self._u)
        np.multiply(self._hv, ih, out=self._v)
