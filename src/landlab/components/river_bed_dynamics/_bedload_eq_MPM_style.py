"""
Implements several bed load transport equations to calculate bed load rates at
links. All equations are of the style qb = a * (tau* - tau*c) **b

Temperature effects
-------------------
When ``rbd._variable_fluid_properties is True``, three temperature-driven
corrections are applied simultaneously:

1. **Shields stress** — tau* = tau / ((rho_s - rho) g D50)
   ``rbd._rho`` is a per-link array (Heggen 1983), so the denominator
   changes spatially with water temperature. Because ``rho`` and ``R`` are
   read from ``rbd`` at call-time, this correction is already captured by the
   existing ``tau_star`` line — no formula change required.

2. **Critical Shields stress** — tau*_cr from Paphitis (2001) via Re_s
   Replaces the fixed constants (0.047 MPM, 0.045 FLvB …) and the existing
   slope-based ``variable_critical_shear_stress`` path with the full
   temperature-aware iterative solve from ``_critical_shear_stress.py``.

3. **Viscous sublayer** — delta_v = 11.6 nu / u*
   Handled inside ``compute_critical_shear_stress`` via the temperature-aware
   log-law (Eq. 2-3 of Link et al. 2019). The result is stored on ``rbd``
   as ``rbd._delta_v_link`` for inspection but does not enter the qb formula
   directly.

When ``variable_fluid_properties=False`` (default) the file is byte-for-byte
identical to v1: fixed tau*_cr constants, optional slope correction.

.. codeauthor:: Angel Monsalve
.. codecoauthors: Sam Anderson, Nicole Gasparini, Elowyn Yager

Examples
--------
This is the same base example described extensively in river bed dynamics, so
we removed comments that are already available in the main component

>>> import numpy as np
>>> from landlab import RasterModelGrid, imshow_grid
>>> from landlab.components import RiverBedDynamics
>>> import copy

>>> grid = RasterModelGrid((5, 5))
>>> grid.at_node["topographic__elevation"] = [
...     [1.07, 1.06, 1.00, 1.06, 1.07],
...     [1.08, 1.07, 1.03, 1.07, 1.08],
...     [1.09, 1.08, 1.07, 1.08, 1.09],
...     [1.09, 1.09, 1.08, 1.09, 1.09],
...     [1.09, 1.09, 1.09, 1.09, 1.09],
... ]
>>> z = copy.deepcopy(grid.at_node["topographic__elevation"])

>>> grid.set_watershed_boundary_condition(grid.at_node["topographic__elevation"])
>>> grid.at_node["surface_water__depth"] = np.full(grid.number_of_nodes, 0.102)
>>> grid.at_node["surface_water__velocity"] = np.full(grid.number_of_nodes, 0.25)
>>> grid.at_link["surface_water__depth"] = np.full(grid.number_of_links, 0.102)
>>> grid.at_link["surface_water__velocity"] = np.full(grid.number_of_links, 0.25)

>>> gsd_loc = [
...     [0, 1.0, 1.0, 1.0, 0],
...     [0, 1.0, 1.0, 1.0, 0],
...     [0, 1.0, 1.0, 1.0, 0],
...     [0, 1.0, 1.0, 1.0, 0],
...     [0, 1.0, 1.0, 1.0, 0],
... ]

>>> gsd = [[32, 100, 100], [16, 25, 50], [8, 0, 0]]

Case 1, we calculate using the default equation, MPM

>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.002297, -0.006891,  0.002297,  0.      ],
       [ 0.      ,  0.      ,  0.002297,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Case 1b, MPM with variable_critical_shear_stress=True

>>> grid.at_node["topographic__elevation"] = z.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bed_surf__gsd_loc_node=gsd_loc,
...     variable_critical_shear_stress=True,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.00065 , -0.001949,  0.00065 ,  0.      ],
       [ 0.      ,  0.      ,  0.00065 ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Case 1c, MPM with a bed load rate fixed in one link

>>> grid.at_node["topographic__elevation"] = z.copy()
>>> qb_fix_link = np.zeros([grid.number_of_links, 1])
>>> qb_fix_link[[29]] = 0.01
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bed_surf__gsd_loc_node=gsd_loc,
...     variable_critical_shear_stress=True,
...     sed_transp__bedload_rate_fix_link=qb_fix_link,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.00065 , -0.001949,  0.00065 ,  0.      ],
       [ 0.      ,  0.      ,  0.00065 ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.01    , -0.01    ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Case 2, we use Fernandez Luque and Van Beek

>>> grid.at_node["topographic__elevation"] = z.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="FLvB",
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.001682, -0.005047,  0.001682,  0.      ],
       [ 0.      ,  0.      ,  0.001682,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Case 2b, FLvB with variable_critical_shear_stress=True

>>> grid.at_node["topographic__elevation"] = z.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="FLvB",
...     variable_critical_shear_stress=True,
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.000463, -0.001389,  0.000463,  0.      ],
       [ 0.      ,  0.      ,  0.000463,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Case 3, we use Wong and Parker

>>> grid.at_node["topographic__elevation"] = z.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="WongAndParker",
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.001133, -0.003398,  0.001133,  0.      ],
       [ 0.      ,  0.      ,  0.001133,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Case 3b, WongAndParker with variable_critical_shear_stress=True

>>> grid.at_node["topographic__elevation"] = z.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="WongAndParker",
...     variable_critical_shear_stress=True,
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.000295, -0.000884,  0.000295,  0.      ],
       [ 0.      ,  0.      ,  0.000295,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Case 4, we use Huang

>>> grid.at_node["topographic__elevation"] = z.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="Huang",
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.001188, -0.003564,  0.001188,  0.      ],
       [ 0.      ,  0.      ,  0.001188,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Case 4b, Huang with variable_critical_shear_stress=True

>>> grid.at_node["topographic__elevation"] = z.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="Huang",
...     variable_critical_shear_stress=True,
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.000292, -0.000876,  0.000292,  0.      ],
       [ 0.      ,  0.      ,  0.000292,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])
"""

import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Module-level helper: temperature-aware critical Shields stress
# ---------------------------------------------------------------------------


def _get_tau_star_cr(rbd, tau_star, tau_star_cr_fixed, tau_star_cr_0):
    """Return the per-link critical Shields stress array.

    Three mutually exclusive paths, selected by flags on the parent component:

    1. ``variable_fluid_properties=True`` — Paphitis (2001) solve via Re_s,
       fully temperature-aware (Shields stress + critical tau*_cr + viscous
       sublayer). Overrides ``variable_critical_shear_stress``.

    2. ``variable_critical_shear_stress=True`` (legacy slope-based) — Mueller
       et al. (2005) slope correction, no temperature correction.

    3. Neither flag — fixed constant (0.047 MPM, 0.045 FLvB …).

    Parameters
    ----------
    rbd : RiverBedDynamics
        Parent component instance (provides all state).
    tau_star : ndarray
        Current dimensionless Shields stress at every link [-].
    tau_star_cr_fixed : float
        Equation-specific default critical Shields stress (e.g. 0.047 MPM).
    tau_star_cr_0 : ndarray
        Slope-based critical Shields stress from Mueller et al. (2005) [–].
        Only used when ``variable_critical_shear_stress=True``.

    Returns
    -------
    tau_star_cr : ndarray, shape (n_links,)
        Critical dimensionless Shields stress at every link.
    """
    # ── Path 1: temperature-aware Paphitis solve ──────────────────────────
    if getattr(rbd, "_variable_fluid_properties", False):
        from ._critical_shear_stress import compute_critical_shear_stress

        result = compute_critical_shear_stress(
            U=rbd._u,
            h=rbd._grid.at_link["surface_water__depth"],
            D50_m=rbd._bed_surf__median_size_link / 1000.0,
            rho_s=rbd._rho_s,  # fix: was rbd._rhos (AttributeError); correct attr is _rho_s
            rho=np.broadcast_to(np.asarray(rbd._rho), rbd._u.shape).copy(),
            mu=np.broadcast_to(np.asarray(rbd._mu), rbd._u.shape).copy(),
            g=rbd._g,
        )
        # Store diagnostics for external inspection / output fields
        rbd._tau_star_cr_link = result["tau_cr_star"]
        rbd._delta_v_link = result["delta_v"]
        rbd._Re_s_cr_link = result["Re_s_cr"]
        rbd._u_star_cr_link = result["u_star_cr"]
        return result["tau_cr_star"]

    # ── Path 2: legacy slope-based correction (variable_critical_shear_stress)
    tau_star_cr = np.full(tau_star.shape, tau_star_cr_fixed)
    if rbd._variable_critical_shear_stress:
        tau_star_cr = np.where(tau_star_cr_0 < 0.021, tau_star_cr, tau_star_cr_0)

    return tau_star_cr


# ---------------------------------------------------------------------------
# Shared computational kernel — used by all four MPM-style wrappers
# ---------------------------------------------------------------------------


def _compute_qb(rbd, equation_fn):
    """Shared kernel for all MPM-style equations.

    Computes the dimensionless Shields stress, delegates the qb* calculation
    to *equation_fn*, then converts back to dimensional transport rate [m²/s].

    This function does **not** read ``rbd._bedload_equation``. The equation
    identity is encoded entirely in the *equation_fn* argument, which is
    supplied by each concrete ``BedloadEquation.calculate()`` wrapper in
    ``bedload_equation_base.py``. This eliminates the dual-location equation
    identity that existed in the previous dispatcher design.

    Parameters
    ----------
    rbd : RiverBedDynamics
        The component instance providing all hydraulic and bed state.
    equation_fn : callable
        One of :func:`MeyerPeter_Muller`, :func:`FernandezLuque_VanBeek`,
        :func:`Wong_Parker`, or :func:`Huang`.
        Signature: ``(rbd, qb_star, tau_star, var_cr, tau_star_cr_0) -> qb_star``.

    Returns
    -------
    qb : ndarray, shape (n_links,)
        Volumetric bedload transport rate per unit width [m²/s] at each link.
        Signed: positive in the positive link direction.
    """
    shear_stress        = rbd._surface_water__shear_stress_link
    shear_stress_signed = rbd._shear_stress
    rho   = rbd._rho          # scalar OR per-link ndarray
    R     = rbd._R            # (rho_s - rho) / rho — scalar OR array
    g     = rbd._g
    gs_D50 = rbd._bed_surf__median_size_link
    dz_ds  = rbd._dz_ds
    var_cr_shear_stress      = rbd._variable_critical_shear_stress
    bedload_rate_fix_link    = rbd._sed_transp__bedload_rate_fix_link
    bedload_rate_fix_link_id = rbd._sed_transp__bedload_rate_fix_link_id

    # ── Dimensionless Shields stress ───────────────────────────────────────
    # rho * R = (rho_s - rho), so tau* = tau / ((rho_s-rho) * g * D50)
    # When rho is a per-link array (temperature path) this is already
    # spatially variable; no formula change is needed.
    tau_star = shear_stress / (rho * R * g * (gs_D50 / 1000))

    # ── Slope-based critical Shields stress (Mueller et al. 2005) ─────────
    # Computed once and passed to the sub-function via _get_tau_star_cr.
    # Only active when variable_fluid_properties=False and
    # variable_critical_shear_stress=True.
    bed_slope     = dz_ds * np.sign(shear_stress_signed)
    tau_star_cr_0 = np.where(bed_slope > 0.03, 2.18 * bed_slope + 0.021, 0)

    qb_star = np.zeros_like(tau_star)
    qb_star = equation_fn(rbd, qb_star, tau_star, var_cr_shear_stress, tau_star_cr_0)
    qb_star = np.where(np.isnan(qb_star), 0, qb_star)

    qb = (
        qb_star * (np.sqrt(R * g * (gs_D50 / 1000)) * (gs_D50 / 1000))
    ) * np.sign(shear_stress_signed)

    # ── Restore fixed bed load rates at fixed links ────────────────────────
    if bedload_rate_fix_link_id.size > 0:
        qb[bedload_rate_fix_link_id] = bedload_rate_fix_link[bedload_rate_fix_link_id]

    return qb


# ---------------------------------------------------------------------------
# Legacy dispatcher — retained for backward compatibility only
# ---------------------------------------------------------------------------


def bedload_equation(rbd):
    """Deprecated entry point kept for backward compatibility.

    New code should use ``EQUATION_REGISTRY`` in ``bedload_equation_base.py``
    and call ``BedloadEquation.calculate(rbd)`` instead.  That path calls
    :func:`_compute_qb` directly with the correct sub-function, removing the
    need to re-read ``rbd._bedload_equation`` at dispatch time.

    .. deprecated::
        Pass the equation key to :class:`RiverBedDynamics` via
        ``bedload_equation=`` and let the registry handle dispatch.
    """
    warnings.warn(
        "bedload_equation() is deprecated. Instantiate RiverBedDynamics with "
        "the desired bedload_equation= key and let EQUATION_REGISTRY dispatch "
        "via BedloadEquation.calculate() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _dispatch = {
        "MPM":           MeyerPeter_Muller,
        "FLvB":          FernandezLuque_VanBeek,
        "WongAndParker": Wong_Parker,
        "Huang":         Huang,
    }
    fn = _dispatch[rbd._bedload_equation]
    return _compute_qb(rbd, fn)


# ---------------------------------------------------------------------------
# Individual transport equations
# Each function is a pure computation: it receives all required state as
# arguments and returns the updated qb_star array.  None of these functions
# read rbd._bedload_equation — equation identity is determined by which
# function is called, not by a string lookup.
# ---------------------------------------------------------------------------


def MeyerPeter_Muller(rbd, qb_star, tau_star, var_cr_shear_stress, tau_star_cr_0):
    """Surface-based bedload transport equation of Meyer-Peter and Müller.

    Meyer-Peter, E. and Müller, R., 1948, Formulas for Bed-Load Transport,
    Proceedings, 2nd Congress, International Association of Hydraulic
    Research, Stockholm: 39-64.
    """
    tau_star_cr = _get_tau_star_cr(rbd, tau_star, 0.047, tau_star_cr_0)
    qb_star_coeff = 8
    qb_star_exp   = 3 / 2

    mask = tau_star > tau_star_cr
    qb_star[mask] = qb_star_coeff * (tau_star[mask] - tau_star_cr[mask]) ** qb_star_exp
    return qb_star


def FernandezLuque_VanBeek(rbd, qb_star, tau_star, var_cr_shear_stress, tau_star_cr_0):
    """Surface-based bedload transport equation of Fernandez Luque and van Beek.

    Fernandez Luque, R. and R. van Beek, 1976, Erosion and transport of
    bedload sediment, Journal of Hydraulic Research, 14(2): 127-144.
    """
    tau_star_cr = _get_tau_star_cr(rbd, tau_star, 0.045, tau_star_cr_0)
    qb_star_coeff = 5.7
    qb_star_exp   = 3 / 2

    mask = tau_star > tau_star_cr
    qb_star[mask] = qb_star_coeff * (tau_star[mask] - tau_star_cr[mask]) ** qb_star_exp
    return qb_star


def Wong_Parker(rbd, qb_star, tau_star, var_cr_shear_stress, tau_star_cr_0):
    """Surface-based bedload transport equation of Wong and Parker.

    Wong and Parker 2006, Reanalysis and Correction of Bed-Load Relation of
    Meyer-Peter and Müller Using Their Own Database. Journal of Hydraulic
    Engineering. Volume 132, Issue 11.
    """
    tau_star_cr = _get_tau_star_cr(rbd, tau_star, 0.047, tau_star_cr_0)
    qb_star_coeff = 4.93
    qb_star_exp   = 1.6

    mask = tau_star > tau_star_cr
    qb_star[mask] = qb_star_coeff * (tau_star[mask] - tau_star_cr[mask]) ** qb_star_exp
    return qb_star


def Huang(rbd, qb_star, tau_star, var_cr_shear_stress, tau_star_cr_0):
    """Surface-based bedload transport equation of He Qing Huang.

    Huang, H. Q. (2010), Reformulation of the bed load equation of Meyer-Peter
    and Müller in light of the linearity theory for alluvial channel flow,
    Water Resour. Res., 46, W09533, doi:10.1029/2009WR008974.
    """
    tau_star_cr = _get_tau_star_cr(rbd, tau_star, 0.047, tau_star_cr_0)
    qb_star_coeff = 6.0
    qb_star_exp   = 5 / 3

    mask = tau_star > tau_star_cr
    qb_star[mask] = qb_star_coeff * (tau_star[mask] - tau_star_cr[mask]) ** qb_star_exp
    return qb_star
