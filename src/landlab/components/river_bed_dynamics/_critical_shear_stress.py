"""
Temperature-aware critical Shields stress calculator for RiverBedDynamics.

This module replaces the fixed ``tau_star_cr = 0.047`` (MPM constant) and the
existing ``variable_critical_shear_stress`` slope-based logic with a physically
consistent iterative solver that couples three effects simultaneously:

1. **Shields stress** — tau* = tau / ((rho_s - rho) g D50)
   Both rho and therefore the denominator change with temperature.

2. **Critical Shields stress via Paphitis (2001)** — tau*_cr = f(Re_s)
   The particle Reynolds number Re_s = rho u* D50 / mu depends on both
   rho and mu, so the whole Shields curve shifts with temperature.

3. **Viscous sublayer** — delta_v = 11.6 nu / u*
   A grain is hydraulically smooth when D50 < delta_v, transitional when
   delta_v < D50 < 70 nu/u*, and fully rough when D50 > 70 nu/u*.
   This modulates B_s and hence the log-law reconstruction of tau from U.

Design
------
The critical condition is implicit:  tau = tau_cr requires u* = u*_cr, which
enters Re_s, which determines tau*_cr, which gives tau_cr = tau*_cr (rho_s-rho)
g D50.  The solver iterates (3-5 Newton steps converge to 1e-10 relative
tolerance for all natural stream conditions).

API
---
The public entry point is :func:`compute_critical_shear_stress`.  It accepts
per-link arrays and returns tau_cr [Pa] at every link, fully compatible with
the existing ``bedload_equation`` logic (just substitute tau_star_cr below).

References
----------
Paphitis, D. (2001). Sediment movement under unidirectional flows: an
    assessment of empirical threshold curves. *Coastal Engineering*, 43(3-4),
    227-245. https://doi.org/10.1016/S0378-3839(01)00015-1
Link et al. (2019). Temperature effects on critical Shields stress.
Heggen, R. J. (1983). Journal of Hydraulic Engineering, 109(2), 298-302.

.. codeauthor:: RiverDynamics v1 team
"""

from __future__ import annotations

import numpy as np

from ._fluid_properties import (
    paphitis_tau_cr_star,
    particle_reynolds,
    shields_stress,
    viscous_sublayer_thickness,
    water_density,
    water_viscosity,
)

# von Kármán constant
_KAPPA = 0.41
# y^+ at the viscous-sublayer edge
_DELTA_V_CONST = 11.6
# y^+ at the smooth-to-rough transition
_ROUGH_CONST = 70.0


def _log_law_u_star(
    U: np.ndarray,
    h: np.ndarray,
    D50_m: np.ndarray,
    rho: np.ndarray,
    mu: np.ndarray,
    u_star_guess: np.ndarray | None = None,
    max_iter: int = 20,
    tol: float = 1e-9,
) -> np.ndarray:
    """Compute shear velocity u* from depth-averaged velocity U via the law of
    the wall (Eq. 2), accounting for the roughness-regime-dependent B_s (Eq. 3).

    The log-law is:

        U / u* = (1/kappa) ln(0.368 h / k_s) + B_s(Re_s)

    with k_s = D50 and B_s from Eq. 3.  Because B_s depends on Re_s = rho u*
    D50 / mu, this is solved iteratively.

    Parameters
    ----------
    U : ndarray
        Depth-averaged velocity [m s^-1].
    h : ndarray
        Flow depth [m].
    D50_m : ndarray
        Median grain diameter [m].
    rho : ndarray
        Water density [kg m^-3].
    mu : ndarray
        Dynamic viscosity [Pa s].
    u_star_guess : ndarray or None
        Initial guess for u* [m s^-1].  Defaults to U / 10.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Relative convergence tolerance.

    Returns
    -------
    u_star : ndarray
        Shear velocity [m s^-1].
    """
    if u_star_guess is None:
        u_star = np.where(U > 0, U / 10.0, 1e-6)
    else:
        u_star = np.maximum(u_star_guess, 1e-10)

    for _ in range(max_iter):
        Re_s = particle_reynolds(u_star, D50_m, rho, mu)
        ln_Re = np.log(np.maximum(Re_s, 1e-10))
        B_s = 8.5 + (2.5 * ln_Re - 3.0) * np.exp(-0.127 * ln_Re ** 2)
        log_term = np.log(np.maximum(0.368 * h / D50_m, 1.0))
        rhs = (log_term / _KAPPA + B_s)  # U/u* from log-law
        u_star_new = np.where(rhs > 0, U / rhs, u_star)
        err = np.max(np.abs(u_star_new - u_star) / np.maximum(u_star, 1e-10))
        u_star = u_star_new
        if err < tol:
            break

    return u_star


def compute_critical_shear_stress(
    U: np.ndarray,
    h: np.ndarray,
    D50_m: np.ndarray,
    rho_s: float,
    T: float | np.ndarray = 20.0,
    rho: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    g: float = 9.80665,
    max_iter: int = 20,
    tol: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Compute temperature-aware critical shear stress at every link.

    This is the primary public API of the module.  It computes tau_cr [Pa]
    by solving the implicit system:

        Re_s_cr  = rho * u*_cr * D50 / mu
        tau*_cr  = Paphitis(Re_s_cr)                          [Eq. 4]
        tau_cr   = tau*_cr * (rho_s - rho) * g * D50         [Eq. 1]
        u*_cr    = sqrt(tau_cr / rho)

    simultaneously, as well as the dimensionless Shields stress for the
    current flow:

        tau*     = tau / ((rho_s - rho) g D50)                [Eq. 1]

    where tau is recovered from U via the temperature-aware log-law (Eq. 2-3).

    Parameters
    ----------
    U : ndarray, shape (n_links,)
        Depth-averaged velocity magnitude at links [m s^-1].
    h : ndarray, shape (n_links,)
        Water depth at links [m].
    D50_m : ndarray, shape (n_links,)
        Median surface grain diameter at links [m].
    rho_s : float
        Sediment density [kg m^-3].
    T : float or ndarray, shape (n_links,) or scalar
        Water temperature [°C].  Scalar applies uniformly; an array allows
        spatially variable temperature (e.g. from RiverTemperatureDynamics).
        Default is 20 °C (standard reference temperature).
    rho : ndarray or None
        Override for water density [kg m^-3].  If None (default), computed
        from ``T`` via :func:`water_density`.
    mu : ndarray or None
        Override for dynamic viscosity [Pa s].  If None (default), computed
        from ``T`` via :func:`water_viscosity`.
    g : float
        Gravitational acceleration [m s^-2].
    max_iter : int
        Maximum Newton iterations for the implicit u*_cr solve.
    tol : float
        Relative convergence tolerance.

    Returns
    -------
    result : dict with keys
        ``tau_cr``        — critical shear stress [Pa], shape (n_links,)
        ``tau_cr_star``   — dimensionless critical Shields stress [-]
        ``tau_star``      — current dimensionless Shields stress [-]
        ``Re_s_cr``       — critical particle Reynolds number [-]
        ``u_star``        — shear velocity from current flow [m s^-1]
        ``u_star_cr``     — critical shear velocity [m s^-1]
        ``delta_v``       — viscous sublayer thickness [m]
        ``rho``           — water density used [kg m^-3]
        ``mu``            — dynamic viscosity used [Pa s]

    Notes
    -----
    When ``T`` is a Landlab field (e.g. ``grid.at_link["surface_water__temperature"]``
    mapped to links), this function naturally inherits spatially variable
    fluid properties from RiverTemperatureDynamics.
    """
    T = np.broadcast_to(np.asarray(T, dtype=float), U.shape)

    if rho is None:
        rho = water_density(T)
    if mu is None:
        mu = water_viscosity(T)

    # ── Step 1: u* from current flow via temperature-aware log-law ───────────
    u_star = _log_law_u_star(U, h, D50_m, rho, mu)
    tau = rho * u_star ** 2

    # ── Step 2: tau* for current flow ─────────────────────────────────────────
    submerged_weight = np.maximum((rho_s - rho) * g * D50_m, 1e-30)
    tau_star = tau / submerged_weight

    # ── Step 3: Implicit solve for tau*_cr and tau_cr ─────────────────────────
    # Initial guess: use standard MPM value
    u_star_cr = np.full_like(U, np.sqrt(0.047 * submerged_weight / rho))

    for _ in range(max_iter):
        Re_s_cr = particle_reynolds(u_star_cr, D50_m, rho, mu)
        tau_cr_star = paphitis_tau_cr_star(Re_s_cr)
        tau_cr = tau_cr_star * submerged_weight
        u_star_cr_new = np.sqrt(np.maximum(tau_cr / rho, 0.0))
        err = np.max(
            np.abs(u_star_cr_new - u_star_cr) / np.maximum(u_star_cr, 1e-12)
        )
        u_star_cr = u_star_cr_new
        if err < tol:
            break

    # ── Step 4: Viscous sublayer thickness ────────────────────────────────────
    delta_v = viscous_sublayer_thickness(u_star, rho, mu)

    return {
        "tau_cr": tau_cr,
        "tau_cr_star": tau_cr_star,
        "tau_star": tau_star,
        "Re_s_cr": Re_s_cr,
        "u_star": u_star,
        "u_star_cr": u_star_cr,
        "delta_v": delta_v,
        "rho": rho,
        "mu": mu,
    }
