"""
Temperature-dependent fluid (water) properties for RiverBedDynamics.

Implements the Heggen (1983) empirical formulae for liquid-water density
and dynamic viscosity as a function of temperature, together with their
analytical partial derivatives with respect to temperature.  These are the
same expressions used in Link et al. (2019) to derive the temperature
sensitivity of the critical Shields stress and bedload transport rate.

Equations
---------
Density (Heggen 1983):

    rho(T) = 1000 * (1 - 1.9549e-5 * |T - 4|^1.68)          [kg m^-3]

    d_rho/dT = -0.0328 * (T - 4) / |T - 4|^0.32             [kg m^-3 °C^-1]

Dynamic viscosity (Heggen 1983):

    mu(T) = (0.20319 + 1.5883 * exp(-T^0.9 / 22)) * 1e-3    [Pa s]

    d_mu/dT = -6.4979e-5 * exp(-T^0.9 / 22) * T^{-0.1}      [Pa s °C^-1]

The viscous sublayer thickness is derived from the local shear velocity:

    delta_v = 11.6 * nu / u*    (nu = mu / rho)              [m]

References
----------
Heggen, R. J. (1983). Thermal dependent physical properties of water.
    *Journal of Hydraulic Engineering*, 109(2), 298-302.
    https://doi.org/10.1061/(ASCE)0733-9429(1983)109:2(298)

Link, O., Clarck, M., García, A., & García, C. (2019). Temperature effects
    on critical Shields stress and bedload transport.  Unpublished manuscript.

.. codeauthor:: RiverDynamics v1 team
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Density
# ──────────────────────────────────────────────────────────────────────────────

def water_density(T: float | np.ndarray) -> float | np.ndarray:
    """Water density as a function of temperature (Heggen 1983).

    Parameters
    ----------
    T : float or ndarray
        Water temperature [°C].  Scalars and arrays are both accepted.

    Returns
    -------
    rho : float or ndarray
        Water density [kg m^-3].

    Examples
    --------
    >>> import numpy as np
    >>> from ._fluid_properties import water_density
    >>> float(np.round(water_density(4.0), 4))
    1000.0
    >>> float(np.round(water_density(20.0), 3))
    998.204
    """
    T = np.asarray(T, dtype=float)
    return 1000.0 * (1.0 - 1.9549e-5 * np.abs(T - 4.0) ** 1.68)


def d_rho_dT(T: float | np.ndarray) -> float | np.ndarray:
    """Partial derivative of water density with respect to temperature (Eq. 6).

    Parameters
    ----------
    T : float or ndarray
        Water temperature [°C].

    Returns
    -------
    drho_dT : float or ndarray
        d rho / dT  [kg m^-3 °C^-1].  Zero exactly at T = 4 °C (density maximum).

    Examples
    --------
    >>> import numpy as np
    >>> from ._fluid_properties import d_rho_dT
    >>> float(np.round(d_rho_dT(4.0), 6))   # density maximum — slope is zero
    0.0
    >>> d_rho_dT(20.0) < 0   # density decreases above 4 °C
    True
    """
    T = np.asarray(T, dtype=float)
    diff = T - 4.0
    # Avoid 0/0 at T = 4; result is 0 there by L'Hôpital / physical argument
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            diff == 0.0,
            0.0,
            -0.0328 * diff / np.abs(diff) ** 0.32,
        )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic viscosity
# ──────────────────────────────────────────────────────────────────────────────

def water_viscosity(T: float | np.ndarray) -> float | np.ndarray:
    """Dynamic viscosity of water as a function of temperature (Heggen 1983).

    Parameters
    ----------
    T : float or ndarray
        Water temperature [°C].  Must be > 0 °C (Eq. 7 has T^{-0.1}).

    Returns
    -------
    mu : float or ndarray
        Dynamic viscosity [Pa s].

    Examples
    --------
    >>> import numpy as np
    >>> from ._fluid_properties import water_viscosity
    >>> float(np.round(water_viscosity(20.0), 6))  # ≈ 1.002e-3 Pa s
    0.001002
    """
    T = np.asarray(T, dtype=float)
    return (0.20319 + 1.5883 * np.exp(-(T ** 0.9) / 22.0)) * 1e-3


def d_mu_dT(T: float | np.ndarray) -> float | np.ndarray:
    """Partial derivative of dynamic viscosity with respect to temperature (Eq. 7).

    Parameters
    ----------
    T : float or ndarray
        Water temperature [°C].  Must be > 0 °C.

    Returns
    -------
    dmu_dT : float or ndarray
        d mu / dT  [Pa s °C^-1].

    Examples
    --------
    >>> import numpy as np
    >>> from ._fluid_properties import d_mu_dT
    >>> d_mu_dT(20.0) < 0   # viscosity decreases with temperature
    True
    """
    T = np.asarray(T, dtype=float)
    return -6.4979e-5 * np.exp(-(T ** 0.9) / 22.0) * T ** (-0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Particle Reynolds number and derived quantities
# ──────────────────────────────────────────────────────────────────────────────

def particle_reynolds(
    u_star: float | np.ndarray,
    D50_m: float | np.ndarray,
    rho: float | np.ndarray,
    mu: float | np.ndarray,
) -> float | np.ndarray:
    """Particle (shear) Reynolds number  Re_s = rho * u* * D50 / mu.

    Parameters
    ----------
    u_star : float or ndarray
        Shear velocity [m s^-1].
    D50_m : float or ndarray
        Median grain diameter [m].
    rho : float or ndarray
        Water density [kg m^-3].
    mu : float or ndarray
        Dynamic viscosity [Pa s].

    Returns
    -------
    Re_s : float or ndarray
        Dimensionless particle Reynolds number [-].
    """
    return rho * u_star * D50_m / mu


def viscous_sublayer_thickness(
    u_star: float | np.ndarray,
    rho: float | np.ndarray,
    mu: float | np.ndarray,
) -> float | np.ndarray:
    """Viscous sublayer thickness  delta_v = 11.6 * nu / u*   [m].

    The constant 11.6 corresponds to the edge of the viscous sublayer in the
    universal law-of-the-wall (y^+ = 11.6).

    Parameters
    ----------
    u_star : float or ndarray
        Shear velocity [m s^-1].
    rho : float or ndarray
        Water density [kg m^-3].
    mu : float or ndarray
        Dynamic viscosity [Pa s].

    Returns
    -------
    delta_v : float or ndarray
        Viscous sublayer thickness [m].  Set to np.inf where u_star = 0.
    """
    nu = mu / rho  # kinematic viscosity [m^2 s^-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_v = np.where(u_star == 0.0, np.inf, 11.6 * nu / u_star)
    return delta_v


# ──────────────────────────────────────────────────────────────────────────────
# Critical Shields stress (Paphitis 2001) — temperature-aware
# ──────────────────────────────────────────────────────────────────────────────

def _B_s(Re_s: np.ndarray) -> np.ndarray:
    """Log-layer intercept B_s (Eq. 3 in Link et al. 2019)."""
    ln_Re = np.log(Re_s)
    return 8.5 + (2.5 * ln_Re - 3.0) * np.exp(-0.127 * ln_Re ** 2)


def paphitis_tau_cr_star(Re_s: np.ndarray) -> np.ndarray:
    """Critical Shields stress from Paphitis (2001) — Eq. 4.

    Valid for  0.01 < Re_s < 1e5.

    Parameters
    ----------
    Re_s : ndarray
        Particle Reynolds number [-].

    Returns
    -------
    tau_cr_star : ndarray
        Dimensionless critical Shields stress [-].
    """
    return 0.188 / (1.0 + Re_s) + 0.0475 * (1.0 - 0.699 * np.exp(-0.015 * Re_s))


def shields_stress(
    tau: np.ndarray,
    rho: np.ndarray,
    rho_s: float,
    g: float,
    D50_m: np.ndarray,
) -> np.ndarray:
    """Dimensionless Shields stress  tau* = tau / ((rho_s - rho) g D50)  (Eq. 1).

    Parameters
    ----------
    tau : ndarray
        Bed shear stress [Pa].
    rho : ndarray
        Water density [kg m^-3].  Can be spatially variable when
        ``variable_fluid_properties=True``.
    rho_s : float
        Sediment density [kg m^-3].
    g : float
        Gravitational acceleration [m s^-2].
    D50_m : ndarray
        Median grain diameter [m].

    Returns
    -------
    tau_star : ndarray
        Dimensionless Shields stress [-].
    """
    return tau / ((rho_s - rho) * g * D50_m)
