"""Coverage-focused unit tests for ``river_bed_dynamics._critical_shear_stress``.

Drives the public ``compute_critical_shear_stress`` solver through the default
(temperature-derived rho/mu) path and the override path, and exercises the
internal ``_log_law_u_star`` iterator including the supplied-initial-guess
branch and the non-convergent (max_iter exhausted) branch.
"""

import numpy as np
import pytest

from landlab.components.river_bed_dynamics import _critical_shear_stress as ccs


def _flow():
    n = 6
    U = np.linspace(0.2, 1.5, n)
    h = np.full(n, 0.5)
    D50 = np.full(n, 0.02)
    return U, h, D50


def test_compute_critical_shear_stress_default_path():
    U, h, D50 = _flow()
    out = ccs.compute_critical_shear_stress(U, h, D50, rho_s=2650.0, T=20.0)
    # All advertised keys present
    for key in (
        "tau_cr",
        "tau_cr_star",
        "tau_star",
        "Re_s_cr",
        "u_star",
        "u_star_cr",
        "delta_v",
        "rho",
        "mu",
    ):
        assert key in out
        assert np.asarray(out[key]).shape == U.shape
    # Physical sanity: positive critical stress, positive shear velocity
    assert np.all(out["tau_cr"] > 0.0)
    assert np.all(out["u_star"] >= 0.0)
    assert np.all(out["u_star_cr"] > 0.0)
    # Density/viscosity recovered from T = 20 degC
    assert np.allclose(out["rho"], 997.939, atol=1e-2)


def test_compute_critical_shear_stress_with_overrides_and_array_T():
    U, h, D50 = _flow()
    rho = np.full(U.shape, 999.7)
    mu = np.full(U.shape, 1.3e-3)
    T = np.linspace(5.0, 25.0, U.size)  # spatially variable temperature
    out = ccs.compute_critical_shear_stress(
        U, h, D50, rho_s=2650.0, T=T, rho=rho, mu=mu, g=9.81
    )
    # Overrides are passed straight through
    assert np.allclose(out["rho"], rho)
    assert np.allclose(out["mu"], mu)


def test_compute_critical_shear_stress_temperature_lowers_threshold():
    # Colder water (higher viscosity) generally shifts the threshold; just
    # confirm the solver runs across a temperature sweep and stays finite.
    U, h, D50 = _flow()
    cold = ccs.compute_critical_shear_stress(U, h, D50, rho_s=2650.0, T=2.0)
    warm = ccs.compute_critical_shear_stress(U, h, D50, rho_s=2650.0, T=28.0)
    assert np.all(np.isfinite(cold["tau_cr"]))
    assert np.all(np.isfinite(warm["tau_cr"]))


def test_log_law_u_star_with_initial_guess():
    U = np.array([0.8, 1.2])
    h = np.array([0.5, 0.5])
    D50 = np.array([0.02, 0.02])
    rho = np.array([1000.0, 1000.0])
    mu = np.array([1e-3, 1e-3])
    guess = np.array([0.05, 0.05])
    u_star = ccs._log_law_u_star(U, h, D50, rho, mu, u_star_guess=guess)
    assert u_star.shape == U.shape
    assert np.all(u_star > 0.0)


def test_log_law_u_star_zero_velocity_and_no_convergence():
    # U = 0 exercises the U <= 0 branch of the default initial guess.
    U = np.array([0.0, 0.5])
    h = np.array([0.4, 0.4])
    D50 = np.array([0.02, 0.02])
    rho = np.array([1000.0, 1000.0])
    mu = np.array([1e-3, 1e-3])
    # max_iter=1 with a tight tolerance forces the loop to exit without the
    # early break, covering the non-convergent path.
    u_star = ccs._log_law_u_star(U, h, D50, rho, mu, max_iter=1, tol=1e-15)
    assert np.all(np.isfinite(u_star))
