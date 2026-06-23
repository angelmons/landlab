"""Coverage-focused unit tests for ``river_bed_dynamics._fluid_properties``.

These exercise every public helper (density, viscosity, their temperature
derivatives, particle Reynolds number, viscous-sublayer thickness, the
Paphitis critical-Shields curve and its log-layer intercept, and the
dimensionless Shields stress) with both scalar and array inputs, plus the
edge cases (T = 4 degC density maximum, u_star = 0).
"""

import numpy as np
import pytest

from landlab.components.river_bed_dynamics import _fluid_properties as fp


def test_water_density_scalar_and_array():
    # Density maximum at 4 degC
    assert np.isclose(float(fp.water_density(4.0)), 1000.0, atol=1e-4)
    # Known value at 20 degC
    assert np.isclose(float(fp.water_density(20.0)), 997.939, atol=1e-3)
    # Array input is handled element-wise and monotonic away from 4 degC
    rho = fp.water_density(np.array([0.0, 4.0, 10.0, 25.0]))
    assert rho.shape == (4,)
    assert rho[1] >= rho.max() - 1e-9  # 4 degC is the maximum


def test_d_rho_dT_zero_at_maximum_and_sign():
    assert np.isclose(float(fp.d_rho_dT(4.0)), 0.0, atol=1e-6)
    assert float(fp.d_rho_dT(20.0)) < 0.0  # density falls above 4 degC
    assert float(fp.d_rho_dT(1.0)) > 0.0  # density rises below 4 degC
    # Array path including the singular point T = 4
    arr = fp.d_rho_dT(np.array([1.0, 4.0, 20.0]))
    assert np.isclose(arr[1], 0.0, atol=1e-6)


def test_water_viscosity_known_value_and_decreasing():
    assert np.isclose(float(fp.water_viscosity(20.0)), 0.0010129, atol=1e-6)
    mu = fp.water_viscosity(np.array([5.0, 20.0, 30.0]))
    assert np.all(np.diff(mu) < 0)  # viscosity decreases with temperature


def test_d_mu_dT_negative():
    assert float(fp.d_mu_dT(20.0)) < 0.0
    arr = fp.d_mu_dT(np.array([10.0, 20.0, 30.0]))
    assert np.all(arr < 0.0)


def test_particle_reynolds():
    re = fp.particle_reynolds(
        u_star=np.array([0.05, 0.1]),
        D50_m=np.array([0.01, 0.01]),
        rho=np.array([1000.0, 1000.0]),
        mu=np.array([1e-3, 1e-3]),
    )
    # Re_s = rho u* D50 / mu = 1000 * 0.05 * 0.01 / 1e-3 = 500
    assert np.isclose(re[0], 500.0)
    assert np.isclose(re[1], 1000.0)


def test_viscous_sublayer_thickness_finite_and_infinite():
    delta = fp.viscous_sublayer_thickness(
        u_star=np.array([0.1, 0.0]),
        rho=np.array([1000.0, 1000.0]),
        mu=np.array([1e-3, 1e-3]),
    )
    # nu = 1e-6; delta = 11.6 * 1e-6 / 0.1 = 1.16e-4
    assert np.isclose(delta[0], 11.6 * 1e-6 / 0.1)
    assert np.isinf(delta[1])  # u_star = 0 -> infinite sublayer


def test_B_s_log_layer_intercept():
    # _B_s should be finite and around the law-of-the-wall constant ~8.5
    b = fp._B_s(np.array([1.0, 100.0, 1e4]))
    assert np.all(np.isfinite(b))


def test_paphitis_tau_cr_star_range():
    re = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
    tau_cr_star = fp.paphitis_tau_cr_star(re)
    # Physically plausible band for the Shields curve
    assert np.all(tau_cr_star > 0.0)
    assert np.all(tau_cr_star < 0.2)


def test_shields_stress():
    tau = np.array([5.0, 10.0])
    tau_star = fp.shields_stress(
        tau=tau,
        rho=np.array([1000.0, 1000.0]),
        rho_s=2650.0,
        g=9.80665,
        D50_m=np.array([0.01, 0.01]),
    )
    expected = tau / ((2650.0 - 1000.0) * 9.80665 * 0.01)
    assert np.allclose(tau_star, expected)
