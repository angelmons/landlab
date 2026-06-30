"""Compute depth-averaged eddy viscosity for river-flow components."""

from __future__ import annotations

import numpy as np


class DepthAveragedTurbulenceModel:
    """Compute an algebraic depth-averaged eddy-viscosity closure.

    Parameters
    ----------
    model : str, optional
        Turbulence closure name. Supported values are ``"none"``,
        ``"constant"``, ``"smagorinsky"``, ``"parabolic"``, and
        ``"hybrid_additive"``.
    dx : float
        Grid spacing in the x direction.
    dy : float
        Grid spacing in the y direction.
    gravity : float, optional
        Gravitational acceleration.
    constant_eddy_viscosity : float, optional
        Constant eddy viscosity used by the ``"constant"`` model.
    smagorinsky_cs : float, optional
        Smagorinsky coefficient.
    filter_width_model : str, optional
        Filter-width model. Supported values are ``"grid"``, ``"depth"``,
        and ``"depth_limited_grid"``.
    filter_width_coefficient : float, optional
        Multiplicative coefficient for the selected filter width.
    parabolic_alpha : float, optional
        Coefficient in ``nu_t = alpha u_star h``.
    background_eddy_viscosity : float, optional
        Non-negative lower bound applied in wet cells.
    max_eddy_viscosity : float or None, optional
        Optional upper bound applied before the background floor.
    mask_dry_cells : bool, optional
        If True, set eddy viscosity to zero where depth is below
        ``dry_depth_threshold``.
    dry_depth_threshold : float, optional
        Depth threshold used for wet/dry masking.
    """

    _VALID_MODELS = {
        "none",
        "constant",
        "smagorinsky",
        "parabolic",
        "hybrid_additive",
    }
    _VALID_FILTER_WIDTHS = {"grid", "depth", "depth_limited_grid"}

    def __init__(
        self,
        model="none",
        dx=1.0,
        dy=1.0,
        gravity=9.80665,
        constant_eddy_viscosity=0.0,
        smagorinsky_cs=0.15,
        filter_width_model="grid",
        filter_width_coefficient=1.0,
        parabolic_alpha=0.067,
        background_eddy_viscosity=0.0,
        max_eddy_viscosity=None,
        mask_dry_cells=True,
        dry_depth_threshold=1.0e-6,
    ):
        model = str(model).lower()
        if model not in self._VALID_MODELS:
            raise ValueError(
                "turbulence_model must be one of "
                f"{tuple(sorted(self._VALID_MODELS))}, got {model!r}"
            )

        filter_width_model = str(filter_width_model).lower()
        if filter_width_model not in self._VALID_FILTER_WIDTHS:
            raise ValueError(
                "filter_width_model must be one of "
                f"{tuple(sorted(self._VALID_FILTER_WIDTHS))}, "
                f"got {filter_width_model!r}"
            )

        self.model = model
        self.dx = float(dx)
        self.dy = float(dy)
        self.gravity = float(gravity)
        self.constant_eddy_viscosity = float(constant_eddy_viscosity)
        self.smagorinsky_cs = float(smagorinsky_cs)
        self.filter_width_model = filter_width_model
        self.filter_width_coefficient = float(filter_width_coefficient)
        self.parabolic_alpha = float(parabolic_alpha)
        self.background_eddy_viscosity = float(background_eddy_viscosity)
        self.max_eddy_viscosity = (
            None if max_eddy_viscosity is None else float(max_eddy_viscosity)
        )
        self.mask_dry_cells = bool(mask_dry_cells)
        self.dry_depth_threshold = float(dry_depth_threshold)

        if self.constant_eddy_viscosity < 0.0:
            raise ValueError("constant_eddy_viscosity must be non-negative")
        if self.smagorinsky_cs < 0.0:
            raise ValueError("smagorinsky_cs must be non-negative")
        if self.filter_width_coefficient <= 0.0:
            raise ValueError("filter_width_coefficient must be positive")
        if self.parabolic_alpha < 0.0:
            raise ValueError("parabolic_alpha must be non-negative")
        if self.background_eddy_viscosity < 0.0:
            raise ValueError("background_eddy_viscosity must be non-negative")
        if self.max_eddy_viscosity is not None and self.max_eddy_viscosity < 0.0:
            raise ValueError("max_eddy_viscosity must be non-negative or None")
        if self.dry_depth_threshold < 0.0:
            raise ValueError("dry_depth_threshold must be non-negative")

    @property
    def is_active(self):
        """Return True if the turbulence closure contributes eddy viscosity."""
        return self.model != "none"

    @property
    def requires_shear_velocity(self):
        """Return True if the selected closure uses bed shear velocity."""
        return self.model in {"parabolic", "hybrid_additive"}

    def update(self, h, u, v, mannings_n=0.0):
        """Return eddy viscosity for the current hydraulic state.

        Parameters
        ----------
        h : ndarray
            Water depth at nodes.
        u : ndarray
            Depth-averaged x velocity at nodes.
        v : ndarray
            Depth-averaged y velocity at nodes.
        mannings_n : float or ndarray, optional
            Manning roughness used to estimate bed shear velocity for the
            parabolic closure.

        Returns
        -------
        ndarray
            Eddy viscosity at nodes with the same shape as ``h``.
        """
        h = np.asarray(h, dtype=float)
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        n_field = np.asarray(mannings_n, dtype=float)

        nu_t = np.zeros_like(h, dtype=float)
        if self.model in {"constant"}:
            nu_t += self.constant_eddy_viscosity
        if self.model in {"smagorinsky", "hybrid_additive"}:
            nu_t += self._smagorinsky(h, u, v)
        if self.model in {"parabolic", "hybrid_additive"}:
            nu_t += self._parabolic(h, u, v, n_field)

        return self._apply_safeguards(nu_t, h)

    def _smagorinsky(self, h, u, v):
        """Calculate the depth-averaged Smagorinsky eddy viscosity."""
        dx = self.dx
        dy = self.dy
        dudx = np.zeros_like(u, dtype=float)
        dudy = np.zeros_like(u, dtype=float)
        dvdx = np.zeros_like(v, dtype=float)
        dvdy = np.zeros_like(v, dtype=float)

        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dy)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * dx)
        dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dy)

        strain_magnitude = np.sqrt(2.0 * (dudx**2 + dvdy**2) + (dudy + dvdx) ** 2)
        delta = self._filter_width(h)
        return (self.smagorinsky_cs * delta) ** 2 * strain_magnitude

    def _parabolic(self, h, u, v, mannings_n):
        """Calculate bed-shear-controlled parabolic eddy viscosity."""
        h_safe = np.maximum(h, self.dry_depth_threshold)
        speed = np.sqrt(u**2 + v**2)
        shear_velocity = (
            np.sqrt(self.gravity) * mannings_n * speed / h_safe ** (1.0 / 6.0)
        )
        return self.parabolic_alpha * shear_velocity * h_safe

    def _filter_width(self, h):
        """Calculate the local filter width."""
        grid_width = self.filter_width_coefficient * np.sqrt(self.dx * self.dy)
        if self.filter_width_model == "grid":
            return grid_width
        depth_width = self.filter_width_coefficient * np.maximum(
            h, self.dry_depth_threshold
        )
        if self.filter_width_model == "depth":
            return depth_width
        return np.minimum(grid_width, depth_width)

    def _apply_safeguards(self, nu_t, h):
        """Apply non-negativity, caps, floors, and dry-cell masking."""
        nu_t = np.maximum(nu_t, 0.0)
        if self.max_eddy_viscosity is not None:
            nu_t = np.minimum(nu_t, self.max_eddy_viscosity)
        if self.background_eddy_viscosity > 0.0:
            nu_t = np.maximum(nu_t, self.background_eddy_viscosity)
        if self.mask_dry_cells:
            nu_t = np.where(h > self.dry_depth_threshold, nu_t, 0.0)
        return nu_t
