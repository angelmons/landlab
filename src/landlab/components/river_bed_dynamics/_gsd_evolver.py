"""
GSD evolution via the Toro-Escobar fractional Exner equation.

Extracts the grain-size distribution update logic from
``RiverBedDynamics.update_bed_surf_gsd()`` into a focused, testable class.
All physics and numerics are unchanged; only the housing changed.

Reference
---------
Toro-Escobar, C. M., Paola, C., & Parker, G. (1996). Transfer function for
the deposition of poorly sorted gravel in response to streambed aggradation.
Journal of Hydraulic Research, 34(1), 35–53.
https://doi.org/10.1080/00221689609498763

.. codeauthor:: Angel Monsalve (original implementation)
.. codeauthor:: Phase-3C refactor — class extraction
.. codeauthor:: Phase-4B — TVD minmod advection scheme
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# TVD minmod helpers (Phase 4B)                                                 #
# --------------------------------------------------------------------------- #


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise minmod limiter: smaller magnitude when same sign, else 0.

    Examples
    --------
    >>> import numpy as np
    >>> from . import _gsd_evolver as ev
    >>> ev._minmod(np.array([1.0, -1.0, 2.0, 0.0]), np.array([2.0, -3.0, -1.0, 1.0]))
    array([ 1., -1.,  0.,  0.])
    """
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


class GSDEvolver:
    """Abstract base class for grain-size distribution evolution schemes.

    Concrete subclasses implement :meth:`evolve`, which updates the bed
    surface GSD for one time step.
    """

    def evolve(self, rbd) -> None:  # pragma: no cover
        """Update the bed surface GSD for one time step.

        Parameters
        ----------
        rbd : RiverBedDynamics
            The component instance.  Implementations read and write GSD
            state directly on this object.
        """
        raise NotImplementedError


class ToroEscobarEvolver(GSDEvolver):
    """Toro-Escobar (1996) fractional Exner GSD evolution.

    Implements the upwind finite-difference Exner equation for each grain
    fraction, using the TVD-like flux scheme of Toro-Escobar et al. (1996).
    The exchange layer fraction ``gsd_FIexc`` blends subsurface and active-
    layer GSD depending on the local divergence sign.

    Parameters
    ----------
    gsd_advection_scheme : str, optional
        Spatial discretisation for the fractional bedload flux divergence.

        * ``"upwind"`` (default) — blended upwind / Lax–Wendroff scheme
          controlled by ``rbd._alpha``.
        * ``"tvd_minmod"`` — TVD minmod-limited scheme; second-order in
          smooth regions, first-order at GSD fronts.  Produces sharper
          fronts with less numerical diffusion than pure upwind.

    Notes
    -----
    This class holds no mutable state between calls; all arrays are read
    from (and written back to) the ``rbd`` component instance.
    """

    _VALID_SCHEMES = frozenset({"upwind", "tvd_minmod"})

    def __init__(self, gsd_advection_scheme: str = "upwind") -> None:
        """Initialise the GSD evolver.

        Parameters
        ----------
        gsd_advection_scheme : str, optional
            Spatial scheme for the fractional bedload flux divergence:

            * ``"upwind"`` (default) — blended upwind / Lax–Wendroff
              controlled by ``rbd._alpha``.  First-order when ``alpha = 1``,
              second-order when ``alpha = 0``.  Diffusive but stable.
            * ``"tvd_minmod"`` — Total Variation Diminishing scheme with
              minmod limiter.  Second-order in smooth regions; reverts to
              first-order upwind at GSD fronts.  Sharper fronts, less
              numerical diffusion than pure upwind.

        Raises
        ------
        ValueError
            If ``gsd_advection_scheme`` is not one of the valid options.
        """
        if gsd_advection_scheme not in self._VALID_SCHEMES:
            raise ValueError(
                f"Unknown gsd_advection_scheme {gsd_advection_scheme!r}. "
                f"Valid options: {sorted(self._VALID_SCHEMES)}"
            )
        self._scheme = gsd_advection_scheme

    def evolve(self, rbd) -> None:
        """Evolve the bed surface GSD by one time step.

        Implements the fractional Exner equation of Toro-Escobar et al. (1996)
        for each grain-size fraction :math:`j`:

        .. math::

            (1 - \\lambda_p) L_a \\frac{\\partial F_j}{\\partial t}
            = -\\nabla \\cdot (q_b p_j) + (F^{exc}_j - F_j)
              \\nabla \\cdot q_b

        where :math:`F_j` is the surface fraction, :math:`p_j` is the
        fractional bedload GSD, :math:`q_b` is the total bedload rate,
        :math:`L_a` is the active-layer thickness, and :math:`F^{exc}_j`
        is the exchange-layer fraction (subsurface during erosion, blended
        during deposition).

        Reads all required state from ``rbd`` and writes back:

        * ``rbd._bed_surf__gsd_link`` — updated surface GSD at links [−]
        * ``rbd._bed_surf__gsd_node`` — updated surface GSD at nodes [−]
        * ``rbd._topogr__elev_orig_link`` — updated reference elevation [m]
        * ``rbd._bed_surf__gsd_residual_max`` — max ``|Σf_i − 1|`` before
          renormalisation (diagnostic)
        * ``rbd._bed_surf__gsd_residual_mean`` — mean diagnostic

        The method is a no-op when ``rbd._track_stratigraphy is False``.

        Parameters
        ----------
        rbd : RiverBedDynamics
            The component instance.  Must have been updated through
            ``bedload_equation()`` and ``calculate_net_bedload()`` before
            this method is called.

        Raises
        ------
        UserWarning
            When ``rbd._check_gsd_residual`` is ``True`` and the
            pre-renormalisation residual exceeds ``rbd._gsd_residual_threshold``.
        """
        if not rbd._track_stratigraphy:
            return

        # ── Unpack frequently accessed state ─────────────────────────────
        g = rbd._grid
        n_links = g.number_of_links
        n_cols = g.number_of_node_columns

        gsd_F = rbd._bed_surf__gsd_link
        gsd_Fs = rbd._bed_subsurf__gsd_link
        pl = rbd._sed_transp__bedload_gsd_link
        qbT = rbd._sed_transp__bedload_rate_link
        la = np.reshape(rbd._bed_surf__act_layer_thick_link, [n_links, 1])
        la0 = np.reshape(rbd._bed_surf__act_layer_thick_prev_time_link, [n_links, 1])
        z = g["link"]["topographic__elevation"]
        z0 = rbd._topogr__elev_orig_link

        lps = rbd._lambda_p
        dx = g.dx
        dy = g.dy
        alpha = rbd._alpha
        dt = g._dt
        dv = 2 * n_cols - 1

        qbT = np.reshape(qbT, [n_links, 1])

        # Reuse pre-allocated scratch arrays.
        # Proof of coverage: hl ∪ hlL ∪ hlR = all horizontal links,
        # vl ∪ vlB ∪ vlT = all vertical links → every index written before read.
        qbTdev = rbd._scratch_qbTdev
        qjj1dev = rbd._scratch_qjj1dev

        # ── Horizontal link topology ──────────────────────────────────────
        hlL = rbd._topo_hlL
        hlR = rbd._topo_hlR
        hl = rbd._topo_hl

        hl_pos = hl[np.where(qbT[hl][:, 0] >= 0)]
        hl_neg = hl[np.where(qbT[hl][:, 0] < 0)]
        hlL_pos = hlL[np.where(qbT[hlL][:, 0] >= 0)]
        hlL_neg = hlL[np.where(qbT[hlL][:, 0] < 0)]
        hlR_pos = hlR[np.where(qbT[hlR][:, 0] >= 0)]
        hlR_neg = hlR[np.where(qbT[hlR][:, 0] < 0)]

        # ── Horizontal: total qb divergence ──────────────────────────────
        qbTdev[hl_pos] = (
            alpha * (qbT[hl_pos] - qbT[hl_pos - 1]) / dy
            + (1 - alpha) * (qbT[hl_pos + 1] - qbT[hl_pos]) / dy
        )
        qbTdev[hl_neg] = (
            alpha * (qbT[hl_neg] - qbT[hl_neg + 1]) / dy
            + (1 - alpha) * (qbT[hl_neg - 1] - qbT[hl_neg]) / dy
        )
        qbTdev[hlL_pos] = (qbT[hlL_pos + 1] - qbT[hlL_pos]) / dy
        qbTdev[hlL_neg] = (qbT[hlL_neg] - qbT[hlL_neg - 1]) / dy
        qbTdev[hlR_pos] = (qbT[hlR_pos] - qbT[hlR_pos - 1]) / dy
        qbTdev[hlR_neg] = (qbT[hlR_neg - 1] - qbT[hlR_neg]) / dy

        # ── Horizontal: fractional qb flux divergence ─────────────────────
        if self._scheme == "tvd_minmod":
            # TVD minmod — second-order in smooth regions (Phase 4B)
            for idx in [hl_pos]:
                if idx.size:
                    up = qbT[idx] * pl[idx, :] - qbT[idx - 1] * pl[idx - 1, :]
                    dn = qbT[idx + 1] * pl[idx + 1, :] - qbT[idx] * pl[idx, :]
                    qjj1dev[idx, :] = (up + _minmod(up, dn)) / dy
            for idx in [hl_neg]:
                if idx.size:
                    up = qbT[idx] * pl[idx, :] - qbT[idx + 1] * pl[idx + 1, :]
                    dn = qbT[idx - 1] * pl[idx - 1, :] - qbT[idx] * pl[idx, :]
                    qjj1dev[idx, :] = (up + _minmod(up, dn)) / dy
            # Boundary: pure first-order upwind (no neighbour on one side)
            qjj1dev[hlL_pos, :] = (
                qbT[hlL_pos + 1] * pl[hlL_pos + 1, :] - qbT[hlL_pos] * pl[hlL_pos, :]
            ) / dy
            qjj1dev[hlL_neg, :] = (
                qbT[hlL_neg] * pl[hlL_neg, :] - qbT[hlL_neg - 1] * pl[hlL_neg - 1, :]
            ) / dy
            qjj1dev[hlR_pos, :] = (
                qbT[hlR_pos] * pl[hlR_pos, :] - qbT[hlR_pos - 1] * pl[hlR_pos - 1, :]
            ) / dy
            qjj1dev[hlR_neg, :] = (
                qbT[hlR_neg - 1] * pl[hlR_neg - 1, :] - qbT[hlR_neg] * pl[hlR_neg, :]
            ) / dy
        else:
            # Original blended upwind/Lax-Wendroff scheme (default)
            qjj1dev[hl_pos, :] = (
                alpha
                * (qbT[hl_pos] * pl[hl_pos, :] - qbT[hl_pos - 1] * pl[hl_pos - 1, :])
                / dy
                + (1 - alpha)
                * (qbT[hl_pos + 1] * pl[hl_pos + 1] - qbT[hl_pos] * pl[hl_pos])
                / dy
            )
            qjj1dev[hl_neg, :] = (
                alpha
                * (qbT[hl_neg] * pl[hl_neg, :] - qbT[hl_neg + 1] * pl[hl_neg + 1, :])
                / dy
                + (1 - alpha)
                * (qbT[hl_neg - 1] * pl[hl_neg - 1] - qbT[hl_neg] * pl[hl_neg])
                / dy
            )
            qjj1dev[hlL_pos, :] = (
                qbT[hlL_pos + 1] * pl[hlL_pos + 1, :] - qbT[hlL_pos] * pl[hlL_pos, :]
            ) / dy
            qjj1dev[hlL_neg, :] = (
                qbT[hlL_neg] * pl[hlL_neg, :] - qbT[hlL_neg - 1] * pl[hlL_neg - 1, :]
            ) / dy
            qjj1dev[hlR_pos, :] = (
                qbT[hlR_pos] * pl[hlR_pos, :] - qbT[hlR_pos - 1] * pl[hlR_pos - 1, :]
            ) / dy
            qjj1dev[hlR_neg, :] = (
                qbT[hlR_neg - 1] * pl[hlR_neg - 1, :] - qbT[hlR_neg] * pl[hlR_neg, :]
            ) / dy

        # ── Vertical link topology ────────────────────────────────────────
        vlB = rbd._topo_vlB
        vlT = rbd._topo_vlT
        vl = rbd._topo_vl

        vl_pos = vl[np.where(qbT[vl][:, 0] >= 0)]
        vl_neg = vl[np.where(qbT[vl][:, 0] < 0)]
        vlB_pos = vlB[np.where(qbT[vlB][:, 0] >= 0)]
        vlB_neg = vlB[np.where(qbT[vlB][:, 0] < 0)]
        vlT_pos = vlT[np.where(qbT[vlT][:, 0] >= 0)]
        vlT_neg = vlT[np.where(qbT[vlT][:, 0] < 0)]

        # ── Vertical: total qb divergence ─────────────────────────────────
        qbTdev[vl_pos] = (
            alpha * (qbT[vl_pos] - qbT[vl_pos - dv]) / dx
            + (1 - alpha) * (qbT[vl_pos + dv] - qbT[vl_pos]) / dx
        )
        qbTdev[vl_neg] = (
            alpha * (qbT[vl_neg] - qbT[vl_neg + dv]) / dx
            + (1 - alpha) * (qbT[vl_neg - dv] - qbT[vl_neg]) / dx
        )
        qbTdev[vlB_pos] = (qbT[vlB_pos + dv] - qbT[vlB_pos]) / dx
        qbTdev[vlB_neg] = (qbT[vlB_neg] - qbT[vlB_neg + dv]) / dx
        qbTdev[vlT_pos] = (qbT[vlT_pos] - qbT[vlT_pos - dv]) / dx
        qbTdev[vlT_neg] = (qbT[vlT_neg - dv] - qbT[vlT_neg]) / dx

        # Round away floating-point noise before sign-based branching
        qbTdev = np.round(qbTdev, 8)

        # ── Exchange layer GSD ────────────────────────────────────────────
        # gsd_FIexc = gsd_Fs everywhere, overridden where divergence ≤ 0
        gsd_FIexc = rbd._scratch_gsd_FIexc
        np.copyto(gsd_FIexc, gsd_Fs)
        (id0,) = np.where(qbTdev[:, 0] <= 0)
        gsd_FIexc[id0, :] = 0.7 * gsd_F[id0, :] + 0.3 * pl[id0, :]

        # ── Vertical: fractional qb flux divergence ───────────────────────
        if self._scheme == "tvd_minmod":
            for idx in [vl_pos]:
                if idx.size:
                    up = qbT[idx] * pl[idx, :] - qbT[idx - dv] * pl[idx - dv, :]
                    dn = qbT[idx + dv] * pl[idx + dv, :] - qbT[idx] * pl[idx, :]
                    qjj1dev[idx, :] = (up + _minmod(up, dn)) / dx
            for idx in [vl_neg]:
                if idx.size:
                    up = qbT[idx] * pl[idx, :] - qbT[idx + dv] * pl[idx + dv, :]
                    dn = qbT[idx - dv] * pl[idx - dv, :] - qbT[idx] * pl[idx, :]
                    qjj1dev[idx, :] = (up + _minmod(up, dn)) / dx
            # Boundary: pure first-order upwind
            qjj1dev[vlB_pos, :] = (
                qbT[vlB_pos + dv] * pl[vlB_pos + dv, :] - qbT[vlB_pos] * pl[vlB_pos, :]
            ) / dx
            qjj1dev[vlB_neg, :] = (
                qbT[vlB_neg] * pl[vlB_neg, :] - qbT[vlB_neg + dv] * pl[vlB_neg + dv, :]
            ) / dx
            qjj1dev[vlT_pos, :] = (
                qbT[vlT_pos] * pl[vlT_pos, :] - qbT[vlT_pos - dv] * pl[vlT_pos - dv, :]
            ) / dx
            qjj1dev[vlT_neg, :] = (
                qbT[vlT_neg - dv] * pl[vlT_neg - dv, :] - qbT[vlT_neg] * pl[vlT_neg, :]
            ) / dx
        else:
            qjj1dev[vl_pos, :] = (
                alpha
                * (qbT[vl_pos] * pl[vl_pos, :] - qbT[vl_pos - dv] * pl[vl_pos - dv, :])
                / dx
                + (1 - alpha)
                * (qbT[vl_pos + dv] * pl[vl_pos + dv] - qbT[vl_pos] * pl[vl_pos])
                / dx
            )
            qjj1dev[vl_neg, :] = (
                alpha
                * (qbT[vl_neg] * pl[vl_neg, :] - qbT[vl_neg + dv] * pl[vl_neg + dv, :])
                / dx
                + (1 - alpha)
                * (qbT[vl_neg - dv] * pl[vl_neg - dv] - qbT[vl_neg] * pl[vl_neg])
                / dx
            )
            qjj1dev[vlB_pos, :] = (
                qbT[vlB_pos + dv] * pl[vlB_pos + dv, :] - qbT[vlB_pos] * pl[vlB_pos, :]
            ) / dx
            qjj1dev[vlB_neg, :] = (
                qbT[vlB_neg] * pl[vlB_neg, :] - qbT[vlB_neg + dv] * pl[vlB_neg + dv, :]
            ) / dx
            qjj1dev[vlT_pos, :] = (
                qbT[vlT_pos] * pl[vlT_pos, :] - qbT[vlT_pos - dv] * pl[vlT_pos - dv, :]
            ) / dx
            qjj1dev[vlT_neg, :] = (
                qbT[vlT_neg - dv] * pl[vlT_neg - dv, :] - qbT[vlT_neg] * pl[vlT_neg, :]
            ) / dx

        # ── Fractional Exner update ───────────────────────────────────────
        qjj2dev = gsd_FIexc * np.reshape(qbTdev, [n_links, 1])
        gsd_Fnew = gsd_F + dt * (-qjj1dev + qjj2dev) / (1 - lps) / la

        if rbd._current_t > 0:  # skip layer-thickness term at t = 0
            gsd_Fnew += (gsd_FIexc - gsd_F) / la * (la - la0)

        # ── Renormalize ───────────────────────────────────────────────────
        # Phase 4C.1: compute residual |Σfi - 1| before renormalisation
        row_sums = np.sum(gsd_Fnew, axis=1)
        residuals = np.abs(row_sums - 1.0)
        rbd._bed_surf__gsd_residual_max = float(residuals.max())
        rbd._bed_surf__gsd_residual_mean = float(residuals.mean())

        # Phase 4C.2: warn if residual exceeds threshold
        if (
            rbd._check_gsd_residual
            and rbd._bed_surf__gsd_residual_max > rbd._gsd_residual_threshold
        ):
            import warnings

            warnings.warn(
                f"GSD residual max = {rbd._bed_surf__gsd_residual_max:.2e} "
                f"> threshold {rbd._gsd_residual_threshold:.2e} at t = "
                f"{rbd._current_t:.4g} s.  This indicates numerical drift "
                f"in the fractional Exner equation; consider reducing dt or "
                f"switching to gsd_advection_scheme='tvd_minmod'.",
                UserWarning,
                stacklevel=4,
            )

        # Phase 4C.3: N−1 fraction tracking — recover last fraction from rest
        # Evolve only the first (n_grains − 1) fractions; set the last as
        # 1 − Σ(rest).  This eliminates the accumulated drift from renorm.
        if rbd._gsd_n_minus_1:
            gsd_Fnew[:, -1] = np.maximum(0.0, 1.0 - np.sum(gsd_Fnew[:, :-1], axis=1))

        gsd_Fnew[gsd_Fnew <= 0] = 0
        gsd_Fnew = gsd_Fnew / np.reshape(np.sum(gsd_Fnew, axis=1), [n_links, 1])
        gsd_Fnew = np.nan_to_num(gsd_Fnew)

        # ── Boundary conditions ───────────────────────────────────────────
        gsd_Fnew[rbd._outlet_links] = gsd_F[rbd._outlet_links]
        gsd_Fnew[rbd._bed_surf__gsd_fix_link] = gsd_F[rbd._bed_surf__gsd_fix_link]
        gsd_Fnew[rbd._boundary_links] = gsd_F[rbd._boundary_links]

        # ── Erosion-below-original restoration ────────────────────────────
        id_eroded = np.where(z < z0)[0]
        if id_eroded.size > 0:
            gsd_Fnew[id_eroded] = gsd_F[id_eroded]
            z0[id_eroded] = z[id_eroded]

        # ── Write results back to component ───────────────────────────────
        from . import _utilities as utilities

        rbd._topogr__elev_orig_link = z0.copy()
        rbd._bed_surf__gsd_link = gsd_Fnew.copy()
        rbd._bed_surf__gsd_node = utilities.map_gsd_from_link_to_node(rbd)
