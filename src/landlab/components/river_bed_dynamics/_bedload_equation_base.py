"""
Abstract base class for bedload transport equations used by RiverBedDynamics.

All concrete equation classes must implement :meth:`BedloadEquation.calculate`,
which takes the RiverBedDynamics component instance and returns the bedload
transport rate (and optionally the fractional GSD) at every link.

Adding a new equation
---------------------
1. Subclass :class:`BedloadEquation`.
2. Implement ``calculate(self, rbd) → (ndarray, ndarray | None)``.
3. Register it in :data:`EQUATION_REGISTRY` with a string key.
4. Pass that key as ``bedload_equation=`` in :class:`RiverBedDynamics.__init__`.

.. codeauthor:: Angel Monsalve (original equations)
.. codeauthor:: Phase-3 refactor — class hierarchy
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np


class BedloadEquation(ABC):
    """Abstract base class for all bedload transport equations.

    Concrete subclasses implement the :meth:`calculate` method, which reads
    whatever it needs from the ``rbd`` component instance and returns the
    per-link bedload transport rate and, optionally, the fractional grain-size
    distribution of the transported sediment.

    The interface is deliberately minimal: subclasses are free to use any
    fields stored on ``rbd`` without restriction.  Future refactors can narrow
    the interface once the class hierarchy is stable.
    """

    #: Human-readable name, used in error messages and repr.
    name: str = "unknown"

    @abstractmethod
    def calculate(self, rbd) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute bedload transport rate (and GSD) at every link.

        Parameters
        ----------
        rbd : RiverBedDynamics
            The component instance.  Implementations read shear stress,
            grain-size distributions, and other state from this object.

        Returns
        -------
        sed_transp__bedload_rate_link : ndarray, shape (n_links,)
            Volumetric bedload transport rate per unit width [m²/s] at each
            link.  Signed: positive in the positive link direction.
        sed_transp__bedload_gsd_link : ndarray, shape (n_links, n_grains) or None
            Fractional grain-size distribution of the bedload at each link,
            or ``None`` for equations that do not resolve grain fractions
            (e.g. MPM-style total-load equations).
        """

    def __repr__(self) -> str:  # pragma: no cover
        """Return a short string representation showing the class name."""
        return f"<{type(self).__name__}>"


# --------------------------------------------------------------------------- #
# Concrete wrappers — thin shells around the module-level functions
# --------------------------------------------------------------------------- #
# Each wrapper delegates to the existing function so that doctests and
# backward-compatible call sites continue to work unchanged.


class MPMEquation(BedloadEquation):
    """Meyer-Peter and Müller (1948) total bedload transport equation.

    Computes the total volumetric bedload rate per unit width using the
    classic MPM power-law relationship with critical Shields stress.  Does
    not resolve grain-size fractions in the bedload — all sediment is
    treated as a single class.

    Notes
    -----
    The critical Shields stress and transport coefficient are taken from
    the implementation in ``_bedload_eq_MPM_style`` and may be configured
    via the ``variable_critical_shear_stress`` flag on the component.

    References
    ----------
    Meyer-Peter, E., & Müller, R. (1948). Formulas for bed-load transport.
    *Proceedings of the 2nd IAHR Congress*, Stockholm, Sweden, 39–64.
    """

    name = "MPM"

    def calculate(self, rbd):
        """Delegate to the underlying module-level function and return results."""
        from . import _bedload_eq_MPM_style as _m

        return _m.bedload_equation(rbd), None


class FLvBEquation(BedloadEquation):
    """Fernandez Luque & van Beek (1976) total bedload transport equation.

    An alternative total-load formula derived from experiments with
    uniform sediment.  Returns a single transport rate with no fractional
    GSD breakdown.

    References
    ----------
    Fernandez Luque, R., & van Beek, R. (1976). Erosion and transport of
    bed-load sediment. *Journal of Hydraulic Research*, 14(2), 127–144.
    https://doi.org/10.1080/00221687609499677
    """

    name = "FLvB"

    def calculate(self, rbd):
        """Delegate to the underlying module-level function and return results."""
        from . import _bedload_eq_MPM_style as _m

        return _m.bedload_equation(rbd), None


class WongAndParkerEquation(BedloadEquation):
    """Wong and Parker (2006) corrected MPM equation.

    Reanalysis of the MPM dataset that corrects for form drag and produces
    a lower transport coefficient.  Total-load only (no fractional GSD).

    References
    ----------
    Wong, M., & Parker, G. (2006). Reanalysis and correction of bed-load
    relation of Meyer-Peter and Müller using their own database.
    *Journal of Hydraulic Engineering*, 132(11), 1159–1168.
    https://doi.org/10.1061/(ASCE)0733-9429(2006)132:11(1159)
    """

    name = "WongAndParker"

    def calculate(self, rbd):
        """Delegate to the underlying module-level function and return results."""
        from . import _bedload_eq_MPM_style as _m

        return _m.bedload_equation(rbd), None


class HuangEquation(BedloadEquation):
    """He Qing Huang (2010) reformulation of the MPM equation.

    Re-derived from dimensional analysis and fits to a broad dataset.
    Total-load only (no fractional GSD).

    References
    ----------
    Huang, H. Q. (2010). Reformulation of the bed load equation of Meyer-Peter
    and Müller in light of the linearity theory for alluvial channel flow.
    *Water Resources Research*, 46(9), W09533.
    https://doi.org/10.1029/2009WR008974
    """

    name = "Huang"

    def calculate(self, rbd):
        """Delegate to the underlying module-level function and return results."""
        from . import _bedload_eq_MPM_style as _m

        return _m.bedload_equation(rbd), None


class Parker1990Equation(BedloadEquation):
    """Parker (1990) surface-based fractional bedload transport equation.

    A hiding-function-based formula for mixed-size gravel transport.
    Computes a fractional GSD for the bedload as well as the total rate.
    Requires a multi-fraction GSD to be set up on the component.

    Notes
    -----
    Transport is resolved per grain-size fraction, using the surface GSD
    (``_bed_surf__gsd_link``) and the hiding/exposure correction of Parker
    (1990).  Results are written to both ``_sed_transp__bedload_rate_link``
    and ``_sed_transp__bedload_gsd_link``.

    References
    ----------
    Parker, G. (1990). Surface-based bedload transport relation for gravel
    rivers. *Journal of Hydraulic Research*, 28(4), 417–436.
    https://doi.org/10.1080/00221689009499058
    """

    name = "Parker1990"

    def calculate(self, rbd):
        """Delegate to the underlying module-level function and return results."""
        from . import _bedload_eq_Parker_1990 as _p

        return _p.bedload_equation(rbd)


class WilcockCrowe2003Equation(BedloadEquation):
    """Wilcock and Crowe (2003) surface-based mixed-size transport equation.

    Extends the two-fraction (sand–gravel) hiding function to a continuous
    GSD.  Produces fractional bedload transport rates suitable for tracking
    gravel–sand exchange and GSD evolution.

    Notes
    -----
    Requires a multi-fraction GSD that includes both sand (< 2 mm) and
    gravel fractions.  The sand fraction strongly influences the reference
    Shields stress for gravel through a phase-diagram hiding correction.

    References
    ----------
    Wilcock, P. R., & Crowe, J. C. (2003). Surface-based transport model
    for mixed-size sediment. *Journal of Hydraulic Engineering*, 129(2),
    120–128. https://doi.org/10.1061/(ASCE)0733-9429(2003)129:2(120)
    """

    name = "WilcockAndCrowe"

    def calculate(self, rbd):
        """Delegate to the underlying module-level function and return results."""
        from . import _bedload_eq_Wilcock_Crowe_2003 as _wc

        return _wc.bedload_equation(rbd)


# --------------------------------------------------------------------------- #
# Registry — maps user-facing string keys to equation classes
# --------------------------------------------------------------------------- #

#: Maps ``bedload_equation`` parameter strings to :class:`BedloadEquation`
#: subclasses.  Add new entries here to register additional equations.
EQUATION_REGISTRY: dict[str, type[BedloadEquation]] = {
    "MPM": MPMEquation,
    "FLvB": FLvBEquation,
    "WongAndParker": WongAndParkerEquation,
    "Huang": HuangEquation,
    "Parker1990": Parker1990Equation,
    "WilcockAndCrowe": WilcockCrowe2003Equation,
}
