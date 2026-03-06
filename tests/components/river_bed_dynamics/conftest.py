"""
Shared pytest fixtures for RiverBedDynamics tests.

Fixtures
--------
rbd
    Basic 5×5 component with a realistic multi-fraction GSD.
    Velocity at link 20 is set to 10 m/s so that the
    velocity-previous-time seeding logic can be verified.

grid_5x5
    Raw 5×5 grid + flow fields only — no component attached.
    Useful for tests that need to control instantiation arguments.

rbd_parker
    5×5 component using the Parker 1990 equation and two GSD zones.
    Provides fractional transport, needed for diffusion and GSD tests.

rbd_diffusion
    Same as rbd_parker but with gravitational diffusion enabled
    (nonlinear mode, mu=0.5). Used for diffusion-specific tests.
"""

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverBedDynamics
from landlab.grid.mappers import map_mean_of_link_nodes_to_link

# ---------------------------------------------------------------------------
# Shared grid geometry constants
# ---------------------------------------------------------------------------
_SHAPE = (5, 5)
_SPACING = (100, 100)


# ---------------------------------------------------------------------------
# Helper: build a configured 5×5 grid ready for RiverBedDynamics
# ---------------------------------------------------------------------------
def _make_5x5_grid():
    """Return a 5×5 RasterModelGrid with all mandatory fields populated."""
    grid = RasterModelGrid(_SHAPE, xy_spacing=_SPACING[0])

    grid.at_node["topographic__elevation"] = np.array(
        [
            [1.07, 1.08, 1.09, 1.09, 1.09],
            [1.06, 1.07, 1.08, 1.09, 1.09],
            [1.00, 1.03, 1.07, 1.08, 1.09],
            [1.06, 1.07, 1.08, 1.09, 1.09],
            [1.07, 1.08, 1.09, 1.09, 1.09],
        ],
        dtype=float,
    ).flatten()

    grid.set_watershed_boundary_condition(grid.at_node["topographic__elevation"])

    # Required link fields
    grid.add_zeros("surface_water__depth", at="link")
    grid.add_zeros("surface_water__velocity", at="link")

    # Set a non-zero velocity at link 20 so that the velocity_prev_time
    # seeding (velocity_at_link copies surface_water__velocity when
    # surface_water__velocity_prev_time_link=None) can be tested.
    grid["link"]["surface_water__velocity"][20] = 10.0

    return grid


# ---------------------------------------------------------------------------
# Fixture: basic component (MPM, multi-fraction GSD)
# ---------------------------------------------------------------------------
@pytest.fixture
def rbd():
    """5×5 RiverBedDynamics component with MPM bedload equation.

    GSD covers gravel and sand fractions.  Velocity at link 20 is 10 m/s
    so that ``_surface_water__velocity_prev_time_link[20] == 10`` can be
    asserted in unit tests.
    """
    grid = _make_5x5_grid()

    gsd = [
        [128, 100],
        [64, 90],
        [32, 80],
        [16, 50],
        [8, 20],
        [4, 10],
        [2, 1],
        [1, 0],
    ]

    return RiverBedDynamics(grid, gsd=gsd)


# ---------------------------------------------------------------------------
# Fixture: raw grid only (no component)
# ---------------------------------------------------------------------------
@pytest.fixture
def grid_5x5():
    """5×5 grid with mandatory flow fields but no RiverBedDynamics instance.

    Use this fixture when the test needs to control instantiation arguments
    directly, e.g. to test that certain parameter combinations raise errors.
    """
    return _make_5x5_grid()


# ---------------------------------------------------------------------------
# Fixture: Parker 1990 component with two GSD zones
# ---------------------------------------------------------------------------
@pytest.fixture
def rbd_parker():
    """5×5 RiverBedDynamics with Parker 1990 equation and two GSD zones.

    Provides fractional bedload transport, which is needed for GSD-evolution
    and diffusion tests.
    """
    grid = RasterModelGrid(_SHAPE, xy_spacing=_SPACING[0])

    grid.at_node["topographic__elevation"] = np.array(
        [
            [1.07, 1.06, 1.00, 1.06, 1.07],
            [1.08, 1.07, 1.03, 1.07, 1.08],
            [1.09, 1.08, 1.07, 1.08, 1.09],
            [1.09, 1.09, 1.08, 1.09, 1.09],
            [1.09, 1.09, 1.09, 1.09, 1.09],
        ],
        dtype=float,
    ).flatten()

    grid.set_watershed_boundary_condition(grid.at_node["topographic__elevation"])

    grid.at_node["surface_water__depth"] = np.full(grid.number_of_nodes, 0.102)
    grid.at_node["surface_water__velocity"] = np.full(grid.number_of_nodes, 0.25)
    grid.at_link["surface_water__depth"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__depth"
    )
    grid.at_link["surface_water__velocity"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__velocity"
    )

    gsd = [[32, 100, 100], [16, 25, 50], [8, 0, 0]]
    gsd_loc = [
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
    ]

    return RiverBedDynamics(
        grid,
        gsd=gsd,
        bedload_equation="Parker1990",
        bed_surf__gsd_loc_node=gsd_loc,
        dt=1.0,
    )


# ---------------------------------------------------------------------------
# Fixture: Parker 1990 + gravitational diffusion enabled
# ---------------------------------------------------------------------------
@pytest.fixture
def rbd_diffusion():
    """5×5 RiverBedDynamics with Parker 1990 and gravitational diffusion.

    Identical setup to ``rbd_parker`` except ``use_bed_diffusion=True``
    with ``bed_diffusion_mode='nonlinear'`` and ``bed_diffusion_mu=0.5``.
    """
    grid = RasterModelGrid(_SHAPE, xy_spacing=_SPACING[0])

    grid.at_node["topographic__elevation"] = np.array(
        [
            [1.07, 1.06, 1.00, 1.06, 1.07],
            [1.08, 1.07, 1.03, 1.07, 1.08],
            [1.09, 1.08, 1.07, 1.08, 1.09],
            [1.09, 1.09, 1.08, 1.09, 1.09],
            [1.09, 1.09, 1.09, 1.09, 1.09],
        ],
        dtype=float,
    ).flatten()

    grid.set_watershed_boundary_condition(grid.at_node["topographic__elevation"])

    grid.at_node["surface_water__depth"] = np.full(grid.number_of_nodes, 0.102)
    grid.at_node["surface_water__velocity"] = np.full(grid.number_of_nodes, 0.25)
    grid.at_link["surface_water__depth"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__depth"
    )
    grid.at_link["surface_water__velocity"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__velocity"
    )

    gsd = [[32, 100, 100], [16, 25, 50], [8, 0, 0]]
    gsd_loc = [
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
    ]

    return RiverBedDynamics(
        grid,
        gsd=gsd,
        bedload_equation="Parker1990",
        bed_surf__gsd_loc_node=gsd_loc,
        dt=1.0,
        use_bed_diffusion=True,
        bed_diffusion_mode="nonlinear",
        bed_diffusion_mu=0.5,
        check_diffusion_cfl=False,  # suppress warnings in tests
    )
