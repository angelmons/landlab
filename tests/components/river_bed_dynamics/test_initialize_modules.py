"""Direct unit tests for the bed-dynamics field and GSD initializers.

These cover the branches that only execute when the user *supplies* a field
(correct size, wrong size, and empty), plus the sub-2 mm grain-size handling
(``adds_2mm_to_gsd`` and ``remove_sand_from_gsd``) that default coarse GSDs
never trigger.
"""

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components.river_bed_dynamics import _initialize_fields as initf
from landlab.components.river_bed_dynamics import _initialize_gsd as initg


@pytest.fixture
def small_grid():
    return RasterModelGrid((4, 4))


# ── _initialize_fields ───────────────────────────────────────────────────────


def test_field_at_node_paths(small_grid):
    n = small_grid.number_of_nodes
    # None -> zeros
    assert initf.field_at_node(small_grid, None).shape == (n,)
    # correct size -> flattened copy
    good = np.ones(n, dtype=int)
    assert np.array_equal(initf.field_at_node(small_grid, good), good)
    # wrong size -> falls back to zeros
    bad = np.ones(n + 3, dtype=int)
    assert np.all(initf.field_at_node(small_grid, bad) == 0)
    # empty array -> returned as-is (size 0)
    empty = initf.field_at_node(small_grid, np.array([], dtype=int))
    assert empty.size == 0


def test_field_at_link_paths(small_grid):
    n = small_grid.number_of_links
    assert initf.field_at_link(small_grid, None).shape == (n,)
    good = np.arange(n, dtype=float)
    assert np.allclose(initf.field_at_link(small_grid, good), good)
    bad = np.ones(n + 2, dtype=float)
    assert np.all(initf.field_at_link(small_grid, bad) == 0)


def test_gsd_at_link_paths(small_grid):
    n = small_grid.number_of_links
    gsd = np.array([[32, 100], [8, 0]], dtype=float)  # 2 size classes -> 1 fraction col
    ncols = gsd.shape[0] - 1
    # None -> zeros of shape (n_links, ncols)
    out = initf.gsd_at_link(small_grid, None, gsd)
    assert out.shape == (n, ncols)
    # correct shape passes through
    good = np.ones((n, ncols), dtype=float)
    assert np.allclose(initf.gsd_at_link(small_grid, good, gsd), good)
    # wrong shape -> zeros
    bad = np.ones((n + 1, ncols), dtype=float)
    assert np.all(initf.gsd_at_link(small_grid, bad, gsd) == 0)


def test_velocity_at_link_paths(small_grid):
    n = small_grid.number_of_links
    small_grid.add_zeros("surface_water__velocity", at="link")
    small_grid["link"]["surface_water__velocity"][:] = 0.3
    # None -> copy of the grid field
    out = initf.velocity_at_link(small_grid, None)
    assert np.allclose(out, 0.3)
    # correct size passes through
    good = np.full(n, 0.5)
    assert np.allclose(initf.velocity_at_link(small_grid, good), good)
    # wrong size -> falls back to grid field copy
    bad = np.full(n + 4, 0.7)
    assert np.allclose(initf.velocity_at_link(small_grid, bad), 0.3)


# ── _initialize_gsd ──────────────────────────────────────────────────────────


def test_adds_2mm_to_gsd_inserts_row():
    gsd = [[32, 100, 100], [16, 25, 50], [1, 0, 0]]  # has a sub-2 mm class, no 2 mm
    out = initg.adds_2mm_to_gsd(gsd)
    # A 2 mm row should now be present
    assert np.any(np.isclose(out[:, 0], 2.0))
    # And the array grew by one row
    assert out.shape[0] == len(gsd) + 1


def test_adds_2mm_to_gsd_noop_for_coarse_gsd():
    gsd = [[32, 100, 100], [16, 25, 50], [8, 0, 0]]  # all >= 2 mm
    out = initg.adds_2mm_to_gsd(gsd)
    assert out.shape[0] == len(gsd)  # unchanged


def test_remove_sand_from_gsd_parker():
    gsd = [[32, 100, 100], [16, 60, 70], [8, 30, 40], [1, 0, 0]]
    out = initg.remove_sand_from_gsd(gsd, "Parker1990")
    # Sand (< 2 mm) removed, fractions renormalised so coarsest ends at 100
    assert np.all(out[:, 0] >= 2)
    assert np.isclose(out[0, 1], 100.0)


def test_remove_sand_from_gsd_other_eq_is_noop():
    gsd = [[32, 100, 100], [16, 60, 70], [1, 0, 0]]
    out = initg.remove_sand_from_gsd(gsd, "MPM")
    assert out.shape[0] == len(gsd)
