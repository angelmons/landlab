"""Coverage tests for ``_utilities``, the velocity-driven shear-stress branch,
the fixed-link bedload restore paths in the MPM and Wilcock-Crowe kernels, and
the stratigraphy deposition/erosion bookkeeping plus CSV export.
"""

import os

import numpy as np
import pytest

from landlab import RasterModelGrid
from landlab.components import RiverBedDynamics
from landlab.components.river_bed_dynamics import _bedload_eq_MPM_style as mpm
from landlab.components.river_bed_dynamics import _stratigraphy as stratigraphy
from landlab.components.river_bed_dynamics import _utilities as utilities
from landlab.grid.mappers import map_mean_of_link_nodes_to_link

GSD = [[32, 100, 100], [16, 25, 50], [8, 0, 0]]
GSD_LOC = [
    [0, 1.0, 1.0, 1.0, 0],
    [0, 1.0, 1.0, 1.0, 0],
    [0, 1.0, 1.0, 1.0, 0],
    [0, 1.0, 1.0, 1.0, 0],
    [0, 1.0, 1.0, 1.0, 0],
]


def build_grid():
    grid = RasterModelGrid((5, 5))
    grid.at_node["topographic__elevation"] = [
        [1.07, 1.06, 1.00, 1.06, 1.07],
        [1.08, 1.07, 1.03, 1.07, 1.08],
        [1.09, 1.08, 1.07, 1.08, 1.09],
        [1.09, 1.09, 1.08, 1.09, 1.09],
        [1.09, 1.09, 1.09, 1.09, 1.09],
    ]
    grid.set_watershed_boundary_condition(grid.at_node["topographic__elevation"])
    grid.at_node["surface_water__depth"] = np.full(grid.number_of_nodes, 0.102)
    grid.at_node["surface_water__velocity"] = np.full(grid.number_of_nodes, 0.25)
    grid.at_link["surface_water__depth"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__depth"
    )
    grid.at_link["surface_water__velocity"] = map_mean_of_link_nodes_to_link(
        grid, "surface_water__velocity"
    )
    return grid


def _run(**kwargs):
    grid = build_grid()
    rbd = RiverBedDynamics(grid, gsd=GSD, bed_surf__gsd_loc_node=GSD_LOC, **kwargs)
    rbd.run_one_step()
    return grid, rbd


# ── _utilities ───────────────────────────────────────────────────────────────


def test_vector_mapper():
    grid = build_grid()
    u = np.zeros(grid.number_of_links)
    u[15] = -0.01
    u[19] = -0.02
    vector, magnitude = utilities.vector_mapper(grid, u)
    assert vector.shape == (grid.number_of_nodes, 2)
    assert magnitude.shape == (grid.number_of_nodes,)
    assert np.all(magnitude >= 0)


def test_map_gsd_from_link_to_node_both_locations():
    _, rbd = _run(bedload_equation="Parker1990")
    surf = utilities.map_gsd_from_link_to_node(rbd, location="bed_surf")
    load = utilities.map_gsd_from_link_to_node(rbd, location="bedload")
    assert surf.shape[0] == rbd._grid.number_of_nodes
    assert load.shape[0] == rbd._grid.number_of_nodes


def test_format_gsd_link_and_node():
    _, rbd = _run(bedload_equation="Parker1990")
    df_link = utilities.format_gsd(rbd, rbd._sed_transp__bedload_gsd_link)
    assert df_link.index[0].startswith("Link_")
    node_gsd = utilities.map_gsd_from_link_to_node(rbd, location="bedload")
    df_node = utilities.format_gsd(rbd, node_gsd)
    assert df_node.index[0].startswith("Node_")


def test_get_available_fields():
    fields = utilities.get_available_fields()
    assert isinstance(fields, list)
    names = [f[0] for f in fields]
    assert "rbd._surface_water__shear_stress_link" in names
    # Sorted, with (name, units) pairs
    assert names == sorted(names)


# ── _shear_stress: velocity-driven formulation ───────────────────────────────


def test_velocity_driven_formulation():
    grid, rbd = _run(
        bedload_equation="MPM",
        shear_stress_formulation="velocity_driven",
        mannings_n=0.03,
    )
    assert np.all(np.isfinite(rbd._surface_water__shear_stress_link))


def test_velocity_driven_requires_mannings_n():
    grid = build_grid()
    with pytest.raises(ValueError):
        RiverBedDynamics(
            grid,
            gsd=GSD,
            bed_surf__gsd_loc_node=GSD_LOC,
            shear_stress_formulation="velocity_driven",
            mannings_n=None,
        )


# ── fixed-link bedload restore (MPM and Wilcock-Crowe) ────────────────────────


def test_mpm_fixed_bedload_rate_link():
    grid = build_grid()
    fix_link = np.zeros(grid.number_of_links, dtype=int)
    fix_link[15] = 1
    rbd = RiverBedDynamics(
        grid,
        gsd=GSD,
        bed_surf__gsd_loc_node=GSD_LOC,
        bedload_equation="MPM",
        sed_transp__bedload_rate_fix_link=fix_link,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._sed_transp__bedload_rate_link))


def test_wilcock_crowe_fixed_rate_and_gsd_links():
    grid = build_grid()
    n_links = grid.number_of_links
    n_frac = len(GSD) - 1  # GSD value columns
    fix_rate = np.zeros(n_links, dtype=float)
    fix_rate[15] = 1.0
    # GSD fix array is (n_links, n_frac); a non-zero row marks an imposed link
    fix_gsd = np.zeros((n_links, n_frac), dtype=float)
    fix_gsd[15, :] = [0.6, 0.4]
    rbd = RiverBedDynamics(
        grid,
        gsd=GSD,
        bed_surf__gsd_loc_node=GSD_LOC,
        bedload_equation="WilcockAndCrowe",
        sed_transp__bedload_rate_fix_link=fix_rate,
        sed_transp__bedload_gsd_fix_link=fix_gsd,
    )
    rbd.run_one_step()
    assert np.all(np.isfinite(rbd._sed_transp__bedload_gsd_link))


def test_deprecated_bedload_equation_dispatcher():
    _, rbd = _run(bedload_equation="MPM")
    rbd._bedload_equation = "MPM"  # legacy attribute consumed by the old dispatcher
    with pytest.warns(DeprecationWarning):
        qb = mpm.bedload_equation(rbd)
    assert qb.shape[0] == rbd._grid.number_of_links


# ── _stratigraphy: deposition/erosion bookkeeping and CSV export ──────────────


def test_stratigraphy_layer_cycling_and_write(tmp_path):
    grid = build_grid()
    rbd = RiverBedDynamics(
        grid,
        gsd=GSD,
        bed_surf__gsd_loc_node=GSD_LOC,
        bedload_equation="Parker1990",
        track_stratigraphy=True,
        num_cycles_to_process_strat=1,
        bed_surf_new_layer_thick=1e-4,  # tiny layer so deposition/erosion crosses it
        morfac=50,  # amplify morphological change to force layer turnover
    )
    for _ in range(8):
        rbd.run_one_step()

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        stratigraphy.write_evolution(rbd)
        assert (tmp_path / "Stratigraphy_evolution.csv").exists()
    finally:
        os.chdir(cwd)
