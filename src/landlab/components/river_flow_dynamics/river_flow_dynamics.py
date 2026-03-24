"""Simulate surface fluid flow based on Casulli and Cheng (1992).

This component implements a semi-implicit, semi-Lagrangian finite-volume approximation of
the depth-averaged shallow water equations originally proposed by Casulli and Cheng in 1992,
and subsequent related work.

Written by Sebastian Bernal and Angel Monsalve.

Compared to the version published on JOSS (v1.0), the current version includes:

Numerical robustness:
- Cached raster link adjacency tables (_build_raster_link_tables): O(1) neighbour
  lookup via precomputed _adj_hlinks / _adj_vlinks arrays, replacing the original
  coordinate-search implementation that built O(N^2) temporary matrices on every
  call.
- O(1) nearest-link mapping in find_nearest_link via integer index arithmetic on
  _hlink_id / _vlink_id grids, replacing an O(N^2) coordinate-search and
  boolean-mask approach.
- Dynamic link-length mapping (_link_lengths): assigns dx to horizontal links and
  dy to vertical links for correct directional pressure gradient scaling in both
  G-faces and velocity update.
- Zero-Gradient (Neumann) open boundary conditions replacing the previous
  depth-preserving form.
- mode="clip" in find_nearest_node to prevent index-out-of-bounds exceptions at
  grid edges.
- min_chezy_depth floor in _compute_a_faces to prevent singular Chezy friction in
  very thin films.
- max_pathline_substeps cap in path_line_tracing to prevent infinite loops in
  backward tracking.
- coord_tol floating-point guard for boundary hit detection in path_line_tracing.

Code architecture:
- Refactored run_one_step (previously a single ~35,000-character monolithic method)
  into 12 private methods: _compute_a_faces, _advect_u_velocity,
  _advect_v_velocity, _compute_g_faces, _solve_pressure_correction,
  _apply_boundary_conditions_eta, _update_velocity,
  _apply_boundary_conditions_vel, _update_depth, _write_grid_fields,
  _advance_time_history, _find_upstream_nodes.
- Public property accessors: wet_nodes, wet_links, water_depth, water_velocity.
- Public property accessors elapsed_time and current_dt for API symmetry with
  RiverFlowDynamics_HLLC, enabling drop-in substitution between the two
  components.
- Input validation in __init__ with clear error messages for dt, theta,
  threshold_depth, and mannings_n.
- np.broadcast_to(...).copy() fix for the read-only view bug on the initial
  time-history arrays.
- Precomputed topology arrays _core_adjacent_nodes and _core_adjacent_links used
  to assemble the sparse pressure matrix each step, replacing per-step O(N^2)
  list comprehensions. Note: the COO matrix is assembled fresh each step using
  these precomputed arrays; the sparsity structure itself is not cached across
  steps.
- Sparse COO->CSR PCG solve with Jacobi preconditioner (core nodes only),
  replacing the original dense A[np.ix_(core_nodes, core_nodes)] extraction and
  solve.
- np.hypot for speed magnitude (numerically cleaner than manual sqrt(u^2 + v^2)).
- In-place [:] field assignment in _write_grid_fields to avoid per-step array
  reallocation.
- Correct time-history ordering in _advance_time_history (bug fix: original
  overwrote time N before copying to N-1).
- Two new output fields: surface_water__x_velocity and surface_water__y_velocity
  (node-centred velocity components averaged from horizontal and vertical links
  respectively), enabling direct velocity comparison with RiverFlowDynamics_HLLC.
- Frictionless shortcut in _compute_a_faces: when mannings_n == 0, the Chezy
  division is bypassed entirely (setting a_links = h_at_N_at_links directly) to
  avoid a RuntimeWarning: divide by zero that was otherwise harmless but noisy.

Boundary condition robustness:
- Stage-preserving zero-gradient outlet BC: enforces eta_b = eta_i (WSE
  continuity) instead of the previous depth-preserving form
  eta_b = eta_i + z_i - z_b (h_b = h_i). The old form incorrectly assigned
  positive depth to low-lying boundary cells on real DEMs, seeding spurious
  wetting that cascaded through the depth-update logic.
- Current-time interior values used in post-solve BC application:
  _apply_boundary_conditions_eta and _apply_boundary_conditions_vel now use
  self._eta and self._vel (time N+1) rather than self._eta_at_N and
  self._vel_at_N (time N), eliminating a one-timestep lag at every open boundary
  node that caused reflection and artificial storage near wetting fronts.
- -1 sentinel stripped from _open_boundary_links: links_at_node returns -1 for
  missing directions on corner and edge nodes; the previous np.unique call
  preserved this sentinel, causing a silent out-of-bounds write
  (self._h_at_links[-1] = ...) on every timestep.
- closed_boundary_nodes parameter: allows the user to designate specific boundary
  nodes as impermeable walls (zero flux, dry WSE). Nodes not listed as closed
  remain open (zero-gradient). Essential for complex-DEM simulations where only
  part of a grid edge is a real channel outlet.
- fixed_exit_nodes / exit_nodes_h_values / outlet_max_depth parameters: prescribe
  a Dirichlet fixed-stage outlet BC at specified right-edge nodes, with an
  optional outlet_max_depth ramp threshold that delays enforcement until local
  depth exceeds a minimum value. Required for sloped-channel simulations where
  the zero-gradient BC alone equilibrates to a flat (bathtub) water surface.

Examples
--------

This example demonstrates basic usage of the RiverFlowDynamics component to simulate
a simple channel flow:

>>> import numpy as np
>>> from landlab import RasterModelGrid
>>> from landlab.components import RiverFlowDynamics

Create a small grid for demonstration purposes:

>>> grid = RasterModelGrid((8, 6), xy_spacing=0.1)

Set up a sloped channel with elevated sides (slope of 0.01).

>>> z = grid.add_zeros("topographic__elevation", at="node")
>>> z += 0.005 - 0.01 * grid.x_of_node
>>> z[grid.y_of_node > 0.5] = 1.0
>>> z[grid.y_of_node < 0.2] = 1.0

Instantiating the Component. To check the names of the required inputs, use
the 'input_var_names' class property.

>>> RiverFlowDynamics.input_var_names
('surface_water__depth', 'surface_water__elevation',
'surface_water__velocity', 'topographic__elevation')

Initialize required fields:

>>> h = grid.add_zeros("surface_water__depth", at="node")
>>> vel = grid.add_zeros("surface_water__velocity", at="link")
>>> wse = grid.add_zeros("surface_water__elevation", at="node")
>>> wse += h + z

Set up inlet boundary conditions (left side of channel):
Water flows from left to right at a depth of 0.5 meters with a velocity of 0.45 m/s.

>>> fixed_entry_nodes = np.arange(12, 36, 6)
>>> fixed_entry_links = grid.links_at_node[fixed_entry_nodes][:, 0]
>>> entry_nodes_h_values = np.full(4, 0.5)
>>> entry_links_vel_values = np.full(4, 0.45)

Instantiate 'RiverFlowDynamics'

>>> rfd = RiverFlowDynamics(
...     grid,
...     dt=0.1,
...     mannings_n=0.012,
...     fixed_entry_nodes=fixed_entry_nodes,
...     fixed_entry_links=fixed_entry_links,
...     entry_nodes_h_values=entry_nodes_h_values,
...     entry_links_vel_values=entry_links_vel_values,
... )

Run the simulation for 100 timesteps (equivalent to 10 seconds).

>>> n_timesteps = 100
>>> for timestep in range(n_timesteps):
...     rfd.run_one_step()
...

Examine the flow depth at the center of the channel after 10 seconds.

>>> flow_depth = np.reshape(grid["node"]["surface_water__depth"], (8, 6))[3, :]
>>> np.round(flow_depth, 3)
array([0.5  , 0.5  , 0.5  , 0.501, 0.501, 0.502])

And the velocity at links along the center of the channel.

>>> linksAtCenter = grid.links_at_node[np.array(np.arange(24, 30))][:-1, 0]
>>> flow_velocity = grid["link"]["surface_water__velocity"][linksAtCenter]
>>> np.round(flow_velocity, 3)
array([0.45 , 0.457, 0.456, 0.451, 0.471])

"""

import warnings

import numpy as np
import scipy as sp

from landlab import Component


class RiverFlowDynamics(Component):
    """Simulate surface fluid flow based on Casulli and Cheng (1992).

    This Landlab component simulates surface fluid flow using the approximations of the
    2D shallow water equations developed by Casulli and Cheng in 1992. It calculates water
    depth and velocity across the raster grid, given a specific input discharge.

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    Monsalve et al., (2025). RiverFlowDynamics v1.0: A Landlab component for computing
    two-dimensional river flow dynamics. Journal of Open Source Software, 10(110), 7823,
    https://doi.org/10.21105/joss.07823

    **Additional References**

    Casulli, V., Cheng, R.T. (1992). "Semi-implicit finite difference methods for
    three-dimensional shallow water flow". International Journal for Numerical Methods
    in Fluids. 15: 629-648.
    https://doi.org/10.1002/fld.1650150602
    """

    _name = "RiverFlowDynamics"

    _unit_agnostic = False

    _info = {
        "surface_water__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of water on the surface",
        },
        "surface_water__velocity": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m/s",
            "mapping": "link",
            "doc": "Speed of water flow above the surface",
        },
        "surface_water__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Water surface elevation at time N",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "surface_water__x_velocity": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "Depth-averaged x-velocity",
        },
        "surface_water__y_velocity": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "Depth-averaged y-velocity",
        },
    }

    def __init__(
        self,
        grid,
        dt=0.01,
        eddy_viscosity=1e-4,
        mannings_n=0.012,
        threshold_depth=0.01,
        theta=0.5,
        fixed_entry_nodes=None,
        fixed_entry_links=None,
        entry_nodes_h_values=None,
        entry_links_vel_values=None,
        fixed_exit_nodes=None,
        exit_nodes_h_values=None,
        outlet_max_depth=None,
        closed_boundary_nodes=None,
        pcg_tolerance=1e-05,
        pcg_max_iterations=None,
        surface_water__elevation_at_N_1=0.0,
        surface_water__elevation_at_N_2=0.0,
        surface_water__velocity_at_N_1=0.0,
    ):
        """Simulate the vertical-averaged surface fluid flow.

        Simulate vertical-averaged surface fluid flow using the Casulli and Cheng (1992)
        approximations of the 2D shallow water equations. This Landlab component calculates
        water depth and velocity across the raster grid based on a given input discharge.

        Parameters
        ----------
        grid : RasterModelGrid
            A grid.
        dt : float, optional
            Time step in seconds. Must be positive. Default = 0.01 s.
        eddy_viscosity : float, optional
            Eddy viscosity coefficient. Default = 1e-4 m^2/s.
        mannings_n : float or array_like, optional
            Manning's roughness coefficient. Default = 0.012 s/m^(1/3).
            Use 0.0 for frictionless (inviscid) simulations such as the
            Thacker oscillating lake or circular dam-break benchmarks.
        threshold_depth : float, optional
            Threshold at which a cell is considered wet. Default = 0.01 m.
        theta : float, optional
            Degree of implicitness of the solution, in [0.5, 1.0].
            Default = 0.5 (centered in time). theta=1.0 is fully implicit.
        fixed_entry_nodes : array_like or None, optional
            Node IDs where flow enters the domain (Dirichlet BC).
        fixed_entry_links : array_like or None, optional
            Link IDs where flow enters the domain (Dirichlet BC).
        entry_nodes_h_values : array_like, optional
            Water depth at each fixed_entry_node.
        entry_links_vel_values : array_like, optional
            Water velocity at each fixed_entry_link.
        fixed_exit_nodes : array_like or None, optional
            Node IDs where a fixed-stage outlet BC is imposed (Dirichlet exit).
        exit_nodes_h_values : array_like, optional
            Target water depth [m] at each fixed_exit_node.
        outlet_max_depth : float or None, optional
            Ramp-up depth threshold [m]. The outlet BC is only enforced when
            the local water depth >= outlet_max_depth, preventing the outlet
            from pulling water from a dry domain at simulation start.
            Default None (hard Dirichlet outlet from t = 0).
        closed_boundary_nodes : array_like or None, optional
            Node IDs on the grid boundary that should be treated as closed walls
            (zero flux, dry WSE = bed elevation). All boundary nodes NOT listed
            here (and not in fixed_entry_nodes or fixed_exit_nodes) receive the
            default zero-gradient open BC. Use this parameter to mask off
            floodplain or non-outlet boundary segments on complex DEMs, where
            leaving the entire grid edge open causes spurious inundation.
            Example for a Kootenai-style setup where only the bottom-left corner
            of the left and bottom edges form the real outlet::

                closed = np.concatenate([
                    grid.nodes_at_left_edge[grid.y_of_node[grid.nodes_at_left_edge] > 539.5],
                    grid.nodes_at_bottom_edge[grid.x_of_node[grid.nodes_at_bottom_edge] > 10.0],
                ])
                rfd = RiverFlowDynamics(grid, ..., closed_boundary_nodes=closed)

        pcg_tolerance : float, optional
            PCG convergence tolerance. Default = 1e-05.
        pcg_max_iterations : int or None, optional
            Maximum PCG iterations. Default = None (scipy default).
        surface_water__elevation_at_N_1 : float or array_like, optional
            Water surface elevation at time N-1 [m]. Default = 0.0.
        surface_water__elevation_at_N_2 : float or array_like, optional
            Water surface elevation at time N-2 [m]. Default = 0.0.
        surface_water__velocity_at_N_1 : float or array_like, optional
            Water velocity at links at time N-1 [m/s]. Default = 0.0.
        """
        super().__init__(grid)

        # ── Input validation ──────────────────────────────────────────────────
        if dt <= 0:
            raise ValueError(
                f"dt must be positive, got {dt}. "
                "Use the CFL condition to estimate a stable timestep."
            )
        if not 0.5 <= theta <= 1.0:
            raise ValueError(
                f"theta must be in [0.5, 1.0], got {theta}. "
                "theta=0.5 gives a centered-in-time scheme; theta=1.0 is fully implicit."
            )
        if threshold_depth <= 0:
            raise ValueError(
                f"threshold_depth must be positive, got {threshold_depth}."
            )
        if np.ndim(mannings_n) == 0 and mannings_n < 0:
            raise ValueError(f"mannings_n must be non-negative, got {mannings_n}.")

        # ── Precompute O(1) raster link lookup tables ─────────────────────────
        # Must be called before any link queries (including __init__ topology setup)
        self._build_raster_link_tables()

        # ── User parameters ───────────────────────────────────────────────────
        self._dt = dt
        self._elapsed_time = 0.0  # total simulated time [s], updated each step
        self._eddy_viscosity = eddy_viscosity
        self._g = sp.constants.g
        self._mannings_n = mannings_n
        self._threshold_depth = threshold_depth
        self._theta = theta
        self._pcg_tolerance = pcg_tolerance
        self._pcg_max_iterations = pcg_max_iterations

        # ── Numerical robustness constants ────────────────────────────────────
        self._coord_tol = 10.0 * np.finfo(float).eps * max(grid.dx, grid.dy)
        self._velocity_tol = 1e-12
        self._time_tol = max(1e-12, 10.0 * np.finfo(float).eps)
        self._gradient_tol = 1e-14
        self._max_pathline_substeps = 100
        self._min_chezy_depth = 1e-8  # floor for Chezy depth to prevent singularity

        # ── Topography (virtual reference frame) ──────────────────────────────
        self._additional_z = 10  # virtual datum offset so eta stays positive
        self._max_elevation = self._grid.at_node["topographic__elevation"].max()
        self._z = (
            self._max_elevation
            + self._additional_z
            - self._grid.at_node["topographic__elevation"]
        )

        # ── Link length array (dx for horizontal links, dy for vertical links) ─
        # Used for correct spatial gradient calculation in G-faces and velocity update
        self._link_lengths = np.full(grid.number_of_links, float(grid.dy), dtype=float)
        self._link_lengths[grid.horizontal_links] = float(grid.dx)

        # ── Entry / exit boundary node and link lists ─────────────────────────
        self._fixed_entry_nodes = [] if fixed_entry_nodes is None else fixed_entry_nodes
        self._fixed_entry_links = [] if fixed_entry_links is None else fixed_entry_links
        self._entry_nodes_h_values = (
            [] if entry_nodes_h_values is None else entry_nodes_h_values
        )
        self._entry_links_vel_values = (
            [] if entry_links_vel_values is None else entry_links_vel_values
        )

        # ── Create grid fields if not already present ─────────────────────────
        if "surface_water__depth" not in grid.at_node:
            grid.add_zeros(
                "surface_water__depth",
                at="node",
                units=self._info["surface_water__depth"]["units"],
            )
        if "surface_water__velocity" not in grid.at_link:
            grid.add_zeros(
                "surface_water__velocity",
                at="link",
                units=self._info["surface_water__velocity"]["units"],
            )
        if "surface_water__x_velocity" not in grid.at_node:
            grid.add_zeros(
                "surface_water__x_velocity",
                at="node",
                units=self._info["surface_water__x_velocity"]["units"],
            )
        if "surface_water__y_velocity" not in grid.at_node:
            grid.add_zeros(
                "surface_water__y_velocity",
                at="node",
                units=self._info["surface_water__y_velocity"]["units"],
            )
        # note: if the user passed topographic__elevation directly into
        # add_field("surface_water__elevation", te) without copy=True, both fields
        # share the same underlying array. The [:] in-place write in _write_grid_fields
        # would then corrupt topographic__elevation. Break the alias here.
        if "surface_water__elevation" in grid.at_node:
            if (
                grid.at_node["surface_water__elevation"]
                is grid.at_node["topographic__elevation"]
            ):
                grid.at_node["surface_water__elevation"] = grid.at_node[
                    "surface_water__elevation"
                ].copy()

        if "surface_water__elevation" not in grid.at_node:
            grid.add_field(
                "surface_water__elevation",
                (
                    grid.at_node["surface_water__depth"]
                    + grid.at_node["topographic__elevation"]
                ),
                at="node",
                units=self._info["surface_water__elevation"]["units"],
            )

        # Pre-create N-1 output fields so _write_grid_fields can use in-place [:]
        if "surface_water__velocity_at_N-1" not in grid.at_link:
            grid.add_zeros("surface_water__velocity_at_N-1", at="link")
        if "surface_water__elevation_at_N-1" not in grid.at_node:
            grid.add_zeros("surface_water__elevation_at_N-1", at="node")

        # ── Previous time-level arrays (.copy() makes them writable) ──────────
        self._surface_water__elevation_at_N_1 = np.broadcast_to(
            np.asarray(surface_water__elevation_at_N_1).flat, grid.number_of_nodes
        ).copy()
        self._surface_water__elevation_at_N_2 = np.broadcast_to(
            np.asarray(surface_water__elevation_at_N_2).flat, grid.number_of_nodes
        ).copy()
        self._surface_water__velocity_at_N_1 = np.broadcast_to(
            np.asarray(surface_water__velocity_at_N_1).flat, grid.number_of_links
        ).copy()

        # ── Live views into grid field arrays ─────────────────────────────────
        self._h = grid.at_node["surface_water__depth"]
        self._vel = grid.at_link["surface_water__velocity"]
        self._vel_at_N_1 = self._surface_water__velocity_at_N_1
        self._eta = grid.at_node["surface_water__elevation"] - (
            self._max_elevation + self._additional_z
        )
        self._eta_at_N_1 = self._surface_water__elevation_at_N_1 - (
            self._max_elevation + self._additional_z
        )
        self._eta_at_N_2 = self._surface_water__elevation_at_N_2 - (
            self._max_elevation + self._additional_z
        )

        # ── Open boundary conditions ──────────────────────────────────────────
        for edge_nodes in [
            grid.nodes_at_left_edge,
            grid.nodes_at_right_edge,
            grid.nodes_at_bottom_edge,
            grid.nodes_at_top_edge,
        ]:
            grid.status_at_node[edge_nodes] = grid.BC_NODE_IS_FIXED_VALUE

        self._adjacent_nodes_at_corner_nodes = np.array(
            [
                [grid.nodes_at_top_edge[-2], grid.nodes_at_right_edge[-2]],  # TR
                [grid.nodes_at_top_edge[1], grid.nodes_at_left_edge[-2]],  # TL
                [grid.nodes_at_left_edge[1], grid.nodes_at_bottom_edge[1]],  # BL
                [grid.nodes_at_right_edge[1], grid.nodes_at_bottom_edge[-2]],  # BR
            ]
        )

        self._open_boundary_nodes = grid.boundary_nodes

        self._open_boundary_links = np.unique(
            grid.links_at_node[self._open_boundary_nodes]
        )
        self._open_boundary_links = self._open_boundary_links[
            self._open_boundary_links >= 0
        ]

        self._open_boundary_nodes = np.setdiff1d(
            self._open_boundary_nodes, self._fixed_entry_nodes
        )
        self._open_boundary_links = np.setdiff1d(
            self._open_boundary_links, self._fixed_entry_links
        )
        self._fixed_corner_nodes = np.setdiff1d(
            grid.corner_nodes, self._open_boundary_nodes
        )
        self._open_corner_nodes = np.setdiff1d(
            grid.corner_nodes, self._fixed_corner_nodes
        )
        self._open_boundary_nodes = np.setdiff1d(
            self._open_boundary_nodes, self._open_corner_nodes
        )

        self._fixed_nodes_exist = len(self._fixed_entry_nodes) > 0
        self._fixed_links_exist = len(self._fixed_entry_links) > 0

        # ── Exit boundary (fixed-stage outlet) ───────────────────────────────
        if fixed_exit_nodes is not None:
            self._fixed_exit_nodes = np.asarray(fixed_exit_nodes, dtype=int)
            if exit_nodes_h_values is None:
                raise ValueError(
                    "exit_nodes_h_values is required when fixed_exit_nodes is provided."
                )
            self._exit_nodes_h_values = np.asarray(exit_nodes_h_values, dtype=float)
        else:
            self._fixed_exit_nodes = np.array([], dtype=int)
            self._exit_nodes_h_values = np.array([], dtype=float)
        self._outlet_max_depth = (
            None if outlet_max_depth is None else float(outlet_max_depth)
        )
        self._fixed_exit_nodes_exist = len(self._fixed_exit_nodes) > 0

        # Remove exit nodes and their links from the open boundary set so the
        # zero-gradient BC does not overwrite the fixed-stage outlet.
        if self._fixed_exit_nodes_exist:
            self._open_boundary_nodes = np.setdiff1d(
                self._open_boundary_nodes, self._fixed_exit_nodes
            )
            exit_links = np.unique(grid.links_at_node[self._fixed_exit_nodes])
            exit_links = exit_links[exit_links >= 0]
            self._open_boundary_links = np.setdiff1d(
                self._open_boundary_links, exit_links
            )

            # Precompute the single interior neighbour for each exit node,
            # chosen according to which edge the node sits on.
            # Convention: adjacent_nodes_at_node indices are [E=0, N=1, W=2, S=3].
            # Left  edge → only horizontal (E) link matters → neighbour index 0
            # Right edge → only horizontal (W) link matters → neighbour index 2
            # Bottom edge → only vertical   (N) link matters → neighbour index 1
            # Top   edge → only vertical   (S) link matters → neighbour index 3
            x_min = grid.x_of_node.min()
            x_max = grid.x_of_node.max()
            y_min = grid.y_of_node.min()
            xn = grid.x_of_node[self._fixed_exit_nodes]
            yn = grid.y_of_node[self._fixed_exit_nodes]

            dir_idx = np.where(
                np.isclose(xn, x_min),
                0,  # left  → E
                np.where(
                    np.isclose(xn, x_max),
                    2,  # right → W
                    np.where(np.isclose(yn, y_min), 1, 3),  # bottom→ N
                ),
            )  # top   → S

            self._exit_interior_nbr = np.array(
                [
                    grid.adjacent_nodes_at_node[node, d]
                    for node, d in zip(self._fixed_exit_nodes, dir_idx)
                ],
                dtype=int,
            )

        # ── Closed boundary nodes (impermeable walls on selected edge segments)
        # These nodes are excluded from the open-BC set and enforced dry (h = 0,
        # WSE = bed elevation, zero velocity on adjacent links) every timestep.
        # This allows the user to restrict open boundaries to real outlet segments
        # on complex-DEM simulations, preventing spurious inundation of floodplain
        # boundary cells that have no physical connection to the channel exit.
        if closed_boundary_nodes is not None:
            self._closed_boundary_nodes = np.asarray(closed_boundary_nodes, dtype=int)
            # Verify all supplied nodes are actually on the grid boundary
            not_boundary = np.setdiff1d(
                self._closed_boundary_nodes, grid.boundary_nodes
            )
            if len(not_boundary) > 0:
                raise ValueError(
                    f"closed_boundary_nodes contains {len(not_boundary)} node(s) that "
                    "are not on the grid boundary. Only boundary nodes may be closed."
                )
            # Collect all links touching closed nodes; strip -1 sentinels
            closed_links_raw = np.unique(
                grid.links_at_node[self._closed_boundary_nodes]
            )
            self._closed_boundary_links = closed_links_raw[closed_links_raw >= 0]
        else:
            self._closed_boundary_nodes = np.array([], dtype=int)
            self._closed_boundary_links = np.array([], dtype=int)
        self._closed_nodes_exist = len(self._closed_boundary_nodes) > 0

        # Remove closed nodes and their links from the open boundary set so
        # the zero-gradient BC does not apply to wall segments.
        if self._closed_nodes_exist:
            self._open_boundary_nodes = np.setdiff1d(
                self._open_boundary_nodes, self._closed_boundary_nodes
            )
            self._open_boundary_links = np.setdiff1d(
                self._open_boundary_links, self._closed_boundary_links
            )

        # ── Apply Dirichlet inflow BCs to initial state ───────────────────────
        if self._fixed_nodes_exist:
            self._h[self._fixed_entry_nodes] = entry_nodes_h_values
            self._eta[self._fixed_entry_nodes] = (
                entry_nodes_h_values - self._z[self._fixed_entry_nodes]
            )
        if self._fixed_links_exist:
            self._vel[self._fixed_entry_links] = entry_links_vel_values

        # ── Apply initial dry state to closed boundary nodes ──────────────────
        if self._closed_nodes_exist:
            self._eta[self._closed_boundary_nodes] = -self._z[
                self._closed_boundary_nodes
            ]
            self._h[self._closed_boundary_nodes] = 0.0
            self._vel[self._closed_boundary_links] = 0.0

        # ── Link-averaged topography and initial state ────────────────────────
        self._z_at_links = grid.map_mean_of_link_nodes_to_link(self._z)
        self._h_at_links = grid.map_mean_of_link_nodes_to_link(self._h)
        self._eta_at_links = self._h_at_links - self._z_at_links

        self._h_at_N = self._h.copy()
        self._h_at_N_at_links = grid.map_mean_of_link_nodes_to_link(self._h_at_N)
        self._vel_at_N = self._vel.copy()
        self._eta_at_N = self._eta.copy()

        self._wet_nodes = self._h_at_N >= self._threshold_depth
        self._wet_links = self._h_at_N_at_links >= self._threshold_depth

        # ── Precomputed topology masks (replaces O(N^2) list comprehensions) ──
        self._is_active_horizontal = np.isin(grid.active_links, grid.horizontal_links)
        self._active_horizontal_links = grid.active_links[self._is_active_horizontal]
        self._is_active_vertical = ~self._is_active_horizontal
        self._active_vertical_links = grid.active_links[self._is_active_vertical]

        self._horizontal_is_active = np.isin(grid.horizontal_links, grid.active_links)
        self._vertical_is_active = np.isin(grid.vertical_links, grid.active_links)

        # ── Precomputed sparse matrix topology (fixed by grid, reused every step)
        self._core_adjacent_nodes = grid.adjacent_nodes_at_node[grid.core_nodes]
        self._core_adjacent_links = grid.links_at_node[grid.core_nodes]
        _nodes_location = np.append(
            self._core_adjacent_nodes,
            np.array([grid.core_nodes]).T,
            axis=1,
        )  # shape [N_core, 5]: E, N, W, S, Center
        self._nodes_location = _nodes_location
        self._is_core_mask = np.isin(_nodes_location, grid.core_nodes)
        self._is_boundary_mask = ~self._is_core_mask

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def elapsed_time(self):
        """Total simulated time elapsed [s].

        Accumulates ``dt`` on every call to :meth:`run_one_step`.
        Matches the ``elapsed_time`` API of ``RiverFlowDynamics_HLLC`` for
        easy coupling and comparison between the two components.
        """
        return self._elapsed_time

    @property
    def current_dt(self):
        """Timestep used by this component [s].

        Returns the fixed ``dt`` supplied at construction.  Provided for
        API symmetry with ``RiverFlowDynamics_HLLC``, which uses an
        adaptive CFL-based timestep.
        """
        return self._dt

    @property
    def wet_nodes(self):
        """Boolean array: True where water depth >= threshold_depth."""
        return self._wet_nodes

    @property
    def wet_links(self):
        """Boolean array: True where link-averaged depth >= threshold_depth."""
        return self._wet_links

    @property
    def water_depth(self):
        """Current water depth at nodes [m]."""
        return self._h

    @property
    def water_velocity(self):
        """Current water velocity at links [m/s]."""
        return self._vel

    # ── Raster-grid O(1) lookup tables ───────────────────────────────────────

    def _build_raster_link_tables(self):
        """Precompute fast lookup tables for RasterModelGrid link operations.

        These tables make neighbor queries and nearest-link searches O(1) using
        simple index arithmetic — no coordinate searching, no large temporary arrays.
        Called once during __init__ before any link queries.
        """
        nrows, ncols = self.grid.shape
        self._nrows = int(nrows)
        self._ncols = int(ncols)
        self._dx = float(self.grid.dx)
        self._dy = float(self.grid.dy)
        self._x0 = float(self.grid.x_of_node.min())
        self._y0 = float(self.grid.y_of_node.min())

        # 2-D link-ID arrays for direct index arithmetic
        self._hlink_id = self.grid.horizontal_links.reshape((nrows, ncols - 1))
        self._vlink_id = self.grid.vertical_links.reshape((nrows - 1, ncols))

        # Adjacent link tables: columns = [E, N, W, S], -1 = boundary sentinel
        self._adj_hlinks = np.full((self.grid.number_of_links, 4), -1, dtype=int)
        self._adj_vlinks = np.full((self.grid.number_of_links, 4), -1, dtype=int)

        # Horizontal-link adjacency
        h = self._hlink_id
        self._adj_hlinks[h[:, :-1].ravel(), 0] = h[:, 1:].ravel()  # E
        self._adj_hlinks[h[:, 1:].ravel(), 2] = h[:, :-1].ravel()  # W
        if nrows > 1:
            self._adj_hlinks[h[:-1, :].ravel(), 1] = h[1:, :].ravel()  # N
            self._adj_hlinks[h[1:, :].ravel(), 3] = h[:-1, :].ravel()  # S

        # Vertical-link adjacency
        v = self._vlink_id
        self._adj_vlinks[v[:, :-1].ravel(), 0] = v[:, 1:].ravel()  # E
        self._adj_vlinks[v[:, 1:].ravel(), 2] = v[:, :-1].ravel()  # W
        if (nrows - 1) > 1:
            self._adj_vlinks[v[:-1, :].ravel(), 1] = v[1:, :].ravel()  # N
            self._adj_vlinks[v[1:, :].ravel(), 3] = v[:-1, :].ravel()  # S

    def _is_on_link_x(self, x):
        """Boolean mask: x-coordinate aligns with raster link x-positions."""
        x = np.asarray(x, dtype=float)
        tol = 10.0 * np.finfo(float).eps * max(self._dx, self._dy)
        fx = ((x - self._x0) / self._dx) % 1.0
        return (
            (np.abs(fx - 0.0) < tol / self._dx)
            | (np.abs(fx - 0.5) < tol / self._dx)
            | (np.abs(fx - 1.0) < tol / self._dx)
        )

    def _is_on_link_y(self, y):
        """Boolean mask: y-coordinate aligns with raster link y-positions."""
        y = np.asarray(y, dtype=float)
        tol = 10.0 * np.finfo(float).eps * max(self._dx, self._dy)
        fy = ((y - self._y0) / self._dy) % 1.0
        return (
            (np.abs(fy - 0.0) < tol / self._dy)
            | (np.abs(fy - 0.5) < tol / self._dy)
            | (np.abs(fy - 1.0) < tol / self._dy)
        )

    def find_nearest_link(self, x_coordinates, y_coordinates, objective_links="all"):
        """Return nearest link IDs for given (x, y) coordinates — O(1) on RasterModelGrid.

        Parameters
        ----------
        x_coordinates, y_coordinates : array_like
            Coordinates of query points.
        objective_links : {'horizontal', 'vertical'}
            Which link set to search.

        Returns
        -------
        ndarray of int
            Link IDs nearest to each query point.
        """
        x = np.asarray(x_coordinates, dtype=float)
        y = np.asarray(y_coordinates, dtype=float)
        dx, dy = self._dx, self._dy
        x0, y0 = self._x0, self._y0

        if objective_links == "horizontal":
            r = np.rint((y - y0) / dy).astype(int)
            c = np.rint((x - (x0 + 0.5 * dx)) / dx).astype(int)
            r = np.clip(r, 0, self._nrows - 1)
            c = np.clip(c, 0, self._ncols - 2)
            return self._hlink_id[r, c].astype(int)

        if objective_links == "vertical":
            r = np.rint((y - (y0 + 0.5 * dy)) / dy).astype(int)
            c = np.rint((x - x0) / dx).astype(int)
            r = np.clip(r, 0, self._nrows - 2)
            c = np.clip(c, 0, self._ncols - 1)
            return self._vlink_id[r, c].astype(int)

        raise ValueError(
            "objective_links must be 'horizontal' or 'vertical' for fast lookup."
        )

    def find_adjacent_links_at_link(self, current_link, objective_links="horizontal"):
        """Return adjacent link IDs (E, N, W, S) for each link — O(1) table lookup.

        Parameters
        ----------
        current_link : array_like of int
            Link IDs to query.
        objective_links : {'horizontal', 'vertical'}
            Link orientation to search for neighbours.

        Returns
        -------
        ndarray of int, shape (n, 4)
            Adjacent link IDs in order [E, N, W, S]. -1 means no neighbour.
        """
        links = np.asarray(current_link, dtype=int)
        if objective_links == "horizontal":
            return self._adj_hlinks[links]
        if objective_links == "vertical":
            return self._adj_vlinks[links]
        raise ValueError("objective_links must be 'horizontal' or 'vertical'.")

    # ── Semi-Lagrangian path-line tracing ─────────────────────────────────────

    def path_line_tracing(self):
        """Semi-analytical path-line tracing (Pollock, 1988).

        Traces particle trajectories backwards in time from each link face to find
        the departure point used for semi-Lagrangian velocity interpolation.
        Includes coordinate clipping, substep cap, and Eulerian shock-capturing
        stabilisation for robust operation on irregular DEMs.
        """
        dx, dy = self.grid.dx, self.grid.dy

        sum_partial_times = np.zeros_like(self._u_vel_of_particle)
        remaining_time = self._dt - sum_partial_times
        keep_tracing = remaining_time > 0

        substep_count = 0
        while np.any(remaining_time > 0):
            substep_count += 1
            if substep_count > self._max_pathline_substeps:
                warnings.warn(
                    "path_line_tracing: reached max_pathline_substeps; "
                    "clipping remaining tracing time.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            # Update entry point from last exit point for active particles
            self._x_of_particle = np.where(
                keep_tracing, self._x_at_exit_point, self._x_of_particle
            )
            self._y_of_particle = np.where(
                keep_tracing, self._y_at_exit_point, self._y_of_particle
            )

            # Detect particles on link positions vs. cell interiors
            tempBx = self._is_on_link_x(self._x_of_particle)
            tempBy = self._is_on_link_y(self._y_of_particle)
            tempBxy = tempBx | tempBy

            # Surrounding links for interior particles (mode="clip" prevents exceptions)
            xy_stack1 = np.append(
                np.array([self._x_of_particle]), np.array([self._y_of_particle]), axis=0
            )
            nodes_interior = self._grid.find_nearest_node(xy_stack1, mode="clip")
            temp_links_interior = self._grid.links_at_node[nodes_interior]
            nodes_from_particle = nodes_interior

            # Surrounding links for link-position particles (offset by dx/10)
            off_x = np.where(
                self._u_vel_of_particle >= 0,
                np.array([self._x_of_particle]) - dx / 10,
                np.array([self._x_of_particle]) + dx / 10,
            )
            off_y = np.where(
                self._v_vel_of_particle >= 0,
                np.array([self._y_of_particle]) - dy / 10,
                np.array([self._y_of_particle]) + dy / 10,
            )
            xy_stack2 = np.append(off_x, off_y, axis=0)
            nodes_link = self._grid.find_nearest_node(xy_stack2, mode="clip")
            temp_links_link = self._grid.links_at_node[nodes_link]
            nodes_from_particle = np.where(tempBxy, nodes_link, nodes_from_particle)

            # Select link set based on particle position type
            tempBxy_tiled = np.tile(tempBxy, 4).reshape(4, len(tempBxy)).T
            links_at_particle = np.where(
                tempBxy_tiled, temp_links_link, temp_links_interior
            )

            # Assign face links based on velocity direction
            link_x2 = np.where(
                self._u_vel_of_particle >= 0,
                links_at_particle[:, 0],
                links_at_particle[:, 2],
            )
            link_x1 = np.where(
                self._u_vel_of_particle >= 0,
                links_at_particle[:, 2],
                links_at_particle[:, 0],
            )
            link_y2 = np.where(
                self._v_vel_of_particle >= 0,
                links_at_particle[:, 1],
                links_at_particle[:, 3],
            )
            link_y1 = np.where(
                self._v_vel_of_particle >= 0,
                links_at_particle[:, 3],
                links_at_particle[:, 1],
            )

            xn = self._grid.x_of_node[nodes_from_particle]
            yn = self._grid.y_of_node[nodes_from_particle]
            x_at_x2 = np.where(self._u_vel_of_particle >= 0, xn + dx / 2, xn - dx / 2)
            x_at_x1 = np.where(self._u_vel_of_particle >= 0, xn - dx / 2, xn + dx / 2)
            y_at_y2 = np.where(self._v_vel_of_particle >= 0, yn + dy / 2, yn - dy / 2)
            y_at_y1 = np.where(self._v_vel_of_particle >= 0, yn - dy / 2, yn + dy / 2)

            # Face velocities (fallback to opposite face when link index is -1)
            u_at_x2 = np.where(
                link_x2 >= 0, self._vel_at_N[link_x2], self._vel_at_N[link_x1]
            )
            u_at_x1 = np.where(
                link_x1 >= 0, self._vel_at_N[link_x1], self._vel_at_N[link_x2]
            )
            v_at_y2 = np.where(
                link_y2 >= 0, self._vel_at_N[link_y2], self._vel_at_N[link_y1]
            )
            v_at_y1 = np.where(
                link_y1 >= 0, self._vel_at_N[link_y1], self._vel_at_N[link_y2]
            )

            grad_x = (u_at_x2 - u_at_x1) / dx
            grad_y = (v_at_y2 - v_at_y1) / dy

            # Particle velocity at entry point (Eulerian stabilisation: zero tiny values)
            self._u_vel_of_particle = u_at_x2 - grad_x * (x_at_x2 - self._x_of_particle)
            self._v_vel_of_particle = v_at_y2 - grad_y * (y_at_y2 - self._y_of_particle)
            self._u_vel_of_particle = np.where(
                np.abs(self._u_vel_of_particle) < 1e-10, 0.0, self._u_vel_of_particle
            )
            self._v_vel_of_particle = np.where(
                np.abs(self._v_vel_of_particle) < 1e-10, 0.0, self._v_vel_of_particle
            )

            # ── TAUx: x-direction partial time ───────────────────────────────
            safe_ux = np.where(
                self._u_vel_of_particle == 0, 9999.0, self._u_vel_of_particle
            )
            safe_ux1 = np.where(u_at_x1 == 0, 9999.0, u_at_x1)
            safe_gx = np.where(grad_x == 0, 9999.0, grad_x)
            TAUx = (1.0 / safe_gx) * np.log(np.abs(safe_ux / safe_ux1))

            dist_x = np.abs((self._x_of_particle - x_at_x1) / safe_ux1)
            TAUx = np.where(grad_x == 0, dist_x, TAUx)
            ratio_x = self._u_vel_of_particle / safe_ux1
            TAUx = np.where(ratio_x == 1, dist_x, TAUx)
            TAUx = np.where(self._u_vel_of_particle == 0, remaining_time, TAUx)
            TAUx = np.where(u_at_x1 == 0, remaining_time, TAUx)
            TAUx = np.where(ratio_x < 0, remaining_time, TAUx)
            TAUx = np.clip(np.abs(TAUx), 0, self._dt)

            # ── TAUy: y-direction partial time ───────────────────────────────
            safe_vy = np.where(
                self._v_vel_of_particle == 0, 9999.0, self._v_vel_of_particle
            )
            safe_vy1 = np.where(v_at_y1 == 0, 9999.0, v_at_y1)
            safe_gy = np.where(grad_y == 0, 9999.0, grad_y)
            TAUy = (1.0 / safe_gy) * np.log(np.abs(safe_vy / safe_vy1))

            dist_y = np.abs((self._y_of_particle - y_at_y1) / safe_vy1)
            TAUy = np.where(grad_y == 0, dist_y, TAUy)
            ratio_y = self._v_vel_of_particle / safe_vy1
            TAUy = np.where(ratio_y == 1, dist_y, TAUy)
            TAUy = np.where(self._v_vel_of_particle == 0, remaining_time, TAUy)
            TAUy = np.where(v_at_y1 == 0, remaining_time, TAUy)
            TAUy = np.where(ratio_y < 0, remaining_time, TAUy)
            TAUy = np.clip(np.abs(TAUy), 0, self._dt)

            # Minimum partial time step
            TAU = np.minimum.reduce([TAUx, TAUy, remaining_time])

            # ── Exit points ───────────────────────────────────────────────────
            safe_gx2 = np.where(grad_x == 0, 9999.0, grad_x)
            safe_gy2 = np.where(grad_y == 0, 9999.0, grad_y)

            xe = x_at_x2 - (1.0 / safe_gx2) * (
                u_at_x2 - self._u_vel_of_particle / np.exp(grad_x * TAU)
            )
            ye = y_at_y2 - (1.0 / safe_gy2) * (
                v_at_y2 - self._v_vel_of_particle / np.exp(grad_y * TAU)
            )
            xe = np.where(grad_x == 0, self._x_of_particle - u_at_x2 * TAU, xe)
            ye = np.where(grad_y == 0, self._y_of_particle - v_at_y2 * TAU, ye)
            xe = np.where(self._u_vel_of_particle == 0, self._x_of_particle, xe)
            ye = np.where(self._v_vel_of_particle == 0, self._y_of_particle, ye)

            # Clamp exit coordinates to grid extent (prevents out-of-bounds nodes)
            x_min = float(self._grid.x_of_node.min())
            x_max = float(self._grid.x_of_node.max())
            y_min = float(self._grid.y_of_node.min())
            y_max = float(self._grid.y_of_node.max())
            xe = np.clip(xe, x_min, x_max)
            ye = np.clip(ye, y_min, y_max)

            self._x_at_exit_point = np.where(keep_tracing, xe, self._x_at_exit_point)
            self._y_at_exit_point = np.where(keep_tracing, ye, self._y_at_exit_point)

            # ── Update remaining time ─────────────────────────────────────────
            sum_partial_times = np.where(
                keep_tracing, sum_partial_times + TAU, self._dt
            )
            remaining_time = np.where(
                remaining_time == 0, 0.0, self._dt - sum_partial_times
            )

            # Stop particles that are stationary
            both_zero = (self._u_vel_of_particle == 0) & (self._v_vel_of_particle == 0)
            remaining_time = np.where(both_zero, 0.0, remaining_time)

            # Stop particles that have not moved
            remaining_time = np.where(
                np.abs(self._x_of_particle - self._x_at_exit_point) < self._coord_tol,
                0.0,
                remaining_time,
            )
            remaining_time = np.where(
                np.abs(self._y_of_particle - self._y_at_exit_point) < self._coord_tol,
                0.0,
                remaining_time,
            )

            # Stop particles that hit a domain edge
            at_left = np.isin(
                self._x_at_exit_point,
                self._grid.x_of_node[self.grid.nodes_at_left_edge],
            )
            at_right = np.isin(
                self._x_at_exit_point,
                self._grid.x_of_node[self.grid.nodes_at_right_edge],
            )
            at_top = np.isin(
                self._y_at_exit_point, self._grid.y_of_node[self.grid.nodes_at_top_edge]
            )
            at_bottom = np.isin(
                self._y_at_exit_point,
                self._grid.y_of_node[self.grid.nodes_at_bottom_edge],
            )
            at_edge = at_left | at_right | at_top | at_bottom
            remaining_time = np.where(at_edge, 0.0, remaining_time)

            keep_tracing = remaining_time > 0

    # ── Private computation methods ───────────────────────────────────────────

    def _compute_a_faces(self):
        """Compute Chezy friction coefficient and implicit A-face denominators.

        The A-face (implicit denominator) absorbs the bottom-friction term into
        the momentum equation following Casulli & Cheng (1992).
        Uses min_chezy_depth floor to prevent singularity in very thin films.

        For frictionless simulations (mannings_n == 0) the Chezy coefficient is
        infinite and the friction term in the A-face vanishes exactly, so
        a_links = h_at_N_at_links. This path is taken explicitly to avoid a
        divide-by-zero RuntimeWarning from the Chezy formula.
        """
        # Depth-split velocity components
        self._u_vel = self._vel_at_N[self.grid.horizontal_links]
        self._v_vel = self._vel_at_N[self.grid.vertical_links]

        # Cross-component velocity interpolation to face centres
        u_mean_at_nodes = self._grid.map_mean_of_horizontal_links_to_node(
            self._vel_at_N
        )
        self._u_vel_at_v_links = np.mean(
            u_mean_at_nodes[self._grid.nodes_at_link[self.grid.vertical_links]], axis=1
        )
        v_mean_at_nodes = self._grid.map_mean_of_vertical_links_to_node(self._vel_at_N)
        self._v_vel_at_u_links = np.mean(
            v_mean_at_nodes[self._grid.nodes_at_link[self.grid.horizontal_links]],
            axis=1,
        )

        # ── Frictionless shortcut (mannings_n == 0) ───────────────────────────
        # When n=0, Chezy C = h^(1/6)/n → ∞, so g*dt*|U|/C² → 0 and
        # a_links = h exactly. Skip the Chezy division to avoid a RuntimeWarning.
        if np.ndim(self._mannings_n) == 0 and self._mannings_n == 0.0:
            self._chezy_at_links = np.full(
                self.grid.number_of_links, np.inf, dtype=float
            )
            self._a_links = np.where(
                self._wet_links,
                self._h_at_N_at_links,
                1.0,
            )
            return

        # ── Frictional case ───────────────────────────────────────────────────
        # Chezy coefficient with depth floor (prevents singular friction)
        chezy_depth_links = np.maximum(self._h_at_N_at_links, self._min_chezy_depth)
        self._chezy_at_links = chezy_depth_links ** (1.0 / 6.0) / self._mannings_n

        # Use Chezy=1 on dry links to avoid division by zero
        chezy_safe = np.where(self._wet_links, self._chezy_at_links, 1.0)

        # A-face = h + g*dt * |U|^2 / C^2
        self._a_links = np.zeros_like(self._vel_at_N)
        self._a_links[self.grid.horizontal_links] = (
            self._h_at_N_at_links[self.grid.horizontal_links]
            + self._g
            * self._dt
            * np.hypot(
                self._vel_at_N[self.grid.horizontal_links],
                self._v_vel_at_u_links,
            )
            / chezy_safe[self.grid.horizontal_links] ** 2
        )
        self._a_links[self.grid.vertical_links] = (
            self._h_at_N_at_links[self.grid.vertical_links]
            + self._g
            * self._dt
            * np.hypot(
                self._vel_at_N[self.grid.vertical_links],
                self._u_vel_at_v_links,
            )
            / chezy_safe[self.grid.vertical_links] ** 2
        )
        # Set dry links to 1 to avoid downstream division by zero
        self._a_links = np.where(self._wet_links, self._a_links, 1.0)

    def _advect_u_velocity(self):
        """Semi-Lagrangian advection + viscosity for U (horizontal) velocity."""
        dx, dy = self.grid.dx, self.grid.dy

        # Particle start positions at active horizontal link centres
        self._x_of_particle = self._grid.xy_of_link[:, 0][self._active_horizontal_links]
        self._y_of_particle = self._grid.xy_of_link[:, 1][self._active_horizontal_links]
        self._u_vel_of_particle = self._u_vel[self._horizontal_is_active]
        self._v_vel_of_particle = self._v_vel_at_u_links[self._horizontal_is_active]
        self._x_at_exit_point = self._x_of_particle.copy()
        self._y_at_exit_point = self._y_of_particle.copy()

        self.path_line_tracing()

        # Semi-Lagrangian interpolation stencil for U
        self._UsL = np.zeros_like(self._u_vel)
        temp_v_at_h = np.zeros_like(self._vel_at_N)
        temp_v_at_h[self.grid.horizontal_links] = self._v_vel_at_u_links

        link_B2 = self.find_nearest_link(
            self._x_at_exit_point, self._y_at_exit_point, objective_links="horizontal"
        )
        adj_B2 = self.find_adjacent_links_at_link(link_B2, objective_links="horizontal")

        link_A2 = adj_B2[:, 1]  # North
        link_C2 = adj_B2[:, 3]  # South
        flip_v = temp_v_at_h[link_B2] < 0
        link_A2 = np.where(flip_v, adj_B2[:, 3], adj_B2[:, 1])
        link_C2 = np.where(flip_v, adj_B2[:, 1], adj_B2[:, 3])
        link_A2 = np.where(link_A2 >= 0, link_A2, link_B2)
        link_C2 = np.where(link_C2 >= 0, link_C2, link_B2)

        flip_u = self._vel_at_N[link_B2] < 0
        adj_A2 = self.find_adjacent_links_at_link(link_A2, objective_links="horizontal")
        adj_C2 = self.find_adjacent_links_at_link(link_C2, objective_links="horizontal")

        link_A1 = np.where(flip_u, adj_A2[:, 2], adj_A2[:, 0])
        link_A3 = np.where(flip_u, adj_A2[:, 0], adj_A2[:, 2])
        link_B1 = np.where(flip_u, adj_B2[:, 2], adj_B2[:, 0])
        link_B3 = np.where(flip_u, adj_B2[:, 0], adj_B2[:, 2])
        link_C1 = np.where(flip_u, adj_C2[:, 2], adj_C2[:, 0])
        link_C3 = np.where(flip_u, adj_C2[:, 0], adj_C2[:, 2])

        def _sv(lnk, fall):
            return np.where(lnk >= 0, self._vel_at_N[lnk], self._vel_at_N[fall])

        vel_A1 = _sv(link_A1, link_A2)
        vel_A2 = self._vel_at_N[link_A2]
        vel_A3 = _sv(link_A3, link_A2)
        vel_B1 = _sv(link_B1, link_B2)
        vel_B2 = self._vel_at_N[link_B2]
        vel_B3 = _sv(link_B3, link_B2)
        vel_C1 = _sv(link_C1, link_C2)
        vel_C2 = self._vel_at_N[link_C2]
        vel_C3 = _sv(link_C3, link_C2)

        x2 = self._grid.xy_of_link[link_B2][:, 0]
        x1 = np.where(flip_u, x2 - dx, x2 + dx)
        x3 = np.where(flip_u, x2 + dx, x2 - dx)
        yB = self._grid.xy_of_link[link_B2][:, 1]
        yA = np.where(flip_v, yB - dy, yB + dy)
        yC = np.where(flip_v, yB + dy, yB - dy)

        xp = self._x_at_exit_point
        W1x = (xp - x2) * (xp - x3) / ((x1 - x2) * (x1 - x3))
        W2x = (xp - x1) * (xp - x3) / ((x2 - x1) * (x2 - x3))
        W3x = (xp - x1) * (xp - x2) / ((x3 - x1) * (x3 - x2))
        A_r = W1x * vel_A1 + W2x * vel_A2 + W3x * vel_A3
        B_r = W1x * vel_B1 + W2x * vel_B2 + W3x * vel_B3
        C_r = W1x * vel_C1 + W2x * vel_C2 + W3x * vel_C3

        yp = self._y_at_exit_point
        W1y = (yp - yB) * (yp - yC) / ((yA - yB) * (yA - yC))
        W2y = (yp - yA) * (yp - yC) / ((yB - yA) * (yB - yC))
        W3y = (yp - yA) * (yp - yB) / ((yC - yA) * (yC - yB))
        self._UsL[self._horizontal_is_active] = W1y * A_r + W2y * B_r + W3y * C_r

        # Central-difference viscous diffusion for U
        self._Uvis = np.zeros_like(self._u_vel)
        self._Uvis[self._horizontal_is_active] = (
            self._eddy_viscosity * self._dt * (vel_B3 - 2 * vel_B2 + vel_B1) / dx**2
            + self._eddy_viscosity * self._dt * (vel_C2 - 2 * vel_B2 + vel_A2) / dy**2
        )

    def _advect_v_velocity(self):
        """Semi-Lagrangian advection + viscosity for V (vertical) velocity."""
        dx, dy = self.grid.dx, self.grid.dy

        # Particle start positions at active vertical link centres
        self._x_of_particle = self._grid.xy_of_link[:, 0][self._active_vertical_links]
        self._y_of_particle = self._grid.xy_of_link[:, 1][self._active_vertical_links]
        self._v_vel_of_particle = self._v_vel[self._vertical_is_active]
        self._u_vel_of_particle = self._u_vel_at_v_links[self._vertical_is_active]
        self._x_at_exit_point = self._x_of_particle.copy()
        self._y_at_exit_point = self._y_of_particle.copy()

        self.path_line_tracing()

        # Semi-Lagrangian interpolation stencil for V
        self._VsL = np.zeros_like(self._v_vel)
        temp_u_at_v = np.zeros_like(self._vel_at_N)
        temp_u_at_v[self.grid.vertical_links] = self._u_vel_at_v_links

        link_B2 = self.find_nearest_link(
            self._x_at_exit_point, self._y_at_exit_point, objective_links="vertical"
        )
        adj_B2 = self.find_adjacent_links_at_link(link_B2, objective_links="vertical")

        flip_v = self._vel_at_N[link_B2] < 0
        link_A2 = np.where(flip_v, adj_B2[:, 3], adj_B2[:, 1])
        link_C2 = np.where(flip_v, adj_B2[:, 1], adj_B2[:, 3])
        link_A2 = np.where(link_A2 >= 0, link_A2, link_B2)
        link_C2 = np.where(link_C2 >= 0, link_C2, link_B2)

        flip_u = temp_u_at_v[link_B2] < 0
        adj_A2 = self.find_adjacent_links_at_link(link_A2, objective_links="vertical")
        adj_C2 = self.find_adjacent_links_at_link(link_C2, objective_links="vertical")

        link_A1 = np.where(flip_u, adj_A2[:, 2], adj_A2[:, 0])
        link_A3 = np.where(flip_u, adj_A2[:, 0], adj_A2[:, 2])
        link_B1 = np.where(flip_u, adj_B2[:, 2], adj_B2[:, 0])
        link_B3 = np.where(flip_u, adj_B2[:, 0], adj_B2[:, 2])
        link_C1 = np.where(flip_u, adj_C2[:, 2], adj_C2[:, 0])
        link_C3 = np.where(flip_u, adj_C2[:, 0], adj_C2[:, 2])

        def _sv(lnk, fall):
            return np.where(lnk >= 0, self._vel_at_N[lnk], self._vel_at_N[fall])

        vel_A1 = _sv(link_A1, link_A2)
        vel_A2 = self._vel_at_N[link_A2]
        vel_A3 = _sv(link_A3, link_A2)
        vel_B1 = _sv(link_B1, link_B2)
        vel_B2 = self._vel_at_N[link_B2]
        vel_B3 = _sv(link_B3, link_B2)
        vel_C1 = _sv(link_C1, link_C2)
        vel_C2 = self._vel_at_N[link_C2]
        vel_C3 = _sv(link_C3, link_C2)

        x2 = self._grid.xy_of_link[link_B2][:, 0]
        x1 = np.where(flip_u, x2 - dx, x2 + dx)
        x3 = np.where(flip_u, x2 + dx, x2 - dx)
        yB = self._grid.xy_of_link[link_B2][:, 1]
        yA = np.where(flip_v, yB - dy, yB + dy)
        yC = np.where(flip_v, yB + dy, yB - dy)

        xp = self._x_at_exit_point
        W1x = (xp - x2) * (xp - x3) / ((x1 - x2) * (x1 - x3))
        W2x = (xp - x1) * (xp - x3) / ((x2 - x1) * (x2 - x3))
        W3x = (xp - x1) * (xp - x2) / ((x3 - x1) * (x3 - x2))
        A_r = W1x * vel_A1 + W2x * vel_A2 + W3x * vel_A3
        B_r = W1x * vel_B1 + W2x * vel_B2 + W3x * vel_B3
        C_r = W1x * vel_C1 + W2x * vel_C2 + W3x * vel_C3

        yp = self._y_at_exit_point
        W1y = (yp - yB) * (yp - yC) / ((yA - yB) * (yA - yC))
        W2y = (yp - yA) * (yp - yC) / ((yB - yA) * (yB - yC))
        W3y = (yp - yA) * (yp - yB) / ((yC - yA) * (yC - yB))
        self._VsL[self._vertical_is_active] = W1y * A_r + W2y * B_r + W3y * C_r

        # Central-difference viscous diffusion for V
        self._Vvis = np.zeros_like(self._v_vel)
        self._Vvis[self._vertical_is_active] = (
            self._eddy_viscosity * self._dt * (vel_B3 - 2 * vel_B2 + vel_B1) / dx**2
            + self._eddy_viscosity * self._dt * (vel_C2 - 2 * vel_B2 + vel_A2) / dy**2
        )

    def _compute_g_faces(self):
        """Compute explicit G-face momentum fluxes.

        Uses _link_lengths for correct dx vs. dy pressure gradient mapping
        on both horizontal and vertical links.
        """
        f_vel = np.zeros_like(self._vel_at_N)
        f_vel[self.grid.horizontal_links] = self._UsL + self._Uvis
        f_vel[self.grid.vertical_links] = self._VsL + self._Vvis

        # Pressure gradient using correct link-direction length
        eta_grad_N = self._grid.calc_diff_at_link(self._eta_at_N) / self._link_lengths
        self._g_links = (
            self._h_at_N_at_links * f_vel
            - self._h_at_N_at_links
            * (1.0 - self._theta)
            * self._g
            * self._dt
            * eta_grad_N
        )

        self._g_links = np.where(self._wet_links, self._g_links, 0.0)
        self._g_links = np.where(self._grid.status_at_link == 4, 0.0, self._g_links)

    def _solve_pressure_correction(self, dx, dy):
        """Build and solve the sparse pressure-correction (WSE) linear system.

        Assembles a COO sparse matrix for core nodes only (core-only solve),
        converts to CSR, and applies PCG with a Jacobi preconditioner.

        Returns
        -------
        pcg_solution : ndarray
            WSE correction at core nodes.
        pcg_info : int
            Scipy CG flag (0 = converged, >0 = max-iter reached, <0 = breakdown).
        """
        core = self.grid.core_nodes
        n_core = core.size

        adj_nodes = self._core_adjacent_nodes  # [N_core, 4]
        adj_links = self._core_adjacent_links  # [N_core, 4]

        # ── RHS ──────────────────────────────────────────────────────────────
        tmp_flux = self._h_at_N_at_links[adj_links] * self._vel_at_N[adj_links]
        eta_star = (
            self._eta_at_N[core]
            - (1.0 - self._theta) * self._dt / dx * (tmp_flux[:, 0] - tmp_flux[:, 2])
            - (1.0 - self._theta) * self._dt / dy * (tmp_flux[:, 1] - tmp_flux[:, 3])
        )
        tmp_g = (
            self._h_at_N_at_links[adj_links]
            * self._g_links[adj_links]
            / self._a_links[adj_links]
        )
        rhs = (
            eta_star
            - self._theta * self._dt / dx * (tmp_g[:, 0] - tmp_g[:, 2])
            - self._theta * self._dt / dy * (tmp_g[:, 1] - tmp_g[:, 3])
        )

        # ── LHS coefficients ─────────────────────────────────────────────────
        tmp_c = self._h_at_N_at_links[adj_links] ** 2 / self._a_links[adj_links]
        cE = -tmp_c[:, 0] * (self._g * self._theta * self._dt / dx) ** 2
        cN = -tmp_c[:, 1] * (self._g * self._theta * self._dt / dy) ** 2
        cW = -tmp_c[:, 2] * (self._g * self._theta * self._dt / dx) ** 2
        cS = -tmp_c[:, 3] * (self._g * self._theta * self._dt / dy) ** 2
        cC = (
            1.0
            + (tmp_c[:, 0] + tmp_c[:, 2]) * (self._g * self._theta * self._dt / dx) ** 2
            + (tmp_c[:, 1] + tmp_c[:, 3]) * (self._g * self._theta * self._dt / dy) ** 2
        )

        # Map core nodes to contiguous 0..n_core-1 indices
        core_index = np.full(self.grid.number_of_nodes, -1, dtype=int)
        core_index[core] = np.arange(n_core, dtype=int)

        row = np.arange(n_core, dtype=int)

        # Move known boundary-node contributions to RHS
        for j, cj in enumerate([cE, cN, cW, cS]):
            nbr = adj_nodes[:, j]
            is_bnd = core_index[nbr] < 0
            if np.any(is_bnd):
                rhs[is_bnd] -= cj[is_bnd] * self._eta_at_N[nbr[is_bnd]]

        # Assemble sparse COO matrix (core nodes only)
        cols_all = [core_index[adj_nodes[:, j]] for j in range(4)] + [row]
        coefs_all = [cE, cN, cW, cS, cC]

        rows_coo, cols_coo, data_coo = [], [], []
        for c_arr, d_arr in zip(cols_all, coefs_all):
            valid = c_arr >= 0
            rows_coo.append(row[valid])
            cols_coo.append(c_arr[valid])
            data_coo.append(d_arr[valid])

        LHS = sp.sparse.coo_matrix(
            (
                np.concatenate(data_coo),
                (np.concatenate(rows_coo), np.concatenate(cols_coo)),
            ),
            shape=(n_core, n_core),
        ).tocsr()

        # Jacobi preconditioner
        diag = LHS.diagonal()
        inv_diag = np.where(diag != 0.0, 1.0 / diag, 1.0)
        M = sp.sparse.linalg.LinearOperator(
            (n_core, n_core),
            matvec=lambda x: inv_diag * x,
            dtype=float,
        )

        pcg_solution, pcg_info = sp.sparse.linalg.cg(
            LHS,
            rhs,
            M=M,
            rtol=self._pcg_tolerance,
            maxiter=self._pcg_max_iterations,
            atol=0.0,
        )
        if pcg_info < 0:
            raise RuntimeError(
                f"PCG solver failed with illegal input or breakdown (info={pcg_info})."
            )
        return pcg_solution, pcg_info

    def _apply_boundary_conditions_eta(self, nodes_1_back):
        """Apply boundary conditions to water surface elevation (WSE).

        Open boundary nodes receive a zero-gradient (Neumann) condition on
        water surface elevation (stage): WSE_boundary = WSE_interior.  This is
        a true stage-gradient-free condition and is more robust than the
        depth-preserving form (h_b = h_i) used in earlier versions, which
        incorrectly assigned positive depth to low-lying boundary cells on real
        DEMs and seeded spurious wetting cascades.

        Interior values are taken at the current time level (self._eta, just
        solved by PCG) rather than the previous time level (self._eta_at_N),
        eliminating the one-timestep lag that caused reflection near wetting
        fronts.

        Dirichlet inflow nodes are set to their prescribed WSE values.
        Closed boundary nodes are enforced dry (WSE = bed elevation) so they
        act as impermeable walls.

        Parameters
        ----------
        nodes_1_back : ndarray of int
            One-cell-upstream interior nodes for each open boundary node.
        """
        # Dirichlet: prescribed inflow WSE
        if self._fixed_nodes_exist:
            self._eta[self._fixed_entry_nodes] = (
                self._entry_nodes_h_values - self._z[self._fixed_entry_nodes]
            )

        # Zero-gradient on stage (WSE): copy current-time interior WSE directly.
        self._eta[self._open_boundary_nodes] = self._eta[nodes_1_back]

        # Fixed-stage outlet BC (ramp-up: only activate when h >= outlet_max_depth)
        if self._fixed_exit_nodes_exist:
            eta_target = self._exit_nodes_h_values - self._z[self._fixed_exit_nodes]
            if self._outlet_max_depth is not None:
                # Use self._h for the depth check — eta at boundary nodes has
                # been zeroed by the pressure-solve reset before this call.
                h_local = self._h[self._fixed_exit_nodes]
                use_target = h_local >= self._outlet_max_depth

                # Where ramp is inactive: directional zero-gradient (stage) using
                # the single interior neighbour along the flow-relevant link only
                # (horizontal for left/right edges, vertical for top/bottom).
                # Uses current-time eta (lag fix) and preserves stage (not depth).
                nbr = self._exit_interior_nbr
                eta_zg = self._eta[nbr]
                self._eta[self._fixed_exit_nodes] = np.where(
                    use_target, eta_target, eta_zg
                )
            else:
                self._eta[self._fixed_exit_nodes] = eta_target

        # Closed boundary nodes: enforce dry WSE = bed elevation (h = 0).
        # These nodes act as impermeable walls; their eta is set last so it
        # cannot be overwritten by the open-BC branch above.
        if self._closed_nodes_exist:
            self._eta[self._closed_boundary_nodes] = -self._z[
                self._closed_boundary_nodes
            ]

        # Clip WSE below topographic surface
        self._eta = np.where(np.abs(self._eta) > np.abs(self._z), -self._z, self._eta)

        self._eta_at_links = self._grid.map_mean_of_link_nodes_to_link(self._eta)

        # Corner nodes: average of their two non-corner boundary neighbours
        self._eta[self.grid.corner_nodes] = np.mean(
            self._eta[self._adjacent_nodes_at_corner_nodes], axis=1
        )

    def _update_velocity(self):
        """Update water velocity from pressure gradient (momentum equation)."""
        wet_head = np.where(
            np.abs(self._eta[self._grid.node_at_link_head])
            <= np.abs(self._z[self._grid.node_at_link_head] - self._threshold_depth),
            1,
            0,
        )
        bed_above = np.where(
            np.abs(self._z[self._grid.node_at_link_head])
            > np.abs(self._eta[self._grid.node_at_link_tail]),
            1,
            0,
        )
        apply_mask = np.where((wet_head + bed_above) > 1, 1, 0)

        # Pressure gradient using direction-correct link lengths
        eta_grad = self._grid.calc_diff_at_link(self._eta) / self._link_lengths
        pressure_term = (
            self._theta
            * self._g
            * self._dt
            * eta_grad
            * self._h_at_N_at_links
            / self._a_links
        )
        self._vel = self._g_links / self._a_links - pressure_term * apply_mask

        # Zero velocity on dry links
        self._vel = np.where(self._wet_links, self._vel, 0.0)

        # Dirichlet inflow velocity
        if self._fixed_links_exist:
            self._vel[self._fixed_entry_links] = self._entry_links_vel_values

    def _apply_boundary_conditions_vel(self, nodes_1_back):
        """Apply Zero-Gradient open boundary conditions to velocity.

        Copies velocity from the one-cell-upstream interior link to each open
        boundary link — the Neumann equivalent for velocity.

        Uses the current-time velocity field (self._vel, just updated by
        _update_velocity) rather than the previous time level (self._vel_at_N),
        eliminating the one-timestep lag that caused reflection and artificial
        storage near outlet boundaries.

        Closed boundary links are zeroed to enforce the no-flux wall condition.

        Parameters
        ----------
        nodes_1_back : ndarray of int
            One-cell-upstream interior nodes for each open boundary node.
        """
        # Identify active open boundary links
        is_open_bnd = np.isin(self.grid.active_links, self._open_boundary_links)
        open_bnd_active_links = self.grid.active_links[is_open_bnd]

        # Find the upstream link for each open boundary link
        tiled_bnd = np.tile(open_bnd_active_links, (4, 1)).T
        surrounding = self._grid.links_at_node[nodes_1_back]
        pos_arr = np.tile([0, 1, 2, 3], (len(self._open_boundary_nodes), 1))
        pos = pos_arr[tiled_bnd == surrounding]

        # Reverse direction: find the link pointing back inward
        rev = np.where(pos == 0, 2, pos)
        rev = np.where(pos == 1, 3, rev)
        rev = np.where(pos == 2, 0, rev)
        rev = np.where(pos == 3, 1, rev)

        upstream_links = surrounding[[range(surrounding.shape[0])], rev][0]

        # Zero-Gradient: copy current-time velocity from upstream link.
        self._vel[open_bnd_active_links] = self._vel[upstream_links]

        # Closed boundary: enforce zero velocity on all links touching wall nodes.
        if self._closed_nodes_exist:
            self._vel[self._closed_boundary_links] = 0.0

    def _update_depth(self):
        """Derive water depth from WSE and update wet/dry status."""
        is_wet_node = np.where(
            np.abs(self._eta) <= np.abs(self._z - self._threshold_depth), 1, 0
        )
        h_head = (
            self._z_at_links + self._eta[self._grid.node_at_link_head]
        ) * is_wet_node[self._grid.node_at_link_head]
        h_tail = (
            self._z_at_links + self._eta[self._grid.node_at_link_tail]
        ) * is_wet_node[self._grid.node_at_link_tail]
        self._h_at_links = np.maximum.reduce([h_head, h_tail, np.zeros_like(h_head)])

        # Override at open boundary links
        self._h_at_links[self._open_boundary_links] = (
            self._z_at_links[self._open_boundary_links]
            + self._eta_at_links[self._open_boundary_links]
        )

        # Apply wet/dry threshold
        self._h_at_links = np.where(
            self._h_at_links < self._threshold_depth, 0.0, self._h_at_links
        )
        self._wet_links = self._h_at_links >= self._threshold_depth
        self._vel *= self._wet_links

        # Node depth: average of qualifying wet surrounding link depths
        surrounding = self._grid.links_at_node[self.grid.core_nodes]
        surr_wet = self._wet_links[surrounding]
        wse_above_bed = (
            np.abs(self._eta_at_links[surrounding])
            < np.abs(self._z[self.grid.core_nodes] - self._threshold_depth)[:, None]
        )
        n_qual = np.sum(surr_wet * wse_above_bed, axis=1)
        n_qual = np.where(n_qual == 0, -9999, n_qual)
        self._h[self.grid.core_nodes] = np.where(
            n_qual > 0,
            np.sum(self._h_at_links[surrounding] * surr_wet * wse_above_bed, axis=1)
            / n_qual,
            0.0,
        )

        # Boundary depth updates
        if self._fixed_nodes_exist:
            self._h[self._fixed_entry_nodes] = self._entry_nodes_h_values
        self._h[self._open_boundary_nodes] = (
            self._eta[self._open_boundary_nodes] + self._z[self._open_boundary_nodes]
        )
        # Fixed-stage outlet: derive h from the (possibly ramp-clamped) eta
        if self._fixed_exit_nodes_exist:
            self._h[self._fixed_exit_nodes] = (
                self._eta[self._fixed_exit_nodes] + self._z[self._fixed_exit_nodes]
            )
        # Closed boundary nodes: enforce h = 0 (dry wall)
        if self._closed_nodes_exist:
            self._h[self._closed_boundary_nodes] = 0.0

        self._h = np.where(self._h < self._threshold_depth, 0.0, self._h)
        self._h[self.grid.corner_nodes] = np.mean(
            self._h[self._adjacent_nodes_at_corner_nodes], axis=1
        )
        self._wet_nodes = self._h >= self._threshold_depth

    def _write_grid_fields(self):
        """Write current and N-1 state to the Landlab grid fields.

        Uses in-place [:] assignment to update existing arrays without
        triggering reallocation on every step.
        """
        self._grid.at_node["surface_water__depth"][:] = self._h
        self._grid.at_link["surface_water__velocity"][:] = self._vel
        self._grid.at_node["surface_water__elevation"][:] = self._eta + (
            self._max_elevation + self._additional_z
        )
        self._grid.at_link["surface_water__velocity_at_N-1"][:] = self._vel_at_N
        self._grid.at_node["surface_water__elevation_at_N-1"][:] = self._eta_at_N + (
            self._max_elevation + self._additional_z
        )

        # Project link-scalar velocity onto node-centred x/y components.
        # Horizontal links carry the u (x-direction) component;
        # vertical links carry the v (y-direction) component.
        # map_mean_of_*_links_to_node averages the signed link values of
        # all horizontal (or vertical) links incident on each node.
        # Zero velocity at dry nodes: without this, non-zero link velocities
        # adjacent to dry/thin cells produce spurious high-velocity artifacts
        # (u = vel at link, h ≈ 0 → apparent speed >> physical).
        x_vel = self._grid.map_mean_of_horizontal_links_to_node(self._vel)
        y_vel = self._grid.map_mean_of_vertical_links_to_node(self._vel)
        x_vel[~self._wet_nodes] = 0.0
        y_vel[~self._wet_nodes] = 0.0
        self._grid.at_node["surface_water__x_velocity"][:] = x_vel
        self._grid.at_node["surface_water__y_velocity"][:] = y_vel

    def _advance_time_history(self):
        """Advance the two-level time history by one step.

        Oldest level is updated first so each variable retains its correct
        time-lagged state (bug fix: original code overwrote N before copying to N-1).
        """
        self._eta_at_N_1 = self._eta_at_N.copy()  # N   -> N-1
        self._eta_at_N = self._eta.copy()  # cur -> N

        self._h_at_N = self._h.copy()
        self._h_at_N_at_links = self._h_at_links.copy()

        self._vel_at_N_1 = self._vel_at_N.copy()  # N   -> N-1
        self._vel_at_N = self._vel.copy()  # cur -> N

    def _find_upstream_nodes(self):
        """Return the one-cell-upstream interior nodes of each open boundary node.

        Used by both WSE and velocity Zero-Gradient boundary conditions.

        Returns
        -------
        nodes_1_back : ndarray of int
            Interior nodes immediately upstream of each open boundary node.
        """
        surrounding_1 = self._grid.active_adjacent_nodes_at_node[
            self._open_boundary_nodes
        ]
        return np.extract(surrounding_1 >= 0, surrounding_1)

    # ── Main public method ────────────────────────────────────────────────────

    def run_one_step(self):
        """Advance the shallow-water solution by one timestep dt.

        Computes water depth and velocity across the grid using the
        Casulli & Cheng (1992) semi-implicit, semi-Lagrangian scheme.
        The algorithm proceeds in ten stages:

        1.  Chezy friction + A-face implicit denominators
        2.  Semi-Lagrangian advection of U (horizontal) velocity
        3.  Semi-Lagrangian advection of V (vertical) velocity
        4.  Explicit G-face momentum fluxes
        5.  Sparse pressure-correction solve (PCG, core nodes only)
        6.  Zero-Gradient BC applied to WSE; closed walls enforced dry
        7.  Velocity update from pressure gradient
        8.  Zero-Gradient BC applied to velocity; closed walls zeroed
        9.  Depth and wet/dry update; closed walls enforced h = 0
        10. Write grid fields and advance time history
        """
        dx, dy = self.grid.dx, self.grid.dy

        # Upstream neighbour nodes for Zero-Gradient BCs (once per step)
        nodes_1_back = self._find_upstream_nodes()

        # ── Steps 1-4: advection and G-faces ─────────────────────────────────
        self._compute_a_faces()
        self._advect_u_velocity()
        self._advect_v_velocity()
        self._compute_g_faces()

        # ── Step 5: pressure-correction solve ────────────────────────────────
        pcg_solution, _pcg_flag = self._solve_pressure_correction(dx, dy)
        self._eta = np.zeros_like(self._eta_at_N)
        self._eta[self.grid.core_nodes] = pcg_solution

        # ── Steps 6-9: boundary conditions, velocity, depth ──────────────────
        self._apply_boundary_conditions_eta(nodes_1_back)
        self._update_velocity()
        self._apply_boundary_conditions_vel(nodes_1_back)
        self._update_depth()

        # ── Step 10: persist to grid and advance history ──────────────────────
        self._write_grid_fields()
        self._advance_time_history()
        self._elapsed_time += self._dt
