"""Simulate surface fluid flow based on Casulli and Cheng (1992).

This component implements a semi-implicit, semi-Lagrangian finite-volume approximation of
the depth-averaged shallow water equations originally proposed by Casulli and Cheng in 1992,
and subsequent related work.

Written by Sebastian Bernal and Angel Monsalve.

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
('surface_water__depth',
 'surface_water__elevation',
 'surface_water__velocity',
 'topographic__elevation')

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
array([0.5  , 0.5  , 0.5  , 0.501, 0.502, 0.502])

And the velocity at links along the center of the channel.

>>> linksAtCenter = grid.links_at_node[np.array(np.arange(24, 30))][:-1, 0]
>>> flow_velocity = grid["link"]["surface_water__velocity"][linksAtCenter]
>>> np.round(flow_velocity, 3)
array([0.45 , 0.457, 0.455, 0.452, 0.453])


Compared to the previous version published on JOSS, the current version
includes:
- Cached raster link adjacency tables (O(1) neighbor lookup)
- O(1) nearest-link mapping on raster grids
- Robust, fast raster-alignment checks in pathline tracing
- Sparse core-only pressure correction solve (PCG + Jacobi preconditioner)
"""

import numpy as np
import scipy as sp
import warnings

from landlab import Component


class RiverFlowDynamics(Component):
    """Simulate surface fluid flow based on Casulli and Cheng (1992).

    This Landlab component simulates surface fluid flow using the approximations of the
    2D shallow water equations developed by Casulli and Cheng in 1992. It calculates water
    depth and velocity across the raster grid, given a specific input discharge.

    References
    ----------
    Monsalve et al., (2025). RiverFlowDynamics v1.0: A Landlab component for computing two-
    dimensional river flow dynamics. Journal of Open Source Software, 10(110), 7823, https://
    doi.org/10.21105/joss.07823

    **Additional References**

    Casulli, V., Cheng, R.T. (1992). “Semi-implicit finite difference methods for
    three-dimensional shallow water flow”. International Journal for Numerical Methods
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
    }

    def __init__(
        self,
        grid,
        dt=0.01,  # Sets the time step (s)
        eddy_viscosity=1e-4,  # ddy viscosity coefficient
        mannings_n=0.012,  # Manning's n
        threshold_depth=0.01,  # Sets the wet/dry threshold
        theta=0.5,  # Degree of 'implicitness' of the solution
        fixed_entry_nodes=None,  # Node IDs where flow enters the domain
        fixed_entry_links=None,  # Link IDs where flow enters the domain
        entry_nodes_h_values=None,  # Water depth at nodes where flow enters the domain
        entry_links_vel_values=None,  # Water velocity at links where flow enters the domain
        pcg_tolerance=1e-05,  # Preconditioned Conjugate Gradient convergence tolerance
        pcg_max_iterations=None,  # Preconditioned Conjugate Gradient max iterations
        surface_water__elevation_at_N_1=0.0,  # Surf water elev at prev. time
        surface_water__elevation_at_N_2=0.0,  # Surf water elev at prev prev time
        surface_water__velocity_at_N_1=0.0,  # Speed of water at prev time
    ):
        """Simulate the vertical-averaged surface fluid flow

        Simulate vertical-averaged surface fluid flow using the Casulli and Cheng (1992)
        approximations of the 2D shallow water equations. This Landlab component calculates
        water depth and velocity across the raster grid based on a given input discharge.

        Parameters
        ----------
        grid : RasterModelGrid
            A grid.
        dt : float, optional
            Time step in seconds. If not provided, it is calculated from the CFL condition.
        eddy_viscosity : float, optional
            Eddy viscosity coefficient. Default = 1e-4 :math:`m^2 / s`
        mannings_n : float or array_like, optional
            Manning's roughness coefficient. Default = 0.012 :math:`s / m^1/3`
        threshold_depth : float, optional
            Threshold at which a cell is considered wet. Default = 0.01 m
        theta : float, optional
            Degree of 'implicitness' of the solution, ranging between 0.5 and 1.0.
            Default: 0.5. When set to 0.5, the approximation is centered in time;
            when set to 1.0, it is fully implicit.
        fixed_entry_nodes : array_like or None, optional
            Node IDs where flow enters the domain (Dirichlet boundary condition).
            If not provided, existing water in the domain is not renewed.
        fixed_entry_links : array_like or None, optional
            Link IDs where flow enters the domain (Dirichlet boundary condition).
            If not provided, existing water in the domain is not renewed.
        entry_nodes_h_values : array_like, optional
            Water depth values at nodes where flow enters the domain
            (Dirichlet boundary condition).
            If not provided, existing water in the domain is not renewed.
        entry_links_vel_values : array_like, optional
            Water velocity values at links where flow enters the domain
            (Dirichlet boundary condition).
            If not provided, existing water in the domain is not renewed.
        pcg_tolerance : float, optional
            Tolerance for convergence in the Preconditioned Conjugate Gradient
            method. Default: 1e-05.
        pcg_max_iterations : integer, optional
            Maximum number of iterations for the Preconditioned Conjugate Gradient
            method. Iteration stops after maxiter steps, even if the specified
            tolerance is not achieved. Default: None.
        surface_water__elevation_at_N_1: array_like of float, optional
            Water surface elevation at nodes at time N-1 [m].
        surface_water__elevation_at_N_2: array_like of float, optional
            Water surface elevation at nodes at time N-2 [m].
        surface_water__velocity_at_N_1: array_like of float, optional
            Speed of water flow at links above the surface at time N-1 [m/s].
        """
        super().__init__(grid)

        # Precompute raster-grid lookup tables used throughout the component
        self._build_raster_link_tables()

        # User inputs
        self._dt = dt
        self._eddy_viscosity = eddy_viscosity
        self._g = sp.constants.g
        self._mannings_n = mannings_n
        self._threshold_depth = threshold_depth
        self._theta = theta
        self._pcg_tolerance = pcg_tolerance
        self._pcg_max_iterations = pcg_max_iterations

        # Getting topography for further calculations
        self._additional_z = 10  # To set the virtual reference elevation (z=0)
        self._max_elevation = self._grid.at_node["topographic__elevation"].max()
        self._z = (
            self._max_elevation
            + self._additional_z
            - self._grid.at_node["topographic__elevation"]
        )

        self._fixed_entry_nodes = [] if fixed_entry_nodes is None else fixed_entry_nodes
        self._fixed_entry_links = [] if fixed_entry_links is None else fixed_entry_links
        self._entry_nodes_h_values = (
            [] if entry_nodes_h_values is None else entry_nodes_h_values
        )
        self._entry_links_vel_values = (
            [] if entry_links_vel_values is None else entry_links_vel_values
        )

        # Creating fields if they don't exist
        if "surface_water__depth" not in self.grid.at_node:
            grid.add_zeros(
                "surface_water__depth",
                at="node",
                units=self._info["surface_water__depth"]["units"],
            )

        if "surface_water__velocity" not in self.grid.at_link:
            grid.add_zeros(
                "surface_water__velocity",
                at="link",
                units=self._info["surface_water__velocity"]["units"],
            )

        if "surface_water__elevation" not in self.grid.at_node:
            grid.add_field(
                "surface_water__elevation",
                self.grid.at_node["surface_water__depth"] - self._z,
                at="node",
                units=self._info["surface_water__elevation"]["units"],
            )

        self._surface_water__elevation_at_N_1 = np.broadcast_to(
            np.asarray(surface_water__elevation_at_N_1).flat, grid.number_of_nodes
        )

        self._surface_water__elevation_at_N_2 = np.broadcast_to(
            np.asarray(surface_water__elevation_at_N_2).flat, grid.number_of_nodes
        )

        self._surface_water__velocity_at_N_1 = np.broadcast_to(
            np.asarray(surface_water__velocity_at_N_1).flat, grid.number_of_links
        )

        # Assigning a class variable to the fields
        self._h = self._grid.at_node["surface_water__depth"]
        self._vel = self._grid.at_link["surface_water__velocity"]
        self._vel_at_N_1 = self._surface_water__velocity_at_N_1
        self._eta = self._grid.at_node["surface_water__elevation"] - (
            self._max_elevation + self._additional_z
        )
        self._eta_at_N_1 = self._surface_water__elevation_at_N_1 - (
            self._max_elevation + self._additional_z
        )
        self._eta_at_N_2 = self._surface_water__elevation_at_N_2 - (
            self._max_elevation + self._additional_z
        )

        # Open boundary conditions
        # water can leave the domain at everywhere, only limited by topography
        self.grid.status_at_node[self.grid.nodes_at_left_edge] = (
            self._grid.BC_NODE_IS_FIXED_VALUE
        )
        self.grid.status_at_node[self.grid.nodes_at_right_edge] = (
            self._grid.BC_NODE_IS_FIXED_VALUE
        )
        self.grid.status_at_node[self.grid.nodes_at_bottom_edge] = (
            self._grid.BC_NODE_IS_FIXED_VALUE
        )
        self.grid.status_at_node[self.grid.nodes_at_top_edge] = (
            self._grid.BC_NODE_IS_FIXED_VALUE
        )

        self._adjacent_nodes_at_corner_nodes = np.array(
            [
                # Top right
                [self.grid.nodes_at_top_edge[-2], self.grid.nodes_at_right_edge[-2]],
                # Top left
                [self.grid.nodes_at_top_edge[1], self.grid.nodes_at_left_edge[-2]],
                # Bottom left
                [self.grid.nodes_at_left_edge[1], self.grid.nodes_at_bottom_edge[1]],
                # Bottom right
                [self.grid.nodes_at_right_edge[1], self.grid.nodes_at_bottom_edge[-2]],
            ]
        )

        # Updating open boundary nodes/links
        self._open_boundary_nodes = self._grid.boundary_nodes
        self._open_boundary_links = np.unique(
            self._grid.links_at_node[self._open_boundary_nodes]
        )

        self._open_boundary_nodes = np.setdiff1d(
            self._open_boundary_nodes, self._fixed_entry_nodes
        )
        self._open_boundary_links = np.setdiff1d(
            self._open_boundary_links, self._fixed_entry_links
        )

        self._fixed_corner_nodes = np.setdiff1d(
            self.grid.corner_nodes, self._open_boundary_nodes
        )
        self._open_corner_nodes = np.setdiff1d(
            self.grid.corner_nodes, self._fixed_corner_nodes
        )

        self._open_boundary_nodes = np.setdiff1d(
            self._open_boundary_nodes, self._open_corner_nodes
        )

        # Using fixed entry nodes/links only when they exist
        self._fixed_nodes_exist = len(self._fixed_entry_nodes) > 0
        self._fixed_links_exist = len(self._fixed_entry_links) > 0

        # Updating grid fixed values according to the user input
        if self._fixed_nodes_exist:
            self._h[self._fixed_entry_nodes] = entry_nodes_h_values
            self._eta[self._fixed_entry_nodes] = (
                entry_nodes_h_values - self._z[self._fixed_entry_nodes]
            )
        if self._fixed_links_exist:
            self._vel[self._fixed_entry_links] = entry_links_vel_values

        # Mapping node values at links
        self._z_at_links = self._grid.map_mean_of_link_nodes_to_link(self._z)
        self._h_at_links = self._grid.map_mean_of_link_nodes_to_link(self._h)
        self._eta_at_links = self._h_at_links - self._z_at_links

        # Passing values to the time step N
        self._h_at_N = self._h.copy()
        self._h_at_N_at_links = self._grid.map_mean_of_link_nodes_to_link(self._h_at_N)

        self._vel_at_N = self._vel.copy()
        self._eta_at_N = self._eta.copy()

        # Boolean for wet nodes/links
        self._wet_nodes = np.where(self._h_at_N >= self._threshold_depth, True, False)
        self._wet_links = np.where(
            self._h_at_N_at_links >= self._threshold_depth, True, False
        )

    # Defining some functions

    # -------------------------------------------------------------------------
    # Raster-grid lookup tables (speed-ups)
    # -------------------------------------------------------------------------
    def _build_raster_link_tables(self):
        """Precompute fast lookup tables for RasterModelGrid link operations.

        These tables make neighbor queries and nearest-link searches O(1) using
        simple index arithmetic (no coordinate searching / no large temporary arrays).
        """
        nrows, ncols = self.grid.shape
        self._nrows = int(nrows)
        self._ncols = int(ncols)
        self._dx = float(self.grid.dx)
        self._dy = float(self.grid.dy)
        self._x0 = float(self.grid.x_of_node.min())
        self._y0 = float(self.grid.y_of_node.min())

        # Reshaped link-id tables for arithmetic lookup
        self._hlink_id = self.grid.horizontal_links.reshape((nrows, ncols - 1))
        self._vlink_id = self.grid.vertical_links.reshape((nrows - 1, ncols))

        # Adjacent links (same orientation): columns are [E, N, W, S]
        self._adj_hlinks = np.full((self.grid.number_of_links, 4), -1, dtype=int)
        self._adj_vlinks = np.full((self.grid.number_of_links, 4), -1, dtype=int)

        # Horizontal-link adjacency
        h = self._hlink_id
        # East/West
        self._adj_hlinks[h[:, :-1].ravel(), 0] = h[:, 1:].ravel()  # E
        self._adj_hlinks[h[:, 1:].ravel(), 2] = h[:, :-1].ravel()  # W
        # North/South (row shifts)
        if nrows > 1:
            self._adj_hlinks[h[:-1, :].ravel(), 1] = h[1:, :].ravel()  # N
            self._adj_hlinks[h[1:, :].ravel(), 3] = h[:-1, :].ravel()  # S

        # Vertical-link adjacency
        v = self._vlink_id
        # East/West
        self._adj_vlinks[v[:, :-1].ravel(), 0] = v[:, 1:].ravel()  # E
        self._adj_vlinks[v[:, 1:].ravel(), 2] = v[:, :-1].ravel()  # W
        # North/South
        if (nrows - 1) > 1:
            self._adj_vlinks[v[:-1, :].ravel(), 1] = v[1:, :].ravel()  # N
            self._adj_vlinks[v[1:, :].ravel(), 3] = v[:-1, :].ravel()  # S

    def _is_on_link_x(self, x):
        """Return boolean mask: x aligns with raster link x-coordinates."""
        x = np.asarray(x, dtype=float)
        tol = 10.0 * np.finfo(float).eps * max(self._dx, self._dy)
        fx = ((x - self._x0) / self._dx) % 1.0
        return (
            (np.abs(fx - 0.0) < tol / self._dx)
            | (np.abs(fx - 0.5) < tol / self._dx)
            | (np.abs(fx - 1.0) < tol / self._dx)
        )

    def _is_on_link_y(self, y):
        """Return boolean mask: y aligns with raster link y-coordinates."""
        y = np.asarray(y, dtype=float)
        tol = 10.0 * np.finfo(float).eps * max(self._dx, self._dy)
        fy = ((y - self._y0) / self._dy) % 1.0
        return (
            (np.abs(fy - 0.0) < tol / self._dy)
            | (np.abs(fy - 0.5) < tol / self._dy)
            | (np.abs(fy - 1.0) < tol / self._dy)
        )

    def find_nearest_link(self, x_coordinates, y_coordinates, objective_links="all"):
        """Return the nearest link IDs for points on a RasterModelGrid.

        Notes
        -----
        This implementation is O(1) per point using index arithmetic.
        It is intended for objective_links in {"horizontal", "vertical"}.
        """
        x = np.asarray(x_coordinates, dtype=float)
        y = np.asarray(y_coordinates, dtype=float)

        dx, dy = self._dx, self._dy
        x0, y0 = self._x0, self._y0

        if objective_links == "horizontal":
            # Horizontal link centers at (x0 + (c + 0.5)*dx, y0 + r*dy)
            r = np.rint((y - y0) / dy).astype(int)
            c = np.rint((x - (x0 + 0.5 * dx)) / dx).astype(int)
            r = np.clip(r, 0, self._nrows - 1)
            c = np.clip(c, 0, self._ncols - 2)
            return self._hlink_id[r, c].astype(int)

        if objective_links == "vertical":
            # Vertical link centers at (x0 + c*dx, y0 + (r + 0.5)*dy)
            r = np.rint((y - (y0 + 0.5 * dy)) / dy).astype(int)
            c = np.rint((x - x0) / dx).astype(int)
            r = np.clip(r, 0, self._nrows - 2)
            c = np.clip(c, 0, self._ncols - 1)
            return self._vlink_id[r, c].astype(int)

        raise ValueError(
            "objective_links must be 'horizontal' or 'vertical' for fast lookup"
        )

    def find_adjacent_links_at_link(self, current_link, objective_links="horizontal"):
        """Adjacent links to each link ID (E, N, W, S).

        This is a fast table lookup replacement for the older coordinate-search version.
        """
        links = np.asarray(current_link, dtype=int)
        if objective_links == "horizontal":
            return self._adj_hlinks[links]
        if objective_links == "vertical":
            return self._adj_vlinks[links]
        raise ValueError("objective_links must be 'horizontal' or 'vertical'")

    def path_line_tracing(self):
        """ Path line tracing algorithm.

        This function implements the semi-analytical path line tracing method
        of Pollock (1988).
        """
        dx, dy = self.grid.dx, self.grid.dy
        tol_v = self._velocity_tol
        tol_g = self._gradient_tol
        tol_x = self._coord_tol

        sum_partial_times = np.zeros_like(self._u_vel_of_particle)
        remaining_time = np.maximum(self._dt - sum_partial_times, 0.0)
        keep_tracing = remaining_time > self._time_tol

        x_left = self._grid.x_of_node[self.grid.nodes_at_left_edge][0]
        x_right = self._grid.x_of_node[self.grid.nodes_at_right_edge][0]
        y_bottom = self._grid.y_of_node[self.grid.nodes_at_bottom_edge][0]
        y_top = self._grid.y_of_node[self.grid.nodes_at_top_edge][0]

        substep_count = 0
        while np.any(keep_tracing):
            substep_count += 1
            if substep_count > self._max_pathline_substeps:
                warnings.warn(
                    "Pathline tracing reached max_pathline_substeps; clipping remaining tracing time.",
                    RuntimeWarning,
                )
                break
            self._x_of_particle = np.where(
                keep_tracing, self._x_at_exit_point, self._x_of_particle
            )
            self._y_of_particle = np.where(
                keep_tracing, self._y_at_exit_point, self._y_of_particle
            )

            tempBx = self._is_on_link_x(self._x_of_particle)
            tempBy = self._is_on_link_y(self._y_of_particle)
            tempBxy = tempBx | tempBy

            tempCalc1 = np.append(
                np.array([self._x_of_particle]), np.array([self._y_of_particle]), axis=0
            )
            tempCalc2 = self._grid.find_nearest_node(tempCalc1, mode="clip")
            temp_links_from_node = self._grid.links_at_node[tempCalc2]
            nodes_from_particle = tempCalc2

            tempCalc1 = np.where(
                self._u_vel_of_particle >= 0,
                np.array([self._x_of_particle]) - dx / 10,
                np.array([self._x_of_particle]) + dx / 10,
            )
            tempCalc2 = np.where(
                self._v_vel_of_particle >= 0,
                np.array([self._y_of_particle]) - dy / 10,
                np.array([self._y_of_particle]) + dy / 10,
            )
            tempCalc3 = np.append(tempCalc1, tempCalc2, axis=0)
            tempCalc4 = self._grid.find_nearest_node(tempCalc3, mode="clip")
            temp_links_from_link = self._grid.links_at_node[tempCalc4]
            nodes_from_particle = np.where(tempBxy, tempCalc4, nodes_from_particle)

            tempBxy = np.tile(tempBxy, 4).reshape(4, len(tempBxy)).T
            links_at_particle = np.where(
                tempBxy, temp_links_from_link, temp_links_from_node
            )

            link_at_x2 = np.where(
                self._u_vel_of_particle >= 0,
                links_at_particle[:, 0],
                links_at_particle[:, 2],
            )
            link_at_x1 = np.where(
                self._u_vel_of_particle >= 0,
                links_at_particle[:, 2],
                links_at_particle[:, 0],
            )
            link_at_y2 = np.where(
                self._v_vel_of_particle >= 0,
                links_at_particle[:, 1],
                links_at_particle[:, 3],
            )
            link_at_y1 = np.where(
                self._v_vel_of_particle >= 0,
                links_at_particle[:, 3],
                links_at_particle[:, 1],
            )

            x_at_x2 = np.where(
                self._u_vel_of_particle >= 0,
                self._grid.x_of_node[nodes_from_particle] + dx / 2,
                self._grid.x_of_node[nodes_from_particle] - dx / 2,
            )
            x_at_x1 = np.where(
                self._u_vel_of_particle >= 0,
                self._grid.x_of_node[nodes_from_particle] - dx / 2,
                self._grid.x_of_node[nodes_from_particle] + dx / 2,
            )
            y_at_y2 = np.where(
                self._v_vel_of_particle >= 0,
                self._grid.y_of_node[nodes_from_particle] + dy / 2,
                self._grid.y_of_node[nodes_from_particle] - dy / 2,
            )
            y_at_y1 = np.where(
                self._v_vel_of_particle >= 0,
                self._grid.y_of_node[nodes_from_particle] - dy / 2,
                self._grid.y_of_node[nodes_from_particle] + dy / 2,
            )

            u_vel_at_x2 = np.where(
                link_at_x2 >= 0, self._vel_at_N[link_at_x2], self._vel_at_N[link_at_x1]
            )
            u_vel_at_x1 = np.where(
                link_at_x1 >= 0, self._vel_at_N[link_at_x1], self._vel_at_N[link_at_x2]
            )
            v_vel_at_y2 = np.where(
                link_at_y2 >= 0, self._vel_at_N[link_at_y2], self._vel_at_N[link_at_y1]
            )
            v_vel_at_y1 = np.where(
                link_at_y1 >= 0, self._vel_at_N[link_at_y1], self._vel_at_N[link_at_y2]
            )

            gradient_x_direction = (u_vel_at_x2 - u_vel_at_x1) / dx
            gradient_y_direction = (v_vel_at_y2 - v_vel_at_y1) / dy

            self._u_vel_of_particle = u_vel_at_x2 - gradient_x_direction * (
                x_at_x2 - self._x_of_particle
            )
            self._v_vel_of_particle = v_vel_at_y2 - gradient_y_direction * (
                y_at_y2 - self._y_of_particle
            )
            self._u_vel_of_particle = np.where(
                np.abs(self._u_vel_of_particle) < tol_v, 0.0, self._u_vel_of_particle
            )
            self._v_vel_of_particle = np.where(
                np.abs(self._v_vel_of_particle) < tol_v, 0.0, self._v_vel_of_particle
            )

            TAUx = remaining_time.copy()
            const_x = np.abs(gradient_x_direction) <= tol_g
            valid_x = (
                ~const_x
                & (np.abs(self._u_vel_of_particle) > tol_v)
                & (np.abs(u_vel_at_x1) > tol_v)
                & ((self._u_vel_of_particle * u_vel_at_x1) > 0.0)
            )
            if np.any(valid_x):
                ratio_x = np.abs(self._u_vel_of_particle[valid_x] / u_vel_at_x1[valid_x])
                TAUx[valid_x] = np.abs(np.log(ratio_x) / gradient_x_direction[valid_x])
            fallback_x = const_x & (np.abs(u_vel_at_x1) > tol_v)
            TAUx[fallback_x] = np.abs(
                (self._x_of_particle[fallback_x] - x_at_x1[fallback_x])
                / u_vel_at_x1[fallback_x]
            )
            TAUx = np.clip(TAUx, 0.0, remaining_time)

            TAUy = remaining_time.copy()
            const_y = np.abs(gradient_y_direction) <= tol_g
            valid_y = (
                ~const_y
                & (np.abs(self._v_vel_of_particle) > tol_v)
                & (np.abs(v_vel_at_y1) > tol_v)
                & ((self._v_vel_of_particle * v_vel_at_y1) > 0.0)
            )
            if np.any(valid_y):
                ratio_y = np.abs(self._v_vel_of_particle[valid_y] / v_vel_at_y1[valid_y])
                TAUy[valid_y] = np.abs(np.log(ratio_y) / gradient_y_direction[valid_y])
            fallback_y = const_y & (np.abs(v_vel_at_y1) > tol_v)
            TAUy[fallback_y] = np.abs(
                (self._y_of_particle[fallback_y] - y_at_y1[fallback_y])
                / v_vel_at_y1[fallback_y]
            )
            TAUy = np.clip(TAUy, 0.0, remaining_time)

            TAU = np.minimum(np.minimum(TAUx, TAUy), remaining_time)

            exp_x = np.exp(np.clip(gradient_x_direction * TAU, -700.0, 700.0))
            exp_y = np.exp(np.clip(gradient_y_direction * TAU, -700.0, 700.0))

            tempCalc3 = self._x_of_particle - u_vel_at_x2 * TAU
            tempCalc4 = self._y_of_particle - v_vel_at_y2 * TAU

            smooth_x = np.abs(gradient_x_direction) > tol_g
            smooth_y = np.abs(gradient_y_direction) > tol_g
            tempCalc3[smooth_x] = x_at_x2[smooth_x] - (
                u_vel_at_x2[smooth_x]
                - self._u_vel_of_particle[smooth_x] / exp_x[smooth_x]
            ) / gradient_x_direction[smooth_x]
            tempCalc4[smooth_y] = y_at_y2[smooth_y] - (
                v_vel_at_y2[smooth_y]
                - self._v_vel_of_particle[smooth_y] / exp_y[smooth_y]
            ) / gradient_y_direction[smooth_y]

            tempCalc3 = np.where(
                np.abs(self._u_vel_of_particle) <= tol_v, self._x_of_particle, tempCalc3
            )
            tempCalc4 = np.where(
                np.abs(self._v_vel_of_particle) <= tol_v, self._y_of_particle, tempCalc4
            )

            self._x_at_exit_point = np.where(
                keep_tracing, tempCalc3, self._x_at_exit_point
            )
            self._y_at_exit_point = np.where(
                keep_tracing, tempCalc4, self._y_at_exit_point
            )

            sum_partial_times = np.where(
                keep_tracing, sum_partial_times + TAU, self._dt
            )
            remaining_time = np.maximum(self._dt - sum_partial_times, 0.0)

            static_particle = (
                np.abs(self._x_of_particle - self._x_at_exit_point) < tol_x
            ) & (
                np.abs(self._y_of_particle - self._y_at_exit_point) < tol_x
            )
            remaining_time = np.where(static_particle, 0.0, remaining_time)

            hit_boundary = (
                (self._x_at_exit_point <= x_left + tol_x)
                | (self._x_at_exit_point >= x_right - tol_x)
                | (self._y_at_exit_point <= y_bottom + tol_x)
                | (self._y_at_exit_point >= y_top - tol_x)
            )
            remaining_time = np.where(hit_boundary, 0.0, remaining_time)
            keep_tracing = remaining_time > self._time_tol

    def run_one_step(self):
        """Calculate water depth and water velocity for a time period dt."""
        dx, dy = self.grid.dx, self.grid.dy

        # Getting velocity as U,V components
        self._u_vel = self._vel_at_N[self.grid.horizontal_links]
        self._v_vel = self._vel_at_N[self.grid.vertical_links]

        # Calculating Chezy coefficient
        self._chezy_at_nodes = self._h_at_N ** (1 / 6) / self._mannings_n
        self._chezy_at_links = self._h_at_N_at_links ** (1 / 6) / self._mannings_n

        # Computing V-velocity (vertical links) at U-velocity positions (horizontal links)
        tempCalc1 = self._grid.map_mean_of_horizontal_links_to_node(self._vel_at_N)
        self._u_vel_at_v_links = np.mean(
            tempCalc1[self._grid.nodes_at_link[self.grid.vertical_links]], axis=1
        )

        # Computing U-velocity (horizontal links) at V-velocity positions (vertical links)
        tempCalc1 = self._grid.map_mean_of_vertical_links_to_node(self._vel_at_N)
        self._v_vel_at_u_links = np.mean(
            tempCalc1[self._grid.nodes_at_link[self.grid.horizontal_links]], axis=1
        )

        # Setting A-faces
        self._a_links = np.zeros_like(self._vel_at_N)

        # Setting dry links equal to 1 to avoid divisions by zero
        tempCalc1 = np.where(self._wet_links, self._chezy_at_links, 1)

        # Computing A-faces
        self._a_links[self.grid.horizontal_links] = (
            self._h_at_N_at_links[self.grid.horizontal_links]
            + self._g
            * self._dt
            * (
                self._vel_at_N[self.grid.horizontal_links] ** 2
                + self._v_vel_at_u_links**2
            )
            ** (1 / 2)
            / tempCalc1[self.grid.horizontal_links]
        )
        self._a_links[self.grid.vertical_links] = (
            self._h_at_N_at_links[self.grid.vertical_links]
            + self._g
            * self._dt
            * (
                self._vel_at_N[self.grid.vertical_links] ** 2
                + self._u_vel_at_v_links**2
            )
            ** (1 / 2)
            / tempCalc1[self.grid.vertical_links]
        )

        # Using only wet-link values, and setting dry links equal to 1 to avoid
        # divisions by zero
        self._a_links = np.where(self._wet_links, self._a_links, 1)

        # Path line tracing
        # U-velocity, x-direction, horizontal links
        # Getting the initial particle location at each volume faces
        tempB1 = np.isin(self.grid.active_links, self.grid.horizontal_links)
        self._x_of_particle = self._grid.xy_of_link[:, 0][self.grid.active_links][
            tempB1
        ]
        self._y_of_particle = self._grid.xy_of_link[:, 1][self.grid.active_links][
            tempB1
        ]

        # Getting the initial particle velocity
        tempB2 = [i in self.grid.active_links for i in self.grid.horizontal_links]
        self._u_vel_of_particle = self._u_vel[tempB2]
        self._v_vel_of_particle = self._v_vel_at_u_links[tempB2]

        # Getting a first 'exit' point to begin the loop
        self._x_at_exit_point = self._x_of_particle
        self._y_at_exit_point = self._y_of_particle

        # Calculating path line backwards on time
        self.path_line_tracing()

        # Bicuatradic interpolation
        # U-velocity, x-direction, around (p,q) location
        self._UsL = np.zeros_like(self._u_vel)

        # Getting V-velocity at U-links
        temp_Vvel = np.zeros_like(self._vel_at_N)
        temp_Vvel[self.grid.horizontal_links] = self._v_vel_at_u_links

        # Getting links around the particle and defining downstream direction based on velocity
        nearest_link_to_particle = self.find_nearest_link(
            self._x_at_exit_point, self._y_at_exit_point, objective_links="horizontal"
        )
        adjacent_links_to_particle = self.find_adjacent_links_at_link(
            nearest_link_to_particle, objective_links="horizontal"
        )

        link_at_B2 = nearest_link_to_particle
        link_at_A2 = adjacent_links_to_particle[:, 1]  # 1: N, top
        link_at_C2 = adjacent_links_to_particle[:, 3]  # 3: S, bottom
        link_at_A2 = np.where(temp_Vvel[link_at_B2] >= 0, link_at_A2, link_at_C2)
        link_at_C2 = np.where(temp_Vvel[link_at_B2] >= 0, link_at_C2, link_at_A2)
        # We avoid "-1" links close to the boundary
        link_at_A2 = np.where(link_at_A2 >= 0, link_at_A2, link_at_B2)
        link_at_C2 = np.where(link_at_C2 >= 0, link_at_C2, link_at_B2)

        # Getting the surrounding links to every particle from closest link
        # 0: E, Right
        # 2: W, Left
        link_at_A1 = self.find_adjacent_links_at_link(
            link_at_A2, objective_links="horizontal"
        )[:, 0]
        link_at_A3 = self.find_adjacent_links_at_link(
            link_at_A2, objective_links="horizontal"
        )[:, 2]

        # Selecting downstream link based on velocity direction
        link_at_A1 = np.where(self._vel_at_N[link_at_B2] >= 0, link_at_A1, link_at_A3)
        link_at_A3 = np.where(self._vel_at_N[link_at_B2] >= 0, link_at_A3, link_at_A1)

        link_at_B1 = self.find_adjacent_links_at_link(
            link_at_B2, objective_links="horizontal"
        )[
            :, 0
        ]  # 0: E, Right
        link_at_B3 = self.find_adjacent_links_at_link(
            link_at_B2, objective_links="horizontal"
        )[
            :, 2
        ]  # 2: W, Left

        # Selecting downstream link based on velocity direction
        link_at_B1 = np.where(self._vel_at_N[link_at_B2] >= 0, link_at_B1, link_at_B3)
        link_at_B3 = np.where(self._vel_at_N[link_at_B2] >= 0, link_at_B3, link_at_B1)

        link_at_C1 = self.find_adjacent_links_at_link(
            link_at_C2, objective_links="horizontal"
        )[
            :, 0
        ]  # 0: E, Right
        link_at_C3 = self.find_adjacent_links_at_link(
            link_at_C2, objective_links="horizontal"
        )[
            :, 2
        ]  # 2: W, Left

        # Selecting downstream link based on velocity direction
        link_at_C1 = np.where(self._vel_at_N[link_at_B2] >= 0, link_at_C1, link_at_C3)
        link_at_C3 = np.where(self._vel_at_N[link_at_B2] >= 0, link_at_C3, link_at_C1)

        # Getting velocity around the particle
        vel_at_A1 = np.where(
            link_at_A1 >= 0, self._vel_at_N[link_at_A1], self._vel_at_N[link_at_A2]
        )
        vel_at_A2 = self._vel_at_N[link_at_A2]
        vel_at_A3 = np.where(
            link_at_A3 >= 0, self._vel_at_N[link_at_A3], self._vel_at_N[link_at_A2]
        )

        vel_at_B1 = np.where(
            link_at_B1 >= 0, self._vel_at_N[link_at_B1], self._vel_at_N[link_at_B2]
        )
        vel_at_B2 = self._vel_at_N[link_at_B2]
        vel_at_B3 = np.where(
            link_at_B3 >= 0, self._vel_at_N[link_at_B3], self._vel_at_N[link_at_B2]
        )

        vel_at_C1 = np.where(
            link_at_C1 >= 0, self._vel_at_N[link_at_C1], self._vel_at_N[link_at_C2]
        )
        vel_at_C2 = self._vel_at_N[link_at_C2]
        vel_at_C3 = np.where(
            link_at_C3 >= 0, self._vel_at_N[link_at_C3], self._vel_at_N[link_at_C2]
        )

        # Getting coordinates around the particle
        x_at_2 = self._grid.xy_of_link[link_at_B2][:, 0]
        x_at_1 = np.where(self._vel_at_N[link_at_B2] >= 0, x_at_2 + dx, x_at_2 - dx)
        x_at_3 = np.where(self._vel_at_N[link_at_B2] >= 0, x_at_2 - dx, x_at_2 + dx)

        y_at_B = self._grid.xy_of_link[link_at_B2][:, 1]
        y_at_A = np.where(temp_Vvel[link_at_B2] >= 0, y_at_B + dy, y_at_B - dy)
        y_at_C = np.where(temp_Vvel[link_at_B2] >= 0, y_at_B - dy, y_at_B + dy)

        # Calculating the weights W(i,j) for k around x-direction
        W1 = (
            (self._x_at_exit_point - x_at_2)
            * (self._x_at_exit_point - x_at_3)
            / ((x_at_1 - x_at_2) * (x_at_1 - x_at_3))
        )
        W2 = (
            (self._x_at_exit_point - x_at_1)
            * (self._x_at_exit_point - x_at_3)
            / ((x_at_2 - x_at_1) * (x_at_2 - x_at_3))
        )
        W3 = (
            (self._x_at_exit_point - x_at_1)
            * (self._x_at_exit_point - x_at_2)
            / ((x_at_3 - x_at_1) * (x_at_3 - x_at_2))
        )

        # Interpolation by row around 'x_at_exit_point'
        A = W1 * vel_at_A1 + W2 * vel_at_A2 + W3 * vel_at_A3
        B = W1 * vel_at_B1 + W2 * vel_at_B2 + W3 * vel_at_B3
        C = W1 * vel_at_C1 + W2 * vel_at_C2 + W3 * vel_at_C3

        # Calculating the weghts W(i,j) for l around y-direction
        W1 = (
            (self._y_at_exit_point - y_at_B)
            * (self._y_at_exit_point - y_at_C)
            / ((y_at_A - y_at_B) * (y_at_A - y_at_C))
        )
        W2 = (
            (self._y_at_exit_point - y_at_A)
            * (self._y_at_exit_point - y_at_C)
            / ((y_at_B - y_at_A) * (y_at_B - y_at_C))
        )
        W3 = (
            (self._y_at_exit_point - y_at_A)
            * (self._y_at_exit_point - y_at_B)
            / ((y_at_C - y_at_A) * (y_at_C - y_at_B))
        )

        # Calculating UsL by bicuadratic interpolation
        self._UsL[tempB2] = W1 * A + W2 * B + W3 * C

        # Computing viscous terms
        # U-located particles
        # Central difference scheme around x- and y- direction for U-located particles
        self._Uvis = np.zeros_like(self._u_vel)

        tempCalc1 = (
            self._eddy_viscosity
            * self._dt
            * (vel_at_B3 - 2 * vel_at_B2 + vel_at_B1)
            / (dx**2)
        )
        tempCalc2 = (
            self._eddy_viscosity
            * self._dt
            * (vel_at_C2 - 2 * vel_at_B2 + vel_at_A2)
            / (dy**2)
        )

        self._Uvis[tempB2] = tempCalc1 + tempCalc2

        # Path line tracing
        # V-velocity, y-direction, vertical links
        # Getting the initial particle location at each volume faces
        tempB1 = [j in self.grid.vertical_links for j in self.grid.active_links]
        self._x_of_particle = self._grid.xy_of_link[:, 0][self.grid.active_links][
            tempB1
        ]
        self._y_of_particle = self._grid.xy_of_link[:, 1][self.grid.active_links][
            tempB1
        ]

        # Getting the initial particle velocity
        tempB2 = [j in self.grid.active_links for j in self.grid.vertical_links]
        self._v_vel_of_particle = self._v_vel[tempB2]
        self._u_vel_of_particle = self._u_vel_at_v_links[tempB2]

        # Getting a first 'exit' point to begin the loop
        self._x_at_exit_point = self._x_of_particle
        self._y_at_exit_point = self._y_of_particle

        # Calculating path line backwards on time
        self.path_line_tracing()

        # Bicuatradic interpolation
        # V-velocity, y-direction, around (p,q) location
        self._VsL = np.zeros_like(self._v_vel)

        # Getting V-velocity at U-links
        temp_Uvel = np.zeros_like(self._vel_at_N)
        temp_Uvel[self.grid.vertical_links] = self._u_vel_at_v_links

        # Getting links around the particle and defining downstream direction based on velocity
        nearest_link_to_particle = self.find_nearest_link(
            self._x_at_exit_point, self._y_at_exit_point, objective_links="vertical"
        )
        adjacent_links_to_particle = self.find_adjacent_links_at_link(
            nearest_link_to_particle, objective_links="vertical"
        )

        link_at_B2 = nearest_link_to_particle
        link_at_A2 = adjacent_links_to_particle[:, 1]  # 1: N, top
        link_at_C2 = adjacent_links_to_particle[:, 3]  # 3: S, bottom
        link_at_A2 = np.where(self._vel_at_N[link_at_B2] >= 0, link_at_A2, link_at_C2)
        link_at_C2 = np.where(self._vel_at_N[link_at_B2] >= 0, link_at_C2, link_at_A2)
        link_at_A2 = np.where(link_at_A2 >= 0, link_at_A2, link_at_B2)
        link_at_C2 = np.where(link_at_C2 >= 0, link_at_C2, link_at_B2)
        # We avoid "-1" links close to the boundary

        # Getting the surrounding links to every particle from closest link
        link_at_A1 = self.find_adjacent_links_at_link(
            link_at_A2, objective_links="vertical"
        )[
            :, 0
        ]  # 0: E, Left
        link_at_A3 = self.find_adjacent_links_at_link(
            link_at_A2, objective_links="vertical"
        )[
            :, 2
        ]  # 2: W, Right

        # Selecting downstream link based on velocity direction
        link_at_A1 = np.where(temp_Uvel[link_at_B2] >= 0, link_at_A1, link_at_A3)
        link_at_A3 = np.where(temp_Uvel[link_at_B2] >= 0, link_at_A3, link_at_A1)

        link_at_B1 = self.find_adjacent_links_at_link(
            link_at_B2, objective_links="vertical"
        )[
            :, 0
        ]  # 0: E, Left
        link_at_B3 = self.find_adjacent_links_at_link(
            link_at_B2, objective_links="vertical"
        )[
            :, 2
        ]  # 2: W, Right

        # Selecting downstream link based on velocity direction
        link_at_B1 = np.where(temp_Uvel[link_at_B2] >= 0, link_at_B1, link_at_B3)
        link_at_B3 = np.where(temp_Uvel[link_at_B2] >= 0, link_at_B3, link_at_B1)

        link_at_C1 = self.find_adjacent_links_at_link(
            link_at_C2, objective_links="vertical"
        )[
            :, 0
        ]  # 0: E, Left
        link_at_C3 = self.find_adjacent_links_at_link(
            link_at_C2, objective_links="vertical"
        )[
            :, 2
        ]  # 2: W, Right

        # Selecting downstream link based on velocity direction
        link_at_C1 = np.where(temp_Uvel[link_at_B2] >= 0, link_at_C1, link_at_C3)
        link_at_C3 = np.where(temp_Uvel[link_at_B2] >= 0, link_at_C3, link_at_C1)

        # Getting velocity around the particle
        vel_at_A1 = np.where(
            link_at_A1 >= 0, self._vel_at_N[link_at_A1], self._vel_at_N[link_at_A2]
        )
        vel_at_A2 = self._vel_at_N[link_at_A2]
        vel_at_A3 = np.where(
            link_at_A3 >= 0, self._vel_at_N[link_at_A3], self._vel_at_N[link_at_A2]
        )

        vel_at_B1 = np.where(
            link_at_B1 >= 0, self._vel_at_N[link_at_B1], self._vel_at_N[link_at_B2]
        )
        vel_at_B2 = self._vel_at_N[link_at_B2]
        vel_at_B3 = np.where(
            link_at_B3 >= 0, self._vel_at_N[link_at_B3], self._vel_at_N[link_at_B2]
        )

        vel_at_C1 = np.where(
            link_at_C1 >= 0, self._vel_at_N[link_at_C1], self._vel_at_N[link_at_C2]
        )
        vel_at_C2 = self._vel_at_N[link_at_C2]
        vel_at_C3 = np.where(
            link_at_C3 >= 0, self._vel_at_N[link_at_C3], self._vel_at_N[link_at_C2]
        )

        # Getting coordinates around the particle
        x_at_2 = self._grid.xy_of_link[link_at_B2][:, 0]
        x_at_1 = np.where(temp_Uvel[link_at_B2] >= 0, x_at_2 + dx, x_at_2 - dx)
        x_at_3 = np.where(temp_Uvel[link_at_B2] >= 0, x_at_2 - dx, x_at_2 + dx)

        y_at_B = self._grid.xy_of_link[link_at_B2][:, 1]
        y_at_A = np.where(self._vel_at_N[link_at_B2] >= 0, y_at_B + dy, y_at_B - dy)
        y_at_C = np.where(self._vel_at_N[link_at_B2] >= 0, y_at_B - dy, y_at_B + dy)

        # Calculating the weights W(i,j) for k around x-direction
        W1 = (
            (self._x_at_exit_point - x_at_2)
            * (self._x_at_exit_point - x_at_3)
            / ((x_at_1 - x_at_2) * (x_at_1 - x_at_3))
        )
        W2 = (
            (self._x_at_exit_point - x_at_1)
            * (self._x_at_exit_point - x_at_3)
            / ((x_at_2 - x_at_1) * (x_at_2 - x_at_3))
        )
        W3 = (
            (self._x_at_exit_point - x_at_1)
            * (self._x_at_exit_point - x_at_2)
            / ((x_at_3 - x_at_1) * (x_at_3 - x_at_2))
        )

        # Interpolation by row around 'y_at_exit_point'
        A = W1 * vel_at_A1 + W2 * vel_at_A2 + W3 * vel_at_A3
        B = W1 * vel_at_B1 + W2 * vel_at_B2 + W3 * vel_at_B3
        C = W1 * vel_at_C1 + W2 * vel_at_C2 + W3 * vel_at_C3

        # Calculating the weghts W(i,j) for l around y-direction
        W1 = (
            (self._y_at_exit_point - y_at_B)
            * (self._y_at_exit_point - y_at_C)
            / ((y_at_A - y_at_B) * (y_at_A - y_at_C))
        )
        W2 = (
            (self._y_at_exit_point - y_at_A)
            * (self._y_at_exit_point - y_at_C)
            / ((y_at_B - y_at_A) * (y_at_B - y_at_C))
        )
        W3 = (
            (self._y_at_exit_point - y_at_A)
            * (self._y_at_exit_point - y_at_B)
            / ((y_at_C - y_at_A) * (y_at_C - y_at_B))
        )

        # Calculating VsL by bicuadratic interpolation
        self._VsL[tempB2] = W1 * A + W2 * B + W3 * C

        # Computing viscous terms
        # V-located particles
        # Central difference scheme around x- and y- direction for V-located particles
        self._Vvis = np.zeros_like(self._v_vel)

        tempCalc1 = (
            self._eddy_viscosity
            * self._dt
            * (vel_at_B3 - 2 * vel_at_B2 + vel_at_B1)
            / (dx**2)
        )
        tempCalc2 = (
            self._eddy_viscosity
            * self._dt
            * (vel_at_C2 - 2 * vel_at_B2 + vel_at_A2)
            / (dy**2)
        )

        self._Vvis[tempB2] = tempCalc1 + tempCalc2

        # Computing advective terms (FU, FV)
        self._f_vel = np.zeros_like(self._vel_at_N)

        # Adding semi-lagrangian and viscous terms
        tempCalc1 = self._UsL + self._Uvis
        tempCalc2 = self._VsL + self._Vvis

        # Including the results according with links directions
        self._f_vel[self.grid.horizontal_links] = tempCalc1
        self._f_vel[self.grid.vertical_links] = tempCalc2

        # Setting G-faces
        self._g_links = np.zeros_like(self._vel_at_N)

        # Computing G-faces
        self._g_links = self._h_at_N_at_links * self._f_vel - self._h_at_N_at_links * (
            1 - self._theta
        ) * self._g * self._dt / dx * self._grid.calc_diff_at_link(self._eta_at_N)

        # Using only wet-link values, and setting dry links equal to 0 to avoid
        # using wrong values
        self._g_links = np.where(self._wet_links, self._g_links, 0)
        self._g_links = np.where(
            self._grid.status_at_link == 4, 0, self._g_links
        )  # link is active (0), fixed (2), inactive (4)

        # Solving semi-implicit scheme with PCG method
        # Building the system of equations 'A*x=b'
        # Full 'A' matrix with all nodes on it

        # Solving semi-implicit scheme with PCG method
        # Building the sparse system of equations 'LHS * eta_core = RHS'
        core = self.grid.core_nodes
        n_core = core.size

        # Surrounding locations for core nodes (E, N, W, S)
        adjacent_nodes = self._grid.adjacent_nodes_at_node[core]  # shape (n_core, 4)
        adjacent_links = self._grid.links_at_node[core]  # shape (n_core, 4)

        # ---------------------------------------------------------------------
        # Build RHS (core nodes only)
        # ---------------------------------------------------------------------
        # Flux divergence term (explicit part)
        tmp_flux = (
            self._h_at_N_at_links[adjacent_links] * self._vel_at_N[adjacent_links]
        )
        eta_star = (
            self._eta_at_N[core]
            - (1 - self._theta) * self._dt / dx * (tmp_flux[:, 0] - tmp_flux[:, 2])
            - (1 - self._theta) * self._dt / dy * (tmp_flux[:, 1] - tmp_flux[:, 3])
        )

        # Gravity/friction term
        tmp_g = (
            self._h_at_N_at_links[adjacent_links]
            * self._g_links[adjacent_links]
            / self._a_links[adjacent_links]
        )
        rhs = (
            eta_star
            - self._theta * self._dt / dx * (tmp_g[:, 0] - tmp_g[:, 2])
            - self._theta * self._dt / dy * (tmp_g[:, 1] - tmp_g[:, 3])
        )

        # ---------------------------------------------------------------------
        # Build LHS (5-point stencil) as a sparse CSR matrix on core nodes
        # ---------------------------------------------------------------------
        tmp_c = (
            self._h_at_N_at_links[adjacent_links] ** 2 / self._a_links[adjacent_links]
        )
        cE = -tmp_c[:, 0] * (self._g * self._theta * self._dt / dx) ** 2
        cN = -tmp_c[:, 1] * (self._g * self._theta * self._dt / dy) ** 2
        cW = -tmp_c[:, 2] * (self._g * self._theta * self._dt / dx) ** 2
        cS = -tmp_c[:, 3] * (self._g * self._theta * self._dt / dy) ** 2
        cC = (
            1
            + (tmp_c[:, 0] + tmp_c[:, 2]) * (self._g * self._theta * self._dt / dx) ** 2
            + (tmp_c[:, 1] + tmp_c[:, 3]) * (self._g * self._theta * self._dt / dy) ** 2
        )

        # Map global node ids -> [0..n_core-1] indices; non-core = -1
        core_index = np.full(self.grid.number_of_nodes, -1, dtype=int)
        core_index[core] = np.arange(n_core, dtype=int)

        # Prepare sparse triplets
        row = np.arange(n_core, dtype=int)

        rows = [row, row, row, row, row]
        cols = [
            core_index[adjacent_nodes[:, 0]],
            core_index[adjacent_nodes[:, 1]],
            core_index[adjacent_nodes[:, 2]],
            core_index[adjacent_nodes[:, 3]],
            row,
        ]
        data = [cE, cN, cW, cS, cC]

        # Move boundary-neighbor contributions to RHS
        for j, cj in enumerate([cE, cN, cW, cS]):
            nbr = adjacent_nodes[:, j]
            is_boundary = core_index[nbr] < 0
            if np.any(is_boundary):
                rhs[is_boundary] -= cj[is_boundary] * self._eta_at_N[nbr[is_boundary]]

        # Keep only core-neighbor entries in the sparse matrix
        rows_coo = []
        cols_coo = []
        data_coo = []
        for r_arr, c_arr, d_arr in zip(rows, cols, data):
            valid = c_arr >= 0
            rows_coo.append(r_arr[valid])
            cols_coo.append(c_arr[valid])
            data_coo.append(d_arr[valid])

        rows_coo = np.concatenate(rows_coo)
        cols_coo = np.concatenate(cols_coo)
        data_coo = np.concatenate(data_coo)

        left_hand_side = sp.sparse.coo_matrix(
            (data_coo, (rows_coo, cols_coo)), shape=(n_core, n_core)
        ).tocsr()

        # Jacobi (diagonal) preconditioner as a LinearOperator
        diag = left_hand_side.diagonal()
        # Avoid division by zero (should not happen for well-posed stencil)
        inv_diag = np.where(diag != 0.0, 1.0 / diag, 1.0)
        M = sp.sparse.linalg.LinearOperator(
            (n_core, n_core), matvec=lambda x: inv_diag * x, dtype=float
        )

        pcg_solution, pcg_info = sp.sparse.linalg.cg(
            left_hand_side,
            rhs,
            M=M,
            rtol=self._pcg_tolerance,
            maxiter=self._pcg_max_iterations,
            atol=0.0,
        )
        if pcg_info < 0:
            raise RuntimeError(
                f"PCG failed with illegal input/breakdown (info={pcg_info})."
            )

        # Getting the new water surface elevation
        self._eta = np.zeros_like(self._eta_at_N)
        self._eta[core] = pcg_solution
        # Boundary conditions
        # Zero-Gradient Boundary Condition applied on open boundaries
        # Water surface elevation
        ## Updating the new WSE ('eta') with the fixed nodes values
        if self._fixed_nodes_exist is True:
            self._eta[self._fixed_entry_nodes] = (
                self._entry_nodes_h_values - self._z[self._fixed_entry_nodes]
            )

        ## Getting the 1-line-upstream nodes from boundary nodes
        tempCalc1 = self._grid.active_adjacent_nodes_at_node[self._open_boundary_nodes]
        open_boundary_nodes_1_backwards = np.extract(tempCalc1 >= 0, tempCalc1)

        ## Applying Zero-Gradient Open Boundary for WATER DEPTH (Sloped WSE)
        self._eta[self._open_boundary_nodes] = (
            self._eta_at_N[open_boundary_nodes_1_backwards]
            + self._z[open_boundary_nodes_1_backwards]
            - self._z[self._open_boundary_nodes]
        )

        self._eta = np.where(
            abs(self._eta) > abs(self._z), -self._z, self._eta
        )  # Correcting WSE below topographic elevation

        self._eta_at_links = self._grid.map_mean_of_link_nodes_to_link(self._eta)

        # Corner nodes treatment
        self._eta[self.grid.corner_nodes] = np.mean(
            self._eta[self._adjacent_nodes_at_corner_nodes], axis=1
        )

        # Updating water velocity
        # tempB1 :  Considering only wet cells
        # tempB2 :  Cells with elevation below the water surface elevation
        tempB1 = np.where(
            abs(self._eta[self._grid.node_at_link_head])
            <= abs(self._z[self._grid.node_at_link_head] - self._threshold_depth),
            1,
            0,
        )
        tempB2 = np.where(
            abs(self._z[self._grid.node_at_link_head])
            > abs(self._eta[self._grid.node_at_link_tail]),
            1,
            0,
        )
        tempB1 = tempB1 + tempB2
        tempB1 = np.where(tempB1 > 1, 1, 0)

        # Updating water velocity
        tempCalc1 = (
            self._theta
            * self._g
            * self._dt
            / dx
            * (self._grid.calc_diff_at_link(self._eta))
            * self._h_at_N_at_links
            / self._a_links
        )
        self._vel = self._g_links / self._a_links - tempCalc1 * tempB1

        # Only updating velocity on wet cells
        self._vel = np.where(self._wet_links, self._vel, 0)

        # Boundary conditions
        # Zero-Gradient Boundary Condition applied on open boundaries
        # Water velocity
        ## Updating the new Velocity with the fixed links values
        if self._fixed_links_exist is True:
            self._vel[self._fixed_entry_links] = self._entry_links_vel_values

        ## Getting the boundary links
        tempB1 = np.isin(self.grid.active_links, self._open_boundary_links)
        open_boundary_active_links = self._grid.active_links[tempB1]

        ## Getting the 1-line-upstream links from boundary links
        tempCalc1 = np.tile(open_boundary_active_links, (4, 1)).T
        tempCalc2 = self._grid.links_at_node[open_boundary_nodes_1_backwards]

        # Getting the link positions to extract them from all the surrounding nodes
        tempB1 = np.tile([0, 1, 2, 3], (len(self._open_boundary_nodes), 1))
        tempB2 = tempB1[tempCalc1 == tempCalc2]

        # It gives me the link indices to extract
        # folowing the face direction
        tempB1 = np.where(tempB2 == 0, 2, tempB2)
        tempB1 = np.where(tempB2 == 1, 3, tempB1)
        # tempB1 is where the target link is located
        tempB1 = np.where(tempB2 == 2, 0, tempB1)
        # tempB2 is where the upstream link is located
        tempB1 = np.where(tempB2 == 3, 1, tempB1)

        open_boundary_active_links_1_backwards = tempCalc2[
            [range(tempCalc2.shape[0])], tempB1
        ][0]

        ## Applying Zero-Gradient Open Boundary for Velocity
        self._vel[open_boundary_active_links] = self._vel_at_N[open_boundary_active_links_1_backwards]

        # Updating water depth at links
        # Using only values where the WSE is above the topographic elevation
        tempB1 = np.where(abs(self._eta) <= abs(self._z - self._threshold_depth), 1, 0)

        # Updating water depth at links
        tempCalc1 = (
            self._z_at_links + self._eta[self._grid.node_at_link_head]
        ) * tempB1[self._grid.node_at_link_head]
        tempCalc2 = (
            self._z_at_links + self._eta[self._grid.node_at_link_tail]
        ) * tempB1[self._grid.node_at_link_tail]
        tempCalc3 = np.zeros_like(self._h_at_N_at_links)

        self._h_at_links = np.array((tempCalc1, tempCalc2, tempCalc3)).max(axis=0)

        # Applying boundary condition at links
        self._h_at_links[self._open_boundary_links] = (
            self._z_at_links[self._open_boundary_links]
            + self._eta_at_links[self._open_boundary_links]
        )

        # Wet cells threshold
        self._h_at_links = np.where(
            self._h_at_links < self._threshold_depth, 0, self._h_at_links
        )

        # Updating wet links
        self._wet_links = np.where(
            self._h_at_links >= self._threshold_depth, True, False
        )
        self._vel = self._vel * self._wet_links

        # Calculating average water depth at nodes
        # If a node is dry, using only surrounding links such that 'WSE' is above 'z'
        # If a node is wet, using all surrounding links even if 'WSE' is below 'z' (jumps)

        # Checking surrounding wet links
        surrounding_links = self._grid.links_at_node[self.grid.core_nodes]

        # Checking whether the core node is wet (T) or dry (F)
        tempB1 = abs(self._eta[self.grid.core_nodes]) < abs(
            self._z[self.grid.core_nodes] - self._threshold_depth
        )

        # Checking whether surrounding links are wet (T) or dry (F)
        tempB2 = self._wet_links[surrounding_links]

        # Checking whether surrounding 'WSE' links are above (T) or below (F) 'z' at nodes
        tempB3 = (
            abs(self._eta_at_links[surrounding_links])
            < abs(self._z[self.grid.core_nodes] - self._threshold_depth)[:, None]
        )

        # Getting the number of wet links around each core node, satisfying tempB2,
        # and avoiding divisions by zero
        tempCalc2 = np.sum(tempB2 * 1, axis=1)
        tempCalc2 = np.where(tempCalc2 == 0, -9999, tempCalc2)

        # Getting the number of wet links around each core node, satisfying tempB3,
        # and avoiding divisions by zero
        tempCalc3 = np.sum(tempB2 * tempB3 * 1, axis=1)
        tempCalc3 = np.where(tempCalc3 == 0, -9999, tempCalc3)

        # Updating water depth
        # h = h_at_N - rmg.calc_net_flux_at_node(h_at_links*vel) # (influx if negative)
        self._h[self.grid.core_nodes] = np.where(
            tempCalc3 > 0,
            np.sum(self._h_at_links[surrounding_links] * tempB2 * tempB3, axis=1)
            / tempCalc3,
            0,
        )  # Dry nodes, tempB1 == False

        ### Updating boundary nodes
        if self._fixed_nodes_exist is True:
            self._h[self._fixed_entry_nodes] = self._entry_nodes_h_values
        self._h[self._open_boundary_nodes] = (
            self._eta[self._open_boundary_nodes] + self._z[self._open_boundary_nodes]
        )
        self._h = np.where(self._h < self._threshold_depth, 0, self._h)

        # Corner nodes treatment
        self._h[self.grid.corner_nodes] = np.mean(
            self._h[self._adjacent_nodes_at_corner_nodes], axis=1
        )

        # Updating wet nodes
        self._wet_nodes = np.where(self._h >= self._threshold_depth, True, False)

        # Storing values in the grid
        self._grid.at_node["surface_water__depth"] = self._h
        self._grid.at_link["surface_water__velocity"] = self._vel
        self._grid.at_node["surface_water__elevation"] = self._eta + (
            self._max_elevation + self._additional_z
        )

        self._grid.at_link["surface_water__velocity_at_N-1"] = self._vel_at_N
        self._grid.at_node["surface_water__elevation_at_N-1"] = self._eta_at_N + (
            self._max_elevation + self._additional_z
        )
        self._grid.at_node["surface_water__elevation_at_N-2"] = self._eta_at_N_1 + (
            self._max_elevation + self._additional_z
        )

        # Storing values at previous time steps
        self._eta_at_N = self._eta.copy()
        self._eta_at_N_1 = self._eta_at_N.copy()
        self._eta_at_N_2 = self._eta_at_N_1.copy()

        self._h_at_N = self._h.copy()
        self._h_at_N_at_links = self._h_at_links.copy()

        self._vel_at_N = self._vel.copy()
        self._vel_at_N_1 = self._vel_at_N.copy()