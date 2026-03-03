"""Landlab component that calculates temporal and spatial changes in river
bed elevation and grain size distribution. Also, this component predicts
fractional or total sediment transport based on the specified bed load
transport model. Hydraulic properties are obtained from an external flow
component. We recommend coupling it with the OverlandFlow component from
Adams et al, 2017.

.. codeauthor:: Angel Monsalve
.. codecoauthors: Sam Anderson, Nicole Gasparini, Elowyn Yager

Examples
--------

Let's import all the required libraries

>>> import numpy as np
>>> import copy
>>> from landlab import RasterModelGrid, imshow_grid
>>> from landlab.components import RiverBedDynamics
>>> from landlab.grid.mappers import map_mean_of_link_nodes_to_link

Create a grid on which to calculate sediment transport

>>> grid = RasterModelGrid((5, 5))

The grid will need some data to run the RiverBedDynamics component.
To check the names of the fields that provide input use the *input_var_names*
class property.

>>> RiverBedDynamics.input_var_names
('surface_water__depth', 'surface_water__velocity', 'topographic__elevation')

Create fields of data for each of these input variables. When running a
complete simulation some of these variables will be created by the flow model.
Notice that surface water depth and velocity are required at links. However,
specifying these variables at nodes is easier and then we can map the fields
onto links. By doing so, we don't have to deal with links numbering. When this
component is coupled to OverlandFlow there is no need to map fields because it
is done automatically within the component.

We start by creating the topography data

>>> grid.at_node["topographic__elevation"] = [
...     [1.07, 1.06, 1.00, 1.06, 1.07],
...     [1.08, 1.07, 1.03, 1.07, 1.08],
...     [1.09, 1.08, 1.07, 1.08, 1.09],
...     [1.09, 1.09, 1.08, 1.09, 1.09],
...     [1.09, 1.09, 1.09, 1.09, 1.09],
... ]

Let's save a copy of this topography, we will use it later.

>>> z0 = copy.deepcopy(grid.at_node["topographic__elevation"])

We set the boundary conditions

>>> grid.set_watershed_boundary_condition(grid.at_node["topographic__elevation"])

And check the node status

>>> grid.status_at_node.reshape(grid.shape)
array([[4, 4, 1, 4, 4],
       [4, 0, 0, 0, 4],
       [4, 0, 0, 0, 4],
       [4, 0, 0, 0, 4],
       [4, 4, 4, 4, 4]], dtype=uint8)

Which tell us that there is one outlet located on the 3rd node

The topography data can be display using

>>> imshow_grid(grid, "topographic__elevation")

Now we add some water into the watershed. In this case is specified in nodes

>>> grid.at_node["surface_water__depth"] = [
...     [0.102, 0.102, 0.102, 0.102, 0.102],
...     [0.102, 0.102, 0.102, 0.102, 0.102],
...     [0.102, 0.102, 0.102, 0.102, 0.102],
...     [0.102, 0.102, 0.102, 0.102, 0.102],
...     [0.102, 0.102, 0.102, 0.102, 0.102],
... ]

Now, we give the water a velocity.

>>> grid.at_node["surface_water__velocity"] = [
...     [0.25, 0.25, 0.25, 0.25, 0.25],
...     [0.25, 0.25, 0.25, 0.25, 0.25],
...     [0.25, 0.25, 0.25, 0.25, 0.25],
...     [0.25, 0.25, 0.25, 0.25, 0.25],
...     [0.25, 0.25, 0.25, 0.25, 0.25],
... ]

Note that in this example, we are attempting to specify a vector at a node
using a single value. This is done intentionally to emphasize the process.
The component will interpret this as the vector's magnitude, and, given its
location in the grid, it will manifest different components. When using
OverlandFlow, there is no need to specify a velocity because it is a
byproduct of the component.

For the purpose of this illustration, we will make an assumption that the
conditions remain identical to the previous time step.

By default, when creating our grid we used a spacing of 1 m in the x and y
directions. Therefore, the discharge is 0.0255 m3/s. Discharge is always in
units of m3/s.

Now we map nodes into links when it is required

>>> grid.at_link["surface_water__depth"] = map_mean_of_link_nodes_to_link(
...     grid, "surface_water__depth"
... )
>>> grid.at_link["surface_water__velocity"] = map_mean_of_link_nodes_to_link(
...     grid, "surface_water__velocity"
... )

We will assume, for the sake of demonstration, that we have two sectors with
different bed surface grain size (GSD). We can tell the component the location
of these two different GSD within the watershed (labeled as 0 and 1). This will
be included during the instantiation

>>> gsd_loc = [
...     [0, 1.0, 1.0, 1.0, 0],
...     [0, 1.0, 1.0, 1.0, 0],
...     [0, 1.0, 1.0, 1.0, 0],
...     [0, 1.0, 1.0, 1.0, 0],
...     [0, 1.0, 1.0, 1.0, 0],
... ]

We assign a GSD to each location. The structure of this array is:
First column contains the grain sizes in milimiters
Second column is location 0 in 'bed_grainSizeDistribution__location'
Third column is location 1 in 'bed_grainSizeDistribution__location', and so on

>>> gsd = [[32, 100, 100], [16, 25, 50], [8, 0, 0]]

Instantiate the `RiverBedDynamics` component to work on the grid, and run it.

>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="Parker1990",
...     bed_surf__gsd_loc_node=gsd_loc,
...     track_stratigraphy=True,
... )
>>> rbd.run_one_step()

After running RiverBedDynamics, we can check if the different GSD locations
were correctly included

>>> gsd_loc_rbd = rbd._bed_surf__gsd_loc_node.reshape(grid.shape)
>>> gsd_loc_rbd
array([[0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 1, 1, 1, 0]])

Let's check at the calculated shear_stress

>>> shearStress = rbd._surface_water__shear_stress_link
>>> np.round(shearStress, decimals=3)  # doctest: +NORMALIZE_WHITESPACE
array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   , 40.011, 40.011,  0.   ,  0.   ,
       10.003, 40.011, 10.003,  0.   ,  0.   , 10.003, 10.003,
        0.   ,  0.   , 10.003, 10.003, 10.003,  0.   ,  0.   ,
       10.003, 10.003,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
        0.   ,  0.   ,  0.   ,  0.   ,  0.   ])

Notice that links at borders have zero shear stress. Let's check at the calculated
net bedload. Hereinafter, most of the results will show between 3 to 6 decimals.
This is just to avoid roundoff errors problems

>>> qb = rbd._sed_transp__net_bedload_node.reshape(grid.shape)
>>> np.around(qb, decimals=6)
array([[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.001598, -0.004793,  0.001598,  0.      ],
       [ 0.      ,  0.      ,  0.001597,  0.      ,  0.      ],
       [ 0.      ,  0.      , -0.      ,  0.      ,  0.      ],
       [ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ]])

Considering the link upstream the watershed exit, link Id 15, we can obtain the
bed load transport rate

>>> qb_l15 = rbd._sed_transp__bedload_rate_link[15]
>>> formatted_number = "{:.4e}".format(qb_l15)
>>> print(formatted_number)
-1.5977e-03

Therefore, the bed load transport rate according to Parker 1990 surface-based
equation is 1.598 * 10^-3 m2/s. Negative means that is going in the negative
Y direction (towards bottom or towards south)

The GSD at this place is:

>>> qb_gsd_l15 = rbd._sed_transp__bedload_gsd_link[15]
>>> np.round(qb_gsd_l15, decimals=3)
array([0.475, 0.525])

Which in cummulative percentage is equivalent to

==== =======
D mm % Finer
==== =======
32   100.000
16   52.498
8    0.000
==== =======

Grain sizes are always given in mm.
We can also check the bed load grain size distribution in all links

>>> qb_gsd_l = rbd._sed_transp__bedload_gsd_link
>>> np.round(qb_gsd_l, decimals=3)
array([[0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.475, 0.525],
       [0.475, 0.525],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.282, 0.718],
       [0.475, 0.525],
       [0.282, 0.718],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.282, 0.718],
       [0.282, 0.718],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.282, 0.718],
       [0.282, 0.718],
       [0.282, 0.718],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.282, 0.718],
       [0.282, 0.718],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ],
       [0.   , 0.   ]])

Zeros indicate that there is no sediment transport of that grain size
at that location.

After the flow acted on the bed and sediment transport occured we can check
the new topographic elevation field

>>> z = grid.at_node["topographic__elevation"].reshape(grid.shape)
>>> np.round(z, decimals=3)
array([[1.07 , 1.06 , 1.007, 1.06 , 1.07 ],
       [1.08 , 1.068, 1.037, 1.068, 1.08 ],
       [1.09 , 1.08 , 1.068, 1.08 , 1.09 ],
       [1.09 , 1.09 , 1.08 , 1.09 , 1.09 ],
       [1.09 , 1.09 , 1.09 , 1.09 , 1.09 ]])

Let's take a look at bed load transport rate when we use the different bedload equations.
First, let's recover the original topography.

>>> grid.at_node["topographic__elevation"] = z0.copy()

For the defaul MPM model we get:

>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="MPM",
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb_l15 = rbd._sed_transp__bedload_rate_link[15]
>>> print("{:.4e}".format(qb_l15))
-2.2970e-03

For Fernandez Luque and Van Beek:

>>> grid.at_node["topographic__elevation"] = z0.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="FLvB",
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb_l15 = rbd._sed_transp__bedload_rate_link[15]
>>> print("{:.4e}".format(qb_l15))
-1.6825e-03

For Wong and Parker:

>>> grid.at_node["topographic__elevation"] = z0.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="WongAndParker",
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb_l15 = rbd._sed_transp__bedload_rate_link[15]
>>> print("{:.4e}".format(qb_l15))
-1.1326e-03

For Huang:

>>> grid.at_node["topographic__elevation"] = z0.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="Huang",
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb_l15 = rbd._sed_transp__bedload_rate_link[15]
>>> print("{:.4e}".format(qb_l15))
-1.1880e-03

For Wilcock and Crowe 2003:

>>> grid.at_node["topographic__elevation"] = z0.copy()
>>> rbd = RiverBedDynamics(
...     grid,
...     gsd=gsd,
...     bedload_equation="WilcockAndCrowe",
...     bed_surf__gsd_loc_node=gsd_loc,
... )
>>> rbd.run_one_step()
>>> qb_l15 = rbd._sed_transp__bedload_rate_link[15]
>>> print("{:.4e}".format(qb_l15))
-5.3081e-04

The previous example, covers a relatively complete case. For demonstration purposes
let's see some other options that show how to use or change the default setting.
If the grain size distribution is not specified, what value will river bed dynamics use?

>>> rbd = RiverBedDynamics(grid)
>>> rbd._gsd
array([[ 32, 100],
       [ 16,  25],
       [  8,   0]])

The sand content can be calculated from a grain size distribution

>>> gsd = [[128, 100], [64, 90], [32, 80], [16, 50], [8, 20], [2, 10], [1, 0]]
>>> rbd = RiverBedDynamics(grid, gsd=gsd, bedload_equation="MPM")
>>> sandContent = rbd._bed_surf__sand_fract_node[20]
>>> print(float("{:.4f}".format(round(sandContent, 4))))
0.1

But it is different if we use Parker 1990, because it removes sand content

>>> rbd = RiverBedDynamics(grid, gsd=gsd, bedload_equation="Parker1990")
>>> float(rbd._bed_surf__sand_fract_node[20])
0.0

What happens if we give it a set of wrong optional fields. The following
fields will have only two elements, which is different than the number of
nodes and links

>>> qb_imposed = np.array([1, 2])
>>> rbd = RiverBedDynamics(grid, sed_transp__bedload_rate_fix_link=qb_imposed)
>>> rbd._sed_transp__bedload_rate_fix_link
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0.])

>>> gsd_loc = np.array([1, 2])
>>> rbd = RiverBedDynamics(grid, bed_surf__gsd_loc_node=gsd_loc)
>>> rbd._bed_surf__gsd_loc_node.reshape(grid.shape)
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])

>>> gsd_fix = np.array([1, 2])
>>> rbd = RiverBedDynamics(grid, bed_surf__gsd_fix_node=gsd_fix)
>>> rbd._bed_surf__gsd_fix_node.reshape(grid.shape)
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])

>>> elev_fix = np.array([1, 2])
>>> rbd = RiverBedDynamics(grid, bed_surf__elev_fix_node=elev_fix)
>>> rbd._bed_surf__elev_fix_node.reshape(grid.shape)
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])

>>> vel_n_1 = np.array([1, 2])
>>> rbd = RiverBedDynamics(grid, surface_water__velocity_prev_time_link=vel_n_1)
>>> rbd._surface_water__velocity_prev_time_link
array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
       0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
       0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
       0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

For sed_transp__bedload_gsd_fix_link, for simplicity, we only show
links 0, 5, and 10

>>> qb_gsd_imposed = np.array([1, 2])
>>> rbd = RiverBedDynamics(grid, sed_transp__bedload_gsd_fix_link=qb_gsd_imposed)
>>> rbd._sed_transp__bedload_gsd_fix_link[[0, 5, 10], :]
array([[0., 0.],
       [0., 0.],
       [0., 0.]])

In summary, in all these cases the wrong given value is override by default values.
But, if the size of the array is correct the specified condition is used.
For simplicity, we only show links 0, 5, and 10

>>> qb_imposed = np.full(grid.number_of_links, 1)
>>> rbd = RiverBedDynamics(grid, sed_transp__bedload_rate_fix_link=qb_imposed)
>>> rbd._sed_transp__bedload_rate_fix_link[[0, 5, 10]]
array([1., 1., 1.])

>>> gsd_loc = np.full(grid.number_of_nodes, 1)
>>> rbd = RiverBedDynamics(grid, bed_surf__gsd_loc_node=gsd_loc)
>>> rbd._bed_surf__gsd_loc_node.reshape(grid.shape)
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])

>>> gsd_fix = np.full(grid.number_of_nodes, 1)
>>> rbd = RiverBedDynamics(grid, bed_surf__gsd_fix_node=gsd_fix)
>>> rbd._bed_surf__gsd_fix_node.reshape(grid.shape)
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])

>>> elev_fix = np.full(grid.number_of_nodes, 1)
>>> rbd = RiverBedDynamics(grid, bed_surf__elev_fix_node=elev_fix)
>>> rbd._bed_surf__elev_fix_node.reshape(grid.shape)
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])

>>> vel_n_1 = np.full(grid.number_of_links, 1)
>>> rbd = RiverBedDynamics(grid, surface_water__velocity_prev_time_link=vel_n_1)
>>> rbd._surface_water__velocity_prev_time_link[[0, 5, 10]]
array([1., 1., 1.])

>>> qb_gsd_imposed = np.ones((grid.number_of_links, 2))  # 2 comes from gsd.shape[0]-1
>>> rbd = RiverBedDynamics(grid, sed_transp__bedload_gsd_fix_link=qb_gsd_imposed)
>>> rbd._sed_transp__bedload_gsd_fix_link[[0, 5, 10]]
array([[1., 1.],
       [1., 1.],
       [1., 1.]])

Using the hydraulics radius is also possible. Let's compare the shear stress with and
without that option. First, without including the hydraulics radius.

>>> grid.at_node["topographic__elevation"] = z0.copy()
>>> rbd = RiverBedDynamics(grid)
>>> rbd.run_one_step()
>>> print(np.around(rbd._shear_stress[15], decimals=3))
-40.011

Now, we will consider the hydraulics radius

>>> grid.at_node["topographic__elevation"] = z0.copy()
>>> rbd = RiverBedDynamics(grid, use_hydraulics_radius_in_shear_stress=True)
>>> rbd.run_one_step()
>>> print(np.around(rbd._shear_stress[15], decimals=3))
-33.232

So, there is an important difference between the two ways of calculating it.

Architecture Overview (Phases 3–5)
------------------------------------
The component is structured around four pluggable subsystems, each implemented
as a standalone class that reads and writes state on the ``RiverBedDynamics``
instance.  This makes each piece independently testable and extensible without
touching the core component.

**Bedload equation registry** (``_bedload_equation_base.py``)
    The ``bedload_equation`` parameter selects a concrete subclass of
    ``BedloadEquation``.  All equations expose a uniform ``calculate(rbd)``
    interface, so adding a new formula requires only subclassing and
    registration — no changes to the core.

    Available keys: ``"MPM"``, ``"FLvB"``, ``"WongAndParker"``, ``"Huang"``,
    ``"Parker1990"``, ``"WilcockAndCrowe"``.

**Shear stress calculator** (``_shear_stress.py``)
    ``ShearStressCalculator`` encapsulates the two shear-stress formulations
    (depth-slope and hydraulic-radius-slope).  Controlled by
    ``use_hydraulics_radius_in_shear_stress``.

**GSD evolver** (``_gsd_evolver.py``)
    ``ToroEscobarEvolver`` implements the Toro-Escobar, Paola & Parker (1996)
    fractional Exner equation for bed surface sorting.  The spatial flux
    scheme is selectable via ``gsd_advection_scheme``:

    * ``"upwind"`` (default) — blended upwind / Lax–Wendroff, first-order.
    * ``"tvd_minmod"`` — TVD minmod-limited, second-order in smooth regions.

**Time integrators for the Exner equation** (``RiverBedDynamics.py``)
    Controlled by ``time_stepping``:

    * ``"euler"`` (default) — explicit first-order forward Euler.  Stable
      only for ``dt ≤ dt_CFL``; cheap (one transport evaluation per step).
    * ``"rk2"`` — Heun's predictor-corrector (second-order in time).  Costs
      two transport evaluations per step but can use ~2× larger ``dt`` for
      the same accuracy.
    * ``"implicit"`` — linearised backward Euler.  Unconditionally stable;
      allows ``dt`` far beyond the explicit CFL limit.  Costs one Jacobian
      computation (``O(n_core)`` transport evaluations) and a sparse LU
      solve per step.  Best for long morphodynamic simulations where
      temporal accuracy is less critical than stability.

New Parameters Added in Phases 2–5
-------------------------------------
The following parameters were added after the initial release and are not
described in the original ``__init__`` docstring above:

``time_stepping`` : ``{"euler", "rk2", "implicit"}``, default ``"euler"``
    Exner time-integration scheme (see above).

``check_advective_cfl`` : bool, default ``True``
    Emit a ``UserWarning`` when the advective Courant number exceeds 1.

``adaptive_dt`` : bool, default ``False``
    Automatically reduce ``dt`` to the CFL-safe value each step.

``gsd_advection_scheme`` : ``{"upwind", "tvd_minmod"}``, default ``"upwind"``
    Spatial scheme for the fractional bedload flux divergence in the GSD
    evolver.

``check_gsd_residual`` : bool, default ``True``
    Warn when the pre-renormalisation GSD residual ``|Σf_i - 1|`` exceeds
    ``gsd_residual_threshold``.  Useful for detecting numerical drift.

``gsd_residual_threshold`` : float, default ``1e-3``
    Residual threshold for the GSD normalisation warning.

``gsd_n_minus_1`` : bool, default ``False``
    If ``True``, evolve only N-1 grain-size fractions and recover the last
    as ``1 - Σ(rest)``.  Eliminates accumulated normalisation drift.

``use_bed_diffusion`` : bool, default ``False``
    Enable gravitational diffusion correction to the Exner equation
    (Talmon et al. 1995).

``bed_diffusion_mode`` : ``{"nonlinear", "constant"}``, default ``"nonlinear"``
    Diffusion coefficient formulation.

``bed_diffusion_mu`` : float, default ``1.0``
    Calibration coefficient for nonlinear diffusion (``D = |qb| / mu``).

``bed_diffusion_coeff`` : float, default ``0.0``
    Constant diffusion coefficient [m² s⁻¹] for ``bed_diffusion_mode="constant"``.

``check_diffusion_cfl`` : bool, default ``True``
    Warn when the diffusive CFL number exceeds 0.5.

Diagnostic Attributes Added in Phases 2–5
------------------------------------------
``_bed_surf__gsd_residual_max`` : float
    Maximum ``|Σf_i - 1|`` across all links before the last renormalisation.

``_bed_surf__gsd_residual_mean`` : float
    Mean ``|Σf_i - 1|`` across all links before the last renormalisation.

References
----------
Toro-Escobar, C. M., Paola, C., & Parker, G. (1996). Transfer function for
the deposition of poorly sorted gravel in response to streambed aggradation.
*Journal of Hydraulic Research*, 34(1), 35–53.

Talmon, A. M., Struiksma, N., & Van Mierlo, M. C. L. M. (1995). Laboratory
measurements of the direction of sediment transport on transverse alluvial-bed
slopes. *Journal of Hydraulic Research*, 33(4), 495–517.

Soni, J. P. (1981). Laboratory study of aggradation in alluvial channels.
*Journal of Hydrology*, 49(1–2), 87–106.

Seal, R., Paola, C., Parker, G., Southard, J. B., & Wilcock, P. R. (1997).
Experiments on downstream fining of gravel: I. Narrow-channel runs.
*Journal of Hydraulic Engineering*, 123(10), 874–884.

"""

import numpy as np
import scipy.constants

from landlab import Component

# _bedload_equation_base and _shear_stress are imported lazily inside
# __init__ to avoid a load-order race with the package __init__.py.
from . import _initialize_fields as initialize
from . import _initialize_gsd as initialize_gsd
from . import _nodes_and_links_info as info
from . import _stratigraphy as stratigraphy


class RiverBedDynamics(Component):
    """Predicts the evolution of a river bed.

    Landlab component that predicts the evolution of a river bed
    considering changes in elevation and grain size distribution in response to
    bed load transport according to the Exner equation and the transfer
    function of Toro-Ecobar et al., (1996).

    To estimate temporal and spatial changes in river bed properties, this
    component predicts the bedload transport rate and fractional transport at
    each link using unsteady shear stress. Time-varying hydraulic
    variables are obtained from a surface flow, for instance,
    :class:`~OverlandFlow`.

    The primary method of this class is :func:`run_one_step`.

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    Not required but recommended

    Adams, J., Gasparini, N., Hobley, D., Tucker, G., Hutton, E., Nudurupati,
    S., Istanbulluoglu, E. (2017). The Landlab v1. 0 OverlandFlow component:
    a Python tool for computing shallow-water flow across watersheds.
    Geoscientific Model Development  10(4), 1645.
    https://dx.doi.org/10.5194/gmd-10-1645-2017

    **Additional References**

    G. Parker (1990) Surface-based bedload transport relation for gravel
    rivers, Journal of Hydraulic Research, 28:4, 417-436,
    DOI: 10.1080/00221689009499058

    Wilcock, P. R., & Crowe, J. C. (2003). Surface-based transport model for
    mixed-size sediment. Journal of hydraulic engineering, 129(2), 120-128.
    DOI: 10.1061/(ASCE)0733-9429(2003)129:2(120)

    Meyer-Peter, E. and Müller, R., 1948, Formulas for Bed-Load Transport,
    Proceedings, 2nd Congress, International Association of Hydraulic Research,
    Stockholm: 39-64.

    Fernandez Luque, R. and R. van Beek, 1976, Erosion and transport of
    bedload sediment, Journal of Hydraulic Research, 14(2): 127-144.
    https://doi.org/10.1080/00221687609499677

    Mueller, E. R., J. Pitlick, and J. M. Nelson (2005), Variation in the
    reference Shields stress for bed load transport in gravelbed streams and
    rivers, Water Resour. Res., 41, W04006, doi:10.1029/2004WR003692

    Carlos M. Toro-Escobar, Chris Paola & Gary Parker (1996) Transfer function
    for the deposition of poorly sorted gravel in response to streambed
    aggradation, Journal of Hydraulic Research, 34:1, 35-53,
    DOI: 10.1080/00221689609498763
    """

    _name = "RiverBedDynamics"

    _unit_agnostic = False

    _info = {
        "surface_water__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "Depth of water on the surface",
        },
        "surface_water__velocity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/s",
            "mapping": "link",
            "doc": "Speed of water flow above the surface",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
    }

    def __init__(
        self,
        grid,
        gsd=None,
        rho=1000,
        rho_s=2650,
        bedload_equation="MPM",
        variable_critical_shear_stress=False,
        use_hydraulics_radius_in_shear_stress=False,
        lambda_p=0.35,
        outlet_boundary_condition="zeroGradient",
        dt=1,
        alpha=1.0,
        bed_surf__elev_fix_node=None,
        bed_surf__gsd_fix_node=None,
        sed_transp__bedload_rate_fix_link=None,
        sed_transp__bedload_gsd_fix_link=None,
        bed_surf__gsd_loc_node=None,
        surface_water__velocity_prev_time_link=None,
        current_t=0.0,
        track_stratigraphy=False,
        num_cycles_to_process_strat=10,
        bed_surf_new_layer_thick=1,
        use_bed_diffusion=False,
        bed_diffusion_mode="nonlinear",
        bed_diffusion_mu=0.5,
        bed_diffusion_coeff=0.0,
        check_diffusion_cfl=True,
        check_advective_cfl=True,
        adaptive_dt=False,
        time_stepping="euler",
        gsd_advection_scheme="upwind",
        check_gsd_residual=True,
        gsd_residual_threshold=1e-3,
        gsd_n_minus_1=False,
    ):
        """Calculates the evolution of a river bed based on bed load transport
        and fractional rates on links. An external flow hydraulics solver, such
        as OverlandFlow, is required to predict flow variables in time and space.
        The shear stress used in sediment transport equations takes into account
        the time and spatial variability of water depth and flow velocity.

        This component adjusts topographic elevation and grain size
        distribution at each node within a Landlab grid.

        Parameters
        ----------
        grid : RasterModelGrid
            area Landlab grid.
        gsd : ndarray of float
            Grain size distribution. Must contain as many GSDs as there are
            different indexes in GSD Location.
        rho : float, optional
            Fluid density. Defaults to the density of water at 1000 kg/m^3.
        rho_s : float, optional
            Sediment density. Defaults to sediment density of 2650 kg/m^3.
        bedload_equation : String, optional
            Selects the bedload transport equation. Options include:

            * ``'MPM'`` for Meyer Peter and Muller,
            * ``'FLvB'`` for Fernandez Luque & van Beek (1976),
            * ``'Parker1990'`` for Parker 1990,
            * ``'WilcockAndCrowe'`` for Wilcock and Crowe 2003.
            * ``'WongAndParker'`` for Wong and Parker 2006.
            * ``'Huang'`` for Huang 2010.
        variable_critical_shear_stress: bool, optional
            If ``True``, the critical shear stress in ``'MPM'`` or ``'FLvB'`` will be
            obtained using Mueller et al. (2005) equation.
        use_hydraulics_radius_in_shear_stress: bool, optional
            If ``True``, shear stress calculations will use the hydraulic radius.
            Defaults to False, which uses water depth.
        lambda_p : float, optional
            Sediment porosity. Default value is 0.35.
        outlet_boundary_condition : str, optional
            Sets the boundary condition at the watershed outlet. Options are:
            ``'zeroGradient'`` (maps the outlet to the upstream node, default) or
            ``'fixedValue'`` (does not change the value at the outlet during the
            run).
        dt: float, optional
            Time step in seconds. When this component is coupled to a flow model,
            it is dynamically updated.
        alpha : float, optional
            An upwinding coefficient for a central difference scheme when
            updating the bed GSD - default value is 1.0 - a value of 0.5
            generates a central differences scheme.
        bed_surf__elev_fix_node: ndarray of int, optional
            Sets a node as a fixed elevation, this means that it's elevation does
            not change during the simulation. Use 1 to denote a fix node and 0 for
            nodes that can change elevation
            Units: - , mapping: node
        sed_transp__bedload_rate_fix_link: ndarray of float, optional
            Sets the sediment transport rate per unit width as fixed at a given link.
            This means that its value will not change unless is manually redefined.
            When defining it, use 0 for non-fixed and other float numbers to specify the
            bed load rate. Specifing actually a zero bed load transport is not supported,
            but a very small value could be used instead.
            Units: m^2/s, mapping: link
        bed_surf__gsd_fix_node: ndarray of int, optional
            Sets a node as fixed gsd, this means that it's gsd does not change during
            the simulation.
            Units: - , mapping: node
        bed_surf__gsd_loc_node: ndarray of int, optional
            Sets the location at each node in which the GSD applies
            Units: - , mapping: node
        sed_transp__bedload_gsd_fix_link: ndarray of float, optional
            Sets the sediment transport GSD where sediment supply is imposed. It's
            size should be columns: number of links; rows: Number of grain sizes - 1
            Units: - , mapping: link
        surface_water__velocity_prev_time_link: ndarray of float, optional
            Speed of water flow above the surface in the previous time step
            Units: m/s, mapping: link
        current_t : float, optional
            Current simulation time or elapsed time. It does not update automatically
            Units: s
        track_stratigraphy : bool, optional
            If ``True``, the component stores the GSD of each layer at every node.
            This is computationally demanding as it needs to read/write data
            at every time step. Recommended only when cycles of erosion and
            deposition are frequent and very important.
        num_cycles_to_process_strat : int, optional
            If ``track_stratigraphy`` is ``True``, data will be read and stored every
            ``'num_cycles_to_process_strat'`` time steps. Must be larger
            or equal to 1. All data are written to the hard drive. It does not use
            *Landlab* layers.
        bed_surf_new_layer_thick : float, optional
            When this thickness is reached in a deposition or erosion process, meaning
            that the new bed at a node is "bed_surf_new_layer_thick" higher or
            deeper a new layer is created, in which the time changes in gsd are
            consolidated and becomes a single value.
        use_bed_diffusion : bool, optional
            If ``True``, adds a gravitational diffusion correction term to the
            Exner equation::

                (1 - λp) ∂z/∂t = -∇·q_b  +  ∇·(D ∇z)

            The second term represents the downslope gravitational deflection of
            bedload trajectories (Engelund 1974, Talmon et al. 1995). It acts
            isotropically and smooths staircase artefacts that arise from the
            square-cell discretization. Disabled by default.
        bed_diffusion_mode : str, optional
            Controls how the diffusion coefficient D [m²/s] is determined.
            Options:

            * ``'nonlinear'`` — D = |qb| / μ, where qb is the local bedload
              transport rate per unit width and μ is the Talmon friction
              parameter ``bed_diffusion_mu``. D is zero wherever there is no
              transport, which is physically consistent.
            * ``'constant'`` — D is the fixed value ``bed_diffusion_coeff``
              everywhere (simpler, but requires empirical calibration).

            Default is ``'nonlinear'``.
        bed_diffusion_mu : float, optional
            Talmon friction parameter μ used when ``bed_diffusion_mode='nonlinear'``.
            Controls the strength of gravitational diffusion relative to the
            bedload transport rate. Typical range: 0.3–1.5 for gravel beds;
            smaller values produce stronger smoothing. Default is 0.5.
        bed_diffusion_coeff : float, optional
            Constant diffusion coefficient D [m²/s] used when
            ``bed_diffusion_mode='constant'``. Default is 0.0.
        check_diffusion_cfl : bool, optional
            If ``True`` (default), emits a ``UserWarning`` when the diffusive
            CFL number exceeds 1, indicating a potentially unstable time step.
            The stability limit is Δt ≤ (1-λp)·Δx² / (2·D_max).
        check_advective_cfl : bool, optional
            If ``True`` (default), emits a ``UserWarning`` inside
            :meth:`update_bed_elevation` when the advective Exner CFL number
            exceeds 1.  The stability limit is
            Δt ≤ (1-λp)·Δx_min / |qb_max|.  Suppressed when
            ``adaptive_dt=True`` (which enforces the limit automatically).
        adaptive_dt : bool, optional
            If ``True``, :meth:`run_one_step` computes the CFL-safe time step
            via :meth:`calc_max_stable_dt` before each step and uses that
            value instead of the user-supplied ``dt``.  A ``UserWarning`` is
            emitted whenever the step is reduced.  Default is ``False``.
        time_stepping : str, optional
            Time-integration scheme for the advective Exner equation.

            * ``"euler"`` (default) — explicit first-order forward Euler.
            * ``"rk2"`` — Heun's method (explicit RK2 / predictor-corrector).
              The hydraulics (shear stress, water depth) are held fixed during
              the two-stage update; only the bed elevation changes between
              stages.  Second-order accurate in time — halving ``dt`` quarters
              the elevation error.  Costs two bedload evaluations per step but
              can use ~2× the stable ``dt``, so net cost is similar to Euler.
        gsd_advection_scheme : str, optional
            Spatial discretisation for the fractional bedload flux divergence
            inside the GSD evolution step (Toro-Escobar equation).

            * ``"upwind"`` (default) — blended upwind / Lax–Wendroff scheme
              controlled by the ``alpha`` parameter.  Stable for all transport
              directions but numerically diffusive at GSD fronts.
            * ``"tvd_minmod"`` — TVD-limited scheme using the minmod limiter.
              Second-order accurate in smooth regions; reverts to first-order
              upwind at discontinuities (sharp GSD fronts).  Reduces numerical
              smearing without introducing spurious oscillations.
        check_gsd_residual : bool, optional
            If ``True`` (default), emit a ``UserWarning`` when the pre-
            renormalisation GSD residual :math:`|\\sum_i f_i - 1|` exceeds
            ``gsd_residual_threshold`` at any link.  Useful for detecting
            numerical drift in long simulations.
        gsd_residual_threshold : float, optional
            Threshold for the GSD residual warning (default ``1e-3``).
            Residuals below this value are considered normal round-off.
        gsd_n_minus_1 : bool, optional
            If ``True`` (default), evolve only the first ``n_grains − 1``
            fractions explicitly and recover the last as
            ``f_last = 1 − Σ(rest)``.  This eliminates the drift source
            that makes renormalisation necessary.
        """
        super().__init__(grid)

        self._g = scipy.constants.g  # Acceleration due to gravity (m/s**2).
        self._rho = rho
        self._rho_s = rho_s
        self._R = (rho_s - rho) / rho

        self._normal = -self._grid.link_dirs_at_node  # Define faces normal vector

        if gsd is None:
            self._gsd = np.array([[32, 100], [16, 25], [8, 0]])
        else:
            self._gsd = np.array(gsd)

        self._bedload_equation = bedload_equation
        # Phase-3A: instantiate the equation class from the registry.
        # Lazy import avoids a load-order race with the package __init__.py.
        from ._bedload_equation_base import EQUATION_REGISTRY
        from ._shear_stress import ShearStressCalculator
        from ._gsd_evolver import ToroEscobarEvolver
        if bedload_equation not in EQUATION_REGISTRY:
            known = sorted(EQUATION_REGISTRY)
            raise ValueError(
                f"Unknown bedload_equation {bedload_equation!r}. "
                f"Valid options: {known}"
            )
        self._bedload_eq = EQUATION_REGISTRY[bedload_equation]()
        # Phase-3B: shear stress logic lives in ShearStressCalculator.
        self._shear_calc = ShearStressCalculator(
            use_hydraulics_radius=use_hydraulics_radius_in_shear_stress
        )
        # Phase-3C: GSD evolution logic lives in ToroEscobarEvolver.
        self._gsd_evolver = ToroEscobarEvolver(gsd_advection_scheme=gsd_advection_scheme)
        self._check_gsd_residual   = check_gsd_residual
        self._gsd_residual_threshold = gsd_residual_threshold
        self._gsd_n_minus_1        = gsd_n_minus_1
        # Phase 4C diagnostic fields — updated each step by GSDEvolver
        self._bed_surf__gsd_residual_max  = 0.0
        self._bed_surf__gsd_residual_mean = 0.0

        self._variable_critical_shear_stress = variable_critical_shear_stress
        self._use_hydraulics_radius_in_shear_stress = (
            use_hydraulics_radius_in_shear_stress
        )
        self._lambda_p = lambda_p
        self._alpha = alpha
        self._outlet_boundary_condition = outlet_boundary_condition

        self._grid._dt = dt
        self._current_t = current_t

        # Initialize required fields and identify nodes and links with
        # fixed surface elevation and/or gsd
        self._bed_surf__gsd_loc_node = initialize.field_at_node(
            grid, bed_surf__gsd_loc_node
        )

        self._bed_surf__elev_fix_node = initialize.field_at_node(
            grid, bed_surf__elev_fix_node
        ).astype(int)
        self._bed_surf__elev_fix_node_id = np.where(
            (self._bed_surf__elev_fix_node) == 1
        )

        self._bed_surf__elev_fix_link_id = np.unique(
            self._grid.links_at_node[self._bed_surf__elev_fix_node_id]
        )

        self._bed_surf__gsd_fix_node = initialize.field_at_node(
            grid, bed_surf__gsd_fix_node
        ).astype(int)
        self._bed_surf__gsd_fix_node_id = np.where(self._bed_surf__gsd_fix_node == 1)

        # Now self._bed_surf__gsd_fix_node changes from nodes to links
        bed_surf__gsd_fix_link = np.unique(
            grid.links_at_node[self._bed_surf__gsd_fix_node_id]
        )
        self._bed_surf__gsd_fix_link = bed_surf__gsd_fix_link[
            np.where(bed_surf__gsd_fix_link >= 0)
        ]

        self._surface_water__velocity_prev_time_link = initialize.velocity_at_link(
            grid, surface_water__velocity_prev_time_link
        )
        self._sed_transp__bedload_rate_link = self._grid.zeros(
            at="link"
        )  # Volumetric bed load transport rate per unit width
        self._sed_transp__net_bedload_node = self._grid.zeros(at="node")
        self._sed_transp__bedload_gsd_link = self._grid.zeros(at="link")
        self._surface_water__shear_stress_link = self._grid.zeros(at="link")

        # Initialize the bed surface grain size properties using inputs
        self.define_initial_bed_properties()

        self._sed_transp__bedload_rate_fix_link = initialize.field_at_link(
            grid, sed_transp__bedload_rate_fix_link
        )
        self._sed_transp__bedload_rate_fix_link_id = np.where(
            self._sed_transp__bedload_rate_fix_link != 0
        )[0]
        self._sed_transp__bedload_gsd_fix_link = initialize.gsd_at_link(
            self._grid, sed_transp__bedload_gsd_fix_link, self._gsd
        )
        self._sed_transp__bedload_gsd_fix_link_id = info.fixed_links(
            self._sed_transp__bedload_gsd_fix_link
        )

        # Identify the node upstream of the outlet
        # Used for boundary conditions when updating bed elevation and bed gsd
        (
            self._out_id,
            self._upstream_out_id,
            self._outlet_links,
            self._closed_nodes,
            self._boundary_links,
        ) = info.outlet_nodes(self._grid)

        # This flag is used to activate or deactivate the bed GSD updating part
        # of the component.
        self._update_bed_surf_GSD = False

        # Activates option to store the GSD of individual layers in each node.
        self._track_stratigraphy = track_stratigraphy

        # ------------------------------------------------------------------ #
        # Gravitational diffusion correction (optional)                        #
        # ------------------------------------------------------------------ #
        self._use_bed_diffusion = use_bed_diffusion

        _valid_modes = ("nonlinear", "constant")
        if bed_diffusion_mode not in _valid_modes:
            raise ValueError(
                f"bed_diffusion_mode must be one of {_valid_modes}, "
                f"got '{bed_diffusion_mode}'."
            )
        self._bed_diffusion_mode = bed_diffusion_mode
        self._bed_diffusion_mu = float(bed_diffusion_mu)
        self._bed_diffusion_coeff = float(bed_diffusion_coeff)
        self._check_diffusion_cfl = check_diffusion_cfl
        self._check_advective_cfl = check_advective_cfl
        self._adaptive_dt = adaptive_dt

        _valid_ts = {"euler", "rk2", "implicit"}
        if time_stepping not in _valid_ts:
            raise ValueError(
                f"Unknown time_stepping {time_stepping!r}. "
                f"Valid options: {sorted(_valid_ts)}"
            )
        self._time_stepping = time_stepping

        # Threshold to deposit layers in a new subsurface layer
        self._bed_surf_new_layer_thick = bed_surf_new_layer_thick

        # When bed_surf_new_layer_thick is reached this flag is used to
        self._update_stratigraphy = False  # record and read the data
        self._update_subsurface_deposited = (
            False  # update subsurface data when there is deposition
        )
        self._update_subsurface_eroded = (
            False  # update subsurface data when there is erosion
        )

        # Sets initial values to enter into the write and read stratigraphy routine
        self._stratigraphy_cycle = 0
        self._num_cycles_to_process_strat = num_cycles_to_process_strat

        # Makes a copy of the original bed surface elevation and maps into links
        self._grid["link"]["topographic__elevation"] = (
            self._grid.map_mean_of_link_nodes_to_link(
                self._grid["node"]["topographic__elevation"]
            )
        )
        self._topogr__elev_orig_node = self._grid["node"][
            "topographic__elevation"
        ].copy()
        self._topogr__elev_orig_link = self._grid.map_mean_of_link_nodes_to_link(
            self._topogr__elev_orig_node
        )
        self._topogr__elev_subsurf_link = self._topogr__elev_orig_link.copy()
        self._bed_surf__thick_new_layer_link = np.zeros_like(
            self._topogr__elev_orig_link
        )

        # Check that parker 1990 or Wilcock and Crowe were selected when
        # tracking stratigraphy
        stratigraphy.checks_correct_equation_to_track_stratigraphy(self)

        # Creates links dictionary to store changes in stratigraphy
        stratigraphy.create_links_dictionary(self)

        # ------------------------------------------------------------------ #
        # Topology cache — fixed arrays that never change after __init__       #
        # Computed once here so that run_one_step() methods only do arithmetic #
        # ------------------------------------------------------------------ #
        self._cache_topology()

    def _cache_topology(self):
        """Pre-compute and store all topology arrays that are fixed for the
        lifetime of the component.

        This is called once at the end of ``__init__``.  The cached arrays are
        used in :meth:`shear_stress` and :meth:`update_bed_surface_gsd` every
        timestep, avoiding repeated calls to ``np.isin``, ``np.hstack``, and
        repeated attribute lookups on the Landlab grid.

        Cached attributes
        -----------------
        _topo_hlL, _topo_hlR, _topo_hl
            Left-edge, right-edge, and interior horizontal link indices.
        _topo_vlB, _topo_vlT, _topo_vl
            Bottom-edge, top-edge, and interior vertical link indices.
        _topo_horizontal_links, _topo_vertical_links
            Full horizontal/vertical link index arrays (alias for the grid
            properties, kept as plain ndarray attributes for fast access).
        _topo_du_ds_scratch
            Pre-allocated zero array of shape (n_links,) reused in
            :meth:`shear_stress` to avoid a ``np.zeros_like`` each step.
        """
        g = self._grid

        # --- Horizontal border / interior link indices ---
        # hlL: links on the left edge of the domain (flow enters from left)
        # hlR: links on the right edge
        # hl : interior horizontal links (2nd-order upwind scheme applies)
        hlL = g.links_at_node[g.nodes_at_left_edge][:, 0]
        hlR = g.links_at_node[g.nodes_at_right_edge][:, 2]
        hl_border = np.hstack((hlL, hlR))
        hl_is_border = np.isin(g.horizontal_links, hl_border)
        self._topo_hlL = hlL
        self._topo_hlR = hlR
        self._topo_hl  = g.horizontal_links[~hl_is_border]

        # --- Vertical border / interior link indices ---
        vlB = g.links_at_node[g.nodes_at_bottom_edge][:, 1]
        vlT = g.links_at_node[g.nodes_at_top_edge][:, 3]
        vl_border = np.hstack((vlB, vlT))
        vl_is_border = np.isin(g.vertical_links, vl_border)
        self._topo_vlB = vlB
        self._topo_vlT = vlT
        self._topo_vl  = g.vertical_links[~vl_is_border]

        # --- Full link-type index arrays (plain ndarray, no property overhead) ---
        self._topo_horizontal_links = g.horizontal_links
        self._topo_vertical_links   = g.vertical_links

        # --- Pre-allocated scratch arrays ---
        # _topo_du_ds_scratch: reset with du_ds[:] = 0.0 each shear_stress step.
        # np.zeros beats a pre-alloc+reset here because the OS zero-page trick
        # makes calloc nearly free; the scratch is kept only to avoid repeated
        # attribute lookups on the grid object.
        self._topo_du_ds_scratch = np.zeros(g.number_of_links)

        # _scratch_qbTdev / _scratch_qjj1dev:
        # Every link index is written before being read inside update_bed_surf_gsd
        # (hl ∪ hlL ∪ hlR and vl ∪ vlB ∪ vlT partition all links, proven in
        # _cache_topology docstring above).  np.empty skips zero-fill entirely
        # (~98× faster than np.zeros for these shapes).
        n_grains = self._gsd.shape[0] - 1   # finalised after define_initial_bed_properties
        self._scratch_qbTdev  = np.empty((g.number_of_links, 1))
        self._scratch_qjj1dev = np.empty((g.number_of_links, n_grains))

        # _scratch_area / _scratch_perimeter: used in shear_stress() hydraulic
        # radius branch. area[hl] and area[vl] together cover all links, so
        # np.empty is safe here too.
        self._scratch_area      = np.empty(g.number_of_links)
        self._scratch_perimeter = np.empty(g.number_of_links)

        # _scratch_D_link: used in bed_diffusion() constant mode. Filled fully
        # by np.full-equivalent before use.
        self._scratch_D_link = np.empty(g.number_of_links)

        # _scratch_gsd_FIexc: (n_links, n_grains) buffer for the exchange GSD
        # in update_bed_surf_gsd().  Filled via np.copyto, avoiding per-step alloc.
        self._scratch_gsd_FIexc = np.empty((g.number_of_links, n_grains))

        # Note: static GSD math caches (_topo_gs_Psi_scale_D, _topo_gs_D_eq_Psi,
        # tiled arrays, padded scratch, etc.) are bootstrapped earlier in
        # define_initial_bed_properties(), right after self._gs is finalised,
        # because calculate_DX is called there before _cache_topology() runs.

        # _scratch_cum_link / _scratch_cum_node: output buffer for np.cumsum
        # inside calculate_DX.  Using out= avoids allocating a (n, n_grains+1)
        # array on every call (~3.7 ms saved per call on a 100×100 grid).
        n_g1 = self._gsd.shape[0]   # n_grains + 1 (includes 2mm sentinel)
        self._scratch_cum_link = np.empty((g.number_of_links, n_g1))
        self._scratch_cum_node = np.empty((g.number_of_nodes, n_g1))

    def define_initial_bed_properties(self):
        """This method performs the initial setup of the bed grain size distribution properties.
        It reads the input data and populates the necessary variables.

        This configuration is only performed during the first time step. Subsequent time steps
        will utilize the bed information, which has already been calculated or updated and
        formatted appropriately.
        """
        # Adds the 2mm fraction
        self._gsd = initialize_gsd.adds_2mm_to_gsd(self._gsd)

        # Removes sand fractions in case Parker 1999 is selected
        self._gsd = initialize_gsd.remove_sand_from_gsd(
            self._gsd, self._bedload_equation
        )

        # Maps inputs to nodes - self._gs contains the equivalent grain sizes D_eq
        (
            sand_fraction,
            gs_D_equiv_freq,
            self._gs,
        ) = initialize_gsd.map_initial_bed_properties_to_nodes(
            self._gsd, self._bed_surf__gsd_loc_node
        )

        # ── Bootstrap static-math caches required by calculate_DX and
        # calculate_gsd_geo_mean_and_geo_std.  self._gsd and self._gs are
        # finalised here; _cache_topology() runs after __init__ completes.
        _g   = self._grid
        _ng  = self._gsd.shape[0] - 1
        _nl, _nn = _g.number_of_links, _g.number_of_nodes
        self._topo_gs_Psi_scale_D       = np.flip(np.log2(self._gsd[:, 0]))
        self._topo_freq_gs_list_link    = np.arange(_nl)
        self._topo_freq_gs_list_node    = np.arange(_nn)
        self._scratch_gs_freq_pad_link  = np.zeros((_nl, _ng + 1))
        self._scratch_gs_freq_pad_node  = np.zeros((_nn, _ng + 1))
        _gs_psi = np.log2(self._gs)
        self._topo_gs_D_eq_Psi          = _gs_psi
        self._topo_gs_D_eq_Psi_t_link   = np.tile(_gs_psi, (_nl, 1))
        self._topo_gs_D_eq_Psi_t_node   = np.tile(_gs_psi, (_nn, 1))
        # cumsum output buffers for calculate_DX (n_grains+1 cols)
        _ng1 = self._gsd.shape[0]
        self._scratch_cum_link = np.empty((_nl, _ng1))
        self._scratch_cum_node = np.empty((_nn, _ng1))

        median_size_D50 = self.calculate_DX(
            gs_D_equiv_freq, 0.5
        )  # Median grain size in each node
        (gs_geom_mean, gs_geo_std) = self.calculate_gsd_geo_mean_and_geo_std(
            gs_D_equiv_freq
        )

        # Bed grain sizes frequency in each node
        self._bed_surf__gsd_node = gs_D_equiv_freq
        self._bed_surf__gsd_orig_node = gs_D_equiv_freq.copy()
        self._bed_subsurf__gsd_node = gs_D_equiv_freq.copy()
        self._bed_surf__median_size_node = median_size_D50
        self._bed_surf__geom_mean_size_node = gs_geom_mean
        self._bed_surf__geo_std_size_node = gs_geo_std
        self._bed_surf__sand_fract_node = sand_fraction

        def map_mean_of_nodes_to_link(var, r_node, l_node):
            """Return the arithmetic mean of *var* at the two end-nodes of each link."""
            return 0.5 * (var[r_node] + var[l_node])

        # GSD properties is mapped from nodes onto links
        r_node = self.grid.nodes_at_link[:, 0]
        l_node = self.grid.nodes_at_link[:, 1]

        self._bed_surf__gsd_link = map_mean_of_nodes_to_link(
            gs_D_equiv_freq, r_node, l_node
        )
        self._bed_surf__median_size_link = map_mean_of_nodes_to_link(
            median_size_D50, r_node, l_node
        )
        self._bed_surf__geom_mean_size_link = map_mean_of_nodes_to_link(
            gs_geom_mean, r_node, l_node
        )
        self._bed_surf__geo_std_size_link = map_mean_of_nodes_to_link(
            gs_geo_std, r_node, l_node
        )
        self._bed_surf__sand_fract_link = map_mean_of_nodes_to_link(
            sand_fraction, r_node, l_node
        )

        self._bed_surf__gsd_orig_link = self._bed_surf__gsd_link.copy()
        self._bed_subsurf__gsd_link = self._bed_surf__gsd_link.copy()

        self._bed_surf__act_layer_thick_link = (
            2 * self.calculate_DX(self._bed_surf__gsd_link, 0.9) / 1000
        )
        self._bed_surf__act_layer_thick_prev_time_link = (
            self._bed_surf__act_layer_thick_link.copy()
        )

    def run_one_step(self):
        """The component can be divided into two parts. In the first part, all bed
        load transport and GSD calculations, including shear stress estimates,
        are conducted. In the second part, bed GSD and bed elevation can evolve.

        **First part**

        Calculates shear stress and bed load transport rates across the grid.

        For one time step, this generates the shear stress across a given
        grid by accounting for the local water depth, bed surface slope, and
        water velocity at links. Then, based on this shear stress and the local
        bed surface grain size distribution, the bed load transport rate is
        calculated at each link and mapped onto each node. Bed load grain size
        distributions are calculated when using Parker's 1990 or Wilcock and
        Crowe's 2003 equations. Meyer-Peter and Muller and Fernandez Luque and
        van Beek models will only calculate the total bed load transport.

        Outputs the following bed surface properties over time at every link
        in the input grid: geometric mean size, GSD, median size, sand fraction,
        standard deviation size.

        Also outputs the bed load GSD, bed load rate, and shear stress over
        time at every link in the input grid. The net bed load is output
        over time at every node.

        **Second Part**

        Changes grid topography. Starts at self.update_bed_elevation()

        For one time step, this erodes the grid topography according to
        Exner equation::

            (1-λp) ∂Z/∂t = - (∂qbx/∂x + ∂qby/∂y)

        Simplifying, we get::

            ∂Z/∂t = - (1 / (1-λp)) * (∂Qb/∂A)
            Z_t+1 = -(Δt * ΔQb)/(1-λp) + Z_t

        The grid field ``"topographic__elevation"`` is altered each time step.
        """
        # -- Adaptive dt (Task 2.4) -----------------------------------------
        # When adaptive_dt=True, override dt with the CFL-safe value derived
        # from the *previous* step's bedload rates (standard for explicit
        # adaptive schemes).  safety=0.9 keeps us just inside the envelope.
        if self._adaptive_dt:
            import warnings
            dt_safe = self.calc_max_stable_dt(safety=0.9)
            dt_requested = self._grid._dt
            if dt_safe < dt_requested:
                warnings.warn(
                    f"adaptive_dt: reducing dt from {dt_requested:.4g} s "
                    f"to CFL-safe {dt_safe:.4g} s.",
                    UserWarning,
                    stacklevel=2,
                )
                self._grid._dt = dt_safe

        self.shear_stress()  # Shear stress calculation
        self.bedload_equation()  # Bedload calculation
        self.calculate_net_bedload()  # Calculates bedload transport from links into nodes
        if self._use_bed_diffusion:
            self.bed_diffusion()  # Gravitational diffusion correction (optional)
        self.update_bed_elevation()  # Changes bed elevation
        stratigraphy.checks_erosion_or_deposition(self)
        stratigraphy.evolve(self)
        self.update_bed_surf_gsd()  # Changes bed surface grain size distribution
        self.update_bed_surf_properties()  # Updates gsd properties

    def shear_stress(self):
        """Compute unsteady shear stress at every link.

        Delegates to :class:`~._shear_stress.ShearStressCalculator` (Phase 3B).
        Results are stored on the component as:

        * ``_dz_ds`` — bed-slope gradient [m m⁻¹]
        * ``_u`` — link velocity [m s⁻¹]
        * ``_shear_stress`` — signed shear stress [Pa]
        * ``_surface_water__shear_stress_link`` — absolute value [Pa]

        See :class:`~._shear_stress.ShearStressCalculator` for the full
        formulation.
        """
        self._shear_calc.calculate(self)

    def bedload_equation(self):
        """Dispatch to the active bedload equation via the Phase-3A registry.

        Calls ``self._bedload_eq.calculate(self)`` and stores the results on
        the component.  For equations that resolve grain fractions (Parker 1990,
        Wilcock & Crowe 2003) the returned GSD array is stored in
        ``_sed_transp__bedload_gsd_link``; for total-load equations it is
        left unchanged from the previous step.
        """
        qb, qb_gsd = self._bedload_eq.calculate(self)
        self._sed_transp__bedload_rate_link = qb
        if qb_gsd is not None:
            self._sed_transp__bedload_gsd_link = qb_gsd

    def calculate_net_bedload(self):
        """Calculates the net volumetric bedload coming from all links (m2/s)
        onto nodes (m3/s).

        This method takes the volumetric bedload entering and exiting through a
        face and determines the net volumetric bedload on a given node.
        """

        qb_x = (
            np.sum(
                self._sed_transp__bedload_rate_link[self._grid.links_at_node[:, [0, 2]]]
                * self._normal[:, [0, 2]],
                axis=1,
            )
            * self._grid.dy
        )
        qb_y = (
            np.sum(
                self._sed_transp__bedload_rate_link[self._grid.links_at_node[:, [1, 3]]]
                * self._normal[:, [1, 3]],
                axis=1,
            )
            * self._grid.dx
        )

        self._sed_transp__net_bedload_node = qb_x + qb_y

        ## At the boundary, there is no exiting link, so we assume a zero flux
        # exiting. This assumption is overridden in the Exner equation, where a
        # zero gradient boundary condition can be used.
        self._sed_transp__net_bedload_node[self._grid.boundary_nodes] = 0

    # ------------------------------------------------------------------ #
    # CFL helpers — Tasks 2.1 / 2.3                                        #
    # ------------------------------------------------------------------ #

    def calc_max_stable_dt_advective(self, safety=0.5):
        r"""Return the CFL-limited time step for the advective Exner term.

        Von Neumann stability for the explicit upwind Exner equation:

        .. math::

            C = \frac{|q_b|_{\max} \, \Delta t}{(1-\lambda_p) \, \Delta x_{\min}} \le 1

        Rearranging:

        .. math::

            \Delta t_{\text{safe}} = \text{safety} \times
            \frac{(1-\lambda_p) \, \Delta x_{\min}}{|q_b|_{\max}}

        Parameters
        ----------
        safety : float, optional
            Safety factor (default 0.5).  Values < 1 guard against
            non-linearity and 2-D effects.

        Returns
        -------
        float
            Maximum stable ``dt`` [s] for the advective term, or
            ``np.inf`` when transport is everywhere zero.
        """
        qb_max = np.abs(self._sed_transp__bedload_rate_link).max()
        if qb_max == 0.0:
            return np.inf
        dx_min = min(self._grid.dx, self._grid.dy)
        return safety * (1.0 - self._lambda_p) * dx_min / qb_max

    def calc_max_stable_dt_diffusive(self, safety=0.5):
        """Return the CFL-limited time step for the diffusive Exner term.

        Uses the parabolic stability criterion::

            dt_safe = safety × (1-λp) × dx_min² / (2 × D_max)

        Returns ``np.inf`` when diffusion is disabled or D_max is zero.

        Parameters
        ----------
        safety : float, optional
            Safety factor (default 0.5).

        Returns
        -------
        float
            Maximum stable ``dt`` [s] for the diffusive term.
        """
        if not self._use_bed_diffusion:
            return np.inf
        if self._bed_diffusion_mode == "nonlinear":
            D_max = (
                np.abs(self._sed_transp__bedload_rate_link)
                / self._bed_diffusion_mu
            ).max()
        else:
            D_max = self._bed_diffusion_coeff
        if D_max == 0.0:
            return np.inf
        dx_min = min(self._grid.dx, self._grid.dy)
        return safety * (1.0 - self._lambda_p) * dx_min**2 / (2.0 * D_max)

    def calc_max_stable_dt(self, safety=0.5):
        """Return the combined CFL-limited time step.

        Returns ``min(dt_advective, dt_diffusive)`` both evaluated with
        ``safety``.  Use this to set ``dt`` before calling
        :meth:`run_one_step` when ``adaptive_dt=False``.

        Parameters
        ----------
        safety : float, optional
            Safety factor (default 0.5).

        Returns
        -------
        float
            Maximum stable ``dt`` [s], or ``np.inf`` when all transport
            is zero and diffusion is disabled.
        """
        return min(
            self.calc_max_stable_dt_advective(safety=safety),
            self.calc_max_stable_dt_diffusive(safety=safety),
        )

    # ------------------------------------------------------------------ #
    # Exner RHS + time integration (Phase 4A)                              #
    # ------------------------------------------------------------------ #

    def _exner_rhs(self) -> np.ndarray:
        r"""Return the advective Exner RHS: dz/dt [m/s] at every node.

        Reads the *current* net bedload divergence
        (``_sed_transp__net_bedload_node``) and returns the rate of elevation
        change per unit time due to sediment flux divergence:

        .. math::

            \frac{\partial z}{\partial t} =
            -\frac{\nabla \cdot q_b}{(1 - \lambda_p)}

        The diffusive correction term (when ``use_bed_diffusion=True``) is
        **not** included here; it is added separately in
        :meth:`update_bed_elevation` so that both the Euler and RK2 paths can
        share this method.

        Returns
        -------
        ndarray, shape (n_nodes,)
            ``dz/dt`` at every node [m s⁻¹].
        """
        area = self._grid.dx * self._grid.dy
        return -self._sed_transp__net_bedload_node / ((1 - self._lambda_p) * area)

    def _compute_transport_jacobian(self, eps: float = 1e-4):
        """Compute the finite-difference Jacobian of the Exner RHS w.r.t. z.

        Returns the sparse matrix

        .. math::

            J_{ij} = \\frac{\\partial (\\dot{z}_i)}{\\partial z_j}
                   \\approx \\frac{f(z + \\varepsilon e_j)_i - f(z)_i}{\\varepsilon}

        where :math:`f(z)` is the advective Exner RHS vector
        (``_exner_rhs()``), evaluated at the current bed elevation.

        Only interior (core) nodes are perturbed; boundary nodes have
        zero columns in the Jacobian (their elevation is fixed by BCs).

        The matrix is returned in **CSR** format so it can be fed directly
        to :func:`scipy.sparse.linalg.spsolve` in Phase 5.2.

        Parameters
        ----------
        eps : float, optional
            Finite-difference step size [m].  Default ``1e-4`` m (0.1 mm)
            gives good accuracy for typical bedload sensitivity without
            triggering transport threshold nonlinearities.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_nodes, n_nodes)
            Jacobian of the Exner RHS.  Columns corresponding to boundary
            nodes are structurally zero.

        Notes
        -----
        Cost is O(n_core_nodes) transport evaluations.  For a 100×100 grid
        with ~9 800 core nodes this is expensive (~seconds); Phase 5.2
        exploits the sparsity pattern to avoid the full column-by-column
        sweep in the implicit solver.
        """
        import scipy.sparse as sp

        z_node = self._grid.at_node["topographic__elevation"]
        z_link = self._grid.at_link["topographic__elevation"]
        n = self._grid.number_of_nodes
        core = self._grid.core_nodes   # interior nodes only

        # Base RHS at current state (already computed; read it directly)
        f0 = self._exner_rhs().copy()

        # Save state that will be temporarily modified
        z_save      = z_node.copy()
        z_link_save = z_link.copy()
        qb_save     = self._sed_transp__bedload_rate_link.copy()
        nb_save     = self._sed_transp__net_bedload_node.copy()
        tau_save    = self._shear_stress.copy()
        tau_abs_save = self._surface_water__shear_stress_link.copy()

        rows, cols, vals = [], [], []

        for j in core:
            # Perturb node j
            z_node[j] += eps
            self._grid.at_link["topographic__elevation"][:] = (
                self._grid.map_mean_of_link_nodes_to_link(z_node)
            )

            # Recompute transport chain (hydraulics held fixed)
            self.shear_stress()
            self.bedload_equation()
            self.calculate_net_bedload()
            fj = self._exner_rhs()

            # Finite-difference column j
            df = (fj - f0) / eps

            # Store only non-negligible entries (threshold: 1e-16)
            nz = np.where(np.abs(df) > 1e-16)[0]
            rows.extend(nz.tolist())
            cols.extend([j] * len(nz))
            vals.extend(df[nz].tolist())

            # Restore node j
            z_node[j] = z_save[j]

        # Restore full state
        self._grid.at_node["topographic__elevation"][:] = z_save
        self._grid.at_link["topographic__elevation"][:] = z_link_save
        self._sed_transp__bedload_rate_link[:]  = qb_save
        self._sed_transp__net_bedload_node[:]   = nb_save
        self._shear_stress[:]                   = tau_save
        self._surface_water__shear_stress_link[:] = tau_abs_save

        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(n, n), dtype=float
        )

    def _assemble_implicit_system(self, J, dt: float):
        """Assemble the sparse linear system for the semi-implicit Exner step.

        The linearised implicit Exner equation is:

        .. math::

            \\left(I - \\frac{\\Delta t}{1-\\lambda_p} \\cdot J\\right)
            \\delta z = \\Delta t \\cdot f(z^n)

        where :math:`J = \\partial f / \\partial z` is the transport Jacobian
        (from :meth:`_compute_transport_jacobian`), :math:`f(z^n)` is the
        explicit Exner RHS, and :math:`\\delta z = z^{n+1} - z^n`.

        Boundary rows are replaced by identity rows so that the solver
        returns :math:`\\delta z = 0` at every fixed/closed/outlet node.

        Parameters
        ----------
        J : scipy.sparse.csr_matrix
            Transport Jacobian, shape ``(n, n)``.
        dt : float
            Time step [s].

        Returns
        -------
        A : scipy.sparse.csr_matrix
            Left-hand side matrix, shape ``(n, n)``.
        b : ndarray, shape (n,)
            Right-hand side vector [m].
        """
        import scipy.sparse as sp

        n = self._grid.number_of_nodes

        # LHS: I - dt/(1-lp) * J
        eye = sp.eye(n, format="csr")
        lhs = eye - (dt / (1.0 - self._lambda_p)) * J

        # RHS: dt * f(z^n)
        rhs = dt * self._exner_rhs()

        # Enforce zero delta-z at boundary nodes by replacing their rows
        # with identity rows (lhs[i,i]=1, lhs[i,j≠i]=0, rhs[i]=0)
        bnd = np.concatenate([
            np.asarray(self._out_id).ravel(),
            np.asarray(self._bed_surf__elev_fix_node_id).ravel(),
            np.asarray(self._closed_nodes).ravel(),
        ])
        bnd = np.unique(bnd).astype(int)

        lhs = lhs.tolil()
        for i in bnd:
            lhs[i, :] = 0.0
            lhs[i, i] = 1.0
        lhs = lhs.tocsr()
        rhs[bnd] = 0.0

        return lhs, rhs

    def _apply_elevation_bcs(self, z: np.ndarray, z0: np.ndarray,
                              dz: np.ndarray) -> None:
        """Apply all boundary conditions to elevation array *z* in-place.

        Sets outlet, fixed, and closed nodes to their correct post-step
        values and writes the result back to the grid.

        Parameters
        ----------
        z : ndarray
            Elevation array (modified in-place).
        z0 : ndarray
            Elevation at the start of the step (unchanged copy).
        dz : ndarray
            Total elevation change applied this step (used for the outlet
            zero-gradient BC).
        """
        z[self._out_id]                  = z0[self._out_id]
        z[self._bed_surf__elev_fix_node_id] = z0[self._bed_surf__elev_fix_node_id]
        z[self._closed_nodes]            = z0[self._closed_nodes]

        if self._outlet_boundary_condition == "zeroGradient":
            dz_outlet = dz[self._upstream_out_id]
        else:  # "fixedValue"
            dz_outlet = 0
        z[self._out_id] = z0[self._out_id] + dz_outlet

        self._grid["link"]["topographic__elevation"] = (
            self._grid.map_mean_of_link_nodes_to_link(z)
        )
        self._grid["node"]["topographic__elevation"] = z

    def update_bed_elevation(self):
        """Applies the Exner equation and boundary conditions to predict
        the change in bed surface elevation.

        The time-integration scheme is controlled by ``time_stepping``:

        ``"euler"`` (default) — first-order forward Euler::

            (1 - λp) ∂z/∂t = - ∇·q_b
            Z_{t+1} = Z_t - Δt · ΔQb / ((1 - λp) · A)

        ``"rk2"`` — Heun's predictor-corrector (second-order)::

            k1 = dz/dt evaluated at z^n (current bedload rates)
            z* = z^n + dt · k1          (predictor, with BCs)
            qb* = bedload(z*)            (one extra transport evaluation)
            k2 = dz/dt evaluated at z*
            z^{n+1} = z^n + dt/2 · (k1 + k2)  (corrector, with BCs)

        ``"implicit"`` — linearised backward Euler (unconditionally stable)::

            (I - dt · J) · δz = dt · f(z^n)
            z^{n+1} = z^n + δz

            where J = ∂f/∂z is the transport Jacobian (finite-difference),
            f(z^n) is the Exner RHS at the current elevation, and the linear
            system is solved with scipy.sparse.linalg.spsolve.

        When ``use_bed_diffusion=True``, the gravitational diffusion correction
        is added to the corrector step only (operator splitting).

        The same boundary conditions (fixed elevation, closed nodes, outlet)
        are applied after every stage.
        """
        z  = self._grid["node"]["topographic__elevation"]
        z0 = z.copy()
        dt = self._grid._dt

        # -- Advective CFL check (Phase 2.2) --------------------------------
        if self._check_advective_cfl and not self._adaptive_dt:
            import warnings
            dt_safe = self.calc_max_stable_dt_advective(safety=1.0)
            if dt_safe < dt:
                warnings.warn(
                    f"Advective Exner CFL = {dt/dt_safe:.2f} > 1 — solution "
                    f"may be unstable. Reduce dt to ≤ {dt_safe:.4g} s "
                    f"(current dt = {dt:.4g} s).",
                    UserWarning,
                    stacklevel=2,
                )

        if self._time_stepping == "euler":
            # ── Forward Euler ────────────────────────────────────────────── #
            dz = dt * self._exner_rhs()
            if self._use_bed_diffusion:
                dz += self._bed_surf__diffusive_dz_node
            z += dz
            self._apply_elevation_bcs(z, z0, dz)

        elif self._time_stepping == "rk2":
            # ── Heun's method (RK2) ──────────────────────────────────────── #
            # Stage 1 — k1 at z^n (bedload already computed for this step)
            k1 = self._exner_rhs()          # dz/dt at z^n  [m/s]

            # Predictor: advance z to z* and apply BCs
            z_star = z0 + dt * k1
            dz_pred = dt * k1
            self._apply_elevation_bcs(z_star, z0, dz_pred)

            # Stage 2 — k2: recompute bedload at predictor elevation z*
            # Write z* into the grid so shear_stress / bedload can read it
            self._grid["node"]["topographic__elevation"][:] = z_star
            self._grid["link"]["topographic__elevation"][:] = (
                self._grid.map_mean_of_link_nodes_to_link(z_star)
            )
            self.shear_stress()
            self.bedload_equation()
            self.calculate_net_bedload()
            k2 = self._exner_rhs()          # dz/dt at z*  [m/s]

            # Corrector: Heun average
            dz = dt / 2.0 * (k1 + k2)
            if self._use_bed_diffusion:
                dz += self._bed_surf__diffusive_dz_node
            z_new = z0 + dz
            self._apply_elevation_bcs(z_new, z0, dz)

        else:
            # ── Linearised backward Euler (implicit) ─────────────────────── #
            dz = self._solve_implicit_exner(dt)
            if self._use_bed_diffusion:
                dz += self._bed_surf__diffusive_dz_node
            z_new = z0 + dz
            self._apply_elevation_bcs(z_new, z0, dz)

    def _solve_implicit_exner(self, dt: float) -> np.ndarray:
        """Assemble and solve the linearised implicit Exner system.

        Implements the linearised backward-Euler update:

        .. math::

            (I - \\Delta t \\, J) \\, \\delta z = \\Delta t \\, f(z^n)

        where :math:`J = \\partial f / \\partial z` is the transport Jacobian
        (:meth:`_compute_transport_jacobian`), :math:`f(z^n)` is the current
        Exner RHS (:meth:`_exner_rhs`), and :math:`\\delta z = z^{n+1} - z^n`.

        Boundary conditions are enforced by replacing each boundary row of the
        system with the identity equation :math:`\\delta z_b = 0`, which fixes
        all outlet, closed, and Dirichlet nodes.  The outlet zero-gradient BC
        is applied afterwards in :meth:`_apply_elevation_bcs`.

        The resulting sparse linear system is solved with
        :func:`scipy.sparse.linalg.spsolve` (direct LU factorisation).

        Parameters
        ----------
        dt : float
            Time step [s].

        Returns
        -------
        ndarray, shape (n_nodes,)
            Elevation change :math:`\\delta z` [m] for this step.

        Raises
        ------
        RuntimeError
            If ``scipy.sparse.linalg.spsolve`` fails (singular matrix).

        Notes
        -----
        The implicit scheme is unconditionally stable for the linearised
        Exner equation, allowing :math:`\\Delta t` far beyond the explicit
        CFL limit.  Accuracy degrades as :math:`O(\\Delta t)` (first-order
        in time), so use large ``dt`` only when the morphodynamic response
        is smooth.
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        n   = self._grid.number_of_nodes
        f0  = self._exner_rhs()                        # RHS at z^n
        J   = self._compute_transport_jacobian()       # (n × n) CSR

        # Assemble system: A · δz = rhs
        # A = I - dt · J,  rhs = dt · f0
        eye = sp.eye(n, format="csr")
        A   = eye - dt * J
        rhs = dt * f0

        # Enforce δz = 0 at every boundary node (all BC types):
        # fixed, closed, and outlet.  Outlet zero-gradient is handled later.
        bnd = np.union1d(
            self._grid.boundary_nodes,
            np.union1d(
                self._bed_surf__elev_fix_node_id,
                self._closed_nodes,
            )
        ).astype(int)

        # Convert to LIL for efficient row assignment, then back to CSR
        A_lil = A.tolil()
        for b in bnd:
            A_lil.rows[b] = [b]
            A_lil.data[b] = [1.0]
            rhs[b]        = 0.0
        A = A_lil.tocsr()

        try:
            dz = spla.spsolve(A, rhs)
        except Exception as exc:
            raise RuntimeError(
                "Implicit Exner solver failed. "
                "The transport Jacobian may be ill-conditioned at this dt. "
                f"scipy error: {exc}"
            ) from exc

        return dz

    def bed_diffusion(self):
        """Computes the gravitational diffusion correction to bed elevation.

        Evaluates the diffusive term of the extended Exner equation::

            (1 - λp) ∂z/∂t |_diff = ∇·(D ∇z)

        and stores the result in ``_bed_surf__diffusive_dz_node`` [m], ready
        to be added inside :func:`update_bed_elevation`.

        The computation follows the native Landlab link → node framework:

        1. Bed-slope gradient at every link:
           ``grad_z = calc_grad_at_link(z)``  [m m⁻¹]

        2. Diffusion coefficient at every link [m² s⁻¹]:

           * *nonlinear* mode: ``D = |qb| / μ``
             (Talmon et al. 1995 — larger transport → stronger smoothing)
           * *constant* mode: ``D = bed_diffusion_coeff`` everywhere

        3. Diffusive sediment flux at links [m² s⁻¹]:
           ``q_diff = D · grad_z``

        4. Divergence of that flux at nodes [m s⁻¹]:
           ``div_diff = calc_flux_div_at_node(q_diff)``

        5. Elevation change contribution [m]:
           ``dz_diff = dt · div_diff / (1 - λp)``

        The diffusion coefficient is set to zero at boundary links so that no
        sediment flux crosses the domain boundary through the diffusive term.

        A CFL warning is raised when ``check_diffusion_cfl=True`` and the
        diffusive CFL number ``Δt · 2·D_max / ((1-λp)·Δx²)`` exceeds 1.

        Notes
        -----
        Physical basis
            On a sloping bed, gravity deflects individual grain trajectories
            downslope, adding a component of transport proportional to the
            local bed gradient.  Integrating over the grain-size distribution
            and the stochastic hop-length distribution recovers the ∇·(D∇z)
            form (Engelund 1974; Furbish et al. 2012).

        Grid artefacts
            On a raster grid, sediment flux is constrained to the four
            cardinal link directions, producing staircase profiles along
            diagonal channels.  The diffusion term acts isotropically and
            suppresses this artefact without altering the net mass budget.
        """
        import warnings

        z = self._grid.at_node["topographic__elevation"]
        dt = self._grid._dt

        # -- 1. Bed-slope gradient at links [m/m] ---------------------------
        grad_z = self._grid.calc_grad_at_link(z)

        # -- 2. Diffusion coefficient at links [m²/s] -----------------------
        if self._bed_diffusion_mode == "nonlinear":
            # D = |qb| / mu  (Talmon et al. 1995)
            # qb is already available at links after bedload_equation()
            D_link = (
                np.abs(self._sed_transp__bedload_rate_link) / self._bed_diffusion_mu
            )
        else:
            # Constant diffusion coefficient
            # Reuse scratch — fill every element so np.empty is safe
            self._scratch_D_link[:] = self._bed_diffusion_coeff
            D_link = self._scratch_D_link

        # No diffusive flux crosses the domain boundary
        D_link[self._boundary_links] = 0.0

        # -- 3. Optional CFL stability check --------------------------------
        if self._check_diffusion_cfl:
            D_max = D_link.max()
            if D_max > 0.0:
                dx_min = min(self._grid.dx, self._grid.dy)
                dt_cfl = (1.0 - self._lambda_p) * dx_min**2 / (2.0 * D_max)
                cfl = dt / dt_cfl
                if cfl > 1.0:
                    warnings.warn(
                        f"Diffusive CFL = {cfl:.2f} > 1 — solution may be "
                        f"unstable. Reduce dt to ≤ {dt_cfl:.4g} s "
                        f"(current dt = {dt:.4g} s), or decrease "
                        f"bed_diffusion_mu to lower D.",
                        UserWarning,
                        stacklevel=2,
                    )

        # -- 4. Diffusive flux at links and its divergence at nodes ---------
        #   q_diff  [m²/s]:  D · ∇z
        #   div_diff [m/s]:  ∇·(D · ∇z)  — Landlab accounts for cell area
        q_diff = D_link * grad_z
        div_diff = self._grid.calc_flux_div_at_node(q_diff)

        # -- 5. Elevation change [m] ----------------------------------------
        #   (1 - λp) dz/dt|_diff = div_diff
        #   => dz_diff = dt · div_diff / (1 - λp)
        self._bed_surf__diffusive_dz_node = (
            dt / (1.0 - self._lambda_p) * div_diff
        )

    def update_bed_surf_gsd(self):
        """Update the bed surface GSD via the Toro-Escobar fractional Exner equation.

        Delegates to :class:`~._gsd_evolver.ToroEscobarEvolver` (Phase 3C).
        Results written to the component:

        * ``_bed_surf__gsd_link`` — updated surface GSD at links
        * ``_bed_surf__gsd_node`` — updated surface GSD at nodes
        * ``_topogr__elev_orig_link`` — updated reference elevation

        See :class:`~._gsd_evolver.ToroEscobarEvolver` for the full algorithm.
        """
        self._gsd_evolver.evolve(self)

    def update_bed_surf_properties(self):
        """Calculates the updated GSD properties"""

        self._bed_surf__median_size_link = self.calculate_DX(
            self._bed_surf__gsd_link, 0.5
        )  # Median grain size in each node
        (
            self._bed_surf__geom_mean_size_link,
            self._bed_surf__geo_std_size_link,
        ) = self.calculate_gsd_geo_mean_and_geo_std(self._bed_surf__gsd_link)
        self._bed_surf__act_layer_thick_prev_time_link = (
            self._bed_surf__act_layer_thick_link.copy()
        )
        self._bed_surf__act_layer_thick_link = (
            2 * self.calculate_DX(self._bed_surf__gsd_link, 0.9) / 1000
        )

        self._bed_surf__median_size_node = self.calculate_DX(
            self._bed_surf__gsd_node, 0.5
        )
        (
            self._bed_surf__geom_mean_size_node,
            self._bed_surf__geo_std_size_node,
        ) = self.calculate_gsd_geo_mean_and_geo_std(self._bed_surf__gsd_node)

    def calculate_DX(self, gs_D_equiv_freq, fX):
        """Calculate the grain size corresponding to any fraction.
        For example, 50%, which is the median_size_D50. In that case use fX = 0.5

        This method takes the user specified fraction, from 0 to 1, and outputs
        the corresponding grain size in nodes or links, which is considered
        by taking the size of the passed argument"""

        # Use cached static arrays (computed once in _cache_topology).
        # Avoids: np.log2, np.flip, np.arange, np.hstack+zeros every call.
        # out= on np.cumsum avoids allocating a (n, n_grains+1) array each call.
        if gs_D_equiv_freq.shape[0] == self.grid.number_of_links:
            freq_gs_list = self._topo_freq_gs_list_link
            gs_freq      = self._scratch_gs_freq_pad_link
            cum_out      = self._scratch_cum_link
        else:
            freq_gs_list = self._topo_freq_gs_list_node
            gs_freq      = self._scratch_gs_freq_pad_node
            cum_out      = self._scratch_cum_node

        gs_Psi_scale_D = self._topo_gs_Psi_scale_D

        # Fill padded scratch in-place (last column stays zero permanently)
        n_grains = gs_D_equiv_freq.shape[1]
        gs_freq[:, :n_grains] = gs_D_equiv_freq

        # cumsum on reversed view → cum_out; avoids allocating the output array
        np.cumsum(gs_freq[:, ::-1], axis=1, out=cum_out)
        i0 = np.argmin(cum_out <= fX, axis=1) - 1
        i1 = np.argmax(cum_out > fX, axis=1)

        gs_Psi_scale_DX = gs_Psi_scale_D[i0] + (
            (gs_Psi_scale_D[i1] - gs_Psi_scale_D[i0])
            / (
                cum_out[freq_gs_list, i1]
                - cum_out[freq_gs_list, i0]
            )
        ) * (fX - cum_out[freq_gs_list, i0])

        return 2**gs_Psi_scale_DX

    def calculate_gsd_geo_mean_and_geo_std(self, gs_D_equiv_freq):
        """Calculates the geometric mean and standard deviation in links or nodes
        depending on the input"""

        # Use cached Psi arrays (computed once in _cache_topology).
        # Avoids: np.log2, np.tile (64 µs on 100×100), np.reshape every call.
        if gs_D_equiv_freq.shape[0] == self.grid.number_of_links:
            gs_D_eq_Psi_tiled = self._topo_gs_D_eq_Psi_t_link  # (n_links, n_grains)
        else:
            gs_D_eq_Psi_tiled = self._topo_gs_D_eq_Psi_t_node  # (n_nodes, n_grains)

        gs_D_eq_Psi = self._topo_gs_D_eq_Psi               # (n_grains,)

        # Geometric mean: exp2( sum(f_i * psi_i) )
        gs_D_eq_Psi_mean = np.sum(gs_D_equiv_freq * gs_D_eq_Psi, axis=1)  # (n,)
        gs_geom_mean = 2**gs_D_eq_Psi_mean

        # Geometric std: exp2( sqrt( sum(f_i * (psi_i - psi_mean)^2) ) )
        # psi_mean needs shape (n,1) for broadcast against (n, n_grains)
        gs_geo_std = 2 ** np.sqrt(
            np.sum(
                ((gs_D_eq_Psi_tiled - gs_D_eq_Psi_mean[:, np.newaxis]) ** 2)
                * gs_D_equiv_freq,
                axis=1,
            )
        )

        return gs_geom_mean, gs_geo_std

    def stratigraphy_write_evolution(self):
        """Writes the stratigraphy time evolution into a csv file

        The example below is a shorter version of the one used in _stratigraphy.py

        Examples
        --------

        As per usual, we define import the required libraries and create a grid
        and configure the mandatory fields.

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import RiverBedDynamics
        >>> import os

        >>> grid = RasterModelGrid((8, 3), xy_spacing=100)

        >>> grid.at_node["topographic__elevation"] = [
        ...     [1.12, 1.00, 1.12],
        ...     [1.12, 1.01, 1.12],
        ...     [1.12, 1.01, 1.12],
        ...     [1.12, 1.01, 1.12],
        ...     [1.12, 1.01, 1.12],
        ...     [1.12, 1.01, 1.12],
        ...     [1.12, 1.01, 1.12],
        ...     [1.12, 1.12, 1.12],
        ... ]

        >>> grid.at_node["surface_water__depth"] = np.full(grid.number_of_nodes, 0.40)
        >>> grid.at_link["surface_water__depth"] = np.full(grid.number_of_links, 0.40)
        >>> grid.at_link["surface_water__velocity"] = np.full(
        ...     grid.number_of_links, 0.40
        ... )
        >>> grid.set_watershed_boundary_condition(
        ...     grid.at_node["topographic__elevation"]
        ... )
        >>> gsd = [[8, 100], [4, 90], [2, 0]]

        >>> fixed_nodes = np.zeros(grid.number_of_nodes)
        >>> fixed_nodes[[1, 4]] = 1

        >>> fixed_bed_gsd_nodes = np.zeros(grid.number_of_nodes)
        >>> fixed_bed_gsd_nodes[[1, 4]] = 1

        >>> qb = np.full(grid.number_of_links, 0.0)
        >>> qb[[28, 33]] = -0.002

        >>> rbd = RiverBedDynamics(
        ...     grid,
        ...     gsd=gsd,
        ...     bedload_equation="Parker1990",
        ...     outlet_boundary_condition="fixedValue",
        ...     bed_surf__elev_fix_node=fixed_nodes,
        ...     bed_surf__gsd_fix_node=fixed_bed_gsd_nodes,
        ...     sed_transp__bedload_rate_fix_link=qb,
        ...     track_stratigraphy=True,
        ...     bed_surf_new_layer_thick=0.02,
        ...     num_cycles_to_process_strat=2,
        ... )

        We will run the model for 1299 s. This is exactly the time required for the
        first link to reach a deposition of 2 cm (Notice bed_surf_new_layer_thick=0.02).
        The evolution can is written to a file called Stratigraphy_evolution.csv

        >>> for t in range(1300):
        ...     rbd._current_t = t
        ...     rbd.run_one_step()
        ...

        >>> rbd.stratigraphy_write_evolution()

        In this case we will delete the file to keep Landlab clean

        >>> os.remove("Stratigraphy_evolution.csv")

        """
        stratigraphy.write_evolution(self)
