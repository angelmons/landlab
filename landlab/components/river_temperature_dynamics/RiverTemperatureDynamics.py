"""Landlab component that calculates temporal and spatial changes in river
temperature.

.. codeauthor:: Angel Monsalve

Examples
--------

>>> import numpy as np
>>> import copy
>>> from landlab import RasterModelGrid
>>> from landlab.components import RiverTemperatureDynamics
>>> from landlab import imshow_grid
>>> from landlab.grid.mappers import map_mean_of_link_nodes_to_link

Create a grid on which to calculate river temperature

>>> grid = RasterModelGrid((5, 5))

The grid will need some data to run the RiverTemperatureDynamics component.
To check the names of the fields that provide input use the *input_var_names*
class property.

>>> RiverTemperatureDynamics.input_var_names
('surface_water__depth',
 'surface_water__temperature',
 'surface_water__velocity',
 'topographic__elevation')

Create fields of data for each of these input variables. When running a
complete simulation some of these variables will be created by the flow model.
Notice that surface water depth and velocity are required at links. However,
specifying these variables at nodes is easier and then we can map the fields
onto links. By doing so, we don't have to deal with links numbering. When this
component is coupled to OverlandFlow there is no need to map fields because it
is done automatically within the component.

We create the bed surface grain size (GSD) distribution location. We assume
that there are two different GSD within the watershed (labeled as 0 and 1)

>>> grid.at_node['surface_water__temperature'] = np.array([
... 20, 21., 21., 21., 20,
... 20, 21., 21., 21., 20,
... 20, 21., 21., 21., 20,
... 20, 21., 21., 21., 20,
... 20, 21., 21., 21., 20,])

Now we create the topography data

>>> grid.at_node['topographic__elevation'] = np.array([
... 1.07, 1.06, 1.00, 1.06, 1.07,
... 1.08, 1.07, 1.03, 1.07, 1.08,
... 1.09, 1.08, 1.07, 1.08, 1.09,
... 1.09, 1.09, 1.08, 1.09, 1.09,
... 1.09, 1.09, 1.09, 1.09, 1.09,])

and set the boundary conditions

>>> grid.set_watershed_boundary_condition(grid.at_node['topographic__elevation'])

And check the node status

>>> grid.status_at_node
array([4, 4, 1, 4, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 4, 4,
       4, 4], dtype=uint8)

Which tell us that there is one outlet located on the 3rd node

the topography data can be display using
>>> imshow_grid(grid,'topographic__elevation')

Now we add some water into the watershed. In this case is specified in nodes

>>> grid.at_node['surface_water__depth'] = np.array([
... 0.102, 0.102, 0.102, 0.102, 0.102,
... 0.102, 0.102, 0.102, 0.102, 0.102,
... 0.102, 0.102, 0.102, 0.102, 0.102,
... 0.102, 0.102, 0.102, 0.102, 0.102,
... 0.102, 0.102, 0.102, 0.102, 0.102,])

There are other most efficient ways to fill the 'surface_water__depth', but for
demonstration purposes we show the extended version. A more efficient way to
set the previous field could be:
grid.at_node['surface_water__depth'] = np.full(grid.number_of_nodes,0.102)

Now, we give the water a velocity.

>>> grid.at_node['surface_water__velocity'] = np.array([
... 0.25, 0.25, 0.25, 0.25, 0.25,
... 0.25, 0.25, 0.25, 0.25, 0.25,
... 0.25, 0.25, 0.25, 0.25, 0.25,
... 0.25, 0.25, 0.25, 0.25, 0.25,
... 0.25, 0.25, 0.25, 0.25, 0.25,])

Note that in this example, we are attempting to specify a vector at a node
using a single value. This is done intentionally to emphasize the process.
The component will interpret this as the vector's magnitude, and, given its
location in the grid, it will manifest different components. When using
OverlandFlow, there is no need to specify a velocity because it is a
byproduct of the component.

By default, when creating our grid we used a spacing of 1 m in the x and y
directions. Therefore, the discharge is 0.0255 m3/s. Discharge is always in
units of m3/s.

Now we map nodes into links when it is required

>>> grid['link']['surface_water__depth'] = \
    map_mean_of_link_nodes_to_link(grid,'surface_water__depth')
>>> grid['link']['surface_water__velocity'] = \
    map_mean_of_link_nodes_to_link(grid,'surface_water__velocity')

Provide a time step, usually an output of OverlandFlow, but it can be
overridden with a custom value.

>>> timeStep = 1 # time step in seconds

Instantiate the `RiverTemperatureDynamics` component to work on the grid, and run it.

>>> RTD = RiverTemperatureDynamics(grid)
>>> RTD.run_one_step()

After instantiating RiverTemperatureDynamics, new fields have been added to the grid.
Use the *output_var_names* property to see the names of the fields
that have been changed.

>>> RTD.output_var_names
('bed_surface__geometric_mean_size',
 'bed_surface__geometric_standard_deviation_size',
 'bed_surface__grain_size_distribution',
 'bed_surface__median_size',
 'bed_surface__sand_fraction',
 'sediment_transport__bedload_grain_size_distribution',
 'sediment_transport__bedload_rate',
 'sediment_transport__net_bedload',
 'surface_water__shear_stress')

The `sediment_transport__bedload_rate` field is defined at nodes.

>>> RBD.var_loc('sediment_transport__net_bedload')
'node'

>>> grid.at_node['sediment_transport__net_bedload'] # doctest: +NORMALIZE_WHITESPACE
array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         1.59750963e-03,  -4.79298187e-03,   1.59750963e-03,
         0.00000000e+00,   0.00000000e+00,   1.50988732e-07,
         1.59720766e-03,   1.50988732e-07,   0.00000000e+00,
         0.00000000e+00,   3.01977464e-07,  -1.50988732e-07,
         3.01977464e-07,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
         0.00000000e+00])

The 'surface_water__shear_stress' field is defined at links.

>>> RBD.var_loc('surface_water__shear_stress')
'link'

>>> grid.at_link['surface_water__shear_stress'] # doctest: +NORMALIZE_WHITESPACE
array([  0.      ,  60.016698,  60.016698,   0.      ,   0.      ,
         0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
        40.011132,  40.011132,   0.      ,  10.002783,  10.002783,
        40.011132,  10.002783,  10.002783,   0.      ,  10.002783,
        10.002783,   0.      ,   0.      ,  10.002783,  10.002783,
        10.002783,   0.      ,   0.      ,  10.002783,  10.002783,
         0.      ,   0.      ,   0.      ,   0.      ,   0.      ,
         0.      ,   0.      ,   0.      ,   0.      ,   0.      ])

Considering the link upstream the watershed exit, link Id 15, we can obtain the
bed load transport rate

>>> grid.at_link['sediment_transport__bedload_rate'][15]
-0.0015976606223666004

Therefore, the bed load transport rate according to Parker 1991 surface-based
equation is 1.598 * 10^-3 m2/s. Negative means that is going in the negative
Y direction

The GSD at this place is:

>>> grid.at_link['sediment_transport__bedload_grain_size_distribution'][15]
array([ 0.47501858,  0.52498142])

Which in cummulative percentage is equivalent to
D mm    % Finer
32      100.000
16      52.498
8       0.000

Grain sizes are always given in mm.
We can also check the bed load grain size distribution in all links

>>> grid.at_link['sediment_transport__bedload_grain_size_distribution']
array([[ 0.        ,  0.        ],
       [ 0.48479122,  0.51520878],
       [ 0.48479122,  0.51520878],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.47501858,  0.52498142],
       [ 0.47501858,  0.52498142],
       [ 0.        ,  0.        ],
       [ 0.54055384,  0.45944616],
       [ 0.28225526,  0.71774474],
       [ 0.47501858,  0.52498142],
       [ 0.28225526,  0.71774474],
       [ 0.54055384,  0.45944616],
       [ 0.        ,  0.        ],
       [ 0.28225526,  0.71774474],
       [ 0.28225526,  0.71774474],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.28225526,  0.71774474],
       [ 0.28225526,  0.71774474],
       [ 0.28225526,  0.71774474],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.28225526,  0.71774474],
       [ 0.28225526,  0.71774474],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ],
       [ 0.        ,  0.        ]])

Zeros indicate that there is no sediment transport of that grain size
at that location.

After the flow acted on the bed and sediment transport occured we can check
the new topographic elevation field

>>> grid.at_node['topographic__elevation']  # doctest: +NORMALIZE_WHITESPACE
array([ 1.07      ,  1.06      ,  1.00737382,  1.06      ,  1.07      ,
        1.08      ,  1.06754229,  1.03737382,  1.06754229,  1.08      ,
        1.09      ,  1.07999977,  1.06754276,  1.07999977,  1.09      ,
        1.09      ,  1.08999954,  1.08000023,  1.08999954,  1.09      ,
        1.09      ,  1.09      ,  1.09      ,  1.09      ,  1.09      ])
"""
import copy
import os
import shutil
import time

import numpy as np
import pandas as pd
import scipy.constants
from scipy.interpolate import interp1d

from landlab import Component, FieldError


class RiverTemperatureDynamics(Component):

    """Landlab component that predicts the evolution of the river temperature
    considering changes in atmospheric conditions.

    To estimate temporal and spatial changes in river temperature properties,
    this component accounts for the sources and sinks of heat. Time-varying 
    hydraulic variables are obtained from a surface flow, for instance, 
    OverlandFlow.

    The primary method of this class is :func:`run_one_step`.

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    None

    **Additional References**

    """

    _name = "RiverTemperatureDynamics"

    _unit_agnostic = False

    _info = {
        "air__temperature": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "deg C",
            "mapping": "node",
            "doc": "Air temperature over the time step",
        },
        "air__relative_humidity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "percentage",
            "mapping": "node",
            "doc": "Air relative humidity over the time step",
        },
        "air__velocity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m/s",
            "mapping": "node",
            "doc": "Air velocity (wind speed) 2m-above-soil over the time step",
        },
        "atmospheric__net_heat_exchange": {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "W/m^2",
            "mapping": "node",
            "doc": "Atmospheric net heat exchange over the time step",
        },        
        "radiation__incoming_longwave_flux": {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "W/m^2",
            "mapping": "node",
            "doc": "total incident longwave radiation over the time step",
        },
        "radiation__incoming_shortwave_flux": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "W/m^2",
            "mapping": "node",
            "doc": "total incident shortwave radiation over the time step",
        },
        "radiation__emitted_longwave_flux": {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "W/m^2",
            "mapping": "node",
            "doc": "total emitted longwave radiation over the time step",
        },
        "radiation__net_shortwave_flux": {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "W/m^2",
            "mapping": "cell",
            "doc": "net incident shortwave radiation over the time step",
        },
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
            "doc": "Average velocity of the surface water",
        },
        "surface_water__temperature": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "deg C",
            "mapping": "node",
            "doc": "Water temperature, assumed constant on the water column",
        },
    }

    def __init__(
        self,
        grid,
        rho=1000,  # Sets the fluid density (kg/m**3).
        shade_factor=0.2,  # Shade factor [0-1] H_sn = 0.97 H_si (1-shade_factor).
        dt=1,  # Sets the time step (s). When coupled to
        # OverlandFlow, this value is adjusted dynamically.
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
        grid :  RasterModelGrid
            A Landlab grid.
        gsd :   Array float, mandatory
            Grain size distribution. Must contain as many GSDs as there are
            different indexes in GSDLocation.
        rho :   float, optional
            Fluid density. Defaults to the density of water at 1000 kg/m^3.
        shade_factor: float, optional   
            Shade factor, Default value is 0.2, must be between 0 and 1. It is used
            to correct incoming shortwave radiation according to:
            H_sn = 0.97 H_si (1-shade_factor).    
        dt: float, optional
            Time step in seconds. When this component is coupled to a flow model,
            it is dynamically updated.

        """
        super().__init__(grid)

        self._Ca = 0.6 # Brundt's coefficient
        self._e = 0.97 # Emmisivity - For calculating emitted long wave radiation 
        self._g = scipy.constants.g  # Acceleration due to gravity (m/s**2).
        self._rho = rho # Water density (kg/m^3)
        self._shade_factor = shade_factor
        self._stefan_boltzmann_const=0.0000000567 # ( Wm^(-2) K^(-4) )
        self._Rl = 0.03 # Reflective coefficient) 
        self._dt = dt

        # Creating optional grid fields at time zero- If the grid field was not
        # defined before instantiation it will create it and fill all values
        # with zeros, otherwise, it will print a message and use the given values
        try:
            self._grid["node"]["air__temperature"] = grid.add_zeros(
                "air__temperature",
                at="node",
                units=self._info["air__temperature"]["units"],
            )
        except FieldError:
            print("'air__temperature' at links - Initialized")

        try:
            self._grid["node"]["air__relative_humidity"] = grid.add_zeros(
                "air__relative_humidity",
                at="node",
                units=self._info["air__relative_humidity"]["units"],
            )
        except FieldError:
            print("'air__relative_humidity' at links - Initialized")

        try:
            self._grid["node"]["air__velocity"] = grid.add_zeros(
                "air__velocity",
                at="node",
                units=self._info["air__velocity"]["units"],
            )
        except FieldError:
            print("'air__velocity' at links - Initialized")

        try:
            self._grid["node"]["radiation__incoming_shortwave_flux"] = grid.add_zeros(
                "radiation__incoming_shortwave_flux",
                at="node",
                units=self._info["radiation__incoming_shortwave_flux"]["units"],
            )
        except FieldError:
            print("'radiation__incoming_shortwave_flux' at nodes - Initialized")

        try:
            self._grid["node"]["surface_water__temperature"] = grid.add_zeros(
                "surface_water__temperature",
                at="node",
                units=self._info["surface_water__temperature"]["units"],
            )
        except FieldError:
            print("'surface_water__temperature' at nodes - Initialized")
        
        # Creates the "air__vapor_pressure" field
        self._grid["node"]["air__vapor_pressure"] = np.full(
            grid.number_of_nodes,0)

        # Creates the "atmospheric__net_heat_exchange" field
        self._grid["node"]["atmospheric__net_heat_exchange"] = np.full(
            grid.number_of_nodes,0)
        
        # Creates the "radiation__incoming_longwave_flux" field
        self._grid["node"]["radiation__incoming_longwave_flux"] = np.full(
            grid.number_of_nodes,0)

        # Creates the "radiation__emitted_longwave_flux" field
        self._grid["node"]["radiation__emitted_longwave_flux"] = np.full(
            grid.number_of_nodes,0)

        # Creates the "radiation__net_shortwave_flux" field
        self._grid["node"]["radiation__net_shortwave_flux"] = np.full(
            grid.number_of_nodes,0)

        # Creates the "vapor_pressure__saturation" field
        self._grid["node"]["vapor_pressure__saturation"] = np.full(
            grid.number_of_nodes,0)

        # Creates the "heat_flux__evaporation" field
        self._grid["node"]["heat_flux__evaporation"] = np.full(
            grid.number_of_nodes,0)

        # Creates the "heat_flux__convective" field
        self._grid["node"]["heat_flux__convective"] = np.full(
            grid.number_of_nodes,0)
                
        # Define faces normal vector
        self._normal = -(self._grid.link_dirs_at_node)

        # Define water specific heat values as a function of temperature
        # Columns are: T [c] and c_w  [Jkg^(-1) K^(-1)].
        # # Then it creates an interpolation function
        specific_heat = np.array([
            [0.01,4.2174],
            [10,4.1910],
            [20,4.1570], 
            [25,4.1379],
            [30,4.1175],
            [40,4.0737],
            [50,4.0264]])

        self._specific_heat_f = interp1d(specific_heat[:, 0],specific_heat[:, 1])

        self.define_node_location()
    
    def define_node_location(self):
        # These are used to calculate fluxes in and out at nodes faces

        self._r_l = self._grid.links_at_node[:,0] # right link
        self._l_l = self._grid.links_at_node[:,2] # left link
        self._u_l = self._grid.links_at_node[:,1] # upper link
        self._d_l = self._grid.links_at_node[:,3] # lower link

        self._r_n = self._grid.adjacent_nodes_at_node[:,0] # right node
        self._l_n = self._grid.adjacent_nodes_at_node[:,2] # left node
        self._u_n = self._grid.adjacent_nodes_at_node[:,1] # upper node
        self._d_n = self._grid.adjacent_nodes_at_node[:,3] # lower node

    def temperature_advection(self):
        """ Calculates temperature advection """
        V = self._grid["link"]["surface_water__velocity"]
        h = self._grid["link"]["surface_water__depth"]
        dx , dy = self._grid.dx , self._grid.dy
        T = self._grid["node"]["surface_water__temperature"]
        T0 = self._grid["node"]["surface_water__temperature"]
        n = self._normal
        dt = self._dt

        # flux at right face - X direction
        u_r = V[self._r_l]
        A_r = h[self._r_l] * dy
        T_r = T
        (Id,) = np.where(u_r<0)
        T_r[Id] = T[self._r_n][Id]
        f_r = n[:,0] * u_r * A_r * T_r
        #print('f_r\n');print(f_r)

        # flux at left face - X direction
        u_l = V[self._l_l]
        A_l = h[self._l_l] * dy
        T_l = T[self._l_n]
        (Id,) = np.where(u_l<0)
        T_l[Id] = T[Id]
        f_l = n[:,2] * u_l * A_l * T_l
        #print('f_l\n');print(f_l)

        # flux at upper face - Y direction
        u_u = V[self._u_l]
        A_u = h[self._u_l] * dx
        T_u = T
        (Id,) = np.where(u_u<0)
        T_u[Id] = T[self._u_n][Id]
        f_u = n[:,1] * u_u * A_u * T_u
        #print('f_u\n');print(f_u)

        # flux at lower face - Y direction
        u_d = V[self._d_l]
        A_d = h[self._d_l] * dx
        T_d = T[self._d_n]
        (Id,) = np.where(u_d<0)
        T_d[Id] = T[Id]
        f_d = n[:,3] * u_d * A_d * T_d
        #print('f_d\n');print(f_d)
               
        self._grid["node"]["surface_water__temperature"] = T - \
            (dt/dx) * (f_r + f_l) * ( 1 / (0.5 * (A_r + A_l) ) ) - \
            (dt/dy) * (f_u + f_d) * ( 1 / (0.5 * (A_u + A_d) ) )

        self._grid["node"]["surface_water__temperature"][self._grid.boundary_nodes] = \
            T0[self._grid.boundary_nodes]
        a = (dt/dx) * (f_r + f_l) * ( 1 / (0.5 * (A_r + A_l) ) )
        b = (dt/dy) * (f_u + f_d) * ( 1 / (0.5 * (A_u + A_d) ) )
        print('k1\n');print(a)
        print('k2\n');print(b)
    def run_one_step(self):
        """ Calculates the water temperature across the grid.

        For one time step, it generates the atmospheric net heat exchange and
        the temperature change associated with this flux across a given
        grid 
        ..."""
        self.water_specific_heat()
        self.temperature_advection()
        self.atmospheric_net_heat_exchange()
        #DT_atm = self._grid["node"]["atmospheric__net_heat_exchange"] / \
        #    (self._grid["node"]["surface_water__specific_heat"] * self._rho)
        #print(DT_atm)

    def atmospheric_net_heat_exchange(self):
        """ The atmospheric net heat exchange H_atm [W/m^2] to or from the water column
        (positive for incoming fluxes) is:
        H_atm = H_sn + H_an + H_br + H_e + H_c
        
        The incoming net short-wave radiation flux H_sn is:
        H_sn = 0.97 H_si (1.0-SF)
        H_si is the total incoming radiation in [W/m^2]
        which is corrected with albedo (3%) and a user-provided shade factor SF [0 to 1] 
        (0.2 by default). 

        Incoming long wave radiation H_an is:
        H_an =σ (T_a + 273.15)^4 (Ca + 0.084900481√(e_a ))(1 - R_l)
        σ=5.67x10^(-8) Wm^(-2) K^(-4) is the Stefan-Boltzman constant
        T_a [°C] is the air temperature, 
        e_a [kPa] is the air vapor pressure, e_a = 0.01 RH e_v
        e_v [kPa] is the saturation of vapor pressure (Tetens equation)
        e_v = 0.61078 e^((17.27 T_a)/(T_a+273.3))
        RH [%] is the relative humidity 
        Ca=0.6 Brundt's coefficient
        R_l=0.03 is the reflective coefficient

        The emitted long wave radiation H_br is :
        H_br = -σ ϵ ( T + 273.15)^4,
        T is the water temperature
        ϵ=0.97 is the emissivity

        The evaporation (latent) heat flux H_e is :
        H_e = - ( 9.2 + 0.46 v_w^2 )( e_v-e_a),
        where v_w [m/s] is the 2m-above-soil wind speed.

        The convective (sensible) heat flux H_c is:
        H_c = - 0.47 ( 9.2 + 0.46 v_w^2 )(T-T_a).
        """
        # H_sn = 0.97 H_si (1.0-SF)
        self._grid["node"]["radiation__net_shortwave_flux"] = 0.97 * \
            self._grid["node"]["radiation__incoming_shortwave_flux"] * \
            (1 - self._shade_factor)
        
        # e_v =0.61078 e^((17.27 T_a)/(T_a+273.3))
        self._grid["node"]["vapor_pressure__saturation"] = 0.61078 * \
            np.exp( (17.27 * self._grid["node"]["air__temperature"]) / \
                   (self._grid["node"]["air__temperature"] + 273.3) )
        # e_a = 0.01 RH e_v
        self._grid["node"]["air__vapor_pressure"] = 0.01 * \
            self._grid["node"]["air__relative_humidity"] * \
            self._grid["node"]["vapor_pressure__saturation"]

        # H_an = σ (T_a + 273.15)^4 (Ca + 0.084900481√(e_a ))(1 - R_l)
        self._grid["node"]["radiation__incoming_longwave_flux"] = \
            self._stefan_boltzmann_const * \
            (self._grid["node"]["air__temperature"] + 273.15 )**4 * \
            (self._Ca + 0.084900481 * \
             np.sqrt(self._grid["node"]["air__vapor_pressure"])) * \
             (1 - self._Rl)
        
        # H_br = -σ ϵ ( T + 273.15)^4
        self._grid["node"]["radiation__emitted_longwave_flux"] = \
        - (self._stefan_boltzmann_const * self._e) * \
        ( self._grid["node"]["surface_water__temperature"] + 273.15)**4

        # H_e = - ( 9.2 + 0.46 v_w^2 )( e_v-e_a)
        self._grid["node"]["heat_flux__evaporation"] = \
            - (9.2 + 0.46 * self._grid["node"]["air__velocity"]**2) * \
            (self._grid["node"]["vapor_pressure__saturation"] - 
             self._grid["node"]["air__vapor_pressure"])

        # H_c = - 0.47 ( 9.2 + 0.46 v_w^2 )(T-T_a)
        self._grid["node"]["heat_flux__convective"] = -0.47 * \
            ( 9.2 + 0.46 * self._grid["node"]["air__velocity"]**2) * \
            (self._grid["node"]["surface_water__temperature"] - 
             self._grid["node"]["air__temperature"])

        # H_atm = H_sn + H_an + H_br + H_e + H_c
        self._grid["node"]["atmospheric__net_heat_exchange"] = \
            self._grid["node"]["radiation__net_shortwave_flux"] + \
            self._grid["node"]["radiation__incoming_longwave_flux"] + \
            self._grid["node"]["radiation__emitted_longwave_flux"] + \
            self._grid["node"]["heat_flux__evaporation"] + \
            self._grid["node"]["heat_flux__convective"]
            
    def water_specific_heat(self):
        """ Calculates the water specific heat in [kJ/(kg K)] 
        Water temperature (input) is specified in degrees """

        self._grid["node"]["surface_water__specific_heat"] = \
            self._specific_heat_f(self._grid["node"]["surface_water__temperature"])
