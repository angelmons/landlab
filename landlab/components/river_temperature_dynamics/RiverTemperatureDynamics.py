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

>>> grid = RasterModelGrid((10, 15))

The grid will need some data to run the RiverTemperatureDynamics component.
To check the names of the fields that provide input use the *input_var_names*
class property.

>>> RiverTemperatureDynamics.input_var_names
('surface_water__depth',
 'surface_water__dispersion_coefficient',
 'surface_water__temperature',
 'surface_water__velocity')

Create fields of data for each of these input variables. When running a
complete simulation some of these variables could be created by the flow model.
Notice that surface water depth and velocity are required at links. However,
specifying these variables at nodes is easier and then we can map the fields
onto links. By doing so, we don't have to deal with links numbering. 

In this example we set uniform and constant values for depth, dispersion coeffiente, and
velocity

>>> grid.at_node['surface_water__depth'] = np.full(grid.number_of_nodes,0.5,dtype='float')
>>> grid.at_node['surface_water__dispersion_coefficient'] = np.full(grid.number_of_nodes,0.1,dtype='float')
>>> grid.at_node['surface_water__velocity'] = np.full(grid.number_of_nodes,0.25,dtype='float')

Note that in this example, we are attempting to specify a vector at a node
using a single value. This is done intentionally to emphasize the process.
The component will interpret this as the vector's magnitude, and, given its
location in the grid, it will manifest different components.

Now we set the water temperature, at one point the temperature is 2 degrees higher than the rest
of the domain. 

>>> grid.at_node['surface_water__temperature'] = np.full(grid.number_of_nodes,10.0,dtype='float')
>>> grid.at_node['surface_water__temperature'][17] = 12

the temperature data can be display using

>>> imshow_grid(grid,'surface_water__temperature',vmin = 10, vmax =12)

Let's set some of the atmospheric conditions too:

>>> grid.at_node["radiation__incoming_shortwave_flux"] = np.full(grid.number_of_nodes,250.0,dtype='float')
>>> grid.at_node["air__temperature"] = np.full(grid.number_of_nodes,15.0,dtype='float')
>>> grid.at_node["air__relative_humidity"] = np.full(grid.number_of_nodes,5,dtype='float')
>>> grid.at_node["air__velocity"] = np.full(grid.number_of_nodes,2,dtype='float')

Now we map nodes into links when it is required

>>> grid['link']['surface_water__depth'] = \
    map_mean_of_link_nodes_to_link(grid,'surface_water__depth')
>>> grid['link']['surface_water__velocity'] = \
    map_mean_of_link_nodes_to_link(grid,'surface_water__velocity')

Provide a time step

>>> dt = 0.25 # time step in seconds

Instantiate the `RiverTemperatureDynamics` component to work on the grid, and run it.

>>> RTD = RiverTemperatureDynamics(grid,dt=dt)

And we let it run for 10 seconds to observe changes in water temperature
we kept the temperature at 12 degrees at the 17th node.
>>> t = 0
>>> while t < 10:
>>>     grid.at_node['surface_water__temperature'][17] = 12
>>>     RTD.run_one_step()
>>>     t += dt 

the updated temperature data can be display using

>>> imshow_grid(grid,'surface_water__temperature',vmin = 10, vmax =12)

"""
import copy
import numpy as np
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
            "mapping": "node",
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
        "surface_water__dispersion_coefficient": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m2/s",
            "mapping": "node",
            "doc": "Dispersion coefficient of the surface water",
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
        """Calculates the evolution of a river temperature based on the advection - 
        dispersion equation. Also, account for heat exchange with the atmosphere. 
        An external flow hydraulics solver, such as OverlandFlow, is required to
        predict flow variables in time and space.

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

        self._Ca = 0.6  # Brundt's coefficient
        self._e = 0.97  # Emmisivity - For calculating emitted long wave radiation 
        self._g = scipy.constants.g  # Acceleration due to gravity (m/s**2).
        self._rho = rho # Water density (kg/m^3)
        self._shade_factor = shade_factor
        self._stefan_boltzmann_const=0.0000000567 # ( Wm^(-2) K^(-4) )
        self._Rl = 0.03 # Reflective coefficient) 
        self._dt = dt   # time step

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
            print("'air__temperature' at nodes - Initialized")

        try:
            self._grid["node"]["air__relative_humidity"] = grid.add_zeros(
                "air__relative_humidity",
                at="node",
                units=self._info["air__relative_humidity"]["units"],
            )
        except FieldError:
            print("'air__relative_humidity' at nodes - Initialized")

        try:
            self._grid["node"]["air__velocity"] = grid.add_zeros(
                "air__velocity",
                at="node",
                units=self._info["air__velocity"]["units"],
            )
        except FieldError:
            print("'air__velocity' at nodes - Initialized")

        try:
            self._grid["node"]["radiation__incoming_shortwave_flux"] = grid.add_zeros(
                "radiation__incoming_shortwave_flux",
                at="node",
                units=self._info["radiation__incoming_shortwave_flux"]["units"],
            )
        except FieldError:
            print("'radiation__incoming_shortwave_flux' at nodes - Initialized")

        try:
            self._grid["node"]["surface_water__depth"] = \
                grid.add_zeros(
                "surface_water__depth",
                at="link",
                units=self._info["surface_water__depth"]["units"],
            )
        except FieldError:
            print("'surface_water__depth' at links - Initialized")


        try:
            self._grid["node"]["surface_water__dispersion_coefficient"] = \
                grid.add_zeros(
                "surface_water__dispersion_coefficient",
                at="node",
                units=self._info["surface_water__dispersion_coefficient"]["units"],
            )
        except FieldError:
            print("'surface_water__dispersion_coefficient' at nodes - Initialized")

        try:
            self._grid["node"]["surface_water__velocity"] = grid.add_zeros(
                "surface_water__velocity",
                at="link",
                units=self._info["surface_water__velocity"]["units"],
            )
        except FieldError:
            print("'surface_water__velocity' at links - Initialized")


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
        # Columns are: T [c] and c_w  [kg^(-1) K^(-1)].
        # # Then it creates an interpolation function
        specific_heat = np.array([
            [0.01,4217.4],
            [10,4191.0],
            [20,4157.0], 
            [25,4137.9],
            [30,4117.5],
            [40,4073.7],
            [50,4026.4]])

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
 
    def water_specific_heat(self):
        """ Calculates the water specific heat in [kJ/(kg K)] 
        Water temperature (input) is specified in degrees """

        self._grid["node"]["surface_water__specific_heat"] = \
            self._specific_heat_f(self._grid["node"]["surface_water__temperature"])
        
    def temperature_advection(self):
        """ Calculates temperature advection """

        V = self._grid["link"]["surface_water__velocity"]
        h = self._grid["link"]["surface_water__depth"]
        dx , dy = self._grid.dx , self._grid.dy
        T  = copy.deepcopy(self._grid["node"]["surface_water__temperature"])
        n = self._normal
        dt = self._dt

        # flux at right face - X direction
        u_r = V[self._r_l]
        A_r = h[self._r_l] * dy
        T_r = copy.deepcopy(T)
        T_r[u_r<0] = T[self._r_n][u_r<0]       
        f_r = n[:,0] * u_r * A_r * T_r

        # flux at left face - X direction
        u_l = V[self._l_l]
        A_l = h[self._l_l] * dy
        T_l = T[self._l_n]          
        T_l[u_l<0] = T[u_l<0]
        f_l = n[:,2] * u_l * A_l * T_l

        # flux at upper face - Y direction
        u_u = V[self._u_l]
        A_u = h[self._u_l] * dx
        T_u = copy.deepcopy(T)
        T_u[u_u<0] = T[self._u_n][u_u<0]
        f_u = n[:,1] * u_u * A_u * T_u

        # flux at lower face - Y direction     
        u_d = V[self._d_l]
        A_d = h[self._d_l] * dx
        T_d = T[self._d_n]
        T_d[u_d<0] = T[u_d<0]
        f_d = n[:,3] * u_d * A_d * T_d
               
        self._grid["node"]["surface_water__temperature"] = T - \
            (dt/dx) * (f_r + f_l) * ( 1 / (0.5 * (A_r + A_l) ) ) - \
            (dt/dy) * (f_u + f_d) * ( 1 / (0.5 * (A_u + A_d) ) )

        self._grid["node"]["surface_water__temperature"][self._grid.boundary_nodes] = \
            T[self._grid.boundary_nodes]
        
    def temperature_dispersion(self):
        """ Calculates temperature dispersion """

        D = self._grid["node"]["surface_water__dispersion_coefficient"]
        dx , dy = self._grid.dx , self._grid.dy
        Ax = self._grid["node"]["surface_water__depth"] * dy
        Ay = self._grid["node"]["surface_water__depth"] * dx
        dt = self._dt
        T  = copy.deepcopy(self._grid["node"]["surface_water__temperature"])

        # Right face flux
        D = 0.5 * (D[self._r_n] + D)
        A = 0.5 * (Ax[self._r_n] + Ax)
        T_diff = np.abs(T[self._r_n] - T)
        T_0 = (D * A * T_diff) * (1/dx)
        f_in_r = np.zeros_like(T)
        f_out_r = np.zeros_like(T)

        id = T[self._r_n] > T
        f_in_r[id] = T_0[id]    # Flow from x+i to x
        f_out_r[~id] = T_0[~id] # Flow from x to x+1        
        
        # Left face flux
        D = 0.5 * (D[self._l_n] + D)
        A = 0.5 * (Ax[self._l_n] + Ax)
        T_diff = np.abs(T[self._l_n] - T)
        T_0 = (D * A * T_diff) * (1/dx)
        f_in_l = np.zeros_like(T)
        f_out_l = np.zeros_like(T)

        id = T[self._l_n] > T
        f_in_l[id] = T_0[id]    # Flow from x-i to x
        f_out_l[~id] = T_0[~id] # Flow from x to x-i

        # Upper face flux
        D = 0.5 * (D[self._u_n] + D)
        A = 0.5 * (Ax[self._u_n] + Ax)
        T_diff = np.abs(T[self._u_n] - T)
        T_0 = (D * A * T_diff) * (1/dx)
        f_in_u = np.zeros_like(T)
        f_out_u = np.zeros_like(T)

        id = T[self._u_n] > T 
        f_in_u[id] = T_0[id]    # Flow from y+i to x
        f_out_u[~id] = T_0[~id] # Flow from y to y+1

        # Lower face flux
        D = 0.5 * (D[self._d_n] + D)
        A = 0.5 * (Ax[self._d_n] + Ax)
        T_diff = np.abs(T[self._d_n] - T)
        T_0 = (D * A * T_diff) * (1/dx)
        f_in_d = np.zeros_like(T)
        f_out_d = np.zeros_like(T)

        id = T[self._d_n] > T
        f_in_d[id] = T_0[id]    # Flow from y-i to y
        f_out_d[~id] = T_0[~id] # Flow from y to y-i

        f_in_x = f_in_l + f_in_r
        f_out_x = f_out_l + f_out_r
        f_in_y = f_in_u + f_in_d
        f_out_y = f_out_u + f_out_d

        self._grid["node"]["surface_water__temperature"] = T - \
            (dt/dx) * (f_out_x - f_in_x) * ( 1 / Ax ) - \
            (dt/dy) * (f_out_y - f_in_y) * ( 1 / Ay )
        
        self._grid["node"]["surface_water__temperature"][self._grid.boundary_nodes] = \
            T[self._grid.boundary_nodes]

    def atmospheric_net_heat_exchange(self):
        """ The atmospheric net heat exchange H_atm [W/m^2] to or from the water 
        column (positive for incoming fluxes) is:
        H_atm = H_sn + H_an + H_br + H_e + H_c
        """

        """ Incoming net short-wave radiation flux H_sn
        # H_sn = 0.97 H_si (1.0-SF)
        # H_si is the total incoming radiation in [W/m^2]
        # corrected with albedo (3%) and a user-provided shade factor SF [0 to 1]
        # (0.2 by default). """
        self._grid["node"]["radiation__net_shortwave_flux"] = 0.97 * \
            self._grid["node"]["radiation__incoming_shortwave_flux"] * \
            (1 - self._shade_factor)

        # Saturation of vapor pressure (Tetens equation)
        # e_v =0.61078 e^((17.27 T_a)/(T_a+273.3)) [kPa]
        # T_a [°C] is the air temperature
        self._grid["node"]["vapor_pressure__saturation"] = 0.61078 * \
            np.exp( (17.27 * self._grid["node"]["air__temperature"]) / \
                   (self._grid["node"]["air__temperature"] + 273.3) )

        # air vapor pressure
        # e_a = 0.01 RH e_v [kPa]
        # RH [%] is the relative humidity 
        self._grid["node"]["air__vapor_pressure"] = 0.01 * \
            self._grid["node"]["air__relative_humidity"] * \
            self._grid["node"]["vapor_pressure__saturation"]

        # Incoming long wave radiation
        # σ=5.67x10^(-8) [W/[m^2 K^4]] is the Stefan-Boltzman constant
        # R_l=0.03 is the reflective coefficient
        # Ca=0.6 Brundt's coefficient
        # H_an = σ (T_a + 273.15)^4 (Ca + 0.084900481√(e_a ))(1 - R_l)
        self._grid["node"]["radiation__incoming_longwave_flux"] = \
            self._stefan_boltzmann_const * \
            (self._grid["node"]["air__temperature"] + 273.15 )**4 * \
            (self._Ca + 0.084900481 * \
             np.sqrt(self._grid["node"]["air__vapor_pressure"])) * \
             (1 - self._Rl)
        
        # Emitted long wave radiation
        # H_br = -σ ϵ ( T + 273.15)^4
        # ϵ=0.97 is the emissivity
        # T is the water temperature
        self._grid["node"]["radiation__emitted_longwave_flux"] = \
        - (self._stefan_boltzmann_const * self._e) * \
        ( self._grid["node"]["surface_water__temperature"] + 273.15)**4

        # Evaporation (latent) heat flux
        # H_e = - ( 9.2 + 0.46 v_w^2 )( e_v-e_a)
        # v_w [m/s] is the 2m-above-soil wind speed.
        self._grid["node"]["heat_flux__evaporation"] = \
            - (9.2 + 0.46 * self._grid["node"]["air__velocity"]**2) * \
            (self._grid["node"]["vapor_pressure__saturation"] - 
             self._grid["node"]["air__vapor_pressure"])

        # Convective (sensible) heat flux
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
        
        DT = self._grid["node"]["atmospheric__net_heat_exchange"] / \
            (self._grid["node"]["surface_water__depth"] * \
             self._grid["node"]["surface_water__specific_heat"] * \
             self._rho)
        
        self._grid["node"]["surface_water__temperature"] = \
            self._grid["node"]["surface_water__temperature"] + \
            self._dt * DT

    def run_one_step(self):
        """ Calculates the water temperature across the grid.

        For one time step, it calculates the advection, dispersion and atmospheric net
        heat exchange of water temperatura across a given grid  """

        self.water_specific_heat()
        self.temperature_advection()
        self.temperature_dispersion()
        self.atmospheric_net_heat_exchange()     
