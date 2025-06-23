---
title: 'RiverFlowDynamics v1.0: A Landlab component for computing two-dimensional river flow dynamics'

tags:
  - Landlab
  - Python
  - Shallow water equations
  - Saint-Venant equation
  - river

authors:
  - name: Angel Monsalve
    orcid: 0000-0002-7369-1602
    equal-contrib: true
    affiliation: 1
    corresponding: true
  - name: Sebastian Bernal
    orcid: 0009-0006-7758-3648
    equal-contrib: true
    affiliation: 1
  - name: Oscar Link
    orcid: 0000-0002-2188-6504
    equal-contrib: true
    affiliation: 2

affiliations:
 - name: Center for Ecohydraulics Research, Civil and Environmental Engineering, University of Idaho, Boise, ID, USA
   index: 1
 - name: Departamento de Ingeniería Civil, Universidad de Concepción, Concepción, Chile
   index: 2

date: 05 September 2024

bibliography: paper.bib

---
# Summary

RiverFlowDynamics enables researchers to simulate how water flows through rivers, streams, and flood plains using realistic physics. The software can predict water depths, flow velocities, and how water dynamics evolve over complex terrain, making it valuable for flood risk assessment, environmental studies, and water resource management. Numerical modeling of surface water flow is a critical tool in hydrology, hydraulics, and environmental science. These models play a crucial role in predicting and analyzing flow patterns in rivers, flood plains, and coastal areas, informing decisions in water resource management, flood risk assessment, and ecosystem conservation. This paper introduces a novel two-dimensional flow model, RiverFlowDynamics, developed as a component of the Landlab Python Package [@hobley:2017;@barnhart:2020;@Hutton:2020;@Hutton:2020], designed to simulate the behavior of rivers and streams under various flow conditions over natural and artificial topography.

RiverFlowDynamics is founded on the depth-averaged Saint-Venant equations, also known as the shallow water equations [@casulli1990semi;@casulli_semi-implicit_1999], which capture the essential dynamics of free-surface flows by assuming that vertical accelerations are negligible compared to horizontal ones. For the numerical solution, the component employs the finite volume method chosen for its robustness, capacity to handle complex geometries, and inherent conservation properties [@andersson2011computational;@fletcher2012computational], with the model's structure allowing for easy parallelization to enable simulations of extensive river networks or large coastal areas.

A key feature of the model is its semi-implicit and semi-Lagrangian representation that treats water surface elevation and velocity implicitly while handling advective terms explicitly, allowing for larger time steps and enhanced computational efficiency [@robert1985semi;@robert_stable_1981]. The semi-Lagrangian advection scheme tracks fluid particles backwards along their flow lines using Pollock's semi-analytical method [@pollock1988semianalytical], which assumes linear velocity variation within each grid cell and relaxes the Courant-Friedrichs-Lewy condition [@staniforth1991semi;@bates1982multiply]. Source terms primarily account for bottom friction using the Manning-Chezy formula [@he2017numerical;@brufau2000two], while wind stress and Coriolis effects are considered negligible for river applications. The model features robust handling of dry/wet cell transitions using the method of @casulli1992semi, which automatically determines wet and dry cell faces based on local flow conditions, and implements boundary conditions including Dirichlet conditions for inlet boundaries and gradient-based or radiation-based conditions for open boundaries to minimize artificial reflections. The model has been validated through a comprehensive test suite including analytical solutions, numerical stability tests, sensitivity analyses, mass conservation checks, and boundary condition validations, with the current implementation supporting uniform rectilinear grids and planned extensions for non-uniform rectilinear and curvilinear meshes.

RiverFlowDynamics, as a full 2D flow model, offers several advantages over simpler flow models, including traditional overland flow models available in Landlab [@adams2017landlab;@de2012improving]. While overland flow models typically focus on shallow sheet flow and often use simplified equations like the kinematic wave approximation, RiverFlowDynamics solves the complete depth-averaged Saint-Venant equations. This approach allows for a more comprehensive representation of complex flow dynamics, including subcritical and supercritical flows, hydraulic jumps, and intricate channel-floodplain interactions. The model's ability to capture these phenomena makes it superior in scenarios involving rapid flood propagation in urban areas, detailed floodplain mapping, or the analysis of complex river morphodynamics. Furthermore, the semi-Lagrangian scheme employed in RiverFlowDynamics provides enhanced stability and accuracy for advection-dominated flows, a critical advantage when modeling high-velocity currents or steep terrain where simpler models might fail. This makes RiverFlowDynamics particularly well-suited for applications in mountainous regions, urban flood modeling, from small-scale stream dynamics to large-scale flood simulations, or any situation where capturing the full range of flow regimes and their transitions is crucial for accurate predictions. The accessibility of this code within the Landlab framework will make it easier for future users to modify and contribute to its continual evolution.

Source code for RiverFlowDynamics is available as part of the Landlab Python package (v2.7.0 and later). Tutorials and examples are available in the notebook gallery of the main Landlab documentation and repository.

# Statement of need

RiverFlowDynamics is a Python-based 2D flow model developed as a component of the Landlab framework, addressing a critical gap in the modeling of complex river systems and flood dynamics. Prior to RiverFlowDynamics, Landlab lacked a comprehensive 2D flow model capable of handling fully advective-dominated problems, particularly in rivers with complex topographies. This limitation hindered accurate simulations of diverse flow regimes and transitions crucial for advanced hydrological and environmental studies. The model's integration into Landlab's component framework enables future coupling with sediment transport components to simulate morphodynamic processes and assess impacts on aquatic habitat and riverine vegetation dynamics under changing flow conditions.

Compared to existing hydraulic modeling software such as TELEMAC and Delft3D, RiverFlowDynamics offers unique advantages through its native Python implementation and integration with the Landlab framework, providing enhanced interoperability with other Earth surface process models and the broader scientific Python ecosystem while maintaining comparable numerical accuracy.. RiverFlowDynamics solves the complete depth-averaged Saint-Venant equations, offering a significant advancement over existing Landlab components that typically use simplified equations like the kinematic wave approximation. This approach enables the model to capture complex flow dynamics, including subcritical and supercritical flows, hydraulic jumps, and intricate channel-floodplain interactions. The model's capabilities make it particularly valuable for a wide range of applications, from small-scale stream dynamics to large-scale flood simulations. It is design to be  applicable in scenarios involving rapid flood propagation in urban areas, detailed floodplain mapping, and the analysis of complex river morphodynamics in mountainous regions. By integrating RiverFlowDynamics into the Landlab framework, we provide researchers, students, and practitioners with a powerful, accessible tool for hydraulics modeling. This integration facilitates future modifications and contributions, ensuring the model's continual evolution to meet emerging challenges in river system analysis and flood risk assessment. Integration with Landlab leverages its established grid structure, visualization tools, and interoperability with other components, facilitating multi-process simulations and reducing development overhead

## Basic Usage Example

RiverFlowDynamics integrates seamlessly with Landlab's grid structure. Here's a simple example demonstrating water flow in a rectangular channel:

```python
import numpy as np
from landlab import RasterModelGrid
from landlab.components import RiverFlowDynamics

# Create grid and topography
grid = RasterModelGrid((20, 60), xy_spacing=0.1)
z = grid.add_zeros("topographic__elevation", at="node")
z += 0.059 - 0.01 * grid.x_of_node
z[(grid.y_of_node > 1.5) | (grid.y_of_node < 0.5)] = 1.0

# Initialize fields
grid.add_zeros("surface_water__depth", at="node")
grid.add_zeros("surface_water__velocity", at="link")
wse = grid.add_zeros("surface_water__elevation", at="node")
wse += z

# Set boundary conditions
fixed_entry_nodes = np.arange(300, 910, 60)
fixed_entry_links = grid.links_at_node[fixed_entry_nodes][:, 0]

# Run model
rfd = RiverFlowDynamics(
    grid,
    fixed_entry_nodes=fixed_entry_nodes,
    fixed_entry_links=fixed_entry_links,
    entry_nodes_h_values=np.full(11, 0.5),
    entry_links_vel_values=np.full(11, 0.45),
)

for _ in range(100):
    rfd.run_one_step()
```	

# Acknowledgements
This research was supported by NSF EPSCoR (Award #2242769) and Chile's ANID FONDECYT Iniciación (Grant #11200949 to AM). Landlab development was funded by NSF (Awards #1147454, #1148305, #1450409, #1450338, #1450412) and the Community Surface Dynamics Modeling System (NSF Awards #1226297, #1831623).

# References
