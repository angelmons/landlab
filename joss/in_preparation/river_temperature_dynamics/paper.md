---
title: 'RiverTemperatureDynamics v1.0: A Landlab component for computing two-dimensional river temperature dynamics and energy budgets'

tags:
  - Landlab
  - Python
  - Ecohydraulics
  - Thermodynamics
  - Advection-dispersion
  - River temperature

authors:
  - name: Angel Monsalve
    orcid: 0000-0002-7369-1602
    equal-contrib: true
    affiliation: 1
    corresponding: true
  - name: Oscar Link
    orcid: 0000-0002-2188-6504
    equal-contrib: true
    affiliation: 2

affiliations:
 - name: Center for Ecohydraulics Research, Civil and Environmental Engineering, University of Idaho, Boise, ID, USA
   index: 1
   ror: 036jqmy94
 - name: Departamento de Ingeniería Civil, Universidad de Concepción, Concepción, Chile
   index: 2
   ror: 0460jpj73

date: 25 February 2026
bibliography: paper.bib
---

# Summary

The thermal regime of a river is a primary driver of aquatic ecosystem health, governing dissolved oxygen solubility, nutrient cycling rates, and the metabolic performance of fish and macroinvertebrates [@caissie:2006; @johnson:2024]. `RiverTemperatureDynamics` is a two-dimensional, depth-averaged numerical model developed as a component of the Landlab Python package [@hobley:2017; @barnhart:2020]. It simulates the spatial and temporal evolution of water temperature in rivers, streams, and floodplains by solving the advection-dispersion equation coupled with a comprehensive energy budget. The energy budget accounts for net shortwave radiation with dynamic surface reflectance and riparian shading, incoming and outgoing longwave radiation with cloud-cover corrections, evaporative and convective heat losses, bed heat conduction through the active sediment layer, and groundwater or hyporheic exchange. The component is designed to couple with Landlab's `RiverFlowDynamics` [@monsalve:2025] or `OverlandFlow` [@adams:2017] components, and can also operate on user-prescribed steady-state hydraulic fields when fully dynamic hydrodynamics are not required.

# Statement of need

River temperatures are rising globally in response to climate change, land-use modification, and flow regulation [@vanvliet:2011; @johnson:2024]. Predicting where and when critical thermal thresholds are exceeded requires spatially explicit models that capture the interplay between fluid transport and atmospheric heat exchange [@leach:2023]. Prior to `RiverTemperatureDynamics`, the Landlab framework lacked a component capable of simulating two-dimensional thermal transport with a physically complete energy budget. Existing monolithic hydraulic suites (e.g., Delft3D, TELEMAC, HEC-RAS) incorporate temperature modules, but their architectures are not designed for seamless coupling with the geomorphic, ecological, and landscape evolution models that Landlab supports. At the other end of the spectrum, lightweight Python-based thermal models are typically restricted to one-dimensional stream networks and lack the high-resolution advection schemes needed to preserve sharp thermal fronts in complex geometries [@dugdale:2017].

`RiverTemperatureDynamics` addresses this gap. Its target audience includes ecohydraulic researchers studying thermal refugia for cold-water species, environmental engineers assessing thermal pollution from point-source discharges, and Earth surface modelers investigating coupled climate–hydrology–geomorphology feedbacks. By providing an open-source, modular thermal solver within Landlab, users can simulate fully coupled hydro-thermal events—such as tracking a diurnal snowmelt pulse or a warm-water discharge as it advects, disperses, and exchanges heat along a river corridor—while retaining the ability to easily inject custom empirical laws (e.g., spatially variable riparian shading models) or to couple the temperature field with sediment transport and vegetation dynamics components.

# State of the field

Process-based stream temperature models span a wide range of complexity, from regression-based empirical tools to fully three-dimensional computational fluid dynamics solvers. As reviewed by @dugdale:2017 and @leach:2023, the most commonly used mechanistic models solve either a one-dimensional advection-dispersion equation along a stream network (e.g., QUAL2K, HFLUX, Heat Source) or embed a temperature module within a two-dimensional or three-dimensional hydrodynamic framework (e.g., Delft3D-WAQ, CE-QUAL-W2, TELEMAC). One-dimensional models cannot resolve lateral thermal gradients across floodplains, secondary channels, or wide braided reaches—precisely the habitats where cold-water refugia are most ecologically significant [@caissie:2006]. Multidimensional hydraulic suites can resolve these features, but typically require their own proprietary grid structures and are difficult to couple with external geomorphic or ecological process models.

`RiverTemperatureDynamics` occupies a distinct niche: it combines a two-dimensional transport solver featuring a high-order Total Variation Diminishing (TVD) advection scheme with a complete atmospheric and subsurface energy budget, all within the interoperable Landlab ecosystem. The TVD scheme uses a Van Leer flux limiter to prevent both numerical diffusion and unphysical oscillations near sharp thermal gradients [@leonard:1991], a critical advantage over the first-order upwind methods common in simpler thermal models [@link:2012]. The build-versus-contribute decision was driven by the fact that no existing Landlab component provided two-dimensional thermal transport, and retrofitting an external temperature module into Landlab's grid structure would sacrifice the native field-sharing and component-coupling capabilities that are central to the framework's design philosophy.

# Software design

## Governing equations

Assuming constant water density ($\rho_w$) and specific heat capacity ($c_{pw}$), the depth-averaged 2D thermal transport equation is:

$$\frac{\partial (h T_w)}{\partial t} + \nabla \cdot (h \mathbf{V} T_w) = \nabla \cdot (h \mathbf{D} \nabla T_w) + \frac{\Phi_{\mathrm{net}}}{\rho_w \, c_{pw}} \label{eq:governing}$$

where $T_w$ is the water temperature, $h$ is the local water depth, $\mathbf{V} = (u, v)$ is the depth-averaged velocity vector, and $\mathbf{D}$ is the anisotropic dispersion tensor with longitudinal and transverse coefficients $D_L = \alpha_L \, h \, u_*$ and $D_T = \alpha_T \, h \, u_*$ [@fischer:1979], where $u_*$ is the shear velocity. The source term $\Phi_{\mathrm{net}}$ (W/m$^2$) represents the total net heat flux at the water surface, which is the sum of seven components:

$$\Phi_{\mathrm{net}} = Q_{sw} + Q_{lw,\mathrm{in}} - Q_{lw,\mathrm{refl}} - Q_{lw,\mathrm{out}} - Q_e - Q_h + Q_{\mathrm{bed}} + Q_{\mathrm{gw}} \label{eq:heatbudget}$$

where $Q_{sw}$ is net shortwave solar radiation (mediated by surface albedo and riparian shading), $Q_{lw,\mathrm{in}}$, $Q_{lw,\mathrm{refl}}$, and $Q_{lw,\mathrm{out}}$ are the incoming, reflected, and emitted longwave radiation terms, $Q_e$ is the latent heat flux from evaporation (driven by the water–air vapor-pressure deficit), $Q_h$ is the sensible heat flux (Bowen ratio), $Q_{\mathrm{bed}}$ is heat conduction through the active sediment layer, and $Q_{\mathrm{gw}}$ is the advective heat exchange from groundwater or hyporheic inflow.

## Numerical architecture

The core design decision is operator splitting, which separates the transport physics from the thermodynamic source terms at each time step (\autoref{fig:flowchart}). This isolates the hyperbolic advection step from the spatially uniform atmospheric fluxes, allowing each sub-problem to be solved with the most appropriate numerical method.

First, meteorological boundary conditions are updated. If a CSV file of time-series data is provided, the component uses `scipy.interpolate.interp1d` to evaluate air temperature, relative humidity, wind speed, shortwave radiation, and cloud cover at the current simulation time. Alternatively, Landlab's `Radiation` component [@nudurupati:2023] can supply terrain-corrected incoming shortwave radiation (after a cell-to-node field mapping), enabling topographically shaded solar forcing over complex terrain.

Second, the advection-dispersion equation (Eq. 1) is solved. Thermal advection is handled by Landlab's `AdvectionSolverTVD`, which applies a second-order TVD finite-volume scheme with a Van Leer flux limiter to prevent both numerical diffusion and unphysical oscillations [@leonard:1991]. Turbulent dispersion is then computed via an explicit finite-volume diffusion step with the anisotropic coefficients defined above.

Third, the net heat flux $\Phi_{\mathrm{net}}$ (Eq. 2) is evaluated explicitly. Incoming longwave radiation uses the Brunt-type atmospheric emissivity with a cloud-cover amplification factor [@brunt:1932]. Surface reflectance is computed from a polynomial function of solar altitude. Bed conduction is modeled through a finite-thickness sediment layer with user-specified thermal properties, and groundwater exchange is driven by the specific discharge and temperature difference between the aquifer and the stream.

Finally, the outlet boundary condition is applied, with three user-selectable options: zero-gradient (recommended for outflow), gradient-preserving (linear extrapolation), or fixed-value (Dirichlet).

A practical trade-off of the explicit formulation is adherence to the Courant-Friedrichs-Lewy (CFL) stability condition; however, because the thermal solver is decoupled from the hydrodynamic solver, users may prescribe pre-computed or steady-state velocity and depth fields, allowing the thermal model to run at sub-second time steps with negligible overhead. When fully dynamic flow fields are needed, the component reads depth and velocity directly from `RiverFlowDynamics` [@monsalve:2025] or `OverlandFlow` [@adams:2017]; when the flow field can be simplified (e.g., a uniform channel), no external flow solver is required, although one is recommended for physical realism.

![Operator-splitting architecture of `RiverTemperatureDynamics`, showing the data flow from external forcings through the four computational stages within each time step. \label{fig:flowchart}](flowchart.png){ width=100% }

# Research impact statement

The component has been validated through a comprehensive test suite comprising 49 automated tests organized into four categories: component metadata, transport physics (mass conservation, advection direction, dispersion spreading, anisotropy, and three outlet boundary condition modes), energy budget closure (independently hand-calculated reference values for every flux term), and integration tests including comparison against the exact analytical solution for two-dimensional Gaussian plume advection-dispersion (\autoref{fig:analytical}). The analytical test demonstrates peak-amplitude agreement within 3% and peak-position agreement within one grid cell on a $\Delta x = 2$ m mesh.

The immediate near-term significance of the component lies in its capacity to be coupled with Landlab's shallow-water components to investigate how spatial variations in velocity and depth across river meanders, side channels, and floodplains create localized thermal micro-habitats—information that is critical for conservation planning of cold-water species such as salmon and trout [@caissie:2006; @johnson:2024]. The component is distributed with Jupyter Notebook tutorials that guide users through both isolated analytical validation and fully coupled hydro-thermal simulations.

![Comparison of the numerical solution from `RiverTemperatureDynamics` (solid lines) against the exact analytical solution for a two-dimensional Gaussian thermal plume (dashed lines) advected and dispersed in a uniform flow field at five time snapshots. \label{fig:analytical}](analytical_validation.png){ width=100% }

# AI usage disclosure

During the development of this software the authors utilized generative AI assistants (Claude, Anthropic) for code debugging. All AI-generated code was rigorously reviewed, modified where necessary, and validated by the authors against independent hand calculations and analytical solutions to ensure physical accuracy, scientific integrity, and adherence to Landlab's contribution standards.

# Acknowledgements

This research was supported by NSF EPSCoR (Award #2242769 to AM) and Chile's ANID FONDECYT Iniciación (Grant #11200949 to AM). Landlab development was funded by NSF (Awards #1147454, #1148305, #1450409, #1450338, #1450412) and the Community Surface Dynamics Modeling System (NSF Awards #1226297, #1831623).

# References
