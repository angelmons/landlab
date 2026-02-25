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
 - name: Departamento de Ingeniería Civil, Universidad de Concepción, Concepción, Chile
   index: 2

date: 24 February 2026
bibliography: paper.bib
---

# Summary

The thermal regime of a river is a primary driver of aquatic ecosystem health, governing water density, dissolved oxygen capacity, and the metabolic rates of aquatic organisms. `RiverTemperatureDynamics` is a two-dimensional, depth-averaged numerical model developed as a component for the Landlab Python package [@hobley:2017; @barnhart:2020]. It simulates the spatial and temporal evolution of water temperature in rivers, streams, and floodplains by solving the 2D advection-dispersion equation coupled with a comprehensive environmental energy budget. The model accounts for net shortwave radiation (including dynamic surface reflectance and riparian shading), incoming and outgoing longwave radiation (adjusted for cloud cover), evaporative and convective heat fluxes, bed heat conduction, and groundwater/hyporheic exchange. 

# Statement of need

Predicting how river temperatures will respond to climate change, land-use alterations, and flow regulation requires sophisticated modeling that couples hydrodynamics with thermodynamics. Prior to `RiverTemperatureDynamics`, the Landlab framework lacked a comprehensive component capable of modeling 2D thermal transport and atmospheric heat exchange. 

This component was developed to seamlessly integrate with Landlab's existing hydrological components, specifically `RiverFlowDynamics` [@monsalve:2025]. By providing an open-source, modular, grid-based thermal solver, researchers can now simulate fully coupled hydro-thermal events—such as tracking a diurnal snowmelt pulse or a flash flood as it heats and cools along a complex river corridor. The inclusion of hyporheic exchange and bed conduction makes this tool exceptionally well-suited for identifying and analyzing thermal refugia for aquatic species, a critical need for ecohydraulics and conservation planning.

# State of the field

While monolithic commercial and open-source software suites (e.g., Delft3D, TELEMAC, HEC-RAS) contain robust water quality and temperature modules, their rigid architectures make it difficult to couple them directly with custom geomorphic, ecological, or landscape evolution models. Conversely, lightweight Python-based thermal models are often restricted to 1D stream networks (e.g., StreamTemp) or lack the sophisticated advection schemes necessary to prevent numerical diffusion in complex geometries.

`RiverTemperatureDynamics` bridges this gap. It provides the advanced numerical rigor of specialized hydraulic modeling suites—such as the utilization of high-order Total Variation Diminishing (TVD) schemes to prevent the smearing of thermal fronts [@monsalve:2012]—within the highly interoperable, Python-native Landlab ecosystem. This allows researchers to easily inject their own customized empirical laws (e.g., dynamic riparian shading models) or couple the thermal solver directly with landscape evolution models without navigating the overhead of external compiled codebases.

# Software design

The core design philosophy of `RiverTemperatureDynamics` relies on operator splitting to separate the transport physics from the thermodynamic source terms. This architecture allows the model to compute complex, non-linear atmospheric fluxes efficiently while maintaining numerical stability during the advection-dispersion steps. 

1. **Advection:** Thermal plumes are transported using Landlab's `AdvectionSolverTVD`. As demonstrated in earlier 1D/2D thermal modeling studies [@monsalve:2012; @leonard:1991], simple upwind or central difference schemes introduce severe numerical diffusion or unphysical oscillations. Utilizing a TVD scheme with a Van Leer flux limiter ensures sharp, oscillation-free thermal gradients.
2. **Anisotropic Dispersion:** The model implements explicit finite-volume diffusion, dynamically computing longitudinal ($D_L$) and transverse ($D_T$) dispersion coefficients as a function of local shear velocity and water depth based on standard scaling laws [@fischer:1979].
3. **Atmospheric and Benthic Exchange:** The net heat flux ($\Phi_{net}$) is calculated explicitly. To support long-duration simulations, the component features a built-in meteorological interpolator using `pandas` and `scipy`, allowing the model to automatically ingest and smoothly apply highly resolved time-series data (e.g., air temperature, wind speed, solar radiation) at every computational time step.

A notable trade-off of the explicit finite-volume design is the strict adherence to the Courant-Friedrichs-Lewy (CFL) stability condition. However, because the component separates the thermal calculations from the hydrodynamic solver, users can prescribe steady-state hydraulic fields when appropriate, allowing the thermal solver to run extremely fast even at high temporal resolutions.

# Research impact statement

`RiverTemperatureDynamics` provides a rigorous, verifiable foundation for ecohydraulic research. The component is validated through a comprehensive test suite, including perfect agreement with the exact analytical solution for 2D Gaussian plume advection-dispersion in an infinite domain. 

Furthermore, the model demonstrates high fidelity in closing the environmental energy budget, successfully simulating continuous diurnal cycles buffered by bed conduction and groundwater upwelling. Its immediate near-term significance lies in its capacity to be coupled with Landlab's shallow water components to assess how spatial variations in velocity and depth across river meanders and floodplains create localized thermal micro-habitats. The component is fully documented with Jupyter Notebook tutorials that guide users through both isolated analytical tests and fully coupled hydro-thermal simulations.

# AI usage disclosure

During the development of this software and manuscript, the authors utilized generative AI (Google Gemini) to assist with code refactoring, generating unit tests to ensure 100% code coverage, and drafting initial structural outlines of this manuscript. All AI-generated code and text were rigorously reviewed, modified, and validated by the human authors to ensure physical accuracy, scientific integrity, and adherence to Landlab's contribution standards.

# Acknowledgements
This research was supported by NSF EPSCoR (Award #2242769 to AM) and Chile's ANID FONDECYT Iniciación (Grant #11200949 to AM). Landlab development was funded by NSF (Awards #1147454, #1148305, #1450409, #1450338, #1450412) and the Community Surface Dynamics Modeling System (NSF Awards #1226297, #1831623).

# References