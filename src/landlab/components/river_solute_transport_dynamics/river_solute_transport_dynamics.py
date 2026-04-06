import numpy as np
from landlab import Component
from landlab.components import AdvectionSolverTVD

class RiverSoluteTransportDynamics(Component):
    """Simulate multi-solute river transport using 2D advection-dispersion and OTIS dynamics.

    This component solves the depth-averaged 2D advection-dispersion equation
    coupled with transient storage, first-order decay, and kinetic sorption
    for N simultaneous solutes.
    """

    _name = "RiverSoluteTransportDynamics"
    _unit_agnostic = False

    def __init__(
        self,
        grid,
        solutes=["tracer"],
        h_min=0.01,
        alpha_L=10.0,
        alpha_T=0.6,
        ustar_fraction=0.1,
        outlet_boundary_condition="zero_gradient",
        fixed_outlet_concentration=None,
        alpha_exchange=None,
        h_storage=None,
        lambda_decay=None,
        lambda_s_decay=None,
        lambda_hat_sorption=None,
        lambda_hat_s_sorption=None,
        kd_sorption=None,
        rho_sediment=1500.0,
    ):
        super().__init__(grid)

        self._solutes = solutes
        self._h_min = h_min
        self._alpha_L = alpha_L
        self._alpha_T = alpha_T
        self._ustar_fraction = ustar_fraction
        self._rho_sed = rho_sediment

        valid_bcs = ("zero_gradient", "gradient_preserving", "fixed_value")
        if outlet_boundary_condition not in valid_bcs:
            raise ValueError(f"Invalid boundary condition: {outlet_boundary_condition}")
        self._outlet_bc = outlet_boundary_condition
        self._fixed_outlet_conc = fixed_outlet_concentration or {}
        
        self._outlet_nodes = grid.nodes_at_right_edge
        self._outlet_interior_1 = self._outlet_nodes - 1
        self._outlet_interior_2 = self._outlet_nodes - 2

        # Parameter dictionaries (default to 0.0 if not provided for a solute)
        self._alpha = alpha_exchange or {}
        self._h_s = h_storage or {}
        self._lambda = lambda_decay or {}
        self._lambda_s = lambda_s_decay or {}
        self._lambda_hat = lambda_hat_sorption or {}
        self._lambda_hat_s = lambda_hat_s_sorption or {} # <--- SAVE DICT
        self._Kd = kd_sorption or {}

        # Core hydraulic fields (must exist)
        self._h = grid.at_node["surface_water__depth"]
        self._adv_vel = grid.at_link["advection__velocity"]
        
        # Check for optional lateral flow
        if "lateral__water_specific_discharge" in grid.at_node:
            self._q_lat = grid.at_node["lateral__water_specific_discharge"]
        else:
            self._q_lat = np.zeros(grid.number_of_nodes)

        self._advectors = {}

        # Initialize dynamically named fields for each solute
        for solute in self._solutes:
            fields = [
                f"surface_water__{solute}__concentration",
                f"storage_zone__{solute}__concentration",
                f"streambed__{solute}__sorbate_concentration",
                f"lateral__{solute}__concentration"
            ]
            for field in fields:
                if field not in grid.at_node:
                    grid.add_zeros(field, at="node")
            
            # Create a dedicated TVD advection solver for this solute
            self._advectors[solute] = AdvectionSolverTVD(
                grid, fields_to_advect=f"surface_water__{solute}__concentration"
            )

    def _advection_dispersion(self, dt):
            """Solve advection and anisotropic dispersion for all solutes."""
            h_link = self._grid.map_mean_of_link_nodes_to_link("surface_water__depth")
            h_link = np.maximum(h_link, self._h_min)
            u_star = np.abs(self._adv_vel) * self._ustar_fraction
    
            D_link = np.zeros(self._grid.number_of_links, dtype=float)
            D_link[self._grid.horizontal_links] = self._alpha_L * h_link[self._grid.horizontal_links] * u_star[self._grid.horizontal_links]
            D_link[self._grid.vertical_links] = self._alpha_T * h_link[self._grid.vertical_links] * u_star[self._grid.vertical_links]
            
            core = self._grid.core_nodes
            
            # CALCULATE VELOCITY DIVERGENCE ONCE
            div_u = self._grid.calc_flux_div_at_node(self._adv_vel)
    
            for solute in self._solutes:
                # 1. Advection (Conservative)
                self._advectors[solute].run_one_step(dt)
                
                # 2. CORRECT FOR ARTIFICIAL MASS COMPRESSION
                # TVD solver causes spikes where velocity slows down. We correct it here.
                field_name = f"surface_water__{solute}__concentration"
                C = self._grid.at_node[field_name]
                C[core] += C[core] * div_u[core] * dt
                
                # 3. Dispersion
                grad_C = self._grid.calc_grad_at_link(field_name)
                diff_flux = D_link * grad_C
                dCdt_diff = self._grid.calc_flux_div_at_node(diff_flux)
                
                C[core] += dCdt_diff[core] * dt

    def _otis_reactions(self, dt):
            h = np.maximum(self._h, self._h_min)
    
            for solute in self._solutes:
                C = self._grid.at_node[f"surface_water__{solute}__concentration"]
                C_s = self._grid.at_node[f"storage_zone__{solute}__concentration"]
                C_sed = self._grid.at_node[f"streambed__{solute}__sorbate_concentration"]
                C_lat = self._grid.at_node[f"lateral__{solute}__concentration"]
    
                alpha = self._alpha.get(solute, 0.0)
                h_s = self._h_s.get(solute, 0.1) 
                lam = self._lambda.get(solute, 0.0)
                lam_s = self._lambda_s.get(solute, 0.0)
                lam_hat = self._lambda_hat.get(solute, 0.0)      # Main channel (LAMHAT)
                lam_hat_s = self._lambda_hat_s.get(solute, 0.0)  # Storage zone (LAMHATS)
                Kd = self._Kd.get(solute, 0.0)
    
                lat_flux = (self._q_lat / h) * (C_lat - C)
                storage_flux_main = alpha * (C_s - C)
                storage_flux_zone = alpha * (h / h_s) * (C - C_s)
                decay_main = -lam * C
                decay_zone = -lam_s * C_s
                
                # --- CORRECTED SORPTION KINETICS ---
                sorption_main = self._rho_sed * lam_hat * (C_sed - Kd * C)
                sorption_storage = self._rho_sed * lam_hat_s * (C_sed - Kd * C_s)
                sorption_bed = lam_hat * (Kd * C - C_sed) + lam_hat_s * (Kd * C_s - C_sed)
    
                # Apply updates
                C += (lat_flux + storage_flux_main + decay_main + sorption_main) * dt
                C_s += (storage_flux_zone + decay_zone + sorption_storage) * dt
                C_sed += sorption_bed * dt

    def _apply_boundaries(self):
        """Apply downstream outlet boundary conditions for all solutes."""
        for solute in self._solutes:
            C = self._grid.at_node[f"surface_water__{solute}__concentration"]
            
            if self._outlet_bc == "zero_gradient":
                C[self._outlet_nodes] = C[self._outlet_interior_1]
            elif self._outlet_bc == "gradient_preserving":
                C[self._outlet_nodes] = 2.0 * C[self._outlet_interior_1] - C[self._outlet_interior_2]
            elif self._outlet_bc == "fixed_value":
                fixed_val = self._fixed_outlet_conc.get(solute, 0.0)
                C[self._outlet_nodes] = fixed_val

    def run_one_step(self, dt):
        """Advance all solutes by one time step *dt*."""
        self._advection_dispersion(dt)
        self._otis_reactions(dt)
        self._apply_boundaries()