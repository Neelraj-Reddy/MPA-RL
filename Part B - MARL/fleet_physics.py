import numpy as np

class FishingFleet:
    def __init__(self, num_boats, ports, env_width, env_height, refuel_requires_sale=True):
        self.num_boats = num_boats
        self.ports = np.array(ports)
        
        # State Arrays
        self.positions = np.tile(self.ports[0], (num_boats, 1)) + (np.random.rand(num_boats, 2) - 0.5) * 5.0
        self.headings = np.random.rand(num_boats) * 2 * np.pi 
        self.velocities = np.zeros(num_boats)
        
        # Resource constraints (Realistic fuel limits)
        self.max_fuel = 12000.0  # Liters of marine diesel (increased from 5000 for long deployments)
        self.max_cargo = 500.0  # Tons of fish
        self.fuel_levels = np.full(num_boats, self.max_fuel)
        self.cargo_levels = np.zeros(num_boats)
        self.nets_deployed = np.zeros(num_boats, dtype=bool)
        
        # Trawl Types [0: Light Pelagic, 1: Heavy Bottom]
        self.trawl_types = np.random.choice([0, 1], size=num_boats)
        self.catch_radii = np.array([2.0, 3.5])      # Reduced to slow down cargo fill
        self.catch_rates = np.array([0.08, 0.20])    # Increased from [0.05, 0.15] for better yield
        self.max_catch_per_tick = np.array([30.0, 80.0]) # Increased from [25.0, 75.0]
        
        # Realistic Physics Constants
        self.max_speed = 2.0    # Reduced from 4.0 for more controlled movement
        self.idle_fuel_cost = 8.0    # 8 L/hr just to run generators/systems (reduced from 15 for sustainability)
        
        # Boundary constraints
        self.env_width = env_width
        self.env_height = env_height
        self.hull_drag_empty = 1.5    # Base drag
        self.hull_drag_full = 3.5     # Drag when sitting deep in the water with 500 Tons
        self.net_drag_coeffs = np.array([2.0, 8.0]) # REDUCED from [5.0, 25.0] - less drag penalty!
        self.wind_drag_coeff = 0.5    # Wind resistance against the superstructure

        # Port policy: when True, boats refuel only if they actually landed catch.
        self.refuel_requires_sale = bool(refuel_requires_sale)

        # Sale tracking (set each step for reward system)
        self.just_sold = np.zeros(num_boats, dtype=bool)
        self.cargo_sold = np.zeros(num_boats)

    def step(self, actions, fish_pop, env):
        if self.num_boats == 0:
            return

        # 1. PARSE ACTIONS
        heading_changes = actions[:, 0]
        throttles = np.clip(actions[:, 1], 0.0, 1.0)
        self.nets_deployed = actions[:, 2] > 0.2  # Lowered threshold for easier deployment
        
        self.headings = (self.headings + heading_changes) % (2 * np.pi)
        self.velocities = throttles * self.max_speed

        # Convert boat scalar velocity & heading to 2D vectors
        boat_v_x = np.cos(self.headings) * self.velocities
        boat_v_y = np.sin(self.headings) * self.velocities
        boat_vectors = np.column_stack((boat_v_x, boat_v_y))

        # 2. FETCH ENVIRONMENTAL FORCES
        # We assume env has these utility functions implemented
        grid_coords = np.floor(self.positions).astype(int)
        grid_coords[:, 0] = np.clip(grid_coords[:, 0], 0, env.width - 1)
        grid_coords[:, 1] = np.clip(grid_coords[:, 1], 0, env.height - 1)
        
        current_vectors = np.array([env.get_current(x, y) for x, y in grid_coords])
        wind_vectors = np.array([env.get_wind(x, y) for x, y in grid_coords])
        local_temps = np.array([env.get_temperature(x, y) for x, y in grid_coords])

        # 3. RELATIVE VELOCITY PHYSICS
        water_rel_vectors = boat_vectors - current_vectors
        wind_rel_vectors = boat_vectors - wind_vectors
        
        # Magnitudes of relative speeds
        speed_thru_water = np.linalg.norm(water_rel_vectors, axis=1)
        speed_thru_air = np.linalg.norm(wind_rel_vectors, axis=1)

        # 4. DYNAMIC DRAG & FUEL BURN
        # Cargo penalty: Linearly scale hull drag based on how full the boat is
        cargo_ratio = self.cargo_levels / self.max_cargo
        dynamic_hull_drag = self.hull_drag_empty + (self.hull_drag_full - self.hull_drag_empty) * cargo_ratio
        
        # Temperature density penalty (colder water = denser = slightly more drag)
        temp_modifier = np.where(local_temps < 15.0, 1.05, 1.0) 
        
        # Calculate actual fuel burned based on REALISTIC PHYSICS
        # 1. Hull drag scales with velocity cubed (hydrodynamic resistance)
        water_resistance = (dynamic_hull_drag * temp_modifier) * (speed_thru_water ** 3)
        
        # 2. Cargo weight creates additional drag (heavier boats burn more fuel)
        # Formula: Extra fuel ∝ cargo_weight × speed²
        cargo_weight_factor = cargo_ratio * 50.0  # 50 extra units of drag per full cargo
        cargo_drag = cargo_weight_factor * (speed_thru_water ** 2) * 0.5
        
        # 3. Net deployment drag (fishing gear resistance)
        net_resistance = (self.net_drag_coeffs[self.trawl_types] * self.nets_deployed) * (speed_thru_water ** 2)
        
        # 4. Wind/air resistance
        wind_resistance = self.wind_drag_coeff * (speed_thru_air ** 2)
        
        # Total fuel burn (realistic physics)
        fuel_burned = self.idle_fuel_cost + water_resistance + cargo_drag + net_resistance + wind_resistance
        self.fuel_levels -= fuel_burned
        
        # Out of fuel logic
        out_of_fuel = self.fuel_levels <= 0
        self.fuel_levels = np.maximum(self.fuel_levels, 0)
        boat_vectors[out_of_fuel] = [0.0, 0.0] # Engines stop
        self.nets_deployed[out_of_fuel] = False

        # 5. KINEMATICS (Applying actual movement over ground)
        # If engines are dead, the boat drifts entirely with the ocean current
        actual_movement = np.where(out_of_fuel[:, None], current_vectors, boat_vectors)
        self.positions += actual_movement
        
        # BOUNDARY ENFORCEMENT - Keep boats within ocean bounds
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.env_width - 0.1)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.env_height - 0.1)

        # 6. PORT UNLOADING & REFUELING
        self.just_sold[:] = False
        self.cargo_sold[:] = 0.0
        for port in self.ports:
            dist_to_port = np.linalg.norm(self.positions - port, axis=1)
            at_port = dist_to_port < 1.5
            selling = at_port & (self.cargo_levels > 0)
            self.just_sold |= selling
            self.cargo_sold[selling] += self.cargo_levels[selling]
            self.cargo_levels[at_port] = 0.0

            if self.refuel_requires_sale:
                refuel_mask = selling
            else:
                refuel_mask = at_port
            self.fuel_levels[refuel_mask] = self.max_fuel

        # 7. CONTINUOUS EXTRACTION (REALISTIC - With Success Rates & Fish Escape)
        active_fishers = np.where(self.nets_deployed & (~out_of_fuel) & (self.cargo_levels < self.max_cargo))[0]
        
        if len(active_fishers) > 0 and fish_pop.num_schools > 0:
            for i in active_fishers:
                boat_pos = self.positions[i]
                t_type = self.trawl_types[i]
                radius = self.catch_radii[t_type]
                q_rate = self.catch_rates[t_type]
                
                # --- REALISTIC CATCH MECHANICS ---
                
                # 1. Base catch capacity (with dynamic factors)
                base_cap = self.max_catch_per_tick[t_type]
                boat_speed_water = speed_thru_water[i]
                
                # Speed factor: Optimal trawling speed is ~2.5 units/hr
                # Too slow = net collapses (0.1x), too fast = escapees (1.0x)
                speed_factor = np.clip(boat_speed_water / 2.5, 0.1, 1.2)
                
                # Saturation penalty: As cargo fills, fish clog the net
                cargo_ratio = self.cargo_levels[i] / self.max_cargo
                saturation_penalty = 1.0 - (0.4 * cargo_ratio)  # Up to 40% penalty at full
                
                # 2. CATCH SUCCESS RATE (30-80% depending on conditions)
                # Base success: 50%
                success_rate = 0.50
                
                # Temperature matters: Fish school up in good temps
                local_temp = env.get_temperature(*boat_pos)
                if 15.0 <= local_temp <= 21.0:
                    success_rate += 0.20  # +20% in comfort zone
                elif 12.0 <= local_temp <= 24.0:
                    success_rate += 0.10  # +10% in acceptable range
                else:
                    success_rate -= 0.15  # -15% outside range (fish dispersed)
                
                # Speed factor affects catch reliability
                if boat_speed_water < 1.5:
                    success_rate -= 0.10  # Too slow, fish escape
                elif boat_speed_water > 3.5:
                    success_rate -= 0.15  # Too fast, net damage/escapees
                
                # 3. Fish ESCAPE rate (10-30%)
                escape_rate = 0.10 + (0.1 * max(0, 1.0 - success_rate))  # Worse conditions = more escape
                
                # 4. Net stress/damage risk (can reduce efficiency)
                net_stress = (boat_speed_water ** 2) * (1.0 + cargo_ratio)
                net_efficiency = 1.0
                if net_stress > 10.0:
                    net_efficiency = 0.8  # Net damaged, only 80% efficiency
                elif net_stress > 15.0:
                    net_efficiency = 0.6  # Severe damage
                
                # Calculate actual catch with all factors
                dynamic_hard_cap = base_cap * speed_factor * saturation_penalty * net_efficiency
                
                # Stochastic success - will this deployment succeed?
                if np.random.random() < success_rate:
                    dist_to_fish = np.linalg.norm(fish_pop.positions - boat_pos, axis=1)
                    catchable_schools = np.where(dist_to_fish < radius)[0]
                    
                    caught_total = 0.0
                    for fish_idx in catchable_schools:
                        if fish_pop.energies[fish_idx] <= 0:
                            continue
                        
                        theoretical_catch = fish_pop.energies[fish_idx] * q_rate
                        space_left = self.max_cargo - self.cargo_levels[i]
                        
                        # Apply escape: some fish get away
                        actual_catch = theoretical_catch * (1.0 - escape_rate)
                        
                        # Hard cap from net capacity
                        actual_catch = min(actual_catch, dynamic_hard_cap, space_left)
                        
                        fish_pop.energies[fish_idx] -= actual_catch
                        self.cargo_levels[i] += actual_catch
                        caught_total += actual_catch
                        
                        if self.cargo_levels[i] >= self.max_cargo:
                            break
                # else: Net deployment failed (no catch this step)
                