import numpy as np

class FishPopulation:
    def __init__(self, initial_population, env_width, env_height):
        """
        Initialize fish population with realistic density.
        
        For realistic ocean fishing:
        - World map: 100x100 units (~10000 sq units)
        - Real fish population should be dense enough to find
        - Increased from 1000 to 2500+ schools for 0.25+ density
        """
        self.num_schools = initial_population
        
        # Spawn fish across the map to increase encounter chances
        # Keep a small margin so fish don't spawn exactly on the boundary
        safe_margin = 5.0
        self.positions = np.column_stack([
            np.random.uniform(safe_margin, env_width - safe_margin, size=self.num_schools),
            np.random.uniform(safe_margin, env_height - safe_margin, size=self.num_schools)
        ])
        self.velocities = (np.random.rand(self.num_schools, 2) - 0.5) * 0.5  # Slower initial velocity
        self.energies = np.full(self.num_schools, 70.0)  # Start below breeding threshold - must earn energy first
        self.ages = np.zeros(self.num_schools)
        
        self.max_speed = 0.5  # Fish move slower for realism
        # Metabolism: LOW base cost, but INTENSE food consumption when eating
        self.basal_metabolism = 0.08  # Very low - only 0.08/step base cost
        self.move_cost_multiplier = 0.02  # Very low movement penalty
        self.nutrition_value = 20.0  # HIGH - food is very valuable when consumed
        self.breeding_threshold = 150.0  # High threshold - must accumulate for 6 months
        
        # NEW: Flexible Comfort Zone instead of a rigid 18 degrees
        self.T_min = 14.0  
        self.T_max = 22.0  
        
        # Age-based mortality - VERY gentle (fish have long lifespans)
        self.m_base = 0.00001  # Nearly zero base mortality
        self.alpha = 0.00001   # Tiny aging effect
        self.beta = 0.00005    # Slow mortality growth with age

    def step(self, env):
        if self.num_schools == 0:
            return 
            
        self.ages += 1.0

        # Map current positions to grid
        xs = np.clip(np.floor(self.positions[:, 0]).astype(int), 0, env.width - 1)
        ys = np.clip(np.floor(self.positions[:, 1]).astype(int), 0, env.height - 1)
        
        local_currents_u = env.current_u[xs, ys]
        local_currents_v = env.current_v[xs, ys]
        local_currents = np.column_stack((local_currents_u, local_currents_v))
        
        current_temps = env.temperature_grid[xs, ys]
        current_plankton = env.plankton_grid[xs, ys]

        # ==========================================
        # 1. LOCALIZED SENSORY PROBES (The Fix)
        # ==========================================
        # Generate a random angle for each school to "sniff" the water nearby
        angles = np.random.rand(self.num_schools) * 2 * np.pi
        sense_radius = 3.0 # They can sense 3 units away
        
        sense_x = np.clip(self.positions[:, 0] + np.cos(angles) * sense_radius, 0, env.width - 1)
        sense_y = np.clip(self.positions[:, 1] + np.sin(angles) * sense_radius, 0, env.height - 1)
        
        sense_xs = np.floor(sense_x).astype(int)
        sense_ys = np.floor(sense_y).astype(int)
        
        sensed_plankton = env.plankton_grid[sense_xs, sense_ys]
        sensed_temps = env.temperature_grid[sense_xs, sense_ys]

        # A. Foraging Vector (Steer towards the sniffed spot if it has more food)
        food_pull = np.where((sensed_plankton > current_plankton)[:, None], 
                             np.column_stack((np.cos(angles), np.sin(angles))), 
                             0.0)

        # B. Thermotaxis Vector (Steer towards the sniffed spot if we are uncomfortable)
        def dist_to_comfort(t):
            return np.maximum(0, np.maximum(self.T_min - t, t - self.T_max))
            
        current_t_dist = dist_to_comfort(current_temps)
        sensed_t_dist = dist_to_comfort(sensed_temps)
        
        # Only pull if the sniffed spot is closer to the comfort zone
        temp_pull = np.where((sensed_t_dist < current_t_dist)[:, None],
                             np.column_stack((np.cos(angles), np.sin(angles))),
                             0.0)

        # C. Apply Forces
        # Damped movement to avoid abrupt jumps
        self.velocities *= 0.85
        wander = (np.random.rand(self.num_schools, 2) - 0.5) * 0.2
        
        # Temp survival overrides food searching (reduced multipliers for smoother motion)
        self.velocities += wander + (food_pull * 1.0) + (temp_pull * 1.5)
        
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(speeds > self.max_speed, 
                                   (self.velocities / speeds) * self.max_speed, 
                                   self.velocities)
                                   
        # Move with currents
        actual_movement = self.velocities + local_currents
        self.positions += actual_movement
        
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, env.width - 1.001)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, env.height - 1.001)

        # ==========================================
        # 2. FEEDING
        # ==========================================
        xs = np.clip(np.floor(self.positions[:, 0]).astype(int), 0, env.width - 1)
        ys = np.clip(np.floor(self.positions[:, 1]).astype(int), 0, env.height - 1)
        
        # Limit per-school intake and prevent overconsumption in crowded cells
        eat_capacity = 0.35
        available_food = env.plankton_grid[xs, ys]
        desired_bites = np.minimum(available_food, eat_capacity)

        total_demand = np.zeros_like(env.plankton_grid)
        np.add.at(total_demand, (xs, ys), desired_bites)

        cell_available = env.plankton_grid[xs, ys]
        cell_demand = total_demand[xs, ys]
        scale = np.minimum(1.0, cell_available / (cell_demand + 1e-8))
        plankton_consumed = desired_bites * scale

        np.add.at(env.plankton_grid, (xs, ys), -plankton_consumed)

        # ==========================================
        # 3. METABOLISM & DEATH
        # ==========================================
        # Smooth temperature efficiency (broad comfort band, never zero)
        temp_efficiency = 0.4 + 0.6 * np.exp(-((current_temps - 18.0) ** 2) / (2 * (6.0 ** 2)))
        current_speeds_sq = np.sum(self.velocities**2, axis=1)
        kinetic_cost = current_speeds_sq * self.move_cost_multiplier
        
        nutrition_gain = plankton_consumed * self.nutrition_value * temp_efficiency
        self.energies += nutrition_gain - (self.basal_metabolism + kinetic_cost)
        
        mortality_probs = self.m_base + self.alpha * np.exp(self.beta * self.ages)
        death_rolls = np.random.rand(self.num_schools)
        
        alive_mask = (self.energies > 0) & (death_rolls > mortality_probs)
        
        self.positions = self.positions[alive_mask]
        self.velocities = self.velocities[alive_mask]
        self.energies = self.energies[alive_mask]
        self.ages = self.ages[alive_mask]
        self.num_schools = len(self.energies)

        # ==========================================
        # 4. PLANKTON-LIMITED REPRODUCTION
        # ==========================================
        # Fish breed when they have ENERGY (from eating) AND plankton is available
        # This couples reproduction to food availability naturally
        
        if self.num_schools > 0:
            xs = np.clip(np.floor(self.positions[:, 0]).astype(int), 0, env.width - 1)
            ys = np.clip(np.floor(self.positions[:, 1]).astype(int), 0, env.height - 1)
            
            # Get plankton density at each fish's location
            local_plankton = env.plankton_grid[xs, ys]
            
            # Breeding requires: high energy AND decent food locally
            # Threshold is HIGH (150) so it takes ~6 months to accumulate
            has_energy = self.energies > self.breeding_threshold
            has_food = local_plankton > 0.5  # Breeding only possible when food is decent (above 0.5)
            can_breed = has_energy & has_food
            
            # Temperature bonus for breeding success
            temps = env.temperature_grid[xs, ys]
            temp_eff = 0.5 + 0.5 * np.exp(-((temps - 18.0) ** 2) / (2 * (7.0 ** 2)))
            
            # Breeding probability: 10% chance when conditions met + temp bonus
            # Since threshold is high, breeding is rare event
            breeding_chance = 0.10 * temp_eff
            breed_rolls = np.random.rand(self.num_schools)
            breed_mask = can_breed & (breed_rolls < breeding_chance)
            num_new_fish = np.sum(breed_mask)
            
            if num_new_fish > 0:
                breeding_positions = self.positions[breed_mask]
                
                # SIGNIFICANT COST: Breeding depletes 50% of energy - creates 6-month cycle
                # At 150 energy, loses 75, drops back to 75 (takes 6 months to regain 75)
                spawn_fractions = 0.50
                spawned_energy = self.energies[breed_mask] * spawn_fractions
                self.energies[breed_mask] -= spawned_energy
                
                # New fish spawn with good energy from parent investment
                new_positions = breeding_positions + (np.random.rand(num_new_fish, 2) - 0.5) * 0.5
                new_velocities = (np.random.rand(num_new_fish, 2) - 0.5) * self.max_speed
                new_energies = spawned_energy.copy()  # Inherit parent's energy investment (50%!)
                new_ages = np.zeros(num_new_fish)
                
                self.positions = np.vstack((self.positions, new_positions))
                self.velocities = np.vstack((self.velocities, new_velocities))
                self.energies = np.concatenate((self.energies, new_energies))
                self.ages = np.concatenate((self.ages, new_ages))
                self.num_schools += num_new_fish