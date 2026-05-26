import numpy as np

class OceanEnvironment:
    def __init__(self, width=100, height=100, hours_per_tick=1):
        self.width = width
        self.height = height
        self.hours_per_tick = hours_per_tick
        self.time_step = 0
        self.ticks_per_year = (365 * 24) / self.hours_per_tick
        
        # 1. Biological Grids
        self.K = 1.5
        self.plankton_grid = np.random.uniform(1.0, 1.3, (width, height))  # High initial plankton - food is abundant
        self.temperature_grid = np.zeros((width, height))
        
        # 2. Marine Protected Areas (MPA) - Fish-density based
        self.mpa_grid = np.zeros((width, height), dtype=np.float32)  # 0.0 = open, 1.0 = protected
        self.mpa_update_interval = 720  # Update MPAs every 30 days (720 hours)
        self.mpa_coverage_target = 0.15  # 15% of ocean should be protected
        self.mpa_persistence = 0.85  # 85% of MPA cells persist between updates (stability)
        
        # 3. Physical Vector Grids (U = X-axis force, V = Y-axis force)
        self.current_u = np.zeros((width, height))
        self.current_v = np.zeros((width, height))
        self.wind_u = np.zeros((width, height))
        self.wind_v = np.zeros((width, height))
        
        # Plankton Growth Parameters (Supports slow-cycling fish ecosystem)
        self.r_max = 0.010 * self.hours_per_tick  # Reduced from 0.015 - slower growth needed now
        self.T_opt = 18.0   
        self.T_sigma = 5.0  
        
        # Initialize the first frame of the environment
        self._update_environmentals()

    def _update_environmentals(self):
        """
        Updates Temperatures, Ocean Currents, and Wind Vectors.
        Uses procedural math to create organic, shifting patterns.
        """
        # --- 1. Temperature (Latitudinal gradient + seasons + eddies + regional features) ---
        y_indices = np.arange(self.height)
        x_indices = np.arange(self.width)
        time_factor = (2 * np.pi * self.time_step) / self.ticks_per_year

        # Latitude-like gradient: warm in the middle, cool toward edges
        lat = (y_indices / (self.height - 1)) * 2.0 - 1.0
        lat_2d = np.tile(lat[None, :], (self.width, 1))
        base_temp = 15.5 + 6.5 * np.cos(np.pi * lat_2d)

        # Seasonal cycle (hemispheric phase shift)
        seasonal = 2.5 * np.sin(time_factor + (lat_2d * 0.8))

        # Mesoscale eddies/fronts (slowly drifting wavefield)
        X, Y = np.meshgrid(x_indices / self.width, y_indices / self.height, indexing='ij')
        eddies = (
            1.8 * np.sin(2 * np.pi * (X * 2.0 + Y * 1.5 + time_factor * 0.03)) +
            1.2 * np.cos(2 * np.pi * (X * 1.2 - Y * 1.7 + time_factor * 0.02))
        )

        # Regional features (warm tongue + cold upwelling)
        warm_cx, warm_cy = 0.75, 0.60
        cold_cx, cold_cy = 0.25, 0.30
        warm_tongue = 2.2 * np.exp(-(((X - warm_cx) ** 2) / 0.02 + ((Y - warm_cy) ** 2) / 0.05))
        cold_upwelling = -2.0 * np.exp(-(((X - cold_cx) ** 2) / 0.03 + ((Y - cold_cy) ** 2) / 0.02))

        self.temperature_grid = base_temp + seasonal + eddies + warm_tongue + cold_upwelling
        self.temperature_grid = np.clip(self.temperature_grid, 5.0, 28.0)

        # --- 2. Ocean Currents (Procedural Swirls) ---
        # Create normalized coordinate grids (-1.0 to 1.0)
        x_norm = np.linspace(-1, 1, self.width)
        y_norm = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x_norm, y_norm, indexing='ij')
        
        current_speed = 0.5 # Max current speed (e.g., 0.5 knots)
        # Create a swirling gyre effect that shifts slightly over time
        self.current_u = current_speed * np.sin(np.pi * Y + time_factor * 0.5)
        self.current_v = current_speed * np.cos(np.pi * X - time_factor * 0.5)

        # Slight temperature adjustment based on current strength
        current_mag = np.sqrt(self.current_u ** 2 + self.current_v ** 2)
        temp_current_influence = (current_mag / current_speed - 0.5) * 1.0
        self.temperature_grid = self.temperature_grid + temp_current_influence

        # --- 3. Wind (Directional with Turbulence) ---
        wind_speed = 3.0 # Wind is generally faster than water
        # Prevailing wind blowing primarily East to West, but shifting North/South seasonally
        self.wind_u = np.full((self.width, self.height), wind_speed * 0.8) # Base Eastern flow
        self.wind_v = wind_speed * np.sin(time_factor + X * 2.0) # Seasonal North/South shift

    def step(self):
        """Advances the environment and applies biological growth."""
        self.time_step += 1
        self._update_environmentals()
        
        r_temp = self.r_max * np.exp(-((self.temperature_grid - self.T_opt)**2) / (2 * self.T_sigma**2))
        growth = r_temp * self.plankton_grid * (1.0 - (self.plankton_grid / self.K))
        baseline_regen = 0.002 * self.hours_per_tick  # Good baseline - ensures food availability
        
        self.plankton_grid = np.clip(self.plankton_grid + growth + baseline_regen, 0.0, self.K)

    # ==========================================
    # UTILITY APIs (For Boats & Fish)
    # ==========================================
    
    def _get_grid_coords(self, x, y):
        """Helper to safely map continuous coordinates to grid indices."""
        grid_x = int(np.clip(x, 0, self.width - 1))
        grid_y = int(np.clip(y, 0, self.height - 1))
        return grid_x, grid_y

    def get_temperature(self, x, y):
        gx, gy = self._get_grid_coords(x, y)
        return self.temperature_grid[gx, gy]

    def get_current(self, x, y):
        """Returns the [u, v] water current vector at this coordinate."""
        gx, gy = self._get_grid_coords(x, y)
        return np.array([self.current_u[gx, gy], self.current_v[gx, gy]])

    def get_wind(self, x, y):
        """Returns the [u, v] wind vector at this coordinate."""
        gx, gy = self._get_grid_coords(x, y)
        return np.array([self.wind_u[gx, gy], self.wind_v[gx, gy]])

    # ... (Keep get_plankton, consume_plankton, get_plankton_gradient as previously defined) ...

    def get_plankton(self, x, y):
        """Returns plankton density at a specific continuous coordinate."""
        grid_x, grid_y = int(np.clip(x, 0, self.width - 1)), int(np.clip(y, 0, self.height - 1))
        return self.plankton_grid[grid_x, grid_y]

    def consume_plankton(self, x, y, amount):
        """Fish call this to eat. Returns the actual amount consumed."""
        grid_x, grid_y = int(np.clip(x, 0, self.width - 1)), int(np.clip(y, 0, self.height - 1))
        available = self.plankton_grid[grid_x, grid_y]
        consumed = min(available, amount)
        self.plankton_grid[grid_x, grid_y] -= consumed
        return consumed

    def get_plankton_gradient(self, x, y, radius=1):
        """
        Calculates the direction a fish should swim to find more food.
        Returns a vector (dx, dy) pointing to the densest plankton nearby.
        """
        grid_x, grid_y = int(np.clip(x, 0, self.width - 1)), int(np.clip(y, 0, self.height - 1))
        
        # Define search boundaries
        min_x, max_x = max(0, grid_x - radius), min(self.width, grid_x + radius + 1)
        min_y, max_y = max(0, grid_y - radius), min(self.height, grid_y + radius + 1)
        
        # Extract the local patch of plankton
        local_patch = self.plankton_grid[min_x:max_x, min_y:max_y]
        
        # Find the coordinates of the maximum plankton in this patch
        max_idx = np.unravel_index(np.argmax(local_patch), local_patch.shape)
        
        # Convert patch coordinates back to global grid vector directions
        target_x = min_x + max_idx[0]
        target_y = min_y + max_idx[1]
        
        # Return direction vector
        return float(target_x - grid_x), float(target_y - grid_y)
    
    def update_mpas(self, fish_population):
        """
        Update Marine Protected Areas based on fish density.
        MPAs form where fish are abundant to protect spawning/feeding grounds.
        """
        # Only update every N steps to maintain stability
        if self.time_step % self.mpa_update_interval != 0:
            return
        
        # Compute fish density grid
        fish_density = self._compute_fish_density(fish_population)
        
        # Apply persistence - keep 85% of existing MPAs
        persistent_mpa = (self.mpa_grid > 0.5) & (np.random.random((self.width, self.height)) < self.mpa_persistence)
        
        # Identify high-density areas for new MPAs
        density_threshold = np.percentile(fish_density, 80)  # Top 20% fish density
        high_density_zones = fish_density >= density_threshold
        
        # Combine persistent + new high-density zones
        candidate_mpa = persistent_mpa | high_density_zones
        
        # Smooth MPAs with connected component expansion (make them contiguous)
        expanded_mpa = self._expand_mpa_regions(candidate_mpa.astype(float))
        
        # Enforce coverage target (don't protect too much ocean)
        total_cells = self.width * self.height
        target_cells = int(total_cells * self.mpa_coverage_target)
        current_cells = int(expanded_mpa.sum())
        
        if current_cells > target_cells:
            # Trim excess by removing lowest-density cells
            mpa_mask = expanded_mpa > 0.5
            protected_densities = fish_density * mpa_mask
            trim_threshold = np.percentile(protected_densities[mpa_mask], 
                                          (1 - target_cells/current_cells) * 100)
            expanded_mpa[protected_densities < trim_threshold] = 0.0
        
        self.mpa_grid = expanded_mpa
    
    def _compute_fish_density(self, fish_population):
        """Create a density grid showing where fish are concentrated"""
        density = np.zeros((self.width, self.height))
        
        if fish_population.num_schools == 0:
            return density
        
        # Add Gaussian kernels around each fish school
        for i in range(fish_population.num_schools):
            x, y = fish_population.positions[i]
            energy = fish_population.energies[i]
            
            # Weight density by fish energy (larger/healthier schools matter more)
            ix, iy = int(np.clip(x, 0, self.width-1)), int(np.clip(y, 0, self.height-1))
            
            # Add density in 5x5 neighborhood (smoothing)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance < 3:
                            density[nx, ny] += energy * np.exp(-distance / 2.0)
        
        return density
    
    def _expand_mpa_regions(self, mpa_seed, iterations=2):
        """
        Expand MPA regions to make them more contiguous (connected components).
        Uses morphological dilation to grow protected zones.
        """
        mpa = mpa_seed.copy()
        
        for _ in range(iterations):
            new_mpa = mpa.copy()
            for x in range(self.width):
                for y in range(self.height):
                    if mpa[x, y] > 0.5:
                        # Spread to 4-connected neighbors
                        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                # 30% chance to spread to neighbor
                                if np.random.random() < 0.3:
                                    new_mpa[nx, ny] = 1.0
            mpa = new_mpa
        
        return mpa
    
    def is_in_mpa(self, x, y):
        """Check if a position is inside an MPA"""
        ix, iy = int(np.clip(x, 0, self.width-1)), int(np.clip(y, 0, self.height-1))
        return self.mpa_grid[ix, iy] > 0.5