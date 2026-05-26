# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------
# HIGHLY OPTIMIZED AGENT-BASED FISHERY MODEL (NUMBA / NUMPY)
#
# Logic: Dynamic Multi-Layered Environmental Model (Depth, Temp, Food-based Habitat)
# Movement: Combined Taxis (Food/Habitat + Density + Random)
# Scaling: Domain 300 km x 300 km, 100k Agents.
#
# By : OWUSU, Kwabena Afriyie (Optimized and Refactored by Gemini)
# Date : October 2025
#---------------------------------------------------------------------------

import random as rd
import math
import numpy as np
import time # Import time for runtime calculation
from statistics import mean
import csv # Keep CSV import for optional logging
import sys # Import sys for safe exit on no plot

# Conditional import for Numba for mandatory performance
try:
    import numba as nb
    from numba import njit
    from numba.typed import List as NumbaList # Import Numba's explicit list type
except ImportError:
    print("FATAL ERROR: Numba not found. Performance will be severely degraded.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    NumbaList = list
    prange = range 

# Conditional import for Matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.ndimage import convolve # For smoothing procedural habitat
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False

# --- GLOBAL CONSTANTS ---

# === 1. CONFIGURATION BLOCK (USER-DEFINED INPUTS) ===
LENGTH_AREA = 300                     # Linear length of the domain (km). 
ENV_GRID_DIM = 100                    # Resolution of the fine environmental grid (e.g., 100x100 cells)
MPA_AREA_PERCENTAGE = 0.20            # MPA size relative to total area (e.g., 20% of total area)

# Simulation Timing
SIMULATION_YEARS = 3
WEEKS_PER_YEAR = 52
n = SIMULATION_YEARS * WEEKS_PER_YEAR # Total simulation steps (weeks)
INIT_FISH = 10000                    # Initial agent count

# Catla Biology Parameters
OPTIMAL_TEMP = 25.0                   # Catla optimal temperature (°C)
OPTIMAL_DEPTH = 30.0                  # Catla optimal depth (m)

# Multi-Port Configuration (NEW)
NUM_RANDOM_PORTS = 15                 # Total number of ports to generate (approx. 2 West, 2 South)
MAX_PORT_LENGTH_KM = 10.0             # Maximum length of the port along the boundary
PORT_POLLUTION_DEPTH_FACTOR = 1/3     # Pollution depth factor (Pollution Zone Depth = Port Length * 2/3)
MIN_PORT_SPACING = 5.0                # Minimum required spacing between ports and boundary edges (km)


# === 2. VESSEL CONSTANTS (FIXED PROPERTIES - DO NOT CHANGE) ===
NUM_FISHERS = NUM_RANDOM_PORTS * 50
MOVE_FISHERS = 5000.0                 # MAX MECHANICAL SPEED (km/week).
FUEL_CAPACITY_LITERS = 600.0          
FUEL_EFFICIENCY_KMPL = 5.0            
MAX_FUEL_RANGE_KM = FUEL_CAPACITY_LITERS * FUEL_EFFICIENCY_KMPL 
Q = 0.7                               # Catchability coefficient.
R = 5.0                               # Sonar search radius (km). 
VESSEL_WEIGHT_CAPACITY = 300.0        # Realistic medium weekly capacity (kg).
MIN_HARVEST_LENGTH_M = 0.50           # Realistic Catch Length (m).
FIXED_OPERATIONAL_COST = 5000.0       # Fixed operational cost per vessel (Rupees).
UNIT_PRICE_PER_KG = 150.0             # Average price per kg (Rupees).
EXPLORATION_PROBABILITY = 0.50        
FUEL_PRICE_PER_LITER = 90.0           
PORT_SPAN_LENGTH = 6.0                # Removed in favor of dynamic PortAgent length, but kept for old reference
EXPLOIT_MEMORY_LIMIT = 5              # Max sub-steps a vessel can stay at the last known spot before searching.

# Vessel Return Status Codes for Logging
FUEL_EXHAUSTION_STATUS = {
    'NORMAL': 0,
    'FUEL_LOW_ABORT': 1,
    'HOLD_FULL_ABORT': 2,
    'FUEL_LOW_RETURNED': 3,
    'HOLD_FULL_RETURNED': 4
}


# === 3. DERIVED PARAMETERS (AUTOMATICALLY SCALED) ===
AREA = LENGTH_AREA * LENGTH_AREA
HALF_LENGTH_AREA = LENGTH_AREA / 2    

ENV_CELL_SIZE = LENGTH_AREA / ENV_GRID_DIM 

GRID_SIZE = LENGTH_AREA / 20.0        
GRID_DIM = int(LENGTH_AREA / GRID_SIZE) 

MPA_SIZE = AREA * MPA_AREA_PERCENTAGE
HALF_MPA_LENGTH = (math.sqrt(MPA_SIZE)) / 2 
Xa, Xb, Ya, Yb = -HALF_MPA_LENGTH, HALF_MPA_LENGTH, -HALF_MPA_LENGTH, HALF_MPA_LENGTH

# Vessel Calculations derived from FIXED R
R_SQR = R ** 2


# === 4. ECOSYSTEM & BIOLOGY CONSTANTS ===
K = 70000000                          
MOVE_FISH = 450.0                     
LARVAL_WEEKS = 4
JUVENILE_WEEKS = 52
ADULT_WEEKS_START = LARVAL_WEEKS + JUVENILE_WEEKS
# ADJUSTED MORTALITY: Lowered Juvenile and Adult rates for stability
MORTALITY = np.array([0.02, 0.001, 0.0005], dtype=np.float64) 
BREEDING_WEEKS_START = 23             
BREEDING_WEEKS_END = 35               
OBSERVE_FREQUENCY = 5                 
BIOMASS_OBSERVE_FREQUENCY = 12        
HABITAT_PLOT_FREQUENCY = 26           

# VBGF
L_INF = 1.20                          
K_GROWTH = 0.40                       
T0 = -0.10                            
A_W = 18.0                            
B_W = 3.0                             

# Recruitment
ALPHA = 0.05                          
BETA = 1.0e-7                         
RECRUITMENT_WEEK = BREEDING_WEEKS_END 
HABITAT_BONUS_FACTOR = 0.30           

# Primary Production / Consumption
MAX_PRIMARY_PRODUCTION = 1000.0       
LIGHT_EXTINCTION_COEFFICIENT = 0.05   
CONSUMPTION_PER_KG_FISH = 0.10        
MIN_HABITAT_SCORE = 0.1               
HABITAT_DEGRADE_RATE = 0.005          
HABITAT_RECOVERY_RATE = 0.001         
PORT_DEGRADATION_RATE = 0.002         # Maximum constant habitat pressure near port
HABITAT_NO_RECOVERY_ZONE = True       


# Taxis Weights (Defined here, passed to Numba function)
W_RANDOM = 0.70                       
W_DENSITY = 0.20                      
W_ENVIRONMENT = 0.10                  


# --- GLOBAL DATA STRUCTURES ---

FISH_DTYPE = [('x', 'f8'), ('y', 'f8'), ('age_weeks', 'i4'), ('length_m', 'f8'), 
             ('biomass_kg', 'f8'), ('stage', 'i1'), ('is_alive', 'b1')] 
FISH_DATA = np.empty((0,), dtype=FISH_DTYPE)

DEPTH_GRID = np.zeros((ENV_GRID_DIM, ENV_GRID_DIM), dtype=np.float64)
HABITAT_GRID = np.zeros((ENV_GRID_DIM, ENV_GRID_DIM), dtype=np.float64)
TEMP_GRID = np.zeros((ENV_GRID_DIM, ENV_GRID_DIM), dtype=np.float64)

BIOMASS_GRID = np.zeros((ENV_GRID_DIM, ENV_GRID_DIM), dtype=np.float64) 
FOOD_AVAILABILITY_GRID = np.zeros((ENV_GRID_DIM, ENV_GRID_DIM), dtype=np.float64) 

FISHER_AGENTS = []
GRID = {} 
DENSITY_GRID = np.zeros((ENV_GRID_DIM, ENV_GRID_DIM), dtype=np.int32) 

# Multi-Port Agents List (NEW)
PORT_AGENTS = [] 

# Simulation Loggers
LOG_FISH_BIOMASS = []
LOG_WEEKLY_CATCH_KG = [0.0] 
LOG_TOTAL_HARVEST_KG = [0.0]
LOG_PROFIT = [0.0] 
LOG_TIME = [0]
LOG_POPULATION = []
LOG_FISHER_WEEKLY = [] 
LOG_FLEET_SUMMARY = [] 

# --- PORT AGENT CLASS AND INITIALIZATION (NEW) ---

class PortAgent:
    def __init__(self, port_id, boundary, start_coord, end_coord, length_km):
        self.port_id = port_id
        self.boundary = boundary # 'West' or 'South'
        self.length_km = length_km
        self.pollution_depth = length_km * PORT_POLLUTION_DEPTH_FACTOR
        
        # Coordinates defining the physical span along the boundary
        if boundary == 'West':
            self.fixed_x = -HALF_LENGTH_AREA
            self.span_y = (start_coord, end_coord)
            self.center_x = self.fixed_x + self.pollution_depth * 0.5 
            self.center_y = (start_coord + end_coord) / 2
        else: # South
            self.fixed_y = -HALF_LENGTH_AREA
            self.span_x = (start_coord, end_coord)
            self.center_x = (start_coord + end_coord) / 2
            self.center_y = self.fixed_y + self.pollution_depth * 0.5
            
        self.center_coords = (self.center_x, self.center_y)
        self.decay_constant = self.pollution_depth / 2.0 # Controls decay speed

def initialize_ports():
    """Generates random ports along the West and South boundaries with overlap checking."""
    global PORT_AGENTS
    
    # Reset ports and initialize coordinates of existing ports
    PORT_AGENTS = []
    
    # We will try to place half the ports on the West and half on the South (rounded down for safety)
    num_west_ports = NUM_RANDOM_PORTS // 2
    num_south_ports = NUM_RANDOM_PORTS - num_west_ports
    
    port_counter = 1
    
    # --- Function to check overlap ---
    def check_overlap(new_start, new_end, existing_ports):
        """Checks if new segment (start, end) overlaps with any existing port segments."""
        for port in existing_ports:
            # Check for intersection between [port.start, port.end] and [new_start, new_end]
            # No overlap if new_end <= port.start OR new_start >= port.end
            port_start = port.span_y[0] if port.boundary == 'West' else port.span_x[0]
            port_end = port.span_y[1] if port.boundary == 'West' else port.span_x[1]
            
            # Use small buffer to account for spacing
            buffer = 0.1 
            
            if (new_end + buffer > port_start) and (new_start - buffer < port_end):
                return True # Overlap detected
        return False

    # --- 1. Place West Ports ---
    west_ports_list = []
    
    for i in range(num_west_ports):
        attempts = 0
        while attempts < 100:
            length = rd.uniform(3.0, MAX_PORT_LENGTH_KM)
            
            # Max possible starting coordinate, accounting for the length and min spacing at both ends
            max_start_y = HALF_LENGTH_AREA - length - MIN_PORT_SPACING
            min_start_y = -HALF_LENGTH_AREA + MIN_PORT_SPACING
            
            if max_start_y < min_start_y:
                 # Domain is too small or ports too big/numerous for this boundary
                 print(f"[WARNING] Cannot place {num_west_ports} West ports on {LENGTH_AREA}km edge without overlap.")
                 break
            
            start_y = rd.uniform(min_start_y, max_start_y)
            end_y = start_y + length

            # Check for overlap with already placed West ports
            if not check_overlap(start_y, end_y, west_ports_list):
                port = PortAgent(port_id=port_counter, boundary='West', start_coord=start_y, end_coord=end_y, length_km=length)
                PORT_AGENTS.append(port)
                west_ports_list.append(port)
                port_counter += 1
                break # Successfully placed
            attempts += 1
            
    # --- 2. Place South Ports ---
    south_ports_list = []

    for i in range(num_south_ports):
        attempts = 0
        while attempts < 100:
            length = rd.uniform(3.0, MAX_PORT_LENGTH_KM)
            
            max_start_x = HALF_LENGTH_AREA - length - MIN_PORT_SPACING
            min_start_x = -HALF_LENGTH_AREA + MIN_PORT_SPACING
            
            if max_start_x < min_start_x:
                 print(f"[WARNING] Cannot place {num_south_ports} South ports on {LENGTH_AREA}km edge without overlap.")
                 break

            start_x = rd.uniform(min_start_x, max_start_x)
            end_x = start_x + length

            # Check for overlap with already placed South ports
            if not check_overlap(start_x, end_x, south_ports_list):
                port = PortAgent(port_id=port_counter, boundary='South', start_coord=start_x, end_coord=end_x, length_km=length)
                PORT_AGENTS.append(port)
                south_ports_list.append(port)
                port_counter += 1
                break # Successfully placed
            attempts += 1

# --- CORE SIMULATION FUNCTIONS (NUMBA ACCELERATED) ---

# Helper function to convert coordinates to environmental grid indices
@njit(cache=True)
def get_env_grid_coords(x, y):
    cell_x = int((x + HALF_LENGTH_AREA) / ENV_CELL_SIZE)
    cell_y = int((y + HALF_LENGTH_AREA) / ENV_CELL_SIZE)
    # Clamp to grid bounds (0 to ENV_GRID_DIM - 1)
    cell_x = max(0, min(cell_x, ENV_GRID_DIM - 1))
    cell_y = max(0, min(cell_y, ENV_GRID_DIM - 1))
    return cell_x, cell_y

# Stage index conversion 
@njit(cache=True)
def get_stage_index(age_weeks):
    if age_weeks <= LARVAL_WEEKS:
        return 0 # Larval
    elif age_weeks <= ADULT_WEEKS_START:
        return 1 # Juvenile
    else:
        return 2 # Adult

# Core calculation for fish size and biomass using VBGF
@njit(cache=True)
def calculate_length_and_biomass(age_weeks):
    # Convert age from weeks to years
    age_years = age_weeks / WEEKS_PER_YEAR 
    
    # 1. Von Bertalanffy Growth Function (VBGF)
    length_m = L_INF * (1.0 - np.exp(-K_GROWTH * (age_years - T0)))
    
    # 2. Weight-Length Relationship
    biomass_kg = A_W * (length_m ** B_W)
    
    stage = get_stage_index(age_weeks)
    
    return length_m, biomass_kg, stage

# Main physics loop: Movement, Aging, and Mortality
@njit(cache=True)
def run_fish_physics_numba(fish_array, mortality_rates, current_week_of_year, carrying_capacity, depth_grid, temp_grid, habitat_grid, density_grid, w_random, w_density, w_environment):
    """
    Handles movement, aging, and mortality checks using Combined Taxis.
    """
    N = len(fish_array)
    
    rand_theta_array = 2.0 * np.pi * np.random.rand(N) 
    
    for i in range(N):
        if not fish_array['is_alive'][i]:
            continue 

        x, y = fish_array['x'][i], fish_array['y'][i]
        
        # --- 1. MOVEMENT Taxis Calculation ---
        
        theta_rand = rand_theta_array[i]
        
        # 1b. Density Taxis (Repulsion from high density)
        cx, cy = get_env_grid_coords(x, y)
        current_density = density_grid[cx, cy]
        
        K_per_cell = carrying_capacity / (ENV_GRID_DIM * ENV_GRID_DIM) 
        
        if current_density > K_per_cell * 0.5: 
            center_x = cx * ENV_CELL_SIZE + ENV_CELL_SIZE / 2 - HALF_LENGTH_AREA
            center_y = cy * ENV_CELL_SIZE + ENV_CELL_SIZE / 2 - HALF_LENGTH_AREA
            
            deltax_density = x - center_x
            deltay_density = y - center_y
            theta_density = np.arctan2(deltay_density, deltax_density)
        else:
            theta_density = theta_rand 
            
        # 1c. Environmental Taxis (Attraction to optimal T/D/H)
        best_suitability = -1.0
        theta_env = theta_rand 
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0: continue
                
                nx = cx + dx
                ny = cy + dy
                
                nx_clamped = (nx + ENV_GRID_DIM) % ENV_GRID_DIM
                ny_clamped = (ny + ENV_GRID_DIM) % ENV_GRID_DIM
                
                temp = temp_grid[nx_clamped, ny_clamped]
                depth = depth_grid[nx_clamped, ny_clamped]
                habitat = habitat_grid[nx_clamped, ny_clamped]
                
                # Suitability Index
                temp_diff = np.abs(temp - OPTIMAL_TEMP) / 10.0 
                depth_diff = np.abs(depth - OPTIMAL_DEPTH) / 50.0 
                
                suitability = habitat - temp_diff - depth_diff 
                
                if suitability > best_suitability:
                    best_suitability = suitability
                    
                    target_x = nx_clamped * ENV_CELL_SIZE + ENV_CELL_SIZE / 2 - HALF_LENGTH_AREA
                    target_y = ny_clamped * ENV_CELL_SIZE + ENV_CELL_SIZE / 2 - HALF_LENGTH_AREA
                    
                    deltax_env = target_x - x
                    deltay_env = target_y - y
                    theta_env = np.arctan2(deltay_env, deltax_env)

        # 1d. Final Movement: Weighted Taxis Combination
        theta_final = (w_random * theta_rand + 
                       w_density * theta_density + 
                       w_environment * theta_env)

        # Apply movement
        fish_array['x'][i] += MOVE_FISH * np.cos(theta_final)
        fish_array['y'][i] += MOVE_FISH * np.sin(theta_final)

        # Apply Periodic Boundary Conditions (wrap-around)
        if fish_array['x'][i] > HALF_LENGTH_AREA: fish_array['x'][i] -= LENGTH_AREA
        elif fish_array['x'][i] < -HALF_LENGTH_AREA: fish_array['x'][i] += LENGTH_AREA

        if fish_array['y'][i] > HALF_LENGTH_AREA: fish_array['y'][i] -= LENGTH_AREA
        elif fish_array['y'][i] < -HALF_LENGTH_AREA: fish_array['y'][i] += LENGTH_AREA

        # --- 2. LIFE CYCLE (Aging and Mortality) ---
        stage_index = fish_array['stage'][i]
        
        # Mortality Check
        if np.random.rand() < mortality_rates[stage_index]:
            fish_array['is_alive'][i] = False # Mark as dead
            continue 

        # Aging
        fish_array['age_weeks'][i] += 1
        
        # Recalculate size and stage based on new age (using VBGF)
        new_L, new_B, new_S = calculate_length_and_biomass(fish_array['age_weeks'][i])
        fish_array['length_m'][i] = new_L
        fish_array['biomass_kg'][i] = new_B
        fish_array['stage'][i] = new_S
        
    return NumbaList([
        (0.0, 0.0, 0, 0.0, 0.0, 0, True)
    ])[:0] 

# Function to check if a point is inside the single MPA.
@njit(cache=True)
def is_in_mpa_numba(x, y):
    """Checks if a point is inside the single MPA (Numba compatible)."""
    return (Xa <= x <= Xb) and (Ya <= y <= Yb)

# --- ENVIRONMENTAL SETUP ---

def initialize_environmental_layers():
    """Generates the static Depth and initial Habitat layers."""
    global DEPTH_GRID, HABITAT_GRID
    
    # --- 1. DEPTH GRID (Procedural Terrain) ---
    x = np.linspace(-HALF_LENGTH_AREA, HALF_LENGTH_AREA, ENV_GRID_DIM)
    y = np.linspace(-HALF_LENGTH_AREA, HALF_LENGTH_AREA, ENV_GRID_DIM)
    X, Y = np.meshgrid(x, y)
    
    # 1a. Generate Pure Procedural Noise for Terrain Variation (No Central Radial Bias)
    
    # Layer 1: Small scale roughness
    depth_raw_1 = np.random.rand(ENV_GRID_DIM, ENV_GRID_DIM)
    kernel_depth_1 = np.ones((10, 10)) / 100.0 
    depth_noise_1 = convolve(depth_raw_1, kernel_depth_1, mode='wrap') * 40.0 

    # Layer 2: Medium scale features (Hills/Valleys)
    depth_raw_2 = np.random.rand(ENV_GRID_DIM, ENV_GRID_DIM)
    kernel_depth_2 = np.ones((25, 25)) / 625.0 
    depth_noise_2 = convolve(depth_raw_2, kernel_depth_2, mode='wrap') * 60.0 

    # Layer 3: Large scale features (Plateaus/Trenches)
    depth_raw_3 = np.random.rand(ENV_GRID_DIM, ENV_GRID_DIM)
    kernel_depth_3 = np.ones((40, 40)) / 1600.0 
    depth_noise_3 = convolve(depth_raw_3, kernel_depth_3, mode='wrap') * 80.0 

    # Weighted combination: 
    # Use Layer 3 for dominant topography, Layer 2 for main features, Layer 1 for small roughness.
    # Normalize noise sum to range from 0 to 1
    total_noise = 0.4 * depth_noise_3 + 0.4 * depth_noise_2 + 0.2 * depth_noise_1
    total_noise_normalized = (total_noise - total_noise.min()) / (total_noise.max() - total_noise.min())
    
    # Scale normalized noise to a full depth range (e.g., 5m to 200m)
    DEPTH_RANGE = 195.0 # 200m - 5m
    MIN_DEPTH = 5.0
    
    DEPTH_GRID = MIN_DEPTH + total_noise_normalized * DEPTH_RANGE
    
    # 1b. Shallow Water Buffer Enforcement (CRITICAL FIX for Ports)
    SHALLOW_SHELF_KM = 20.0 
    MAX_SHORE_DEPTH = 10.0  
    
    dist_x = np.minimum(X + HALF_LENGTH_AREA, HALF_LENGTH_AREA - X)
    dist_y = np.minimum(Y + HALF_LENGTH_AREA, HALF_LENGTH_AREA - Y)
    dist_to_shore = np.minimum(dist_x, dist_y)
    
    shallow_mask = dist_to_shore < SHALLOW_SHELF_KM
    
    shallow_factor = dist_to_shore[shallow_mask] / SHALLOW_SHELF_KM
    
    blended_depth = MAX_SHORE_DEPTH + (DEPTH_GRID[shallow_mask] - MAX_SHORE_DEPTH) * shallow_factor
    
    DEPTH_GRID[shallow_mask] = np.minimum(DEPTH_GRID[shallow_mask], blended_depth)
    DEPTH_GRID = np.clip(DEPTH_GRID, 2.0, 200.0) 

    # --- 2. HABITAT GRID (Probabilistic Multi-Layered Noise) ---
    
    habitat_raw_1 = np.random.rand(ENV_GRID_DIM, ENV_GRID_DIM)
    kernel_1 = np.ones((5, 5)) / 25.0
    habitat_layer_1 = convolve(habitat_raw_1, kernel_1, mode='wrap')
    
    habitat_raw_2 = np.random.rand(ENV_GRID_DIM, ENV_GRID_DIM)
    kernel_2 = np.ones((15, 15)) / 225.0
    habitat_layer_2 = convolve(habitat_raw_2, kernel_2, mode='wrap')
    
    habitat_raw_3 = np.random.rand(ENV_GRID_DIM, ENV_GRID_DIM)
    kernel_3 = np.ones((30, 30)) / 900.0
    habitat_layer_3 = convolve(habitat_raw_3, kernel_3, mode='wrap')
    
    HABITAT_GRID = (0.5 * habitat_layer_3 + 0.3 * habitat_layer_2 + 0.2 * habitat_layer_1)
    
    HABITAT_GRID = (HABITAT_GRID - HABITAT_GRID.min()) / (HABITAT_GRID.max() - HABITAT_GRID.min())

def update_temperature_grid(time1):
    """Generates the dynamic Temperature layer based on season and depth."""
    global TEMP_GRID
    
    current_week_of_year = time1 % WEEKS_PER_YEAR
    
    seasonal_factor = (np.sin(2 * np.pi * (current_week_of_year - 13) / WEEKS_PER_YEAR) + 1) / 2 
    
    BASE_TEMP = 20.0
    SEASONAL_RANGE = 10.0
    DEPTH_PENALTY = 0.05 
    
    shallow_temp = BASE_TEMP + SEASONAL_RANGE * seasonal_factor
    
    TEMP_GRID = shallow_temp - DEPTH_PENALTY * (DEPTH_GRID - 10.0) 
    TEMP_GRID = np.clip(TEMP_GRID, 15.0, 35.0) 

def calculate_primary_production():
    """Calculates weekly food production based on depth (sunlight penetration)."""
    global FOOD_AVAILABILITY_GRID, DEPTH_GRID
    
    light_factor = np.exp(-LIGHT_EXTINCTION_COEFFICIENT * DEPTH_GRID)
    PP_PER_KM2_WEEK = MAX_PRIMARY_PRODUCTION * light_factor
    
    ENV_AREA_KM2 = (ENV_CELL_SIZE**2) 
    FOOD_AVAILABILITY_GRID = PP_PER_KM2_WEEK * ENV_AREA_KM2 

def build_biomass_grid():
    """Calculates the current fish biomass in the 100x100 environment grid (O(N))."""
    global BIOMASS_GRID, FISH_DATA
    BIOMASS_GRID.fill(0.0)
    
    alive_indices = np.where(FISH_DATA['is_alive'])[0]
    
    for i in alive_indices:
        x, y = FISH_DATA['x'][i], FISH_DATA['y'][i]
        cx, cy = get_env_grid_coords(x, y)
        BIOMASS_GRID[cx, cy] += FISH_DATA['biomass_kg'][i]
        
def update_habitat_dynamics():
    """
    Updates Habitat Quality based on Food Pressure and applies Multi-Port Exponential Degradation.
    """
    global HABITAT_GRID, BIOMASS_GRID, FOOD_AVAILABILITY_GRID
    
    # 1. Food Pressure Dynamics (Consumption vs. Production)
    consumption_grid = BIOMASS_GRID * CONSUMPTION_PER_KG_FISH
    
    epsilon = 1e-6
    food_pressure_ratio = consumption_grid / (FOOD_AVAILABILITY_GRID + epsilon)
    
    change_factor = food_pressure_ratio - 1.0
    
    degradation = np.clip(change_factor, 0, None) * HABITAT_DEGRADE_RATE
    recovery = np.clip(-change_factor, 0, None) * HABITAT_RECOVERY_RATE
    
    # Apply change globally
    HABITAT_GRID += recovery - degradation
    
    # 2. Apply Multi-Port Exponential Degradation (NEW LOGIC)
    
    # Generate grid coordinates (cell center points) once
    x_coords = np.linspace(-HALF_LENGTH_AREA + ENV_CELL_SIZE/2, HALF_LENGTH_AREA - ENV_CELL_SIZE/2, ENV_GRID_DIM)
    y_coords = np.linspace(-HALF_LENGTH_AREA + ENV_CELL_SIZE/2, HALF_LENGTH_AREA - ENV_CELL_SIZE/2, ENV_GRID_DIM)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    # Initialize pollution effect array
    pollution_effect = np.zeros_like(HABITAT_GRID)
    
    for port in PORT_AGENTS:
        # Calculate shortest distance from every grid cell center to the port's active boundary segment.
        if port.boundary == 'West':
            # Distance from fixed X boundary segment
            # Clamp Y coordinates to the port's span
            y_clamped = np.clip(Y_grid, port.span_y[0], port.span_y[1])
            
            # Calculate distance from the fixed X plane to the clamped Y point
            dist_to_span = np.sqrt( (X_grid - port.fixed_x)**2 + (Y_grid - y_clamped)**2 )
            
            # Distance outward from the boundary
            distance = np.abs(X_grid - port.fixed_x)
            
        else: # South boundary
            # Clamp X coordinates to the port's span
            x_clamped = np.clip(X_grid, port.span_x[0], port.span_x[1])
            
            # Calculate distance from the fixed Y plane to the clamped X point
            dist_to_span = np.sqrt( (X_grid - x_clamped)**2 + (Y_grid - port.fixed_y)**2 )
            
            # Distance outward from the boundary
            distance = np.abs(Y_grid - port.fixed_y)

        # Apply exponential decay based on distance from the span
        # Decay is maximum (PORT_DEGRADATION_RATE) at the shoreline and falls off exponentially.
        # Max pollution zone depth is port.pollution_depth
        
        # Only apply degradation within the pollution depth
        mask = distance < port.pollution_depth
        
        # Exponential decay factor (1 at boundary, near 0 at pollution_depth)
        decay_factor = np.exp(-dist_to_span / port.decay_constant) 
        
        # Add this port's pollution to the total effect
        pollution_effect[mask] += PORT_DEGRADATION_RATE * decay_factor[mask] 

    # Apply the total max pollution effect (clamped so pollution doesn't exceed 1.0)
    HABITAT_GRID -= pollution_effect
    
    # 3. Clamp HABITAT_GRID between MIN_HABITAT_SCORE and 1.0
    HABITAT_GRID = np.clip(HABITAT_GRID, MIN_HABITAT_SCORE, 1.0)


# --- RECRUITMENT AND COMPACTION LOGIC ---

def calculate_recruitment(fish_data):
    """
    Calculates total annual recruitment using the Beverton-Holt equation 
    and distributes new recruits spatially based on Habitat Quality.
    
    CRITICAL: This function ensures all births are spatially biased (non-uniform).
    """
    adult_mask = (fish_data['stage'] == 2) & (fish_data['is_alive'])
    ssb_kg = fish_data['biomass_kg'][adult_mask].sum()

    if ssb_kg < 1.0: total_recruits = 0
    else: total_recruits = int(round((ALPHA * ssb_kg) / (1.0 + BETA * ssb_kg)))
    if total_recruits <= 0: return []

    adult_fish = fish_data[adult_mask]
    
    # 1. Calculate weighted SSB contribution based on habitat quality for spatial bias
    habitat_contributions = []
    spawning_adults = []
    
    # If no adults are left, distribution defaults to uniform (but that should be rare)
    if len(adult_fish) == 0:
        recruits_uniformly = total_recruits
        new_larvae = []
    else:
        for i in range(len(adult_fish)):
              x, y = adult_fish['x'][i], adult_fish['y'][i]
              cx, cy = get_env_grid_coords(x, y)
              habitat_quality = HABITAT_GRID[cx, cy]
              # Weight is proportional to biomass * quality
              weight = adult_fish['biomass_kg'][i] * (1.0 + habitat_quality * HABITAT_BONUS_FACTOR)
              habitat_contributions.append(weight)
              spawning_adults.append((x,y))

        total_weight = np.sum(habitat_contributions)
        
        # If total weight is near zero (floating point safety), default to uniform placement
        if total_weight < 1e-6:
            recruits_uniformly = total_recruits
        else:
            # All recruits are generated based on biased placement
            recruits_in_biased_zones = total_recruits 
            recruits_uniformly = 0
            
            # Generate Biased Recruits (placed near high habitat quality adults)
            parent_indices = np.random.choice(len(spawning_adults), size=recruits_in_biased_zones, 
                                             p=np.array(habitat_contributions) / total_weight)
            
            new_larvae = []
            for index in parent_indices:
                px, py = spawning_adults[index]
                # Spawn near the parent (local dispersal)
                x_pos = px + np.random.uniform(-ENV_CELL_SIZE, ENV_CELL_SIZE)
                y_pos = py + np.random.uniform(-ENV_CELL_SIZE, ENV_CELL_SIZE)
                
                new_larvae.append((x_pos, y_pos, 0, 0.0, 0.0, 0, True))
    
    # Fallback to uniform distribution if no adults could be weighted/selected
    if recruits_uniformly > 0:
        for _ in range(recruits_uniformly):
            x_pos = np.random.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
            y_pos = np.random.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
            new_larvae.append((x_pos, y_pos, 0, 0.0, 0.0, 0, True))
            
    return new_larvae

def compact_fish_data(new_larvae):
    """
    Removes dead/caught agents and adds new larvae (Annual Compaction).
    """
    global FISH_DATA
    
    # 1. Filter out dead/caught fish (is_alive == False)
    FISH_DATA = FISH_DATA[FISH_DATA['is_alive']]
    
    # 2. Add new larvae (births)
    if new_larvae:
        new_larvae_array = np.array(new_larvae, dtype=FISH_DTYPE)
        FISH_DATA = np.concatenate((FISH_DATA, new_larvae_array))
    

def build_density_grid():
    """Calculates the current fish density in the 100x100 environment grid (O(N))."""
    global DENSITY_GRID
    DENSITY_GRID.fill(0) # Reset
    
    alive_indices = np.where(FISH_DATA['is_alive'])[0]
    
    for i in alive_indices:
        x, y = FISH_DATA['x'][i], FISH_DATA['y'][i]
        cx, cy = get_env_grid_coords(x, y)
        DENSITY_GRID[cx, cy] += 1
        
def build_fisher_search_grid():
    """Populates the global GRID dictionary with indices of ALIVE fish (O(N))."""
    global GRID
    GRID.clear()
    
    alive_indices = np.where(FISH_DATA['is_alive'])[0]
    
    for i in alive_indices:
        x, y = FISH_DATA['x'][i], FISH_DATA['y'][i]
        
        cell_x = int((x + HALF_LENGTH_AREA) / GRID_SIZE)
        cell_y = int((y + HALF_LENGTH_AREA) / GRID_SIZE)

        cell_x = max(0, min(cell_x, GRID_DIM - 1))
        cell_y = max(0, min(cell_y, GRID_DIM - 1))

        cell_key = (cell_x, cell_y)
        if cell_key not in GRID:
            GRID[cell_key] = []
        GRID[cell_key].append(i)

# --- AGENT CREATION AND INITIALIZATION ---

def create_fish_agents(num_agents):
    """
    Initializes the NumPy array with fish data using schooled dispersal.
    """
    new_data = np.zeros(num_agents, dtype=FISH_DTYPE)
    total_placed = 0
    
    # Schooled Dispersal Logic
    while total_placed < num_agents:
        school_size = rd.randint(50, 100)
        school_size = min(school_size, num_agents - total_placed)
        
        # Center of the school is randomly placed
        center_x = np.random.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
        center_y = np.random.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
        
        # Small dispersal radius for the school
        school_radius = 5.0 # km
        
        start_index = total_placed
        end_index = total_placed + school_size
        
        # Place school members near the center point
        new_data['x'][start_index:end_index] = np.random.normal(center_x, school_radius, school_size)
        new_data['y'][start_index:end_index] = np.random.normal(center_y, school_radius, school_size)
        
        # Assign random ages and mark as alive
        new_data['age_weeks'][start_index:end_index] = np.random.randint(0, SIMULATION_YEARS * WEEKS_PER_YEAR, school_size)
        new_data['is_alive'][start_index:end_index] = True 
        
        total_placed += school_size

    # Calculate initial physical properties (Length, Biomass, Stage)
    for i in range(num_agents):
        L, B, S = calculate_length_and_biomass(new_data['age_weeks'][i])
        new_data['length_m'][i] = L
        new_data['biomass_kg'][i] = B
        new_data['stage'][i] = S
    
    return new_data

# Fisher agent object (minimal Python object overhead)
class FisherAgent:
    def __init__(self, num):
        self.num = f'fisher{num}'
        self.harvest = 0.0 # Cumulative harvest
        self.weekly_weight_catch = 0.0
        self.weekly_revenue = 0.0
        self.weekly_distance_traveled = 0.0
        self.effort = 0.7
        
        # Home port coordinates will be set during initialize()
        self.home_port_coords = (0.0, 0.0) 
        
        self.x = 0.0 
        self.y = 0.0
        self.last_successful_x = self.x
        self.last_successful_y = self.y
        
        self.fuel_L = FUEL_CAPACITY_LITERS 
        self.weekly_fish_count = 0 
        self.return_status = FUEL_EXHAUSTION_STATUS['NORMAL'] 
        self.last_trip_profitable = True 
        
        # NEW: Counter for exploitation limit
        self.exploit_steps = 0 

# --- FISHERMAN LOGIC AND TRANSACTIONS ---

def move_fisherman(fisherman_ag, target_x, target_y):
    """Calculates and applies movement toward a target, handling fuel/distance budget."""
    
    current_x, current_y = fisherman_ag.x, fisherman_ag.y
    
    # Maximum single-step distance based on remaining capacity (time/fuel)
    max_weekly_move = MAX_FUEL_RANGE_KM - fisherman_ag.weekly_distance_traveled
    remaining_fuel_range = fisherman_ag.fuel_L * FUEL_EFFICIENCY_KMPL
    
    deltax = target_x - current_x
    deltay = target_y - current_y
    distance_to_target = math.sqrt(deltax**2 + deltay**2)
    
    # The actual distance the vessel can cover this step:
    move_distance = min(
        max_weekly_move, 
        distance_to_target,
        remaining_fuel_range
    )
    
    if move_distance < 0: move_distance = 0.0 

    if distance_to_target > 0 and move_distance > 0:
        theta = math.atan2(deltay, deltax)
        new_x = current_x + move_distance * math.cos(theta)
        new_y = current_y + move_distance * math.sin(theta)
    else:
        return

    # Clamping boundaries 
    fisherman_ag.x = max(-HALF_LENGTH_AREA, min(HALF_LENGTH_AREA, new_x))
    fisherman_ag.y = max(-HALF_LENGTH_AREA, min(HALF_LENGTH_AREA, new_y))
    
    # Log distance traveled (and consume fuel)
    fisherman_ag.weekly_distance_traveled += move_distance
    fisherman_ag.fuel_L -= move_distance / FUEL_EFFICIENCY_KMPL


def get_harvestable_fish_indices(fisherman_ag):
    """Finds all harvestable fish indices within sonar range R."""
    focal_cell_x = int((fisherman_ag.x + HALF_LENGTH_AREA) / GRID_SIZE)
    focal_cell_y = int((fisherman_ag.y + HALF_LENGTH_AREA) / GRID_SIZE)

    harvestable_indices = []
    
    # Search the 9 surrounding cells 
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell_x, cell_y = focal_cell_x + dx, focal_cell_y + dy
            
            if 0 <= cell_x < GRID_DIM and 0 <= cell_y < GRID_DIM:
                cell_key = (cell_x, cell_y)
                if cell_key in GRID:
                    for fish_index in GRID[cell_key]:
                        dist_sqr = (FISH_DATA['x'][fish_index] - fisherman_ag.x)**2 + (FISH_DATA['y'][fish_index] - fisherman_ag.y)**2
                        
                        if dist_sqr < R_SQR:
                            if not is_in_mpa_numba(FISH_DATA['x'][fish_index], FISH_DATA['y'][fish_index]):
                                if FISH_DATA['length_m'][fish_index] >= MIN_HARVEST_LENGTH_M:
                                    harvestable_indices.append(fish_index)
                                    
    return harvestable_indices

def get_best_fishing_spot(fisherman_ag):
    """Finds the best fishing grid cell within sonar/search radius R."""
    best_biomass = -1
    best_cell_coords = None

    search_radius_cells = int(R / GRID_SIZE) + 1 
    
    focal_cell_x = int((fisherman_ag.x + HALF_LENGTH_AREA) / GRID_SIZE)
    focal_cell_y = int((fisherman_ag.y + HALF_LENGTH_AREA) / GRID_SIZE)

    for dx in range(-search_radius_cells, search_radius_cells + 1):
        for dy in range(-search_radius_cells, search_radius_cells + 1):
            cell_x, cell_y = focal_cell_x + dx, focal_cell_y + dy
            
            if 0 <= cell_x < GRID_DIM and 0 <= cell_y < GRID_DIM:
                cell_key = (cell_x, cell_y)
                
                if cell_key in GRID:
                    cell_center_x = cell_x * GRID_SIZE + GRID_SIZE/2 - HALF_LENGTH_AREA
                    cell_center_y = cell_y * GRID_SIZE + GRID_SIZE/2 - HALF_LENGTH_AREA

                    if not is_in_mpa_numba(cell_center_x, cell_center_y):
                        current_biomass = sum(FISH_DATA['biomass_kg'][idx] for idx in GRID[cell_key] if FISH_DATA['length_m'][idx] >= MIN_HARVEST_LENGTH_M)
                        
                        if current_biomass > best_biomass:
                            best_biomass = current_biomass
                            best_cell_coords = (cell_center_x, cell_center_y)
                            
    return best_cell_coords

def fisher_transaction_and_reset(time1):
    """
    Handles the economic transaction and logging for the *previous* week (time1 - 1).
    Enforces that the vessel starts the current week AT its home port.
    """
    global FISHER_AGENTS, LOG_PROFIT, LOG_WEEKLY_CATCH_KG
    
    if time1 == 0:
        LOG_PROFIT.append(0.0)
        LOG_WEEKLY_CATCH_KG.append(0.0)
        return 0.0, 0.0

    total_weekly_profit_realized = 0.0
    current_weekly_catch_kg_fleet = 0.0 
    FIXED_OPERATIONAL_COST_VESSEL = FIXED_OPERATIONAL_COST

    for f in FISHER_AGENTS:
        
        # Logged position is the Port (where the transaction occurs)
        # We only log the port coordinates for the transaction since mandatory return is enforced
        logged_x, logged_y = f.home_port_coords

        distance_km = f.weekly_distance_traveled
        fuel_spent_L = distance_km / FUEL_EFFICIENCY_KMPL
        variable_fuel_cost = fuel_spent_L * FUEL_PRICE_PER_LITER 
        
        total_trip_cost = FIXED_OPERATIONAL_COST_VESSEL + variable_fuel_cost
        
        current_trip_profit = f.weekly_revenue - total_trip_cost
        
        f.last_trip_profitable = current_trip_profit >= 0
        
        # --- LOGGING THE RESULTS OF WEEK (time1 - 1) ---
        LOG_FISHER_WEEKLY.append({
            'Week': time1, 
            'Fisherman_ID': f.num,
            'Catch_Count': f.weekly_fish_count,
            'Catch_KG': f.weekly_weight_catch,
            'Revenue_Realized': f.weekly_revenue,
            'Fixed_Cost': FIXED_OPERATIONAL_COST_VESSEL, 
            'Variable_Fuel_Cost': variable_fuel_cost, 
            'Total_Trip_Cost': total_trip_cost,
            'Profit_Realized': current_trip_profit, 
        })
        
        # 1a. Realize profit and catch from the *previous* week's trip (T-1)
        total_weekly_profit_realized += current_trip_profit
        current_weekly_catch_kg_fleet += f.weekly_weight_catch

        # 1b. Reset state for the *current* week (T) - CRITICAL: Vessel starts at Port
        f.weekly_weight_catch = 0.0 
        f.weekly_revenue = 0.0
        f.weekly_distance_traveled = 0.0
        f.fuel_L = FUEL_CAPACITY_LITERS # Full tank
        f.weekly_fish_count = 0
        f.return_status = FUEL_EXHAUSTION_STATUS['NORMAL'] 
        f.exploit_steps = 0 # Reset exploitation counter
        
        # Set vessel starting position for the current week (T) to its Home Port
        f.x, f.y = f.home_port_coords
        
    # Update fleet-level logs
    LOG_PROFIT.append(LOG_PROFIT[-1] + total_weekly_profit_realized)
    LOG_WEEKLY_CATCH_KG.append(current_weekly_catch_kg_fleet) 

    return total_weekly_profit_realized, current_weekly_catch_kg_fleet


def execute_fishing_and_movement(time1):
    """Executes all fisherman movement and harvest logic for the current week (time1)."""
    global FISH_DATA, FISHER_AGENTS
    
    current_week_of_year = time1 % WEEKS_PER_YEAR
    is_fishing_banned = BREEDING_WEEKS_START <= current_week_of_year <= BREEDING_WEEKS_END
    
    t = 0.
    while t < 1. and len(FISHER_AGENTS) > 0:
        t += 1. / len(FISHER_AGENTS)
        
        fisherman_ag = rd.choice(FISHER_AGENTS)
        
        # HARVESTING 
        if not is_fishing_banned:
            fish_indices_in_range = get_harvestable_fish_indices(fisherman_ag)
            
            potential_harvest_count = int(round(Q * fisherman_ag.effort * len(fish_indices_in_range)))

            current_harvest_weight = 0.0
            remaining_capacity = VESSEL_WEIGHT_CAPACITY - fisherman_ag.weekly_weight_catch
            indices_to_remove = []
            
            if fish_indices_in_range:
                sample_indices = rd.sample(fish_indices_in_range, min(len(fish_indices_in_range), potential_harvest_count))
                
                for index in sample_indices:
                    fish_weight = FISH_DATA['biomass_kg'][index]
                    if current_harvest_weight + fish_weight <= remaining_capacity:
                        indices_to_remove.append(index)
                        current_harvest_weight += fish_weight
                        
                if indices_to_remove:
                    # Update success memory and exploitation counter
                    fisherman_ag.last_successful_x = fisherman_ag.x
                    fisherman_ag.last_successful_y = fisherman_ag.y
                    fisherman_ag.harvest += current_harvest_weight
                    fisherman_ag.weekly_weight_catch += current_harvest_weight
                    fisherman_ag.weekly_revenue += current_harvest_weight * UNIT_PRICE_PER_KG
                    fisherman_ag.weekly_fish_count += len(indices_to_remove) 
                    
                    for index in indices_to_remove:
                        FISH_DATA['is_alive'][index] = False 

        # MOVEMENT 
        dist_to_port = math.sqrt((fisherman_ag.x - fisherman_ag.home_port_coords[0])**2 + (fisherman_ag.y - fisherman_ag.home_port_coords[1])**2)
        required_fuel_home = dist_to_port / FUEL_EFFICIENCY_KMPL
        
        is_critically_low_on_fuel = fisherman_ag.fuel_L < (required_fuel_home + (0.10 * FUEL_CAPACITY_LITERS))
        is_near_capacity = fisherman_ag.weekly_weight_catch >= (0.80 * VESSEL_WEIGHT_CAPACITY)
        
        # Determine Target Coordinates for this step
        if is_critically_low_on_fuel or is_near_capacity:
            target_x, target_y = fisherman_ag.home_port_coords
            
            if is_critically_low_on_fuel:
                fisherman_ag.return_status = FUEL_EXHAUSTION_STATUS['FUEL_LOW_ABORT']
            elif is_near_capacity:
                fisherman_ag.return_status = FUEL_EXHAUSTION_STATUS['HOLD_FULL_ABORT']
        else:
            # Strategic Movement logic (Exploit/Explore)
            best_spot_local = get_best_fishing_spot(fisherman_ag)
            
            if best_spot_local and fisherman_ag.exploit_steps < EXPLOIT_MEMORY_LIMIT:
                # Exploitation: move to the detected local hotspot
                target_x, target_y = best_spot_local
                fisherman_ag.exploit_steps += 1
            elif fisherman_ag.last_trip_profitable and fisherman_ag.exploit_steps < EXPLOIT_MEMORY_LIMIT:
                # Exploitation fallback: move back to last successful spot (memory)
                target_x, target_y = fisherman_ag.last_successful_x, fisherman_ag.last_successful_y
                fisherman_ag.exploit_steps += 1
            else:
                # Targeted Exploration (triggered by loss or hitting exploitation limit)
                theta = 2 * math.pi * rd.random()
                explore_distance = R * 5.0 * rd.random() 
                target_x = fisherman_ag.last_successful_x + explore_distance * math.cos(theta) 
                target_y = fisherman_ag.last_successful_y + explore_distance * math.sin(theta)
                
                target_x = max(-HALF_LENGTH_AREA, min(HALF_LENGTH_AREA, target_x))
                target_y = max(-HALF_LENGTH_AREA, min(HALF_LENGTH_AREA, target_y))
                # Reset exploitation counter as they are now exploring
                fisherman_ag.exploit_steps = 0 
                
        
        # Core Constraint Check: Fuel safety for round trip when moving away from home port
        if target_x != fisherman_ag.home_port_coords[0] or target_y != fisherman_ag.home_port_coords[1]:
            dist_to_target_current = math.sqrt((fisherman_ag.x - target_x)**2 + (fisherman_ag.y - target_y)**2)
            dist_target_to_port = math.sqrt((target_x - fisherman_ag.home_port_coords[0])**2 + (target_y - fisherman_ag.home_port_coords[1])**2)
            
            total_trip_range_needed = dist_to_target_current + dist_target_to_port
            
            if fisherman_ag.fuel_L * FUEL_EFFICIENCY_KMPL < total_trip_range_needed:
                # OVERRULE: Cannot start the trip. Mandate return/docking at port.
                target_x, target_y = fisherman_ag.home_port_coords
                fisherman_ag.return_status = FUEL_EXHAUSTION_STATUS['FUEL_LOW_ABORT'] 
        
        move_fisherman(fisherman_ag, target_x, target_y)


# --- PLOTTING AND EXPORT FUNCTIONS ---

def plot_habitat_status(time1):
    """Generates a snapshot of the current environment and fish distribution (4-plot grid)."""
    if not CAN_PLOT:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    x_centers = np.linspace(-HALF_LENGTH_AREA, HALF_LENGTH_AREA, ENV_GRID_DIM + 1)
    y_centers = np.linspace(-HALF_LENGTH_AREA, HALF_LENGTH_AREA, ENV_GRID_DIM + 1)
    
    # Helper to draw boundaries
    def draw_boundaries(ax):
        mpa_rect = patches.Rectangle((Xa, Ya), HALF_MPA_LENGTH * 2, HALF_MPA_LENGTH * 2,
                                     edgecolor='darkblue', linewidth=1, facecolor='none', alpha=0.8)
        ax.add_patch(mpa_rect)
        ax.set_xlim(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
        ax.set_ylim(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
        ax.set_aspect('equal', adjustable='box')

    # PLOT A: Dynamic Habitat Quality & Port Pollution
    ax1 = axes[0, 0]
    heatmap = ax1.pcolormesh(x_centers, y_centers, HABITAT_GRID.T, cmap='YlGn', vmin=0.1, vmax=1.0)
    fig.colorbar(heatmap, ax=ax1, orientation='vertical', shrink=0.7, label='Dynamic Habitat Quality (0.1 - 1.0)')

    # Draw All Port Pollution Zones (VISUALIZATION)
    for port in PORT_AGENTS:
        if port.boundary == 'West':
            x_start, y_start, width, height = port.fixed_x, port.span_y[0], port.pollution_depth, port.length_km
        else: # South Port
            x_start, y_start, width, height = port.span_x[0], port.fixed_y, port.length_km, port.pollution_depth
            
        port_rect = patches.Rectangle((x_start, y_start), width, height,
                                      edgecolor='maroon', linewidth=1.0, facecolor='darkred', alpha=0.4)
        ax1.add_patch(port_rect)
        ax1.plot(port.center_coords[0], port.center_coords[1], 's', color='white', markeredgecolor='black', markersize=5)
    
    draw_boundaries(ax1)
    ax1.set_title(f'A. Habitat Quality (Week {time1})', fontsize=14)
    ax1.set_xlabel('X Coordinate (km)'); ax1.set_ylabel('Y Coordinate (km)')

    # PLOT B: Depth / Terrain
    ax2 = axes[0, 1]
    depth_map = ax2.pcolormesh(x_centers, y_centers, DEPTH_GRID.T, cmap='ocean', vmin=2.0, vmax=200.0)
    fig.colorbar(depth_map, ax=ax2, orientation='vertical', shrink=0.7, label='Depth (m)')
    draw_boundaries(ax2)
    ax2.set_title('B. Ocean Depth / Terrain (m)', fontsize=14)
    ax2.set_xlabel('X Coordinate (km)'); ax2.set_ylabel('Y Coordinate (km)')

    # PLOT C: Temperature
    ax3 = axes[1, 0]
    temp_map = ax3.pcolormesh(x_centers, y_centers, TEMP_GRID.T, cmap='Reds_r', vmin=15.0, vmax=35.0)
    fig.colorbar(temp_map, ax=ax3, orientation='vertical', shrink=0.7, label='Temperature (°C)')
    draw_boundaries(ax3)
    ax3.set_title(f'C. Temperature (Week {time1})', fontsize=14)
    ax3.set_xlabel('X Coordinate (km)'); ax3.set_ylabel('Y Coordinate (km)')


    # PLOT D: Fish Agent Distribution (Red dots on white)
    ax4 = axes[1, 1]
    alive_fish = FISH_DATA[FISH_DATA['is_alive']]
    if len(alive_fish) > 0:
        # Sample 5% or 1000 agents for visualization speed
        sample_size = min(len(alive_fish), max(1000, int(len(alive_fish) * 0.05))) 
        sample_indices = np.random.choice(len(alive_fish), size=sample_size, replace=False)
        sampled_fish = alive_fish[sample_indices]
        
        ax4.plot(sampled_fish['x'], sampled_fish['y'], '.', color='red', markersize=1, alpha=0.7)

    draw_boundaries(ax4)
    ax4.set_facecolor('white')
    ax4.set_title(f'D. Fish Agent Distribution (Week {time1})', fontsize=14)
    ax4.set_xlabel('X Coordinate (km)'); ax4.set_ylabel('Y Coordinate (km)')
    
    plt.tight_layout()
    plt.savefig(f'habitat_status_week_{time1:03d}.png')
    plt.close() 

def plot_initial_setup():
    """Generates the initial setup visualization (4-plot grid)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    x_centers = np.linspace(-HALF_LENGTH_AREA, HALF_LENGTH_AREA, ENV_GRID_DIM + 1)
    y_centers = np.linspace(-HALF_LENGTH_AREA, HALF_LENGTH_AREA, ENV_GRID_DIM + 1)
    
    # Helper to draw boundaries
    def draw_boundaries(ax, legend=False):
        mpa_rect = patches.Rectangle((Xa, Ya), HALF_MPA_LENGTH * 2, HALF_MPA_LENGTH * 2,
                                     edgecolor='darkblue', linewidth=1, facecolor='none', alpha=0.8,
                                     label='MPA')
        ax.add_patch(mpa_rect)
        
        # Draw All Ports
        for port in PORT_AGENTS:
            if port.boundary == 'West':
                x_start, y_start, width, height = port.fixed_x, port.span_y[0], port.pollution_depth, port.length_km
            else:
                x_start, y_start, width, height = port.span_x[0], port.fixed_y, port.length_km, port.pollution_depth
                
            port_rect = patches.Rectangle((x_start, y_start), width, height,
                                          edgecolor='maroon', linewidth=1.0, facecolor='darkred', alpha=0.4)
            ax.add_patch(port_rect)
            ax.plot(port.center_coords[0], port.center_coords[1], 's', color='white', markeredgecolor='black', markersize=5, label=f'Port {port.port_id}')
        
        ax.set_xlim(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
        ax.set_ylim(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X Coordinate (km)'); ax.set_ylabel('Y Coordinate (km)')
        if legend:
            # Create a simple legend handle for the port box since we loop it
            port_box_legend = patches.Patch(facecolor='darkred', edgecolor='maroon', alpha=0.4, label='Port Pollution Zone')
            handles, labels = ax.get_legend_handles_labels()
            # Remove duplicate port labels if any
            unique_labels = sorted(list(set([l for l in labels if l.startswith('Port')])))
            
            # Combine generic box and unique port points
            new_handles = [h for h, l in zip(handles, labels) if not l.startswith('Port')] + [port_box_legend] + [h for h, l in zip(handles, labels) if l.startswith('Port')]
            new_labels = [l for l in labels if not l.startswith('Port')] + ['Port Pollution Zone'] + unique_labels
            
            ax.legend(new_handles, new_labels, loc='upper right', prop={'size': 8}, title='Map Features')
        
    # PLOT A: Initial Habitat Quality
    ax1 = axes[0, 0]
    heatmap = ax1.pcolormesh(x_centers, y_centers, HABITAT_GRID.T, cmap='YlGn', vmin=0.1, vmax=1.0)
    fig.colorbar(heatmap, ax=ax1, orientation='vertical', shrink=0.7, label='Initial Habitat Quality (0.1 - 1.0)')
    draw_boundaries(ax1, legend=True)
    ax1.set_title('A. Initial Habitat Quality (Time = 0)', fontsize=14)

    # PLOT B: Ocean Depth / Terrain
    ax2 = axes[0, 1]
    depth_map = ax2.pcolormesh(x_centers, y_centers, DEPTH_GRID.T, cmap='ocean', vmin=2.0, vmax=200.0)
    fig.colorbar(depth_map, ax=ax2, orientation='vertical', shrink=0.7, label='Depth (m)')
    draw_boundaries(ax2)
    ax2.set_title('B. Ocean Depth / Terrain (m)', fontsize=14)
    
    # PLOT C: Initial Temperature (Week 0)
    ax3 = axes[1, 0]
    temp_map = ax3.pcolormesh(x_centers, y_centers, TEMP_GRID.T, cmap='Reds_r', vmin=15.0, vmax=35.0)
    fig.colorbar(temp_map, ax=ax3, orientation='vertical', shrink=0.7, label='Temperature (°C)')
    draw_boundaries(ax3)
    ax3.set_title('C. Initial Temperature (Week 0)', fontsize=14)

    # PLOT D: Initial Fish Spread (Schooled Dispersal)
    ax4 = axes[1, 1]
    all_fish = FISH_DATA[FISH_DATA['is_alive']] 
    
    sample_size = min(len(all_fish), max(1000, int(len(all_fish) * 0.05)))
    sample_indices = np.random.choice(len(all_fish), size=sample_size, replace=False)
    sampled_fish = all_fish[sample_indices]
    
    ax4.plot(sampled_fish['x'], sampled_fish['y'], '.', color='red', markersize=2, alpha=0.5, label='Fish Agents (Sampled)')
    
    draw_boundaries(ax4)
    ax4.set_facecolor('white')
    ax4.set_title(f'D. Initial Fish Spread (Schooled)', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('initial_setup.png')
    plt.close()


def generate_final_plots():
    """Generates the final scatter plot and time series line graphs (decoupled)."""
    if not CAN_PLOT:
        return

    # --- 1. FINAL SPATIAL DISTRIBUTION PLOT ---
    plt.figure(figsize=(8, 8))
    
    alive_fish = FISH_DATA[FISH_DATA['is_alive']]
    if len(alive_fish) > 0:
        sample_size = min(len(alive_fish), int(len(alive_fish) * 0.05)) 
        sample_indices = np.random.choice(len(alive_fish), size=sample_size, replace=False)
        sampled_fish = alive_fish[sample_indices]
        
        plt.plot(sampled_fish['x'], sampled_fish['y'], '.', color='lightgreen', markersize=2, alpha=0.5, label=f'Fish Agents (Sampled: {sample_size})')

    mpa_rect = patches.Rectangle((Xa, Ya), HALF_MPA_LENGTH * 2, HALF_MPA_LENGTH * 2,
                                 edgecolor='darkblue', linewidth=2, facecolor='lightblue', alpha=0.3,
                                 label='MPA')
    plt.gca().add_patch(mpa_rect)
    
    # Plot Last Fished Spots
    fisher_x = [f.last_successful_x for f in FISHER_AGENTS]
    fisher_y = [f.last_successful_y for f in FISHER_AGENTS]
    plt.plot(fisher_x, fisher_y, 'v', color='red', markersize=8, markeredgecolor='k', label='Vessel Last Fished Spot')

    # Draw Port Centers
    for port in PORT_AGENTS:
        plt.plot(port.center_coords[0], port.center_coords[1], 's', color='saddlebrown', markersize=10, label=f'Port {port.port_id}')

    plt.xlim(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
    plt.ylim(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
    plt.title(f'Final Spatial Distribution (Week {LOG_TIME[-1]})', fontsize=14)
    plt.xlabel('X Coordinate (km)'); plt.ylabel('Y Coordinate (km)')
    plt.legend(loc='lower right', prop={'size': 8})
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('final_spatial_distribution.png')
    plt.close()

    # --- 2. DECOUPLED TIME SERIES PLOTS (2x2 Grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    time_steps = np.array(LOG_TIME)
    
    # PLOT 1: Total Fish Agents (Population)
    axes[0, 0].plot(time_steps, LOG_POPULATION, color='darkgreen', label='Total Agents')
    axes[0, 0].set_title('A. Total Fish Agents (Count)')
    axes[0, 0].set_xlabel('Time (Weeks)')
    axes[0, 0].set_ylabel('Population Count')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # PLOT 2: Total Biomass (Sampled)
    biomass_time = time_steps[::BIOMASS_OBSERVE_FREQUENCY]
    biomass_values = LOG_FISH_BIOMASS[::BIOMASS_OBSERVE_FREQUENCY]
    axes[0, 1].plot(biomass_time, biomass_values, color='dodgerblue', marker='.', linestyle='--', label='Total Biomass')
    axes[0, 1].set_title('B. Total Biomass (kg)')
    axes[0, 1].set_xlabel('Time (Weeks)')
    axes[0, 1].set_ylabel('Biomass (kg)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # PLOT 3: Weekly Catch Landed (kg)
    axes[1, 0].plot(time_steps[1:], LOG_WEEKLY_CATCH_KG[1:], color='red', label='Weekly Catch')
    axes[1, 0].set_title('C. Weekly Catch Landed (kg)')
    axes[1, 0].set_xlabel('Time (Weeks)')
    axes[1, 0].set_ylabel('Catch (kg)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # PLOT 4: Weekly Profit (₹)
    weekly_profit = np.diff(LOG_PROFIT)
    axes[1, 1].plot(time_steps[1:], weekly_profit, color='purple', label='Weekly Profit')
    axes[1, 1].axhline(0, color='grey', linestyle=':', linewidth=0.8)
    axes[1, 1].set_title('D. Weekly Fleet Profit (₹)')
    axes[1, 1].set_xlabel('Time (Weeks)')
    axes[1, 1].set_ylabel('Profit (₹)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('final_time_series.png')
    plt.close()


def export_vessel_metrics_to_csv():
    """Writes the detailed per-vessel weekly log to a CSV file."""
    if not LOG_FISHER_WEEKLY:
        print("[WARNING] No vessel-specific data was logged.")
        return

    csv_file = 'vessel_metrics.csv'
    fieldnames = LOG_FISHER_WEEKLY[0].keys()

    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(LOG_FISHER_WEEKLY)
        print(f"[INFO] Successfully exported {len(LOG_FISHER_WEEKLY)} vessel metrics records to {csv_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV file: {e}")

def export_fleet_summary_to_csv():
    """Writes the detailed fleet summary log (population breakdown, catch, profit) to a CSV file."""
    if not LOG_FLEET_SUMMARY:
        print("[WARNING] No fleet summary data was logged.")
        return

    csv_file = 'fleet_summary.csv'
    fieldnames = LOG_FLEET_SUMMARY[0].keys()

    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(LOG_FLEET_SUMMARY)
        print(f"[INFO] Successfully exported {len(LOG_FLEET_SUMMARY)} fleet summary records to {csv_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV file: {e}")


def initialize():
    global FISH_DATA, FISHER_AGENTS, LOG_FISH_BIOMASS
    
    # 0. Initialize Ports first
    initialize_ports()
    
    # 1. Initialize Environmental Layers
    initialize_environmental_layers()
    calculate_primary_production()

    # 2. Initialize Fish
    FISH_DATA = create_fish_agents(INIT_FISH)

    # 3. Initialize Fishers and assign home ports
    FISHER_AGENTS = [FisherAgent(i + 1) for i in range(NUM_FISHERS)]
    
    # Assign each fisherman to a randomly selected home port
    num_ports = len(PORT_AGENTS)
    if num_ports == 0:
        print("[ERROR] Cannot initialize fishermen: No ports were successfully placed. Check port spacing/length constraints.")
        sys.exit(1)
        
    for f in FISHER_AGENTS:
        port = PORT_AGENTS[rd.randint(0, num_ports - 1)]
        f.home_port_coords = port.center_coords
        # Initialize vessel's starting position at its home port
        f.x, f.y = f.home_port_coords 
        f.last_successful_x, f.last_successful_y = f.home_port_coords

    # 4. Initial Logging
    total_biomass = FISH_DATA['biomass_kg'].sum()
    LOG_FISH_BIOMASS.append(total_biomass)
    LOG_POPULATION.append(len(FISH_DATA))
    print(f"INITIAL STATE | Agents: {len(FISH_DATA) + len(FISHER_AGENTS)} | Biomass: {total_biomass:.2f} kg")

    # 5. Initial visualization (Week 0)
    plot_initial_setup()

def update_one_unit_time():
    """Executes all agent updates for one time step (one week)."""

    global FISH_DATA, LOG_FISH_BIOMASS, LOG_TOTAL_HARVEST_KG, LOG_PROFIT, LOG_TIME, LOG_POPULATION, LOG_WEEKLY_CATCH_KG
    time1 = LOG_TIME[-1] + 1
    LOG_TIME.append(time1)
    
    current_week_of_year = time1 % WEEKS_PER_YEAR
    
    # --- STEP 0A: FISHERMAN TRANSACTION & LOGGING (T-1 Results) ---
    total_fleet_profit_realized, current_weekly_catch_log = fisher_transaction_and_reset(time1)
    
    # --- STEP 0B: UPDATE DYNAMIC ENVIRONMENT ---
    update_temperature_grid(time1)
    calculate_primary_production() 
    
    build_density_grid() 
    build_biomass_grid() 
    update_habitat_dynamics() # Includes exponential multi-port degradation
    
    # --- STEP 1: FISH PHYSICS (Movement, Aging, Mortality) ---
    run_fish_physics_numba(
        FISH_DATA, 
        MORTALITY, 
        current_week_of_year, 
        K, 
        DEPTH_GRID, 
        TEMP_GRID, 
        HABITAT_GRID, 
        DENSITY_GRID,
        W_RANDOM, 
        W_DENSITY, 
        W_ENVIRONMENT 
    )
    
    # --- STEP 2: FISHERMAN ACTIONS (Harvesting & Movement for T) ---
    if np.sum(FISH_DATA['is_alive']) > 0:
        build_fisher_search_grid() 
        execute_fishing_and_movement(time1) 

    # --- STEP 3: ANNUAL RECRUITMENT AND ARRAY COMPACTION ---
    new_larvae = []
    if current_week_of_year == RECRUITMENT_WEEK:
        new_larvae = calculate_recruitment(FISH_DATA)
        
    if time1 % WEEKS_PER_YEAR == 0:
        compact_fish_data(new_larvae)
    elif new_larvae:
        new_larvae_array = np.array(new_larvae, dtype=FISH_DTYPE)
        FISH_DATA = np.concatenate((FISH_DATA, new_larvae_array))

    # --- STEP 4: LOGGING & VISUALIZATION ---
    
    alive_fish = FISH_DATA[FISH_DATA['is_alive']]
    current_fish_count = len(alive_fish)
    
    if time1 % BIOMASS_OBSERVE_FREQUENCY == 0:
        current_biomass = alive_fish['biomass_kg'].sum()
    else:
        current_biomass = LOG_FISH_BIOMASS[-1] if len(LOG_FISH_BIOMASS) > 0 else 0.0

    total_cumulative_catch = sum(f.harvest for f in FISHER_AGENTS)
    
    LOG_TOTAL_HARVEST_KG.append(total_cumulative_catch)
    LOG_FISH_BIOMASS.append(current_biomass) 
    LOG_POPULATION.append(current_fish_count)
    
    # Fleet Summary Logging 
    larval_count = np.sum(alive_fish['stage'] == 0)
    juvenile_count = np.sum(alive_fish['stage'] == 1)
    adult_count = np.sum(alive_fish['stage'] == 2)
    
    LOG_FLEET_SUMMARY.append({
        'Week': time1,
        'Total_Fish_Agents': current_fish_count,
        'Larval_Count': larval_count,
        'Juvenile_Count': juvenile_count,
        'Adult_Count': adult_count,
        'Weekly_Catch_KG': current_weekly_catch_log,
        'Weekly_Profit_Rs': total_fleet_profit_realized
    })

    # Generate Habitat Image every 26 weeks
    if time1 % HABITAT_PLOT_FREQUENCY == 0:
        plot_habitat_status(time1)

    # Console logging only
    if time1 % OBSERVE_FREQUENCY == 0:
        current_year = time1 // WEEKS_PER_YEAR
        current_week = time1 % WEEKS_PER_YEAR + 1
        
        print(f"WEEK {time1} ({current_year}-{current_week:02d}) | Total: {current_fish_count} | Biomass: {current_biomass:.2f} kg | Weekly Catch: {current_weekly_catch_log:.2f} kg | Week Profit: ₹{total_fleet_profit_realized:.2f}")

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    if not CAN_PLOT:
        print("\n[CRITICAL WARNING] Matplotlib is required to run this version!")
        sys.exit(1)
        
    start_time = time.time()

    initialize()

    # Run the simulation loop
    for j in range(1, n + 1):
        update_one_unit_time()

    end_time = time.time()
    runtime = end_time - start_time

    # Generate Final Plots 
    generate_final_plots()
    
    # Export all per-vessel metrics and fleet summary
    export_vessel_metrics_to_csv()
    export_fleet_summary_to_csv()

    # E. Print final runtime
    print(f"\n--- Simulation finished. ---")
    print(f"Total simulation steps: {n} weeks ({SIMULATION_YEARS} years)")
    print(f"Final Fish Agents: {len(FISH_DATA[FISH_DATA['is_alive']])}")
    print(f"Total runtime: {runtime:.2f} seconds")
