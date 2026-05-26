# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------
# HIGHLY OPTIMIZED AGENT-BASED FISHERY MODEL (NUMBA / NUMPY)
#
# Optimization: Implemented Von Bertalanffy Growth, Beverton-Holt Recruitment, and FUEL CONSTRAINTS.
# Scaling: Domain scaled to 1000 km x 1000 km. All constants scaled accordingly.
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

# Conditional import for Numba for mandatory performance
try:
    import numba as nb
    from numba import njit
    from numba.typed import List as NumbaList # Import Numba's explicit list type
except ImportError:
    # If Numba is missing, the simulation will be EXTREMELY slow at this scale.
    print("FATAL ERROR: Numba not found. Performance will be severely degraded.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    NumbaList = list
    prange = range 

# Conditional import for Matplotlib for initial plot visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False

# --- GLOBAL CONSTANTS ---

# Simulation Setup (MASSIVE SCALE)
K = 70000000                          # Carrying Capacity (Scaled 100x for new area)
SIMULATION_YEARS = 3
WEEKS_PER_YEAR = 52
n = SIMULATION_YEARS * WEEKS_PER_YEAR # Total simulation steps (weeks)
INIT_FISH = 10000                   # Initial number of fish agents (1 Million)
OBSERVE_FREQUENCY = 5                 # Console log frequency (weeks)
BIOMASS_OBSERVE_FREQUENCY = 12        # Frequency for expensive biomass calculation (weeks)

# Fish Parameters (CATLA ECOLOGICAL MODELS)
GROWTH_PROB = 0.26 # Base growth probability (now only used as a base factor, not per-agent probability)
MOVE_FISH = 450.0                     # Scaled 10x for 1000 km domain
LARVAL_WEEKS = 4
JUVENILE_WEEKS = 52
ADULT_WEEKS_START = LARVAL_WEEKS + JUVENILE_WEEKS
MORTALITY = np.array([0.10, 0.01, 0.005], dtype=np.float64) # [Larval, Juvenile, Adult]
BREEDING_WEEKS_START = 23             # Fishing Ban: Monsoon (June)
BREEDING_WEEKS_END = 35               # Fishing Ban: Monsoon (August)

# *** VON BERTALANFFY GROWTH PARAMETERS (VBGF) ***
L_INF = 1.20                          # Theoretical Max Length (m)
K_GROWTH = 0.40                       # Annual Growth Rate (k, year^-1)
T0 = -0.10                            # Age at Zero Length (years)
A_W = 18.0                            # Weight-Length 'a' parameter 
B_W = 3.0                             # Weight-Length 'b' parameter

# *** BEVERTON-HOLT RECRUITMENT PARAMETERS ***
# R = (alpha * SSB) / (1 + beta * SSB)
# SSB: Spawning Stock Biomass (kg)
ALPHA = 0.05                          # Max survival/recruitment per unit SSB (tuned for scale)
BETA = 1.0e-7                         # Density dependence factor (tuned for scale)
RECRUITMENT_WEEK = BREEDING_WEEKS_END # Week 35 is when larvae are added after spawning season

# Fisher Parameters (INDIAN ARTISANAL VESSEL - SCALED)
NUM_FISHERS = 20
MOVE_FISHERS = 5000.0                 # Scaled 10x Max Weekly Displacement (km) - MAX MECHANICAL SPEED
# --- NEW FUEL CONSTRAINTS (Adjusted for 15-20 km/hr effective speed) ---
FUEL_CAPACITY_LITERS = 600.0          # Adjusted (was 300L) to achieve ~18 km/hr effective speed
FUEL_EFFICIENCY_KMPL = 5.0            # Kilometers per Liter
MAX_FUEL_RANGE_KM = FUEL_CAPACITY_LITERS * FUEL_EFFICIENCY_KMPL # 3000 km total range (approx 17.85 km/hr)
# -----------------------------
Q = 0.7                               # Catchability coefficient
R = 30.0                              # Scaled 10x Fisher neighborhood radius (km)
R_SQR = R ** 2
VESSEL_WEIGHT_CAPACITY = 300.0        # Realistic medium weekly capacity (kg)
MIN_HARVEST_LENGTH_M = 0.50           # Realistic Catch Length (m)
WEEKLY_TRIP_COST = 5000.0             # Increased fixed operational cost (Rupees)
UNIT_PRICE_PER_KG = 150.0             # Average price per kg (Rupees)
EXPLORATION_PROBABILITY = 0.50
PORT_COORDINATES = (-500.0, 0.0)      # Port coordinates scaled 10x

# Spatial Grid / MPA Parameters
AREA = 100000.0                      # New area: 1,000,000 sq km
LENGTH_AREA = 1000.0                  # New linear length: 1000 km
HALF_LENGTH_AREA = LENGTH_AREA / 2    # 500.0 km
GRID_SIZE = 30.0                      # Scaled 10x grid size for sparser large-scale search
GRID_DIM = int(LENGTH_AREA / GRID_SIZE)

# MPA Design 
MPA_ACTIVE = 'yes'
MPA_TYPE = 'single'
MPA_SIZE = 1500.0                     # Scaled 100x area (1500 sq km)
HALF_MPA_LENGTH = (math.sqrt(MPA_SIZE)) / 2 # ~38.7 km
Xa, Xb, Ya, Yb = -HALF_MPA_LENGTH, HALF_MPA_LENGTH, -HALF_MPA_LENGTH, HALF_MPA_LENGTH

# --- NEW BIOME / HABITAT PARAMETERS ---
BIOME_BONUS_FACTOR = 0.30             # 30% spatial bias in recruitment probability
# Fixed centers for Rich Biomes (e.g., warmer, shallower, higher nutrient zones)
RICH_BIOMES = np.array([
    [-150.0, 150.0],  # Biome 1: North-West quadrant
    [200.0, -100.0]    # Biome 2: South-East quadrant
], dtype=np.float64)
BIOME_RADIUS_SQR = 2500.0             # Biome influence radius (50 km radius)
BIAS_FACTOR = 0.10                    # 10% movement bias toward the nearest rich biome

# --- GLOBAL DATA STRUCTURES ---

# Fish Data Dtype includes 'is_alive' flag for efficient deletion handling
FISH_DTYPE = [('x', 'f8'), ('y', 'f8'), ('age_weeks', 'i4'), ('length_m', 'f8'), 
              ('biomass_kg', 'f8'), ('stage', 'i1'), ('is_alive', 'b1')] # b1 is boolean
FISH_DATA = np.empty((0,), dtype=FISH_DTYPE)

# Fisher Data remains Python objects (only 20, so overhead is minimal)
FISHER_AGENTS = []
GRID = {} # Used for quick fisherman search

# Simulation Loggers
LOG_FISH_BIOMASS = []
LOG_HARVEST_KG = [0.0]
LOG_PROFIT = [0.0]
LOG_TIME = [0]


# --- CORE SIMULATION FUNCTIONS (NUMBA ACCELERATED) ---

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
    # L(t) = L_inf * (1 - e^(-K * (t - t0)))
    length_m = L_INF * (1.0 - np.exp(-K_GROWTH * (age_years - T0)))
    
    # 2. Weight-Length Relationship
    biomass_kg = A_W * (length_m ** B_W)
    
    # Determine stage (stage calculation is still based on strict age classes)
    stage = get_stage_index(age_weeks)
    
    return length_m, biomass_kg, stage

# Main function to run Movement, Aging, and Mortality checks (SEQUENTIAL NUMBA)
@njit(cache=True) # Removed parallel=True for stability
def run_fish_physics_numba(fish_array, mortality_rates, current_week_of_year, carrying_capacity, biome_centers, biome_radius_sqr, bias_factor):
    """
    Handles movement, aging, and mortality checks. 
    Returns an empty NumbaList of the expected tuple structure for type stability.
    """
    N = len(fish_array)
    
    # Pre-calculate random movement vector for the week
    rand_theta_array = 2.0 * np.pi * np.random.rand(N) 
    
    for i in range(N):
        if not fish_array['is_alive'][i]:
            continue # Skip dead/caught fish

        # --- 1. BIOME-DRIVEN MOVEMENT (Taxis) ---
        
        theta_rand = rand_theta_array[i]
        
        # Find nearest rich biome center (Taxis target)
        min_dist_sqr = 1e18 
        closest_biome_center_x = 0.0
        closest_biome_center_y = 0.0
        
        for j in range(biome_centers.shape[0]):
            bx = biome_centers[j, 0]
            by = biome_centers[j, 1]
            dist_sqr = (fish_array['x'][i] - bx)**2 + (fish_array['y'][i] - by)**2
            if dist_sqr < min_dist_sqr:
                min_dist_sqr = dist_sqr
                closest_biome_center_x = bx
                closest_biome_center_y = by

        # Calculate preferred move direction (Taxis) towards the closest rich biome
        deltax_preferred = closest_biome_center_x - fish_array['x'][i]
        deltay_preferred = closest_biome_center_y - fish_array['y'][i]
        
        theta_preferred = np.arctan2(deltay_preferred, deltax_preferred)
        
        # Weighted combination of random and preferred directions
        theta = (1.0 - bias_factor) * theta_rand + bias_factor * theta_preferred
        
        # Apply movement
        fish_array['x'][i] += MOVE_FISH * np.cos(theta)
        fish_array['y'][i] += MOVE_FISH * np.sin(theta)

        # Apply Periodic Boundary Conditions (simplified wrap-around)
        if fish_array['x'][i] > HALF_LENGTH_AREA:
            fish_array['x'][i] -= LENGTH_AREA
        elif fish_array['x'][i] < -HALF_LENGTH_AREA:
            fish_array['x'][i] += LENGTH_AREA

        if fish_array['y'][i] > HALF_LENGTH_AREA:
            fish_array['y'][i] -= LENGTH_AREA
        elif fish_array['y'][i] < -HALF_LENGTH_AREA:
            fish_array['y'][i] += LENGTH_AREA

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
        
    # FIX: Explicitly define the type of the empty list returned for Numba stability
    return NumbaList([
        (0.0, 0.0, 0, 0.0, 0.0, 0, True) # Dummy tuple to define type: (f8, f8, i4, f8, f8, i1, b1)
    ])[:0] # Slice to create an empty NumbaList of the correct type

# Function to check if a point is inside the single MPA.
@njit(cache=True)
def is_in_mpa_numba(x, y):
    """Checks if a point is inside the single MPA (Numba compatible)."""
    return (Xa <= x <= Xb) and (Ya <= y <= Yb)

# --- POPULATION RECRUITMENT LOGIC (Replaces Per-Agent Reproduction) ---

def calculate_recruitment(fish_data, biome_centers, biome_radius_sqr):
    """
    Calculates total annual recruitment using the Beverton-Holt equation 
    and distributes new recruits spatially (O(N_adults) + O(N_recruits)).
    """
    # 1. Calculate Spawning Stock Biomass (SSB)
    # Only adult (stage=2) and alive fish contribute
    adult_mask = (fish_data['stage'] == 2) & (fish_data['is_alive'])
    ssb_kg = fish_data['biomass_kg'][adult_mask].sum()

    # 2. Apply Beverton-Holt Model
    # R = (alpha * SSB) / (1 + beta * SSB)
    if ssb_kg < 1.0: # Prevent division by zero/numerical instability with tiny SSB
        total_recruits = 0
    else:
        total_recruits = int(round((ALPHA * ssb_kg) / (1.0 + BETA * ssb_kg)))
    
    if total_recruits <= 0:
        return []

    # 3. Spatial Distribution of Recruits (Biased Recruitment)
    # Determine the number of adults in each biome vs. outside to create spatial probability
    adult_fish = fish_data[adult_mask]
    
    # Numba function to get spatial probabilities
    @njit(cache=True)
    def get_spatial_prob(adults, biomes, radius_sqr, bonus_factor):
        # Calculate base reproductive weight (SSB) inside and outside biomes
        ssb_in_biome = 0.0
        ssb_outside = 0.0
        
        for i in range(len(adults)):
            is_in_rich_biome = False
            
            # Check proximity to any rich biome
            for j in range(biomes.shape[0]):
                dist_sqr = (adults['x'][i] - biomes[j, 0])**2 + (adults['y'][i] - biomes[j, 1])**2
                if dist_sqr < radius_sqr:
                    is_in_rich_biome = True
                    break
            
            if is_in_rich_biome:
                # Add bonus factor to SSB contribution from rich biomes
                ssb_in_biome += adults['biomass_kg'][i] * (1.0 + bonus_factor)
            else:
                ssb_outside += adults['biomass_kg'][i]

        # Calculate probability based on weighted SSB
        total_weighted_ssb = ssb_in_biome + ssb_outside
        
        # If no weighted SSB, distribute uniformly
        if total_weighted_ssb == 0:
            return 0.5, 0.5 # Default to 50/50

        prob_in_biome = ssb_in_biome / total_weighted_ssb
        prob_outside = ssb_outside / total_weighted_ssb
        return prob_in_biome, prob_outside

    prob_in_biome, prob_outside = get_spatial_prob(adult_fish, biome_centers, biome_radius_sqr, BIOME_BONUS_FACTOR)
    
    recruits_in_biome = int(round(total_recruits * prob_in_biome))
    
    new_larvae = []
    
    # 4. Generate Larvae in Rich Biomes
    if recruits_in_biome > 0:
        for _ in range(recruits_in_biome):
            # Choose a random Rich Biome to spawn in
            biome = rd.choice(RICH_BIOMES)
            biome_radius = math.sqrt(BIOME_RADIUS_SQR)
            
            # Spawn near the center of the chosen rich biome
            x_pos = np.random.uniform(biome[0] - biome_radius, biome[0] + biome_radius)
            y_pos = np.random.uniform(biome[1] - biome_radius, biome[1] + biome_radius)
            
            # Append new larva tuple (age 0, larval stage)
            new_larvae.append((x_pos, y_pos, 0, 0.0, 0.0, 0, True))

    # 5. Generate Larvae Outside Biomes (Uniformly Random)
    recruits_outside = total_recruits - recruits_in_biome
    if recruits_outside > 0:
        for _ in range(recruits_outside):
            x_pos = np.random.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
            y_pos = np.random.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
            
            new_larvae.append((x_pos, y_pos, 0, 0.0, 0.0, 0, True))
            
    # Numba requires the output to be a list of tuples with the correct dtype
    return new_larvae


# --- PYTHON/NUMPY OPERATIONAL LOGIC ---
# (Rest of the file remains similar, but the main loop handles recruitment)

def create_fish_agents(num_agents):
    """Initializes the NumPy array with fish data."""
    new_data = np.zeros(num_agents, dtype=FISH_DTYPE)
    
    # Vectorized initial positions and ages
    new_data['x'] = np.random.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA, num_agents)
    new_data['y'] = np.random.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA, num_agents)
    new_data['age_weeks'] = np.random.randint(0, SIMULATION_YEARS * WEEKS_PER_YEAR, num_agents)
    new_data['is_alive'] = True # All start alive

    # Calculate initial length, biomass, and stage using the Numba function
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
        self.harvest = 0.0
        self.weekly_weight_catch = 0.0
        self.weekly_revenue = 0.0
        self.weekly_distance_traveled = 0.0
        self.effort = 0.7
        # Initial position outside MPA (simplified)
        self.x = rd.uniform(Xb, HALF_LENGTH_AREA)
        self.y = rd.uniform(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
        self.last_successful_x = self.x
        self.last_successful_y = self.y
        # NEW: Fuel attribute
        self.fuel_L = FUEL_CAPACITY_LITERS # Start with full tank

def plot_initial_setup():
    """Generates a one-time visualization of the MPA, Biomes, and Domain boundaries."""
    global CAN_PLOT

    if not CAN_PLOT:
        print("\n[INFO] Matplotlib is not available. Skipping initial plot.")
        return

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 1. Draw Simulation Boundary
    ax.set_xlim(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
    ax.set_ylim(-HALF_LENGTH_AREA, HALF_LENGTH_AREA)
    ax.set_aspect('equal', adjustable='box')
    plt.title(f'Initial Simulation Setup ({LENGTH_AREA:.0f}km x {LENGTH_AREA:.0f}km)', fontsize=14)
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    
    # 2. Draw MPA (Centered Rectangle)
    mpa_rect = patches.Rectangle((Xa, Ya), HALF_MPA_LENGTH * 2, HALF_MPA_LENGTH * 2,
                                 edgecolor='darkblue', linewidth=2, facecolor='lightblue', alpha=0.4,
                                 label=f'MPA ({MPA_SIZE:.0f} km²)')
    ax.add_patch(mpa_rect)
    
    # 3. Draw Rich Biomes (Circles)
    biome_radius = math.sqrt(BIOME_RADIUS_SQR)
    for i, center in enumerate(RICH_BIOMES):
        circle = patches.Circle((center[0], center[1]), biome_radius, 
                                edgecolor='green', linewidth=1.5, facecolor='lime', alpha=0.3, 
                                label=f'Rich Biome {i+1}' if i == 0 else "")
        ax.add_patch(circle)
        ax.text(center[0], center[1], f'B{i+1}', ha='center', va='center', color='darkgreen', fontsize=10, fontweight='bold')

    # 4. Draw Port Location
    ax.plot(PORT_COORDINATES[0], PORT_COORDINATES[1], 's', color='saddlebrown', markersize=10, label='Port (Vessel Home)')
    ax.text(PORT_COORDINATES[0], PORT_COORDINATES[1] + 20, 'Port', ha='center', color='saddlebrown', fontsize=10)

    # 5. Add Legend and Show
    ax.legend(loc='upper right', framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def initialize():
    global FISH_DATA, FISHER_AGENTS, LOG_FISH_BIOMASS
    
    # 1. Initialize Fish (NumPy Array)
    FISH_DATA = create_fish_agents(INIT_FISH)

    # 2. Initialize Fishers (Python Objects)
    FISHER_AGENTS = [FisherAgent(i + 1) for i in range(NUM_FISHERS)]

    # 3. Initial Logging
    total_biomass = FISH_DATA['biomass_kg'].sum()
    LOG_FISH_BIOMASS.append(total_biomass)
    print(f"INITIAL STATE | Agents: {len(FISH_DATA) + len(FISHER_AGENTS)} | Biomass: {total_biomass:.2f} kg")

    # 4. ONE-TIME VISUALIZATION
    plot_initial_setup()

def compact_fish_data(new_larvae):
    """
    Removes dead/caught agents and adds new larvae. 
    This expensive operation is now limited to once per year.
    """
    global FISH_DATA
    
    # 1. Filter out dead/caught fish (is_alive == False)
    FISH_DATA = FISH_DATA[FISH_DATA['is_alive']]
    
    # 2. Add new larvae (births)
    if new_larvae:
        # NOTE: new_larvae is already a list of tuples with correct dtype properties
        new_larvae_array = np.array(new_larvae, dtype=FISH_DTYPE)
        FISH_DATA = np.concatenate((FISH_DATA, new_larvae_array))
    

def build_grid_for_fishers():
    """Populates the global GRID dictionary with indices of ALIVE fish for quick access (O(N))."""
    global GRID
    GRID.clear()
    
    # Only iterate over alive fish (more efficient than checking inside the loop if array is large)
    alive_indices = np.where(FISH_DATA['is_alive'])[0]
    
    for i in alive_indices:
        x, y = FISH_DATA['x'][i], FISH_DATA['y'][i]
        
        # Calculate cell coordinates
        cell_x = int((x + HALF_LENGTH_AREA) / GRID_SIZE)
        cell_y = int((y + HALF_LENGTH_AREA) / GRID_SIZE)

        # Clamp to grid bounds
        cell_x = max(0, min(cell_x, GRID_DIM - 1))
        cell_y = max(0, min(cell_y, GRID_DIM - 1))

        cell_key = (cell_x, cell_y)
        if cell_key not in GRID:
            GRID[cell_key] = []
        GRID[cell_key].append(i)

def get_harvestable_fish_indices(fisherman_ag):
    """Finds all harvestable fish indices within range of the fisherman using the grid."""
    focal_cell_x = int((fisherman_ag.x + HALF_LENGTH_AREA) / GRID_SIZE)
    focal_cell_y = int((fisherman_ag.y + HALF_LENGTH_AREA) / GRID_SIZE)

    harvestable_indices = []
    
    # Search the 9 surrounding cells (including the center cell)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell_x, cell_y = focal_cell_x + dx, focal_cell_y + dy
            
            if 0 <= cell_x < GRID_DIM and 0 <= cell_y < GRID_DIM:
                cell_key = (cell_x, cell_y)
                if cell_key in GRID:
                    for fish_index in GRID[cell_key]:
                        # Check if fish is still alive (essential check)
                        if not FISH_DATA['is_alive'][fish_index]:
                            continue 
                        
                        # Quick Distance Check (NumPy data)
                        dist_sqr = (FISH_DATA['x'][fish_index] - fisherman_ag.x)**2 + (FISH_DATA['y'][fish_index] - fisherman_ag.y)**2
                        
                        if dist_sqr < R_SQR:
                            # MPA Check using Numba-compiled function
                            if not is_in_mpa_numba(FISH_DATA['x'][fish_index], FISH_DATA['y'][fish_index]):
                                # Gear Selectivity Check
                                if FISH_DATA['length_m'][fish_index] >= MIN_HARVEST_LENGTH_M:
                                    harvestable_indices.append(fish_index)
                                    
    return harvestable_indices

def move_fisherman(fisherman_ag, target_x, target_y):
    """Calculates and applies movement toward a target, handling budget AND FUEL."""
    
    current_x, current_y = fisherman_ag.x, fisherman_ag.y
    
    # 1. Calculate mechanical distance budget remaining
    remaining_mech_budget = MOVE_FISHERS - fisherman_ag.weekly_distance_traveled

    # 2. Calculate fuel distance budget remaining
    remaining_fuel_L = fisherman_ag.fuel_L
    remaining_fuel_range = remaining_fuel_L * FUEL_EFFICIENCY_KMPL

    if remaining_mech_budget <= 0 or remaining_fuel_range <= 0: return

    deltax = target_x - current_x
    deltay = target_y - current_y
    distance_to_target = math.sqrt(deltax**2 + deltay**2)
    
    # Move distance is the minimum of 3 constraints:
    # 1. Max mechanical distance remaining
    # 2. Distance allowed by remaining fuel
    # 3. Distance to the target
    move_distance = min(remaining_mech_budget, remaining_fuel_range, distance_to_target)

    if move_distance > 0:
        theta = math.atan2(deltay, deltax)
        new_x = current_x + move_distance * math.cos(theta)
        new_y = current_y + move_distance * math.sin(theta)
        
        # Calculate fuel consumption for the move
        fuel_consumed = move_distance / FUEL_EFFICIENCY_KMPL
        
        # Update fuel and distance logs
        fisherman_ag.fuel_L -= fuel_consumed
        fisherman_ag.weekly_distance_traveled += move_distance
    else:
        return

    # Clamping boundaries
    fisherman_ag.x = max(-HALF_LENGTH_AREA, min(HALF_LENGTH_AREA, new_x))
    fisherman_ag.y = max(-HALF_LENGTH_AREA, min(HALF_LENGTH_AREA, new_y))

def update_fishermen(time1):
    """Executes all fisherman movement and harvest logic."""
    global FISH_DATA, FISHER_AGENTS, LOG_HARVEST_KG, LOG_PROFIT
    
    total_fleet_profit_this_week = 0.0
    total_fleet_distance_traveled = 0.0
    
    current_week_of_year = time1 % WEEKS_PER_YEAR

    # 1. Weekly Profit Realization and Reset (runs at the start of the week)
    for f in FISHER_AGENTS:
        current_trip_profit = f.weekly_revenue - WEEKLY_TRIP_COST
        total_fleet_profit_this_week += current_trip_profit
        total_fleet_distance_traveled += f.weekly_distance_traveled
        
        f.weekly_weight_catch = 0.0 
        f.weekly_revenue = 0.0
        f.weekly_distance_traveled = 0.0
        # NEW: Refill tank at the start of the trip
        f.fuel_L = FUEL_CAPACITY_LITERS 

    LOG_PROFIT.append(LOG_PROFIT[-1] + total_fleet_profit_this_week)

    # --- FISHING BAN IMPLEMENTATION ---
    if BREEDING_WEEKS_START <= current_week_of_year <= BREEDING_WEEKS_END:
        # Fishing ban is active: Skip all harvesting and directed movement for the week.
        return
    # -----------------------------------

    # 2. Iterative Fishing/Movement Steps (approximating continuous effort)
    t = 0.
    while t < 1. and len(FISHER_AGENTS) > 0:
        t += 1. / len(FISHER_AGENTS)
        
        fisherman_ag = rd.choice(FISHER_AGENTS)
        
        # HARVESTING
        # Use the grid structure, which only contains indices of ALIVE fish
        fish_indices_in_range = get_harvestable_fish_indices(fisherman_ag)
        
        # Calculate potential harvest based on local fish concentration
        potential_harvest_count = int(round(Q * fisherman_ag.effort * len(fish_indices_in_range)))

        current_harvest_weight = 0.0
        remaining_capacity = VESSEL_WEIGHT_CAPACITY - fisherman_ag.weekly_weight_catch
        indices_to_remove = []
        
        if fish_indices_in_range:
            # Randomly sample fish to be caught (Indices are still valid at this point)
            sample_indices = rd.sample(fish_indices_in_range, min(len(fish_indices_in_range), potential_harvest_count))
            
            for index in sample_indices:
                fish_weight = FISH_DATA['biomass_kg'][index]
                if current_harvest_weight + fish_weight <= remaining_capacity:
                    indices_to_remove.append(index)
                    current_harvest_weight += fish_weight
                    
            # Apply harvest to fisherman metrics
            if indices_to_remove:
                fisherman_ag.last_successful_x = fisherman_ag.x
                fisherman_ag.last_successful_y = fisherman_ag.y
                fisherman_ag.harvest += current_harvest_weight
                fisherman_ag.weekly_weight_catch += current_harvest_weight
                fisherman_ag.weekly_revenue += current_harvest_weight * UNIT_PRICE_PER_KG
                
                # --- ACTUAL DELETION (Set is_alive to False) ---
                for index in indices_to_remove:
                    # Mark fish for removal
                    FISH_DATA['is_alive'][index] = False 
                    
        # MOVEMENT
        if fish_indices_in_range:
            # Exploitation: Move towards Center of Harvestable Biomass (CHB)
            # Find the actual indices that are ALIVE and in range
            valid_indices = [i for i in fish_indices_in_range if FISH_DATA['is_alive'][i]]
            if valid_indices:
                # Use NumPy indexing for vectorized sum
                valid_indices_np = np.array(valid_indices, dtype=np.int32)
                
                weights = FISH_DATA['biomass_kg'][valid_indices_np]
                total_biomass_in_r = weights.sum()
                
                weighted_x = (FISH_DATA['x'][valid_indices_np] * weights).sum()
                weighted_y = (FISH_DATA['y'][valid_indices_np] * weights).sum()

                chb_x = weighted_x / total_biomass_in_r
                chb_y = weighted_y / total_biomass_in_r
                move_fisherman(fisherman_ag, chb_x, chb_y)
            else:
                # If all fish nearby were just caught (is_alive=False), fall back to exploration
                theta = 2*math.pi*rd.random()
                target_x = fisherman_ag.x + MOVE_FISHERS * math.cos(theta) * 0.50 
                target_y = fisherman_ag.y + MOVE_FISHERS * math.sin(theta) * 0.50
                move_fisherman(fisherman_ag, target_x, target_y)
        else:
            # Exploration/Memory
            if fisherman_ag.weekly_weight_catch == 0.0:
                # Forced Exploration (Random walk)
                theta = 2*math.pi*rd.random()
                target_x = fisherman_ag.x + MOVE_FISHERS * math.cos(theta) * 0.75 
                target_y = fisherman_ag.y + MOVE_FISHERS * math.sin(theta) * 0.75
                move_fisherman(fisherman_ag, target_x, target_y)
            elif rd.random() < EXPLORATION_PROBABILITY:
                # Exploration (Random Search)
                theta = 2*math.pi*rd.random()
                target_x = fisherman_ag.x + MOVE_FISHERS * math.cos(theta) * 0.50 
                target_y = fisherman_ag.y + MOVE_FISHERS * math.sin(theta) * 0.50
                move_fisherman(fisherman_ag, target_x, target_y)
            else:
                # Exploitation (Memory: Move back to last successful spot)
                move_fisherman(fisherman_ag, fisherman_ag.last_successful_x, fisherman_ag.last_successful_y)

def update_one_unit_time():
    """Executes all agent updates for one time step (one week)."""

    global FISH_DATA, LOG_FISH_BIOMASS, LOG_HARVEST_KG, LOG_PROFIT, LOG_TIME
    time1 = LOG_TIME[-1] + 1
    LOG_TIME.append(time1)
    
    current_week_of_year = time1 % WEEKS_PER_YEAR
    
    # --- STEP 1: FISH PHYSICS (Movement, Aging, Mortality) ---
    # Note: New larvae list is empty here, as it's generated annually
    new_larvae_dummy = run_fish_physics_numba(
        FISH_DATA, 
        MORTALITY, 
        current_week_of_year, 
        K, 
        RICH_BIOMES,
        BIOME_RADIUS_SQR,
        BIAS_FACTOR
    )
    
    # --- STEP 2: FISHERMAN ACTIONS (Harvesting & Movement) ---
    
    # Build a new spatial index for fishermen using current ALIVE fish positions
    if np.sum(FISH_DATA['is_alive']) > 0:
        build_grid_for_fishers() 
        update_fishermen(time1)

    # --- STEP 3: ANNUAL RECRUITMENT AND ARRAY COMPACTION (The Major Optimization) ---
    new_larvae = []
    if current_week_of_year == RECRUITMENT_WEEK:
        # 3a. Generate recruits using the Beverton-Holt model
        new_larvae = calculate_recruitment(FISH_DATA, RICH_BIOMES, BIOME_RADIUS_SQR)
        
    if time1 % WEEKS_PER_YEAR == 0:
        # 3b. Annual Compaction (remove dead/caught fish, add new larvae)
        compact_fish_data(new_larvae)
    elif new_larvae:
        # 3c. Recruitment happens mid-year (Week 35), so add larvae immediately if not compaction time
        new_larvae_array = np.array(new_larvae, dtype=FISH_DTYPE)
        FISH_DATA = np.concatenate((FISH_DATA, new_larvae_array))

    # --- STEP 4: LOGGING ---
    
    # Calculate current biomass and counts from ALIVE fish
    alive_fish = FISH_DATA[FISH_DATA['is_alive']]
    current_fish_count = len(alive_fish)
    
    # --- PERFORMANCE OPTIMIZATION: Biomass Sum only every 12 weeks ---
    if time1 % BIOMASS_OBSERVE_FREQUENCY == 0:
        current_biomass = alive_fish['biomass_kg'].sum()
    else:
        # Use the last logged biomass value for reporting
        current_biomass = LOG_FISH_BIOMASS[-1] if len(LOG_FISH_BIOMASS) > 0 else 0.0

    total_cumulative_catch = sum(f.harvest for f in FISHER_AGENTS)
    
    LOG_HARVEST_KG.append(total_cumulative_catch)
    LOG_FISH_BIOMASS.append(current_biomass) # Log the calculated or last known value
    
    # Console logging only
    if time1 % OBSERVE_FREQUENCY == 0:
        current_year = time1 // WEEKS_PER_YEAR
        current_week = time1 % WEEKS_PER_YEAR + 1
        total_fleet_profit_this_week = LOG_PROFIT[-1] - LOG_PROFIT[-2] if len(LOG_PROFIT) > 1 else 0.0
        
        # Calculate stage counts from the 'alive_fish' array
        larval_count = np.sum(alive_fish['stage'] == 0)
        juvenile_count = np.sum(alive_fish['stage'] == 1)
        adult_count = np.sum(alive_fish['stage'] == 2)
        
        print(f"WEEK {time1} ({current_year}-{current_week:02d}) | Total: {current_fish_count} (L:{larval_count} J:{juvenile_count} A:{adult_count}) | Biomass: {current_biomass:.2f} kg | Cum. Catch: {total_cumulative_catch:.2f} kg | Week Profit: ₹{total_fleet_profit_this_week:.2f}")

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    start_time = time.time()

    initialize()

    # Run the simulation loop
    for j in range(1, n + 1):
        # The main loop calls update_one_unit_time for every step, which handles all logic now
        update_one_unit_time()


    end_time = time.time()
    runtime = end_time - start_time

    # E. Print final runtime
    print(f"\n--- Simulation finished. ---")
    print(f"Total simulation steps: {n} weeks ({SIMULATION_YEARS} years)")
    print(f"Final Fish Agents: {len(FISH_DATA[FISH_DATA['is_alive']])}")
    print(f"Total runtime: {runtime:.2f} seconds")

#----------------------------------------------------------------------------------------------------------------
