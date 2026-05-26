# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------------

# Agent-Based Model (ABM) representative of an idealised small-scale, artisanal fishery.
# This version features:
# 1. Performance optimization using KD-Tree for fish schooling (Fastest neighbor search).
# 2. Weekly time steps with continuous length/weight growth and strict life stages (Larval, Juvenile, Adult).
# 3. Biomass-based harvesting limits (Vessel Weight Capacity) and Gear Selectivity.
# 4. Habitat Heterogeneity: Fixed resource patches influence initial distribution and fish growth.

# By : OWUSU, Kwabena Afriyie (Optimized and Refactored by Gemini)
# Date : October 2025

#---------------------------------------------------------------------------

## Create a subfolder to save data ##
import shutil, os
from pylab import *
import copy as cp
import random as rd
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from statistics import mean
import time # Import time for runtime calculation

# >>> REQUIRED LIBRARIES <<<
try:
    from scipy.spatial import KDTree # Fastest spatial search
except ImportError:
    print("Warning: SciPy not found. KDTree optimization will fail. Install SciPy for max performance.")
    # Define a dummy class if KDTree is not available
    class KDTree:
        def __init__(self, data): self.data = data
        def query_ball_point(self, x, r): return []

try:
    # Hypothetical Numba import for demonstration
    import numba as nb
    from numba import njit
except ImportError:
    # Define njit as a dummy decorator if numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
# >>> END LIBRARIES <<<

subdir = 'simulation_output' # subfolder name
# Only remove/create directory if not in a restricted environment
try:
    if os.path.isdir(subdir):
        shutil.rmtree(subdir)
    os.mkdir(subdir)
    os.chdir(subdir)
    # Flag to indicate indicate we can save files
    CAN_SAVE_FILES = True
except:
    # If file operations fail (e.g., in a restricted environment), proceed without saving
    CAN_SAVE_FILES = False

#---------------------------------------------------------------------------

# Parameters #

# Fishing ground and time (Simulation runs in WEEKLY time steps) #
K = 35000                          # carrying capacity of fishing ground (agents count) - INCREASED
SIMULATION_YEARS = 3
WEEKS_PER_YEAR = 52
n = SIMULATION_YEARS * WEEKS_PER_YEAR # Total simulation steps (weeks)
init_fish = 1000          # initial number of fish agents - SET BY USER

# NEW: Observation Frequency (For plotting/saving visuals)
OBSERVE_FREQUENCY = 8      # Observe and save plot once every 8 weeks

# Attributes of fish agents #
growth_prob = 0.26      # maximum intrinsic growth rate
# >>> AREA-SCALED SPEED FOR FISH (10000 km^2 domain) <<<
move_fish = 45.0         # speed of fish (Scaled for 10000 sq km domain)
rad_repulsion = 0.025   # radius of repulsion zone
rad_orientation = 0.06  # radius of orientation zone
rad_attraction = 0.1    # radius of attraction zone
rad_repulsion_sqr = rad_repulsion ** 2
rad_orientation_sqr = rad_orientation ** 2
rad_attraction_sqr = rad_attraction ** 2

# NEW: Continuous Growth Parameters (Approximation for fish size)
LARVAL_WEEKS = 4        # ~30 days
JUVENILE_WEEKS = 52     # ~1 year
ADULT_WEEKS_START = LARVAL_WEEKS + JUVENILE_WEEKS # Week 56

L_INF = 0.90             # Theoretical Maximum Length (m)
L_AT_LARVAL_END = 0.10   # Length (m) at the end of larval stage (Week 4)
L_AT_JUVENILE_END = 0.60 # Length (m) at the end of juvenile stage (Week 56)

# >>> FIX APPLIED HERE: Adjusted A_W for realistic biomass (0.01 -> 15.0) <<<
A_W = 15.0               # Weight-Length 'a' parameter (W = a * L^b) - CORRECTED FOR KG/M UNITS
B_W = 3.0                # Weight-Length 'b' parameter

MORTALITY = {
    'larval': 0.10,     # 10% weekly mortality
    'juvenile': 0.01,   # 1% weekly mortality
    'adult': 0.005      # 0.5% weekly mortality
}

# Attributes of fishing agents (pirogues) #
num_fishers = 20       # number of pirogues
# >>> AREA-SCALED SPEED FOR FISHERS (Indian/Practical Conditions - THIS IS THE MAX WEEKLY BUDGET) <<<
move_fishers = 221.3     # MAX WEEKLY displacement budget of a pirogue (km)
q = 0.7                # catchability coefficient
# >>> FIX: INCREASED NEIGHBORHOOD RADIUS FOR BETTER DETECTION IN LARGE/SPARSE AREA <<<
r = 3.0                # neighbourhood radius (for fishing/social interaction)
r_sqr = r ** 2         # neighbourhood radius squared

# NEW: Operational and Ecological Constraints (Weekly Cycle)
BREEDING_WEEKS_START = 10 # Week 10 of the year
BREEDING_WEEKS_END = 16   # Week 14 of the year
VESSEL_WEIGHT_CAPACITY = 500.0 # Max fish weight (kg) a vessel can catch per week
MIN_HARVEST_LENGTH_M = 0.50    # Minimum length (m) for a fish to be caught (Gear Selectivity)

# >>> NEW ECONOMIC AND LOGISTICAL PARAMETERS (Tuned for Indian Scale) <<<
UNIT_PRICE_PER_KG = 150.0    # Average price per kg of caught fish (Generic Currency Unit/kg)
WEEKLY_TRIP_COST = 1500.0    # Fixed operational cost per vessel per week (Generic Currency Unit)
CAPACITY_THRESHOLD_PCT = 0.8 # Percentage of max capacity that triggers return-home logic (Now unused due to forced weekly trip)

# >>> FIX: SET PORT AT LEFT MIDDLE BOUNDARY (-50, 0) <<<
PORT_COORDINATES = (-50.0, 0.0) 
EXPLORATION_PROBABILITY = 0.50 # 50% chance of random move when desperate (USER REQUESTED)
PLOT_SIZE_REDUCTION_FACTOR = 0.4 # Factor to visually shrink agents on the 1000 sq km plot

# Spatial Grid Optimization Parameters
GRID_SIZE = 0.3 # Still used for fishermen only
grid = {}

# MPA Parameters (Unchanged)
# Design of the MPA
MPA = 'yes'          # Presence or absence of MPA ('yes' for presence, 'no' for absence)
Both = 'no'          # Curfew/Part-time MPA: ('no' for full-time presence, 'yes' for part-time presence)
Time_MPA = 50        # Period of time (weeks) over which MPA is active (when Both = 'yes')
Type_MPA = 'single'  # Spacial configuration of MPA ('spaced' for two MPAs, 'single' for one MPA)

# Area and MPA Calculations
# >>> FIX: UPDATED AREA to 10,000 km^2 <<<
Area = 100.0 # 10000 km^2 - 10 x 10 grid
Frac_MPA = 15.0 / Area # 15 km^2 / 10000 km^2 = 0.0015

# Coordinates of the fishing ground
Length_Area = math.sqrt(Area) # 100.0 km
Half_Length_Area = Length_Area / 2 # 50.0 km
GRID_DIM = int(Length_Area / GRID_SIZE) # Number of cells along one side of the grid

# Coordinates of the MPA
Half_Length = (math.sqrt(Frac_MPA * Area)) / 2 # Half length of the MPA side (MPA size remains 15 sq km)

# Coordinates for a single MPA (centered at 0,0)
Xa = -Half_Length
Xb = Half_Length
Ya = -Half_Length
Yb = Half_Length

# Distance between two MPAs
Dist_MPA = 0.2

# Coordinates of first spaced MPA (centered on the left)
Xm = -Half_Length - (Dist_MPA / 2)
Xn = -(Dist_MPA / 2)
Ym = -Half_Length
Yn = Half_Length

# Coordinates of second spaced MPA (centered on the right)
Xp = (Dist_MPA / 2)
Xq = Half_Length + (Dist_MPA / 2)
Yp = -Half_Length
Yq = Half_Length

# >>> NEW: HABITAT HETEROGENEITY DEFINITIONS <<<
RESOURCE_PATCHES = [
    # Patch 1: Center-Top
    {'center_x': 0.0, 'center_y': 30.0, 'radius_sqr': 25.0, 'growth_bonus': 0.10},
    # Patch 2: Bottom-Right
    {'center_x': 30.0, 'center_y': -30.0, 'radius_sqr': 25.0, 'growth_bonus': 0.10}
]

#######################################################################################################################################################

class agent: # create an empty class
    pass

#----------------------------------------------------------------------------------------------------------

# Numba is highly effective here as it's a pure mathematical function
@njit(cache=True)
def calculate_length_and_biomass(age_weeks):
    """Calculates fish length and biomass (kg) based on age using continuous growth."""

    # 1. Determine Length (Simplified age-based growth)
    length_m = 0.0
    stage = ''

    if age_weeks <= LARVAL_WEEKS:
        # Larval phase (Linear growth to L_AT_LARVAL_END)
        length_m = L_AT_LARVAL_END * (age_weeks / LARVAL_WEEKS) if LARVAL_WEEKS > 0 else L_AT_LARVAL_END
        stage = 'larval'
    elif age_weeks <= ADULT_WEEKS_START:
        # Juvenile phase (Linear growth from L_AT_LARVAL_END to L_AT_JUVENILE_END)
        age_in_stage = age_weeks - LARVAL_WEEKS
        stage_duration = JUVENILE_WEEKS
        growth_range = L_AT_JUVENILE_END - L_AT_LARVAL_END
        # Linear interpolation during Juvenile stage
        length_m = L_AT_LARVAL_END + growth_range * (age_in_stage / stage_duration)
        stage = 'juvenile'
    else:
        # Adult phase (Asymptotic growth approximation towards L_INF)
        age_in_adult = age_weeks - ADULT_WEEKS_START
        # Simplified asymptotic growth: L = L_JUVENILE_END + (L_INF - L_JUVENILE_END) * (1 - e^(-k*t))
        # Note: We use a fixed factor for simple progression here.
        length_m = L_AT_JUVENILE_END + (L_INF - L_AT_JUVENILE_END) * min(1.0, age_in_adult / 50.0)
        stage = 'adult'

    # 2. Determine Biomass (Weight-Length relationship: W = a * L^b)
    biomass_kg = A_W * (length_m ** B_W)
    return length_m, biomass_kg, stage

# Global variable to hold the KDTree and the list of fish agents for indexing
fish_agent_list = []
fish_kdtree = None

def initialize():

    global time1, agents, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen_data1, fishermen_data2, fishermen_data3, fishermen_data4, fishermen_data5
    time1 = 0. # time
    agents = [] # list containing fishes and fishermen
    # fish_data will be initialized with total biomass after all agents are created

    total_hav_data = {} # dictionary containing total catch of fishermen
    current_hav_data = {} # dictionary containing current catch of fishermen
    
    # Global Logging Lists
    fishermen_data1 = [0.0] # list containing total cumulative catch weight (kg)
    fishermen_data2 = [0.0] # list containing current weekly catch weight (kg)
    fishermen_data3 = [0.0] # list containing biomass outside MPA
    fishermen_data4 = [0.0] # NEW: Total weekly profit ($)
    fishermen_data5 = [0.0] # NEW: Total weekly distance traveled (km)

#----------------------------------------------------------------------------------------------------------

    # Attributes of agents (fishermen and fish) #
    for j in range(num_fishers + init_fish):
        ag = agent()
        ag.type = 'fishers' if j < num_fishers else 'fish'

        if ag.type == 'fishers':
            ag.harvest = 0.0 # total harvest weight (kg)
            ag.weekly_weight_catch = 0.0 # weekly catch weight (kg)
            ag.weekly_revenue = 0.0 # NEW: Weekly revenue ($)
            ag.weekly_distance_traveled = 0.0 # NEW: Weekly distance (km)
            ag.effort = 0.7 # Single baseline effort for all fishers
            ag.trait = 'fisher' # Simplified trait
            ag.num = 'fisher%d' % (1 + j)
            
            # NEW: Track where the fisher was last successful (initialize to random spot)
            ag.last_successful_x = rd.uniform(-Half_Length_Area, Half_Length_Area) 
            ag.last_successful_y = rd.uniform(-Half_Length_Area, Half_Length_Area)


            total_hav_data[ag.num] = [ag.harvest]
            current_hav_data [ag.num] = [ag.harvest]

            # --- Initial Fisherman Placement (Outside MPA) ---
            if (MPA == 'no' and Both == 'no'): # No MPA at all
                ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
                ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
            # Single MPA cases (Always On or Curfew Active)
            elif any([(MPA == 'yes' and Type_MPA == 'single' and Both == 'no'),(MPA == 'no' and Both == 'yes' and Type_MPA == 'single')]):
                while True: # randomly assign spatial_position to fishermen
                    ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
                    ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
                    if not((Xa <= ag.x <= Xb) and (Ya <= ag.y <= Yb)) : # keep looping until spatial_position falls outside the MPA
                        break
            # Spaced MPA cases (Always On or Curfew Active)
            elif any([(MPA == 'yes' and Type_MPA == 'spaced' and Both == 'no'), (MPA == 'no' and Both == 'yes' and Type_MPA == 'spaced')]):
                while True: # randomly assign spatial_position to fishermen
                    ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
                    ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
                    if all([not((Xm <= ag.x <= Xn) and (Ym <= ag.y <= Yn)),
                            not((Xp <= ag.x <= Xq) and (Yp <= ag.y <= Yq))]): # keep looping until spatial_position falls outside the MPA
                        break
        else: # if a fish
            # >>> HABITAT HETEROGENEITY: INITIAL FISH DISTRIBUTION <<<
            # 30% of fish are placed in patches, 70% placed randomly
            if rd.random() < 0.3:
                patch = rd.choice(RESOURCE_PATCHES)
                # Place fish within the patch radius
                patch_radius = math.sqrt(patch['radius_sqr'])
                ag.x = rd.uniform(patch['center_x'] - patch_radius, patch['center_x'] + patch_radius)
                ag.y = rd.uniform(patch['center_y'] - patch_radius, patch['center_y'] + patch_radius)
            else:
                ag.x = rd.uniform(-Half_Length_Area, Half_Length_Area)
                ag.y = rd.uniform(-Half_Length_Area, Half_Length_Area)
            # --- END HABITAT INITIALIZATION ---

            # --- NEW: Randomly initialize age across all stages ---
            # Max age now set to 3 years (156 weeks) to ensure adequate starting biomass of adult fish.
            max_age_init = SIMULATION_YEARS * WEEKS_PER_YEAR 
            ag.age_weeks = rd.randint(0, max_age_init)

            # Calculate initial properties using the continuous growth model
            ag.length_m, ag.biomass_kg, ag.stage = calculate_length_and_biomass(ag.age_weeks)


        agents.append(ag) # put all agents

#----------------------------------------------------------------------------------------------------------

    # Initialise the number of fishes in an MPA
    def is_in_mpa(x, y):
        if Type_MPA == 'single':
            return (Xa <= x <= Xb) and (Ya <= y <= Yb)
        elif Type_MPA == 'spaced':
            in_mpa1 = (Xm <= x <= Xn) and (Ym <= y <= Yn)
            in_mpa2 = (Xp <= x <= Xq) and (Yp <= y <= Yq)
            return in_mpa1 or in_mpa2
        return False

    def calculate_mpa_biomass(agents_list):
        if (MPA == 'no' and Both == 'no') :
            return 0.0
        # FIX: Use python's sum() on list(generator) to suppress DeprecationWarning
        fish_in_mpa = (j.biomass_kg for j in agents_list if j.type == 'fish' and is_in_mpa(j.x, j.y))
        return sum(list(fish_in_mpa))
        
    fish_data_MPA = [calculate_mpa_biomass(agents)]
    # FIX: Use python's sum() on list(generator) to suppress DeprecationWarning
    total_biomass = sum(list(j.biomass_kg for j in agents if j.type == 'fish'))
    
    # Initialize fish_data with the actual total biomass (This part was conceptually correct)
    fish_data = [total_biomass] 
    
    fishermen_data3[0] = total_biomass - fish_data_MPA[-1] # biomass outside MPA

#----------------------------------------------------------------------------------------------------------
# SPATIAL OPTIMIZATION IMPLEMENTATION (KD-Tree for Fish, Grid for Fishermen)
#----------------------------------------------------------------------------------------------------------

def build_fish_kdtree():
    """Builds the KDTree from the current positions of all fish agents."""
    global fish_agent_list, fish_kdtree
    
    # 1. Separate fish agents from the main list (required for re-indexing)
    fish_agent_list = [ag for ag in agents if ag.type == 'fish']

    # 2. Convert positions to a NumPy array
    fish_positions = np.array([[ag.x, ag.y] for ag in fish_agent_list])

    # 3. Build the KDTree (O(N log N) but highly optimized C code)
    if len(fish_positions) > 0:
        fish_kdtree = KDTree(fish_positions)
    else:
        fish_kdtree = None
        
def get_fish_neighbors_kdtree(focal_ag, radius):
    """
    Finds all fish agents within radius using the KDTree.
    Note: Returns agents, not indices.
    """
    if fish_kdtree is None:
        return []

    # Get the index of the focal agent in the fish_agent_list
    try:
        focal_index = fish_agent_list.index(focal_ag)
    except ValueError:
        # Should not happen if build_fish_kdtree is called first, but handled defensively
        return []

    # Query the tree (O(log N + k))
    # query_ball_point returns the indices in fish_positions that are within radius
    neighbor_indices = fish_kdtree.query_ball_point([focal_ag.x, focal_ag.y], radius)
    
    # Map indices back to agent objects, excluding the focal agent itself
    neighbors = [fish_agent_list[i] for i in neighbor_indices if i != focal_index]
    
    return neighbors


# Fishermen still use the old grid for simplicity, as they are only 20 agents
def build_grid_for_fishers():
    """Populates the global grid dictionary with agents based on their coordinates (Used for Fishermen)."""
    global grid
    grid.clear()
    # Populate the grid with all agents (needed for fisherman neighbor search)
    for ag in agents:
        # Normalize position to [0, Length_Area] and convert to integer cell coordinates
        cell_x = int((ag.x + Half_Length_Area) / GRID_SIZE)
        cell_y = int((ag.y + Half_Length_Area) / GRID_SIZE)

        # Boundary check for agents exactly at the edge
        cell_x = min(cell_x, GRID_DIM - 1)
        cell_y = min(cell_y, GRID_DIM - 1)

        if (cell_x, cell_y) not in grid:
            grid[(cell_x, cell_y)] = []
        grid[(cell_x, cell_y)].append(ag)

def get_fisherman_neighbors(focal_ag):
    """
    Retrieves all agents from adjacent cells for fisherman interaction checks.
    This function still uses the basic spatial grid structure.
    """
    focal_cell_x = int((focal_ag.x + Half_Length_Area) / GRID_SIZE)
    focal_cell_y = int((focal_ag.y + Half_Length_Area) / GRID_SIZE)

    candidates = []
    # Search the 9 surrounding cells (including the center cell)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell_x, cell_y = focal_cell_x + dx, focal_cell_y + dy
            # Simple boundary check for fishermen (no wrap-around needed here)
            if 0 <= cell_x < GRID_DIM and 0 <= cell_y < GRID_DIM and (cell_x, cell_y) in grid:
                candidates.extend(grid[(cell_x, cell_y)])
    return candidates


######################################################################################################################################################

def observe():
    """Plots the current state of the simulation (expensive operation)."""

    global time1, agents, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen , fishermen_data1, fishermen_data2, fishermen_data3

    if not CAN_SAVE_FILES:
        # If saving is disabled, only print the required log to the console
        current_biomass = fish_data[-1]
        total_harvest_kg = fishermen_data1[-1] if len(fishermen_data1) > 0 else 0.0
        print(f"Time: {int(time1)} | Biomass: {current_biomass:.2f} kg | Harvest: {total_harvest_kg:.2f} kg")
        return # Skip plotting if in a restricted environment

    plt.clf() # clear figure
    plt.subplot(111, facecolor='lightskyblue') # background color

    # APPLY VISUAL SCALING FACTOR TO MARKER SIZES
    scale_factor = PLOT_SIZE_REDUCTION_FACTOR
    
    fishermen = [ag for ag in agents if ag.type == 'fishers']
    if len(fishermen) > 0:
        # Simplified plotting: all fishers are now a single type
        X_fishers = [ag.x for ag in fishermen]
        Y_fishers = [ag.y for ag in fishermen]

        # Plot all fishermen with a single color/marker
        plt.plot(X_fishers, Y_fishers, 'o', color='dimgrey', markersize=7.5 * scale_factor, label='Fisher Vessel')

#----------------------------------------------------------------------------------------------------------
    # OPTIMIZED: Re-use the agent list to calculate counts
    fish = [ag for ag in agents if ag.type == 'fish']
    num_fish = len(fish)
    current_biomass = fish_data[-1]

    if num_fish > 0:
        # Separate fish by stage for plotting
        larval_fish = [ag for ag in fish if ag.stage == 'larval']
        juvenile_fish = [ag for ag in fish if ag.stage == 'juvenile']
        adult_fish = [ag for ag in fish if ag.stage == 'adult']

        # Plot Larval (small circle, light color)
        if larval_fish:
            fish_to_plot_larval = rd.sample(larval_fish, min(len(larval_fish), 300))
            plt.plot([ag.x for ag in fish_to_plot_larval], [ag.y for ag in fish_to_plot_larval],
                     'o', color='yellow', markersize=1.5 * scale_factor, alpha=0.7, label=f'Larval ({len(larval_fish)})')

        # Plot Juvenile (medium triangle, medium color)
        if juvenile_fish:
            fish_to_plot_juvenile = rd.sample(juvenile_fish, min(len(juvenile_fish), 400))
            plt.plot([ag.x for ag in fish_to_plot_juvenile], [ag.y for ag in fish_to_plot_juvenile],
                     '^', color='lightgreen', markersize=2.5 * scale_factor, alpha=0.7, label=f'Juvenile ({len(juvenile_fish)})')

        # Plot Adult (larger inverted triangle, dark color)
        if adult_fish:
            fish_to_plot_adult = rd.sample(adult_fish, min(len(adult_fish), 300))
            plt.plot([ag.x for ag in fish_to_plot_adult], [ag.y for ag in fish_to_plot_adult],
                     'v', color='darkgreen', markersize=3.5 * scale_factor, alpha=0.9, label=f'Adult ({len(adult_fish)})')
    
    # Draw MPA boundaries (Curfew logic applied here: time1 <= Time_MPA)
    mpa_active = ((MPA == 'yes' and Both == 'no') or (MPA == 'no' and Both == 'yes' and time1 <= Time_MPA))

    if mpa_active:
        mpa_color = 'k'
        if Type_MPA == 'single':
            # Lines enclosing single MPA
            plt.vlines(Xa, Ya, Yb, lw=2, color=mpa_color)
            plt.vlines(Xb, Ya, Yb, lw=2, color=mpa_color)
            plt.hlines(Ya, Xa, Xb, lw=2, color=mpa_color)
            plt.hlines(Yb, Xa, Xb, lw=2, color=mpa_color)

        elif Type_MPA == 'spaced':
            # Lines enclosing the first spaced MPA
            plt.vlines(Xm, Ym, Yn, lw=2, color=mpa_color)
            plt.vlines(Xn, Ym, Yn, lw=2, color=mpa_color)
            plt.hlines(Ym, Xm, Xn, lw=2, color=mpa_color)
            plt.hlines(Yn, Xm, Xn, lw=2, color=mpa_color)
            # Lines enclosing the second spaced MPA
            plt.vlines(Xp, Yp, Yq, lw=2, color=mpa_color)
            plt.vlines(Xq, Yp, Yq, lw=2, color=mpa_color)
            plt.hlines(Yp, Xp, Xq, lw=2, color=mpa_color)
            plt.hlines(Yq, Xp, Xq, lw=2, color=mpa_color)

    # >>> FIX: Plot the Port Location <<<
    port_x, port_y = PORT_COORDINATES
    plt.plot(port_x, port_y, 's', color='saddlebrown', markersize=15 * scale_factor, label='Port')
    
    # >>> NEW: Plot Resource Patches <<<
    for patch in RESOURCE_PATCHES:
        patch_radius = math.sqrt(patch['radius_sqr'])
        circle = plt.Circle((patch['center_x'], patch['center_y']), patch_radius, 
                            color='sandybrown', alpha=0.3, fill=True, zorder=0)
        plt.gca().add_artist(circle)


    current_year = int(time1) // WEEKS_PER_YEAR
    current_week = int(time1) % WEEKS_PER_YEAR + 1 # +1 to be 1-indexed for display
    total_harvest_kg = fishermen_data1[-1] if len(fishermen_data1) > 0 else 0.0

    plt.title(f'Year: {current_year}, Week: {current_week} | Biomass: {current_biomass:.2f} kg | Harvest: {total_harvest_kg:.2f} kg')
    plt.legend(numpoints=1, loc= 'center', bbox_to_anchor=(0.5, -0.072), ncol=3, prop={'size':11}, facecolor='lightskyblue')

    axis('image') ; axis([-Half_Length_Area, Half_Length_Area,-Half_Length_Area, Half_Length_Area]) ; plt.grid(False) ; plt.xticks([], []) ; plt.yticks([], [])
    plt.savefig('year_%04d.png' % time1, bbox_inches='tight', pad_inches=0 ,dpi=200)

######################################################################################################################################################

def update_fish():
    """Updates the position and state of a randomly selected fish agent using the KDTree for neighbor search."""

    global time1, agents
    
    if not agents: return
    
    ag = rd.choice(agents)
    if ag.type != 'fish':
        return # Skip if it's a fisherman

    fish_ag = ag
    # FIX: Use sum(list(generator)) to suppress DeprecationWarning
    current_fish_count = sum(list(1 for j in agents if j.type == 'fish')) 
    
    # >>> KD-TREE QUERY: Find all neighbors within the maximum attraction radius <<<
    # Note: KDTree returns agents, not just positions.
    all_neighbors = get_fish_neighbors_kdtree(fish_ag, rad_attraction) 

    # Filter neighbors into Boids model zones (Repulsion, Alignment, Attraction)
    repulsion = [nb for nb in all_neighbors if ((fish_ag.x - nb.x)**2 + (fish_ag.y - nb.y)**2) < rad_repulsion_sqr]
    alignment = [nb for nb in all_neighbors if rad_repulsion_sqr < ((fish_ag.x - nb.x)**2 + (fish_ag.y - nb.y)**2) < rad_orientation_sqr ]
    attraction =[nb for nb in all_neighbors if rad_orientation_sqr < ((fish_ag.x - nb.x)**2 + (fish_ag.y - nb.y)**2) < rad_attraction_sqr ]

    # Boids Movement Logic (remains unchanged)
    theta = 2*math.pi*rd.random() # Default random move

    if len(repulsion) > 0:
        repulsion_x = mean([j.x for j in repulsion])
        repulsion_y = mean([j.y for j in repulsion])
        repulsion_1 = (math.atan2((repulsion_y - fish_ag.y), (repulsion_x - fish_ag.x)) + math.pi ) % (2 * math.pi)
        theta = repulsion_1
    elif len(alignment) > 0:
        alignment_1 = mean([math.atan2((j.y - fish_ag.y),(j.x - fish_ag.x)) for j in alignment])
        theta = alignment_1
    elif len(attraction) > 0:
        attraction_x = mean([j.x for j in attraction ])
        attraction_y = mean([j.y for j in attraction])
        attraction_1 = math.atan2((attraction_y - fish_ag.y), (attraction_x - fish_ag.x))
        theta = attraction_1

    # Apply movement and wrap-around (periodic boundary conditions)
    fish_ag.x += move_fish*math.cos(theta)
    fish_ag.y += move_fish*math.sin(theta)

    fish_ag.x = (fish_ag.x % -Half_Length_Area) if fish_ag.x > Half_Length_Area else (fish_ag.x % Half_Length_Area) if fish_ag.x < -Half_Length_Area else fish_ag.x
    fish_ag.y = (fish_ag.y % -Half_Length_Area) if fish_ag.y > Half_Length_Area else (fish_ag.y % Half_Length_Area) if fish_ag.y < -Half_Length_Area else fish_ag.y

    # NEW: Life Stage Updates, Mortality, and Reproduction
    current_week_of_year = int(time1) % WEEKS_PER_YEAR

    # 1. Mortality Check
    if rd.random() < MORTALITY[fish_ag.stage]:
        agents.remove(fish_ag)
        return

    # 2. Aging and Stage Transition (Strictly Time-Based)
    fish_ag.age_weeks += 1

    # Update length, weight, and stage based strictly on the new age
    fish_ag.length_m, fish_ag.biomass_kg, fish_ag.stage = calculate_length_and_biomass(fish_ag.age_weeks)

    # 3. Reproduction (Adults only, during breeding weeks)
    is_breeding_week = BREEDING_WEEKS_START <= current_week_of_year <= BREEDING_WEEKS_END
    
    # >>> HABITAT HETEROGENEITY: GROWTH BONUS <<<
    current_growth_prob = growth_prob
    for patch in RESOURCE_PATCHES:
        dist_sqr = (fish_ag.x - patch['center_x'])**2 + (fish_ag.y - patch['center_y'])**2
        if dist_sqr < patch['radius_sqr']:
            current_growth_prob += patch['growth_bonus']
            break
    
    if fish_ag.stage == 'adult' and is_breeding_week:
        # Use adjusted growth probability
        if rd.random() < current_growth_prob * (1 - current_fish_count / float(K)):
            new_larva = cp.copy(fish_ag)
            new_larva.age_weeks = 0 # Newborn is age 0
            # Calculate initial properties for the newborn (age 0)
            new_larva.length_m, new_larva.biomass_kg, new_larva.stage = calculate_length_and_biomass(0)
            # Place larva randomly near parent or current position
            new_larva.x += rd.uniform(-0.1, 0.1)
            new_larva.y += rd.uniform(-0.1, 0.1)
            # Ensure new position wraps if needed
            new_larva.x = (new_larva.x % -Half_Length_Area) if new_larva.x > Half_Length_Area else (new_larva.x % Half_Length_Area) if new_larva.x < -Half_Length_Area else new_larva.x
            new_larva.y = (new_larva.y % -Half_Length_Area) if new_larva.y > Half_Length_Area else (new_larva.y % Half_Length_Area) if new_larva.y < -Half_Length_Area else new_larva.y
            agents.append(new_larva)

######################################################################################################################################################

def move_fisherman(fisherman_ag, target_x, target_y):
    """
    Calculates and applies movement toward a target, handling boundaries AND tracking distance,
    while enforcing the strict WEEKLY movement budget.
    """
    
    current_x = fisherman_ag.x
    current_y = fisherman_ag.y
    
    # Check remaining budget
    remaining_budget = move_fishers - fisherman_ag.weekly_distance_traveled

    if remaining_budget <= 0:
        return # No movement allowed this step

    deltax = target_x - current_x
    deltay = target_y - current_y
    
    distance_to_target = math.sqrt(deltax**2 + deltay**2)
    
    # Move distance is the minimum of (remaining budget, distance to target)
    move_distance = min(remaining_budget, distance_to_target)

    # Update position
    if distance_to_target > 0 and move_distance > 0:
        theta = math.atan2(deltay, deltax)
        new_x = current_x + move_distance * math.cos(theta)
        new_y = current_y + move_distance * math.sin(theta)
    else:
        # If already at target or no budget, no change
        return

    # Clamping boundaries
    fisherman_ag.x = max(-Half_Length_Area, min(Half_Length_Area, new_x))
    fisherman_ag.y = max(-Half_Length_Area, min(Half_Length_Area, new_y))
    
    # Log distance traveled (only if movement occurred)
    fisherman_ag.weekly_distance_traveled += move_distance

def no_mpa():
    """
    Fisherman logic for no MPA scenario, implementing Exploration/Exploitation.
    """

    fisherman_list = [j for j in agents if j.type == 'fishers']
    if not fisherman_list: return

    fisherman_ag = rd.sample(fisherman_list, 1)[-1]

    # --- 1. Fishing Action (Harvest and Revenue) ---
    candidates = get_fisherman_neighbors(fisherman_ag)

    # Use r_sqr for the search radius
    harvestable_neighbors = [nb for nb in candidates if nb.type == 'fish' and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr
        and nb.length_m >= MIN_HARVEST_LENGTH_M]

    potential_harvest_count = int(round(q * fisherman_ag.effort * len(harvestable_neighbors)))

    current_harvest_weight = 0.0
    remaining_weight_capacity = VESSEL_WEIGHT_CAPACITY - fisherman_ag.weekly_weight_catch
    fish_to_remove = []

    rd.shuffle(harvestable_neighbors) 

    fish_caught_in_step = 0.0
    for j in harvestable_neighbors:
        if len(fish_to_remove) >= potential_harvest_count: 
            break
        if current_harvest_weight + j.biomass_kg <= remaining_weight_capacity:
            fish_to_remove.append(j)
            current_harvest_weight += j.biomass_kg
            fish_caught_in_step += j.biomass_kg

    if fish_caught_in_step > 0:
        fisherman_ag.last_successful_x = fisherman_ag.x # Update last successful spot (Exploitation Memory)
        fisherman_ag.last_successful_y = fisherman_ag.y
        fisherman_ag.weekly_revenue += fish_caught_in_step * UNIT_PRICE_PER_KG
        
    for j in fish_to_remove:
        if j in agents: # Check if the fish has already been removed by another fisherman
            agents.remove(j)
            fisherman_ag.harvest += j.biomass_kg # Total harvest is in kg
            fisherman_ag.weekly_weight_catch += j.biomass_kg # Weekly catch is in kg
            

    # --- 2. Movement Logic (Exploration vs. Exploitation Tradeoff) ---
    
    if len(harvestable_neighbors) > 0:
        # EXPLOITATION (LOCAL): Move towards Center of Harvestable Biomass (CHB)
        total_biomass_in_r = sum(f.biomass_kg for f in harvestable_neighbors)
        weighted_x = sum(f.x * f.biomass_kg for f in harvestable_neighbors)
        weighted_y = sum(f.y * f.biomass_kg for f in harvestable_neighbors)

        chb_x = weighted_x / total_biomass_in_r
        chb_y = weighted_y / total_biomass_in_r
        
        move_fisherman(fisherman_ag, chb_x, chb_y)
    
    else:
        # NO LOCAL FISH FOUND: Decide between memory-based search or exploration
        
        if fisherman_ag.weekly_weight_catch == 0.0:
            # FIX: Forced Exploration (Random walk to find new patches when desperate)
            theta = 2*math.pi*rd.random()
            new_x = fisherman_ag.x + move_fishers * math.cos(theta) * 0.75 # Use 75% of remaining budget capacity for big random moves
            new_y = fisherman_ag.y + move_fishers * math.sin(theta) * 0.75
            move_fisherman(fisherman_ag, new_x, new_y) 
            
        elif rd.random() < EXPLORATION_PROBABILITY:
             # Exploration (Random Search when not desperate)
            theta = 2*math.pi*rd.random()
            new_x = fisherman_ag.x + move_fishers * math.cos(theta) * 0.50 # Use 50% of remaining budget capacity
            new_y = fisherman_ag.y + move_fishers * math.sin(theta) * 0.50
            move_fisherman(fisherman_ag, new_x, new_y)
            
        else:
            # EXPLOITATION (MEMORY): Move back to the last known successful spot
            move_fisherman(fisherman_ag, fisherman_ag.last_successful_x, fisherman_ag.last_successful_y)


######################################################################################################################################################

def single_mpa():
    """
    Fisherman logic for single MPA scenario, implementing Exploration/Exploitation and MPA avoidance.
    """

    fisherman_list = [j for j in agents if j.type == 'fishers']
    if not fisherman_list: return

    fisherman_ag = rd.sample(fisherman_list, 1)[-1]
    
    # MPA check function
    def is_in_single_mpa(x, y):
        return (Xa <= x <= Xb) and (Ya <= y <= Yb)

    # --- 1. Fishing Action ---
    candidates = get_fisherman_neighbors(fisherman_ag)

    harvestable_neighbors = [nb for nb in candidates if nb.type == 'fish' and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr
        and not is_in_single_mpa(nb.x, nb.y) and nb.length_m >= MIN_HARVEST_LENGTH_M]

    potential_harvest_count = int(round(q * fisherman_ag.effort * len(harvestable_neighbors)))

    current_harvest_weight = 0.0
    remaining_weight_capacity = VESSEL_WEIGHT_CAPACITY - fisherman_ag.weekly_weight_catch
    fish_to_remove = []

    rd.shuffle(harvestable_neighbors) 
    
    fish_caught_in_step = 0.0
    for j in harvestable_neighbors:
        if len(fish_to_remove) >= potential_harvest_count:
            break
        if current_harvest_weight + j.biomass_kg <= remaining_weight_capacity:
            fish_to_remove.append(j)
            current_harvest_weight += j.biomass_kg
            fish_caught_in_step += j.biomass_kg
            
    if fish_caught_in_step > 0:
        fisherman_ag.last_successful_x = fisherman_ag.x # Update last successful spot
        fisherman_ag.last_successful_y = fisherman_ag.y
        fisherman_ag.weekly_revenue += fish_caught_in_step * UNIT_PRICE_PER_KG

    for j in fish_to_remove:
        if j in agents:
            agents.remove(j)
            fisherman_ag.harvest += j.biomass_kg
            fisherman_ag.weekly_weight_catch += j.biomass_kg

    # --- 2. Movement Logic (Exploration vs. Exploitation Tradeoff) ---
    
    target_x, target_y = fisherman_ag.x, fisherman_ag.y # Default target is self

    if len(harvestable_neighbors) > 0:
        # EXPLOITATION (LOCAL): Move towards Center of Harvestable Biomass (CHB)
        total_biomass_in_r = sum(f.biomass_kg for f in harvestable_neighbors)
        weighted_x = sum(f.x * f.biomass_kg for f in harvestable_neighbors)
        weighted_y = sum(f.y * f.biomass_kg for f in harvestable_neighbors)
        target_x, target_y = weighted_x / total_biomass_in_r, weighted_y / total_biomass_in_r
        
    else:
        # NO LOCAL FISH FOUND: Decide between memory-based search or exploration
        if fisherman_ag.weekly_weight_catch == 0.0:
            # FIX: Forced Exploration (Random walk to find new patches when desperate)
            theta = 2*math.pi*rd.random()
            target_x = fisherman_ag.x + move_fishers * math.cos(theta) * 0.75 
            target_y = fisherman_ag.y + move_fishers * math.sin(theta) * 0.75
        
        elif rd.random() < EXPLORATION_PROBABILITY:
            # Exploration (Random Search when not desperate)
            theta = 2*math.pi*rd.random()
            target_x = fisherman_ag.x + move_fishers * math.cos(theta) * 0.50 
            target_y = fisherman_ag.y + move_fishers * math.sin(theta) * 0.50
        else:
            # EXPLOITATION (MEMORY): Move back to the last known successful spot
            target_x, target_y = fisherman_ag.last_successful_x, fisherman_ag.last_successful_y

    
    # Calculate movement angle
    deltax = target_x - fisherman_ag.x
    deltay = target_y - fisherman_ag.y
    theta = math.atan2(deltay, deltax)
    
    # Calculate move distance (full speed toward the target)
    move_distance_full = move_fishers
    
    # Proposed new position
    new_x = fisherman_ag.x + move_distance_full * math.cos(theta)
    new_y = fisherman_ag.y + move_distance_full * math.sin(theta)

    # Check and avoid MPA
    if is_in_single_mpa(new_x, new_y):
        # If the move lands in the MPA, use a random evasive move instead
        while True:
            theta_rand = 2*math.pi*rd.random()
            new_x = fisherman_ag.x + move_distance_full * math.cos(theta_rand)
            new_y = fisherman_ag.y + move_distance_full * math.sin(theta_rand)

            if not is_in_single_mpa(new_x, new_y):
                # Apply move through move_fisherman to enforce weekly budget
                move_fisherman(fisherman_ag, new_x, new_y)
                break
    else:
        # If the move is safe, apply it and log distance
        move_fisherman(fisherman_ag, new_x, new_y)


######################################################################################################################################################

def spaced_mpa():
    """
    Fisherman logic for spaced MPA scenario, implementing Exploration/Exploitation and MPA avoidance.
    """

    fisherman_list = [j for j in agents if j.type == 'fishers']
    if not fisherman_list: return

    fisherman_ag = rd.sample(fisherman_list, 1)[-1]

    # Function to check if a position is inside any spaced MPA
    def is_in_spaced_mpa(x, y):
        in_mpa1 = (Xm <= x <= Xn) and (Ym <= y <= Yn)
        in_mpa2 = (Xp <= x <= Xq) and (Yp <= y <= Yq)
        return in_mpa1 or in_mpa2

    # --- 1. Fishing Action ---
    candidates = get_fisherman_neighbors(fisherman_ag)

    harvestable_neighbors = [nb for nb in candidates if nb.type == 'fish' and ((fisherman_ag.x - nb.x)**2 + (fisherman_ag.y - nb.y)**2) < r_sqr
        and not is_in_spaced_mpa(nb.x, nb.y) and nb.length_m >= MIN_HARVEST_LENGTH_M]

    potential_harvest_count = int(round(q * fisherman_ag.effort * len(harvestable_neighbors)))

    current_harvest_weight = 0.0
    remaining_weight_capacity = VESSEL_WEIGHT_CAPACITY - fisherman_ag.weekly_weight_catch
    fish_to_remove = []

    rd.shuffle(harvestable_neighbors) 
    
    fish_caught_in_step = 0.0
    for j in harvestable_neighbors:
        if len(fish_to_remove) >= potential_harvest_count:
            break
        if current_harvest_weight + j.biomass_kg <= remaining_weight_capacity:
            fish_to_remove.append(j)
            current_harvest_weight += j.biomass_kg
            fish_caught_in_step += j.biomass_kg
            
    if fish_caught_in_step > 0:
        fisherman_ag.last_successful_x = fisherman_ag.x # Update last successful spot
        fisherman_ag.last_successful_y = fisherman_ag.y
        fisherman_ag.weekly_revenue += fish_caught_in_step * UNIT_PRICE_PER_KG

    for j in fish_to_remove:
        if j in agents:
            agents.remove(j)
            fisherman_ag.harvest += j.biomass_kg
            fisherman_ag.weekly_weight_catch += j.biomass_kg

    # --- 2. Movement Logic (Exploration vs. Exploitation Tradeoff) ---
    
    target_x, target_y = fisherman_ag.x, fisherman_ag.y # Default target is self

    if len(harvestable_neighbors) > 0:
        # EXPLOITATION (LOCAL): Move towards Center of Harvestable Biomass (CHB)
        total_biomass_in_r = sum(f.biomass_kg for f in harvestable_neighbors)
        weighted_x = sum(f.x * f.biomass_kg for f in harvestable_neighbors)
        weighted_y = sum(f.y * f.biomass_kg for f in harvestable_neighbors)
        target_x, target_y = weighted_x / total_biomass_in_r, weighted_y / total_biomass_in_r
    else:
        # NO LOCAL FISH FOUND: Decide between memory-based search or exploration
        if fisherman_ag.weekly_weight_catch == 0.0:
            # FIX: Forced Exploration (Random walk to find new patches when desperate)
            theta = 2*math.pi*rd.random()
            target_x = fisherman_ag.x + move_fishers * math.cos(theta) * 0.75 
            target_y = fisherman_ag.y + move_fishers * math.sin(theta) * 0.75
        
        elif rd.random() < EXPLORATION_PROBABILITY:
            # Exploration (Random Search when not desperate)
            theta = 2*math.pi*rd.random()
            target_x = fisherman_ag.x + move_fishers * math.cos(theta) * 0.50 
            target_y = fisherman_ag.y + move_fishers * math.sin(theta) * 0.50
        else:
            # EXPLOITATION (MEMORY): Move back to the last known successful spot
            target_x, target_y = fisherman_ag.last_successful_x, fisherman_ag.last_successful_y

    
    # Calculate desired move angle
    deltax = target_x - fisherman_ag.x
    deltay = target_y - fisherman_ag.y
    theta = math.atan2(deltay, deltax)
    
    # Calculate move distance (full speed toward the target)
    move_distance_full = move_fishers
    
    # Proposed new position
    new_x = fisherman_ag.x + move_distance_full * math.cos(theta)
    new_y = fisherman_ag.y + move_distance_full * math.sin(theta)

    # Check and avoid MPA
    if is_in_spaced_mpa(new_x, new_y):
        # If the move lands in the MPA, use a random evasive move instead
        while True:
            theta_rand = 2*math.pi*rd.random()
            new_x = fisherman_ag.x + move_distance_full * math.cos(theta_rand)
            new_y = fisherman_ag.y + move_distance_full * math.sin(theta_rand)

            if not is_in_spaced_mpa(new_x, new_y):
                # Apply move through move_fisherman to enforce weekly budget
                move_fisherman(fisherman_ag, new_x, new_y)
                break
    else:
        # If the move is safe, apply it and log distance
        move_fisherman(fisherman_ag, new_x, new_y)


######################################################################################################################################################

def update_one_unit_time():
    """Executes all agent updates for one time step (one week)."""

    global time1, agents, fish_data, fish_data_MPA, total_hav_data, current_hav_data, fishermen, fishermen_data1, fishermen_data2, fishermen_data3, fishermen_data4, fishermen_data5
    time1 += 1 # update time (this is now a weekly step)

    # STEP 1: Build KDTree for fish and Grid for fishermen
    build_fish_kdtree()
    build_grid_for_fishers()

    # STEP 2: Update Fish Agents (Boids model, aging, mortality, reproduction)
    fish = [j for j in agents if j.type == 'fish'] 
    if fish:
        t = 0.
        # Run sub-steps based on current total number of agents for stability
        while t < 1. and len(agents) > 0: 
            t += 1. / len(agents)
            update_fish()

    # STEP 3: Update Fisherman Agents (Fishing and Movement)
    fishermen = [j for j in agents if j.type == 'fishers']

    # --- NEW: Mandatory Weekly Profit Realization and Reset ---
    total_fleet_profit_this_week = 0.0
    total_fleet_distance_this_week = 0.0
    
    # FIX: Loop through fishers to realize profit, reset state, and track distance
    for f in fishermen:
        # 1. Profit Realization (Mandatory Weekly Trip)
        current_trip_profit = f.weekly_revenue - WEEKLY_TRIP_COST
        total_fleet_profit_this_week += current_trip_profit
        
        # 2. Distance Logging
        total_fleet_distance_this_week += f.weekly_distance_traveled
        
        # 3. Reset State for New Week
        f.weekly_weight_catch = 0.0 
        f.weekly_revenue = 0.0
        f.weekly_distance_traveled = 0.0 # Reset distance for the new week
    # -----------------------------

    if fishermen:
        t = 0.
        # Determine the current MPA regime
        mpa_active = ((MPA == 'yes' and Both == 'no') or (MPA == 'no' and Both == 'yes' and time1 <= Time_MPA))
        no_mpa_regime = (MPA == 'no' and Both == 'no') or (MPA == 'no' and Both == 'yes' and time1 > Time_MPA)

        # Run sub-steps based on number of fisherman agents
        while t < 1.:
            t += 1. / len(fishermen)
            if no_mpa_regime:
                no_mpa()
            elif mpa_active and Type_MPA == 'single':
                single_mpa()
            elif mpa_active and Type_MPA == 'spaced':
                spaced_mpa()
            else:
                no_mpa()


    # STEP 4: Data Preparation and Saving (LOGGING)

    # Recalculate total biomass after all updates
    # FIX: Use python's sum() on list(generator) to suppress DeprecationWarning
    total_biomass = sum(list(j.biomass_kg for j in agents if j.type == 'fish'))
    fish_data.append(total_biomass)

    # A. Calculate Biomass in MPA for the current time step
    current_mpa_biomass = 0.0
    mpa_regime_active = ((MPA == 'yes' and Both == 'no') or (MPA == 'no' and Both == 'yes' and time1 <= Time_MPA))

    if mpa_regime_active:
        if Type_MPA == 'single':
            # FIX: Use python's sum() on list(generator) to suppress DeprecationWarning
            current_mpa_biomass = sum(list(j.biomass_kg for j in agents if j.type == 'fish' and ((Xa <= j.x <= Xb) and (Ya <= j.y <= Yb))))
        elif Type_MPA == 'spaced':
            def is_in_spaced_mpa(x, y):
                in_mpa1 = (Xm <= x <= Xn) and (Ym <= y <= Yn)
                in_mpa2 = (Xp <= x <= Xq) and (Yp <= y <= Yq)
                return in_mpa1 or in_mpa2
            # FIX: Use python's sum() on list(generator) to suppress DeprecationWarning
            current_mpa_biomass = sum(list(j.biomass_kg for j in agents if j.type == 'fish' and is_in_spaced_mpa(j.x, j.y)))

    fish_data_MPA.append(current_mpa_biomass)
    fish_data_OUTSIDE_MPA = total_biomass - current_mpa_biomass
    fishermen_data3.append(fish_data_OUTSIDE_MPA) # biomass outside MPA

    # B. Calculate Harvest Data
    all_fishers = [j for j in agents if j.type =='fishers'] # Re-filter fishers after updates
    current_step_catch = 0.0
    for j in all_fishers:
        # Append each fisherman's total catch (weight)
        total_hav_data[j.num].append(j.harvest)
        # Calculate current catch (weight) and append to current_hav_data
        catch_now = total_hav_data[j.num][-1] - total_hav_data[j.num][-2]
        current_hav_data[j.num].append(catch_now)
        current_step_catch += catch_now

    # FIX: Use python's sum() on list(generator) to suppress DeprecationWarning
    fishermen_data1.append(sum(list(j.harvest for j in agents if j.type == 'fishers'))) # total catch weight
    fishermen_data2.append(current_step_catch) # current catch weight
    
    # NEW LOGGING: Log total profit and distance
    fishermen_data4.append(total_fleet_profit_this_week)
    fishermen_data5.append(total_fleet_distance_this_week)


    # C. Write to CSV (Data logging runs every week)
    if CAN_SAVE_FILES:
        csvfile = "simulation_data.csv"
        header = [key for key in sorted(current_hav_data)]
        header.append('total_catch_kg') 
        header.append('total_biomass_kg') 
        header.append('biomass_inside_MPA_kg') 
        header.append('biomass_outside_MPA_kg')
        # NEW CSV HEADERS
        header.append('total_weekly_profit_$') 
        header.append('total_weekly_distance_km')
        
        main_data = [current_hav_data[key] for key in sorted(current_hav_data)]
        main_data.append(fishermen_data2) 
        main_data.append(fish_data) 
        main_data.append(fish_data_MPA) 
        main_data.append(fishermen_data3)
        # NEW CSV DATA
        main_data.append(fishermen_data4) 
        main_data.append(fishermen_data5)
        
        with open(csvfile, "w") as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(zip(*main_data))


    # D. Print console log every week (optional logging, always runs)
    current_year = int(time1) // WEEKS_PER_YEAR
    current_week = int(time1) % WEEKS_PER_YEAR + 1
    # FIX: Use python's sum() on list(generator) to suppress DeprecationWarning
    current_fish_count_for_print = sum(list(1 for j in agents if j.type == 'fish'))
    print(f"WEEK {int(time1)} ({current_year}-{current_week:02d}) | Agents: {current_fish_count_for_print+len(fishermen)} | Biomass: {total_biomass:.2f} kg | Catch: {current_step_catch:.2f} kg | Profit: ${total_fleet_profit_this_week:.2f} | Dist: {total_fleet_distance_this_week:.2f} km")

######################################################################################################################################################

# Runtime variable to capture the start time
start_time = time.time()

initialize()
# Observe initial state regardless of frequency
observe() 

# Run the simulation loop
for j in range(1, n):
    update_one_unit_time()
    
    # Conditional observation for plotting/saving images (Runs once every OBSERVE_FREQUENCY weeks)
    if time1 % OBSERVE_FREQUENCY == 0:
        observe()

end_time = time.time()
runtime = end_time - start_time

if CAN_SAVE_FILES:
    # If the last step wasn't an observation step, run it one last time to ensure final plot is saved
    if n % OBSERVE_FREQUENCY != 0:
        observe()
        
    # Use ffmpeg to create a movie from the saved frames (will only use the frames that were saved)
    os.system("ffmpeg -v quiet -r 5 -i year_%04d.png -vcodec mpeg4 -y -s:v 1920x1080 simulation_movie.mp4")
    os.chdir(os.pardir) # move up to parent folder

# E. Print final runtime
print(f"\n--- Simulation finished. ---")
print(f"Total simulation steps: {n} weeks ({SIMULATION_YEARS} years)")
print(f"Total runtime: {runtime:.2f} seconds")

#----------------------------------------------------------------------------------------------------------------
