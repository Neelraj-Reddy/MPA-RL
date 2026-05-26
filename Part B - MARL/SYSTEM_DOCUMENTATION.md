# Fishing Fleet Simulation - Complete System Documentation

**Version:** 2.0 with Marine Protected Areas  
**Last Updated:** February 27, 2026  
**Status:** Production-ready with MPA integration

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Environment Configuration](#environment-configuration)
3. [Component Details](#component-details)
4. [Observation Space](#observation-space)
5. [Action Space](#action-space)
6. [Reward Function](#reward-function)
7. [Physics Model](#physics-model)
8. [Marine Protected Areas](#marine-protected-areas)
9. [Training Configuration](#training-configuration)
10. [Deployment Configuration](#deployment-configuration)
11. [Performance Metrics](#performance-metrics)
12. [Known Issues & Solutions](#known-issues--solutions)

---

## System Overview

### Architecture

The simulation consists of five core components:

```
OceanEnvironment (environment.py)
├── Temperature Grid (100×100)
├── Plankton Distribution
├── Ocean Currents
├── Wind Patterns
└── Marine Protected Areas (MPA Grid)

FishPopulation (fish_ecosystem.py)
├── Individual Fish Schools (2,500 initial)
├── Energy-based Metabolism
├── Temperature-driven Behavior
└── Plankton-based Reproduction

FishingFleet (fleet_physics.py)
├── Boat Physics (realistic drag model)
├── Fuel Consumption (multi-factor)
├── Cargo Management
└── Net Deployment Mechanics

FishingAgent (fishing_agent.py)
├── Deep Q-Network (DQN)
├── 33-dimensional Observations
├── Hybrid RL + Rule-based Control
└── Experience Replay Buffer

TrainingSession / FleetDeployer
├── Episode Management
├── Multi-agent Coordination
├── Logging & Visualization
└── Checkpoint System
```

### Design Philosophy

**Hybrid Intelligence:**
- **Rule-based:** Navigation to home port, emergency protocols, safety constraints
- **RL-learned:** Fish-finding strategies, net deployment timing, fuel optimization

**Realism:**
- Cubic drag for water resistance (realistic hydrodynamics)
- Cargo weight affects fuel consumption
- Temperature drives fish behavior
- Self-regulating ecosystem (no hard caps)

---

## Environment Configuration

### OceanEnvironment (`environment.py`)

#### Initialization Parameters

```python
OceanEnvironment(
    width=100,              # Grid width (units: km)
    height=100,             # Grid height (units: km)
    hours_per_tick=1        # Simulation timestep (1 hour)
)
```

#### Grid Systems

**Temperature Grid (100×100)**
```python
# Generation: Procedural with eddies and gradients
base_temp = 18°C
range = 14-22°C (optimal for fish)
variation = ±4°C (spatial eddies)
temporal_drift = ±1°C (seasonal simulation)
```

**Plankton Grid (100×100)**
```python
initial_value = 1.0 (baseline food density)
carrying_capacity K = 1.5
max_growth_rate r_max = 0.010
baseline_regen = 0.002 per hour (ensures availability)
temperature_optimal T_opt = 18°C
temperature_sigma = 5.0 (bell curve)
```

**Growth Model:**
```
growth = r_max × exp(-(T - T_opt)² / (2σ²)) × plankton × (1 - plankton/K)
plankton(t+1) = clip(plankton(t) + growth + baseline_regen, 0, K)
```

**Current Grid (u, v components)**
```python
# Procedural generation with vortices
magnitude = 0.1-0.3 units/hr
patterns = gyres + eddies + linear flow
update_frequency = every 24 hours
```

**Wind Grid (u, v components)**
```python
magnitude = 0.2-0.5 units/hr
patterns = trade winds + local effects
update_frequency = every 6 hours
```

#### Marine Protected Area System

```python
# MPA Configuration
mpa_grid = np.zeros((100, 100), dtype=np.float32)
mpa_update_interval = 720 hours (30 days)
mpa_coverage_target = 0.20 (20% of ocean)
mpa_persistence = 0.85 (85% stability between updates)
```

**MPA Formation Algorithm:**
1. Compute fish density using Gaussian kernels (radius=5) around each school
2. Identify top 20% density zones (80th percentile)
3. Keep 85% of existing MPAs (persistence)
4. Expand new high-density zones using morphological dilation (2 iterations)
5. Enforce 20% coverage constraint

**Update Trigger:**
```python
if env.time_step % 720 == 0:  # Every 30 days
    env.update_mpas(fish_population)
```

---

## Component Details

### FishPopulation (`fish_ecosystem.py`)

#### Initialization

```python
FishPopulation(
    initial_population=2500,  # Number of fish schools
    env_width=100,
    env_height=100
)
```

#### Per-School Attributes

```python
positions: np.array([2500, 2])  # (x, y) coordinates
energies: np.array([2500])      # Energy reserves (starts at 70)
last_ate: np.array([2500])      # Hours since last meal
```

#### Metabolism Parameters

```python
basal_metabolism = 0.08 per hour     # Very low for sustainability
move_cost = 0.02 per hour            # Movement energy expenditure
starvation_period = 360 hours        # 15 days without food = death
mortality_base = 0.00001             # Natural death rate (extremely low)
```

**Energy Loss:**
```
energy_loss = basal_metabolism + move_cost
energy(t+1) = energy(t) - energy_loss
if energy < 0: death
```

#### Feeding Mechanics

```python
# Fish schools eat plankton from their cell
food_available = plankton_grid[x, y]
nutrition_value = 20.0  # High reward for eating

if food_available > 0.05:
    food_consumed = min(0.3, food_available) × nutrition_value
    fish_energy += food_consumed
    plankton_grid[x, y] -= food_consumed / nutrition_value
    last_ate = 0
```

#### Reproduction System

```python
breeding_threshold = 150 energy  # Must be well-fed
breeding_cost = 50% of parent energy
breeding_probability = 0.10 (10% per hour if conditions met)

# Conditions for breeding:
if energy >= 150 AND food_available > 0.3 AND random() < 0.10:
    offspring_energy = parent_energy × 0.5
    parent_energy = parent_energy × 0.5
    create_new_school(offspring)
```

**6-Month Breeding Cycle:**
- Threshold of 150 energy requires ~6 months of feeding
- Most fish hover around 70-100 energy during survival
- Only well-fed schools in rich zones reproduce
- Self-regulating: population grows when food is abundant

#### Movement Behavior

```python
# Temperature-driven movement
optimal_temp = 18°C
temp_gradient = calculate_gradient(position)

# Move toward better temperature if current is unsuitable
if abs(current_temp - 18) > 3:
    move_toward_optimal_temp()
else:
    random_walk(distance=0.5 to 2.0 units)
```

#### Death Conditions

```python
# 1. Energy depletion
if energy <= 0:
    death

# 2. Starvation (15 days without food)
if hours_since_last_meal > 360:
    death

# 3. Natural mortality (very rare)
if random() < mortality_base:
    death
```

---

### FishingFleet (`fleet_physics.py`)

#### Initialization

```python
FishingFleet(
    num_boats=15,                        # Fleet size
    ports=[np.array([x, y]), ...],       # Port locations
    env_width=100,
    env_height=100
)
```

#### Boat State Arrays

```python
positions: np.array([15, 2])        # (x, y) coordinates
headings: np.array([15])            # Radians [0, 2π)
velocities: np.array([15])          # Speed units/hr
fuel_levels: np.array([15])         # Liters
cargo_levels: np.array([15])        # Tons of fish
nets_deployed: np.array([15])       # Boolean
```

#### Physical Constants

```python
# Capacity
max_fuel = 12000.0 liters           # Increased from 5000 for sustainability
max_cargo = 500.0 tons              # Fish storage

# Speed
max_speed = 2.0 units/hr            # Maximum velocity

# Fuel consumption base
idle_fuel_cost = 8.0 L/hr           # Reduced from 15 for better range

# Drag coefficients
hull_drag_empty = 1.5               # Hydrodynamic resistance (empty)
hull_drag_full = 3.5                # Hydrodynamic resistance (full cargo)
net_drag_coeffs = [2.0, 8.0]        # Light net, Heavy net
wind_drag_coeff = 0.5               # Air resistance
```

#### Net Types

```python
# Two trawl types per boat (randomly assigned)
trawl_types = [0, 1]  # 0: Light Pelagic, 1: Heavy Bottom

# Characteristics
catch_radii = [2.0, 3.5] units         # Effective fishing radius
catch_rates = [0.08, 0.20]             # Fish extraction rate per tick
max_catch_per_tick = [30.0, 80.0] tons # Hard cap per hour
```

#### Fuel Consumption Model

**Physics-based calculation:**

```python
# 1. Hull drag (cubic with velocity)
dynamic_hull_drag = hull_drag_empty + (hull_drag_full - hull_drag_empty) × cargo_ratio
temp_modifier = 1.05 if temp < 15°C else 1.0
water_resistance = dynamic_hull_drag × temp_modifier × (speed_thru_water)³

# 2. Cargo weight drag (quadratic)
cargo_weight_factor = cargo_ratio × 50.0
cargo_drag = cargo_weight_factor × (speed_thru_water)² × 0.5

# 3. Net resistance (quadratic)
net_resistance = net_drag_coeff × net_deployed × (speed_thru_water)²

# 4. Wind resistance (quadratic)
wind_resistance = wind_drag_coeff × (speed_thru_air)²

# Total fuel burn per hour:
fuel_burned = idle_fuel_cost + water_resistance + cargo_drag + net_resistance + wind_resistance
```

**Fuel Consumption Examples:**

| Scenario | Speed | Cargo | Net | Total Fuel | Days of Operation |
|----------|-------|-------|-----|------------|-------------------|
| Idle (docked) | 0.0 | 0% | No | 8 L/hr | 62.5 days |
| Cruising empty | 2.0 | 0% | No | 22 L/hr | 22.7 days |
| Fishing (light net) | 1.0 | 20% | Yes | 17 L/hr | 28.7 days |
| Fishing (heavy net) | 1.0 | 20% | Yes | 23 L/hr | 21.4 days |
| Returning full cargo | 2.0 | 100% | No | **138 L/hr** | **3.6 days** |
| Maximum burn | 2.0 | 100% | Yes | **170 L/hr** | **2.9 days** |

**Critical Insight:** Full-speed travel with full cargo **burns fuel 17× faster than idle**. Agents must learn efficient return strategies.

#### Port Refueling

```python
# Automatic refueling when within 1.5 units of any port
for port in ports:
    dist_to_port = ||position - port||
    if dist_to_port < 1.5:
        # Sell all cargo
        if cargo > 0:
            just_sold = True
            cargo_sold = cargo
            cargo = 0
        # Refuel to maximum
        fuel = max_fuel (12000.0 L)
```

**Key Point:** Boats refuel EVERY time they reach port. Running out of fuel means they're not returning frequently enough or traveling too inefficiently.

#### Fishing Mechanics

**Success Rate Factors:**

```python
# 1. Speed factor (slower = better catch)
speed_penalty = exp(-speed / 2.5)
success_rate = base_success × speed_penalty
# At speed 1.0: ~85% success
# At speed 2.0: ~45% success

# 2. Temperature suitability
temp_suitability = exp(-|(temp - 18)|² / 18)
success_rate × temp_suitability

# 3. Fish escape rate
escape_rate = 0.10 + 0.20 × normalized_speed
# At speed 1.0: 20% escape
# At speed 2.0: 30% escape

# Final catch:
actual_catch = theoretical_catch × success_rate × (1 - escape_rate)
```

**Extraction Process:**

```python
if net_deployed and not out_of_fuel and cargo < max_cargo:
    catchable_schools = fish within catch_radius
    for each school:
        theoretical = school_energy × catch_rate × speed_penalty
        actual = theoretical × success_rate × (1 - escape_rate)
        actual = min(actual, dynamic_hard_cap, space_left_in_cargo)
        
        fish_school.energy -= actual
        boat.cargo += actual
```

---

## Observation Space

### FishingAgent (`fishing_agent.py`)

**Total Dimensions: 33**

#### Breakdown by Category

| Index | Category | Dimension | Description | Normalization |
|-------|----------|-----------|-------------|---------------|
| 0-1 | Position | 2 | (x, y) coordinates | x ÷ 100, y ÷ 100 |
| 2-3 | Velocity | 2 | Magnitude, direction | mag ÷ 4.0, dir ÷ π |
| 4 | Heading | 1 | Current heading | heading ÷ 2π |
| 5-6 | Resources | 2 | Fuel %, cargo % | fuel ÷ max_fuel, cargo ÷ max_cargo |
| 7-10 | Navigation | 4 | Home vector (x, y), distance, fuel_check | normalized |
| 11 | Environment | 1 | Current temperature | (temp - 15) ÷ 10 |
| 12-19 | Fish Sensing | 8 | Fish density in 8 directions | biomass ÷ 1000 |
| 20-27 | Temperature Gradient | 8 | Temp suitability in 8 directions | normalized |
| 28 | Equipment | 1 | Net deployed | 0.0 or 1.0 |
| 29-30 | Economics | 2 | Expected revenue, fuel value | normalized |
| **31** | **MPA Status** | **1** | **Inside protected area** | **0.0 or 1.0** |
| **32** | **Temporal** | **1** | **Time of year (seasonal)** | **(time_step % 8760) ÷ 8760** |

#### Detailed Observation Construction

```python
def get_observation(boat_state, fish_pop, env):
    obs = []
    
    # POSITION (2 dims)
    obs.extend([
        boat_state['position'][0] / env_width,
        boat_state['position'][1] / env_height
    ])
    
    # VELOCITY (2 dims)
    velocity_mag = ||boat_state['velocity']||
    velocity_dir = arctan2(vy, vx)
    obs.extend([velocity_mag / 4.0, velocity_dir / π])
    
    # HEADING (1 dim)
    obs.append(boat_state['heading'] / (2π))
    
    # RESOURCES (2 dims)
    obs.extend([
        boat_state['fuel'] / max_fuel,
        boat_state['cargo'] / max_cargo
    ])
    
    # NAVIGATION (4 dims)
    home_vector = home_port - boat_state['position']
    home_distance = ||home_vector||
    home_direction = home_vector / (home_distance + ε)
    
    obs.extend([
        home_direction[0],  # X component to home
        home_direction[1],  # Y component to home
        home_distance / 100.0,  # Normalized distance
        fuel_sufficient_for_return(boat_state, home_distance)  # 1.0 or 0.0
    ])
    
    # TEMPERATURE (1 dim)
    current_temp = env.get_temperature(*boat_state['position'])
    obs.append((current_temp - 15.0) / 10.0)
    
    # FISH DETECTION (8 dims) - Sonar sweep
    angles = [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]
    for angle in angles:
        direction = [cos(angle), sin(angle)]
        biomass_in_cone = 0.0
        for school in fish_within_vision_range(10 units):
            if school_in_cone(school, direction, cone_angle=30°):
                distance = ||school.position - boat.position||
                biomass_in_cone += school.energy / (distance + 1)
        obs.append(biomass_in_cone / 1000.0)  # Normalized
    
    # TEMPERATURE GRADIENT (8 dims)
    for angle in angles:
        direction = [cos(angle), sin(angle)]
        forward_pos = boat_position + direction × 5.0
        forward_temp = env.get_temperature(*forward_pos)
        temp_suitability = exp(-((forward_temp - 18)² / 18))
        obs.append(temp_suitability)
    
    # EQUIPMENT (1 dim)
    obs.append(float(boat_state['net_deployed']))
    
    # ECONOMICS (2 dims)
    expected_revenue = boat_state['cargo'] × fish_price (1500 $/ton)
    current_fuel_value = boat_state['fuel'] × fuel_cost (1.2 $/L)
    obs.extend([
        min(expected_revenue / 750000.0, 1.0),
        current_fuel_value / 6000.0
    ])
    
    # MPA STATUS (1 dim) ⭐ NEW
    obs.append(float(env.is_in_mpa(*boat_state['position'])))
    
    # TEMPORAL AWARENESS (1 dim) ⭐ NEW
    # Allows learning seasonal patterns
    obs.append((env.time_step % 8760) / 8760.0)
    
    return np.array(obs, dtype=np.float32)
```

---

## Action Space

### Continuous Action Vector

**3 dimensions: [heading_change, throttle, net_deploy]**

#### 1. Heading Change
```python
heading_change ∈ [-0.5, +0.5] radians per step
# Applied to current heading
new_heading = old_heading + heading_change
```

#### 2. Throttle
```python
throttle ∈ [0.0, 1.0]
# Applied to max_speed
new_velocity = throttle × max_speed (2.0 units/hr)
```

#### 3. Net Deployment
```python
net_deploy ∈ {0.0, 1.0}
# Typical agent output: continuous [-1, +1]
# Converted: net_deploy = 1.0 if raw_output > 0 else 0.0
```

### Action Selection Process

```python
def select_action(observation, boat_state):
    # ========== RULE-BASED SAFETY OVERRIDES ==========
    
    # 1. Out of fuel → Drift
    if fuel <= 0:
        return [0.0, 0.0, 0.0]
    
    # 2. Must return home → Navigate to port
    cargo_full = cargo >= max_cargo × 0.95
    fuel_insufficient = fuel < distance_to_home × 50 × 1.3
    fuel_critical = fuel < max_fuel × 0.15
    
    if cargo_full OR fuel_insufficient OR fuel_critical:
        target_heading = arctan2(home_y - y, home_x - x)
        heading_diff = angle_difference(target_heading, current_heading)
        
        if distance_to_home < 2.0:
            return [heading_diff × 0.6, 0.3, 0.0]  # Slow approach
        else:
            throttle = min(1.0, fuel / (distance × 30))  # Fuel-aware speed
            return [heading_diff, throttle, 0.0]
    
    # ========== DEEP Q-NETWORK DECISION ==========
    
    with torch.no_grad():
        observation_tensor = torch.FloatTensor(observation)
        q_values = policy_net(observation_tensor)
    
    # Epsilon-greedy exploration
    if random() < epsilon:
        heading_change = uniform(-0.5, 0.5)
        throttle = uniform(0.0, 1.0)
        net_deploy = choice([0.0, 1.0])
    else:
        heading_change = clip(q_values[0], -0.5, 0.5)
        throttle = clip(q_values[1], 0.0, 1.0)
        net_deploy = 1.0 if q_values[2] > 0 else 0.0
    
    # ========== SAFETY OVERRIDES FOR NET ==========
    
    # Don't deploy net if:
    # - Cargo full (98%+)
    if cargo >= max_cargo × 0.98:
        net_deploy = 0.0
    
    # - Speed too high (>4.0 units/hr, net damage risk)
    if speed > 4.0:
        net_deploy = 0.0
    
    # - No fish detected (wasteful drag)
    fish_density = observation[12:20]  # 8-direction fish sensors
    if max(fish_density) < 0.02:
        net_deploy = 0.0
    
    # - Temperature unsuitable (<14°C or >22°C)
    if temp < 14 or temp > 22:
        net_deploy = 0.0
    
    return [heading_change, throttle, net_deploy]
```

---

## Reward Function

### FishingAgent.calculate_reward()

**Design Goal:** Shape agents toward profitable, fuel-efficient, law-abiding fishing.

#### Reward Components

```python
def calculate_reward(prev_state, action, new_state, just_sold, cargo_sold):
    reward = 0.0
    
    # ========== POSITIVE REWARDS ==========
    
    # 1. JACKPOT: Successful sale at port
    if just_sold and cargo_sold > 0:
        revenue = cargo_sold × fish_price (1500 $/ton)
        trip_profit = revenue / 1000.0  # Scale for NN stability
        reward += trip_profit
        
        # Efficiency bonus
        fuel_used_this_trip = prev_state.get('trip_fuel', 0)
        if fuel_used_this_trip > 0:
            efficiency = revenue / (fuel_used_this_trip × fuel_cost)
            reward += efficiency × 10.0
    
    # 2. PROGRESS: Catching fish
    cargo_gained = new_state['cargo'] - prev_state['cargo']
    if cargo_gained > 0:
        reward += cargo_gained × 2.0  # Increased from 0.8
    
    # 3. HOLDING BONUS: Reward for having cargo
    if new_state['cargo'] > 0:
        reward += new_state['cargo'] × 0.01
    
    # ========== NEGATIVE REWARDS (PENALTIES) ==========
    
    # 4. FUEL COST: Continuous efficiency pressure
    fuel_used = max(0, prev_state['fuel'] - new_state['fuel'])
    fuel_cost = fuel_used × fuel_cost_per_liter (1.2 $/L) / 1000.0
    reward -= fuel_cost
    
    # 5. TIME COST: Encourage decisive action
    reward -= 0.02  # Small penalty per timestep
    
    # 6. WASTEFUL DRAG: Empty net deployment
    if action[2] > 0.5 and cargo_gained < 0.5:
        reward -= 1.0  # Deployed net but caught nothing
    
    # 7. FUEL DEATH PENALTY (CATASTROPHIC) ⚠️
    if new_state['fuel'] <= 0 and prev_state['fuel'] > 0:
        reward -= 200.0  # Massive failure (increased from -100)
    
    # 8. LOW FUEL WARNING
    fuel_pct = new_state['fuel'] / new_state['max_fuel']
    dist_to_port = ||self.home_port - new_state['position']||
    if fuel_pct < 0.2 and dist_to_port > 30:
        reward -= 10.0  # Risky situation
    
    # 9. MARINE PROTECTED AREA VIOLATIONS ⚠️⚠️
    inside_mpa = new_state.get('inside_mpa', False)
    if inside_mpa:
        # Light penalty for just being there (fuel waste)
        reward -= 2.0
        
        # HEAVY PENALTY for illegal fishing
        if action[2] > 0.5:  # Net deployed
            reward -= 50.0  # Ecological damage, legal violation
    
    # 10. DEAD ZONE PENALTY: Bad temperature areas
    temp_suitability = exp(-((new_state['current_temp'] - 18)² / 18))
    if temp_suitability < 0.2:
        reward -= 0.5  # Discourage staying in unproductive zones
    
    return reward
```

#### Reward Shaping Philosophy

**Positive Signals:**
- Large reward for successful sales (hundreds of points)
- Incremental rewards for catching fish (builds motivation)
- Small bonus for holding cargo (prevents dumping)

**Negative Signals:**
- Proportional fuel costs (teaches efficiency)
- Small time penalty (prevents idling)
- Heavy penalties for violations (safety-critical)

**Priority Order:**
1. Don't die (fuel death = -200)
2. Don't violate MPAs (illegal fishing = -50)
3. Return home efficiently (efficiency bonus)
4. Catch fish profitably (progressive rewards)

---

## Physics Model

### Hydrodynamics

**Speed Through Water:**
```
V_water = V_boat - V_current
drag ∝ V_water³  (Cubic relationship for hydrodynamic resistance)
```

**Speed Through Air:**
```
V_air = V_boat - V_wind
drag ∝ V_air²  (Quadratic relationship for aerodynamic resistance)
```

### Fuel Burn Equation

```
F_total = F_idle + F_hull + F_cargo + F_net + F_wind

where:
F_idle = 8.0 L/hr (constant)

F_hull = dynamic_hull_drag × temp_modifier × V_water³
  dynamic_hull_drag = 1.5 + (3.5 - 1.5) × cargo_ratio
  temp_modifier = 1.05 if T < 15°C, else 1.0

F_cargo = 50 × cargo_ratio × V_water² × 0.5

F_net = net_drag_coeff × net_deployed × V_water²
  net_drag_coeff ∈ {2.0, 8.0}  // Light, Heavy

F_wind = 0.5 × V_air²
```

### Example Calculations

**Scenario: Full-speed return with full cargo**
```
V_boat = 2.0 units/hr
cargo_ratio = 1.0
net_deployed = False

F_idle = 8.0
F_hull = 3.5 × 1.0 × (2.0)³ = 28.0
F_cargo = 50 × 1.0 × (2.0)² × 0.5 = 100.0
F_net = 0.0
F_wind = 0.5 × (2.0)² = 2.0

F_total = 138.0 L/hr
```

**Result:** With 12,000L tank, can operate for 87 hours (3.6 days) at this burn rate.

---

## Marine Protected Areas

### Purpose

**Ecological Protection:**
- Protect fish spawning grounds (high-density zones)
- Create refugia for population recovery
- Simulate real-world marine conservation

**Agent Challenge:**
- Agents must learn spatial avoidance strategies
- Penalty for fishing in MPAs teaches compliance
- Balances fishing pressure across ocean

### Implementation

#### MPA Grid (`environment.py`)

```python
# Initialization
self.mpa_grid = np.zeros((width, height), dtype=np.float32)
# Values: 0.0 = open water, 1.0 = protected area

self.mpa_update_interval = 720  # Update every 30 days
self.mpa_coverage_target = 0.20  # Target 20% coverage
self.mpa_persistence = 0.85     # 85% of MPAs persist between updates
```

#### Update Algorithm

```python
def update_mpas(fish_population):
    # Only update every 30 days
    if time_step % mpa_update_interval != 0:
        return
    
    # Step 1: Compute fish density grid
    fish_density = np.zeros((width, height))
    for school in fish_population:
        x, y = school.position
        energy = school.energy
        
        # Create Gaussian kernel (radius=5) around fish
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                grid_x = clip(x + dx, 0, width-1)
                grid_y = clip(y + dy, 0, height-1)
                distance = sqrt(dx² + dy²)
                contribution = energy × exp(-(distance² / (2 × 2²)))
                fish_density[grid_x, grid_y] += contribution
    
    # Step 2: Persistence (keep 85% of existing MPAs)
    existing_mpas = (mpa_grid > 0.5)
    keep_mask = (random(size=(width, height)) < 0.85)
    persistent_mpa = existing_mpas & keep_mask
    
    # Step 3: Identify high-density zones (top 20%)
    density_threshold = percentile(fish_density, 80)
    high_density_zones = fish_density >= density_threshold
    
    # Step 4: Combine persistent + new zones
    candidate_mpa = persistent_mpa | high_density_zones
    
    # Step 5: Expand to connected regions (morphological dilation)
    expanded_mpa = binary_dilation(candidate_mpa, iterations=2)
    
    # Step 6: Enforce coverage target (20%)
    if sum(expanded_mpa) > width × height × 0.20:
        # Remove lowest-density cells until target reached
        mpa_density = fish_density × expanded_mpa
        priority = flatten(mpa_density)
        sorted_indices = argsort(priority)
        num_to_remove = sum(expanded_mpa) - int(width × height × 0.20)
        remove_indices = sorted_indices[:num_to_remove]
        expanded_mpa[remove_indices] = False
    
    self.mpa_grid = expanded_mpa.astype(np.float32)
```

#### Checking MPA Status

```python
def is_in_mpa(x, y):
    grid_x = clip(int(x), 0, width - 1)
    grid_y = clip(int(y), 0, height - 1)
    return mpa_grid[grid_x, grid_y] > 0.5
```

### Visualization

MPAs shown as **red/pink overlay** in all GIFs and visualizations:
```python
if hasattr(env, 'mpa_grid') and np.any(env.mpa_grid > 0.5):
    mpa_mask = env.mpa_grid > 0.5
    ax.imshow(mpa_mask.T, origin='lower',
             cmap='Reds', alpha=0.35, vmin=0, vmax=2)
```

---

## Training Configuration

### Standard Training (`train_fishing_agents.py`)

```python
config = {
    # Environment
    'env_width': 100,
    'env_height': 100,
    'hours_per_tick': 1,
    'initial_fish': 2500,
    
    # Fleet
    'num_boats': 5,          # 5 independent agents
    'num_ports': 3,          # Randomized each episode
    
    # Training
    'num_episodes': 200,
    'steps_per_episode': 2190,  # ~3 months per episode
    
    # Agent hyperparameters
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.995,
    'learning_rate': 0.0003,
    'batch_size': 64,
    'gamma': 0.99,           # Discount factor
    'memory_size': 10000,     # Replay buffer
    'target_update': 100      # Target network sync frequency
}
```

### DQN Architecture

```python
# fishing_agent.py: _build_network()
nn.Sequential(
    nn.Linear(33, 128),     # Input: 33-dim observation
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3)        # Output: 3 Q-values (heading, throttle, net)
)

Optimizer: Adam (lr=0.0003)
Loss: Huber Loss (SmoothL1Loss)
```

### Training Loop

```python
for episode in range(num_episodes):
    # 1. Reset environment
    env = OceanEnvironment(100, 100, 1)
    fish = FishPopulation(2500, 100, 100)
    
    # 2. Randomize ports (generalization)
    ports = generate_random_ports(num_ports=3)
    fleet = FishingFleet(num_boats=5, ports=ports)
    
    # 3. Run episode
    for step in range(steps_per_episode):
        # Get observations and actions
        for boat_idx in range(num_boats):
            state = get_boat_state(boat_idx)
            obs = agent[boat_idx].get_observation(state, fish, env)
            action = agent[boat_idx].select_action(obs, state)
        
        # Step simulation
        fleet.step(actions, fish, env)
        fish.step(env)
        env.step()
        env.update_mpas(fish)  # Update MPAs every 30 days
        
        # Calculate rewards
        for boat_idx in range(num_boats):
            new_state = get_boat_state(boat_idx)
            reward = agent[boat_idx].calculate_reward(
                prev_state, action, new_state,
                fleet.just_sold[boat_idx],
                fleet.cargo_sold[boat_idx]
            )
            
            # Store experience
            agent[boat_idx].store_experience(
                prev_obs, action, reward, new_obs, done=False
            )
            
            # Train step (if enough experiences)
            loss = agent[boat_idx].train_step()
    
    # 4. Decay exploration
    for agent in agents:
        agent.epsilon = max(epsilon_end, agent.epsilon × epsilon_decay)
    
    # 5. Update target networks
    if episode % target_update_freq == 0:
        for agent in agents:
            agent.update_target_network()
    
    # 6. Save checkpoints
    if episode % 50 == 0:
        save_checkpoint(episode)
```

### Checkpoint System

```python
# Saved every 50 episodes + best performance
checkpoints/
    agent_0_ep50.pt
    agent_0_ep100.pt
    ...
    agent_0_best.pt        # Best average reward
    agent_0_epfinal.pt     # Final model

# Each checkpoint contains:
torch.save({
    'policy_net_state_dict': policy_net.state_dict(),
    'target_net_state_dict': target_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epsilon': epsilon,
    'total_reward': total_reward,
    'trips_completed': trips_completed,
    'total_catch': total_catch,
    'memory': list(memory)  # Optional: save replay buffer
}, filepath)
```

---

## Deployment Configuration

### Fleet Deployment (`deploy_fleet.py`)

```python
deployment_config = {
    # Environment
    'env_width': 100,
    'env_height': 100,
    'hours_per_tick': 1,
    'initial_fish': 2500,
    
    # Fleet composition
    'num_ports': 5,
    'model_checkpoints': [
        'checkpoints/agent_0_best.pt',
        'checkpoints/agent_1_best.pt',
        'checkpoints/agent_2_best.pt',
        'checkpoints/agent_3_best.pt',
        'checkpoints/agent_4_best.pt'
    ],
    'boats_per_model': 3,     # Total: 5 models × 3 = 15 boats
    
    # Deployment duration
    'deployment_steps': 8760,  # 1 full year simulation
    
    # Visualization
    'generate_gifs': True,
    'gif_fps': 5,
    'frames_to_capture': 500   # ~every 17.5 hours
}
```

### Deployment Process

```python
class FleetDeployer:
    def __init__(self, config):
        # Load trained agents
        self.agents = []
        for model_idx, checkpoint_path in enumerate(model_checkpoints):
            for instance in range(boats_per_model):
                agent = FishingAgent(...)
                agent.load_checkpoint(checkpoint_path)
                agent.epsilon = 0.05  # Minimal exploration for deployment
                self.agents.append(agent)
        
        # Total: 15 agents (5 models × 3 instances)
        self.total_boats = 15
        
        # Initialize environment
        self.env = OceanEnvironment(100, 100, 1)
        self.fish = FishPopulation(2500, 100, 100)
        self.fleet = FishingFleet(15, ports, 100, 100)
    
    def deploy(self):
        for step in range(deployment_steps):
            # Get actions
            actions = []
            for boat_idx in range(total_boats):
                state = get_boat_state(boat_idx)
                obs = agent[boat_idx].get_observation(state, fish, env)
                action = agent[boat_idx].select_action(obs, state)
                actions.append(action)
            
            # Step simulation
            fleet.step(actions, fish, env)
            fish.step(env)
            env.step()
            env.update_mpas(fish)
            
            # Track metrics
            for boat_idx in range(total_boats):
                deployment_catches[boat_idx] += fleet.cargo_sold[boat_idx]
                if fleet.just_sold[boat_idx]:
                    deployment_trips[boat_idx] += 1
            
            # Capture frame for GIFs
            if step % frame_interval == 0:
                frame = create_frame(step)
                frames.append(frame)
        
        # Generate GIFs
        generate_gifs()
        print_deployment_summary()
```

### GIF Generation

Generated GIFs:
1. `deployment_all_boats.gif` (all 15 boats)
2. `deployment_model_0_fleet.gif` (Model 0's 3 boats)
3. `deployment_model_1_fleet.gif` (Model 1's 3 boats)
4. `deployment_model_2_fleet.gif` (Model 2's 3 boats)
5. `deployment_model_3_fleet.gif` (Model 3's 3 boats)
6. `deployment_model_4_fleet.gif` (Model 4's 3 boats)

Visualization elements:
- **Temperature background** (blue=cold, red=warm)
- **Fish schools** (cyan circles, size ∝ energy)
- **Ports** (white stars)
- **Boats** (colored circles with heading arrows)
- **Nets deployed** (red halos)
- **Marine Protected Areas** (red overlay) ⭐ NEW

---

## Performance Metrics

### Training Metrics

Tracked per episode:
```python
episode_rewards = [agent.total_reward for agent in agents]
episode_catches = [agent.total_catch for agent in agents]
episode_trips = [agent.trips_completed for agent in agents]
training_losses = [agent.last_loss for agent in agents]
```

Logged output:
```
Episode 50/200 | Avg Reward: 1523.4 | Avg Catch: 245.3 tons | Trips: 3.2 | Loss: 0.054
Episode 100/200 | Avg Reward: 2847.1 | Avg Catch: 512.7 tons | Trips: 5.8 | Loss: 0.032
```

### Deployment Metrics

Per-boat tracking:
```python
deployment_catches[15]   # Total tons caught per boat
deployment_trips[15]     # Number of successful sales
deployment_rewards[15]   # Cumulative reward
fuel_deaths              # Boats that ran out of fuel
mpa_violations           # Net deployments in MPAs
```

Per-model aggregation:
```python
model_catches = [sum(catches[i:i+3]) for i in range(0, 15, 3)]
model_trips = [sum(trips[i:i+3]) for i in range(0, 15, 3)]
```

Example output:
```
Model 0: 89,432 tons, 1,124 trips
Model 1: 115,982 tons, 1,359 trips  ← Best performer
Model 2: 78,654 tons, 982 trips
Model 3: 92,101 tons, 1,087 trips
Model 4: 101,264 tons, 1,217 trips

Fleet Total: 277,433 tons, 2,069 trips over 1 year
```

### Ecosystem Health

Tracked during simulation:
```python
fish_population_timeline = [num_schools at each step]
total_energy_timeline = [sum of all fish energies]
plankton_mean_timeline = [average plankton density]
```

Healthy ecosystem indicators:
- Fish population stable or growing: 2,500 → 2,900+ schools
- Total energy increasing: 175,000 → 200,000+
- Plankton regenerating: Mean > 0.8

---

## Known Issues & Solutions

### Issue 1: Boats Running Out of Fuel

**Symptoms:**
- 14/15 boats dead by day 90 in deployment
- Boats drifting at boundaries with full cargo

**Root Cause:**
- Full-speed travel with full cargo = 138 L/hr (extreme burn rate)
- Agents not returning to port frequently enough
- Training episodes too short to learn long-term fuel management

**Analysis:**
```
Fuel capacity: 12,000 L
Worst-case burn: 138 L/hr (full speed + full cargo)
Operational time: 87 hours = 3.6 days
```

**Solutions Implemented:**
1. ✅ Increased fuel capacity: 5,000 → 12,000 liters
2. ✅ Reduced idle cost: 15 → 8 L/hr
3. ✅ Enhanced fuel death penalty: -100 → -200
4. ✅ Added low-fuel warning penalty: -10 when <20% fuel far from port
5. ✅ Increased training episode lengths for better learning

**Future Improvements:**
- Weekly training episodes with random starts (temporal generalization)
- Progressive fuel penalty (increases as fuel drops)
- Efficiency bonus in rewards (tons caught per liter burned)

### Issue 2: Fish Population Instability

**Historical Problems:**
- Exponential growth: 2,500 → 386,000 (unsustainable)
- Mass starvation: 2,500 → 0 (ecosystem collapse)

**Solutions Implemented:**
1. ✅ Lowered basal metabolism: 0.3 → 0.08 (fish survive longer)
2. ✅ Increased nutrition value: 10.0 → 20.0 (feeding more rewarding)
3. ✅ High breeding threshold: 150 energy (requires well-fed state)
4. ✅ Breeding cost: 50% of parent energy (expensive reproduction)
5. ✅ Very low natural mortality: 0.00001 (prevents random deaths)
6. ✅ Baseline plankton regen: 0.002 per hour (ensures food availability)

**Current Status:** Population stable at 2,500-2,900 schools over 1 year

### Issue 3: MPA Formation

**Challenge:**
- MPAs must be ecologically meaningful (protect fish concentrations)
- Must maintain 20% coverage consistently
- Should persist between updates (85% stability)

**Implementation:**
1. ✅ Fish-density based formation (Gaussian kernels, radius=5)
2. ✅ persistence mechanism (85% of MPAs kept each update)
3. ✅ Connected component growth (morphological dilation)
4. ✅ Coverage enforcement (trim to 20% if exceeded)

**Testing:**
- Run `test_mpa_system.py` to validate MPA formation
- Check GIFs for red overlay showing protected areas
- Monitor fish populations in МРА vs non-MPA zones

### Issue 4: Training Convergence

**Challenge:**
- Agents need to learn complex multi-objective strategies
- Fuel management vs. profit maximization trade-off
- Seasonal patterns and temporal awareness

**Solutions:**
1. ✅ Experience replay (10,000 sample buffer)
2. ✅ Target network stabilization (update every 100 episodes)
3. ✅ Epsilon decay (1.0 → 0.05 over 200 episodes)
4. ✅ Randomized port locations (prevents overfitting)
5. ✅ Checkpoint system (best model saved)

**Training Duration:**
- 200 episodes × 2,190 steps ≈ 438,000 timesteps
- ~2-3 hours on modern CPU
- GPU acceleration available but not required

### Issue 5: Observation Dimension Mismatch

**Problem:**
- Old agents trained with 31-dim observations
- New system uses 33-dim (MPA + temporal awareness)

**Solution:**
- ⚠️ Existing checkpoints incompatible with new observation space
- Must retrain all agents from scratch with 33-dim observations
- Backward compatibility not maintained (intentional upgrade)

**Migration Path:**
1. Archive old checkpoints: `checkpoints/` → `checkpoints_old/`
2. Run new training: `python train_fishing_agents.py`
3. Generate new checkpoints with 33-dim observation space
4. Deploy with updated agents

---

## Quick Reference

### File Structure

```
fishing_simulation/
├── environment.py              # Ocean physics + MPAs
├── fish_ecosystem.py           # Fish population dynamics
├── fleet_physics.py            # Boat physics + fuel
├── fishing_agent.py            # DQN agent (33-dim obs)
├── train_fishing_agents.py     # Training script
├── deploy_fleet.py             # Deployment script
├── make_gif.py                 # Single-model GIF generator
├── fish_energy_analysis.py     # Ecosystem comparison analysis
├── test_mpa_system.py          # MPA integration tests
├── analyze_fuel_consumption.py # Fuel burn analysis
├── SYSTEM_DOCUMENTATION.md     # This file
├── TRAINING_RESULTS_ANALYSIS.md # Training metrics analysis
└── checkpoints/                # Saved agent models
    ├── agent_0_best.pt
    ├── agent_1_best.pt
    ...
```

### Key Commands

```bash
# Training (200 episodes, ~3 hours)
python train_fishing_agents.py

# Deployment (1 year simulation, generates 6 GIFs)
python deploy_fleet.py

# Quick visualization (3 boats, short simulation)
python make_gif.py

# Ecosystem analysis (fish alone vs. with boats)
python fish_energy_analysis.py

# Test MPA system
python test_mpa_system.py

# Analyze fuel consumption
python analyze_fuel_consumption.py
```

### Key Parameters to Tune

**For faster training:**
- Reduce `steps_per_episode`: 2190 → 1000
- Reduce `num_episodes`: 200 → 100
- Increase `epsilon_decay`: 0.995 → 0.98

**For better fuel management:**
- Increase fuel death penalty: -200 → -500
- Add progressive fuel warnings at 50%, 30%, 20%
- Reduce hull drag coefficients

**For stronger MPA protection:**
- Increase MPA penalty: -50 → -100
- Increase MPA coverage: 0.20 → 0.30
- Decrease update interval: 720 → 360 hours

**For ecosystem stability:**
- Increase plankton baseline regen: 0.002 → 0.003
- Adjust breeding threshold: 150 → 120 (faster reproduction)
- Lower breeding cost: 50% → 40% (less expensive)

---

## Version History

**v2.0 (Current) - February 27, 2026**
- ✅ Added Marine Protected Area system (intelligent, fish-density based)
- ✅ Expanded observation space to 33 dimensions (MPA + temporal)
- ✅ Enhanced reward penalties (MPA violations, fuel death)
- ✅ Added MPA visualization to all GIFs
- ✅ Integrated MPA updates in all simulation loops
- ✅ Created comprehensive fuel consumption analysis
- ✅ Full system documentation

**v1.5 - Previous**
- ✅ Fixed fuel crisis (12K liters, 8 L/hr idle)
- ✅ Balanced fish ecosystem (sustainable population)
- ✅ Deployment script with 15 boats
- ✅ Individual GIF generation per model
- ✅ Checkpoint system for training

**v1.0 - Initial**
- ✅ Core simulation framework
- ✅ DQN-based agents
- ✅ Realistic physics model
- ✅ Basic training and deployment

---

## Future Roadmap

### Planned Features

1. **Weekly Training Episodes**
   - Random week selection (temporal generalization)
   - Week number as model input
   - Learn seasonal patterns

2. **Advanced Fuel Management**
   - Progressive fuel warnings (50%, 30%, 20% thresholds)
   - Fuel efficiency metric in logs
   - Efficiency bonus in rewards

3. **Multi-species Ecosystem**
   - Predator-prey dynamics
   - Different fish types with varying values
   - Quota systems per species

4. **Weather Systems**
   - Storms (increased fuel consumption, reduced catch rates)
   - Seasonal patterns (fish migration)
   - Wave height (affects fishing success)

5. **Economic Complexity**
   - Dynamic fish prices (supply/demand)
   - Fuel price fluctuations
   - Maintenance costs

6. **Social Dynamics**
   - Competition between boats (resource depletion in local areas)
   - Cooperation (information sharing about fish locations)
   - Port capacity limits

---

## Contact & Support

For questions, issues, or contributions related to this simulation system, refer to:
- Training results analysis: `TRAINING_RESULTS_ANALYSIS.md`
- Code comments in each `.py` file
- Test scripts for validation examples

**System Status:** Production-ready with Marine Protected Areas integration complete.

**Last Validated:** February 27, 2026

---

*End of Documentation*
