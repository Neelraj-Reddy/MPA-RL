import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class FishingAgent:
    """
    Hybrid RL + Rule-Based Fishing Boat Agent
    
    Philosophy: Humans know certain things explicitly (like "sail towards home port"),
    but learn through experience how to find fish efficiently.
    
    Rule-Based Components:
    - Navigation to home port (we always know which direction is home)
    - Emergency protocols (out of fuel, cargo full)
    - Safety constraints (net deployment rules)
    
    RL Components:
    - Where to search for fish
    - When to deploy nets
    - Speed/fuel optimization
    - Exploration vs exploitation
    """
    
    def __init__(self, boat_id, home_port, env_width, env_height):
        self.boat_id = boat_id
        self.home_port = np.array(home_port)
        self.env_width = env_width
        self.env_height = env_height
        
        # Observation space dimension
        self.observation_dim = 33  # Detailed below in get_observation() - includes MPA and temporal
        
        # Domain Knowledge (given to agent, not learned)
        self.fish_temp_min = 14.0
        self.fish_temp_max = 22.0
        self.fish_temp_optimal = 18.0
        
        # Sensory Capabilities
        self.vision_range = 10.0  # Can detect fish within 10 units
        self.temp_sensor_range = 5.0  # Can sense temperature patterns ahead
        
        # Economic Knowledge
        self.fish_price_per_ton = 1500.0  # USD per ton
        self.fuel_cost_per_liter = 1.2    # USD per liter
        
        # Deep Q-Network for learning fishing strategies
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.memory = deque(maxlen=10000)
        
        # Training hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate (start at 100%)
        self.epsilon_min = 0.10  # Keep some exploration (raised from 0.05)
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        
        # Performance tracking
        self.total_reward = 0.0
        self.trips_completed = 0
        self.total_catch = 0.0
        self.fuel_efficiency_history = []
        
    def _build_network(self):
        """Build Deep Q-Network for decision making"""
        return nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 Q-values for action components
        )
    
    def get_observation(self, boat_state, fish_pop, env):
        """
        Construct observation vector - what the captain "sees"
        
        Total: 31 dimensions
        - Position: 2 (normalized x, y)
        - Velocity: 2 (magnitude, direction)
        - Heading: 1
        - Fuel & cargo: 2
        - Navigation info: 4 (home direction + distance + fuel check)
        - Temperature: 1 (current location)
        - Fish detection (8 directions): 8
        - Temperature gradient (8 directions): 8
        - Net status: 1
        - Economic factors: 2 (expected revenue, fuel value)
        """
        obs = []
        
        # === SELF STATE (7 dims) ===
        obs.extend([
            boat_state['position'][0] / self.env_width,  # Normalized x
            boat_state['position'][1] / self.env_height,  # Normalized y
        ])
        
        velocity_mag = np.linalg.norm(boat_state['velocity'])
        velocity_dir = np.arctan2(boat_state['velocity'][1], boat_state['velocity'][0])
        obs.extend([velocity_mag / 4.0, velocity_dir / np.pi])  # Normalized velocity
        
        obs.append(boat_state['heading'] / (2 * np.pi))  # Normalized heading
        obs.append(boat_state['fuel'] / boat_state['max_fuel'])  # Fuel percentage
        obs.append(boat_state['cargo'] / boat_state['max_cargo'])  # Cargo percentage
        
        # === NAVIGATION (4 dims) - EXPLICIT KNOWLEDGE ===
        home_vector = self.home_port - boat_state['position']
        home_distance = np.linalg.norm(home_vector)
        home_direction = home_vector / (home_distance + 1e-6)
        
        obs.extend([
            home_direction[0],  # X component to home
            home_direction[1],  # Y component to home
            home_distance / max(self.env_width, self.env_height),  # Normalized distance
            self._fuel_sufficient_for_return(boat_state, home_distance)  # 1.0 if can return, 0.0 if risky
        ])
        
        # === ENVIRONMENTAL SENSING (1 dim) ===
        current_temp = env.get_temperature(*boat_state['position'])
        obs.append((current_temp - 15.0) / 10.0)  # Normalized around typical range
        
        # === FISH DETECTION (8 dims) - Sonar/Visual in 8 directions ===
        fish_density = self._sense_fish_nearby(boat_state['position'], fish_pop)
        obs.extend(fish_density)
        
        # === TEMPERATURE GRADIENT (8 dims) - Finding optimal fishing zones ===
        temp_gradient = self._sense_temperature_gradient(boat_state['position'], env)
        obs.extend(temp_gradient)
        
        # === EQUIPMENT STATE (1 dim) ===
        obs.append(float(boat_state['net_deployed']))
        
        # === ECONOMIC FACTORS (2 dims) ===
        expected_revenue = boat_state['cargo'] * self.fish_price_per_ton
        current_fuel_value = boat_state['fuel'] * self.fuel_cost_per_liter
        obs.append(min(expected_revenue / 750000.0, 1.0))  # Normalized expected revenue
        obs.append(current_fuel_value / 6000.0)  # Normalized fuel asset value
        
        # === MARINE PROTECTED AREA (1 dim) - CRITICAL FOR COMPLIANCE ===
        obs.append(float(boat_state.get('inside_mpa', False)))  # 1.0 if in MPA, 0.0 if safe
        
        # === TEMPORAL AWARENESS (1 dim) - For seasonal patterns ===
        # Normalized time of year [0-1] for understanding fish migration/breeding cycles
        obs.append((env.time_step % 8760) / 8760.0)  # Position in annual cycle
        
        return np.array(obs, dtype=np.float32)
    
    def _sense_fish_nearby(self, position, fish_pop):
        """
        Detect fish density in 8 directions (like sonar sweep)
        Returns array of 8 values [0-1] indicating biomass in each direction
        """
        if fish_pop.num_schools == 0:
            return [0.0] * 8
        
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        densities = []
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Calculate vectors to all fish
            fish_vectors = fish_pop.positions - position
            distances = np.linalg.norm(fish_vectors, axis=1)
            
            # Filter by vision range
            in_range_mask = distances < self.vision_range
            if not np.any(in_range_mask):
                densities.append(0.0)
                continue
            
            # Check if fish are in this directional cone (±22.5° = 45° cone)
            fish_vectors_norm = fish_vectors[in_range_mask] / (distances[in_range_mask, None] + 1e-8)
            dots = np.dot(fish_vectors_norm, direction)
            in_cone_mask = dots > 0.92  # cos(22.5°)
            
            if np.any(in_cone_mask):
                # Sum biomass in this cone
                fish_indices = np.where(in_range_mask)[0][in_cone_mask]
                total_biomass = np.sum(fish_pop.energies[fish_indices])
                densities.append(min(total_biomass / 1000.0, 1.0))  # Normalized
            else:
                densities.append(0.0)
        
        return densities
    
    def _sense_temperature_gradient(self, position, env):
        """
        Sample temperature in 8 directions to identify optimal fishing zones
        Returns how much more fish-suitable each direction is
        """
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        gradients = []
        
        current_temp = env.get_temperature(*position)
        current_suitability = self._temperature_suitability(current_temp)
        
        for angle in angles:
            # Probe temperature in this direction
            probe_x = np.clip(
                position[0] + np.cos(angle) * self.temp_sensor_range,
                0, self.env_width - 1
            )
            probe_y = np.clip(
                position[1] + np.sin(angle) * self.temp_sensor_range,
                0, self.env_height - 1
            )
            
            probe_temp = env.get_temperature(probe_x, probe_y)
            probe_suitability = self._temperature_suitability(probe_temp)
            
            # Positive gradient = better for fish in that direction
            gradient = probe_suitability - current_suitability
            gradients.append(np.clip(gradient, -1.0, 1.0))
        
        return gradients
    
    def _temperature_suitability(self, temp):
        """Calculate how suitable temperature is for fish [0-1]"""
        if self.fish_temp_min <= temp <= self.fish_temp_max:
            # Inside comfort zone - higher score closer to optimal
            deviation = abs(temp - self.fish_temp_optimal)
            return 1.0 - (deviation / 8.0)
        else:
            # Outside comfort zone - score degrades with distance
            if temp < self.fish_temp_min:
                return max(0.0, 1.0 - (self.fish_temp_min - temp) / 10.0)
            else:
                return max(0.0, 1.0 - (temp - self.fish_temp_max) / 10.0)
    
    def _fuel_sufficient_for_return(self, boat_state, home_distance):
        """Rule-based calculation: Can we make it home?"""
        # Conservative fuel estimate: 50 L/unit + 20% safety margin
        fuel_needed = home_distance * 50.0 * 1.2
        return 1.0 if boat_state['fuel'] >= fuel_needed else 0.0
    
    def select_action(self, observation, boat_state):
        """
        HYBRID DECISION SYSTEM: Rule-based safety + RL strategy
        
        Actions returned: [heading_change, throttle, net_deploy]
        - heading_change: -0.5 to +0.5 radians per step
        - throttle: 0.0 to 1.0
        - net_deploy: 0.0 (off) or 1.0 (on)
        """
        
        # ========================================
        # RULE-BASED OVERRIDES (Safety First)
        # ========================================
        
        # 1. EMERGENCY: Out of fuel
        if boat_state['fuel'] <= 0:
            return np.array([0.0, 0.0, 0.0])  # Drift with current
        
        # 2. AUTO-PILOT HOME MODE (Explicit Navigation)
        should_return = self._check_return_home_condition(boat_state)
        if should_return:
            return self._navigate_to_port(boat_state)
        
        # ========================================
        # RL-BASED FISHING STRATEGY
        # ========================================
        
        if np.random.random() < self.epsilon:
            # EXPLORATION: Random valid actions
            heading_change = np.random.uniform(-0.3, 0.3)
            throttle = np.random.uniform(0.4, 1.0)
            net_deploy = np.random.choice([0.0, 1.0], p=[0.3, 0.7])  # 70% chance to deploy during exploration
        else:
            # EXPLOITATION: Use learned policy
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(obs_tensor).squeeze()
            
            # Map Q-values to actions
            heading_change = torch.tanh(q_values[0]).item() * 0.5  # ±0.5 radians
            throttle = torch.sigmoid(q_values[1]).item()  # 0 to 1
            net_deploy = 1.0 if q_values[2].item() > 0 else 0.0
        
        # Smart net control (rule-based safety overrides)
        net_deploy = self._smart_net_control(boat_state, observation, net_deploy)
        
        return np.array([heading_change, throttle, net_deploy])
    
    def _check_return_home_condition(self, boat_state):
        """
        RULE: Determine if boat must return to port
        Any of these triggers immediate return:
        1. Cargo nearly full (95%+)
        2. Fuel insufficient for safe return
        3. Critical fuel level (<15%)
        """
        home_distance = np.linalg.norm(boat_state['position'] - self.home_port)
        
        # Calculate fuel needed with safety margin
        fuel_to_home = home_distance * 50.0 * 1.3  # 30% safety buffer
        
        cargo_full = boat_state['cargo'] >= boat_state['max_cargo'] * 0.95
        fuel_insufficient = boat_state['fuel'] < fuel_to_home
        fuel_critical = boat_state['fuel'] < boat_state['max_fuel'] * 0.15
        
        return cargo_full or fuel_insufficient or fuel_critical
    
    def _navigate_to_port(self, boat_state):
        """
        RULE-BASED NAVIGATION: Direct path to home port
        This is NOT learned - we always know how to go home!
        """
        home_vector = self.home_port - boat_state['position']
        home_distance = np.linalg.norm(home_vector)
        
        # Calculate target heading
        target_heading = np.arctan2(home_vector[1], home_vector[0])
        heading_diff = self._angle_difference(target_heading, boat_state['heading'])
        
        if home_distance < 2.0:
            # Approaching port - slow down for docking
            return np.array([heading_diff * 0.6, 0.3, 0.0])
        else:
            # En route - full speed ahead
            throttle = min(1.0, boat_state['fuel'] / (home_distance * 30.0))  # Fuel-aware throttle
            return np.array([heading_diff, throttle, 0.0])
    
    def _angle_difference(self, target, current):
        """Calculate shortest angular path"""
        diff = target - current
        # Normalize to [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return np.clip(diff, -0.5, 0.5)  # Limit turning rate
    
    def _smart_net_control(self, boat_state, observation, rl_decision):
        """
        RULE-BASED NET SAFETY SYSTEM
        Overrides RL decision if deployment would be inefficient/dangerous
        """
        # Condition 1: Cargo full - no point catching more
        if boat_state['cargo'] >= boat_state['max_cargo'] * 0.98:
            return 0.0
        
        # Condition 2: Speed too high - net damage risk
        speed = np.linalg.norm(boat_state['velocity'])
        if speed > 4.0:  # Raised from 3.5 to allow more deployment
            return 0.0
        
        # Condition 3: No fish detected - wasteful drag (RELAXED)
        fish_density = observation[11:19]  # Fish sensor readings
        if np.max(fish_density) < 0.02:  # Lowered from 0.05 - deploy even with weak signals
            return 0.0
        
        # Condition 4: Temperature unsuitable - fish unlikely (RELAXED)
        current_temp_normalized = observation[10]
        current_temp = current_temp_normalized * 10.0 + 15.0
        if current_temp < self.fish_temp_min - 4.0 or current_temp > self.fish_temp_max + 4.0:  # Wider tolerance
            return 0.0
        
        # Condition 5: Fuel too low - minimize drag to get home
        if boat_state['fuel'] < boat_state['max_fuel'] * 0.15:  # Lowered from 0.2
            return 0.0
        
        # All safety checks passed - trust RL decision
        return rl_decision
    
    def calculate_reward(self, prev_state, action, new_state, just_sold, cargo_sold):
        """
        REWARD ENGINEERING: Shape agent behavior towards profitability
        
        Positive rewards:
        - Successful fish sale (MAJOR)
        - Catching fish (incremental progress)
        
        Negative rewards:
        - Fuel consumption (efficiency pressure)
        - Time cost (faster trips = better)
        - Empty net drag (inefficiency penalty)
        - Running out of fuel (MAJOR penalty)
        """
        reward = 0.0
        
        # 1. JACKPOT: Successful sale at port
        if just_sold and cargo_sold > 0:
            revenue = cargo_sold * self.fish_price_per_ton
            trip_profit = revenue / 1000.0  # Scale for NN stability
            reward += trip_profit
            
            # Efficiency bonus: High profit with low fuel use
            fuel_used_this_trip = prev_state.get('trip_fuel', 0)
            if fuel_used_this_trip > 0:
                efficiency = revenue / (fuel_used_this_trip * self.fuel_cost_per_liter)
                reward += efficiency * 10.0
            
            self.trips_completed += 1
            self.total_catch += cargo_sold
        
        # 2. FUEL COST: Continuous pressure for efficiency
        fuel_used = max(0, prev_state['fuel'] - new_state['fuel'])
        fuel_cost = fuel_used * self.fuel_cost_per_liter / 1000.0
        reward -= fuel_cost
        
        # 3. TIME COST: Encourage decisive action (reduced to be less punishing)
        reward -= 0.02
        
        # 4. PROGRESS: Incremental reward for catching fish (INCREASED)
        cargo_gained = new_state['cargo'] - prev_state['cargo']
        if cargo_gained > 0:
            reward += cargo_gained * 2.0  # Increased from 0.8 - reward catching more!
        
        # 4b. BONUS: Reward for having cargo (encourages keeping fish)
        if new_state['cargo'] > 0:
            reward += new_state['cargo'] * 0.01  # Small reward for holding fish
        
        # 5. PENALTY: Empty net drag (wasteful)
        if action[2] > 0.5 and cargo_gained < 0.5:
            reward -= 1.0
        
        # 6. PENALTY: Running out of fuel (catastrophic failure)
        if new_state['fuel'] <= 0 and prev_state['fuel'] > 0:
            reward -= 200.0  # SEVERE - boat is dead in water, mission failure
        
        # 6b. WARNING PENALTY: Low fuel far from port (risky situation)
        fuel_pct = new_state['fuel'] / new_state['max_fuel']
        pos = np.array(new_state['position'])
        dist_to_port = np.linalg.norm(self.home_port - pos)
        if fuel_pct < 0.2 and dist_to_port > 30:
            reward -= 10.0  # Dangerous situation - might not make it back
        
        # 7. PENALTY: Marine Protected Area violations (CRITICAL)
        inside_mpa = new_state.get('inside_mpa', False)
        if inside_mpa:
            # Lighter penalty for just being in MPA (fuel waste, opportunity cost)
            reward -= 2.0
            
            # HEAVY PENALTY for illegal fishing in MPA
            if action[2] > 0.5:  # Net deployed
                reward -= 50.0  # Major violation - ecological damage
        
        # 8. PENALTY: Wandering too far from viable fishing areas
        temp_suitability = self._temperature_suitability(
            new_state.get('current_temp', 18.0)
        )
        if temp_suitability < 0.2:
            reward -= 0.5  # Discourage staying in dead zones
        
        self.total_reward += reward
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """
        Single training step using Deep Q-Learning with experience replay
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random minibatch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Compute current Q-values
        current_q_values = self.policy_net(states)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Loss: TD error on first Q-value (simplified DQN)
        loss = nn.MSELoss()(current_q_values[:, 0], target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Periodically sync target network (every N episodes)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_checkpoint(self, filepath):
        """Save agent for later training/deployment"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'trips_completed': self.trips_completed,
            'total_catch': self.total_catch,
            'memory': list(self.memory)
        }, filepath)
        print(f"Agent {self.boat_id} saved: {self.trips_completed} trips, {self.total_catch:.1f} tons caught")
    
    def load_checkpoint(self, filepath):
        """Load trained agent"""
        checkpoint = torch.load(filepath, weights_only=False)  # PyTorch 2.6 compatibility
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_reward = checkpoint['total_reward']
        self.trips_completed = checkpoint['trips_completed']
        self.total_catch = checkpoint['total_catch']
        if 'memory' in checkpoint:
            self.memory = deque(checkpoint['memory'], maxlen=10000)
        print(f"Agent {self.boat_id} loaded: {self.trips_completed} trips, {self.total_catch:.1f} tons caught")
    
    def get_stats(self):
        """Return agent performance statistics"""
        return {
            'boat_id': self.boat_id,
            'trips_completed': self.trips_completed,
            'total_catch': self.total_catch,
            'total_reward': self.total_reward,
            'epsilon': self.epsilon,
            'avg_catch_per_trip': self.total_catch / max(1, self.trips_completed)
        }
