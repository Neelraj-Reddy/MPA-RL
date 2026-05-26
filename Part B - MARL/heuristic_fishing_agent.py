import numpy as np


class HeuristicFishingAgent:
    """
    Training-free rule-based fishing controller.

    Designed to plug into the same fleet/environment pipeline used by RL agents.
    """

    def __init__(self, boat_id, home_port, env_width, env_height):
        self.boat_id = boat_id
        self.home_port = np.array(home_port, dtype=float)
        self.env_width = env_width
        self.env_height = env_height

        # Keep same domain assumptions as the RL agent for compatibility.
        self.fish_temp_min = 14.0
        self.fish_temp_max = 22.0
        self.fish_temp_optimal = 18.0
        self.vision_range = 12.0

        # Economic constants reused by reward accounting in deployment stats.
        self.fish_price_per_ton = 1500.0
        self.fuel_cost_per_liter = 1.2

        # Internal navigation memory.
        self.patrol_phase = np.random.uniform(0.0, 2.0 * np.pi)
        self.patrol_radius = 8.0 + (boat_id % 5) * 2.5

        self.total_reward = 0.0
        self.trips_completed = 0
        self.total_catch = 0.0

        # Filled during get_observation for use by select_action.
        self._latest_context = {}

    def get_observation(self, boat_state, fish_pop, env):
        """Collect local signals for rule-based control and return a compact vector."""
        pos = np.array(boat_state['position'], dtype=float)

        fish_signal = self._nearest_fish_signal(pos, fish_pop)
        temp = env.get_temperature(*pos)
        temp_suitability = self._temperature_suitability(temp)

        home_vec = self.home_port - pos
        home_dist = np.linalg.norm(home_vec)
        home_dir = home_vec / (home_dist + 1e-8)

        self._latest_context = {
            'nearest_fish_vector': fish_signal['vector'],
            'nearest_fish_distance': fish_signal['distance'],
            'nearest_fish_energy': fish_signal['energy'],
            'temp': temp,
            'temp_suitability': temp_suitability,
            'home_distance': home_dist,
            'home_direction': home_dir,
            'inside_mpa': bool(boat_state.get('inside_mpa', False)),
        }

        return np.array([
            pos[0] / max(1.0, self.env_width),
            pos[1] / max(1.0, self.env_height),
            boat_state['fuel'] / max(1.0, boat_state['max_fuel']),
            boat_state['cargo'] / max(1.0, boat_state['max_cargo']),
            temp_suitability,
            fish_signal['distance_norm'],
            fish_signal['bearing_x'],
            fish_signal['bearing_y'],
            float(boat_state.get('inside_mpa', False)),
        ], dtype=np.float32)

    def select_action(self, observation, boat_state):
        """Return action [heading_change, throttle, net_deploy]."""
        if boat_state['fuel'] <= 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        pos = np.array(boat_state['position'], dtype=float)
        heading = boat_state['heading']

        # 1) Hard safety return logic.
        if self._should_return_to_port(boat_state):
            return self._navigate_to_port(pos, heading, boat_state)

        # 2) MPA compliance override: exit quickly and never fish inside MPA.
        if self._latest_context.get('inside_mpa', False):
            action = self._exit_mpa_action(pos, heading)
            action[2] = 0.0
            return action

        nearest_fish_distance = self._latest_context.get('nearest_fish_distance', np.inf)
        nearest_fish_vector = self._latest_context.get('nearest_fish_vector', np.zeros(2, dtype=float))
        temp_suitability = self._latest_context.get('temp_suitability', 0.0)

        # 3) Primary fishing behavior: chase nearest fish cluster if viable.
        can_fish = (
            nearest_fish_distance < self.vision_range
            and temp_suitability > 0.35
            and boat_state['cargo'] < boat_state['max_cargo'] * 0.95
        )

        if can_fish:
            target_heading = np.arctan2(nearest_fish_vector[1], nearest_fish_vector[0])
            heading_change = self._angle_difference(target_heading, heading)

            # Stable trawling speed in catch-optimal range.
            throttle = 0.55 if nearest_fish_distance < 4.0 else 0.75
            net_deploy = 1.0

            if boat_state['fuel'] < boat_state['max_fuel'] * 0.20:
                throttle = min(throttle, 0.55)
                net_deploy = 0.0

            return np.array([heading_change, throttle, net_deploy], dtype=np.float32)

        # 4) Patrol/search strategy around home port when fish not detected.
        return self._patrol_action(pos, heading, boat_state)

    def calculate_reward(self, prev_state, action, new_state, just_sold, cargo_sold):
        """Reward used for deployment diagnostics and summaries."""
        reward = 0.0

        if just_sold and cargo_sold > 0:
            revenue = cargo_sold * self.fish_price_per_ton
            reward += revenue / 1000.0
            self.trips_completed += 1
            self.total_catch += cargo_sold

        fuel_used = max(0.0, prev_state['fuel'] - new_state['fuel'])
        reward -= (fuel_used * self.fuel_cost_per_liter) / 1000.0
        reward -= 0.02

        cargo_gain = new_state['cargo'] - prev_state['cargo']
        if cargo_gain > 0:
            reward += cargo_gain * 1.5

        if new_state.get('inside_mpa', False) and action[2] > 0.5:
            reward -= 50.0

        if new_state['fuel'] <= 0 < prev_state['fuel']:
            reward -= 200.0

        self.total_reward += reward
        return reward

    def get_stats(self):
        return {
            'boat_id': self.boat_id,
            'trips_completed': self.trips_completed,
            'total_catch': self.total_catch,
            'total_reward': self.total_reward,
            'epsilon': 0.0,
            'avg_catch_per_trip': self.total_catch / max(1, self.trips_completed)
        }

    def _nearest_fish_signal(self, pos, fish_pop):
        if fish_pop.num_schools == 0:
            return {
                'vector': np.zeros(2, dtype=float),
                'distance': np.inf,
                'distance_norm': 1.0,
                'bearing_x': 0.0,
                'bearing_y': 0.0,
                'energy': 0.0,
            }

        vectors = fish_pop.positions - pos
        distances = np.linalg.norm(vectors, axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_idx])
        nearest_vector = vectors[nearest_idx]

        norm = np.linalg.norm(nearest_vector)
        if norm > 1e-8:
            bearing = nearest_vector / norm
        else:
            bearing = np.zeros(2, dtype=float)

        return {
            'vector': nearest_vector,
            'distance': nearest_distance,
            'distance_norm': float(np.clip(nearest_distance / 25.0, 0.0, 1.0)),
            'bearing_x': float(bearing[0]),
            'bearing_y': float(bearing[1]),
            'energy': float(fish_pop.energies[nearest_idx])
        }

    def _should_return_to_port(self, boat_state):
        pos = np.array(boat_state['position'], dtype=float)
        home_distance = np.linalg.norm(self.home_port - pos)
        fuel_needed = home_distance * 50.0 * 1.25

        cargo_high = boat_state['cargo'] >= boat_state['max_cargo'] * 0.90
        fuel_low = boat_state['fuel'] < boat_state['max_fuel'] * 0.18
        fuel_risky = boat_state['fuel'] < fuel_needed

        return cargo_high or fuel_low or fuel_risky

    def _navigate_to_port(self, pos, heading, boat_state):
        to_home = self.home_port - pos
        distance = np.linalg.norm(to_home)
        target_heading = np.arctan2(to_home[1], to_home[0])
        heading_change = self._angle_difference(target_heading, heading)

        if distance < 2.0:
            throttle = 0.25
        elif distance < 8.0:
            throttle = 0.45
        else:
            throttle = 0.90 if boat_state['fuel'] > boat_state['max_fuel'] * 0.30 else 0.65

        return np.array([heading_change, throttle, 0.0], dtype=np.float32)

    def _exit_mpa_action(self, pos, heading):
        away = pos - self.home_port
        if np.linalg.norm(away) < 1e-6:
            away = np.array([np.cos(self.patrol_phase), np.sin(self.patrol_phase)])
        target_heading = np.arctan2(away[1], away[0])
        heading_change = self._angle_difference(target_heading, heading)
        return np.array([heading_change, 0.80, 0.0], dtype=np.float32)

    def _patrol_action(self, pos, heading, boat_state):
        self.patrol_phase = (self.patrol_phase + 0.08 + 0.02 * (self.boat_id % 3)) % (2 * np.pi)

        waypoint = self.home_port + np.array([
            np.cos(self.patrol_phase) * self.patrol_radius,
            np.sin(self.patrol_phase) * self.patrol_radius,
        ])
        waypoint[0] = np.clip(waypoint[0], 2.0, self.env_width - 2.0)
        waypoint[1] = np.clip(waypoint[1], 2.0, self.env_height - 2.0)

        to_waypoint = waypoint - pos
        target_heading = np.arctan2(to_waypoint[1], to_waypoint[0])
        heading_change = self._angle_difference(target_heading, heading)

        fuel_ratio = boat_state['fuel'] / max(1.0, boat_state['max_fuel'])
        throttle = 0.60 if fuel_ratio > 0.30 else 0.45

        return np.array([heading_change, throttle, 0.0], dtype=np.float32)

    def _temperature_suitability(self, temp):
        if self.fish_temp_min <= temp <= self.fish_temp_max:
            return 1.0 - abs(temp - self.fish_temp_optimal) / 8.0
        if temp < self.fish_temp_min:
            return max(0.0, 1.0 - (self.fish_temp_min - temp) / 10.0)
        return max(0.0, 1.0 - (temp - self.fish_temp_max) / 10.0)

    def _angle_difference(self, target, current):
        diff = target - current
        while diff > np.pi:
            diff -= 2.0 * np.pi
        while diff < -np.pi:
            diff += 2.0 * np.pi
        return float(np.clip(diff, -0.5, 0.5))