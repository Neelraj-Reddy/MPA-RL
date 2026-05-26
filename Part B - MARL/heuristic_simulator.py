import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import imageio

from environment import OceanEnvironment
from fish_ecosystem import FishPopulation
from fleet_physics import FishingFleet
from heuristic_fishing_agent import HeuristicFishingAgent


class HeuristicSimulation:
    def __init__(
        self,
        env_width=100,
        env_height=100,
        hours_per_tick=1,
        initial_fish=2500,
        num_ports=5,
        boats_per_port=None,
        port_locations=None,
        simulation_days=365.0,
        years=None,
        mpa_enabled=True,
        seed=None,
        disable_fishing=False,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.env_width = int(env_width)
        self.env_height = int(env_height)
        self.hours_per_tick = int(hours_per_tick)
        self.initial_fish = int(initial_fish)
        self.num_ports = int(num_ports)
        if years is not None:
            # Backward compatibility for older callers.
            simulation_days = float(years) * 365.0
        self.simulation_days = float(simulation_days)
        self.total_steps = max(1, int(self.simulation_days * 24 / self.hours_per_tick))
        self.mpa_enabled = bool(mpa_enabled)
        self.disable_fishing = bool(disable_fishing)

        if port_locations is not None and len(port_locations) >= self.num_ports:
            ports = np.array(port_locations[:self.num_ports], dtype=float)
            ports[:, 0] = np.clip(ports[:, 0], 0.0, self.env_width - 0.1)
            ports[:, 1] = np.clip(ports[:, 1], 0.0, self.env_height - 0.1)
            self.ports = ports
        else:
            self.ports = self._generate_random_ports(self.num_ports, self.env_width, self.env_height)

        if boats_per_port is None:
            boats_per_port = [3] * self.num_ports
        boats_per_port = [int(max(0, b)) for b in boats_per_port]
        if len(boats_per_port) < self.num_ports:
            boats_per_port.extend([0] * (self.num_ports - len(boats_per_port)))
        self.boats_per_port = boats_per_port[: self.num_ports]
        self.total_boats = int(sum(self.boats_per_port))

        self.env = OceanEnvironment(
            width=self.env_width,
            height=self.env_height,
            hours_per_tick=self.hours_per_tick,
        )

        if not self.mpa_enabled:
            self.env.mpa_coverage_target = 0.0
            self.env.mpa_persistence = 0.0
            self.env.mpa_grid[:] = 0.0

            def no_mpa_update(_fish_population):
                return

            self.env.update_mpas = no_mpa_update

        self.fish = FishPopulation(
            initial_population=self.initial_fish,
            env_width=self.env_width,
            env_height=self.env_height,
        )

        self.fleet = FishingFleet(
            num_boats=self.total_boats,
            ports=self.ports,
            env_width=self.env_width,
            env_height=self.env_height,
        )

        self.agents = []
        self.boat_home_port_index = []

        boat_idx = 0
        for port_idx, count in enumerate(self.boats_per_port):
            for _ in range(count):
                agent = HeuristicFishingAgent(
                    boat_id=boat_idx,
                    home_port=self.ports[port_idx],
                    env_width=self.env_width,
                    env_height=self.env_height,
                )
                self.agents.append(agent)
                self.boat_home_port_index.append(port_idx)
                boat_idx += 1

        self._reset_fleet_positions_near_ports()

        colors = ['#000033', '#001a66', '#0055cc', '#00a3a3', '#7cc244', '#ffd24d', '#ff6a3d']
        self.cmap = LinearSegmentedColormap.from_list('ocean_thermal', colors, N=200)

        self.metrics = {
            'step': [],
            'day': [],
            'fish_schools': [],
            'fish_school_delta_prev': [],
            'fish_school_change_from_start': [],
            'fish_biomass_total': [],
            'fish_biomass_change_from_start': [],
            'avg_fish_school_energy': [],
            'total_energy': [],
            'captured_biomass': [],
            'plankton_total': [],
            'fleet_avg_fuel': [],
            'fleet_avg_cargo': [],
            'cumulative_catch': [],
            'boats_out_of_fuel': [],
            'inside_mpa_boats': [],
            'boat_fuel_by_step': [],
            'boat_cargo_by_step': [],
            'boat_step_catch_by_step': [],
            'boat_cum_catch_by_step': [],
        }

        self.port_catch = np.zeros(self.num_ports, dtype=float)
        self.boat_catch = np.zeros(self.total_boats, dtype=float)

        # Interactive stepping state (used by frame-by-frame UI mode).
        self.interactive_step = 0
        self.interactive_cumulative_catch = 0.0
        self.interactive_prev_states = [self.get_boat_state(i) for i in range(self.total_boats)]
        self.interactive_mpa_start_step = 0
        self.interactive_mpa_end_step = self.total_steps - 1
        self.interactive_daily_catch = []

    def _generate_random_ports(self, num_ports, width, height):
        ports = []
        min_spacing = 20.0
        margin = 10.0

        for _ in range(num_ports):
            for _attempt in range(200):
                candidate = np.array([
                    np.random.uniform(margin, width - margin),
                    np.random.uniform(margin, height - margin),
                ])
                if len(ports) == 0:
                    ports.append(candidate)
                    break
                min_dist = min(np.linalg.norm(candidate - p) for p in ports)
                if min_dist >= min_spacing:
                    ports.append(candidate)
                    break

        if len(ports) < num_ports:
            while len(ports) < num_ports:
                ports.append(np.array([width * 0.5, height * 0.5]))

        return np.array(ports)

    def _reset_fleet_positions_near_ports(self):
        for i, port_idx in enumerate(self.boat_home_port_index):
            self.fleet.positions[i] = self.ports[port_idx] + (np.random.rand(2) - 0.5) * 2.5
            self.fleet.positions[i, 0] = np.clip(self.fleet.positions[i, 0], 0, self.env_width - 0.1)
            self.fleet.positions[i, 1] = np.clip(self.fleet.positions[i, 1], 0, self.env_height - 0.1)
            self.fleet.fuel_levels[i] = self.fleet.max_fuel
            self.fleet.cargo_levels[i] = 0.0
            self.fleet.velocities[i] = 0.0
            self.fleet.headings[i] = np.random.rand() * 2 * np.pi

    def get_boat_state(self, boat_idx):
        return {
            'position': self.fleet.positions[boat_idx].copy(),
            'velocity': np.array([
                np.cos(self.fleet.headings[boat_idx]) * self.fleet.velocities[boat_idx],
                np.sin(self.fleet.headings[boat_idx]) * self.fleet.velocities[boat_idx],
            ]),
            'heading': self.fleet.headings[boat_idx],
            'fuel': self.fleet.fuel_levels[boat_idx],
            'max_fuel': self.fleet.max_fuel,
            'cargo': self.fleet.cargo_levels[boat_idx],
            'max_cargo': self.fleet.max_cargo,
            'net_deployed': self.fleet.nets_deployed[boat_idx],
            'current_temp': self.env.get_temperature(*self.fleet.positions[boat_idx]),
            'inside_mpa': self.env.is_in_mpa(*self.fleet.positions[boat_idx]),
        }

    def run(
        self,
        capture_start_day=0.0,
        capture_end_day=None,
        target_frames=400,
        fps=12,
        media_format='mp4',
        mpa_start_day=0.0,
        mpa_end_day=None,
        highlight_boat=None,
        focus_port=None,
        metrics_interval_steps=24,
        output_dir='outputs',
        run_name=None,
        progress_callback=None,
    ):
        os.makedirs(output_dir, exist_ok=True)

        if capture_end_day is None:
            capture_end_day = self.simulation_days

        capture_start_step = int(max(0.0, capture_start_day * 24 / self.hours_per_tick))
        capture_end_step = int(min(self.total_steps - 1, capture_end_day * 24 / self.hours_per_tick))
        if capture_end_step < capture_start_step:
            capture_end_step = capture_start_step

        if mpa_end_day is None:
            mpa_end_day = self.simulation_days
        mpa_start_step = int(max(0.0, mpa_start_day * 24 / self.hours_per_tick))
        mpa_end_step = int(min(self.total_steps - 1, mpa_end_day * 24 / self.hours_per_tick))
        if mpa_end_step < mpa_start_step:
            mpa_end_step = mpa_start_step

        capture_span = max(1, capture_end_step - capture_start_step + 1)
        target_frames = int(max(10, target_frames))
        frame_stride = max(1, capture_span // target_frames)

        focus_mask = None
        if focus_port is not None and 0 <= int(focus_port) < self.num_ports:
            focus_mask = np.array([p == int(focus_port) for p in self.boat_home_port_index], dtype=bool)

        if run_name is None:
            run_name = f'fleet_run_{int(time.time())}'

        mp4_path = os.path.join(output_dir, f'{run_name}.mp4')
        gif_path = os.path.join(output_dir, f'{run_name}.gif')

        requested_format = str(media_format).lower().strip()
        if requested_format not in ('mp4', 'gif'):
            requested_format = 'mp4'

        media_type = 'video/mp4' if requested_format == 'mp4' else 'image/gif'
        media_path = mp4_path if requested_format == 'mp4' else gif_path

        writer = None
        fallback_used = False
        fallback_message = ''

        try:
            if requested_format == 'mp4':
                writer = imageio.get_writer(
                    mp4_path,
                    fps=max(2, int(fps)),
                    codec='libx264',
                    macro_block_size=None,
                )
            else:
                writer = imageio.get_writer(
                    gif_path,
                    mode='I',
                    fps=max(2, int(fps)),
                )
        except Exception as e:
            writer = imageio.get_writer(
                gif_path,
                mode='I',
                fps=max(2, int(fps)),
            )
            media_type = 'image/gif'
            media_path = gif_path
            fallback_used = True
            fallback_message = f'MP4 unavailable, fell back to GIF: {e}'

        captured_frames = 0

        prev_states = [self.get_boat_state(i) for i in range(self.total_boats)]
        cumulative_catch = 0.0
        prev_fish_schools = int(self.fish.num_schools)
        initial_fish_biomass = float(np.sum(self.fish.energies)) if self.fish.num_schools > 0 else 0.0
        interval_boat_catch = np.zeros(self.total_boats, dtype=float)

        for step in range(self.total_steps):
            actions = []
            for boat_idx in range(self.total_boats):
                boat_state = self.get_boat_state(boat_idx)
                obs = self.agents[boat_idx].get_observation(boat_state, self.fish, self.env)
                action = self.agents[boat_idx].select_action(obs, boat_state)
                if self.disable_fishing:
                    # Disable fishing by setting net_deployed to 0
                    action = (action[0], action[1], 0)
                actions.append(action)

            self.fleet.step(np.array(actions), self.fish, self.env)
            self.fish.step(self.env)
            self.env.step()

            mpa_active_now = self.mpa_enabled and (mpa_start_step <= step <= mpa_end_step)
            if mpa_active_now:
                self.env.update_mpas(self.fish)
            else:
                # Keep ocean unprotected outside configured MPA window.
                self.env.mpa_grid[:] = 0.0

            for boat_idx in range(self.total_boats):
                new_state = self.get_boat_state(boat_idx)
                sold = float(self.fleet.cargo_sold[boat_idx])
                if sold > 0:
                    home_port_idx = self.boat_home_port_index[boat_idx]
                    self.port_catch[home_port_idx] += sold
                    self.boat_catch[boat_idx] += sold
                    interval_boat_catch[boat_idx] += sold
                    cumulative_catch += sold

                self.agents[boat_idx].calculate_reward(
                    prev_states[boat_idx],
                    actions[boat_idx],
                    new_state,
                    bool(self.fleet.just_sold[boat_idx]),
                    sold,
                )
                prev_states[boat_idx] = new_state

            if step % max(1, int(metrics_interval_steps)) == 0 or step == self.total_steps - 1:
                inside_mpa = 0
                for b in range(self.total_boats):
                    if self.env.is_in_mpa(*self.fleet.positions[b]):
                        inside_mpa += 1

                self.metrics['step'].append(step)
                self.metrics['day'].append(step * self.hours_per_tick / 24.0)
                fish_schools_now = int(self.fish.num_schools)
                self.metrics['fish_schools'].append(fish_schools_now)
                self.metrics['fish_school_delta_prev'].append(fish_schools_now - prev_fish_schools)
                self.metrics['fish_school_change_from_start'].append(fish_schools_now - self.initial_fish)

                fish_biomass_now = float(np.sum(self.fish.energies)) if self.fish.num_schools > 0 else 0.0
                plankton_total_now = float(np.sum(self.env.plankton_grid))
                avg_fish_school_energy = float(np.mean(self.fish.energies)) if self.fish.num_schools > 0 else 0.0
                captured_biomass_now = float(np.sum(self.boat_catch) + np.sum(self.fleet.cargo_levels))

                self.metrics['fish_biomass_total'].append(fish_biomass_now)
                self.metrics['fish_biomass_change_from_start'].append(fish_biomass_now - initial_fish_biomass)
                self.metrics['avg_fish_school_energy'].append(avg_fish_school_energy)
                self.metrics['total_energy'].append(fish_biomass_now + plankton_total_now)
                self.metrics['captured_biomass'].append(captured_biomass_now)

                self.metrics['plankton_total'].append(plankton_total_now)
                self.metrics['fleet_avg_fuel'].append(float(np.mean(self.fleet.fuel_levels)))
                self.metrics['fleet_avg_cargo'].append(float(np.mean(self.fleet.cargo_levels)))
                self.metrics['cumulative_catch'].append(float(cumulative_catch))
                self.metrics['boats_out_of_fuel'].append(int(np.sum(self.fleet.fuel_levels <= 0)))
                self.metrics['inside_mpa_boats'].append(int(inside_mpa))
                self.metrics['boat_fuel_by_step'].append(self.fleet.fuel_levels.astype(float).tolist())
                self.metrics['boat_cargo_by_step'].append(self.fleet.cargo_levels.astype(float).tolist())
                self.metrics['boat_step_catch_by_step'].append(interval_boat_catch.astype(float).tolist())
                self.metrics['boat_cum_catch_by_step'].append(self.boat_catch.astype(float).tolist())

                prev_fish_schools = fish_schools_now
                interval_boat_catch[:] = 0.0

            if capture_start_step <= step <= capture_end_step:
                should_capture = ((step - capture_start_step) % frame_stride == 0) or (step == capture_end_step)
                if should_capture:
                    frame = self._render_frame(step, focus_mask, highlight_boat=highlight_boat)
                    writer.append_data(np.asarray(frame))
                    captured_frames += 1

            if progress_callback is not None:
                progress_callback(step + 1, self.total_steps)

        if captured_frames == 0:
            writer.append_data(np.asarray(self._render_frame(self.total_steps - 1, focus_mask, highlight_boat=highlight_boat)))
            captured_frames = 1

        writer.close()

        port_report = []
        for p_idx in range(self.num_ports):
            mask = np.array([h == p_idx for h in self.boat_home_port_index], dtype=bool)
            boats_here = int(np.sum(mask))
            avg_fuel_here = float(np.mean(self.fleet.fuel_levels[mask])) if boats_here > 0 else 0.0
            out_of_fuel_here = int(np.sum(self.fleet.fuel_levels[mask] <= 0)) if boats_here > 0 else 0
            port_report.append({
                'port_index': p_idx,
                'boats': boats_here,
                'total_catch': float(self.port_catch[p_idx]),
                'avg_catch_per_boat': float(self.port_catch[p_idx] / boats_here) if boats_here > 0 else 0.0,
                'avg_fuel_final': avg_fuel_here,
                'boats_out_of_fuel': out_of_fuel_here,
            })

        return {
            'media_path': media_path,
            'media_type': media_type,
            'gif_path': media_path,
            'metrics': self.metrics,
            'port_report': port_report,
            'fallback_used': fallback_used,
            'fallback_message': fallback_message,
            'summary': {
                'total_steps': self.total_steps,
                'total_boats': self.total_boats,
                'final_fish_schools': int(self.fish.num_schools),
                'total_catch': float(np.sum(self.boat_catch)),
                'final_plankton_total': float(np.sum(self.env.plankton_grid)),
                'boats_out_of_fuel_final': int(np.sum(self.fleet.fuel_levels <= 0)),
                'capture_start_day': float(capture_start_day),
                'capture_end_day': float(capture_end_day),
                'mpa_start_day': float(mpa_start_day),
                'mpa_end_day': float(mpa_end_day),
                'captured_frames': int(captured_frames),
                'fps': int(max(2, int(fps))),
                'frame_stride': int(frame_stride),
            },
        }

    def _render_frame(self, step, focus_mask=None, highlight_boat=None):
        fig = plt.figure(figsize=(12, 7), dpi=90)
        ax = fig.add_subplot(111)

        ax.imshow(
            self.env.temperature_grid.T,
            origin='lower',
            cmap=self.cmap,
            extent=[0, self.env_width, 0, self.env_height],
            alpha=0.65,
            vmin=5,
            vmax=28,
        )

        if np.any(self.env.mpa_grid > 0.5):
            mpa_mask = self.env.mpa_grid > 0.5
            ax.imshow(
                mpa_mask.T,
                origin='lower',
                extent=[0, self.env_width, 0, self.env_height],
                cmap='Reds',
                alpha=0.25,
                vmin=0,
                vmax=2,
            )

        if self.fish.num_schools > 0:
            fish_sizes = np.clip(self.fish.energies / 9.0, 2.0, 70.0)
            ax.scatter(
                self.fish.positions[:, 0],
                self.fish.positions[:, 1],
                s=fish_sizes,
                c='#00e5ff',
                alpha=0.65,
                edgecolors='#0b3d91',
                linewidths=0.4,
            )

        for p in self.ports:
            ax.plot(p[0], p[1], marker='*', color='white', markersize=16, markeredgecolor='black')

        for b in range(self.total_boats):
            pos = self.fleet.positions[b]
            heading = self.fleet.headings[b]
            selected = True if focus_mask is None else bool(focus_mask[b])
            is_highlight = (highlight_boat is not None and int(highlight_boat) == b)

            if is_highlight:
                boat_color = '#ff1744'
                alpha = 1.0
                zorder = 8
            elif selected:
                boat_color = '#ffd54f'
                alpha = 0.95
                zorder = 5
            else:
                boat_color = '#9e9e9e'
                alpha = 0.45
                zorder = 3

            ax.plot(
                pos[0], pos[1],
                marker='o',
                color=boat_color,
                markersize=8,
                markeredgecolor='black',
                markeredgewidth=0.8,
                alpha=alpha,
                zorder=zorder,
            )

            dx = 3.2 * np.cos(heading)
            dy = 3.2 * np.sin(heading)
            ax.arrow(
                pos[0], pos[1], dx, dy,
                head_width=1.0,
                head_length=0.6,
                fc=boat_color,
                ec='black',
                linewidth=0.8,
                alpha=alpha,
                zorder=zorder,
            )

            if self.fleet.nets_deployed[b] and selected:
                circle = plt.Circle(pos, 2.6, color='#ff5252', alpha=0.16)
                ax.add_patch(circle)

            if is_highlight:
                # Strong ring marker to track chosen boat across steps.
                ring = plt.Circle(pos, 3.8, color='#ffffff', fill=False, linewidth=1.8, alpha=0.95)
                ax.add_patch(ring)

        day = step * self.hours_per_tick / 24.0
        ax.set_title(
            f'Fleet Simulation | Step {step} | Day {day:.1f} | Fish {self.fish.num_schools} | Boats {self.total_boats}',
            fontsize=11,
            fontweight='bold',
        )
        ax.set_xlim(0, self.env_width)
        ax.set_ylim(0, self.env_height)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(alpha=0.18)

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        frame = Image.fromarray(buf)
        plt.close(fig)
        return frame

    def start_interactive(self, mpa_start_day=0.0, mpa_end_day=None):
        """Initialize interactive stepping counters on top of current world state."""
        if mpa_end_day is None:
            mpa_end_day = self.simulation_days

        self.interactive_step = 0
        self.interactive_cumulative_catch = 0.0
        self.interactive_prev_states = [self.get_boat_state(i) for i in range(self.total_boats)]
        self.interactive_daily_catch = [0.0]
        self.interactive_mpa_start_step = int(max(0.0, mpa_start_day * 24 / self.hours_per_tick))
        self.interactive_mpa_end_step = int(min(self.total_steps - 1, mpa_end_day * 24 / self.hours_per_tick))
        if self.interactive_mpa_end_step < self.interactive_mpa_start_step:
            self.interactive_mpa_end_step = self.interactive_mpa_start_step

        # Seed visible MPA state at interactive start if the policy window is active.
        if self.mpa_enabled and (self.interactive_mpa_start_step <= 0 <= self.interactive_mpa_end_step):
            self.env.update_mpas(self.fish)
        else:
            self.env.mpa_grid[:] = 0.0

    def interactive_next_step(self):
        """Advance simulation by one step. Returns False if already at the end."""
        if self.interactive_step >= self.total_steps:
            return False

        step = self.interactive_step
        steps_per_day = max(1, int(round(24 / self.hours_per_tick)))

        actions = []
        for boat_idx in range(self.total_boats):
            boat_state = self.get_boat_state(boat_idx)
            obs = self.agents[boat_idx].get_observation(boat_state, self.fish, self.env)
            action = self.agents[boat_idx].select_action(obs, boat_state)
            if self.disable_fishing:
                # Disable fishing by setting net_deployed to 0
                action = (action[0], action[1], 0)
            actions.append(action)

        self.fleet.step(np.array(actions), self.fish, self.env)
        self.fish.step(self.env)
        self.env.step()

        mpa_active_now = self.mpa_enabled and (self.interactive_mpa_start_step <= step <= self.interactive_mpa_end_step)
        if mpa_active_now:
            self.env.update_mpas(self.fish)
        else:
            self.env.mpa_grid[:] = 0.0

        step_sold_total = 0.0
        for boat_idx in range(self.total_boats):
            new_state = self.get_boat_state(boat_idx)
            sold = float(self.fleet.cargo_sold[boat_idx])
            if sold > 0:
                home_port_idx = self.boat_home_port_index[boat_idx]
                self.port_catch[home_port_idx] += sold
                self.boat_catch[boat_idx] += sold
                self.interactive_cumulative_catch += sold
            step_sold_total += sold

            self.agents[boat_idx].calculate_reward(
                self.interactive_prev_states[boat_idx],
                actions[boat_idx],
                new_state,
                bool(self.fleet.just_sold[boat_idx]),
                sold,
            )
            self.interactive_prev_states[boat_idx] = new_state

        day_idx = step // steps_per_day
        while len(self.interactive_daily_catch) <= day_idx:
            self.interactive_daily_catch.append(0.0)
        self.interactive_daily_catch[day_idx] += step_sold_total

        self.interactive_step += 1
        return True

    def get_interactive_snapshot(self):
        """Capture full simulation state so UI can go backward instantly."""
        return {
            'interactive_step': int(self.interactive_step),
            'interactive_cumulative_catch': float(self.interactive_cumulative_catch),
            'interactive_mpa_start_step': int(self.interactive_mpa_start_step),
            'interactive_mpa_end_step': int(self.interactive_mpa_end_step),
            'interactive_daily_catch': self.interactive_daily_catch.copy(),
            'interactive_prev_states': [
                {
                    'position': s['position'].copy(),
                    'velocity': s['velocity'].copy(),
                    'heading': float(s['heading']),
                    'fuel': float(s['fuel']),
                    'max_fuel': float(s['max_fuel']),
                    'cargo': float(s['cargo']),
                    'max_cargo': float(s['max_cargo']),
                    'net_deployed': bool(s['net_deployed']),
                    'current_temp': float(s['current_temp']),
                    'inside_mpa': bool(s['inside_mpa']),
                }
                for s in self.interactive_prev_states
            ],
            'env': {
                'time_step': int(self.env.time_step),
                'plankton_grid': self.env.plankton_grid.copy(),
                'temperature_grid': self.env.temperature_grid.copy(),
                'mpa_grid': self.env.mpa_grid.copy(),
                'current_u': self.env.current_u.copy(),
                'current_v': self.env.current_v.copy(),
                'wind_u': self.env.wind_u.copy(),
                'wind_v': self.env.wind_v.copy(),
            },
            'fish': {
                'num_schools': int(self.fish.num_schools),
                'positions': self.fish.positions.copy(),
                'velocities': self.fish.velocities.copy(),
                'energies': self.fish.energies.copy(),
                'ages': self.fish.ages.copy(),
            },
            'fleet': {
                'positions': self.fleet.positions.copy(),
                'headings': self.fleet.headings.copy(),
                'velocities': self.fleet.velocities.copy(),
                'fuel_levels': self.fleet.fuel_levels.copy(),
                'cargo_levels': self.fleet.cargo_levels.copy(),
                'nets_deployed': self.fleet.nets_deployed.copy(),
                'just_sold': self.fleet.just_sold.copy(),
                'cargo_sold': self.fleet.cargo_sold.copy(),
            },
            'port_catch': self.port_catch.copy(),
            'boat_catch': self.boat_catch.copy(),
            'agent_stats': [
                {
                    'total_reward': float(a.total_reward),
                    'trips_completed': int(a.trips_completed),
                    'total_catch': float(a.total_catch),
                }
                for a in self.agents
            ],
        }

    def load_interactive_snapshot(self, snapshot):
        """Restore a previously captured simulation snapshot."""
        self.interactive_step = int(snapshot['interactive_step'])
        self.interactive_cumulative_catch = float(snapshot['interactive_cumulative_catch'])
        self.interactive_mpa_start_step = int(snapshot['interactive_mpa_start_step'])
        self.interactive_mpa_end_step = int(snapshot['interactive_mpa_end_step'])
        self.interactive_daily_catch = snapshot.get('interactive_daily_catch', [0.0]).copy()

        self.interactive_prev_states = [
            {
                'position': s['position'].copy(),
                'velocity': s['velocity'].copy(),
                'heading': float(s['heading']),
                'fuel': float(s['fuel']),
                'max_fuel': float(s['max_fuel']),
                'cargo': float(s['cargo']),
                'max_cargo': float(s['max_cargo']),
                'net_deployed': bool(s['net_deployed']),
                'current_temp': float(s['current_temp']),
                'inside_mpa': bool(s['inside_mpa']),
            }
            for s in snapshot['interactive_prev_states']
        ]

        self.env.time_step = int(snapshot['env']['time_step'])
        self.env.plankton_grid = snapshot['env']['plankton_grid'].copy()
        self.env.temperature_grid = snapshot['env']['temperature_grid'].copy()
        self.env.mpa_grid = snapshot['env']['mpa_grid'].copy()
        self.env.current_u = snapshot['env']['current_u'].copy()
        self.env.current_v = snapshot['env']['current_v'].copy()
        self.env.wind_u = snapshot['env']['wind_u'].copy()
        self.env.wind_v = snapshot['env']['wind_v'].copy()

        self.fish.num_schools = int(snapshot['fish']['num_schools'])
        self.fish.positions = snapshot['fish']['positions'].copy()
        self.fish.velocities = snapshot['fish']['velocities'].copy()
        self.fish.energies = snapshot['fish']['energies'].copy()
        self.fish.ages = snapshot['fish']['ages'].copy()

        self.fleet.positions = snapshot['fleet']['positions'].copy()
        self.fleet.headings = snapshot['fleet']['headings'].copy()
        self.fleet.velocities = snapshot['fleet']['velocities'].copy()
        self.fleet.fuel_levels = snapshot['fleet']['fuel_levels'].copy()
        self.fleet.cargo_levels = snapshot['fleet']['cargo_levels'].copy()
        self.fleet.nets_deployed = snapshot['fleet']['nets_deployed'].copy()
        self.fleet.just_sold = snapshot['fleet']['just_sold'].copy()
        self.fleet.cargo_sold = snapshot['fleet']['cargo_sold'].copy()

        self.port_catch = snapshot['port_catch'].copy()
        self.boat_catch = snapshot['boat_catch'].copy()

        for i, stats in enumerate(snapshot['agent_stats']):
            self.agents[i].total_reward = float(stats['total_reward'])
            self.agents[i].trips_completed = int(stats['trips_completed'])
            self.agents[i].total_catch = float(stats['total_catch'])
