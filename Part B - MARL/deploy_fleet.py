"""
Fishing Fleet Deployment System

Deploys trained fishing agents in a realistic ocean environment.
- Loads 5 trained agent models
- Deploys 3 instances of each model (15 boats total)
- Uses randomized port locations for each episode
- Generates individual GIFs for each trained agent model
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
import imageio
import os
import time
from datetime import timedelta

from environment import OceanEnvironment
from fish_ecosystem import FishPopulation
from fleet_physics import FishingFleet
from fishing_agent import FishingAgent
from heuristic_fishing_agent import HeuristicFishingAgent


class FleetDeployer:
    def __init__(self, deployment_config):
        """Initialize deployment system with trained RL models or training-free heuristics."""
        self.config = deployment_config
        self.controller_mode = self.config.get('controller_mode', 'rl').lower()
        if self.controller_mode not in ('rl', 'heuristic'):
            raise ValueError("controller_mode must be 'rl' or 'heuristic'")

        self.num_trained_agents = 5
        self.instances_per_agent = 3
        self.total_boats = self.num_trained_agents * self.instances_per_agent  # 15 boats
        
        # Setup logging
        self.log_file = open('deployment_log.txt', 'w', encoding='utf-8')
        header = "="*80 + "\n"
        header += "FISHING FLEET DEPLOYMENT LOG\n"
        header += "="*80 + "\n"
        header += f"Deployment Start Time: {timedelta(seconds=0)}\n"
        header += f"Configuration: {self.config}\n"
        header += "="*80 + "\n\n"
        self.log_file.write(header)
        
        # Environment setup
        self.env = OceanEnvironment(
            width=self.config['env_width'],
            height=self.config['env_height'],
            hours_per_tick=self.config['hours_per_tick']
        )
        
        # Random ports for this deployment
        self.ports = self._generate_random_ports(
            self.config['num_ports'],
            self.config['env_width'],
            self.config['env_height']
        )
        
        # Fish population
        self.fish = FishPopulation(
            initial_population=self.config['initial_fish'],
            env_width=self.config['env_width'],
            env_height=self.config['env_height']
        )
        
        # Fleet
        self.fleet = FishingFleet(
            num_boats=self.total_boats,
            ports=self.ports,
            env_width=self.config['env_width'],
            env_height=self.config['env_height']
        )
        
        # Build controllers (RL checkpoint-based or heuristic training-free)
        self.agents = []
        self.agent_model_ids = []  # Track controller group each boat uses (0-4)
        
        print("\n" + "="*70)
        if self.controller_mode == 'rl':
            print("LOADING TRAINED MODELS")
        else:
            print("INITIALIZING HEURISTIC CONTROLLERS (TRAINING-FREE)")
        print("="*70)
        
        for model_id in range(self.num_trained_agents):
            checkpoint_path = f'checkpoints/agent_{model_id}_best.pt'
            
            if self.controller_mode == 'rl' and not os.path.exists(checkpoint_path):
                print(f"⚠️  Warning: {checkpoint_path} not found!")
                continue

            # Create instances of this controller group
            for instance in range(self.instances_per_agent):
                boat_id = len(self.agents)
                agent_home_port = self.ports[boat_id % len(self.ports)]

                if self.controller_mode == 'rl':
                    agent = FishingAgent(
                        boat_id=boat_id,
                        home_port=agent_home_port,
                        env_width=self.config['env_width'],
                        env_height=self.config['env_height']
                    )
                    agent.load_checkpoint(checkpoint_path)
                    agent.epsilon = 0.05  # Low exploration - deployment mode
                else:
                    agent = HeuristicFishingAgent(
                        boat_id=boat_id,
                        home_port=agent_home_port,
                        env_width=self.config['env_width'],
                        env_height=self.config['env_height']
                    )
                
                self.agents.append(agent)
                self.agent_model_ids.append(model_id)

                if self.controller_mode == 'rl':
                    print(f"✅ Loaded Model {model_id}, Instance {instance+1} (Boat {len(self.agents)-1})")
                else:
                    print(f"✅ Ready Heuristic Group {model_id}, Instance {instance+1} (Boat {len(self.agents)-1})")

        self.total_boats = len(self.agents)
        self.num_trained_agents = len(set(self.agent_model_ids)) if len(self.agent_model_ids) > 0 else 0

        if self.total_boats == 0:
            raise RuntimeError("No controllers initialized. Check checkpoints for RL mode or config.")
        
        # Reset fleet positions near ports
        for i, agent in enumerate(self.agents):
            port_idx = i % len(self.ports)
            self.fleet.positions[i] = self.ports[port_idx] + (np.random.rand(2) - 0.5) * 3.0
            self.fleet.fuel_levels[i] = self.fleet.max_fuel
            self.fleet.cargo_levels[i] = 0.0
            self.fleet.velocities[i] = 0.0
            self.fleet.headings[i] = np.random.rand() * 2 * np.pi
        
        # Tracking
        self.deployment_rewards = np.zeros(self.total_boats)
        self.deployment_catches = np.zeros(self.total_boats)
        self.deployment_trips = np.zeros(self.total_boats)
        
        # GIF frame storage per agent model
        self.frames_per_model = {i: [] for i in range(self.num_trained_agents)}
        self.sales_per_model = {i: [] for i in range(self.num_trained_agents)}
        self.total_catch_per_model = {i: 0.0 for i in range(self.num_trained_agents)}
        
        # Colormap
        colors = ['#000033', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        self.cmap = LinearSegmentedColormap.from_list('ocean', colors, N=100)
        self.boat_colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', '#FF44FF', 
                           '#44FFFF', '#FF8844', '#88FF44', '#4488FF', '#FF4488',
                           '#44FF88', '#8844FF', '#FFFF88', '#88FFFF', '#FF88FF']
        
        print("\n" + "="*70)
        print("DEPLOYMENT CONFIGURATION")
        print("="*70)
        print(f"Controller Mode: {self.controller_mode.upper()}")
        print(f"Total Boats: {self.total_boats} ({self.num_trained_agents} groups × {self.instances_per_agent} each)")
        print(f"Environment: {self.config['env_width']}x{self.config['env_height']}")
        print(f"Ports: {len(self.ports)} randomized locations")
        print(f"Deployment Steps: {self.config['deployment_steps']}")
        print(f"Simulated Duration: {self.config['deployment_steps']} hours (1 year / {self.config['deployment_steps']/24:.0f} days)")
        print("="*70 + "\n")
    
    def _generate_random_ports(self, num_ports, width, height):
        """Generate random port locations with minimum spacing"""
        ports = []
        min_spacing = 20.0
        margin = 10.0
        
        for _ in range(num_ports):
            max_attempts = 100
            for attempt in range(max_attempts):
                x = np.random.uniform(margin, width - margin)
                y = np.random.uniform(margin, height - margin)
                candidate = [x, y]
                
                if len(ports) == 0:
                    ports.append(candidate)
                    break
                
                min_dist = min([np.linalg.norm(np.array(candidate) - np.array(p)) for p in ports])
                if min_dist >= min_spacing:
                    ports.append(candidate)
                    break
        
        return ports
    
    def get_boat_state(self, boat_idx):
        """Extract state information for a specific boat"""
        return {
            'position': self.fleet.positions[boat_idx].copy(),
            'velocity': np.array([
                np.cos(self.fleet.headings[boat_idx]) * self.fleet.velocities[boat_idx],
                np.sin(self.fleet.headings[boat_idx]) * self.fleet.velocities[boat_idx]
            ]),
            'heading': self.fleet.headings[boat_idx],
            'fuel': self.fleet.fuel_levels[boat_idx],
            'max_fuel': self.fleet.max_fuel,
            'cargo': self.fleet.cargo_levels[boat_idx],
            'max_cargo': self.fleet.max_cargo,
            'net_deployed': self.fleet.nets_deployed[boat_idx],
            'current_temp': self.env.get_temperature(*self.fleet.positions[boat_idx]),
            'inside_mpa': self.env.is_in_mpa(*self.fleet.positions[boat_idx])
        }
    
    def _log_step(self, step, step_catch):
        """Detailed logging of deployment step"""
        log_output = f"\n{'='*80}\n"
        log_output += f"STEP {step:,} / 🕐 {step} hours ({step/24:.1f} days)\n"
        log_output += f"{'='*80}\n"
        
        # Fleet-wide stats
        log_output += f"\n📊 FLEET OVERVIEW:\n"
        log_output += f"   🐟 Fish Schools: {self.fish.num_schools}\n"
        log_output += f"   🎣 Tonnes Caught This Step: {step_catch:.2f}t\n"
        log_output += f"   📈 Total Catch So Far: {np.sum(self.deployment_catches):.1f}t\n"
        log_output += f"   ⛽ Avg Fleet Fuel: {np.mean(self.fleet.fuel_levels):.0f}L / {self.fleet.max_fuel:.0f}L\n"
        log_output += f"   📦 Avg Fleet Cargo: {np.mean(self.fleet.cargo_levels):.1f}t / {self.fleet.max_cargo:.0f}t\n"
        
        # Per-boat details
        log_output += f"\n🚢 BOAT-BY-BOAT STATUS:\n"
        log_output += f"   {'Boat':<5} {'Model':<6} {'Pos':<18} {'Heading':<8} {'Net':<5} {'Fuel':<10} {'Cargo':<10} {'Trips':<6}\n"
        log_output += f"   {'-'*90}\n"
        
        for boat_idx in range(self.total_boats):
            model_id = self.agent_model_ids[boat_idx]
            pos = self.fleet.positions[boat_idx]
            heading_deg = np.degrees(self.fleet.headings[boat_idx]) % 360
            net_deployed = "YES" if self.fleet.nets_deployed[boat_idx] else "NO"
            fuel = self.fleet.fuel_levels[boat_idx]
            cargo = self.fleet.cargo_levels[boat_idx]
            trips = int(self.deployment_trips[boat_idx])
            
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})"
            heading_str = f"{heading_deg:.0f}°"
            fuel_str = f"{fuel:.0f}L"
            cargo_str = f"{cargo:.1f}t"
            
            log_output += f"   {boat_idx:<5} {model_id:<6} {pos_str:<18} {heading_str:<8} {net_deployed:<5} {fuel_str:<10} {cargo_str:<10} {trips:<6}\n"
        
        # Net deployment summary
        num_nets_deployed = np.sum(self.fleet.nets_deployed)
        log_output += f"\n🎣 FISHING STATUS:\n"
        log_output += f"   Nets Currently Deployed: {num_nets_deployed}/{self.total_boats}\n"
        log_output += f"   Boats Out of Fuel: {np.sum(self.fleet.fuel_levels <= 0)}\n"
        log_output += f"   Boats at Port: {np.sum([np.min(np.linalg.norm(self.fleet.positions[i] - self.ports, axis=1)) < 1.5 for i in range(self.total_boats)])}\n"
        
        # Temperature info
        temps_at_boats = [self.env.get_temperature(*self.fleet.positions[i]) for i in range(self.total_boats)]
        log_output += f"\n🌡️  ENVIRONMENT:\n"
        log_output += f"   Avg Water Temp at Boats: {np.mean(temps_at_boats):.1f}°C\n"
        log_output += f"   Min/Max Temp: {np.min(temps_at_boats):.1f}°C / {np.max(temps_at_boats):.1f}°C\n"
        log_output += f"   Optimal Fish Temp Zone: 14-22°C\n"
        
        # Catches per model
        log_output += f"\n📊 CATCHES BY MODEL:\n"
        for model_id in range(self.num_trained_agents):
            model_boats = [i for i, mid in enumerate(self.agent_model_ids) if mid == model_id]
            model_catch = np.sum(self.deployment_catches[model_boats])
            model_trips = np.sum(self.deployment_trips[model_boats])
            log_output += f"   Model {model_id}: {model_catch:.1f}t total ({int(model_trips)} trips, avg {model_catch/3:.2f}t/boat)\n"
        
        log_output += f"{'='*80}\n"
        
        # Print to console and file
        print(log_output)
        self.log_file.write(log_output)
        self.log_file.flush()
    
    def deploy(self):
        """Run deployment simulation"""
        print(f"🚀 Starting deployment for {self.config['deployment_steps']} steps...\n")
        
        start_time = time.time()
        prev_states = [self.get_boat_state(i) for i in range(self.total_boats)]
        
        # Logging tracking
        log_interval = max(1, self.config['deployment_steps'] // 50)  # Log 50 times during deployment
        fish_population_history = []
        total_catch_history = []
        
        for step in range(self.config['deployment_steps']):
            # Get actions from all agents
            actions = []
            for boat_idx in range(self.total_boats):
                state = self.get_boat_state(boat_idx)
                obs = self.agents[boat_idx].get_observation(state, self.fish, self.env)
                action = self.agents[boat_idx].select_action(obs, state)
                actions.append(action)
            
            # Step simulation
            self.fleet.step(np.array(actions), self.fish, self.env)
            self.fish.step(self.env)
            self.env.step()
            
            # Update Marine Protected Areas based on fish density
            if hasattr(self.env, 'update_mpas'):
                self.env.update_mpas(self.fish)
            
            # Track metrics
            step_catch_total = 0.0
            for boat_idx in range(self.total_boats):
                new_state = self.get_boat_state(boat_idx)
                
                reward = self.agents[boat_idx].calculate_reward(
                    prev_states[boat_idx],
                    actions[boat_idx],
                    new_state,
                    self.fleet.just_sold[boat_idx],
                    self.fleet.cargo_sold[boat_idx]
                )
                
                self.deployment_rewards[boat_idx] += reward
                self.deployment_catches[boat_idx] += self.fleet.cargo_sold[boat_idx]
                if self.fleet.just_sold[boat_idx]:
                    self.deployment_trips[boat_idx] += 1
                
                step_catch_total += self.fleet.cargo_sold[boat_idx]
                
                prev_states[boat_idx] = new_state
            
            # Logging
            if step % log_interval == 0:
                self._log_step(step, step_catch_total)
                fish_population_history.append(self.fish.num_schools)
                total_catch_history.append(np.sum(self.deployment_catches))
            
            # Capture frames for GIFs - one per model
            if step % max(1, self.config['deployment_steps'] // 100) == 0 or step < 5:
                # Create model-specific frames
                for model_id in range(self.num_trained_agents):
                    frame = self._create_frame_for_model(step, model_id)
                    self.frames_per_model[model_id].append(frame)
                
                # Track sales for each model
                for boat_idx in range(self.total_boats):
                    model_id = self.agent_model_ids[boat_idx]
                    if self.fleet.just_sold[boat_idx]:
                        self.sales_per_model[model_id].append(
                            (step, boat_idx, self.fleet.cargo_sold[boat_idx])
                        )
                        self.total_catch_per_model[model_id] += self.fleet.cargo_sold[boat_idx]
            
            # Progress
            if (step + 1) % max(1, self.config['deployment_steps'] // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Step {step + 1}/{self.config['deployment_steps']} | "
                      f"Fish: {self.fish.num_schools} | "
                      f"Total Catch: {np.sum(self.deployment_catches):.1f} tons | "
                      f"Time: {elapsed:.1f}s")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Deployment complete in {elapsed:.1f}s\n")
        
        # Fish population analysis
        print("\n" + "="*70)
        print("FISH POPULATION ANALYSIS")
        print("="*70)
        print(f"Initial Fish Schools: {fish_population_history[0]} (if available)")
        print(f"Final Fish Schools: {self.fish.num_schools}")
        print(f"Total Fish Caught: {np.sum(self.deployment_catches):.1f} tons")
        if len(fish_population_history) > 1:
            change = self.fish.num_schools - fish_population_history[0]
            print(f"Net Population Change: {change:+d} schools")
            print("\n⚠️  NOTE: If fish population stayed same or increased,")
            print("    reproduction is outpacing fishing pressure!")
        print("="*70 + "\n")
    
    def _create_frame(self, step):
        """Create a single PIL Image frame for GIFs (all boats)"""
        fig = plt.figure(figsize=(14, 8), dpi=80)
        ax = fig.add_subplot(111)
        
        # Temperature background
        im = ax.imshow(self.env.temperature_grid.T, origin='lower',
                      cmap=self.cmap, extent=[0, self.config['env_width'], 0, self.config['env_height']],
                      alpha=0.6, vmin=5, vmax=25)
        
        # Marine Protected Areas (MPAs) - Red overlay
        if hasattr(self.env, 'mpa_grid') and np.any(self.env.mpa_grid > 0.5):
            mpa_mask = self.env.mpa_grid > 0.5
            ax.imshow(mpa_mask.T, origin='lower',
                     extent=[0, self.config['env_width'], 0, self.config['env_height']],
                     cmap='Reds', alpha=0.35, vmin=0, vmax=2)
        
        # Fish
        if self.fish.num_schools > 0:
            sizes = np.clip(self.fish.energies / 10, 2, 80)
            ax.scatter(self.fish.positions[:, 0], self.fish.positions[:, 1],
                      s=sizes, c='cyan', alpha=0.7, edgecolors='blue', linewidths=0.5)
        
        # Ports
        for port in self.ports:
            ax.plot(port[0], port[1], 'w*', markersize=25,
                   markeredgecolor='black', markeredgewidth=2)
        
        # Boats
        for i in range(self.total_boats):
            pos = self.fleet.positions[i]
            heading = self.fleet.headings[i]
            
            # Boat circle
            ax.plot(pos[0], pos[1], 'o', color=self.boat_colors[i],
                   markersize=12, markeredgecolor='black', markeredgewidth=1)
            
            # Heading arrow
            dx = 4 * np.cos(heading)
            dy = 4 * np.sin(heading)
            ax.arrow(pos[0], pos[1], dx, dy,
                    head_width=1.5, head_length=0.8,
                    fc=self.boat_colors[i], ec='black', linewidth=1)
            
            # Net deployed
            if self.fleet.nets_deployed[i]:
                circle = plt.Circle(pos, 3, color='red', alpha=0.15)
                ax.add_patch(circle)
        
        # Labels and formatting
        ax.set_xlim(0, self.config['env_width'])
        ax.set_ylim(0, self.config['env_height'])
        ax.set_xlabel('X Position (km)', fontsize=10)
        ax.set_ylabel('Y Position (km)', fontsize=10)
        ax.set_title(f'Fishing Fleet Deployment - Step {step}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        # Convert to PIL
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        frame = Image.fromarray(buf)
        
        plt.close(fig)
        return frame
    
    def _create_frame_for_model(self, step, model_id):
        """Create a frame showing only boats for a specific trained model"""
        fig = plt.figure(figsize=(14, 8), dpi=80)
        ax = fig.add_subplot(111)
        
        # Temperature background
        im = ax.imshow(self.env.temperature_grid.T, origin='lower',
                      cmap=self.cmap, extent=[0, self.config['env_width'], 0, self.config['env_height']],
                      alpha=0.6, vmin=5, vmax=25)
        
        # Marine Protected Areas (MPAs) - Red overlay
        if hasattr(self.env, 'mpa_grid') and np.any(self.env.mpa_grid > 0.5):
            mpa_mask = self.env.mpa_grid > 0.5
            ax.imshow(mpa_mask.T, origin='lower',
                     extent=[0, self.config['env_width'], 0, self.config['env_height']],
                     cmap='Reds', alpha=0.35, vmin=0, vmax=2)
        
        # Fish
        if self.fish.num_schools > 0:
            sizes = np.clip(self.fish.energies / 10, 2, 80)
            ax.scatter(self.fish.positions[:, 0], self.fish.positions[:, 1],
                      s=sizes, c='cyan', alpha=0.7, edgecolors='blue', linewidths=0.5)
        
        # Ports
        for port in self.ports:
            ax.plot(port[0], port[1], 'w*', markersize=25,
                   markeredgecolor='black', markeredgewidth=2)
        
        # Only boats for this model
        model_boat_indices = [i for i, mid in enumerate(self.agent_model_ids) if mid == model_id]
        
        for boat_idx in model_boat_indices:
            pos = self.fleet.positions[boat_idx]
            heading = self.fleet.headings[boat_idx]
            
            # Boat circle
            ax.plot(pos[0], pos[1], 'o', color=self.boat_colors[boat_idx],
                   markersize=15, markeredgecolor='black', markeredgewidth=2)
            
            # Heading arrow
            dx = 5 * np.cos(heading)
            dy = 5 * np.sin(heading)
            ax.arrow(pos[0], pos[1], dx, dy,
                    head_width=2, head_length=1,
                    fc=self.boat_colors[boat_idx], ec='black', linewidth=2)
            
            # Net deployed
            if self.fleet.nets_deployed[boat_idx]:
                circle = plt.Circle(pos, 4, color='red', alpha=0.2)
                ax.add_patch(circle)
        
        # Labels and formatting
        ax.set_xlim(0, self.config['env_width'])
        ax.set_ylim(0, self.config['env_height'])
        ax.set_xlabel('X Position (km)', fontsize=10)
        ax.set_ylabel('Y Position (km)', fontsize=10)
        ax.set_title(f'Model {model_id} - Step {step} | Fish Schools: {self.fish.num_schools}', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        # Convert to PIL and add text overlay
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        frame = Image.fromarray(buf)
        
        # Add fish count and info using PIL
        draw = ImageDraw.Draw(frame)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Fish schools info
        fish_info = f"🐟 Fish Schools: {self.fish.num_schools}"
        draw.text((20, 20), fish_info, fill=(255, 255, 255), font=font)
        
        plt.close(fig)
        return frame
    
    def generate_gifs(self):
        """Generate individual GIFs for each trained agent model"""
        print("\n" + "="*70)
        print("GENERATING GIFs FOR EACH TRAINED MODEL")
        print("="*70 + "\n")
        
        for model_id in range(self.num_trained_agents):
            if len(self.frames_per_model[model_id]) == 0:
                print(f"⚠️  No frames captured for Model {model_id}")
                continue
            
            output_file = f'deployment_model_{model_id}_fleet.gif'
            
            print(f"📹 Model {model_id}: Creating GIF with {len(self.frames_per_model[model_id])} frames...")
            print(f"   Output: {output_file}")
            
            imageio.mimsave(output_file, self.frames_per_model[model_id], fps=5)
            
            print(f"✅ Saved: {output_file}")
            print(f"   Instances: 3 boats | Total Catch: {self.total_catch_per_model[model_id]:.1f} tons")
            print(f"   Completed Sales: {len(self.sales_per_model[model_id])}\n")
    
    def print_deployment_summary(self):
        """Print detailed deployment statistics"""
        summary = "\n" + "="*70 + "\n"
        summary += "DEPLOYMENT SUMMARY\n"
        summary += "="*70 + "\n"
        
        for model_id in range(self.num_trained_agents):
            summary += f"\n🤖 MODEL {model_id} (3 instances deployed):\n"
            
            # Find boats for this model
            model_boats = [i for i, mid in enumerate(self.agent_model_ids) if mid == model_id]
            
            model_reward = np.sum(self.deployment_rewards[model_boats])
            model_catch = np.sum(self.deployment_catches[model_boats])
            model_trips = np.sum(self.deployment_trips[model_boats])
            
            summary += f"   Total Reward: {model_reward:+.2f}\n"
            summary += f"   Total Catch: {model_catch:.1f} tons\n"
            summary += f"   Port Returns: {int(model_trips)}\n"
            summary += f"   Avg per Boat: {model_catch/3:.2f} tons, {model_trips/3:.1f} trips\n"
        
        summary += f"\n{'='*70}\n"
        summary += "OVERALL FLEET PERFORMANCE:\n"
        summary += f"{'='*70}\n"
        summary += f"🚢 Total Boats: {self.total_boats}\n"
        total_catch = np.sum(self.deployment_catches)
        total_trips = int(np.sum(self.deployment_trips))
        summary += f"🐟 Total Catch: {total_catch:.1f} tons\n"
        summary += f"💰 Total Trips: {total_trips}\n"
        summary += f"🎯 Average per Boat: {np.mean(self.deployment_catches):.2f} tons\n"
        summary += f"⚡ Avg Reward: {np.mean(self.deployment_rewards):+.2f}\n"
        summary += f"🐠 Fish Remaining: {self.fish.num_schools} schools\n"
        
        summary += f"\n{'='*70}\n"
        summary += "FISH POPULATION DIAGNOSIS\n"
        summary += f"{'='*70}\n"
        summary += f"Total Fish Caught (biomass): {total_catch:.1f} tons\n"
        summary += f"Final Fish Schools: {self.fish.num_schools}\n"
        summary += f"\n⚠️  WHY FISH MAY NOT BE DEPLETING:\n"
        summary += f"   1. Reproduction Rate: Fish reproduce faster than you catch them\n"
        summary += f"      → Check if catch_rates in fleet_physics.py are too low\n"
        summary += f"      → Or reduce reproduction_threshold in fish_ecosystem.py\n"
        summary += f"\n   2. Fish Detection: Boats may not be sensing fish correctly\n"
        summary += f"      → Check vision_range in fishing_agent.py (currently {self._get_agent_stat('vision_range')})\n"
        summary += f"\n   3. Catch Radius: Too small to catch nearby fish\n"
        summary += f"      → Check catch_radii in fleet_physics.py (currently [2.0, 3.5])\n"
        summary += f"\n   4. Net Deployment: Agents may not be deploying nets enough\n"
        summary += f"      → Check if agents have learned to deploy networks (ε = 0.05)\n"
        
        summary += "="*70 + "\n\n"
        
        # Print and log
        print(summary)
        self.log_file.write(summary)
        self.log_file.close()
        
        print(f"📝 Full log saved to: deployment_log.txt\n")
    
    def _get_agent_stat(self, stat_name):
        """Get a statistic from the first agent"""
        if len(self.agents) > 0:
            agent = self.agents[0]
            if stat_name == 'vision_range':
                return agent.vision_range
        return "unknown"


if __name__ == "__main__":
    config = {
        # Environment
        'env_width': 100,
        'env_height': 100,
        'hours_per_tick': 1,
        'initial_fish': 2500,
        
        # Fleet
        'num_ports': 5,
        
        # Deployment
        'deployment_steps': 8760,  # 1 year (365 days * 24 hours)
    }
    
    print("\n" + "="*70)
    print("FISHING FLEET DEPLOYMENT SYSTEM")
    print("="*70)
    print(f"\n🚀 Preparing to deploy 15 boats (5 models × 3 instances each)")
    print(f"📍 Random port locations will be generated")
    print(f"⏱️  Deployment duration: {config['deployment_steps']} hours (1 year / 365 days)")
    print(f"\n📊 Each GIF will show only the 3 boats for that trained model")
    print("="*70)
    
    try:
        input("\nPress Enter to start deployment...")
    except KeyboardInterrupt:
        print("\nDeployment cancelled.")
        exit(0)
    
    try:
        deployer = FleetDeployer(config)
        deployer.deploy()
        deployer.generate_gifs()
        deployer.print_deployment_summary()
        
        print("🎉 Deployment completed successfully!")
        print("Check deployment_model_X_fleet.gif files for visualizations.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during deployment: {e}")
        import traceback
        traceback.print_exc()
