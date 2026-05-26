"""
Lightweight GIF Generator - Creates animated fishing simulation
Simplified version without complex text rendering that was causing issues
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image, ImageDraw, ImageFont
import imageio

from environment import OceanEnvironment
from fish_ecosystem import FishPopulation
from fleet_physics import FishingFleet
from fishing_agent import FishingAgent


class SimpleGIFMaker:
    def __init__(self, num_steps=1000, fps=15):
        self.num_steps = num_steps
        self.fps = fps
        
        # Setup
        self.env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
        self.fish = FishPopulation(initial_population=800, env_width=100, env_height=100)
        self.ports = [[10.0, 10.0], [90.0, 10.0], [50.0, 90.0]]
        self.fleet = FishingFleet(num_boats=3, ports=self.ports, env_width=100, env_height=100)
        
        # Load agents
        self.agents = []
        for boat_id in range(3):
            agent = FishingAgent(
                boat_id=boat_id,
                home_port=self.ports[boat_id % len(self.ports)],
                env_width=100,
                env_height=100
            )
            checkpoint_path = f'checkpoints/agent_{boat_id}_best.pt'
            if os.path.exists(checkpoint_path):
                agent.load_checkpoint(checkpoint_path)
                agent.epsilon = 0.05
            self.agents.append(agent)
        
        # Colormap for temperature
        colors = ['#000033', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        self.cmap = LinearSegmentedColormap.from_list('ocean', colors, N=100)
        self.boat_colors = ['#FF4444', '#44FF44', '#4444FF']
        
        self.frames = []
        self.total_catch = [0, 0, 0]
        self.sales = []
        
        print(f"\n🎬 GIF Generator ready: {num_steps} steps at {fps} FPS")
        print(f"   Expected duration: {num_steps/fps:.1f} seconds")
        print(f"   File size estimate: {(num_steps/fps) * 1.5:.1f} MB\n")
    
    def simulate_and_capture(self):
        """Run simulation and capture matplotlib frames as PIL images"""
        print("📹 Running simulation and capturing frames...")
        
        # Temperature colormap for reuse
        norm = plt.Normalize(vmin=5, vmax=25)
        
        for step in range(self.num_steps):
            # Get actions
            actions = []
            for boat_idx in range(3):
                state = {
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
                obs = self.agents[boat_idx].get_observation(state, self.fish, self.env)
                action = self.agents[boat_idx].select_action(obs, state)
                actions.append(action)
            
            # Step simulation
            self.fleet.step(np.array(actions), self.fish, self.env)
            
            # Track catches
            for i in range(3):
                if self.fleet.just_sold[i]:
                    self.sales.append((step, i, self.fleet.cargo_sold[i]))
                    self.total_catch[i] += self.fleet.cargo_sold[i]
            
            self.fish.step(self.env)
            self.env.step()
            
            # Update Marine Protected Areas
            if hasattr(self.env, 'update_mpas'):
                self.env.update_mpas(self.fish)
            
            # Create frame
            if step % max(1, self.num_steps // 100) == 0 or step < 5:
                frame = self._create_frame(step)
                self.frames.append(frame)
            
            if (step + 1) % 50 == 0:
                print(f"  Progress: {step + 1}/{self.num_steps} | " +
                      f"Fish: {self.fish.num_schools} | " +
                      f"Sales: {len(self.sales)}")
        
        print(f"✅ Captured {len(self.frames)} frames")
    
    def _create_frame(self, step):
        """Create a single PIL Image frame"""
        fig = plt.figure(figsize=(14, 8), dpi=80)
        ax = fig.add_subplot(111)
        
        # Temperature background
        im = ax.imshow(self.env.temperature_grid.T, origin='lower', 
                      cmap=self.cmap, extent=[0, 100, 0, 100], 
                      alpha=0.6, vmin=5, vmax=25)
        
        # Marine Protected Areas (MPAs) - Red overlay
        if hasattr(self.env, 'mpa_grid') and np.any(self.env.mpa_grid > 0.5):
            mpa_mask = self.env.mpa_grid > 0.5
            ax.imshow(mpa_mask.T, origin='lower',
                     extent=[0, 100, 0, 100],
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
        for i in range(3):
            pos = self.fleet.positions[i]
            heading = self.fleet.headings[i]
            
            # Boat circle
            ax.plot(pos[0], pos[1], 'o', color=self.boat_colors[i],
                   markersize=18, markeredgecolor='black', markeredgewidth=2)
            
            # Heading arrow
            dx = 6 * np.cos(heading)
            dy = 6 * np.sin(heading)
            ax.arrow(pos[0], pos[1], dx, dy,
                    head_width=2, head_length=1.2,
                    fc=self.boat_colors[i], ec='black', linewidth=2)
            
            # Net deployed indicator
            if self.fleet.nets_deployed[i]:
                circle = plt.Circle(pos, 5, color='red', alpha=0.15)
                ax.add_patch(circle)
                ax.plot(pos[0], pos[1], 'o', markerfacecolor='none',
                       markeredgecolor='red', markersize=32, markeredgewidth=2)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_title(f'Fishing Simulation - Hour {step}\nFish: {self.fish.num_schools} | '
                    f'Cargo: {np.sum(self.fleet.cargo_levels):.0f}t | Sales: {len(self.sales)}',
                    fontsize=12, fontweight='bold')
        
        # Convert to PIL Image using buffer approach
        fig.canvas.draw()
        
        # Get the RGBA buffer from the canvas
        rgba_buf = fig.canvas.buffer_rgba()
        (w, h) = fig.canvas.get_width_height()
        
        # Convert to PIL Image
        image = Image.frombytes('RGBA', (w, h), rgba_buf)
        image = image.convert('RGB')  # Convert RGBA to RGB for GIF
        plt.close(fig)
        
        return image
    
    def save_gif(self, filename='fishing_simulation.gif'):
        """Save frames as GIF using imageio"""
        print(f"\n💾 Saving GIF: {filename}")
        print(f"   Frames: {len(self.frames)}")
        print(f"   FPS: {self.fps}")
        
        # Convert PIL images to numpy arrays
        frames_array = [np.array(img) for img in self.frames]
        
        # Save with imageio (more reliable than pillow writer)
        imageio.mimsave(filename, frames_array, fps=self.fps, loop=0)
        
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"✅ Saved: {filename} ({file_size_mb:.1f} MB)\n")
        
        # Print summary
        print("="*70)
        print("GIF SUMMARY")
        print("="*70)
        print(f"Duration: {len(self.frames) / self.fps:.1f} seconds")
        print(f"Resolution: 1120x640 pixels")
        print(f"Total Sales: {len(self.sales)}")
        print(f"Total Catch: {sum(self.total_catch):.1f} tons")
        print(f"\nCatch by boat:")
        for i, catch in enumerate(self.total_catch):
            print(f"  Boat {i}: {catch:.1f} tons")
        print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("FISHING SIMULATION GIF GENERATOR")
    print("="*70)
    
    # Create generator
    generator = SimpleGIFMaker(num_steps=500, fps=20)
    
    # Run simulation and capture frames
    generator.simulate_and_capture()
    
    # Save as GIF
    generator.save_gif('fishing_simulation.gif')
    
    print("\n🎉 GIF COMPLETE!")
    print("   Open 'fishing_simulation.gif' to watch the trained agents fish!\n")
