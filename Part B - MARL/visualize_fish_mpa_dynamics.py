"""
Fish & MPA Dynamics Visualization (No Boats)

Visualizes how fish populations and Marine Protected Areas evolve over time
without fishing pressure. Shows the natural ecosystem dynamics and intelligent
MPA formation based on fish density.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import imageio
import os

from environment import OceanEnvironment
from fish_ecosystem import FishPopulation


class FishMPAVisualizer:
    def __init__(self, duration_hours=2190, frames_per_day=1):
        """
        Initialize visualizer
        
        Parameters:
        - duration_hours: Simulation length (default: 2190 = ~3 months)
        - frames_per_day: How many frames to capture per day (default: 1)
        """
        self.duration_hours = duration_hours
        self.capture_interval = int(24 / frames_per_day)  # Hours between frame captures
        
        # Initialize environment
        print("🌊 Initializing ocean environment...")
        self.env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
        
        # Initialize fish population
        print("🐟 Initializing fish population...")
        self.fish = FishPopulation(initial_population=2500, env_width=100, env_height=100)
        
        # Temperature colormap
        self.cmap = LinearSegmentedColormap.from_list(
            'ocean', ['#1a237e', '#0277bd', '#00bcd4', '#4caf50', '#ffeb3b', '#ff9800', '#d32f2f']
        )
        
        # Storage for frames
        self.frames = []
        
        # Metrics tracking
        self.fish_population_history = []
        self.total_energy_history = []
        self.mpa_coverage_history = []
        self.plankton_mean_history = []
        
        print(f"✓ Simulation ready: {duration_hours} hours ({duration_hours/24:.1f} days)")
        print(f"  Capturing 1 frame every {self.capture_interval} hours")
        print()
    
    def create_frame(self, step):
        """Create a single visualization frame"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Fish & MPA Dynamics - Hour {step} ({step/24:.1f} days)', 
                     fontsize=16, fontweight='bold')
        
        # ========== Panel 1: Fish Distribution + MPAs ==========
        ax1 = axes[0, 0]
        
        # Temperature background
        ax1.imshow(self.env.temperature_grid.T, origin='lower',
                  cmap=self.cmap, extent=[0, 100, 0, 100],
                  alpha=0.6, vmin=5, vmax=25)
        
        # Marine Protected Areas
        if hasattr(self.env, 'mpa_grid') and np.any(self.env.mpa_grid > 0.5):
            mpa_mask = self.env.mpa_grid > 0.5
            ax1.imshow(mpa_mask.T, origin='lower', extent=[0, 100, 0, 100],
                      cmap='Reds', alpha=0.4, vmin=0, vmax=2)
            mpa_coverage = np.sum(mpa_mask) / (100 * 100) * 100
        else:
            mpa_coverage = 0.0
        
        # Fish schools
        if self.fish.num_schools > 0:
            sizes = np.clip(self.fish.energies / 10, 2, 100)
            ax1.scatter(self.fish.positions[:, 0], self.fish.positions[:, 1],
                       s=sizes, c='cyan', alpha=0.8, edgecolors='blue', linewidths=0.5)
        
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('X Position (km)', fontsize=11)
        ax1.set_ylabel('Y Position (km)', fontsize=11)
        ax1.set_title(f'Fish Distribution & MPAs\n{self.fish.num_schools} schools | MPA Coverage: {mpa_coverage:.1f}%',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        
        # ========== Panel 2: Fish Density Heatmap ==========
        ax2 = axes[0, 1]
        
        # Compute fish density grid
        fish_density = np.zeros((100, 100))
        if self.fish.num_schools > 0:
            for i in range(self.fish.num_schools):
                x, y = self.fish.positions[i]
                energy = self.fish.energies[i]
                gx, gy = int(np.clip(x, 0, 99)), int(np.clip(y, 0, 99))
                
                # Gaussian spread (radius 5)
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < 100 and 0 <= ny < 100:
                            distance = np.sqrt(dx**2 + dy**2)
                            contribution = energy * np.exp(-(distance**2) / (2 * 2**2))
                            fish_density[nx, ny] += contribution
        
        # Plot density
        im2 = ax2.imshow(fish_density.T, origin='lower', extent=[0, 100, 0, 100],
                        cmap='YlOrRd', interpolation='bilinear')
        
        # Overlay MPA boundaries
        if hasattr(self.env, 'mpa_grid') and np.any(self.env.mpa_grid > 0.5):
            from matplotlib.patches import Rectangle
            mpa_mask = self.env.mpa_grid > 0.5
            # Draw MPA borders
            ax2.contour(mpa_mask.T, levels=[0.5], colors='red', linewidths=2, 
                       extent=[0, 100, 0, 100], alpha=0.8)
        
        plt.colorbar(im2, ax=ax2, label='Fish Density')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel('X Position (km)', fontsize=11)
        ax2.set_ylabel('Y Position (km)', fontsize=11)
        ax2.set_title('Fish Density Heatmap\n(MPAs form around high-density zones)',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2)
        
        # ========== Panel 3: Population Metrics Over Time ==========
        ax3 = axes[1, 0]
        
        hours = np.arange(len(self.fish_population_history))
        days = hours / 24.0
        
        # Plot population and energy on dual y-axes
        color1 = 'tab:blue'
        ax3.set_xlabel('Time (days)', fontsize=11)
        ax3.set_ylabel('Fish Population (schools)', color=color1, fontsize=11)
        ax3.plot(days, self.fish_population_history, color=color1, linewidth=2, label='Population')
        ax3.tick_params(axis='y', labelcolor=color1)
        ax3.grid(True, alpha=0.3)
        
        ax3_twin = ax3.twinx()
        color2 = 'tab:orange'
        ax3_twin.set_ylabel('Total Energy', color=color2, fontsize=11)
        ax3_twin.plot(days, self.total_energy_history, color=color2, linewidth=2, 
                     linestyle='--', label='Total Energy')
        ax3_twin.tick_params(axis='y', labelcolor=color2)
        
        ax3.set_title('Population & Energy Dynamics', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # ========== Panel 4: MPA Coverage and Plankton ==========
        ax4 = axes[1, 1]
        
        # Plot MPA coverage and plankton on dual y-axes
        color3 = 'tab:red'
        ax4.set_xlabel('Time (days)', fontsize=11)
        ax4.set_ylabel('MPA Coverage (%)', color=color3, fontsize=11)
        ax4.plot(days, self.mpa_coverage_history, color=color3, linewidth=2, label='MPA Coverage')
        ax4.tick_params(axis='y', labelcolor=color3)
        ax4.grid(True, alpha=0.3)
        
        ax4_twin = ax4.twinx()
        color4 = 'tab:green'
        ax4_twin.set_ylabel('Mean Plankton Density', color=color4, fontsize=11)
        ax4_twin.plot(days, self.plankton_mean_history, color=color4, linewidth=2, 
                     linestyle='--', label='Plankton')
        ax4_twin.tick_params(axis='y', labelcolor=color4)
        
        ax4.set_title('MPA Coverage & Food Availability', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        # Add statistics text box
        stats_text = (
            f"Current State:\n"
            f"Fish: {self.fish.num_schools} schools\n"
            f"Total Energy: {self.fish.energies.sum():.0f}\n"
            f"Avg Energy/School: {self.fish.energies.mean():.1f}\n"
            f"MPA Coverage: {mpa_coverage:.1f}%\n"
            f"Plankton: {self.env.plankton_grid.mean():.3f}"
        )
        ax4.text(0.02, 0.02, stats_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Convert to PIL Image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = Image.fromarray(buf)
        
        plt.close(fig)
        return frame
    
    def run_simulation(self):
        """Run the simulation and capture frames"""
        print(f"🎬 Starting simulation for {self.duration_hours} hours...")
        print()
        
        for step in range(self.duration_hours):
            # Step ecosystem
            self.env.step()
            self.fish.step(self.env)
            
            # Update MPAs (every 30 days)
            if hasattr(self.env, 'update_mpas'):
                self.env.update_mpas(self.fish)
            
            # Track metrics (every hour)
            self.fish_population_history.append(self.fish.num_schools)
            self.total_energy_history.append(self.fish.energies.sum())
            self.plankton_mean_history.append(self.env.plankton_grid.mean())
            
            mpa_coverage = 0.0
            if hasattr(self.env, 'mpa_grid') and np.any(self.env.mpa_grid > 0.5):
                mpa_coverage = np.sum(self.env.mpa_grid > 0.5) / (100 * 100) * 100
            self.mpa_coverage_history.append(mpa_coverage)
            
            # Capture frame
            if step % self.capture_interval == 0 or step == self.duration_hours - 1:
                frame = self.create_frame(step)
                self.frames.append(frame)
                print(f"⏱️  Hour {step:>4} ({step/24:>6.2f} days) | "
                      f"Fish: {self.fish.num_schools:>4} | "
                      f"Energy: {self.fish.energies.sum():>8.0f} | "
                      f"MPA: {mpa_coverage:>5.1f}% | "
                      f"Frame: {len(self.frames):>3}")
        
        print()
        print(f"✓ Simulation complete! Captured {len(self.frames)} frames")
    
    def save_gif(self, filename='fish_mpa_dynamics.gif', fps=5):
        """Save frames as animated GIF"""
        print(f"\n💾 Saving GIF: {filename}")
        print(f"   Frames: {len(self.frames)}")
        print(f"   FPS: {fps}")
        
        imageio.mimsave(filename, self.frames, fps=fps, loop=0)
        
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"✓ GIF saved successfully! Size: {file_size:.2f} MB")
    
    def print_summary(self):
        """Print simulation summary"""
        print("\n" + "="*70)
        print("SIMULATION SUMMARY")
        print("="*70)
        
        initial_pop = self.fish_population_history[0]
        final_pop = self.fish_population_history[-1]
        pop_change = final_pop - initial_pop
        pop_change_pct = (pop_change / initial_pop) * 100
        
        initial_energy = self.total_energy_history[0]
        final_energy = self.total_energy_history[-1]
        energy_change = final_energy - initial_energy
        energy_change_pct = (energy_change / initial_energy) * 100
        
        max_mpa = max(self.mpa_coverage_history)
        final_mpa = self.mpa_coverage_history[-1]
        
        print(f"\nPopulation:")
        print(f"  Initial: {initial_pop} schools")
        print(f"  Final:   {final_pop} schools")
        print(f"  Change:  {pop_change:+d} schools ({pop_change_pct:+.1f}%)")
        
        print(f"\nTotal Energy:")
        print(f"  Initial: {initial_energy:,.0f}")
        print(f"  Final:   {final_energy:,.0f}")
        print(f"  Change:  {energy_change:+,.0f} ({energy_change_pct:+.1f}%)")
        
        print(f"\nMPA Coverage:")
        print(f"  Maximum: {max_mpa:.1f}%")
        print(f"  Final:   {final_mpa:.1f}%")
        
        print(f"\nPlankton:")
        print(f"  Initial: {self.plankton_mean_history[0]:.3f}")
        print(f"  Final:   {self.plankton_mean_history[-1]:.3f}")
        print(f"  Min:     {min(self.plankton_mean_history):.3f}")
        print(f"  Max:     {max(self.plankton_mean_history):.3f}")
        
        # Ecosystem health assessment
        print(f"\nEcosystem Health:")
        if pop_change > 0 and final_energy > initial_energy:
            print(f"  ✓ HEALTHY - Population growing, energy increasing")
        elif pop_change > -100 and final_pop > 2000:
            print(f"  ✓ STABLE - Population maintained above 2000 schools")
        elif final_pop < 1000:
            print(f"  ⚠️  AT RISK - Population declining significantly")
        else:
            print(f"  ~ TRANSITIONING - Population adjusting to equilibrium")
        
        print("="*70)


def main():
    """Main execution"""
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "FISH & MPA DYNAMICS VISUALIZER" + " "*23 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Configuration
    duration_hours = 2190  # ~3 months (91.25 days)
    frames_per_day = 2     # Capture 2 frames per day
    output_filename = 'fish_mpa_dynamics.gif'
    fps = 8                # 8 frames per second in GIF
    
    print(f"Configuration:")
    print(f"  Duration: {duration_hours} hours ({duration_hours/24:.1f} days)")
    print(f"  Frame rate: {frames_per_day} frames/day")
    print(f"  Total frames: ~{int(duration_hours/24 * frames_per_day)}")
    print(f"  GIF speed: {fps} FPS")
    print()
    
    # Create visualizer
    viz = FishMPAVisualizer(duration_hours=duration_hours, frames_per_day=frames_per_day)
    
    # Run simulation
    viz.run_simulation()
    
    # Save GIF
    viz.save_gif(filename=output_filename, fps=fps)
    
    # Print summary
    viz.print_summary()
    
    print(f"\n🎉 Complete! Watch the dynamics in: {output_filename}")
    print()


if __name__ == "__main__":
    main()
