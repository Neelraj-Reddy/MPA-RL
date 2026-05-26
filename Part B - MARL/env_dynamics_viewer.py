"""
Environment Dynamics Viewer

Visualize how the ocean environment evolves independently over time.
Shows temperature gradients, plankton distribution, currents, and wind patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from environment import OceanEnvironment

def visualize_environment_dynamics(steps=1000, sample_interval=100):
    """
    Run environment simulation and visualize key frames.
    
    Args:
        steps: Total simulation steps
        sample_interval: Sample every N steps for visualization
    """
    
    # Initialize environment
    env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
    
    print("="*70)
    print("OCEAN ENVIRONMENT DYNAMICS VIEWER")
    print("="*70)
    print(f"Map Size: {env.width}x{env.height} units")
    print(f"Simulation: {steps} steps ({steps} simulated hours)")
    print(f"Sampling every {sample_interval} steps for visualization")
    print("="*70)
    
    # Storage for sampled frames
    # Include final step by going to steps + 1
    sample_steps = list(range(0, steps + 1, sample_interval))
    
    frames = {
        'steps': sample_steps,
        'temperature': [],
        'plankton': [],
        'current_u': [],
        'current_v': [],
        'wind_u': [],
        'wind_v': []
    }
    
    # Run simulation
    print("\nSimulating environment...")
    for step in range(steps + 1):
        if step % 200 == 0:
            print(f"  Step {step}/{steps}")
        
        # Capture frame if at sample point
        if step in sample_steps:
            frames['temperature'].append(env.temperature_grid.copy())
            frames['plankton'].append(env.plankton_grid.copy())
            frames['current_u'].append(env.current_u.copy())
            frames['current_v'].append(env.current_v.copy())
            frames['wind_u'].append(env.wind_u.copy())
            frames['wind_v'].append(env.wind_v.copy())
        
        # Step environment
        env.step()
    
    print(f"✓ Simulation complete - captured {len(sample_steps)} frames")
    
    # Generate comprehensive visualization
    create_visualization(frames, sample_steps)
    
    # Print statistical summary
    print_statistics(frames, sample_steps)

def create_visualization(frames, sample_steps):
    """Create multi-panel visualization of environment dynamics"""
    
    num_frames = len(sample_steps)
    
    # Create main figure with grid layout
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Ocean Environment Dynamics Over Time', fontsize=18, fontweight='bold')
    
    temp_frames = frames['temperature']
    plank_frames = frames['plankton']
    curr_u_frames = frames['current_u']
    curr_v_frames = frames['current_v']
    wind_u_frames = frames['wind_u']
    wind_v_frames = frames['wind_v']
    
    # Temperature row
    print("\nGenerating temperature visualization...")
    temp_norm = Normalize(vmin=5, vmax=25)
    for idx, step in enumerate(sample_steps):
        ax = plt.subplot(3, num_frames, idx + 1)
        im = ax.imshow(temp_frames[idx].T, cmap='RdYlBu_r', norm=temp_norm, origin='lower')
        ax.set_title(f'Temp @ Step {step}h', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == num_frames - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('°C', rotation=270, labelpad=15)
    
    # Plankton row
    print("Generating plankton visualization...")
    plank_norm = Normalize(vmin=0, vmax=1)
    for idx, step in enumerate(sample_steps):
        ax = plt.subplot(3, num_frames, num_frames + idx + 1)
        im = ax.imshow(plank_frames[idx].T, cmap='YlGn', norm=plank_norm, origin='lower')
        ax.set_title(f'Plankton @ Step {step}h', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == num_frames - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Density', rotation=270, labelpad=15)
    
    # Currents + Wind row (quiver plots)
    print("Generating current/wind visualization...")
    for idx, step in enumerate(sample_steps):
        ax = plt.subplot(3, num_frames, 2*num_frames + idx + 1)
        
        # Sample every 5 cells for visibility
        downsampled = 5
        y_indices, x_indices = np.mgrid[0:100:downsampled, 0:100:downsampled]
        
        u_current = curr_u_frames[idx][x_indices, y_indices]
        v_current = curr_v_frames[idx][x_indices, y_indices]
        u_wind = wind_u_frames[idx][x_indices, y_indices]
        v_wind = wind_v_frames[idx][x_indices, y_indices]
        
        # Plot currents as arrows
        ax.quiver(y_indices, x_indices, v_current, u_current, 
                 alpha=0.6, scale=0.5, scale_units='inches', color='blue', label='Current')
        
        # Overlay wind (slightly offset for visibility)
        ax.quiver(y_indices+0.2, x_indices+0.2, v_wind*2, u_wind*2, 
                 alpha=0.4, scale=0.5, scale_units='inches', color='red', label='Wind (2x)')
        
        ax.set_xlim(-1, 100)
        ax.set_ylim(-1, 100)
        ax.set_aspect('equal')
        ax.set_title(f'Currents @ Step {step}h', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('environment_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved: environment_dynamics.png")
    plt.close()

def print_statistics(frames, sample_steps):
    """Print statistical summary of environment behavior"""
    
    print("\n" + "="*70)
    print("ENVIRONMENT STATISTICS")
    print("="*70)
    
    temp_frames = frames['temperature']
    plank_frames = frames['plankton']
    
    print("\nTemperature Evolution:")
    print(f"{'Step':>8} {'Min':>8} {'Mean':>8} {'Max':>8} {'Std Dev':>10}")
    print("-" * 45)
    for idx, step in enumerate(sample_steps):
        temps = temp_frames[idx]
        print(f"{step:>8.0f} {temps.min():>8.2f} {temps.mean():>8.2f} {temps.max():>8.2f} {temps.std():>10.2f}")
    
    print("\nPlankton Evolution:")
    print(f"{'Step':>8} {'Min':>8} {'Mean':>8} {'Max':>8} {'Std Dev':>10}")
    print("-" * 45)
    for idx, step in enumerate(sample_steps):
        planks = plank_frames[idx]
        print(f"{step:>8.0f} {planks.min():>8.4f} {planks.mean():>8.4f} {planks.max():>8.4f} {planks.std():>10.6f}")
    
    print("\nCurrent Strengths (RMS):")
    print(f"{'Step':>8} {'Magnitude':>15}")
    print("-" * 25)
    for idx, step in enumerate(sample_steps):
        u = frames['current_u'][idx]
        v = frames['current_v'][idx]
        magnitude = np.sqrt(u**2 + v**2).mean()
        print(f"{step:>8.0f} {magnitude:>15.4f}")
    
    print("\nWind Strengths (RMS):")
    print(f"{'Step':>8} {'Magnitude':>15}")
    print("-" * 25)
    for idx, step in enumerate(sample_steps):
        u = frames['wind_u'][idx]
        v = frames['wind_v'][idx]
        magnitude = np.sqrt(u**2 + v**2).mean()
        print(f"{step:>8.0f} {magnitude:>15.4f}")
    
    print("="*70)

if __name__ == "__main__":
    print("\n")
    visualize_environment_dynamics(steps=1000, sample_interval=100)
    print("\n✅ Environment dynamics analysis complete!\n")
