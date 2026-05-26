"""
Fish Population Energy Timeline Analysis

Compares ecosystem health (total energy) with and without fishing boats.
Helps identify if boats are causing unrealistic energy drain.
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import OceanEnvironment
from fish_ecosystem import FishPopulation
from fleet_physics import FishingFleet
from fishing_agent import FishingAgent
import torch

def run_ecosystem_alone(duration_steps=8760):
    """Run fish ecosystem WITHOUT boats for baseline"""
    print("="*70)
    print("SCENARIO 1: FISH ECOSYSTEM ALONE (NO FISHING)")
    print("="*70)
    
    env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
    fish = FishPopulation(initial_population=2500, env_width=100, env_height=100)
    
    energy_timeline = []
    population_timeline = []
    plankton_timeline = []
    
    print(f"\n[ANALYSIS] Running {duration_steps:,} steps...")
    print(f"   Start: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy\n")
    
    for step in range(duration_steps):
        # Ecosystem step
        env.step()
        fish.step(env)
        
        # Update MPAs
        if hasattr(env, 'update_mpas'):
            env.update_mpas(fish)
        
        # Track metrics
        total_energy = fish.energies.sum()
        avg_plankton = env.plankton_grid.mean()
        
        energy_timeline.append(total_energy)
        population_timeline.append(fish.num_schools)
        plankton_timeline.append(avg_plankton)
        
        if (step + 1) % (duration_steps // 20) == 0:
            print(f"  Step {step+1:>6} | Fish: {fish.num_schools:>5} | "
                  f"Energy: {total_energy:>10.1f} | Plankton: {avg_plankton:.3f}")
    
    print(f"\n[OK] Final: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy")
    
    return {
        'energy': np.array(energy_timeline),
        'population': np.array(population_timeline),
        'plankton': np.array(plankton_timeline),
        'label': 'Fish Alone (No Boats)'
    }

def run_ecosystem_with_boats(duration_steps=8760, checkpoint_path='checkpoints/'):
    """Run fish ecosystem WITH 15 fishing boats"""
    print("\n" + "="*70)
    print("SCENARIO 2: FISH ECOSYSTEM + 15 FISHING BOATS")
    print("="*70)
    
    env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
    fish = FishPopulation(initial_population=2500, env_width=100, env_height=100)
    
    # Setup 15 boats (5 models × 3 instances)
    ports = [
        (20, 20), (80, 20), (50, 50),
        (20, 80), (80, 80)
    ]
    fleet = FishingFleet(num_boats=15, ports=ports, env_width=100, env_height=100)
    
    # Load trained agents
    agents = []
    try:
        for i in range(5):
            agent = FishingAgent(
                boat_id=i,
                home_port=ports[i % len(ports)],
                env_width=100,
                env_height=100
            )
            agent_path = f'{checkpoint_path}agent_{i}_best.pt'
            agent.load_checkpoint(agent_path)
            agent.epsilon = 0.05  # Minimal exploration
            agents.append(agent)
        print(f"[OK] Loaded 5 trained models")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    energy_timeline = []
    population_timeline = []
    plankton_timeline = []
    total_catch = 0
    
    print(f"\n[ANALYSIS] Running {duration_steps:,} steps with 15 boats...")
    print(f"   Start: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy\n")
    
    for step in range(duration_steps):
        # Fleet actions
        actions = []
        
        for boat_idx in range(15):
            agent_idx = boat_idx // 3  # Which trained agent (0-4)
            agent = agents[agent_idx]
            
            # Get boat state (matching deploy_fleet format)
            boat_pos = fleet.positions[boat_idx]
            boat_state = {
                'position': boat_pos.copy(),
                'velocity': np.array([
                    np.cos(fleet.headings[boat_idx]) * fleet.velocities[boat_idx],
                    np.sin(fleet.headings[boat_idx]) * fleet.velocities[boat_idx]
                ]),
                'heading': fleet.headings[boat_idx],
                'fuel': fleet.fuel_levels[boat_idx],
                'max_fuel': fleet.max_fuel,
                'cargo': fleet.cargo_levels[boat_idx],
                'max_cargo': fleet.max_cargo,
                'net_deployed': fleet.nets_deployed[boat_idx],
                'current_temp': env.get_temperature(*boat_pos),
                'inside_mpa': env.is_in_mpa(*boat_pos)
            }
            
            # Get observation and action
            obs = agent.get_observation(boat_state, fish, env)
            action = agent.select_action(obs, boat_state)
            actions.append(action)
        
        actions = np.array(actions)
        
        # Fleet step
        fleet.step(actions, fish, env)
        
        # Ecosystem step
        env.step()
        fish.step(env)
        
        # Update MPAs
        if hasattr(env, 'update_mpas'):
            env.update_mpas(fish)
        
        # Track metrics
        total_energy = fish.energies.sum()
        avg_plankton = env.plankton_grid.mean()
        step_catch = fleet.cargo_sold.sum()
        total_catch += step_catch
        
        energy_timeline.append(total_energy)
        population_timeline.append(fish.num_schools)
        plankton_timeline.append(avg_plankton)
        
        if (step + 1) % (duration_steps // 20) == 0:
            print(f"  Step {step+1:>6} | Fish: {fish.num_schools:>5} | "
                  f"Energy: {total_energy:>10.1f} | Catch so far: {total_catch:>8.1f}t")
    
    print(f"\n[OK] Final: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy")
    print(f"   Total caught: {total_catch:.1f} tons")
    
    return {
        'energy': np.array(energy_timeline),
        'population': np.array(population_timeline),
        'plankton': np.array(plankton_timeline),
        'label': 'Fish + 15 Boats'
    }

def plot_comparison(data_alone, data_with_boats):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fish Ecosystem Energy Analysis: Alone vs With 15 Boats', fontsize=16, fontweight='bold')
    
    steps = np.arange(len(data_alone['energy']))
    hours = steps
    days = hours / 24
    
    # Plot 1: Total Energy Timeline
    ax = axes[0, 0]
    ax.plot(days, data_alone['energy'], label=data_alone['label'], linewidth=2, alpha=0.8)
    ax.plot(days, data_with_boats['energy'], label=data_with_boats['label'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Total Energy (all fish combined)')
    ax.set_title('Total Fish Energy Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Population Timeline
    ax = axes[0, 1]
    ax.plot(days, data_alone['population'], label=data_alone['label'], linewidth=2, alpha=0.8)
    ax.plot(days, data_with_boats['population'], label=data_with_boats['label'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Number of Fish Schools')
    ax.set_title('Fish Population Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Energy per Fish
    ax = axes[1, 0]
    energy_per_fish_alone = data_alone['energy'] / (data_alone['population'] + 1)
    energy_per_fish_boats = data_with_boats['energy'] / (data_with_boats['population'] + 1)
    ax.plot(days, energy_per_fish_alone, label=data_alone['label'], linewidth=2, alpha=0.8)
    ax.plot(days, energy_per_fish_boats, label=data_with_boats['label'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Average Energy per Fish School')
    ax.set_title('Energy per Fish School Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Plankton Timeline
    ax = axes[1, 1]
    ax.plot(days, data_alone['plankton'], label=data_alone['label'], linewidth=2, alpha=0.8)
    ax.plot(days, data_with_boats['plankton'], label=data_with_boats['label'], linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Average Plankton Abundance')
    ax.set_title('Plankton Availability Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fish_energy_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Graph: fish_energy_analysis.png")
    plt.show()

def print_statistics(data_alone, data_with_boats):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("ENERGY ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<35} {'Fish Alone':>15} {'With Boats':>15}")
    print("-" * 65)
    
    print(f"{'Initial Total Energy':<35} {data_alone['energy'][0]:>15.1f} {data_with_boats['energy'][0]:>15.1f}")
    print(f"{'Final Total Energy':<35} {data_alone['energy'][-1]:>15.1f} {data_with_boats['energy'][-1]:>15.1f}")
    print(f"{'Energy Change':<35} {data_alone['energy'][-1] - data_alone['energy'][0]:>15.1f} {data_with_boats['energy'][-1] - data_with_boats['energy'][0]:>15.1f}")
    
    print(f"\n{'Initial Population':<35} {data_alone['population'][0]:>15.0f} {data_with_boats['population'][0]:>15.0f}")
    print(f"{'Final Population':<35} {data_alone['population'][-1]:>15.0f} {data_with_boats['population'][-1]:>15.0f}")
    print(f"{'Population Change':<35} {data_alone['population'][-1] - data_alone['population'][0]:>15.0f} {data_with_boats['population'][-1] - data_with_boats['population'][0]:>15.0f}")
    
    print(f"\n{'Avg Plankton (Alone)':<35} {data_alone['plankton'].mean():>15.3f}")
    print(f"{'Avg Plankton (With Boats)':<35} {data_with_boats['plankton'].mean():>15.3f}")
    
    energy_loss_alone = data_alone['energy'][0] - data_alone['energy'][-1]
    energy_loss_boats = data_with_boats['energy'][0] - data_with_boats['energy'][-1]
    print(f"\n{'Total Energy Lost (Alone)':<35} {energy_loss_alone:>15.1f}")
    print(f"{'Total Energy Lost (With Boats)':<35} {energy_loss_boats:>15.1f}")
    print(f"{'Extra Loss from Fishing':<35} {energy_loss_boats - energy_loss_alone:>15.1f}")

if __name__ == "__main__":
    print("\n[FISH ENERGY ANALYSIS: ECOSYSTEM BASELINE VS FISHING IMPACT]\n")
    
    # Run both scenarios
    print("\n[Starting Scenario 1 (Ecosystem Alone)]...")
    data_alone = run_ecosystem_alone(duration_steps=8760)
    
    print("\n\n[Starting Scenario 2 (Ecosystem + Boats)]...")
    data_with_boats = run_ecosystem_with_boats(duration_steps=8760)
    
    if data_alone and data_with_boats:
        # Print statistics
        print_statistics(data_alone, data_with_boats)
        
        # Plot comparison
        print("\n[ANALYSIS] Generating comparison plots...")
        plot_comparison(data_alone, data_with_boats)
        
        print("\n[OK] Analysis complete!")
    else:
        print("\n[ERROR] Analysis failed")
