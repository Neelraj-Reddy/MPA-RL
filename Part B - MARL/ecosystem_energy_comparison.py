"""
Comprehensive Ecosystem Energy Analysis

Compares three scenarios to understand system dynamics:
1. Fish Alone (baseline - no boats, no MPAs)
2. Fish + Boats (fishing impact without protection)
3. Fish + Boats + MPAs (fishing with marine protected areas)

This helps identify:
- Natural ecosystem energy dynamics
- Impact of fishing pressure
- Effectiveness of MPA protection
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import OceanEnvironment
from fish_ecosystem import FishPopulation
from fleet_physics import FishingFleet
from fishing_agent import FishingAgent
import torch


def run_fish_alone(duration_steps=8760):
    """SCENARIO 1: Fish ecosystem alone (baseline)"""
    print("="*70)
    print("SCENARIO 1: FISH ALONE (No Boats, No MPAs)")
    print("="*70)
    
    env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
    fish = FishPopulation(initial_population=2500, env_width=100, env_height=100)
    
    energy_timeline = []
    population_timeline = []
    plankton_timeline = []
    mpa_coverage_timeline = []
    catch_timeline = []
    
    print(f"\n[ANALYSIS] Running {duration_steps:,} steps...")
    print(f"   Start: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy\n")
    
    for step in range(duration_steps):
        # Ecosystem step only
        env.step()
        fish.step(env)
        
        # Track metrics
        total_energy = fish.energies.sum()
        avg_plankton = env.plankton_grid.mean()
        mpa_coverage = 0.0  # No MPAs in this scenario
        
        energy_timeline.append(total_energy)
        population_timeline.append(fish.num_schools)
        plankton_timeline.append(avg_plankton)
        mpa_coverage_timeline.append(mpa_coverage)
        catch_timeline.append(0.0)
        
        if (step + 1) % (duration_steps // 20) == 0:
            print(f"  Step {step+1:>6} | Fish: {fish.num_schools:>5} | "
                  f"Energy: {total_energy:>10.1f} | Plankton: {avg_plankton:.3f}")
    
    print(f"\n[OK] Final: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy")
    
    return {
        'energy': np.array(energy_timeline),
        'population': np.array(population_timeline),
        'plankton': np.array(plankton_timeline),
        'mpa_coverage': np.array(mpa_coverage_timeline),
        'catch_timeline': np.array(catch_timeline),
        'total_catch': 0.0,
        'label': 'Fish Alone',
        'color': '#2ecc71'  # Green
    }


def run_fish_with_boats(duration_steps=8760, checkpoint_path='checkpoints/'):
    """SCENARIO 2: Fish + Boats (no MPAs)"""
    print("\n" + "="*70)
    print("SCENARIO 2: FISH + BOATS (No MPAs)")
    print("="*70)
    
    env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
    fish = FishPopulation(initial_population=2500, env_width=100, env_height=100)
    
    # Setup fleet
    ports = [(20, 20), (80, 20), (50, 50), (20, 80), (80, 80)]
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
            agent.epsilon = 0.05
            agents.append(agent)
        print(f"[OK] Loaded 5 trained models for 15 boats")
    except Exception as e:
        print(f"[ERROR] Could not load models: {e}")
        return None
    
    energy_timeline = []
    population_timeline = []
    plankton_timeline = []
    mpa_coverage_timeline = []
    catch_timeline = []
    total_catch = 0
    
    print(f"\n[ANALYSIS] Running {duration_steps:,} steps with 15 boats (NO MPAs)...")
    print(f"   Start: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy\n")
    
    for step in range(duration_steps):
        # Fleet actions
        actions = []
        for boat_idx in range(15):
            agent_idx = boat_idx // 3
            agent = agents[agent_idx]
            
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
                'inside_mpa': False  # No MPAs in this scenario
            }
            
            obs = agent.get_observation(boat_state, fish, env)
            action = agent.select_action(obs, boat_state)
            actions.append(action)
        
        actions = np.array(actions)
        
        # Execute fleet step
        fleet.step(actions, fish, env)
        
        # Ecosystem step (NO MPA updates)
        env.step()
        fish.step(env)
        
        # Track metrics
        total_energy = fish.energies.sum()
        avg_plankton = env.plankton_grid.mean()
        step_catch = fleet.cargo_sold.sum()
        total_catch += step_catch
        mpa_coverage = 0.0  # No MPAs
        
        energy_timeline.append(total_energy)
        population_timeline.append(fish.num_schools)
        plankton_timeline.append(avg_plankton)
        mpa_coverage_timeline.append(mpa_coverage)
        catch_timeline.append(total_catch)
        
        if (step + 1) % (duration_steps // 20) == 0:
            print(f"  Step {step+1:>6} | Fish: {fish.num_schools:>5} | "
                  f"Energy: {total_energy:>10.1f} | Catch: {total_catch:>8.1f}t")
    
    print(f"\n[OK] Final: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy")
    print(f"   Total caught: {total_catch:.1f} tons")
    
    return {
        'energy': np.array(energy_timeline),
        'population': np.array(population_timeline),
        'plankton': np.array(plankton_timeline),
        'mpa_coverage': np.array(mpa_coverage_timeline),
        'catch_timeline': np.array(catch_timeline),
        'total_catch': total_catch,
        'label': 'Fish + Boats (No MPAs)',
        'color': '#e74c3c'  # Red
    }


def run_fish_boats_with_mpas(duration_steps=8760, checkpoint_path='checkpoints/'):
    """SCENARIO 3: Fish + Boats + MPAs"""
    print("\n" + "="*70)
    print("SCENARIO 3: FISH + BOATS + MPAs")
    print("="*70)
    
    env = OceanEnvironment(width=100, height=100, hours_per_tick=1)
    fish = FishPopulation(initial_population=2500, env_width=100, env_height=100)
    
    # Setup fleet
    ports = [(20, 20), (80, 20), (50, 50), (20, 80), (80, 80)]
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
            agent.epsilon = 0.05
            agents.append(agent)
        print(f"[OK] Loaded 5 trained models for 15 boats")
    except Exception as e:
        print(f"[ERROR] Could not load models: {e}")
        return None
    
    energy_timeline = []
    population_timeline = []
    plankton_timeline = []
    mpa_coverage_timeline = []
    catch_timeline = []
    total_catch = 0
    
    print(f"\n[ANALYSIS] Running {duration_steps:,} steps with 15 boats + MPAs...")
    print(f"   Start: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy\n")
    
    for step in range(duration_steps):
        # Fleet actions
        actions = []
        for boat_idx in range(15):
            agent_idx = boat_idx // 3
            agent = agents[agent_idx]
            
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
            
            obs = agent.get_observation(boat_state, fish, env)
            action = agent.select_action(obs, boat_state)
            actions.append(action)
        
        actions = np.array(actions)
        
        # Execute fleet step
        fleet.step(actions, fish, env)
        
        # Ecosystem step
        env.step()
        fish.step(env)
        
        # UPDATE MPAs (this is the key difference!)
        env.update_mpas(fish)
        
        # Track metrics
        total_energy = fish.energies.sum()
        avg_plankton = env.plankton_grid.mean()
        step_catch = fleet.cargo_sold.sum()
        total_catch += step_catch
        mpa_coverage = (env.mpa_grid > 0.5).sum() / (env.width * env.height)
        
        energy_timeline.append(total_energy)
        population_timeline.append(fish.num_schools)
        plankton_timeline.append(avg_plankton)
        mpa_coverage_timeline.append(mpa_coverage)
        catch_timeline.append(total_catch)
        
        if (step + 1) % (duration_steps // 20) == 0:
            print(f"  Step {step+1:>6} | Fish: {fish.num_schools:>5} | "
                  f"Energy: {total_energy:>10.1f} | Catch: {total_catch:>8.1f}t | MPA: {mpa_coverage*100:.1f}%")
    
    print(f"\n[OK] Final: {fish.num_schools} schools, {fish.energies.sum():.1f} total energy")
    print(f"   Total caught: {total_catch:.1f} tons")
    print(f"   Avg MPA coverage: {np.mean(mpa_coverage_timeline)*100:.1f}%")
    
    return {
        'energy': np.array(energy_timeline),
        'population': np.array(population_timeline),
        'plankton': np.array(plankton_timeline),
        'mpa_coverage': np.array(mpa_coverage_timeline),
        'catch_timeline': np.array(catch_timeline),
        'total_catch': total_catch,
        'label': 'Fish + Boats + MPAs',
        'color': '#3498db'  # Blue
    }


def plot_three_way_comparison(data1, data2, data3):
    """Create comprehensive comparison plots for all three scenarios"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Ecosystem Energy Analysis: Three-Way Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    steps = np.arange(len(data1['energy']))
    days = steps / 24
    
    datasets = [data1, data2, data3]
    
    # Plot 1: Total Energy Timeline
    ax = fig.add_subplot(gs[0, 0])
    for data in datasets:
        ax.plot(days, data['energy'], label=data['label'], 
                linewidth=2.5, alpha=0.8, color=data['color'])
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Total Energy', fontsize=11)
    ax.set_title('Total Fish Energy Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Population Timeline
    ax = fig.add_subplot(gs[0, 1])
    for data in datasets:
        ax.plot(days, data['population'], label=data['label'], 
                linewidth=2.5, alpha=0.8, color=data['color'])
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Number of Fish Schools', fontsize=11)
    ax.set_title('Fish Population Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Energy per Fish
    ax = fig.add_subplot(gs[0, 2])
    for data in datasets:
        energy_per_fish = data['energy'] / (data['population'] + 1)
        ax.plot(days, energy_per_fish, label=data['label'], 
                linewidth=2.5, alpha=0.8, color=data['color'])
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Avg Energy per Fish', fontsize=11)
    ax.set_title('Energy per Fish School', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Plankton Timeline
    ax = fig.add_subplot(gs[1, 0])
    for data in datasets:
        ax.plot(days, data['plankton'], label=data['label'], 
                linewidth=2.5, alpha=0.8, color=data['color'])
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Avg Plankton Abundance', fontsize=11)
    ax.set_title('Plankton Availability', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: MPA Coverage (only relevant for scenario 3)
    ax = fig.add_subplot(gs[1, 1])
    for data in datasets:
        if data['mpa_coverage'].max() > 0:
            ax.plot(days, data['mpa_coverage'] * 100, label=data['label'], 
                    linewidth=2.5, alpha=0.8, color=data['color'])
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('MPA Coverage (%)', fontsize=11)
    ax.set_title('Marine Protected Area Coverage', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=15, color='gray', linestyle='--', alpha=0.5, label='Target 15%')
    
    # Plot 6: Energy Change Rate (derivative)
    ax = fig.add_subplot(gs[1, 2])
    for data in datasets:
        energy_change = np.diff(data['energy'], prepend=data['energy'][0])
        # Smooth with moving average
        window = 24  # 1 day
        smoothed = np.convolve(energy_change, np.ones(window)/window, mode='same')
        ax.plot(days, smoothed, label=data['label'], 
                linewidth=2, alpha=0.8, color=data['color'])
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Energy Change Rate', fontsize=11)
    ax.set_title('Energy Change Rate (24h smoothed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Cumulative Energy Loss
    ax = fig.add_subplot(gs[2, 0])
    for data in datasets:
        initial_energy = data['energy'][0]
        cumulative_loss = initial_energy - data['energy']
        ax.plot(days, cumulative_loss, label=data['label'], 
                linewidth=2.5, alpha=0.8, color=data['color'])
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Cumulative Energy Loss', fontsize=11)
    ax.set_title('Cumulative Energy Lost from Start', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Population Sustainability Index
    ax = fig.add_subplot(gs[2, 1])
    for data in datasets:
        # Sustainability = current pop / initial pop
        sustainability = data['population'] / data['population'][0]
        ax.plot(days, sustainability, label=data['label'], 
                linewidth=2.5, alpha=0.8, color=data['color'])
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='80% threshold')
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Population Ratio', fontsize=11)
    ax.set_title('Population Sustainability (vs Initial)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Fish Catch Comparison (with and without MPAs)
    ax = fig.add_subplot(gs[2, 2])
    no_mpa = data2
    with_mpa = data3
    ax.plot(days, no_mpa['catch_timeline'], label='Catch (No MPAs)',
            linewidth=2.5, alpha=0.9, color=no_mpa['color'])
    ax.plot(days, with_mpa['catch_timeline'], label='Catch (With MPAs)',
            linewidth=2.5, alpha=0.9, color=with_mpa['color'])
    ax.set_xlabel('Time (Days)', fontsize=11)
    ax.set_ylabel('Cumulative Catch (tons)', fontsize=11)
    ax.set_title('Fish Catch: With vs Without MPAs', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.savefig('ecosystem_energy_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n[SAVED] Graph: ecosystem_energy_comparison.png")
    plt.show()


def print_comprehensive_statistics(data1, data2, data3):
    """Print detailed comparison statistics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE ENERGY ANALYSIS - THREE SCENARIOS")
    print("="*80)
    
    datasets = [data1, data2, data3]
    
    # Energy Statistics
    print("\n┌─ ENERGY METRICS " + "─"*60 + "┐")
    print(f"│ {'Metric':<30} │ {'Fish Alone':>14} │ {'Fish+Boats':>14} │ {'Fish+Boats+MPA':>14} │")
    print("├" + "─"*32 + "┼" + "─"*16 + "┼" + "─"*16 + "┼" + "─"*16 + "┤")
    
    metrics = [
        ('Initial Energy', [d['energy'][0] for d in datasets]),
        ('Final Energy', [d['energy'][-1] for d in datasets]),
        ('Energy Lost', [d['energy'][0] - d['energy'][-1] for d in datasets]),
        ('% Energy Retained', [(d['energy'][-1]/d['energy'][0])*100 for d in datasets])
    ]
    
    for metric_name, values in metrics:
        print(f"│ {metric_name:<30} │ {values[0]:>14.1f} │ {values[1]:>14.1f} │ {values[2]:>14.1f} │")
    print("└" + "─"*32 + "┴" + "─"*16 + "┴" + "─"*16 + "┴" + "─"*16 + "┘")
    
    # Population Statistics
    print("\n┌─ POPULATION METRICS " + "─"*56 + "┐")
    print(f"│ {'Metric':<30} │ {'Fish Alone':>14} │ {'Fish+Boats':>14} │ {'Fish+Boats+MPA':>14} │")
    print("├" + "─"*32 + "┼" + "─"*16 + "┼" + "─"*16 + "┼" + "─"*16 + "┤")
    
    metrics = [
        ('Initial Population', [d['population'][0] for d in datasets]),
        ('Final Population', [d['population'][-1] for d in datasets]),
        ('Population Change', [d['population'][-1] - d['population'][0] for d in datasets]),
        ('% Sustained', [(d['population'][-1]/d['population'][0])*100 for d in datasets])
    ]
    
    for metric_name, values in metrics:
        print(f"│ {metric_name:<30} │ {values[0]:>14.0f} │ {values[1]:>14.0f} │ {values[2]:>14.0f} │")
    print("└" + "─"*32 + "┴" + "─"*16 + "┴" + "─"*16 + "┴" + "─"*16 + "┘")
    
    # Fishing Impact
    print("\n┌─ FISHING METRICS " + "─"*59 + "┐")
    print(f"│ {'Metric':<30} │ {'Fish Alone':>14} │ {'Fish+Boats':>14} │ {'Fish+Boats+MPA':>14} │")
    print("├" + "─"*32 + "┼" + "─"*16 + "┼" + "─"*16 + "┼" + "─"*16 + "┤")
    
    print(f"│ {'Total Catch (tons)':<30} │ {data1['total_catch']:>14.1f} │ {data2['total_catch']:>14.1f} │ {data3['total_catch']:>14.1f} │")
    print(f"│ {'Avg MPA Coverage (%)':<30} │ {data1['mpa_coverage'].mean()*100:>14.1f} │ {data2['mpa_coverage'].mean()*100:>14.1f} │ {data3['mpa_coverage'].mean()*100:>14.1f} │")
    print("└" + "─"*32 + "┴" + "─"*16 + "┴" + "─"*16 + "┴" + "─"*16 + "┘")
    
    # Ecosystem Health Comparison
    print("\n┌─ ECOSYSTEM HEALTH ANALYSIS " + "─"*48 + "┐")
    
    # Compare scenario 2 vs 1 (boats impact)
    energy_loss_boats = (data2['energy'][-1] - data1['energy'][-1])
    pop_loss_boats = (data2['population'][-1] - data1['population'][-1])
    
    # Compare scenario 3 vs 2 (MPA benefit)
    energy_recovery_mpa = (data3['energy'][-1] - data2['energy'][-1])
    pop_recovery_mpa = (data3['population'][-1] - data2['population'][-1])
    
    # Compare scenario 3 vs 1 (net impact with MPAs)
    net_energy_impact = (data3['energy'][-1] - data1['energy'][-1])
    net_pop_impact = (data3['population'][-1] - data1['population'][-1])
    
    print(f"│ Fishing Impact (Boats vs Baseline):")
    print(f"│   • Energy difference: {energy_loss_boats:+.1f} ({energy_loss_boats/data1['energy'][-1]*100:+.1f}%)")
    print(f"│   • Population difference: {pop_loss_boats:+.0f} fish ({pop_loss_boats/data1['population'][-1]*100:+.1f}%)")
    print(f"│")
    print(f"│ MPA Protection Effect (MPAs vs No MPAs):")
    print(f"│   • Energy recovered: {energy_recovery_mpa:+.1f} ({energy_recovery_mpa/data2['energy'][-1]*100:+.1f}%)")
    print(f"│   • Population recovered: {pop_recovery_mpa:+.0f} fish ({pop_recovery_mpa/data2['population'][-1]*100:+.1f}%)")
    print(f"│")
    print(f"│ Net Impact (Boats+MPAs vs Baseline):")
    print(f"│   • Energy difference: {net_energy_impact:+.1f} ({net_energy_impact/data1['energy'][-1]*100:+.1f}%)")
    print(f"│   • Population difference: {net_pop_impact:+.0f} fish ({net_pop_impact/data1['population'][-1]*100:+.1f}%)")
    print("└" + "─"*76 + "┘")
    
    # Sustainability Assessment
    print("\n┌─ SUSTAINABILITY ASSESSMENT " + "─"*47 + "┐")
    for data in datasets:
        sustainability = (data['population'][-1] / data['population'][0]) * 100
        status = "✓ SUSTAINABLE" if sustainability >= 80 else "✗ DECLINING"
        print(f"│ {data['label']:<25} │ {sustainability:>6.1f}% population │ {status:>18} │")
    print("└" + "─"*76 + "┘")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE ECOSYSTEM ENERGY ANALYSIS")
    print("Three-Scenario Comparison: Fish Alone | Fish+Boats | Fish+Boats+MPAs")
    print("="*80)
    
    duration = 8760  # 1 year = 365 days * 24 hours
    
    # Run all three scenarios
    print("\n[Starting Scenario 1: Fish Alone]")
    data_fish_alone = run_fish_alone(duration_steps=duration)
    
    print("\n[Starting Scenario 2: Fish + Boats (No MPAs)]")
    data_fish_boats = run_fish_with_boats(duration_steps=duration)
    
    print("\n[Starting Scenario 3: Fish + Boats + MPAs]")
    data_fish_boats_mpas = run_fish_boats_with_mpas(duration_steps=duration)
    
    # Analyze results
    if data_fish_alone and data_fish_boats and data_fish_boats_mpas:
        print_comprehensive_statistics(data_fish_alone, data_fish_boats, data_fish_boats_mpas)
        
        print("\n[ANALYSIS] Generating comprehensive comparison plots...")
        plot_three_way_comparison(data_fish_alone, data_fish_boats, data_fish_boats_mpas)
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80)
        print("\nKey Insights:")
        print("  1. Fish Alone: Natural ecosystem dynamics without human intervention")
        print("  2. Fish + Boats: Impact of fishing pressure on population and energy")
        print("  3. Fish + Boats + MPAs: Effectiveness of marine protection in sustaining stocks")
        print("\nCheck 'ecosystem_energy_comparison.png' for visual analysis")
        print("="*80 + "\n")
    else:
        print("\n[ERROR] One or more scenarios failed to complete")
