"""
Fishing Fleet RL Training System

This script trains fishing boat agents using Deep Q-Learning in a realistic
ocean ecosystem simulation. Boats learn to balance fish catching efficiency
with fuel consumption while navigating dynamic ocean conditions.
"""

import numpy as np
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from environment import OceanEnvironment
from fish_ecosystem import FishPopulation
from fleet_physics import FishingFleet
from fishing_agent import FishingAgent


class TrainingSession:
    def __init__(self, config):
        """Initialize training environment and agents"""
        self.config = config
        
        # Create environment
        self.env = OceanEnvironment(
            width=config['env_width'],
            height=config['env_height'],
            hours_per_tick=config['hours_per_tick']
        )
        
        # Generate initial random ports for first episode
        self.ports = self._generate_random_ports(config['num_ports'], config['env_width'], config['env_height'])
        
        # Create fish population
        self.fish = FishPopulation(
            initial_population=config['initial_fish'],
            env_width=config['env_width'],
            env_height=config['env_height']
        )
        
        # Create fishing fleet
        self.fleet = FishingFleet(
            num_boats=config['num_boats'],
            ports=self.ports,
            env_width=config['env_width'],
            env_height=config['env_height']
        )
        
        # Create agents (one per boat)
        self.agents = []
        for boat_id in range(config['num_boats']):
            agent = FishingAgent(
                boat_id=boat_id,
                home_port=self.ports[boat_id % len(self.ports)],  # Distribute boats across ports
                env_width=config['env_width'],
                env_height=config['env_height']
            )
            self.agents.append(agent)
        
        # Training tracking
        self.episode = 0
        self.global_step = 0
        self.episode_rewards = []
        self.episode_catches = []
        self.episode_trips = []
        self.training_losses = []
        
        # Performance metrics
        self.best_avg_reward = -float('inf')
        self.convergence_check = []
        
        print("\n" + "="*70)
        print("FISHING FLEET TRAINING SYSTEM INITIALIZED")
        print("="*70)
        print(f"Environment: {config['env_width']}x{config['env_height']} ocean grid")
        print(f"Fish Population: {config['initial_fish']} schools")
        print(f"Fleet Size: {config['num_boats']} boats")
        print(f"Ports: {config['num_ports']} locations (randomized each episode)")
        print(f"Episode Length: {config['steps_per_episode']} steps ({config['steps_per_episode'] * config['hours_per_tick']} simulated hours = {config['steps_per_episode']/24:.0f} days = {config['steps_per_episode']/168:.1f} weeks)")
        print(f"Temporal Variance: Episodes start at random weeks (0-51) for generalization")
        print(f"MPA System: Intelligent protection zones update every 30 days")
        print("="*70 + "\n")
    
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
    
    def _generate_random_ports(self, num_ports, width, height):
        """Generate random port locations with minimum spacing"""
        ports = []
        min_spacing = 20.0  # Minimum distance between ports
        margin = 10.0  # Keep ports away from edges
        
        for _ in range(num_ports):
            max_attempts = 100
            for attempt in range(max_attempts):
                x = np.random.uniform(margin, width - margin)
                y = np.random.uniform(margin, height - margin)
                candidate = [x, y]
                
                # Check spacing from existing ports
                if len(ports) == 0:
                    ports.append(candidate)
                    break
                
                min_dist = min([np.linalg.norm(np.array(candidate) - np.array(p)) for p in ports])
                if min_dist >= min_spacing:
                    ports.append(candidate)
                    break
        
        return ports
    
    def run_episode(self, episode_num):
        """Run a single training episode"""
        episode_start = time.time()
        
        # Randomize port locations each episode for generalization
        self.ports = self._generate_random_ports(self.config['num_ports'], self.config['env_width'], self.config['env_height'])
        
        # Update agent home ports to new randomized locations
        for boat_id, agent in enumerate(self.agents):
            agent.home_port = np.array(self.ports[boat_id % len(self.ports)])
        
        # TEMPORAL RANDOMIZATION: Start at random week of year for generalization
        # This teaches agents to handle different seasons/conditions
        random_week = np.random.randint(0, 52)  # Week 0-51 of year
        time_offset = random_week * 168  # Convert to hours
        
        # Reset environment for new episode
        self.env = OceanEnvironment(
            width=self.config['env_width'],
            height=self.config['env_height'],
            hours_per_tick=self.config['hours_per_tick']
        )
        
        # Fast-forward environment to random starting week
        # This establishes ecosystem state for that time of year
        self.env.time_step = time_offset
        
        # Pre-warm MPAs based on starting week (every 30 days MPAs update)
        # Calculate how many MPA updates should have occurred
        mpa_updates_due = time_offset // 720
        if mpa_updates_due > 0:
            # Align environment's MPA update tracking
            self.env.time_step = (mpa_updates_due * 720)  # Set to last MPA update boundary
        
        # Reset fish with some variation
        fish_variation = int(np.random.uniform(-200, 200))
        self.fish = FishPopulation(
            initial_population=max(100, self.config['initial_fish'] + fish_variation),
            env_width=self.config['env_width'],
            env_height=self.config['env_height']
        )
        
        # Reset fleet with new randomized ports
        self.fleet = FishingFleet(
            num_boats=self.config['num_boats'],
            ports=self.ports,
            env_width=self.config['env_width'],
            env_height=self.config['env_height']
        )
        
        # Reset fleet positions near new ports
        for i, agent in enumerate(self.agents):
            port_idx = i % len(self.ports)
            self.fleet.positions[i] = self.ports[port_idx] + (np.random.rand(2) - 0.5) * 3.0
            self.fleet.fuel_levels[i] = self.fleet.max_fuel
            self.fleet.cargo_levels[i] = 0.0
            self.fleet.velocities[i] = 0.0
            self.fleet.headings[i] = np.random.rand() * 2 * np.pi
        
        # Episode tracking
        episode_reward = np.zeros(self.config['num_boats'])
        episode_total_catch = np.zeros(self.config['num_boats'])
        episode_trips = np.zeros(self.config['num_boats'])
        
        # Store previous states for experience replay
        prev_states = []
        prev_observations = []
        prev_actions = []
        
        for boat_idx in range(self.config['num_boats']):
            state = self.get_boat_state(boat_idx)
            obs = self.agents[boat_idx].get_observation(state, self.fish, self.env)
            prev_states.append(state)
            prev_observations.append(obs)
            prev_actions.append(None)
        
        # Run episode steps
        for step in range(self.config['steps_per_episode']):
            # Get actions from all agents
            actions = []
            for boat_idx in range(self.config['num_boats']):
                state = self.get_boat_state(boat_idx)
                obs = self.agents[boat_idx].get_observation(state, self.fish, self.env)
                action = self.agents[boat_idx].select_action(obs, state)
                actions.append(action)
            
            actions = np.array(actions)
            
            # Step fleet physics
            self.fleet.step(actions, self.fish, self.env)
            
            # Step ecosystem
            self.fish.step(self.env)
            self.env.step()
            
            # MPA system updates (every 30 days based on fish density)
            self.env.update_mpas(self.fish)
            
            # Calculate rewards and store experiences
            for boat_idx in range(self.config['num_boats']):
                new_state = self.get_boat_state(boat_idx)
                new_obs = self.agents[boat_idx].get_observation(new_state, self.fish, self.env)
                
                # Calculate reward
                reward = self.agents[boat_idx].calculate_reward(
                    prev_states[boat_idx],
                    actions[boat_idx],
                    new_state,
                    self.fleet.just_sold[boat_idx],
                    self.fleet.cargo_sold[boat_idx]
                )
                
                episode_reward[boat_idx] += reward
                episode_total_catch[boat_idx] += self.fleet.cargo_sold[boat_idx]
                if self.fleet.just_sold[boat_idx]:
                    episode_trips[boat_idx] += 1
                
                # Store experience
                done = (step == self.config['steps_per_episode'] - 1)
                self.agents[boat_idx].store_experience(
                    prev_observations[boat_idx],
                    actions[boat_idx],
                    reward,
                    new_obs,
                    done
                )
                
                # Update previous state
                prev_states[boat_idx] = new_state
                prev_observations[boat_idx] = new_obs
            
            # Train agents periodically
            if step % self.config['train_frequency'] == 0:
                for agent in self.agents:
                    loss = agent.train_step()
                    if loss is not None:
                        self.training_losses.append(loss)
            
            self.global_step += 1
        
        # Update target networks periodically
        if episode_num % self.config['target_update_frequency'] == 0:
            for agent in self.agents:
                agent.update_target_network()
        
        # Episode statistics
        episode_time = time.time() - episode_start
        avg_reward = np.mean(episode_reward)
        total_catch = np.sum(episode_total_catch)
        total_trips = np.sum(episode_trips)
        
        self.episode_rewards.append(avg_reward)
        self.episode_catches.append(total_catch)
        self.episode_trips.append(total_trips)
        
        # Convergence tracking
        if len(self.episode_rewards) >= 10:
            recent_avg = np.mean(self.episode_rewards[-10:])
            self.convergence_check.append(recent_avg)
            
            if recent_avg > self.best_avg_reward:
                self.best_avg_reward = recent_avg
                self.save_best_agents()
        
        return {
            'episode': episode_num,
            'time': episode_time,
            'avg_reward': avg_reward,
            'total_catch': total_catch,
            'total_trips': total_trips,
            'fish_remaining': self.fish.num_schools,
            'avg_epsilon': np.mean([agent.epsilon for agent in self.agents]),
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            'avg_cargo': np.mean(self.fleet.cargo_levels),
            'avg_fuel': np.mean(self.fleet.fuel_levels)
        }
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['num_episodes']} episodes...\n")
        
        training_start = time.time()
        
        for episode in range(1, self.config['num_episodes'] + 1):
            self.episode = episode
            
            # Run episode
            stats = self.run_episode(episode)
            
            # Progress reporting
            if episode % self.config['log_frequency'] == 0 or episode == 1:
                self._print_progress(episode, stats, training_start)
            
            # Save checkpoint
            if episode % self.config['save_frequency'] == 0:
                self.save_checkpoint(episode)
        
        # Training complete
        training_time = time.time() - training_start
        self._print_final_summary(training_time)
        
        # Save final models
        self.save_checkpoint('final')
        
        # Generate training plots
        self.plot_training_results()
    
    def _print_progress(self, episode, stats, training_start):
        """Print formatted progress update"""
        elapsed = time.time() - training_start
        eps_per_sec = episode / elapsed
        remaining_eps = self.config['num_episodes'] - episode
        eta_seconds = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Episode {episode}/{self.config['num_episodes']} " +
              f"({100*episode/self.config['num_episodes']:.1f}% complete)")
        print(f"{'='*70}")
        print(f"⏱️  Episode Time: {stats['time']:.2f}s | ETA: {timedelta(seconds=int(eta_seconds))}")
        print(f"🎯 Avg Reward: {stats['avg_reward']:+.2f} | Best 10-ep Avg: {self.best_avg_reward:+.2f}")
        print(f"🐟 Fish Caught: {stats['total_catch']:.1f} tons | Trips: {int(stats['total_trips'])}")
        print(f"🧠 Exploration ε: {stats['avg_epsilon']:.3f} | Loss: {stats['avg_loss']:.4f}")
        print(f"🌊 Fish Population: {stats['fish_remaining']} schools")
        print(f"🚢 Avg Cargo: {stats['avg_cargo']:.1f}t | Avg Fuel: {stats['avg_fuel']:.0f}L")
        print(f"{'='*70}")
    
    def _print_final_summary(self, training_time):
        """Print training completion summary"""
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Total Time: {timedelta(seconds=int(training_time))}")
        print(f"Episodes: {self.config['num_episodes']}")
        print(f"Total Steps: {self.global_step:,}")
        print(f"Avg Time/Episode: {training_time/self.config['num_episodes']:.2f}s")
        print(f"\nFinal Performance (last 10 episodes):")
        print(f"  Avg Reward: {np.mean(self.episode_rewards[-10:]):+.2f}")
        print(f"  Avg Catch: {np.mean(self.episode_catches[-10:]):.1f} tons/episode")
        print(f"  Avg Trips: {np.mean(self.episode_trips[-10:]):.1f} trips/episode")
        print(f"\nBest Performance:")
        print(f"  Best 10-ep Avg Reward: {self.best_avg_reward:+.2f}")
        print(f"  Best Single Episode: {max(self.episode_rewards):+.2f}")
        print(f"  Max Catch: {max(self.episode_catches):.1f} tons")
        
        # Agent statistics
        print(f"\nAgent Statistics:")
        for i, agent in enumerate(self.agents):
            stats = agent.get_stats()
            print(f"  Boat {i}: {stats['trips_completed']} trips, " +
                  f"{stats['total_catch']:.1f} tons total")
        print("="*70 + "\n")
    
    def save_checkpoint(self, episode):
        """Save training checkpoint"""
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            filepath = f"{checkpoint_dir}/agent_{i}_ep{episode}.pt"
            agent.save_checkpoint(filepath)
        
        # Save training metadata
        metadata_path = f"{checkpoint_dir}/training_metadata_ep{episode}.npy"
        np.save(metadata_path, {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_catches': self.episode_catches,
            'episode_trips': self.episode_trips,
            'best_avg_reward': self.best_avg_reward
        })
        
        print(f"✅ Checkpoint saved: Episode {episode}")
    
    def save_best_agents(self):
        """Save best performing agents"""
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            filepath = f"{checkpoint_dir}/agent_{i}_best.pt"
            agent.save_checkpoint(filepath)
    
    def plot_training_results(self):
        """Generate training visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fishing Agent Training Results', fontsize=16)
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # Subplot 1: Rewards
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= 10:
            smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(10, len(self.episode_rewards) + 1), smoothed, 
                           'r-', linewidth=2, label='10-Episode Moving Avg')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Learning Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Subplot 2: Catch Performance
        axes[0, 1].plot(episodes, self.episode_catches, 'g-', alpha=0.6)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Catch (tons)')
        axes[0, 1].set_title('Fish Catch per Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Subplot 3: Trips Completed
        axes[1, 0].plot(episodes, self.episode_trips, 'b-', alpha=0.6)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Trips Completed')
        axes[1, 0].set_title('Port Returns per Episode')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Subplot 4: Exploration Rate
        eps_values = []
        for ep in episodes:
            if ep <= len(self.agents):
                eps_values.append(1.0)
            else:
                eps_values.append(max(0.05, 1.0 * (0.9995 ** ep)))
        axes[1, 1].plot(episodes, eps_values, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon (Exploration Rate)')
        axes[1, 1].set_title('Exploration Decay')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150)
        print("📊 Training plots saved to 'training_results.png'")
        plt.close()


def estimate_training_time(config):
    """Provide time estimates for training configuration"""
    # Rough estimates based on typical performance
    steps_per_second = 50  # Conservative estimate for simulation speed
    episode_seconds = config['steps_per_episode'] / steps_per_second
    total_seconds = episode_seconds * config['num_episodes']
    
    print("\n" + "="*70)
    print("TRAINING TIME ESTIMATES")
    print("="*70)
    print(f"Configuration:")
    print(f"  • {config['num_episodes']} episodes")
    print(f"  • {config['steps_per_episode']} steps/episode")
    print(f"  • {config['num_boats']} boats")
    print(f"\nEstimated Times:")
    print(f"  • Per Episode: ~{episode_seconds:.1f} seconds")
    print(f"  • Total Training: ~{timedelta(seconds=int(total_seconds))}")
    print(f"  • Simulated Time: {config['num_episodes'] * config['steps_per_episode'] * config['hours_per_tick']:,} hours " +
          f"({config['num_episodes'] * config['steps_per_episode'] * config['hours_per_tick'] / 8760:.1f} years)")
    print("\nNote: Actual time depends on your CPU/GPU performance.")
    print("First few episodes may be slower as PyTorch optimizes.")
    print("="*70 + "\n")


if __name__ == "__main__":
    # ===================================
    # TRAINING CONFIGURATION
    # ===================================
    
    config = {
        # Environment
        'env_width': 100,
        'env_height': 100,
        'hours_per_tick': 1,
        'initial_fish': 2500,  # Increased from 1000 - realistic density 0.25/unit² for larger world
        
        # Fleet
        'num_boats': 5,  # 5 agents learning seasonal patterns
        'ports': 'random',  # Will be randomized each episode for generalization
        'num_ports': 5,  # One port per agent
        
        # Training
        'num_episodes': 500,        # 500 episodes of 1 week each
        'steps_per_episode': 168,    # 1 week = 7 days * 24 hours (faster training, random temporal start)
        'train_frequency': 4,       # Train every 4 steps
        'target_update_frequency': 10,  # Update target network every 10 episodes
        
        # Logging
        'log_frequency': 5,         # Print progress every 5 episodes
        'save_frequency': 50,       # Save checkpoint every 50 episodes
    }
    
    # Show time estimates
    estimate_training_time(config)
    
    # Confirm before starting
    print("Ready to begin training.")
    print("Checkpoints will be saved to ./checkpoints/")
    print("\nPress Ctrl+C at any time to stop training (progress will be saved).\n")
    
    try:
        input("Press Enter to start training...")
    except KeyboardInterrupt:
        print("\nTraining cancelled.")
        exit(0)
    
    # Run training
    try:
        session = TrainingSession(config)
        session.train()
        
        print("\n🎉 Training completed successfully!")
        print("Load best agents from './checkpoints/agent_X_best.pt'")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("Progress has been saved. You can resume by adjusting the config.")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
