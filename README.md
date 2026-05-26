Architecture (components & data flow)
------------------------------------
3. Analysis & visualization
	- Analysis scripts (e.g., `fish_energy_analysis.py`, `ecosystem_energy_comparison.py`) process CSV outputs and generate figures or summary statistics.
	- Visualization (e.g., `make_gif.py`, `visualize_fish_mpa_dynamics.py`) produces GIFs or animations of agent/fish dynamics.

Typical experiments
-------------------
AI & Models (summary)
----------------------
Hybrid control notes
--------------------
Practical notes
---------------
Next steps I can take
---------------------
If you want one of these, tell me which and I will implement it.

Architecture — Detailed
-----------------------

This section expands the high-level architecture into concrete modules, class interactions, data flows and operational pipelines so you (or collaborators) can extend, test, or reproduce experiments easily.

1) Component map (logical modules)
	 - Simulation Core (Part A)
		 • `fleet_physics.py` — physical model for vessel motion and fuel consumption. Exposes `FishingFleet.step(actions, fish_pop, env)` to apply agent actions and update position, velocity, fuel, cargo, and port interactions.
		 • `fish_ecosystem.py` — fish population and biomass dynamics, sampling/consumption APIs like `consume_plankton` or fish-school interactions.
		 • Scenario drivers (`final1.py`, `final2.py`) — orchestration scripts that create environments, fleets, MPAs, and write CSV outputs. They act as canonical experiments for reproducibility.

	 - RL Layer (Part B)
		 • `environment.py` — adapter turning simulator state into agent observations and exposing utility queries (temperature, current, MPA status). Also responsible for advancing temporal state (`env.step()`).
		 • `fishing_agent.py` — per-boat hybrid agent: DQN policy net + rule-based safety controllers (navigation, net safety). Implements observation assembly, action selection, experience replay and training steps.
		 • `train_fishing_agents.py` — training orchestration (episode lifecycle, randomization, logging, checkpointing). Coordinates environment, fish population, fleet and agents.

	 - Analysis & Visualization
		 • `fish_energy_analysis.py`, `ecosystem_energy_comparison.py` — post-processing of CSV outputs; compute aggregated metrics (fuel per ton, yield distribution, spatial catch heatmaps).
		 • `make_gif.py`, `visualize_fish_mpa_dynamics.py` — create visual artifacts to inspect agent behavior over time.

2) Dataflow (per-episode)
	 - Initialization: scenario runner or training session constructs `OceanEnvironment`, `FishPopulation`, `FishingFleet`, and agents. Ports may be randomized to encourage generalization.
	 - Per-step loop:
		 1. For each agent, `get_observation()` reads `FishingFleet` and `environment` state and composes the observation vector.
		 2. Agents compute actions via `select_action()` (hybrid: rule-checks then RL mapping).
		 3. `FishingFleet.step(actions, fish_pop, env)` applies physics and fishing mechanics: positions updated, fuel consumed, cargo changes, sales handled.
		 4. Ecosystem updated via `fish_pop.step(env)` and environmental dynamics advanced via `env.step()`.
		 5. Rewards computed per-agent by `FishingAgent.calculate_reward()` using pre/post states and events (e.g., sale, fuel use, MPA violation).
		 6. Experience tuples appended to per-agent replay buffers; periodic `train_step()` calls update policy nets.
		 7. Logs and per-step metrics appended for CSV export and live plots.

3) Class-level responsibilities & contracts
	 - `OceanEnvironment`:
		 • Pure state and procedural update rules (temperature, currents, plankton). Stateless relative to agents (agents query it).
		 • Must provide deterministic (seedable) procedural generation for reproducibility. Consider adding a `random_seed` argument and using a shared RNG.

	 - `FishingFleet`:
		 • Encapsulates actuators and sensors for boats: positions, headings, velocities, fuel, cargo, net state.
		 • Public API: `step(actions, fish_pop, env)`, and status arrays for agents to read.

	 - `FishPopulation`:
		 • Encapsulates schools, biomass, and regeneration rules. Exposes sampling/consumption APIs.

	 - `FishingAgent`:
		 • Observation assembly, action selection (hybrid), experience storage, training step and checkpoint serialization.
		 • Contract: `get_observation(state, fish_pop, env)` must return same-dimension numpy array. `select_action(obs, state)` must return continuous action array interpreted by `FishingFleet.step()`.

4) Training pipeline & reproducibility
	 - Episode randomization: ports, fish population and start week are randomized to improve generalization. To reproduce exact runs, snapshot `env.time_step`, RNG seeds and any per-episode metadata to the `checkpoints/training_metadata_ep{n}.npy` file (already done for core metrics).
	 - Checkpointing: per-agent checkpoints save network states, optimizer states and replay buffer. To reproduce evaluation exactly, load both model state and replay buffer if deterministic replay is required.

5) Reward & safety design considerations
	 - Reward shaping intentionally combines sparse (sales) and dense (catch, fuel) components. This guides learning while punishing catastrophic failures sharply (heavy negative reward for running out of fuel or illegal MPA fishing).
	 - Rule-based overrides protect agents during learning. When changing reward magnitudes, re-evaluate safety rules to avoid conflicting signals.

6) Performance & scaling
	 - Simulation-heavy loops (per-step updates, grid operations, MPA expansions) run in Python+NumPy. For larger worlds or many agents, consider:
		 • Vectorizing `FishingFleet` updates further and reducing Python-level loops.
		 • Moving heavy kernels (MPA expansion, density calculations) to Numba/Cython or small C++ extensions.
		 • Using datatypes carefully (float32 vs float64) to reduce memory and cache pressure.
	 - For RL training, using a GPU accelerates neural net updates but the simulator may become the bottleneck; consider asynchronous experience generation (multiple simulator workers) with a centralized learner.

7) Extensibility & recommended refactors
	 - Centralize experiment config into `config.py` or a YAML/JSON config to avoid editing scripts.
	 - Provide deterministic seeds across `numpy` and `torch` and propagate them into `OceanEnvironment` to support exact reproducibility.
	 - Add lightweight unit tests for core numerical invariants (e.g., fuel non-negativity, mass balance in fish consumption) under `tests/`.
	 - Consider an `AbstractAgent` interface and an alternate `PPOAgent` implementation for continuous control (actor-critic) if fine throttle/heading control is needed.

8) Evaluation & analysis best-practices
	 - Always compare RL policies to several heuristic baselines generated by `Part A` scenario drivers.
	 - Keep evaluation episodes deterministic (fixed seeds & ports) and export full per-timestep CSVs for reproducible figures.
	 - Use `training_metadata` saved with checkpoints to annotate plots and reproduce conditions.

If you'd like, I can now:
- Add a `CONFIG.md` that maps every major file to its responsibilities (helpful for new contributors),
- Implement a `config.py` and wire it into the main scripts, or
- Create a small reproducibility script that runs one baseline scenario and then evaluates a checkpoint and saves comparison CSVs.


MPA RL — Fisheries Simulation & Multi-Agent Reinforcement Learning
================================================================

Overview
--------

This repository implements a research platform for studying fishing fleet behavior and Marine Protected Area (MPA) policies using both a deterministic simulatory baseline and Multi-Agent Reinforcement Learning (MARL). The codebase is split into two cooperating parts:

- `Part A` — a high-fidelity simulatory environment and scenario runner that models fleet dynamics, vessel fuel consumption, and a simplified fish ecosystem. Use this to run controlled experiments and generate baseline CSV outputs.
- `Part B` — MARL experiments and tooling. Provides an RL environment wrapper around the simulator, agent implementations, training loops, evaluation/deployment scripts, analysis and visualization utilities, and saved checkpoints.

Why this structure
------------------

The split lets you:
- validate dynamics and heuristics in `Part A` without the complexity of RL training, and
- iterate on RL algorithms in `Part B` while reusing the same physics and ecosystem model for reproducible comparisons.

Architecture (components & data flow)
------------------------------------

1. Simulation core (`Part A`)
	- Fleet physics: `fleet_physics.py` models vessel motion, speed, fuel burn, and interactions with environmental state.
	- Ecosystem model: `fish_ecosystem.py` approximates local fish biomass, regeneration, and catch dynamics.
	- Scenario drivers: `final1.py` / `final2.py` run scripted fleet deployments and export `simulation_data.csv`, `fleet_summary.csv`, and `vessel_metrics.csv` into `simulation_output*` directories.

2. RL layer (`Part B`)
	- Env wrapper: `environment.py` adapts the simulation core into an observation/action/reward interface expected by RL libraries.
	- Agents & training: `train_fishing_agents.py` implements the training loop (supports episodic runs, checkpointing to `checkpoints/`).
	- Deployment & evaluation: `deploy_fleet.py` and variants load checkpoints and run evaluation episodes, producing the same CSV outputs for direct comparison with heuristic baselines.

3. Analysis & visualization
	- Analysis scripts (e.g., `fish_energy_analysis.py`, `ecosystem_energy_comparison.py`) process CSV outputs and generate figures or summary statistics.
	- Visualization (e.g., `make_gif.py`, `visualize_fish_mpa_dynamics.py`) produces GIFs or animations of agent/fish dynamics.

Typical experiments
-------------------

- Baseline: run a heuristic deployment in `Part A` and collect `fleet_summary.csv`.
- RL training: train agents in `Part B`, checkpoint policies to `checkpoints/` and compare learning curves.
- Evaluation: use `deploy_fleet.py` with saved checkpoints to produce evaluation CSVs and analyze fuel vs. yield tradeoffs.

Quick setup
-----------

1. Create and activate a Python virtual environment (from repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies for the part you want to run:

```powershell
pip install -r "Part A - Simulatory Environment/requirements.txt"
pip install -r "Part B - MARL/requirements.txt"
```

Example commands
----------------

- Run a short deterministic scenario (Part A):

```powershell
python "Part A - Simulatory Environment\final1.py"
```

- Train agents (Part B) for 200 episodes:

```powershell
python "Part B - MARL\train_fishing_agents.py" --episodes 200
```

- Evaluate a checkpoint:

```powershell
python "Part B - MARL\deploy_fleet.py" --checkpoint "Part B - MARL\checkpoints\agent_0_best.pt"
```

Files & outputs
---------------

- Simulation CSVs: `Part A - Simulatory Environment/simulation_output*/simulation_data.csv`, `fleet_summary.csv`, `vessel_metrics.csv`.
- Checkpoints: `Part B - MARL/checkpoints/` (policy files, e.g., `agent_0_best.pt`).
- Analysis outputs: `Part B - MARL/outputs/` and generated figures from the analysis scripts.

Developer notes
---------------

- Start experiments with small fleets and short episodes to validate configuration quickly.
- Many scripts accept configuration via command-line args or constants at the top of the files; inspect `if __name__ == '__main__'` blocks.
- To adapt the environment, modify `fleet_physics.py` and `fish_ecosystem.py` and re-run both heuristic and RL experiments to maintain comparability.

Next steps I can take
---------------------

- Add concrete command-line flags/examples for main scripts.
- Extract common config into a single `config.py` for easier experimentation.
- Add a short walkthrough reproducing a paper figure using existing checkpoints.

If you want one of these, tell me which and I will implement it.

AI & Models (summary)
----------------------

This project uses a hybrid approach combining rule-based heuristics with Deep Reinforcement Learning:

- Algorithm: Deep Q-Learning (DQN) per-agent with experience replay and a target network.
- Policy architecture: small feed-forward neural network (input -> 128 -> 128 -> 64 -> outputs).
- Observation: per-agent feature vector including normalized position/velocity, fuel/cargo, navigation/home info, temperature, 8-direction fish density sensors, 8-direction temperature gradient probes, net status, expected revenue and a time-of-year feature.
- Actions: continuous-sanitized outputs mapped from Q-values into [heading_change, throttle, net_deploy] where `heading_change` is clipped to ±0.5 rad/step, `throttle` is 0–1, and `net_deploy` is binary (0/1) derived from a Q-value.

Training particulars (default config present in `Part B - MARL/train_fishing_agents.py`):

- Optimizer: Adam, learning rate 0.0005.
- Replay buffer: capacity 10,000; minibatch size 64.
- Discount factor: γ = 0.99.
- Epsilon-greedy exploration: start ε=1.0, decay multiplicatively by 0.9995 per training step, with ε_min=0.10.
- Target network: updated every 10 episodes.
- Training frequency: agent.train_step() called every 4 simulation steps.
- Checkpointing: saved every 50 episodes; best-performing agents saved separately.

Reward engineering highlights
----------------------------

- Large positive reward for successful sale at port (scaled revenue + efficiency bonus).
- Continuous negative reward for fuel consumption and a small time penalty.
- Penalties for empty-net deployments, running out of fuel, risky low-fuel far-from-port situations, and illegal fishing inside MPAs (heavy penalty for net deployed inside MPA).

Hybrid control notes
--------------------

The `FishingAgent` implements several hard-coded safety and navigation behaviors (return-to-port, emergency fuel handling, net safety rules). RL is focused on strategic choices: where to search, when to deploy nets, and throttle control for fuel optimization. This hybrid design reduces catastrophic failures during training while leaving meaningful strategic decisions to learn.
