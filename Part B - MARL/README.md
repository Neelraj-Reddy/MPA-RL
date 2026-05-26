
Part B — Multi-Agent Reinforcement Learning (MARL)
=================================================

Purpose
-------

`Part B` contains the RL experiments that use the simulator from `Part A` as the environment backend. It provides an environment wrapper, training harness, deployment scripts, analysis tools, visualization utilities, and saved checkpoints for evaluation and reproduction.

System architecture
-------------------

- Environment adapter (`environment.py`): converts simulator state into observations and applies agent actions back to the simulator. Observations typically include vessel positions, remaining fuel, local fish biomass, and time/simulation flags.
- Policy & training loop (`train_fishing_agents.py`): trains policies using episodic rollouts, supports checkpointing to `checkpoints/` and logging to `outputs/`.
- Deployment & evaluation (`deploy_fleet.py`, `deploy_fleet_no_mpa.py`): load policies and run deterministic evaluation episodes to produce CSV outputs for direct comparison with heuristics.
- Analysis & visualization: `fish_energy_analysis.py`, `ecosystem_energy_comparison.py`, `make_gif.py`, `visualize_fish_mpa_dynamics.py`.

Agent & training details
------------------------

- Execution model: the codebase uses centralized training with decentralized execution patterns (train policies on joint data, deploy per-agent policies independently) — check `train_fishing_agents.py` for exact algorithm.
- Checkpoints: trained policies and optimizer state are saved in `Part B - MARL/checkpoints/` (policy files like `agent_0_best.pt`).
- Observations & actions: inspect `environment.py` to confirm the observation vector layout and action space (discrete/continuous).

Quick setup
-----------

Create and activate a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "Part B - MARL/requirements.txt"
```

Common commands
---------------

- Train a quick experiment:

```powershell
python "Part B - MARL\train_fishing_agents.py" --episodes 200 --num-agents 4
```

- Deploy a saved checkpoint for evaluation:

```powershell
python "Part B - MARL\deploy_fleet.py" --checkpoint "Part B - MARL\checkpoints\agent_0_best.pt" --episodes 10
```

- Reproduce an analysis figure (example):

```powershell
python "Part B - MARL\fish_energy_analysis.py" --input "Part B - MARL\outputs\training_metrics.csv"
```

Outputs
-------

- `Part B - MARL/checkpoints/`: saved model checkpoints.
- `Part B - MARL/outputs/`: training logs, metrics, and evaluation CSVs.

Practical tips
--------------

- Use small `--episodes` and `--num-agents` values while developing. Training full experiments can take hours depending on hardware.
- If you change the simulator API (in `Part A`), update `environment.py` accordingly and re-run both heuristic and RL experiments for a fair comparison.

Next steps I can implement for you
---------------------------------

- Add a small example config and CLI parser for the main scripts to standardize runs.
- Create a short reproducibility script that runs a baseline scenario and an evaluation with a provided checkpoint.

AI & Models (detailed)
----------------------

This project uses a hybrid Deep RL + rule-based agent design. Key details from the implementation:

- Algorithm: Per-agent Deep Q-Network (DQN) with experience replay and a target network. Training loop lives in `train_fishing_agents.py`.
- Network architecture: a feed-forward MLP with layers [observation_dim -> 128 -> 128 -> 64 -> 3]. The network outputs three Q-values which are mapped to the three action components.
- Observation vector: ~33 dimensions (normalized position, velocity, heading, fuel, cargo, home direction/distance/fuel-check, local temperature, 8-direction fish density sensors, 8-direction temperature gradients, net status, expected revenue/fuel value, MPA flag, and time-of-year).
- Action mapping: the network returns 3 Q-values; these are converted to actions as follows:
	- `heading_change` = tanh(q0) * 0.5 radians
	- `throttle` = sigmoid(q1) → [0,1]
	- `net_deploy` = 1 if q2 > 0 else 0 (binary)

Training hyperparameters (defaults in `train_fishing_agents.py` & `fishing_agent.py`):

- Optimizer: Adam, lr=0.0005.
- Replay buffer size: 10,000; batch_size=64.
- Discount γ=0.99.
- Exploration: ε-start=1.0, ε_decay=0.9995, ε_min=0.10.
- Train frequency: every 4 simulation steps.
- Target network update: every 10 episodes.
- Checkpointing: save every 50 episodes; best agents saved as `agent_X_best.pt`.

Reward shaping
--------------

- Positive rewards for catching fish and selling at port (includes an efficiency bonus scaled by revenue/fuel used).
- Negative rewards for fuel usage, time cost, empty-net deployments, running out of fuel, and illegal fishing inside MPAs (heavy penalty for deploying nets in protected zones).

Hybrid & safety systems
-----------------------

- Several rule-based policies ensure safety and domain constraints: emergency out-of-fuel behavior, automatic return-to-port when fuel/cargo thresholds are met, and smart net control to prevent wasteful or illegal deployments. These reduce catastrophic failures during learning and make training practical without complex constraints handling in the RL loss.

Practical notes
---------------

- Because the action outputs are derived from Q-values (not raw continuous actor outputs), the implementation implements a simplified DQN mapping where continuous controls are obtained by squashing Q-values. This hybrid mapping is pragmatic for this domain but could be replaced with an actor-critic (e.g., PPO/SAC) if continuous control becomes the primary objective.
- Training is CPU/GPU dependent—start with fewer episodes and smaller worlds to iterate quickly.

