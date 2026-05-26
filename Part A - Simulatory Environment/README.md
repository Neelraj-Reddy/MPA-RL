
Part A — Simulatory Environment
================================

Purpose
-------

`Part A` is the deterministic simulatory baseline used across the project. It provides:

- a fleet dynamics model (movement, speed, fuel consumption),
- a simplified fish ecosystem (local biomass, catch rates, regeneration), and
- scenario runners and exporters that create canonical CSV outputs for analysis and for direct comparison with RL agents.

Core components
---------------

- `fleet_physics.py` — vessel motion, thrust-to-speed mapping, fuel burn calculations, collision/interaction handling.
- `fish_ecosystem.py` — cell-based or continuous biomass model, catch calculation when vessels fish, regeneration/seasonality rules.
- `final1.py`, `final2.py` — scenario drivers: compose fleets, MPAs, environment settings, run episodes, and export results.
- `test*.py` — smoke tests and short example runs useful for quick validation.
- `simulation_output/`, `simulation_output_1/` — stored `simulation_data.csv` files produced by scenario runs.

Data model & outputs
--------------------

- `simulation_data.csv`: per-timestep records (vessel positions, speed, local biomass, catch, fuel state).
- `fleet_summary.csv`: per-run aggregated metrics (total catch, total fuel consumed, average yield per vessel).
- `vessel_metrics.csv`: per-vessel time-averaged metrics suitable for agent-level comparisons.

Running locally
---------------

1. Install dependencies:

```powershell
pip install -r "Part A - Simulatory Environment/requirements.txt"
```

2. Run a short scenario to generate outputs:

```powershell
python "Part A - Simulatory Environment\final1.py" --episodes 10 --fleet-size 5
```

Configuration
-------------

- Scenario parameters are configured via command-line flags (when available) or at the top of the driver scripts. Typical knobs include `--episodes`, `--fleet-size`, `--mpa-radius`, and simulation timestep settings.

Validation & extension points
----------------------------

- Use `test_realistic_physics.py` (if present) to validate energy and speed relationships against known values.
- To extend the ecosystem model, modify `fish_ecosystem.py` and re-run both `final1.py` and RL experiments so comparisons remain consistent.

