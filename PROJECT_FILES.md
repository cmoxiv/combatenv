# Project Files Documentation

This document provides an overview of all files in the combatenv project.

---

## Root Python Files

### Entry Points

| File            | Purpose                                                          |
|-----------------|------------------------------------------------------------------|
| `main.py`       | Primary entry point - runs the visual tactical combat simulation |
| `run_visual.py` | Alternative visual runner for the simulation                     |
| `demo_units.py` | Demonstration of the unit system                                 |

### Training Scripts

| File                           | Purpose                               | Usage                                              |
|--------------------------------|---------------------------------------|----------------------------------------------------|
| `train_tactical_movement.py`   | Phase 1 tactical training (movement)  | `python train_tactical_movement.py --episodes 100` |
| `train_tactical_combat.py`     | Phase 2 tactical training (combat)    | `python train_tactical_combat.py --episodes 100`   |
| `train_operational.py`         | Operational-level Q-learning training | `python train_operational.py`                      |
| `train_waypoint_navigation.py` | Waypoint navigation training          | `python train_waypoint_navigation.py`              |
| `train_movement.py`            | Basic movement training               | `python train_movement.py`                         |
| `train_multi_agent.py`         | Multi-agent RL training               | `python train_multi_agent.py`                      |

### Support Files

| File                       | Purpose                                   |
|----------------------------|-------------------------------------------|
| `movement_training_env.py` | Environment wrapper for movement training |
| `test_custom_map.py`       | Testing custom map functionality          |
|                            |                                           |

---

## Core Package (`combatenv/`)

### Main Modules

| File             | Purpose                                             |
|------------------|-----------------------------------------------------|
| `__init__.py`    | Package exports (public API)                        |
| `environment.py` | `TacticalCombatEnv` - main Gymnasium environment    |
| `agent.py`       | `Agent` class - movement, combat, resources         |
| `config.py`      | All configuration constants                         |
| `terrain.py`     | `TerrainType` enum and `TerrainGrid` class          |
| `fov.py`         | Field of view calculations (ray-casting)            |
| `spatial.py`     | Spatial grid optimization for O(1) neighbor queries |
| `renderer.py`    | Pygame rendering system                             |
| `map_io.py`      | Save/load custom maps to JSON                       |
| `projectile.py`  | Projectile entity and factory                       |

### Extended Modules

| File                 | Purpose                                   |
|----------------------|-------------------------------------------|
| `boids.py`           | Boids flocking behavior implementation    |
| `gridworld.py`       | Base GridWorld environment                |
| `strategic_env.py`   | Strategic-level environment               |
| `strategic_grid.py`  | Strategic grid implementation             |
| `unit.py`            | Unit management system                    |
| `wrapper_factory.py` | Factory for creating wrapper combinations |

---

## Wrappers (`combatenv/wrappers/`)

Wrappers extend the base environment with additional functionality following the Gymnasium wrapper pattern.

### Core Infrastructure

| File              | Purpose                             |
|-------------------|-------------------------------------|
| `__init__.py`     | Wrapper package exports             |
| `base_wrapper.py` | Base wrapper class for all wrappers |

### Environment Wrappers

| File                    | Purpose               |
|-------------------------|-----------------------|
| `agent_wrapper.py`      | Agent management      |
| `combat_wrapper.py`     | Combat system         |
| `fov_wrapper.py`        | Field of view         |
| `movement_wrapper.py`   | Agent movement        |
| `projectile_wrapper.py` | Projectile management |
| `team_wrapper.py`       | Team management       |
| `terrain_wrapper.py`    | Terrain system        |
| `unit_wrapper.py`       | Unit management       |
| `empty_terrain.py`      | Empty terrain setup   |
| `terrain_gen.py`        | Terrain generation    |

### Observation & Action Wrappers

| File                        | Purpose                            |
|-----------------------------|------------------------------------|
| `observation_wrapper.py`    | Observation processing             |
| `discrete_action.py`        | Discrete action space              |
| `discrete_obs.py`           | Discrete observation space         |
| `action_mask.py`            | Action masking                     |
| `action_suppression.py`     | Action suppression                 |
| `movement_obs.py`           | Movement observations              |
| `single_discrete_action.py` | Single agent discrete actions      |
| `single_discrete_obs.py`    | Single agent discrete observations |

### Reward Wrappers

| File                        | Purpose                            |
|-----------------------------|------------------------------------|
| `reward.py`                 | Base reward wrapper                |
| `tactical_reward.py`        | Tactical-level rewards             |
| `tactical_combat_reward.py` | Combat-specific rewards            |
| `chebyshev_reward.py`       | Distance-based rewards (Chebyshev) |
| `cohesion_reward.py`        | Team cohesion rewards              |
| `anti_clump_reward.py`      | Anti-clumping rewards              |
| `fov_coverage_reward.py`    | FOV coverage rewards               |
| `goldilocks_reward.py`      | Goldilocks distance rewards        |
| `unit_chebyshev_reward.py`  | Unit distance rewards              |

### Training Wrappers

| File                       | Purpose                   |
|----------------------------|---------------------------|
| `qlearning_wrapper.py`     | Q-learning integration    |
| `tactical_qlearning.py`    | Tactical Q-learning       |
| `waypoint_task_wrapper.py` | Waypoint navigation tasks |
| `termination_wrapper.py`   | Episode termination       |

### Level-Specific Wrappers

| File                             | Purpose                           |
|----------------------------------|-----------------------------------|
| `tactical.py`                    | Tactical-level composite wrapper  |
| `operational.py`                 | Operational-level wrapper         |
| `operational_training.py`        | Operational training wrapper      |
| `operational_discrete_action.py` | Operational discrete actions      |
| `operational_discrete_obs.py`    | Operational discrete observations |
| `strategic.py`                   | Strategic-level wrapper           |
| `strategic_discrete_action.py`   | Strategic discrete actions        |
| `strategic_discrete_obs.py`      | Strategic discrete observations   |
| `multi_agent.py`                 | Multi-agent wrapper               |

### Rendering & Debug

| File                | Purpose                 |
|---------------------|-------------------------|
| `render_wrapper.py` | Rendering integration   |
| `debug_overlay.py`  | Debug overlay display   |
| `keybindings.py`    | Keyboard input handling |
| `terminal_log.py`   | Terminal logging        |

---

## RL Student (`rl_student/`)

Learning implementations for students.

| File                  | Purpose                         |
|-----------------------|---------------------------------|
| `__init__.py`         | Package exports                 |
| `q_table.py`          | Basic Q-table manager           |
| `circular_q_table.py` | Circular Q-table implementation |
| `multi_agent_q.py`    | Multi-agent Q-learning          |
| `train.py`            | Training utilities              |
| `test_integration.py` | Integration tests               |
| `README.md`           | Documentation                   |
| `wrappers/`           | Student-specific wrappers       |

---

## Tests (`tests/`)

Unit and integration tests.

| File                     | Purpose               |
|--------------------------|-----------------------|
| `__init__.py`            | Test package          |
| `test_agent.py`          | Agent class tests     |
| `test_environment.py`    | Environment tests     |
| `test_terrain.py`        | Terrain system tests  |
| `test_fov.py`            | FOV calculation tests |
| `test_spatial.py`        | Spatial grid tests    |
| `test_projectile.py`     | Projectile tests      |
| `test_renderer.py`       | Renderer tests        |
| `test_map_io.py`         | Map I/O tests         |
| `test_integration.py`    | Integration tests     |
| `test_boids.py`          | Boids behavior tests  |
| `test_gridworld.py`      | GridWorld tests       |
| `test_strategic_grid.py` | Strategic grid tests  |
| `test_unit.py`           | Unit system tests     |

**Run all tests:**
```bash
source ~/.venvs/pg/bin/activate
python -m pytest tests/ -v
```

---

## Documentation (`docs/`)

| File                   | Purpose                         |
|------------------------|---------------------------------|
| `README.md`            | User-facing documentation       |
| `ARCHITECTURE.md`      | System design and data flow     |
| `API.md`               | Configuration and API reference |
| `RENDERER_GUIDE.md`    | Rendering system guide          |
| `RENDERING_SUMMARY.md` | Rendering summary               |
| `HISTORY.md`           | Project history                 |
| `TODO.md`              | Future work                     |
| `rl-training/`         | RL training guides              |
| `media/`               | Images and media assets         |

---

## Models (`models/`)

Trained model checkpoints (`.pkl` files).

| File                            | Description                   |
|---------------------------------|-------------------------------|
| `operational_agent.pkl`         | Final operational agent       |
| `operational_agent_ep*.pkl`     | Operational agent checkpoints |
| `q_table_operational.pkl`       | Final Q-table                 |
| `q_table_operational_ep*.pkl`   | Q-table checkpoints           |
| `waypoint_qtables.pkl`          | Waypoint navigation Q-tables  |
| `waypoint_checkpoint_ep100.pkl` | Waypoint training checkpoint  |

---

## To Delete (`to-delete/`)

Files staged for deletion:
- Editor backup files (`#...#`)
- `example_layered_architecture.py` - has broken imports
- `other/` - archived old code
