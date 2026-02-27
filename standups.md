# Standups

## 2026-01-14

**Branch:** main

### Completed
- No commits in the last 24 hours

### In Progress
- **RL Student Module** (`rl_student/`): Developing a reinforcement learning training framework
  - Q-table implementation (`q_table.py`)
  - Training script (`train.py`)
  - Wrappers for environment integration
  - Integration tests
  - README documentation
- **Q-table artifacts**: Generated trained Q-table files (`q_table_shared.pkl`, `q_table_shared_ep100.pkl`)
- **Custom map testing**: Script to test loading custom terrain maps (`test_custom_map.py`)
- **Legacy/reference code**: Archived simulation main script in `other/`

### Blockers
- None

### Next Steps
- Evaluate RL training results from Q-table experiments
- Consider committing the `rl_student/` module once validated
- Clean up temporary files (`.pkl` files, `test_custom_map.py`) or integrate into main test suite

## 2026-01-14 16:09

**Branch:** main

### Completed
- No commits in the last 24 hours

### In Progress
- **RL Student Module** (`rl_student/`): Reinforcement learning training framework
  - Q-table implementation with state discretization
  - Training script for agent learning
  - Environment wrappers for RL integration
  - Integration tests for validation
- **Q-table experiments**: Trained Q-table artifacts (`q_table_shared.pkl`, `q_table_shared_ep100.pkl`)
- **Custom map testing**: `test_custom_map.py` for terrain loading tests
- **Archived code**: Legacy files in `other/`

### Blockers
- None

### Next Steps
- Evaluate RL training results from Q-table experiments
- Validate and commit `rl_student/` module if ready
- Decide whether to integrate `test_custom_map.py` into main test suite or remove
- Clean up temporary `.pkl` files or move to appropriate location

## Session Wrap-up 2026-01-14

**Date:** 2026-01-14

### What Got Done
- Fixed missing import for `render_strategic_grid` in `demo_units.py`
- Fixed all pyright type errors in `demo_units.py`:
  - Added null checks for `spatial_grid` operations
  - Added type guard for `Projectile` append
  - Wrapped screen drawing operations in null check
- Ran and verified the units demo simulation (200 agents, 20 units)
- Ran full test suite: **267 passed**, 1 skipped

### Summary
Quick bug fix session to resolve type errors and runtime issues in the demo script, ensuring the strategic unit/swarm system runs cleanly with proper type safety.

## 2026-01-15 00:00

**Branch:** main

### Completed
- No commits in the last 24 hours

### In Progress
- **Unit/Squad System** (`combatenv/unit.py`): New module for organizing agents into tactical squads
  - Unit dataclass with centroid calculation and cohesion scoring
  - Waypoint-based movement commands
  - Spawn functions for unit creation (`spawn_unit`, `spawn_units_for_team`, `spawn_all_units`)

- **Boids Flocking Algorithm** (`combatenv/boids.py`): Reynolds' flocking behaviors
  - Cohesion, separation, alignment, and waypoint forces
  - Configurable weights and steering force limits

- **Strategic Grid System** (`combatenv/strategic_grid.py`): High-level observation/action spaces
  - 4x4 strategic grid over 64x64 tactical grid
  - Terrain summarization and occupancy detection
  - 32-value observation array (16 terrain + 16 occupancy)

- **Strategic Environment** (`combatenv/strategic_env.py`): New environment variant using strategic layer

- **Core Module Updates**:
  - `config.py`: Added unit, boids, formation, cohesion reward, and strategic grid configuration
  - `agent.py`: Added `unit_id` and `following_unit` fields
  - `renderer.py`: Added strategic grid overlay rendering
  - `environment.py`: Added `show_strategic_grid` toggle
  - `__init__.py`: Exported all new modules and functions

- **Tests**: New test files for unit, boids, and strategic grid

- **RL Wrappers** (`rl_student/wrappers/`): Cohesion reward wrapper and unit wrapper

### Blockers
- None

### Next Steps
- Add tests for strategic environment
- Validate boids flocking behavior in simulation
- Document new Unit API in docs/
- Consider committing unit/strategic system once validated

## Session Wrap-up 2026-01-15

**Date:** 2026-01-15

### What Got Done
- Integrated unit system into `TacticalCombatEnv` with `use_units=True` default
- Added conditional spawning: 8 units × 8 agents = 64 per team (vs 100 individual agents)
- Implemented boids steering in `_update_unit_movement()` for waypoint-following behavior
- Added waypoint control methods: `set_unit_waypoint()`, `clear_unit_waypoint()`, `clear_all_waypoints()`, `get_unit_by_id()`
- Added keyboard controls (1-8 to select units, SPACE to clear waypoints)
- Added mouse click waypoint setting for selected unit
- Added `render_waypoints()` and `render_selected_unit()` with yellow circle highlighting
- Updated keybindings overlay with new unit controls
- Updated tests to handle both unit and non-unit spawning modes
- Brainstormed logic programming integration (Prolog-style, strategic level, pyswip)
- Changed config to 8×8 agents per team (from 10×10)
- Full test suite: **268 passed**, 1 skipped

### Summary
Major integration session merging the unit/squad system from `demo_units.py` into the main `TacticalCombatEnv`. The game now supports interactive unit waypoint control with boids flocking, visual feedback for selected units, and backward-compatible non-unit mode.

## 2026-01-15 09:00

**Branch:** main

### Completed
- No commits in the last 24 hours

### In Progress
- **Unit/Squad System Integrated**: Full unit management in `TacticalCombatEnv`
  - `use_units=True` config option with conditional spawning (8 units × 8 agents = 64/team)
  - Boids flocking steering for waypoint-following behavior
  - Waypoint control methods (`set_unit_waypoint()`, `clear_unit_waypoint()`, etc.)
  - Keyboard (1-8 unit selection, SPACE clear) and mouse click waypoint controls
  - Visual feedback: `render_waypoints()`, `render_selected_unit()` with yellow highlighting

- **Strategic Grid System** (`combatenv/strategic_grid.py`):
  - 4x4 overlay on 64x64 tactical grid
  - `render_strategic_grid()` added to renderer
  - Configurable colors and line widths

- **Configuration Additions** (`config.py`):
  - Unit config: `NUM_UNITS_PER_TEAM=8`, `AGENTS_PER_UNIT=8`
  - Boids weights: cohesion, separation, alignment, waypoint
  - Formation breaking: combat range, health threshold
  - Cohesion reward bonuses/penalties
  - Strategic grid styling

- **New Untracked Modules**:
  - `combatenv/unit.py`, `combatenv/boids.py`, `combatenv/strategic_env.py`
  - Tests: `test_boids.py`, `test_strategic_grid.py`, `test_unit.py`
  - `demo_units.py`, `rl_student/` framework

### Blockers
- None

### Next Steps
- Run full test suite to validate current changes
- Commit unit/strategic system to main branch
- Document Unit and Strategic Grid APIs
- Clean up emacs backup files (`#agent.py#`, `#environment.py#`)

## Session Wrap-up 2026-01-16

**Date:** 2026-01-16

### What Got Done
- Implemented 8x8 coarse grid terrain generation system
  - Each coarse cell (8x8 tactical cells) gets a major terrain type
  - Connected "puddle" algorithm for organic terrain shapes (lakes, forests)
  - 40% empty, 60% major terrain distribution
  - Buildings placed last across entire map
- Updated strategic grid lines to match 8x8 coarse grid size
- Implemented unit reserve/dispatch system for strategic control
  - Units start in reserve (off-screen at -100, -100)
  - `dispatch()` teleports units to team spawn corner (blue: top-left, red: bottom-right)
  - Added `dispatch_unit()`, `get_next_reserve_unit()` methods to environment
  - Reserved units skip autonomous movement and combat
  - Controlled agent's unit auto-dispatched on reset
- Updated spawning to use `is_walkable()` for terrain compatibility
- Full test suite: **273 passed**, 1 skipped

### Summary
Significant terrain and strategic gameplay session. Refactored terrain generation to use a coarse grid system with connected terrain regions, and implemented a unit dispatch system where the strategic agent releases units one at a time from reserve to spawn corners.

## 2026-01-16 12:00

**Branch:** main

### Completed
- No commits in the last 24 hours (all work is uncommitted)

### In Progress
- **Coarse Grid Terrain System**: Major refactor of terrain generation
  - 8x8 coarse grid system with major terrain types per cell
  - Connected "puddle" algorithm for organic terrain shapes
  - Updated strategic grid lines to match coarse grid

- **Unit Reserve/Dispatch System**: Strategic deployment mechanics
  - Units start in reserve (off-screen at -100, -100)
  - `dispatch()` teleports to team spawn corners
  - Reserved units skip autonomous movement and combat

- **Core Module Updates** (~1,140 lines added/changed):
  - `environment.py`: +544 lines (unit integration, dispatch system)
  - `terrain.py`: +280 lines (coarse grid generation)
  - `renderer.py`: +162 lines (strategic overlays, waypoint rendering)
  - `config.py`: +104 lines (unit, boids, formation configs)
  - `agent.py`: +58 lines (unit_id, following_unit fields)

- **New Modules** (untracked):
  - `combatenv/unit.py` (16KB): Unit/squad management
  - `combatenv/boids.py` (13KB): Reynolds' flocking algorithm
  - `combatenv/strategic_grid.py` (7KB): Strategic layer observations
  - `combatenv/strategic_env.py` (14KB): Strategic environment variant
  - `demo_units.py` (12KB): Demo script for unit system

- **Test Updates**: 273 tests passing, 1 skipped

### Blockers
- None

### Next Steps
- Commit unit/strategic system to main branch
- Document Unit and Strategic Grid APIs in docs/
- Clean up emacs backup files (`#agent.py#`, `#environment.py#`)
- Validate boids flocking behavior in extended simulation runs

## Session Wrap-up 2026-01-17

**Date:** 2026-01-17

### What Got Done
- Implemented symmetric edge spawning for clearer training signal
  - Blue units spawn in row 0 (top edge, y ≈ 4)
  - Red units spawn in row 7 (bottom edge, y ≈ 60)
  - All waypoints set to center (32, 32)
- Created `UnitChebyshevWrapper` (`combatenv/wrappers/unit_chebyshev_reward.py`)
  - Calculates Chebyshev distance from unit centroid to waypoint
  - All agents in a unit receive the same distance-based reward
  - Encourages group movement vs individual agent drift
- Combined both Chebyshev reward wrappers in `MultiAgentDiscreteEnv`:
  - `ChebyshevRewardWrapper` (agent-based, scale=0.5)
  - `UnitChebyshevWrapper` (unit centroid, scale=0.5)
  - Total max reward = 1.0 when both individual and unit centroid move closer
- Disabled `FOVCoverageRewardWrapper` for cleaner reward signal
- Analyzed Q-learning implementation: N-table tracks visit counts but isn't used for exploration (epsilon-greedy only)

### Summary
Focused session on improving RL training signal quality. Refactored spawning to create symmetric edge-to-center navigation task, and implemented dual Chebyshev reward shaping to balance individual agent movement with unit cohesion.

## 2026-01-17 18:30

**Branch:** main

### Completed
- No commits in the last 24 hours (all work is uncommitted)

### In Progress
- **Dual Chebyshev Reward System**: Combined reward shaping for movement training
  - `ChebyshevRewardWrapper` (agent-based distance, scale=0.5)
  - `UnitChebyshevWrapper` (unit centroid distance, scale=0.5)
  - Total max reward = 1.0 when both components improve

- **New Wrapper Module** (`combatenv/wrappers/`):
  - `unit_chebyshev_reward.py`: Unit centroid-based Chebyshev distance rewards
  - `movement_obs.py`: Movement observation wrapper
  - Multiple reward wrappers for training experiments

- **Movement Training Environment** (`movement_training_env.py`):
  - Edge spawning: Blue at row 0, Red at row 7
  - All waypoints at center (32, 32)
  - `MultiAgentDiscreteEnv` with 128 agents (8 units × 8 agents × 2 teams)

- **RL Student Module** (`rl_student/`):
  - `circular_q_table.py`: Fixed-size Q-table with FIFO eviction
  - `multi_agent_q.py`: Multi-agent Q-table manager with knowledge sharing
  - `q_table.py`: Base Q-table implementation with N-table tracking

- **Training Scripts**:
  - `train_multi_agent.py`: Multi-agent training loop
  - `train_movement.py`: Movement training script
  - `run_visual.py`: Visual debugging script

- **Core Module Updates** (uncommitted):
  - `environment.py`, `renderer.py`, `config.py`, `agent.py`
  - Unit system, boids flocking, strategic grid

### Blockers
- None

### Next Steps
- Validate RL training with dual Chebyshev reward system
- Consider implementing UCB exploration using N-table visit counts
- Commit wrappers module and training environment
- Run extended training sessions to evaluate learning progress

## Session Wrap-up 2026-01-17

**Date:** 2026-01-17

### What Got Done
- Renamed "coarse grid" terminology to "operational grid" across the codebase
  - `COARSE_GRID_SIZE` → `OPERATIONAL_GRID_SIZE` (8×8)
  - `TACTICAL_CELLS_PER_COARSE` → `TACTICAL_CELLS_PER_OPERATIONAL` (8)
  - Updated 12+ files: config.py, environment.py, renderer.py, terrain.py, unit.py, movement_training_env.py, and multiple wrappers
- Fixed spawn position bug for operational training
  - Blue units now spawn in top row (operational cells 0-7, row 0)
  - Red units spawn in bottom row (operational cells 0-7, row 7)
  - Added `skip_auto_dispatch` option to prevent base env from overriding positions
- Implemented clean combat suppression via `EnvConfig.autonomous_combat` flag
  - Deleted `CombatSuppressionWrapper` (was modifying agent attributes)
  - Added `autonomous_combat: bool = True` to `EnvConfig`
  - TacticalCombatEnv skips `_execute_autonomous_combat()` when disabled
  - Updated `train_operational.py` to use config option with `--no-combat` flag
- Verified all tests pass (20 terrain tests confirmed)
- Training with `--no-combat` now keeps all agents at full health (avg reward ~229 vs ~70 with combat)

### Summary
Refactoring session focused on standardizing grid terminology (coarse → operational) and implementing proper combat suppression through environment configuration rather than wrapper hacks. The operational training system now has clean spawn positions and reliable combat toggling.

## Session Wrap-up 2026-01-18

**Date:** 2026-01-18

### What Got Done
- **Q-Learning Action Space Expansion**
  - Added diagonal movement actions (NE, NW, SE, SW) - actions 5-8
  - Added "Think" action placeholder (action 15) for future logic programming
  - Fixed MOVE_OFFSETS ASCII diagram to match actual code
- **Reward System Rebalancing**
  - Increased `DISTANCE_REWARD_PER_CELL` from 1.0 to 10.0 for stronger directional signal
  - Reduced `STANDING_STILL_PENALTY` from -0.5 to -0.1 (was overwhelming distance reward)
  - Fixed NaN rewards caused by `inf - inf` when units had no valid goal
  - Fixed standing still penalty applying even when goal was reached
- **Goal System Fixes**
  - Fixed goal setting bug (`dispatch_to` was only setting intermediate waypoint)
  - Added explicit `set_goal()` call in WaypointTaskWrapper
  - Implemented independent random goals for each unit (was all going to center)
- **Visual/UI Improvements**
  - Unit names: Blue=U1-U8 (uppercase), Red=u1-u8 (lowercase)
  - Goal labels: Blue=G1-G8 (uppercase), Red=g1-g8 (lowercase)
  - Reward panel now shows action AND goal cell: `[NE D57]`
  - Dead agents now highlighted (dimmer) when unit is selected
  - Fire terrain: replaced emoji with polygon flame icon
- **Training Script Enhancements**
  - Added `resume` command to auto-detect and continue from latest checkpoint
  - Changed reporting frequency from every 10 to every 3 episodes
  - Added Q-table entry count to training output
  - Deleted stale Q-table files for fresh training

### Summary
Major Q-learning debugging and improvement session. Fixed critical reward calculation bugs (NaN, inverted incentives) and expanded the action space with diagonal movements. Enhanced visual feedback with differentiated team naming and goal cell display in the reward panel.

## 2026-01-18 12:00

**Branch:** main

### Completed
- No commits in the last 24 hours (all work remains uncommitted)

### In Progress
- **Core Module Refactoring** (~2,000 lines changed across 14 files):
  - `environment.py`: +873 lines (unit integration, dispatch system, operational grid)
  - `renderer.py`: +652 lines (strategic overlays, waypoint rendering, flame icons)
  - `terrain.py`: +333 lines (operational grid terrain generation)
  - `config.py`: +139 lines (unit, boids, formation, operational configs)
  - `agent.py`: +112 lines (unit_id, following_unit, goal system)

- **Gymnasium Wrappers Library** (`combatenv/wrappers/`): 43 wrapper modules
  - Reward shaping: Chebyshev, unit Chebyshev, cohesion, anti-clump, FOV coverage, goldilocks
  - Observation: discrete obs, movement obs, operational/strategic discrete
  - Action: discrete action, action mask, action suppression
  - Training: operational, strategic, tactical, waypoint task, qlearning
  - Utilities: debug overlay, keybindings, terminal log, terrain gen

- **RL Training Infrastructure**:
  - Multiple training scripts: `train_operational.py`, `train_movement.py`, `train_multi_agent.py`, `train_waypoint_navigation.py`
  - Q-table checkpoints: `operational_agent_ep100-500.pkl`, `q_table_operational_ep100-500.pkl`
  - Waypoint training checkpoint: `waypoint_checkpoint_ep100.pkl`

- **Tests**: Updated test files for environment, integration, terrain, renderer

### Blockers
- None

### Next Steps
- Evaluate Q-learning training results from operational agent checkpoints
- Commit the wrappers module and training infrastructure
- Document wrapper APIs in docs/
- Clean up emacs backup files (`#*.py#`) and decide on `.pkl` file locations

## Session Wrap-up 2026-01-18

**Date:** 2026-01-18

### What Got Done
- **Project Cleanup and Organization**
  - Created `models/` folder and moved all 14 `.pkl` checkpoint files (operational agents, Q-tables, waypoint checkpoints)
  - Created `to-delete/` folder and moved 6 items: 4 emacs backup files (`#*.py#`), broken `example_layered_architecture.py`, and `other/` archive folder
  - Created comprehensive `PROJECT_FILES.md` documenting all project files:
    - Root Python files (entry points, training scripts)
    - Core package modules (`combatenv/`)
    - 43 Gymnasium wrappers organized by category
    - RL student module, tests, and documentation
- Verified tests: **285 passed**, 1 failed (pre-existing), 1 skipped

### Summary
Housekeeping session to organize the project structure. Moved trained model checkpoints to a dedicated `models/` folder, staged unused/backup files for deletion, and created detailed file documentation for the growing codebase.

## 2026-02-17 12:00

**Branch:** main

### Completed
- No new commits since last session (Jan 18)

### In Progress
- **Core Module Refactoring** (~2,020 lines changed across 15 tracked files):
  - `environment.py`: +873 lines (unit integration, dispatch system, operational grid)
  - `renderer.py`: +652 lines (strategic overlays, waypoint rendering, flame icons)
  - `terrain.py`: +333 lines (operational grid terrain generation)
  - `config.py`: +139 lines (unit, boids, formation, operational configs)
  - `agent.py`: +112 lines (unit_id, following_unit, goal system)

- **New Modules** (2,186 lines across 6 files, untracked):
  - `combatenv/unit.py` (670 lines): Unit/squad management with waypoints
  - `combatenv/boids.py` (423 lines): Reynolds' flocking algorithm
  - `combatenv/strategic_env.py` (473 lines): Strategic environment variant
  - `combatenv/strategic_grid.py` (258 lines): 4x4 strategic observation layer
  - `combatenv/gridworld.py` (190 lines): Base GridWorld for wrapper architecture
  - `combatenv/wrapper_factory.py` (172 lines): Factory functions for env creation

- **Gymnasium Wrappers Library** (`combatenv/wrappers/`): ~50 wrapper modules for RL training (reward shaping, observation, action, training utilities)

- **RL Training Artifacts**: 14 model checkpoints in `models/` (Q-tables, operational agents, waypoint checkpoints)

- **Training Scripts**: `train_operational.py`, `train_movement.py`, `train_multi_agent.py`, `train_waypoint_navigation.py`, `train_tactical_combat.py`, `train_tactical_movement.py`

### Blockers
- None

### Next Steps
- Commit the substantial uncommitted work (2,000+ lines of core changes + 6 new modules + 50 wrappers)
- Run full test suite to validate current state
- Document wrapper APIs and Unit system in docs/
- Evaluate Q-learning training results from model checkpoints
- Clean up `to-delete/` folder

## Session Wrap-up 2026-02-17

**Date:** 2026-02-17

### What Got Done
- Reviewed project status after a month-long break (last session: Jan 18)
- Confirmed ~2,020 lines of core module changes across 15 tracked files (environment, renderer, terrain, config, agent)
- Confirmed 6 new untracked modules totaling 2,186 lines (unit.py, boids.py, strategic_env.py, strategic_grid.py, gridworld.py, wrapper_factory.py)
- Confirmed ~50 Gymnasium wrapper modules in `combatenv/wrappers/` for RL training
- Confirmed 14 trained model checkpoints organized in `models/`
- Confirmed 6 training scripts for various RL experiments (operational, movement, multi-agent, waypoint, tactical combat, tactical movement)

### Summary
Status review session after a month-long break. All prior work (unit/squad system, boids flocking, strategic grid, wrapper library, RL training infrastructure) remains intact but uncommitted. The project has grown substantially with ~4,200+ new lines of code awaiting commit.

## 2026-02-21 00:00

**Branch:** main

### Completed
- No commits or file changes in the last 24 hours

### In Progress
- **Core Module Refactoring** (~2,020 lines changed across 15 tracked files, uncommitted):
  - `environment.py`, `renderer.py`, `terrain.py`, `config.py`, `agent.py`, `fov.py`, `main.py`, plus test files
- **New Modules** (6 untracked, ~2,186 lines): `unit.py`, `boids.py`, `strategic_env.py`, `strategic_grid.py`, `gridworld.py`, `wrapper_factory.py`
- **Gymnasium Wrappers Library** (`combatenv/wrappers/`): ~50 wrapper modules for RL training
- **Training Scripts** (6 scripts): operational, movement, multi-agent, waypoint, tactical combat, tactical movement
- **Trained Models** (`models/`): 14 Q-table and agent checkpoints

### Blockers
- None

### Next Steps
- Commit the large body of uncommitted work (~4,200+ lines across core changes, new modules, and wrappers)
- Run full test suite to validate current state
- Document wrapper APIs and Unit system in docs/
- Clean up `to-delete/` folder

## Session Wrap-up 2026-02-28

**Date:** 2026-02-28

### What Got Done
- Added elevation mechanic to empty tiles in SSS visualiser (`teaching/env.py`)
  - Perlin noise elevation field (1024x1024 float32) generated per terrain reset
  - `get_elevation(cx, cy)` accessor for cell-level elevation queries
  - 4 new constants: `ELEVATION_MAX_STAMINA_DRAIN`, `ELEVATION_SPEED_BONUS`, `ELEVATION_HP_DRAIN`, `ELEVATION_HP_THRESHOLD`
- Implemented elevation cost in `CombatReadinessRewardWrapper.step()` — scales from 1.0 (flat) to ~3.6 (peak)
- Added `render_elevation()` cached pixel-level overlay with two-ramp coloring (green-brown lowlands / tan highlands)
- Wired elevation into render loop, cache invalidation, and terrain generation
- Updated `sss_solution.py` and `sss_question.py` with elevation-aware `_apply_terrain_effects()` and `_combat_readiness_cost()`
- Verified all changes: syntax checks, import checks, and elevation field generation test all passing

### Summary
Implemented the full elevation mechanic for empty tiles across three files, giving Dijkstra/A* a richer cost landscape to exploit with visual feedback so students can see why informed search picks certain routes.
