# Architecture Documentation

This document describes the system architecture, design patterns, and data flow of the `combatenv` package - a Gymnasium-compatible multi-agent tactical combat environment.

## Table of Contents

- [System Overview](#system-overview)
- [Package Structure](#package-structure)
- [Module Dependency Graph](#module-dependency-graph)
- [Environment Step Flow](#environment-step-flow)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Design Patterns](#design-patterns)
- [Coordinate Systems](#coordinate-systems)
- [Performance Considerations](#performance-considerations)

## System Overview

The simulation is built as a Gymnasium-compatible environment with clear separation of concerns:

```
+-------------------------------------------------------------------------+
|                         SYSTEM ARCHITECTURE                              |
+-------------------------------------------------------------------------+
|                                                                         |
|   +-------------+                                                       |
|   |   main.py   |  Entry point, creates env, runs episodes              |
|   +------+------+                                                       |
|          |                                                              |
|          v                                                              |
|   +------------------------------------------------------------------+  |
|   |                    combatenv PACKAGE                              |  |
|   |  +------------+  +------------+  +------------+  +------------+  |  |
|   |  |environment |->|   agent    |->| projectile |->|  renderer  |  |  |
|   |  +------------+  +------------+  +------------+  +------------+  |  |
|   |        |              |                               |          |  |
|   |        v              v                               v          |  |
|   |  +------------+  +------------+  +------------+  +------------+  |  |
|   |  |  terrain   |  |    fov     |  |  spatial   |  |   config   |  |  |
|   |  +------------+  +------------+  +------------+  +------------+  |  |
|   +------------------------------------------------------------------+  |
|                                                                         |
+-------------------------------------------------------------------------+
```

## Package Structure

```
combatenv/
├── __init__.py        # Public API exports
├── environment.py     # TacticalCombatEnv (Gymnasium interface)
├── agent.py           # Agent class, spawning functions
├── terrain.py         # TerrainType enum, TerrainGrid class
├── projectile.py      # Projectile class, factory function
├── fov.py             # Field of view calculations
├── spatial.py         # SpatialGrid for collision optimization
├── renderer.py        # Pygame rendering functions
└── config.py          # All configuration constants
```

## Module Dependency Graph

```
+---------------------------------------------------------------------+
|                    MODULE DEPENDENCIES                               |
+---------------------------------------------------------------------+
|                                                                     |
|                        +------------+                               |
|                        | config.py  | (No dependencies)             |
|                        +-----+------+                               |
|                              |                                      |
|         +--------+-------+---+---+-------+--------+                |
|         v        v       v       v       v        v                |
|   +--------+ +-------+ +-----+ +------+ +------+ +----------+      |
|   |terrain | | fov   | |agent| |proj- | |render| | spatial  |      |
|   |  .py   | |  .py  | | .py | |ectile| |er.py | |   .py    |      |
|   +--------+ +---+---+ +--+--+ +------+ +------+ +----------+      |
|                  |        |                                        |
|                  +---+----+                                        |
|                      |                                             |
|                      v                                             |
|              +----------------+                                    |
|              | environment.py |  (Orchestrates all modules)        |
|              +----------------+                                    |
|                      |                                             |
|                      v                                             |
|                +----------+                                        |
|                | main.py  |                                        |
|                +----------+                                        |
|                                                                     |
+---------------------------------------------------------------------+
```

### Module Responsibilities

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `config.py` | Centralized configuration | All constants (GRID_SIZE, speeds, etc.) |
| `terrain.py` | Terrain system | `TerrainType`, `TerrainGrid`, `generate_random_terrain()` |
| `agent.py` | Agent entity and behavior | `Agent`, `spawn_team()`, `spawn_all_teams()` |
| `fov.py` | Field of view calculations | `get_layered_fov_overlap()`, `is_agent_visible_to_agent()` |
| `spatial.py` | Spatial partitioning | `SpatialGrid` |
| `projectile.py` | Projectile entity | `Projectile`, `create_projectile()` |
| `renderer.py` | All rendering functions | `render_all()`, `render_debug_overlay()` |
| `environment.py` | Gymnasium environment | `TacticalCombatEnv`, `EnvConfig` |

## Environment Step Flow

The main game loop follows a Gymnasium pattern with internal phases:

```
+-------------------------------------------------------------------------+
|                       ENVIRONMENT STEP PHASES                            |
+-------------------------------------------------------------------------+
|                                                                         |
|   +-------------------------------------------------------------+      |
|   |  PHASE 1: PROCESS ACTION                                     |      |
|   |  ├── Parse action array [movement, rotation, shoot]          |      |
|   |  ├── Apply movement to controlled agent                      |      |
|   |  └── Process shoot action if requested                       |      |
|   +-------------------------------------------------------------+      |
|                              |                                          |
|                              v                                          |
|   +-------------------------------------------------------------+      |
|   |  PHASE 2: SPATIAL GRID REBUILD                               |      |
|   |  └── spatial_grid.build(all_agents)                          |      |
|   |      (Agents have moved, grid must be refreshed)             |      |
|   +-------------------------------------------------------------+      |
|                              |                                          |
|                              v                                          |
|   +-------------------------------------------------------------+      |
|   |  PHASE 3: RESOURCE UPDATES                                   |      |
|   |  For each living agent:                                      |      |
|   |  ├── update_cooldown(dt)      # Shooting cooldown            |      |
|   |  ├── update_reload(dt)        # Magazine reload timer        |      |
|   |  └── update_stamina(dt)       # Stamina drain/regen          |      |
|   +-------------------------------------------------------------+      |
|                              |                                          |
|                              v                                          |
|   +-------------------------------------------------------------+      |
|   |  PHASE 4: COMBAT PHASE                                       |      |
|   |  For each AI agent that can_shoot():                         |      |
|   |  ├── Query nearby agents via spatial grid                    |      |
|   |  ├── Check targets in near/far FOV                           |      |
|   |  ├── Select closest target (near FOV prioritized)            |      |
|   |  ├── Apply accuracy modifiers (FOV layer + movement)         |      |
|   |  └── Create projectile with shoot_at_target()                |      |
|   +-------------------------------------------------------------+      |
|                              |                                          |
|                              v                                          |
|   +-------------------------------------------------------------+      |
|   |  PHASE 5: PROJECTILE UPDATES                                 |      |
|   |  For each projectile:                                        |      |
|   |  ├── Update position (projectile.update(dt, terrain_grid))   |      |
|   |  ├── Check expiration (lifetime, out of bounds)              |      |
|   |  ├── Check collision with buildings (destroyed on hit)       |      |
|   |  ├── Check collision with all agents                         |      |
|   |  └── Apply damage on hit (agent.take_damage())               |      |
|   +-------------------------------------------------------------+      |
|                              |                                          |
|                              v                                          |
|   +-------------------------------------------------------------+      |
|   |  PHASE 6: AI AGENT MOVEMENT                                  |      |
|   |  For each living AI agent:                                   |      |
|   |  ├── Query nearby agents for collision check                 |      |
|   |  └── Execute wander() behavior                               |      |
|   +-------------------------------------------------------------+      |
|                              |                                          |
|                              v                                          |
|   +-------------------------------------------------------------+      |
|   |  PHASE 7: TERRAIN EFFECTS                                    |      |
|   |  For each living agent:                                      |      |
|   |  ├── Check terrain at position                               |      |
|   |  ├── Apply fire damage (bypasses armor)                      |      |
|   |  └── Update swamp stuck state                                |      |
|   +-------------------------------------------------------------+      |
|                              |                                          |
|                              v                                          |
|   +-------------------------------------------------------------+      |
|   |  PHASE 8: OBSERVATION & REWARD                               |      |
|   |  ├── Generate observation vector                             |      |
|   |  ├── Calculate reward                                        |      |
|   |  └── Check termination conditions                            |      |
|   +-------------------------------------------------------------+      |
|                              |                                          |
|                              v                                          |
|   +-------------------------------------------------------------+      |
|   |  PHASE 9: RENDERING (if render_mode="human")                 |      |
|   |  render_all() draws in order:                                |      |
|   |  ├── 1. Background (white)                                   |      |
|   |  ├── 2. Grid lines (faint gray)                              |      |
|   |  ├── 3. Terrain (buildings, fire, swamp, water)              |      |
|   |  ├── 4. FOV highlights (layered transparency)                |      |
|   |  ├── 5. Agents (circles with orientation)                    |      |
|   |  ├── 6. Projectiles                                          |      |
|   |  └── 7. Muzzle flashes                                       |      |
|   +-------------------------------------------------------------+      |
|                                                                         |
+-------------------------------------------------------------------------+
```

## Component Details

### Agent Component

```
+---------------------------------------------------------------------+
|                         AGENT CLASS                                  |
+---------------------------------------------------------------------+
|                                                                     |
|   @dataclass                                                        |
|   class Agent:                                                      |
|       +-----------------------------------------------------+      |
|       |  CORE ATTRIBUTES                                     |      |
|       |  ├── position: (float, float)  # Grid coordinates    |      |
|       |  ├── orientation: float        # Degrees (0-360)     |      |
|       |  └── team: "blue" | "red"                            |      |
|       +-----------------------------------------------------+      |
|                                                                     |
|       +-----------------------------------------------------+      |
|       |  COMBAT ATTRIBUTES                                   |      |
|       |  ├── health: int = 100                               |      |
|       |  ├── armor: int = 100                                |      |
|       |  └── shoot_cooldown: float = 0.0                     |      |
|       +-----------------------------------------------------+      |
|                                                                     |
|       +-----------------------------------------------------+      |
|       |  RESOURCE ATTRIBUTES                                 |      |
|       |  ├── stamina: float = 100.0                          |      |
|       |  ├── ammo_reserve: int = 1000                        |      |
|       |  ├── magazine_ammo: int = 30                         |      |
|       |  ├── reload_timer: float = 0.0                       |      |
|       |  └── stuck_steps: int = 0                            |      |
|       +-----------------------------------------------------+      |
|                                                                     |
+---------------------------------------------------------------------+
```

### Terrain System

```
+---------------------------------------------------------------------+
|                      TERRAIN SYSTEM                                  |
+---------------------------------------------------------------------+
|                                                                     |
|   TerrainType (IntEnum):                                            |
|   ├── EMPTY (0)    - Passable, no effect                            |
|   ├── BUILDING (1) - Blocks movement AND line of sight              |
|   ├── FIRE (2)     - Passable, damages agents (bypasses armor)      |
|   ├── SWAMP (3)    - Passable, temporarily immobilizes agents       |
|   └── WATER (4)    - Blocks movement, allows line of sight          |
|                                                                     |
|   TerrainGrid:                                                      |
|   ├── 2D array of TerrainType values                                |
|   ├── is_walkable(x, y) -> bool                                     |
|   ├── blocks_los(x, y) -> bool                                      |
|   └── Used by FOV calculations for LOS blocking                     |
|                                                                     |
+---------------------------------------------------------------------+
```

### Two-Layer FOV System

```
+---------------------------------------------------------------------+
|                       TWO-LAYER FOV SYSTEM                           |
+---------------------------------------------------------------------+
|                                                                     |
|                        Agent Facing Right (0 deg)                   |
|                                                                     |
|                              120 deg (Far FOV)                      |
|                         /---------------------\                     |
|                       /   90 deg (Near FOV)     \                   |
|                     /   /---------------\         \                 |
|                   /   /                   \         \               |
|                 /   /         NEAR          \         \             |
|               /   /        (90% acc)          \         \           |
|             /   /    +---------------+           \         \        |
|           /   /      |               |            \         \       |
|          /   /       |               |             \         \      |
|      [Agent]---------|---------------|----------------------->      |
|          \   \       |    3 cells    |             /         /      |
|           \   \      |               |            /         /       |
|             \   \    +---------------+           /         /        |
|               \   \        FAR ONLY            /         /          |
|                 \   \      (50% acc)         /         /            |
|                   \   \                    /         /              |
|                     \   \--------------/         /                  |
|                       \                        /                    |
|                         \------------------/                        |
|                              5 cells                                |
|                                                                     |
|   Near FOV: 3 cells range, 90 deg angle, 90% accuracy               |
|   Far FOV:  5 cells range, 120 deg angle, 50% accuracy              |
|                                                                     |
|   Note: Far FOV ring = cells in far FOV but NOT in near FOV         |
|                                                                     |
+---------------------------------------------------------------------+
```

### Spatial Grid

```
+---------------------------------------------------------------------+
|                       SPATIAL GRID SYSTEM                            |
+---------------------------------------------------------------------+
|                                                                     |
|   Grid World (64x64 cells) divided into spatial buckets:            |
|                                                                     |
|   +---+---+---+---+---+---+---+---+                                 |
|   |   |   |   |   |   |   |   |   |  Each bucket = 2x2 cells        |
|   +---+---+---+---+---+---+---+---+                                 |
|   |   | A |   |   |   |   |   |   |  A = Query agent                |
|   +---+---+---+---+---+---+---+---+                                 |
|   |   |   |   |   |   |   |   |   |                                 |
|   +---+---+---+---+---+---+---+---+                                 |
|                                                                     |
|   Query Process:                                                    |
|   +-----------------------------------------------------+          |
|   |  1. Get agent's bucket: (x // 2.0, y // 2.0)         |          |
|   |  2. Check 3x3 neighborhood of buckets                |          |
|   |  3. Return all agents in those 9 buckets             |          |
|   +-----------------------------------------------------+          |
|                                                                     |
|   +---+---+---+                                                     |
|   | * | * | * |  * = Buckets checked for neighbors                  |
|   +---+---+---+                                                     |
|   | * | A | * |  A = Agent's bucket                                 |
|   +---+---+---+                                                     |
|   | * | * | * |                                                     |
|   +---+---+---+                                                     |
|                                                                     |
|   Performance Benefit:                                              |
|   Without spatial grid: 200 agents x 200 agents = 40,000 checks     |
|   With spatial grid: 200 agents x ~16 nearby = 3,200 checks         |
|   Speedup: ~12x                                                     |
|                                                                     |
+---------------------------------------------------------------------+
```

## Data Flow

### Observation Space (88 floats)

```
+---------------------------------------------------------------------+
|                       OBSERVATION STRUCTURE                          |
+---------------------------------------------------------------------+
|                                                                     |
|   AGENT STATE (indices 0-9):                                        |
|   +-----------------------------------------------------+          |
|   |  0: x position / GRID_SIZE                          |          |
|   |  1: y position / GRID_SIZE                          |          |
|   |  2: orientation / 360                                |          |
|   |  3: health / MAX_HEALTH                              |          |
|   |  4: stamina / MAX_STAMINA                            |          |
|   |  5: armor / MAX_ARMOR                                |          |
|   |  6: ammo_reserve / MAX_AMMO                          |          |
|   |  7: magazine_ammo / 30                               |          |
|   |  8: can_shoot (0 or 1)                               |          |
|   |  9: is_reloading (0 or 1)                            |          |
|   +-----------------------------------------------------+          |
|                                                                     |
|   NEAREST ENEMIES (indices 10-29, 5 enemies x 4 floats):            |
|   +-----------------------------------------------------+          |
|   |  For each enemy:                                     |          |
|   |    - relative_x (normalized)                         |          |
|   |    - relative_y (normalized)                         |          |
|   |    - health (normalized)                             |          |
|   |    - distance (normalized)                           |          |
|   +-----------------------------------------------------+          |
|                                                                     |
|   NEAREST ALLIES (indices 30-49, 5 allies x 4 floats):              |
|   +-----------------------------------------------------+          |
|   |  Same structure as enemies                           |          |
|   +-----------------------------------------------------+          |
|                                                                     |
|   TERRAIN IN FOV (indices 50-87, up to 38 cells):                   |
|   +-----------------------------------------------------+          |
|   |  Terrain type for each visible cell (normalized):    |          |
|   |    - EMPTY = 0.0                                     |          |
|   |    - BUILDING = 0.25                                 |          |
|   |    - FIRE = 0.5                                      |          |
|   |    - SWAMP = 0.75                                    |          |
|   |    - WATER = 1.0                                     |          |
|   |  Cells sorted by distance from agent                 |          |
|   +-----------------------------------------------------+          |
|                                                                     |
+---------------------------------------------------------------------+
```

### Combat Data Flow

```
+---------------------------------------------------------------------+
|                       COMBAT DATA FLOW                               |
+---------------------------------------------------------------------+
|                                                                     |
|   SHOOTER AGENT                                                     |
|   +----------------------------------------------------------+     |
|   | 1. can_shoot() checks:                                    |     |
|   |    ├── is_alive (health > 0)                              |     |
|   |    ├── shoot_cooldown <= 0                                |     |
|   |    ├── magazine_ammo > 0                                  |     |
|   |    └── not is_reloading                                   |     |
|   +----------------------------------------------------------+     |
|                               | If true                             |
|                               v                                     |
|   +----------------------------------------------------------+     |
|   | 2. get_nearby_agents(spatial_grid)                        |     |
|   |    Returns ~8-16 agents in adjacent cells                 |     |
|   +----------------------------------------------------------+     |
|                               |                                     |
|                               v                                     |
|   +----------------------------------------------------------+     |
|   | 3. get_targets_in_fov(nearby_agents, terrain)             |     |
|   |    Returns (near_targets, far_targets)                    |     |
|   |    - Filters by team (enemies only)                       |     |
|   |    - Filters by alive status                              |     |
|   |    - Checks FOV cone intersection                         |     |
|   |    - Checks terrain LOS blocking                          |     |
|   +----------------------------------------------------------+     |
|                               |                                     |
|                               v                                     |
|   +----------------------------------------------------------+     |
|   | 4. Target Selection                                       |     |
|   |    - Prefer near_targets (higher accuracy)                |     |
|   |    - Select closest target in preferred layer             |     |
|   |    - Apply accuracy = base * movement_penalty             |     |
|   +----------------------------------------------------------+     |
|                               |                                     |
|                               v                                     |
|   +----------------------------------------------------------+     |
|   | 5. shoot_at_target(target, accuracy)                      |     |
|   |    - consume_ammo()                                       |     |
|   |    - create_projectile() with accuracy deviation          |     |
|   |    - Reset shoot_cooldown = 0.5 seconds                   |     |
|   +----------------------------------------------------------+     |
|                               |                                     |
|                               v                                     |
|   PROJECTILE                                                        |
|   +----------------------------------------------------------+     |
|   | 6. Projectile travels:                                    |     |
|   |    - Speed: 15 cells/second                               |     |
|   |    - Checks collision each frame                          |     |
|   +----------------------------------------------------------+     |
|                               | On collision                        |
|                               v                                     |
|   TARGET AGENT                                                      |
|   +----------------------------------------------------------+     |
|   | 7. take_damage(25):                                       |     |
|   |    ├── If armor > 0: armor -= min(damage, armor)          |     |
|   |    └── Remaining damage: health -= damage                 |     |
|   |                                                           |     |
|   | 8. If health <= 0: is_alive becomes False                 |     |
|   +----------------------------------------------------------+     |
|                                                                     |
+---------------------------------------------------------------------+
```

## Design Patterns

### Pattern: Gymnasium Environment
The `TacticalCombatEnv` follows the standard Gymnasium API with `reset()`, `step()`, `render()`, `close()`, and defined observation/action spaces.

### Pattern: Entity-Component (Partial)
The Agent class uses a partial entity-component pattern where all agent data is in one dataclass, but behaviors are methods that could be extracted to separate systems.

### Pattern: Spatial Hashing
The SpatialGrid implements spatial hashing for O(1) neighbor lookups instead of O(n) scans.

### Pattern: Lazy Initialization
Renderer overlay surfaces are created on first use, allowing headless imports.

### Pattern: Factory Function
`create_projectile()`, `spawn_team()`, and `generate_random_terrain()` are factory functions that encapsulate entity creation logic.

### Pattern: Package-Based Organization
All modules are organized in the `combatenv` package with clear public API exports through `__init__.py`.

## Coordinate Systems

```
+---------------------------------------------------------------------+
|                      COORDINATE SYSTEMS                              |
+---------------------------------------------------------------------+
|                                                                     |
|   GRID COORDINATES (game logic)                                     |
|   +-------------------------------------+                           |
|   | (0,0)                    (63,0)     |                           |
|   |   +------------------------+        |                           |
|   |   |                        |        |                           |
|   |   |   Agent position is    |        |                           |
|   |   |   (x, y) as floats     |        |                           |
|   |   |   Range: 0.0 to 64.0   |        |                           |
|   |   |                        |        |                           |
|   |   +------------------------+        |                           |
|   | (0,63)                   (63,63)    |                           |
|   +-------------------------------------+                           |
|                                                                     |
|   PIXEL COORDINATES (rendering)                                     |
|   +-------------------------------------+                           |
|   | (0,0)                  (1024,0)     |                           |
|   |   +------------------------+        |                           |
|   |   |                        |        |                           |
|   |   |   Pixel = Grid * 16    |        |                           |
|   |   |   (CELL_SIZE = 16)     |        |                           |
|   |   |                        |        |                           |
|   |   +------------------------+        |                           |
|   | (0,1024)              (1024,1024)   |                           |
|   +-------------------------------------+                           |
|                                                                     |
|   ORIENTATION (degrees)                                             |
|   +-------------------------------------+                           |
|   |                                     |                           |
|   |            270 (up)                 |                           |
|   |               |                     |                           |
|   |               |                     |                           |
|   |  180 --------*-------- 0 (right)    |                           |
|   |  (left)      |                      |                           |
|   |               |                     |                           |
|   |            90 (down)                |                           |
|   |                                     |                           |
|   |   Pygame uses Y-down coordinates    |                           |
|   |   (0 = right, 90 = down)            |                           |
|   +-------------------------------------+                           |
|                                                                     |
+---------------------------------------------------------------------+
```

## Performance Considerations

### Optimization Strategies

1. **Spatial Grid**: Reduces collision checks from O(n^2) to O(n)
2. **FOV Caching**: FOVCache only recalculates when agent moves >0.3 cells or rotates >5 degrees
3. **Ray Casting FOV**: More efficient than bounding box for cone shapes
4. **Pre-allocated Surfaces**: FOV overlay surfaces created once, reused every frame
5. **Set Operations**: FOV overlap uses efficient set intersection/union
6. **Delta-time Updates**: Frame-rate independent physics
7. **Terrain LOS Caching**: LOS checks use simple grid lookups

### Bottlenecks

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Spatial grid build | O(n) | Runs once per frame |
| Neighbor query | O(1) | Constant time hash lookup |
| FOV calculation | O(agents * rays) | Cached; only recalculates on significant movement |
| Projectile collision | O(projectiles * agents) | Also checks building collision |
| Rendering | O(cells + agents) | Linear in visible elements |

### Memory Usage

- Agent: ~200 bytes per agent (dataclass with floats/ints)
- Projectile: ~100 bytes per projectile
- Spatial Grid: O(agents) dictionary entries
- Terrain Grid: O(grid_size^2) = 4KB for 64x64
- FOV Cache: O(agents) cached FOV cells per agent
- FOV Sets: O(visible cells) per team per layer
- Overlay Surfaces: 7 surfaces x 16x16 pixels x 4 bytes = ~7KB fixed
