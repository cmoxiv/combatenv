# API Reference

This document provides a comprehensive reference for the `combatenv` package - a Gymnasium-compatible multi-agent tactical combat environment.

## Table of Contents

- [Package Overview](#package-overview)
- [Environment (combatenv.environment)](#environment-combatenvenvironment)
- [Configuration (combatenv.config)](#configuration-combatenvconfig)
- [Agent Module (combatenv.agent)](#agent-module-combatenvagent)
- [Terrain Module (combatenv.terrain)](#terrain-module-combatenvterrain)
- [FOV Module (combatenv.fov)](#fov-module-combatenvfov)
- [Projectile Module (combatenv.projectile)](#projectile-module-combatenvprojectile)
- [Spatial Module (combatenv.spatial)](#spatial-module-combatenvspatial)
- [Renderer Module (combatenv.renderer)](#renderer-module-combatenvrenderer)

---

## Package Overview

### Installation
```python
# All imports come from the combatenv package
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv import Agent, spawn_team, TerrainType, TerrainGrid
from combatenv import config  # Access configuration constants
```

### Quick Start
```python
from combatenv import TacticalCombatEnv

env = TacticalCombatEnv(render_mode="human")
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

---

## Environment (combatenv.environment)

### Class: TacticalCombatEnv

A Gymnasium-compatible environment for multi-agent tactical combat.

```python
from combatenv import TacticalCombatEnv, EnvConfig
```

#### Constructor

```python
TacticalCombatEnv(render_mode: Optional[str] = None, config: Optional[EnvConfig] = None)
```

**Parameters:**
- `render_mode`: `None` for headless, `"human"` for visual display
- `config`: Optional `EnvConfig` for custom settings

#### EnvConfig

```python
@dataclass
class EnvConfig:
    num_agents_per_team: int = 100
    respawn_enabled: bool = False
    respawn_delay: float = 1.0
    max_steps: Optional[int] = 1000
    terminate_on_team_elimination: bool = True
    terminate_on_controlled_death: bool = True
    allow_escape_exit: bool = True  # Allow Shift+Q to exit simulation
```

#### Spaces

**Observation Space:** `Box(shape=(88,), dtype=float32)`
- Normalized values in range [0, 1]
- Structure:
  - Indices 0-9: Agent state (position, orientation, health, stamina, armor, ammo, etc.)
  - Indices 10-29: 5 nearest enemies (4 floats each: rel_x, rel_y, health, distance)
  - Indices 30-49: 5 nearest allies (4 floats each: rel_x, rel_y, health, distance)
  - Indices 50-87: Terrain types for up to 38 FOV cells (normalized 0-1)
    - EMPTY=0.0, BUILDING=0.25, FIRE=0.5, SWAMP=0.75, WATER=1.0

**Action Space:** `Box(low=[-1, -1, 0], high=[1, 1, 1], shape=(3,), dtype=float32)`
- `action[0]`: Forward/backward movement (-1 to 1)
- `action[1]`: Rotation (-1=left, 1=right)
- `action[2]`: Shoot (>0.5 to fire)

#### Methods

##### `reset(seed=None, options=None) -> Tuple[np.ndarray, dict]`
Reset the environment to initial state.

**Returns:** `(observation, info)`

##### `step(action) -> Tuple[np.ndarray, float, bool, bool, dict]`
Execute one environment step.

**Returns:** `(observation, reward, terminated, truncated, info)`

##### `render() -> Optional[np.ndarray]`
Render the environment (if render_mode is set).

##### `close()`
Clean up resources.

##### `process_events() -> bool`
Process pygame events (window close, keyboard input).

**Returns:** True if simulation should continue, False if user requested exit (Shift+Q or window close)

**Note:** This method respects the `allow_escape_exit` option in EnvConfig. In headless mode (render_mode=None), always returns True.

#### Info Dictionary

```python
{
    'blue_kills': int,
    'red_kills': int,
    'blue_alive': int,
    'red_alive': int,
    'step_count': int,
    'controlled_agent_kills': int
}
```

---

## Configuration (combatenv.config)

All game parameters are centralized in `combatenv/config.py`.

```python
from combatenv import config

# Access any constant
grid_size = config.GRID_SIZE
damage = config.PROJECTILE_DAMAGE
```

### Window and Grid Configuration

| Constant      | Default | Type | Description                                                  |
|---------------|---------|------|--------------------------------------------------------------|
| `WINDOW_SIZE` | 1024    | int  | Window resolution in pixels (square)                         |
| `CELL_SIZE`   | 16      | int  | Size of each grid cell in pixels                             |
| `GRID_SIZE`   | 64      | int  | Derived: `WINDOW_SIZE // CELL_SIZE`. Grid dimensions (64x64) |

### Agent Configuration

| Constant              | Default | Type  | Description                                             |
|-----------------------|---------|-------|---------------------------------------------------------|
| `NUM_AGENTS_PER_TEAM` | 100     | int   | Number of agents spawned per team                       |
| `AGENT_SIZE_RATIO`    | 0.7     | float | Agent circle diameter as fraction of `CELL_SIZE`        |
| `AGENT_NOSE_RATIO`    | 0.4     | float | Orientation indicator length as fraction of `CELL_SIZE` |

### Movement Configuration

| Constant                  | Default | Type  | Description                                                  |
|---------------------------|---------|-------|--------------------------------------------------------------|
| `AGENT_MOVE_SPEED`        | 3.0     | float | Movement speed in grid cells per second                      |
| `AGENT_ROTATION_SPEED`    | 180.0   | float | Rotation speed in degrees per second                         |
| `WANDER_DIRECTION_CHANGE` | 0.02    | float | Probability per frame of changing wander direction (0.0-1.0) |

### Boundary and Collision Configuration

| Constant                       | Default | Type  | Description                               |
|--------------------------------|---------|-------|-------------------------------------------|
| `BOUNDARY_MARGIN`              | 0.5     | float | Minimum distance from grid edge in cells  |
| `BOUNDARY_DETECTION_THRESHOLD` | 1.0     | float | Distance to trigger boundary avoidance    |
| `AGENT_SPAWN_SPACING`          | 1.0     | float | Minimum distance between spawned agents   |
| `AGENT_COLLISION_RADIUS`       | 0.8     | float | Agent-to-agent collision detection radius |

### Combat FOV Layers

| Constant                    | Default | Type  | Description                     |
|-----------------------------|---------|-------|---------------------------------|
| `NEAR_FOV_RANGE`            | 3.0     | float | Near FOV range in grid cells    |
| `NEAR_FOV_ANGLE`            | 90.0    | float | Near FOV cone angle in degrees  |
| `NEAR_FOV_ACCURACY`         | 0.90    | float | Base accuracy in near FOV (90%) |
| `FAR_FOV_RANGE`             | 5.0     | float | Far FOV range in grid cells     |
| `FAR_FOV_ANGLE`             | 120.0   | float | Far FOV cone angle in degrees   |
| `FAR_FOV_ACCURACY`          | 0.50    | float | Base accuracy in far FOV (50%)  |
| `MOVEMENT_ACCURACY_PENALTY` | 0.5     | float | Accuracy multiplier when moving |

### Combat Projectiles

| Constant                | Default | Type  | Description                       |
|-------------------------|---------|-------|-----------------------------------|
| `PROJECTILE_SPEED`      | 15.0    | float | Projectile speed in cells/second  |
| `PROJECTILE_DAMAGE`     | 25      | int   | Damage dealt per hit              |
| `PROJECTILE_LIFETIME`   | 2.0     | float | Seconds before projectile expires |
| `PROJECTILE_RADIUS`     | 0.3     | float | Collision detection radius        |

### Combat Agent

| Constant                | Default | Type  | Description                          |
|-------------------------|---------|-------|--------------------------------------|
| `AGENT_MAX_HEALTH`      | 100     | int   | Maximum health points                |
| `SHOOT_COOLDOWN`        | 0.5     | float | Seconds between shots                |
| `FRIENDLY_FIRE_ENABLED` | True    | bool  | Allow teammates to damage each other |

### Terrain Configuration

| Constant                 | Default | Type  | Description                        |
|--------------------------|---------|-------|------------------------------------|
| `FIRE_DAMAGE_PER_STEP`   | 5       | int   | Damage per step on fire terrain    |
| `SWAMP_STUCK_MIN_STEPS`  | 3       | int   | Minimum steps stuck in swamp       |
| `SWAMP_STUCK_MAX_STEPS`  | 6       | int   | Maximum steps stuck in swamp       |

### Resource Management - Stamina

| Constant                             | Default | Type  | Description                             |
|--------------------------------------|---------|-------|-----------------------------------------|
| `AGENT_MAX_STAMINA`                  | 100.0   | float | Maximum stamina points                  |
| `STAMINA_REGEN_RATE_IDLE`            | 20.0    | float | Stamina per second when stationary      |
| `STAMINA_REGEN_RATE_MOVING`          | 5.0     | float | Stamina per second while moving         |
| `STAMINA_DRAIN_RATE`                 | 15.0    | float | Stamina consumed per second of movement |
| `LOW_STAMINA_THRESHOLD`              | 20.0    | float | Stamina level triggering speed penalty  |
| `MOVEMENT_SPEED_PENALTY_LOW_STAMINA` | 0.5     | float | Speed multiplier at low stamina (50%)   |

### Resource Management - Armor & Ammo

| Constant               | Default | Type  | Description                       |
|------------------------|---------|-------|-----------------------------------|
| `AGENT_MAX_ARMOR`      | 100     | int   | Maximum armor points              |
| `AGENT_MAX_AMMO`       | 1000    | int   | Total ammunition reserve          |
| `MAGAZINE_SIZE`        | 30      | int   | Rounds per magazine               |
| `RELOAD_TIME`          | 2.0     | float | Seconds to reload                 |
| `AUTO_RELOAD_ON_EMPTY` | True    | bool  | Auto-reload when magazine empties |

### Colors (RGB)

| Constant                | Default         | Description             |
|-------------------------|-----------------|-------------------------|
| `COLOR_BACKGROUND`      | (255, 255, 255) | White background        |
| `COLOR_GRID_LINES`      | (200, 200, 200) | Light gray grid lines   |
| `COLOR_BLUE_TEAM`       | (0, 0, 255)     | Blue team agents        |
| `COLOR_RED_TEAM`        | (255, 0, 0)     | Red team agents         |
| `COLOR_DEAD_AGENT`      | (128, 128, 128) | Dead agent color (gray) |
| `COLOR_BUILDING`        | (64, 64, 64)    | Building terrain        |
| `COLOR_FIRE`            | (255, 100, 0)   | Fire terrain            |
| `COLOR_SWAMP`           | (100, 150, 100) | Swamp terrain           |
| `COLOR_WATER`           | (100, 150, 255) | Water terrain           |

---

## Agent Module (combatenv.agent)

```python
from combatenv import Agent, spawn_team, spawn_all_teams
```

### Class: Agent

```python
@dataclass
class Agent:
    position: Tuple[float, float]
    orientation: float
    team: str  # "blue" | "red"
    health: int = AGENT_MAX_HEALTH
    stamina: float = AGENT_MAX_STAMINA
    armor: int = AGENT_MAX_ARMOR
    ammo_reserve: int = AGENT_MAX_AMMO
    magazine_ammo: int = MAGAZINE_SIZE
    shoot_cooldown: float = 0.0
    reload_timer: float = 0.0
    stuck_steps: int = 0
```

#### Properties

| Property          | Return Type | Description                               |
|-------------------|-------------|-------------------------------------------|
| `is_alive`        | bool        | True if `health > 0`                      |
| `is_moving`       | bool        | True if `wander_direction != 0`           |
| `is_reloading`    | bool        | True if `reload_timer > 0`                |
| `has_low_stamina` | bool        | True if `stamina < LOW_STAMINA_THRESHOLD` |

#### Movement Methods

##### `move_forward(speed, dt, other_agents, terrain_grid) -> bool`
Move agent forward in direction of orientation.

##### `rotate_left(degrees, dt) -> None`
Rotate agent counter-clockwise.

##### `rotate_right(degrees, dt) -> None`
Rotate agent clockwise.

##### `wander(dt, other_agents, terrain_grid) -> None`
Execute autonomous wandering behavior.

#### Combat Methods

##### `can_shoot() -> bool`
Check if agent can fire (alive, not on cooldown, has ammo, not reloading).

##### `get_targets_in_fov(potential_targets, terrain_grid) -> Tuple[List[Agent], List[Agent]]`
Get visible targets in near and far FOV layers.

##### `shoot_at_target(target, accuracy) -> Optional[Projectile]`
Fire at a target with specified accuracy.

##### `take_damage(damage) -> None`
Apply damage (armor absorbs first).

##### `apply_terrain_damage(damage) -> None`
Apply terrain damage (bypasses armor).

#### Resource Methods

##### `update_stamina(dt, is_moving) -> None`
Update stamina based on movement state.

##### `update_cooldown(dt) -> None`
Decrease shooting cooldown timer.

##### `update_reload(dt) -> None`
Update reload timer and complete reload when ready.

##### `respawn() -> None`
Reset agent to full health at spawn location.

### Factory Functions

##### `spawn_team(team, num_agents, terrain_grid) -> List[Agent]`
Spawn a team of agents in their designated quadrant.

##### `spawn_all_teams(num_blue, num_red, terrain_grid) -> Tuple[List[Agent], List[Agent]]`
Spawn both teams.

---

## Terrain Module (combatenv.terrain)

```python
from combatenv import TerrainType, TerrainGrid
```

### Enum: TerrainType

```python
class TerrainType(IntEnum):
    EMPTY = 0     # Passable, no effect
    BUILDING = 1  # Blocks movement and LOS
    FIRE = 2      # Damages agents (bypasses armor)
    SWAMP = 3     # Temporarily immobilizes agents
    WATER = 4     # Blocks movement, allows LOS
```

### Class: TerrainGrid

```python
class TerrainGrid:
    def __init__(self, width: int, height: int)
    def get(self, x: int, y: int) -> TerrainType
    def set(self, x: int, y: int, terrain: TerrainType)
    def is_walkable(self, x: int, y: int) -> bool
    def blocks_los(self, x: int, y: int) -> bool
    def clear(self)
```

### Factory Function

##### `generate_random_terrain(grid_size, seed) -> TerrainGrid`
Generate a random terrain grid with buildings, fire, swamp, and water.

---

## FOV Module (combatenv.fov)

```python
from combatenv import (
    get_fov_cells, get_team_fov_cells, get_fov_overlap,
    get_layered_fov_overlap, is_agent_visible_to_agent,
    get_visible_agents, normalize_angle, angle_difference,
    is_point_in_fov_cone
)
```

### Utility Functions

##### `normalize_angle(angle) -> float`
Normalize angle to [0, 360) range.

##### `angle_difference(angle1, angle2) -> float`
Calculate smallest angular difference (-180 to 180).

### FOV Calculation Functions

##### `is_point_in_fov_cone(agent_pos, agent_orientation, point, fov_angle, max_range) -> bool`
Check if a point falls within FOV cone.

##### `get_fov_cells(agent_pos, agent_orientation, fov_angle, max_range, terrain_grid) -> Set[Tuple[int, int]]`
Calculate all grid cells within FOV using ray casting.

##### `get_team_fov_cells(agents, fov_angle, max_range, terrain_grid) -> Set[Tuple[int, int]]`
Calculate combined FOV for an entire team.

##### `get_fov_overlap(blue_agents, red_agents, terrain_grid) -> Tuple[Set, Set, Set]`
Calculate FOV coverage and overlaps.

**Returns:** `(blue_only_cells, red_only_cells, overlap_cells)`

##### `get_layered_fov_overlap(...) -> Tuple[7 Sets]`
Calculate two-layer FOV with overlap detection.

**Returns:** Tuple of 7 sets:
1. `blue_near`: Blue team near FOV cells
2. `blue_far`: Blue team far FOV cells (excluding near)
3. `red_near`: Red team near FOV cells
4. `red_far`: Red team far FOV cells (excluding near)
5. `overlap_near_near`: Both teams near (darkest)
6. `overlap_mixed`: One near, one far (medium)
7. `overlap_far_far`: Both teams far (lightest)

##### `is_agent_visible_to_agent(observer, target, fov_angle, max_range, terrain_grid) -> bool`
Check if one agent can see another (considering terrain LOS blocking).

##### `get_visible_agents(observer, potential_targets, fov_angle, max_range, terrain_grid) -> List[Agent]`
Get all agents visible to an observer.

### Class: FOVCache

Performance optimization for FOV calculations. Only recalculates FOV when an agent has moved or rotated significantly.

```python
class FOVCache:
    def __init__(self, position_threshold: float = 0.3, angle_threshold: float = 5.0)
    def clear(self)
    def get_agent_fov(agent, near_angle, near_range, far_angle, far_range, terrain_grid) -> Tuple[Set, Set]
    def remove_agent(agent_id: int)
```

**Parameters:**
- `position_threshold`: Recalculate if position changed by this many cells (default: 0.3)
- `angle_threshold`: Recalculate if orientation changed by this many degrees (default: 5.0)

##### `get_fov_cache() -> FOVCache`
Get the global FOV cache instance.

---

## Projectile Module (combatenv.projectile)

```python
from combatenv import Projectile, create_projectile
```

### Class: Projectile

```python
@dataclass
class Projectile:
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    owner_team: str  # "blue" | "red"
    shooter_id: int
    damage: int = PROJECTILE_DAMAGE
    lifetime_remaining: float = PROJECTILE_LIFETIME
```

#### Methods

##### `update(dt, terrain_grid=None) -> bool`
Update position and lifetime.

**Parameters:**
- `dt`: Delta time in seconds
- `terrain_grid`: Optional TerrainGrid for building collision detection

**Returns:** True if projectile should be removed (expired, out of bounds, or hit building)

##### `check_collision(agent) -> bool`
Check collision with an agent.

**Collision Rules:**
- Never hits the shooter (`shooter_id` check)
- Never hits dead agents
- Respects `FRIENDLY_FIRE_ENABLED` setting

### Factory Function

##### `create_projectile(shooter_position, shooter_orientation, shooter_team, shooter_id, accuracy) -> Projectile`
Create a projectile fired from an agent with accuracy-based deviation.

---

## Spatial Module (combatenv.spatial)

```python
from combatenv import SpatialGrid
```

### Class: SpatialGrid

Spatial partitioning for efficient neighbor queries.

```python
class SpatialGrid:
    def __init__(self, cell_size: float = 2.0)
    def clear(self)
    def insert(self, agent: Agent)
    def build(self, agents: List[Agent])
    def get_nearby_agents(self, agent: Agent) -> List[Agent]
    def get_agents_in_cell(self, x: float, y: float) -> List[Agent]
    def get_statistics(self) -> Dict[str, float]
```

#### Statistics Dictionary

```python
{
    'num_cells': int,           # Number of occupied cells
    'total_agents': int,        # Total agents in grid
    'avg_agents_per_cell': float,
    'max_agents_per_cell': int
}
```

---

## Renderer Module (combatenv.renderer)

```python
from combatenv import renderer
# Or access specific functions
from combatenv.renderer import render_all, render_debug_overlay
```

### Rendering Functions

##### `render_background(surface) -> None`
Fill surface with white background.

##### `render_grid_lines(surface) -> None`
Draw grid lines.

##### `render_terrain(surface, terrain_grid) -> None`
Render terrain cells with appropriate colors.

##### `render_fov_highlights(surface, blue_near, blue_far, red_near, red_far, overlap_nn, overlap_mix, overlap_ff) -> None`
Render layered FOV highlighting.

##### `render_agent(surface, agent) -> None`
Render a single agent (circle with orientation indicator).

##### `render_agents(surface, agents) -> None`
Render list of agents.

##### `render_projectile(surface, projectile) -> None`
Render a single projectile.

##### `render_projectiles(surface, projectiles) -> None`
Render list of projectiles.

### Complete Rendering

##### `render_all(surface, ...) -> None`
Complete frame rendering pipeline.

**Render Order:**
1. Background (white)
2. Grid lines (gray)
3. Terrain
4. FOV highlights (layered)
5. Agents (circles)
6. Projectiles
7. Muzzle flashes

### Overlay Functions

##### `render_debug_overlay(surface, debug_info) -> None`
Render debug information panel.

##### `render_keybindings_overlay(surface) -> None`
Render keybindings help panel (centered).
