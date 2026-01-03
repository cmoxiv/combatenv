# Renderer Module Guide

## Overview

The `combatenv.renderer` module provides a complete, modular rendering system for the grid-world Pygame application. All rendering functions are pure visual components that accept game state data without modifying it.

## Core Design Principles

1. **Separation of Concerns**: Rendering functions only draw visuals; they never modify game state
2. **Modular Architecture**: Each rendering aspect has its own function
3. **Layered Rendering**: Proper rendering order ensures correct visual composition
4. **Performance**: Efficient drawing operations suitable for 60 FPS

## Rendering Functions

### Background & Grid

#### `render_background(surface)`
Renders solid white background.
- **Parameters**: `surface` - Pygame display surface
- **Usage**: Call first in rendering pipeline

#### `render_grid_lines(surface)`
Renders faint gray grid lines (64x64 grid, 16px cells).
- **Parameters**: `surface` - Pygame display surface
- **Usage**: Call after background, before terrain

### Terrain Rendering

#### `render_terrain(surface, terrain_grid)`
Renders terrain cells with appropriate colors.
- **Parameters**:
  - `surface` - Pygame display surface
  - `terrain_grid` - TerrainGrid object
- **Colors**:
  - Building: Dark gray
  - Fire: Orange
  - Swamp: Muted green
  - Water: Light blue
- **Usage**: Call after grid lines, before FOV highlights

### FOV Highlighting

#### `render_fov_highlights(surface, blue_near, blue_far, red_near, red_far, overlap_nn, overlap_mix, overlap_ff)`
Renders semi-transparent FOV overlays with team colors.
- **Parameters**:
  - `surface` - Pygame display surface
  - `blue_near/far` - Sets of (x, y) tuples for blue team FOV layers
  - `red_near/far` - Sets of (x, y) tuples for red team FOV layers
  - `overlap_*` - Sets of (x, y) tuples for overlapping FOV regions
- **Usage**: Call after terrain, before agents

### Agent Rendering

#### `render_agent(surface, agent)`
Renders a single agent as a colored circle with orientation indicator.
- **Parameters**:
  - `surface` - Pygame display surface
  - `agent` - Agent object with `position`, `orientation`, `team`, `is_alive` attributes
- **Visual Elements**:
  - Circle: 70% of cell size, team color (blue/red) or gray if dead
  - Nose line: Shows orientation direction, extends 40% beyond circle
- **Usage**: Call for individual agents or use `render_agents()`

#### `render_agents(surface, agents)`
Convenience function to render multiple agents.

### Complete Pipeline

#### `render_all(surface, ...)`
**Primary function for complete frame rendering.**

Renders entire scene in correct layer order:
1. Background (white)
2. Grid lines (gray)
3. Terrain (buildings, fire, swamp, water)
4. FOV highlights (semi-transparent)
5. Agents (both teams)
6. Projectiles
7. Muzzle flashes

## Integration Example

```python
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv.renderer import render_all, render_debug_overlay

# Using the environment (handles rendering internally)
env = TacticalCombatEnv(render_mode="human")
obs, info = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # Rendering happens automatically in step() when render_mode="human"
    if terminated or truncated:
        break
```

### Manual Rendering (Advanced)

```python
import pygame
from combatenv import Agent, spawn_all_teams, TerrainGrid
from combatenv import get_fov_overlap, config
from combatenv.renderer import render_all

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((config.WINDOW_SIZE, config.WINDOW_SIZE))
clock = pygame.time.Clock()

# Game state
terrain = TerrainGrid(config.GRID_SIZE, config.GRID_SIZE)
blue_agents, red_agents = spawn_all_teams(terrain_grid=terrain)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate FOV
    blue_fov, red_fov, overlap = get_fov_overlap(blue_agents, red_agents, terrain)

    # Render
    render_all(screen, blue_agents, red_agents, ...)

    pygame.display.flip()
    clock.tick(config.FPS)
```

## Configuration

All visual parameters are controlled via `combatenv/config.py`:

- `CELL_SIZE`: Grid cell size (16px)
- `GRID_SIZE`: Grid dimensions (64x64)
- `GRID_LINE_WIDTH`: Line thickness (1px)
- `AGENT_SIZE_RATIO`: Agent circle size (0.7 = 70%)
- `AGENT_NOSE_RATIO`: Nose line length (0.4 = 40%)
- `FOV_NEAR_ALPHA`: Near FOV transparency (50)
- `FOV_FAR_ALPHA`: Far FOV transparency (25)
- `OVERLAP_*_ALPHA`: Overlap transparency values
- `COLOR_*`: RGB color values

## Performance Considerations

- **Efficient Overlays**: Uses single surface blit per cell for alpha blending
- **Minimal Overdraw**: Proper layer ordering prevents redundant drawing
- **Cached Surfaces**: Temporary overlay surfaces created once, reused
- **60 FPS Target**: Optimized for real-time rendering

## Testing

Run the simulation:
```bash
source ~/.venvs/pg/bin/activate
python main.py
```

Run renderer tests:
```bash
source ~/.venvs/pg/bin/activate
python -m pytest tests/test_renderer.py -v
```

## Future Extensions

The modular design supports easy addition of:
- Particle effects
- UI overlays
- Animation systems
- Camera/viewport systems
- Debug visualization layers
- Sprite-based rendering (replace circle agents)

All extensions should follow the same pattern: accept data, render visuals, don't modify state.
