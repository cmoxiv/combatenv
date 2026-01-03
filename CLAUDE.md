# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent tactical combat simulation built with Python and Pygame, packaged as `combatenv`. The simulation features 200 autonomous agents (100 blue, 100 red) engaged in team-based combat within a 64x64 grid world. It provides a Gymnasium-compatible reinforcement learning environment.

**Key Features:**
- Gymnasium API (`reset()`, `step()`, observation/action spaces)
- Two-layer FOV system (near: 3 cells/90 deg, far: 5 cells/120 deg)
- Projectile-based combat with accuracy modifiers (projectiles collide with buildings)
- Resource management (stamina, armor, ammo)
- Terrain system (buildings, fire, swamp, water)
- Spatial grid optimization for efficient collision detection
- FOV caching for performance optimization (~58 FPS with 200 agents)

## Development Environment

**Virtual Environment**: The project uses a Python virtual environment located at `~/.venvs/pg/`.

**Activation**:
- Activate: `source ~/.venvs/pg/bin/activate`
- Deactivate: `deactivate`

**Running the Application**:
```bash
source ~/.venvs/pg/bin/activate
python main.py
```

**Running Tests**:
```bash
source ~/.venvs/pg/bin/activate
python -m pytest tests/ -v
```

**Installing Dependencies**:
```bash
source ~/.venvs/pg/bin/activate
pip install pygame numpy gymnasium
```

## Project Structure

```
/Users/mo/Projects/tmp/
├── main.py              # Entry point - runs the simulation
├── map_editor.py        # Standalone map editor for custom terrain
├── combatenv/           # Main package
│   ├── __init__.py      # Package exports (public API)
│   ├── environment.py   # TacticalCombatEnv (Gymnasium env)
│   ├── agent.py         # Agent class with movement, combat, resources
│   ├── config.py        # All configuration constants
│   ├── projectile.py    # Projectile class and factory
│   ├── fov.py           # Field of view calculations
│   ├── spatial.py       # Spatial grid optimization
│   ├── terrain.py       # TerrainType enum and TerrainGrid
│   ├── renderer.py      # All rendering functions
│   └── map_io.py        # Save/load maps to JSON
├── tests/               # Test suite (193 tests)
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_environment.py
│   ├── test_terrain.py
│   ├── test_projectile.py
│   ├── test_fov.py
│   ├── test_spatial.py
│   ├── test_renderer.py
│   ├── test_map_io.py
│   └── test_integration.py
└── docs/                # Documentation
    ├── README.md        # User-facing documentation
    ├── ARCHITECTURE.md  # System design and data flow
    ├── API.md           # Configuration and API reference
    ├── CLAUDE.md        # This file
    ├── RENDERER_GUIDE.md
    ├── RENDERING_SUMMARY.md
    ├── HISTORY.md
    └── TODO.md
```

## Package Imports

Students and developers import from the `combatenv` package:

```python
# Main environment
from combatenv import TacticalCombatEnv, EnvConfig

# Agent and spawning
from combatenv import Agent, spawn_team, spawn_all_teams

# Terrain system
from combatenv import TerrainType, TerrainGrid

# Map I/O (save/load custom maps)
from combatenv import save_map, load_map

# Projectiles
from combatenv import Projectile, create_projectile

# Spatial optimization
from combatenv import SpatialGrid

# FOV functions
from combatenv import (
    get_fov_cells, get_team_fov_cells, get_fov_overlap,
    get_layered_fov_overlap, is_agent_visible_to_agent,
    get_visible_agents, normalize_angle, angle_difference,
    is_point_in_fov_cone
)

# Configuration values
from combatenv import config
GRID_SIZE = config.GRID_SIZE
PROJECTILE_DAMAGE = config.PROJECTILE_DAMAGE

# Renderer (usually not needed directly)
from combatenv import renderer
```

## Code Architecture

The application follows a Gymnasium environment pattern with game loop rendering:

1. **Environment Reset**: Spawns agents, generates terrain, initializes state
2. **Step Processing**: Processes action, updates all agents, handles combat
3. **Observation Generation**: Returns normalized observation vector
4. **Rendering** (optional): Visual display with Pygame

### Game Loop Phases (in environment.step())

1. Process controlled agent action
2. Rebuild spatial grid
3. Update resource timers (cooldowns, reload, stamina)
4. Combat phase (target detection, shooting)
5. Projectile updates (movement, collision detection with agents and buildings)
6. Agent movement (wandering AI)
7. Terrain effects (fire damage, swamp stuck)
8. FOV calculation (for rendering)
9. Render frame (if render_mode="human")

## Key Modules

- **combatenv/config.py**: All tunable parameters - modify here to change game behavior
- **combatenv/environment.py**: TacticalCombatEnv Gymnasium environment
- **combatenv/agent.py**: Agent dataclass with movement, combat, and resource methods
- **combatenv/terrain.py**: TerrainType enum and TerrainGrid class
- **combatenv/fov.py**: Ray-casting FOV calculation and overlap detection
- **combatenv/spatial.py**: O(1) neighbor queries via spatial hashing
- **combatenv/projectile.py**: Projectile entity with accuracy-based deviation
- **combatenv/renderer.py**: Modular rendering with layered FOV visualization

## Documentation

Comprehensive documentation is available in `docs/`:
- **README.md**: Installation, usage, game mechanics
- **ARCHITECTURE.md**: System design, data flow diagrams, design patterns
- **API.md**: Complete configuration reference and module APIs

## Controls

- **Q (Shift+Q)**: Exit simulation
- **` (backtick)**: Toggle debug overlay (on by default)
- **F**: Toggle FOV overlay (on by default)
- **? (Shift+/)**: Toggle keybindings help

## Common Development Tasks

**Adjusting Game Balance:**
Modify constants in `combatenv/config.py`:
- `NUM_AGENTS_PER_TEAM`: Agent count
- `NEAR_FOV_ACCURACY`, `FAR_FOV_ACCURACY`: Combat accuracy
- `PROJECTILE_DAMAGE`, `AGENT_MAX_HEALTH`: Health/damage balance
- `STAMINA_*`: Stamina economy
- `FIRE_DAMAGE_PER_STEP`: Fire terrain damage

**Adding New Behaviors:**
Agent behavior is in `combatenv/agent.py`:
- `wander()`: Autonomous movement AI
- `get_targets_in_fov()`: Target selection
- `shoot_at_target()`: Combat execution

**Modifying Terrain:**
Terrain system is in `combatenv/terrain.py`:
- `TerrainType`: Enum of terrain types
- `TerrainGrid`: 2D grid of terrain
- `generate_random_terrain()`: Procedural generation

**Modifying Visuals:**
Rendering is in `combatenv/renderer.py`:
- `render_agent()`: Agent appearance
- `render_terrain()`: Terrain rendering
- `render_fov_highlights()`: FOV visualization
- Color constants in `combatenv/config.py`

**Running Specific Tests:**
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_agent.py -v

# Run specific test class
python -m pytest tests/test_agent.py::TestAgent -v

# Run with coverage
python -m pytest tests/ --cov=combatenv
```
