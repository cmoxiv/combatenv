# combatenv

A 2D tactical combat simulation environment for AI testing, built with Python and Pygame. Gymnasium-compatible for reinforcement learning research.

## Features

- **Gymnasium API**: Standard `reset()`, `step()`, `render()`, `close()` interface compatible with RL libraries
- **Two-Layer FOV System**: Near (3 cells, 90°, 99% accuracy) and far (5 cells, 120°, 80% accuracy) field of view
- **Projectile Combat**: Speed-based projectiles with collision detection and accuracy modifiers
- **Resource Management**: Stamina, armor, health, and ammunition systems
- **Dynamic Terrain**: Buildings, fire, swamp, and water with unique effects
- **200 Autonomous Agents**: 100 blue vs 100 red with wandering, targeting, and combat AI
- **Performance Optimized**: Spatial grid hashing for efficient collision detection (~12x speedup)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/combatenv.git
cd combatenv

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pygame numpy gymnasium
```

## Quick Start

```bash
python main.py
```

### Basic Usage

```python
from combatenv import TacticalCombatEnv, EnvConfig

# Create environment
env = TacticalCombatEnv(render_mode="human")
obs, info = env.reset(seed=42)

# Game loop
while env.process_events():
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Headless Training

```python
from combatenv import TacticalCombatEnv

env = TacticalCombatEnv(render_mode=None)  # No rendering
obs, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Controls

| Key | Action |
|-----|--------|
| ESC | Exit simulation |
| ` (backtick) | Toggle debug overlay |
| F | Toggle FOV overlay |
| ? (Shift+/) | Toggle keybindings help |

## Configuration

```python
from combatenv import TacticalCombatEnv, EnvConfig

config = EnvConfig(
    num_agents_per_team=50,      # Agents per team (default: 100)
    max_steps=1000,              # Episode length (default: 2000)
    respawn_enabled=False,       # Agent respawning
    terminate_on_team_elimination=True
)

env = TacticalCombatEnv(render_mode="human", config=config)
```

See `combatenv/config.py` for all 160+ tunable parameters including:
- Agent speed, health, and collision radius
- FOV ranges, angles, and accuracy
- Projectile speed, damage, and range
- Terrain density and effects
- Stamina/armor/ammo regeneration rates

## Observation Space

88-dimensional float vector (normalized 0-1):
- **0-9**: Agent state (position, orientation, health, stamina, armor, ammo)
- **10-29**: 5 nearest enemies (relative position, health, distance)
- **30-49**: 5 nearest allies (relative position, health, distance)
- **50-87**: Terrain types in FOV

## Action Space

Continuous 3D vector:
- `action[0]`: Forward/backward movement (-1 to 1)
- `action[1]`: Left/right rotation (-1 to 1)
- `action[2]`: Shoot (fires if > 0.5)

## Project Structure

```
combatenv/
├── main.py                 # Entry point
├── combatenv/
│   ├── environment.py      # Gymnasium environment
│   ├── agent.py            # Agent class and spawning
│   ├── terrain.py          # Terrain system
│   ├── projectile.py       # Projectile mechanics
│   ├── fov.py              # Field of view calculations
│   ├── spatial.py          # Spatial grid optimization
│   ├── renderer.py         # Pygame rendering
│   ├── input_handler.py    # Keyboard input
│   └── config.py           # Configuration constants
├── tests/                  # Unit tests (190 tests)
├── docs/                   # Documentation
└── diagnostics/            # Performance analysis tools
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Terrain Types

| Type | Effect |
|------|--------|
| Empty | Normal traversable terrain |
| Building | Blocks movement and line of sight |
| Fire | Deals 2 HP/step damage (bypasses armor) |
| Swamp | Traps agents for 0.5-1.5 seconds |
| Water | Impassable |

## License

MIT License - see LICENSE file for details.
