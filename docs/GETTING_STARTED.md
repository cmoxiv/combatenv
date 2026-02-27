# Getting Started: Building Environments with Wrappers

This guide shows how to build custom training environments by stacking Gymnasium wrappers. The combatenv wrapper system follows a composable architecture where each wrapper adds a specific capability.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [The Wrapper Stack](#the-wrapper-stack)
3. [Building Your First Environment](#building-your-first-environment)
4. [Adding Capabilities Step by Step](#adding-capabilities-step-by-step)
5. [Abstraction Levels](#abstraction-levels)
6. [Reward Shaping](#reward-shaping)
7. [Complete Examples](#complete-examples)
8. [Quick Reference](#quick-reference)

---

## Core Concepts

### What is a Wrapper?

A wrapper extends a Gymnasium environment with additional functionality:

```python
import gymnasium as gym

# Base environment
env = SomeEnvironment()

# Wrap it to add new behavior
env = SomeWrapper(env)
env = AnotherWrapper(env)

# Use it normally
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Each wrapper:
- Receives calls from outside (your code)
- Can modify observations, actions, or rewards
- Passes calls to the inner environment
- Can expose new attributes/methods

### Attribute Forwarding

All combatenv wrappers inherit from `BaseWrapper`, which forwards attribute access through the wrapper chain:

```python
from combatenv.wrappers import BaseWrapper

# If outer wrapper doesn't have 'agents', it checks inner wrappers
env = OuterWrapper(MiddleWrapper(InnerWrapper(base_env)))
agents = env.agents  # Found in InnerWrapper, accessible from outside
```

This means you can access `agents`, `units`, `terrain_grid`, etc. from any wrapper level.

---

## The Wrapper Stack

Environments are built by stacking wrappers from bottom to top:

```
┌─────────────────────────────────────┐
│  Your Code / RL Agent               │  ← You interact here
├─────────────────────────────────────┤
│  Utility Wrappers (optional)        │  ← Rendering, debug, keybindings
├─────────────────────────────────────┤
│  Discrete Wrappers                  │  ← Convert to discrete spaces
├─────────────────────────────────────┤
│  Reward Wrappers                    │  ← Shape rewards
├─────────────────────────────────────┤
│  Abstraction Wrappers               │  ← Units, strategic grid
├─────────────────────────────────────┤
│  Core Wrappers                      │  ← Combat, movement, FOV
├─────────────────────────────────────┤
│  Base Environment                   │  ← GridWorld or TacticalCombatEnv
└─────────────────────────────────────┘
```

---

## Building Your First Environment

### Option 1: Start from TacticalCombatEnv (Recommended)

The simplest approach uses the pre-built environment:

```python
from combatenv import TacticalCombatEnv, EnvConfig

# Create with default settings
env = TacticalCombatEnv()

# Or customize with EnvConfig
config = EnvConfig(
    num_agents_per_team=50,
    use_units=True,
    autonomous_combat=False,  # Disable combat for movement training
)
env = TacticalCombatEnv(config=config, render_mode="human")

# Standard Gymnasium interface
obs, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Option 2: Build from GridWorld (Full Control)

For maximum control, build from the base grid:

```python
from combatenv.gridworld import GridWorld
from combatenv.wrappers import (
    AgentWrapper,
    TeamWrapper,
    BaseTerrainWrapper,
    MovementWrapper,
    TerminationWrapper,
    RenderWrapper,
)

# Start with empty grid (render_mode must be set here on the base)
env = GridWorld(grid_size=64, render_mode="human")

# Add agents
env = AgentWrapper(env, num_agents=100)

# Assign teams
env = TeamWrapper(env)

# Add terrain
env = BaseTerrainWrapper(env)

# Enable movement
env = MovementWrapper(env)

# Add episode termination
env = TerminationWrapper(env, max_steps=500)

# Add rendering (must be last - wraps the complete environment)
env = RenderWrapper(env)

# Now use it
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Display the frame
    if terminated or truncated:
        obs, info = env.reset()
```

---

## Adding Capabilities Step by Step

### Step 1: Empty Grid

```python
from combatenv.gridworld import GridWorld

env = GridWorld(grid_size=64)
# Result: 64x64 empty grid, no agents, no terrain
```

### Step 2: Add Agents

```python
from combatenv.wrappers import AgentWrapper

env = AgentWrapper(env, num_agents=128)
# Result: 128 agents spawned at random positions
# New attributes: env.agents, env.alive_agents, env.spatial_grid
```

### Step 3: Assign Teams

```python
from combatenv.wrappers import TeamWrapper

env = TeamWrapper(env)
# Result: Agents assigned to blue (team=0) or red (team=1)
# Agents spawn in team corners (blue: top-left, red: bottom-right)
```

### Step 4: Add Terrain

```python
from combatenv.wrappers import BaseTerrainWrapper

env = BaseTerrainWrapper(env)
# Result: Procedural terrain generation
# New attributes: env.terrain_grid
# Terrain types: EMPTY, WATER, FOREST, FIRE, OBSTACLE
```

### Step 5: Add Combat System

```python
from combatenv.wrappers import (
    ProjectileWrapper,
    BaseFOVWrapper,
    BaseCombatWrapper,
)

env = ProjectileWrapper(env)
# Result: Projectile entity system
# New attributes: env.projectiles

env = BaseFOVWrapper(env)
# Result: Field of view calculations
# Agents can only see enemies in their FOV cone

env = BaseCombatWrapper(env)
# Result: Shooting mechanics, damage, health
```

### Step 6: Add Movement

```python
from combatenv.wrappers import MovementWrapper

env = MovementWrapper(env, enable_wandering=True)
# Result: Agents move autonomously (wandering AI)
# For training, you may want enable_wandering=False
```

### Step 7: Add Observations

```python
from combatenv.wrappers import BaseObservationWrapper

env = BaseObservationWrapper(env)
# Result: Normalized observation vectors
# Observation includes: position, orientation, health, stamina, nearby agents
```

### Step 8: Add Termination

```python
from combatenv.wrappers import TerminationWrapper

env = TerminationWrapper(env, max_steps=500)
# Result: Episode ends after max_steps or when team eliminated
```

### Complete Minimal Stack

```python
from combatenv.gridworld import GridWorld
from combatenv.wrappers import (
    AgentWrapper,
    TeamWrapper,
    BaseTerrainWrapper,
    ProjectileWrapper,
    BaseFOVWrapper,
    BaseCombatWrapper,
    MovementWrapper,
    BaseObservationWrapper,
    TerminationWrapper,
)

def create_combat_env(num_agents=128, max_steps=500):
    env = GridWorld(grid_size=64)
    env = AgentWrapper(env, num_agents=num_agents)
    env = TeamWrapper(env)
    env = BaseTerrainWrapper(env)
    env = ProjectileWrapper(env)
    env = BaseFOVWrapper(env)
    env = BaseCombatWrapper(env)
    env = MovementWrapper(env, enable_wandering=False)
    env = BaseObservationWrapper(env)
    env = TerminationWrapper(env, max_steps=max_steps)
    return env

env = create_combat_env()
```

---

## Abstraction Levels

The wrapper system supports three abstraction levels for training:

### Tactical Level (Individual Agents)

Control individual agents with fine-grained actions:

```python
from combatenv import TacticalCombatEnv
from combatenv.wrappers import (
    DiscreteObservationWrapper,
    DiscreteActionWrapper,
)

env = TacticalCombatEnv()
env = DiscreteObservationWrapper(env)  # 8,192 discrete states
env = DiscreteActionWrapper(env)        # 10 discrete actions

# Actions: NOOP, FORWARD, BACKWARD, LEFT, RIGHT,
#          NE, NW, SE, SW, SHOOT
```

### Operational Level (Units/Squads)

Control units of 8 agents with waypoint commands:

```python
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv.wrappers import (
    OperationalWrapper,
    OperationalDiscreteObsWrapper,
    OperationalDiscreteActionWrapper,
)

config = EnvConfig(use_units=True, num_units_per_team=8, agents_per_unit=8)
env = TacticalCombatEnv(config=config)
env = OperationalWrapper(env)
env = OperationalDiscreteObsWrapper(env)   # 432 discrete states
env = OperationalDiscreteActionWrapper(env) # 19 discrete actions

# Actions: Move to operational grid cell (0-63), change stance, dispatch
```

### Strategic Level (High-Level Commands)

Observe and command across a 4x4 strategic grid:

```python
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv.wrappers import (
    OperationalWrapper,
    StrategicWrapper,
    StrategicDiscreteObsWrapper,
    StrategicDiscreteActionWrapper,
)

config = EnvConfig(use_units=True)
env = TacticalCombatEnv(config=config)
env = OperationalWrapper(env)
env = StrategicWrapper(env)
env = StrategicDiscreteObsWrapper(env)    # 2,916 discrete states
env = StrategicDiscreteActionWrapper(env)  # 25 discrete actions

# Observations: 4x4 terrain summary + unit positions
# Actions: Dispatch units to strategic grid cells
```

---

## Reward Shaping

Stack reward wrappers to shape learning:

### Distance-Based Rewards

```python
from combatenv.wrappers import ChebyshevRewardWrapper

env = ChebyshevRewardWrapper(env, scale=1.0)
# Reward: +scale when agent moves closer to waypoint
# Reward: -scale when agent moves farther from waypoint
```

### Unit Cohesion Rewards

```python
from combatenv.wrappers import (
    CohesionRewardWrapper,
    UnitChebyshevWrapper,
)

env = CohesionRewardWrapper(env, bonus=0.5, penalty=-0.5)
# Reward: +bonus when unit stays together
# Reward: penalty when unit spreads too far

env = UnitChebyshevWrapper(env, scale=0.5)
# Reward based on unit centroid distance to waypoint
```

### Combined Reward Stack

```python
from combatenv.wrappers import (
    ChebyshevRewardWrapper,
    UnitChebyshevWrapper,
    AntiClumpRewardWrapper,
)

env = TacticalCombatEnv(config=config)
env = OperationalWrapper(env)

# Stack multiple reward shapers
env = ChebyshevRewardWrapper(env, scale=0.5)    # Individual distance
env = UnitChebyshevWrapper(env, scale=0.5)       # Unit distance
env = AntiClumpRewardWrapper(env, penalty=-0.1)  # Spread out

# Total reward = base + chebyshev + unit_chebyshev + anti_clump
```

---

## Complete Examples

### Example 1: Movement Training Environment

Train units to navigate to waypoints without combat:

```python
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv.wrappers import (
    OperationalWrapper,
    WaypointTaskWrapper,
    OperationalDiscreteObsWrapper,
    OperationalDiscreteActionWrapper,
    ChebyshevRewardWrapper,
    RenderWrapper,
)

def create_movement_training_env(render=False):
    # Base environment with combat disabled
    config = EnvConfig(
        use_units=True,
        num_units_per_team=8,
        agents_per_unit=8,
        autonomous_combat=False,  # No shooting
        autonomous_wandering=False,  # No random movement
    )
    env = TacticalCombatEnv(config=config, render_mode="human" if render else None)

    # Add unit abstraction
    env = OperationalWrapper(env)

    # Add waypoint task (assigns goals, tracks completion)
    env = WaypointTaskWrapper(env)

    # Add reward shaping
    env = ChebyshevRewardWrapper(env, scale=1.0)

    # Discretize for Q-learning
    env = OperationalDiscreteObsWrapper(env)
    env = OperationalDiscreteActionWrapper(env)

    # Optional rendering
    if render:
        env = RenderWrapper(env)

    return env

# Usage
env = create_movement_training_env(render=True)
obs, info = env.reset()

for episode in range(100):
    obs, info = env.reset()
    total_reward = 0

    for step in range(500):
        action = env.action_space.sample()  # Replace with your policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Episode {episode}: reward={total_reward:.2f}")
```

### Example 2: Combat Training Environment

Train tactical combat with full action space:

```python
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv.wrappers import (
    DiscreteObservationWrapper,
    DiscreteActionWrapper,
    TacticalCombatRewardWrapper,
)

def create_combat_training_env():
    config = EnvConfig(
        num_agents_per_team=50,
        use_units=False,  # Individual agent control
    )
    env = TacticalCombatEnv(config=config)

    # Combat-focused rewards
    env = TacticalCombatRewardWrapper(env)

    # Discretize for Q-learning
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)

    return env
```

### Example 3: Multi-Agent Training

Train all agents simultaneously:

```python
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv.wrappers import (
    MultiAgentWrapper,
    DiscreteObservationWrapper,
    DiscreteActionWrapper,
)

def create_multi_agent_env():
    config = EnvConfig(num_agents_per_team=64)
    env = TacticalCombatEnv(config=config)

    # Enable multi-agent interface
    env = MultiAgentWrapper(env)

    # Discretize
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)

    return env

env = create_multi_agent_env()
obs_dict, info = env.reset()

# obs_dict = {0: obs_agent_0, 1: obs_agent_1, ...}
# Take action for each agent
actions = {agent_id: env.action_space.sample() for agent_id in obs_dict}
obs_dict, rewards, terminated, truncated, info = env.step(actions)
```

### Example 4: Interactive Demo with Debug

```python
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv.wrappers import (
    OperationalWrapper,
    KeybindingsWrapper,
    DebugOverlayWrapper,
    RenderWrapper,
)

def create_interactive_demo():
    config = EnvConfig(use_units=True)
    env = TacticalCombatEnv(config=config, render_mode="human")

    # Add unit control
    env = OperationalWrapper(env)

    # Add keyboard controls (1-8 select units, click sets waypoint)
    env = KeybindingsWrapper(env)

    # Add debug overlay (press ` to toggle)
    env = DebugOverlayWrapper(env)

    # Rendering
    env = RenderWrapper(env)

    return env

# Run interactive demo
env = create_interactive_demo()
obs, info = env.reset()

running = True
while running:
    action = 0  # NOOP - keyboard controls handle input
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

---

## Quick Reference

### Factory Functions

Use `wrapper_factory.py` for common configurations:

```python
from combatenv.wrapper_factory import (
    create_minimal_env,   # GridWorld + Agents + Teams
    create_terrain_env,   # + Terrain
    create_combat_env,    # + Combat system
    create_full_env,      # Full TacticalCombatEnv equivalent
)

env = create_combat_env(num_agents=100, render_mode="human")
```

### Common Wrapper Imports

```python
# Core wrappers
from combatenv.wrappers import (
    AgentWrapper,
    TeamWrapper,
    BaseTerrainWrapper,
    ProjectileWrapper,
    BaseFOVWrapper,
    BaseCombatWrapper,
    MovementWrapper,
    BaseObservationWrapper,
    TerminationWrapper,
)

# Abstraction wrappers
from combatenv.wrappers import (
    UnitWrapper,
    OperationalWrapper,
    StrategicWrapper,
    MultiAgentWrapper,
)

# Discrete wrappers (for Q-learning)
from combatenv.wrappers import (
    DiscreteObservationWrapper,
    DiscreteActionWrapper,
    OperationalDiscreteObsWrapper,
    OperationalDiscreteActionWrapper,
    StrategicDiscreteObsWrapper,
    StrategicDiscreteActionWrapper,
)

# Reward wrappers
from combatenv.wrappers import (
    ChebyshevRewardWrapper,
    UnitChebyshevWrapper,
    CohesionRewardWrapper,
    AntiClumpRewardWrapper,
    FOVCoverageRewardWrapper,
    TacticalCombatRewardWrapper,
)

# Utility wrappers
from combatenv.wrappers import (
    RenderWrapper,
    KeybindingsWrapper,
    DebugOverlayWrapper,
    TerminalLogWrapper,
    WaypointTaskWrapper,
    QLearningWrapper,
)
```

### Environment Config Options

```python
from combatenv import EnvConfig

config = EnvConfig(
    # Agent settings
    num_agents_per_team=100,

    # Unit settings
    use_units=True,
    num_units_per_team=8,
    agents_per_unit=8,

    # Behavior toggles
    autonomous_combat=True,    # Agents shoot automatically
    autonomous_wandering=True, # Agents move randomly

    # Episode settings
    max_steps=500,
)
```

---

## Next Steps

- See `docs/API.md` for complete wrapper API documentation
- See `docs/ARCHITECTURE.md` for system design details
- Check `train_*.py` scripts for complete training examples
- Explore `combatenv/wrappers/` source code for implementation details
