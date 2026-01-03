"""
Combat Environment - Grid-World Multi-Agent Tactical Simulation

A Gymnasium-compatible environment for multi-agent reinforcement learning
featuring 200 autonomous agents (100 blue, 100 red) in team-based combat.

Key Features:
    - Two-layer FOV system (near: 3 cells/90°, far: 5 cells/120°)
    - Projectile-based combat with accuracy modifiers
    - Resource management (stamina, armor, ammo)
    - Terrain system (buildings, fire, swamp, water)
    - Spatial grid optimization for collision detection

Basic Usage:
    >>> from combatenv import TacticalCombatEnv
    >>> env = TacticalCombatEnv(render_mode="human")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)

Advanced Usage:
    >>> from combatenv import TacticalCombatEnv, EnvConfig
    >>> config = EnvConfig(num_agents_per_team=50, max_steps=500)
    >>> env = TacticalCombatEnv(render_mode="human", config=config)

Classes:
    TacticalCombatEnv: Main Gymnasium environment
    EnvConfig: Environment configuration dataclass
    Agent: Agent entity with movement, combat, resources
    TerrainGrid: Grid-based terrain storage
    TerrainType: Enum of terrain types (EMPTY, BUILDING, FIRE, SWAMP, WATER)
    Projectile: Projectile entity
    SpatialGrid: Spatial partitioning for collision detection

Functions:
    spawn_team: Spawn a team of agents
    spawn_all_teams: Spawn both blue and red teams
    create_projectile: Factory function for projectiles
"""

# Version
__version__ = "0.1.0"

# Core environment
from .environment import TacticalCombatEnv, EnvConfig, OBS_SIZE

# Agent system
from .agent import Agent, spawn_team, spawn_all_teams

# Terrain system
from .terrain import TerrainType, TerrainGrid

# Projectile system
from .projectile import Projectile, create_projectile

# Spatial partitioning
from .spatial import SpatialGrid

# Field of view
from .fov import (
    get_fov_cells,
    get_team_fov_cells,
    get_fov_overlap,
    get_layered_fov_overlap,
    is_point_in_fov_cone,
    is_agent_visible_to_agent,
    get_visible_agents,
    normalize_angle,
    angle_difference,
    FOVCache,
    get_fov_cache
)

# Configuration (for advanced users)
from . import config

# Renderer (for custom rendering)
from . import renderer

# Public API
__all__ = [
    # Environment
    "TacticalCombatEnv",
    "EnvConfig",
    "OBS_SIZE",

    # Agents
    "Agent",
    "spawn_team",
    "spawn_all_teams",

    # Terrain
    "TerrainType",
    "TerrainGrid",

    # Projectiles
    "Projectile",
    "create_projectile",

    # Spatial
    "SpatialGrid",

    # FOV
    "get_fov_cells",
    "get_team_fov_cells",
    "get_fov_overlap",
    "get_layered_fov_overlap",
    "is_point_in_fov_cone",
    "is_agent_visible_to_agent",
    "get_visible_agents",
    "normalize_angle",
    "angle_difference",
    "FOVCache",
    "get_fov_cache",

    # Modules
    "config",
    "renderer",
]
