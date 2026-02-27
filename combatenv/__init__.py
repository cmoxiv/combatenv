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
    TerrainType: Enum of terrain types (EMPTY, OBSTACLE, FIRE, FOREST, WATER)
    Projectile: Projectile entity
    SpatialGrid: Spatial partitioning for collision detection

Functions:
    spawn_team: Spawn a team of agents
    spawn_all_teams: Spawn both blue and red teams
    create_projectile: Factory function for projectiles
    save_map: Save a terrain grid to JSON file
    load_map: Load a terrain grid from JSON file
"""

# Version
__version__ = "0.1.1"

# Core environment
from .environment import TacticalCombatEnv, EnvConfig, OBS_SIZE

# Base environment (for wrapper architecture)
from .gridworld import GridWorld

# Factory functions for wrapper-based environments
from .wrapper_factory import (
    create_minimal_env,
    create_terrain_env,
    create_combat_env,
    create_full_env,
    create_tactical_env,
)

# Strategic environment
from .strategic_env import StrategicCombatEnv

# Wrappers (for RL training)
from .wrappers import (
    MultiAgentWrapper,
    RewardWrapper,
    DiscreteObservationWrapper,
    DiscreteActionWrapper,
    UnitWrapper,
    CohesionRewardWrapper,
    ChebyshevRewardWrapper,
    MovementObservationWrapper,
    EmptyTerrainWrapper,
)

# Agent system
from .agent import Agent, spawn_team, spawn_all_teams

# Unit system
from .unit import (
    Unit,
    spawn_unit,
    spawn_units_for_team,
    spawn_all_units,
    get_all_agents_from_units,
    get_unit_for_agent,
)

# Terrain system
from .terrain import TerrainType, TerrainGrid

# Map I/O
from .map_io import save_map, load_map

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

# Boids flocking
from .boids import (
    calculate_cohesion_force,
    calculate_separation_force,
    calculate_alignment_force,
    calculate_waypoint_force,
    calculate_boids_steering,
    steering_to_orientation,
    blend_steering_with_random,
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
    "GridWorld",
    "StrategicCombatEnv",
    # Factory functions
    "create_minimal_env",
    "create_terrain_env",
    "create_combat_env",
    "create_full_env",
    "create_tactical_env",

    # Wrappers
    "MultiAgentWrapper",
    "RewardWrapper",
    "DiscreteObservationWrapper",
    "DiscreteActionWrapper",
    "UnitWrapper",
    "CohesionRewardWrapper",
    "ChebyshevRewardWrapper",
    "MovementObservationWrapper",
    "EmptyTerrainWrapper",

    # Agents
    "Agent",
    "spawn_team",
    "spawn_all_teams",

    # Units
    "Unit",
    "spawn_unit",
    "spawn_units_for_team",
    "spawn_all_units",
    "get_all_agents_from_units",
    "get_unit_for_agent",

    # Terrain
    "TerrainType",
    "TerrainGrid",

    # Map I/O
    "save_map",
    "load_map",

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

    # Boids
    "calculate_cohesion_force",
    "calculate_separation_force",
    "calculate_alignment_force",
    "calculate_waypoint_force",
    "calculate_boids_steering",
    "steering_to_orientation",
    "blend_steering_with_random",

    # Modules
    "config",
    "renderer",
]
