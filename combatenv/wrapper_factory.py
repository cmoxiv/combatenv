"""
Wrapper Factory - Create environments using the composable wrapper stack.

This module provides factory functions to create environments using the new
layered wrapper architecture built on GridWorld.

The wrapper stack provides equivalent functionality to TacticalCombatEnv but
with a composable, testable architecture.

Usage:
    # Create full tactical environment (equivalent to TacticalCombatEnv)
    from combatenv import create_tactical_env
    env = create_tactical_env(render_mode="human")

    # Create minimal environment for testing
    from combatenv import create_minimal_env
    env = create_minimal_env(num_agents=20)

    # Create custom environment with specific wrappers
    from combatenv import GridWorld
    from combatenv.wrappers import AgentWrapper, TeamWrapper
    env = GridWorld()
    env = AgentWrapper(env, num_agents=100)
    env = TeamWrapper(env)
"""

from typing import Optional

from .gridworld import GridWorld
from .wrappers import (
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
from .config import GRID_SIZE, NUM_AGENTS_PER_TEAM


def create_minimal_env(
    grid_size: int = GRID_SIZE,
    num_agents: int = 20,
    render_mode: Optional[str] = None,
):
    """
    Create a minimal environment with just agents and teams.

    Args:
        grid_size: Size of the grid
        num_agents: Number of agents to spawn
        render_mode: "human" for pygame, None for headless

    Returns:
        Wrapped environment
    """
    env = GridWorld(grid_size=grid_size, render_mode=render_mode)
    env = AgentWrapper(env, num_agents=num_agents)
    env = TeamWrapper(env, teams=["blue", "red"])

    return env


def create_terrain_env(
    grid_size: int = GRID_SIZE,
    num_agents: int = 200,
    render_mode: Optional[str] = None,
):
    """
    Create an environment with agents, teams, and terrain.

    Args:
        grid_size: Size of the grid
        num_agents: Number of agents to spawn
        render_mode: "human" for pygame, None for headless

    Returns:
        Wrapped environment
    """
    env = GridWorld(grid_size=grid_size, render_mode=render_mode)
    env = AgentWrapper(env, num_agents=num_agents)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = BaseTerrainWrapper(env)

    return env


def create_combat_env(
    grid_size: int = GRID_SIZE,
    num_agents: int = 200,
    render_mode: Optional[str] = None,
    autonomous_combat: bool = True,
):
    """
    Create an environment with full combat capabilities.

    Args:
        grid_size: Size of the grid
        num_agents: Number of agents to spawn
        render_mode: "human" for pygame, None for headless
        autonomous_combat: Whether AI agents shoot automatically

    Returns:
        Wrapped environment
    """
    env = GridWorld(grid_size=grid_size, render_mode=render_mode)
    env = AgentWrapper(env, num_agents=num_agents)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = BaseTerrainWrapper(env)
    env = ProjectileWrapper(env)
    env = BaseFOVWrapper(env)
    env = BaseCombatWrapper(env, autonomous_combat=autonomous_combat)
    env = MovementWrapper(env, enable_wandering=True)

    return env


def create_full_env(
    grid_size: int = GRID_SIZE,
    num_agents: int = NUM_AGENTS_PER_TEAM * 2,
    render_mode: Optional[str] = None,
    autonomous_combat: bool = True,
    max_steps: int = 1000,
):
    """
    Create the full environment stack (similar to TacticalCombatEnv).

    This creates an environment with:
    - GridWorld base
    - Agent spawning and teams
    - Terrain generation
    - Projectile physics
    - FOV calculations
    - Combat system
    - Movement/wandering
    - 89-float observations
    - Termination conditions

    Args:
        grid_size: Size of the grid
        num_agents: Number of agents to spawn
        render_mode: "human" for pygame, None for headless
        autonomous_combat: Whether AI agents shoot automatically
        max_steps: Maximum steps per episode

    Returns:
        Fully wrapped environment
    """
    env = GridWorld(grid_size=grid_size, render_mode=render_mode)
    env = AgentWrapper(env, num_agents=num_agents)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = BaseTerrainWrapper(env)
    env = ProjectileWrapper(env)
    env = BaseFOVWrapper(env)
    env = BaseCombatWrapper(env, autonomous_combat=autonomous_combat)
    env = MovementWrapper(env, enable_wandering=True)
    env = BaseObservationWrapper(env)
    env = TerminationWrapper(
        env,
        max_steps=max_steps,
        terminate_on_controlled_death=True,
        terminate_on_team_elimination=True,
    )

    return env


# Alias for backwards compatibility hint
create_tactical_env = create_full_env
