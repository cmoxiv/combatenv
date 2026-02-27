"""
AgentWrapper - Spawns and manages agents on the GridWorld.

This wrapper adds agent entities to the base GridWorld environment.
Agents are generic entities with position, orientation, health, and resources.
Team assignment is handled by TeamWrapper.

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base_wrapper import BaseWrapper
from ..agent import Agent, TeamType
from ..spatial import SpatialGrid
from ..config import (
    GRID_SIZE,
    AGENT_MAX_HEALTH,
    AGENT_MAX_STAMINA,
    AGENT_MAX_ARMOR,
    AGENT_MAX_AMMO,
    MAGAZINE_SIZE,
)


class AgentWrapper(BaseWrapper):
    """
    Wrapper that spawns and manages agents.

    Adds agent entities with position, orientation, health, and resources
    to the base GridWorld environment.

    Attributes:
        agents: List of all agents
        alive_agents: List of currently alive agents
        spatial_grid: Spatial partitioning for efficient queries
        controlled_agent: Agent controlled by the RL agent (first agent)
    """

    def __init__(
        self,
        env: gym.Env,
        num_agents: int = 200,
        cell_size: float = 2.0,
    ):
        """
        Initialize the AgentWrapper.

        Args:
            env: Base environment (GridWorld or wrapped)
            num_agents: Total number of agents to spawn
            cell_size: Spatial grid cell size for collision queries
        """
        super().__init__(env)

        self.num_agents = num_agents
        self.cell_size = cell_size

        # Agent state (initialized on reset)
        self.agents: List[Agent] = []
        self.alive_agents: List[Agent] = []
        self.spatial_grid: Optional[SpatialGrid] = None
        self.controlled_agent: Optional[Agent] = None

        # Override observation space to include agent data
        # For now, keep minimal - higher wrappers will define full obs
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Override action space for agent control
        # [move_x, move_y, shoot, think]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and spawn agents.

        Args:
            seed: Random seed for reproducibility
            options: Additional options:
                - spawn_positions: List of (x, y) positions for agents
                - skip_spawn: If True, don't spawn agents (for custom handling)

        Returns:
            Tuple of (observation, info)
        """
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Get grid size from base environment
        grid_size = getattr(self.env, 'grid_size', GRID_SIZE)

        # Initialize spatial grid
        self.spatial_grid = SpatialGrid(cell_size=self.cell_size)

        # Check if we should skip spawning
        skip_spawn = options.get("skip_spawn", False) if options else False
        if skip_spawn:
            self.agents = []
            self.alive_agents = []
            self.controlled_agent = None
            return self._get_obs(), info

        # Spawn agents
        spawn_positions = options.get("spawn_positions") if options else None
        self._spawn_agents(grid_size, spawn_positions, seed)

        # Build spatial grid
        self.spatial_grid.build(self.alive_agents)

        # Update info with agent data
        info["num_agents"] = len(self.agents)
        info["num_alive"] = len(self.alive_agents)

        return self._get_obs(), info

    def _spawn_agents(
        self,
        grid_size: int,
        spawn_positions: Optional[List[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Spawn agents at random or specified positions.

        Args:
            grid_size: Size of the grid
            spawn_positions: Optional list of positions
            seed: Random seed
        """
        import random

        if seed is not None:
            random.seed(seed)

        self.agents = []

        for i in range(self.num_agents):
            if spawn_positions and i < len(spawn_positions):
                x, y = spawn_positions[i]
            else:
                # Random position with margin from edges
                margin = 2.0
                x = random.uniform(margin, grid_size - margin)
                y = random.uniform(margin, grid_size - margin)

            # Agent dataclass requires position tuple and team
            # TeamWrapper will assign real teams; use "blue" as placeholder
            agent = Agent(
                position=(x, y),
                orientation=random.uniform(0, 360),
                team="blue",  # Placeholder - TeamWrapper reassigns
                health=AGENT_MAX_HEALTH,
                stamina=AGENT_MAX_STAMINA,
                armor=AGENT_MAX_ARMOR,
                ammo_reserve=AGENT_MAX_AMMO,
                magazine_ammo=MAGAZINE_SIZE,
            )
            self.agents.append(agent)

        self.alive_agents = list(self.agents)
        self.controlled_agent = self.agents[0] if self.agents else None

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with agent updates.

        Args:
            action: Action for the controlled agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update alive agents list
        self.alive_agents = [a for a in self.agents if a.is_alive]

        # Rebuild spatial grid
        if self.spatial_grid is not None:
            self.spatial_grid.build(self.alive_agents)

        # Update info
        info["num_alive"] = len(self.alive_agents)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Generate observation for the controlled agent.

        Returns:
            Normalized observation array
        """
        if self.controlled_agent is None or not self.controlled_agent.is_alive:
            return np.zeros(10, dtype=np.float32)

        agent = self.controlled_agent
        grid_size = getattr(self.env, 'grid_size', GRID_SIZE)

        # Basic agent state (10 floats)
        obs = np.array([
            agent.position[0] / grid_size,
            agent.position[1] / grid_size,
            agent.orientation / 360.0,
            agent.health / AGENT_MAX_HEALTH,
            agent.stamina / AGENT_MAX_STAMINA,
            agent.armor / AGENT_MAX_ARMOR,
            agent.magazine_ammo / MAGAZINE_SIZE,
            1.0 if agent.can_shoot else 0.0,
            1.0 if agent.is_reloading else 0.0,
            1.0 if agent.is_moving else 0.0,
        ], dtype=np.float32)

        return obs

    def get_neighbors(
        self,
        agent: Agent,
        radius: float,
    ) -> List[Agent]:
        """
        Get agents within radius of given agent.

        Args:
            agent: Center agent
            radius: Search radius

        Returns:
            List of nearby agents (excluding the center agent)
        """
        if self.spatial_grid is None:
            return []

        return self.spatial_grid.get_neighbors(agent, radius)

    @property
    def grid_size(self) -> int:
        """Get grid size from base environment."""
        return getattr(self.env, 'grid_size', GRID_SIZE)
