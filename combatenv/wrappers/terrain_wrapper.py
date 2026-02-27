"""
TerrainWrapper - Adds terrain system to the environment.

This wrapper adds a 64x64 terrain grid with different terrain types
(empty, building, fire, forest, water) and processes terrain effects
on agents.

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = TerrainWrapper(env, obstacle_pct=0.05)
"""

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..terrain import TerrainType, TerrainGrid
from ..config import (
    GRID_SIZE,
    TERRAIN_OBSTACLE_PCT,
    FIRE_DAMAGE_PER_STEP,
)


class TerrainWrapper(BaseWrapper):
    """
    Wrapper that adds terrain to the environment.

    Provides a terrain grid with procedural generation and processes
    terrain effects on agents (fire damage, speed modifiers, etc.).

    Attributes:
        terrain_grid: The terrain grid for the environment
    """

    def __init__(
        self,
        env: gym.Env,
        obstacle_pct: float = TERRAIN_OBSTACLE_PCT,
    ):
        """
        Initialize the TerrainWrapper.

        Args:
            env: Base environment
            obstacle_pct: Percentage of cells to be obstacles (0.0-1.0)
        """
        super().__init__(env)

        self.obstacle_pct = obstacle_pct
        self.terrain_grid: Optional[TerrainGrid] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset and generate terrain.

        Args:
            seed: Random seed
            options: Additional options:
                - terrain_grid: Pre-made TerrainGrid to use
                - skip_terrain: If True, don't generate terrain

        Returns:
            Tuple of (observation, info)
        """
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Get grid size
        grid_size = getattr(self.env, 'grid_size', GRID_SIZE)

        # Check for pre-made terrain grid
        if options and "terrain_grid" in options:
            self.terrain_grid = options["terrain_grid"]
        elif options and options.get("skip_terrain", False):
            self.terrain_grid = None
        else:
            # Generate random terrain
            self.terrain_grid = TerrainGrid(grid_size, grid_size)
            terrain_rng = random.Random()
            if seed is not None:
                terrain_rng.seed(seed)
            self.terrain_grid.generate_random(
                obstacle_pct=self.obstacle_pct,
                rng=terrain_rng
            )

        # Relocate agents that spawned on non-empty terrain
        if self.terrain_grid is not None:
            self._relocate_agents_on_terrain(seed)

        # Update info with terrain data
        if self.terrain_grid:
            info["terrain_generated"] = True
            info["grid_size"] = (self.terrain_grid.width, self.terrain_grid.height)

        return obs, info

    def _relocate_agents_on_terrain(self, seed: Optional[int] = None) -> None:
        """
        Move agents off non-empty terrain to empty cells.

        Called after terrain generation to ensure agents only occupy empty cells.

        Args:
            seed: Random seed for reproducible relocations
        """
        agents = getattr(self.env, 'agents', [])
        if not agents or self.terrain_grid is None:
            return

        rng = random.Random(seed)
        grid_size = self.terrain_grid.width
        margin = 2.0

        for agent in agents:
            cell_x = int(agent.position[0])
            cell_y = int(agent.position[1])

            # Check if agent is on non-empty terrain
            terrain = self.terrain_grid.get(cell_x, cell_y)
            if terrain != TerrainType.EMPTY:
                # Find a new empty position
                for _ in range(100):  # Max attempts
                    new_x = rng.uniform(margin, grid_size - margin)
                    new_y = rng.uniform(margin, grid_size - margin)
                    new_cell_x = int(new_x)
                    new_cell_y = int(new_y)

                    if self.terrain_grid.get(new_cell_x, new_cell_y) == TerrainType.EMPTY:
                        agent.position = (new_x, new_y)
                        break

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step and process terrain effects.

        Args:
            action: Action for controlled agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Process terrain effects on agents
        if self.terrain_grid is not None:
            self._process_terrain_effects()

        return obs, reward, terminated, truncated, info

    def _process_terrain_effects(self) -> None:
        """
        Apply terrain effects to all alive agents.

        Effects:
        - Fire: Damage agents
        - Forest: Set in_forest flag (speed penalty)
        - Water: Set in_water flag (speed penalty, no shooting)
        """
        # Get alive agents from AgentWrapper
        alive_agents = getattr(self.env, 'alive_agents', [])
        if not alive_agents or self.terrain_grid is None:
            return

        for agent in alive_agents:
            # Get agent's cell position
            cell_x = int(agent.position[0])
            cell_y = int(agent.position[1])

            # Clamp to grid bounds
            cell_x = max(0, min(cell_x, self.terrain_grid.width - 1))
            cell_y = max(0, min(cell_y, self.terrain_grid.height - 1))

            terrain = self.terrain_grid.get(cell_x, cell_y)

            # Reset terrain flags
            agent.in_forest = False
            agent.in_water = False

            # Apply terrain effects
            if terrain == TerrainType.FIRE:
                old_health = agent.health
                agent.health -= FIRE_DAMAGE_PER_STEP
                if agent.health <= 0:
                    agent.health = 0
                    if old_health > 0:
                        print(f"DEBUG: Agent died from FIRE at cell ({cell_x}, {cell_y}), pos {agent.position}")
            elif terrain == TerrainType.FOREST:
                agent.in_forest = True
            elif terrain == TerrainType.WATER:
                agent.in_water = True

    def is_walkable(self, x: int, y: int) -> bool:
        """
        Check if a cell is walkable.

        Args:
            x: Cell x coordinate
            y: Cell y coordinate

        Returns:
            True if walkable, False if blocked (building)
        """
        if self.terrain_grid is None:
            return True
        return self.terrain_grid.is_walkable(x, y)

    def get_terrain(self, x: int, y: int) -> TerrainType:
        """
        Get terrain type at a cell.

        Args:
            x: Cell x coordinate
            y: Cell y coordinate

        Returns:
            TerrainType at the cell
        """
        if self.terrain_grid is None:
            return TerrainType.EMPTY
        return self.terrain_grid.get(x, y)

    @property
    def grid_size(self) -> int:
        """Get grid size from base environment."""
        return getattr(self.env, 'grid_size', GRID_SIZE)
