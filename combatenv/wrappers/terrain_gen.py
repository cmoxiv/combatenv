"""
Terrain generation wrapper for configurable terrain.

This wrapper allows configuring terrain generation at reset time,
enabling different terrain types for training and testing.

Usage:
    from combatenv.wrappers import TerrainGenWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = TerrainGenWrapper(env, terrain_type="corridors")

    # Empty terrain
    env = TerrainGenWrapper(env, terrain_type="empty")

    # Random terrain with custom parameters
    env = TerrainGenWrapper(env, terrain_type="random", obstacle_pct=0.1)
"""

from typing import Any, Dict, Optional, Tuple
import random

import gymnasium as gym

from combatenv.terrain import TerrainGrid
from combatenv.config import GRID_SIZE


class TerrainGenWrapper(gym.Wrapper):
    """
    Wrapper for terrain generation at reset time.

    Generates terrain before the base environment reset, allowing
    different terrain configurations without modifying the base env.

    Attributes:
        terrain_type: Type of terrain ("empty", "random", "corridors")
        seed: Optional seed for reproducible terrain
        obstacle_pct: Obstacle percentage for random terrain
        spawn_margin: Margin from edges to keep clear
    """

    def __init__(
        self,
        env,
        terrain_type: str = "random",
        seed: Optional[int] = None,
        obstacle_pct: float = 0.05,
        spawn_margin: int = 5,
        corridor_width: int = 2,
        grid_size: int = GRID_SIZE,
    ):
        """
        Initialize the terrain generation wrapper.

        Args:
            env: Base environment to wrap
            terrain_type: Type of terrain to generate:
                - "empty": No terrain (all walkable)
                - "random": Random terrain using operational grid
                - "corridors": Obstacles with corridors
            seed: Random seed for terrain generation
            obstacle_pct: Obstacle percentage for corridor terrain
            spawn_margin: Cells from edge to keep clear for spawning
            corridor_width: Width of corridors in corridor mode
            grid_size: Size of the terrain grid
        """
        super().__init__(env)

        self.terrain_type = terrain_type
        self.seed = seed
        self.obstacle_pct = obstacle_pct
        self.spawn_margin = spawn_margin
        self.corridor_width = corridor_width
        self.grid_size = grid_size

        # Track if we've generated terrain this episode
        self._terrain_generated = False

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset and generate terrain."""
        # Generate terrain grid
        terrain_grid = self._generate_terrain()

        # Pass to base env via options
        if 'options' not in kwargs or kwargs['options'] is None:
            kwargs['options'] = {}
        kwargs['options']['terrain_grid'] = terrain_grid

        self._terrain_generated = True
        return self.env.reset(**kwargs)

    def _generate_terrain(self) -> TerrainGrid:
        """
        Generate terrain based on configuration.

        Returns:
            TerrainGrid with generated terrain
        """
        grid = TerrainGrid(self.grid_size, self.grid_size)

        # Create RNG with seed if provided
        rng = random.Random(self.seed) if self.seed is not None else None

        if self.terrain_type == "empty":
            # Empty terrain - do nothing, grid is already empty
            pass
        elif self.terrain_type == "corridors":
            grid.generate_corridors(
                spawn_margin=self.spawn_margin,
                rng=rng,
            )
        elif self.terrain_type == "random":
            grid.generate_random(
                obstacle_pct=self.obstacle_pct,
                spawn_margin=self.spawn_margin,
                rng=rng,
            )
        else:
            # Default to empty if unknown type
            pass

        return grid

    def set_terrain_type(self, terrain_type: str) -> None:
        """
        Change terrain type for next reset.

        Args:
            terrain_type: New terrain type
        """
        self.terrain_type = terrain_type

    def set_seed(self, seed: Optional[int]) -> None:
        """
        Set seed for terrain generation.

        Args:
            seed: Random seed or None for random
        """
        self.seed = seed

    @property
    def current_terrain(self) -> Optional[TerrainGrid]:
        """Get the current terrain grid from base environment."""
        base_env = self.env.unwrapped
        return getattr(base_env, 'terrain_grid', None)
