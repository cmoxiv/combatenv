"""
Empty terrain wrapper for movement training.

Replaces all terrain with EMPTY cells, creating an obstacle-free
environment for basic movement learning.

Usage:
    from combatenv import TacticalCombatEnv
    from combatenv.wrappers import EmptyTerrainWrapper

    env = TacticalCombatEnv(render_mode="human")
    env = EmptyTerrainWrapper(env)
    # All terrain is now EMPTY - no buildings, fire, forest, or water
"""

from typing import Tuple, Dict, Any, Optional

import numpy as np
import gymnasium as gym

from combatenv.terrain import TerrainType, TerrainGrid


class EmptyTerrainWrapper(gym.Wrapper):
    """
    Wrapper that ensures all terrain cells are EMPTY.

    On reset, this wrapper clears the terrain grid to all EMPTY cells,
    providing an obstacle-free environment for movement training.

    This is useful for:
    - Basic movement learning without terrain complexity
    - Testing agent navigation in open spaces
    - Isolating movement behavior from terrain effects
    """

    def __init__(self, env):
        """
        Initialize the empty terrain wrapper.

        Args:
            env: Base environment (TacticalCombatEnv or wrapped)
        """
        super().__init__(env)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset and clear all terrain to EMPTY.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Clear terrain to all EMPTY
        self._clear_terrain()

        return obs, info

    def _clear_terrain(self) -> None:
        """Set all terrain cells to EMPTY."""
        # Find the base environment with terrain_grid
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'terrain_grid') and env.terrain_grid is not None:
                break
            env = env.env

        if hasattr(env, 'terrain_grid') and env.terrain_grid is not None:
            env.terrain_grid.clear()
