"""
GridWorld - Minimal base environment for the combat simulation.

This provides only the core grid coordinate space and Gymnasium interface.
All game mechanics (agents, terrain, combat, etc.) are added via wrappers.

Usage:
    # Minimal environment
    env = GridWorld(grid_size=64, render_mode="human")

    # Add functionality via wrappers
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = TerrainWrapper(env)
    # ... etc

    # Or use TacticalCombatEnv() for the full stack
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from combatenv.config import GRID_SIZE, FPS, WINDOW_SIZE, CELL_SIZE


class GridWorld(gym.Env):
    """
    Minimal base environment - just a coordinate space.

    This is the foundation that all wrappers build upon.
    By itself, it does almost nothing - just maintains grid dimensions
    and provides the Gymnasium interface structure.

    Attributes:
        grid_size: Size of the grid (default 64x64)
        render_mode: "human" for pygame, None for headless
        step_count: Number of steps taken this episode
    """

    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": FPS}

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the GridWorld environment.

        Args:
            grid_size: Size of the grid (default 64)
            render_mode: "human" for pygame rendering, None for headless
        """
        super().__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode

        # Step counter
        self.step_count = 0

        # Pygame resources (initialized lazily)
        self._screen = None
        self._clock = None
        self._pygame_initialized = False

        # Minimal action/observation spaces
        # These will be overridden by wrappers
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)

        self.step_count = 0

        # Minimal observation (just zeros - wrappers add real data)
        obs = np.zeros(2, dtype=np.float32)
        info: Dict[str, Any] = {}

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (interpreted by wrappers)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Minimal response (wrappers add real logic)
        obs = np.zeros(2, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        For render_mode="human", initializes pygame and clears screen.
        Wrappers add actual content to render.

        Returns:
            RGB array if render_mode="rgb_array", None otherwise
        """
        if self.render_mode == "human":
            self._init_pygame()
            self._screen.fill((40, 40, 40))  # Dark gray background
            return None
        elif self.render_mode == "rgb_array":
            self._init_pygame()
            self._screen.fill((40, 40, 40))
            import pygame
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)),
                axes=(1, 0, 2)
            )
        return None

    def _init_pygame(self) -> None:
        """Initialize pygame if not already done."""
        if self._pygame_initialized:
            return

        import pygame
        pygame.init()
        pygame.display.set_caption("GridWorld")

        self._screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self._clock = pygame.time.Clock()
        self._pygame_initialized = True

    def close(self) -> None:
        """Clean up resources."""
        if self._pygame_initialized:
            import pygame
            pygame.quit()
            self._pygame_initialized = False
            self._screen = None
            self._clock = None

    @property
    def screen(self):
        """Get the pygame screen surface."""
        return self._screen

    @property
    def clock(self):
        """Get the pygame clock."""
        return self._clock

    def flip_display(self) -> None:
        """Update the display (call after all wrappers have rendered)."""
        if self._pygame_initialized and self._screen is not None:
            import pygame
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(FPS)
