"""
Single-agent discrete action wrapper for tabular Q-learning.

Converts discrete action indices to continuous action arrays for single-agent environments.

Action Space Design (8 actions):
    - 4 movement directions: North, South, East, West
    - 2 shoot options: shoot or no-shoot
    - Total: 4 × 2 = 8 discrete actions

Action Mapping:
    0: North (no shoot)     -> [0.0, -1.0, 0.0, 0.0]
    1: South (no shoot)     -> [0.0,  1.0, 0.0, 0.0]
    2: East (no shoot)      -> [1.0,  0.0, 0.0, 0.0]
    3: West (no shoot)      -> [-1.0, 0.0, 0.0, 0.0]
    4: North + Shoot        -> [0.0, -1.0, 1.0, 0.0]
    5: South + Shoot        -> [0.0,  1.0, 1.0, 0.0]
    6: East + Shoot         -> [1.0,  0.0, 1.0, 0.0]
    7: West + Shoot         -> [-1.0, 0.0, 1.0, 0.0]

Usage:
    from combatenv.wrappers import SingleAgentDiscreteObsWrapper, SingleAgentDiscreteActionWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = SingleAgentDiscreteObsWrapper(env)
    env = SingleAgentDiscreteActionWrapper(env)

    obs, info = env.reset()  # obs is int in [0, 2047]
    next_obs, reward, terminated, truncated, info = env.step(3)  # Move West
"""

from typing import Tuple, Dict, List

import numpy as np
import gymnasium as gym


class SingleAgentDiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts discrete action index to continuous action array.

    This wrapper maps 8 discrete actions to the environment's continuous
    action space [move_x, move_y, shoot, think].

    Attributes:
        n_actions: Number of discrete actions (8)
        n_states: Number of discrete states (from observation wrapper)
    """

    # Action mapping: discrete index -> (move_x, move_y, shoot, think)
    ACTION_MAP = {
        0: np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),   # North
        1: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),    # South
        2: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),    # East
        3: np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),   # West
        4: np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float32),   # North + Shoot
        5: np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),    # South + Shoot
        6: np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),    # East + Shoot
        7: np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32),   # West + Shoot
    }

    # Human-readable action names (for debugging)
    ACTION_NAMES = [
        "North",
        "South",
        "East",
        "West",
        "North + Shoot",
        "South + Shoot",
        "East + Shoot",
        "West + Shoot",
    ]

    def __init__(self, env):
        """
        Initialize the discrete action wrapper.

        Args:
            env: Environment (should have n_states attribute)
        """
        super().__init__(env)
        self.n_actions = len(self.ACTION_MAP)

        # Expose n_states from the observation wrapper
        self.n_states = getattr(env, 'n_states', 2048)

        # Update action space
        self.action_space = gym.spaces.Discrete(self.n_actions)

        print(f"SingleAgentDiscreteActionWrapper: {self.n_actions} discrete actions")

    def action(self, action: int) -> np.ndarray:
        """
        Convert discrete action to continuous action array.

        Args:
            action: Discrete action index (0-7)

        Returns:
            Continuous action array [move_x, move_y, shoot, think]
        """
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action {action}. Must be 0-{self.n_actions-1}")
        return self.ACTION_MAP[action].copy()

    def reverse_action(self, continuous_action: np.ndarray) -> int:
        """
        Convert continuous action back to discrete index (for debugging).

        Args:
            continuous_action: [move_x, move_y, shoot, think] array

        Returns:
            Discrete action index (0-7)
        """
        move_x, move_y, shoot = continuous_action[:3]

        # Determine direction
        if abs(move_y) > abs(move_x):
            # Vertical movement
            direction = 0 if move_y < 0 else 1  # North or South
        else:
            # Horizontal movement
            direction = 2 if move_x > 0 else 3  # East or West

        # Add shoot offset
        if shoot > 0.5:
            direction += 4

        return direction

    def get_action_name(self, action: int) -> str:
        """Get human-readable name for an action."""
        if 0 <= action < len(self.ACTION_NAMES):
            return self.ACTION_NAMES[action]
        return f"Unknown({action})"

    def get_movement_actions(self) -> List[int]:
        """Get list of movement-only actions (no shooting)."""
        return [0, 1, 2, 3]

    def get_shoot_actions(self) -> List[int]:
        """Get list of shooting actions."""
        return [4, 5, 6, 7]
