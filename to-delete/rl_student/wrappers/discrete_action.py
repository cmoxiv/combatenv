"""
Discrete action wrapper for tabular Q-learning.

Converts discrete action indices to continuous action arrays for the environment.

Action Space Design (8 actions):
    - 4 movement directions: North, South, East, West
    - 2 shoot options: shoot or no-shoot
    - Total: 4 × 2 = 8 discrete actions

Action Mapping:
    0: North (no shoot)     -> [0.0, -1.0, 0.0]
    1: South (no shoot)     -> [0.0,  1.0, 0.0]
    2: East (no shoot)      -> [1.0,  0.0, 0.0]
    3: West (no shoot)      -> [-1.0, 0.0, 0.0]
    4: North + Shoot        -> [0.0, -1.0, 1.0]
    5: South + Shoot        -> [0.0,  1.0, 1.0]
    6: East + Shoot         -> [1.0,  0.0, 1.0]
    7: West + Shoot         -> [-1.0, 0.0, 1.0]

Usage:
    from rl_student.wrappers import MultiAgentWrapper, DiscreteObservationWrapper, DiscreteActionWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)

    observations, info = env.reset()
    # observations is Dict[int, int] mapping agent_idx -> discrete state

    actions = {agent_idx: 0 for agent_idx in observations}  # All move North
    next_obs, rewards, terminated, truncated, info = env.step(actions)
"""

from typing import Dict, Tuple, List

import numpy as np
import gymnasium as gym


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts discrete action indices to continuous action arrays.

    This wrapper maps 8 discrete actions to the environment's continuous
    action space [move_x, move_y, shoot].

    Attributes:
        n_actions: Number of discrete actions (8)
        ACTION_MAP: Dict mapping action index to continuous action array
        ACTION_NAMES: List of human-readable action names
    """

    # Action mapping: discrete index -> (move_x, move_y, shoot)
    ACTION_MAP = {
        0: np.array([0.0, -1.0, 0.0], dtype=np.float32),   # North
        1: np.array([0.0, 1.0, 0.0], dtype=np.float32),    # South
        2: np.array([1.0, 0.0, 0.0], dtype=np.float32),    # East
        3: np.array([-1.0, 0.0, 0.0], dtype=np.float32),   # West
        4: np.array([0.0, -1.0, 1.0], dtype=np.float32),   # North + Shoot
        5: np.array([0.0, 1.0, 1.0], dtype=np.float32),    # South + Shoot
        6: np.array([1.0, 0.0, 1.0], dtype=np.float32),    # East + Shoot
        7: np.array([-1.0, 0.0, 1.0], dtype=np.float32),   # West + Shoot
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
            env: Environment (should be DiscreteObservationWrapper)
        """
        super().__init__(env)
        self.n_actions = len(self.ACTION_MAP)

        # Expose n_states from the observation wrapper
        self.n_states = env.n_states

        print(f"DiscreteActionWrapper: {self.n_actions} discrete actions")

    def action(self, action_dict: Dict[int, int]) -> Dict[int, np.ndarray]:
        """
        Convert discrete actions to continuous action arrays.

        Args:
            action_dict: Dict mapping agent_idx -> discrete action index (0-7)

        Returns:
            Dict mapping agent_idx -> continuous action array [move_x, move_y, shoot]
        """
        continuous_actions = {}
        for agent_idx, discrete_action in action_dict.items():
            if discrete_action < 0 or discrete_action >= self.n_actions:
                raise ValueError(f"Invalid action {discrete_action}. Must be 0-{self.n_actions-1}")
            continuous_actions[agent_idx] = self.ACTION_MAP[discrete_action].copy()
        return continuous_actions

    def reverse_action(self, continuous_action: np.ndarray) -> int:
        """
        Convert continuous action back to discrete index (for debugging).

        Args:
            continuous_action: [move_x, move_y, shoot] array

        Returns:
            Discrete action index (0-7)
        """
        move_x, move_y, shoot = continuous_action

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
        """
        Get human-readable name for an action.

        Args:
            action: Discrete action index (0-7)

        Returns:
            Action name string
        """
        if 0 <= action < len(self.ACTION_NAMES):
            return self.ACTION_NAMES[action]
        return f"Unknown({action})"

    def get_movement_actions(self) -> List[int]:
        """
        Get list of movement-only actions (no shooting).

        Returns:
            List of action indices [0, 1, 2, 3]
        """
        return [0, 1, 2, 3]

    def get_shoot_actions(self) -> List[int]:
        """
        Get list of shooting actions.

        Returns:
            List of action indices [4, 5, 6, 7]
        """
        return [4, 5, 6, 7]

    def get_direction_for_action(self, action: int) -> str:
        """
        Get the direction component of an action.

        Args:
            action: Discrete action index (0-7)

        Returns:
            Direction string ("North", "South", "East", "West")
        """
        directions = ["North", "South", "East", "West"]
        return directions[action % 4]

    def does_action_shoot(self, action: int) -> bool:
        """
        Check if an action includes shooting.

        Args:
            action: Discrete action index (0-7)

        Returns:
            True if action includes shooting
        """
        return action >= 4
