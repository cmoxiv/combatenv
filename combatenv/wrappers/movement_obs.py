"""
Movement-focused discrete observation wrapper.

Converts continuous observations to a simplified discrete state space
optimized for movement/navigation learning.

State Space (64 states):
    - Chebyshev distance to waypoint: 8 levels (0-7)
    - Direction to waypoint: 4 levels (N, S, E, W)
    - Blocked ahead: 2 levels (yes/no)

Total: 8 × 4 × 2 = 64 states

This smaller state space enables faster Q-table convergence for movement
training compared to the full 2,048 state combat observation space.

Usage:
    from movement_training_env import MovementTrainingEnv
    from combatenv.wrappers import MovementObservationWrapper

    env = MovementTrainingEnv(render_mode=None)
    env = MovementObservationWrapper(env)

    obs, info = env.reset()  # obs is an int in [0, 63]
"""

import math
from typing import Tuple, Dict, Any

import numpy as np
import gymnasium as gym


class MovementObservationWrapper(gym.ObservationWrapper):
    """
    Simplified discrete observation for movement training.

    Converts the 89-dim continuous observation into a 64-state
    discrete space focused on navigation to waypoint.

    State encoding:
        state = chebyshev + direction * 8 + blocked * 32

    Where:
        - chebyshev: 0-7 (distance to waypoint in operational cells)
        - direction: 0-3 (N=0, S=1, E=2, W=3)
        - blocked: 0-1 (building in movement direction)

    Attributes:
        n_states: Total discrete states (64)
        n_chebyshev: Chebyshev distance levels (8)
        n_direction: Direction levels (4)
        n_blocked: Blocked flag levels (2)
    """

    def __init__(self, env):
        """
        Initialize the movement observation wrapper.

        Args:
            env: Environment (should be MovementTrainingEnv)
        """
        super().__init__(env)

        # State space dimensions
        self.n_chebyshev = 8  # Distance levels 0-7
        self.n_direction = 4  # N, S, E, W
        self.n_blocked = 2    # Yes/No

        self.n_states = self.n_chebyshev * self.n_direction * self.n_blocked
        # 8 * 4 * 2 = 64

        # Update observation space
        self.observation_space = gym.spaces.Discrete(self.n_states)

        print(f"MovementObservationWrapper: {self.n_states} discrete states")

    def reset(self, **kwargs) -> Tuple[int, Dict]:
        """
        Reset and return discrete observation.

        Returns:
            Tuple of (discrete_state, info)
        """
        obs, info = self.env.reset(**kwargs)
        discrete_obs = self.observation(obs)
        return discrete_obs, info

    def step(self, action) -> Tuple[int, float, bool, bool, Dict]:
        """
        Step and return discrete observation.

        Args:
            action: Action (can be discrete int or continuous array)

        Returns:
            Tuple of (discrete_state, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        discrete_obs = self.observation(obs)
        return discrete_obs, reward, terminated, truncated, info

    def observation(self, obs: np.ndarray) -> int:
        """
        Convert continuous observation to discrete state index.

        Args:
            obs: 89-dim continuous observation array

        Returns:
            Discrete state index (0-63)
        """
        return self._discretize(obs)

    def _discretize(self, obs: np.ndarray) -> int:
        """
        Convert observation to discrete state.

        Args:
            obs: 89-dim observation array

        Returns:
            State index in [0, 63]
        """
        # Handle dead agent (all zeros)
        if np.all(obs == 0):
            return 0

        # 1. Chebyshev distance (obs[88] is normalized 0-1, multiply by 7 for 0-7)
        chebyshev_norm = obs[88] if len(obs) > 88 else 0.5
        chebyshev = min(7, int(chebyshev_norm * 8))

        # 2. Direction to waypoint
        # Use agent position and waypoint from environment
        direction = self._get_waypoint_direction(obs)

        # 3. Blocked ahead (check terrain in FOV)
        blocked = self._is_blocked_ahead(obs)

        # Encode state
        state = chebyshev + direction * 8 + blocked * 32

        return state

    def _get_waypoint_direction(self, obs: np.ndarray) -> int:
        """
        Determine direction to waypoint.

        Uses the relative position encoded in observation or
        queries the underlying environment.

        Args:
            obs: Observation array

        Returns:
            Direction: 0=North, 1=South, 2=East, 3=West
        """
        # Get agent position from observation (obs[0:2] are x,y normalized)
        agent_x = obs[0] * 64  # Denormalize to grid coords
        agent_y = obs[1] * 64

        # Try to get waypoint from environment
        waypoint = getattr(self.env, '_target_waypoint', None)
        if waypoint is None:
            # Fallback: use controlled agent's unit waypoint
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'controlled_agent') and env.controlled_agent is not None:
                from combatenv import get_unit_for_agent
                unit = get_unit_for_agent(env.controlled_agent, getattr(env, 'blue_units', []))
                if unit is not None and unit.waypoint is not None:
                    waypoint = unit.waypoint

        if waypoint is None:
            return 0  # Default to North

        wp_x, wp_y = waypoint
        dx = wp_x - agent_x
        dy = wp_y - agent_y

        # Determine primary direction
        if abs(dy) >= abs(dx):
            # Vertical dominant
            return 0 if dy < 0 else 1  # North=0, South=1
        else:
            # Horizontal dominant
            return 2 if dx > 0 else 3  # East=2, West=3

    def _is_blocked_ahead(self, obs: np.ndarray) -> int:
        """
        Check if there's a building blocking movement ahead.

        Uses terrain observation from FOV (obs[50:88]).

        Args:
            obs: Observation array

        Returns:
            1 if blocked, 0 if clear
        """
        # Terrain is in obs[50:88] (38 cells)
        # Check first few cells which are typically directly ahead
        if len(obs) < 54:
            return 0

        terrain_ahead = obs[50:54]  # First 4 terrain cells

        # Building terrain is encoded as ~0.25 normalized
        for val in terrain_ahead:
            if 0.2 <= val <= 0.3:  # Building
                return 1

        return 0

    def decode_state(self, state: int) -> Dict[str, Any]:
        """
        Decode a state index back to its components (for debugging).

        Args:
            state: Discrete state index (0-63)

        Returns:
            Dict with decoded components
        """
        chebyshev = state % 8
        direction = (state // 8) % 4
        blocked = (state // 32) % 2

        direction_names = ["North", "South", "East", "West"]

        return {
            "chebyshev_distance": chebyshev,
            "waypoint_direction": direction_names[direction],
            "blocked_ahead": "yes" if blocked else "no",
        }
