"""
Single-agent discrete observation wrapper for tabular Q-learning.

Converts the 89-dimensional continuous observation space into a discrete
state index suitable for Q-table lookup. Works with single-agent environments.

State Space Design (2,048 states):
    - Health: 2 levels (high ≥0.5, low <0.5)
    - Ammo: 2 levels (high ≥0.5, low <0.5)
    - Armor: 2 levels (high ≥0.5, low <0.5)
    - Enemy distance: 2 levels (near <0.15, far ≥0.15)
    - Ally distance: 2 levels (near <0.15, far ≥0.15)
    - Enemy direction: 4 levels (N, S, E, W)
    - Terrain flags: 16 combinations (obstacle, fire, forest, water present)

Total: 2 × 2 × 2 × 2 × 2 × 4 × 16 = 2,048 states

Usage:
    from combatenv.wrappers import SingleAgentDiscreteObsWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = SingleAgentDiscreteObsWrapper(env)

    obs, info = env.reset()
    # obs is an int in [0, 2047]
"""

from typing import Tuple, Dict, Any

import numpy as np
import gymnasium as gym


class SingleAgentDiscreteObsWrapper(gym.ObservationWrapper):
    """
    Converts continuous 89-dim observations to discrete state indices.

    This wrapper reduces the observation space from 89 continuous dimensions
    to a single discrete state index for Q-table lookup.

    Attributes:
        n_states: Total number of discrete states (2,048)
    """

    # Thresholds for discretization
    RESOURCE_THRESHOLD = 0.5   # high/low threshold for health, ammo, armor
    DISTANCE_THRESHOLD = 0.15  # near/far threshold (normalized, ~10 cells)

    def __init__(self, env):
        """
        Initialize the discrete observation wrapper.

        Args:
            env: Base environment (single-agent)
        """
        super().__init__(env)

        # State space dimensions
        self.n_health = 2      # high/low
        self.n_ammo = 2        # high/low
        self.n_armor = 2       # high/low
        self.n_enemy_dist = 2  # near/far
        self.n_ally_dist = 2   # near/far
        self.n_enemy_dir = 4   # N/S/E/W
        self.n_terrain = 16    # 4 binary flags = 2^4

        # Total state space size
        self.n_states = (
            self.n_health *
            self.n_ammo *
            self.n_armor *
            self.n_enemy_dist *
            self.n_ally_dist *
            self.n_enemy_dir *
            self.n_terrain
        )  # 2,048

        # Update observation space
        self.observation_space = gym.spaces.Discrete(self.n_states)

        print(f"SingleAgentDiscreteObsWrapper: {self.n_states} discrete states")

    def reset(self, **kwargs) -> Tuple[int, Dict]:
        """Reset and return discrete observation."""
        obs, info = self.env.reset(**kwargs)
        discrete_obs = self.observation(obs)
        return discrete_obs, info

    def step(self, action) -> Tuple[int, float, bool, bool, Dict]:
        """Step and return discrete observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        discrete_obs = self.observation(obs)
        return discrete_obs, reward, terminated, truncated, info

    def observation(self, obs: np.ndarray) -> int:
        """
        Convert continuous observation to discrete state index.

        Args:
            obs: 89-float observation array

        Returns:
            Discrete state index (0-2047)
        """
        return self._discretize(obs)

    def _discretize(self, obs: np.ndarray) -> int:
        """
        Convert a single 89-dim observation to discrete state index.

        Observation structure:
            - obs[0-1]: position (x, y)
            - obs[2]: orientation
            - obs[3]: health (normalized 0-1)
            - obs[4]: stamina
            - obs[5]: armor (normalized 0-1)
            - obs[6]: ammo_reserve
            - obs[7]: magazine_ammo (normalized 0-1)
            - obs[8]: can_shoot
            - obs[9]: is_reloading
            - obs[10-13]: nearest enemy (rel_x, rel_y, health, distance)
            - obs[30-33]: nearest ally (rel_x, rel_y, health, distance)
            - obs[50-87]: terrain in FOV
            - obs[88]: Chebyshev distance to waypoint

        Args:
            obs: 89-float numpy array

        Returns:
            Discrete state index (0 to n_states-1)
        """
        # Handle dead agents (zero observation)
        if np.all(obs == 0):
            return 0

        # Extract features and discretize
        health = 1 if obs[3] >= self.RESOURCE_THRESHOLD else 0      # high/low
        ammo = 1 if obs[7] >= self.RESOURCE_THRESHOLD else 0        # high/low (magazine)
        armor = 1 if obs[5] >= self.RESOURCE_THRESHOLD else 0       # high/low

        # Enemy distance (obs[13] is distance to nearest enemy, normalized)
        enemy_dist = 1 if obs[13] >= self.DISTANCE_THRESHOLD else 0  # far/near

        # Ally distance (obs[33] is distance to nearest ally, normalized)
        ally_dist = 1 if obs[33] >= self.DISTANCE_THRESHOLD else 0   # far/near

        # Enemy direction (4 compass quadrants)
        # obs[10] and obs[11] are relative x, y (centered at 0.5)
        enemy_dir = self._get_direction(obs[10], obs[11])

        # Terrain flags from FOV (indices 50-87)
        terrain_flags = self._encode_terrain_flags(obs[50:88])

        # Encode state as single integer using mixed-radix encoding
        state = (
            health +
            ammo * 2 +
            armor * 4 +
            enemy_dist * 8 +
            ally_dist * 16 +
            enemy_dir * 32 +
            terrain_flags * 128
        )

        return state

    def _get_direction(self, rel_x: float, rel_y: float) -> int:
        """
        Convert relative position to compass direction.

        Args:
            rel_x: Relative x position (centered at 0.5)
            rel_y: Relative y position (centered at 0.5)

        Returns:
            Direction index: 0=North, 1=South, 2=East, 3=West
        """
        # Center at 0
        dx = rel_x - 0.5
        dy = rel_y - 0.5

        # Handle case where no enemy visible (default to North)
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return 0

        # Determine primary direction
        if abs(dy) > abs(dx):
            # Vertical movement dominates
            return 0 if dy < 0 else 1  # North=0, South=1
        else:
            # Horizontal movement dominates
            return 2 if dx > 0 else 3  # East=2, West=3

    def _encode_terrain_flags(self, terrain_obs: np.ndarray) -> int:
        """
        Encode terrain observations as 4-bit flags.

        Terrain types (normalized values):
            - EMPTY = 0.0
            - OBSTACLE = 0.25
            - FIRE = 0.5
            - FOREST = 0.75
            - WATER = 1.0

        Flags:
            - bit 0: obstacle present in FOV
            - bit 1: fire present in FOV
            - bit 2: forest present in FOV
            - bit 3: water present in FOV

        Args:
            terrain_obs: Array of terrain values from observation (indices 50-87)

        Returns:
            4-bit integer (0-15) representing terrain flags
        """
        flags = 0

        for val in terrain_obs:
            if val < 0.01:  # EMPTY
                continue
            elif 0.2 <= val <= 0.3:  # OBSTACLE (0.25)
                flags |= 1  # bit 0
            elif 0.45 <= val <= 0.55:  # FIRE (0.5)
                flags |= 2  # bit 1
            elif 0.7 <= val <= 0.8:  # FOREST (0.75)
                flags |= 4  # bit 2
            elif val >= 0.95:  # WATER (1.0)
                flags |= 8  # bit 3

        return flags

    def decode_state(self, state: int) -> Dict[str, Any]:
        """
        Decode a state index back to its components (for debugging).

        Args:
            state: Discrete state index

        Returns:
            Dict with decoded components
        """
        health = state % 2
        ammo = (state // 2) % 2
        armor = (state // 4) % 2
        enemy_dist = (state // 8) % 2
        ally_dist = (state // 16) % 2
        enemy_dir = (state // 32) % 4
        terrain_flags = (state // 128) % 16

        direction_names = ["North", "South", "East", "West"]
        terrain_names = []
        if terrain_flags & 1:
            terrain_names.append("obstacle")
        if terrain_flags & 2:
            terrain_names.append("fire")
        if terrain_flags & 4:
            terrain_names.append("forest")
        if terrain_flags & 8:
            terrain_names.append("water")

        return {
            "health": "high" if health else "low",
            "ammo": "high" if ammo else "low",
            "armor": "high" if armor else "low",
            "enemy_distance": "far" if enemy_dist else "near",
            "ally_distance": "far" if ally_dist else "near",
            "enemy_direction": direction_names[enemy_dir],
            "terrain_in_fov": terrain_names if terrain_names else ["none"],
        }
