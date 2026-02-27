"""
Discrete observation wrapper for tabular Q-learning.

Converts the 92-dimensional continuous observation space into a discrete
state index suitable for Q-table lookup.

State Space Design (1,920 states):
    - Health: 2 levels (high ≥0.5, low <0.5)
    - Ammo: 2 levels (high ≥0.5, low <0.5)
    - Armor: 2 levels (high ≥0.5, low <0.5)
    - Enemy distance: 2 levels (near <0.15, far ≥0.15)
    - Ally distance: 2 levels (near <0.15, far ≥0.15)
    - Enemy direction: 4 levels (N, S, E, W)
    - Majority terrain: 5 types (empty, obstacle, fire, forest, water)
    - Centroid distance: 3 levels (close <2, optimal 2-6, far >6)

Total: 2 × 2 × 2 × 2 × 2 × 4 × 5 × 3 = 1,920 states

Note: Waypoint direction is NOT encoded - waypoints are an operational-level
concern. Tactical agents maneuver relative to unit centroid.

Usage:
    from combatenv.wrappers import MultiAgentWrapper, DiscreteObservationWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = DiscreteObservationWrapper(env)

    observations, info = env.reset()
    # observations is Dict[int, int] mapping agent_idx -> discrete state index
"""

import math
from typing import Dict, Tuple, Any

import numpy as np
import gymnasium as gym


class DiscreteObservationWrapper(gym.ObservationWrapper):
    """
    Converts continuous 92-dim observations to discrete state indices.

    This wrapper reduces the observation space from 92 continuous dimensions
    to a single discrete state index for Q-table lookup.

    Attributes:
        n_states: Total number of discrete states (6,144)
        n_health: Number of health buckets (2)
        n_ammo: Number of ammo buckets (2)
        n_armor: Number of armor buckets (2)
        n_enemy_dist: Number of enemy distance buckets (2)
        n_ally_dist: Number of ally distance buckets (2)
        n_enemy_dir: Number of enemy direction buckets (4)
        n_terrain: Number of terrain flag combinations (16)
        n_centroid_dist: Number of centroid distance buckets (3)
    """

    # Thresholds for discretization
    RESOURCE_THRESHOLD = 0.5   # high/low threshold for health, ammo, armor
    DISTANCE_THRESHOLD = 0.15  # near/far threshold (normalized, ~10 cells)

    def __init__(self, env):
        """
        Initialize the discrete observation wrapper.

        Args:
            env: Environment (should be MultiAgentWrapper)
        """
        super().__init__(env)

        # Import radius constants
        from combatenv.config import UNIT_MIN_RADIUS, UNIT_MAX_RADIUS
        self.unit_min_radius = UNIT_MIN_RADIUS
        self.unit_max_radius = UNIT_MAX_RADIUS

        # State space dimensions
        self.n_health = 2      # high/low
        self.n_ammo = 2        # high/low
        self.n_armor = 2       # high/low
        self.n_enemy_dist = 2  # near/far
        self.n_ally_dist = 2   # near/far
        self.n_enemy_dir = 4   # N/S/E/W
        self.n_terrain = 5     # majority terrain: EMPTY/OBSTACLE/FIRE/FOREST/WATER
        self.n_centroid_dist = 3  # close/optimal/far

        # Total state space size
        self.n_states = (
            self.n_health *
            self.n_ammo *
            self.n_armor *
            self.n_enemy_dist *
            self.n_ally_dist *
            self.n_enemy_dir *
            self.n_terrain *
            self.n_centroid_dist
        )

        print(f"DiscreteObservationWrapper: {self.n_states} discrete states")

    def reset(self, **kwargs) -> Tuple[Dict[int, int], Dict]:
        """
        Reset and return discrete observations.

        Returns:
            observations: Dict mapping agent_idx -> discrete state index
            info: Environment info dictionary
        """
        obs_dict, info = self.env.reset(**kwargs)
        discrete_obs = self.observation(obs_dict)
        return discrete_obs, info

    def step(self, action) -> Tuple[Dict[int, int], Dict[int, float], bool, bool, Dict]:
        """
        Step and return discrete observations.

        Returns:
            observations: Dict mapping agent_idx -> discrete state index
            rewards: Dict mapping agent_idx -> reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Environment info dictionary
        """
        obs_dict, rewards, terminated, truncated, info = self.env.step(action)
        discrete_obs = self.observation(obs_dict)
        return discrete_obs, rewards, terminated, truncated, info

    def observation(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Convert continuous observations to discrete state indices.

        Args:
            obs_dict: Dict mapping agent_idx -> 88-float observation array

        Returns:
            Dict mapping agent_idx -> discrete state index (int)
        """
        discrete_obs = {}
        for agent_idx, obs in obs_dict.items():
            discrete_obs[agent_idx] = self._discretize(obs)
        return discrete_obs

    def _discretize(self, obs: np.ndarray) -> int:
        """
        Convert a single 92-dim observation to discrete state index.

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
            - obs[89]: waypoint_rel_x (-1 to 1, west to east)
            - obs[90]: waypoint_rel_y (-1 to 1, north to south)
            - obs[91]: centroid distance (normalized 0-1)

        Args:
            obs: 92-float numpy array

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

        # Majority terrain type from FOV (indices 50-87)
        majority_terrain = self._get_majority_terrain(obs[50:88])

        # Centroid distance (3 levels: close/optimal/far)
        # obs[91] is normalized by GRID_SIZE, so denormalize first
        centroid_dist = self._get_centroid_distance_bucket(obs)

        # Encode state as single integer
        # Using mixed-radix encoding
        state = (
            health +
            ammo * 2 +
            armor * 4 +
            enemy_dist * 8 +
            ally_dist * 16 +
            enemy_dir * 32 +
            majority_terrain * 128 +
            centroid_dist * 640
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

    def _get_centroid_distance_bucket(self, obs: np.ndarray) -> int:
        """
        Get centroid distance bucket from observation.

        Buckets based on unit radius constraints:
            0 = close (< UNIT_MIN_RADIUS from centroid)
            1 = optimal (UNIT_MIN_RADIUS to UNIT_MAX_RADIUS - in the sweet spot)
            2 = far (> UNIT_MAX_RADIUS - outside allowed radius)

        Args:
            obs: 92-float observation array

        Returns:
            Bucket index: 0=close, 1=optimal, 2=far
        """
        from combatenv.config import GRID_SIZE

        # obs[91] = centroid distance (normalized 0-1 by GRID_SIZE)
        normalized_dist = obs[91] if len(obs) > 91 else 0.5
        dist = normalized_dist * GRID_SIZE  # denormalize to cells

        if dist < self.unit_min_radius:
            return 0  # too close to centroid
        elif dist <= self.unit_max_radius:
            return 1  # optimal (within allowed radius)
        else:
            return 2  # too far (outside leash)

    def _get_majority_terrain(self, terrain_obs: np.ndarray) -> int:
        """
        Get the majority terrain type in FOV.

        Terrain types (normalized values):
            - EMPTY = 0.0
            - OBSTACLE = 0.25
            - FIRE = 0.5
            - FOREST = 0.75
            - WATER = 1.0

        Args:
            terrain_obs: Array of terrain values from observation (indices 50-87)

        Returns:
            Majority terrain index: 0=EMPTY, 1=OBSTACLE, 2=FIRE, 3=FOREST, 4=WATER
        """
        # Count each terrain type
        counts = [0, 0, 0, 0, 0]  # EMPTY, OBSTACLE, FIRE, FOREST, WATER

        for val in terrain_obs:
            if val < 0.01:  # EMPTY
                counts[0] += 1
            elif 0.2 <= val <= 0.3:  # OBSTACLE (0.25)
                counts[1] += 1
            elif 0.45 <= val <= 0.55:  # FIRE (0.5)
                counts[2] += 1
            elif 0.7 <= val <= 0.8:  # FOREST (0.75)
                counts[3] += 1
            elif val >= 0.95:  # WATER (1.0)
                counts[4] += 1

        # Return index of maximum count
        return counts.index(max(counts))

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
        majority_terrain = (state // 128) % 5
        centroid_dist = (state // 640) % 3

        direction_names = ["North", "South", "East", "West"]
        centroid_dist_names = ["close", "optimal", "far"]
        terrain_names = ["empty", "obstacle", "fire", "forest", "water"]

        return {
            "health": "high" if health else "low",
            "ammo": "high" if ammo else "low",
            "armor": "high" if armor else "low",
            "enemy_distance": "far" if enemy_dist else "near",
            "ally_distance": "far" if ally_dist else "near",
            "enemy_direction": direction_names[enemy_dir],
            "majority_terrain": terrain_names[majority_terrain],
            "centroid_distance": centroid_dist_names[centroid_dist],
        }
