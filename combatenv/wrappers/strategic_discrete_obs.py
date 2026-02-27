"""
Discrete observation wrapper for strategic-level (grid) Q-learning.

Converts 4x4 strategic grid observations into discrete state indices for Q-table lookup.

State Space Design (65,536 states):
    The 4x4 grid is encoded as follows:
    - For each of 16 cells: 4 states (empty, blue, contested, red)
    - Total: 4^16 = 4,294,967,296 states (too large!)

    Simplified encoding (recommended):
    - Quadrant dominance: 4 quadrants × 3 states (blue/neutral/red) = 81 combinations
    - Force balance: 3 levels (blue advantage / balanced / red advantage)
    - Frontline position: 4 positions (near blue spawn / mid-left / mid-right / near red spawn)
    - Contested count: 3 levels (0 / 1-2 / 3+)

    Total: 81 × 3 × 4 × 3 = 2,916 states

Usage:
    from combatenv.wrappers import StrategicWrapper, StrategicDiscreteObsWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = StrategicWrapper(env)
    env = StrategicDiscreteObsWrapper(env)

    obs, info = env.reset()
    # obs is a single discrete state index
"""

from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym

from combatenv.config import STRATEGIC_GRID_SIZE


class StrategicDiscreteObsWrapper(gym.ObservationWrapper):
    """
    Converts strategic grid observations to discrete state indices.

    This wrapper provides a single discrete state for strategic-level
    Q-learning, encoding the overall battlefield situation.

    Attributes:
        n_states: Total discrete states (2,916)
        n_quadrant_combos: Quadrant dominance combinations (81)
        n_balance: Force balance levels (3)
        n_frontline: Frontline position levels (4)
        n_contested: Contested cell count levels (3)
    """

    # Occupancy thresholds
    BLUE_THRESHOLD = 0.25    # < 0.25 = blue dominant
    RED_THRESHOLD = 0.75     # > 0.75 = red dominant

    def __init__(self, env):
        """
        Initialize the strategic discrete observation wrapper.

        Args:
            env: Environment (should have StrategicWrapper)
        """
        super().__init__(env)

        # State space dimensions
        self.n_quadrant_states = 3   # blue/neutral/red per quadrant
        self.n_quadrant_combos = 81  # 3^4 quadrant combinations
        self.n_balance = 3           # blue advantage / balanced / red advantage
        self.n_frontline = 4         # frontline position
        self.n_contested = 3         # contested cell count levels

        # Total states
        self.n_states = (
            self.n_quadrant_combos *
            self.n_balance *
            self.n_frontline *
            self.n_contested
        )

        print(f"StrategicDiscreteObsWrapper: {self.n_states} discrete states")

    def reset(self, **kwargs) -> Tuple[int, Dict]:
        """
        Reset and return discrete strategic observation.

        Returns:
            observation: Discrete state index
            info: Environment info dictionary
        """
        obs, info = self.env.reset(**kwargs)
        discrete_obs = self.observation(obs)
        return discrete_obs, info

    def step(self, action) -> Tuple[int, float, bool, bool, Dict]:
        """
        Step and return discrete strategic observation.

        Returns:
            observation: Discrete state index
            reward: Strategic-level reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Environment info dictionary
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        discrete_obs = self.observation(obs)
        return discrete_obs, reward, terminated, truncated, info

    def observation(self, obs) -> int:
        """
        Convert strategic observation to discrete state index.

        Args:
            obs: Raw observation from environment

        Returns:
            Discrete state index
        """
        # Get occupancy board from wrapper or compute it
        if hasattr(self.env, 'get_occupancy_board'):
            occupancy = self.env.get_occupancy_board()
        else:
            occupancy = self._compute_occupancy()

        # Encode quadrant dominance
        quadrant_code = self._encode_quadrants(occupancy)

        # Encode force balance
        balance = self._get_force_balance(occupancy)

        # Encode frontline position
        frontline = self._get_frontline_position(occupancy)

        # Encode contested count
        contested = self._get_contested_level(occupancy)

        # Combine into single state index
        state = (
            quadrant_code +
            balance * self.n_quadrant_combos +
            frontline * (self.n_quadrant_combos * self.n_balance) +
            contested * (self.n_quadrant_combos * self.n_balance * self.n_frontline)
        )

        return state

    def _compute_occupancy(self) -> np.ndarray:
        """
        Compute occupancy board from base environment.

        Returns:
            4x4 numpy array with occupancy values
        """
        from combatenv.strategic_grid import get_occupancy_board

        base_env = self.env.unwrapped
        blue_agents = getattr(base_env, 'blue_agents', [])
        red_agents = getattr(base_env, 'red_agents', [])

        return get_occupancy_board(blue_agents, red_agents)

    def _encode_quadrants(self, occupancy: np.ndarray) -> int:
        """
        Encode 4 quadrants as base-3 number.

        Quadrants:
            [0][1]   (top-left, top-right)
            [2][3]   (bottom-left, bottom-right)

        Each quadrant is 2x2 cells from the 4x4 grid.

        Args:
            occupancy: 4x4 occupancy array

        Returns:
            Quadrant code (0 to 80)
        """
        quadrant_states = []

        # Define quadrant boundaries (each quadrant is 2x2 cells)
        quadrants = [
            (0, 0, 2, 2),  # Top-left
            (2, 0, 4, 2),  # Top-right
            (0, 2, 2, 4),  # Bottom-left
            (2, 2, 4, 4),  # Bottom-right
        ]

        for x1, y1, x2, y2 in quadrants:
            # Average occupancy in quadrant
            quad_occ = occupancy[y1:y2, x1:x2].mean()

            if quad_occ < self.BLUE_THRESHOLD:
                quadrant_states.append(0)  # Blue
            elif quad_occ > self.RED_THRESHOLD:
                quadrant_states.append(2)  # Red
            else:
                quadrant_states.append(1)  # Neutral

        # Encode as base-3 number
        code = (
            quadrant_states[0] +
            quadrant_states[1] * 3 +
            quadrant_states[2] * 9 +
            quadrant_states[3] * 27
        )

        return code

    def _get_force_balance(self, occupancy: np.ndarray) -> int:
        """
        Get overall force balance.

        Args:
            occupancy: 4x4 occupancy array

        Returns:
            0 = blue advantage, 1 = balanced, 2 = red advantage
        """
        avg_occ = occupancy.mean()

        if avg_occ < 0.4:
            return 0  # Blue advantage
        elif avg_occ > 0.6:
            return 2  # Red advantage
        else:
            return 1  # Balanced

    def _get_frontline_position(self, occupancy: np.ndarray) -> int:
        """
        Get frontline position (where blue and red meet).

        Args:
            occupancy: 4x4 occupancy array

        Returns:
            0-3 representing frontline position from blue to red spawn
        """
        # Check each column for contested/transition zones
        col_scores = []
        for col in range(STRATEGIC_GRID_SIZE):
            col_occ = occupancy[:, col].mean()
            col_scores.append(col_occ)

        # Find where occupancy transitions from blue to red
        # (lowest column where occupancy > 0.4)
        for i, score in enumerate(col_scores):
            if score > 0.4:
                return min(i, 3)

        return 3  # Default to far right

    def _get_contested_level(self, occupancy: np.ndarray) -> int:
        """
        Get contested cell count level.

        A cell is contested if 0.3 < occupancy < 0.7

        Args:
            occupancy: 4x4 occupancy array

        Returns:
            0 = no contested, 1 = 1-2 contested, 2 = 3+ contested
        """
        contested_count = np.sum((occupancy > 0.3) & (occupancy < 0.7))

        if contested_count == 0:
            return 0
        elif contested_count <= 2:
            return 1
        else:
            return 2

    def decode_state(self, state: int) -> Dict[str, Any]:
        """
        Decode a state index back to its components (for debugging).

        Args:
            state: Discrete state index

        Returns:
            Dict with decoded components
        """
        quadrant_code = state % self.n_quadrant_combos
        balance = (state // self.n_quadrant_combos) % self.n_balance
        frontline = (state // (self.n_quadrant_combos * self.n_balance)) % self.n_frontline
        contested = (state // (self.n_quadrant_combos * self.n_balance * self.n_frontline)) % self.n_contested

        # Decode quadrants
        quadrants = []
        remaining = quadrant_code
        dominance_names = ["blue", "neutral", "red"]
        for _ in range(4):
            quadrants.append(dominance_names[remaining % 3])
            remaining //= 3

        balance_names = ["blue_advantage", "balanced", "red_advantage"]
        frontline_names = ["near_blue", "mid_left", "mid_right", "near_red"]
        contested_names = ["none", "low", "high"]

        return {
            "quadrants": {
                "top_left": quadrants[0],
                "top_right": quadrants[1],
                "bottom_left": quadrants[2],
                "bottom_right": quadrants[3],
            },
            "balance": balance_names[balance],
            "frontline": frontline_names[frontline],
            "contested": contested_names[contested],
        }
