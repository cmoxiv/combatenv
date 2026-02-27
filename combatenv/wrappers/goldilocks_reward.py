"""
Goldilocks zone reward wrapper for formation training.

This wrapper shapes rewards based on distance from unit centroid,
encouraging agents to maintain optimal spacing (not too close, not too far).

Reward Zones:
    - Too close (< optimal_min): penalty
    - Goldilocks zone (optimal_min to optimal_max): bonus
    - Too far (> optimal_max): penalty

Usage:
    from combatenv.wrappers import GoldilocksRewardWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = GoldilocksRewardWrapper(env, optimal_min=3.0, optimal_max=6.0)
"""

from typing import Dict, Tuple, Any

import numpy as np
import gymnasium as gym

from combatenv.config import GRID_SIZE


class GoldilocksRewardWrapper(gym.Wrapper):
    """
    Reward shaping based on distance to unit centroid.

    Encourages agents to maintain optimal spacing from their unit's centroid.
    Agents in the "Goldilocks zone" (optimal distance range) receive a bonus,
    while agents too close or too far receive a penalty.

    Attributes:
        optimal_min: Minimum distance for optimal zone (cells)
        optimal_max: Maximum distance for optimal zone (cells)
        optimal_bonus: Reward for being in optimal zone
        close_penalty: Penalty for being too close
        far_penalty: Penalty for being too far
    """

    def __init__(
        self,
        env,
        optimal_min: float = 3.0,
        optimal_max: float = 6.0,
        optimal_bonus: float = 1.0,
        close_penalty: float = 0.5,
        far_penalty: float = 0.5
    ):
        """
        Initialize the Goldilocks reward wrapper.

        Args:
            env: Base environment
            optimal_min: Minimum distance for Goldilocks zone (cells)
            optimal_max: Maximum distance for Goldilocks zone (cells)
            optimal_bonus: Reward when in optimal zone
            close_penalty: Penalty when too close to centroid
            far_penalty: Penalty when too far from centroid
        """
        super().__init__(env)

        self.optimal_min = optimal_min
        self.optimal_max = optimal_max
        self.optimal_bonus = optimal_bonus
        self.close_penalty = close_penalty
        self.far_penalty = far_penalty

        print(f"GoldilocksRewardWrapper: Formation reward shaping")
        print(f"  Optimal zone: {optimal_min}-{optimal_max} cells")
        print(f"  Optimal bonus: +{optimal_bonus}")
        print(f"  Close penalty (<{optimal_min}): -{close_penalty}")
        print(f"  Far penalty (>{optimal_max}): -{far_penalty}")

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        """Step and apply Goldilocks zone reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply formation reward shaping
        shaped_reward = self._shape_reward(obs, reward)

        return obs, shaped_reward, terminated, truncated, info

    def _shape_reward(self, obs: Any, base_reward: Any) -> Any:
        """
        Apply Goldilocks zone reward shaping.

        Args:
            obs: Observation (array or dict)
            base_reward: Original reward

        Returns:
            Shaped reward (float or dict)
        """
        # Single agent case
        if isinstance(obs, np.ndarray):
            formation_reward = self._compute_formation_reward(obs)
            return base_reward + formation_reward

        # Multi-agent case
        elif isinstance(obs, dict) and isinstance(base_reward, dict):
            shaped_rewards = {}
            for agent_idx, agent_obs in obs.items():
                agent_reward = base_reward.get(agent_idx, 0.0)
                formation_reward = self._compute_formation_reward(agent_obs)
                shaped_rewards[agent_idx] = agent_reward + formation_reward
            return shaped_rewards

        return base_reward

    def _compute_formation_reward(self, obs: np.ndarray) -> float:
        """
        Compute formation reward for a single agent.

        Args:
            obs: Agent's observation array (92 floats)

        Returns:
            Formation reward (positive for optimal, negative otherwise)
        """
        # obs[91] = centroid distance (normalized 0-1 by GRID_SIZE)
        if len(obs) <= 91:
            return 0.0  # No centroid distance in observation

        normalized_dist = obs[91]
        dist = normalized_dist * GRID_SIZE  # denormalize to cells

        # Goldilocks zone check
        if self.optimal_min <= dist <= self.optimal_max:
            return self.optimal_bonus
        elif dist < self.optimal_min:
            return -self.close_penalty
        else:
            return -self.far_penalty
