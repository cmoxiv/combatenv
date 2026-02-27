"""
Anti-clump reward wrapper for formation training.

This wrapper penalizes agents that are too close to their nearest ally,
encouraging spacing without penalizing forward movement.

Unlike the Goldilocks wrapper (which uses centroid distance), this wrapper
uses nearest-neighbor distance to avoid the "moving target" problem where
agents that advance faster than the group get penalized.

Usage:
    from combatenv.wrappers import AntiClumpRewardWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = AntiClumpRewardWrapper(env, min_distance=2.0, clump_penalty=0.5)
"""

from typing import Dict, Tuple, Any

import numpy as np
import gymnasium as gym

from combatenv.config import GRID_SIZE


class AntiClumpRewardWrapper(gym.Wrapper):
    """
    Penalize agents that are too close to their nearest ally.

    This encourages agents to spread out while moving toward their objective,
    without penalizing forward movement (unlike centroid-based rewards).

    Attributes:
        min_distance: Minimum acceptable distance to nearest ally (cells)
        clump_penalty: Penalty when closer than min_distance
    """

    def __init__(
        self,
        env,
        min_distance: float = 2.0,
        clump_penalty: float = 0.5
    ):
        """
        Initialize the anti-clump reward wrapper.

        Args:
            env: Base environment
            min_distance: Minimum acceptable distance to nearest ally (cells)
            clump_penalty: Penalty when closer than min_distance
        """
        super().__init__(env)

        self.min_distance = min_distance
        self.clump_penalty = clump_penalty

        print(f"AntiClumpRewardWrapper: Neighbor spacing reward")
        print(f"  Min distance: {min_distance} cells")
        print(f"  Clump penalty (<{min_distance}): -{clump_penalty}")

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        """Step and apply anti-clump reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply anti-clump reward shaping
        shaped_reward = self._shape_reward(obs, reward)

        return obs, shaped_reward, terminated, truncated, info

    def _shape_reward(self, obs: Any, base_reward: Any) -> Any:
        """
        Apply anti-clump reward shaping.

        Args:
            obs: Observation (array or dict)
            base_reward: Original reward

        Returns:
            Shaped reward (float or dict)
        """
        # Single agent case
        if isinstance(obs, np.ndarray):
            clump_penalty = self._compute_clump_penalty(obs)
            return base_reward + clump_penalty

        # Multi-agent case
        elif isinstance(obs, dict) and isinstance(base_reward, dict):
            shaped_rewards = {}
            for agent_idx, agent_obs in obs.items():
                agent_reward = base_reward.get(agent_idx, 0.0)
                clump_penalty = self._compute_clump_penalty(agent_obs)
                shaped_rewards[agent_idx] = agent_reward + clump_penalty
            return shaped_rewards

        return base_reward

    def _compute_clump_penalty(self, obs: np.ndarray) -> float:
        """
        Compute clump penalty for a single agent.

        Args:
            obs: Agent's observation array (92 floats)
                 obs[33] = distance to nearest ally (normalized 0-1)

        Returns:
            Clump penalty (negative if too close, 0 otherwise)
        """
        # obs[33] = nearest ally distance (normalized 0-1 by GRID_SIZE)
        if len(obs) <= 33:
            return 0.0  # No ally distance in observation

        normalized_dist = obs[33]
        ally_dist = normalized_dist * GRID_SIZE  # denormalize to cells

        # Penalize if too close to nearest ally
        if ally_dist < self.min_distance:
            return -self.clump_penalty

        return 0.0
