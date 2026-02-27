"""
Cohesion reward wrapper for unit-based coordination.

This wrapper modifies survival rewards based on how close agents are to their
unit centroid. Agents that stay grouped receive higher survival rewards,
while isolated agents receive reduced rewards.

Reward Multiplier:
    - At centroid: 1.0 + COHESION_REWARD_BONUS (e.g., 1.5x)
    - Within cohesion radius: Linear interpolation (1.0 to 1.5x)
    - Outside cohesion radius: 1.0 - COHESION_ISOLATION_PENALTY (e.g., 0.7x)

Usage:
    from rl_student.wrappers import MultiAgentWrapper, CohesionRewardWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = UnitWrapper(env)  # Must be applied first to track units
    env = CohesionRewardWrapper(env)
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)
"""

from typing import Dict, Tuple, Any, List, Optional
import math

import numpy as np
import gymnasium as gym

from combatenv.config import (
    COHESION_REWARD_BONUS,
    COHESION_ISOLATION_PENALTY,
    UNIT_COHESION_RADIUS,
)


class CohesionRewardWrapper(gym.Wrapper):
    """
    Reward shaping wrapper that multiplies survival rewards by cohesion score.

    Encourages agents to stay near their unit centroid through reward shaping.
    The survival reward component (default 0.01 per step) is multiplied by a
    cohesion factor that rewards grouping and penalizes isolation.

    Attributes:
        cohesion_bonus: Bonus multiplier at centroid (default: 0.5 -> 1.5x)
        isolation_penalty: Penalty when isolated (default: 0.3 -> 0.7x)
        cohesion_radius: Radius for grouping (default: 4.0 cells)
        base_survival_reward: Expected survival reward per step (default: 0.01)
    """

    def __init__(
        self,
        env,
        cohesion_bonus: float = COHESION_REWARD_BONUS,
        isolation_penalty: float = COHESION_ISOLATION_PENALTY,
        cohesion_radius: float = UNIT_COHESION_RADIUS,
        base_survival_reward: float = 0.01
    ):
        """
        Initialize the cohesion reward wrapper.

        Args:
            env: Environment (should have units attribute from UnitWrapper)
            cohesion_bonus: Bonus multiplier at centroid
            isolation_penalty: Penalty multiplier when isolated
            cohesion_radius: Distance threshold for being "grouped"
            base_survival_reward: The survival reward to apply multiplier to
        """
        super().__init__(env)

        self.cohesion_bonus = cohesion_bonus
        self.isolation_penalty = isolation_penalty
        self.cohesion_radius = cohesion_radius
        self.base_survival_reward = base_survival_reward

        print(f"CohesionRewardWrapper: Shaping rewards for unit cohesion")
        print(f"  At centroid: {1.0 + cohesion_bonus:.1f}x survival reward")
        print(f"  Isolated: {1.0 - isolation_penalty:.1f}x survival reward")
        print(f"  Cohesion radius: {cohesion_radius} cells")

    def step(self, actions) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict]:
        """Step and apply cohesion-based reward shaping."""
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)

        # Apply cohesion multiplier to rewards
        shaped_rewards = {}
        for agent_idx, reward in rewards.items():
            multiplier = self._get_cohesion_multiplier(agent_idx)
            shaped_reward = self._apply_cohesion_shaping(reward, multiplier)
            shaped_rewards[agent_idx] = shaped_reward

        return obs_dict, shaped_rewards, terminated, truncated, info

    def _get_cohesion_multiplier(self, agent_idx: int) -> float:
        """
        Calculate cohesion multiplier for an agent.

        Args:
            agent_idx: Agent index

        Returns:
            Multiplier for survival reward (0.7 to 1.5 typically)
        """
        # Try to get unit info from wrapped environment
        if not hasattr(self.env, 'blue_units') or not hasattr(self.env, 'red_units'):
            return 1.0  # No unit info available, use neutral multiplier

        # Determine team and get units
        if agent_idx < 100:
            units = self.env.blue_units
            agent_list = getattr(self.env, 'blue_agents', None)
        else:
            units = self.env.red_units
            agent_list = getattr(self.env, 'red_agents', None)

        if not units or not agent_list:
            return 1.0

        # Find the agent
        local_idx = agent_idx if agent_idx < 100 else agent_idx - 100
        if local_idx >= len(agent_list):
            return 1.0

        agent = agent_list[local_idx]

        # Dead agents don't get cohesion bonus
        if not agent.is_alive:
            return 1.0

        # Find the unit this agent belongs to
        unit = None
        for u in units:
            if agent.unit_id == u.id:
                unit = u
                break

        if unit is None:
            return 1.0 - self.isolation_penalty  # No unit = isolated

        # Calculate distance to centroid
        centroid = unit.centroid
        dist = math.sqrt(
            (agent.position[0] - centroid[0])**2 +
            (agent.position[1] - centroid[1])**2
        )

        # Calculate multiplier based on distance
        if dist < 1.0:
            # At centroid - full bonus
            return 1.0 + self.cohesion_bonus
        elif dist < self.cohesion_radius:
            # Within radius - linear interpolation
            t = (dist - 1.0) / (self.cohesion_radius - 1.0)
            return 1.0 + self.cohesion_bonus * (1.0 - t)
        else:
            # Outside radius - penalty
            return 1.0 - self.isolation_penalty

    def _apply_cohesion_shaping(self, reward: float, multiplier: float) -> float:
        """
        Apply cohesion multiplier to survival component of reward.

        Only the survival reward component is shaped. Kill rewards and
        death penalties pass through unchanged.

        Args:
            reward: Original reward
            multiplier: Cohesion multiplier (0.7 to 1.5)

        Returns:
            Shaped reward
        """
        # If reward is exactly the base survival reward, apply full multiplier
        if abs(reward - self.base_survival_reward) < 0.001:
            return reward * multiplier

        # For other rewards (kills, death), separate survival component
        # Assume any positive small reward includes survival
        if reward > 0 and reward <= self.base_survival_reward:
            return reward * multiplier

        # For larger rewards (kills included), apply multiplier only to survival portion
        if reward > self.base_survival_reward:
            non_survival = reward - self.base_survival_reward
            survival_shaped = self.base_survival_reward * multiplier
            return non_survival + survival_shaped

        # Negative rewards (death) pass through unchanged
        return reward


def calculate_cohesion_multiplier(
    agent_position: Tuple[float, float],
    centroid: Tuple[float, float],
    cohesion_radius: float = UNIT_COHESION_RADIUS,
    bonus: float = COHESION_REWARD_BONUS,
    penalty: float = COHESION_ISOLATION_PENALTY
) -> float:
    """
    Calculate cohesion reward multiplier for an agent position.

    Standalone function for use outside the wrapper.

    Args:
        agent_position: Agent (x, y) position
        centroid: Unit centroid (x, y)
        cohesion_radius: Radius threshold for being grouped
        bonus: Multiplier bonus at centroid
        penalty: Multiplier penalty when isolated

    Returns:
        Multiplier value (penalty to 1+bonus, e.g., 0.7 to 1.5)
    """
    dist = math.sqrt(
        (agent_position[0] - centroid[0])**2 +
        (agent_position[1] - centroid[1])**2
    )

    if dist < 1.0:
        return 1.0 + bonus
    elif dist < cohesion_radius:
        t = (dist - 1.0) / (cohesion_radius - 1.0)
        return 1.0 + bonus * (1.0 - t)
    else:
        return 1.0 - penalty
