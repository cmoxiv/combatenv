"""
FOV coverage reward wrapper for tactical training.

This wrapper rewards agents based on their unit's total FOV coverage,
encouraging agents to spread out and face different directions to
maximize map awareness.

Coverage is calculated as percentage of MAXIMUM POSSIBLE coverage
(not grid size). Max possible = num_agents × cells_per_agent_fov.
This means 100% is achievable when agents have zero FOV overlap.

Usage:
    from combatenv.wrappers import FOVCoverageRewardWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = FOVCoverageRewardWrapper(env, coverage_bonus=0.1)
"""

import math
from typing import Dict, Tuple, Any, Set

import gymnasium as gym

from combatenv.config import FAR_FOV_RANGE, FAR_FOV_ANGLE
from combatenv.fov import get_fov_cells


# Calculate cells per agent FOV (circular sector area)
# Area = pi * r^2 * (angle / 360)
CELLS_PER_AGENT_FOV = math.pi * (FAR_FOV_RANGE ** 2) * (FAR_FOV_ANGLE / 360.0)


class FOVCoverageRewardWrapper(gym.Wrapper):
    """
    Reward agents based on unit FOV coverage efficiency.

    Coverage is measured as percentage of maximum possible coverage
    (num_agents × cells_per_agent). This encourages agents to spread
    out and minimize FOV overlap with allies.

    100% coverage = all agents have zero FOV overlap (perfect spread)
    50% coverage = half the potential FOV is wasted on overlap

    Attributes:
        coverage_bonus: Reward multiplier based on coverage percentage
    """

    def __init__(
        self,
        env,
        coverage_bonus: float = 0.1
    ):
        """
        Initialize the FOV coverage reward wrapper.

        Args:
            env: Base environment
            coverage_bonus: Bonus multiplied by coverage ratio (0.0-1.0)
        """
        super().__init__(env)

        self.coverage_bonus = coverage_bonus

        print(f"FOVCoverageRewardWrapper: FOV coverage efficiency reward")
        print(f"  Cells per agent FOV: ~{CELLS_PER_AGENT_FOV:.0f}")
        print(f"  Coverage bonus: +{coverage_bonus} × coverage_ratio (0-1)")

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        self._unit_fov_cache = {}
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        """Step and apply FOV coverage reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply FOV coverage reward shaping
        shaped_reward = self._shape_reward(obs, reward)

        return obs, shaped_reward, terminated, truncated, info

    def _shape_reward(self, obs: Any, base_reward: Any) -> Any:
        """
        Apply FOV coverage reward shaping.

        Args:
            obs: Observation (array or dict)
            base_reward: Original reward

        Returns:
            Shaped reward (float or dict)
        """
        # Only works with multi-agent (dict) observations
        if not isinstance(obs, dict) or not isinstance(base_reward, dict):
            return base_reward

        # Get base environment
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, '_base_env'):
            base_env = base_env._base_env

        # Get units
        blue_units = getattr(base_env, 'blue_units', [])
        if not blue_units:
            return base_reward

        # Calculate FOV coverage per unit
        unit_coverages = self._calculate_unit_coverages(blue_units)

        # Shape rewards based on unit coverage
        shaped_rewards = {}
        num_blue = len(getattr(base_env, 'blue_agents', []))

        for agent_idx, agent_reward in base_reward.items():
            # Only shape blue team rewards (first half of agents)
            if agent_idx >= num_blue:
                shaped_rewards[agent_idx] = agent_reward
                continue

            # Find which unit this agent belongs to
            unit_idx = agent_idx // 8  # Assuming 8 agents per unit

            if unit_idx < len(unit_coverages):
                coverage_ratio = unit_coverages[unit_idx]
                # Reward based on coverage ratio (0.0 to 1.0)
                coverage_reward = coverage_ratio * self.coverage_bonus
                shaped_rewards[agent_idx] = agent_reward + coverage_reward
            else:
                shaped_rewards[agent_idx] = agent_reward

        return shaped_rewards

    def _calculate_unit_coverages(self, units) -> Dict[int, float]:
        """
        Calculate FOV coverage ratio for each unit.

        Coverage is measured against maximum POSSIBLE coverage (not grid size).
        Max possible = num_alive_agents × CELLS_PER_AGENT_FOV

        This means:
        - 100% = perfect spread, zero overlap
        - 50% = half the FOV is wasted on overlap
        - Obstacles reduce actual coverage but max remains theoretical

        Args:
            units: List of Unit objects

        Returns:
            Dict mapping unit_idx -> coverage ratio (0.0-1.0)
        """
        coverages = {}

        for unit_idx, unit in enumerate(units):
            alive_agents = unit.alive_agents if hasattr(unit, 'alive_agents') else []
            if not alive_agents:
                coverages[unit_idx] = 0.0
                continue

            # Max possible coverage = num_agents × cells_per_agent
            max_possible = len(alive_agents) * CELLS_PER_AGENT_FOV

            # Calculate combined FOV for all agents in unit
            combined_fov: Set[Tuple[int, int]] = set()

            for agent in alive_agents:
                agent_fov = get_fov_cells(
                    agent.position,
                    agent.orientation,
                    fov_angle=FAR_FOV_ANGLE,
                    max_range=FAR_FOV_RANGE
                )
                combined_fov.update(agent_fov)

            # Calculate coverage ratio (0.0 to 1.0, can exceed 1.0 in theory)
            coverage_ratio = len(combined_fov) / max_possible if max_possible > 0 else 0.0
            coverages[unit_idx] = min(1.0, coverage_ratio)  # Cap at 1.0

        return coverages
