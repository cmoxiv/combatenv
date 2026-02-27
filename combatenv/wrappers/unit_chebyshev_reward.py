"""
Unit centroid-based Chebyshev distance reward wrapper.

Unlike ChebyshevRewardWrapper which uses individual agent positions,
this wrapper calculates distance from the UNIT CENTROID to waypoint.
All agents in a unit receive the same distance-based reward, encouraging
group movement toward the waypoint.

Reward Shaping:
    - Decreasing centroid distance: +scale * (prev_dist - curr_dist)
    - Reaching waypoint (centroid distance <= goal_radius): +goal_bonus
    - No progress: -no_progress_penalty per step

Usage:
    from combatenv.wrappers import UnitChebyshevWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = UnitChebyshevWrapper(env)
"""

from typing import Dict, Tuple, Any, Optional

import numpy as np
import gymnasium as gym

from combatenv.config import GRID_SIZE


class UnitChebyshevWrapper(gym.Wrapper):
    """
    Reward shaping based on unit centroid Chebyshev distance to waypoint.

    All agents in a unit get the same reward based on their unit's
    collective position (centroid), encouraging group movement.

    Attributes:
        distance_reward_scale: Multiplier for distance reduction reward
        goal_bonus: Bonus for reaching waypoint area
        goal_radius: Tactical cells from waypoint to consider "reached"
        no_progress_penalty: Penalty per step without progress
    """

    def __init__(
        self,
        env,
        distance_reward_scale: float = 1.0,
        goal_bonus: float = 50.0,
        goal_radius: float = 5.0,
        no_progress_penalty: float = 0.1
    ):
        """
        Initialize the unit Chebyshev reward wrapper.

        Args:
            env: Base environment (must have MultiAgentWrapper in chain)
            distance_reward_scale: Multiplier for distance reduction
            goal_bonus: Bonus when unit centroid reaches waypoint area
            goal_radius: Tactical cells from waypoint to consider "reached"
            no_progress_penalty: Penalty per step without distance reduction
        """
        super().__init__(env)

        self.distance_reward_scale = distance_reward_scale
        self.goal_bonus = goal_bonus
        self.goal_radius = goal_radius
        self.no_progress_penalty = no_progress_penalty

        # Track previous Chebyshev distances per unit (keyed by unit_id)
        self._prev_distances: Dict[int, float] = {}

        print(f"UnitChebyshevWrapper: Unit centroid-based movement reward")
        print(f"  Distance reduction: +{distance_reward_scale} per tactical cell")
        print(f"  Goal reached (centroid <= {goal_radius} cells): +{goal_bonus}")
        print(f"  No progress: -{no_progress_penalty}")

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset and initialize distance tracking per unit."""
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        # Initialize distances from unit centroids
        self._init_unit_distances()

        return obs, info

    def _init_unit_distances(self) -> None:
        """Initialize distance tracking for all units."""
        self._prev_distances = {}

        base_env = self.env.unwrapped
        blue_units = getattr(base_env, 'blue_units', [])
        red_units = getattr(base_env, 'red_units', [])

        # Blue units: unit_id = 0 to len(blue_units)-1
        for i, unit in enumerate(blue_units):
            dist = self._get_unit_centroid_distance(unit)
            self._prev_distances[i] = dist

        # Red units: unit_id = len(blue_units) to len(blue_units)+len(red_units)-1
        offset = len(blue_units)
        for i, unit in enumerate(red_units):
            dist = self._get_unit_centroid_distance(unit)
            self._prev_distances[offset + i] = dist

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        """Step and apply unit centroid Chebyshev distance reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate shaped reward
        shaped_reward = self._shape_reward(obs, reward)

        return obs, shaped_reward, terminated, truncated, info

    def _shape_reward(self, obs: Any, base_reward: Any) -> Any:
        """
        Apply unit centroid Chebyshev distance reward shaping.

        Args:
            obs: Observation (array or dict)
            base_reward: Original reward

        Returns:
            Shaped reward (float or dict)
        """
        # Only works with multi-agent (dict) rewards
        if not isinstance(base_reward, dict):
            return base_reward

        shaped_rewards = {}
        for agent_idx, agent_reward in base_reward.items():
            shaped_rewards[agent_idx] = self._compute_shaped_reward(
                agent_idx, agent_reward
            )
        return shaped_rewards

    def _compute_shaped_reward(
        self,
        agent_idx: int,
        base_reward: float
    ) -> float:
        """
        Compute shaped reward for an agent based on unit centroid distance.

        All agents in the same unit receive the same distance-based reward.

        Args:
            agent_idx: Agent index
            base_reward: Original reward

        Returns:
            Shaped reward
        """
        unit, unit_id = self._get_unit_for_agent_idx(agent_idx)
        if unit is None:
            return base_reward

        curr_dist = self._get_unit_centroid_distance(unit)
        prev_dist = self._prev_distances.get(unit_id, curr_dist)

        shaping_reward = 0.0

        # Reward for reducing distance
        if curr_dist < prev_dist:
            shaping_reward += (prev_dist - curr_dist) * self.distance_reward_scale
        # Penalty for moving away
        elif curr_dist > prev_dist:
            shaping_reward -= (curr_dist - prev_dist) * self.distance_reward_scale

        # Bonus for reaching goal area
        if curr_dist <= self.goal_radius:
            shaping_reward += self.goal_bonus

        # Penalty for no progress
        if curr_dist >= prev_dist and curr_dist > self.goal_radius:
            shaping_reward -= self.no_progress_penalty

        # Update tracking per unit
        self._prev_distances[unit_id] = curr_dist

        return base_reward + shaping_reward

    def _get_unit_for_agent_idx(self, agent_idx: int) -> Tuple[Any, int]:
        """
        Map agent index to (unit, unit_id).

        Args:
            agent_idx: Agent index (0 to num_agents-1)

        Returns:
            Tuple of (unit object, unit_id) or (None, -1) if not found
        """
        base_env = self.env.unwrapped
        blue_units = getattr(base_env, 'blue_units', [])
        red_units = getattr(base_env, 'red_units', [])

        agents_per_unit = 8
        num_blue_agents = len(blue_units) * agents_per_unit

        if agent_idx < num_blue_agents:
            unit_idx = agent_idx // agents_per_unit
            if unit_idx < len(blue_units):
                return blue_units[unit_idx], unit_idx
        else:
            unit_idx = (agent_idx - num_blue_agents) // agents_per_unit
            if unit_idx < len(red_units):
                # Offset red unit IDs to avoid collision with blue
                return red_units[unit_idx], len(blue_units) + unit_idx

        return None, -1

    def _get_unit_centroid_distance(self, unit) -> float:
        """
        Calculate Chebyshev distance from unit centroid to waypoint.

        Args:
            unit: Unit object with centroid and waypoint attributes

        Returns:
            Chebyshev distance in tactical cells
        """
        if unit is None or unit.waypoint is None:
            return float(GRID_SIZE)

        centroid = unit.centroid  # (x, y)
        waypoint = unit.waypoint  # (x, y)

        dx = abs(centroid[0] - waypoint[0])
        dy = abs(centroid[1] - waypoint[1])

        # Chebyshev distance = max(|dx|, |dy|)
        return max(dx, dy)

    def reinit_distances(self) -> None:
        """
        Reinitialize distance tracking.

        Call this after spawning/waypoint changes to ensure correct initial distances.
        """
        self._init_unit_distances()
