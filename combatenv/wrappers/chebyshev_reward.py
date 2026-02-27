"""
Chebyshev distance reward wrapper for movement training.

This wrapper shapes rewards based on Chebyshev distance (max of |dx|, |dy|)
between the agent's tactical position and the waypoint's tactical position.

Uses FINE-GRAINED (tactical cell) distance for dense rewards. Each step
toward the waypoint is rewarded immediately, not just at operational cell boundaries.

Reward Shaping:
    - Decreasing distance: +1.0 * (prev_dist - curr_dist) per tactical cell
    - Reaching waypoint (distance ≤ 2): +50 bonus
    - No progress: -0.1 penalty per step

Usage:
    from combatenv.wrappers import ChebyshevRewardWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = ChebyshevRewardWrapper(env)
"""

from typing import Dict, Tuple, Any, Optional

import numpy as np
import gymnasium as gym

from combatenv.config import OPERATIONAL_GRID_SIZE, TACTICAL_CELLS_PER_OPERATIONAL, STRATEGIC_GRID_SIZE, GRID_SIZE
from combatenv.unit import get_unit_for_agent

# Strategic cell size in tactical cells
TACTICAL_CELLS_PER_STRATEGIC = GRID_SIZE // STRATEGIC_GRID_SIZE  # 64/4 = 16


class ChebyshevRewardWrapper(gym.Wrapper):
    """
    Reward shaping based on Chebyshev distance to waypoint.

    Uses FINE-GRAINED tactical cell distance for dense rewards.
    Each step toward the waypoint is rewarded immediately.

    Attributes:
        distance_reward_scale: Multiplier for distance reduction reward
        strategic_reward_scale: Bonus for crossing into closer strategic cell
        goal_bonus: Bonus for reaching waypoint area
        goal_radius: Tactical cells from waypoint to consider "reached"
        no_progress_penalty: Penalty per step without progress
    """

    def __init__(
        self,
        env,
        distance_reward_scale: float = 1.0,
        strategic_reward_scale: float = 2.0,
        goal_bonus: float = 50.0,
        goal_radius: int = 2,
        no_progress_penalty: float = 0.1
    ):
        """
        Initialize the Chebyshev reward wrapper.

        Args:
            env: Base environment
            distance_reward_scale: Multiplier for distance reduction (per tactical cell)
            strategic_reward_scale: Bonus for moving one strategic cell closer
            goal_bonus: Bonus when reaching waypoint area
            goal_radius: Tactical cells from waypoint to consider "reached"
            no_progress_penalty: Penalty per step without distance reduction
        """
        super().__init__(env)

        self.distance_reward_scale = distance_reward_scale
        self.strategic_reward_scale = strategic_reward_scale
        self.goal_bonus = goal_bonus
        self.goal_radius = goal_radius
        self.no_progress_penalty = no_progress_penalty

        # Track previous Chebyshev distances per agent (in tactical cells)
        self._prev_distances: Dict[int, float] = {}
        # Track previous strategic cell distances per agent
        self._prev_strategic_distances: Dict[int, int] = {}

        print(f"ChebyshevRewardWrapper: Fine-grained movement reward")
        print(f"  Distance reduction: +{distance_reward_scale} per tactical cell")
        print(f"  Strategic cell bonus: +{strategic_reward_scale} per strategic cell")
        print(f"  Goal reached (≤{goal_radius} cells): +{goal_bonus}")
        print(f"  No progress: -{no_progress_penalty}")

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset and initialize distance tracking."""
        result = self.env.reset(**kwargs)

        # Handle both single-agent and multi-agent returns
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        # Initialize distances from observations
        self._init_distances_from_obs(obs)

        return obs, info

    def _init_distances_from_obs(self, obs: Any) -> None:
        """
        Initialize distance tracking from observations.

        Can be called after reset if positions/waypoints change.

        Args:
            obs: Observation (array for single agent, dict for multi-agent)
        """
        self._prev_distances = {}
        self._prev_strategic_distances = {}

        # Single agent case
        if isinstance(obs, np.ndarray):
            dist = self._get_tactical_distance_from_obs(obs)
            self._prev_distances[0] = dist
            self._prev_strategic_distances[0] = self._get_strategic_distance_from_obs(obs)
        # Multi-agent case
        elif isinstance(obs, dict):
            for agent_idx, agent_obs in obs.items():
                dist = self._get_tactical_distance_from_obs(agent_obs)
                self._prev_distances[agent_idx] = dist
                self._prev_strategic_distances[agent_idx] = self._get_strategic_distance_from_obs(agent_obs)

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        """Step and apply Chebyshev distance reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate shaped reward
        shaped_reward = self._shape_reward(obs, reward)

        return obs, shaped_reward, terminated, truncated, info

    def _shape_reward(self, obs: Any, base_reward: float) -> Any:
        """
        Apply Chebyshev distance reward shaping.

        Args:
            obs: Observation (array or dict)
            base_reward: Original reward

        Returns:
            Shaped reward (float or dict)
        """
        # Single agent case
        if isinstance(obs, np.ndarray):
            return self._compute_shaped_reward(0, obs, base_reward)

        # Multi-agent case
        elif isinstance(obs, dict) and isinstance(base_reward, dict):
            shaped_rewards = {}
            for agent_idx, agent_obs in obs.items():
                agent_reward = base_reward.get(agent_idx, 0.0)
                shaped_rewards[agent_idx] = self._compute_shaped_reward(
                    agent_idx, agent_obs, agent_reward
                )
            return shaped_rewards

        return base_reward

    def _compute_shaped_reward(
        self,
        agent_idx: int,
        obs: np.ndarray,
        base_reward: float
    ) -> float:
        """
        Compute shaped reward for a single agent.

        Uses fine-grained tactical cell distance for dense rewards.

        Args:
            agent_idx: Agent index
            obs: Agent's observation
            base_reward: Original reward

        Returns:
            Shaped reward
        """
        curr_dist = self._get_tactical_distance_from_obs(obs)
        prev_dist = self._prev_distances.get(agent_idx, curr_dist)

        curr_strategic = self._get_strategic_distance_from_obs(obs)
        prev_strategic = self._prev_strategic_distances.get(agent_idx, curr_strategic)

        shaping_reward = 0.0

        # Reward for reducing distance (per tactical cell moved closer)
        if curr_dist < prev_dist:
            shaping_reward += (prev_dist - curr_dist) * self.distance_reward_scale
        # Penalty for moving away (per tactical cell)
        elif curr_dist > prev_dist:
            shaping_reward -= (curr_dist - prev_dist) * 1.0

        # Bonus for crossing into closer strategic cell
        if curr_strategic < prev_strategic:
            shaping_reward += (prev_strategic - curr_strategic) * self.strategic_reward_scale
        # Penalty for moving into farther strategic cell
        elif curr_strategic > prev_strategic:
            shaping_reward -= (curr_strategic - prev_strategic) * self.strategic_reward_scale

        # Bonus for reaching goal area
        if curr_dist <= self.goal_radius:
            shaping_reward += self.goal_bonus

        # Penalty for no progress (or moving away)
        if curr_dist >= prev_dist and curr_dist > self.goal_radius:
            shaping_reward -= self.no_progress_penalty

        # Update tracking
        self._prev_distances[agent_idx] = curr_dist
        self._prev_strategic_distances[agent_idx] = curr_strategic

        return base_reward + shaping_reward

    def _get_tactical_distance_from_obs(self, obs: np.ndarray) -> float:
        """
        Extract tactical cell Chebyshev distance from observation.

        Uses obs[89] and obs[90] which are the normalized waypoint relative
        position (-1 to 1, normalized by GRID_SIZE).

        Args:
            obs: Observation array (91 floats)

        Returns:
            Chebyshev distance in tactical cells (0 to ~64)
        """
        from combatenv.config import GRID_SIZE

        if len(obs) < 91:
            # Fallback to operational distance if new obs not available
            return float(self._get_chebyshev_from_obs(obs) * TACTICAL_CELLS_PER_OPERATIONAL)

        # obs[89] = waypoint_rel_x (-1 to 1, normalized by GRID_SIZE)
        # obs[90] = waypoint_rel_y (-1 to 1, normalized by GRID_SIZE)
        dx = abs(obs[89] * GRID_SIZE)
        dy = abs(obs[90] * GRID_SIZE)

        # Chebyshev distance = max(|dx|, |dy|)
        return max(dx, dy)

    def _get_strategic_distance_from_obs(self, obs: np.ndarray) -> int:
        """
        Extract strategic cell Chebyshev distance from observation.

        Strategic grid is 4x4, so each strategic cell is 16 tactical cells.

        Args:
            obs: Observation array (91 floats)

        Returns:
            Chebyshev distance in strategic cells (0 to 3)
        """
        # Get tactical distance first
        tactical_dist = self._get_tactical_distance_from_obs(obs)

        # Convert to strategic cells (16 tactical = 1 strategic)
        strategic_dist = int(tactical_dist / TACTICAL_CELLS_PER_STRATEGIC)

        return min(strategic_dist, STRATEGIC_GRID_SIZE - 1)

    def _get_chebyshev_from_obs(self, obs: np.ndarray) -> int:
        """
        Extract operational grid Chebyshev distance from observation (legacy).

        The observation index 88 contains the normalized Chebyshev distance
        (0.0 = at waypoint, 1.0 = max distance of 7 cells).

        Args:
            obs: Observation array (89+ floats)

        Returns:
            Chebyshev distance as integer (0-7) in operational cells
        """
        if len(obs) < 89:
            # Old observation format without Chebyshev
            return self._compute_chebyshev_directly()

        # Observation index 88 is normalized Chebyshev (0-1)
        normalized_dist = obs[88]

        # Convert back to integer distance (0-7)
        chebyshev = int(round(normalized_dist * (OPERATIONAL_GRID_SIZE - 1)))
        return max(0, min(chebyshev, OPERATIONAL_GRID_SIZE - 1))

    def _compute_chebyshev_directly(self) -> int:
        """
        Compute Chebyshev distance directly from environment state.

        Fallback for environments without the new observation format.

        Returns:
            Chebyshev distance (0-7) or max distance if unavailable
        """
        # Try to access base environment
        base_env = self.env.unwrapped
        if not hasattr(base_env, 'controlled_agent'):
            return OPERATIONAL_GRID_SIZE - 1

        agent = base_env.controlled_agent
        if agent is None or not hasattr(agent, 'unit_id'):
            return OPERATIONAL_GRID_SIZE - 1

        # Get unit and waypoint
        units = getattr(base_env, 'blue_units', []) + getattr(base_env, 'red_units', [])
        unit = get_unit_for_agent(agent, units)

        if unit is None or unit.waypoint is None:
            return OPERATIONAL_GRID_SIZE - 1

        # Compute operational cell positions
        agent_cx = int(agent.position[0] / TACTICAL_CELLS_PER_OPERATIONAL)
        agent_cy = int(agent.position[1] / TACTICAL_CELLS_PER_OPERATIONAL)
        wp_cx = int(unit.waypoint[0] / TACTICAL_CELLS_PER_OPERATIONAL)
        wp_cy = int(unit.waypoint[1] / TACTICAL_CELLS_PER_OPERATIONAL)

        # Chebyshev = max(|dx|, |dy|)
        return max(abs(agent_cx - wp_cx), abs(agent_cy - wp_cy))
