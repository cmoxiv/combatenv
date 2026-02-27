"""
WaypointTaskWrapper - Task wrapper for waypoint navigation training.

This wrapper sets up the training scenario for waypoint navigation:
- Spawns blue units at top row operational cells
- Spawns red units at bottom row operational cells
- Assigns independent goal waypoint to each unit (opposite side)
- Calculates distance-based rewards per unit

Usage:
    env = TacticalCombatEnv(render_mode=None)
    env = OperationalWrapper(env)
    env = WaypointTaskWrapper(env)
"""

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..config import (
    TACTICAL_CELLS_PER_OPERATIONAL,
    OPERATIONAL_GRID_SIZE,
)


class WaypointTaskWrapper(BaseWrapper):
    """
    Task wrapper for waypoint navigation training.

    Spawns units at designated positions and calculates distance-based
    rewards for each unit based on progress toward their individual goals.

    Each unit gets an independent random goal waypoint assigned on reset.

    Attributes:
        previous_distances: Dict mapping (team, unit_id) to previous distance
        episode_rewards: Dict mapping (team, unit_id) to cumulative episode reward
    """

    # Reward configuration - distance is primary signal
    DISTANCE_REWARD_PER_CELL = 10.0   # Per cell closer (+) or farther (-) from goal
    WAYPOINT_REACHED_BONUS = 50.0     # One-off bonus when unit first reaches goal
    WAYPOINT_THRESHOLD = 4.0          # Distance to consider goal "reached"

    # Secondary rewards (small, don't overwhelm distance signal)
    COHESION_BONUS = 0.01             # Per step when formation spread < 2.0
    DEAD_PENALTY_PER_AGENT = -0.1     # Penalty per dead agent per step
    STANDING_STILL_PENALTY = -0.1     # Penalty per step when not moving (smaller than distance reward)
    STANDING_STILL_THRESHOLD = 0.01   # Min distance delta to count as movement (agents move ~0.05/step)

    def __init__(
        self,
        env: gym.Env,
    ):
        """
        Initialize the WaypointTaskWrapper.

        Args:
            env: Base environment (should have OperationalWrapper)
        """
        super().__init__(env)

        self.previous_distances: Dict[Tuple[str, int], float] = {}
        self.episode_rewards: Dict[Tuple[str, int], float] = {}
        self.waypoint_reached: Set[Tuple[str, int]] = set()  # Track one-off bonus

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset and set up training scenario.

        Spawns units at designated positions and assigns waypoints.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Get units
        blue_units = getattr(self.env, 'blue_units', [])
        red_units = getattr(self.env, 'red_units', [])

        # Spawn blue units at top row (row 0) with random goals
        self._spawn_units_with_random_goals(blue_units, spawn_row=0)

        # Spawn red units at bottom row (row 7) with random goals
        self._spawn_units_with_random_goals(red_units, spawn_row=OPERATIONAL_GRID_SIZE - 1)

        # Initialize distance tracking
        self.previous_distances = {}
        self.episode_rewards = {}
        self.waypoint_reached = set()

        for unit in blue_units:
            key = ("blue", unit.id)
            self.previous_distances[key] = self._get_distance_to_goal(unit)
            self.episode_rewards[key] = 0.0

        for unit in red_units:
            key = ("red", unit.id)
            self.previous_distances[key] = self._get_distance_to_goal(unit)
            self.episode_rewards[key] = 0.0

        return obs, info

    def _spawn_units_with_random_goals(
        self,
        units: List,
        spawn_row: int,
    ) -> None:
        """
        Spawn units along an operational grid row with random goals.

        Each unit spawns in a column and gets a random goal position
        anywhere on the grid.

        Args:
            units: List of Unit objects
            spawn_row: Operational grid row to spawn at (0-7)
        """
        cell_size = TACTICAL_CELLS_PER_OPERATIONAL  # 8 tactical cells per op cell

        for i, unit in enumerate(units):
            # Calculate spawn position (center of operational cell)
            col = i % OPERATIONAL_GRID_SIZE
            spawn_x = (col + 0.5) * cell_size
            spawn_y = (spawn_row + 0.5) * cell_size

            # Random goal position (center of a random operational cell)
            goal_col = random.randint(0, OPERATIONAL_GRID_SIZE - 1)
            goal_row = random.randint(0, OPERATIONAL_GRID_SIZE - 1)
            goal_x = (goal_col + 0.5) * cell_size
            goal_y = (goal_row + 0.5) * cell_size

            # Dispatch unit to spawn position
            unit.dispatch_to(
                (spawn_x, spawn_y),
                (goal_x, goal_y),  # This sets intermediate waypoint
            )

            # Set the goal waypoint (for reward calculation)
            unit.set_goal(goal_x, goal_y)

            # Clear intermediate waypoint - units learn to navigate via actions
            unit.clear_waypoint()

    def step(
        self,
        action: Any
    ) -> Tuple[np.ndarray, Dict[Tuple[str, int], float], bool, bool, Dict[str, Any]]:
        """
        Execute one step and calculate per-unit rewards.

        Args:
            action: Action(s) for units

        Returns:
            Tuple of (observation, rewards_dict, terminated, truncated, info)
            rewards_dict maps (team, unit_id) to reward
        """
        # Step base environment (base_reward unused - we calculate per-unit rewards)
        obs, _, terminated, truncated, info = self.env.step(action)

        # Calculate per-unit rewards
        rewards = self._calculate_rewards()

        # Update episode rewards
        for key, reward in rewards.items():
            self.episode_rewards[key] = self.episode_rewards.get(key, 0.0) + reward

        # Add to info
        info["unit_rewards"] = rewards
        info["episode_rewards"] = dict(self.episode_rewards)

        return obs, rewards, terminated, truncated, info

    def _calculate_rewards(self) -> Dict[Tuple[str, int], float]:
        """
        Calculate rewards for all units based on waypoint progress.

        Returns:
            Dict mapping (team, unit_id) to reward
        """
        rewards = {}

        blue_units = getattr(self.env, 'blue_units', [])
        red_units = getattr(self.env, 'red_units', [])

        for unit in blue_units:
            key = ("blue", unit.id)
            rewards[key] = self._calculate_unit_reward(unit, key)

        for unit in red_units:
            key = ("red", unit.id)
            rewards[key] = self._calculate_unit_reward(unit, key)

        return rewards

    def _calculate_unit_reward(
        self,
        unit,
        key: Tuple[str, int],
    ) -> float:
        """
        Calculate reward for a single unit.

        Reward components:
        - Distance-based: +1.0/cell closer, -1.0/cell farther (primary signal)
        - Goal reached bonus: +50.0 (one-off, first time only)
        - Cohesion bonus: +0.01 when spread < 2.0
        - Dead penalty: -0.1 per dead agent per step

        Args:
            unit: Unit object
            key: (team, unit_id) tuple

        Returns:
            Reward value
        """
        reward = 0.0

        # Get all agents and alive agents
        all_agents = unit.agents
        alive_agents = unit.alive_agents
        num_total = len(all_agents)
        num_alive = len(alive_agents)
        num_dead = num_total - num_alive

        # No reward if unit is completely dead
        if num_alive == 0:
            return 0.0

        # Get current distance to goal
        current_distance = self._get_distance_to_goal(unit)

        # No reward if no valid goal (distance is inf)
        if math.isinf(current_distance):
            return 0.0

        previous_distance = self.previous_distances.get(key, current_distance)

        # Handle case where previous distance was inf (first step after getting goal)
        if math.isinf(previous_distance):
            previous_distance = current_distance

        # Distance-based reward: +1.0/cell closer, -1.0/cell farther
        distance_delta = previous_distance - current_distance
        reward += distance_delta * self.DISTANCE_REWARD_PER_CELL

        # Check if goal is reached
        goal_reached = current_distance < self.WAYPOINT_THRESHOLD

        # Penalty for standing still (not making progress) - skip if at goal
        if abs(distance_delta) < self.STANDING_STILL_THRESHOLD and not goal_reached:
            reward += self.STANDING_STILL_PENALTY

        # Goal reached bonus (one-off)
        if goal_reached:
            if key not in self.waypoint_reached:
                reward += self.WAYPOINT_REACHED_BONUS
                self.waypoint_reached.add(key)

        # Cohesion bonus (small)
        spread = unit.get_formation_spread()
        if spread < 2.0:
            reward += self.COHESION_BONUS

        # Dead agent penalty
        reward += num_dead * self.DEAD_PENALTY_PER_AGENT

        # Update previous distance
        self.previous_distances[key] = current_distance

        return reward

    def _get_distance_to_goal(self, unit) -> float:
        """
        Get Euclidean distance from unit centroid to its goal waypoint.

        Args:
            unit: Unit object

        Returns:
            Distance in grid cells, or inf if no goal or no alive agents
        """
        if not unit.alive_agents:
            return float('inf')

        if unit.goal_waypoint is None:
            return float('inf')

        centroid = unit.centroid
        dx = centroid[0] - unit.goal_waypoint[0]
        dy = centroid[1] - unit.goal_waypoint[1]
        return math.sqrt(dx * dx + dy * dy)

    def get_episode_rewards(self) -> Dict[Tuple[str, int], float]:
        """
        Get cumulative episode rewards for all units.

        Returns:
            Dict mapping (team, unit_id) to cumulative reward
        """
        return dict(self.episode_rewards)

    def get_team_episode_reward(self, team: str) -> float:
        """
        Get total episode reward for a team.

        Args:
            team: "blue" or "red"

        Returns:
            Sum of rewards for all units in the team
        """
        return sum(
            reward for (t, _), reward in self.episode_rewards.items()
            if t == team
        )
