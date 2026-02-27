"""
Operational training wrapper with reward shaping for waypoint navigation.

This wrapper provides:
1. Custom spawn positions (blue at top row, red at bottom row)
2. All units assigned waypoints at center
3. Reward shaping for navigation learning

Usage:
    from combatenv.wrappers import (
        OperationalWrapper,
        OperationalTrainingWrapper,
        OperationalDiscreteObsWrapper,
        OperationalDiscreteActionWrapper,
    )

    env = TacticalCombatEnv(render_mode=None)
    env = OperationalWrapper(env)
    env = OperationalTrainingWrapper(env, team="blue")
    env = OperationalDiscreteObsWrapper(env, team="blue")
    env = OperationalDiscreteActionWrapper(env, team="blue")
"""

import math
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym

from combatenv.config import GRID_SIZE, OPERATIONAL_GRID_SIZE


class OperationalTrainingWrapper(gym.Wrapper):
    """
    Training wrapper with spawn positions and reward shaping for waypoint navigation.

    Spawn Layout (8x8 operational grid):
        ┌────┬────┬────┬────┬────┬────┬────┬────┐
        │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │  ← Blue spawn (row 0)
        ├────┼────┼────┼────┼────┼────┼────┼────┤
        │    │    │    │    │    │    │    │    │
        ├────┼────┼────┼────┼────┼────┼────┼────┤
        │    │    │    │    │    │    │    │    │
        ├────┼────┼────┼────┼────┼────┼────┼────┤
        │    │    │    │ CENTER  │    │    │    │  ← All waypoints here
        ├────┼────┼────┼────┼────┼────┼────┼────┤
        │    │    │    │    │    │    │    │    │
        ├────┼────┼────┼────┼────┼────┼────┼────┤
        │    │    │    │    │    │    │    │    │
        ├────┼────┼────┼────┼────┼────┼────┼────┤
        │ R0 │ R1 │ R2 │ R3 │ R4 │ R5 │ R6 │ R7 │  ← Red spawn (row 7)
        └────┴────┴────┴────┴────┴────┴────┴────┘

    Reward Shaping:
        - Distance reduction: +0.5 per cell closer to waypoint
        - Waypoint reached: +10.0 when within 4 cells of center
        - Formation cohesive: +0.1/step when spread < 2.0
        - No progress penalty: -0.05/step when not moving toward waypoint

    Attributes:
        team: Which team to train ("blue" or "red")
        waypoint_target: Target waypoint position (center of grid)
        prev_distances: Previous distances to waypoint for each unit
    """

    # Reward shaping constants
    DISTANCE_REDUCTION_BONUS = 0.5   # Per cell closer
    WAYPOINT_REACHED_BONUS = 10.0    # When within arrival distance
    WAYPOINT_ARRIVAL_DIST = 4.0      # Distance to consider waypoint reached
    FORMATION_COHESION_BONUS = 0.1   # Per step when cohesive
    FORMATION_COHESION_THRESHOLD = 2.0  # Max spread for cohesion bonus
    NO_PROGRESS_PENALTY = -0.05      # Per step when not progressing

    def __init__(self, env, team: str = "blue"):
        """
        Initialize the training wrapper.

        Args:
            env: Environment (should have OperationalWrapper)
            team: Which team to train ("blue" or "red")
        """
        super().__init__(env)
        self.team = team
        self.cell_size = GRID_SIZE / OPERATIONAL_GRID_SIZE
        self.waypoint_target = (GRID_SIZE / 2, GRID_SIZE / 2)

        # Track previous distances for reward shaping
        self.prev_distances: Dict[int, float] = {}

        print(f"OperationalTrainingWrapper: team={team}, waypoint=center")

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Reset environment with custom spawn positions.

        Blue units spawn at top row (row 0), Red units at bottom row (row 7).
        All units get waypoints set to center.

        Returns:
            observation: Initial observation
            info: Environment info dictionary
        """
        # Skip base auto-dispatch so we can set custom spawn positions
        options = kwargs.get("options") or {}
        options["skip_auto_dispatch"] = True
        kwargs["options"] = options

        obs, info = self.env.reset(**kwargs)

        base_env = self.env.unwrapped
        blue_units = getattr(base_env, 'blue_units', [])
        red_units = getattr(base_env, 'red_units', [])

        # Dispatch blue units to top row (row 0, cells 0-7)
        for i, unit in enumerate(blue_units):
            cell_x = i % OPERATIONAL_GRID_SIZE
            cell_y = 0  # Top row
            spawn_x = (cell_x + 0.5) * self.cell_size
            spawn_y = (cell_y + 0.5) * self.cell_size
            unit.dispatch_to((spawn_x, spawn_y), self.waypoint_target)

        # Dispatch red units to bottom row (row 7, cells 56-63)
        for i, unit in enumerate(red_units):
            cell_x = i % OPERATIONAL_GRID_SIZE
            cell_y = OPERATIONAL_GRID_SIZE - 1  # Bottom row
            spawn_x = (cell_x + 0.5) * self.cell_size
            spawn_y = (cell_y + 0.5) * self.cell_size
            unit.dispatch_to((spawn_x, spawn_y), self.waypoint_target)

        # Initialize previous distances for reward shaping
        self.prev_distances = {}
        units = blue_units if self.team == "blue" else red_units
        for unit in units:
            self.prev_distances[unit.id] = self._get_distance_to_waypoint(unit)

        return obs, info

    def step(self, action) -> Tuple[Any, Dict[int, float], bool, bool, Dict]:
        """
        Step with reward shaping for navigation.

        Args:
            action: Action dictionary

        Returns:
            observation: Next observation
            rewards: Shaped rewards per unit
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Environment info dictionary
        """
        obs, rewards, terminated, truncated, info = self.env.step(action)

        # Get units for the training team
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{self.team}_units', [])

        # Calculate shaped rewards
        shaped_rewards = {}
        for unit in units:
            if unit.alive_count == 0:
                shaped_rewards[unit.id] = 0.0
                continue

            # Get current distance to waypoint
            curr_dist = self._get_distance_to_waypoint(unit)
            prev_dist = self.prev_distances.get(unit.id, curr_dist)

            # Calculate shaped reward
            shaped_reward = self._calculate_shaped_reward(unit, prev_dist, curr_dist)

            # Add base reward if available
            base_reward = rewards.get(unit.id, 0.0) if isinstance(rewards, dict) else 0.0
            shaped_rewards[unit.id] = base_reward + shaped_reward

            # Update previous distance
            self.prev_distances[unit.id] = curr_dist

        return obs, shaped_rewards, terminated, truncated, info

    def _get_distance_to_waypoint(self, unit) -> float:
        """
        Get Euclidean distance from unit centroid to waypoint target.

        Args:
            unit: Unit instance

        Returns:
            Distance in grid cells
        """
        centroid = unit.centroid
        dx = centroid[0] - self.waypoint_target[0]
        dy = centroid[1] - self.waypoint_target[1]
        return math.sqrt(dx * dx + dy * dy)

    def _calculate_shaped_reward(self, unit, prev_dist: float, curr_dist: float) -> float:
        """
        Calculate shaped reward for a unit.

        Args:
            unit: Unit instance
            prev_dist: Previous distance to waypoint
            curr_dist: Current distance to waypoint

        Returns:
            Shaped reward value
        """
        reward = 0.0

        # 1. Distance reduction bonus (+0.5 per cell closer)
        dist_reduction = prev_dist - curr_dist
        if dist_reduction > 0:
            reward += dist_reduction * self.DISTANCE_REDUCTION_BONUS
        else:
            # Penalty for no progress
            reward += self.NO_PROGRESS_PENALTY

        # 2. Waypoint arrival bonus (+10.0)
        if curr_dist < self.WAYPOINT_ARRIVAL_DIST:
            reward += self.WAYPOINT_REACHED_BONUS

        # 3. Formation cohesion bonus (+0.1 if cohesive)
        if unit.get_formation_spread() < self.FORMATION_COHESION_THRESHOLD:
            reward += self.FORMATION_COHESION_BONUS

        return reward

    def get_unit_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Get current stats for all units in training team.

        Returns:
            Dict mapping unit_id -> stats dict
        """
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{self.team}_units', [])

        stats = {}
        for unit in units:
            stats[unit.id] = {
                "distance_to_waypoint": self._get_distance_to_waypoint(unit),
                "formation_spread": unit.get_formation_spread(),
                "alive_count": unit.alive_count,
                "at_waypoint": self._get_distance_to_waypoint(unit) < self.WAYPOINT_ARRIVAL_DIST,
            }
        return stats
