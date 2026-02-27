"""
Discrete observation wrapper for operational-level (unit) Q-learning.

Converts unit-level observations into discrete state indices for Q-table lookup.

State Space Design (per unit, 576 states):
    - Formation state: 3 levels (cohesive, scattered, broken)
    - Average health: 2 levels (high ≥0.5, low <0.5)
    - Alive ratio: 2 levels (high ≥0.5, low <0.5)
    - Waypoint status: 3 levels (no waypoint, has waypoint, at waypoint)
    - Stance: 3 levels (aggressive, defensive, patrol)
    - Nearest enemy unit: 4 directions (N, S, E, W)

Total per unit: 3 × 2 × 2 × 3 × 3 × 4 = 432 states

For a team with N units, combined state is encoded as base-432 number,
or we track each unit independently (recommended for independent learners).

Usage:
    from combatenv.wrappers import OperationalWrapper, OperationalDiscreteObsWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = OperationalWrapper(env)
    env = OperationalDiscreteObsWrapper(env, team="blue")

    obs, info = env.reset()
    # obs is Dict[int, int] mapping unit_id -> discrete state index
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from combatenv.config import GRID_SIZE, NUM_UNITS_PER_TEAM


class OperationalDiscreteObsWrapper(gym.ObservationWrapper):
    """
    Converts unit-level observations to discrete state indices.

    This wrapper provides discrete states for unit-level Q-learning,
    where each unit is treated as an independent agent.

    Attributes:
        team: Which team to observe ("blue" or "red")
        n_states: Total discrete states per unit (432)
        n_formation: Formation state buckets (3)
        n_health: Health buckets (2)
        n_alive: Alive ratio buckets (2)
        n_waypoint: Waypoint status buckets (3)
        n_stance: Stance buckets (3)
        n_enemy_dir: Enemy direction buckets (4)
    """

    # Formation thresholds (based on formation spread)
    COHESIVE_THRESHOLD = 2.0   # spread < 2.0 = cohesive
    SCATTERED_THRESHOLD = 4.0  # spread < 4.0 = scattered, else broken

    # Resource threshold
    RESOURCE_THRESHOLD = 0.5

    def __init__(self, env, team: str = "blue"):
        """
        Initialize the operational discrete observation wrapper.

        Args:
            env: Environment (should have OperationalWrapper)
            team: Which team to observe ("blue" or "red")
        """
        super().__init__(env)

        self.team = team

        # State space dimensions per unit
        self.n_formation = 3   # cohesive/scattered/broken
        self.n_health = 2      # high/low
        self.n_alive = 2       # high/low
        self.n_waypoint = 3    # none/has/at
        self.n_stance = 3      # aggressive/defensive/patrol
        self.n_enemy_dir = 4   # N/S/E/W

        # Total states per unit
        self.n_states = (
            self.n_formation *
            self.n_health *
            self.n_alive *
            self.n_waypoint *
            self.n_stance *
            self.n_enemy_dir
        )

        print(f"OperationalDiscreteObsWrapper: {self.n_states} states per unit")

    def reset(self, **kwargs) -> Tuple[Dict[int, int], Dict]:
        """
        Reset and return discrete unit observations.

        Returns:
            observations: Dict mapping unit_id -> discrete state index
            info: Environment info dictionary
        """
        obs, info = self.env.reset(**kwargs)
        discrete_obs = self.observation(obs)
        return discrete_obs, info

    def step(self, action) -> Tuple[Dict[int, int], Dict[int, float], bool, bool, Dict]:
        """
        Step and return discrete unit observations.

        Returns:
            observations: Dict mapping unit_id -> discrete state index
            rewards: Dict mapping unit_id -> reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Environment info dictionary
        """
        obs, rewards, terminated, truncated, info = self.env.step(action)
        discrete_obs = self.observation(obs)

        # Convert rewards to unit-level if needed
        unit_rewards = self._aggregate_rewards(rewards)

        return discrete_obs, unit_rewards, terminated, truncated, info

    def observation(self, obs) -> Dict[int, int]:
        """
        Convert continuous observations to discrete unit state indices.

        Args:
            obs: Raw observation from environment

        Returns:
            Dict mapping unit_id -> discrete state index
        """
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{self.team}_units', [])
        enemy_units = getattr(base_env, f'{"red" if self.team == "blue" else "blue"}_units', [])

        discrete_obs = {}
        for unit in units:
            discrete_obs[unit.id] = self._discretize_unit(unit, enemy_units)

        return discrete_obs

    def _discretize_unit(self, unit, enemy_units: List) -> int:
        """
        Convert a single unit's state to discrete index.

        Args:
            unit: Unit instance
            enemy_units: List of enemy units

        Returns:
            Discrete state index (0 to n_states-1)
        """
        # Formation state (0=cohesive, 1=scattered, 2=broken)
        spread = unit.get_formation_spread()
        if spread < self.COHESIVE_THRESHOLD:
            formation = 0
        elif spread < self.SCATTERED_THRESHOLD:
            formation = 1
        else:
            formation = 2

        # Average health
        alive_agents = unit.alive_agents
        if alive_agents:
            avg_health = sum(a.health for a in alive_agents) / len(alive_agents) / 100.0
        else:
            avg_health = 0.0
        health = 1 if avg_health >= self.RESOURCE_THRESHOLD else 0

        # Alive ratio
        alive_ratio = unit.alive_count / max(1, len(unit.agents))
        alive = 1 if alive_ratio >= self.RESOURCE_THRESHOLD else 0

        # Waypoint status (0=none, 1=has, 2=at)
        if unit.waypoint is None:
            waypoint = 0
        elif unit.is_at_waypoint():
            waypoint = 2
        else:
            waypoint = 1

        # Stance (0=aggressive, 1=defensive, 2=patrol)
        stance_map = {"aggressive": 0, "defensive": 1, "patrol": 2}
        stance = stance_map.get(unit.stance, 0)

        # Nearest enemy unit direction
        enemy_dir = self._get_nearest_enemy_direction(unit, enemy_units)

        # Encode as mixed-radix number
        state = (
            formation +
            health * 3 +
            alive * 6 +
            waypoint * 12 +
            stance * 36 +
            enemy_dir * 108
        )

        return state

    def _get_nearest_enemy_direction(self, unit, enemy_units: List) -> int:
        """
        Get direction to nearest enemy unit.

        Args:
            unit: Our unit
            enemy_units: List of enemy units

        Returns:
            Direction index: 0=North, 1=South, 2=East, 3=West
        """
        if not enemy_units:
            return 0  # Default to North

        unit_centroid = unit.centroid
        nearest_dist = float('inf')
        nearest_dir = 0

        for enemy in enemy_units:
            if enemy.alive_count == 0:
                continue

            enemy_centroid = enemy.centroid
            dx = enemy_centroid[0] - unit_centroid[0]
            dy = enemy_centroid[1] - unit_centroid[1]
            dist = dx * dx + dy * dy

            if dist < nearest_dist:
                nearest_dist = dist
                # Determine direction
                if abs(dy) > abs(dx):
                    nearest_dir = 0 if dy < 0 else 1  # North/South
                else:
                    nearest_dir = 2 if dx > 0 else 3  # East/West

        return nearest_dir

    def _aggregate_rewards(self, rewards) -> Dict[int, float]:
        """
        Aggregate agent rewards to unit level.

        Args:
            rewards: Agent-level rewards (or already unit-level)

        Returns:
            Dict mapping unit_id -> reward
        """
        if isinstance(rewards, dict):
            base_env = self.env.unwrapped
            units = getattr(base_env, f'{self.team}_units', [])

            if not units:
                return {}

            # Check if rewards are already unit-level (keyed by unit.id)
            unit_ids = {unit.id for unit in units}
            if rewards and all(k in unit_ids for k in rewards.keys()):
                # Already unit-level rewards, return as-is
                return rewards

            # Otherwise, rewards are agent-indexed - aggregate by unit
            # Build agent index mapping
            all_agents = list(base_env.blue_agents) + list(base_env.red_agents)
            unit_rewards = {}
            for unit in units:
                unit_reward = 0.0
                for agent in unit.agents:
                    # Find agent's index in the combined list
                    if agent in all_agents:
                        agent_idx = all_agents.index(agent)
                        if agent_idx in rewards:
                            unit_reward += rewards[agent_idx]
                unit_rewards[unit.id] = unit_reward

            return unit_rewards
        else:
            # Single reward, distribute to all units
            base_env = self.env.unwrapped
            units = getattr(base_env, f'{self.team}_units', [])
            return {unit.id: float(rewards) / max(1, len(units)) for unit in units}

    def decode_state(self, state: int) -> Dict[str, Any]:
        """
        Decode a state index back to its components (for debugging).

        Args:
            state: Discrete state index

        Returns:
            Dict with decoded components
        """
        formation = state % 3
        health = (state // 3) % 2
        alive = (state // 6) % 2
        waypoint = (state // 12) % 3
        stance = (state // 36) % 3
        enemy_dir = (state // 108) % 4

        formation_names = ["cohesive", "scattered", "broken"]
        waypoint_names = ["none", "has", "at"]
        stance_names = ["aggressive", "defensive", "patrol"]
        direction_names = ["North", "South", "East", "West"]

        return {
            "formation": formation_names[formation],
            "health": "high" if health else "low",
            "alive_ratio": "high" if alive else "low",
            "waypoint": waypoint_names[waypoint],
            "stance": stance_names[stance],
            "enemy_direction": direction_names[enemy_dir],
        }
