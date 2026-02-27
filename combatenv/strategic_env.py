"""
Strategic-level environment for hierarchical multi-agent RL.

This environment provides a high-level interface for commanding units
via waypoint dispatch. The strategic agent observes unit-level information
and assigns waypoints to units, while the tactical layer (individual agents)
handles movement and combat.

Architecture:
    Strategic Agent (this environment)
         |
         | Assigns waypoints to units
         v
    Tactical Layer (TacticalCombatEnv + wrappers)
         |
         | Individual agent actions
         v
    Simulation

Usage:
    from combatenv import TacticalCombatEnv
    from combatenv.strategic_env import StrategicCombatEnv
    from rl_student.wrappers import MultiAgentWrapper, UnitWrapper

    # Create tactical env with unit support
    tactical_env = TacticalCombatEnv(render_mode=None)
    tactical_env = MultiAgentWrapper(tactical_env)
    tactical_env = UnitWrapper(tactical_env)

    # Create strategic wrapper
    strategic_env = StrategicCombatEnv(tactical_env)
"""

from typing import Dict, Tuple, List, Optional, Any
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import (
    GRID_SIZE,
    NUM_UNITS_PER_TEAM,
)


# Strategic observation size per unit: 7 floats
# (centroid_x, centroid_y, health, alive_count, waypoint_x, waypoint_y, formation_state)
UNIT_OBS_SIZE = 7

# Total strategic observation:
# - 7 floats * 10 units * 2 teams = 140
# - Plus 20 floats for global state (team health, kill counts, etc.)
STRATEGIC_OBS_SIZE = UNIT_OBS_SIZE * NUM_UNITS_PER_TEAM * 2 + 20

# Waypoint grid discretization: 8x8 = 64 possible positions
WAYPOINT_GRID_SIZE = 8


class StrategicCombatEnv(gym.Env):
    """
    High-level Gymnasium environment for strategic unit control.

    This environment wraps a tactical environment (with UnitWrapper) and
    provides a macro-level interface for commanding unit movements.

    Observation Space:
        Box(160,) containing:
        - Per unit (7 floats * 10 units * 2 teams = 140):
            - Centroid position (x, y) normalized
            - Average unit health (0-1)
            - Alive agent count (0-1)
            - Current waypoint (x, y) or (0.5, 0.5) if none
            - Formation state (0=cohesive, 0.5=scattered, 1=broken)
        - Global state (20 floats):
            - Blue team average health
            - Red team average health
            - Blue team alive count (normalized)
            - Red team alive count (normalized)
            - Blue kills, Red kills
            - Game progress (steps / max_steps)
            - [Padding to 20]

    Action Space:
        MultiDiscrete([64] * 10) - waypoint index for each blue unit
        Each action is an index into an 8x8 grid covering the battlefield.

    Attributes:
        tactical_env: The wrapped tactical environment with UnitWrapper
        tactical_steps_per_strategic: How many tactical steps per strategic step
        controlled_team: Which team the strategic agent controls ("blue" or "red")
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        tactical_env,
        tactical_steps_per_strategic: int = 30,
        controlled_team: str = "blue",
        render_mode: Optional[str] = None
    ):
        """
        Initialize the strategic environment.

        Args:
            tactical_env: Tactical environment with UnitWrapper applied
            tactical_steps_per_strategic: Tactical steps per strategic action
            controlled_team: Which team to control ("blue" or "red")
            render_mode: Render mode for visualization
        """
        super().__init__()

        self.tactical_env = tactical_env
        self.tactical_steps_per_strategic = tactical_steps_per_strategic
        self.controlled_team = controlled_team
        self.render_mode = render_mode

        # Strategic observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(STRATEGIC_OBS_SIZE,),
            dtype=np.float32
        )

        # Action space: waypoint for each controlled unit
        # 8x8 grid = 64 discrete waypoint positions
        self.action_space = spaces.MultiDiscrete(
            [WAYPOINT_GRID_SIZE * WAYPOINT_GRID_SIZE] * NUM_UNITS_PER_TEAM
        )

        # State tracking
        self._step_count = 0
        self._max_steps = 3000  # Default max tactical steps

        print(f"StrategicCombatEnv initialized:")
        print(f"  Controlled team: {controlled_team}")
        print(f"  Tactical steps per strategic: {tactical_steps_per_strategic}")
        print(f"  Waypoint grid: {WAYPOINT_GRID_SIZE}x{WAYPOINT_GRID_SIZE}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)

        # Reset tactical env
        _, info = self.tactical_env.reset(seed=seed, options=options)

        self._step_count = 0

        # Get strategic observation
        obs = self._get_strategic_observation()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute a strategic action.

        Assigns waypoints to controlled units, then runs multiple
        tactical steps before returning.

        Args:
            action: Array of waypoint indices for each unit

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Assign waypoints to units
        self._assign_waypoints(action)

        # Run tactical steps
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.tactical_steps_per_strategic):
            # Get tactical actions (could be from tactical policy or autonomous)
            tactical_actions = self._get_tactical_actions()

            # Step tactical env
            _, rewards, terminated, truncated, step_info = self.tactical_env.step(tactical_actions)

            # Accumulate rewards for controlled team
            team_reward = self._get_team_reward(rewards)
            total_reward += team_reward

            self._step_count += 1
            info = step_info

            if terminated or truncated:
                break

        # Get strategic observation
        obs = self._get_strategic_observation()

        return obs, total_reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.tactical_env.render()

    def close(self):
        """Clean up resources."""
        self.tactical_env.close()

    def _assign_waypoints(self, action: np.ndarray) -> None:
        """
        Convert action indices to waypoints and assign to units.

        Args:
            action: Array of waypoint indices (0-63) for each unit
        """
        units = self._get_controlled_units()

        for i, unit in enumerate(units):
            if i >= len(action):
                break

            waypoint_idx = action[i]
            x, y = self._index_to_position(waypoint_idx)
            unit.set_waypoint(x, y)

    def _index_to_position(self, index: int) -> Tuple[float, float]:
        """
        Convert waypoint index to grid position.

        Args:
            index: Waypoint index (0 to WAYPOINT_GRID_SIZE^2 - 1)

        Returns:
            (x, y) position in grid coordinates
        """
        grid_x = index % WAYPOINT_GRID_SIZE
        grid_y = index // WAYPOINT_GRID_SIZE

        # Convert to actual grid position (with margins)
        margin = 2.0
        cell_size = (GRID_SIZE - 2 * margin) / WAYPOINT_GRID_SIZE

        x = margin + (grid_x + 0.5) * cell_size
        y = margin + (grid_y + 0.5) * cell_size

        return (x, y)

    def _position_to_index(self, x: float, y: float) -> int:
        """
        Convert grid position to waypoint index.

        Args:
            x: X position
            y: Y position

        Returns:
            Waypoint index
        """
        margin = 2.0
        cell_size = (GRID_SIZE - 2 * margin) / WAYPOINT_GRID_SIZE

        grid_x = int((x - margin) / cell_size)
        grid_y = int((y - margin) / cell_size)

        # Clamp to valid range
        grid_x = max(0, min(WAYPOINT_GRID_SIZE - 1, grid_x))
        grid_y = max(0, min(WAYPOINT_GRID_SIZE - 1, grid_y))

        return grid_y * WAYPOINT_GRID_SIZE + grid_x

    def _get_controlled_units(self) -> List:
        """Get units for the controlled team."""
        if self.controlled_team == "blue":
            return getattr(self.tactical_env, 'blue_units', [])
        else:
            return getattr(self.tactical_env, 'red_units', [])

    def _get_enemy_units(self) -> List:
        """Get units for the enemy team."""
        if self.controlled_team == "blue":
            return getattr(self.tactical_env, 'red_units', [])
        else:
            return getattr(self.tactical_env, 'blue_units', [])

    def _get_tactical_actions(self) -> Dict[int, np.ndarray]:
        """
        Get tactical actions for all agents.

        For now, returns no-op actions (let boids handle movement).
        Could be extended to use a tactical policy.

        Returns:
            Dict of agent_idx -> action array
        """
        actions = {}

        # Get agent count
        num_blue = len(getattr(self.tactical_env, 'blue_agents', []) or
                       getattr(self.tactical_env.unwrapped, 'blue_agents', []))
        num_red = len(getattr(self.tactical_env, 'red_agents', []) or
                      getattr(self.tactical_env.unwrapped, 'red_agents', []))

        # Create no-op actions (let boids/autonomous behavior work)
        for i in range(num_blue):
            actions[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        for i in range(num_red):
            actions[100 + i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return actions

    def _get_team_reward(self, rewards: Dict[int, float]) -> float:
        """
        Sum rewards for the controlled team.

        Args:
            rewards: Dict of agent_idx -> reward

        Returns:
            Total team reward
        """
        total = 0.0

        if self.controlled_team == "blue":
            for idx in range(100):
                total += rewards.get(idx, 0.0)
        else:
            for idx in range(100, 200):
                total += rewards.get(idx, 0.0)

        return total

    def _get_strategic_observation(self) -> np.ndarray:
        """
        Build strategic observation from current state.

        Returns:
            Array of shape (STRATEGIC_OBS_SIZE,)
        """
        obs = np.zeros(STRATEGIC_OBS_SIZE, dtype=np.float32)

        # Get units
        blue_units = getattr(self.tactical_env, 'blue_units', [])
        red_units = getattr(self.tactical_env, 'red_units', [])

        # Fill unit observations
        idx = 0

        # Blue units (first 70 values)
        for i in range(NUM_UNITS_PER_TEAM):
            if i < len(blue_units):
                unit_obs = self._get_unit_observation(blue_units[i])
            else:
                unit_obs = np.zeros(UNIT_OBS_SIZE, dtype=np.float32)
            obs[idx:idx + UNIT_OBS_SIZE] = unit_obs
            idx += UNIT_OBS_SIZE

        # Red units (next 70 values)
        for i in range(NUM_UNITS_PER_TEAM):
            if i < len(red_units):
                unit_obs = self._get_unit_observation(red_units[i])
            else:
                unit_obs = np.zeros(UNIT_OBS_SIZE, dtype=np.float32)
            obs[idx:idx + UNIT_OBS_SIZE] = unit_obs
            idx += UNIT_OBS_SIZE

        # Global state (last 20 values)
        global_obs = self._get_global_observation()
        obs[idx:idx + 20] = global_obs

        return obs

    def _get_unit_observation(self, unit) -> np.ndarray:
        """
        Get observation for a single unit.

        Args:
            unit: Unit instance

        Returns:
            Array of shape (UNIT_OBS_SIZE,)
        """
        obs = np.zeros(UNIT_OBS_SIZE, dtype=np.float32)

        # Centroid position (normalized)
        centroid = unit.centroid
        obs[0] = centroid[0] / GRID_SIZE
        obs[1] = centroid[1] / GRID_SIZE

        # Average health
        alive_agents = unit.alive_agents
        if alive_agents:
            avg_health = sum(a.health for a in alive_agents) / len(alive_agents) / 100.0
        else:
            avg_health = 0.0
        obs[2] = avg_health

        # Alive count (normalized)
        obs[3] = unit.alive_count / len(unit.agents)

        # Waypoint (or 0.5, 0.5 if none)
        if unit.waypoint is not None:
            obs[4] = unit.waypoint[0] / GRID_SIZE
            obs[5] = unit.waypoint[1] / GRID_SIZE
        else:
            obs[4] = 0.5
            obs[5] = 0.5

        # Formation state
        spread = unit.get_formation_spread()
        if spread < 2.0:
            obs[6] = 0.0  # Cohesive
        elif spread < 4.0:
            obs[6] = 0.5  # Scattered
        else:
            obs[6] = 1.0  # Broken

        return obs

    def _get_global_observation(self) -> np.ndarray:
        """
        Get global game state observation.

        Returns:
            Array of shape (20,)
        """
        obs = np.zeros(20, dtype=np.float32)

        # Get agent lists
        blue_agents = getattr(self.tactical_env, 'blue_agents', None)
        red_agents = getattr(self.tactical_env, 'red_agents', None)

        if blue_agents is None:
            unwrapped = self.tactical_env.unwrapped
            blue_agents = getattr(unwrapped, 'blue_agents', [])
            red_agents = getattr(unwrapped, 'red_agents', [])

        # Blue team stats
        if blue_agents:
            blue_alive = [a for a in blue_agents if a.is_alive]
            obs[0] = sum(a.health for a in blue_alive) / (len(blue_agents) * 100) if blue_alive else 0
            obs[1] = len(blue_alive) / len(blue_agents)

        # Red team stats
        if red_agents:
            red_alive = [a for a in red_agents if a.is_alive]
            obs[2] = sum(a.health for a in red_alive) / (len(red_agents) * 100) if red_alive else 0
            obs[3] = len(red_alive) / len(red_agents)

        # Kill counts (from info)
        info = getattr(self.tactical_env, '_info', {})
        obs[4] = info.get('blue_kills', 0) / 100.0
        obs[5] = info.get('red_kills', 0) / 100.0

        # Game progress
        obs[6] = self._step_count / self._max_steps

        return obs
