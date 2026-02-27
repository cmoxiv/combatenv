"""
Unit wrapper for tracking unit membership and applying boids behavior.

This wrapper:
1. Creates and tracks Unit instances for agents
2. Applies boids flocking forces to agent movement
3. Extends observations with unit-related information
4. Provides unit data for strategic-level coordination

Usage:
    from rl_student.wrappers import MultiAgentWrapper, UnitWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = UnitWrapper(env)  # Add after MultiAgentWrapper
    env = CohesionRewardWrapper(env)  # Can now access units
"""

from typing import Dict, Tuple, List, Optional, Any
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from combatenv.unit import (
    Unit,
    spawn_units_for_team,
    get_all_agents_from_units,
    get_unit_for_agent,
)
from combatenv.config import (
    NUM_UNITS_PER_TEAM,
    AGENTS_PER_UNIT,
    UNIT_COHESION_RADIUS,
    GRID_SIZE,
)
from .base_wrapper import BaseWrapper


# Number of additional observation values for unit info
UNIT_OBS_SIZE = 12


class UnitWrapper(BaseWrapper):
    """
    Wrapper that adds unit awareness to the multi-agent environment.

    Features:
        - Creates units from spawned agents at reset
        - Tracks unit membership via agent.unit_id
        - Optionally applies boids forces to wandering agents
        - Extends observations with unit information (optional)

    Attributes:
        blue_units: List of blue team Unit instances
        red_units: List of red team Unit instances
        enable_boids: Whether to apply boids forces
        extend_obs: Whether to add unit info to observations
    """

    def __init__(
        self,
        env,
        num_units_per_team: int = NUM_UNITS_PER_TEAM,
        agents_per_unit: int = AGENTS_PER_UNIT,
        enable_boids: bool = True,
        extend_obs: bool = True
    ):
        """
        Initialize the unit wrapper.

        Args:
            env: Environment (should be MultiAgentWrapper)
            num_units_per_team: Number of units per team
            agents_per_unit: Agents per unit
            enable_boids: Whether to apply boids forces
            extend_obs: Whether to extend observations with unit info
        """
        super().__init__(env)

        self.num_units_per_team = num_units_per_team
        self.agents_per_unit = agents_per_unit
        self.enable_boids = enable_boids
        self.extend_obs = extend_obs

        self.blue_units: List[Unit] = []
        self.red_units: List[Unit] = []

        # Extended observation space if enabled
        if extend_obs:
            # Get base observation size
            base_obs_space = env.observation_space
            if isinstance(base_obs_space, spaces.Dict):
                # Multi-agent: get single agent obs size
                sample_space = list(base_obs_space.spaces.values())[0]
                base_size = sample_space.shape[0]
            elif isinstance(base_obs_space, spaces.Box):
                base_size = base_obs_space.shape[0]
            else:
                base_size = 88  # Default

            new_size = base_size + UNIT_OBS_SIZE

            # Create new observation space
            new_spaces = {}
            for key in range(200):  # 200 agents
                new_spaces[key] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(new_size,),
                    dtype=np.float32
                )
            self.observation_space = spaces.Dict(new_spaces)

        print(f"UnitWrapper: {num_units_per_team} units per team, {agents_per_unit} agents each")
        print(f"  Boids: {'enabled' if enable_boids else 'disabled'}")
        print(f"  Extended obs: {'enabled' if extend_obs else 'disabled'}")

    def reset(self, **kwargs) -> Tuple[Dict[int, np.ndarray], Dict]:
        """Reset environment and create units."""
        obs_dict, info = self.env.reset(**kwargs)

        # Create units from existing agents
        self._create_units()

        # Extend observations if enabled
        if self.extend_obs:
            obs_dict = self._extend_observations(obs_dict)

        # Add unit info to info dict
        info['blue_units'] = len(self.blue_units)
        info['red_units'] = len(self.red_units)

        return obs_dict, info

    def step(self, actions) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict]:
        """Step environment with optional boids influence."""
        # Apply boids influence before step if enabled
        if self.enable_boids:
            self._apply_boids_influence()

        # Step the environment
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)

        # Extend observations if enabled
        if self.extend_obs:
            obs_dict = self._extend_observations(obs_dict)

        # Add unit info to info dict
        info['blue_units_alive'] = sum(1 for u in self.blue_units if not u.is_eliminated)
        info['red_units_alive'] = sum(1 for u in self.red_units if not u.is_eliminated)

        return obs_dict, rewards, terminated, truncated, info

    def _create_units(self) -> None:
        """Create units from existing agents after environment reset."""
        # Get agent lists from wrapped env
        blue_agents = getattr(self.env, 'blue_agents', None)
        red_agents = getattr(self.env, 'red_agents', None)

        if blue_agents is None or red_agents is None:
            # Try to get from unwrapped env
            unwrapped = self.env.unwrapped
            blue_agents = getattr(unwrapped, 'blue_agents', [])
            red_agents = getattr(unwrapped, 'red_agents', [])

        # Create blue units
        self.blue_units = self._partition_agents_into_units(
            blue_agents, "blue", 0
        )

        # Create red units
        self.red_units = self._partition_agents_into_units(
            red_agents, "red", self.num_units_per_team
        )

    def _partition_agents_into_units(
        self,
        agents: List,
        team: str,
        start_id: int
    ) -> List[Unit]:
        """
        Partition a list of agents into units.

        Uses spatial clustering based on agent positions to create
        cohesive units.

        Args:
            agents: List of Agent instances
            team: Team identifier
            start_id: Starting unit ID

        Returns:
            List of Unit instances
        """
        if not agents:
            return []

        # Simple partitioning: divide agents evenly
        num_agents = len(agents)
        agents_per_unit = max(1, num_agents // self.num_units_per_team)

        units = []
        for i in range(self.num_units_per_team):
            start_idx = i * agents_per_unit
            if i == self.num_units_per_team - 1:
                # Last unit gets remaining agents
                end_idx = num_agents
            else:
                end_idx = start_idx + agents_per_unit

            unit_agents = agents[start_idx:end_idx]
            if not unit_agents:
                continue

            unit_id = start_id + i

            # Assign unit_id to agents
            for agent in unit_agents:
                agent.unit_id = unit_id
                agent.following_unit = True

            unit = Unit(
                id=unit_id,
                team=team,
                agents=list(unit_agents)
            )
            units.append(unit)

        return units

    def _apply_boids_influence(self) -> None:
        """Apply boids steering to agent following_unit flags and orientations."""
        # Boids influence is applied through wander_with_boids in agent
        # This method could be used for additional unit-level behaviors
        pass

    def _extend_observations(
        self,
        obs_dict: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Extend observations with unit information.

        Adds 12 floats to each observation:
            0-1: Unit centroid relative X, Y
            2: Distance to centroid (normalized)
            3: Unit average heading (normalized)
            4: Unit alive count (normalized)
            5-6: Waypoint relative X, Y (0.5 if no waypoint)
            7: Distance to waypoint (normalized)
            8: Cohesion score
            9: Formation state (0=cohesive, 0.5=scattered, 1=broken)
            10: Nearby squadmates count (normalized)
            11: Is following unit (0 or 1)

        Args:
            obs_dict: Original observations

        Returns:
            Extended observations
        """
        extended_obs = {}

        for agent_idx, obs in obs_dict.items():
            unit_info = self._get_unit_observation(agent_idx)
            extended = np.concatenate([obs, unit_info])
            extended_obs[agent_idx] = extended

        return extended_obs

    def _get_unit_observation(self, agent_idx: int) -> np.ndarray:
        """
        Get unit-related observation values for an agent.

        Args:
            agent_idx: Agent index

        Returns:
            Array of 12 unit observation values
        """
        # Default values for agents without units
        default_obs = np.array([
            0.5, 0.5,  # centroid relative
            0.0,       # distance to centroid
            0.0,       # unit heading
            0.0,       # alive count
            0.5, 0.5,  # waypoint relative
            0.0,       # distance to waypoint
            0.0,       # cohesion score
            0.5,       # formation state
            0.0,       # nearby squadmates
            0.0,       # following unit
        ], dtype=np.float32)

        # Get agent and unit
        agent, unit = self._get_agent_and_unit(agent_idx)
        if agent is None or unit is None:
            return default_obs

        # Dead agents get default obs
        if not agent.is_alive:
            return default_obs

        # Calculate unit observations
        centroid = unit.centroid
        agent_pos = agent.position

        # Relative centroid position (0-1)
        rel_centroid_x = (centroid[0] - agent_pos[0]) / GRID_SIZE + 0.5
        rel_centroid_y = (centroid[1] - agent_pos[1]) / GRID_SIZE + 0.5

        # Distance to centroid (normalized by grid size)
        dist_to_centroid = math.sqrt(
            (agent_pos[0] - centroid[0])**2 +
            (agent_pos[1] - centroid[1])**2
        )
        norm_dist_centroid = min(1.0, dist_to_centroid / GRID_SIZE)

        # Unit average heading (normalized 0-1)
        norm_heading = unit.average_heading / 360.0

        # Alive count (normalized by original unit size)
        norm_alive = unit.alive_count / len(unit.agents)

        # Waypoint info
        if unit.waypoint is not None:
            rel_waypoint_x = (unit.waypoint[0] - agent_pos[0]) / GRID_SIZE + 0.5
            rel_waypoint_y = (unit.waypoint[1] - agent_pos[1]) / GRID_SIZE + 0.5
            dist_to_waypoint = math.sqrt(
                (agent_pos[0] - unit.waypoint[0])**2 +
                (agent_pos[1] - unit.waypoint[1])**2
            )
            norm_dist_waypoint = min(1.0, dist_to_waypoint / GRID_SIZE)
        else:
            rel_waypoint_x = 0.5
            rel_waypoint_y = 0.5
            norm_dist_waypoint = 0.0

        # Cohesion score (0-1)
        cohesion_score = unit.get_cohesion_score(agent)

        # Formation state based on spread
        spread = unit.get_formation_spread()
        if spread < 2.0:
            formation_state = 0.0  # Cohesive
        elif spread < UNIT_COHESION_RADIUS:
            formation_state = 0.5  # Scattered
        else:
            formation_state = 1.0  # Broken

        # Nearby squadmates (normalized)
        nearby = len(unit.get_nearby_squadmates(agent, radius=3.0))
        max_nearby = self.agents_per_unit - 1
        norm_nearby = nearby / max(1, max_nearby)

        # Following unit flag
        following = 1.0 if agent.following_unit else 0.0

        return np.array([
            np.clip(rel_centroid_x, 0, 1),
            np.clip(rel_centroid_y, 0, 1),
            norm_dist_centroid,
            norm_heading,
            norm_alive,
            np.clip(rel_waypoint_x, 0, 1),
            np.clip(rel_waypoint_y, 0, 1),
            norm_dist_waypoint,
            cohesion_score,
            formation_state,
            norm_nearby,
            following,
        ], dtype=np.float32)

    def _get_agent_and_unit(self, agent_idx: int) -> Tuple[Optional[Any], Optional[Unit]]:
        """
        Get agent and their unit from agent index.

        Args:
            agent_idx: Agent index (0-99 blue, 100-199 red)

        Returns:
            Tuple of (agent, unit) or (None, None)
        """
        # Get agent lists
        blue_agents = getattr(self.env, 'blue_agents', None)
        red_agents = getattr(self.env, 'red_agents', None)

        if blue_agents is None or red_agents is None:
            unwrapped = self.env.unwrapped
            blue_agents = getattr(unwrapped, 'blue_agents', [])
            red_agents = getattr(unwrapped, 'red_agents', [])

        # Get agent
        if agent_idx < 100:
            if agent_idx < len(blue_agents):
                agent = blue_agents[agent_idx]
                units = self.blue_units
            else:
                return None, None
        else:
            local_idx = agent_idx - 100
            if local_idx < len(red_agents):
                agent = red_agents[local_idx]
                units = self.red_units
            else:
                return None, None

        # Find unit
        unit = get_unit_for_agent(agent, units)
        return agent, unit

    def set_waypoint(self, unit_id: int, x: float, y: float) -> bool:
        """
        Set a waypoint for a specific unit.

        Args:
            unit_id: Unit ID
            x: Waypoint X coordinate
            y: Waypoint Y coordinate

        Returns:
            True if waypoint was set, False if unit not found
        """
        for unit in self.blue_units + self.red_units:
            if unit.id == unit_id:
                unit.set_waypoint(x, y)
                return True
        return False

    def clear_waypoint(self, unit_id: int) -> bool:
        """
        Clear waypoint for a specific unit.

        Args:
            unit_id: Unit ID

        Returns:
            True if cleared, False if unit not found
        """
        for unit in self.blue_units + self.red_units:
            if unit.id == unit_id:
                unit.clear_waypoint()
                return True
        return False

    def get_unit_centroids(self, team: str = None) -> Dict[int, Tuple[float, float]]:
        """
        Get centroids for all units.

        Args:
            team: Optional team filter ("blue" or "red")

        Returns:
            Dict mapping unit_id to centroid position
        """
        centroids = {}

        if team is None or team == "blue":
            for unit in self.blue_units:
                centroids[unit.id] = unit.centroid

        if team is None or team == "red":
            for unit in self.red_units:
                centroids[unit.id] = unit.centroid

        return centroids
