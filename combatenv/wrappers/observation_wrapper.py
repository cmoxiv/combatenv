"""
ObservationWrapper - Generates the observation vector for RL.

This wrapper generates the 89-float observation vector for the controlled agent,
including agent state, nearby enemies/allies, terrain, and waypoint distance.

Observation Format (89 floats):
    - Indices 0-9: Agent state (position, orientation, health, stamina, armor, ammo, etc.)
    - Indices 10-29: 5 nearest enemies (4 floats each: rel_x, rel_y, health, distance)
    - Indices 30-49: 5 nearest allies (4 floats each: rel_x, rel_y, health, distance)
    - Indices 50-87: Terrain types for up to 38 FOV cells (normalized 0-1)
    - Index 88: Chebyshev distance to waypoint (normalized 0-1)

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = TerrainWrapper(env)
    env = ObservationWrapper(env)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..agent import Agent
from ..fov import get_fov_cells
from ..terrain import TerrainType
from ..config import (
    GRID_SIZE,
    AGENT_MAX_HEALTH,
    AGENT_MAX_STAMINA,
    AGENT_MAX_ARMOR,
    AGENT_MAX_AMMO,
    FAR_FOV_RANGE,
    FAR_FOV_ANGLE,
    OBS_SIZE,
)


# Observation constants
NUM_NEARBY_ENEMIES = 5
NUM_NEARBY_ALLIES = 5
MAX_FOV_CELLS = 38
NUM_TERRAIN_TYPES = 5


class ObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that generates the 89-float observation vector.

    Transforms the environment observation into a normalized array containing
    agent state, nearby agents, visible terrain, and waypoint distance.

    Attributes:
        observation_space: Box(89,) with values in [0, 1]
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the ObservationWrapper.

        Args:
            env: Base environment
        """
        super().__init__(env)

        # Define observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped environment."""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.env, name)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Transform observation to 89-float vector.

        Args:
            observation: Original observation (ignored, we compute fresh)

        Returns:
            89-float normalized observation
        """
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """
        Generate the full observation vector.

        Returns:
            Normalized numpy array of shape (OBS_SIZE,)
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # Get controlled agent
        controlled_agent = getattr(self.env, 'controlled_agent', None)
        if controlled_agent is None or not controlled_agent.is_alive:
            return obs

        agent = controlled_agent
        grid_size = getattr(self.env, 'grid_size', GRID_SIZE)

        # Agent state (indices 0-9)
        obs[0] = np.clip(agent.position[0] / grid_size, 0, 1)
        obs[1] = np.clip(agent.position[1] / grid_size, 0, 1)
        obs[2] = np.clip((agent.orientation % 360) / 360.0, 0, 1)
        obs[3] = np.clip(agent.health / AGENT_MAX_HEALTH, 0, 1)
        obs[4] = np.clip(agent.stamina / AGENT_MAX_STAMINA, 0, 1)
        obs[5] = np.clip(agent.armor / AGENT_MAX_ARMOR, 0, 1)
        obs[6] = np.clip(agent.ammo_reserve / AGENT_MAX_AMMO, 0, 1)
        obs[7] = np.clip(agent.magazine_ammo / 30.0, 0, 1)
        obs[8] = 1.0 if agent.can_shoot else 0.0
        obs[9] = 1.0 if agent.is_reloading else 0.0

        # Get team agents
        team_agents = getattr(self.env, 'team_agents', {})
        blue_agents = team_agents.get("blue", [])
        red_agents = team_agents.get("red", [])

        # Determine which are enemies/allies based on agent's team
        if agent.team == "blue":
            enemy_agents = red_agents
            ally_agents = [a for a in blue_agents if a is not agent]
        else:
            enemy_agents = blue_agents
            ally_agents = [a for a in red_agents if a is not agent]

        # Get nearby enemies (indices 10-29)
        enemies = [(e, self._distance_to(agent, e)) for e in enemy_agents if e.is_alive]
        enemies.sort(key=lambda x: x[1])

        for i, (enemy, dist) in enumerate(enemies[:NUM_NEARBY_ENEMIES]):
            base_idx = 10 + i * 4
            rel_x = (enemy.position[0] - agent.position[0]) / grid_size + 0.5
            rel_y = (enemy.position[1] - agent.position[1]) / grid_size + 0.5
            obs[base_idx] = np.clip(rel_x, 0, 1)
            obs[base_idx + 1] = np.clip(rel_y, 0, 1)
            obs[base_idx + 2] = np.clip(enemy.health / AGENT_MAX_HEALTH, 0, 1)
            obs[base_idx + 3] = np.clip(dist / grid_size, 0, 1)

        # Get nearby allies (indices 30-49)
        allies = [(a, self._distance_to(agent, a)) for a in ally_agents if a.is_alive]
        allies.sort(key=lambda x: x[1])

        for i, (ally, dist) in enumerate(allies[:NUM_NEARBY_ALLIES]):
            base_idx = 30 + i * 4
            rel_x = (ally.position[0] - agent.position[0]) / grid_size + 0.5
            rel_y = (ally.position[1] - agent.position[1]) / grid_size + 0.5
            obs[base_idx] = np.clip(rel_x, 0, 1)
            obs[base_idx + 1] = np.clip(rel_y, 0, 1)
            obs[base_idx + 2] = np.clip(ally.health / AGENT_MAX_HEALTH, 0, 1)
            obs[base_idx + 3] = np.clip(dist / grid_size, 0, 1)

        # Get terrain types for FOV cells (indices 50-87)
        terrain_grid = getattr(self.env, 'terrain_grid', None)
        if terrain_grid is not None:
            fov_cells = get_fov_cells(
                agent.position,
                agent.orientation,
                fov_angle=FAR_FOV_ANGLE,
                max_range=FAR_FOV_RANGE,
                terrain_grid=terrain_grid
            )

            # Sort cells by distance
            agent_x, agent_y = agent.position
            sorted_cells = sorted(
                fov_cells,
                key=lambda c: (c[0] - agent_x) ** 2 + (c[1] - agent_y) ** 2
            )

            # Add terrain type for each cell
            for i, (cx, cy) in enumerate(sorted_cells[:MAX_FOV_CELLS]):
                terrain_type = terrain_grid.get(cx, cy)
                obs[50 + i] = terrain_type.value / (NUM_TERRAIN_TYPES - 1)

        # Chebyshev distance to waypoint (index 88)
        obs[88] = self._get_chebyshev_to_waypoint(agent)

        return obs

    def _distance_to(self, agent1: Agent, agent2: Agent) -> float:
        """Calculate Euclidean distance between two agents."""
        dx = agent1.position[0] - agent2.position[0]
        dy = agent1.position[1] - agent2.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def _get_chebyshev_to_waypoint(self, agent: Agent) -> float:
        """
        Get normalized Chebyshev distance to agent's unit waypoint.

        Returns:
            Normalized distance (0 = at waypoint, 1 = max distance)
        """
        # Get unit waypoint if agent is in a unit
        from ..unit import get_unit_for_agent

        # Get units from environment
        blue_units = getattr(self.env, 'blue_units', [])
        red_units = getattr(self.env, 'red_units', [])
        all_units = blue_units + red_units

        unit = get_unit_for_agent(agent, all_units)
        if unit is None or unit.waypoint is None:
            return 1.0  # No waypoint = max distance

        # Calculate Chebyshev distance (max of |dx|, |dy|)
        dx = abs(agent.position[0] - unit.waypoint[0])
        dy = abs(agent.position[1] - unit.waypoint[1])
        chebyshev_dist = max(dx, dy)

        # Normalize by max possible distance (7 cells for operational grid)
        from ..config import OPERATIONAL_GRID_SIZE
        max_dist = (OPERATIONAL_GRID_SIZE - 1) * 8  # Max cells
        return min(chebyshev_dist / max_dist, 1.0)
