"""
FOVWrapper - Calculates field of view for agents.

This wrapper adds FOV (field of view) calculation capabilities to the environment,
providing visibility checks and target detection within FOV cones.

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = TerrainWrapper(env)
    env = FOVWrapper(env)
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..agent import Agent
from ..fov import (
    get_fov_cells,
    is_agent_visible_to_agent,
    get_fov_cache,
)
from ..config import (
    NEAR_FOV_RANGE,
    NEAR_FOV_ANGLE,
    FAR_FOV_RANGE,
    FAR_FOV_ANGLE,
    NEAR_FOV_ACCURACY,
    FAR_FOV_ACCURACY,
)


class FOVWrapper(BaseWrapper):
    """
    Wrapper that calculates field of view for agents.

    Provides two-layer FOV system (near and far) with visibility checking
    and target detection.

    Attributes:
        near_range: Near FOV range in cells
        near_angle: Near FOV angle in degrees
        far_range: Far FOV range in cells
        far_angle: Far FOV angle in degrees
    """

    def __init__(
        self,
        env: gym.Env,
        near_range: float = NEAR_FOV_RANGE,
        near_angle: float = NEAR_FOV_ANGLE,
        far_range: float = FAR_FOV_RANGE,
        far_angle: float = FAR_FOV_ANGLE,
    ):
        """
        Initialize the FOVWrapper.

        Args:
            env: Base environment
            near_range: Near FOV range in cells (default 3.0)
            near_angle: Near FOV angle in degrees (default 90)
            far_range: Far FOV range in cells (default 5.0)
            far_angle: Far FOV angle in degrees (default 120)
        """
        super().__init__(env)

        self.near_range = near_range
        self.near_angle = near_angle
        self.far_range = far_range
        self.far_angle = far_angle

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset and clear FOV cache.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Clear FOV cache from previous episode
        get_fov_cache().clear()

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: Action for controlled agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self.env.step(action)

    def get_fov_cells(
        self,
        agent: Agent,
        max_range: Optional[float] = None,
        fov_angle: Optional[float] = None,
    ) -> Set[Tuple[int, int]]:
        """
        Get cells visible to an agent.

        Args:
            agent: Agent to get FOV for
            max_range: Override FOV range (default uses far range)
            fov_angle: Override FOV angle (default uses far angle)

        Returns:
            Set of (x, y) cell coordinates visible to the agent
        """
        terrain_grid = getattr(self.env, 'terrain_grid', None)

        return get_fov_cells(
            agent_pos=agent.position,
            agent_orientation=agent.orientation,
            fov_angle=fov_angle or self.far_angle,
            max_range=max_range or self.far_range,
            terrain_grid=terrain_grid,
        )

    def get_visible_targets(
        self,
        agent: Agent,
        targets: List[Agent],
    ) -> List[Tuple[Agent, str]]:
        """
        Get targets visible to an agent with FOV layer.

        Args:
            agent: Agent doing the looking
            targets: Potential targets

        Returns:
            List of (target_agent, fov_layer) tuples
            where fov_layer is "near" or "far"
        """
        visible = []
        terrain_grid = getattr(self.env, 'terrain_grid', None)

        for target in targets:
            if not target.is_alive or target is agent:
                continue

            # Check near FOV first
            if is_agent_visible_to_agent(
                observer=agent,
                target=target,
                max_range=self.near_range,
                fov_angle=self.near_angle,
                terrain_grid=terrain_grid,
            ):
                visible.append((target, "near"))
            # Check far FOV
            elif is_agent_visible_to_agent(
                observer=agent,
                target=target,
                max_range=self.far_range,
                fov_angle=self.far_angle,
                terrain_grid=terrain_grid,
            ):
                visible.append((target, "far"))

        return visible

    def get_targets_in_fov(
        self,
        agent: Agent,
        enemy_agents: List[Agent],
    ) -> Tuple[Optional[Agent], float]:
        """
        Get best target in FOV with accuracy.

        Args:
            agent: Agent looking for targets
            enemy_agents: List of enemy agents

        Returns:
            Tuple of (target_agent, accuracy) or (None, 0.0) if no target
        """
        visible = self.get_visible_targets(agent, enemy_agents)

        if not visible:
            return None, 0.0

        # Prioritize near FOV targets
        near_targets = [t for t, layer in visible if layer == "near"]
        far_targets = [t for t, layer in visible if layer == "far"]

        if near_targets:
            return near_targets[0], NEAR_FOV_ACCURACY
        elif far_targets:
            return far_targets[0], FAR_FOV_ACCURACY

        return None, 0.0
