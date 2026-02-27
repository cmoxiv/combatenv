"""
MovementWrapper - Agent movement and wandering AI.

This wrapper handles agent movement, including processing movement actions
for the controlled agent and autonomous wandering for AI agents.

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = TerrainWrapper(env)
    env = MovementWrapper(env, enable_wandering=True)
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..agent import Agent
from ..config import (
    FPS,
    AGENT_MOVE_SPEED,
    AGENT_ROTATION_SPEED,
)


class MovementWrapper(BaseWrapper):
    """
    Wrapper that handles agent movement.

    Processes movement actions for the controlled agent and provides
    autonomous wandering for AI agents.

    Attributes:
        enable_wandering: Whether AI agents wander autonomously
    """

    def __init__(
        self,
        env: gym.Env,
        enable_wandering: bool = True,
    ):
        """
        Initialize the MovementWrapper.

        Args:
            env: Base environment
            enable_wandering: Whether AI agents should wander
        """
        super().__init__(env)

        self.enable_wandering = enable_wandering

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with movement processing.

        Args:
            action: Action for controlled agent [move_x, move_y, shoot, think]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        dt = 1.0 / FPS

        # Apply movement to controlled agent (only if action is provided)
        controlled_agent = getattr(self.env, 'controlled_agent', None)
        if action is not None and controlled_agent and controlled_agent.is_alive:
            self._apply_movement_action(controlled_agent, action, dt)

        # Update stamina for all agents
        alive_agents = getattr(self.env, 'alive_agents', [])
        for agent in alive_agents:
            agent.update_stamina(dt, agent.is_moving)

        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Autonomous wandering for AI agents
        if self.enable_wandering:
            self._execute_wandering(dt)

        return obs, reward, terminated, truncated, info

    def _apply_movement_action(
        self,
        agent: Agent,
        action: np.ndarray,
        dt: float,
    ) -> None:
        """
        Apply movement action to an agent.

        Args:
            agent: Agent to move
            action: Action array [move_x, move_y, shoot, think]
            dt: Delta time in seconds
        """
        move_x = float(action[0])
        move_y = float(action[1])

        # Get terrain grid for collision
        terrain_grid = getattr(self.env, 'terrain_grid', None)

        # Get other agents for collision
        alive_agents = getattr(self.env, 'alive_agents', [])

        # Check if there's significant movement input
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            # Calculate target orientation from input
            target_orientation = math.degrees(math.atan2(move_y, move_x))
            target_orientation = target_orientation % 360

            # Smoothly rotate toward target orientation
            angle_diff = (target_orientation - agent.orientation + 180) % 360 - 180

            # Rotate towards target
            max_rotation = AGENT_ROTATION_SPEED * dt
            if abs(angle_diff) <= max_rotation:
                agent.orientation = target_orientation
            else:
                agent.orientation += max_rotation if angle_diff > 0 else -max_rotation
            agent.orientation = agent.orientation % 360

            # Move forward at speed proportional to input magnitude
            magnitude = min(1.0, math.sqrt(move_x * move_x + move_y * move_y))
            agent.move_forward(
                speed=AGENT_MOVE_SPEED * magnitude,
                dt=dt,
                other_agents=alive_agents,
                terrain_grid=terrain_grid,
            )
        else:
            # Agent not moving - reset wander_direction so is_moving returns False
            agent.wander_direction = 0

    def _execute_wandering(self, dt: float) -> None:
        """
        Execute autonomous wandering for AI agents.

        Args:
            dt: Delta time in seconds
        """
        # Get agents
        alive_agents = getattr(self.env, 'alive_agents', [])
        controlled_agent = getattr(self.env, 'controlled_agent', None)
        terrain_grid = getattr(self.env, 'terrain_grid', None)

        for agent in alive_agents:
            # Skip controlled agent
            if agent is controlled_agent:
                continue

            # Skip stuck agents
            if agent.is_stuck:
                continue

            # Wander
            agent.wander(
                dt=dt,
                other_agents=alive_agents,
                terrain_grid=terrain_grid,
            )

    def move_agent(
        self,
        agent: Agent,
        direction: Tuple[float, float],
        dt: float,
    ) -> bool:
        """
        Move an agent in a direction.

        Args:
            agent: Agent to move
            direction: (dx, dy) direction vector
            dt: Delta time

        Returns:
            True if movement succeeded
        """
        terrain_grid = getattr(self.env, 'terrain_grid', None)
        alive_agents = getattr(self.env, 'alive_agents', [])

        # Calculate orientation from direction
        dx, dy = direction
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            agent.orientation = math.degrees(math.atan2(dy, dx)) % 360
            return agent.move_forward(
                speed=AGENT_MOVE_SPEED,
                dt=dt,
                other_agents=alive_agents,
                terrain_grid=terrain_grid,
            )
        return False
