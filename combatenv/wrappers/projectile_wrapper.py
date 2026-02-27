"""
ProjectileWrapper - Manages projectile physics and collision.

This wrapper adds projectile system to the environment, handling projectile
creation, movement, collision detection with agents and terrain.

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = TerrainWrapper(env)
    env = ProjectileWrapper(env)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..projectile import Projectile, create_projectile
from ..config import (
    FPS,
    MUZZLE_FLASH_LIFETIME,
    MUZZLE_FLASH_OFFSET,
)


class ProjectileWrapper(BaseWrapper):
    """
    Wrapper that manages projectiles and muzzle flashes.

    Handles projectile creation, movement, collision detection with agents
    and terrain, and muzzle flash effects.

    Attributes:
        projectiles: List of active projectiles
        muzzle_flashes: List of (position, lifetime) tuples for visual effects
    """

    def __init__(self, env: gym.Env, friendly_fire: bool = True):
        """
        Initialize the ProjectileWrapper.

        Args:
            env: Base environment (should have TerrainWrapper)
            friendly_fire: If True, projectiles can hit teammates
        """
        super().__init__(env)

        self.friendly_fire = friendly_fire
        self.projectiles: List[Projectile] = []
        self.muzzle_flashes: List[Tuple[Tuple[float, float], float]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset and clear all projectiles.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Clear projectiles and muzzle flashes
        self.projectiles = []
        self.muzzle_flashes = []

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step and update projectiles.

        Args:
            action: Action for controlled agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update projectiles
        dt = 1.0 / FPS
        hits = self._update_projectiles(dt)

        # Update muzzle flashes
        self._update_muzzle_flashes(dt)

        # Add hit info
        info["projectile_hits"] = hits

        return obs, reward, terminated, truncated, info

    def _update_projectiles(self, dt: float) -> List[Tuple[Projectile, "Agent"]]:
        """
        Update all projectiles and check for collisions.

        Args:
            dt: Delta time in seconds

        Returns:
            List of (projectile, hit_agent) tuples
        """
        # Get terrain grid for building collision
        terrain_grid = getattr(self.env, 'terrain_grid', None)

        # Get alive agents for collision detection
        alive_agents = getattr(self.env, 'alive_agents', [])

        hits = []
        surviving_projectiles = []

        for projectile in self.projectiles:
            # Check collision with agents first
            hit_agent = None
            for agent in alive_agents:
                # Skip same-team agents when friendly fire is disabled
                if not self.friendly_fire and agent.team == projectile.owner_team:
                    continue
                if projectile.check_collision(agent):
                    hit_agent = agent
                    break

            if hit_agent:
                # Apply damage
                old_health = hit_agent.health
                hit_agent.health -= projectile.damage
                if hit_agent.health <= 0:
                    hit_agent.health = 0
                    if old_health > 0:
                        print(f"DEBUG: Agent died from projectile at {hit_agent.position}")
                hits.append((projectile, hit_agent))
            else:
                # Update position and check expiration
                expired = projectile.update(dt, terrain_grid)
                if not expired:
                    surviving_projectiles.append(projectile)

        self.projectiles = surviving_projectiles
        return hits

    def _update_muzzle_flashes(self, dt: float) -> None:
        """
        Update muzzle flash lifetimes.

        Args:
            dt: Delta time in seconds
        """
        self.muzzle_flashes = [
            (pos, lifetime - dt)
            for pos, lifetime in self.muzzle_flashes
            if lifetime - dt > 0
        ]

    def create_projectile(
        self,
        shooter_position: Tuple[float, float],
        shooter_orientation: float,
        shooter_team: str,
        shooter_id: int,
        accuracy: float = 1.0,
    ) -> Projectile:
        """
        Create and register a new projectile.

        Args:
            shooter_position: Position of the shooter
            shooter_orientation: Orientation of the shooter in degrees
            shooter_team: Team of the shooter
            shooter_id: ID of the shooter (to avoid self-hit)
            accuracy: Accuracy modifier (0.0-1.0)

        Returns:
            The created Projectile
        """
        import math

        projectile = create_projectile(
            shooter_position=shooter_position,
            shooter_orientation=shooter_orientation,
            shooter_team=shooter_team,
            shooter_id=shooter_id,
            accuracy=accuracy,
        )

        self.projectiles.append(projectile)

        # Add muzzle flash
        rad = math.radians(shooter_orientation)
        flash_x = shooter_position[0] + math.cos(rad) * MUZZLE_FLASH_OFFSET
        flash_y = shooter_position[1] + math.sin(rad) * MUZZLE_FLASH_OFFSET
        self.muzzle_flashes.append(((flash_x, flash_y), MUZZLE_FLASH_LIFETIME))

        return projectile

    def get_projectile_count(self) -> int:
        """Get number of active projectiles."""
        return len(self.projectiles)

    def get_muzzle_flash_count(self) -> int:
        """Get number of active muzzle flashes."""
        return len(self.muzzle_flashes)
