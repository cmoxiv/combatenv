"""
CombatWrapper - Shooting mechanics and autonomous combat.

This wrapper adds shooting mechanics to the environment, handling agent combat
including target acquisition, shooting, and kill tracking.

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = TerrainWrapper(env)
    env = ProjectileWrapper(env)
    env = FOVWrapper(env)
    env = CombatWrapper(env, autonomous_combat=True)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..agent import Agent
from ..config import (
    FPS,
    NEAR_FOV_ACCURACY,
    FAR_FOV_ACCURACY,
    MOVEMENT_ACCURACY_PENALTY,
    SHOOT_COOLDOWN,
)


class CombatWrapper(BaseWrapper):
    """
    Wrapper that handles shooting mechanics and combat.

    Provides autonomous combat for AI agents and shoot action processing
    for the controlled agent.

    Attributes:
        autonomous_combat: Whether AI agents shoot automatically
        blue_kills: Number of kills by blue team
        red_kills: Number of kills by red team
        controlled_agent_kills: Kills by the controlled agent
    """

    def __init__(
        self,
        env: gym.Env,
        autonomous_combat: bool = True,
    ):
        """
        Initialize the CombatWrapper.

        Args:
            env: Base environment (should have FOVWrapper, ProjectileWrapper)
            autonomous_combat: Whether AI agents shoot autonomously
        """
        super().__init__(env)

        self.autonomous_combat = autonomous_combat

        # Kill counters
        self.blue_kills = 0
        self.red_kills = 0
        self.controlled_agent_kills = 0
        self.controlled_agent_damage_dealt = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset and clear kill counters.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset kill counters
        self.blue_kills = 0
        self.red_kills = 0
        self.controlled_agent_kills = 0
        self.controlled_agent_damage_dealt = 0.0

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step with combat processing.

        Args:
            action: Action for controlled agent [move_x, move_y, shoot, think]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        dt = 1.0 / FPS

        # Update cooldowns for all agents
        alive_agents = getattr(self.env, 'alive_agents', [])
        for agent in alive_agents:
            agent.update_cooldown(dt)
            agent.update_reload(dt)

        # Step base environment (handles projectile updates)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Process projectile hits for kill tracking
        hits = info.get("projectile_hits", [])
        for projectile, hit_agent in hits:
            if not hit_agent.is_alive:
                # Agent was killed
                if projectile.owner_team == "blue":
                    self.blue_kills += 1
                else:
                    self.red_kills += 1

                # Check if controlled agent got the kill
                controlled_agent = getattr(self.env, 'controlled_agent', None)
                if controlled_agent and projectile.shooter_id == id(controlled_agent):
                    self.controlled_agent_kills += 1

        # Execute autonomous combat if enabled
        if self.autonomous_combat:
            self._execute_autonomous_combat(dt)

        # Add combat stats to info
        info["blue_kills"] = self.blue_kills
        info["red_kills"] = self.red_kills
        info["controlled_agent_kills"] = self.controlled_agent_kills

        return obs, reward, terminated, truncated, info

    def _execute_autonomous_combat(self, dt: float) -> None:
        """
        Execute autonomous shooting for AI agents.

        Args:
            dt: Delta time in seconds
        """
        # Get team agents
        team_agents = getattr(self.env, 'team_agents', {})
        blue_agents = team_agents.get("blue", [])
        red_agents = team_agents.get("red", [])

        # Get controlled agent to skip
        controlled_agent = getattr(self.env, 'controlled_agent', None)

        # Blue agents shoot at red
        for agent in blue_agents:
            if agent is controlled_agent or not agent.is_alive:
                continue
            if agent.can_shoot and not agent.in_water:
                self._try_shoot(agent, red_agents)

        # Red agents shoot at blue
        for agent in red_agents:
            if not agent.is_alive:
                continue
            if agent.can_shoot and not agent.in_water:
                self._try_shoot(agent, blue_agents)

    def _try_shoot(self, agent: Agent, enemy_agents: List[Agent]) -> bool:
        """
        Try to shoot at a visible enemy.

        Args:
            agent: Agent doing the shooting
            enemy_agents: List of potential targets

        Returns:
            True if shot was fired
        """
        # Get FOV wrapper methods
        fov_wrapper = self._find_wrapper("FOVWrapper")
        if fov_wrapper is None:
            return False

        # Get best target
        target, accuracy = fov_wrapper.get_targets_in_fov(agent, enemy_agents)
        if target is None:
            return False

        # Apply movement penalty
        if agent.is_moving:
            accuracy *= MOVEMENT_ACCURACY_PENALTY

        # Create projectile
        projectile_wrapper = self._find_wrapper("ProjectileWrapper")
        if projectile_wrapper is None:
            return False

        projectile_wrapper.create_projectile(
            shooter_position=agent.position,
            shooter_orientation=agent.orientation,
            shooter_team=agent.team,
            shooter_id=id(agent),
            accuracy=accuracy,
        )

        # Consume ammo and start cooldown
        agent.magazine_ammo -= 1
        agent.shoot_cooldown = SHOOT_COOLDOWN

        return True

    def _find_wrapper(self, wrapper_name: str) -> Optional[gym.Wrapper]:
        """
        Find a wrapper in the chain by name.

        Args:
            wrapper_name: Name of wrapper class to find

        Returns:
            Wrapper instance or None
        """
        env = self.env
        while env is not None:
            if type(env).__name__ == wrapper_name:
                return env
            env = getattr(env, 'env', None)
        return None

    def shoot_controlled_agent(
        self,
        accuracy: float = NEAR_FOV_ACCURACY,
    ) -> bool:
        """
        Fire the controlled agent's weapon.

        Args:
            accuracy: Shot accuracy (0.0-1.0)

        Returns:
            True if shot was fired
        """
        controlled_agent = getattr(self.env, 'controlled_agent', None)
        if controlled_agent is None or not controlled_agent.is_alive:
            return False

        if not controlled_agent.can_shoot or controlled_agent.in_water:
            return False

        # Get projectile wrapper
        projectile_wrapper = self._find_wrapper("ProjectileWrapper")
        if projectile_wrapper is None:
            return False

        # Apply movement penalty
        if controlled_agent.is_moving:
            accuracy *= MOVEMENT_ACCURACY_PENALTY

        projectile_wrapper.create_projectile(
            shooter_position=controlled_agent.position,
            shooter_orientation=controlled_agent.orientation,
            shooter_team=controlled_agent.team,
            shooter_id=id(controlled_agent),
            accuracy=accuracy,
        )

        # Consume ammo and start cooldown
        controlled_agent.magazine_ammo -= 1
        controlled_agent.shoot_cooldown = SHOOT_COOLDOWN

        return True
