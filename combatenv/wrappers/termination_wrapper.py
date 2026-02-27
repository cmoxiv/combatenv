"""
TerminationWrapper - Episode termination conditions.

This wrapper handles episode termination and truncation conditions,
including controlled agent death, team elimination, and step limits.

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = TerminationWrapper(env, max_steps=1000)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper


class TerminationWrapper(BaseWrapper):
    """
    Wrapper that handles episode termination conditions.

    Checks for termination (controlled agent death, team elimination)
    and truncation (step limit).

    Attributes:
        max_steps: Maximum steps before truncation (None = no limit)
        terminate_on_controlled_death: Terminate if controlled agent dies
        terminate_on_team_elimination: Terminate if either team is eliminated
        step_count: Current step count
    """

    def __init__(
        self,
        env: gym.Env,
        max_steps: Optional[int] = 1000,
        terminate_on_controlled_death: bool = True,
        terminate_on_team_elimination: bool = True,
    ):
        """
        Initialize the TerminationWrapper.

        Args:
            env: Base environment
            max_steps: Maximum steps before truncation (None for no limit)
            terminate_on_controlled_death: Terminate if controlled agent dies
            terminate_on_team_elimination: Terminate if a team is eliminated
        """
        super().__init__(env)

        self.max_steps = max_steps
        self.terminate_on_controlled_death = terminate_on_controlled_death
        self.terminate_on_team_elimination = terminate_on_team_elimination
        self.step_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset and clear step counter.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset step counter
        self.step_count = 0

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step and check termination conditions.

        Args:
            action: Action for controlled agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Increment step counter
        self.step_count += 1

        # Check termination conditions
        terminated = terminated or self._check_terminated()
        truncated = truncated or self._check_truncated()

        # Add step count to info
        info["step_count"] = self.step_count

        return obs, reward, terminated, truncated, info

    def _check_terminated(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if episode should terminate
        """
        # Terminate if controlled agent dies
        if self.terminate_on_controlled_death:
            controlled_agent = getattr(self.env, 'controlled_agent', None)
            if controlled_agent and not controlled_agent.is_alive:
                return True

        # Terminate if either team is eliminated
        if self.terminate_on_team_elimination:
            team_agents = getattr(self.env, 'team_agents', {})
            blue_agents = team_agents.get("blue", [])
            red_agents = team_agents.get("red", [])

            living_blue = sum(1 for a in blue_agents if a.is_alive)
            living_red = sum(1 for a in red_agents if a.is_alive)

            if living_blue == 0 or living_red == 0:
                return True

        return False

    def _check_truncated(self) -> bool:
        """
        Check if episode should be truncated.

        Returns:
            True if step limit reached
        """
        if self.max_steps is None:
            return False
        return self.step_count >= self.max_steps
