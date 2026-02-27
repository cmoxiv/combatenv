"""
TeamWrapper - Assigns agents to teams.

This wrapper takes agents from AgentWrapper and assigns them to teams
(blue, red, etc.). It provides team-based access to agents.

Usage:
    env = GridWorld()
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env, teams=["blue", "red"])  # 100 per team
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..agent import Agent, TeamType


class TeamWrapper(BaseWrapper):
    """
    Wrapper that assigns agents to teams.

    Takes agents from AgentWrapper and divides them equally among teams.
    Provides team-based properties for accessing agents.

    Attributes:
        teams: List of team names
        team_agents: Dict mapping team name to list of agents
    """

    def __init__(
        self,
        env: gym.Env,
        teams: Optional[List[str]] = None,
    ):
        """
        Initialize the TeamWrapper.

        Args:
            env: Environment with AgentWrapper
            teams: List of team names (default: ["blue", "red"])
        """
        super().__init__(env)

        self.teams = teams or ["blue", "red"]
        self.team_agents: Dict[str, List[Agent]] = {team: [] for team in self.teams}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset and assign agents to teams.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        # Reset base environment (which spawns agents)
        obs, info = self.env.reset(seed=seed, options=options)

        # Get agents from AgentWrapper
        agents = getattr(self.env, 'agents', [])

        # Assign agents to teams
        self._assign_teams(agents)

        # Update info with team data
        for team in self.teams:
            info[f"{team}_count"] = len(self.team_agents[team])
            info[f"{team}_alive"] = len([a for a in self.team_agents[team] if a.is_alive])

        return obs, info

    def _assign_teams(self, agents: List[Agent]) -> None:
        """
        Assign agents to teams evenly.

        Args:
            agents: List of all agents
        """
        # Clear previous assignments
        self.team_agents = {team: [] for team in self.teams}

        if not agents:
            return

        # Divide agents among teams
        num_teams = len(self.teams)
        agents_per_team = len(agents) // num_teams

        for i, agent in enumerate(agents):
            team_index = min(i // agents_per_team, num_teams - 1)
            team_name = self.teams[team_index]
            # Cast to TeamType for proper typing
            agent.team = cast(TeamType, team_name)
            self.team_agents[team_name].append(agent)

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
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update team counts in info
        for team in self.teams:
            info[f"{team}_alive"] = len([a for a in self.team_agents[team] if a.is_alive])

        return obs, reward, terminated, truncated, info

    @property
    def blue_agents(self) -> List[Agent]:
        """Get blue team agents."""
        return self.team_agents.get("blue", [])

    @property
    def red_agents(self) -> List[Agent]:
        """Get red team agents."""
        return self.team_agents.get("red", [])

    @property
    def alive_blue_agents(self) -> List[Agent]:
        """Get alive blue team agents."""
        return [a for a in self.blue_agents if a.is_alive]

    @property
    def alive_red_agents(self) -> List[Agent]:
        """Get alive red team agents."""
        return [a for a in self.red_agents if a.is_alive]

    def get_team_agents(self, team: str) -> List[Agent]:
        """
        Get agents for a specific team.

        Args:
            team: Team name

        Returns:
            List of agents on that team
        """
        return self.team_agents.get(team, [])

    def get_alive_team_agents(self, team: str) -> List[Agent]:
        """
        Get alive agents for a specific team.

        Args:
            team: Team name

        Returns:
            List of alive agents on that team
        """
        return [a for a in self.get_team_agents(team) if a.is_alive]

    def get_enemy_agents(self, team: str) -> List[Agent]:
        """
        Get agents from opposing teams.

        Args:
            team: Team name

        Returns:
            List of agents not on that team
        """
        enemies = []
        for other_team in self.teams:
            if other_team != team:
                enemies.extend(self.team_agents.get(other_team, []))
        return enemies

    def get_alive_enemy_agents(self, team: str) -> List[Agent]:
        """
        Get alive agents from opposing teams.

        Args:
            team: Team name

        Returns:
            List of alive enemy agents
        """
        return [a for a in self.get_enemy_agents(team) if a.is_alive]
