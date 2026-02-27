"""
Tactical reward wrapper for Phase 1 movement/survival training.

This wrapper shapes rewards to encourage agents to:
- Maintain health and armor
- Manage stamina effectively
- Stay in formation (within unit cohesion radius)
- Avoid hazardous terrain (fire, forest)

Usage:
    from combatenv.wrappers import TacticalRewardWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = TacticalRewardWrapper(env)

    # Rewards are automatically shaped based on agent state
"""

import math
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym

from combatenv.config import (
    AGENT_MAX_HEALTH,
    AGENT_MAX_ARMOR,
    AGENT_MAX_STAMINA,
    UNIT_COHESION_RADIUS,
)
from combatenv.terrain import TerrainType


class TacticalRewardWrapper(gym.Wrapper):
    """
    Reward shaping wrapper for Phase 1 tactical movement training.

    Shapes rewards to encourage survival and formation behavior
    without combat rewards.

    Reward Signals:
        - Health maintained: +0.01/step (bonus for staying above 50% HP)
        - Armor preserved: +0.005/step (bonus for keeping armor intact)
        - Stamina managed: +0.005/step (bonus for not depleting stamina)
        - Formation cohesion: +0.02/step (within UNIT_COHESION_RADIUS of centroid)
        - Hazard avoidance: +0.03/step (not standing on fire/swamp)
        - Death penalty: -1.0 (agent eliminated)
        - Fire damage: -0.1/step (standing on fire)
        - Forest stuck: -0.05/step (movement impaired)

    Attributes:
        health_threshold: Health threshold for bonus (default: 0.5)
        armor_threshold: Armor threshold for bonus (default: 0.5)
        stamina_threshold: Stamina threshold for bonus (default: 0.3)
    """

    # Reward values (can be tuned)
    HEALTH_BONUS = 0.01
    ARMOR_BONUS = 0.005
    STAMINA_BONUS = 0.005
    FORMATION_BONUS = 0.02
    HAZARD_AVOIDANCE_BONUS = 0.03
    DEATH_PENALTY = -1.0
    FIRE_PENALTY = -0.1
    FOREST_PENALTY = -0.05

    # Thresholds
    HEALTH_THRESHOLD = 0.5
    ARMOR_THRESHOLD = 0.5
    STAMINA_THRESHOLD = 0.3

    def __init__(
        self,
        env: gym.Env,
        health_bonus: float = 0.01,
        armor_bonus: float = 0.005,
        stamina_bonus: float = 0.005,
        formation_bonus: float = 0.02,
        hazard_avoidance_bonus: float = 0.03,
        death_penalty: float = -1.0,
        fire_penalty: float = -0.1,
        forest_penalty: float = -0.05,
    ):
        """
        Initialize the tactical reward wrapper.

        Args:
            env: Environment (should be MultiAgentWrapper)
            health_bonus: Bonus for maintaining health above threshold
            armor_bonus: Bonus for maintaining armor above threshold
            stamina_bonus: Bonus for maintaining stamina above threshold
            formation_bonus: Bonus for staying in formation
            hazard_avoidance_bonus: Bonus for avoiding hazards
            death_penalty: Penalty for agent death
            fire_penalty: Penalty for standing on fire
            forest_penalty: Penalty for being stuck in forest
        """
        super().__init__(env)

        self.health_bonus = health_bonus
        self.armor_bonus = armor_bonus
        self.stamina_bonus = stamina_bonus
        self.formation_bonus = formation_bonus
        self.hazard_avoidance_bonus = hazard_avoidance_bonus
        self.death_penalty = death_penalty
        self.fire_penalty = fire_penalty
        self.forest_penalty = forest_penalty

        # Track previous alive state for death detection
        self._prev_alive: Dict[int, bool] = {}

        print("TacticalRewardWrapper: Phase 1 movement reward shaping enabled")
        print(f"  Health bonus: +{health_bonus}/step (above {self.HEALTH_THRESHOLD * 100:.0f}% HP)")
        print(f"  Formation bonus: +{formation_bonus}/step (within cohesion radius)")
        print(f"  Hazard avoidance bonus: +{hazard_avoidance_bonus}/step")

    def reset(self, **kwargs) -> Tuple[Dict[int, Any], Dict]:
        """Reset and initialize tracking state."""
        obs_dict, info = self.env.reset(**kwargs)

        # Track initial alive state
        self._prev_alive = {}
        for agent_idx in obs_dict.keys():
            agent = self._get_agent(agent_idx)
            if agent is not None:
                self._prev_alive[agent_idx] = agent.is_alive

        return obs_dict, info

    def step(self, actions) -> Tuple[Dict[int, Any], Dict[int, float], bool, bool, Dict]:
        """Step and add shaped rewards."""
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)

        # Shape rewards based on agent state
        shaped_rewards = {}
        for agent_idx in obs_dict.keys():
            base_reward = rewards.get(agent_idx, 0.0)
            shaping_reward = self._compute_shaping_reward(agent_idx)
            shaped_rewards[agent_idx] = base_reward + shaping_reward

        # Update alive tracking
        for agent_idx in obs_dict.keys():
            agent = self._get_agent(agent_idx)
            if agent is not None:
                self._prev_alive[agent_idx] = agent.is_alive

        return obs_dict, shaped_rewards, terminated, truncated, info

    def _compute_shaping_reward(self, agent_idx: int) -> float:
        """
        Compute shaping reward for an agent.

        Args:
            agent_idx: Agent index

        Returns:
            Total shaping reward
        """
        agent = self._get_agent(agent_idx)
        if agent is None:
            return 0.0

        reward = 0.0

        # Death penalty
        was_alive = self._prev_alive.get(agent_idx, True)
        if was_alive and not agent.is_alive:
            return self.death_penalty

        # Only shape rewards for alive agents
        if not agent.is_alive:
            return 0.0

        # Health maintained bonus
        health_ratio = agent.health / AGENT_MAX_HEALTH
        if health_ratio >= self.HEALTH_THRESHOLD:
            reward += self.health_bonus

        # Armor preserved bonus
        armor_ratio = agent.armor / AGENT_MAX_ARMOR if AGENT_MAX_ARMOR > 0 else 1.0
        if armor_ratio >= self.ARMOR_THRESHOLD:
            reward += self.armor_bonus

        # Stamina managed bonus
        stamina_ratio = agent.stamina / AGENT_MAX_STAMINA if AGENT_MAX_STAMINA > 0 else 1.0
        if stamina_ratio >= self.STAMINA_THRESHOLD:
            reward += self.stamina_bonus

        # Formation cohesion bonus
        if self._is_in_formation(agent, agent_idx):
            reward += self.formation_bonus

        # Terrain hazard checks
        terrain = self._get_agent_terrain(agent)
        if terrain == TerrainType.FIRE:
            reward += self.fire_penalty
        elif terrain == TerrainType.FOREST:
            if agent.is_stuck:
                reward += self.forest_penalty
            # Not stuck on swamp = hazard navigated successfully
        else:
            # Not on hazard = bonus
            reward += self.hazard_avoidance_bonus

        return reward

    def _get_agent(self, agent_idx: int):
        """Get agent by index from the environment."""
        # Try MultiAgentWrapper's agent_list first (walks up the chain)
        agent_list = self._find_attr('agent_list')
        if agent_list is not None:
            if 0 <= agent_idx < len(agent_list):
                return agent_list[agent_idx]

        # Try blue_agents + red_agents from wrapper chain
        blue_agents = self._find_attr('blue_agents')
        red_agents = self._find_attr('red_agents')
        if blue_agents is not None and red_agents is not None:
            combined = list(blue_agents) + list(red_agents)
            if 0 <= agent_idx < len(combined):
                return combined[agent_idx]

        return None

    def _find_attr(self, attr_name: str):
        """Find attribute by walking up the wrapper chain."""
        env = self.env
        while env is not None:
            if hasattr(env, attr_name):
                val = getattr(env, attr_name)
                if val is not None:
                    return val
            env = getattr(env, 'env', None)
        return None

    def _is_in_formation(self, agent, agent_idx: int) -> bool:
        """
        Check if agent is within cohesion radius of unit centroid.

        Args:
            agent: Agent instance
            agent_idx: Agent index

        Returns:
            True if agent is in formation
        """
        # Get units from wrapper chain
        blue_agents = self._find_attr('blue_agents') or []
        is_blue = agent_idx < len(blue_agents)
        units = self._find_attr('blue_units') if is_blue else self._find_attr('red_units')
        units = units or []

        # Find agent's unit
        for unit in units:
            if agent in unit.agents:
                centroid = unit.centroid
                dx = agent.position[0] - centroid[0]
                dy = agent.position[1] - centroid[1]
                dist = math.sqrt(dx * dx + dy * dy)
                return dist <= UNIT_COHESION_RADIUS

        return False

    def _get_agent_terrain(self, agent) -> TerrainType:
        """Get terrain type at agent's position."""
        terrain_grid = self._find_attr('terrain_grid')
        if terrain_grid is None:
            return TerrainType.EMPTY

        cell_x = int(agent.position[0])
        cell_y = int(agent.position[1])
        return terrain_grid.get(cell_x, cell_y)
