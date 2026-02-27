"""
Tactical combat reward wrapper for Phase 2 combat training.

This wrapper shapes rewards to encourage effective combat behavior:
- Killing enemies
- Landing hits
- Efficient ammo usage
- Avoiding friendly fire
- Survival and cover usage

Usage:
    from combatenv.wrappers import TacticalCombatRewardWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = TacticalCombatRewardWrapper(env)

    # Rewards are automatically shaped based on combat events
"""

import math
from typing import Any, Dict, Set, Tuple

import gymnasium as gym

from combatenv.terrain import TerrainType


class TacticalCombatRewardWrapper(gym.Wrapper):
    """
    Reward shaping wrapper for Phase 2 tactical combat training.

    Shapes rewards to encourage effective combat engagement
    while maintaining survival instincts from Phase 1.

    Reward Signals:
        - Kill: +5.0 (enemy eliminated)
        - Hit: +0.5 (successful shot on enemy)
        - Miss: -0.1 (shot that missed)
        - Ammo efficiency: +0.1 (kill with < 10 rounds used)
        - Friendly fire: -2.0 (hit ally)
        - Death: -2.0 (agent eliminated)
        - Survival: +0.01/step (alive bonus)
        - Cover use: +0.02/step (adjacent to building)
        - Low ammo: -0.05/step (magazine < 20%)

    Attributes:
        kill_reward: Reward for killing enemy
        hit_reward: Reward for hitting enemy
        miss_penalty: Penalty for missing
        ammo_efficiency_bonus: Bonus for efficient kills
        friendly_fire_penalty: Penalty for hitting ally
        death_penalty: Penalty for dying
        survival_bonus: Bonus for staying alive
        cover_bonus: Bonus for using cover
        low_ammo_penalty: Penalty for low ammo
    """

    # Reward values (can be tuned)
    KILL_REWARD = 5.0
    HIT_REWARD = 0.5
    MISS_PENALTY = -0.1
    AMMO_EFFICIENCY_BONUS = 0.1
    FRIENDLY_FIRE_PENALTY = -2.0
    DEATH_PENALTY = -2.0
    SURVIVAL_BONUS = 0.01
    COVER_BONUS = 0.02
    LOW_AMMO_PENALTY = -0.05

    # Thresholds
    LOW_AMMO_THRESHOLD = 0.2  # 20% of magazine
    EFFICIENT_KILL_ROUNDS = 10  # Rounds used for efficient kill bonus

    def __init__(
        self,
        env: gym.Env,
        kill_reward: float = 5.0,
        hit_reward: float = 0.5,
        miss_penalty: float = -0.1,
        ammo_efficiency_bonus: float = 0.1,
        friendly_fire_penalty: float = -2.0,
        death_penalty: float = -2.0,
        survival_bonus: float = 0.01,
        cover_bonus: float = 0.02,
        low_ammo_penalty: float = -0.05,
    ):
        """
        Initialize the tactical combat reward wrapper.

        Args:
            env: Environment (should be MultiAgentWrapper)
            kill_reward: Reward for killing enemy
            hit_reward: Reward for hitting enemy
            miss_penalty: Penalty for missing shot
            ammo_efficiency_bonus: Bonus for efficient kills
            friendly_fire_penalty: Penalty for hitting ally
            death_penalty: Penalty for dying
            survival_bonus: Per-step bonus for staying alive
            cover_bonus: Per-step bonus for using cover
            low_ammo_penalty: Per-step penalty for low ammo
        """
        super().__init__(env)

        self.kill_reward = kill_reward
        self.hit_reward = hit_reward
        self.miss_penalty = miss_penalty
        self.ammo_efficiency_bonus = ammo_efficiency_bonus
        self.friendly_fire_penalty = friendly_fire_penalty
        self.death_penalty = death_penalty
        self.survival_bonus = survival_bonus
        self.cover_bonus = cover_bonus
        self.low_ammo_penalty = low_ammo_penalty

        # Track previous state for event detection
        self._prev_alive: Dict[int, bool] = {}
        self._prev_health: Dict[int, float] = {}
        self._prev_ammo: Dict[int, int] = {}
        self._shots_fired: Dict[int, int] = {}  # Agent idx -> shots since last kill

        print("TacticalCombatRewardWrapper: Phase 2 combat reward shaping enabled")
        print(f"  Kill reward: +{kill_reward}")
        print(f"  Hit reward: +{hit_reward}")
        print(f"  Miss penalty: {miss_penalty}")
        print(f"  Survival bonus: +{survival_bonus}/step")

    def reset(self, **kwargs) -> Tuple[Dict[int, Any], Dict]:
        """Reset and initialize tracking state."""
        obs_dict, info = self.env.reset(**kwargs)

        # Track initial state
        self._prev_alive = {}
        self._prev_health = {}
        self._prev_ammo = {}
        self._shots_fired = {}

        for agent_idx in obs_dict.keys():
            agent = self._get_agent(agent_idx)
            if agent is not None:
                self._prev_alive[agent_idx] = agent.is_alive
                self._prev_health[agent_idx] = agent.health
                self._prev_ammo[agent_idx] = agent.magazine_ammo
                self._shots_fired[agent_idx] = 0

        return obs_dict, info

    def step(self, actions) -> Tuple[Dict[int, Any], Dict[int, float], bool, bool, Dict]:
        """Step and add shaped rewards."""
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)

        # Shape rewards based on combat events
        shaped_rewards = {}
        for agent_idx in obs_dict.keys():
            base_reward = rewards.get(agent_idx, 0.0)
            shaping_reward = self._compute_shaping_reward(agent_idx, info)
            shaped_rewards[agent_idx] = base_reward + shaping_reward

        # Update tracking state
        for agent_idx in obs_dict.keys():
            agent = self._get_agent(agent_idx)
            if agent is not None:
                self._prev_alive[agent_idx] = agent.is_alive
                self._prev_health[agent_idx] = agent.health
                self._prev_ammo[agent_idx] = agent.magazine_ammo

        return obs_dict, shaped_rewards, terminated, truncated, info

    def _compute_shaping_reward(self, agent_idx: int, info: Dict) -> float:
        """
        Compute shaping reward for an agent.

        Args:
            agent_idx: Agent index
            info: Step info dict (may contain kill/hit info)

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

        # Survival bonus
        reward += self.survival_bonus

        # Shot detection (ammo decreased)
        prev_ammo = self._prev_ammo.get(agent_idx, agent.magazine_ammo)
        ammo_used = prev_ammo - agent.magazine_ammo
        if ammo_used > 0:
            self._shots_fired[agent_idx] = self._shots_fired.get(agent_idx, 0) + ammo_used

        # Kill reward (check from info if available)
        kills = self._get_agent_kills(agent_idx, info)
        if kills > 0:
            reward += kills * self.kill_reward

            # Ammo efficiency bonus
            shots = self._shots_fired.get(agent_idx, 0)
            if shots > 0 and shots <= self.EFFICIENT_KILL_ROUNDS:
                reward += self.ammo_efficiency_bonus
            self._shots_fired[agent_idx] = 0  # Reset after kill

        # Hit reward (enemy health decreased by this agent's shots)
        hits = self._get_agent_hits(agent_idx, info)
        if hits > 0:
            reward += hits * self.hit_reward

        # Miss penalty (shot fired but no hit detected)
        if ammo_used > 0 and hits == 0:
            reward += self.miss_penalty

        # Friendly fire penalty
        friendly_hits = self._get_friendly_hits(agent_idx, info)
        if friendly_hits > 0:
            reward += friendly_hits * self.friendly_fire_penalty

        # Cover usage bonus (adjacent to building)
        if self._is_near_cover(agent):
            reward += self.cover_bonus

        # Low ammo penalty
        magazine_ratio = agent.magazine_ammo / 30.0  # Assuming 30 round mag
        if magazine_ratio < self.LOW_AMMO_THRESHOLD:
            reward += self.low_ammo_penalty

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

    def _get_agent_kills(self, agent_idx: int, info: Dict) -> int:
        """
        Get number of kills by this agent this step.

        Checks info dict for kill tracking from MultiAgentWrapper.
        """
        # Check info for kill tracking
        enemy_kills = info.get("enemy_kills", {})
        return enemy_kills.get(agent_idx, 0)

    def _get_agent_hits(self, agent_idx: int, info: Dict) -> int:
        """
        Get number of hits by this agent this step.

        Checks info dict for hit tracking.
        """
        hits = info.get("hits_by_shooter", {})
        return hits.get(agent_idx, 0)

    def _get_friendly_hits(self, agent_idx: int, info: Dict) -> int:
        """
        Get number of friendly fire hits by this agent.
        """
        friendly_kills = info.get("friendly_kills", {})
        return friendly_kills.get(agent_idx, 0)

    def _is_near_cover(self, agent) -> bool:
        """
        Check if agent is adjacent to a building (cover).

        Args:
            agent: Agent instance

        Returns:
            True if agent is near cover
        """
        terrain_grid = self._find_attr('terrain_grid')
        if terrain_grid is None:
            return False

        agent_x, agent_y = int(agent.position[0]), int(agent.position[1])

        # Check adjacent cells for buildings
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                check_x = agent_x + dx
                check_y = agent_y + dy

                # Bounds check
                if check_x < 0 or check_y < 0:
                    continue
                if check_x >= terrain_grid.width or check_y >= terrain_grid.height:
                    continue

                terrain = terrain_grid.get(check_x, check_y)
                if terrain == TerrainType.OBSTACLE:
                    return True

        return False
