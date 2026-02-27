"""
Reward wrapper to shape agent behavior.

This wrapper modifies rewards to incentivize agents to move toward
the enemy team's corner:
    - Blue team (spawns top-left) → rewarded for moving to bottom-right
    - Red team (spawns bottom-right) → rewarded for moving to top-left

The reward shaping encourages aggressive engagement between teams
rather than passive wandering.

Usage:
    from rl_student.wrappers import MultiAgentWrapper, RewardWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = RewardWrapper(env)  # Add before discretization wrappers
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)
"""

from typing import Dict, Tuple, Any

import numpy as np
import gymnasium as gym


class RewardWrapper(gym.Wrapper):
    """
    Reward shaping wrapper to incentivize movement toward enemy territory.

    Blue team goal: bottom-right corner (1.0, 1.0)
    Red team goal: top-left corner (0.0, 0.0)

    Rewards:
        - Progress toward goal: +reward_scale * distance_improvement
        - Reaching goal area: +goal_bonus

    Attributes:
        reward_scale: Multiplier for distance-based rewards (default: 1.0)
        goal_bonus: Bonus for reaching the goal corner (default: 5.0)
        goal_threshold: Distance to corner to count as "reached" (default: 0.2)
    """

    def __init__(
        self,
        env,
        reward_scale: float = 1.0,
        goal_bonus: float = 5.0,
        goal_threshold: float = 0.2
    ):
        """
        Initialize the reward wrapper.

        Args:
            env: Environment (should be MultiAgentWrapper)
            reward_scale: Scale factor for distance rewards
            goal_bonus: Bonus reward for reaching goal corner
            goal_threshold: Normalized distance to consider goal reached
        """
        super().__init__(env)

        self.reward_scale = reward_scale
        self.goal_bonus = goal_bonus
        self.goal_threshold = goal_threshold

        # Store previous distances for reward shaping
        self._prev_distances: Dict[int, float] = {}

        # Goal positions (normalized 0-1)
        # Blue team goal: bottom-right (1.0, 1.0)
        # Red team goal: top-left (0.0, 0.0)
        self.blue_goal = (1.0, 1.0)
        self.red_goal = (0.0, 0.0)

        print(f"RewardWrapper: Incentivizing movement to enemy corners")
        print(f"  Blue team goal: bottom-right {self.blue_goal}")
        print(f"  Red team goal: top-left {self.red_goal}")

    def reset(self, **kwargs) -> Tuple[Dict[int, np.ndarray], Dict]:
        """Reset and initialize distance tracking."""
        obs_dict, info = self.env.reset(**kwargs)

        # Initialize previous distances
        self._prev_distances = {}
        for agent_idx, obs in obs_dict.items():
            self._prev_distances[agent_idx] = self._get_distance_to_goal(agent_idx, obs)

        return obs_dict, info

    def step(self, actions) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict]:
        """Step and add shaped rewards."""
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)

        # Shape rewards based on progress toward goal
        shaped_rewards = {}
        for agent_idx, obs in obs_dict.items():
            base_reward = rewards.get(agent_idx, 0.0)
            shaping_reward = self._compute_shaping_reward(agent_idx, obs)
            shaped_rewards[agent_idx] = base_reward + shaping_reward

        return obs_dict, shaped_rewards, terminated, truncated, info

    def _compute_shaping_reward(self, agent_idx: int, obs: np.ndarray) -> float:
        """
        Compute reward shaping based on movement toward goal.

        Args:
            agent_idx: Agent index (0-99 = blue, 100-199 = red)
            obs: Agent's observation array

        Returns:
            Shaping reward
        """
        # Dead agents get no shaping reward
        if np.all(obs == 0):
            return 0.0

        current_dist = self._get_distance_to_goal(agent_idx, obs)
        prev_dist = self._prev_distances.get(agent_idx, current_dist)

        # Reward for getting closer (positive when distance decreases)
        progress_reward = (prev_dist - current_dist) * self.reward_scale

        # Bonus for reaching goal area
        goal_reward = 0.0
        if current_dist < self.goal_threshold:
            goal_reward = self.goal_bonus

        # Update previous distance
        self._prev_distances[agent_idx] = current_dist

        return progress_reward + goal_reward

    def _get_distance_to_goal(self, agent_idx: int, obs: np.ndarray) -> float:
        """
        Calculate normalized distance from agent to their goal corner.

        Args:
            agent_idx: Agent index (0-99 = blue, 100-199 = red)
            obs: Agent's observation array

        Returns:
            Euclidean distance to goal (normalized, 0 to ~1.41)
        """
        # Get agent position from observation (indices 0-1 are normalized x, y)
        agent_x = obs[0]
        agent_y = obs[1]

        # Determine goal based on team
        if agent_idx < 100:  # Blue team
            goal_x, goal_y = self.blue_goal
        else:  # Red team
            goal_x, goal_y = self.red_goal

        # Euclidean distance
        dx = agent_x - goal_x
        dy = agent_y - goal_y
        return np.sqrt(dx * dx + dy * dy)

    def _is_blue_team(self, agent_idx: int) -> bool:
        """Check if agent is on blue team."""
        return agent_idx < 100
