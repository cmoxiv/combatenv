"""
Debug overlay wrapper for rendering debug information.

This wrapper tracks agent rewards and other debug data, providing
visual overlays for debugging and development.

Usage:
    from combatenv.wrappers import DebugOverlayWrapper

    env = TacticalCombatEnv(render_mode="human")
    env = DebugOverlayWrapper(env, show_debug=True)
"""

from typing import Any, Dict, Tuple

import gymnasium as gym


class DebugOverlayWrapper(gym.Wrapper):
    """
    Wrapper for tracking and displaying debug information.

    Tracks rewards per agent, episode statistics, and provides
    methods for accessing debug data.

    Attributes:
        show_debug: Whether debug overlay is visible
        show_rewards: Whether per-agent rewards are displayed
        agent_rewards: Dict mapping agent index to cumulative reward
        episode_rewards: Dict mapping agent index to episode total
        step_rewards: Dict mapping agent index to last step reward
    """

    def __init__(
        self,
        env,
        show_debug: bool = True,
        show_rewards: bool = False,
        track_all_agents: bool = True,
    ):
        """
        Initialize the debug overlay wrapper.

        Args:
            env: Base environment to wrap
            show_debug: Initial state of debug overlay
            show_rewards: Whether to show per-agent rewards
            track_all_agents: Whether to track all agents or just controlled
        """
        super().__init__(env)

        self.show_debug = show_debug
        self.show_rewards = show_rewards
        self.track_all_agents = track_all_agents

        # Reward tracking
        self._agent_rewards: Dict[int, float] = {}  # Last step rewards
        self._episode_rewards: Dict[int, float] = {}  # Cumulative episode rewards
        self._agent_list: list = []  # List of agent indices being tracked

        # Episode statistics
        self._episode_steps = 0
        self._total_episodes = 0

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset and clear reward tracking."""
        result = self.env.reset(**kwargs)

        # Clear reward tracking
        self._agent_rewards = {}
        self._episode_rewards = {}
        self._episode_steps = 0
        self._total_episodes += 1

        # Build agent list
        self._build_agent_list()

        # Sync to base env
        self._sync_to_base_env()

        return result

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        """Step and track rewards."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._episode_steps += 1

        # Track rewards
        self._track_rewards(reward)

        # Sync to base env
        self._sync_to_base_env()

        return obs, reward, terminated, truncated, info

    def _track_rewards(self, reward: Any) -> None:
        """
        Track rewards for all agents.

        Args:
            reward: Reward from step (float or dict)
        """
        if isinstance(reward, dict):
            # Multi-agent rewards
            self._agent_rewards = dict(reward)
            for agent_idx, r in reward.items():
                if agent_idx not in self._episode_rewards:
                    self._episode_rewards[agent_idx] = 0.0
                self._episode_rewards[agent_idx] += r
        else:
            # Single agent reward
            self._agent_rewards = {0: reward}
            if 0 not in self._episode_rewards:
                self._episode_rewards[0] = 0.0
            self._episode_rewards[0] += reward

    def _build_agent_list(self) -> None:
        """Build list of agents to track."""
        self._agent_list = []

        base_env = self.env.unwrapped
        if self.track_all_agents:
            # Get all agents from base env
            blue_agents = getattr(base_env, 'blue_agents', [])
            red_agents = getattr(base_env, 'red_agents', [])
            num_agents = len(blue_agents) + len(red_agents)
            self._agent_list = list(range(num_agents))
        else:
            # Just track controlled agent
            controlled = getattr(base_env, 'controlled_agent', None)
            if controlled is not None:
                self._agent_list = [getattr(controlled, 'id', 0)]

    def _sync_to_base_env(self) -> None:
        """Sync debug data to base environment for rendering."""
        base_env = self.env.unwrapped
        if hasattr(base_env, '_agent_rewards'):
            base_env._agent_rewards = self._agent_rewards
        if hasattr(base_env, '_episode_rewards'):
            base_env._episode_rewards = self._episode_rewards
        if hasattr(base_env, '_agent_list'):
            base_env._agent_list = self._agent_list
        if hasattr(base_env, 'show_debug'):
            base_env.show_debug = self.show_debug
        if hasattr(base_env, 'show_rewards'):
            base_env.show_rewards = self.show_rewards

    def render(self):
        """Render with debug overlay."""
        # Sync state before rendering
        self._sync_to_base_env()
        return self.env.render()

    # Public accessors for debug data
    @property
    def agent_rewards(self) -> Dict[int, float]:
        """Last step rewards per agent."""
        return dict(self._agent_rewards)

    @property
    def episode_rewards(self) -> Dict[int, float]:
        """Cumulative episode rewards per agent."""
        return dict(self._episode_rewards)

    @property
    def episode_steps(self) -> int:
        """Steps in current episode."""
        return self._episode_steps

    @property
    def total_episodes(self) -> int:
        """Total episodes run."""
        return self._total_episodes

    def get_agent_stats(self, agent_idx: int) -> Dict[str, float]:
        """
        Get statistics for a specific agent.

        Args:
            agent_idx: Agent index

        Returns:
            Dict with 'last_reward', 'episode_reward', 'avg_reward'
        """
        last_reward = self._agent_rewards.get(agent_idx, 0.0)
        episode_reward = self._episode_rewards.get(agent_idx, 0.0)
        avg_reward = episode_reward / max(1, self._episode_steps)

        return {
            'last_reward': last_reward,
            'episode_reward': episode_reward,
            'avg_reward': avg_reward,
        }

    def print_summary(self) -> None:
        """Print debug summary to console."""
        print(f"\n=== Debug Summary (Episode {self._total_episodes}) ===")
        print(f"Steps: {self._episode_steps}")

        if self._episode_rewards:
            total = sum(self._episode_rewards.values())
            avg = total / len(self._episode_rewards)
            print(f"Total reward: {total:.2f}")
            print(f"Avg reward per agent: {avg:.4f}")

            # Top 5 agents
            sorted_agents = sorted(
                self._episode_rewards.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            print("Top 5 agents:")
            for idx, reward in sorted_agents:
                print(f"  Agent {idx}: {reward:.2f}")
