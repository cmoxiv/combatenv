"""
Terminal logging wrapper for console output.

This wrapper logs environment events to the terminal, useful for
debugging and monitoring training runs.

Usage:
    from combatenv.wrappers import TerminalLogWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = TerminalLogWrapper(env, log_level="info")
"""

from typing import Any, Dict, Tuple

import gymnasium as gym


class TerminalLogWrapper(gym.Wrapper):
    """
    Wrapper for logging environment events to terminal.

    Logs episode starts/ends, kills, rewards, and other events
    based on configuration.

    Attributes:
        log_level: Logging verbosity ("debug", "info", "warning", "error")
        log_kills: Whether to log kill events
        log_episodes: Whether to log episode start/end
        log_rewards: Whether to log rewards each step
        episode_count: Total episodes run
    """

    def __init__(
        self,
        env,
        log_level: str = "info",
        log_kills: bool = False,
        log_episodes: bool = True,
        log_rewards: bool = False,
        log_interval: int = 100,
        prefix: str = "",
    ):
        """
        Initialize the terminal log wrapper.

        Args:
            env: Base environment to wrap
            log_level: Logging verbosity level
            log_kills: Whether to log kill events
            log_episodes: Whether to log episode start/end
            log_rewards: Whether to log rewards each step
            log_interval: How often to log periodic stats (in steps)
            prefix: Prefix string for all log messages
        """
        super().__init__(env)

        self.log_level = log_level
        self.log_kills = log_kills
        self.log_episodes = log_episodes
        self.log_rewards = log_rewards
        self.log_interval = log_interval
        self.prefix = prefix

        # Tracking
        self.episode_count = 0
        self.total_steps = 0
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._prev_blue_kills = 0
        self._prev_red_kills = 0

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset and log episode start."""
        # Log previous episode summary if not first episode
        if self.episode_count > 0 and self.log_episodes:
            self._log_episode_end()

        result = self.env.reset(**kwargs)

        self.episode_count += 1
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._prev_blue_kills = 0
        self._prev_red_kills = 0

        if self.log_episodes:
            self._log(f"Episode {self.episode_count} started")

        return result

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        """Step and log events."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.total_steps += 1
        self._episode_steps += 1

        # Track reward
        if isinstance(reward, dict):
            step_reward = sum(reward.values())
        else:
            step_reward = float(reward)
        self._episode_reward += step_reward

        # Log rewards
        if self.log_rewards and self._episode_steps % self.log_interval == 0:
            self._log(f"Step {self._episode_steps}: reward={step_reward:.4f}, "
                     f"cumulative={self._episode_reward:.2f}")

        # Log kills
        if self.log_kills:
            self._check_and_log_kills()

        # Log episode end
        if (terminated or truncated) and self.log_episodes:
            self._log_episode_end()

        return obs, reward, terminated, truncated, info

    def _check_and_log_kills(self) -> None:
        """Check for new kills and log them."""
        base_env = self.env.unwrapped
        blue_kills = getattr(base_env, 'blue_kills', 0)
        red_kills = getattr(base_env, 'red_kills', 0)

        if blue_kills > self._prev_blue_kills:
            diff = blue_kills - self._prev_blue_kills
            self._log(f"Blue team: +{diff} kills (total: {blue_kills})", level="debug")
            self._prev_blue_kills = blue_kills

        if red_kills > self._prev_red_kills:
            diff = red_kills - self._prev_red_kills
            self._log(f"Red team: +{diff} kills (total: {red_kills})", level="debug")
            self._prev_red_kills = red_kills

    def _log_episode_end(self) -> None:
        """Log episode summary."""
        base_env = self.env.unwrapped
        blue_kills = getattr(base_env, 'blue_kills', 0)
        red_kills = getattr(base_env, 'red_kills', 0)

        self._log(f"Episode {self.episode_count} ended: "
                 f"steps={self._episode_steps}, "
                 f"reward={self._episode_reward:.2f}, "
                 f"blue_kills={blue_kills}, "
                 f"red_kills={red_kills}")

    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message if it meets the log level threshold.

        Args:
            message: Message to log
            level: Log level for this message
        """
        levels = {"debug": 0, "info": 1, "warning": 2, "error": 3}
        current_level = levels.get(self.log_level, 1)
        message_level = levels.get(level, 1)

        if message_level >= current_level:
            prefix = f"[{self.prefix}] " if self.prefix else ""
            print(f"{prefix}{message}")

    def print_summary(self) -> None:
        """Print overall training summary."""
        print(f"\n{'='*50}")
        print(f"Training Summary")
        print(f"{'='*50}")
        print(f"Total episodes: {self.episode_count}")
        print(f"Total steps: {self.total_steps}")
        if self.episode_count > 0:
            avg_steps = self.total_steps / self.episode_count
            print(f"Avg steps/episode: {avg_steps:.1f}")
        print(f"{'='*50}\n")
