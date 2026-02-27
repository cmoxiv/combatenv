"""
TacticalQLearningWrapper - Q-learning wrapper for tactical agent training.

This wrapper handles action selection and learning for individual tactical agents
using tabular Q-learning with UCB exploration. It can load pre-trained operational
Q-tables to control units while tactical agents learn.

Features:
- Per-team shared Q-tables for tactical agents (blue agents share one, red share another)
- UCB1 exploration for action selection during training
- Loads pre-trained operational Q-tables for unit control
- Q-table save/load functionality

Usage:
    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)
    env = TacticalQLearningWrapper(env, ops_qtable_path="operational_agent.pkl")
"""

import math
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper


class TacticalQLearningWrapper(BaseWrapper):
    """
    Q-learning wrapper for tactical agent training.

    Each team (blue/red) shares a Q-table among all its agents.
    Uses UCB1 exploration for action selection during training.
    Can load pre-trained operational Q-tables for unit control.

    State Space: 1,920 states (from DiscreteObservationWrapper)
    Action Space: 8 actions (from DiscreteActionWrapper, shoot may be masked)

    Attributes:
        training: Whether in training mode
        learning_rate: Q-learning alpha parameter
        discount_factor: Q-learning gamma parameter
        exploration_bonus: UCB exploration coefficient
        blue_q_table: Q-table for blue team
        red_q_table: Q-table for red team
        blue_n_table: Visit count table for blue team
        red_n_table: Visit count table for red team
        ops_q_tables: Pre-trained operational Q-tables (optional)
        total_steps: Total steps taken (for UCB calculation)
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_bonus: float = 1.0,
        training: bool = True,
        n_actions: int = 8,
        ops_qtable_path: Optional[str] = None,
    ):
        """
        Initialize the TacticalQLearningWrapper.

        Args:
            env: Base environment (should have DiscreteActionWrapper)
            learning_rate: Q-learning alpha (default: 0.1)
            discount_factor: Q-learning gamma (default: 0.95)
            exploration_bonus: UCB exploration coefficient c (default: 1.0)
            training: Whether to train (True) or evaluate (False)
            n_actions: Number of discrete actions (default: 8)
            ops_qtable_path: Path to pre-trained operational Q-tables (optional)
        """
        super().__init__(env)

        self.training = training
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_bonus = exploration_bonus
        self.n_actions = n_actions

        # Get state space size from observation wrapper
        self.n_states = getattr(env, 'n_states', 1920)

        # Per-team Q-tables and N-tables (visit counts)
        self.blue_q_table: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )
        self.red_q_table: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )
        self.blue_n_table: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )
        self.red_n_table: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        # Step counter for UCB
        self.total_steps = 1

        # Store previous observations and actions for learning
        self._prev_obs: Dict[int, int] = {}
        self._prev_actions: Dict[int, int] = {}

        # Episode statistics
        self.episode_blue_reward = 0.0
        self.episode_red_reward = 0.0
        self.episode_steps = 0

        # Load operational Q-tables if provided
        self.ops_q_tables = None
        if ops_qtable_path:
            self._load_ops_qtables(ops_qtable_path)

        print(f"TacticalQLearningWrapper: {self.n_states} states, {n_actions} actions")
        print(f"  Training: {training}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Discount factor: {discount_factor}")
        print(f"  Exploration bonus: {exploration_bonus}")
        if self.ops_q_tables:
            print(f"  Operational Q-tables loaded from: {ops_qtable_path}")

    def _load_ops_qtables(self, filepath: str) -> None:
        """
        Load pre-trained operational Q-tables.

        Args:
            filepath: Path to operational Q-tables file
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            self.ops_q_tables = {
                "blue": data.get("blue_q_table", {}),
                "red": data.get("red_q_table", {}),
            }
            print(f"  Loaded operational Q-tables with {len(self.ops_q_tables['blue'])} blue states")
        except FileNotFoundError:
            print(f"  Warning: Operational Q-table file not found: {filepath}")
            self.ops_q_tables = None
        except Exception as e:
            print(f"  Warning: Failed to load operational Q-tables: {e}")
            self.ops_q_tables = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation_dict, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset episode tracking
        self._prev_obs = {}
        self._prev_actions = {}
        self.episode_blue_reward = 0.0
        self.episode_red_reward = 0.0
        self.episode_steps = 0

        return obs, info

    def step(
        self,
        actions: Optional[Dict[int, int]] = None
    ) -> Tuple[Dict[int, int], Dict[int, float], bool, bool, Dict[str, Any]]:
        """
        Execute one step with Q-learning.

        If actions is None and training=True, selects actions using UCB.
        Updates Q-tables based on rewards.

        Args:
            actions: Optional dict mapping agent_idx to action index.
                     If None, actions are selected automatically.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Get current observations (already discrete from DiscreteObservationWrapper)
        # We need to get observations before taking action
        obs = self._get_current_observations()

        # Select actions if not provided
        if actions is None:
            actions = self._select_all_actions(obs)

        # Store current obs/actions for learning update
        prev_obs = dict(obs)
        prev_actions = dict(actions)

        # Step underlying environment
        next_obs, rewards, terminated, truncated, info = self.env.step(actions)

        # Update Q-tables if training
        if self.training:
            self._update_q_tables(prev_obs, prev_actions, rewards, next_obs, terminated or truncated)

        # Track episode rewards
        blue_count = self._get_blue_count()

        for agent_idx, reward in rewards.items():
            if agent_idx < blue_count:
                self.episode_blue_reward += reward
            else:
                self.episode_red_reward += reward

        self.total_steps += 1
        self.episode_steps += 1

        # Add Q-learning stats to info
        info["blue_episode_reward"] = self.episode_blue_reward
        info["red_episode_reward"] = self.episode_red_reward
        info["total_steps"] = self.total_steps
        info["episode_steps"] = self.episode_steps

        return next_obs, rewards, terminated, truncated, info

    def _get_current_observations(self) -> Dict[int, int]:
        """
        Get current discrete observations from the environment.

        Returns:
            Dict mapping agent_idx to discrete state
        """
        # Try to get from previous observation wrapper's output
        # The observations should already be discrete (from DiscreteObservationWrapper)
        if hasattr(self, '_last_obs'):
            return self._last_obs

        # If not available, return empty dict (will be populated on first step)
        return {}

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

    def _get_blue_count(self) -> int:
        """Get the number of blue agents by walking the wrapper chain."""
        # Try agent_list first (from MultiAgentWrapper)
        agent_list = self._find_attr('agent_list')
        if agent_list is not None:
            # In MultiAgentWrapper, blue agents come first
            blue_agents = self._find_attr('blue_agents')
            if blue_agents is not None:
                return len(blue_agents)
            # Default to half if we can't determine
            return len(agent_list) // 2

        # Try blue_agents directly
        blue_agents = self._find_attr('blue_agents')
        if blue_agents is not None:
            return len(blue_agents)

        # Default fallback
        return 64  # Half of 128 agents

    def _select_all_actions(self, obs: Dict[int, int]) -> Dict[int, int]:
        """
        Select actions for all agents using UCB.

        Args:
            obs: Dict mapping agent_idx to discrete state

        Returns:
            Dict mapping agent_idx to action
        """
        actions = {}
        blue_count = self._get_blue_count()

        for agent_idx, state in obs.items():
            team = "blue" if agent_idx < blue_count else "red"
            actions[agent_idx] = self.select_action(state, team)

        return actions

    def select_action(self, state: int, team: str) -> int:
        """
        Select action using UCB1 exploration.

        UCB1: Q(s,a) + c * sqrt(log(t) / N(s,a))

        Args:
            state: Discrete state index
            team: "blue" or "red"

        Returns:
            Selected action index
        """
        q_table = self.blue_q_table if team == "blue" else self.red_q_table
        n_table = self.blue_n_table if team == "blue" else self.red_n_table

        q_values = q_table[state]
        n_values = n_table[state]

        if self.training:
            # UCB1 exploration
            ucb_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                if n_values[a] == 0:
                    # Unvisited action - explore with high priority
                    ucb_values[a] = float('inf')
                else:
                    exploration = self.exploration_bonus * math.sqrt(
                        math.log(self.total_steps) / n_values[a]
                    )
                    ucb_values[a] = q_values[a] + exploration

            return int(np.argmax(ucb_values))
        else:
            # Greedy action selection
            return int(np.argmax(q_values))

    def _update_q_tables(
        self,
        states: Dict[int, int],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_states: Dict[int, int],
        done: bool,
    ) -> None:
        """
        Update Q-tables using Q-learning update rule.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))

        Args:
            states: Dict mapping agent_idx to state
            actions: Dict mapping agent_idx to action
            rewards: Dict mapping agent_idx to reward
            next_states: Dict mapping agent_idx to next state
            done: Whether episode is done
        """
        blue_count = self._get_blue_count()

        for agent_idx, reward in rewards.items():
            if agent_idx not in states or agent_idx not in actions:
                continue

            state = states[agent_idx]
            action = actions[agent_idx]
            next_state = next_states.get(agent_idx, state)
            team = "blue" if agent_idx < blue_count else "red"

            # Get Q-table and N-table for this team
            q_table = self.blue_q_table if team == "blue" else self.red_q_table
            n_table = self.blue_n_table if team == "blue" else self.red_n_table

            # Q-learning update
            current_q = q_table[state][action]

            if done:
                target = reward
            else:
                max_next_q = np.max(q_table[next_state])
                target = reward + self.discount_factor * max_next_q

            q_table[state][action] += self.learning_rate * (target - current_q)

            # Update visit count
            n_table[state][action] += 1

    def save(self, filepath: str) -> None:
        """
        Save Q-tables and N-tables to file.

        Args:
            filepath: Path to save file
        """
        data = {
            "blue_q_table": dict(self.blue_q_table),
            "red_q_table": dict(self.red_q_table),
            "blue_n_table": dict(self.blue_n_table),
            "red_n_table": dict(self.red_n_table),
            "total_steps": self.total_steps,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_bonus": self.exploration_bonus,
            "n_actions": self.n_actions,
            "n_states": self.n_states,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved tactical Q-tables to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load Q-tables and N-tables from file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Restore Q-tables as defaultdicts
        self.blue_q_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            data["blue_q_table"]
        )
        self.red_q_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            data["red_q_table"]
        )
        self.blue_n_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            data["blue_n_table"]
        )
        self.red_n_table = defaultdict(
            lambda: np.zeros(self.n_actions),
            data["red_n_table"]
        )
        self.total_steps = data.get("total_steps", 1)

        print(f"Loaded tactical Q-tables from {filepath}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Blue states: {len(self.blue_q_table)}")
        print(f"  Red states: {len(self.red_q_table)}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get training statistics.

        Returns:
            Dict with Q-table statistics
        """
        return {
            "total_steps": self.total_steps,
            "blue_states_visited": len(self.blue_q_table),
            "red_states_visited": len(self.red_q_table),
            "blue_total_visits": sum(
                n.sum() for n in self.blue_n_table.values()
            ),
            "red_total_visits": sum(
                n.sum() for n in self.red_n_table.values()
            ),
            "episode_blue_reward": self.episode_blue_reward,
            "episode_red_reward": self.episode_red_reward,
            "episode_steps": self.episode_steps,
        }

    def get_q_values(self, state: int, team: str) -> np.ndarray:
        """
        Get Q-values for a state.

        Args:
            state: Discrete state index
            team: "blue" or "red"

        Returns:
            Array of Q-values for all actions
        """
        q_table = self.blue_q_table if team == "blue" else self.red_q_table
        return q_table[state].copy()

    def get_best_action(self, state: int, team: str) -> int:
        """
        Get the greedy (best) action for a state.

        Args:
            state: Discrete state index
            team: "blue" or "red"

        Returns:
            Best action index
        """
        q_table = self.blue_q_table if team == "blue" else self.red_q_table
        return int(np.argmax(q_table[state]))
