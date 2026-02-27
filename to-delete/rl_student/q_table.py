"""
Q-Table management for tabular Q-learning.

Supports two training modes:
    1. Shared Q-table: All agents share one Q-table (faster learning, less memory)
    2. Individual Q-tables: Each agent has its own Q-table (more specialized behavior)

Q-Learning Update Rule:
    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

Where:
    - s: current state
    - a: action taken
    - r: reward received
    - s': next state
    - alpha: learning rate
    - gamma: discount factor

Usage:
    from rl_student import QTableManager

    # Shared Q-table mode
    q_manager = QTableManager(n_states=2048, n_actions=8, mode="shared")

    # Individual Q-tables mode
    q_manager = QTableManager(n_states=2048, n_actions=8, n_agents=200, mode="individual")

    # Get action for agent
    action = q_manager.get_action(agent_idx=0, state=123, training=True)

    # Update Q-value
    q_manager.update(agent_idx=0, state=123, action=2, reward=1.0, next_state=456, done=False)
"""

import pickle
from typing import Dict, Optional

import numpy as np


class QTableManager:
    """
    Manages Q-tables for multi-agent tabular Q-learning.

    Attributes:
        n_states: Number of discrete states
        n_actions: Number of discrete actions
        n_agents: Number of agents (only used for individual mode)
        mode: "shared" or "individual"
        learning_rate: Alpha parameter for Q-learning
        discount_factor: Gamma parameter for Q-learning
        epsilon: Current exploration rate
        epsilon_decay: Decay rate per episode
        epsilon_min: Minimum epsilon value
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_agents: int = 200,
        mode: str = "shared",
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        """
        Initialize Q-table manager.

        Args:
            n_states: Number of discrete states
            n_actions: Number of discrete actions
            n_agents: Number of agents (only used for individual mode)
            mode: "shared" or "individual"
            learning_rate: Q-learning alpha (default: 0.1)
            discount_factor: Q-learning gamma (default: 0.99)
            epsilon: Initial exploration rate (default: 1.0)
            epsilon_decay: Epsilon decay rate per episode (default: 0.995)
            epsilon_min: Minimum epsilon value (default: 0.05)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.mode = mode
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Statistics
        self.total_updates = 0
        self.episodes_trained = 0

        # Initialize Q-table(s) and N-table(s) for visit counts
        if mode == "shared":
            self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
            self.n_table = np.zeros((n_states, n_actions), dtype=np.int32)
            memory_kb = (self.q_table.nbytes + self.n_table.nbytes) / 1024
            print(f"QTableManager: Shared Q-table ({n_states} x {n_actions})")
            print(f"  Memory usage: {memory_kb:.1f} KB (includes N-table)")
        elif mode == "individual":
            self.q_tables = {
                agent_idx: np.zeros((n_states, n_actions), dtype=np.float32)
                for agent_idx in range(n_agents)
            }
            self.n_tables = {
                agent_idx: np.zeros((n_states, n_actions), dtype=np.int32)
                for agent_idx in range(n_agents)
            }
            total_bytes = sum(q.nbytes for q in self.q_tables.values())
            total_bytes += sum(n.nbytes for n in self.n_tables.values())
            memory_mb = total_bytes / 1024 / 1024
            print(f"QTableManager: {n_agents} individual Q-tables ({n_states} x {n_actions} each)")
            print(f"  Total memory usage: {memory_mb:.1f} MB (includes N-tables)")
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'shared' or 'individual'.")

    def get_action(self, agent_idx: int, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            agent_idx: Agent index (0-199)
            state: Discrete state index
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action index (0 to n_actions-1)
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Greedy action selection
        q_values = self._get_q_values(agent_idx, state)

        # Break ties randomly
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def get_actions(self, states: Dict[int, int], training: bool = True) -> Dict[int, int]:
        """
        Select actions for all agents.

        Args:
            states: Dict mapping agent_idx -> discrete state
            training: If True, use epsilon-greedy

        Returns:
            Dict mapping agent_idx -> selected action
        """
        return {
            agent_idx: self.get_action(agent_idx, state, training)
            for agent_idx, state in states.items()
        }

    def update(
        self,
        agent_idx: int,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Update Q-value using Q-learning update rule.

        Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

        Also increments N(s, a) visit count for probability derivation.

        Args:
            agent_idx: Agent index
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            TD error (for debugging)
        """
        # Increment visit count N(s, a)
        self._increment_n(agent_idx, state, action)

        current_q = self._get_q_value(agent_idx, state, action)

        if done:
            target = reward
        else:
            next_q_values = self._get_q_values(agent_idx, next_state)
            target = reward + self.discount_factor * np.max(next_q_values)

        # TD error
        td_error = target - current_q

        # Q-learning update
        new_q = current_q + self.learning_rate * td_error
        self._set_q_value(agent_idx, state, action, new_q)

        self.total_updates += 1
        return td_error

    def batch_update(
        self,
        states: Dict[int, int],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_states: Dict[int, int],
        done: bool
    ) -> Dict[int, float]:
        """
        Update Q-values for all agents at once.

        Args:
            states: Dict mapping agent_idx -> current state
            actions: Dict mapping agent_idx -> action taken
            rewards: Dict mapping agent_idx -> reward received
            next_states: Dict mapping agent_idx -> next state
            done: Whether episode ended (applies to all agents)

        Returns:
            Dict mapping agent_idx -> TD error
        """
        td_errors = {}
        for agent_idx in states.keys():
            td_errors[agent_idx] = self.update(
                agent_idx,
                states[agent_idx],
                actions[agent_idx],
                rewards[agent_idx],
                next_states[agent_idx],
                done
            )
        return td_errors

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    def _get_q_values(self, agent_idx: int, state: int) -> np.ndarray:
        """Get Q-values for all actions in a state."""
        if self.mode == "shared":
            return self.q_table[state]
        else:
            return self.q_tables[agent_idx][state]

    def _get_q_value(self, agent_idx: int, state: int, action: int) -> float:
        """Get Q-value for specific state-action pair."""
        if self.mode == "shared":
            return self.q_table[state, action]
        else:
            return self.q_tables[agent_idx][state, action]

    def _set_q_value(self, agent_idx: int, state: int, action: int, value: float) -> None:
        """Set Q-value for specific state-action pair."""
        if self.mode == "shared":
            self.q_table[state, action] = value
        else:
            self.q_tables[agent_idx][state, action] = value

    def _get_n_value(self, agent_idx: int, state: int, action: int) -> int:
        """Get visit count N(s, a) for specific state-action pair."""
        if self.mode == "shared":
            return int(self.n_table[state, action])
        else:
            return int(self.n_tables[agent_idx][state, action])

    def _get_n_values(self, agent_idx: int, state: int) -> np.ndarray:
        """Get visit counts N(s, :) for all actions in a state."""
        if self.mode == "shared":
            return self.n_table[state]
        else:
            return self.n_tables[agent_idx][state]

    def _increment_n(self, agent_idx: int, state: int, action: int) -> None:
        """Increment visit count N(s, a)."""
        if self.mode == "shared":
            self.n_table[state, action] += 1
        else:
            self.n_tables[agent_idx][state, action] += 1

    def get_action_probabilities(self, agent_idx: int, state: int) -> np.ndarray:
        """
        Get action probabilities P(a|s) = N(s,a) / Σ_a N(s,a).

        Args:
            agent_idx: Agent index
            state: Discrete state index

        Returns:
            Array of action probabilities (sums to 1.0)
        """
        n_values = self._get_n_values(agent_idx, state)
        total = n_values.sum()
        if total == 0:
            return np.ones(self.n_actions, dtype=np.float32) / self.n_actions
        return n_values.astype(np.float32) / total

    def get_best_action(self, agent_idx: int, state: int) -> int:
        """
        Get the greedy (best) action for a state.

        Args:
            agent_idx: Agent index
            state: Discrete state index

        Returns:
            Best action index
        """
        return self.get_action(agent_idx, state, training=False)

    def get_value(self, agent_idx: int, state: int) -> float:
        """
        Get the value of a state (max Q-value across actions).

        Args:
            agent_idx: Agent index
            state: Discrete state index

        Returns:
            State value (max Q-value)
        """
        return float(np.max(self._get_q_values(agent_idx, state)))

    def get_statistics(self) -> Dict:
        """
        Get training statistics.

        Returns:
            Dict with training statistics
        """
        if self.mode == "shared":
            non_zero = np.count_nonzero(self.q_table)
            total_entries = self.q_table.size
            mean_q = np.mean(self.q_table)
            max_q = np.max(self.q_table)
            min_q = np.min(self.q_table)
        else:
            all_q = np.concatenate([q.flatten() for q in self.q_tables.values()])
            non_zero = np.count_nonzero(all_q)
            total_entries = all_q.size
            mean_q = np.mean(all_q)
            max_q = np.max(all_q)
            min_q = np.min(all_q)

        return {
            "mode": self.mode,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "total_updates": self.total_updates,
            "non_zero_entries": non_zero,
            "total_entries": total_entries,
            "coverage": non_zero / total_entries if total_entries > 0 else 0,
            "mean_q": float(mean_q),
            "max_q": float(max_q),
            "min_q": float(min_q),
        }

    def save(self, filepath: str) -> None:
        """
        Save Q-table(s) and training state to file.

        Args:
            filepath: Path to save file (should end in .pkl)
        """
        data = {
            "mode": self.mode,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "total_updates": self.total_updates,
            "episodes_trained": self.episodes_trained,
        }

        if self.mode == "shared":
            data["q_table"] = self.q_table
            data["n_table"] = self.n_table
        else:
            data["q_tables"] = self.q_tables
            data["n_tables"] = self.n_tables

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved Q-table and N-table to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load Q-table(s) and training state from file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Verify compatibility
        if data["mode"] != self.mode:
            raise ValueError(f"Mode mismatch: file has '{data['mode']}', expected '{self.mode}'")
        if data["n_states"] != self.n_states:
            raise ValueError(f"State space mismatch: file has {data['n_states']}, expected {self.n_states}")
        if data["n_actions"] != self.n_actions:
            raise ValueError(f"Action space mismatch: file has {data['n_actions']}, expected {self.n_actions}")

        # Load parameters
        self.epsilon = data["epsilon"]
        self.total_updates = data["total_updates"]
        self.episodes_trained = data["episodes_trained"]

        # Load Q-table(s) and N-table(s)
        if self.mode == "shared":
            self.q_table = data["q_table"]
            # Backward compatible: create empty N-table if not present
            self.n_table = data.get("n_table", np.zeros_like(self.q_table, dtype=np.int32))
        else:
            self.q_tables = data["q_tables"]
            # Backward compatible: create empty N-tables if not present
            if "n_tables" in data:
                self.n_tables = data["n_tables"]
            else:
                self.n_tables = {
                    agent_idx: np.zeros_like(q, dtype=np.int32)
                    for agent_idx, q in self.q_tables.items()
                }

        print(f"Loaded Q-table and N-table from {filepath}")
        print(f"  Episodes trained: {self.episodes_trained}")
        print(f"  Epsilon: {self.epsilon:.4f}")

    def reset_epsilon(self, new_epsilon: float = 1.0) -> None:
        """
        Reset epsilon to a new value (useful for continuing training).

        Args:
            new_epsilon: New epsilon value (default: 1.0)
        """
        self.epsilon = new_epsilon
        print(f"Reset epsilon to {self.epsilon}")
