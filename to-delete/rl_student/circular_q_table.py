"""
Circular Q-Table with limited memory for tabular Q-learning.

Uses a ring buffer to store Q-values with FIFO eviction when full.
This is useful when:
- Memory is constrained
- Most states are rarely visited
- You want bounded memory usage

The circular buffer overwrites the oldest entries when the table is full,
providing a form of recency-based forgetting.

Usage:
    from rl_student import CircularQTable

    # Create Q-table with 500 entry limit
    q_table = CircularQTable(max_entries=500, n_actions=8)

    # Get action using epsilon-greedy
    action = q_table.get_action(state=42, epsilon=0.1)

    # Update Q-value
    q_table.update(state=42, action=2, reward=1.0, next_state=43, done=False)

    # Check statistics
    print(q_table.get_statistics())
"""

from typing import Dict, Optional

import numpy as np


class CircularQTable:
    """
    Q-table with fixed memory using circular buffer eviction.

    Instead of storing Q-values for all possible states, this class
    maintains a fixed-size buffer that evicts the oldest entries
    when full (FIFO eviction).

    Attributes:
        max_entries: Maximum number of state entries to store
        n_actions: Number of actions per state
        learning_rate: Q-learning alpha parameter
        discount_factor: Q-learning gamma parameter
        epsilon: Current exploration rate
        count: Number of unique states currently stored
    """

    def __init__(
        self,
        max_entries: int = 500,
        n_actions: int = 8,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        """
        Initialize the circular Q-table.

        Args:
            max_entries: Maximum number of states to store (default: 500)
            n_actions: Number of discrete actions (default: 8)
            learning_rate: Q-learning alpha (default: 0.1)
            discount_factor: Q-learning gamma (default: 0.99)
            epsilon: Initial exploration rate (default: 1.0)
            epsilon_decay: Epsilon decay per episode (default: 0.995)
            epsilon_min: Minimum epsilon (default: 0.05)
        """
        self.max_entries = max_entries
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Circular buffer storage
        self.states = np.zeros(max_entries, dtype=np.int32)
        self.q_values = np.zeros((max_entries, n_actions), dtype=np.float32)
        self.n_values = np.zeros((max_entries, n_actions), dtype=np.int32)  # Visit counts
        self.write_idx = 0  # Next write position (circular)
        self.count = 0  # Number of entries used (up to max_entries)

        # Hash map for O(1) lookup: state -> buffer_idx
        self.state_to_idx: Dict[int, int] = {}

        # Statistics
        self.total_updates = 0
        self.episodes_trained = 0
        self.eviction_count = 0

        memory_kb = (self.states.nbytes + self.q_values.nbytes + self.n_values.nbytes) / 1024
        print(f"CircularQTable: max {max_entries} entries × {n_actions} actions")
        print(f"  Memory usage: {memory_kb:.1f} KB (includes N-table)")

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Get Q-values for a state.

        Args:
            state: State index

        Returns:
            Q-values array of shape (n_actions,). Returns zeros if state not stored.
        """
        if state in self.state_to_idx:
            idx = self.state_to_idx[state]
            return self.q_values[idx].copy()
        return np.zeros(self.n_actions, dtype=np.float32)

    def set_q_values(self, state: int, values: np.ndarray, n_values: Optional[np.ndarray] = None) -> None:
        """
        Store Q-values (and optionally N-values) for a state, evicting oldest if full.

        Args:
            state: State index
            values: Q-values array of shape (n_actions,)
            n_values: Optional N-values array of shape (n_actions,)
        """
        if state in self.state_to_idx:
            # Update existing entry
            idx = self.state_to_idx[state]
            self.q_values[idx] = values
            if n_values is not None:
                self.n_values[idx] = n_values
        else:
            # Add new entry
            if self.count >= self.max_entries:
                # Evict oldest entry at write_idx
                old_state = self.states[self.write_idx]
                if old_state in self.state_to_idx:
                    del self.state_to_idx[old_state]
                self.eviction_count += 1

            # Write new entry
            idx = self.write_idx
            self.states[idx] = state
            self.q_values[idx] = values
            self.n_values[idx] = n_values if n_values is not None else np.zeros(self.n_actions, dtype=np.int32)
            self.state_to_idx[state] = idx

            # Advance write pointer (circular)
            self.write_idx = (self.write_idx + 1) % self.max_entries
            self.count = min(self.count + 1, self.max_entries)

    def get_n_values(self, state: int) -> np.ndarray:
        """
        Get visit counts N(s, :) for a state.

        Args:
            state: State index

        Returns:
            N-values array of shape (n_actions,). Returns zeros if state not stored.
        """
        if state in self.state_to_idx:
            idx = self.state_to_idx[state]
            return self.n_values[idx].copy()
        return np.zeros(self.n_actions, dtype=np.int32)

    def get_action_probabilities(self, state: int) -> np.ndarray:
        """
        Get action probabilities P(a|s) = N(s,a) / Σ_a N(s,a).

        Args:
            state: State index

        Returns:
            Array of action probabilities (sums to 1.0)
        """
        n_values = self.get_n_values(state)
        total = n_values.sum()
        if total == 0:
            return np.ones(self.n_actions, dtype=np.float32) / self.n_actions
        return n_values.astype(np.float32) / total

    def get_action(self, state: int, epsilon: Optional[float] = None, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: State index
            epsilon: Override epsilon value (uses self.epsilon if None)
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action index (0 to n_actions-1)
        """
        eps = epsilon if epsilon is not None else self.epsilon

        # Epsilon-greedy exploration
        if training and np.random.random() < eps:
            return np.random.randint(self.n_actions)

        # Greedy action selection
        q_values = self.get_q_values(state)

        # Break ties randomly
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
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
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            TD error (for debugging)
        """
        # Get current Q-values and N-values
        q_values = self.get_q_values(state)
        n_values = self.get_n_values(state)
        current_q = q_values[action]

        # Increment visit count N(s, a)
        n_values[action] += 1

        # Calculate target
        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target = reward + self.discount_factor * np.max(next_q_values)

        # TD error
        td_error = target - current_q

        # Q-learning update
        q_values[action] = current_q + self.learning_rate * td_error
        self.set_q_values(state, q_values, n_values)

        self.total_updates += 1
        return td_error

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    def reset_epsilon(self, new_epsilon: float = 1.0) -> None:
        """Reset epsilon to a new value."""
        self.epsilon = new_epsilon

    def contains(self, state: int) -> bool:
        """Check if state is in the Q-table."""
        return state in self.state_to_idx

    def get_statistics(self) -> Dict:
        """
        Get Q-table statistics.

        Returns:
            Dict with statistics including memory usage, coverage, etc.
        """
        if self.count > 0:
            active_q = self.q_values[:self.count] if self.count < self.max_entries else self.q_values
            non_zero = np.count_nonzero(active_q)
            mean_q = float(np.mean(active_q))
            max_q = float(np.max(active_q))
            min_q = float(np.min(active_q))
        else:
            non_zero = 0
            mean_q = 0.0
            max_q = 0.0
            min_q = 0.0

        return {
            "type": "circular",
            "max_entries": self.max_entries,
            "n_actions": self.n_actions,
            "entries_used": self.count,
            "capacity_pct": self.count / self.max_entries * 100,
            "evictions": self.eviction_count,
            "total_updates": self.total_updates,
            "episodes_trained": self.episodes_trained,
            "epsilon": self.epsilon,
            "non_zero_q_values": non_zero,
            "mean_q": mean_q,
            "max_q": max_q,
            "min_q": min_q,
            "memory_kb": (self.states.nbytes + self.q_values.nbytes + self.n_values.nbytes) / 1024,
        }

    def clear(self) -> None:
        """Clear all entries from the Q-table and N-table."""
        self.states.fill(0)
        self.q_values.fill(0)
        self.n_values.fill(0)
        self.state_to_idx.clear()
        self.write_idx = 0
        self.count = 0
        self.eviction_count = 0
        self.total_updates = 0
