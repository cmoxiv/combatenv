"""
Multi-Agent Q-Table Manager for independent Q-learning agents.

Manages N Q-tables, one per agent, enabling independent learning where
each agent has its own Q-table and learns from its own experiences.

Usage:
    from rl_student import MultiAgentQManager

    # Create manager for 100 agents
    q_manager = MultiAgentQManager(n_agents=100, n_states=2048, n_actions=10)

    # Training loop
    observations, _ = env.reset()  # Dict[int, int]
    actions = q_manager.get_actions(observations)  # Dict[int, int]
    next_obs, rewards, done, truncated, info = env.step(actions)
    q_manager.update(observations, actions, rewards, next_obs, done or truncated)
"""

from typing import Dict, Any, Optional

from rl_student.circular_q_table import CircularQTable


class MultiAgentQManager:
    """
    Manages individual Q-tables for each agent.

    Each agent maintains its own CircularQTable and learns independently
    from its own state-action-reward experiences.

    Attributes:
        n_agents: Number of agents
        n_states: Number of discrete states per agent
        n_actions: Number of discrete actions per agent
        q_tables: Dict mapping agent_idx -> CircularQTable
    """

    def __init__(
        self,
        n_agents: int,
        n_states: int = 2048,
        n_actions: int = 10,
        max_q_entries: int = 500,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        """
        Initialize the multi-agent Q-table manager.

        Args:
            n_agents: Number of agents to manage
            n_states: Number of discrete states (default: 2048)
            n_actions: Number of discrete actions (default: 10)
            max_q_entries: Maximum entries per Q-table (default: 500)
            learning_rate: Q-learning alpha (default: 0.1)
            discount_factor: Q-learning gamma (default: 0.99)
            epsilon: Initial exploration rate (default: 1.0)
            epsilon_decay: Epsilon decay per episode (default: 0.995)
            epsilon_min: Minimum epsilon (default: 0.05)
        """
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions

        # Create one Q-table per agent
        self.q_tables: Dict[int, CircularQTable] = {}
        for i in range(n_agents):
            self.q_tables[i] = CircularQTable(
                max_entries=max_q_entries,
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )

        print(f"MultiAgentQManager: {n_agents} agents, {n_actions} actions each")
        total_memory = sum(qt.get_statistics()['memory_kb'] for qt in self.q_tables.values())
        print(f"  Total memory: {total_memory:.1f} KB")

    def get_actions(
        self,
        observations: Dict[int, int],
        epsilon: Optional[float] = None,
        training: bool = True
    ) -> Dict[int, int]:
        """
        Get actions for all agents from their Q-tables.

        Args:
            observations: Dict mapping agent_idx -> discrete state
            epsilon: Override epsilon (uses each Q-table's epsilon if None)
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Dict mapping agent_idx -> discrete action
        """
        actions = {}
        for agent_idx, state in observations.items():
            if agent_idx in self.q_tables:
                actions[agent_idx] = self.q_tables[agent_idx].get_action(
                    state, epsilon=epsilon, training=training
                )
            else:
                # Agent not managed - use random action
                import numpy as np
                actions[agent_idx] = np.random.randint(self.n_actions)
        return actions

    def update(
        self,
        states: Dict[int, int],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_states: Dict[int, int],
        done: bool
    ) -> Dict[int, float]:
        """
        Update Q-tables for all agents.

        Args:
            states: Dict mapping agent_idx -> current state
            actions: Dict mapping agent_idx -> action taken
            rewards: Dict mapping agent_idx -> reward received
            next_states: Dict mapping agent_idx -> next state
            done: Whether episode ended

        Returns:
            Dict mapping agent_idx -> TD error
        """
        td_errors = {}
        for agent_idx in states:
            if agent_idx in self.q_tables:
                td_errors[agent_idx] = self.q_tables[agent_idx].update(
                    state=states[agent_idx],
                    action=actions[agent_idx],
                    reward=rewards.get(agent_idx, 0.0),
                    next_state=next_states.get(agent_idx, states[agent_idx]),
                    done=done
                )
        return td_errors

    def decay_epsilon(self) -> None:
        """Decay epsilon for all Q-tables after each episode."""
        for q_table in self.q_tables.values():
            q_table.decay_epsilon()

    def reset_epsilon(self, new_epsilon: float = 1.0) -> None:
        """Reset epsilon to a new value for all Q-tables."""
        for q_table in self.q_tables.values():
            q_table.reset_epsilon(new_epsilon)

    @property
    def epsilon(self) -> float:
        """Current exploration rate (from first Q-table)."""
        if self.q_tables:
            return next(iter(self.q_tables.values())).epsilon
        return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics across all Q-tables.

        Returns:
            Dict with aggregated statistics
        """
        if not self.q_tables:
            return {"n_agents": 0}

        stats_list = [qt.get_statistics() for qt in self.q_tables.values()]

        total_entries = sum(s['entries_used'] for s in stats_list)
        total_updates = sum(s['total_updates'] for s in stats_list)
        total_evictions = sum(s['evictions'] for s in stats_list)
        total_memory = sum(s['memory_kb'] for s in stats_list)

        max_q_vals = [s['max_q'] for s in stats_list]
        min_q_vals = [s['min_q'] for s in stats_list]

        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "total_entries_used": total_entries,
            "avg_entries_per_agent": total_entries / self.n_agents,
            "total_updates": total_updates,
            "total_evictions": total_evictions,
            "epsilon": self.epsilon,
            "max_q": max(max_q_vals),
            "min_q": min(min_q_vals),
            "total_memory_kb": total_memory,
        }

    def get_agent_stats(self, agent_idx: int) -> Dict[str, Any]:
        """
        Get statistics for a specific agent's Q-table.

        Args:
            agent_idx: Agent index

        Returns:
            Dict with Q-table statistics
        """
        if agent_idx in self.q_tables:
            return self.q_tables[agent_idx].get_statistics()
        return {"error": f"Agent {agent_idx} not found"}

    def clear(self) -> None:
        """Clear all entries from all Q-tables."""
        for q_table in self.q_tables.values():
            q_table.clear()

    def share_knowledge(self) -> Dict[str, Any]:
        """
        Share knowledge across agents after an episode.

        For each state, the agent with highest total visits (confidence)
        donates their Q-values to all other agents with lower confidence.

        Returns:
            Dict with sharing statistics
        """
        # 1. Collect all unique states across all agents
        all_states = set()
        for q_table in self.q_tables.values():
            for state_idx in range(q_table.count):
                all_states.add(int(q_table.states[state_idx]))

        states_shared = 0
        updates_made = 0

        # 2. For each state, find the most confident agent
        for state in all_states:
            best_agent_idx = None
            best_confidence = 0
            best_q_values = None
            best_n_values = None

            # Find agent with highest N(s) = sum of N(s,a)
            for agent_idx, q_table in self.q_tables.items():
                n_values = q_table.get_n_values(state)
                confidence = int(n_values.sum())

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_agent_idx = agent_idx
                    best_q_values = q_table.get_q_values(state)
                    best_n_values = n_values

            # 3. Distribute to all agents (if we found a confident source)
            if best_confidence > 0 and best_q_values is not None and best_n_values is not None:
                for agent_idx, q_table in self.q_tables.items():
                    if agent_idx != best_agent_idx:
                        # Only update if target has lower confidence
                        target_n = q_table.get_n_values(state)
                        if target_n.sum() < best_confidence:
                            q_table.set_q_values(state, best_q_values.copy(), best_n_values.copy())
                            updates_made += 1
                states_shared += 1

        return {
            "total_states": len(all_states),
            "states_shared": states_shared,
            "updates_made": updates_made,
        }
