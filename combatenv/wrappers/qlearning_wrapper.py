"""
QLearningWrapper - Q-learning wrapper with per-team shared Q-tables.

This wrapper handles action selection and learning for unit-level
waypoint navigation training using tabular Q-learning with UCB exploration.

Features:
- Per-team shared Q-tables (blue units share one, red units share another)
- UCB1 exploration for action selection during training
- Built-in state discretization (432 states per unit)
- Action application for both teams
- Q-table save/load and merge functionality

Usage:
    env = TacticalCombatEnv(render_mode=None)
    env = OperationalWrapper(env)
    env = WaypointTaskWrapper(env)
    env = QLearningWrapper(env, training=True)
"""

import math
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..config import GRID_SIZE, TACTICAL_CELLS_PER_OPERATIONAL
from ..terrain import TerrainType


class QLearningWrapper(BaseWrapper):
    """
    Q-learning wrapper with per-team shared Q-tables.

    Each team (blue/red) shares a Q-table among all its units.
    Uses UCB1 exploration for action selection during training.
    Includes built-in state discretization and action application.

    State Space (7,776 states per unit):
        - Formation: 3 levels (cohesive, scattered, broken)
        - Health: 2 levels (high ≥0.5, low <0.5)
        - Alive ratio: 2 levels (high ≥0.5, low <0.5)
        - Ammo: 2 levels (high ≥0.5, low <0.5)
        - Stamina: 2 levels (high ≥0.5, low <0.5)
        - Waypoint H dist: 3 levels (left, center, right)
        - Waypoint V dist: 3 levels (above, center, below)
        - Waypoint terrain: 6 types (empty, forest, water, fire,
                                     majority-enemy, majority-friend)
        Total: 3 × 2 × 2 × 2 × 2 × 3 × 3 × 6 = 7,776

    Action Space (16 actions per unit):
        - 0: Hold (clear intermediate waypoint)
        - 1-4: Set intermediate waypoint (N, W, E, S) for smooth boids movement
        - 5-8: Set intermediate waypoint (NE, NW, SE, SW) diagonal movement
        - 9-12: Dispatch to strategic cells (unused)
        - 13: Aggressive stance
        - 14: Defensive stance
        - 15: Think (placeholder for logic programming reasoning)

    Note: The goal waypoint (for reward) is set externally via mouse/strategic level.
    Actions set intermediate waypoints to allow obstacle avoidance while navigating.

    Attributes:
        training: Whether in training mode
        learning_rate: Q-learning alpha parameter
        discount_factor: Q-learning gamma parameter
        exploration_bonus: UCB exploration coefficient
        blue_q_table: Q-table for blue team
        red_q_table: Q-table for red team
        blue_n_table: Visit count table for blue team
        red_n_table: Visit count table for red team
        total_steps: Total steps taken (for UCB calculation)
    """

    # State discretization constants
    COHESIVE_THRESHOLD = 2.0   # spread < 2.0 = cohesive
    SCATTERED_THRESHOLD = 4.0  # spread < 4.0 = scattered, else broken
    RESOURCE_THRESHOLD = 0.5

    # Movement direction offsets (cardinal and diagonal)
    # [6]NW [1]N [5]NE
    #  [2]W [X]  [3]E
    # [7]SW [4]S [8]SE
    MOVE_OFFSETS = {
        1: (0, -1),    # N (up)
        2: (-1, 0),    # W (left)
        3: (1, 0),     # E (right)
        4: (0, 1),     # S (down)
        5: (1, -1),    # NE (up-right)
        6: (-1, -1),   # NW (up-left)
        7: (-1, 1),    # SW (down-left)
        8: (1, 1),     # SE (down-right)
    }

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_bonus: float = 1.0,
        training: bool = True,
        n_actions: int = 16,
        move_step: float = 2.0,
    ):
        """
        Initialize the QLearningWrapper.

        Args:
            env: Base environment (should have OperationalWrapper)
            learning_rate: Q-learning alpha (default: 0.1)
            discount_factor: Q-learning gamma (default: 0.95)
            exploration_bonus: UCB exploration coefficient c (default: 1.0)
            training: Whether to train (True) or evaluate (False)
            n_actions: Number of discrete actions (default: 16)
            move_step: Distance for intermediate waypoint (default: 2.0 = 1/4 op cell)
        """
        super().__init__(env)

        self.training = training
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_bonus = exploration_bonus
        self.n_actions = n_actions
        self.move_step = move_step

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
        self._prev_obs: Dict[Tuple[str, int], int] = {}
        self._prev_actions: Dict[Tuple[str, int], int] = {}

        # Episode statistics
        self.episode_blue_reward = 0.0
        self.episode_red_reward = 0.0

        # Expose discrete observations for external use
        self.blue_observations: Dict[int, int] = {}
        self.red_observations: Dict[int, int] = {}

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

        # Compute initial discrete observations
        self._update_discrete_observations()

        return obs, info

    def step(
        self,
        actions: Optional[Dict[Tuple[str, int], int]] = None
    ) -> Tuple[Dict[int, int], Dict[Tuple[str, int], float], bool, bool, Dict[str, Any]]:
        """
        Execute one step with Q-learning.

        If actions is None and training=True, selects actions using UCB.
        Updates Q-tables based on rewards.

        Args:
            actions: Optional dict mapping (team, unit_id) to action index.
                     If None, actions are selected automatically.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Get current discrete observations
        blue_obs = dict(self.blue_observations)
        red_obs = dict(self.red_observations)

        # Select actions if not provided
        if actions is None:
            actions = self._select_all_actions(blue_obs, red_obs)

        # Apply actions to units
        self._apply_all_actions(actions)

        # Step underlying environment with no-op action
        no_op = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, rewards, terminated, truncated, info = self.env.step(no_op)

        # Update discrete observations after step
        self._update_discrete_observations()

        # Update Q-tables if training
        if self.training:
            self._update_q_tables(rewards, terminated or truncated)

        # Store current obs/actions for next update
        self._prev_obs = {}
        self._prev_actions = {}

        for (team, unit_id), action in actions.items():
            self._prev_obs[(team, unit_id)] = (
                blue_obs.get(unit_id, 0) if team == "blue" else red_obs.get(unit_id, 0)
            )
            self._prev_actions[(team, unit_id)] = action

        # Track episode rewards
        for (team, unit_id), reward in rewards.items():
            if team == "blue":
                self.episode_blue_reward += reward
            else:
                self.episode_red_reward += reward

        self.total_steps += 1

        # Add Q-learning stats to info
        info["blue_episode_reward"] = self.episode_blue_reward
        info["red_episode_reward"] = self.episode_red_reward
        info["total_steps"] = self.total_steps

        return obs, rewards, terminated, truncated, info

    def _select_all_actions(
        self,
        blue_obs: Dict[int, int],
        red_obs: Dict[int, int],
    ) -> Dict[Tuple[str, int], int]:
        """
        Select actions for all units using UCB.

        Args:
            blue_obs: Dict mapping blue unit_id to state
            red_obs: Dict mapping red unit_id to state

        Returns:
            Dict mapping (team, unit_id) to action
        """
        actions = {}

        for unit_id, state in blue_obs.items():
            actions[("blue", unit_id)] = self.select_action(state, "blue")

        for unit_id, state in red_obs.items():
            actions[("red", unit_id)] = self.select_action(state, "red")

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

    def _update_discrete_observations(self) -> None:
        """
        Update discrete observations for all units.

        Computes discrete state indices from unit properties.
        """
        blue_units = getattr(self.env, 'blue_units', [])
        red_units = getattr(self.env, 'red_units', [])

        self.blue_observations = {}
        self.red_observations = {}

        for unit in blue_units:
            self.blue_observations[unit.id] = self._discretize_unit(unit)

        for unit in red_units:
            self.red_observations[unit.id] = self._discretize_unit(unit)

    def _discretize_unit(self, unit) -> int:
        """
        Convert a unit's state to discrete index.

        Args:
            unit: Unit instance

        Returns:
            Discrete state index (0 to 10367)
        """
        alive_agents = unit.alive_agents

        # Formation state (0=cohesive, 1=scattered, 2=broken)
        spread = unit.get_formation_spread()
        if spread < self.COHESIVE_THRESHOLD:
            formation = 0
        elif spread < self.SCATTERED_THRESHOLD:
            formation = 1
        else:
            formation = 2

        # Average health (0=low, 1=high)
        if alive_agents:
            avg_health = sum(a.health for a in alive_agents) / len(alive_agents) / 100.0
        else:
            avg_health = 0.0
        health = 1 if avg_health >= self.RESOURCE_THRESHOLD else 0

        # Alive ratio (0=low, 1=high)
        alive_ratio = unit.alive_count / max(1, len(unit.agents))
        alive = 1 if alive_ratio >= self.RESOURCE_THRESHOLD else 0

        # Average ammo (0=low, 1=high) - normalized by magazine size
        if alive_agents:
            avg_ammo = sum(a.magazine_ammo for a in alive_agents) / len(alive_agents) / 30.0
        else:
            avg_ammo = 0.0
        ammo = 1 if avg_ammo >= self.RESOURCE_THRESHOLD else 0

        # Average stamina (0=low, 1=high) - normalized by max stamina
        if alive_agents:
            avg_stamina = sum(a.stamina for a in alive_agents) / len(alive_agents) / 100.0
        else:
            avg_stamina = 0.0
        stamina = 1 if avg_stamina >= self.RESOURCE_THRESHOLD else 0

        # Goal horizontal distance (0=left, 1=center, 2=right)
        # Goal vertical distance (0=above, 1=center, 2=below)
        # Use unit's goal_waypoint (set by strategic level or mouse)
        centroid = unit.centroid
        goal = unit.goal_waypoint
        if goal is not None:
            dx = goal[0] - centroid[0]
            dy = goal[1] - centroid[1]
            # Horizontal: negative=goal is left, positive=goal is right
            if dx < -4.0:
                h_dist = 0  # goal is left
            elif dx > 4.0:
                h_dist = 2  # goal is right
            else:
                h_dist = 1  # center
            # Vertical: negative=goal is above, positive=goal is below
            if dy < -4.0:
                v_dist = 0  # goal is above
            elif dy > 4.0:
                v_dist = 2  # goal is below
            else:
                v_dist = 1  # center
        else:
            h_dist = 1  # center (no goal)
            v_dist = 1  # center (no goal)

        # Terrain at intermediate waypoint (or centroid if no waypoint)
        # Includes agent presence: majority-enemy (6) or majority-friend (7)
        waypoint = unit.waypoint if unit.waypoint else centroid
        terrain = self._get_waypoint_terrain(waypoint, unit.team)

        # Encode as mixed-radix number
        # 3 × 2 × 2 × 2 × 2 × 3 × 3 × 6 = 7,776
        state = (
            formation +
            health * 3 +
            alive * 6 +
            ammo * 12 +
            stamina * 24 +
            h_dist * 48 +
            v_dist * 144 +
            terrain * 432
        )

        return state

    def _get_majority_terrain(self, centroid: tuple) -> int:
        """
        Get majority terrain type in the operational cell containing the centroid.

        Args:
            centroid: (x, y) position

        Returns:
            Terrain type index (0-5): grass, forest, water, swamp, fire, building
        """
        terrain_grid = getattr(self, 'terrain_grid', None)
        if terrain_grid is None:
            return 0  # Default to grass

        # Find operational cell boundaries
        cell_size = TACTICAL_CELLS_PER_OPERATIONAL
        op_x = int(centroid[0] // cell_size)
        op_y = int(centroid[1] // cell_size)

        # Count terrain types in the 8x8 cell
        terrain_counts = {t: 0 for t in TerrainType}

        start_x = op_x * cell_size
        start_y = op_y * cell_size

        for dx in range(cell_size):
            for dy in range(cell_size):
                x = int(start_x + dx)
                y = int(start_y + dy)
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    t = terrain_grid.get(x, y)
                    terrain_counts[t] = terrain_counts.get(t, 0) + 1

        # Find majority terrain (default to GRASS if no terrain found)
        if not terrain_counts:
            return 0  # empty

        majority = max(terrain_counts.keys(), key=lambda t: terrain_counts[t])

        # Map terrain type to simplified index
        # 0=empty (grass/swamp/building), 1=forest, 2=water, 3=fire
        terrain_indices = {
            "GRASS": 0,
            "FOREST": 0,
            "OBSTACLE": 0,
            "FOREST": 1,
            "WATER": 2,
            "FIRE": 3,
        }
        return terrain_indices.get(majority.name, 0)

    def _get_waypoint_terrain(self, position: tuple, team: str) -> int:
        """
        Get terrain/presence type at waypoint's operational cell.

        Returns terrain type (0-3), or majority-enemy (4), or majority-friend (5)
        based on agent presence in the cell.

        Args:
            position: (x, y) position to check
            team: Unit's team ("blue" or "red")

        Returns:
            0=empty, 1=forest, 2=water, 3=fire,
            4=majority-enemy, 5=majority-friend
        """
        # Find operational cell boundaries
        cell_size = TACTICAL_CELLS_PER_OPERATIONAL
        op_x = int(position[0] // cell_size)
        op_y = int(position[1] // cell_size)

        start_x = op_x * cell_size
        start_y = op_y * cell_size
        end_x = start_x + cell_size
        end_y = start_y + cell_size

        # Count agents in this operational cell
        friendly_count = 0
        enemy_count = 0

        # Get all agents
        team_agents = getattr(self, 'team_agents', {})
        blue_agents = team_agents.get("blue", [])
        red_agents = team_agents.get("red", [])

        friendly_agents = blue_agents if team == "blue" else red_agents
        enemy_agents = red_agents if team == "blue" else blue_agents

        for agent in friendly_agents:
            if agent.is_alive:
                ax, ay = agent.position
                if start_x <= ax < end_x and start_y <= ay < end_y:
                    friendly_count += 1

        for agent in enemy_agents:
            if agent.is_alive:
                ax, ay = agent.position
                if start_x <= ax < end_x and start_y <= ay < end_y:
                    enemy_count += 1

        # If significant agent presence, report that instead of terrain
        min_presence = 3  # Minimum agents to count as "majority"
        if enemy_count >= min_presence and enemy_count > friendly_count:
            return 4  # majority-enemy
        if friendly_count >= min_presence and friendly_count > enemy_count:
            return 5  # majority-friend

        # Otherwise return terrain type
        return self._get_majority_terrain(position)

    def _apply_all_actions(
        self,
        actions: Dict[Tuple[str, int], int]
    ) -> None:
        """
        Apply actions to all units.

        Args:
            actions: Dict mapping (team, unit_id) to action index
        """
        blue_units = getattr(self.env, 'blue_units', [])
        red_units = getattr(self.env, 'red_units', [])

        blue_map = {unit.id: unit for unit in blue_units}
        red_map = {unit.id: unit for unit in red_units}

        for (team, unit_id), action in actions.items():
            if team == "blue" and unit_id in blue_map:
                self._apply_unit_action(blue_map[unit_id], action)
            elif team == "red" and unit_id in red_map:
                self._apply_unit_action(red_map[unit_id], action)

    def _apply_unit_action(self, unit, action: int) -> None:
        """
        Apply a discrete action to a unit.

        Actions set intermediate waypoints for smooth boids movement.
        The goal waypoint (for reward) is set externally via mouse/strategic level.
        This allows units to learn trajectories that avoid obstacles.

        Args:
            unit: Unit instance
            action: Discrete action index (0-15)
        """
        if action == 0:
            # Hold position - clear intermediate waypoint
            unit.clear_waypoint()

        elif 1 <= action <= 8:
            # Set intermediate waypoint in direction (N, W, E, S, NE, NW, SW, SE)
            offset = self.MOVE_OFFSETS[action]
            centroid = unit.centroid

            # Place intermediate waypoint in chosen direction
            new_x = centroid[0] + offset[0] * self.move_step
            new_y = centroid[1] + offset[1] * self.move_step

            # Clamp to grid bounds
            new_x = max(0.5, min(GRID_SIZE - 0.5, new_x))
            new_y = max(0.5, min(GRID_SIZE - 0.5, new_y))

            # Check if waypoint is walkable (not a building)
            terrain_grid = getattr(self.env, 'terrain_grid', None)
            if terrain_grid is not None:
                cell_x, cell_y = int(new_x), int(new_y)
                if not terrain_grid.is_walkable(cell_x, cell_y):
                    # Don't set waypoint inside building - try shorter step
                    new_x = centroid[0] + offset[0] * (self.move_step * 0.5)
                    new_y = centroid[1] + offset[1] * (self.move_step * 0.5)
                    new_x = max(0.5, min(GRID_SIZE - 0.5, new_x))
                    new_y = max(0.5, min(GRID_SIZE - 0.5, new_y))
                    cell_x, cell_y = int(new_x), int(new_y)
                    if not terrain_grid.is_walkable(cell_x, cell_y):
                        # Still blocked, don't set waypoint
                        return

            unit.set_waypoint(new_x, new_y)

        elif 9 <= action <= 12:
            # Dispatch to strategic cell (unused)
            pass

        elif action == 13:
            # Set stance to aggressive
            unit.stance = "aggressive"

        elif action == 14:
            # Set stance to defensive
            unit.stance = "defensive"

        elif action == 15:
            # Think - placeholder for logic programming reasoning
            # Currently a no-op; can be extended to invoke reasoning engine
            pass

    def _update_q_tables(
        self,
        rewards: Dict[Tuple[str, int], float],
        done: bool,
    ) -> None:
        """
        Update Q-tables using Q-learning update rule.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))

        Args:
            rewards: Dict mapping (team, unit_id) to reward
            done: Whether episode is done
        """
        # Get next observations (already updated)
        blue_next_obs = self.blue_observations
        red_next_obs = self.red_observations

        for (team, unit_id), reward in rewards.items():
            key = (team, unit_id)

            if key not in self._prev_obs or key not in self._prev_actions:
                continue

            state = self._prev_obs[key]
            action = self._prev_actions[key]

            # Get next state
            if team == "blue":
                next_state = blue_next_obs.get(unit_id, state)
            else:
                next_state = red_next_obs.get(unit_id, state)

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
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

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
        self.total_steps = data["total_steps"]

    @staticmethod
    def load_tables(filepath: str) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Load Q-tables and N-tables from file without creating wrapper.

        Args:
            filepath: Path to load file

        Returns:
            Tuple of (blue_q, red_q, blue_n, red_n)
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        return (
            data["blue_q_table"],
            data["red_q_table"],
            data["blue_n_table"],
            data["red_n_table"],
        )

    @staticmethod
    def save_tables(
        tables: Tuple[Dict, Dict, Dict, Dict],
        filepath: str,
        n_actions: int = 16,
    ) -> None:
        """
        Save Q-tables and N-tables to file without wrapper.

        Args:
            tables: Tuple of (blue_q, red_q, blue_n, red_n)
            filepath: Path to save file
            n_actions: Number of actions
        """
        blue_q, red_q, blue_n, red_n = tables
        data = {
            "blue_q_table": blue_q,
            "red_q_table": red_q,
            "blue_n_table": blue_n,
            "red_n_table": red_n,
            "total_steps": 1,
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "exploration_bonus": 1.0,
            "n_actions": n_actions,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def merge_tables(
        tables_list: List[Tuple[Dict, Dict, Dict, Dict]],
        n_actions: int = 16,
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Merge multiple Q-tables and N-tables into one.

        Uses weighted average based on visit counts.

        Args:
            tables_list: List of (blue_q, red_q, blue_n, red_n) tuples
            n_actions: Number of actions

        Returns:
            Merged (blue_q, red_q, blue_n, red_n)
        """
        # Merge blue tables
        merged_blue_q, merged_blue_n = QLearningWrapper._merge_team_tables(
            [(t[0], t[2]) for t in tables_list],
            n_actions
        )

        # Merge red tables
        merged_red_q, merged_red_n = QLearningWrapper._merge_team_tables(
            [(t[1], t[3]) for t in tables_list],
            n_actions
        )

        return merged_blue_q, merged_red_q, merged_blue_n, merged_red_n

    @staticmethod
    def _merge_team_tables(
        qn_pairs: List[Tuple[Dict, Dict]],
        n_actions: int,
    ) -> Tuple[Dict, Dict]:
        """
        Merge Q-tables and N-tables for one team.

        Args:
            qn_pairs: List of (q_table, n_table) pairs
            n_actions: Number of actions

        Returns:
            Merged (q_table, n_table)
        """
        merged_q: Dict[int, np.ndarray] = {}
        merged_n: Dict[int, np.ndarray] = {}

        # Collect all states
        all_states = set()
        for q_table, n_table in qn_pairs:
            all_states.update(q_table.keys())
            all_states.update(n_table.keys())

        for state in all_states:
            merged_q[state] = np.zeros(n_actions)
            merged_n[state] = np.zeros(n_actions)

            for q_table, n_table in qn_pairs:
                q_vals = q_table.get(state, np.zeros(n_actions))
                n_vals = n_table.get(state, np.zeros(n_actions))

                # Weighted sum for later averaging
                merged_q[state] += q_vals * n_vals
                merged_n[state] += n_vals

            # Convert weighted sum to weighted average
            mask = merged_n[state] > 0
            if mask.any():
                merged_q[state][mask] /= merged_n[state][mask]

        return merged_q, merged_n

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
        }
