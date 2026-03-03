"""
Movement Training Environment for teaching agents navigation.

This specialized environment focuses on movement training:
- Uses corridor map (grid pattern of walkable paths)
- Sets random waypoint on episode start
- Rewards agent for reducing Chebyshev distance to waypoint
- Terminates when agent reaches waypoint operational cell

Usage:
    >>> from movement_training_env import MovementTrainingEnv
    >>> env = MovementTrainingEnv(render_mode="human")
    >>> obs, info = env.reset()
    >>> # Agent must navigate to waypoint through corridors
"""

import random
from typing import Optional, Dict, Any, Tuple

import numpy as np

from combatenv import TacticalCombatEnv, EnvConfig, TerrainGrid, get_unit_for_agent
from combatenv.config import (
    GRID_SIZE, OPERATIONAL_GRID_SIZE, TACTICAL_CELLS_PER_OPERATIONAL,
)


class MovementTrainingEnv(TacticalCombatEnv):
    """
    Specialized environment for movement training.

    Features:
    - Corridor map (grid pattern through buildings)
    - Random waypoint set on each episode
    - Only one blue unit active (simplified)
    - Terminates when agent reaches waypoint operational cell
    - Reward shaping based on Chebyshev distance reduction

    Observation: Same as TacticalCombatEnv (89 floats)
    Action: Same as TacticalCombatEnv (4 floats: move_x, move_y, shoot, think)
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[EnvConfig] = None
    ):
        """
        Initialize the movement training environment.

        Args:
            render_mode: "human" for pygame rendering, None for headless
            config: Optional environment configuration
        """
        # Default config for movement training
        if config is None:
            config = EnvConfig(
                use_units=True,
                terminate_on_controlled_death=True,
                terminate_on_team_elimination=False,  # Don't care about team elimination
                max_steps=500,  # Shorter episodes for training
            )

        super().__init__(render_mode=render_mode, config=config)

        # Track previous Chebyshev distance for reward shaping
        self._prev_chebyshev: float = 1.0
        self._target_waypoint: Optional[Tuple[float, float]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment with corridor map and random waypoint.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (map_style is forced to 'corridors')

        Returns:
            Tuple of (observation, info)
        """
        # Force corridor map
        if options is None:
            options = {}

        # Generate corridor terrain
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        terrain_rng = random.Random()
        if seed is not None:
            terrain_rng.seed(seed)
        terrain_grid.generate_corridors(spawn_margin=5, rng=terrain_rng)
        options['terrain_grid'] = terrain_grid

        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)

        # Clear red team - movement training only uses blue
        self.red_agents = []
        self.red_units = []
        self.all_agents = list(self.blue_agents)
        self.alive_agents = [a for a in self.blue_agents if a.is_alive]

        # Set random waypoint for controlled agent's unit
        if self.config.use_units and self.controlled_agent is not None:
            waypoint = self._generate_random_waypoint(seed)
            self._target_waypoint = waypoint

            unit = get_unit_for_agent(self.controlled_agent, self.blue_units)
            if unit is not None:
                unit.set_waypoint(waypoint[0], waypoint[1])

        # Initialize previous Chebyshev distance
        self._prev_chebyshev = self._get_chebyshev_to_waypoint(self.controlled_agent)

        # Add waypoint info
        info['target_waypoint'] = self._target_waypoint
        info['chebyshev_distance'] = self._prev_chebyshev

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step with movement reward shaping.

        Args:
            action: numpy array [move_x, move_y, shoot, think]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # Calculate Chebyshev distance reward shaping
        if self.controlled_agent is not None and self.controlled_agent.is_alive:
            curr_chebyshev = self._get_chebyshev_to_waypoint(self.controlled_agent)

            # Reward for reducing distance
            distance_reward = (self._prev_chebyshev - curr_chebyshev) * 10.0

            # Bonus for reaching waypoint
            if curr_chebyshev == 0:
                distance_reward += 100.0
                terminated = True  # End episode on reaching waypoint

            # Small penalty for no progress
            if curr_chebyshev == self._prev_chebyshev:
                distance_reward -= 0.1

            reward += distance_reward
            self._prev_chebyshev = curr_chebyshev

            info['chebyshev_distance'] = curr_chebyshev

        return obs, reward, terminated, truncated, info

    def _generate_random_waypoint(self, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Generate a random waypoint position on the operational grid.

        Avoids placing waypoint too close to agent's starting position.

        Args:
            seed: Random seed

        Returns:
            (x, y) position in grid coordinates
        """
        rng = random.Random(seed)

        # Agent starts near (0, 0) corner typically
        # Generate waypoint in a different operational cell
        agent_pos = (0, 0)
        if self.controlled_agent is not None:
            agent_pos = self.controlled_agent.position
            agent_op_x = int(agent_pos[0] / TACTICAL_CELLS_PER_OPERATIONAL)
            agent_op_y = int(agent_pos[1] / TACTICAL_CELLS_PER_OPERATIONAL)
        else:
            agent_op_x, agent_op_y = 0, 0

        # Try to find a waypoint at least 3 operational cells away (Chebyshev)
        attempts = 0
        wp_op_x, wp_op_y = 0, 0
        while attempts < 50:
            wp_op_x = rng.randint(0, OPERATIONAL_GRID_SIZE - 1)
            wp_op_y = rng.randint(0, OPERATIONAL_GRID_SIZE - 1)

            chebyshev = max(abs(wp_op_x - agent_op_x), abs(wp_op_y - agent_op_y))
            if chebyshev >= 3:
                break
            attempts += 1

        # Convert operational cell to grid coordinates (random position within cell)
        x = wp_op_x * TACTICAL_CELLS_PER_OPERATIONAL + rng.random() * TACTICAL_CELLS_PER_OPERATIONAL
        y = wp_op_y * TACTICAL_CELLS_PER_OPERATIONAL + rng.random() * TACTICAL_CELLS_PER_OPERATIONAL

        return (x, y)


class DiscreteMovementTrainingEnv:
    """
    Movement training with discrete observation/action and circular Q-table.

    This wraps MovementTrainingEnv with standard wrappers:
    - EmptyTerrainWrapper: All terrain set to EMPTY
    - SingleAgentDiscreteObsWrapper: 2,048 discrete states
    - SingleAgentDiscreteActionWrapper: 8 discrete actions
    - ChebyshevRewardWrapper: Reward shaping for waypoint navigation
    - CircularQTable: Memory-limited Q-table

    Usage:
        env = DiscreteMovementTrainingEnv(render_mode="human", max_q_entries=500)
        obs, info = env.reset()

        # Training loop
        action = env.get_action(obs, epsilon=0.1)
        next_obs, reward, done, truncated, info = env.step(action)
        env.update_q(obs, action, reward, next_obs, done or truncated)
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[EnvConfig] = None,
        max_q_entries: int = 500,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        """
        Initialize discrete movement training environment.

        Args:
            render_mode: "human" for pygame, None for headless
            config: Environment configuration
            max_q_entries: Maximum entries in circular Q-table
            learning_rate: Q-learning alpha
            discount_factor: Q-learning gamma
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
        """
        from combatenv.wrappers import (
            EmptyTerrainWrapper,
            SingleAgentDiscreteObsWrapper,
            SingleAgentDiscreteActionWrapper,
            ChebyshevRewardWrapper,
        )
        from rl_student import CircularQTable

        # Create base environment
        self._base_env = MovementTrainingEnv(render_mode=render_mode, config=config)

        # Apply wrappers in order:
        # 1. EmptyTerrainWrapper - clear all terrain to EMPTY
        env = EmptyTerrainWrapper(self._base_env)

        # 2. ChebyshevRewardWrapper - reward shaping for waypoint navigation
        env = ChebyshevRewardWrapper(env)

        # 3. SingleAgentDiscreteObsWrapper - 2,048 discrete states
        env = SingleAgentDiscreteObsWrapper(env)

        # 4. SingleAgentDiscreteActionWrapper - 8 discrete actions
        self._env = SingleAgentDiscreteActionWrapper(env)

        # Create circular Q-table
        self.q_table = CircularQTable(
            max_entries=max_q_entries,
            n_actions=self._env.n_actions,  # 8 actions
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min
        )

        # Expose state/action sizes
        self.n_states = self._env.n_states  # 2,048
        self.n_actions = self._env.n_actions  # 8

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """Reset and return discrete observation."""
        return self._env.reset(seed=seed)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execute step with discrete action.

        Args:
            action: Discrete action index (0-7)

        Returns:
            Tuple of (discrete_obs, reward, terminated, truncated, info)
        """
        return self._env.step(action)

    def get_action(self, state: int, epsilon: Optional[float] = None, training: bool = True) -> int:
        """Get action from Q-table using epsilon-greedy."""
        return self.q_table.get_action(state, epsilon=epsilon, training=training)

    def update_q(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """Update Q-table and return TD error."""
        return self.q_table.update(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        """Decay epsilon after episode."""
        self.q_table.decay_epsilon()

    def render(self) -> None:
        """Render the environment."""
        self._base_env.render()

    def close(self) -> None:
        """Close the environment."""
        self._base_env.close()

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self.q_table.epsilon

    def get_statistics(self) -> Dict[str, Any]:
        """Get Q-table and training statistics."""
        return self.q_table.get_statistics()


class MultiAgentDiscreteEnv:
    """
    Multi-agent discrete environment with N agents, each with their own Q-table.

    This wraps TacticalCombatEnv with multi-agent wrappers:
    - MultiAgentWrapper: Control all N agents
    - EmptyTerrainWrapper: All terrain set to EMPTY
    - UnitChebyshevWrapper: Reward shaping for waypoint navigation (per unit centroid)
    - DiscreteObservationWrapper: Dict[int, int] observations
    - DiscreteActionWrapper: Dict[int, int] actions (10 discrete actions)

    Action Space (10 actions per agent):
        0: Think (invoke logic)
        1: North (no shoot)
        2: South (no shoot)
        3: East (no shoot)
        4: West (no shoot)
        5: North + Shoot
        6: South + Shoot
        7: East + Shoot
        8: West + Shoot
        9: No action

    Usage:
        env = MultiAgentDiscreteEnv(render_mode="human")
        observations, info = env.reset()  # Dict[int, int]

        for episode in range(100):
            actions = env.get_actions(observations)  # Dict[int, int]
            next_obs, rewards, done, truncated, info = env.step(actions)
            env.update(observations, actions, rewards, next_obs, done)
            observations = next_obs
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[EnvConfig] = None,
        max_q_entries: int = 500,
        max_steps: int = 10000,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        """
        Initialize multi-agent discrete environment.

        Args:
            render_mode: "human" for pygame, None for headless
            config: Environment configuration
            max_q_entries: Maximum entries per Q-table
            max_steps: Maximum steps per episode before truncation
            learning_rate: Q-learning alpha
            discount_factor: Q-learning gamma
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
        """
        from combatenv import TacticalCombatEnv
        from combatenv.wrappers import (
            MultiAgentWrapper,
            EmptyTerrainWrapper,
            ChebyshevRewardWrapper,
            UnitChebyshevWrapper,
            FOVCoverageRewardWrapper,
            DiscreteObservationWrapper,
            DiscreteActionWrapper,
            ActionMaskWrapper,
        )
        from rl_student.multi_agent_q import MultiAgentQManager

        # Default config - 8 units per team (8 agents each = 64 agents per team)
        if config is None:
            config = EnvConfig(
                use_units=True,
                terminate_on_controlled_death=False,
                terminate_on_team_elimination=True,
                max_steps=max_steps,
                num_agents_per_team=64,  # 8 units × 8 agents each
                num_units_per_team=8,    # 8 units per team
            )

        # Create base environment
        self._base_env = TacticalCombatEnv(render_mode=render_mode, config=config)

        # Apply wrappers in order:
        # 1. MultiAgentWrapper - control all agents
        env = MultiAgentWrapper(self._base_env)

        # 2. EmptyTerrainWrapper - clear all terrain to EMPTY
        env = EmptyTerrainWrapper(env)

        # 3. ChebyshevRewardWrapper - agent-based distance reward (max 0.5)
        env = ChebyshevRewardWrapper(env, distance_reward_scale=0.5, strategic_reward_scale=0.0)

        # 4. UnitChebyshevWrapper - unit centroid distance reward (max 0.5)
        env = UnitChebyshevWrapper(env, distance_reward_scale=0.5)

        # 5. FOVCoverageRewardWrapper - DISABLED
        # env = FOVCoverageRewardWrapper(env, coverage_bonus=0.1)

        # 5. DiscreteObservationWrapper - returns Dict[int, int]
        env = DiscreteObservationWrapper(env)

        # 6. DiscreteActionWrapper - takes Dict[int, int], 10 discrete actions
        env = DiscreteActionWrapper(env)

        # 7. ActionMaskWrapper - disable move+shoot actions (5, 6, 7, 8)
        # Move+shoot is disabled to simplify learning - agents learn to move OR shoot
        self._env = ActionMaskWrapper(env, disabled_actions=[5, 6, 7, 8])

        # Store config
        self._max_q_entries = max_q_entries
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min

        # Q-manager will be created after first reset (when we know n_agents)
        self.q_manager: Optional[MultiAgentQManager] = None

        # Expose state/action sizes
        self.n_states = self._env.n_states  # 8,192
        self.n_actions = self._env.n_actions  # 10

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, int], Dict[str, Any]]:
        """
        Reset and return discrete observations for all agents.

        Returns:
            observations: Dict[int, int] mapping agent_idx -> discrete state
            info: Environment info dictionary
        """
        observations, info = self._env.reset(seed=seed)

        # Spawn blue units at top edge (row 0), red at bottom edge (row 7)
        self._spawn_units_at_edges(seed)

        # Set all unit waypoints to center (32, 32)
        self._set_center_waypoints(seed)

        # Get fresh observations after spawning/waypoints are set
        # Need to re-query from the multi-agent wrapper
        fresh_obs = self._get_fresh_observations()

        # Reinitialize distance tracking in both Chebyshev wrappers
        self._reinit_distances()

        # Create Q-manager on first reset (once we know n_agents)
        if self.q_manager is None:
            from rl_student.multi_agent_q import MultiAgentQManager
            n_agents = len(observations)
            self.q_manager = MultiAgentQManager(
                n_agents=n_agents,
                n_states=self.n_states,
                n_actions=self.n_actions,
                max_q_entries=self._max_q_entries,
                learning_rate=self._learning_rate,
                discount_factor=self._discount_factor,
                epsilon=self._epsilon,
                epsilon_decay=self._epsilon_decay,
                epsilon_min=self._epsilon_min
            )

        return fresh_obs, info

    def step(
        self,
        actions: Dict[int, int]
    ) -> Tuple[Dict[int, int], Dict[int, float], bool, bool, Dict[str, Any]]:
        """
        Execute step with discrete actions for all agents.

        Args:
            actions: Dict[int, int] mapping agent_idx -> discrete action (0-9)

        Returns:
            observations: Dict[int, int] mapping agent_idx -> discrete state
            rewards: Dict[int, float] mapping agent_idx -> reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Environment info dictionary
        """
        return self._env.step(actions)

    def log_rewards_breakdown(self, rewards: Dict[int, float]) -> None:
        """
        Log reward breakdown for all units.

        Shows per-unit totals and component breakdown (Chebyshev, FOV coverage).

        Args:
            rewards: Dict[int, float] from step()
        """
        unit_size = 8
        num_blue_units = len(self._base_env.blue_units)
        num_red_units = len(self._base_env.red_units)
        num_blue_agents = num_blue_units * unit_size

        # Get FOV coverage wrapper for coverage info
        fov_wrapper = None
        env = self._env
        while env is not None:
            if type(env).__name__ == 'FOVCoverageRewardWrapper':
                fov_wrapper = env
                break
            env = getattr(env, 'env', None)

        # Calculate FOV coverages
        blue_coverages = {}
        red_coverages = {}
        if fov_wrapper:
            blue_coverages = fov_wrapper._calculate_unit_coverages(self._base_env.blue_units)
            red_coverages = fov_wrapper._calculate_unit_coverages(self._base_env.red_units)

        print("\n" + "=" * 60)
        print("REWARDS BREAKDOWN")
        print("=" * 60)

        # Blue units
        print("\nBLUE TEAM:")
        print("-" * 50)
        blue_total = 0.0
        for u in range(num_blue_units):
            start = u * unit_size
            unit_rewards = [rewards.get(i, 0.0) for i in range(start, start + unit_size)]
            unit_total = sum(unit_rewards)
            blue_total += unit_total
            coverage = blue_coverages.get(u, 0.0)
            fov_contrib = coverage * 0.1 * unit_size  # coverage_bonus * num_agents

            print(f"  U{u+1}: total={unit_total:+7.2f} | "
                  f"FOV={coverage:.0%} (+{fov_contrib:.2f}) | "
                  f"per_agent=[{unit_rewards[0]:+.2f}, {unit_rewards[1]:+.2f}, ...]")

        print(f"  BLUE TOTAL: {blue_total:+.2f}")

        # Red units
        print("\nRED TEAM:")
        print("-" * 50)
        red_total = 0.0
        for u in range(num_red_units):
            start = num_blue_agents + u * unit_size
            unit_rewards = [rewards.get(i, 0.0) for i in range(start, start + unit_size)]
            unit_total = sum(unit_rewards)
            red_total += unit_total
            coverage = red_coverages.get(u, 0.0)
            fov_contrib = coverage * 0.1 * unit_size

            print(f"  U{u+1}: total={unit_total:+7.2f} | "
                  f"FOV={coverage:.0%} (+{fov_contrib:.2f}) | "
                  f"per_agent=[{unit_rewards[0]:+.2f}, {unit_rewards[1]:+.2f}, ...]")

        print(f"  RED TOTAL: {red_total:+.2f}")

        print("\n" + "=" * 60)
        print(f"GRAND TOTAL: {blue_total + red_total:+.2f}")
        print("=" * 60 + "\n")

    def get_actions(
        self,
        observations: Dict[int, int],
        epsilon: Optional[float] = None,
        training: bool = True
    ) -> Dict[int, int]:
        """
        Get actions for all agents from their Q-tables.

        Args:
            observations: Dict[int, int] mapping agent_idx -> discrete state
            epsilon: Override epsilon (uses Q-manager's epsilon if None)
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Dict[int, int] mapping agent_idx -> discrete action
        """
        if self.q_manager is None:
            raise RuntimeError("Must call reset() before get_actions()")
        return self.q_manager.get_actions(observations, epsilon=epsilon, training=training)

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
            states: Dict[int, int] mapping agent_idx -> current state
            actions: Dict[int, int] mapping agent_idx -> action taken
            rewards: Dict[int, float] mapping agent_idx -> reward received
            next_states: Dict[int, int] mapping agent_idx -> next state
            done: Whether episode ended

        Returns:
            Dict[int, float] mapping agent_idx -> TD error
        """
        if self.q_manager is None:
            raise RuntimeError("Must call reset() before update()")
        return self.q_manager.update(states, actions, rewards, next_states, done)

    def decay_epsilon(self) -> None:
        """Decay epsilon for all Q-tables after episode."""
        if self.q_manager is not None:
            self.q_manager.decay_epsilon()

    def share_knowledge(self) -> Dict[str, Any]:
        """
        Share knowledge across agents after an episode.

        Returns:
            Dict with sharing statistics
        """
        if self.q_manager is not None:
            return self.q_manager.share_knowledge()
        return {"total_states": 0, "states_shared": 0, "updates_made": 0}

    def _set_center_waypoints(self, seed: Optional[int] = None) -> None:
        """
        Set all unit waypoints to center (32, 32).

        This creates a symmetric setup where both teams converge at the center.

        Args:
            seed: Random seed (unused, kept for API compatibility)
        """
        from combatenv.config import GRID_SIZE

        center = GRID_SIZE / 2  # 32

        for unit in self._base_env.blue_units:
            unit.set_waypoint(center, center)

        for unit in self._base_env.red_units:
            unit.set_waypoint(center, center)

    def _spawn_units_at_edges(self, seed: Optional[int] = None) -> None:
        """
        Spawn blue units in row 0 (top), red units in row 7 (bottom).

        Uses operational grid (8x8), 8 cells per row = 1 unit per cell.
        Blue units: y ~ 4 (center of row 0)
        Red units: y ~ 60 (center of row 7)

        Args:
            seed: Random seed for reproducibility
        """
        import random
        from combatenv.config import GRID_SIZE, OPERATIONAL_GRID_SIZE

        if seed is not None:
            random.seed(seed + 1000)

        op_cell_size = GRID_SIZE // OPERATIONAL_GRID_SIZE  # 64/8 = 8

        # Blue: row 0 (cells 0-7), y in [0, 8)
        for i, unit in enumerate(self._base_env.blue_units):
            cell_x = i % OPERATIONAL_GRID_SIZE  # 0-7
            center_x = (cell_x + 0.5) * op_cell_size
            center_y = 0.5 * op_cell_size  # Row 0 center (y=4)

            for agent in unit.agents:
                agent.position = (
                    center_x + random.uniform(-2, 2),
                    center_y + random.uniform(-2, 2)
                )

        # Red: row 7 (cells 56-63), y in [56, 64)
        for i, unit in enumerate(self._base_env.red_units):
            cell_x = i % OPERATIONAL_GRID_SIZE  # 0-7
            center_x = (cell_x + 0.5) * op_cell_size
            center_y = (OPERATIONAL_GRID_SIZE - 0.5) * op_cell_size  # Row 7 center (y=60)

            for agent in unit.agents:
                agent.position = (
                    center_x + random.uniform(-2, 2),
                    center_y + random.uniform(-2, 2)
                )

    def _get_fresh_observations(self) -> Dict[int, int]:
        """
        Get fresh observations after spawning/waypoint changes.

        Queries the multi-agent wrapper directly for current observations,
        then passes through the discrete observation wrapper.

        Returns:
            Dict[int, int] mapping agent_idx -> discrete state
        """
        # Navigate to find both wrappers in the chain
        env = self._env
        multi_agent_wrapper = None
        discrete_obs_wrapper = None

        # Walk down the chain to find wrappers
        while env is not None:
            # MultiAgentWrapper has _get_all_observations
            if hasattr(env, '_get_all_observations') and multi_agent_wrapper is None:
                multi_agent_wrapper = env
            # DiscreteObservationWrapper has observation() method
            if type(env).__name__ == 'DiscreteObservationWrapper' and discrete_obs_wrapper is None:
                discrete_obs_wrapper = env
            env = getattr(env, 'env', None)

        if multi_agent_wrapper is None:
            return {}

        # Get raw observations from multi-agent wrapper
        raw_obs = multi_agent_wrapper._get_all_observations()

        # Pass through discrete observation wrapper if present
        if discrete_obs_wrapper is not None:
            return discrete_obs_wrapper.observation(raw_obs)

        return raw_obs

    def _reinit_distances(self) -> None:
        """
        Reinitialize both Chebyshev wrappers' distance tracking.

        Must be called after spawning/waypoint changes to ensure correct initial distances.
        """
        env = self._env
        multi_agent_wrapper = None

        # Find wrappers and reinitialize
        while env is not None:
            # UnitChebyshevWrapper
            if hasattr(env, 'reinit_distances'):
                env.reinit_distances()
            # ChebyshevRewardWrapper
            if hasattr(env, '_init_distances_from_obs'):
                if multi_agent_wrapper is None:
                    # Find MultiAgentWrapper first
                    search_env = env
                    while search_env is not None:
                        if hasattr(search_env, '_get_all_observations'):
                            multi_agent_wrapper = search_env
                            break
                        search_env = getattr(search_env, 'env', None)
                if multi_agent_wrapper is not None:
                    raw_obs = multi_agent_wrapper._get_all_observations()
                    env._init_distances_from_obs(raw_obs)
            env = getattr(env, 'env', None)

    def render(self) -> bool:
        """
        Render the environment and process events.

        Returns:
            True if simulation should continue, False if user requested exit
        """
        # Render through wrapper chain so MultiAgentWrapper can pass rewards to debug panel
        self._env.render()
        return self._base_env.process_events()

    def close(self) -> None:
        """Close the environment."""
        self._base_env.close()

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        if self.q_manager is not None:
            return self.q_manager.epsilon
        return self._epsilon

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated Q-table statistics."""
        if self.q_manager is not None:
            return self.q_manager.get_statistics()
        return {"error": "Not initialized"}


if __name__ == "__main__":
    # Test discrete movement training environment
    print("Testing DiscreteMovementTrainingEnv...")

    env = DiscreteMovementTrainingEnv(render_mode="human", max_q_entries=100)
    obs, info = env.reset()

    print(f"Discrete state: {obs}")
    print(f"State space: {env.n_states}")
    print(f"Action space: {env.n_actions}")
    print(f"Target waypoint: {info.get('target_waypoint')}")

    # Training loop demo
    total_reward = 0.0
    for i in range(200):
        action = env.get_action(obs, epsilon=0.3)
        next_obs, reward, terminated, truncated, info = env.step(action)

        td_error = env.update_q(obs, action, reward, next_obs, terminated or truncated)
        total_reward += reward

        env.render()

        if terminated or truncated:
            print(f"Episode ended at step {i+1}, total reward: {total_reward:.2f}")
            break

        obs = next_obs

    print(f"\nQ-table statistics: {env.get_statistics()}")
    env.close()
