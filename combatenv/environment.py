"""
Gymnasium-compatible environment for the Grid-World Multi-Agent Tactical Simulation.

This module wraps the tactical combat simulation as a Gymnasium environment, enabling
integration with reinforcement learning libraries and standardized training loops.

Environment Details:
    - Observation: 89-float numpy array with structure:
        - Indices 0-9: Agent state (position, orientation, health, stamina, armor, ammo, etc.)
        - Indices 10-29: 5 nearest enemies (4 floats each: rel_x, rel_y, health, distance)
        - Indices 30-49: 5 nearest allies (4 floats each: rel_x, rel_y, health, distance)
        - Indices 50-87: Terrain types for up to 38 FOV cells (normalized 0-1)
            - EMPTY=0.0, OBSTACLE=0.25, FIRE=0.5, FOREST=0.75, WATER=1.0
        - Index 88: Chebyshev distance to waypoint (normalized 0-1, 0=at waypoint)
    - Action: Box(4,) continuous [move_x, move_y, shoot, think]
        - move_x, move_y: -1.0 to 1.0 (direction and speed)
        - shoot: 0.0 to 1.0 (fire if > 0.5)
        - think: 0.0 to 1.0 (invoke logical thinking if > 0.5, placeholder for logic programming)
    - Reward: Based on kills, damage, survival
    - Termination: When controlled agent dies or team eliminated

Wrapper Architecture:
    The environment supports a layered wrapper architecture for different abstraction levels:

    Base Environment (this class):
        - Core physics (projectiles, collisions)
        - Agent state (health, position, resources)
        - FOV calculations and spatial grid
        - Basic rendering

    Utility Wrappers (combatenv.wrappers):
        - KeybindingsWrapper: Keyboard/mouse input handling
        - DebugOverlayWrapper: Debug info rendering
        - TerminalLogWrapper: Console logging
        - TerrainGenWrapper: Terrain generation

    Tactical Level (agent control):
        - TacticalWrapper: Combat execution, terrain effects
        - DiscreteObservationWrapper: Q-learning state encoding
        - DiscreteActionWrapper: Discrete action mapping

    Operational Level (unit control):
        - OperationalWrapper: Unit management, boids steering
        - OperationalDiscreteObsWrapper: Unit-level states
        - OperationalDiscreteActionWrapper: Unit commands

    Strategic Level (grid control):
        - StrategicWrapper: 4x4 grid observations
        - StrategicDiscreteObsWrapper: Strategic states
        - StrategicDiscreteActionWrapper: Strategic commands

Usage:
    >>> from combatenv import TacticalCombatEnv
    >>> env = TacticalCombatEnv(render_mode="human")
    >>> obs, info = env.reset()
    >>> for _ in range(1000):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if terminated or truncated:
    ...         obs, info = env.reset()
    >>> env.close()

With Wrappers:
    >>> from combatenv import TacticalCombatEnv
    >>> from combatenv.wrappers import OperationalWrapper, StrategicWrapper
    >>> env = TacticalCombatEnv(render_mode="human")
    >>> env = OperationalWrapper(env)
    >>> env = StrategicWrapper(env)
"""

import math
import random
from typing import Optional, Tuple, Dict, Any, List, Set
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

from .config import (
    WINDOW_SIZE, FPS, CELL_SIZE,
    NEAR_FOV_ACCURACY, FAR_FOV_ACCURACY,
    NEAR_FOV_RANGE, NEAR_FOV_ANGLE,
    FAR_FOV_RANGE, FAR_FOV_ANGLE,
    MOVEMENT_ACCURACY_PENALTY,
    MUZZLE_FLASH_OFFSET, MUZZLE_FLASH_LIFETIME,
    RESPAWN_DELAY_SECONDS,
    NUM_AGENTS_PER_TEAM, GRID_SIZE,
    AGENT_MAX_HEALTH, AGENT_MAX_STAMINA, AGENT_MAX_ARMOR, AGENT_MAX_AMMO,
    AGENT_MOVE_SPEED, AGENT_ROTATION_SPEED,
    FIRE_DAMAGE_PER_STEP, FOREST_SPEED_MULTIPLIER,
    TERRAIN_OBSTACLE_PCT,
    NUM_UNITS_PER_TEAM, AGENTS_PER_UNIT,
    COHESION_SURVIVAL_BONUS, COHESION_ACCURACY_BONUS, COHESION_ARMOR_BONUS,
    UnitStance, WAYPOINT_DISTANCE_REWARD_SCALE, STANCE_COMPLIANCE_REWARD,
    MAX_WAYPOINT_REWARD_DISTANCE, AGENT_KILL_REWARD, UNIT_COHESION_RADIUS,
    OPERATIONAL_GRID_SIZE, TACTICAL_CELLS_PER_OPERATIONAL,
    OBS_SIZE as CONFIG_OBS_SIZE, ACTION_SIZE,
    FRIENDLY_FIRE_PENALTY,
)
from .agent import Agent, spawn_all_teams, try_shoot_at_visible_target
from .fov import get_layered_fov_overlap, get_fov_cells, get_fov_cache
from .renderer import render_all, render_debug_overlay, render_keybindings_overlay, render_terrain, render_waypoints, render_selected_unit, render_unit_rewards_overlay
from .spatial import SpatialGrid
from .unit import Unit, spawn_all_units, get_all_agents_from_units, get_unit_for_agent
from .boids import calculate_boids_steering, calculate_stance_steering, steering_to_orientation
from .projectile import Projectile
from .terrain import TerrainType, TerrainGrid


# Observation space constants
NUM_NEARBY_ENEMIES = 5
NUM_NEARBY_ALLIES = 5
MAX_FOV_CELLS = 38  # Maximum cells visible in combined near+far FOV
NUM_TERRAIN_TYPES = 5  # EMPTY, OBSTACLE, FIRE, FOREST, WATER
# OBS_SIZE = 10 + (NUM_NEARBY_ENEMIES * 4) + (NUM_NEARBY_ALLIES * 4) + MAX_FOV_CELLS + 1 = 89 floats
# Last float is Chebyshev distance to waypoint
OBS_SIZE = CONFIG_OBS_SIZE  # Use config value (89)


@dataclass
class EnvConfig:
    """Configuration for the tactical combat environment."""
    num_agents_per_team: int = NUM_AGENTS_PER_TEAM
    num_units_per_team: int = NUM_UNITS_PER_TEAM  # Number of units per team
    agents_per_unit: int = AGENTS_PER_UNIT  # Agents per unit
    respawn_enabled: bool = False  # Disabled for RL training
    respawn_delay: float = RESPAWN_DELAY_SECONDS
    max_steps: Optional[int] = 1000
    terminate_on_team_elimination: bool = True
    terminate_on_controlled_death: bool = True
    allow_escape_exit: bool = True  # Allow ESC key to exit simulation
    use_units: bool = True  # Enable unit-based spawning and waypoint control
    autonomous_combat: bool = True  # Enable autonomous shooting by all agents


class TacticalCombatEnv(gym.Env):
    """
    Gymnasium-compatible environment for tactical combat simulation.

    This environment wraps the multi-agent tactical combat simulation,
    providing a standard interface for reinforcement learning.

    The agent controls one blue team agent while all other agents
    act autonomously.

    Attributes:
        config: Environment configuration
        render_mode: "human" for visual rendering, None for headless
        observation_space: Box(88,) normalized floats (see module docstring for structure)
        action_space: Box(3,) for [move_x, move_y, shoot]
        show_fov: Whether FOV overlay is displayed (default: True)
        show_debug: Whether debug overlay is displayed (default: True)
    """

    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": FPS}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[EnvConfig] = None
    ):
        """
        Initialize the tactical combat environment.

        Args:
            render_mode: "human" for pygame rendering, None for headless
            config: Optional environment configuration
        """
        super().__init__()

        self.render_mode = render_mode
        self.config = config or EnvConfig()

        # Define action space: [move_x, move_y, shoot, think]
        # move_x, move_y: -1.0 to 1.0 (direction and speed)
        # shoot: 0.0 to 1.0 (fire if > 0.5)
        # think: 0.0 to 1.0 (invoke logical thinking if > 0.5)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Define observation space: normalized floats
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )

        # Pygame state (initialized on first render)
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._pygame_initialized = False

        # Game state (initialized on reset)
        self.blue_agents: List[Agent] = []
        self.red_agents: List[Agent] = []
        self.all_agents: List[Agent] = []
        self.alive_agents: List[Agent] = []
        self.spatial_grid: Optional[SpatialGrid] = None
        self.terrain_grid: Optional[TerrainGrid] = None
        self.projectiles: List[Projectile] = []
        self.muzzle_flashes: List[Tuple[Tuple[float, float], float]] = []
        self.respawn_timers: Dict[int, float] = {}

        # Unit management (when use_units=True)
        self.blue_units: List[Unit] = []
        self.red_units: List[Unit] = []

        # Controlled agent (first blue agent)
        self.controlled_agent: Optional[Agent] = None

        # Statistics
        self.blue_kills = 0
        self.red_kills = 0
        self.step_count = 0
        self.controlled_agent_kills = 0
        self.controlled_agent_damage_dealt = 0.0

        # UI state
        self.show_debug = False  # Debug overlay off by default (toggle with `)
        self.show_keybindings = False
        self.show_fov = True  # FOV visualization on by default
        self.show_strategic_grid = True  # Strategic 4x4 grid overlay
        self.show_rewards = False  # Unit rewards overlay (toggle with R)
        self._user_quit = False  # Set True when user requests exit via ESC or window close
        self.selected_unit_id: Optional[int] = None  # Currently selected blue unit for waypoint control
        self.selected_red_unit_id: Optional[int] = None  # Currently selected red unit (SHIFT+1-8)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options:
                - terrain_grid: Pre-made TerrainGrid to use instead of random generation
                - skip_auto_dispatch: If True, don't auto-dispatch units (for custom spawn positions)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Clear FOV cache from previous episode
        get_fov_cache().clear()

        # Seed the random module for deterministic spawning
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize terrain grid (use provided or generate random)
        if options and "terrain_grid" in options:
            self.terrain_grid = options["terrain_grid"]
        else:
            self.terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
            terrain_rng = random.Random()
            if seed is not None:
                terrain_rng.seed(seed)
            self.terrain_grid.generate_random(
                obstacle_pct=TERRAIN_OBSTACLE_PCT,
                rng=terrain_rng
            )

        # Spawn agents (conditional: units vs individual)
        if self.config.use_units:
            # Unit-based spawning
            self.blue_units, self.red_units = spawn_all_units(
                num_units_per_team=self.config.num_units_per_team,
                agents_per_unit=self.config.agents_per_unit,
                terrain_grid=self.terrain_grid
            )
            self.blue_agents = get_all_agents_from_units(self.blue_units)
            self.red_agents = get_all_agents_from_units(self.red_units)
        else:
            # Traditional individual spawning
            self.blue_agents, self.red_agents = spawn_all_teams(terrain_grid=self.terrain_grid)
            self.blue_units = []
            self.red_units = []

        self.all_agents = self.blue_agents + self.red_agents
        self.alive_agents = list(self.all_agents)

        # Set controlled agent (first blue agent)
        self.controlled_agent = self.blue_agents[0]

        # Auto-dispatch all units so they're on the battlefield
        # (skip if wrapper will handle custom spawn positions)
        skip_auto_dispatch = options.get("skip_auto_dispatch", False) if options else False
        if self.config.use_units and not skip_auto_dispatch:
            # Dispatch blue units toward center/enemy territory
            for unit in self.blue_units:
                if unit.in_reserve:
                    # Blue starts top-left, dispatch toward center
                    waypoint = (GRID_SIZE / 2, GRID_SIZE / 2)
                    unit.dispatch(waypoint)
            # Dispatch red units toward center/enemy territory
            for unit in self.red_units:
                if unit.in_reserve:
                    # Red starts bottom-right, dispatch toward center
                    waypoint = (GRID_SIZE / 2, GRID_SIZE / 2)
                    unit.dispatch(waypoint)

        # Initialize spatial grid
        self.spatial_grid = SpatialGrid(cell_size=2.0)

        # Clear game state
        self.projectiles = []
        self.muzzle_flashes = []
        self.respawn_timers = {}
        self.selected_unit_id = None  # Reset unit selection
        self.selected_red_unit_id = None  # Reset red unit selection

        # Reset statistics
        self.blue_kills = 0
        self.red_kills = 0
        self.step_count = 0
        self.controlled_agent_kills = 0
        self.controlled_agent_damage_dealt = 0.0

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: numpy array [move_x, move_y, shoot]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        dt = 1.0 / FPS
        self.step_count += 1

        # Update alive agents list and rebuild spatial grid
        self.alive_agents = [a for a in self.all_agents if a.is_alive]
        self.spatial_grid.build(self.alive_agents)

        # Update agent resources
        for agent in self.alive_agents:
            agent.update_cooldown(dt)
            agent.update_reload(dt)
            agent.update_stamina(dt, agent.is_moving)

        # Handle respawning (if enabled)
        if self.config.respawn_enabled:
            self._handle_respawns(dt)

        # Apply action to controlled agent
        reward = 0.0
        if self.controlled_agent and self.controlled_agent.is_alive:
            reward += self._apply_action(action, dt)

        # Autonomous combat for other agents (if enabled)
        if self.config.autonomous_combat:
            self._execute_autonomous_combat(dt)

        # Unit-based movement for agents with waypoints
        if self.config.use_units:
            self._update_unit_movement(dt)

        # Autonomous movement for other agents (skip stuck agents and unit-following agents)
        for agent in self.alive_agents:
            if agent is not self.controlled_agent:
                if not agent.is_stuck:
                    # Skip agents in reserve units (they don't move until dispatched)
                    if self._is_agent_in_reserve(agent):
                        continue
                    # Skip agents following unit waypoints (handled by _update_unit_movement)
                    if self.config.use_units and agent.following_unit:
                        unit = get_unit_for_agent(agent, self.blue_units + self.red_units)
                        if unit is not None and unit.waypoint is not None:
                            continue
                    agent.wander(dt=dt, other_agents=self.alive_agents, terrain_grid=self.terrain_grid)

        # Process terrain effects for all alive agents
        self._process_terrain_effects()

        # Update projectiles
        kills_this_step = self._update_projectiles(dt)
        reward += self._calculate_reward(kills_this_step)

        # Update muzzle flashes
        self.muzzle_flashes = [
            (pos, lifetime - dt)
            for pos, lifetime in self.muzzle_flashes
            if lifetime - dt > 0
        ]

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _apply_action(self, action: np.ndarray, dt: float) -> float:
        """
        Apply action to controlled agent.

        Args:
            action: [move_x, move_y, shoot, think]
            dt: Delta time

        Returns:
            Immediate reward from action
        """
        agent = self.controlled_agent
        if agent is None or not agent.is_alive:
            return 0.0

        # Handle both 3D (legacy) and 4D action spaces
        if len(action) >= 4:
            move_x, move_y, shoot, think = action[:4]
        else:
            move_x, move_y, shoot = action[:3]
            think = 0.0

        reward = 0.0

        # Handle logical thinking action
        if think > 0.5:
            suggested_action = self._invoke_logical_thinking(agent, self._get_obs())
            # For now, just log that thinking was triggered (future: blend/override action)
            _ = suggested_action  # Placeholder - will be used in future

        # Apply movement (only if not stuck)
        if not agent.is_stuck:
            move_magnitude = math.sqrt(move_x**2 + move_y**2)
            if move_magnitude > 0.1:  # Dead zone
                # Calculate target orientation from movement direction
                target_orientation = math.degrees(math.atan2(move_y, move_x))
                agent.orientation = target_orientation

                # Move with speed proportional to magnitude (capped at 1.0)
                speed = min(move_magnitude, 1.0) * AGENT_MOVE_SPEED
                agent.move_forward(speed=speed, dt=dt, other_agents=self.alive_agents, terrain_grid=self.terrain_grid)
                agent.wander_direction = 1  # Mark as moving
            else:
                agent.wander_direction = 0  # Not moving
        else:
            agent.wander_direction = 0  # Stuck, not moving

        # Apply shoot action
        if shoot > 0.5 and agent.can_shoot():
            nearby = self.spatial_grid.get_nearby_agents(agent)

            # Apply cohesion bonus to accuracy
            near_acc = self._get_accuracy_with_cohesion(agent, NEAR_FOV_ACCURACY)
            far_acc = self._get_accuracy_with_cohesion(agent, FAR_FOV_ACCURACY)

            projectile = try_shoot_at_visible_target(
                agent, nearby,
                near_acc, far_acc,
                MOVEMENT_ACCURACY_PENALTY,
                self.terrain_grid
            )
            if projectile:
                self.projectiles.append(projectile)
                # Add muzzle flash
                flash_x = agent.position[0] + math.cos(math.radians(agent.orientation)) * MUZZLE_FLASH_OFFSET
                flash_y = agent.position[1] + math.sin(math.radians(agent.orientation)) * MUZZLE_FLASH_OFFSET
                self.muzzle_flashes.append(((flash_x, flash_y), MUZZLE_FLASH_LIFETIME))

        return reward

    def _is_agent_cohesive(self, agent: Agent) -> bool:
        """
        Check if an agent is within cohesion radius of its unit's centroid.

        Args:
            agent: The agent to check

        Returns:
            True if agent is within UNIT_COHESION_RADIUS of unit centroid
        """
        if not self.config.use_units or agent.unit_id is None:
            return False

        all_units = self.blue_units + self.red_units
        unit = get_unit_for_agent(agent, all_units)
        if unit is None:
            return False

        centroid = unit.centroid
        dist_sq = (agent.position[0] - centroid[0])**2 + (agent.position[1] - centroid[1])**2
        return dist_sq <= UNIT_COHESION_RADIUS**2

    def _get_accuracy_with_cohesion(self, agent: Agent, base_accuracy: float) -> float:
        """
        Get accuracy with cohesion bonus applied if applicable.

        Args:
            agent: The shooting agent
            base_accuracy: Base accuracy value

        Returns:
            Accuracy multiplied by cohesion bonus if agent is cohesive
        """
        if self._is_agent_cohesive(agent):
            return base_accuracy * (1.0 + COHESION_ACCURACY_BONUS)
        return base_accuracy

    def _get_chebyshev_to_waypoint(self, agent: Agent) -> float:
        """
        Calculate Chebyshev distance from agent's operational cell to waypoint's operational cell.

        The operational grid is 8x8, so max Chebyshev distance is 7.
        Returns normalized value in [0, 1] by dividing by 7.

        Args:
            agent: The agent to calculate distance for

        Returns:
            Normalized Chebyshev distance (0-1), or 1.0 if no waypoint set
        """
        if not self.config.use_units or agent.unit_id is None:
            return 1.0  # No unit = max distance

        all_units = self.blue_units + self.red_units
        unit = get_unit_for_agent(agent, all_units)
        if unit is None or unit.waypoint is None:
            return 1.0  # No waypoint = max distance

        # Get agent's operational cell (0-7, 0-7)
        agent_op_x = int(agent.position[0] / TACTICAL_CELLS_PER_OPERATIONAL)
        agent_op_y = int(agent.position[1] / TACTICAL_CELLS_PER_OPERATIONAL)

        # Get waypoint's operational cell (0-7, 0-7)
        wp_op_x = int(unit.waypoint[0] / TACTICAL_CELLS_PER_OPERATIONAL)
        wp_op_y = int(unit.waypoint[1] / TACTICAL_CELLS_PER_OPERATIONAL)

        # Clamp to valid range
        agent_op_x = max(0, min(agent_op_x, OPERATIONAL_GRID_SIZE - 1))
        agent_op_y = max(0, min(agent_op_y, OPERATIONAL_GRID_SIZE - 1))
        wp_op_x = max(0, min(wp_op_x, OPERATIONAL_GRID_SIZE - 1))
        wp_op_y = max(0, min(wp_op_y, OPERATIONAL_GRID_SIZE - 1))

        # Chebyshev distance = max(|dx|, |dy|)
        chebyshev = max(abs(agent_op_x - wp_op_x), abs(agent_op_y - wp_op_y))

        # Normalize by max possible distance (7 for 8x8 grid)
        return chebyshev / (OPERATIONAL_GRID_SIZE - 1)

    def _invoke_logical_thinking(self, agent: Agent, observation: np.ndarray) -> np.ndarray:
        """
        Invoke logical thinking to get a suggested action.

        This is a placeholder for future logic programming integration.
        Currently returns a no-op action.

        Args:
            agent: The agent requesting logical thinking
            observation: Current observation vector

        Returns:
            Suggested action as numpy array [move_x, move_y, shoot, think]
        """
        # Placeholder: return no-op action
        # Future: integrate Prolog/pyswip for strategic reasoning
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _is_agent_in_reserve(self, agent: Agent) -> bool:
        """Check if agent belongs to a unit that is in reserve."""
        if not self.config.use_units or not agent.following_unit:
            return False
        all_units = self.blue_units + self.red_units
        unit = get_unit_for_agent(agent, all_units)
        return unit is not None and unit.in_reserve

    def _execute_autonomous_combat(self, dt: float) -> None:
        """Execute combat for autonomous agents (not controlled)."""
        for agent in self.alive_agents:
            if agent is self.controlled_agent:
                continue

            # Skip agents in reserve units
            if self._is_agent_in_reserve(agent):
                continue

            nearby = self.spatial_grid.get_nearby_agents(agent)

            # Apply cohesion bonus to accuracy
            near_acc = self._get_accuracy_with_cohesion(agent, NEAR_FOV_ACCURACY)
            far_acc = self._get_accuracy_with_cohesion(agent, FAR_FOV_ACCURACY)

            projectile = try_shoot_at_visible_target(
                agent, nearby,
                near_acc, far_acc,
                MOVEMENT_ACCURACY_PENALTY,
                self.terrain_grid
            )

            if projectile:
                self.projectiles.append(projectile)
                flash_x = agent.position[0] + math.cos(math.radians(agent.orientation)) * MUZZLE_FLASH_OFFSET
                flash_y = agent.position[1] + math.sin(math.radians(agent.orientation)) * MUZZLE_FLASH_OFFSET
                self.muzzle_flashes.append(((flash_x, flash_y), MUZZLE_FLASH_LIFETIME))

    def _update_projectiles(self, dt: float) -> Dict[str, int]:
        """Update projectiles and handle collisions."""
        projectiles_to_remove: Set[int] = set()
        kills_this_step = {'blue': 0, 'red': 0, 'controlled': 0, 'friendly_fire': 0}

        for i, projectile in enumerate(self.projectiles):
            if projectile.update(dt, self.terrain_grid):
                projectiles_to_remove.add(i)
                continue

            for agent in self.alive_agents:
                if projectile.check_collision(agent):
                    was_alive = agent.is_alive

                    # Check for friendly fire (shooter hits teammate)
                    is_friendly_fire = False
                    if projectile.owner_team == "blue" and agent in self.blue_agents:
                        is_friendly_fire = True
                    elif projectile.owner_team == "red" and agent in self.red_agents:
                        is_friendly_fire = True

                    # Track friendly fire by controlled agent
                    if is_friendly_fire and projectile.shooter_id == id(self.controlled_agent):
                        kills_this_step['friendly_fire'] += 1

                    # Apply cohesion armor bonus (reduce damage if target is cohesive)
                    damage = projectile.damage
                    if self._is_agent_cohesive(agent):
                        damage = int(damage * (1.0 - COHESION_ARMOR_BONUS))

                    agent.take_damage(damage)

                    if was_alive and not agent.is_alive:
                        # Clear dead agent from FOV cache
                        get_fov_cache().remove_agent(id(agent))

                        # Track kill on the shooter agent
                        for shooter in self.all_agents:
                            if id(shooter) == projectile.shooter_id:
                                shooter.kills += 1
                                break

                        if projectile.owner_team == "blue":
                            self.blue_kills += 1
                            kills_this_step['blue'] += 1
                            # Check if controlled agent got the kill
                            if projectile.shooter_id == id(self.controlled_agent):
                                self.controlled_agent_kills += 1
                                kills_this_step['controlled'] += 1
                        else:
                            self.red_kills += 1
                            kills_this_step['red'] += 1

                    # Track damage dealt by controlled agent
                    if projectile.shooter_id == id(self.controlled_agent):
                        self.controlled_agent_damage_dealt += projectile.damage

                    projectiles_to_remove.add(i)
                    break

        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        return kills_this_step

    def _get_obs(self) -> np.ndarray:
        """
        Get observation for controlled agent.

        Returns:
            Normalized numpy array of shape (OBS_SIZE,) with values in [0, 1]
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        agent = self.controlled_agent
        if agent is None or not agent.is_alive:
            return obs

        # Agent state (indices 0-9) - all clipped to [0, 1]
        obs[0] = np.clip(agent.position[0] / GRID_SIZE, 0, 1)
        obs[1] = np.clip(agent.position[1] / GRID_SIZE, 0, 1)
        obs[2] = np.clip((agent.orientation % 360) / 360.0, 0, 1)
        obs[3] = np.clip(agent.health / AGENT_MAX_HEALTH, 0, 1)
        obs[4] = np.clip(agent.stamina / AGENT_MAX_STAMINA, 0, 1)
        obs[5] = np.clip(agent.armor / AGENT_MAX_ARMOR, 0, 1)
        obs[6] = np.clip(agent.ammo_reserve / AGENT_MAX_AMMO, 0, 1)
        obs[7] = np.clip(agent.magazine_ammo / 30.0, 0, 1)  # Magazine size
        obs[8] = 1.0 if agent.can_shoot() else 0.0
        obs[9] = 1.0 if agent.is_reloading else 0.0

        # Get nearby enemies (indices 10-29)
        enemies = [(e, self._distance_to(agent, e)) for e in self.red_agents if e.is_alive]
        enemies.sort(key=lambda x: x[1])

        for i, (enemy, dist) in enumerate(enemies[:NUM_NEARBY_ENEMIES]):
            base_idx = 10 + i * 4
            # Relative position (normalized and clipped)
            rel_x = (enemy.position[0] - agent.position[0]) / GRID_SIZE + 0.5
            rel_y = (enemy.position[1] - agent.position[1]) / GRID_SIZE + 0.5
            obs[base_idx] = np.clip(rel_x, 0, 1)
            obs[base_idx + 1] = np.clip(rel_y, 0, 1)
            obs[base_idx + 2] = np.clip(enemy.health / AGENT_MAX_HEALTH, 0, 1)
            obs[base_idx + 3] = np.clip(dist / GRID_SIZE, 0, 1)

        # Get nearby allies (indices 30-49)
        allies = [(a, self._distance_to(agent, a)) for a in self.blue_agents if a.is_alive and a is not agent]
        allies.sort(key=lambda x: x[1])

        for i, (ally, dist) in enumerate(allies[:NUM_NEARBY_ALLIES]):
            base_idx = 30 + i * 4
            rel_x = (ally.position[0] - agent.position[0]) / GRID_SIZE + 0.5
            rel_y = (ally.position[1] - agent.position[1]) / GRID_SIZE + 0.5
            obs[base_idx] = np.clip(rel_x, 0, 1)
            obs[base_idx + 1] = np.clip(rel_y, 0, 1)
            obs[base_idx + 2] = np.clip(ally.health / AGENT_MAX_HEALTH, 0, 1)
            obs[base_idx + 3] = np.clip(dist / GRID_SIZE, 0, 1)

        # Get terrain types for FOV cells (indices 50-87)
        # Use far FOV range/angle to get all visible cells
        fov_cells = get_fov_cells(
            agent.position,
            agent.orientation,
            fov_angle=FAR_FOV_ANGLE,
            max_range=FAR_FOV_RANGE,
            terrain_grid=self.terrain_grid
        )

        # Sort cells by distance for consistent ordering
        agent_x, agent_y = agent.position
        sorted_cells = sorted(
            fov_cells,
            key=lambda c: (c[0] - agent_x) ** 2 + (c[1] - agent_y) ** 2
        )

        # Add terrain type for each cell (block majority, normalized to [0, 1])
        for i, (cx, cy) in enumerate(sorted_cells[:MAX_FOV_CELLS]):
            terrain_type = self.terrain_grid.get_block_majority(cx, cy)
            obs[50 + i] = terrain_type / (NUM_TERRAIN_TYPES - 1)  # Normalize to [0, 1]

        # Chebyshev distance to waypoint (index 88)
        # Normalized [0, 1] where 0 = at waypoint, 1 = max distance (7 cells on operational grid)
        obs[88] = self._get_chebyshev_to_waypoint(agent)

        return obs

    def _distance_to(self, agent1: Agent, agent2: Agent) -> float:
        """Calculate distance between two agents."""
        dx = agent1.position[0] - agent2.position[0]
        dy = agent1.position[1] - agent2.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information about environment state."""
        return {
            'blue_kills': self.blue_kills,
            'red_kills': self.red_kills,
            'blue_alive': sum(1 for a in self.blue_agents if a.is_alive),
            'red_alive': sum(1 for a in self.red_agents if a.is_alive),
            'controlled_alive': self.controlled_agent.is_alive if self.controlled_agent else False,
            'controlled_kills': self.controlled_agent_kills,
            'controlled_damage': self.controlled_agent_damage_dealt,
            'step_count': self.step_count,
        }

    def _calculate_reward(self, kills_this_step: Dict[str, int]) -> float:
        """
        Calculate reward for the current step.

        Rewards:
        - +AGENT_KILL_REWARD per kill by controlled agent
        - +0.01 per step survived (with cohesion bonus if applicable)
        - Cohesion survival bonus: +10% if within unit cohesion radius
        - Waypoint distance reward: Based on proximity to waypoint
        - Stance compliance reward: Bonus for following stance behavior

        Penalties:
        - FRIENDLY_FIRE_PENALTY per friendly fire hit by controlled agent
        """
        reward = 0.0

        # Reward for kills by controlled agent
        reward += kills_this_step.get('controlled', 0) * AGENT_KILL_REWARD

        # Penalty for friendly fire by controlled agent
        reward += kills_this_step.get('friendly_fire', 0) * FRIENDLY_FIRE_PENALTY

        # Survival reward (with cohesion bonus)
        if self.controlled_agent and self.controlled_agent.is_alive:
            survival_reward = 0.01
            if self._is_agent_cohesive(self.controlled_agent):
                survival_reward *= (1.0 + COHESION_SURVIVAL_BONUS)
            reward += survival_reward

            # Waypoint distance and stance compliance rewards (only if using units)
            if self.config.use_units and self.controlled_agent.unit_id is not None:
                unit = get_unit_for_agent(self.controlled_agent, self.blue_units)
                if unit is not None and unit.waypoint is not None:
                    # Waypoint distance reward (inversely proportional to distance)
                    dist = math.sqrt(
                        (self.controlled_agent.position[0] - unit.waypoint[0])**2 +
                        (self.controlled_agent.position[1] - unit.waypoint[1])**2
                    )
                    waypoint_reward = max(0, 1 - dist / MAX_WAYPOINT_REWARD_DISTANCE) * WAYPOINT_DISTANCE_REWARD_SCALE
                    reward += waypoint_reward

                    # Stance compliance reward
                    stance = unit.stance
                    if stance == UnitStance.AGGRESSIVE:
                        # Bonus if moving toward enemy
                        enemies = self.red_agents
                        if enemies:
                            # Find nearest enemy
                            nearest_dist = float('inf')
                            for enemy in enemies:
                                if enemy.is_alive:
                                    dx = enemy.position[0] - self.controlled_agent.position[0]
                                    dy = enemy.position[1] - self.controlled_agent.position[1]
                                    enemy_dist = math.sqrt(dx*dx + dy*dy)
                                    nearest_dist = min(nearest_dist, enemy_dist)

                            # Check if facing toward enemies (moving toward them)
                            if nearest_dist < MAX_WAYPOINT_REWARD_DISTANCE and self.controlled_agent.is_moving:
                                reward += STANCE_COMPLIANCE_REWARD

                    elif stance == UnitStance.DEFENSIVE:
                        # Bonus if near waypoint
                        if dist <= UNIT_COHESION_RADIUS:
                            reward += STANCE_COMPLIANCE_REWARD

                    elif stance == UnitStance.PATROL:
                        # Bonus if moving toward waypoint
                        if self.controlled_agent.is_moving and dist > 0.5:
                            reward += STANCE_COMPLIANCE_REWARD

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if controlled agent dies
        if self.config.terminate_on_controlled_death:
            if self.controlled_agent and not self.controlled_agent.is_alive:
                return True

        # Terminate if either team is eliminated
        if self.config.terminate_on_team_elimination:
            living_blue = sum(1 for a in self.blue_agents if a.is_alive)
            living_red = sum(1 for a in self.red_agents if a.is_alive)
            if living_blue == 0 or living_red == 0:
                return True

        return False

    def _check_truncated(self) -> bool:
        """Check if episode should be truncated (step limit)."""
        if self.config.max_steps is None:
            return False
        return self.step_count >= self.config.max_steps

    def _process_terrain_effects(self) -> None:
        """Process terrain effects based on block majority terrain type."""
        for agent in self.alive_agents:
            cell = agent.get_grid_position()
            terrain = self.terrain_grid.get_block_majority(*cell)

            if terrain == TerrainType.FIRE:
                # Fire damage bypasses armor
                agent.apply_terrain_damage(FIRE_DAMAGE_PER_STEP)

            # Track terrain effects (affects speed, detection, and shooting)
            agent.in_forest = (terrain == TerrainType.FOREST)
            agent.in_water = (terrain == TerrainType.WATER)

    def _handle_respawns(self, dt: float) -> None:
        """Handle agent respawning after death."""
        for agent in self.all_agents:
            if not agent.is_alive:
                agent_id = id(agent)
                if agent_id not in self.respawn_timers:
                    self.respawn_timers[agent_id] = self.config.respawn_delay
                else:
                    self.respawn_timers[agent_id] -= dt
                    if self.respawn_timers[agent_id] <= 0:
                        agent.respawn()
                        del self.respawn_timers[agent_id]

    def _update_unit_movement(self, dt: float) -> None:
        """
        Apply boids steering to agents that belong to units with active waypoints.

        For each agent in a unit with a waypoint, calculate boids steering force
        and apply it as the agent's movement direction instead of random wandering.
        Stance affects steering behavior:
        - AGGRESSIVE: Target nearest enemy
        - DEFENSIVE: Stronger cohesion, reduced waypoint force
        - PATROL: Normal boids behavior

        Args:
            dt: Delta time in seconds
        """
        if not self.config.use_units:
            return

        all_units = self.blue_units + self.red_units

        for agent in self.alive_agents:
            # Skip controlled agent (handled by _apply_action)
            if agent is self.controlled_agent:
                continue

            # Skip agents that are stuck
            if agent.is_stuck:
                continue

            # Skip agents not following unit
            if not agent.following_unit:
                continue

            # Find agent's unit
            unit = get_unit_for_agent(agent, all_units)
            if unit is None or unit.waypoint is None:
                continue

            # Skip units in reserve
            if unit.in_reserve:
                continue

            # Get enemies for stance-aware steering (AGGRESSIVE targets enemies)
            enemies = self.red_agents if agent.team == "blue" else self.blue_agents

            # Calculate stance-aware boids steering
            steering = calculate_stance_steering(agent, unit, enemies=enemies)
            target_orientation = steering_to_orientation(steering)

            if target_orientation is not None:
                # Smoothly rotate toward steering direction
                angle_diff = (target_orientation - agent.orientation + 180) % 360 - 180
                max_rotation = AGENT_ROTATION_SPEED * dt

                if abs(angle_diff) > max_rotation:
                    if angle_diff > 0:
                        agent.orientation = (agent.orientation + max_rotation) % 360
                    else:
                        agent.orientation = (agent.orientation - max_rotation) % 360
                else:
                    agent.orientation = target_orientation

                # Move forward in steering direction
                agent.move_forward(dt=dt, other_agents=self.alive_agents, terrain_grid=self.terrain_grid)
                agent.wander_direction = 1  # Mark as moving
            else:
                agent.wander_direction = 0  # Not moving

    def set_unit_waypoint(self, unit_id: int, team: str, x: float, y: float) -> bool:
        """
        Set a waypoint for a specific unit.

        Args:
            unit_id: The unit's ID (0 to NUM_UNITS_PER_TEAM-1)
            team: Team affiliation ("blue" or "red")
            x: X coordinate in grid space
            y: Y coordinate in grid space

        Returns:
            True if waypoint was set, False if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                unit.set_waypoint(x, y)
                return True
        return False

    def dispatch_unit(self, unit_id: int, team: str, x: float, y: float) -> bool:
        """
        Dispatch a unit from reserve - releases it and sets its waypoint.

        Args:
            unit_id: The unit's ID (0 to NUM_UNITS_PER_TEAM-1)
            team: Team affiliation ("blue" or "red")
            x: X coordinate in grid space for waypoint
            y: Y coordinate in grid space for waypoint

        Returns:
            True if unit was dispatched, False if unit not found or already dispatched
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                if not unit.in_reserve:
                    return False  # Already dispatched
                unit.dispatch((x, y))
                return True
        return False

    def get_next_reserve_unit(self, team: str) -> Optional[int]:
        """
        Get the ID of the next unit in reserve for a team.

        Args:
            team: Team affiliation ("blue" or "red")

        Returns:
            Unit ID of next reserve unit, or None if all units dispatched
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.in_reserve:
                return unit.id
        return None

    def clear_unit_waypoint(self, unit_id: int, team: str) -> bool:
        """
        Clear the waypoint for a specific unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            True if waypoint was cleared, False if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                unit.clear_waypoint()
                return True
        return False

    def clear_all_waypoints(self, team: Optional[str] = None) -> None:
        """
        Clear all waypoints for a team or both teams.

        Args:
            team: "blue", "red", or None for both teams
        """
        if team is None or team == "blue":
            for unit in self.blue_units:
                unit.clear_waypoint()
        if team is None or team == "red":
            for unit in self.red_units:
                unit.clear_waypoint()

    def get_unit_by_id(self, unit_id: int, team: str) -> Optional[Unit]:
        """
        Get a unit by its ID and team.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            Unit if found, None otherwise
        """
        units = self.blue_units if team == "blue" else self.red_units
        for unit in units:
            if unit.id == unit_id:
                return unit
        return None

    def add_unit_waypoint(self, unit_id: int, team: str, x: float, y: float) -> bool:
        """
        Append a waypoint to a unit's waypoint sequence.

        Args:
            unit_id: The unit's ID (0 to NUM_UNITS_PER_TEAM-1)
            team: Team affiliation ("blue" or "red")
            x: X coordinate in grid space
            y: Y coordinate in grid space

        Returns:
            True if waypoint was added, False if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                unit.add_waypoint(x, y)
                return True
        return False

    def advance_unit_waypoint(self, unit_id: int, team: str) -> bool:
        """
        Advance a unit to its next waypoint in the sequence.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            True if advanced to next waypoint, False if at end or unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                return unit.advance_waypoint()
        return False

    def get_unit_waypoints(self, unit_id: int, team: str) -> List[Tuple[float, float]]:
        """
        Get all waypoints for a unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            List of (x, y) waypoint positions, empty list if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                return list(unit.waypoints)
        return []

    def set_unit_stance(self, unit_id: int, team: str, stance: UnitStance) -> bool:
        """
        Set the stance for a unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")
            stance: UnitStance (AGGRESSIVE, DEFENSIVE, PATROL)

        Returns:
            True if stance was set, False if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                unit.stance = stance
                return True
        return False

    def get_unit_stance(self, unit_id: int, team: str) -> Optional[UnitStance]:
        """
        Get the current stance for a unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            UnitStance if unit found, None otherwise
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                return unit.stance
        return None

    def process_events(self) -> bool:
        """
        Process pygame events (window close, keyboard input).

        This method handles user input for the simulation:
        - Window close (X button): Returns False
        - ESC key: Returns False (if allow_escape_exit is True in config)
        - Backtick (`): Toggle debug overlay
        - ? (Shift+/): Toggle keybindings help

        Returns:
            True if simulation should continue, False if user requested exit

        Note:
            This method only processes events when render_mode is not None.
            In headless mode, it always returns True.
        """
        # In headless mode, no events to process
        if self.render_mode is None:
            return not self._user_quit

        # Initialize pygame if needed
        if not self._pygame_initialized:
            self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._user_quit = True
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Left click to set/add waypoint for selected unit
                if self.config.use_units and self.selected_unit_id is not None:
                    mouse_x, mouse_y = event.pos
                    grid_x = mouse_x / CELL_SIZE
                    grid_y = mouse_y / CELL_SIZE
                    # Shift+click adds to waypoint sequence, regular click sets single waypoint
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        self.add_unit_waypoint(self.selected_unit_id, "blue", grid_x, grid_y)
                    else:
                        self.set_unit_waypoint(self.selected_unit_id, "blue", grid_x, grid_y)

        return not self._user_quit

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        """
        Handle keyboard key press events.

        Key bindings:
            - Shift+Q: Quit simulation (if allow_escape_exit is True)
            - ` (backtick): Toggle debug overlay
            - ? (Shift+/): Toggle keybindings help
            - F: Toggle FOV overlay
            - G: Toggle strategic grid overlay
            - 1-8: Select blue unit
            - SHIFT+1-8: Select red unit
            - SPACE: Clear selected/all waypoints

        Args:
            event: The pygame KEYDOWN event
        """
        if event.key == pygame.K_q and (event.mod & pygame.KMOD_SHIFT):
            if self.config.allow_escape_exit:
                self._user_quit = True
        elif event.key == pygame.K_BACKQUOTE:
            self.show_debug = not self.show_debug
        elif event.key == pygame.K_SLASH and (event.mod & pygame.KMOD_SHIFT):
            self.show_keybindings = not self.show_keybindings
        elif event.key == pygame.K_f:
            self.show_fov = not self.show_fov
        elif event.key == pygame.K_g:
            self.show_strategic_grid = not self.show_strategic_grid
        elif event.key == pygame.K_r:
            self.show_rewards = not self.show_rewards
        # Unit selection keys (1-8, only when units are enabled)
        # SHIFT+1-8 selects red units, 1-8 selects blue units
        elif self.config.use_units and event.key in (
            pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
            pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8
        ):
            unit_idx = event.key - pygame.K_1  # K_1 -> 0, K_2 -> 1, etc.
            if event.mod & pygame.KMOD_SHIFT:
                # SHIFT+1-8: Select red unit
                self.selected_red_unit_id = unit_idx
                self.selected_unit_id = None  # Deselect blue unit
            else:
                # 1-8: Select blue unit
                self.selected_unit_id = unit_idx
                self.selected_red_unit_id = None  # Deselect red unit
        # Clear waypoints
        elif self.config.use_units and event.key == pygame.K_SPACE:
            if self.selected_unit_id is not None:
                self.clear_unit_waypoint(self.selected_unit_id, "blue")
            else:
                self.clear_all_waypoints("blue")
        # Advance to next waypoint in sequence
        elif self.config.use_units and event.key == pygame.K_RETURN:
            if self.selected_unit_id is not None:
                self.advance_unit_waypoint(self.selected_unit_id, "blue")

    def render(self) -> Optional[np.ndarray]:
        """
        Render the current environment state.

        Returns:
            None for human mode, RGB array for rgb_array mode
        """
        if self.render_mode is None:
            return None

        # Initialize pygame on first render
        if not self._pygame_initialized:
            self._init_pygame()

        # Calculate FOV for visualization (only if enabled - expensive operation)
        living_blue = [a for a in self.blue_agents if a.is_alive]
        living_red = [a for a in self.red_agents if a.is_alive]

        if self.show_fov:
            blue_near, blue_far, red_near, red_far, overlap_nn, overlap_mix, overlap_ff = get_layered_fov_overlap(
                living_blue, living_red,
                NEAR_FOV_RANGE, NEAR_FOV_ANGLE,
                FAR_FOV_RANGE, FAR_FOV_ANGLE,
                self.terrain_grid
            )
        else:
            # Empty sets when FOV visualization is disabled
            blue_near = blue_far = red_near = red_far = set()
            overlap_nn = overlap_mix = overlap_ff = set()

        # Render everything
        render_all(
            self._screen, self.blue_agents, self.red_agents,
            blue_near, blue_far, red_near, red_far,
            overlap_nn, overlap_mix, overlap_ff,
            self.projectiles, self.muzzle_flashes,
            self.terrain_grid,
            self.show_strategic_grid
        )

        # Render overlays
        if self.show_debug:
            debug_info = {
                'fps': self._clock.get_fps() if self._clock else FPS,
                'num_agents': len(self.all_agents),
                'blue_agents': len(living_blue),
                'red_agents': len(living_red),
                'blue_dead': len(self.blue_agents) - len(living_blue),
                'red_dead': len(self.red_agents) - len(living_red),
                'num_projectiles': len(self.projectiles),
                'spatial_stats': self.spatial_grid.get_statistics() if self.spatial_grid else {},
                'blue_kills': self.blue_kills,
                'red_kills': self.red_kills,
                'selected_unit': self.selected_unit_id,
                'agent_rewards': getattr(self, '_agent_rewards', {}),
                'agent_list': getattr(self, '_agent_list', []),
            }
            render_debug_overlay(self._screen, debug_info)

        # Render unit waypoints and selection (always visible when units are used)
        if self.config.use_units and self._screen:
            render_waypoints(self._screen, self.blue_units, self.selected_unit_id)
            render_waypoints(self._screen, self.red_units, self.selected_red_unit_id)

            # Render circles around agents in selected unit
            if self.selected_unit_id is not None:
                selected_unit = self.get_unit_by_id(self.selected_unit_id, "blue")
                if selected_unit is not None:
                    render_selected_unit(self._screen, selected_unit, circle_color=(135, 206, 250))  # Light blue
            if self.selected_red_unit_id is not None:
                selected_red_unit = self.get_unit_by_id(self.selected_red_unit_id, "red")
                if selected_red_unit is not None:
                    render_selected_unit(self._screen, selected_red_unit, circle_color=(255, 165, 0))  # Orange

        if self.show_keybindings:
            render_keybindings_overlay(self._screen)

        if self.show_rewards and self.config.use_units:
            reward_info = {
                'selected_unit_id': self.selected_unit_id,
                'selected_red_unit_id': self.selected_red_unit_id,
                'blue_units': self.blue_units,
                'red_units': self.red_units,
                'agent_rewards': getattr(self, '_agent_rewards', {}),
                'episode_rewards': getattr(self, '_episode_rewards', {}),
            }
            render_unit_rewards_overlay(self._screen, reward_info)

        pygame.display.flip()

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)),
                axes=(1, 0, 2)
            )

        # Tick clock for human mode
        if self._clock:
            self._clock.tick(FPS)

        return None

    def _init_pygame(self) -> None:
        """Initialize pygame for rendering."""
        pygame.init()
        self._screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Grid-World Multi-Agent System")
        self._clock = pygame.time.Clock()
        self._pygame_initialized = True

    def close(self) -> None:
        """Clean up environment resources."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
            self._screen = None
            self._clock = None


# Tests are in tests/test_environment.py
# Run with: python -m pytest tests/test_environment.py -v
