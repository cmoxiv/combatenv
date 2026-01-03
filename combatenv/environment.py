"""
Gymnasium-compatible environment for the Grid-World Multi-Agent Tactical Simulation.

This module wraps the tactical combat simulation as a Gymnasium environment, enabling
integration with reinforcement learning libraries and standardized training loops.

Environment Details:
    - Observation: 88-float numpy array with structure:
        - Indices 0-9: Agent state (position, orientation, health, stamina, armor, ammo, etc.)
        - Indices 10-29: 5 nearest enemies (4 floats each: rel_x, rel_y, health, distance)
        - Indices 30-49: 5 nearest allies (4 floats each: rel_x, rel_y, health, distance)
        - Indices 50-87: Terrain types for up to 38 FOV cells (normalized 0-1)
            - EMPTY=0.0, BUILDING=0.25, FIRE=0.5, SWAMP=0.75, WATER=1.0
    - Action: Box(3,) continuous [move_x, move_y, shoot]
    - Reward: Based on kills, damage, survival
    - Termination: When controlled agent dies or team eliminated

Usage:
    >>> from environment import TacticalCombatEnv
    >>> env = TacticalCombatEnv(render_mode="human")
    >>> obs, info = env.reset()
    >>> for _ in range(1000):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if terminated or truncated:
    ...         obs, info = env.reset()
    >>> env.close()
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
    WINDOW_SIZE, FPS,
    NEAR_FOV_ACCURACY, FAR_FOV_ACCURACY,
    NEAR_FOV_RANGE, NEAR_FOV_ANGLE,
    FAR_FOV_RANGE, FAR_FOV_ANGLE,
    MOVEMENT_ACCURACY_PENALTY,
    MUZZLE_FLASH_OFFSET, MUZZLE_FLASH_LIFETIME,
    RESPAWN_DELAY_SECONDS,
    NUM_AGENTS_PER_TEAM, GRID_SIZE,
    AGENT_MAX_HEALTH, AGENT_MAX_STAMINA, AGENT_MAX_ARMOR, AGENT_MAX_AMMO,
    AGENT_MOVE_SPEED,
    FIRE_DAMAGE_PER_STEP, SWAMP_STUCK_MIN_STEPS, SWAMP_STUCK_MAX_STEPS,
    TERRAIN_BUILDING_PCT, TERRAIN_FIRE_PCT, TERRAIN_SWAMP_PCT, TERRAIN_WATER_PCT
)
from .agent import Agent, spawn_all_teams, try_shoot_at_visible_target
from .fov import get_layered_fov_overlap, get_fov_cells, get_fov_cache
from .renderer import render_all, render_debug_overlay, render_keybindings_overlay, render_terrain
from .spatial import SpatialGrid
from .projectile import Projectile
from .terrain import TerrainType, TerrainGrid


# Observation space constants
NUM_NEARBY_ENEMIES = 5
NUM_NEARBY_ALLIES = 5
MAX_FOV_CELLS = 38  # Maximum cells visible in combined near+far FOV
NUM_TERRAIN_TYPES = 5  # EMPTY, BUILDING, FIRE, SWAMP, WATER
OBS_SIZE = 10 + (NUM_NEARBY_ENEMIES * 4) + (NUM_NEARBY_ALLIES * 4) + MAX_FOV_CELLS  # 88 floats


@dataclass
class EnvConfig:
    """Configuration for the tactical combat environment."""
    num_agents_per_team: int = NUM_AGENTS_PER_TEAM
    respawn_enabled: bool = False  # Disabled for RL training
    respawn_delay: float = RESPAWN_DELAY_SECONDS
    max_steps: Optional[int] = 1000
    terminate_on_team_elimination: bool = True
    terminate_on_controlled_death: bool = True
    allow_escape_exit: bool = True  # Allow ESC key to exit simulation


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

        # Define action space: [move_x, move_y, shoot]
        # move_x, move_y: -1.0 to 1.0 (direction and speed)
        # shoot: 0.0 to 1.0 (fire if > 0.5)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
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

        # Controlled agent (first blue agent)
        self.controlled_agent: Optional[Agent] = None

        # Statistics
        self.blue_kills = 0
        self.red_kills = 0
        self.step_count = 0
        self.controlled_agent_kills = 0
        self.controlled_agent_damage_dealt = 0.0

        # UI state
        self.show_debug = True  # Debug overlay on by default
        self.show_keybindings = False
        self.show_fov = True  # FOV visualization on by default
        self._user_quit = False  # Set True when user requests exit via ESC or window close

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
                building_pct=TERRAIN_BUILDING_PCT,
                fire_pct=TERRAIN_FIRE_PCT,
                swamp_pct=TERRAIN_SWAMP_PCT,
                water_pct=TERRAIN_WATER_PCT,
                rng=terrain_rng
            )

        # Spawn agents (only on empty terrain)
        self.blue_agents, self.red_agents = spawn_all_teams(terrain_grid=self.terrain_grid)
        self.all_agents = self.blue_agents + self.red_agents
        self.alive_agents = list(self.all_agents)

        # Set controlled agent (first blue agent)
        self.controlled_agent = self.blue_agents[0]

        # Initialize spatial grid
        self.spatial_grid = SpatialGrid(cell_size=2.0)

        # Clear game state
        self.projectiles = []
        self.muzzle_flashes = []
        self.respawn_timers = {}

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

        # Autonomous combat for other agents
        self._execute_autonomous_combat(dt)

        # Autonomous movement for other agents (skip stuck agents)
        for agent in self.alive_agents:
            if agent is not self.controlled_agent:
                if not agent.is_stuck:
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
            action: [move_x, move_y, shoot]
            dt: Delta time

        Returns:
            Immediate reward from action
        """
        agent = self.controlled_agent
        if agent is None or not agent.is_alive:
            return 0.0

        move_x, move_y, shoot = action
        reward = 0.0

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
            projectile = try_shoot_at_visible_target(
                agent, nearby,
                NEAR_FOV_ACCURACY, FAR_FOV_ACCURACY,
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

    def _execute_autonomous_combat(self, dt: float) -> None:
        """Execute combat for autonomous agents (not controlled)."""
        for agent in self.alive_agents:
            if agent is self.controlled_agent:
                continue

            nearby = self.spatial_grid.get_nearby_agents(agent)
            projectile = try_shoot_at_visible_target(
                agent, nearby,
                NEAR_FOV_ACCURACY, FAR_FOV_ACCURACY,
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
        kills_this_step = {'blue': 0, 'red': 0, 'controlled': 0}

        for i, projectile in enumerate(self.projectiles):
            if projectile.update(dt, self.terrain_grid):
                projectiles_to_remove.add(i)
                continue

            for agent in self.alive_agents:
                if projectile.check_collision(agent):
                    was_alive = agent.is_alive
                    agent.take_damage(projectile.damage)

                    if was_alive and not agent.is_alive:
                        # Clear dead agent from FOV cache
                        get_fov_cache().remove_agent(id(agent))

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

        # Add terrain type for each cell (normalized to [0, 1])
        for i, (cx, cy) in enumerate(sorted_cells[:MAX_FOV_CELLS]):
            terrain_type = self.terrain_grid.get(cx, cy)
            obs[50 + i] = terrain_type / (NUM_TERRAIN_TYPES - 1)  # Normalize to [0, 1]

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
        - +10.0 per kill by controlled agent
        - +0.1 per damage dealt by controlled agent
        - -10.0 if controlled agent dies
        - +1.0 per step survived
        """
        reward = 0.0

        # Reward for kills by controlled agent
        reward += kills_this_step.get('controlled', 0) * 10.0

        # Small survival reward
        if self.controlled_agent and self.controlled_agent.is_alive:
            reward += 0.01

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
        """Process terrain effects (fire damage, swamp stuck) for all alive agents."""
        for agent in self.alive_agents:
            cell = agent.get_grid_position()
            terrain = self.terrain_grid.get(*cell)

            if terrain == TerrainType.FIRE:
                # Fire damage bypasses armor
                agent.apply_terrain_damage(FIRE_DAMAGE_PER_STEP)

            elif terrain == TerrainType.SWAMP:
                # Only apply stuck if not already stuck
                if not agent.is_stuck:
                    agent.stuck_steps = random.randint(
                        SWAMP_STUCK_MIN_STEPS,
                        SWAMP_STUCK_MAX_STEPS
                    )

            # Update stuck timer
            agent.update_stuck()

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

        return not self._user_quit

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        """
        Handle keyboard key press events.

        Key bindings:
            - ESC: Quit simulation (if allow_escape_exit is True)
            - ` (backtick): Toggle debug overlay
            - ? (Shift+/): Toggle keybindings help

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
            self.terrain_grid
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
            }
            render_debug_overlay(self._screen, debug_info)

        if self.show_keybindings:
            render_keybindings_overlay(self._screen)

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


if __name__ == "__main__":
    """Basic self-tests for environment module."""
    import sys

    def test_env_creation():
        """Test environment creation."""
        env = TacticalCombatEnv(render_mode=None)

        assert env.action_space.shape == (3,)
        assert env.observation_space.shape == (OBS_SIZE,)  # 88 floats
        print("  environment creation: OK")

    def test_env_reset():
        """Test environment reset."""
        env = TacticalCombatEnv(render_mode=None)
        obs, info = env.reset(seed=42)

        assert obs.shape == (OBS_SIZE,)
        assert isinstance(info, dict)
        assert len(env.blue_agents) == NUM_AGENTS_PER_TEAM
        assert len(env.red_agents) == NUM_AGENTS_PER_TEAM
        assert env.controlled_agent is not None
        print("  environment reset: OK")

    def test_env_step():
        """Test environment step."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.step_count == 1
        print("  environment step: OK")

    def test_observation_normalized():
        """Test that observations are normalized to [0, 1]."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)

            assert np.all(obs >= 0.0), "Observations should be >= 0"
            assert np.all(obs <= 1.0), "Observations should be <= 1"
        print("  observation normalized: OK")

    def test_terrain_generated():
        """Test that terrain is generated on reset."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        assert env.terrain_grid is not None

        # Count non-empty terrain
        non_empty = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if env.terrain_grid.get(x, y) != TerrainType.EMPTY:
                    non_empty += 1

        assert non_empty > 0, "Should have some terrain"
        print(f"  terrain generated: OK ({non_empty} non-empty cells)")

    def test_deterministic_seed():
        """Test that seed produces deterministic results."""
        env = TacticalCombatEnv(render_mode=None)

        # First run
        obs1, _ = env.reset(seed=12345)
        agent1_pos = env.controlled_agent.position

        # Second run with same seed
        obs2, _ = env.reset(seed=12345)
        agent2_pos = env.controlled_agent.position

        assert np.allclose(obs1, obs2), "Same seed should produce same observations"
        assert agent1_pos == agent2_pos, "Same seed should produce same agent positions"
        print("  deterministic seed: OK")

    def test_max_steps_truncation():
        """Test truncation at max steps."""
        config = EnvConfig(max_steps=5)
        env = TacticalCombatEnv(render_mode=None, config=config)
        env.reset(seed=42)

        for i in range(10):
            action = np.array([0.0, 0.0, 0.0])  # No movement
            _, _, terminated, truncated, _ = env.step(action)

            if i >= 4:  # After 5 steps
                assert truncated or terminated, f"Should be done at step {i+1}"
                break
        print("  max steps truncation: OK")

    # Run all tests
    print("Running environment.py self-tests...")
    try:
        test_env_creation()
        test_env_reset()
        test_env_step()
        test_observation_normalized()
        test_terrain_generated()
        test_deterministic_seed()
        test_max_steps_truncation()
        print("All environment.py self-tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
