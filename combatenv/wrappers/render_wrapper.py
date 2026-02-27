"""
RenderWrapper - Visual rendering for the wrapper stack.

This wrapper adds pygame-based visual rendering to the environment,
displaying agents, terrain, projectiles, FOV overlays, and debug info.

Usage:
    env = GridWorld(render_mode="human")
    env = AgentWrapper(env, num_agents=200)
    env = TeamWrapper(env)
    env = TerrainWrapper(env)
    env = RenderWrapper(env)
"""

from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np
import gymnasium as gym

from .base_wrapper import BaseWrapper
from ..config import (
    CELL_SIZE,
    FPS,
    NEAR_FOV_RANGE,
    NEAR_FOV_ANGLE,
    FAR_FOV_RANGE,
    FAR_FOV_ANGLE,
)
from ..renderer import (
    render_background,
    render_grid_lines,
    render_operational_grid,
    render_strategic_grid,
    render_terrain,
    render_agent,
    render_projectile,
    render_fov_highlights,
    render_debug_overlay,
    render_keybindings_overlay,
    render_waypoints,
    render_selected_unit,
    render_unit_rewards_overlay,
)
from ..fov import get_layered_fov_overlap


class RenderWrapper(BaseWrapper):
    """
    Wrapper that handles visual rendering.

    Renders agents, terrain, projectiles, FOV overlays, and debug information
    using pygame.

    Attributes:
        show_debug: Whether to show debug overlay
        show_fov: Whether to show FOV visualization
        show_keybindings: Whether to show keybindings help
        show_strategic_grid: Whether to show strategic grid overlay
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the RenderWrapper.

        Args:
            env: Base environment (should have GridWorld with render_mode="human")
        """
        super().__init__(env)

        # UI state
        self.show_debug = False
        self.show_fov = True
        self.show_keybindings = False
        self.show_strategic_grid = True
        self.show_rewards = True  # Unit rewards overlay (R key)
        self.show_waypoints = True  # Intermediate waypoint lines (W key)
        self.show_goals = True  # Goal waypoint lines (SHIFT+W key)

        # Selected unit for waypoint control (0-indexed internally, 1-8 for display)
        self.selected_unit_id: Optional[int] = None      # Blue team
        self.selected_red_unit_id: Optional[int] = None  # Red team

        # Track user quit request
        self._user_quit = False

        # Reward tracking
        self._episode_reward = 0.0
        self._last_step_reward = 0.0
        self._unit_episode_rewards: Dict[Tuple[str, int], float] = {}  # Per-unit rewards

        # FPS tracking
        self._last_fps = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.selected_unit_id = None
        self.selected_red_unit_id = None
        self._user_quit = False
        self._episode_reward = 0.0
        self._last_step_reward = 0.0
        self._unit_episode_rewards = {}
        return obs, info

    def step(
        self,
        action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Any, bool, bool, Dict[str, Any]]:
        """Execute one step and track reward."""
        if action is None:
            # Pass through to underlying env (for QLearningWrapper)
            obs, reward, terminated, truncated, info = self.env.step()
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)

        # Track reward (handle dict rewards from task wrappers)
        if isinstance(reward, dict):
            total_reward = sum(reward.values())
            self._last_step_reward = float(total_reward)
            self._episode_reward += float(total_reward)
        else:
            self._last_step_reward = float(reward)
            self._episode_reward += float(reward)

        # Track per-unit episode rewards from info
        if "episode_rewards" in info:
            self._unit_episode_rewards = info["episode_rewards"]

        return obs, reward, terminated, truncated, info

    # TODO: Revise rendering mechanism - consider propagating render() through
    # wrapper chain instead of handling all rendering in one place
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment visually.

        Returns:
            RGB array if render_mode="rgb_array", None otherwise
        """
        # Get render mode and screen from base GridWorld
        render_mode = getattr(self.env, 'render_mode', None)
        if render_mode is None:
            return None

        # Initialize pygame if needed
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        if hasattr(base_env, '_init_pygame'):
            base_env._init_pygame()

        screen = getattr(base_env, '_screen', None)
        if screen is None:
            return None

        # Render white background
        render_background(screen)

        # Get terrain grid
        terrain_grid = getattr(self.env, 'terrain_grid', None)

        # Render terrain
        if terrain_grid is not None:
            render_terrain(screen, terrain_grid)

        # Render grid overlays (tactical -> operational -> strategic, light to dark)
        if self.show_strategic_grid:
            render_grid_lines(screen)        # Light black tactical grid
            render_operational_grid(screen)  # Darker black operational grid
            render_strategic_grid(screen)    # Full black strategic grid

        # Get all agents (including dead for bodies)
        all_agents = getattr(self.env, 'agents', [])

        # Render FOV for all agents
        if self.show_fov:
            blue_near, blue_far, red_near, red_far, overlap_nn, overlap_mix, overlap_ff = \
                self._calculate_team_fov(terrain_grid)
            render_fov_highlights(
                screen,
                blue_near, blue_far,
                red_near, red_far,
                overlap_nn, overlap_mix, overlap_ff
            )

        # Render agents (dead first, then alive on top; pass terrain for underwater transparency)
        # First agent (index 0) gets yellow highlight
        for agent in all_agents:
            if not agent.is_alive:
                render_agent(screen, agent, terrain_grid)
        for i, agent in enumerate(all_agents):
            if agent.is_alive:
                render_agent(screen, agent, terrain_grid, highlight=(i == 0))

        # Render projectiles
        projectiles = getattr(self.env, 'projectiles', [])
        for projectile in projectiles:
            render_projectile(screen, projectile)

        # Render muzzle flashes
        muzzle_flashes = getattr(self.env, 'muzzle_flashes', [])
        import pygame
        for pos, lifetime in muzzle_flashes:
            alpha = int(255 * (lifetime / 0.1))
            flash_surface = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(flash_surface, (255, 255, 200, alpha), (3, 3), 3)
            screen_x = int(pos[0] * CELL_SIZE)
            screen_y = int(pos[1] * CELL_SIZE)
            screen.blit(flash_surface, (screen_x - 3, screen_y - 3))

        # Render waypoints and selected unit (blue)
        blue_units = getattr(self.env, 'blue_units', [])
        if blue_units:
            render_waypoints(
                screen, blue_units,
                show_waypoints=self.show_waypoints,
                show_goals=self.show_goals
            )
            if self.selected_unit_id is not None and self.selected_unit_id < len(blue_units):
                # Use index-based lookup (selected_unit_id is an index, not unit.id)
                selected_unit = blue_units[self.selected_unit_id]
                render_selected_unit(screen, selected_unit)

        # Render waypoints and selected unit (red)
        red_units = getattr(self.env, 'red_units', [])
        if red_units:
            render_waypoints(
                screen, red_units,
                show_waypoints=self.show_waypoints,
                show_goals=self.show_goals
            )
            if self.selected_red_unit_id is not None and self.selected_red_unit_id < len(red_units):
                # Use index-based lookup (selected_red_unit_id is an index, not unit.id)
                selected_unit = red_units[self.selected_red_unit_id]
                # Orange highlight for red team
                render_selected_unit(screen, selected_unit, circle_color=(255, 165, 0))

        # Render debug overlay
        if self.show_debug:
            # Build stats dict
            stats = self._build_stats()
            render_debug_overlay(screen, stats)

        # Render keybindings
        if self.show_keybindings:
            render_keybindings_overlay(screen)

        # Render unit rewards overlay
        if self.show_rewards:
            # Get current actions from QLearningWrapper by traversing wrapper chain
            unit_actions = {}
            env = self.env
            while env is not None:
                if hasattr(env, '_prev_actions'):
                    unit_actions = env._prev_actions
                    break
                env = getattr(env, 'env', None)
            reward_info = {
                'selected_unit_id': self.selected_unit_id,
                'selected_red_unit_id': self.selected_red_unit_id,
                'blue_units': blue_units,
                'red_units': red_units,
                'episode_rewards': self._unit_episode_rewards,
                'unit_actions': unit_actions,
            }
            render_unit_rewards_overlay(screen, reward_info)

        # Update display
        import pygame
        pygame.display.flip()

        # Tick clock and track FPS
        clock = getattr(base_env, '_clock', None)
        if clock:
            clock.tick(FPS)
            self._last_fps = clock.get_fps()

        return None

    def _calculate_team_fov(self, terrain_grid) -> Tuple[
        Set[Tuple[int, int]],  # blue_near
        Set[Tuple[int, int]],  # blue_far
        Set[Tuple[int, int]],  # red_near
        Set[Tuple[int, int]],  # red_far
        Set[Tuple[int, int]],  # overlap_near_near
        Set[Tuple[int, int]],  # overlap_mixed
        Set[Tuple[int, int]],  # overlap_far_far
    ]:
        """Calculate FOV cells for both teams with overlap detection."""
        team_agents = getattr(self.env, 'team_agents', {})
        blue_agents = [a for a in team_agents.get("blue", []) if a.is_alive]
        red_agents = [a for a in team_agents.get("red", []) if a.is_alive]

        # Use get_layered_fov_overlap to calculate all FOV sets with overlaps
        return get_layered_fov_overlap(
            blue_agents=blue_agents,
            red_agents=red_agents,
            near_range=NEAR_FOV_RANGE,
            near_angle=NEAR_FOV_ANGLE,
            far_range=FAR_FOV_RANGE,
            far_angle=FAR_FOV_ANGLE,
            terrain_grid=terrain_grid,
        )

    def _build_stats(self) -> Dict[str, Any]:
        """Build stats dict for debug overlay."""
        team_agents = getattr(self.env, 'team_agents', {})
        blue_agents = team_agents.get("blue", [])
        red_agents = team_agents.get("red", [])

        controlled_agent = getattr(self.env, 'controlled_agent', None)

        # Get step count from TerminationWrapper or base
        step_count = getattr(self.env, 'step_count', 0)

        # Get kill counts from CombatWrapper
        blue_kills = getattr(self.env, 'blue_kills', 0)
        red_kills = getattr(self.env, 'red_kills', 0)

        # Count alive and dead
        blue_alive = sum(1 for a in blue_agents if a.is_alive)
        red_alive = sum(1 for a in red_agents if a.is_alive)
        blue_dead = len(blue_agents) - blue_alive
        red_dead = len(red_agents) - red_alive

        # Get projectile count
        projectiles = getattr(self.env, 'projectiles', [])

        return {
            # FPS
            'fps': self._last_fps,
            # Agent counts (required by render_debug_overlay)
            'num_agents': blue_alive + red_alive,
            'blue_agents': blue_alive,
            'red_agents': red_alive,
            # Combat stats
            'blue_dead': blue_dead,
            'red_dead': red_dead,
            'num_projectiles': len(projectiles),
            'blue_kills': blue_kills,
            'red_kills': red_kills,
            # Step count
            'step_count': step_count,
            # Reward tracking
            'step_reward': self._last_step_reward,
            'episode_reward': self._episode_reward,
            # Controlled agent stats
            'controlled_health': controlled_agent.health if controlled_agent else 0,
            'controlled_stamina': controlled_agent.stamina if controlled_agent else 0,
            'controlled_ammo': controlled_agent.magazine_ammo if controlled_agent else 0,
        }

    def process_events(self) -> bool:
        """
        Process pygame events.

        Returns:
            False if user wants to quit, True otherwise
        """
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._user_quit = True
                return False

            if event.type == pygame.KEYDOWN:
                # SHIFT+Q to quit
                if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self._user_quit = True
                    return False

                # Toggle debug overlay
                if event.key == pygame.K_BACKQUOTE:
                    self.show_debug = not self.show_debug

                # Toggle FOV overlay
                if event.key == pygame.K_f:
                    self.show_fov = not self.show_fov

                # Toggle keybindings
                if event.key == pygame.K_SLASH and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.show_keybindings = not self.show_keybindings

                # Toggle strategic grid
                if event.key == pygame.K_g:
                    self.show_strategic_grid = not self.show_strategic_grid

                # Toggle rewards overlay
                if event.key == pygame.K_r:
                    self.show_rewards = not self.show_rewards

                # Toggle waypoint/goal lines
                if event.key == pygame.K_w:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.show_goals = not self.show_goals
                    else:
                        self.show_waypoints = not self.show_waypoints

                # Unit selection (1-8 for blue, SHIFT+1-8 for red)
                if pygame.K_1 <= event.key <= pygame.K_8:
                    unit_num = event.key - pygame.K_1
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        # Red team selection (SHIFT + 1-8)
                        red_units = getattr(self.env, 'red_units', [])
                        if unit_num < len(red_units):
                            self.selected_red_unit_id = unit_num
                            self.selected_unit_id = None  # Clear blue selection
                    else:
                        # Blue team selection (1-8)
                        blue_units = getattr(self.env, 'blue_units', [])
                        if unit_num < len(blue_units):
                            self.selected_unit_id = unit_num
                            self.selected_red_unit_id = None  # Clear red selection

                # Clear unit selection
                if event.key == pygame.K_SPACE:
                    self.selected_unit_id = None
                    self.selected_red_unit_id = None

            # Mouse click for goal waypoint setting
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_x, mouse_y = event.pos
                    grid_x = mouse_x / CELL_SIZE
                    grid_y = mouse_y / CELL_SIZE

                    # Set goal waypoint for selected blue unit
                    if self.selected_unit_id is not None:
                        blue_units = getattr(self.env, 'blue_units', [])
                        if self.selected_unit_id < len(blue_units):
                            selected_unit = blue_units[self.selected_unit_id]
                            selected_unit.set_goal(grid_x, grid_y)

                    # Set goal waypoint for selected red unit
                    elif self.selected_red_unit_id is not None:
                        red_units = getattr(self.env, 'red_units', [])
                        if self.selected_red_unit_id < len(red_units):
                            selected_unit = red_units[self.selected_red_unit_id]
                            selected_unit.set_goal(grid_x, grid_y)

        return True

    def close(self) -> None:
        """Clean up resources."""
        # Find and close base GridWorld
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'close'):
            base_env.close()
