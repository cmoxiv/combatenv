"""
Keybindings wrapper for handling keyboard and mouse input.

This wrapper extracts input handling from the base environment, allowing
clean separation of concerns. It handles:
- Keyboard shortcuts (toggle overlays, unit selection, quit)
- Mouse clicks (waypoint setting)
- UI state (selected units, overlay toggles)

Usage:
    from combatenv.wrappers import KeybindingsWrapper

    env = TacticalCombatEnv(render_mode="human")
    env = KeybindingsWrapper(env)

    # In game loop:
    if not env.process_events():
        break  # User requested quit
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None  # type: ignore[assignment]
    PYGAME_AVAILABLE = False

from combatenv.config import CELL_SIZE


class KeybindingsWrapper(gym.Wrapper):
    """
    Wrapper for keyboard and mouse input handling.

    Extracts input processing from TacticalCombatEnv to allow clean
    separation of concerns. Manages UI state like selected units and
    overlay toggles.

    Attributes:
        show_debug: Whether debug overlay is visible
        show_keybindings: Whether keybindings help is visible
        show_fov: Whether FOV visualization is visible
        show_strategic_grid: Whether strategic grid overlay is visible
        show_rewards: Whether rewards overlay is visible
        selected_unit_id: Currently selected blue unit (0-7 or None)
        selected_red_unit_id: Currently selected red unit (0-7 or None)
    """

    def __init__(
        self,
        env,
        show_debug: bool = False,
        show_keybindings: bool = False,
        show_fov: bool = True,
        show_strategic_grid: bool = True,
        show_rewards: bool = False,
        allow_escape_exit: bool = True,
    ):
        """
        Initialize the keybindings wrapper.

        Args:
            env: Base environment to wrap
            show_debug: Initial state of debug overlay
            show_keybindings: Initial state of keybindings help
            show_fov: Initial state of FOV visualization
            show_strategic_grid: Initial state of strategic grid overlay
            show_rewards: Initial state of rewards overlay
            allow_escape_exit: Whether Shift+Q exits the simulation
        """
        super().__init__(env)

        # UI toggles
        self.show_debug = show_debug
        self.show_keybindings = show_keybindings
        self.show_fov = show_fov
        self.show_strategic_grid = show_strategic_grid
        self.show_rewards = show_rewards
        self.allow_escape_exit = allow_escape_exit

        # Selection state
        self.selected_unit_id: Optional[int] = None
        self.selected_red_unit_id: Optional[int] = None

        # Quit flag
        self._user_quit = False

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset and clear quit flag."""
        self._user_quit = False
        return self.env.reset(**kwargs)

    def process_events(self) -> bool:
        """
        Process pygame events (window close, keyboard input, mouse clicks).

        Returns:
            True if simulation should continue, False if user requested exit

        Note:
            In headless mode (render_mode=None), always returns True.
        """
        if not PYGAME_AVAILABLE or pygame is None:
            return not self._user_quit

        # Check base env render mode
        base_env = self.env.unwrapped
        if getattr(base_env, 'render_mode', None) is None:
            return not self._user_quit

        for event in pygame.event.get():  # type: ignore[union-attr]
            if event.type == pygame.QUIT:
                self._user_quit = True
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_click(event)

        return not self._user_quit

    def _handle_keydown(self, event) -> None:
        """
        Handle keyboard key press events.

        Key bindings:
            - Shift+Q: Quit simulation
            - ` (backtick): Toggle debug overlay
            - ? (Shift+/): Toggle keybindings help
            - F: Toggle FOV overlay
            - G: Toggle strategic grid overlay
            - R: Toggle rewards overlay
            - 1-8: Select blue unit
            - Shift+1-8: Select red unit
            - SPACE: Clear selected/all waypoints
            - ENTER: Advance to next waypoint

        Args:
            event: The pygame KEYDOWN event
        """
        assert pygame is not None  # Guaranteed by process_events guard

        # Quit
        if event.key == pygame.K_q and (event.mod & pygame.KMOD_SHIFT):
            if self.allow_escape_exit:
                self._user_quit = True
            return

        # Toggle overlays
        if event.key == pygame.K_BACKQUOTE:
            self.show_debug = not self.show_debug
            # Sync with base env if it has this attribute
            self._sync_to_base_env('show_debug', self.show_debug)
        elif event.key == pygame.K_SLASH and (event.mod & pygame.KMOD_SHIFT):
            self.show_keybindings = not self.show_keybindings
            self._sync_to_base_env('show_keybindings', self.show_keybindings)
        elif event.key == pygame.K_f:
            self.show_fov = not self.show_fov
            self._sync_to_base_env('show_fov', self.show_fov)
        elif event.key == pygame.K_g:
            self.show_strategic_grid = not self.show_strategic_grid
            self._sync_to_base_env('show_strategic_grid', self.show_strategic_grid)
        elif event.key == pygame.K_r:
            self.show_rewards = not self.show_rewards
            self._sync_to_base_env('show_rewards', self.show_rewards)

        # Unit selection (1-8)
        elif self._units_enabled() and event.key in (
            pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
            pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8
        ):
            unit_idx = event.key - pygame.K_1
            if event.mod & pygame.KMOD_SHIFT:
                # Shift+1-8: Select red unit
                self.selected_red_unit_id = unit_idx
                self.selected_unit_id = None
            else:
                # 1-8: Select blue unit
                self.selected_unit_id = unit_idx
                self.selected_red_unit_id = None
            # Sync selection to base env
            self._sync_to_base_env('selected_unit_id', self.selected_unit_id)
            self._sync_to_base_env('selected_red_unit_id', self.selected_red_unit_id)

        # Clear waypoints
        elif self._units_enabled() and event.key == pygame.K_SPACE:
            base_env = self.env.unwrapped
            if self.selected_unit_id is not None:
                if hasattr(base_env, 'clear_unit_waypoint'):
                    base_env.clear_unit_waypoint(self.selected_unit_id, "blue")
            else:
                if hasattr(base_env, 'clear_all_waypoints'):
                    base_env.clear_all_waypoints("blue")

        # Advance to next waypoint
        elif self._units_enabled() and event.key == pygame.K_RETURN:
            if self.selected_unit_id is not None:
                base_env = self.env.unwrapped
                if hasattr(base_env, 'advance_unit_waypoint'):
                    base_env.advance_unit_waypoint(self.selected_unit_id, "blue")

    def _handle_mouse_click(self, event) -> None:
        """
        Handle mouse click events for waypoint setting.

        Left click sets waypoint for selected unit.
        Shift+left click adds to waypoint sequence.

        Args:
            event: The pygame MOUSEBUTTONDOWN event
        """
        if not self._units_enabled():
            return

        if self.selected_unit_id is None:
            return

        base_env = self.env.unwrapped
        mouse_x, mouse_y = event.pos
        grid_x = mouse_x / CELL_SIZE
        grid_y = mouse_y / CELL_SIZE

        mods = pygame.key.get_mods()
        if mods & pygame.KMOD_SHIFT:
            # Shift+click: add to waypoint sequence
            if hasattr(base_env, 'add_unit_waypoint'):
                base_env.add_unit_waypoint(self.selected_unit_id, "blue", grid_x, grid_y)
        else:
            # Regular click: set single waypoint
            if hasattr(base_env, 'set_unit_waypoint'):
                base_env.set_unit_waypoint(self.selected_unit_id, "blue", grid_x, grid_y)

    def _units_enabled(self) -> bool:
        """Check if units are enabled in base environment."""
        base_env = self.env.unwrapped
        config = getattr(base_env, 'config', None)
        if config is not None:
            return getattr(config, 'use_units', False)
        return False

    def _sync_to_base_env(self, attr: str, value: Any) -> None:
        """Sync an attribute to the base environment if it exists."""
        base_env = self.env.unwrapped
        if hasattr(base_env, attr):
            setattr(base_env, attr, value)

    def render(self):
        """
        Render with keybindings overlay.

        Syncs UI state to base env before rendering, then calls base render.
        """
        # Sync all UI state to base env before rendering
        self._sync_to_base_env('show_debug', self.show_debug)
        self._sync_to_base_env('show_keybindings', self.show_keybindings)
        self._sync_to_base_env('show_fov', self.show_fov)
        self._sync_to_base_env('show_strategic_grid', self.show_strategic_grid)
        self._sync_to_base_env('show_rewards', self.show_rewards)
        self._sync_to_base_env('selected_unit_id', self.selected_unit_id)
        self._sync_to_base_env('selected_red_unit_id', self.selected_red_unit_id)

        return self.env.render()

    @property
    def user_quit(self) -> bool:
        """Whether user has requested to quit."""
        return self._user_quit
