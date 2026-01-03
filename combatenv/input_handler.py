"""
Input handling for the Grid-World Multi-Agent Tactical Simulation.

This module provides the InputHandler class for processing pygame events
and managing keyboard input separately from the main game loop.
"""

import pygame
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import TacticalCombatEnv


class InputHandler:
    """
    Handles pygame events and keyboard input.

    Processes quit events and keyboard shortcuts for toggling
    debug overlays and other UI elements.

    Attributes:
        env: The environment instance to control
        running: Whether the simulation should continue running
    """

    def __init__(self, env: 'TacticalCombatEnv'):
        """
        Initialize the input handler.

        Args:
            env: The tactical combat environment instance
        """
        self.env = env
        self.running = True

    def process_events(self) -> bool:
        """
        Process all pending pygame events.

        Handles window close events and keyboard input.
        Updates internal state and environment properties as needed.

        Returns:
            True if simulation should continue, False if should quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

        return self.running

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        """
        Handle a keyboard key press event.

        Key bindings:
            - ESC: Quit simulation
            - ` (backtick): Toggle debug overlay
            - ? (Shift+/): Toggle keybindings help

        Args:
            event: The pygame KEYDOWN event
        """
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_BACKQUOTE:
            self.env.show_debug = not self.env.show_debug
        elif event.key == pygame.K_SLASH and (event.mod & pygame.KMOD_SHIFT):
            self.env.show_keybindings = not self.env.show_keybindings


if __name__ == "__main__":
    """Basic self-tests for input_handler module."""
    import sys

    def test_input_handler_creation():
        """Test InputHandler creation without pygame."""
        # Create a mock environment
        class MockEnv:
            def __init__(self):
                self.show_debug = False
                self.show_keybindings = False

        env = MockEnv()
        handler = InputHandler(env)

        assert handler.env is env
        assert handler.running == True
        print("  InputHandler creation: OK")

    def test_initial_state():
        """Test initial running state."""
        class MockEnv:
            def __init__(self):
                self.show_debug = False
                self.show_keybindings = False

        env = MockEnv()
        handler = InputHandler(env)

        assert handler.running == True
        print("  initial running state: OK")

    # Run all tests
    print("Running input_handler.py self-tests...")
    print("  (Note: pygame event tests require pygame initialization)")
    try:
        test_input_handler_creation()
        test_initial_state()
        print("All input_handler.py self-tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
