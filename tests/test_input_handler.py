"""
Unit tests for the input handler.

Tests cover:
- InputHandler creation
- Running state management
- Event handling (limited without pygame display)

Run with: pytest tests/test_input_handler.py -v

Note: Full event testing requires pygame initialization which
may not be available in headless environments.
"""

import pytest
from combatenv import InputHandler


class MockEnv:
    """Mock environment for testing InputHandler."""

    def __init__(self):
        self.show_debug = False
        self.show_keybindings = False


class TestInputHandlerInit:
    """Tests for InputHandler initialization."""

    def test_init(self):
        """Test InputHandler creation."""
        env = MockEnv()
        handler = InputHandler(env)

        assert handler.env is env

    def test_initial_running_state(self):
        """Test initial running state is True."""
        env = MockEnv()
        handler = InputHandler(env)

        assert handler.running == True


class TestInputHandlerState:
    """Tests for InputHandler state management."""

    def test_running_can_be_set_false(self):
        """Test running state can be set to False."""
        env = MockEnv()
        handler = InputHandler(env)

        handler.running = False

        assert handler.running == False

    def test_env_reference(self):
        """Test handler maintains env reference."""
        env = MockEnv()
        handler = InputHandler(env)

        # Modify env through handler
        handler.env.show_debug = True

        assert env.show_debug == True


# Pygame-dependent tests are marked as skipped if pygame is not available
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


@pytest.mark.skipif(not PYGAME_AVAILABLE, reason="pygame not available")
class TestInputHandlerEvents:
    """Tests for event handling (requires pygame)."""

    @pytest.fixture(autouse=True)
    def setup_pygame(self):
        """Setup and teardown pygame for tests."""
        # Initialize pygame in headless mode if possible
        import os
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        pygame.init()
        yield
        pygame.quit()

    def test_escape_stops_running(self):
        """Test ESC key stops running."""
        env = MockEnv()
        handler = InputHandler(env)

        # Create escape key event
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
        handler._handle_keydown(event)

        assert handler.running == False

    def test_backtick_toggles_debug(self):
        """Test backtick toggles debug overlay."""
        env = MockEnv()
        handler = InputHandler(env)

        assert env.show_debug == False

        # Toggle on
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKQUOTE)
        handler._handle_keydown(event)

        assert env.show_debug == True

        # Toggle off
        handler._handle_keydown(event)

        assert env.show_debug == False

    def test_question_toggles_keybindings(self):
        """Test Shift+/ (?) toggles keybindings help."""
        env = MockEnv()
        handler = InputHandler(env)

        assert env.show_keybindings == False

        # Toggle on
        event = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_SLASH,
            mod=pygame.KMOD_SHIFT
        )
        handler._handle_keydown(event)

        assert env.show_keybindings == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
