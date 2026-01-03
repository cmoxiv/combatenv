"""
Unit tests for the rendering system.

Tests cover:
- Configuration value imports
- Coordinate conversion
- Surface initialization logic

Note: Full rendering tests require pygame display which may not
be available in headless CI environments.

Run with: pytest tests/test_renderer.py -v
"""

import pytest
from combatenv import config
WINDOW_SIZE = config.WINDOW_SIZE
CELL_SIZE = config.CELL_SIZE
GRID_SIZE = config.GRID_SIZE


class TestConfigImports:
    """Test that renderer can access required config values."""

    def test_window_size(self):
        """Test WINDOW_SIZE is valid."""
        assert WINDOW_SIZE > 0
        assert WINDOW_SIZE == 1024  # Expected default

    def test_cell_size(self):
        """Test CELL_SIZE is valid."""
        assert CELL_SIZE > 0
        assert CELL_SIZE == 16  # Expected default

    def test_grid_size(self):
        """Test GRID_SIZE is derived correctly."""
        assert GRID_SIZE > 0
        assert GRID_SIZE == WINDOW_SIZE // CELL_SIZE


class TestCoordinateConversion:
    """Test coordinate conversion logic."""

    def test_grid_to_pixel(self):
        """Test grid coordinates convert to pixels."""
        grid_x = 10
        pixel_x = grid_x * CELL_SIZE

        assert pixel_x == 160

    def test_grid_to_pixel_zero(self):
        """Test grid position 0 is pixel 0."""
        grid_x = 0
        pixel_x = grid_x * CELL_SIZE

        assert pixel_x == 0

    def test_grid_to_pixel_max(self):
        """Test max grid position."""
        grid_x = GRID_SIZE - 1
        pixel_x = grid_x * CELL_SIZE

        assert pixel_x == (GRID_SIZE - 1) * CELL_SIZE

    def test_fractional_grid_position(self):
        """Test fractional grid positions convert correctly."""
        grid_x = 10.5
        pixel_x = int(grid_x * CELL_SIZE)

        assert pixel_x == 168


class TestAgentSizing:
    """Test agent size calculations."""

    def test_agent_size_ratio(self):
        """Test agent size ratio produces valid radius."""
        from combatenv.config import AGENT_SIZE_RATIO

        radius = int((CELL_SIZE / 2) * AGENT_SIZE_RATIO)

        assert radius > 0
        assert radius < CELL_SIZE

    def test_nose_ratio(self):
        """Test nose length ratio is reasonable."""
        from combatenv.config import AGENT_NOSE_RATIO

        nose_length = (CELL_SIZE / 2) * AGENT_NOSE_RATIO

        assert nose_length > 0
        assert nose_length < CELL_SIZE


class TestColorFormats:
    """Test color format definitions."""

    def test_background_color(self):
        """Test background color format."""
        from combatenv.config import COLOR_BACKGROUND

        assert len(COLOR_BACKGROUND) == 3
        assert all(0 <= c <= 255 for c in COLOR_BACKGROUND)

    def test_team_colors(self):
        """Test team color formats."""
        from combatenv.config import COLOR_BLUE_TEAM, COLOR_RED_TEAM

        assert len(COLOR_BLUE_TEAM) == 3
        assert len(COLOR_RED_TEAM) == 3

    def test_terrain_colors(self):
        """Test terrain color formats."""
        from combatenv.config import COLOR_BUILDING, COLOR_FIRE, COLOR_SWAMP, COLOR_WATER

        for color in [COLOR_BUILDING, COLOR_FIRE, COLOR_SWAMP, COLOR_WATER]:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


# Pygame-dependent tests
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


@pytest.mark.skipif(not PYGAME_AVAILABLE, reason="pygame not available")
class TestRendererFunctions:
    """Test renderer functions (requires pygame)."""

    @pytest.fixture(autouse=True)
    def setup_pygame(self):
        """Setup and teardown pygame for tests."""
        import os
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        pygame.init()
        yield
        pygame.quit()

    def test_render_background(self):
        """Test background rendering doesn't crash."""
        from combatenv.renderer import render_background

        surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        render_background(surface)  # Should not raise

    def test_render_grid_lines(self):
        """Test grid lines rendering doesn't crash."""
        from combatenv.renderer import render_grid_lines

        surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        render_grid_lines(surface)  # Should not raise

    def test_render_agent_alive(self):
        """Test rendering a living agent."""
        from combatenv.renderer import render_agent
        from combatenv import Agent

        surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        agent = Agent(position=(10.0, 10.0), orientation=45.0, team="blue")

        render_agent(surface, agent)  # Should not raise

    def test_render_agent_dead(self):
        """Test rendering a dead agent."""
        from combatenv.renderer import render_agent
        from combatenv import Agent

        surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        agent = Agent(position=(10.0, 10.0), orientation=45.0, team="red")
        agent.health = 0

        render_agent(surface, agent)  # Should not raise

    def test_render_terrain(self):
        """Test terrain rendering."""
        from combatenv.renderer import render_terrain
        from combatenv import TerrainGrid, TerrainType

        surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        terrain = TerrainGrid(GRID_SIZE, GRID_SIZE)
        terrain.set(5, 5, TerrainType.BUILDING)
        terrain.set(10, 10, TerrainType.FIRE)

        render_terrain(surface, terrain)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
