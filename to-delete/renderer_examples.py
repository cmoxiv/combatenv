"""
Advanced rendering examples and alternative usage patterns.

This module demonstrates various ways to use the rendering system
for different use cases and custom rendering pipelines.
"""

import pygame
from typing import List, Set, Tuple

from renderer import (
    render_background,
    render_grid_lines,
    render_fov_highlights,
    render_agent,
    render_agents,
    render_debug_info
)
from fov import get_fov_overlap


def render_minimal(surface: pygame.Surface, agents: List) -> None:
    """
    Minimal rendering: just background and agents (no grid, no FOV).

    Useful for:
    - Simple visualizations
    - Performance testing
    - Clean screenshots without UI elements

    Args:
        surface: Pygame display surface
        agents: List of all agents (both teams)
    """
    render_background(surface)
    render_agents(surface, agents)


def render_grid_only(surface: pygame.Surface) -> None:
    """
    Static grid visualization (no agents or FOV).

    Useful for:
    - Level design
    - Grid calibration
    - UI mockups

    Args:
        surface: Pygame display surface
    """
    render_background(surface)
    render_grid_lines(surface)


def render_team_separated(
    surface: pygame.Surface,
    blue_agents: List,
    red_agents: List,
    show_blue_fov: bool = True,
    show_red_fov: bool = True
) -> None:
    """
    Render with optional team-specific FOV visualization.

    Allows toggling FOV display per team for analysis or debugging.

    Args:
        surface: Pygame display surface
        blue_agents: List of blue team agents
        red_agents: List of red team agents
        show_blue_fov: Whether to show blue team FOV
        show_red_fov: Whether to show red team FOV
    """
    render_background(surface)
    render_grid_lines(surface)

    # Calculate FOV
    blue_fov, red_fov, overlap = get_fov_overlap(blue_agents, red_agents)

    # Conditionally render FOV based on flags
    blue_display = blue_fov if show_blue_fov else set()
    red_display = red_fov if show_red_fov else set()
    overlap_display = overlap if (show_blue_fov and show_red_fov) else set()

    render_fov_highlights(surface, blue_display, red_display, overlap_display)

    # Always render agents
    render_agents(surface, blue_agents)
    render_agents(surface, red_agents)


def render_with_highlights(
    surface: pygame.Surface,
    blue_agents: List,
    red_agents: List,
    highlight_cells: Set[Tuple[int, int]],
    highlight_color: Tuple[int, int, int, int] = (255, 255, 0, 80)
) -> None:
    """
    Render with custom cell highlighting (e.g., for waypoints, objectives).

    Demonstrates how to add custom overlay layers beyond FOV.

    Args:
        surface: Pygame display surface
        blue_agents: List of blue team agents
        red_agents: List of red team agents
        highlight_cells: Set of (x, y) grid cells to highlight
        highlight_color: RGBA color for highlight overlay (default: yellow)
    """
    from config import CELL_SIZE

    render_background(surface)
    render_grid_lines(surface)

    # Render custom highlights
    if highlight_cells:
        highlight_overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        highlight_overlay.fill(highlight_color)

        for cell_x, cell_y in highlight_cells:
            pixel_x = cell_x * CELL_SIZE
            pixel_y = cell_y * CELL_SIZE
            surface.blit(highlight_overlay, (pixel_x, pixel_y))

    # Render FOV
    blue_fov, red_fov, overlap = get_fov_overlap(blue_agents, red_agents)
    render_fov_highlights(surface, blue_fov, red_fov, overlap)

    # Render agents
    render_agents(surface, blue_agents)
    render_agents(surface, red_agents)


def render_with_stats(
    surface: pygame.Surface,
    blue_agents: List,
    red_agents: List,
    show_fps: bool = True,
    clock: pygame.time.Clock = None
) -> None:
    """
    Render complete scene with debug statistics overlay.

    Args:
        surface: Pygame display surface
        blue_agents: List of blue team agents
        red_agents: List of red team agents
        show_fps: Whether to display FPS counter
        clock: Pygame clock for FPS calculation
    """
    render_background(surface)
    render_grid_lines(surface)

    # FOV rendering
    blue_fov, red_fov, overlap = get_fov_overlap(blue_agents, red_agents)
    render_fov_highlights(surface, blue_fov, red_fov, overlap)

    # Agent rendering
    render_agents(surface, blue_agents)
    render_agents(surface, red_agents)

    # Stats overlay
    if show_fps and clock:
        fps = int(clock.get_fps())
        stats_text = f"FPS: {fps} | Blue: {len(blue_agents)} | Red: {len(red_agents)}"
        render_debug_info(surface, stats_text, (10, 10))


def render_single_agent_focus(
    surface: pygame.Surface,
    all_agents: List,
    focused_agent
) -> None:
    """
    Render with emphasis on a specific agent (e.g., for following camera).

    Demonstrates custom rendering for agent-centric views.

    Args:
        surface: Pygame display surface
        all_agents: List of all agents in the scene
        focused_agent: Agent to emphasize
    """
    from config import CELL_SIZE, AGENT_SIZE_RATIO
    from fov import get_fov_cells
    import math

    render_background(surface)
    render_grid_lines(surface)

    # Highlight focused agent's FOV only
    focused_fov = get_fov_cells(focused_agent.position, focused_agent.orientation)
    empty_set = set()

    if focused_agent.team == "blue":
        render_fov_highlights(surface, focused_fov, empty_set, empty_set)
    else:
        render_fov_highlights(surface, empty_set, focused_fov, empty_set)

    # Render all agents normally
    render_agents(surface, all_agents)

    # Draw emphasis ring around focused agent
    pixel_x = int(focused_agent.position[0] * CELL_SIZE)
    pixel_y = int(focused_agent.position[1] * CELL_SIZE)
    radius = int((CELL_SIZE / 2) * AGENT_SIZE_RATIO) + 3

    pygame.draw.circle(surface, (255, 255, 0), (pixel_x, pixel_y), radius, 2)


def render_layered_custom(
    surface: pygame.Surface,
    layers: List[str],
    blue_agents: List,
    red_agents: List
) -> None:
    """
    Custom layer-based rendering for experimental visualizations.

    Allows selective rendering of specific layers only.

    Args:
        surface: Pygame display surface
        layers: List of layer names to render (e.g., ['background', 'agents'])
        blue_agents: List of blue team agents
        red_agents: List of red team agents

    Supported layers: 'background', 'grid', 'fov', 'agents'
    """
    if 'background' in layers:
        render_background(surface)

    if 'grid' in layers:
        render_grid_lines(surface)

    if 'fov' in layers:
        blue_fov, red_fov, overlap = get_fov_overlap(blue_agents, red_agents)
        render_fov_highlights(surface, blue_fov, red_fov, overlap)

    if 'agents' in layers:
        render_agents(surface, blue_agents)
        render_agents(surface, red_agents)


# Example usage in main loop:
"""
# Standard usage
render_all(screen, blue_agents, red_agents, blue_fov, red_fov, overlap)

# Minimal (no FOV)
render_minimal(screen, blue_agents + red_agents)

# Toggle FOV per team
render_team_separated(screen, blue_agents, red_agents,
                      show_blue_fov=True, show_red_fov=False)

# With stats
render_with_stats(screen, blue_agents, red_agents, show_fps=True, clock=clock)

# Custom layers
render_layered_custom(screen, ['background', 'agents'], blue_agents, red_agents)
"""
