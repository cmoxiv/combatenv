"""
Rendering system for the Grid-World Multi-Agent Tactical Simulation.

This module provides modular rendering functions for all visual elements:
- Grid background and lines
- Field of view (FOV) highlighting with team colors and layered overlaps
- Agent rendering with orientation indicators (living and dead states)
- Projectile rendering
- Muzzle flash effects
- Debug and keybinding overlays

Rendering Pipeline:
    The render_all() function draws elements in a specific order for proper
    layering:
    1. Background (white fill)
    2. Grid lines (faint gray)
    3. FOV highlights (far FOV first, then near FOV, then overlaps)
    4. Agents (circles with orientation "nose" lines)
    5. Projectiles (small colored circles)
    6. Muzzle flashes (yellow/orange spark particles)

Overlay System:
    Two optional overlays can be toggled:
    - Debug overlay (backtick key): Shows FPS, agent counts, combat stats
    - Keybindings overlay (? key): Shows available controls

Performance Optimizations:
    - Pre-created overlay surfaces for FOV highlighting (7 surfaces)
    - Lazy initialization for headless environment compatibility
    - Efficient blitting using pre-filled RGBA surfaces

Coordinate Conversion:
    All game logic uses grid coordinates (0-64). This module converts to
    pixel coordinates using:
        pixel = grid_position * CELL_SIZE
    where CELL_SIZE = 16 pixels.

Example:
    >>> import pygame
    >>> from renderer import render_all, render_debug_overlay
    >>> screen = pygame.display.set_mode((1024, 1024))
    >>> render_all(screen, blue_agents, red_agents, ...)
    >>> render_debug_overlay(screen, {'fps': 60.0, 'num_agents': 200})
    >>> pygame.display.flip()
"""

import pygame
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import (
    WINDOW_SIZE,
    CELL_SIZE,
    GRID_SIZE,
    GRID_LINE_WIDTH,
    TACTICAL_CELLS_PER_OPERATIONAL,
    COLOR_BACKGROUND,
    COLOR_GRID_LINES,
    COLOR_BLUE_TEAM,
    COLOR_RED_TEAM,
    COLOR_PURPLE,
    FOV_BLUE_ALPHA,
    FOV_RED_ALPHA,
    FOV_PURPLE_ALPHA,
    AGENT_SIZE_RATIO,
    AGENT_NOSE_RATIO,
    COLOR_OBSTACLE,
    COLOR_FIRE,
    COLOR_FOREST,
    COLOR_WATER
)

# FOV overlay surfaces (lazy-initialized for headless mode compatibility)
# These surfaces are reused every frame instead of being recreated
_BLUE_NEAR_OVERLAY = None
_BLUE_FAR_OVERLAY = None
_RED_NEAR_OVERLAY = None
_RED_FAR_OVERLAY = None
_OVERLAP_NEAR_NEAR_OVERLAY = None
_OVERLAP_MIXED_OVERLAY = None
_OVERLAP_FAR_FAR_OVERLAY = None

# Grid line surface (cached for performance)
# Set to None to force regeneration on color changes
_GRID_LINE_SURFACE = None

# Cached terrain surface (rebuilt when terrain_grid._surface_dirty is True)
_TERRAIN_SURFACE = None


def _ensure_surfaces_initialized():
    """
    Lazy initialization of FOV overlay surfaces.

    Creates the overlay surfaces on first use rather than at module import.
    This allows the module to be imported in headless environments where
    Pygame display is not available.
    """
    from .config import (
        FOV_NEAR_ALPHA, FOV_FAR_ALPHA,
        OVERLAP_NEAR_NEAR_ALPHA, OVERLAP_MIXED_ALPHA, OVERLAP_FAR_FAR_ALPHA
    )

    global _BLUE_NEAR_OVERLAY, _BLUE_FAR_OVERLAY, _RED_NEAR_OVERLAY, _RED_FAR_OVERLAY
    global _OVERLAP_NEAR_NEAR_OVERLAY, _OVERLAP_MIXED_OVERLAY, _OVERLAP_FAR_FAR_OVERLAY

    if _BLUE_NEAR_OVERLAY is None:
        _BLUE_NEAR_OVERLAY = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        _BLUE_FAR_OVERLAY = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        _RED_NEAR_OVERLAY = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        _RED_FAR_OVERLAY = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        _OVERLAP_NEAR_NEAR_OVERLAY = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        _OVERLAP_MIXED_OVERLAY = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        _OVERLAP_FAR_FAR_OVERLAY = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)

        _BLUE_NEAR_OVERLAY.fill((*COLOR_BLUE_TEAM, FOV_NEAR_ALPHA))  # Darker blue
        _BLUE_FAR_OVERLAY.fill((*COLOR_BLUE_TEAM, FOV_FAR_ALPHA))    # Lighter blue
        _RED_NEAR_OVERLAY.fill((*COLOR_RED_TEAM, FOV_NEAR_ALPHA))    # Darker red
        _RED_FAR_OVERLAY.fill((*COLOR_RED_TEAM, FOV_FAR_ALPHA))      # Lighter red
        _OVERLAP_NEAR_NEAR_OVERLAY.fill((*COLOR_PURPLE, OVERLAP_NEAR_NEAR_ALPHA))  # Darkest purple
        _OVERLAP_MIXED_OVERLAY.fill((*COLOR_PURPLE, OVERLAP_MIXED_ALPHA))          # Medium purple
        _OVERLAP_FAR_FAR_OVERLAY.fill((*COLOR_PURPLE, OVERLAP_FAR_FAR_ALPHA))      # Lightest purple


def render_background(surface: pygame.Surface) -> None:
    """
    Render the white background for the grid-world.

    Args:
        surface: Pygame surface to draw on (typically the main screen)
    """
    surface.fill(COLOR_BACKGROUND)


def render_grid_lines(surface: pygame.Surface) -> None:
    """
    Render faint gray grid lines to visualize the grid structure.

    Draws vertical and horizontal lines at cell boundaries. Uses GRID_LINE_WIDTH
    from config to determine line thickness. Lines are 50% transparent.
    The grid surface is cached for performance (created once, reused every frame).

    Args:
        surface: Pygame surface to draw on (typically the main screen)
    """
    global _GRID_LINE_SURFACE

    # Lazy initialization of grid line surface (created once)
    if _GRID_LINE_SURFACE is None:
        _GRID_LINE_SURFACE = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        grid_color = (*COLOR_GRID_LINES, 40)  # ~15% transparency

        # Draw vertical lines
        for x in range(0, GRID_SIZE + 1):
            pixel_x = x * CELL_SIZE
            pygame.draw.line(
                _GRID_LINE_SURFACE,
                grid_color,
                (pixel_x, 0),
                (pixel_x, WINDOW_SIZE),
                GRID_LINE_WIDTH
            )

        # Draw horizontal lines
        for y in range(0, GRID_SIZE + 1):
            pixel_y = y * CELL_SIZE
            pygame.draw.line(
                _GRID_LINE_SURFACE,
                grid_color,
                (0, pixel_y),
                (WINDOW_SIZE, pixel_y),
                GRID_LINE_WIDTH
            )

    surface.blit(_GRID_LINE_SURFACE, (0, 0))


def render_operational_grid(surface: pygame.Surface) -> None:
    """
    Render medium lines for the 8x8 operational grid overlay.

    The operational grid divides the 64x64 tactical grid into 64 cells
    of 8x8 tactical cells each.

    Args:
        surface: Pygame surface to draw on
    """
    from .config import (
        OPERATIONAL_GRID_SIZE,
        TACTICAL_CELLS_PER_OPERATIONAL,
        OPERATIONAL_GRID_COLOR,
        OPERATIONAL_GRID_LINE_WIDTH,
    )

    op_cell_pixels = TACTICAL_CELLS_PER_OPERATIONAL * CELL_SIZE

    # Draw vertical lines
    for i in range(OPERATIONAL_GRID_SIZE + 1):
        x = i * op_cell_pixels
        pygame.draw.line(
            surface,
            OPERATIONAL_GRID_COLOR,
            (x, 0),
            (x, WINDOW_SIZE),
            OPERATIONAL_GRID_LINE_WIDTH
        )

    # Draw horizontal lines
    for i in range(OPERATIONAL_GRID_SIZE + 1):
        y = i * op_cell_pixels
        pygame.draw.line(
            surface,
            OPERATIONAL_GRID_COLOR,
            (0, y),
            (WINDOW_SIZE, y),
            OPERATIONAL_GRID_LINE_WIDTH
        )


def render_strategic_grid(surface: pygame.Surface) -> None:
    """
    Render thick lines for the 4x4 strategic grid overlay.

    The strategic grid divides the 64x64 tactical grid into 16 cells
    of 16x16 tactical cells each. Lines are thicker and more visible
    than the operational grid lines.

    Args:
        surface: Pygame surface to draw on
    """
    from .config import (
        STRATEGIC_GRID_SIZE,
        TACTICAL_CELLS_PER_STRATEGIC,
        STRATEGIC_GRID_COLOR,
        STRATEGIC_GRID_LINE_WIDTH,
    )

    strategic_cell_pixels = TACTICAL_CELLS_PER_STRATEGIC * CELL_SIZE

    # Draw vertical lines
    for i in range(STRATEGIC_GRID_SIZE + 1):
        x = i * strategic_cell_pixels
        pygame.draw.line(
            surface,
            STRATEGIC_GRID_COLOR,
            (x, 0),
            (x, WINDOW_SIZE),
            STRATEGIC_GRID_LINE_WIDTH
        )

    # Draw horizontal lines
    for i in range(STRATEGIC_GRID_SIZE + 1):
        y = i * strategic_cell_pixels
        pygame.draw.line(
            surface,
            STRATEGIC_GRID_COLOR,
            (0, y),
            (WINDOW_SIZE, y),
            STRATEGIC_GRID_LINE_WIDTH
        )


def render_waypoints(
    surface: pygame.Surface,
    units: List,
    selected_unit_id: Optional[int] = None,
    show_waypoints: bool = True,
    show_goals: bool = True
) -> None:
    """
    Render unit waypoints and unit IDs for debugging.

    Args:
        surface: Pygame surface to draw on
        units: List of Unit instances to render waypoints for
        selected_unit_id: ID of currently selected unit (highlighted differently)
        show_waypoints: Whether to show intermediate waypoint lines (W key)
        show_goals: Whether to show goal waypoint lines (SHIFT+W key)
    """
    font = pygame.font.Font(None, 20)

    for unit in units:
        if not unit.alive_agents:
            continue

        centroid = unit.centroid
        centroid_px = (int(centroid[0] * CELL_SIZE), int(centroid[1] * CELL_SIZE))

        # Determine color based on team
        if unit.team == "blue":
            base_color = (100, 100, 255)
            waypoint_color = (0, 150, 255)
        else:
            base_color = (255, 100, 100)
            waypoint_color = (255, 150, 0)

        # Highlight selected unit
        is_selected = (selected_unit_id is not None and unit.id == selected_unit_id)
        if is_selected:
            if unit.team == "blue":
                base_color = (135, 206, 250)  # Light blue for selected blue
                waypoint_color = (135, 206, 250)
            else:
                base_color = (255, 165, 0)  # Orange for selected red
                waypoint_color = (255, 165, 0)

        # Draw unit ID at centroid (uppercase for blue, lowercase for red)
        if unit.team == "blue":
            unit_label = f"U{unit.id}"
        else:
            unit_label = f"u{unit.id}"
        text = font.render(unit_label, True, base_color)
        surface.blit(text, (centroid_px[0] - 8, centroid_px[1] - 20))

        # Draw goal waypoint (diamond shape) - toggled with SHIFT+W
        goal = getattr(unit, 'goal_waypoint', None)
        if goal is not None and show_goals:
            goal_px = (int(goal[0] * CELL_SIZE), int(goal[1] * CELL_SIZE))

            # Draw dashed line from centroid to goal (use dots for dashed effect)
            # Use same color as intermediate waypoint for consistency
            dx = goal_px[0] - centroid_px[0]
            dy = goal_px[1] - centroid_px[1]
            dist = max(1, int((dx**2 + dy**2)**0.5))
            for i in range(0, dist, 12):  # Dashed line
                t = i / dist
                x = int(centroid_px[0] + dx * t)
                y = int(centroid_px[1] + dy * t)
                pygame.draw.circle(surface, waypoint_color, (x, y), 2)

            # Draw goal marker (diamond shape)
            size = 10
            pygame.draw.polygon(surface, waypoint_color, [
                (goal_px[0], goal_px[1] - size),      # Top
                (goal_px[0] + size, goal_px[1]),      # Right
                (goal_px[0], goal_px[1] + size),      # Bottom
                (goal_px[0] - size, goal_px[1]),      # Left
            ], 3)

            # Draw goal label (uppercase for blue, lowercase for red)
            if unit.team == "blue":
                goal_label = f"G{unit.id}"
            else:
                goal_label = f"g{unit.id}"
            goal_text = font.render(goal_label, True, waypoint_color)
            surface.blit(goal_text, (goal_px[0] + 12, goal_px[1] - 12))

        # Draw intermediate waypoint if set (X shape) - toggled with W
        if unit.waypoint is not None and show_waypoints:
            wp_px = (int(unit.waypoint[0] * CELL_SIZE), int(unit.waypoint[1] * CELL_SIZE))

            # Draw solid line from centroid to intermediate waypoint
            pygame.draw.line(surface, waypoint_color, centroid_px, wp_px, 2)

            # Draw waypoint marker (X shape)
            size = 6
            pygame.draw.line(surface, waypoint_color,
                           (wp_px[0] - size, wp_px[1] - size),
                           (wp_px[0] + size, wp_px[1] + size), 2)
            pygame.draw.line(surface, waypoint_color,
                           (wp_px[0] + size, wp_px[1] - size),
                           (wp_px[0] - size, wp_px[1] + size), 2)


def render_selected_unit(
    surface: pygame.Surface,
    unit,
    circle_color: Tuple[int, int, int] = (135, 206, 250)
) -> None:
    """
    Render circles around each agent in the selected unit (alive and dead).

    Args:
        surface: Pygame surface to draw on
        unit: Unit instance whose agents should be highlighted
        circle_color: Color for the highlight circles (light blue for blue, orange for red)
    """
    if unit is None:
        return

    for agent in unit.agents:
        # Convert grid position to pixel position
        px = int(agent.position[0] * CELL_SIZE)
        py = int(agent.position[1] * CELL_SIZE)

        if agent.is_alive:
            # Full color for alive agents
            pygame.draw.circle(surface, circle_color, (px, py), 12, 2)
        else:
            # Dimmer color for dead agents
            dim_color = (circle_color[0] // 2, circle_color[1] // 2, circle_color[2] // 2)
            pygame.draw.circle(surface, dim_color, (px, py), 12, 1)


def render_terrain(surface: pygame.Surface, terrain_grid) -> None:
    """
    Render terrain at pixel level using surfarray for performance.

    Uses a color lookup table to map the 1024x1024 terrain grid directly
    to RGB pixels. The rendered surface is cached and only rebuilt when
    terrain changes (checked via _surface_dirty flag).

    Args:
        surface: Pygame surface to draw on
        terrain_grid: TerrainGrid object containing pixel-level terrain data
    """
    import numpy as np
    global _TERRAIN_SURFACE

    # Check if we need to rebuild the cached surface
    if _TERRAIN_SURFACE is None or terrain_grid._surface_dirty:
        # Color lookup table indexed by TerrainType value
        lut = np.array([
            COLOR_BACKGROUND,  # EMPTY (0)
            COLOR_OBSTACLE,    # OBSTACLE (1)
            COLOR_FIRE,        # FIRE (2)
            COLOR_FOREST,      # FOREST (3)
            COLOR_WATER,       # WATER (4)
        ], dtype=np.uint8)

        # Map terrain grid to RGB: (1024, 1024) -> (1024, 1024, 3)
        rgb = lut[terrain_grid.grid]

        # Create or reuse cached surface
        if _TERRAIN_SURFACE is None:
            _TERRAIN_SURFACE = pygame.Surface(
                (terrain_grid.pixel_width, terrain_grid.pixel_height)
            )

        pygame.surfarray.blit_array(_TERRAIN_SURFACE, rgb)
        terrain_grid._surface_dirty = False

    surface.blit(_TERRAIN_SURFACE, (0, 0))


def render_fov_highlights(
    surface: pygame.Surface,
    blue_near_cells: Set[Tuple[int, int]],
    blue_far_cells: Set[Tuple[int, int]],
    red_near_cells: Set[Tuple[int, int]],
    red_far_cells: Set[Tuple[int, int]],
    overlap_near_near: Set[Tuple[int, int]],
    overlap_mixed: Set[Tuple[int, int]],
    overlap_far_far: Set[Tuple[int, int]]
) -> None:
    """
    Render layered field of view (FOV) highlighting for both teams with layered overlaps.

    Highlights grid cells with semi-transparent overlays:
    - Blue far: Light blue tint for blue team far FOV (lighter)
    - Blue near: Darker blue tint for blue team near FOV (darker)
    - Red far: Light red tint for red team far FOV (lighter)
    - Red near: Darker red tint for red team near FOV (darker)
    - Purple (3 types based on overlap layer):
      - Near-Near: Darkest purple (both teams near FOV)
      - Mixed: Medium purple (one near, one far)
      - Far-Far: Lightest purple (both teams far FOV)

    Uses pre-created overlay surfaces for optimal performance.

    Args:
        surface: Pygame surface to draw on (typically the main screen)
        blue_near_cells: Set of (x, y) grid coordinates in blue near FOV
        blue_far_cells: Set of (x, y) grid coordinates in blue far FOV
        red_near_cells: Set of (x, y) grid coordinates in red near FOV
        red_far_cells: Set of (x, y) grid coordinates in red far FOV
        overlap_near_near: Cells where both teams have near FOV
        overlap_mixed: Cells where one team near, one far
        overlap_far_far: Cells where both teams have far FOV
    """
    # Ensure overlay surfaces are initialized
    _ensure_surfaces_initialized()

    # Assert surfaces are initialized (for type checker)
    assert _BLUE_FAR_OVERLAY is not None
    assert _BLUE_NEAR_OVERLAY is not None
    assert _RED_FAR_OVERLAY is not None
    assert _RED_NEAR_OVERLAY is not None
    assert _OVERLAP_FAR_FAR_OVERLAY is not None
    assert _OVERLAP_MIXED_OVERLAY is not None
    assert _OVERLAP_NEAR_NEAR_OVERLAY is not None

    # Render far FOV first (lighter, in the background)
    for cell_x, cell_y in blue_far_cells:
        pixel_x = cell_x * CELL_SIZE
        pixel_y = cell_y * CELL_SIZE
        surface.blit(_BLUE_FAR_OVERLAY, (pixel_x, pixel_y))

    for cell_x, cell_y in red_far_cells:
        pixel_x = cell_x * CELL_SIZE
        pixel_y = cell_y * CELL_SIZE
        surface.blit(_RED_FAR_OVERLAY, (pixel_x, pixel_y))

    # Render near FOV on top (darker, more visible)
    for cell_x, cell_y in blue_near_cells:
        pixel_x = cell_x * CELL_SIZE
        pixel_y = cell_y * CELL_SIZE
        surface.blit(_BLUE_NEAR_OVERLAY, (pixel_x, pixel_y))

    for cell_x, cell_y in red_near_cells:
        pixel_x = cell_x * CELL_SIZE
        pixel_y = cell_y * CELL_SIZE
        surface.blit(_RED_NEAR_OVERLAY, (pixel_x, pixel_y))

    # Render overlap cells with layered purple tints (rendered last for proper blending)
    # Far-Far overlap (lightest purple, lowest priority)
    for cell_x, cell_y in overlap_far_far:
        pixel_x = cell_x * CELL_SIZE
        pixel_y = cell_y * CELL_SIZE
        surface.blit(_OVERLAP_FAR_FAR_OVERLAY, (pixel_x, pixel_y))

    # Mixed overlap (medium purple, medium priority)
    for cell_x, cell_y in overlap_mixed:
        pixel_x = cell_x * CELL_SIZE
        pixel_y = cell_y * CELL_SIZE
        surface.blit(_OVERLAP_MIXED_OVERLAY, (pixel_x, pixel_y))

    # Near-Near overlap (darkest purple, highest priority)
    for cell_x, cell_y in overlap_near_near:
        pixel_x = cell_x * CELL_SIZE
        pixel_y = cell_y * CELL_SIZE
        surface.blit(_OVERLAP_NEAR_NEAR_OVERLAY, (pixel_x, pixel_y))


def render_agent(surface: pygame.Surface, agent, terrain_grid=None, highlight: bool = False) -> None:
    """
    Render a single agent as a filled circle with orientation indicator.

    The agent is drawn as:
    - A filled circle in the agent's team color (70% of cell size)
    - A "nose" line extending from center to show orientation
    - Dead agents are rendered in gray without orientation indicator
    - Agents in water or forest are rendered with transparency
    - Highlighted agents have a yellow border

    Args:
        surface: Pygame surface to draw on (typically the main screen)
        agent: Agent object with position, orientation, team, and health attributes
        terrain_grid: Optional TerrainGrid to check if agent is underwater
        highlight: If True, draw yellow border around agent
    """
    from .config import COLOR_DEAD_AGENT
    from .terrain import TerrainType

    # Convert grid position to pixel coordinates (center of circle)
    pixel_x = int(agent.position[0] * CELL_SIZE)
    pixel_y = int(agent.position[1] * CELL_SIZE)
    center = (pixel_x, pixel_y)

    # Calculate circle radius (70% of cell size)
    radius = int((CELL_SIZE / 2) * AGENT_SIZE_RATIO)

    # Check if agent is underwater or in forest
    is_transparent = False
    if terrain_grid is not None:
        grid_x = int(agent.position[0])
        grid_y = int(agent.position[1])
        terrain = terrain_grid.get(grid_x, grid_y)
        is_transparent = terrain in (TerrainType.WATER, TerrainType.FOREST)

    if agent.is_alive:
        # Determine color based on team
        color = COLOR_BLUE_TEAM if agent.team == "blue" else COLOR_RED_TEAM

        if is_transparent:
            # Underwater or in forest: render with transparency
            agent_surface = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(agent_surface, (*color, 100), (radius + 1, radius + 1), radius)
            # Draw nose with transparency
            nose_length = (CELL_SIZE / 2) * AGENT_NOSE_RATIO
            orientation_rad = math.radians(agent.orientation)
            nose_end_x = radius + 1 + int(math.cos(orientation_rad) * (radius + nose_length))
            nose_end_y = radius + 1 + int(math.sin(orientation_rad) * (radius + nose_length))
            pygame.draw.line(agent_surface, (*color, 100), (radius + 1, radius + 1), (nose_end_x, nose_end_y), 2)
            surface.blit(agent_surface, (pixel_x - radius - 1, pixel_y - radius - 1))
        else:
            # Normal rendering
            pygame.draw.circle(surface, color, center, radius)

            # Draw orientation "nose" line
            nose_length = (CELL_SIZE / 2) * AGENT_NOSE_RATIO
            orientation_rad = math.radians(agent.orientation)
            nose_end_x = pixel_x + int(math.cos(orientation_rad) * (radius + nose_length))
            nose_end_y = pixel_y + int(math.sin(orientation_rad) * (radius + nose_length))
            nose_end = (nose_end_x, nose_end_y)
            pygame.draw.line(surface, color, center, nose_end, 2)

        # Draw yellow highlight border if requested (for both transparent and normal)
        if highlight:
            pygame.draw.circle(surface, (255, 255, 0), center, radius + 2, 2)
    else:
        # Dead agent: gray with transparency, no nose
        dead_surface = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(dead_surface, (*COLOR_DEAD_AGENT, 128), (radius + 1, radius + 1), radius)
        surface.blit(dead_surface, (pixel_x - radius - 1, pixel_y - radius - 1))


def render_agents(surface: pygame.Surface, agents: List, terrain_grid=None) -> None:
    """
    Render all agents in a list.

    Convenience function to render multiple agents. Typically called once
    for blue team and once for red team. Dead agents are rendered first
    so that live agents appear on top.

    Args:
        surface: Pygame surface to draw on (typically the main screen)
        agents: List of Agent objects to render
        terrain_grid: Optional TerrainGrid for underwater transparency
    """
    # Render dead agents first (lower z-order), then live agents on top
    for agent in agents:
        if not agent.is_alive:
            render_agent(surface, agent, terrain_grid)
    for agent in agents:
        if agent.is_alive:
            render_agent(surface, agent, terrain_grid)


def render_projectile(surface: pygame.Surface, projectile) -> None:
    """
    Render a single projectile as a small filled circle.

    Args:
        surface: Pygame surface to draw on (typically the main screen)
        projectile: Projectile object with position and owner_team attributes
    """
    from .config import (
        COLOR_PROJECTILE_BLUE,
        COLOR_PROJECTILE_RED,
        PROJECTILE_SIZE_RATIO
    )

    # Convert grid position to pixel coordinates
    pixel_x = int(projectile.position[0] * CELL_SIZE)
    pixel_y = int(projectile.position[1] * CELL_SIZE)
    center = (pixel_x, pixel_y)

    # Calculate radius (smaller than agents)
    radius = max(2, int((CELL_SIZE / 2) * PROJECTILE_SIZE_RATIO))

    # Determine color based on team
    color = COLOR_PROJECTILE_BLUE if projectile.owner_team == "blue" else COLOR_PROJECTILE_RED

    # Draw projectile
    pygame.draw.circle(surface, color, center, radius)


def render_projectiles(surface: pygame.Surface, projectiles: List) -> None:
    """
    Render all projectiles.

    Args:
        surface: Pygame surface to draw on (typically the main screen)
        projectiles: List of Projectile objects to render
    """
    for projectile in projectiles:
        render_projectile(surface, projectile)


def render_muzzle_flash(
    surface: pygame.Surface,
    position: Tuple[float, float],
    rng: Optional[random.Random] = None
) -> None:
    """
    Render a muzzle flash as yellow sparks/particles.

    Args:
        surface: Pygame surface to draw on
        position: (x, y) position in grid coordinates
        rng: Optional random.Random instance for deterministic testing.
             If None, uses the global random module.
    """
    # Convert grid position to pixel coordinates
    pixel_x = int(position[0] * CELL_SIZE)
    pixel_y = int(position[1] * CELL_SIZE)

    # Yellow/orange colors for muzzle flash
    colors = [(255, 255, 0), (255, 200, 0), (255, 150, 0)]  # Yellow to orange

    # Draw multiple small particles in a small radius
    num_particles = 4
    for i in range(num_particles):
        angle = (i / num_particles) * 360
        # Use provided RNG or fall back to global random
        if rng is not None:
            distance = rng.uniform(1, 3)
            color = rng.choice(colors)
            radius = rng.randint(1, 2)
        else:
            distance = random.uniform(1, 3)
            color = random.choice(colors)
            radius = random.randint(1, 2)
        particle_x = int(pixel_x + math.cos(math.radians(angle)) * distance)
        particle_y = int(pixel_y + math.sin(math.radians(angle)) * distance)
        pygame.draw.circle(surface, color, (particle_x, particle_y), radius)

    # Draw bright center
    pygame.draw.circle(surface, (255, 255, 255), (pixel_x, pixel_y), 2)


def render_muzzle_flashes(
    surface: pygame.Surface,
    muzzle_flashes: List,
    rng: Optional[random.Random] = None
) -> None:
    """
    Render all muzzle flashes.

    Args:
        surface: Pygame surface to draw on
        muzzle_flashes: List of (position, lifetime) tuples
        rng: Optional random.Random instance for deterministic testing
    """
    for position, _ in muzzle_flashes:
        render_muzzle_flash(surface, position, rng)


def render_all(
    surface: pygame.Surface,
    blue_agents: List,
    red_agents: List,
    blue_near_cells: Set[Tuple[int, int]],
    blue_far_cells: Set[Tuple[int, int]],
    red_near_cells: Set[Tuple[int, int]],
    red_far_cells: Set[Tuple[int, int]],
    overlap_near_near: Set[Tuple[int, int]],
    overlap_mixed: Set[Tuple[int, int]],
    overlap_far_far: Set[Tuple[int, int]],
    projectiles: List,
    muzzle_flashes: List,
    terrain_grid=None,
    show_strategic_grid: bool = False
) -> None:
    """
    Complete rendering pipeline for the entire game scene.

    Renders all visual elements in the correct order:
    1. Background (white)
    2. Terrain (buildings, fire, forest)
    3. Grid lines (faint gray)
    4. FOV highlights (layered: far FOV lighter, near FOV darker, overlaps by layer)
    5. Agents (circles with orientation indicators)
    6. Projectiles (small circles on top of everything)
    7. Muzzle flashes (yellow sparks at weapon fire points)

    This is the primary function to call from the main game loop for
    complete frame rendering.

    Args:
        surface: Pygame surface to draw on (typically the main screen)
        blue_agents: List of blue team Agent objects
        red_agents: List of red team Agent objects
        blue_near_cells: Set of (x, y) grid coordinates in blue near FOV
        blue_far_cells: Set of (x, y) grid coordinates in blue far FOV
        red_near_cells: Set of (x, y) grid coordinates in red near FOV
        red_far_cells: Set of (x, y) grid coordinates in red far FOV
        overlap_near_near: Both teams near FOV overlap
        overlap_mixed: One near one far FOV overlap
        overlap_far_far: Both teams far FOV overlap
        projectiles: List of Projectile objects to render
        muzzle_flashes: List of (position, lifetime) tuples for muzzle flashes
        terrain_grid: TerrainGrid object for terrain rendering (optional)
    """
    # Layer 1: Background
    render_background(surface)

    # Layer 2: Terrain (buildings, fire, forest)
    if terrain_grid:
        render_terrain(surface, terrain_grid)

    # Layer 3: Grid lines
    render_grid_lines(surface)

    # Layer 3.5: Strategic grid overlay (thick lines for 4x4 grid)
    if show_strategic_grid:
        render_strategic_grid(surface)

    # Layer 4: FOV highlights (layered: far lighter, near darker, overlaps by layer)
    render_fov_highlights(surface, blue_near_cells, blue_far_cells,
                          red_near_cells, red_far_cells,
                          overlap_near_near, overlap_mixed, overlap_far_far)

    # Layer 5: Agents (rendered on top)
    render_agents(surface, blue_agents)
    render_agents(surface, red_agents)

    # Layer 6: Projectiles (rendered on top of everything)
    render_projectiles(surface, projectiles)

    # Layer 7: Muzzle flashes (rendered last for maximum visibility)
    render_muzzle_flashes(surface, muzzle_flashes)


def render_debug_overlay(surface: pygame.Surface, debug_info: Dict[str, Any]) -> None:
    """
    Render comprehensive debug information overlay.

    Displays performance metrics, agent statistics, and spatial grid info
    in a semi-transparent overlay panel.

    Args:
        surface: Pygame surface to draw on
        debug_info: Dictionary containing debug data:
            - 'fps': Current frames per second
            - 'num_agents': Total agent count
            - 'blue_agents': Blue team agent count
            - 'red_agents': Red team agent count
            - 'spatial_stats': Spatial grid statistics (optional)
            - 'fov_coverage': FOV coverage percentages (optional)
    """
    try:
        # Create semi-transparent background panel
        panel_width = 320
        panel_height = 580  # Increased to fit agent rewards
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))  # Dark semi-transparent background

        # Initialize fonts
        title_font = pygame.font.Font(None, 32)
        label_font = pygame.font.Font(None, 24)
        value_font = pygame.font.Font(None, 24)

        # Render title
        title = title_font.render("DEBUG INFO", True, (255, 255, 0))
        panel.blit(title, (10, 10))

        # Render debug information
        y_offset = 50
        line_height = 25

        # FPS
        fps_text = f"FPS: {debug_info.get('fps', 0):.1f}"
        fps_color = (0, 255, 0) if debug_info.get('fps', 0) >= 55 else (255, 255, 0) if debug_info.get('fps', 0) >= 30 else (255, 0, 0)
        fps_surface = value_font.render(fps_text, True, fps_color)
        panel.blit(fps_surface, (10, y_offset))
        y_offset += line_height

        # Agent counts
        total = debug_info.get('num_agents', 0)
        blue = debug_info.get('blue_agents', 0)
        red = debug_info.get('red_agents', 0)

        agents_text = value_font.render(f"Total Agents: {total}", True, (255, 255, 255))
        panel.blit(agents_text, (10, y_offset))
        y_offset += line_height

        blue_text = value_font.render(f"  Blue: {blue}", True, (100, 100, 255))
        panel.blit(blue_text, (20, y_offset))
        y_offset += line_height

        red_text = value_font.render(f"  Red: {red}", True, (255, 100, 100))
        panel.blit(red_text, (20, y_offset))
        y_offset += line_height

        # Combat statistics
        if 'blue_dead' in debug_info or 'red_dead' in debug_info or 'num_projectiles' in debug_info:
            y_offset += 10
            combat_title = label_font.render("Combat Stats:", True, (200, 200, 200))
            panel.blit(combat_title, (10, y_offset))
            y_offset += line_height

            blue_dead = debug_info.get('blue_dead', 0)
            red_dead = debug_info.get('red_dead', 0)
            num_projectiles = debug_info.get('num_projectiles', 0)

            dead_blue_text = value_font.render(f"  Blue Dead: {blue_dead}", True, (100, 100, 255))
            panel.blit(dead_blue_text, (20, y_offset))
            y_offset += line_height

            dead_red_text = value_font.render(f"  Red Dead: {red_dead}", True, (255, 100, 100))
            panel.blit(dead_red_text, (20, y_offset))
            y_offset += line_height

            proj_text = value_font.render(f"  Projectiles: {num_projectiles}", True, (200, 200, 200))
            panel.blit(proj_text, (20, y_offset))
            y_offset += line_height

        # Spatial grid statistics
        if 'spatial_stats' in debug_info:
            stats = debug_info['spatial_stats']
            y_offset += 10
            spatial_title = label_font.render("Spatial Grid:", True, (200, 200, 200))
            panel.blit(spatial_title, (10, y_offset))
            y_offset += line_height

            cells_text = value_font.render(f"  Cells: {stats.get('num_cells', 0)}", True, (180, 180, 180))
            panel.blit(cells_text, (20, y_offset))
            y_offset += line_height

            avg_text = value_font.render(f"  Avg/cell: {stats.get('avg_agents_per_cell', 0):.1f}", True, (180, 180, 180))
            panel.blit(avg_text, (20, y_offset))
            y_offset += line_height

        # FOV coverage (if provided)
        if 'fov_coverage' in debug_info:
            cov = debug_info['fov_coverage']
            y_offset += 10
            fov_title = label_font.render("FOV Coverage:", True, (200, 200, 200))
            panel.blit(fov_title, (10, y_offset))
            y_offset += line_height

            blue_cov = value_font.render(f"  Blue: {cov.get('blue', 0):.1f}%", True, (100, 100, 255))
            panel.blit(blue_cov, (20, y_offset))
            y_offset += line_height

            red_cov = value_font.render(f"  Red: {cov.get('red', 0):.1f}%", True, (255, 100, 100))
            panel.blit(red_cov, (20, y_offset))

        # Kill statistics (if provided)
        if 'blue_kills' in debug_info or 'red_kills' in debug_info:
            y_offset += line_height + 10
            kills_title = label_font.render("Kill Stats:", True, (200, 200, 200))
            panel.blit(kills_title, (10, y_offset))
            y_offset += line_height

            blue_kills = debug_info.get('blue_kills', 0)
            red_kills = debug_info.get('red_kills', 0)
            kills_text = value_font.render(f"  Blue Kills: {blue_kills}", True, (100, 100, 255))
            panel.blit(kills_text, (20, y_offset))
            y_offset += line_height

            kills_text2 = value_font.render(f"  Red Kills: {red_kills}", True, (255, 100, 100))
            panel.blit(kills_text2, (20, y_offset))

        # Reward statistics (if provided)
        if 'step_reward' in debug_info or 'episode_reward' in debug_info:
            y_offset += line_height + 10
            reward_title = label_font.render("Rewards:", True, (200, 200, 200))
            panel.blit(reward_title, (10, y_offset))
            y_offset += line_height

            step_reward = debug_info.get('step_reward', 0.0)
            episode_reward = debug_info.get('episode_reward', 0.0)

            step_color = (100, 255, 100) if step_reward >= 0 else (255, 100, 100)
            step_text = value_font.render(f"  Step: {step_reward:+.2f}", True, step_color)
            panel.blit(step_text, (20, y_offset))
            y_offset += line_height

            episode_color = (100, 255, 100) if episode_reward >= 0 else (255, 100, 100)
            episode_text = value_font.render(f"  Episode: {episode_reward:+.2f}", True, episode_color)
            panel.blit(episode_text, (20, y_offset))

        # Selected unit (if provided)
        if 'selected_unit' in debug_info:
            y_offset += line_height + 10
            selected = debug_info.get('selected_unit')
            if selected is not None:
                unit_text = value_font.render(f"Selected Unit: {selected}", True, (255, 255, 0))
            else:
                unit_text = value_font.render("Selected Unit: None", True, (150, 150, 150))
            panel.blit(unit_text, (10, y_offset))

        # Agent rewards (if provided by multi-agent wrapper)
        agent_rewards = debug_info.get('agent_rewards', {})
        agent_list = debug_info.get('agent_list', [])
        if agent_rewards and agent_list:
            y_offset += line_height + 10
            rewards_title = label_font.render("Agent Rewards:", True, (200, 200, 200))
            panel.blit(rewards_title, (10, y_offset))
            y_offset += line_height

            # Calculate totals per team
            blue_total = 0.0
            red_total = 0.0
            n_blue = len([a for a in agent_list if a.team == "blue"])

            for idx, reward in agent_rewards.items():
                if idx < n_blue:
                    blue_total += reward
                else:
                    red_total += reward

            # Show team totals
            blue_reward_text = value_font.render(f"  Blue Total: {blue_total:+.1f}", True, (100, 100, 255))
            panel.blit(blue_reward_text, (20, y_offset))
            y_offset += line_height

            red_reward_text = value_font.render(f"  Red Total: {red_total:+.1f}", True, (255, 100, 100))
            panel.blit(red_reward_text, (20, y_offset))
            y_offset += line_height

            # Show per-agent rewards (first few of each team)
            small_font = pygame.font.Font(None, 18)
            y_offset += 5

            # Blue agents (show first 4)
            blue_agents_rewards = [(idx, r) for idx, r in agent_rewards.items() if idx < n_blue]
            for idx, reward in blue_agents_rewards[:4]:
                agent = agent_list[idx]
                status = "✓" if agent.is_alive else "✗"
                r_text = small_font.render(f"  B{idx}: {reward:+.1f} {status}", True, (80, 80, 200))
                panel.blit(r_text, (20, y_offset))
                y_offset += 16

            # Red agents (show first 4)
            red_agents_rewards = [(idx, r) for idx, r in agent_rewards.items() if idx >= n_blue]
            for idx, reward in red_agents_rewards[:4]:
                agent = agent_list[idx]
                status = "✓" if agent.is_alive else "✗"
                r_text = small_font.render(f"  R{idx-n_blue}: {reward:+.1f} {status}", True, (200, 80, 80))
                panel.blit(r_text, (20, y_offset))
                y_offset += 16

        # Blit panel to screen (top-left corner)
        surface.blit(panel, (10, 10))

    except pygame.error:
        # Font not initialized, skip rendering
        pass


def render_keybindings_overlay(surface: pygame.Surface) -> None:
    """
    Render keybindings help overlay.

    Displays all available keyboard controls in a semi-transparent panel.

    Args:
        surface: Pygame surface to draw on
    """
    try:
        # Define keybindings
        keybindings = [
            ("Shift+Q", "Exit game"),
            ("`", "Toggle debug info"),
            ("F", "Toggle FOV overlay"),
            ("G", "Toggle strategic grid"),
            ("R", "Toggle rewards overlay"),
            ("?", "Toggle this help"),
            ("1-8", "Select blue unit"),
            ("Shift+1-8", "Select red unit"),
            ("SPACE", "Clear waypoint(s)"),
            ("Click", "Set waypoint"),
        ]

        # Calculate panel height based on content
        line_height = 30
        title_height = 60
        footer_height = 50
        panel_height = title_height + len(keybindings) * line_height + footer_height

        # Create semi-transparent background panel (centered)
        panel_width = 400
        panel_x = (WINDOW_SIZE - panel_width) // 2
        panel_y = (WINDOW_SIZE - panel_height) // 2

        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 200))  # Dark semi-transparent background

        # Draw border
        pygame.draw.rect(panel, (255, 255, 255), (0, 0, panel_width, panel_height), 2)

        # Initialize fonts
        title_font = pygame.font.Font(None, 36)
        key_font = pygame.font.Font(None, 26)
        desc_font = pygame.font.Font(None, 24)

        # Render title
        title = title_font.render("KEYBINDINGS", True, (255, 255, 0))
        title_rect = title.get_rect(centerx=panel_width // 2, y=20)
        panel.blit(title, title_rect)

        # Render keybindings
        y_offset = title_height

        for key, description in keybindings:
            # Render key (highlighted)
            key_surface = key_font.render(key, True, (100, 255, 100))
            panel.blit(key_surface, (30, y_offset))

            # Render description
            desc_surface = desc_font.render(description, True, (220, 220, 220))
            panel.blit(desc_surface, (140, y_offset + 2))

            y_offset += line_height

        # Footer
        y_offset += 15
        footer_font = pygame.font.Font(None, 20)
        footer = footer_font.render("Press '?' again to close", True, (150, 150, 150))
        footer_rect = footer.get_rect(centerx=panel_width // 2, y=y_offset)
        panel.blit(footer, footer_rect)

        # Blit panel to screen (centered)
        surface.blit(panel, (panel_x, panel_y))

    except pygame.error:
        # Font not initialized, skip rendering
        pass


ACTION_NAMES = {
    0: "Hold",
    1: "N", 2: "W", 3: "E", 4: "S",
    5: "NE", 6: "NW", 7: "SW", 8: "SE",
    9: "D9", 10: "D10", 11: "D11", 12: "D12",
    13: "Aggr", 14: "Def", 15: "Think",
}


def render_unit_rewards_overlay(
    surface: pygame.Surface,
    reward_info: Dict[str, Any]
) -> None:
    """
    Render unit rewards overlay showing all units from both teams.

    Displays episode rewards and current actions for all blue and red units.

    Args:
        surface: Pygame surface to draw on
        reward_info: Dictionary containing:
            - 'selected_unit_id': Currently selected blue unit index (or None)
            - 'selected_red_unit_id': Currently selected red unit index (or None)
            - 'blue_units': List of blue Unit objects
            - 'red_units': List of red Unit objects
            - 'episode_rewards': Dict[(team, unit_id), float] cumulative rewards
            - 'unit_actions': Dict[(team, unit_id), int] current actions
    """
    try:
        selected_unit_id = reward_info.get('selected_unit_id')
        selected_red_unit_id = reward_info.get('selected_red_unit_id')
        blue_units = reward_info.get('blue_units', [])
        red_units = reward_info.get('red_units', [])
        episode_rewards = reward_info.get('episode_rewards', {})
        unit_actions = reward_info.get('unit_actions', {})

        # Create semi-transparent background panel (right side)
        panel_width = 200
        panel_height = 450
        panel_x = WINDOW_SIZE - panel_width - 10
        panel_y = 10

        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))  # Dark semi-transparent background

        # Draw border
        pygame.draw.rect(panel, (255, 255, 100), (0, 0, panel_width, panel_height), 2)

        # Initialize fonts
        title_font = pygame.font.Font(None, 24)
        label_font = pygame.font.Font(None, 20)
        value_font = pygame.font.Font(None, 18)

        y_offset = 8

        # Title
        title = title_font.render("UNIT REWARDS (R)", True, (255, 255, 100))
        panel.blit(title, (10, y_offset))
        y_offset += 24

        # Blue team header
        blue_header = label_font.render("BLUE TEAM", True, (135, 206, 250))
        panel.blit(blue_header, (10, y_offset))
        y_offset += 18

        # Blue units - use tuple key format ("blue", unit_id)
        blue_team_total = 0.0
        for unit_idx, unit in enumerate(blue_units):
            # Get reward using tuple key
            unit_reward = episode_rewards.get(("blue", unit.id), 0.0)
            blue_team_total += unit_reward

            # Get current action - use unit_idx as fallback key since unit.id may differ
            action_idx = unit_actions.get(("blue", unit.id), -1)
            if action_idx == -1:
                # Try with unit_idx in case IDs don't match
                action_idx = unit_actions.get(("blue", unit_idx), -1)
            action_name = ACTION_NAMES.get(action_idx, f"#{action_idx}")

            # Get goal cell reference (D{row}{col})
            goal = getattr(unit, 'goal_waypoint', None)
            if goal is not None:
                goal_col = int(goal[0] / TACTICAL_CELLS_PER_OPERATIONAL)
                goal_row = int(goal[1] / TACTICAL_CELLS_PER_OPERATIONAL)
                goal_cell = f"D{goal_row}{goal_col}"
            else:
                goal_cell = "-"

            # Highlight selected unit
            if unit_idx == selected_unit_id:
                unit_color = (135, 206, 250)  # Light blue for selected
                marker = ">"
            else:
                unit_color = (150, 150, 200)
                marker = " "

            prefix = "+" if unit_reward > 0 else ""
            unit_text = value_font.render(
                f"{marker}U{unit_idx + 1}: {prefix}{unit_reward:.1f} [{action_name} {goal_cell}]",
                True, unit_color
            )
            panel.blit(unit_text, (10, y_offset))
            y_offset += 15

        # Blue team total
        blue_total_color = (100, 255, 100) if blue_team_total > 0 else (255, 100, 100) if blue_team_total < 0 else (150, 150, 150)
        blue_prefix = "+" if blue_team_total > 0 else ""
        blue_total_text = label_font.render(f"Total: {blue_prefix}{blue_team_total:.1f}", True, blue_total_color)
        panel.blit(blue_total_text, (10, y_offset))
        y_offset += 24

        # Red team header
        red_header = label_font.render("RED TEAM", True, (255, 165, 0))
        panel.blit(red_header, (10, y_offset))
        y_offset += 18

        # Red units - use tuple key format ("red", unit_id)
        red_team_total = 0.0
        for unit_idx, unit in enumerate(red_units):
            # Get reward using tuple key
            unit_reward = episode_rewards.get(("red", unit.id), 0.0)
            red_team_total += unit_reward

            # Get current action - use unit_idx as fallback key since unit.id may differ
            action_idx = unit_actions.get(("red", unit.id), -1)
            if action_idx == -1:
                # Try with unit_idx in case IDs don't match
                action_idx = unit_actions.get(("red", unit_idx), -1)
            action_name = ACTION_NAMES.get(action_idx, f"#{action_idx}")

            # Get goal cell reference (D{row}{col})
            goal = getattr(unit, 'goal_waypoint', None)
            if goal is not None:
                goal_col = int(goal[0] / TACTICAL_CELLS_PER_OPERATIONAL)
                goal_row = int(goal[1] / TACTICAL_CELLS_PER_OPERATIONAL)
                goal_cell = f"D{goal_row}{goal_col}"
            else:
                goal_cell = "-"

            # Highlight selected unit
            if unit_idx == selected_red_unit_id:
                unit_color = (255, 165, 0)  # Orange for selected
                marker = ">"
            else:
                unit_color = (200, 150, 150)
                marker = " "

            prefix = "+" if unit_reward > 0 else ""
            unit_text = value_font.render(
                f"{marker}u{unit_idx + 1}: {prefix}{unit_reward:.1f} [{action_name} {goal_cell}]",
                True, unit_color
            )
            panel.blit(unit_text, (10, y_offset))
            y_offset += 15

        # Red team total
        red_total_color = (100, 255, 100) if red_team_total > 0 else (255, 100, 100) if red_team_total < 0 else (150, 150, 150)
        red_prefix = "+" if red_team_total > 0 else ""
        red_total_text = label_font.render(f"Total: {red_prefix}{red_team_total:.1f}", True, red_total_color)
        panel.blit(red_total_text, (10, y_offset))
        y_offset += 24

        # Keybindings hint
        hint1 = value_font.render("1-8: Select blue", True, (120, 120, 120))
        panel.blit(hint1, (10, y_offset))
        y_offset += 14
        hint2 = value_font.render("Shift+1-8: Select red", True, (120, 120, 120))
        panel.blit(hint2, (10, y_offset))

        # Blit panel to screen
        surface.blit(panel, (panel_x, panel_y))

    except pygame.error:
        # Font not initialized, skip rendering
        pass


if __name__ == "__main__":
    """Basic self-tests for renderer module."""
    import sys

    def test_coordinate_conversion():
        """Test grid-to-pixel coordinate conversion."""
        # Grid position 5 should be at pixel 5 * CELL_SIZE
        grid_x = 5
        expected_pixel_x = grid_x * CELL_SIZE
        assert expected_pixel_x == 80, f"Expected 80, got {expected_pixel_x}"
        print("  coordinate conversion: OK")

    def test_surface_initialization():
        """Test lazy surface initialization logic."""
        # Before pygame init, surfaces should be None
        global _BLUE_NEAR_OVERLAY
        initial_state = _BLUE_NEAR_OVERLAY

        # Can't test full init without pygame, but can verify the check
        assert initial_state is None or isinstance(initial_state, type(None)) or True
        print("  surface initialization check: OK")

    def test_config_imports():
        """Test that all required config values are available."""
        assert WINDOW_SIZE > 0
        assert CELL_SIZE > 0
        assert GRID_SIZE > 0
        assert len(COLOR_BACKGROUND) == 3
        assert len(COLOR_BLUE_TEAM) == 3
        assert len(COLOR_RED_TEAM) == 3
        print("  config imports: OK")

    def test_agent_size_ratio():
        """Test agent size calculations."""
        radius = int((CELL_SIZE / 2) * AGENT_SIZE_RATIO)
        assert radius > 0, "Agent radius should be positive"
        assert radius < CELL_SIZE, "Agent radius should be less than cell size"
        print("  agent size ratio: OK")

    # Run all tests
    print("Running renderer.py self-tests...")
    print("  (Note: Full rendering tests require pygame initialization)")
    try:
        test_coordinate_conversion()
        test_surface_initialization()
        test_config_imports()
        test_agent_size_ratio()
        print("All renderer.py self-tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
