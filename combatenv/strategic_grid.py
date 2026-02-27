"""
Strategic grid system for high-level observation and action spaces.

The 64x64 tactical grid is divided into a 4x4 strategic grid where each
strategic cell covers 16x16 tactical cells.

This module provides:
- Terrain summarization (major terrain type per strategic cell)
- Occupancy detection (dominant team per strategic cell)
- Observation building (32-value array: 16 terrain + 16 occupancy)
- Action conversion (strategic cell index to tactical position)
"""

from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np

from .config import (
    GRID_SIZE,
    STRATEGIC_GRID_SIZE,
    TACTICAL_CELLS_PER_STRATEGIC,
)
from .terrain import TerrainType, TerrainGrid


# Terrain type to normalized value mapping
TERRAIN_TO_VALUE = {
    TerrainType.EMPTY: 0.0,
    TerrainType.OBSTACLE: 0.25,
    TerrainType.FIRE: 0.5,
    TerrainType.FOREST: 0.75,
    TerrainType.WATER: 1.0,
}


def get_strategic_cell_bounds(cell_index: int) -> Tuple[int, int, int, int]:
    """
    Get tactical grid bounds for a strategic cell.

    Args:
        cell_index: Strategic cell index (0-15 for 4x4 grid)

    Returns:
        (x_start, y_start, x_end, y_end) in tactical grid coordinates
    """
    strategic_x = cell_index % STRATEGIC_GRID_SIZE
    strategic_y = cell_index // STRATEGIC_GRID_SIZE

    x_start = strategic_x * TACTICAL_CELLS_PER_STRATEGIC
    y_start = strategic_y * TACTICAL_CELLS_PER_STRATEGIC
    x_end = x_start + TACTICAL_CELLS_PER_STRATEGIC
    y_end = y_start + TACTICAL_CELLS_PER_STRATEGIC

    return (x_start, y_start, x_end, y_end)


def get_major_terrain(
    terrain_grid: TerrainGrid,
    strategic_x: int,
    strategic_y: int
) -> float:
    """
    Get the most common terrain type in a 16x16 strategic cell.

    Args:
        terrain_grid: The terrain grid
        strategic_x: Strategic grid X coordinate (0-3)
        strategic_y: Strategic grid Y coordinate (0-3)

    Returns:
        Normalized terrain value (0.0-1.0)
    """
    x_start = strategic_x * TACTICAL_CELLS_PER_STRATEGIC
    y_start = strategic_y * TACTICAL_CELLS_PER_STRATEGIC

    terrain_counts: Counter = Counter()

    for x in range(x_start, x_start + TACTICAL_CELLS_PER_STRATEGIC):
        for y in range(y_start, y_start + TACTICAL_CELLS_PER_STRATEGIC):
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                terrain_type = terrain_grid.get(x, y)
                terrain_counts[terrain_type] += 1

    if not terrain_counts:
        return 0.0  # Default to EMPTY

    # Get most common terrain type
    major_terrain = terrain_counts.most_common(1)[0][0]
    return TERRAIN_TO_VALUE.get(major_terrain, 0.0)


def get_major_occupancy(
    blue_agents: List,
    red_agents: List,
    strategic_x: int,
    strategic_y: int
) -> float:
    """
    Get the dominant team in a 16x16 strategic cell.

    Args:
        blue_agents: List of blue team agents
        red_agents: List of red team agents
        strategic_x: Strategic grid X coordinate (0-3)
        strategic_y: Strategic grid Y coordinate (0-3)

    Returns:
        Occupancy value: 0.0 (blue dominant), 0.5 (empty/neutral), 1.0 (red dominant)
    """
    x_start = strategic_x * TACTICAL_CELLS_PER_STRATEGIC
    y_start = strategic_y * TACTICAL_CELLS_PER_STRATEGIC
    x_end = x_start + TACTICAL_CELLS_PER_STRATEGIC
    y_end = y_start + TACTICAL_CELLS_PER_STRATEGIC

    blue_count = 0
    red_count = 0

    for agent in blue_agents:
        if agent.is_alive:
            x, y = agent.position
            if x_start <= x < x_end and y_start <= y < y_end:
                blue_count += 1

    for agent in red_agents:
        if agent.is_alive:
            x, y = agent.position
            if x_start <= x < x_end and y_start <= y < y_end:
                red_count += 1

    # Determine dominance
    if blue_count == 0 and red_count == 0:
        return 0.5  # Empty
    elif blue_count > red_count:
        return 0.0  # Blue dominant
    elif red_count > blue_count:
        return 1.0  # Red dominant
    else:
        return 0.5  # Neutral (equal)


def build_strategic_observation(
    terrain_grid: TerrainGrid,
    blue_agents: List,
    red_agents: List
) -> np.ndarray:
    """
    Build 32-value strategic observation array.

    The observation consists of:
    - First 16 values: Major terrain type per strategic cell
    - Last 16 values: Major occupancy per strategic cell

    Args:
        terrain_grid: The terrain grid
        blue_agents: List of blue team agents
        red_agents: List of red team agents

    Returns:
        numpy array of shape (32,) with values in [0, 1]
    """
    obs = np.zeros(32, dtype=np.float32)

    # Fill terrain values (indices 0-15)
    for cell_idx in range(STRATEGIC_GRID_SIZE * STRATEGIC_GRID_SIZE):
        strategic_x = cell_idx % STRATEGIC_GRID_SIZE
        strategic_y = cell_idx // STRATEGIC_GRID_SIZE
        obs[cell_idx] = get_major_terrain(terrain_grid, strategic_x, strategic_y)

    # Fill occupancy values (indices 16-31)
    for cell_idx in range(STRATEGIC_GRID_SIZE * STRATEGIC_GRID_SIZE):
        strategic_x = cell_idx % STRATEGIC_GRID_SIZE
        strategic_y = cell_idx // STRATEGIC_GRID_SIZE
        obs[16 + cell_idx] = get_major_occupancy(
            blue_agents, red_agents, strategic_x, strategic_y
        )

    return obs


def strategic_cell_to_position(cell_index: int) -> Tuple[float, float]:
    """
    Convert strategic cell index to tactical grid center position.

    Args:
        cell_index: Strategic cell index (0-15 for 4x4 grid)

    Returns:
        (x, y) center position in tactical grid coordinates
    """
    strategic_x = cell_index % STRATEGIC_GRID_SIZE
    strategic_y = cell_index // STRATEGIC_GRID_SIZE

    # Center of the strategic cell
    x = (strategic_x + 0.5) * TACTICAL_CELLS_PER_STRATEGIC
    y = (strategic_y + 0.5) * TACTICAL_CELLS_PER_STRATEGIC

    return (x, y)


def position_to_strategic_cell(x: float, y: float) -> int:
    """
    Convert tactical grid position to strategic cell index.

    Args:
        x: X position in tactical grid
        y: Y position in tactical grid

    Returns:
        Strategic cell index (0-15)
    """
    strategic_x = int(x / TACTICAL_CELLS_PER_STRATEGIC)
    strategic_y = int(y / TACTICAL_CELLS_PER_STRATEGIC)

    # Clamp to valid range
    strategic_x = max(0, min(STRATEGIC_GRID_SIZE - 1, strategic_x))
    strategic_y = max(0, min(STRATEGIC_GRID_SIZE - 1, strategic_y))

    return strategic_y * STRATEGIC_GRID_SIZE + strategic_x


def get_terrain_board(terrain_grid: TerrainGrid) -> np.ndarray:
    """
    Get 4x4 terrain board as 2D array.

    Args:
        terrain_grid: The terrain grid

    Returns:
        numpy array of shape (4, 4) with terrain values
    """
    board = np.zeros((STRATEGIC_GRID_SIZE, STRATEGIC_GRID_SIZE), dtype=np.float32)

    for y in range(STRATEGIC_GRID_SIZE):
        for x in range(STRATEGIC_GRID_SIZE):
            board[y, x] = get_major_terrain(terrain_grid, x, y)

    return board


def get_occupancy_board(blue_agents: List, red_agents: List) -> np.ndarray:
    """
    Get 4x4 occupancy board as 2D array.

    Args:
        blue_agents: List of blue team agents
        red_agents: List of red team agents

    Returns:
        numpy array of shape (4, 4) with occupancy values
    """
    board = np.zeros((STRATEGIC_GRID_SIZE, STRATEGIC_GRID_SIZE), dtype=np.float32)

    for y in range(STRATEGIC_GRID_SIZE):
        for x in range(STRATEGIC_GRID_SIZE):
            board[y, x] = get_major_occupancy(blue_agents, red_agents, x, y)

    return board
