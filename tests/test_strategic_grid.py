"""
Unit tests for combatenv/strategic_grid.py - Strategic grid observation system.
"""

import pytest
import numpy as np
from combatenv.strategic_grid import (
    get_strategic_cell_bounds,
    get_major_terrain,
    get_major_occupancy,
    build_strategic_observation,
    strategic_cell_to_position,
    position_to_strategic_cell,
    get_terrain_board,
    get_occupancy_board,
    TERRAIN_TO_VALUE,
)
from combatenv.terrain import TerrainType, TerrainGrid
from combatenv.agent import Agent
from combatenv.config import GRID_SIZE, STRATEGIC_GRID_SIZE, TACTICAL_CELLS_PER_STRATEGIC


class TestStrategicCellBounds:
    """Test strategic cell bounds calculation."""

    def test_cell_0_bounds(self):
        """Test bounds for cell 0 (top-left)."""
        x_start, y_start, x_end, y_end = get_strategic_cell_bounds(0)
        assert x_start == 0
        assert y_start == 0
        assert x_end == 16
        assert y_end == 16

    def test_cell_3_bounds(self):
        """Test bounds for cell 3 (top-right)."""
        x_start, y_start, x_end, y_end = get_strategic_cell_bounds(3)
        assert x_start == 48
        assert y_start == 0
        assert x_end == 64
        assert y_end == 16

    def test_cell_12_bounds(self):
        """Test bounds for cell 12 (bottom-left)."""
        x_start, y_start, x_end, y_end = get_strategic_cell_bounds(12)
        assert x_start == 0
        assert y_start == 48
        assert x_end == 16
        assert y_end == 64

    def test_cell_15_bounds(self):
        """Test bounds for cell 15 (bottom-right)."""
        x_start, y_start, x_end, y_end = get_strategic_cell_bounds(15)
        assert x_start == 48
        assert y_start == 48
        assert x_end == 64
        assert y_end == 64


class TestMajorTerrain:
    """Test major terrain type detection."""

    def test_empty_grid_returns_empty(self):
        """Test that empty grid returns EMPTY terrain value."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        value = get_major_terrain(terrain_grid, 0, 0)
        assert value == TERRAIN_TO_VALUE[TerrainType.EMPTY]

    def test_mostly_buildings(self):
        """Test cell with mostly buildings."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        # Fill cell 0 with buildings (more than half of 16x16=256 cells)
        for x in range(14):
            for y in range(14):
                terrain_grid.set(x, y, TerrainType.OBSTACLE)

        value = get_major_terrain(terrain_grid, 0, 0)
        assert value == TERRAIN_TO_VALUE[TerrainType.OBSTACLE]

    def test_mostly_fire(self):
        """Test cell with mostly fire."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        # Fill cell 1 (x=16-31) with fire
        for x in range(16, 32):
            for y in range(16):
                terrain_grid.set(x, y, TerrainType.FIRE)

        value = get_major_terrain(terrain_grid, 1, 0)
        assert value == TERRAIN_TO_VALUE[TerrainType.FIRE]

    def test_terrain_values_normalized(self):
        """Test that all terrain types map to values between 0 and 1."""
        for terrain_type in TerrainType:
            value = TERRAIN_TO_VALUE.get(terrain_type, 0.0)
            assert 0.0 <= value <= 1.0


class TestMajorOccupancy:
    """Test major occupancy detection."""

    def test_empty_cell_returns_neutral(self):
        """Test that empty cell returns neutral (0.5)."""
        blue_agents = []
        red_agents = []
        value = get_major_occupancy(blue_agents, red_agents, 0, 0)
        assert value == 0.5

    def test_blue_dominant(self):
        """Test cell with more blue agents."""
        blue_agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(6.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(7.0, 5.0), orientation=0.0, team="blue"),
        ]
        red_agents = [
            Agent(position=(8.0, 5.0), orientation=0.0, team="red"),
        ]
        value = get_major_occupancy(blue_agents, red_agents, 0, 0)
        assert value == 0.0  # Blue dominant

    def test_red_dominant(self):
        """Test cell with more red agents."""
        blue_agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
        ]
        red_agents = [
            Agent(position=(6.0, 5.0), orientation=0.0, team="red"),
            Agent(position=(7.0, 5.0), orientation=0.0, team="red"),
            Agent(position=(8.0, 5.0), orientation=0.0, team="red"),
        ]
        value = get_major_occupancy(blue_agents, red_agents, 0, 0)
        assert value == 1.0  # Red dominant

    def test_equal_occupancy(self):
        """Test cell with equal agents returns neutral."""
        blue_agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
        ]
        red_agents = [
            Agent(position=(6.0, 5.0), orientation=0.0, team="red"),
        ]
        value = get_major_occupancy(blue_agents, red_agents, 0, 0)
        assert value == 0.5  # Neutral

    def test_dead_agents_ignored(self):
        """Test that dead agents are not counted."""
        blue_agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(6.0, 5.0), orientation=0.0, team="blue"),
        ]
        blue_agents[1].health = 0  # Kill one blue agent
        red_agents = [
            Agent(position=(7.0, 5.0), orientation=0.0, team="red"),
            Agent(position=(8.0, 5.0), orientation=0.0, team="red"),
        ]
        value = get_major_occupancy(blue_agents, red_agents, 0, 0)
        assert value == 1.0  # Red dominant (1 blue vs 2 red)

    def test_agents_in_different_cells(self):
        """Test that only agents in target cell are counted."""
        # Blue in cell 0
        blue_agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
        ]
        # Red in cell 1 (x=16-31)
        red_agents = [
            Agent(position=(20.0, 5.0), orientation=0.0, team="red"),
            Agent(position=(21.0, 5.0), orientation=0.0, team="red"),
        ]

        # Cell 0 should be blue dominant
        value0 = get_major_occupancy(blue_agents, red_agents, 0, 0)
        assert value0 == 0.0  # Blue dominant

        # Cell 1 should be red dominant
        value1 = get_major_occupancy(blue_agents, red_agents, 1, 0)
        assert value1 == 1.0  # Red dominant


class TestStrategicObservation:
    """Test strategic observation building."""

    def test_observation_shape(self):
        """Test that observation has correct shape."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        obs = build_strategic_observation(terrain_grid, [], [])
        assert obs.shape == (32,)

    def test_observation_dtype(self):
        """Test that observation has correct dtype."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        obs = build_strategic_observation(terrain_grid, [], [])
        assert obs.dtype == np.float32

    def test_observation_range(self):
        """Test that observation values are in [0, 1]."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        # Add some terrain and agents
        terrain_grid.set(5, 5, TerrainType.OBSTACLE)
        blue_agents = [Agent(position=(10.0, 10.0), orientation=0.0, team="blue")]
        red_agents = [Agent(position=(50.0, 50.0), orientation=0.0, team="red")]

        obs = build_strategic_observation(terrain_grid, blue_agents, red_agents)

        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_terrain_in_first_16_values(self):
        """Test that terrain values are in first 16 indices."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        # Fill cell 0 with buildings
        for x in range(16):
            for y in range(16):
                terrain_grid.set(x, y, TerrainType.OBSTACLE)

        obs = build_strategic_observation(terrain_grid, [], [])

        # First value should be OBSTACLE
        assert obs[0] == TERRAIN_TO_VALUE[TerrainType.OBSTACLE]

    def test_occupancy_in_last_16_values(self):
        """Test that occupancy values are in last 16 indices."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        # Blue agents in cell 0
        blue_agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(6.0, 5.0), orientation=0.0, team="blue"),
        ]

        obs = build_strategic_observation(terrain_grid, blue_agents, [])

        # Index 16 (first occupancy) should be 0.0 (blue dominant)
        assert obs[16] == 0.0


class TestCellConversion:
    """Test cell index to position conversion."""

    def test_cell_0_position(self):
        """Test center position for cell 0."""
        x, y = strategic_cell_to_position(0)
        assert x == 8.0  # Center of 0-16
        assert y == 8.0

    def test_cell_3_position(self):
        """Test center position for cell 3."""
        x, y = strategic_cell_to_position(3)
        assert x == 56.0  # Center of 48-64
        assert y == 8.0

    def test_cell_15_position(self):
        """Test center position for cell 15."""
        x, y = strategic_cell_to_position(15)
        assert x == 56.0
        assert y == 56.0

    def test_position_to_cell_0(self):
        """Test position in cell 0 maps to cell 0."""
        cell = position_to_strategic_cell(5.0, 5.0)
        assert cell == 0

    def test_position_to_cell_15(self):
        """Test position in cell 15 maps to cell 15."""
        cell = position_to_strategic_cell(60.0, 60.0)
        assert cell == 15

    def test_roundtrip_conversion(self):
        """Test that cell -> position -> cell returns same cell."""
        for cell_idx in range(16):
            x, y = strategic_cell_to_position(cell_idx)
            result_cell = position_to_strategic_cell(x, y)
            assert result_cell == cell_idx


class TestBoards:
    """Test terrain and occupancy board functions."""

    def test_terrain_board_shape(self):
        """Test terrain board has correct shape."""
        terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        board = get_terrain_board(terrain_grid)
        assert board.shape == (STRATEGIC_GRID_SIZE, STRATEGIC_GRID_SIZE)

    def test_occupancy_board_shape(self):
        """Test occupancy board has correct shape."""
        board = get_occupancy_board([], [])
        assert board.shape == (STRATEGIC_GRID_SIZE, STRATEGIC_GRID_SIZE)

    def test_occupancy_board_all_neutral_when_empty(self):
        """Test all cells are neutral when no agents."""
        board = get_occupancy_board([], [])
        assert np.all(board == 0.5)
