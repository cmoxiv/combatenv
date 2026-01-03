"""
Unit tests for the spatial partitioning system.

Tests cover:
- SpatialGrid initialization
- Agent insertion
- Nearby agent queries
- Grid rebuilding
- Statistics

Run with: pytest tests/test_spatial.py -v
"""

import pytest
from combatenv import SpatialGrid, Agent


class TestSpatialGridInit:
    """Tests for SpatialGrid initialization."""

    def test_default_cell_size(self):
        """Test default cell size is 2.0."""
        grid = SpatialGrid()
        assert grid.cell_size == 2.0

    def test_custom_cell_size(self):
        """Test custom cell size."""
        grid = SpatialGrid(cell_size=5.0)
        assert grid.cell_size == 5.0

    def test_empty_on_creation(self):
        """Test grid is empty on creation."""
        grid = SpatialGrid()
        assert len(grid.grid) == 0


class TestSpatialGridInsert:
    """Tests for agent insertion."""

    def test_insert_single_agent(self):
        """Test inserting a single agent."""
        grid = SpatialGrid(cell_size=2.0)
        agent = Agent(position=(5.0, 5.0), orientation=0.0, team="blue")

        grid.insert(agent)

        assert len(grid.grid) == 1

    def test_insert_multiple_same_cell(self):
        """Test inserting agents in same cell."""
        grid = SpatialGrid(cell_size=2.0)
        agent1 = Agent(position=(5.0, 5.0), orientation=0.0, team="blue")
        agent2 = Agent(position=(5.5, 5.5), orientation=0.0, team="red")

        grid.insert(agent1)
        grid.insert(agent2)

        # Both in same cell (cell_size=2.0, so (5,5) and (5.5,5.5) are in cell (2,2))
        stats = grid.get_statistics()
        assert stats['total_agents'] == 2

    def test_insert_different_cells(self):
        """Test inserting agents in different cells."""
        grid = SpatialGrid(cell_size=2.0)
        agent1 = Agent(position=(5.0, 5.0), orientation=0.0, team="blue")
        agent2 = Agent(position=(50.0, 50.0), orientation=0.0, team="red")

        grid.insert(agent1)
        grid.insert(agent2)

        stats = grid.get_statistics()
        assert stats['num_cells'] == 2


class TestSpatialGridBuild:
    """Tests for grid building from agent list."""

    def test_build_clears_existing(self):
        """Test build clears existing entries."""
        grid = SpatialGrid(cell_size=2.0)

        agents1 = [Agent(position=(5.0, 5.0), orientation=0.0, team="blue")]
        grid.build(agents1)
        assert grid.get_statistics()['total_agents'] == 1

        agents2 = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(20.0, 20.0), orientation=0.0, team="red")
        ]
        grid.build(agents2)
        assert grid.get_statistics()['total_agents'] == 2

    def test_build_empty_list(self):
        """Test building with empty list."""
        grid = SpatialGrid(cell_size=2.0)
        grid.build([])

        assert len(grid.grid) == 0


class TestSpatialGridNearby:
    """Tests for nearby agent queries."""

    def test_get_nearby_finds_self(self):
        """Test agent finds itself in nearby query."""
        grid = SpatialGrid(cell_size=2.0)
        agent = Agent(position=(5.0, 5.0), orientation=0.0, team="blue")

        grid.build([agent])
        nearby = grid.get_nearby_agents(agent)

        assert agent in nearby

    def test_get_nearby_finds_close_agents(self):
        """Test nearby query finds close agents."""
        grid = SpatialGrid(cell_size=2.0)
        agent1 = Agent(position=(5.0, 5.0), orientation=0.0, team="blue")
        agent2 = Agent(position=(5.5, 5.5), orientation=0.0, team="red")

        grid.build([agent1, agent2])
        nearby = grid.get_nearby_agents(agent1)

        assert agent1 in nearby
        assert agent2 in nearby

    def test_get_nearby_excludes_far_agents(self):
        """Test nearby query excludes far agents."""
        grid = SpatialGrid(cell_size=2.0)
        agent1 = Agent(position=(5.0, 5.0), orientation=0.0, team="blue")
        agent2 = Agent(position=(50.0, 50.0), orientation=0.0, team="red")

        grid.build([agent1, agent2])
        nearby = grid.get_nearby_agents(agent1)

        assert agent1 in nearby
        assert agent2 not in nearby

    def test_get_nearby_checks_adjacent_cells(self):
        """Test nearby query checks 3x3 neighborhood."""
        grid = SpatialGrid(cell_size=2.0)
        # Agent at cell boundary
        agent1 = Agent(position=(4.0, 4.0), orientation=0.0, team="blue")
        # Agent in adjacent cell
        agent2 = Agent(position=(5.5, 5.5), orientation=0.0, team="red")

        grid.build([agent1, agent2])
        nearby = grid.get_nearby_agents(agent1)

        assert agent2 in nearby


class TestSpatialGridClear:
    """Tests for clearing the grid."""

    def test_clear(self):
        """Test clear empties the grid."""
        grid = SpatialGrid(cell_size=2.0)
        agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(10.0, 10.0), orientation=0.0, team="red")
        ]

        grid.build(agents)
        assert len(grid.grid) > 0

        grid.clear()
        assert len(grid.grid) == 0


class TestSpatialGridGetAgentsInCell:
    """Tests for getting agents in a specific cell."""

    def test_get_agents_in_cell(self):
        """Test getting agents in a specific cell."""
        grid = SpatialGrid(cell_size=2.0)
        agent1 = Agent(position=(5.0, 5.0), orientation=0.0, team="blue")
        agent2 = Agent(position=(5.5, 5.5), orientation=0.0, team="red")
        agent3 = Agent(position=(50.0, 50.0), orientation=0.0, team="red")

        grid.build([agent1, agent2, agent3])

        # Both agent1 and agent2 should be in same cell
        cell_agents = grid.get_agents_in_cell(5.0, 5.0)
        assert agent1 in cell_agents
        assert agent2 in cell_agents
        assert agent3 not in cell_agents

    def test_get_agents_in_empty_cell(self):
        """Test getting agents from empty cell."""
        grid = SpatialGrid(cell_size=2.0)
        grid.build([])

        cell_agents = grid.get_agents_in_cell(100.0, 100.0)
        assert cell_agents == []


class TestSpatialGridStatistics:
    """Tests for grid statistics."""

    def test_statistics_empty(self):
        """Test statistics for empty grid."""
        grid = SpatialGrid(cell_size=2.0)
        stats = grid.get_statistics()

        assert stats['num_cells'] == 0
        assert stats['total_agents'] == 0
        assert stats['avg_agents_per_cell'] == 0.0
        assert stats['max_agents_per_cell'] == 0

    def test_statistics_with_agents(self):
        """Test statistics with agents."""
        grid = SpatialGrid(cell_size=2.0)
        agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(5.5, 5.5), orientation=0.0, team="red"),
            Agent(position=(50.0, 50.0), orientation=0.0, team="red")
        ]

        grid.build(agents)
        stats = grid.get_statistics()

        assert stats['total_agents'] == 3
        assert stats['num_cells'] == 2
        assert stats['max_agents_per_cell'] == 2


class TestSpatialGridPerformance:
    """Performance-related tests."""

    def test_handles_many_agents(self):
        """Test grid handles many agents."""
        grid = SpatialGrid(cell_size=2.0)
        agents = [
            Agent(position=(float(i % 50), float(i // 50)), orientation=0.0, team="blue")
            for i in range(500)
        ]

        grid.build(agents)
        stats = grid.get_statistics()

        assert stats['total_agents'] == 500

    def test_nearby_query_is_efficient(self):
        """Test nearby query returns subset of agents."""
        grid = SpatialGrid(cell_size=2.0)
        agents = [
            Agent(position=(float(i % 50), float(i // 50)), orientation=0.0, team="blue")
            for i in range(500)
        ]

        grid.build(agents)

        # Query for agent in corner
        nearby = grid.get_nearby_agents(agents[0])

        # Should return far fewer than total agents
        assert len(nearby) < len(agents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
