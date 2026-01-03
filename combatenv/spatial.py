"""
Spatial partitioning system for efficient collision detection.

This module provides a spatial hash grid for O(n) collision queries instead of O(n^2).
With 200 agents, this reduces collision checks from ~40,000 to ~3,200 per frame,
providing approximately 12x speedup for neighbor queries.

Algorithm:
    The spatial grid divides the world into fixed-size cells (buckets). Each agent
    is inserted into exactly one bucket based on its position. When querying for
    nearby agents, only the 9 buckets in a 3x3 neighborhood are checked.

Configuration:
    - Default cell size: 2.0 grid units
    - Recommended: 2-3x the agent collision radius (AGENT_COLLISION_RADIUS = 0.8)
    - Larger cells = fewer hash lookups, but more agents per cell
    - Smaller cells = more precise queries, but more hash operations

Usage:
    >>> from spatial import SpatialGrid
    >>> grid = SpatialGrid(cell_size=2.0)
    >>> grid.build(all_agents)  # Called once per frame
    >>> nearby = grid.get_nearby_agents(agent)  # O(1) query
    >>> # nearby contains ~8-16 agents instead of all 200

Performance Notes:
    - build() is O(n) - must be called each frame as agents move
    - get_nearby_agents() is O(1) - constant time hash lookup
    - Memory: O(n) for grid dictionary entries

Example:
    >>> from agent import spawn_all_teams
    >>> from spatial import SpatialGrid
    >>> blue, red = spawn_all_teams()
    >>> spatial = SpatialGrid(cell_size=2.0)
    >>> spatial.build(blue + red)
    >>> stats = spatial.get_statistics()
    >>> print(f"Grid has {stats['num_cells']} occupied cells")
"""

from typing import List, Set, Tuple, Dict
from collections import defaultdict
from .config import GRID_SIZE


class SpatialGrid:
    """
    Spatial hash grid for efficient neighbor queries.

    Divides the world into cells and maintains a mapping of which agents
    are in which cells. This allows querying only nearby agents instead
    of checking all agents.

    Attributes:
        cell_size: Size of each spatial cell in grid units
        grid: Dictionary mapping (cell_x, cell_y) to list of agents in that cell
    """

    def __init__(self, cell_size: float = 2.0):
        """
        Initialize the spatial grid.

        Args:
            cell_size: Size of each spatial cell in grid units.
                      Larger cells = fewer cells to check, but more agents per cell.
                      Recommended: 2-3x the agent collision radius.
        """
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List] = defaultdict(list)

    def clear(self) -> None:
        """Clear all agents from the grid."""
        self.grid.clear()

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Get the cell coordinates for a world position.

        Args:
            x: X position in world coordinates
            y: Y position in world coordinates

        Returns:
            (cell_x, cell_y) tuple representing the cell
        """
        cell_x = int(x // self.cell_size)
        cell_y = int(y // self.cell_size)
        return (cell_x, cell_y)

    def insert(self, agent) -> None:
        """
        Insert an agent into the spatial grid.

        Args:
            agent: Agent to insert (must have position attribute)
        """
        cell = self._get_cell(agent.position[0], agent.position[1])
        self.grid[cell].append(agent)

    def build(self, agents: List) -> None:
        """
        Build the spatial grid from a list of agents.

        Clears the existing grid and inserts all agents.

        Args:
            agents: List of agents to insert into the grid
        """
        self.clear()
        for agent in agents:
            self.insert(agent)

    def get_nearby_agents(self, agent, radius: float = None) -> List:
        """
        Get agents near the given agent.

        Checks the agent's cell and all 8 neighboring cells (3x3 grid).
        This is much faster than checking all agents in the world.

        Args:
            agent: Agent to query around (must have position attribute)
            radius: Optional radius for additional filtering (not yet implemented)

        Returns:
            List of agents in nearby cells (including the query agent itself)

        Example:
            >>> spatial_grid = SpatialGrid(cell_size=2.0)
            >>> spatial_grid.build(all_agents)
            >>> nearby = spatial_grid.get_nearby_agents(my_agent)
            >>> # nearby contains only ~8-12 agents instead of all 400
        """
        cell_x, cell_y = self._get_cell(agent.position[0], agent.position[1])

        nearby = []
        # Check 3x3 grid of cells around the agent
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.grid:
                    nearby.extend(self.grid[neighbor_cell])

        return nearby

    def get_agents_in_cell(self, x: float, y: float) -> List:
        """
        Get all agents in the cell containing the given position.

        Args:
            x: X position in world coordinates
            y: Y position in world coordinates

        Returns:
            List of agents in that cell (may be empty)
        """
        cell = self._get_cell(x, y)
        return self.grid.get(cell, [])

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the spatial grid.

        Returns:
            Dictionary containing:
                - num_cells: Number of occupied cells
                - total_agents: Total number of agents in grid
                - avg_agents_per_cell: Average agents per occupied cell
                - max_agents_per_cell: Maximum agents in any single cell
        """
        if not self.grid:
            return {
                'num_cells': 0,
                'total_agents': 0,
                'avg_agents_per_cell': 0.0,
                'max_agents_per_cell': 0
            }

        cell_counts = [len(agents) for agents in self.grid.values()]
        total = sum(cell_counts)

        return {
            'num_cells': len(self.grid),
            'total_agents': total,
            'avg_agents_per_cell': total / len(self.grid) if self.grid else 0.0,
            'max_agents_per_cell': max(cell_counts) if cell_counts else 0
        }


def test_spatial_grid():
    """
    Simple test to verify spatial grid functionality.

    This is a basic sanity check, not a comprehensive unit test.
    """
    from agent import Agent

    # Create some test agents
    agents = [
        Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
        Agent(position=(5.5, 5.5), orientation=0.0, team="blue"),  # Near first agent
        Agent(position=(10.0, 10.0), orientation=0.0, team="red"),  # Far away
        Agent(position=(50.0, 50.0), orientation=0.0, team="red"),  # Very far away
    ]

    # Build spatial grid
    spatial = SpatialGrid(cell_size=2.0)
    spatial.build(agents)

    # Test: nearby query should return agents in same/adjacent cells
    nearby = spatial.get_nearby_agents(agents[0])
    assert agents[0] in nearby, "Agent should find itself"
    assert agents[1] in nearby, "Should find nearby agent"
    assert len(nearby) <= len(agents), "Should not return more agents than exist"

    # Test: statistics
    stats = spatial.get_statistics()
    assert stats['total_agents'] == len(agents)
    assert stats['num_cells'] > 0

    print("Spatial grid tests passed!")


if __name__ == "__main__":
    test_spatial_grid()
