"""
Terrain system for the Grid-World Multi-Agent Tactical Simulation.

This module provides terrain types that affect agent movement, visibility,
and combat. Each grid cell can contain one terrain type.

Terrain Types:
    - EMPTY: Normal traversable terrain with no effects
    - BUILDING: Blocks movement and line of sight
    - FIRE: Causes damage that bypasses armor
    - SWAMP: Traps agents for a random number of steps

Example:
    >>> from terrain import TerrainType, TerrainGrid
    >>> grid = TerrainGrid(64, 64)
    >>> grid.generate_random(building_pct=0.05, fire_pct=0.02, swamp_pct=0.03)
    >>> grid.is_walkable(10, 10)
    True
"""

from enum import IntEnum
import numpy as np
from typing import Optional
import random


class TerrainType(IntEnum):
    """Enumeration of terrain types."""
    EMPTY = 0
    BUILDING = 1
    FIRE = 2
    SWAMP = 3
    WATER = 4


class TerrainGrid:
    """
    Grid-based terrain storage and query system.

    Stores terrain types for each cell in the simulation grid and provides
    efficient queries for movement and visibility checks.

    Attributes:
        width: Grid width in cells
        height: Grid height in cells
        grid: 2D numpy array of terrain types
    """

    def __init__(self, width: int, height: int):
        """
        Initialize an empty terrain grid.

        Args:
            width: Grid width in cells
            height: Grid height in cells
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=np.int8)

    def get(self, x: int, y: int, strict: bool = False) -> TerrainType:
        """
        Get the terrain type at a specific cell.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            strict: If True, raise ValueError for out-of-bounds coordinates.
                    If False (default), return EMPTY for out-of-bounds.

        Returns:
            TerrainType at the specified cell, or EMPTY if out of bounds (non-strict)

        Raises:
            ValueError: If strict=True and coordinates are out of bounds
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return TerrainType(self.grid[x, y])
        if strict:
            raise ValueError(
                f"Coordinates ({x}, {y}) out of bounds for grid size "
                f"({self.width}, {self.height})"
            )
        return TerrainType.EMPTY

    def set(self, x: int, y: int, terrain: TerrainType, strict: bool = False) -> None:
        """
        Set the terrain type at a specific cell.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            terrain: TerrainType to set
            strict: If True, raise ValueError for out-of-bounds coordinates.
                    If False (default), silently ignore out-of-bounds sets.

        Raises:
            ValueError: If strict=True and coordinates are out of bounds
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x, y] = terrain
        elif strict:
            raise ValueError(
                f"Coordinates ({x}, {y}) out of bounds for grid size "
                f"({self.width}, {self.height})"
            )

    def is_walkable(self, x: int, y: int) -> bool:
        """
        Check if a cell can be entered by agents.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate

        Returns:
            True if the cell is walkable (not a building or water)
        """
        terrain = self.get(x, y)
        return terrain != TerrainType.BUILDING and terrain != TerrainType.WATER

    def blocks_los(self, x: int, y: int) -> bool:
        """
        Check if a cell blocks line of sight.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate

        Returns:
            True if the cell blocks visibility (is a building)
        """
        return self.get(x, y) == TerrainType.BUILDING

    def clear(self) -> None:
        """Reset all cells to empty terrain."""
        self.grid.fill(TerrainType.EMPTY)

    def _is_valid_placement(self, x: int, y: int, spawn_margin: int) -> bool:
        """Check if position is valid for terrain placement (not in spawn areas)."""
        if x < spawn_margin or x >= self.width - spawn_margin:
            return False
        if y < spawn_margin or y >= self.height - spawn_margin:
            return False
        return True

    def _generate_puddle(
        self,
        center_x: int,
        center_y: int,
        target_size: int,
        terrain_type: TerrainType,
        spawn_margin: int,
        rng: random.Random
    ) -> int:
        """
        Generate a puddle-shaped patch using random walk.

        Returns number of cells placed.
        """
        placed = 0
        to_visit = [(center_x, center_y)]
        visited = set()

        while to_visit and placed < target_size:
            x, y = to_visit.pop(rng.randint(0, len(to_visit) - 1))

            if (x, y) in visited:
                continue
            visited.add((x, y))

            if not self._is_valid_placement(x, y, spawn_margin):
                continue
            if self.get(x, y) != TerrainType.EMPTY:
                continue

            self.set(x, y, terrain_type)
            placed += 1

            # Add neighbors with random probability for organic shape
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and rng.random() < 0.7:
                    to_visit.append((nx, ny))

        return placed

    def _generate_fire_line(
        self,
        start_x: int,
        start_y: int,
        length: int,
        spawn_margin: int,
        rng: random.Random
    ) -> int:
        """
        Generate a fire line/arc using random walk with momentum.

        Returns number of cells placed.
        """
        placed = 0
        x, y = float(start_x), float(start_y)
        angle = rng.uniform(0, 2 * 3.14159)

        for _ in range(length):
            ix, iy = int(x), int(y)

            if self._is_valid_placement(ix, iy, spawn_margin):
                if self.get(ix, iy) == TerrainType.EMPTY:
                    self.set(ix, iy, TerrainType.FIRE)
                    placed += 1

            # Move with slight curve
            angle += rng.uniform(-0.5, 0.5)
            x += np.cos(angle)
            y += np.sin(angle)

            # Bounds check
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                break

        return placed

    def _generate_building_cluster(
        self,
        center_x: int,
        center_y: int,
        target_size: int,
        spawn_margin: int,
        rng: random.Random
    ) -> int:
        """
        Generate a cluster of buildings forming corridor-like structures.
        Uses L-shaped and rectangular patterns.

        Returns number of cells placed.
        """
        placed = 0

        # Generate several connected rectangular/L-shaped segments
        num_segments = rng.randint(2, 4)
        x, y = center_x, center_y

        for _ in range(num_segments):
            if placed >= target_size:
                break

            # Choose segment type
            segment_type = rng.choice(['horizontal', 'vertical', 'l_shape'])
            length = rng.randint(3, 6)

            if segment_type == 'horizontal':
                for i in range(length):
                    px = x + i
                    if self._is_valid_placement(px, y, spawn_margin):
                        if self.get(px, y) == TerrainType.EMPTY:
                            self.set(px, y, TerrainType.BUILDING)
                            placed += 1
                x += length - 1

            elif segment_type == 'vertical':
                for i in range(length):
                    py = y + i
                    if self._is_valid_placement(x, py, spawn_margin):
                        if self.get(x, py) == TerrainType.EMPTY:
                            self.set(x, py, TerrainType.BUILDING)
                            placed += 1
                y += length - 1

            else:  # L-shape
                half = length // 2
                # Horizontal part
                for i in range(half):
                    px = x + i
                    if self._is_valid_placement(px, y, spawn_margin):
                        if self.get(px, y) == TerrainType.EMPTY:
                            self.set(px, y, TerrainType.BUILDING)
                            placed += 1
                # Vertical part
                direction = rng.choice([-1, 1])
                for i in range(1, half + 1):
                    py = y + i * direction
                    if self._is_valid_placement(x + half - 1, py, spawn_margin):
                        if self.get(x + half - 1, py) == TerrainType.EMPTY:
                            self.set(x + half - 1, py, TerrainType.BUILDING)
                            placed += 1
                x += half - 1
                y += (half) * direction

            # Small random offset for next segment
            x += rng.randint(-1, 1)
            y += rng.randint(-1, 1)

        return placed

    def generate_random(
        self,
        building_pct: float = 0.05,
        fire_pct: float = 0.02,
        swamp_pct: float = 0.03,
        water_pct: float = 0.03,
        spawn_margin: int = 5,
        rng: Optional[random.Random] = None
    ) -> None:
        """
        Generate random terrain with natural-looking formations.

        - Buildings: Clustered into corridor-like structures
        - Fire: Lines and arcs
        - Swamp: Puddle-shaped patches
        - Water: Puddle-shaped patches

        Args:
            building_pct: Percentage of cells to be buildings (0.0-1.0)
            fire_pct: Percentage of cells to be fire (0.0-1.0)
            swamp_pct: Percentage of cells to be swamp (0.0-1.0)
            water_pct: Percentage of cells to be water (0.0-1.0)
            spawn_margin: Cells from edges to keep clear for spawning
            rng: Random number generator (uses global random if None)
        """
        self.clear()

        if rng is None:
            rng = random.Random()

        total_cells = self.width * self.height
        valid_width = self.width - 2 * spawn_margin
        valid_height = self.height - 2 * spawn_margin

        # Calculate target counts
        building_target = int(total_cells * building_pct)
        fire_target = int(total_cells * fire_pct)
        swamp_target = int(total_cells * swamp_pct)
        water_target = int(total_cells * water_pct)

        # Generate building clusters (corridors)
        building_placed = 0
        num_clusters = max(1, building_target // 15)
        for _ in range(num_clusters):
            if building_placed >= building_target:
                break
            cx = rng.randint(spawn_margin, self.width - spawn_margin - 1)
            cy = rng.randint(spawn_margin, self.height - spawn_margin - 1)
            cluster_size = rng.randint(10, 20)
            building_placed += self._generate_building_cluster(
                cx, cy, min(cluster_size, building_target - building_placed),
                spawn_margin, rng
            )

        # Generate fire lines/arcs
        fire_placed = 0
        num_fires = max(1, fire_target // 8)
        for _ in range(num_fires):
            if fire_placed >= fire_target:
                break
            fx = rng.randint(spawn_margin, self.width - spawn_margin - 1)
            fy = rng.randint(spawn_margin, self.height - spawn_margin - 1)
            line_length = rng.randint(5, 15)
            fire_placed += self._generate_fire_line(
                fx, fy, min(line_length, fire_target - fire_placed),
                spawn_margin, rng
            )

        # Generate swamp puddles
        swamp_placed = 0
        num_swamps = max(1, swamp_target // 12)
        for _ in range(num_swamps):
            if swamp_placed >= swamp_target:
                break
            sx = rng.randint(spawn_margin, self.width - spawn_margin - 1)
            sy = rng.randint(spawn_margin, self.height - spawn_margin - 1)
            puddle_size = rng.randint(8, 20)
            swamp_placed += self._generate_puddle(
                sx, sy, min(puddle_size, swamp_target - swamp_placed),
                TerrainType.SWAMP, spawn_margin, rng
            )

        # Generate water puddles
        water_placed = 0
        num_waters = max(1, water_target // 12)
        for _ in range(num_waters):
            if water_placed >= water_target:
                break
            wx = rng.randint(spawn_margin, self.width - spawn_margin - 1)
            wy = rng.randint(spawn_margin, self.height - spawn_margin - 1)
            puddle_size = rng.randint(8, 20)
            water_placed += self._generate_puddle(
                wx, wy, min(puddle_size, water_target - water_placed),
                TerrainType.WATER, spawn_margin, rng
            )


if __name__ == "__main__":
    """Basic self-tests for terrain module."""
    import sys

    def test_terrain_type_enum():
        """Test TerrainType enum values."""
        assert TerrainType.EMPTY == 0
        assert TerrainType.BUILDING == 1
        assert TerrainType.FIRE == 2
        assert TerrainType.SWAMP == 3
        assert TerrainType.WATER == 4
        print("  TerrainType enum: OK")

    def test_terrain_grid_creation():
        """Test TerrainGrid creation."""
        grid = TerrainGrid(64, 64)
        assert grid.width == 64
        assert grid.height == 64
        assert grid.grid.shape == (64, 64)
        print("  TerrainGrid creation: OK")

    def test_terrain_grid_get_set():
        """Test get and set operations."""
        grid = TerrainGrid(10, 10)

        # Default is EMPTY
        assert grid.get(5, 5) == TerrainType.EMPTY

        # Set and get
        grid.set(5, 5, TerrainType.BUILDING)
        assert grid.get(5, 5) == TerrainType.BUILDING

        # Out of bounds returns EMPTY
        assert grid.get(-1, 5) == TerrainType.EMPTY
        assert grid.get(100, 5) == TerrainType.EMPTY
        print("  get/set: OK")

    def test_is_walkable():
        """Test walkability checks."""
        grid = TerrainGrid(10, 10)

        # EMPTY is walkable
        assert grid.is_walkable(5, 5) == True

        # BUILDING is not walkable
        grid.set(5, 5, TerrainType.BUILDING)
        assert grid.is_walkable(5, 5) == False

        # WATER is not walkable
        grid.set(6, 6, TerrainType.WATER)
        assert grid.is_walkable(6, 6) == False

        # FIRE is walkable
        grid.set(7, 7, TerrainType.FIRE)
        assert grid.is_walkable(7, 7) == True

        # SWAMP is walkable
        grid.set(8, 8, TerrainType.SWAMP)
        assert grid.is_walkable(8, 8) == True
        print("  is_walkable: OK")

    def test_blocks_los():
        """Test line of sight blocking."""
        grid = TerrainGrid(10, 10)

        # EMPTY doesn't block
        assert grid.blocks_los(5, 5) == False

        # BUILDING blocks
        grid.set(5, 5, TerrainType.BUILDING)
        assert grid.blocks_los(5, 5) == True

        # WATER doesn't block
        grid.set(6, 6, TerrainType.WATER)
        assert grid.blocks_los(6, 6) == False
        print("  blocks_los: OK")

    def test_generate_random():
        """Test random terrain generation."""
        grid = TerrainGrid(64, 64)
        rng = random.Random(42)  # Fixed seed for reproducibility

        grid.generate_random(
            building_pct=0.05,
            fire_pct=0.02,
            swamp_pct=0.03,
            water_pct=0.03,
            spawn_margin=5,
            rng=rng
        )

        # Count terrain types
        counts = {t: 0 for t in TerrainType}
        for x in range(64):
            for y in range(64):
                counts[grid.get(x, y)] += 1

        # Should have some of each type
        assert counts[TerrainType.BUILDING] > 0, "Should have buildings"
        assert counts[TerrainType.FIRE] > 0, "Should have fire"
        assert counts[TerrainType.SWAMP] > 0, "Should have swamp"
        assert counts[TerrainType.WATER] > 0, "Should have water"
        print(f"  generate_random: OK (B:{counts[TerrainType.BUILDING]}, F:{counts[TerrainType.FIRE]}, S:{counts[TerrainType.SWAMP]}, W:{counts[TerrainType.WATER]})")

    def test_clear():
        """Test clearing the grid."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.BUILDING)
        grid.clear()
        assert grid.get(5, 5) == TerrainType.EMPTY
        print("  clear: OK")

    # Run all tests
    print("Running terrain.py self-tests...")
    try:
        test_terrain_type_enum()
        test_terrain_grid_creation()
        test_terrain_grid_get_set()
        test_is_walkable()
        test_blocks_los()
        test_generate_random()
        test_clear()
        print("All terrain.py self-tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
