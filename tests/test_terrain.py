"""
Unit tests for the terrain system.

Tests cover:
- TerrainType enum values
- TerrainGrid creation and manipulation
- Walkability and LOS blocking
- Terrain generation algorithms

Run with: pytest tests/test_terrain.py -v
"""

import pytest
import random
from combatenv import TerrainType, TerrainGrid


class TestTerrainType:
    """Tests for TerrainType enum."""

    def test_enum_values(self):
        """Test terrain type enum has correct integer values."""
        assert TerrainType.EMPTY == 0
        assert TerrainType.BUILDING == 1
        assert TerrainType.FIRE == 2
        assert TerrainType.SWAMP == 3
        assert TerrainType.WATER == 4

    def test_enum_ordering(self):
        """Test terrain types can be compared."""
        assert TerrainType.EMPTY < TerrainType.BUILDING
        assert TerrainType.BUILDING < TerrainType.FIRE


class TestTerrainGrid:
    """Tests for TerrainGrid class."""

    def test_init(self):
        """Test grid initialization."""
        grid = TerrainGrid(64, 64)
        assert grid.width == 64
        assert grid.height == 64
        assert grid.grid.shape == (64, 64)

    def test_get_set(self):
        """Test get and set operations."""
        grid = TerrainGrid(10, 10)

        # Default is EMPTY
        assert grid.get(5, 5) == TerrainType.EMPTY

        # Set and get
        grid.set(5, 5, TerrainType.BUILDING)
        assert grid.get(5, 5) == TerrainType.BUILDING

        grid.set(3, 3, TerrainType.FIRE)
        assert grid.get(3, 3) == TerrainType.FIRE

    def test_get_out_of_bounds(self):
        """Test get returns EMPTY for out of bounds."""
        grid = TerrainGrid(10, 10)

        assert grid.get(-1, 5) == TerrainType.EMPTY
        assert grid.get(100, 5) == TerrainType.EMPTY
        assert grid.get(5, -1) == TerrainType.EMPTY
        assert grid.get(5, 100) == TerrainType.EMPTY

    def test_set_out_of_bounds_silent(self):
        """Test set ignores out of bounds silently."""
        grid = TerrainGrid(10, 10)
        # Should not raise
        grid.set(-1, 5, TerrainType.BUILDING)
        grid.set(100, 5, TerrainType.BUILDING)

    def test_is_walkable_empty(self):
        """Test empty terrain is walkable."""
        grid = TerrainGrid(10, 10)
        assert grid.is_walkable(5, 5) == True

    def test_is_walkable_building(self):
        """Test buildings are not walkable."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.BUILDING)
        assert grid.is_walkable(5, 5) == False

    def test_is_walkable_water(self):
        """Test water is not walkable."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.WATER)
        assert grid.is_walkable(5, 5) == False

    def test_is_walkable_fire(self):
        """Test fire is walkable (but damages)."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.FIRE)
        assert grid.is_walkable(5, 5) == True

    def test_is_walkable_swamp(self):
        """Test swamp is walkable (but slows)."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.SWAMP)
        assert grid.is_walkable(5, 5) == True

    def test_blocks_los_building(self):
        """Test buildings block line of sight."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.BUILDING)
        assert grid.blocks_los(5, 5) == True

    def test_blocks_los_empty(self):
        """Test empty doesn't block line of sight."""
        grid = TerrainGrid(10, 10)
        assert grid.blocks_los(5, 5) == False

    def test_blocks_los_water(self):
        """Test water doesn't block line of sight."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.WATER)
        assert grid.blocks_los(5, 5) == False

    def test_blocks_los_fire(self):
        """Test fire doesn't block line of sight."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.FIRE)
        assert grid.blocks_los(5, 5) == False

    def test_clear(self):
        """Test clearing the grid."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.BUILDING)
        grid.set(3, 3, TerrainType.FIRE)

        grid.clear()

        assert grid.get(5, 5) == TerrainType.EMPTY
        assert grid.get(3, 3) == TerrainType.EMPTY


class TestTerrainGeneration:
    """Tests for terrain generation."""

    def test_generate_random_creates_terrain(self):
        """Test random generation creates terrain of each type."""
        grid = TerrainGrid(64, 64)
        rng = random.Random(42)

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

        assert counts[TerrainType.BUILDING] > 0, "Should have buildings"
        assert counts[TerrainType.FIRE] > 0, "Should have fire"
        assert counts[TerrainType.SWAMP] > 0, "Should have swamp"
        assert counts[TerrainType.WATER] > 0, "Should have water"

    def test_respects_spawn_margin(self):
        """Test terrain generation respects spawn margins."""
        grid = TerrainGrid(64, 64)
        rng = random.Random(42)
        margin = 10

        grid.generate_random(
            building_pct=0.10,
            fire_pct=0.05,
            swamp_pct=0.05,
            water_pct=0.05,
            spawn_margin=margin,
            rng=rng
        )

        # Check margin areas are empty
        for x in range(margin):
            for y in range(64):
                assert grid.get(x, y) == TerrainType.EMPTY, f"Left margin at ({x},{y}) should be empty"

        for x in range(64 - margin, 64):
            for y in range(64):
                assert grid.get(x, y) == TerrainType.EMPTY, f"Right margin at ({x},{y}) should be empty"

    def test_deterministic_with_seed(self):
        """Test same seed produces same terrain."""
        grid1 = TerrainGrid(32, 32)
        grid2 = TerrainGrid(32, 32)

        grid1.generate_random(rng=random.Random(12345))
        grid2.generate_random(rng=random.Random(12345))

        for x in range(32):
            for y in range(32):
                assert grid1.get(x, y) == grid2.get(x, y), f"Mismatch at ({x},{y})"

    def test_puddle_generates_connected(self):
        """Test puddle generation creates connected regions."""
        grid = TerrainGrid(64, 64)
        rng = random.Random(42)

        # Generate just water
        grid.generate_random(
            building_pct=0.0,
            fire_pct=0.0,
            swamp_pct=0.0,
            water_pct=0.05,
            spawn_margin=5,
            rng=rng
        )

        # Find a water cell
        water_cells = []
        for x in range(64):
            for y in range(64):
                if grid.get(x, y) == TerrainType.WATER:
                    water_cells.append((x, y))

        if water_cells:
            # Check at least one water cell has a water neighbor
            has_neighbor = False
            for x, y in water_cells:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if grid.get(x + dx, y + dy) == TerrainType.WATER:
                        has_neighbor = True
                        break
                if has_neighbor:
                    break
            assert has_neighbor, "Water should form connected puddles"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
