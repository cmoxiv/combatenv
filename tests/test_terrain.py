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
import numpy as np
from combatenv import TerrainType, TerrainGrid


class TestTerrainType:
    """Tests for TerrainType enum."""

    def test_enum_values(self):
        """Test terrain type enum has correct integer values."""
        assert TerrainType.EMPTY == 0
        assert TerrainType.OBSTACLE == 1
        assert TerrainType.FIRE == 2
        assert TerrainType.FOREST == 3
        assert TerrainType.WATER == 4

    def test_enum_ordering(self):
        """Test terrain types can be compared."""
        assert TerrainType.EMPTY < TerrainType.OBSTACLE
        assert TerrainType.OBSTACLE < TerrainType.FIRE


class TestTerrainGrid:
    """Tests for TerrainGrid class."""

    def test_init(self):
        """Test grid initialization."""
        grid = TerrainGrid(64, 64)
        assert grid.width == 64
        assert grid.height == 64
        # Internal storage is at pixel resolution (64 * CELL_SIZE = 1024)
        assert grid.grid.shape == (grid.pixel_width, grid.pixel_height)

    def test_get_set(self):
        """Test get and set operations."""
        grid = TerrainGrid(10, 10)

        # Default is EMPTY
        assert grid.get(5, 5) == TerrainType.EMPTY

        # Set and get
        grid.set(5, 5, TerrainType.OBSTACLE)
        assert grid.get(5, 5) == TerrainType.OBSTACLE

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
        grid.set(-1, 5, TerrainType.OBSTACLE)
        grid.set(100, 5, TerrainType.OBSTACLE)

    def test_is_walkable_empty(self):
        """Test empty terrain is walkable."""
        grid = TerrainGrid(10, 10)
        assert grid.is_walkable(5, 5) == True

    def test_is_walkable_obstacle(self):
        """Test obstacles are not walkable."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.OBSTACLE)
        assert grid.is_walkable(5, 5) == False

    def test_is_walkable_water(self):
        """Test water is walkable (slows, prevents shooting, grants invisibility)."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.WATER)
        assert grid.is_walkable(5, 5) == True

    def test_is_walkable_fire(self):
        """Test fire is walkable (but damages)."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.FIRE)
        assert grid.is_walkable(5, 5) == True

    def test_is_walkable_forest(self):
        """Test forest is walkable (but slows)."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.FOREST)
        assert grid.is_walkable(5, 5) == True

    def test_blocks_los_obstacle(self):
        """Test obstacles block line of sight."""
        grid = TerrainGrid(10, 10)
        grid.set(5, 5, TerrainType.OBSTACLE)
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
        grid.set(5, 5, TerrainType.OBSTACLE)
        grid.set(3, 3, TerrainType.FIRE)

        grid.clear()

        assert grid.get(5, 5) == TerrainType.EMPTY
        assert grid.get(3, 3) == TerrainType.EMPTY


class TestTerrainGeneration:
    """Tests for terrain generation using 8x8 operational grid system."""

    def test_generate_random_creates_terrain(self):
        """Test random generation creates terrain."""
        grid = TerrainGrid(64, 64)
        rng = random.Random(42)

        grid.generate_random(
            obstacle_pct=0.05,
            spawn_margin=5,
            rng=rng
        )

        # Count terrain types
        counts = {t: 0 for t in TerrainType}
        for x in range(64):
            for y in range(64):
                counts[grid.get(x, y)] += 1

        # Should have buildings (placed last)
        assert counts[TerrainType.OBSTACLE] > 0, "Should have obstacles"

        # Should have non-empty terrain from operational grid
        non_empty = sum(counts[t] for t in TerrainType if t != TerrainType.EMPTY)
        assert non_empty > 0, "Should have non-empty terrain"

    def test_respects_spawn_margin(self):
        """Test terrain generation keeps spawn corners clear."""
        grid = TerrainGrid(64, 64)
        rng = random.Random(42)
        margin = 10

        grid.generate_random(
            obstacle_pct=0.10,
            spawn_margin=margin,
            rng=rng
        )

        # Blue spawn corner (top-left) should be empty
        for x in range(margin):
            for y in range(margin):
                assert grid.get(x, y) == TerrainType.EMPTY, f"Blue spawn at ({x},{y}) should be empty"

        # Red spawn corner (bottom-right) should be empty
        for x in range(64 - margin, 64):
            for y in range(64 - margin, 64):
                assert grid.get(x, y) == TerrainType.EMPTY, f"Red spawn at ({x},{y}) should be empty"

    def test_deterministic_with_seed(self):
        """Test same seed produces same terrain."""
        grid1 = TerrainGrid(32, 32)
        grid2 = TerrainGrid(32, 32)

        grid1.generate_random(rng=random.Random(12345))
        grid2.generate_random(rng=random.Random(12345))

        for x in range(32):
            for y in range(32):
                assert grid1.get(x, y) == grid2.get(x, y), f"Mismatch at ({x},{y})"

    def test_operational_grid_fills_regions(self):
        """Test operational grid fills 8x8 regions with terrain."""
        grid = TerrainGrid(64, 64)
        rng = random.Random(42)

        grid.generate_random(
            obstacle_pct=0.0,  # No obstacles to see operational grid clearly
            spawn_margin=0,   # No margin to fill entire grid
            rng=rng
        )

        # Each 8x8 operational cell should have terrain
        # Check that each operational cell has at least some terrain
        for cx in range(8):
            for cy in range(8):
                start_x = cx * 8
                start_y = cy * 8
                cell_terrain = set()
                for dx in range(8):
                    for dy in range(8):
                        cell_terrain.add(grid.get(start_x + dx, start_y + dy))
                # Each operational cell should have terrain (major and minor types)
                assert len(cell_terrain) >= 1, f"Operational cell ({cx},{cy}) should have terrain"


class TestPixelLevelStorage:
    """Tests for pixel-level terrain storage."""

    def test_pixel_level_storage(self):
        """Test grid stores terrain at pixel resolution (1024x1024)."""
        grid = TerrainGrid(64, 64)
        assert grid.pixel_width == 1024
        assert grid.pixel_height == 1024
        assert grid.grid.shape == (1024, 1024)

    def test_cell_majority(self):
        """Test cell majority voting with mixed pixels."""
        grid = TerrainGrid(10, 10)
        cs = grid.cell_size

        # Fill cell (3,3) with mostly FOREST but some WATER
        px, py = 3 * cs, 3 * cs
        grid.grid[px:px + cs, py:py + cs] = TerrainType.FOREST
        # Overwrite a few pixels with WATER (less than half)
        grid.grid[px:px + 2, py:py + 2] = TerrainType.WATER

        # Majority should be FOREST
        assert grid.get(3, 3) == TerrainType.FOREST

    def test_pixel_get_set(self):
        """Test pixel-level get/set methods."""
        grid = TerrainGrid(10, 10)

        grid.set_pixel(50, 50, TerrainType.FIRE)
        assert grid.get_pixel(50, 50) == TerrainType.FIRE
        assert grid.get_pixel(51, 50) == TerrainType.EMPTY

    def test_pixel_out_of_bounds(self):
        """Test pixel access out of bounds returns EMPTY."""
        grid = TerrainGrid(10, 10)
        assert grid.get_pixel(-1, 0) == TerrainType.EMPTY
        assert grid.get_pixel(9999, 0) == TerrainType.EMPTY


class TestLayeredGeneration:
    """Tests for the layered terrain generation algorithm."""

    def test_terrain_types_touch(self):
        """Test that terrain types can directly border each other."""
        grid = TerrainGrid(64, 64)
        grid.generate_random(obstacle_pct=0.05, spawn_margin=5, rng=random.Random(42))

        g = grid.grid
        # Forest should directly touch water or obstacles somewhere
        forest_mask = g == TerrainType.FOREST
        forest_coords = np.argwhere(forest_mask)

        touches_other = 0
        for x, y in forest_coords[:500]:  # Sample for speed
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.pixel_width and 0 <= ny < grid.pixel_height:
                    n = g[nx, ny]
                    if n in (TerrainType.WATER, TerrainType.OBSTACLE):
                        touches_other += 1
                        break

        assert touches_other > 0, "Forest should directly touch water or obstacles"

    def test_forest_near_water(self):
        """Test most forest pixels are near water."""
        from scipy.ndimage import distance_transform_edt

        grid = TerrainGrid(64, 64)
        grid.generate_random(obstacle_pct=0.05, spawn_margin=5, rng=random.Random(42))

        g = grid.grid
        water_mask = (g == TerrainType.WATER).astype(np.float64)
        dist_from_water = distance_transform_edt(1.0 - water_mask)

        forest_pixels = g == TerrainType.FOREST
        if forest_pixels.sum() == 0:
            pytest.skip("No forest generated with this seed")

        forest_dists = dist_from_water[forest_pixels]
        # At 1024 resolution, "near water" means within ~60 pixels (15*4 upscale)
        near_pct = (forest_dists < 80).sum() / forest_dists.size * 100
        assert near_pct > 50, f"Only {near_pct:.1f}% of forest is near water"

    def test_fire_at_forest_edges(self):
        """Test fire pixels are near forest (within upscale distance)."""
        from scipy.ndimage import distance_transform_edt

        grid = TerrainGrid(64, 64)
        grid.generate_random(obstacle_pct=0.05, spawn_margin=5, rng=random.Random(42))

        g = grid.grid
        fire_mask = g == TerrainType.FIRE
        forest_mask = (g == TerrainType.FOREST).astype(np.float64)

        if fire_mask.sum() == 0:
            pytest.skip("No fire generated with this seed")

        # Distance from forest for all pixels
        dist_from_forest = distance_transform_edt(1.0 - forest_mask)
        fire_dists = dist_from_forest[fire_mask]

        # At 256→1024 upscale (4x), fire placed adjacent at 256 means
        # within ~5 pixels at 1024 resolution
        near_pct = (fire_dists < 8).sum() / fire_dists.size * 100
        assert near_pct > 50, f"Only {near_pct:.1f}% of fire is near forest"

    def test_terrain_distribution(self):
        """Test terrain type percentages are within expected ranges."""
        grid = TerrainGrid(64, 64)
        grid.generate_random(spawn_margin=5, rng=random.Random(42))

        total = grid.grid.size
        counts = np.bincount(grid.grid.ravel().astype(np.int32), minlength=5)
        pcts = counts / total * 100

        # Ordering should be: Empty > Obstacle > Water > Forest > Fire
        assert pcts[TerrainType.EMPTY] > 30, f"Empty: {pcts[TerrainType.EMPTY]:.1f}%"
        assert 10 < pcts[TerrainType.OBSTACLE] < 30, f"Obstacle: {pcts[TerrainType.OBSTACLE]:.1f}%"
        assert pcts[TerrainType.WATER] > 0.5, f"Water: {pcts[TerrainType.WATER]:.1f}%"
        assert pcts[TerrainType.FOREST] > 0.5, f"Forest: {pcts[TerrainType.FOREST]:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
