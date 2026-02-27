"""
Map I/O utilities for saving and loading terrain maps.

This module provides functions to save and load TerrainGrid maps.

Format v2 (current): Compressed numpy (.npz) with pixel-level terrain data.
Format v1 (legacy): JSON with cell-level terrain data. Loading v1 upscales
    to pixel resolution automatically.

Usage:
    >>> from combatenv import TerrainGrid, save_map, load_map
    >>> grid = TerrainGrid(64, 64)
    >>> save_map(grid, "my_map.npz")
    >>> loaded_grid = load_map("my_map.npz")
"""

import json
import numpy as np

from .terrain import TerrainGrid, TerrainType
from .config import GRID_SIZE


MAP_FORMAT_VERSION = 2


def save_map(terrain_grid: TerrainGrid, filepath: str) -> None:
    """
    Save a terrain grid to a compressed numpy file (.npz).

    Args:
        terrain_grid: The TerrainGrid to save
        filepath: Path to the output file (recommended extension: .npz)

    Example:
        >>> grid = TerrainGrid(64, 64)
        >>> grid.set(10, 10, TerrainType.OBSTACLE)
        >>> save_map(grid, "my_map.npz")
    """
    np.savez_compressed(
        filepath,
        version=MAP_FORMAT_VERSION,
        width=terrain_grid.width,
        height=terrain_grid.height,
        cell_size=terrain_grid.cell_size,
        grid=terrain_grid.grid,
    )


def load_map(filepath: str) -> TerrainGrid:
    """
    Load a terrain grid from a file.

    Supports both v2 (.npz pixel-level) and v1 (.json cell-level) formats.
    V1 maps are automatically upscaled to pixel resolution.

    Args:
        filepath: Path to the map file (.npz or .json)

    Returns:
        TerrainGrid populated with the loaded terrain data

    Raises:
        ValueError: If the map format version is unsupported
        FileNotFoundError: If the file doesn't exist

    Example:
        >>> grid = load_map("my_map.npz")
        >>> print(grid.get(10, 10))
        TerrainType.OBSTACLE
    """
    if filepath.endswith('.json'):
        return _load_v1_json(filepath)

    return _load_v2_npz(filepath)


def _load_v2_npz(filepath: str) -> TerrainGrid:
    """Load a v2 pixel-level map from compressed numpy format."""
    data = np.load(filepath)

    version = int(data['version'])
    if version != 2:
        raise ValueError(
            f"Unsupported npz map format version {version}. "
            f"Expected version 2."
        )

    width = int(data['width'])
    height = int(data['height'])

    grid = TerrainGrid(width, height)
    grid.grid = data['grid'].astype(np.int8)
    grid._surface_dirty = True

    return grid


def _load_v1_json(filepath: str) -> TerrainGrid:
    """Load a v1 cell-level map from JSON and upscale to pixel resolution."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    version = data.get("version", 0)
    if version != 1:
        raise ValueError(
            f"Unsupported JSON map format version {version}. "
            f"Expected version 1."
        )

    width = data["width"]
    height = data["height"]
    terrain_data = data["terrain"]

    grid = TerrainGrid(width, height)

    # Upscale cell-level data to pixel-level by filling 16x16 blocks
    for x in range(width):
        for y in range(height):
            terrain_value = terrain_data[x][y]
            if terrain_value != 0:  # Skip EMPTY for speed
                grid.set(x, y, TerrainType(terrain_value))

    return grid


if __name__ == "__main__":
    """Basic self-tests for map_io module."""
    import tempfile
    import os

    def test_save_load_roundtrip():
        """Test saving and loading a map (v2 npz format)."""
        grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        grid.set(10, 10, TerrainType.OBSTACLE)
        grid.set(20, 20, TerrainType.FIRE)
        grid.set(30, 30, TerrainType.FOREST)
        grid.set(40, 40, TerrainType.WATER)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_map(grid, filepath)
            loaded = load_map(filepath)

            assert loaded.get(10, 10) == TerrainType.OBSTACLE
            assert loaded.get(20, 20) == TerrainType.FIRE
            assert loaded.get(30, 30) == TerrainType.FOREST
            assert loaded.get(40, 40) == TerrainType.WATER
            assert loaded.get(0, 0) == TerrainType.EMPTY
            print("  save/load roundtrip (v2 npz): OK")
        finally:
            os.unlink(filepath)

    def test_load_v1_json():
        """Test loading a v1 JSON map with upscaling."""
        # Create a v1-format JSON file
        cell_grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        cell_grid[10][10] = int(TerrainType.OBSTACLE)
        cell_grid[20][20] = int(TerrainType.FOREST)

        v1_data = {
            "version": 1,
            "width": GRID_SIZE,
            "height": GRID_SIZE,
            "terrain": cell_grid,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(v1_data, f)
            filepath = f.name

        try:
            loaded = load_map(filepath)
            assert loaded.get(10, 10) == TerrainType.OBSTACLE
            assert loaded.get(20, 20) == TerrainType.FOREST
            assert loaded.get(0, 0) == TerrainType.EMPTY
            # Verify pixel-level storage exists
            assert loaded.grid.shape == (loaded.pixel_width, loaded.pixel_height)
            print("  load v1 JSON with upscaling: OK")
        finally:
            os.unlink(filepath)

    def test_load_invalid_version():
        """Test loading a map with invalid version."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"version": 999, "width": 64, "height": 64, "terrain": []}, f)
            filepath = f.name

        try:
            try:
                load_map(filepath)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Unsupported" in str(e)
                print("  invalid version handling: OK")
        finally:
            os.unlink(filepath)

    print("Running map_io.py self-tests...")
    test_save_load_roundtrip()
    test_load_v1_json()
    test_load_invalid_version()
    print("All map_io.py self-tests passed!")
