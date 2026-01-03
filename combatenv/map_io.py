"""
Map I/O utilities for saving and loading terrain maps.

This module provides functions to save and load TerrainGrid maps in JSON format.
Maps can be created with the map editor and loaded into the simulation.

JSON Format:
    {
        "version": 1,
        "width": 64,
        "height": 64,
        "terrain": [[0, 1, 0, ...], ...]
    }

Usage:
    >>> from combatenv import TerrainGrid, save_map, load_map
    >>> grid = TerrainGrid(64, 64)
    >>> save_map(grid, "my_map.json")
    >>> loaded_grid = load_map("my_map.json")
"""

import json
from typing import Any, Dict

from .terrain import TerrainGrid, TerrainType
from .config import GRID_SIZE


MAP_FORMAT_VERSION = 1


def save_map(terrain_grid: TerrainGrid, filepath: str) -> None:
    """
    Save a terrain grid to a JSON file.

    Args:
        terrain_grid: The TerrainGrid to save
        filepath: Path to the output JSON file

    Example:
        >>> grid = TerrainGrid(64, 64)
        >>> grid.set(10, 10, TerrainType.BUILDING)
        >>> save_map(grid, "my_map.json")
    """
    data: Dict[str, Any] = {
        "version": MAP_FORMAT_VERSION,
        "width": terrain_grid.width,
        "height": terrain_grid.height,
        "terrain": terrain_grid.grid.tolist()
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_map(filepath: str) -> TerrainGrid:
    """
    Load a terrain grid from a JSON file.

    Args:
        filepath: Path to the JSON map file

    Returns:
        TerrainGrid populated with the loaded terrain data

    Raises:
        ValueError: If the map format version is unsupported
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON

    Example:
        >>> grid = load_map("my_map.json")
        >>> print(grid.get(10, 10))
        TerrainType.BUILDING
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    version = data.get("version", 0)
    if version != MAP_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported map format version {version}. "
            f"Expected version {MAP_FORMAT_VERSION}."
        )

    width = data["width"]
    height = data["height"]
    terrain_data = data["terrain"]

    grid = TerrainGrid(width, height)

    for x in range(width):
        for y in range(height):
            terrain_value = terrain_data[x][y]
            grid.set(x, y, TerrainType(terrain_value))

    return grid


if __name__ == "__main__":
    """Basic self-tests for map_io module."""
    import tempfile
    import os

    def test_save_load_roundtrip():
        """Test saving and loading a map."""
        grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        grid.set(10, 10, TerrainType.BUILDING)
        grid.set(20, 20, TerrainType.FIRE)
        grid.set(30, 30, TerrainType.SWAMP)
        grid.set(40, 40, TerrainType.WATER)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            save_map(grid, filepath)
            loaded = load_map(filepath)

            assert loaded.get(10, 10) == TerrainType.BUILDING
            assert loaded.get(20, 20) == TerrainType.FIRE
            assert loaded.get(30, 30) == TerrainType.SWAMP
            assert loaded.get(40, 40) == TerrainType.WATER
            assert loaded.get(0, 0) == TerrainType.EMPTY
            print("  save/load roundtrip: OK")
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
                assert "Unsupported map format version" in str(e)
                print("  invalid version handling: OK")
        finally:
            os.unlink(filepath)

    print("Running map_io.py self-tests...")
    test_save_load_roundtrip()
    test_load_invalid_version()
    print("All map_io.py self-tests passed!")
