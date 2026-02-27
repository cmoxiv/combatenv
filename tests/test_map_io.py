"""Tests for map I/O functionality."""

import json
import os
import tempfile
import numpy as np
import pytest

from combatenv import TerrainGrid, TerrainType, save_map, load_map
from combatenv.map_io import MAP_FORMAT_VERSION


class TestSaveMap:
    """Tests for save_map function (v2 npz format)."""

    def test_save_creates_file(self):
        """Test that save_map creates a file."""
        grid = TerrainGrid(64, 64)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_map(grid, filepath)
            assert os.path.exists(filepath)
        finally:
            os.unlink(filepath)

    def test_save_npz_structure(self):
        """Test that saved npz has correct structure."""
        grid = TerrainGrid(64, 64)
        grid.set(10, 10, TerrainType.OBSTACLE)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_map(grid, filepath)

            data = np.load(filepath)
            assert 'version' in data
            assert 'width' in data
            assert 'height' in data
            assert 'grid' in data
            assert int(data['version']) == MAP_FORMAT_VERSION
            assert int(data['width']) == 64
            assert int(data['height']) == 64
            # Grid is pixel-level (1024x1024)
            assert data['grid'].shape == (grid.pixel_width, grid.pixel_height)
        finally:
            os.unlink(filepath)

    def test_save_preserves_terrain(self):
        """Test that terrain values are preserved in save."""
        grid = TerrainGrid(64, 64)
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
        finally:
            os.unlink(filepath)


class TestLoadMap:
    """Tests for load_map function."""

    def test_load_returns_terrain_grid(self):
        """Test that load_map returns a TerrainGrid."""
        grid = TerrainGrid(64, 64)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_map(grid, filepath)
            loaded = load_map(filepath)
            assert isinstance(loaded, TerrainGrid)
        finally:
            os.unlink(filepath)

    def test_load_preserves_dimensions(self):
        """Test that loaded grid has correct dimensions."""
        grid = TerrainGrid(64, 64)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_map(grid, filepath)
            loaded = load_map(filepath)
            assert loaded.width == 64
            assert loaded.height == 64
        finally:
            os.unlink(filepath)

    def test_load_preserves_terrain(self):
        """Test that terrain values are preserved after load."""
        grid = TerrainGrid(64, 64)
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
        finally:
            os.unlink(filepath)

    def test_load_v1_json(self):
        """Test that v1 JSON maps are loaded and upscaled to pixel level."""
        cell_grid = [[0] * 64 for _ in range(64)]
        cell_grid[10][10] = int(TerrainType.OBSTACLE)
        cell_grid[20][20] = int(TerrainType.FOREST)

        v1_data = {
            "version": 1,
            "width": 64,
            "height": 64,
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
            # Verify pixel-level storage
            assert loaded.grid.shape == (loaded.pixel_width, loaded.pixel_height)
        finally:
            os.unlink(filepath)

    def test_load_invalid_json_version_raises(self):
        """Test that loading unsupported JSON version raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "version": 999,
                "width": 64,
                "height": 64,
                "terrain": [[0] * 64 for _ in range(64)]
            }, f)
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported"):
                load_map(filepath)
        finally:
            os.unlink(filepath)

    def test_load_missing_file_raises(self):
        """Test that loading non-existent file raises error."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_map("/nonexistent/path/map.npz")

    def test_load_invalid_json_raises(self):
        """Test that loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json{{{")
            filepath = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_map(filepath)
        finally:
            os.unlink(filepath)


class TestRoundTrip:
    """Tests for save/load roundtrip."""

    def test_full_roundtrip(self):
        """Test complete save/load cycle preserves all data."""
        grid = TerrainGrid(64, 64)

        # Set various terrain types
        for i in range(5):
            for j in range(5):
                grid.set(i * 10, j * 10, TerrainType((i + j) % 5))

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_map(grid, filepath)
            loaded = load_map(filepath)

            # Verify all cells match
            for x in range(64):
                for y in range(64):
                    assert loaded.get(x, y) == grid.get(x, y), \
                        f"Mismatch at ({x}, {y})"
        finally:
            os.unlink(filepath)
