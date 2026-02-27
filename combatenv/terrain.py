"""
Terrain system for the Grid-World Multi-Agent Tactical Simulation.

This module provides terrain types that affect agent movement, visibility,
and combat. Terrain is stored at pixel level (1024x1024) for organic shapes,
with cell-level API (64x64) for backward compatibility.

Terrain Types:
    - EMPTY: Normal traversable terrain with no effects
    - OBSTACLE: Blocks movement and line of sight (mountains)
    - FIRE: Causes damage that bypasses armor
    - FOREST: Slows agents by 50%, reduces enemy detection range
    - WATER: Slows agents, prevents shooting, grants invisibility

Example:
    >>> from terrain import TerrainType, TerrainGrid
    >>> grid = TerrainGrid(64, 64)
    >>> grid.generate_random()
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
    OBSTACLE = 1
    FIRE = 2
    FOREST = 3
    WATER = 4


class TerrainGrid:
    """
    Pixel-level terrain storage with cell-level API.

    Internal storage is at pixel resolution (width*CELL_SIZE x height*CELL_SIZE).
    Cell-level methods (get, set, is_walkable, blocks_los) use majority voting
    over the 16x16 pixel block for each cell.

    Attributes:
        width: Grid width in cells (e.g. 64)
        height: Grid height in cells (e.g. 64)
        pixel_width: Grid width in pixels (e.g. 1024)
        pixel_height: Grid height in pixels (e.g. 1024)
        grid: 2D numpy array of terrain types at pixel resolution
    """

    def __init__(self, width: int, height: int):
        """
        Initialize an empty terrain grid.

        Args:
            width: Grid width in cells
            height: Grid height in cells
        """
        from combatenv.config import CELL_SIZE

        self.width = width
        self.height = height
        self.cell_size = CELL_SIZE
        self.pixel_width = width * CELL_SIZE
        self.pixel_height = height * CELL_SIZE
        self.grid = np.zeros((self.pixel_width, self.pixel_height), dtype=np.int8)
        self._surface_dirty = True

    # ── Pixel-level methods ─────────────────────────────────────────────

    def get_pixel(self, px: int, py: int) -> TerrainType:
        """
        Get terrain type at a specific pixel.

        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate

        Returns:
            TerrainType at the pixel, or EMPTY if out of bounds
        """
        if 0 <= px < self.pixel_width and 0 <= py < self.pixel_height:
            return TerrainType(self.grid[px, py])
        return TerrainType.EMPTY

    def set_pixel(self, px: int, py: int, terrain: TerrainType) -> None:
        """
        Set terrain type at a specific pixel.

        Args:
            px: Pixel x coordinate
            py: Pixel y coordinate
            terrain: TerrainType to set
        """
        if 0 <= px < self.pixel_width and 0 <= py < self.pixel_height:
            self.grid[px, py] = terrain
            self._surface_dirty = True

    # ── Cell-level methods (backward compatible) ────────────────────────

    def get(self, x: int, y: int, strict: bool = False) -> TerrainType:
        """
        Get the terrain type at a cell via majority vote over 16x16 pixels.

        Args:
            x: Grid x coordinate (cell)
            y: Grid y coordinate (cell)
            strict: If True, raise ValueError for out-of-bounds coordinates.

        Returns:
            Majority TerrainType in the cell, or EMPTY if out of bounds
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.get_cell_majority(x, y)
        if strict:
            raise ValueError(
                f"Coordinates ({x}, {y}) out of bounds for grid size "
                f"({self.width}, {self.height})"
            )
        return TerrainType.EMPTY

    def set(self, x: int, y: int, terrain: TerrainType, strict: bool = False) -> None:
        """
        Set terrain type for a cell by filling the entire 16x16 pixel block.

        Args:
            x: Grid x coordinate (cell)
            y: Grid y coordinate (cell)
            terrain: TerrainType to set
            strict: If True, raise ValueError for out-of-bounds coordinates.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            cs = self.cell_size
            px, py = x * cs, y * cs
            self.grid[px:px + cs, py:py + cs] = terrain
            self._surface_dirty = True
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
            True if the cell is walkable (not an obstacle)
        """
        terrain = self.get(x, y)
        return terrain != TerrainType.OBSTACLE

    def blocks_los(self, x: int, y: int) -> bool:
        """
        Check if a cell blocks line of sight.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate

        Returns:
            True if the cell blocks visibility (is an obstacle)
        """
        return self.get(x, y) == TerrainType.OBSTACLE

    def get_cell_majority(self, cell_x: int, cell_y: int) -> TerrainType:
        """
        Get the majority terrain type for a single cell (16x16 pixel block).

        Args:
            cell_x: Cell x coordinate
            cell_y: Cell y coordinate

        Returns:
            Most common TerrainType in the cell
        """
        cs = self.cell_size
        px, py = cell_x * cs, cell_y * cs
        block = self.grid[px:px + cs, py:py + cs].ravel()
        counts = np.bincount(block.astype(np.int32), minlength=5)
        return TerrainType(int(np.argmax(counts)))

    def get_block_majority(self, x: int, y: int) -> TerrainType:
        """
        Get the majority terrain type for the terrain block containing (x, y).

        Each terrain block is TACTICAL_CELLS_PER_TERRAIN_BLOCK x
        TACTICAL_CELLS_PER_TERRAIN_BLOCK tactical cells. This method counts
        terrain types within that block and returns the most common one.

        Args:
            x: Tactical grid x coordinate
            y: Tactical grid y coordinate

        Returns:
            Most common TerrainType in the block (ties broken by enum order)
        """
        from combatenv.config import TACTICAL_CELLS_PER_TERRAIN_BLOCK as BLK

        # Find block origin in cells
        bx = (x // BLK) * BLK
        by = (y // BLK) * BLK

        # Convert to pixel coordinates
        cs = self.cell_size
        px = bx * cs
        py = by * cs
        size = BLK * cs

        # Clamp to grid bounds
        end_x = min(px + size, self.pixel_width)
        end_y = min(py + size, self.pixel_height)

        block = self.grid[px:end_x, py:end_y].ravel()
        if block.size == 0:
            return TerrainType.EMPTY
        counts = np.bincount(block.astype(np.int32), minlength=5)
        return TerrainType(int(np.argmax(counts)))

    def clear(self) -> None:
        """Reset all pixels to empty terrain."""
        self.grid.fill(TerrainType.EMPTY)
        self._surface_dirty = True

    # ── Terrain generation ──────────────────────────────────────────────

    @staticmethod
    def _enforce_empty_border(
        grid: np.ndarray,
        terrain_type: int,
        allowed: Optional[set] = None
    ) -> np.ndarray:
        """
        Remove pixels of terrain_type that touch a different non-empty type.

        Ensures empty separation between different terrain types.
        Uses 4-neighbor connectivity (up/down/left/right).

        Args:
            grid: 2D terrain array to modify in-place
            terrain_type: The terrain type to erode
            allowed: Set of terrain types that ARE allowed as neighbors
                     (e.g. fire is allowed to touch forest)

        Returns:
            The modified grid
        """
        if allowed is None:
            allowed = set()

        mask = grid == terrain_type

        # Check 4 neighbors via shifts (wrapping is harmless — cleared by margin)
        bad = np.zeros_like(mask)
        for shift, axis in [(-1, 0), (1, 0), (-1, 1), (1, 1)]:
            neighbor = np.roll(grid, shift, axis=axis)
            is_bad = (neighbor != TerrainType.EMPTY) & (neighbor != terrain_type)
            for a in allowed:
                is_bad &= (neighbor != a)
            bad |= is_bad

        grid[mask & bad] = TerrainType.EMPTY
        return grid

    @staticmethod
    def _sample_noise(
        size: int, seed: int,
        scale: float = 40.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0
    ) -> np.ndarray:
        """
        Sample 2D Perlin noise into a (size, size) float32 array.

        Args:
            size: Grid dimension
            seed: Offset seed for unique maps
            scale: Noise zoom (lower = larger features)
            octaves: Number of noise octaves
            persistence: Amplitude decay per octave
            lacunarity: Frequency growth per octave

        Returns:
            (size, size) float32 array of noise values
        """
        import noise as _noise

        field = np.empty((size, size), dtype=np.float32)
        for x in range(size):
            for y in range(size):
                field[x, y] = _noise.pnoise2(
                    (x + seed) / scale, (y + seed) / scale,
                    octaves=octaves, persistence=persistence,
                    lacunarity=lacunarity
                )
        return field

    def generate_random(
        self,
        obstacle_pct: float = 0.10,
        spawn_margin: int = 8,
        rng: Optional[random.Random] = None
    ) -> None:
        """
        Generate layered terrain with ecological rules.

        Generates at 256x256 resolution and upscales to 1024x1024 for
        natural pixel-level detail. Four layers placed in order:

        1. Obstacles from high elevation noise
        2. Water from low elevation + high moisture
        3. Forest near water (distance transform) and near obstacles
        4. Fire at forest edges (sparse)

        Empty terrain separates all non-empty types, except fire may
        touch forest.

        Args:
            obstacle_pct: Target fraction of pixels as obstacles (0.0-1.0)
            spawn_margin: Cells from edges to keep clear for spawning
            rng: Random number generator (uses global random if None)
        """
        from scipy.ndimage import binary_dilation

        self.clear()

        if rng is None:
            rng = random.Random()

        # Work at 256x256 resolution, upscale 4x to 1024x1024
        gen_size = self.pixel_width // 4  # 256
        upscale = 4

        # Spawn margin in gen_size pixels: cells * cell_size / upscale
        margin_px = max(1, spawn_margin * self.cell_size // upscale)

        # Noise seeds
        seed_e = rng.randint(0, 65535)
        seed_m = rng.randint(0, 65535)
        seed_f = rng.randint(0, 65535)

        # Generate noise fields at 256x256
        elev = self._sample_noise(gen_size, seed_e, scale=40.0)
        moist = self._sample_noise(gen_size, seed_m, scale=40.0)

        grid_256 = np.zeros((gen_size, gen_size), dtype=np.int8)

        # ── Layer 1: Obstacles from high elevation ───────────────────
        obstacle_threshold = np.percentile(elev, 100 - obstacle_pct * 100)
        grid_256[elev > obstacle_threshold] = TerrainType.OBSTACLE

        # ── Layer 2: Water from low elevation + high moisture ────────
        water_elev_pct = np.percentile(elev, 15)
        water_mask = (
            (elev < water_elev_pct) &
            (moist > -0.05) &
            (grid_256 == TerrainType.EMPTY)
        )
        grid_256[water_mask] = TerrainType.WATER

        # ── Layer 3: Forest from independent noise field ───────────
        forest_noise = self._sample_noise(gen_size, seed_f, scale=35.0)
        empty_mask = grid_256 == TerrainType.EMPTY
        empty_count = empty_mask.sum()
        if empty_count > 0:
            # Target ~10% of total grid as forest
            target_forest = int(gen_size * gen_size * 0.10)
            forest_vals = forest_noise[empty_mask]
            threshold = np.percentile(forest_vals, max(0, 100 - target_forest / empty_count * 100))
            forest_mask = empty_mask & (forest_noise > threshold)
            grid_256[forest_mask] = TerrainType.FOREST

        # ── Layer 4: Fire at forest edges (sparse) ───────────────────
        forest_binary = (grid_256 == TerrainType.FOREST).astype(np.int8)
        forest_dilated = binary_dilation(forest_binary, iterations=1)
        fire_candidates = forest_dilated & (grid_256 == TerrainType.EMPTY)

        fire_noise = self._sample_noise(gen_size, seed_f, scale=20.0, octaves=2)
        grid_256[fire_candidates & (fire_noise > 0.2)] = TerrainType.FIRE

        # ── Spawn margin: clear spawn corners only ─────────────────────
        # Blue spawns top-left, Red spawns bottom-right
        grid_256[:margin_px, :margin_px] = TerrainType.EMPTY
        grid_256[-margin_px:, -margin_px:] = TerrainType.EMPTY

        # ── Upscale 256x256 → 1024x1024 ─────────────────────────────
        self.grid = np.kron(
            grid_256, np.ones((upscale, upscale), dtype=np.int8)
        )
        self._surface_dirty = True

    def generate_corridors(
        self,
        spawn_margin: int = 5,
        rng: Optional[random.Random] = None
    ) -> None:
        """
        Generate a corridor map for movement training.

        Creates a grid pattern of corridors (EMPTY cells) through a field of
        OBSTACLE terrain.

        Args:
            spawn_margin: Cells from edges to keep clear for spawning
            rng: Random number generator (unused, kept for API consistency)
        """
        from combatenv.config import CORRIDOR_WIDTH, CORRIDOR_SPACING, OPERATIONAL_GRID_SIZE

        self.clear()

        # Step 1: Fill entire map with obstacles (except spawn areas)
        for x in range(self.width):
            for y in range(self.height):
                in_blue_spawn = x < spawn_margin and y < spawn_margin
                in_red_spawn = x >= self.width - spawn_margin and y >= self.height - spawn_margin
                if not in_blue_spawn and not in_red_spawn:
                    self.set(x, y, TerrainType.OBSTACLE)

        # Step 2: Carve corridors aligned with operational grid
        for cy in range(OPERATIONAL_GRID_SIZE + 1):
            base_y = cy * CORRIDOR_SPACING
            for w in range(CORRIDOR_WIDTH):
                y = base_y + w
                if 0 <= y < self.height:
                    for x in range(self.width):
                        self.set(x, y, TerrainType.EMPTY)

        for cx in range(OPERATIONAL_GRID_SIZE + 1):
            base_x = cx * CORRIDOR_SPACING
            for w in range(CORRIDOR_WIDTH):
                x = base_x + w
                if 0 <= x < self.width:
                    for y in range(self.height):
                        self.set(x, y, TerrainType.EMPTY)

        self._surface_dirty = True
