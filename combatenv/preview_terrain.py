"""
Standalone terrain preview script.

Generates terrain and displays it in a pygame window.

Controls:
    SPACE: Regenerate with a new random seed
    S: Set a specific seed (type in terminal)
    C / Cmd+C / Ctrl+C: Copy current seed to clipboard
    Q: Quit

Usage:
    combatenv-preview-terrain
    combatenv-preview-terrain --seed 42
"""

import argparse
import platform
import subprocess

import numpy as np
import pygame

from combatenv.terrain import TerrainGrid, TerrainType
from combatenv.config import (
    WINDOW_SIZE, GRID_SIZE,
    COLOR_BACKGROUND, COLOR_OBSTACLE, COLOR_FIRE, COLOR_FOREST, COLOR_WATER,
)
from combatenv.renderer import (
    render_water_depth, render_forest_depth,
    render_lava_variation, render_mountain_elevation,
)
import combatenv.renderer as _renderer


TERRAIN_NAMES = {
    TerrainType.EMPTY: "Empty",
    TerrainType.OBSTACLE: "Obstacle",
    TerrainType.FIRE: "Fire",
    TerrainType.FOREST: "Forest",
    TerrainType.WATER: "Water",
}

TERRAIN_COLORS_RGB = {
    TerrainType.EMPTY: COLOR_BACKGROUND,
    TerrainType.OBSTACLE: COLOR_OBSTACLE,
    TerrainType.FIRE: COLOR_FIRE,
    TerrainType.FOREST: COLOR_FOREST,
    TerrainType.WATER: COLOR_WATER,
}

# Color LUT for surfarray rendering
COLOR_LUT = np.array([
    COLOR_BACKGROUND,
    COLOR_OBSTACLE,
    COLOR_FIRE,
    COLOR_FOREST,
    COLOR_WATER,
], dtype=np.uint8)


def compute_stats(grid: TerrainGrid) -> dict:
    """Compute terrain distribution percentages."""
    total = grid.grid.size
    counts = np.bincount(grid.grid.ravel().astype(np.int32), minlength=5)
    return {
        TerrainType(i): counts[i] / total * 100
        for i in range(5)
    }


def render_terrain(surface: pygame.Surface, grid: TerrainGrid) -> None:
    """Render terrain at pixel level using surfarray with color variation overlays."""
    rgb = COLOR_LUT[grid.grid]
    pygame.surfarray.blit_array(surface, rgb)
    render_water_depth(surface, grid)
    render_forest_depth(surface, grid)
    render_lava_variation(surface, grid)
    render_mountain_elevation(surface, grid)


def render_stats_overlay(surface: pygame.Surface, stats: dict, seed_val: int) -> None:
    """Render terrain distribution stats as overlay text."""
    font = pygame.font.Font(None, 24)
    panel_width = 220
    panel_height = 200
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 180))

    y = 10
    title = font.render(f"Seed: {seed_val}", True, (255, 255, 0))
    panel.blit(title, (10, y))
    y += 28

    for terrain_type in TerrainType:
        pct = stats.get(terrain_type, 0.0)
        color = TERRAIN_COLORS_RGB[terrain_type]
        # Draw color swatch
        pygame.draw.rect(panel, color, (10, y + 2, 14, 14))
        pygame.draw.rect(panel, (200, 200, 200), (10, y + 2, 14, 14), 1)
        # Draw text
        text = font.render(f"{TERRAIN_NAMES[terrain_type]}: {pct:.1f}%", True, (220, 220, 220))
        panel.blit(text, (30, y))
        y += 24

    # Controls hint
    y += 4
    hint_font = pygame.font.Font(None, 20)
    for hint_line in ["SPACE=regen  S=seed", "C=copy seed  Q=quit"]:
        hint = hint_font.render(hint_line, True, (150, 150, 150))
        panel.blit(hint, (10, y))
        y += 16

    surface.blit(panel, (10, 10))


def copy_to_clipboard(text: str) -> None:
    """Copy text to system clipboard (cross-platform)."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        elif system == "Linux":
            subprocess.run(["xclip", "-selection", "clipboard"],
                           input=text.encode(), check=True)
        elif system == "Windows":
            subprocess.run(["clip"], input=text.encode(), check=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Preview terrain generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Terrain Preview")
    clock = pygame.time.Clock()

    import random
    seed_val = args.seed if args.seed is not None else random.randint(0, 99999)

    def invalidate_overlays():
        _renderer._WATER_DEPTH_DIRTY = True
        _renderer._FOREST_DEPTH_DIRTY = True
        _renderer._LAVA_DIRTY = True
        _renderer._MOUNTAIN_DIRTY = True

    def generate(seed):
        nonlocal grid, stats
        rng = random.Random(seed)
        grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        grid.generate_random(rng=rng)
        stats = compute_stats(grid)
        invalidate_overlays()

    # Generate initial terrain
    grid = None
    stats = {}
    generate(seed_val)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif event.key == pygame.K_SPACE:
                    seed_val = random.randint(0, 99999)
                    generate(seed_val)

                elif event.key == pygame.K_s:
                    # Prompt for seed in terminal
                    try:
                        raw = input(f"Enter seed (current: {seed_val}): ").strip()
                        if raw:
                            seed_val = int(raw)
                            generate(seed_val)
                    except (ValueError, EOFError):
                        pass

                elif event.key == pygame.K_c:
                    copy_to_clipboard(str(seed_val))
                    pygame.display.set_caption(f"Terrain Preview — Seed {seed_val} copied")

        render_terrain(screen, grid)
        render_stats_overlay(screen, stats, seed_val)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
