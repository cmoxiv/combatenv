"""
Standalone terrain preview script.

Generates terrain and displays it in a pygame window.
Press SPACE to regenerate with a new seed.
Press Q to quit.

Usage:
    python preview_terrain.py
    python preview_terrain.py --seed 42
"""

import argparse
import numpy as np
import pygame

from combatenv.terrain import TerrainGrid, TerrainType
from combatenv.config import (
    WINDOW_SIZE, GRID_SIZE,
    COLOR_BACKGROUND, COLOR_OBSTACLE, COLOR_FIRE, COLOR_FOREST, COLOR_WATER,
)


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
    """Render terrain at pixel level using surfarray."""
    rgb = COLOR_LUT[grid.grid]
    pygame.surfarray.blit_array(surface, rgb)


def render_stats_overlay(surface: pygame.Surface, stats: dict, seed_val: int) -> None:
    """Render terrain distribution stats as overlay text."""
    font = pygame.font.Font(None, 24)
    panel_width = 220
    panel_height = 180
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
    hint = pygame.font.Font(None, 20).render("SPACE=regen  Q=quit", True, (150, 150, 150))
    panel.blit(hint, (10, y))

    surface.blit(panel, (10, 10))


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

    # Generate initial terrain
    grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
    rng = random.Random(seed_val)
    grid.generate_random(rng=rng)
    stats = compute_stats(grid)

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
                    rng = random.Random(seed_val)
                    grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
                    grid.generate_random(rng=rng)
                    stats = compute_stats(grid)

        render_terrain(screen, grid)
        render_stats_overlay(screen, stats, seed_val)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
