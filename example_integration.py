"""
Example integration of the rendering system with the main game loop.

This demonstrates how to use the modular rendering functions from renderer.py
with agent and FOV data to create a complete game visualization.
"""

import pygame
import sys

from config import WINDOW_SIZE, FPS
from agent import spawn_all_teams
from fov import get_fov_overlap
from renderer import render_all


def main():
    """Main game loop demonstrating renderer integration."""
    # Initialize Pygame
    pygame.init()

    # Create display surface
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Grid-World - Rendering Demo")
    clock = pygame.time.Clock()

    # Spawn agents (game state initialization)
    blue_agents, red_agents = spawn_all_teams()

    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q and (event.mod & pygame.KMOD_SHIFT):
                    running = False

        # Update game state (agent movement/logic)
        for agent in blue_agents:
            agent.wander()  # Simple autonomous movement

        for agent in red_agents:
            agent.wander()

        # Calculate FOV data (game state calculation)
        blue_only_fov, red_only_fov, overlap_fov = get_fov_overlap(
            blue_agents,
            red_agents
        )

        # RENDERING: Call the complete rendering pipeline
        # This is the only rendering call needed per frame
        render_all(
            surface=screen,
            blue_agents=blue_agents,
            red_agents=red_agents,
            blue_fov_cells=blue_only_fov,
            red_fov_cells=red_only_fov,
            overlap_cells=overlap_fov
        )

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

    # Clean shutdown
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
