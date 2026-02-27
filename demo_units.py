"""
Demo script for the strategic unit/swarm system.

This demonstrates:
- Unit-based spawning (10 units per team, 10 agents each)
- Boids flocking behavior (cohesion, separation, alignment)
- Waypoint-based movement
- Visual representation of unit cohesion

Usage:
    source ~/.venvs/pg/bin/activate
    python demo_units.py

Controls:
    - ESC: Exit
    - 1-5: Set waypoint for blue unit 0-4 to mouse position
    - 6-0: Set waypoint for blue unit 5-9 to mouse position
    - SPACE: Clear all waypoints
"""

import random
import pygame
from combatenv import TacticalCombatEnv, EnvConfig
from combatenv.unit import spawn_all_units, get_all_agents_from_units
from combatenv.config import GRID_SIZE, CELL_SIZE, NUM_UNITS_PER_TEAM, AGENTS_PER_UNIT
from combatenv.renderer import render_strategic_grid


def generate_attack_waypoint(unit, enemy_units, grid_size):
    """
    Generate a waypoint toward enemy territory or enemy units.

    Args:
        unit: The unit to generate waypoint for
        enemy_units: List of enemy units to target
        grid_size: Size of the grid

    Returns:
        (x, y) waypoint coordinates
    """
    # Find nearest alive enemy unit
    my_centroid = unit.centroid
    nearest_enemy = None
    nearest_dist = float('inf')

    for enemy in enemy_units:
        if not enemy.is_eliminated:
            enemy_centroid = enemy.centroid
            dist = ((my_centroid[0] - enemy_centroid[0])**2 +
                    (my_centroid[1] - enemy_centroid[1])**2)**0.5
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_enemy = enemy

    if nearest_enemy:
        # Move toward enemy centroid with some randomness
        target = nearest_enemy.centroid
        # Add randomness to avoid all units converging on same spot
        offset_x = random.uniform(-5, 5)
        offset_y = random.uniform(-5, 5)
        x = max(2, min(grid_size - 2, target[0] + offset_x))
        y = max(2, min(grid_size - 2, target[1] + offset_y))
        return (x, y)
    else:
        # No enemies left, move to center
        return (grid_size / 2, grid_size / 2)


def main():
    """Run the unit/swarm demo."""

    # Create environment
    config = EnvConfig(
        respawn_enabled=False,
        terminate_on_controlled_death=False,
        terminate_on_team_elimination=True,
        max_steps=5000
    )

    env = TacticalCombatEnv(render_mode="human", config=config)

    # Reset to get terrain
    env.reset(seed=42)

    # Replace agents with unit-spawned agents
    blue_units, red_units = spawn_all_units(
        num_units_per_team=NUM_UNITS_PER_TEAM,
        agents_per_unit=AGENTS_PER_UNIT,
        terrain_grid=env.terrain_grid
    )

    # Get all agents from units
    blue_agents = get_all_agents_from_units(blue_units)
    red_agents = get_all_agents_from_units(red_units)

    # Replace environment agents
    env.blue_agents = blue_agents
    env.red_agents = red_agents
    env.all_agents = blue_agents + red_agents
    env.alive_agents = blue_agents + red_agents  # Critical: needed for projectile collision
    env.controlled_agent = blue_agents[0]

    # Rebuild spatial grid
    env.spatial_grid.clear()
    for agent in env.all_agents:
        env.spatial_grid.insert(agent)

    # Enable strategic grid overlay
    env.show_strategic_grid = True

    print(f"Created {len(blue_units)} blue units and {len(red_units)} red units")
    print(f"Total agents: {len(blue_agents)} blue, {len(red_agents)} red")
    print("\nControls:")
    print("  1-5: Set waypoint for blue units 0-4")
    print("  6-0: Set waypoint for blue units 5-9")
    print("  SPACE: Clear all waypoints")
    print("  Click: Set waypoint for unit 0")
    print("  ` (backtick): Toggle debug overlay")
    print("  F: Toggle FOV overlay")
    print("  ESC: Exit")

    # Initial render
    env.render()

    running = True
    step_count = 0
    waypoint_update_interval = 120  # Update AI waypoints every 2 seconds (60 fps)

    # Initial waypoints for red team (attack blue territory)
    for unit in red_units:
        wp = generate_attack_waypoint(unit, blue_units, GRID_SIZE)
        unit.set_waypoint(wp[0], wp[1])

    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_BACKQUOTE:
                    # Toggle debug overlay
                    env.show_debug = not env.show_debug
                    print(f"Debug overlay: {'ON' if env.show_debug else 'OFF'}")
                elif event.key == pygame.K_f:
                    # Toggle FOV overlay
                    env.show_fov = not env.show_fov
                    print(f"FOV overlay: {'ON' if env.show_fov else 'OFF'}")
                elif event.key == pygame.K_SPACE:
                    # Clear all waypoints
                    for unit in blue_units + red_units:
                        unit.clear_waypoint()
                    print("Cleared all waypoints")
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    # Set waypoint for blue units 0-4
                    unit_idx = event.key - pygame.K_1
                    if unit_idx < len(blue_units):
                        mouse_pos = pygame.mouse.get_pos()
                        grid_x = mouse_pos[0] / CELL_SIZE
                        grid_y = mouse_pos[1] / CELL_SIZE
                        blue_units[unit_idx].set_waypoint(grid_x, grid_y)
                        print(f"Set waypoint for blue unit {unit_idx} to ({grid_x:.1f}, {grid_y:.1f})")
                elif event.key in [pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0]:
                    # Set waypoint for blue units 5-9
                    if event.key == pygame.K_0:
                        unit_idx = 9
                    else:
                        unit_idx = event.key - pygame.K_6 + 5
                    if unit_idx < len(blue_units):
                        mouse_pos = pygame.mouse.get_pos()
                        grid_x = mouse_pos[0] / CELL_SIZE
                        grid_y = mouse_pos[1] / CELL_SIZE
                        blue_units[unit_idx].set_waypoint(grid_x, grid_y)
                        print(f"Set waypoint for blue unit {unit_idx} to ({grid_x:.1f}, {grid_y:.1f})")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Click to set waypoint for unit 0
                mouse_pos = pygame.mouse.get_pos()
                grid_x = mouse_pos[0] / CELL_SIZE
                grid_y = mouse_pos[1] / CELL_SIZE
                if len(blue_units) > 0:
                    blue_units[0].set_waypoint(grid_x, grid_y)
                    print(f"Set waypoint for blue unit 0 to ({grid_x:.1f}, {grid_y:.1f})")

        # Update agents with boids behavior
        dt = 1.0 / 60.0

        # Periodically update red team waypoints (AI pursuit)
        if step_count % waypoint_update_interval == 0:
            for unit in red_units:
                if not unit.is_eliminated:
                    wp = generate_attack_waypoint(unit, blue_units, GRID_SIZE)
                    unit.set_waypoint(wp[0], wp[1])

        for unit in blue_units + red_units:
            for agent in unit.alive_agents:
                if not agent.is_stuck:
                    # Normal wandering - flocking emerges from RL training, not hardcoded
                    agent.wander(dt=dt, other_agents=env.all_agents, terrain_grid=env.terrain_grid)
                # Update stamina
                agent.update_stamina(dt, agent.is_moving)

        # Update alive agents list and spatial grid
        env.alive_agents = [a for a in env.all_agents if a.is_alive]
        if env.spatial_grid is not None:
            env.spatial_grid.clear()
            for agent in env.alive_agents:
                env.spatial_grid.insert(agent)

        # Combat phase (autonomous shooting)
        from combatenv.agent import try_shoot_at_visible_target
        from combatenv.config import MOVEMENT_ACCURACY_PENALTY
        from combatenv.projectile import Projectile

        for agent in env.all_agents:
            if agent.is_alive and not agent.is_stuck:
                agent.update_cooldown(dt)
                agent.update_reload(dt)

                # Get nearby enemies
                if env.spatial_grid is not None:
                    nearby = env.spatial_grid.get_nearby_agents(agent)
                    enemies = [a for a in nearby if a.team != agent.team and a.is_alive]

                    if enemies:
                        projectile = try_shoot_at_visible_target(
                            agent, enemies,
                            movement_penalty=MOVEMENT_ACCURACY_PENALTY,
                            terrain_grid=env.terrain_grid
                        )
                        if projectile is not None and isinstance(projectile, Projectile):
                            env.projectiles.append(projectile)

        # Update projectiles
        env._update_projectiles(dt)

        # Process terrain effects
        env._process_terrain_effects()

        # Render
        env.render()

        # Draw strategic grid overlay
        if env._screen is not None:
            render_strategic_grid(env._screen)

        # Draw waypoints and unit info
        screen = env._screen
        if screen is not None:
            font = pygame.font.Font(None, 20)

            for i, unit in enumerate(blue_units):
                if unit.waypoint:
                    # Draw waypoint marker
                    wp_x = int(unit.waypoint[0] * CELL_SIZE)
                    wp_y = int(unit.waypoint[1] * CELL_SIZE)
                    pygame.draw.circle(screen, (0, 100, 255), (wp_x, wp_y), 8, 2)
                    pygame.draw.line(screen, (0, 100, 255), (wp_x - 5, wp_y), (wp_x + 5, wp_y), 2)
                    pygame.draw.line(screen, (0, 100, 255), (wp_x, wp_y - 5), (wp_x, wp_y + 5), 2)

                # Draw unit number at centroid
                centroid = unit.centroid
                cx = int(centroid[0] * CELL_SIZE)
                cy = int(centroid[1] * CELL_SIZE)
                text = font.render(str(i), True, (0, 0, 200))
                screen.blit(text, (cx - 4, cy - 6))

            for i, unit in enumerate(red_units):
                if unit.waypoint:
                    wp_x = int(unit.waypoint[0] * CELL_SIZE)
                    wp_y = int(unit.waypoint[1] * CELL_SIZE)
                    pygame.draw.circle(screen, (255, 100, 0), (wp_x, wp_y), 8, 2)

                centroid = unit.centroid
                cx = int(centroid[0] * CELL_SIZE)
                cy = int(centroid[1] * CELL_SIZE)
                text = font.render(str(i), True, (200, 0, 0))
                screen.blit(text, (cx - 4, cy - 6))

            pygame.display.flip()

        step_count += 1

        # Check for team elimination
        blue_alive = sum(1 for a in blue_agents if a.is_alive)
        red_alive = sum(1 for a in red_agents if a.is_alive)

        # Periodic status update
        if step_count % 60 == 0:
            print(f"Step {step_count}: Blue={blue_alive}, Red={red_alive}, Projectiles={len(env.projectiles)}")

        if blue_alive == 0 or red_alive == 0:
            winner = "Blue" if red_alive == 0 else "Red"
            print(f"\n{winner} team wins! Steps: {step_count}")
            print(f"Blue alive: {blue_alive}, Red alive: {red_alive}")
            break

        # Cap frame rate
        pygame.time.Clock().tick(60)

    env.close()
    print(f"\nDemo ended after {step_count} steps")


if __name__ == "__main__":
    main()
