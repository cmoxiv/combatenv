"""
Projectile system for the Grid-World Multi-Agent Tactical Simulation.

This module implements the projectile-based combat system. Projectiles are
created when agents fire at targets and travel at constant velocity until
they hit an agent, expire, or leave the grid boundaries.

Projectile Characteristics:
    - Speed: 15 grid cells per second (configurable via PROJECTILE_SPEED)
    - Damage: 25 HP per hit (configurable via PROJECTILE_DAMAGE)
    - Lifetime: 2.0 seconds maximum (configurable via PROJECTILE_LIFETIME)
    - Collision Radius: 0.3 grid cells (configurable via PROJECTILE_RADIUS)

Accuracy System:
    Projectile direction is affected by accuracy, which determines the
    maximum angular deviation from the aimed direction:
        - accuracy 1.0: Perfect aim, no deviation
        - accuracy 0.9: +/- 18 degrees deviation (near FOV)
        - accuracy 0.5: +/- 90 degrees deviation (far FOV)
        - accuracy 0.0: +/- 180 degrees deviation (random)

    The actual deviation is uniformly distributed within the range.

Collision Detection:
    - Projectiles check collision with all agents each frame
    - Collision occurs when distance < PROJECTILE_RADIUS + AGENT_COLLISION_RADIUS
    - Self-collision is prevented using shooter_id
    - Friendly fire can be enabled/disabled via FRIENDLY_FIRE_ENABLED

Spawn Offset:
    Projectiles spawn 0.6 grid cells in front of the shooter to prevent
    immediate self-collision.

Example:
    >>> from projectile import create_projectile, Projectile
    >>> proj = create_projectile(
    ...     shooter_position=(5.0, 5.0),
    ...     shooter_orientation=45.0,
    ...     shooter_team="blue",
    ...     shooter_id=12345,
    ...     accuracy=0.9
    ... )
    >>> expired = proj.update(dt=0.016)  # Update for one frame
    >>> print(f"Projectile at {proj.position}, expired: {expired}")
"""

import math
import random
from typing import Tuple, Optional
from dataclasses import dataclass

from .config import (
    PROJECTILE_SPEED,
    PROJECTILE_DAMAGE,
    PROJECTILE_LIFETIME,
    PROJECTILE_RADIUS,
    GRID_SIZE,
    AGENT_COLLISION_RADIUS,
    FRIENDLY_FIRE_ENABLED
)
from .terrain import TerrainType


@dataclass
class Projectile:
    """
    Represents a projectile fired by an agent.

    Attributes:
        position: (x, y) coordinates in grid space
        velocity: (vx, vy) velocity in grid cells per second
        owner_team: Team that fired this projectile ("blue" or "red")
        shooter_id: ID of the agent who shot (to avoid self-hit)
        damage: Damage dealt on hit
        lifetime_remaining: Seconds until projectile expires
    """
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    owner_team: str
    shooter_id: int
    damage: int = PROJECTILE_DAMAGE
    lifetime_remaining: float = PROJECTILE_LIFETIME

    def update(self, dt: float, terrain_grid=None) -> bool:
        """
        Update projectile position and lifetime.

        Args:
            dt: Delta time in seconds
            terrain_grid: Optional TerrainGrid for building collision detection

        Returns:
            True if projectile should be removed (expired, out of bounds, or hit building)
        """
        # Update position
        new_x = self.position[0] + self.velocity[0] * dt
        new_y = self.position[1] + self.velocity[1] * dt
        self.position = (new_x, new_y)

        # Update lifetime
        self.lifetime_remaining -= dt

        # Check if should be removed
        if self.lifetime_remaining <= 0:
            return True

        # Check if out of bounds
        if (new_x < 0 or new_x >= GRID_SIZE or
            new_y < 0 or new_y >= GRID_SIZE):
            return True

        # Check if hit a building
        if terrain_grid is not None:
            cell_x = int(new_x)
            cell_y = int(new_y)
            if terrain_grid.get(cell_x, cell_y) == TerrainType.BUILDING:
                return True

        return False

    def check_collision(self, agent) -> bool:
        """
        Check if projectile hits an agent.

        Args:
            agent: Agent to check collision with

        Returns:
            True if collision detected
        """
        # Don't hit the shooter (use id() for object identity)
        if id(agent) == self.shooter_id:
            return False

        # Don't hit dead agents
        if not agent.is_alive:
            return False

        # Check friendly fire setting
        if not FRIENDLY_FIRE_ENABLED and self.owner_team == agent.team:
            return False

        # Check distance using squared values (faster than sqrt)
        dx = self.position[0] - agent.position[0]
        dy = self.position[1] - agent.position[1]
        distance_squared = dx * dx + dy * dy

        # Collision if within combined radius (compare squared values)
        collision_radius = PROJECTILE_RADIUS + AGENT_COLLISION_RADIUS
        return distance_squared < (collision_radius * collision_radius)


def create_projectile(
    shooter_position: Tuple[float, float],
    shooter_orientation: float,
    shooter_team: str,
    shooter_id: int,
    accuracy: float = 1.0,
    rng: Optional[random.Random] = None
) -> Projectile:
    """
    Create a projectile fired from an agent.

    Args:
        shooter_position: (x, y) position of shooting agent
        shooter_orientation: Orientation of shooting agent in degrees
        shooter_team: Team of shooting agent ("blue" or "red")
        shooter_id: ID of the shooting agent (to avoid self-collision)
        accuracy: Accuracy modifier (0.0-1.0), affects angle deviation
        rng: Optional random.Random instance for deterministic testing.
             If None, uses the global random module.

    Returns:
        New Projectile instance
    """
    # Use provided RNG or fall back to global random
    if rng is None:
        rng = random

    # Calculate angle deviation based on accuracy
    # accuracy 1.0 = no deviation
    # accuracy 0.5 = ±90° deviation
    # accuracy 0.9 = ±18° deviation
    max_deviation = (1.0 - accuracy) * 180.0
    deviation = rng.uniform(-max_deviation, max_deviation)

    # Apply deviation to orientation
    final_angle = shooter_orientation + deviation
    angle_rad = math.radians(final_angle)

    # Calculate velocity components
    vx = math.cos(angle_rad) * PROJECTILE_SPEED
    vy = math.sin(angle_rad) * PROJECTILE_SPEED

    # Spawn projectile slightly in front of shooter to avoid self-collision
    spawn_offset = 0.6  # Just beyond agent radius
    spawn_x = shooter_position[0] + math.cos(math.radians(shooter_orientation)) * spawn_offset
    spawn_y = shooter_position[1] + math.sin(math.radians(shooter_orientation)) * spawn_offset

    return Projectile(
        position=(spawn_x, spawn_y),
        velocity=(vx, vy),
        owner_team=shooter_team,
        shooter_id=shooter_id
    )


if __name__ == "__main__":
    """Basic self-tests for projectile module."""
    import sys

    def test_projectile_creation():
        """Test basic projectile creation."""
        proj = create_projectile(
            shooter_position=(5.0, 5.0),
            shooter_orientation=0.0,
            shooter_team="blue",
            shooter_id=12345,
            accuracy=1.0
        )

        # Should spawn in front of shooter
        assert proj.position[0] > 5.0, "Should spawn in front"
        assert abs(proj.position[1] - 5.0) < 0.1, "Should be on same y-axis"

        # Velocity should be positive x for 0 degree orientation
        assert proj.velocity[0] > 0, "Should move right"
        assert abs(proj.velocity[1]) < 0.1, "Should not move vertically"

        assert proj.owner_team == "blue"
        assert proj.shooter_id == 12345
        print("  projectile creation: OK")

    def test_projectile_update():
        """Test projectile movement."""
        proj = Projectile(
            position=(5.0, 5.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        initial_x = proj.position[0]
        initial_lifetime = proj.lifetime_remaining

        # Update for 0.1 seconds
        expired = proj.update(0.1)

        assert not expired, "Should not be expired yet"
        assert proj.position[0] > initial_x, "Should have moved"
        assert proj.lifetime_remaining < initial_lifetime, "Lifetime should decrease"
        print("  projectile update: OK")

    def test_projectile_expiration():
        """Test projectile lifetime expiration."""
        proj = Projectile(
            position=(5.0, 5.0),
            velocity=(1.0, 0.0),
            owner_team="blue",
            shooter_id=1,
            lifetime_remaining=0.05
        )

        # Update past lifetime
        expired = proj.update(0.1)
        assert expired, "Should be expired"
        print("  projectile expiration: OK")

    def test_projectile_out_of_bounds():
        """Test projectile goes out of bounds."""
        proj = Projectile(
            position=(63.9, 5.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        # Update to move past grid boundary
        expired = proj.update(0.1)
        assert expired, "Should be expired (out of bounds)"
        print("  projectile out of bounds: OK")

    def test_accuracy_deviation():
        """Test accuracy affects projectile direction."""
        random.seed(42)

        # Perfect accuracy - should go straight
        proj_perfect = create_projectile(
            shooter_position=(5.0, 5.0),
            shooter_orientation=0.0,
            shooter_team="blue",
            shooter_id=1,
            accuracy=1.0
        )

        # Low accuracy - may deviate
        proj_low = create_projectile(
            shooter_position=(5.0, 5.0),
            shooter_orientation=0.0,
            shooter_team="blue",
            shooter_id=2,
            accuracy=0.5
        )

        # Perfect accuracy should be along x-axis
        assert abs(proj_perfect.velocity[1]) < 0.1, "Perfect accuracy should be straight"
        print("  accuracy deviation: OK")

    # Run all tests
    print("Running projectile.py self-tests...")
    try:
        test_projectile_creation()
        test_projectile_update()
        test_projectile_expiration()
        test_projectile_out_of_bounds()
        test_accuracy_deviation()
        print("All projectile.py self-tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
