"""
Unit tests for the projectile system.

Tests cover:
- Projectile creation and factory function
- Movement and position updates
- Lifetime and boundary expiration
- Collision detection
- Accuracy-based deviation

Run with: pytest tests/test_projectile.py -v
"""

import pytest
import math
import random
from combatenv import Projectile, create_projectile, Agent, config
PROJECTILE_SPEED = config.PROJECTILE_SPEED
PROJECTILE_DAMAGE = config.PROJECTILE_DAMAGE
GRID_SIZE = config.GRID_SIZE


class TestProjectileCreation:
    """Tests for projectile creation."""

    def test_position(self):
        """Test projectile spawns in front of shooter."""
        proj = create_projectile(
            shooter_position=(10.0, 10.0),
            shooter_orientation=0.0,  # Facing right
            shooter_team="blue",
            shooter_id=123,
            accuracy=1.0
        )

        # Should spawn in front (higher x for orientation 0)
        assert proj.position[0] > 10.0

    def test_velocity(self):
        """Test projectile has correct velocity direction."""
        proj = create_projectile(
            shooter_position=(10.0, 10.0),
            shooter_orientation=0.0,  # Facing right
            shooter_team="blue",
            shooter_id=123,
            accuracy=1.0
        )

        # Velocity should be positive x for facing right
        assert proj.velocity[0] > 0
        assert abs(proj.velocity[1]) < 0.1  # Minimal y component

    def test_velocity_magnitude(self):
        """Test projectile velocity magnitude equals PROJECTILE_SPEED."""
        proj = create_projectile(
            shooter_position=(10.0, 10.0),
            shooter_orientation=45.0,
            shooter_team="blue",
            shooter_id=123,
            accuracy=1.0
        )

        speed = math.sqrt(proj.velocity[0]**2 + proj.velocity[1]**2)
        assert abs(speed - PROJECTILE_SPEED) < 0.01

    def test_team(self):
        """Test projectile has correct team assignment."""
        proj_blue = create_projectile(
            shooter_position=(10.0, 10.0),
            shooter_orientation=0.0,
            shooter_team="blue",
            shooter_id=123
        )
        assert proj_blue.owner_team == "blue"

        proj_red = create_projectile(
            shooter_position=(10.0, 10.0),
            shooter_orientation=0.0,
            shooter_team="red",
            shooter_id=456
        )
        assert proj_red.owner_team == "red"

    def test_shooter_id(self):
        """Test projectile stores shooter ID."""
        proj = create_projectile(
            shooter_position=(10.0, 10.0),
            shooter_orientation=0.0,
            shooter_team="blue",
            shooter_id=99999
        )
        assert proj.shooter_id == 99999

    def test_accuracy_deviation_perfect(self):
        """Test perfect accuracy has no deviation."""
        random.seed(42)
        proj = create_projectile(
            shooter_position=(10.0, 10.0),
            shooter_orientation=0.0,
            shooter_team="blue",
            shooter_id=123,
            accuracy=1.0
        )

        # With perfect accuracy, should go straight
        assert abs(proj.velocity[1]) < 0.1

    def test_accuracy_deviation_low(self):
        """Test low accuracy can have deviation."""
        random.seed(42)

        # Create multiple projectiles with low accuracy
        deviations = []
        for i in range(20):
            proj = create_projectile(
                shooter_position=(10.0, 10.0),
                shooter_orientation=0.0,
                shooter_team="blue",
                shooter_id=i,
                accuracy=0.5
            )
            # Calculate angle from velocity
            angle = math.degrees(math.atan2(proj.velocity[1], proj.velocity[0]))
            deviations.append(abs(angle))

        # With 0.5 accuracy, max deviation is 90 degrees
        # Should have some variation
        assert max(deviations) > 1.0, "Low accuracy should produce some deviation"


class TestProjectileUpdate:
    """Tests for projectile movement updates."""

    def test_moves_position(self):
        """Test update moves projectile position."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(15.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        initial_x = proj.position[0]
        proj.update(0.1)

        assert proj.position[0] > initial_x

    def test_movement_amount(self):
        """Test movement is velocity * dt."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(10.0, 5.0),
            owner_team="blue",
            shooter_id=1
        )

        proj.update(0.5)

        expected_x = 10.0 + 10.0 * 0.5
        expected_y = 10.0 + 5.0 * 0.5
        assert abs(proj.position[0] - expected_x) < 0.01
        assert abs(proj.position[1] - expected_y) < 0.01

    def test_decreases_lifetime(self):
        """Test update decreases lifetime."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1,
            lifetime_remaining=1.0
        )

        proj.update(0.1)
        assert proj.lifetime_remaining < 1.0
        assert abs(proj.lifetime_remaining - 0.9) < 0.01

    def test_returns_true_expired(self):
        """Test update returns True when expired."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(1.0, 0.0),
            owner_team="blue",
            shooter_id=1,
            lifetime_remaining=0.05
        )

        expired = proj.update(0.1)
        assert expired == True

    def test_returns_false_active(self):
        """Test update returns False when still active."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1,
            lifetime_remaining=1.0
        )

        expired = proj.update(0.1)
        assert expired == False

    def test_returns_true_out_of_bounds_right(self):
        """Test projectile expires when leaving grid right."""
        proj = Projectile(
            position=(GRID_SIZE - 0.5, 10.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        expired = proj.update(0.1)
        assert expired == True

    def test_returns_true_out_of_bounds_left(self):
        """Test projectile expires when leaving grid left."""
        proj = Projectile(
            position=(0.5, 10.0),
            velocity=(-10.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        expired = proj.update(0.1)
        assert expired == True


class TestProjectileCollision:
    """Tests for projectile collision detection."""

    def test_hit_enemy(self):
        """Test projectile hits enemy agent."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        enemy = Agent(position=(10.2, 10.0), orientation=0.0, team="red")
        assert proj.check_collision(enemy) == True

    def test_miss_far_away(self):
        """Test projectile misses agent far away."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        enemy = Agent(position=(50.0, 50.0), orientation=0.0, team="red")
        assert proj.check_collision(enemy) == False

    def test_skips_dead(self):
        """Test projectile doesn't hit dead agents."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        enemy = Agent(position=(10.2, 10.0), orientation=0.0, team="red")
        enemy.health = 0  # Dead

        assert proj.check_collision(enemy) == False

    def test_skips_same_team_friendly_fire_off(self):
        """Test projectile skips same team when friendly fire disabled."""
        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=1
        )

        # This test depends on FRIENDLY_FIRE_ENABLED config
        # Import and check the setting
        from combatenv.config import FRIENDLY_FIRE_ENABLED

        ally = Agent(position=(10.2, 10.0), orientation=0.0, team="blue")

        if not FRIENDLY_FIRE_ENABLED:
            assert proj.check_collision(ally) == False

    def test_skips_shooter(self):
        """Test projectile doesn't hit the shooter."""
        shooter = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        shooter_id = id(shooter)

        proj = Projectile(
            position=(10.0, 10.0),
            velocity=(10.0, 0.0),
            owner_team="blue",
            shooter_id=shooter_id
        )

        assert proj.check_collision(shooter) == False


class TestProjectileDamage:
    """Tests for projectile damage values."""

    def test_default_damage(self):
        """Test projectile has default damage from config."""
        proj = create_projectile(
            shooter_position=(10.0, 10.0),
            shooter_orientation=0.0,
            shooter_team="blue",
            shooter_id=1
        )

        assert proj.damage == PROJECTILE_DAMAGE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
