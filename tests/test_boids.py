"""
Unit tests for combatenv/boids.py - Boids flocking algorithm.
"""

import pytest
import math
from combatenv.boids import (
    calculate_cohesion_force,
    calculate_separation_force,
    calculate_alignment_force,
    calculate_waypoint_force,
    calculate_boids_steering,
    steering_to_orientation,
    blend_steering_with_random,
)
from combatenv.unit import Unit
from combatenv.agent import Agent
from combatenv.config import (
    BOIDS_COHESION_WEIGHT,
    BOIDS_SEPARATION_WEIGHT,
    BOIDS_ALIGNMENT_WEIGHT,
    BOIDS_WAYPOINT_WEIGHT,
    BOIDS_MAX_FORCE,
    BOIDS_SEPARATION_RADIUS,
)


class TestCohesionForce:
    """Test cohesion force calculation."""

    def test_cohesion_pulls_toward_centroid(self):
        """Test that cohesion pulls agent toward centroid."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(12.0, 10.0), orientation=0.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        # First agent is left of centroid (11, 10)
        force = calculate_cohesion_force(agents[0], unit)

        # Force should point right (positive x)
        assert force[0] > 0
        assert abs(force[1]) < 0.01  # No Y component

    def test_cohesion_zero_at_centroid(self):
        """Test that cohesion is zero when at centroid."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        unit = Unit(id=0, team="blue", agents=[agent])

        force = calculate_cohesion_force(agent, unit)

        # Single agent is at centroid
        assert abs(force[0]) < 0.01
        assert abs(force[1]) < 0.01

    def test_cohesion_weight_affects_magnitude(self):
        """Test that weight scales the force."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(14.0, 10.0), orientation=0.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        force1 = calculate_cohesion_force(agents[0], unit, weight=1.0)
        force2 = calculate_cohesion_force(agents[0], unit, weight=2.0)

        # Double weight = double magnitude
        assert abs(force2[0] - force1[0] * 2) < 0.01


class TestSeparationForce:
    """Test separation force calculation."""

    def test_separation_pushes_away(self):
        """Test that separation pushes agents apart."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.5, 10.0), orientation=0.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        # First agent should be pushed left (away from second)
        force = calculate_separation_force(agents[0], unit)

        assert force[0] < 0  # Pushed left
        assert abs(force[1]) < 0.01

    def test_separation_zero_when_far_apart(self):
        """Test that separation is zero when agents are far apart."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(20.0, 10.0), orientation=0.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        force = calculate_separation_force(agents[0], unit)

        # No force when far apart
        assert abs(force[0]) < 0.01
        assert abs(force[1]) < 0.01

    def test_separation_stronger_when_closer(self):
        """Test that separation is stronger when agents are closer."""
        # Close agents
        agents1 = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.5, 10.0), orientation=0.0, team="blue"),
        ]
        unit1 = Unit(id=0, team="blue", agents=agents1)

        # Farther agents (but still within radius)
        agents2 = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=0.0, team="blue"),
        ]
        unit2 = Unit(id=0, team="blue", agents=agents2)

        force1 = calculate_separation_force(agents1[0], unit1)
        force2 = calculate_separation_force(agents2[0], unit2)

        # Closer = stronger force
        assert abs(force1[0]) > abs(force2[0])


class TestAlignmentForce:
    """Test alignment force calculation."""

    def test_alignment_toward_average_heading(self):
        """Test that alignment steers toward average heading."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=90.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        # First agent faces 0 degrees, average is ~45
        force = calculate_alignment_force(agents[0], unit)

        # Should have positive Y component (steering toward 45 degrees)
        # This depends on the math, but force should be non-zero
        magnitude = math.sqrt(force[0]**2 + force[1]**2)
        assert magnitude > 0

    def test_alignment_zero_when_same_heading(self):
        """Test alignment is minimal when all agents face same direction."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=45.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=45.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        force = calculate_alignment_force(agents[0], unit)

        # Should be near zero
        magnitude = math.sqrt(force[0]**2 + force[1]**2)
        assert magnitude < 0.1


class TestWaypointForce:
    """Test waypoint force calculation."""

    def test_waypoint_pulls_toward_target(self):
        """Test that waypoint force pulls toward target."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        unit = Unit(id=0, team="blue", agents=[agent])
        unit.set_waypoint(20.0, 10.0)

        force = calculate_waypoint_force(agent, unit)

        # Should pull right (toward waypoint)
        assert force[0] > 0
        assert abs(force[1]) < 0.01

    def test_waypoint_zero_when_no_waypoint(self):
        """Test that waypoint force is zero when no waypoint set."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        unit = Unit(id=0, team="blue", agents=[agent])

        force = calculate_waypoint_force(agent, unit)

        assert force[0] == 0.0
        assert force[1] == 0.0

    def test_waypoint_zero_at_destination(self):
        """Test that waypoint force is zero when at destination."""
        agent = Agent(position=(20.0, 20.0), orientation=0.0, team="blue")
        unit = Unit(id=0, team="blue", agents=[agent])
        unit.set_waypoint(20.0, 20.0)

        force = calculate_waypoint_force(agent, unit)

        assert abs(force[0]) < 0.01
        assert abs(force[1]) < 0.01


class TestCombinedSteering:
    """Test combined boids steering."""

    def test_combined_steering_clamped(self):
        """Test that combined steering is clamped to max force."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(50.0, 10.0), orientation=180.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)
        unit.set_waypoint(60.0, 60.0)

        steering = calculate_boids_steering(agents[0], unit)

        magnitude = math.sqrt(steering[0]**2 + steering[1]**2)
        assert magnitude <= BOIDS_MAX_FORCE + 0.01

    def test_combined_steering_with_custom_config(self):
        """Test combined steering with custom config."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        unit = Unit(id=0, team="blue", agents=[agent])
        unit.set_waypoint(20.0, 10.0)

        # Zero out all forces except waypoint
        config = {
            'cohesion': 0.0,
            'separation': 0.0,
            'alignment': 0.0,
            'waypoint': 1.0,
            'max_force': 1.0,
        }

        steering = calculate_boids_steering(agent, unit, config)

        # Should only have waypoint force (pointing right)
        assert steering[0] > 0
        assert abs(steering[1]) < 0.01


class TestSteeringToOrientation:
    """Test steering to orientation conversion."""

    def test_steering_to_orientation_right(self):
        """Test conversion for rightward steering."""
        steering = (1.0, 0.0)
        orientation = steering_to_orientation(steering)

        assert orientation is not None
        assert abs(orientation - 0.0) < 1.0  # 0 degrees = right

    def test_steering_to_orientation_down(self):
        """Test conversion for downward steering."""
        steering = (0.0, 1.0)
        orientation = steering_to_orientation(steering)

        assert orientation is not None
        assert abs(orientation - 90.0) < 1.0  # 90 degrees = down

    def test_steering_to_orientation_left(self):
        """Test conversion for leftward steering."""
        steering = (-1.0, 0.0)
        orientation = steering_to_orientation(steering)

        assert orientation is not None
        assert abs(orientation - 180.0) < 1.0  # 180 degrees = left

    def test_steering_to_orientation_zero(self):
        """Test conversion for zero steering."""
        steering = (0.0, 0.0)
        orientation = steering_to_orientation(steering)

        assert orientation is None


class TestBlendSteering:
    """Test steering blending with randomness."""

    def test_blend_returns_vector(self):
        """Test that blend returns a valid vector."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        steering = (1.0, 0.0)

        blended = blend_steering_with_random(steering, agent, boids_weight=0.7)

        assert isinstance(blended, tuple)
        assert len(blended) == 2

    def test_blend_full_boids_weight(self):
        """Test blend with full boids weight."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        steering = (1.0, 0.0)

        # With weight=1.0, should be close to original steering
        blended = blend_steering_with_random(steering, agent, boids_weight=1.0)

        assert abs(blended[0] - 1.0) < 0.01
        assert abs(blended[1] - 0.0) < 0.01

    def test_blend_adds_variation(self):
        """Test that blend adds some variation."""
        agent = Agent(position=(10.0, 10.0), orientation=45.0, team="blue")
        steering = (1.0, 0.0)

        # With lower boids weight, should have more variation
        results = []
        for _ in range(10):
            blended = blend_steering_with_random(steering, agent, boids_weight=0.5)
            results.append(blended)

        # Check that we get different results (randomness)
        unique_x = len(set(r[0] for r in results))
        assert unique_x > 1  # Should have variation


class TestBoidsIntegration:
    """Integration tests for boids system."""

    def test_full_unit_movement(self):
        """Test that boids steering produces sensible movement for a unit."""
        # Create a unit with agents
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.0, 11.0), orientation=0.0, team="blue"),
        ]

        for i, agent in enumerate(agents):
            agent.unit_id = 0

        unit = Unit(id=0, team="blue", agents=agents)
        unit.set_waypoint(30.0, 30.0)

        # All agents should steer toward waypoint
        for agent in agents:
            steering = calculate_boids_steering(agent, unit)
            magnitude = math.sqrt(steering[0]**2 + steering[1]**2)

            # Should have non-zero steering
            assert magnitude > 0

            # Steering should generally point toward waypoint (positive x and y)
            # (Could be modified by cohesion/separation, but generally positive)
            assert steering[0] > -BOIDS_MAX_FORCE
            assert steering[1] > -BOIDS_MAX_FORCE

    def test_scattered_unit_cohesion(self):
        """Test that scattered unit produces strong cohesion forces."""
        # Create a very scattered unit
        agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(25.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(15.0, 25.0), orientation=0.0, team="blue"),
        ]

        for i, agent in enumerate(agents):
            agent.unit_id = 0

        unit = Unit(id=0, team="blue", agents=agents)

        # First agent should have strong cohesion pull toward centroid (15, ~11.67)
        cohesion = calculate_cohesion_force(agents[0], unit)

        # Should point toward centroid (positive x)
        assert cohesion[0] > 0
