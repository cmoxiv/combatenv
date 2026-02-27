"""
Unit tests for combatenv/unit.py - Unit dataclass and spawning functions.
"""

import pytest
import math
from combatenv.unit import (
    Unit,
    spawn_unit,
    spawn_units_for_team,
    spawn_all_units,
    get_all_agents_from_units,
    get_unit_for_agent,
)
from combatenv.agent import Agent
from combatenv.config import (
    NUM_UNITS_PER_TEAM,
    AGENTS_PER_UNIT,
    UNIT_COHESION_RADIUS,
    GRID_SIZE,
)


class TestUnitBasics:
    """Test basic Unit functionality."""

    def test_unit_creation(self):
        """Test creating a Unit with agents."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=45.0, team="blue"),
            Agent(position=(10.0, 11.0), orientation=90.0, team="blue"),
        ]

        unit = Unit(id=0, team="blue", agents=agents)

        assert unit.id == 0
        assert unit.team == "blue"
        assert len(unit.agents) == 3
        assert unit.waypoint is None

    def test_unit_centroid(self):
        """Test centroid calculation."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(12.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 12.0), orientation=0.0, team="blue"),
        ]

        unit = Unit(id=0, team="blue", agents=agents)
        centroid = unit.centroid

        # Centroid should be at (11, 10.67)
        assert abs(centroid[0] - 11.0) < 0.01
        assert abs(centroid[1] - 10.67) < 0.1

    def test_unit_centroid_with_dead_agents(self):
        """Test centroid only considers alive agents."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(20.0, 10.0), orientation=0.0, team="blue"),
        ]
        agents[1].health = 0  # Kill second agent

        unit = Unit(id=0, team="blue", agents=agents)
        centroid = unit.centroid

        # Centroid should only use first agent
        assert centroid[0] == 10.0
        assert centroid[1] == 10.0

    def test_unit_alive_agents(self):
        """Test alive_agents property."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(12.0, 10.0), orientation=0.0, team="blue"),
        ]
        agents[1].health = 0  # Kill middle agent

        unit = Unit(id=0, team="blue", agents=agents)

        assert len(unit.alive_agents) == 2
        assert unit.alive_count == 2

    def test_unit_is_eliminated(self):
        """Test is_eliminated property."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=0.0, team="blue"),
        ]

        unit = Unit(id=0, team="blue", agents=agents)

        assert not unit.is_eliminated

        # Kill all agents
        for agent in agents:
            agent.health = 0

        assert unit.is_eliminated


class TestUnitCohesion:
    """Test cohesion-related functionality."""

    def test_cohesion_score_at_centroid(self):
        """Test cohesion score when agent is at centroid."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        unit = Unit(id=0, team="blue", agents=[agent])

        score = unit.get_cohesion_score(agent)
        assert score == 1.0

    def test_cohesion_score_far_from_centroid(self):
        """Test cohesion score when agent is far from centroid."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.0 + UNIT_COHESION_RADIUS * 2, 10.0), orientation=0.0, team="blue"),
        ]

        unit = Unit(id=0, team="blue", agents=agents)

        # Third agent is far from centroid
        score = unit.get_cohesion_score(agents[2])
        assert score == 0.0

    def test_cohesion_score_mid_range(self):
        """Test cohesion score at mid-range."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.0 + UNIT_COHESION_RADIUS / 2, 10.0), orientation=0.0, team="blue"),
        ]

        unit = Unit(id=0, team="blue", agents=agents)

        score = unit.get_cohesion_score(agents[2])
        assert 0.0 < score < 1.0

    def test_formation_spread(self):
        """Test formation spread calculation."""
        # Tight formation
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.5, 10.0), orientation=0.0, team="blue"),
            Agent(position=(10.0, 10.5), orientation=0.0, team="blue"),
        ]

        unit = Unit(id=0, team="blue", agents=agents)
        spread = unit.get_formation_spread()

        assert spread < 1.0  # Tight formation


class TestUnitWaypoints:
    """Test waypoint functionality."""

    def test_set_waypoint(self):
        """Test setting a waypoint."""
        unit = Unit(id=0, team="blue", agents=[])
        unit.set_waypoint(30.0, 30.0)

        assert unit.waypoint == (30.0, 30.0)

    def test_set_waypoint_clamped(self):
        """Test waypoint is clamped to grid bounds."""
        unit = Unit(id=0, team="blue", agents=[])
        unit.set_waypoint(-10.0, GRID_SIZE + 10)

        assert unit.waypoint[0] == 0.5
        assert unit.waypoint[1] == GRID_SIZE - 0.5

    def test_clear_waypoint(self):
        """Test clearing a waypoint."""
        unit = Unit(id=0, team="blue", agents=[])
        unit.set_waypoint(30.0, 30.0)
        unit.clear_waypoint()

        assert unit.waypoint is None

    def test_is_at_waypoint(self):
        """Test waypoint arrival detection."""
        agents = [
            Agent(position=(30.0, 30.0), orientation=0.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        # No waypoint = at waypoint
        assert unit.is_at_waypoint()

        # Set waypoint at centroid
        unit.set_waypoint(30.0, 30.0)
        assert unit.is_at_waypoint()

        # Set far waypoint
        unit.set_waypoint(50.0, 50.0)
        assert not unit.is_at_waypoint()


class TestUnitAverageHeading:
    """Test average heading calculation."""

    def test_average_heading_single_agent(self):
        """Test average heading with single agent."""
        agent = Agent(position=(10.0, 10.0), orientation=90.0, team="blue")
        unit = Unit(id=0, team="blue", agents=[agent])

        assert abs(unit.average_heading - 90.0) < 1.0

    def test_average_heading_opposite_directions(self):
        """Test average heading with agents facing opposite directions."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=180.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        # Average of 0 and 180 should be ~90 or ~270
        heading = unit.average_heading
        assert heading in [90.0, 270.0] or abs(heading - 90) < 1 or abs(heading - 270) < 1

    def test_average_heading_same_direction(self):
        """Test average heading with agents facing same direction."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=45.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=45.0, team="blue"),
        ]
        unit = Unit(id=0, team="blue", agents=agents)

        assert abs(unit.average_heading - 45.0) < 1.0


class TestUnitSpawning:
    """Test unit spawning functions."""

    def test_spawn_unit_basic(self):
        """Test spawning a basic unit."""
        unit = spawn_unit(
            team="blue",
            unit_id=0,
            num_agents=5,
            center=(15.0, 15.0)
        )

        assert unit.id == 0
        assert unit.team == "blue"
        assert len(unit.agents) == 5
        assert all(a.team == "blue" for a in unit.agents)
        assert all(a.unit_id == 0 for a in unit.agents)

    def test_spawn_unit_agents_off_screen_in_reserve(self):
        """Test that spawned agents are off-screen when in reserve."""
        unit = spawn_unit(
            team="blue",
            unit_id=0,
            num_agents=5,
            center=(15.0, 15.0)
        )

        # Units start in reserve with agents off-screen
        assert unit.in_reserve is True
        for agent in unit.agents:
            assert agent.position[0] < 0  # Off-screen

    def test_dispatch_teleports_to_spawn_corner(self):
        """Test that dispatch moves agents to spawn corner."""
        unit = spawn_unit(
            team="blue",
            unit_id=0,
            num_agents=5,
            center=(15.0, 15.0)
        )

        # Dispatch to a waypoint
        unit.dispatch((32.0, 32.0))

        # Should no longer be in reserve
        assert unit.in_reserve is False

        # Agents should be at blue spawn corner (top-left, around 3,3)
        centroid = unit.centroid
        assert 0 < centroid[0] < 10  # Near left edge
        assert 0 < centroid[1] < 10  # Near top edge

    def test_spawn_units_for_team(self):
        """Test spawning all units for a team."""
        units = spawn_units_for_team(
            team="blue",
            num_units=5,
            agents_per_unit=5
        )

        assert len(units) == 5
        assert all(u.team == "blue" for u in units)

        # Check total agents
        total_agents = sum(len(u.agents) for u in units)
        assert total_agents == 25

    def test_spawn_all_units(self):
        """Test spawning units for both teams."""
        blue_units, red_units = spawn_all_units(
            num_units_per_team=5,
            agents_per_unit=5
        )

        assert len(blue_units) == 5
        assert len(red_units) == 5

        assert all(u.team == "blue" for u in blue_units)
        assert all(u.team == "red" for u in red_units)


class TestUnitHelpers:
    """Test helper functions."""

    def test_get_all_agents_from_units(self):
        """Test extracting all agents from units."""
        agents1 = [Agent(position=(10.0, 10.0), orientation=0.0, team="blue")]
        agents2 = [Agent(position=(11.0, 10.0), orientation=0.0, team="blue")]

        units = [
            Unit(id=0, team="blue", agents=agents1),
            Unit(id=1, team="blue", agents=agents2),
        ]

        all_agents = get_all_agents_from_units(units)
        assert len(all_agents) == 2

    def test_get_unit_for_agent(self):
        """Test finding unit for an agent."""
        agents = [Agent(position=(10.0, 10.0), orientation=0.0, team="blue")]
        agents[0].unit_id = 5

        units = [
            Unit(id=5, team="blue", agents=agents),
            Unit(id=6, team="blue", agents=[]),
        ]

        found = get_unit_for_agent(agents[0], units)
        assert found is not None
        assert found.id == 5

    def test_get_unit_for_agent_not_found(self):
        """Test finding unit for agent that doesn't belong to any."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent.unit_id = 99  # Non-existent unit

        units = [Unit(id=0, team="blue", agents=[])]

        found = get_unit_for_agent(agent, units)
        assert found is None

    def test_get_nearby_squadmates(self):
        """Test getting nearby squadmates."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(11.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(20.0, 20.0), orientation=0.0, team="blue"),
        ]

        unit = Unit(id=0, team="blue", agents=agents)

        nearby = unit.get_nearby_squadmates(agents[0], radius=3.0)
        assert len(nearby) == 1  # Only agent[1] is nearby
        assert agents[1] in nearby
        assert agents[2] not in nearby
