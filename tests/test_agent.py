"""
Unit tests for the grid-world multi-agent system.

Tests cover:
- Agent movement and rotation
- Boundary collision detection
- Agent-to-agent collision
- Field of view calculations
- Spatial partitioning
- Spawning logic

Run with: pytest tests/test_agent.py -v
"""

import pytest
import math
from combatenv import (
    Agent, spawn_team, spawn_all_teams,
    get_fov_cells, is_point_in_fov_cone, get_fov_overlap,
    get_visible_agents, normalize_angle, angle_difference,
    SpatialGrid, config
)
GRID_SIZE = config.GRID_SIZE
BOUNDARY_MARGIN = config.BOUNDARY_MARGIN
AGENT_COLLISION_RADIUS = config.AGENT_COLLISION_RADIUS


class TestAgent:
    """Tests for Agent class."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        assert agent.position == (10.0, 10.0)
        assert agent.orientation == 0.0
        assert agent.team == "blue"

    def test_orientation_normalization(self):
        """Test orientation is normalized to [0, 360) on creation."""
        agent = Agent(position=(10.0, 10.0), orientation=400.0, team="blue")
        assert 0 <= agent.orientation < 360
        assert agent.orientation == 40.0

        agent2 = Agent(position=(10.0, 10.0), orientation=-90.0, team="red")
        assert agent2.orientation == 270.0

    def test_move_forward_basic(self):
        """Test basic forward movement."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent.move_forward(speed=1.0, dt=1.0)

        # Orientation 0° = right/east, so x should increase
        assert agent.position[0] > 10.0
        assert abs(agent.position[1] - 10.0) < 0.01  # y should stay same

    def test_move_forward_all_directions(self):
        """Test movement in all cardinal directions."""
        # Right (0°)
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent.move_forward(speed=1.0, dt=1.0)
        assert agent.position[0] > 10.0

        # Down (90°)
        agent = Agent(position=(10.0, 10.0), orientation=90.0, team="blue")
        agent.move_forward(speed=1.0, dt=1.0)
        assert agent.position[1] > 10.0

        # Left (180°)
        agent = Agent(position=(10.0, 10.0), orientation=180.0, team="blue")
        agent.move_forward(speed=1.0, dt=1.0)
        assert agent.position[0] < 10.0

        # Up (270°)
        agent = Agent(position=(10.0, 10.0), orientation=270.0, team="blue")
        agent.move_forward(speed=1.0, dt=1.0)
        assert agent.position[1] < 10.0

    def test_move_backward(self):
        """Test backward movement."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent.move_backward(speed=1.0, dt=1.0)

        # Moving backward from 0° should move left (decrease x)
        assert agent.position[0] < 10.0

    def test_boundary_collision(self):
        """Test agent cannot move past boundaries."""
        agent = Agent(position=(0.5, 10.0), orientation=180.0, team="blue")
        agent.move_forward(speed=10.0, dt=1.0)

        # Should not go below BOUNDARY_MARGIN
        assert agent.position[0] >= BOUNDARY_MARGIN

    def test_agent_collision_detection(self):
        """Test agent-to-agent collision detection."""
        agent1 = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent2 = Agent(position=(12.0, 10.0), orientation=0.0, team="red")

        # Agents start 2.0 units apart
        # Collision radius is 0.8, so they need to be > 0.8 apart

        # Movement that keeps them far enough apart (> collision radius)
        result1 = agent1.move_forward(speed=0.5, dt=1.0, other_agents=[agent2])
        assert result1 == True  # Should succeed (1.5 units apart after move)

        # Now try to move much closer (would violate collision radius)
        result2 = agent1.move_forward(speed=1.0, dt=1.0, other_agents=[agent2])

        # Should be blocked by collision detection (would be 0.5 apart < 0.8)
        assert result2 == False

    def test_rotate_left(self):
        """Test counter-clockwise rotation."""
        agent = Agent(position=(10.0, 10.0), orientation=90.0, team="blue")
        agent.rotate_left(degrees=90.0, dt=1.0)
        assert agent.orientation == 0.0

    def test_rotate_right(self):
        """Test clockwise rotation."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent.rotate_right(degrees=90.0, dt=1.0)
        assert agent.orientation == 90.0

    def test_rotation_wrapping(self):
        """Test rotation wraps around 360°."""
        agent = Agent(position=(10.0, 10.0), orientation=350.0, team="blue")
        agent.rotate_right(degrees=20.0, dt=1.0)
        assert agent.orientation == 10.0

        agent2 = Agent(position=(10.0, 10.0), orientation=10.0, team="blue")
        agent2.rotate_left(degrees=20.0, dt=1.0)
        assert agent2.orientation == 350.0

    def test_is_at_boundary(self):
        """Test boundary detection."""
        # Near left boundary
        agent = Agent(position=(0.8, 10.0), orientation=0.0, team="blue")
        assert agent.is_at_boundary()

        # Far from boundaries
        agent2 = Agent(position=(32.0, 32.0), orientation=0.0, team="blue")
        assert not agent2.is_at_boundary()

    def test_is_facing_boundary(self):
        """Test boundary facing detection."""
        # Near left boundary, facing left
        agent = Agent(position=(0.8, 32.0), orientation=180.0, team="blue")
        assert agent.is_facing_boundary()

        # Near left boundary, facing right (away)
        agent2 = Agent(position=(0.8, 32.0), orientation=0.0, team="blue")
        assert not agent2.is_facing_boundary()

    def test_get_grid_position(self):
        """Test grid position calculation."""
        agent = Agent(position=(10.5, 20.7), orientation=0.0, team="blue")
        grid_pos = agent.get_grid_position()
        assert grid_pos == (10, 20)

    def test_delta_time_independence(self):
        """Test movement is frame-rate independent."""
        # Two agents, same speed, different dt
        agent1 = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent2 = Agent(position=(10.0, 10.0), orientation=0.0, team="red")

        # Agent1: 1 frame at dt=1.0
        agent1.move_forward(speed=5.0, dt=1.0)

        # Agent2: 2 frames at dt=0.5
        agent2.move_forward(speed=5.0, dt=0.5)
        agent2.move_forward(speed=5.0, dt=0.5)

        # Should end up at approximately the same position
        assert abs(agent1.position[0] - agent2.position[0]) < 0.01
        assert abs(agent1.position[1] - agent2.position[1]) < 0.01


class TestSpawning:
    """Tests for agent spawning."""

    def test_spawn_team_count(self):
        """Test correct number of agents spawned."""
        agents = spawn_team("blue", num_agents=10)
        assert len(agents) == 10

    def test_spawn_team_correct_team(self):
        """Test agents have correct team assignment."""
        blue_agents = spawn_team("blue", num_agents=5)
        assert all(a.team == "blue" for a in blue_agents)

        red_agents = spawn_team("red", num_agents=5)
        assert all(a.team == "red" for a in red_agents)

    def test_spawn_team_quadrants(self):
        """Test teams spawn in correct quadrants."""
        blue_agents = spawn_team("blue", num_agents=10)
        red_agents = spawn_team("red", num_agents=10)

        # Blue should be in top-left (low x, low y)
        for agent in blue_agents:
            assert agent.position[0] < GRID_SIZE / 2
            assert agent.position[1] < GRID_SIZE / 2

        # Red should be in bottom-right (high x, high y)
        for agent in red_agents:
            assert agent.position[0] > GRID_SIZE / 2
            assert agent.position[1] > GRID_SIZE / 2

    def test_spawn_all_teams(self):
        """Test spawning both teams."""
        blue_agents, red_agents = spawn_all_teams()
        assert len(blue_agents) > 0
        assert len(red_agents) > 0
        assert all(a.team == "blue" for a in blue_agents)
        assert all(a.team == "red" for a in red_agents)

    def test_spawn_failure_too_many_agents(self):
        """Test spawn raises error when too many agents requested."""
        with pytest.raises(RuntimeError):
            # Try to spawn more agents than can fit
            spawn_team("blue", num_agents=10000)


class TestFOV:
    """Tests for field of view calculations."""

    def test_normalize_angle(self):
        """Test angle normalization."""
        assert normalize_angle(400) == 40
        assert normalize_angle(-90) == 270
        assert normalize_angle(0) == 0
        assert normalize_angle(359) == 359

    def test_angle_difference(self):
        """Test angle difference calculation."""
        # Same angle
        assert angle_difference(0, 0) == 0

        # 90° difference
        assert angle_difference(0, 90) == 90
        assert angle_difference(90, 0) == -90

        # Wrapping around 360°
        assert angle_difference(350, 10) == 20
        assert angle_difference(10, 350) == -20

    def test_point_in_fov_straight_ahead(self):
        """Test point directly ahead is visible."""
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(12.0, 10.0),
            fov_angle=90,
            max_range=5.0
        )
        assert result == True

    def test_point_behind_not_visible(self):
        """Test point behind agent is not visible."""
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(8.0, 10.0),
            fov_angle=90,
            max_range=5.0
        )
        assert result == False

    def test_point_out_of_range(self):
        """Test point beyond range is not visible."""
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(20.0, 10.0),
            fov_angle=90,
            max_range=5.0
        )
        assert result == False

    def test_point_at_fov_edge(self):
        """Test point at edge of FOV cone."""
        # 90° FOV = ±45° from orientation
        # Point at exactly 45° to the right should be visible
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(12.0, 12.0),  # 45° angle
            fov_angle=90,
            max_range=5.0
        )
        assert result == True

    def test_get_fov_cells(self):
        """Test FOV cell calculation."""
        cells = get_fov_cells(
            agent_pos=(32.0, 32.0),
            agent_orientation=0.0,
            fov_angle=90,
            max_range=3.0
        )

        # Should return some cells
        assert len(cells) > 0

        # All cells should be roughly within range
        # Note: Due to ray casting, cells at corners can be slightly beyond range
        for cell_x, cell_y in cells:
            dist = math.sqrt((cell_x - 32)**2 + (cell_y - 32)**2)
            assert dist <= 4.0  # Allow some tolerance for corner cells

    def test_get_visible_agents(self):
        """Test agent visibility detection."""
        observer = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")

        # Agent directly ahead
        target1 = Agent(position=(12.0, 10.0), orientation=0.0, team="red")

        # Agent behind
        target2 = Agent(position=(8.0, 10.0), orientation=0.0, team="red")

        visible = get_visible_agents(observer, [target1, target2], fov_angle=90, max_range=5.0)

        assert target1 in visible
        assert target2 not in visible

    def test_get_fov_overlap(self):
        """Test FOV overlap calculation."""
        blue_agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        red_agent = Agent(position=(12.0, 10.0), orientation=180.0, team="red")

        blue_only, red_only, overlap = get_fov_overlap([blue_agent], [red_agent])

        # Should have cells visible to both (overlap)
        assert isinstance(overlap, set)


class TestSpatialGrid:
    """Tests for spatial partitioning."""

    def test_spatial_grid_creation(self):
        """Test spatial grid initialization."""
        grid = SpatialGrid(cell_size=2.0)
        assert grid.cell_size == 2.0
        assert len(grid.grid) == 0

    def test_spatial_grid_insert(self):
        """Test inserting agents into spatial grid."""
        grid = SpatialGrid(cell_size=2.0)
        agent = Agent(position=(5.0, 5.0), orientation=0.0, team="blue")

        grid.insert(agent)
        assert len(grid.grid) > 0

    def test_spatial_grid_build(self):
        """Test building spatial grid from agent list."""
        agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(10.0, 10.0), orientation=0.0, team="red"),
        ]

        grid = SpatialGrid(cell_size=2.0)
        grid.build(agents)

        stats = grid.get_statistics()
        assert stats['total_agents'] == 2

    def test_spatial_grid_nearby_query(self):
        """Test querying nearby agents."""
        agents = [
            Agent(position=(5.0, 5.0), orientation=0.0, team="blue"),
            Agent(position=(5.5, 5.5), orientation=0.0, team="blue"),
            Agent(position=(50.0, 50.0), orientation=0.0, team="red"),
        ]

        grid = SpatialGrid(cell_size=2.0)
        grid.build(agents)

        # Query around first agent
        nearby = grid.get_nearby_agents(agents[0])

        # Should find itself and nearby agent, but not far one
        assert agents[0] in nearby
        assert agents[1] in nearby
        assert len(nearby) < len(agents)  # Shouldn't return all agents

    def test_spatial_grid_statistics(self):
        """Test spatial grid statistics."""
        agents = [
            Agent(position=(i * 5.0, i * 5.0), orientation=0.0, team="blue")
            for i in range(10)
        ]

        grid = SpatialGrid(cell_size=2.0)
        grid.build(agents)

        stats = grid.get_statistics()
        assert stats['total_agents'] == 10
        assert stats['num_cells'] > 0
        assert stats['avg_agents_per_cell'] > 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_agent_at_grid_corner(self):
        """Test agent at grid corner."""
        agent = Agent(position=(0.5, 0.5), orientation=0.0, team="blue")
        assert agent.is_at_boundary()

    def test_agent_at_exact_boundary(self):
        """Test agent at exact boundary position."""
        agent = Agent(position=(BOUNDARY_MARGIN, 32.0), orientation=0.0, team="blue")
        # Should handle this gracefully
        agent.move_backward(speed=1.0, dt=1.0)
        assert agent.position[0] >= BOUNDARY_MARGIN

    def test_zero_delta_time(self):
        """Test movement with zero delta time."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        original_pos = agent.position

        agent.move_forward(speed=5.0, dt=0.0)

        # Should not move
        assert agent.position == original_pos

    def test_very_large_delta_time(self):
        """Test movement with very large delta time."""
        agent = Agent(position=(32.0, 32.0), orientation=0.0, team="blue")

        # Large dt should still respect boundaries
        agent.move_forward(speed=100.0, dt=10.0)

        assert agent.position[0] < GRID_SIZE
        assert agent.position[0] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
