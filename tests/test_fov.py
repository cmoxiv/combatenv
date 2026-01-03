"""
Unit tests for the field of view system.

Tests cover:
- Angle normalization and difference
- Point-in-FOV-cone detection
- FOV cell calculation
- Team FOV aggregation
- LOS blocking by terrain

Run with: pytest tests/test_fov.py -v
"""

import pytest
import math
from combatenv import (
    normalize_angle,
    angle_difference,
    is_point_in_fov_cone,
    get_fov_cells,
    get_team_fov_cells,
    get_fov_overlap,
    get_layered_fov_overlap,
    is_agent_visible_to_agent,
    get_visible_agents,
    Agent,
    TerrainGrid,
    TerrainType
)


class TestNormalizeAngle:
    """Tests for angle normalization."""

    def test_zero(self):
        """Test zero stays zero."""
        assert normalize_angle(0) == 0

    def test_full_rotation(self):
        """Test 360 becomes 0."""
        assert normalize_angle(360) == 0

    def test_over_360(self):
        """Test angles over 360 are normalized."""
        assert normalize_angle(450) == 90
        assert normalize_angle(720) == 0

    def test_negative(self):
        """Test negative angles are normalized."""
        assert normalize_angle(-90) == 270
        assert normalize_angle(-180) == 180
        assert normalize_angle(-360) == 0

    def test_normal_range(self):
        """Test angles in normal range stay same."""
        assert normalize_angle(45) == 45
        assert normalize_angle(180) == 180
        assert normalize_angle(359) == 359


class TestAngleDifference:
    """Tests for angle difference calculation."""

    def test_same_angle(self):
        """Test same angle gives zero difference."""
        assert abs(angle_difference(0, 0)) < 0.01
        assert abs(angle_difference(180, 180)) < 0.01

    def test_simple_difference(self):
        """Test simple angle differences."""
        assert abs(angle_difference(0, 90) - 90) < 0.01
        assert abs(angle_difference(90, 0) - (-90)) < 0.01

    def test_wrap_around(self):
        """Test wrap-around at 360 degrees."""
        assert abs(angle_difference(350, 10) - 20) < 0.01
        assert abs(angle_difference(10, 350) - (-20)) < 0.01

    def test_opposite(self):
        """Test opposite directions."""
        assert abs(abs(angle_difference(0, 180)) - 180) < 0.01


class TestIsPointInFovCone:
    """Tests for point-in-FOV-cone detection."""

    def test_in_front(self):
        """Test point directly in front is visible."""
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(12.0, 10.0),
            fov_angle=90,
            max_range=5.0
        )
        assert result == True

    def test_behind(self):
        """Test point behind is not visible."""
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(8.0, 10.0),
            fov_angle=90,
            max_range=5.0
        )
        assert result == False

    def test_out_of_range(self):
        """Test point out of range is not visible."""
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(20.0, 10.0),
            fov_angle=90,
            max_range=5.0
        )
        assert result == False

    def test_at_fov_edge(self):
        """Test point at edge of FOV cone."""
        # 90 degree FOV = 45 degrees on each side
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(12.0, 12.0),  # 45 degrees
            fov_angle=90,
            max_range=5.0
        )
        assert result == True

    def test_outside_fov_angle(self):
        """Test point outside FOV angle."""
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(10.0, 15.0),  # 90 degrees off
            fov_angle=90,
            max_range=5.0
        )
        assert result == False

    def test_at_agent_position(self):
        """Test point at agent position is visible."""
        result = is_point_in_fov_cone(
            agent_pos=(10.0, 10.0),
            agent_orientation=0.0,
            point=(10.0, 10.0),
            fov_angle=90,
            max_range=5.0
        )
        assert result == True


class TestGetFovCells:
    """Tests for FOV cell calculation."""

    def test_returns_set(self):
        """Test function returns a set."""
        cells = get_fov_cells((10.0, 10.0), 0.0, 90, 3)
        assert isinstance(cells, set)

    def test_includes_agent_cell(self):
        """Test agent's own cell is included."""
        cells = get_fov_cells((10.5, 10.5), 0.0, 90, 3)
        assert (10, 10) in cells

    def test_includes_cells_in_front(self):
        """Test cells in front are included."""
        cells = get_fov_cells((10.0, 10.0), 0.0, 90, 3)
        assert (11, 10) in cells
        assert (12, 10) in cells

    def test_excludes_cells_behind(self):
        """Test cells behind are excluded."""
        cells = get_fov_cells((10.0, 10.0), 0.0, 90, 3)
        assert (8, 10) not in cells
        assert (7, 10) not in cells

    def test_with_terrain_blocking(self):
        """Test buildings block FOV."""
        terrain = TerrainGrid(20, 20)
        terrain.set(11, 10, TerrainType.BUILDING)  # Building in front

        cells = get_fov_cells((10.0, 10.0), 0.0, 90, 5, terrain_grid=terrain)

        # Cell after building should be blocked
        assert (12, 10) not in cells
        assert (13, 10) not in cells


class TestTeamFov:
    """Tests for team FOV calculation."""

    def test_combines_multiple_agents(self):
        """Test team FOV combines all agents."""
        agents = [
            Agent(position=(10.0, 10.0), orientation=0.0, team="blue"),
            Agent(position=(30.0, 30.0), orientation=90.0, team="blue")
        ]

        team_fov = get_team_fov_cells(agents, fov_angle=90, max_range=3)

        # Should include cells from both agents
        assert (10, 10) in team_fov or (11, 10) in team_fov
        assert (30, 30) in team_fov or (30, 31) in team_fov

    def test_empty_team(self):
        """Test empty team returns empty set."""
        team_fov = get_team_fov_cells([], fov_angle=90, max_range=3)
        assert team_fov == set()


class TestFovOverlap:
    """Tests for FOV overlap detection."""

    def test_returns_three_sets(self):
        """Test get_fov_overlap returns three sets."""
        blue = [Agent(position=(10.0, 10.0), orientation=0.0, team="blue")]
        red = [Agent(position=(12.0, 10.0), orientation=180.0, team="red")]

        blue_only, red_only, overlap = get_fov_overlap(blue, red)

        assert isinstance(blue_only, set)
        assert isinstance(red_only, set)
        assert isinstance(overlap, set)

    def test_overlap_exists_when_facing(self):
        """Test overlap exists when agents face each other."""
        blue = [Agent(position=(10.0, 10.0), orientation=0.0, team="blue")]
        red = [Agent(position=(12.0, 10.0), orientation=180.0, team="red")]

        _, _, overlap = get_fov_overlap(blue, red)

        # Should have some overlap in the middle
        assert len(overlap) > 0


class TestLayeredFovOverlap:
    """Tests for layered FOV overlap."""

    def test_returns_seven_sets(self):
        """Test layered overlap returns correct number of sets."""
        blue = [Agent(position=(10.0, 10.0), orientation=0.0, team="blue")]
        red = [Agent(position=(12.0, 10.0), orientation=180.0, team="red")]

        result = get_layered_fov_overlap(
            blue, red,
            near_range=3, near_angle=90,
            far_range=5, far_angle=120
        )

        assert len(result) == 7


class TestAgentVisibility:
    """Tests for agent-to-agent visibility."""

    def test_visible_in_front(self):
        """Test agent in front is visible."""
        observer = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        target = Agent(position=(12.0, 10.0), orientation=0.0, team="red")

        visible = is_agent_visible_to_agent(observer, target, fov_angle=90, max_range=5)
        assert visible == True

    def test_not_visible_behind(self):
        """Test agent behind is not visible."""
        observer = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        target = Agent(position=(8.0, 10.0), orientation=0.0, team="red")

        visible = is_agent_visible_to_agent(observer, target, fov_angle=90, max_range=5)
        assert visible == False

    def test_blocked_by_building(self):
        """Test building blocks visibility."""
        terrain = TerrainGrid(20, 20)
        terrain.set(11, 10, TerrainType.BUILDING)

        observer = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        target = Agent(position=(13.0, 10.0), orientation=0.0, team="red")

        visible = is_agent_visible_to_agent(
            observer, target,
            fov_angle=90, max_range=5,
            terrain_grid=terrain
        )
        assert visible == False


class TestGetVisibleAgents:
    """Tests for getting all visible agents."""

    def test_returns_visible_only(self):
        """Test only visible agents are returned."""
        observer = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        in_front = Agent(position=(12.0, 10.0), orientation=0.0, team="red")
        behind = Agent(position=(8.0, 10.0), orientation=0.0, team="red")

        visible = get_visible_agents(
            observer, [in_front, behind],
            fov_angle=90, max_range=5
        )

        assert in_front in visible
        assert behind not in visible


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
