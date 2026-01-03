"""
Field of View (FOV) calculation system for the Grid-World Multi-Agent Tactical Simulation.

This module provides comprehensive FOV calculations for agent visibility, including:
- Single-agent FOV cone calculations
- Team-wide FOV aggregation
- Two-layer FOV system (near/far) for tactical awareness
- Overlap detection between opposing teams

Two-Layer FOV System:
    Near FOV:
        - Range: 3 grid cells (NEAR_FOV_RANGE)
        - Angle: 90 degrees (NEAR_FOV_ANGLE)
        - Purpose: High-accuracy targeting zone

    Far FOV:
        - Range: 5 grid cells (FAR_FOV_RANGE)
        - Angle: 120 degrees (FAR_FOV_ANGLE)
        - Purpose: Awareness zone with reduced accuracy

    The far FOV is a wider cone that extends beyond the near FOV, providing
    peripheral vision at the cost of accuracy.

FOV Cone Geometry:
    - Center: Agent's position
    - Direction: Agent's orientation (0=right, 90=down, 180=left, 270=up)
    - Half-angle: FOV angle / 2 on each side of center direction
    - Range: Maximum distance from agent center

Ray Casting Implementation:
    The get_fov_cells() function uses ray casting for efficient FOV calculation:
    1. Cast rays at regular angular intervals across the FOV cone
    2. Sample points along each ray from agent to max range
    3. Collect grid cells touched by the rays
    This is more efficient than checking all cells in a bounding box.

Overlap Detection:
    The get_layered_fov_overlap() function identifies three types of overlaps:
    - Near-Near: Both teams have near FOV (highest tactical engagement)
    - Mixed: One team near, other far (asymmetric engagement)
    - Far-Far: Both teams have far FOV (early warning zone)

Example:
    >>> from fov import get_fov_cells, is_agent_visible_to_agent
    >>> # Check if target is visible to observer
    >>> visible = is_agent_visible_to_agent(observer, target, fov_angle=90, max_range=3)
    >>> # Get all cells in an agent's FOV
    >>> cells = get_fov_cells((5.0, 5.0), orientation=45.0, fov_angle=90, max_range=3)
"""

import math
from typing import Tuple, Set, TYPE_CHECKING, Protocol, List

from .config import LOS, FOV_ANGLE, GRID_SIZE

if TYPE_CHECKING:
    from .agent import Agent


class HasPositionOrientation(Protocol):
    """Protocol for objects with position and orientation attributes."""
    position: Tuple[float, float]
    orientation: float
    team: str


class FOVCache:
    """
    Cache for FOV calculations to avoid redundant ray casting.

    Only recalculates FOV when an agent has moved or rotated significantly.
    """

    def __init__(self, position_threshold: float = 0.3, angle_threshold: float = 5.0):
        """
        Initialize FOV cache.

        Args:
            position_threshold: Recalculate if position changed by this many cells
            angle_threshold: Recalculate if orientation changed by this many degrees
        """
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold
        self._cache = {}  # agent_id -> (position, orientation, near_cells, far_cells)

    def clear(self):
        """Clear all cached FOV data."""
        self._cache.clear()

    def get_agent_fov(
        self,
        agent: HasPositionOrientation,
        near_angle: float,
        near_range: float,
        far_angle: float,
        far_range: float,
        terrain_grid=None
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Get FOV cells for an agent, using cache if valid.

        Args:
            agent: Agent to get FOV for
            near_angle: Near FOV angle
            near_range: Near FOV range
            far_angle: Far FOV angle
            far_range: Far FOV range
            terrain_grid: Terrain grid for LOS blocking

        Returns:
            Tuple of (near_cells, far_cells)
        """
        agent_id = id(agent)
        pos = agent.position
        orient = agent.orientation

        # Check if cache is valid
        if agent_id in self._cache:
            cached_pos, cached_orient, near_cells, far_cells = self._cache[agent_id]

            # Check position change
            dx = pos[0] - cached_pos[0]
            dy = pos[1] - cached_pos[1]
            pos_change = math.sqrt(dx * dx + dy * dy)

            # Check orientation change
            orient_change = abs((orient - cached_orient + 180) % 360 - 180)

            if pos_change < self.position_threshold and orient_change < self.angle_threshold:
                return near_cells, far_cells

        # Cache miss or invalid - recalculate
        near_cells = get_fov_cells(pos, orient, near_angle, near_range, terrain_grid)
        far_cells = get_fov_cells(pos, orient, far_angle, far_range, terrain_grid)

        self._cache[agent_id] = (pos, orient, near_cells, far_cells)
        return near_cells, far_cells

    def remove_agent(self, agent_id: int):
        """Remove an agent from the cache (e.g., when they die)."""
        self._cache.pop(agent_id, None)


# Global FOV cache instance
_fov_cache = FOVCache()


def get_fov_cache() -> FOVCache:
    """Get the global FOV cache instance."""
    return _fov_cache


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [0, 360).

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle in [0, 360)
    """
    return angle % 360


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest difference between two angles.

    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees

    Returns:
        Smallest angular difference in degrees (-180 to 180)
    """
    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff


def is_point_in_fov_cone(
    agent_pos: Tuple[float, float],
    agent_orientation: float,
    point: Tuple[float, float],
    fov_angle: float = FOV_ANGLE,
    max_range: float = LOS
) -> bool:
    """
    Check if a point falls within the agent's FOV cone.

    Args:
        agent_pos: Agent's (x, y) position
        agent_orientation: Agent's orientation in degrees
        point: Point to check (x, y)
        fov_angle: Total FOV angle in degrees (default 90)
        max_range: Maximum visibility range in grid cells

    Returns:
        True if the point is within the FOV cone
    """
    # Calculate vector from agent to point
    dx = point[0] - agent_pos[0]
    dy = point[1] - agent_pos[1]

    # Calculate distance
    distance = math.sqrt(dx * dx + dy * dy)

    # Check if within range
    if distance > max_range:
        return False

    # Handle case where point is at agent position
    if distance < 0.01:
        return True

    # Calculate angle to the point (in degrees)
    angle_to_point = math.degrees(math.atan2(dy, dx))

    # Normalize to [0, 360)
    angle_to_point = normalize_angle(angle_to_point)
    agent_orientation = normalize_angle(agent_orientation)

    # Calculate angular difference
    diff = abs(angle_difference(agent_orientation, angle_to_point))

    # Check if within FOV cone (half-angle on each side)
    half_fov = fov_angle / 2.0
    return diff <= half_fov


def get_fov_cells(
    agent_pos: Tuple[float, float],
    agent_orientation: float,
    fov_angle: float = FOV_ANGLE,
    max_range: float = LOS,
    terrain_grid=None
) -> Set[Tuple[int, int]]:
    """
    Calculate all grid cells within an agent's field of view using ray casting.

    Uses optimized ray casting to scan the FOV cone, which is more efficient
    than checking all cells in a bounding box. Rays are blocked by buildings
    if terrain_grid is provided.

    Args:
        agent_pos: Agent's (x, y) position in grid coordinates
        agent_orientation: Agent's orientation in degrees (0 = right/east)
        fov_angle: Total FOV angle in degrees (default 90)
        max_range: Maximum visibility range in grid cells (default LOS)
        terrain_grid: TerrainGrid for LOS blocking by buildings (optional)

    Returns:
        Set of (x, y) grid cell coordinates within the FOV cone
    """
    visible_cells = set()
    agent_x, agent_y = agent_pos

    # Calculate FOV cone bounds
    half_fov = fov_angle / 2.0
    start_angle = agent_orientation - half_fov
    end_angle = agent_orientation + half_fov

    # Number of rays to cast (adaptive based on FOV angle and range)
    # Use more rays for wider FOV angles and longer ranges
    num_rays = max(int(fov_angle / 3), 15)  # At least 15 rays, ~3 degrees apart

    # Cast rays at regular angular intervals
    for i in range(num_rays + 1):
        angle = start_angle + (fov_angle * i / num_rays)
        angle_rad = math.radians(angle)

        # Cast ray from agent position to max range
        # Use sub-cell steps for accuracy
        steps = int(max_range * 2) + 1  # 2 steps per cell for good coverage
        for step in range(steps + 1):
            distance = (step / 2.0)  # Convert steps to distance
            if distance > max_range:
                break

            # Calculate point along ray
            ray_x = agent_x + math.cos(angle_rad) * distance
            ray_y = agent_y + math.sin(angle_rad) * distance

            # Convert to grid cell coordinates using floor for correct rounding
            cell_x = math.floor(ray_x)
            cell_y = math.floor(ray_y)

            # Check if cell is within grid bounds
            if 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
                # Check if ray hits a building (blocks LOS)
                if terrain_grid and terrain_grid.blocks_los(cell_x, cell_y):
                    # Building blocks LOS - stop this ray
                    break

                visible_cells.add((cell_x, cell_y))

    return visible_cells


def get_fov_cells_for_agent(agent: HasPositionOrientation) -> Set[Tuple[int, int]]:
    """
    Convenience function to get FOV cells directly from an Agent instance.

    Args:
        agent: Agent instance with position and orientation

    Returns:
        Set of (x, y) grid cell coordinates within the agent's FOV
    """
    return get_fov_cells(agent.position, agent.orientation)


def get_team_fov_cells(
    agents: List[HasPositionOrientation],
    fov_angle: float = FOV_ANGLE,
    max_range: float = LOS,
    terrain_grid=None
) -> Set[Tuple[int, int]]:
    """
    Calculate the combined FOV for an entire team.

    Args:
        agents: List of Agent instances
        fov_angle: FOV cone angle in degrees (defaults to FOV_ANGLE)
        max_range: Maximum visibility range in grid cells (defaults to LOS)
        terrain_grid: TerrainGrid for LOS blocking by buildings (optional)

    Returns:
        Set of all grid cells visible to at least one team member
    """
    combined_fov = set()

    for agent in agents:
        agent_fov = get_fov_cells(agent.position, agent.orientation, fov_angle, max_range, terrain_grid)
        combined_fov.update(agent_fov)

    return combined_fov


def get_fov_overlap(
    blue_agents: List[HasPositionOrientation],
    red_agents: List[HasPositionOrientation]
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Calculate FOV coverage and overlaps for both teams.

    Args:
        blue_agents: List of blue team agents
        red_agents: List of red team agents

    Returns:
        Tuple of (blue_only_cells, red_only_cells, overlap_cells)
        - blue_only_cells: Cells visible only to blue team
        - red_only_cells: Cells visible only to red team
        - overlap_cells: Cells visible to both teams
    """
    blue_fov = get_team_fov_cells(blue_agents)
    red_fov = get_team_fov_cells(red_agents)

    overlap = blue_fov.intersection(red_fov)
    blue_only = blue_fov - overlap
    red_only = red_fov - overlap

    return blue_only, red_only, overlap


def calculate_fov_coverage_percentage(agents: List[HasPositionOrientation], grid_size: int = GRID_SIZE) -> float:
    """
    Calculate the percentage of the grid covered by a team's FOV.

    Args:
        agents: List of Agent instances
        grid_size: Size of the grid (grid_size x grid_size)

    Returns:
        Percentage of grid cells visible (0.0 to 100.0)
    """
    visible_cells = get_team_fov_cells(agents)
    total_cells = grid_size * grid_size
    coverage = (len(visible_cells) / total_cells) * 100.0

    return coverage


def is_agent_visible_to_agent(
    observer: HasPositionOrientation,
    target: HasPositionOrientation,
    fov_angle: float = FOV_ANGLE,
    max_range: float = LOS,
    terrain_grid=None
) -> bool:
    """
    Check if one agent can see another agent.

    Args:
        observer: The agent doing the observing
        target: The agent being observed
        fov_angle: FOV angle in degrees
        max_range: Maximum visibility range
        terrain_grid: TerrainGrid for LOS blocking by buildings (optional)

    Returns:
        True if target is within observer's FOV and not blocked by buildings
    """
    # First check if target is in FOV cone
    if not is_point_in_fov_cone(
        observer.position,
        observer.orientation,
        target.position,
        fov_angle,
        max_range
    ):
        return False

    # If no terrain grid, visibility is just based on FOV cone
    if terrain_grid is None:
        return True

    # Check LOS blocking by buildings using ray casting
    ox, oy = observer.position
    tx, ty = target.position
    dx = tx - ox
    dy = ty - oy
    distance = math.sqrt(dx * dx + dy * dy)

    if distance < 0.1:
        return True  # Same cell

    # Cast ray from observer to target, check for building collisions
    steps = int(distance * 2) + 1
    for step in range(1, steps):
        t = step / steps
        ray_x = ox + dx * t
        ray_y = oy + dy * t
        cell_x = math.floor(ray_x)
        cell_y = math.floor(ray_y)

        if terrain_grid.blocks_los(cell_x, cell_y):
            return False  # LOS blocked by building

    return True


def get_visible_agents(
    observer: HasPositionOrientation,
    potential_targets: List[HasPositionOrientation],
    fov_angle: float = FOV_ANGLE,
    max_range: float = LOS,
    terrain_grid=None
) -> List[HasPositionOrientation]:
    """
    Get all agents visible to an observer.

    Args:
        observer: The observing agent
        potential_targets: List of agents that could be visible
        fov_angle: FOV angle in degrees
        max_range: Maximum visibility range
        terrain_grid: TerrainGrid for LOS blocking by buildings (optional)

    Returns:
        List of agents visible to the observer
    """
    visible = []

    for target in potential_targets:
        if is_agent_visible_to_agent(observer, target, fov_angle, max_range, terrain_grid):
            visible.append(target)

    return visible


def get_layered_fov_overlap(
    blue_agents: List[HasPositionOrientation],
    red_agents: List[HasPositionOrientation],
    near_range: float,
    near_angle: float,
    far_range: float,
    far_angle: float,
    terrain_grid=None,
    fov_cache: FOVCache = None
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]],
           Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Calculate layered FOV (near and far) for both teams with overlap detection by layer.

    Uses FOV caching to avoid redundant calculations when agents haven't moved significantly.

    Args:
        blue_agents: List of blue team agents
        red_agents: List of red team agents
        near_range: Range for near FOV layer
        near_angle: Angle for near FOV layer
        far_range: Range for far FOV layer
        far_angle: Angle for far FOV layer
        terrain_grid: TerrainGrid for LOS blocking by buildings (optional)
        fov_cache: FOVCache instance for caching (uses global cache if None)

    Returns:
        Tuple of (blue_near, blue_far, red_near, red_far, overlap_near_near, overlap_mixed, overlap_far_far)
        - blue_near: Cells in blue team's near FOV
        - blue_far: Cells in blue team's far FOV (excluding near)
        - red_near: Cells in red team's near FOV
        - red_far: Cells in red team's far FOV (excluding near)
        - overlap_near_near: Both teams have near FOV (darkest)
        - overlap_mixed: One near, one far (medium)
        - overlap_far_far: Both teams have far FOV (lightest)
    """
    # Use global cache if none provided
    if fov_cache is None:
        fov_cache = _fov_cache

    # Collect FOV cells for all agents using cache
    blue_near_fov = set()
    blue_far_fov_full = set()
    red_near_fov = set()
    red_far_fov_full = set()

    for agent in blue_agents:
        near_cells, far_cells = fov_cache.get_agent_fov(
            agent, near_angle, near_range, far_angle, far_range, terrain_grid
        )
        blue_near_fov.update(near_cells)
        blue_far_fov_full.update(far_cells)

    for agent in red_agents:
        near_cells, far_cells = fov_cache.get_agent_fov(
            agent, near_angle, near_range, far_angle, far_range, terrain_grid
        )
        red_near_fov.update(near_cells)
        red_far_fov_full.update(far_cells)

    # Far FOV excluding near (only show the far ring, not the near area)
    blue_far_only = blue_far_fov_full - blue_near_fov
    red_far_only = red_far_fov_full - red_near_fov

    # Calculate different types of overlaps based on FOV layers
    # Near-Near: Both teams have near FOV (highest engagement, darkest purple)
    overlap_near_near = blue_near_fov & red_near_fov

    # Mixed overlaps: One team near, one team far
    blue_near_red_far = blue_near_fov & red_far_only
    blue_far_red_near = blue_far_only & red_near_fov
    overlap_mixed = blue_near_red_far | blue_far_red_near

    # Far-Far: Both teams only have far FOV (lowest engagement, lightest purple)
    overlap_far_far = blue_far_only & red_far_only

    return blue_near_fov, blue_far_only, red_near_fov, red_far_only, overlap_near_near, overlap_mixed, overlap_far_far


if __name__ == "__main__":
    """Basic self-tests for fov module."""
    import sys

    def test_normalize_angle():
        """Test angle normalization."""
        assert normalize_angle(0) == 0
        assert normalize_angle(360) == 0
        assert normalize_angle(450) == 90
        assert normalize_angle(-90) == 270
        assert normalize_angle(-360) == 0
        print("  normalize_angle: OK")

    def test_angle_difference():
        """Test angle difference calculation."""
        # Same angle
        assert abs(angle_difference(0, 0)) < 0.01

        # Simple difference
        assert abs(angle_difference(0, 45) - 45) < 0.01

        # Wrapping around
        assert abs(angle_difference(350, 10) - 20) < 0.01
        assert abs(angle_difference(10, 350) - (-20)) < 0.01
        print("  angle_difference: OK")

    def test_is_point_in_fov_cone():
        """Test point-in-FOV-cone detection."""
        agent_pos = (5.0, 5.0)

        # Point directly in front (facing right, 0 degrees)
        assert is_point_in_fov_cone(agent_pos, 0.0, (7.0, 5.0), 90, 3) == True

        # Point behind (180 degrees away)
        assert is_point_in_fov_cone(agent_pos, 0.0, (3.0, 5.0), 90, 3) == False

        # Point to the side (90 degrees away, outside 45-degree half-angle)
        assert is_point_in_fov_cone(agent_pos, 0.0, (5.0, 7.0), 90, 3) == False

        # Point within cone at angle
        assert is_point_in_fov_cone(agent_pos, 0.0, (7.0, 5.5), 90, 3) == True

        # Point out of range
        assert is_point_in_fov_cone(agent_pos, 0.0, (20.0, 5.0), 90, 3) == False
        print("  is_point_in_fov_cone: OK")

    def test_get_fov_cells():
        """Test FOV cell calculation."""
        agent_pos = (5.0, 5.0)

        cells = get_fov_cells(agent_pos, 0.0, 90, 3)

        # Should have some cells
        assert len(cells) > 0, "Should have visible cells"

        # Agent's own cell should be included
        assert (5, 5) in cells, "Agent's cell should be visible"

        # Cells in front should be included
        assert (6, 5) in cells, "Cell in front should be visible"
        assert (7, 5) in cells, "Cell 2 ahead should be visible"

        # Cells behind should not be included
        assert (3, 5) not in cells, "Cell behind should not be visible"
        print("  get_fov_cells: OK")

    def test_get_team_fov_cells():
        """Test team FOV calculation."""
        class MockAgent:
            def __init__(self, pos, orient):
                self.position = pos
                self.orientation = orient
                self.team = "blue"

        agents = [
            MockAgent((5.0, 5.0), 0.0),
            MockAgent((10.0, 10.0), 90.0)
        ]

        team_fov = get_team_fov_cells(agents, fov_angle=90, max_range=3)

        # Should combine both agents' FOV
        assert (5, 5) in team_fov, "First agent's cell should be visible"
        assert (10, 10) in team_fov, "Second agent's cell should be visible"
        print("  get_team_fov_cells: OK")

    def test_fov_with_building_blocking():
        """Test that buildings block FOV."""
        from .terrain import TerrainGrid, TerrainType

        terrain = TerrainGrid(20, 20)
        terrain.set(6, 5, TerrainType.BUILDING)  # Place building in front

        agent_pos = (5.0, 5.0)
        cells = get_fov_cells(agent_pos, 0.0, 90, 5, terrain_grid=terrain)

        # Cells before building should be visible
        assert (5, 5) in cells, "Agent's cell should be visible"

        # Cells after building should NOT be visible (blocked)
        assert (7, 5) not in cells, "Cell behind building should not be visible"
        print("  FOV building blocking: OK")

    # Run all tests
    print("Running fov.py self-tests...")
    try:
        test_normalize_angle()
        test_angle_difference()
        test_is_point_in_fov_cone()
        test_get_fov_cells()
        test_get_team_fov_cells()
        test_fov_with_building_blocking()
        print("All fov.py self-tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
