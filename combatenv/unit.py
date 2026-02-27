"""
Unit class for organizing agents into cohesive squads.

This module provides the Unit abstraction layer for grouping agents into
tactical squads that move and fight together. Units support:
- Centroid calculation for group center of mass
- Cohesion scoring to measure how grouped agents are
- Waypoint-based movement commands
- Formation state tracking

Example:
    >>> from combatenv.unit import Unit, spawn_units_for_team
    >>> blue_units = spawn_units_for_team("blue", num_units=10, agents_per_unit=10)
    >>> unit = blue_units[0]
    >>> print(f"Unit {unit.id} centroid: {unit.centroid}")
"""

import math
import random
from typing import Tuple, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from .config import (
    GRID_SIZE,
    AGENT_SPAWN_SPACING,
    AGENT_MAX_HEALTH,
    NUM_UNITS_PER_TEAM,
    AGENTS_PER_UNIT,
    UNIT_SPAWN_SPREAD,
    UNIT_COHESION_RADIUS,
    UNIT_WAYPOINT_ARRIVAL_DIST,
    TACTICAL_CELLS_PER_OPERATIONAL,
    UnitStance,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .terrain import TerrainGrid

# Import TeamType from agent module
from .agent import TeamType


@dataclass
class Unit:
    """
    A squad of agents that move and fight together.

    Attributes:
        id: Unique identifier for the unit
        team: Team affiliation ("blue" or "red")
        agents: List of Agent instances in this unit
        waypoints: List of waypoint positions (intermediate movement targets)
        current_waypoint_index: Index of current waypoint in sequence
        stance: Movement behavior mode (AGGRESSIVE, DEFENSIVE, PATROL)
        in_reserve: If True, unit is hidden and inactive (not moving/fighting)
        goal_waypoint: Final destination set by strategic level or mouse (for reward)
    """
    id: int
    team: TeamType
    agents: List['Agent'] = field(default_factory=list)
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    current_waypoint_index: int = 0
    stance: UnitStance = UnitStance.PATROL
    in_reserve: bool = True  # Units start in reserve until dispatched
    goal_waypoint: Optional[Tuple[float, float]] = None  # Final destination for reward

    @property
    def waypoint(self) -> Optional[Tuple[float, float]]:
        """
        Get the current active waypoint (for backward compatibility).

        Returns:
            Current waypoint position or None if no waypoints
        """
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_waypoint_index]

    @property
    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        """Alias for waypoint property."""
        return self.waypoint

    @property
    def centroid(self) -> Tuple[float, float]:
        """
        Calculate the unit's center of mass based on alive agents.

        Returns:
            (x, y) centroid position, or (0, 0) if no alive agents
        """
        alive = self.alive_agents
        if not alive:
            return (0.0, 0.0)

        sum_x = sum(a.position[0] for a in alive)
        sum_y = sum(a.position[1] for a in alive)
        count = len(alive)

        return (sum_x / count, sum_y / count)

    @property
    def alive_agents(self) -> List['Agent']:
        """Return only living agents in the unit."""
        return [a for a in self.agents if a.is_alive]

    @property
    def alive_count(self) -> int:
        """Return count of living agents in the unit."""
        return len(self.alive_agents)

    @property
    def average_heading(self) -> float:
        """
        Calculate average orientation of all alive agents.

        Uses circular mean to properly average angles.

        Returns:
            Average heading in degrees (0-360), or 0 if no alive agents
        """
        alive = self.alive_agents
        if not alive:
            return 0.0

        # Use circular mean for angles
        sum_sin = sum(math.sin(math.radians(a.orientation)) for a in alive)
        sum_cos = sum(math.cos(math.radians(a.orientation)) for a in alive)

        avg_rad = math.atan2(sum_sin, sum_cos)
        return math.degrees(avg_rad) % 360

    @property
    def is_eliminated(self) -> bool:
        """True if all agents in unit are dead."""
        return all(not a.is_alive for a in self.agents)

    def get_cohesion_score(self, agent: 'Agent') -> float:
        """
        Measure how close an agent is to the unit centroid.

        Args:
            agent: The agent to measure cohesion for

        Returns:
            Score from 0.0 (far from centroid) to 1.0 (at centroid)
        """
        if agent not in self.agents:
            return 0.0

        centroid = self.centroid
        dist = math.sqrt(
            (agent.position[0] - centroid[0])**2 +
            (agent.position[1] - centroid[1])**2
        )

        # Score decreases linearly with distance up to cohesion radius
        if dist >= UNIT_COHESION_RADIUS:
            return 0.0

        return 1.0 - (dist / UNIT_COHESION_RADIUS)

    def get_formation_spread(self) -> float:
        """
        Calculate how spread out the unit is.

        Returns:
            Average distance from centroid for all alive agents
        """
        alive = self.alive_agents
        if len(alive) <= 1:
            return 0.0

        centroid = self.centroid
        total_dist = sum(
            math.sqrt(
                (a.position[0] - centroid[0])**2 +
                (a.position[1] - centroid[1])**2
            )
            for a in alive
        )

        return total_dist / len(alive)

    def is_at_waypoint(self) -> bool:
        """
        Check if unit centroid has reached the waypoint.

        Returns:
            True if waypoint is set and centroid is within arrival distance
        """
        if self.waypoint is None:
            return True

        centroid = self.centroid
        dist = math.sqrt(
            (centroid[0] - self.waypoint[0])**2 +
            (centroid[1] - self.waypoint[1])**2
        )

        return dist <= UNIT_WAYPOINT_ARRIVAL_DIST

    def dispatch(self, waypoint: Tuple[float, float]) -> None:
        """
        Release unit from reserve and set its waypoint.

        Teleports all agents randomly within their team's spawn strategic cell,
        letting RL learn optimal formation through rewards.

        Args:
            waypoint: Target position (x, y) for the unit to move to
        """
        self.in_reserve = False
        self.set_waypoint(waypoint[0], waypoint[1])

        # Define spawn strategic cell based on team
        # Blue: top-left cell (0,0) = tactical coords (0-7, 0-7)
        # Red: bottom-right cell (7,7) = tactical coords (56-63, 56-63)
        cell_size = TACTICAL_CELLS_PER_OPERATIONAL
        margin = 0.5  # Keep agents slightly inside cell boundaries

        if self.team == "blue":
            cell_min_x, cell_min_y = margin, margin
            cell_max_x = cell_size - margin
            cell_max_y = cell_size - margin
        else:
            cell_min_x = GRID_SIZE - cell_size + margin
            cell_min_y = GRID_SIZE - cell_size + margin
            cell_max_x = GRID_SIZE - margin
            cell_max_y = GRID_SIZE - margin

        # Randomly position agents within the strategic cell
        # Try to avoid overlapping with minimum spacing
        placed_positions = []
        max_attempts = 50

        for agent in self.agents:
            for attempt in range(max_attempts):
                # Random position within strategic cell
                x = random.uniform(cell_min_x, cell_max_x)
                y = random.uniform(cell_min_y, cell_max_y)

                # Check spacing with already placed agents
                too_close = False
                for px, py in placed_positions:
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    if dist < AGENT_SPAWN_SPACING * 0.8:  # Slightly relaxed spacing
                        too_close = True
                        break

                if not too_close or attempt == max_attempts - 1:
                    agent.position = (x, y)
                    placed_positions.append((x, y))
                    break

    def dispatch_to(self, spawn_position: Tuple[float, float], waypoint: Tuple[float, float]) -> None:
        """
        Release unit from reserve, spawn at specific position, and set waypoint.

        Args:
            spawn_position: Center position (x, y) to spawn agents around
            waypoint: Target position (x, y) for the unit to move to
        """
        self.in_reserve = False
        self.set_waypoint(waypoint[0], waypoint[1])

        # Spawn agents around the specified position
        spawn_radius = UNIT_SPAWN_SPREAD
        margin = 0.5

        placed_positions = []
        max_attempts = 50

        for agent in self.agents:
            for attempt in range(max_attempts):
                # Random position within spawn radius of center
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(0, spawn_radius)
                x = spawn_position[0] + dist * math.cos(angle)
                y = spawn_position[1] + dist * math.sin(angle)

                # Clamp to grid bounds
                x = max(margin, min(GRID_SIZE - margin, x))
                y = max(margin, min(GRID_SIZE - margin, y))

                # Check spacing with already placed agents
                too_close = False
                for px, py in placed_positions:
                    if math.sqrt((x - px)**2 + (y - py)**2) < AGENT_SPAWN_SPACING * 0.8:
                        too_close = True
                        break

                if not too_close or attempt == max_attempts - 1:
                    agent.position = (x, y)
                    agent.health = AGENT_MAX_HEALTH  # Revive agent
                    placed_positions.append((x, y))
                    break

    def clear_waypoints(self) -> None:
        """Clear all waypoints."""
        self.waypoints = []
        self.current_waypoint_index = 0

    def clear_waypoint(self) -> None:
        """Clear all waypoints (alias for clear_waypoints)."""
        self.clear_waypoints()

    def move_direction(self, dx: float, dy: float) -> None:
        """
        Move all alive agents directly in a direction.

        This bypasses waypoint-based movement and directly updates agent positions.
        Used for direct movement control in RL training.

        Args:
            dx: Movement in x direction (grid cells)
            dy: Movement in y direction (grid cells)
        """
        for agent in self.alive_agents:
            new_x = agent.position[0] + dx
            new_y = agent.position[1] + dy

            # Clamp to grid bounds
            new_x = max(0.5, min(GRID_SIZE - 0.5, new_x))
            new_y = max(0.5, min(GRID_SIZE - 0.5, new_y))

            agent.position = (new_x, new_y)

    def set_waypoint(self, x: float, y: float) -> None:
        """
        Set a single waypoint for the unit (clears existing waypoints).

        Args:
            x: X coordinate in grid space
            y: Y coordinate in grid space
        """
        # Clamp to grid bounds
        x = max(0.5, min(GRID_SIZE - 0.5, x))
        y = max(0.5, min(GRID_SIZE - 0.5, y))
        self.waypoints = [(x, y)]
        self.current_waypoint_index = 0

    def set_goal(self, x: float, y: float) -> None:
        """
        Set the goal waypoint (final destination for reward calculation).

        This is separate from intermediate waypoints used for movement.
        Set by strategic level or mouse click.

        Args:
            x: X coordinate in grid space
            y: Y coordinate in grid space
        """
        x = max(0.5, min(GRID_SIZE - 0.5, x))
        y = max(0.5, min(GRID_SIZE - 0.5, y))
        self.goal_waypoint = (x, y)

    def clear_goal(self) -> None:
        """Clear the goal waypoint."""
        self.goal_waypoint = None

    def add_waypoint(self, x: float, y: float) -> None:
        """
        Append a waypoint to the sequence.

        Args:
            x: X coordinate in grid space
            y: Y coordinate in grid space
        """
        # Clamp to grid bounds
        x = max(0.5, min(GRID_SIZE - 0.5, x))
        y = max(0.5, min(GRID_SIZE - 0.5, y))
        self.waypoints.append((x, y))

    def advance_waypoint(self) -> bool:
        """
        Move to the next waypoint in the sequence.

        Returns:
            True if advanced to next waypoint, False if at end of sequence
        """
        if self.current_waypoint_index < len(self.waypoints) - 1:
            self.current_waypoint_index += 1
            return True
        return False

    def has_more_waypoints(self) -> bool:
        """Check if there are more waypoints after the current one."""
        return self.current_waypoint_index < len(self.waypoints) - 1

    def get_waypoint_progress(self) -> Tuple[int, int]:
        """
        Get current waypoint progress.

        Returns:
            Tuple of (current_index, total_waypoints)
        """
        return (self.current_waypoint_index, len(self.waypoints))

    def get_nearby_squadmates(self, agent: 'Agent', radius: float = 3.0) -> List['Agent']:
        """
        Get squadmates within a radius of the given agent.

        Args:
            agent: The reference agent
            radius: Search radius in grid cells

        Returns:
            List of nearby alive squadmates (excluding the agent itself)
        """
        nearby = []
        for other in self.alive_agents:
            if other is agent:
                continue

            dist = math.sqrt(
                (other.position[0] - agent.position[0])**2 +
                (other.position[1] - agent.position[1])**2
            )
            if dist <= radius:
                nearby.append(other)

        return nearby


def spawn_unit(
    team: TeamType,
    unit_id: int,
    num_agents: int = AGENTS_PER_UNIT,
    center: Tuple[float, float] = None,
    spread: float = UNIT_SPAWN_SPREAD,
    terrain_grid: 'TerrainGrid' = None
) -> Unit:
    """
    Spawn a cohesive unit of agents around a center point.

    Args:
        team: Team affiliation ("blue" or "red")
        unit_id: Unique identifier for this unit
        num_agents: Number of agents to spawn in the unit
        center: Center position for the unit spawn (random if None)
        spread: Maximum radius for agent positions around center
        terrain_grid: TerrainGrid for terrain collision checks

    Returns:
        Unit containing the spawned agents
    """
    from .agent import Agent
    from .terrain import TerrainType

    # Determine spawn area based on team
    margin = 2.0
    if team == "blue":
        x_min, x_max = margin, GRID_SIZE // 2 - margin
        y_min, y_max = margin, GRID_SIZE // 2 - margin
    else:
        x_min, x_max = GRID_SIZE // 2 + margin, GRID_SIZE - margin
        y_min, y_max = GRID_SIZE // 2 + margin, GRID_SIZE - margin

    # Generate center if not provided
    if center is None:
        center = (
            random.uniform(x_min + spread, x_max - spread),
            random.uniform(y_min + spread, y_max - spread)
        )

    agents = []
    attempts = 0
    max_attempts = 2000
    current_spread = spread

    while len(agents) < num_agents and attempts < max_attempts:
        attempts += 1

        # Progressively expand search radius if struggling to find positions
        if attempts > 0 and attempts % 500 == 0:
            current_spread = min(current_spread * 1.5, max(x_max - x_min, y_max - y_min) / 2)

        # Random position around center
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, current_spread)
        x = center[0] + dist * math.cos(angle)
        y = center[1] + dist * math.sin(angle)

        # Clamp to spawn bounds
        x = max(x_min, min(x_max, x))
        y = max(y_min, min(y_max, y))

        # Check terrain - must be walkable (not a building)
        if terrain_grid:
            cell_x, cell_y = int(x), int(y)
            if not terrain_grid.is_walkable(cell_x, cell_y):
                continue

        # Check spacing with existing agents
        too_close = False
        for existing in agents:
            d = math.sqrt(
                (x - existing.position[0])**2 +
                (y - existing.position[1])**2
            )
            if d < AGENT_SPAWN_SPACING:
                too_close = True
                break

        if not too_close:
            agent = Agent(
                position=(x, y),
                orientation=random.uniform(0, 360),
                team=team,
                unit_id=unit_id,
                following_unit=True
            )
            agents.append(agent)

    if len(agents) < num_agents:
        raise RuntimeError(
            f"Failed to spawn {num_agents} agents for unit {unit_id} "
            f"(only spawned {len(agents)}). Reduce spread or check terrain."
        )

    unit = Unit(id=unit_id, team=team, agents=agents)

    # Units start in reserve - move agents off-screen
    for agent in agents:
        agent.position = (-100.0, -100.0)

    return unit


def spawn_units_for_team(
    team: TeamType,
    num_units: int = NUM_UNITS_PER_TEAM,
    agents_per_unit: int = AGENTS_PER_UNIT,
    terrain_grid: 'TerrainGrid' = None
) -> List[Unit]:
    """
    Spawn all units for a team distributed across their quadrant.

    Args:
        team: Team affiliation ("blue" or "red")
        num_units: Number of units to spawn
        agents_per_unit: Agents per unit
        terrain_grid: TerrainGrid for terrain collision checks

    Returns:
        List of Unit instances
    """
    units = []

    # Calculate spawn area
    margin = 2.0 + UNIT_SPAWN_SPREAD
    if team == "blue":
        # Blue spawns at top-left
        x_min, x_max = margin, GRID_SIZE // 2 - margin
        y_min, y_max = margin, GRID_SIZE // 2 - margin
        # Corner position for single unit
        corner_x, corner_y = margin + UNIT_SPAWN_SPREAD, margin + UNIT_SPAWN_SPREAD
    else:
        # Red spawns at bottom-right
        x_min, x_max = GRID_SIZE // 2 + margin, GRID_SIZE - margin
        y_min, y_max = GRID_SIZE // 2 + margin, GRID_SIZE - margin
        # Corner position for single unit (bottom-right)
        corner_x, corner_y = GRID_SIZE - margin, GRID_SIZE - margin

    # Special case: single unit spawns at corner
    if num_units == 1:
        unit = spawn_unit(
            team=team,
            unit_id=0,
            num_agents=agents_per_unit,
            center=(corner_x, corner_y),
            terrain_grid=terrain_grid
        )
        return [unit]

    # Calculate grid dimensions for unit placement
    grid_cols = int(math.ceil(math.sqrt(num_units)))
    grid_rows = int(math.ceil(num_units / grid_cols))

    x_step = (x_max - x_min) / max(1, grid_cols)
    y_step = (y_max - y_min) / max(1, grid_rows)

    unit_id = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if unit_id >= num_units:
                break

            # Calculate center for this unit
            center_x = x_min + (col + 0.5) * x_step
            center_y = y_min + (row + 0.5) * y_step

            # Add some randomness to avoid perfect grid
            center_x += random.uniform(-x_step * 0.2, x_step * 0.2)
            center_y += random.uniform(-y_step * 0.2, y_step * 0.2)

            unit = spawn_unit(
                team=team,
                unit_id=unit_id,
                num_agents=agents_per_unit,
                center=(center_x, center_y),
                terrain_grid=terrain_grid
            )
            units.append(unit)
            unit_id += 1

    return units


def spawn_all_units(
    num_units_per_team: int = NUM_UNITS_PER_TEAM,
    agents_per_unit: int = AGENTS_PER_UNIT,
    terrain_grid: 'TerrainGrid' = None
) -> Tuple[List[Unit], List[Unit]]:
    """
    Spawn units for both teams.

    Args:
        num_units_per_team: Number of units per team
        agents_per_unit: Agents per unit
        terrain_grid: TerrainGrid for terrain collision checks

    Returns:
        Tuple of (blue_units, red_units)
    """
    blue_units = spawn_units_for_team(
        "blue",
        num_units=num_units_per_team,
        agents_per_unit=agents_per_unit,
        terrain_grid=terrain_grid
    )
    red_units = spawn_units_for_team(
        "red",
        num_units=num_units_per_team,
        agents_per_unit=agents_per_unit,
        terrain_grid=terrain_grid
    )

    return blue_units, red_units


def get_all_agents_from_units(units: List[Unit]) -> List['Agent']:
    """
    Extract all agents from a list of units.

    Args:
        units: List of Unit instances

    Returns:
        Flat list of all Agent instances
    """
    agents = []
    for unit in units:
        agents.extend(unit.agents)
    return agents


def get_unit_for_agent(agent: 'Agent', units: List[Unit]) -> Optional[Unit]:
    """
    Find the unit that contains a given agent.

    Args:
        agent: The agent to find
        units: List of units to search

    Returns:
        The Unit containing the agent, or None if not found
    """
    if agent.unit_id is not None:
        for unit in units:
            if unit.id == agent.unit_id:
                return unit
    return None
