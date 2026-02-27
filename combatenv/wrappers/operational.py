"""
Operational wrapper for unit management, boids steering, and waypoints.

This wrapper handles the operational level of the tactical simulation:
- Unit spawning and management
- Boids flocking behavior for unit movement
- Waypoint setting and path following
- Unit stance and formation control
- Dispatch system for releasing units from reserve

Usage:
    from combatenv.wrappers import OperationalWrapper

    env = TacticalCombatEnv(render_mode="human")
    env = OperationalWrapper(env)

    # Set waypoint for unit 0
    env.set_unit_waypoint(0, "blue", 32.0, 32.0)

    # Dispatch a unit from reserve
    env.dispatch_unit(0, "blue", 32.0, 32.0)
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym

import math

from combatenv.config import (
    AGENT_ROTATION_SPEED,
    UNIT_COHESION_RADIUS,
    COHESION_ACCURACY_BONUS,
    UNIT_MAX_RADIUS,
)
from combatenv.unit import (
    Unit,
    UnitStance,
    get_unit_for_agent,
)
from combatenv.boids import calculate_stance_steering, steering_to_orientation
from combatenv.agent import Agent
from .base_wrapper import BaseWrapper


class OperationalWrapper(BaseWrapper):
    """
    Wrapper for operational-level unit management.

    Manages units, boids steering, waypoints, and formations on top of
    the tactical environment.

    Attributes:
        blue_units: List of blue team units
        red_units: List of red team units
        use_units: Whether units are enabled
    """

    def __init__(
        self,
        env,
        use_units: bool = True,
        auto_dispatch_controlled: bool = True,
    ):
        """
        Initialize the operational wrapper.

        Args:
            env: Base environment to wrap
            use_units: Whether to use unit system
            auto_dispatch_controlled: Auto-dispatch controlled agent's unit on reset
        """
        super().__init__(env)

        self.use_units = use_units
        self.auto_dispatch_controlled = auto_dispatch_controlled

        # Unit tracking (synced with base env)
        self._blue_units: List[Unit] = []
        self._red_units: List[Unit] = []

    @property
    def blue_units(self) -> List[Unit]:
        """Get blue team units (from wrapper chain or base env)."""
        # First check wrapper chain (e.g., UnitWrapper)
        units = self._find_units_in_chain('blue_units')
        if units:
            return units
        return self._blue_units

    @property
    def red_units(self) -> List[Unit]:
        """Get red team units (from wrapper chain or base env)."""
        # First check wrapper chain (e.g., UnitWrapper)
        units = self._find_units_in_chain('red_units')
        if units:
            return units
        return self._red_units

    def _find_units_in_chain(self, attr_name: str) -> List[Unit]:
        """Find units by walking up the wrapper chain."""
        env = self.env
        while env is not None:
            units = getattr(env, attr_name, None)
            if units is not None and len(units) > 0:
                return units
            env = getattr(env, 'env', None)
        return []

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Reset and optionally spawn units.

        If use_units is True, units are spawned and agents assigned to them.
        The controlled agent's unit is auto-dispatched if configured.
        """
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        # Units are now accessed via properties that search the wrapper chain
        # No need to sync here - the properties handle it dynamically

        return obs, info

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        """
        Step and apply boids steering to units.

        Boids steering is applied before the base step to ensure agents
        have updated movement directions.
        """
        # Apply boids steering before base step
        if self.use_units:
            self._update_unit_movement(dt=1.0/60.0)  # Assume 60 FPS

        # Call base step
        return self.env.step(action)

    def _update_unit_movement(self, dt: float) -> None:
        """
        Apply boids steering to agents in units with active waypoints.

        Args:
            dt: Delta time in seconds
        """
        # Use attribute forwarding through self (not unwrapped) to access agents
        alive_agents = getattr(self, 'alive_agents', [])
        terrain_grid = getattr(self, 'terrain_grid', None)
        red_agents = getattr(self, 'red_agents', [])
        blue_agents = getattr(self, 'blue_agents', [])

        all_units = self.blue_units + self.red_units

        for agent in alive_agents:
            # Skip agents that are stuck
            if agent.is_stuck:
                continue

            # Skip agents not following unit
            if not agent.following_unit:
                continue

            # Find agent's unit
            unit = get_unit_for_agent(agent, all_units)
            if unit is None:
                continue

            # Get target waypoint - prefer intermediate waypoint, fall back to goal
            target_waypoint = unit.waypoint
            if target_waypoint is None:
                target_waypoint = getattr(unit, 'goal_waypoint', None)
            if target_waypoint is None:
                continue

            # Skip units in reserve
            if unit.in_reserve:
                continue

            # Get enemies for stance-aware steering
            enemies = red_agents if agent.team == "blue" else blue_agents

            # Calculate stance-aware boids steering
            steering = calculate_stance_steering(agent, unit, enemies=enemies)
            target_orientation = steering_to_orientation(steering)

            if target_orientation is not None:
                # Smoothly rotate toward steering direction
                angle_diff = (target_orientation - agent.orientation + 180) % 360 - 180
                max_rotation = AGENT_ROTATION_SPEED * dt

                if abs(angle_diff) > max_rotation:
                    if angle_diff > 0:
                        agent.orientation = (agent.orientation + max_rotation) % 360
                    else:
                        agent.orientation = (agent.orientation - max_rotation) % 360
                else:
                    agent.orientation = target_orientation

                # Move forward in steering direction
                # Don't block on agent collision - use soft collision (push apart after)
                agent.move_forward(dt=dt, other_agents=None, terrain_grid=terrain_grid)
                agent.wander_direction = 1  # Mark as moving

                # Enforce unit radius constraint (hard leash)
                self._enforce_unit_radius(agent, unit)

                # Soft collision - push overlapping agents apart
                self._push_overlapping_agents(agent, alive_agents)
            else:
                agent.wander_direction = 0  # Not moving

    def _enforce_unit_radius(self, agent, unit: Unit) -> None:
        """
        Enforce the unit radius constraint (hard leash).

        If an agent moves beyond UNIT_MAX_RADIUS from the unit centroid,
        it is pulled back to the boundary. This is a hard constraint that
        ensures agents stay within tactical range of their unit.

        Args:
            agent: Agent to check/adjust
            unit: Unit the agent belongs to
        """
        terrain_grid = getattr(self, 'terrain_grid', None)
        centroid = unit.centroid
        agent_x, agent_y = agent.position
        dx = agent_x - centroid[0]
        dy = agent_y - centroid[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > UNIT_MAX_RADIUS:
            # Agent is outside allowed radius, pull back to boundary
            scale = UNIT_MAX_RADIUS / dist
            new_x = centroid[0] + dx * scale
            new_y = centroid[1] + dy * scale

            # Check if position is safe (not building or fire)
            if not self._is_safe_position(new_x, new_y, terrain_grid):
                return  # Don't push into hazard

            agent.position = (new_x, new_y)

    def _is_safe_position(self, x: float, y: float, terrain_grid) -> bool:
        """Check if position is safe (walkable and not fire)."""
        if terrain_grid is None:
            return True
        cell_x, cell_y = int(x), int(y)
        if not terrain_grid.is_walkable(cell_x, cell_y):
            return False
        # Also check for fire terrain
        from combatenv.terrain import TerrainType
        terrain = terrain_grid.get(cell_x, cell_y)
        if terrain == TerrainType.FIRE:
            return False
        return True

    def _push_overlapping_agents(self, agent, all_agents: list) -> None:
        """
        Soft collision - push overlapping agents apart.

        If two agents overlap, push them away from each other by half
        the overlap distance each.

        Args:
            agent: Agent that just moved
            all_agents: List of all alive agents to check against
        """
        from combatenv.config import AGENT_COLLISION_RADIUS, GRID_SIZE

        terrain_grid = getattr(self, 'terrain_grid', None)
        ax, ay = agent.position

        for other in all_agents:
            if other is agent or not other.is_alive:
                continue

            ox, oy = other.position
            dx = ax - ox
            dy = ay - oy
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < AGENT_COLLISION_RADIUS and dist > 0.01:
                # Overlap detected - push apart
                overlap = AGENT_COLLISION_RADIUS - dist
                push = overlap * 0.5  # Each agent moves half

                # Normalize direction
                nx = dx / dist
                ny = dy / dist

                # Push this agent away (if safe position)
                new_ax = ax + nx * push
                new_ay = ay + ny * push
                new_ax = max(0.5, min(GRID_SIZE - 0.5, new_ax))
                new_ay = max(0.5, min(GRID_SIZE - 0.5, new_ay))
                if self._is_safe_position(new_ax, new_ay, terrain_grid):
                    agent.position = (new_ax, new_ay)

                # Push other agent away (if safe position)
                new_ox = ox - nx * push
                new_oy = oy - ny * push
                new_ox = max(0.5, min(GRID_SIZE - 0.5, new_ox))
                new_oy = max(0.5, min(GRID_SIZE - 0.5, new_oy))
                if self._is_safe_position(new_ox, new_oy, terrain_grid):
                    other.position = (new_ox, new_oy)

                # Update our position for next check
                ax, ay = agent.position

    # ==================== Unit Management API ====================

    def set_unit_waypoint(self, unit_id: int, team: str, x: float, y: float) -> bool:
        """
        Set a waypoint for a specific unit.

        Args:
            unit_id: The unit's ID (0 to NUM_UNITS_PER_TEAM-1)
            team: Team affiliation ("blue" or "red")
            x: X coordinate in grid space
            y: Y coordinate in grid space

        Returns:
            True if waypoint was set, False if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                unit.set_waypoint(x, y)
                return True
        return False

    def add_unit_waypoint(self, unit_id: int, team: str, x: float, y: float) -> bool:
        """
        Append a waypoint to a unit's waypoint sequence.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")
            x: X coordinate in grid space
            y: Y coordinate in grid space

        Returns:
            True if waypoint was added, False if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                unit.add_waypoint(x, y)
                return True
        return False

    def advance_unit_waypoint(self, unit_id: int, team: str) -> bool:
        """
        Advance a unit to its next waypoint in the sequence.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            True if advanced to next waypoint, False if at end or not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                return unit.advance_waypoint()
        return False

    def clear_unit_waypoint(self, unit_id: int, team: str) -> bool:
        """
        Clear the waypoint for a specific unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            True if waypoint was cleared, False if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                unit.clear_waypoint()
                return True
        return False

    def clear_all_waypoints(self, team: Optional[str] = None) -> None:
        """
        Clear all waypoints for a team or both teams.

        Args:
            team: "blue", "red", or None for both teams
        """
        if team is None or team == "blue":
            for unit in self.blue_units:
                unit.clear_waypoint()
        if team is None or team == "red":
            for unit in self.red_units:
                unit.clear_waypoint()

    def get_unit_waypoints(self, unit_id: int, team: str) -> List[Tuple[float, float]]:
        """
        Get all waypoints for a unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            List of (x, y) waypoint positions, empty list if not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                return list(unit.waypoints)
        return []

    # ==================== Dispatch System ====================

    def dispatch_unit(self, unit_id: int, team: str, x: float, y: float) -> bool:
        """
        Dispatch a unit from reserve - releases it and sets its waypoint.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")
            x: X coordinate for waypoint
            y: Y coordinate for waypoint

        Returns:
            True if dispatched, False if not found or already dispatched
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                if not unit.in_reserve:
                    return False  # Already dispatched
                unit.dispatch((x, y))
                return True
        return False

    def get_next_reserve_unit(self, team: str) -> Optional[int]:
        """
        Get the ID of the next unit in reserve for a team.

        Args:
            team: Team affiliation ("blue" or "red")

        Returns:
            Unit ID of next reserve unit, or None if all dispatched
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.in_reserve:
                return unit.id
        return None

    def dispatch_all_units(self, team: str, waypoint: Tuple[float, float]) -> int:
        """
        Dispatch all units in reserve for a team.

        Args:
            team: Team affiliation ("blue" or "red")
            waypoint: (x, y) waypoint for all units

        Returns:
            Number of units dispatched
        """
        units = self.blue_units if team == "blue" else self.red_units
        count = 0

        for unit in units:
            if unit.in_reserve:
                unit.dispatch(waypoint)
                count += 1

        return count

    # ==================== Stance Control ====================

    def set_unit_stance(self, unit_id: int, team: str, stance: UnitStance) -> bool:
        """
        Set the stance for a unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")
            stance: UnitStance (AGGRESSIVE, DEFENSIVE, PATROL)

        Returns:
            True if stance was set, False if unit not found
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                unit.stance = stance
                return True
        return False

    def get_unit_stance(self, unit_id: int, team: str) -> Optional[UnitStance]:
        """
        Get the current stance for a unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            UnitStance if unit found, None otherwise
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                return unit.stance
        return None

    def set_all_units_stance(self, team: str, stance: UnitStance) -> int:
        """
        Set stance for all units in a team.

        Args:
            team: Team affiliation ("blue" or "red")
            stance: UnitStance to set

        Returns:
            Number of units updated
        """
        units = self.blue_units if team == "blue" else self.red_units
        count = 0

        for unit in units:
            unit.stance = stance
            count += 1

        return count

    # ==================== Unit Queries ====================

    def get_unit_by_id(self, unit_id: int, team: str) -> Optional[Unit]:
        """
        Get a unit by its ID and team.

        Args:
            unit_id: The unit's ID
            team: Team affiliation ("blue" or "red")

        Returns:
            Unit if found, None otherwise
        """
        units = self.blue_units if team == "blue" else self.red_units

        for unit in units:
            if unit.id == unit_id:
                return unit
        return None

    def get_unit_for_agent(self, agent: Agent) -> Optional[Unit]:
        """
        Get the unit that an agent belongs to.

        Args:
            agent: The agent to look up

        Returns:
            Unit if found, None otherwise
        """
        all_units = self.blue_units + self.red_units
        return get_unit_for_agent(agent, all_units)

    def get_units_at_waypoint(self, team: str) -> List[Unit]:
        """
        Get units whose centroid has reached their waypoint.

        Uses the unit's built-in arrival distance threshold.

        Args:
            team: Team affiliation ("blue" or "red")

        Returns:
            List of units at their waypoint
        """
        units = self.blue_units if team == "blue" else self.red_units
        result = []

        for unit in units:
            if unit.is_at_waypoint():
                result.append(unit)

        return result

    # ==================== Cohesion Checks ====================

    def is_agent_cohesive(self, agent: Agent) -> bool:
        """
        Check if an agent is within cohesion radius of its unit's centroid.

        Args:
            agent: The agent to check

        Returns:
            True if agent is within UNIT_COHESION_RADIUS of centroid
        """
        if not self.use_units or agent.unit_id is None:
            return False

        unit = self.get_unit_for_agent(agent)
        if unit is None:
            return False

        centroid = unit.centroid
        dist_sq = (agent.position[0] - centroid[0])**2 + (agent.position[1] - centroid[1])**2
        return dist_sq <= UNIT_COHESION_RADIUS**2

    def get_accuracy_with_cohesion(self, agent: Agent, base_accuracy: float) -> float:
        """
        Get accuracy with cohesion bonus applied if applicable.

        Args:
            agent: The shooting agent
            base_accuracy: Base accuracy value

        Returns:
            Accuracy with cohesion bonus if agent is cohesive
        """
        if self.is_agent_cohesive(agent):
            return base_accuracy * (1.0 + COHESION_ACCURACY_BONUS)
        return base_accuracy

    def get_unit_cohesion_score(self, unit_id: int, team: str, agent: Optional[Agent] = None) -> float:
        """
        Get the cohesion score for an agent relative to its unit.

        If no agent is specified, returns average cohesion for all agents.

        Args:
            unit_id: The unit's ID
            team: Team affiliation
            agent: Specific agent to check, or None for average

        Returns:
            Cohesion score (0.0-1.0), or 0.0 if unit not found
        """
        unit = self.get_unit_by_id(unit_id, team)
        if unit is None:
            return 0.0

        if agent is not None:
            return unit.get_cohesion_score(agent)

        # Average cohesion for all alive agents
        alive = unit.alive_agents
        if not alive:
            return 0.0
        total = sum(unit.get_cohesion_score(a) for a in alive)
        return total / len(alive)

    # ==================== Statistics ====================

    def get_unit_stats(self, unit_id: int, team: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a unit.

        Args:
            unit_id: The unit's ID
            team: Team affiliation

        Returns:
            Dict with unit statistics
        """
        unit = self.get_unit_by_id(unit_id, team)
        if unit is None:
            return {}

        # Calculate average cohesion across alive agents
        alive = unit.alive_agents
        avg_cohesion = 0.0
        if alive:
            total = sum(unit.get_cohesion_score(a) for a in alive)
            avg_cohesion = total / len(alive)

        return {
            'id': unit.id,
            'team': team,
            'alive_agents': len(alive),
            'total_agents': len(unit.agents),
            'centroid': unit.centroid,
            'waypoint': unit.waypoint,
            'waypoints': list(unit.waypoints),
            'stance': unit.stance.name,
            'in_reserve': unit.in_reserve,
            'is_eliminated': unit.is_eliminated,
            'avg_cohesion': avg_cohesion,
            'formation_spread': unit.get_formation_spread(),
        }

    def get_all_unit_stats(self, team: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get statistics for all units.

        Args:
            team: Team to filter ("blue", "red", or None for all)

        Returns:
            List of unit statistics dicts
        """
        result = []

        if team is None or team == "blue":
            for unit in self.blue_units:
                result.append(self.get_unit_stats(unit.id, "blue"))

        if team is None or team == "red":
            for unit in self.red_units:
                result.append(self.get_unit_stats(unit.id, "red"))

        return result
