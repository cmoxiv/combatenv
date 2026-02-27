"""
Boids flocking algorithm for unit cohesion.

This module implements Reynolds' flocking algorithm with four steering behaviors:
1. Cohesion: Steer toward the unit centroid
2. Separation: Avoid crowding nearby squadmates
3. Alignment: Match heading of nearby squadmates
4. Waypoint: Steer toward unit waypoint

The combined steering force influences agent movement direction while
maintaining agent autonomy for combat and individual decisions.

Stance-based behavior modifications:
- AGGRESSIVE: Waypoint force targets nearest enemy instead of waypoint
- DEFENSIVE: Stronger cohesion, reduced waypoint force
- PATROL: Normal boids behavior

Example:
    >>> from combatenv.boids import calculate_boids_steering
    >>> steering = calculate_boids_steering(agent, unit)
    >>> print(f"Steering force: ({steering[0]:.2f}, {steering[1]:.2f})")
"""

import math
from typing import Tuple, Dict, Optional, List, TYPE_CHECKING

from .config import (
    BOIDS_COHESION_WEIGHT,
    BOIDS_SEPARATION_WEIGHT,
    BOIDS_ALIGNMENT_WEIGHT,
    BOIDS_WAYPOINT_WEIGHT,
    BOIDS_MAX_FORCE,
    BOIDS_SEPARATION_RADIUS,
    UnitStance,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .unit import Unit


def calculate_cohesion_force(
    agent: 'Agent',
    unit: 'Unit',
    weight: float = BOIDS_COHESION_WEIGHT
) -> Tuple[float, float]:
    """
    Steer toward the unit centroid.

    This force pulls agents toward the center of their squad,
    encouraging group cohesion.

    Args:
        agent: The agent to calculate steering for
        unit: The unit the agent belongs to
        weight: Multiplier for the cohesion force

    Returns:
        (dx, dy) force vector pointing toward centroid
    """
    centroid = unit.centroid

    # Vector from agent to centroid
    dx = centroid[0] - agent.position[0]
    dy = centroid[1] - agent.position[1]

    # Normalize and apply weight
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 0.001:  # Already at centroid
        return (0.0, 0.0)

    # Scale by distance (farther = stronger pull)
    # Normalize to unit vector, then scale by weight
    return (dx / dist * weight, dy / dist * weight)


def calculate_separation_force(
    agent: 'Agent',
    unit: 'Unit',
    separation_radius: float = BOIDS_SEPARATION_RADIUS,
    weight: float = BOIDS_SEPARATION_WEIGHT
) -> Tuple[float, float]:
    """
    Steer away from nearby squadmates to avoid crowding.

    This force pushes agents apart when they get too close,
    preventing clumping and maintaining tactical spacing.

    Args:
        agent: The agent to calculate steering for
        unit: The unit the agent belongs to
        separation_radius: Minimum comfortable distance between agents
        weight: Multiplier for the separation force

    Returns:
        (dx, dy) force vector pointing away from nearby agents
    """
    force_x = 0.0
    force_y = 0.0

    for other in unit.alive_agents:
        if other is agent:
            continue

        # Vector from other to agent (pointing away)
        dx = agent.position[0] - other.position[0]
        dy = agent.position[1] - other.position[1]

        dist = math.sqrt(dx*dx + dy*dy)

        # Only apply force within separation radius
        if dist < separation_radius and dist > 0.001:
            # Strength inversely proportional to distance (closer = stronger push)
            strength = (separation_radius - dist) / separation_radius
            force_x += (dx / dist) * strength
            force_y += (dy / dist) * strength

    # Apply weight
    return (force_x * weight, force_y * weight)


def calculate_alignment_force(
    agent: 'Agent',
    unit: 'Unit',
    weight: float = BOIDS_ALIGNMENT_WEIGHT
) -> Tuple[float, float]:
    """
    Match the average heading of nearby squadmates.

    This force steers agents to face the same direction as their squad,
    creating coordinated movement patterns.

    Args:
        agent: The agent to calculate steering for
        unit: The unit the agent belongs to
        weight: Multiplier for the alignment force

    Returns:
        (dx, dy) force vector representing target heading direction
    """
    # Get average heading of the unit
    avg_heading = unit.average_heading

    # Convert to direction vector
    avg_rad = math.radians(avg_heading)
    target_dx = math.cos(avg_rad)
    target_dy = math.sin(avg_rad)

    # Get agent's current heading
    agent_rad = math.radians(agent.orientation)
    agent_dx = math.cos(agent_rad)
    agent_dy = math.sin(agent_rad)

    # Steering force = target - current
    force_x = (target_dx - agent_dx) * weight
    force_y = (target_dy - agent_dy) * weight

    return (force_x, force_y)


def calculate_waypoint_force(
    agent: 'Agent',
    unit: 'Unit',
    weight: float = BOIDS_WAYPOINT_WEIGHT
) -> Tuple[float, float]:
    """
    Steer toward unit waypoint if active.

    This force pulls agents toward the unit's strategic waypoint,
    enabling coordinated squad movement across the battlefield.

    Prefers intermediate waypoint (unit.waypoint), falls back to goal_waypoint
    if no intermediate waypoint is set.

    Args:
        agent: The agent to calculate steering for
        unit: The unit the agent belongs to
        weight: Multiplier for the waypoint force

    Returns:
        (dx, dy) force vector pointing toward waypoint, or (0,0) if no waypoint
    """
    # Prefer intermediate waypoint, fall back to goal_waypoint
    waypoint = unit.waypoint
    if waypoint is None:
        waypoint = getattr(unit, 'goal_waypoint', None)
    if waypoint is None:
        return (0.0, 0.0)

    # Vector from agent to waypoint
    dx = waypoint[0] - agent.position[0]
    dy = waypoint[1] - agent.position[1]

    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 0.001:  # Already at waypoint
        return (0.0, 0.0)

    # Normalize and apply weight
    return (dx / dist * weight, dy / dist * weight)


def calculate_boids_steering(
    agent: 'Agent',
    unit: 'Unit',
    config: Optional[Dict[str, float]] = None
) -> Tuple[float, float]:
    """
    Combine all boids forces into a single steering vector.

    This function calculates the combined influence of cohesion, separation,
    alignment, and waypoint forces. The result is clamped to MAX_FORCE to
    prevent excessive steering.

    Args:
        agent: The agent to calculate steering for
        unit: The unit the agent belongs to
        config: Optional dict to override default weights:
            - 'cohesion': Cohesion weight (default: BOIDS_COHESION_WEIGHT)
            - 'separation': Separation weight (default: BOIDS_SEPARATION_WEIGHT)
            - 'alignment': Alignment weight (default: BOIDS_ALIGNMENT_WEIGHT)
            - 'waypoint': Waypoint weight (default: BOIDS_WAYPOINT_WEIGHT)
            - 'max_force': Maximum force magnitude (default: BOIDS_MAX_FORCE)
            - 'separation_radius': Separation radius (default: BOIDS_SEPARATION_RADIUS)

    Returns:
        (dx, dy) combined steering force, clamped to max force magnitude
    """
    # Get weights from config or use defaults
    if config is None:
        config = {}

    cohesion_weight = config.get('cohesion', BOIDS_COHESION_WEIGHT)
    separation_weight = config.get('separation', BOIDS_SEPARATION_WEIGHT)
    alignment_weight = config.get('alignment', BOIDS_ALIGNMENT_WEIGHT)
    waypoint_weight = config.get('waypoint', BOIDS_WAYPOINT_WEIGHT)
    max_force = config.get('max_force', BOIDS_MAX_FORCE)
    separation_radius = config.get('separation_radius', BOIDS_SEPARATION_RADIUS)

    # Calculate individual forces
    cohesion = calculate_cohesion_force(agent, unit, cohesion_weight)
    separation = calculate_separation_force(agent, unit, separation_radius, separation_weight)
    alignment = calculate_alignment_force(agent, unit, alignment_weight)
    waypoint = calculate_waypoint_force(agent, unit, waypoint_weight)

    # Combine forces
    total_x = cohesion[0] + separation[0] + alignment[0] + waypoint[0]
    total_y = cohesion[1] + separation[1] + alignment[1] + waypoint[1]

    # Clamp to max force
    magnitude = math.sqrt(total_x*total_x + total_y*total_y)
    if magnitude > max_force:
        total_x = (total_x / magnitude) * max_force
        total_y = (total_y / magnitude) * max_force

    return (total_x, total_y)


def steering_to_orientation(steering: Tuple[float, float]) -> Optional[float]:
    """
    Convert a steering vector to an orientation angle.

    Args:
        steering: (dx, dy) steering vector

    Returns:
        Orientation in degrees (0-360), or None if steering is zero
    """
    dx, dy = steering
    magnitude = math.sqrt(dx*dx + dy*dy)

    if magnitude < 0.001:
        return None

    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360


def blend_steering_with_random(
    steering: Tuple[float, float],
    agent: 'Agent',
    boids_weight: float = 0.7
) -> Tuple[float, float]:
    """
    Blend boids steering with some randomness for natural movement.

    This prevents perfectly mechanical movement while maintaining
    overall unit cohesion.

    Args:
        steering: (dx, dy) boids steering vector
        agent: The agent (used for current orientation)
        boids_weight: How much to weight boids (0-1), rest is random wander

    Returns:
        (dx, dy) blended steering vector
    """
    import random

    # Get random direction based on current orientation with some variance
    random_angle = agent.orientation + random.uniform(-30, 30)
    random_rad = math.radians(random_angle)
    random_dx = math.cos(random_rad)
    random_dy = math.sin(random_rad)

    random_weight = 1.0 - boids_weight

    # Blend
    blended_x = steering[0] * boids_weight + random_dx * random_weight
    blended_y = steering[1] * boids_weight + random_dy * random_weight

    return (blended_x, blended_y)


def calculate_target_force(
    agent: 'Agent',
    target_pos: Tuple[float, float],
    weight: float = BOIDS_WAYPOINT_WEIGHT
) -> Tuple[float, float]:
    """
    Steer toward a specific target position.

    Args:
        agent: The agent to calculate steering for
        target_pos: (x, y) position to steer toward
        weight: Multiplier for the force

    Returns:
        (dx, dy) force vector pointing toward target
    """
    # Vector from agent to target
    dx = target_pos[0] - agent.position[0]
    dy = target_pos[1] - agent.position[1]

    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 0.001:  # Already at target
        return (0.0, 0.0)

    # Normalize and apply weight
    return (dx / dist * weight, dy / dist * weight)


def calculate_stance_steering(
    agent: 'Agent',
    unit: 'Unit',
    enemies: Optional[List['Agent']] = None,
    config: Optional[Dict[str, float]] = None
) -> Tuple[float, float]:
    """
    Calculate boids steering adjusted for unit stance.

    Stance effects:
    - AGGRESSIVE: Waypoint force targets nearest enemy instead of waypoint
    - DEFENSIVE: Stronger cohesion (1.5x), reduced waypoint force (0.5x)
    - PATROL: Normal boids behavior

    Args:
        agent: The agent to calculate steering for
        unit: The unit the agent belongs to
        enemies: List of enemy agents (used for AGGRESSIVE stance)
        config: Optional dict to override default weights

    Returns:
        (dx, dy) combined steering force, clamped to max force magnitude
    """
    # Get weights from config or use defaults
    if config is None:
        config = {}

    cohesion_weight = config.get('cohesion', BOIDS_COHESION_WEIGHT)
    separation_weight = config.get('separation', BOIDS_SEPARATION_WEIGHT)
    alignment_weight = config.get('alignment', BOIDS_ALIGNMENT_WEIGHT)
    waypoint_weight = config.get('waypoint', BOIDS_WAYPOINT_WEIGHT)
    max_force = config.get('max_force', BOIDS_MAX_FORCE)
    separation_radius = config.get('separation_radius', BOIDS_SEPARATION_RADIUS)

    # Adjust weights based on stance
    stance = unit.stance

    if stance == UnitStance.DEFENSIVE:
        # Stronger cohesion, reduced waypoint force
        cohesion_weight *= 1.5
        waypoint_weight *= 0.5

    # Calculate individual forces
    cohesion = calculate_cohesion_force(agent, unit, cohesion_weight)
    separation = calculate_separation_force(agent, unit, separation_radius, separation_weight)
    alignment = calculate_alignment_force(agent, unit, alignment_weight)

    # Calculate target force based on stance
    if stance == UnitStance.AGGRESSIVE and enemies:
        # Find nearest alive enemy
        nearest_enemy = None
        nearest_dist_sq = float('inf')
        for enemy in enemies:
            if enemy.is_alive:
                dx = enemy.position[0] - agent.position[0]
                dy = enemy.position[1] - agent.position[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < nearest_dist_sq:
                    nearest_dist_sq = dist_sq
                    nearest_enemy = enemy

        if nearest_enemy is not None:
            # Target nearest enemy
            target = calculate_target_force(agent, nearest_enemy.position, waypoint_weight)
        else:
            # No enemies, fall back to waypoint
            target = calculate_waypoint_force(agent, unit, waypoint_weight)
    else:
        # Normal waypoint behavior (PATROL and DEFENSIVE)
        target = calculate_waypoint_force(agent, unit, waypoint_weight)

    # Combine forces
    total_x = cohesion[0] + separation[0] + alignment[0] + target[0]
    total_y = cohesion[1] + separation[1] + alignment[1] + target[1]

    # Clamp to max force
    magnitude = math.sqrt(total_x*total_x + total_y*total_y)
    if magnitude > max_force:
        total_x = (total_x / magnitude) * max_force
        total_y = (total_y / magnitude) * max_force

    return (total_x, total_y)
