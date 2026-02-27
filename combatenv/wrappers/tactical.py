"""
Tactical wrapper for agent-level combat and movement logic.

This wrapper handles the tactical level of the simulation:
- Action execution for controlled agent
- Autonomous combat for AI agents
- Terrain effects (fire damage, forest tracking)
- Respawn handling
- Termination/truncation checks
- Reward calculation

Usage:
    from combatenv.wrappers import TacticalWrapper

    env = TacticalCombatEnv(render_mode="human")
    env = TacticalWrapper(env)
"""

from typing import Any, Dict, List, Optional, Tuple
import math

import gymnasium as gym

from combatenv.config import (
    AGENT_MOVE_SPEED,
    NEAR_FOV_ACCURACY,
    FAR_FOV_ACCURACY,
    MOVEMENT_ACCURACY_PENALTY,
    MUZZLE_FLASH_OFFSET,
    MUZZLE_FLASH_LIFETIME,
    AGENT_KILL_REWARD,
    FRIENDLY_FIRE_PENALTY,
    FIRE_DAMAGE_PER_STEP,
    UNIT_COHESION_RADIUS,
    COHESION_ACCURACY_BONUS,
    COHESION_SURVIVAL_BONUS,
    COHESION_ARMOR_BONUS,
    WAYPOINT_DISTANCE_REWARD_SCALE,
    MAX_WAYPOINT_REWARD_DISTANCE,
    STANCE_COMPLIANCE_REWARD,
)
from combatenv.terrain import TerrainType
from combatenv.unit import UnitStance, get_unit_for_agent
from combatenv.agent import Agent, try_shoot_at_visible_target


class TacticalWrapper(gym.Wrapper):
    """
    Wrapper for tactical-level agent combat and movement.

    Handles individual agent actions, combat execution, terrain effects,
    and reward calculation.

    This wrapper can work independently for single-agent training or
    alongside OperationalWrapper for unit-based training.

    Attributes:
        enable_respawn: Whether agents respawn after death
        respawn_delay: Time before respawn (in seconds)
        enable_autonomous_combat: Whether AI agents shoot automatically
        enable_terrain_effects: Whether terrain affects agents
    """

    def __init__(
        self,
        env,
        enable_respawn: bool = True,
        respawn_delay: float = 3.0,
        enable_autonomous_combat: bool = True,
        enable_terrain_effects: bool = True,
        enable_cohesion_bonuses: bool = True,
    ):
        """
        Initialize the tactical wrapper.

        Args:
            env: Base environment to wrap
            enable_respawn: Whether agents respawn after death
            respawn_delay: Seconds before respawn
            enable_autonomous_combat: Whether AI agents shoot automatically
            enable_terrain_effects: Whether terrain affects agents
            enable_cohesion_bonuses: Whether cohesion provides accuracy/armor bonuses
        """
        super().__init__(env)

        self.enable_respawn = enable_respawn
        self.respawn_delay = respawn_delay
        self.enable_autonomous_combat = enable_autonomous_combat
        self.enable_terrain_effects = enable_terrain_effects
        self.enable_cohesion_bonuses = enable_cohesion_bonuses

        # Respawn timers (agent_id -> time_remaining)
        self._respawn_timers: Dict[int, float] = {}

        # Kill tracking for current step
        self._kills_this_step: Dict[str, int] = {}

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset and clear respawn timers."""
        self._respawn_timers = {}
        self._kills_this_step = {}
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[Any, Any, bool, bool, Dict]:
        """
        Step with tactical processing.

        This wraps the base step and adds:
        - Autonomous combat for AI agents
        - Terrain effect processing
        - Respawn handling

        Note: The base env already handles action application and projectiles.
        This wrapper adds optional processing that can be enabled/disabled.
        """
        # Let base env handle the step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Additional tactical processing
        dt = 1.0 / 60.0  # Assume 60 FPS

        if self.enable_terrain_effects:
            self._process_terrain_effects()

        if self.enable_respawn:
            self._handle_respawns(dt)

        return obs, reward, terminated, truncated, info

    # ==================== Combat Methods ====================

    def execute_combat_for_agent(self, agent: Agent) -> Optional[Any]:
        """
        Execute combat action for a specific agent.

        Args:
            agent: Agent to execute combat for

        Returns:
            Projectile if shot was fired, None otherwise
        """
        base_env = self.env.unwrapped
        spatial_grid = getattr(base_env, 'spatial_grid', None)
        terrain_grid = getattr(base_env, 'terrain_grid', None)

        if spatial_grid is None or not agent.can_shoot():
            return None

        nearby = spatial_grid.get_nearby_agents(agent)

        # Apply cohesion bonus to accuracy
        near_acc = self._get_accuracy_with_cohesion(agent, NEAR_FOV_ACCURACY)
        far_acc = self._get_accuracy_with_cohesion(agent, FAR_FOV_ACCURACY)

        projectile = try_shoot_at_visible_target(
            agent, nearby,
            near_acc, far_acc,
            MOVEMENT_ACCURACY_PENALTY,
            terrain_grid
        )

        if projectile:
            # Add to projectile list
            projectiles = getattr(base_env, 'projectiles', [])
            projectiles.append(projectile)

            # Add muzzle flash
            flash_x = agent.position[0] + math.cos(math.radians(agent.orientation)) * MUZZLE_FLASH_OFFSET
            flash_y = agent.position[1] + math.sin(math.radians(agent.orientation)) * MUZZLE_FLASH_OFFSET
            muzzle_flashes = getattr(base_env, 'muzzle_flashes', [])
            muzzle_flashes.append(((flash_x, flash_y), MUZZLE_FLASH_LIFETIME))

        return projectile

    def apply_movement(
        self,
        agent: Agent,
        move_x: float,
        move_y: float,
        dt: float = 1/60
    ) -> None:
        """
        Apply movement to an agent.

        Args:
            agent: Agent to move
            move_x: X movement direction (-1 to 1)
            move_y: Y movement direction (-1 to 1)
            dt: Delta time
        """
        if agent.is_stuck or not agent.is_alive:
            return

        base_env = self.env.unwrapped
        alive_agents = getattr(base_env, 'alive_agents', [])
        terrain_grid = getattr(base_env, 'terrain_grid', None)

        move_magnitude = math.sqrt(move_x**2 + move_y**2)
        if move_magnitude > 0.1:  # Dead zone
            # Calculate target orientation from movement direction
            target_orientation = math.degrees(math.atan2(move_y, move_x))
            agent.orientation = target_orientation

            # Move with speed proportional to magnitude (capped at 1.0)
            speed = min(move_magnitude, 1.0) * AGENT_MOVE_SPEED
            agent.move_forward(speed=speed, dt=dt, other_agents=alive_agents, terrain_grid=terrain_grid)
            agent.wander_direction = 1  # Mark as moving
        else:
            agent.wander_direction = 0  # Not moving

    # ==================== Terrain Effects ====================

    def _process_terrain_effects(self) -> None:
        """Process terrain effects for all alive agents."""
        base_env = self.env.unwrapped
        alive_agents = getattr(base_env, 'alive_agents', [])
        terrain_grid = getattr(base_env, 'terrain_grid', None)

        if terrain_grid is None:
            return

        for agent in alive_agents:
            cell = agent.get_grid_position()
            terrain = terrain_grid.get(*cell)

            if terrain == TerrainType.FIRE:
                # Fire damage bypasses armor
                agent.apply_terrain_damage(FIRE_DAMAGE_PER_STEP)

            # Track terrain effects (affects speed, detection, shooting)
            agent.in_forest = (terrain == TerrainType.FOREST)
            agent.in_water = (terrain == TerrainType.WATER)

    def get_terrain_at_agent(self, agent: Agent) -> TerrainType:
        """
        Get terrain type at agent's position.

        Args:
            agent: Agent to check

        Returns:
            TerrainType at agent's grid position
        """
        base_env = self.env.unwrapped
        terrain_grid = getattr(base_env, 'terrain_grid', None)

        if terrain_grid is None:
            return TerrainType.EMPTY

        cell = agent.get_grid_position()
        return terrain_grid.get(*cell)

    # ==================== Respawn Handling ====================

    def _handle_respawns(self, dt: float) -> None:
        """Handle agent respawning after death."""
        base_env = self.env.unwrapped
        all_agents = getattr(base_env, 'all_agents', [])

        for agent in all_agents:
            if not agent.is_alive:
                agent_id = id(agent)
                if agent_id not in self._respawn_timers:
                    self._respawn_timers[agent_id] = self.respawn_delay
                else:
                    self._respawn_timers[agent_id] -= dt
                    if self._respawn_timers[agent_id] <= 0:
                        agent.respawn()
                        del self._respawn_timers[agent_id]

    def get_respawn_time(self, agent: Agent) -> Optional[float]:
        """
        Get remaining respawn time for an agent.

        Args:
            agent: Agent to check

        Returns:
            Seconds until respawn, or None if agent is alive
        """
        if agent.is_alive:
            return None
        return self._respawn_timers.get(id(agent), self.respawn_delay)

    def force_respawn(self, agent: Agent) -> bool:
        """
        Force immediate respawn of an agent.

        Args:
            agent: Agent to respawn

        Returns:
            True if respawned, False if agent was already alive
        """
        if agent.is_alive:
            return False

        agent.respawn()
        agent_id = id(agent)
        if agent_id in self._respawn_timers:
            del self._respawn_timers[agent_id]
        return True

    # ==================== Cohesion Bonuses ====================

    def _is_agent_cohesive(self, agent: Agent) -> bool:
        """Check if agent is within cohesion radius of unit centroid."""
        if not self.enable_cohesion_bonuses:
            return False

        base_env = self.env.unwrapped
        config = getattr(base_env, 'config', None)
        if config is None or not getattr(config, 'use_units', False):
            return False

        if agent.unit_id is None:
            return False

        blue_units = getattr(base_env, 'blue_units', [])
        red_units = getattr(base_env, 'red_units', [])
        all_units = blue_units + red_units

        unit = get_unit_for_agent(agent, all_units)
        if unit is None:
            return False

        centroid = unit.centroid
        dist_sq = (agent.position[0] - centroid[0])**2 + (agent.position[1] - centroid[1])**2
        return dist_sq <= UNIT_COHESION_RADIUS**2

    def _get_accuracy_with_cohesion(self, agent: Agent, base_accuracy: float) -> float:
        """Get accuracy with cohesion bonus if applicable."""
        if self._is_agent_cohesive(agent):
            return base_accuracy * (1.0 + COHESION_ACCURACY_BONUS)
        return base_accuracy

    def get_damage_with_cohesion(self, target: Agent, base_damage: int) -> int:
        """
        Get damage reduced by cohesion armor bonus.

        Args:
            target: Agent receiving damage
            base_damage: Original damage amount

        Returns:
            Damage after cohesion armor reduction
        """
        if self._is_agent_cohesive(target):
            return int(base_damage * (1.0 - COHESION_ARMOR_BONUS))
        return base_damage

    # ==================== Termination Checks ====================

    def check_terminated(self) -> bool:
        """
        Check if episode should terminate.

        Termination conditions:
        - Controlled agent dies (if configured)
        - Either team is eliminated (if configured)

        Returns:
            True if episode should terminate
        """
        base_env = self.env.unwrapped
        config = getattr(base_env, 'config', None)

        if config is None:
            return False

        # Terminate if controlled agent dies
        if getattr(config, 'terminate_on_controlled_death', True):
            controlled = getattr(base_env, 'controlled_agent', None)
            if controlled and not controlled.is_alive:
                return True

        # Terminate if either team is eliminated
        if getattr(config, 'terminate_on_team_elimination', True):
            blue_agents = getattr(base_env, 'blue_agents', [])
            red_agents = getattr(base_env, 'red_agents', [])
            living_blue = sum(1 for a in blue_agents if a.is_alive)
            living_red = sum(1 for a in red_agents if a.is_alive)
            if living_blue == 0 or living_red == 0:
                return True

        return False

    def check_truncated(self) -> bool:
        """
        Check if episode should be truncated (step limit reached).

        Returns:
            True if step limit reached
        """
        base_env = self.env.unwrapped
        config = getattr(base_env, 'config', None)
        step_count = getattr(base_env, 'step_count', 0)

        if config is None:
            return False

        max_steps = getattr(config, 'max_steps', None)
        if max_steps is None:
            return False

        return step_count >= max_steps

    # ==================== Reward Calculation ====================

    def calculate_tactical_reward(
        self,
        kills: Dict[str, int],
        controlled_agent: Optional[Agent] = None
    ) -> float:
        """
        Calculate tactical reward for controlled agent.

        Rewards:
        - Kill reward per kill by controlled agent
        - Survival reward per step
        - Cohesion survival bonus if within unit

        Penalties:
        - Friendly fire penalty

        Args:
            kills: Dict with 'controlled' and 'friendly_fire' counts
            controlled_agent: Agent to calculate reward for

        Returns:
            Reward value
        """
        reward = 0.0

        # Kill reward
        reward += kills.get('controlled', 0) * AGENT_KILL_REWARD

        # Friendly fire penalty
        reward += kills.get('friendly_fire', 0) * FRIENDLY_FIRE_PENALTY

        # Survival reward with cohesion bonus
        if controlled_agent and controlled_agent.is_alive:
            survival_reward = 0.01
            if self._is_agent_cohesive(controlled_agent):
                survival_reward *= (1.0 + COHESION_SURVIVAL_BONUS)
            reward += survival_reward

        return reward

    def calculate_waypoint_reward(
        self,
        agent: Agent,
        waypoint: Optional[Tuple[float, float]]
    ) -> float:
        """
        Calculate reward based on distance to waypoint.

        Args:
            agent: Agent to calculate reward for
            waypoint: Target waypoint (x, y) or None

        Returns:
            Waypoint distance reward (0 to WAYPOINT_DISTANCE_REWARD_SCALE)
        """
        if waypoint is None or not agent.is_alive:
            return 0.0

        dist = math.sqrt(
            (agent.position[0] - waypoint[0])**2 +
            (agent.position[1] - waypoint[1])**2
        )

        return max(0, 1 - dist / MAX_WAYPOINT_REWARD_DISTANCE) * WAYPOINT_DISTANCE_REWARD_SCALE

    def calculate_stance_reward(
        self,
        agent: Agent,
        stance: UnitStance,
        waypoint: Optional[Tuple[float, float]],
        enemies: List[Agent]
    ) -> float:
        """
        Calculate reward for stance compliance.

        Args:
            agent: Agent to check
            stance: Unit stance
            waypoint: Unit waypoint
            enemies: List of enemy agents

        Returns:
            Stance compliance reward
        """
        if waypoint is None or not agent.is_alive:
            return 0.0

        dist_to_waypoint = math.sqrt(
            (agent.position[0] - waypoint[0])**2 +
            (agent.position[1] - waypoint[1])**2
        )

        if stance == UnitStance.AGGRESSIVE:
            # Bonus if moving toward enemy
            if enemies:
                nearest_dist = float('inf')
                for enemy in enemies:
                    if enemy.is_alive:
                        dx = enemy.position[0] - agent.position[0]
                        dy = enemy.position[1] - agent.position[1]
                        enemy_dist = math.sqrt(dx*dx + dy*dy)
                        nearest_dist = min(nearest_dist, enemy_dist)

                if nearest_dist < MAX_WAYPOINT_REWARD_DISTANCE and agent.is_moving:
                    return STANCE_COMPLIANCE_REWARD

        elif stance == UnitStance.DEFENSIVE:
            # Bonus if near waypoint
            if dist_to_waypoint <= UNIT_COHESION_RADIUS:
                return STANCE_COMPLIANCE_REWARD

        elif stance == UnitStance.PATROL:
            # Bonus if moving toward waypoint
            if agent.is_moving and dist_to_waypoint > 0.5:
                return STANCE_COMPLIANCE_REWARD

        return 0.0

    # ==================== Utility Methods ====================

    def is_agent_in_reserve(self, agent: Agent) -> bool:
        """
        Check if agent belongs to a unit that is in reserve.

        Args:
            agent: Agent to check

        Returns:
            True if agent's unit is in reserve
        """
        base_env = self.env.unwrapped
        config = getattr(base_env, 'config', None)

        if config is None or not getattr(config, 'use_units', False):
            return False

        if not agent.following_unit:
            return False

        blue_units = getattr(base_env, 'blue_units', [])
        red_units = getattr(base_env, 'red_units', [])
        all_units = blue_units + red_units

        unit = get_unit_for_agent(agent, all_units)
        return unit is not None and unit.in_reserve

    def get_nearby_enemies(self, agent: Agent, max_distance: float = 10.0) -> List[Agent]:
        """
        Get enemies within range of an agent.

        Args:
            agent: Agent to check from
            max_distance: Maximum distance to consider

        Returns:
            List of enemy agents sorted by distance
        """
        base_env = self.env.unwrapped
        blue_agents = getattr(base_env, 'blue_agents', [])
        red_agents = getattr(base_env, 'red_agents', [])

        enemies = red_agents if agent.team == "blue" else blue_agents
        result = []

        for enemy in enemies:
            if not enemy.is_alive:
                continue
            dx = enemy.position[0] - agent.position[0]
            dy = enemy.position[1] - agent.position[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist <= max_distance:
                result.append((enemy, dist))

        result.sort(key=lambda x: x[1])
        return [e for e, _ in result]

    def get_nearby_allies(self, agent: Agent, max_distance: float = 10.0) -> List[Agent]:
        """
        Get allies within range of an agent.

        Args:
            agent: Agent to check from
            max_distance: Maximum distance to consider

        Returns:
            List of ally agents sorted by distance
        """
        base_env = self.env.unwrapped
        blue_agents = getattr(base_env, 'blue_agents', [])
        red_agents = getattr(base_env, 'red_agents', [])

        allies = blue_agents if agent.team == "blue" else red_agents
        result = []

        for ally in allies:
            if not ally.is_alive or ally is agent:
                continue
            dx = ally.position[0] - agent.position[0]
            dy = ally.position[1] - agent.position[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist <= max_distance:
                result.append((ally, dist))

        result.sort(key=lambda x: x[1])
        return [a for a, _ in result]

    def distance_between(self, agent1: Agent, agent2: Agent) -> float:
        """
        Calculate distance between two agents.

        Args:
            agent1: First agent
            agent2: Second agent

        Returns:
            Euclidean distance
        """
        dx = agent1.position[0] - agent2.position[0]
        dy = agent1.position[1] - agent2.position[1]
        return math.sqrt(dx*dx + dy*dy)
