"""
Multi-agent wrapper that enables control of all 200 agents.

The base TacticalCombatEnv only supports controlling one agent (the first blue agent).
This wrapper intercepts the environment's step process and allows external control
of all 200 agents.

Usage:
    from combatenv import TacticalCombatEnv
    from rl_student.wrappers import MultiAgentWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)

    observations, info = env.reset()
    # observations is Dict[int, np.ndarray] mapping agent_idx -> 88-float observation

    actions = {agent_idx: np.array([move_x, move_y, shoot]) for agent_idx in observations}
    next_obs, rewards, terminated, truncated, info = env.step(actions)
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import gymnasium as gym

from combatenv import config
from combatenv.config import OPERATIONAL_GRID_SIZE, TACTICAL_CELLS_PER_OPERATIONAL
from combatenv.agent import try_shoot_at_visible_target
from combatenv.fov import get_fov_cells, get_fov_cache
from combatenv.terrain import TerrainType
from combatenv.unit import get_unit_for_agent


class MultiAgentWrapper(gym.Wrapper):
    """
    Wrapper that enables multi-agent control for combatenv.

    The base environment only supports controlling one agent.
    This wrapper provides an interface for controlling all 200 agents.

    Attributes:
        num_agents: Total number of agents (200)
        agent_list: Ordered list of all agents (blue first, then red)
        agent_to_idx: Map from agent id() to index
    """

    def __init__(self, env):
        """
        Initialize the multi-agent wrapper.

        Args:
            env: The base TacticalCombatEnv instance
        """
        super().__init__(env)

        # Store reference to unwrapped environment for direct access
        self._base_env = env.unwrapped

        # Will be populated after reset()
        self.num_agents = 0
        self.agent_list = []  # Ordered list of all agents
        self.agent_to_idx = {}  # Map agent id() to index
        self.idx_to_agent = {}  # Map index to agent

        # Track kills for reward attribution
        self._pending_kills = {}  # projectile_id -> shooter_agent_idx

        # Track friendly collisions per step
        self._friendly_collisions = {}  # agent_idx -> collision count

        # Track last rewards for debug display
        self._last_rewards = {}  # agent_idx -> reward (per step)
        self._episode_rewards = {}  # agent_idx -> cumulative reward (per episode)

    def reset(self, **kwargs) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        Reset environment and return observations for all agents.

        Returns:
            observations: Dict mapping agent_idx -> observation array (88 floats)
            info: Environment info dictionary
        """
        # Call base reset
        _, info = self.env.reset(**kwargs)

        # Build agent list (blue first, then red for consistent ordering)
        self.agent_list = list(self._base_env.blue_agents) + list(self._base_env.red_agents)
        self.num_agents = len(self.agent_list)
        self.agent_to_idx = {id(agent): idx for idx, agent in enumerate(self.agent_list)}
        self.idx_to_agent = {idx: agent for idx, agent in enumerate(self.agent_list)}

        # Clear kill tracking
        self._pending_kills = {}

        # Reset episode rewards
        self._episode_rewards = {}

        # Generate observations for all agents
        observations = self._get_all_observations()

        return observations, info

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict]:
        """
        Execute one step with actions for all agents.

        Args:
            actions: Dict mapping agent_idx -> action array (3 floats: move_x, move_y, shoot)

        Returns:
            observations: Dict mapping agent_idx -> observation array
            rewards: Dict mapping agent_idx -> reward float
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Environment info dictionary
        """
        dt = 1.0 / config.FPS
        self._base_env.step_count += 1

        # Phase 1: Update alive agents and spatial grid
        self._base_env.alive_agents = [a for a in self._base_env.all_agents if a.is_alive]
        self._base_env.spatial_grid.build(self._base_env.alive_agents)

        # Phase 2: Update resources for all agents
        for agent in self._base_env.alive_agents:
            agent.update_cooldown(dt)
            agent.update_reload(dt)
            agent.update_stamina(dt, agent.is_moving)

        # Phase 3: Apply actions to ALL agents (tracks friendly collisions)
        rewards = {idx: 0.0 for idx in range(self.num_agents)}
        self._friendly_collisions = {}  # Reset per-step tracking
        for agent_idx, action in actions.items():
            agent = self.agent_list[agent_idx]
            if agent.is_alive:
                self._apply_agent_action(agent, agent_idx, action, dt)

        # Phase 4: Update projectiles and track kills/hits
        enemy_kills, friendly_kills, hits_by_victim = self._update_projectiles(dt)

        # Apply rewards for enemy kills: +10.0 each
        for agent_idx, count in enemy_kills.items():
            rewards[agent_idx] += count * 10.0

        # Apply penalty for friendly fire kills: -2000.0 each
        for agent_idx, count in friendly_kills.items():
            rewards[agent_idx] += count * -2000.0

        # Apply penalty for hits taken: -0.2 each
        for agent_idx, count in hits_by_victim.items():
            rewards[agent_idx] += count * -0.2

        # Apply penalty for friendly collisions: -0.1 each
        for agent_idx, count in self._friendly_collisions.items():
            rewards[agent_idx] += count * -0.1

        # Phase 5: Process terrain effects
        self._process_terrain_effects()

        # Phase 6: Update muzzle flashes
        self._base_env.muzzle_flashes = [
            (pos, lifetime - dt)
            for pos, lifetime in self._base_env.muzzle_flashes
            if lifetime - dt > 0
        ]

        # Add survival reward for living agents, death penalty for dead
        for agent_idx, agent in enumerate(self.agent_list):
            if agent.is_alive:
                rewards[agent_idx] += 0.01  # Small survival reward
            else:
                rewards[agent_idx] += -1000.0  # Death penalty

        # Store rewards for debug display
        self._last_rewards = rewards.copy()

        # Accumulate episode rewards
        for agent_idx, reward in rewards.items():
            self._episode_rewards[agent_idx] = self._episode_rewards.get(agent_idx, 0.0) + reward

        # Generate observations for all agents
        observations = self._get_all_observations()

        # Check termination
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        info = self._base_env._get_info()

        return observations, rewards, terminated, truncated, info

    def _apply_agent_action(self, agent, agent_idx: int, action: np.ndarray, dt: float) -> None:
        """
        Apply action to a single agent.

        This mirrors the logic in TacticalCombatEnv._apply_action()
        but works for any agent.

        Args:
            agent: The Agent instance
            agent_idx: The agent's index
            action: [move_x, move_y, shoot, think] array (4 floats)
            dt: Delta time
        """
        if not agent.is_alive or agent.is_stuck:
            return

        # Support both 3-element [move_x, move_y, shoot] and 4-element [move_x, move_y, shoot, think]
        move_x = action[0]
        move_y = action[1]
        shoot = action[2]
        think = action[3] if len(action) > 3 else 0.0

        # If "think" action is triggered, agent invokes its own logic
        if think > 0.5:
            # For now, think action just makes the agent use its built-in wander behavior
            # This could be extended to invoke more complex decision making
            agent.wander(
                dt=dt,
                other_agents=self._base_env.alive_agents,
                terrain_grid=self._base_env.terrain_grid
            )
            return

        # Apply movement
        move_magnitude = math.sqrt(move_x**2 + move_y**2)
        if move_magnitude > 0.1:  # Dead zone
            target_orientation = math.degrees(math.atan2(move_y, move_x))
            agent.orientation = target_orientation

            speed = min(move_magnitude, 1.0) * config.AGENT_MOVE_SPEED
            old_pos = agent.position

            move_succeeded = agent.move_forward(
                speed=speed,
                dt=dt,
                other_agents=self._base_env.alive_agents,
                terrain_grid=self._base_env.terrain_grid
            )

            # Check for friendly collision if movement was blocked
            if not move_succeeded:
                # Check if blocked by a friendly agent
                for other in self._base_env.alive_agents:
                    if other is agent:
                        continue
                    if other.team == agent.team:  # Same team = friendly
                        dist = math.sqrt(
                            (other.position[0] - agent.position[0])**2 +
                            (other.position[1] - agent.position[1])**2
                        )
                        if dist < 1.5:  # Close enough to be the blocker
                            self._friendly_collisions[agent_idx] = self._friendly_collisions.get(agent_idx, 0) + 1
                            break

            agent.wander_direction = 1
        else:
            agent.wander_direction = 0

        # Apply shoot action
        if shoot > 0.5 and agent.can_shoot():
            nearby = self._base_env.spatial_grid.get_nearby_agents(agent)

            projectile = try_shoot_at_visible_target(
                agent, nearby,
                config.NEAR_FOV_ACCURACY, config.FAR_FOV_ACCURACY,
                config.MOVEMENT_ACCURACY_PENALTY,
                self._base_env.terrain_grid
            )

            if projectile:
                self._base_env.projectiles.append(projectile)
                # Track shooter for reward attribution
                self._pending_kills[id(projectile)] = agent_idx

                # Add muzzle flash
                flash_x = agent.position[0] + math.cos(math.radians(agent.orientation)) * config.MUZZLE_FLASH_OFFSET
                flash_y = agent.position[1] + math.sin(math.radians(agent.orientation)) * config.MUZZLE_FLASH_OFFSET
                self._base_env.muzzle_flashes.append(((flash_x, flash_y), config.MUZZLE_FLASH_LIFETIME))

    def _update_projectiles(self, dt: float) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        """
        Update projectiles and track kills/hits by agent.

        Returns:
            Tuple of:
                - enemy_kills: Dict mapping shooter_agent_idx -> number of enemy kills
                - friendly_kills: Dict mapping shooter_agent_idx -> number of friendly kills
                - hits_by_victim: Dict mapping victim_agent_idx -> number of hits taken
        """
        projectiles_to_remove: Set[int] = set()
        enemy_kills: Dict[int, int] = {}
        friendly_kills: Dict[int, int] = {}
        hits_by_victim: Dict[int, int] = {}

        for i, projectile in enumerate(self._base_env.projectiles):
            if projectile.update(dt, self._base_env.terrain_grid):
                projectiles_to_remove.add(i)
                # Clean up kill tracking
                if id(projectile) in self._pending_kills:
                    del self._pending_kills[id(projectile)]
                continue

            for agent in self._base_env.alive_agents:
                if projectile.check_collision(agent):
                    was_alive = agent.is_alive
                    agent.take_damage(projectile.damage)

                    # Track hit for the victim (regardless of death)
                    victim_idx = self.agent_to_idx[id(agent)]
                    hits_by_victim[victim_idx] = hits_by_victim.get(victim_idx, 0) + 1

                    if was_alive and not agent.is_alive:
                        # Clear dead agent from FOV cache
                        get_fov_cache().remove_agent(id(agent))

                        # Determine if this was friendly fire or enemy kill
                        is_friendly_fire = (projectile.owner_team == agent.team)

                        # Track kill for the shooter
                        if id(projectile) in self._pending_kills:
                            shooter_idx = self._pending_kills[id(projectile)]
                            if is_friendly_fire:
                                friendly_kills[shooter_idx] = friendly_kills.get(shooter_idx, 0) + 1
                            else:
                                enemy_kills[shooter_idx] = enemy_kills.get(shooter_idx, 0) + 1

                        # Update team kill counts
                        if projectile.owner_team == "blue":
                            self._base_env.blue_kills += 1
                        else:
                            self._base_env.red_kills += 1

                    # Clean up
                    if id(projectile) in self._pending_kills:
                        del self._pending_kills[id(projectile)]
                    projectiles_to_remove.add(i)
                    break

        self._base_env.projectiles = [p for i, p in enumerate(self._base_env.projectiles) if i not in projectiles_to_remove]
        return enemy_kills, friendly_kills, hits_by_victim

    def _process_terrain_effects(self) -> None:
        """Process terrain effects (fire damage, swamp stuck) for all alive agents."""
        for agent in self._base_env.alive_agents:
            cell = agent.get_grid_position()
            terrain = self._base_env.terrain_grid.get(*cell)

            if terrain == TerrainType.FIRE:
                agent.apply_terrain_damage(config.FIRE_DAMAGE_PER_STEP)

            elif terrain == TerrainType.FOREST:
                # Forest terrain - handled via speed reduction in movement
                pass

            agent.update_stuck()

    def _get_all_observations(self) -> Dict[int, np.ndarray]:
        """
        Generate observations for all agents.

        Returns:
            Dict mapping agent_idx -> 88-float observation array
        """
        observations = {}

        for idx, agent in enumerate(self.agent_list):
            if agent.is_alive:
                obs = self._get_agent_observation(agent, idx)
            else:
                obs = np.zeros(92, dtype=np.float32)
            observations[idx] = obs

        return observations

    def _get_agent_observation(self, agent, agent_idx: int) -> np.ndarray:
        """
        Generate observation for a single agent.

        This mirrors TacticalCombatEnv._get_obs() but works for any agent.

        Args:
            agent: The Agent instance
            agent_idx: The agent's index (0-99 = blue, 100-199 = red)

        Returns:
            89-float numpy array (includes Chebyshev distance at index 88)
        """
        obs = np.zeros(92, dtype=np.float32)

        if not agent.is_alive:
            return obs

        # Agent state (indices 0-9)
        obs[0] = np.clip(agent.position[0] / config.GRID_SIZE, 0, 1)
        obs[1] = np.clip(agent.position[1] / config.GRID_SIZE, 0, 1)
        obs[2] = np.clip((agent.orientation % 360) / 360.0, 0, 1)
        obs[3] = np.clip(agent.health / config.AGENT_MAX_HEALTH, 0, 1)
        obs[4] = np.clip(agent.stamina / config.AGENT_MAX_STAMINA, 0, 1)
        obs[5] = np.clip(agent.armor / config.AGENT_MAX_ARMOR, 0, 1)
        obs[6] = np.clip(agent.ammo_reserve / config.AGENT_MAX_AMMO, 0, 1)
        obs[7] = np.clip(agent.magazine_ammo / 30.0, 0, 1)
        obs[8] = 1.0 if agent.can_shoot() else 0.0
        obs[9] = 1.0 if agent.is_reloading else 0.0

        # Determine enemies and allies based on agent's team
        is_blue = agent_idx < len(self._base_env.blue_agents)
        if is_blue:
            enemies = self._base_env.red_agents
            allies = [a for a in self._base_env.blue_agents if a is not agent]
        else:
            enemies = self._base_env.blue_agents
            allies = [a for a in self._base_env.red_agents if a is not agent]

        # Nearest enemies (indices 10-29)
        enemies_with_dist = [
            (e, self._distance(agent, e))
            for e in enemies if e.is_alive
        ]
        enemies_with_dist.sort(key=lambda x: x[1])

        for i, (enemy, dist) in enumerate(enemies_with_dist[:5]):
            base_idx = 10 + i * 4
            rel_x = (enemy.position[0] - agent.position[0]) / config.GRID_SIZE + 0.5
            rel_y = (enemy.position[1] - agent.position[1]) / config.GRID_SIZE + 0.5
            obs[base_idx] = np.clip(rel_x, 0, 1)
            obs[base_idx + 1] = np.clip(rel_y, 0, 1)
            obs[base_idx + 2] = np.clip(enemy.health / config.AGENT_MAX_HEALTH, 0, 1)
            obs[base_idx + 3] = np.clip(dist / config.GRID_SIZE, 0, 1)

        # Nearest allies (indices 30-49)
        allies_with_dist = [
            (a, self._distance(agent, a))
            for a in allies if a.is_alive
        ]
        allies_with_dist.sort(key=lambda x: x[1])

        for i, (ally, dist) in enumerate(allies_with_dist[:5]):
            base_idx = 30 + i * 4
            rel_x = (ally.position[0] - agent.position[0]) / config.GRID_SIZE + 0.5
            rel_y = (ally.position[1] - agent.position[1]) / config.GRID_SIZE + 0.5
            obs[base_idx] = np.clip(rel_x, 0, 1)
            obs[base_idx + 1] = np.clip(rel_y, 0, 1)
            obs[base_idx + 2] = np.clip(ally.health / config.AGENT_MAX_HEALTH, 0, 1)
            obs[base_idx + 3] = np.clip(dist / config.GRID_SIZE, 0, 1)

        # Terrain in FOV (indices 50-87)
        fov_cells = get_fov_cells(
            agent.position,
            agent.orientation,
            fov_angle=config.FAR_FOV_ANGLE,
            max_range=config.FAR_FOV_RANGE,
            terrain_grid=self._base_env.terrain_grid
        )

        # Sort cells by distance
        agent_x, agent_y = agent.position
        sorted_cells = sorted(
            fov_cells,
            key=lambda c: (c[0] - agent_x) ** 2 + (c[1] - agent_y) ** 2
        )

        # Add terrain type for each cell (normalized)
        for i, (cx, cy) in enumerate(sorted_cells[:38]):
            terrain_type = self._base_env.terrain_grid.get(cx, cy)
            obs[50 + i] = terrain_type / 4.0  # 5 terrain types: 0-4

        # Chebyshev distance to waypoint (index 88)
        # Normalized [0, 1] where 0 = at waypoint, 1 = max distance (7 cells on operational grid)
        obs[88] = self._get_chebyshev_to_waypoint(agent, agent_idx)

        # Waypoint relative position (index 89, 90)
        # Normalized [-1, 1] where negative = west/north, positive = east/south
        wp_rel_x, wp_rel_y = self._get_waypoint_relative_pos(agent, agent_idx)
        obs[89] = wp_rel_x  # -1 = waypoint west, +1 = waypoint east
        obs[90] = wp_rel_y  # -1 = waypoint north, +1 = waypoint south

        # Centroid distance (index 91)
        # Normalized [0, 1] where 0 = at centroid, 1 = max distance (GRID_SIZE)
        obs[91] = self._get_centroid_distance(agent, agent_idx)

        return obs

    def _get_chebyshev_to_waypoint(self, agent, agent_idx: int) -> float:
        """
        Calculate Chebyshev distance from agent's operational cell to waypoint's operational cell.

        Args:
            agent: The agent
            agent_idx: The agent's index (used to determine team)

        Returns:
            Normalized Chebyshev distance [0, 1]
        """
        # Get unit for this agent
        is_blue = agent_idx < len(self._base_env.blue_agents)
        units = self._base_env.blue_units if is_blue else self._base_env.red_units

        unit = get_unit_for_agent(agent, units)
        if unit is None or unit.waypoint is None:
            return 1.0  # No waypoint = max distance

        # Get agent's operational cell (0-7, 0-7)
        agent_op_x = int(agent.position[0] / TACTICAL_CELLS_PER_OPERATIONAL)
        agent_op_y = int(agent.position[1] / TACTICAL_CELLS_PER_OPERATIONAL)

        # Get waypoint's operational cell (0-7, 0-7)
        wp_op_x = int(unit.waypoint[0] / TACTICAL_CELLS_PER_OPERATIONAL)
        wp_op_y = int(unit.waypoint[1] / TACTICAL_CELLS_PER_OPERATIONAL)

        # Clamp to valid range
        agent_op_x = max(0, min(agent_op_x, OPERATIONAL_GRID_SIZE - 1))
        agent_op_y = max(0, min(agent_op_y, OPERATIONAL_GRID_SIZE - 1))
        wp_op_x = max(0, min(wp_op_x, OPERATIONAL_GRID_SIZE - 1))
        wp_op_y = max(0, min(wp_op_y, OPERATIONAL_GRID_SIZE - 1))

        # Chebyshev distance = max(|dx|, |dy|)
        chebyshev = max(abs(agent_op_x - wp_op_x), abs(agent_op_y - wp_op_y))

        # Normalize by max possible distance (7 for 8x8 grid)
        return chebyshev / (OPERATIONAL_GRID_SIZE - 1)

    def _get_waypoint_relative_pos(self, agent, agent_idx: int) -> Tuple[float, float]:
        """
        Get normalized relative position from agent to waypoint.

        Args:
            agent: The agent
            agent_idx: The agent's index (used to determine team)

        Returns:
            Tuple (rel_x, rel_y) normalized to [-1, 1]
            rel_x: -1 = waypoint west, +1 = waypoint east
            rel_y: -1 = waypoint north, +1 = waypoint south
        """
        # Get unit for this agent
        is_blue = agent_idx < len(self._base_env.blue_agents)
        units = self._base_env.blue_units if is_blue else self._base_env.red_units

        unit = get_unit_for_agent(agent, units)
        if unit is None or unit.waypoint is None:
            return (0.0, 0.0)  # No waypoint = no direction

        # Calculate relative position
        dx = unit.waypoint[0] - agent.position[0]
        dy = unit.waypoint[1] - agent.position[1]

        # Normalize by grid size (max possible distance)
        max_dist = float(config.GRID_SIZE)
        rel_x = max(-1.0, min(1.0, dx / max_dist))
        rel_y = max(-1.0, min(1.0, dy / max_dist))

        return (rel_x, rel_y)

    def _get_centroid_distance(self, agent, agent_idx: int) -> float:
        """
        Calculate distance from agent to unit centroid.

        Args:
            agent: The agent
            agent_idx: The agent's index (used to determine team)

        Returns:
            Normalized distance [0, 1] where 0 = at centroid
        """
        import math

        # Get unit for this agent
        is_blue = agent_idx < len(self._base_env.blue_agents)
        units = self._base_env.blue_units if is_blue else self._base_env.red_units

        unit = get_unit_for_agent(agent, units)
        if unit is None:
            return 1.0  # No unit = max distance

        # Get centroid
        centroid = unit.centroid
        if centroid == (0.0, 0.0) and not unit.alive_agents:
            return 1.0  # No alive agents = max distance

        # Calculate Euclidean distance to centroid
        dx = agent.position[0] - centroid[0]
        dy = agent.position[1] - centroid[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # Normalize by grid size (max possible distance)
        return min(1.0, dist / float(config.GRID_SIZE))

    def _distance(self, agent1, agent2) -> float:
        """Calculate Euclidean distance between two agents."""
        dx = agent1.position[0] - agent2.position[0]
        dy = agent1.position[1] - agent2.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def _check_terminated(self) -> bool:
        """Check if episode should terminate (only when BOTH teams eliminated)."""
        living_blue = sum(1 for a in self._base_env.blue_agents if a.is_alive)
        living_red = sum(1 for a in self._base_env.red_agents if a.is_alive)
        # Only terminate when both teams are completely dead
        return living_blue == 0 and living_red == 0

    def _check_truncated(self) -> bool:
        """Check if episode should be truncated."""
        if self._base_env.config.max_steps is None:
            return False
        return self._base_env.step_count >= self._base_env.config.max_steps

    def render(self):
        """Render the environment (delegates to base env)."""
        # Pass last rewards to base env for debug display
        self._base_env._agent_rewards = self._last_rewards
        self._base_env._episode_rewards = self._episode_rewards
        self._base_env._agent_list = self.agent_list
        return self._base_env.render()
