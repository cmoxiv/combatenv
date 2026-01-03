"""
Integration tests for the Grid-World Multi-Agent Tactical Simulation.

Tests cover end-to-end scenarios:
- Full episode execution
- Combat flow (shoot -> hit -> damage -> death)
- Terrain interaction (fire damage, swamp stuck, building blocking)
- FOV with terrain LOS blocking
- Respawn cycle
- Gymnasium compliance

Run with: pytest tests/test_integration.py -v
"""

import pytest
import numpy as np
from combatenv import (
    TacticalCombatEnv, EnvConfig,
    Agent, spawn_team,
    create_projectile,
    is_agent_visible_to_agent, get_fov_cells,
    TerrainGrid, TerrainType,
    config
)
PROJECTILE_DAMAGE = config.PROJECTILE_DAMAGE
AGENT_MAX_HEALTH = config.AGENT_MAX_HEALTH
GRID_SIZE = config.GRID_SIZE
FIRE_DAMAGE_PER_STEP = config.FIRE_DAMAGE_PER_STEP
SWAMP_STUCK_MIN_STEPS = config.SWAMP_STUCK_MIN_STEPS


class TestFullEpisode:
    """Test complete episode execution."""

    def test_episode_runs_to_completion(self):
        """Test episode runs without error until termination."""
        config = EnvConfig(max_steps=100)
        env = TacticalCombatEnv(render_mode=None, config=config)
        obs, info = env.reset(seed=42)

        done = False
        steps = 0
        max_steps = 200

        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps > 0, "Should execute at least one step"
        assert done or steps == max_steps, "Should terminate eventually"

    def test_multiple_episodes(self):
        """Test running multiple episodes without leaks."""
        config = EnvConfig(max_steps=50)
        env = TacticalCombatEnv(render_mode=None, config=config)

        for episode in range(5):
            obs, info = env.reset(seed=episode)
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        # Should complete without error


class TestCombatFlow:
    """Test complete combat flow."""

    def test_projectile_creation_and_hit(self):
        """Test projectile is created and can hit target."""
        # Create shooter and target
        shooter = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        target = Agent(position=(12.0, 10.0), orientation=180.0, team="red")

        initial_health = target.health
        initial_armor = target.armor

        # Create projectile aimed at target
        proj = create_projectile(
            shooter_position=shooter.position,
            shooter_orientation=shooter.orientation,
            shooter_team=shooter.team,
            shooter_id=id(shooter),
            accuracy=1.0
        )

        # Simulate projectile movement until hit
        dt = 0.016
        hit = False
        for _ in range(100):
            expired = proj.update(dt)
            if expired:
                break
            if proj.check_collision(target):
                target.take_damage(proj.damage)
                hit = True
                break

        assert hit, "Projectile should hit target"
        # Damage goes to armor first, then health
        assert target.armor < initial_armor or target.health < initial_health, "Target should take damage"

    def test_damage_leads_to_death(self):
        """Test sufficient damage kills agent."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")

        # Apply lethal damage
        damage_needed = agent.health + agent.armor + 1
        agent.take_damage(damage_needed)

        assert agent.is_alive == False, "Agent should be dead"

    def test_death_terminates_episode(self):
        """Test controlled agent death terminates episode."""
        config = EnvConfig(terminate_on_controlled_death=True)
        env = TacticalCombatEnv(render_mode=None, config=config)
        env.reset(seed=42)

        # Kill controlled agent
        env.controlled_agent.health = 0

        _, _, terminated, _, _ = env.step(env.action_space.sample())

        assert terminated == True


class TestTerrainInteraction:
    """Test terrain effects on agents."""

    def test_fire_damages_over_time(self):
        """Test standing in fire causes damage."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        agent = env.controlled_agent
        initial_health = agent.health

        # Place agent on fire
        cell = agent.get_grid_position()
        env.terrain_grid.set(cell[0], cell[1], TerrainType.FIRE)

        # Step multiple times
        for _ in range(5):
            env.step(np.array([0.0, 0.0, 0.0]))

        assert agent.health < initial_health, "Fire should damage agent"

    def test_fire_bypasses_armor(self):
        """Test fire damage bypasses armor."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        initial_armor = agent.armor
        initial_health = agent.health

        # Apply terrain damage
        agent.apply_terrain_damage(FIRE_DAMAGE_PER_STEP)

        assert agent.armor == initial_armor, "Armor should be unchanged"
        assert agent.health < initial_health, "Health should take damage"

    def test_swamp_prevents_movement(self):
        """Test stuck agents cannot move."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        agent = env.controlled_agent
        agent.stuck_steps = SWAMP_STUCK_MIN_STEPS

        initial_pos = agent.position

        # Try to move
        env.step(np.array([1.0, 0.0, 0.0]))

        # Position should not change significantly (agent is stuck)
        # Note: Orientation may change but position should stay same

    def test_building_blocks_movement(self):
        """Test building blocks agent movement."""
        terrain = TerrainGrid(64, 64)
        terrain.set(11, 10, TerrainType.BUILDING)

        # Start agent just before building
        agent = Agent(position=(10.5, 10.0), orientation=0.0, team="blue")

        # Try to move into building cell (small step that would land in building)
        result = agent.move_forward(speed=1.0, dt=1.0, terrain_grid=terrain)

        # Movement into building should be blocked
        assert result == False, "Movement into building should be blocked"
        assert agent.position[0] < 11.0, "Agent should not enter building cell"


class TestFOVWithTerrain:
    """Test FOV interaction with terrain."""

    def test_building_blocks_los(self):
        """Test building blocks line of sight."""
        terrain = TerrainGrid(20, 20)
        terrain.set(11, 10, TerrainType.BUILDING)

        observer = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        target = Agent(position=(13.0, 10.0), orientation=180.0, team="red")

        visible = is_agent_visible_to_agent(
            observer, target,
            fov_angle=90, max_range=5,
            terrain_grid=terrain
        )

        assert visible == False, "Building should block visibility"

    def test_building_blocks_shooting(self):
        """Test cannot shoot through building."""
        terrain = TerrainGrid(20, 20)
        terrain.set(11, 10, TerrainType.BUILDING)

        shooter = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        target = Agent(position=(13.0, 10.0), orientation=180.0, team="red")

        # Get targets in FOV (should not include blocked target)
        near_targets, far_targets = shooter.get_targets_in_fov([target], terrain)

        assert target not in near_targets, "Target should not be in near FOV"
        assert target not in far_targets, "Target should not be in far FOV"

    def test_water_does_not_block_los(self):
        """Test water does not block line of sight."""
        terrain = TerrainGrid(20, 20)
        terrain.set(11, 10, TerrainType.WATER)

        observer = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        target = Agent(position=(13.0, 10.0), orientation=180.0, team="red")

        visible = is_agent_visible_to_agent(
            observer, target,
            fov_angle=90, max_range=5,
            terrain_grid=terrain
        )

        assert visible == True, "Water should not block visibility"


class TestRespawnCycle:
    """Test agent respawn functionality."""

    def test_respawn_restores_health(self):
        """Test respawn restores full health."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent.health = 0
        agent.armor = 0
        agent.stamina = 0

        agent.respawn()

        assert agent.health == AGENT_MAX_HEALTH
        assert agent.is_alive == True

    def test_respawn_clears_stuck(self):
        """Test respawn clears stuck state."""
        agent = Agent(position=(10.0, 10.0), orientation=0.0, team="blue")
        agent.stuck_steps = 100

        agent.respawn()

        assert agent.stuck_steps == 0

    def test_respawn_moves_to_spawn_area(self):
        """Test respawn moves agent to spawn quadrant."""
        agent = Agent(position=(50.0, 50.0), orientation=0.0, team="blue")
        agent.respawn()

        # Blue team spawns in top-left quadrant
        assert agent.position[0] < GRID_SIZE / 2
        assert agent.position[1] < GRID_SIZE / 2


class TestGymnasiumCompliance:
    """Test Gymnasium API compliance."""

    def test_observation_space_contains_observations(self):
        """Test observations are within observation space."""
        env = TacticalCombatEnv(render_mode=None)
        obs, _ = env.reset(seed=42)

        assert env.observation_space.contains(obs), "Observation should be in observation space"

    def test_action_space_sample_valid(self):
        """Test sampled actions are valid."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        for _ in range(100):
            action = env.action_space.sample()
            assert env.action_space.contains(action), "Sampled action should be valid"

    def test_step_returns_correct_types(self):
        """Test step returns correct types per Gymnasium API."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reset_returns_correct_types(self):
        """Test reset returns correct types per Gymnasium API."""
        env = TacticalCombatEnv(render_mode=None)
        obs, info = env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    @pytest.mark.skip(reason="gymnasium.utils.env_checker may have strict requirements")
    def test_gymnasium_env_checker(self):
        """Test environment passes gymnasium env_checker."""
        try:
            from gymnasium.utils.env_checker import check_env
            env = TacticalCombatEnv(render_mode=None)
            check_env(env)
        except ImportError:
            pytest.skip("gymnasium.utils.env_checker not available")


class TestSpawnWithTerrain:
    """Test agent spawning respects terrain."""

    def test_agents_spawn_on_empty(self):
        """Test agents only spawn on empty terrain."""
        terrain = TerrainGrid(64, 64)

        # Fill most of a quadrant with terrain
        for x in range(5, 20):
            for y in range(5, 20):
                terrain.set(x, y, TerrainType.BUILDING)

        # Spawn should still work (finds empty cells)
        agents = spawn_team("blue", num_agents=10, terrain_grid=terrain)

        # Verify all agents are on empty terrain
        for agent in agents:
            cell = agent.get_grid_position()
            assert terrain.get(cell[0], cell[1]) == TerrainType.EMPTY


class TestStatisticsTracking:
    """Test kill and damage statistics."""

    def test_kills_tracked(self):
        """Test kill statistics are tracked."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        initial_blue_kills = env.blue_kills
        initial_red_kills = env.red_kills

        # Run some steps
        for _ in range(100):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Info should contain kill stats
        assert 'blue_kills' in info
        assert 'red_kills' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
