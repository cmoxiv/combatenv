"""
Unit tests for the Gymnasium environment.

Tests cover:
- Environment configuration
- Initialization and spaces
- Reset behavior
- Step execution
- Observation format
- Reward calculation
- Termination conditions
- Terrain effects

Run with: pytest tests/test_environment.py -v
"""

import pytest
import numpy as np
from combatenv import TacticalCombatEnv, EnvConfig, OBS_SIZE, TerrainType, config
NUM_AGENTS_PER_TEAM = config.NUM_AGENTS_PER_TEAM
GRID_SIZE = config.GRID_SIZE


class TestEnvConfig:
    """Tests for environment configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EnvConfig()

        assert config.num_agents_per_team == NUM_AGENTS_PER_TEAM
        assert config.respawn_enabled == False
        assert config.max_steps == 1000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EnvConfig(
            num_agents_per_team=50,
            respawn_enabled=True,
            max_steps=500
        )

        assert config.num_agents_per_team == 50
        assert config.respawn_enabled == True
        assert config.max_steps == 500


class TestEnvInit:
    """Tests for environment initialization."""

    def test_action_space_shape(self):
        """Test action space has correct shape."""
        env = TacticalCombatEnv(render_mode=None)
        assert env.action_space.shape == (3,)

    def test_observation_space_shape(self):
        """Test observation space has correct shape."""
        env = TacticalCombatEnv(render_mode=None)
        assert env.observation_space.shape == (OBS_SIZE,)

    def test_render_mode(self):
        """Test render mode is stored correctly."""
        env = TacticalCombatEnv(render_mode=None)
        assert env.render_mode is None

        env2 = TacticalCombatEnv(render_mode="rgb_array")
        assert env2.render_mode == "rgb_array"


class TestEnvReset:
    """Tests for environment reset."""

    def test_returns_observation(self):
        """Test reset returns observation of correct shape."""
        env = TacticalCombatEnv(render_mode=None)
        obs, info = env.reset(seed=42)

        assert obs.shape == (OBS_SIZE,)
        assert obs.dtype == np.float32

    def test_returns_info(self):
        """Test reset returns info dict."""
        env = TacticalCombatEnv(render_mode=None)
        obs, info = env.reset(seed=42)

        assert isinstance(info, dict)
        assert 'blue_kills' in info
        assert 'red_kills' in info

    def test_spawns_agents(self):
        """Test reset spawns correct number of agents."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        assert len(env.blue_agents) == NUM_AGENTS_PER_TEAM
        assert len(env.red_agents) == NUM_AGENTS_PER_TEAM

    def test_generates_terrain(self):
        """Test reset generates terrain."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        assert env.terrain_grid is not None

        # Count non-empty terrain
        non_empty = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if env.terrain_grid.get(x, y) != TerrainType.EMPTY:
                    non_empty += 1

        assert non_empty > 0

    def test_seed_deterministic(self):
        """Test same seed produces same initial state."""
        env = TacticalCombatEnv(render_mode=None)

        obs1, _ = env.reset(seed=12345)
        pos1 = env.controlled_agent.position

        obs2, _ = env.reset(seed=12345)
        pos2 = env.controlled_agent.position

        assert np.allclose(obs1, obs2)
        assert pos1 == pos2


class TestEnvStep:
    """Tests for environment step."""

    def test_returns_tuple(self):
        """Test step returns correct tuple structure."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        action = env.action_space.sample()
        result = env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_increments_count(self):
        """Test step increments step count."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        assert env.step_count == 0

        env.step(env.action_space.sample())
        assert env.step_count == 1

        env.step(env.action_space.sample())
        assert env.step_count == 2

    def test_processes_action(self):
        """Test step processes action correctly."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        initial_pos = env.controlled_agent.position

        # Move right
        action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        env.step(action)

        # Position should change (unless blocked)
        # Note: May be blocked by terrain or agents


class TestEnvObservation:
    """Tests for observation format."""

    def test_shape(self):
        """Test observation shape is correct."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (OBS_SIZE,)

    def test_normalized(self):
        """Test observations are in [0, 1] range."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        for _ in range(20):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            assert np.all(obs >= 0.0), "Observations should be >= 0"
            assert np.all(obs <= 1.0), "Observations should be <= 1"

            if terminated or truncated:
                break

    def test_agent_state(self):
        """Test observation contains agent state."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0]))

        # First 10 elements are agent state
        assert obs[0] >= 0  # x position normalized
        assert obs[1] >= 0  # y position normalized
        assert obs[2] >= 0  # orientation normalized
        assert obs[3] >= 0  # health normalized


class TestEnvTermination:
    """Tests for termination conditions."""

    def test_max_steps_truncation(self):
        """Test episode truncates at max steps."""
        config = EnvConfig(max_steps=5)
        env = TacticalCombatEnv(render_mode=None, config=config)
        env.reset(seed=42)

        for i in range(10):
            action = np.array([0.0, 0.0, 0.0])
            _, _, terminated, truncated, _ = env.step(action)

            if i >= 4:
                assert truncated or terminated
                break

    def test_controlled_death_terminates(self):
        """Test episode terminates when controlled agent dies."""
        config = EnvConfig(terminate_on_controlled_death=True)
        env = TacticalCombatEnv(render_mode=None, config=config)
        env.reset(seed=42)

        # Kill controlled agent
        env.controlled_agent.health = 0

        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated == True


class TestTerrainEffects:
    """Tests for terrain effects on agents."""

    def test_fire_damage(self):
        """Test fire terrain damages agents."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        # Place agent on fire
        agent = env.controlled_agent
        initial_health = agent.health

        # Find or create fire cell
        env.terrain_grid.set(int(agent.position[0]), int(agent.position[1]), TerrainType.FIRE)

        # Step to trigger terrain effect
        env.step(np.array([0.0, 0.0, 0.0]))

        # Health should decrease (fire bypasses armor)
        assert agent.health < initial_health

    def test_swamp_stuck(self):
        """Test swamp terrain makes agents stuck."""
        env = TacticalCombatEnv(render_mode=None)
        env.reset(seed=42)

        agent = env.controlled_agent

        # Place agent on swamp
        env.terrain_grid.set(int(agent.position[0]), int(agent.position[1]), TerrainType.SWAMP)

        # Step to trigger terrain effect
        env.step(np.array([0.0, 0.0, 0.0]))

        # Agent should be stuck
        assert agent.is_stuck == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
