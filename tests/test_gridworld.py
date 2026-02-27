"""Tests for GridWorld base environment."""

import numpy as np
import pytest

from combatenv.gridworld import GridWorld
from combatenv.config import GRID_SIZE


class TestGridWorld:
    """Tests for GridWorld base environment."""

    def test_init_default(self):
        """Test default initialization."""
        env = GridWorld()
        assert env.grid_size == GRID_SIZE
        assert env.render_mode is None
        assert env.step_count == 0

    def test_init_custom_size(self):
        """Test initialization with custom grid size."""
        env = GridWorld(grid_size=32)
        assert env.grid_size == 32

    def test_reset(self):
        """Test reset returns valid observation and info."""
        env = GridWorld()
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2,)
        assert isinstance(info, dict)
        assert env.step_count == 0

    def test_reset_with_seed(self):
        """Test reset with seed for reproducibility."""
        env = GridWorld()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_step(self):
        """Test step returns valid tuple."""
        env = GridWorld()
        env.reset()

        action = np.array([0.5, -0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.step_count == 1

    def test_step_increments_counter(self):
        """Test step counter increments correctly."""
        env = GridWorld()
        env.reset()

        for i in range(10):
            env.step(np.zeros(2, dtype=np.float32))
            assert env.step_count == i + 1

    def test_action_space(self):
        """Test action space is valid Box."""
        env = GridWorld()
        assert env.action_space.shape == (2,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_observation_space(self):
        """Test observation space is valid Box."""
        env = GridWorld()
        assert env.observation_space.shape == (2,)
        assert env.observation_space.low[0] == 0.0
        assert env.observation_space.high[0] == 1.0

    def test_render_headless(self):
        """Test render in headless mode returns None."""
        env = GridWorld(render_mode=None)
        env.reset()
        result = env.render()
        assert result is None

    def test_close(self):
        """Test close doesn't raise errors."""
        env = GridWorld()
        env.reset()
        env.close()  # Should not raise


class TestGridWorldGymnasiumAPI:
    """Test Gymnasium API compliance."""

    def test_env_check(self):
        """Test environment passes Gymnasium's env_checker."""
        from gymnasium.utils.env_checker import check_env

        env = GridWorld()
        # This will raise if the env doesn't comply
        check_env(env, skip_render_check=True)

    def test_sample_action(self):
        """Test action space sampling works."""
        env = GridWorld()
        action = env.action_space.sample()
        assert action.shape == (2,)
        assert -1.0 <= action[0] <= 1.0
        assert -1.0 <= action[1] <= 1.0

    def test_sample_observation(self):
        """Test observation space sampling works."""
        env = GridWorld()
        obs = env.observation_space.sample()
        assert obs.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
