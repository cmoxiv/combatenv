"""
Integration test for the rl_student Q-learning package.

This script verifies that all components work together correctly.

Usage:
    python -m rl_student.test_integration
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    from rl_student import create_wrapped_env, QTableManager
    from rl_student.wrappers import (
        MultiAgentWrapper,
        DiscreteObservationWrapper,
        DiscreteActionWrapper
    )
    print("  All imports successful!")


def test_environment_creation():
    """Test environment creation and wrapper stack."""
    print("\nTesting environment creation...")
    from rl_student import create_wrapped_env

    env = create_wrapped_env(render_mode=None, max_steps=10)
    print(f"  State space: {env.n_states} states")
    print(f"  Action space: {env.n_actions} actions")

    assert env.n_states == 2048, f"Expected 2048 states, got {env.n_states}"
    assert env.n_actions == 8, f"Expected 8 actions, got {env.n_actions}"
    print("  Environment creation successful!")

    return env


def test_reset_and_observations(env):
    """Test environment reset and observation generation."""
    print("\nTesting reset and observations...")

    states, info = env.reset(seed=42)

    print(f"  Number of agents: {len(states)}")
    print(f"  Sample state (agent 0): {states[0]}")
    print(f"  Info keys: {list(info.keys())}")

    assert len(states) == 200, f"Expected 200 agents, got {len(states)}"
    assert all(0 <= s < 2048 for s in states.values()), "States out of range"

    print("  Reset and observations successful!")
    return states, info


def test_qtable_manager():
    """Test Q-table manager creation and operations."""
    print("\nTesting QTableManager...")
    from rl_student import QTableManager

    # Test shared mode
    q_shared = QTableManager(n_states=2048, n_actions=8, mode="shared")
    print(f"  Shared Q-table created")

    # Test action selection
    action = q_shared.get_action(agent_idx=0, state=100, training=True)
    assert 0 <= action < 8, f"Invalid action: {action}"
    print(f"  Action selection works: {action}")

    # Test Q-value update
    q_shared.update(
        agent_idx=0,
        state=100,
        action=action,
        reward=1.0,
        next_state=200,
        done=False
    )
    print(f"  Q-value update works")

    print("  QTableManager successful!")
    return q_shared


def test_training_step(env, q_manager):
    """Test a complete training step."""
    print("\nTesting training step...")

    # Reset
    states, _ = env.reset(seed=123)

    # Get actions
    actions = q_manager.get_actions(states, training=True)
    print(f"  Actions generated for {len(actions)} agents")

    # Step
    next_states, rewards, terminated, truncated, info = env.step(actions)
    print(f"  Step completed: terminated={terminated}, truncated={truncated}")
    print(f"  Blue alive: {info['blue_alive']}, Red alive: {info['red_alive']}")

    # Update Q-values
    done = terminated or truncated
    q_manager.batch_update(states, actions, rewards, next_states, done)
    print(f"  Q-updates performed: {q_manager.total_updates}")

    print("  Training step successful!")


def test_state_decoding():
    """Test state decoding for debugging."""
    print("\nTesting state decoding...")
    from rl_student.wrappers import DiscreteObservationWrapper
    from combatenv import TacticalCombatEnv, EnvConfig
    from rl_student.wrappers import MultiAgentWrapper

    # Create minimal wrapped env
    base_env = TacticalCombatEnv(render_mode=None)
    multi_env = MultiAgentWrapper(base_env)
    discrete_env = DiscreteObservationWrapper(multi_env)

    # Test decoding
    state_info = discrete_env.decode_state(1234)
    print(f"  State 1234 decodes to: {state_info}")

    assert "health" in state_info
    assert "enemy_direction" in state_info
    print("  State decoding successful!")


def test_action_wrapper():
    """Test action wrapper functionality."""
    print("\nTesting action wrapper...")
    from rl_student.wrappers import DiscreteActionWrapper
    import numpy as np

    # Create a mock environment with the wrapper
    class MockEnv:
        pass

    # Test action mapping directly
    for action_idx in range(8):
        continuous = DiscreteActionWrapper.ACTION_MAP[action_idx]
        name = DiscreteActionWrapper.ACTION_NAMES[action_idx]
        print(f"  Action {action_idx}: {name} -> {continuous}")

    print("  Action wrapper successful!")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("rl_student Integration Tests")
    print("=" * 60)

    try:
        test_imports()
        env = test_environment_creation()
        test_reset_and_observations(env)
        q_manager = test_qtable_manager()
        test_training_step(env, q_manager)
        test_state_decoding()
        test_action_wrapper()

        print("\n" + "=" * 60)
        print("All integration tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
