"""
rl_student - Tabular Q-Learning for combatenv

This package provides wrappers and utilities for training Q-learning agents
on the combatenv tactical combat simulation.

Example (multi-agent):
    from rl_student import create_wrapped_env, MultiAgentQManager

    env = create_wrapped_env(render_mode=None)
    q_manager = MultiAgentQManager(n_agents=200, n_states=2048, n_actions=10)

    # Training loop
    observations, _ = env.reset()  # Dict[int, int]
    actions = q_manager.get_actions(observations)  # Dict[int, int]
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    q_manager.update(observations, actions, rewards, next_obs, terminated)
"""

from rl_student.wrappers import (
    MultiAgentWrapper,
    RewardWrapper,
    DiscreteObservationWrapper,
    DiscreteActionWrapper,
)
from rl_student.q_table import QTableManager
from rl_student.circular_q_table import CircularQTable
from rl_student.multi_agent_q import MultiAgentQManager


def create_wrapped_env(render_mode=None, max_steps=1000, reward_shaping=False):
    """
    Create a fully wrapped environment for multi-agent Q-learning.

    Args:
        render_mode: "human" for visualization, None for headless
        max_steps: Maximum steps per episode
        reward_shaping: If True, add RewardWrapper to incentivize
                       movement toward enemy corner

    Returns:
        Wrapped environment with discrete state/action spaces.
        - Observations: Dict[int, int] mapping agent_idx -> discrete state (0-2047)
        - Actions: Dict[int, int] mapping agent_idx -> discrete action (0-9)
        - Rewards: Dict[int, float] mapping agent_idx -> reward
    """
    from combatenv import TacticalCombatEnv, EnvConfig

    config = EnvConfig(
        max_steps=max_steps,
        terminate_on_team_elimination=True,
        terminate_on_controlled_death=False,
    )

    env = TacticalCombatEnv(render_mode=render_mode, config=config)
    env = MultiAgentWrapper(env)

    if reward_shaping:
        env = RewardWrapper(env)

    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)

    return env


__all__ = [
    "MultiAgentWrapper",
    "RewardWrapper",
    "DiscreteObservationWrapper",
    "DiscreteActionWrapper",
    "QTableManager",
    "CircularQTable",
    "MultiAgentQManager",
    "create_wrapped_env",
]
