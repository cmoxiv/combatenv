"""
Performance profiling for the tactical combat simulation.
"""

import cProfile
import pstats
import io
from pstats import SortKey

from combatenv import TacticalCombatEnv, EnvConfig


def profile_simulation():
    """Profile 100 steps of the simulation."""
    config = EnvConfig(
        respawn_enabled=False,
        terminate_on_controlled_death=False,
        terminate_on_team_elimination=False,
        max_steps=100
    )

    env = TacticalCombatEnv(render_mode=None, config=config)  # No rendering
    env.reset(seed=42)

    # Profile 100 steps
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)

    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)
    print(s.getvalue())

    env.close()


def profile_with_rendering():
    """Profile with rendering and FOV enabled."""
    config = EnvConfig(
        respawn_enabled=False,
        terminate_on_controlled_death=False,
        terminate_on_team_elimination=False,
        max_steps=50
    )

    env = TacticalCombatEnv(render_mode="human", config=config)
    env.reset(seed=42)
    env.show_fov = True  # Enable FOV visualization
    env.render()

    # Profile 50 steps with rendering and FOV
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(50):
        action = env.action_space.sample()
        env.step(action)
        env.render()

    profiler.disable()

    # Print results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)
    print(s.getvalue())

    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PROFILING WITHOUT RENDERING")
    print("=" * 60)
    profile_simulation()

    print("\n" + "=" * 60)
    print("PROFILING WITH RENDERING")
    print("=" * 60)
    profile_with_rendering()
