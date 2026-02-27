"""
Example: Layered Wrapper Architecture

This script demonstrates the three-level wrapper architecture:
1. Tactical Level - Agent-level control (movement, combat)
2. Operational Level - Unit-level control (waypoints, formations, boids)
3. Strategic Level - Grid-level control (4x4 strategic cells)

Each level can be used independently or stacked together.

Usage:
    python example_layered_architecture.py [tactical|operational|strategic|full]
"""

import sys
import numpy as np
from combatenv import TacticalCombatEnv, EnvConfig


def demo_tactical_level():
    """
    Tactical Level: Agent-level control.

    The TacticalWrapper provides methods for:
    - Combat execution (shooting, damage)
    - Terrain effects (fire damage, forest slowing)
    - Agent movement and actions
    - Tactical rewards (kills, damage)
    """
    from combatenv.wrappers import (
        MultiAgentWrapper,
        DiscreteObservationWrapper,
        DiscreteActionWrapper,
    )

    print("\n" + "=" * 60)
    print("TACTICAL LEVEL DEMO")
    print("=" * 60)
    print("Agent-level control with discrete observation/action spaces")

    # Create environment with tactical wrappers
    env = TacticalCombatEnv(render_mode="human", config=EnvConfig(max_steps=500))
    env = MultiAgentWrapper(env)
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)

    print(f"State space: {env.n_states} discrete states")
    print(f"Action space: {env.n_actions} discrete actions")
    print("\nActions: 0=Hold, 1=Contract, 2=Expand, 3=FlankL, 4=FlankR")
    print("         5=Shoot, 6=Think, 7=No-op")
    print("\nPress Q to exit\n")

    obs, info = env.reset(seed=42)

    for step in range(500):
        # Random actions for all agents
        actions = {idx: np.random.randint(0, env.n_actions) for idx in obs.keys()}

        obs, rewards, terminated, truncated, info = env.step(actions)
        env.render()

        # Process events
        if not env.unwrapped.process_events():
            break

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    env.close()


def demo_operational_level():
    """
    Operational Level: Unit-level control.

    The OperationalWrapper provides methods for:
    - Unit waypoint management (set_unit_waypoint, dispatch_unit)
    - Boids flocking behavior (cohesion, separation, alignment)
    - Unit formations and stances
    - Cohesion bonuses (accuracy, armor)
    """
    from combatenv.wrappers import (
        OperationalWrapper,
        OperationalDiscreteObsWrapper,
        OperationalDiscreteActionWrapper,
    )

    print("\n" + "=" * 60)
    print("OPERATIONAL LEVEL DEMO")
    print("=" * 60)
    print("Unit-level control with waypoints and formations")

    # Create environment with operational wrappers
    env = TacticalCombatEnv(render_mode="human", config=EnvConfig(max_steps=500))
    env = OperationalWrapper(env)
    env = OperationalDiscreteObsWrapper(env, team="blue")
    env = OperationalDiscreteActionWrapper(env, team="blue")

    print(f"State space: {env.n_states} states per unit")
    print(f"Action space: {env.n_actions} actions")
    print("\nActions: 0=Hold, 1-8=Waypoint directions, 9-16=Dispatch, 17-18=Stance")
    print("\nPress Q to exit\n")

    obs, info = env.reset(seed=42)

    for step in range(500):
        # Random actions for all units
        actions = {uid: np.random.randint(0, env.n_actions) for uid in obs.keys()}

        obs, rewards, terminated, truncated, info = env.step(actions)
        env.render()

        # Process events
        if not env.unwrapped.process_events():
            break

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

        # Print unit stats periodically
        if step % 100 == 0:
            for uid in list(obs.keys())[:2]:
                decoded = env.env.decode_state(obs[uid])
                print(f"Unit {uid}: {decoded}")

    env.close()


def demo_strategic_level():
    """
    Strategic Level: Grid-level control.

    The StrategicWrapper provides:
    - 4x4 strategic grid observation (terrain + occupancy)
    - Cell-based unit coordination
    - Strategic analysis (contested cells, frontline)
    """
    from combatenv.wrappers import (
        OperationalWrapper,
        StrategicWrapper,
        StrategicDiscreteObsWrapper,
        StrategicDiscreteActionWrapper,
    )

    print("\n" + "=" * 60)
    print("STRATEGIC LEVEL DEMO")
    print("=" * 60)
    print("Grid-level control with 4x4 strategic cells")

    # Create environment with strategic wrappers
    env = TacticalCombatEnv(render_mode="human", config=EnvConfig(max_steps=500))
    env = OperationalWrapper(env)
    env = StrategicWrapper(env)
    env = StrategicDiscreteObsWrapper(env)
    env = StrategicDiscreteActionWrapper(env, team="blue")

    print(f"State space: {env.n_states} strategic states")
    print(f"Action space: {env.n_actions} actions")
    print("\nActions: 0=Hold, 1-16=Focus cell, 17-20=Push dir, 21-24=Defend quadrant")
    print("\nPress Q to exit\n")

    obs, info = env.reset(seed=42)

    for step in range(500):
        # Random strategic action
        action = np.random.randint(0, env.n_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Process events
        if not env.unwrapped.process_events():
            break

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

        # Print strategic state periodically
        if step % 50 == 0:
            decoded = env.env.decode_state(obs)
            print(f"Step {step}: State={obs}, Decoded={decoded}")

    env.close()


def demo_full_stack():
    """
    Full Stack: All levels combined.

    Shows how to stack all wrappers together for a complete system.
    """
    from combatenv.wrappers import (
        OperationalWrapper,
        StrategicWrapper,
        KeybindingsWrapper,
        DebugOverlayWrapper,
    )

    print("\n" + "=" * 60)
    print("FULL STACK DEMO")
    print("=" * 60)
    print("All wrapper levels combined")

    # Create full wrapper stack
    env = TacticalCombatEnv(render_mode="human", config=EnvConfig(max_steps=1000))
    env = OperationalWrapper(env)
    env = StrategicWrapper(env)
    env = KeybindingsWrapper(env)
    env = DebugOverlayWrapper(env, show_debug=True)

    print("\nWrapper stack:")
    print("  Base: TacticalCombatEnv")
    print("  + OperationalWrapper (units, boids)")
    print("  + StrategicWrapper (4x4 grid)")
    print("  + KeybindingsWrapper (keyboard input)")
    print("  + DebugOverlayWrapper (debug info)")
    print("\nControls:")
    print("  1-8: Select blue unit")
    print("  Click: Set waypoint for selected unit")
    print("  `: Toggle debug overlay")
    print("  Q: Exit\n")

    obs, info = env.reset(seed=42)

    # Find the StrategicWrapper in the chain
    strategic_wrapper = env.env.env  # DebugOverlay -> Keybindings -> Strategic
    print(f"Strategic observation: {strategic_wrapper.get_strategic_observation().shape}")
    print(f"Occupancy board:\n{strategic_wrapper.get_occupancy_board()}")

    for step in range(1000):
        # No-op action (let agents act autonomously)
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        # KeybindingsWrapper handles rendering and input
        if not env.render():
            break

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            obs, info = env.reset()

    env.close()


def main():
    """Run the specified demo."""
    demo = sys.argv[1] if len(sys.argv) > 1 else "full"

    demos = {
        "tactical": demo_tactical_level,
        "operational": demo_operational_level,
        "strategic": demo_strategic_level,
        "full": demo_full_stack,
    }

    if demo not in demos:
        print(f"Usage: python {sys.argv[0]} [tactical|operational|strategic|full]")
        print("\nAvailable demos:")
        print("  tactical    - Agent-level discrete control")
        print("  operational - Unit-level waypoint control")
        print("  strategic   - Grid-level strategic control")
        print("  full        - All wrappers combined (default)")
        sys.exit(1)

    demos[demo]()


if __name__ == "__main__":
    main()
