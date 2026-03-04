"""
Main entry point for the Grid-World Multi-Agent Tactical Simulation.

This module demonstrates the Gymnasium-compatible environment with random
agent control for testing purposes.

Usage:
    combatenv-demo
    combatenv-demo --seed 42
    combatenv-demo --map my_map.npz

Controls:
    - ESC: Exit simulation
    - ` (backtick): Toggle debug overlay
    - F: Toggle FOV overlay
    - G: Toggle strategic grid
    - ? (shift+/): Toggle keybindings help
"""

import argparse
import sys
from combatenv import GridWorld, load_map
from combatenv.wrappers import (
    AgentWrapper,
    TeamWrapper,
    BaseTerrainWrapper,
    ProjectileWrapper,
    BaseFOVWrapper,
    BaseCombatWrapper,
    MovementWrapper,
    BaseObservationWrapper,
    TerminationWrapper,
    RenderWrapper,
)
from combatenv.config import NUM_AGENTS_PER_TEAM


def create_visual_env(
    num_agents: int = NUM_AGENTS_PER_TEAM * 2,
    max_steps: int = 2000,
    autonomous_combat: bool = True,
):
    """Create a visual environment using the wrapper stack."""
    env = GridWorld(render_mode="human")
    env = AgentWrapper(env, num_agents=num_agents)
    env = BaseTerrainWrapper(env)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = ProjectileWrapper(env, friendly_fire=False)
    env = BaseFOVWrapper(env)
    env = BaseCombatWrapper(env, autonomous_combat=autonomous_combat)
    env = MovementWrapper(env, enable_wandering=True)
    env = BaseObservationWrapper(env)
    env = TerminationWrapper(
        env,
        max_steps=max_steps,
        terminate_on_controlled_death=False,
        terminate_on_team_elimination=True,
    )
    env = RenderWrapper(env)
    return env


def main():
    """
    Run the tactical combat simulation with random agent control.

    Demonstrates the Gymnasium environment interface with visualization.
    The controlled agent takes random actions while other agents act autonomously.
    """
    parser = argparse.ArgumentParser(description="Combat environment demo")
    parser.add_argument("--map", type=str, default=None, help="Path to a .npz map file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for terrain generation")
    args = parser.parse_args()

    # Build reset options
    reset_options = {}
    if args.map:
        reset_options["terrain_grid"] = load_map(args.map)
    seed = args.seed if args.seed is not None else 42

    # Create environment with wrapper stack
    env = create_visual_env(max_steps=2000)

    # Reset environment
    observation, info = env.reset(seed=seed, options=reset_options or None)
    episode = 0
    total_reward = 0.0

    # Initial render to initialize pygame
    env.render()

    while env.process_events():
        # Sample random action (replace with RL policy for training)
        action = env.action_space.sample()

        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render
        env.render()

        # Handle episode end
        if terminated or truncated:
            episode += 1
            total_reward = 0.0
            observation, info = env.reset(seed=seed, options=reset_options or None)

    env.close()
    sys.exit()


if __name__ == "__main__":
    main()
