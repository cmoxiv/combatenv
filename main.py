"""
Main entry point for the Grid-World Multi-Agent Tactical Simulation.

This module demonstrates the Gymnasium-compatible environment with random
agent control for testing purposes.

Usage:
    source ~/.venvs/pg/bin/activate
    python main.py

Controls:
    - ESC: Exit simulation
    - ` (backtick): Toggle debug overlay
    - F: Toggle FOV overlay
    - G: Toggle strategic grid
    - ? (shift+/): Toggle keybindings help
"""

import sys
from combatenv import GridWorld
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
    # Create environment with wrapper stack
    env = create_visual_env(max_steps=2000)

    # Reset environment
    observation, info = env.reset(seed=42)
    blue_alive = info.get('blue_alive', NUM_AGENTS_PER_TEAM)
    red_alive = info.get('red_alive', NUM_AGENTS_PER_TEAM)
    print(f"Environment created with {blue_alive} blue and {red_alive} red agents")
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")

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
            reason = "terminated" if terminated else "truncated"
            step_count = info.get('step_count', 0)
            blue_kills = info.get('blue_kills', 0)
            red_kills = info.get('red_kills', 0)
            blue_alive = info.get('blue_alive', 0)
            red_alive = info.get('red_alive', 0)

            print(f"Episode {episode} {reason} at step {step_count}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Blue kills: {blue_kills}, Red kills: {red_kills}")
            print(f"  Blue alive: {blue_alive}, Red alive: {red_alive}")

            # Reset for next episode
            total_reward = 0.0
            observation, info = env.reset()

    # Cleanup
    env.close()
    print(f"\nSimulation ended after {episode} episodes")
    sys.exit()


if __name__ == "__main__":
    main()
