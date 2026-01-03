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
    - ? (shift+/): Toggle keybindings help
"""

import sys
from combatenv import TacticalCombatEnv, EnvConfig


def main():
    """
    Run the tactical combat simulation with random agent control.

    Demonstrates the Gymnasium environment interface with visualization.
    The controlled agent takes random actions while other agents act autonomously.
    """
    # Configure environment for visualization
    config = EnvConfig(
        respawn_enabled=False,
        terminate_on_controlled_death=False,
        terminate_on_team_elimination=True,
        max_steps=2000
    )

    # Create environment with human rendering
    env = TacticalCombatEnv(render_mode="human", config=config)

    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"Environment created with {info['blue_alive']} blue and {info['red_alive']} red agents")
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
            print(f"Episode {episode} {reason} at step {info['step_count']}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Controlled agent kills: {info['controlled_kills']}")
            print(f"  Blue kills: {info['blue_kills']}, Red kills: {info['red_kills']}")
            print(f"  Blue alive: {info['blue_alive']}, Red alive: {info['red_alive']}")

            # Reset for next episode
            total_reward = 0.0
            observation, info = env.reset()

    # Cleanup
    env.close()
    print(f"\nSimulation ended after {episode} episodes")
    sys.exit()


if __name__ == "__main__":
    main()
