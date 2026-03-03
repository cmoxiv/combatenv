"""
Multi-agent training script for combat environment.

Trains N agents (200 by default) with individual Q-tables using:
- Discrete observation space (2,048 states)
- Discrete action space (10 actions)
- Individual Q-tables per agent (circular buffer with limited memory)
- Epsilon-greedy exploration with decay

Usage:
    python train_multi_agent.py
"""

import time
from movement_training_env import MultiAgentDiscreteEnv


def train(
    num_episodes: int = 100,
    max_steps: int = 500,
    render: bool = True,
    render_every: int = 10,
    report_every: int = 5,
    max_q_entries: int = 500,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.05,
):
    """
    Train multiple agents in combat environment.

    Args:
        num_episodes: Number of training episodes
        max_steps: Max steps per episode
        render: Whether to render
        render_every: Render every N episodes
        max_q_entries: Q-table memory limit per agent
        epsilon_start: Initial exploration rate
        epsilon_decay: Epsilon decay per episode
        epsilon_min: Minimum epsilon
    """
    # Create environment
    render_mode = "human" if render else None
    env = MultiAgentDiscreteEnv(
        render_mode=render_mode,
        max_q_entries=max_q_entries,
        max_steps=max_steps,
        epsilon=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    print(f"\nStarting multi-agent training for {num_episodes} episodes...")
    print(f"State space: {env.n_states}, Action space: {env.n_actions}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Q-table max entries per agent: {max_q_entries}")
    print("-" * 60)

    # Training stats
    total_rewards_per_episode = []
    blue_wins = 0
    red_wins = 0

    for episode in range(num_episodes):
        observations, info = env.reset()
        n_agents = len(observations)
        episode_rewards = {i: 0.0 for i in range(n_agents)}
        steps = 0

        for step in range(max_steps):
            # Get actions from Q-tables (epsilon-greedy)
            actions = env.get_actions(observations, training=True)

            # Take step
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            # Update Q-tables
            done = terminated or truncated
            env.update(observations, actions, rewards, next_obs, done)

            # Share knowledge across agents every step
            env.share_knowledge()

            # Track rewards
            for agent_idx, reward in rewards.items():
                episode_rewards[agent_idx] += reward

            steps += 1
            observations = next_obs

            # Render if enabled
            if render and (episode % render_every == 0 or episode == num_episodes - 1):
                if not env.render():
                    print("\nUser requested exit.")
                    env.close()
                    return env.q_manager

            if done:
                break

        # Decay epsilon after each episode
        env.decay_epsilon()

        # Track stats
        total_reward = sum(episode_rewards.values())
        total_rewards_per_episode.append(total_reward)

        # Track wins
        blue_alive = info.get('blue_alive', 0)
        red_alive = info.get('red_alive', 0)
        if blue_alive > 0 and red_alive == 0:
            blue_wins += 1
        elif red_alive > 0 and blue_alive == 0:
            red_wins += 1

        # Print progress
        if (episode + 1) % report_every == 0 or episode == 0:
            avg_reward = sum(total_rewards_per_episode[-report_every:]) / min(report_every, len(total_rewards_per_episode))
            stats = env.get_statistics()
            print(
                f"Episode {episode + 1:3d} | "
                f"Total Reward: {total_reward:8.1f} | "
                f"Avg(10): {avg_reward:8.1f} | "
                f"Steps: {steps:3d} | "
                f"ε: {env.epsilon:.3f} | "
                f"Q-entries: {stats.get('total_entries_used', 0)}"
            )

    # Final summary
    print("-" * 60)
    print(f"Training complete!")
    print(f"  Episodes: {num_episodes}")
    print(f"  Blue wins: {blue_wins}/{num_episodes} ({100*blue_wins/num_episodes:.1f}%)")
    print(f"  Red wins: {red_wins}/{num_episodes} ({100*red_wins/num_episodes:.1f}%)")
    print(f"  Draws: {num_episodes - blue_wins - red_wins}/{num_episodes}")
    print(f"  Avg reward (last 10): {sum(total_rewards_per_episode[-10:])/10:.1f}")
    print(f"  Final epsilon: {env.epsilon:.4f}")

    stats = env.get_statistics()
    print(f"\nQ-table statistics:")
    print(f"  Agents: {stats.get('n_agents', 0)}")
    print(f"  Total entries used: {stats.get('total_entries_used', 0)}")
    print(f"  Avg entries per agent: {stats.get('avg_entries_per_agent', 0):.1f}")
    print(f"  Total updates: {stats.get('total_updates', 0)}")
    print(f"  Total evictions: {stats.get('total_evictions', 0)}")
    print(f"  Max Q-value: {stats.get('max_q', 0):.2f}")
    print(f"  Min Q-value: {stats.get('min_q', 0):.2f}")
    print(f"  Total memory: {stats.get('total_memory_kb', 0):.1f} KB")

    env.close()
    return env.q_manager


if __name__ == "__main__":
    q_manager = train(
        num_episodes=150,
        max_steps=1000,  # Very long episodes
        render=True,
        render_every=1,  # Render every episode to watch learning
        max_q_entries=2000,  # Larger Q-tables for better learning
        epsilon_start=1.0,
        epsilon_decay=0.97,
        epsilon_min=0.05,
    )
