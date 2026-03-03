"""
Training script for movement learning with Q-learning.

Trains an agent to navigate to waypoints using:
- Discrete observation space (2,048 states)
- Discrete action space (8 actions)
- Circular Q-table with limited memory
- Epsilon-greedy exploration with decay

Usage:
    python train_movement.py
"""

import time
from movement_training_env import DiscreteMovementTrainingEnv


def train(
    num_episodes: int = 100,
    max_steps: int = 500,
    render: bool = True,
    render_every: int = 10,
    max_q_entries: int = 500,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.05,
):
    """
    Train agent to navigate to waypoints.

    Args:
        num_episodes: Number of training episodes
        max_steps: Max steps per episode
        render: Whether to render
        render_every: Render every N episodes
        max_q_entries: Q-table memory limit
        epsilon_start: Initial exploration rate
        epsilon_decay: Epsilon decay per episode
        epsilon_min: Minimum epsilon
    """
    # Create environment
    render_mode = "human" if render else None
    env = DiscreteMovementTrainingEnv(
        render_mode=render_mode,
        max_q_entries=max_q_entries,
        epsilon=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"State space: {env.n_states}, Action space: {env.n_actions}")
    print(f"Q-table max entries: {max_q_entries}")
    print("-" * 60)

    # Training stats
    total_rewards = []
    episode_lengths = []
    goals_reached = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0

        for step in range(max_steps):
            # Select action using greedy policy (no random exploration)
            action = env.get_action(obs, training=False)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Update Q-table
            done = terminated or truncated
            env.update_q(obs, action, reward, next_obs, done)

            episode_reward += reward
            steps += 1
            obs = next_obs

            # Render if enabled
            if render and (episode % render_every == 0 or episode == num_episodes - 1):
                env.render()

            if done:
                break

        # Decay epsilon after each episode
        env.decay_epsilon()

        # Track stats
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        if episode_reward > 50:  # Reached goal (got +100 bonus)
            goals_reached += 1

        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
            avg_length = sum(episode_lengths[-10:]) / min(10, len(episode_lengths))
            stats = env.get_statistics()
            print(
                f"Episode {episode + 1:3d} | "
                f"Reward: {episode_reward:7.1f} | "
                f"Avg(10): {avg_reward:7.1f} | "
                f"Steps: {steps:3d} | "
                f"ε: {env.epsilon:.3f} | "
                f"Q-entries: {stats['entries_used']}"
            )

    # Final summary
    print("-" * 60)
    print(f"Training complete!")
    print(f"  Episodes: {num_episodes}")
    print(f"  Goals reached: {goals_reached}/{num_episodes} ({100*goals_reached/num_episodes:.1f}%)")
    print(f"  Avg reward (last 10): {sum(total_rewards[-10:])/10:.1f}")
    print(f"  Final epsilon: {env.epsilon:.4f}")

    stats = env.get_statistics()
    print(f"\nQ-table statistics:")
    print(f"  Entries used: {stats['entries_used']}/{stats['max_entries']}")
    print(f"  Evictions: {stats['evictions']}")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Max Q-value: {stats['max_q']:.2f}")
    print(f"  Min Q-value: {stats['min_q']:.2f}")

    env.close()
    return env.q_table


if __name__ == "__main__":
    q_table = train(
        num_episodes=100,
        max_steps=300,
        render=True,
        render_every=1,  # Render every episode to watch learning
        max_q_entries=500,
        epsilon_start=1.0,
        epsilon_decay=0.97,
        epsilon_min=0.05,
    )
