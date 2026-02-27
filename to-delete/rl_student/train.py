"""
Training script for multi-agent tabular Q-learning on combatenv.

This script trains Q-learning agents to play the tactical combat simulation.
All 200 agents (both teams) are controlled by the Q-learning policy.

Two training modes are available:
    - shared: All agents share one Q-table (faster learning, ~64 KB)
    - individual: Each agent has its own Q-table (more specialized, ~13 MB)

Usage:
    # Train with shared Q-table (default)
    python -m rl_student.train --mode shared --episodes 1000

    # Train with individual Q-tables
    python -m rl_student.train --mode individual --episodes 1000

    # Train with rendering (slower but visual)
    python -m rl_student.train --render --episodes 100

    # Evaluate a trained model
    python -m rl_student.train --evaluate q_table_shared.pkl --render
"""

import argparse
import time
from typing import Optional

import numpy as np

from rl_student import create_wrapped_env, QTableManager


def train(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 500,
    mode: str = "shared",
    render: bool = False,
    reward_shaping: bool = False,
    save_every: int = 100,
    log_every: int = 10,
    save_path: Optional[str] = None
) -> QTableManager:
    """
    Train agents using tabular Q-learning.

    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        mode: "shared" or "individual" Q-tables
        render: Whether to render during training
        reward_shaping: Whether to add reward shaping (move to enemy corner)
        save_every: Save Q-table every N episodes
        log_every: Log progress every N episodes
        save_path: Path to save final Q-table (default: q_table_{mode}.pkl)

    Returns:
        Trained QTableManager instance
    """
    print("=" * 60)
    print("Multi-Agent Q-Learning Training")
    print("=" * 60)

    # Create wrapped environment
    render_mode = "human" if render else None
    env = create_wrapped_env(
        render_mode=render_mode,
        max_steps=max_steps_per_episode,
        reward_shaping=reward_shaping
    )

    # Get state/action space sizes from wrappers
    n_states = env.n_states  # From DiscreteObservationWrapper
    n_actions = env.n_actions  # From DiscreteActionWrapper

    print(f"State space: {n_states} states")
    print(f"Action space: {n_actions} actions")
    print(f"Mode: {mode}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    # Initialize Q-table manager
    q_manager = QTableManager(
        n_states=n_states,
        n_actions=n_actions,
        n_agents=200,
        mode=mode,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    blue_wins = 0
    red_wins = 0

    print("\nStarting training...")
    print("Press Shift+Q to stop training early\n")
    start_time = time.time()
    user_quit = False

    for episode in range(num_episodes):
        if user_quit:
            break

        # Reset environment
        states, info = env.reset(seed=episode)

        episode_reward = 0.0
        step = 0
        done = False

        while not done and step < max_steps_per_episode:
            # Select actions for all agents
            actions = q_manager.get_actions(states, training=True)

            # Execute actions
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Update Q-values
            q_manager.batch_update(states, actions, rewards, next_states, done)

            # Track rewards
            episode_reward += sum(rewards.values())

            states = next_states
            step += 1

            if render:
                # Process pygame events (keyboard input, window close)
                base_env = env.unwrapped  # Unwrap to get TacticalCombatEnv
                if not base_env.process_events():
                    user_quit = True
                    print("\nUser requested quit...")
                    break
                env.render()

        if user_quit:
            break

        # Episode complete
        q_manager.decay_epsilon()

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        # Track wins
        if info["blue_alive"] > 0 and info["red_alive"] == 0:
            blue_wins += 1
        elif info["red_alive"] > 0 and info["blue_alive"] == 0:
            red_wins += 1

        # Logging
        if (episode + 1) % log_every == 0:
            avg_reward = np.mean(episode_rewards[-log_every:])
            avg_length = np.mean(episode_lengths[-log_every:])
            elapsed = time.time() - start_time

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Avg length: {avg_length:.1f} steps")
            print(f"  Epsilon: {q_manager.epsilon:.4f}")
            print(f"  Blue wins: {blue_wins}, Red wins: {red_wins}")
            print(f"  Time elapsed: {elapsed:.1f}s")

        # Save checkpoint
        if (episode + 1) % save_every == 0:
            checkpoint_path = f"q_table_{mode}_ep{episode + 1}.pkl"
            q_manager.save(checkpoint_path)

    # Training complete
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Episodes per second: {num_episodes / elapsed:.2f}")
    print(f"Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final epsilon: {q_manager.epsilon:.4f}")
    print(f"Blue wins: {blue_wins}, Red wins: {red_wins}")

    # Print Q-table statistics
    stats = q_manager.get_statistics()
    print(f"\nQ-Table Statistics:")
    print(f"  Coverage: {stats['coverage'] * 100:.1f}% of state-action pairs visited")
    print(f"  Mean Q-value: {stats['mean_q']:.4f}")
    print(f"  Max Q-value: {stats['max_q']:.4f}")
    print(f"  Min Q-value: {stats['min_q']:.4f}")

    # Save final model
    if save_path is None:
        save_path = f"q_table_{mode}.pkl"
    q_manager.save(save_path)

    env.close()
    return q_manager


def evaluate(
    q_manager: QTableManager,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = True
) -> None:
    """
    Evaluate trained Q-learning agents.

    Args:
        q_manager: Trained QTableManager
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render
    """
    print("\n" + "=" * 60)
    print("Evaluating Q-Learning Agents")
    print("=" * 60)
    if render:
        print("Press Shift+Q to stop early\n")

    render_mode = "human" if render else None
    env = create_wrapped_env(render_mode=render_mode, max_steps=max_steps)

    episode_rewards = []
    blue_wins = 0
    red_wins = 0
    user_quit = False

    for episode in range(num_episodes):
        if user_quit:
            break

        states, info = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            # Greedy action selection (no exploration)
            actions = q_manager.get_actions(states, training=False)
            next_states, rewards, terminated, truncated, info = env.step(actions)

            total_reward += sum(rewards.values())
            states = next_states
            done = terminated or truncated
            step += 1

            if render:
                # Process pygame events (keyboard input, window close)
                base_env = env.unwrapped  # Unwrap to get TacticalCombatEnv
                if not base_env.process_events():
                    user_quit = True
                    print("\nUser requested quit...")
                    break
                env.render()

        if user_quit:
            break

        episode_rewards.append(total_reward)

        # Track wins
        if info["blue_alive"] > 0 and info["red_alive"] == 0:
            blue_wins += 1
            result = "Blue wins!"
        elif info["red_alive"] > 0 and info["blue_alive"] == 0:
            red_wins += 1
            result = "Red wins!"
        else:
            result = "Draw"

        print(f"Episode {episode + 1}: reward={total_reward:.2f}, "
              f"steps={step}, {result}")

    print("\n" + "-" * 40)
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Blue wins: {blue_wins}/{num_episodes}")
    print(f"Red wins: {red_wins}/{num_episodes}")
    print(f"Draws: {num_episodes - blue_wins - red_wins}/{num_episodes}")

    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Q-learning agents on combatenv tactical combat simulation"
    )

    parser.add_argument(
        "--mode",
        choices=["shared", "individual"],
        default="shared",
        help="Q-table mode: 'shared' (all agents share one Q-table) or "
             "'individual' (each agent has its own Q-table)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during training (slower but visual)"
    )
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        help="Enable reward shaping (incentivize movement to enemy corner)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N episodes (default: 100)"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N episodes (default: 10)"
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        metavar="PATH",
        help="Path to Q-table file to evaluate (skips training)"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )

    args = parser.parse_args()

    if args.evaluate:
        # Load and evaluate
        q_manager = QTableManager(
            n_states=2048,
            n_actions=8,
            n_agents=200,
            mode=args.mode
        )
        q_manager.load(args.evaluate)
        evaluate(
            q_manager,
            num_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            render=args.render
        )
    else:
        # Train
        q_manager = train(
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            mode=args.mode,
            render=args.render,
            reward_shaping=args.reward_shaping,
            save_every=args.save_every,
            log_every=args.log_every
        )


if __name__ == "__main__":
    main()
