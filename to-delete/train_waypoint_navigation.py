"""
Waypoint Navigation Training Script.

This script trains units to navigate toward waypoints using Q-learning.
Both blue and red teams learn to navigate toward random goal positions.

Usage:
    # Train for 500 episodes (no rendering)
    python train_waypoint_navigation.py train --episodes 500

    # Train with visual rendering
    python train_waypoint_navigation.py train --episodes 100 --render

    # Continue training from checkpoint
    python train_waypoint_navigation.py train --episodes 500 --load checkpoint.pkl

    # Resume training from latest checkpoint (auto-detects)
    python train_waypoint_navigation.py resume --episodes 500

    # Evaluate trained Q-tables
    python train_waypoint_navigation.py eval waypoint_qtables.pkl

    # Merge Q-tables from multiple runs
    python train_waypoint_navigation.py merge run1.pkl run2.pkl -o merged.pkl
"""

import argparse
import glob
import os
import re
import time
from typing import Optional

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
    UnitWrapper,
    OperationalWrapper,
    RenderWrapper,
)
from combatenv.wrappers.waypoint_task_wrapper import WaypointTaskWrapper
from combatenv.wrappers.qlearning_wrapper import QLearningWrapper
from combatenv.config import AGENTS_PER_UNIT, NUM_UNITS_PER_TEAM


def create_training_env(
    render_mode: Optional[str] = None,
    training: bool = True,
    max_steps: int = 500,
    autonomous_combat: bool = False,
):
    """
    Create wrapped environment stack for waypoint training.

    Wrapper stack (bottom to top):
        GridWorld (base)
        -> AgentWrapper (128 agents: 8 units × 8 agents × 2 teams)
        -> BaseTerrainWrapper (terrain generation)
        -> TeamWrapper (blue/red teams)
        -> ProjectileWrapper (projectile management)
        -> BaseFOVWrapper (field of view)
        -> BaseCombatWrapper (combat system, disabled for waypoint training)
        -> MovementWrapper (agent movement, wandering disabled)
        -> BaseObservationWrapper (observation generation)
        -> TerminationWrapper (episode termination)
        -> UnitWrapper (creates units from agents)
        -> OperationalWrapper (boids steering for units)
        -> WaypointTaskWrapper (spawn positions, waypoint assignment, rewards)
        -> QLearningWrapper (per-team Q-tables, action selection, learning)
        -> RenderWrapper (optional, for visualization)

    Args:
        render_mode: "human" for visual rendering, None for headless
        training: Whether in training mode (UCB exploration) or evaluation (greedy)
        max_steps: Maximum steps per episode
        autonomous_combat: Whether agents shoot autonomously (default: False for waypoint training)

    Returns:
        Wrapped environment with Q-learning
    """
    # Calculate total agents: 8 units × 8 agents × 2 teams = 128
    num_agents = NUM_UNITS_PER_TEAM * AGENTS_PER_UNIT * 2

    # Create base environment with wrapper stack (same as main.py)
    env = GridWorld(render_mode=render_mode)
    env = AgentWrapper(env, num_agents=num_agents)
    env = BaseTerrainWrapper(env)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = ProjectileWrapper(env)
    env = BaseFOVWrapper(env)
    env = BaseCombatWrapper(env, autonomous_combat=autonomous_combat)
    env = MovementWrapper(env, enable_wandering=False)  # Disable wandering, use boids
    env = BaseObservationWrapper(env)
    env = TerminationWrapper(
        env,
        max_steps=max_steps,
        terminate_on_controlled_death=False,
        terminate_on_team_elimination=False,
    )

    # Add unit wrapper (creates units from agents)
    env = UnitWrapper(
        env,
        num_units_per_team=NUM_UNITS_PER_TEAM,
        agents_per_unit=AGENTS_PER_UNIT,
        enable_boids=True,   # Enable boids for smooth waypoint navigation
        extend_obs=False,    # Don't extend observations
    )

    # Add operational wrapper (units, boids steering)
    env = OperationalWrapper(env)

    # Add waypoint task wrapper (spawning, individual goals, rewards)
    env = WaypointTaskWrapper(env)

    # Add Q-learning wrapper (handles discretization, action selection, learning)
    env = QLearningWrapper(
        env,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_bonus=1.0,
        training=training,
    )

    # Add rendering wrapper if visual mode
    if render_mode == "human":
        env = RenderWrapper(env)

    return env


def train(
    episodes: int = 500,
    render: bool = False,
    save_path: str = "waypoint_qtables.pkl",
    load_path: Optional[str] = None,
    checkpoint_interval: int = 100,
    max_steps: int = 500,
) -> None:
    """
    Train units to navigate toward waypoints.

    Args:
        episodes: Number of training episodes
        render: Whether to render visually
        save_path: Path to save final Q-tables
        load_path: Path to load existing Q-tables (continue training)
        checkpoint_interval: Save checkpoint every N episodes
        max_steps: Maximum steps per episode
    """
    print(f"Creating training environment...")
    env = create_training_env(
        render_mode="human" if render else None,
        training=True,
        max_steps=max_steps,
    )

    # Get the Q-learning wrapper (may be wrapped by RenderWrapper)
    qlearning_env = env
    while not isinstance(qlearning_env, QLearningWrapper):
        qlearning_env = qlearning_env.env

    # Load existing Q-tables if specified
    if load_path and os.path.exists(load_path):
        print(f"Loading Q-tables from {load_path}...")
        qlearning_env.load(load_path)
        print(f"  Total steps: {qlearning_env.total_steps}")

    print(f"Starting training for {episodes} episodes...")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    print()

    total_blue_reward = 0.0
    total_red_reward = 0.0
    start_time = time.time()

    for episode in range(episodes):
        _, info = env.reset()
        done = False
        step_count = 0

        while not done:
            # QLearningWrapper handles action selection internally
            _, _, terminated, truncated, info = env.step()
            done = terminated or truncated
            step_count += 1

            if render:
                env.render()
                # Process events for RenderWrapper
                if hasattr(env, 'process_events'):
                    if not env.process_events():
                        print("User quit requested")
                        env.close()
                        return

        # Track rewards
        blue_reward = info.get("blue_episode_reward", 0.0)
        red_reward = info.get("red_episode_reward", 0.0)
        total_blue_reward += blue_reward
        total_red_reward += red_reward

        # Log progress every 3 episodes
        if (episode + 1) % 3 == 0:
            stats = qlearning_env.get_statistics()
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed

            # Count Q-table entries
            blue_q_entries = len(qlearning_env.blue_q_table)
            red_q_entries = len(qlearning_env.red_q_table)

            print(f"Episode {episode + 1:4d}/{episodes}")
            print(f"  Blue reward: {blue_reward:8.2f} (avg: {total_blue_reward / (episode + 1):8.2f})")
            print(f"  Red reward:  {red_reward:8.2f} (avg: {total_red_reward / (episode + 1):8.2f})")
            print(f"  Steps: {step_count}, States visited: B={stats['blue_states_visited']}, R={stats['red_states_visited']}")
            print(f"  Q-table entries: Blue={blue_q_entries}, Red={red_q_entries}")
            print(f"  Speed: {eps_per_sec:.2f} ep/s, Elapsed: {elapsed:.1f}s")
            print()

        # Save checkpoint
        if (episode + 1) % checkpoint_interval == 0:
            checkpoint_path = f"waypoint_checkpoint_ep{episode + 1}.pkl"
            qlearning_env.save(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

    # Save final Q-tables
    qlearning_env.save(save_path)
    print(f"\nTraining complete!")
    print(f"Saved Q-tables to {save_path}")

    # Print final statistics
    stats = qlearning_env.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Blue states visited: {stats['blue_states_visited']}")
    print(f"  Red states visited: {stats['red_states_visited']}")
    print(f"  Avg blue reward: {total_blue_reward / episodes:.2f}")
    print(f"  Avg red reward: {total_red_reward / episodes:.2f}")

    env.close()


def evaluate(
    qtables_path: str,
    episodes: int = 10,
    render: bool = True,
    max_steps: int = 500,
) -> None:
    """
    Evaluate trained Q-tables.

    Args:
        qtables_path: Path to Q-tables file
        episodes: Number of evaluation episodes
        render: Whether to render visually
        max_steps: Maximum steps per episode
    """
    print(f"Creating evaluation environment...")
    env = create_training_env(
        render_mode="human" if render else None,
        training=False,  # Greedy action selection
        max_steps=max_steps,
    )

    # Get the Q-learning wrapper
    qlearning_env = env
    while not isinstance(qlearning_env, QLearningWrapper):
        qlearning_env = qlearning_env.env

    # Load Q-tables
    print(f"Loading Q-tables from {qtables_path}...")
    qlearning_env.load(qtables_path)

    print(f"Running {episodes} evaluation episodes...")
    total_blue_reward = 0.0
    total_red_reward = 0.0

    for episode in range(episodes):
        _, info = env.reset()
        done = False

        while not done:
            _, _, terminated, truncated, info = env.step()
            done = terminated or truncated

            if render:
                env.render()
                if hasattr(env, 'process_events'):
                    if not env.process_events():
                        print("User quit requested")
                        env.close()
                        return

        blue_reward = info.get("blue_episode_reward", 0.0)
        red_reward = info.get("red_episode_reward", 0.0)
        total_blue_reward += blue_reward
        total_red_reward += red_reward

        print(f"Episode {episode + 1}: Blue={blue_reward:.2f}, Red={red_reward:.2f}")

    print(f"\nEvaluation complete!")
    print(f"Avg blue reward: {total_blue_reward / episodes:.2f}")
    print(f"Avg red reward: {total_red_reward / episodes:.2f}")

    env.close()


def find_latest_checkpoint(pattern: str = "waypoint_checkpoint_ep*.pkl") -> Optional[str]:
    """
    Find the latest checkpoint file by episode number.

    Args:
        pattern: Glob pattern for checkpoint files

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    # Extract episode numbers and find max
    def get_episode_num(path):
        match = re.search(r'ep(\d+)', path)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=get_episode_num)
    return checkpoints[-1]


def resume_training(
    episodes: int = 500,
    render: bool = False,
    save_path: str = "waypoint_qtables.pkl",
    checkpoint_interval: int = 100,
    max_steps: int = 500,
) -> None:
    """
    Resume training from the latest checkpoint.

    Args:
        episodes: Number of ADDITIONAL episodes to train
        render: Whether to render visually
        save_path: Path to save final Q-tables
        checkpoint_interval: Save checkpoint every N episodes
        max_steps: Maximum steps per episode
    """
    # Find latest checkpoint
    latest = find_latest_checkpoint()

    if latest is None:
        # Also check for the main save file
        if os.path.exists(save_path):
            latest = save_path
        else:
            print("No checkpoint found. Starting fresh training...")
            train(
                episodes=episodes,
                render=render,
                save_path=save_path,
                load_path=None,
                checkpoint_interval=checkpoint_interval,
                max_steps=max_steps,
            )
            return

    # Extract episode number from checkpoint name
    match = re.search(r'ep(\d+)', latest)
    start_episode = int(match.group(1)) if match else 0

    print(f"Resuming from checkpoint: {latest}")
    print(f"  Starting at episode: {start_episode}")
    print(f"  Training for {episodes} more episodes...")

    train(
        episodes=episodes,
        render=render,
        save_path=save_path,
        load_path=latest,
        checkpoint_interval=checkpoint_interval,
        max_steps=max_steps,
    )


def merge_qtables(
    input_files: list,
    output_file: str,
) -> None:
    """
    Merge Q-tables from multiple training runs.

    Args:
        input_files: List of Q-table file paths
        output_file: Output file path
    """
    print(f"Merging {len(input_files)} Q-table files...")

    tables_list = []
    for filepath in input_files:
        print(f"  Loading {filepath}...")
        tables = QLearningWrapper.load_tables(filepath)
        tables_list.append(tables)

    print("Merging...")
    merged = QLearningWrapper.merge_tables(tables_list)

    print(f"Saving merged Q-tables to {output_file}...")
    QLearningWrapper.save_tables(merged, output_file)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Train units to navigate toward waypoints using Q-learning."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train Q-tables")
    train_parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of training episodes (default: 500)"
    )
    train_parser.add_argument(
        "--render", action="store_true",
        help="Enable visual rendering during training"
    )
    train_parser.add_argument(
        "--save", type=str, default="waypoint_qtables.pkl",
        help="Path to save Q-tables (default: waypoint_qtables.pkl)"
    )
    train_parser.add_argument(
        "--load", type=str, default=None,
        help="Path to load existing Q-tables (continue training)"
    )
    train_parser.add_argument(
        "--checkpoint", type=int, default=100,
        help="Save checkpoint every N episodes (default: 100)"
    )
    train_parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum steps per episode (default: 500)"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained Q-tables")
    eval_parser.add_argument(
        "qtables", type=str,
        help="Path to Q-tables file"
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    eval_parser.add_argument(
        "--no-render", action="store_true",
        help="Disable visual rendering"
    )
    eval_parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum steps per episode (default: 500)"
    )

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume training from latest checkpoint")
    resume_parser.add_argument(
        "--episodes", type=int, default=500,
        help="Number of additional episodes to train (default: 500)"
    )
    resume_parser.add_argument(
        "--render", action="store_true",
        help="Enable visual rendering during training"
    )
    resume_parser.add_argument(
        "--save", type=str, default="waypoint_qtables.pkl",
        help="Path to save Q-tables (default: waypoint_qtables.pkl)"
    )
    resume_parser.add_argument(
        "--checkpoint", type=int, default=100,
        help="Save checkpoint every N episodes (default: 100)"
    )
    resume_parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum steps per episode (default: 500)"
    )

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge Q-tables from multiple runs")
    merge_parser.add_argument(
        "files", type=str, nargs="+",
        help="Q-table files to merge"
    )
    merge_parser.add_argument(
        "-o", "--output", type=str, default="merged_qtables.pkl",
        help="Output file path (default: merged_qtables.pkl)"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            episodes=args.episodes,
            render=args.render,
            save_path=args.save,
            load_path=args.load,
            checkpoint_interval=args.checkpoint,
            max_steps=args.max_steps,
        )
    elif args.command == "eval":
        evaluate(
            qtables_path=args.qtables,
            episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps,
        )
    elif args.command == "resume":
        resume_training(
            episodes=args.episodes,
            render=args.render,
            save_path=args.save,
            checkpoint_interval=args.checkpoint,
            max_steps=args.max_steps,
        )
    elif args.command == "merge":
        merge_qtables(
            input_files=args.files,
            output_file=args.output,
        )
    else:
        # Default to train if no subcommand
        parser.print_help()


if __name__ == "__main__":
    main()
