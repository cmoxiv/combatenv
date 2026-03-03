"""
Tactical Combat Training Script (Phase 2).

Train tactical agents for combat, ammo economy, and terrain leverage.
Shooting is enabled. Starts from Phase 1 checkpoint for movement skills.

Uses pre-trained operational Q-tables for unit control while tactical agents learn.

Usage:
    # Train for 1000 episodes (headless)
    python train_tactical_combat.py --episodes 1000 \\
        --ops-qtable operational_agent.pkl \\
        --resume tactical_movement_final.pkl

    # Train with visualization
    python train_tactical_combat.py --episodes 100 --render \\
        --ops-qtable operational_agent.pkl \\
        --resume tactical_movement_final.pkl

    # Evaluate trained Q-tables
    python train_tactical_combat.py --eval tactical_combat_ep500.pkl --render
"""

import argparse
import os
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
    autonomous_combat: bool = True,
):
    """
    Create wrapped environment stack for Phase 2 tactical combat training.

    Uses the same pattern as train_waypoint_navigation.py with
    autonomous combat enabled for combat learning.

    Wrapper stack (bottom to top):
        GridWorld (base)
        -> AgentWrapper (agents)
        -> BaseTerrainWrapper (terrain generation)
        -> TeamWrapper (blue/red teams)
        -> ProjectileWrapper (projectile management)
        -> BaseFOVWrapper (field of view)
        -> BaseCombatWrapper (combat system enabled)
        -> MovementWrapper (agent movement)
        -> BaseObservationWrapper (observation generation)
        -> TerminationWrapper (episode termination)
        -> UnitWrapper (creates units from agents)
        -> OperationalWrapper (boids steering for units)
        -> WaypointTaskWrapper (spawning, waypoint assignment, rewards)
        -> QLearningWrapper (handles discretization, action selection, learning)
        -> RenderWrapper (optional, for visualization)

    Args:
        render_mode: "human" for visual rendering, None for headless
        training: Whether in training mode (UCB exploration) or evaluation (greedy)
        max_steps: Maximum steps per episode
        autonomous_combat: Whether agents shoot autonomously (True for Phase 2)

    Returns:
        Wrapped environment with Q-learning
    """
    # Calculate total agents: 8 units x 8 agents x 2 teams = 128
    num_agents = NUM_UNITS_PER_TEAM * AGENTS_PER_UNIT * 2

    # Create base environment with wrapper stack
    env = GridWorld(render_mode=render_mode)
    env = AgentWrapper(env, num_agents=num_agents)
    env = BaseTerrainWrapper(env)
    env = TeamWrapper(env, teams=["blue", "red"])
    env = ProjectileWrapper(env)
    env = BaseFOVWrapper(env)
    env = BaseCombatWrapper(env, autonomous_combat=autonomous_combat)  # Combat enabled in Phase 2
    env = MovementWrapper(env, enable_wandering=False)  # Use boids for movement
    env = BaseObservationWrapper(env)
    env = TerminationWrapper(
        env,
        max_steps=max_steps,
        terminate_on_controlled_death=False,
        terminate_on_team_elimination=True,  # End when a team is eliminated
    )

    # Add unit wrapper (creates units from agents)
    env = UnitWrapper(
        env,
        num_units_per_team=NUM_UNITS_PER_TEAM,
        agents_per_unit=AGENTS_PER_UNIT,
        enable_boids=True,
        extend_obs=False,
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
    episodes: int = 1000,
    max_steps: int = 500,
    render: bool = False,
    save_interval: int = 100,
    log_interval: int = 10,
    resume_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Train tactical agents for combat (Phase 2).

    Uses the same approach as train_waypoint_navigation.py with
    autonomous combat enabled.

    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        render: Whether to render visually
        save_interval: Save checkpoint every N episodes
        log_interval: Log stats every N episodes
        resume_path: Path to Phase 1 checkpoint to load Q-tables from
        seed: Random seed
    """
    print("=" * 60)
    print("Phase 2: Tactical Combat Training")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Resume from: {resume_path or 'None'}")
    print()

    if not resume_path:
        print("INFO: Starting fresh (no Phase 1 checkpoint)")
        print()

    # Create training environment
    env = create_training_env(
        render_mode="human" if render else None,
        training=True,
        max_steps=max_steps,
        autonomous_combat=True,  # Combat enabled in Phase 2
    )

    # Get the Q-learning wrapper (may be wrapped by RenderWrapper)
    qlearning_env = env
    while not isinstance(qlearning_env, QLearningWrapper):
        qlearning_env = qlearning_env.env

    # Load Phase 1 checkpoint (movement skills)
    if resume_path and os.path.exists(resume_path):
        print(f"Loading Phase 1 checkpoint: {resume_path}")
        qlearning_env.load(resume_path)

    print(f"\nStarting training for {episodes} episodes...")
    print()

    # Training statistics
    total_blue_reward = 0.0
    total_red_reward = 0.0
    total_blue_kills = 0
    total_red_kills = 0
    start_time = time.time()

    for episode in range(episodes):
        _, info = env.reset(seed=seed + episode if seed else None)
        done = False
        step_count = 0
        episode_blue_kills = 0
        episode_red_kills = 0

        while not done:
            # Q-learning wrapper handles action selection internally
            _, rewards, terminated, truncated, info = env.step(None)
            done = terminated or truncated
            step_count += 1

            # Track kills from info
            blue_kills = info.get("blue_kills", 0)
            red_kills = info.get("red_kills", 0)
            episode_blue_kills = blue_kills
            episode_red_kills = red_kills

            if render:
                env.render()
                # Process events for RenderWrapper
                if hasattr(env, 'process_events'):
                    if not env.process_events():
                        print("\nUser quit requested")
                        env.close()
                        return

        # Track rewards
        blue_reward = info.get("blue_episode_reward", 0.0)
        red_reward = info.get("red_episode_reward", 0.0)
        total_blue_reward += blue_reward
        total_red_reward += red_reward
        total_blue_kills += episode_blue_kills
        total_red_kills += episode_red_kills

        # Log progress
        if (episode + 1) % log_interval == 0:
            stats = qlearning_env.get_statistics()
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0

            # Calculate K/D ratio
            blue_kd = episode_blue_kills / max(1, episode_red_kills) if episode_red_kills else episode_blue_kills
            red_kd = episode_red_kills / max(1, episode_blue_kills) if episode_blue_kills else episode_red_kills

            print(f"Episode {episode + 1:5d}/{episodes}")
            print(f"  Blue: reward={blue_reward:10.2f} (avg: {total_blue_reward / (episode + 1):10.2f}), kills={episode_blue_kills}, K/D={blue_kd:.2f}")
            print(f"  Red:  reward={red_reward:10.2f} (avg: {total_red_reward / (episode + 1):10.2f}), kills={episode_red_kills}, K/D={red_kd:.2f}")
            print(f"  Steps: {step_count:4d}")
            print(f"  States visited: Blue={stats['blue_states_visited']:4d}, Red={stats['red_states_visited']:4d}")
            print(f"  Speed: {eps_per_sec:.2f} ep/s, Elapsed: {elapsed:.1f}s")
            print()

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = f"tactical_combat_ep{episode + 1}.pkl"
            qlearning_env.save(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            print()

    # Save final Q-tables
    final_path = "tactical_combat_final.pkl"
    qlearning_env.save(final_path)

    # Print final statistics
    stats = qlearning_env.get_statistics()
    elapsed = time.time() - start_time

    # Calculate overall K/D
    overall_blue_kd = total_blue_kills / max(1, total_red_kills)
    overall_red_kd = total_red_kills / max(1, total_blue_kills)

    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total episodes: {episodes}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")
    print(f"Average speed: {episodes / elapsed:.2f} ep/s")
    print(f"Total steps: {stats['total_steps']}")
    print()
    print("Combat Statistics:")
    print(f"  Total blue kills: {total_blue_kills} (K/D: {overall_blue_kd:.2f})")
    print(f"  Total red kills: {total_red_kills} (K/D: {overall_red_kd:.2f})")
    print()
    print("Learning Statistics:")
    print(f"  Blue states visited: {stats['blue_states_visited']}")
    print(f"  Red states visited: {stats['red_states_visited']}")
    print(f"  Avg blue reward: {total_blue_reward / episodes:.2f}")
    print(f"  Avg red reward: {total_red_reward / episodes:.2f}")
    print()
    print(f"Final Q-tables saved to: {final_path}")

    env.close()


def evaluate(
    qtables_path: str,
    episodes: int = 10,
    max_steps: int = 500,
    render: bool = True,
) -> None:
    """
    Evaluate trained tactical combat Q-tables.

    Args:
        qtables_path: Path to Q-tables file
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render visually
    """
    print("=" * 60)
    print("Phase 2: Tactical Combat Evaluation")
    print("=" * 60)
    print(f"Q-tables: {qtables_path}")
    print(f"Episodes: {episodes}")
    print()

    # Create evaluation environment
    env = create_training_env(
        render_mode="human" if render else None,
        training=False,  # Greedy action selection
        max_steps=max_steps,
        autonomous_combat=True,
    )

    # Get the Q-learning wrapper
    qlearning_env = env
    while not isinstance(qlearning_env, QLearningWrapper):
        qlearning_env = qlearning_env.env

    # Load Q-tables
    qlearning_env.load(qtables_path)

    print(f"\nRunning {episodes} evaluation episodes...")
    print()

    total_blue_reward = 0.0
    total_red_reward = 0.0
    total_blue_kills = 0
    total_red_kills = 0

    for episode in range(episodes):
        _, info = env.reset()
        done = False

        while not done:
            _, _, terminated, truncated, info = env.step(None)
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
        blue_kills = info.get("blue_kills", 0)
        red_kills = info.get("red_kills", 0)

        total_blue_reward += blue_reward
        total_red_reward += red_reward
        total_blue_kills += blue_kills
        total_red_kills += red_kills

        print(f"Episode {episode + 1}: Blue reward={blue_reward:.2f}, kills={blue_kills} | Red reward={red_reward:.2f}, kills={red_kills}")

    # Calculate K/D ratios
    blue_kd = total_blue_kills / max(1, total_red_kills)
    red_kd = total_red_kills / max(1, total_blue_kills)

    print()
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Avg blue reward: {total_blue_reward / episodes:.2f}")
    print(f"Avg red reward: {total_red_reward / episodes:.2f}")
    print(f"Blue K/D ratio: {blue_kd:.2f}")
    print(f"Red K/D ratio: {red_kd:.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Tactical Combat Training (shooting enabled)"
    )

    # Training parameters
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Number of training episodes (default: 1000)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps per episode (default: 500)"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable visual rendering"
    )
    parser.add_argument(
        "--save-interval", type=int, default=100,
        help="Save checkpoint every N episodes (default: 100)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="Log stats every N episodes (default: 10)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint (can use Phase 1 checkpoint)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )

    # Evaluation mode
    parser.add_argument(
        "--eval", type=str, default=None,
        help="Evaluate Q-tables from file (disables training)"
    )

    args = parser.parse_args()

    if args.eval:
        # Evaluation mode
        evaluate(
            qtables_path=args.eval,
            episodes=args.episodes if args.episodes != 1000 else 10,
            max_steps=args.max_steps,
            render=args.render or True,  # Default to render in eval mode
        )
    else:
        # Training mode
        train(
            episodes=args.episodes,
            max_steps=args.max_steps,
            render=args.render,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            resume_path=args.resume,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
