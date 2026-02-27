# RL Training Requirements Definition

## Project Overview

Enhance the existing tabular Q-learning training system for tactical combat agents in `combatenv`. The focus is on improving combat effectiveness training with detailed stat tracking while building on the existing infrastructure.

## Existing Infrastructure

The following components already exist:

### Training Script (`rl_student/train.py`)
- Multi-agent Q-learning supporting 200 agents
- Shared or individual Q-table modes
- Episode-based training with checkpoints
- Basic win/loss tracking

### State Space (1,920 discrete states)
- Health: 2 levels (high/low)
- Ammo: 2 levels (high/low)
- Armor: 2 levels (high/low)
- Enemy distance: 2 levels (near/far)
- Ally distance: 2 levels (near/far)
- Enemy direction: 4 levels (N/S/E/W)
- Terrain: 5 types (empty/building/fire/swamp/water)
- Centroid distance: 3 levels (close/optimal/far)

### Action Space (8 discrete actions)
- Hold: Maintain position relative to unit
- Contract: Move toward unit centroid
- Expand: Move away from unit centroid
- FlankLeft: Orbit counter-clockwise
- FlankRight: Orbit clockwise
- Shoot: Stationary combat
- Think: Scan for enemies
- No-op: Do nothing

### Q-Table Manager (`rl_student/q_table.py`)
- Epsilon-greedy exploration with decay
- Visit count tracking (N-tables)
- Save/load functionality
- Training statistics

## Requirements

### R1: Detailed Combat Statistics
Track and report granular combat metrics during training:

| Metric | Description |
|--------|-------------|
| Shots Fired | Total shots fired per agent/team/episode |
| Hits | Successful hits on enemy agents |
| Hit Rate | Hits / Shots Fired (accuracy) |
| Kills | Enemies eliminated |
| Deaths | Friendly agents eliminated |
| K/D Ratio | Kills / Deaths |
| Damage Dealt | Total HP damage inflicted |
| Damage Taken | Total HP damage received |
| Ammo Efficiency | Kills / Rounds Fired |
| Survival Time | Steps agent remained alive |

### R2: Enhanced Training Output
Improve logging to show combat effectiveness:
- Per-episode combat summary
- Rolling averages (last N episodes)
- Team comparison (Blue vs Red performance)
- Progress visualization in terminal

### R3: Checkpoint Enhancement
Extend periodic checkpoints to include:
- Combat stats history
- Learning curves data
- Best model tracking (by win rate or kill ratio)

### R4: Training Configuration
Configurable training parameters:
- Episode count
- Max steps per episode
- Learning rate, discount factor
- Epsilon decay schedule
- Checkpoint frequency
- Logging verbosity

### R5: Full Environment Training
Train in the complete tactical environment:
- All terrain types (buildings, fire, swamp, water)
- Two-layer FOV system
- Resource management (stamina, armor, ammo)
- Team-based combat (100 vs 100)

### R6: Single-Machine Execution
Training runs on a single machine:
- Vectorized environment not required initially
- Pygame rendering optional
- Progress saved to disk

## Out of Scope

The following are explicitly NOT in scope:
- Deep RL algorithms (PPO, DQN, A2C)
- Multi-agent RL (MAPPO, QMIX)
- Hierarchical RL
- Distributed training
- Experiment tracking (W&B, MLflow)
- Strategic or operational level training

## Success Criteria

1. Training produces agents with positive kill/death ratios
2. Combat stats are tracked and reported accurately
3. Checkpoints include full training state and stats
4. Training completes within reasonable time on single machine
5. Trained agents demonstrate combat effectiveness vs random baseline

## Next Steps

Proceed to **Stage 2: Requirements Specification** to detail the implementation approach.
