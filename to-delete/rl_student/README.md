# Tabular Q-Learning for combatenv

This module provides a complete implementation of tabular Q-learning for training agents in the `combatenv` tactical combat simulation.

## Learning Objectives

After completing this module, you will understand:

1. **State Space Discretization**: How to convert continuous observations into discrete states for Q-table lookup
2. **Multi-Agent Learning**: Two approaches - shared Q-table vs. individual Q-tables
3. **Q-Learning Algorithm**: The update rule and exploration-exploitation tradeoff
4. **Gymnasium Wrappers**: How to extend environment functionality without modifying source code

## Quick Start

```bash
# Activate virtual environment
source ~/.venvs/pg/bin/activate

# Train with shared Q-table (recommended to start)
python -m rl_student.train --mode shared --episodes 1000

# Train with visualization (slower but informative)
python -m rl_student.train --mode shared --episodes 100 --render

# Evaluate a trained model
python -m rl_student.train --evaluate q_table_shared.pkl --render
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Wrapper Stack                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TacticalCombatEnv        (Base: 1 controlled agent)        │
│         ↓                                                   │
│  MultiAgentWrapper        (Enables 200-agent control)       │
│         ↓                                                   │
│  DiscreteObservationWrapper  (88 floats → 2,048 states)     │
│         ↓                                                   │
│  DiscreteActionWrapper    (3 floats → 8 discrete actions)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## State Space Design

The environment provides an 88-dimensional continuous observation. We discretize this into **2,048 discrete states** using feature engineering:

### Discretized Features

| Feature | Source | Levels | Values |
|---------|--------|--------|--------|
| Health | obs[3] | 2 | high (≥0.5), low (<0.5) |
| Ammo | obs[7] | 2 | high (≥0.5), low (<0.5) |
| Armor | obs[5] | 2 | high (≥0.5), low (<0.5) |
| Enemy Distance | obs[13] | 2 | near (<0.15), far (≥0.15) |
| Ally Distance | obs[33] | 2 | near (<0.15), far (≥0.15) |
| Enemy Direction | obs[10:12] | 4 | North, South, East, West |
| Terrain in FOV | obs[50:87] | 16 | 4 binary flags |

### State Calculation

```
Total States = 2 × 2 × 2 × 2 × 2 × 4 × 16 = 2,048 states
```

### Terrain Flags

The terrain observation encodes which terrain types are visible in the agent's field of view:

| Bit | Terrain Type | Effect |
|-----|--------------|--------|
| 0 | Building | Blocks movement and line of sight |
| 1 | Fire | Deals damage (bypasses armor) |
| 2 | Swamp | Traps agents temporarily |
| 3 | Water | Impassable |

Example: If an agent sees both fire and swamp, the terrain flag = 0b0110 = 6

## Action Space Design

We define **8 discrete actions** (4 directions × 2 shoot options):

| Action | Direction | Shoot | Continuous Values |
|--------|-----------|-------|-------------------|
| 0 | North | No | [0.0, -1.0, 0.0] |
| 1 | South | No | [0.0, 1.0, 0.0] |
| 2 | East | No | [1.0, 0.0, 0.0] |
| 3 | West | No | [-1.0, 0.0, 0.0] |
| 4 | North | Yes | [0.0, -1.0, 1.0] |
| 5 | South | Yes | [0.0, 1.0, 1.0] |
| 6 | East | Yes | [1.0, 0.0, 1.0] |
| 7 | West | Yes | [-1.0, 0.0, 1.0] |

## Training Modes

### Shared Q-Table (Recommended to Start)

All 200 agents share and update the same Q-table in real-time.

**Pros:**
- Faster learning (200× more experience per episode)
- Lower memory usage (~64 KB)
- Learns general tactics applicable to all agents

**Cons:**
- All agents behave identically
- Cannot specialize for different situations

```python
q_manager = QTableManager(n_states=2048, n_actions=8, mode="shared")
```

### Individual Q-Tables

Each agent has its own Q-table.

**Pros:**
- Agents can specialize
- More diverse behaviors

**Cons:**
- Slower learning (each agent only learns from own experience)
- Higher memory usage (~13 MB)

```python
q_manager = QTableManager(n_states=2048, n_actions=8, mode="individual")
```

## Q-Learning Algorithm

### Update Rule

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') - Q(s, a)]
```

Where:
- `s`: current state
- `a`: action taken
- `r`: reward received
- `s'`: next state
- `α`: learning rate (default: 0.1)
- `γ`: discount factor (default: 0.99)

### Exploration Strategy

We use **ε-greedy** exploration:

- With probability ε: take random action (explore)
- With probability 1-ε: take best action (exploit)

Epsilon decays over training:
- Start: ε = 1.0 (100% exploration)
- Decay: ε = ε × 0.995 per episode
- Minimum: ε = 0.05 (5% exploration)

## Reward Structure

| Event | Reward |
|-------|--------|
| Kill enemy | +10.0 |
| Survive step | +0.01 |
| Agent dies | -1.0 |

## Code Structure

```
rl_student/
├── __init__.py              # Package exports
├── wrappers/
│   ├── __init__.py
│   ├── multi_agent.py       # MultiAgentWrapper
│   ├── discrete_obs.py      # DiscreteObservationWrapper
│   └── discrete_action.py   # DiscreteActionWrapper
├── q_table.py               # QTableManager
├── train.py                 # Training/evaluation script
└── README.md                # This file
```

## Usage Examples

### Training

```python
from rl_student import create_wrapped_env, QTableManager

# Create wrapped environment
env = create_wrapped_env(render_mode=None, max_steps=500)

# Create Q-table manager
q_manager = QTableManager(
    n_states=env.n_states,      # 2048
    n_actions=env.n_actions,    # 8
    mode="shared",
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0
)

# Training loop
for episode in range(1000):
    states, info = env.reset()
    done = False

    while not done:
        # Select actions (epsilon-greedy)
        actions = q_manager.get_actions(states, training=True)

        # Step environment
        next_states, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Update Q-values
        q_manager.batch_update(states, actions, rewards, next_states, done)

        states = next_states

    # Decay exploration rate
    q_manager.decay_epsilon()

# Save trained model
q_manager.save("my_qtable.pkl")
```

### Evaluation

```python
# Load trained model
q_manager = QTableManager(n_states=2048, n_actions=8, mode="shared")
q_manager.load("my_qtable.pkl")

# Evaluate with rendering
env = create_wrapped_env(render_mode="human", max_steps=1000)
states, _ = env.reset()
done = False

while not done:
    # Greedy action selection (no exploration)
    actions = q_manager.get_actions(states, training=False)
    states, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    env.render()
```

### Debugging State Encoding

```python
from rl_student.wrappers import DiscreteObservationWrapper

# Decode a state index to see its components
wrapper = DiscreteObservationWrapper(env)
state_info = wrapper.decode_state(1234)
print(state_info)
# Output: {'health': 'high', 'ammo': 'low', 'enemy_direction': 'East', ...}
```

## Exercises

### Exercise 1: Hyperparameter Tuning

Try different hyperparameters and compare results:

```python
# Try different learning rates
for lr in [0.01, 0.1, 0.5]:
    q_manager = QTableManager(n_states=2048, n_actions=8, learning_rate=lr)
    # Train and compare...
```

Questions:
- How does learning rate affect convergence speed?
- What happens with very high learning rate?

### Exercise 2: Compare Training Modes

Train both shared and individual modes for the same number of episodes:

```bash
python -m rl_student.train --mode shared --episodes 1000
python -m rl_student.train --mode individual --episodes 1000
```

Questions:
- Which mode achieves better win rate?
- How does state coverage differ between modes?

### Exercise 3: Analyze Learned Policy

After training, analyze what the agent learned:

```python
# Find states where agent prefers shooting
for state in range(2048):
    best_action = q_manager.get_best_action(0, state)
    if best_action >= 4:  # Shooting action
        state_info = wrapper.decode_state(state)
        print(f"Shoots when: {state_info}")
```

### Exercise 4: Modify State Representation

The current state representation uses 2,048 states. Try modifying `discrete_obs.py` to:

1. Add more enemy direction bins (8 directions instead of 4)
2. Add stamina as a feature
3. Remove terrain flags

How do these changes affect learning?

## Troubleshooting

### "Q-values not updating"

Check that:
1. Epsilon is not too high (agents only exploring)
2. Rewards are being received
3. State representation is working correctly

### "All agents doing the same thing"

This is expected in shared mode. In individual mode, agents should eventually develop different behaviors.

### "Training too slow"

- Use headless mode (no `--render`)
- Reduce `max_steps_per_episode`
- Use shared mode instead of individual

## Memory Usage

| Mode | Q-Table Size | Memory |
|------|--------------|--------|
| Shared | 2,048 × 8 | ~64 KB |
| Individual | 200 × 2,048 × 8 | ~13 MB |

## References

1. Watkins, C.J.C.H. (1989). Learning from Delayed Rewards. PhD thesis.
2. Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction.
