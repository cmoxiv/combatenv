# TODO List - combatenv Package

## Completed Features

### Core Systems
- [x] **Gymnasium Environment**: TacticalCombatEnv with standard API
- [x] **Observation Space**: 50-dimension normalized observation vector
- [x] **Action Space**: Movement, rotation, and shooting actions
- [x] **Reward Function**: Kill rewards, survival rewards
- [x] **Terminal Conditions**: Death, team elimination, max steps

### Combat System
- [x] **Projectile System**: Accuracy-based shooting with deviation
- [x] **Two-Layer FOV**: Near (3 cells, 90 deg) and Far (5 cells, 120 deg)
- [x] **Damage System**: Armor absorbs before health
- [x] **Cooldowns**: Shooting cooldown, reload timer

### Resource Management
- [x] **Stamina System**: Drain while moving, regen while idle
- [x] **Ammo System**: Magazine + reserve, auto-reload
- [x] **Armor System**: Depleting damage absorption

### Terrain System
- [x] **Terrain Types**: EMPTY, BUILDING, FIRE, SWAMP, WATER
- [x] **Terrain Effects**: Fire damage, swamp stuck, building blocking
- [x] **LOS Integration**: Buildings block line of sight
- [x] **Procedural Generation**: Random terrain placement

### Infrastructure
- [x] **Package Structure**: `combatenv` package with `__init__.py`
- [x] **Comprehensive Tests**: 190 unit and integration tests
- [x] **Documentation**: README, API, Architecture, guides

---

## Future Enhancements

### High Priority

- [ ] **Multi-Agent Control**: Control multiple agents simultaneously
  - Allow actions for all agents in observation/action spaces
  - Support centralized training with decentralized execution

- [ ] **Observation Enhancements**:
  - Add enemy/ally health visibility in observation
  - Add terrain information in local view
  - Add projectile trajectory awareness

- [ ] **Reward Shaping**:
  - Add reward for dealing damage
  - Add reward for surviving damage
  - Add team-based coordination rewards

### Medium Priority

- [ ] **Action Space Variants**:
  - Discrete action space option
  - Multi-discrete action space option
  - Dictionary action space for complex actions

- [ ] **Advanced AI Behaviors**:
  - Target prioritization by health
  - Cover-seeking behavior near buildings
  - Team coordination (flanking, focus fire)

- [ ] **Environment Variants**:
  - Different map layouts (pre-defined)
  - Varying team sizes
  - Asymmetric scenarios (defend vs attack)

- [ ] **Performance Profiling**:
  - Add profiling hooks
  - Optimize hot paths
  - Memory usage analysis

### Low Priority

- [ ] **Visual Enhancements**:
  - Sprite-based rendering option
  - Particle effects (explosions, smoke)
  - Health bars above agents
  - Minimap view

- [ ] **Replay System**:
  - Record episodes to file
  - Playback recorded episodes
  - Step-by-step replay controls

- [ ] **Training Utilities**:
  - Vectorized environment wrapper
  - Curriculum learning support
  - Self-play framework

- [ ] **Configuration UI**:
  - In-game parameter adjustment
  - Real-time config changes
  - Preset loading/saving

---

## Student Assignment Ideas

### Beginner
- Modify agent colors and sizes in `config.py`
- Change FOV angles and ranges
- Adjust movement and rotation speeds

### Intermediate
- Implement a simple rule-based AI policy
- Add a new terrain type with custom effect
- Create a custom reward function

### Advanced
- Train an RL agent using Stable Baselines3
- Implement multi-agent reinforcement learning
- Add communication between team agents
- Create a tournament framework

---

## Known Issues

- [ ] Projectiles don't use spatial grid for collision (O(projectiles * agents))
- [ ] No support for different map sizes at runtime
- [ ] Gymnasium env_checker has strict requirements (test skipped)

---

## Notes

The package is designed for educational use. Students import from `combatenv` and can:

```python
from combatenv import TacticalCombatEnv, EnvConfig

# Create and run environment
env = TacticalCombatEnv(render_mode="human")
obs, info = env.reset()

# Custom configuration
config = EnvConfig(num_blue_agents=50, num_red_agents=50)
env = TacticalCombatEnv(config=config)
```
