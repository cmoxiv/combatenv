from combatenv.gridworld import GridWorld
from combatenv.wrappers import (
    AgentWrapper,
    TeamWrapper,
    BaseTerrainWrapper,
    MovementWrapper,
    RenderWrapper,
    KeybindingsWrapper,
    DebugOverlayWrapper,
    TerminationWrapper,
)

# Start with empty grid
env = GridWorld(grid_size=64, render_mode="human")

# Add agents
env = AgentWrapper(env, num_agents=100)

# Assign teams
env = TeamWrapper(env, teams=["blue", "red"])

# Add terrain
env = BaseTerrainWrapper(env)

# Enable movement
env = MovementWrapper(env)

# Add episode termination
env = TerminationWrapper(env, max_steps=500)


env = KeybindingsWrapper(env)

# Add debug overlay (press ` to toggle)
env = DebugOverlayWrapper(env)

env = RenderWrapper(env)


# Now use it
obs, info = env.reset()

# Standard Gymnasium interface
obs, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()
        pass
    pass

env.close()
