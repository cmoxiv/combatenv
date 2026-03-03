"""Run environment with rendering to visualize edge spawns and center waypoints."""
from movement_training_env import MultiAgentDiscreteEnv

env = MultiAgentDiscreteEnv(render_mode="human")
obs, info = env.reset(seed=42)

print("Blue units spawn at top (row 0), Red at bottom (row 7)")
print("All waypoints at center (32, 32)")
print("Press Q to quit")

running = True
step = 0
while running:
    actions = env.get_actions(obs, training=True)
    obs, rewards, terminated, truncated, info = env.step(actions)
    running = env.render()
    step += 1

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()
        step = 0

env.close()
