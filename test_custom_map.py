"""Test script for custom map loading."""
import time
from combatenv import TacticalCombatEnv, load_map

print("Loading custom map...")
terrain = load_map('map.json')
print(f"Map has: 109 buildings, 22 fire, 24 swamp, 34 water")

env = TacticalCombatEnv(render_mode='human')
obs, info = env.reset(options={'terrain_grid': terrain})
env.render()

print("Running for 20 seconds - check if you see your terrain!")
start = time.time()
while env.process_events() and (time.time() - start) < 20:
    env.step(env.action_space.sample())
    env.render()

env.close()
print('Done')
