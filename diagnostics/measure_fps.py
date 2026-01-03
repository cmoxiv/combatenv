"""
FPS measurement diagnostic for the tactical combat simulation.
"""

import time
from combatenv import TacticalCombatEnv, EnvConfig


def measure_fps(num_frames: int = 200, show_fov: bool = True):
    """
    Measure average FPS over a number of frames.

    Args:
        num_frames: Number of frames to measure
        show_fov: Whether to enable FOV visualization
    """
    config = EnvConfig(
        respawn_enabled=False,
        terminate_on_controlled_death=False,
        terminate_on_team_elimination=False,
        max_steps=num_frames + 100
    )

    env = TacticalCombatEnv(render_mode='human', config=config)
    env.reset(seed=42)
    env.show_fov = show_fov
    env.render()

    # Warm up
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
        env.render()

    # Measure FPS
    start = time.time()
    for i in range(num_frames):
        action = env.action_space.sample()
        env.step(action)
        env.render()
    elapsed = time.time() - start

    fps = num_frames / elapsed
    print(f"\n{'='*50}")
    print(f"FPS MEASUREMENT (FOV {'ON' if show_fov else 'OFF'})")
    print(f"{'='*50}")
    print(f"Frames: {num_frames}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Average FPS: {fps:.1f}")
    print(f"{'='*50}\n")

    env.close()
    return fps


if __name__ == "__main__":
    print("Testing FPS with FOV OFF...")
    fps_off = measure_fps(num_frames=200, show_fov=False)

    print("Testing FPS with FOV ON...")
    fps_on = measure_fps(num_frames=200, show_fov=True)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"FOV OFF: {fps_off:.1f} FPS")
    print(f"FOV ON:  {fps_on:.1f} FPS")
    print("="*50)
