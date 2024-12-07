import gymnasium as gym
import panda_gym
import imageio

env = gym.make("PandaReach-v3", render_mode="rgb_array")
observation, info = env.reset()
frames = []

for _ in range(10):
    current_position = observation["observation"][0:3]
    desired_position = observation["desired_goal"][0:3]
    action = 5.0 * (desired_position - current_position)
    observation, reward, terminated, truncated, info = env.step(action)

    # Capture frame
    frame = env.render()
    frames.append(frame)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# Save frames to a video file
imageio.mimsave('simulation.mp4', frames, fps=30)
