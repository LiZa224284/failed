import gymnasium as gym
import panda_gym
import imageio
from MyPush import MyPandaPushEnv

# env = gym.make("PandaReach-v3", render_mode="rgb_array")
env_name = 'MyPandaPushEnv'
env = gym.make(env_name, render_mode="rgb_array")
observation, info = env.reset()
frames = []

# for _ in range(10):
#     current_position = observation["observation"][0:3]
#     desired_position = observation["desired_goal"][0:3]
#     action = 5.0 * (desired_position - current_position)
#     observation, reward, terminated, truncated, info = env.step(action)

#     # Capture frame
#     frame = env.render()
#     frames.append(frame)

#     if terminated or truncated:
#         observation, info = env.reset()

num_episodes = 5
for episode in range(num_episodes):
    observation, info = env.reset()
    # terminated = truncated = False
    done = False
    while not done:
        current_position = observation["observation"][0:3]
        desired_position = observation["desired_goal"][0:3]
        action = 5.0 * (desired_position - current_position)
        # action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        done = terminated or truncated
    print('record one episode')


env.close()
imageio.mimsave('/home/xlx9645/failed/PandaRobot/PandaPush/tools/simulation.mp4', frames, fps=30)
