import gymnasium as gym
import gymnasium_robotics
import numpy as np
import imageio

gym.register_envs(gymnasium_robotics)

example_map = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
       [1, 'c', 0, 'c', 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 1, 1, 1, 1]]

env = gym.make('PointMaze_UMaze-v3', maze_map=example_map, render_mode="rgb_array", max_episode_steps=100)


# Define a random policy
def random_policy(env):
    return env.action_space.sample()

# Define the video writer
# video_writer = imageio.get_writer("trajectory.mp4", fps=30)


# Run the environment for two episodes
frames = []
for _ in range(5):
    state, _ = env.reset()
    truncated, done = False, False

    while not (truncated or done) :
        # action = random_policy(env)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        # Capture the frame
        frame = env.render()
        frames.append(frame)
    
    # Append frames to the video
    # for frame in frames:
    #     video_writer.append_data(frame)


# frames = []
# state, _ = env.reset()
# for _ in range(10):
#     #action = random_policy(env)
#     action = env.action_space.sample()
#     state, reward, done, truncated, info = env.step(action)
    
#     frame = env.render()
#     frames.append(frame)

#     print(_)
#     # Break if the episode ends before 100 timesteps
#     if done or truncated:
#         break

# Close video writer and environment
imageio.mimsave('/home/yuxuanli/failed_IRL_new/Deep_Model/Maze/simulation.mp4', frames, fps=30)
env.close()