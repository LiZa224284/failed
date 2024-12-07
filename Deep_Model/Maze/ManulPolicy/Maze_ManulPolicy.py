import numpy as np
import gymnasium as gym
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import imageio

gym.register_envs(gymnasium_robotics)
# Initialize the environment
example_map = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
       [1, 'c', 1, 'c', 1],
       [1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1],
       [1, 1, 1, 1, 1]]

env = gym.make('PointMaze_UMaze-v3', maze_map=example_map, render_mode="rgb_array", max_episode_steps=100)


def manual_policy(observation):
    agent_pos = obs['achieved_goal'] # Agent's current position (x, y)
    goal_pos = obs['desired_goal']  # Goal position (x, y)
    
    # Calculate the direction vector to the goal
    direction = goal_pos - agent_pos
    
    # Define the magnitude of the force (constant or scaled by distance)
    force_magnitude = 1.0  # Max force (tuned for environment)
    
    # Normalize direction vector and scale by the force magnitude
    if np.linalg.norm(direction) > 1e-6:  # Avoid division by zero
        action = (direction / np.linalg.norm(direction)) * force_magnitude
    else:
        action = np.array([0.0, 0.0])  # No force if already at the goal

    # Ensure the action stays within [-1, 1] for each dimension
    action = np.clip(action, -1, 1)
    
    return action

# Run a single episode with the manual policy
frames = []
obs, _ = env.reset()
truncated, done = False, False
while not (truncated or done):
    env.render()  # Optional, for visualization
    action = manual_policy(obs)
    obs, reward, truncated, done, info = env.step(action)
    print(obs['achieved_goal'])
    print(info)

    frame = env.render()
    frames.append(frame)

imageio.mimsave('/home/yuxuanli/failed_IRL_new/Deep_Model/Maze/simulation.mp4', frames, fps=30)
env.close()