import numpy as np
import gymnasium as gym
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from typing import List, Union, Optional, Dict
from os import path
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium.envs.registration import register
import imageio

class TrapMazeEnv(PointMazeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_trap = False  

    def step(self, action):
        current_position = self.point_env.data.qpos[:2].copy()

        for trap_pos in self.maze._trap_locations:
            if np.linalg.norm(current_position - trap_pos) <= 0.2:  
                self.in_trap = True
                break

        if self.in_trap:
            self.point_env.data.qvel[:] = 0  
            action = np.zeros_like(action)  

        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.in_trap = False  
        return super().reset(*args, **kwargs)

# Define an example maze map with 't' as a trap cell
example_map = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 'g', 1], 
    [1, 1, 't', 0, 1],
    [1, 'r', 0, 0, 1],
    [1, 1, 1, 1, 1]
]
register(
    id="TrapMazeEnv",  # 自定义环境的唯一 ID
    entry_point="__main__:TrapMazeEnv"  # 指向 TrapMazeEnv 的类路径
)
env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="rgb_array", max_episode_steps=500)


def manual_policy(observation):
    agent_pos = obs['achieved_goal'] # Agent's current position (x, y)
    goal_pos = obs['desired_goal']  # Goal position (x, y)
    
    # Calculate the direction vector to the goal
    direction = goal_pos - agent_pos
    
    # Define the magnitude of the force (constant or scaled by distance)
    force_magnitude = 0.1  # Max force (tuned for environment)
    
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