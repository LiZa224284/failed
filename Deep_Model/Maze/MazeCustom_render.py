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
from gymnasium_robotics.envs.maze.maps import U_MAZE


class TrapMazeEnv(PointMazeEnv):
    def __init__(self, *args, **kwargs):
        self.in_trap = False 
        point_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "/home/yuxuanli/failed_IRL_new/Deep_Model/Maze/assets/point.xml"
        )
        super().__init__(*args, point_xml_file_path=point_xml_file_path, **kwargs)

    def step(self, action):
        current_position = self.point_env.data.qpos[:2].copy()

        for trap_pos in self.maze._trap_locations:
            if np.linalg.norm(current_position - trap_pos) <= 0.01:  
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
    [1, 'r', 0, 't', 1],
    [1, 1, 1, 1, 1]
]

# Instantiate the custom point maze environment
# env = TrapMazeEnv(maze_map=example_map, render_mode="rgb_array")

register(
    id="TrapMazeEnv",  # 自定义环境的唯一 ID
    entry_point="__main__:TrapMazeEnv",  # 指向 TrapMazeEnv 的类路径
    max_episode_steps=100,  # 设置最大步数
)
env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

obs, info = env.reset()
done = False
# env.unwrapped.model.cam_name2id("topview")
# print(dir(env.unwrapped))
# print(dir(env.unwrapped.point_env))

frames = []


for _ in range(1):
    state, _ = env.reset()
    truncated, done = False, False

    while not (truncated or done) :
        # action = random_policy(env)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        # Capture the frame
        # frame = env.render()
        frame = env.render()
        frames.append(frame)

imageio.mimsave('/home/yuxuanli/failed_IRL_new/Deep_Model/Maze/new_trap_simulation.mp4', frames, fps=30)
env.close()