import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pygame
import gymnasium as gym
import gymnasium_robotics

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

from gymnasium_robotics.envs.maze.maze_v4 import Maze
import xml.etree.ElementTree as ET
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium_robotics.envs.maze.maps import COMBINED, GOAL, RESET, U_MAZE
from gymnasium_robotics.core import GoalEnv

import tempfile
import time
TRAP = T = "t"

class modifyMaze(Maze):
    def __init__(
        self,
        maze_map: List[List[Union[str, int]]],
        maze_size_scaling: float,
        maze_height: float,
    ):

        self._maze_map = maze_map
        self._maze_size_scaling = maze_size_scaling
        self._maze_height = maze_height

        self._unique_goal_locations = []
        self._unique_reset_locations = []
        self._combined_locations = []
        self._trap_locations = []

        # Get the center cell Cartesian position of the maze. This will be the origin
        self._map_length = len(maze_map)
        self._map_width = len(maze_map[0])
        self._x_map_center = self.map_width / 2 * maze_size_scaling
        self._y_map_center = self.map_length / 2 * maze_size_scaling

    @classmethod
    def make_maze(
        cls,
        agent_xml_path: str,
        maze_map: list,
        maze_size_scaling: float,
        maze_height: float,
    ):
        tree = ET.parse(agent_xml_path)
        worldbody = tree.find(".//worldbody")

        maze = cls(maze_map, maze_size_scaling, maze_height)
        empty_locations = []
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze_size_scaling
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that maze is centered.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                        size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {maze_height / 2 * maze_size_scaling}",
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.7 0.5 0.3 1.0",
                    )

                elif struct == RESET:
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif struct == GOAL:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    maze._combined_locations.append(np.array([x, y]))
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))
                elif struct == TRAP:  
                    ET.SubElement(
                        worldbody,
                        "site",
                        name=f"trap_{i}_{j}",
                        pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                        size=f"{0.2 * maze_size_scaling}",
                        rgba="0 0 1 0.7",  
                        type="sphere",
                    )
                    maze._trap_locations.append(np.array([x, y]))  
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))


        # Add target site for visualization
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 {maze_height / 2 * maze_size_scaling}",
            size=f"{0.2 * maze_size_scaling}",
            rgba="1 0 0 0.7",
            type="sphere",
        )

        # Add the combined cell locations (goal/reset) to goal and reset
        if (
            not maze._unique_goal_locations
            and not maze._unique_reset_locations
            and not maze._combined_locations
        ):
            # If there are no given "r", "g" or "c" cells in the maze data structure,
            # any empty cell can be a reset or goal location at initialization.
            maze._combined_locations = empty_locations
        elif not maze._unique_reset_locations and not maze._combined_locations:
            # If there are no given "r" or "c" cells in the maze data structure,
            # any empty cell can be a reset location at initialization.
            maze._unique_reset_locations = empty_locations
        elif not maze._unique_goal_locations and not maze._combined_locations:
            # If there are no given "g" or "c" cells in the maze data structure,
            # any empty cell can be a gaol location at initialization.
            maze._unique_goal_locations = empty_locations

        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_name = f"ant_maze{str(time.time())}.xml"
            temp_xml_path = path.join(path.dirname(tmp_dir), temp_xml_name)
            tree.write(temp_xml_path)

        return maze, temp_xml_path

class modifyPointMazeEnv(PointMazeEnv):
    def __init__(
        self,
        agent_xml_path=None,
        maze_map=None,
        maze_size_scaling=1,
        maze_height=0.4,
        position_noise_range: float = 0.25,
        render_mode: Optional[str] = None,
        reward_type=None,
        continuing_task=None,
        reset_target=None,
        point_xml_file_path = None,
        **kwargs,
    ):

        self.reward_type = reward_type
        self.continuing_task = continuing_task
        self.reset_target = reset_target
        self.maze, self.tmp_xml_file_path = modifyMaze.make_maze(
            agent_xml_path , maze_map, maze_size_scaling, maze_height
        )

        self.position_noise_range = position_noise_range

        maze_length = len(maze_map)
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs_shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            maze_map,
            render_mode,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

class TrapMazeEnv(modifyPointMazeEnv):
    def __init__(self, *args, goal_threshould=0.2, trap_threshould=0.2, **kwargs):
        self.reached_goal = False
        self.goal_threshould = goal_threshould
        self.trap_threshould = trap_threshould
        agent_xml_path = path.join(
            path.dirname(path.realpath(__file__)), "/home/yuxuanli/failed_IRL_new/Deep_Model/Maze/assets/point.xml"
        )
        super().__init__(*args, agent_xml_path=agent_xml_path, **kwargs)

    def in_goal(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> bool:
        if np.linalg.norm(achieved_goal - desired_goal) <= self.goal_threshould:
            return True
        else:
            return False
    
    def in_trap(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> bool:
        for trap_pos in self.maze._trap_locations:
            if np.linalg.norm(achieved_goal - trap_pos) <= self.trap_threshould:
                return True
        return False

    def step(self, action):
        obs, _, _, _, info = self.point_env.step(action)
        obs_dict = self._get_obs(obs)

        reward = self.compute_reward(obs_dict["achieved_goal"], self.goal, info)
        terminated = self.compute_terminated(obs_dict["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs_dict["achieved_goal"], self.goal, info)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= self.goal_threshould
        )

        # Update the goal position if necessary
        self.update_goal(obs_dict["achieved_goal"])

        if self.in_goal(obs_dict["achieved_goal"], self.goal, info) or self.in_trap(obs_dict["achieved_goal"], self.goal, info):
            self.point_env.data.qvel[:] = 0  
            self.point_env.data.qacc[:] = 0
            action = np.zeros_like(action) 

        return obs_dict, reward, terminated, truncated, info
    
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "dense":
            return np.exp(-distance)
        elif self.reward_type == "sparse":
            return (distance <= self.goal_threshould).astype(np.float64)

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        if not self.continuing_task:
            # If task is episodic terminate the episode when the goal is reached
            # return bool(np.linalg.norm(achieved_goal - desired_goal) <= 3)
            return False
        else:
            # Continuing tasks don't terminate, episode will be truncated when time limit is reached (`max_episode_steps`)
            return False

example_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1], 
    [1, 0, 1, 0, 1, 0, 1], 
    [1, 'r', 1, 'g', 't', 0, 1], 
    [1, 0, 't', 0, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1]
]

register(
    id="TrapMazeEnv",  
    entry_point="__main__:TrapMazeEnv", 
)
env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="rgb_array", max_episode_steps=800, camera_name="topview")

# 加载成功和失败的演示数据
success_demo_path = '/home/yuxuanli/failed_IRL_new/Deep_Model/Maze/demos/success/all_success_demos.pkl'  # 替换为成功演示文件的路径
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Deep_Model/Maze/demos/failed/all_failed_demos.pkl'  # 替换为失败演示文件的路径

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

# 提取状态（obs）和标签（reward）
def extract_obs_and_labels(demos, label):
    obs = []
    for traj in demos:
        obs.extend([step["state"]['observation'] for step in traj])
    labels = [label] * len(obs)
    return np.array(obs), np.array(labels)

success_obs, success_labels = extract_obs_and_labels(success_demos, 1)
failed_obs, failed_labels = extract_obs_and_labels(failed_demos, 0)

# 合并数据
X = np.concatenate((success_obs, failed_obs), axis=0)
y = np.concatenate((success_labels, failed_labels), axis=0)

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 定义奖励函数的神经网络
class RewardNetwork(nn.Module):
    def __init__(self, input_dim):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出范围为 [0, 1]
        )
    
    def forward(self, x):
        return self.net(x)

# 初始化神经网络和优化器
reward_net = RewardNetwork(input_dim=X.shape[1])
optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# 训练奖励函数
epochs = 50
for epoch in range(epochs):
    reward_net.train()
    optimizer.zero_grad()
    predictions = reward_net(X_tensor).squeeze()
    loss = loss_fn(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 可视化奖励函数
def visualize_reward_function(env, reward_net):
    # 提取环境范围
    x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
    y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]

    # 创建网格
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid_states = np.c_[xx.ravel(), yy.ravel()]

    # 计算奖励
    grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32)
    with torch.no_grad():
        rewards = reward_net(grid_states_tensor).numpy().reshape(xx.shape)

    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="Reward")
    plt.title("Reward Function Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# 创建 Point Maze 环境
# env = PointMazeEnv()  # 替换为你实际的环境
visualize_reward_function(env, reward_net)