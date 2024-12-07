import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
# import pygame
import gymnasium as gym
# import gymnasium_robotics
import pickle

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

# 加载成功和失败的演示数据
# success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'  # 替换为成功演示文件的路径
# failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_failed_demos.pkl'  # 替换为失败演示文件的路径
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_success_demos.pkl'
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_failed_demos.pkl'
with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

# 提取状态（obs）和标签（reward）
def extract_obs_and_labels(demos, label):
    obs = []
    for traj in demos:
        obs.extend([step["state"]['achieved_goal'] for step in traj])
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
def visualize_reward_function(reward_net):
    # 提取环境范围
    # x_min, x_max = env.observation_space.low[0], env.observation_space.high[0]
    # y_min, y_max = env.observation_space.low[1], env.observation_space.high[1]

    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2

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
visualize_reward_function(reward_net)