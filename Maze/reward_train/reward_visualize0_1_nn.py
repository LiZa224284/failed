import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
# import pygame
import gymnasium as gym
import gymnasium_robotics
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

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# 加载成功和失败的演示数据
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'  # 替换为成功演示文件的路径
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_failed_demos.pkl'  # 替换为失败演示文件的路径
# success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_success_demos.pkl'
# failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_failed_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

# 提取状态（obs）和标签（reward）
def extract_obs_and_labels(demos, label, exp_k=0):
    obs = []
    rewards = []
    for traj in demos:
        demo_length = len(traj)
        for i, step in enumerate(traj):
            # 提取 achieved_goal
            achieved_goal = step["state"]['achieved_goal']
            obs.append(achieved_goal)

            # 计算时间步对应的权重
            weight = (i + 1) / demo_length  # 线性增加
            weight = weight * np.exp(exp_k * weight)  # 应用指数调制

            # 成功和失败的 reward 设置
            if label > 0:  # 成功 demo
                # reward = label * weight
                reward = 1
            else:  # 失败 demo
                # reward = label * weight
                reward = -1

            rewards.append(reward)
    return np.array(obs), np.array(rewards)

# 从成功和失败的 demos 中提取数据
success_obs, success_rewards = extract_obs_and_labels(success_demos, 1, exp_k=1)
failed_obs, failed_rewards = extract_obs_and_labels(failed_demos, -1, exp_k=1)

# 合并数据
X = np.concatenate((success_obs, failed_obs), axis=0)
y = np.concatenate((success_rewards, failed_rewards), axis=0)

# 转换为 PyTorch 张量
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=512, shuffle=True) #256
# 定义奖励函数的神经网络
class RewardNetwork(nn.Module):
    def __init__(self, input_dim):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出为连续值，不使用 Sigmoid
        )
    
    def forward(self, x):
        return self.net(x)

# 初始化神经网络和优化器
reward_net = RewardNetwork(input_dim=X.shape[1]).to(device)
optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
loss_fn = nn.MSELoss()

# 训练奖励函数
epochs = 500
for epoch in tqdm(range(epochs), desc='Epochs'):
    reward_net.train()
    epoch_loss = 0.0

    for X_batch, Y_batch in tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False):
        # predictions = reward_net(X_batch).squeeze()
        predictions = reward_net(X_batch)
        loss = loss_fn(predictions, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_loader):.4f}, Learning Rate: {current_lr:.6f}")

    # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

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
    reward_net.eval()

    grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32).to(device)
    grid_dataset = TensorDataset(grid_states_tensor)
    grid_loader = DataLoader(grid_dataset, batch_size=512, shuffle=False)

    predictions = []
    for batch in grid_loader:
        batch_grid_states = batch[0]  # 从 DataLoader 中取出当前批次
        with torch.no_grad():  # 禁用梯度计算以节省显存
            batch_predictions = reward_net(batch_grid_states)
            predictions.append(batch_predictions.cpu().numpy())  # 将结果移动到 CPU 并存储

    # with torch.no_grad():
    #     rewards = reward_net(grid_states_tensor).numpy().reshape(xx.shape)
    # 拼接所有批次的预测结果
    predictions = np.concatenate(predictions, axis=0)

    # 将预测结果重塑为网格形状
    rewards = predictions.reshape(xx.shape)
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