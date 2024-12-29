import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import wandb  
import sys
import os
import copy
import pickle
import gymnasium as gym
import gymnasium_robotics
from TrapMazeEnv import TrapMazeEnv
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from TD3 import TD3, ReplayBuffer
import matplotlib.pyplot as plt
import argparse
# 环境初始化
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_success_demos_16.pkl'
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_failed_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

def extract_obs_and_actions(demos):
    obs = []
    actions = []
    for traj in demos:
        for step in traj:
            state = step["state"]
            obs_input = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            obs.append(obs_input)
            actions.append(step["action"])
    return np.array(obs), np.array(actions)
    
expert_states, expert_actions = extract_obs_and_actions(success_demos)

class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出奖励值
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

reward_net = RewardNetwork(state_dim, action_dim)
reward_optimizer = optim.Adam(reward_net.parameters(), lr=0.01)

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

if __name__ == "__main__":
    wandb.init(
        project="TrapMaze_1200_0",  # 替换为你的项目名称
        name='BCIRL_v2',
        config={
            "batch_size": 256,
            "buffer_size": int(1e6),
            "episodes": 10,
            "actor_lr": 1e-3,
            "critic_lr": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "noise_std": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "max_episode_steps": 10,
        },
    )

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    example_map = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 'r', 1, 'g', 't', 0, 1],
        [1, 0, 't', 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
        ]
    env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="sparse", render_mode="rgb_array", max_episode_steps=300, camera_name="topview")

    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(device=device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)

    reward_net = RewardNetwork(state_dim, action_dim).to(device=device)
    reward_optimizer = optim.Adam(reward_net.parameters(), lr=0.001)

    for episode in range(500):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device=device)
        done, truncated = False, False
        episode_reward = 0
        while not (done or truncated):
            idx = np.random.choice(len(expert_states), batch_size)
            expert_states_batch = torch.FloatTensor(expert_states[idx]).to(device)
            expert_actions_batch = torch.FloatTensor(expert_actions[idx]).to(device)
            pred_actions = td3_agent.actor(expert_states_batch)

            actor_loss = nn.MSELoss(expert_actions_batch, pred_actions)
            # Actor网络更新（使用梯度上升优化策略）
            actor_optimizer.zero_grad()
            actor_loss = -reward.mean()  # 最大化奖励
            actor_loss.backward()
            actor_optimizer.step()
            wandb.log({"actor loss": actor_loss})

            # # 使用Actor网络选择动作
            # action = actor(state)
            # next_state, _, done, _ = env.step(action.detach().numpy())
            # next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device=device)
            
            # 计算模拟的奖励
            reward = reward_net(state, action).to(device=device)
            
            
            
            # 奖励网络更新
            reward_optimizer.zero_grad()
            reward_loss = -actor_loss.detach()  # 最小化负的策略损失，即最大化策略损失
            reward_loss.backward()
            reward_optimizer.step()
            wandb.log({"reward net loss": reward_loss})

            state = next_state

            episode_reward += reward
            wandb.log({"episode reward": episode_reward})