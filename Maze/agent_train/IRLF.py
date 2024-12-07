import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from collections import deque
import numpy as np
import wandb
import gymnasium as gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze import TrapMazeEnv
from TD3 import TD3, ReplayBuffer

success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_failed_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

# 提取状态和动作
def extract_obs_and_actions(demos):
    obs = []
    actions = []
    for traj in demos:
        for step in traj:
            obs.append(step["state"]["achieved_goal"])
            actions.append(step["action"])
    return np.array(obs), np.array(actions)

success_states, success_actions = extract_obs_and_actions(success_demos)
failed_states, failed_actions = extract_obs_and_actions(failed_demos)

# 定义特征提取网络
class FeatureNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=32):
        super(FeatureNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

example_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1], 
    [1, 0, 1, 0, 1, 0, 1], 
    [1, 0, 1, 'g', 't', 0, 1], 
    [1, 0, 't', 0, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1]
]
# 初始化环境
env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

expert_states, expert_actions = extract_obs_and_actions(success_demos)
state_dim = expert_states.shape[1]
action_dim = expert_actions.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 奖励网络和策略网络
reward_net = FeatureNetwork(state_dim, action_dim).to(device)
reward_optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)
td3_agent = TD3(state_dim, action_dim, env.action_space.high[0])
replay_buffer = ReplayBuffer(buffer_size=100000)

# 超参数
batch_size = 32
episodes = 500
lambda_factor = 1.0
alpha = 0.01  # 学习率
min_lambda = 0.01  # 最小的 lambda 值

# 训练过程
for episode in range(episodes):
    state, _ = env.reset()
    done, truncated = False, False
    episode_reward = 0

    while not (done or truncated):
        # 使用策略选择动作
        action = td3_agent.select_action(state["achieved_goal"])
        next_state, _, done, truncated, _ = env.step(action)

        # 计算奖励
        state_tensor = torch.FloatTensor(state["achieved_goal"]).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
        features = reward_net(state_tensor, action_tensor)
        reward = torch.sum(features).item()  # 奖励是特征的加权和

        # 存入经验回放
        replay_buffer.add(state["achieved_goal"], action, reward, next_state["achieved_goal"], done)
        state = next_state
        episode_reward += reward

        if replay_buffer.size() > batch_size:
            td3_agent.train(replay_buffer, batch_size)

    # 更新奖励网络
    for _ in range(10):  # 每轮训练多次
        # 从成功轨迹中采样
        idx = np.random.choice(len(success_states), batch_size)
        success_states_batch = torch.FloatTensor(success_states[idx]).to(device)
        success_actions_batch = torch.FloatTensor(success_actions[idx]).to(device)

        # 从失败轨迹中采样
        idx = np.random.choice(len(failed_states), batch_size)
        failed_states_batch = torch.FloatTensor(failed_states[idx]).to(device)
        failed_actions_batch = torch.FloatTensor(failed_actions[idx]).to(device)

        # 成功特征期望
        success_features = reward_net(success_states_batch, success_actions_batch)
        success_loss = torch.mean(success_features)  # 匹配成功轨迹的特征

        # 失败特征期望
        failed_features = reward_net(failed_states_batch, failed_actions_batch)
        failed_loss = -torch.mean(failed_features) / lambda_factor  # 增加失败轨迹的约束

        # 总损失
        reward_loss = success_loss + failed_loss

        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()

    # 动态调整 lambda
    lambda_factor = max(min_lambda, lambda_factor * 0.99)

    print(f"Episode {episode}: Reward {episode_reward}, Reward Loss {reward_loss.item()}")

# 保存模型
torch.save(td3_agent.actor.state_dict(), "irlf_actor.pth")
torch.save(reward_net.state_dict(), "irlf_reward_net.pth")