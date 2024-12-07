import wandb
import sys
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze import TrapMazeEnv
from TD3 import TD3, ReplayBuffer
# TD3 相关定义（略，直接使用您已有的 TD3 和 ReplayBuffer 实现）

# 定义奖励函数网络
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

# 加载训练的奖励函数模型
def load_reward_network(model_path, input_dim, device):
    reward_net = RewardNetwork(input_dim=input_dim).to(device)
    reward_net.load_state_dict(torch.load(model_path, map_location=device))
    reward_net.eval()
    return reward_net

# 定义替代环境奖励的函数
def compute_custom_reward(reward_net, state):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        reward = reward_net(state_tensor).item()
    return reward

if __name__ == "__main__":
    # 初始化 wandb
    wandb.init(
        project="TD3-CustomReward",  # 替换为你的项目名称
        name='TD3 my Method',
        config={
            "batch_size": 256,
            "buffer_size": 100000,
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

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载环境
    example_map = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1], 
        [1, 0, 1, 0, 1, 0, 1], 
        [1, 0, 1, 'g', 't', 0, 1], 
        [1, 0, 't', 0, 0, 0, 1], 
        [1, 1, 1, 1, 1, 1, 1]
    ]
    env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

    # 环境和模型参数
    state_dim = env.observation_space["achieved_goal"].shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # 加载训练好的 Reward Network
    reward_model_path = "/home/yuxuanli/failed_IRL_new/Maze/reward_train/reward_function_model.pth"
    reward_net = load_reward_network(reward_model_path, state_dim, device)

    # 初始化 TD3 Agent
    td3_agent = TD3(state_dim, action_dim, max_action, device=device)

    # ReplayBuffer
    replay_buffer = ReplayBuffer(buffer_size=100000)  # 调整为适合的 buffer size
    batch_size = 32
    episodes = int(3e6)

    # 开始训练
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            # 选择动作并执行
            action = td3_agent.select_action(state["achieved_goal"])
            next_state, _, done, truncated, _ = env.step(action)

            # 使用自定义奖励函数
            custom_reward = compute_custom_reward(reward_net, state["achieved_goal"])

            # 存入 ReplayBuffer
            replay_buffer.add(state["achieved_goal"], action, custom_reward, next_state["achieved_goal"], done)
            state = next_state
            episode_reward += custom_reward

            # 训练 TD3
            if replay_buffer.size() > batch_size:
                td3_agent.train(replay_buffer, batch_size)

        # 记录当前 Episode 的奖励
        rewards.append(episode_reward)
        print(f"Episode: {episode}, Custom Reward: {episode_reward}")

        # 使用 wandb 记录每个 Episode 的奖励
        wandb.log({"Episode": episode, "Custom Reward": episode_reward})

    wandb.finish()

    # 保存训练好的 TD3 模型
    td3_actor_path = "/home/yuxuanli/failed_IRL_new/Maze/td3_actor_myMethod.pth"
    torch.save(td3_agent.actor.state_dict(), td3_actor_path)
    print(f"TD3 Actor model saved to {td3_actor_path}")