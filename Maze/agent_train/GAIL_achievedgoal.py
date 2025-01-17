import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import wandb
import gymnasium as gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze import TrapMazeEnv
from TD3 import TD3, ReplayBuffer

# 引入专家演示数据
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

expert_states, expert_actions = extract_obs_and_actions(success_demos)

# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出为 [0, 1] 概率
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
env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="rgb_array", max_episode_steps=100, camera_name="topview")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化判别器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = expert_states.shape[1]
action_dim = expert_actions.shape[1]
max_action = env.action_space.high[0]
discriminator = Discriminator(state_dim, action_dim).to(device)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
bce_loss = nn.BCELoss()

# 定义伪奖励函数
def compute_airl_reward(state, action):
    with torch.no_grad():
        logits = discriminator(state, action)
        reward = -torch.log(1 - logits + 1e-8)
    return reward

# 初始化生成器（TD3）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td3_agent = TD3(state_dim, action_dim, max_action)

# ReplayBuffer
replay_buffer = ReplayBuffer(buffer_size=100000)

# 训练参数
batch_size = 32
episodes = 500
disc_epochs = 5  # 每次训练判别器的轮数

wandb.init(
    project="TD3-TrapMaze",  # 替换为你的项目名称
    name='GAIL',
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

# 开始训练
for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    done, truncated = False, False

    while not (done or truncated):
        # 从生成器中选择动作
        action = td3_agent.select_action(state["achieved_goal"])
        next_state, reward, done, truncated, _ = env.step(action)

        # 计算伪奖励
        state_tensor = torch.FloatTensor(state["achieved_goal"]).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
        pseudo_reward = compute_airl_reward(state_tensor, action_tensor).item()

        # 存入经验回放
        replay_buffer.add(state["achieved_goal"], action, pseudo_reward, next_state["achieved_goal"], done, truncated)
        state = next_state
        episode_reward += reward

        # 训练生成器
        if replay_buffer.size() > batch_size:
            td3_agent.train(replay_buffer, batch_size)

    # 每轮更新判别器
    for _ in range(disc_epochs):
        # 从经验回放中采样生成器数据
        gen_states, gen_actions, _, _, _ = replay_buffer.sample(batch_size)

        # 转换为 GPU 张量
        gen_states = gen_states.to(device)
        gen_actions = gen_actions.to(device)

        # 从专家演示中采样
        idx = np.random.choice(len(expert_states), batch_size)
        expert_states_batch = torch.FloatTensor(expert_states[idx]).to(device)
        expert_actions_batch = torch.FloatTensor(expert_actions[idx]).to(device)

        # 构造判别器标签
        expert_labels = torch.ones((batch_size, 1)).to(device)
        gen_labels = torch.zeros((batch_size, 1)).to(device)

        # 计算判别器损失
        expert_logits = discriminator(expert_states_batch, expert_actions_batch)
        gen_logits = discriminator(gen_states, gen_actions)
        disc_loss = bce_loss(expert_logits, expert_labels) + bce_loss(gen_logits, gen_labels)

        # 优化判别器
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

    # 使用 wandb 记录奖励和判别器损失
    wandb.log({"Episode": episode, "Episode Reward": episode_reward, "Discriminator Loss": disc_loss.item()})
    print(f"Episode: {episode}, Reward: {episode_reward}, Discriminator Loss: {disc_loss.item()}")

# 保存训练结果
torch.save(td3_agent.actor.state_dict(), "td3_airl_actor.pth")
torch.save(discriminator.state_dict(), "airl_discriminator.pth")