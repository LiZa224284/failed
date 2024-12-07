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
import matplotlib.pyplot as plts
from TD3 import TD3, ReplayBuffer

success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_success_demos.pkl'
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_failed_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

def extract_obs_and_actions(demos, label, exp_k=0):
    obs = []
    actions = []
    rewards = []
    for traj in demos:
        demo_length = len(traj)
        for i, step in enumerate(traj):
            state = step["state"]
            obs_input = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            obs.append(obs_input)
            actions.append(step["action"])

            # exp_k = 1
            weight = (i + 1) / demo_length  # 线性增加
            weight = weight * np.exp(exp_k * weight)  # 应用指数调制

            # 成功和失败的 reward 设置
            reward = label * weight
            rewards.append(reward)

    return np.array(obs), np.array(actions), np.array(rewards)
    
expert_states, expert_actions, success_rewards = extract_obs_and_actions(success_demos, 1, exp_k=1)
failed_states, failed_actions, failed_rewards = extract_obs_and_actions(failed_demos, -1, exp_k=1)

# 合并数据
X = np.concatenate((expert_states, failed_states), axis=0)
y = np.concatenate((success_rewards, failed_rewards), axis=0)

# 转换为 PyTorch 张量并移动到 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

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
reward_net = RewardNetwork(input_dim=X.shape[1]).to(device)  # 将模型移动到 GPU
optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 训练奖励函数
epochs = 200
model_save_path = "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/my_reward.pth"  # 保存模型的路径

for epoch in range(epochs):
    reward_net.train()
    optimizer.zero_grad()
    predictions = reward_net(X_tensor).squeeze()
    loss = loss_fn(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 保存模型
torch.save(reward_net.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


def compute_gail_reward(state, action):
    with torch.no_grad():
        logits = discriminator(state, action)
        reward = -torch.log(1 - logits + 1e-8)
    return reward


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
        x = torch.cat([state, action], 1)
        return self.net(x)

if __name__ == "__main__":

    wandb.init(
        project="TrapMaze_1205",  
        name='MyMethod_org_sparse',
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    example_map = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 'g', 't', 0, 1],
        [1, 0, 't', 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="sparse", render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

    # 定义状态维度和动作维度
    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6))  
    td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)
    
    discriminator = Discriminator(state_dim, action_dim).to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=1000, gamma=0.9)
    bce_loss = nn.BCELoss()
    disc_epochs = 5

    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(1e5)
    start_timesteps = 100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    pseudo_episode_reward = 0
    episode_num = 0

    state, _ = env.reset()
    state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
    done, truncated = False, False


    # 加载模型
    model_path = model_save_path
    reward_net = RewardNetwork(input_dim=8).to(device)
    reward_net.load_state_dict(torch.load(model_path, map_location=device))
    reward_net.eval()  # 设置为评估模式

    for t in range(max_timsteps):
        episode_timesteps += 1
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            noise = 0.2
            action = action = td3_agent.select_action(state=state)
            action += noise * np.random.normal(size=action.shape)
            action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.concatenate([next_state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
        done = torch.tensor(done, dtype=torch.bool)
        truncated = torch.tensor(truncated, dtype=torch.bool)
        done_bool = torch.logical_or(done, truncated).float()

        state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
        action_tensor = torch.from_numpy(action).float().to(device).unsqueeze(0)
        # pseudo_reward = compute_gail_reward(state_tensor, action_tensor)
        with torch.no_grad():
            pseudo_reward = reward_net(state_tensor)
            pseudo_reward = pseudo_reward.cpu().numpy()
            pseudo_reward += reward
        replay_buffer.add(state, action, next_state, pseudo_reward, done_bool)

        state = next_state
        episode_reward += reward
        pseudo_episode_reward += pseudo_reward

        if t > start_timesteps:
            td3_agent.train()
        
        if (done or truncated):
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} PseudoReward: {pseudo_episode_reward}")
            wandb.log({"Episode Reward": episode_reward})
            wandb.log({"Pseudo Episode Reward": pseudo_episode_reward})
            state, _ = env.reset()
            state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            done, truncated = False, False
            episode_reward = 0
            pseudo_episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    
    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/my_actor.pth")
    torch.save(discriminator.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/my_discriminator.pth")