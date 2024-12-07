import wandb  # 添加 wandb
import sys
import os
import gymnasium as gym
import gymnasium_robotics
# 获取当前脚本所在路径的上一级路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze_action import TrapMazeEnv
# from Maze.TrapMaze import TrapMazeEnv
import imageio

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

# 定义网络结构
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            # nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        # x = state
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

# 定义经验回放
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done, truncated):
        self.buffer.append((state, action, reward, next_state, done, truncated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, truncated = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),       
            torch.FloatTensor(np.array(actions)),      
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),  
            torch.FloatTensor(np.array(next_states)), 
            torch.FloatTensor(np.array(dones)).unsqueeze(1),   
            torch.FloatTensor(np.array(truncated)).unsqueeze(1),  
        )

    def size(self):
        return len(self.buffer)

# 定义 TD3 算法
class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        noise_std=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_size=100000,
        batch_size=256,
        device='cuda'
    ):
        self.device = device

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Training Parameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        # Add exploration noise
        action += noise * np.random.normal(size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return None, None, None  # Return default values when there's not enough data

        self.total_it += 1

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, truncateds = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device).squeeze(1) 
        # states = np.concatenate([states[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']]).to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device).squeeze(1) 
        # next_states = np.concatenate([next_state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']]).to(self.device)
        
        dones = dones.to(self.device)
        truncateds = truncateds.to(self.device)
        # done_flag = float(dones or truncateds)
        done_flags = torch.logical_or(dones, truncateds).float()

        # Target Policy Smoothing
        noise = (torch.randn_like(actions) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-values
        target_q1 = self.critic_target_1(next_states, next_actions)
        target_q2 = self.critic_target_2(next_states, next_actions)
        target_q = rewards + (1 - done_flags) * self.gamma * torch.min(target_q1, target_q2)

        # Optimize Critic
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_1_loss = nn.MSELoss()(current_q1, target_q.detach())
        critic_2_loss = nn.MSELoss()(current_q2, target_q.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        actor_loss = None  # Initialize actor_loss to a default value
        # Delayed Policy Updates
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(states, self.actor(states)).mean()

            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Return the computed losses (actor_loss will be None if not updated)
        return critic_1_loss.item(), critic_2_loss.item(), actor_loss.item() if actor_loss is not None else None 

if __name__ == "__main__":

    wandb.init(
        project="TrapMaze_1127",  # 替换为你的项目名称
        name='TD3_dense',
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    example_map = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 'g', 't', 0, 1],
        [1, 0, 't', 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="dense", render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

    # 定义状态维度和动作维度
    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    td3_agent = TD3(state_dim, action_dim, max_action, device=device)

    # ReplayBuffer
    replay_buffer = ReplayBuffer(buffer_size=100000)  
    batch_size = 32
    episodes = int(5e6)

    # 开始训练
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            # 提取并处理当前状态
            obs = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            # obs = np.expand_dims(obs, axis=0)

            # 选择动作
            action = td3_agent.select_action(obs)
            next_state, reward, done, truncated, _ = env.step(action)

            # 提取并处理下一状态
            next_obs = np.concatenate([next_state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            next_obs = np.expand_dims(next_obs, axis=0)

            # 存入经验回放
            td3_agent.replay_buffer.add(obs, action, reward, next_obs, done, truncated)
            td3_agent.train()
            state = next_state
            episode_reward += reward

        # 记录当前 episode 的奖励
        wandb.log({"Episode Reward": episode_reward})
        print(f"Episode: {episode}, Reward: {episode_reward}")

    wandb.finish()

    # 保存训练后的 actor 网络
    torch.save(td3_agent.actor.state_dict(), "td3_actor_2.pth")