import wandb  # 添加 wandb
import sys
import os
import copy
import gymnasium as gym
import gymnasium_robotics
# 获取当前脚本所在路径的上一级路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze_action import TrapMazeEnv
# from Maze.TrapMaze import TrapMazeEnv
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plts

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0, hidden_dim=256):
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

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done_bool = np.zeros((max_size, 1))
        # self.truncated = np.zeros((max_size, 1))

        self.device = device


    def add(self, state, action, next_state, reward, done_bool):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done_bool[self.ptr] = done
        # self.truncated[self.ptr] = truncated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
        torch.FloatTensor(self.state[ind]).to(self.device),
        torch.FloatTensor(self.action[ind]).to(self.device),
        torch.FloatTensor(self.reward[ind]).to(self.device),
        torch.FloatTensor(self.next_state[ind]).to(self.device),
        torch.FloatTensor(self.done_bool[ind]).to(self.device),
        # torch.FloatTensor(self.truncated[ind]).to(self.device),
        )

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
        buffer_size=int(1e6),
        batch_size=256,
        device='cuda',
        ReplayBuffer=None,
    ):
        self.device = device

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim=256).to(self.device)
        # self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer
        self.batch_size = batch_size

        # Training Parameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self):
        self.total_it += 1

        # Sample batch from replay buffer
        states, actions, rewards, next_states, done_bool = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Target Policy Smoothing
            noise = (torch.randn_like(actions) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - done_bool) * self.gamma * torch.min(target_q1, target_q2)

        # Optimize Critic
        current_q1, current_q2 = self.critic(states, actions)
        critic_1_loss = nn.MSELoss()(current_q1, target_q.detach())
        critic_2_loss = nn.MSELoss()(current_q2, target_q.detach())
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        wandb.log({"Critic Loss": critic_loss})

        # Delayed Policy Updates
        if self.total_it % self.policy_delay == 0:

            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            wandb.log({"Actor Loss": actor_loss})

            # Update Target Networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

if __name__ == "__main__":

    wandb.init(
        project="TrapMaze_1203",  # 替换为你的项目名称
        name='TD3_dense',
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
    env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="dense", render_mode="rgb_array", max_episode_steps=100, camera_name="topview")
    # env = gym.make('InvertedPendulum-v4')
    # 定义状态维度和动作维度
    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6))  
    td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)

    # ReplayBuffer
    
    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(5e6)
    start_timesteps = 100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state, _ = env.reset()
    state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
    done, truncated = False, False

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

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t > start_timesteps:
            td3_agent.train()
        
        if (done or truncated):
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            wandb.log({"Episode Reward": episode_reward})
            state, _ = env.reset()
            state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            done, truncated = False, False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
    
    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "td3_actor_2.pth")