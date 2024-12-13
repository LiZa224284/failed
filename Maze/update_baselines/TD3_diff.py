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
import higher

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
        # self.reward = np.zeros((max_size, 1))
        self.reward = torch.zeros_like((torch.Tensor(1)))
        self.done_bool = np.zeros((max_size, 1))

        self.device = device


    def add(self, state, action, next_state, reward, done_bool):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done_bool[self.ptr] = done_bool

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        # ind = np.random.randint(0, self.size, size=batch_size)
        ind = np.random.randint(0, self.size, size=batch_size)
        # effective_batch_size = min(batch_size, self.size)
        # ind = np.random.choice(self.size, size=effective_batch_size, replace=False)

        return (
        torch.FloatTensor(self.state[ind]).to(self.device),
        torch.FloatTensor(self.action[ind]).to(self.device),
        torch.FloatTensor(self.reward[ind]).to(self.device),
        torch.FloatTensor(self.next_state[ind]).to(self.device),
        torch.FloatTensor(self.done_bool[ind]).to(self.device),
        )

# 定义 TD3 算法
class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        actor_lr=1e-4, # 1e-3
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        noise_std=0.2,
        noise_clip=0.5,
        policy_delay=1,
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

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10000, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=10000, gamma=0.9)

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

    def get_actor_params(self):
        # 返回actor的参数，用于outer loop获取
        return [p for p in self.actor.parameters()]

    def set_actor_params(self, params):
        # 将params拷贝给actor的参数，用于outer loop还原actor初始参数
        with torch.no_grad():
            for p, dp in zip(self.actor.parameters(), params):
                p.copy_(dp)

    def inner_update_actor_critic(self, reward_net, inner_steps=2):

        self.total_it += 1

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Sample batch from replay buffer
        states, actions, rewards, next_states, done_bool = self.replay_buffer.sample(self.batch_size)

        initial_actor_params = [p.clone() for p in self.actor.parameters()]

        with higher.innerloop_ctx(self.actor, self.actor_optimizer, copy_initial_weights=True) as (actor_diff, actor_optimizer_diff):
            # inner loop对critic可以直接用梯度下降更新（如果希望critic也在inner loop中可微跟踪，可用higher包装critic）
            # 这里简化处理，只对actor做inner update，可根据需要将critic也用higher包装。
            for step in range(inner_steps):
                # Critic更新（不一定需要higher，这里简化用普通更新）
                with torch.no_grad():
                    noise = (torch.randn_like(actions)*self.noise_std).clamp(-self.noise_clip, self.noise_clip)
                    next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
                    target_q1, target_q2 = self.critic_target(next_states, next_actions)
                    target_q = (1 - done_bool)*self.gamma*torch.min(target_q1, target_q2)

                with higher.innerloop_ctx(self.critic, self.critic_optimizer, copy_initial_weights=True) as (critic_diff, critic_optimizer_diff):
                    current_q1, current_q2 = critic_diff(states, actions)
                    critic_loss = F.mse_loss(current_q1, rewards + target_q.detach()) + F.mse_loss(current_q2, rewards + target_q.detach())
                    # critic_optimizer_diff.zero_grad()
                    # critic_loss.backward()
                    # critic_optimizer_diff.step()
                    critic_optimizer_diff.step(critic_loss)
                    wandb.log({"Critic Loss": critic_loss})

                    # Actor更新（通过actor_diff更新）
                    actor_loss = -critic_diff.Q1(states, actor_diff(states)).mean()
                    actor_optimizer_diff.step(actor_loss)
                    wandb.log({"Actor Loss": actor_loss})

            # inner loop结束后，actor_diff中是更新后的参数
            updated_actor_params = [p.clone() for p in actor_diff.parameters()]

        # 更新target网络
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

        # 将actor参数恢复到初始状态（outer loop需要对actor再次进行测试时保持初始参数）
        self.set_actor_params(initial_actor_params)

        return updated_actor_params