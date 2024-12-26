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
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return  self.net(state)
    
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
        self.value = np.zeros((max_size, 1))
        self.log_prob = np.zeros((max_size, 1))

        self.device = device


    def add(self, state, action, next_state, reward, done_bool, value, log_prob):
        # idx = self.ptr % self.max_size
        # self.state[idx] = state
        # self.action[idx] = action
        # self.next_state[idx] = next_state
        # self.reward[idx] = reward
        # self.done_bool[idx] = done_bool
        # self.value[idx] = value
        # self.log_prob[idx] = log_prob

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done_bool[self.ptr] = done_bool
        self.value[self.ptr] = value
        self.log_prob[self.ptr] = log_prob

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        # ind = np.random.randint(0, self.ptr, size=batch_size)
        # ind = np.random.randint(0, self.size, size=batch_size)
        # effective_batch_size = min(batch_size, self.size)
        # ind = np.random.choice(self.size, size=effective_batch_size, replace=False)
        size = min(self.ptr, self.max_size)

        return (
            torch.FloatTensor(self.state[:size]).to(self.device),
            torch.FloatTensor(self.action[:size]).to(self.device),
            torch.FloatTensor(self.reward[:size]).to(self.device),
            torch.FloatTensor(self.next_state[:size]).to(self.device),
            torch.FloatTensor(self.done_bool[:size]).to(self.device),
            torch.FloatTensor(self.value[:size]).to(self.device),
            torch.FloatTensor(self.log_prob[:size]).to(self.device),
        )
    
    def clear(self):
        self.ptr = 0

# 定义 TD3 算法
class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        actor_lr=1e-4, # 1e-3
        critic_lr=1e-3,
        gamma=0.99,
        # tau=0.005,
        lmbda=0.95,
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
        self.critic = Critic(state_dim, action_dim).to(self.device)


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
        # self.tau = tau
        self.lmbda = lmbda
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action
    
    def compute_gae(self, rewards, values, dones, next_values):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas).to(self.device)
        advantage = 0.0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.lmbda * (1 - dones[t]) * advantage
            advantages[t] = advantage
        return advantages

    def train(self):
        self.total_it += 1

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Sample batch from replay buffer
        states, actions, rewards, next_states, done_bools, values, old_log_probs = self.replay_buffer.sample(self.batch_size)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        next_values = torch.cat((values[1:], torch.zeros(1, 1).to(self.device)))
        # next_values = self.critic(next_states)
        advantages = self.compute_gae(rewards, values, done_bools, next_values)
        td_target = rewards + self.gamma * next_values * (1 - done_bools)

        # td_target = rewards + self.gamma * self.critic(next_states) * (1 - done_bool)
        # td_delta = td_target - self.critic(states)

        # advantages = []
        # advantage = 0.0
        # td_delta = td_delta.cpu().detach().numpy()  # Convert to numpy for computation
        # for delta in reversed(td_delta):  # Iterate from the end to the beginning
        #     advantage = delta + self.gamma * self.lmbda * advantage
        #     advantages.insert(0, advantage)
        # advantages = torch.FloatTensor(advantages).to(self.device)

        # log_std = torch.zeros_like(self.actor(states)) + 0.2  # 假设固定标准差
        # std = torch.exp(log_std)
        # dist = torch.distributions.Normal(self.actor(states), std)
        # old_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True).detach()

        for i in range(50):
            dist = torch.distributions.Normal(self.actor(states), 0.2)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            wandb.log({"Critic Loss": critic_loss})
            wandb.log({"Actor Loss": actor_loss})
            



if __name__ == "__main__":

    wandb.init(
        project="TrapMaze_1225",  # 替换为你的项目名称
        name='PPO',
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
    env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="sparse", render_mode="rgb_array", max_episode_steps=300, camera_name="topview")
    # env = gym.make('InvertedPendulum-v4')
    # 定义状态维度和动作维度
    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6))  
    td3_agent = PPO(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)

    # ReplayBuffer
    
    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(5e6)#int(500e3)
    start_timesteps = 100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state, _ = env.reset()
    state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
    # state = np.concatenate([state[key].flatten() for key in ['achieved_goal', 'desired_goal']])
    done, truncated = False, False
    success_buffer = []
    episode_rewards = []  # 用于存储最近的 episode 奖励
    avg_episode_reward_window = 50

    for t in range(max_timsteps):
        episode_timesteps += 1

        if t < 10000:
            action = env.action_space.sample()
        else:
            action = td3_agent.select_action(state=state)

        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.concatenate([next_state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
        done = torch.tensor(done, dtype=torch.bool)
        truncated = torch.tensor(truncated, dtype=torch.bool)
        done_bool = done
        value = td3_agent.critic(torch.FloatTensor(state).to(device)).item()
        log_prob = torch.distributions.Normal(torch.FloatTensor(action), 0.2).log_prob(torch.FloatTensor(action)).sum().item()

        replay_buffer.add(state, action, next_state, reward, done_bool, value, log_prob)

        state = next_state
        episode_reward += reward

        
        if (done or truncated):
            td3_agent.train()
            replay_buffer.clear()

            episode_rewards.append(episode_reward)
            if len(episode_rewards) > avg_episode_reward_window:
                episode_rewards.pop(0)
            avg_episode_reward = np.mean(episode_rewards)

            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            wandb.log({"Episode Reward": episode_reward})
            wandb.log({"average Episode Reward": avg_episode_reward})

            if info['success'] == True:
                success_buffer.append(1)
            elif info['success'] == False:
                success_buffer.append(0)
            else:
                print('############### Wrong!! ##########################')
            
            # 每10个episode计算一次平均success rate
            if (t + 1) % 10 == 0:
                avg_success = np.mean(success_buffer[-10:])  # 最近10个episode的平均成功率
                print(f"Episode {episode_num+1}, Average Success Rate (last 10 eps): {avg_success:.2f}")
                wandb.log({"Average Success Rate (last 10 eps)": avg_success})


            state, _ = env.reset()
            state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            # state = np.concatenate([state[key].flatten() for key in ['achieved_goal', 'desired_goal']])
            done, truncated = False, False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
    
    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "/failed/Maze/update_baselines/models/td3_actor_obs.pth")