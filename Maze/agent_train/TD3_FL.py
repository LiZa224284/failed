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

import os
import wandb
from gymnasium import spaces
import gymnasium as gym
# from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
# from stable_baselines3.common import results_plotter
# from stable_baselines3 import TD3
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.results_plotter import load_results, ts2xy
# from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.common.callbacks import BaseCallback
import torch
# from stable_baselines3.common.evaluation import evaluate_policy

# 定义网络结构
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
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

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),       
            torch.FloatTensor(np.array(actions)),      
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),  
            torch.FloatTensor(np.array(next_states)), 
            torch.FloatTensor(np.array(dones)).unsqueeze(1),    
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
        device='cuda'
    ):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target_2 = Critic(state_dim, action_dim).to(self.device)

        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size):
        self.total_it += 1

        # 从经验回放中采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 添加目标噪声并裁剪
        noise = (
            torch.randn_like(actions) * self.noise_std
        ).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (
            self.actor_target(next_states) + noise
        ).clamp(-1, 1)

        # 计算目标 Q 值
        target_q1 = self.critic_target_1(next_states, next_actions)
        target_q2 = self.critic_target_2(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        # 更新 Critic 网络
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

        # 延迟更新 Actor 网络和目标网络
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic_1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ContinuousFrozenLakeEnv(gym.Env):
    def __init__(self, lake_size=4, hole_radius=0.1, goal_radius=0.1, max_steps=20):
        super(ContinuousFrozenLakeEnv, self).__init__()
        
        # 定义连续状态空间
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
                                            high=np.array([lake_size, lake_size]), 
                                            dtype=np.float32)
        
        # 定义连续动作空间
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
                                       high=np.array([1.0, 1.0]), 
                                       dtype=np.float32)
        
        self.lake_size = lake_size
        self.hole_radius = hole_radius
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.current_step = 0  # 初始化步数计数器
        
        # 定义洞和目标的位置
        self.holes = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [1.0, 3.0]])
        self.goal = np.array([3.5, 3.5])
        
        self.seed_value = None
        self.reset()

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed_value = seed
            np.random.seed(self.seed_value)
        
        # 将智能体置于远离洞和目标的随机位置
        while True:
            self.state = np.random.uniform(0, self.lake_size, size=(2,))
            # self.state = np.array([1.0, 3.0])
            if not self._is_in_hole(self.state) and not self._is_in_goal(self.state):
                break
        
        self.current_step = 0  # 重置步数计数器
        return self.state, {}

    def step(self, action):
        self.current_step += 1  # Increment step counter

        distance_to_goal = np.linalg.norm(self.state - self.goal)
        distance_to_holes = min(np.linalg.norm(self.state - hole) for hole in self.holes)
        reward = (-0.01 / distance_to_holes + 0.01) + (0.01 / distance_to_goal + 0.01) - 2
        # reward = -0.01 - 2

        # Check if the agent is in a hole
        if self._is_in_hole(self.state):
            # Agent is stuck in the hole but can still take actions within the hole's boundary
            hole_center = self._get_hole_center(self.state)
            potential_next_state = self.state + action
            
            # if distance_to_hole_center <= self.hole_radius:
            if  self._is_in_hole(potential_next_state):
                self.state = potential_next_state
                info = {"result": "1, The agent is now stuck in the hole"}
            else:
                while True:
                    random_tiny_action = np.random.uniform(-0.1, 0.1, size=self.state.shape)
                    tmp_state = self.state + random_tiny_action

                    if self._is_in_hole(tmp_state):
                        self.state = tmp_state
                        break
                info = {"result": "2, The agent is now stuck in the hole"}
            
            # info = {"result": "The agent is now stuck in the hole"}
            if self.current_step >= self.max_steps:
                info = {"result": "failure", "truncated": True}
                return self.state, reward, False, True, info #-0.5
            return self.state, reward, False, False, info # The episode doesn't end, but the agent is stuck

        else:
            # Update the state based on the action if the agent is not in a hole
            self.state = np.clip(self.state + action, 0.0, self.lake_size)

        # Check if the agent has reached the goal
        if self._is_in_goal(self.state):
            info = {"result": "success"}
            return self.state, 1.0 - 2 , True, False, info
        
        # Check if the agent has exceeded the maximum number of steps
        if self.current_step >= self.max_steps:
            info = {"result": "failure", "truncated": True}
            return self.state, reward, False, True, info #-0.5

        # If neither, return a small negative reward to encourage reaching the goal
        return self.state, reward, False, False, {}
    
    def render(self, mode='human'):
        plt.figure(figsize=(6, 6))
        plt.xlim(0, self.lake_size)
        plt.ylim(0, self.lake_size)
        
        # 绘制洞
        for hole in self.holes:
            circle = plt.Circle(hole, self.hole_radius, color='blue', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # 绘制目标
        goal_circle = plt.Circle(self.goal, self.goal_radius, color='green', alpha=0.5)
        plt.gca().add_patch(goal_circle)
        
        # 绘制智能体
        agent_circle = plt.Circle(self.state, 0.05, color='red')
        plt.gca().add_patch(agent_circle)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
    
    def _is_in_hole(self, pos):
        for hole in self.holes:
            if np.linalg.norm(pos - hole) <= self.hole_radius:
                return True
        return False
    
    def _is_in_goal(self, pos):
        return np.linalg.norm(pos - self.goal) <= self.goal_radius

    def _get_hole_center(self, state):
        for hole in self.holes:
            if np.linalg.norm(state - hole) <= self.hole_radius:
                self.hole_center = hole
                return True
        return None

if __name__ == "__main__":

    wandb.init(
        project="FL_1127",  # 替换为你的项目名称
        name='TD3_dense_FL',
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

    # example_map = [
    #     [1, 1, 1, 1, 1, 1, 1],
    #     [1, 0, 0, 0, 0, 0, 1], 
    #     [1, 0, 1, 0, 1, 0, 1], 
    #     [1, 0, 1, 'g', 't', 0, 1], 
    #     [1, 0, 't', 0, 0, 0, 1], 
    #     [1, 1, 1, 1, 1, 1, 1]
    # ]
    # env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="dense", render_mode="rgb_array", max_episode_steps=100, camera_name="topview")
    env = ContinuousFrozenLakeEnv(max_steps=20)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    td3_agent = TD3(state_dim, action_dim, max_action, device=device)

    # ReplayBuffer
    replay_buffer = ReplayBuffer(buffer_size=100000) #100000
    batch_size = 32
    episodes = int(2e5)

    # 开始训练
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            # 选择动作并执行
            input_state = state
            action = td3_agent.select_action(input_state)
            next_state, reward, done, truncated, _ = env.step(action)
            input_next_state = next_state
            replay_buffer.add(input_state, action, reward, input_next_state, done)
            state = next_state
            episode_reward += reward
            # print(f'state: {state}')
            print(f"replay_buffer.size() {replay_buffer.size()}, batch_size: {batch_size:}")
            # 训练 TD3
            if replay_buffer.size() > batch_size:
                td3_agent.train(replay_buffer, batch_size)
                print('Training!!!!')

        # 记录当前 Episode 的奖励
        rewards.append(episode_reward)
        print(f"Episode: {episode}, Reward: {episode_reward}")

        # 使用 wandb 记录每个 Episode 的奖励
        wandb.log({"Episode": episode, "Reward": episode_reward})

    wandb.finish()
    
    torch.save(td3_agent.actor.state_dict(), "td3_actor_2.pth")   