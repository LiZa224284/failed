import wandb  # 添加 wandb
import sys
import os
import copy
import gymnasium as gym
import gymnasium_robotics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from Maze.TrapMaze import TrapMazeEnv
from Maze.TrapMaze_action import TrapMazeEnv
from TD3_3 import TD3, ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plts



if __name__ == "__main__":

    wandb.init(
        project="TrapMaze_1203",  # 替换为你的项目名称
        name='TD3_action_My_org',
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
    # 定义奖励函数的神经网络
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

    # 加载预训练的奖励函数模型
    achieved_goal_dim = env.observation_space['achieved_goal'].shape[0]
    desired_goal_dim = env.observation_space['desired_goal'].shape[0]
    reward_input_dim = achieved_goal_dim + desired_goal_dim

    reward_net = RewardNetwork(input_dim=reward_input_dim).to(device)
    model_save_path = "/home/yuxuanli/failed_IRL_new/Maze/reward_train/reward_function_model.pth"  # 替换为你的模型路径
    reward_net.load_state_dict(torch.load(model_save_path, map_location=device))
    reward_net.eval()  # 设置为评估模式

    
    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(5e6)
    start_timesteps = 100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    org_episode_reward = 0

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
        
        next_state, org_reward, done, truncated, info = env.step(action)
        

        # 提取 achieved_goal 和 desired_goal
        achieved_goal = next_state['achieved_goal']
        desired_goal = next_state['desired_goal']
        # 准备奖励函数的输入
        reward_input = np.concatenate([achieved_goal, desired_goal])
        reward_input_tensor = torch.tensor(reward_input, dtype=torch.float32).unsqueeze(0).to(device)
        # 使用奖励函数模型计算奖励
        with torch.no_grad():
            reward = reward_net(reward_input_tensor).cpu().item()
            reward = reward + org_reward

        next_state = np.concatenate([next_state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
        done = torch.tensor(done, dtype=torch.bool)
        truncated = torch.tensor(truncated, dtype=torch.bool)
        # done_bool = torch.logical_or(done, truncated).float()
        done_bool = done

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        org_episode_reward += org_reward

        if t > start_timesteps:
            td3_agent.train()
        
        if (done or truncated):
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} org Reward: {org_episode_reward:.3f}")
            wandb.log({"Episode Reward": episode_reward})
            wandb.log({"Org Episode Reward": org_episode_reward})
            state, _ = env.reset()
            state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            done, truncated = False, False
            episode_reward = 0
            org_episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
    
    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "td3_actor_2.pth")