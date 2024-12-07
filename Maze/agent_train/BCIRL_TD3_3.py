import wandb  # 添加 wandb
import sys
import os
import copy
import pickle
import gymnasium as gym
import gymnasium_robotics
# 获取当前脚本所在路径的上一级路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from Maze.TrapMaze_action import TrapMazeEnv
from Maze.TrapMaze import TrapMazeEnv
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze import TrapMazeEnv
from TD3_3 import TD3, ReplayBuffer

success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_failed_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

def extract_obs_and_actions(demos):
    obs = []
    actions = []
    for traj in demos:
        for step in traj:
            state = step["state"]
            obs_input = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            obs.append(obs_input)
            actions.append(step["action"])
    return np.array(obs), np.array(actions)
    
expert_states, expert_actions = extract_obs_and_actions(success_demos)


def compute_maxent_reward(state, action):
    with torch.no_grad():
        reward = reward_net(state_tensor, action_tensor)
    return reward


class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出奖励值
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

if __name__ == "__main__":

    wandb.init(
        project="TrapMaze_1203",  # 替换为你的项目名称
        name='BCIRL_motor',
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

    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6))  
    td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)
    
    reward_net = RewardNetwork(state_dim, action_dim).to(device)
    reward_optimizer = optim.Adam(reward_net.parameters(), lr=3e-4)
    reward_scheduler = optim.lr_scheduler.StepLR(reward_optimizer, step_size=1000, gamma=0.9)
    reward_epochs = 5 #5

    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(7e4)
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

        state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
        action_tensor = torch.from_numpy(action).float().to(device).unsqueeze(0)
        pseudo_reward = compute_maxent_reward(state_tensor, action_tensor)
        print(f'pseudo_r: {pseudo_reward}')
        pseudo_reward = torch.clamp(pseudo_reward, min=-10, max=10)
        pseudo_reward = pseudo_reward.cpu().numpy()
        print(f'clamp_pseudo_r: {pseudo_reward}')
        replay_buffer.add(state, action, next_state, pseudo_reward, done_bool)

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

            for _ in range(reward_epochs):
                idx = np.random.choice(len(expert_states), batch_size)
                expert_states_batch = torch.FloatTensor(expert_states[idx]).to(device)
                expert_actions_batch = torch.FloatTensor(expert_actions[idx]).to(device)

                # 策略动作预测
                pred_actions = td3_agent.actor(expert_states_batch)

                # 行为克隆损失
                bc_loss = ((pred_actions - expert_actions_batch) ** 2).mean()
                reward_optimizer.zero_grad()
                bc_loss.backward()
                reward_optimizer.step()

                wandb.log({"BC Loss": bc_loss})
    
    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "td3_actor_2.pth")
    torch.save(reward_net.state_dict(), "bcirl_rewardnet_new.pth")