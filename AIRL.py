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


def compute_airl_reward(state, action):
    with torch.no_grad():
        logits = discriminator(state, action)
        sigmoid_logits = torch.sigmoid(logits)  # 转换为概率 D(s, a)
        reward = torch.log(sigmoid_logits + 1e-8) - torch.log(1 - sigmoid_logits + 1e-8)

        # reward = torch.log(1 + torch.exp(-logits)) 

        # reward = logits - torch.log(torch.exp(logits) + 1)
        # reward = -torch.log(1 - logits + 1e-8)
    return reward


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            # nn.Sigmoid()  # 输出为 [0, 1] 概率
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.net(x)

if __name__ == "__main__":

    wandb.init(
        project="TrapMaze_1205",  # 替换为你的项目名称
        name='AIRL_sparse',
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

    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6))  
    td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)
    
    discriminator = Discriminator(state_dim, action_dim).to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=1000, gamma=0.9)
    disc_epochs = 5

    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(6e4)
    start_timesteps = 100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    pseudo_episode_reward = 0
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
        pseudo_reward = compute_airl_reward(state_tensor, action_tensor)
        print(f'pseudo_r: {pseudo_reward}')
        pseudo_reward = torch.clamp(pseudo_reward, min=-10, max=10)
        pseudo_reward = pseudo_reward.cpu().numpy()
        print(f'clamp_pseudo_r: {pseudo_reward}')
        replay_buffer.add(state, action, next_state, pseudo_reward, done_bool)

        state = next_state
        episode_reward += reward
        pseudo_episode_reward += pseudo_reward

        if t > start_timesteps:
            td3_agent.train()
        
        if (done or truncated):
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            wandb.log({"Episode Reward": episode_reward})
            wandb.log({"Pseudo Episode Reward": pseudo_episode_reward})
            state, _ = env.reset()
            state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            done, truncated = False, False
            episode_reward = 0
            pseudo_episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

            # Train Discriminator
            for _ in range(disc_epochs):
                # Sample from Generator(Actor)
                gen_states, gen_actions, rewards, next_states, done_bool = replay_buffer.sample(batch_size=512)

                # Sample from Expert
                idx = np.random.choice(len(expert_states), batch_size)
                expert_states_batch = torch.FloatTensor(expert_states[idx]).to(device)
                expert_actions_batch = torch.FloatTensor(expert_actions[idx]).to(device)

                gen_rewards = discriminator(gen_states, gen_actions)  # AIRL: 判别器直接输出奖励
                expert_rewards = discriminator(expert_states_batch, expert_actions_batch)
                reg_loss = 0.01 * torch.mean(gen_rewards**2 + expert_rewards**2)
                disc_loss = torch.mean(torch.exp(-expert_rewards)) + torch.mean(gen_rewards) + reg_loss

                # 优化判别器
                disc_scheduler.step()
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()
                wandb.log({"Discriminator Loss": disc_loss})

    
    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/AIRL_models/airl_actor.pth")
    torch.save(discriminator.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/AIRL_models/airl_discriminator.pth")