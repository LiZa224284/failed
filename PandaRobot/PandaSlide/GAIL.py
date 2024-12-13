import wandb  
import sys
import os
import copy
import pickle
import gymnasium as gym
import gymnasium_robotics
import imageio
import panda_gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from TD3 import TD3, ReplayBuffer
import argparse

success_demo_path = '/home/yuxuanli/failed_IRL_new/PandaRobot/PandaSlide/model/success_500.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)


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

def construct_obs(achieved_goal):
    observation = np.zeros(4)  # 初始化 observation
    observation[:2] = achieved_goal  # 前两维为 achieved_goal
    observation[2:] = [0, 0]  # 后两维为 action，固定为 (0, 0)

    obs_dict = {
        "observation": np.array(observation),
        "achieved_goal": np.array(achieved_goal),
        "desired_goal": np.array([0, 0]),  # 固定为 (0, 0)
    }
    obs_array = np.concatenate([obs_dict[key].flatten() for key in sorted(obs_dict.keys())])
    return np.expand_dims(obs_array, axis=0)  # 增加 batch 维度
# 可视化 BC-IRL 奖励函数
def visualize_bcirl_reward_function(reward_net_path, state_dim, action_dim, device, figure_save_path):
    # 加载训练好的 Reward Network
    reward_net = Discriminator(state_dim=8, action_dim=2).to(device)
    reward_net.load_state_dict(torch.load(reward_net_path, map_location=device))
    reward_net.eval()

    # 定义状态范围（例如 x 和 y 坐标）
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2

    # 创建网格
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)

    # 构造 achieved_goal
    achieved_goals = np.c_[xx.ravel(), yy.ravel()]  # 网格点作为 achieved_goal

    # 构造 observation
    obs_list = [construct_obs(achieved_goal) for achieved_goal in achieved_goals]
    obs_tensor = torch.tensor(np.vstack(obs_list), dtype=torch.float32).to(device)
    actions_tensor = torch.zeros((obs_tensor.shape[0], action_dim), dtype=torch.float32).to(device)
    # 计算奖励
    with torch.no_grad():
        rewards = compute_gail_reward(obs_tensor,actions_tensor)
        rewards = rewards.cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU
    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="BC-IRL Reward")
    plt.title("BC-IRL Reward Function Visualization")
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
    plt.savefig(figure_save_path)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Run kernel regression with specified parameters.")
    parser.add_argument('--update_timesteps', type=int, default=3000)
    parser.add_argument('--reward_epochs', type=int, default=200)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    wandb.init(
        project="PandaSlide",  
        name='GAIL',
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

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # example_map = [
    #     [1, 1, 1, 1, 1, 1, 1],
    #     [1, 0, 0, 0, 0, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 'g', 't', 0, 1],
    #     [1, 0, 't', 0, 0, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 1]
    # ]
    # env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="sparse", render_mode="rgb_array", max_episode_steps=300, camera_name="topview")
    env_name = 'PandaSlide-v3'
    env = gym.make(env_name)
    # 定义状态维度和动作维度
    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6),  device=device)  
    td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)
    
    discriminator = Discriminator(state_dim, action_dim).to(device)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=1000, gamma=0.9)
    bce_loss = nn.BCELoss()
    disc_epochs = 200 #5

    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(300e10) #int(2e4)
    start_timesteps = 100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    pseudo_episode_reward = 0
    episode_num = 0

    state, _ = env.reset()
    state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
    done, truncated = False, False

    success_buffer = []
    success_count = 0

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
        pseudo_reward = compute_gail_reward(state_tensor, action_tensor)
        pseudo_reward = pseudo_reward.cpu().numpy()
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

            if info['is_success'] == True:
                success_buffer.append(1)
            elif info['is_success'] == False:
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
            done, truncated = False, False
            episode_reward = 0
            pseudo_episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Train Discriminator
        # update_timesteps = 4500
        update_timesteps = args.update_timesteps
        if (t+1) % update_timesteps == 1:

            # disc_epochs = 5
            # reward_epochs = 100
            reward_epochs = args.reward_epochs
            for _ in range(reward_epochs):
                # Sample from Generator(Actor)
                gen_states, gen_actions, rewards, next_states, done_bool = replay_buffer.sample(batch_size=512)

                # Sample from Expert
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
                disc_scheduler.step()
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()
                wandb.log({"Discriminator Loss": disc_loss})

    
    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/GAIL_models/map2_gail_actor.pth")
    torch.save(discriminator.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/GAIL_models/map2_gail_discriminator.pth")