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
from TD3 import TD3, ReplayBuffer
import matplotlib.pyplot as plt
import argparse

success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_success_demos_16.pkl'
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


# def compute_bcirl_reward(state, action):
#     with torch.no_grad():
#         reward = reward_net(state_tensor, action_tensor)
#     return reward


class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
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

# 定义 BC-IRL Reward 计算函数
def compute_bcirl_reward(reward_net, state, action):
    with torch.no_grad():
        reward = reward_net(state, action)
    return reward

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
    reward_net = RewardNetwork(state_dim=8, action_dim=2).to(device)
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
        rewards = compute_bcirl_reward(reward_net, obs_tensor,actions_tensor)
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
        project="TrapMaze_1200",  # 替换为你的项目名称
        name='BCIRL_org_r',
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

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    example_map = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 'g', 't', 0, 1],
        [1, 0, 't', 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="sparse", render_mode="rgb_array", max_episode_steps=300, camera_name="topview")

    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6), device=device)  
    td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)
    
    reward_net = RewardNetwork(state_dim, action_dim).to(device)
    reward_optimizer = optim.Adam(reward_net.parameters(), lr=1e-5)
    reward_scheduler = optim.lr_scheduler.StepLR(reward_optimizer, step_size=1000, gamma=0.9)
    reward_epochs = 200 #5

    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(300e10) #int(6e4)
    start_timesteps = 0 #100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    pseudo_episode_reward = 0
    episode_num = 0

    state, _ = env.reset()
    state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
    done, truncated = False, False

    success_buffer = []

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
        pseudo_reward = compute_bcirl_reward(reward_net, state_tensor, action_tensor)
        print(f'pseudo_r: {pseudo_reward}')
        pseudo_reward = torch.clamp(pseudo_reward, min=-10, max=10)
        pseudo_reward = pseudo_reward.cpu().numpy()
        print(f'clamp_pseudo_r: {pseudo_reward}')
        pseudo_reward += reward
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
            done, truncated = False, False
            episode_reward = 0
            pseudo_episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        update_timesteps = 1
        # update_timesteps = args.update_timesteps
        if (t+1) % update_timesteps == 0:

            # reward_epochs = 100
            reward_epochs = args.reward_epochs
            for _ in range(reward_epochs):
                idx = np.random.choice(len(expert_states), batch_size)
                expert_states_batch = torch.FloatTensor(expert_states[idx]).to(device)
                expert_actions_batch = torch.FloatTensor(expert_actions[idx]).to(device)

            
                pred_actions = td3_agent.actor(expert_states_batch)

                
                bc_loss = ((pred_actions - expert_actions_batch) ** 2).mean()
                reward_optimizer.zero_grad()
                bc_loss.backward()
                reward_optimizer.step()

                wandb.log({"Discriminator Loss": bc_loss})
            
        if (t+1) % 1500 == 0:
            save_path = f'/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/BCIRL_models/mid_16/mid_reward_{t+1}.pth'
            torch.save(reward_net.state_dict(), save_path)

            fig_save_path = f"/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/BCIRL_models/mid_16/bcirl_map2_rewardnet_{t+1}.png"
            visualize_bcirl_reward_function(
                reward_net_path=save_path,
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                figure_save_path=fig_save_path
            )

    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/BCIRL_models/bcirl_actor.pth")
    torch.save(reward_net.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/BCIRL_models/bcirl_rewardnet.pth")