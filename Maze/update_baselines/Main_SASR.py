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
import matplotlib.pyplot as plt
from TD3 import TD3, ReplayBuffer
from scipy.stats import beta
from sklearn.neighbors import KernelDensity
import joblib

success_demo_path = '/home/xlx9645/failed/Maze/demo_generate/demos/action_trapMaze/all_success_demos_16.pkl'
failed_demo_path = '/home/xlx9645/failed/Maze/demo_generate/demos/action_trapMaze/all_failed_demos_12.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

class SASRRewardShaper:
    def __init__(self, input_dim, rff_dim=128, retention_rate=0.8, bandwidth=0.1):
        # RFF 参数
        self.rff_dim = rff_dim
        self.retention_rate = retention_rate
        self.bandwidth = bandwidth

        # 随机 Fourier 参数
        self.W = np.random.normal(size=(input_dim, rff_dim))
        self.b = np.random.uniform(0, 2 * np.pi, size=rff_dim)

        # 根据 Retention Rate 保留部分特征
        selected_indices = np.random.choice(rff_dim, size=int(rff_dim * retention_rate), replace=False)
        self.reduced_W = self.W[:, selected_indices]
        self.reduced_b = self.b[selected_indices]

        # KDE 初始化
        self.success_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self.failed_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)

        # 数据缓冲
        self.success_data = []
        self.failed_data = []

    def rff_transform(self, x):
        """将输入状态映射到 RFF 空间"""
        return np.sqrt(2 / self.reduced_W.shape[1]) * np.cos(np.dot(x, self.reduced_W) + self.reduced_b)

    def update_kde(self, success_states, failed_states):
        """更新 KDE 模型"""
        if len(success_states) > 0:
            transformed_success = np.array([self.rff_transform(s) for s in success_states])
            self.success_data.extend(transformed_success)
            self.success_kde.fit(np.array(self.success_data))

        if len(failed_states) > 0:
            transformed_failed = np.array([self.rff_transform(s) for s in failed_states])
            self.failed_data.extend(transformed_failed)
            self.failed_kde.fit(np.array(self.failed_data))

    def compute_success_rate(self, state):
        """计算成功率"""
        transformed_state = self.rff_transform(state)
        success_prob = (
            np.exp(self.success_kde.score_samples([transformed_state])) if len(self.success_data) > 0 else 0.0
        )
        failed_prob = (
            np.exp(self.failed_kde.score_samples([transformed_state])) if len(self.failed_data) > 0 else 0.0
        )
        alpha = success_prob + 1
        beta_val = failed_prob + 1
        return beta(alpha, beta_val).mean()

    def compute_shaped_reward(self, state, base_reward):
        """计算塑形奖励"""
        success_rate = self.compute_success_rate(state)
        return 0.4*base_reward + 0.6*success_rate
    
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

def construct_obs(achieved_goal):
    # 假设其余 6 个维度都为 0，只演示把 2D 的 (x,y) 放到最后
    obs = np.zeros(8, dtype=np.float32)
    obs[3:5] = achieved_goal  # 或者你自己的规则
    obs[5:6] = achieved_goal 
    return obs

def visualize_sasr_reward_function(sasr_shaper, figure_save_path):
    # 1. 定义你想要可视化的状态范围（例如 x 和 y 坐标）
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.0, 2.0

    # 2. 创建网格
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)

    # 3. 将网格上的点当作 "achieved_goal"（或者你环境中的 2D 部分）
    achieved_goals = np.c_[xx.ravel(), yy.ravel()]

    # 4. 构造观测 obs_list，并计算塑形奖励
    shaped_rewards = []
    for ag in achieved_goals:
        obs = construct_obs(ag)
        # 这里 base_reward=0.0，方便看“纯粹的塑形值”；可自行修改
        obs = obs.squeeze(axis=0)
        sr = sasr_shaper.compute_shaped_reward(obs, base_reward=0.0)
        shaped_rewards.append(sr)
    shaped_rewards = np.array(shaped_rewards).reshape(xx.shape)

    # 5. 绘制奖励函数的等高线图
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, shaped_rewards, levels=50, cmap="viridis")
    plt.colorbar(label="SASR Shaped Reward")
    plt.title("SASR Reward Function Visualization")
    plt.xlabel("State Dimension 1 (x)")
    plt.ylabel("State Dimension 2 (y)")
    plt.savefig(figure_save_path)
    plt.close()

# 定义 BC-IRL Reward 计算函数
def compute_bcirl_reward(reward_net, state, action):
    with torch.no_grad():
        reward = reward_net(state)
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
    reward_net = RewardNetwork(input_dim=8).to(device)
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

if __name__ == "__main__":

    wandb.init(
        project="Main_1229",  
        name='SASR5',
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

    example_map = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 'g', 't', 0, 1],
        [1, 0, 't', 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="sparse", render_mode="rgb_array", max_episode_steps=300, camera_name="topview")

    # 定义状态维度和动作维度
    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6),  device=device)  
    td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)
    
    success_demo_buffer= extract_obs_and_actions(success_demos)[0].tolist() 
    failed_demo_buffer = extract_obs_and_actions(failed_demos)[0].tolist() 


    batch_size = 512
    max_timsteps = int(300e100)
    start_timesteps = 0 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    pseudo_episode_reward = 0
    episode_num = 0
    traj = [] #应该是字典 [{'state:'.., 'info':...}, {'state:'.., 'info':...}]

    state, _ = env.reset()
    state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
    done, truncated = False, False


    success_buffer = []
    success_count = 0
    episode_rewards = []  # 用于存储最近的 episode 奖励
    avg_episode_reward_window = 50

    sasr_shaper = SASRRewardShaper(bandwidth=0.1, input_dim=state_dim, rff_dim=128, retention_rate=0.1)
 
    for t in range(max_timsteps):
        episode_timesteps += 1

        for i in range(300):
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
            pseudo_reward = sasr_shaper.compute_shaped_reward(state, reward)
            replay_buffer.add(state, action, next_state, pseudo_reward, done_bool)

            traj.append({"state": state, "info": info})

            state = next_state
            episode_reward += reward
            pseudo_episode_reward += pseudo_reward

        if t > start_timesteps:
            td3_agent.train()
        
        

        if (done or truncated):
            episode_rewards.append(episode_reward)
            if len(episode_rewards) > avg_episode_reward_window:
                episode_rewards.pop(0)
            avg_episode_reward = np.mean(episode_rewards)
            wandb.log({"average Episode Reward": avg_episode_reward})

            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} PseudoReward: {pseudo_episode_reward}")
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

            if info['success'] == True:
                success_demo_buffer.append(state)

            elif info['success'] == False:
                failed_demo_buffer.append(state)
            else:
                print('Wrong!!!!!')
            traj = []

        if (t+1) % 15000 == 1:
        # if (t+1) % 10 == 1:
            sasr_shaper.update_kde(success_demo_buffer, failed_demo_buffer)

        if (t+1) % int(50e3) == 0:
        # if (t+1) % 10 == 1:
            save_path = f'/home/xlx9645/failed/Maze/update_baselines/models/SASR/mid_16/sasr_shaper_{t+1}_kde_models.joblib'
            # save kde model
            joblib.dump(sasr_shaper, save_path)

            fig_save_path = f"/home/xlx9645/failed/Maze/update_baselines/models/SASR/mid_16/sasr_shaper_{t+1}.png"
            visualize_sasr_reward_function(sasr_shaper, fig_save_path)

    wandb.finish()
    torch.save(td3_agent.actor.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/myit_actor.pth")
    # torch.save(reward_net.state_dict(), '/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/myit_reward.pth')