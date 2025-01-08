import wandb  
import sys
import os
import copy
import pickle
import gymnasium as gym
import gymnasium_robotics
import panda_gym
import imageio
import argparse
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
from MyPush import MyPandaPushEnv
from MyPush import MyPandaPushEnv_2

success_demo_path = '/home/xlx9645/failed/PandaRobot/PandaPush/model/success_50.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

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
    
expert_states, expert_actions = extract_obs_and_actions(success_demos)

def parse_args():
    parser = argparse.ArgumentParser(description="Run kernel regression with specified parameters.")
    parser.add_argument('--update_timesteps', type=int, default=3000)
    parser.add_argument('--reward_epochs', type=int, default=200)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    wandb.init(
        project="Main_PandaPush",  
        name='SASR',
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

    env_name = 'MyPandaPushEnv_2'
    env = gym.make(env_name)
    # 定义状态维度和动作维度
    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6),  device=device)  
    td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)
    
    success_demo_buffer= extract_obs_and_actions(success_demos)[0].tolist() 
    failed_demo_buffer = []

    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(300e100)
    start_timesteps = 100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    pseudo_episode_reward = 0
    episode_num = 0
    traj = []  

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

            if info['is_success'] == True:
                success_demo_buffer.append(state)

            elif info['is_success'] == False:
                failed_demo_buffer.append(state)
            else:
                print('Wrong!!!!!')
            traj = []

        if (t+1) % 15000 == 1:
        # if (t+1) % 1 == 1:
            sasr_shaper.update_kde(success_demo_buffer, failed_demo_buffer)

        
    wandb.finish()
    # torch.save(td3_agent.actor.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/myit_actor.pth")
    # torch.save(reward_net.state_dict(), '/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/myit_reward.pth')