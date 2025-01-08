import wandb  
import sys
import os
import copy
import pickle
import gymnasium as gym
import gymnasium_robotics
import panda_gym
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
from MyPush import MyPandaPushEnv
from MyPush import MyPandaPushEnv_2

# success_demo_path = '/home/xlx9645/failed/PandaRobot/PandaReach/model/success_500.pkl'
success_demo_path = '/home/xlx9645/failed/PandaRobot/PandaPush/model/success_50.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)


def extract_obs_and_actions(demos, label, exp_k=0):
    obs = []
    actions = []
    rewards = []
    for traj in demos:
        demo_length = len(traj)
        for i, step in enumerate(traj):
            state = step["state"]
            obs_input = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            obs.append(obs_input)
            actions.append(step["action"])

            # exp_k = 1
            weight = (i + 1) / demo_length  # 线性增加
            weight = weight * np.exp(exp_k * weight)  # 应用指数调制

            # 成功和失败的 reward 设置
            reward = label * weight
            rewards.append(reward)

    return np.array(obs), np.array(actions), np.array(rewards)

def extract_reward(traj, label, exp_k=1):
    obs = []
    actions = []
    rewards = []
    demo_length = len(traj)

    for i, step in enumerate(traj):
        state = step["state"]
        # obs_input = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
        obs.append(state)
        # actions.append(step["action"])

        # exp_k = 1
        weight = (i + 1) / demo_length  # 线性增加
        weight = weight * np.exp(exp_k * weight)  # 应用指数调制

        # 成功和失败的 reward 设置
        reward = label * weight
        rewards.append(reward)

    return np.array(obs), np.array(rewards)

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

# 定义 BC-IRL Reward 计算函数
def compute_bcirl_reward(reward_net, state, action):
    with torch.no_grad():
        reward = reward_net(state)
    return reward

if __name__ == "__main__":

    wandb.init(
        project="Main_PandaPush",  
        name='My_start100',
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
    
    expert_states, expert_actions, success_rewards = extract_obs_and_actions(success_demos, 1, exp_k=1)
    X = expert_states
    y = success_rewards
    # 转换为 PyTorch 张量并移动到 GPU
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    reward_net = RewardNetwork(input_dim=X.shape[1]).to(device)  # 将模型移动到 GPU
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 512
    # episodes = int(5e6)
    max_timsteps = int(300e100)
    start_timesteps = 100 #int(25e3)
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
        with torch.no_grad():
            pseudo_reward = reward_net(state_tensor)
            pseudo_reward = pseudo_reward.cpu().numpy()
        replay_buffer.add(state, action, next_state, pseudo_reward, done_bool)

        '''
        traj.append([state,info])  #应该是字典 [{'state:'.., 'info':...}, {'state:'.., 'info':...}]
        其中的 state 已经是 np array 了
        '''
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

            '''
            # judge success/failed
                if info['success'] == True:
                    # assigm reward
                    success_state, sucess_rewarad = extract_reward(traj, 1)
                    x = np.concatenate(x, succcess_state)
                    y = np.concatenate(y, success_rewrad)
                else:         # failed
                    failed_state, failed_reward = extract_reward(traj, 0)
                    x = np.concatenate(x, failed_state)
                    y = np.concatenate(y, failed_rewrad)

                traj = [] # 清空 traj 以备储存新traj([state, info])  #应该是字典 [{'state:'.., 'info':...}, {'state:'.., 'info':...}]
            
            '''
            if info['is_success'] == True:
                success_states_new, success_rewards_new = extract_reward(traj, 1)
                # X = np.concatenate((X, success_states_new), axis=0)
                # y = np.concatenate((y, success_rewards_new), axis=0)
                success_states_new_tensor = torch.tensor(success_states_new, dtype=torch.float32).to(device)
                success_rewards_new_tensor = torch.tensor(success_rewards_new, dtype=torch.float32).to(device)
                X_tensor = torch.cat((X_tensor, success_states_new_tensor), dim=0)
                y_tensor = torch.cat((y_tensor, success_rewards_new_tensor), dim=0)

            elif info['is_success'] == False:
                failed_states_new, failed_rewards_new = extract_reward(traj, -1)
                # X = np.concatenate((X, failed_states_new), axis=0)
                # y = np.concatenate((y, failed_rewards_new), axis=0)
                failed_states_new_tensor = torch.tensor(failed_states_new, dtype=torch.float32).to(device)
                failed_rewards_new_tensor = torch.tensor(failed_rewards_new, dtype=torch.float32).to(device)   
                X_tensor = torch.cat((X_tensor, failed_states_new_tensor), dim=0)
                y_tensor = torch.cat((y_tensor, failed_rewards_new_tensor), dim=0)
            else:
                print('Wrong!!!!!')
            traj = []

        if (t+1) % 3000 == 1:

            epochs = 200
            for epoch in range(epochs):
                reward_net.train()
                optimizer.zero_grad()
                predictions = reward_net(X_tensor).squeeze()
                loss = loss_fn(predictions, y_tensor)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    wandb.finish()
    # torch.save(td3_agent.actor.state_dict(), "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/myit_actor.pth")
    # torch.save(reward_net.state_dict(), '/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/myit_reward.pth')