import gymnasium as gym
import panda_gym
from stable_baselines3 import TD3
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import torch
import os
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

class WandbCallback(BaseCallback):
    def __init__(self, log_dir, check_interval, verbose=1):
        super(WandbCallback, self).__init__(verbose)
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        self.save_path = os.path.join(log_dir, "best_model")
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.check_interval = check_interval  # 每隔 10 个 episode 进行一次统计
        self.episode_rewards = []  # 用于记录每个 episode 的奖励
        self.current_episode_reward = 0  # 用于累积当前 episode 的奖励
        self.episode_count = 0  # 用来统计运行了多少个 episode      
        
    def _on_step(self) -> bool:
        # 累加当前 timestep 的奖励
        self.current_episode_reward += self.locals['rewards'][0]

        # 检查是否完成了一个 episode
        if self.locals['dones']:
            # 将当前 episode 的总奖励添加到记录中
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1  # 增加 episode 计数

            # 每 10 个 episode 计算一次平均 reward
            if self.episode_count % self.check_interval == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                # wandb.log({
                #     "mean_reward_last_10_episode": mean_reward
                #     # "episodes": self.episode_count
                # }, step = self.episode_count)
                wandb.log({
                    "mean_reward_last_10_timestep": mean_reward
                }, step = self.num_timesteps)
            # 重置当前 episode 的奖励累加器
            self.current_episode_reward = 0

        return True

wandb.init(
        project="PandaPickAndPlace",  # 同一个项目
        name=f"sb3_tqc_her_sparse_2",  # 根据算法和环境生成不同的 run name
        config={'total_timesteps': int(5e6),'check_interval': 10}
    )

env_name = 'PandaPickAndPlace-v3'
env = gym.make(env_name)    

# policy_kwargs = dict(n_critics=2, n_quantiles=25)
model = TQC("MultiInputPolicy", 
            env, 
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_class=HerReplayBuffer, 
            replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
        ),
            batch_size=2048, 
            learning_rate=0.001, 
            gamma=0.95, 
            top_quantiles_to_drop_per_net=2, verbose=1, device='cuda:1')

model.learn(total_timesteps=10_000, log_interval=4)

callback = WandbCallback(log_dir='/home/yuxuanli/failed_IRL_new/PandaRobot/PandaPickAndPlace/log', check_interval=10)
model.learn(total_timesteps=int(5e6), callback=callback)