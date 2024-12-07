import os
import wandb
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms import density as db
import pickle
from sklearn.neighbors import KernelDensity
from typing import Any, Dict, Iterable, List, Optional, cast
from imitation.data import types
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KernelDensity
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout
from stable_baselines3.common.base_class import BaseAlgorithm
from sklearn.kernel_ridge import KernelRidge
import imageio
from imitation.data.types import TrajectoryWithRew
from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC
import panda_gym
from sb3_contrib import TQC

class RandomPolicyModel(BaseAlgorithm):
    def __init__(self, policy, env, verbose=0):
        super(RandomPolicyModel, self).__init__(policy=policy, env=env, verbose=verbose,  learning_rate=0.1)
    
    def _setup_model(self):
        # 这里不设置模型，因为我们不需要模型来生成随机策略
        pass

    def learn(self, total_timesteps, callback=None, log_interval=None, tb_log_name="run", reset_num_timesteps=True):
        # Learn 方法不执行任何操作，因为没有实际的学习过程
        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # 随机选择一个动作
        action = self.env.action_space.sample()
        return [action], None
    
    def save(self, path):
        # 这个方法通常用于保存模型的状态，但对于随机策略来说不适用
        pass
    
    def load(self, path):
        # 从文件加载模型状态，但随机策略不需要这样做
        return self

rng = np.random.default_rng(seed=42) 
env_name = "PandaPush-v3"
env = gym.make(env_name)
# # env = CustomReacherEnv(base_env, success_threshold=0.1)
env = Monitor(env)
env = DummyVecEnv([lambda: env])
rollout_env = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_name)) for _ in range(1)])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
torch.cuda.set_device(2)
# model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device="cuda")
rollout_model = TQC("MultiInputPolicy", env, verbose=1)
expert = rollout_model.load("/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/PandaPush_TQC_model.zip", env) 
# expert = model.load("/files1/Yuxuan_Li/failed_demos/Experiments/General_Env/Reacher/assets/Reacher_TD3_model.zip")
# checkpoint = load_from_hub(
#     repo_id="jren123/sac-humanoid-v4",
#     filename="SAC-Humanoid-v4.zip",
# )
# expert = SAC.load(checkpoint)
failed_expert = RandomPolicyModel(policy=None, env=env)

rollouts = rollout.rollout(
    expert,
    rollout_env,
    rollout.make_sample_until(min_timesteps=20, min_episodes=10),
    rng=rng,
)
failed_rollouts = rollout.rollout(
    failed_expert,
    rollout_env,
    rollout.make_sample_until(min_timesteps=20, min_episodes=10),
    rng=rng,
)

with open('/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/Rollout_visualize_tools/rollouts.pkl', 'wb') as f:
    pickle.dump(rollouts, f)
with open('/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/Rollout_visualize_tools/failed_rollouts.pkl', 'wb') as f:
    pickle.dump(failed_rollouts, f)
