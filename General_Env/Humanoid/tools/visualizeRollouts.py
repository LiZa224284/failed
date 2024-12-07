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
import gymnasium as gym
import pickle
import imageio
import numpy as np
from imitation.data.types import TrajectoryWithRew

class RandomPolicyModel(BaseAlgorithm):
    def __init__(self, policy, env, verbose=0):
        super(RandomPolicyModel, self).__init__(policy=policy, env=env, verbose=verbose,  learning_rate=0.1)
    
    def _setup_model(self):
        pass

    def learn(self, total_timesteps, callback=None, log_interval=None, tb_log_name="run", reset_num_timesteps=True):
        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action = self.env.action_space.sample()
        return [action], None
    
    def save(self, path):
        pass
    
    def load(self, path):
        return self



# Load rollouts from the .pkl file
with open("/files1/Yuxuan_Li/failed_demos/Experiments/General_Env/Humanoid/rollouts.pkl", "rb") as f:
    rollouts = pickle.load(f)

env_name = "Humanoid-v4"
# Initialize the environment
env = gym.make("Humanoid-v4", render_mode="rgb_array")  # Replace "YourEnv-v3" with the environment you used for rollouts
# env = RolloutInfoWrapper(gym.make(env_name))

frames = []

# Replay each trajectory in rollouts and record as video
for trajectory in rollouts:
    observation = env.reset()

    # qpos, qvel = trajectory.obs[0][:env.model.nq], trajectory.obs[0][env.model.nq:]
    qpos = trajectory.obs[0][:env.model.nq].reshape((env.model.nq,))
    qvel = trajectory.obs[0][env.model.nq:env.model.nq + env.model.nv].reshape((env.model.nv,))
    env.set_state(qpos, qvel)  # Set the initial state with qpos and qvel
    # env.set_state(trajectory.obs[0])  # Set the initial state if applicable

    for action in trajectory.acts:
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Render and store the frame
        frame = env.render()
        frames.append(frame)

        # End the episode if done
        if terminated or truncated:
            break

env.close()

# Save frames to a video file
imageio.mimsave("/files1/Yuxuan_Li/failed_demos/Experiments/General_Env/Humanoid/rollout_video.mp4", frames, fps=30)

