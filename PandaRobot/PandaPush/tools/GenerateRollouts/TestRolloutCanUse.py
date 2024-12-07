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
from collections import namedtuple

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


TrajectoryWithRew = namedtuple('TrajectoryWithRew', ['obs', 'acts', 'rews', 'infos', 'terminal'])

# Load rollouts from the .pkl file
with open("/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/ManualControl_SuccessfulDemos.pkl", "rb") as f:
    rollouts = pickle.load(f)

env_name = "PandaPush-v3"
env = gym.make(env_name, render_mode="rgb_array")  # Replace "YourEnv-v3" with the environment you used for rollouts

frames = []

for trajectory in rollouts:
    # Reset the environment
    # observation, info = env.reset()
    # env.reset()
    # env.env.sim.set_state_from_flattened(trajectory.obs[0])  # 将初始状态设置为 trajectory 的第一个观测
    # env.env.sim.forward() 
    # 将初始状态设置为 trajectory 的第一个观测
    
    env.reset()

    # 将初始状态设置为 trajectory 的第一个观测
    initial_observation = trajectory.obs[0]

    # 假设 obs[0] 的前几个元素是机器人关节的状态，之后是物体的位置
    # 根据实际情况分解 initial_observation 为关节状态和物体状态
    robot_joint_positions = initial_observation[:7]  # 假设机器人有 7 个关节
    object_position = initial_observation[7:10]      # 物体位置 (x, y, z)
    object_orientation = initial_observation[10:14]  # 物体的四元数旋转 (x, y, z, w)

    # 设置 Panda 机器人的关节位置
    for joint_index, joint_position in enumerate(robot_joint_positions):
        env.env._p.resetJointState(env.env.robot_id, joint_index, joint_position)

    # 设置物体的位置和旋转
    object_id = env.env.block_id  # 假设物体的 ID 为 block_id，具体取决于您的环境定义
    env.env._p.resetBasePositionAndOrientation(object_id, object_position, object_orientation)

    # 在仿真器中推进一步以应用新的初始状态
    env.env._p.stepSimulation()

    for action in trajectory.acts:
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            break

env.close()

# Save frames to a video file
imageio.mimsave("/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/rollout_video.mp4", frames, fps=10)

