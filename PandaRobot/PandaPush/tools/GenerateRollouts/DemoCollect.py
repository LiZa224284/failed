import pprint
from imitation.algorithms import density as db
from imitation.data import types
from imitation.util import util
# Set FAST = False for longer training. Use True for testing and CI.
from imitation.policies.serialize import load_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.wrappers import RolloutInfoWrapper
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import TD3
import torch
import enum
import itertools
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, cast
import numpy as np
from gymnasium.spaces import utils as space_utils
from sklearn import neighbors, preprocessing
from stable_baselines3.common import base_class, vec_env
from imitation.algorithms import base
from imitation.data import rollout, types, wrappers
from imitation.rewards import reward_wrapper
from imitation.util import logger as imit_logger
from imitation.util import util
import argparse
import os
import numpy as np
import pickle
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from imitation.data import rollout
from imitation.data import types
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import density 
from imitation.algorithms.density import DensityAlgorithm
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterable, List, Optional, cast
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from stable_baselines3.common.monitor import Monitor
from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms import density as db
import pprint
import mujoco 
from sklearn.kernel_ridge import KernelRidge
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
import numpy as np
from sklearn.neighbors import KernelDensity
import joblib
from typing import Any, Dict, Iterable, List, Optional, cast
from imitation.data import types
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KernelDensity
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout
from stable_baselines3.common.base_class import BaseAlgorithm
from sklearn.kernel_ridge import KernelRidge
import joblib
import pybullet as p
from typing import Any, Dict
from panda_gym.utils import distance
import panda_gym
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from sb3_contrib import TQC
import numpy as np
import pickle
import os
from collections import namedtuple
import imageio

class CustomPushEnv(gym.Wrapper):
    def __init__(self, env, success_threshold=0.1):
        super(CustomPushEnv, self).__init__(env)
        self.success_threshold = success_threshold  # Distance threshold for success
        self.target_id = None  # Replace with actual target ID if needed
        self.end_effector_id = None  # Replace with actual end effector ID if needed

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return np.array(d < self.success_threshold, dtype=bool)

    def step(self, action):
        # Perform the step action in the environment
        obs, reward, termination, truncation, info = self.env.step(action)
        
        # Check if the task is successful
        if self.is_success(obs['achieved_goal'], obs['desired_goal']):
            info["is_success"] = True
            info["result"] = "success"
            done = True
        else:
            info["is_success"] = False
            info["result"] = "failed"
            done = False

        return obs, reward, termination, truncation, info 

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

# def collect_successful_demonstrations(model, env, num_demos=50):
#     successful_demos = []
#     num_successes = 0
    
#     while num_successes < num_demos:
#         state, _ = env.reset()
#         episode = []
#         terminated = False
#         truncated =False
        
#         while not terminated and not truncated:
#             action, _states = model.predict(state)
#             next_state, reward, terminated, truncated, info = env.step(action)
#             success = np.array([True])
            
#             # Store the transition (state, action, reward, next_state, done)
#             episode.append((state, action, reward, next_state, terminated, truncated, success))
            
#             # Move to the next state
#             state = next_state
            
#             if terminated or truncated:
#                 if info['result'] == 'success':
#                     successful_demos.append(episode)
#                     num_successes += 1
#                     print(f"Collected {num_successes}/{num_demos} successful demonstrations")
#                 else:
#                     print("in a fail demo")
                
        
#     return successful_demos
 
def collect_failed_demonstrations(model, env, num_demos=50):
    failed_demos = []
    num_failed = 0
    
    while num_failed < num_demos:
        state, _ = env.reset()
        episode = []
        terminated = False
        truncated =False
        
        while not terminated and not truncated:
            action, _states = model.predict(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            success = np.array([False])
            
            # Store the transition (state, action, reward, next_state, done)
            episode.append((state, action, reward, next_state, terminated, truncated, success))
            
            # Move to the next state
            state = next_state
            
            if terminated or truncated:
                if info['result'] == 'failed':
                    failed_demos.append(episode)
                    num_failed += 1
                    print(f"Collected {num_failed}/{num_demos} failed demonstrations")
                else:
                    print("in a successful demo")
                
        
    return successful_demos

TrajectoryWithRew = namedtuple('TrajectoryWithRew', ['obs', 'acts', 'rews', 'infos', 'terminal'])

def collect_successful_demonstrations(model, env, num_demos=50):
    successful_demos = []
    num_successes = 0

    while num_successes < num_demos:
        state, _ = env.reset()
        obs, acts, rews, infos = [], [], [], []
        terminated, truncated = False, False

        while not terminated and not truncated:
            # Collect the action from the model
            action, _states = model.predict(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store data for this step
            obs.append(state)
            acts.append(action)
            rews.append(reward)
            infos.append(info)
            
            # Move to the next state
            state = next_state

        # Check if the episode was successful
        if info.get('result') == 'success':
            trajectory = TrajectoryWithRew(
                obs=np.array(obs),
                acts=np.array(acts),
                rews=np.array(rews),
                infos=infos,
                terminal=terminated
            )
            successful_demos.append(trajectory)
            num_successes += 1
            print(f"Collected {num_successes}/{num_demos} successful demonstrations")
        else:
            print("Encountered a failed demo")

    return successful_demos

if __name__ == "__main__":
    demo_save_dir = '/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/'
    os.makedirs(demo_save_dir, exist_ok=True)

    # Create the environment and wrap it in DummyVecEnv
    env_name = 'PandaPush-v3'
    base_env = gym.make(env_name)
    env = CustomPushEnv(base_env, success_threshold=0.1)
    env = Monitor(env)

    # Load the trained model
    # successful_model = TD3.load("/home/yuxuanli/skrl_frozen_lake/checkpoints/TD3_continuous_frozenlake_stuck_WellTrained.zip", env=env)
    # failed_model = TD3.load("/home/yuxuanli/skrl_frozen_lake/checkpoints/TD3_continuous_frozenlake_stuck_BadTrained.zip", env=env)

    model = TQC("MultiInputPolicy", env, verbose=1)
    successful_model = model.load("/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/PandaPush_TQC_model.zip", env) 
    failed_model = RandomPolicyModel(policy=None, env=env)

    # Collect 50 expert demonstrations
    successful_demos = collect_successful_demonstrations(successful_model, env, num_demos=5)
    # failed_demos = collect_failed_demonstrations(failed_model, env, num_demos=2)

    # Save the demonstrations to a file for later use
    # dis_successful_demos = [pair for demo in successful_demos for pair in demo]
    # dis_failed_demos = [pair for demo in failed_demos for pair in demo]

    successful_demonstrations_path = os.path.join(demo_save_dir, "/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/successful_demonstrations_2.pkl")
    # failed_demonstrations_path = os.path.join(demo_save_dir, "/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/failed_demonstrations_2.pkl")
    # save demos in pickle file
    with open(successful_demonstrations_path, "wb") as f:
        pickle.dump(successful_demos, f)
    # with open(failed_demonstrations_path, "wb") as f:
    #     pickle.dump(failed_demos, f)

    print("Successfully collected 100 expert demonstrations.")

    env_name = "PandaPush-v3"
    env = gym.make(env_name, render_mode="rgb_array")  # Replace "YourEnv-v3" with the environment you used for rollouts

    frames = []

    for trajectory in successful_demos:
        # Reset the environment
        observation, info = env.reset()

        for action in trajectory.acts:
            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            frames.append(frame)

            if terminated or truncated:
                break

    env.close()

    # Save frames to a video file
    imageio.mimsave("/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/rollout_video.mp4", frames, fps=30)

