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
FAST = True

if FAST:
    N_VEC = 1
    N_TRAJECTORIES = 1
    N_ITERATIONS = 1
    N_RL_TRAIN_STEPS = 100

else:
    N_VEC = 8
    N_TRAJECTORIES = 10
    N_ITERATIONS = 10
    N_RL_TRAIN_STEPS = 100_000

import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

class ContinuousFrozenLakeEnv(gym.Env):
    def __init__(self, lake_size=4, hole_radius=0.1, goal_radius=0.1, max_steps=20):
        super(ContinuousFrozenLakeEnv, self).__init__()
        
        # 定义连续状态空间
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
                                            high=np.array([lake_size, lake_size]), 
                                            dtype=np.float32)
        
        # 定义连续动作空间
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
                                       high=np.array([1.0, 1.0]), 
                                       dtype=np.float32)
        
        self.lake_size = lake_size
        self.hole_radius = hole_radius
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.current_step = 0  # 初始化步数计数器
        
        # 定义洞和目标的位置
        self.holes = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [1.0, 3.0]])
        self.goal = np.array([3.5, 3.5])
        
        self.seed_value = None
        self.reset()

    def reset(self, seed=None, **args):
        if seed is not None:
            self.seed_value = seed
            np.random.seed(self.seed_value)
        
        # 将智能体置于远离洞和目标的随机位置
        while True:
            self.state = np.random.uniform(0, self.lake_size, size=(2,))
            # self.state = np.array([1.0, 3.0])
            if not self._is_in_hole(self.state) and not self._is_in_goal(self.state):
                break
        
        self.current_step = 0  # 重置步数计数器
        return self.state, {}

    def step(self, action):
        self.current_step += 1  # Increment step counter

        # Check if the agent is in a hole
        if self._is_in_hole(self.state):
            # Agent is stuck in the hole but can still take actions within the hole's boundary
            hole_center = self._get_hole_center(self.state)
            potential_next_state = self.state + action
            
            # if distance_to_hole_center <= self.hole_radius:
            if  self._is_in_hole(potential_next_state):
                self.state = potential_next_state
                info = {"result": "1, The agent is now stuck in the hole"}
            else:
                while True:
                    random_tiny_action = np.random.uniform(-0.1, 0.1, size=self.state.shape)
                    tmp_state = self.state + random_tiny_action

                    if self._is_in_hole(tmp_state):
                        self.state = tmp_state
                        break
                info = {"result": "2, The agent is now stuck in the hole"}
            
            # info = {"result": "The agent is now stuck in the hole"}
            if self.current_step >= self.max_steps:
                info = {"result": "failure", "truncated": True}
                return self.state, -0.01, False, True, info #-0.5
            return self.state, -0.01, False, False, info # The episode doesn't end, but the agent is stuck

        else:
            # Update the state based on the action if the agent is not in a hole
            self.state = np.clip(self.state + action, 0.0, self.lake_size)

        # Check if the agent has reached the goal
        if self._is_in_goal(self.state):
            info = {"result": "success"}
            return self.state, 1.0, True, False, info
        
        # Check if the agent has exceeded the maximum number of steps
        if self.current_step >= self.max_steps:
            info = {"result": "failure", "truncated": True}
            return self.state, -0.01, False, True, info #-0.5

        # If neither, return a small negative reward to encourage reaching the goal
        return self.state, -0.01, False, False, {}
    
    def render(self, mode='human'):
        plt.figure(figsize=(6, 6))
        plt.xlim(0, self.lake_size)
        plt.ylim(0, self.lake_size)
        
        # 绘制洞
        for hole in self.holes:
            circle = plt.Circle(hole, self.hole_radius, color='blue', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # 绘制目标
        goal_circle = plt.Circle(self.goal, self.goal_radius, color='green', alpha=0.5)
        plt.gca().add_patch(goal_circle)
        
        # 绘制智能体
        agent_circle = plt.Circle(self.state, 0.05, color='red')
        plt.gca().add_patch(agent_circle)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
    
    def _is_in_hole(self, pos):
        for hole in self.holes:
            if np.linalg.norm(pos - hole) <= self.hole_radius:
                return True
        return False
    
    def _is_in_goal(self, pos):
        return np.linalg.norm(pos - self.goal) <= self.goal_radius

    def _get_hole_center(self, state):
        for hole in self.holes:
            if np.linalg.norm(state - hole) <= self.hole_radius:
                self.hole_center = hole
                return True
        return None


rng = np.random.default_rng(0)
# env = make_vec_env(
#     "seals:seals/Ant-v1",
#     rng=rng,
#     n_envs=1,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
# )
# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-Ant-v1",
#     venv=env,
# )
# rollouts = rollout.rollout(
#     expert,
#     env,
#     rollout.make_sample_until(min_timesteps=None, min_episodes=100),
#     rng=rng,
# )
env = ContinuousFrozenLakeEnv(max_steps=20)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

rollout_env = DummyVecEnv([lambda: RolloutInfoWrapper(ContinuousFrozenLakeEnv(max_steps=20)) for _ in range(1)])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
expert = model.load("/home/yuxuanli/failed_IRL_new/FL/models/FL_TD3_model.zip")

rollouts = rollout.rollout(
    expert,
    rollout_env,
    rollout.make_sample_until(min_timesteps=2000, min_episodes=200),
    rng=rng,
)

transitions = rollout.flatten_trajectories(rollouts)
print('start traning')
device = 'cuda'
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
    device=device,
)
bc_trainer.train(n_epochs=1)
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)