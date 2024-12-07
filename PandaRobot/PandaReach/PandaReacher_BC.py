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
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

class CustomReacherEnv(gym.Wrapper):
    def __init__(self, env, success_threshold=0.1):
        super(CustomReacherEnv, self).__init__(env)
        self.success_threshold = success_threshold  # The distance threshold for success
        self.target_id = None
        self.end_effector_id = None
        # self.link_index = end_effector_link_inde
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def step(self, action):
        # obs, reward, done, info = self.env.step(action)
        obs, reward, termination, truncation, info = self.env.step(action)
        # Check if success
        if info["is_success"] == True:
            info["result"] = "success"
            done = True
        elif info["is_success"] == False:
            info["result"] = "failed"

        return obs, reward, termination, truncation, info 


rng = np.random.default_rng(0)
env_name = 'PandaReach-v3'
base_env = gym.make(env_name)
env = CustomReacherEnv(base_env, success_threshold=0.1)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

rollout_env = DummyVecEnv([lambda: RolloutInfoWrapper(CustomReacherEnv(base_env, success_threshold=0.1)) for _ in range(1)])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
expert = model.load("//files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaReacher_TD3_model")

rollouts = rollout.rollout(
    expert,
    rollout_env,
    rollout.make_sample_until(min_timesteps=2000, min_episodes=200),
    rng=rng,
)
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