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

class CustomAIRLAlgorithm(DensityAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        """Fits the density model to demonstration data `self.transitions`."""
        # No self._scaler!
        # now fit density model
        self._density_models = {
            k: self._fit_density(v)
            for k, v in self.transitions.items()
        }
    def __call__(
        self,
        state: types.Observation,
        action: np.ndarray,
        next_state: types.Observation,
        done: np.ndarray,
        steps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Compute reward from given (s,a,s') transition batch.

        This handles *batches* of observations, since it's designed to work with
        VecEnvs.

        Args:
            state: current batch of observations.
            action: batch of actions that agent took in response to those
                observations.
            next_state: batch of observations encountered after the
                agent took those actions.
            done: is it terminal state?
            steps: What timestep is this from? Used if `self.is_stationary` is false,
                otherwise ignored.

        Returns:
            Array of scalar rewards of the form `r_t(s,a,s') = \log \hat p_t(s,a,s')`
            (one for each environment), where `\log \hat p` is the underlying density
            model (and may be independent of s', a, or t, depending on options passed
            to constructor).

        Raises:
            ValueError: Non-stationary model (`self.is_stationary` false) and `steps`
                is not provided.
        """
        if not self.is_stationary and steps is None:
            raise ValueError("steps must be provided with non-stationary models")

        del done  # TODO(adam): should we handle terminal state specially in any way?

        rew_list = []
        assert len(state) == len(action) and len(state) == len(next_state)
        state = types.maybe_wrap_in_dictobs(state)
        next_state = types.maybe_wrap_in_dictobs(next_state)
        for idx, (obs, act, next_obs) in enumerate(zip(state, action, next_state)):
            flat_trans = self._preprocess_transition(obs, act, next_obs)
            # assert self._scaler is not None
            if self._scaler is not None:
                scaled_padded_trans = self._scaler.transform(flat_trans[np.newaxis])
            else:
                scaled_padded_trans = flat_trans[np.newaxis]
            if self.is_stationary:
                rew = self._density_models[None].score(scaled_padded_trans)
            else:
                assert steps is not None
                time = steps[idx]
                if time >= len(self._density_models):
                    # Can't do anything sensible here yet. Correct solution is to use
                    # hierarchical model in which we first check whether state is
                    # absorbing, then assign either constant score or a score based on
                    # density.
                    raise ValueError(
                        f"Time {time} out of range (0, {len(self._density_models)}], "
                        "and absorbing states not currently supported",
                    )
                else:
                    time_model = self._density_models[time]
                    rew = 0.1*time_model.score(scaled_padded_trans) 
            rew_list.append(rew)
        rew_array = np.asarray(rew_list, dtype="float32")
        return rew_array

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

    def reset(self, seed=None):
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

def load_successful_demos(filepath):
    with open(filepath, 'rb') as f:
        successful_demos = pickle.load(f)
    rollouts = []

    for demo in successful_demos:
        # 将每个演示解包成不同的数据类型
        states = np.array([step[0] for step in demo])  # state
        actions = np.array([step[1] for step in demo])  # action
        rewards = np.array([step[2] for step in demo])  # reward
        next_states = np.array([step[3] for step in demo])  # next_state
        dones = np.array([step[4] for step in demo])  # done (terminated or truncated)

        rewards = rewards.reshape(-1)
        states = np.vstack([states, np.expand_dims(states[-1], axis=0)]) 

        # 创建 TrajectoryWithRew 对象
        traj = TrajectoryWithRew(
            obs=states,        # 观测值
            acts=actions,      # 动作
            rews=rewards,      # 奖励
            infos=None,        # 如果有额外的信息，传入它们；没有可以设为 None
            terminal=dones[-1] # 是否为终止状态（最后一个 done 表示这个演示是否完成）
        )
        
        rollouts.append(traj)

    return rollouts

def print_stats(density_trainer, n_trajectories, epoch=""):
    stats = density_trainer.test_policy(n_trajectories=n_trajectories)
    print("True reward function stats:")
    pprint.pprint(stats)
    stats_im = density_trainer.test_policy(
        true_reward=False,
        n_trajectories=n_trajectories,
    )
    print(f"Imitation reward function stats, epoch {epoch}:")
    pprint.pprint(stats_im)

if __name__ == "__main__":
   # Initialize wandb
    project = "FL_1019"
    algorithm_name = 'KD_vanila_bandwidth01'
    env_name = 'FL'
    total_timesteps = 100
    n_iterations = 200
    log_dir = "/home/yuxuanli/failed_IRL_new/FL/logs"
    check_interval = 10
    callback = WandbCallback(log_dir=log_dir, check_interval=check_interval)
    
    wandb.init(
        project=project,  # 同一个项目
        name=f"{algorithm_name}-{env_name}",  # 根据算法和环境生成不同的 run name
        group=algorithm_name,  # 用 group 将同一类算法归到一起
        config={"env_name": env_name, "algorithm": algorithm_name, 'total_timesteps': total_timesteps, 'log_dir': log_dir, 'check_interval': check_interval}
    )

    

    env = ContinuousFrozenLakeEnv(max_steps=20)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])

    rollouts = load_successful_demos('/files1/Yuxuan_Li/failed_demos/ImitationLearning_demos/FL/successful_demonstrations_200.pkl')

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    imitation_trainer = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0, device="cuda" if torch.cuda.is_available() else "cpu")


    rng = np.random.default_rng(seed=42)

    density_trainer = db.DensityAlgorithm(
        venv=env,
        rng=rng,
        demonstrations=rollouts,
        rl_algo=imitation_trainer,
        density_type=db.DensityType.STATE_DENSITY,
        is_stationary=True,
        kernel="gaussian",
        kernel_bandwidth=0.1,  # found using divination & some palm reading
        standardise_inputs=True,
        allow_variable_horizon= True,
        # callback = callback,
    )
    density_trainer.train()

    for i in range(n_iterations):
        density_trainer.train_policy(total_timesteps)
        print_stats(density_trainer, 1, epoch=str(i))







    def evaluate_agent(env, model, num_episodes=10):
        success_count = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action, _states = model.predict(state)
                state, reward, done, info = env.step(action)
                if done[0]:
                    if info[0].get('result') == 'success':
                        success_count += 1
        success_rate = success_count / num_episodes
        print(f"Success Rate: {success_rate * 100}% over {num_episodes} episodes")

    evaluate_agent(env, imitation_trainer)
    #没有wandb callback
    #没有model。save




    