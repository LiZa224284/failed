from functools import partial

from seals import base_envs
from seals.diagnostics.cliff_world import CliffWorldEnv
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
)
from imitation.data import rollout
from imitation.rewards import reward_nets

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

class ContinuousFrozenLakeEnv(gym.Env):
    def __init__(self, lake_size=4, hole_radius=0.1, goal_radius=0.1, horizon=20):
        super(ContinuousFrozenLakeEnv, self).__init__()
        
        # 定义连续状态空间
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
                                            high=np.array([lake_size, lake_size]), 
                                            dtype=np.float32)
        
        # 定义连续动作空间
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
                                       high=np.array([1.0, 1.0]), 
                                       dtype=np.float32)
        self.horizon = horizon
        self.lake_size = lake_size
        self.hole_radius = hole_radius
        self.goal_radius = goal_radius
        self.max_steps = horizon
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


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    env = ContinuousFrozenLakeEnv()
    env = Monitor(env)
    env_single = env
    env = DummyVecEnv([lambda: env])
    
    # env_creator = partial(env, height=4, horizon=8, width=7, use_xy_obs=True)
    env_creator = lambda: ContinuousFrozenLakeEnv()
    env_creator = partial(env_creator)
    env_single = env_creator()

    state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_creator())

    # This is just a vectorized environment because `generate_trajectories` expects one
    # state_venv = DummyVecEnv([state_env_creator] * 4)
    state_venv = env

    _, _, pi = mce_partition_fh(env_single)

    _, om = mce_occupancy_measures(env_single, pi=pi)

    reward_net = reward_nets.BasicRewardNet(
        env_single.observation_space,
        env_single.action_space,
        hid_sizes=[256],
        use_action=False,
        use_done=False,
        use_next_state=False,
    )

    # training on analytically computed occupancy measures
    mce_irl = MCEIRL(
        om,
        env_single,
        reward_net,
        log_interval=250,
        optimizer_kwargs={"lr": 0.01},
        rng=rng,
    )
    occ_measure = mce_irl.train()

    imitation_trajs = rollout.generate_trajectories(
        policy=mce_irl.policy,
        venv=state_venv,
        sample_until=rollout.make_min_timesteps(5000),
        rng=rng,
    )
    print("Imitation stats: ", rollout.rollout_stats(imitation_trajs))