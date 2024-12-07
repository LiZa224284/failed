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

class CustomAIRLAlgorithm(DensityAlgorithm):
    def __init__(self, *args, failed_demonstrations: Optional[base.AnyTransitions] = None, exp_k, **kwargs):
        self.exp_k = exp_k
        super().__init__(*args, **kwargs)
        self.failed_demonstrations = failed_demonstrations

        if failed_demonstrations is not None:
            self.set_failed_demonstrations(failed_demonstrations)
        
        

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        """Sets the demonstration data."""
        transitions: Dict[Optional[int], List[np.ndarray]] = {}

        if isinstance(demonstrations, types.TransitionsMinimal):
            next_obs_b = getattr(demonstrations, "next_obs", None)
            transitions.update(
                self._get_demo_from_batch(
                    demonstrations.obs,
                    demonstrations.acts,
                    next_obs_b,
                ),
            )
        elif isinstance(demonstrations, Iterable):
            # Inferring the correct type here is difficult with generics.
            (
                first_item,
                demonstrations,
            ) = util.get_first_iter_element(  # type: ignore[assignment]
                demonstrations,
            )
            if isinstance(first_item, types.Trajectory):
                # we assume that all elements are also types.Trajectory.
                # (this means we have timestamp information)
                # It's not perfectly type safe, but it allows for the flexibility of
                # using iterables, which is useful for large data structures.
                demonstrations = cast(Iterable[types.Trajectory], demonstrations)
                states = []
                weights = []

                for traj in demonstrations:
                    demo_length = len(traj)
                    for i, (obs, act, next_obs) in enumerate(
                        zip(traj.obs[:-1], traj.acts, traj.obs[1:]),
                    ):
                        flat_trans = self._preprocess_transition(obs, act, next_obs)
                        weight = (i + 1) / demo_length
                        weight = weight * np.exp(self.exp_k * weight)
                        states.append(flat_trans)
                        weights.append(weight)
                states = np.array(states)
                weights = np.array(weights)
                transitions = {0:np.array(states)}
                        
            elif isinstance(first_item, Mapping):
                # analogous to cast above.
                demonstrations = cast(Iterable[types.TransitionMapping], demonstrations)

                def to_np_maybe_dictobs(x):
                    if isinstance(x, types.DictObs):
                        return x
                    else:
                        return util.safe_to_numpy(x, warn=True)

                for batch in demonstrations:
                    obs = to_np_maybe_dictobs(batch["obs"])
                    acts = util.safe_to_numpy(batch["acts"], warn=True)
                    next_obs = to_np_maybe_dictobs(batch.get("next_obs"))
                    transitions.update(self._get_demo_from_batch(obs, acts, next_obs))
            else:
                raise TypeError(
                    f"Unsupported demonstration type {type(demonstrations)}",
                )
        else:
            raise TypeError(f"Unsupported demonstration type {type(demonstrations)}")

        self.transitions = transitions
        self.weights = weights

        if not self.is_stationary and None in self.transitions:
            raise ValueError(
                "Non-stationary model incompatible with non-trajectory demonstrations.",
            )
        if self.is_stationary:
            self.transitions = {
                None: np.concatenate(list(self.transitions.values()), axis=0),
            }

    def set_failed_demonstrations(self, failed_demonstrations: base.AnyTransitions) -> None:
        """Sets the demonstration data."""
        transitions: Dict[Optional[int], List[np.ndarray]] = {}

        if isinstance(failed_demonstrations, types.TransitionsMinimal):
            next_obs_b = getattr(failed_demonstrations, "next_obs", None)
            transitions.update(
                self._get_demo_from_batch(
                    failed_demonstrations.obs,
                    failed_demonstrations.acts,
                    next_obs_b,
                ),
            )
        elif isinstance(failed_demonstrations, Iterable):
            # Inferring the correct type here is difficult with generics.
            (
                first_item,
                failed_demonstrations,
            ) = util.get_first_iter_element(  # type: ignore[assignment]
                failed_demonstrations,
            )
            if isinstance(first_item, types.Trajectory):
                # we assume that all elements are also types.Trajectory.
                # (this means we have timestamp information)
                # It's not perfectly type safe, but it allows for the flexibility of
                # using iterables, which is useful for large data structures.
                demonstrations = cast(Iterable[types.Trajectory], failed_demonstrations)
                states = []
                weights = []

                for traj in demonstrations:
                    demo_length = len(traj)
                    for i, (obs, act, next_obs) in enumerate(
                        zip(traj.obs[:-1], traj.acts, traj.obs[1:]),
                    ):
                        flat_trans = self._preprocess_transition(obs, act, next_obs)
                        weight = (i + 1) / demo_length
                        weight = weight * np.exp(self.exp_k * weight)
                        states.append(flat_trans)
                        weights.append(weight)
                states = np.array(states)
                weights = np.array(weights)
                transitions = {0:np.array(states)}
                        
            elif isinstance(first_item, Mapping):
                # analogous to cast above.
                demonstrations = cast(Iterable[types.TransitionMapping], demonstrations)

                def to_np_maybe_dictobs(x):
                    if isinstance(x, types.DictObs):
                        return x
                    else:
                        return util.safe_to_numpy(x, warn=True)

                for batch in demonstrations:
                    obs = to_np_maybe_dictobs(batch["obs"])
                    acts = util.safe_to_numpy(batch["acts"], warn=True)
                    next_obs = to_np_maybe_dictobs(batch.get("next_obs"))
                    transitions.update(self._get_demo_from_batch(obs, acts, next_obs))
            else:
                raise TypeError(
                    f"Unsupported demonstration type {type(demonstrations)}",
                )
        else:
            raise TypeError(f"Unsupported demonstration type {type(demonstrations)}")

        self.failed_transitions = transitions
        self.failed_weights = weights

        if not self.is_stationary and None in self.transitions:
            raise ValueError(
                "Non-stationary model incompatible with non-trajectory demonstrations.",
            )
        if self.is_stationary:
            self.transitions = {
                None: np.concatenate(list(self.transitions.values()), axis=0),
            }

    def train(self) -> None:
        """Fits the kernel regression model to demonstration data `self.transitions`."""
        self._regression_models = {
            k: self._fit_regression(v, self.weights)
            for k, v in self.transitions.items()
        }
        joblib.dump(self._regression_models, "SFD_success_kr_model")

        self._failed_regression_models = {
            k: self._fit_regression(v, self.failed_weights)
            for k, v in self.failed_transitions.items()
        }
        joblib.dump(self._failed_regression_models, "SFD_failed_kr_model")

    def _fit_regression(self, transitions: np.ndarray, weights: np.array) -> KernelRidge:
        regression_model = KernelRidge(
            kernel=self.kernel,  # 可以选择不同的核函数
            alpha=self.kernel_bandwidth  # 使用带宽作为正则化参数
        )
        regression_model.fit(transitions, weights)  # 使用 transitions 作为 X，weights 作为 y
        return regression_model
    
    def __call__(
        self,
        state: types.Observation,
        action: np.ndarray,
        next_state: types.Observation,
        done: np.ndarray,
        steps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Compute reward from given (s,a,s') transition batch.
        """
        if not self.is_stationary and steps is None:
            raise ValueError("steps must be provided with non-stationary models")

        del done  # TODO(adam): should we handle terminal state specially in any way?

        rew_list = []
        # assert len(state) == len(action) and len(state) == len(next_state)
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
                # rew = self._regression_models[None].predict(scaled_padded_trans)
                # rew = self._regression_models[None].predict(scaled_padded_trans)*100
                rew = (self._regression_models[None].predict(scaled_padded_trans) - self._failed_regression_models[0].predict(scaled_padded_trans)) - 100
            else:
                assert steps is not None
                time = steps[idx]
                if time >= len(self._regression_models):
                    # Can't do anything sensible here yet. Correct solution is to use
                    # hierarchical model in which we first check whether state is
                    # absorbing, then assign either constant score or a score based on
                    # density.
                    raise ValueError(
                        f"Time {time} out of range (0, {len(self._regression_models)}], "
                        "and absorbing states not currently supported",
                    )
                else:
                    time_model = self._regression_models[time]
                    time_failed_model = self._failed_regression_models[time]
                    # rew = 0.5*time_model.predict(scaled_padded_trans) - 2
                    rew = (self._regression_models[None].predict(scaled_padded_trans) - self._failed_regression_models[0].predict(scaled_padded_trans)) - 100
            rew_list.append(rew)
        rew_array = np.asarray(rew_list, dtype="float32")
        rew_array = rew_array.reshape(1,)
        return rew_array

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run kernel regression with specified parameters.")
    parser.add_argument('--n_iterations', type=int, default=600, help='Number of training iterations.')
    parser.add_argument('--exp_k', type=float, default=2.0, help='Exponent factor for weighting in kernel regression.')
    parser.add_argument('--kernel_bandwidth', type=float, default=0.1, help='Bandwidth for the kernel in kernel regression.')
    parser.add_argument('--n_s_demos', type=int, default=500, help='Number of training iterations.')
    parser.add_argument('--n_f_demos', type=int, default=200, help='Number of training iterations.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    project = "PandaReacher_1105"
    # algorithm_name = 'KR_SFD_exp1_bw01'
    algorithm_name = f"KR_SFD_exp{args.exp_k}_bw{args.kernel_bandwidth}_sdmeo{args.n_s_demos}__fdmeo{args.n_f_demos}"
    env_name = 'PandaReach-v3'
    exp_k = args.exp_k
    kernel_bandwidth = args.kernel_bandwidth
    n_s_demos = args.n_s_demos
    n_f_demos = args.n_f_demos
    total_timesteps = 100
    n_iterations = args.n_iterations
    log_dir = "/files1/Yuxuan_Li/failed_demos/Experiments/General_Env/logs/PPO_KD_logs"
    check_interval = 10
    callback = WandbCallback(log_dir=log_dir, check_interval=check_interval)

    wandb.init(
        project=project,  # 同一个项目
        name=f"{algorithm_name}-{env_name}",  # 根据算法和环境生成不同的 run name
        group=algorithm_name,  # 用 group 将同一类算法归到一起
        config={"env_name": env_name, "algorithm": algorithm_name, 'total_timesteps': total_timesteps, 'log_dir': log_dir, 'check_interval': check_interval}
    )
    rng = np.random.default_rng(seed=42)

    base_env = gym.make(env_name)
    env = CustomReacherEnv(base_env, success_threshold=0.1)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])

    rollout_env = DummyVecEnv([lambda: RolloutInfoWrapper(CustomReacherEnv(base_env, success_threshold=0.1)) for _ in range(1)])

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    torch.cuda.set_device(0)
    model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1, device='cuda')
    expert = model.load("/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaReacher_TD3_model")
    failed_expert = RandomPolicyModel(policy=None, env=env)

    rollouts = rollout.rollout(
        expert,
        rollout_env,
        rollout.make_sample_until(min_timesteps=10, min_episodes=n_s_demos),
        rng=rng,
    )
    failed_rollouts = rollout.rollout(
        failed_expert,
        rollout_env,
        rollout.make_sample_until(min_timesteps=10, min_episodes=n_f_demos),
        rng=rng,
    )

    density_trainer = CustomAIRLAlgorithm(
        venv=env,
        rng=rng,
        demonstrations=rollouts,
        failed_demonstrations=failed_rollouts,
        rl_algo=model,
        density_type=db.DensityType.STATE_ACTION_DENSITY,
        is_stationary=True,
        kernel="rbf",
        kernel_bandwidth=kernel_bandwidth,  # found using divination & some palm reading
        exp_k = exp_k,
        standardise_inputs=True,
        allow_variable_horizon= True,
    )
    density_trainer.train()

    # print("Starting the training!")
    for i in range(n_iterations):
        density_trainer.train_policy(total_timesteps)
        print_stats(density_trainer, 1, epoch=str(i))

