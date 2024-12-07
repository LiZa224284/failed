import gymnasium as gym
import panda_gym
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
import gymnasium as gym

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

class CustomAIRLAlgorithm(DensityAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def train(self) -> None:
        """Fits the density model to demonstration data `self.transitions`."""
        # if requested, we'll scale demonstration transitions so that they have
        # zero mean and unit variance (i.e. all components are equally important)
        # self._scaler = preprocessing.StandardScaler(
        #     with_mean=self.standardise,
        #     with_std=self.standardise,
        # )
        # flattened_dataset = np.concatenate(list(self.transitions.values()[0]), axis=0)
        # self._scaler.fit(flattened_dataset)

        # now fit density model
        self._density_models = {
            k: self._fit_density(v, self.weights)
            for k, v in self.transitions.items()
        }

    def _fit_density(self, transitions: np.ndarray, weights:np.array) -> neighbors.KernelDensity:


        density_model = neighbors.KernelDensity(
            kernel=self.kernel,
            bandwidth=self.kernel_bandwidth,  # 使用自定义的 bandwidth
        )
        density_model.fit(transitions, weights)  #[0]:states, [1]:weiights
        return density_model
    
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
                    rew = 0.5*time_model.score(scaled_padded_trans) - 2
            rew_list.append(rew)
        rew_array = np.asarray(rew_list, dtype="float32")
        return rew_array


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

if __name__ == "__main__":
    project = "PandaReacher_1025"
    algorithm_name = 'PandaReacher_Td3'
    env_name = 'PandaReach-v3'
    total_timesteps = int(8e4)#10000
    n_iterations = 200
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


    # env_name = "Pendulum-v1"
    rollout_env = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_name)) for _ in range(N_VEC)])
    env = gym.make(env_name)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    # env = util.make_vec_env(env_name, n_envs=N_VEC, rng=rng)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3('MultiInputPolicy', env, action_noise=action_noise, verbose=0, device="cuda" if torch.cuda.is_available() else "cpu")
    print("Model is using device:", model.device)

    learner_rewards_before_training, _ = evaluate_policy(
        model, env, 100, return_episode_rewards=True
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaReacher_TD3_model.zip")



    # expert_rewards, _ = evaluate_policy(expert, env, 100, return_episode_rewards=True)

    # evaluate the learner before training
    
    learner_rewards_after_training, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)


    # print("Mean expert reward:", np.mean(expert_rewards))
    print("Mean reward before training:", np.mean(learner_rewards_before_training))
    print("Mean reward after training:", np.mean(learner_rewards_after_training))