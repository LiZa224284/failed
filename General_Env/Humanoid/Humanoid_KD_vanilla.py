import pprint
from imitation.algorithms import density as db
from imitation.data import types
from imitation.util import util
# Set FAST = False for longer training. Use True for testing and CI.
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
    project = "Humanoid_1026"
    algorithm_name = 'Humanoid_KD_vanilla'
    env_name = 'seals:seals/Humanoid-v1'
    total_timesteps = 100
    n_iterations = 15000
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


    # env_name = "seals:seals/Ant-v1"
    rollout_env = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_name)) for _ in range(N_VEC)])
    env = gym.make(env_name)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    # env = util.make_vec_env(env_name, n_envs=N_VEC, rng=rng)


    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-Humanoid-v1",
        venv=rollout_env,
    )
    rollouts = rollout.rollout(
        expert,
        rollout_env,
        rollout.make_sample_until(min_timesteps=2000, min_episodes=57),
        rng=rng,
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    torch.cuda.set_device(1)
    imitation_trainer = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device="cuda")
    # imitation_trainer = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0, device="cuda" if torch.cuda.is_available() else "cpu")

    # imitation_trainer = PPO(
    #     ActorCriticPolicy, env, learning_rate=3e-4, gamma=0.95, ent_coef=1e-4, n_steps=2048
    # )
    density_trainer = db.DensityAlgorithm(
        venv=env,
        rng=rng,
        demonstrations=rollouts,
        rl_algo=imitation_trainer,
        density_type=db.DensityType.STATE_ACTION_DENSITY,
        is_stationary=True,
        kernel="gaussian",
        kernel_bandwidth=0.4,  # found using divination & some palm reading
        standardise_inputs=True,
    )
    density_trainer.train()


    expert_rewards, _ = evaluate_policy(expert, env, 100, return_episode_rewards=True)

    # evaluate the learner before training
    learner_rewards_before_training, _ = evaluate_policy(
        density_trainer.policy, env, 100, return_episode_rewards=True
    )


    # print("Starting the training!")
    for i in range(n_iterations):
        density_trainer.train_policy(total_timesteps)
        print_stats(density_trainer, 1, epoch=str(i))

    # evaluate_agent(env, imitation_trainer)
    # evaluate the learner after training
    learner_rewards_after_training, _ = evaluate_policy(
        density_trainer.policy, env, 100, return_episode_rewards=True
    )

    print("Mean expert reward:", np.mean(expert_rewards))
    print("Mean reward before training:", np.mean(learner_rewards_before_training))
    print("Mean reward after training:", np.mean(learner_rewards_after_training))