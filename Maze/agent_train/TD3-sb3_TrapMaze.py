import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import sys
import os
import gymnasium_robotics
import wandb
from wandb.integration.sb3 import WandbCallback
# 获取当前脚本所在路径的上一级路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze_action import TrapMazeEnv
# from Maze.TrapMaze import TrapMazeEnv

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": int(5e6),
}
run = wandb.init(
    project="sb3_td3_TrapMaze",
    name = 'TD3_TrapMaze_action',
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

example_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1], 
    [1, 0, 1, 0, 1, 0, 1], 
    [1, 0, 1, 'g', 't', 0, 1], 
    [1, 0, 't', 0, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1]
]
env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="dense", render_mode="rgb_array", max_episode_steps=100, camera_name="topview")


# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1, device="cuda", tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=int(2.5e6), 
    log_interval=10,
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        )
)
model.save("td3_TrapMaze")

run.finish()