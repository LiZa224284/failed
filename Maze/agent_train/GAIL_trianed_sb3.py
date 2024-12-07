import sys
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze_action import TrapMazeEnv  # Ensure this import is correct

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# Load the environment
example_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 'g', 't', 0, 1],
    [1, 0, 't', 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="dense",
               render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
action_dim = env.action_space.shape[0]

# Initialize and load the trained discriminator
trained_discriminator_path = "/home/yuxuanli/failed_IRL_new/gail_discriminator.pth"
discriminator = Discriminator(state_dim, action_dim).to(device)
discriminator.load_state_dict(torch.load(trained_discriminator_path, map_location=device))
discriminator.eval()

# Define the pseudo-reward function using the trained discriminator
def compute_gail_reward(state, action):
    with torch.no_grad():
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        logits = discriminator(state, action)
        reward = -torch.log(1 - logits + 1e-8)
    return reward.cpu().numpy()

# Create a custom environment wrapper to replace rewards
# Create a custom environment wrapper to replace rewards
class GAILRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GAILRewardWrapper, self).__init__(env)
        self.current_obs = None  # 用于保存当前的观测

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_obs = obs  # 保存当前观测
        return obs

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        # 获取当前观测并处理
        obs, info = self.current_obs
        obs = np.concatenate([obs[key].flatten() for key in sorted(obs.keys())])
        obs = np.expand_dims(obs, axis=0)
        action = np.expand_dims(action, axis=0)
        # 计算 GAIL 奖励
        gail_reward = compute_gail_reward(obs, action)
        info["original_reward"] = reward
        # 更新 current_obs 为下一状态的观测
        self.current_obs = next_state
        return next_state, gail_reward, done, truncated, info
# Wrap the environment
env = GAILRewardWrapper(env)

# Noise for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize Wandb
config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": int(2.5e6),
    "batch_size": 256,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "noise_std": 0.1,
    "max_episode_steps": 100,
}

run = wandb.init(
    project="sb3_td3_TrapMaze",
    name='TD3_GAIL_TrapMaze_Using_Trained_Discriminator',
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

# Initialize the TD3 model
model = TD3(
    config["policy_type"],
    env,
    action_noise=action_noise,
    verbose=1,
    device=device,
    tensorboard_log=f"runs/{run.id}"
)

# Start training
model.learn(
    total_timesteps=config["total_timesteps"],
    log_interval=10,
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
)

# Save the final model
model.save("td3_gail_TrapMaze_with_trained_discriminator")
run.finish()