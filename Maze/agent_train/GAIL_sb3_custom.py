import sys
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_util import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze import TrapMazeEnv  # Make sure this import is correct

# Load expert demonstrations
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

# Extract states and actions from demos
def extract_obs_and_actions(demos):
    obs = []
    actions = []
    for traj in demos:
        for step in traj:
            obs_input = step["state"]
            obs_input = np.concatenate([obs_input[key].flatten() for key in sorted(obs_input.keys())])
            obs.append(obs_input)
            actions.append(step["action"])
    return np.array(obs), np.array(actions)

expert_states, expert_actions = extract_obs_and_actions(success_demos)

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

# Initialize the environment
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
# state_dim = env.observation_space["observation"].shape[0]
state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# Initialize the discriminator
discriminator = Discriminator(state_dim, action_dim).to(device)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
bce_loss = nn.BCELoss()

# Define the pseudo-reward function
def compute_gail_reward(state, action):
    with torch.no_grad():
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        logits = discriminator(state, action)
        reward = -torch.log(1 - logits + 1e-8)
    return reward.cpu().numpy()

# Create a custom environment wrapper to replace rewards
class GAILRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GAILRewardWrapper, self).__init__(env)
        self.current_obs = None 

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.current_obs = obs 
        return obs

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        obs, info = self.current_obs
        obs = np.concatenate([obs[key].flatten() for key in sorted(obs.keys())])
        obs = np.expand_dims(obs, axis=0)  # 或者 obs = obs[None, :]
        action = np.expand_dims(action, axis=0)
        # Compute the GAIL reward
        gail_reward = compute_gail_reward(obs, action)
        info["original_reward"] = reward
        # return next_state, gail_reward, done, truncated, info
        return next_state, reward, done, truncated, info

# Wrap the environment
env = GAILRewardWrapper(env)

# The noise objects for TD3
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
    "disc_epochs": 5,
}

run = wandb.init(
    project="sb3_td3_TrapMaze",
    name='TD3_GAIL_TrapMaze_orgReward',
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

# Custom callback for GAIL training
from stable_baselines3.common.callbacks import BaseCallback

class GAILCallback(BaseCallback):
    def __init__(self, expert_states, expert_actions, discriminator, disc_optimizer,
                 disc_epochs=5, batch_size=64, verbose=0, update_frequency=10):
        super(GAILCallback, self).__init__(verbose)
        self.expert_states = torch.FloatTensor(expert_states).to(device)
        self.expert_actions = torch.FloatTensor(expert_actions).to(device)
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.disc_epochs = disc_epochs
        self.batch_size = batch_size
        self.update_frequency = update_frequency

    def _on_step(self) -> bool:
        if self.n_calls % self.update_frequency != 0:
            return True
        # Update the discriminator every training iteration
        for _ in range(self.disc_epochs):
            # Sample generator data from replay buffer
            if self.model.replay_buffer.size() < self.batch_size:
                continue

            gen_samples = self.model.replay_buffer.sample(self.batch_size)
            gen_obs = gen_samples.observations
            # gen_states = torch.FloatTensor(gen_samples.observations).to(device)
            gen_actions = gen_samples.actions.float()

            gen_states = torch.stack(
                [
                    torch.cat([gen_obs[key][i] for key in ['observation', 'achieved_goal', 'desired_goal']])
                    for i in range(self.batch_size)
                ]
            ).to(device).float()

            # Sample expert data
            idx = np.random.choice(len(self.expert_states), self.batch_size, replace=False)
            expert_states_batch = self.expert_states[idx].float()
            expert_actions_batch = self.expert_actions[idx].float()

            # Create labels
            expert_labels = torch.ones((self.batch_size, 1)).to(device)
            gen_labels = torch.zeros((self.batch_size, 1)).to(device)

            # Compute discriminator loss
            expert_logits = self.discriminator(expert_states_batch, expert_actions_batch)
            gen_logits = self.discriminator(gen_states, gen_actions)
            disc_loss = bce_loss(expert_logits, expert_labels) + bce_loss(gen_logits, gen_labels)

            # Optimize discriminator
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            # Log discriminator loss
            wandb.log({"discriminator_loss": disc_loss.item()})

        return True

# class LogOriginalRewardCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(LogOriginalRewardCallback, self).__init__(verbose)

#     def _on_step(self) -> bool:
#         # Access the `info` dictionary to log the original reward
#         for info in self.locals["infos"]:
#             if "original_reward" in info:
#                 wandb.log({"original_reward": info["original_reward"]})
#         return True

class LogOriginalRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LogOriginalRewardCallback, self).__init__(verbose)
        self.episode_rewards = []  # 用于存储每个 episode 的总奖励
        self.current_episode_reward = 0  # 当前 episode 的累计奖励

    def _on_step(self) -> bool:
        # 获取当前 step 的原始奖励
        for info in self.locals["infos"]:
            if "original_reward" in info:
                self.current_episode_reward += info["original_reward"]

        # 如果 episode 结束，记录总奖励并重置
        if self.locals["dones"].any():
            self.episode_rewards.append(self.current_episode_reward)
            wandb.log({"Episode Return": self.current_episode_reward})
            self.current_episode_reward = 0  # 重置当前 episode 奖励

        return True

# Create the callback list
callback = [
    WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
    GAILCallback(
        expert_states=expert_states,
        expert_actions=expert_actions,
        discriminator=discriminator,
        disc_optimizer=disc_optimizer,
        disc_epochs=config["disc_epochs"],
        batch_size=config["batch_size"],
        update_frequency=10,
    ),
    LogOriginalRewardCallback(),
]

# Start training
model.learn(
    total_timesteps=config["total_timesteps"],
    log_interval=100,
    callback=callback
)

# Save the final model and discriminator
model.save("td3_gail_TrapMaze")
torch.save(discriminator.state_dict(), "gail_discriminator.pth")

run.finish()