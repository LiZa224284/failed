import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import wandb
import gymnasium as gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze import TrapMazeEnv
from TD3 import TD3, ReplayBuffer

# 加载专家演示数据
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'
with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

# 提取状态和动作
def extract_obs_and_actions(demos):
    obs = []
    actions = []
    for traj in demos:
        for step in traj:
            obs.append(step["state"]["achieved_goal"])
            actions.append(step["action"])
    return np.array(obs), np.array(actions)

expert_states, expert_actions = extract_obs_and_actions(success_demos)

# 定义奖励网络
class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出奖励值
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# 初始化环境
example_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1], 
    [1, 0, 1, 0, 1, 0, 1], 
    [1, 0, 1, 'g', 't', 0, 1], 
    [1, 0, 't', 0, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1]
]
env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

# 定义参数和网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = expert_states.shape[1]
action_dim = expert_actions.shape[1]
max_action = env.action_space.high[0]

reward_net = RewardNetwork(state_dim, action_dim).to(device)
reward_optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)

# 初始化生成器（TD3）
td3_agent = TD3(state_dim, action_dim, max_action)

# ReplayBuffer
replay_buffer = ReplayBuffer(buffer_size=100000)

# 训练参数
batch_size = 32
episodes = 600
reward_epochs = 5  # 每次训练奖励网络的轮数

wandb.init(project="TD3-BCIRL", name="BC-IRL")

# 计算伪奖励
def compute_bcirl_reward(state, action):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
        reward = reward_net(state_tensor, action_tensor).item()
    return reward

# 训练循环
for episode in range(episodes):
    state, _ = env.reset()
    done, truncated = False, False
    episode_reward = 0

    while not (done or truncated):
        action = td3_agent.select_action(state["achieved_goal"])
        next_state, reward, done, truncated, _ = env.step(action)

        # 计算 BC-IRL 奖励
        pseudo_reward = compute_bcirl_reward(state["achieved_goal"], action)

        # 存入经验回放
        replay_buffer.add(state["achieved_goal"], action, pseudo_reward, next_state["achieved_goal"], done)
        state = next_state
        episode_reward += pseudo_reward

        # 更新生成器
        if replay_buffer.size() > batch_size:
            td3_agent.train(replay_buffer, batch_size)

    # 更新奖励网络
    for _ in range(reward_epochs):
        gen_states, gen_actions, _, _, _ = replay_buffer.sample(batch_size)
        idx = np.random.choice(len(expert_states), batch_size)
        expert_states_batch = torch.FloatTensor(expert_states[idx]).to(device)
        expert_actions_batch = torch.FloatTensor(expert_actions[idx]).to(device)

        # BC 损失
        gen_states = gen_states.to(device)
        gen_actions = gen_actions.to(device)
        gen_policy_actions = td3_agent.actor(gen_states)
        bc_loss = nn.MSELoss()(gen_policy_actions, gen_actions)

        reward_optimizer.zero_grad()
        bc_loss.backward()
        reward_optimizer.step()

    wandb.log({"Episode": episode, "Episode Reward": episode_reward, "BC Loss": bc_loss.item()})
    print(f"Episode {episode}: Reward {episode_reward}, BC Loss {bc_loss.item()}")

torch.save(reward_net.state_dict(), "bcirl_reward_net.pth")
torch.save(td3_agent.actor.state_dict(), "bcirl_actor.pth")