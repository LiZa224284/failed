import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import numpy as np
import wandb
import gymnasium as gym
from collections import deque
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze import TrapMazeEnv

# 引入专家演示数据
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

# 提取状态和动作
def extract_obs_and_actions(demos):
    obs = []
    actions = []
    for traj in demos:
        for step in traj:
            # 拼接 achieved_goal 和 desired_goal
            achieved_goal = step["state"]["achieved_goal"]
            desired_goal = step["state"]["desired_goal"]
            combined_obs = np.concatenate((achieved_goal, desired_goal))
            obs.append(combined_obs)
            actions.append(step["action"])
    return np.array(obs), np.array(actions)

expert_states, expert_actions = extract_obs_and_actions(success_demos)

# 定义 BC 的网络结构 (与 TD3 的 Actor 网络一致)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

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

# 超参数定义
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
state_dim = expert_states.shape[1]  # 输入维度为拼接后的维度
action_dim = expert_actions.shape[1]
max_action = env.action_space.high[0]

batch_size = 64
epochs = 5000
learning_rate = 1e-3

# 初始化 BC 模型和优化器
bc_agent = Actor(state_dim, action_dim, max_action=max_action).to(device)
optimizer = torch.optim.Adam(bc_agent.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# 将专家数据转换为 PyTorch 张量
expert_states_tensor = torch.FloatTensor(expert_states).to(device)
expert_actions_tensor = torch.FloatTensor(expert_actions).to(device)

# 启用 wandb
wandb.init(
    project="BC-TrapMaze",  # 替换为您的项目名称
    name="Behavior Cloning",
    config={
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
    },
)

# 定义 checkpoint 保存路径
checkpoint_dir = "/home/yuxuanli/failed_IRL_new/Maze/agent_train/BC_checkpoints"  # 替换为您的保存路径
os.makedirs(checkpoint_dir, exist_ok=True)  # 如果路径不存在，则创建

# 开始训练
for epoch in range(epochs):
    bc_agent.train()

    # 随机采样一个 batch
    indices = np.random.choice(len(expert_states), batch_size)
    states_batch = expert_states_tensor[indices]
    actions_batch = expert_actions_tensor[indices]

    # 前向传播
    predicted_actions = bc_agent(states_batch)
    loss = loss_fn(predicted_actions, actions_batch)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失
    wandb.log({"Epoch": epoch, "Loss": loss.item()})
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # 每 100 轮保存一次 checkpoint
    if (epoch + 1) % 100 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"bc_agent_checkpoint_epoch_{epoch+1}.pth")
        torch.save(bc_agent.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# 保存最终模型
final_model_path = os.path.join(checkpoint_dir, "bc_agent_final.pth")
torch.save(bc_agent.state_dict(), final_model_path)
print(f"Final BC Agent model saved to {final_model_path}")

# 测试 BC 模型
bc_agent.eval()
state, _ = env.reset()
done, truncated = False, False
episode_reward = 0

while not (done or truncated):
    # 拼接 achieved_goal 和 desired_goal
    achieved_goal = state["achieved_goal"]
    desired_goal = state["desired_goal"]
    combined_obs = np.concatenate((achieved_goal, desired_goal))

    state_tensor = torch.FloatTensor(combined_obs).unsqueeze(0).to(device)
    with torch.no_grad():
        action = bc_agent(state_tensor).cpu().numpy().flatten()

    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    episode_reward += reward

print(f"Behavior Cloning Test Episode Reward: {episode_reward}")
wandb.finish()