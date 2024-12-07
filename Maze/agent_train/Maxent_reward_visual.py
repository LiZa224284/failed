import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 定义奖励网络 (Reward Network)
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

# 定义 MaxEnt 奖励计算函数
def compute_maxent_reward(reward_net, state, action):
    with torch.no_grad():
        reward = reward_net(state, action)
    return reward

# 可视化 MaxEnt IRL 奖励函数
def visualize_maxent_reward_function(reward_net_path, state_dim, action_dim, device):
    # 加载训练好的 Reward Network
    reward_net = RewardNetwork(state_dim, action_dim).to(device)
    reward_net.load_state_dict(torch.load(reward_net_path, map_location=device))
    reward_net.eval()

    # 定义状态范围（例如 x 和 y 坐标）
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2

    # 创建网格
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid_states = np.c_[xx.ravel(), yy.ravel()]
    grid_actions = np.zeros_like(grid_states)  # 假设动作为零向量

    # 转换为 PyTorch 张量
    grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32).to(device)
    grid_actions_tensor = torch.tensor(grid_actions, dtype=torch.float32).to(device)

    # 计算奖励
    with torch.no_grad():
        rewards = compute_maxent_reward(reward_net, grid_states_tensor, grid_actions_tensor)
        rewards = rewards.cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU

    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="MaxEnt IRL Reward")
    plt.title("MaxEnt IRL Reward Function Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# 可视化奖励函数
reward_net_path = "/home/yuxuanli/failed_IRL_new/maxent_reward_net.pth"  # 替换为你保存的 Reward Network 模型路径
state_dim = 2  # 根据你的环境设置
action_dim = 2  # 根据你的环境设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(reward_net_path):
    visualize_maxent_reward_function(reward_net_path, state_dim, action_dim, device)
else:
    print(f"Reward Network file not found at {reward_net_path}")