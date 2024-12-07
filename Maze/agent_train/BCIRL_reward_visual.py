import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 定义 BC-IRL 的 Reward Network
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

# 定义 BC-IRL Reward 计算函数
def compute_bcirl_reward(reward_net, state, action):
    with torch.no_grad():
        reward = reward_net(state, action)
    return reward

def construct_obs(achieved_goal):
    """
    根据 achieved_goal 和固定的 action (0, 0) 构造 observation
    """
    observation = np.zeros(4)  # 初始化 observation
    observation[:2] = achieved_goal  # 前两维为 achieved_goal
    observation[2:] = [0, 0]  # 后两维为 action，固定为 (0, 0)

    obs_dict = {
        "observation": np.array(observation),
        "achieved_goal": np.array(achieved_goal),
        "desired_goal": np.array([0, 0]),  # 固定为 (0, 0)
    }
    obs_array = np.concatenate([obs_dict[key].flatten() for key in sorted(obs_dict.keys())])
    return np.expand_dims(obs_array, axis=0)  # 增加 batch 维度
# 可视化 BC-IRL 奖励函数
def visualize_bcirl_reward_function(reward_net_path, state_dim, action_dim, device):
    # 加载训练好的 Reward Network
    reward_net = RewardNetwork(state_dim=8, action_dim=2).to(device)
    reward_net.load_state_dict(torch.load(reward_net_path, map_location=device))
    reward_net.eval()

    # 定义状态范围（例如 x 和 y 坐标）
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2

    # 创建网格
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)

    # 构造 achieved_goal
    achieved_goals = np.c_[xx.ravel(), yy.ravel()]  # 网格点作为 achieved_goal

    # 构造 observation
    obs_list = [construct_obs(achieved_goal) for achieved_goal in achieved_goals]
    obs_tensor = torch.tensor(np.vstack(obs_list), dtype=torch.float32).to(device)
    actions_tensor = torch.zeros((obs_tensor.shape[0], action_dim), dtype=torch.float32).to(device)
    # 计算奖励
    with torch.no_grad():
        rewards = compute_bcirl_reward(reward_net, obs_tensor,actions_tensor)
        rewards = rewards.cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU
    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="BC-IRL Reward")
    plt.title("BC-IRL Reward Function Visualization")
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
    plt.show()

# 可视化奖励函数
reward_net_path = "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/BCIRL_models/mid/mid_reward_100.pth"  # 替换为你保存的 Reward Network 模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = 8
action_dim = 2
visualize_bcirl_reward_function(reward_net_path, state_dim, action_dim, device)
