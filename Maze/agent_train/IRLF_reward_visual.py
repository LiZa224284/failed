import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 定义 IRLF 的 Feature Network
# class FeatureNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, feature_dim=32):
#         super(FeatureNetwork, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, feature_dim)
#         )

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         return self.net(x)
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
# 定义 IRLF 奖励计算函数
def compute_irlf_reward(reward_net, state, action):
    with torch.no_grad():
        features = reward_net(state, action)
        reward = torch.sum(features, dim=1, keepdim=True)  # 奖励是特征的加权和
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
# 可视化 IRLF 奖励函数
def visualize_irlf_reward_function(reward_net_path, state_dim, action_dim, device):
    # 加载训练好的 Reward Network
    reward_net = RewardNetwork(state_dim, action_dim).to(device)
    reward_net.load_state_dict(torch.load(reward_net_path, map_location=device))
    reward_net.eval()

    # 定义状态范围（例如 x 和 y 坐标）
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2

    # # 创建状态-动作空间的网格
    # x = np.linspace(x_min, x_max, 100)
    # y = np.linspace(y_min, y_max, 100)
    # xx, yy = np.meshgrid(x, y)
    # grid_states = np.c_[xx.ravel(), yy.ravel()]
    # grid_actions = np.zeros_like(grid_states)  # 假设动作为零向量

    # # 转换为 PyTorch 张量
    # grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32).to(device)
    # grid_actions_tensor = torch.tensor(grid_actions, dtype=torch.float32).to(device)

    # # 计算奖励
    # with torch.no_grad():
    #     rewards = compute_irlf_reward(reward_net, grid_states_tensor, grid_actions_tensor)
    #     rewards = rewards.cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU
     # 定义状态范围（achieved_goal 的 x 和 y 坐标范围）
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
        rewards = compute_irlf_reward(reward_net, obs_tensor,actions_tensor)
        rewards = rewards.cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU

    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="IRLF Reward")
    plt.title("IRLF Reward Function Visualization")
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
    plt.show()

# 可视化奖励函数
reward_net_path = "/home/yuxuanli/failed_IRL_new/irlf_rewardnet_new.pth"  # 替换为你的 Reward Network 模型路径
state_dim = 8  # 根据你的环境设置
action_dim = 2  # 根据你的环境设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(reward_net_path):
    visualize_irlf_reward_function(reward_net_path, state_dim, action_dim, device)
else:
    print(f"Reward Network file not found at {reward_net_path}")