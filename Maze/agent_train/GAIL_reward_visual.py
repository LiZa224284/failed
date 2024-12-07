import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 定义 GAIL 的 Reward Function 网络
# class Discriminator(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256):
#         super(Discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#         )

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         return self.net(x)
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出为 [0, 1] 概率
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)
# 定义 GAIL Reward 计算函数
def compute_gail_reward(discriminator, state, action):
    with torch.no_grad():
        logits = discriminator(state, action)
        reward = -torch.log(1 - torch.sigmoid(logits) + 1e-8)
    return reward

# # 可视化 GAIL 生成的奖励函数
# def visualize_gail_reward_function(discriminator_path, state_dim, action_dim, device):
#     # 加载训练好的 Discriminator
#     discriminator = Discriminator(state_dim, action_dim).to(device)
#     discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
#     discriminator.eval()

#     # 定义状态范围（例如 x 和 y 坐标）
#     x_min, x_max = -2.5, 2.5
#     y_min, y_max = -2, 2

#     # 创建网格
#     x = np.linspace(x_min, x_max, 100)
#     y = np.linspace(y_min, y_max, 100)
#     xx, yy = np.meshgrid(x, y)
#     grid_states = np.c_[xx.ravel(), yy.ravel()]
#     grid_actions = np.zeros_like(grid_states)  # 假设动作为零向量

#     # 转换为 PyTorch 张量
#     grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32).to(device)
#     grid_actions_tensor = torch.tensor(grid_actions, dtype=torch.float32).to(device)

#     # 计算奖励
#     with torch.no_grad():
#         rewards = compute_gail_reward(discriminator, grid_states_tensor, grid_actions_tensor)
#         rewards = rewards.cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU

#     # 绘制奖励函数
#     plt.figure(figsize=(8, 6))
#     plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
#     plt.colorbar(label="GAIL Reward")
#     plt.title("GAIL Reward Function Visualization")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()
# 构造 obs 的函数
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

# 可视化 GAIL Reward 函数
def visualize_gail_reward_function(discriminator_path, device):
    # 加载训练好的 Discriminator
    state_dim = 8  # observation + achieved_goal + desired_goal 的总维度
    action_dim = 2  # 动作已包含在 observation 中，不需要额外输入
    discriminator = Discriminator(state_dim, action_dim).to(device)
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    discriminator.eval()

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
        rewards = compute_gail_reward(discriminator, obs_tensor,actions_tensor)
        rewards = rewards.cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU

    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="GAIL Reward")
    plt.title("GAIL Reward Function Visualization")
    plt.xlabel("Achieved Goal X")
    plt.ylabel("Achieved Goal Y")
    plt.show()



# 可视化奖励函数
discriminator_path = "/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/GAIL_models/map2_gail_discriminator.pth"  # 替换为你保存的 Discriminator 模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(discriminator_path):
    visualize_gail_reward_function(discriminator_path, device)
else:
    print(f"Discriminator file not found at {discriminator_path}")