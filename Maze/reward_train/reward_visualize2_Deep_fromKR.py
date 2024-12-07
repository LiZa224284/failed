import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

def extract_last_achieved_goals_with_rewards(demos, reward_value, exp_k=1, last_n=100):

    achieved_goals = []
    rewards = []
    for traj in demos:
        demo_length = len(traj)
        start_idx = max(0, demo_length - last_n)
        for i, step in enumerate(traj[start_idx:], start=start_idx):
            achieved_goals.append(step["state"]['achieved_goal'])
            weight = (i + 1) / demo_length 
            weight = weight * np.exp(exp_k * weight)
            rewards.append(reward_value * weight)  # Assign the specified reward
    return np.array(achieved_goals), np.array(rewards)

class RewardNetwork(nn.Module):
    def __init__(self, input_dim):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_data(filepath):
    """
    Load grid_states and rewards from a .pkl file.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["grid_states"], data["rewards"]

total_data_path = "/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/from_KR/total.pkl"
total_obs, total_rewards = load_data(total_data_path)
# 使用标准化数据
# scaler = StandardScaler()
# total_obs = scaler.fit_transform(total_obs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_obs_tensor = torch.tensor(total_obs, dtype=torch.float32).to(device)
total_rewards_tensor = torch.tensor(total_rewards, dtype=torch.float32).unsqueeze(1).to(device)
print(f"数据样本数量: {total_obs.shape[0]}")  # 样本数
print(f"输入数据维度: {total_obs.shape[1]}")  # 每个样本的维度
total_reward_net = RewardNetwork(input_dim=total_obs.shape[1]).to(device)
optimizer_total = torch.optim.Adam(total_reward_net.parameters(), lr=1e-4) #5e-4
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_total, step_size=50, gamma=0.5)
loss_fn = nn.MSELoss()
# 查看模型参数总数
total_params = sum(p.numel() for p in total_reward_net.parameters())
print(f"模型参数总数: {total_params}")

def visualize_dataset(total_obs, total_rewards, title="Dataset Visualization"):
    """
    可视化数据集，显示样本点及其对应的奖励值。
    total_obs: 样本点位置 (N, 2)
    total_rewards: 奖励值 (N,)
    """
    if total_obs.shape[1] != 2:
        print("输入数据不是二维的，无法可视化")
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(total_obs[:, 0], total_obs[:, 1], c=total_rewards, cmap="coolwarm", s=20)
    plt.colorbar(scatter, label="Reward Value")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

# 调用可视化函数
visualize_dataset(total_obs, total_rewards, title="Dataset with Rewards")

print("\nTraining total reward network...")
for epoch in range(200):  # 可调整 epoch
    total_reward_net.train()
    optimizer_total.zero_grad()
    total_predictions = total_reward_net(total_obs_tensor).squeeze()
    loss_total = loss_fn(total_predictions, total_rewards_tensor)
    loss_total.backward()
    optimizer_total.step()
    # 调整学习率
    scheduler.step()

    # 打印当前学习率和 loss（每 10 个 epoch 打印一次）
    if epoch % 10 == 0:
        current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
        print(f"Epoch {epoch}, Loss: {loss_total.item():.4f}, Learning Rate: {current_lr:.6f}")

    print(f"Total Reward - Epoch {epoch + 1}, Loss: {loss_total.item():.4f}")
# 可视化单独的奖励函数
def visualize_reward_function(reward_net, title="Reward Function"):
    # 定义环境范围
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2

    # 创建网格
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid_states = np.c_[xx.ravel(), yy.ravel()]

    # 计算奖励
    grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32).to(device)
    with torch.no_grad():
        rewards = reward_net(grid_states_tensor).cpu().numpy()

    # 重塑奖励用于可视化
    rewards = rewards.reshape(xx.shape)

    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="Reward")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

visualize_reward_function(total_reward_net, title="Total Reward Function")

