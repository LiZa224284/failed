import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class DeepKernelNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(DeepKernelNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

class DeepKernelRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, kernel_scale=1.0):
        super(DeepKernelRegressionModel, self).__init__()
        self.kernel_network = DeepKernelNetwork(input_dim, hidden_dim)
        self.kernel_scale = kernel_scale

    def forward(self, X, Y, Y_target):
        # 使用深度网络生成高维特征
        X_features = self.kernel_network(X)
        Y_features = self.kernel_network(Y)
        
        # 计算核矩阵 (使用 RBF 核)
        diff = X_features.unsqueeze(1) - Y_features.unsqueeze(0)  # (N, M, D)
        squared_distances = torch.sum(diff ** 2, dim=2)  # (N, M)
        kernel_matrix = torch.exp(-squared_distances / (2 * self.kernel_scale ** 2))  # (N, M)

        # 核回归预测
        weights = kernel_matrix / torch.sum(kernel_matrix, dim=1, keepdim=True)  # (N, M)
        predictions = torch.mm(weights, Y_target)  # (N, 1)
        return predictions

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

def load_data(filepath):
    """
    Load grid_states and rewards from a .pkl file.
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["grid_states"], data["rewards"]

total_data_path = "/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/from_KR/total.pkl"
total_obs, total_rewards = load_data(total_data_path)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
total_obs_tensor = torch.tensor(total_obs, dtype=torch.float32).to(device)
total_rewards_tensor = torch.tensor(total_rewards, dtype=torch.float32).unsqueeze(1).to(device)
dataset = TensorDataset(total_obs_tensor, total_rewards_tensor)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True) #256

# 初始化模型和优化器
input_dim = total_obs.shape[1]
hidden_dim = 32
kernel_scale = 1.0

model = DeepKernelRegressionModel(input_dim, hidden_dim, kernel_scale).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
loss_fn = nn.MSELoss()

# 训练深度核回归模型
epochs = 5
print("\nTraining Deep Kernel Regression Model...")
for epoch in tqdm(range(epochs), desc="Epochs"):
    model.train()
    epoch_loss = 0.0  # 用于统计每个 epoch 的总 loss
    
    # tqdm 包装 DataLoader 显示批次进度
    for X_batch, Y_batch in tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False):
        # 前向传播
        predictions = model(X_batch, total_obs_tensor, total_rewards_tensor)
        loss = loss_fn(predictions, Y_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(data_loader):.4f}, Learning Rate: {current_lr:.6f}")

# 测试模型
# model.eval()
# with torch.no_grad():
#     test_predictions = model(total_obs_tensor, total_obs_tensor, total_rewards_tensor)

# 可视化核回归结果
# def visualize_predictions(predictions, title="Predicted Rewards"):
#     # 重塑预测值
#     rewards = predictions.numpy().reshape(-1)
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(total_obs[:, 0], total_obs[:, 1], c=rewards, cmap="coolwarm", s=20)
#     plt.colorbar(scatter, label="Predicted Reward Value")
#     plt.title(title)
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.grid(True)
#     plt.show()

def visualize_deep_kr_predictions(deep_kr_model, X_train, Y_train, batch_size=512, title="Deep-KR Predicted Rewards"):
    """
    使用 DataLoader 实现 Deep Kernel Regression 的奖励分布可视化。

    Args:
        deep_kr_model: 训练好的 Deep-KR 模型
        X_train: 训练数据 (torch.Tensor)
        Y_train: 对应的目标奖励值 (torch.Tensor)
        batch_size: 每次预测的网格点批量大小
        title: 可视化图标题
    """
    # 确定模型设备
    device = next(deep_kr_model.parameters()).device

    # 设置模型为评估模式
    deep_kr_model.eval()

    # 定义可视化网格范围
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2

    # 创建网格点
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid_states = np.c_[xx.ravel(), yy.ravel()]  # 转换为 (N, 2)

    # 转换为 Tensor 并创建 DataLoader
    grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32).to(device)
    grid_dataset = TensorDataset(grid_states_tensor)
    grid_loader = DataLoader(grid_dataset, batch_size=batch_size, shuffle=False)

    # 分批处理网格点预测
    predictions = []
    for batch in grid_loader:
        batch_grid_states = batch[0]  # 从 DataLoader 中取出当前批次
        with torch.no_grad():  # 禁用梯度计算以节省显存
            batch_predictions = deep_kr_model(batch_grid_states, X_train, Y_train)
            predictions.append(batch_predictions.cpu().numpy())  # 将结果移动到 CPU 并存储

    # 拼接所有批次的预测结果
    predictions = np.concatenate(predictions, axis=0)

    # 将预测结果重塑为网格形状
    rewards = predictions.reshape(xx.shape)

    # 绘制奖励分布图
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="Predicted Reward")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# visualize_deep_kr_predictions(model, title="Predicted Rewards (Deep Kernel Regression)")
visualize_deep_kr_predictions(
    deep_kr_model=model, 
    X_train=total_obs_tensor, 
    Y_train=total_rewards_tensor, 
    title="Predicted Rewards (Deep-KR)"
)