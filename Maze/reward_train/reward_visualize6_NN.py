import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 加载成功和失败的演示数据
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'  # 替换为成功演示文件的路径
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_failed_demos.pkl'  # 替换为失败演示文件的路径
# success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_success_demos.pkl'
# failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_failed_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

# 提取状态（obs）和标签（reward）
def extract_obs_and_labels(demos, label, exp_k=0):
    obs = []
    rewards = []
    for traj in demos:
        demo_length = len(traj)
        for i, step in enumerate(traj):
            # 提取 achieved_goal
            achieved_goal = step["state"]['achieved_goal']
            obs.append(achieved_goal)

            # 计算时间步对应的权重
            weight = (i + 1) / demo_length  # 线性增加
            weight = weight * np.exp(exp_k * weight)  # 应用指数调制

            # 成功和失败的 reward 设置
            if label > 0:  # 成功 demo
                reward = label * weight
                # reward = 1
            else:  # 失败 demo
                reward = label * weight
                # reward = -1

            rewards.append(reward)
    return np.array(obs), np.array(rewards)

# 从成功和失败的 demos 中提取数据
success_obs, success_rewards = extract_obs_and_labels(success_demos, 1, exp_k=1)
failed_obs, failed_rewards = extract_obs_and_labels(failed_demos, -1, exp_k=1)

# 合并数据
X = np.concatenate((success_obs, failed_obs), axis=0)
y = np.concatenate((success_rewards, failed_rewards), axis=0)

# 转换为 PyTorch 张量并移动到 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# 定义奖励函数的神经网络
class RewardNetwork(nn.Module):
    def __init__(self, input_dim):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出为连续值，不使用 Sigmoid
        )
    
    def forward(self, x):
        return self.net(x)

# 初始化神经网络和优化器
reward_net = RewardNetwork(input_dim=X.shape[1]).to(device)  # 将模型移动到 GPU
optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 训练奖励函数
epochs = 200
for epoch in range(epochs):
    reward_net.train()
    optimizer.zero_grad()
    predictions = reward_net(X_tensor).squeeze()
    loss = loss_fn(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 可视化奖励函数
def visualize_reward_function(reward_net):
    # 提取环境范围
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
        rewards = reward_net(grid_states_tensor).cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU

    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="Reward")
    plt.title("Reward Function Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# 可视化奖励函数
visualize_reward_function(reward_net)