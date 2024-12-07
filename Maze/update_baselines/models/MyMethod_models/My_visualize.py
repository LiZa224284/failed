import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
import gymnasium as gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from Maze.TrapMaze import TrapMazeEnv

# 加载成功和失败的演示数据
# success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_success_demos.pkl'
# failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/random_demo/all_failed_demos.pkl'
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_success_demos.pkl'
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_failed_demos.pkl'
with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

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

def extract_obs_and_actions(demos, label, exp_k=0):
    obs = []
    actions = []
    rewards = []
    for traj in demos:
        demo_length = len(traj)
        for i, step in enumerate(traj):
            state = step["state"]
            obs_input = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            obs.append(obs_input)
            actions.append(step["action"])

            # exp_k = 1
            weight = (i + 1) / demo_length  # 线性增加
            weight = weight * np.exp(exp_k * weight)  # 应用指数调制

            # 成功和失败的 reward 设置
            reward = label * weight
            rewards.append(reward)

    return np.array(obs), np.array(actions), np.array(rewards)

def extract_reward(traj, label, exp_k=1):
    obs = []
    actions = []
    rewards = []
    demo_length = len(traj)

    for i, step in enumerate(traj):
        state = step["state"]
        # obs_input = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
        obs.append(state)
        # actions.append(step["action"])

        # exp_k = 1
        weight = (i + 1) / demo_length  # 线性增加
        weight = weight * np.exp(exp_k * weight)  # 应用指数调制

        # 成功和失败的 reward 设置
        reward = label * weight
        rewards.append(reward)

    return np.array(obs), np.array(rewards)


# 从成功和失败的 demos 中提取数据
# success_obs, success_rewards = extract_obs_and_labels(success_demos, 1, exp_k=1)
# failed_obs, failed_rewards = extract_obs_and_labels(failed_demos, -1, exp_k=1)

expert_states, expert_actions, success_rewards = extract_obs_and_actions(success_demos, 1, exp_k=1)
failed_states, failed_actions, failed_rewards = extract_obs_and_actions(failed_demos, -1, exp_k=1)
    
# 合并数据
X = np.concatenate((expert_states, failed_states), axis=0)
y = np.concatenate((success_rewards, failed_rewards), axis=0)
# 转换为 PyTorch 张量并移动到 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
reward_net = RewardNetwork(input_dim=X.shape[1]).to(device)  # 将模型移动到 GPU
optimizer = torch.optim.Adam(reward_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()



# 训练奖励函数
epochs = 200
# model_save_path = "/home/yuxuanli/failed_IRL_new/Maze/reward_train/reward_function_model.pth"  # 保存模型的路径
model_save_path = '/home/yuxuanli/failed_IRL_new/Maze/update_baselines/models/MyMethod_models/my_reward_trained.pth'

for epoch in range(epochs):
    reward_net.train()
    optimizer.zero_grad()
    predictions = reward_net(X_tensor).squeeze()
    loss = loss_fn(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 保存模型
torch.save(reward_net.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

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

# 可视化奖励函数
def visualize_reward_function_with_fixed_desired_goal(model_path, device):
    # 加载模型
    # reward_net = RewardNetwork(input_dim=input_dim).to(device)
    reward_net = RewardNetwork(input_dim=8).to(device)
    reward_net.load_state_dict(torch.load(model_path, map_location=device))
    reward_net.eval()  # 设置为评估模式

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
    # actions_tensor = torch.zeros((obs_tensor.shape[0], action_dim), dtype=torch.float32).to(device)
    # 计算奖励
    with torch.no_grad():
        rewards = reward_net(obs_tensor)
        rewards = rewards.cpu().numpy().reshape(xx.shape)

    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="GAIL Reward")
    plt.title("GAIL Reward Function Visualization")
    plt.xlabel("Achieved Goal X")
    plt.ylabel("Achieved Goal Y")
    plt.show()
    # 获取 achieved_goal 和 desired_goal 的维度
    # achieved_goal_dim = env.observation_space["achieved_goal"].shape[0]
    # desired_goal_dim = env.observation_space["desired_goal"].shape[0]
    # input_dim = achieved_goal_dim + desired_goal_dim

    # 获取环境中的固定 desired_goal
    # state, _ = env.reset()
    # fixed_desired_goal = state['desired_goal']  # 固定值

    # # 定义 achieved_goal 的范围
    # achieved_goal_min = -2.5  # 根据具体环境调整
    # achieved_goal_max = 2.5
    # achieved_goal = np.linspace(achieved_goal_min, achieved_goal_max, 50)

    # # 创建 achieved_goal 的网格
    # ag = np.meshgrid(*[achieved_goal for _ in range(achieved_goal_dim)])
    # grid_achieved_goal = np.vstack([x.ravel() for x in ag]).T

    # # 将 fixed_desired_goal 与每个 achieved_goal 组合
    # grid_desired_goal = np.tile(fixed_desired_goal, (grid_achieved_goal.shape[0], 1))
    # grid_states_combined = np.hstack([grid_achieved_goal, grid_desired_goal])

    # # 转换为 PyTorch 张量
    # grid_states_tensor = torch.tensor(grid_states_combined, dtype=torch.float32).to(device)

    # # 计算奖励
    # with torch.no_grad():
    #     rewards = reward_net(grid_states_tensor).cpu().numpy().reshape(achieved_goal_dim * [50])

    # # 绘制奖励函数
    # if achieved_goal_dim == 1:  # 一维情况下的可视化
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(achieved_goal, rewards)
    #     plt.xlabel("Achieved Goal")
    #     plt.ylabel("Reward")
    #     plt.title("Reward Function with Fixed Desired Goal")
    #     plt.show()
    # elif achieved_goal_dim == 2:  # 二维情况下的可视化
    #     plt.figure(figsize=(8, 6))
    #     plt.contourf(ag[0], ag[1], rewards, levels=50, cmap="viridis")
    #     plt.colorbar(label="Reward")
    #     plt.title("Reward Function with Fixed Desired Goal")
    #     plt.xlabel("Achieved Goal Dim 1")
    #     plt.ylabel("Achieved Goal Dim 2")
    #     plt.show()
    # else:
    #     print("Visualization for dimensions higher than 2 is not supported.")

# 初始化环境
# 初始化环境
example_map = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1], 
    [1, 0, 1, 0, 1, 0, 1], 
    [1, 0, 1, 'g', 't', 0, 1], 
    [1, 0, 't', 0, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1]
]
# env = gym.make('TrapMazeEnv', maze_map=example_map, render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

# 可视化奖励函数
if os.path.exists(model_save_path):
    visualize_reward_function_with_fixed_desired_goal(model_save_path, device)
else:
    print(f"Model file not found at {model_save_path}")