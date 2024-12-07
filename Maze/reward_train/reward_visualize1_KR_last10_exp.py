import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

def extract_last_achieved_goals_with_rewards(demos, reward_value, exp_k=1, last_n=10):

    achieved_goals = []
    rewards = []
    for traj in demos:
        demo_length = len(traj)
        start_idx = max(0, demo_length - last_n)
        for i, step in enumerate(traj[start_idx:], start=start_idx):
            achieved_goals.append(step["state"]['achieved_goal'])
            weight = (i + 1) / demo_length 
            weight = weight * np.exp(exp_k * weight)
            rewards.append(weight)  # Assign the specified reward
    return np.array(achieved_goals), np.array(rewards)

# Load success and failed demos
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_success_demos.pkl'
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_failed_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

# # Extract data
# success_obs, success_rewards = extract_obs_and_labels(success_demos, 1, exp_k=1, last_steps=10)
# failed_obs, failed_rewards = extract_obs_and_labels(failed_demos, -1, exp_k=1, last_steps=100)
success_obs, success_rewards = extract_last_achieved_goals_with_rewards(success_demos, reward_value=1, last_n=1)
failed_obs, failed_rewards = extract_last_achieved_goals_with_rewards(failed_demos, reward_value=-1, last_n=1)



# Initialize Kernel Ridge Regression models
success_kr = KernelRidge(kernel="rbf", alpha=1e-3, gamma=1)
failed_kr = KernelRidge(kernel="rbf", alpha=1e-3, gamma=1)

# Train success and failed reward functions
print("Training success reward function...")
success_kr.fit(success_obs, success_rewards)

print("Training failed reward function...")
failed_kr.fit(failed_obs, failed_rewards)


# Visualize reward functions
def visualize_kr_reward_function(kr_model, title="Reward Function"):
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid_states = np.c_[xx.ravel(), yy.ravel()]
    # grid_states = scaler.transform(grid_states)

    rewards = kr_model.predict(grid_states)
    rewards = rewards.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="Reward")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def visualize_total_kr_reward_function(kr_success_model, kr_failed_model, title="Reward Function"):
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid_states = np.c_[xx.ravel(), yy.ravel()]

    success_rewards = kr_success_model.predict(grid_states)
    failed_rewards = kr_failed_model.predict(grid_states)
    rewards = success_rewards - failed_rewards
    rewards = rewards.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="Reward")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
# Visualize success and failed reward functions
visualize_kr_reward_function(success_kr, title="Success Reward Function")
visualize_kr_reward_function(failed_kr, title="Failed Reward Function")
visualize_total_kr_reward_function(success_kr, failed_kr, title="Total Reward Function")