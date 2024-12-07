import pickle
import numpy as np
import matplotlib.pyplot as plt

# File paths
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_success_demos.pkl'  # Replace with the actual path
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/acc_demo/all_failed_demos.pkl'  # Replace with the actual path

# Load success and failed demos
with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

# # Extract achieved_goal from success and failed demos
# def extract_achieved_goals(demos):
#     achieved_goals = []
#     for traj in demos:
#         for step in traj:
#             achieved_goals.append(step["state"]['achieved_goal'])
#     return np.array(achieved_goals)

# success_achieved_goals = extract_achieved_goals(success_demos)
# failed_achieved_goals = extract_achieved_goals(failed_demos)

# # Plot achieved goals
# plt.figure(figsize=(8, 6))
# plt.scatter(success_achieved_goals[:, 0], success_achieved_goals[:, 1], color='red', label='Success', alpha=0.7)
# plt.scatter(failed_achieved_goals[:, 0], failed_achieved_goals[:, 1], color='blue', label='Failed', alpha=0.7)
# plt.title("Achieved Goals in Demos")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.xlim(-2.5, 2.5)  # Set x-axis range
# plt.ylim(-2, 2) 
# plt.legend()
# plt.grid(True)
# plt.show()

# Extract last 10 achieved_goal from each trajectory
def extract_last_achieved_goals(demos, last_n=10):
    achieved_goals = []
    for traj in demos:
        for step in traj[-last_n:]:  # Get only the last N timesteps
            achieved_goals.append(step["state"]['achieved_goal'])
    return np.array(achieved_goals)

success_achieved_goals = extract_last_achieved_goals(success_demos, last_n=10)
failed_achieved_goals = extract_last_achieved_goals(failed_demos, last_n=10)

# Plot achieved goals
plt.figure(figsize=(8, 6))
plt.scatter(success_achieved_goals[:, 0], success_achieved_goals[:, 1], color='red', label='Success', alpha=0.7)
plt.scatter(failed_achieved_goals[:, 0], failed_achieved_goals[:, 1], color='blue', label='Failed', alpha=0.7)
plt.title("Achieved Goals in Last 10 Timesteps")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-2.5, 2.5)  # Set x-axis range
plt.ylim(-2, 2) 
plt.legend()
plt.grid(True)
plt.show()