import matplotlib.pyplot as plt
import os
import pandas as pd

file_path = 'wandbtest/wandb_export_2024-12-27T19_59_46.857-06_00.csv'
new_data = pd.read_csv(file_path)

# Define the save directory
save_dir = 'wandbtest/figs'
os.makedirs(save_dir, exist_ok=True)

# Define the file path for saving the plot
save_path = os.path.join(save_dir, 'success_rate_plot.png')

# Methods to plot
methods = [
    "My_SFD - Average Success Rate (last 10 eps)",
    "TD3_sparse - Average Success Rate (last 10 eps)",
    "My_SD - Average Success Rate (last 10 eps)",
    "My_FD - Average Success Rate (last 10 eps)"
]

# Plotting
plt.figure(figsize=(10, 6))
for method in methods:
    if method in new_data.columns:
        plt.plot(new_data["Step"], new_data[method], label=method)

plt.xlabel("Step")
plt.ylabel("Average Success Rate (last 10 episodes)")
plt.title("Success Rate vs Step for Different Methods")
plt.legend()
plt.grid(True)

# Save and display the plot
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {save_path}")