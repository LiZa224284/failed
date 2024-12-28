import pandas as pd
import matplotlib.pyplot as plt
import os

# 加载 CSV 文件
file_path = 'wandbtest/wandb_export_2024-12-27T19_44_27.165-06_00.csv'
data = pd.read_csv(file_path)
n = 500

# 设置要保存图像的文件夹
save_dir = 'wandbtest/figs'
os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在，创建它

# 图像保存路径
save_path = os.path.join(save_dir, 'average_episode_reward_plot.png')

# 要绘制的列
methods = [
    "SASR - average Episode Reward",
    "MyMethod - average Episode Reward",
    "GAIL - average Episode Reward",
    "TD3 - average Episode Reward"
]

# 绘图
plt.figure(figsize=(10, 6))
for method in methods:
    if method in data.columns:
        valid_data = data[method][~pd.isna(data[method])]
        moving_avg = []  # 用于存储每个点的前 10 个点（包括当前点）的平均值
        for i in range(len(valid_data)):
            # 取前 10 个点（包括当前点），不足 10 个点则取所有点
            window = valid_data[max(0, i - (n-1)):i + 1]
            moving_avg.append(window.mean())
        # plt.axhline(y=avg_value, linestyle='--', label=f"{method} (First 10 Avg)")
        plt.plot(data["Step"][~pd.isna(data[method])], moving_avg, label=method)
        # plt.plot(data["Step"][~pd.isna(data[method])], data[method][~pd.isna(data[method])], label=method)

plt.xlabel("Step")
plt.ylabel("Average Episode Reward")
plt.title("Average Episode Reward vs Step for Different Methods")
plt.legend()
plt.grid(True)

# 显示图像并保存
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"图像已保存到: {save_path}")
plt.show()