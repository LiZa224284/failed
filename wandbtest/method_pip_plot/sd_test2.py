import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义网格范围
x = np.linspace(0, 3, 400)
y = np.linspace(0, 2, 400)
X, Y = np.meshgrid(x, y)

# 定义高斯函数以在 (1.5, 1) 处产生剧烈凸起
sigma = 0.2  # 控制凸起的锐利程度，sigma 越小，凸起越尖锐
Z = np.exp(-(((X - 1.5) ** 2 + (Y - 1) ** 2) / (2 * sigma ** 2)))

# 归一化 Z 以便更好地可视化
Z = Z / np.max(Z)

# 创建 3D 图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',  # 使用默认的色图，可以根据需要更改
    linewidth=0,
    antialiased=False
)

# 添加颜色条（可选）
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# 设置轴标签
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')

# 设置标题
ax.set_title('以 (1.5, 1) 为中心的剧烈凸起3D曲面图')

# 设置视角（可根据需要调整）
ax.view_init(elev=30, azim=45)

# 显示图形
save_path = '/home/xlx9645/failed/wandbtest/method_pip_plot'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()