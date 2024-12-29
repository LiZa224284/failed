import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

# 使用 Seaborn 调色板创建自定义 colormap
sns_palette = sns.color_palette("Reds", as_cmap=True)

# 定义数据
x = np.linspace(-3, 3, 400)  # 修改 x 范围为 (-3, 3)
y = np.linspace(-2, 2, 400)  # 修改 y 范围为 (-2, 2)
X, Y = np.meshgrid(x, y)

# 定义高斯函数以在 (0, 0) 处产生剧烈凸起
sigma = 0.2  # 控制凸起的锐利程度，sigma 越小，凸起越尖锐
Z = np.exp(-(((X - 0) ** 2 + (Y - 0) ** 2) / (2 * sigma ** 2)))

# 归一化 Z 以便更好地可视化
Z = Z / np.max(Z)

# 计算每个 y 值对应的 alpha（透明度）
# y 从最小值到最大值映射到 alpha 从 1 到 0
y_min, y_max = y.min(), y.max()
alpha = 1 - (Y - y_min) / (y_max - y_min)

# 创建 3D 图形
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 7))

# 使用 Seaborn 的调色板作为 colormap
surf = ax.plot_surface(X, Y, Z, cmap=sns_palette, linewidth=0, antialiased=False)

# 设置 Z 轴范围和刻度
ax.set_zlim(-0.1, 1.01)  # 修改 z 轴范围
ax.zaxis.set_major_locator(plt.LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# 设置标题
ax.set_title("3D Surface Plot with Seaborn Palette")

# 显示图形
plt.show()

# 保存图形
save_path = '/home/xlx9645/failed/wandbtest/method_pip_plot3.png'
fig.savefig(save_path, dpi=300, bbox_inches='tight')