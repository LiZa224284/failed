import pickle
import os

# 存储所有轨迹的列表
all_demos = []

# 文件路径格式
base_path = '/Users/yuxuanli/Maze/demos/failed/'
file_prefix = 'failed_demo_'
file_extension = '.pkl'

# 合并文件
for i in range(1, 52):  # 从 1 到 51
    file_path = os.path.join(base_path, f"{file_prefix}{i}{file_extension}")
    try:
        with open(file_path, 'rb') as f:
            demo = pickle.load(f)
            all_demos.append(demo)
            print(f"Loaded {file_path}.")  # 打印已加载的文件名
    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")

# 保存合并后的数据
output_file = os.path.join(base_path, 'all_failed_demos.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(all_demos, f)

print(f"All demonstrations saved to {output_file}.")


################################ Read #################################

combined_demo_path = '/Users/yuxuanli/Maze/demos/failed/all_failed_demos.pkl'

# 读取整合后的专家演示数据
with open(combined_demo_path, 'rb') as f:
    all_demos = pickle.load(f)

# 查看数据结构和内容
print(f"Loaded {len(all_demos)} trajectories.")
print("Example trajectory (first):")
print(all_demos[0])  # 查看第一个轨迹的内容