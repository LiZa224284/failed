import pickle

# 文件路径
demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/success_demo_1.pkl'

# 读取专家演示数据
with open(demo_path, 'rb') as f:
    expert_demo = pickle.load(f)

# 查看数据结构和内容
print(f"Loaded {len(expert_demo)} demonstrations.")
print("Example demonstration step:")
print(expert_demo[-1])  # 查看第一个时间步的数据