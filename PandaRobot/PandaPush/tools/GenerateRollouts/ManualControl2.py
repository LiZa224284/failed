import gymnasium as gym
import panda_gym
import imageio
import numpy as np
import pickle
import logging
from collections import namedtuple

# 设置日志记录
log_path = "/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/ManualControl_Log.txt"
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# 定义 TrajectoryWithRew 数据结构，使用 'terminal' 而不是 'terminals'
TrajectoryWithRew = namedtuple('TrajectoryWithRew', ['obs', 'acts', 'rews', 'infos', 'terminal'])

# 初始化 PandaPush 环境
env = gym.make("PandaPush-v3", render_mode="rgb_array")
total_episodes = 5  
episode_count = 0
all_frames = []  # 用于保存所有成功 episode 的帧
successful_demos = []  # 存储所有成功的 demo

while episode_count < total_episodes:
    observation, info = env.reset()
    
    aligned = False  
    align_steps = 0
    forward_steps = 0
    total_steps = 0
    max_forward_attempts = 40
    max_total_steps = 100
    success = False

    # 用于存储当前 episode 的数据
    episode_obs = []
    episode_acts = []
    episode_rews = []
    episode_infos = []
    episode_terminal = []

    while total_steps < max_total_steps:
        current_position = observation["observation"][0:3]   
        object_position = observation["achieved_goal"][0:3]   
        desired_position = observation["desired_goal"][0:3] 

        object_to_goal = desired_position - object_position
        object_to_goal /= np.linalg.norm(object_to_goal) 

        if not aligned:
            # align 阶段
            target_position = object_position - 0.05 * object_to_goal 
            target_position[2] = 0.05 

            action = 4.0 * (target_position - current_position)
            align_steps += 1
            total_steps += 1

            # 检查是否已经对齐
            if np.linalg.norm(current_position - target_position) < 0.002:
                aligned = True  
                forward_steps = 0
                logger.info("Effector Aligned")
        else:
            # forward 阶段
            action = 4.0 * (desired_position - current_position)
            forward_steps += 1
            total_steps += 1

            # 检查是否成功
            if np.linalg.norm(object_position - desired_position) < 0.01:
                success = True
                logger.info("Success!")
                break  

            # 检查 forward 阶段步数是否超过限制
            if forward_steps > max_forward_attempts:
                aligned = False  # 重新进入 align 阶段
                logger.info("Re-aligning due to forward step limit")

        # 执行动作并存储数据
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 存储当前步的数据
        episode_obs.append(observation["observation"])
        episode_acts.append(np.array(action))
        episode_rews.append(np.float64(reward))
        episode_infos.append(info)
        # episode_terminal.append(terminated)

    # 如果成功，保存 episode 的数据
    if success:
        logger.info(f"Episode {episode_count + 1} completed successfully:")
        
        # 创建 TrajectoryWithRew 对象并添加到成功的 demo 列表
        trajectory = TrajectoryWithRew(
            obs=np.array(episode_obs),
            acts=np.array(episode_acts),
            rews=np.array(episode_rews),
            infos=episode_infos,
            terminal=terminated
        )
        successful_demos.append(trajectory)

        episode_count += 1  # 仅在成功时增加 episode_count
    else:
        logger.info(f"Episode {episode_count + 1} failed due to step limit:")
    
    logger.info(f"  Align steps: {align_steps}")
    logger.info(f"  Forward steps: {forward_steps}")
    logger.info(f"  Total steps: {total_steps}")

# 关闭环境
env.close()

# 保存成功的 demo 到 pkl 文件中
save_path = "/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/ManualControl_SuccessfulDemos.pkl"
with open(save_path, "wb") as f:
    pickle.dump(successful_demos, f)

logger.info(f"所有成功的 demo 已保存至 {save_path}")
print(f"日志已保存至 {log_path}，所有成功的 demo 已保存至 {save_path}")