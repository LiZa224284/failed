import gymnasium as gym
import panda_gym
import numpy as np
import pickle
from imitation.data.types import TrajectoryWithRew, DictObs

# 初始化 PandaPush 环境
env = gym.make("PandaPush-v3", render_mode="rgb_array")
total_episodes = 500  
episode_count = 0
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
    episode_obs_observation = []
    episode_obs_achieved_goal = []
    episode_obs_desired_goal = []
    episode_acts = []
    episode_rews = []
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
                print("Effector Aligned")
        else:
            # forward 阶段
            action = 4.0 * (desired_position - current_position)
            forward_steps += 1
            total_steps += 1

            # 检查是否成功
            if np.linalg.norm(object_position - desired_position) < 0.01:
                success = True
                print("Success!")
                break  

            # 检查 forward 阶段步数是否超过限制
            if forward_steps > max_forward_attempts:
                aligned = False  # 重新进入 align 阶段
                print("Re-aligning due to forward step limit")

        # 执行动作并存储数据
        observation, reward, terminated, truncated, info = env.step(action)

        # 逐步追加数据到各个字段的列表中
        episode_obs_observation.append(observation["observation"])
        episode_obs_achieved_goal.append(observation["achieved_goal"])
        episode_obs_desired_goal.append(observation["desired_goal"])

        # 存储当前步的数据
        episode_acts.append(np.array(action, dtype=np.float64))  # 确保动作为 float64
        episode_rews.append(np.float64(reward))  # 确保奖励为 float64
        episode_terminal.append(terminated)

    # 如果成功，保存 episode 的数据
    if success:
        print(f"Episode {episode_count + 1} completed successfully:")
        # episode_obs.append(obs)
        episode_obs_observation.append(observation["observation"])
        episode_obs_achieved_goal.append(observation["achieved_goal"])
        episode_obs_desired_goal.append(observation["desired_goal"])

        obs = DictObs({
            'observation': np.array(episode_obs_observation, dtype=np.float64),
            'achieved_goal': np.array(episode_obs_achieved_goal, dtype=np.float64),
            'desired_goal': np.array(episode_obs_desired_goal, dtype=np.float64)
        })

        trajectory = TrajectoryWithRew(
            obs=obs,
            acts=np.array(episode_acts, dtype=np.float64),
            rews=np.array(episode_rews, dtype=np.float64),
            infos=None,
            terminal=bool(terminated)
        )
        successful_demos.append(trajectory)

        episode_count += 1  # 仅在成功时增加 episode_count

# 关闭环境
env.close()

# 保存成功的 demo 到 pkl 文件中
save_path = "/files1/Yuxuan_Li/failed_demos/Experiments/Robots/PandaPush/GenerateDemos/ManualControl_SuccessfulDemos.pkl"
with open(save_path, "wb") as f:
    pickle.dump(successful_demos, f)

print(f"所有成功的 demo 已保存至 {save_path}")