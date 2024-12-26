import os
import wandb
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import torch
from stable_baselines3.common.evaluation import evaluate_policy

class WandbCallback(BaseCallback):
    def __init__(self, log_dir, check_interval, verbose=1):
        super(WandbCallback, self).__init__(verbose)
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        self.save_path = os.path.join(log_dir, "best_model")
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.check_interval = check_interval  # 每隔 10 个 episode 进行一次统计
        self.episode_rewards = []  # 用于记录每个 episode 的奖励
        self.current_episode_reward = 0  # 用于累积当前 episode 的奖励
        self.episode_count = 0  # 用来统计运行了多少个 episode     
        self.save_freq = 1000
        
    def _on_step(self) -> bool:
        # 累加当前 timestep 的奖励
        self.current_episode_reward += self.locals['rewards'][0]

        # 检查是否完成了一个 episode
        if self.locals['dones']:
            # 将当前 episode 的总奖励添加到记录中
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1  # 增加 episode 计数

            # 每 10 个 episode 计算一次平均 reward
            if self.episode_count % self.check_interval == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                # wandb.log({
                #     "mean_reward_last_10_episode": mean_reward
                #     # "episodes": self.episode_count
                # }, step = self.episode_count)
                wandb.log({
                    "mean_reward_last_10_timestep": mean_reward
                }, step = self.num_timesteps)
            # 重置当前 episode 的奖励累加器
            self.current_episode_reward = 0
        
        if self.num_timesteps % self.save_freq == 0:
            save_path = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(save_path)

        return True

# class ContinuousFrozenLakeEnv(gym.Env):
#     def __init__(self, lake_size=4, hole_radius=0.1, goal_radius=0.1, max_steps=20):
#         super(ContinuousFrozenLakeEnv, self).__init__()
        
#         # 定义连续状态空间
#         self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
#                                             high=np.array([lake_size, lake_size]), 
#                                             dtype=np.float32)
        
#         # 定义连续动作空间
#         self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
#                                        high=np.array([1.0, 1.0]), 
#                                        dtype=np.float32)
        
#         self.lake_size = lake_size
#         self.hole_radius = hole_radius
#         self.goal_radius = goal_radius
#         self.max_steps = max_steps
#         self.current_step = 0  # 初始化步数计数器
        
#         # 定义洞和目标的位置
#         self.holes = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [1.0, 3.0]])
#         self.goal = np.array([3.5, 3.5])
        
#         self.seed_value = None
#         self.reset()

#     def reset(self, seed=None):
#         if seed is not None:
#             self.seed_value = seed
#             np.random.seed(self.seed_value)
        
#         # 将智能体置于远离洞和目标的随机位置
#         while True:
#             self.state = np.random.uniform(0, self.lake_size, size=(2,))
#             # self.state = np.array([1.0, 3.0])
#             if not self._is_in_hole(self.state) and not self._is_in_goal(self.state):
#                 break
        
#         self.current_step = 0  # 重置步数计数器
#         return self.state, {}

#     def step(self, action):
#         self.current_step += 1  # Increment step counter
#         reward = -0.1

#         # Check if the agent is in a hole
#         if self._is_in_hole(self.state):
#             # Agent is stuck in the hole but can still take actions within the hole's boundary
#             hole_center = self._get_hole_center(self.state)
#             potential_next_state = self.state + action
            
#             # if distance_to_hole_center <= self.hole_radius:
#             if  self._is_in_hole(potential_next_state):
#                 self.state = potential_next_state
#                 info = {"result": "1, The agent is now stuck in the hole"}
#             else:
#                 while True:
#                     random_tiny_action = np.random.uniform(-0.1, 0.1, size=self.state.shape)
#                     tmp_state = self.state + random_tiny_action

#                     if self._is_in_hole(tmp_state):
#                         self.state = tmp_state
#                         break
#                 info = {"result": "2, The agent is now stuck in the hole"}
            
#             # info = {"result": "The agent is now stuck in the hole"}
#             if self.current_step >= self.max_steps:
#                 info = {"result": "failure", "truncated": True}
#                 return self.state, reward, False, True, info #-0.5
#             return self.state, reward, False, False, info # The episode doesn't end, but the agent is stuck

#         else:
#             # Update the state based on the action if the agent is not in a hole
#             self.state = np.clip(self.state + action, 0.0, self.lake_size)

#         # Check if the agent has reached the goal
#         if self._is_in_goal(self.state):
#             info = {"result": "success"}
#             reward = 1 #(1-2)
#             return self.state, reward , True, False, info
        
#         # Check if the agent has exceeded the maximum number of steps
#         if self.current_step >= self.max_steps:
#             info = {"result": "failure", "truncated": True}
#             return self.state, reward, False, True, info #-0.5

#         # If neither, return a small negative reward to encourage reaching the goal
#         return self.state, reward, False, False, {}
    
#     def render(self, mode='human'):
#         plt.figure(figsize=(6, 6))
#         plt.xlim(0, self.lake_size)
#         plt.ylim(0, self.lake_size)
        
#         # 绘制洞
#         for hole in self.holes:
#             circle = plt.Circle(hole, self.hole_radius, color='blue', alpha=0.5)
#             plt.gca().add_patch(circle)
        
#         # 绘制目标
#         goal_circle = plt.Circle(self.goal, self.goal_radius, color='green', alpha=0.5)
#         plt.gca().add_patch(goal_circle)
        
#         # 绘制智能体
#         agent_circle = plt.Circle(self.state, 0.05, color='red')
#         plt.gca().add_patch(agent_circle)
        
#         plt.gca().set_aspect('equal', adjustable='box')
#         plt.grid(True)
#         plt.show()
    
#     def _is_in_hole(self, pos):
#         for hole in self.holes:
#             if np.linalg.norm(pos - hole) <= self.hole_radius:
#                 return True
#         return False
    
#     def _is_in_goal(self, pos):
#         return np.linalg.norm(pos - self.goal) <= self.goal_radius

#     def _get_hole_center(self, state):
#         for hole in self.holes:
#             if np.linalg.norm(state - hole) <= self.hole_radius:
#                 self.hole_center = hole
#                 return True
#         return None

class ContinuousFrozenLakeEnv(gym.Env):
    def __init__(self, lake_size=4, hole_radius=0.1, goal_radius=0.1, max_steps=20):
        super(ContinuousFrozenLakeEnv, self).__init__()
        
        # 定义连续状态空间
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), 
                                            high=np.array([lake_size, lake_size]), 
                                            dtype=np.float32)
        
        # 定义连续动作空间
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
                                       high=np.array([1.0, 1.0]), 
                                       dtype=np.float32)
        
        self.lake_size = lake_size
        self.hole_radius = hole_radius
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.current_step = 0  # 初始化步数计数器
        
        # 定义洞和目标的位置
        self.holes = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [1.0, 3.0]])
        self.goal = np.array([3.5, 3.5])
        
        self.seed_value = None
        self.reset()

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed_value = seed
            np.random.seed(self.seed_value)
        
        # 将智能体置于远离洞和目标的随机位置
        while True:
            self.state = np.random.uniform(0, self.lake_size, size=(2,))
            # self.state = np.array([1.0, 3.0])
            if not self._is_in_hole(self.state) and not self._is_in_goal(self.state):
                break
        
        self.current_step = 0  # 重置步数计数器
        return self.state, {}

    def step(self, action):
        self.current_step += 1  # Increment step counter

        distance_to_goal = np.linalg.norm(self.state - self.goal)
        distance_to_holes = min(np.linalg.norm(self.state - hole) for hole in self.holes)
        reward = (-0.01 / distance_to_holes + 0.01) + (0.01 / distance_to_goal + 0.01) - 2
        # reward = -0.01 - 2

        # Check if the agent is in a hole
        if self._is_in_hole(self.state):
            # Agent is stuck in the hole but can still take actions within the hole's boundary
            hole_center = self._get_hole_center(self.state)
            potential_next_state = self.state + action
            
            # if distance_to_hole_center <= self.hole_radius:
            if  self._is_in_hole(potential_next_state):
                self.state = potential_next_state
                info = {"result": "1, The agent is now stuck in the hole"}
            else:
                while True:
                    random_tiny_action = np.random.uniform(-0.1, 0.1, size=self.state.shape)
                    tmp_state = self.state + random_tiny_action

                    if self._is_in_hole(tmp_state):
                        self.state = tmp_state
                        break
                info = {"result": "2, The agent is now stuck in the hole"}
            
            # info = {"result": "The agent is now stuck in the hole"}
            if self.current_step >= self.max_steps:
                info = {"result": "failure", "truncated": True}
                return self.state, reward, False, True, info #-0.5
            return self.state, reward, False, False, info # The episode doesn't end, but the agent is stuck

        else:
            # Update the state based on the action if the agent is not in a hole
            self.state = np.clip(self.state + action, 0.0, self.lake_size)

        # Check if the agent has reached the goal
        if self._is_in_goal(self.state):
            info = {"result": "success"}
            return self.state, 1.0 - 2 , True, False, info
        
        # Check if the agent has exceeded the maximum number of steps
        if self.current_step >= self.max_steps:
            info = {"result": "failure", "truncated": True}
            return self.state, reward, False, True, info #-0.5

        # If neither, return a small negative reward to encourage reaching the goal
        return self.state, reward, False, False, {}
    
    def render(self, mode='human'):
        plt.figure(figsize=(6, 6))
        plt.xlim(0, self.lake_size)
        plt.ylim(0, self.lake_size)
        
        # 绘制洞
        for hole in self.holes:
            circle = plt.Circle(hole, self.hole_radius, color='blue', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # 绘制目标
        goal_circle = plt.Circle(self.goal, self.goal_radius, color='green', alpha=0.5)
        plt.gca().add_patch(goal_circle)
        
        # 绘制智能体
        agent_circle = plt.Circle(self.state, 0.05, color='red')
        plt.gca().add_patch(agent_circle)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
    
    def _is_in_hole(self, pos):
        for hole in self.holes:
            if np.linalg.norm(pos - hole) <= self.hole_radius:
                return True
        return False
    
    def _is_in_goal(self, pos):
        return np.linalg.norm(pos - self.goal) <= self.goal_radius

    def _get_hole_center(self, state):
        for hole in self.holes:
            if np.linalg.norm(state - hole) <= self.hole_radius:
                self.hole_center = hole
                return True
        return None

if __name__ == "__main__":
    # Initialize wandb
    project = "FL_1019"
    algorithm_name = 'TD3_model_prepare'
    env_name = 'FL'
    total_timesteps = int(5e4)
    log_dir = "/home/xlx9645/failed/FL/logs"
    check_interval = 10
    callback = WandbCallback(log_dir=log_dir, check_interval=check_interval)
    
    wandb.init(
        project=project,  # 同一个项目
        name=f"{algorithm_name}-{env_name}",  # 根据算法和环境生成不同的 run name
        group=algorithm_name,  # 用 group 将同一类算法归到一起
        config={"env_name": env_name, "algorithm": algorithm_name, 'total_timesteps': total_timesteps, 'log_dir': log_dir, 'check_interval': check_interval}
    )

    # Create the environment
    env = ContinuousFrozenLakeEnv(max_steps=20)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    
    

    # Add action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Create the TD3 model
    model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0, device="cuda" if torch.cuda.is_available() else "cpu")
    print("Model is using device:", model.device)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("/home/xlx9645/failed/FL/models/FL_TD3_success")

    # Evaluate Model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    def evaluate_agent(env, model, num_episodes=10):
        success_count = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action, _states = model.predict(state)
                state, reward, done, info = env.step(action)
                if done[0]:
                    if info[0].get('result') == 'success':
                        success_count += 1
        success_rate = success_count / num_episodes
        print(f"Success Rate: {success_rate * 100}% over {num_episodes} episodes")

    evaluate_agent(env, model)

