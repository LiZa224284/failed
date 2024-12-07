import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle
import os
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm

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

    def reset(self, seed=None):
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
        reward = 0

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
            reward = 1 
            return self.state, reward , True, False, info
        
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

# Function to collect expert demonstrations
def collect_successful_demonstrations(model, env, num_demos=50):
    successful_demos = []
    num_successes = 0
    
    while num_successes < num_demos:
        state, _ = env.reset()
        episode = []
        terminated = False
        truncated =False
        
        while not terminated and not truncated:
            action, _states = model.predict(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            success = np.array([True])
            
            # Store the transition (state, action, reward, next_state, done)
            episode.append((state, action, reward, next_state, terminated, truncated, success))
            
            # Move to the next state
            state = next_state
            
            if terminated or truncated:
                if info['result'] == 'success':
                    successful_demos.append(episode)
                    num_successes += 1
                    print(f"Collected {num_successes}/{num_demos} successful demonstrations")
                else:
                    print("in a fail demo")
                
        
    return successful_demos
 
def collect_failed_demonstrations(model, env, num_demos=50):
    failed_demos = []
    num_failed = 0
    
    while num_failed < num_demos:
        state, _ = env.reset()
        episode = []
        done = False
        terminated = False
        truncated =False
        
        # while not terminated and not truncated:
        while not done:
            action, _states = model.predict(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            success = np.array([False])
            
            # Store the transition (state, action, reward, next_state, done)
            episode.append((state, action, reward, next_state, terminated, truncated, success))
            
            # Move to the next state
            state = next_state
            
            if terminated or truncated:
                if info['result'] == 'failure':
                    failed_demos.append(episode)
                    num_failed += 1
                    print(f"Collected {num_failed}/{num_demos} failed demonstrations")
                else:
                    print("in a successful demo")
                
        
    return successful_demos

def DummyVecEnv_collect_successful_demonstrations(model, env, num_demos=50):
    successful_demos = []
    num_successes = 0
    
    while num_successes < num_demos:
        state = env.reset()
        episode = []
        done = False
        
        while not done:
            action, _states = model.predict(state)
            next_state, reward, done, info = env.step(action)
            success = np.array([True])
            
            # Store the transition (state, action, reward, next_state, done)
            episode.append((state, action, reward, next_state, done, success))
            
            # Move to the next state
            state = next_state
            
            if done[0]:  # Since done is an array (due to DummyVecEnv)
                if info[0].get('result') == 'success':
                    successful_demos.append(episode)
                    num_successes += 1
                    print(f"Collected {num_successes}/{num_demos} successful demonstrations")
                else:
                    print("in a fail demo")
                
        
    return successful_demos

# Function to collect failed demonstrations
def DummyVecEnv_collect_failed_demonstrations(model, env, num_demos=50):
    failed_demos = []
    num_successes = 0
    
    while num_successes < num_demos:
        state = env.reset()
        episode = []
        done = False
        
        while not done:
            action, _states = model.predict(state)
            next_state, reward, done, info = env.step(action)
            success = np.array([False])
            
            # Store the transition (state, action, reward, next_state, done)
            episode.append((state, action, reward, next_state, done, success))
            
            # Move to the next state
            state = next_state
            
            if done[0]:  # Since done is an array (due to DummyVecEnv)
                if info[0].get('result') == 'failure':
                    failed_demos.append(episode)
                    num_successes += 1
                    print(f"Collected {num_successes}/{num_demos} failed demonstrations")
                else:
                    print("in a successful demo")
                
        
    return failed_demos

class RandomPolicyModel(BaseAlgorithm):
    def __init__(self, policy, env, verbose=0):
        super(RandomPolicyModel, self).__init__(policy=policy, env=env, verbose=verbose,  learning_rate=0.1)
    
    def _setup_model(self):
        # 这里不设置模型，因为我们不需要模型来生成随机策略
        pass

    def learn(self, total_timesteps, callback=None, log_interval=None, tb_log_name="run", reset_num_timesteps=True):
        # Learn 方法不执行任何操作，因为没有实际的学习过程
        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # 随机选择一个动作
        action = self.env.action_space.sample()
        return [action], None
    
    def save(self, path):
        # 这个方法通常用于保存模型的状态，但对于随机策略来说不适用
        pass
    
    def load(self, path):
        # 从文件加载模型状态，但随机策略不需要这样做
        return self


if __name__ == "__main__":
    demo_save_dir = '/home/yuxuanli/failed_IRL_new/FL/models'
    os.makedirs(demo_save_dir, exist_ok=True)

    # Create the environment and wrap it in DummyVecEnv
    env = ContinuousFrozenLakeEnv(max_steps=20)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Load the trained model
    successful_model = TD3.load("/home/yuxuanli/failed_IRL_new/FL/models/FL_TD3_merge_success.zip", env=env)
    # failed_model = TD3.load("/home/yuxuanli/skrl_frozen_lake/checkpoints/TD3_continuous_frozenlake_stuck_BadTrained.zip", env=env)
    # failed_model = TD3.load('/home/yuxuanli/failed_IRL_new/FL/models/FL_TD3_model.zip', env=env)
    failed_model = RandomPolicyModel(policy=None, env=env)

    # Collect 50 expert demonstrations
    successful_demos = DummyVecEnv_collect_successful_demonstrations(successful_model, env, num_demos=1000)
    failed_demos = DummyVecEnv_collect_failed_demonstrations(failed_model, env, num_demos=1000)

    # successful_demos = collect_successful_demonstrations(successful_model, env, num_demos=1000)
    # failed_demos = collect_failed_demonstrations(failed_model, env, num_demos=1000)

    # Save the demonstrations to a file for later use
    # dis_successful_demos = [pair for demo in successful_demos for pair in demo]
    # dis_failed_demos = [pair for demo in failed_demos for pair in demo]

    successful_demonstrations_path = os.path.join(demo_save_dir, "successful_demonstrations_1000.pkl")
    failed_demonstrations_path = os.path.join(demo_save_dir, "failed_demonstrations_1000.pkl")
    # save demos in pickle file
    with open(successful_demonstrations_path, "wb") as f:
        pickle.dump(successful_demos, f)
    with open(failed_demonstrations_path, "wb") as f:
        pickle.dump(failed_demos, f)

    print("Successfully collected 100 expert demonstrations.")