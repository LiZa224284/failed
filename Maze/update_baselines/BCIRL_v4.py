import wandb  # 添加 wandb
import sys
import os
import copy
import gymnasium as gym
import gymnasium_robotics
# 获取当前脚本所在路径的上一级路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Maze.TrapMaze_action import TrapMazeEnv
# from Maze.TrapMaze import TrapMazeEnv
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plts
import higher
 
import sys
import os
import copy
import pickle
import gymnasium as gym
import gymnasium_robotics
from TrapMazeEnv import TrapMazeEnv
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from TD3_diff import TD3, ReplayBuffer
import matplotlib.pyplot as plt
import argparse
import higher

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0, hidden_dim=256):
        super(Actor, self).__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        # self.state = torch.zeros_like((state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        # self.reward = torch.zeros_like((torch.Tensor(1)))
        self.done_bool = np.zeros((max_size, 1))

        self.device = device


    def add(self, state, action, next_state, reward, done_bool):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done_bool[self.ptr] = done_bool

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        # ind = np.random.randint(0, self.size, size=batch_size)
        ind = np.random.randint(0, self.size, size=batch_size)
        # effective_batch_size = min(batch_size, self.size)
        # ind = np.random.choice(self.size, size=effective_batch_size, replace=False)

        return (
        torch.FloatTensor(self.state[ind]).to(self.device),
        torch.FloatTensor(self.action[ind]).to(self.device),
        torch.FloatTensor(self.reward[ind]).to(self.device),
        torch.FloatTensor(self.next_state[ind]).to(self.device),
        torch.FloatTensor(self.done_bool[ind]).to(self.device),
        )

class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        actor_lr=1e-4, # 1e-3
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        noise_std=0.2,
        noise_clip=0.5,
        policy_delay=1,
        buffer_size=int(1e6),
        batch_size=256,
        device='cuda',
        ReplayBuffer=None,
    ):
        self.device = device

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim=256).to(self.device)
        # self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10000, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=10000, gamma=0.9)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer
        self.batch_size = batch_size

        # Training Parameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def get_actor_params(self):
        # 返回actor的参数，用于outer loop获取
        return [p for p in self.actor.parameters()]

    def set_actor_params(self, params):
        # 将params拷贝给actor的参数，用于outer loop还原actor初始参数
        with torch.no_grad():
            for p, dp in zip(self.actor.parameters(), params):
                p.copy_(dp)

    def inner_update_actor_critic(self, reward_net, inner_steps=2):

        self.total_it += 1

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # Sample batch from replay buffer
        states, actions, rewards, next_states, done_bool = self.replay_buffer.sample(self.batch_size)

        initial_actor_params = [p.clone() for p in self.actor.parameters()]

        with higher.innerloop_ctx(self.actor, self.actor_optimizer, copy_initial_weights=True) as (actor_diff, actor_optimizer_diff):
            # inner loop对critic可以直接用梯度下降更新（如果希望critic也在inner loop中可微跟踪，可用higher包装critic）
            # 这里简化处理，只对actor做inner update，可根据需要将critic也用higher包装。
            for step in range(inner_steps):
                # Critic更新（不一定需要higher，这里简化用普通更新）
                with torch.no_grad():
                    noise = (torch.randn_like(actions)*self.noise_std).clamp(-self.noise_clip, self.noise_clip)
                    next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
                    target_q1, target_q2 = self.critic_target(next_states, next_actions)
                    target_q = (1 - done_bool)*self.gamma*torch.min(target_q1, target_q2)

                with higher.innerloop_ctx(self.critic, self.critic_optimizer, copy_initial_weights=True) as (critic_diff, critic_optimizer_diff):
                    current_q1, current_q2 = critic_diff(states, actions)
                    critic_loss = F.mse_loss(current_q1, rewards + target_q.detach()) + F.mse_loss(current_q2, rewards + target_q.detach())
                    # critic_optimizer_diff.zero_grad()
                    # critic_loss.backward()
                    # critic_optimizer_diff.step()
                    critic_optimizer_diff.step(critic_loss)
                    wandb.log({"Critic Loss": critic_loss})

                    # Actor更新（通过actor_diff更新）
                    actor_loss = -critic_diff.Q1(states, actor_diff(states)).mean()
                    actor_optimizer_diff.step(actor_loss)
                    wandb.log({"Actor Loss": actor_loss})

            # inner loop结束后，actor_diff中是更新后的参数
            updated_actor_params = [p.clone() for p in actor_diff.parameters()]

        # 更新target网络
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

        # 将actor参数恢复到初始状态（outer loop需要对actor再次进行测试时保持初始参数）
        self.set_actor_params(initial_actor_params)

        return updated_actor_params

class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出奖励值
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

def compute_bcirl_reward(reward_net, state, action):
    with torch.no_grad():
        reward = reward_net(state, action)
    return reward

def construct_obs(achieved_goal):
    observation = np.zeros(4)  # 初始化 observation
    observation[:2] = achieved_goal  # 前两维为 achieved_goal
    observation[2:] = [0, 0]  # 后两维为 action，固定为 (0, 0)

    obs_dict = {
        "observation": np.array(observation),
        "achieved_goal": np.array(achieved_goal),
        "desired_goal": np.array([0, 0]),  # 固定为 (0, 0)
    }
    obs_array = np.concatenate([obs_dict[key].flatten() for key in sorted(obs_dict.keys())])
    return np.expand_dims(obs_array, axis=0)  # 增加 batch 维度

def visualize_bcirl_reward_function(reward_net_path, state_dim, action_dim, device, figure_save_path):
    # 加载训练好的 Reward Network
    reward_net = RewardNetwork(state_dim=8, action_dim=2).to(device)
    reward_net.load_state_dict(torch.load(reward_net_path, map_location=device))
    reward_net.eval()

    # 定义状态范围（例如 x 和 y 坐标）
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2, 2

    # 创建网格
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x, y)

    # 构造 achieved_goal
    achieved_goals = np.c_[xx.ravel(), yy.ravel()]  # 网格点作为 achieved_goal

    # 构造 observation
    obs_list = [construct_obs(achieved_goal) for achieved_goal in achieved_goals]
    obs_tensor = torch.tensor(np.vstack(obs_list), dtype=torch.float32).to(device)
    actions_tensor = torch.zeros((obs_tensor.shape[0], action_dim), dtype=torch.float32).to(device)
    # 计算奖励
    with torch.no_grad():
        rewards = compute_bcirl_reward(reward_net, obs_tensor,actions_tensor)
        rewards = rewards.cpu().numpy().reshape(xx.shape)  # 将结果移回 CPU
    # 绘制奖励函数
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, rewards, levels=50, cmap="viridis")
    plt.colorbar(label="BC-IRL Reward")
    plt.title("BC-IRL Reward Function Visualization")
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
    plt.savefig(figure_save_path)
    plt.close()

def extract_obs_and_actions(demos):
    obs = []
    actions = []
    for traj in demos:
        for step in traj:
            state = step["state"]
            obs_input = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            obs.append(obs_input)
            actions.append(step["action"])
    return np.array(obs), np.array(actions)

def parse_args():
    parser = argparse.ArgumentParser(description="Run kernel regression with specified parameters.")
    parser.add_argument('--update_timesteps', type=int, default=3000)
    parser.add_argument('--reward_epochs', type=int, default=200)
    return parser.parse_args()
   
success_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_success_demos_16.pkl'
failed_demo_path = '/home/yuxuanli/failed_IRL_new/Maze/demo_generate/demos/action_trapMaze/all_failed_demos.pkl'

with open(success_demo_path, 'rb') as f:
    success_demos = pickle.load(f)

with open(failed_demo_path, 'rb') as f:
    failed_demos = pickle.load(f)

expert_states, expert_actions = extract_obs_and_actions(success_demos)

if __name__ == "__main__":
    args = parse_args()
    wandb.init(
        project="Map0",  # 替换为你的项目名称
        name='BCIRL_lr_test',
        config={
            "batch_size": 256,
            "buffer_size": int(1e6),
        },
    )
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    example_map = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 'g', 't', 0, 1],
        [1, 0, 't', 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    env = gym.make('TrapMazeEnv', maze_map=example_map, reward_type="sparse", render_mode="rgb_array", max_episode_steps=100, camera_name="topview")

    state_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=int(1e6), device=device)  
    # td3_agent = TD3(state_dim, action_dim, max_action, device=device, ReplayBuffer=replay_buffer)
    
    reward_net = RewardNetwork(state_dim, action_dim).to(device)
    reward_optimizer = optim.Adam(reward_net.parameters(), lr=1e-4)
    reward_scheduler = optim.lr_scheduler.StepLR(reward_optimizer, step_size=1000, gamma=0.9)
    # reward_epochs = 200 #5
    actor = Actor(state_dim, action_dim, max_action, hidden_dim=256).to(device)
    actor_target = Actor(state_dim, action_dim, max_action, hidden_dim=256).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

    batch_size = 512 #128 
    # episodes = int(5e6)
    max_timsteps = int(300e10) #int(6e4)
    start_timesteps = 0 #100 #int(25e3)
    episode_timesteps = 0
    episode_reward = 0
    pseudo_episode_reward = 0
    episode_num = 0
    outer_steps = 10 # 每隔一定episode或一定条件后对reward_net进行外层更新
    inner_steps = 2 

    gamma = 0.09
    noise_std=0.2
    noise_clip=0.5
    tau=0.005

    state, _ = env.reset()
    state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
    done, truncated = False, False

    success_buffer = []

    for t in range(max_timsteps):
        episode_timesteps += 1
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            noise = 0.2
            # action = action = td3_agent.select_action(state=state)
            state = torch.FloatTensor(state).to(device).unsqueeze(0)
            action = actor(state).cpu().data.numpy().flatten()
            action += noise * np.random.normal(size=action.shape)
            action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.concatenate([next_state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
        done = torch.tensor(done, dtype=torch.bool)
        truncated = torch.tensor(truncated, dtype=torch.bool)
        done_bool = torch.logical_or(done, truncated).float()

        # state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
        state_tensor = state
        action_tensor = torch.from_numpy(action).float().to(device).unsqueeze(0)
        state = state_tensor.cpu().numpy()
        # pseudo_reward = compute_bcirl_reward(reward_net, state_tensor, action_tensor)
        pseudo_reward = reward_net(state_tensor, action_tensor)
        # print(f'pseudo_r: {pseudo_reward}')
        pseudo_reward = torch.clamp(pseudo_reward, min=-10, max=10)
        # pseudo_reward = pseudo_reward.cpu().numpy()
        pseudo_reward = pseudo_reward
        # print(f'clamp_pseudo_r: {pseudo_reward}')
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        pseudo_episode_reward += pseudo_reward

        if (done or truncated):
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            wandb.log({"Episode Reward": episode_reward})
            wandb.log({"Pseudo Episode Reward": pseudo_episode_reward})

            if info['success'] == True:
                success_buffer.append(1)
            elif info['success'] == False:
                success_buffer.append(0)

            if (t + 1) % 10 == 0:
                avg_success = np.mean(success_buffer[-10:])  # 最近10个episode的平均成功率
                print(f"Episode {episode_num+1}, Average Success Rate (last 10 eps): {avg_success:.2f}")
                wandb.log({"Average Success Rate (last 10 eps)": avg_success})

            state, _ = env.reset()
            state = np.concatenate([state[key].flatten() for key in ['observation', 'achieved_goal', 'desired_goal']])
            done, truncated = False, False
            episode_reward = 0
            pseudo_episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # with higher.innerloop_ctx(reward_net, reward_optimizer, copy_initial_weights=True) as (reward_net_diff, reward_optimizer_diff):
            
        # 在给定的reward_net_diff下，进行actor的inner loop更新
        # inner_update_actor_critic会返回更新后的actor参数
        # updated_actor_params = td3_agent.inner_update_actor_critic(reward_net=reward_net_diff, inner_steps=inner_steps)
        states, actions, rewards, next_states, done_bool = replay_buffer.sample(batch_size)

        with higher.innerloop_ctx(actor, actor_optimizer, copy_initial_weights=True) as (actor_diff, actor_optimizer_diff):
            with higher.innerloop_ctx(critic, critic_optimizer, copy_initial_weights=True) as (critic_diff, critic_optimizer_diff):
                for step in range(inner_steps):
                    with torch.no_grad():
                        noise = (torch.randn_like(actions)*noise_std).clamp(-noise_clip, noise_clip)
                        next_actions = (actor_target(next_states) + noise).clamp(-max_action, max_action)
                        target_q1, target_q2 = critic_target(next_states, next_actions)
                        target_q = (1 - done_bool)*gamma*torch.min(target_q1, target_q2)
                    
                    current_q1, current_q2 = critic_diff(states, actions)
                    rewards = reward_net(states, actions)
                    target_q = rewards + (1 - done_bool) * gamma * torch.min(target_q1, target_q2)
                    critic_loss = F.mse_loss(current_q1, rewards + target_q.detach()) + F.mse_loss(current_q2, rewards + target_q.detach())
                    critic_optimizer_diff.step(critic_loss)
                    wandb.log({"Critic Loss": critic_loss})

                    actor_loss = -critic_diff.Q1(states, actor_diff(states)).mean()
                    actor_optimizer_diff.step(actor_loss)
                    wandb.log({"Actor Loss": actor_loss})

                # updated_actor_params = [p.clone() for p in actor_diff.parameters()]

        # for param, target_param in zip(actor_diff.parameters(), actor_target.parameters()):
        #     target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
                for param, target_param in zip(critic_diff.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)

                # 用updated_actor_params来计算IRL/BC损失
                # 将actor参数替换成updated后的参数，用于评估策略输出
                # original_actor_params = td3_agent.get_actor_params()
                # td3_agent.set_actor_params(updated_actor_params)

                # 计算BC损失(或IRL损失)
                idx = np.random.choice(len(expert_states), batch_size)
                expert_states_batch = torch.FloatTensor(expert_states[idx]).to(device)
                expert_actions_batch = torch.FloatTensor(expert_actions[idx]).to(device)

                pred_actions = actor_diff(expert_states_batch)
                bc_loss = ((pred_actions - expert_actions_batch)**2).mean()
                reward_optimizer.zero_grad()
                bc_loss.backward()
                reward_optimizer.step()
                wandb.log({"Discriminator Loss": bc_loss})

        # 对reward_net_diff求梯度并更新
        # reward_optimizer_diff.step(bc_loss)
        # wandb.log({"Discriminator Loss": bc_loss})

        # 恢复actor参数到original
        # td3_agent.set_actor_params(original_actor_params)