import gymnasium as gym
import panda_gym
from sb3_contrib import TQC
import numpy as np
import pickle
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


env_name = 'PandaPickAndPlace-v3'
env = gym.make(env_name)
checkpoint = '/home/yuxuanli/failed_IRL_new/PandaRobot/PandaPickAndPlace/model/PandaPickAndPlace-v3.zip'
model = TQC("MultiInputPolicy", env, verbose=1, device="cuda")
expert = model.load(checkpoint, env)
vec_env = model.get_env()
demos = []

for i in range(50):
    obs = vec_env.reset()
    done = False
    expert_demo = []
    while not (done):
        action, _state = expert.predict(obs, deterministic=True)
        action = np.squeeze(action)
        obs, reward, done, info = vec_env.step(action)
        # print(obs, reward, done, info )

        expert_demo.append({
            "state": obs,
            "action": action.copy(),
            "reward": reward,
            'truncated':None,
            "done": done,
            "info": info
            })
    
    demos.append(expert_demo)
    

with open('/home/yuxuanli/failed_IRL_new/PandaRobot/PandaPickAndPlace/model/success_50.pkl', 'wb') as f:
    pickle.dump(demos, f)