import gymnasium as gym
import panda_gym
from sb3_contrib import TQC
import numpy as np
import pickle
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from MyPush import MyPandaPushEnv


env_name = 'MyPandaPushEnv'
env = gym.make(env_name)
checkpoint = '/home/yuxuanli/failed_IRL_new/PandaRobot/PandaPush/model/MyPush_TQC_model.zip'
model = TQC("MultiInputPolicy", env, verbose=1, device="cuda")
expert = model.load(checkpoint, env)
vec_env = model.get_env()
demos = []

for i in range(500):
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
    

with open('/home/yuxuanli/failed_IRL_new/PandaRobot/PandaPush/model/success_500.pkl', 'wb') as f:
    pickle.dump(demos, f)