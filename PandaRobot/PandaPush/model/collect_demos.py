import gymnasium as gym
import panda_gym
from sb3_contrib import TQC
import numpy as np
import pickle
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from MyPush import MyPandaPushEnv
from MyPush import MyPandaPushEnv_2


env_name = 'MyPandaPushEnv_2'
env = gym.make(env_name)
checkpoint = 'PandaRobot/PandaPush/model/PandaPush_TQC_3_model.zip'
model = TQC("MultiInputPolicy", env, verbose=1, device="cuda")
expert = model.load(checkpoint, env)
vec_env = model.get_env()
demos = []

for i in range(5):
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
    

with open('/home/xlx9645/failed/PandaRobot/PandaPush/model/success_5_2.pkl', 'wb') as f:
    pickle.dump(demos, f)