import gymnasium as gym
import panda_gym
from sb3_contrib import TQC
import numpy as np
import pickle
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


env_name = 'PandaReach-v3'
env = gym.make(env_name)
checkpoint = '/home/yuxuanli/failed_IRL_new/logs/tqc/PandaReach-v3_1/PandaReach-v3.zip'
model = TQC("MultiInputPolicy", env, verbose=1, device="cuda")
expert = model.load(checkpoint, env)


vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = expert.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)