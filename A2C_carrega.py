#Imports
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import gym
import os

#Faz o environment
models_dir = "models/A2C"
env = make_vec_env('LunarLander-v2', n_envs=1)
#Carrega o melhor modelo
model = A2C.load(f"{models_dir}/best/best_model")
#Roda
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")




