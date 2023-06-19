#Imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
import os

#Faz o environment
models_dir = "models/PPO"
env = make_vec_env('LunarLander-v2', n_envs=1)
#Carrega o melhor modelo
model = PPO.load(f"{models_dir}/best/best_model")
#Roda
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")




