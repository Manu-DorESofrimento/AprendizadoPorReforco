#Mesmo esquema do treina, mas reforca o treinamento ao inves de treinar do 0
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import gym
import os

models_dir = "models/PPO"

env = make_vec_env('LunarLander-v2', n_envs=8)

eval_callback = EvalCallback(env, best_model_save_path=f'{models_dir}/best/', eval_freq=500, deterministic=True, render=False)
#Aqui muda, carrega o melhor modelo e seta o environment pra poder aprender "mais"
model = PPO.load(f"{models_dir}/best/best_model")
model.set_env(env)
model.learn(total_timesteps=200000, reset_num_timesteps=True, tb_log_name="PPO")


model.save(f"{models_dir}/best-model")




