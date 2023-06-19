#Imports
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import gym
import os
#Pasta de modelos
models_dir = "models/A2C"
#Verifica se a pasta existe, se nao, cria
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
#Faz o environment
env = make_vec_env('LunarLander-v2', n_envs=8)
#Pra salvar o melhor modelo
eval_callback = EvalCallback(env, best_model_save_path=f'{models_dir}/best/', eval_freq=500, deterministic=True, render=False)
#Inicia o modelo e o treinamento
model = A2C('MlpPolicy', env, verbose=1, learning_rate=0.00081, gamma=0.995, n_steps=5, ent_coef=0.00001)
model.learn(total_timesteps=200000, reset_num_timesteps=True, tb_log_name="A2C", callback=eval_callback)
#Salva o modelo completo
model.save(f"{models_dir}/model")



