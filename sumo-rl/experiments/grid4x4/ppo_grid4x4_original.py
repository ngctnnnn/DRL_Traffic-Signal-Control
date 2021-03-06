
from stable_baselines3 import PPO
import sumo_rl
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from pyvirtualdisplay.smartdisplay import SmartDisplay
import os
import subprocess
from tqdm import trange
import shutil
import traci

if __name__ == '__main__':

    RESOLUTION = (3200, 1800)

    env = sumo_rl.grid4x4(use_gui=True, out_csv_name='outputs/grid4x4/ppo_test', virtual_display=RESOLUTION)

    max_time = env.unwrapped.env.sim_max_time
    delta_time = env.unwrapped.env.delta_time

    print("Environment created")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)

    model = PPO("MlpPolicy",
                env,
                verbose=3,
                gamma=0.95,
                n_steps=256,
                ent_coef=0.0905168,
                learning_rate=0.00062211,
                vf_coef=0.042202,
                max_grad_norm=0.9,
                gae_lambda=0.99,
                n_epochs=5,
                clip_range=0.3,
                batch_size=256,
                tensorboard_log="./logs/grid4x4/ppo_test",)

    print("Starting training")
    model.learn(total_timesteps=80000)

    print("Training finished. Starting evaluation")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(mean_reward)
    print(std_reward)

