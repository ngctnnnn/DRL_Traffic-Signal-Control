from stable_baselines3 import PPO
# from sb3.PPO import PPO
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

    #env = sumo_rl.grid4x4(use_gui=True, out_csv_name='outputs/grid4x4/ppo_test', virtual_display=RESOLUTION)
    env = sumo_rl.ingolstadt1(use_gui=True, out_csv_name='outputs/ingostadt1/ppo_test', virtual_display=RESOLUTION)
    max_time = env.unwrapped.env.sim_max_time
    delta_time = env.unwrapped.env.delta_time

    print("Environment created")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=0, base_class='stable_baselines3')
    env = VecMonitor(env)
   
#
    #print(env.step())
#
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                gamma=0.95,
                n_steps=256,
                ent_coef=0.0905168,
                learning_rate=0.00062211,
                vf_coef=0.042202,
                max_grad_norm=0.9,
                gae_lambda=0.99,
                n_epochs=1,
                clip_range=0.3,
                batch_size=256,
                #tensorboard_log="./logs/grid4x4/ppo_test",)
   		tensorboard_log=None)
#   class TensorboardCallback(BaseCallback):
# 	def __init__(self, verbose=0)
#	    super(TensorboardCallback, self).__init__(verbose)
#	def _on_step(self) -> bool:
#	    value =

    print("Starting training")
    model.learn(total_timesteps=1)

    print("Training finished. Starting evaluation")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

    print(mean_reward)
    print(std_reward)

    # Maximum number of steps before reset, +1 because I'm scared of OBOE
    print("Starting rendering")
    num_steps = (max_time // delta_time) + 1

    obs = env.reset()

    if os.path.exists("temp"):
        shutil.rmtree("temp")

    os.mkdir("temp")
    # img = disp.grab()
    # img.save(f"temp/img0.jpg")
    print(type(env))
    assert 1 == 2
    #img = env.render()
    for t in range(num_steps):
        actions, _ = model.predict(obs, state=None, deterministic=False)
        obs, reward, done, info = env.step(actions)
        print(f"Action: {actions}\nObs: {obs}\nReward: {reward}\nInfo: {info}\n")
        #img = env.render()
        #img.save(f"temp/img{t}.jpg")
    
    #subprocess.run(["ffmpeg", "-y", "-framerate", "5", "-i", "temp/img%d.jpg", "output.mp4"])

    print("All done, cleaning up")
    shutil.rmtree("temp")
    env.close()
