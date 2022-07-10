import gym
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from sumo_rl import SumoEnvironment
import traci

from stable_baselines3 import DQN

env = SumoEnvironment(net_file='nets/big-intersection/big-intersection.net.xml',
                        single_agent=True,
                        route_file='nets/big-intersection/routes.rou.xml',
                        out_csv_name='outputs/big-intersection/dqn',
                        use_gui=False,
                        num_seconds=100000,
                        min_green=5)

model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=1e-3,
    buffer_size=50000,
    exploration_fraction=0.05,
    exploration_final_eps=0.02
)
model.learn(total_timesteps=100000)
