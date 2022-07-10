import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
import ray
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from gym import spaces
import numpy as np
import sumo_rl
import traci
from pprint import pprint as print

if __name__ == '__main__':
    ray.init()

    register_env("4x4grid", lambda _ : PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                    route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                    out_csv_name='outputs/4x4grid/dqn/dqn',
                                                    use_gui=False,
                                                    num_seconds=int(30))))
    config = {
         "multiagent": {
            "policies": {
                '0': (DQNTFPolicy, spaces.Box(low=np.zeros(11), high=np.ones(11)), spaces.Discrete(2), {})
            },
            "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
        },      

        "double_q": False,
        "dueling": False,
        "num_cpus_per_worker": 0,
        "num_cpus_for_driver": 0,
        "timesteps_per_iteration": 0,
        "gamma": 0.99, 
        "num_workers": 1,
	    "rollout_fragment_length": 1700,
        "lr": 0.0005,
        "n_step": 0,
        "num_gpus": 0,
        "no_done_at_end": False,
        "framework": "tf2.x",
        "num_atoms": 1,
        "batch_mode": "complete_episodes"
    }
    trainer = DQNTrainer(env="4x4grid", config=config)
    print(trainer.__dict__["config"])
    trainer.train()  # distributed training step
    print("Cho Dang")
