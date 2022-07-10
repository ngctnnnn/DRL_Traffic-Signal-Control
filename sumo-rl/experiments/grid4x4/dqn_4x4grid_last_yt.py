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
from ray import tune


if __name__ == '__main__':
    ray.init()
    total_step = 1000
    out_csv= 'outputs/4x4grid/dqn'
    
    env = PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                    route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                    out_csv_name=out_csv,
                                    use_gui=False,
                                    single_agent=False,
                                    num_seconds=int(total_step)))
    
    register_env("4x4grid", lambda _ : PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                    route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                    out_csv_name=out_csv,
                                    use_gui=False,
                                    single_agent=False,
                                    num_seconds=int(total_step))))


    agent = DQNTrainer(env="4x4grid", config={
        "multiagent": {
            "policies": {
                id: (DQNTFPolicy, env.env.observation_spaces[id], env.env.action_spaces[id], {}) for id in env.env.agents
            },
            "policy_mapping_fn": (lambda id: id)  # Traffic lights are always controlled by this policy
        },
        "lr": 0.001,
        "gamma": 0.99,
        "no_done_at_end": True,
        # "rollout_fragment_length": 1,
        # "num_envs_per_worker": 1,
        "double_q": False,
        "dueling": False,
        "train_batch_size": 1,
        "num_workers": 1,
        "batch_mode": "complete_episodes",
    })
    folder_name = os.getcwd() 
    agent.restore(f"{os.path.join(folder_name, 'outputs/4x4grid/dqn/last_policy/DQNTrainer_2022-06-30_10-53-48/DQNTrainer_4x4grid_3f80d_00000_0_2022-06-30_10-53-49/checkpoint_000001/checkpoint-1')}")
    tune.run(agent, stop={
            "training_iteration": 0
        }, local_dir=f"{os.path.join(folder_name, 'outputs/4x4grid/dqn/last_policy/DQNTrainer_2022-06-30_10-53-48/DQNTrainer_4x4grid_3f80d_00000_0_2022-06-30_10-53-49/checkpoint_000001/')}")
