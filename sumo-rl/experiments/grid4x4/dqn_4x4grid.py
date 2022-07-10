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

    runs = 10
    time_step = 80000 
    for run in range(runs):
        env = PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                        route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                        out_csv_name=f"outputs/4x4grid/dqn_train/dqn_train_{run}",
                                                        use_gui=False,
                                                        num_seconds=time_step,
                                                        sumo_seed = run))

        register_env("4x4grid", lambda _ : PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                        route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                        out_csv_name=f"outputs/4x4grid/dqn_train/dqn_train_{run}",
                                                        use_gui=False,
                                                        num_seconds=time_step,
                                                        sumo_seed = run)))
        print("run:", run)
        tune.run(DQNTrainer, config={
            "env": "4x4grid",
            "multiagent": {
                "policies": {
                    id: (DQNTFPolicy, env.env.observation_spaces[id], env.env.action_spaces[id], {}) for id in env.env.agents
                },
                "policy_mapping_fn": (lambda id: id)  # Traffic lights are always controlled by this policy
            },
            "lr": 0.0005,
            "num_envs_per_worker": 1,
            "train_batch_size": 1,
            "num_workers": 1,
            "batch_mode": "complete_episodes",
            "evaluation_num_episodes": 1,
            "dueling": False,
            "double_q": False,
            # "buffer_size": 50000,
            "replay_buffer_config": {
                # "_enable_replay_buffer_api": True,
                # "type": "MultiAgentReplayBuffer", 
                "capacity": 50000},
            "gamma": 0.9,
            "target_network_update_freq": 500,
            "exploration_config": {
                # Exploration sub-class by name or full path to module+class
                # (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
                "type": "EpsilonGreedy",
                # Parameters for the Exploration class' constructor:
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": time_step, 
                 # Timesteps over which to anneal epsilon.
            },

        }, local_dir='outputs/4x4grid/dqn_train/last_policy', checkpoint_at_end=True, 
            stop={"timesteps_total": time_step,  "episodes_total":1} )

        # trainer.train() # distributed training step
        # trainer.save('outputs/4x4grid/dqn')
        print('KET THUC')
