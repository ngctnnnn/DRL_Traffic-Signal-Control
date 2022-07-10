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
import math
from sumo_rl import SumoEnvironment


if __name__ == '__main__':
    ray.init()

    time_step = 80000
    runs = 1

    for run in range(runs):
        env = PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                        route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                        out_csv_name=f'outputs/4x4grid/dqn_last/dqn_last_{run}',
                                                        use_gui=False,
                                                        single_agent=False,
                                                        num_seconds=time_step,
                                                        sumo_seed = run))

        env_sumo = SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                            route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                            use_gui=False,
                            num_seconds=time_step,
                            min_green=5,
                            delta_time=5,
                            sumo_seed = run)

        register_env("4x4grid", lambda _ : PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                                        route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                                        out_csv_name=f'outputs/4x4grid/dqn_last/dqn_last_{run}',
                                                        use_gui=False,
                                                        single_agent=False,
                                                        num_seconds=time_step,
                                                        sumo_seed = run)))

        env.reset()
        env_sumo.reset()
        agent = DQNTrainer(env="4x4grid", config={
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
        })
        
        agent.restore(os.path.join(os.getcwd(), 'outputs/4x4grid/dqn_train/last_policy/DQNTrainer_2022-07-06_02-02-40/DQNTrainer_4x4grid_b7176_00000_0_2022-07-06_02-02-40/checkpoint_000001/checkpoint-1'))

        done = False
    
        obs = env_sumo.reset()
        while not done:
            action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = agent.config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id)
            
            obs, reward, done, info = env_sumo.step(action=action)
            done = done['__all__']
            print("run:", run)
            print(info)
        print('KET THUC')
