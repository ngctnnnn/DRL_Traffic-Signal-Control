import math, numpy as np
import argparse
import os
import sys
import pandas as pd
import pickle

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == '__main__':

    alpha = 0.005
    gamma = 0.95
    decay = 0.99
    runs = 10

    # with open('outputs/grid4x4/ql_random_policy.pkl','wb') as f:
    #     pickle.dump(ql_agents, f)

    for run in range(1, runs + 1):
        env = SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                            route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                            use_gui=False,
                            num_seconds=80000,
                            min_green=5,
                            delta_time=5,
                            sumo_seed = int(run-1))

        initial_states = env.reset()
        infos = []
        done = {'__all__': False}
        agent_id = [str(_) for _ in range(0, 16)]
        while not done['__all__']:
            actions = {ts: math.floor(np.random.randint(0, 2)) for ts in agent_id}
            print(actions)    
            s, r, done, info = env.step(action=actions)
            print(f"Run: {run}")
            print(f"Info: {info}")
#            for agent_id in s.keys():
#                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
        env.save_csv('outputs/grid4x4/random_policy', run)
        env.close()


