import os, sys, sumo_rl, traci 
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import pandas as pd, numpy as np 
from tianshou.env import PettingZooEnv
import sumo_rl
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy, MultiAgentPolicyManager
 

out_csv = 'outputs/grid4x4/tianshou-policy/random-policy'
total_step = 1000

env = PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                out_csv_name=out_csv,
                                use_gui=False,
                                single_agent=False,
                                num_seconds=int(total_step)))

obs = env.reset()

policy = MultiAgentPolicyManager([RandomPolicy() for _ in range(16)], env)
env = DummyVectorEnv([lambda: env])
collector = Collector(policy, env)
result = collector.collect(n_episode=1)
print(result)
