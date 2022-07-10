import argparse
import os
import sys
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl.util.gen_route import write_route_file
from sumo_rl import SumoEnvironment
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda

if __name__ == '__main__':
    kwargs = {
        'alpha': 1e-4,
        'gamma': 0.95,
        'epsilon': 0.01,
        'min_green': 5,
        'max_green': 300,
        'gui': False,
        'fixed': False,
        'seconds': int(4e5),
        'runs': 1
    }
    
    out_csv = 'outputs/single-intersection/sarsa_lambda'
    env = SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                        route_file='nets/single-intersection/single-intersection.rou.xml',
                        out_csv_name='outputs/single-intersection/sarsa_lambda',
                        single_agent=True,
                        use_gui=kwargs['gui'],
                        num_seconds=kwargs['seconds'],
                        min_green=kwargs['min_green'],
                        max_green=kwargs['max_green'])
                        
    for run in range(1, kwargs['runs'] + 1):
        obs = env.reset()
        agent = TrueOnlineSarsaLambda(env.observation_space, env.action_space, alpha=kwargs['alpha'], gamma=kwargs['gamma'], epsilon=kwargs['epsilon'], fourier_order=7, lamb=0.95)
        
        done = False
        if kwargs['fixed']:
            while not done:
                _, _, done, _ = env.step({})
        else:
            while not done:
                action = agent.act(obs)

                next_obs, r, done, _ = env.step(action=action)

                agent.learn(state=obs, action=action, reward=r, next_state=next_obs, done=done)

                obs = next_obs

        env.save_csv(out_csv, run)