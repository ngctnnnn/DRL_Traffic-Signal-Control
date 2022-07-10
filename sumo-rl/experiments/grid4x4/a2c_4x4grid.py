import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
from sumo_rl.util.gen_route import write_route_file
import traci

#from stable_baselines3.common.vec_env import VecMonitor
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, DQN


if __name__ == '__main__':

    #write_route_file('nets/single-intersection/single-intersection-gen.rou.xml', 400000, 100000)

    env = DummyVecEnv([lambda: SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                                        route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                        out_csv_name='outputs/grid4x4/a2c',
                                        single_agent=True,
                                        use_gui=False,
                                        num_seconds=100000,
                                        min_green=5)])
    model = A2C("MlpPolicy", 
				env, 
				verbose=1, 
				learning_rate=0.001)
    model.learn(total_timesteps=100000)
    model.save('a2c-grid4x4.pkl')
