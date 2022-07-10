import os, sys, sumo_rl, traci 
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import pandas as pd, numpy as np 
import sumo_rl
from sumo_rl.util.utils import *

import torch 
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.trainer import (
    offpolicy_trainer
)
from tianshou.env import (
    PettingZooEnv
)
from tianshou.data import (
    Collector,
    ReplayBuffer
)
from tianshou.env import (
    DummyVectorEnv
)
from tianshou.policy import (
    RandomPolicy,
    MultiAgentPolicyManager,
    DQNPolicy
) 
from tianshou.utils import (
    TensorboardLogger, 
    WandbLogger
)
 
"""
DNN for DQN
Double DQN implementation -> is_double = True
Dueling DQN implementation -> change in the Network itself
"""
class Net(nn.Module):
    """
    Target network for DQN
    """
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    @override(nn.Module)
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


"""
Constants and parameters
"""
out_csv = 'outputs/grid4x4/tianshou-policy/dqn_train'
total_step = 80000
# global policy_path
policy_path = os.path.join(os.getcwd(), 'outputs/grid4x4/tianshou-policy/dqn/')

"""
Environment settings
"""
env = PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                out_csv_name=out_csv,
                                use_gui=True,
                                single_agent=False,
                                num_seconds=int(total_step),
                                sumo_seed=3))
# obs = env.reset()

"""
Network configuration
"""
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(
            state_shape, 
            action_shape
)
optim = torch.optim.Adam(
                        net.parameters(), 
                        lr=1e-3
)


"""
Policy configuration
"""
dqn_policy = DQNPolicy(
                net, 
                optim, 
                discount_factor=0.99, # gamma = 0.99
                is_double=False,      
)
dqn_policy.set_eps(0.7) # epsilon = 0.7

def save_fn(policy):
    """
    Save policy function
    """
    torch.save(policy.state_dict(), os.path.join(policy_path, 'policy.pth'))

def save_checkpoint_fn(epoch, env_step, gradient_step):
    """
    Save checkpoint function
    """
    policy_path = os.path.join(os.getcwd(), 'outputs/grid4x4/tianshou-policy/dqn/')
    torch.save({
        'model': policy.state_dict(),
        'optim': optim.state_dict(),
    }, os.path.join(policy_path, f"checkpoint_{epoch}.pth"))  

"""
Multi-agent manager
"""
policy = MultiAgentPolicyManager([dqn_policy for _ in range(16)], env)
env = DummyVectorEnv([lambda: env])
buffer = ReplayBuffer(size=int(1e5))

"""
Tianshou collector
"""
collector = Collector(policy, env, buffer)
#result = collector.collect(
#                            n_episode=1, 
#                            no_grad=True,
#)
#print(result)
#assert 1==2,'End'
"""
Config save data
"""
seed = 0
log_name = os.path.join('grid4x4', 'dqn', str(seed))
log_path = os.path.join(policy_path, log_name)


"""
Config wandb
"""
logger = WandbLogger(
    save_interval=1,
    name=log_name.replace(os.path.sep, "__"),
    run_id=None,
    # config=args,
    project="tianshou-dqn-4x4",
)
writer = SummaryWriter(log_path)
logger.load(writer)

"""
Off-policy trainer
"""
result = offpolicy_trainer(
    policy, 
    train_collector = collector,
    test_collector = None,
    max_epoch = 0,
    step_per_epoch = int(total_step*3.216),  
    step_per_collect = 1,
    episode_per_test = 0,
    update_per_step = 0.1,
    batch_size = 64,
    logger=logger,
    # stop_fn = stop_fn,
    save_fn = save_fn,
    # save_checkpoint_fn=save_checkpoint_fn,
)
print(result)
# ### Try to finish offpolicy trainer & save - load policy & save results
