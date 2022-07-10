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
                                sumo_seed=10))
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
                        lr=5e-3
)


"""
Policy configuration
"""
from typing import Any, Dict, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as

decay = 0.99
min_eps = 0.02
class CustomDQNPolicy(DQNPolicy):
    def __init__(self, 
                 epsilon, 
                 net,
                 optim,
                 discount_factor,
                 target_update_freq,
                 is_double):
        super().__init__(net, optim, discount_factor, target_update_freq, is_double)
        self.set_eps(epsilon)
    
    @override(DQNPolicy)
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        print("Forward--------------------------------")
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=hidden)
    
    @override(DQNPolicy)
    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.iesclose(self.eps, 0.0):
            bsz = len(act)
            print(self.eps); assert 1==2
            rand_mask = np.random.rand(bsz) < self.eps
            if self.eps * decay >= min_eps:
                self.eps *= decay
            else:
                self.eps = min_eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act

custom_dqn_policy = CustomDQNPolicy(
                epsilon=1.0,
                net=net, 
                optim=optim, 
                discount_factor=0.9, # gamma
                target_update_freq=500,
                is_double=False,      
)
# print(custom_dqn_policy.__dict__)
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
        'model': mapolicy.state_dict(),
        'optim': optim.state_dict(),
    }, os.path.join(policy_path, f"checkpoint_{epoch}.pth"))  

"""
Multi-agent manager
"""
mapolicy = MultiAgentPolicyManager([custom_dqn_policy for _ in range(16)], env)
env = DummyVectorEnv([lambda: env])
buffer = ReplayBuffer(size=int(5e4))

"""
Tianshou collector
"""
collector = Collector(mapolicy, 
                      env, 
                      buffer)
collector.collect(n_step=1000, random=True)
# buffer.reset()
# collector.reset()
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


# """
# Config wandb
# """
# logger = WandbLogger(
#     save_interval=1,
#     name=log_name.replace(os.path.sep, "__"),
#     run_id=None,
#     # config=args,
#     project="tianshou-dqn-4x4",
# )
# writer = SummaryWriter(log_path)
# logger.load(writer)

"""
Off-policy trainer
"""
# print(mapolicy.__dict__); assert 1==2

result = offpolicy_trainer(
    mapolicy, 
    train_collector = collector,
    test_collector = None,
    max_epoch = 1,
    step_per_epoch = int(total_step*3.216),
    step_per_collect = 5,
    update_per_step = 1, 
    batch_size = 32,
    episode_per_test = 0,
    save_fn = save_fn,
    # logger=logger,
    # stop_fn = stop_fn, 
    # save_checkpoint_fn=save_checkpoint_fn,
)
print(result)
# ### Try to finish offpolicy trainer & save - load policy & save results
