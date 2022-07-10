# import os, sys, sumo_rl, traci 
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")

# import pandas as pd, numpy as np 
# import sumo_rl
# from sumo_rl.util.utils import *

# import torch 
# from torch import nn
# from torch.utils.tensorboard import SummaryWriter

# from tianshou.trainer import (
#     offpolicy_trainer
# )
# from tianshou.env import (
#     PettingZooEnv
# )
# from tianshou.data import (
#     Collector,
#     ReplayBuffer
# )
# from tianshou.env import (
#     DummyVectorEnv
# )
# from tianshou.policy import (
#     RandomPolicy,
#     MultiAgentPolicyManager,
#     DQNPolicy
# ) 
# from tianshou.utils import (
#     TensorboardLogger, 
#     WandbLogger
# )
 
# """
# DNN for DQN
# Double DQN implementation -> is_double = True
# Dueling DQN implementation -> change in the Network itself
# """
# class Net(nn.Module):
#     """
#     Target network for DQN
#     """
#     def __init__(self, state_shape, action_shape):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(np.prod(state_shape), 128), 
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128), 
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 128), 
#             nn.ReLU(inplace=True),
#             nn.Linear(128, np.prod(action_shape)),
#         )

#     @override(nn.Module)
#     def forward(self, obs, state=None, info={}):
#         if not isinstance(obs, torch.Tensor):
#             obs = torch.tensor(obs, dtype=torch.float)
#         batch = obs.shape[0]
#         logits = self.model(obs.view(batch, -1))
#         return logits, state


# """
# Constants and parameters
# """
# out_csv = 'outputs/grid4x4/tianshou-policy/dqn_train'
# total_step = 80000
# # global policy_path
# policy_path = os.path.join(os.getcwd(), 'outputs/grid4x4/tianshou-policy/dqn/')

# """
# Environment settings
# """
# env = PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
#                                 route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
#                                 out_csv_name=out_csv,
#                                 use_gui=True,
#                                 single_agent=False,
#                                 num_seconds=int(total_step),
#                                 sumo_seed=8))
# # obs = env.reset()

# """
# Network configuration
# """
# state_shape = env.observation_space.shape or env.observation_space.n
# action_shape = env.action_space.shape or env.action_space.n
# net = Net(
#             state_shape, 
#             action_shape
# )
# optim = torch.optim.Adam(
#                         net.parameters(), 
#                         lr=1e-3
# )


# """
# Policy configuration
# """
# dqn_policy = DQNPolicy(
#                 net, 
#                 optim, 
#                 discount_factor=0.99, # gamma = 0.99
#                 is_double=False,      
# )
# dqn_policy.set_eps(0.7) # epsilon = 0.7

# def save_fn(policy):
#     """
#     Save policy function
#     """
#     torch.save(policy.state_dict(), os.path.join(policy_path, 'policy.pth'))


# def save_fn(policy):
#     """
#     Save policy function
#     """
#     torch.save(policy.state_dict(), os.path.join(policy_path, 'policy.pth'))

# def save_checkpoint_fn(epoch, env_step, gradient_step):
#     """
#     Save checkpoint function
#     """
#     policy_path = os.path.join(os.getcwd(), 'outputs/grid4x4/tianshou-policy/dqn/')
#     torch.save({
#         'model': policy.state_dict(),
#         'optim': optim.state_dict(),
#     }, os.path.join(policy_path, f"checkpoint_{epoch}.pth"))  

# """
# Multi-agent manager
# """
# policy = MultiAgentPolicyManager([dqn_policy for _ in range(16)], env)
# env = DummyVectorEnv([lambda: env])
# buffer = ReplayBuffer(size=int(1e5))

# """
# Tianshou collector
# """
# collector = Collector(policy, env)
# result = collector.collect(
#                            n_episode=1, 
#                            no_grad=True,
# )
# print(result)

import os, sys, sumo_rl, traci, pickle
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

    def save_fn(self, policy_path, model_idx):
        """
        Save policy function
        Args:
        policy_path -- path to save policy
        """
        torch.save(self.model.state_dict(), os.path.join(policy_path, f'policy_{model_idx}.pth'))

"""
Constants and parameters
"""
out_csv = 'outputs/grid4x4/tianshou-policy/dqn_train'
total_step = 200
# global policy_path
policy_path = os.path.join(os.getcwd(), 'outputs/grid4x4/tianshou-policy/dqn/')

"""
Environment settings
"""
env = PettingZooEnv(sumo_rl.env(net_file='nets/4x4-Lucas/4x4.net.xml',
                                route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                                out_csv_name=out_csv,
                                use_gui=not not not not True,
                                single_agent=False,
                                num_seconds=int(total_step),
                                sumo_seed=8))
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
print(dqn_policy.__dict__)

"""
Multi-agent manager
"""
mapolicy = MultiAgentPolicyManager([dqn_policy for _ in range(16)], env)
env = DummyVectorEnv([lambda: env])
buffer = ReplayBuffer(size=int(1e5))

"""
Tianshou collector
"""
print('Cho Dung')
# collector = Collector(mapolicy, env)
train_collector = Collector(mapolicy, 
                            env, 
                            ReplayBuffer(size=20000), 
                            exploration_noise=True)
# test_collector = Collector(mapolicy, 
#                            env, 
#                            exploration_noise=True)
# train_collector.collect(n_step=5, random=True)

print('Cho Dang')
for idx in range(16):
    mapolicy.policies[str(idx)].set_eps(0.1)


train_collector.reset()
env.reset()
# test_envs.reset()
buffer.reset()
for i in range(int(total_step)):  # total step
    collect_result = train_collector.collect(n_step=15)

    for idx in range(16):
        mapolicy.policies[str(idx)].set_eps(0.05)
        # result = test_collector.collect(n_episode=100)
    for idx in range(16):
        mapolicy.policies[str(idx)].set_eps(0.1)
    losses = mapolicy.update(64, train_collector.buffer, batch_size=32)
    train_collector.reset_buffer()
    print(f"Cho Dang")
    # print('Cho Dang cho Dang')

with open(f"{policy_path}/mapolicy.pkl", "wb") as f:
    pickle.dump(mapolicy, f)