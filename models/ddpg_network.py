import torch
import os
from torch import nn

class CriticNetwork(nn.Module):
    def __init__(self, obs_space, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir='tmp/ddpg') -> None:
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg')

        self.fc1 = nn.Linear(obs_space, self.fc1_dims)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

    def forward(self, state, action):
        print(state.shape)
        print(action.shape)
        x = self.relu(self.fc1(torch.concat(state, action, dim=1)))
        x = self.relu(self.fc2(x))
        q = self.q(x)
        return q
    
class ActorNetwork(nn.Module):
    def __init__(self, obs_space, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor', chkpt_dir='tmp/ddpg') -> None:
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg')

        self.fc1 = nn.Linear(obs_space, self.fc1_dims)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.tanh = nn.Tanh()
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

    def forward(self, state):
        prob = self.relu(self.fc1(state))
        prob = self.relu(self.fc2(prob))

        # If the actikn not bound +/- 1, we can modify it here
        mu = self.tanh(self.mu(prob))
        return mu

