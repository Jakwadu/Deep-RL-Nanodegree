import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np

DEFAULT_HIDDEN = [64, 64]


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, layer_nodes=None):
        super(ActorCriticNetwork, self).__init__()

        if layer_nodes is None:
            layer_nodes = DEFAULT_HIDDEN
        is_list = isinstance(layer_nodes, list)
        has_ints = all([isinstance(n_nodes, int) for n_nodes in layer_nodes])
        assert (is_list and has_ints), 'layer_nodes must be list of ints'
        assert len(layer_nodes) > 0, 'No layer parameters provided'

        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.bn = nn.BatchNorm1d(state_size)

        layers = []
        n_hidden_nodes = len(layer_nodes)
        for i in range(n_hidden_nodes):
            if i == 0:
                layers += [nn.Linear(state_size, layer_nodes[i]), nn.ReLU()]
            else:
                layers += [nn.Linear(layer_nodes[i-1], layer_nodes[i]), nn.ReLU()]
        self.features = nn.Sequential(*layers)

        self.mu = nn.Linear(layer_nodes[-1], self.action_size)
        self.sigma = nn.Parameter(torch.ones((1, action_size)))
        self.value = nn.Linear(layer_nodes[-1], 1)

    def forward(self, x_in):
        bn_state = self.bn(x_in)
        features = self.features(bn_state)
        mu = torch.tanh(self.mu(features))
        sigma = F.softplus(self.sigma)
        value = self.value(features)
        return Normal(mu, sigma), value

#############################################################


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed=0, fc1_units=128, fc2_units=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed=0, fcs1_units=128, fc2_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        xs = F.relu(self.fcs1(state))
        xs = self.bn1(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
