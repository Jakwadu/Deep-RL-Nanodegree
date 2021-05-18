import torch
import torch.nn as nn
import torch.nn.functional as F

class StateBasedQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(StateBasedQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x_in):
        x = F.relu(self.fc1(x_in))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
class PixelBasedQNetwork(nn.Module):
    def __init__(self, input_shape=(84,84,3)):
        super(PixelBasedQNetwork, self).__init__()
        ...
        
    def forward(self, x_in):
        ...