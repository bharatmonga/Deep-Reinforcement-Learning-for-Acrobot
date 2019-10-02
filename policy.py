"""  Created by Bharat Monga  """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

torch.manual_seed(0)  # set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):

    def __init__(self, s_size=4, h_size1=12, h_size2=24, a_size=3):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size1)
        self.fc2 = nn.Linear(h_size1, h_size2)
        self.fc3 = nn.Linear(h_size2, h_size2)
        self.fc4 = nn.Linear(h_size2, h_size1)
        self.fc5 = nn.Linear(h_size1, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)