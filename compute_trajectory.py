"""  Created by Bharat Monga  """

import numpy as np
from acrobot import Acrobot
from policy import Policy
import torch
import matplotlib.pyplot as plt
from numpy import cos
torch.manual_seed(0) # set random seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parameters = torch.load('checkpoint.pth')
m1 = parameters['m1']
l1 = parameters['l1']
m2 = parameters['m2']
l2 = parameters['l2']

policy = Policy()
policy.load_state_dict(parameters['policy_state_dic'])  # load saved parameters to policy network
policy = policy.to(device)

N = 2000

ac = Acrobot(m1, l1, m2, l2)  # create acrobot object with saved parameters
ac.reset()

torque, t = np.zeros((N, 1), dtype=int), np.zeros((N, 1), dtype=float)
r = np.zeros((N, 1), dtype=float)
s = np.zeros((N, 4), dtype=float)
for i in range(N):  # generate a trajectory with the optimized policy network
    a, _ = policy.act(ac.state)
    s[i, :], r[i] = ac.step(a)
    torque[i], t[i] = ac.torque, i * ac.dt

height = -l1*cos(s[:, 0]) - l2*cos(s[:, 0] + s[:, 1])
plt.figure(1)
plt.plot(t, height, linewidth=3, label="height")
plt.legend(fontsize=20, loc='best')
plt.grid()
plt.savefig('acrobot_height.png', dpi=300)

plt.figure(2)
plt.plot(t, torque, 'k', linewidth=3, label="Motor torque")
plt.legend(fontsize=20, loc='best')
plt.grid()
plt.savefig('acrobot_torque.png', dpi=300)
ac.render(s, torque, t)  # save the generated trajectory as a cool animation
plt.show()
