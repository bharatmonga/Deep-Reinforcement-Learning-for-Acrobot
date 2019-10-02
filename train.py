"""  Created by Bharat Monga  """

import time
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)  # set random seed
import torch.optim as optim
from acrobot import Acrobot
from policy import Policy

m1, m2 = 1.0, 1.0  # mass of the links of the acrobot
l1, l2 = 1.0, 1.0  # length of the links of the acrobot

ac = Acrobot(m1, l1, m2, l2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.03)


def reinforce(n_episodes=61, max_t=2000, print_every=10):
    """
    :param n_episodes: number of training episodes
    :param max_t: time steps for each episode
    :param print_every: print value and save policy next work parameters every print_every episodes
    :return: Value at time 0 for all the episodes
    """
    Value = []
    for i_episode in range(n_episodes):
        log_probability = []
        rewards = []
        state = ac.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            log_probability.append(log_prob)
            state, reward = ac.step(action)
            rewards.append(reward)

        discounted_rewards = []
        value_t = 0

        for t in range(len(rewards)-1, -1, -1):  # for loop backward in time for calculating value at every time step
            value_t = ac.gama*value_t + rewards[t]
            discounted_rewards.append(value_t)
        discounted_rewards.reverse()  # flip the list for forward time from 0 to max_t
        discounted_rewards = torch.tensor(discounted_rewards)
        Value.append(discounted_rewards[0])
        #

        policy_gradient = []
        for log_prob, value_t in zip(log_probability, discounted_rewards):
            policy_gradient.append(-log_prob * value_t)
        policy_gradient = torch.stack(policy_gradient).sum()

        optimizer.zero_grad()
        policy_gradient.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            torch.save({'l1': l1, 'm1': m1, 'l2': l2, 'm2': m2, 'policy_state_dic': policy.state_dict()}, 'checkpoint.pth')
            print('Episode {}\tValue: {:.2f}'.format(i_episode, Value[i_episode]))
    return Value

tt = time.time()
value = reinforce()

print(time.time()-tt)
plt.plot(value, linewidth=3, label="Value(t=0) vs episode #")
plt.grid()
plt.legend(fontsize=20, loc=0)
plt.savefig('value_vs_episode#.png', dpi=300)
plt.show()
