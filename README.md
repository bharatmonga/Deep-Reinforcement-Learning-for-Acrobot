# Deep-Reinforcement-Learning-for-Acrobot

This repository contains the code to implement deep reinforcement learning algorithm REINFORCE to swing up an acrobot.

Acrobot is a 2 rigid link system actuated at the elbow joint. It is shown below. The goal of this project is to design a control stimulus (motor torque) which swings up the acrobot from the downward position (shown on the left) to the upward position (shown on the right).
<img src="https://github.com/bharatmonga/Deep-Reinforcement-Learning-for-Acrobot/blob/master/pics/acrobot.PNG" width="300">

REINFORCE is a Monte-Carlo variant of policy gradient methods where an agent learns the policy (mapping from state space to action space) by maximizing the value function at the start of the trajectory. I used a 5 fully connected layer neural network as the policy network. Reinforce algorithm optimizes the parameters of this network to obtain an optimal polcy that can swing up the acrobot. 

[policy.py](https://github.com/bharatmonga/Deep-Reinforcement-Learning-for-Acrobot/blob/master/policy.py) contains this 5 layer network.

[acrobot.py](https://github.com/bharatmonga/Deep-Reinforcement-Learning-for-Acrobot/blob/master/acrobot.py) is the environment that I created. It creates an acrobot object with attributes like reseting the state of the object, evolving the system one time step by a 4th order Runge-Kutta solver, and a render function which saves the animation of the trained policy in action.  
