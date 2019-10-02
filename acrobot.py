"""  Created by Bharat Monga  """

import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Acrobot:
    """
    Create an acrobot object by specifying it's link masses and link lengths as inputs
    at initialization of the object
    """

    def __init__(self, link_mass_1, link_length_1, link_mass_2, link_length_2):
        self.LINK_MASS_1 = link_mass_1  # mass of link 1 in [kg]
        self.LINK_LENGTH_1 = link_length_1  #: length of link 1 in [m]
        self.LINK_MOI_1 = (1 / 12) * self.LINK_MASS_1 * self.LINK_LENGTH_1 ** 2  #: moment of inertia of link 1
        self.LINK_MASS_2 = link_mass_2  # mass of link 2 in [kg]
        self.LINK_LENGTH_2 = link_length_2  #: length of link 2 in [m]
        self.LINK_MOI_2 = (1 / 12) * self.LINK_MASS_2 * self.LINK_LENGTH_2 ** 2  #: moment of inertia of link 2
        self.torque = 0  #: initialize torque at start as 0
        self.action_space = [-20.0, 0.0, 20.0]
        self.gama = 0.999  # discount factor
        self.dt = 0.001  #: time step for the Runge-Kutta solver
        self.seed = np.random.seed(0)
        self.state = None

    def reset(self):
        """
        :return: new state of the acrobot object
        """
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))
        return self.state

    def step(self, index):
        """
        takes input the index of the action space
        calculates next state by calling 4th order Runge-Kutta solver
        returns state and reward at the next time step
        """
        s = self.state
        self.torque = self.action_space[index]
        new_s = self.rk4(s)
        self.state = new_s
        height = -self.LINK_LENGTH_1 * cos(self.state[0]) - self.LINK_LENGTH_2 * cos(self.state[1] + self.state[0])
        return self.state, height

    def dsdt(self, s):
        """
        :param s: current state of the acrobot
        :return: right hand side of acrobot ODEs
        """
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        l2 = self.LINK_LENGTH_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        g = 9.8
        a = self.torque
        ds1 = s[2]
        ds2 = s[3]
        D11 = 0.25 * m1 * l1 ** 2 + I1 + I2 + 0.25 * m2 * (4 * l1 ** 2 + l2 ** 2 + 4 * l1 * l2 * cos(s[1]))
        D12 = I2 + 0.25 * m2 * (l2 ** 2 + 2 * l1 * l2 * cos(s[1]))
        c = -0.5 * (m2 * l1 * l2 * sin(s[1]))
        phi1 = sin(s[0]) * (0.5 * g * l1 * (m1 + 2 * m2)) + sin(s[0] + s[1]) * (0.5 * m2 * g * l2)
        D22 = (0.25 * m2 * l2 ** 2 + I2)
        phi2 = sin(s[0] + s[1]) * (0.5 * m2 * g * l2)
        d = D11 * D22 - D12 ** 2
        d11 = D11 / d
        d22 = D22 / d
        d12 = D12 / d
        ds3 = -c * d12 * s[2] ** 2 - 2 * c * d22 * s[2] * s[3] - c * d22 * s[3] ** 2 + d12 * (phi2 - a) - d22 * phi1
        ds4 = c * d11 * s[2] ** 2 + 2 * c * d12 * s[2] * s[3] + c * d12 * s[3] ** 2 + d12 * phi1 - d11 * (phi2 - a)
        return ds1, ds2, ds3, ds4


    def rk4(self, y0):
        """
        :param y0: current state of the acrobot object
        :return: state y of the acrobot object at next time step using 4th order Runge-Kutta method
        """
        h = self.dt
        f = self.dsdt
        k1 = h * np.asarray(f(y0))
        k2 = h * np.asarray(f(y0 + k1 / 2))
        k3 = h * np.asarray(f(y0 + k2 / 2))
        k4 = h * np.asarray(f(y0 + k3))
        y = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def render(self, x, u, t):
        """
        :param x: state history/trajectory
        :param u: motor torque history
        :param t: time vector
        :return: save animation
        """
        fig = plt.figure(figsize=(5, 3), dpi=300)
        ax = fig.add_subplot(121, autoscale_on=False, xlim=(-1.1 * (self.LINK_LENGTH_1+self.LINK_LENGTH_2),
                                                            1.1 * (self.LINK_LENGTH_1+self.LINK_LENGTH_2)),
                             ylim=(-1.1 * (self.LINK_LENGTH_1+self.LINK_LENGTH_2),
                                   1.1 * (self.LINK_LENGTH_1+self.LINK_LENGTH_2)))
        ax1 = fig.add_subplot(122, autoscale_on=False, xlim=(-1.1 * (self.LINK_LENGTH_1+self.LINK_LENGTH_2),
                                                             1.1 * (self.LINK_LENGTH_1+self.LINK_LENGTH_2)),
                              ylim=(-1.1 * (self.LINK_LENGTH_1+self.LINK_LENGTH_2), 1.1 * (self.LINK_LENGTH_1+self.LINK_LENGTH_2)))
        ax.set_aspect('equal')
        ax.axis('off')
        ax1.axis('off')
        line, = ax.plot([], [], color='dodgerblue', lw=8, solid_joinstyle='round', solid_capstyle='round')
        line2, = ax.plot([], [], 'ro', markersize=6)
        line3, = ax1.plot([], [], color='blue', lw=1, solid_joinstyle='round', solid_capstyle='round')
        ax.text(1.6, 1.05, 'Motor Torque', transform=ax.transAxes)
        i = np.arange(0, t.shape[0], 1)

        def animate(i):
            sx = self.LINK_LENGTH_1 * sin(x[i, 0])
            cx = -self.LINK_LENGTH_1 * cos(x[i, 0])
            sx1 = sx + self.LINK_LENGTH_2 * sin(x[i, 0] + x[i, 1])
            cx1 = cx - self.LINK_LENGTH_2 * cos(x[i, 0] + x[i, 1])
            line.set_data([0, sx, sx1], [0, cx, cx1])
            line2.set_data(sx, cx)
            line3.set_data(-2-10*t[:i, 0] + 10*t[i, 0], u[:i, 0] / 15)
            return line, line2, line3

        ani = animation.FuncAnimation(fig, animate, frames=i, interval=1, blit=True, repeat=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=60000)
        ani.save('acrobot_deep_reinforce.mp4', writer=writer)

        fig.clear()
        plt.close(fig)
