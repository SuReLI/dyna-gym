"""
Non Stationary Chain Walk
"""

import logging
import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gym import error, spaces, utils
from gym.utils import seeding

logger = logging.getLogger(__name__)

class NSChain(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.n_states = 5
        self.tau = 1

        # Dynamic parameters
        self.r_evolution_speed = 1
        self.r_center = 0 # Initial center
        self.r_variance = self.n_states / 6
        self.r_period = 10 * self.tau

        self.action_space = spaces.Discrete(3) # left, noop, right

        self._seed()
        self.viewer = None
        self.state = self.initial_state()
        self.steps_beyond_done = None

        # rendering
        '''
        self.fig = plt.gcf()
        self.fig.show()
        gs1 = gridspec.GridSpec(2, 1)
        self.ax1 = self.fig.add_subplot(gs1[0])
        self.ax2 = self.fig.add_subplot(gs1[1])
        '''
        '''
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.fig.show()
        '''

    def initial_state(self):
        return (int(self.n_states / 2),0)

    def equality_operator(self, s1, s2):
        '''
        Equality operator, return True if the two input states are equal.
        '''
        return (s1 == s2)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self, pos, time):
        self.r_center = (self.n_states / 2) * (1 + math.sin(2 * math.pi * time / self.r_period))
        return math.exp(- (pos - self.r_center)**2 / self.r_variance)

    def transition(self, state, action, is_model_dynamic):
        '''
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        '''
        position_p, time_p = self.state
        if action == 0:
            if position_p > 0:
                position_p -= 1
        elif action == 2:
            if position_p < self.n_states:
                position_p += 1
        if is_model_dynamic:
            time_p += self.tau
        state_p = (int(position_p),time_p)
        # Termination criterion
        done = False
        # Reward
        if not done:
            reward = self.reward(position_p, time_p)
        else:
            reward = 0.0
        return state_p, reward, done

    def step(self, action):
        '''
        Step function equivalent to transition and reward function.
        Actually modifies the environment's state attribute.
        Return (observation, reward, termination criterion (boolean), informations)
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.state, reward, done = self.transition(self.state, action, True)
        return self.state, reward, done, {}

    def print_state(self):
        print('pos: {}; t: {}'.format(self.state[0],self.state[1]))

    def reset(self):
        self.state = self.initial_state()
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human', close=False):
        '''
        rwds = []
        for i in range(self.n_states + 1):
            rwds.append(self.reward(i, self.state[1]))
        self.ax1.clear()
        self.ax1.set_xlim([0, self.n_states])
        self.ax1.set_ylim([0, 1])
        self.ax1.plot(rwds)

        self.ax2.clear()
        self.ax2.set_xlim([0, self.n_states])
        self.ax2.set_ylim([-0.1, 0.1])
        self.ax2.scatter(self.state[0], 0, c='r')

        self.fig.canvas.draw()
        '''
        #plt.pause(1.0)
