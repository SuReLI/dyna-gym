"""
Non-Stationary Randomly generated MDP
"""

import logging
import math
import numpy as np
import dyna_gym.utils.distribution as distribution
from gym import Env, error, spaces, utils

logger = logging.getLogger(__name__)

class State:
    """
    State class
    """
    def __init__(self, index, time):
        self.index = index
        self.time = time

class RandomNSMDP(Env):
    def __init__(self):
        self.nS = 3 # n states
        self.nTS = 3 # n terminal states
        self.nA = 2 # n actions
        self.nT = 100 # n timesteps
        self.pos_space = np.array(range(self.nS))
        self.action_space = spaces.Discrete(self.nA)

        self.timestep = 1 # timestep duration
        self.L_p = 1 # transition kernel Lipschitz constant
        self.L_r = 10 # reward function Lipschitz constant

        self.T = self.generate_transition_matrix()
        self.R = self.generate_reward_matrix()

        self._seed()
        self.reset()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = (0, 0)
        self.steps_beyond_done = None

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)
        for i in range(self.nS): # s
            for j in range(self.nA): # a
                # 1. Generate distribution for t=0
                T[i,j,0,:] = distribution.random_tabular(size=self.nS)
                # 2. Build subsequent distributions st LC constraint is respected
                for t in range(1, self.nT): # t
                    T[i,j,t,:] = distribution.random_constrained(T[i,j,t-1,:], self.L_p * self.timestep)
        return T

    def transition_probability_distribution(self, s, t, a):
        return self.T[s.index, a, t]

    def transition_probability(self, s_p, s, t, a):
        return self.T[s.index, a, t, s_p.index]

    def generate_reward_matrix(self):
        R = np.zeros(shape=(self.nS, self.nA, self.nT), dtype=float)
        for i in range(self.nS): # s
            for j in range(self.nA): # a
                # 1. Generate instant reward for t=0
                R[i,j,0] = np.random.random(size=None)
                # 2. Build subsequent instant rewards st LC constraint is respected
                for t in range(1, self.nT): # t
                    R[i,j,t] = R[i,j,t-1] + self.L_r * self.timestep * (2 * np.random.random(size=None) - 1)
                    if R[i,j,t] > 1:
                        R[i,j,t] = 1
                    if R[i,j,t] < 0:
                        R[i,j,t] = 0
        return R

    def reward(self, s, t, a):
        return self.R[s.index, a, t]

    def equality_operator(self, s1, s2):
        """
        Return True if the input states have the same indexes.
        """
        return (s1.index == s2.index)

    def transition(self, state, action, is_model_dynamic):
        """
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        """
        p_p, t = self.state
        reward = self.reward(p_p, t, action)
        transition_model = self.transition_probability_distribution(p_p, t, action)
        p_p = np.random.choice(self.pos_space, size=None, replace=False, p=transition_model)
        if is_model_dynamic:
            t += self.timestep
        state_p = (int(p_p), t)
        if t >= self.nT - 1: # Timeout
            done = True
        else:
            done = False
        return state_p, reward, done

    def step(self, action):
        """
        Step function equivalent to transition and reward function.
        Actually modifies the environment's state attribute.
        Return (observation, reward, termination criterion (boolean), informations)
        """
        self.state, reward, done = self.transition(self.state, action, True)
        return (self.state, reward, done, {})

    def is_terminal(self, state):
        return False

    def print_state(self):
        print('position: {}; t: {}'.format(self.state[0],self.state[1]))

    def get_state_space_at_time(self, t):
        return [(x, t) for x in self.pos_space]

    def get_time(self):
        return self.state[1]

    def render(self, mode='human', close=False):
        """
        No rendering yet
        """
        return None
