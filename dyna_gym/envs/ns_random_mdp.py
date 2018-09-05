"""
Non-Stationary Randomly generated MDP
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

class NSRandomMDP(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.n_states = 2
        self.n_timestep = 3 # maximal number of timesteps
        self.timestep = 1 # timestep duration
        self.L_p = 1 # transition kernel Lipschitz constant
        self.L_r = 1 # reward function Lipschitz constant

        self.transition_matrix = self.generate_transition_matrix()

        self.action_space = spaces.Discrete(self.n_states) # each action corresponds to the position the agent wants to reach

        self._seed()
        self.viewer = None
        self.state = self.initial_state()
        self.steps_beyond_done = None

    def initial_state(self):
        '''
        Initial state is [position=0, time=0]
        '''
        return [0, 0]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_tabular_distribution(self, size):
        d = np.random.random(size)
        d = d / np.sum(d)
        return d

    def randomly_evolve_distribution(self, d, maxdist):
        '''
        Randomly generate a new distribution wrt previous distribution d
        st Wasserstein distance between both distributions is smaller than maxdist.
        '''
        #TODO
        return d + 1

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.n_states, self.n_states, self.n_timestep, self.n_states), dtype=float)
        for i in range(self.n_states): # s
            for j in range(self.n_states): # a
                # Generate distribution for t=0
                T[i,j,0,:] = self.random_tabular_distribution(size=self.n_states)
                for t in range(1, self.n_timestep): # t
                    # Build subsequent distributions st LC constraint is respected
                    T[i,j,t,:] = self.randomly_evolve_distribution(T[i,j,t-1,:], self.L_p * self.timestep)
        print(T)
        return T

    def equality_operator(self, s1, s2):
        '''
        Equality operator, return True if the two input states are equal.
        '''
        return (s1 == s2)

    def reward(self, state, action):
        #TODO
        return 0

    def transition(self, state, action, is_model_dynamic):
        '''
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        '''
        position_p, time_p = self.state
        # TODO assert that action is valid
        # TODO process position_p wrt action
        if is_model_dynamic:
            time_p += self.timestep
        state_p = (int(position_p), time_p)
        done = False # Termination criterion
        # TODO process Reward
        reward = 0
        return state_p, reward, done

    def step(self, action):
        '''
        Step function equivalent to transition and reward function.
        Actually modifies the environment's state attribute.
        Return (observation, reward, termination criterion (boolean), informations)
        '''
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
        No rendering yet
        '''
