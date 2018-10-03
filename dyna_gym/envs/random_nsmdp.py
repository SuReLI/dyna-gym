"""
Non-Stationary Randomly generated MDP
"""

import logging
import math
import numpy as np
import dyna_gym.utils.distribution as distribution
from gym import Env, error, spaces, utils

logger = logging.getLogger(__name__)

class RandomNSMDP(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.nS = 3 # n states
        self.nA = 2 # n actions
        self.nT = 100 # n timesteps
        self.pos_space = np.array(range(self.nS))
        self.action_space = spaces.Discrete(self.nA)

        self.timestep = 1 # timestep duration
        self.L_p = 1 # transition kernel Lipschitz constant
        self.L_r = 10 # reward function Lipschitz constant

        self.T = self.generate_transition_matrix()
        self.R = self.generate_reward_matrix()

        #self._seed()
        self.viewer = None
        self.state = self.initial_state()
        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def initial_state(self):
        '''
        Initial state is [position=0, time=0]
        '''
        return [0, 0]

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
        '''
        Return the distribution of the transition probability conditionned by (s, t, a)
        If a full state (time-enhanced) is provided as argument , only the position is used
        '''
        pos = s
        if (type(pos) == list):
            pos = pos[0]
        assert(isinstance(pos,np.int64) or isinstance(pos, int))
        return self.T[pos, a, t]

    def transition_probability(self, s_p, s, t, a):
        '''
        Return the probability of transition to s_p conditionned by (s, t, a)
        If a full state (time-enhanced) is provided as argument , only the position is used
        '''
        pos = s
        pos_p = s_p
        if (type(pos) == list):
            pos = pos[0]
        if (type(pos_p) == list):
            pos_p = pos_p[0]
        assert(isinstance(pos,np.int64) or isinstance(pos, int))
        assert(isinstance(pos_p,np.int64) or isinstance(pos_p, int))
        return self.T[pos, a, t, pos_p]

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
        '''
        Return the instant reward r(s, t, a)
        If a full state (time-enhanced) is provided as argument , only the position is used
        '''
        pos = s
        if (type(pos) == list):
            pos = pos[0]
        assert(isinstance(pos,np.int64) or isinstance(pos, int))
        return self.R[pos, a, t]

    def equality_operator(self, s1, s2):
        '''
        Equality operator, return True if the two input states are equal.
        '''
        return (s1 == s2)

    def transition(self, state, action, is_model_dynamic):
        '''
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        '''
        position_p, time_p = self.state
        reward = self.reward(position_p, time_p, action)
        transition_model = self.transition_probability_distribution(position_p, time_p, action)
        position_p = np.random.choice(self.pos_space, size=None, replace=False, p=transition_model)
        if is_model_dynamic:
            time_p += self.timestep
        state_p = [int(position_p), time_p]
        if t >= self.nT - 1: # Timeout
            done = True
        else:
            done = False
        return state_p, reward, done

    def step(self, action):
        '''
        Step function equivalent to transition and reward function.
        Actually modifies the environment's state attribute.
        Return (observation, reward, termination criterion (boolean), informations)
        '''
        self.state, reward, done = self.transition(self.state, action, True)
        return (self.state, reward, done, {})

    def is_terminal(self, state):
        return False

    def print_state(self):
        print('position: {}; t: {}'.format(self.state[0],self.state[1]))

    def get_state_space_at_time(self, t):
        space = []
        for i in range(self.nS):
            space.append([self.pos_space[i], t])
        return space

    def get_time(self):
        return self.state[1]

    def reset(self):
        self.state = self.initial_state()
        self.steps_beyond_done = None
        return self.state

    def render(self, mode='human', close=False):
        '''
        No rendering yet
        '''
        return None
