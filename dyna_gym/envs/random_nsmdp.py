"""
Randomly generated NSMDP
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
        self.nS = 4 # n states
        self.nTS = 1 # n terminal states TODO
        self.minSBF = 1 # minimum state branching factor after taking an action
        self.maxSBF = 3 # maximum state branching factor after taking an action
        self.nA = 2 # n actions
        self.nT = 10 # n timesteps
        self.pos_space = np.array(range(self.nS))
        #self.action_space = spaces.Discrete(self.nA)
        self.Rmin = 0.0
        self.Rmax = 1.0

        self.epsilon = 1.0 # 0 = random; 1 = adversarial; in-between = mixture of both
        self.L_p = 0.1 # transition kernel Lipschitz constant
        self.L_r = 0.1 # reward function Lipschitz constant
        self.tau = 1 # timestep duration

        self.RS = self.generate_reachable_states()
        self.TS = self.generate_terminal_states()
        self.T = self.generate_transition_matrix()
        self.R = self.generate_reward_matrix()

        #self._seed()
        self.reset()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = (0, 0)
        self.steps_beyond_done = None

    def generate_terminal_states(self):
        TS = np.zeros(shape=self.nS, dtype=int)
        ts = np.random.randint(low=0, high=self.nS, size=self.nTS)
        for s in ts:
            TS[s] = 1
        return TS

    def generate_reachable_states(self):
        RS = np.zeros(shape=(self.nS, self.nA, self.nS), dtype=int)
        for i in range(self.nS):
            for j in range(self.nA):
                RS[i, j, (i+1)%self.nS] = 1
                if self.maxSBF > 1:
                    bf = np.random.randint(low=self.minSBF, high=self.maxSBF)
                    rs = np.random.randint(low=0, high=self.nS, size=bf)
                    for s in rs:
                        RS[i, j, s] = 1
        return RS

    def generate_transition_matrix(self):
        """
        Initialize the transition matrix.
        When the step function is called, a new one is generated based on the previous one.
        The behavior of the environment (adversarial or not) is set with the epsilon factor.
        """
        T = np.zeros(shape=(self.nS, self.nA, self.nS), dtype=float)
        for i in range(self.nS):
            for j in range(self.nA):
                rs = self.RS[i, j, :]
                nrs = sum(rs)
                d = distribution.random_tabular(size=nrs)
                index = 0
                for k in range(self.nS):
                    if rs[k] == 1:
                        T[i,j,k] = d[index]
                        index += 1
        return T

    def generate_reward_matrix(self):
        R = np.zeros(shape=(self.nS, self.nA, self.nS), dtype=float)
        for i in range(self.nS):
            for j in range(self.nA):
                for k in range(self.nS):
                    if self.TS[k] == 1:
                        R[i, j, k] = self.Rmin
                    else:
                        R[i, j, k] = np.random.uniform(low=self.Rmin, high=self.Rmax)
        return R
        '''
        #TODO
        Questions:
        - Catastrophic terminal states?
        - Depth of the evolution?

        Faire un pull
        '''

    def transition_probability_distribution(self, s, t, a):
        return self.T[s.index, a, t]

    def transition_probability(self, s_p, s, t, a):
        return self.T[s.index, a, t, s_p.index]

    def reward(self, s, t, a):
        #TODO
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
        #TODO
        p_p, t = self.state
        reward = self.reward(p_p, t, action)
        transition_model = self.transition_probability_distribution(p_p, t, action)
        p_p = np.random.choice(self.pos_space, size=None, replace=False, p=transition_model)
        if is_model_dynamic:
            t += self.tau
        state_p = (int(p_p), t)
        if t >= self.nT - 1: # Timeout
            done = True
        else:
            done = False
        return state_p, reward, done

    def evolve(self):
        """
        Change the transition and reward matrices.
        Evolutions are either random or adversarial given the epsilon factor.
        """
        #TODO
        print('TODO')

    def step(self, action):
        """
        Step function equivalent to transition and reward function.
        Actually modifies the environment's state attribute.
        Return (observation, reward, termination criterion (boolean), informations)
        """
        self.state, reward, done = self.transition(self.state, action, True)
        self.evolve()
        return (self.state, reward, done, {})

    def is_terminal(self, state):
        return False

    def print_state(self):
        print('s: {}; t: {}'.format(self.state[0],self.state[1]))

    def get_state_space_at_time(self, t):
        return [(x, t) for x in self.pos_space]

    def get_time(self):
        return self.state[1]

    def render(self, mode='human', close=False):
        """
        No rendering yet
        """
        return None
