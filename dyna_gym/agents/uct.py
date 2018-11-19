"""
UCT Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""

import itertools
import dyna_gym.agents.mcts as mcts
from math import sqrt, log
from gym import spaces

def uct_tree_policy(ag, children):
    return max(children, key=ag.ucb)

class UCT(object):
    """
    UCT agent
    """
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, ucb_constant=6.36396103068, is_model_dynamic=True):
        self.action_space = list(mcts.combinations(action_space))
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.is_model_dynamic = is_model_dynamic

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying UCT agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('UCB constant       :', self.ucb_constant)
        print('Is model dynamic   :', self.is_model_dynamic)

    def reset(self, p):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        assert len(p) == 6, 'Error: expected 6 parameters received {}'.format(len(p))
        assert type(p[0]) == spaces.discrete.Discrete, 'Error: wrong type, expected "gym.spaces.discrete.Discrete", received {}'.format(type(p[0]))
        assert type(p[1]) == int, 'Error: wrong type, expected "int", received {}'.format(type(p[1]))
        assert type(p[2]) == int, 'Error: wrong type, expected "int", received {}'.format(type(p[2]))
        assert type(p[3]) == float, 'Error: wrong type, expected "float", received {}'.format(type(p[3]))
        assert type(p[4]) == float, 'Error: wrong type, expected "float", received {}'.format(type(p[4]))
        assert type(p[5]) == bool, 'Error: wrong type, expected "bool", received {}'.format(type(p[5]))
        self.__init__(p[0], p[1], p[2], p[3], p[4], p[5])

    def ucb(self, node):
        """
        Upper Confidence Bound of a chance node
        """
        return mcts.chance_node_value(node) + self.ucb_constant * sqrt(log(node.parent.visits)/len(node.sampled_returns))

    def act(self, env, done):
        return mcts.mcts_procedure(self, uct_tree_policy, env, done)
