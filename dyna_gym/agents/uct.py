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

def uct_tree_policy(ag, children):
    return max(children, key=ag.ucb)

class UCT(object):
    '''
    UCT agent
    '''
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, ucb_constant=6.36396103068, is_model_dynamic=True):
        self.action_space = list(mcts.combinations(action_space))
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.is_model_dynamic = is_model_dynamic

    def reset(self):
        ''' Reset Agent's attributes. Nothing to reset for UCT agent. '''

    def ucb(self, node):
        '''
        Upper Confidence Bound of a chance node
        '''
        return mcts.chance_node_value(node) + self.ucb_constant * sqrt(log(node.parent.visits)/len(node.sampled_returns))

    def act(self, env, done):
        return mcts.mcts_procedure(self, uct_tree_policy, env, done)
