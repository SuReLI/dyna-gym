"""
Risk Averse Tree Search (RATS) algorithm
"""

import gym
import random
import itertools

class DecisionNode:
    '''
    Decision node class, labelled by a state
    '''
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.children = []

class ChanceNode:
    '''
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    '''
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []

class RATS(object):
    '''
    RATS agent
    '''
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self):
        '''
        Reset Agent's attributes. Nothing to reset for RATS agent.
        '''

    def act(self, env, done):
        '''
        Compute the entire RATS procedure
        '''
        self.root = DecisionNode(None, env.state, done)
        #TODO
        return max(self.root.children, key=chance_node_value).action
