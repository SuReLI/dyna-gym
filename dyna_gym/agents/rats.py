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
    def __init__(self, parent, state, is_terminal):
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None: # Root node
            self.depth = 0
        else: # Non root node
            self.depth = parent.depth + 1
        self.children = []

class ChanceNode:
    '''
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    '''
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []

class RATS(object):
    '''
    RATS agent
    '''
    def __init__(self, action_space, max_depth):
        self.action_space = action_space
        self.max_depth = max_depth

    def reset(self):
        '''
        Reset Agent's attributes. Nothing to reset for RATS agent.
        '''

    def build_tree(self, node):
        if type(node) is DecisionNode:
            #TODO
        else: #ChanceNode
            #TODO

    def initialize_tree(self, env, done):
        node = DecisionNode(None, env.state, done)
        self.build_tree(node)
        return node

    def minimax(self, node):
        '''
        Pseudocode:
        minimax(node, depth, maximizingPlayer) is
            if depth = 0 or node is a terminal node then
                return the heuristic value of node
            if maximizingPlayer then
                value := −∞
                for each child of node do
                value := max(value, minimax(child, depth − 1, FALSE))
                return value
            else (* minimizing player *)
                value := +∞
                for each child of node do
                value := min(value, minimax(child, depth − 1, TRUE))
                return value

        Call:
        minimax(origin, depth, TRUE)
        '''
        if (node.depth == self.max_depth) or ((type(node) is DecisionNode) and node.is_terminal):
            return self.heuristic_value(node)
        #TODO etc

    def heuristic_value(self):
        #TODO
        return 0

    def act(self, env, done):
        '''
        Compute the entire RATS procedure
        '''
        self.root = self.initialize_tree(env, done)
        self.minimax(self.root)
        return max(self.root.children, key=chance_node_value).action
