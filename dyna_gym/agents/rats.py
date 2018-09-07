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
    def __init__(self, parent, state, weight, is_terminal):
        self.parent = parent
        self.state = state
        self.weight = weight # Probability to occur
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
        self.n_actions = self.action_space.shape[0]
        self.max_depth = max_depth

    def reset(self):
        '''
        Reset Agent's attributes. Nothing to reset for RATS agent.
        '''

    def build_tree(self, node, env):
        if type(node) is DecisionNode:
            if (node.depth < self.max_depth):
                for a in range(self.n_actions):
                    node.children.append(ChanceNode(node, a))
                for ch in node.children:
                    self.build_tree(ch, env)
        else: #ChanceNode
            for s_p in env.get_state_space_at_time(env.get_time()):
                node.children.append(
                    DecisionNode(
                        node,
                        s_p,
                        env.transition_probability(s_p[0], node.parent.state[0], env.get_time(), node.action),
                        env.is_terminal(s_p)
                    )
                )
            for ch in node.children:
                self.build_tree(ch, env)
            #TODO test

    def initialize_tree(self, env, done):
        node = DecisionNode(None, env.state, 1, done)
        self.build_tree(node, env)
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
