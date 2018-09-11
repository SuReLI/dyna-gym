"""
Risk Averse Tree Search (RATS) algorithm
"""

import gym
import random
import itertools
import numpy as np
import dyna_gym.utils.distribution as distribution

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
    def __init__(self, action_space, max_depth, gamma, L_v, horizon):
        self.action_space = action_space
        self.n_actions = self.action_space.shape[0]
        self.max_depth = max_depth

        self.gamma = gamma # discount factor
        self.L_v = L_v # value function's Lipschitz constant
        self.horizon = horizon # maximum number of timesteps for MC simulations

    def reset(self):
        '''
        Reset Agent's attributes. Nothing to reset for RATS agent.
        '''

    def build_tree(self, node, env):
        if type(node) is DecisionNode: #DecisionNode
            if (node.depth < self.max_depth):
                for a in range(self.n_actions):
                    node.children.append(ChanceNode(node, a))
            else: #Reached maximum depth
                return None
        else: #ChanceNode
            for s_p in env.get_state_space_at_time(env.get_time()):
                node.children.append(
                    DecisionNode(
                        parent=node,
                        state=s_p,
                        weight=env.transition_probability(s_p, node.parent.state, env.get_time(), node.action), # TODO test if good weight
                        is_terminal=env.is_terminal(s_p)
                    )
                )
        for ch in node.children:
            self.build_tree(ch, env)

    def initialize_tree(self, env, done):
        '''
        Initialize an empty tree.
        The tree is composed with all the possible actions as chance nodes and all the possible state-outcome as decision nodes.
        The used model is the snapshot MDP provided by the environment at the time of the environment.
        The depth of the tree is defined by the self.max_depth attribute of the agent.
        The used heuristic for the evaluation of the leaf nodes that are not terminal nodes is defined by the function self.heuristic_value.
        '''
        root = DecisionNode(None, env.state, 1, done)
        self.build_tree(root, env)
        return root

    def minimax(self, node, env):
        if (type(node) is DecisionNode) and (node.is_terminal or (node.depth == self.max_depth)):
            return self.heuristic_value(node, env)
        if (type(node) is DecisionNode):
            value = -1e99
            for ch in node.children:
                value = max(value, self.minimax(ch, env))
            return value
        else: # ChanceNode
            self.set_worst_case_distribution(node, env) # min operator
            value = env.reward(node.parent.state, env.get_time(), node.action) - env.L_r * env.timestep * node.depth # pessimistic reward value
            for ch in node.children: # pessimistic look-ahead value
                value = value + self.gamma * ch.weight * self.minimax(ch, env)
            return value

    def set_worst_case_distribution(self, node, env):
        '''
        Modify the weights of the children so that the worst distribution is set wrt their values.
        '''
        v_0 = np.zeros(shape=len(node.children), dtype=float)
        for i in range(len(node.children)):
            v_0[i] = node.children[i].weight
        print(v_0)
        exit()
        return v_0

    def heuristic_value(self, node, env):
        '''
        Return the heuristic value of the input state.
        This value is computed via Monte-Carlo simulations using the snapshot MDP provided by the environment at the time of the environment.
        '''
        assert(type(node) == DecisionNode)
        value = 0
        s = node.state
        for t in range(self.horizon):
            a = self.action_space.sample()
            s, r, done = env.transition(s, a, is_model_dynamic=False)
            value = value + self.gamma**t * r
        return value - self.L_v * env.timestep * node.depth

    def act(self, env, done):
        '''
        Compute the entire RATS procedure
        '''
        self.root = self.initialize_tree(env, done)
        self.minimax(self.root, env)
        return max(self.root.children, key=chance_node_value).action
