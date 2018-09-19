"""
Risk Averse Tree Search (RATS) algorithm
"""

import gym
import random
import itertools
import numpy as np
import dyna_gym.utils.distribution as distribution
from scipy.stats import wasserstein_distance

def node_value(node):
    assert(node.value != None)
    return node.value

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
        self.value = None

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
        self.value = None

class RATS(object):
    '''
    RATS agent
    '''
    def __init__(self, action_space, max_depth, gamma, L_v, horizon):
        self.action_space = action_space
        self.n_actions = self.action_space.shape[0]
        self.max_depth = max_depth
        self.t_call = None

        self.gamma = gamma # discount factor
        self.L_v = L_v # value function's Lipschitz constant
        self.horizon = horizon # maximum number of timesteps for MC simulations

    def reset(self):
        '''
        Reset Agent's attributes.
        '''
        self.t_call = None

    def build_tree(self, node, env):
        if type(node) is DecisionNode: #DecisionNode
            if (node.depth < self.max_depth):
                for a in range(self.n_actions):
                    node.children.append(ChanceNode(node, a))
            else: #Reached maximum depth
                return None
        else: #ChanceNode
            for s_p in env.get_state_space_at_time(self.t_call):
                node.children.append(
                    DecisionNode(
                        parent=node,
                        state=s_p,
                        weight=env.transition_probability(s_p, node.parent.state, self.t_call, node.action),
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
        if (type(node) is DecisionNode):
            if (node.is_terminal or (node.depth == self.max_depth)):
                assert(node.parent.parent.parent.parent.parent == None)
                assert(node.value == None)
                node.value = self.heuristic_value(node, env)
            else:
                v = -1e99
                for ch in node.children:
                    v = max(v, self.minimax(ch, env))
                assert(node.value == None)
                node.value = v
        else: # ChanceNode
            self.set_worst_case_distribution(node, env) # min operator
            v = 0
            for ch in node.children: # pessimistic look-ahead values
                v += ch.weight * ch.value #self.minimax(ch, env)
            v *= self.gamma
            v += env.reward(node.parent.state, self.t_call, node.action) - env.L_r * env.timestep * node.depth # pessimistic reward value
            assert(node.value == None)
            node.value = v
        return node.value

    def set_worst_case_distribution(self, node, env):
        '''
        Modify the weights of the children so that the worst distribution is set wrt their values.
        '''
        assert(type(node) is ChanceNode)
        n_states = len(node.children)
        n_trials = 1000

        # 1. Generate random distributions
        v_0 = np.zeros(shape=n_states, dtype=float)
        for i in range(n_states):
            v_0[i] = node.children[i].weight
        buff = np.empty(shape=(n_trials, n_states), dtype=float)
        buff[0] = v_0
        for i in range(1,n_trials):
            buff[i] = distribution.random_tabular(n_states)

        # 2. Delete distributions too far from v_0
        todelete = []
        for i in range(n_trials):
            if wasserstein_distance(range(n_states),range(n_states),v_0,buff[i]) > (node.depth * env.L_p * env.timestep):
                todelete.append(i)
        buff = np.delete(buff, todelete, 0)

        #3. Unique recursive call st values are set
        for ch in node.children:
            self.minimax(ch, env)

        #4. Pick the minimizing distribution
        minval = 1e99
        argmin = 0
        for i in range(len(buff)):
            val = 0
            for j in range(len(node.children)):
                val += buff[i][j] * node.children[j].value
            if val < minval:
                minval = val
                argmin = i

        return buff[argmin]

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
        self.t_call = env.get_time()
        root = self.initialize_tree(env, done)
        self.minimax(root, env)
        return max(root.children, key=node_value).action
