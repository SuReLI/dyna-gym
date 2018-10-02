"""
UCT Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""

import random
import itertools
from gym import spaces as gspaces
from math import sqrt, log
from copy import copy

def decision_node_value(node):
    '''
    Value of a decision node
    '''
    return chance_node_value(max(node.children, key=chance_node_value))

def chance_node_value(node):
    '''
    Value of a chance node
    '''
    return sum(node.sampled_returns) / len(node.sampled_returns)

def combinations(space):
    if isinstance(space, gspaces.Discrete):
        return range(space.n)
    elif isinstance(space, gspaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

class DecisionNode:
    '''
    Decision node class, labelled by a state
    '''
    def __init__(self, parent, state, possible_actions, is_terminal):
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None: # Root node
            self.depth = 0
        else: # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.possible_actions = possible_actions
        random.shuffle(self.possible_actions)
        self.explored_children = 0
        self.visits = 0

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
        self.sampled_returns = []

class UCT(object):
    '''
    UCT agent
    '''
    def __init__(self, action_space, rollouts, gamma=0.9, ucb_constant=6.36396103068, is_model_dynamic=True):
        self.action_space = list(combinations(action_space))
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.is_model_dynamic = is_model_dynamic

    def reset(self):
        '''
        Reset Agent's attributes.
        Nothing to reset for UCT agent.
        '''

    def ucb(self, node):
        '''
        Upper Confidence Bound of a chance node
        '''
        return chance_node_value(node) + self.ucb_constant * sqrt(log(node.parent.visits)/len(node.sampled_returns))

    def act(self, env, done):
        '''
        Compute the entire UCT procedure
        '''
        self.root = DecisionNode(None, env.state, self.action_space.copy(), done)
        for _ in range(self.rollouts):
            rewards = [] # Rewards collected along the tree for the current rollout
            node = self.root # Current node
            terminal = done

            # Selection
            select = True
            while select:
                if (type(node) == DecisionNode): # DecisionNode
                    if node.is_terminal:
                        select = False # Selected a terminal DecisionNode
                    else:
                        if len(node.children) < self.n_actions:
                            select = False # Selected a non-fully-expanded DecisionNode
                        else: # Go to chance node maximizing UCB
                            node = max(node.children, key=self.ucb)
                else: # Chance Node
                    state_p, reward, terminal = env.transition(node.parent.state, node.action, self.is_model_dynamic)
                    rewards.append(reward)
                    if (len(node.children) == 0):
                        select = False # Selected a ChanceNode
                    else:
                        new_state = True
                        for i in range(len(node.children)):
                            if env.equality_operator(node.children[i].state, state_p):
                                node = node.children[i]
                                new_state = False
                                break
                        if new_state:
                            select = False # Selected a ChanceNode

            # Expansion
            if (type(node) == ChanceNode) or ((type(node) == DecisionNode) and not node.is_terminal):
                if (type(node) == DecisionNode):
                    node.children.append(ChanceNode(node, node.possible_actions.pop()))
                    node = node.children[-1]
                    state_p, reward, terminal = env.transition(node.parent.state ,node.action, self.is_model_dynamic)
                    rewards.append(reward)
                # ChanceNode
                node.children.append(DecisionNode(node, state_p, self.action_space.copy(), terminal))
                node = node.children[-1]

            # Evaluation
            assert(type(node) == DecisionNode)
            t = 0
            estimate = reward
            state = node.state
            while not terminal:
                action = env.action_space.sample() # default policy
                state, reward, terminal = env.transition(state, action, self.is_model_dynamic)
                estimate += reward * (self.gamma**t)
                t += 1

            # Backpropagation
            node.visits += 1
            node = node.parent
            assert(type(node) == ChanceNode)
            while node:
                node.sampled_returns.append(estimate)
                if len(rewards) != 0:
                    estimate = rewards.pop() + self.gamma * estimate
                node.parent.visits += 1
                node = node.parent.parent
        return max(self.root.children, key=chance_node_value).action
