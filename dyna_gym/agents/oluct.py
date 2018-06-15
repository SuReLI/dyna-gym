"""
OLUCT (Open Loop UCT) Algorithm
"""

import gym
import random
import itertools
from math import sqrt, log
from copy import copy

def value(node):
    '''
    Value of the node
    '''
    return sum(node.sampled_returns) / len(node.sampled_returns)

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        if self.parent is None: # Root node
            self.depth = 0
        else: # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.explored_children = 0
        self.sampled_returns = []

class OLUCT(object):
    '''
    OLUCT agent
    '''
    def __init__(self, action_space, gamma, rollouts, max_depth, is_model_dynamic, ucb_constant):
        self.action_space = action_space
        self.gamma = gamma
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.is_model_dynamic = is_model_dynamic
        self.ucb_constant = ucb_constant

    def reset(self):
        '''
        Reset Agent's attributes.
        Nothing to reset.
        '''

    def ucb(self, node):
        '''
        Upper Confidence Bound
        '''
        return value(node) + self.ucb_constant * sqrt(log(len(node.parent.sampled_returns))/len(node.sampled_returns))

    def act(self, env, done):
        '''
        Compute the entire OLUCT procedure
        '''
        root = Node(None, None)
        for _ in range(self.rollouts):
            rewards = [] # Rewards collected along the tree for the current rollout
            node = root # Current node
            terminal = done
            state = env.state

            # Selection
            while len(node.children) != 0: # While node has children
                if node.explored_children < len(node.children): # Explore a new child
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else: # Go to UCB child
                    node = max(node.children, key=self.ucb)
                state, reward, terminal = env.transition(state,node.action,self.is_model_dynamic)
                rewards.append(reward)

            # Expansion
            if not terminal:
                node.children = [Node(node, a) for a in combinations(env.action_space)]
                random.shuffle(node.children)

            # Evaluation
            t = 0
            estimate = 0
            while not terminal:
                action = env.action_space.sample() # default policy
                state, reward, terminal = env.transition(state,action,self.is_model_dynamic)
                estimate += reward * (self.gamma**t)
                t += 1
                if node.depth + t > self.max_depth:
                    break

            # Backpropagation
            while node:
                node.sampled_returns.append(estimate)
                if len(rewards) != 0:
                    estimate = rewards.pop() + self.gamma * estimate
                node = node.parent
        return max(root.children, key=value).action
