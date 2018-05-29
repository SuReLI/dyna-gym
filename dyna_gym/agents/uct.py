"""
UCT Algorithm
"""

import gym
import random
import itertools
from math import sqrt, log
from copy import copy

def decision_node_value(node):
    '''
    Value of a decision node
    '''
    return chance_node_value(max(node.children, key=chance_node_value)) #TODO can be improved

def chance_node_value(node):
    '''
    Value of a chance node
    '''
    return sum(node.sampled_returns) / len(node.sampled_returns)

def ucb(node):
    '''
    Upper Confidence Bound of a chance node
    '''
    return chance_node_value(node) + sqrt(log(node.parent.visits)/len(node.sampled_returns))

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

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
    def __init__(self, action_space, gamma, rollouts, max_depth, is_model_dynamic):
        self.action_space = action_space
        self.gamma = gamma
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.is_model_dynamic = is_model_dynamic

    def reset(self):
        '''
        Reset Agent's attributes.
        Nothing to reset.
        '''

    def act(self, env, done):
        '''
        Compute the entire UCT procedure
        '''
        root = DecisionNode(None, env.get_state(), done)
        for _ in range(self.rollouts):
            rewards = [] # Rewards collected along the tree for the current rollout
            node = root # Current node
            terminal = done

            # Selection
            select = True
            expand_chance_node = False
            while select and (len(root.children) != 0):
                if (type(node) == DecisionNode): # Decision node
                    if node.is_terminal: # Terminal, evaluate parent
                        node = node.parent
                        select = False
                    else: # Decision node is not terminal
                        if node.explored_children < len(node.children): # Go to unexplored chance node
                            child = node.children[node.explored_children]
                            node.explored_children += 1
                            node = child
                            select = False
                        else: # Go to chance node maximizing UCB
                            node = max(node.children, key=ucb)
                else: # Chance Node
                    state_p, reward, terminal = env.transition(node.parent.state,node.action,self.is_model_dynamic)
                    rewards.append(reward)
                    if (len(node.children) == 0): # No child
                        expand_chance_node = True
                        select = False
                    else: # Already has children
                        for i in range(len(node.children)):
                            if env.equality_operator(node.children[i].state,state_p): # State already sampled
                                node = node.children[i]
                                break
                            else: # New state sampled
                                expand_chance_node = True
                                select = False
                                break

            # Expansion
            if expand_chance_node and (type(node) == ChanceNode): # Expand a chance node
                node.children.append(DecisionNode(node,state_p,terminal))
                node = node.children[-1]
            if (type(node) == DecisionNode): # Expand a decision node
                if terminal:
                    node = node.parent
                else:
                    node.children = [ChanceNode(node, a) for a in combinations(env.action_space)]
                    random.shuffle(node.children)
                    child = node.children[0]
                    node.explored_children += 1
                    node = child

            # Evaluation
            t = 0
            estimate = 0
            state = node.parent.state
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
                node.parent.visits += 1
                node = node.parent.parent
        return max(root.children, key=chance_node_value).action
