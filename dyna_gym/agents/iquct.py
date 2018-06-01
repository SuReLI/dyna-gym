"""
Inferred Q-values UCT Algorithm
"""

import gym
import random
import itertools
import numpy as np
from math import sqrt, log
from copy import copy
from sklearn.linear_model import Ridge

def snapshot_value(node):
    '''
    Value estimate of a chance node wrt current snapshot model of the MDP
    '''
    return sum(node.sampled_returns) / len(node.sampled_returns)

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

def update_histories(histories, node, env):
    '''
    Update the collected histories.
    Recursive method.
    @TODO discard the elements once their duration is zero?
    '''
    for child in node.children:
        if child.sampled_returns: # Ensure there are sampled returns
            # Update history of child
            match = False
            duration = child.depth * env.tau
            for h in histories:
                if (h[1] == child.action) and env.equality_operator(h[0],child.parent.state): # The child's state-action pair matches an already built history
                    h[2].append([snapshot_value(child),duration])
                    match = True
                    break
            if not match: # No match occured, add a new history
                histories.append([
                    child.parent.state,
                    child.action,
                    [[snapshot_value(child),duration]]
                ])
            # Recursive call
            for grandchild in child.children:
                update_histories(histories, grandchild, env)

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
    def __init__(self, parent, action, env, histories):
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.sampled_returns = []

        self.history = []
        for h in histories:
            if (h[1] == self.action) and env.equality_operator(h[0],self.parent.state):
                self.history = copy(h[2])

class IQUCT(object):
    '''
    IQUCT agent
    '''
    def __init__(self, action_space, gamma, rollouts, max_depth, ucb_constant, regularization, degree):
        self.action_space = action_space
        self.gamma = gamma
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.is_model_dynamic = False # default
        self.histories = [] # saved histories
        self.ucb_constant = ucb_constant

        # Regression parameters
        self.reg = regularization
        self.deg = degree

        self.reg_datasz = []#TRM

    def reset(self):
        '''
        Reset Agent's attributes.
        '''
        self.histories = [] # saved histories

    def poly_feature(self, x):
        result = np.array([],float)
        for i in range(self.deg+1):
            result = np.append(result, [x**i])
        return result

    def poly_reg(self, data, x):
        '''
        Perform Linear Regression with Polynomial features and returns the predictor.
        Data should have the form [[x,y], ...].
        Return the prediction at the value specified by x
        '''
        self.reg_datasz.append(len(data))#TRM
        X = []
        y = []
        for d in data:
            X = np.append(X, self.poly_feature(d[0]))
            y = np.append(y, d[1])
        X = X.reshape((len(data), self.deg+1))
        clf = Ridge(alpha=self.reg)
        clf.fit(X, y)
        return clf.predict(self.poly_feature(x).reshape(1,-1))

    def inferred_value(self, node):
        '''
        Value estimate of a chance node wrt selected predictor
        No inference is performed if the history is empty or has too few data points.
        @TODO maybe consider inferring only with a higher number of data points
        '''
        if(len(node.history) > 1):
            return self.poly_reg(node.history, 0)
        else:
            return snapshot_value(node)

    def ucb(self, node):
        '''
        Upper Confidence Bound of a chance node
        '''
        return self.inferred_value(node) + self.ucb_constant * sqrt(log(node.parent.visits)/len(node.sampled_returns))

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
                        else: # Go to chance node maximizing UCB using inferred values
                            node = max(node.children, key=self.ucb)
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
                    node.children = [ChanceNode(node,a,env,self.histories) for a in combinations(env.action_space)]
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
        update_histories(self.histories, root, env)


        print('Bilan reg:')#TRM
        for i in range(100): #TRM
            if self.reg_datasz.count(i) > 0:
                print('{} reg with {} pt'.format(self.reg_datasz.count(i),i))
        self.reg_datasz = []
        print()#TRM


        return max(root.children, key=self.inferred_value).action
