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

def poly_feature(x, deg):
    result = np.array([],float)
    for i in range(deg+1):
        result = np.append(result, [x**i])
    return result

def poly_reg(data, x):
    '''
    Perform Linear Regression with Polynomial features and returns the predictor.
    Data should have the form [[x,y], ...].
    Return the prediction at the value specified by x
    '''
    reg = 1 # Regularization
    deg = 1 # Degree of the polynomial
    X = []
    y = []
    for d in data:
        X = np.append(X, poly_feature(d[0], deg))
        y = np.append(y, d[1])
    X = X.reshape((len(data), deg+1))
    clf = Ridge(alpha=reg)
    clf.fit(X, y)
    return clf.predict(poly_feature(x, deg).reshape(1,-1))

def snapshot_value(node):
    '''
    Value estimate of a chance node wrt current snapshot model of the MDP
    '''
    return sum(node.sampled_returns) / len(node.sampled_returns)

def inferred_value(node):
    '''
    Value estimate of a chance node wrt selected predictor
    No inference is performed if the history is empty or has too few data points.
    @TODO maybe consider inferring only with a higher number of data points
    '''
    if(len(node.history) > 1):
        return poly_reg(node.history, 0)
    else:
        return snapshot_value(node)

def ucb(node):
    '''
    Upper Confidence Bound of a chance node
    '''
    return inferred_value(node) + 0.7 * sqrt(log(node.parent.visits)/len(node.sampled_returns))

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

def print_state(prefix, s): #TRM
    if s is None:
        print('None')
    else:
        print('{}x: {:.6f}; xd: {:.6f}; t: {:.6f}; td: {:.6f}'.format(prefix,s[0],s[1],s[2],s[3]))

def print_tree(node, prev_root, env):#TRM
    assert(type(node) == DecisionNode)
    print('print tree--------------start')

    print('ROOT:')
    print_state('',node.state)
    '''
    if prev_root is not None:
        print('Prev root grandchild:')
        for pvchild in prev_root.children:
            for pvgrandchild in pvchild.children:
                print_state(pvgrandchild.state)
                answ = env.equality_operator(pvgrandchild.state,node.state)
                print(' -> is equal to new root: {}'.format(answ))
    '''

    print('  CHILDREN:')
    for child in node.children:
        print('  a = {}'.format(child.action))
        print_state('  ',child.parent.state)

    print('    GRANDx2 CHILDREN:')
    for child in node.children:
        for gchild in child.children:
            for ggchild in gchild.children:
                print('    a = {}'.format(ggchild.action))
                print_state('    ',ggchild.parent.state)

    print('      GRANDx4 CHILDREN:')
    for child in node.children:
        for gchild in child.children:
            for ggchild in gchild.children:
                for gggchild in ggchild.children:
                    for ggggchild in gggchild.children:
                        print('      a = {}'.format(ggggchild.action))
                        print_state('      ',ggggchild.parent.state)
    print('--------------------------end')

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
    def __init__(self, action_space, gamma, rollouts, max_depth, is_model_dynamic):
        self.action_space = action_space
        self.gamma = gamma
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.is_model_dynamic = is_model_dynamic
        self.histories = [] # saved histories

    def reset(self):
        '''
        Reset Agent's attributes.
        '''
        self.histories = [] # saved histories

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
                #state, reward, terminal = env.transition(state,action,self.is_model_dynamic)#TODO put back
                #estimate += reward * (self.gamma**t)#TODO put back
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
        return max(root.children, key=inferred_value).action
