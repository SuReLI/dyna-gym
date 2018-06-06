import gym
import random
import itertools
import numpy as np
import statistics as stat
from math import sqrt, log
import dyna_gym.agents.lwpr_for_iquct as lwprmdl

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
    def __init__(self, parent, action, env):
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.sampled_returns = []

class ApproxIQUCT(object):
    '''
    Approximated Inferred Q-values UCT Algorithm

    Tabular wrt action space only.
    Generalization is made through state-temporal space.
    '''
    def __init__(self, action_space, state_dim, gamma, rollouts, max_depth, ucb_constant, use_averaged_qval):
        self.action_space = action_space
        self.gamma = gamma
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.is_model_dynamic = False # default
        self.ucb_constant = ucb_constant
        self.is_model_initialized = False
        # Model parameters
        self.use_averaged_qval = use_averaged_qval
        self.model = lwprmdl.LWPRForIQUCT(state_dim)

    def reset(self):
        '''
        Reset Agent's attributes.
        '''
        self.model.reset()

    def inferred_value(self, node):
        '''
        Value estimate of a chance node wrt selected predictor
        No inference is performed if the history is empty or has too few data points.
        '''
        if self.is_model_initialized:
            return self.model.prediction_at(np.array(node.parent.state), 0.0, node.action)
        else:
            return snapshot_value(node) #TODO write the model inference TRM

    def ucb(self, node):
        '''
        Upper Confidence Bound of a chance node
        '''
        return self.inferred_value(node) + self.ucb_constant * sqrt(log(node.parent.visits)/len(node.sampled_returns))

    def extract_data(self, data, node, env):
        '''
        Extract the data from the tree starting at the input node.
        Recursive function.
        Data have the form (s, t, a, Q)
        '''
        for child in node.children:
            duration = child.depth * env.tau
            if child.sampled_returns: # Ensure there are sampled returns
                if self.use_averaged_qval:
                    data.append([np.array(child.parent.state), duration, child.action, snapshot_value(child)])
                else:
                    for q in child.sampled_returns:
                        data.append([np.array(child.parent.state), duration, child.action, q])
                # Recursive call
                for grandchild in child.children:
                    self.extract_data(data, grandchild, env)

    def update_model(self, root_node, env):
        '''
        Collect the data in the tree and update the prediction model.
        '''
        data = []
        self.extract_data(data, root_node, env)
        self.model.update(data)

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
                    node.children = [ChanceNode(node,a,env) for a in combinations(env.action_space)]
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
        self.update_model(root, env)
        self.is_model_initialized = True

        print('model trained with {} data pts'.format(self.model.model.n_data))#TRM

        return max(root.children, key=self.inferred_value).action
