"""
MCTS Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""

import random
import itertools
import dyna_gym.utils.utils as utils
from gym import spaces
from math import sqrt, log
from copy import copy

def chance_node_value(node):
    """
    Value of a chance node
    """
    return sum(node.sampled_returns) / len(node.sampled_returns)

def decision_node_value(node):
    """
    Value of a decision node
    """
    return chance_node_value(max(node.children, key=chance_node_value))

def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    elif isinstance(space, spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

def mcts_tree_policy(ag, children):
    return random.choice(children)

def mcts_procedure(ag, tree_policy, env, done):
    """
    Compute the entire MCTS procedure wrt to the selected tree policy.
    Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
    and returning the one chosen by the tree policy.
    """
    root = DecisionNode(None, env.state, ag.action_space.copy(), done)
    for _ in range(ag.rollouts):
        rewards = [] # Rewards collected along the tree for the current rollout
        node = root # Current node
        terminal = done

        # Selection
        select = True
        while select:
            if (type(node) == DecisionNode): # DecisionNode
                if node.is_terminal:
                    select = False # Selected a terminal DecisionNode
                else:
                    if len(node.children) < ag.n_actions:
                        select = False # Selected a non-fully-expanded DecisionNode
                    else:
                        #node = random.choice(node.children) #TODO remove
                        node = tree_policy(ag, node.children)
            else: # ChanceNode
                state_p, reward, terminal = env.transition(node.parent.state, node.action, ag.is_model_dynamic)
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
                state_p, reward, terminal = env.transition(node.parent.state ,node.action, ag.is_model_dynamic)
                rewards.append(reward)
            # ChanceNode
            node.children.append(DecisionNode(node, state_p, ag.action_space.copy(), terminal))
            node = node.children[-1]

        # Evaluation
        assert(type(node) == DecisionNode)
        t = 0
        estimate = reward
        state = node.state
        while (not terminal) and (t < ag.horizon):
            action = env.action_space.sample() # default policy
            state, reward, terminal = env.transition(state, action, ag.is_model_dynamic)
            estimate += reward * (ag.gamma**t)
            t += 1

        # Backpropagation
        node.visits += 1
        node = node.parent
        assert(type(node) == ChanceNode)
        while node:
            node.sampled_returns.append(estimate)
            if len(rewards) != 0:
                estimate = rewards.pop() + ag.gamma * estimate
            node.parent.visits += 1
            node = node.parent.parent

    return max(root.children, key=chance_node_value).action

class DecisionNode:
    """
    Decision node class, labelled by a state
    """
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
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.sampled_returns = []

class MCTS(object):
    """
    MCTS agent
    """
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, is_model_dynamic=True):
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(action_space))
        else:
            self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.is_model_dynamic = is_model_dynamic

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p == None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p,[spaces.discrete.Discrete, int, int, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4])

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying MCTS agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('Is model dynamic   :', self.is_model_dynamic)

    def act(self, env, done):
        return mcts_procedure(self, mcts_tree_policy, env, done)
