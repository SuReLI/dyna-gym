"""
Generic benchmark method:

Require:
agent.reset(param)
agent.display()
agent.act(env, done)

env.reset()
"""

import gym
import numpy as np
import statistics as stat
from multiprocessing import Pool

import dyna_gym.agents.uct as uct
import csv

def csv_write(row, path, mode):
    with open(path, mode) as csvfile:
        w = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(row)

def run(agent, env, tmax, verbose=False):
    """
    Run single episode
    """
    done = False
    cr = 0.0
    if verbose:
        env.render()
    for t in range(tmax):
        action = agent.act(env,done)
        _, r, done, _ = env.step(action)
        cr += r
        if verbose:
            env.render()
        if (t+1 == tmax) or done:
            break
    return cr

def benchmark(agent_name, agent, param_pool, env_name, nenv, nepi, tmax, save=True, path='log.csv', verbose=True):
    """
    Benchmark a single agent within an environment.
    agent_name : name of the agent for saving parameter
    agent      : agent object
    param_pool : tested combinations of parameters
    env_name   : name of the generated environment
    nenv       : number of generated environment
    nepi       : number of episodes per generated environment
    tmax       : timeout for each episode
    save       : save the results or not
    path       : path of the save
    verbose    : if true, display informations during benchmark
    """
    nag = len(param_pool)
    if save:
        csv_write(['env_name', 'env_number', 'agent_name', 'agent_number', 'epi_number', 'score'], path, 'w')
    for j in range(nenv):
        env = gym.make(env_name)
        if verbose:
            print('Created environment', j+1, '/', nenv)
        for i in range(nag):
            agent.reset(param_pool[i])
            if verbose:
                print('Created agent', i+1, '/', nag)
                agent.display()
            for k in range(nepi):
                if verbose:
                    print('Run episode', k+1, '/', nepi)
                env.reset()
                score = run(agent, env, tmax)
                if save:
                    csv_write([env_name, j, agent_name, i, k, score], path, 'a')

def test():
    """
    Example
    """
    env = gym.make('NSFrozenLakeEnv-v0')
    nenv = 1
    nepi = 3
    agent = uct.UCT(env.action_space)

    param_pool = [
        [env.action_space,  10, 100, 0.9, 6.36396103068, True],
        [env.action_space,  50, 100, 0.9, 6.36396103068, True],
        [env.action_space, 100, 100, 0.9, 6.36396103068, True]
    ]

    benchmark('UCT', agent, param_pool, 'NSFrozenLakeEnv-v0', nenv, nepi, 100)
