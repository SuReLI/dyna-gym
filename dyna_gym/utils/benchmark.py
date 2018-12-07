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
import dyna_gym.agents.my_random_agent as ra
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

def benchmark(env_name, n_env, agent_name_pool, agent_pool, param_pool, n_epi, tmax, save=True, paths_pool=['log.csv'], verbose=True):
    """
    Benchmark a single agent within an environment.
    env_name        : name of the generated environment
    n_env           : number of generated environment
    agent_name_pool : list containing the names of the agents for saving purpose
    agent_pool      : list containing the agent objects
    param_pool      : list containing lists of parameters for each agent object
    n_epi           : number of episodes per generated environment
    tmax            : timeout for each episode
    save            : save the results or not
    paths_pool      : list containing the saving path for each agent
    verbose         : if true, display informations during benchmark
    """
    assert len(agent_name_pool) == len(agent_pool) == len(param_pool)
    n_agents = len(param_pool)
    if save:
        assert len(paths_pool) == n_agents
    for i in range(n_env):
        env = gym.make(env_name)
        if verbose:
            print('Created environment', i+1, '/', n_env)
        for j in range(n_agents):
            if save:
                csv_write(['env_name', 'env_number', 'agent_name', 'agent_number', 'epi_number', 'score'], paths_pool[j], 'w')
            agent = agent_pool[j]
            n_agents_j = len(param_pool[j])
            for k in range(n_agents_j):
                agent.reset(param_pool[j][k])
                if verbose:
                    print('Created agent', j+1, '/', n_agents,'with parameters', k+1, '/', n_agents_j)
                    agent.display()
                for l in range(n_epi):
                    if verbose:
                        print('Run episode', l+1, '/', n_epi)
                    env.reset()
                    score = run(agent, env, tmax)
                    if save:
                        csv_write([env_name, i, agent_name_pool[j], k, l, score], paths_pool[j], 'a')

def test():
    """
    Example
    """
    env = gym.make('NSFrozenLakeEnv-v0')
    nenv = 1
    nepi = 3
    tmax = 100

    agent_name_pool = ['UCT','RANDOM']
    agent_pool = [uct.UCT(env.action_space), ra.MyRandomAgent(env.action_space)]
    param_pool = [
        [[env.action_space,  10, 100, 0.9, 6.36396103068, True],[env.action_space, 100, 100, 0.9, 6.36396103068, True]],
        [[env.action_space],[env.action_space],[env.action_space]]
    ]
    paths_pool = ['uct.csv','random.csv']

    benchmark('NSFrozenLakeEnv-v0', nenv, agent_name_pool, agent_pool, param_pool, nepi, tmax, save=True, paths_pool=paths_pool, verbose=True)
