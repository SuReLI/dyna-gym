"""
Generic benchmark method:

Require:
agent.reset(param)
agent.display()
agent.act(env, done)

env.reset()
"""

import gym
import csv
import numpy as np
import statistics as stat
import dyna_gym.agents.uct as uct
import dyna_gym.agents.my_random_agent as ra
from multiprocessing import Pool

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

def multi_run(env_name, env_number, env, agent_name_j, agent_number, agent_param, agent, tmax, n_epi, thread_number, save, path_j, verbose):
    for epi_number in range(n_epi):
        if verbose:
            print('Thread', thread_number, 'running episode', epi_number+1, '/', n_epi)
        env.reset()
        score = run(agent, env, tmax)
        if save:
            csv_write([env_name, env_number, agent_name_j, agent_number] + agent_param + [thread_number, score], path_j, 'a')

def benchmark(env_name, n_env, agent_name_pool, agent_pool, param_pool, param_names_pool, n_epi, tmax, save=True, paths_pool=['log.csv'], verbose=True):
    """
    Benchmark a single agent within an environment.
    env_name         : name of the generated environment
    n_env            : number of generated environment
    agent_name_pool  : list containing the names of the agents for saving purpose
    agent_pool       : list containing the agent objects
    param_pool       : list containing lists of parameters for each agent object
    param_names_pool : list containing the parameters names
    n_epi            : number of episodes per generated environment
    tmax             : timeout for each episode
    save             : save the results or not
    paths_pool       : list containing the saving path for each agent
    verbose          : if true, display informations during benchmark
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
            agent = agent_pool[j]
            n_agents_j = len(param_pool[j])
            param_names_j = param_names_pool[j]
            if save:
                csv_write(['env_name', 'env_number', 'agent_name', 'agent_number'] + param_names_j + ['epi_number', 'score'], paths_pool[j], 'w')
            for k in range(n_agents_j):
                agent.reset(param_pool[j][k])
                if verbose:
                    print('Created agent', j+1, '/', n_agents,'with parameters', k+1, '/', n_agents_j)
                    agent.display()
                for l in range(n_epi):
                    if verbose:
                        print('Running episode', l+1, '/', n_epi)
                    env.reset()
                    score = run(agent, env, tmax)
                    if save:
                        csv_write([env_name, i, agent_name_pool[j], k] + param_pool[j][k] + [l, score], paths_pool[j], 'a')

def multithread_benchmark(env_name, n_env, agent_name_pool, agent_pool, param_pool, param_names_pool, n_epi, tmax, save=True, paths_pool=['log.csv'], n_thread=2, verbose=True):
    """
    Benchmark a single agent within an environment.
    env_name         : name of the generated environment
    n_env            : number of generated environment
    agent_name_pool  : list containing the names of the agents for saving purpose
    agent_pool       : list containing the agent objects
    param_pool       : list containing lists of parameters for each agent object
    param_names_pool : list containing the parameters names
    n_epi            : number of episodes per generated environment
    tmax             : timeout for each episode
    save             : save the results or not
    paths_pool       : list containing the saving path for each agent
    n_thread         : number of threads
    verbose          : if true, display informations during benchmark
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
            agent = agent_pool[j]
            n_agents_j = len(param_pool[j])
            param_names_j = param_names_pool[j]
            path_j = paths_pool[j]
            agent_name_j = agent_name_pool[j]
            if save:
                csv_write(['env_name', 'env_number', 'agent_name', 'agent_number'] + param_names_j + ['thread_number', 'score'], path_j, 'w')
            for k in range(n_agents_j):
                agent.reset(param_pool[j][k])
                if verbose:
                    print('Created agent', j+1, '/', n_agents,'with parameters', k+1, '/', n_agents_j)
                    agent.display()
                # EPISODES
                pool = Pool()
                results_pool = []
                agent_number = k
                agent_param = param_pool[j][k]
                n_epi_per_thread = int(n_epi / n_thread)
                for l in range(n_thread):
                    results_pool.append(pool.apply_async(multi_run, [env_name, i, env, agent_name_j, agent_number, agent_param, agent, tmax, n_epi_per_thread, l+1, save, path_j, verbose]))
                for result in results_pool:
                    result.get()

def test_multithread():
    env = gym.make('NSFrozenLakeEnv-v0')
    nenv = 1
    nepi = 1000
    tmax = 100
    n_thread = 5

    agent_name_pool = ['UCT']
    agent_pool = [uct.UCT(env.action_space)]
    param_names_pool = [
        ['action_space','rollouts','horizon','gamma','ucb_constant','is_model_dynamic']
    ]
    param_pool = [
        [[env.action_space,  10, 10, 0.9, 6.36396103068, True]]
    ]
    paths_pool = ['multitest.csv']

    multithread_benchmark('NSFrozenLakeEnv-v0', nenv, agent_name_pool, agent_pool, param_pool, param_names_pool, nepi, tmax, save=True, paths_pool=paths_pool, n_thread=n_thread, verbose=True)

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
    param_names_pool = [
        ['action_space','rollouts','horizon','gamma','ucb_constant','is_model_dynamic'],
        ['action_space']
    ]
    param_pool = [
        [[env.action_space,  10, 100, 0.9, 6.36396103068, True],[env.action_space, 100, 100, 0.9, 6.36396103068, True]],
        [[env.action_space],[env.action_space],[env.action_space]]
    ]
    paths_pool = ['uct.csv','random.csv']

    benchmark('NSFrozenLakeEnv-v0', nenv, agent_name_pool, agent_pool, param_pool, param_names_pool, nepi, tmax, save=True, paths_pool=paths_pool, verbose=True)
