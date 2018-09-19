import gym
import numpy as np
import dyna_gym.envs.cartpole_dynamic_reward_v2
import dyna_gym.agents.uct as uct
import dyna_gym.agents.tabular_iquct as tabiquct
import statistics as stat

from multiprocessing import Pool
import csv

def print_stat(l):
	print('    mean   : {}'.format(stat.mean(l)))
	print('    median : {}'.format(stat.median(l)))
	if(len(l) > 1):
		print('    stdev  : {}'.format(stat.stdev(l)))

def write_row(path, row, mode):
    '''
    Write a row in the csv file at the given path.
    The format of the row should be a list of strings eg ['a', 'b', 'c']
    '''
    with open(path, mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)

def run(env, agent, timesteps, verbose):
    '''
    Run a simulation and return the score.
    Parameters such as agent and environment are reseted at the begining of the episode.
    '''
    env.reset()
    agent.reset()
    done = False
    cumulative_reward = 0
    for ts in range(timesteps):
        __, reward, done, __ = env.step(agent.act(env,done))
        if verbose:
            env.print_state()
        cumulative_reward += reward
        # Termination
        if ts+1 == timesteps:
            if verbose:
                print("Episode reached the end ({} timesteps)".format(ts+1))
        if done:
            if verbose:
                print("Episode finished after {} timesteps".format(ts+1))
            break
    return cumulative_reward

def loop(env, agent, path, n_epi, epi_length):
    '''
    Main loop
    '''
    write_row(path,['simulation_nb', 'score', 'mean', 'median', 'stdev'], 'w')
    scores = []
    for i in range(n_epi):
        score_i = run(env,agent,epi_length,False)
        scores.append(score_i)
        stdev = 0
        if(len(scores) > 1):
            stdev = stat.stdev(scores)
        write_row(path,[i, score_i, stat.mean(scores), stat.median(scores), stdev], 'a')

def dobatch(n_envs, n_epi):
    '''
    n_env : number of different environments
    n_epi :
    '''

## Parameters
env = gym.make('CartPoleDynamicReward-v2')
pool = Pool()
n_epi = 100 # Nb of episodes
epi_length = 1000 # Nb of timesteps within a single episode

p_gamma = 0.9
p_maxdepth = 1000 # Max depth of tree search
p_ucbcst = 5

results_pool = []

## Pools
p_rollouts_pool = [100, 1000] # Budget of tree search
p_keep_data_pool = [0,1]
p_nlstm_pool = [1,2]
p_layer_pool1 = [
    [7, 50, 1],
    [7, 100, 1],
    [7, 200, 1]
]
p_layer_pool2 = [
    [7, 50, 50, 1],
    [7, 50, 100, 1],
    [7, 50, 200, 1],
    [7, 100, 100, 1],
    [7, 100, 200, 1],
    [7, 200, 200, 1]
]
p_nepoch_pool = [1, 10, 20]

for
    ## Run IQUCT
    p_model = lstm.LSTMModel(
        input_dim=len(env.state)+2,
        n_actions=env.action_space.n,
        nb_epoch=10
    )
    agent = appiquct.ApproxIQUCT(
        action_space=env.action_space,
        gamma=p_gamma,
        rollouts=p_rollouts,
        max_depth=p_max_depth,
        ucb_constant=p_ucb_cst,
        model=p_model
    )
    path = 'local/data/iquct_' +\
        'rollout' + str(p_rollouts) + '_'\
        'keep' + str(p_keep_data) + '_'\
        'nLSTM' + str(p_nlstm) + '_'\
        'layer' + str(p_layer) + '_'\
        'nepoch' + str(p_nepoch) +\
        '.csv'
    results_pool.append(pool.apply_async(loop, [env, agent, path, n_epi, epi_length]))

## Get the results
for result in results_pool:
	result.get()
