import gym
import numpy as np
import statistics as stat
from multiprocessing import Pool

import dyna_gym.agents.uct as uct
import dyna_gym.agents.rats as rats

#np.random.seed(1993)

def print_stat(l):
    print('    nb elements : {}'.format(len(l)))
    print('    mean        : {}'.format(stat.mean(l)))
    print('    median      : {}'.format(stat.median(l)))
    if(len(l) > 1):
        print('    stdev       : {}'.format(stat.stdev(l)))

def run(env, agent, reset_agent, epi_length, n_epi):
    CR = []
    for i in range(n_epi):
        if reset_agent:
            agent.reset()
        done = False
        cumulative_reward = 0
        env.reset()
        for t in range(epi_length):
            action = agent.act(env,done)
            _, reward, done, _ = env.step(action)
            cumulative_reward += reward
            if (t+1 == epi_length) or done:
                break
        print('episode {} done, cr={}'.format(i, cumulative_reward))
        CR.append(cumulative_reward)
    print('average')
    print_stat(CR)

def dobatch(env_pool, epi_length, n_epi, agent, reset_agent):
    '''
    env_pool    : different environments
    epi_length  : number of timesteps in one episode
    n_epi       : number of episodes per environment
    agent       : agent used for the simulation
    reset_agent : boolean, reset the agent before each episode if True
    '''
    pool = Pool()
    results_pool = []
    for env in env_pool:
        results_pool.append(pool.apply_async(run, [env, agent, reset_agent, epi_length, n_epi]))
    for result in results_pool:
        result.get()

def generate_env_pool(env_name, n_env):
    pool = []
    for _ in range(n_env):
        pool.append(gym.make(env_name))
    action_space = pool[0].action_space
    return pool, action_space

### Parameters

epi_length = 100
n_epi = 3
env_pool, acsp = generate_env_pool(env_name='RandomNSMDP-v0',n_env=5)

p_gamma = 0.9
p_ucb_cst = 0.707 * p_gamma / (1 - p_gamma)
agent0 = uct.UCT(
    action_space=acsp,
    gamma=p_gamma,
    rollouts=20,
    max_depth=100,
    is_model_dynamic=True,
    ucb_constant=p_ucb_cst
)
agent1 = uct.UCT(
    action_space=acsp,
    gamma=p_gamma,
    rollouts=20,
    max_depth=100,
    is_model_dynamic=False,
    ucb_constant=p_ucb_cst
)
agent2 = rats.RATS(
    action_space=acsp,
    max_depth=2,
    gamma=p_gamma,
    L_v=1,
    horizon=3
)

### Call

print('\n---------------- UCT nsmdp ----------------')
dobatch(env_pool, epi_length, n_epi, agent0, True)
print('\n---------------- UCT snapshot ----------------')
dobatch(env_pool, epi_length, n_epi, agent1, True)
print('\n---------------- RATS ----------------')
dobatch(env_pool, epi_length, n_epi, agent2, True)
