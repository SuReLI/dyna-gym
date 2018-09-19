"""
Helpful functions when dealing with distributions
"""

import numpy as np
from scipy.stats import wasserstein_distance

def random_tabular(size):
    '''
    Generate a 1D numpy array whose coefficients sum to 1
    '''
    u_weights = np.random.random(size)
    return u_weights / np.sum(u_weights)

def random_constrained(u_values, u_weights, maxdist):
    '''
    Randomly generate a new distribution st the Wasserstein distance between the input
    distribution u and the generated distribution is smaller than the input maxdist.
    Notice that the generated distribution has the same values as the input distribution.
    '''
    max_n_trial = 10000 # Maximum number of trials
    v_weights = random_tabular(u_values.size)
    for i in range(max_n_trial):
        if wasserstein_distance(u_values,u_values,u_weights,v_weights) <= maxdist:
            return v_weights
        else:
            v_weights = random_tabular(u_values.size)
    print('Failed to generate constrained distribution after {} trials'.format(max_n_trial))
    exit()
