"""
Helpful functions when dealing with distributions
"""

import numpy as np
from scipy.stats import wasserstein_distance

def random_tabular(size):
    '''
    Generate a 1D numpy array whose coefficients sum to 1
    '''
    w = np.random.random(size)
    return w / np.sum(w)

def random_constrained(u, maxdist):
    '''
    Randomly generate a new distribution st the Wasserstein distance between the input
    distribution u and the generated distribution is smaller than the input maxdist.
    Notice that the generated distribution has the same support as the input distribution.
    '''
    max_n_trial = int(1e4) # Maximum number of trials
    val = np.asarray(range(len(u)))
    v = random_tabular(val.size)
    for i in range(max_n_trial):
        if wasserstein_distance(val,val,u,v) <= maxdist:
            return v
        else:
            v = random_tabular(val.size)
    print('Failed to generate constrained distribution after {} trials'.format(max_n_trial))
    exit()
