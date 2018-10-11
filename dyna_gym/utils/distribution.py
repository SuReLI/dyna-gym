"""
Helpful functions when dealing with distributions
"""

import numpy as np
import cvxpy as cp
import dyna_gym.utils.utils as utl
from math import sqrt
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

def step_zero_coeff_active(w0, g):
        l = list((w0[i] / g[i]) for i in range(len(w0)) if (g[i] > 0))#TRM
        if len(l) == 0:#TRM
            print('\n\n')
            print(w0)
            print(g)
            print('\n\n')
        return min((w0[i] / g[i]) for i in range(len(w0)) if (g[i] > 0))

def step_wass_active(c, g):
    n = len(g)
    f = cp.Variable(n)
    A = np.eye(n, dtype=int)
    A = np.delete(A, n-1, axis=0)
    for i in range(n-1):
        A[i, i+1] = -1
    A = np.concatenate((A,-A), axis=0)
    objective = cp.Maximize(cp.sum(cp.diag(g) * f))
    constraints = [A * f <= 1]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    return c / result

def clean_almost_zero_coefficients(w):
    for i in range(len(w)):
        if utl.close(w[i], 0.0):
            w[i] = 0.0
        elif w[i] < 0.0:
            print('Error: negative weight computed ({}th index): w={}'.format(i, w))
            exit()
    return w

def worst_dist(v, w0, c):
    '''
    Generate argmin_w (w^T v) st W1(w,w0) <= c where W1 is the Wasserstein distance
    '''
    if utl.are_coeff_equal(v):
        return w0
    else:
        n = len(v)
        u = np.ones(shape=n)
        g = v - (np.dot(u, v) / float(n)) * u
        g = g / sqrt(np.dot(g, g))
        alpha = min(
            step_zero_coeff_active(w0, g),
            step_wass_active(c, g)
        )
        wstar = w0 - alpha * g
        wstar = clean_almost_zero_coefficients(wstar)
        return wstar
