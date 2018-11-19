"""
Helpful functions when dealing with distributions
"""

import numpy as np
import dyna_gym.utils.utils as utl
from scipy.stats import wasserstein_distance
from scipy.optimize import linprog
from math import sqrt

def random_tabular(size):
    """
    Generate a 1D numpy array whose coefficients sum to 1
    """
    w = np.random.random(size)
    return w / np.sum(w)

def random_constrained(u, maxdist):
    """
    Randomly generate a new distribution st the Wasserstein distance between the input
    distribution u and the generated distribution is smaller than the input maxdist.
    Notice that the generated distribution has the same support as the input distribution.
    """
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

def clean_distribution(w):
    for i in range(len(w)):
        if utl.close(w[i], 0.0):
            w[i] = 0.0
        else:
            assert w[i] > 0.0, 'Error: negative weight computed ({}th index): w={}'.format(i, w)
    return w

def worst_dist(v, w0, c):
    """
    Generate argmin_w (w^T v) st W1(w,w0) <= c where W1 is the Wasserstein distance
    """
    n = len(v)
    obj = np.concatenate((v, np.zeros(shape=n)), axis=0)

    U = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(i+1):
            U[i][j] = 1.0
    A1 = np.concatenate((-np.eye(n),np.zeros(shape=(n,n))), axis=1)
    A2 = np.reshape(np.concatenate((np.zeros(n), np.ones(n))), newshape=(1,2*n))
    A3 = np.concatenate((U,-np.eye(n)), axis=1)
    A4 = np.concatenate((-U,-np.eye(n)), axis=1)
    A = np.concatenate((A1,A2,A3,A4), axis=0)

    b1 = np.zeros(n)
    b2 = np.asarray([c])
    b3 = np.dot(U,w0)
    b4 = np.dot(-U,w0)
    b = np.concatenate((b1,b2,b3,b4), axis=0)

    Ae = np.reshape(np.concatenate((np.ones(n),np.zeros(n))), newshape=(1,2*n))
    be = np.asarray([1])

    res = linprog(obj, A_eq=Ae, b_eq=be, A_ub=A, b_ub=b)
    x = res.x[:n]
    return clean_distribution(x)
