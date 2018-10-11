import numpy as np
from math import isclose

def close(a, b, r=13):
    return isclose(round(a,r), round(b,r), rel_tol=1e-12, abs_tol=0.0)

def are_coeff_equal(v):
    return bool(np.prod(list(v[i] == v[i+1] for i in range(len(v)-1)), axis=0))

def are_coeff_close(v):
    return bool(np.prod(list(close(v[i],v[i+1]) for i in range(len(v)-1)), axis=0))
