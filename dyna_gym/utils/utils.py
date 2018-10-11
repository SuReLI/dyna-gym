from math import isclose

def close(a, b, r=13):
    return isclose(round(a,r), round(b,r), rel_tol=1e-12, abs_tol=0.0)
