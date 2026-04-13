import numpy as np

def compute_velocity(state, t):
    p = state.p
    dr = state.dr
    k = state.k
    mu = state.mu
    
    dp_dr = np.zeros_like(p)
    
    #forward difference for inner boundary
    dp_dr[0] = (-3.0 * p[0] + 4.0 * p[1] - p[2]) / (2.0 * dr)
    
    #backward difference for outer boundary
    dp_dr[-1] = (3.0 * p[-1] - 4.0 * p[-2] + p[-3]) / (2.0 * dr)
    
    #central differences for interior
    dp_dr[1:-1] = (p[2:] - p[:-2]) / (2.0 * dr)
    
    state.u[:] = - (k / mu) * dp_dr