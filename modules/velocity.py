import numpy as np

def compute_velocity(state, t):
    p = state.p
    dr = state.dr
    
    dp_dr = np.zeros_like(p)
    
    #inner boundary
    dp_dr[0] = (-3.0 * p[0] + 4.0 * p[1] - p[2]) / (2.0 * dr)
    
    #outer boundary
    dp_dr[-1] = (3.0 * p[-1] - 4.0 * p[-2] + p[-3]) / (2.0 * dr)
    
    #interior
    dp_dr[1:-1] = (p[2:] - p[:-2]) / (2.0 * dr)
    
    #Darcy's law
    u = - state.k_hat * dp_dr
    
    state.u[:] = u